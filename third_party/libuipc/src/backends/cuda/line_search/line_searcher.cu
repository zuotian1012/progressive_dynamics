#include <finite_element/constitutions/psq_trial_energy.h> // PSQ positive-infinity trial rejection
#include <finite_element/constitutions/psq_proxy.h>
#include <cstdlib>
#include <line_search/line_searcher.h>
#include <uipc/common/enumerate.h>
#include <uipc/common/zip.h>
#include <line_search/line_search_reporter.h>
#include <uipc/common/timer.h>
#include <cuda_tool/cub.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(LineSearcher);

void LineSearcher::do_build() {}

void LineSearcher::init()
{
    auto scene = world().scene();

    auto report_enery_attr = scene.config().find<IndexT>("line_search/report_energy");
    m_report_energy = report_enery_attr->view()[0] != 0;

    auto max_iter_attr = scene.config().find<IndexT>("line_search/max_iter");
    m_max_iter         = max_iter_attr->view()[0];

    m_dt_attr = scene.config().find<Float>("dt");
    UIPC_ASSERT(m_dt_attr, "Scene config must have a 'dt' attribute.");

    const auto energy_count = m_reporters.view().size() + 1;
    m_device_energy_values.resize(energy_count);
    m_energy_values.resize(energy_count, 0);

    auto reporter_view = m_reporters.view();

    for(auto&& [i, R] : enumerate(reporter_view))
        R->m_index = i;

    for(auto&& [i, R] : enumerate(reporter_view))
        R->init();
}

void LineSearcher::record_start_point()
{
    for(auto&& R : m_reporters.view())
    {
        RecordInfo info;
        R->record_start_point(info);
    }
}

void LineSearcher::step_forward(Float alpha)
{
    for(auto&& R : m_reporters.view())
    {
        StepInfo info;
        info.alpha = alpha;
        R->step_forward(info);
    }
}

Float LineSearcher::compute_energy(bool is_initial)
{
    Timer timer{"Compute Energy"};

    auto reporter_view = m_reporters.view();

    for(auto&& [i, R] : enumerate(reporter_view))
    {
        ComputeEnergyInfo info{
            this, cuda_tool::VarView<Float>{m_device_energy_values.data() + i}};
        info.m_is_initial = is_initial;
        R->compute_energy(info);
        UIPC_ASSERT(info.m_energy_set,
                    "Energy[{}] not set by reporter, did you forget to call energy()?",
                    R->name());
    }

    const auto reporter_count = reporter_view.size();
    auto* total_energy_device = m_device_energy_values.data() + reporter_count;
    if(reporter_count > 0)
    {
        cuda_tool::DeviceReduce().Sum(m_device_energy_values.data(), total_energy_device, reporter_count);
    }
    else
    {
        cuda_tool::VarView<Float>{total_energy_device}.fill(0.0);
    }

    m_device_energy_values.copy_to(m_energy_values.data());

    const char* inf_env=std::getenv("UIPC_PSQ_REJECT_INF_TRIAL");
    const bool allow_psq_inf=inf_env && std::string(inf_env)=="1" && psq_replaces_bending();
    bool saw_rejectable_inf=false;
    auto reporter_energies = span{m_energy_values}.first(reporter_count);
    for(auto&& [energy, reporter] : zip(reporter_energies, reporter_view))
    {
        const bool rejectable=psq_trial_energy::rejectable_positive_infinity(
            energy,is_initial,std::string(reporter->name()).find("FEMLineSearchReporter")!=std::string::npos,allow_psq_inf);
        saw_rejectable_inf=saw_rejectable_inf || rejectable;
        UIPC_ASSERT(std::isfinite(energy) || rejectable,
                    "Energy [{}] is {}",
                    reporter->name(),
                    energy);
    }

    Float total_energy = m_energy_values.back();
    UIPC_ASSERT(std::isfinite(total_energy) || (saw_rejectable_inf && std::isinf(total_energy) && total_energy>0),
                "Total line-search energy is {}",
                total_energy);

    if(m_report_energy)
    {
        m_report_stream << R"(
-------------------------------------------------------------------------------
*                             Compute Energy                                  *
-------------------------------------------------------------------------------
)";
        m_report_stream << "Total:" << total_energy << "\n";
        for(auto&& [R, value] : zip(reporter_view, reporter_energies))
        {
            m_report_stream << "  > " << R->name() << "=" << value << "\n";
        }
        m_report_stream << "-------------------------------------------------------------------------------";
        logger::info(m_report_stream.str());
        m_report_stream.str("");
    }

    return total_energy;
}

void LineSearcher::add_reporter(LineSearchReporter* reporter)
{
    UIPC_ASSERT(reporter, "reporter is nullptr");
    check_state(SimEngineState::BuildSystems, "add_reporter()");
    m_reporters.register_sim_system(*reporter);
}

LineSearcher::ComputeEnergyInfo::ComputeEnergyInfo(LineSearcher* impl,
                                                   cuda_tool::VarView<Float> energy) noexcept
    : m_impl(impl)
    , m_energy(energy)
{
}

Float LineSearcher::ComputeEnergyInfo::dt() noexcept
{
    return m_impl->m_dt_attr->view()[0];
}

cuda_tool::VarView<Float> LineSearcher::ComputeEnergyInfo::energy() noexcept
{
    m_energy_set = true;
    return m_energy;
}

bool LineSearcher::ComputeEnergyInfo::is_initial() noexcept
{
    return m_is_initial;
}

SizeT LineSearcher::max_iter() const noexcept
{
    return m_max_iter;
}
}  // namespace uipc::backend::cuda
