#include <uipc/common/timer.h>
#include <finite_element/finite_element_elastics.h>
#include <finite_element/finite_element_constitution.h>
#include <finite_element/finite_element_extra_constitution.h>
#include <finite_element/fem_linear_subsystem_reporter.h>
#include <finite_element/fem_line_search_subreporter.h>
#include <finite_element/progressive_cloth.h>
#include <utils/report_extent_check.h>
#include <sim_engine.h>
#include <uipc/builtin/attribute_name.h>

namespace uipc::backend::cuda
{
REGISTER_SIM_SYSTEM(FiniteElementElastics);

void FiniteElementElastics::ReportExtentInfo::check(std::string_view name) const
{
    check_report_extent(m_gradient_only_checked, m_gradient_only, m_hessian_count, name);
}

void FiniteElementElastics::do_build()
{
    m_impl.finite_element_method = require<FiniteElementMethod>();
    m_impl.finite_element_method->add_elastics(this);
}

void FiniteElementElastics::init()
{
    m_impl.progressive_cloths.clear();
    auto slots = world().scene().geometries();
    for(const auto& info : m_impl.fem().geo_infos)
    {
        const auto& geo = slots[info.geo_slot_index]->geometry();
        if(auto asset = geo.meta().find<std::string>("pd_cloth_proxy_asset"))
        {
            auto* cloth = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT_THROW(cloth && info.dim_uid.dim == 2 && info.dim_uid.uid == 0,
                              "Progressive cloth requires the Empty FEM constitution on the coarse mesh");
            if(auto extra = geo.meta().find<VectorXu64>(builtin::extra_constitution_uids))
            {
                // FEM constraints use their own registration and are unaffected.
                UIPC_ASSERT_THROW(extra->view().front().size() == 0,
                                  "Proxy cloth cannot share geometry slot {} with native FEM extra elasticity",
                                  info.geo_slot_index);
            }
            m_impl.progressive_cloths.push_back(uipc::make_shared<ProgressiveCloth>(
                *cloth,
                span{m_impl.fem().h_rest_positions}.subspan(info.vertex_offset, info.vertex_count),
                static_cast<IndexT>(info.vertex_offset),
                asset->view().front()));
        }
    }

    auto  constitutions = m_impl.fem().constitutions.view();
    auto  extras        = m_impl.fem().extra_constitutions.view();
    SizeT N             = constitutions.size() + extras.size();

    m_impl.constitution_energy_offsets_counts.resize(N);
    m_impl.constitution_gradient_offsets_counts.resize(N);
    m_impl.constitution_hessian_offsets_counts.resize(N);

    auto energy_span = m_impl.constitution_energy_offsets_counts.counts();
    auto grad_span   = m_impl.constitution_gradient_offsets_counts.counts();
    auto hess_span   = m_impl.constitution_hessian_offsets_counts.counts();

    auto offset = 0;
    for(auto&& [I, c] : enumerate(constitutions))
    {
        ReportExtentInfo this_info;
        c->report_extent(this_info);
        energy_span[I] = this_info.m_energy_count;
        grad_span[I]   = this_info.m_gradient_count;
        hess_span[I]   = this_info.m_hessian_count;
    }

    offset += constitutions.size();
    for(auto&& [I, c] : enumerate(extras))
    {
        ReportExtentInfo this_info;
        c->report_extent(this_info);
        energy_span[offset + I] = this_info.m_energy_count;
        grad_span[offset + I]   = this_info.m_gradient_count;
        hess_span[offset + I]   = this_info.m_hessian_count;
    }
    offset += extras.size();

    UIPC_ASSERT(offset == N, "offset mismatching, offset = {}, expected = {}", offset, N);

    m_impl.constitution_energy_offsets_counts.scan();
    m_impl.constitution_gradient_offsets_counts.scan();
    m_impl.constitution_hessian_offsets_counts.scan();
    if(!m_impl.progressive_cloths.empty())
        logger::info("PD FEM elastic routing: proxies={} native_energy_slots={} native_gradient_blocks={} "
                     "native_hessian_blocks={}", m_impl.progressive_cloths.size(),
                     m_impl.constitution_energy_offsets_counts.total_count(),
                     m_impl.constitution_gradient_offsets_counts.total_count(),
                     m_impl.constitution_hessian_offsets_counts.total_count());
}

SizeT FiniteElementElastics::Impl::energy_count() const
{
    SizeT count = constitution_energy_offsets_counts.total_count();
    for(const auto& cloth : progressive_cloths)
        count += cloth->energy_count();
    return count;
}

SizeT FiniteElementElastics::Impl::gradient_count() const
{
    SizeT count = constitution_gradient_offsets_counts.total_count();
    for(const auto& cloth : progressive_cloths)
        count += cloth->gradient_count();
    return count;
}

SizeT FiniteElementElastics::Impl::hessian_count() const
{
    SizeT count = constitution_hessian_offsets_counts.total_count();
    for(const auto& cloth : progressive_cloths)
        count += cloth->hessian_count();
    return count;
}

cuda_tool::DoubletVectorView<Float, 3> FiniteElementElastics::ComputeGradientHessianInfo::gradients() const noexcept
{
    auto [offset, count] = m_impl->constitution_gradient_offsets_counts[m_index];
    return m_gradients.subview(offset, count);
}

cuda_tool::TripletMatrixView<Float, 3> FiniteElementElastics::ComputeGradientHessianInfo::hessians() const noexcept
{
    if(m_gradient_only)
        return cuda_tool::TripletMatrixView<Float, 3>{};

    auto [offset, count] = m_impl->constitution_hessian_offsets_counts[m_index];
    return m_hessians.subview(offset, count);
}

void FiniteElementElastics::Impl::assemble(FEMLinearSubsystem::AssembleInfo& info,
                                           SizeT frame, SizeT newton_iter)
{
    auto constitutions = fem().constitutions.view();
    auto extras        = fem().extra_constitutions.view();

    auto offset = 0;

    for(auto&& [I, c] : enumerate(constitutions))
    {
        ComputeGradientHessianInfo this_info{
            this, I, info.gradient_only(), info.dt(), info.gradients(), info.hessians()};

        c->compute_gradient_hessian(this_info);
    }
    offset += constitutions.size();

    for(auto&& [I, c] : enumerate(extras))
    {
        ComputeGradientHessianInfo this_info{
            this, offset + I, info.gradient_only(), info.dt(), info.gradients(), info.hessians()};
        c->compute_gradient_hessian(this_info);
    }
    offset += extras.size();

    UIPC_ASSERT(offset == constitutions.size() + extras.size(),
                "offset mismatching, offset = {}, expected = {}",
                offset,
                constitutions.size() + extras.size());

    SizeT g_offset = constitution_gradient_offsets_counts.total_count();
    SizeT h_offset = constitution_hessian_offsets_counts.total_count();
    for(auto& cloth : progressive_cloths)
    {
        Timer timer{"Assemble PD Proxy Elasticity"};
        auto gradients = info.gradients().subview(g_offset, cloth->gradient_count());
        cuda_tool::TripletMatrixView<Float, 3> hessians;
        if(!info.gradient_only())
            hessians = info.hessians().subview(h_offset, cloth->hessian_count());
        cloth->assemble(fem().xs.cview(), info.dt(), gradients, hessians, info.gradient_only(),
                        frame, newton_iter);
        g_offset += cloth->gradient_count();
        h_offset += cloth->hessian_count();
    }
}

void FiniteElementElastics::Impl::compute_energy(FEMLineSearchReporter::ComputeEnergyInfo& info)
{
    auto constitutions = fem().constitutions.view();
    auto extras        = fem().extra_constitutions.view();
    auto offset        = 0;
    for(auto&& [I, c] : enumerate(constitutions))
    {
        ComputeEnergyInfo this_info{this, I, info.dt(), info.energies()};
        c->compute_energy(this_info);
    }
    offset += constitutions.size();
    for(auto&& [I, c] : enumerate(extras))
    {
        ComputeEnergyInfo this_info{this, offset + I, info.dt(), info.energies()};
        c->compute_energy(this_info);
    }
    offset += extras.size();

    UIPC_ASSERT(offset == constitutions.size() + extras.size(),
                "offset mismatching, offset = {}, expected = {}",
                offset,
                constitutions.size() + extras.size());

    SizeT e_offset = constitution_energy_offsets_counts.total_count();
    for(auto& cloth : progressive_cloths)
    {
        cloth->compute_energy(fem().xs.cview(),
                              info.dt(),
                              info.energies().subview(e_offset, cloth->energy_count()));
        e_offset += cloth->energy_count();
    }
}

class FiniteElementElasticsLinearSubsystemReporter final : public FEMLinearSubsystemReporter
{
  public:
    using FEMLinearSubsystemReporter::FEMLinearSubsystemReporter;

    SimSystemSlot<FiniteElementElastics> elastics;

    FiniteElementElastics::Impl& el() { return elastics->m_impl; }

    virtual void do_build(BuildInfo& info) override
    {
        elastics = require<FiniteElementElastics>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        auto grad_count = el().gradient_count();
        info.gradient_count(grad_count);
        SizeT hess_count = 0;
        if(!info.gradient_only())
            hess_count = el().hessian_count();
        info.hessian_count(hess_count);
    }

    virtual void do_assemble(AssembleInfo& info) override
    {
        el().assemble(info, engine().frame(), engine().newton_iter());
    }
};

REGISTER_SIM_SYSTEM(FiniteElementElasticsLinearSubsystemReporter);

class FiniteElementElasticsLineSearchSubreporter final : public FEMLineSearchSubreporter
{
  public:
    using FEMLineSearchSubreporter::FEMLineSearchSubreporter;

    SimSystemSlot<FiniteElementElastics> elastics;

    FiniteElementElastics::Impl& el() { return elastics->m_impl; }

    virtual void do_build(BuildInfo& info) override
    {
        elastics = require<FiniteElementElastics>();
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_report_extent(ReportExtentInfo& info) override
    {
        auto energy_count = el().energy_count();
        info.energy_count(energy_count);
    }

    virtual void do_compute_energy(ComputeEnergyInfo& info) override
    {
        el().compute_energy(info);
    }
};

REGISTER_SIM_SYSTEM(FiniteElementElasticsLineSearchSubreporter);

}  // namespace uipc::backend::cuda
