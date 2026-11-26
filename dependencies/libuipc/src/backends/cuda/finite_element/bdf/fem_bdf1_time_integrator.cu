#include <finite_element/fem_time_integrator.h>
#include <time_integrator/bdf1_flag.h>
#include <sim_engine.h>
#include <uipc/builtin/attribute_name.h>
#include <fstream>

namespace uipc::backend::cuda
{
namespace
{
    enum class Predictor { DeltaVelocity, PaperEq5, VelPro };

    __global__ void progressive_prediction(cuda_tool::CBufferView<IndexT> fixed,
                                           cuda_tool::CBufferView<IndexT> dynamic,
                                           cuda_tool::CBufferView<Vector3> xs,
                                           cuda_tool::CBufferView<Vector3> vs,
                                           cuda_tool::CBufferView<Vector3> input,
                                           cuda_tool::CBufferView<Vector3> gravity,
                                           cuda_tool::CBufferView<Vector3> external_acc,
                                           cuda_tool::BufferView<Vector3> predicted,
                                           IndexT offset, int count, Float dt, Predictor mode)
    {
        int local = blockIdx.x * blockDim.x + threadIdx.x;
        if(local >= count)
            return;
        int i = offset + local;
        if(!fixed(i) && dynamic(i))
        {
            // Eq. (5): input planes are P*x_coarse^t and V*v_coarse^t.
            // The old fine positions still initialize Newton and define its velocity update.
            if(mode == Predictor::DeltaVelocity)
                predicted(i) = xs(i) + dt * (vs(i) + input(local));
            else
            {
                // VelPro retains the current fine position and receives only the coarse velocity.
                const Vector3 center = mode == Predictor::PaperEq5
                                           ? Vector3(input(local) + dt * input(count + local))
                                           : Vector3(xs(i) + dt * input(local));
                predicted(i) = center + (dt * dt) * (gravity(i) + external_acc(i));
            }
        }
    }

    __global__ void FEMBDF1Integrator_do_predict_dof_kernel(
        cuda_tool::CBufferView<IndexT>  is_fixed,
        cuda_tool::CBufferView<IndexT>  is_dynamic,
        cuda_tool::BufferView<Vector3>  x_prevs,
        cuda_tool::CBufferView<Vector3> xs,
        cuda_tool::CBufferView<Vector3> vs,
        cuda_tool::BufferView<Vector3>  x_tildes,
        cuda_tool::CBufferView<Vector3> gravities,
        cuda_tool::CBufferView<Vector3> external_force_accs,
        Float                           dt,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        // record previous position
        Vector3& x_prev = x_prevs(i);
        x_prev          = xs(i);

        const Vector3& v = vs(i);

        // 0) fixed: x_tilde = x_prev
        Vector3 x_tilde = x_prev;

        if(!is_fixed(i))
        {
            const Vector3& g         = gravities(i);
            const Vector3& f_ext_acc = external_force_accs(i);

            // 1) static problem: x_tilde = x_prev + (g + f_ext_acc) * dt * dt
            x_tilde += (g + f_ext_acc) * dt * dt;

            // 2) dynamic problem: x_tilde = x_prev + v * dt + (g + f_ext_acc) * dt * dt
            if(is_dynamic(i))
            {
                x_tilde += v * dt;
            }
        }

        x_tildes(i) = x_tilde;
    }

    __global__ void FEMBDF1Integrator_do_update_state_kernel(
        cuda_tool::CBufferView<Vector3> xs,
        cuda_tool::BufferView<Vector3>  vs,
        cuda_tool::CBufferView<Vector3> x_prevs,
        Float                           dt,
        int                             n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        Vector3&       v      = vs(i);
        const Vector3& x_prev = x_prevs(i);
        const Vector3& x      = xs(i);

        v = (x - x_prev) * (1.0 / dt);
    }
}  // namespace

class FEMBDF1Integrator final : public FEMTimeIntegrator
{
    struct PredictionInput
    {
        IndexT offset;
        IndexT count;
        Predictor mode;
        SizeT steps;
        SizeT audit_frame = 0;
        std::string audit_directory;
        vector<Vector3> host;
        cuda_tool::DeviceBuffer<Vector3> device;
        std::ifstream file;
    };
    vector<U<PredictionInput>> prediction_inputs;

  public:
    using FEMTimeIntegrator::FEMTimeIntegrator;

    void do_build(BuildInfo& info) override
    {
        // require the BDF1 flag
        require<BDF1Flag>();
    }

    virtual void do_init(InitInfo& info) override
    {
        prediction_inputs.clear();
        for(auto& slot : world().scene().geometries())
        {
            auto& geo = slot->geometry();
            auto delta_velocity = geo.meta().find<std::string>("pd_cloth_delta_velocity");
            auto coarse_state = geo.meta().find<std::string>("pd_cloth_paper_eq5_state");
            auto coarse_velocity = geo.meta().find<std::string>("pd_cloth_velpro_velocity");
            UIPC_ASSERT_THROW(int(bool(delta_velocity)) + int(bool(coarse_state))
                                  + int(bool(coarse_velocity)) <= 1,
                              "Select only one progressive cloth predictor per mesh");
            auto filename = coarse_state ? coarse_state : (coarse_velocity ? coarse_velocity : delta_velocity);
            if(!filename)
                continue;
            auto offset = geo.meta().find<IndexT>(builtin::backend_fem_vertex_offset);
            auto* mesh = geo.as<geometry::SimplicialComplex>();
            UIPC_ASSERT_THROW(mesh && offset && mesh->vertices().size() > 0,
                              "Progressive predictor input requires a FEM mesh");
            auto input = uipc::make_unique<PredictionInput>();
            input->offset = offset->view().front();
            input->count = static_cast<IndexT>(mesh->vertices().size());
            input->mode = coarse_state ? Predictor::PaperEq5
                                       : (coarse_velocity ? Predictor::VelPro : Predictor::DeltaVelocity);
            if(auto directory = geo.meta().find<std::string>("pd_cloth_audit_directory"))
            {
                input->audit_directory = directory->view().front();
                input->audit_frame = std::stoull(geo.meta().find<std::string>("pd_cloth_audit_frame")
                                                    ->view().front());
            }
            auto vectors = static_cast<SizeT>(input->count) * (coarse_state ? 2 : 1);
            input->host.resize(vectors);
            input->device.resize(vectors);
            input->file.open(filename->view().front(), std::ios::binary | std::ios::ate);
            auto bytes = input->file.tellg();
            auto stride = input->host.size() * sizeof(Vector3);
            static_assert(sizeof(Vector3) == 3 * sizeof(Float));
            UIPC_ASSERT_THROW(bytes > 0 && static_cast<SizeT>(bytes) % stride == 0,
                              "Invalid progressive predictor file {}", filename->view().front());
            input->steps = static_cast<SizeT>(bytes) / stride;
            prediction_inputs.push_back(std::move(input));
        }
    }

    virtual void do_predict_dof(PredictDofInfo& info) override
    {
        auto k = FEMBDF1Integrator_do_predict_dof_kernel;
        int  n = (int)info.xs().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                info.is_fixed().cview(),
                info.is_dynamic().cview(),
                info.x_prevs(),
                info.xs().cview(),
                info.vs().cview(),
                info.x_tildes(),
                info.gravities().cview(),
                info.external_force_accs().cview(),
                info.dt(),
                n);
        }
        for(auto& input : prediction_inputs)
        {
            auto frame = engine().frame();
            UIPC_ASSERT_THROW(frame > 0 && frame <= input->steps,
                              "No progressive predictor input for IPC frame {}", frame);
            auto bytes = input->host.size() * sizeof(Vector3);
            input->file.seekg((frame - 1) * bytes);
            input->file.read(reinterpret_cast<char*>(input->host.data()), bytes);
            UIPC_ASSERT_THROW(input->file.good(), "Failed to read progressive predictor frame {}", frame);
            input->device.view().copy_from(input->host.data());
            auto kernel = progressive_prediction;
            int count = input->count;
            kernel<<<cuda_tool::best_grid_dim(count, kernel), cuda_tool::best_block_dim(kernel)>>>(
                info.is_fixed().cview(), info.is_dynamic().cview(), info.xs().cview(),
                info.vs().cview(), input->device.cview(), info.gravities().cview(),
                info.external_force_accs().cview(), info.x_tildes(), input->offset, count,
                info.dt(), input->mode);
            if(frame == input->audit_frame && !input->audit_directory.empty())
            {
                vector<Vector3> predicted(count);
                info.x_tildes().subview(input->offset, count).copy_to(predicted.data());
                auto path = input->audit_directory + "/x_tilde.f64";
                std::ofstream output(path, std::ios::binary);
                output.write(reinterpret_cast<const char*>(predicted.data()), count * sizeof(Vector3));
                UIPC_ASSERT_THROW(output.good(), "Could not write {}", path);
            }
        }
    }

    virtual void do_update_state(UpdateVelocityInfo& info) override
    {
        auto k = FEMBDF1Integrator_do_update_state_kernel;
        int  n = (int)info.xs().size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                info.xs().cview(), info.vs(), info.x_prevs().cview(), info.dt(), n);
        }
    }
};

REGISTER_SIM_SYSTEM(FEMBDF1Integrator);
}  // namespace uipc::backend::cuda
