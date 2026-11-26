#include <newton_tolerance/newton_tolerance_checker.h>
#include <finite_element/finite_element_method.h>
#include <uipc/geometry/attribute_slot.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace uipc::backend::cuda
{
// Isolated bunny build. Its only moving object is the FEM cloth. C-IPC's
// dynamic stopping residual is ||dx_free||_2 / (sqrt(n_free) * dt).
class BunnyRmsTranslationChecker final : public NewtonToleranceChecker
{
  public:
    using NewtonToleranceChecker::NewtonToleranceChecker;

  private:
    SimSystemSlot<FiniteElementMethod> fem;
    vector<Vector3> displacement;
    vector<SizeT> free_vertices;
    Float dt = 0;
    Float tolerance = 0;
    Float residual = 0;
    SizeT consecutive = 0;
    SizeT frame = 0;
    SizeT iteration = 0;

    void do_build(BuildInfo&) override
    {
        fem = require<FiniteElementMethod>();
        auto& config = world().scene().config();
        dt = config.find<Float>("dt")->view()[0];
        tolerance = config.find<Float>("newton/velocity_tol")->view()[0];
        if(dt <= 0 || tolerance <= 0 || config.find<Float>("newton/velocity_tol_relative")->view()[0] != 0)
            throw std::runtime_error("Bunny RMS checker requires a positive absolute velocity tolerance");
    }

    void do_init(InitInfo&) override {}

    void do_pre_newton(PreNewtonInfo& info) override
    {
        frame = info.frame();
        consecutive = 0;
    }

    void do_check(CheckResultInfo& info) override
    {
        iteration = info.newton_iter();
        if(displacement.empty())
        {
            vector<IndexT> fixed(fem->is_fixed().size());
            fem->is_fixed().copy_to(fixed.data());
            for(SizeT i = 0; i < fixed.size(); ++i)
                if(!fixed[i])
                    free_vertices.push_back(i);
            if(free_vertices.empty())
                throw std::runtime_error("Bunny RMS checker found no free FEM vertices");
            displacement.resize(fixed.size());
        }
        // Sum in vertex order. This also keeps the diagnostic residual
        // reproducible for the same back-solved direction.
        fem->dxs().copy_to(displacement.data());
        Float squared_norm = 0;
        for(SizeT i : free_vertices)
            squared_norm += displacement[i].squaredNorm();
        residual = std::sqrt(squared_norm / free_vertices.size()) / dt;
        if(!std::isfinite(residual))
            throw std::runtime_error("Nonfinite bunny Newton RMS velocity");
        consecutive = residual <= tolerance ? std::min<SizeT>(consecutive + 1, 3) : 0;
        info.converged(consecutive == 3);
    }

    std::string do_report() override
    {
        return fmt::format("BUNNY_RMS frame={} newton={} velocity={:.17g} tolerance={:.17g} free={} consecutive={}",
                           frame, iteration, residual, tolerance, free_vertices.size(), consecutive);
    }
};

REGISTER_SIM_SYSTEM(BunnyRmsTranslationChecker);
}  // namespace uipc::backend::cuda
