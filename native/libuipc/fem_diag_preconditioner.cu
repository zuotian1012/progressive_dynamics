#include <type_define.h>
#include "diag_cholesky.h"
#include <Eigen/Dense>
#include <linear_system/local_preconditioner.h>
#include <finite_element/finite_element_method.h>
#include <linear_system/global_linear_system.h>
#include <finite_element/fem_linear_subsystem.h>
#include <global_geometry/global_vertex_manager.h>
#include <kernel_cout.h>
#include <cuda_tool/cuda_tool.h>
#include <uipc/geometry/simplicial_complex.h>

namespace uipc::backend::cuda
{
namespace
{
    namespace eigen = cuda_tool::eigen;

    __global__ void FEMDiagPreconditioner_do_assemble_kernel(
        cuda_tool::CBCOOMatrixView<Float, 3> triplet,
        cuda_tool::BufferView<uipc_diag::Factor> factors,
        cuda_tool::VarView<IndexT> fallback_count,
        SizeT                                fem_segment_offset,
        SizeT                                fem_segment_count,
        int                                  n)
    {
        int I = blockIdx.x * blockDim.x + threadIdx.x;
        if(I >= n)
            return;
        auto&& [g_i, g_j, H3x3] = triplet(I);

        IndexT i = g_i - fem_segment_offset;
        IndexT j = g_j - fem_segment_offset;

        if(i < 0 || j < 0 || i >= fem_segment_count || j >= fem_segment_count)
        {
            return;
        }

        if(i == j)
        {
            double a[9];
            for(int r=0;r<3;++r) for(int c=0;c<3;++c) a[3*r+c]=H3x3(r,c);
            int result=uipc_diag::factor(a,factors(i));
            UIPC_KERNEL_ASSERT(result>=0, "FEM diag: non-positive/non-finite diagonal at vertex %d", i);
            if(result>0) cuda_tool::atomic_add(fallback_count.data(), IndexT{1});
        }
    }

    __global__ void FEMDiagPreconditioner_do_apply_kernel(
        cuda_tool::CDenseVectorView<Float> r,
        cuda_tool::DenseVectorView<Float>  z,
        cuda_tool::CDense<IndexT>          converged,
        cuda_tool::CBufferView<uipc_diag::Factor> factors,
        int                                n)
    {
        int i = blockIdx.x * blockDim.x + threadIdx.x;
        if(i >= n)
            return;
        if(*converged != 0)
            return;
        double rhs[3],result[3];
        for(int k=0;k<3;++k) rhs[k]=r(i*3+k);
        uipc_diag::solve(factors(i),rhs,result);
        for(int k=0;k<3;++k) z(i*3+k)=result[k];
    }
}  // namespace

class FEMDiagPreconditioner : public LocalPreconditioner
{
  public:
    using LocalPreconditioner::LocalPreconditioner;

    FiniteElementMethod* finite_element_method = nullptr;
    GlobalLinearSystem*  global_linear_system  = nullptr;
    FEMLinearSubsystem*  fem_linear_subsystem  = nullptr;

    cuda_tool::DeviceBuffer<uipc_diag::Factor> factors;
    cuda_tool::DeviceVar<IndexT> fallback_count;

    virtual void do_build(BuildInfo& info) override
    {
        finite_element_method       = &require<FiniteElementMethod>();
        global_linear_system        = &require<GlobalLinearSystem>();
        fem_linear_subsystem        = &require<FEMLinearSubsystem>();
        auto& global_vertex_manager = require<GlobalVertexManager>();

        // MAS is selected via config; defer to FEMMASPreconditioner when on.
        auto precond = world().scene().config().find<std::string>("linear_system/fem_preconditioner");
        if(precond && precond->view()[0] == "mas")
        {
            throw SimSystemException(
                "FEMDiagPreconditioner: linear_system/fem_preconditioner == \"mas\", "
                "deferring to FEMMASPreconditioner.");
        }

        // This FEMDiagPreconditioner depends on FEMLinearSubsystem
        info.connect(fem_linear_subsystem);
    }

    virtual void do_init(InitInfo& info) override {}

    virtual void do_assemble(GlobalLinearSystem::LocalPreconditionerAssemblyInfo& info) override
    {
        factors.resize(finite_element_method->xs().size());
        fallback_count = 0;

        // 1) collect diagonal blocks
        auto k = FEMDiagPreconditioner_do_assemble_kernel;
        int  n = (int)info.A().triplet_count();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, nullptr>>>(
                info.A(), factors.view(), fallback_count.view(), info.dof_offset() / 3, info.dof_count() / 3, n);
            IndexT count=fallback_count;
            if(count) logger::warn("FEM diag scaled Cholesky: {} blocks use positive diagonal fallback", count);
        }
    }

    virtual void do_apply(GlobalLinearSystem::ApplyPreconditionerInfo& info) override
    {
        auto converged = info.converged();

        auto k = FEMDiagPreconditioner_do_apply_kernel;
        int  n = (int)factors.size();
        if(n > 0)
        {
            k<<<cuda_tool::best_grid_dim(n, k), cuda_tool::best_block_dim(k), 0, info.stream()>>>(
                info.r(), info.z(), converged.cviewer(), factors.cview(), n);
        }
    }
};

REGISTER_SIM_SYSTEM(FEMDiagPreconditioner);
}  // namespace uipc::backend::cuda
