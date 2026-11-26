#pragma once
#include <type_define.h>
#include <cuda_tool/cuda_tool.h>
#include <memory>
#include <uipc/common/vector.h>
#include "psq_strain.cuh"

namespace uipc::backend::cuda
{
class FiniteElementMethod;
// Experimental single-cloth adapter. Each instance owns its proxy buffers.
// Explicit binary bundle opt-in; native FEM coordinates remain the only DOFs.
class PsqProxy
{
public:
    PsqProxy();
    ~PsqProxy();
    void init(FiniteElementMethod& fem, cuda_tool::CBufferView<Vector3i> faces,
              cuda_tool::CBufferView<Matrix2x2> IB,
              const vector<Float>& lambdas, const vector<Float>& mus);
    bool active() const;
    bool replaces_elastic() const;
    bool preview_eligible() const;
    SizeT energy_count() const;
    SizeT gradient_count() const;
    psq_native::StrainParameters strain_parameters() const;
    void energy(cuda_tool::CBufferView<Vector3> x, cuda_tool::BufferView<Float> E, Float dt);
    void gradient(cuda_tool::CBufferView<Vector3> x, cuda_tool::DoubletVectorView<Float,3> G, Float dt);
    Float feasible_step(Float alpha,SizeT frame=0,SizeT iteration=0);
    // Failure-only diagnostic: freeze b(x) at the current iterate for energy probes.
    void diagnostic_freeze_offset(bool enabled);
private:
    struct Impl;
    std::unique_ptr<Impl> m;
};
bool psq_replaces_bending();
bool psq_preview_eligible(FiniteElementMethod& fem);
Float psq_feasible_step(FiniteElementMethod& fem, Float alpha,SizeT frame=0,SizeT iteration=0);
void psq_diagnostic_freeze_offset(FiniteElementMethod& fem, bool enabled);
}
