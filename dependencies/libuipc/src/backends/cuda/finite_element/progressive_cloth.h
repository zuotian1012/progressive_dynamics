#pragma once
#include <finite_element/finite_element_method.h>

namespace uipc::backend::cuda
{
class ProgressiveCloth
{
  public:
    ProgressiveCloth(const geometry::SimplicialComplex& geometry,
                     span<const Vector3> coarse_rest,
                     IndexT coarse_offset,
                     std::string_view asset);
    ~ProgressiveCloth();

    SizeT energy_count() const;
    SizeT gradient_count() const;
    SizeT hessian_count() const;

    void compute_energy(cuda_tool::CBufferView<Vector3> xs,
                        Float dt,
                        cuda_tool::BufferView<Float> energies);
    void assemble(cuda_tool::CBufferView<Vector3> xs,
                  Float dt,
                  cuda_tool::DoubletVectorView<Float, 3> gradients,
                  cuda_tool::TripletMatrixView<Float, 3> hessians,
                  bool gradient_only, SizeT frame, SizeT newton_iter);

  private:
    struct Impl;
    U<Impl> m_impl;
};
}  // namespace uipc::backend::cuda
