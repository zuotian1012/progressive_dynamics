#include <finite_element/progressive_cloth.h>
#include <finite_element/constitutions/discrete_shell_bending_function.h>
#include <finite_element/constitutions/strain_limiting_baraff_witkin_shell_2d.h>
#include <utils/make_spd.h>
#include <utils/matrix_market.h>
#include <uipc/common/json.h>
#include <uipc/builtin/attribute_name.h>

#include <filesystem>
#include <fstream>
#include <map>
#include <limits>

namespace uipc::backend::cuda
{
namespace
{
namespace BWS = sym::strainlimiting_baraff_witkin_shell_2d;
namespace DSB = sym::discrete_shell_bending;

struct WeightedVertex
{
    IndexT id;
    Float weight;
};

struct BarycentricRow
{
    IndexT begin, end;
};

struct TransferColumn
{
    IndexT id;
    Vector4 weights;
};

struct ShellOffset
{
    Vector3i ids;
    Matrix2x2 inv_rest;
    Vector3 local;
};

UIPC_GENERIC Vector3 shell_offset(const Vector3& a, const Vector3& b, const Vector3& c,
                                 const ShellOffset& offset)
{
    if(offset.local.squaredNorm() == 0)
        return Vector3::Zero();
    const Vector3 e = b - a;
    const Vector3 f = c - a;
    const Vector3 cross = e.cross(f);
    const Float length = cross.norm();
    UIPC_KERNEL_ASSERT(length > 0, "Degenerate PSQ support triangle");
    const Vector3 normal = cross / length;
    Matrix<Float, 3, 2> edges;
    edges << e, f;
    const Matrix<Float, 3, 2> F = edges * offset.inv_rest;
    // Thin polar factor R = F (F^T F)^(-1/2), using the 2x2 SPD square root.
    const Matrix2x2 C = F.transpose() * F;
    const Float det_sqrt = length * offset.inv_rest.determinant();
    const Matrix2x2 shifted = C + det_sqrt * Matrix2x2::Identity();
    // det(C + d I) = d (trace(C) + 2d); avoid cancellation in Eigen's 2x2 inverse.
    const Float trace = C.trace() + 2 * det_sqrt;
    Matrix2x2 adjugate;
    adjugate << shifted(1, 1), -shifted(0, 1), -shifted(1, 0), shifted(0, 0);
    return normal * offset.local[2]
           + F * (adjugate * offset.local.head<2>()) / (det_sqrt * sqrt(trace));
}

// Differential of the nonlinear PSQ offset. For the thin polar frame
// R=[r0,r1,n], dq=omega x q, where omega=n x dn + twist*n.
struct OffsetJacobian
{
    Vector3i ids;
    Matrix3x3 db, dc;  // da = -(db + dc) by translation invariance
};

__global__ void compute_offset_jacobians(cuda_tool::CBufferView<Vector3> coarse,
    cuda_tool::CBufferView<ShellOffset> offsets, IndexT base,
    cuda_tool::BufferView<OffsetJacobian> result)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= offsets.size()) return;
    const auto& shell = offsets(i);
    auto& jac = result(i);
    jac.ids = shell.ids.array() + base;
    jac.db.setZero();
    jac.dc.setZero();
    if(shell.local.squaredNorm() == 0) return;
    const Vector3 a = coarse(base + shell.ids[0]);
    const Vector3 e = coarse(base + shell.ids[1]) - a;
    const Vector3 f = coarse(base + shell.ids[2]) - a;
    const Vector3 cross = e.cross(f);
    const Float area2 = cross.norm();
    const Vector3 n = cross / area2;
    Matrix<Float,3,2> edges;
    edges << e, f;
    const Matrix<Float,3,2> F = edges * shell.inv_rest;
    const Vector3 polar0 = F.col(0) + F.col(1).cross(n);
    const Float trace_stretch = polar0.norm();
    const Vector3 r0 = polar0 / trace_stretch;
    const Vector3 r1 = n.cross(r0);
    const Vector3 q = r0 * shell.local[0] + r1 * shell.local[1] + n * shell.local[2];
    for(int k = 0; k < 3; ++k)
    {
        const Vector3 axis = Vector3::Unit(k);
        for(int side = 0; side < 2; ++side)
        {
            const Vector3 dcross = side == 0 ? axis.cross(f) : e.cross(axis);
            const Float twist = (r1[k] * shell.inv_rest(side,0) - r0[k] * shell.inv_rest(side,1)) / trace_stretch;
            const Vector3 omega = n.cross(dcross) / area2 + twist * n;
            (side == 0 ? jac.db : jac.dc).col(k) = omega.cross(q);
        }
    }
}

template <int N>
struct Transfer
{
    Vector<IndexT, N> fine_ids;
    IndexT count = 0;
    IndexT gradient_offset = 0;
    IndexT hessian_offset = 0;
};

struct Membrane
{
    Transfer<3> transfer;
    Matrix2x2 inv_rest;
    Float area;
};

struct Hinge
{
    Transfer<4> transfer;
    Float length, height, angle, weight;
};

struct Material
{
    Float stretch, shear, strain_rate, kappa;
};

template <typename T>
void upload(cuda_tool::DeviceBuffer<T>& device, const vector<T>& host)
{
    device.resize(host.size());
    device.view().copy_from(host.data());
}

template <typename View>
void write_audit_buffer(const std::filesystem::path& path, View view)
{
    using T = std::remove_cv_t<std::remove_pointer_t<decltype(view.data())>>;
    vector<T> host(view.size());
    view.copy_to(host.data());
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(host.data()), host.size() * sizeof(T));
    UIPC_ASSERT_THROW(file.good(), "Could not write {}", path.string());
}

template <int N>
Transfer<N> make_transfer(const Vector<IndexT, N>& fine_ids,
                          span<const BarycentricRow> rows,
                          span<const WeightedVertex> weights,
                          span<const ShellOffset> offsets,
                          IndexT coarse_offset,
                          vector<TransferColumn>& columns,
                          SizeT& hessian_count)
{
    Transfer<N> transfer;
    transfer.fine_ids = fine_ids;
    std::map<IndexT, Vector4> local_columns;
    for(int i = 0; i < N; ++i)
    {
        const auto row = rows[fine_ids[i]];
        if(!offsets.empty() && offsets[fine_ids[i]].local.squaredNorm() > 0)
            for(int k = 0; k < 3; ++k)
                local_columns.try_emplace(offsets[fine_ids[i]].ids[k] + coarse_offset, Vector4::Zero());
        for(auto j = row.begin; j < row.end; ++j)
        {
            const auto entry = weights[j];
            auto it = local_columns.try_emplace(entry.id + coarse_offset, Vector4::Zero()).first;
            it->second[i] += entry.weight;
        }
    }
    transfer.gradient_offset = static_cast<IndexT>(columns.size());
    transfer.count = static_cast<IndexT>(local_columns.size());
    for(const auto& [id, weight] : local_columns)
        columns.push_back({id, weight});
    transfer.hessian_offset = static_cast<IndexT>(hessian_count);
    hessian_count += static_cast<SizeT>(transfer.count) * (transfer.count + 1) / 2;
    UIPC_ASSERT_THROW(hessian_count <= std::numeric_limits<IndexT>::max()
                          && columns.size() <= std::numeric_limits<IndexT>::max(),
                      "Progressive cloth exceeds CUDA sparse assembly index range");
    return transfer;
}

__global__ void prolongate(cuda_tool::CBufferView<Vector3> coarse,
                          cuda_tool::CBufferView<BarycentricRow> rows,
                          cuda_tool::CBufferView<WeightedVertex> weights,
                          cuda_tool::CBufferView<ShellOffset> offsets,
                          cuda_tool::BufferView<Vector3> fine,
                          IndexT offset)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= rows.size())
        return;
    const auto row = rows(i);
    Vector3 x = Vector3::Zero();
    for(auto j = row.begin; j < row.end; ++j)
        x += weights(j).weight * coarse(offset + weights(j).id);
    if(offsets.size() > 0)
    {
        const auto& shell = offsets(i);
        x += shell_offset(coarse(offset + shell.ids[0]), coarse(offset + shell.ids[1]),
                          coarse(offset + shell.ids[2]), shell);
    }
    fine(i) = x;
}

template <int N>
__device__ void restrict_gradient(const Transfer<N>& transfer,
                                 const Vector<Float, 3 * N>& local,
                                 cuda_tool::CBufferView<TransferColumn> columns,
                                 cuda_tool::CBufferView<OffsetJacobian> offset_jacobians,
                                 cuda_tool::DoubletVectorView<Float, 3> gradients)
{
    for(int col = 0; col < transfer.count; ++col)
    {
        const auto& column = columns(transfer.gradient_offset + col);
        Vector3 block = Vector3::Zero();
        for(int i = 0; i < N; ++i)
        {
            const Vector3 g = local.template segment<3>(3 * i);
            block += column.weights[i] * g;
            if(offset_jacobians.size())
            {
                const auto& jac = offset_jacobians(transfer.fine_ids[i]);
                if(column.id == jac.ids[0]) block -= (jac.db + jac.dc).transpose() * g;
                if(column.id == jac.ids[1]) block += jac.db.transpose() * g;
                if(column.id == jac.ids[2]) block += jac.dc.transpose() * g;
            }
        }
        gradients(transfer.gradient_offset + col).write(column.id, block);
    }
}

template <int N>
__device__ void restrict_hessian(const Transfer<N>& transfer,
                                const Matrix<Float, 3 * N, 3 * N>& local,
                                cuda_tool::CBufferView<TransferColumn> columns,
                                cuda_tool::TripletMatrixView<Float, 3> hessians)
{
    IndexT entry = transfer.hessian_offset;
    for(int row = 0; row < transfer.count; ++row)
    {
        for(int col = row; col < transfer.count; ++col)
        {
            const auto& left = columns(transfer.gradient_offset + row);
            const auto& right = columns(transfer.gradient_offset + col);
            Matrix3x3 block = Matrix3x3::Zero();
            for(int i = 0; i < N; ++i)
                for(int j = 0; j < N; ++j)
                    block += (left.weights[i] * right.weights[j])
                             * local.template block<3, 3>(3 * i, 3 * j);
            hessians(entry++).write(left.id, right.id, block);
        }
    }
}

__device__ Matrix<Float, 3, 2> deformation(const Membrane& triangle,
                                            cuda_tool::CBufferView<Vector3> xs)
{
    const auto& ids = triangle.transfer.fine_ids;
    return BWS::Ds3x2(xs(ids[0]), xs(ids[1]), xs(ids[2])) * triangle.inv_rest;
}

// Analytic derivative of the existing BWS gradient, before its per-term projection.
__device__ Matrix6x6 raw_membrane_hessian(const Matrix<Float, 3, 2>& F, Material material)
{
    Matrix6x6 H = Matrix6x6::Zero();
    for(int axis = 0; axis < 2; ++axis)
    {
        Vector3 f = F.col(axis);
        Float squared = f.squaredNorm();
        Float r = sqrt(max(squared, Float{1e-12}));
        Float scaled = material.strain_rate;
        Float tangent = 2 * (1 - 1 / r);
        Float radial = 2;
        if(r > 1)
        {
            tangent += 3 * scaled * (r + 1 / r - 2);
            radial += 6 * scaled * (r - 1);
        }
        Matrix3x3 block = tangent * Matrix3x3::Identity();
        if(squared > 1e-12)
            block += ((radial - tangent) / squared) * (f * f.transpose());
        H.block<3, 3>(3 * axis, 3 * axis) = material.stretch * block;
    }
    Vector6 shear_gradient;
    shear_gradient << F.col(1), F.col(0);
    H += (2 * material.shear) * (shear_gradient * shear_gradient.transpose());
    Float coupling = 2 * material.shear * F.col(0).dot(F.col(1));
    H.block<3, 3>(0, 3).diagonal().array() += coupling;
    H.block<3, 3>(3, 0).diagonal().array() += coupling;
    return H;
}

__global__ void membrane_energy(cuda_tool::CBufferView<Membrane> triangles,
                                cuda_tool::CBufferView<Vector3> xs,
                                Material material,
                                Float dt2,
                                cuda_tool::BufferView<Float> energies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= triangles.size())
        return;
    const auto& tri = triangles(i);
    energies(i) = (tri.area * dt2)
                  * BWS::E(deformation(tri, xs), Vector2::UnitX(), Vector2::UnitY(),
                           material.stretch, material.shear, material.strain_rate);
}

__global__ void hinge_energy(cuda_tool::CBufferView<Hinge> hinges,
                             cuda_tool::CBufferView<Vector3> xs,
                             Material material,
                             Float dt2,
                             cuda_tool::BufferView<Float> energies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= hinges.size())
        return;
    const auto& hinge = hinges(i);
    const auto& ids = hinge.transfer.fine_ids;
    energies(i) = (hinge.weight * dt2)
                  * DSB::E(xs(ids[0]), xs(ids[1]), xs(ids[2]), xs(ids[3]),
                           hinge.length, hinge.height, hinge.angle, material.kappa);
}

__global__ void assemble_membrane(cuda_tool::CBufferView<Membrane> triangles,
                                  cuda_tool::CBufferView<Vector3> xs,
                                  cuda_tool::CBufferView<TransferColumn> columns,
                                cuda_tool::CBufferView<OffsetJacobian> offset_jacobians,
                                Material material,
                                  Float dt2,
                                  cuda_tool::DoubletVectorView<Float, 3> gradients,
                                  cuda_tool::TripletMatrixView<Float, 3> hessians,
                                  bool gradient_only,
                                  bool project)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= triangles.size())
        return;
    const auto& tri = triangles(i);
    const auto F = deformation(tri, xs);
    const auto jacobian = BWS::dFdX(tri.inv_rest);
    Float scale = tri.area * dt2;
    Matrix<Float, 3, 2> dEdF;
    BWS::dEdF(dEdF, F, Vector2::UnitX(), Vector2::UnitY(),
              material.stretch, material.shear, material.strain_rate);
    Vector9 local_gradient = scale * (jacobian.transpose() * BWS::flatten(dEdF));
    restrict_gradient(tri.transfer, local_gradient, columns, offset_jacobians, gradients);
    if(gradient_only)
        return;

    Matrix6x6 H;
    if(project)
        BWS::ddEddF(H, F, Vector2::UnitX(), Vector2::UnitY(),
                    material.stretch, material.shear, material.strain_rate);
    else
        H = raw_membrane_hessian(F, material);
    Matrix9x9 local_hessian = jacobian.transpose() * (scale * H) * jacobian;
    restrict_hessian(tri.transfer, local_hessian, columns, hessians);
}

__global__ void assemble_hinges(cuda_tool::CBufferView<Hinge> hinges,
                                cuda_tool::CBufferView<Vector3> xs,
                                cuda_tool::CBufferView<TransferColumn> columns,
                                cuda_tool::CBufferView<OffsetJacobian> offset_jacobians,
                                Material material,
                                Float dt2,
                                cuda_tool::DoubletVectorView<Float, 3> gradients,
                                cuda_tool::TripletMatrixView<Float, 3> hessians,
                                bool gradient_only,
                                bool project)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= hinges.size())
        return;
    const auto& hinge = hinges(i);
    const auto& ids = hinge.transfer.fine_ids;
    const auto& a = xs(ids[0]);
    const auto& b = xs(ids[1]);
    const auto& c = xs(ids[2]);
    const auto& d = xs(ids[3]);
    Float scale = hinge.weight * dt2;
    Vector12 local_gradient;
    DSB::dEdx(local_gradient, a, b, c, d, hinge.length, hinge.height,
              hinge.angle, material.kappa);
    local_gradient *= scale;
    restrict_gradient(hinge.transfer, local_gradient, columns, offset_jacobians, gradients);
    if(gradient_only)
        return;

    Matrix12x12 H;
    DSB::ddEddx(H, a, b, c, d, hinge.length, hinge.height, hinge.angle, material.kappa);
    H *= scale;
    if(project)
        make_spd(H);
    restrict_hessian(hinge.transfer, H, columns, hessians);
}
}  // namespace

struct ProgressiveCloth::Impl
{
    IndexT coarse_offset;
    SizeT gradient_count = 0;
    SizeT hessian_count = 0;
    Material material;
    std::filesystem::path validation_path;
    std::filesystem::path audit_directory;
    SizeT audit_frame = 0;
    SizeT coarse_count = 0;
    bool audit_recorded = false;
    bool recorded_energy = false;
    bool logged_line_search = false;
    cuda_tool::DeviceBuffer<BarycentricRow> rows;
    cuda_tool::DeviceBuffer<WeightedVertex> row_weights;
    cuda_tool::DeviceBuffer<ShellOffset> shell_offsets;
    cuda_tool::DeviceBuffer<OffsetJacobian> offset_jacobians;
    cuda_tool::DeviceBuffer<TransferColumn> transfer_columns;
    cuda_tool::DeviceBuffer<Vector3> fine_positions;
    cuda_tool::DeviceBuffer<Membrane> triangles;
    cuda_tool::DeviceBuffer<Hinge> hinges;

    void predict(cuda_tool::CBufferView<Vector3> xs)
    {
        auto kernel = prolongate;
        int count = static_cast<int>(rows.size());
        kernel<<<cuda_tool::best_grid_dim(count, kernel), cuda_tool::best_block_dim(kernel)>>>(
            xs, rows.cview(), row_weights.cview(), shell_offsets.cview(), fine_positions.view(), coarse_offset);
    }

    void assemble(Float dt,
                  cuda_tool::DoubletVectorView<Float, 3> gradients,
                  cuda_tool::TripletMatrixView<Float, 3> hessians,
                  bool gradient_only,
                  bool project)
    {
        auto membrane = assemble_membrane;
        int count = static_cast<int>(triangles.size());
        membrane<<<cuda_tool::best_grid_dim(count, membrane), cuda_tool::best_block_dim(membrane)>>>(
            triangles.cview(), fine_positions.cview(), transfer_columns.cview(), offset_jacobians.cview(), material, dt * dt, gradients, hessians,
            gradient_only, project);
        if(hinges.size())
        {
            auto bending = assemble_hinges;
            count = static_cast<int>(hinges.size());
            bending<<<cuda_tool::best_grid_dim(count, bending), cuda_tool::best_block_dim(bending)>>>(
                hinges.cview(), fine_positions.cview(), transfer_columns.cview(), offset_jacobians.cview(), material, dt * dt, gradients, hessians,
                gradient_only, project);
        }
    }
};

ProgressiveCloth::ProgressiveCloth(const geometry::SimplicialComplex& mesh,
                                   span<const Vector3> coarse_rest,
                                   IndexT coarse_offset,
                                   std::string_view asset)
    : m_impl(uipc::make_unique<Impl>())
{
    auto& impl = *m_impl;
    impl.coarse_offset = coarse_offset;
    impl.coarse_count = coarse_rest.size();
    if(auto directory = mesh.meta().find<std::string>("pd_cloth_audit_directory"))
    {
        impl.audit_directory = directory->view().front();
        impl.audit_frame = std::stoull(mesh.meta().find<std::string>("pd_cloth_audit_frame")
                                          ->view().front());
    }
    auto stretch = mesh.triangles().find<Float>("lambda");
    auto shear = mesh.triangles().find<Float>("mu");
    auto strain_rate = mesh.triangles().find<Float>("strain_rate");
    auto kappa = mesh.meta().find<Float>("pd_cloth_proxy_kappa");
    UIPC_ASSERT_THROW(stretch && shear && strain_rate && kappa && mesh.triangles().size(),
                      "Progressive cloth requires uniform native BWS material and bending stiffness");
    auto uniform = [](auto& slot)
    {
        auto values = slot->view();
        Float value = values.front();
        bool is_uniform = std::all_of(values.begin(), values.end(),
                                      [value](Float v) { return v == value; });
        UIPC_ASSERT_THROW(std::isfinite(value) && value >= 0 && is_uniform,
                          "Progressive cloth material must be finite, nonnegative and uniform");
        return value;
    };
    impl.material = {uniform(stretch), uniform(shear), uniform(strain_rate), uniform(kappa)};
    auto thickness = mesh.vertices().find<Float>(builtin::thickness);
    UIPC_ASSERT_THROW(thickness, "Progressive cloth requires a thickness attribute");
    logger::info("PD proxy material: stretch={} shear={} strain_rate={} kappa={} thickness={}",
                 impl.material.stretch, impl.material.shear, impl.material.strain_rate,
                 impl.material.kappa, uniform(thickness));

    std::ifstream input{std::string{asset}};
    UIPC_ASSERT_THROW(input.good(), "Cannot open progressive cloth asset {}", asset);
    auto data = Json::parse(input);
    const auto& positions = data.at("rest");
    const auto& columns = data.at("columns");
    const auto& weights = data.at("weights");
    UIPC_ASSERT_THROW(positions.is_array() && !positions.empty()
                         && positions.size() == columns.size() && positions.size() == weights.size()
                         && data.at("coarse_vertices") == coarse_rest.size(),
                      "Progressive cloth asset dimensions do not match the coarse geometry");
    vector<Vector3> rest(positions.size());
    vector<BarycentricRow> rows(positions.size());
    vector<WeightedVertex> row_weights;
    vector<ShellOffset> shell_offsets;
    vector<TransferColumn> transfer_columns;
    const bool use_psq = data.contains("psq");
    if(use_psq)
    {
        const auto& psq = data.at("psq");
        UIPC_ASSERT_THROW(psq.at("faces").size() == rest.size() && psq.at("offsets").size() == rest.size(),
                          "PSQ support triangles and offsets must match fine vertices");
        shell_offsets.reserve(rest.size());
    }
    for(SizeT i = 0; i < rest.size(); ++i)
    {
        const auto xyz = positions[i].get<std::array<Float, 3>>();
        rest[i] = Vector3{xyz[0], xyz[1], xyz[2]};
        const auto ids = columns[i].get<std::vector<IndexT>>();
        const auto w = weights[i].get<std::vector<Float>>();
        UIPC_ASSERT_THROW(ids.size() == w.size() && !ids.empty(), "Invalid progressive cloth row {}", i);
        Vector3 reconstructed = Vector3::Zero();
        Float sum = 0;
        rows[i].begin = static_cast<IndexT>(row_weights.size());
        for(SizeT j = 0; j < ids.size(); ++j)
        {
            UIPC_ASSERT_THROW(ids[j] >= 0 && ids[j] < coarse_rest.size()
                                 && std::isfinite(w[j]) && w[j] >= 0,
                              "Invalid progressive cloth barycentric row {}", i);
            if(w[j] != 0)
                row_weights.push_back({ids[j], w[j]});
            reconstructed += w[j] * coarse_rest[ids[j]];
            sum += w[j];
        }
        UIPC_ASSERT_THROW(row_weights.size() <= std::numeric_limits<IndexT>::max(),
                          "Progressive cloth exceeds CUDA interpolation index range");
        rows[i].end = static_cast<IndexT>(row_weights.size());
        if(use_psq)
        {
            auto face = data["psq"]["faces"][i].get<std::array<IndexT, 3>>();
            auto local = data["psq"]["offsets"][i].get<std::array<Float, 3>>();
            for(auto id : face)
                UIPC_ASSERT_THROW(id >= 0 && id < coarse_rest.size(), "Invalid PSQ support triangle");
            const auto& a = coarse_rest[face[0]];
            const auto& b = coarse_rest[face[1]];
            const auto& c = coarse_rest[face[2]];
            UIPC_ASSERT_THROW((b - a).cross(c - a).norm() > 1e-16, "Degenerate PSQ rest triangle");
            const Vector3 edge = b - a;
            const Vector3 side = c - a;
            Matrix2x2 reference;
            reference << edge.norm(), edge.normalized().dot(side),
                         0, edge.cross(side).norm() / edge.norm();
            ShellOffset support{Vector3i{face[0], face[1], face[2]},
                                reference.inverse(),
                                Vector3{local[0], local[1], local[2]}};
            reconstructed += shell_offset(a, b, c, support);
            shell_offsets.push_back(support);
        }
        // Genesis applies its OBJ rotation and translation in FP32 by default.
        Float rest_error = (rest[i] - reconstructed).lpNorm<Eigen::Infinity>();
        UIPC_ASSERT_THROW(std::abs(sum - 1.0) < 1e-12 && rest_error < 1e-5,
                          "Progressive cloth rest pose or coarse vertex order mismatch at {} (error {})",
                          i, rest_error);
    }
    vector<Membrane> triangles;
    std::map<std::pair<IndexT, IndexT>, vector<IndexT>> opposite_vertices;
    for(const auto& face : data.at("triangles"))
    {
        auto ids = face.get<std::array<IndexT, 3>>();
        for(auto id : ids)
            UIPC_ASSERT_THROW(id >= 0 && id < rest.size(), "Invalid progressive cloth triangle vertex");
        Vector3i tri{ids[0], ids[1], ids[2]};
        const auto& a = rest[tri[0]];
        const auto& b = rest[tri[1]];
        const auto& c = rest[tri[2]];
        Float area = (b - a).cross(c - a).norm() / 2;
        UIPC_ASSERT_THROW(area > 0 && std::isfinite(area), "Degenerate progressive cloth triangle");
        triangles.push_back({make_transfer(tri, rows, row_weights, shell_offsets, coarse_offset, transfer_columns,
                                           impl.hessian_count),
                             BWS::Dm2x2(a, b, c).inverse(), area});
        for(int i = 0; i < 3; ++i)
        {
            auto [left, right] = std::minmax(tri[(i + 1) % 3], tri[(i + 2) % 3]);
            opposite_vertices[{left, right}].push_back(tri[i]);
        }
    }
    UIPC_ASSERT_THROW(!triangles.empty(), "Progressive cloth requires fine triangles");
    vector<Hinge> hinges;
    if(impl.material.kappa > 0)
    {
        for(auto& [edge, opposite] : opposite_vertices)
        {
            UIPC_ASSERT_THROW(opposite.size() <= 2, "Non-manifold progressive cloth edge");
            if(opposite.size() != 2)
                continue;
            std::sort(opposite.begin(), opposite.end());
            Vector4i ids{opposite[0], edge.first, edge.second, opposite[1]};
            Hinge hinge;
            hinge.transfer = make_transfer(ids, rows, row_weights, shell_offsets, coarse_offset, transfer_columns,
                                           impl.hessian_count);
            DSB::compute_constants(hinge.length, hinge.height, hinge.angle, hinge.weight,
                                   rest[ids[0]], rest[ids[1]], rest[ids[2]], rest[ids[3]],
                                   0, 0, 0, 0);
            hinges.push_back(hinge);
        }
    }
    if(auto validation = mesh.meta().find<std::string>("pd_cloth_proxy_validation"))
        impl.validation_path = validation->view().front();
    impl.gradient_count = transfer_columns.size();
    upload(impl.rows, rows);
    upload(impl.row_weights, row_weights);
    if(use_psq)
        {
        upload(impl.shell_offsets, shell_offsets);
        impl.offset_jacobians.resize(shell_offsets.size());
    }
    upload(impl.transfer_columns, transfer_columns);
    upload(impl.triangles, triangles);
    if(!hinges.empty())
        upload(impl.hinges, hinges);
    impl.fine_positions.resize(rest.size());
}

ProgressiveCloth::~ProgressiveCloth() = default;

SizeT ProgressiveCloth::energy_count() const
{
    return m_impl->triangles.size() + m_impl->hinges.size();
}

SizeT ProgressiveCloth::gradient_count() const { return m_impl->gradient_count; }
SizeT ProgressiveCloth::hessian_count() const { return m_impl->hessian_count; }

void ProgressiveCloth::compute_energy(cuda_tool::CBufferView<Vector3> xs,
                                       Float dt,
                                       cuda_tool::BufferView<Float> energies)
{
    auto& impl = *m_impl;
    impl.predict(xs);
    if(!impl.logged_line_search)
    {
        logger::info("PD proxy compute_energy: membrane_elements={} bending_hinges={}",
                     impl.triangles.size(), impl.hinges.size());
        impl.logged_line_search = true;
    }
    int count = static_cast<int>(impl.triangles.size());
    auto membrane = membrane_energy;
    membrane<<<cuda_tool::best_grid_dim(count, membrane), cuda_tool::best_block_dim(membrane)>>>(
        impl.triangles.cview(), impl.fine_positions.cview(), impl.material, dt * dt,
        energies.subview(0, count));
    if(impl.hinges.size())
    {
        int offset = count;
        count = static_cast<int>(impl.hinges.size());
        auto bending = hinge_energy;
        bending<<<cuda_tool::best_grid_dim(count, bending), cuda_tool::best_block_dim(bending)>>>(
            impl.hinges.cview(), impl.fine_positions.cview(), impl.material, dt * dt,
            energies.subview(offset, count));
    }
    if(!impl.validation_path.empty() && !impl.recorded_energy)
    {
        vector<Float> values(energy_count());
        energies.copy_to(values.data());
        std::ofstream output{impl.validation_path.string() + ".energy"};
        output.precision(17);
        Float sum = 0;
        for(Float value : values)
            sum += value;
        output << sum << '\n';
        UIPC_ASSERT_THROW(output.good(), "Could not write progressive cloth line-search energy");
        impl.recorded_energy = true;
    }
}

void ProgressiveCloth::assemble(cuda_tool::CBufferView<Vector3> xs,
                                 Float dt,
                                 cuda_tool::DoubletVectorView<Float, 3> gradients,
                                 cuda_tool::TripletMatrixView<Float, 3> hessians,
                                 bool gradient_only, SizeT frame, SizeT newton_iter)
{
    auto& impl = *m_impl;
    impl.predict(xs);
    // Use the exact elastic gradient with the existing positive-semidefinite
    // U-transferred Hessian approximation as the search metric.
    if(impl.shell_offsets.size())
    {
        const int n = static_cast<int>(impl.shell_offsets.size());
        auto kernel = compute_offset_jacobians;
        kernel<<<cuda_tool::best_grid_dim(n,kernel),cuda_tool::best_block_dim(kernel)>>>(
            xs,impl.shell_offsets.cview(),impl.coarse_offset,impl.offset_jacobians.view());
    }
    impl.assemble(dt, gradients, hessians, gradient_only, true);

    if(!gradient_only && !impl.audit_recorded && impl.audit_frame == frame
       && newton_iter == 0 && !impl.audit_directory.empty())
    {
        static_assert(sizeof(Vector3) == 3 * sizeof(Float));
        static_assert(sizeof(Matrix3x3) == 9 * sizeof(Float));
        const auto& dir = impl.audit_directory;
        logger::info("PD proxy assemble: frame={} Newton={} gradient_blocks={} hessian_blocks={}",
                     frame, newton_iter, gradient_count(), hessian_count());
        write_audit_buffer(dir / "x_newton.f64", xs.subview(impl.coarse_offset, impl.coarse_count));
        if(impl.shell_offsets.size())
            write_audit_buffer(dir / "psq_fine_positions.f64", impl.fine_positions.cview());
        write_audit_buffer(dir / "g_indices.i32", gradients.indices());
        write_audit_buffer(dir / "g_blocks.f64", gradients.values());
        write_audit_buffer(dir / "H_rows.i32", hessians.row_indices());
        write_audit_buffer(dir / "H_cols.i32", hessians.col_indices());
        write_audit_buffer(dir / "H_blocks.f64", hessians.values());
        auto h_offset = hessians.submatrix_offset();
        std::ofstream offsets(dir / "block_offsets.txt");
        offsets << gradients.subvector_offset() + impl.coarse_offset << ' '
                << h_offset.x + impl.coarse_offset << ' ' << h_offset.y + impl.coarse_offset << '\n';
        UIPC_ASSERT_THROW(offsets.good(), "Could not write audit block offsets");
        impl.audit_recorded = true;
    }

    if(!gradient_only && !impl.validation_path.empty()
       && !std::filesystem::exists(impl.validation_path))
    {
        // Self-check uses the first Newton iterate and retains the projected
        // production assembly in the FEM reporter buffers.
        cuda_tool::DeviceDoubletVector<Float, 3> raw_gradients;
        cuda_tool::DeviceTripletMatrix<Float, 3> raw_hessians;
        raw_gradients.resize_discard(xs.size(), gradient_count());
        raw_hessians.resize_discard(xs.size(), xs.size(), hessian_count());
        impl.assemble(dt, raw_gradients.view(), raw_hessians.view(), false, false);
        UIPC_ASSERT_THROW(export_matrix_market(impl.validation_path.string(), raw_hessians.cview()),
                          "Could not write raw progressive cloth Hessian");
    }
}
}  // namespace uipc::backend::cuda
