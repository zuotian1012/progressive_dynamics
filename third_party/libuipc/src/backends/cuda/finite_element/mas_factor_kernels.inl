// Included inside mas_preconditioner_engine.cu's anonymous namespace.
// Full precision factor solves; no explicit inverse in the PCG application.
using ClusterFactor = MASPreconditionerEngine::ClusterFactor;

__device__ double mas_entry(const ClusterMatrixSym& h, int r, int c)
{
    const int br = r / 3, bc = c / 3;
    return br <= bc ? h.M[sym_index(br, bc)](r % 3, c % 3)
                    : h.M[sym_index(bc, br)](c % 3, r % 3);
}

__global__ void MASPreconditionerEngine_factor_kernel(
    cuda_tool::BufferView<ClusterFactor> factors,
    cuda_tool::CBufferView<ClusterMatrixSym> hess,
    cuda_tool::VarView<int> shifted_count)
{
    constexpr int N = BANKSIZE * 3;
    const int b = blockIdx.x, row = threadIdx.x;
    __shared__ double L[N][N], scale[N], bounds[N], shift;
    __shared__ int failed;
    if(row < N)
    {
        double d = mas_entry(hess(b), row, row);
        UIPC_KERNEL_ASSERT(::isfinite(d),
                           "MAS invalid diagonal: cluster=%d row=%d", b, row);
        double row_max = 0;
        for(int c = 0; c < N; ++c)
        {
            double v = mas_entry(hess(b), row, c);
            UIPC_KERNEL_ASSERT(::isfinite(v), "MAS non-finite entry: cluster=%d row=%d col=%d", b, row, c);
            row_max = fmax(row_max, fabs(v));
        }
        // Galerkin summation can lose a small positive diagonal to cancellation.
        // Scale without changing its sign; the uniform shift below repairs only P.
        scale[row] = row_max > 0 ? sqrt(fmax(fabs(d), row_max * 1e-12)) : 1.0;
    }
    if(row == 0) shift = 0;
    __syncthreads();
    for(int attempt = 0; attempt < 10; ++attempt)
    {
        if(row == 0) failed = 0;
        if(row < N)
        {
            for(int c = 0; c < N; ++c)
            {
                double v = .5 * (mas_entry(hess(b), row, c) / scale[row] / scale[c]
                               + mas_entry(hess(b), c, row) / scale[c] / scale[row]);
                // An exactly empty padded row gets identity.
                if(row == c && v == 0) {
                    bool empty = true;
                    for(int j = 0; j < N; ++j) empty = empty && mas_entry(hess(b), row, j) == 0;
                    if(empty) v = 1;
                }
                L[row][c] = v + (row == c ? shift : 0);
            }
            double bound = -L[row][row] + shift;
            for(int c = 0; c < N; ++c) if(c != row) bound += fabs(L[row][c]);
            bounds[row] = bound;
        }
        __syncthreads();
        for(int k = 0; k < N; ++k)
        {
            if(row == k)
            {
                double pivot = L[k][k];
                for(int j = 0; j < k; ++j) pivot = fma(-L[k][j], L[k][j], pivot);
                if(!::isfinite(pivot) || pivot <= 1e-10) failed = 1;
                else L[k][k] = sqrt(pivot);
            }
            __syncthreads();
            if(failed) break;
            if(row > k && row < N)
            {
                double v = L[row][k];
                for(int j = 0; j < k; ++j) v = fma(-L[row][j], L[k][j], v);
                L[row][k] = v / L[k][k];
            }
            __syncthreads();
        }
        if(!failed) break;
        if(row == 0)
        {
            shift = shift == 0 ? 1e-10 : shift * 10;
            if(attempt == 8) {
                double bound = 0;
                for(int i = 0; i < N; ++i) bound = fmax(bound, bounds[i]);
                shift = fmax(shift, bound + fmax(1e-8, bound * 1e-10));
            }
        }
        __syncthreads();
    }
    UIPC_KERNEL_ASSERT(!failed, "MAS factorization failed: cluster=%d", b);
    if(row < N)
    {
        factors(b).scale[row] = scale[row];
        for(int c = 0; c < N; ++c) factors(b).lower[row * N + c] = c <= row ? L[row][c] : 0;
    }
    if(row == 0)
    {
        factors(b).shift = shift;
        if(shift > 0) atomicAdd(shifted_count.data(), 1);
    }
}

// A warp owns one cluster. Each lane owns up to two scalar rows.
__global__ void MASPreconditionerEngine_factor_solve_kernel(
    cuda_tool::CBufferView<ClusterFactor> factors,
    cuda_tool::CBufferView<Eigen::Vector3d> rhs,
    cuda_tool::BufferView<double3> out,
    cuda_tool::CDense<IndexT> converged, int clusters)
{
    constexpr int N = BANKSIZE * 3;
    int lane = threadIdx.x % 32;
    int b = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if(b >= clusters || *converged != 0) return;
    auto& f = factors(b);
    double v[2];
    for(int t = 0; t < 2; ++t)
    {
        int row = lane + 32 * t;
        v[t] = row < N ? rhs(b * BANKSIZE + row / 3)[row % 3] / f.scale[row] : 0;
    }
    for(int k = 0; k < N; ++k)
    {
        double y = __shfl_sync(0xffffffff, v[k / 32], k % 32) / f.lower[k * N + k];
        for(int t = 0; t < 2; ++t)
        {
            int row = lane + 32 * t;
            if(row == k) v[t] = y;
            if(row > k && row < N) v[t] = fma(-f.lower[row * N + k], y, v[t]);
        }
    }
    for(int k = N - 1; k >= 0; --k)
    {
        double z = __shfl_sync(0xffffffff, v[k / 32], k % 32) / f.lower[k * N + k];
        for(int t = 0; t < 2; ++t)
        {
            int row = lane + 32 * t;
            if(row == k) v[t] = z;
            if(row < k) v[t] = fma(-f.lower[k * N + row], z, v[t]);
        }
    }
    for(int t = 0; t < 2; ++t)
    {
        int row = lane + 32 * t;
        if(row < N)
        {
            double value = v[t] / f.scale[row];
            double3& z = out(b * BANKSIZE + row / 3);
            if(row % 3 == 0) z.x = value;
            if(row % 3 == 1) z.y = value;
            if(row % 3 == 2) z.z = value;
        }
    }
}
