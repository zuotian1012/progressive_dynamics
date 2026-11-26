// Solve the identical assembled Newton system on the host for a direct-solve
// control. The native matrix stores one copy of each symmetric 3x3 block.
const auto matrix = bcoo_A.cview();
const auto count = matrix.triplet_count();
std::vector<IndexT> rows(count), cols(count);
std::vector<Matrix3x3> blocks(count);
matrix.row_indices().copy_to(rows.data());
matrix.col_indices().copy_to(cols.data());
matrix.values().copy_to(blocks.data());

using Sparse = Eigen::SparseMatrix<Float>;
using Dense = Eigen::Matrix<Float, Eigen::Dynamic, 1>;
std::vector<Eigen::Triplet<Float>> entries;
entries.reserve(9 * count);
for(SizeT k = 0; k < count; ++k)
{
    const bool diagonal = rows[k] == cols[k];
    for(int i = 0; i < 3; ++i)
        for(int j = 0; j < 3; ++j)
        {
            const Float value = blocks[k](i,j);
            if((diagonal && i > j) || value == 0)
                continue;
            const int row = 3 * rows[k] + i, col = 3 * cols[k] + j;
            entries.emplace_back(std::min(row,col), std::max(row,col), value);
        }
}
Sparse upper(b.size(), b.size());
upper.setFromTriplets(entries.begin(), entries.end());
Dense rhs(b.size());
b.cview().buffer_view().copy_to(rhs.data());
Eigen::SimplicialLDLT<Sparse, Eigen::Upper> factor;
factor.compute(upper);
if(factor.info() != Eigen::Success || !factor.vectorD().allFinite() || factor.vectorD().minCoeff() <= 0)
    throw std::runtime_error("Bunny sparse LDLT could not factor the positive-definite Newton matrix");
const Dense solution = factor.solve(rhs);
if(factor.info() != Eigen::Success || !solution.allFinite())
    throw std::runtime_error("Bunny sparse LDLT returned an invalid Newton solution");
const Dense residual = upper.selfadjointView<Eigen::Upper>() * solution - rhs;
const Float denominator = std::sqrt(2.) * upper.norm() * solution.norm() + rhs.norm();
const Float backward_error = denominator > 0 ? residual.norm() / denominator : residual.norm();
if(!std::isfinite(backward_error) || backward_error > 1e-12)
    throw std::runtime_error("Bunny sparse LDLT failed the backward error check");
x.view().buffer_view().copy_from(solution.data());
last_solve_iterations = 1;
logger::info("BUNNY_DIRECT_SOLVE dofs={} min_pivot={:.17g} rhs_norm={:.17g} residual_norm={:.17g} backward_error={:.17g}",
             b.size(), factor.vectorD().minCoeff(), rhs.norm(), residual.norm(), backward_error);
