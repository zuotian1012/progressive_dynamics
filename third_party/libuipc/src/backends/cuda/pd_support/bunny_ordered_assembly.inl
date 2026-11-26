// Host reference for the zero-friction bunny. Keep element evaluations on
// CUDA, but accumulate equal matrix or vector indices in a fixed order.
template <class Values>
static std::vector<int> bunny_term_order(const std::vector<int>& row,
                                        const std::vector<int>& col,
                                        const Values& values)
{
    std::vector<int> order(row.size());
    for(size_t i=0; i<row.size(); ++i)
    {
        if(!values[i].allFinite())
            throw std::runtime_error("Nonfinite FEM assembly term");
        order[i]=static_cast<int>(i);
    }
    std::sort(order.begin(),order.end(),[&](int a,int b) {
        if(row[a]!=row[b]) return row[a]<row[b];
        if(!col.empty() && col[a]!=col[b]) return col[a]<col[b];
        const auto& x=values[a];
        const auto& y=values[b];
        return std::lexicographical_compare(x.data(),x.data()+x.size(),y.data(),y.data()+y.size());
    });
    return order;
}

#if defined(BUNNY_FEM_GRADIENT)
static void bunny_assemble_gradient(cuda_tool::CDoubletVectorView<Float,3> src,
                                    cuda_tool::DenseVectorView<Float> dst,
                                    cuda_tool::CBufferView<IndexT> fixed, int offset)
{
    if(!src.doublet_count()) return;
    std::vector<int> index(src.doublet_count()), pinned(fixed.size());
    std::vector<Vector3> values(index.size());
    std::vector<Float> dense(dst.size());
    src.indices().copy_to(index.data());
    src.values().copy_to(values.data());
    fixed.copy_to(pinned.data());
    dst.buffer_view().copy_to(dense.data());
    for(int k : bunny_term_order(index,{},values))
    {
        int i=index[k]-src.subvector_offset()-offset;
        if(i<0 || i>=static_cast<int>(pinned.size()) || 3*i+2>=static_cast<int>(dense.size()))
            throw std::runtime_error("FEM term outside its destination");
        if(!pinned[i])
            for(int c=0; c<3; ++c) dense[3*i+c]+=values[k][c];
    }
    dst.buffer_view().copy_from(dense.data());
}
#elif defined(BUNNY_FEM_MATRIX)
static void bunny_assemble_blocks(const cuda_tool::DeviceTripletMatrix<Float,3>& from,
                                  cuda_tool::DeviceBCOOMatrix<Float,3>& to)
{
    std::vector<int> rows(from.triplet_count()),cols(rows.size());
    std::vector<Matrix3x3> values(rows.size());
    if(!rows.empty())
    {
        from.cview().row_indices().copy_to(rows.data());
        from.cview().col_indices().copy_to(cols.data());
        from.cview().values().copy_to(values.data());
    }
    std::vector<int> unique_rows,unique_cols;
    std::vector<Matrix3x3> sums;
    for(int k : bunny_term_order(rows,cols,values))
    {
        if(sums.empty() || unique_rows.back()!=rows[k] || unique_cols.back()!=cols[k])
        {
            unique_rows.push_back(rows[k]);
            unique_cols.push_back(cols[k]);
            sums.push_back(values[k]);
        }
        else sums.back()+=values[k];
    }
    to.reshape(from.rows(),from.cols());
    to.resize_triplets_discard(sums.size());
    if(!sums.empty())
    {
        to.view().row_indices().copy_from(unique_rows.data());
        to.view().col_indices().copy_from(unique_cols.data());
        to.view().values().copy_from(sums.data());
    }
}
#endif
