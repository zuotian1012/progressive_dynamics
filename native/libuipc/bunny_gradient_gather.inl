// Element slots and U-transpose rows have fixed topology. Sum each output
// vertex's contributions in mesh order, independent of GPU thread scheduling.
struct PsqGatherTerm
{
    int destination, source;
    Float weight;
};

__global__ void psq_gather_rows(int n, const int* row, const int* source,
                                const Float* weight, const Vector3* input,
                                Vector3* output, bool add)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= n)
        return;
    Vector3 value = Vector3::Zero();
    if(add)
        value = output[i];
    for(int k = row[i]; k < row[i+1]; ++k)
        value += (weight ? weight[k] : 1.) * input[source[k]];
    output[i] = value;
}

struct PsqGatherRows
{
    int vertices = 0;
    cuda_tool::DeviceBuffer<int> row, source;
    cuda_tool::DeviceBuffer<Float> weight;

    void build(int count, const std::vector<PsqGatherTerm>& terms, bool weighted)
    {
        vertices = count;
        vector<int> offsets(count+1,0), indices(terms.size());
        vector<Float> weights(weighted ? terms.size() : 0);
        for(const auto& term : terms)
        {
            check(term.destination >= 0 && term.destination < count, "Invalid PSQ gather destination");
            ++offsets[term.destination+1];
        }
        for(int i=0; i<count; ++i)
            offsets[i+1] += offsets[i];
        auto next = offsets;
        for(const auto& term : terms)
        {
            const int k = next[term.destination]++;
            indices[k] = term.source;
            if(weighted)
                weights[k] = term.weight;
        }
        upload(row,offsets);
        upload(source,indices);
        if(weighted)
            upload(weight,weights);
    }

    void sum(const Vector3* input, Vector3* output, bool add=false)
    {
        psq_gather_rows<<<(vertices+127)/128,128>>>(vertices,row.data(),source.data(),
            weight.size() ? weight.data() : nullptr,input,output,add);
    }
};

struct OrderedPsqGradients
{
    PsqGatherRows elements, transpose, offsets;
    cuda_tool::DeviceBuffer<Vector3> slots;

    void build(int nf, int nc, const vector<Vector3i>& faces, const vector<Vector4i>& hinges,
               const vector<int>& row, const vector<int>& col, const vector<Float>& val,
               const vector<int>& anchors, const vector<Vector3i>& coarse_faces)
    {
        std::vector<PsqGatherTerm> terms;
        terms.reserve(3*faces.size()+4*hinges.size());
        auto element = [&](const auto& vertices) {
            for(int j=0; j<vertices.size(); ++j)
                terms.push_back({vertices[j],static_cast<int>(terms.size()),1.});
        };
        for(const auto& face : faces) element(face);
        for(const auto& hinge : hinges) element(hinge);
        elements.build(nf,terms,false);
        slots.resize(std::max(terms.size(),static_cast<size_t>(3)*nf));
        terms.clear();
        for(int i=0; i<nf; ++i)
            for(int k=row[i]; k<row[i+1]; ++k)
                terms.push_back({col[k],i,val[k]});
        transpose.build(nc,terms,true);
        terms.clear();
        for(int i=0; i<nf; ++i)
            for(int j=0; j<3; ++j)
                terms.push_back({coarse_faces[anchors[i]][j],3*i+j,1.});
        offsets.build(nc,terms,false);
    }
};
