"""Prepare the fixed-order GPU assembly overlay against the pinned PSQ source."""


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError(f'Unexpected PSQ source at {old[:60]}')
    return source.replace(old, new)


def prepare(source, build):
    code = source.read_text()
    marker = '__global__ void proxy_membrane'
    code = replace_once(code, marker, '#include "bunny_gradient_gather.inl"\n\n' + marker)
    code = replace_once(code, 'cuda_tool::DeviceBuffer<Vector3> y,fg,cg;',
                        'cuda_tool::DeviceBuffer<Vector3> y,fg,cg;\n    OrderedPsqGradients ordered;')
    marker = 'm->replace=psq_replaces_bending();m->enabled=true;m->fem=&fem;'
    code = replace_once(code, marker, marker + '\n    if(m->replace) m->ordered.build(m->nf,m->nc,faces,hinges,row,col,val,anchors,coarse_faces);')
    code = replace_once(code, 'for(int a=0;a<3;++a) atomicAdd(&gradient[ids(j)](a),G(3*j+a));',
                        'gradient[3*i+j] = G.segment<3>(3*j);')
    code = replace_once(code, 'for(int a=0;a<3;++a) atomicAdd(&gradient[t(j)](a),G(3*j+a));',
                        'gradient[4*i+j] = G.segment<3>(3*j);')
    code = replace_once(code, 'nullptr,m->fg.data(),m->lambda', 'nullptr,m->ordered.slots.data(),m->lambda')
    code = replace_once(code, 'nullptr,m->fg.data(),m->bending', 'nullptr,m->ordered.slots.data()+3*m->nt,m->bending')
    scatter = next(line for line in code.splitlines() if 'psq::restrict_gradient<<<' in line)
    code = replace_once(code, scatter, '    m->ordered.elements.sum(m->ordered.slots.data(),m->fg.data());\n'
                        '    m->ordered.transpose.sum(m->fg.data(),m->cg.data());')
    marker = 'if(m->exact_offset || exact_once) psq::restrict_offset_gradient'
    code = replace_once(code, marker, 'if(m->exact_offset || exact_once) {\n        psq::restrict_offset_gradient')
    code = replace_once(code, 'reinterpret_cast<double*>(m->cg.data()));',
                        'reinterpret_cast<double*>(m->ordered.slots.data()));\n'
                        '        m->ordered.offsets.sum(m->ordered.slots.data(),m->cg.data(),true);\n    }')
    patched = build / source.name
    patched.write_text(code)
    # Reuse the exact per-anchor differential. Store its three vertex terms
    # separately until they can be summed in stable anchor order.
    offset = source.with_name('psq_exact_offset.cuh')
    header = replace_once(offset.read_text(), 'tangent[2*i+1]==0)return;',
                          'tangent[2*i+1]==0){ for(int k=0;k<9;++k) cg[9*i+k]=0.; return; }')
    header = replace_once(header, 'atomicAdd(cg+3*a+j,-gb-gc);atomicAdd(cg+3*b+j,gb);atomicAdd(cg+3*c+j,gc);',
                          'cg[9*i+j]=-gb-gc; cg[9*i+3+j]=gb; cg[9*i+6+j]=gc;')
    (build / offset.name).write_text(header)
    return patched


def prepare_fem_units(cuda, direct_build, build):
    """Use the same element forces and symmetric triplets as the direct build."""
    fem_source=cuda/'finite_element/fem_linear_subsystem.cu'
    direct_source=direct_build/'global_linear_system.cu'
    energy_source=cuda/'dytopo_effect_system/dytopo_effect_line_search_reporter.cu'
    namespace='namespace uipc::backend::cuda\n{'
    header='\n#include "bunny_ordered_assembly.inl"\n'
    fem=replace_once(fem_source.read_text(),namespace,namespace+'\n#define BUNNY_FEM_GRADIENT'+header)
    begin=fem.index('        auto k = FEMLinearSubsystem_assemble_reporters_kernel;')
    end=fem.index('\n    }',begin)
    fem=fem[:begin]+'''        bunny_assemble_gradient(reporter_gradients.cview(),info.gradients(),
                                fem().is_fixed.cview(),0);'''+fem[end:]
    begin=fem.index('        auto k = FEMLinearSubsystem_assemble_dytopo_effect_k1_kernel;')
    end=fem.index('\n    }',begin)
    fem=fem[:begin]+'''        bunny_assemble_gradient(dytopo_effect_receiver->gradients(),info.gradients(),
                                fem().is_fixed.cview(),finite_element_vertex_reporter->vertex_offset());'''+fem[end:]
    start='REGISTER_SIM_SYSTEM(GlobalLinearSystem);'
    matrix=replace_once(direct_source.read_text(),start,'#define BUNNY_FEM_MATRIX'+header+start)
    matrix=replace_once(matrix,'converter.convert(triplet_A, bcoo_A);','bunny_assemble_blocks(triplet_A,bcoo_A);')
    energy=replace_once(energy_source.read_text(),
                        'DeviceReduce().Sum(\n        m_impl.energies.data(), info.energy().data(), m_impl.energies.size());',
                        '''std::vector<Float> terms(m_impl.energies.size());
    if(!terms.empty()) m_impl.energies.cview().copy_to(terms.data());
    Float sum=0;
    if(std::all_of(terms.begin(),terms.end(),[](Float v){return std::isfinite(v);}))
    {
        std::sort(terms.begin(),terms.end());
        for(Float v : terms) sum+=v;
    }
    else sum=std::numeric_limits<Float>::infinity();
    info.energy().copy_from(&sum);''')
    result=[]
    for original,code in ((fem_source,fem),(direct_source,matrix),(energy_source,energy)):
        target=build/original.name
        target.write_text(code)
        result.append(target)
    return result
