"""Prepare an isolated fail-and-dump solver without changing successful PCG arithmetic."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
src=ROOT/'external/libuipc-psq-src/src/backends/cuda/linear_system'
out=ROOT/'native/libuipc/l3_solver/capture'
out.mkdir(parents=True,exist_ok=True)
def read(n):return (src/n).read_text(encoding='utf-8')
def write(n,s):(out/n).write_text(s,encoding='utf-8')
s=read('linear_fused_pcg.h').replace('cuda_tool::DeviceVar<Float>  d_pAp;', 'cuda_tool::DeviceVar<Float>  d_pAp;\n    cuda_tool::DeviceVar<Float> d_failed_pAp, d_failed_rz;')
write('linear_fused_pcg.h',s)
s=read('linear_fused_pcg.cu').replace('#include <linear_system/linear_fused_pcg.h>','#include "linear_fused_pcg.h"\n#include <backends/common/backend_path_tool.h>\n#include <utils/matrix_market.h>\n#include <fstream>')
s=s.replace('cuda_tool::CDense<IndexT> d_converged,\n                                           cuda_tool::DenseVectorView<Float>  x,','cuda_tool::Dense<IndexT> d_converged,\n                                           cuda_tool::Dense<Float> failed_pAp,\n                                           cuda_tool::Dense<Float> failed_rz,\n                                           cuda_tool::DenseVectorView<Float>  x,')
s=s.replace('if(i == 0) UIPC_KERNEL_ASSERT(false, "FusedPCG breakdown: pAp=%.17g rz=%.17g", *d_pAp, *d_rz);','if(i == 0) { *failed_pAp = *d_pAp; *failed_rz = *d_rz; *d_converged = -1; }')
s=s.replace('        Float rz_new = *d_rz_new;\n        UIPC_KERNEL_ASSERT(::isfinite(rz_new) && rz_new >= 0,\n                           "FusedPCG non-positive preconditioner: rz=%.17g", rz_new);','        if(*d_converged < 0) return;\n        Float rz_new = *d_rz_new;\n        if(!::isfinite(rz_new) || rz_new < 0) { *d_converged = -2; return; }')
s=s.replace('        Float rz_new = *d_rz_new;\n        UIPC_KERNEL_ASSERT(::isfinite(rz_new) && rz_new >= 0, "FusedPCG Iter invalid r^T*z = %.17g", rz_new);','        if(*d_converged < 0) { cudaGraphSetConditional(handle, 0u); return; }\n        Float rz_new = *d_rz_new;\n        if(!::isfinite(rz_new) || rz_new < 0) { *d_converged=-2; cudaGraphSetConditional(handle, 0u); return; }')
s=s.replace('cuda_tool::CVarView<IndexT>        d_converged,\n                     cuda_tool::DenseVectorView<Float>  x,','cuda_tool::VarView<IndexT>         d_converged,\n                     cuda_tool::VarView<Float> failed_pAp,\n                     cuda_tool::VarView<Float> failed_rz,\n                     cuda_tool::DenseVectorView<Float>  x,')
s=s.replace('            d_converged.cviewer(),\n            x.viewer(),','            d_converged.viewer(),\n            failed_pAp.viewer(), failed_rz.viewer(),\n            x.viewer(),')
s=s.replace('d_rz.view(), d_pAp.view(), d_converged.view(), x, p.cview(), r.view(), Ap.cview(), stream);','d_rz.view(), d_pAp.view(), d_converged.view(), d_failed_pAp.view(), d_failed_rz.view(), x, p.cview(), r.view(), Ap.cview(), stream);')
needle='    Timer pcg_timer{"FusedPCG"};'
insert=r'''    Timer pcg_timer{"FusedPCG"};
    auto fail = [&](SizeT block_end, IndexT reason) -> void {
        auto folder = BackendPathTool(workspace()).workspace("linear_system/linear_fused_pcg.cu", "debug");
        auto prefix = fmt::format("{}failure.{}.{}.", folder.string(), engine().frame(), engine().newton_iter());
        export_vector_market(prefix+"x.mtx", x.as_const());
        export_vector_market(prefix+"b.mtx", b);
        export_vector_market(prefix+"p.mtx", p.cview());
        export_vector_market(prefix+"Ap.mtx", Ap.cview());
        export_vector_market(prefix+"r.mtx", r.cview());
        export_vector_market(prefix+"z.mtx", z.cview());
        Float fp=d_failed_pAp, fr=d_failed_rz, rn=d_rz_new;
        std::ofstream(prefix+"meta.txt") << fmt::format("frame={} newton={} block_end={} reason={} failed_pAp={:.17g} failed_rz={:.17g} rz_new={:.17g}\n", engine().frame(),engine().newton_iter(),block_end,reason,fp,fr,rn);
        throw SimSystemException(fmt::format("PCG failed; preserved vectors at {} (reason {}, pAp={}, rz={})", prefix,reason,fp,fr));
    };
    d_failed_pAp=0; d_failed_rz=0;'''
assert needle in s;s=s.replace(needle,insert)
s=s.replace('            IndexT iters = d_iter;','            IndexT failure = d_converged;\n            if(failure < 0) fail((IndexT)d_iter, failure);\n            IndexT iters = d_iter;')
s=s.replace('        Float rz_new_host = d_rz_new;','        IndexT failure = d_converged;\n        if(failure < 0) fail(iter_done, failure);\n        Float rz_new_host = d_rz_new;')
s=s.replace('    return converged ? iter_done : max_iter;','    if(!converged) fail(iter_done, -3);\n    return iter_done;')
write('linear_fused_pcg.cu',s)
s=read('global_linear_system.cu');needle='    m_impl.solve_linear_system();';assert s.count(needle)==1
s=s.replace(needle,'    try { m_impl.solve_linear_system(); }\n    catch(...) { _dump_A_b(); throw; }')
write('global_linear_system.cu',s)
print('Prepared isolated capture source.')
