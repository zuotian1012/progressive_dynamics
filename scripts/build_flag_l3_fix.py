"""Compile one isolated CUDA translation unit and relink read-only cached objects.

Never invokes the shared xmake build/install, replaces default DLLs, or edits
the shirt source tree. The cached compiler/link commands are exported by xmake.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from native.libuipc.apply_psq_patch import preserve_contact_on_velocity_write
OUT=ROOT/'output/flag_L3_fix_20261118'
SRC=ROOT/'external/libuipc-psq-src'

# Each comparison reuses its parent's object files except for the units
# explicitly replaced below. Legacy experiments keep their original binaries.
BUNNY_PARENT = {
    'bunny_control': 'bunny',
    'bunny_stable': 'bunny',
    'bunny_energy': 'bunny',
    'bunny_rms': 'bunny_energy',
    'bunny_backtrack': 'bunny_energy',
    'bunny_frozen': 'bunny_energy',
    'bunny_paper': 'bunny_frozen',
    'bunny_precision': 'bunny_paper',
    'bunny_direct': 'bunny_precision',
    'bunny_ordered': 'bunny_direct',
    'bunny_ordered_fem': 'bunny_ordered',
    'bunny_ordered_pcg': 'bunny_ordered_fem',
    'bunny_ordered_mkl': 'bunny_ordered_fem',
    'bunny_ordered_cached': 'bunny_ordered_fem',
}
CORRECTED_ACCESSOR = {'bunny', *BUNNY_PARENT} - {'bunny_control'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['exact','rescue','bunny',*BUNNY_PARENT],default='exact');a=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    if a.variant in BUNNY_PARENT:
        parent=BUNNY_PARENT[a.variant]
        if not (OUT/(parent+'_build')/'provenance.json').is_file():
            subprocess.run([sys.executable,__file__,'--variant',parent],check=True)
    # Query xmake's recorded commands and MSVC environment without building or
    # installing any shared target. Only these JSON files are written.
    lua='''import("core.base.json");import("core.tool.toolchain");
local base="build/.deps/cuda/windows/x64/release/";
local c=io.load(base.."src/backends/cuda/finite_element/constitutions/psq_proxy.cu.obj.d");
local l=io.load(base.."uipc_backend_cuda.dll.d");
local a=io.load(base.."src/backends/cuda/engine/advance_ipc.cu.obj.d");
local d=io.load(base.."rules/cuda/devlink/uipc_backend_cuda_gpucode.cu.obj.d");
local t=toolchain.load("msvc",{plat="windows",arch="x64"});
local out="../../output/flag_L3_fix_20261118/";
io.writefile(out.."build_cache.json",json.encode({compile=c.values,link=l.values,linkfiles=l.files}));
io.writefile(out.."build_env.json",json.encode({env=t:runenvs(),devlink=d}));
io.writefile(out.."advance_compile.json",json.encode(a.values));'''
    subprocess.run(['xmake','lua','-c',lua],cwd=SRC,check=True)
    cache=json.loads((OUT/'build_cache.json').read_text())
    data=json.loads((OUT/'build_env.json').read_text())
    env=os.environ.copy();env.update(data['env'])
    build=OUT/(a.variant+'_build');build.mkdir(exist_ok=True)
    runtime=OUT/(a.variant+'_runtime/src')
    if not runtime.exists():shutil.copytree(SRC/'build/python/src',runtime)
    obj=build/'psq_proxy.cu.obj';dev=build/'gpucode.obj'
    library=runtime/'uipc/_native/uipc_backend_cuda.dll'
    source=ROOT/'native/flag_l3_repair/psq_proxy.cu'
    commands=[]
    def run(program,args,name):
        args=[str(a) for a in args];commands.append([program,*args])
        (build/'commands.json').write_text(json.dumps(commands,indent=2))
        # Response files avoid the Windows command length limit.
        rsp=build/(name+'.rsp');rsp.write_text(subprocess.list2cmdline(args))
        command=[program,'@'+str(rsp)]
        if 'nvcc' in program:
            # nvcc option-file escaping differs from MSVC command-line escaping.
            # Its parser accepts one quoted, forward-slash path per argument.
            rsp.write_text('\n'.join('"'+a.replace('\\','/').replace('"','\\"')+'"' for a in args))
            command=[program,'-optf',str(rsp)]
            if name.startswith('compile'):command=[program,*args]
        subprocess.run(command,cwd=SRC,env=env,check=True)
    advance=build/'advance_ipc.cu.obj'
    if a.variant in BUNNY_PARENT:
        reference=OUT/(BUNNY_PARENT[a.variant]+'_build')
        for target in (obj,advance):
            shutil.copyfile(reference/target.name,target)
    else:
        run(cache['compile'][0],[*cache['compile'][1],'-c',source,'-o',obj],'compile')
    if a.variant in ('rescue','bunny'):
        c=json.loads((OUT/'advance_compile.json').read_text())
        run(c[0],[*c[1],'-c',ROOT/'native/flag_l3_repair/advance_ipc.cu','-o',advance],'compile_advance')
    accessor_source=SRC/'src/backends/cuda/finite_element/finite_element_state_accessor_feature.cu'
    accessor_obj=build/'finite_element_state_accessor_feature.cu.obj'
    stable_parts={}
    if a.variant in ('bunny_ordered_pcg','bunny_ordered_mkl','bunny_ordered_cached'):
        # Keep the ordered energy, gradient and symmetric matrix assembly.
        # Switch only the final linear solve back to the native GPU PCG path.
        shutil.copyfile(reference/accessor_obj.name, accessor_obj)
        for name in ('max_translation_checker.cu.obj', 'fem_linear_subsystem.cu.obj',
                     'dytopo_effect_line_search_reporter.cu.obj'):
            stable_parts[name] = build/name
            shutil.copyfile(reference/name, stable_parts[name])
        patched = build/'global_linear_system.cu'
        code = (reference/patched.name).read_text()
        direct_call = '#include "bunny_sparse_ldlt.inl"\n    return;'
        if code.count(direct_call) != 1:
            raise ValueError('Expected one direct solve in the ordered reference')
        if a.variant=='bunny_ordered_mkl':
            direct=ROOT/'native/libuipc/bunny_sparse_ldlt.inl'
            optimized=build/'bunny_sparse_mkl.inl'
            text=direct.read_text().replace('Eigen::SimplicialLDLT<Sparse, Eigen::Upper> factor;','Eigen::PardisoLLT<Sparse, Eigen::Upper> factor;')
            text=text.replace(' || !factor.vectorD().allFinite() || factor.vectorD().minCoeff() <= 0','')
            text=text.replace('factor.vectorD().minCoeff()','1.0').replace('BUNNY_DIRECT_SOLVE','FLAG_MKL_DIRECT_SOLVE')
            text=text.replace(' min_pivot={:.17g}','').replace('b.size(), 1.0, rhs.norm()','b.size(), rhs.norm()')
            optimized.write_text(text)
            code=code.replace('#include <Eigen/Sparse>','#include <Eigen/Sparse>\n#include <Eigen/PardisoSupport>')
            patched.write_text(code.replace(direct_call,'#include "bunny_sparse_mkl.inl"\n    return;'))
        elif a.variant=='bunny_ordered_cached':
            optimized=build/'bunny_sparse_cached.inl'
            text=(ROOT/'native/libuipc/bunny_sparse_ldlt.inl').read_text()
            old='Eigen::SimplicialLDLT<Sparse, Eigen::Upper> factor;\nfactor.compute(upper);'
            new='''static thread_local Eigen::SimplicialLDLT<Sparse, Eigen::Upper> factor;
static thread_local std::vector<int> cached_outer, cached_inner;
const bool same_pattern = cached_outer.size() == upper.outerSize()+1 && cached_inner.size() == upper.nonZeros()
    && std::equal(cached_outer.begin(),cached_outer.end(),upper.outerIndexPtr())
    && std::equal(cached_inner.begin(),cached_inner.end(),upper.innerIndexPtr());
if(!same_pattern) {
    factor.analyzePattern(upper);
    cached_outer.assign(upper.outerIndexPtr(),upper.outerIndexPtr()+upper.outerSize()+1);
    cached_inner.assign(upper.innerIndexPtr(),upper.innerIndexPtr()+upper.nonZeros());
}
factor.factorize(upper);'''
            if old not in text:raise ValueError('Unexpected direct factor source')
            optimized.write_text(text.replace(old,new))
            patched.write_text(code.replace(direct_call,'#include "bunny_sparse_cached.inl"\n    return;'))
        else:patched.write_text(code.replace(direct_call, ''))
        target = build/(patched.name+'.obj')
        stable_parts[target.name] = target
        extra=['-IC:/Program Files (x86)/Intel/oneAPI/mkl/latest/include','-DEIGEN_USE_MKL'] if a.variant=='bunny_ordered_mkl' else []
        run(cache['compile'][0], [*cache['compile'][1],*extra, '-I'+str(ROOT/'native/libuipc'),
                                 '-c', patched, '-o', target], 'compile_ordered_pcg')
    if a.variant=='bunny_ordered_fem':
        from native.libuipc.prepare_bunny_ordered_proxy import prepare_fem_units
        gather=ROOT/'native/libuipc/bunny_ordered_assembly.inl'
        units=prepare_fem_units(SRC/'src/backends/cuda',OUT/'bunny_direct_build',build)
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
        checker_obj=build/'max_translation_checker.cu.obj'
        shutil.copyfile(reference/checker_obj.name,checker_obj)
        stable_parts[checker_obj.name]=checker_obj
        for unit in units:
            target=build/(unit.name+'.obj')
            stable_parts[target.name]=target
            run(cache['compile'][0],[*cache['compile'][1],'-I'+str(gather.parent),'-c',unit,'-o',target],'compile_'+unit.stem)
    if a.variant=='bunny_ordered':
        from native.libuipc.prepare_bunny_ordered_proxy import prepare
        gather=ROOT/'native/libuipc/bunny_gradient_gather.inl'
        patched=prepare(source,build)
        run(cache['compile'][0],[*cache['compile'][1],'-I'+str(source.parent),'-I'+str(gather.parent),'-c',patched,'-o',obj],'compile_ordered')
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
        for unit in ('max_translation_checker.cu.obj','global_linear_system.cu.obj'):
            stable_parts[unit]=build/unit
            shutil.copyfile(reference/unit,stable_parts[unit])
    if a.variant=='bunny_direct':
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
        checker_obj=build/'max_translation_checker.cu.obj'
        shutil.copyfile(reference/checker_obj.name,checker_obj)
        stable_parts[checker_obj.name]=checker_obj
        original=SRC/'src/backends/cuda/linear_system/global_linear_system.cu'
        direct=ROOT/'native/libuipc/bunny_sparse_ldlt.inl'
        old=original.read_text()
        hook='    Timer timer{"Solve Linear System"};'
        if old.count(hook)!=1:raise ValueError('Unrecognized Newton linear-solve entry')
        patched=build/original.name
        patched.write_text(old.replace(hook,hook+'\n#include "bunny_sparse_ldlt.inl"\n    return;'))
        linear_obj=build/(original.name+'.obj')
        stable_parts[linear_obj.name]=linear_obj
        run(cache['compile'][0],[*cache['compile'][1],'-I'+str(direct.parent),'-c',patched,'-o',linear_obj],'compile_direct')
    if a.variant in ('bunny_rms','bunny_paper'):
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
        checker=ROOT/'native/libuipc/bunny_rms_translation_checker.cu'
        checker_obj=build/'max_translation_checker.cu.obj'
        stable_parts[checker_obj.name]=checker_obj
        run(cache['compile'][0],[*cache['compile'][1],'-c',checker,'-o',checker_obj],'compile_rms')
    if a.variant=='bunny_precision':
        original=OUT/'bunny_frozen_build/advance_ipc.cu'
        candidate=ROOT/'native/libuipc/bunny_representable_search.inl'
        hook='#include "bunny_exhausted_search.inl"'
        old=original.read_text()
        if old.count(hook)!=1:
            raise ValueError('Missing verified backtracking hook')
        patched=build/'advance_ipc.cu'
        patched.write_text(old.replace(hook,'#include "bunny_representable_search.inl"\n                    '+hook))
        flags=json.loads((OUT/'advance_compile.json').read_text())
        run(flags[0],[*flags[1],'-I'+str(source.parent),'-I'+str(candidate.parent),'-c',patched,'-o',advance],'compile_advance')
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
        checker_obj=build/'max_translation_checker.cu.obj'
        shutil.copyfile(reference/checker_obj.name,checker_obj)
        stable_parts[checker_obj.name]=checker_obj
    if a.variant in ('bunny_backtrack','bunny_frozen'):
        original=reference/'advance_ipc.cu'
        candidate=ROOT/'native/libuipc/bunny_exhausted_search.inl'
        old=original.read_text()
        final_check='if(!forced_iteration) check_line_search_iter(line_search_iter, E0, last_E);'
        if old.count(final_check)!=1:
            raise ValueError('Unrecognized line-search exhaustion check')
        patched=build/'advance_ipc.cu'
        recovery='#include "bunny_exhausted_search.inl"\n                    '+final_check
        if a.variant=='bunny_frozen':
            derivative=candidate.with_name('bunny_preview_derivative.inl')
            recovery='#include "bunny_preview_derivative.inl"\n                    '+recovery
        patched.write_text(old.replace(final_check,recovery))
        flags=json.loads((OUT/'advance_compile.json').read_text())
        run(flags[0],[*flags[1],'-I'+str(source.parent),'-I'+str(candidate.parent),'-c',patched,'-o',advance],'compile_advance')
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
    if a.variant=='bunny_energy':
        original=ROOT/'native/flag_l3_repair/advance_ipc.cu'
        guard=ROOT/'native/libuipc/psq_converged_trial.inl'
        old=original.read_text()
        unchecked='if(converged && !(newton_iter==flag_rescue_iteration))\n                            break;'
        insertion='// PSQ reject nonfinite trial before convergence acceptance'
        if old.count(unchecked)!=1 or old.count(insertion)!=1:
            raise ValueError('Unrecognized Newton line search in source overlay')
        patched=build/'advance_ipc.cu'
        patched.write_text(old.replace(unchecked,'').replace(insertion,'#include "psq_converged_trial.inl"\n                        '+insertion))
        c=json.loads((OUT/'advance_compile.json').read_text())
        run(c[0],[*c[1],'-I'+str(original.parent),'-I'+str(guard.parent),'-c',patched,'-o',advance],'compile_advance')
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
    if a.variant=='bunny':
        original=accessor_source.read_text()
        corrected=preserve_contact_on_velocity_write(original)
        if corrected!=original:
            accessor_source=build/accessor_source.name
            accessor_source.write_text(corrected)
        run(c[0],[*c[1],'-c',accessor_source,'-o',accessor_obj],'compile_accessor')
    if a.variant=='bunny_stable':
        shutil.copyfile(reference/accessor_obj.name,accessor_obj)
        stable_src=ROOT/'native/libuipc/l3_solver/stabilized'
        for unit in sorted(stable_src.glob('*.cu')):
            target=build/(unit.name+'.obj')
            stable_parts[target.name]=target
            run(cache['compile'][0],[*cache['compile'][1],'-c',unit,'-o',target],'compile_'+unit.stem)
    objects={}
    def replace(path):
        p=SRC/path
        if str(path).endswith('psq_proxy.cu.obj'):return obj
        if a.variant!='exact' and str(path).endswith('advance_ipc.cu.obj'):return advance
        if a.variant in CORRECTED_ACCESSOR and str(path).endswith(accessor_obj.name):return accessor_obj
        if p.name in stable_parts:return stable_parts[p.name]
        if 'devlink' in str(path):return dev
        objects[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    d=data['devlink'];run(d['values'][0],[*d['values'][1],'-Xcompiler=-MD',*[replace(p) for p in d['files']],'-o',dev],'device_link')
    extra_libs=['C:/Program Files (x86)/Intel/oneAPI/mkl/latest/lib/intel64/mkl_rt.lib'] if a.variant=='bunny_ordered_mkl' else []
    run(cache['link'][0],[*cache['link'][1],*[replace(p) for p in cache['linkfiles']],*extra_libs,'/DLL','/NODEFAULTLIB:LIBCMT','ucrt.lib','vcruntime.lib','msvcrt.lib','/OUT:'+str(library),'/IMPLIB:'+str(build/'flag.lib')],'link')
    manifest=dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'native/flag_l3_repair').glob('*') if p.is_file()},
        read_only_object_hashes=objects,dll_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        runtime=str(runtime),default_runtime_written=False,shared_sources_written=False)
    if a.variant in CORRECTED_ACCESSOR:
        manifest['source_hashes'][str(accessor_source)]=hashlib.sha256(accessor_source.read_bytes()).hexdigest()
        manifest['velocity_only_preserves_friction']=True
    if a.variant=='bunny_energy':
        corrected=json.loads((reference/'provenance.json').read_text())
        if objects!=corrected['read_only_object_hashes']:
            raise ValueError('Unmatched object files for line-search control')
        manifest['source_hashes'][str(guard)]=hashlib.sha256(guard.read_bytes()).hexdigest()
        manifest['source_hashes'].update(corrected['source_hashes'])
        manifest['source_hashes'][str(patched)]=hashlib.sha256(patched.read_bytes()).hexdigest()
        manifest['energy_checked_convergence']=True
    if a.variant in ('bunny_backtrack','bunny_frozen'):
        baseline=json.loads((reference/'provenance.json').read_text())
        if objects!=baseline['read_only_object_hashes']:
            raise ValueError('Unmatched object files for the exhaustion check')
        manifest['source_hashes'].update(baseline['source_hashes'])
        for path in (candidate,patched):
            manifest['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        manifest.update(energy_checked_convergence=True,checked_search_exhaustion=True,
                        exhausted_search_acceptance='strict_decrease',
                        reference_dll_sha256=baseline['dll_sha256'])
    if a.variant=='bunny_frozen':
        manifest['source_hashes'][str(derivative)]=hashlib.sha256(derivative.read_bytes()).hexdigest()
        manifest['frozen_direction_derivative']=True
    if a.variant in ('bunny_rms','bunny_paper'):
        baseline=json.loads((reference/'provenance.json').read_text())
        unchanged={path:sha for path,sha in baseline['read_only_object_hashes'].items()
                   if Path(path).name!=checker_obj.name}
        if unchanged!=objects or len(unchanged)+1!=len(baseline['read_only_object_hashes']):
            raise ValueError('Unmatched objects in Newton RMS control')
        manifest['source_hashes'].update(baseline['source_hashes'])
        manifest['source_hashes'][str(checker)]=hashlib.sha256(checker.read_bytes()).hexdigest()
        manifest.update(energy_checked_convergence=True,bunny_newton_norm='free_vertex_rms_3',
                        reference_dll_sha256=baseline['dll_sha256'])
        if a.variant=='bunny_paper':
            for key in ('checked_search_exhaustion','exhausted_search_acceptance','frozen_direction_derivative'):
                manifest[key]=baseline[key]
    if a.variant=='bunny_precision':
        baseline=json.loads((reference/'provenance.json').read_text())
        if objects!=baseline['read_only_object_hashes']:
            raise ValueError('Unmatched object files in the representable-step comparison')
        manifest['source_hashes'].update(baseline['source_hashes'])
        for path in (candidate,patched):
            manifest['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('energy_checked_convergence','checked_search_exhaustion','exhausted_search_acceptance',
                    'frozen_direction_derivative','bunny_newton_norm'):
            manifest[key]=baseline[key]
        manifest.update(representable_preview=True,reference_dll_sha256=baseline['dll_sha256'])
    if a.variant=='bunny_stable':
        corrected=json.loads((reference/'provenance.json').read_text())
        assert all(corrected['read_only_object_hashes'].get(p)==h for p,h in objects.items())
        assert {Path(p).name for p in corrected['read_only_object_hashes'].keys()-objects.keys()}==set(stable_parts)
        manifest['source_hashes'].update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in stable_src.iterdir() if p.is_file()})
        manifest['linear_solver_parent_sha256']=corrected['dll_sha256']
    if a.variant=='bunny_direct':
        baseline=json.loads((reference/'provenance.json').read_text())
        unchanged={p:h for p,h in baseline['read_only_object_hashes'].items() if Path(p).name!=linear_obj.name}
        if unchanged!=objects or len(unchanged)+1!=len(baseline['read_only_object_hashes']):
            raise ValueError('Unmatched object files in direct linear-solve control')
        manifest['source_hashes'].update(baseline['source_hashes'])
        for path in (original,patched,direct):
            manifest['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('energy_checked_convergence','checked_search_exhaustion','exhausted_search_acceptance',
                    'frozen_direction_derivative','bunny_newton_norm','representable_preview'):
            manifest[key]=baseline[key]
        manifest.update(bunny_linear_solver='eigen_sparse_ldlt',reference_dll_sha256=baseline['dll_sha256'])
    if a.variant=='bunny_ordered':
        baseline=json.loads((reference/'provenance.json').read_text())
        if objects!=baseline['read_only_object_hashes']:
            raise ValueError('Unmatched object files in ordered PSQ assembly')
        manifest['source_hashes'].update(baseline['source_hashes'])
        for path in (patched,build/'psq_exact_offset.cuh',gather):
            manifest['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('energy_checked_convergence','checked_search_exhaustion','exhausted_search_acceptance',
                    'frozen_direction_derivative','bunny_newton_norm','representable_preview','bunny_linear_solver'):
            manifest[key]=baseline[key]
        manifest.update(ordered_psq_gradient=True,reference_dll_sha256=baseline['dll_sha256'])
    if a.variant=='bunny_ordered_fem':
        baseline=json.loads((reference/'provenance.json').read_text())
        changed={unit.name+'.obj' for unit in units}
        unchanged={p:h for p,h in baseline['read_only_object_hashes'].items() if Path(p).name not in changed}
        if unchanged!=objects or len(unchanged)+2!=len(baseline['read_only_object_hashes']):
            raise ValueError('Unmatched object files in ordered FEM assembly')
        manifest['source_hashes'].update(baseline['source_hashes'])
        for path in [*units,gather]:
            manifest['source_hashes'][str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        for key in ('energy_checked_convergence','checked_search_exhaustion','exhausted_search_acceptance',
                    'frozen_direction_derivative','bunny_newton_norm','representable_preview',
                    'bunny_linear_solver','ordered_psq_gradient'):
            manifest[key]=baseline[key]
        manifest.update(ordered_fem_assembly=True,reference_dll_sha256=baseline['dll_sha256'])
    if a.variant in ('bunny_ordered_pcg','bunny_ordered_mkl','bunny_ordered_cached'):
        baseline = json.loads((reference/'provenance.json').read_text())
        if objects != baseline['read_only_object_hashes']:
            raise ValueError('Unmatched object files in ordered PCG build')
        manifest['source_hashes'].update(baseline['source_hashes'])
        manifest['source_hashes'][str(patched)] = hashlib.sha256(patched.read_bytes()).hexdigest()
        for key in ('energy_checked_convergence', 'checked_search_exhaustion', 'exhausted_search_acceptance',
                    'frozen_direction_derivative', 'bunny_newton_norm', 'representable_preview',
                    'ordered_psq_gradient', 'ordered_fem_assembly'):
            manifest[key] = baseline[key]
        manifest.update(bunny_linear_solver='mkl_pardiso_llt' if a.variant=='bunny_ordered_mkl' else 'native_pcg', reference_dll_sha256=baseline['dll_sha256'])
        if a.variant=='bunny_ordered_cached':manifest['bunny_linear_solver']='eigen_sparse_ldlt_cached_symbolic'
        if a.variant in ('bunny_ordered_mkl','bunny_ordered_cached'):manifest['source_hashes'][str(optimized)]=hashlib.sha256(optimized.read_bytes()).hexdigest()
    if a.variant=='bunny_control':
        corrected=json.loads((reference/'provenance.json').read_text())
        historical=json.loads((OUT/'rescue_build/provenance.json').read_text())
        control_only=set(objects)-set(corrected['read_only_object_hashes'])
        assert len(control_only)==1 and Path(next(iter(control_only))).name==accessor_obj.name
        old_accessor=next(iter(control_only))
        assert objects[old_accessor]==historical['read_only_object_hashes'][old_accessor]
        assert all(objects.get(p)==h for p,h in corrected['read_only_object_hashes'].items())
        manifest['paired_corrected_dll_sha256']=corrected['dll_sha256']
        manifest['identical_solver_objects']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (obj,advance)}
        manifest['historical_accessor_sha256']=objects[old_accessor]
    (build/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print('BUILT',manifest['dll_sha256'])

if __name__=='__main__':main()

