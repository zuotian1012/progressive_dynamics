"""Compile one isolated CUDA translation unit and relink read-only cached objects.

Never invokes the shared xmake build/install, replaces default DLLs, or edits
the shirt source tree. The cached compiler/link commands are exported by xmake.
"""
from pathlib import Path
import argparse,hashlib,json,os,shutil,subprocess

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/flag_L2_fix_20261118'
SRC=ROOT/'external/libuipc-psq-src'

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['exact','rescue'],default='exact');a=parser.parse_args()
    OUT.mkdir(exist_ok=True)
    # Query xmake's recorded commands and MSVC environment without building or
    # installing any shared target. Only these JSON files are written.
    lua='''import("core.base.json");import("core.tool.toolchain");
local base="build/.deps/cuda/windows/x64/release/";
local c=io.load(base.."src/backends/cuda/finite_element/constitutions/psq_proxy.cu.obj.d");
local l=io.load(base.."uipc_backend_cuda.dll.d");
local a=io.load(base.."src/backends/cuda/engine/advance_ipc.cu.obj.d");
local d=io.load(base.."rules/cuda/devlink/uipc_backend_cuda_gpucode.cu.obj.d");
local t=toolchain.load("msvc",{plat="windows",arch="x64"});
local out="../../output/flag_L2_fix_20261118/";
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
    source=ROOT/'native/flag_l2_repair/psq_proxy.cu'
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
    run(cache['compile'][0],[*cache['compile'][1],'-c',source,'-o',obj],'compile')
    advance=build/'advance_ipc.cu.obj'
    if a.variant=='rescue':
        c=json.loads((OUT/'advance_compile.json').read_text())
        run(c[0],[*c[1],'-c',ROOT/'native/flag_l2_repair/advance_ipc.cu','-o',advance],'compile_advance')
    objects={}
    def replace(path):
        p=SRC/path
        if str(path).endswith('psq_proxy.cu.obj'):return obj
        if a.variant=='rescue' and str(path).endswith('advance_ipc.cu.obj'):return advance
        if 'devlink' in str(path):return dev
        objects[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return p
    d=data['devlink'];run(d['values'][0],[*d['values'][1],'-Xcompiler=-MD',*[replace(p) for p in d['files']],'-o',dev],'device_link')
    run(cache['link'][0],[*cache['link'][1],*[replace(p) for p in cache['linkfiles']],'/DLL','/NODEFAULTLIB:LIBCMT','ucrt.lib','vcruntime.lib','msvcrt.lib','/OUT:'+str(library),'/IMPLIB:'+str(build/'flag.lib')],'link')
    manifest=dict(source_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'native/flag_l2_repair').glob('*') if p.is_file()},
        read_only_object_hashes=objects,dll_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        runtime=str(runtime),default_runtime_written=False,shared_sources_written=False)
    (build/'provenance.json').write_text(json.dumps(manifest,indent=2))
    print('BUILT',manifest['dll_sha256'])

if __name__=='__main__':main()

