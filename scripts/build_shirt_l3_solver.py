"""Build isolated solver candidates from cached CUDA objects; do not install them."""
import argparse, hashlib, json, os, shutil, subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SRC=ROOT/'external/libuipc-psq-src'
OUT=ROOT/'output/shirt_L3_solver_fix_20261119'

def main():
    p=argparse.ArgumentParser();p.add_argument('variant');a=p.parse_args()
    source=ROOT/'native/libuipc/l3_solver'/a.variant
    units=sorted(source.glob('*.cu'));assert units
    build=OUT/(a.variant+'_build');build.mkdir(parents=True,exist_ok=True)
    rel=build.relative_to(ROOT).as_posix()
    lua='''import("core.base.json");import("core.tool.toolchain");
local base="build/.deps/cuda/windows/x64/release/";
local l=io.load(base.."uipc_backend_cuda.dll.d");
local d=io.load(base.."rules/cuda/devlink/uipc_backend_cuda_gpucode.cu.obj.d");
local t=toolchain.load("msvc",{plat="windows",arch="x64"});
local cs={};
'''
    for u in units:
        lua+=f'cs["{u.name}"]=io.load(base.."src/backends/cuda/linear_system/{u.name}.obj.d").values;\n'
    lua+=f'io.writefile("../../{rel}/cache.json",json.encode({{compile=cs,link=l.values,files=l.files,devlink=d,env=t:runenvs()}}));'
    subprocess.run(['xmake','lua','-c',lua],cwd=SRC,check=True)
    cache=json.loads((build/'cache.json').read_text());env=os.environ.copy();env.update(cache['env'])
    runtime=OUT/(a.variant+'_runtime/src')
    if not runtime.exists():shutil.copytree(ROOT/'output/shirt_L2_mas_fix_20261119/runtime/src',runtime)
    commands=[];hashes={};objects={};dev=build/'gpucode.obj'
    def run(program,args,name):
        args=[str(x) for x in args];commands.append([program,*args]);(build/'commands.json').write_text(json.dumps(commands,indent=2))
        if name.startswith('compile'):cmd=[program,*args]
        else:
            rsp=build/(name+'.rsp')
            if 'nvcc' in program:
                rsp.write_text('\n'.join('"'+x.replace('\\','/').replace('"','\\"')+'"' for x in args));cmd=[program,'-optf',str(rsp)]
            else:rsp.write_text(subprocess.list2cmdline(args));cmd=[program,'@'+str(rsp)]
        subprocess.run(cmd,cwd=SRC,env=env,check=True)
    for u in units:
        c=cache['compile'][u.name];obj=build/(u.name+'.obj');objects[u.name+'.obj']=obj
        run(c[0],[*c[1],'-c',u,'-o',obj],'compile_'+u.stem)
    def replace(s):
        if Path(s).name in objects:return objects[Path(s).name]
        if 'devlink' in s:return dev
        p=SRC/s;hashes[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
    d=cache['devlink'];run(d['values'][0],[*d['values'][1],'-Xcompiler=-MD',*[replace(s) for s in d['files']],'-o',dev],'device_link')
    dll=runtime/'uipc/_native/uipc_backend_cuda.dll';c=cache['link']
    run(c[0],[*c[1],*[replace(s) for s in cache['files']],'/DLL','/NODEFAULTLIB:LIBCMT','ucrt.lib','vcruntime.lib','msvcrt.lib','/OUT:'+str(dll),'/IMPLIB:'+str(build/'solver.lib')],'link')
    manifest=dict(variant=a.variant,runtime=str(runtime),dll_sha256=hashlib.sha256(dll.read_bytes()).hexdigest(),
        sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir() if p.is_file()},
        read_only_object_hashes=hashes,shared_runtime_or_sources_modified=False)
    (build/'provenance.json').write_text(json.dumps(manifest,indent=2));print('BUILT',manifest['dll_sha256'],flush=True)

if __name__=='__main__':main()
