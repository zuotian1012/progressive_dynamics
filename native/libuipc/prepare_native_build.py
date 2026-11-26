"""Prepare an isolated, pinned libuipc source build without changing installed wheels."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import urllib.request
import zipfile
import io

ROOT = Path(__file__).resolve().parents[2]
COMMIT = '9c5710ee4b716502e3757e7312360e5ee68cb79f'
SRC = ROOT/'external/libuipc-psq-src'
PYBIND_SHA = '99983aa3b93e940d5d58662692fc7c3fd18e03ec8187157b48fe4432c4af257d'

def main():
    if not SRC.exists():
        with urllib.request.urlopen('https://codeload.github.com/spiriMirror/libuipc/zip/'+COMMIT,timeout=40) as r:
            data=r.read()
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            prefix=z.namelist()[0]
            for name in z.namelist():
                rel=name[len(prefix):]
                if not rel or name.endswith('/'): continue
                target=(SRC/rel).resolve()
                if not target.is_relative_to(SRC.resolve()): raise ValueError('Archive path escape')
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    archive=ROOT/'native/libuipc/downloads/pybind11-v3.0.4.zip'
    if not archive.exists():
        archive.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen('https://codeload.github.com/pybind/pybind11/zip/refs/tags/v3.0.4',timeout=30) as r:
            archive.write_bytes(r.read())
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==PYBIND_SHA
    with zipfile.ZipFile(archive) as z:
        prefix=z.namelist()[0]
        for name in z.namelist():
            rel=name[len(prefix):]
            if rel.startswith('include/') and not name.endswith('/'):
                target=SRC/'vendor/pybind11'/rel
                target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
        for rel in ['LICENSE','CMakeLists.txt']:
            target=SRC/'vendor/pybind11'/rel;target.write_bytes(z.read(prefix+rel))
        for name in z.namelist():
            rel=name[len(prefix):]
            if rel.startswith('tools/') and not name.endswith('/'):
                target=SRC/'vendor/pybind11'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    info=dict(upstream_commit=COMMIT,pybind11_version='3.0.4',pybind11_archive_sha256=PYBIND_SHA,
              source=str(SRC),installed_wheel_modified=False,
              status='source_build_preparation',
              validation_report=str(ROOT/'output/libuipc_psq_native/integration_result.json'))
    env=os.environ.copy()
    env['UIPC_BUILD_PYTHON']=str(Path(sys.base_prefix)/'python.exe')
    support=ROOT/'native/libuipc/build_support'
    for local,relative in [('xmake.lua','xmake.lua'),('pybind_xmake.lua','src/pybind/xmake.lua'),('pybind_CMakeLists.txt','src/pybind/CMakeLists.txt')]:
        shutil.copyfile(support/local,SRC/relative)
    if '--with-psq' in sys.argv:
        subprocess.run([sys.executable,str(ROOT/'native/libuipc/apply_psq_patch.py')],check=True)
    for scheme,url in urllib.request.getproxies().items():
        if scheme in ('http','https'): env[scheme+'_proxy']=url
    if '--configure' in sys.argv:
        args=['xmake','f','-y','--pybind=true','--python_system=true','--python_version=3.11.x',
              '--tests=false','--examples=false','--benchmarks=false','--dev=true',
              '--cuda=C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6',
              '--pkg_searchdirs='+str(archive.parent)]
        subprocess.run(args,cwd=SRC,env=env,check=True)
    if '--build' in sys.argv:
        subprocess.run(['xmake','build','-y','-j','6'],cwd=SRC,env=env,check=True)
        info['status']='build_completed'
    info['psq_adapter_source_present']=(SRC/'src/backends/cuda/finite_element/constitutions/psq_proxy.cu').exists()
    (ROOT/'native/libuipc/build_provenance.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':main()
