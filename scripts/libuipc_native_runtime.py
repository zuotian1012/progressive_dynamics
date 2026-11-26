"""Select the isolated build, never silently fall back to the installed wheel."""
from pathlib import Path
import os
import sys
import ctypes
import shutil

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT/'build/runtime/src'
HANDLES = []

def load(package=None):
    global PACKAGE
    if 'uipc' in sys.modules:
        raise RuntimeError('Select isolated libuipc before importing uipc')
    if os.name != 'nt':
        raise RuntimeError('This packaged native runtime currently supports Windows only')
    if package is not None:
        PACKAGE = Path(package).resolve()
        if not (PACKAGE/'uipc/_native/uipc_backend_cuda.dll').is_file():
            raise FileNotFoundError('Explicit runtime must contain uipc/_native/uipc_backend_cuda.dll')
    native = PACKAGE/'uipc/_native'
    HANDLES.append(os.add_dll_directory(str(native)))
    # Third-party DLLs are staged from the exact xmake package directories.
    # Never add the wheel directory: Windows may select its incompatible core.
    cuda_dirs = []
    if os.environ.get('CUDA_PATH'):
        cuda_dirs.append(Path(os.environ['CUDA_PATH'])/'bin')
    nvcc = shutil.which('nvcc')
    if nvcc:
        cuda_dirs.append(Path(nvcc).parent)
    for path in cuda_dirs:
        if path.exists():HANDLES.append(os.add_dll_directory(str(path)))
    sys.path.insert(0,str(PACKAGE))
    import uipc
    assert Path(uipc.__file__).resolve().is_relative_to(PACKAGE.resolve()), uipc.__file__
    assert Path(uipc.pyuipc.__file__).resolve().is_relative_to(native.resolve())
    return uipc

def loaded_libraries():
    kernel = ctypes.WinDLL('kernel32',use_last_error=True)
    kernel.GetModuleHandleW.argtypes=[ctypes.c_wchar_p]
    kernel.GetModuleHandleW.restype=ctypes.c_void_p
    kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint]
    results={}
    for name in ['uipc_core.dll','uipc_geometry.dll','uipc_constitution.dll','uipc_backend_cuda.dll']:
        handle=kernel.GetModuleHandleW(name)
        if not handle: raise RuntimeError(f'Expected native module not loaded: {name}')
        buf=ctypes.create_unicode_buffer(32768)
        if not kernel.GetModuleFileNameW(handle,buf,len(buf)):raise ctypes.WinError(ctypes.get_last_error())
        path=Path(buf.value).resolve()
        if not path.is_relative_to((PACKAGE/'uipc/_native').resolve()):
            raise RuntimeError(f'Wrong native runtime loaded: {path}')
        results[name]=str(path)
    return results
