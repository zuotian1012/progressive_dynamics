"""Read Windows hybrid core topology and the flag workers' hot-thread affinity."""
import ctypes as C
from ctypes import wintypes as W
import json
from pathlib import Path
import struct

k=C.WinDLL('kernel32',use_last_error=True)
k.GetLogicalProcessorInformationEx.argtypes=[W.DWORD,C.c_void_p,C.POINTER(W.DWORD)]
k.GetLogicalProcessorInformationEx.restype=W.BOOL
n=W.DWORD();k.GetLogicalProcessorInformationEx(0,None,C.byref(n));buf=C.create_string_buffer(n.value)
if not k.GetLogicalProcessorInformationEx(0,buf,C.byref(n)):raise C.WinError(C.get_last_error())
cores=[];offset=0
while offset<n.value:
    relationship,size=struct.unpack_from('<II',buf.raw,offset)
    flags,efficiency=struct.unpack_from('<BB',buf.raw,offset+8)
    groups=struct.unpack_from('<H',buf.raw,offset+30)[0]
    assert groups==1
    mask,group=struct.unpack_from('<QH',buf.raw,offset+32);assert group==0
    cores.append(dict(logical=[i for i in range(64) if mask&(1<<i)],efficiency=efficiency,smt=bool(flags&1)))
    offset+=size
print('CORES',json.dumps(cores))
k.OpenThread.argtypes=[W.DWORD,W.BOOL,W.DWORD];k.OpenThread.restype=W.HANDLE
k.CloseHandle.argtypes=[W.HANDLE]
class Processor(C.Structure):_fields_=[('group',W.WORD),('number',W.BYTE),('reserved',W.BYTE)]
k.GetThreadIdealProcessorEx.argtypes=[W.HANDLE,C.POINTER(Processor)];k.GetThreadIdealProcessorEx.restype=W.BOOL
root=Path(__file__).resolve().parents[1]
for row in json.loads((root/'output/flag_cpu_inspect.json').read_text(encoding='utf-8-sig')):
    handle=k.OpenThread(0x40,False,row['hot_thread']);cpu=Processor()
    if not handle:raise C.WinError(C.get_last_error())
    try:
        if not k.GetThreadIdealProcessorEx(handle,C.byref(cpu)):raise C.WinError(C.get_last_error())
    finally:k.CloseHandle(handle)
    row['ideal_processor']=cpu.number
    print(json.dumps(row))
