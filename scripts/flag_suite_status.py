"""Compact status, including independently prefetched Direct runs."""
import json
from pathlib import Path
import ctypes
from ctypes import wintypes

kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD]
kernel.OpenProcess.restype=wintypes.HANDLE
kernel.GetExitCodeProcess.argtypes=[wintypes.HANDLE,ctypes.POINTER(wintypes.DWORD)]
kernel.GetExitCodeProcess.restype=wintypes.BOOL
kernel.CloseHandle.argtypes=[wintypes.HANDLE]

suite=Path(__file__).resolve().parents[1]/'output/flag_to_L9_20261122'
index=json.loads((suite/'index.json').read_text())
active=[]
for file in sorted(suite.glob('L*/L*/run.json')):
    meta=json.loads(file.read_text())
    if meta['status']=='completed':continue
    progress=file.parent/'progress.json'
    state=json.loads(progress.read_text()) if progress.exists() else meta
    entry=dict(run=file.parent.name,status=meta['status'],frames=state.get('accepted_frames',0),
        seconds=round(state.get('elapsed_s',0),1),error=meta.get('error'))
    if meta.get('process_id'):
        handle=kernel.OpenProcess(0x1000,False,meta['process_id'])
        if not handle:entry['process_missing']=True
        else:
            try:
                code=wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle,ctypes.byref(code)):raise ctypes.WinError(ctypes.get_last_error())
                if code.value!=259:entry['process_exit_code']=code.value
            finally:kernel.CloseHandle(handle)
    active.append(entry)
issues=[dict(run=Path(e['run']).name,status=e['status'],details=e.get('details'))
    for entries in index['levels'].values() for e in entries.values()
    if e['status'] in ('failed','audit_failed','parent_failed')]
print(json.dumps(dict(completed_levels=[int(k) for k,v in index['levels'].items() if all(e['status']=='completed' for e in v.values())],
    active=active,issues=issues,finished=(suite/'summary.json').exists()),ensure_ascii=False))
