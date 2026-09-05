"""Fresh, read-only Windows measurements for this project's bounded jobs.

This does not change the shared governor or authorize another workload.
The shared state is always read; an explicit distress/throttle state vetoes
admission. A stale non-distress snapshot is supplemented by live measurements,
never treated as proof of headroom. Owned child affinity is verified separately.

References:
https://learn.microsoft.com/en-us/windows/win32/api/psapi/nf-psapi-getperformanceinfo
https://learn.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-performance_information
https://learn.microsoft.com/en-us/windows/win32/api/winuser/nf-winuser-ishungappwindow
"""
import ctypes as C
from ctypes import wintypes as W
from datetime import datetime,timezone
import json
import math
from pathlib import Path
import time
import psutil

STATE=Path('C:/Users/owner/.claude/compute/compute_state.json')
MiB=1024**2

class Performance(C.Structure):
    _fields_=[('cb',W.DWORD)]+[(name,C.c_size_t) for name in (
        'CommitTotal','CommitLimit','CommitPeak','PhysicalTotal','PhysicalAvailable',
        'SystemCache','KernelTotal','KernelPaged','KernelNonpaged','PageSize')]+[
        ('HandleCount',W.DWORD),('ProcessCount',W.DWORD),('ThreadCount',W.DWORD)]

def live_memory():
    library=C.WinDLL('psapi.dll',use_last_error=True)
    function=library.GetPerformanceInfo
    function.argtypes=[C.POINTER(Performance),W.DWORD];function.restype=W.BOOL
    data=Performance();data.cb=C.sizeof(data)
    if not function(C.byref(data),data.cb):raise C.WinError(C.get_last_error())
    assert 512<=data.PageSize<=1024**2 and data.PageSize&(data.PageSize-1)==0
    assert data.PhysicalAvailable<=data.PhysicalTotal and data.CommitTotal<=data.CommitLimit
    independent=psutil.virtual_memory()
    assert abs(independent.total-data.PhysicalTotal*data.PageSize)<MiB
    # These calls are sequential. A large movement is a reason to resample,
    # not to certify a machine whose memory is changing rapidly.
    assert abs(independent.available-data.PhysicalAvailable*data.PageSize)<2048*MiB
    return dict(mem_avail_mb=data.PhysicalAvailable*data.PageSize/MiB,
        commit_used_mb=data.CommitTotal*data.PageSize/MiB,
        commit_limit_mb=data.CommitLimit*data.PageSize/MiB,
        commit_avail_mb=(data.CommitLimit-data.CommitTotal)*data.PageSize/MiB,
        physical_total_mb=data.PhysicalTotal*data.PageSize/MiB,
        page_size=data.PageSize)

def hung_visible_windows():
    user=C.WinDLL('user32.dll',use_last_error=True)
    callback_type=C.WINFUNCTYPE(W.BOOL,W.HWND,W.LPARAM)
    user.EnumWindows.argtypes=[callback_type,W.LPARAM];user.EnumWindows.restype=W.BOOL
    for name in ('IsWindowVisible','IsHungAppWindow'):
        function=getattr(user,name);function.argtypes=[W.HWND];function.restype=W.BOOL
    counts={'visible':0,'hung':0}
    def visit(hwnd,_):
        if user.IsWindowVisible(hwnd):
            counts['visible']+=1
            counts['hung']+=int(bool(user.IsHungAppWindow(hwnd)))
        return True
    if not user.EnumWindows(callback_type(visit),0):raise C.WinError(C.get_last_error())
    return counts

def rejection_reasons(snapshot,governor):
    reasons=[]
    if governor.get('status') not in {'ok','warn','distress'} or not governor.get('mode'):
        reasons.append('shared governor state is incomplete')
    for name in ('cpu_pct','mem_avail_mb','commit_avail_mb','disk_free_gb','hung_windows'):
        if not math.isfinite(snapshot[name]) or snapshot[name]<0:
            reasons.append('invalid live measurement: '+name)
    if governor.get('status')=='distress' or governor.get('mode')=='throttle':
        reasons.append('shared governor pressure veto')
    if snapshot['cpu_pct']>=85:reasons.append('CPU pressure')
    if snapshot['hung_windows']:reasons.append('hung UI')
    if snapshot['mem_avail_mb']<=8192:reasons.append('physical memory headroom')
    if snapshot['commit_avail_mb']<=16384:reasons.append('commit headroom')
    if snapshot['disk_free_gb']<=30:reasons.append('disk headroom')
    return reasons

def health():
    started=time.time()
    governor=json.loads(STATE.read_text(encoding='utf-8-sig'))
    source_time=datetime.strptime(governor['ts'],'%Y-%m-%d %H:%M:%S').timestamp()
    if source_time>started+5:raise ValueError('Governor timestamp is in the future')
    cpu=psutil.cpu_percent(interval=1)
    memory=live_memory();windows=hung_visible_windows()
    snapshot=dict(ts=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),cpu_pct=cpu,
        **memory,hung_windows=windows['hung'],visible_windows=windows['visible'],
        disk_free_gb=psutil.disk_usage('C:/').free/1024**3)
    reasons=rejection_reasons(snapshot,governor)
    pressure=bool(reasons)
    duration=time.time()-started
    if duration>10:reasons.append('live measurement took too long; resample')
    # A delayed sample blocks new admission. It is not itself evidence of
    # actual pressure and must not be used to kill an existing light reader.
    snapshot.update(status='distress' if pressure else ('warn' if reasons else 'ok'),
        mode='throttle' if pressure else ('hold' if reasons else 'bounded_work'))
    return dict(at=datetime.now(timezone.utc).isoformat(),allow=not reasons,hold_reasons=reasons,
        age_seconds=0,measurement_seconds=duration,snapshot=snapshot,
        governor_age_seconds=started-source_time,governor_stale=started-source_time>90,
        governor={k:governor.get(k) for k in ('ts','status','mode','reasons')},
        scope='Live CPU, PSAPI memory/commit, visible-window hung status and disk; no global guard repair claim')

if __name__=='__main__':print(json.dumps(health(),indent=2))
