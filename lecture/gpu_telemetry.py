"""Read device-wide NVML telemetry without spawning nvidia-smi per sample.

API reference: https://docs.nvidia.com/deploy/nvml-api/group__nvmlDeviceQueries.html
These counters include concurrent workloads and do not identify our own share.
NVML v1 used memory includes reserved framebuffer memory. It can therefore
exceed nvidia-smi's allocated-only used field; free and total are comparable.
"""
import ctypes as C
from pathlib import Path
import time


class Memory(C.Structure):
    _fields_=[('total',C.c_ulonglong),('free',C.c_ulonglong),('used',C.c_ulonglong)]


class Utilization(C.Structure):
    _fields_=[('gpu',C.c_uint),('memory',C.c_uint)]


class Telemetry:
    def __init__(self,index=0):
        self.library=C.WinDLL(str(Path('C:/Windows/System32/nvml.dll')))
        self.handle=C.c_void_p()
        signatures={
            'nvmlInit_v2':[], 'nvmlShutdown':[],
            'nvmlDeviceGetHandleByIndex_v2':[C.c_uint,C.POINTER(C.c_void_p)],
            'nvmlSystemGetDriverVersion':[C.c_char_p,C.c_uint],
            'nvmlDeviceGetName':[C.c_void_p,C.c_char_p,C.c_uint],
            'nvmlDeviceGetMemoryInfo':[C.c_void_p,C.POINTER(Memory)],
            'nvmlDeviceGetUtilizationRates':[C.c_void_p,C.POINTER(Utilization)],
            'nvmlDeviceGetEncoderUtilization':[C.c_void_p,C.POINTER(C.c_uint),C.POINTER(C.c_uint)],
            'nvmlDeviceGetTemperature':[C.c_void_p,C.c_uint,C.POINTER(C.c_uint)],
        }
        for name,arguments in signatures.items():
            function=getattr(self.library,name);function.argtypes=arguments;function.restype=C.c_int
        self.call('nvmlInit_v2')
        self.open=True
        try:
            self.call('nvmlDeviceGetHandleByIndex_v2',index,C.byref(self.handle))
            name=C.create_string_buffer(128);driver=C.create_string_buffer(80)
            self.call('nvmlDeviceGetName',self.handle,name,len(name))
            self.call('nvmlSystemGetDriverVersion',driver,len(driver))
            self.identity={'index':index,'name':name.value.decode(),'driver':driver.value.decode()}
        except Exception:
            self.close();raise

    def call(self,name,*arguments):
        code=getattr(self.library,name)(*arguments)
        if code:raise RuntimeError(f'{name} returned NVML error {code}')

    def snapshot(self):
        start=time.time();memory=Memory();util=Utilization();encoder=C.c_uint();period=C.c_uint();temperature=C.c_uint()
        self.call('nvmlDeviceGetMemoryInfo',self.handle,C.byref(memory))
        self.call('nvmlDeviceGetUtilizationRates',self.handle,C.byref(util))
        self.call('nvmlDeviceGetEncoderUtilization',self.handle,C.byref(encoder),C.byref(period))
        self.call('nvmlDeviceGetTemperature',self.handle,0,C.byref(temperature))
        if memory.used+memory.free!=memory.total:raise RuntimeError('NVML memory identity failed')
        if max(util.gpu,util.memory,encoder.value)>100:raise RuntimeError('Invalid utilization percentage')
        return {'at':start,'duration_seconds':time.time()-start,'device':self.identity,
            'memory_total_bytes':memory.total,'memory_used_and_reserved_bytes':memory.used,'memory_free_bytes':memory.free,
            'gpu_percent':util.gpu,'memory_percent':util.memory,'encoder_percent':encoder.value,
            'encoder_sampling_period_us':period.value,'temperature_c':temperature.value,
            'scope':'device-wide; includes concurrent workloads; WDDM framebuffer accounting'}

    def close(self):
        if self.open:
            self.call('nvmlShutdown');self.open=False
