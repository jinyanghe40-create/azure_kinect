#!/usr/bin/env python3
"""
CUDA Runtime API 的精简 ctypes 封装 —— 替代 pycuda

只需要 5 个函数就能跑 TensorRT 推理：
    cudaMalloc / cudaFree / cudaMemcpy / cudaMemset / cudaDeviceSynchronize

JetPack 自带 libcudart.so，不需要装任何东西。
"""
import ctypes
import ctypes.util
import os

_cudart = None


def _load():
    global _cudart
    if _cudart is not None:
        return _cudart
    cands = []
    # 1. 标准库路径
    for name in ("libcudart.so", "libcudart.so.10.2", "libcudart.so.10.0"):
        cands.append(name)
    # 2. CUDA 安装目录
    for base in ("/usr/local/cuda", "/usr/local/cuda-10.2", "/usr/local/cuda-10.0"):
        for name in ("lib64/libcudart.so", "lib/libcudart.so"):
            p = os.path.join(base, name)
            if os.path.exists(p):
                cands.insert(0, p)
    # 3. 让系统找
    f = ctypes.util.find_library("cudart")
    if f:
        cands.insert(0, f)

    last = None
    for c in cands:
        try:
            _cudart = ctypes.CDLL(c)
            _setup(_cudart)
            return _cudart
        except OSError as e:
            last = e
    raise OSError(
        f"找不到 libcudart.so。试过: {cands[:5]}\n"
        f"最后一次错误: {last}\n"
        f"检查: ls /usr/local/cuda/lib64/libcudart*"
    )


def _setup(lib):
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]

    lib.cudaFree.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]

    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                               ctypes.c_size_t, ctypes.c_int]

    lib.cudaDeviceSynchronize.restype = ctypes.c_int
    lib.cudaDeviceSynchronize.argtypes = []

    lib.cudaGetErrorString.restype = ctypes.c_char_p
    lib.cudaGetErrorString.argtypes = [ctypes.c_int]


MemcpyHostToDevice = 1
MemcpyDeviceToHost = 2
MemcpyDeviceToDevice = 3


def _check(rc, what):
    if rc != 0:
        lib = _load()
        msg = lib.cudaGetErrorString(rc)
        raise RuntimeError(f"{what} 失败: {msg.decode(errors='replace')} (code {rc})")


def mem_alloc(nbytes):
    """分配 GPU 内存，返回 device pointer（c_void_p）"""
    lib = _load()
    ptr = ctypes.c_void_p()
    _check(lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cudaMalloc")
    return ptr          # ★ 保持 c_void_p 类型，不要转成 int


def mem_free(ptr):
    lib = _load()
    if ptr:
        lib.cudaFree(ptr)


def memcpy_htod(dst_dev, src_host_array):
    """host numpy array -> device"""
    lib = _load()
    # ★ dst_dev 本身就是 c_void_p（来自 cudaMalloc），不要再包一层
    _check(lib.cudaMemcpy(dst_dev,
                          src_host_array.ctypes.data_as(ctypes.c_void_p),
                          ctypes.c_size_t(src_host_array.nbytes),
                          MemcpyHostToDevice), "cudaMemcpy H2D")


def memcpy_dtoh(dst_host_array, src_dev):
    """device -> host numpy array"""
    lib = _load()
    _check(lib.cudaMemcpy(dst_host_array.ctypes.data_as(ctypes.c_void_p),
                          src_dev,
                          ctypes.c_size_t(dst_host_array.nbytes),
                          MemcpyDeviceToHost), "cudaMemcpy D2H")


def device_sync():
    lib = _load()
    _check(lib.cudaDeviceSynchronize(), "cudaDeviceSynchronize")


def selftest():
    """自检：分配 → 传输 → 回读 → 释放"""
    import numpy as np
    lib = _load()
    print(f"[CUDA] 已加载 libcudart: {lib._name}")
    a = np.arange(16, dtype=np.float32)
    dev = mem_alloc(a.nbytes)
    try:
        memcpy_htod(dev, a)
        b = np.zeros_like(a)
        memcpy_dtoh(b, dev)
        device_sync()
        ok = np.array_equal(a, b)
        print(f"[CUDA] 传输自检: {'✅ 通过' if ok else '❌ 数据不符'}")
        return ok
    finally:
        mem_free(dev)


if __name__ == "__main__":
    import sys
    sys.exit(0 if selftest() else 1)
