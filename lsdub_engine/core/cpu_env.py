# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""cpu_env.py: keep MediaPipe/GL work off the GPUs. `cpu_env()` is the environment for a child process (no CUDA,
MediaPipe's GL context on Mesa llvmpipe, not the NVIDIA driver); `guard()` applies the same variables in-process, before
MediaPipe is imported, but only where the caller has not set them (an operator-granted run keeps its own values)."""

import os, sys

CPU_VARS = {
    "CUDA_VISIBLE_DEVICES": "",
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    "LIBGL_ALWAYS_SOFTWARE": "1",
    "OMP_NUM_THREADS": "4",
}


def cpu_env():
    """child processes stay off the GPUs: no CUDA, and MediaPipe's GL context on Mesa (llvmpipe), not the NVIDIA driver."""
    return dict(os.environ, **CPU_VARS)


def guard():
    """the same variables in this process, each only if unset."""
    for k, v in CPU_VARS.items():
        os.environ.setdefault(k, v)


def thread_report():
    """one observation line: the CPU count and the thread pools this process has. Sets nothing, loads nothing (torch is
    read only if something already imported it, so asking never changes which OpenMP runtime the process ends up with)."""
    cv2 = sys.modules.get("cv2")
    torch = sys.modules.get("torch")
    return (
        f"lsdub threads: cpu_count={os.cpu_count()} cv2={getattr(cv2, '__version__', None)} "
        f"cv2_threads={cv2.getNumThreads() if cv2 else None} cv2_file={getattr(cv2, '__file__', None)} "
        f"torch_threads={torch.get_num_threads() if torch else None} OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}"
    )
