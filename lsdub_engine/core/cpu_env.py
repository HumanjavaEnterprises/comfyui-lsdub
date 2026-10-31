# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""cpu_env.py: keep MediaPipe/GL work off the GPUs. `cpu_env()` is the environment for a child process (no CUDA, a thread cap,
MediaPipe's GL context on Mesa llvmpipe, not the NVIDIA driver); `guard()` applies the GL variables in-process, before MediaPipe is
imported, only where the caller has not set them. guard() never touches CUDA_VISIBLE_DEVICES or OMP_NUM_THREADS: in a host process
that shares torch, those belong to the host."""

import os, sys

GL_VARS = {
    "__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json",
    "MESA_LOADER_DRIVER_OVERRIDE": "llvmpipe",
    "LIBGL_ALWAYS_SOFTWARE": "1",
}
CHILD_VARS = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}


def cpu_env():
    """child processes stay off the GPUs: no CUDA, and MediaPipe's GL context on Mesa (llvmpipe), not the NVIDIA driver."""
    return dict(os.environ, **GL_VARS, **CHILD_VARS)


def guard():
    """the GL variables in this process, each only if unset."""
    for k, v in GL_VARS.items():
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
