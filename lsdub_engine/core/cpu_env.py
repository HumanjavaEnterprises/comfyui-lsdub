# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Humanjava Enterprises. LSDub: https://github.com/HumanjavaEnterprises/comfyui-lsdub
"""cpu_env.py: the one observation line the project node logs."""

import os, sys


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
