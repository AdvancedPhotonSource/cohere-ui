"""Device discovery + current-utilization summary for the Device(s) hint."""

import ast
import os
import platform
import re
import shutil
import subprocess
import sys


def list_devices(backend: str) -> list:
    """Return [(index_or_None, line)] for the resolved backend.

    index is the device index that the Device(s) field accepts (so the caller
    can validate user input against it); None for non-indexed entries (CPU /
    error / status lines).
    """
    try:
        resolved = _resolve_auto() if backend == 'auto' else backend
        if resolved == 'np':
            return [(None, _summarize_cpu())]
        if resolved == 'cp':
            rows = _summarize_cuda(prefer='cupy')
            return rows or [(None, _no_gpu_line('cupy'))]
        if resolved == 'torch':
            rows = _summarize_cuda(prefer='torch')
            if rows:
                return rows
            if _torch_has_mps():
                return [(0, '[0] ' + _summarize_mps())]
            return [(None, _torch_cpu_fallback_line()), (None, _summarize_cpu())]
        return [(None, f'unknown backend {resolved!r}')]
    except Exception as e:
        return [(None, f'device probe failed: {type(e).__name__}: {e}')]


def format_devices(devices, selected=None) -> str:
    """HTML wrapper; indices in ``selected`` get a bold-green ``[N]`` prefix."""
    rows = []
    for idx, line in devices:
        if selected and idx is not None and idx in selected:
            line = re.sub(
                r'^\[(\d+)\]',
                r'<span style="color:#1e7a1e;font-weight:600;">[\1]</span>',
                line,
            )
        rows.append(f'<div>{line}</div>')
    return f'<small style="color:#666;">{"".join(rows)}</small>'


def parse_device_field(value, devices):
    """Parse Device(s) text against the probed device list.

    Returns the set of valid indices the user has selected. ``"all"`` (with or
    without quotes) expands to every available index. Indices that don't exist
    in ``devices`` are silently dropped, so they never get highlighted as
    valid. Empty / unparseable input returns ``set()``.
    """
    available = {idx for idx, _ in devices if idx is not None}
    text = (value or '').strip()
    if not text:
        return set()
    bare = text.strip('"\'').lower()
    if bare == 'all':
        return available
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        return set()
    if isinstance(parsed, int):
        return {parsed} & available
    if isinstance(parsed, (list, tuple, set)):
        return {int(x) for x in parsed if isinstance(x, int) and x in available}
    return set()


def summarize_devices(backend: str, selected_devices=None) -> str:
    """Convenience: list + format in one call."""
    return format_devices(list_devices(backend), selected=selected_devices)


def _resolve_auto():
    """Mirror of RecTab._resolve_backend('auto'): cupy -> torch -> numpy."""
    if sys.platform != 'darwin' and _have('cupy'):
        return 'cp'
    if _have('torch'):
        return 'torch'
    return 'np'


def _have(mod):
    try:
        __import__(mod)
        return True
    except Exception:
        return False


def _summarize_cuda(prefer='torch'):
    """nvidia-smi first (system-wide truth), then torch, then cupy.

    Returns ``[(index, line)]`` so the caller can validate Device(s) input
    against the actual indices.
    """
    if shutil.which('nvidia-smi'):
        try:
            out = subprocess.run(
                ['nvidia-smi',
                 '--query-gpu=index,name,memory.used,memory.total,utilization.gpu',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                rows = []
                for row in out.stdout.strip().splitlines():
                    parts = [p.strip() for p in row.split(',')]
                    if len(parts) != 5:
                        continue
                    idx_str, name, used_mb, total_mb, util = parts
                    idx = int(idx_str)
                    used_gb = int(used_mb) / 1024
                    total_gb = int(total_mb) / 1024
                    rows.append((
                        idx,
                        f'[{idx}] {name}, {used_gb:.1f}/{total_gb:.1f} GB used, '
                        f'{util}% busy',
                    ))
                if rows:
                    return rows
        except Exception:
            pass
    if prefer != 'cupy' and _have('torch'):
        try:
            import torch
            if torch.cuda.is_available():
                rows = []
                for i in range(torch.cuda.device_count()):
                    name = torch.cuda.get_device_name(i)
                    free, total = torch.cuda.mem_get_info(i)
                    used_gb = (total - free) / (1024 ** 3)
                    total_gb = total / (1024 ** 3)
                    rows.append((i, f'[{i}] {name}, {used_gb:.1f}/{total_gb:.1f} GB used'))
                if rows:
                    return rows
        except Exception:
            pass
    if _have('cupy'):
        try:
            import cupy
            n = cupy.cuda.runtime.getDeviceCount()
            rows = []
            for i in range(n):
                with cupy.cuda.Device(i):
                    free, total = cupy.cuda.runtime.memGetInfo()
                    props = cupy.cuda.runtime.getDeviceProperties(i)
                raw = props.get('name', f'GPU {i}')
                name = raw.decode() if isinstance(raw, bytes) else str(raw)
                used_gb = (total - free) / (1024 ** 3)
                total_gb = total / (1024 ** 3)
                rows.append((i, f'[{i}] {name}, {used_gb:.1f}/{total_gb:.1f} GB used'))
            if rows:
                return rows
        except Exception:
            pass
    return []


def _no_gpu_line(tool):
    return f'no CUDA devices visible (probed via {tool})'


def _torch_has_mps():
    if not _have('torch'):
        return False
    try:
        import torch
        return hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
    except Exception:
        return False


def _torch_cpu_fallback_line():
    return 'torch: no CUDA/MPS detected, falling back to CPU'


def _summarize_mps():
    """Apple Silicon GPU via MPS; memory is unified with system RAM."""
    name = _cpu_name() or 'Apple Silicon'
    used_gb, total_gb = _ram_usage_gb()
    base = f'Apple GPU (MPS) on {name}, {used_gb:.1f}/{total_gb:.1f} GB unified used'
    util = _mps_util_percent()
    if util is not None:
        base += f', {util}% busy'
    return base


def _mps_util_percent():
    """Apple GPU utilization % via ioreg AGXAccelerator; None if unavailable."""
    try:
        out = subprocess.run(
            ['ioreg', '-r', '-d', '1', '-c', 'AGXAccelerator'],
            capture_output=True, text=True, timeout=2,
        )
        if out.returncode != 0:
            return None
        m = re.search(r'"Device Utilization %"=(\d+)', out.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _summarize_cpu():
    """CPU + system RAM + current CPU utilization."""
    name = _cpu_name() or 'CPU'
    cores = os.cpu_count() or 0
    used_gb, total_gb = _ram_usage_gb()
    util = _cpu_util_percent()
    parts = [name, f'{cores} cores', f'{used_gb:.1f}/{total_gb:.1f} GB used']
    if util is not None:
        parts.append(f'{util:.0f}% busy')
    return ', '.join(parts)


def _cpu_util_percent():
    """System-wide CPU utilization % via psutil; None if unavailable."""
    try:
        import psutil
        return psutil.cpu_percent(interval=0.1)
    except Exception:
        return None


def _cpu_name():
    """Best-effort CPU brand string across platforms."""
    if sys.platform == 'darwin':
        try:
            out = subprocess.run(
                ['sysctl', '-n', 'machdep.cpu.brand_string'],
                capture_output=True, text=True, timeout=2,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:
            pass
    if sys.platform.startswith('linux'):
        try:
            with open('/proc/cpuinfo') as f:
                for line in f:
                    if line.startswith('model name'):
                        return line.split(':', 1)[1].strip()
        except Exception:
            pass
    return platform.processor() or platform.machine() or ''


def _ram_usage_gb():
    """(used, total) RAM in GB via psutil. Returns (0, 0) if psutil missing."""
    try:
        import psutil
        vm = psutil.virtual_memory()
        return vm.used / (1024 ** 3), vm.total / (1024 ** 3)
    except Exception:
        return 0.0, 0.0
