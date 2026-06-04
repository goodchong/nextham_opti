import json
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


NXRAW_VERSION = 1


_DTYPE_TO_NUMPY = {
    "complex64": np.complex64,
    "float32": np.float32,
    "int64": np.int64,
}


def format_bytes(num_bytes):
    if num_bytes is None:
        return "unknown"
    sign = "-" if num_bytes < 0 else ""
    size = abs(float(num_bytes))
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024.0:
            return f"{sign}{size:.2f}{unit}"
        size /= 1024.0
    return f"{sign}{size:.2f}PiB"


def current_rss_bytes():
    try:
        with open("/proc/self/status", "r") as status_file:
            for line in status_file:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    return None


class StepProfiler:
    def __init__(self, enabled=False, prefix="nxraw-profile"):
        self.enabled = enabled
        self.prefix = prefix
        self.records = []

    def log(self, message):
        if self.enabled:
            print(f"[{self.prefix}] {message}", flush=True)

    @contextmanager
    def step(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter()
        rss_before = current_rss_bytes()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            rss_after = current_rss_bytes()
            rss_delta = None if rss_before is None or rss_after is None else rss_after - rss_before
            self.records.append((name, elapsed))
            self.log(
                f"{name}: {elapsed:.6f}s, "
                f"rss={format_bytes(rss_after)}, delta={format_bytes(rss_delta)}"
            )

    def summary(self):
        if not self.enabled:
            return
        totals = {}
        counts = {}
        for name, elapsed in self.records:
            totals[name] = totals.get(name, 0.0) + elapsed
            counts[name] = counts.get(name, 0) + 1
        self.log("summary by step:")
        for name, total in sorted(totals.items(), key=lambda item: item[1], reverse=True):
            count = counts[name]
            self.log(f"  {name}: total={total:.6f}s, count={count}, avg={total / count:.6f}s")


def load_manifest(sample_dir):
    sample_dir = Path(sample_dir)
    with open(sample_dir / "manifest.json", "r") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("format") != "nextham-nxraw":
        raise ValueError(f"unsupported nxraw format in {sample_dir}")
    if manifest.get("version") != NXRAW_VERSION:
        raise ValueError(f"unsupported nxraw version {manifest.get('version')} in {sample_dir}")
    return manifest


def load_raw_array(sample_dir, manifest, name, mmap=True):
    tensor_info = manifest["tensors"][name]
    dtype = _DTYPE_TO_NUMPY[tensor_info["dtype"]]
    shape = tuple(tensor_info["shape"])
    path = Path(sample_dir) / tensor_info["file"]
    if mmap:
        return np.memmap(path, dtype=dtype, mode="c", shape=shape)
    return np.fromfile(path, dtype=dtype).reshape(shape)


def torch_from_array(array):
    return torch.from_numpy(np.asarray(array))


def load_nxraw_sample(sample_dir, mmap=True, profiler=None):
    sample_dir = Path(sample_dir)
    profiler = profiler or StepProfiler(False)
    with profiler.step("nxraw load manifest"):
        manifest = load_manifest(sample_dir)
    with profiler.step("nxraw map descriptor"):
        descriptor = torch_from_array(load_raw_array(sample_dir, manifest, "descriptor", mmap=mmap))
    with profiler.step("nxraw map mask"):
        mask = torch_from_array(load_raw_array(sample_dir, manifest, "mask", mmap=mmap))
    with profiler.step("nxraw map edge_vec"):
        edge_vec = torch_from_array(load_raw_array(sample_dir, manifest, "edge_vec", mmap=mmap))
    with profiler.step("nxraw map edge_src"):
        edge_src = torch_from_array(load_raw_array(sample_dir, manifest, "edge_src", mmap=mmap))
    with profiler.step("nxraw map edge_dst"):
        edge_dst = torch_from_array(load_raw_array(sample_dir, manifest, "edge_dst", mmap=mmap))
    return {
        "sample_dir": str(sample_dir),
        "manifest": manifest,
        "descriptor": descriptor,
        "mask": mask,
        "edge_vec": edge_vec,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "atom_elements": manifest["atom_elements"],
        "output_path": manifest.get("output_path", "output_inference.pth"),
    }
