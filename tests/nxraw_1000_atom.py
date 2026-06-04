#!/usr/bin/env python3
import argparse
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import infer
from nxraw_io import StepProfiler, load_nxraw_sample

DEFAULT_SAMPLE_ROOT = Path("/home/goodchong/geths/test_geths/1000-atom/OUT.ABACUS")
DEFAULT_BINARY = REPO_ROOT / "pre_post_process/cpp/build/nextham_preprocess"


def run_command(command):
    print("$", " ".join(shlex.quote(str(part)) for part in command), flush=True)
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise AssertionError(f"command failed with exit code {result.returncode}")


def run_preprocess(binary, sample_root, output_path, output_format):
    run_command([
        binary,
        sample_root / "STRU.cif",
        sample_root,
        "4",
        "8.0",
        output_path,
        "--format",
        "cif",
        "--output-format",
        output_format,
    ])


def assert_raw_matches(torch_pth, nxraw_dir):
    profiler = StepProfiler(True, prefix="nxraw-test")
    with profiler.step("load torch raw pth"):
        input_data, _ = torch.load(torch_pth, weights_only=False, map_location="cpu")
        H0, _, mask, edge_vec, edge_src, edge_dst, _, output_path = input_data
    with profiler.step("load nxraw raw mmap"):
        nxraw_sample = load_nxraw_sample(nxraw_dir, mmap=True, profiler=profiler)

    with profiler.step("compare raw tensors"):
        checks = [
            ("descriptor", torch.equal(H0, nxraw_sample["descriptor"])),
            ("mask", torch.equal(mask, nxraw_sample["mask"])),
            ("edge_vec", torch.equal(edge_vec, nxraw_sample["edge_vec"])),
            ("edge_src", torch.equal(edge_src, nxraw_sample["edge_src"])),
            ("edge_dst", torch.equal(edge_dst, nxraw_sample["edge_dst"])),
        ]
        failed = [name for name, ok in checks if not ok]
        if failed:
            raise AssertionError(f"raw tensor mismatch: {failed}")

    args = argparse.ArgumentParser(parents=[infer.get_args_parser()]).parse_args([])
    _, construct_kernel = infer.get_hamiltonian_size(args, spinful=True)
    raw_from_torch = {
        "sample_dir": str(torch_pth),
        "output_path": output_path,
        "descriptor": H0,
        "mask": mask,
        "edge_vec": edge_vec,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
        "atom_elements": nxraw_sample["atom_elements"],
    }

    with profiler.step("combine torch raw in memory"):
        torch_combined = infer.combine_nxraw_sample(raw_from_torch, construct_kernel, profiler)
    with profiler.step("combine nxraw raw in memory"):
        nxraw_combined = infer.combine_nxraw_sample(nxraw_sample, construct_kernel, profiler)
    with profiler.step("compare combined tensors"):
        for name in ("H0_ds", "H0_raw", "mask_tensor_raw", "edge_vec", "edge_src", "edge_dst", "node_atom"):
            if not torch.equal(torch_combined[name], nxraw_combined[name]):
                raise AssertionError(f"{name} mismatch")
    profiler.summary()


def main():
    parser = argparse.ArgumentParser(description="1000-atom nxraw correctness and timing test")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--skip-preprocess", action="store_true")
    args = parser.parse_args()

    if not args.sample_root.exists():
        raise AssertionError(f"sample root does not exist: {args.sample_root}")
    if not args.binary.exists():
        raise AssertionError(f"preprocess binary does not exist: {args.binary}")

    temp_dir = None
    if args.work_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        work_dir = Path(temp_dir.name)
    else:
        work_dir = args.work_dir
        work_dir.mkdir(parents=True, exist_ok=True)

    try:
        torch_pth = work_dir / "raw_torch.pth"
        nxraw_dir = work_dir / "raw.nxraw"
        if not args.skip_preprocess:
            run_preprocess(args.binary, args.sample_root, torch_pth, "torch")
            run_preprocess(args.binary, args.sample_root, nxraw_dir, "nxraw")
        assert_raw_matches(torch_pth, nxraw_dir)
        print("nxraw 1000-atom checks passed", flush=True)
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
