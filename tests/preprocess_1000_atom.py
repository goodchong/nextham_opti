#!/usr/bin/env python3
import argparse
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_ROOT = Path("/home/goodchong/geths/test_geths/1000-atom/OUT.ABACUS")
DEFAULT_BINARY = REPO_ROOT / "pre_post_process/cpp/build/nextham_preprocess"
EXPECTED_EDGE_COUNT = 381000


def cif_tokens(line):
    tokens = []
    token = []
    quote = None
    for char in line:
        if quote:
            if char == quote:
                tokens.append("".join(token))
                token = []
                quote = None
            else:
                token.append(char)
        elif char in ("'", '"'):
            if token:
                tokens.append("".join(token))
                token = []
            quote = char
        elif char == "#":
            break
        elif char.isspace():
            if token:
                tokens.append("".join(token))
                token = []
        else:
            token.append(char)
    if token:
        tokens.append("".join(token))
    return tokens


def cif_number(value):
    if value in {".", "?"}:
        raise ValueError("missing CIF numeric value")
    return float(value.split("(", 1)[0])


def parse_cif_like_cpp(cif_path):
    lines = cif_path.read_text().splitlines()
    scalars = {}
    for line in lines:
        tokens = cif_tokens(line)
        if len(tokens) < 2:
            continue
        key = tokens[0].lower()
        if key in {
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        }:
            scalars[key] = cif_number(tokens[1])

    a = scalars["_cell_length_a"]
    b = scalars["_cell_length_b"]
    c = scalars["_cell_length_c"]
    alpha = math.radians(scalars["_cell_angle_alpha"])
    beta = math.radians(scalars["_cell_angle_beta"])
    gamma = math.radians(scalars["_cell_angle_gamma"])
    ca, cb, cg = math.cos(alpha), math.cos(beta), math.cos(gamma)
    sg = math.sin(gamma)

    lattice = np.zeros((3, 3), dtype=np.float64)
    lattice[0] = [a, 0.0, 0.0]
    lattice[1] = [b * cg, b * sg, 0.0]
    cx = c * cb
    cy = c * (ca - cb * cg) / sg
    lattice[2] = [cx, cy, math.sqrt(max(0.0, c * c - cx * cx - cy * cy))]

    atoms = []
    species = []
    species_seen = set()
    is_direct = False

    def column(headers, name):
        lowered = [h.lower() for h in headers]
        return lowered.index(name) if name in lowered else -1

    def starts_control(line):
        value = line.strip().lower()
        return value == "loop_" or value.startswith("data_") or value.startswith("_")

    i = 0
    while i < len(lines):
        if lines[i].strip().lower() != "loop_":
            i += 1
            continue
        headers = []
        row_start = i + 1
        while row_start < len(lines):
            tokens = cif_tokens(lines[row_start])
            if not tokens:
                row_start += 1
                continue
            if not tokens[0].startswith("_"):
                break
            headers.append(tokens[0])
            row_start += 1

        symbol_col = column(headers, "_atom_site_type_symbol")
        if symbol_col < 0:
            symbol_col = column(headers, "_atom_site_label")
        fx, fy, fz = (column(headers, name) for name in (
            "_atom_site_fract_x", "_atom_site_fract_y", "_atom_site_fract_z"
        ))
        cx_col, cy_col, cz_col = (column(headers, name) for name in (
            "_atom_site_cartn_x", "_atom_site_cartn_y", "_atom_site_cartn_z"
        ))
        fractional = fx >= 0 and fy >= 0 and fz >= 0
        cartesian = cx_col >= 0 and cy_col >= 0 and cz_col >= 0
        if symbol_col < 0 or not (fractional or cartesian):
            i += 1
            continue

        is_direct = fractional
        values = []
        j = row_start
        while j < len(lines) and not starts_control(lines[j]):
            row_tokens = cif_tokens(lines[j])
            values.extend(row_tokens)
            while len(values) >= len(headers):
                symbol = re.match(r"[A-Za-z]+", values[symbol_col]).group(0).capitalize()
                x_col, y_col, z_col = (fx, fy, fz) if fractional else (cx_col, cy_col, cz_col)
                pos = [cif_number(values[x_col]), cif_number(values[y_col]), cif_number(values[z_col])]
                atoms.append((symbol, pos))
                if symbol not in species_seen:
                    species_seen.add(symbol)
                    species.append(symbol)
                del values[:len(headers)]
            j += 1
        if values:
            raise ValueError("incomplete CIF atom row")
        i = j

    if not atoms:
        raise ValueError(f"no atoms parsed from {cif_path}")

    pos = np.asarray([p for _, p in atoms], dtype=np.float64)
    if is_direct:
        frac = pos
    else:
        frac = pos @ np.linalg.inv(lattice)
    frac = frac - np.floor(frac + 1e-9)
    elements = [element for element, _ in atoms]
    return lattice, frac, elements


def lattice_heights(lattice):
    v0, v1, v2 = lattice

    def height(a, b, c):
        n_vec = np.cross(b, c)
        n_vec /= np.linalg.norm(n_vec)
        return abs(np.dot(a, n_vec))

    return height(v0, v1, v2), height(v1, v0, v2), height(v2, v0, v1)


def build_reference_edges(lattice, frac, cutoff):
    h0, h1, h2 = lattice_heights(lattice)
    nrx, nry, nrz = (math.ceil(cutoff / h) for h in (h0, h1, h2))
    cutoff_sq = cutoff * cutoff
    n = len(frac)

    parts = []
    for i in range(n):
        pi = frac[i]
        for rx in range(-nrx, nrx + 1):
            for ry in range(-nry, nry + 1):
                for rz in range(-nrz, nrz + 1):
                    r_vec = np.array([rx, ry, rz], dtype=np.float64)
                    d_frac = frac + r_vec - pi
                    d_cart = d_frac @ lattice
                    d2 = np.einsum("ij,ij->i", d_cart, d_cart)
                    js = np.nonzero(d2 < cutoff_sq)[0].astype(np.int32)
                    if js.size == 0:
                        continue
                    parts.append(np.column_stack([
                        np.full(js.size, rx, dtype=np.int32),
                        np.full(js.size, ry, dtype=np.int32),
                        np.full(js.size, rz, dtype=np.int32),
                        np.full(js.size, i, dtype=np.int32),
                        js,
                    ]))

    if not parts:
        return np.empty((0, 5), dtype=np.int32)
    edges = np.concatenate(parts, axis=0)
    order = np.lexsort((edges[:, 4], edges[:, 3], edges[:, 2], edges[:, 1], edges[:, 0]))
    return edges[order]


def load_torch(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def run_preprocess(binary, sample_root, output_path, nspin, cutoff, threads):
    stru = sample_root / "STRU.cif"
    command = [
        str(binary),
        str(stru),
        str(sample_root),
        str(nspin),
        str(cutoff),
        str(output_path),
        "--format",
        "cif",
    ]
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    print("$", " ".join(shlex.quote(part) for part in command))
    result = subprocess.run(command, env=env, text=True, capture_output=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        raise AssertionError(f"preprocess failed with exit code {result.returncode}")
    match = re.search(r"Neighbor list build took:\s+([0-9.]+)\s+s\s+\((\d+)\s+edges found\)", result.stdout)
    if not match:
        raise AssertionError("could not parse neighbor-list timing and edge count")
    return float(match.group(1)), int(match.group(2)), result.stdout


def tensor_sample(tensor, max_edges=2048):
    if tensor.shape[0] <= max_edges:
        return tensor
    idx = torch.linspace(0, tensor.shape[0] - 1, steps=max_edges).long()
    return tensor.index_select(0, idx)


def assert_preprocess_payload(pth_path, expected_edges):
    sample = load_torch(pth_path)
    if not isinstance(sample, tuple) or len(sample) != 2:
        raise AssertionError("preprocess output should be a (data_list, label) tuple")
    data_list, label = sample
    if label is not None:
        raise AssertionError("inference preprocess label should be None")
    if len(data_list) != 8:
        raise AssertionError(f"expected 8 input fields, got {len(data_list)}")

    h0, overlap, mask, edge_vec, edge_src, edge_dst, ele_list, output_path = data_list
    assert overlap is None
    assert output_path == "output_inference.pth"
    assert h0.shape == (expected_edges, 54, 54)
    assert mask.shape == (expected_edges, 54, 54)
    assert edge_vec.shape == (expected_edges, 3)
    assert edge_src.shape == (expected_edges,)
    assert edge_dst.shape == (expected_edges,)
    assert len(ele_list) == expected_edges
    assert h0.dtype == torch.complex64
    assert mask.dtype == torch.float32
    assert edge_vec.dtype == torch.float32
    assert edge_src.dtype == torch.int64
    assert edge_dst.dtype == torch.int64

    if int(edge_src.min()) < 0 or int(edge_dst.min()) < 0:
        raise AssertionError("edge indices must be non-negative")
    if int(edge_src.max()) >= 1000 or int(edge_dst.max()) >= 1000:
        raise AssertionError("edge indices exceed the 1000-atom sample size")

    h0_sample = tensor_sample(h0)
    mask_sample = tensor_sample(mask)
    if not torch.isfinite(h0_sample.real).all() or not torch.isfinite(h0_sample.imag).all():
        raise AssertionError("sampled H0 tensor contains non-finite values")
    if not torch.isfinite(mask_sample).all():
        raise AssertionError("sampled mask tensor contains non-finite values")
    if not torch.all((mask_sample == 0.0) | (mask_sample == 1.0)):
        raise AssertionError("sampled mask tensor is not binary")

    return data_list


def assert_edges_match_reference(data_list, sample_root, cutoff):
    _, _, _, edge_vec, edge_src, edge_dst, _, _ = data_list
    lattice, frac, _ = parse_cif_like_cpp(sample_root / "STRU.cif")
    reference = build_reference_edges(lattice, frac, cutoff)

    edge_vec_np = edge_vec.numpy().astype(np.float64)
    src = edge_src.numpy().astype(np.int32)
    dst = edge_dst.numpy().astype(np.int32)
    d_frac = edge_vec_np @ np.linalg.inv(lattice)
    r_float = d_frac - (frac[dst] - frac[src])
    r = np.rint(r_float).astype(np.int32)
    max_roundoff = float(np.max(np.abs(r_float - r))) if len(r) else 0.0
    if max_roundoff > 2e-3:
        raise AssertionError(f"could not recover integer R offsets, max roundoff={max_roundoff}")

    observed = np.column_stack([r, src, dst]).astype(np.int32)
    order = np.lexsort((observed[:, 4], observed[:, 3], observed[:, 2], observed[:, 1], observed[:, 0]))
    if not np.array_equal(order, np.arange(len(observed))):
        raise AssertionError("C++ preprocess edges are not sorted by (R, src, dst)")
    if reference.shape != observed.shape:
        raise AssertionError(f"edge shape mismatch: reference={reference.shape}, observed={observed.shape}")
    if not np.array_equal(reference, observed):
        mismatch = int(np.nonzero(np.any(reference != observed, axis=1))[0][0])
        raise AssertionError(
            "edge mismatch at sorted index "
            f"{mismatch}: reference={reference[mismatch].tolist()}, observed={observed[mismatch].tolist()}"
        )


def main():
    parser = argparse.ArgumentParser(description="1000-atom preprocessing correctness tests")
    parser.add_argument("--sample-root", type=Path, default=DEFAULT_SAMPLE_ROOT)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--work-dir", type=Path, default=None)
    parser.add_argument("--output-pth", type=Path, default=None)
    parser.add_argument("--keep-output", action="store_true")
    parser.add_argument("--skip-run", action="store_true", help="reuse --output-pth instead of running C++ preprocess")
    parser.add_argument("--skip-reference", action="store_true", help="skip Python brute-force edge comparison")
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=8.0)
    parser.add_argument("--expected-edges", type=int, default=EXPECTED_EDGE_COUNT)
    args = parser.parse_args()

    if not args.sample_root.exists():
        raise AssertionError(f"sample root does not exist: {args.sample_root}")
    if not args.binary.exists():
        raise AssertionError(f"preprocess binary does not exist: {args.binary}")
    if args.skip_run and not args.output_pth:
        raise AssertionError("--skip-run requires --output-pth")

    temp_dir = None
    try:
        if args.output_pth:
            output_pth = args.output_pth
            output_pth.parent.mkdir(parents=True, exist_ok=True)
        else:
            temp_dir = tempfile.TemporaryDirectory(dir=args.work_dir)
            output_pth = Path(temp_dir.name) / "cpp_input_inference_raw.pth"

        if not args.skip_run:
            neighbor_seconds, edge_count, _ = run_preprocess(
                args.binary, args.sample_root, output_pth, 4, args.cutoff, args.threads
            )
            if edge_count != args.expected_edges:
                raise AssertionError(f"expected {args.expected_edges} edges, got {edge_count}")
            print(f"neighbor_seconds={neighbor_seconds:.6f}")

        print(f"loading {output_pth}")
        data_list = assert_preprocess_payload(output_pth, args.expected_edges)
        print("payload checks passed")

        if not args.skip_reference:
            assert_edges_match_reference(data_list, args.sample_root, args.cutoff)
            print("brute-force edge comparison passed")

        print("all preprocessing correctness tests passed")
    finally:
        if temp_dir and not args.keep_output:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
