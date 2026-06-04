#include "neighbor_list.h"
#include <cmath>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <iterator>
#include <omp.h>

std::vector<Edge> NeighborList::build(const StruData& data, double cutoff) {
    std::vector<Edge> edges;
    int n = data.atoms.size();
    if (n == 0) return edges;

    Eigen::Matrix3d lat = data.lattice_vectors * data.lattice_constant;
    Eigen::Matrix3d inv_lat = lat.inverse();
    
    // 1. Determine the range of R-vectors to check
    Eigen::Vector3d v0 = lat.row(0);
    Eigen::Vector3d v1 = lat.row(1);
    Eigen::Vector3d v2 = lat.row(2);
    
    auto get_h = [&](const Eigen::Vector3d& a, const Eigen::Vector3d& b, const Eigen::Vector3d& c) {
        Eigen::Vector3d n_vec = b.cross(c).normalized();
        return std::abs(a.dot(n_vec));
    };
    
    double h0 = get_h(v0, v1, v2);
    double h1 = get_h(v1, v0, v2);
    double h2 = get_h(v2, v0, v1);
    
    int nrx = std::ceil(cutoff / h0);
    int nry = std::ceil(cutoff / h1);
    int nrz = std::ceil(cutoff / h2);

    // 2. Fractional coordinates in [0, 1)
    std::vector<Eigen::Vector3d> frac_pos(n);
    for (int i = 0; i < n; ++i) {
        if (data.is_direct) {
            frac_pos[i] = data.atoms[i].pos;
        } else {
            // Cartesian in Bohr or Angstrom? 
            // After our fix, lattice_constant is in Angstroms if it was Bohr.
            // ABACUS Cartesian positions are typically pos * lattice_constant.
            Eigen::Vector3d p = data.atoms[i].pos * data.lattice_constant;
            frac_pos[i] = inv_lat.transpose() * p;
        }
        for (int k = 0; k < 3; ++k) {
            frac_pos[i](k) -= std::floor(frac_pos[i](k) + 1e-9); // Small epsilon
        }
    }

    double cutoff_sq = cutoff * cutoff;

    // Cell-list search.  Grid cells are chosen in fractional coordinates, with
    // each axis bounded by the real-space cell height.  For a candidate within
    // cutoff, each fractional component must be within cutoff / height, so the
    // per-axis interval below is a safe superset even for skewed cells.
    int gx = std::max(1, (int)std::floor(h0 / (cutoff + 1e-6)));
    int gy = std::max(1, (int)std::floor(h1 / (cutoff + 1e-6)));
    int gz = std::max(1, (int)std::floor(h2 / (cutoff + 1e-6)));
    std::array<int, 3> grid_shape = {gx, gy, gz};
    std::array<double, 3> search_margin = {
        cutoff / h0,
        cutoff / h1,
        cutoff / h2,
    };
    
    std::vector<std::vector<int>> grid(gx * gy * gz);
    auto get_grid_idx = [&](int ix, int iy, int iz) -> int {
        return ix * gy * gz + iy * gz + iz;
    };

    for (int i = 0; i < n; ++i) {
        int ix = std::min(gx - 1, (int)(frac_pos[i](0) * gx));
        int iy = std::min(gy - 1, (int)(frac_pos[i](1) * gy));
        int iz = std::min(gz - 1, (int)(frac_pos[i](2) * gz));
        grid[ix * gy * gz + iy * gz + iz].push_back(i);
    }

    int num_threads = std::max(1, omp_get_max_threads());
    std::vector<std::vector<Edge>> thread_edges(num_threads);

    auto get_cell_range = [](double lo, double hi, int g, int& c0, int& c1) -> bool {
        if (hi < 0.0 || lo >= 1.0) {
            return false;
        }
        lo = std::max(0.0, lo);
        hi = std::min(std::nextafter(1.0, 0.0), hi);
        c0 = std::max(0, std::min(g - 1, (int)std::floor(lo * g)));
        c1 = std::max(0, std::min(g - 1, (int)std::floor(hi * g)));
        return c0 <= c1;
    };

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        auto& local_edges = thread_edges[tid];

        #pragma omp for schedule(dynamic, 16)
        for (int i = 0; i < n; ++i) {
            Eigen::Vector3d pi = frac_pos[i];

            for (int rx = -nrx; rx <= nrx; ++rx) {
                for (int ry = -nry; ry <= nry; ++ry) {
                    for (int rz = -nrz; rz <= nrz; ++rz) {
                        Eigen::Vector3d center = pi - Eigen::Vector3d(rx, ry, rz);
                        int c0[3];
                        int c1[3];
                        bool has_cells = true;
                        for (int axis = 0; axis < 3; ++axis) {
                            double lo = center(axis) - search_margin[axis];
                            double hi = center(axis) + search_margin[axis];
                            if (!get_cell_range(lo, hi, grid_shape[axis], c0[axis], c1[axis])) {
                                has_cells = false;
                                break;
                            }
                        }
                        if (!has_cells) {
                            continue;
                        }

                        Eigen::Vector3d R_vec(rx, ry, rz);
                        for (int cx = c0[0]; cx <= c1[0]; ++cx) {
                            for (int cy = c0[1]; cy <= c1[1]; ++cy) {
                                for (int cz = c0[2]; cz <= c1[2]; ++cz) {
                                    const auto& candidates = grid[get_grid_idx(cx, cy, cz)];
                                    for (int j : candidates) {
                                        Eigen::Vector3d d_frac = frac_pos[j] + R_vec - pi;
                                        Eigen::Vector3d d_cart = lat.transpose() * d_frac;
                                        double d2 = d_cart.squaredNorm();
                                        if (d2 < cutoff_sq) {
                                            local_edges.push_back({i, j, d_cart, {rx, ry, rz}, std::sqrt(d2)});
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    size_t total_edges = 0;
    for (const auto& local_edges : thread_edges) {
        total_edges += local_edges.size();
    }
    edges.reserve(total_edges);
    for (auto& local_edges : thread_edges) {
        edges.insert(edges.end(),
                     std::make_move_iterator(local_edges.begin()),
                     std::make_move_iterator(local_edges.end()));
    }

    // Sort edges for consistency as in Python
    std::sort(edges.begin(), edges.end(), [](const Edge& a, const Edge& b) {
        if (a.r_offset(0) != b.r_offset(0)) return a.r_offset(0) < b.r_offset(0);
        if (a.r_offset(1) != b.r_offset(1)) return a.r_offset(1) < b.r_offset(1);
        if (a.r_offset(2) != b.r_offset(2)) return a.r_offset(2) < b.r_offset(2);
        if (a.atom_i != b.atom_i) return a.atom_i < b.atom_i;
        return a.atom_j < b.atom_j;
    });

    return edges;
}
