// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include "tracking.h"
#include "nn_inference.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <algorithm>
#include <numeric>
#include <queue>
#include <cmath>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <random>

namespace freetrace {

// ============================================================
// Global state for empirical PDF (matches Python module globals)
// ============================================================
static std::vector<float> g_emp_pdf; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<float> g_emp_bins; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static int g_emp_nb_bins = 40; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static float g_emp_max_val = 20.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================
// Global state for qt_99 data (abnormal detection)
// ============================================================
static std::vector<double> g_qt_alpha_arr; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<double> g_qt_k_arr; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<double> g_qt_mean; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 [n_alpha * n_k, row-major]
static bool g_qt_loaded = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// Global NN models for alpha/k inference // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static NNModels g_nn_models; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

static bool load_qt_data(const std::string& path) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) return false;
    int32_t n_alpha, n_k; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    f.read(reinterpret_cast<char*>(&n_alpha), 4);
    f.read(reinterpret_cast<char*>(&n_k), 4);
    g_qt_alpha_arr.resize(n_alpha); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    g_qt_k_arr.resize(n_k);
    g_qt_mean.resize(n_alpha * n_k);
    f.read(reinterpret_cast<char*>(g_qt_alpha_arr.data()), n_alpha * 8); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    f.read(reinterpret_cast<char*>(g_qt_k_arr.data()), n_k * 8);
    f.read(reinterpret_cast<char*>(g_qt_mean.data()), n_alpha * n_k * 8);
    g_qt_loaded = f.good(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return g_qt_loaded;
}

static std::pair<int,int> indice_fetch(double alpha, double k) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int alpha_index = (int)g_qt_alpha_arr.size() - 1;
    int k_index = (int)g_qt_k_arr.size() - 1;
    for (int i = 0; i < (int)g_qt_alpha_arr.size(); i++) {
        if (alpha < g_qt_alpha_arr[i]) { alpha_index = i; break; }
    }
    for (int i = 0; i < (int)g_qt_k_arr.size(); i++) {
        if (k < g_qt_k_arr[i]) { k_index = i; break; }
    }
    return {alpha_index, k_index};
}

static double qt_fetch(int alpha_i, int k_i) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return g_qt_mean[alpha_i * (int)g_qt_k_arr.size() + k_i];
}

// ============================================================
// DiGraph implementation
// ============================================================

void DiGraph::add_node(const Node& n) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    nodes_.insert(n);
}

void DiGraph::add_edge(const Node& from, const Node& to, const EdgeData& data) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    nodes_.insert(from);
    nodes_.insert(to);
    fwd_[from][to] = data;
    rev_[to].insert(from);
}

void DiGraph::remove_node(const Node& n) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Remove all edges from predecessors to n
    if (rev_.count(n)) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& pred : rev_[n]) {
            if (fwd_.count(pred)) fwd_[pred].erase(n);
        }
        rev_.erase(n);
    }
    // Remove all edges from n to successors
    if (fwd_.count(n)) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& [succ, _] : fwd_[n]) {
            if (rev_.count(succ)) rev_[succ].erase(n);
        }
        fwd_.erase(n);
    }
    nodes_.erase(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

void DiGraph::remove_edge(const Node& from, const Node& to) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (fwd_.count(from)) fwd_[from].erase(to);
    if (rev_.count(to)) rev_[to].erase(from);
}

bool DiGraph::has_node(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return nodes_.count(n) > 0;
}

bool DiGraph::has_edge(const Node& from, const Node& to) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto it = fwd_.find(from);
    if (it == fwd_.end()) return false;
    return it->second.count(to) > 0;
}

bool DiGraph::has_path(const Node& from, const Node& to) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (from == to) return true;
    std::set<Node> visited;
    std::queue<Node> q;
    q.push(from);
    visited.insert(from);
    while (!q.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Node cur = q.front(); q.pop();
        auto it = fwd_.find(cur);
        if (it == fwd_.end()) continue;
        for (const auto& [next, _] : it->second) {
            if (next == to) return true;
            if (!visited.count(next)) {
                visited.insert(next);
                q.push(next);
            }
        }
    }
    return false;
}

std::vector<Node> DiGraph::successors(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> result;
    auto it = fwd_.find(n);
    if (it != fwd_.end()) {
        for (const auto& [succ, _] : it->second) result.push_back(succ);
    }
    return result;
}

std::vector<Node> DiGraph::predecessors(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> result;
    auto it = rev_.find(n);
    if (it != rev_.end()) {
        for (const auto& pred : it->second) result.push_back(pred);
    }
    return result;
}

const EdgeData& DiGraph::get_edge_data(const Node& from, const Node& to) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    static EdgeData empty;
    auto it = fwd_.find(from);
    if (it == fwd_.end()) return empty;
    auto it2 = it->second.find(to);
    if (it2 == it->second.end()) return empty;
    return it2->second;
}

const std::set<Node>& DiGraph::get_nodes() const { return nodes_; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

size_t DiGraph::size() const { return nodes_.size(); } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

DiGraph DiGraph::copy() const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    DiGraph g;
    g.nodes_ = nodes_;
    g.fwd_ = fwd_;
    g.rev_ = rev_;
    return g;
}

std::vector<Node> DiGraph::ancestors(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> result;
    std::set<Node> visited;
    std::queue<Node> q;
    auto it = rev_.find(n);
    if (it != rev_.end()) {
        for (const auto& pred : it->second) {
            if (!visited.count(pred)) {
                visited.insert(pred);
                q.push(pred);
            }
        }
    }
    while (!q.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Node cur = q.front(); q.pop();
        result.push_back(cur);
        auto it2 = rev_.find(cur);
        if (it2 != rev_.end()) {
            for (const auto& pred : it2->second) {
                if (!visited.count(pred)) {
                    visited.insert(pred);
                    q.push(pred);
                }
            }
        }
    }
    return result;
}

// ============================================================
// TrajectoryObj
// ============================================================

void TrajectoryObj::add_trajectory_tuple(int frame, int idx) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    tuples.push_back({frame, idx});
}

int TrajectoryObj::get_trajectory_length() const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return (int)tuples.size();
}

std::vector<int> TrajectoryObj::get_times() const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> times;
    for (const auto& t : tuples) times.push_back(t.first);
    return times;
}

// ============================================================
// Distance utilities
// ============================================================

float euclidean_displacement_single(const std::array<float,3>& a, const std::array<float,3>& b) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

std::vector<float> euclidean_displacement_batch(const std::vector<std::array<float,3>>& a, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                                const std::vector<std::array<float,3>>& b) {
    std::vector<float> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        float dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = a[i][2] - b[i][2];
        result[i] = std::sqrt(dx*dx + dy*dy + dz*dz);
    }
    return result;
}

// ============================================================
// I/O: read localization CSV, write trajectory CSV
// ============================================================

Localizations read_localization_csv(const std::string& path, int nb_frames) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    Localizations locs;
    std::ifstream file(path);
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << path << std::endl;
        return locs;
    }
    std::string line;
    std::getline(file, line); // skip header // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Parse header to find column indices // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::istringstream hss(line);
    std::string col;
    std::vector<std::string> headers;
    while (std::getline(hss, col, ',')) {
        // lowercase // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (auto& c : col) c = std::tolower(c);
        // trim whitespace // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        while (!col.empty() && (col.front() == ' ' || col.front() == '\r')) col.erase(col.begin());
        while (!col.empty() && (col.back() == ' ' || col.back() == '\r')) col.pop_back();
        headers.push_back(col);
    }
    int frame_col = -1, x_col = -1, y_col = -1, z_col = -1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < (int)headers.size(); i++) {
        if (headers[i] == "frame") frame_col = i;
        else if (headers[i] == "x") x_col = i;
        else if (headers[i] == "y") y_col = i;
        else if (headers[i] == "z") z_col = i;
    }
    if (frame_col < 0 || x_col < 0 || y_col < 0) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cerr << "Error: CSV must have frame, x, y columns" << std::endl;
        return locs;
    }

    while (std::getline(file, line)) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::vector<std::string> vals;
        std::string val;
        while (std::getline(ss, val, ',')) vals.push_back(val);

        int frame = std::stoi(vals[frame_col]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float x = std::stof(vals[x_col]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float y = std::stof(vals[y_col]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float z = (z_col >= 0 && z_col < (int)vals.size()) ? std::stof(vals[z_col]) : 0.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        locs[frame].push_back({x, y, z});
    }

    // Fill empty frames (matching Python: locals[t] = [[]] -> empty vector) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int t = 1; t <= nb_frames; t++) {
        if (locs.find(t) == locs.end()) {
            locs[t] = {}; // empty vector = no particles // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }
    return locs;
}

void write_trajectory_csv(const std::string& path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                          const std::vector<TrajectoryObj>& trajectories,
                          const Localizations& locs) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write to " << path << std::endl;
        return;
    }
    f << "traj_idx,frame,x,y,z\n"; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& traj : trajectories) {
        for (const auto& [frame, idx] : traj.tuples) {
            const auto& pos = locs.at(frame)[idx]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            f << traj.index << "," << frame << "," << pos[0] << "," << pos[1] << "," << pos[2] << "\n";
        }
    }
}

// ============================================================
// Greedy shortest matching (segmentation helper)
// ============================================================

GreedyResult greedy_shortest(const std::vector<std::array<float,3>>& srcs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                             const std::vector<std::array<float,3>>& dests) {
    GreedyResult result;
    int ns = (int)srcs.size(), nd = (int)dests.size();
    if (ns == 0 || nd == 0) return result;

    // Compute all pairwise distances // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> linkage(ns * nd, 0.0f);
    std::vector<float> x_diff(ns * nd), y_diff(ns * nd), z_diff(ns * nd);
    for (int i = 0; i < ns; i++) {
        for (int j = 0; j < nd; j++) {
            int idx = i * nd + j;
            x_diff[idx] = srcs[i][0] - dests[j][0];
            y_diff[idx] = srcs[i][1] - dests[j][1];
            z_diff[idx] = srcs[i][2] - dests[j][2];
            linkage[idx] = std::sqrt(x_diff[idx]*x_diff[idx] + y_diff[idx]*y_diff[idx] + z_diff[idx]*z_diff[idx]);
        }
    }

    // Sort by distance // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> minargs(ns * nd);
    std::iota(minargs.begin(), minargs.end(), 0);
    std::sort(minargs.begin(), minargs.end(), [&](int a, int b) {
        return linkage[a] < linkage[b];
    });

    std::vector<bool> linked_src(ns, false), linked_dest(nd, false); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int minarg : minargs) {
        int src = minarg / nd;
        int dest = minarg % nd;
        if (linked_src[src] || linked_dest[dest]) continue;
        linked_src[src] = true;
        linked_dest[dest] = true;
        result.x_dist.push_back(x_diff[minarg]);
        result.y_dist.push_back(y_diff[minarg]);
        result.z_dist.push_back(z_diff[minarg]);
        result.jump_dist.push_back(linkage[src * nd + dest]);
    }

    // Remove last element from x/y/z (matching Python's [:-1]) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (!result.x_dist.empty()) result.x_dist.pop_back();
    if (!result.y_dist.empty()) result.y_dist.pop_back();
    if (!result.z_dist.empty()) result.z_dist.pop_back();

    return result;
}

// ============================================================
// Segmentation
// ============================================================

SegmentationResult segmentation(const Localizations& loc, const std::vector<int>& time_steps, int lag) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    lag = 0; // Python forces lag=0 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    SegmentationResult result;
    for (int i = 0; i <= lag; i++) result.seg_distribution[i] = {};

    int n_steps = (int)time_steps.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n_steps - lag - 1; i++) {
        int t_src = time_steps[i];
        const auto& srcs = loc.at(t_src);
        if (srcs.empty()) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        for (int j = i + 1; j <= i + lag + 1; j++) {
            int t_dest = time_steps[j];
            const auto& dests = loc.at(t_dest);
            if (dests.empty()) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            auto gr = greedy_shortest(srcs, dests); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            result.dist_x.insert(result.dist_x.end(), gr.x_dist.begin(), gr.x_dist.end());
            result.dist_y.insert(result.dist_y.end(), gr.y_dist.begin(), gr.y_dist.end());
            result.dist_z.insert(result.dist_z.end(), gr.z_dist.begin(), gr.z_dist.end());
            result.seg_distribution[lag].insert(result.seg_distribution[lag].end(),
                                                gr.jump_dist.begin(), gr.jump_dist.end());
        }
    }

    // Outlier filtering (2 rounds of 4*mean_std filter + diffraction limit) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float diffraction_light_limit = 10.0f;
    auto compute_std = [](const std::vector<float>& v) -> float {
        if (v.size() < 2) return 0.0f;
        float mean = 0;
        for (size_t i = 0; i + 1 < v.size(); i++) mean += v[i];
        mean /= (v.size() - 1);
        float var = 0;
        for (size_t i = 0; i + 1 < v.size(); i++) var += (v[i] - mean) * (v[i] - mean);
        return std::sqrt(var / (v.size() - 2)); // sample std, n-1
    };

    // Check dimensionality // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float z_var = 0;
    if (!result.dist_z.empty()) {
        float z_mean = 0;
        for (float v : result.dist_z) z_mean += v;
        z_mean /= result.dist_z.size();
        for (float v : result.dist_z) z_var += (v - z_mean) * (v - z_mean);
        z_var /= result.dist_z.size();
    }
    int ndim = (z_var < 1e-5f) ? 2 : 3; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    for (int round = 0; round < 2; round++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float std_x = compute_std(result.dist_x);
        float std_y = compute_std(result.dist_y);
        float std_z = compute_std(result.dist_z);
        float estim_limit = (ndim == 2) ? 4.0f * (std_x + std_y) / 2.0f
                                        : 4.0f * (std_x + std_y + std_z) / 3.0f;
        float filter_min = std::max(estim_limit, diffraction_light_limit);

        std::vector<float> fx, fy, fz;
        for (size_t i = 0; i + 1 < result.dist_x.size(); i++) {
            if (std::abs(result.dist_x[i]) < filter_min &&
                std::abs(result.dist_y[i]) < filter_min &&
                std::abs(result.dist_z[i]) < filter_min) {
                fx.push_back(result.dist_x[i]);
                fy.push_back(result.dist_y[i]);
                fz.push_back(result.dist_z[i]);
            }
        }
        result.dist_x = fx;
        result.dist_y = fy;
        result.dist_z = fz;
    }

    // Final diffraction limit filter // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> fx, fy, fz;
    for (size_t i = 0; i + 1 < result.dist_x.size(); i++) {
        if (std::abs(result.dist_x[i]) < diffraction_light_limit &&
            std::abs(result.dist_y[i]) < diffraction_light_limit &&
            std::abs(result.dist_z[i]) < diffraction_light_limit) {
            fx.push_back(result.dist_x[i]);
            fy.push_back(result.dist_y[i]);
            fz.push_back(result.dist_z[i]);
        }
    }
    result.dist_x = fx;
    result.dist_y = fy;
    result.dist_z = fz;
    return result;
}

// ============================================================
// Approximation: simplified GMM replacement
// When jump_threshold > 0, uses fixed threshold.
// When jump_threshold <= 0, estimates from std of distributions.
// ============================================================

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// 1D Gaussian Mixture Model with EM algorithm
// ============================================================

// Fit 1D GMM with n_components using Expectation-Maximization // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Returns (weights, means, variances, log_likelihood)
struct GMM1DResult { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> weights; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> means; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> variances; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double log_likelihood; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};

static double gauss_pdf_1d(double x, double mean, double var) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double d = x - mean; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return std::exp(-0.5 * d * d / var) / std::sqrt(2.0 * M_PI * var); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

static GMM1DResult fit_gmm_1d(const std::vector<double>& data, int n_comp, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                               int max_iter = 100, int n_init = 3,
                               bool use_mean_prior = false,
                               double mean_prior = 0.0,
                               double mean_precision_prior = 1e7) {
    int n = (int)data.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (n == 0) return {{}, {}, {}, -1e30}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    GMM1DResult best_result; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    best_result.log_likelihood = -1e30; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Compute data variance for initialization // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double data_mean = 0, data_var = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (double x : data) data_mean += x; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    data_mean /= n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (double x : data) data_var += (x - data_mean) * (x - data_mean); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    data_var /= n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (data_var < 1e-10) data_var = 1e-10; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    std::mt19937 rng(42); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    for (int init = 0; init < n_init; init++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Initialize: all means at 0 (matching Python's means_init=[[0],[0],...]) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> weights(n_comp, 1.0 / n_comp); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> means(n_comp, 0.0); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> vars(n_comp, data_var); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Add small perturbation for multi-init // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (init > 0) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            std::normal_distribution<double> nd(0, std::sqrt(data_var) * 0.1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int k = 0; k < n_comp; k++) means[k] = nd(rng); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }

        // Responsibilities matrix // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::vector<double>> resp(n, std::vector<double>(n_comp)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        double prev_ll = -1e30; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int iter = 0; iter < max_iter; iter++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            // E-step // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double ll = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double total = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int k = 0; k < n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    resp[i][k] = weights[k] * gauss_pdf_1d(data[i], means[k], vars[k]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    total += resp[i][k]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                }
                if (total < 1e-300) total = 1e-300; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                ll += std::log(total); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int k = 0; k < n_comp; k++) resp[i][k] /= total; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }

            // Check convergence // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (std::abs(ll - prev_ll) < 1e-6) break; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            prev_ll = ll; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            // M-step // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int k = 0; k < n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double nk = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int i = 0; i < n; i++) nk += resp[i][k]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (nk < 1e-10) nk = 1e-10; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

                weights[k] = nk / n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

                // Mean update with optional Bayesian prior // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double sum_x = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int i = 0; i < n; i++) sum_x += resp[i][k] * data[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (use_mean_prior) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    means[k] = (sum_x + mean_precision_prior * mean_prior) / (nk + mean_precision_prior); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                } else {
                    means[k] = sum_x / nk; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                }

                double sum_var = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    double d = data[i] - means[k]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    sum_var += resp[i][k] * d * d; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                }
                vars[k] = sum_var / nk; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (vars[k] < 1e-10) vars[k] = 1e-10; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }

        // Compute final log-likelihood // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double ll = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double total = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int k = 0; k < n_comp; k++) total += weights[k] * gauss_pdf_1d(data[i], means[k], vars[k]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (total < 1e-300) total = 1e-300; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            ll += std::log(total); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }

        if (ll > best_result.log_likelihood) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            best_result.weights = weights; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            best_result.means = means; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            best_result.variances = vars; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            best_result.log_likelihood = ll; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }
    return best_result; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// BIC score for 1D GMM: -2*LL + k*log(n) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static double gmm_bic(const GMM1DResult& result, int n_comp, int n_data) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int n_params = 3 * n_comp - 1; // weights(k-1) + means(k) + vars(k) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return -2.0 * result.log_likelihood + n_params * std::log((double)n_data); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// approx_gauss: full GMM-based jump threshold estimation (matches Python) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static float approx_gauss(const std::vector<std::vector<float>>& distributions) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float min_euclid = 5.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> max_xyz; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    for (const auto& raw_dist : distributions) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (raw_dist.empty()) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Check variance // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double mean_v = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (float x : raw_dist) mean_v += x; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        mean_v /= raw_dist.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double var_v = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (float x : raw_dist) var_v += (x - mean_v) * (x - mean_v); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        var_v /= raw_dist.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (var_v <= 1e-5) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Quantile filter (2.5% - 97.5%) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> sorted_dist = raw_dist; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::sort(sorted_dist.begin(), sorted_dist.end()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int n = (int)sorted_dist.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float q025 = sorted_dist[(int)(0.025 * (n - 1))]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float q975 = sorted_dist[(int)(0.975 * (n - 1))]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> filtered; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (float x : raw_dist) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (x > q025 && x < q975) filtered.push_back((double)x); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
        if (filtered.empty()) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Recheck variance after filtering // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double fmean = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (double x : filtered) fmean += x; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        fmean /= filtered.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double fvar = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (double x : filtered) fvar += (x - fmean) * (x - fmean); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        fvar /= filtered.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (fvar <= 1e-5) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // GridSearch: fit GMM with 1,2,3 components, select by BIC // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int best_n_comp = 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double best_bic = 1e30; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int nc = 1; nc <= 3; nc++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto result = fit_gmm_1d(filtered, nc, 100, 3, false); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double bic = gmm_bic(result, nc, (int)filtered.size()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (bic < best_bic) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                best_bic = bic; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                best_n_comp = nc; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }

        // Bayesian GMM with optimal components and mean prior at 0 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto cluster = fit_gmm_1d(filtered, best_n_comp, 100, 3, true, 0.0, 1e7); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Select components near zero with weight > 0.05 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> selec_var; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int k = 0; k < best_n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (cluster.means[k] > -1.0 && cluster.means[k] < 1.0 && cluster.weights[k] > 0.05) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                selec_var.push_back(cluster.variances[k]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
        if (selec_var.empty()) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Take the component with largest variance // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double max_var = *std::max_element(selec_var.begin(), selec_var.end()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        max_xyz.push_back((float)(std::sqrt(max_var) * 2.5)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    // Compute Euclidean norm across dimensions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float max_euclid_sq = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (float v : max_xyz) max_euclid_sq += v * v; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return std::max(std::sqrt(max_euclid_sq), min_euclid); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

std::map<int, float> approximation(const std::vector<float>& dist_x, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                   const std::vector<float>& dist_y,
                                   const std::vector<float>& dist_z,
                                   int time_forecast, float jump_threshold) {
    std::map<int, float> approx;
    if (jump_threshold > 0) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int t = 0; t <= time_forecast; t++) approx[t] = jump_threshold;
    } else {
        // Full GMM-based threshold estimation (matches Python approx_gauss) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::vector<float>> distributions; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        distributions.push_back(dist_x); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        distributions.push_back(dist_y); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (!dist_z.empty()) distributions.push_back(dist_z); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float max_euclid = approx_gauss(distributions); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int t = 0; t <= time_forecast; t++) approx[t] = max_euclid;
    }
    return approx;
}

// ============================================================
// Empirical PDF
// ============================================================

void build_emp_pdf(const std::vector<float>& emp_distribution, int nb_bins, float max_val) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    g_emp_nb_bins = nb_bins;
    g_emp_max_val = max_val;
    g_emp_bins.resize(nb_bins + 1);
    for (int i = 0; i <= nb_bins; i++) g_emp_bins[i] = max_val * i / nb_bins;

    std::vector<float> data; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if ((int)emp_distribution.size() < 1000) {
        // Generate exponential(1.0) samples // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::mt19937 rng(42);
        std::exponential_distribution<float> exp_dist(1.0f);
        data.resize(10000);
        for (auto& v : data) v = exp_dist(rng);
    } else {
        data = emp_distribution;
    }

    // Compute histogram (density=True) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    g_emp_pdf.assign(nb_bins, 0.0f);
    for (float v : data) {
        int bin = (int)(v / max_val * nb_bins);
        if (bin >= 0 && bin < nb_bins) g_emp_pdf[bin] += 1.0f;
    }
    // Normalize to density // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float bin_width = max_val / nb_bins;
    float total = 0;
    for (float c : g_emp_pdf) total += c;
    if (total > 0) {
        for (float& c : g_emp_pdf) c /= (total * bin_width);
    }
}

float empirical_pdf_lookup(const std::array<float,3>& coords) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float jump_d = std::sqrt(coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2]);
    int idx = (int)std::min(jump_d / g_emp_bins.back() * g_emp_nb_bins, (float)(g_emp_nb_bins - 1));
    idx = std::max(0, idx);
    return std::log(g_emp_pdf[idx] + 1e-7f);
}

// ============================================================
// Regularization helpers (for predict_cauchy_tracking)
// ============================================================

static float scaling_func(float numer) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return std::max(0.0f, std::min(2.0f, -std::log10(std::abs(numer))));
}

static float regularization(float numer, float denom, float expect) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (numer < 0) numer -= 2e-1f; else numer += 2e-1f;
    if (denom < 0) denom -= 2e-1f; else denom += 2e-1f;
    float scaler = std::min(std::sqrt(scaling_func(numer) + 2.0f * scaling_func(denom)) / 2.0f, 1.0f);
    float scaled_ratio = (numer / denom) + (expect - numer / denom) * std::pow(scaler, 1.0f / 3.0f);
    return scaled_ratio;
}

// ============================================================
// Cauchy cost (tracking version)
// ============================================================

CauchyResult predict_cauchy_tracking(const std::array<float,3>& next_vec, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                     const std::array<float,3>& prev_vec,
                                     float k, float alpha,
                                     int before_lag, int lag,
                                     float precision, int dimension) {
    CauchyResult result = {0.0f, false};
    float delta_s = (float)(before_lag + 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float delta_t = (float)(lag + 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    if (std::abs(alpha - 1.0f) < 1e-4f) alpha += 1e-4f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float rho = std::pow(2.0f, alpha - 1.0f) - 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float std_ratio = std::sqrt( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        (std::pow(2.0f * delta_t, alpha) - 2.0f * std::pow(delta_t, alpha)) /
        (std::pow(2.0f * delta_s, alpha) - 2.0f * std::pow(delta_s, alpha))
    );
    float scale = std::sqrt(1.0f - rho * rho) * std_ratio; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    std::vector<float> coord_ratios; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int d = 0; d < dimension; d++) {
        float cr = regularization(next_vec[d], prev_vec[d], rho * std_ratio);
        coord_ratios.push_back(cr);
    }

    for (float cr : coord_ratios) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float density = (1.0f / ((float)M_PI * scale)) *
                        (1.0f / (((cr - rho * std_ratio) * (cr - rho * std_ratio)) /
                                 (scale * scale * std_ratio) + std_ratio));
        result.log_pdf += std::log(density);
    }

    // Abnormal check: |coord_ratio - std_ratio*rho| > 6*scale // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (float cr : coord_ratios) {
        if (std::abs(cr - std_ratio * rho) > 6.0f * scale) {
            result.abnormal = true;
            break;
        }
    }

    // qt_fetch-based abnormal check // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (g_qt_loaded) {
        auto [ai, ki] = indice_fetch((double)alpha, (double)k); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double qt_val = qt_fetch(ai, ki);
        float jump_mag = std::sqrt(next_vec[0]*next_vec[0] + next_vec[1]*next_vec[1] + next_vec[2]*next_vec[2]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (jump_mag - (float)qt_val * 2.5f > 0) {
            result.abnormal = true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }

    return result;
}

// ============================================================
// DFS path enumeration
// ============================================================

void find_paths_dfs(const DiGraph& G, const Node& current, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    Path& path, std::set<Node>& seen,
                    std::vector<Path>& results) {
    seen.insert(current);
    auto neighbors = G.successors(current);

    if (neighbors.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        results.push_back(path); // leaf: yield the path
    }
    for (const auto& neighbor : neighbors) {
        if (!seen.count(neighbor)) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            path.push_back(neighbor);
            find_paths_dfs(G, neighbor, path, seen, results);
            path.pop_back();
        }
    }
    seen.erase(current);
}

std::vector<Path> find_paths_as_list(const DiGraph& G, const Node& source) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Path> all_paths;
    Path path = {source};
    std::set<Node> seen;
    find_paths_dfs(G, source, path, seen, all_paths);

    // Filter: only paths with length > 1 (matching Python) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Path> filtered;
    for (auto& p : all_paths) {
        if (p.size() > 1) filtered.push_back(std::move(p));
    }
    return filtered;
}

// ============================================================
// Split to subgraphs (connected components from source)
// Matches Python split_to_subgraphs exactly
// ============================================================

std::vector<DiGraph> split_to_subgraphs(const DiGraph& G, const Node& source_node) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<DiGraph> subgraphs;

    // Get first edges (source -> direct neighbors) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto first_neighbors = G.successors(source_node);

    // Build undirected graph without source // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Represent as adjacency set (undirected)
    std::set<Node> all_nodes;
    std::map<Node, std::set<Node>> undirected;
    for (const auto& n : G.get_nodes()) {
        if (n == source_node) continue;
        all_nodes.insert(n);
        for (const auto& succ : G.successors(n)) {
            if (succ == source_node) continue;
            undirected[n].insert(succ);
            undirected[succ].insert(n);
        }
        for (const auto& pred : G.predecessors(n)) {
            if (pred == source_node) continue;
            undirected[n].insert(pred);
            undirected[pred].insert(n);
        }
    }

    std::set<Node> remaining = all_nodes; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    while (!remaining.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // BFS from arbitrary node in undirected graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Node arb_node = *remaining.begin();
        std::vector<Node> bfs_nodes;
        std::set<Node> bfs_visited;
        std::queue<Node> bfs_q;
        bfs_q.push(arb_node);
        bfs_visited.insert(arb_node);
        bool has_edges = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        while (!bfs_q.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            Node cur = bfs_q.front(); bfs_q.pop();
            bfs_nodes.push_back(cur);
            if (undirected.count(cur)) {
                for (const auto& nb : undirected[cur]) {
                    has_edges = true;
                    if (!bfs_visited.count(nb)) {
                        bfs_visited.insert(nb);
                        bfs_q.push(nb);
                    }
                }
            }
        }

        // Build directed subgraph from BFS component // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        DiGraph sub_graph;
        for (const auto& node : bfs_nodes) {
            if (undirected.count(node)) {
                for (const auto& nb : undirected[node]) {
                    // Add edge in time-forward direction // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    if (node.first < nb.first) {
                        sub_graph.add_edge(node, nb);
                    } else if (nb.first < node.first) {
                        sub_graph.add_edge(nb, node);
                    }
                }
            }
        }

        // Remove processed nodes // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& n : sub_graph.get_nodes()) remaining.erase(n);

        // Handle isolated nodes (no edges in BFS) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (!has_edges && !remaining.empty()) {
            // Actually check: if bfs found no edges but node exists
        }
        if (bfs_nodes.size() == 1 && !has_edges) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            sub_graph.add_edge(source_node, arb_node);
            remaining.erase(arb_node);
        }

        // Add source edges to subgraph for any first_neighbor in this component // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& fn : first_neighbors) {
            if (sub_graph.has_node(fn)) {
                sub_graph.add_edge(source_node, fn);
            }
        }

        subgraphs.push_back(std::move(sub_graph));
    }
    return subgraphs;
}

// ============================================================
// Terminal check
// ============================================================

bool is_terminal_node(const Node& node, const Localizations& locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                      float max_jump_d, const DiGraph& selected_graph,
                      const std::set<Node>& final_graph_nodes) {
    int node_t = node.first; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Check if any future node is reachable
    auto node_loc = locs.at(node_t)[node.second]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& [t, particles] : locs) {
        if (t <= node_t) continue;
        if (t - node_t > 5) break; // limit search range
        for (int idx = 0; idx < (int)particles.size(); idx++) {
            if (particles.empty()) continue;
            Node next_node = {t, idx};
            if (final_graph_nodes.count(next_node)) continue;
            if (selected_graph.has_node(next_node)) continue;
            float d = euclidean_displacement_single(node_loc, particles[idx]);
            if (d < max_jump_d) return false; // can still connect
        }
    }
    return true; // no reachable future nodes // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// ============================================================
// Generate next paths (build graph edges)
// ============================================================

GenerateResult generate_next_paths(DiGraph next_graph, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                   const std::set<Node>& final_graph_nodes,
                                   const Localizations& locs,
                                   const std::vector<int>& next_times,
                                   const std::map<int, float>& distribution,
                                   const Node& source_node) {
    std::vector<Node> cumulative_last_nodes;

    while (true) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        size_t start_g_len = next_graph.size();
        int index = 0;
        cumulative_last_nodes.clear();

        while (true) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            // Get last nodes of all paths // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto paths = find_paths_as_list(next_graph, source_node);
            // Also include single-node paths (source only yields leaf neighbors)
            for (const auto& p : paths) {
                Node last = p.back();
                bool found = false;
                for (const auto& ln : cumulative_last_nodes) {
                    if (ln == last) { found = true; break; }
                }
                if (!found) cumulative_last_nodes.push_back(last);
            }

            // Connect last nodes to particles in next_times[index] // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (index < (int)next_times.size()) {
                int cur_time = next_times[index]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

                for (const auto& last_node : cumulative_last_nodes) {
                    if (last_node.first >= cur_time || last_node == source_node) continue;
                    if (!locs.count(last_node.first) || locs.at(last_node.first).empty()) continue;
                    if (!locs.count(cur_time) || locs.at(cur_time).empty()) continue;

                    auto node_loc = locs.at(last_node.first)[last_node.second]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    const auto& cur_particles = locs.at(cur_time);

                    for (int next_idx = 0; next_idx < (int)cur_particles.size(); next_idx++) {
                        float jump_d = euclidean_displacement_single(cur_particles[next_idx], node_loc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        int time_gap = cur_time - last_node.first - 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        auto it = distribution.find(time_gap);
                        if (it != distribution.end()) {
                            if (jump_d < it->second) {
                                Node next_node = {cur_time, next_idx};
                                if (!final_graph_nodes.count(next_node)) {
                                    EdgeData ed; ed.jump_d = jump_d;
                                    next_graph.add_edge(last_node, next_node, ed);
                                }
                            }
                        }
                    }
                }

                // Add unconnected particles as orphans from source // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (locs.count(cur_time) && !locs.at(cur_time).empty()) {
                    for (int idx = 0; idx < (int)locs.at(cur_time).size(); idx++) {
                        Node n = {cur_time, idx};
                        if (!next_graph.has_node(n) && !final_graph_nodes.count(n)) {
                            EdgeData ed; ed.jump_d = -1;
                            next_graph.add_edge(source_node, n, ed);
                        }
                    }
                }
            }

            index++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (index >= (int)next_times.size()) break;
        }

        size_t end_g_len = next_graph.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (start_g_len == end_g_len) break;
    }
    return {next_graph, cumulative_last_nodes};
}

// ============================================================
// Match previous path to next path
// ============================================================

const Path* match_prev_next(const std::vector<Path>& prev_paths, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                            const Path& next_path,
                            std::map<Path, int>& hashed_prev_next) {
    auto it = hashed_prev_next.find(next_path);
    if (it != hashed_prev_next.end()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (it->second < 0) return nullptr;
        return &prev_paths[it->second];
    }

    for (int i = 0; i < (int)prev_paths.size(); i++) {
        if (prev_paths[i].size() > 1) {
            Node last_prev = prev_paths[i].back();
            for (const auto& n : next_path) {
                if (n == last_prev) {
                    hashed_prev_next[next_path] = i;
                    return &prev_paths[i];
                }
            }
        }
    }
    hashed_prev_next[next_path] = -1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return nullptr;
}

// ============================================================
// Predict long sequence cost
// ============================================================

static const float UNCOMPUTED = -999999.0f; // sentinel for uncomputed cost // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

PredictResult predict_long_seq(const Path& next_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                               std::map<Path, float>& trajectories_costs,
                               const Localizations& locs,
                               float prev_alpha, float prev_k,
                               const std::vector<int>& next_times,
                               const Path* prev_path_ptr,
                               std::map<Path, int>& start_indice,
                               int last_time, float jump_threshold,
                               const DiGraph& selected_graph,
                               const std::set<Node>& final_graph_nodes,
                               int time_forecast, int dimension,
                               float loc_precision_err) {
    PredictResult presult;
    float abnormal_penalty = 1000.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float time_penalty = abnormal_penalty / (float)(time_forecast + 1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float cutting_threshold = 2.0f * abnormal_penalty; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float initial_cost = cutting_threshold - 100.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Already computed or single node? // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto cost_it = trajectories_costs.find(next_path);
    if ((cost_it != trajectories_costs.end() && cost_it->second != UNCOMPUTED) || next_path.size() <= 1) {
        presult.terminal = -1; // None equivalent
        return presult;
    }

    // Terminal check // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool terminal = is_terminal_node(next_path.back(), locs, jump_threshold, selected_graph, final_graph_nodes);
    presult.terminal = terminal ? 1 : 0;

    // Check for excessive time gaps // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int idx = 1; idx < (int)next_path.size() - 1; idx++) {
        if ((next_path[idx + 1].first - next_path[idx].first) - 1 > time_forecast) {
            trajectories_costs[next_path] = initial_cost;
            presult.ab_index.push_back(idx);
            return presult;
        }
    }

    if (next_path.size() <= 1) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Should not happen
        trajectories_costs[next_path] = initial_cost;
        return presult;
    } else if (next_path.size() == 2) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        trajectories_costs[next_path] = time_penalty;
    } else if (next_path.size() == 3) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        Node before_node = next_path[1];
        Node next_node = next_path[2];
        int time_gap = next_node.first - before_node.first - 1;
        std::array<float,3> dummy = {0.15f, 0.15f, 0.0f};
        float log_p0 = empirical_pdf_lookup(dummy);
        float time_score = time_gap * time_penalty;
        trajectories_costs[next_path] = time_score + std::abs(log_p0 - 5.0f);
    } else { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // len >= 4
        int last_idx = terminal ? -1 : -2; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int path_len = (int)next_path.size();

        // First edge cost (empirical PDF) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> traj_cost;
        std::vector<int> ab_index;
        Node before_node = next_path[1];
        Node next_node = next_path[2];
        auto next_coord = locs.at(next_node.first)[next_node.second]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto cur_coord = locs.at(before_node.first)[before_node.second]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::array<float,3> input_mu = {next_coord[0] - cur_coord[0],
                                        next_coord[1] - cur_coord[1],
                                        next_coord[2] - cur_coord[2]};
        float log_p0 = empirical_pdf_lookup(input_mu); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        traj_cost.push_back(std::abs(log_p0 - 5.0f));

        // Cauchy cost for subsequent edges // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int end_idx = path_len + last_idx - 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> tmpx, tmpy; // for k re-estimation after abnormal
        Node j_prev_node, j_next_node; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        for (int edge_i = 1; edge_i < end_idx; edge_i++) {
            int edge_j = edge_i + 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            // Re-estimate k after abnormal detections (fixed mode: k=0.5) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (!ab_index.empty()) {
                prev_alpha = 1.0f;
                prev_k = 0.5f; // predict_ks returns 0.5 when TF=False
            }

            Node i_prev_node = next_path[edge_i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            Node i_next_node = next_path[edge_i + 1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            j_prev_node = next_path[edge_j];
            j_next_node = next_path[edge_j + 1];
            int i_time_gap = i_next_node.first - i_prev_node.first - 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            int j_time_gap = j_next_node.first - j_prev_node.first - 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            auto loc_i_next = locs.at(i_next_node.first)[i_next_node.second]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto loc_i_prev = locs.at(i_prev_node.first)[i_prev_node.second];
            auto loc_j_next = locs.at(j_next_node.first)[j_next_node.second];
            auto loc_j_prev = locs.at(j_prev_node.first)[j_prev_node.second];

            std::array<float,3> vec_i = {loc_i_next[0] - loc_i_prev[0], // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                         loc_i_next[1] - loc_i_prev[1],
                                         loc_i_next[2] - loc_i_prev[2]};
            std::array<float,3> vec_j = {loc_j_next[0] - loc_j_prev[0],
                                         loc_j_next[1] - loc_j_prev[1],
                                         loc_j_next[2] - loc_j_prev[2]};

            auto cauchy = predict_cauchy_tracking(vec_j, vec_i, prev_k, prev_alpha, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                                  i_time_gap, j_time_gap, loc_precision_err, dimension);
            if (cauchy.abnormal) ab_index.push_back(edge_j); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            traj_cost.push_back(std::abs(cauchy.log_pdf - 5.0f));
        }

        // Compute time gaps penalty // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int time_gaps = 0;
        for (int ni = 1; ni < (int)next_path.size() - 1; ni++) {
            time_gaps += (next_path[ni + 1].first - next_path[ni].first) - 1;
        }

        // Sort and deduplicate ab_index // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::sort(ab_index.begin(), ab_index.end());
        ab_index.erase(std::unique(ab_index.begin(), ab_index.end()), ab_index.end());

        float abnormal_jump_score = abnormal_penalty * (float)ab_index.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        float time_score = (float)time_gaps * time_penalty;

        float final_score; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (traj_cost.size() > 1) {
            float mean_cost = 0;
            for (float c : traj_cost) mean_cost += c;
            mean_cost /= traj_cost.size();
            final_score = mean_cost + abnormal_jump_score + time_score;
        } else {
            final_score = time_score + traj_cost[0];
        }
        trajectories_costs[next_path] = final_score;

        if (final_score > cutting_threshold) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            presult.ab_index = ab_index;
        }
    }

    return presult;
}

// ============================================================
// Helper: run one pass of greedy assignment on a subgraph
// Used by both the NB_TO_OPTIMUM probing passes and the final pass
// ============================================================

static void greedy_assign_pass( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    DiGraph& sub_graph,
    DiGraph& out_graph,
    const Node& source_node,
    const std::set<Node>& final_graph_node_set_hashed,
    const Localizations& locs,
    const std::vector<int>& next_times,
    const std::map<int, float>& distribution,
    bool first_step,
    int last_time,
    const TrackingConfig& config,
    const std::vector<Path>* prev_paths_ptr,
    const std::map<Path, float>* alpha_values_ptr,
    const std::map<Path, float>* k_values_ptr,
    std::map<Path, int>& hashed_prev_next,
    int initial_pick_idx,
    float& cost_sum)
{
    int nb_to_optimum = 1 << config.graph_depth; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    (void)nb_to_optimum;

    std::map<Path, float> trajectories_costs; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Initialize all paths with UNCOMPUTED
    {
        Path path_buf = {source_node};
        std::set<Node> seen_init;
        std::vector<Path> init_paths;
        find_paths_dfs(sub_graph, source_node, path_buf, seen_init, init_paths);
        for (auto& p : init_paths) {
            if (p.size() > 1) trajectories_costs[p] = UNCOMPUTED;
        }
    }

    std::map<Path, std::vector<int>> ab_indice; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::map<Path, bool> is_terminals; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::map<Path, int> start_indice_map;
    bool initial_ = true;
    Path prev_lowest = {source_node};

    while (true) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Rebuild cost_copy (prune stale entries) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::map<Path, float> cost_copy;
        auto next_paths = find_paths_as_list(sub_graph, source_node);
        if (next_paths.empty()) break;

        if (initial_ && initial_pick_idx >= (int)next_paths.size()) break; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        for (auto& np : next_paths) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            // Count how many nodes are in final_graph
            int index_ind = 0;
            for (const auto& n : np) {
                if (final_graph_node_set_hashed.count(n)) index_ind++;
            }
            start_indice_map[np] = index_ind;
            if (trajectories_costs.count(np)) cost_copy[np] = trajectories_costs[np];
        }
        trajectories_costs = cost_copy;

        // Compute costs // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (auto& np : next_paths) {
            const Path* pp = nullptr;
            float use_alpha = config.init_alpha; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            float use_k = config.init_k; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            if (!first_step && prev_paths_ptr) {
                pp = match_prev_next(*prev_paths_ptr, np, hashed_prev_next);
                if (pp && alpha_values_ptr && k_values_ptr) {
                    auto ait = alpha_values_ptr->find(*pp);
                    auto kit = k_values_ptr->find(*pp);
                    if (ait != alpha_values_ptr->end()) use_alpha = ait->second;
                    if (kit != k_values_ptr->end()) use_k = kit->second;
                }
            }

            auto pr = predict_long_seq(np, trajectories_costs, locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                       use_alpha, use_k, next_times, pp,
                                       start_indice_map, last_time,
                                       distribution.at(0), out_graph,
                                       final_graph_node_set_hashed,
                                       config.graph_depth, config.dimension,
                                       config.loc_precision_err);
            if (pr.terminal >= 0) is_terminals[np] = (pr.terminal == 1);
            if (!pr.ab_index.empty()) ab_indice[np] = pr.ab_index;
        }

        // Sort by cost // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::pair<float, int>> cost_idx_vec;
        for (int i = 0; i < (int)next_paths.size(); i++) {
            float c = trajectories_costs.count(next_paths[i]) ? trajectories_costs[next_paths[i]] : 1e9f;
            if (c == UNCOMPUTED) c = 1e9f;
            cost_idx_vec.push_back({c, i});
        }
        std::sort(cost_idx_vec.begin(), cost_idx_vec.end());

        Path lowest_cost_traj; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (initial_) {
            if (initial_pick_idx < (int)cost_idx_vec.size()) {
                lowest_cost_traj = next_paths[cost_idx_vec[initial_pick_idx].second];
            } else {
                break;
            }
            initial_ = false;
        } else {
            lowest_cost_traj = next_paths[cost_idx_vec[0].second];
        }

        // Abnormal trajectory cutting // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (ab_indice.count(lowest_cost_traj) && !ab_indice[lowest_cost_traj].empty()) {
            int ab_i = ab_indice[lowest_cost_traj][0]; // only first // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (sub_graph.has_edge(lowest_cost_traj[ab_i], lowest_cost_traj[ab_i + 1])) {
                sub_graph.remove_edge(lowest_cost_traj[ab_i], lowest_cost_traj[ab_i + 1]);
            }
            if (!sub_graph.has_edge(source_node, lowest_cost_traj[ab_i + 1])) {
                sub_graph.add_edge(source_node, lowest_cost_traj[ab_i + 1]);
            }
            // Add split paths to costs // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            Path added1 = {source_node};
            for (int k = ab_i + 1; k < (int)lowest_cost_traj.size(); k++) added1.push_back(lowest_cost_traj[k]);
            trajectories_costs[added1] = UNCOMPUTED;
            Path added2 = {source_node};
            for (int k = 1; k <= ab_i; k++) added2.push_back(lowest_cost_traj[k]);
            trajectories_costs[added2] = UNCOMPUTED;

            // Register new paths // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto new_paths = find_paths_as_list(sub_graph, source_node);
            for (auto& np : new_paths) {
                if (!trajectories_costs.count(np)) trajectories_costs[np] = UNCOMPUTED;
            }
            continue; // retry with modified graph
        }

        // Prune graph: remove selected nodes, add bypass edges // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        while (true) {
            size_t before = sub_graph.size();
            for (size_t ri = 1; ri < lowest_cost_traj.size(); ri++) {
                Node rm_node = lowest_cost_traj[ri];
                auto preds = sub_graph.predecessors(rm_node);
                auto succs = sub_graph.successors(rm_node);
                for (const auto& pred : preds) {
                    for (const auto& suc : succs) {
                        if (!final_graph_node_set_hashed.count(pred) &&
                            !final_graph_node_set_hashed.count(suc) &&
                            pred != source_node &&
                            !sub_graph.has_edge(pred, suc)) {
                            auto pred_loc = locs.at(pred.first)[pred.second]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                            auto suc_loc = locs.at(suc.first)[suc.second];
                            float jump_d = euclidean_displacement_single(pred_loc, suc_loc);
                            int time_gap = suc.first - pred.first - 1;
                            auto dit = distribution.find(time_gap);
                            if (dit != distribution.end() && jump_d < dit->second) {
                                EdgeData ed; ed.jump_d = jump_d;
                                sub_graph.add_edge(pred, suc, ed);
                            }
                        }
                    }
                }
            }
            if (sub_graph.size() == before) break;
        }

        // Remove nodes of selected trajectory // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        bool term = is_terminals.count(lowest_cost_traj) ? is_terminals[lowest_cost_traj] : true;
        if (term) {
            for (size_t i = 1; i < lowest_cost_traj.size(); i++) {
                sub_graph.remove_node(lowest_cost_traj[i]);
            }
        } else {
            if (lowest_cost_traj.size() == 2) {
                sub_graph.remove_node(lowest_cost_traj.back());
            } else {
                for (size_t i = 1; i + 1 < lowest_cost_traj.size(); i++) {
                    sub_graph.remove_node(lowest_cost_traj[i]);
                }
            }
        }

        // Reconnect orphaned nodes to source // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& sn : std::set<Node>(sub_graph.get_nodes())) {
            if (sn != source_node && !sub_graph.has_path(source_node, sn)) {
                sub_graph.add_edge(source_node, sn);
            }
        }

        float pop_cost = trajectories_costs.count(lowest_cost_traj) ? trajectories_costs[lowest_cost_traj] : 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        cost_sum += pop_cost;

        // Update output graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (size_t ei = 1; ei < lowest_cost_traj.size(); ei++) {
            EdgeData ed;
            ed.terminal = term;
            out_graph.add_edge(lowest_cost_traj[ei - 1], lowest_cost_traj[ei], ed);
        }

        // Check termination // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto src_neighbors = sub_graph.successors(source_node);
        if (src_neighbors.empty() || lowest_cost_traj == prev_lowest) break;

        // Register new paths // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        {
            Path pb = {source_node};
            std::set<Node> si;
            std::vector<Path> new_paths;
            find_paths_dfs(sub_graph, source_node, pb, si, new_paths);
            for (auto& np : new_paths) {
                if (np.size() > 1 && !trajectories_costs.count(np)) {
                    trajectories_costs[np] = UNCOMPUTED;
                }
            }
        }

        prev_lowest = lowest_cost_traj;
    }
}

// ============================================================
// select_opt_graph2: optimal graph selection
// ============================================================

SelectResult select_opt_graph2(const std::set<Node>& final_graph_node_set_hashed, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                               const DiGraph& saved_graph,
                               DiGraph next_graph,
                               const Localizations& locs,
                               const std::vector<int>& next_times,
                               const std::map<int, float>& distribution,
                               bool first_step, int last_time,
                               const TrackingConfig& config) {
    Node source_node = {0, 0}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    DiGraph selected_graph;
    selected_graph.add_node(source_node);
    int nb_to_optimum = 1 << config.graph_depth; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Get prev paths and their alpha/k values (all fixed) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Path> prev_paths;
    std::map<Path, float> alpha_values, k_values;
    std::map<Path, int> hashed_prev_next;

    if (!first_step) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        prev_paths = find_paths_as_list(saved_graph, source_node);
        for (auto& pp : prev_paths) {
            if (config.use_nn && g_nn_models.loaded) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                // Extract trajectory positions (last ALPHA_MAX_LENGTH=10 points) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                std::vector<float> xs, ys; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                int start = std::max(1, (int)pp.size() - 10); // skip source node // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (int i = start; i < (int)pp.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    auto lit = locs.find(pp[i].first); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    if (lit != locs.end() && pp[i].second < (int)lit->second.size()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        xs.push_back(lit->second[pp[i].second][0]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        ys.push_back(lit->second[pp[i].second][1]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    }
                }
                alpha_values[pp] = predict_alpha_nn(g_nn_models, xs, ys); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                k_values[pp] = predict_k_nn(g_nn_models, xs, ys); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            } else { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                alpha_values[pp] = config.init_alpha;
                k_values[pp] = config.init_k;
            }
        }
    }

    // Generate next graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto gen_result = generate_next_paths(next_graph, final_graph_node_set_hashed,
                                          locs, next_times, distribution, source_node);
    next_graph = gen_result.graph;
    auto& last_nodes = gen_result.last_nodes;

    // Split into connected subgraphs // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto subgraphs = split_to_subgraphs(next_graph, source_node);

    for (auto& sub_graph_ : subgraphs) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Phase 1: try NB_TO_OPTIMUM alternatives // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<float> cost_sums(nb_to_optimum, -1e-5f);

        for (int lowest_idx = 0; lowest_idx < nb_to_optimum; lowest_idx++) {
            DiGraph sub_copy = sub_graph_.copy();
            DiGraph tmp_graph;
            tmp_graph.add_node(source_node);
            std::map<Path, int> hpn_copy = hashed_prev_next;

            greedy_assign_pass(sub_copy, tmp_graph, source_node, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                              final_graph_node_set_hashed, locs, next_times,
                              distribution, first_step, last_time, config,
                              first_step ? nullptr : &prev_paths,
                              first_step ? nullptr : &alpha_values,
                              first_step ? nullptr : &k_values,
                              hpn_copy, lowest_idx, cost_sums[lowest_idx]);
        }

        // Find best starting index // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (auto& cs : cost_sums) { if (cs < 0) cs = 99999.0f; }
        int lowest_cost_idx = 0;
        for (int i = 1; i < (int)cost_sums.size(); i++) {
            if (cost_sums[i] < cost_sums[lowest_cost_idx]) lowest_cost_idx = i;
        }

        // Phase 2: rebuild with best starting index into selected_graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        DiGraph sub_copy2 = sub_graph_.copy();
        std::map<Path, int> hpn2 = hashed_prev_next;
        float dummy_cost = -1e-5f;
        greedy_assign_pass(sub_copy2, selected_graph, source_node, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                          final_graph_node_set_hashed, locs, next_times,
                          distribution, first_step, last_time, config,
                          first_step ? nullptr : &prev_paths,
                          first_step ? nullptr : &alpha_values,
                          first_step ? nullptr : &k_values,
                          hpn2, lowest_cost_idx, dummy_cost);
    }

    // Check for orphan nodes // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> orphans;
    for (int time : next_times) {
        if (!locs.count(time) || locs.at(time).empty()) continue;
        for (int idx = 0; idx < (int)locs.at(time).size(); idx++) {
            Node cur_node = {time, idx};
            if (!selected_graph.has_node(cur_node) && !final_graph_node_set_hashed.count(cur_node)) {
                orphans.push_back(cur_node);
            }
        }
    }

    return {selected_graph, !orphans.empty()}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// ============================================================
// forecast: main tracking loop
// ============================================================

std::vector<TrajectoryObj> forecast(const Localizations& locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                   const std::vector<int>& t_avail_steps,
                                   const std::map<int, float>& distribution,
                                   int image_length,
                                   const TrackingConfig& config) {
    bool first_construction = true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int last_time = image_length;
    Node source_node = {0, 0};
    int time_forecast = std::max(1, std::min(5, config.graph_depth)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    DiGraph final_graph; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    DiGraph light_prev_graph;
    DiGraph next_graph;
    final_graph.add_node(source_node);
    light_prev_graph.add_node(source_node);
    next_graph.add_node(source_node);

    // Initialize next_graph with edges to first time step // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (!t_avail_steps.empty() && locs.count(t_avail_steps[0])) {
        for (int idx = 0; idx < (int)locs.at(t_avail_steps[0]).size(); idx++) {
            EdgeData ed; ed.jump_d = -1;
            next_graph.add_edge(source_node, {t_avail_steps[0], idx}, ed);
        }
    }

    // Selected time steps // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> selected_time_steps;
    if (!t_avail_steps.empty()) {
        int end_t = std::min(t_avail_steps[0] + 1 + time_forecast, t_avail_steps.back() + 1);
        for (int t = t_avail_steps[0]; t < end_t; t++) selected_time_steps.push_back(t);
    }

    std::set<Node> final_graph_node_set_hashed = {source_node}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int saved_time_steps = 1;

    while (true) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::vector<Node>> node_pairs;
        int start_time = selected_time_steps.empty() ? last_time : selected_time_steps.back();

        if (config.verbose && !selected_time_steps.empty()) {
            std::cerr << "\rTracking frame " << selected_time_steps[0] << "-" << selected_time_steps.back()
                      << " / " << last_time << std::flush;
        }

        // Check if selected time steps intersect with available steps // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        bool has_intersection = false;
        for (int t : selected_time_steps) {
            for (int ta : t_avail_steps) {
                if (t == ta) { has_intersection = true; break; }
            }
            if (has_intersection) break;
        }

        DiGraph selected_sub_graph; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        bool has_orphan = false;
        selected_sub_graph.add_node(source_node);

        if (has_intersection) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto sr = select_opt_graph2(final_graph_node_set_hashed, light_prev_graph,
                                        next_graph, locs, selected_time_steps,
                                        distribution, first_construction, last_time, config);
            selected_sub_graph = sr.selected_graph;
            has_orphan = sr.has_orphan;
        }

        first_construction = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        light_prev_graph = DiGraph();
        light_prev_graph.add_node(source_node);

        auto selected_paths = find_paths_as_list(selected_sub_graph, source_node); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Check if we're at the end // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        bool last_time_in_sel = false;
        for (int t : selected_time_steps) { if (t == last_time) { last_time_in_sel = true; break; } }

        if (selected_sub_graph.size() <= 1 && last_time_in_sel) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            break;
        }

        if (last_time_in_sel && !has_orphan) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            // Final: add all remaining paths to final_graph
            for (const auto& path : selected_paths) {
                auto without_source = std::vector<Node>(path.begin() + 1, path.end());
                if (without_source.size() == 1) {
                    if (!final_graph_node_set_hashed.count(without_source[0])) {
                        final_graph.add_edge(source_node, without_source[0]);
                    }
                } else {
                    for (size_t i = 0; i + 1 < without_source.size(); i++) {
                        if (!final_graph_node_set_hashed.count(without_source[i + 1])) {
                            final_graph.add_edge(without_source[i], without_source[i + 1]);
                        }
                    }
                }
                if (!final_graph.has_path(source_node, without_source[0])) {
                    final_graph.add_edge(source_node, without_source[0]);
                }
            }
            break;
        }

        // Process selected paths // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& path : selected_paths) {
            if (path.size() < 2) continue;
            bool terminal = selected_sub_graph.get_edge_data(path[0], path[1]).terminal; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            if (path.size() == 2) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (terminal && !final_graph_node_set_hashed.count(path[1])) {
                    final_graph.add_edge(source_node, path[1]);
                    final_graph_node_set_hashed.insert(path[1]);
                } else {
                    start_time = std::min(start_time, path[1].first);
                }
            } else { // path.size() >= 3 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (!terminal) {
                    start_time = std::min(start_time, path[(int)path.size() - 2].first); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

                    if (path.size() == 3) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        Node before_node = path[1];
                        if (!final_graph_node_set_hashed.count(before_node)) {
                            final_graph.add_edge(source_node, before_node);
                            final_graph_node_set_hashed.insert(before_node);
                        }
                        node_pairs.push_back({path[1]});
                    } else { // path.size() > 3, not terminal // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        Node first_node = path[1];
                        if (final_graph_node_set_hashed.count(first_node)) {
                            for (size_t ei = 2; ei + 1 < path.size(); ei++) {
                                final_graph.add_edge(path[ei - 1], path[ei]);
                                final_graph_node_set_hashed.insert(path[ei]);
                            }
                        } else {
                            final_graph.add_edge(source_node, first_node);
                            final_graph_node_set_hashed.insert(first_node);
                            for (size_t ei = 2; ei + 1 < path.size(); ei++) {
                                final_graph.add_edge(path[ei - 1], path[ei]);
                                final_graph_node_set_hashed.insert(path[ei]);
                            }
                        }

                        node_pairs.push_back({path[path.size() - 3], path[path.size() - 2]}); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

                        // Build light_prev_graph from ancestors // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        int alpha_max_length = 10;
                        auto ancestors = final_graph.ancestors(path[path.size() - 2]);
                        std::sort(ancestors.begin(), ancestors.end(), [](const Node& a, const Node& b) {
                            return a.first > b.first; // descending by time
                        });
                        int limit = std::min((int)ancestors.size(), alpha_max_length + 3);
                        for (int i = 0; i + 1 < limit; i++) {
                            light_prev_graph.add_edge(ancestors[i + 1], ancestors[i]);
                        }
                        if (limit > 1 && ancestors[limit - 1] != source_node) {
                            light_prev_graph.add_edge(source_node, ancestors[limit - 1]);
                        }
                    }
                } else { // terminal, path.size() >= 3 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    Node first_node = path[1];
                    Node second_node = path[2];
                    if (final_graph_node_set_hashed.count(first_node)) {
                        final_graph.add_edge(first_node, second_node);
                        final_graph_node_set_hashed.insert(second_node);
                        for (size_t ei = 3; ei < path.size(); ei++) {
                            final_graph.add_edge(path[ei - 1], path[ei]);
                            final_graph_node_set_hashed.insert(path[ei]);
                        }
                    } else {
                        final_graph.add_edge(source_node, first_node);
                        final_graph.add_edge(first_node, second_node);
                        final_graph_node_set_hashed.insert(first_node);
                        final_graph_node_set_hashed.insert(second_node);
                        for (size_t ei = 3; ei < path.size(); ei++) {
                            final_graph.add_edge(path[ei - 1], path[ei]);
                            final_graph_node_set_hashed.insert(path[ei]);
                        }
                    }
                }
            }
        }

        // Check for unassigned nodes in selected_time_steps // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int time : selected_time_steps) {
            if (!locs.count(time) || locs.at(time).empty()) continue;
            for (int idx = 0; idx < (int)locs.at(time).size(); idx++) {
                Node node = {time, idx};
                if (!final_graph_node_set_hashed.count(node)) {
                    start_time = std::min(start_time, node.first);
                }
            }
        }

        // Advance time window // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        saved_time_steps = selected_time_steps.back();
        int next_first_time = selected_time_steps.back() + 1;
        next_graph = DiGraph();
        next_graph.add_node(source_node);

        // Build new selected_time_steps // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        selected_time_steps.clear();
        int end_t = std::min(last_time + 1, next_first_time + time_forecast);
        for (int t = start_time; t < end_t; t++) selected_time_steps.push_back(t);

        // Carry over node_pairs into next_graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& np : node_pairs) {
            if (np.size() == 1) {
                EdgeData ed; ed.jump_d = -1;
                next_graph.add_edge(source_node, np[0], ed);
            } else if (np.size() >= 2) {
                EdgeData ed; ed.jump_d = -1;
                next_graph.add_edge(source_node, np[0], ed);
                auto last_xyz = locs.at(np.back().first)[np.back().second];
                auto second_last_xyz = locs.at(np[0].first)[np[0].second];
                float jd = std::sqrt(
                    (last_xyz[0] - second_last_xyz[0]) * (last_xyz[0] - second_last_xyz[0]) +
                    (last_xyz[1] - second_last_xyz[1]) * (last_xyz[1] - second_last_xyz[1])
                );
                EdgeData ed2; ed2.jump_d = jd;
                next_graph.add_edge(np[0], np.back(), ed2);
            }
        }
    }

    if (config.verbose) std::cerr << "\rTracking complete.                    " << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Extract trajectories from final_graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<TrajectoryObj> trajectory_list;
    auto all_final_paths = find_paths_as_list(final_graph, source_node);
    int traj_idx = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& path : all_final_paths) {
        if ((int)path.size() >= config.cutoff + 1) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            TrajectoryObj traj;
            traj.index = traj_idx;
            for (size_t i = 1; i < path.size(); i++) {
                traj.add_trajectory_tuple(path[i].first, path[i].second);
            }
            trajectory_list.push_back(traj);
            traj_idx++;
        }
    }
    return trajectory_list;
}

// ============================================================
// Trajectory visualization (make_trajectory_image)
// ============================================================

#ifdef USE_LIBPNG
#include <png.h>
#endif

// Bresenham line drawing on RGB buffer // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static void draw_line(std::vector<uint8_t>& img, int rows, int cols, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                      int x0, int y0, int x1, int y1,
                      uint8_t r, uint8_t g, uint8_t b) {
    int dx = std::abs(x1 - x0), dy = std::abs(y1 - y0); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int sx = (x0 < x1) ? 1 : -1, sy = (y0 < y1) ? 1 : -1;
    int err = dx - dy;
    while (true) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (x0 >= 0 && x0 < cols && y0 >= 0 && y0 < rows) {
            int idx = (y0 * cols + x0) * 3; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            img[idx] = r; img[idx + 1] = g; img[idx + 2] = b;
        }
        if (x0 == x1 && y0 == y1) break;
        int e2 = 2 * err; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (e2 > -dy) { err -= dy; x0 += sx; }
        if (e2 < dx) { err += dx; y0 += sy; }
    }
}

// Deterministic color from trajectory index (matches Python's np.random.default_rng(idx)) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Uses splitmix64 hash for simplicity — visually similar random colors
static std::array<uint8_t, 3> traj_color(int idx) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    uint64_t x = (uint64_t)idx;
    x += 0x9e3779b97f4a7c15ULL; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return {(uint8_t)(x & 0xFF), (uint8_t)((x >> 8) & 0xFF), (uint8_t)((x >> 16) & 0xFF)}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// 2D Spherical GMM with k-means++ initialization
// ============================================================

struct GMM2DResult { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> weights; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<std::array<double,2>> means; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> variances; // spherical: single variance per component // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> labels; // predicted labels for each point // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};

static double dist2d(double x1, double y1, double x2, double y2) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double dx = x1 - x2, dy = y1 - y2; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return std::sqrt(dx*dx + dy*dy); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// k-means++ initialization for 2D data // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static std::vector<std::array<double,2>> kmeans_pp_init(const std::vector<std::array<double,2>>& data, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                                         int k, std::mt19937& rng) {
    int n = (int)data.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<std::array<double,2>> centers; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::uniform_int_distribution<int> uid(0, n-1); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    centers.push_back(data[uid(rng)]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    std::vector<double> min_dist(n, 1e30); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int c = 1; c < k; c++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double total = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double d = dist2d(data[i][0], data[i][1], centers.back()[0], centers.back()[1]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            min_dist[i] = std::min(min_dist[i], d * d); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            total += min_dist[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
        std::uniform_real_distribution<double> urd(0, total); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double r = urd(rng); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double cumul = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int chosen = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            cumul += min_dist[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (cumul >= r) { chosen = i; break; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
        centers.push_back(data[chosen]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }
    return centers; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// Fit 2D spherical GMM with k-means++ init // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static GMM2DResult fit_gmm_2d_spherical(const std::vector<std::array<double,2>>& data, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                         int n_comp, int max_iter = 1000) {
    int n = (int)data.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (n == 0) return {{}, {}, {}, {}}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    std::mt19937 rng(42); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto centers = kmeans_pp_init(data, n_comp, rng); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    std::vector<double> weights(n_comp, 1.0 / n_comp); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<std::array<double,2>> means = centers; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Initialize variance from data spread // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double data_var = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double mx = 0, my = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& p : data) { mx += p[0]; my += p[1]; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    mx /= n; my /= n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& p : data) { data_var += (p[0]-mx)*(p[0]-mx) + (p[1]-my)*(p[1]-my); } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    data_var /= (2.0 * n); // spherical: average per dimension // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (data_var < 1e-10) data_var = 1e-10; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> vars(n_comp, data_var); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    std::vector<std::vector<double>> resp(n, std::vector<double>(n_comp)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    for (int iter = 0; iter < max_iter; iter++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double prev_ll = -1e30; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // E-step // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double ll = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double total = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int k = 0; k < n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double dx = data[i][0] - means[k][0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double dy = data[i][1] - means[k][1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double exponent = -0.5 * (dx*dx + dy*dy) / vars[k]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                resp[i][k] = weights[k] * std::exp(exponent) / (2.0 * M_PI * vars[k]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                total += resp[i][k]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
            if (total < 1e-300) total = 1e-300; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            ll += std::log(total); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int k = 0; k < n_comp; k++) resp[i][k] /= total; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }

        if (std::abs(ll - prev_ll) < 1e-6) break; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        prev_ll = ll; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // M-step // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int k = 0; k < n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double nk = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int i = 0; i < n; i++) nk += resp[i][k]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (nk < 1e-10) nk = 1e-10; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            weights[k] = nk / n; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            double sx = 0, sy = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                sx += resp[i][k] * data[i][0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                sy += resp[i][k] * data[i][1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
            means[k][0] = sx / nk; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            means[k][1] = sy / nk; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            // Spherical variance: average of both dimensions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double sv = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double dx = data[i][0] - means[k][0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double dy = data[i][1] - means[k][1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                sv += resp[i][k] * (dx*dx + dy*dy); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
            vars[k] = sv / (2.0 * nk); // divide by 2 for spherical (d dimensions) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (vars[k] < 1e-10) vars[k] = 1e-10; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }

    // Predict labels // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> labels(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (int i = 0; i < n; i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int best_k = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double best_r = resp[i][0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int k = 1; k < n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (resp[i][k] > best_r) { best_r = resp[i][k]; best_k = k; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
        labels[i] = best_k; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    return {weights, means, vars, labels}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// PostProcessing — trajectory splitting at false state transitions
// ============================================================

std::vector<TrajectoryObj> post_processing(const std::vector<TrajectoryObj>& trajectory_list, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                            const Localizations& locs, int cutoff, bool verbose) {
    // Helper: extract sorted positions and frames for a trajectory // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto extract_positions = [&](const TrajectoryObj& traj, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                  std::vector<double>& xs, std::vector<double>& ys,
                                  std::vector<int>& frames) {
        struct Entry { int frame; double x, y; }; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<Entry> entries; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& [f, idx] : traj.tuples) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto it = locs.find(f); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (it != locs.end() && idx < (int)it->second.size()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                entries.push_back({f, (double)it->second[idx][0], (double)it->second[idx][1]}); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
        std::sort(entries.begin(), entries.end(), [](const Entry& a, const Entry& b) { return a.frame < b.frame; }); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        xs.clear(); ys.clear(); frames.clear(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& e : entries) { xs.push_back(e.x); ys.push_back(e.y); frames.push_back(e.frame); } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    };

    // Helper: GMM clustering + label merging (shared by both passes) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto cluster_trajectory = [](const std::vector<double>& xs, const std::vector<double>& ys, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                  int n_comp) -> std::pair<GMM2DResult, std::vector<int>> {
        int n = (int)xs.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::array<double,2>> pos(n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i < n; i++) pos[i] = {xs[i], ys[i]}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        auto gm = fit_gmm_2d_spherical(pos, n_comp, 1000); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto labels = gm.labels; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Merge nearby clusters (distance < 1.5 px) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::map<int, int> change_label; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int a = 0; a < n_comp; a++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int b = a + 1; b < n_comp; b++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double d = dist2d(gm.means[a][0], gm.means[a][1], gm.means[b][0], gm.means[b][1]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (d < 1.5) change_label[b] = a; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
        // Apply relabeling in reverse key order // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (auto it = change_label.rbegin(); it != change_label.rend(); ++it) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (auto& lb : labels) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (lb == it->first) lb = it->second; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
        return {gm, labels}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    };

    // ---- Pass 1: compute gap distribution and seuil ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> gap_distrib; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& traj : trajectory_list) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (traj.get_trajectory_length() < cutoff) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> xs, ys; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<int> frames; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        extract_positions(traj, xs, ys, frames); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if ((int)xs.size() < 5) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        int n_comp = ((int)xs.size() <= 10) ? 2 : ((int)xs.size() < 100 ? 3 : 4); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto [gm, labels] = cluster_trajectory(xs, ys, n_comp); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Compute mean displacement per cluster, take max gap // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::set<int> unique_labels(labels.begin(), labels.end()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double max_gap = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int lb : unique_labels) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double sum_disp = 0; int count = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int i = 0; i < (int)labels.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (labels[i] == lb) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    double dx = xs[i] - gm.means[lb][0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    double dy = ys[i] - gm.means[lb][1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    sum_disp += std::sqrt(dx*dx + dy*dy); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    count++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                }
            }
            if (count > 0) max_gap = std::max(max_gap, sum_disp / count); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
        gap_distrib.push_back(max_gap); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    // Compute seuil (threshold for splitting) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    double seuil = 0.0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (!gap_distrib.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int below_1 = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (double g : gap_distrib) if (g < 1.0) below_1++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if ((double)below_1 / gap_distrib.size() > 0.8) seuil = 1.0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    // ---- Pass 2: split trajectories at false state transitions ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<TrajectoryObj> filtered; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int traj_index = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int post_processed_count = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    for (const auto& traj : trajectory_list) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (traj.get_trajectory_length() < cutoff) continue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> xs, ys; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<int> frames; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        extract_positions(traj, xs, ys, frames); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        // Find original (frame, idx) mapping sorted by frame // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        struct TupleEntry { int frame; int idx; }; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<TupleEntry> sorted_tuples; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& [f, idx] : traj.tuples) sorted_tuples.push_back({f, idx}); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::sort(sorted_tuples.begin(), sorted_tuples.end(), [](const TupleEntry& a, const TupleEntry& b) { return a.frame < b.frame; }); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        std::set<std::pair<int,int>> predicted_false_pairs; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

        if ((int)xs.size() >= 5) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            int n_comp = ((int)xs.size() <= 10) ? 2 : ((int)xs.size() < 100 ? 3 : 4); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto [gm, labels] = cluster_trajectory(xs, ys, n_comp); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            // Compute overall gap (mean displacement to cluster centers) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            double gap_sum = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (int i = 0; i < (int)labels.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                int lb = labels[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double dx = xs[i] - gm.means[lb][0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                double dy = ys[i] - gm.means[lb][1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                gap_sum += std::sqrt(dx*dx + dy*dy); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
            double gap = gap_sum / labels.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            // Collect unique label pairs // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            std::set<int> unique_labels(labels.begin(), labels.end()); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            std::vector<std::pair<int,int>> pairs; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (auto it1 = unique_labels.begin(); it1 != unique_labels.end(); ++it1) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (auto it2 = std::next(it1); it2 != unique_labels.end(); ++it2) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    pairs.push_back({*it1, *it2}); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                }
            }

            // If gap < seuil, all pairs are false transitions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (gap < seuil) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                for (const auto& p : pairs) predicted_false_pairs.insert(p); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }

        if (!predicted_false_pairs.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            // Need labels from pass 2 clustering // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            int n_comp = ((int)xs.size() <= 10) ? 2 : ((int)xs.size() < 100 ? 3 : 4); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto [gm2, labels2] = cluster_trajectory(xs, ys, n_comp); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            post_processed_count++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            int prev_label = labels2[0]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            TrajectoryObj new_traj; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            new_traj.index = traj_index; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            new_traj.add_trajectory_tuple(sorted_tuples[0].frame, sorted_tuples[0].idx); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

            for (int i = 1; i < (int)labels2.size(); i++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                int cur_label = labels2[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                int a = std::min(prev_label, cur_label); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                int b = std::max(prev_label, cur_label); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (predicted_false_pairs.count({a, b})) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    if (new_traj.get_trajectory_length() >= cutoff) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        filtered.push_back(new_traj); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                        traj_index++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    }
                    new_traj = TrajectoryObj(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    new_traj.index = traj_index; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                }
                new_traj.add_trajectory_tuple(sorted_tuples[i].frame, sorted_tuples[i].idx); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                prev_label = cur_label; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
            if (new_traj.get_trajectory_length() >= cutoff) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                filtered.push_back(new_traj); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                traj_index++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        } else { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            TrajectoryObj new_traj; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            new_traj.index = traj_index; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            for (const auto& te : sorted_tuples) new_traj.add_trajectory_tuple(te.frame, te.idx); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (new_traj.get_trajectory_length() >= cutoff) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                filtered.push_back(new_traj); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                traj_index++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
    }

    if (verbose && !gap_distrib.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int below_1 = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (double g : gap_distrib) if (g < 1.0) below_1++; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cerr << "Post processed nb of trajectories: " << post_processed_count // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                  << " with ratio: " << (double)below_1 / gap_distrib.size()
                  << ", seuil: " << seuil << std::endl;
    }
    return filtered; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

void make_trajectory_image(const std::string& output_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                           const std::vector<TrajectoryObj>& trajectories,
                           const Localizations& locs,
                           int img_rows, int img_cols) {
#ifdef USE_LIBPNG
    // Compute upscaling factor (matching Python: if < 1024*1024, upscale) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int upscale = 1;
    if (img_rows * img_cols < 1024 * 1024) {
        upscale = 1024 / img_rows; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (upscale < 1) upscale = 1;
    }
    int out_rows = img_rows * upscale; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int out_cols = img_cols * upscale;

    // Create black RGB image // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<uint8_t> img(out_rows * out_cols * 3, 0);

    // Draw trajectories // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (const auto& traj : trajectories) {
        auto color = traj_color(traj.index); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int i = 0; i + 1 < (int)traj.tuples.size(); i++) {
            auto [f1, idx1] = traj.tuples[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto [f2, idx2] = traj.tuples[i + 1];
            auto pos1 = locs.at(f1)[idx1]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            auto pos2 = locs.at(f2)[idx2];
            // Python uses xy = [int(round(x * upscale)), int(round(y * upscale))]
            // cv2.polylines takes (x,y) = (col,row) order
            // pos = (x, y, z) where x=col, y=row in image space
            int x0 = (int)std::round(pos1[0] * upscale); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            int y0 = (int)std::round(pos1[1] * upscale);
            int x1 = (int)std::round(pos2[0] * upscale);
            int y1 = (int)std::round(pos2[1] * upscale);
            draw_line(img, out_rows, out_cols, x0, y0, x1, y1, color[0], color[1], color[2]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }

    // Write PNG // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    FILE* fp = fopen(output_path.c_str(), "wb");
    if (!fp) { std::cerr << "Error: cannot write " << output_path << std::endl; return; }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_set_IHDR(png, info, out_cols, out_rows, 8, PNG_COLOR_TYPE_RGB, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (int r = 0; r < out_rows; r++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        png_write_row(png, &img[r * out_cols * 3]);
    }
    png_write_end(png, nullptr); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    png_destroy_write_struct(&png, &info);
    fclose(fp);
#else
    std::cerr << "Warning: libpng not available, skipping trajectory image" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#endif
}

// ============================================================
// count_localizations helper
// ============================================================

static std::vector<int> get_sorted_time_steps(const Localizations& loc) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> steps;
    for (const auto& [t, _] : loc) steps.push_back(t);
    std::sort(steps.begin(), steps.end());
    return steps;
}

// ============================================================
// run_tracking: top-level entry point
// ============================================================

bool run_tracking(const std::string& loc_csv_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                  const std::string& output_path,
                  int nb_frames,
                  const TrackingConfig& config) {
    if (config.verbose) std::cerr << "Reading localization CSV: " << loc_csv_path << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto loc = read_localization_csv(loc_csv_path, nb_frames);
    if (loc.empty()) {
        std::cerr << "Error: no localization data" << std::endl;
        return false;
    }

    // Get available time steps (frames with particles) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto all_steps = get_sorted_time_steps(loc);
    std::vector<int> t_avail_steps;
    for (int t : all_steps) {
        if (!loc[t].empty()) t_avail_steps.push_back(t);
    }

    if (t_avail_steps.size() < 2) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::cerr << "Error: need at least 2 frames with particles" << std::endl;
        return false;
    }

    int time_forecast = std::max(1, std::min(5, config.graph_depth)); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Segmentation // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (config.verbose) std::cerr << "Running segmentation..." << std::endl;
    auto seg = segmentation(loc, t_avail_steps, time_forecast);

    // Approximation (jump thresholds) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (config.verbose) std::cerr << "Computing jump thresholds..." << std::endl;
    auto max_jumps = approximation(seg.dist_x, seg.dist_y, seg.dist_z,
                                   time_forecast, config.jump_threshold);

    if (config.verbose) {
        std::cerr << "Jump threshold: " << max_jumps[0] << " px" << std::endl;
    }

    // Build empirical PDF // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> emp_bins_edges(41);
    for (int i = 0; i <= 40; i++) emp_bins_edges[i] = 20.0f * i / 40;
    build_emp_pdf(seg.seg_distribution[0], 40, 20.0f);

    // Count particles per frame // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (config.verbose) {
        int total = 0;
        for (const auto& [t, ps] : loc) total += (int)ps.size();
        float mean_nb = (float)total / all_steps.size();
        std::cerr << "Mean particles/frame: " << mean_nb << std::endl;
    }

    // Load qt_99 data for abnormal detection // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    {
        // Try multiple paths for qt_99.bin // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::string> qt_paths = {"models/qt_99.bin", "../models/qt_99.bin"};
        // Also try relative to loc_csv_path directory
        auto slash = loc_csv_path.rfind('/');
        if (slash != std::string::npos) {
            qt_paths.push_back(loc_csv_path.substr(0, slash) + "/../models/qt_99.bin");
        }
        for (const auto& qp : qt_paths) {
            if (load_qt_data(qp)) {
                if (config.verbose) std::cerr << "Loaded qt_99 data from " << qp << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                break;
            }
        }
        if (!g_qt_loaded && config.verbose) {
            std::cerr << "Warning: qt_99.bin not found, abnormal detection will be limited" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }

    // Load NN models if requested // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (config.use_nn) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<std::string> nn_dirs = {"models", "../models"}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        auto nn_slash = loc_csv_path.rfind('/'); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (nn_slash != std::string::npos) nn_dirs.push_back(loc_csv_path.substr(0, nn_slash) + "/../models"); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (const auto& nd : nn_dirs) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (load_nn_models(g_nn_models, nd)) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                if (config.verbose) std::cerr << "Loaded NN models from " << nd << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                break; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
        if (!g_nn_models.loaded && config.verbose) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            std::cerr << "Warning: ONNX models not found, using fixed alpha/k" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        }
    }

    // Run forecast // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (config.verbose) std::cerr << "Starting trajectory inference..." << std::endl;
    auto trajectories = forecast(loc, t_avail_steps, max_jumps, nb_frames, config);

    if (config.verbose) {
        std::cerr << "Found " << trajectories.size() << " trajectories" << std::endl;
    }

    // PostProcessing (optional) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (config.post_process) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (config.verbose) std::cerr << "Running post-processing..." << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        trajectories = post_processing(trajectories, loc, config.cutoff, config.verbose); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (config.verbose) std::cerr << "After post-processing: " << trajectories.size() << " trajectories" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    }

    // Extract output filename from loc_csv_path // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::string base = loc_csv_path;
    auto slash_pos = base.rfind('/');
    if (slash_pos != std::string::npos) base = base.substr(slash_pos + 1);
    auto loc_pos = base.find("_loc.csv");
    if (loc_pos != std::string::npos) base = base.substr(0, loc_pos);

    std::string out_csv = output_path + "/" + base + "_traces.csv"; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    write_trajectory_csv(out_csv, trajectories, loc);
    if (config.verbose) std::cerr << "Written: " << out_csv << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Image dimensions for trajectory visualization // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int img_rows = config.img_rows;
    int img_cols = config.img_cols;
    if (img_rows <= 0 || img_cols <= 0) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        // Infer from localization data
        float max_x = 0, max_y = 0;
        for (const auto& [t, particles] : loc) {
            for (const auto& p : particles) {
                if (p[0] > max_x) max_x = p[0];
                if (p[1] > max_y) max_y = p[1];
            }
        }
        img_rows = (int)std::ceil(max_y) + 2;
        img_cols = (int)std::ceil(max_x) + 2;
    }
    std::string out_img = output_path + "/" + base + "_traces.png"; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    make_trajectory_image(out_img, trajectories, loc, img_rows, img_cols);
    if (config.verbose) std::cerr << "Written: " << out_img << " (" << img_cols << "x" << img_rows << " * upscale)" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Cleanup NN models // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    if (g_nn_models.loaded) free_nn_models(g_nn_models); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    return true;
}

} // namespace freetrace
