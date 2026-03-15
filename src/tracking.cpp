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
#include <iomanip>
#include <charconv>
// Platform-specific includes for get_exe_dir() // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// x86-simd-sort: only available on x86/x64 with AVX2 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
#if defined(__AVX2__) || defined(__AVX512F__)
#include "../x86-simd-sort/src/x86simdsort-static-incl.h"
#define HAS_X86_SIMD_SORT 1
#else
#define HAS_X86_SIMD_SORT 0
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

namespace freetrace {

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
// CPython set iteration order replication
// Python's find_paths_as_iter uses nx.descendants_at_distance which returns
// a set. To match Python's DFS path order, we must iterate successors in
// CPython set iteration order (hash-table slot order).
// ============================================================

// CPython tuple hash for (int, int) — matches tuplehash() in tupleobject.c (Python 3.8+, 64-bit)
static uint64_t cpython_tuple_hash(int a, int b) {
    const uint64_t XXPRIME_1 = 11400714785074694791ULL;
    const uint64_t XXPRIME_2 = 14029467366897019727ULL;
    const uint64_t XXPRIME_5 = 2870177450012600261ULL;

    // hash(small_int) = small_int for non-negative, hash(-1) = -2
    uint64_t ha = (uint64_t)(int64_t)a;
    uint64_t hb = (uint64_t)(int64_t)b;

    uint64_t acc = XXPRIME_5;
    acc += ha * XXPRIME_2;
    acc = (acc << 31) | (acc >> 33);
    acc *= XXPRIME_1;
    acc += hb * XXPRIME_2;
    acc = (acc << 31) | (acc >> 33);
    acc *= XXPRIME_1;
    acc += 2ULL ^ (XXPRIME_5 ^ 3527539ULL);

    if (acc == (uint64_t)-1) return 1546275796ULL;
    return acc;
}

// Reorder nodes to match CPython set({nodes}) iteration order // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// CPython 3.12 set uses perturbation probing + LINEAR_PROBES (9 linear slots)
static void cpython_set_order(std::vector<Node>& nodes) {
    if (nodes.size() <= 1) return;

    const int PERTURB_SHIFT = 5;
    const int LINEAR_PROBES = 9;

    // Compute hashes
    std::vector<uint64_t> hashes(nodes.size());
    for (size_t i = 0; i < nodes.size(); i++) {
        hashes[i] = cpython_tuple_hash(nodes[i].first, nodes[i].second);
    }

    // Determine table size: start at 8, insert elements one by one
    // CPython resizes when fill*5 >= mask*3 (mask = table_size - 1)
    int table_size = 8;
    int fill = 0;

    struct Slot { bool occupied = false; Node node; uint64_t hash_val; };
    std::vector<Slot> table(table_size);

    // Insert function matching CPython set_add_entry with linear probing
    auto insert_into_table = [&](std::vector<Slot>& tbl, int tbl_size,
                                  const Node& node, uint64_t h) {
        uint64_t mask = (uint64_t)(tbl_size - 1);
        uint64_t i = h & mask;
        if (!tbl[(int)i].occupied) {
            tbl[(int)i] = {true, node, h};
            return;
        }
        uint64_t perturb = h;
        while (true) {
            // Linear probing: check i+1 through i+LINEAR_PROBES
            if (i + LINEAR_PROBES <= mask) {
                for (int j = 0; j < LINEAR_PROBES; j++) {
                    uint64_t li = i + (uint64_t)j + 1;
                    if (!tbl[(int)li].occupied) {
                        tbl[(int)li] = {true, node, h};
                        return;
                    }
                }
            }
            // Perturbation step
            perturb >>= PERTURB_SHIFT;
            i = ((i * 5) + perturb + 1) & mask;
            if (!tbl[(int)i].occupied) {
                tbl[(int)i] = {true, node, h};
                return;
            }
        }
    };

    for (size_t idx = 0; idx < nodes.size(); idx++) {
        uint64_t h = hashes[idx];
        insert_into_table(table, table_size, nodes[idx], h);
        fill++;

        // Check resize: CPython uses fill*5 >= mask*3
        uint64_t mask = (uint64_t)(table_size - 1);
        if ((uint64_t)fill * 5 >= mask * 3) {
            int used = fill;
            int minused = (used <= 50000) ? used * 4 : used * 2;
            int new_size = 8;
            while (new_size <= minused) new_size <<= 1;

            // Rehash into new table
            std::vector<Slot> new_table(new_size);
            for (int s = 0; s < table_size; s++) {
                if (table[s].occupied) {
                    insert_into_table(new_table, new_size, table[s].node, table[s].hash_val);
                }
            }
            table = std::move(new_table);
            table_size = new_size;
        }
    }

    // Extract in table-slot order (matching CPython set iteration)
    nodes.clear();
    for (int s = 0; s < table_size; s++) {
        if (table[s].occupied) {
            nodes.push_back(table[s].node);
        }
    }
}

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// NumPy-compatible argsort.
// On x86 with AVX2: uses x86-simd-sort (same library NumPy 2.0+ uses),
//   guaranteeing identical tie-breaking behavior.
// On other platforms (ARM/Apple Silicon): falls back to std::sort.
// ============================================================
static std::vector<int> numpy_argsort(const std::vector<double>& costs) {
    int n = (int)costs.size();
    if (n <= 1) {
        std::vector<int> idx(n);
        if (n == 1) idx[0] = 0;
        return idx;
    }
#if HAS_X86_SIMD_SORT
    auto result = x86simdsortStatic::argsort(costs.data(), (size_t)n);
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = (int)result[i];
    return idx;
#else
    // Fallback: stable sort preserving insertion order for ties (approximates // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    // NumPy's x86-simd-sort behavior for small arrays)
    std::vector<int> idx(n);
    for (int i = 0; i < n; i++) idx[i] = i;
    std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
        return costs[a] < costs[b];
    });
    return idx;
#endif
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// ============================================================
// Global state for empirical PDF (matches Python module globals)
// ============================================================
static std::vector<double> g_emp_pdf; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
static std::vector<double> g_emp_bins; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
static int g_emp_nb_bins = 40; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static double g_emp_max_val = 20.0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

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

void DiGraph::add_node(const Node& n) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    if (nodes_.insert(n).second) {
        nodes_ordered_.push_back(n);
    }
}

void DiGraph::add_edge(const Node& from, const Node& to, const EdgeData& data) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    if (nodes_.insert(from).second) nodes_ordered_.push_back(from);
    if (nodes_.insert(to).second) nodes_ordered_.push_back(to);
    // Insertion-ordered: update if exists, else append (matches Python dict behavior)
    auto& adj = fwd_[from];
    bool found = false;
    for (auto& [n, d] : adj) {
        if (n == to) { d = data; found = true; break; }
    }
    if (!found) adj.push_back({to, data});
    // Reverse: append if not already present
    auto& rev_list = rev_[to];
    if (std::find(rev_list.begin(), rev_list.end(), from) == rev_list.end())
        rev_list.push_back(from);
}

void DiGraph::remove_node(const Node& n) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    // Remove all edges from predecessors to n
    if (rev_.count(n)) {
        for (const auto& pred : rev_[n]) {
            if (fwd_.count(pred)) {
                auto& adj = fwd_[pred];
                adj.erase(std::remove_if(adj.begin(), adj.end(),
                    [&](const std::pair<Node, EdgeData>& e) { return e.first == n; }),
                    adj.end());
            }
        }
        rev_.erase(n);
    }
    // Remove all edges from n to successors
    if (fwd_.count(n)) {
        for (const auto& [succ, _] : fwd_[n]) {
            if (rev_.count(succ)) {
                auto& rl = rev_[succ];
                rl.erase(std::remove(rl.begin(), rl.end(), n), rl.end());
            }
        }
        fwd_.erase(n);
    }
    nodes_.erase(n);
    nodes_ordered_.erase(std::remove(nodes_ordered_.begin(), nodes_ordered_.end(), n), nodes_ordered_.end());
}

void DiGraph::remove_edge(const Node& from, const Node& to) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    if (fwd_.count(from)) {
        auto& adj = fwd_[from];
        adj.erase(std::remove_if(adj.begin(), adj.end(),
            [&](const std::pair<Node, EdgeData>& e) { return e.first == to; }),
            adj.end());
    }
    if (rev_.count(to)) {
        auto& rl = rev_[to];
        rl.erase(std::remove(rl.begin(), rl.end(), from), rl.end());
    }
}

bool DiGraph::has_node(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return nodes_.count(n) > 0;
}

bool DiGraph::has_edge(const Node& from, const Node& to) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    auto it = fwd_.find(from);
    if (it == fwd_.end()) return false;
    for (const auto& [n, _] : it->second) {
        if (n == to) return true;
    }
    return false;
}

bool DiGraph::has_path(const Node& from, const Node& to) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    if (from == to) return true;
    std::set<Node> visited;
    std::queue<Node> q;
    q.push(from);
    visited.insert(from);
    while (!q.empty()) {
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

std::vector<Node> DiGraph::successors(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    std::vector<Node> result;
    auto it = fwd_.find(n);
    if (it != fwd_.end()) {
        for (const auto& [succ, _] : it->second) result.push_back(succ);
    }
    return result; // Returns in insertion order (matches Python dict)
}

std::vector<Node> DiGraph::predecessors(const Node& n) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> result;
    auto it = rev_.find(n);
    if (it != rev_.end()) {
        for (const auto& pred : it->second) result.push_back(pred);
    }
    return result;
}

const EdgeData& DiGraph::get_edge_data(const Node& from, const Node& to) const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    static EdgeData empty;
    auto it = fwd_.find(from);
    if (it == fwd_.end()) return empty;
    for (const auto& [n, d] : it->second) {
        if (n == to) return d;
    }
    return empty;
}

const std::set<Node>& DiGraph::get_nodes() const { return nodes_; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
const std::vector<Node>& DiGraph::get_nodes_ordered() const { return nodes_ordered_; } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

size_t DiGraph::size() const { return nodes_.size(); } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

DiGraph DiGraph::copy() const { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    DiGraph g;
    g.nodes_ = nodes_;
    g.nodes_ordered_ = nodes_ordered_;
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

double euclidean_displacement_single(const std::array<double,3>& a, const std::array<double,3>& b) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx*dx + dy*dy + dz*dz);
}

std::vector<double> euclidean_displacement_batch(const std::vector<std::array<double,3>>& a, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                                                 const std::vector<std::array<double,3>>& b) {
    std::vector<double> result(a.size());
    for (size_t i = 0; i < a.size(); i++) {
        double dx = a[i][0] - b[i][0], dy = a[i][1] - b[i][1], dz = a[i][2] - b[i][2];
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
        double x = std::stod(vals[x_col]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        double y = std::stod(vals[y_col]);
        double z = (z_col >= 0 && z_col < (int)vals.size()) ? std::stod(vals[z_col]) : 0.0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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

// Format a double like Python's repr(): shortest representation that round-trips exactly // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
static std::string fmt_double(double v) {
    if (v == 0.0) return "0.0";
    char buf[32];
    auto [ptr, ec] = std::to_chars(buf, buf + sizeof(buf), v);
    std::string s(buf, ptr);
    // Ensure decimal point exists (like Python's repr, never bare integer form)
    if (s.find('.') == std::string::npos && s.find('e') == std::string::npos) s += ".0";
    return s;
}

void write_trajectory_csv(const std::string& path,
                          const std::vector<TrajectoryObj>& trajectories,
                          const Localizations& locs) {
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write to " << path << std::endl;
        return;
    }
    f << "traj_idx,frame,x,y,z\n";
    for (const auto& traj : trajectories) {
        for (const auto& [frame, idx] : traj.tuples) {
            const auto& pos = locs.at(frame)[idx];
            f << traj.index << "," << frame << "," << fmt_double(pos[0]) << "," << fmt_double(pos[1]) << "," << fmt_double(pos[2]) << "\n";
        }
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// ============================================================
// Greedy shortest matching (segmentation helper)
// ============================================================

GreedyResult greedy_shortest(const std::vector<std::array<double,3>>& srcs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                             const std::vector<std::array<double,3>>& dests) {
    GreedyResult result;
    int ns = (int)srcs.size(), nd = (int)dests.size();
    if (ns == 0 || nd == 0) return result;

    // Compute all pairwise distances // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<double> linkage(ns * nd, 0.0); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    std::vector<double> x_diff(ns * nd), y_diff(ns * nd), z_diff(ns * nd);
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

    // Outlier filtering (2 rounds of 4*mean_std filter + diffraction limit) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    double diffraction_light_limit = 10.0;
    auto compute_std = [](const std::vector<double>& v) -> double {
        if (v.size() < 2) return 0.0;
        double mean = 0;
        for (size_t i = 0; i + 1 < v.size(); i++) mean += v[i];
        mean /= (v.size() - 1);
        double var = 0;
        for (size_t i = 0; i + 1 < v.size(); i++) var += (v[i] - mean) * (v[i] - mean);
        return std::sqrt(var / (v.size() - 1)); // population std ddof=0, matching numpy np.std
    };

    // Check dimensionality
    double z_var = 0;
    if (!result.dist_z.empty()) {
        double z_mean = 0;
        for (double v : result.dist_z) z_mean += v;
        z_mean /= result.dist_z.size();
        for (double v : result.dist_z) z_var += (v - z_mean) * (v - z_mean);
        z_var /= result.dist_z.size();
    }
    int ndim = (z_var < 1e-5) ? 2 : 3;

    for (int round = 0; round < 2; round++) {
        double std_x = compute_std(result.dist_x);
        double std_y = compute_std(result.dist_y);
        double std_z = compute_std(result.dist_z);
        double estim_limit = (ndim == 2) ? 4.0 * (std_x + std_y) / 2.0
                                         : 4.0 * (std_x + std_y + std_z) / 3.0;
        double filter_min = std::max(estim_limit, diffraction_light_limit);

        std::vector<double> fx, fy, fz;
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

    // Final diffraction limit filter
    std::vector<double> fx, fy, fz;
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

// 1D k-means for GMM initialization (matches sklearn init_params='kmeans') // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
static void kmeans_1d(const std::vector<double>& data, int n_comp,
                      std::vector<double>& out_weights, std::vector<double>& out_means,
                      std::vector<double>& out_vars, std::mt19937& rng) {
    int n = (int)data.size();
    const double reg_covar = 1e-6;

    // k-means++ init for centroids
    std::vector<double> centroids(n_comp);
    std::uniform_int_distribution<int> uid(0, n - 1);
    centroids[0] = data[uid(rng)];
    for (int k = 1; k < n_comp; k++) {
        std::vector<double> dists(n);
        double total_d = 0;
        for (int i = 0; i < n; i++) {
            double min_d = 1e30;
            for (int j = 0; j < k; j++) {
                double d = (data[i] - centroids[j]) * (data[i] - centroids[j]);
                if (d < min_d) min_d = d;
            }
            dists[i] = min_d;
            total_d += min_d;
        }
        std::uniform_real_distribution<double> urd(0, total_d);
        double r = urd(rng);
        double cum = 0;
        for (int i = 0; i < n; i++) {
            cum += dists[i];
            if (cum >= r) { centroids[k] = data[i]; break; }
        }
    }

    // Run k-means iterations
    std::vector<int> labels(n);
    for (int iter = 0; iter < 100; iter++) {
        // Assign
        for (int i = 0; i < n; i++) {
            double best_d = 1e30;
            for (int k = 0; k < n_comp; k++) {
                double d = (data[i] - centroids[k]) * (data[i] - centroids[k]);
                if (d < best_d) { best_d = d; labels[i] = k; }
            }
        }
        // Update centroids
        std::vector<double> new_c(n_comp, 0);
        std::vector<int> counts(n_comp, 0);
        for (int i = 0; i < n; i++) { new_c[labels[i]] += data[i]; counts[labels[i]]++; }
        bool converged = true;
        for (int k = 0; k < n_comp; k++) {
            if (counts[k] > 0) {
                double nc = new_c[k] / counts[k];
                if (std::abs(nc - centroids[k]) > 1e-8) converged = false;
                centroids[k] = nc;
            }
        }
        if (converged) break;
    }

    // Compute initial weights, means, vars from k-means labels
    out_weights.assign(n_comp, 0);
    out_means.assign(n_comp, 0);
    out_vars.assign(n_comp, 0);
    std::vector<int> counts(n_comp, 0);
    for (int i = 0; i < n; i++) { out_means[labels[i]] += data[i]; counts[labels[i]]++; }
    for (int k = 0; k < n_comp; k++) {
        out_weights[k] = (double)counts[k] / n;
        if (counts[k] > 0) out_means[k] /= counts[k];
    }
    for (int i = 0; i < n; i++) {
        double d = data[i] - out_means[labels[i]];
        out_vars[labels[i]] += d * d;
    }
    for (int k = 0; k < n_comp; k++) {
        if (counts[k] > 0) out_vars[k] = out_vars[k] / counts[k] + reg_covar;
        else out_vars[k] = reg_covar;
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

static GMM1DResult fit_gmm_1d(const std::vector<double>& data, int n_comp, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                               int max_iter = 100, int n_init = 3,
                               bool use_mean_prior = false,
                               double mean_prior = 0.0,
                               double mean_precision_prior = 1e7) {
    int n = (int)data.size();
    if (n == 0) return {{}, {}, {}, -1e30};
    const double reg_covar = 1e-6; // matches sklearn default

    GMM1DResult best_result;
    best_result.log_likelihood = -1e30;

    std::mt19937 rng(42);

    for (int init = 0; init < n_init; init++) {
        // K-means initialization, then override means to 0 (matching sklearn with means_init)
        std::vector<double> weights, means, vars;
        kmeans_1d(data, n_comp, weights, means, vars, rng);
        // Override means with means_init=[[0],[0],...]
        for (int k = 0; k < n_comp; k++) means[k] = 0.0;

        // Responsibilities matrix
        std::vector<std::vector<double>> resp(n, std::vector<double>(n_comp));

        double prev_ll = -1e30;
        for (int iter = 0; iter < max_iter; iter++) {
            // E-step
            double ll = 0;
            for (int i = 0; i < n; i++) {
                double total = 0;
                for (int k = 0; k < n_comp; k++) {
                    resp[i][k] = weights[k] * gauss_pdf_1d(data[i], means[k], vars[k]);
                    total += resp[i][k];
                }
                if (total < 1e-300) total = 1e-300;
                ll += std::log(total);
                for (int k = 0; k < n_comp; k++) resp[i][k] /= total;
            }

            // Check convergence
            if (std::abs(ll - prev_ll) < 1e-6) break;
            prev_ll = ll;

            // M-step
            for (int k = 0; k < n_comp; k++) {
                double nk = 0;
                for (int i = 0; i < n; i++) nk += resp[i][k];
                if (nk < 1e-10) nk = 1e-10;

                weights[k] = nk / n;

                double sum_x = 0;
                for (int i = 0; i < n; i++) sum_x += resp[i][k] * data[i];
                if (use_mean_prior) {
                    means[k] = (sum_x + mean_precision_prior * mean_prior) / (nk + mean_precision_prior);
                } else {
                    means[k] = sum_x / nk;
                }

                double sum_var = 0;
                for (int i = 0; i < n; i++) {
                    double d = data[i] - means[k];
                    sum_var += resp[i][k] * d * d;
                }
                vars[k] = sum_var / nk + reg_covar;
            }
        }

        // Compute final log-likelihood
        double ll = 0;
        for (int i = 0; i < n; i++) {
            double total = 0;
            for (int k = 0; k < n_comp; k++) total += weights[k] * gauss_pdf_1d(data[i], means[k], vars[k]);
            if (total < 1e-300) total = 1e-300;
            ll += std::log(total);
        }

        if (ll > best_result.log_likelihood) {
            best_result.weights = weights;
            best_result.means = means;
            best_result.variances = vars;
            best_result.log_likelihood = ll;
        }
    }
    return best_result;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

// BIC score for 1D GMM: -2*LL + k*log(n) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static double gmm_bic(const GMM1DResult& result, int n_comp, int n_data) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int n_params = 3 * n_comp - 1; // weights(k-1) + means(k) + vars(k) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    return -2.0 * result.log_likelihood + n_params * std::log((double)n_data); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// Compute log-likelihood of test data given fitted GMM parameters // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
static double gmm_score(const GMM1DResult& model, const std::vector<double>& test_data) {
    double ll = 0;
    int n_comp = (int)model.weights.size();
    for (double x : test_data) {
        double total = 0;
        for (int k = 0; k < n_comp; k++) {
            total += model.weights[k] * gauss_pdf_1d(x, model.means[k], model.variances[k]);
        }
        if (total < 1e-300) total = 1e-300;
        ll += std::log(total);
    }
    return ll;
}

// 5-fold CV BIC model selection (matches sklearn GridSearchCV) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
static int gmm_cv_bic_select(const std::vector<double>& data, int max_comp = 3) {
    int n = (int)data.size();
    int n_folds = 5;

    // Create fold indices (contiguous blocks, matching sklearn KFold default)
    std::vector<int> fold_ids(n);
    int fold_size = n / n_folds;
    int remainder = n % n_folds;
    int idx = 0;
    for (int f = 0; f < n_folds; f++) {
        int sz = fold_size + (f < remainder ? 1 : 0);
        for (int j = 0; j < sz; j++) fold_ids[idx++] = f;
    }

    int best_nc = 1;
    double best_mean_bic = 1e30;

    for (int nc = 1; nc <= max_comp; nc++) {
        double bic_sum = 0;
        for (int fold = 0; fold < n_folds; fold++) {
            std::vector<double> train, test;
            for (int i = 0; i < n; i++) {
                if (fold_ids[i] == fold) test.push_back(data[i]);
                else train.push_back(data[i]);
            }
            // Fit on train
            auto model = fit_gmm_1d(train, nc, 100, 3, false);
            // BIC on test: -2*LL + n_params*log(n_test)
            double test_ll = gmm_score(model, test);
            int n_params = 3 * nc - 1;
            double bic = -2.0 * test_ll + n_params * std::log((double)test.size());
            bic_sum += bic;
        }
        double mean_bic = bic_sum / n_folds;
        if (mean_bic < best_mean_bic) {
            best_mean_bic = mean_bic;
            best_nc = nc;
        }
    }
    return best_nc;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

// np.quantile linear interpolation (method='linear', default) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
static double np_quantile(const std::vector<double>& sorted_data, double q) {
    int n = (int)sorted_data.size();
    double idx = q * (n - 1);
    int lo = (int)std::floor(idx);
    int hi = (int)std::ceil(idx);
    if (lo == hi || hi >= n) return sorted_data[lo];
    double frac = idx - lo;
    return sorted_data[lo] * (1.0 - frac) + sorted_data[hi] * frac;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 16:30
// Variational Bayesian GMM (1D, full covariance)
// Matches sklearn BayesianGaussianMixture with:
//   weight_concentration_prior_type='dirichlet_process'
//   covariance_type='full', mean_prior=[0], mean_precision_prior=1e7
// ============================================================

// digamma (psi) function via asymptotic expansion
static double digamma_fn(double x) {
    double result = 0;
    // Shift x to x >= 6 using psi(x) = psi(x+1) - 1/x
    while (x < 6.0) { result -= 1.0 / x; x += 1.0; }
    // Asymptotic: psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + 1/(120x^4) - 1/(252x^6)
    double inv_x = 1.0 / x;
    double inv_x2 = inv_x * inv_x;
    result += std::log(x) - 0.5 * inv_x
              - inv_x2 * (1.0/12.0 - inv_x2 * (1.0/120.0 - inv_x2 / 252.0));
    return result;
}

static GMM1DResult fit_bgmm_1d(const std::vector<double>& data, int n_comp,
                                int max_iter = 100, int n_init = 3,
                                double mean_prior_val = 0.0,
                                double mean_precision_prior = 1e7) {
    int n = (int)data.size();
    if (n == 0) return {{}, {}, {}, -1e30};
    const double reg_covar = 1e-6;
    const double tol = 1e-3; // sklearn default for BGMM

    // Compute priors from data
    double data_mean = 0;
    for (double x : data) data_mean += x;
    data_mean /= n;
    double data_cov = 0; // np.cov with ddof=1
    for (double x : data) data_cov += (x - data_mean) * (x - data_mean);
    data_cov /= (n - 1);

    double weight_conc_prior = 1.0 / n_comp; // sklearn default
    double dof_prior = 1.0;        // n_features = 1
    double cov_prior = data_cov;   // np.cov(X.T) for 1D

    std::mt19937 rng(42);
    GMM1DResult best_result;
    best_result.log_likelihood = -1e30;

    for (int init = 0; init < n_init; init++) {
        // K-means initialization → hard responsibilities
        std::vector<double> km_w, km_m, km_v;
        kmeans_1d(data, n_comp, km_w, km_m, km_v, rng);

        // Compute initial responsibilities from k-means labels
        // Recompute labels from centroids
        std::vector<int> labels(n);
        for (int i = 0; i < n; i++) {
            double best_d = 1e30;
            for (int k = 0; k < n_comp; k++) {
                double d = (data[i] - km_m[k]) * (data[i] - km_m[k]);
                if (d < best_d) { best_d = d; labels[i] = k; }
            }
        }

        // Compute initial nk, xk, sk from hard responsibilities
        std::vector<double> nk(n_comp, 0), xk(n_comp, 0), sk(n_comp, 0);
        for (int i = 0; i < n; i++) { nk[labels[i]] += 1.0; xk[labels[i]] += data[i]; }
        double eps10 = 10.0 * 2.2e-16; // 10 * np.finfo(float64).eps
        for (int k = 0; k < n_comp; k++) {
            nk[k] += eps10;
            xk[k] /= nk[k];
        }
        for (int i = 0; i < n; i++) {
            double d = data[i] - xk[labels[i]];
            sk[labels[i]] += d * d;
        }
        for (int k = 0; k < n_comp; k++) {
            sk[k] = sk[k] / nk[k] + reg_covar;
        }

        // Initialize posterior parameters via M-step
        // Weights (Dirichlet Process: stick-breaking)
        std::vector<double> wc_a(n_comp), wc_b(n_comp);
        { // cumsum of nk reversed
            std::vector<double> nk_rev(nk.rbegin(), nk.rend());
            std::vector<double> cs(n_comp, 0);
            cs[0] = nk_rev[0];
            for (int k = 1; k < n_comp; k++) cs[k] = cs[k-1] + nk_rev[k];
            // reverse back and shift: [cs[-2], cs[-3], ..., cs[0], 0]
            for (int k = 0; k < n_comp; k++) {
                wc_a[k] = 1.0 + nk[k];
                wc_b[k] = (k < n_comp - 1) ? weight_conc_prior + cs[n_comp - 2 - k] : weight_conc_prior;
            }
        }

        // Means
        std::vector<double> mean_prec(n_comp), means(n_comp);
        for (int k = 0; k < n_comp; k++) {
            mean_prec[k] = mean_precision_prior + nk[k];
            means[k] = (mean_precision_prior * mean_prior_val + nk[k] * xk[k]) / mean_prec[k];
        }

        // Wishart (covariances and degrees of freedom)
        std::vector<double> dof(n_comp), covs(n_comp);
        for (int k = 0; k < n_comp; k++) {
            dof[k] = dof_prior + nk[k];
            double diff = xk[k] - mean_prior_val;
            covs[k] = (cov_prior + nk[k] * sk[k] + nk[k] * mean_precision_prior / mean_prec[k] * diff * diff) / dof[k];
        }

        // Compute precisions for E-step
        std::vector<double> prec(n_comp);
        for (int k = 0; k < n_comp; k++) prec[k] = 1.0 / covs[k];

        double prev_lower_bound = -1e30;

        for (int iter = 0; iter < max_iter; iter++) {
            // E-step: compute log responsibilities
            // log_weights (Dirichlet Process stick-breaking)
            std::vector<double> log_weights(n_comp);
            {
                std::vector<double> dig_sum(n_comp), dig_a(n_comp), dig_b(n_comp);
                for (int k = 0; k < n_comp; k++) {
                    dig_sum[k] = digamma_fn(wc_a[k] + wc_b[k]);
                    dig_a[k] = digamma_fn(wc_a[k]);
                    dig_b[k] = digamma_fn(wc_b[k]);
                }
                // log_weights[0] = dig_a[0] - dig_sum[0]
                // log_weights[k] = dig_a[k] - dig_sum[k] + sum(dig_b[j] - dig_sum[j], j=0..k-1)
                double cum = 0;
                for (int k = 0; k < n_comp; k++) {
                    log_weights[k] = dig_a[k] - dig_sum[k] + cum;
                    cum += dig_b[k] - dig_sum[k];
                }
            }

            // log_prob for each data point and component
            // For 1D full: log_lambda = log(2) + digamma(0.5 * dof[k])
            // log_prob = log_gauss - 0.5*log(dof[k]) + 0.5*(log_lambda - 1/mean_prec[k])
            // log_gauss = -0.5*log(2*pi) + 0.5*log(prec[k]) - 0.5*prec[k]*(x-mean[k])^2
            // But sklearn normalizes precision by dof, so:
            // log_gauss uses precisions_cholesky_ = sqrt(1/covs[k]) which gives prec = 1/covs[k]
            // Then subtracts 0.5*n_features*log(dof[k]) = 0.5*log(dof[k])

            std::vector<double> log_lambda(n_comp);
            for (int k = 0; k < n_comp; k++) {
                log_lambda[k] = std::log(2.0) + digamma_fn(0.5 * dof[k]);
            }

            // Compute log_resp and log_prob_norm (for ELBO)
            std::vector<std::vector<double>> log_resp(n, std::vector<double>(n_comp));
            double log_prob_norm_sum = 0;

            for (int i = 0; i < n; i++) {
                double max_lr = -1e30;
                for (int k = 0; k < n_comp; k++) {
                    double d = data[i] - means[k];
                    // log_gauss (using precision = 1/covs[k], but NOT divided by dof yet)
                    double log_gauss = -0.5 * std::log(2.0 * M_PI)
                                       + 0.5 * std::log(prec[k])
                                       - 0.5 * prec[k] * d * d;
                    // Subtract 0.5*log(dof[k]) per sklearn
                    log_gauss -= 0.5 * std::log(dof[k]);
                    // Add log_lambda and mean_precision correction
                    double log_prob = log_gauss + 0.5 * (log_lambda[k] - 1.0 / mean_prec[k]);
                    log_resp[i][k] = log_weights[k] + log_prob;
                    if (log_resp[i][k] > max_lr) max_lr = log_resp[i][k];
                }
                // Log-sum-exp normalization
                double lse = 0;
                for (int k = 0; k < n_comp; k++) lse += std::exp(log_resp[i][k] - max_lr);
                double log_norm = max_lr + std::log(lse);
                log_prob_norm_sum += log_norm;
                for (int k = 0; k < n_comp; k++) log_resp[i][k] -= log_norm;
            }

            // M-step: compute sufficient statistics
            std::fill(nk.begin(), nk.end(), 0);
            std::fill(xk.begin(), xk.end(), 0);
            std::fill(sk.begin(), sk.end(), 0);
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n_comp; k++) {
                    double r = std::exp(log_resp[i][k]);
                    nk[k] += r;
                    xk[k] += r * data[i];
                }
            }
            for (int k = 0; k < n_comp; k++) {
                nk[k] += eps10;
                xk[k] /= nk[k];
            }
            for (int i = 0; i < n; i++) {
                for (int k = 0; k < n_comp; k++) {
                    double r = std::exp(log_resp[i][k]);
                    double d = data[i] - xk[k];
                    sk[k] += r * d * d;
                }
            }
            for (int k = 0; k < n_comp; k++) {
                sk[k] = sk[k] / nk[k] + reg_covar;
            }

            // Update weights (Dirichlet Process)
            {
                std::vector<double> nk_rev(nk.rbegin(), nk.rend());
                std::vector<double> cs(n_comp, 0);
                cs[0] = nk_rev[0];
                for (int j = 1; j < n_comp; j++) cs[j] = cs[j-1] + nk_rev[j];
                for (int k = 0; k < n_comp; k++) {
                    wc_a[k] = 1.0 + nk[k];
                    wc_b[k] = (k < n_comp - 1) ? weight_conc_prior + cs[n_comp - 2 - k] : weight_conc_prior;
                }
            }

            // Update means
            for (int k = 0; k < n_comp; k++) {
                mean_prec[k] = mean_precision_prior + nk[k];
                means[k] = (mean_precision_prior * mean_prior_val + nk[k] * xk[k]) / mean_prec[k];
            }

            // Update Wishart
            for (int k = 0; k < n_comp; k++) {
                dof[k] = dof_prior + nk[k];
                double diff = xk[k] - mean_prior_val;
                covs[k] = (cov_prior + nk[k] * sk[k] + nk[k] * mean_precision_prior / mean_prec[k] * diff * diff) / dof[k];
                prec[k] = 1.0 / covs[k];
            }

            // Convergence check on total ELBO (matching sklearn's tol on total lower bound)
            double lower_bound = log_prob_norm_sum;
            double change = lower_bound - prev_lower_bound;
            if (std::abs(change) < tol) break;
            prev_lower_bound = lower_bound;
        }

        // Compute final weights via stick-breaking: E[V_k] = a_k/(a_k+b_k)
        std::vector<double> weights(n_comp);
        {
            double remaining = 1.0;
            for (int k = 0; k < n_comp; k++) {
                double ev = wc_a[k] / (wc_a[k] + wc_b[k]);
                weights[k] = ev * remaining;
                remaining *= (1.0 - ev);
            }
            // Normalize
            double wsum = 0;
            for (double w : weights) wsum += w;
            for (double& w : weights) w /= wsum;
        }

        // Use ELBO as comparison metric across inits
        if (prev_lower_bound > best_result.log_likelihood) {
            best_result.weights = weights;
            best_result.means = means;
            best_result.variances = covs; // posterior covariances (W_k/dof_k)
            best_result.log_likelihood = prev_lower_bound;
        }
    }
    return best_result;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 16:30

// approx_gauss: full GMM-based jump threshold estimation (matches Python) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static double approx_gauss(const std::vector<std::vector<double>>& distributions) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    double min_euclid = 5.0;
    std::vector<double> max_xyz;

    for (const auto& raw_dist : distributions) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        if (raw_dist.empty()) continue;

        // Check variance
        double mean_v = 0;
        for (double x : raw_dist) mean_v += x;
        mean_v /= raw_dist.size();
        double var_v = 0;
        for (double x : raw_dist) var_v += (x - mean_v) * (x - mean_v);
        var_v /= raw_dist.size();
        if (var_v <= 1e-5) continue;

        // Quantile filter (2.5% - 97.5%) with np.quantile-compatible interpolation // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        std::vector<double> sorted_dist(raw_dist.begin(), raw_dist.end());
        std::sort(sorted_dist.begin(), sorted_dist.end());
        double q025 = np_quantile(sorted_dist, 0.025);
        double q975 = np_quantile(sorted_dist, 0.975);
        std::vector<double> filtered;
        for (double x : raw_dist) {
            if (x > q025 && x < q975) filtered.push_back(x);
        }
        if (filtered.empty()) continue;

        // Recheck variance after filtering // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        double fmean = 0;
        for (double x : filtered) fmean += x;
        fmean /= filtered.size();
        double fvar = 0;
        for (double x : filtered) fvar += (x - fmean) * (x - fmean);
        fvar /= filtered.size();
        if (fvar <= 1e-5) continue;

        // 5-fold CV BIC model selection (matches sklearn GridSearchCV) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        int best_n_comp = gmm_cv_bic_select(filtered, 3);

        // Variational Bayesian GMM (matches sklearn BayesianGaussianMixture) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        auto cluster = fit_bgmm_1d(filtered, best_n_comp, 100, 3, 0.0, 1e7);

        // Select components near zero with weight > 0.05 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> selec_var; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        for (int k = 0; k < best_n_comp; k++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            if (cluster.means[k] > -1.0 && cluster.means[k] < 1.0 && cluster.weights[k] > 0.05) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                selec_var.push_back(cluster.variances[k]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            }
        }
        if (selec_var.empty()) continue;

        // Take the component with largest variance // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        double max_var = *std::max_element(selec_var.begin(), selec_var.end());
        max_xyz.push_back(std::sqrt(max_var) * 2.5); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    }

    // Compute Euclidean norm across dimensions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    double max_euclid_sq = 0;
    for (double v : max_xyz) max_euclid_sq += v * v;
    return std::max(std::sqrt(max_euclid_sq), min_euclid); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
}

std::map<int, double> approximation(const std::vector<double>& dist_x, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                                   const std::vector<double>& dist_y,
                                   const std::vector<double>& dist_z,
                                   int time_forecast, float jump_threshold) {
    std::map<int, double> approx;
    if (jump_threshold > 0) {
        for (int t = 0; t <= time_forecast; t++) approx[t] = (double)jump_threshold;
    } else {
        // Full GMM-based threshold estimation (matches Python approx_gauss)
        std::vector<std::vector<double>> distributions;
        distributions.push_back(dist_x);
        distributions.push_back(dist_y);
        if (!dist_z.empty()) distributions.push_back(dist_z);
        double max_euclid = approx_gauss(distributions);
        for (int t = 0; t <= time_forecast; t++) approx[t] = max_euclid;
    }
    return approx;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

// ============================================================
// Empirical PDF
// ============================================================

void build_emp_pdf(const std::vector<double>& emp_distribution, int nb_bins, float max_val) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    // nb_bins = number of edge points (matching Python np.linspace(0, 20, 40) = 40 edges)
    // Actual histogram bins = nb_bins - 1 (39 bins, matching np.histogram)
    int n_edges = nb_bins;  // 40 edges
    int n_hist_bins = n_edges - 1;  // 39 bins
    g_emp_nb_bins = n_edges;  // store edge count for lookup (matching Python len(EMP_BINS)=40)
    g_emp_max_val = (double)max_val;
    g_emp_bins.resize(n_edges);
    for (int i = 0; i < n_edges; i++) g_emp_bins[i] = (double)max_val * (double)i / (double)(n_edges - 1);

    std::vector<double> data;
    if ((int)emp_distribution.size() < 1000) {
        std::mt19937 rng(42);
        std::exponential_distribution<double> exp_dist(1.0);
        data.resize(10000);
        for (auto& v : data) v = exp_dist(rng);
    } else {
        data = emp_distribution;
    }

    // Compute histogram with n_hist_bins bins using n_edges bin edges
    // Matching np.histogram(data, bins=EMP_BINS, density=True)
    g_emp_pdf.assign(n_hist_bins, 0.0);
    double bin_width = g_emp_bins[1] - g_emp_bins[0]; // uniform bin width
    for (double v : data) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        // Use binary search via std::upper_bound (matching numpy bin assignment)
        auto it = std::upper_bound(g_emp_bins.begin(), g_emp_bins.end(), v);
        int bin = (int)(it - g_emp_bins.begin()) - 1;
        // Last bin includes right edge: edge[-2] <= v <= edge[-1]
        if (bin == n_hist_bins && v == g_emp_bins.back()) bin = n_hist_bins - 1;
        if (bin >= 0 && bin < n_hist_bins) g_emp_pdf[bin] += 1.0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    }
    // Normalize to density: count / (total * bin_width)
    double total = 0;
    for (double c : g_emp_pdf) total += c;
    if (total > 0) {
        for (double& c : g_emp_pdf) c /= (total * bin_width);
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

double empirical_pdf_lookup(const std::array<double,3>& coords) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    // Match Python: idx = int(min((jump_d / EMP_BINS[-1] * len(EMP_BINS)), len(EMP_BINS) - 1))
    double jump_d = std::sqrt(coords[0]*coords[0] + coords[1]*coords[1] + coords[2]*coords[2]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    int idx = (int)std::min(jump_d / g_emp_bins.back() * (double)g_emp_nb_bins, (double)(g_emp_nb_bins - 1));
    idx = std::max(0, std::min(idx, (int)g_emp_pdf.size() - 1));
    return std::log(g_emp_pdf[idx] + 1e-7); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
}

// ============================================================
// Regularization helpers (for predict_cauchy_tracking)
// ============================================================

static double scaling_func(double numer) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    return std::max(0.0, std::min(2.0, -std::log10(std::abs(numer))));
}

static double regularization(double numer, double denom, double expect) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    if (numer < 0) numer -= 2e-1; else numer += 2e-1;
    if (denom < 0) denom -= 2e-1; else denom += 2e-1;
    double scaler = std::min(std::sqrt(scaling_func(numer) + 2.0 * scaling_func(denom)) / 2.0, 1.0);
    double scaled_ratio = (numer / denom) + (expect - numer / denom) * std::pow(scaler, 1.0 / 3.0);
    return scaled_ratio;
}

// ============================================================
// Cauchy cost (tracking version)
// ============================================================

CauchyResult predict_cauchy_tracking(const std::array<double,3>& next_vec, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                     const std::array<double,3>& prev_vec,
                                     double k, double alpha,
                                     int before_lag, int lag,
                                     double precision, int dimension) {
    CauchyResult result = {0.0, false};
    double d_alpha = alpha; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    double delta_s = (double)(before_lag + 1);
    double delta_t = (double)(lag + 1);

    if (std::abs(d_alpha - 1.0) < 1e-4) d_alpha += 1e-4;
    double rho = std::pow(2.0, d_alpha - 1.0) - 1.0;
    double std_ratio = std::sqrt(
        (std::pow(2.0 * delta_t, d_alpha) - 2.0 * std::pow(delta_t, d_alpha)) /
        (std::pow(2.0 * delta_s, d_alpha) - 2.0 * std::pow(delta_s, d_alpha))
    );
    double scale = std::sqrt(1.0 - rho * rho) * std_ratio;

    std::vector<double> coord_ratios;
    for (int d = 0; d < dimension; d++) {
        double cr = regularization((double)next_vec[d], (double)prev_vec[d], rho * std_ratio);
        coord_ratios.push_back(cr);
    }

    for (double cr : coord_ratios) {
        double density = (1.0 / (M_PI * scale)) *
                        (1.0 / (((cr - rho * std_ratio) * (cr - rho * std_ratio)) /
                                 (scale * scale * std_ratio) + std_ratio));
        result.log_pdf += std::log(density);
    }

    // Abnormal check: |coord_ratio - std_ratio*rho| > 6*scale
    for (double cr : coord_ratios) {
        if (std::abs(cr - std_ratio * rho) > 6.0 * scale) {
            result.abnormal = true;
            break;
        }
    }

    // qt_fetch-based abnormal check
    if (g_qt_loaded) {
        auto [ai, ki] = indice_fetch(d_alpha, k);
        double qt_val = qt_fetch(ai, ki);
        double jump_mag = std::sqrt(next_vec[0]*next_vec[0] + next_vec[1]*next_vec[1] + next_vec[2]*next_vec[2]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        if (jump_mag - qt_val * 2.5 > 0) {
            result.abnormal = true;
        }
    }

    return result;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

// ============================================================
// DFS path enumeration
// ============================================================

void find_paths_dfs(const DiGraph& G, const Node& current, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                    Path& path, std::set<Node>& seen,
                    std::vector<Path>& results) {
    seen.insert(current);
    auto neighbors = G.successors(current);
    // Reorder successors to match CPython set iteration order
    // (Python's find_paths_as_iter uses nx.descendants_at_distance which returns a set)
    cpython_set_order(neighbors);

    if (neighbors.empty()) {
        results.push_back(path); // leaf: yield the path
    }
    for (const auto& neighbor : neighbors) {
        if (!seen.count(neighbor)) {
            path.push_back(neighbor);
            find_paths_dfs(G, neighbor, path, seen, results);
            path.pop_back();
        } else {
            // Match Python: when n in seen, yield current path
            results.push_back(path);
        }
    }
    seen.erase(current);
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

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

std::vector<DiGraph> split_to_subgraphs(const DiGraph& G, const Node& source_node) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    std::vector<DiGraph> subgraphs;

    // Get first edges (source -> direct neighbors) in insertion order
    auto first_neighbors = G.successors(source_node);

    // Build undirected graph without source, matching Python's to_undirected() order
    // Python iterates directed edges in (node-insertion-order, successor-insertion-order)
    // For each edge (u,v): add v to u's neighbor list and u to v's neighbor list
    std::vector<Node> all_nodes_ordered; // node insertion order for undirected graph
    std::set<Node> all_nodes_set;
    std::map<Node, std::vector<Node>> undirected; // insertion-ordered adjacency

    auto add_undirected_edge = [&](const Node& u, const Node& v) {
        // Add v to u's list if not present, and u to v's list if not present
        auto& ul = undirected[u];
        if (std::find(ul.begin(), ul.end(), v) == ul.end()) ul.push_back(v);
        auto& vl = undirected[v];
        if (std::find(vl.begin(), vl.end(), u) == vl.end()) vl.push_back(u);
        // Track node order
        if (all_nodes_set.insert(u).second) all_nodes_ordered.push_back(u);
        if (all_nodes_set.insert(v).second) all_nodes_ordered.push_back(v);
    };

    // Iterate nodes in insertion order (matching Python's graph node iteration)
    for (const auto& n : G.get_nodes_ordered()) {
        if (n == source_node) continue;
        if (all_nodes_set.insert(n).second) all_nodes_ordered.push_back(n);
        for (const auto& succ : G.successors(n)) {
            if (succ == source_node) continue;
            add_undirected_edge(n, succ);
        }
    }

    // Track remaining nodes in insertion order
    std::vector<Node> remaining_ordered = all_nodes_ordered;

    while (!remaining_ordered.empty()) {
        // Python: arb_node = node_list[0] (first in current node list)
        Node arb_node = remaining_ordered[0];
        std::vector<Node> bfs_nodes;
        std::set<Node> bfs_visited;
        std::queue<Node> bfs_q;
        bfs_q.push(arb_node);
        bfs_visited.insert(arb_node);
        bool has_edges = false;

        while (!bfs_q.empty()) {
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

        // Build directed subgraph matching Python's edge insertion order exactly. // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        // Python iterates bfs_nodes, for each gets undirected edges, and adds
        // directed edges based on time ordering (edge[0][0] < edge[1][0]).
        // This insertion order determines successor ordering in the subgraph,
        // which affects cpython_set_order hash collision resolution, DFS path
        // enumeration, and ultimately the greedy path selection.
        DiGraph sub_graph;
        std::set<Node> component_set_tmp(bfs_nodes.begin(), bfs_nodes.end());
        for (const auto& node : bfs_nodes) {
            if (undirected.count(node)) {
                for (const auto& neighbor : undirected[node]) {
                    if (!component_set_tmp.count(neighbor)) continue;
                    // Add directed edge: earlier time -> later time
                    if (node.first < neighbor.first) {
                        sub_graph.add_edge(node, neighbor, G.get_edge_data(node, neighbor));
                    } else if (neighbor.first < node.first) {
                        sub_graph.add_edge(neighbor, node, G.get_edge_data(neighbor, node));
                    }
                    // If same time (shouldn't happen in tracking), skip
                }
            }
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

        // Remove processed nodes from remaining list
        std::set<Node> component_set(bfs_nodes.begin(), bfs_nodes.end());
        if (bfs_nodes.size() == 1 && !has_edges) {
            sub_graph.add_edge(source_node, arb_node);
            component_set.insert(arb_node);
        }
        // Python: graph_copy_without_source.remove_nodes_from(sub_graph_.nodes)
        std::vector<Node> new_remaining;
        for (const auto& n : remaining_ordered) {
            if (!component_set.count(n)) new_remaining.push_back(n);
        }
        remaining_ordered = std::move(new_remaining);

        // Add source edges to subgraph for any first_neighbor in this component
        for (const auto& fn : first_neighbors) {
            if (sub_graph.has_node(fn)) {
                sub_graph.add_edge(source_node, fn);
            }
        }

        // Debug: dump subgraph composition // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        subgraphs.push_back(std::move(sub_graph));
    }
    return subgraphs;
}

// ============================================================
// Terminal check
// ============================================================

bool is_terminal_node(const Node& node, const Localizations& locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                      double max_jump_d, const DiGraph& selected_graph,
                      const std::set<Node>& final_graph_nodes,
                      int time_forecast) {
    int node_t = node.first;
    auto node_loc = locs.at(node_t)[node.second];
    // Match Python: range(node_t + 1, node_t + TIME_FORECAST + 1)
    for (int next_t = node_t + 1; next_t <= node_t + time_forecast; next_t++) {
        auto it = locs.find(next_t);
        if (it == locs.end()) continue;
        const auto& particles = it->second;
        for (int idx = 0; idx < (int)particles.size(); idx++) {
            if (particles[idx].size() < 3) continue;
            Node next_node = {next_t, idx};
            if (selected_graph.has_node(next_node)) continue;
            if (final_graph_nodes.count(next_node)) continue;
            double d = euclidean_displacement_single(node_loc, particles[idx]); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
            if (d < max_jump_d) return false;
        }
    }
    return true; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
}

// ============================================================
// Generate next paths (build graph edges)
// ============================================================

GenerateResult generate_next_paths(DiGraph next_graph, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                   const std::set<Node>& final_graph_nodes,
                                   const Localizations& locs,
                                   const std::vector<int>& next_times,
                                   const std::map<int, double>& distribution, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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
                        double jump_d = euclidean_displacement_single(cur_particles[next_idx], node_loc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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

static const double UNCOMPUTED = -999999.0; // sentinel for uncomputed cost // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

PredictResult predict_long_seq(const Path& next_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                               std::map<Path, double>& trajectories_costs,
                               const Localizations& locs,
                               double prev_alpha, double prev_k,
                               const std::vector<int>& next_times,
                               const Path* prev_path_ptr,
                               std::map<Path, int>& start_indice,
                               int last_time, double jump_threshold,
                               const DiGraph& selected_graph,
                               const std::set<Node>& final_graph_nodes,
                               int time_forecast, int dimension,
                               float loc_precision_err,
                               bool use_nn) {
    PredictResult presult;
    double abnormal_penalty = 1000.0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    double time_penalty = abnormal_penalty / (double)(time_forecast + 1);
    double cutting_threshold = 2.0 * abnormal_penalty;
    double initial_cost = cutting_threshold - 100.0;

    // Already computed or single node? // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto cost_it = trajectories_costs.find(next_path);
    if ((cost_it != trajectories_costs.end() && cost_it->second != UNCOMPUTED) || next_path.size() <= 1) {
        presult.terminal = -1; // None equivalent
        return presult;
    }

    // Terminal check // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    bool terminal = is_terminal_node(next_path.back(), locs, jump_threshold, selected_graph, final_graph_nodes, time_forecast);
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
    } else if (next_path.size() == 2) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        trajectories_costs[next_path] = time_penalty;
    } else if (next_path.size() == 3) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        Node before_node = next_path[1];
        Node next_node = next_path[2];
        int time_gap = next_node.first - before_node.first - 1;
        std::array<double,3> dummy = {0.15, 0.15, 0.0}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        double log_p0 = empirical_pdf_lookup(dummy);
        double time_score = time_gap * time_penalty;
        trajectories_costs[next_path] = time_score + std::abs(log_p0 - 5.0);
    } else { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        // len >= 4
        int last_idx = terminal ? -1 : -2; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int path_len = (int)next_path.size();

        // First edge cost (empirical PDF) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        std::vector<double> traj_cost;
        std::vector<int> ab_index;
        Node before_node = next_path[1];
        Node next_node = next_path[2];
        auto next_coord = locs.at(next_node.first)[next_node.second];
        auto cur_coord = locs.at(before_node.first)[before_node.second];
        std::array<double,3> input_mu = {next_coord[0] - cur_coord[0],
                                        next_coord[1] - cur_coord[1],
                                        next_coord[2] - cur_coord[2]};
        double log_p0 = empirical_pdf_lookup(input_mu);
        traj_cost.push_back(std::abs(log_p0 - 5.0));

        // Cauchy cost for subsequent edges // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        int end_idx = path_len + last_idx - 1;
        std::vector<double> tmpx, tmpy; // for NN k re-estimation after abnormal // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        Node j_prev_node, j_next_node;

        for (int edge_i = 1; edge_i < end_idx; edge_i++) {
            int edge_j = edge_i + 1;

            // Re-estimate k after abnormal detections // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
            if (!ab_index.empty()) {
                prev_alpha = 1.0;
                if (use_nn && g_nn_models.loaded) {
                    // Accumulate positions for NN k prediction (matching Python)
                    if (tmpx.empty()) {
                        tmpx = {locs.at(j_prev_node.first)[j_prev_node.second][0],
                                locs.at(j_next_node.first)[j_next_node.second][0]};
                        tmpy = {locs.at(j_prev_node.first)[j_prev_node.second][1],
                                locs.at(j_next_node.first)[j_next_node.second][1]};
                    } else {
                        tmpx.push_back(locs.at(j_next_node.first)[j_next_node.second][0]);
                        tmpy.push_back(locs.at(j_next_node.first)[j_next_node.second][1]);
                    }
                    // Use at most 5 points (matching Python tmpx[:5])
                    int nn_len = std::min((int)tmpx.size(), 5); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                    std::vector<float> tx(nn_len), ty(nn_len);
                    for (int ni = 0; ni < nn_len; ni++) { tx[ni] = (float)tmpx[ni]; ty[ni] = (float)tmpy[ni]; }
                    prev_k = predict_k_nn(g_nn_models, tx, ty);
                } else {
                    prev_k = 0.5; // predict_ks returns 0.5 when TF=False
                }
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

            std::array<double,3> vec_i = {loc_i_next[0] - loc_i_prev[0], // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                         loc_i_next[1] - loc_i_prev[1],
                                         loc_i_next[2] - loc_i_prev[2]};
            std::array<double,3> vec_j = {loc_j_next[0] - loc_j_prev[0],
                                         loc_j_next[1] - loc_j_prev[1],
                                         loc_j_next[2] - loc_j_prev[2]};

            auto cauchy = predict_cauchy_tracking(vec_j, vec_i, prev_k, prev_alpha, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                                  i_time_gap, j_time_gap, loc_precision_err, dimension);
            if (cauchy.abnormal) ab_index.push_back(edge_j);
            traj_cost.push_back(std::abs(cauchy.log_pdf - 5.0));
        }

        // Compute time gaps penalty // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int time_gaps = 0;
        for (int ni = 1; ni < (int)next_path.size() - 1; ni++) {
            time_gaps += (next_path[ni + 1].first - next_path[ni].first) - 1;
        }

        // Sort and deduplicate ab_index // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::sort(ab_index.begin(), ab_index.end());
        ab_index.erase(std::unique(ab_index.begin(), ab_index.end()), ab_index.end());

        double abnormal_jump_score = abnormal_penalty * (double)ab_index.size(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        double time_score = (double)time_gaps * time_penalty;

        double final_score;
        if (traj_cost.size() > 1) {
            double mean_cost = 0;
            for (double c : traj_cost) mean_cost += c;
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

static void greedy_assign_pass( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 21:30
    DiGraph& sub_graph,
    DiGraph& out_graph,
    const Node& source_node,
    const std::set<Node>& final_graph_node_set_hashed,
    const Localizations& locs,
    const std::vector<int>& next_times,
    const std::map<int, double>& distribution, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    bool first_step,
    int last_time,
    const TrackingConfig& config,
    const std::vector<Path>* prev_paths_ptr,
    const std::map<Path, double>* alpha_values_ptr, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    const std::map<Path, double>* k_values_ptr,
    std::map<Path, int>& hashed_prev_next,
    int initial_pick_idx,
    double& cost_sum,
    const std::vector<Node>& last_nodes,
    bool reconnect_orphans = true) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
{
    int nb_to_optimum = 1 << config.graph_depth; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    (void)nb_to_optimum;

    std::map<Path, double> trajectories_costs; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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
        // Rebuild cost_copy (prune stale entries) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        std::map<Path, double> cost_copy;
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
            double use_alpha = config.init_alpha; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
            double use_k = config.init_k; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

            if (!first_step && prev_paths_ptr) {
                pp = match_prev_next(*prev_paths_ptr, np, hashed_prev_next);
                if (pp && alpha_values_ptr && k_values_ptr) {
                    auto ait = alpha_values_ptr->find(*pp);
                    auto kit = k_values_ptr->find(*pp);
                    if (ait != alpha_values_ptr->end()) use_alpha = ait->second;
                    if (kit != k_values_ptr->end()) use_k = kit->second;
                }
            }

            auto pr = predict_long_seq(np, trajectories_costs, locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                                       use_alpha, use_k, next_times, pp,
                                       start_indice_map, last_time,
                                       distribution.at(0), out_graph,
                                       final_graph_node_set_hashed,
                                       config.graph_depth, config.dimension,
                                       config.loc_precision_err,
                                       config.use_nn);
            if (pr.terminal >= 0) is_terminals[np] = (pr.terminal == 1);
            if (!pr.ab_index.empty()) ab_indice[np] = pr.ab_index;
        }

        // Sort by cost using numpy-compatible quicksort argsort // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        std::vector<double> cost_values;
        for (int i = 0; i < (int)next_paths.size(); i++) {
            double c = trajectories_costs.count(next_paths[i]) ? trajectories_costs[next_paths[i]] : 1e9;
            if (c == UNCOMPUTED) c = 1e9;
            cost_values.push_back(c);
        }
        auto sorted_indices = numpy_argsort(cost_values);
        std::vector<std::pair<double, int>> cost_idx_vec;
        for (int si : sorted_indices) {
            cost_idx_vec.push_back({cost_values[si], si});
        }

        Path lowest_cost_traj; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        int actual_pick = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
        if (initial_) {
            if (initial_pick_idx < (int)cost_idx_vec.size()) {
                lowest_cost_traj = next_paths[cost_idx_vec[initial_pick_idx].second];
                actual_pick = initial_pick_idx;
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
                            double jump_d = euclidean_displacement_single(pred_loc, suc_loc); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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

        // Reconnect orphaned nodes to source — only in phase 1 (matching Python) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        if (reconnect_orphans) { // phase 1 only — Python phase 2 has no orphan reconnection // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
            std::set<Node> last_nodes_set(last_nodes.begin(), last_nodes.end());
            auto orphan_check_nodes = sub_graph.get_nodes_ordered(); // use insertion order like Python
            for (const auto& sn : orphan_check_nodes) {
                if (sn != source_node && !last_nodes_set.count(sn) && !sub_graph.has_path(source_node, sn)) {
                    sub_graph.add_edge(source_node, sn);
                }
            }
        }

        double pop_cost = trajectories_costs.count(lowest_cost_traj) ? trajectories_costs[lowest_cost_traj] : 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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
                               const std::map<int, double>& distribution, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                               bool first_step, int last_time,
                               const TrackingConfig& config) {
    Node source_node = {0, 0}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    DiGraph selected_graph;
    selected_graph.add_node(source_node);
    int nb_to_optimum = 1 << config.graph_depth; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Get prev paths and their alpha/k values (all fixed) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Path> prev_paths;
    std::map<Path, double> alpha_values, k_values; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    std::map<Path, int> hashed_prev_next;

    if (!first_step) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        prev_paths = find_paths_as_list(saved_graph, source_node);
        if (config.use_nn && g_nn_models.loaded) {
            // Collect all trajectory coordinates for batched NN prediction
            std::vector<std::vector<float>> all_xs, all_ys;
            all_xs.reserve(prev_paths.size());
            all_ys.reserve(prev_paths.size());
            for (auto& pp : prev_paths) {
                std::vector<float> xs, ys;
                int start = std::max(1, (int)pp.size() - 10);
                for (int i = start; i < (int)pp.size(); i++) {
                    auto lit = locs.find(pp[i].first);
                    if (lit != locs.end() && pp[i].second < (int)lit->second.size()) {
                        xs.push_back(lit->second[pp[i].second][0]);
                        ys.push_back(lit->second[pp[i].second][1]);
                    }
                }
                all_xs.push_back(std::move(xs));
                all_ys.push_back(std::move(ys));
            }
            // Batched alpha + k prediction (few ONNX calls instead of N) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
            auto alpha_batch = predict_alpha_nn_batch(g_nn_models, all_xs, all_ys);
            auto k_batch = predict_k_nn_batch(g_nn_models, all_xs, all_ys);
            for (int i = 0; i < (int)prev_paths.size(); i++) {
                alpha_values[prev_paths[i]] = alpha_batch[i];
                k_values[prev_paths[i]] = k_batch[i];
            } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        } else {
            for (auto& pp : prev_paths) {
                alpha_values[pp] = config.init_alpha;
                k_values[pp] = config.init_k;
            }
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    auto gen_result = generate_next_paths(next_graph, final_graph_node_set_hashed,
                                          locs, next_times, distribution, source_node);
    next_graph = gen_result.graph;
    auto& last_nodes = gen_result.last_nodes;

    // Split into connected subgraphs // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto subgraphs = split_to_subgraphs(next_graph, source_node);

    for (int sub_idx = 0; sub_idx < (int)subgraphs.size(); sub_idx++) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        auto& sub_graph_ = subgraphs[sub_idx];
        // Phase 1: try NB_TO_OPTIMUM alternatives // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        std::vector<double> cost_sums(nb_to_optimum, -1e-5);

        for (int lowest_idx = 0; lowest_idx < nb_to_optimum; lowest_idx++) {
            DiGraph sub_copy = sub_graph_.copy();
            DiGraph tmp_graph;
            tmp_graph.add_node(source_node);
            std::map<Path, int> hpn_copy = hashed_prev_next;

            greedy_assign_pass(sub_copy, tmp_graph, source_node, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 21:30
                              final_graph_node_set_hashed, locs, next_times,
                              distribution, first_step, last_time, config,
                              first_step ? nullptr : &prev_paths,
                              first_step ? nullptr : &alpha_values,
                              first_step ? nullptr : &k_values,
                              hpn_copy, lowest_idx, cost_sums[lowest_idx],
                              last_nodes, true); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        }

        // Find best starting index — strict comparison like Python's np.argmin // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        for (auto& cs : cost_sums) { if (cs < 0) cs = 99999.0; }
        int lowest_cost_idx = 0;
        for (int i = 1; i < (int)cost_sums.size(); i++) {
            if (cost_sums[i] < cost_sums[lowest_cost_idx]) lowest_cost_idx = i;
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

        // Phase 2: rebuild with best starting index into selected_graph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        DiGraph sub_copy2 = sub_graph_.copy();
        std::map<Path, int> hpn2 = hashed_prev_next;
        double dummy_cost = -1e-5;
        greedy_assign_pass(sub_copy2, selected_graph, source_node, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                          final_graph_node_set_hashed, locs, next_times,
                          distribution, first_step, last_time, config,
                          first_step ? nullptr : &prev_paths,
                          first_step ? nullptr : &alpha_values,
                          first_step ? nullptr : &k_values,
                          hpn2, lowest_cost_idx, dummy_cost,
                          last_nodes, false); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
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

std::vector<TrajectoryObj> forecast(const Localizations& locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                                   const std::vector<int>& t_avail_steps,
                                   const std::map<int, double>& distribution,
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

        if (config.verbose && !selected_time_steps.empty() && selected_time_steps[0] % 10 == 0) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
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
                double jd = std::sqrt( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
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

// Trajectory color table matching Python's np.random.default_rng(idx).integers(0, 256, 3) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// Loaded from models/traj_colors.bin (pre-generated by Python)
static std::vector<std::array<uint8_t, 3>> g_traj_colors;
static bool g_traj_colors_loaded = false;

static bool load_traj_colors(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    int n = (int)(sz / 3);
    g_traj_colors.resize(n);
    for (int i = 0; i < n; i++) {
        uint8_t rgb[3];
        if (fread(rgb, 1, 3, f) != 3) { fclose(f); return false; }
        g_traj_colors[i] = {rgb[0], rgb[1], rgb[2]};
    }
    fclose(f);
    g_traj_colors_loaded = true;
    return true;
}

static std::array<uint8_t, 3> traj_color(int idx) {
    if (g_traj_colors_loaded && idx >= 0 && idx < (int)g_traj_colors.size()) {
        return g_traj_colors[idx];
    }
    // Fallback: splitmix64 hash
    uint64_t x = (uint64_t)idx;
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    x = x ^ (x >> 31);
    return {(uint8_t)(x & 0xFF), (uint8_t)((x >> 8) & 0xFF), (uint8_t)((x >> 16) & 0xFF)};
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

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
// H-K (diffusion) output: compute and write H, K per trajectory
// ============================================================

void write_hk_csv(const std::string& path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                  const std::vector<TrajectoryObj>& trajectories,
                  const Localizations& locs,
                  bool use_nn,
                  std::vector<double>& out_H,
                  std::vector<double>& out_K) {
    out_H.clear(); out_K.clear();
    std::ofstream f(path);
    if (!f.is_open()) {
        std::cerr << "Error: cannot write to " << path << std::endl;
        return;
    }
    f << std::setprecision(17); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    f << "traj_idx,H,K\n";

    // Collect all trajectory coordinates for batched prediction
    std::vector<std::vector<float>> all_xs, all_ys;
    all_xs.reserve(trajectories.size());
    all_ys.reserve(trajectories.size());
    for (const auto& traj : trajectories) {
        std::vector<float> xs, ys;
        for (const auto& [frame, idx] : traj.tuples) {
            const auto& pos = locs.at(frame)[idx];
            xs.push_back((float)pos[0]);
            ys.push_back((float)pos[1]);
        }
        all_xs.push_back(std::move(xs));
        all_ys.push_back(std::move(ys));
    }

    // Batched alpha + k prediction (few ONNX calls instead of N) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    auto alpha_batch = predict_alpha_nn_batch(g_nn_models, all_xs, all_ys);
    auto k_batch = predict_k_nn_batch(g_nn_models, all_xs, all_ys);

    for (int i = 0; i < (int)trajectories.size(); i++) {
        float alpha = alpha_batch[i];
        float log_k = k_batch[i]; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        double H = (double)alpha / 2.0;
        double K = std::pow(10.0, (double)log_k);
        f << trajectories[i].index << "," << H << "," << K << "\n";
        out_H.push_back(H);
        out_K.push_back(K);
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Seaborn "mako" colormap (256 entries, sampled from seaborn) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
static void mako_colormap(double t, uint8_t& r, uint8_t& g, uint8_t& b) {
    // t in [0,1]. Approximation of mako: dark purple → teal → yellow-green
    // Key stops sampled from seaborn mako:
    //   0.00: (11, 10, 36)     dark purple
    //   0.15: (35, 23, 90)     deep purple
    //   0.30: (62, 42, 127)    purple
    //   0.45: (66, 78, 137)    blue-purple
    //   0.55: (52, 115, 132)   teal
    //   0.65: (40, 152, 127)   green-teal
    //   0.75: (62, 189, 119)   green
    //   0.85: (130, 219, 120)  light green
    //   0.95: (210, 245, 165)  yellow-green
    //   1.00: (222, 249, 183)  pale yellow
    struct Stop { double pos; double r, g, b; };
    static const Stop stops[] = {
        {0.00,  11, 10, 36},
        {0.15,  35, 23, 90},
        {0.30,  62, 42,127},
        {0.45,  66, 78,137},
        {0.55,  52,115,132},
        {0.65,  40,152,127},
        {0.75,  62,189,119},
        {0.85, 130,219,120},
        {0.95, 210,245,165},
        {1.00, 222,249,183},
    };
    static const int n = sizeof(stops)/sizeof(stops[0]);
    if (t <= 0) { r = (uint8_t)stops[0].r; g = (uint8_t)stops[0].g; b = (uint8_t)stops[0].b; return; }
    if (t >= 1) { r = (uint8_t)stops[n-1].r; g = (uint8_t)stops[n-1].g; b = (uint8_t)stops[n-1].b; return; }
    for (int i = 0; i < n - 1; i++) {
        if (t <= stops[i+1].pos) {
            double f = (t - stops[i].pos) / (stops[i+1].pos - stops[i].pos);
            r = (uint8_t)(stops[i].r + f * (stops[i+1].r - stops[i].r) + 0.5);
            g = (uint8_t)(stops[i].g + f * (stops[i+1].g - stops[i].g) + 0.5);
            b = (uint8_t)(stops[i].b + f * (stops[i+1].b - stops[i].b) + 0.5);
            return;
        }
    }
    r = (uint8_t)stops[n-1].r; g = (uint8_t)stops[n-1].g; b = (uint8_t)stops[n-1].b;
}

// ---- Bitmap font for plot text (6x10 glyphs) ---- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// Each glyph is 6 wide x 10 tall, stored as 10 bytes (1 bit per pixel, MSB=left)
static const uint8_t FONT_6x10[][10] = {
    // ' ' (space)
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00,0x00},
    // '0'
    {0x78,0xCC,0xCC,0xDC,0xEC,0xCC,0xCC,0x78,0x00,0x00},
    // '1'
    {0x30,0x70,0x30,0x30,0x30,0x30,0x30,0xFC,0x00,0x00},
    // '2'
    {0x78,0xCC,0x0C,0x18,0x30,0x60,0xC0,0xFC,0x00,0x00},
    // '3'
    {0x78,0xCC,0x0C,0x38,0x0C,0x0C,0xCC,0x78,0x00,0x00},
    // '4'
    {0x1C,0x3C,0x6C,0xCC,0xFC,0x0C,0x0C,0x0C,0x00,0x00},
    // '5'
    {0xFC,0xC0,0xC0,0xF8,0x0C,0x0C,0xCC,0x78,0x00,0x00},
    // '6'
    {0x38,0x60,0xC0,0xF8,0xCC,0xCC,0xCC,0x78,0x00,0x00},
    // '7'
    {0xFC,0x0C,0x18,0x30,0x30,0x30,0x30,0x30,0x00,0x00},
    // '8'
    {0x78,0xCC,0xCC,0x78,0xCC,0xCC,0xCC,0x78,0x00,0x00},
    // '9'
    {0x78,0xCC,0xCC,0xCC,0x7C,0x0C,0x18,0x70,0x00,0x00},
    // '.'
    {0x00,0x00,0x00,0x00,0x00,0x00,0x30,0x30,0x00,0x00},
    // '-'
    {0x00,0x00,0x00,0x00,0xFC,0x00,0x00,0x00,0x00,0x00},
    // 'A'
    {0x30,0x78,0xCC,0xCC,0xFC,0xCC,0xCC,0xCC,0x00,0x00},
    // 'B' (not used but placeholder)
    {0xF8,0xCC,0xCC,0xF8,0xCC,0xCC,0xCC,0xF8,0x00,0x00},
    // 'C'
    {0x78,0xCC,0xC0,0xC0,0xC0,0xC0,0xCC,0x78,0x00,0x00},
    // 'D'
    {0xF0,0xD8,0xCC,0xCC,0xCC,0xCC,0xD8,0xF0,0x00,0x00},
    // 'E'
    {0xFC,0xC0,0xC0,0xF8,0xC0,0xC0,0xC0,0xFC,0x00,0x00},
    // 'H'
    {0xCC,0xCC,0xCC,0xFC,0xCC,0xCC,0xCC,0xCC,0x00,0x00},
    // 'K'
    {0xCC,0xD8,0xF0,0xE0,0xF0,0xD8,0xCC,0xCC,0x00,0x00},
    // '('
    {0x18,0x30,0x60,0x60,0x60,0x60,0x30,0x18,0x00,0x00},
    // ')'
    {0x60,0x30,0x18,0x18,0x18,0x18,0x30,0x60,0x00,0x00},
    // 'a'
    {0x00,0x00,0x78,0x0C,0x7C,0xCC,0xCC,0x76,0x00,0x00},
    // 'c'
    {0x00,0x00,0x78,0xCC,0xC0,0xC0,0xCC,0x78,0x00,0x00},
    // 'd'
    {0x0C,0x0C,0x7C,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00},
    // 'e'
    {0x00,0x00,0x78,0xCC,0xFC,0xC0,0xCC,0x78,0x00,0x00},
    // 'f'
    {0x38,0x6C,0x60,0xF0,0x60,0x60,0x60,0x60,0x00,0x00},
    // 'g'
    {0x00,0x00,0x76,0xCC,0xCC,0x7C,0x0C,0xCC,0x78,0x00},
    // 'h' (not used)
    {0xC0,0xC0,0xF8,0xCC,0xCC,0xCC,0xCC,0xCC,0x00,0x00},
    // 'i'
    {0x30,0x00,0x70,0x30,0x30,0x30,0x30,0x78,0x00,0x00},
    // 'l'
    {0x70,0x30,0x30,0x30,0x30,0x30,0x30,0x78,0x00,0x00},
    // 'n'
    {0x00,0x00,0xB8,0xCC,0xCC,0xCC,0xCC,0xCC,0x00,0x00},
    // 'o'
    {0x00,0x00,0x78,0xCC,0xCC,0xCC,0xCC,0x78,0x00,0x00},
    // 'p'
    {0x00,0x00,0xF8,0xCC,0xCC,0xF8,0xC0,0xC0,0xC0,0x00},
    // 'r'
    {0x00,0x00,0xB8,0xCC,0xC0,0xC0,0xC0,0xC0,0x00,0x00},
    // 's'
    {0x00,0x00,0x78,0xC0,0x78,0x0C,0x0C,0xF8,0x00,0x00},
    // 't'
    {0x30,0x30,0xFC,0x30,0x30,0x30,0x34,0x18,0x00,0x00},
    // 'u'
    {0x00,0x00,0xCC,0xCC,0xCC,0xCC,0xCC,0x76,0x00,0x00},
    // 'x'
    {0x00,0x00,0xCC,0xCC,0x78,0x30,0x78,0xCC,0x00,0x00},
    // 'm'
    {0x00,0x00,0xCC,0xFE,0xFE,0xD6,0xC6,0xC6,0x00,0x00},
    // 'j'
    {0x0C,0x00,0x1C,0x0C,0x0C,0x0C,0xCC,0xCC,0x78,0x00},
};
// Map character to glyph index
static int font_index(char c) {
    if (c == ' ') return 0;
    if (c >= '0' && c <= '9') return c - '0' + 1;
    if (c == '.') return 11;
    if (c == '-') return 12;
    switch(c) {
        case 'A': return 13; case 'B': return 14; case 'C': return 15;
        case 'D': return 16; case 'E': return 17; case 'H': return 18;
        case 'K': return 19; case '(': return 20; case ')': return 21;
        case 'a': return 22; case 'c': return 23; case 'd': return 24;
        case 'e': return 25; case 'f': return 26; case 'g': return 27;
        case 'h': return 28; case 'i': return 29; case 'l': return 30;
        case 'n': return 31; case 'o': return 32; case 'p': return 33;
        case 'r': return 34; case 's': return 35; case 't': return 36;
        case 'u': return 37; case 'x': return 38;
        case 'm': return 39; case 'j': return 40;
        default: return 0;
    }
}

// Draw a character at (x,y) top-left, with scale factor, onto RGB image
static void draw_char(std::vector<uint8_t>& img, int img_w, int img_h,
                      int x, int y, char c, int scale, uint8_t cr, uint8_t cg, uint8_t cb) {
    int gi = font_index(c);
    for (int row = 0; row < 10; row++) {
        uint8_t bits = FONT_6x10[gi][row];
        for (int col = 0; col < 6; col++) {
            if (bits & (0x80 >> col)) {
                for (int sy = 0; sy < scale; sy++) {
                    for (int sx = 0; sx < scale; sx++) {
                        int px = x + col * scale + sx;
                        int py = y + row * scale + sy;
                        if (px >= 0 && px < img_w && py >= 0 && py < img_h) {
                            int idx = (py * img_w + px) * 3;
                            img[idx] = cr; img[idx+1] = cg; img[idx+2] = cb;
                        }
                    }
                }
            }
        }
    }
}

// Draw a string horizontally at (x,y)
static void draw_text(std::vector<uint8_t>& img, int img_w, int img_h,
                      int x, int y, const std::string& text, int scale,
                      uint8_t r = 0, uint8_t g = 0, uint8_t b = 0) {
    for (int i = 0; i < (int)text.size(); i++) {
        draw_char(img, img_w, img_h, x + i * 6 * scale, y, text[i], scale, r, g, b);
    }
}

// Draw a string vertically (rotated 90 CW: tilt head right to read) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// (x, y) = top-left of bounding box; text reads top-to-bottom
// Rotated glyph is 10px wide, 6px tall (before scale)
static void draw_text_vertical(std::vector<uint8_t>& img, int img_w, int img_h,
                               int x, int y, const std::string& text, int scale,
                               uint8_t cr = 0, uint8_t cg = 0, uint8_t cb = 0) {
    int nch = (int)text.size();
    for (int i = 0; i < nch; i++) {
        int ci = nch - 1 - i; // first char at bottom, last at top
        int gi = font_index(text[ci]);
        // 90 CW: (col, row) → screen (row, 5-col)
        for (int row = 0; row < 10; row++) {
            uint8_t bits = FONT_6x10[gi][row];
            for (int col = 0; col < 6; col++) {
                if (bits & (0x80 >> col)) {
                    for (int sy = 0; sy < scale; sy++) {
                        for (int sx = 0; sx < scale; sx++) {
                            int px = x + row * scale + sy;
                            int py = y + i * 6 * scale + (5 - col) * scale + sx;
                            if (px >= 0 && px < img_w && py >= 0 && py < img_h) {
                                int idx = (py * img_w + px) * 3;
                                img[idx] = cr; img[idx+1] = cg; img[idx+2] = cb;
                            }
                        }
                    }
                }
            }
        }
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Format a number for tick labels
static std::string fmt_tick(double val, int decimals = 1) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.*f", decimals, val);
    return buf;
}

static std::string fmt_sci(double val) {
    // Format as e.g. "0.01", "0.1", "1", "10", "100"
    if (val >= 1.0 && val < 10.0) return fmt_tick(val, 0);
    if (val >= 10.0) return fmt_tick(val, 0);
    char buf[32];
    snprintf(buf, sizeof(buf), "%.2g", val);
    return buf;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

void make_hk_distribution_image(const std::string& path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                const std::vector<double>& H_vals,
                                const std::vector<double>& K_vals) {
#ifdef USE_LIBPNG
    if (H_vals.empty() || H_vals.size() != K_vals.size()) return;
    int n = (int)H_vals.size();
    if (n < 2) return;

    // Image dimensions
    const int IMG_W = 900, IMG_H = 900;
    // Margins: left for Y-label+ticks, bottom for X-label+ticks, top for title
    const int margin_left = 110, margin_right = 30, margin_top = 50, margin_bottom = 80;
    const int pw = IMG_W - margin_left - margin_right;
    const int ph = IMG_H - margin_top - margin_bottom;

    // Axis ranges
    const double h_min = 0.0, h_max = 1.0;
    double k_min_val = K_vals[0], k_max_val = K_vals[0];
    for (double k : K_vals) {
        if (k > 0) {
            if (k < k_min_val || k_min_val <= 0) k_min_val = k;
            if (k > k_max_val) k_max_val = k;
        }
    }
    if (k_min_val <= 0) k_min_val = 1e-3;
    if (k_max_val <= k_min_val) k_max_val = k_min_val * 100;
    double log_k_min = std::floor(std::log10(k_min_val) - 0.5);
    double log_k_max = std::ceil(std::log10(k_max_val) + 0.5);

    // 2D KDE computation
    const int grid_size = 200;
    std::vector<double> density(grid_size * grid_size, 0.0);

    double h_mean = 0, lk_mean = 0;
    std::vector<double> log_ks(n);
    for (int i = 0; i < n; i++) {
        h_mean += H_vals[i];
        log_ks[i] = std::log10(std::max(K_vals[i], 1e-10));
        lk_mean += log_ks[i];
    }
    h_mean /= n; lk_mean /= n;
    double h_var = 0, lk_var = 0;
    for (int i = 0; i < n; i++) {
        h_var += (H_vals[i] - h_mean) * (H_vals[i] - h_mean);
        lk_var += (log_ks[i] - lk_mean) * (log_ks[i] - lk_mean);
    }
    h_var /= (n - 1); lk_var /= (n - 1);
    double bw_factor = std::pow((double)n, -1.0/6.0);
    double bw_h = std::sqrt(h_var) * bw_factor;
    double bw_k = std::sqrt(lk_var) * bw_factor;
    if (bw_h < 1e-6) bw_h = 0.05;
    if (bw_k < 1e-6) bw_k = 0.1;

    double dh = (h_max - h_min) / grid_size;
    double dlk = (log_k_max - log_k_min) / grid_size;

    for (int i = 0; i < n; i++) {
        double hi = H_vals[i];
        double lki = log_ks[i];
        int gx_lo = std::max(0, (int)((hi - 4*bw_h - h_min) / dh));
        int gx_hi = std::min(grid_size - 1, (int)((hi + 4*bw_h - h_min) / dh));
        int gy_lo = std::max(0, (int)((lki - 4*bw_k - log_k_min) / dlk));
        int gy_hi = std::min(grid_size - 1, (int)((lki + 4*bw_k - log_k_min) / dlk));
        for (int gy = gy_lo; gy <= gy_hi; gy++) {
            double lk_grid = log_k_min + (gy + 0.5) * dlk;
            double dy = (lk_grid - lki) / bw_k;
            double ey = std::exp(-0.5 * dy * dy);
            for (int gx = gx_lo; gx <= gx_hi; gx++) {
                double h_grid = h_min + (gx + 0.5) * dh;
                double dx = (h_grid - hi) / bw_h;
                density[gy * grid_size + gx] += std::exp(-0.5 * dx * dx) * ey;
            }
        }
    }

    double max_density = 0;
    for (double d : density) if (d > max_density) max_density = d;
    if (max_density <= 0) max_density = 1;

    // Create RGB image (white background)
    std::vector<uint8_t> img(IMG_H * IMG_W * 3, 255);

    // Fill plot area with mako(0) background
    uint8_t bg_r, bg_g, bg_b;
    mako_colormap(0.0, bg_r, bg_g, bg_b);
    for (int py = margin_top; py < IMG_H - margin_bottom; py++) {
        for (int px = margin_left; px < IMG_W - margin_right; px++) {
            int idx = (py * IMG_W + px) * 3;
            img[idx] = bg_r; img[idx+1] = bg_g; img[idx+2] = bg_b;
        }
    }

    // Render KDE
    for (int gy = 0; gy < grid_size; gy++) {
        int py_start = margin_top + (int)((1.0 - (double)(gy + 1) / grid_size) * ph);
        int py_end = margin_top + (int)((1.0 - (double)gy / grid_size) * ph);
        for (int gx = 0; gx < grid_size; gx++) {
            double d = density[gy * grid_size + gx] / max_density;
            if (d < 1e-6) continue;
            uint8_t cr, cg, cb;
            mako_colormap(d, cr, cg, cb);
            int px_start = margin_left + (int)((double)gx / grid_size * pw);
            int px_end = margin_left + (int)((double)(gx + 1) / grid_size * pw);
            for (int py = py_start; py < py_end && py < IMG_H - margin_bottom; py++) {
                for (int px = px_start; px < px_end && px < IMG_W - margin_right; px++) {
                    if (py >= margin_top && px >= margin_left) {
                        int idx = (py * IMG_W + px) * 3;
                        img[idx] = cr; img[idx+1] = cg; img[idx+2] = cb;
                    }
                }
            }
        }
    }

    // Draw axes (black border)
    for (int px = margin_left; px <= IMG_W - margin_right; px++) {
        for (int t = 0; t < 2; t++) {
            int idx_top = ((margin_top + t) * IMG_W + px) * 3;
            int idx_bot = ((IMG_H - margin_bottom - t) * IMG_W + px) * 3;
            img[idx_top] = img[idx_top+1] = img[idx_top+2] = 0;
            img[idx_bot] = img[idx_bot+1] = img[idx_bot+2] = 0;
        }
    }
    for (int py = margin_top; py <= IMG_H - margin_bottom; py++) {
        for (int t = 0; t < 2; t++) {
            int idx_l = (py * IMG_W + margin_left + t) * 3;
            int idx_r = (py * IMG_W + IMG_W - margin_right - t) * 3;
            img[idx_l] = img[idx_l+1] = img[idx_l+2] = 0;
            img[idx_r] = img[idx_r+1] = img[idx_r+2] = 0;
        }
    }

    const int scale = 2; // font scale
    const int tick_len = 6;

    // X-axis ticks (H: 0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
    for (int ti = 0; ti <= 5; ti++) {
        double hval = ti * 0.2;
        int px = margin_left + (int)(hval / (h_max - h_min) * pw);
        // Tick mark
        for (int t = 0; t < tick_len; t++) {
            if (px >= 0 && px < IMG_W) {
                int idx = ((IMG_H - margin_bottom + t) * IMG_W + px) * 3;
                img[idx] = img[idx+1] = img[idx+2] = 0;
            }
        }
        // Tick label
        std::string label = fmt_tick(hval, 1);
        int text_x = px - (int)label.size() * 6 * scale / 2;
        draw_text(img, IMG_W, IMG_H, text_x, IMG_H - margin_bottom + tick_len + 2, label, scale);
    }

    // Y-axis ticks (K: log scale — powers of 10)
    for (int exp = (int)log_k_min; exp <= (int)log_k_max; exp++) {
        double lk = (double)exp;
        double frac = (lk - log_k_min) / (log_k_max - log_k_min);
        if (frac < 0 || frac > 1) continue;
        int py = margin_top + (int)((1.0 - frac) * ph);
        // Tick mark
        for (int t = 0; t < tick_len; t++) {
            if (py >= 0 && py < IMG_H) {
                int idx = (py * IMG_W + margin_left - t) * 3;
                img[idx] = img[idx+1] = img[idx+2] = 0;
            }
        }
        // Tick label
        double kval = std::pow(10.0, exp);
        std::string label = fmt_sci(kval);
        int text_x = margin_left - tick_len - 4 - (int)label.size() * 6 * scale;
        int text_y = py - 5 * scale; // center vertically
        draw_text(img, IMG_W, IMG_H, text_x, text_y, label, scale);
    }

    // X-axis label: "H (Hurst exponent)"
    {
        std::string xlabel = "H (Hurst exponent)";
        int text_x = margin_left + pw / 2 - (int)xlabel.size() * 6 * scale / 2;
        int text_y = IMG_H - margin_bottom + tick_len + 2 + 10 * scale + 8;
        draw_text(img, IMG_W, IMG_H, text_x, text_y, xlabel, scale);
    }

    // Y-axis label: "K (generalised diffusion coefficient)" — vertical (90 CW) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    {
        std::string ylabel = "K (generalised diffusion coefficient, px^2/frame^2H)";  // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        int text_height = (int)ylabel.size() * 6 * scale;
        int center_y = margin_top + ph / 2;
        int text_x = 4; // left side
        int text_start_y = center_y - text_height / 2;
        draw_text_vertical(img, IMG_W, IMG_H, text_x, text_start_y, ylabel, scale);
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    // Title: "Estimated cluster of trajectories"
    {
        std::string title = "Estimated cluster of trajectories";
        int text_x = IMG_W / 2 - (int)title.size() * 6 * scale / 2;
        int text_y = 10;
        draw_text(img, IMG_W, IMG_H, text_x, text_y, title, scale);
    }

    // Scatter points: yellow 'x' markers for each (H, K) data point // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    {
        const int marker_half = 3; // half-size of x marker in pixels
        const uint8_t mr = 255, mg = 220, mb = 50; // yellow
        for (int i = 0; i < n; i++) {
            double hi = H_vals[i];
            double lki = std::log10(std::max(K_vals[i], 1e-10));
            double frac_x = (hi - h_min) / (h_max - h_min);
            double frac_y = (lki - log_k_min) / (log_k_max - log_k_min);
            if (frac_x < 0 || frac_x > 1 || frac_y < 0 || frac_y > 1) continue;
            int cx = margin_left + (int)(frac_x * pw);
            int cy = margin_top + (int)((1.0 - frac_y) * ph);
            // Draw X shape
            for (int d = -marker_half; d <= marker_half; d++) {
                // diagonal 1
                int px1 = cx + d, py1 = cy + d;
                if (px1 >= margin_left && px1 < IMG_W - margin_right &&
                    py1 >= margin_top && py1 < IMG_H - margin_bottom) {
                    int idx = (py1 * IMG_W + px1) * 3;
                    img[idx] = mr; img[idx+1] = mg; img[idx+2] = mb;
                }
                // diagonal 2
                int px2 = cx + d, py2 = cy - d;
                if (px2 >= margin_left && px2 < IMG_W - margin_right &&
                    py2 >= margin_top && py2 < IMG_H - margin_bottom) {
                    int idx = (py2 * IMG_W + px2) * 3;
                    img[idx] = mr; img[idx+1] = mg; img[idx+2] = mb;
                }
            }
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    // Write PNG
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) { std::cerr << "Error: cannot write " << path << std::endl; return; }
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    png_init_io(png, fp);
    png_set_IHDR(png, info, IMG_W, IMG_H, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (int r = 0; r < IMG_H; r++) {
        png_write_row(png, &img[r * IMG_W * 3]);
    }
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
#else
    (void)path; (void)H_vals; (void)K_vals;
    std::cerr << "Warning: libpng not available, skipping H-K distribution image" << std::endl;
#endif
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// ============================================================
// run_tracking: top-level entry point
// ============================================================

// Helper: get directory of current executable (Linux, macOS, Windows) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
static std::string get_exe_dir() {
    char buf[4096];
#ifdef _WIN32
    DWORD len = GetModuleFileNameA(NULL, buf, sizeof(buf));
    if (len > 0 && len < sizeof(buf)) {
        std::string p(buf);
        auto sl = p.rfind('\\');
        if (sl == std::string::npos) sl = p.rfind('/');
        if (sl != std::string::npos) return p.substr(0, sl);
    }
#elif defined(__APPLE__)
    uint32_t size = sizeof(buf);
    if (_NSGetExecutablePath(buf, &size) == 0) {
        std::string p(buf);
        auto sl = p.rfind('/');
        if (sl != std::string::npos) return p.substr(0, sl);
    }
#else
    ssize_t len = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (len > 0) {
        buf[len] = '\0';
        std::string p(buf);
        auto sl = p.rfind('/');
        if (sl != std::string::npos) return p.substr(0, sl);
    }
#endif
    return "";
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Helper: find last path separator (handles both '/' and '\\' for Windows) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
static std::string::size_type rfind_path_sep(const std::string& path) {
    auto fwd = path.rfind('/');
    auto bwd = path.rfind('\\');
    if (fwd == std::string::npos) return bwd;
    if (bwd == std::string::npos) return fwd;
    return std::max(fwd, bwd);
}

// Helper: platform path separator
#ifdef _WIN32
static const char PATH_SEP = '\\';
#else
static const char PATH_SEP = '/';
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Helper: build model search paths (CWD, parent, loc_csv dir, exe dir) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
static std::vector<std::string> model_search_dirs(const std::string& loc_csv_path) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    std::string sep(1, PATH_SEP);
    std::vector<std::string> dirs = {"models", ".." + sep + "models", ".." + sep + ".." + sep + "models"}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    auto slash = rfind_path_sep(loc_csv_path);
    if (slash != std::string::npos)
        dirs.push_back(loc_csv_path.substr(0, slash) + sep + ".." + sep + "models");
    std::string ed = get_exe_dir();
    if (!ed.empty()) {
        dirs.push_back(ed + sep + "models");
        dirs.push_back(ed + sep + ".." + sep + "models");
        dirs.push_back(ed + sep + ".." + sep + ".." + sep + "models"); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    }
    return dirs;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

// Helper: derive output base name from tiff_path or loc_csv_path // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// Python uses: input_video_path.split("/")[-1].split(".tif")[0]
static std::string derive_output_base(const TrackingConfig& config, const std::string& loc_csv_path) {
    std::string src = config.tiff_path.empty() ? loc_csv_path : config.tiff_path;
    auto slash = rfind_path_sep(src);
    std::string fname = (slash != std::string::npos) ? src.substr(slash + 1) : src;
    // Strip .tif/.tiff extension
    auto tif_pos = fname.find(".tif");
    if (tif_pos != std::string::npos) return fname.substr(0, tif_pos);
    // Strip _loc.csv for loc-based naming
    auto loc_pos = fname.find("_loc.csv");
    if (loc_pos != std::string::npos) return fname.substr(0, loc_pos);
    // Strip .csv
    auto csv_pos = fname.rfind(".csv");
    if (csv_pos != std::string::npos) return fname.substr(0, csv_pos);
    return fname;
}

bool run_tracking(const std::string& loc_csv_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
                  const std::string& output_path,
                  int nb_frames,
                  TrackingConfig config) {
    if (config.verbose) std::cerr << "Reading localization CSV: " << loc_csv_path << std::endl;
    auto loc = read_localization_csv(loc_csv_path, nb_frames);
    if (loc.empty()) {
        std::cerr << "Error: no localization data" << std::endl;
        return false;
    }

    // Get available time steps (frames with particles)
    auto all_steps = get_sorted_time_steps(loc);
    std::vector<int> t_avail_steps;
    for (int t : all_steps) {
        if (!loc[t].empty()) t_avail_steps.push_back(t);
    }

    if (t_avail_steps.size() < 2) {
        std::cerr << "Error: need at least 2 frames with particles" << std::endl;
        return false;
    }

    int time_forecast = std::max(1, std::min(5, config.graph_depth));

    // Segmentation
    if (config.verbose) std::cerr << "Running segmentation..." << std::endl;
    auto seg = segmentation(loc, t_avail_steps, time_forecast);

    // Approximation (jump thresholds)
    if (config.verbose) std::cerr << "Computing jump thresholds..." << std::endl;
    auto max_jumps = approximation(seg.dist_x, seg.dist_y, seg.dist_z,
                                   time_forecast, config.jump_threshold);

    if (config.verbose) {
        std::cerr << "Jump threshold: " << max_jumps[0] << " px" << std::endl;
    }

    // Build empirical PDF
    build_emp_pdf(seg.seg_distribution[0], 40, 20.0f);

    // Count particles per frame
    if (config.verbose) {
        int total = 0;
        for (const auto& [t, ps] : loc) total += (int)ps.size();
        double mean_nb = (double)total / all_steps.size();
        std::cerr << "Mean particles/frame: " << mean_nb << std::endl;
    }

    // Build model search directories
    auto mdirs = model_search_dirs(loc_csv_path); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    std::string psep(1, PATH_SEP); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    // Load qt_99 data for abnormal detection
    {
        for (const auto& md : mdirs) {
            if (load_qt_data(md + psep + "qt_99.bin")) {
                if (config.verbose) std::cerr << "Loaded qt_99 data from " << md << psep << "qt_99.bin" << std::endl;
                break;
            }
        }
        if (!g_qt_loaded) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
            std::cerr << "Error: qt_99.bin is required to run FreeTrace tracking." << std::endl;
            std::cerr << "Searched in:";
            for (const auto& md : mdirs) std::cerr << " " << md << psep << "qt_99.bin";
            std::cerr << std::endl;
            return false;
        }
    }

    // Load trajectory color table (required) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    {
        for (const auto& md : mdirs) {
            if (load_traj_colors(md + psep + "traj_colors.bin")) break;
        }
        if (!g_traj_colors_loaded) {
            std::cerr << "Error: traj_colors.bin is required to run FreeTrace tracking." << std::endl;
            std::cerr << "Searched in:";
            for (const auto& md : mdirs) std::cerr << " " << md << psep << "traj_colors.bin";
            std::cerr << std::endl;
            return false;
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    // Load NN models if requested (fBm mode) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    if (config.use_nn) {
        for (const auto& md : mdirs) {
            if (load_nn_models(g_nn_models, md)) {
                if (config.verbose) std::cerr << "Loaded NN models from " << md << std::endl;
                break;
            }
        }
        if (!g_nn_models.loaded) {
            std::cerr << "Warning: NN models (reg_model_*.onnx, reg_k_model.onnx, k_model_weights.bin) not found." << std::endl;
            std::cerr << "  Falling back to fBm=OFF mode (fixed default alpha and K)." << std::endl;
            config.use_nn = false;
            config.fbm_mode = false;
            config.hk_output = false;
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    // Run forecast
    if (config.verbose) std::cerr << "Starting trajectory inference..." << std::endl;
    auto trajectories = forecast(loc, t_avail_steps, max_jumps, nb_frames, config);

    if (config.verbose) {
        std::cerr << "Found " << trajectories.size() << " trajectories" << std::endl;
    }

    // PostProcessing (optional)
    if (config.post_process) {
        if (config.verbose) std::cerr << "Running post-processing..." << std::endl;
        trajectories = post_processing(trajectories, loc, config.cutoff, config.verbose);
        if (config.verbose) std::cerr << "After post-processing: " << trajectories.size() << " trajectories" << std::endl;
    }

    // Derive output base name (from tiff_path if available, else loc_csv)
    std::string base = derive_output_base(config, loc_csv_path); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    std::string sep(1, PATH_SEP); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    // Write trajectory CSV
    std::string out_csv = output_path + sep + base + "_traces.csv";
    write_trajectory_csv(out_csv, trajectories, loc);
    if (config.verbose) std::cerr << "Written: " << out_csv << std::endl;

    // Trajectory visualization image
    int img_rows = config.img_rows;
    int img_cols = config.img_cols;
    if (img_rows <= 0 || img_cols <= 0) {
        double max_x = 0, max_y = 0;
        for (const auto& [t, particles] : loc) {
            for (const auto& p : particles) {
                if (p[0] > max_x) max_x = p[0];
                if (p[1] > max_y) max_y = p[1];
            }
        }
        img_rows = (int)std::ceil(max_y) + 2;
        img_cols = (int)std::ceil(max_x) + 2;
    }
    std::string out_img = output_path + sep + base + "_traces.png";
    make_trajectory_image(out_img, trajectories, loc, img_rows, img_cols);
    if (config.verbose) std::cerr << "Written: " << out_img << " (" << img_cols << "x" << img_rows << " * upscale)" << std::endl;

    // H-K (diffusion) output when fBm mode is on // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    if (config.hk_output) {
        std::vector<double> H_vals, K_vals;
        std::string out_hk_csv = output_path + sep + base + "_diffusion.csv";
        write_hk_csv(out_hk_csv, trajectories, loc, config.use_nn, H_vals, K_vals);
        if (config.verbose) std::cerr << "Written: " << out_hk_csv << std::endl;

        std::string out_hk_img = output_path + sep + base + "_diffusion_distribution.png";
        make_hk_distribution_image(out_hk_img, H_vals, K_vals);
        if (config.verbose) std::cerr << "Written: " << out_hk_img << std::endl;
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

    // Cleanup NN models
    if (g_nn_models.loaded) free_nn_models(g_nn_models);

    return true;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

} // namespace freetrace
