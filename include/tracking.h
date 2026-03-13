#pragma once // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <vector>
#include <map>
#include <set>
#include <array>
#include <string>
#include <utility>
#include <cmath>

namespace freetrace {

// Node in tracking graph: (frame, particle_index) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
using Node = std::pair<int, int>;

// Edge data stored on graph edges // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct EdgeData {
    float jump_d = -1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool terminal = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};

// Directed graph replacing NetworkX DiGraph // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
class DiGraph {
public:
    void add_node(const Node& n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    void add_edge(const Node& from, const Node& to, const EdgeData& data = {}); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    void remove_node(const Node& n); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    void remove_edge(const Node& from, const Node& to); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool has_node(const Node& n) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool has_edge(const Node& from, const Node& to) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool has_path(const Node& from, const Node& to) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> successors(const Node& n) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> predecessors(const Node& n) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    const EdgeData& get_edge_data(const Node& from, const Node& to) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    const std::set<Node>& get_nodes() const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    const std::vector<Node>& get_nodes_ordered() const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    size_t size() const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    DiGraph copy() const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> ancestors(const Node& n) const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

private:
    std::set<Node> nodes_; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    std::vector<Node> nodes_ordered_; // Insertion order tracking // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    // Insertion-ordered adjacency lists to match Python dict insertion order
    std::map<Node, std::vector<std::pair<Node, EdgeData>>> fwd_; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    std::map<Node, std::vector<Node>> rev_; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
};

// Path: sequence of nodes // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
using Path = std::vector<Node>;

// Trajectory object // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct TrajectoryObj {
    int index = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> tuples; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    void add_trajectory_tuple(int frame, int idx); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int get_trajectory_length() const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<int> get_times() const; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};

// Localization data: frame -> list of (x, y, z) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
using Localizations = std::map<int, std::vector<std::array<double, 3>>>;

// Tracking configuration // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
struct TrackingConfig {
    int graph_depth = 3; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    int cutoff = 3; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    float jump_threshold = -1.0f; // -1 = undetermined, infer from data // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float init_alpha = 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float init_k = 0.3f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int dimension = 2; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float loc_precision_err = 1.0f; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool verbose = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool post_process = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool use_nn = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool fbm_mode = false; // fBm mode: enable NN + HK output // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    bool hk_output = false; // produce H-K diffusion CSV and distribution image // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    int img_rows = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11 (0 = auto from loc data)
    int img_cols = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::string tiff_path; // path to TIFF for image dimensions and output naming // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
};

// --- I/O --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
Localizations read_localization_csv(const std::string& path, int nb_frames); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
void write_trajectory_csv(const std::string& path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                          const std::vector<TrajectoryObj>& trajectories,
                          const Localizations& locs);

// --- Distance --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
double euclidean_displacement_single(const std::array<double,3>& a, const std::array<double,3>& b); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
std::vector<double> euclidean_displacement_batch(const std::vector<std::array<double,3>>& a, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                                                 const std::vector<std::array<double,3>>& b);

// --- Segmentation --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct SegmentationResult {
    std::vector<double> dist_x, dist_y, dist_z; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    std::map<int, std::vector<double>> seg_distribution; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
};
SegmentationResult segmentation(const Localizations& loc, const std::vector<int>& time_steps, int lag); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// --- Greedy matching --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct GreedyResult {
    std::vector<double> x_dist, y_dist, z_dist, jump_dist; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
};
GreedyResult greedy_shortest(const std::vector<std::array<double,3>>& srcs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                             const std::vector<std::array<double,3>>& dests);

// --- Approximation (jump thresholds) --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
std::map<int, double> approximation(const std::vector<double>& dist_x, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                                   const std::vector<double>& dist_y,
                                   const std::vector<double>& dist_z,
                                   int time_forecast, float jump_threshold);

// --- Empirical PDF --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
void build_emp_pdf(const std::vector<double>& emp_distribution, int nb_bins = 40, float max_val = 20.0f);
double empirical_pdf_lookup(const std::array<double,3>& coords); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12

// --- Cauchy cost (tracking version with regularization) --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct CauchyResult {
    double log_pdf; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
    bool abnormal; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};
CauchyResult predict_cauchy_tracking(const std::array<double,3>& next_vec, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                     const std::array<double,3>& prev_vec,
                                     float k, float alpha,
                                     int before_lag, int lag,
                                     float precision, int dimension);

// --- Graph algorithms --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
std::vector<Path> find_paths_as_list(const DiGraph& G, const Node& source); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
void find_paths_dfs(const DiGraph& G, const Node& current, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    Path& path, std::set<Node>& seen,
                    std::vector<Path>& results);

std::vector<DiGraph> split_to_subgraphs(const DiGraph& G, const Node& source_node); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// --- Graph building --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct GenerateResult {
    DiGraph graph; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<Node> last_nodes; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};
GenerateResult generate_next_paths(DiGraph next_graph, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                   const std::set<Node>& final_graph_nodes,
                                   const Localizations& locs,
                                   const std::vector<int>& next_times,
                                   const std::map<int, double>& distribution,
                                   const Node& source_node);

// --- Terminal check --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
bool is_terminal_node(const Node& node, const Localizations& locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                      double max_jump_d, const DiGraph& selected_graph,
                      const std::set<Node>& final_graph_nodes,
                      int time_forecast);

// --- Path matching --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
const Path* match_prev_next(const std::vector<Path>& prev_paths, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                            const Path& next_path,
                            std::map<Path, int>& hashed_prev_next);

// --- Cost computation --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct PredictResult {
    std::vector<int> ab_index; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int terminal = -1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};
PredictResult predict_long_seq(const Path& next_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12
                               std::map<Path, double>& trajectories_costs,
                               const Localizations& locs,
                               float prev_alpha, float prev_k,
                               const std::vector<int>& next_times,
                               const Path* prev_path,
                               std::map<Path, int>& start_indice,
                               int last_time, float jump_threshold,
                               const DiGraph& selected_graph,
                               const std::set<Node>& final_graph_nodes,
                               int time_forecast, int dimension,
                               float loc_precision_err,
                               bool use_nn = false);

// --- Optimal graph selection --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
struct SelectResult {
    DiGraph selected_graph; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    bool has_orphan; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
};
SelectResult select_opt_graph2(const std::set<Node>& final_graph_node_set_hashed, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                               const DiGraph& saved_graph,
                               DiGraph next_graph,
                               const Localizations& locs,
                               const std::vector<int>& next_times,
                               const std::map<int, double>& distribution,
                               bool first_step, int last_time,
                               const TrackingConfig& config);

// --- Main forecast loop --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
std::vector<TrajectoryObj> forecast(const Localizations& locs, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                   const std::vector<int>& t_avail_steps,
                                   const std::map<int, double>& distribution,
                                   int image_length,
                                   const TrackingConfig& config);

// --- Trajectory visualization --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
void make_trajectory_image(const std::string& output_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                           const std::vector<TrajectoryObj>& trajectories,
                           const Localizations& locs,
                           int img_rows, int img_cols);

// --- PostProcessing --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
std::vector<TrajectoryObj> post_processing(const std::vector<TrajectoryObj>& trajectory_list, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                                            const Localizations& locs, int cutoff, bool verbose = false);

// --- H-K (diffusion) output --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
void write_hk_csv(const std::string& path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                  const std::vector<TrajectoryObj>& trajectories,
                  const Localizations& locs,
                  bool use_nn);
void make_hk_distribution_image(const std::string& path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
                                const std::vector<double>& H_vals,
                                const std::vector<double>& K_vals);

// --- Top-level tracking entry point --- // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
bool run_tracking(const std::string& loc_csv_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                  const std::string& output_path,
                  int nb_frames,
                  const TrackingConfig& config = {});

} // namespace freetrace
