// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
// Test program: reads trajectories from binary file, runs C++ NN inference,
// compares with Python predictions saved in the same file.
#include "nn_inference.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <iostream>
#include <iomanip>

int main(int argc, char* argv[]) {
    std::string models_dir = "models";
    std::string data_file = "test_nn_trajectories.bin";

    if (argc > 1) models_dir = argv[1];
    if (argc > 2) data_file = argv[2];

    // Load ONNX models
    freetrace::NNModels models;
    if (!freetrace::load_nn_models(models, models_dir)) {
        std::cerr << "Failed to load NN models from " << models_dir << std::endl;
        return 1;
    }

    // Read test data
    FILE* f = fopen(data_file.c_str(), "rb");
    if (!f) {
        std::cerr << "Cannot open " << data_file << std::endl;
        return 1;
    }

    int n_cases;
    fread(&n_cases, sizeof(int), 1, f);

    printf("\n%-22s %4s %12s %12s %12s %12s %12s\n",
           "Label", "N", "Py_Alpha", "Cpp_Alpha", "Py_K", "Cpp_K", "Max_Diff");
    printf("--------------------------------------------------------------------------------------------------------------\n");

    int pass_count = 0;
    int total = 0;
    double worst_alpha_diff = 0.0;
    double worst_k_diff = 0.0;
    std::string worst_alpha_label, worst_k_label;

    for (int c = 0; c < n_cases; c++) {
        // Read label
        int label_len;
        fread(&label_len, sizeof(int), 1, f);
        std::string label(label_len, '\0');
        fread(&label[0], 1, label_len, f);

        // Read trajectory
        int n;
        fread(&n, sizeof(int), 1, f);
        std::vector<double> xd(n), yd(n);
        fread(xd.data(), sizeof(double), n, f);
        fread(yd.data(), sizeof(double), n, f);

        // Read Python results
        double py_alpha, py_k;
        fread(&py_alpha, sizeof(double), 1, f);
        fread(&py_k, sizeof(double), 1, f);

        // Read raw predictions (for debugging)
        int n_raw;
        fread(&n_raw, sizeof(int), 1, f);
        std::vector<double> raw_preds(n_raw);
        fread(raw_preds.data(), sizeof(double), n_raw, f);

        // Convert to float for C++ inference
        std::vector<float> xs(n), ys(n);
        for (int i = 0; i < n; i++) {
            xs[i] = (float)xd[i];
            ys[i] = (float)yd[i];
        }

        // C++ predictions
        float cpp_alpha = freetrace::predict_alpha_nn(models, xs, ys);
        float cpp_k = freetrace::predict_k_nn(models, xs, ys);

        double alpha_diff = std::abs((double)cpp_alpha - py_alpha);
        double k_diff = std::abs((double)cpp_k - py_k);
        double max_diff = std::max(alpha_diff, k_diff);

        const char* status = (max_diff < 1e-6) ? "PASS" : "FAIL";
        if (max_diff < 1e-6) pass_count++;
        total++;

        if (alpha_diff > worst_alpha_diff) {
            worst_alpha_diff = alpha_diff;
            worst_alpha_label = label;
        }
        if (k_diff > worst_k_diff) {
            worst_k_diff = k_diff;
            worst_k_label = label;
        }

        printf("%-22s %4d %12.8f %12.8f %12.8f %12.8f %12.2e  %s\n",
               label.c_str(), n,
               py_alpha, (double)cpp_alpha,
               py_k, (double)cpp_k,
               max_diff, status);
    }

    fclose(f);

    printf("\n--- Summary ---\n");
    printf("Passed: %d / %d\n", pass_count, total);
    printf("Worst alpha diff: %.2e (%s)\n", worst_alpha_diff, worst_alpha_label.c_str());
    printf("Worst K diff:     %.2e (%s)\n", worst_k_diff, worst_k_label.c_str());

    if (pass_count == total) {
        printf("\nALL TESTS PASSED (< 1e-6)\n");
    } else {
        printf("\nSome tests FAILED — investigating needed.\n");
    }

    freetrace::free_nn_models(models);
    return (pass_count == total) ? 0 : 1; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
}
