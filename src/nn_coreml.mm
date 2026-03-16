// Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
// CoreML native inference for alpha models on macOS
// Dispatches ConvLSTM layers to GPU/ANE via Apple's CoreML framework
#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>
#include "nn_coreml.h"
#include <map>
#include <string>
#include <cstring>

static std::map<int, MLModel*> g_alpha_models;
static bool g_coreml_loaded = false;

int coreml_load_alpha_models(const char* models_dir, const int* model_nums, int count) {
    @autoreleasepool {
        MLModelConfiguration* config = [[MLModelConfiguration alloc] init];
        config.computeUnits = MLComputeUnitsAll;  // CPU + GPU + ANE

        for (int i = 0; i < count; i++) {
            int n = model_nums[i];

            // Try .mlmodelc (compiled) first, then .mlpackage
            NSString* dir = [NSString stringWithUTF8String:models_dir];
            NSArray<NSString*>* candidates = @[
                [dir stringByAppendingPathComponent:
                    [NSString stringWithFormat:@"reg_model_%d.mlmodelc", n]],
                [dir stringByAppendingPathComponent:
                    [NSString stringWithFormat:@"reg_model_%d.mlpackage", n]],
            ];

            for (NSString* path in candidates) {
                if (![[NSFileManager defaultManager] fileExistsAtPath:path])
                    continue;

                NSURL* url = [NSURL fileURLWithPath:path];
                NSError* error = nil;

                // For .mlpackage, compile first
                NSURL* compiledURL = url;
                if ([path hasSuffix:@".mlpackage"]) {
                    compiledURL = [MLModel compileModelAtURL:url error:&error];
                    if (!compiledURL) {
                        NSLog(@"CoreML: failed to compile %@: %@", path, error);
                        continue;
                    }
                }

                MLModel* model = [MLModel modelOfContentsOfURL:compiledURL
                                                 configuration:config
                                                         error:&error];
                if (model) {
                    g_alpha_models[n] = model;
                    NSLog(@"CoreML: loaded reg_model_%d from %@", n, path);
                    break;
                } else {
                    NSLog(@"CoreML: failed to load %@: %@", path, error);
                }
            }
        }

        g_coreml_loaded = !g_alpha_models.empty();
        return g_coreml_loaded ? 1 : 0;
    }
}

int coreml_is_available(void) {
    return g_coreml_loaded ? 1 : 0;
}

int coreml_predict_alpha(int model_num, const float* input_data,
                         int batch_size, int seq_len,
                         float* output_data) {
    @autoreleasepool {
        auto it = g_alpha_models.find(model_num);
        if (it == g_alpha_models.end()) return 0;

        MLModel* model = it->second;
        NSError* error = nil;

        // Process each sample in the batch
        // CoreML MLModel.prediction processes one sample at a time
        int stride = seq_len * 1 * 3;

        for (int b = 0; b < batch_size; b++) {
            // Create MLMultiArray for single sample: [1, seq_len, 1, 3]
            NSArray<NSNumber*>* shape = @[@1, @(seq_len), @1, @3];
            MLMultiArray* inputArray = [[MLMultiArray alloc]
                initWithShape:shape
                     dataType:MLMultiArrayDataTypeFloat32
                        error:&error];
            if (!inputArray) return 0;

            // Copy input data
            float* ptr = (float*)inputArray.dataPointer;
            memcpy(ptr, input_data + b * stride, stride * sizeof(float));

            // Run prediction
            NSDictionary* inputDict = @{@"input": inputArray};
            id<MLFeatureProvider> features = [[MLDictionaryFeatureProvider alloc]
                initWithDictionary:inputDict error:&error];
            if (!features) return 0;

            id<MLFeatureProvider> output = [model predictionFromFeatures:features
                                                                  error:&error];
            if (!output) {
                NSLog(@"CoreML predict error: %@", error);
                return 0;
            }

            MLMultiArray* result = [output featureValueForName:@"output"].multiArrayValue; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
            if (!result) return 0;

            output_data[b] = ((float*)result.dataPointer)[0];
        }

        return 1;
    }
}

void coreml_free_models(void) {
    @autoreleasepool {
        g_alpha_models.clear();
        g_coreml_loaded = false;
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-16
