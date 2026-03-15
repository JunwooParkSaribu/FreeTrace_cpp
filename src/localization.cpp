#include "localization.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <cmath>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cassert>
#include <random>
#include <iomanip>
#include <cstdlib>
#include <cstring>
#include <map>
#include <filesystem>
#ifdef _WIN32 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
#include <windows.h>
#elif defined(__APPLE__)
#include <sys/sysctl.h>
#else
#include <sys/sysinfo.h>
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef USE_OPENCV
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#endif

#ifdef USE_LIBTIFF // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <tiffio.h>
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

#include "gpu_module.h" // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50

// Global GPU flag — set once at startup in run()
static bool USE_GPU = false; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50

// Get available system RAM in GB // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
static int get_available_ram_gb() {
#ifdef _WIN32
    MEMORYSTATUSEX memInfo;
    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
    GlobalMemoryStatusEx(&memInfo);
    return static_cast<int>(memInfo.ullAvailPhys / (1024ULL * 1024 * 1024));
#elif defined(__APPLE__)
    int64_t phys_mem = 0;
    size_t len = sizeof(phys_mem);
    sysctlbyname("hw.memsize", &phys_mem, &len, nullptr, 0);
    return static_cast<int>(phys_mem / (1024LL * 1024 * 1024));
#else
    struct sysinfo si;
    sysinfo(&si);
    return static_cast<int>((static_cast<uint64_t>(si.freeram) * si.mem_unit) / (1024ULL * 1024 * 1024));
#endif
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

namespace freetrace {

// ============================================================
// Image I/O
// ============================================================

#ifdef USE_OPENCV
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width) {
    std::vector<cv::Mat> pages;
    cv::imreadmulti(path, pages, cv::IMREAD_UNCHANGED);
    nb_frames = static_cast<int>(pages.size());
    if (nb_frames == 0) return {};
    height = pages[0].rows;
    width = pages[0].cols;

    std::vector<float> data(nb_frames * height * width);
    float global_max = 0.0f;
    for (int n = 0; n < nb_frames; ++n) {
        cv::Mat f;
        pages[n].convertTo(f, CV_32F);
        for (int r = 0; r < height; ++r)
            for (int c = 0; c < width; ++c) {
                float v = f.at<float>(r, c);
                data[n * height * width + r * width + c] = v;
                global_max = std::max(global_max, v);
            }
    }
    if (global_max > 0.0f)
        for (auto& v : data) v /= global_max;
    return data;
}
#elif defined(USE_LIBTIFF) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    // Suppress libtiff warnings (e.g., TIFFReadDirectory: unknown field tags)
    TIFFSetWarningHandler(nullptr);
    TIFF* tif = TIFFOpen(path.c_str(), "r");
    if (!tif) { nb_frames = height = width = 0; return {}; }

    // Count directories (frames)
    nb_frames = 0;
    do { nb_frames++; } while (TIFFReadDirectory(tif));

    // Read dimensions from first frame
    TIFFSetDirectory(tif, 0);
    uint32_t w = 0, h = 0;
    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &w);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &h);
    width = static_cast<int>(w);
    height = static_cast<int>(h);

    uint16_t bits_per_sample = 8, sample_format = SAMPLEFORMAT_UINT;
    TIFFGetFieldDefaulted(tif, TIFFTAG_BITSPERSAMPLE, &bits_per_sample);
    TIFFGetFieldDefaulted(tif, TIFFTAG_SAMPLEFORMAT, &sample_format);

    std::vector<float> data(nb_frames * height * width);

    for (int n = 0; n < nb_frames; ++n) {
        TIFFSetDirectory(tif, n);
        std::vector<uint8_t> buf(TIFFScanlineSize(tif));
        for (int r = 0; r < height; ++r) {
            TIFFReadScanline(tif, buf.data(), r);
            for (int c = 0; c < width; ++c) {
                float v = 0.0f;
                if (bits_per_sample == 8) {
                    v = static_cast<float>(buf[c]);
                } else if (bits_per_sample == 16) {
                    if (sample_format == SAMPLEFORMAT_UINT)
                        v = static_cast<float>(reinterpret_cast<uint16_t*>(buf.data())[c]);
                    else
                        v = static_cast<float>(reinterpret_cast<int16_t*>(buf.data())[c]);
                } else if (bits_per_sample == 32) {
                    if (sample_format == SAMPLEFORMAT_IEEEFP)
                        v = reinterpret_cast<float*>(buf.data())[c];
                    else
                        v = static_cast<float>(reinterpret_cast<uint32_t*>(buf.data())[c]);
                }
                data[n * height * width + r * width + c] = v;
            }
        }
    }
    TIFFClose(tif);

    // Normalization matching Python read_tif: // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // Step 1: global (img - s_min) / (s_max - s_min)
    int total = nb_frames * height * width;
    float s_min = data[0], s_max = data[0];
    for (int i = 1; i < total; ++i) {
        s_min = std::min(s_min, data[i]);
        s_max = std::max(s_max, data[i]);
    }
    float range = s_max - s_min;
    if (range > 0.0f)
        for (int i = 0; i < total; ++i)
            data[i] = (data[i] - s_min) / range;

    // Step 2: per-frame normalization by frame max
    int frame_size = height * width;
    for (int n = 0; n < nb_frames; ++n) {
        int base = n * frame_size;
        float fmax = 0.0f;
        for (int i = 0; i < frame_size; ++i)
            fmax = std::max(fmax, data[base + i]);
        if (fmax > 0.0f)
            for (int i = 0; i < frame_size; ++i)
                data[base + i] /= fmax;
    }

    return data;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#else
std::vector<float> read_tiff(const std::string& path, int& nb_frames, int& height, int& width) {
    std::cerr << "TIFF reading requires OpenCV (-DUSE_OPENCV) or libtiff (-DUSE_LIBTIFF)." << std::endl;
    nb_frames = height = width = 0;
    return {};
}
#endif

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
// ND2 reader (Nikon proprietary chunk-based binary format)
// Supports modern ND2 files (uncompressed). Old JPEG2000 files
// are detected and a clear error message is shown.
// ============================================================

static const uint32_t ND2_CHUNK_MAGIC = 0x0ABECEDA;
static const uint32_t ND2_LEGACY_MAGIC = 0x0C000000;
static const char ND2_FILE_SIGNATURE[] = "ND2 FILE SIGNATURE CHUNK NAME01!";
static const char ND2_CHUNKMAP_SIGNATURE[] = "ND2 CHUNK MAP SIGNATURE 0000001!";

static uint32_t le_u32(const uint8_t* p) {
    return uint32_t(p[0]) | (uint32_t(p[1]) << 8) | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}

static uint64_t le_u64(const uint8_t* p) {
    return uint64_t(le_u32(p)) | (uint64_t(le_u32(p + 4)) << 32);
}

// Search for a UTF-16LE encoded key in ND2 metadata and extract uint32 value
static bool nd2_find_uint32(const uint8_t* data, size_t len, const std::string& key, uint32_t& out_val) {
    std::vector<uint8_t> key_u16;
    for (char c : key) { key_u16.push_back(static_cast<uint8_t>(c)); key_u16.push_back(0); }

    for (size_t i = 0; i + key_u16.size() + 4 < len; ++i) {
        if (std::memcmp(data + i, key_u16.data(), key_u16.size()) == 0) {
            size_t val_offset = i + key_u16.size() + 2;
            if (val_offset + 4 <= len) {
                out_val = le_u32(data + val_offset);
                return true;
            }
        }
    }
    return false;
}

// Read chunk data at given file offset (skips 16-byte header + name)
static std::vector<uint8_t> nd2_read_chunk_data(std::ifstream& file, uint64_t offset) {
    file.seekg(offset, std::ios::beg);
    uint8_t hdr[16];
    file.read(reinterpret_cast<char*>(hdr), 16);
    if (!file.good() || le_u32(hdr) != ND2_CHUNK_MAGIC) return {};
    uint32_t name_len = le_u32(hdr + 4);
    uint64_t data_len = le_u64(hdr + 8);
    file.seekg(name_len, std::ios::cur);
    std::vector<uint8_t> data(data_len);
    file.read(reinterpret_cast<char*>(data.data()), data_len);
    return file.good() ? data : std::vector<uint8_t>{};
}

std::vector<float> read_nd2(const std::string& path, int& nb_frames, int& height, int& width) {
    nb_frames = height = width = 0;

    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open ND2 file: " << path << std::endl;
        return {};
    }

    file.seekg(0, std::ios::end);
    int64_t file_size = file.tellg();
    if (file_size < 48) {
        std::cerr << "ND2 file too small: " << path << std::endl;
        return {};
    }

    // Step 1: Verify first chunk header
    file.seekg(0, std::ios::beg);
    uint8_t first_hdr[16];
    file.read(reinterpret_cast<char*>(first_hdr), 16);
    uint32_t first_magic = le_u32(first_hdr);

    if (first_magic == ND2_LEGACY_MAGIC) {
        std::cerr << "\nERROR: Cannot read ND2 file: " << path << std::endl;
        std::cerr << "  This file uses the old ND2 format (JPEG2000 compression)," << std::endl;
        std::cerr << "  which is not supported by FreeTrace C++." << std::endl;
        std::cerr << "  Please re-export the file from NIS-Elements as a modern ND2 or TIFF." << std::endl;
        std::cerr << "  Alternatively, use the Python version of FreeTrace which supports" << std::endl;
        std::cerr << "  all ND2 formats via the 'nd2' library." << std::endl;
        return {};
    }

    if (first_magic != ND2_CHUNK_MAGIC) {
        std::cerr << "\nERROR: Cannot read ND2 file: " << path << std::endl;
        std::cerr << "  Unrecognized file format (invalid chunk magic)." << std::endl;
        return {};
    }

    // Read chunk name and verify it's the file signature
    uint32_t sig_name_len = le_u32(first_hdr + 4);
    std::vector<char> sig_name(sig_name_len);
    file.read(sig_name.data(), sig_name_len);
    if (sig_name_len < 32 || std::memcmp(sig_name.data(), ND2_FILE_SIGNATURE, 32) != 0) {
        std::cerr << "\nERROR: Cannot read ND2 file: " << path << std::endl;
        std::cerr << "  Missing ND2 file signature." << std::endl;
        return {};
    }

    // Step 2: Read chunk map location from last 40 bytes
    // Format: 32-byte signature ("ND2 CHUNK MAP SIGNATURE 0000001!") + 8-byte offset
    file.seekg(-40, std::ios::end);
    uint8_t tail[40];
    file.read(reinterpret_cast<char*>(tail), 40);
    if (std::memcmp(tail, ND2_CHUNKMAP_SIGNATURE, 32) != 0) {
        std::cerr << "ND2 chunk map signature not found: " << path << std::endl;
        return {};
    }
    uint64_t chunkmap_offset = le_u64(tail + 32);

    // Step 3: Read chunk map data
    auto map_data = nd2_read_chunk_data(file, chunkmap_offset);
    if (map_data.empty()) {
        std::cerr << "Failed to read ND2 chunk map: " << path << std::endl;
        return {};
    }

    // Step 4: Parse chunk map entries
    // Format: chunk names delimited by '!' (not null), followed by 16 bytes (uint64 offset + uint64 size)
    struct ChunkEntry { uint64_t offset; uint64_t size; };
    std::map<std::string, ChunkEntry> chunks;
    size_t pos = 0;
    size_t map_len = map_data.size();
    while (pos < map_len) {
        // Find next '!' to get chunk name
        size_t excl = pos;
        while (excl < map_len && map_data[excl] != '!') excl++;
        if (excl >= map_len) break;
        excl++; // include the '!'

        std::string name(reinterpret_cast<const char*>(map_data.data() + pos), excl - pos);

        // Check for end marker
        if (name == ND2_CHUNKMAP_SIGNATURE) break;

        if (excl + 16 > map_len) break;
        ChunkEntry entry;
        entry.offset = le_u64(map_data.data() + excl);
        entry.size   = le_u64(map_data.data() + excl + 8);
        chunks[name] = entry;
        pos = excl + 16;
    }

    // Step 5: Read ImageAttributesLV! metadata
    auto attr_it = chunks.find("ImageAttributesLV!");
    if (attr_it == chunks.end()) {
        std::cerr << "ND2 file missing ImageAttributesLV! chunk: " << path << std::endl;
        return {};
    }

    auto attr_data = nd2_read_chunk_data(file, attr_it->second.offset);
    if (attr_data.empty()) {
        std::cerr << "Failed to read ND2 attributes: " << path << std::endl;
        return {};
    }

    uint32_t ui_width = 0, ui_height = 0, ui_bpc = 16, ui_comp = 1, ui_seq_count = 0;
    uint32_t ui_compression = 2;
    nd2_find_uint32(attr_data.data(), attr_data.size(), "uiWidth", ui_width);
    nd2_find_uint32(attr_data.data(), attr_data.size(), "uiHeight", ui_height);
    nd2_find_uint32(attr_data.data(), attr_data.size(), "uiBpcInMemory", ui_bpc);
    nd2_find_uint32(attr_data.data(), attr_data.size(), "uiComp", ui_comp);
    nd2_find_uint32(attr_data.data(), attr_data.size(), "uiSequenceCount", ui_seq_count);
    nd2_find_uint32(attr_data.data(), attr_data.size(), "uiCompression", ui_compression);

    if (ui_width == 0 || ui_height == 0) {
        std::cerr << "ND2 file has zero dimensions: " << path << std::endl;
        return {};
    }

    if (ui_compression == 1) {
        std::cerr << "\nERROR: Cannot read ND2 file: " << path << std::endl;
        std::cerr << "  This file uses lossy (JPEG2000) compression," << std::endl;
        std::cerr << "  which is not supported by FreeTrace C++." << std::endl;
        std::cerr << "  Please re-export the file from NIS-Elements as uncompressed ND2 or TIFF." << std::endl;
        std::cerr << "  Alternatively, use the Python version of FreeTrace which supports" << std::endl;
        std::cerr << "  all ND2 formats via the 'nd2' library." << std::endl;
        return {};
    }

    if (ui_seq_count == 0) {
        for (auto& kv : chunks)
            if (kv.first.find("ImageDataSeq|") == 0) ui_seq_count++;
    }

    width = static_cast<int>(ui_width);
    height = static_cast<int>(ui_height);
    nb_frames = static_cast<int>(ui_seq_count);

    if (nb_frames == 0) {
        std::cerr << "ND2 file has 0 frames: " << path << std::endl;
        return {};
    }

    // Step 6: Read image data from ImageDataSeq|N! chunks
    size_t bytes_per_pixel = ui_bpc / 8;
    size_t pixels_per_frame = static_cast<size_t>(ui_width) * ui_height;
    size_t channel_count = ui_comp > 0 ? ui_comp : 1;

    std::vector<float> data(static_cast<size_t>(nb_frames) * height * width);

    for (int n = 0; n < nb_frames; ++n) {
        std::string seq_key = "ImageDataSeq|" + std::to_string(n) + "!";
        auto seq_it = chunks.find(seq_key);
        if (seq_it == chunks.end()) {
            std::cerr << "ND2 missing frame " << n << " (" << seq_key << "): " << path << std::endl;
            return {};
        }

        auto raw = nd2_read_chunk_data(file, seq_it->second.offset);
        if (raw.empty()) {
            std::cerr << "ND2 failed to read frame " << n << ": " << path << std::endl;
            return {};
        }

        // Skip 8-byte timestamp at start
        size_t pixel_offset = 8;
        size_t expected_bytes = pixels_per_frame * channel_count * bytes_per_pixel;
        if (pixel_offset + expected_bytes > raw.size()) {
            pixel_offset = 0;
            if (expected_bytes > raw.size()) {
                std::cerr << "ND2 frame " << n << " data too small: " << path << std::endl;
                return {};
            }
        }

        const uint8_t* px = raw.data() + pixel_offset;
        int base = n * height * width;

        for (int r = 0; r < height; ++r) {
            for (int c = 0; c < width; ++c) {
                size_t pixel_idx = static_cast<size_t>(r) * width + c;
                size_t interleaved_idx = pixel_idx * channel_count;
                float v = 0.0f;
                if (bytes_per_pixel == 2) {
                    const uint8_t* p = px + interleaved_idx * 2;
                    v = static_cast<float>(uint16_t(p[0]) | (uint16_t(p[1]) << 8));
                } else if (bytes_per_pixel == 1) {
                    v = static_cast<float>(px[interleaved_idx]);
                } else if (bytes_per_pixel == 4) {
                    const uint8_t* p = px + interleaved_idx * 4;
                    float fv;
                    std::memcpy(&fv, p, 4);
                    v = fv;
                }
                data[base + r * width + c] = v;
            }
        }
    }

    file.close();

    // Normalization matching Python read_tif
    int total = nb_frames * height * width;
    float s_min = data[0], s_max = data[0];
    for (int i = 1; i < total; ++i) {
        s_min = std::min(s_min, data[i]);
        s_max = std::max(s_max, data[i]);
    }
    float range = s_max - s_min;
    if (range > 0.0f)
        for (int i = 0; i < total; ++i)
            data[i] = (data[i] - s_min) / range;

    int frame_size = height * width;
    for (int n = 0; n < nb_frames; ++n) {
        int base = n * frame_size;
        float fmax = 0.0f;
        for (int i = 0; i < frame_size; ++i)
            fmax = std::max(fmax, data[base + i]);
        if (fmax > 0.0f)
            for (int i = 0; i < frame_size; ++i)
                data[base + i] /= fmax;
    }

    return data;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

// ============================================================
// Unified image reader — dispatches by file extension
// ============================================================

static std::string get_extension(const std::string& path) {
    auto dot = path.rfind('.');
    if (dot == std::string::npos) return "";
    std::string ext = path.substr(dot);
    for (auto& c : ext) c = std::tolower(static_cast<unsigned char>(c));
    return ext;
}

std::vector<float> read_image(const std::string& path, int& nb_frames, int& height, int& width) {
    std::string ext = get_extension(path);
    if (ext == ".nd2") {
        return read_nd2(path, nb_frames, height, width);
    } else if (ext == ".tif" || ext == ".tiff") {
        return read_tiff(path, nb_frames, height, width);
    } else {
        std::cerr << "Unsupported file format: " << ext << std::endl;
        std::cerr << "Supported formats: .tif, .tiff, .nd2" << std::endl;
        nb_frames = height = width = 0;
        return {};
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

// ============================================================
// CSV output
// ============================================================

void write_localization_csv(
    const std::string& output_path,
    const LocalizationResult& result
) {
    std::ofstream ofs(output_path + "_loc.csv"); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 11:00
    ofs << std::setprecision(15);  // Match Python's full float precision output
    ofs << "frame,x,y,z,xvar,yvar,rho,norm_cst,intensity,window_size\n";
    for (int frame = 0; frame < static_cast<int>(result.coords.size()); ++frame) {
        for (int p = 0; p < static_cast<int>(result.coords[frame].size()); ++p) {
            auto& pos = result.coords[frame][p];
            auto& info = result.infos[frame][p];
            auto& pdf = result.pdfs[frame][p];
            int ws = static_cast<int>(std::sqrt(static_cast<float>(pdf.size())));
            float intensity = (pdf.size() > 0) ? pdf[pdf.size() / 2] : 0.0f;
            // x/y swap as in Python version
            ofs << (frame + 1) << ","
                << pos[1] << "," << pos[0] << "," << pos[2] << ","
                << info[0] << "," << info[1] << "," << info[2] << ","
                << info[3] << "," << intensity << "," << ws << "\n";
        }
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 11:00

// ============================================================
// Background estimation
// ============================================================

BackgroundResult compute_background(
    const std::vector<float>& imgs, int nb_imgs, int rows, int cols,
    const std::vector<WinParams>& window_sizes, float alpha
) {
    std::vector<float> bg_means(nb_imgs, 0.0f);
    std::vector<float> bg_stds(nb_imgs, 0.0f);

    if (USE_GPU) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        // GPU path: histogram-based background estimation in CUDA kernel
        gpu::compute_background_gpu(imgs, nb_imgs, rows, cols, bg_means.data(), bg_stds.data());
    } else { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
        // CPU path — match Python GPU gpu_module.background() exactly
        int pixel_count = rows * cols;

        // Pre-quantize all frames
        std::vector<std::vector<double>> all_bg_val(nb_imgs, std::vector<double>(pixel_count));
        std::vector<std::vector<int>> all_bg_ibin(nb_imgs, std::vector<int>(pixel_count));
        for (int n = 0; n < nb_imgs; ++n) {
            int base = n * pixel_count;
            double fmax_d = 0.0;
            for (int i = 0; i < pixel_count; ++i)
                fmax_d = std::max(fmax_d, (double)imgs[base + i]);
            if (fmax_d <= 0.0) fmax_d = 1.0;
            for (int i = 0; i < pixel_count; ++i) {
                float normed_f = imgs[base + i] / static_cast<float>(fmax_d);
                int ival = static_cast<int>(static_cast<uint8_t>(normed_f * 100.0f));
                all_bg_val[n][i] = ival / 100.0;
                all_bg_ibin[n][i] = ival;
            }
        }

        // Per-frame state: post_mask, mode_val, mask_std
        std::vector<std::vector<int>> all_post_mask(nb_imgs);
        std::vector<double> all_mode_val(nb_imgs, 0.0);
        std::vector<double> all_mask_std(nb_imgs, 0.0);
        for (int n = 0; n < nb_imgs; ++n) {
            all_post_mask[n].resize(pixel_count);
            std::iota(all_post_mask[n].begin(), all_post_mask[n].end(), 0);
        }

        // 3 iterations matching Python GPU gpu_module.background()
        for (int iter = 0; iter < 3; ++iter) {
            // Step 1: histogram + mode per frame
            for (int n = 0; n < nb_imgs; ++n) {
                auto& post_mask = all_post_mask[n];
                auto& bg_val = all_bg_val[n];
                auto& bg_ibin = all_bg_ibin[n];
                if (post_mask.empty()) continue;

                int max_ival = 0;
                for (int idx : post_mask)
                    max_ival = std::max(max_ival, bg_ibin[idx]);
                int nb_bins = std::max(1, max_ival);

                std::vector<int> hist(nb_bins, 0);
                for (int idx : post_mask) {
                    int b = bg_ibin[idx];
                    if (b > 0 && bg_val[idx] < b * 0.01) b--;
                    if (b >= nb_bins) b = nb_bins - 1;
                    hist[b]++;
                }
                int mode_bin = 0, mode_count = 0;
                for (int b = 0; b < nb_bins; ++b) {
                    if (hist[b] > mode_count) { mode_count = hist[b]; mode_bin = b; }
                }
                all_mode_val[n] = mode_bin * 0.01 + 0.005;
            }

            // Step 2: per-frame std from masked values (every iteration) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
            for (int n = 0; n < nb_imgs; ++n) {
                auto& post_mask = all_post_mask[n];
                auto& bg_val = all_bg_val[n];
                if (post_mask.empty()) { all_mask_std[n] = 0.0; continue; }
                double sum = 0.0, sum2 = 0.0;
                for (int idx : post_mask) { sum += bg_val[idx]; sum2 += bg_val[idx] * bg_val[idx]; }
                double mean_tmp = sum / post_mask.size();
                all_mask_std[n] = std::sqrt(std::max(0.0, sum2 / post_mask.size() - mean_tmp * mean_tmp));
            } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

            // Step 3: rebuild masks
            for (int n = 0; n < nb_imgs; ++n) {
                auto& bg_val = all_bg_val[n];
                double lo = all_mode_val[n] - 3.0 * all_mask_std[n];
                double hi = all_mode_val[n] + 3.0 * all_mask_std[n];
                std::vector<int> new_mask;
                for (int i = 0; i < pixel_count; ++i)
                    if (bg_val[i] > lo && bg_val[i] < hi)
                        new_mask.push_back(i);
                all_post_mask[n] = std::move(new_mask);
            }
        }

        // Final: compute mean/std from final mask
        for (int n = 0; n < nb_imgs; ++n) {
            auto& post_mask = all_post_mask[n];
            auto& bg_val = all_bg_val[n];
            if (!post_mask.empty()) {
                double sum = 0.0, sum2 = 0.0;
                for (int idx : post_mask) { sum += bg_val[idx]; sum2 += bg_val[idx] * bg_val[idx]; }
                double mean_d = sum / post_mask.size();
                bg_means[n] = static_cast<float>(mean_d);
                bg_stds[n] = static_cast<float>(std::sqrt(std::max(0.0, sum2 / post_mask.size() - mean_d * mean_d)));
            }
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    BackgroundResult result;
    for (auto& ws : window_sizes) {
        int area = ws.w * ws.h;
        std::vector<float> bg(nb_imgs * area);
        for (int n = 0; n < nb_imgs; ++n)
            for (int i = 0; i < area; ++i)
                bg[n * area + i] = bg_means[n];
        result.bgs[ws.w] = std::move(bg);
    }

    result.thresholds.resize(nb_imgs);
    for (int n = 0; n < nb_imgs; ++n) {
        float t = (bg_stds[n] > 0) ? 1.0f / (bg_means[n] * bg_means[n] / (bg_stds[n] * bg_stds[n])) * 2.0f : 1.0f;
        t = std::max(t, 1.0f);
        if (std::isnan(t)) t = 1.0f;
        result.thresholds[n] = t * alpha;
    }

    return result;
}

// ============================================================
// Gaussian PSF
// ============================================================

Image2D gauss_psf(int win_w, int win_h, float radius) {
    Image2D grid(win_h, win_w);
    float cx = win_w / 2.0f;
    float cy = win_h / 2.0f;
    float norm = std::sqrt(M_PI) * radius;

    for (int r = 0; r < win_h; ++r) {
        for (int c = 0; c < win_w; ++c) {
            float dx = (c + 0.5f) - cx;
            float dy = (r + 0.5f) - cy;
            grid.at(r, c) = std::exp(-(dx * dx + dy * dy) / (2.0f * radius * radius)) / norm;
        }
    }
    return grid;
}

// ============================================================
// Region max filter (single-window, forward)
// ============================================================

std::vector<DetIndex> region_max_filter2(
    std::vector<float>& maps, int nb_imgs, int rows, int cols,
    const WinParams& window_size, const std::vector<float>& thresholds,
    int detect_range
) {
    std::vector<DetIndex> indices;
    int r_start = (detect_range == 0) ? window_size.h / 2 : detect_range;
    int c_start = (detect_range == 0) ? window_size.w / 2 : detect_range;

    for (int pass = 0; pass < 2; ++pass) {
        for (int n = 0; n < nb_imgs; ++n) {
            float thresh = thresholds[n];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    int idx = n * rows * cols + r * cols + c;
                    if (maps[idx] <= thresh) { maps[idx] = 0.0f; continue; }

                    int r0 = std::max(0, r - r_start);
                    int r1 = std::min(rows, r + r_start + 1);
                    int c0 = std::max(0, c - c_start);
                    int c1 = std::min(cols, c + c_start + 1);

                    float local_max = 0.0f;
                    for (int ri = r0; ri < r1; ++ri)
                        for (int ci = c0; ci < c1; ++ci)
                            local_max = std::max(local_max, maps[n * rows * cols + ri * cols + ci]);

                    if (maps[idx] == local_max && maps[idx] != 0.0f) {
                        indices.push_back({n, r, c});
                        for (int ri = r0; ri < r1; ++ri)
                            for (int ci = c0; ci < c1; ++ci)
                                maps[n * rows * cols + ri * cols + ci] = 0.0f;
                    }
                }
            }
        }
    }
    return indices;
}

// ============================================================
// Region max filter (multi-window, backward) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// ============================================================

std::vector<DetIndexWin> region_max_filter_multi(
    std::vector<float>& maps,
    int nb_wins, int nb_imgs, int rows, int cols,
    const std::vector<WinParams>& window_sizes,
    const std::vector<float>& thresholds,
    int detect_range
) {
    // maps: [nb_wins][nb_imgs][rows][cols] flattened
    // thresholds: [nb_imgs * nb_wins]
    std::vector<DetIndexWin> all_indices;

    // Per-frame aggregation: collect detections from all windows, sort by score, apply NMS
    // For each frame, gather (win_idx, r, c, score)
    struct CandidateInfo { int win_idx; int r; int c; float score; };
    std::vector<std::vector<CandidateInfo>> per_frame(nb_imgs);

    for (int wi = 0; wi < nb_wins; ++wi) {
        int r_half = (detect_range == 0) ? window_sizes[wi].h / 2 : detect_range;
        int c_half = (detect_range == 0) ? window_sizes[wi].w / 2 : detect_range;

        for (int n = 0; n < nb_imgs; ++n) {
            float thresh = thresholds[n * nb_wins + wi];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    int idx = wi * nb_imgs * rows * cols + n * rows * cols + r * cols + c;
                    if (maps[idx] <= thresh) { maps[idx] = 0.0f; continue; }

                    int r0 = std::max(0, r - r_half);
                    int r1 = std::min(rows, r + r_half + 1);
                    int c0 = std::max(0, c - c_half);
                    int c1 = std::min(cols, c + c_half + 1);

                    float local_max = 0.0f;
                    for (int ri = r0; ri < r1; ++ri)
                        for (int ci = c0; ci < c1; ++ci)
                            local_max = std::max(local_max,
                                maps[wi * nb_imgs * rows * cols + n * rows * cols + ri * cols + ci]);

                    if (maps[idx] == local_max && maps[idx] != 0.0f) {
                        per_frame[n].push_back({wi, r, c, maps[idx]});
                        // Zero out neighborhood for this window
                        for (int ri = r0; ri < r1; ++ri)
                            for (int ci = c0; ci < c1; ++ci)
                                maps[wi * nb_imgs * rows * cols + n * rows * cols + ri * cols + ci] = 0.0f;
                    }
                }
            }
        }
    }

    // NMS per frame: sort by score descending, mask overlaps
    for (int n = 0; n < nb_imgs; ++n) {
        auto& cands = per_frame[n];
        std::sort(cands.begin(), cands.end(),
                  [](const CandidateInfo& a, const CandidateInfo& b) { return a.score > b.score; });

        std::vector<std::vector<bool>> mask(rows, std::vector<bool>(cols, false));
        for (auto& cand : cands) {
            if (mask[cand.r][cand.c]) continue;
            int ext = (detect_range == 0) ? (window_sizes[cand.win_idx].w - 1) / 2 : detect_range;
            int r0 = std::max(0, cand.r - ext);
            int r1 = std::min(rows - 1, cand.r + ext);
            int c0 = std::max(0, cand.c - ext);
            int c1 = std::min(cols - 1, cand.c + ext);

            all_indices.push_back({{n, cand.r, cand.c}, window_sizes[cand.win_idx].w});
            for (int ri = r0; ri <= r1; ++ri)
                for (int ci = c0; ci <= c1; ++ci)
                    mask[ri][ci] = true;
        }
    }
    return all_indices;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================
// Bivariate normal PDF (unnormalized) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// ============================================================

std::vector<float> bi_variate_normal_pdf(
    int nb_imgs, int win_w, int win_h,
    const std::vector<float>& x_var, const std::vector<float>& y_var,
    const std::vector<float>& rho, const std::vector<float>& amp
) {
    int win_area = win_w * win_h;
    std::vector<float> result(nb_imgs * win_area, 0.0f);

    for (int n = 0; n < nb_imgs; ++n) {
        float xv = x_var[n], yv = y_var[n], r = rho[n];
        float k = 1.0f - r * r;
        if (std::abs(k) < 1e-12f) k = 1e-12f;
        float sx = std::sqrt(std::abs(xv));
        float sy = std::sqrt(std::abs(yv));

        // Covariance matrix: [[xv, r*sx*sy], [r*sx*sy, yv]]
        // inv(cov) = 1/det * [[yv, -r*sx*sy], [-r*sx*sy, xv]]
        // det = xv*yv - (r*sx*sy)^2 = xv*yv*k
        float det = xv * yv * k;
        if (std::abs(det) < 1e-12f) det = 1e-12f;

        float inv00 = yv / det;
        float inv01 = -r * sx * sy / det;
        float inv11 = xv / det;

        for (int ri = 0; ri < win_h; ++ri) {
            for (int ci = 0; ci < win_w; ++ci) {
                float dx = static_cast<float>(ci - win_w / 2);
                float dy = static_cast<float>(ri - win_h / 2);
                float exponent = -0.5f * (dx * dx * inv00 + 2.0f * dx * dy * inv01 + dy * dy * inv11);
                result[n * win_area + ri * win_w + ci] = amp[n] * std::exp(exponent);
            }
        }
    }
    return result;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================
// Image regression (Guo + unpack + PDF construction) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// ============================================================

RegressionResult image_regression(
    std::vector<float>& imgs,
    const std::vector<float>& bgs,
    int nb_imgs, int win_w, int win_h,
    const float* p0, int repeat
) {
    int win_area = win_w * win_h;
    RegressionResult res;

    // Run Guo algorithm
    auto coefs = guo_algorithm(imgs, bgs, p0, nb_imgs, win_w, win_h, repeat);

    // Unpack coefficients
    auto unpacked = unpack_coefs(coefs, win_w, win_h);

    // If there are errors, retry with repeat+1 on a fresh copy
    if (!unpacked.err_indices.empty()) {
        // Re-run with more iterations (imgs already modified, but guo handles it)
        auto coefs2 = guo_algorithm(imgs, bgs, p0, nb_imgs, win_w, win_h, repeat + 1);
        unpacked = unpack_coefs(coefs2, win_w, win_h);
    }

    res.xs.resize(nb_imgs);
    res.ys.resize(nb_imgs);
    res.x_vars.resize(nb_imgs);
    res.y_vars.resize(nb_imgs);
    res.amps.resize(nb_imgs);
    res.rhos.resize(nb_imgs);

    for (int i = 0; i < nb_imgs; ++i) {
        res.x_vars[i] = unpacked.x_var[i];
        res.xs[i] = unpacked.x_mu[i];
        res.y_vars[i] = unpacked.y_var[i];
        res.ys[i] = unpacked.y_mu[i];
        res.rhos[i] = unpacked.rho[i];
        res.amps[i] = unpacked.amp[i];
    }

    // Build PDFs: amp * bivariate_normal(unnormalized) + bg
    auto pdf_flat = bi_variate_normal_pdf(nb_imgs, win_w, win_h,
                                           res.x_vars, res.y_vars, res.rhos, res.amps);

    res.pdfs.resize(nb_imgs);
    for (int n = 0; n < nb_imgs; ++n) {
        res.pdfs[n].resize(win_area);
        for (int p = 0; p < win_area; ++p) {
            res.pdfs[n][p] = pdf_flat[n * win_area + p] + bgs[n * win_area + p];
        }
    }

    // Mark error indices with -100 (matching Python — after PDF computation) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    for (int ei : unpacked.err_indices) {
        res.x_vars[ei] = -100.0f;
        res.y_vars[ei] = -100.0f;
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

    return res;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================
// Subtract PDF (deflation) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// ============================================================

void subtract_pdf(
    std::vector<float>& ext_imgs,
    int nb_imgs, int ext_rows, int ext_cols,
    const std::vector<std::vector<float>>& pdfs,
    const std::vector<std::array<int, 3>>& indices,
    int win_w, int win_h,
    const std::vector<float>& bg_means,
    int extend
) {
    int half_w = (win_w - 1) / 2;
    int half_h = (win_h - 1) / 2;
    int half_ext = extend / 2;

    for (size_t i = 0; i < indices.size(); ++i) {
        int n = indices[i][0];
        int r = indices[i][1];
        int c = indices[i][2];
        float bg_val = bg_means[n];

        int r0 = r - half_h + half_ext;
        int c0 = c - half_w + half_ext;

        for (int ri = 0; ri < win_h; ++ri) {
            for (int ci = 0; ci < win_w; ++ci) {
                int er = r0 + ri;
                int ec = c0 + ci;
                if (er < 0 || er >= ext_rows || ec < 0 || ec >= ext_cols) continue;
                int ext_idx = n * ext_rows * ext_cols + er * ext_cols + ec;
                ext_imgs[ext_idx] -= pdfs[i][ri * win_w + ci];
                ext_imgs[ext_idx] = std::max(ext_imgs[ext_idx], bg_val);
            }
        }

        // Boundary smoothing around the subtracted region
        // (simplified — smooth the border of the subtracted area)
        int row_min = r0, row_max = r0 + win_h - 1;
        int col_min = c0, col_max = c0 + win_w - 1;
        if (row_min >= 0 && row_max < ext_rows && col_min >= 0 && col_max < ext_cols) {
            // Create a temporary Image2D view of this frame
            // Use the already-ported boundary_smoothing on a crop
            // For efficiency, do a simple local mean smoothing on the border pixels
            const int erase_space = 2;
            const int border_iters = 2;
            for (int iter = 0; iter < border_iters; ++iter) {
                // Top and bottom edges
                for (int ec = col_min; ec <= col_max; ++ec) {
                    for (int edge_r : {row_min, row_max}) {
                        int er0 = std::max(0, edge_r - erase_space);
                        int er1 = std::min(ext_rows - 1, edge_r + erase_space);
                        int ec0 = std::max(0, ec - erase_space);
                        int ec1 = std::min(ext_cols - 1, ec + erase_space);
                        float sum = 0; int cnt = 0;
                        for (int rr = er0; rr <= er1; ++rr)
                            for (int cc = ec0; cc <= ec1; ++cc) {
                                sum += ext_imgs[n * ext_rows * ext_cols + rr * ext_cols + cc];
                                cnt++;
                            }
                        ext_imgs[n * ext_rows * ext_cols + edge_r * ext_cols + ec] = sum / cnt;
                    }
                }
                // Left and right edges
                for (int er = row_min; er <= row_max; ++er) {
                    for (int edge_c : {col_min, col_max}) {
                        int er0 = std::max(0, er - erase_space);
                        int er1 = std::min(ext_rows - 1, er + erase_space);
                        int ec0 = std::max(0, edge_c - erase_space);
                        int ec1 = std::min(ext_cols - 1, edge_c + erase_space);
                        float sum = 0; int cnt = 0;
                        for (int rr = er0; rr <= er1; ++rr)
                            for (int cc = ec0; cc <= ec1; ++cc) {
                                sum += ext_imgs[n * ext_rows * ext_cols + rr * ext_cols + cc];
                                cnt++;
                            }
                        ext_imgs[n * ext_rows * ext_cols + er * ext_cols + edge_c] = sum / cnt;
                    }
                }
            }
        }
    }
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================
// Parameter generation
// ============================================================

static void params_gen(int win_s,
    std::vector<WinParams>& single_ws, std::vector<float>& single_rad,
    std::vector<WinParams>& multi_ws, std::vector<float>& multi_rad
) {
    if (win_s < 5) win_s = 5;
    if (win_s % 2 == 0) win_s += 1;

    single_ws = {{win_s, win_s}};
    multi_ws = {{win_s - 2, win_s - 2}, {win_s, win_s}, {win_s + 2, win_s + 2}};
    single_rad.clear();
    for (auto& ws : single_ws) single_rad.push_back((ws.w / 2) / 2.0f);
    multi_rad.clear();
    for (auto& ws : multi_ws) multi_rad.push_back((ws.w / 2) / 2.0f);
}

// ============================================================
// Core localization (forward + backward) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// ============================================================

LocalizationResult localize( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    const std::vector<float>& imgs, int nb_imgs, int rows, int cols,
    int window_size, float threshold_alpha, int shift,
    int deflation_loop_backward,
    NumpyRNG* external_rng
) {
    // Generate window params
    std::vector<WinParams> single_ws, multi_ws;
    std::vector<float> single_rad, multi_rad;
    params_gen(window_size, single_ws, single_rad, multi_ws, multi_rad);

    // Compute background
    std::vector<WinParams> all_ws = single_ws;
    all_ws.insert(all_ws.end(), multi_ws.begin(), multi_ws.end());
    auto bg_result = compute_background(imgs, nb_imgs, rows, cols, all_ws, threshold_alpha);

    // Generate Gaussian PSFs
    std::vector<Image2D> forward_grids, backward_grids;
    for (size_t i = 0; i < single_ws.size(); ++i)
        forward_grids.push_back(gauss_psf(single_ws[i].w, single_ws[i].h, single_rad[i]));
    for (size_t i = 0; i < multi_ws.size(); ++i)
        backward_grids.push_back(gauss_psf(multi_ws[i].w, multi_ws[i].h, multi_rad[i]));

    int extend = multi_ws.back().w * 4;
    int ext_rows = rows + extend;
    int ext_cols = cols + extend;
    int half_ext = extend / 2;

    // Create extended images with zero padding
    std::vector<float> ext_imgs(nb_imgs * ext_rows * ext_cols, 0.0f);
    for (int n = 0; n < nb_imgs; ++n)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                ext_imgs[n * ext_rows * ext_cols + (r + half_ext) * ext_cols + (c + half_ext)] =
                    imgs[n * rows * cols + r * cols + c];

    // Add block noise to borders // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    add_block_noise(ext_imgs, nb_imgs, ext_rows, ext_cols, extend, external_rng);

    // DEBUG: dump ext_imgs for specific frame // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    if (std::getenv("FREETRACE_DEBUG_LOC")) {
        static int ext_batch_idx = 0;
        if (ext_batch_idx == 3 && nb_imgs > 53) {
            int lf = 53;
            std::string p = "/home/junwoo/claude/FreeTrace_comparison/debug_cpp_ext_f353.bin";
            std::ofstream ofs(p, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(&ext_imgs[lf * ext_rows * ext_cols]),
                      ext_rows * ext_cols * sizeof(float));
            std::cout << "Saved ext_imgs batch3 frame53 to " << p << std::endl;
        }
        ext_batch_idx++;
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

    // Background means (per-frame scalar)
    auto& bg_vec = bg_result.bgs[multi_ws[0].w];
    std::vector<float> bg_means(nb_imgs);
    for (int n = 0; n < nb_imgs; ++n)
        bg_means[n] = bg_vec[n * multi_ws[0].w * multi_ws[0].h]; // all same per frame

    // DEBUG: dump bg_means and thresholds // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    if (std::getenv("FREETRACE_DEBUG_LOC")) {
        static int bg_batch_idx = 0;
        std::cout << std::setprecision(15);
        std::cout << "BG_BATCH " << bg_batch_idx << ": " << nb_imgs << " frames" << std::endl;
        for (int i = 0; i < nb_imgs; ++i)
            std::cout << "  f" << i << " bg_mean=" << bg_means[i] << " threshold=" << bg_result.thresholds[i] << std::endl;
        bg_batch_idx++;
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

    // Initial parameters for Guo regression
    float p0[6] = {1.5f, 0.0f, 1.5f, 0.0f, 0.0f, 0.5f};

    // Result
    LocalizationResult result;
    result.coords.resize(nb_imgs);
    result.pdfs.resize(nb_imgs);
    result.infos.resize(nb_imgs);

    // ==============================
    // FORWARD PASS (single window) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // ==============================
    {
        auto& ws0 = single_ws[0];
        auto& grid0 = forward_grids[0];

        // Crop images // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50
        int nb_crops = 0;
        std::vector<float> crop_imgs;
        if (USE_GPU) {
            crop_imgs = gpu::image_cropping_gpu(ext_imgs, nb_imgs, ext_rows, ext_cols,
                                                 extend, ws0.w, ws0.h, shift, nb_crops);
        } else {
            crop_imgs = image_cropping(ext_imgs, nb_imgs, ext_rows, ext_cols,
                                         extend, ws0.w, ws0.h, shift, nb_crops);
        }

        // Compute bg_squared_sums
        std::vector<float> bg_sq_sums(nb_imgs);
        for (int n = 0; n < nb_imgs; ++n)
            bg_sq_sums[n] = ws0.w * ws0.h * bg_means[n] * bg_means[n];

        // Compute likelihood
        std::vector<float> lik;
        if (USE_GPU) {
            int sw = ws0.h * ws0.w;
            std::vector<float> gdata(sw);
            float gmean = image_mean(grid0);
            for (int i = 0; i < ws0.h; ++i)
                for (int j = 0; j < ws0.w; ++j)
                    gdata[i * ws0.w + j] = grid0.data[i * grid0.cols + j];
            lik = gpu::likelihood_gpu(crop_imgs, gdata, gmean, bg_sq_sums, bg_means,
                                       nb_imgs, nb_crops, sw);
        } else {
            lik = likelihood(crop_imgs, grid0, bg_sq_sums, bg_means,
                              nb_imgs, nb_crops, ws0.h, ws0.w);
        }

        // Map likelihood back to image space // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50
        auto h_map = mapping(lik, nb_imgs, rows, cols, shift);

        // DEBUG: dump h_map for specific batches // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        if (std::getenv("FREETRACE_DEBUG_LOC")) {
            static int hmap_batch_idx = 0;
            std::string p = "/home/junwoo/claude/FreeTrace_comparison/debug_cpp_hmap_batch" + std::to_string(hmap_batch_idx) + ".bin";
            std::ofstream ofs(p, std::ios::binary);
            ofs.write(reinterpret_cast<const char*>(h_map.data()), nb_imgs * rows * cols * sizeof(float));
            std::cout << "Saved h_map batch" << hmap_batch_idx << " (" << nb_imgs << " frames) to " << p << std::endl;
            hmap_batch_idx++;
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

        // Thresholds for single window — use bg_result.thresholds directly
        auto& thresholds = bg_result.thresholds;

        // Region max filter with NMS (matching Python region_max_filter) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        // Step 1: Find local maxima above threshold
        int r_half = (shift == 0) ? ws0.h / 2 : shift;
        int c_half = (shift == 0) ? ws0.w / 2 : shift;
        struct CandInfo { int frame; int r; int c; float score; };
        std::vector<std::vector<CandInfo>> per_frame_cands(nb_imgs);

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < nb_imgs; ++n) {
            float thresh = thresholds[n];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    int idx = n * rows * cols + r * cols + c;
                    if (h_map[idx] <= thresh) continue;

                    int r0 = std::max(0, r - r_half);
                    int r1 = std::min(rows, r + r_half + 1);
                    int c0 = std::max(0, c - c_half);
                    int c1 = std::min(cols, c + c_half + 1);

                    float local_max = 0.0f;
                    for (int ri = r0; ri < r1; ++ri)
                        for (int ci = c0; ci < c1; ++ci)
                            local_max = std::max(local_max, h_map[n * rows * cols + ri * cols + ci]);

                    if (h_map[idx] == local_max)
                        per_frame_cands[n].push_back({n, r, c, h_map[idx]});
                }
            }
        }

        // Step 2: Per-frame NMS with mask (sort by score desc, suppress overlaps) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        // Matching Python: mask[row_min:row_max, col_min:col_max] = 1 (exclusive upper bound)
        std::vector<std::vector<DetIndex>> per_frame_indices(nb_imgs);
        int ext_nms = (shift == 0) ? (ws0.w - 1) / 2 : shift;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < nb_imgs; ++n) {
            auto& cands = per_frame_cands[n];
            std::sort(cands.begin(), cands.end(),
                      [](const CandInfo& a, const CandInfo& b) { return a.score > b.score; });

            std::vector<std::vector<uint8_t>> mask(rows, std::vector<uint8_t>(cols, 0));
            for (auto& cand : cands) {
                if (mask[cand.r][cand.c] != 0) continue;
                per_frame_indices[n].push_back({n, cand.r, cand.c});
                int r0 = std::max(0, cand.r - ext_nms);
                int r1 = std::min(rows - 1, cand.r + ext_nms);
                int c0 = std::max(0, cand.c - ext_nms);
                int c1 = std::min(cols - 1, cand.c + ext_nms);
                for (int ri = r0; ri < r1; ++ri)
                    for (int ci = c0; ci < c1; ++ci)
                        mask[ri][ci] = 1;
            }
        }
        // Merge in frame order (deterministic)
        std::vector<DetIndex> indices;
        for (int n = 0; n < nb_imgs; ++n)
            indices.insert(indices.end(), per_frame_indices[n].begin(), per_frame_indices[n].end());

        // DEBUG: dump NMS counts and positions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        if (std::getenv("FREETRACE_DEBUG_LOC")) {
            static int nms_batch_idx = 0;
            std::cout << "BATCH " << nms_batch_idx << ": total=" << indices.size() << std::endl;
            for (int fn = 0; fn < nb_imgs; ++fn) {
                std::vector<std::pair<int,int>> fpos;
                for (auto& idx : indices)
                    if (idx.frame == fn) fpos.push_back({idx.row, idx.col});
                if (!fpos.empty()) {
                    std::cout << "  frame" << fn << ": " << fpos.size();
                    for (auto& p : fpos) std::cout << " (" << p.first << "," << p.second << ")";
                    std::cout << std::endl;
                }
            }
            nms_batch_idx++;
        } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

        if (!indices.empty()) {
            int ws = ws0.w;
            int win_area = ws * ws;

            // Gather crop images and background for detected points
            int nb_det = static_cast<int>(indices.size());
            std::vector<float> regress_imgs(nb_det * win_area);
            std::vector<float> bg_regress(nb_det * win_area);
            std::vector<int> ns(nb_det), det_rs(nb_det), det_cs(nb_det);

            int cols_per_row = cols / shift;
            if (shift == 1) cols_per_row = cols;

            for (int d = 0; d < nb_det; ++d) {
                int n = indices[d].frame;
                int r = indices[d].row;
                int c = indices[d].col;
                ns[d] = n;
                det_rs[d] = r;
                det_cs[d] = c;

                // Get the crop from crop_imgs
                int crop_idx = (r / shift) * (cols / shift) + (c / shift);
                if (crop_idx < nb_crops) {
                    for (int p = 0; p < win_area; ++p)
                        regress_imgs[d * win_area + p] = crop_imgs[n * nb_crops * win_area + crop_idx * win_area + p];
                }

                // Background for this detection
                auto& bg_for_ws = bg_result.bgs[ws];
                for (int p = 0; p < win_area; ++p)
                    bg_regress[d * win_area + p] = bg_for_ws[n * win_area + p];
            }


            // Run image regression
            auto reg = image_regression(regress_imgs, bg_regress, nb_det, ws, ws, p0, 5);

            // Filter errors and collect valid results
            std::vector<int> err_indices;
            for (int d = 0; d < nb_det; ++d) {
                if (reg.x_vars[d] < 0 || reg.y_vars[d] < 0 ||
                    reg.x_vars[d] > 3 * ws || reg.y_vars[d] > 3 * ws ||
                    reg.rhos[d] > 1 || reg.rhos[d] < -1)
                    err_indices.push_back(d);
            }

            // Collect valid detections
            std::vector<std::array<int, 3>> del_indices;
            for (int d = 0; d < nb_det; ++d) {
                if (std::find(err_indices.begin(), err_indices.end(), d) != err_indices.end())
                    continue;

                int n = ns[d];
                float r_f = det_rs[d] + reg.ys[d];
                float c_f = det_cs[d] + reg.xs[d];
                if (r_f <= -1 || r_f >= rows || c_f <= -1 || c_f >= cols)
                    continue;

                float x_coord = std::max(0.0f, std::min(r_f, static_cast<float>(rows - 1)));
                float y_coord = std::max(0.0f, std::min(c_f, static_cast<float>(cols - 1)));
                float z_coord = 0.0f;

                result.coords[n].push_back({x_coord, y_coord, z_coord});
                result.pdfs[n].push_back(reg.pdfs[d]);
                result.infos[n].push_back({reg.x_vars[d], reg.y_vars[d], reg.rhos[d], reg.amps[d]});

                // Collect indices for deflation
                del_indices.push_back({n,
                    static_cast<int>(std::round(det_rs[d] + reg.ys[d])),
                    static_cast<int>(std::round(det_cs[d] + reg.xs[d]))});
            }

            // Deflation: subtract fitted PSFs from extended images
            if (!del_indices.empty()) {
                // Gather pdfs for valid detections only
                std::vector<std::vector<float>> valid_pdfs;
                int valid_idx = 0;
                for (int d = 0; d < nb_det; ++d) {
                    if (std::find(err_indices.begin(), err_indices.end(), d) != err_indices.end())
                        continue;
                    int n = ns[d];
                    float r_f = det_rs[d] + reg.ys[d];
                    float c_f = det_cs[d] + reg.xs[d];
                    if (r_f <= -1 || r_f >= rows || c_f <= -1 || c_f >= cols)
                        continue;
                    valid_pdfs.push_back(reg.pdfs[d]);
                }
                subtract_pdf(ext_imgs, nb_imgs, ext_rows, ext_cols,
                             valid_pdfs, del_indices, ws, ws, bg_means, extend);
            }
        }
    }

    // ==============================
    // BACKWARD PASS (multi-window) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    // ==============================
    for (int df_loop = 0; df_loop < deflation_loop_backward; ++df_loop) {
        int nb_multi = static_cast<int>(multi_ws.size());

        // Compute likelihood maps for each backward window size
        // h_maps: [nb_multi][nb_imgs * rows * cols]
        std::vector<std::vector<float>> h_maps(nb_multi);

        for (int wi = 0; wi < nb_multi; ++wi) {
            auto& ws = multi_ws[wi];
            auto& grid = backward_grids[wi];

            int nb_crops = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50
            std::vector<float> crop_imgs;
            if (USE_GPU) {
                crop_imgs = gpu::image_cropping_gpu(ext_imgs, nb_imgs, ext_rows, ext_cols,
                                                     extend, ws.w, ws.h, shift, nb_crops);
            } else {
                crop_imgs = image_cropping(ext_imgs, nb_imgs, ext_rows, ext_cols,
                                             extend, ws.w, ws.h, shift, nb_crops);
            }

            std::vector<float> bg_sq_sums(nb_imgs);
            for (int n = 0; n < nb_imgs; ++n)
                bg_sq_sums[n] = ws.w * ws.h * bg_means[n] * bg_means[n];

            std::vector<float> lik;
            if (USE_GPU) {
                int sw = ws.h * ws.w;
                std::vector<float> gdata(sw);
                float gmean = image_mean(grid);
                for (int i = 0; i < ws.h; ++i)
                    for (int j = 0; j < ws.w; ++j)
                        gdata[i * ws.w + j] = grid.data[i * grid.cols + j];
                lik = gpu::likelihood_gpu(crop_imgs, gdata, gmean, bg_sq_sums, bg_means,
                                           nb_imgs, nb_crops, sw);
            } else {
                lik = likelihood(crop_imgs, grid, bg_sq_sums, bg_means,
                                  nb_imgs, nb_crops, ws.h, ws.w);
            }

            auto h_map = mapping(lik, nb_imgs, rows, cols, shift); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50

            // Scale by window area ratio (as in Python)
            float scale = static_cast<float>(multi_ws[0].w * multi_ws[0].w) /
                          static_cast<float>(ws.w * ws.w);
            for (auto& v : h_map) v *= scale;

            h_maps[wi] = std::move(h_map);
        }

        // Run region_max_filter2 per window size (backward pass)
        // Then do multi-scale selection + regression
        for (int wi = nb_multi - 1; wi >= 0; --wi) {
            auto& ws = multi_ws[wi];
            auto indices = region_max_filter2(h_maps[wi], nb_imgs, rows, cols,
                                              ws, bg_result.thresholds, shift);

            if (indices.empty()) continue;

            int win = ws.w;
            int win_area = win * win;
            int nb_det = static_cast<int>(indices.size());

            // Extract regression images from extended_imgs
            std::vector<float> regress_imgs(nb_det * win_area);
            std::vector<float> bg_regress(nb_det * win_area);

            for (int d = 0; d < nb_det; ++d) {
                int n = indices[d].frame;
                int r = indices[d].row + half_ext;
                int c = indices[d].col + half_ext;

                int r0 = r - (win - 1) / 2;
                int c0 = c - (win - 1) / 2;

                for (int ri = 0; ri < win; ++ri)
                    for (int ci = 0; ci < win; ++ci) {
                        int er = r0 + ri, ec = c0 + ci;
                        if (er >= 0 && er < ext_rows && ec >= 0 && ec < ext_cols)
                            regress_imgs[d * win_area + ri * win + ci] =
                                ext_imgs[n * ext_rows * ext_cols + er * ext_cols + ec];
                    }

                auto& bg_for_ws = bg_result.bgs[win];
                for (int p = 0; p < win_area; ++p)
                    bg_regress[d * win_area + p] = bg_for_ws[n * win_area + p];
            }

            // Regression
            float p0_back[6] = {1.5f, 0.0f, 1.5f, 0.0f, 0.0f, 0.5f};
            auto reg = image_regression(regress_imgs, bg_regress, nb_det, win, win, p0_back, 5);

            // Collect valid results
            std::vector<std::array<int, 3>> del_indices;
            std::vector<std::vector<float>> valid_pdfs;

            for (int d = 0; d < nb_det; ++d) {
                if (reg.x_vars[d] < 0 || reg.y_vars[d] < 0 ||
                    reg.x_vars[d] > 3 * win || reg.y_vars[d] > 3 * win ||
                    reg.rhos[d] > 1 || reg.rhos[d] < -1)
                    continue;

                int n = indices[d].frame;
                float r_f = indices[d].row + reg.ys[d];
                float c_f = indices[d].col + reg.xs[d];
                if (r_f <= -1 || r_f >= rows || c_f <= -1 || c_f >= cols)
                    continue;

                float x_coord = std::max(0.0f, std::min(r_f, static_cast<float>(rows - 1)));
                float y_coord = std::max(0.0f, std::min(c_f, static_cast<float>(cols - 1)));

                result.coords[n].push_back({x_coord, y_coord, 0.0f});
                result.pdfs[n].push_back(reg.pdfs[d]);
                result.infos[n].push_back({reg.x_vars[d], reg.y_vars[d], reg.rhos[d], reg.amps[d]});

                del_indices.push_back({n,
                    static_cast<int>(std::round(indices[d].row + reg.ys[d])),
                    static_cast<int>(std::round(indices[d].col + reg.xs[d]))});
                valid_pdfs.push_back(reg.pdfs[d]);
            }

            // Deflation
            if (!del_indices.empty() && df_loop < deflation_loop_backward - 1) {
                subtract_pdf(ext_imgs, nb_imgs, ext_rows, ext_cols,
                             valid_pdfs, del_indices, win, win, bg_means, 0);
            }
        }
    }

    return result;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Localize from pre-built extended images
// ============================================================
LocalizationResult localize_from_ext( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    const std::vector<float>& ext_imgs_in,
    int nb_imgs, int rows, int cols, int ext_rows, int ext_cols,
    int window_size, float threshold_alpha, int shift,
    int extend
) {
    // Generate window params
    std::vector<WinParams> single_ws, multi_ws; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> single_rad, multi_rad;
    params_gen(window_size, single_ws, single_rad, multi_ws, multi_rad);

    // We need original (non-extended) images for background computation // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int half_ext = extend / 2;
    std::vector<float> imgs(nb_imgs * rows * cols);
    for (int n = 0; n < nb_imgs; ++n)
        for (int r = 0; r < rows; ++r)
            for (int c = 0; c < cols; ++c)
                imgs[n * rows * cols + r * cols + c] = // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
                    ext_imgs_in[n * ext_rows * ext_cols + (r + half_ext) * ext_cols + (c + half_ext)];

    // Compute background from original images // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<WinParams> all_ws = single_ws;
    all_ws.insert(all_ws.end(), multi_ws.begin(), multi_ws.end());
    auto bg_result = compute_background(imgs, nb_imgs, rows, cols, all_ws, threshold_alpha);

    // Make a mutable copy of ext_imgs // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> ext_imgs(ext_imgs_in);

    // Generate Gaussian PSFs
    std::vector<Image2D> forward_grids, backward_grids; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    for (size_t i = 0; i < single_ws.size(); ++i)
        forward_grids.push_back(gauss_psf(single_ws[i].w, single_ws[i].h, single_rad[i]));
    for (size_t i = 0; i < multi_ws.size(); ++i)
        backward_grids.push_back(gauss_psf(multi_ws[i].w, multi_ws[i].h, multi_rad[i]));

    // Background means // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    auto& bg_vec = bg_result.bgs[multi_ws[0].w];
    std::vector<float> bg_means(nb_imgs);
    for (int n = 0; n < nb_imgs; ++n)
        bg_means[n] = bg_vec[n * multi_ws[0].w * multi_ws[0].h];

    // Initial parameters for Guo regression
    float p0[6] = {1.5f, 0.0f, 1.5f, 0.0f, 0.0f, 0.5f}; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    // Result
    LocalizationResult result;
    result.coords.resize(nb_imgs);
    result.pdfs.resize(nb_imgs);
    result.infos.resize(nb_imgs);

    // FORWARD PASS — same as localize() // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    {
        auto& ws0 = single_ws[0];
        auto& grid0 = forward_grids[0];

        int nb_crops = 0; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-12 20:50
        std::vector<float> crop_imgs;
        if (USE_GPU) {
            crop_imgs = gpu::image_cropping_gpu(ext_imgs, nb_imgs, ext_rows, ext_cols,
                                                 extend, ws0.w, ws0.h, shift, nb_crops);
        } else {
            crop_imgs = image_cropping(ext_imgs, nb_imgs, ext_rows, ext_cols,
                                         extend, ws0.w, ws0.h, shift, nb_crops);
        }

        std::vector<float> bg_sq_sums(nb_imgs);
        for (int n = 0; n < nb_imgs; ++n)
            bg_sq_sums[n] = ws0.w * ws0.h * bg_means[n] * bg_means[n];

        std::vector<float> lik;
        if (USE_GPU) {
            int sw = ws0.h * ws0.w;
            std::vector<float> gdata(sw);
            float gmean = image_mean(grid0);
            for (int i = 0; i < ws0.h; ++i)
                for (int j = 0; j < ws0.w; ++j)
                    gdata[i * ws0.w + j] = grid0.data[i * grid0.cols + j];
            lik = gpu::likelihood_gpu(crop_imgs, gdata, gmean, bg_sq_sums, bg_means,
                                       nb_imgs, nb_crops, sw);
        } else {
            lik = likelihood(crop_imgs, grid0, bg_sq_sums, bg_means,
                              nb_imgs, nb_crops, ws0.h, ws0.w);
        }

        auto h_map = mapping(lik, nb_imgs, rows, cols, shift);
        auto& thresholds = bg_result.thresholds;


        // NMS detection — identical to localize() // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
        int r_half = (shift == 0) ? ws0.h / 2 : shift;
        int c_half = (shift == 0) ? ws0.w / 2 : shift;
        struct CandInfo { int frame; int r; int c; float score; };
        std::vector<std::vector<CandInfo>> per_frame_cands(nb_imgs);

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < nb_imgs; ++n) {
            float thresh = thresholds[n];
            for (int r = 0; r < rows; ++r) {
                for (int c = 0; c < cols; ++c) {
                    int idx = n * rows * cols + r * cols + c;
                    if (h_map[idx] <= thresh) continue;
                    int r0 = std::max(0, r - r_half);
                    int r1 = std::min(rows, r + r_half + 1);
                    int c0 = std::max(0, c - c_half);
                    int c1 = std::min(cols, c + c_half + 1);
                    float local_max = 0.0f;
                    for (int ri = r0; ri < r1; ++ri)
                        for (int ci = c0; ci < c1; ++ci)
                            local_max = std::max(local_max, h_map[n * rows * cols + ri * cols + ci]);
                    if (h_map[idx] == local_max)
                        per_frame_cands[n].push_back({n, r, c, h_map[idx]});
                }
            }
        }

        std::vector<std::vector<DetIndex>> per_frame_indices(nb_imgs);
        int ext_nms = (shift == 0) ? (ws0.w - 1) / 2 : shift;

        #pragma omp parallel for schedule(static)
        for (int n = 0; n < nb_imgs; ++n) {
            auto& cands = per_frame_cands[n];
            std::sort(cands.begin(), cands.end(),
                      [](const CandInfo& a, const CandInfo& b) { return a.score > b.score; });
            std::vector<std::vector<uint8_t>> mask(rows, std::vector<uint8_t>(cols, 0));
            for (auto& cand : cands) {
                if (mask[cand.r][cand.c] != 0) continue;
                per_frame_indices[n].push_back({n, cand.r, cand.c});
                int r0 = std::max(0, cand.r - ext_nms);
                int r1 = std::min(rows - 1, cand.r + ext_nms);
                int c0 = std::max(0, cand.c - ext_nms);
                int c1 = std::min(cols - 1, cand.c + ext_nms);
                for (int ri = r0; ri < r1; ++ri)
                    for (int ci = c0; ci < c1; ++ci)
                        mask[ri][ci] = 1;
            }
        }
        std::vector<DetIndex> indices;
        for (int n = 0; n < nb_imgs; ++n)
            indices.insert(indices.end(), per_frame_indices[n].begin(), per_frame_indices[n].end());
        // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13

        if (!indices.empty()) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
            int ws = ws0.w;
            int win_area = ws * ws;
            int nb_det = static_cast<int>(indices.size());
            std::vector<float> regress_imgs(nb_det * win_area);
            std::vector<float> bg_regress(nb_det * win_area);
            std::vector<int> ns(nb_det), det_rs(nb_det), det_cs(nb_det);

            for (int d = 0; d < nb_det; ++d) {
                int n = indices[d].frame;
                int r = indices[d].row;
                int c = indices[d].col;
                ns[d] = n; det_rs[d] = r; det_cs[d] = c;
                int crop_idx = (r / shift) * (cols / shift) + (c / shift);
                int nb_crops_total = nb_crops;
                if (crop_idx < nb_crops_total) {
                    for (int p = 0; p < win_area; ++p)
                        regress_imgs[d * win_area + p] = crop_imgs[n * nb_crops_total * win_area + crop_idx * win_area + p];
                }
                auto& bg_for_ws = bg_result.bgs[ws];
                for (int p = 0; p < win_area; ++p)
                    bg_regress[d * win_area + p] = bg_for_ws[n * win_area + p];
            }

            auto reg = image_regression(regress_imgs, bg_regress, nb_det, ws, ws, p0, 5);

            std::vector<int> err_indices;
            for (int d = 0; d < nb_det; ++d) {
                if (reg.x_vars[d] < 0 || reg.y_vars[d] < 0 ||
                    reg.x_vars[d] > 3 * ws || reg.y_vars[d] > 3 * ws ||
                    reg.rhos[d] > 1 || reg.rhos[d] < -1)
                    err_indices.push_back(d);
            }

            std::vector<std::array<int, 3>> del_indices;
            for (int d = 0; d < nb_det; ++d) {
                if (std::find(err_indices.begin(), err_indices.end(), d) != err_indices.end())
                    continue;
                int n = ns[d];
                float r_f = det_rs[d] + reg.ys[d];
                float c_f = det_cs[d] + reg.xs[d];
                if (r_f <= -1 || r_f >= rows || c_f <= -1 || c_f >= cols)
                    continue;
                float x_coord = std::max(0.0f, std::min(r_f, static_cast<float>(rows - 1)));
                float y_coord = std::max(0.0f, std::min(c_f, static_cast<float>(cols - 1)));
                result.coords[n].push_back({x_coord, y_coord, 0.0f});
                result.pdfs[n].push_back(reg.pdfs[d]);
                result.infos[n].push_back({reg.x_vars[d], reg.y_vars[d], reg.rhos[d], reg.amps[d]});
                del_indices.push_back({n,
                    static_cast<int>(std::round(det_rs[d] + reg.ys[d])),
                    static_cast<int>(std::round(det_cs[d] + reg.xs[d]))});
            }

            if (!del_indices.empty()) {
                std::vector<std::vector<float>> valid_pdfs;
                int valid_idx = 0;
                for (int d = 0; d < nb_det; ++d) {
                    if (std::find(err_indices.begin(), err_indices.end(), d) != err_indices.end())
                        continue;
                    int n = ns[d];
                    float r_f = det_rs[d] + reg.ys[d];
                    float c_f = det_cs[d] + reg.xs[d];
                    if (r_f <= -1 || r_f >= rows || c_f <= -1 || c_f >= cols)
                        continue;
                    valid_pdfs.push_back(reg.pdfs[d]);
                }
                subtract_pdf(ext_imgs, nb_imgs, ext_rows, ext_cols,
                             valid_pdfs, del_indices, ws, ws, bg_means, extend);
            }
        }
    }

    return result; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
}

// ============================================================
// Top-level run
// ============================================================

bool run(const std::string& input_video_path, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
         const std::string& output_path,
         int window_size,
         float threshold,
         int shift,
         bool verbose,
         const std::string& ext_imgs_path,
         int batch_size) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
{
    std::filesystem::create_directories(output_path); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    int nb_frames, height, width;
    auto images = read_image(input_video_path, nb_frames, height, width); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    if (images.empty()) {
        std::cerr << "Failed to read: " << input_video_path << std::endl;
        return false;
    }

    // GPU detection: always use GPU if available // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    USE_GPU = gpu::is_available();
    // TEMP: allow forcing CPU via env var for comparison testing
    if (std::getenv("FREETRACE_FORCE_CPU")) USE_GPU = false;
    if (verbose) {
        std::cout << "Loaded " << nb_frames << " frames (" << height << "x" << width << ")" << std::endl;
        if (USE_GPU) {
            std::cout << "GPU detected (" << gpu::get_gpu_mem_size() << " GB free). Using CUDA acceleration." << std::endl;
        } else {
            std::cout << "No GPU detected. Running on CPU." << std::endl;
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    // Batch size — use parameter if given, otherwise compute dynamically // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    int div_q;
    int ws2 = window_size * window_size;
    if (batch_size > 0) {
        div_q = batch_size;
    } else if (USE_GPU) { // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
        // Estimate GPU memory per frame: ext_imgs + crops (dominant) + background raw imgs
        int ext_est = window_size;
        int ext_h = height + ext_est, ext_w = width + ext_est;
        int nb_crops_est = ((height + 1) / shift) * ((width + 1) / shift);
        size_t per_frame_crop = (size_t)nb_crops_est * ws2 * sizeof(float);
        size_t per_frame_ext = (size_t)ext_h * ext_w * sizeof(float);
        size_t per_frame_bg = (size_t)height * width * sizeof(float);
        size_t per_frame = per_frame_crop + per_frame_ext + per_frame_bg;
        size_t free_mem = gpu::get_gpu_free_mem_bytes();
        div_q = static_cast<int>(0.7 * free_mem / per_frame);
    } else {
        div_q = std::min(50, static_cast<int>(2.7 * 4194304.0 / height / width * (49.0 / ws2)));
    }
    div_q = std::clamp(div_q, 1, 500); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

    if (verbose)
        std::cout << "Batch size: " << div_q << std::endl;

    // Process in batches, matching Python's batch loop
    LocalizationResult result;
    result.coords.resize(nb_frames);
    result.pdfs.resize(nb_frames);
    result.infos.resize(nb_frames);

    // Compute extend for ext_imgs loading // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<WinParams> tmp_single, tmp_multi;
    std::vector<float> tmp_sr, tmp_mr;
    params_gen(window_size, tmp_single, tmp_sr, tmp_multi, tmp_mr);
    int extend = tmp_multi.back().w * 4; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int ext_rows = height + extend;
    int ext_cols = width + extend;

    // Load pre-computed extended images if path provided // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> loaded_ext_imgs;
    if (!ext_imgs_path.empty()) {
        std::ifstream efile(ext_imgs_path, std::ios::binary); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (!efile.is_open()) {
            std::cerr << "Failed to open ext_imgs: " << ext_imgs_path << std::endl;
            return false;
        }
        efile.seekg(0, std::ios::end);
        size_t file_size = efile.tellg(); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        efile.seekg(0, std::ios::beg);
        loaded_ext_imgs.resize(file_size / sizeof(float));
        efile.read(reinterpret_cast<char*>(loaded_ext_imgs.data()), file_size); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
        if (verbose)
            std::cout << "Loaded ext_imgs: " << loaded_ext_imgs.size() << " floats from " << ext_imgs_path << std::endl;
    }

    int frame_size = height * width;
    int ext_frame_size = ext_rows * ext_cols;

    // Create a persistent RNG (seed=42) that carries state across batches, // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14
    // matching Python which calls add_block_noise once for all frames.
    NumpyRNG batch_rng(42);

    for (int batch_start = 0; batch_start < nb_frames; batch_start += div_q) {
        int batch_end = std::min(batch_start + div_q, nb_frames);
        int batch_n = batch_end - batch_start;

        if (verbose) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
            std::cerr << "\rLocalizing frame " << batch_start << "-" << batch_end << " / " << nb_frames << std::flush;

        LocalizationResult batch_result;

        if (!ext_imgs_path.empty()) {
            // Use pre-computed extended images
            std::vector<float> batch_ext(batch_n * ext_frame_size);
            for (int i = 0; i < batch_n * ext_frame_size; ++i)
                batch_ext[i] = loaded_ext_imgs[batch_start * ext_frame_size + i];
            batch_result = localize_from_ext(batch_ext, batch_n, height, width,
                                             ext_rows, ext_cols, window_size, threshold, shift, extend);
        } else {
            // Normal path: pass persistent RNG so noise sequence is continuous across batches
            std::vector<float> batch_imgs(batch_n * frame_size);
            for (int i = 0; i < batch_n * frame_size; ++i)
                batch_imgs[i] = images[batch_start * frame_size + i];
            batch_result = localize(batch_imgs, batch_n, height, width,
                                     window_size, threshold, shift, /*deflation_loop_backward=*/0,
                                     &batch_rng);
        }

        // Merge results with correct frame offset
        for (int i = 0; i < batch_n; ++i) {
            int global_frame = batch_start + i;
            result.coords[global_frame] = std::move(batch_result.coords[i]);
            result.pdfs[global_frame] = std::move(batch_result.pdfs[i]);
            result.infos[global_frame] = std::move(batch_result.infos[i]);
        }
    } // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-14

    if (verbose) std::cerr << std::endl; // end progress line // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15

    // Count total detections
    int total = 0;
    for (auto& frame_coords : result.coords) total += frame_coords.size();

    if (verbose)
        std::cout << "Localization complete: " << total << " detections across "
                  << nb_frames << " frames." << std::endl;

    // Write output // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    auto last_sep = input_video_path.find_last_of("/\\");
    std::string vid_fname = (last_sep != std::string::npos)
        ? input_video_path.substr(last_sep + 1) : input_video_path;
#ifdef _WIN32
    std::string loc_output = output_path + "\\" + vid_fname;
#else
    std::string loc_output = output_path + "/" + vid_fname;
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-13
    // Strip known extensions: .tif, .tiff, .nd2 // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    auto tif_pos = loc_output.find(".tif");
    if (tif_pos != std::string::npos) loc_output = loc_output.substr(0, tif_pos);
    auto nd2_pos = loc_output.find(".nd2");
    if (nd2_pos != std::string::npos) loc_output = loc_output.substr(0, nd2_pos); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-15
    write_localization_csv(loc_output, result);
    make_loc_depth_image(loc_output, result, /*multiplier=*/4, /*winsize=*/window_size, /*resolution=*/2); // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    if (verbose)
        std::cout << "Written to " << loc_output << "_loc.csv and _loc_2d_density.png" << std::endl; // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

    return true;
} // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

// ============================================================ // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// 2D density image (matching Python make_loc_depth_image)
// ============================================================

// Matplotlib 'hot' colormap LUT (256 entries, RGB) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
static void hot_colormap(float t, uint8_t& r, uint8_t& g, uint8_t& b) {
    // t in [0, 1]
    t = std::max(0.0f, std::min(1.0f, t));
    // Red: ramps 0->1 over [0, 0.375]
    float rf = std::min(1.0f, t / 0.375f);
    // Green: ramps 0->1 over [0.375, 0.75]
    float gf = (t < 0.375f) ? 0.0f : std::min(1.0f, (t - 0.375f) / 0.375f);
    // Blue: ramps 0->1 over [0.75, 1.0]
    float bf = (t < 0.75f) ? 0.0f : (t - 0.75f) / 0.25f;
    r = static_cast<uint8_t>(rf * 255.0f);
    g = static_cast<uint8_t>(gf * 255.0f);
    b = static_cast<uint8_t>(bf * 255.0f);
}

#ifdef USE_LIBPNG // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
#include <png.h>
static bool write_png(const std::string& path, const uint8_t* data, int width, int height) {
    FILE* fp = fopen(path.c_str(), "wb");
    if (!fp) return false;
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    png_infop info = png_create_info_struct(png);
    if (setjmp(png_jmpbuf(png))) { fclose(fp); return false; }
    png_init_io(png, fp);
    png_set_IHDR(png, info, width, height, 8, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png, info);
    for (int y = 0; y < height; ++y)
        png_write_row(png, data + y * width * 3);
    png_write_end(png, nullptr);
    png_destroy_write_struct(&png, &info);
    fclose(fp);
    return true;
}
#else // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
// Fallback: write PPM (no dependencies)
static bool write_png(const std::string& path, const uint8_t* data, int width, int height) {
    std::string ppm_path = path.substr(0, path.rfind('.')) + ".ppm";
    std::ofstream f(ppm_path, std::ios::binary);
    if (!f.is_open()) return false;
    f << "P6\n" << width << " " << height << "\n255\n";
    f.write(reinterpret_cast<const char*>(data), width * height * 3);
    return true;
}
#endif // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11

void make_loc_depth_image( // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    const std::string& output_path,
    const LocalizationResult& result,
    int multiplier, int winsize, int resolution
) {
    resolution = std::max(1, std::min(3, resolution));
    if (multiplier % 2 == 1) multiplier -= 1;
    winsize += multiplier * resolution;
    int cov_std = multiplier * resolution;
    int amp = 1;
    float amp_ = static_cast<float>(std::pow(10, amp));
    float margin_pixel = 2.0f * 10.0f * amp_;
    amp_ *= resolution;

    // Collect all coords (Python stores as [y_row, x_col, z]) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<std::array<float, 3>> all_coords;
    for (auto& frame_coords : result.coords)
        for (auto& c : frame_coords)
            all_coords.push_back(c);

    if (all_coords.empty()) return;

    // In Python: coords are [row, col, z], CSV writes x=col, y=row
    // all_coords[i] = {row, col, z} from result.coords
    // Python does: all_coords[:,1] -= x_min (col), all_coords[:,0] -= y_min (row) // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    float x_min = 1e30f, x_max = -1e30f; // x = col = [1]
    float y_min = 1e30f, y_max = -1e30f; // y = row = [0]
    for (auto& c : all_coords) {
        x_min = std::min(x_min, c[1]); x_max = std::max(x_max, c[1]);
        y_min = std::min(y_min, c[0]); y_max = std::max(y_max, c[0]);
    }
    for (auto& c : all_coords) {
        c[1] -= x_min;
        c[0] -= y_min;
    }

    // Image dimensions // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    int img_h = static_cast<int>((y_max - y_min) * amp_ + margin_pixel);
    int img_w = static_cast<int>((x_max - x_min) * amp_ + margin_pixel);

    // Build Gaussian template // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> templ(winsize * winsize);
    int half = winsize / 2;
    float inv_cov = 1.0f / static_cast<float>(cov_std);
    for (int r = 0; r < winsize; ++r) {
        for (int c = 0; c < winsize; ++c) {
            float y = static_cast<float>(r - half);
            float x = static_cast<float>(c - half);
            templ[r * winsize + c] = std::exp(-0.5f * (x * x + y * y) * inv_cov);
        }
    }

    // Accumulate density // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> image(img_h * img_w, 0.0f);
    int margin_half = static_cast<int>(margin_pixel) / 2;

    for (auto& c : all_coords) {
        int coord_col = static_cast<int>(std::round(c[1] * amp_)) + margin_half;
        int coord_row = static_cast<int>(std::round(c[0] * amp_)) + margin_half;
        int row = std::min(std::max(0, coord_row), img_h - 1);
        int col = std::min(std::max(0, coord_col), img_w - 1);

        for (int ri = 0; ri < winsize; ++ri) {
            int ir = row - half + ri;
            if (ir < 0 || ir >= img_h) continue;
            for (int ci = 0; ci < winsize; ++ci) {
                int ic = col - half + ci;
                if (ic < 0 || ic >= img_w) continue;
                image[ir * img_w + ic] += templ[ri * winsize + ci];
            }
        }
    }

    // Normalize: quantile over ALL pixels (including zeros), matching Python np.quantile // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<float> sorted_vals(image.begin(), image.end());
    std::sort(sorted_vals.begin(), sorted_vals.end());
    float img_max = sorted_vals[static_cast<int>(sorted_vals.size() * 0.995)];
    if (img_max <= 0.0f) return;

    for (auto& v : image)
        v = std::min(v, img_max);
    float global_max = *std::max_element(image.begin(), image.end());
    if (global_max > 0.0f)
        for (auto& v : image) v /= global_max;

    // Apply colormap and write PNG // Modified by Claude (claude-opus-4-6, Anthropic AI) - 2026-03-11
    std::vector<uint8_t> rgb(img_h * img_w * 3);
    for (int i = 0; i < img_h * img_w; ++i)
        hot_colormap(image[i], rgb[i * 3], rgb[i * 3 + 1], rgb[i * 3 + 2]);

    std::string png_path = output_path + "_loc_2d_density.png";
    write_png(png_path, rgb.data(), img_w, img_h);
}

} // namespace freetrace
