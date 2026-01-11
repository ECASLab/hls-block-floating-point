#include <iostream>
#include <iomanip>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>

// XRT includes
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

// Profiler
#include "timer.hpp"

// BFP common definitions
#include "common_bfp.h"

// Helper to compute metrics (MAE and MAPE)
void compute_metrics(const float* ref, const float* got, unsigned int len,
                     double& mae, double& mape) {
    double abs_sum = 0.0, ape_sum = 0.0;
    unsigned int mape_cnt = 0;
    for (unsigned int i = 0; i < len; ++i) {
        const double r = ref[i], g = got[i];
        const double ae = std::fabs(g - r);
        abs_sum += ae;
        if (std::fabs(r) > 1e-12) { 
            ape_sum += ae / std::fabs(r); 
            ++mape_cnt; 
        }
    }
    mae = abs_sum / double(len);
    mape = (mape_cnt ? (ape_sum / double(mape_cnt)) * 100.0 : 0.0);
}

// Simple CPU-side BFP encoding (UPDATED with delta calculation)
struct SimpleBFP {
    unsigned int exp_shared;
    std::vector<unsigned int> sign;
    std::vector<unsigned int> mant;
    std::vector<unsigned int> delta; 
    
    SimpleBFP(unsigned int n) : exp_shared(0), sign(n, 0), mant(n, 0), delta(n, 0) {}
};

SimpleBFP encode_fp32_to_bfp(const float* data, unsigned int n) {
    SimpleBFP result(n);
    
    int max_exp = -1000;
    bool has_special = false;  
    
    for (unsigned int i = 0; i < n; i++) {
        if (data[i] == 0.0f) continue;
        
        union {float f; uint32_t u;} u = {data[i]};
        int exp = int((u.u >> 23) & 0xFF);
        
        // NaN/Inf
        if (exp == 0xFF) {
            has_special = true;
            continue;
        }
        
        if (exp > 0) {
            int exp_unbiased = exp - 127;
            if (exp_unbiased > max_exp) max_exp = exp_unbiased;
        }
    }
    
    if (max_exp == -1000) {
        // Special cases
        if (has_special) {
            max_exp = (1 << WE) - 1 - ((1 << (WE - 1)) - 1);
        } else {
            return result;
        }
    }
    
    if (has_special && WM >= 4) {
        max_exp = (1 << WE) - 1 - ((1 << (WE - 1)) - 1);
    }
    
    int exp_shared_bfp = max_exp + ((1 << (WE - 1)) - 1);
    if (exp_shared_bfp < 0) exp_shared_bfp = 0;
    if (exp_shared_bfp > 31) exp_shared_bfp = 31;
    result.exp_shared = exp_shared_bfp;
    
    const uint32_t mant_max = (1u << (WM + 1)) - 1;
    
    for (unsigned int i = 0; i < n; i++) {
        float num_fp32 = data[i];
        
        if (num_fp32 == 0.0f) {
            result.sign[i] = 0;
            result.mant[i] = 0;
            result.delta[i] = 0;
            continue;
        }
        
        union {float f; uint32_t u;} u = {num_fp32};
        uint32_t s = (u.u >> 31) & 0x1;
        int exp_fp32 = int((u.u >> 23) & 0xFF);
        uint32_t mant_fp32 = u.u & 0x7FFFFF;
        
        if (exp_fp32 == 0xFF) {
            result.sign[i] = s;
            if (mant_fp32 == 0) {
                // Infinity
                result.mant[i] = mant_max;
                result.delta[i] = 0;
            } else {
                // NaN
                result.mant[i] = mant_max - 1;
                result.delta[i] = 0;
            }
            continue;
        }
        
        if (exp_fp32 == 0) {
            result.sign[i] = 0;
            result.mant[i] = 0;
            result.delta[i] = 0;
            continue;
        }
        
        uint32_t mant24 = (u.u & 0x7FFFFF) | (1u << 23);
        int exp_unbiased = exp_fp32 - 127;
        
        int delta_val = max_exp - exp_unbiased;
        
        if (delta_val < 0) delta_val = 0;
        if (delta_val > ((1 << WM) - 1)) delta_val = ((1 << WM) - 1);
        
        result.delta[i] = uint32_t(delta_val);
        
        int shift = (23 - WM) + delta_val;
        
        uint32_t mant_reduced;
        if (shift >= 32) {
            mant_reduced = 0;
        } else if (shift < 0) {
            mant_reduced = mant24 << (-shift);
        } else {
            uint32_t q = mant24 >> shift;
            uint32_t rem = mant24 & ((1u << shift) - 1);
            uint32_t half = 1u << (shift - 1);
            if (rem > half || (rem == half && (q & 1))) {
                q++;
            }
            mant_reduced = q;
        }
        
        if (mant_reduced > mant_max) mant_reduced = mant_max;
        
        result.sign[i] = s;
        result.mant[i] = mant_reduced;
    }
    
    return result;
}

// Helper to decode BFP element to FP32 (matches HW rebuild_FP32)
float decode_bfp_to_fp32(uint32_t exp_shared, uint32_t sign, uint32_t mant, uint32_t delta) {
    const uint32_t mant_max = (1u << (WM + 1)) - 1;
    const int bias = (1 << (WE - 1)) - 1;  
    
    // NaN
    if (mant == (mant_max - 1) && delta == 0) {
        union {float f; uint32_t u;} nan_val;
        nan_val.u = 0x7FC00000;
        return nan_val.f;
    }
    
    // Detect Infinity
    if (mant == mant_max && delta == 0) {
        union {float f; uint32_t u;} inf_val;
        inf_val.u = sign ? 0xFF800000 : 0x7F800000;
        return inf_val.f;
    }
    
    // Detect zero
    if (exp_shared == 0 && mant == 0) return 0.0f;
    
    // Reconstruction
    int exp_shared_unbiased = int(exp_shared) - bias;
    int exp_real = exp_shared_unbiased - int(delta);
    
    uint32_t mant_unshifted = mant << delta;
    
    float mant_val = float(mant_unshifted) / float(1u << WM);
    float value = ldexpf(mant_val, exp_real);
    
    return sign ? -value : value;
}

int main(int argc, char** argv) {
    INIT_PROFILER(bfp_profiler)
    int device_index = 0;

    // Initial validation host settings
    std::cout << "========================================" << std::endl;
    std::cout << "BFP Configuration Check:" << std::endl;
    std::cout << "  WE=" << WE << ", WM=" << WM << ", N=" << N << std::endl;
    std::cout << "  BIAS=" << ((1 << (WE - 1)) - 1) << std::endl;
    std::cout << "  MANT_MAX=" << ((1 << (WM + 1)) - 1) << std::endl;
    std::cout << "  DELTA_MAX=" << ((1 << WM) - 1) << std::endl;
    std::cout << "  BFP_BLOCK_SIZE=" << BFP_BLOCK_SIZE << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << std::endl;

    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <operation> <n_blocks>" << std::endl;
        std::cerr << "  operation: 0=ENCODE, 1=DECODE, 2=ADD, 3=SUB, 4=MUL, 5=DIV, 6=RCP" << std::endl;
        std::cerr << "  n_blocks: number of blocks (e.g., 2)" << std::endl;
        return EXIT_FAILURE;
    }

    // Get input parameters
    static std::string binaryFile = "../HW/package.hw/kernels.xclbin";
    unsigned int operation = std::stoi(argv[1]);
    unsigned int n_blocks = std::stoi(argv[2]);

    if (operation > 6) {
        std::cerr << "Error: Invalid operation code. Must be 0-6" << std::endl;
        return EXIT_FAILURE;
    }

    std::cout << "========================================" << std::endl;
    std::cout << "BFP Accelerator Test (COMPACT FORMAT)" << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "Operation: " << OP_NAMES[operation] << " (" << operation << ")" << std::endl;
    std::cout << "Number of blocks: " << n_blocks << std::endl;
    std::cout << "Block size (N): " << N << std::endl;
    std::cout << "BFP Config: WE=" << WE << ", WM=" << WM << std::endl;
    std::cout << "BFP_BLOCK_SIZE: " << BFP_BLOCK_SIZE << " uints/block" << std::endl;
    std::cout << std::endl;

    // Compute sizes
    unsigned int size_fp32 = n_blocks * N;
    unsigned int size_bfp = n_blocks * BFP_BLOCK_SIZE;  // compact format

    GET_PROFILE_INSTANCE(setup_time, bfp_profiler);
    setup_time->reset();

    std::cout << "Opening device " << device_index << "..." << std::endl;
    auto device = xrt::device(device_index);
    
    std::cout << "Loading xclbin: " << binaryFile << "..." << std::endl;
    auto uuid = device.load_xclbin(binaryFile);
    
    std::cout << "Creating kernel handle..." << std::endl;
    auto bfp_kernel = xrt::kernel(device, uuid, "bfp_kernel");
    
    setup_time->tick();

    std::cout << "Allocating buffers in global memory..." << std::endl;
    
    // UPDATED: Kernel arguments for COMPACT format
    //   0: operation (scalar)
    //   1: n_blocks (scalar)
    //   2: in_fp32     -> gmem0
    //   3: in_bfp_a    -> gmem1 (COMPACT: n_blocks * BFP_BLOCK_SIZE uints)
    //   4: in_bfp_b    -> gmem1 (COMPACT: n_blocks * BFP_BLOCK_SIZE uints)
    //   5: out_fp32    -> gmem0
    //   6: out_bfp     -> gmem2 (COMPACT: n_blocks * BFP_BLOCK_SIZE uints)
    
    auto bo_in_fp32  = xrt::bo(device, size_fp32 * sizeof(float), bfp_kernel.group_id(2));
    auto bo_in_bfp_a = xrt::bo(device, size_bfp * sizeof(uint32_t), bfp_kernel.group_id(3));
    auto bo_in_bfp_b = xrt::bo(device, size_bfp * sizeof(uint32_t), bfp_kernel.group_id(4));
    auto bo_out_fp32 = xrt::bo(device, size_fp32 * sizeof(float), bfp_kernel.group_id(5));
    auto bo_out_bfp  = xrt::bo(device, size_bfp * sizeof(uint32_t), bfp_kernel.group_id(6));

    // Map buffers
    auto bo_in_fp32_map  = bo_in_fp32.map<float*>();
    auto bo_in_bfp_a_map = bo_in_bfp_a.map<uint32_t*>();
    auto bo_in_bfp_b_map = bo_in_bfp_b.map<uint32_t*>();
    auto bo_out_fp32_map = bo_out_fp32.map<float*>();
    auto bo_out_bfp_map  = bo_out_bfp.map<uint32_t*>();

    // Test data - Expanded to N=32 elements per block
    std::cout << "Preparing test data..." << std::endl;
    
    // Pattern 0: Mixed values
    float A0[N] = {
        12.35f,  6.50f, 10.20f,  6.60f,  8.80f,  2.56f, 11.11f,  8.00f,
         5.45f,  9.99f,  0.15f, 18.00f,  3.80f, 90.10f, 14.00f, 10.00f,
         7.25f, 13.50f,  4.75f, 16.80f,  9.20f,  1.90f, 12.60f,  5.30f,
        15.40f,  8.70f, 11.90f,  6.10f,  2.40f, 19.50f,  3.15f, 17.80f
    };
    
    // Pattern 1: Sequential values
    float A1[N] = {
         1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
         9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f,
        17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f,
        25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f, 32.0f
    };
    
    float B0[N] = {
        2.0f, 1.0f, 2.0f, 3.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 5.0f, 3.0f, 6.0f, 3.0f, 8.0f, 2.0f,
        4.0f, 2.0f, 3.0f, 2.0f, 5.0f, 2.0f, 3.0f, 4.0f,
        2.0f, 3.0f, 2.0f, 5.0f, 2.0f, 3.0f, 6.0f, 2.0f
    };
    
    float B1[N] = {
        15.0f, 14.0f, 13.0f, 12.0f, 11.0f, 10.0f,  9.0f,  8.0f,
         7.0f,  6.0f,  5.0f,  4.0f,  3.0f,  2.0f,  1.0f,  0.5f,
        16.0f, 15.5f, 14.5f, 13.5f, 12.5f, 11.5f, 10.5f,  9.5f,
         8.5f,  7.5f,  6.5f,  5.5f,  4.5f,  3.5f,  2.5f,  1.5f
    };
    
    // Pattern 2: Powers of 2
    float A2[N] = {
        64.0f, 128.0f, 256.0f, 512.0f,  32.0f,  16.0f,   8.0f,   4.0f,
         2.0f,   1.0f,   0.5f,   0.25f,   0.125f, 96.0f,  48.0f,  24.0f,
       1024.0f, 2048.0f, 4096.0f, 8192.0f, 16384.0f, 0.0625f, 0.03125f, 192.0f,
       384.0f, 768.0f, 1536.0f, 3072.0f, 6144.0f, 12.0f, 6.0f, 3.0f
    };
    
    float B2[N] = {
        2.0f, 4.0f, 8.0f, 16.0f, 2.0f, 2.0f, 2.0f, 2.0f,
        2.0f, 2.0f, 2.0f, 2.0f,  2.0f, 3.0f, 3.0f, 3.0f,
        4.0f, 4.0f, 4.0f, 4.0f,  8.0f, 2.0f, 2.0f, 3.0f,
        3.0f, 3.0f, 3.0f, 3.0f,  3.0f, 2.0f, 2.0f, 2.0f
    };
    
    // Pattern 3: Fractional values
    float A3[N] = {
        0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f,
        0.9f, 1.1f, 1.2f, 1.3f, 1.4f, 1.5f, 1.6f, 1.7f,
        1.8f, 1.9f, 2.1f, 2.2f, 2.3f, 2.4f, 2.5f, 2.6f,
        2.7f, 2.8f, 2.9f, 3.1f, 3.2f, 3.3f, 3.4f, 3.5f
    };
    
    float B3[N] = {
        0.5f, 0.5f, 0.5f, 0.5f, 1.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f,
        0.25f, 0.25f, 0.75f, 0.75f, 1.5f, 1.5f, 2.5f, 2.5f,
        3.5f, 3.5f, 4.0f, 4.0f, 5.0f, 5.0f, 6.0f, 6.0f
    };
    
    // Pattern 4: Mixed signs
    float A4[N] = {
       -12.5f,  8.0f, -6.25f, 15.0f, -3.5f, 20.0f, -9.0f,  7.5f,
        -4.25f, 11.0f, -2.75f, 13.5f, -8.5f, 5.0f, -10.5f, 16.0f,
       -14.0f,  9.5f, -7.0f,  12.0f, -5.5f, 18.0f, -11.0f,  6.5f,
        -3.0f, 14.5f, -9.5f,  17.0f, -6.0f,  4.5f, -13.0f, 19.5f
    };
    
    float B4[N] = {
         2.0f, -2.0f,  2.0f, -2.0f,  2.0f, -2.0f,  2.0f, -2.0f,
         3.0f, -3.0f,  3.0f, -3.0f,  4.0f, -4.0f,  4.0f, -4.0f,
         2.5f, -2.5f,  3.5f, -3.5f,  4.5f, -4.5f,  5.0f, -5.0f,
         2.0f, -2.0f,  3.0f, -3.0f,  4.0f, -4.0f,  5.0f, -5.0f
    };
    
    // Pattern 5: Large values
    float A5[N] = {
       100.0f,  99.5f,  98.25f,  97.0f,  95.5f,  94.0f,  92.5f,  91.0f,
        89.5f,  88.0f,  86.5f,  85.0f,  83.5f,  82.0f,  80.5f,  79.0f,
       105.0f, 104.5f, 103.0f, 101.5f, 200.0f, 198.5f, 196.0f, 193.5f,
       150.0f, 148.5f, 146.0f, 143.5f, 140.0f, 137.5f, 135.0f, 132.5f
    };
    
    float B5[N] = {
        10.0f,  9.5f,  9.0f,  8.5f,  8.0f,  7.5f,  7.0f,  6.5f,
         6.0f,  5.5f,  5.0f,  4.5f,  4.0f,  3.5f,  3.0f,  2.5f,
        11.0f, 10.5f, 10.0f,  9.5f, 20.0f, 19.5f, 18.0f, 16.5f,
        15.0f, 14.5f, 13.0f, 11.5f, 10.0f,  9.0f,  8.0f,  7.0f
    };

    // Initialize all buffers (UPDATED)
    std::fill(bo_in_fp32_map, bo_in_fp32_map + size_fp32, 0.0f);
    std::fill(bo_in_bfp_a_map, bo_in_bfp_a_map + size_bfp, 0u);
    std::fill(bo_in_bfp_b_map, bo_in_bfp_b_map + size_bfp, 0u);
    std::fill(bo_out_fp32_map, bo_out_fp32_map + size_fp32, 0.0f);
    std::fill(bo_out_bfp_map, bo_out_bfp_map + size_bfp, 0u);

    // Prepare test data based on operation (UNCHANGED)
    std::vector<float> A_fp(size_fp32), B_fp(size_fp32);
    std::vector<float> golden_ref(size_fp32);
    
    // Fill data for all blocks
    for (unsigned int blk = 0; blk < n_blocks; ++blk) {
        unsigned int offset = blk * N;
        
        switch(blk % 6) {  // Cycle through 6 patterns
            case 0:
                std::memcpy(&A_fp[offset], A0, sizeof(float) * N);
                std::memcpy(&B_fp[offset], B0, sizeof(float) * N);
                break;
            case 1:
                std::memcpy(&A_fp[offset], A1, sizeof(float) * N);
                std::memcpy(&B_fp[offset], B1, sizeof(float) * N);
                break;
            case 2:
                std::memcpy(&A_fp[offset], A2, sizeof(float) * N);
                std::memcpy(&B_fp[offset], B2, sizeof(float) * N);
                break;
            case 3:
                std::memcpy(&A_fp[offset], A3, sizeof(float) * N);
                std::memcpy(&B_fp[offset], B3, sizeof(float) * N);
                break;
            case 4:
                std::memcpy(&A_fp[offset], A4, sizeof(float) * N);
                std::memcpy(&B_fp[offset], B4, sizeof(float) * N);
                break;
            case 5:
                std::memcpy(&A_fp[offset], A5, sizeof(float) * N);
                std::memcpy(&B_fp[offset], B5, sizeof(float) * N);
                break;
        }
    }

    // Compute golden reference (UNCHANGED)
    for (unsigned int i = 0; i < size_fp32; ++i) {
        switch (operation) {
            case OP_ENCODE:
            case OP_DECODE:
                golden_ref[i] = A_fp[i];
                break;
            case OP_ADD:
                golden_ref[i] = A_fp[i] + B_fp[i];
                break;
            case OP_SUB:
                golden_ref[i] = A_fp[i] - B_fp[i];
                break;
            case OP_MUL:
                golden_ref[i] = A_fp[i] * B_fp[i];
                break;
            case OP_DIV:
                golden_ref[i] = (std::fabs(B_fp[i]) > 1e-30f) ? (A_fp[i] / B_fp[i]) : 0.f;
                break;
            case OP_RCP:
                golden_ref[i] = (std::fabs(B_fp[i]) > 1e-30f) ? (1.0f / B_fp[i]) : 0.f;
                break;
        }
    }

    // Fill input buffers based on operation (UPDATED: pack to compact format)
    if (operation == OP_ENCODE) {
        // ENCODE: input is FP32
        std::memcpy(bo_in_fp32_map, A_fp.data(), sizeof(float) * size_fp32);
        
    } else if (operation == OP_DECODE) {
        // DECODE: input is BFP - encode A on CPU and pack
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP bfp_a = encode_fp32_to_bfp(&A_fp[fp_offset], N);
            pack_bfp_to_compact(bfp_a.exp_shared, bfp_a.sign.data(), 
                               bfp_a.mant.data(), bfp_a.delta.data(),
                               bo_in_bfp_a_map, bfp_offset);
        }
        
    } else if (operation == OP_RCP) {
        // RCP: input is BFP B only
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP bfp_b = encode_fp32_to_bfp(&B_fp[fp_offset], N);
            pack_bfp_to_compact(bfp_b.exp_shared, bfp_b.sign.data(),
                               bfp_b.mant.data(), bfp_b.delta.data(),
                               bo_in_bfp_b_map, bfp_offset);
        }
        
    } else {
        // Binary ops: encode both A and B
        for (unsigned int blk = 0; blk < n_blocks; ++blk) {
            unsigned int fp_offset = blk * N;
            unsigned int bfp_offset = blk * BFP_BLOCK_SIZE;
            
            SimpleBFP bfp_a = encode_fp32_to_bfp(&A_fp[fp_offset], N);
            SimpleBFP bfp_b = encode_fp32_to_bfp(&B_fp[fp_offset], N);
            
            pack_bfp_to_compact(bfp_a.exp_shared, bfp_a.sign.data(),
                               bfp_a.mant.data(), bfp_a.delta.data(),
                               bo_in_bfp_a_map, bfp_offset);
            pack_bfp_to_compact(bfp_b.exp_shared, bfp_b.sign.data(),
                               bfp_b.mant.data(), bfp_b.delta.data(),
                               bo_in_bfp_b_map, bfp_offset);
        }
    }

    std::cout << "Syncing input buffers to device..." << std::endl;
    
    START_PROFILE(kernel_execution, bfp_profiler, 10)
    
    bo_in_fp32.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_bfp_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_in_bfp_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::cout << "Executing kernel: " << OP_NAMES[operation] << "..." << std::endl;
    
    // UPDATED: Kernel call with 7 arguments (compact format)
    auto run = bfp_kernel(
        operation,
        n_blocks,
        bo_in_fp32,
        bo_in_bfp_a,
        bo_in_bfp_b,
        bo_out_fp32,
        bo_out_bfp
    );
    
    run.wait();
    std::cout << "Kernel completed!" << std::endl;

    std::cout << "Reading output buffers from device..." << std::endl;
    bo_out_fp32.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    bo_out_bfp.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    
    END_PROFILE(kernel_execution);

    // Display results (showing first 8 elements + raw compact vector for ENCODE)
    std::cout << "\n========================================" << std::endl;
    std::cout << "Results" << std::endl;
    std::cout << "========================================" << std::endl;
    
    if (operation == OP_ENCODE) {
        // Show raw compact vector for first block
        std::cout << "\nFirst block - Raw compact vector (first 25 values):" << std::endl;
        std::cout << "  [";
        for (int i = 0; i < 25 && i < BFP_BLOCK_SIZE; ++i) {
            std::cout << "0x" << std::hex << std::setw(8) << std::setfill('0') 
                      << bo_out_bfp_map[i] << std::dec;
            if (i < 24) std::cout << ", ";
            if ((i + 1) % 8 == 0) std::cout << "\n   ";
        }
        std::cout << "]" << std::endl;
        
        // Unpack and show interpreted values
        uint32_t exp_out;
        uint32_t sign_out[N], mant_out[N], delta_out[N];
        unpack_compact_to_bfp(bo_out_bfp_map, 0, exp_out, sign_out, mant_out, delta_out);
        
        std::cout << "\nFirst block - Decoded format (first 8 elements):" << std::endl;
        std::cout << "  exp_shared: " << exp_out << std::endl;
        for (int i = 0; i < 8; ++i) {
            std::cout << "  [" << i << "] sign: " << sign_out[i]
                      << ", mant: " << mant_out[i]
                      << ", delta: " << delta_out[i] << std::endl;
        }
        
    } else if (operation == OP_DECODE) {
        std::cout << "\nFirst block - FP32 output (first 8 elements):" << std::endl;
        for (int i = 0; i < 8; ++i) {
            std::cout << "  [" << i << "] FP32: " << bo_out_fp32_map[i] 
                      << " (expected: " << golden_ref[i] << ")" << std::endl;
        }
        
    } else {
        // Arithmetic operations: show BFP result and decode to FP32
        uint32_t exp_out;
        uint32_t sign_out[N], mant_out[N], delta_out[N];
        unpack_compact_to_bfp(bo_out_bfp_map, 0, exp_out, sign_out, mant_out, delta_out);
        
        std::cout << "\nFirst block - " << OP_NAMES[operation] << " result (first 8 elements):" << std::endl;
        std::cout << "  exp_shared: " << exp_out << std::endl;
        std::cout << std::endl;
        
        // Show operation with decoded values
        for (int i = 0; i < 8; ++i) {
            float result_fp32 = decode_bfp_to_fp32(exp_out, sign_out[i], mant_out[i], delta_out[i]);
            
            std::cout << "  [" << i << "] ";
            
            // Show operation in decimal
            if (operation == OP_RCP) {
                std::cout << "1 / " << B_fp[i] << " = " << result_fp32;
            } else {
                std::cout << A_fp[i];
                switch(operation) {
                    case OP_ADD: std::cout << " + "; break;
                    case OP_SUB: std::cout << " - "; break;
                    case OP_MUL: std::cout << " * "; break;
                    case OP_DIV: std::cout << " / "; break;
                }
                std::cout << B_fp[i] << " = " << result_fp32;
            }
            
            // Show expected value
            std::cout << " (expected: " << golden_ref[i] << ")";
            
            // Show BFP representation
            std::cout << " [BFP: sign=" << sign_out[i]
                      << ", mant=" << mant_out[i]
                      << ", delta=" << delta_out[i] << "]" << std::endl;
        }
    }

    // Validate results (UNCHANGED logic)
    if (operation == OP_DECODE) {
        double mae = 0.0, mape = 0.0;
        compute_metrics(golden_ref.data(), bo_out_fp32_map, size_fp32, mae, mape);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Accuracy Metrics" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "MAE:  " << mae << std::endl;
        std::cout << "MAPE: " << mape << "%" << std::endl;
        
        bool passed = (mae < 1.0 && mape < 10.0);
        std::cout << "\n" << (passed ? "✓ TEST PASSED" : "✗ TEST FAILED") << std::endl;
        
    } else if (operation == OP_ENCODE) {
        // For ENCODE, verify that output has non-zero data
        bool has_data = false;
        for (unsigned int i = 0; i < size_bfp; ++i) {
            if (bo_out_bfp_map[i] != 0) {
                has_data = true;
                break;
            }
        }
        std::cout << "\n" << (has_data ? "✓ TEST PASSED (data encoded)" : "✗ TEST FAILED (no data)") << std::endl;
        
    } else {
        // For arithmetic operations
        std::cout << "\nNote: Arithmetic operation completed." << std::endl;
        std::cout << "✓ TEST COMPLETED" << std::endl;
    }

    std::cout << "\n" << bfp_profiler << std::endl;
    std::cout << "\n========================================" << std::endl;

    return 0;
}
