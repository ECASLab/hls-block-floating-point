#ifndef BFP_H
#define BFP_H

#include <ap_int.h>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <array>
#include <cmath>

//* Bias settings for BFP format
template<int WE_BITS, int WM_BITS>
struct BFP_bias {
    static constexpr int we = WE_BITS;                      
    static constexpr int wm = WM_BITS;                      
    static constexpr int bias_bfp = (1 << (WE_BITS - 1)) - 1;
    
    // Additional constants for operations
    static constexpr uint32_t MANT_MAX = (1u << (WM_BITS + 1)) - 1;
    static constexpr int INT_MIN_HLS = -10000;
    static constexpr int EXP_MAX = (1 << WE_BITS) - 1;

    typedef ap_uint<WE_BITS> exp_t;
    typedef ap_uint<1> sign_t;
    typedef ap_uint<WM_BITS+1> mant_t;
    typedef ap_uint<WM_BITS> delta_t;
};

//* Round to Nearest even (shift rigth)
static inline uint32_t helper_rne(uint32_t x, int shift) {
#pragma HLS INLINE
    
    if (shift <= 0) {
        int s = -shift;
        if (s >= 32) return 0u;
        return (s == 0) ? x : (x << s);
    }

    if (shift >= 32) return 0u;

    uint32_t q    = x >> shift;
    uint32_t rem  = x & ((1u << shift) - 1u);
    uint32_t half = 1u << (shift - 1);

    if (rem > half || (rem == half && (q & 1u))) {
        ++q;
    }
    return q;
}

//* Representation of BFP block with globlal exponent
template<class Cfg, std::size_t Block_size>
struct BFP_Global {
    typename Cfg::exp_t exp_shared;
    std::array<typename Cfg::sign_t, Block_size> sign;
    std::array<typename Cfg::mant_t, Block_size> mant;
    std::array<typename Cfg::delta_t, Block_size> delta;

    float rebuid_FP32(std::size_t i) const {
#pragma HLS INLINE
        if (i >= Block_size) return 0.0f;

        const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
        uint32_t exp_s = uint32_t(exp_shared);
        uint32_t mant_i = uint32_t(mant[i]);
        uint32_t delta_i = uint32_t(delta[i]);
        uint32_t sign_i = uint32_t(sign[i]);
        
        // Zero
        if (exp_s == 0 && mant_i == 0) {
            float zero = 0.0f;
            return sign_i ? -zero : zero;
        }
        
        // Special cases (only for WM >= 4)
        if (Cfg::wm >= 4 && delta_i == 0) {
            int exp_shared_unbiased = int(exp_s) - Cfg::bias_bfp;
            int exp_max_unbiased = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
            
            if (exp_shared_unbiased == exp_max_unbiased) {
                if (mant_i == (mant_max - 1)) {
                    // NaN
                    union {float f; uint32_t u;} nan_val;
                    nan_val.u = 0x7FC00000;
                    return nan_val.f;
                }
                if (mant_i == mant_max) {
                    // Infinity
                    union {float f; uint32_t u;} inf_val;
                    inf_val.u = sign_i ? 0xFF800000 : 0x7F800000;
                    return inf_val.f;
                }
            }
        }

        // Normal reconstruction
        int exp_shared_unbiased = int(exp_s) - Cfg::bias_bfp;
        int exp_real = exp_shared_unbiased - int(delta_i);
        
        uint64_t mant_unshifted = uint64_t(mant_i) << delta_i;
        float mant_val = float(mant_unshifted) / float(1ull << Cfg::wm);
        
        float value = std::ldexp(mant_val, exp_real);
        return sign_i ? -value : value;
    }
};

//* Block coding: FP32 array -> BFP_Global
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> encode_block(const std::array<float, Block_size>& xs) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> out{};

    //* Phase 1: find Maximum exponent
    int Emax = std::numeric_limits<int>::min();
    bool has_special = false;

    FIND_EMAX:
        for (std::size_t i = 0; i < Block_size; i++) {
    #pragma HLS PIPELINE II=1
    #pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
            
            float num_fp32 = xs[i];
            union {float f; uint32_t u;} u = {num_fp32};
            
            int exp_fp32 = int((u.u >> 23) & 0xFF);
            
            // NaN/Inf
            if (exp_fp32 == 0xFF) {
                has_special = true;
                continue;
            }
            
            if (exp_fp32 == 0) continue;
            
            int exp_unbiased = exp_fp32 - 127; 
            if (exp_unbiased > Emax) {
                Emax = exp_unbiased;
            }
        }

    if (Emax == std::numeric_limits<int>::min()) {
        if (has_special) {
            Emax = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
        } else {
            out.exp_shared = 0;
            out.sign.fill(0);
            out.mant.fill(0);
            out.delta.fill(0);
            return out;
        }
    }


    // Push Emax to the max for special cases
    if (has_special && Cfg::wm >= 4) {
        Emax = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
    }
        
    //* Phase 2: Calculate shared exponent for BFP with BIAS
    int exp_shared_bfp = Emax + Cfg::bias_bfp;
        
    if (exp_shared_bfp < 0) exp_shared_bfp = 0;
    if (exp_shared_bfp > (1 << Cfg::we) - 1) exp_shared_bfp = (1 << Cfg::we) - 1;
        
    out.exp_shared = (typename Cfg::exp_t)(exp_shared_bfp);

    //* Phase 3: Quantify each element
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;

QUANTIZE_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        float num_fp32 = xs[i];
        
        if (num_fp32 == 0.0f) {
            out.sign[i] = (typename Cfg::sign_t)0;
            out.mant[i] = (typename Cfg::mant_t)0;
            out.delta[i] = (typename Cfg::delta_t)0;
            continue;
        }
        
        union {float f; uint32_t u;} u = {num_fp32};
        
        uint32_t s = (u.u >> 31) & 0x1;
        int exp_fp32 = int((u.u >> 23) & 0xFF);
        uint32_t mant_fp32 = u.u & 0x7FFFFF;

        // NaN/Inf 
        if (exp_fp32 == 0xFF) {
            out.sign[i] = (typename Cfg::sign_t)s;
            
            if (mant_fp32 == 0) {
                // Infinity
                out.mant[i] = (typename Cfg::mant_t)mant_max;
                out.delta[i] = (typename Cfg::delta_t)0;
            } else {
                // NaN
                out.mant[i] = (typename Cfg::mant_t)(mant_max - 1);
                out.delta[i] = (typename Cfg::delta_t)0;
            }
            continue;
        }
        
        if (exp_fp32 == 0) {
            out.sign[i] = (typename Cfg::sign_t)0;
            out.mant[i] = (typename Cfg::mant_t)0;
            out.delta[i] = (typename Cfg::delta_t)0;
            continue;
        }
        
        // Build a 24-bit mantissa
        uint32_t mant24 = (u.u & 0x7FFFFF) | (1u << 23);
        int exp_unbiased = exp_fp32 - 127;

        int delta_val = Emax - exp_unbiased;
        int shift_total = (23 - Cfg::wm) + delta_val;
        
        uint32_t mant_reduced;
        
        if (shift_total >= 31) {
            mant_reduced = 0u;
        } else if (shift_total >= 0) {
            mant_reduced = helper_rne(mant24, shift_total);
        } else {
            mant_reduced = mant24 << (-shift_total);
        }
        
        if (mant_reduced > mant_max) {
            mant_reduced = mant_max;
        }
        
        out.sign[i] = (typename Cfg::sign_t)s;
        out.mant[i] = (typename Cfg::mant_t)mant_reduced;
        out.delta[i] = (typename Cfg::delta_t)delta_val;
    }
    
    return out;
}


//* Blcoj Decoding: BFP_Global -> FP32 ARRAY
template<class Cfg, std::size_t Block_size>
std::array<float, Block_size> decode_block(const BFP_Global<Cfg, Block_size>& blk) {
#pragma HLS INLINE off
    
    std::array<float, Block_size> result;
    
DECODE_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        result[i] = blk.rebuid_FP32(i);
    }
    
    return result;
}

#endif // BFP_H
