#ifndef BFP_OPS_H
#define BFP_OPS_H

#include <ap_int.h>
#include <cstdint>
#include <cstdlib>
#include "bfp_hls.h"

//* Helper:  Clamp exponent to valid range
template<class Cfg>
static inline uint32_t clamp_exponent(int E_real) {
#pragma HLS INLINE
    int E_biased = E_real + Cfg::bias_bfp;
    if (E_biased < 0) E_biased = 0;
    if (E_biased > (1 << Cfg::we) - 1) E_biased = (1 << Cfg::we) - 1;
    return uint32_t(E_biased);
}

//* Helper: Calculate delta from mantissa
template<class Cfg>
static inline uint32_t calculate_delta_from_mant(uint32_t mant) {
#pragma HLS INLINE
    
    if (mant == 0) return 0;
    
    int msb_pos = -1;
    
    #pragma HLS UNROLL factor=8
    for (int b = Cfg::wm; b >= 0; --b) {
        if ((mant >> b) & 0x1) {
            msb_pos = b;
            break;
        }
    }
    
    if (msb_pos < 0) return 0;
    
    int delta = Cfg::wm - msb_pos;
    
    return uint32_t(delta);
}

//* Helper: 64-bit RNE
static inline uint64_t helper_rne_64(uint64_t x, int shift) {
#pragma HLS INLINE
    
    if (shift <= 0) {
        int s = -shift;
        if (s >= 64) return 0ull;
        return (s == 0) ? x : (x << s);
    }

    if (shift >= 64) return 0ull;

    uint64_t q = x >> shift;
    uint64_t rem = x & ((1ull << shift) - 1);
    uint64_t half = 1ull << (shift - 1);

    if (rem > half || (rem == half && (q & 1))) {
        ++q;
    }
    return q;
}

//* Helper: Detect special cases
template<class Cfg>
static inline bool is_infinity(uint32_t mant, uint32_t delta, uint32_t exp_shared) {
#pragma HLS INLINE
    if (Cfg::wm < 4) return false; 
    
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
    const uint32_t exp_max = (1u << Cfg::we) - 1;
    
    int exp_shared_unbiased = int(exp_shared) - Cfg::bias_bfp;
    int exp_max_unbiased = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
    
    return (mant == mant_max) && (delta == 0) && (exp_shared_unbiased == exp_max_unbiased);
}

template<class Cfg>
static inline bool is_nan(uint32_t mant, uint32_t delta, uint32_t exp_shared) {
#pragma HLS INLINE
    if (Cfg::wm < 4) return false; // Not supported for wm < 4
    
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
    const uint32_t exp_max = (1u << Cfg::we) - 1;
    
    int exp_shared_unbiased = int(exp_shared) - Cfg::bias_bfp;
    int exp_max_unbiased = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
    
    return (mant == (mant_max - 1)) && (delta == 0) && (exp_shared_unbiased == exp_max_unbiased);
}


//* Block adder BFP: Z = A + B
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> add_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Z{};
    
    int Ea_shared = int(A.exp_shared) - Cfg::bias_bfp;
    int Eb_shared = int(B.exp_shared) - Cfg::bias_bfp;
    
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
    
    //* Step 1: Finding the maximum exponent considering deltas
    int Emax = std::numeric_limits<int>::min();
    bool has_nonzero = false;

FIND_EMAX:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        uint32_t mant_a = uint32_t(A.mant[i]);
        uint32_t mant_b = uint32_t(B.mant[i]);
        
        // Skip special values for Emax calculation
        if (Cfg::wm >= 4) {
            bool is_inf_a = is_infinity<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared));
            bool is_inf_b = is_infinity<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared));
            bool is_nan_a = is_nan<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared));
            bool is_nan_b = is_nan<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared));
            
            if (is_inf_a || is_inf_b || is_nan_a || is_nan_b) {
                continue;
            }
        }
        
        if (mant_a != 0u) {
            int exp_A_i = Ea_shared - int(A.delta[i]);
            if (exp_A_i > Emax) Emax = exp_A_i;
            has_nonzero = true;
        }
        
        if (mant_b != 0u) {
            int exp_B_i = Eb_shared - int(B.delta[i]);
            if (exp_B_i > Emax) Emax = exp_B_i;
            has_nonzero = true;
        }
    }
    
    if (!has_nonzero) {
        // all zeros, check for special values
        bool has_special = false;
        
        for (std::size_t i = 0; i < Block_size; ++i) {
            if (Cfg::wm >= 4) {
                uint32_t mant_a = uint32_t(A.mant[i]);
                uint32_t mant_b = uint32_t(B.mant[i]);
                
                if (is_infinity<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared)) ||
                    is_infinity<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared)) ||
                    is_nan<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared)) ||
                    is_nan<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared))) {
                    has_special = true;
                    break;
                }
            }
        }
        
        if (has_special) {
            Emax = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
        } else {
            Z.exp_shared = 0;
            Z.sign.fill(0);
            Z.mant.fill(0);
            Z.delta.fill(0);
            return Z;
        }
    }
    
    Z.exp_shared = clamp_exponent<Cfg>(Emax);

    //* Step 2: Sum with alignment by element
    std::array<uint64_t, Block_size> M_temp;
    std::array<uint32_t, Block_size> sign_temp;
    bool overflow_flag = false;
    
ADD_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16

        uint32_t mant_a = uint32_t(A.mant[i]);
        uint32_t mant_b = uint32_t(B.mant[i]);
        uint32_t sign_a = uint32_t(A.sign[i]);
        uint32_t sign_b = uint32_t(B.sign[i]);
        
        // Special cases
        if (Cfg::wm >= 4) {
            bool is_inf_A = is_infinity<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared));
            bool is_inf_B = is_infinity<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared));
            bool is_nan_A = is_nan<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared));
            bool is_nan_B = is_nan<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared));
            
            if (is_nan_A || is_nan_B) {
                sign_temp[i] = 0u;
                M_temp[i] = mant_max - 1;
                continue;
            }
            
            if (is_inf_A && is_inf_B) {
                if (sign_a != sign_b) {
                    sign_temp[i] = 0u;
                    M_temp[i] = mant_max - 1; 
                } else {
                    sign_temp[i] = sign_a;
                    M_temp[i] = mant_max;    
                }
                continue;
            }
            
            if (is_inf_A) {
                sign_temp[i] = sign_a;
                M_temp[i] = mant_max;
                continue;
            }
            if (is_inf_B) {
                sign_temp[i] = sign_b;
                M_temp[i] = mant_max;
                continue;
            }
        }
        
        // Normal sum
        int delta_A_i = int(A.delta[i]);
        int delta_B_i = int(B.delta[i]);
        
        uint64_t Ma_full = uint64_t(mant_a) << delta_A_i;
        uint64_t Mb_full = uint64_t(mant_b) << delta_B_i;

        int exp_A_i = Ea_shared - delta_A_i;
        int exp_B_i = Eb_shared - delta_B_i;
        
        int shift_A = Emax - exp_A_i;
        int shift_B = Emax - exp_B_i;
        
        uint64_t Ma = (shift_A >= 64) ? 0 : (shift_A > 0) ? helper_rne_64(Ma_full, shift_A) : (shift_A < 0) ? (Ma_full << (-shift_A)) : Ma_full;
        uint64_t Mb = (shift_B >= 64) ? 0 : (shift_B > 0) ? helper_rne_64(Mb_full, shift_B) : (shift_B < 0) ? (Mb_full << (-shift_B)) : Mb_full;
        
        // Signed sum
        int64_t Sa = sign_a ? -int64_t(Ma) : int64_t(Ma);
        int64_t Sb = sign_b ? -int64_t(Mb) : int64_t(Mb);
        int64_t S  = Sa + Sb;
        
        uint32_t sign_res = (S < 0) ? 1u : 0u;
        uint64_t Mag = uint64_t((S < 0) ? -S : S);
        
        if (Mag == 0u) {
            sign_res = 0u;
        }
        
        sign_temp[i] = sign_res;
        M_temp[i] = Mag;
        
        if (M_temp[i] > mant_max) {
            overflow_flag = true;
        }
    }
    
    //* Step 3: Normalize if there is overflow
    if (overflow_flag) {
        Emax += 1;
        Z.exp_shared = clamp_exponent<Cfg>(Emax);
        
NORMALIZE_OVERFLOW:
        for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
            
            uint64_t M_adj = helper_rne_64(M_temp[i], 1);
            if (M_adj > mant_max) {
                M_adj = mant_max;
            }
            M_temp[i] = M_adj;
            if (M_temp[i] == 0u) {
                sign_temp[i] = 0u;
            }
        }
    } else {
SATURATE_MANT:
        for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
            
            if (M_temp[i] > mant_max) {
                M_temp[i] = mant_max;
            }
        }
    }
    
    //* Step 4: Save results and calculate deltas
SAVE_RESULTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        Z.sign[i] = sign_temp[i];
        Z.mant[i] = uint32_t(M_temp[i]);
        
        if (Cfg::wm >= 4) {
            bool is_special = (M_temp[i] == mant_max) || (M_temp[i] == mant_max - 1);
            Z.delta[i] = is_special ? 0 : calculate_delta_from_mant<Cfg>(uint32_t(M_temp[i]));
        } else {
            Z.delta[i] = calculate_delta_from_mant<Cfg>(uint32_t(M_temp[i]));
        }
    }
    
    return Z;
}

//* Block substraction BFP: Z = A - B
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> sub_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> B_neg = B;
    
NEGATE_SIGNS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        if (B.mant[i] != 0u) {
            B_neg.sign[i] = uint32_t(B.sign[i]) ^ 1u;
        } else {
            B_neg.sign[i] = 0u; 
        }
    }
    
    return add_blocks<Cfg, Block_size>(A, B_neg);
}

//* Block multiplication BFP: Z = A * B
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> mul_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> Z{};
    
    int Ea_shared = int(A.exp_shared) - Cfg::bias_bfp;
    int Eb_shared = int(B.exp_shared) - Cfg::bias_bfp;
    
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1;
    
    //* Step 1: Find the maximum exponent and calculate the products
    int Emax = std::numeric_limits<int>::min();
    std::array<uint64_t, Block_size> products;
    std::array<int, Block_size> exp_products;
    std::array<uint32_t, Block_size> signs;
    std::array<bool, Block_size> is_special;
    bool has_nonzero = false;
    
COMPUTE_PRODUCTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        uint32_t mant_a = uint32_t(A.mant[i]);
        uint32_t mant_b = uint32_t(B.mant[i]);
        uint32_t sign = uint32_t(A.sign[i]) ^ uint32_t(B.sign[i]);
        signs[i] = sign;
        is_special[i] = false;
        
        // Special cases
        if (Cfg::wm >= 4) {
            bool is_nan_A = is_nan<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared));
            bool is_nan_B = is_nan<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared));
            bool is_inf_A = is_infinity<Cfg>(mant_a, uint32_t(A.delta[i]), uint32_t(A.exp_shared));
            bool is_inf_B = is_infinity<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared));
            bool is_zero_A = (mant_a == 0u);
            bool is_zero_B = (mant_b == 0u);
            
            if (is_nan_A || is_nan_B) {
                signs[i] = 0;
                products[i] = mant_max - 1;
                is_special[i] = true;
                continue;
            }
            
            if ((is_zero_A && is_inf_B) || (is_inf_A && is_zero_B)) {
                signs[i] = 0;
                products[i] = mant_max - 1; 
                is_special[i] = true;
                continue;
            }
            
            if (is_inf_A || is_inf_B) {
                products[i] = mant_max;
                is_special[i] = true;
                continue;
            }
        }
        
        if (mant_a == 0u || mant_b == 0u) {
            signs[i] = 0;
            products[i] = 0;
            exp_products[i] = 0;
            continue;
        }
        
        // Normal multiplication
        int delta_A_i = int(A.delta[i]);
        int delta_B_i = int(B.delta[i]);
        
        uint32_t Ma_full = mant_a << delta_A_i;
        uint32_t Mb_full = mant_b << delta_B_i;
        
        uint64_t P = uint64_t(Ma_full) * uint64_t(Mb_full);
        products[i] = P;
        
        int exp_A_i = Ea_shared - delta_A_i;
        int exp_B_i = Eb_shared - delta_B_i;
        int exp_prod = exp_A_i + exp_B_i;
        exp_products[i] = exp_prod;
        
        if (exp_prod > Emax) {
            Emax = exp_prod;
        }
        has_nonzero = true;
    }
    
    if (!has_nonzero) {
        // Special values verification
        bool has_special_vals = false;
        for (std::size_t i = 0; i < Block_size; ++i) {
            if (is_special[i]) {
                has_special_vals = true;
                break;
            }
        }
        
        if (has_special_vals) {
            Emax = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
        } else {
            Z.exp_shared = 0;
            Z.sign.fill(0);
            Z.mant.fill(0);
            Z.delta.fill(0);
            return Z;
        }
    }
    
    Z.exp_shared = clamp_exponent<Cfg>(Emax);

    //* Step 2: Quantify products to the shared exponent
QUANTIZE_PRODUCTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        if (is_special[i]) {
            Z.sign[i] = signs[i];
            Z.mant[i] = uint32_t(products[i]);
            Z.delta[i] = 0;
            continue;
        }
        
        if (products[i] == 0u) {
            Z.sign[i] = 0;
            Z.mant[i] = 0;
            Z.delta[i] = 0;
            continue;
        }
        
        uint64_t P = products[i];
        int exp_prod = exp_products[i];
        
        // Reduce product to WM bits
        uint64_t q = P >> Cfg::wm;
        uint64_t rem = P & ((1ull << Cfg::wm) - 1);
        uint64_t half = 1ull << (Cfg::wm - 1);
        
        bool tie = (rem == half);
        bool gt = (rem > half);
        bool lsb_odd = (q & 1u) != 0;
        
        if (gt || (tie && lsb_odd)) {
            ++q;
        }
        
        // Align the shared exponent
        int shift = Emax - exp_prod;
        uint64_t M_shifted;
        
        if (shift >= 64) {
            M_shifted = 0;
        } else if (shift > 0) {
            M_shifted = helper_rne_64(q, shift);
        } else if (shift < 0) {
            M_shifted = q << (-shift);
        } else {
            M_shifted = q;
        }
        
        // Normalize if exceeds max_maintenance (with fixed limit)
        for (int j = 0; j < (int)Cfg::wm + 1; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=24 avg=4
            if (M_shifted <= mant_max) break;
            M_shifted = helper_rne_64(M_shifted, 1);
        }
        
        uint32_t M = uint32_t(M_shifted);
        
        if (M == 0u) {
            signs[i] = 0u;
        }
        
        Z.sign[i] = signs[i];
        Z.mant[i] = M;
        Z.delta[i] = calculate_delta_from_mant<Cfg>(M);
    }
    
    return Z;
}

//* Block reciprocal BFP: R = 1/B
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> rcp_blocks(
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    BFP_Global<Cfg, Block_size> R{};
    
    const int Eb_shared = int(B.exp_shared) - Cfg::bias_bfp;
    const uint32_t mant_max = (1u << (Cfg::wm + 1)) - 1u;
    
    std::array<uint64_t, Block_size> q{};
    std::array<int, Block_size> Ei{};
    std::array<bool, Block_size> is_special{};
    bool any_nz = false;
    
RCP_ELEMENTS:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        uint32_t mant_b = uint32_t(B.mant[i]);
        is_special[i] = false;
        
        if (mant_b == 0u) {
            R.sign[i] = B.sign[i];
            q[i] = mant_max;
            Ei[i] = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
            is_special[i] = true;
            continue;
        }
        
        if (Cfg::wm >= 4) {
            if (is_nan<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared))) {
                R.sign[i] = 0;
                q[i] = mant_max - 1;
                Ei[i] = 0;
                is_special[i] = true;
                continue;
            }
            
            if (is_infinity<Cfg>(mant_b, uint32_t(B.delta[i]), uint32_t(B.exp_shared))) {
                R.sign[i] = B.sign[i];
                q[i] = 0;
                Ei[i] = std::numeric_limits<int>::min();
                any_nz = true;
                continue;
            }
        }
        
        R.sign[i] = B.sign[i];

        int delta_B_i = int(B.delta[i]);
        uint64_t Mb_full = (uint64_t)mant_b << delta_B_i;
        
        const uint64_t Num = 1ull << (2 * Cfg::wm);
        const uint64_t Den = Mb_full;
        
        uint64_t qq  = Num / Den;
        uint64_t rem = Num % Den;
        
        const bool gt = (rem << 1) > Den;
        const bool tie = (rem << 1) == Den;
        const bool lsb_odd = (qq & 1ull) != 0ull;
        
        if (gt || (tie && lsb_odd)) {
            ++qq;
        }
        
        int exp_B_i = Eb_shared - int(B.delta[i]);
        int Erec = -exp_B_i;

        const int exp_max_real = (1 << Cfg::we) - 1 - Cfg::bias_bfp;
        const int exp_min_real = -(Cfg::bias_bfp);
        
        if (Erec > exp_max_real) {
            if (Cfg::wm >= 4) {
                q[i] = mant_max - 2;
            } else {
                q[i] = mant_max;
            }
            Ei[i] = exp_max_real;
            any_nz = true;
            continue;
        }
        
        if (Erec < exp_min_real) {
            q[i] = 0;
            Ei[i] = 0;
            R.sign[i] = 0u;
            is_special[i] = true;
            continue;
        }
        
        // Normalize if exceeds mant_max
        for (int j = 0; j < (int)Cfg::wm + 1; ++j) {
#pragma HLS LOOP_TRIPCOUNT min=0 max=16 avg=4
            if (qq <= mant_max) break;
            qq = helper_rne_64(qq, 1);
            ++Erec;
        }
        
        if (qq > mant_max) qq = mant_max;
        
        q[i] = qq;
        Ei[i] = Erec;
        any_nz = true;
    }
    
    if (!any_nz) {
        R.exp_shared = clamp_exponent<Cfg>(0);
        R.sign.fill(0u);
        R.mant.fill(0u);
        R.delta.fill(0u);
        return R;
    }
    
    // Step 2: Find maximum common factor
    int Eshared = std::numeric_limits<int>::min();
    
FIND_MAX_EXP:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        
        if (is_special[i] && q[i] == mant_max) continue;
        
        if (Ei[i] > Eshared) {
            Eshared = Ei[i];
        }
    }
    
    R.exp_shared = clamp_exponent<Cfg>(Eshared);
    
    //* Step 3: Align and calculate deltas
ALIGN_AND_CALC_DELTA:
    for (std::size_t i = 0; i < Block_size; ++i) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
         
        uint64_t M = q[i];
        
        if (is_special[i]) {
            R.mant[i] = uint32_t(M);
            R.delta[i] = 0u;
        } else {
            int diff = Eshared - Ei[i];
            
            if (diff > 0) {
                M = helper_rne_64(M, diff);
            } else if (diff < 0) {
                int shift_left = -diff;
                if (shift_left < 64) {
                    M = M << shift_left;
                }
            }
            
            if (M > mant_max) {
                M = mant_max;
            }
            
            if (M == 0u) {
                R.sign[i] = 0u;
            }
            
            R.mant[i] = uint32_t(M);
            R.delta[i] = calculate_delta_from_mant<Cfg>(uint32_t(M));
        }
    }
 
    return R;
}

//* Block division BFP: Z = A / B
template<class Cfg, std::size_t Block_size>
BFP_Global<Cfg, Block_size> div_blocks(
    const BFP_Global<Cfg, Block_size>& A,
    const BFP_Global<Cfg, Block_size>& B
) {
#pragma HLS INLINE off
    
    auto R = rcp_blocks<Cfg, Block_size>(B);
    return mul_blocks<Cfg, Block_size>(A, R);
}

#endif // BFP_OPS_H
