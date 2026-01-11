
#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <iomanip>
#include <random>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <algorithm>
#include <cstdlib>

#include "bfp_kernel.h"  

//------------------------------------------------------------------------------
// Utilidades numéricas y métricas
//------------------------------------------------------------------------------
static inline bool is_nan_f(float x) {
    return std::isnan(static_cast<double>(x));
}
static inline bool is_inf_f(float x) {
    return std::isinf(static_cast<double>(x));
}

struct ErrorMetrics {
    double mae  = 0.0;
    double rmse = 0.0;
    double mape = 0.0;
    std::size_t count = 0;
    std::size_t skipped = 0;
};

static ErrorMetrics compute_metrics(const std::vector<float>& ref,
                                   const std::vector<float>& got,
                                   double eps = 1e-12) {
    ErrorMetrics m;
    const std::size_t n = std::min(ref.size(), got.size());
    double sum_abs = 0.0;
    double sum_sq  = 0.0;
    double sum_mape = 0.0;

    for (std::size_t i = 0; i < n; ++i) {
        float r = ref[i];
        float g = got[i];

        if (is_nan_f(r) || is_nan_f(g)) {
            m.skipped++;
            continue;
        }
        if (is_inf_f(r) || is_inf_f(g)) {
            if (is_inf_f(r) && is_inf_f(g) && (std::signbit(r) == std::signbit(g))) {
                m.count++;
                continue;
            }
            sum_abs += 1e6;
            sum_sq  += 1e12;
            sum_mape += 100.0;
            m.count++;
            continue;
        }

        double err = std::fabs(double(g) - double(r));
        sum_abs += err;
        sum_sq  += err * err;

        double denom = std::max(std::fabs(double(r)), eps);
        sum_mape += (err / denom) * 100.0;
        m.count++;
    }

    if (m.count > 0) {
        m.mae  = sum_abs / double(m.count);
        m.rmse = std::sqrt(sum_sq / double(m.count));
        m.mape = sum_mape / double(m.count);
    }
    return m;
}

struct RangeCfg {
    const char* name;
    int we;
    int wm;
    float lo;
    float hi;
};

static RangeCfg get_range_cfg(int we, int wm) {
    static const RangeCfg table[] = {
        {"Float4  (WE=2, WM=1)",  2,  1,   -4.0f,     4.0f},
        {"Float6  (WE=3, WM=2)",  3,  2,  -10.0f,    10.0f},
        {"Float8  (WE=4, WM=3)",  4,  3, -20.0f,   20.0f},
        {"Float8  (WE=5, WM=2)",  5,  2,-1024.0f,  1024.0f},
        {"Float10 (WE=5, WM=4)",  5,  4,-1024.0f,  1024.0f},
        {"Float12 (WE=5, WM=6)",  5,  6,-1024.0f,  1024.0f},
        {"Float16 (WE=5, WM=10)", 5, 10,-1024.0f,  1024.0f},
    };

    for (const auto& e : table) {
        if (e.we == we && e.wm == wm) return e;
    }

    RangeCfg fb{"Custom", we, wm, -1024.0f, 1024.0f};
    return fb;
}

enum class Dataset {
    RANDOM,
    EASY,
    SPECIAL,
    NEARZERO,
    MIXED
};

static Dataset parse_dataset(const std::string& s) {
    if (s == "random")  return Dataset::RANDOM;
    if (s == "easy")    return Dataset::EASY;
    if (s == "special") return Dataset::SPECIAL;
    if (s == "nearzero")return Dataset::NEARZERO;
    return Dataset::MIXED;
}

static const char* dataset_name(Dataset d) {
    switch (d) {
        case Dataset::RANDOM:   return "random";
        case Dataset::EASY:     return "easy";
        case Dataset::SPECIAL:  return "special";
        case Dataset::NEARZERO: return "nearzero";
        default:                return "mixed";
    }
}

static float round_decimals(float x, int decimals) {
    const float p = std::pow(10.0f, float(decimals));
    return std::round(x * p) / p;
}

static std::array<float, N> make_block_random(std::mt19937& rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    std::array<float, N> a{};
    for (int i = 0; i < N; ++i) {
        a[i] = round_decimals(dist(rng), 4);
    }
    return a;
}

static std::array<float, N> make_block_easy(int which) {
    static const float patA[] = {
      512.0f,  496.0f,  480.0f,  464.0f,
      448.0f,  432.0f,  416.0f,  400.0f,
      384.0f,  368.0f,  352.0f,  336.0f,
      320.0f,  304.0f,  288.0f,  272.0f,

     -512.0f, -496.0f, -480.0f, -464.0f,
      -448.0f, -432.0f, -416.0f, -400.0f,
      -384.0f, -368.0f, -352.0f, -336.0f,
      -320.0f, -304.0f, -288.0f, -272.0f
    };
    static const float patB[] = {
     1024.0f,  992.0f,  960.0f,  928.0f,
      896.0f,  864.0f,  832.0f,  800.0f,
      768.0f,  736.0f,  704.0f,  672.0f,
      640.0f,  608.0f,  576.0f,  544.0f,

     -1024.0f, -992.0f, -960.0f, -928.0f,
      -896.0f, -864.0f, -832.0f, -800.0f,
      -768.0f, -736.0f, -704.0f, -672.0f,
      -640.0f, -608.0f, -576.0f, -544.0f
    };

    std::array<float, N> out{};
    const float* pat = (which == 0) ? patA : patB;
    constexpr int P = 16;
    for (int i = 0; i < N; ++i) out[i] = pat[i % P];
    return out;
}

static std::array<float, N> make_block_nearzero(std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1e-3f, 1e-3f);
    std::array<float, N> a{};
    for (int i = 0; i < N; ++i) {
        float v = dist(rng);
        a[i] = round_decimals(v, 7);
    }
    if (N >= 4) {
        a[0] = 0.0f;
        a[1] = -0.0f;
    }
    return a;
}

static std::array<float, N> make_block_special(float lo, float hi, int which) {
    std::array<float, N> a{};
    const float big = (which == 0) ? hi * 0.95f : hi * 0.80f;
    const float small = (which == 0) ? lo * 0.95f : lo * 0.80f;

    for (int i = 0; i < N; ++i) a[i] = 0.0f;

    if (N >= 8) {
        a[0] = big;
        a[1] = -big;
        a[2] = small;
        a[3] = -small;
        a[4] = hi * 1.5f;  
        a[5] = lo * 1.5f;
        a[6] = 1e-30f;     
        a[7] = -1e-30f;
    }

    union { uint32_t u; float f; } infp, infn, nanv;
    infp.u = 0x7F800000u;
    infn.u = 0xFF800000u;
    nanv.u = 0x7FC00000u;

    if (N >= 12) {
        a[8]  = infp.f;
        a[9]  = infn.f;
        a[10] = nanv.f;
        a[11] = 1.0f;
    }

    for (int i = 12; i < N; ++i) {
        if ((i % 2) == 0) a[i] = big;
        else a[i] = 1e-6f;
    }

    return a;
}

static void make_blocks(Dataset ds, std::mt19937& rng, float lo, float hi,
                        std::array<float, N>& A,
                        std::array<float, N>& B) {
    switch (ds) {
        case Dataset::RANDOM:
            A = make_block_random(rng, lo, hi);
            B = make_block_random(rng, lo, hi);
            return;
        case Dataset::EASY:
            A = make_block_easy(0);
            B = make_block_easy(1);
            return;
        case Dataset::SPECIAL:
            A = make_block_special(lo, hi, 0);
            B = make_block_special(lo, hi, 1);
            return;
        case Dataset::NEARZERO:
            A = make_block_nearzero(rng);
            B = make_block_nearzero(rng);
            return;
        default:
            // MIXED
            A = make_block_random(rng, lo, hi);
            B = make_block_random(rng, lo, hi);
            if (N >= 8) {
                A[0] = 0.0f;
                A[1] = 1.0f;
                A[2] = -1.0f;
                A[3] = 2.0f;
                B[0] = 0.0f;
                B[1] = 0.5f;
                B[2] = -0.5f;
                B[3] = 4.0f;
            }
            return;
    }
}

static std::string cfg_tag() {
    std::ostringstream oss;
    oss << "WE" << WE << "_WM" << WM << "_N" << N;
    return oss.str();
}

static void write_csv_header(std::ofstream& f, const std::string& title) {
    f << "# " << title << "\n";
    f << "# cfg=" << cfg_tag() << "\n";
    f << "idx,ref_fp32,got_fp32,abs_err,rel_err_pct,exp_shared,sign,mant,delta,is_inf,is_nan\n";
}

static inline double err_pct(double ref, double got, double eps=1e-12) {
    if (std::isnan(ref) || std::isnan(got)) return std::numeric_limits<double>::quiet_NaN();
    if (std::isinf(ref) || std::isinf(got)) {
        if (std::isinf(ref) && std::isinf(got) && (std::signbit(ref) == std::signbit(got))) return 0.0;
        return 100.0;
    }
    const double e = std::fabs(got - ref);
    const double d = std::max(std::fabs(ref), eps);
    return (e / d) * 100.0;
}

static void print_raw_u32(const char* title,
                          const unsigned int* buf,
                          unsigned int offset_words,
                          unsigned int size_words)
{
    std::cout << "\n+- " << title
              << " - Raw Memory (offset=" << offset_words
              << ", size=" << size_words << " uint32_t) -+\n";

    const unsigned int PER_LINE = 8; // cambia a 6, 10, etc.

    for (unsigned int i = 0; i < size_words; ++i) {
        if (i % PER_LINE == 0) std::cout << "  ";

        const unsigned int w = buf[offset_words + i];
        std::cout << "[" << std::setw(2) << i << "] 0x"
                  << std::hex << std::setw(8) << std::setfill('0') << w
                  << std::dec << std::setfill(' ');

        if (i + 1 < size_words) std::cout << " ";

        if ((i % PER_LINE) == (PER_LINE - 1)) std::cout << "\n";
    }
    std::cout << "\n\n";
}

static void print_table_encode(const char* title,
                               const std::array<float, N>& X,
                               const blk_t& blk)
{
    std::cout << "+- Result " << title << " - Raw Memory (offset=0, size="
              << BFP_BLOCK_SIZE << " uint32_t) -+\n";
    std::cout << "i  X           BFP Decode     FP32 Ref       Err (%)\n";
    std::cout << "    -----------------------------------------------------------------------\n";

    for (int i = 0; i < N; ++i) {
        const double ref = (double)X[i];
        const double got = (double)blk.rebuid_FP32(i);
        const double ep  = err_pct(ref, got);

        std::cout << std::setw(2) << i << "  "
                  << std::setw(11) << std::fixed << std::setprecision(4) << ref << "  "
                  << std::setw(11) << std::fixed << std::setprecision(4) << got << "  "
                  << std::setw(11) << std::fixed << std::setprecision(4) << ref << "  "
                  << std::setw(10) << std::fixed << std::setprecision(4) << ep
                  << "\n";
    }
    std::cout << "\n";
}

static void print_table_op(const char* title,
                           const std::array<float, N>& A,
                           const std::array<float, N>& B,
                           const std::vector<float>& Z_ref_block,
                           const blk_t& blkZ)
{
    std::cout << "+- Result " << title << " - Raw Memory (offset=0, size="
              << BFP_BLOCK_SIZE << " uint32_t) -+\n";
    std::cout << "i       A           B           BFP Result      FP32 Ref        Err (%)\n";
    std::cout << "    -----------------------------------------------------------------------\n";

    for (int i = 0; i < N; ++i) {
        const double a   = (double)A[i];
        const double b   = (double)B[i];
        const double ref = (double)Z_ref_block[i];
        const double got = (double)blkZ.rebuid_FP32(i);
        const double ep  = err_pct(ref, got);

        std::cout << std::setw(2) << i << "  "
                  << std::setw(11) << std::fixed << std::setprecision(4) << a   << "  "
                  << std::setw(11) << std::fixed << std::setprecision(4) << b   << "  "
                  << std::setw(13) << std::fixed << std::setprecision(4) << got << "  "
                  << std::setw(13) << std::fixed << std::setprecision(4) << ref << "  "
                  << std::setw(10) << std::fixed << std::setprecision(4) << ep
                  << "\n";
    }
    std::cout << "\n";
}

static void write_csv_row(std::ofstream& f, int idx, float ref, float got,
                          const blk_t& blk, int elem_idx) {
    const double abs_err = (is_nan_f(ref) || is_nan_f(got)) ? std::numeric_limits<double>::quiet_NaN()
                                                            : std::fabs(double(got) - double(ref));
    const double denom = std::max(std::fabs(double(ref)), 1e-12);
    const double rel = (is_nan_f(ref) || is_nan_f(got)) ? std::numeric_limits<double>::quiet_NaN()
                                                        : (abs_err / denom) * 100.0;

    const uint32_t mant = uint32_t(blk.mant[elem_idx]);
    const uint32_t del  = uint32_t(blk.delta[elem_idx]);
    const uint32_t exp  = uint32_t(blk.exp_shared);

    const bool inf = (Cfg::wm >= 4) ? is_infinity<Cfg>(mant, del, exp) : false;
    const bool nan = (Cfg::wm >= 4) ? is_nan<Cfg>(mant, del, exp)      : false;

    f << idx << ","
      << std::setprecision(9) << ref << ","
      << std::setprecision(9) << got << ","
      << std::setprecision(12) << abs_err << ","
      << std::setprecision(8) << rel << ","
      << exp << ","
      << uint32_t(blk.sign[elem_idx]) << ","
      << mant << ","
      << del << ","
      << (inf ? 1 : 0) << ","
      << (nan ? 1 : 0) << "\n";
}

struct DeltaValidation {
    std::size_t checked = 0;
    std::size_t mismatch = 0;
    std::size_t overflow_expected = 0; 
};

static int fp32_unbiased_exp(float x, bool& is_special, bool& is_zero_or_subnormal) {
    union { float f; uint32_t u; } u { x };
    const int exp_fp32 = int((u.u >> 23) & 0xFF);
    is_special = (exp_fp32 == 0xFF);
    is_zero_or_subnormal = (exp_fp32 == 0);
    if (is_special || is_zero_or_subnormal) return 0;
    return exp_fp32 - 127;
}

static DeltaValidation validate_deltas_encode(const std::array<float, N>& xs, const blk_t& blk) {
    DeltaValidation v;

    int Emax = std::numeric_limits<int>::min();
    bool has_special = false;

    for (int i = 0; i < N; ++i) {
        bool sp=false, z=false;
        const int e = fp32_unbiased_exp(xs[i], sp, z);
        if (sp) { has_special = true; continue; }
        if (z)  continue;
        Emax = std::max(Emax, e);
    }

    if (Emax == std::numeric_limits<int>::min()) {
        return v;
    }

    const uint32_t delta_mod = (1u << WM);

    for (int i = 0; i < N; ++i) {
        bool sp=false, z=false;
        const int e = fp32_unbiased_exp(xs[i], sp, z);
        if (sp || z) continue;

        const int expected = Emax - e;
        if (expected >= int(delta_mod)) v.overflow_expected++;

        const uint32_t stored = uint32_t(blk.delta[i]);
        const uint32_t expected_trunc = uint32_t(expected) & (delta_mod - 1u);

        v.checked++;
        if (stored != expected_trunc) v.mismatch++;
    }

    return v;
}

static DeltaValidation validate_deltas_postop(const blk_t& blk) {
    DeltaValidation v;
    for (int i = 0; i < N; ++i) {
        const uint32_t mant = uint32_t(blk.mant[i]);
        const uint32_t del  = uint32_t(blk.delta[i]);
        const uint32_t exp  = uint32_t(blk.exp_shared);

        if (Cfg::wm >= 4) {
            if (is_nan<Cfg>(mant, del, exp) || is_infinity<Cfg>(mant, del, exp)) {
                continue;
            }
        }
        if (mant == 0u) continue;

        const uint32_t expected = calculate_delta_from_mant<Cfg>(mant);
        v.checked++;
        if (del != expected) v.mismatch++;
    }
    return v;
}

static void run_encode(const std::vector<float>& in_fp32,
                       std::vector<unsigned int>& out_bfp,
                       unsigned int n_blocks) {
    std::vector<unsigned int> dummy_in(BFP_BLOCK_SIZE * n_blocks, 0u);
    std::vector<float> dummy_out_fp32(N * n_blocks, 0.0f);

    bfp_kernel(OP_ENCODE, n_blocks,
               in_fp32.data(),
               dummy_in.data(), dummy_in.data(),
               dummy_out_fp32.data(),
               out_bfp.data());
}

static void run_op(unsigned int opcode,
                   const std::vector<unsigned int>& in_a,
                   const std::vector<unsigned int>& in_b,
                   std::vector<unsigned int>& out_z,
                   unsigned int n_blocks) {
    std::vector<float> dummy_in_fp32(N * n_blocks, 0.0f);
    std::vector<float> dummy_out_fp32(N * n_blocks, 0.0f);

    bfp_kernel(opcode, n_blocks,
               dummy_in_fp32.data(),
               in_a.data(), in_b.data(),
               dummy_out_fp32.data(),
               out_z.data());
}

int main(int argc, char** argv) {
    // Defaults
    Dataset ds = Dataset::MIXED;
    unsigned int n_blocks = 1;
    long long seed_arg = -1; 

    bool print_tables = false;   
    unsigned int print_block_limit = 1; 

    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--dataset" && i + 1 < argc) {
            ds = parse_dataset(argv[++i]);
        } else if (a == "--nblocks" && i + 1 < argc) {
            n_blocks = (unsigned int)std::max(1, std::atoi(argv[++i]));
        } else if (a == "--seed" && i + 1 < argc) {
            seed_arg = std::atoll(argv[++i]);
        } else if (a == "--print") {
            print_tables = true;
        } else if (a == "--print_blocks" && i + 1 < argc) {
            print_block_limit = (unsigned int)std::max(1, std::atoi(argv[++i]));
        }

    }

    static constexpr unsigned int TB_MAX_BLOCKS = 128;
    if (n_blocks > TB_MAX_BLOCKS) n_blocks = TB_MAX_BLOCKS;

    const RangeCfg rc = get_range_cfg(WE, WM);
    const float lo = rc.lo;
    const float hi = rc.hi;

    std::uint64_t seed;
    if (seed_arg >= 0) {
        seed = (std::uint64_t)seed_arg;
    } else {
        seed = (std::uint64_t)std::random_device{}() ^
               (std::uint64_t)std::chrono::high_resolution_clock::now().time_since_epoch().count();
    }
    std::mt19937 rng((uint32_t)(seed & 0xFFFFFFFFu));

    std::cout << "\n=== TB BFP Kernel ===\n";
    std::cout << "Config: WE=" << WE << ", WM=" << WM << ", N=" << N << "\n";
    std::cout << "Rango de prueba: [" << lo << ", " << hi << "] (" << rc.name << ")\n";
    std::cout << "Dataset: " << dataset_name(ds) << " | n_blocks=" << n_blocks << " | seed=" << seed << "\n\n";

    std::vector<float> A_fp32(N * n_blocks), B_fp32(N * n_blocks);

    for (unsigned int b = 0; b < n_blocks; ++b) {
        std::array<float, N> Ab{}, Bb{};
        make_blocks(ds, rng, lo, hi, Ab, Bb);
        for (int i = 0; i < N; ++i) {
            A_fp32[b * N + i] = Ab[i];
            B_fp32[b * N + i] = Bb[i];
        }
    }

    // 1) ENCODE A y B
    std::vector<unsigned int> A_bfp(BFP_BLOCK_SIZE * n_blocks, 0u);
    std::vector<unsigned int> B_bfp(BFP_BLOCK_SIZE * n_blocks, 0u);

    run_encode(A_fp32, A_bfp, n_blocks);
    run_encode(B_fp32, B_bfp, n_blocks);

    std::vector<float> A_dec(N * n_blocks), B_dec(N * n_blocks);

    const std::string tag = cfg_tag();
    const std::string baseA = std::string("encode_A_") + tag + "_" + dataset_name(ds) + ".csv";
    const std::string baseB = std::string("encode_B_") + tag + "_" + dataset_name(ds) + ".csv";

    std::ofstream fA(baseA);
    std::ofstream fB(baseB);
    write_csv_header(fA, "ENCODE A: ref vs decode(BFP)");
    write_csv_header(fB, "ENCODE B: ref vs decode(BFP)");

    DeltaValidation dvA_total{}, dvB_total{};

    for (unsigned int b = 0; b < n_blocks; ++b) {
        blk_t blkA{}, blkB{};
        unpack_bfp_block(A_bfp.data(), blkA, b * BFP_BLOCK_SIZE);
        unpack_bfp_block(B_bfp.data(), blkB, b * BFP_BLOCK_SIZE);

        std::array<float, N> Ab{}, Bb{};
        for (int i = 0; i < N; ++i) {
            Ab[i] = A_fp32[b * N + i];
            Bb[i] = B_fp32[b * N + i];
        }

        if (print_tables && b < print_block_limit) {
            const unsigned int offA = b * BFP_BLOCK_SIZE;
            const unsigned int offB = b * BFP_BLOCK_SIZE;

            print_raw_u32("ENCODE A", A_bfp.data(), offA, BFP_BLOCK_SIZE);
            print_table_encode("ENCODE A (FP32 -> BFP -> decode)", Ab, blkA);

            print_raw_u32("ENCODE B", B_bfp.data(), offB, BFP_BLOCK_SIZE);
            print_table_encode("ENCODE B (FP32 -> BFP -> decode)", Bb, blkB);
        }

        const DeltaValidation dvA = validate_deltas_encode(Ab, blkA);
        const DeltaValidation dvB = validate_deltas_encode(Bb, blkB);

        dvA_total.checked += dvA.checked;
        dvA_total.mismatch += dvA.mismatch;
        dvA_total.overflow_expected += dvA.overflow_expected;

        dvB_total.checked += dvB.checked;
        dvB_total.mismatch += dvB.mismatch;
        dvB_total.overflow_expected += dvB.overflow_expected;

        for (int i = 0; i < N; ++i) {
            const float a = A_fp32[b * N + i];
            const float bval = B_fp32[b * N + i];
            const float ad = blkA.rebuid_FP32(i);
            const float bd = blkB.rebuid_FP32(i);
            A_dec[b * N + i] = ad;
            B_dec[b * N + i] = bd;

            const int global_idx = int(b * N + i);
            write_csv_row(fA, global_idx, a, ad, blkA, i);
            write_csv_row(fB, global_idx, bval, bd, blkB, i);
        }
    }

    fA.close();
    fB.close();

    const ErrorMetrics mA = compute_metrics(A_fp32, A_dec);
    const ErrorMetrics mB = compute_metrics(B_fp32, B_dec);

    std::cout << "[ENCODE] A: MAE=" << mA.mae << " | RMSE=" << mA.rmse << " | MAPE=" << mA.mape
              << " | n=" << mA.count << " | skipped=" << mA.skipped << "\n";
    std::cout << "         Delta check: checked=" << dvA_total.checked
              << " | mismatch=" << dvA_total.mismatch
              << " | expected_overflow(>=2^WM)=" << dvA_total.overflow_expected << "\n";

    std::cout << "[ENCODE] B: MAE=" << mB.mae << " | RMSE=" << mB.rmse << " | MAPE=" << mB.mape
              << " | n=" << mB.count << " | skipped=" << mB.skipped << "\n";
    std::cout << "         Delta check: checked=" << dvB_total.checked
              << " | mismatch=" << dvB_total.mismatch
              << " | expected_overflow(>=2^WM)=" << dvB_total.overflow_expected << "\n\n";

    struct OpInfo { unsigned int opcode; const char* name; };
    const OpInfo ops[] = {
        {OP_ADD, "add"},
        {OP_SUB, "sub"},
        {OP_MUL, "mul"},
        {OP_DIV, "div"}
    };

    for (const auto& op : ops) {
    std::vector<unsigned int> Z_bfp(BFP_BLOCK_SIZE * n_blocks, 0u);
    run_op(op.opcode, A_bfp, B_bfp, Z_bfp, n_blocks);

    std::vector<float> Z_dec(N * n_blocks);
    std::vector<float> Z_ref(N * n_blocks);

    const std::string csv_name =
        std::string("op_") + op.name + "_" + tag + "_" + dataset_name(ds) + ".csv";
    std::ofstream fo(csv_name);
    write_csv_header(fo, std::string("OP ") + op.name + ": ref vs decode(BFP)");

    DeltaValidation dv_total{};

    for (unsigned int b = 0; b < n_blocks; ++b) {
        blk_t blkZ{};
        unpack_bfp_block(Z_bfp.data(), blkZ, b * BFP_BLOCK_SIZE);

        const DeltaValidation dv = validate_deltas_postop(blkZ);
        dv_total.checked += dv.checked;
        dv_total.mismatch += dv.mismatch;

        std::array<float, N> Ab{}, Bb{};
        for (int i = 0; i < N; ++i) {
            Ab[i] = A_fp32[b * N + i];
            Bb[i] = B_fp32[b * N + i];
        }

        std::vector<float> Zref_block(N, 0.0f);

        for (int i = 0; i < N; ++i) {
            const float a = A_fp32[b * N + i];
            const float bb = B_fp32[b * N + i];

            float r = 0.0f;
            switch (op.opcode) {
                case OP_ADD: r = a + bb; break;
                case OP_SUB: r = a - bb; break;
                case OP_MUL: r = a * bb; break;
                case OP_DIV: r = a / bb; break;
                default:     r = 0.0f; break;
            }

            const float g = blkZ.rebuid_FP32(i);

            const int global_idx = int(b * N + i);
            Z_ref[global_idx] = r;
            Z_dec[global_idx] = g;
            Zref_block[i]     = r;

            write_csv_row(fo, global_idx, r, g, blkZ, i);
        }

        if (print_tables && b < print_block_limit) {
            const unsigned int offZ = b * BFP_BLOCK_SIZE;

            std::string pretty;
            if      (op.opcode == OP_ADD) pretty = "ADDITION (A + B)";
            else if (op.opcode == OP_SUB) pretty = "SUBTRACTION (A - B)";
            else if (op.opcode == OP_MUL) pretty = "MULTIPLICATION (A * B)";
            else if (op.opcode == OP_DIV) pretty = "DIVISION (A / B)";
            else                          pretty = std::string(op.name);

            print_raw_u32(pretty.c_str(), Z_bfp.data(), offZ, BFP_BLOCK_SIZE);

            print_table_op(pretty.c_str(), Ab, Bb, Zref_block, blkZ);
        }
    }

        fo.close();

        const ErrorMetrics mm = compute_metrics(Z_ref, Z_dec);

        std::cout << "[" << op.name << "] MAE=" << mm.mae 
                  << " | RMSE=" << mm.rmse 
                  << " | MAPE=" << mm.mape
                  << " | n=" << mm.count 
                  << " | skipped=" << mm.skipped << "\n";
                  
        std::cout << "       Delta post-op check: checked=" << dv_total.checked
                  << " | mismatch=" << dv_total.mismatch << "\n";
    }

    std::cout << "\nCSVs generados:\n";
    std::cout << "  - " << baseA << "\n";
    std::cout << "  - " << baseB << "\n";
    std::cout << "  - op_add_... , op_sub_... , op_mul_... , op_div_...\n";
    std::cout << "\nSiguiente paso: corre plot_bfp_tb.py para generar PNGs.\n";

    return 0;
}
