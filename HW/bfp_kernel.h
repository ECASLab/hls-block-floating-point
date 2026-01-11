#ifndef BFP_KERNEL_H
#define BFP_KERNEL_H

#include <ap_int.h>
#include "bfp_hls.h"
#include "bfp_ops_hls.h"

// Configuration float16
#ifndef WE
#define WE 5
#endif

#ifndef WM
#define WM 10
#endif

#ifndef N
#define N 32
#endif

// Global 
using Cfg = BFP_bias<WE, WM>;
using blk_t = BFP_Global<Cfg, N>;

// Operation codes
typedef enum : unsigned int {
    OP_ENCODE = 0,
    OP_DECODE = 1,
    OP_ADD    = 2,
    OP_SUB    = 3,
    OP_MUL    = 4,
    OP_DIV    = 5,
    OP_RCP    = 6
} bfp_op_t;

// Constants
static constexpr unsigned int BFP_BLOCK_SIZE = 1 + 3 * N;

// Declaraciones de funciones auxiliares
void pack_bfp_block(const blk_t& blk, unsigned int* vec, unsigned int offset);
void unpack_bfp_block(const unsigned int* vec, blk_t& blk, unsigned int offset);

// Declaración del kernel principal
extern "C" void bfp_kernel(
    const unsigned int operation,
    const unsigned int n_blocks,
    const float* in_fp32,
    const unsigned int* in_bfp_a,
    const unsigned int* in_bfp_b,
    float* out_fp32,
    unsigned int* out_bfp
);

#endif // BFP_KERNEL_H
