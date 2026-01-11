#include "bfp_kernel.h"

#ifndef MAX_TEST_BLOCKS
#define MAX_TEST_BLOCKS 128
#endif

#define FP32_DEPTH (N * MAX_TEST_BLOCKS)
#define BFP_DEPTH (BFP_BLOCK_SIZE * MAX_TEST_BLOCKS)

// Optimized burst parameters
#define FP32_BURST 16
#define BFP_BURST 256
#define OUTSTANDING 8

// Package BFP_Global in vector
void pack_bfp_block(const blk_t& blk, unsigned int* vec, unsigned int offset) {
#pragma HLS INLINE off
    
    // First element: shared exponent
    vec[offset] = blk.exp_shared;
    unsigned int idx = offset + 1;
    
PACK_ELEMENTS:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        vec[idx++] = blk.sign[i];
        vec[idx++] = blk.mant[i];
        vec[idx++] = blk.delta[i];
    }
}

// Unpack para bloque BFP
void unpack_bfp_block(const unsigned int* vec, blk_t& blk, unsigned int offset) {
#pragma HLS INLINE off
    
    // First element: shared exponent
    blk.exp_shared = vec[offset];
    unsigned int idx = offset + 1;
    
UNPACK_ELEMENTS:
    for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
        blk.sign[i]  = vec[idx++];
        blk.mant[i]  = vec[idx++];
        blk.delta[i] = vec[idx++];
    }
}


extern "C" {

void bfp_kernel(
    // Control
    const unsigned int operation,
    const unsigned int n_blocks,
    // Input FP32
    const float* in_fp32,
    // Input/Output BFP
    const unsigned int* in_bfp_a,     // Vector compacto A
    const unsigned int* in_bfp_b,     // Vector compacto B
    // Output fp32 for decode
    float* out_fp32,
    // Output BFP
    unsigned int* out_bfp

) {
   // FP32 I/O
    #pragma HLS INTERFACE m_axi port=in_fp32 offset=slave bundle=gmem0 \
        depth=FP32_DEPTH \
        max_read_burst_length=FP32_BURST \
        num_read_outstanding=OUTSTANDING
    
    #pragma HLS INTERFACE m_axi port=out_fp32 offset=slave bundle=gmem0 \
        depth=FP32_DEPTH \
        max_write_burst_length=FP32_BURST \
        num_write_outstanding=OUTSTANDING

    // BFP Input A 
    #pragma HLS INTERFACE m_axi port=in_bfp_a offset=slave bundle=gmem1 \
        depth=BFP_DEPTH \
        max_read_burst_length=BFP_BURST \
        num_read_outstanding=OUTSTANDING

    // BFP Input B
    #pragma HLS INTERFACE m_axi port=in_bfp_b offset=slave bundle=gmem1 \
        depth=BFP_DEPTH \
        max_read_burst_length=BFP_BURST \
        num_read_outstanding=OUTSTANDING

    // BFP Output 
    #pragma HLS INTERFACE m_axi port=out_bfp offset=slave bundle=gmem2 \
        depth=BFP_DEPTH \
        max_write_burst_length=BFP_BURST \
        num_write_outstanding=OUTSTANDING

    // Interface pragmas
    #pragma HLS INTERFACE s_axilite port=operation 
    #pragma HLS INTERFACE s_axilite port=n_blocks 
    #pragma HLS INTERFACE s_axilite port=return

    // Main processing loop - Simplified sequential design
    process_blocks: for (unsigned int blk_idx = 0; blk_idx < n_blocks; blk_idx++) {
//#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=1 max=128 avg=32
       
        const unsigned int fp32_offset = blk_idx * N;
        const unsigned int bfp_offset = blk_idx * BFP_BLOCK_SIZE;

        blk_t A{}, B{}, Z{};
        std::array<float, N> fp_in{}, fp_out{};
        
        //* Load data
        if (operation == OP_ENCODE) {
            // Load FP32 for encoding
        load_fp32: 
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
                fp_in[i] = in_fp32[fp32_offset + i];
            }
            
        } else if (operation == OP_DECODE) {

            // Load BFP A for decoding
            unpack_bfp_block(in_bfp_a, A, bfp_offset);
           
            
        } else if (operation == OP_RCP) {

            // Load only B for reciprocal
            unpack_bfp_block(in_bfp_b, B, bfp_offset);
            
        } else {
            // Binary operations: Load both A and B
            unpack_bfp_block(in_bfp_a, A, bfp_offset);
            unpack_bfp_block(in_bfp_b, B, bfp_offset);
        }
        
        //* Compute
        switch (operation) {
            case OP_ENCODE:
                Z = encode_block<Cfg, N>(fp_in);
                break;
                
            case OP_DECODE:
                fp_out = decode_block<Cfg, N>(A);
                break;
                
            case OP_ADD:
                Z = add_blocks<Cfg, N>(A, B);
                break;
                
            case OP_SUB:
                Z = sub_blocks<Cfg, N>(A, B);
                break;
                
            case OP_MUL:
                Z = mul_blocks<Cfg, N>(A, B);
                break;
                
            case OP_DIV:
                Z = div_blocks<Cfg, N>(A, B);
                break;
                
            case OP_RCP:
                Z = rcp_blocks<Cfg, N>(B);
                break;
                
            default:
                Z = A;
                break;
        }
        
        //* Store results
        if (operation == OP_DECODE) {
            // Write FP32 output
        store_fp32: 
            for (int i = 0; i < N; i++) {
#pragma HLS PIPELINE II=1
#pragma HLS LOOP_TRIPCOUNT min=8 max=64 avg=16
                out_fp32[fp32_offset + i] = fp_out[i];
            }
            
        } else {
            // Write BFP output
            pack_bfp_block(Z, out_bfp, bfp_offset);
        }
    }
}

} // extern "C"
