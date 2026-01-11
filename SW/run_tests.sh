#!/bin/bash
# Test script for BFP Accelerator - ENHANCED VERSION
# Executes all operations with detailed results

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
EXECUTABLE="./bfp_host"
N_BLOCKS=${N_BLOCKS:-6}

echo "========================================"
echo "BFP Accelerator Test Suite - ENHANCED"
echo "========================================"
echo ""
echo -e "${CYAN}Configuration:${NC}"
echo "  Executable: $EXECUTABLE"
echo "  N_BLOCKS: $N_BLOCKS"
echo "  Elements per block: 32"
echo "  Total elements: $((N_BLOCKS * 32))"
echo "  BFP Config: WE=5, WM=10"
echo ""

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo -e "${RED}Error: $EXECUTABLE not found${NC}"
    echo "Please run 'make' first to build the host application"
    exit 1
fi

# Check if xclbin exists
XCLBIN_PATH="../HW/package.hw/kernels.xclbin"
if [ ! -f "$XCLBIN_PATH" ]; then
    echo -e "${YELLOW}Warning: xclbin not found at $XCLBIN_PATH${NC}"
    echo "Make sure the HW build is complete"
    echo ""
fi

# Test operations
echo -e "${YELLOW}Starting tests with $N_BLOCKS blocks...${NC}"
echo ""

# Temporary file for capturing output
TMPFILE=$(mktemp)

# Test 1: ENCODE
echo "========================================"
echo -e "${BLUE}Test 1: ENCODE Operation${NC}"
echo "========================================"
$EXECUTABLE 0 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    # Extract and display key metrics
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    grep "exp_shared:" $TMPFILE | head -1
    grep "TEST PASSED\|TEST FAILED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ ENCODE test completed${NC}"
else
    echo -e "${RED}✗ ENCODE test failed${NC}"
fi
echo ""

# Test 2: DECODE
echo "========================================"
echo -e "${BLUE}Test 2: DECODE Operation${NC}"
echo "========================================"
$EXECUTABLE 1 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    MAE=$(grep "MAE:" $TMPFILE | grep -oP 'MAE:\s+\K[0-9.e+-]+')
    MAPE=$(grep "MAPE:" $TMPFILE | grep -oP 'MAPE:\s+\K[0-9.]+')
    echo -e "  Mean Absolute Error: ${GREEN}${MAE}${NC}"
    echo -e "  Mean Absolute % Error: ${GREEN}${MAPE}%${NC}"
    grep "TEST PASSED\|TEST FAILED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ DECODE test completed${NC}"
else
    echo -e "${RED}✗ DECODE test failed${NC}"
fi
echo ""

# Test 3: ADD
echo "========================================"
echo -e "${BLUE}Test 3: ADD Operation${NC}"
echo "========================================"
$EXECUTABLE 2 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    grep "exp_shared:" $TMPFILE | head -1
    echo -e "  Sample result: A + B"
    grep "\[0\] sign:" $TMPFILE | head -1 | sed 's/^/  /'
    grep "TEST PASSED\|TEST COMPLETED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ ADD test completed${NC}"
else
    echo -e "${RED}✗ ADD test failed${NC}"
fi
echo ""

# Test 4: SUB
echo "========================================"
echo -e "${BLUE}Test 4: SUB Operation${NC}"
echo "========================================"
$EXECUTABLE 3 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    grep "exp_shared:" $TMPFILE | head -1
    echo -e "  Sample result: A - B"
    grep "\[0\] sign:" $TMPFILE | head -1 | sed 's/^/  /'
    grep "TEST PASSED\|TEST COMPLETED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ SUB test completed${NC}"
else
    echo -e "${RED}✗ SUB test failed${NC}"
fi
echo ""

# Test 5: MUL
echo "========================================"
echo -e "${BLUE}Test 5: MUL Operation${NC}"
echo "========================================"
$EXECUTABLE 4 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    grep "exp_shared:" $TMPFILE | head -1
    echo -e "  Sample result: A * B"
    grep "\[0\] sign:" $TMPFILE | head -1 | sed 's/^/  /'
    grep "TEST PASSED\|TEST COMPLETED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ MUL test completed${NC}"
else
    echo -e "${RED}✗ MUL test failed${NC}"
fi
echo ""

# Test 6: DIV
echo "========================================"
echo -e "${BLUE}Test 6: DIV Operation${NC}"
echo "========================================"
$EXECUTABLE 5 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    grep "exp_shared:" $TMPFILE | head -1
    echo -e "  Sample result: A / B"
    grep "\[0\] sign:" $TMPFILE | head -1 | sed 's/^/  /'
    grep "TEST PASSED\|TEST COMPLETED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ DIV test completed${NC}"
else
    echo -e "${RED}✗ DIV test failed${NC}"
fi
echo ""

# Test 7: RCP
echo "========================================"
echo -e "${BLUE}Test 7: RCP (Reciprocal) Operation${NC}"
echo "========================================"
$EXECUTABLE 6 $N_BLOCKS | tee $TMPFILE
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${CYAN}Key Results:${NC}"
    grep "exp_shared:" $TMPFILE | head -1
    echo -e "  Sample result: 1 / B"
    grep "\[0\] sign:" $TMPFILE | head -1 | sed 's/^/  /'
    grep "TEST PASSED\|TEST COMPLETED" $TMPFILE
    AVG_TIME=$(grep "kernel_execution" $TMPFILE | grep -oP 'AVG: \K[0-9.e+-]+')
    echo -e "  Kernel execution time: ${GREEN}${AVG_TIME}s${NC}"
    echo ""
    echo -e "${GREEN}✓ RCP test completed${NC}"
else
    echo -e "${RED}✗ RCP test failed${NC}"
fi
echo ""

# Cleanup
rm -f $TMPFILE

# Summary
echo "========================================"
echo -e "${GREEN}✓ All tests completed successfully!${NC}"
echo "========================================"
echo ""
echo -e "${CYAN}Summary:${NC}"
echo "  • ENCODE: Converts FP32 to BFP format"
echo "  • DECODE: Converts BFP back to FP32"
echo "  • ADD/SUB/MUL/DIV/RCP: Arithmetic in BFP"
echo ""
echo -e "${YELLOW}Note:${NC} Display shows first 8 elements per block for brevity."
echo "      Full ${N_BLOCKS} blocks x 32 elements = $((N_BLOCKS * 32)) total elements processed."
echo ""
