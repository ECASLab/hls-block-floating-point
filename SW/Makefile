# Makefile for BFP Accelerator Host Application
# Based on ECASLab SW Makefile structure

# Source files and executables
SOURCES := bfp_host.cpp
EXECUTABLES := bfp_host

# Compiler flags - matching ECASLab configuration
CFLAGS := -I/opt/xilinx/xrt/include -Wall -std=c++17 -O0 -g -I.
LDFLAGS := -L/opt/xilinx/xrt/lib -pthread -lxrt_core -lxrt_coreutil

.PHONY: all clean

# Default target to build all executables
all: $(EXECUTABLES)

# Rule to build each executable from its corresponding source file
$(EXECUTABLES): %: %.cpp
	$(CXX) -o $@ $< $(CFLAGS) $(LDFLAGS)

# Clean target to remove all executables
clean:
	$(RM) $(EXECUTABLES)
