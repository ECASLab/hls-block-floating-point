# run_hls_2024.tcl - Minimum version for Vitis 2024.2 (U55C, top=bfp_kernel)

#==============================================================================
# PROJECT CONFIGURATION
#==============================================================================
open_project -reset bfp_proj
set_top bfp_kernel

#==============================================================================
# SOURCE FILES
#==============================================================================
add_files bfp_kernel.cpp -cflags "-std=c++11"
add_files bfp_kernel.h -cflags "-std=c++11"
add_files bfp_hls.h -cflags "-std=c++11"
add_files bfp_ops_hls.h -cflags "-std=c++11"

#==============================================================================
# Testbench for C simulation (use the TB that calls the kernel)
#==============================================================================
#add_files -tb tb_kernel2.cc -cflags "-std=c++11"
add_files -tb tb_bfp_kernel.cc -cflags "-std=c++11"


#==============================================================================
# SOLUTION CONFIGURATION
#==============================================================================
open_solution -reset "sol1"
set_part {xcu55c-fsvh2892-2L-e}
create_clock -period 5.0 -name default  ;# 200 MHz

#==============================================================================
# SETTINGS
#==============================================================================
config_compile   -name_max_length 80
config_dataflow  -strict_mode warning
config_rtl       -deadlock_detection sim
config_interface -m_axi_conservative_mode=1
config_interface -m_axi_addr64
config_interface -m_axi_auto_max_ports=0
config_export    -format xo -ipname bfp_kernel

#==============================================================================
# FLOW EXECUTION 
#==============================================================================
puts "\n=========================================="
puts "Starting C Simulation (csim)"
puts "==========================================\n"
csim_design -argv "--dataset mixed --nblocks 1"
# --dataset random → uniform, 4 decimal places, variable seed per run (different each time) --seed -1 / 42
# --dataset easy → “clean” vector (powers of 2, simple fractions, signs) designed for exact comparison
# --dataset special → underflow/overflow + Inf/NaN + large outliers
# --dataset nearzero→ values close to 0 with many decimal places
# --dataset mixed → mixture (default)

puts "\n=========================================="
puts "Starting C Synthesis (csynth)"
puts "==========================================\n"
csynth_design
#cosim_design

puts "\n=========================================="
puts "Exporting XO"
puts "==========================================\n"
export_design -format xo -rtl verilog -output bfp_kernel.xo

#==============================================================================
# FINAL REPORT
#==============================================================================
puts "\n=========================================="
puts "HLS Flow Complete!"
puts "=========================================="
puts "Project: bfp_proj"
puts "Reports: bfp_proj/sol1/syn/report/"
puts "XO     : bfp_proj/sol1/bfp_kernel.xo (y copia local: ./bfp_kernel.xo)"
puts "\n"

exit
