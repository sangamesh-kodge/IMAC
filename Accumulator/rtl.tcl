######################################################
# Script for Cadence RTL Compiler synthesis      
# Erik Brunvand, 2008
# Use with syn-rtl -f rtl-script
# Replace items inside <> with your own information
######################################################

# Set the search paths to the libraries and the HDL files
# Remember that "." means your current directory. Add more directories
# after the . if you like. 
# set_attribute hdl_search_path {./} 
set_attribute lib_search_path /home/nano01/a/agrawa64/mosis_65nm/mosis_65nm/IP_LIB/TSMCHOME/digital/Front_End/timing_power_noise/ECSM/tcbn65lp_200a
set_attribute library {tcbn65lpbc_ecsm.lib}

set_attribute interconnect_mode ple
set_attribute lib_search_path /home/nano01/a/agrawa64/mosis_65nm/mosis_65nm/IP_LIB/TSMCHOME/digital/Back_End/lef/tcbn65lp_200a/lef
set_attribute lef_library {tcbn65lp_6lmT1.lef}
# set_attribute information_level 6 

# set myFiles [list <HDL-files>]   ;# All your HDL files
# set basename <top-module-name>   ;# name of top level module
# set myClk <clk>                  ;# clock name
# set myPeriod_ps <num>            ;# Clock period in ps
# set myInDelay_ns <num>           ;# delay from clock to inputs valid
# set myOutDelay_ns <num>          ;# delay from clock to output valid
# set runname <string>             ;# name appended to output files

#*********************************************************
#*   below here shouldn't need to be changed...          *
#*********************************************************
set rda_Input(ui_pwrnet) {VDD}
set rda_Input(ui_gndnet) {VSS}
# Analyze and Elaborate the HDL files
read_hdl accumulator.v 
elaborate accumulator

# Apply Constraints and generate clocks
# set clock [define_clock -period ${myPeriod_ps} -name ${myClk} [clock_ports]]	
# external_delay -input $myInDelay_ns -clock ${myClk} [find / -port ports_in/*]
# external_delay -output $myOutDelay_ns -clock ${myClk} [find / -port ports_out/*]

# Sets transition to default values for Synopsys SDC format, 
# fall/rise 400ps
# dc::set_clock_transition .4 $myClk

# check that the design is OK so far
# check_design -unresolved
# report timing -lint

# Synthesize the design to the target library
synthesize -to_mapped


report area > area_report.rpt
report power > power_report.rpt
report timing > timing_report.rpt
report clocks > clocks_report.rpt
# Write out the structural Verilog and sdc files
write -mapped >  accumulator_synth.v
write_sdc > constraints.sdc

write_script >  script
gui_show;
