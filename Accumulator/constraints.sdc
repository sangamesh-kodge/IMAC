# ####################################################################

#  Created by Encounter(R) RTL Compiler RC14.28 - v14.20-s067_1 on Sun Jul 14 16:44:16 -0400 2019

# ####################################################################

set sdc_version 1.7

set_units -capacitance 1000.0fF
set_units -time 1000.0ps

# Set the current design
current_design accumulator

set_clock_gating_check -setup 0.0 
set_wire_load_selection_group "WireAreaForZero" -library "tcbn65lpbc_ecsm"
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/BHD]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/BUFFD20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/BUFFD24]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/BUFTD20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/BUFTD24]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKBD20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKBD24]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKLHQD20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKLHQD24]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKLNQD20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKLNQD24]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKND20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/CKND24]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GAN2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GAN2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GAOI21D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GAOI21D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GAOI22D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GBUFFD1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GBUFFD2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GBUFFD3]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GBUFFD4]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GBUFFD8]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDCAP]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDCAP10]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDCAP2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDCAP3]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDCAP4]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDFCNQD1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GDFQD1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GFILL]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GFILL10]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GFILL2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GFILL3]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GFILL4]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GINVD1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GINVD2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GINVD3]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GINVD4]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GINVD8]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GMUX2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GMUX2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GMUX2ND1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GMUX2ND2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GND2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GND2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GND2D3]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GND2D4]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GND3D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GND3D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GNR2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GNR2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GNR3D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GNR3D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GOAI21D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GOAI21D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GOR2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GOR2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GSDFCNQD1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GTIEH]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GTIEL]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GXNR2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GXNR2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GXOR2D1]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/GXOR2D2]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/INVD20]
set_dont_use [get_lib_cells tcbn65lpbc_ecsm/INVD24]
