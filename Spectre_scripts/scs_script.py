#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 23:36:48 2019

@author: skodge
"""
import numpy as np
import random 
dorandom= False
weights = 15
inputs = 15

F= open ("mc_mac.scs",'w')
# Initialize the file
F.write('''// Generated for: spectre
// Generated on: Jul 25 11:54:53 2019
// Design library name: Sangamesh_SRAM
// Design cell name: 6T_MAC
// Design view name: schematic

// terminal command:   spectre trans_mult.scs +escchars +log psf/spectre.out -format psfxl -raw psf +lqtimeout 900 -maxw 5 -maxn 5 +fastdc +mt=64 -64 +lqt=1 +aps=liberal
 
simulator lang=spectre\n\n\n
''')


#copy body of the circuit
body= open ("Circuit_body.scs",'r')
F.write(body.read())
body.close()
F.write("\n\n")

# Give the required input signals
#power rials
F.write ("""//SK control signals
//Power rails
Vvdd   (vdd   0)   vsource dc=1 
Vvdd_pre   (vdd_pre   0)   vsource dc=1.2  
Vground   (gnd  0)   vsource dc=0 """)

##precharge
F.write("\n// Precharge Signal\n")
delay = [1.35, 1.3, 1.2, 1.00]
period = 1.0
width= [0.25, 0.3, 0.4, 0.6]
vh=1.2
for j in range (4):
    F.write ("Vvpre{} (Vpre{} 0) vsource type=pwl wave=[0 0".format(j,j))  
    for i in range (25):
        x=delay[j]+i*period
        F.write (" {}n 0 {}n {} {}n {} {}n 0 ".format(round(x,3),round(x+0.001,3),vh,round(x+width[j],3),vh,round(x+width[j]+0.001,3)))
    F.write("]\n")
  
#wordline signals
delay = 1.00
period = 1
width= 0.4
#vwl="vin"
F.write ("\n// Worldline Signal\n")
for j in range (25):
    if dorandom:
        vin=random.randint(0,15)
    else:
        vin=inputs
    x = delay+ j*period
    F.write ("Vwl{} (WL{} 0) vsource  type=pwl wave=[0 0 {}n  0 {}n {} {}n {} {}n 0 ]\n".format(j+1,j+1,round(x,3),round(x+0.001,3),(0.3+(0.7*vin/15)),round(x+width,3),(0.3+(0.7*vin/15)),round(x+width+0.001,3)))
    


#en_sharing
F.write("\n// en_share Signal\n")
delay = 1.4
period = 1.0
width= 0.2
vh=1.2
F.write ("Ven_sharing (en_sharing 0) vsource  type=pwl wave=[0 0",)  
for i in range (25):
    x=delay+i*period
    F.write (" {}n {} {}n {} {}n {} {}n {}".format(round(x,3),0,round(x+0.001,3),vh,round(x+width,3),vh,round(x+width+0.001,3),0))
F.write("]\n")



#en_sample
F.write("\n// en_sample Signal\n")
delay = 1.4
period = 1.0
width= 0.2
vh=1.2
F.write ("Ven_sample (en_sample 0) vsource  type=pwl wave=[0 0",)  
for i in range (25):
    x=delay+i*period
    F.write (" {}n {} {}n {} {}n {} {}n {}".format(round(x,3),0,round(x+0.001,3),vh,round(x+width,3),vh,round(x+width+0.001,3),0))
F.write("]\n")

#en_acc
F.write("\n// en_acc Signal\n")
delay = 1.6
period = 1.0
width= 0.4
vh=1.2
F.write ("Venacc (enacc 0) vsource  type=pwl wave=[0 0",)  
for i in range (25):
    x=delay+i*period
    F.write (" {}n {} {}n {} {}n {} {}n {}".format(round(x,3),0,round(x+0.001,3),vh,round(x+width,3),vh,round(x+width+0.001,3),0))
F.write("]\n")
    
#en_ADC
F.write("\n// en_sample Signal\n")
F.write("""Vvss_adc (vss_adc 0) vsource dc=0.020  
Vvdd_adc (vdd_adc 0) vsource dc=0.630
Vadc_en (ADC_en 0)  vsource type=pwl wave=[0 0 26.0n 0 26.001n 1 31n 1 31.001n 0]
Vreset_shiftreg (Reset_ShiftReg 0) vsource type=pwl wave=[0 0 26.0n 0 26.001n 1 31.0n 1 31.001n 0]
Vreset (Reset 0) vsource type=pwl wave=[0 0 26.0n 0 26.001n 1 31.0n 1 31.001n 0]\n
""")
delay = 26
period = 1.0
width= 0.5
v1=1
v2=0
name="clk_ShiftReg"
F.write ("V{} ({} 0) vsource  type=pwl wave=[0 {}".format(name,name,v1))  
for i in range (5):
    x=delay+i*period
    F.write (" {}n {} {}n {} {}n {} {}n {}".format(round(x,3),v1,round(x+0.001,3),v2,round(x+width,3),v2,round(x+width+0.001,3),v1))
F.write("]\n")
        
delay = 26.75
period = 1.0
width1= 0.25
width2= 0.15
gap=0.35
v1=0
v2=1
name="clk_logic"
F.write ("V{} ({} 0) vsource  type=pwl wave=[0 {}".format(name,name,v1))  
for i in range (4):
    x1=delay+i*period
    x2=delay+i*period+gap
    F.write (" {}n {} {}n {} {}n {} {}n {} {}n {} {}n {} {}n {} {}n {}".format(round(x1,3),v1,round(x1+0.001,3),v2,round(x1+width1,3),v2,round(x1+width1+0.001,3),v1,round(x2,3),v1,round(x2+0.001,3),v2,round(x2+width2,3),v2,round(x2+width2+0.001,3),v1))
F.write("]\n")
    
    
delay = 26
period = 1.0
width= 0.5
v1=1
v2=0
name="clk_SA"
F.write ("V{} ({} 0) vsource  type=pwl wave=[0 {}".format(name,name,v1))  
for i in range (5):
    x=delay+i*period
    F.write (" {}n {} {}n {} {}n {} {}n {}".format(round(x,3),v1,round(x+0.001,3),v2,round(x+width,3),v2,round(x+width+0.001,3),v1))
F.write("]\n")

delay = 27.15
period = 1.0
width= 0.25
v1=0
v2=1
name="Output_valid"
F.write ("V{} ({} 0) vsource  type=pwl wave=[0 {}".format(name,name,v1))  
for i in range (4):
    x=delay+i*period
    F.write (" {}n {} {}n {} {}n {} {}n {}".format(round(x,3),v1,round(x+0.001,3),v2,round(x+width,3),v2,round(x+width+0.001,3),v1))
F.write("]\n")


#initial condition Spectre
F.write ("ic d3=1 d2=1 d1=1 d0=1 D3=0 D2=0 D1=0 D0=0 Vx=450m sample=0 accumulation=0 " )
for i in range (25):
    b=np.zeros(4) 
    if dorandom :
        a=bin(random.randint(0,15))[2:]
    else:
        a=bin(weights)[2:]
        
    c=len(a)
    for k in range(c):
        if(a[k]=='0'):
            b[k+4-c]=0
        else:
            b[k+4-c]=1             
    bb=1-b

    for j in range (4):
        F.write ("I_{x}_{y}.qb={zb} I_{x}_{y}.q={z} ".format(x=i+1,y=j+1,z=b[j],zb=bb[j]))
        
    

#simulation
F.write("""
simulatorOptions options reltol=1e-3 vabstol=1e-6 iabstol=1e-12 temp=27 \
    tnom=27 scalem=1.0 scale=1.0 gmin=1e-12 rforce=1 maxnotes=5 maxwarns=5 \
    digits=5 cols=80 pivrel=1e-3 sensfile="../psf/sens.output" \
    checklimitdest=psf 
mc montecarlo saveprocessparams=yes variations=all donominal=no savefamilyplots=yes appendsd=yes scalarfile="analog_multData.txt" numruns=1000 {     
    tran tran start=0 stop=257n 
    export Ouput_mac = oceanEval(\"int(value(getData(\\\"d0\\\" ?result \\\"tran\\\") 30.75e-09 ?xName \\\"time\\\")/0.9) +int(value(getData(\\\"d1\\\" ?result \\\"tran\\\") 30.75e-09 ?xName \\\"time\\\")/0.9)*2 +int(value(getData(\\\"d2\\\" ?result \\\"tran\\\") 30.75e-09 ?xName \\\"time\\\")/0.9)*4 +int(value(getData(\\\"d3\\\" ?result \\\"tran\\\") 30.75e-09 ?xName \\\"time\\\")/0.9)*8 \") 
}
//tran tran stop=30n write="spectre.ic" writefinal="spectre.fc" \
    annotate=status maxiters=5 
    
finalTimeOP info what=oppoint where=rawfile
modelParameter info what=models where=rawfile
element info what=inst where=rawfile
outputParameter info what=output where=rawfile
designParamVals info what=parameters where=rawfile
primitives info what=primitives where=rawfile
subckts info what=subckts where=rawfile
saveOptions options save=allpub pwr=all currents=all
""")






F.close()


