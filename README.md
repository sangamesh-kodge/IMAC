# 6T-SRAM-Multiplication
Application Level:

    train.py  : Training the parameters of the network on Cifar10.
    test.py   : Inference with error addition as per the hardware variations.
    vgg.py    : The architecture of the network describing each layer and functions for quantisation and error addition.

Circuit Simulation:

    Circuit_Simulation_Results.py   : Code that plots the graph for Vin vs Output, W vs Output, Expected vs Digital output and the effect of variation on output.
    MC.png                          : Effect of variation on Output
    OutvsVin.png                    : Effect of Vin sweep on Output
    OutvsW.png                      : Effect of W sweep on Output
    ExpvsObs.png                    : Expected output vs Observed Output
    
