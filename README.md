# 6T-SRAM-Multiplication
Application Level:

    train.py            : Training the parameters of the network on Cifar10.
    test.py             : Inference with error addition as per the hardware variations.
    vgg.py              : The architecture of the network describing each layer and functions for 
                          quantisation and error addition.
    error_accuracy.pt   : 1000 inferences with Variation as observed in circuit simulation

Circuit Simulation:

    Circuit_Simulation_Results.py   : Code that plots the graph for Vin vs Output, W vs Output, 
                                      Expected vs Digital output and the effect of variation on output.
    MC.png                          : Effect of variation on Output
    OutvsVin.png                    : Effect of Vin sweep on Output
    OutvsW.png                      : Effect of W sweep on Output
    ExpvsObs.png                    : Expected output vs Observed Output
    
Trained Parameters: https://drive.google.com/file/d/1Z_9ZXXWG4P6jbfc6K7xdWg4bPQDYRfGY/view?usp=sharing
