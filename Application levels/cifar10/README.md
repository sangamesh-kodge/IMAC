# In-Memory Multi-Bit Multiplication in 6 Transistor SRAM Cells
Application Level:

    train.py            : Training the parameters of the network on Cifar10.
    test.py             : Inference with error addition as per the hardware variations.
    vgg.py              : The architecture of the network describing each layer and functions for 
                          quantisation and error addition. Refer the paper for detail description.
    error_accuracy.pt   : Test accuracy for 1000 inferences with Variation as observed in circuit simulation.
                          Test accuracy  remains  between  88.5%-89.5%  (mean= 88.99% and sigma= 0.1496) with 
                          variation

    
Trained Parameters: https://drive.google.com/file/d/1yPOBUNTf6hSGDnY10Me16RkXG8tDf1gS/view?usp=sharing
