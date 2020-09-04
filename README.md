# IMAC (Accepted in TCAS)

This repository contains the codes used for the work done in the paper "In-Memory Multi-Bit Multiplication in 6 Transistor SRAM Cells" (https://ieeexplore.ieee.org/document/9050543). 

@author: Mustafa Ali, Akhilesh Jaiswal, Sangamesh Kodge, Amogh Agrawal, Indranil Chakraborty and Kaushik Roy
@copyright: Nanoelectronics Research Laboratory, Purdue

Application Level:

    Contains the code for VGG network simulating the hardware variations and checking the robustness. 
    The  network  is  trained  using  adam  optimizer  with initial  learning  rate  of  0.0001  for  
    200  epochs. The  learning rate  was  dropped  by  a  factor  of  10  at  epoch  number  100, 150  
    and  180. The  training  batch  size  and  testing  batch  size was 32 and 128 respectively. We 
    did small data augmentation (Flipping the training dataset and shifting the training dataset by 4 
    pixels). Cross-entropy loss function is used for training. Test accuracy for this network was found 
    to be 91.38%.

Circuit Simulation:

    Contains the codes for generating the graphs and histograms presented in the paper. The results were
    obtained from circuit simulations in tsmc65nm Technology node.

