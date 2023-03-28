# Metallic-Plate-Permeability-Estimation-using-Single-Frequency-Eddy-Current-Testing
The Deep Learning code for paper Metallic Plate Permeability Estimation using Single Frequency Eddy Current Testing in the Presence of Probe Lift-off

The architecture of the DL model is the modified ResNet18-1D. 
The inductance spectrum is fed to the input layer of the model, while other input variables are appended to the internal layer of the model, cascading with low-dimensional feature vectors extracted from the inductance spectrum.

![model_architecture_3](https://user-images.githubusercontent.com/64902728/228315018-08c626da-946e-4e16-9aab-fc797b374d94.png)

Get dataset from Kaggle ().
Run the train.py to train the network. 
