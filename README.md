# DL-Homework1

#### Problem2

#### MNIST digit classification task using Tensorflow

This is done as part of assignment for deeplearning course-IFT6135.

### Performance

MLP: xxx% (accuracy) 

CNN: 98.2% (accuracy)

### Description

This experiment was done to compare the performance of MLP (multi layer perceptron) and CNN (convolutional neural networks).
This is done on MNIST classification dataset. 

#### Step 1: Evaluate the MLP model

The MLP model contains two hidden layers with ReLU activation function.
Each hidden layer is XX dimension and is initialized with xxx.
Data is a 2-d matirx for MLP.

#### Step 2: Data preparation

Train data (60000)is split to train set (55000) and valid set (5000).
Test data consist of 10000 samples.

For CNN, data is 4-dim tensor.

#### Step 3: CNN model - Decide on the number of CNN layers

The CNN model contains three convolutional layers with max pooling and ReLU function.
Convolutional layer 1 has 8 maps.
Convoluitonal layer 2 has 32 maps.
Convoluitonal layer 3 has 64 maps.
All these layers use 7x7 kernels.
After convolution, the activation is flattened and passed through a fully connected layer which has 128 dimension.

#### Step 4:Set up the Network Architecture

|Layers                                                       |
|:-----------------------------------------------------------:|
| Convolution, Filter shape: (7,7,8), Stride=1, Padding=’SAME’|
| Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’|
 |ReLU|
 |Convolution, Filter shape:(7,7,32), Stride=1, Padding=’SAME’|
| Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’|
| ReLU|
|Convolution, Filter shape:(7,7,64), Stride=1, Padding=’SAME’|
| Max pooling (2x2), Window shape:(2,2), Stride=2, Padding=’Same’|
| ReLU|
 |Fully Connected Layer (128)|
 |ReLU|
| Fully Connected Layer (10)|
 |Softmax|
#### Step 5: Take the parameter count in the decided network
####  Parameter count in the 3 layer CNN


  
  | Name                                        | size                                            |parameters      |                                        
| :-------------------------------------------: |:-----------------------------------------------:| :----------------------------:|
| 0  input      | 1x28x28 | 0 |
| 1  conv2d1     |  8x28x28       |   (7x7x1+1)*8   =     400 |
| 2  maxpool1 |  8x28x28       |    0 |
| 3  conv2d2     |  32x28x28       |    (7x7x8+1)x32  =   12576 |
| 4  maxpool2 |  32x28x28            |    0 |
| 5  conv2d3     |  64x28x28       |    (7x7x32+1)x64  =   100416 |
| 6  maxpool3 |  64x28x28            |    0 |
|7  dense  |128   | (64x7x7+1)*128 = 401536 |
| 8  output |10 |(128+1)x10     =   1290 |

|Total number of learnable parameters in the CNN|
|:-------------------------------------------------------------------------------:|
| 5,16,218  (approx .5million)|

#### Step 6: Training

For every epoch(till 10 epochs for this task), we train network by train set and evaluate by valid set to check overfitting.
We use Stochastic Gradient Descent optimizer for parameter update.

#### Step 7: Evaluate the model

|Train Accuracy|Validation Accuracy|Train Loss|Validation Loss|
|:------------------------:|:--------------------------------:|:------------------------:|:--------------------------------:|
|99.74 |98.20 |0.0178|0.0576|


