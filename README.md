# Self-supervised learning with autoencoders
This is a classical autoencoder written in C, along with a mini-library for matrix operations.

![banner plot](plots/mlp_banner.png)

## Problem description
Suppose we want to train a neural network to encode an input vector with one 1 and seven 0's, corresponding to the 8 cases below.

| n | t0 | t1 | t2 | t3 | t4 | t5 | t6 | t7 |
| - | - | - | - | - | - | - | - | - |
| 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| ... | ... | ... | ... | ... | ... | ... | ... | ... |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

The 8 input vectors correspond to the one-hot encoding of 8 categorical variables — a relatively easy pattern for the network to learn to encode and decode. This can be most efficiently done using 3 binary bits. We will train a network to learn the binary representation of this input vector.

## Encoding 8 categories with 3 bits
I trained a multi-layer perceptron to encode with 3 units a one-hot encoded vector of length 8. For the sake of clarity and terminology of indices used, here is a visual representation of the network, showing the i, j, and k indices used to describe the input, hidden, and output layer units, and the weights between them.

![multi-layer perceptron diagram](plots/mlp_arch.png)

Here are the details of the multi-layer perceptron needed for its reconstruction:

| Hyperparameter | Info | Equation |
|-|-|-|
| activation function *g* | sigmoid | activation <img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cg%28x%29+%3D+%5Cfrac%7B1%7D%7B1+%2B+exp%28-x%29%7D"> , derivative <img src="https://render.githubusercontent.com/render/math?math=%5Cdisplaystyle+%5Cg%27%28x%29+%3D+%5Cg%28x%29+%5Ccdot+%281+-+g%28x%29%5C%2C%29" alt="\g'(x) = \g(x) \cdot (1 - g(x)\,)"> |
| loss | binary cross-entropy | <img src="https://render.githubusercontent.com/render/math?math=\large E=-%5Cfrac%7B1%7D%7BN%7D%5C%2C%5Csum_%7Bk%3D1%7D%5E%7BN%7D%5C%2Ct_%7Bk%7D%5C%2Clog%28p_%7Bk%7D%29%20%2B%20%281-t_%7Bk%7D%29%5C%2Clog%281-p_%7Bk%7D%29"> |
| optimizer | classical momentum | learning rate <img src="https://render.githubusercontent.com/render/math?math=%5Ceta%20%5Cin%20%5B0.01%2C%201.0%5D"> , momentum <img src="https://render.githubusercontent.com/render/math?math=m%20%5Cin%20%5B0%2C%201%5D">|
| weights initialization | random uniform | between -0.1 and 0.1 |

Our loss function is binary cross-entropy, defined as:

<img src="https://render.githubusercontent.com/render/math?math=\huge E=-\frac{1}{N}\,\sum_{k=1}^{N}\,t_{k}\,log(p_{k}) %2B (1-t_{k})\,log(1-p_{k})">

where **t** is the target one-hot encoded vector, and **p** is the output layer activation (a probability vector), summation over *k* from k=1 to k=8 (N) in our case. 

The derivatives of binary cross-entropy loss with respect to weights worked out to be:

<img src="https://render.githubusercontent.com/render/math?math=\huge %5Cdisplaystyle+%5Cfrac%7B%5Cpartial+E%7D%7B%5Cpartial+w_%7Bjk%7D%7D+%3D+-%5Cfrac%7B1%7D%7BN%7D%5C%2C%28%5Cfrac%7Bt_%7Bk%7D%7D%7Bz_%7Bk%7D%7D-%5Cfrac%7B1-t_%7Bk%7D%7D%7B1-z_%7Bk%7D%7D%29%5Ccdot+g%27%28x_%7Bk%7D%29%5Ccdot+z_%7Bj%7D+%3D+%5Cdelta_%7Bk%7D%5Ccdot+z_%7Bj%7D" 
alt="\frac{\partial E}{\partial w_{jk}} = -\frac{1}{N}\,(\frac{t_{k}}{z_{k}}-\frac{1-t_{k}}{1-z_{k}})\cdot g'(x_{k})\cdot z_{j} = \delta_{k}\cdot z_{j}">

for weights between hidden layer j and output layer k, and 

<img src="https://render.githubusercontent.com/render/math?math=\huge %5Cdisplaystyle+%5Cfrac%7B%5Cpartial+E%7D%7B%5Cpartial+w_%7Bij%7D%7D+%3D+-%5Cfrac%7B1%7D%7BN%7D+%5Csum_%7Bk%3D1%7D%5E%7BN%7D+%5C%3A+%28%5Cfrac%7Bt_%7Bk%7D%7D%7Bz_%7Bk%7D%7D-%5Cfrac%7B1-t_%7Bk%7D%7D%7B1-z_%7Bk%7D%7D%29+%5Ccdot+g%27%28x_%7Bk%7D%29+%5Ccdot+w_%7Bjk%7D+%5Ccdot+g%27%28x_%7Bj%7D%29+%5Ccdot+z_%7Bi%7D+%3D+%5Cdelta_%7Bj%7D%5Ccdot+z_%7Bi%7D" 
alt="\frac{\partial E}{\partial w_{ij}} = -\frac{1}{N} \sum_{k=1}^{N} \: (\frac{t_{k}}{z_{k}}-\frac{1-t_{k}}{1-z_{k}}) \cdot g'(x_{k}) \cdot w_{jk} \cdot g'(x_{j}) \cdot z_{i} = \delta_{j}\cdot z_{i}">

for weights between the input and hidden units.

## Training results

![effect of momentum on training at two learning rates](plots/mlp_m.png)

With standard gradient descent, training was slow, even at high learning rates of up to 10 (refer to left insert of above figure). Thus, I added a momentum term involving the previous time step as a modification to vanilla gradient descent. The weight update heuristic at epoch *t* becomes:

<img src="https://render.githubusercontent.com/render/math?math=\huge %5CDelta%20W_%7Bij%7D%28t%29%20%3D%20-%5Ceta%20%5C%2C%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bij%7D%7D%20%2B%20m%20%5C%2C%5CDelta%20W_%7Bij%7D%28t-1%29">

for a pair of connected neurons *i* and *j* (or *j* and *k* for that matter), and where *m* is scalar that practitioners suggest to be kept between 0 and 1. The result in training with varying *m*, for two different learning rates (lr = 0.01 and lr = 0.05) is shown in the plots above.

## How the 8-3-8 autoencoder learns to encode its input

How is the network learning to encode input vectors? We look at the final state of the weights for the best network to see what is happening.

![weight matrices of 8-3-8 autoencoder](plots/mlp_w.png)

Without loss of generality, what the network has learned is to match the signs between mirrored pairs of weights <img src="https://render.githubusercontent.com/render/math?math=\large w_{nj}, \: w_{jn} \: \forall n \in [0,7]">. For example, if *t* is <img src="https://render.githubusercontent.com/render/math?math=\large %28%201%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%20%29">, then only j0 and k0 will fire, producing a large positive activation in output node k0 because <img src="https://render.githubusercontent.com/render/math?math=\large w_{i0,j0}"> and <img src="https://render.githubusercontent.com/render/math?math=\large w_{j0,k0}">
 have the same sign (does not matter whether positive or negative). The two other hidden units also have synergistic weights entering/leaving them, but they will stay silent because the inputs into those units is 0. 

## Encoding a 16-bit one-hot vector with 3 bits

Is it possible to be even more efficient and encode a one-hot vector \textit{twice} the size as our previous vector, with 3 hidden units? If each hidden unit can represent 3 activation states, then we can encode up to a one-hot vector of length 3^n = 3^3 = 27. As we are using a sigmoid activation, the three activation states are naturally 0, 0.5, and 1, requiring asymptotic input values of -Inf, 0, and Inf at each hidden unit. While our loss will never reach 0, it is theoretically possible to train such a network.

To try this out, I trained a multi-layer perceptron to encode with 3 units a one-hot encoded vector of length **16**. The training results are shown below. 

![training comparison between two autoencoders](plots/mlp_8_16.png)

As per our predictions, the autoencoder is still able to encode a one-hot representation of 16 different inputs. Training is slower, because there are more weights to adjust, but binary cross-entropy is reduced to a comparably small value after around 10000 epochs. We now looka at the weights of this 16-3-16 autoencoder.

## How the 16-3-16 autoencoder learns to encode its input

![weight matrices of 16-3-16 autoencoder](plots/mlp16_w.png)

Visualizing the weight matrices in heatmap above, we see that the network is again solving the problem by mirroring the sign of the weights. Additionally, it mirrors the relative magnitude of the weights. This time, each hidden node has three states, as the mirrored incoming and outgoing weights are in one of negative (blue), zero (off-white), and positive (red). The connections entering output node 14 (k14) look all negative, but this is offset by the positive bias (<img src=
"https://render.githubusercontent.com/render/math?math=\large %5Cdisplaystyle+w_%7Bj3%2Ck14%7D" 
alt="w_{j3,k14}">). Again, the bias in the input layer is not doing much (it cannot be any other value than 0 otherwise non-1 units will fire), but the bias in the hidden layer is helping to adjust the median weight to zero.

## Project structure
The code for the 8-unit multi-layer perceptron is available as `testmlp.c`, and the code for the 16-unit multi-layer perceptron is found in `testmlp16.c`. They are hard-coded in terms of the number of units and the inputs they accept. The `utils.c` is the definitions file for simple matrix operations (e.g., dot product, hadamard product, and scalar operations). The `output` directory contains the .csv files used to generate the plots. Jupyter notebooks for generating the plots are not included.
