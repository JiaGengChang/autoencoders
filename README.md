# Self-supervised learning with autoencoders
This is a classical autoencoder for intended for CPU training, that is written in optimized code using C. 

## Problem description
We want to train a neural network to encode a one-hot encoded input vector that represents the digits 1 to 8 into a vector of length < 8, and to also reconstruct the original input given the encoding.

| target | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| - | - | - | - | - | - | - | - | - |
| 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| 2 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| 3 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| 4 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 |
| 5 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| 6 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 |
| 7 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 |

As seen above, the 8 input vectors correspond to the rows of a 8x8 identity matrix — a relatively easy pattern for the network to learn to encode and decode. We will start off with attempting to teach the network a binary representation of this input.

## Encoding a 8-bit one-hot vector with 3 bits
I trained a multi-layer perceptron to encode with 3 units a one-hot encoded vector of length 8. Here are the hyperparameters of the autoencoder:



| Hyperparameter | Info | Equation |
|-|-|-|
| activation function *g* | sigmoid | ![equation](http://www.sciweavers.org/tex2img.php?eq=g%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1%20%2B%20exp%28-x%29%7D%2C%20%5C%3Ag%27%28x%29%20%3D%20g%28x%29%5Ccdot%281-g%28x%29%29&bc=White&fc=Black&im=png&fs=18&ff=arev&edit=0)|
| loss | binary cross-entropy | ![equation](http://www.sciweavers.org/tex2img.php?eq=E=-%5Cfrac%7B1%7D%7BN%7D%5C%2C%5Csum_%7Bk%3D1%7D%5E%7BN%7D%5C%2Ct_%7Bk%7D%5C%2Clog%28p_%7Bk%7D%29%20%2B%20%281-t_%7Bk%7D%29%5C%2Clog%281-p_%7Bk%7D%29&bc=White&fc=Black&im=png&fs=18&ff=arev&edit=0) |
| optimizer | classical momentum | learning rate ![equation](http://www.sciweavers.org/tex2img.php?eq=%5Ceta%20%5Cin%20%5B0.01%2C%201.0%5D&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0) , momentum ![equation](http://www.sciweavers.org/tex2img.php?eq=m%20%5Cin%20%5B0%2C%201%5D&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0)|
| weights initialization | random uniform | between -0.1 and 0.1 |

Our loss function is binary cross-entropy, defined as:

<img src="https://render.githubusercontent.com/render/math?math=\huge E=-\frac{1}{N}\,\sum_{k=1}^{N}\,t_{k}\,log(p_{k}) %2B (1-t_{k})\,log(1-p_{k})">

where **t** is the target one-hot encoded vector, and **p** is the output layer activation (a probability vector), summation over *k* from N=1 to N=8 in our case. 

The derivatives of binary cross-entropy loss with respect to weights worked out to be:

![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bjk%7D%7D%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%5C%2C%28%5Cfrac%7Bt_%7Bk%7D%7D%7Bz_%7Bk%7D%7D-%5Cfrac%7B1-t_%7Bk%7D%7D%7B1-z_%7Bk%7D%7D%29%5Ccdot%20g%27%28x_%7Bk%7D%29%20%5Ccdot%20z_%7Bj%7D%20%3D%20%5Cdelta_%7Bk%7D%5Ccdot%20z_%7Bj%7D&bc=White&fc=Black&im=png&fs=18&ff=arev&edit=0)

for weights between hidden layer j and output layer k, and 

![equation](http://www.sciweavers.org/tex2img.php?eq=%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bij%7D%7D%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%20%5Csum%5Climits_%7Bk%3D1%7D%5E%7BN%7D%20%5C%3A%20%28%5Cfrac%7Bt_%7Bk%7D%7D%7Bz_%7Bk%7D%7D-%5Cfrac%7B1-t_%7Bk%7D%7D%7B1-z_%7Bk%7D%7D%29%20%5Ccdot%20g%27%28x_%7Bk%7D%29%20%5Ccdot%20w_%7Bjk%7D%20%5Ccdot%20g%27%28x_%7Bj%7D%29%20%5Ccdot%20z_%7Bi%7D%20%3D%20%5Cdelta_%7Bj%7D%20%5Ccdot%20z_%7Bi%7D&bc=White&fc=Black&im=png&fs=18&ff=arev&edit=0)

for weights between the input and hidden units.

## Training results

![effect of momentum on training at two learning rates](plots/mlp_m.png)

With standard gradient descent, training was slow, even at high learning rates of up to 10 (refer to left insert of above figure). Thus, I added a momentum term involving the previous time step as a modification to vanilla gradient descent. The weight update heuristic at epoch *t* becomes:

![equation](http://www.sciweavers.org/tex2img.php?eq=%5CDelta%20W_%7Bij%7D%28t%29%20%3D%20-%5Ceta%20%5C%2C%20%5Cfrac%7B%5Cpartial%20E%7D%7B%5Cpartial%20w_%7Bij%7D%7D%20%2B%20m%20%5C%2C%5CDelta%20W_%7Bij%7D%28t-1%29&bc=White&fc=Black&im=png&fs=18&ff=arev&edit=0)

for a pair of connected neurons *i* and *j* (or *j* and *k* for that matter), and where *m* is scalar that practitioners suggest to be kept between 0 and 1. The result in training with varying *m*, for two different learning rates (lr = 0.01 and lr = 0.05) is shown in the plots above.

## How the 8-3-8 autoencoder learns to encode its input

How is the network learning to encode input vectors? We look at the final state of the weights for the best network to see what is happening.

![weight matrices of 8-3-8 autoencoder](plots/mlp_w.png)

Without loss of generality, what the network has learned is to match the signs between mirrored pairs of weights <img src="https://render.githubusercontent.com/render/math?math=w_{nj}, \: w_{jn} \: \forall n \in [0,7]">. For example, if *t* is ![equation](http://www.sciweavers.org/tex2img.php?eq=%28%201%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%5C%3A%200%20%29&bc=White&fc=Black&im=png&fs=12&ff=arev&edit=0), then only j0 and k0 will fire, producing a large positive activation in output node k0 because <img src="https://render.githubusercontent.com/render/math?math=w_{i0,j0}"> and <img src="https://render.githubusercontent.com/render/math?math=w_{j0,k0}">
 have the same sign (does not matter whether positive or negative). The two other hidden units also have synergistic weights entering/leaving them, but they will stay silent because the inputs into those units is 0. 

![training comparison between two autoencoders](plots/mlp_8_16.png)



![weight matrices of 16-3-16 autoencoder](plots/mlp16_w.png)
