# Neural Networks from Scratch (NumPy)

An MLP and a CNN written by hand in NumPy, with no deep learning libraries. I wrote the forward pass, backpropagation and SGD myself, and checked every gradient against a numerical estimate before training.

Built for Applied AIML 2, Adelaide University (2026).

## 1. MLP on Iris (`mlp-iris/`)

- 4 inputs, 16 ReLU hidden units, 3 softmax outputs
- He initialisation, cross-entropy loss, mini-batch SGD (batch 16, lr 0.01, 100 epochs)
- Gradient check: worst relative error 1.7e-08
- **Test accuracy: 93.3% (28/30).** Both errors were Versicolor predicted as Virginica, the two classes that overlap most

## 2. CNN on MNIST (`cnn-mnist/`)

- Conv(6, 3x3) > MaxPool > Conv(16, 3x3) > MaxPool > Dense(128) > Dense(10), 53,558 parameters
- Convolution done with im2col so it runs as one matrix multiply; col2im adds overlapping gradients back
- 48,000 train / 12,000 validation / 10,000 test, best weights picked on validation only
- **Test accuracy: 98.0%** after 5 epochs, 98.3% after 10
- Also tested dropout as a regulariser and compared train/validation gaps

## What I would improve

- The MLP loss was still going down at epoch 100, so it could train longer
- The CNN improved from 98.0% to 98.3% when I trained 10 epochs instead of 5, so it had not finished learning either
- Known issue: in the MLP, W2 is initialised using the output size (3). He initialisation should use the input size of that layer (16)

## Run it

Open either notebook in Jupyter or Colab and run all cells. Only NumPy, Matplotlib, scikit-learn (Iris loading) and torchvision (MNIST download) are needed.
