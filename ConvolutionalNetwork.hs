module ConvolutionalNetwork where
import NeuralNet
--to implement convolutional neural networks, i can make two networks, a convolutional one that feeds into an mlp from NeuralNet, and train the mlp normally, then train the convolutional layer on top of that

data ConvNet = ConvNet [ConvLayer] [Layer] deriving (Show, Read)
--a convolutional set of layers, then the mlp following that which will be trained normally


--stores stride, kernel x dimension, kernel y dimension, output x dimension, output y dimension, and a 2d list of however many neurons are required (x,y (a neuron for each output))
data ConvLayer = ConvLayer Int Int Int Int Int [[Neuron]]