import NeuralNet

----generates a network file
----takes in the file path to a list of layer types in string form, the file path of a 2d list of all of the neuron types in string form, the file path of a 3d list of all of the weights, the file path of a 2d list of all of the Biases and returns a list of layers (or a network), and the desired file path for the output, ouputs a network file

main = do
  layerTypes <- fmap read (readFile "layerTypes.txt")
  neuronTypes <- fmap read (readFile "neuronTypes.txt")
  weights <- fmap read (readFile "weights.txt")
  biases <- fmap read (readFile "biases.txt")
  writeFile "network.txt" $ show $ NeuralNet.generateNetwork layerTypes neuronTypes weights biases
  
--generateFromFiles layerTypesFile neuronTypesFile weightsFile biasesFile outFile = do
--  layerTypes <- fmap read (readFile layerTypesFile)
--  neuronTypes <- fmap read (readFile neuronTypesFile)
--  weights <- fmap read (readFile weightsFile)
--  biases <- fmap read (readFile biasesFile)
--  writeFile outFile $ show $ NeuralNet.generateNetwork layerTypes neuronTypes weights biases