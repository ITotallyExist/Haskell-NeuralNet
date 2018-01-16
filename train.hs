import NeuralNet
----Trains a network
----Takes in the filepath of the network txt file as a string, the path of the training inputs, the path of the corresponding correct outputs, the number of epochs to train the network for, the amount of splits to make to the data (i.e., if you want each epoch to train on only 1/4 of the data, this number will be four.  if this number is less than the number of epochs, then it will keep cycling through each subarray of inputs/answers until it has gone through the number of epochs specified.  If you want each epoch to train on the entire data set, this value should be one.), and the stepsize. updates the network txt file with a new network

main = do
  net <- fmap read (readFile "network.txt")
  inputs <- fmap read (readFile "trainingInputVector.txt")
  outputs <- fmap read (readFile "trainingOutputVector.txt")
  writeFile "trainedNetwork.txt" $ show $ NeuralNet.train 1000 5 net inputs outputs 0.3
-- epochs numberOfSplitsForStochasticDescent ... trainingStepSize

----trainFromFiles :: String -> String -> String -> Int -> Int -> Double -> updates file
--trainFromFiles netFile trainingInputFile trainingOutputFile outputFile epochs splits stepSize = do
--  net <- fmap read (readFile netFile)
--  inputs <- fmap read (readFile trainingInputFile)
--  outputs <- fmap read (readFile trainingOutputFile)
--  writeFile outputFile $ show $ NeuralNet.train epochs splits net inputs outputs stepSize