import NeuralNet
main = do
  net <- fmap read (readFile "blankNetwork.txt")
  inputs <- fmap read (readFile "trainingInputVector.txt")
  outputs <- fmap read (readFile "trainingOutputVector.txt")
  print $ "test"
  writeFile "profilingOutput.txt" $ show $ NeuralNet.train 100 10 net inputs outputs 0.03