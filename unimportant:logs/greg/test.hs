import NeuralNet

main = do
  n <- fmap read (readFile "trainedNetwork.txt")
  x <- fmap read (readFile "inputVector.txt")
  y <- fmap read (readFile "outputVector.txt")
  print $ testClassification n x y
