--i heard that recurrent networks work by creating new layers and storing them, might be easy to do with a layer data type
--consider adding a gradience vector data type
--the bug has been narrowed down do trainEpoch (i.e. derivateNet works and activateNet works, average3d works, and updatelay works, the error starts at trainEpoch)
    --sigmoid, perceptron, relu, softmax (made for last layer), leakyRelu, elu, hyperbolic tangent
    --only sigmoid, perceptron, relu, and leakyRelu currently supported
data NeuClas = Sig | Per | Relu | NeuSoftMax | Elu | LeakyRelu | Htan deriving (Show)

    --class of neuron (basically just the function it uses), then an array of all of the weights going into it, then its bias
    --A clean neuron contains only the weights and bias
    --A clean neuron can be activated by the activateNeu function (technically this actually just creates a new one instead of changing the old one because variables in haskell aren't mutable, but you probably already knew that). The activated neuron will store the same data as the clean neuron, but also stores its activation value as a double and the derivative of its acitvation function at the activation value that is input into it as a double
    --a derived neuron is an activated neuron where the activation level has been replaced the cost function's derivative over it
    --for the list of weights, the first item is the top weigh, last is the bottom weight
    
    --activated has [weight] activation functionDerivative
data Neuron = Neuron NeuClas [Double] Double deriving (Show)
getNeuronWeights :: Neuron -> [Double]
getNeuronWeights (Neuron _ x _) = x

--layer types, e.g. recurrent, convolutional, feed forward
--currently only feed forward is supported
data LayClass = Recurrent | Convolutional | Regular | LaySoftMax deriving (Show)
    --clean activated and derived refer to the sates of all of the layer's neurons
    --In an activated layer, the array of doubles is an array of activation levels of all of the neurons within that layer, 
    --for the list of neurons, the top neuron is the first in the list and the bottom is the last in the list
data Layer = Layer LayClass [Neuron] deriving (Show)
getLayerWeights :: Layer -> [[Double]]
getLayerWeights (Layer _ x) = map getNeuronWeights x

--takes in a list of layer types in string form, a 2d list of all of the neuron types in string form, a 3d list of all of the weights, and a 2d list of all of the Biases and returns a list of layers (or a network)
generateNetwork :: [String] -> [[String]] -> [[[Double]]] -> [[Double]] -> [Layer]
generateNetwork [] _ _ _ = []
generateNetwork (lType:lTypes) (nTypes:nTypeses) (nWeightses:nWeightseses) (nBiases:nBiaseses) = (generateLayer lType nTypes nWeightses nBiases):(generateNetwork lTypes nTypeses nWeightseses nBiaseses)

--Takes in a layer type in the form of a string, a list of neuron types as strings, a 2d list of weights, and a list of biases and returns a layer generated from those parameters
--here is where to add functionality in terms of more layer types (like implementing current ones and adding pooling or something) in the future
generateLayer :: String -> [String] -> [[Double]] -> [Double] -> Layer
generateLayer "Recurrent" x y z = Layer Recurrent (generateNeurons x y z)
generateLayer "Convolutional" x y z = Layer Convolutional (generateNeurons x y z)
generateLayer "Regular" x y z = Layer Regular (generateNeurons x y z)
generateLayer "LaySoftMax" x y z = Layer Regular (generateNeurons x y z)


generateNeurons :: [String] -> [[Double]] -> [Double] -> [Neuron]
generateNeurons [] _ _ = []
generateNeurons (nType:nTypes) (nWeights:nWeightses) (nBias:nBiases) = (generateNeuron nType nWeights nBias):(generateNeurons nTypes nWeightses nBiases)

--Takes in a neuron type in the form of a string and a list of neuron weights and the a neuron bias and returns a neuron generated with those parameters
generateNeuron :: String -> [Double] -> Double -> Neuron
generateNeuron "Sig" weights bias = Neuron Sig weights bias
generateNeuron "Per" weights bias = Neuron Per weights bias
generateNeuron "Relu" weights bias = Neuron Relu weights bias
generateNeuron "NeuSoftMax" weights bias = Neuron NeuSoftMax weights bias
generateNeuron "Elu" weights bias = Neuron Elu weights bias
generateNeuron "LeakyRelu" weights bias = Neuron LeakyRelu weights bias
generateNeuron "Htan" weights bias = Neuron Htan weights bias

--takes in a network and an array of test inputs and their corresponding outputs and returns the accuracy of the network
--only works for classification networks
testClassification :: [Layer] -> [[Double]] -> [[Double]] -> Double
testClassification net inputs outputs = testClassificationHelper net inputs outputs 0 0

testClassificationHelper :: [Layer] -> [[Double]] -> [[Double]] -> Double -> Double -> Double
testClassificationHelper _ [] _ correct incorrect = (correct/(correct+incorrect))
testClassificationHelper net (input:inputs) (output:outputs) correct incorrect | netMax == outMax = next (correct+1) incorrect
                                                                               | otherwise = next correct (incorrect+1)
                                                                               where
                                                                                next = testClassificationHelper net inputs outputs
                                                                                netMax = greatestIndex (getOutput net input)
                                                                                outMax = greatestIndex output

--takes in the number of epochs, the amount of splits to make to the data (i.e., if you want each epoch to train on only 1/4 of the data, this number will be four.  if this number is less than the number of epochs, then it will keep cycling through each subarray of inputs/answers until it has gone through the number of epochs specified.  If you want each epoch to train on the entire data set, this value should be one.), a clean network as an array of layers, a list of training data inputs as an array of array of doubles (each subarray corresponding to one input, whith each element being the activation of an input neuron), a list of the desired results from those inputs, and the stepSize.  Returns a new trained network
train :: Int -> Int -> [Layer] -> [[Double]] -> [[Double]] -> Double -> [Layer]
train epochs splits network inputs outputs stepSize = trainHelper epochs splitInputs splitOutputs network splitInputs splitOutputs stepSize
                                                    where
                                                     splitInputs = splitInto splits inputs
                                                     splitOutputs = splitInto splits outputs

trainHelper :: Int -> [[[Double]]] -> [[[Double]]] -> [Layer] -> [[[Double]]] -> [[[Double]]] -> Double -> [Layer]
trainHelper 1 ogInputs ogOutputs network (input:inputs) (output:outputs) stepSize = trainEpoch network input output stepSize
trainHelper epochs ogInputs ogOutputs network (input:[]) (output:[]) stepSize = trainHelper (epochs - 1) ogInputs ogOutputs (trainEpoch network input output stepSize) ogInputs ogOutputs stepSize
trainHelper epochs ogInputs ogOutputs network (input:inputs) (output:outputs) stepSize = trainHelper (epochs - 1) ogInputs ogOutputs (trainEpoch network input output stepSize) inputs outputs stepSize

--Takes in a clean network as an array of layers, a list of inputs input as an array of arrays of doubles, and the corresponding desired outputs as an array of arrays of doubles, a training step size, then returns the network with each neuron replaced by a derived version of it.
--the training step size is a percentage of the gradience
--this function is probably a prime target for parralelization - look into it later
    --maybe have it get inputted the number of cores? then it splits the input and desired output arrays into that many subsections/processes?
trainEpoch :: [Layer] -> [[Double]] -> [[Double]] -> Double -> [Layer]
trainEpoch network [] desiredOutputs stepSize = network --this line is only here in the edge case that the training inputs where split into more subsections than there were inputs
trainEpoch network inputs desiredOutputs stepSize = zipWith (updateLay stepSize) (average3d (trainEpochHelper [] network inputs desiredOutputs (reverse(map getLayerWeights network)))) network-- fix the reverse lag here, maybe find some sort of reverse map?

--takes in a step size, a gradience, and a layer, and updates the layer
updateLay :: Double -> [[Double]] -> Layer -> Layer
updateLay stepSize gradience (Layer y x) = Layer y (zipWith (updateNeu stepSize) gradience x)

updateNeu :: Double -> [Double] -> Neuron -> Neuron
updateNeu stepSize gradience (Neuron y weights bias) = Neuron y (tail attributes) (head attributes)
                                                     where attributes = zipWith (updateAttribute stepSize) gradience (bias:weights)

updateAttribute :: Double -> Double -> Double -> Double
updateAttribute stepSize gradience attribute = (attribute - (stepSize*gradience))
--keeps track of the gradiences so far and returns the list of them, so that the main function can average them and update the net using stepsize and stuff
--takes in an empty list, the network as an array of layers, the training inputs, the training outputs, and the weights of the network as a 2d array but in reverseand returns a list of gradiences, each in the form of [[[dw1-1-1, dw1-1-2, ..., db1-1],[dw1-2-1, dw1-2-2, ..., db1-2],...],[[dw2-1-1, dw2-1-2, ..., db2-1],[dw2-2-1, dw2-2-2, ..., db2-2],...],...]
trainEpochHelper :: [[[[Double]]]] -> [Layer] -> [[Double]] -> [[Double]] -> [[[Double]]]-> [[[[Double]]]]
trainEpochHelper soFar net (input:[]) (output:[]) weights = (derivateNet input output (map item2 activeNet) (map head activeNet) weights):soFar
                                                  where activeNet = activateNet net input
trainEpochHelper soFar net (input:inputs) (output:outputs) weights = trainEpochHelper ((derivateNet input output (map item2 activeNet) (map head activeNet) weights):soFar) net inputs outputs weights
                                                           where activeNet = activateNet net input


--takes in a list, returns the second item in it
item2 :: [a] -> a
item2 x = head(tail x)

--[[[act1-1,act1-2,...],[der1-1,der1-2,...]],[[act2-1,act2-2,...],[der2-1,der2-2,...]],...]
--side note, "next" in the following few comments actually just means the adjacent layer that is close to the output of the nextwork than the current layer
--and of course "previous" here means closer to input

--takes in the input to the network, the desired output of the network, the derivatives of the functions of each neuron in the network as a 2d aray, each subarray being a layer, the activations of each neuron, the weights of each neuron, and outputs the gradience of the network for this particular training example as [[[db1-1, dw1-1-1, dw1-1-2, ...],[db1-2, dw1-2-1, dw1-2-2, ...],...],[[db2-1, dw2-1-1, dw2-1-2, ...],[db2-2, dw2-2-1, dw2-2-2, ...],...],...]
--note, the arrays taken as inputs and are in reverse order (the output layer is first and the input last)
--the ordering of the neurons within the layers, however, is normal
--the ouput is also normal
--assumes the net has at least two layers (one input and one ouput)
derivateNet :: [Double] -> [Double] -> [[Double]] -> [[Double]] -> [[[Double]]] -> [[[Double]]]
derivateNet input correct (lDFunc:lDFuncs) (lAct:lActs) weights = (derivateNetHelper lDFunc weights (head x) lDFuncs (tail lActs) input ((tail x):[]))
                                                                where x = separateFronts (derivateLastLayer lAct correct lDFunc (head lActs))

derivateNetHelper :: [Double] -> [[[Double]]] -> [Double] -> [[Double]] -> [[Double]] -> [Double] -> [[[Double]]] -> [[[Double]]]
derivateNetHelper nDFuncs (nWeights:weights) nDacts (cDfuncs:dFuncs) [] input soFar = (map tail (derivateLayer nDFuncs nWeights nDacts cDfuncs input)):soFar
derivateNetHelper nDFuncs (nWeights:weights) nDacts (cDfuncs:dFuncs) (pActs:acts) input soFar = derivateNetHelper cDfuncs weights cDacts dFuncs acts input (cResult:soFar)
                                                                                              where
                                                                                               cLOut = derivateLayer nDFuncs nWeights nDacts cDfuncs pActs
                                                                                               x = separateFronts cLOut
                                                                                               cDacts = head x
                                                                                               cResult = tail x
                                                                                        
--needs the derived functions from the next layer, the weights from the next layer (2d array), the derivatives of the activations of the next layer, activations of current layer, derivative of the functions of the current layer of neurons, and the activations of the previous layer
--outputs the deravites for the weights and biases of each neuron in the layer in the form of [[da1,db1,dw1-1, dw1-2, ...],[da2,db2,dw2-1, dw2-2, ...],...]
derivateLayer :: [Double] -> [[Double]] -> [Double] -> [Double] -> [Double] -> [[Double]]
derivateLayer _ _ _ [] _ = []
derivateLayer nDFuncs nWeights nDacts (cDFunc:cDFuncs) pActs = (cDAct:(derivateNeuronWeightsandBias cDAct cDFunc (1:pActs))):(derivateLayer nDFuncs (tail x) nDacts cDFuncs pActs)
                                                             where 
                                                              x = separateFronts nWeights
                                                              cDAct = derivateNeuronActivation (head x) nDFuncs nDacts

--special case of derivateLayer that works for the last layer of the network (output layer)
--takes in an array of the activations of the last layer, and array of the desired outputs of the last layer, an array of the derivatives of the functions of the last layer, and an array of the activations of the previous (second to last) layer, outputs the derivatives for the weights and biases of each neuron in the layer in the form of [[da1,db1,dw1-1, dw1-2, ...],[da2,db2,dw2-1, dw2-2, ...],...]
derivateLastLayer :: [Double] -> [Double] -> [Double] -> [Double] -> [[Double]]
derivateLastLayer [] _ _ _ = []
derivateLastLayer (lAct:lActs) (correct:corrects) (dFunc:dFuncs) pActs = (cDact:(derivateNeuronWeightsandBias cDact dFunc (1:pActs))):(derivateLastLayer lActs corrects dFuncs pActs)
                                                                       where cDact = 2*(lAct - correct)

--takes in an array of the weights from the current neuron to each next neuron, and an array of the derivatives of the functions of the next layer at their inputs, and the derivatives of the activations of the next layer and returns the derivative of the activation of the current neuron
derivateNeuronActivation ::  [Double] -> [Double] -> [Double] -> Double
derivateNeuronActivation [] _ _ = 0
derivateNeuronActivation (weight:weights) (dFunc:dFuncs) (dAct:dActs) = (weight*dFunc*dAct) + (derivateNeuronActivation weights dFuncs dActs)

--takes the derivative of the activation of a neuron, the derivative of the function of that neuron given its inputs, and a list of all of the activations of the previous layer of neurons (with a one appended as the first value) and returns the derivatives of each of the weights of each neuron and the derivative of the bias in the form [dbias,dw1,dw2,...]
derivateNeuronWeightsandBias :: Double -> Double -> [Double] -> [Double]
derivateNeuronWeightsandBias _ _ [] = []
derivateNeuronWeightsandBias dAct dFunc (activation:activations) = (activation*dAct*dFunc):(derivateNeuronWeightsandBias dAct dFunc activations)

--takes in a network as an array of layers and an input to the network, and returns the output
getOutput :: [Layer] -> [Double] -> [Double]
getOutput x y = head (head (activateNet x y))
--takes in a clean network as an array of layers, and a list of inputs and returns the activations and derivative of the functions of each neuron in the form of [[[act1-1,act1-2,...],[der1-1,der1-2,...]],[[act2-1,act2-2,...],[der2-1,der2-2,...]],...]
--to make the back propagation more optimized, this list is backwards (as in the ouput layer of the net is first and the input layer of the net is last)
activateNet :: [Layer] -> [Double] -> [[[Double]]]
activateNet x y = activateNetHelper x y []

activateNetHelper :: [Layer] -> [Double] -> [[[Double]]] -> [[[Double]]]
activateNetHelper (layer:[]) inputs soFar = (activateLay layer inputs):soFar
activateNetHelper (layer:layers) inputs soFar = activateNetHelper layers (head z) (z:soFar)
                                              where z = activateLay layer inputs

--takes two layers of neurons and activates the second one using the activation values from the first
--takes a layer and a list of all of the outputs of the previous layer and returns the activations and derivatives of the functions of each of its neurons in the form of [[act1,act2,...],[der1,der2,...]]
activateLay :: Layer -> [Double] -> [[Double]]
activateLay (Layer Regular neurons) inputs = (fst l):(snd l):[]
                                           where l = unzip (activateNuesRegular neurons inputs)

--takes a list of neurons and a list of inputs and returns a list of the activations and derivations of the input functions of the neurons as [[act1,act2,...],[der1,der2,...]]
activateNuesRegular :: [Neuron] -> [Double] -> [(Double,Double)]
activateNuesRegular [] _ = []
activateNuesRegular (neuron:neurons) inputs = (activateNue neuron inputs):(activateNuesRegular neurons inputs)

--takes a neuron and the list of its inputs (the first item is the top input, the last is the bottom) and returns its activation and the derivation of its input function as [act1,der1]
activateNue :: Neuron -> [Double] -> (Double,Double)
activateNue (Neuron Sig inWeights bias) inputs = (sigVal, (sigVal*(1-sigVal)))
                                               where 
                                                z = calculateZ inputs inWeights bias
                                                sigVal = (1/(1+(2**(-z))))
activateNue (Neuron Relu inWeights bias) inputs | z>0 = (z, 1)
                                                | otherwise = (0, 0)
                                                where z = calculateZ inputs inWeights bias
activateNue (Neuron LeakyRelu inWeights bias) inputs | z>0 = (z, 1)
                                                     | otherwise = ((z/100), 0.01)
                                                     where z = calculateZ inputs inWeights bias
activateNue (Neuron Per inWeights bias) inputs | z>0 = (1, 0)
                                               | otherwise = (0, 0)
                                               where z = calculateZ inputs inWeights bias
--activateNue (CleanNeu SoftMax inWeights bias) inputs
    --source for implementing/understanding this (https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/)
--activateNue (CleanNeu Elu inWeights bias) inputs

--calculates the input to a neuron after all the weights and the bias are added to the original array of inputs
--takes in the array of inputs, the array of weights, and the bias
calculateZ :: [Double] -> [Double] -> Double -> Double
calculateZ [] _ y = y
calculateZ (input:inputs) (weight:weights) bias = calculateZ inputs weights (bias+(input*weight))

--Takes in any 2d array as inputs and outputs a new 2d array made up of an array of all of the heads of each original subarray appended to the front of an array of all the tails of each past array
separateFronts :: [[a]] -> [[a]]
separateFronts x = map head x:map tail x

--original Version, around same speed/order, but messier (kept in case it is needed for whatever reason) (i am like 50% sure that they do the exact same thing though but this one doesnt use map)
--separateFrontsOriginal :: [[a]] -> [[a]]
--separateFrontsOriginal (x:[]) = ((head x):[]):((tail x):[])
--separateFrontsOriginal (x:xs) = ((head x):(head(separateFronts xs))):((tail x):(tail (separateFronts xs)))
--takes in a list of lists of dimension n and returns their average, doesnt work for dumb reasons
--averageNd :: Int -> [a] -> a
--averageNd _ ([]:xs) = []
--averageNd 1 x = (average (head z)):(averageNd 1 (tail z))
--              where z = separateFronts x
--averageNd n x = (averageNd (n-1) (head z)):(averageNd n (tail z))
--              where z = separateFronts x

--takes in a list of 3d lists and returns the average of those in a new list
average3d :: [[[[Double]]]] -> [[[Double]]]
average3d ([]:xs) = []
average3d x = (average2d (head z)):(average3d (tail z))
            where z = separateFronts x

--takes in a list of 2d lists and returns the average of those in a new list
average2d :: [[[Double]]] -> [[Double]]
average2d ([]:xs) = []
average2d x = (average1d (head z)):(average2d (tail z))
            where z = separateFronts x
            
--takes in a list of lists and returns the average of those in a new list
average1d :: [[Double]] -> [Double]
average1d ([]:xs) = []
average1d x = (average (head z)):(average1d (tail z))
            where z = separateFronts x
            
--Takes in a list of numbers and returns its average value
average :: [Double] -> Double
average x = averageHelper x 0 0

averageHelper :: [Double] -> Double -> Double -> Double
averageHelper [] cSum cLength = cSum/cLength
averageHelper (x:xs) cSum cLength = averageHelper xs (cSum+x) (cLength+1)

--takes in an array and a number and splits the array into that many subarrays of equal size
--on the same order as reverse
splitInto :: Int -> [a] -> [[a]]
splitInto n inputs = splitIntoHelper [] (blankList n) inputs

splitIntoHelper :: [[a]] -> [[a]] -> [a] -> [[a]]
splitIntoHelper soFar lists [] = soFar++lists
splitIntoHelper soFar [] inputs = splitIntoHelper [] soFar inputs
splitIntoHelper soFar (list:lists) (input:inputs) = splitIntoHelper ((input:list):soFar) lists inputs

--takes in a number and returns a list of that many empty lists
blankList :: Int -> [[a]]
blankList 0 = []
blankList x = []:(blankList (x-1)) 

--Takes in an array of doubles and outputs the index of the greatest value within the array
greatestIndex :: [Double] -> Int
greatestIndex [] = 0
greatestIndex (num:nums) = greatestIndexHelper num 0 1 nums

greatestIndexHelper :: Double -> Int -> Int -> [Double] -> Int
greatestIndexHelper cGreatest cGreatestIndex cIndex [] = cGreatestIndex
greatestIndexHelper cGreatest cGreatestIndex cIndex (num:nums) | cGreatest > num = greatestIndexHelper cGreatest cGreatestIndex (cIndex+1) nums
                                                               | otherwise = greatestIndexHelper num cIndex (cIndex+1) nums