import Data.Array.Accelerate as A
--[[[[]]]]
--x nets, each with three layers
--layer 0 has 50 neurons, each with 785 parameters
--layer 1 has 50 neurons, each with 51 parameters
--layer 2 has 10 neurons, each with 51 parameters

--arr = fromList (z:.3) ()


----seperateFronts gpu optimized
--separateFronts :: [[a]] -> [[a]]
--separateFronts x = map head x:map tail x
--separateFrontsv2 :: [[a]] -> [A.Acc (A.Array sh e)]
separateFrontsv2 x = (A.map Prelude.head (use arr)) : ((A.map Prelude.tail (use arr)) : []) --dont use run her
                    where 
                     len = Prelude.length x
                     arr = fromList (Z:.len) x
--square 
--getHeads :: [[a]] -> [a]
--
--enumFromN (z:.outLen:.inLen) 
--where 
-- outLen = length x
-- inLen = length $ head x