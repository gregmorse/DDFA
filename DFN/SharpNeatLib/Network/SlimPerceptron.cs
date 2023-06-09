using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Phenomes.NeuralNets;
using System.Diagnostics;
using System.Threading.Tasks;

namespace SharpNeat.Network
{
    /// <summary>
    /// A single layer perceptron that's quite slim and runs on arrays as opposed to connection objects like other networks.
    /// </summary>
    public class SlimPerceptron 
    {
        public ParallelOptions po { get; set; }

        //always 1 bias please
        int biasCount = 1;
        int inputCount;
        int fullInputCount;
        int outputCount;
        double[][] _weights = null;

        bool correctTwoNodes = true;


        bool usePool = false;
        int poolSize;
        int tempIx = 0;
        double[][] tempWeightPool = null;
        object threadWeightLock = new object();

        /// <summary>
        /// Send in the connections you would normally create, and then also the intended feature count for verification of the proper connections
        /// </summary>
        /// <param name="inputFeatureCount"></param>
        /// <param name="connections"></param>
        public SlimPerceptron(int inputFeatureCount, FastConnection[] connections)
        {
            //loop through our connections
            HashSet<int> sourceNodes = new HashSet<int>();
            HashSet<int> targetNodes = new HashSet<int>();

            SortedDictionary<int, SortedList<int, double>> nodesToIncomingWeights = new SortedDictionary<int, SortedList<int, double>>();
            Dictionary<int, int> sourceToIxMap = new Dictionary<int, int>();

            //how many outputs do we have
            for (int i=0; i < connections.Length; i++)
            {
                var conn = connections[i];
                //grab all source nodes
                if (!sourceNodes.Contains(conn._srcNeuronIdx))
                    sourceNodes.Add(conn._srcNeuronIdx);

                //grab all target nodes
                if (!targetNodes.Contains(conn._tgtNeuronIdx))
                {
                    targetNodes.Add(conn._tgtNeuronIdx);
                    nodesToIncomingWeights.Add(conn._tgtNeuronIdx, new SortedList<int, double>());
                }

                //add this indexed object, thanks
                nodesToIncomingWeights[conn._tgtNeuronIdx].Add(conn._srcNeuronIdx, conn._weight);
            }

            Debug.Assert(sourceNodes.Count == (inputFeatureCount + biasCount), "Number of source nodes == number of input featurs + 1 bias");

            for(int i=0; i < sourceNodes.Count; i++)
            {
                sourceToIxMap.Add(sourceNodes.ElementAt(i), i);
            }

            //sanity check debugging -- must be single layer perceptron
            foreach (var tg in targetNodes)
            {
                //NONE of our target nodes should be a source node
                Debug.Assert(!sourceNodes.Contains(tg), "Source nodes should not be output nodes (i.e. not a single layer perceptron)");
                Debug.Assert(nodesToIncomingWeights.ContainsKey(tg), "Outputs must be fully connected, no connections to output");
                Debug.Assert(nodesToIncomingWeights[tg].Count == (inputFeatureCount + biasCount), "Outputs must be fully connected, wrong input node count");
            }
            
            //now we know our output count
            outputCount = targetNodes.Count;

            Debug.Assert(nodesToIncomingWeights.Count == outputCount, "Total number of incoming node targets should equal number of outputs");

            //we have 1 bias node, remember that!
            inputCount = sourceNodes.Count - biasCount;

            fullInputCount = inputCount + biasCount;

            _weights = new double[outputCount][];
            
            int oCount =0;
            foreach(var incoming in nodesToIncomingWeights)
            {
                //here we go, let's create our array
                double[] weightArray = new double[fullInputCount];

                //copying the weights from connections for this output
                _weights[oCount] = weightArray;

                //now grab the incoming weights
                foreach (var ixPair in incoming.Value)
                {
                    //this is a weight
                    //what ix is the source
                    //see, we don't care what the source node ID is, we care about what is the index of the source node
                    int sourceIx = sourceToIxMap[ixPair.Key];

                    //set our weight array here 
                    //our source index is the source node it comes from -- which represents the weight of the connection to this output
                    weightArray[sourceIx] = ixPair.Value;
                }

                //now let's move to the next output
                oCount++;
            }

            if (usePool)
            {
                //how big is our pool?
                poolSize = (int)Math.Ceiling(2.0 * outputCount);

                tempWeightPool = new double[poolSize][];

                //let's create a temporary weight pool
                for (int i = 0; i < poolSize; i++)
                {
                    tempWeightPool[i] = new double[fullInputCount];
                }
            }
            //now we've created all the arrays for our weights, all done
        }

        public double[][] getAllWeights()
        {
            return _weights;
        }

        public double[] threadSafeGetWeightArray()
        {
            if (usePool)
            {
                double[] rWeights;
                lock (threadWeightLock)
                {
                    rWeights = tempWeightPool[tempIx];
                    tempIx = (tempIx + 1) % poolSize;
                }
                return rWeights;
            }
            else
                return new double[fullInputCount];
        }

        int lastChosenIx;
        double[][] lastCrossWeights;
        public double[][] LastDecisionWeights
        {
            get { return lastCrossWeights; }
        }
        public int LastChosenIx
        {
            get { return lastChosenIx; }
        }

        //give the inputs, and what the correct guess should be
        public double[][] ActivateAndCorrect(double[] inputs, int correctDigitIx, bool shouldCorrect, double learningRate, bool addInPlace, out bool correct)
        {
            //let's check activation for our perceptron
            double[] activatedArray = new double[outputCount];

            //first step is to compute the inputs X weights
            double[][] tempWeights = crossInputsAndWeights(inputs, _weights);
            
            //basically our decision making process is contained inside these temporary weights
            lastCrossWeights = tempWeights;
            
            //now we need to do a summation of each array for the activation
            double[] sumWeights = sumWeightArrays(tempWeights, true);

            //get our maximum output node
            int maxOutputIx = maxDigitIx(sumWeights);

            lastChosenIx = maxOutputIx;

            //now tell us, are we right?
            if (outputCount == 1)
            {
                //if we only have 1 output, then correctDigitIx = 0 or correctDigitIx = 1, those are our only options
                correct = (sumWeights[0] > .5) == (correctDigitIx == 1);
            }
            else
                correct = maxOutputIx == correctDigitIx;

            //oops, we're wrong, time to fix our error
            //only do this if you need to fix an error
            if (shouldCorrect && !correct)
            {
                //nope! wrong. 
                //now we have to fix what we have
                double[] error = new double[outputCount];
                
                //how to calculate error? 
                //(proper - guess)*learning rate
                
                //only correct the max, and the weight of the wrong node
                int[] punishNodes;
                if (correctTwoNodes) //both the max output is wrong AND the digit that is supposed to be correct
                    punishNodes = new int[] { maxOutputIx, correctDigitIx };
                else //otherwise, we gather up all the nodes that are inappropriately higher than our max guess -- and knock those down
                    punishNodes = allAboveAndIncludingMax(sumWeights, sumWeights[maxOutputIx], maxOutputIx, correctDigitIx);


                //only punish the nodes found inside punish nodes
                for (int i = 0; i < punishNodes.Length; i++)
                {
                    //get the true node index
                    int ix = punishNodes[i];
                    //were we the right node?
                    double proper = (correctDigitIx == ix) ? 1.0 : -1.0;

                    //the error is set for that particular node, == (proper - guess) * learning
                    error[ix] = (proper - sumWeights[ix]) * learningRate;
                }

                //now multiply error inside our temporary array
                double[][] adjustments = createInputsTimesError(inputs, error, outputCount);
                //multiplyInPlace(error, tempWeights);

                //this is our error to add
                if (!addInPlace)
                    return adjustments;

                //otherwise, we keep going, add the error terms in place
                addWeightsInPlace(_weights, adjustments);
            }
            //if nothing to correct or if we add in place, we simply return null
            return null;
        }

        /// <summary>
        /// these are the activation nodes that are all above the wrong max value -- these need to be adjusted downwards
        /// </summary>
        /// <param name="sums"></param>
        /// <param name="wrongMax"></param>
        /// <param name="maxIx"></param>
        /// <returns></returns>
        public int[] allAboveAndIncludingMax(double[] sums, double wrongMax, int maxIx, int correctIx)
        {
            List<int> above = new List<int>();
            //by default include our max
            above.Add(maxIx);
            //we also include the correct, because it was wrong as well
            above.Add(correctIx);

            for(int i=0; i < sums.Length; i++)
            {
                if (i != maxIx && sums[i] > wrongMax)
                    above.Add(i);
            }

            //sort by index
            above.Sort();

            //send back everything above
            return above.ToArray();
        }
        
        public int maxDigitIx(double[] darr)
        {
            double max = double.MinValue;
            int chosenIx = -1;
            int len = darr.Length;
            for (int i = 0; i < len; i++)
            {
                if (darr[i] > max)
                {
                    max = darr[i];
                    chosenIx = i;
                }
            }

            return chosenIx;
        }

        #region Activation Functions [Steepened Sig]

        static double BipolarSteepenedSigmoid(double x)
        {
            return (2.0 / (1.0 + Math.Exp(-4.9 * x))) - 1.0;
        }

        static double BipolarSigmoid(double x)
        {
            return (2.0 / (1.0 + Math.Exp(-x))) - 1.0;
        }

        #endregion

        public void addWeightsInPlace(double[][] original, double[][] add)
        {
            Parallel.For(0, original.Length, po, i =>
            {
                //don't make the add if it's nulled out
                //that means we had zero error, and don't need to issue a correction
                if (add[i] != null)
                {
                    //grab our original, and the additional
                    double[] warr = original[i];
                    double[] addarr = add[i];

                    //add entire array in place
                    for (int w = 0; w < warr.Length; w++)
                        warr[w] += addarr[w];
                }
            });
        }

        public void multiplyInPlace(double[] mult, double[][] original)
        {
            Parallel.For(0, original.Length, po, i =>
            {
                double[] warr = original[i];
                double multAmount = mult[i];
                //multiply entire array in place
                for (int w = 0; w < warr.Length; w++)
                    warr[w] *= multAmount;
            });
        }

        public double[][] createInputsTimesError(double[] inputs, double[] error, int outputs)
        {
            double[][] errorAmount = new double[outputs][];

            Parallel.For(0, outputs, po, i =>
            {
                //we can skip our calculation if we had 0 error
                if (error[i] != 0.0)
                {
                    //get a new array to hold these objects
                    double[] summed = threadSafeGetWeightArray();
                    errorAmount[i] = summed;
                    double errAmount = error[i];

                    //multiply and store the inputs X the error term
                    for (int w = 0; w < inputs.Length; w++)
                        summed[w] = inputs[w] * errAmount;
                }
            });

            return errorAmount;
        }

        //sum weights in parallel -- apply activation -- 
        //since we decide activation by max, we don't need to apply activation by default
        //however, for fixing error, we need to apply -- so we have true by default
        public double[] sumWeightArrays(double[][] weightArrays, bool applyActivation = true)
        {
            double[] summed = new double[weightArrays.Length];

            Parallel.For(0, summed.Length, po, i =>
            {
                double[] warr = weightArrays[i];
                double c =0;
                for (int w = 0; w < warr.Length; w++)
                    c += warr[w];

                summed[i] = applyActivation ? BipolarSteepenedSigmoid(c) : c;
            });

            return summed;
        }

        public double[][] crossInputsAndWeights(double[] inputs, double[][] weightArrays)
        {
            double[][] multipliedWeights = new double[weightArrays.Length][];

            Parallel.For(0, weightArrays.Length, po, i =>
            {
                multipliedWeights[i] = crossArrays(inputs, weightArrays[i]);
            });

            return multipliedWeights;
        }

        public double[] crossArrays(double[] inputs, double[] weightArray)
        {
            if (inputs.Length != weightArray.Length)
                throw new Exception("Crossing arrays implies they have the same number of elements");

            //let's go across the inputs, and simply multiply into a new array
            //we don't allocate new memory in threads, so we just get the latest free weight array and copy over
            double[] tempWeights = threadSafeGetWeightArray();

            for (int i = 0; i < inputs.Length; i++)
            {
                tempWeights[i] = inputs[i] * weightArray[i];
            }

            return tempWeights;
        }





    }
}
