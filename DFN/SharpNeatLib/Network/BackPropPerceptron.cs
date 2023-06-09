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
    public class BackPropPerceptron 
    {
        //always 1 bias please
        int biasCount = 1;
        int inputCount;
        int fullInputCount;
        int outputCount;
        public double[][] _weights = null;
        double[][] _previousDeltas = null;
        public SingleOutputNetwork[] allFeatures;
        public double[] lastOutputs;

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
        public BackPropPerceptron(SingleOutputNetwork[] featureNetworks, int outputCount, double startingWeightMultiplier)
        {
            usePool = false;
            inputCount = featureNetworks.Length;
            //add the bias
            fullInputCount = inputCount + biasCount;

            //grab output count -- how many outputs to construct
            this.outputCount = outputCount;

            //random initial weight
            Random r = new Random();
            
            //each output is fully connected
            _weights = new double[outputCount][];
            _previousDeltas = new double[outputCount][];

            //each weight going in == starting weight
            for (int i = 0; i < outputCount; i++)
            {
                _weights[i] = new double[fullInputCount];
                _previousDeltas[i] = new double[fullInputCount];
                for (int w = 0; w < fullInputCount; w++)
                    _weights[i][w] = 2 * (startingWeightMultiplier * r.NextDouble()) - startingWeightMultiplier;
            }

            allFeatures = featureNetworks;

            //now we've created all the arrays for our weights, all done
        }

        public BackPropPerceptron(SingleOutputNetwork[] featureNetworks, int outputCount, double[][] startingWeights)
        {
            usePool = false;
            inputCount = featureNetworks.Length;
            //add the bias
            fullInputCount = inputCount + biasCount;

            //grab output count -- how many outputs to construct
            this.outputCount = outputCount;

            //random initial weight
            Random r = new Random();

            //each output is fully connected
            _weights = new double[outputCount][];
            _previousDeltas = new double[outputCount][];

            //each weight going in == starting weight
            for (int i = 0; i < outputCount; i++)
            {
                _weights[i] = new double[fullInputCount];
                _previousDeltas[i] = new double[fullInputCount];
                for (int w = 0; w < fullInputCount; w++)
                    _weights[i][w] = startingWeights[i][w];
            }

            allFeatures = featureNetworks;

            //now we've created all the arrays for our weights, all done
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

        //send in the image, we'll handle creating the inputs to our network, then correcting them
        public bool ActivateFullNetwork(double[] imageWithoutPadding, int correctDigitIx, 
            bool shouldCorrect, double learningRate, double secondLayerLearningRate, double momentum, bool biasIsZero = false)
        {
            bool correct = false;

            //include bias in number of inputs
            double[] featureInputs = new double[fullInputCount];
            //establish the bias, thanks
                        
            featureInputs[0] = 1.0;

            //we must activate all our features
            for(int i = 0; i < inputCount; i++)            
            {
                //we need to cativate each feature
                allFeatures[i].ActivateDotProduct(imageWithoutPadding, biasIsZero);

                //set feature input == activation using image
                featureInputs[i + 1] = allFeatures[i].LastActivation;
            }

            //now we must make corrections to our features AND to our perceptron weight

            //let's figure out our error from the bottom layer (and apply that error)
            double[] errorForFeatures = ActivateCorrectAndGetError(featureInputs, correctDigitIx, shouldCorrect, learningRate, momentum, out correct);

            if (shouldCorrect)
            {
                //now we must calculate the error for our features
                for (int i = 0; i < inputCount; i++)
                {
                    //for each feature, we must send the error, and it will need to do corrections accordingly
                    allFeatures[i].PropogateError(imageWithoutPadding, errorForFeatures[i + 1], secondLayerLearningRate, momentum);
                }
            }

            return correct;
        }

        #region bad multithreaded methods
        // version for multithreaded backprop algorithm.  do not update the network but return deltas for each weight to be applied later
        public bool ActivateFullNetworkMT(double[] imageWithoutPadding, int correctDigitIx, bool shouldCorrect, double learningRate, double secondLayerLearningRate, double[][] topLevelDeltas, double[][] featureDeltas)
        {
            bool correct = false;

            //include bias in number of inputs
            double[] featureInputs = new double[fullInputCount];
            //establish the bias, thanks
            featureInputs[0] = 1.0;

            //we must activate all our features
            for (int i = 0; i < inputCount; i++)   
            {
                //we need to cativate each feature
                allFeatures[i].ActivateDotProduct(imageWithoutPadding);

                //set feature input == activation using image
                featureInputs[i + 1] = allFeatures[i].LastActivation;
            }

            //now we must make corrections to our features AND to our perceptron weight

            //let's figure out our error from the bottom layer (and apply that error)
            double[] errorForFeatures = ActivateCorrectAndGetErrorMT(featureInputs, correctDigitIx, shouldCorrect, learningRate, out correct, topLevelDeltas);

            //now we must calculate the error for our features
            for (int i = 0; i < inputCount; i++)
            {
                //for each feature, we must send the error, and it will need to do corrections accordingly
                featureDeltas[i] = allFeatures[i].getErrorDeltas(imageWithoutPadding, errorForFeatures[i + 1], secondLayerLearningRate);
            }

            return correct;
        }

        // version for multithreaded backprop algorithm.  do not update the network but return deltas for each weight to be applied later
        double[] ActivateCorrectAndGetErrorMT(double[] inputs, int correctDigitIx, bool shouldCorrect, double learningRate, out bool correct, double[][] topLevelDeltas)
        {
            bool addInPlace = true;

            double[] featureErrors = new double[inputs.Length];

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

            //the correct error to propogate
            double[] totalFeatureError = new double[fullInputCount];

            if (shouldCorrect)
            {
                //let's calculate our error
                double[] outputErrors = new double[outputCount];

                for (int i = 0; i < outputCount; i++)
                {
                    double target = (i == correctDigitIx ? 1.0 : 0.0);

                    //convert to [0,1] range so everything is sigmoid activation
                    double o = (sumWeights[i] + 1.0) / 2.0;

                    //our error = 
                    //Sigmoid: 
                    //output X (1 - output) X (target - output)
                    outputErrors[i] = o * (1 - o) * (target - o);
                }


                double[][] featureError = new double[outputCount][];
                
                //we must propogate all error backwards among our connections
                for (int i = 0; i < outputCount; i++)
                {
                    //grab our proper weights to propogate
                    double propError = outputErrors[i];
                    double[] weights = _weights[i];

                    //double[] previousDeltas = _previousDeltas[i];
                    double[] fErrorForOutput = new double[fullInputCount];

                    double[] deltas = new double[fullInputCount];

                    //now propogate based on feature inputs
                    for (int w = 0; w < fullInputCount; w++)
                    {
                        double inputToWeight = inputs[w];

                        double oWeight = weights[w];
                        double delta = learningRate * propError * inputToWeight;

                        deltas[w] = delta;

                        //calculate the error caused by this weight for this output node and feature combination (later to be summed)
                        fErrorForOutput[w] = propError * oWeight;

                        //oWeight += delta;
                        //oWeight += momentum * previousDeltas[w];
                        //previousDeltas[w] = delta;

                        //weight change = 
                        //w_new = w_old + learningRate x error x inActivation
                        //weights[w] = oWeight;
                    }

                    //set this as our feature error -- will sum shortly
                    featureError[i] = fErrorForOutput;

                    topLevelDeltas[i] = deltas;
                }

                for (int i = 0; i < fullInputCount; i++)                   
                {
                    double featureInputErrorSum = 0;
                    //sum across the double array (probably slow, unfortunately)
                    for (int o = 0; o < outputCount; o++)
                        featureInputErrorSum += featureError[o][i];

                    double featureIn = inputs[i];

                    //DFN: Assumption -- using sigmoid as activation 
                    totalFeatureError[i] = featureIn * (1 - featureIn) * featureInputErrorSum;
                }
            }

            //now we have propogated all the weight changes backwards, we must update the features as well
            //send the feature error info backwards, thank you!
            return totalFeatureError;
        }

        public void applyDeltas(double[][] topLevelDeltas, double[][] featureDeltas)
        {
            // update the top level connections
            for (int i = 0; i < outputCount; i++)
                for (int j = 0; j < fullInputCount; j++)
                    _weights[i][j] += topLevelDeltas[i][j];

            // update the feature connections
            for (int i = 0; i < allFeatures.Length; i++)
                allFeatures[i].updateFeatureFromDeltas(featureDeltas[i]);
        }
        #endregion

        //give the inputs, and what the correct guess should be
        double[] ActivateCorrectAndGetError(double[] inputs, int correctDigitIx, bool shouldCorrect, double learningRate, double momentum, out bool correct)
        {
            bool addInPlace = true;

            double[] featureErrors = new double[inputs.Length];

            //let's check activation for our perceptron
            double[] activatedArray = new double[outputCount];

            //first step is to compute the inputs X weights
            double[][] tempWeights = crossInputsAndWeights(inputs, _weights);
            
            //basically our decision making process is contained inside these temporary weights
            lastCrossWeights = tempWeights;
            
            //now we need to do a summation of each array for the activation
            double[] sumWeights = sumWeightArrays(tempWeights, true);
            lastOutputs = sumWeights;

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

            //the correct error to propogate
            double[] totalFeatureError = new double[fullInputCount];

            if (shouldCorrect)
            {
                //let's calculate our error
                double[] outputErrors = new double[outputCount];

                for (int i = 0; i < outputCount; i++)
                {
                    double target = (i == correctDigitIx ? 1.0 : 0.0);

                    //convert to [0,1] range so everything is sigmoid activation
                    double o = (sumWeights[i] + 1.0) / 2.0;

                    //our error = 
                    //Sigmoid: 
                    //output X (1 - output) X (target - output)
                    outputErrors[i] = o * (1 - o) * (target - o);
                }


                double[][] featureError = new double[outputCount][];


                //we must propogate all error backwards among our connections
                for (int i = 0; i < outputCount; i++)                   
                {
                    //grab our proper weights to propogate
                    double propError = outputErrors[i];
                    double[] weights = _weights[i];

                    double[] previousDeltas = _previousDeltas[i];
                    double[] fErrorForOutput = new double[fullInputCount];

                    //now propogate based on feature inputs
                    for (int w = 0; w < fullInputCount; w++)
                    {
                        double inputToWeight = inputs[w];

                        double oWeight = weights[w];
                        double delta = learningRate * propError * inputToWeight;

                        //calculate the error caused by this weight for this output node and feature combination (later to be summed)
                        fErrorForOutput[w] = propError * oWeight;

                        oWeight += delta;
                        oWeight += momentum * previousDeltas[w];
                        previousDeltas[w] = delta;

                        //weight change = 
                        //w_new = w_old + learningRate x error x inActivation
                        weights[w] = oWeight;
                    }

                    //set this as our feature error -- will sum shortly
                    featureError[i] = fErrorForOutput;
                }

                for (int i = 0; i < fullInputCount; i++)
                {
                    double featureInputErrorSum = 0;
                    //sum across the double array (probably slow, unfortunately)
                    for (int o = 0; o < outputCount; o++)
                        featureInputErrorSum += featureError[o][i];

                    double featureIn = inputs[i];

                    //DFN: Assumption -- using sigmoid as activation 
                    totalFeatureError[i] = featureIn * (1 - featureIn) * featureInputErrorSum;
                }
            }

            //now we have propogated all the weight changes backwards, we must update the features as well
            //send the feature error info backwards, thank you!
            return totalFeatureError;
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
            for (int i = 0; i < original.Length; i++)               
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
            }
        }

        public void multiplyInPlace(double[] mult, double[][] original)
        {
            for (int i = 0; i < original.Length; i++)               
            {
                double[] warr = original[i];
                double multAmount = mult[i];
                //multiply entire array in place
                for (int w = 0; w < warr.Length; w++)
                    warr[w] *= multAmount;
            }
        }

        public double[][] createInputsTimesError(double[] inputs, double[] error, int outputs)
        {
            double[][] errorAmount = new double[outputs][];

            for (int i = 0; i < outputs; i++)               
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
            }

            return errorAmount;
        }

        //sum weights in parallel -- apply activation -- 
        //since we decide activation by max, we don't need to apply activation by default
        //however, for fixing error, we need to apply -- so we have true by default
        public double[] sumWeightArrays(double[][] weightArrays, bool applyActivation = true)
        {
            double[] summed = new double[weightArrays.Length];

            for (int i = 0; i < summed.Length; i++)               
            {
                double[] warr = weightArrays[i];
                double c =0;
                for (int w = 0; w < warr.Length; w++)
                    c += warr[w];

                summed[i] = applyActivation ? BipolarSigmoid(c) : c;
            }

            return summed;
        }

        public double[][] crossInputsAndWeights(double[] inputs, double[][] weightArrays)
        {
            double[][] multipliedWeights = new double[weightArrays.Length][];

            for (int i = 0; i < weightArrays.Length; i++)               
            {
                multipliedWeights[i] = crossArrays(inputs, weightArrays[i]);
            }

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


        public void mergeNetworks(BackPropPerceptron[] networks)
        {            
            for (int i = 0; i < networks.Length; i++)
            {
                for (int j = 0; j < _weights.Length; j++)
                    for (int k = 0; k < _weights[j].Length; k++)
                    {
                        _weights[j][k] += networks[i]._weights[j][k];
                        _previousDeltas[j][k] += networks[i]._previousDeltas[j][k]; // _weights and _previousDeltas always of same size, yeah?
                    }

                for (int j = 0; j < allFeatures.Length; j++)                
                    for (int k = 0; k < allFeatures[j].Weights.Length; k++)
                        allFeatures[j].Weights[k] += networks[i].allFeatures[j].Weights[k];                
            }

            for (int j = 0; j < _weights.Length; j++)
                for (int k = 0; k < _weights[j].Length; k++)
                {
                    _weights[j][k] /= networks.Length + 1;
                    _previousDeltas[j][k] /= networks.Length + 1;
                }

            for (int j = 0; j < allFeatures.Length; j++)
                for (int k = 0; k < allFeatures[j].Weights.Length; k++)
                    allFeatures[j].Weights[k] /= networks.Length + 1;
        }

        public void copyNetwork(BackPropPerceptron[] networks)
        {
            for (int i = 0; i < networks.Length; i++)
            {               
                SingleOutputNetwork[] newFeatures = new SingleOutputNetwork[allFeatures.Length];
                double[][] newWeights = new double[this._weights.Length][];
                for (int j = 0; j < this._weights.Length; j++)
                {
                    newWeights[j] = this._weights[j].ToArray();
                }                

                for (int j = 0; j < newFeatures.Length; j++)
                {
                    double[] newSONWeights = allFeatures[j].Weights.ToArray();
                    newFeatures[j] = new SingleOutputNetwork(newSONWeights, allFeatures[j].ActFun, allFeatures[j].AllowBackpropWeights);
                }
                networks[i] = new BackPropPerceptron(newFeatures, outputCount, newWeights);
            }
        }
    }
}
