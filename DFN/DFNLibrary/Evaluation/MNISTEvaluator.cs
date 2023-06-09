using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Core;
using SharpNeat.Phenomes;
using System.Diagnostics;
using DFNLibrary;
using SharpNeat.Phenomes.NeuralNets;
using SharpNeat.Network;

namespace DFN.Evaluation
{
    /// <summary>
    /// A black box evaluator for the XOR logic gate problem domain. 
    /// 
    /// XOR (also known as Exclusive OR) is a type of logical disjunction on two operands that results in
    /// a value of true if and only if exactly one of the operands has a value of 'true'. A simple way 
    /// to state this is 'one or the other but not both.'.
    /// 
    /// This evaulator therefore requires that the black box to be evaluated has has two inputs and one 
    /// output all using the range 0..1
    /// 
    /// In turn each of the four possible test cases are applied to the two inputs, the network is activated
    /// and the output is evaulated. If a 'false' response is requried we expect an output of zero, for true
    /// we expect a 1.0. Fitness for each test case is the difference between the output and the wrong output, 
    /// thus a maximum of 1 can be scored on each test case giving a maximum of 4. In addition each outputs is
    /// compared against a threshold of 0.5, if all four outputs are on teh correct side of the threshold then
    /// 10.0 is added to the total fitness. Therefore a black box that answers correctly but very close to the
    /// threshold will score just above 10, and a black box that answers correctly with perfect 0.0 and 1.0 
    /// answers will score a maximum of 14.0.
    /// 
    /// The first type of evaulation punishes for difference from the required outputs and therefore represents
    /// a smooth fitness space (we can evolve gradually towards better scores). The +10 score for 4 correct
    /// responses is 'all or nothing', in other words it is a fitness space with a large step and no indication
    /// of where the step is, which on it's own would be a poor fitness space as it required evolution to stumble
    /// on the correct network by random rather than ascending a gradient in the fitness space. If however we do 
    /// stumble on a black box that answers correctly but close to the threshold, then we would like that box to 
    /// obtain a higher score than a network with, say, 3 strong correct responses and but wrong overall. We can
    /// improve the correct box's output difference from threshold value gradually, while the box with 3 correct
    /// responses may actually be in the wrong area of the fitness space alltogether - in the wrong 'ballpark'.
    /// </summary>
    public class MNISTEvaluator : IPhenomeEvaluator<IBlackBox>
    {
        const double StopFitness = 10.0;
        ulong _evalCount;
        bool _stopConditionSatisfied;

        #region IPhenomeEvaluator<IBlackBox> Members

        /// <summary>
        /// Gets the total number of evaluations that have been performed.
        /// </summary>
        public ulong EvaluationCount
        {
            get { return _evalCount; }
        }

        /// <summary>
        /// Gets a value indicating whether some goal fitness has been achieved and that
        /// the the evolutionary algorithm/search should stop. This property's value can remain false
        /// to allow the algorithm to run indefinitely.
        /// </summary>
        public bool StopConditionSatisfied
        {
            get { return _stopConditionSatisfied; }
        }

        /// <summary>
        /// Evaluate the provided IBlackBox against the MNIST problem domain and return its fitness/novlety score.
        /// </summary>
        public FitnessInfo Evaluate(IBlackBox box)
        {
            //just keep track of evals
            _evalCount++;

            return StaticEvaluate(box);
        }

        public static FitnessInfo StaticEvaluate(IBlackBox box)
        {
            MNISTManager mm = MNISTManager.Instance;

            //grab the pre-constructed 2D array of images and their pixel values
            double[][] testImages = mm.AllTestImages;

            //how many pixels per image?
            return EvaluateImages(testImages, mm.PixelCount, box, null, false, -1);
        }

        //clamps a value between 0, and 1 regardless of min val and range
        public static double ClampNormalize(double val, double minVal, double range)
        {
            return Math.Min(1.0, Math.Max(0.0, ((val - minVal) / range)));
        }

        /// <summary>
        /// Evaluate the provided IBlackBox against the MNIST problem domain and return its fitness/novelty score.
        /// </summary>
        public static FitnessInfo EvaluateImages(
            double[][] testImages, 
            int pixelCount, 
            IBlackBox box, 
            Dictionary<int, Tuple<double,double>> minMaxDictionary, 
            bool writeToDictionary,
            int boxIx
            )
        {
            //these are our inputs and outputs
            ISignalArray inputArr = box.InputSignalArray;
            ISignalArray outputArr = box.OutputSignalArray;

            //how many total images to check -- all test images
            int totalImageCount = testImages.Length;

            float[] bbBehavior = new float[totalImageCount];
            int bIx = 0;

            double min = double.MaxValue;
            double max = double.MinValue;
            double sum = 0;

            if (box is SingleOutputNetwork)
            {
                //listen here son, we need to do this as simply as possible
                SingleOutputNetwork son = box as SingleOutputNetwork;

                //we need to be absolutely certain its the same image order every time 
                for (int i = 0; i < totalImageCount; i++)
                {
                    double val = son.ActivateDotProduct(testImages[i]);
                    min = Math.Min(val, min);
                    max = Math.Max(val, max);
                    sum += val;
                    //grab the image -- do a direct multiplication from the inputs to create a sum and apply an act function
                    //grab our image behavior and add it to the full list
                    bbBehavior[bIx++] = (float)val;
                }
            }
            else if (box is SingleOutputNetworkWithHidden)
            {
                //listen here son, we need to do this as simply as possible
                SingleOutputNetworkWithHidden son = box as SingleOutputNetworkWithHidden;

                //we need to be absolutely certain its the same image order every time 
                for (int i = 0; i < totalImageCount; i++)
                {
                    double val = son.ActivateDotProduct(testImages[i]);
                    min = Math.Min(val, min);
                    max = Math.Max(val, max);
                    sum += val;
                    //grab the image -- do a direct multiplication from the inputs to create a sum and apply an act function
                    //grab our image behavior and add it to the full list
                    bbBehavior[bIx++] = (float)val;
                }
            }
            else
            {

                //we need to be absolutely certain its the same image order every time 
                for (int i = 0; i < totalImageCount; i++)
                {
                    //grab the image -- copy into inputs
                    inputArr.CopyFrom(testImages[i], 0);

                    //now we activate our blackbox and pull the result from the outputs
                    box.ResetState();
                    box.Activate();

                    //val
                    double val = outputArr[0];
                    min = Math.Min(val, min);
                    max = Math.Max(val, max);
                    sum += val;

                    //grab our image behavior and add it to the full list
                    bbBehavior[bIx++] = (float)val;
                }
            }

            //this only happens during caching behavior
            if(minMaxDictionary != null)
            {
                if (writeToDictionary)
                {
                    lock (minMaxDictionary)
                    {
                        //if we've seen this key before, we need to update the min/max for the whole behavior range
                        if(minMaxDictionary.ContainsKey(boxIx))
                        {
                            var otup = minMaxDictionary[boxIx];
                            minMaxDictionary[boxIx] = new Tuple<double, double>(Math.Min(min, otup.Item1), Math.Max(max, otup.Item2));
                        }
                        else
                            minMaxDictionary.Add(boxIx, new Tuple<double, double>(min, max));
                    }
                }
                else
                {
                    var minMax = minMaxDictionary[boxIx];
                    min = minMax.Item1;
                    max = minMax.Item2;
                }
            }

            //fitness doesn't matter at all in this experiment
            double fitness = 1.0;
            
            if(DFNParams.NORMALIZE_BEHAVIOR_VECTOR_IN_EVALUATE)
            {
                double range = max - min;
                double avg = sum / totalImageCount;
                
                //if max == min, then all behaviors are the same
                //therefore behavior[i] - min == 0 -- so the whole thing zeroes out anyways
                if(range == 0.0)
                    range = 1;

                double variance = 0;
                double temp;
                //we normalize by subtracting the min and adding the range 
                for (int i = 0; i < bbBehavior.Length; i++)
                {
                    temp = bbBehavior[i] - avg;
                    variance += temp * temp;
                }

                variance = Math.Sqrt(variance);

                //if you have 0 variance, then you are the same number -- so minus avg == 0
                if (variance == 0)
                    variance = 1;

                for (int i = 0; i < bbBehavior.Length; i++)
                {
                    temp = (bbBehavior[i] - avg);
                    //z-score normalization -- subtract the avg, normalize it
                    bbBehavior[i] = (float)(temp / variance);
                }
            }

            //either one of these requires collection binary acitvation for every behavior
            if (DFNParams.USE_BINARY_SET_DISTANCE || DFNParams.SET_OVERLAP_FEATURE_COLLECTION)
            {
                bool[] binaryActivation = new bool[bbBehavior.Length];

                for (int i = 0; i < binaryActivation.Length; i++)
                    binaryActivation[i] = bbBehavior[i] > .5f;


                //send back our behavior for these objects
                return new FitnessInfo(fitness, bbBehavior, fitness, null, binaryActivation);
            }

            if (DFNParams.USEUNIQUENESSBEHAVIOR)
            {
                float[] secondBehavior = null;
                var openNetwork = box as OpenAcyclicNetwork;
                if (openNetwork != null)
                {
                    var connections = openNetwork.ConnectionList;

                    //source can be any of the pixels + the bias node
                    secondBehavior = new float[pixelCount + 1];

                    //you can have a connection from each pixel
                    for (var i = 0; i < connections.Length; i++)
                    {
                        var c = connections[i];
                        var src = connections[i]._srcNeuronIdx;

                        secondBehavior[src] = (float)c._weight;
                    }                    
                }

                  //send back our behavior for these objects
                return new FitnessInfo(fitness, bbBehavior, fitness, secondBehavior);
            }

            //send back our behavior for these objects
            return new FitnessInfo(fitness, bbBehavior, fitness);
        }

        /// <summary>
        /// Reset the internal state of the evaluation scheme if any exists.
        /// Note. The XOR problem domain has no internal state. This method does nothing.
        /// </summary>
        public void Reset()
        {

        }

        #endregion
    }
}
