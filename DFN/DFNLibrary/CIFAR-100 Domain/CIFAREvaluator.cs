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
    public class CIFAREvaluator : IDomainEvaluator
    {
        ulong _evalCount;
        bool _stopConditionSatisfied;

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

        public double[][] ReloadImages()
        {
            CIFARManager mm = CIFARManager.Instance;
            mm.LoadImages();
            return mm.AllTestImages;
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
            CIFARManager mm = CIFARManager.Instance;

            //grab the pre-constructed 2D array of images and their pixel values
            double[][] testImages = mm.AllTestImages;

            //how many pixels per image?
            return EvaluateImages(testImages, mm.PixelCount, box, null, false, -1);
        }

        /// <summary>
        /// Reset the internal state of the evaluation scheme if any exists.
        /// Note. The XOR problem domain has no internal state. This method does nothing.
        /// </summary>
        public void Reset()
        {

        }

        /// <summary>
        /// Evaluate the provided IBlackBox against the MNIST problem domain and return its fitness/novelty score.
        /// </summary>
        public static FitnessInfo EvaluateImages(
            double[][] testImages,
            int pixelCount,
            IBlackBox box,
            Dictionary<int, Tuple<double, double>> minMaxDictionary,
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
            double[] bbBehaviorD = new double[totalImageCount];
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
                    bbBehavior[bIx] = (float)val;
                    bbBehaviorD[bIx++] = val;
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
                    bbBehavior[bIx] = (float)val;
                    bbBehaviorD[bIx++] = val;
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
                    bbBehavior[bIx] = (float)val;
                    bbBehaviorD[bIx++] = val;
                }
            }

            //this only happens during caching behavior
            if (minMaxDictionary != null)
            {
                if (writeToDictionary)
                {
                    lock (minMaxDictionary)
                    {
                        //if we've seen this key before, we need to update the min/max for the whole behavior range
                        if (minMaxDictionary.ContainsKey(boxIx))
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

            //send back our behavior for these objects
            return new FitnessInfo(fitness, bbBehavior, fitness, null, null, bbBehaviorD);
        }
    }
}
