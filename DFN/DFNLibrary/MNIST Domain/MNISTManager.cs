using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using DataLibrary;
using System.IO;
using System.Reflection;
using DFNLibrary;
using SharpNeat.Core;
using SharpNeat.Genomes.Neat;
using SharpNeat.Phenomes;
using SharpNeat.Decoders;

namespace DFN.Evaluation
{
    //MNIST Manager is a singleton that handles all our images (only exists once in memory)
    public class MNISTManager : IDomain
    {
        //what digits are we itnerested in?
        static List<int> interestedDigits;

        //this is a sorted list of digits and their images
        static SortedList<int, List<double[]>> allDigitsAndImages;
        static SortedList<int, List<double[]>> allValidationDigitAndImages;

        //this is if we go in order
        static int[] testSetIdentifier;
        static int[] validationSetIdentifiers;

        static double[][] allTestImages;
        static double[][] allValidationImages;

        static int _pixelCount = 28*28;
        static int _totalImageCount;
        static int _totalTestImageCount;
        private static MNISTManager instance;

        //this is only ever called once -- this is a singleton
        private MNISTManager() {

        }


        #region IDomain members

        public int getPixelWidth()
        {
            return PixelWidth;
        }

        public int getPixelHeight()
        {
            return PixelHeight;
        }

        public int getTotalInputs()
        {
            return PixelWidth * PixelHeight + 1;
        }

        public void LoadImages(string jsonFolder = @"")
        {
            if (jsonFolder == "" || jsonFolder == null)
            {
                //no path supplied, we shall build our own to the first chunk
                string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                jsonFolder = currentAssemblyDirectoryName + "/../../../../data/MNIST/";                
            }

            //we load in our pictures from the helper library (it knows how to deal with the parsed data)                        
            allDigitsAndImages = MNISTDataLoader.loadPicturesAsDoubles(jsonFolder + "train.json");
            allValidationDigitAndImages = MNISTDataLoader.loadPicturesAsDoubles(jsonFolder + "test.json");            

            if(DFNParams.CREATE_MNIST_VALIDATION)
            {
                //we need to replace alldigitsanddimages/allvalidation with true validation set and smaller training

                int validationSetSize = 10000;

                //how many we need to take for each digit
                int randomDigitSamples = (int)Math.Ceiling((double)validationSetSize/allDigitsAndImages.Count);

                List<List<int>> randomChosenIxs = new List<List<int>>();
                Random r = new Random();


                SortedList<int, List<double[]>> remainingImages = new SortedList<int, List<double[]>>();
                SortedList<int, List<double[]>> newValidation = new SortedList<int, List<double[]>>();

                foreach (var picKeyVal in allDigitsAndImages)
                {
                    int digit = picKeyVal.Key;

                    List<int> randomOrder = new List<int>();

                    List<double[]> valSelection = new List<double[]>();
                    List<double[]> remainder = new List<double[]>();

                    var allDImages = picKeyVal.Value;

                    for (int i = 0; i < picKeyVal.Value.Count; i++)
                        randomOrder.Add(i);

                    //randomly sort
                    randomOrder = randomOrder.OrderBy(a => r.Next()).ToList();

                    //take the first sampleCount -- grab the random ix
                    for (int w = 0; w < randomDigitSamples; w++){
                    
                        //rando image for validation
                        int randoIx = randomOrder[w];
                        double[] image = allDImages[randoIx];
                        
                        //add to our validation set
                        valSelection.Add(image);
                    }

                    //then the rest is our training set for this digit
                    for (int w = randomDigitSamples; w < allDImages.Count; w++)
                    {
                        //add the rest!
                        //rando image for validation
                        int remainIx = randomOrder[w];
                        double[] image = allDImages[remainIx];

                        remainder.Add(image);
                    }

                    //our training set
                    remainingImages.Add(digit, remainder);
                    //our validation set
                    newValidation.Add(digit, valSelection);
                }



                //simply replace our two sets -- loaded from training and not testing places :)
                allDigitsAndImages = remainingImages;
                allValidationDigitAndImages = newValidation;

            }


            interestedDigits = new List<int>();
            _totalImageCount = 0;
            _pixelCount = 0;

            //this will be in digit ordered list
            foreach (var picKeyVal in allDigitsAndImages)
            {
                int digit = picKeyVal.Key;

                //these are all the digits we're testing/validating!
                if (!interestedDigits.Contains(digit))
                    interestedDigits.Add(digit);

                //how many pixels in any of the images -- we only need it from the first digit
                //but this just checks all the digits -- it's all equal anyways
                _pixelCount = picKeyVal.Value[0].Length;

                //sum up all the images for every digit
                _totalImageCount += picKeyVal.Value.Count;
            }                        

            testSetIdentifier = new int[_totalImageCount];
            allTestImages = new double[_totalImageCount][];

            int imageIx = 0;

            foreach (var picKeyVal in allDigitsAndImages)
            {
                int digit = picKeyVal.Key;

                foreach(var image in picKeyVal.Value)
                {
                    testSetIdentifier[imageIx] = digit;
                    allTestImages[imageIx] = image;

                    //bump the index here
                    imageIx++;
                }
            }

            _totalTestImageCount = 0;
            //this will be in digit ordered list
            foreach (var picKeyVal in allValidationDigitAndImages)
            {
                int digit = picKeyVal.Key;

                //these are all the digits we're testing/validating!
                if (!interestedDigits.Contains(digit))
                    interestedDigits.Add(digit);

                //how many pixels in any of the images -- we only need it from the first digit
                //but this just checks all the digits -- it's all equal anyways

                //sum up all the images for every digit
                _totalTestImageCount += picKeyVal.Value.Count;
            }

            validationSetIdentifiers = new int[_totalTestImageCount];
            allValidationImages = new double[_totalTestImageCount][];

            imageIx = 0;

            foreach (var picKeyVal in allValidationDigitAndImages)
            {
                int digit = picKeyVal.Key;

                foreach (var image in picKeyVal.Value)
                {
                    validationSetIdentifiers[imageIx] = digit;
                    allValidationImages[imageIx] = image;

                    //bump the index here
                    imageIx++;
                }
            }
        }

        public IGenomeDecoder<NeatGenome, IBlackBox> CreateGenomeDecoder(bool normalizeWeight, int pixelWidth, int pixelHeight,
            NetworkActivationScheme cppnSchem, NetworkActivationScheme substrateScheme, bool biasUsesLEO, bool useLEO)
        {
            return MNISTSubstrateDecoder.CreateGenomeDecoder(normalizeWeight, pixelWidth, pixelHeight, cppnSchem, substrateScheme, biasUsesLEO, useLEO);
        }

        public FitnessInfo EvaluateImages(
            double[][] testImages,
            int pixelCount,
            IBlackBox box,
            Dictionary<int, Tuple<double, double>> minMaxDictionary,
            bool writeToDictionary,
            int boxIx
            )
        {
            return MNISTEvaluator.EvaluateImages(testImages, pixelCount, box, minMaxDictionary, writeToDictionary, boxIx);
        }

        public IDomainEvaluator createEvaluator()
        {
            return new MNISTEvaluator();
        }

        public List<int> LeoIndexes(bool biasUsesLEO, bool useLEO)
        {
            return MNISTSubstrateDecoder.LeoIndexes(biasUsesLEO, useLEO);
        }

        public int getCPPNInputs()
        {
            //x, y of the input pixel
            int inputs = 2;

            //doh! We actually need x,y -> x,y cause our input layer has geometry
            if (DFNParams.HiddenNodeSubstrateWidth > 0)
                inputs = 4;

            return inputs;
        }

        public int getCPPNOutputs(bool biasUsesLEO, bool useLEO)
        {
            //output determination 

            //we only have 1 singular output -- that is the weights from inputs to outputs (from visual field to singular output)
            //but we may or may not use LEO
            int outputs = 1;

            //we need more outputs if we're going to have hidden nodes 
            //-- we have 3, 1 for inputs -> outputs, 1 for inputs -> hidden, and the final for hidden -> outputs
            if (DFNLibrary.DFNParams.HiddenNodeSubstrateWidth > 0)
                outputs = 3;

            //if we useLEO then we multiply the number of outputs by 2! For every output we have (if we're doing multimodal), we need 1 node for LEO
            if (useLEO)
                outputs *= 2;

            //then we add 1 output for our bias weights
            outputs++;

            //and if our bias uses LEO, we need a bias LEO node (i.e. 2 bias output nodes in our CPPN)
            if (biasUsesLEO)
                outputs++;

            return outputs;
        }

        public List<int> getInterestedDigits()
        {
            return interestedDigits;
        }

        public double[][] getAllTestImages()
        {
            return allTestImages;
        }

        public double[][] getAllValidationImages()
        {
            return allValidationImages;
        }

        public int[] getAllTestLabels()
        {
            return testSetIdentifier;
        }

        public int[] getAllValidationLabels()
        {
            return AllValidationLabels;
        }

        public int getTotalTestImageCount()
        {
            return _totalImageCount;
        }

        public List<int> getDigitsForString()
        {
            return interestedDigits;
        }

        #endregion






        public int PixelCount
        {
            get { return _pixelCount; }
        }
        //assumed square images here
        public int PixelWidth
        {
            get { return (int)Math.Sqrt(_pixelCount); }
        }
        public int PixelHeight
        {
            get { return (int)Math.Sqrt(_pixelCount); }
        }
        public int TotalTestImageCount
        {
            get { return _totalImageCount; }
        }       

        //public SortedList<int, List<double[]>> AllImages
        //{
        //    get { return allDigitsAndImages; }
        //}

        #region Full Image and Label Arrays

        public List<int> InterestedDigits
        {
            get { return interestedDigits; }
        }
        public double[][] AllTestImages
        {
            get { return allTestImages; }
        }

        public int[] AllTestLabels
        {
            get { return testSetIdentifier; }
        }

        public double[][] AllValidationImages
        {
            get { return allValidationImages; }
        }

        public int[] AllValidationLabels
        {
            get { return validationSetIdentifiers; }
        }

        #endregion

        //public SortedList<int, List<double[]>> ValidationSet
        //{
        //    get { return allValidationDigitAndImages; }
        //}

        public static MNISTManager Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new MNISTManager();
                }
                return instance;
            }
        }
    }
}
