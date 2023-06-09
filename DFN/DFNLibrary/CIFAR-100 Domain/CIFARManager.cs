using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

using System.IO;
using System.Reflection;
using DFNLibrary;
using SharpNeat.Core;
using SharpNeat.Genomes.Neat;
using SharpNeat.Phenomes;
using SharpNeat.Decoders;
using DataLibrary;

namespace DFN.Evaluation
{
    public class CIFARManager : IDomain
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

        static int _pixelCount = 32*32*3;
        static int _totalImageCount;
        static int _totalTestImageCount;
        private static CIFARManager instance;

        //this is only ever called once -- this is a singleton
        private CIFARManager()
        {

        }

        #region IDomain members

        public void LoadImages(string folder = @"")
        {
            if (folder == "" || folder == null)
            {
                //no path supplied, we shall build our own to the first chunk
                string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                folder = currentAssemblyDirectoryName + "/../../../../data/CIFAR/";
            }

            allDigitsAndImages = CIFARDataLoader.loadPicturesAsDoubles(folder + "train.bin");
            allValidationDigitAndImages = CIFARDataLoader.loadPicturesAsDoubles(folder + "test.bin");

            if (DFNParams.CREATE_MNIST_VALIDATION)
            {
                throw new NotImplementedException();
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

                foreach (var image in picKeyVal.Value)
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
            return CIFARSubstrateDecoder.CreateGenomeDecoder(normalizeWeight, pixelWidth, pixelHeight, cppnSchem, substrateScheme, biasUsesLEO, useLEO);
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
            return CIFAREvaluator.EvaluateImages(testImages, pixelCount, box, minMaxDictionary, writeToDictionary, boxIx);
        }

        public IDomainEvaluator createEvaluator()
        {
            return new CIFAREvaluator();
        }

        public List<int> LeoIndexes(bool biasUsesLEO, bool useLEO)
        {
            return CIFARSubstrateDecoder.LeoIndexes(biasUsesLEO, useLEO);
        }

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
            return PixelWidth * PixelHeight * 3 + 1;
        }

        public int getCPPNInputs()
        {
            return 2;
        }

        public int getCPPNOutputs(bool biasUsesLEO, bool useLEO)
        {
            if (DFNParams.HiddenNodeSubstrateWidth > 0)
                throw new NotImplementedException();
            else
            {
                int outputs = 4;

                // use a single LEO output for each of the 3 input planes
                if (useLEO)
                    outputs++;

                //and if our bias uses LEO, we need a bias LEO node (i.e. 2 bias output nodes in our CPPN)
                if (biasUsesLEO)
                    outputs++;

                return outputs;
            }
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
            List<int> digitList = new List<int>();
            digitList.Add(0);
            return digitList;
        }


        #endregion





        public int PixelCount
        {
            get { return _pixelCount; }
        }
        //assumed square images here
        public int PixelWidth
        {
            get { return (int)Math.Sqrt(_pixelCount / 3); }
        }
        public int PixelHeight
        {
            get { return (int)Math.Sqrt(_pixelCount / 3); }
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

        public static CIFARManager Instance
        {
            get
            {
                if (instance == null)
                {
                    instance = new CIFARManager();
                }
                return instance;
            }
        }
    }
}
