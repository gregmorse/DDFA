using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using MNISTDataLibrary;
using System.IO;
using System.Reflection;
using DFNLibrary;

namespace DFN.Evaluation
{
    //MNIST Manager is a singleton that handles all our images (only exists once in memory)
    public class MNISTManager
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

        static void ZScoreNormalize(SortedList<int, List<double[]>> imageHolder)
        {
            foreach (var digit in imageHolder)
            {
                foreach (var image in digit.Value)
                {
                    double avg = image.Average();
                    double variance = 0;

                    double temp;
                    for (int i = 0; i < image.Length; i++)
                    {
                        temp = image[i] - avg;
                        variance += temp * temp;
                    }

                    //impossible unless it's blank for some reason
                    //if(variance == 0)
                    //    variance = 1;

                    //z score normalization of the image inputs
                    for (int i = 0; i < image.Length; i++)
                    {
                        temp = image[i] - avg;
                        image[i] = temp / variance;
                    }

                }
            }
        }


        public void LoadImages(string jsonFolder = @"")
        {
            if (jsonFolder == "")
            {
                //no path supplied, we shall build our own to the first chunk
                string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                jsonFolder = currentAssemblyDirectoryName + "/../../../data/chunks/";
            }

            //we load in our pictures from the helper library (it knows how to deal with the parsed data)
            if (DFNParams.LOADMNISTASCHUNKS)
            {
                allDigitsAndImages = DataLoader.loadFullPicturesInPieces(DFNParams.PARSEDMNISTCHUNKCOUNT, "full", jsonFolder);
                allValidationDigitAndImages = DataLoader.loadFullPicturesInPieces(DFNParams.PARSEDMNISTCHUNKCOUNT, "test", jsonFolder);
            }
            else
            {
                allDigitsAndImages = DataLoader.loadPicturesAsDoubles(jsonFolder + "full.json");
                allValidationDigitAndImages = DataLoader.loadPicturesAsDoubles(jsonFolder + "test.json");
            }

            if (DFNParams.NORMALIZE_IMAGE_INPUTS)
            {
                //normalize the group of test and validation images, please, thank you
                ZScoreNormalize(allDigitsAndImages);
                ZScoreNormalize(allValidationDigitAndImages);
            }

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
                    randomOrder.OrderBy(x => r.NextDouble());

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

            
            if (DFNParams.TRAINING_SET_DIVIDER > 1)
            {
                //reduce the training size by our divider
                int trainingSetSize = (int)Math.Floor(_totalImageCount / DFNParams.TRAINING_SET_DIVIDER);

                //reduce the total count
                _totalImageCount = trainingSetSize;

                //how many we need to take for each digit
                int randomDigitSamples = (int)Math.Ceiling((double)trainingSetSize / allDigitsAndImages.Count);

                List<List<int>> randomChosenIxs = new List<List<int>>();
                Random r = new Random();

                SortedList<int, List<double[]>> newValidation = new SortedList<int, List<double[]>>();

                foreach (var picKeyVal in allDigitsAndImages)
                {
                    int digit = picKeyVal.Key;

                    List<int> randomOrder = new List<int>();

                    List<double[]> valSelection = new List<double[]>();

                    var allDImages = picKeyVal.Value;

                    for (int i = 0; i < picKeyVal.Value.Count; i++)
                        randomOrder.Add(i);

                    //randomly sort
                    randomOrder.OrderBy(x=> r.NextDouble());

                    //take the first sampleCount -- grab the random ix
                    for (int w = 0; w < randomDigitSamples; w++)
                    {

                        //rando image for validation
                        int randoIx = randomOrder[w];
                        double[] image = allDImages[randoIx];

                        //add to our validation set
                        valSelection.Add(image);
                    }

                    //our validation set
                    newValidation.Add(digit, valSelection);
                }



                //simply replace our two sets -- loaded from training and not testing places :)
                allDigitsAndImages = newValidation;

                DFNParams.FEATUREADDITIONTHRESHOLD = (float)(DFNParams.FEATUREADDITIONTHRESHOLD / DFNParams.TRAINING_SET_DIVIDER);
                

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

            //Console.WriteLine("Currently test set images == training set images. Must change soon");

        }
      
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
