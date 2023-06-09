using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Genomes.Neat;
using SharpNeat.Phenomes;
using SharpNeat.Core;
using DFN.Evaluation;
using SharpNeat.Phenomes.NeuralNets;
using SharpNeat.Decoders;
using SharpNeat.Network;
using System.Threading.Tasks;
using DFNLibrary;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Threading;

namespace DFN.Validation
{
    //we take an archive in, and then piece together a full network full of features
    //run those features on a collection of test images using multi-output perceptron learning rules 
    //then we output a result for the big network
    public class FullFeatureValidation
    {
        static BackPropPerceptron[] workerbpps;
        static int[] workerCorrectCount;
        static int[][] workerWrongGuesses;

        #region Perceptron stuff
        #region Create Base Perceptron Object

        public static MultiPerceptron CreateFullConnectedPerceptron(int inputs, int outputs, double startingWeight)
        {
            NeatGenome ng = null;

            //default factory -- initializes genomes with linear/sigmoid activation == fnID == 0
            NeatGenomeFactory neatFactory = new NeatGenomeFactory(inputs, outputs, DefaultActivationFunctionLibrary.CreateLibraryNeat(Linear.__DefaultInstance));

            NeuronGeneList nodes = new NeuronGeneList(inputs + outputs);

            //make sure to start at zero for innovation ids
            neatFactory.InnovationIdGenerator.Reset();

            //add inputs -- all of them!
            for (int i = 0; i < 1; i++)
                nodes.Add(new NeuronGene(neatFactory.NextInnovationId(), SharpNeat.Network.NodeType.Bias, 0));

            //add inputs -- all of them!
            for (int i = 0; i < inputs; i++)
                nodes.Add(new NeuronGene(neatFactory.NextInnovationId(), SharpNeat.Network.NodeType.Input, 0));

            //now add the outsputs
            for (int i = 0; i < outputs; i++)
                nodes.Add(new NeuronGene(neatFactory.NextInnovationId(), SharpNeat.Network.NodeType.Output, 0));

            ConnectionGeneList connections = new ConnectionGeneList(inputs * outputs);

            for (int j = 0; j < outputs; j++)
            {
                //to index into the output node, we have bias + inputs == inputs + 1 + j
                int jx = inputs + 1 + j;
                for (int i = 0; i < inputs + 1; i++)
                {
                    //we make sure to wire up the bias node, this is important for perceptron training purposes
                    int ix = i;

                    //grab our source/tgt genes
                    var src = nodes[ix];
                    var tgt = nodes[jx];

                    //create a connection between output j and input i
                    connections.Add(new ConnectionGene(neatFactory.NextInnovationId(), src.Id, tgt.Id, startingWeight));

                    // Register connection with endpoint neurons.
                    src.TargetNeurons.Add(tgt.Id);
                    tgt.SourceNeurons.Add(src.Id);
                }
            }

            //create a new neat genome matching a fully connected inputs to outputs multi-perceptron
            ng = neatFactory.CreateGenome(0, 0, nodes, connections, inputs, outputs, false);

            //turn the perceptron into a network -- no point delaying
            return MultiPerceptronNetworkFactory.CreateAcyclicMultiPerceptronNetwork(ng);
        }

        public static SlimPerceptron CreateFullConnectedSlimPerceptron(int inputs, int outputs, double startingWeight, bool randomizeWeights)
        {
            FastConnection[] connections = new FastConnection[(inputs + 1)* outputs];
           Random r = new Random();
            int connIx = 0;

            for (int j = 0; j < outputs; j++)
            {
                //to index into the output node, we have bias + inputs == inputs + 1 + j
                int jx = inputs + 1 + j;
                for (int i = 0; i < inputs + 1; i++)
                {
                    //we make sure to wire up the bias node, this is important for perceptron training purposes
                    int ix = i;

                    //create a connection between output j and input i
                    var conn = new FastConnection();

                    conn._srcNeuronIdx = ix;
                    conn._tgtNeuronIdx = jx;
                    conn._weight = (randomizeWeights ? r.NextDouble()*startingWeight : startingWeight);

                    connections[connIx++] = conn; 
                }
            }

            //turn the perceptron into a network -- no point delaying
            return new SlimPerceptron(inputs, connections);
        }

        #endregion

        #region Validation Control

        public static Tuple<MultiPerceptron, double, double> CreateAndTrainPerceptron(int repeatTestSet, IGenomeDecoder<NeatGenome, IBlackBox> genomeDecorder, List<NeatGenome> featureArchive)
        {
            MNISTManager mnist = MNISTManager.Instance;

            List<int> interestedDigits = mnist.InterestedDigits;

            //how many times do we do this?
            //reepate test set times

            //how many images in the test set
            int imageCount = mnist.TotalTestImageCount;

            //how many inputs into the perceptron -- whatever feature archive we send in
            int perceptronInputCount = featureArchive.Count;

            //how many outputs? whatever the total number of interested digits are
            int outputDigits = interestedDigits.Count;

            //initial weight for perceptron connections (.5 chosen originally, maybe 0 is better)
            // normalizing to 1/n where n is number of features
            double startingWeight = ((double)1)/featureArchive.Count;

            //round up all the cached behaviors for our perceptron training
            float[][] cachedBehaviors = new float[perceptronInputCount][];

            //take in all our neat genome features,
            //then cache all image behavior for all the genomes
            for (var i = 0; i < perceptronInputCount; i++)
            {
                //cache behavior for all images tested with this genome -- going to copy the array
                if (featureArchive[i] != null)
                    cachedBehaviors[i] = featureArchive[i].EvaluationInfo.Behavior;
                else //otherwise, just create an empty cache
                    cachedBehaviors[i] = new float[imageCount];

                //this is our image count -- should be consistent
                if(featureArchive[i].EvaluationInfo.Behavior != null)
                    imageCount = featureArchive[i].EvaluationInfo.Behavior.Length;
            }

            double[][] imageToCachedFeatures = new double[imageCount][];
            
            //we want to quickly build cached features
            for (var i = 0; i < imageCount; i++)
            {
                imageToCachedFeatures[i] = new double[perceptronInputCount];

                for (var j = 0; j < perceptronInputCount; j++)
                {
                    if (cachedBehaviors[j] != null)
                        imageToCachedFeatures[i][j] = cachedBehaviors[j][i];
                }
            }
       
            //clear out the previous cache
            cachedBehaviors = null;

            //now we create our perceptron for training
            //we'll need to define the Neat Genome for this

            //create our perceptron, ready for use
            MultiPerceptron perceptron = CreateFullConnectedPerceptron(perceptronInputCount, outputDigits, startingWeight);

            //send in labeled information matching cached behavior order
            double trainingAccuracy = trainPerceptron(repeatTestSet, perceptron, interestedDigits, imageCount, imageToCachedFeatures, mnist.AllTestLabels, genomeDecorder, featureArchive);

            double testAccuracy = runNetworkOnValidationImages(perceptron, genomeDecorder, featureArchive);

            //training accuracy yo, but also send back the trained perceptron for checking the validation/test set
            return new Tuple<MultiPerceptron, double, double>(perceptron, trainingAccuracy, testAccuracy);
        }

        /// <summary>
        /// Creating a slim training set, feature archive should NOT be evaluated before sending in
        /// </summary>
        /// <param name="repeatTestSet"></param>
        /// <param name="genomeDecorder"></param>
        /// <param name="featureArchive"></param>
        /// <returns></returns>
        public static Tuple<SlimPerceptron, double, double> SlimCreateAndTrainPerceptron(
            double[][] fullImageFeatureCache,
            double[][] validationImageCache,
            int repeatTestSet, 
            int decodedGenomeLength,
            //IBlackBox[] decodedGenomes, 
            int imageBatches, 
            ParallelOptions po)
        {
            MNISTManager mnist = MNISTManager.Instance;

            List<int> interestedDigits = mnist.InterestedDigits;

            int[] digitLabels = mnist.AllTestLabels;

            //how many times do we do this?
            //reepate test set times

            //how many images in the test set
            int imageCount = mnist.TotalTestImageCount;

            //how many inputs into the perceptron -- whatever feature archive we send in
            int perceptronInputCount = decodedGenomeLength;

            //how many outputs? whatever the total number of interested digits are
            int outputDigits = interestedDigits.Count;

            //initial weight for perceptron connections (.5 chosen originally, maybe 0 is better)
            // normalizing to 1/n where n is number of features
            double startingWeight = ((double)1) / decodedGenomeLength;


            //create our slim perceptron using input and output counts
            SlimPerceptron slimPerceptron = CreateFullConnectedSlimPerceptron(perceptronInputCount, outputDigits, startingWeight, DFNParams.RANDOMIZESTARTINGWEIGHTS);
            slimPerceptron.po = po;

            double learningRate = DFNParams.INITIALLEARNRATEPERCEPTRON / perceptronInputCount;

            Dictionary<int, double[][]> featureCache = new Dictionary<int, double[][]>();

            int currentStartImageIx = 0;
            int batchImageSize = (int)Math.Floor((double)imageCount/imageBatches);

            double[][] testImages = MNISTManager.Instance.AllTestImages;

            double lastTrainingAccuracy = 0;
            
            //we pad all our inputs with 1 input for hte bias
            int biasPadding = 1;

            List<int> imageOrder = new List<int>(testImages.Length);
            for (int i = 0; i < testImages.Length; i++)
                imageOrder.Add(i);
            Random rng = new Random();
            imageOrder = imageOrder.OrderBy(a => rng.Next()).ToList();


            Stopwatch sw = Stopwatch.StartNew();

            //now we can test in random order
            for (int t = 0; t < repeatTestSet; t++)
            {
                int correctCount = 0;
                currentStartImageIx = 0;

                Stopwatch testSW = Stopwatch.StartNew();

                for (int i = 0; i < imageBatches; i++)
                {
                    int iCount = Math.Min(batchImageSize, imageCount - currentStartImageIx);
                    //now we have all the substrates for feature creation 
                    //we're going to do the training in batches, since this will cut down on the initial startup times
                    double[][] imageToCachedFeatures;

                    //use a cache for our repeated tests, thank you
                    if(!featureCache.TryGetValue(i, out imageToCachedFeatures))
                    {
                        Stopwatch cachedSW = Stopwatch.StartNew();

                        imageToCachedFeatures = new double[iCount][];

                        //grab our test images
                        for (int image = 0; image < iCount; image++)
                        {
                            int shuffleIx = imageOrder[currentStartImageIx + image];

                            //pull from the cache for this object
                            double[] featureCached = fullImageFeatureCache[shuffleIx];
                            imageToCachedFeatures[image] = featureCached;
                        }

                        Console.WriteLine("Feature Cache Creation: {0} for {1} features and {2} images in the batch", cachedSW.ElapsedMilliseconds, decodedGenomeLength, iCount);

                        //cache image behavior for this batch
                        featureCache.Add(i, imageToCachedFeatures);
                    }



                    Stopwatch imageRunSW = Stopwatch.StartNew();

                    int previousCorrect = correctCount;

                    //now we have part of our training set, let's train the slim perceptron
                    for (int image = 0; image < iCount; image++)
                    {
                        bool isCorrect = false;

                        //to read the label, we need to grab the shuffled index (though our feature cache is in order)
                        int shuffleIx = imageOrder[currentStartImageIx + image];
                        int labelDigit = digitLabels[shuffleIx];

                        //activate the network, then correct the weights in a parallel fashion
                        slimPerceptron.ActivateAndCorrect(
                            imageToCachedFeatures[image], //send inputs in 
                            interestedDigits.IndexOf(labelDigit), // what index should be the highest activated?
                            true, //if it's wrong, should we fix it?
                            learningRate, //how much do we fix it  
                            true, //should we fix it in place, or send back the desired corrections?
                            out isCorrect //did we get it right???
                            );

                        //if we got it right, incremenet our correct count
                        if (isCorrect)
                            correctCount++;
                    }

                    Console.WriteLine("Full Image Batch Time: {0} for {1} features and {2} images in the batch, correct % batch: {3}, incorrect: {4}", 
                        imageRunSW.ElapsedMilliseconds,
                        decodedGenomeLength,
                        iCount, 
                        (correctCount - previousCorrect)/(double)iCount,
                        iCount - (correctCount - previousCorrect));


                    //finished icount batch, go to next round
                    currentStartImageIx += iCount;
                }


                Console.WriteLine("Single Test Run Time: {0} for {1} features and {2} images", testSW.ElapsedMilliseconds, decodedGenomeLength, imageCount);

                double trainingAccuracy = ((double)correctCount) / imageCount;
                lastTrainingAccuracy = trainingAccuracy;

                //we do this every time no matter what, it's fast enough now
                //if (t % DFNLibrary.DFNParams.TRAININGOUTPUTFREQ == 0)
                //{
                    // output the current training/test stats
                double testingAccuracy = runSlimNetworkOnValidation(slimPerceptron, validationImageCache, decodedGenomeLength, biasPadding, po);
                    System.IO.File.AppendAllText("aPerceptronResults.txt", String.Format("{3}---Training: {0}, Testing: {1}, Training Pass: {2}", trainingAccuracy, testingAccuracy, t, System.DateTime.Now) + Environment.NewLine);
                    System.IO.File.AppendAllText("aTrainingAccuracy.txt", trainingAccuracy + Environment.NewLine);
                    System.IO.File.AppendAllText("aTestingAccuracy.txt", testingAccuracy + Environment.NewLine);
                //}
                                
                learningRate *= (1 - DFNLibrary.DFNParams.LEARNRATEDECAY);
                if (trainingAccuracy >= DFNParams.PERCEPTRONTRAININGSTOPACCURACY)
                {
                    Console.WriteLine("Full Test Elapse Time: {0} for {1} features and {2} repetitions", sw.ElapsedMilliseconds, decodedGenomeLength, repeatTestSet);

                    double accuracy = runSlimNetworkOnValidation(slimPerceptron, validationImageCache, decodedGenomeLength, biasPadding, po);

                    //send back the network, and it's latest work
                    //TODO: Should this be the BEST activation we saw, or the last?
                    return new Tuple<SlimPerceptron, double, double>(slimPerceptron, lastTrainingAccuracy, accuracy);
                }
            }

            Console.WriteLine("Full Test Elapse Time: {0} for {1} features and {2} repetitions", sw.ElapsedMilliseconds, decodedGenomeLength, repeatTestSet);

            double lastValidationAccuracy = runSlimNetworkOnValidation(slimPerceptron, validationImageCache, decodedGenomeLength, biasPadding, po);

            //send back the network, and it's latest work
            //TODO: Should this be the BEST activation we saw, or the last?
            return new Tuple<SlimPerceptron, double, double>(slimPerceptron, lastTrainingAccuracy, lastValidationAccuracy);
        }

        #endregion

        #region Train Perceptron On Test Set

        //the framework for training our perceptron network using the caches behaviors (which we can run multiple times if necessary)
        /// <summary>
        /// Interested digits == list of outputs to digit maps -- e.g. output 0 is a "3" classifier, output 1 is a "9" classifier. 
        /// 
        /// </summary>
        /// <param name="perceptronNetwork"></param>
        /// <param name="interestedDigits"></param>
        /// <param name="imageCount"></param>
        /// <param name="cachedBehaviors"></param>
        /// <param name="digitsLabels"></param>
        public static double trainPerceptron(int repeatTestCount, MultiPerceptron perceptronNetwork, List<int> interestedDigits, int imageCount, double[][] cachedBehaviors, int[] digitsLabels, IGenomeDecoder<NeatGenome, IBlackBox> genomeDecorder, List<NeatGenome> featureArchive)
        {
            ParallelOptions parallelOptions = new ParallelOptions();
            parallelOptions.MaxDegreeOfParallelism = DFNLibrary.DFNParams.THREADS;

            //we use the cached behaviors to train the pereceptron network according to multi-output perceptron rules... yo

            if(perceptronNetwork.OutputCount != interestedDigits.Count)
                throw new Exception("Outputs should each be assigned a digit when training perceptron network. # of digits != # of perceptron outputs");

            int featureCount = cachedBehaviors.Length;

            float learningRate = DFNLibrary.DFNParams.INITIALLEARNRATEPERCEPTRON;

            //Object[] locks = new Object[perceptronNetwork.ConnectionList.Length];
            //for (int i = 0; i < locks.Length; i++)
            //    locks[i] = new Object();

            //List<double[]> weightAdjustments = new List<double[]>();


            for (var t = 0; t < repeatTestCount; t++)
            {
                //this code doesn't work
                //double[] weightUpdates = new double[perceptronNetwork.ConnectionList.Length];

                //for (int i = 0; i < weightUpdates.Length; i++)
                //{
                //    weightUpdates[i] = 0.0;
                //}

                int correct = 0;

                    //there are all the images we have to train for in parallel
                    //Parallel.For(0, imageCount, parallelOptions, i =>
                for (int i = 0; i < imageCount; i++)
                {
                    int imageToTest;

                    // rotational image selection
                    //if (imageCount % 10 == 0)
                    if (imageCount - i > imageCount % 10)
                        imageToTest = i / 10 + (i % 10) * (imageCount / 10);
                    else
                        imageToTest = i; // in case imageCount isn't divisible by 10, the last few guys won't be in the rotation

                    double[] localWeightAdjust;
                    bool isCorrect = runPerceptronOnImage(
                        perceptronNetwork, //the network
                        interestedDigits, //the mapping from output index to digit classifier (i.e. 0 classified 2, 1 classifies 5)
                        digitsLabels[imageToTest], //what is the correct digit for this image
                        imageToTest, //what index are we inside cached behaviors
                        featureCount, //how many features are cached
                        cachedBehaviors[imageToTest], //what are the cached features
                        true, //do we want to train the network when incorrect?
                        out localWeightAdjust,
                        false, //we dont want to adjust all at once, so dont fill up our local weight adjustments
                        learningRate);

                    if (isCorrect)
                        correct++;

                    //now update serially, please!
                    //perceptronNetwork.MakeAllWeightCorrections(localWeightAdjust);
                    //if(!isCorrect)
                    //{
                    //    Console.Write("");
                    //}
                    // if localWeightAdjust is null, no adjustments were needed --PAUL!
                    //if (localWeightAdjust != null)
                    //{
                    //    lock (weightAdjustments)
                    //    {
                    //        weightAdjustments.Add(localWeightAdjust);

                    //    }
                    //}
                }
                        //});

                //for (var w = 0; w < weightAdjustments.Count; w++)
                //{
                //    double[] wAd = weightAdjustments[w];
                //    for(int c=0; c < wAd.Length; c++)
                //    {
                //        //update the weights with this weight array
                //        weightUpdates[c] += wAd[c];
                //    }
                //}

                //now weight updates has what we need!
                //all at once we correct the network
                //perceptronNetwork.MakeAllWeightCorrections(weightUpdates);

                learningRate *= (1 - DFNLibrary.DFNParams.LEARNRATEDECAY);

                if (t % DFNLibrary.DFNParams.TRAININGOUTPUTFREQ == 0)
                {
                    // output the current training/test stats
                    double trainingAccuracy = ((double)correct) / imageCount;
                    double testingAccuracy = runNetworkOnValidationImages(perceptronNetwork, genomeDecorder, featureArchive);
                    System.IO.File.AppendAllText("aPerceptronResults.txt", String.Format("{3}---Training: {0}, Testing: {1}, Training Pass: {2}", trainingAccuracy, testingAccuracy, t, System.DateTime.Now) + Environment.NewLine);
                    System.IO.File.AppendAllText("aTrainingAccuracy.txt", trainingAccuracy + Environment.NewLine);
                    System.IO.File.AppendAllText("aTestingAccuracy.txt", testingAccuracy + Environment.NewLine);
                }
            }

            int correctGuess = 0;

            object correctLock = new object();

             //there are all the images we have to train for in parallel
            //can't run in parallel because they share the same activation array
            //Parallel.For(0, imageCount, i =>
                for (int i = 0; i < imageCount; i++)
                {

                    //now, after training we check accuracy
                    double [] empty;
                    //we use the correct bool variable to make this part of the code clear
                    bool correct = runPerceptronOnImage(
                        perceptronNetwork, //the network
                        interestedDigits, //the mapping from output index to digit classifier (i.e. 0 classified 2, 1 classifies 5)
                        digitsLabels[i], //what is the correct digit for this image
                        i, //what index are we inside cached behaviors
                        featureCount, //how many features are cached
                        cachedBehaviors[i], //what are the cached features
                        false, //do we want to train the network when incorrect? No, this is an accuracy test only
                        out empty,
                        false
                        );

                    //when eprceptron is run, it returns a bool saying whether or not the perceptron network was correct
                    if (correct)
                    {
                        //must be locked across threads
                        lock (correctLock)
                        {
                            correctGuess++;
                        }
                    }
                }
                //});


            //return the accuracy of the trained perceptron
            return (double)correctGuess / imageCount;
        }
        public static bool runPerceptronOnImage(MultiPerceptron perceptronNetwork, 
            List<int> outputsToDigitGuess, 
            int digitLabel, 
            int imageIx, 
            int featureCount, 
            double[] featureBehaviors,
            bool trainNetworkAfter,
            out double[] weightAdjustments,
            bool adjustWeightsAtOnce,
            float learningRate = 0f
            )
        {
            ParallelOptions po = new ParallelOptions();
            po.MaxDegreeOfParallelism = DFNLibrary.DFNParams.THREADS;
            perceptronNetwork.po = po;

            //Step 1. Gather our cached input features
            //double[] inputs = new double[featureCount];

            //yay. All the feature outputs are belong to us
            //for (int m = 0; m < featureCount; m++)
            //    inputs[m] = cachedBehaviors[m][imageIx];

            //we have our inputs -- cached from the evolutionary runs

            //Step 2. Run the perceptron 

            //clear
            perceptronNetwork.ResetState();

            //copy our inputs into the perceptron (an acyclic neural network)
            perceptronNetwork.InputSignalArray.CopyFrom(featureBehaviors, 0); // TODO: would parallelization be worth the trouble here?

            //activate the network, would you kindly?
            perceptronNetwork.Activate();

            weightAdjustments = null;

            //Step 3. "Correct" the incorrect behavior -- derrr
            if (trainNetworkAfter)
            {
                bool[] correctChoices = new bool[perceptronNetwork.OutputCount];

                //this is what should have happened -- we have 1 correct digit identifier, and the rest should be false 
                for (var d = 0; d < correctChoices.Length; d++)
                {
                    if (outputsToDigitGuess[d] == digitLabel)
                        correctChoices[d] = true;
                    else
                        correctChoices[d] = false;
                }

                //now we have our correct choices, send to perceptron to do it's thang
                //send in the correct objects, it'll handle training with it's learnin' rate

                //if we sent in weight adjusts and its not null, fill it up, please!
                weightAdjustments = perceptronNetwork.LearnFromCorrect(correctChoices, learningRate, adjustWeightsAtOnce);

            }

            //step 4, get accuracy
            var oSignal = perceptronNetwork.OutputSignalArray;
            int guessDigit = -1;
            double maxOut = double.MinValue;
            for (var i = 0; i < perceptronNetwork.OutputCount; i++)
            {
                double osig = oSignal[i];
                if(osig > maxOut)
                {
                    guessDigit = outputsToDigitGuess[i];
                    maxOut = osig;
                }
            } 

            //did we get it right? Our max activation was the right digit
            return guessDigit == digitLabel;
        }

        #endregion

        #region Run Network On Validation

        public static List<double[]> cachedBehaviors = new List<double[]>();

        public static double runSlimNetworkOnValidation(SlimPerceptron slimPerceptron, double[][] validationImageBehavior, 
            int decodedGenomeLength,
            //IBlackBox[] decodedGenomes, 
            int biasPadding, 
            ParallelOptions po)
        {
            MNISTManager mm = MNISTManager.Instance;
            List<int> outputGuesses = mm.InterestedDigits;
            //We pull the 2D array of images
            double[][] validationImages = mm.AllValidationImages;
            int[] validationDigits = mm.AllValidationLabels;

            //how many images are we testing?
            int imageCount = validationImages.Length;

            int correctGuesses = 0;

            for (int i = 0; i < validationImageBehavior.Length; i++)
            {
                double[] inArray = validationImageBehavior[i];

                int correctIx = outputGuesses.IndexOf(validationDigits[i]);

                bool isCorrect = false;

                //this will activate, but not do any correction logic
                slimPerceptron.ActivateAndCorrect(inArray, correctIx, false, 0, false, out isCorrect);

                //did we get it right? add 1, otherwise, don't add any :/
                correctGuesses += (isCorrect ? 1 : 0);
            }

            //how many right?
            return (double)correctGuesses / validationImages.Length;
        }

        //then we need to run our networks on all the validation set to get the full fitness
        public static double runNetworkOnValidationImages(MultiPerceptron trainedPerceptron, IGenomeDecoder<NeatGenome, IBlackBox> genomeDecoder, List<NeatGenome> features)
        {
            MNISTManager mm = MNISTManager.Instance; 

            //We pull the 2D array of images
            double[][] validationImages = mm.AllValidationImages;
            int[] validationDigits = mm.AllValidationLabels;

            //how many images are we testing?
            int imageCount = validationImages.Length;

            //how many features? we call this a lot so just make a variable
            int featureCount = features.Count;

            //mapping from outputIx to what digit is chosen
            List<int> interestedDigits = mm.InterestedDigits;

            IBlackBox[] featureNetwork = new IBlackBox[featureCount];

            Parallel.For(0, featureCount, i =>
            {
                //need to decode our feature genome into an actual feature network
                featureNetwork[i] = genomeDecoder.Decode(features[i]);

            });

            //now we have all our features, let's see what our perceptron does
            int correctGuess = 0;

            double[] imageBehavior = new double[featureCount];

            //now, after training we check accuracy
            for (int i = 0; i < imageCount; i++)
            {
                // check for cached behavior from a previous test
                if (cachedBehaviors.Count > i)
                {
                    imageBehavior = cachedBehaviors[i];
                }
                else
                {
                    //now we have all our networks, we need to create our behavior
                    for (var n = 0; n < featureCount; n++)
                    {

                        //get our network
                        var network = featureNetwork[n];

                        //network is null -- skip please!
                        if (network == null)
                        {
                            imageBehavior[n] = 0.0f;
                            continue;
                        }

                        var inputs = network.InputSignalArray;
                        var outputs = network.OutputSignalArray;

                        //clear blackbox
                        network.ResetState();

                        //load in our uinputs
                        network.InputSignalArray.CopyFrom(validationImages[i], 0);

                        //activate our network
                        network.Activate();

                        imageBehavior[n] = (float)outputs[0];                        
                    }
                    cachedBehaviors.Add((double[])imageBehavior.Clone());
                }
                double[] weights;
                //we use the correct bool variable to make this part of the code clear
                bool correct = runPerceptronOnImage(
                    trainedPerceptron, //the network
                    interestedDigits, //the mapping from output index to digit classifier (i.e. 0 classified 2, 1 classifies 5)
                    validationDigits[i], //what is the correct digit for this image
                    i, //what index are we inside cached behaviors
                    featureCount, //how many features are cached
                    imageBehavior, //what are the cached features
                    false, //do we want to train the network when incorrect? No, this is an accuracy test only
                    out weights, //where we store the desired weight cahnges during learning -- it will remain emtpy
                    false // do we want to adjust weights for each run, or all at once, not applicable here so false
                    );

                //when eprceptron is run, it returns a bool saying whether or not the perceptron network was correct
                if (correct)
                    correctGuess++;
            }

            //return the accuracy of the trained perceptron
            return (double)correctGuess / imageCount;
        }
        #endregion
        #endregion

        #region Backprop Training Perceptron

        public static double runBackpropNetworkOnValidation(BackPropPerceptron bpp,
            int decodedGenomeLength,
            //IBlackBox[] decodedGenomes, 
            int biasPadding,
            ParallelOptions po,
            bool showWrongGuesses = false)
        {
            IDomain domain = DomainManager.Instance;
            List<int> outputGuesses = domain.getInterestedDigits();
            //We pull the 2D array of images
            double[][] validationImages = domain.getAllValidationImages();
            int[] validationDigits = domain.getAllValidationLabels();

            //how many images are we testing?
            int imageCount = validationImages.Length;

            int correctGuesses = 0;

            for (int i = 0; i < validationImages.Length; i++)
            {
                double[] inArray = validationImages[i];

                int correctIx = outputGuesses.IndexOf(validationDigits[i]);

                //this will activate, but not do any correction logic
                bool isCorrect = bpp.ActivateFullNetwork(inArray, correctIx, false, 0,0,0);

                // save this wrong guess to a file
                if (!isCorrect && showWrongGuesses)
                {
                    string outputDirectory = Environment.CurrentDirectory + "/wrongGuesses/";
                    if(!Directory.Exists(outputDirectory))
                        Directory.CreateDirectory(outputDirectory);
                    string outFile = outputDirectory + "A" + correctIx + "-G" + bpp.LastChosenIx;
                    while (File.Exists(outFile + ".jpg"))
                        outFile = outFile + "_";
                    WriteImage(inArray, outFile + ".jpg", 0);
                }

                //did we get it right? add 1, otherwise, don't add any :/
                correctGuesses += (isCorrect ? 1 : 0);
            }

            //how many right?
            return (double)correctGuesses / validationImages.Length;
        }

        public static double runBackpropNetworkOnValidationMulti(BackPropPerceptron bpp,
            int decodedGenomeLength,
            //IBlackBox[] decodedGenomes, 
            int biasPadding,
            ParallelOptions po,
            bool showWrongGuesses = false)
        {
            int threads = po.MaxDegreeOfParallelism - 1;
            IDomain domain = DomainManager.Instance;
            List<int> outputGuesses = domain.getInterestedDigits();
            //We pull the 2D array of images
            double[][] validationImages = domain.getAllValidationImages();
            int[] validationDigits = domain.getAllValidationLabels();

            //how many images are we testing?
            int imageCount = validationImages.Length;
            int batchImageSize = (imageCount / (threads + 1));
            int correctGuesses = 0;
            int currentStartImageIx = 0;

            int[] imageOrder = new int[imageCount];
            for (int i = 0; i < imageOrder.Length; i++)
                imageOrder[i] = i;
            List<int> io = imageOrder.ToList<int>();



            // create the worker threads and start them on a set of the images for this pass
            BackpropWorker[] workers = new BackpropWorker[threads];
            Thread[] workerThreads = new Thread[threads];

            for (int i = 0; i < threads; i++)
            {
                int iCount = Math.Min(batchImageSize, imageCount - currentStartImageIx);


                workers[i] = new BackpropWorker(i, iCount, currentStartImageIx, io, validationDigits, workerbpps[i], validationImages, domain.getInterestedDigits(), 0, 0, 0, false, new PassFinishedCallback(PassFinished), showWrongGuesses);
                workerThreads[i] = new Thread(new ThreadStart(workers[i].DoWork));
                workerThreads[i].Start();

                currentStartImageIx += iCount;
            }

            int myCount = Math.Min(batchImageSize, imageCount - currentStartImageIx);
            BackpropWorker bpw = new BackpropWorker(0, myCount, currentStartImageIx, io, validationDigits, bpp, validationImages, domain.getInterestedDigits(), 0, 0, 0, false, null, showWrongGuesses);
            bpw.DoWork();

            // now collect all the results from the worker threads 
            for (int i = 0; i < threads; i++)
                workerThreads[i].Join();

            correctGuesses = bpw.getCorrect();
            for (int i = 0; i < threads; i++)
            {
                correctGuesses += workerCorrectCount[i];
            }

            //how many right?
            return (double)correctGuesses / validationImages.Length;
        }

        public static void WriteImage(double[] feature, string outputFile, int offset = 1)
        {
            int xSize = DomainManager.Instance.getPixelWidth();
            int ySize = DomainManager.Instance.getPixelHeight();
            int scaleFactor = 1; // set to X to create an image X times wider/taller
            Bitmap bmp = new Bitmap(xSize * scaleFactor, ySize * scaleFactor);

            for (int i = offset; i < feature.Length; i++)
            {
                //if (substrateNetworkDef.ConnectionList[i].SourceNodeId > 0)
                {
                    // color is normalized and a sharpening factor (2000) is applied to be able to distinguish fine differences more easily
                    int alpha;
                    alpha = Math.Max(Math.Min((int)Math.Floor(feature[i] * 255), 255), 0);
                    Color c = Color.FromArgb(alpha, alpha, alpha);

                    // for each weight in the network, generate 4 pixels to enlarge the image

                    for (int j = 0; j < scaleFactor; j++)
                        for (int k = 0; k < scaleFactor; k++)
                        {
                            if(j == 0 && k == 0)
                                bmp.SetPixel((i - offset) % xSize * scaleFactor, (i - offset) / ySize * scaleFactor, c);
                            if(j != 0)
                                bmp.SetPixel((i - offset) % xSize * scaleFactor + j, (i - offset) / ySize * scaleFactor, c);
                            if(k != 0)
                                bmp.SetPixel((i - offset) % xSize * scaleFactor, (i - offset) / ySize * scaleFactor + k, c);
                            if(j != 0 && k != 0)
                                bmp.SetPixel((i - offset) % xSize * scaleFactor + j, (i - offset) / ySize * scaleFactor + k, c);
                        }
                    

                    
                }
            }

            bmp.Save(outputFile, System.Drawing.Imaging.ImageFormat.Png);
        }

        public static void WriteFeature(double[] feature, string outputFile, int offset = 1)
        {
            int xSize = DomainManager.Instance.getPixelWidth();
            int ySize = DomainManager.Instance.getPixelHeight();

            Bitmap bmp = new Bitmap(xSize * 2, ySize * 2);

            for (int i = offset; i < feature.Length; i++)
            {
                //if (substrateNetworkDef.ConnectionList[i].SourceNodeId > 0)
                {
                    // color is normalized and a sharpening factor (2000) is applied to be able to distinguish fine differences more easily
                    int alpha;
                    alpha = Math.Max(Math.Min((int)Math.Floor((feature[i]) * 2000) + 128, 255), 0);
                    Color c = Color.FromArgb(alpha, alpha, alpha);

                    // for each weight in the network, generate 4 pixels to enlarge the image
                    bmp.SetPixel((i - offset) % xSize * 2, (i - offset) / ySize * 2, c);
                    bmp.SetPixel((i - offset) % xSize * 2 + 1, (i - offset) / ySize * 2, c);
                    bmp.SetPixel((i - offset) % xSize * 2, (i - offset) / ySize * 2 + 1, c);
                    bmp.SetPixel((i - offset) % xSize * 2 + 1, (i - offset) / ySize * 2 + 1, c);
                }
            }
            
            bmp.Save(outputFile, System.Drawing.Imaging.ImageFormat.Jpeg);
        }

        #region standardSingleThreadedBackprop
        public static Tuple<BackPropPerceptron, double, double> BackpropCreateAndTrainNetwork(
            int repeatTestSet,
            SingleOutputNetwork[] features,
            int imageBatches,
            ParallelOptions po,
            double[][] startingWeights = null)
        {

            string imageDirectory = Environment.CurrentDirectory + "/featureImages/";
            if (DFNParams.RECORDFEATURES)
            {
                // output the initial features
                if (!Directory.Exists(imageDirectory))
                    Directory.CreateDirectory(imageDirectory);
                for (int k = 0; k < features.Length; k++)
                {
                    WriteFeature(features[k].Weights, imageDirectory + "feature_" + k + "__PRE.jpg");
                }
            }
            IDomain domain = DomainManager.Instance;

            List<int> interestedDigits = domain.getInterestedDigits();

            int[] digitLabels = domain.getAllTestLabels();

            //how many times do we do this?
            //reepate test set times

            //how many images in the test set
            int imageCount = domain.getTotalTestImageCount();

            //how many inputs into the perceptron -- whatever feature archive we send in
            int perceptronInputCount = features.Length;

            //how many outputs? whatever the total number of interested digits are
            int outputDigits = interestedDigits.Count;

            //initial weight for perceptron connections (.5 chosen originally, maybe 0 is better)
            // normalizing to 1/n where n is number of features
            Random r = new Random();
            //http://deeplearning.net/tutorial/mlp.html range sampling for sigmoid starting weights -- whatever
            double startingWeight = 4*Math.Sqrt(6.0/(features.Length + 10));//((double)1) / features.Length;


            //create our slim perceptron using input and output counts
            //SlimPerceptron slimPerceptron = CreateFullConnectedSlimPerceptron(perceptronInputCount, outputDigits, startingWeight, DFNParams.RANDOMIZESTARTINGWEIGHTS);
            //slimPerceptron.po = po;
            BackPropPerceptron bpp;
            if(startingWeights == null)
                bpp = new BackPropPerceptron(features, outputDigits, startingWeight);
            else
                bpp = new BackPropPerceptron(features, outputDigits, startingWeights);

            double learningRate = DFNParams.INITIALLEARNRATEBACKPROP;/// perceptronInputCount;
            double secondLearningRate = 4.9* learningRate;// / perceptronInputCount;
            double momentum = DFNParams.MOMENTUM;

            Dictionary<int, double[][]> featureCache = new Dictionary<int, double[][]>();

            int currentStartImageIx = 0;
            int batchImageSize = (int)Math.Floor((double)imageCount / imageBatches);

            double[][] testImages = DomainManager.Instance.getAllTestImages();

            double lastTrainingAccuracy = 0;

            //we pad all our inputs with 1 input for hte bias
            int biasPadding = 1;

            List<int> imageOrder = new List<int>(testImages.Length);
            for (int i = 0; i < testImages.Length; i++)
                imageOrder.Add(i);
            Random rng = new Random();
            imageOrder = imageOrder.OrderBy(a => rng.Next()).ToList();


            Stopwatch sw = Stopwatch.StartNew();

            //now we can test in random order
            for (int t = 0; t < repeatTestSet; t++)
            {
                int[] wrongGuesses = new int[interestedDigits.Count];
                int correctCount = 0;
                currentStartImageIx = 0;

                Stopwatch testSW = Stopwatch.StartNew();                

                for (int i = 0; i < imageBatches; i++)
                {
                    int iCount = Math.Min(batchImageSize, imageCount - currentStartImageIx);
                    //now we have all the substrates for feature creation 
                    //we're going to do the training in batches, since this will cut down on the initial startup times
                    double[][] imageToCachedFeatures;

                    //use a cache for our repeated tests, thank you
                    if (!featureCache.TryGetValue(i, out imageToCachedFeatures))
                    {
                        Stopwatch cachedSW = Stopwatch.StartNew();

                        imageToCachedFeatures = new double[iCount][];

                        //grab our test images
                        for (int image = 0; image < iCount; image++)
                        {
                            int shuffleIx = imageOrder[currentStartImageIx + image];

                            //pull from the cache for this object
                            double[] featureCached = testImages[shuffleIx];
                            imageToCachedFeatures[image] = featureCached;
                        }

                        Console.WriteLine("Feature Cache Creation: {0} for {1} features and {2} images in the batch", cachedSW.ElapsedMilliseconds, features.Length, iCount);

                        //cache image behavior for this batch
                        featureCache.Add(i, imageToCachedFeatures);
                    }

                    Stopwatch imageRunSW = Stopwatch.StartNew();

                    int previousCorrect = correctCount;

                    //now we have part of our training set, let's train the slim perceptron
                    for (int image = 0; image < iCount; image++)
                    {
                        //to read the label, we need to grab the shuffled index (though our feature cache is in order)
                        int shuffleIx = imageOrder[currentStartImageIx + image];
                        int labelDigit = digitLabels[shuffleIx];

                        //activate the network, then correct the weights in a parallel fashion
                        bool isCorrect = bpp.ActivateFullNetwork(
                            imageToCachedFeatures[image], //send inputs in 
                            interestedDigits.IndexOf(labelDigit), // what index should be the highest activated?
                            true, //if it's wrong, should we fix it?
                            learningRate, //how much do we fix it  
                            secondLearningRate, //how much do we fix the next layer too?
                            momentum //momentum is if your previous delta was in the same direction, we add part of it
                            );

                        //if we got it right, incremenet our correct count
                        if (isCorrect)
                            correctCount++;
                        else
                            wrongGuesses[labelDigit]++;
                    }

                    Console.WriteLine("Full Image Batch Time: {0} for {1} features and {2} images in the batch, correct % batch: {3}, incorrect: {4}",
                        imageRunSW.ElapsedMilliseconds,
                        features.Length,
                        iCount,
                        (correctCount - previousCorrect) / (double)iCount,
                        iCount - (correctCount - previousCorrect));


                    //finished icount batch, go to next round
                    currentStartImageIx += iCount;
                }


                Console.WriteLine("Single Test Run Time: {0} for {1} features and {2} images", testSW.ElapsedMilliseconds, features.Length, imageCount);

                double trainingAccuracy = ((double)correctCount) / imageCount;
                lastTrainingAccuracy = trainingAccuracy;

                //we do this every time no matter what, it's fast enough now
                //if (t % DFNLibrary.DFNParams.TRAININGOUTPUTFREQ == 0)
                //{
                // output the current training/test stats
                if (trainingAccuracy >= DFNParams.TRAININGSTOPACCURACY)
                {
                    if (DFNParams.SAVEWRONGGUESSES)
                    {
                        // we're done with backprop, output the stats regarding which digits were wrong in training/testing, and output the final features
                        System.IO.File.AppendAllText("aTrainingGuessesWrong.txt", String.Format("{0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}:{8}:{9}", wrongGuesses[0], wrongGuesses[1], wrongGuesses[2], wrongGuesses[3], wrongGuesses[4], wrongGuesses[5], wrongGuesses[6], wrongGuesses[7], wrongGuesses[8], wrongGuesses[9]));
                    }
                    if (DFNParams.RECORDFEATURES)
                    {
                        for (int k = 0; k < bpp.allFeatures.Length; k++)
                        {
                            WriteFeature(bpp.allFeatures[k].Weights, imageDirectory + "feature_" + k + "_POST.jpg");
                        }
                    }

                    double testingAccuracy = runBackpropNetworkOnValidation(bpp, features.Length, biasPadding, po, DFNParams.SAVEWRONGGUESSES);
                    System.IO.File.AppendAllText("aPerceptronResults.txt", String.Format("{3}---Training: {0}, Testing: {1}, Training Pass: {2}", trainingAccuracy, testingAccuracy, t, System.DateTime.Now) + Environment.NewLine);
                    System.IO.File.AppendAllText("aTrainingAccuracy.txt", trainingAccuracy + Environment.NewLine);
                    System.IO.File.AppendAllText("aTestingAccuracy.txt", testingAccuracy + Environment.NewLine);
                    Environment.Exit(0);
                }
                else
                {
                    double testingAccuracy = runBackpropNetworkOnValidation(bpp, features.Length, biasPadding, po);
                    System.IO.File.AppendAllText("aPerceptronResults.txt", String.Format("{3}---Training: {0}, Testing: {1}, Training Pass: {2}", trainingAccuracy, testingAccuracy, t, System.DateTime.Now) + Environment.NewLine);
                    System.IO.File.AppendAllText("aTrainingAccuracy.txt", trainingAccuracy + Environment.NewLine);
                    System.IO.File.AppendAllText("aTestingAccuracy.txt", testingAccuracy + Environment.NewLine);                    
                }
                
                learningRate *= (1 - DFNLibrary.DFNParams.LEARNRATEDECAY);
                //decay second layer rate at the same pace
                secondLearningRate *= (1 - DFNLibrary.DFNParams.LEARNRATEDECAY);

            }

            Console.WriteLine("Full Test Elapse Time: {0} for {1} features and {2} repetitions", sw.ElapsedMilliseconds, features.Length, repeatTestSet);

            double lastValidationAccuracy = runBackpropNetworkOnValidation(bpp, features.Length, biasPadding, po);
            //send back the network, and it's latest work
            //TODO: Should this be the BEST activation we saw, or the last?
            return new Tuple<BackPropPerceptron, double, double>(bpp, lastTrainingAccuracy, lastValidationAccuracy);
        }
#endregion

        // multithreaded backprop
        public static Tuple<BackPropPerceptron, double, double> BackpropCreateAndTrainNetworkMulti(
            int repeatTestSet,
            SingleOutputNetwork[] features,
            int imageBatches,
            ParallelOptions po,
            double[][] startingWeights = null)
        {
            int threads = po.MaxDegreeOfParallelism;
            
            string imageDirectory = Environment.CurrentDirectory + "/featureImages/";
            if (DFNParams.RECORDFEATURES)
            {
                // output the initial features
                if (!Directory.Exists(imageDirectory))
                    Directory.CreateDirectory(imageDirectory);
                for (int k = 0; k < features.Length; k++)
                {
                    WriteFeature(features[k].Weights, imageDirectory + "feature_" + k + "__PRE.jpg");
                }
            }
            IDomain domain = DomainManager.Instance;

            List<int> interestedDigits = domain.getInterestedDigits();

            int[] digitLabels = domain.getAllTestLabels();

            //how many times do we do this?
            //reepate test set times

            //how many images in the test set
            int imageCount = domain.getTotalTestImageCount();

            //how many inputs into the perceptron -- whatever feature archive we send in
            int perceptronInputCount = features.Length;

            //how many outputs? whatever the total number of interested digits are
            int outputDigits = interestedDigits.Count;

            //initial weight for perceptron connections (.5 chosen originally, maybe 0 is better)
            // normalizing to 1/n where n is number of features
            Random r = new Random();
            //http://deeplearning.net/tutorial/mlp.html range sampling for sigmoid starting weights -- whatever
            double startingWeight = 4 * Math.Sqrt(6.0 / (features.Length + 10));//((double)1) / features.Length;


            //create our slim perceptron using input and output counts
            //SlimPerceptron slimPerceptron = CreateFullConnectedSlimPerceptron(perceptronInputCount, outputDigits, startingWeight, DFNParams.RANDOMIZESTARTINGWEIGHTS);
            //slimPerceptron.po = po;
            BackPropPerceptron bpp; // main "merged" bpp 
            workerbpps = new BackPropPerceptron[threads - 1];
            workerCorrectCount = new int[threads - 1];
            workerWrongGuesses = new int[threads - 1][];

            if (startingWeights == null)
            {
                bpp = new BackPropPerceptron(features, outputDigits, startingWeight);
                for (int i = 0; i < threads - 1; i++)
                    workerbpps[i] = new BackPropPerceptron(copyFeatures(features), outputDigits, startingWeight);
            }
            else
            {
                bpp = new BackPropPerceptron(features, outputDigits, startingWeights);
                for (int i = 0; i < threads - 1; i++)
                    workerbpps[i] = new BackPropPerceptron(copyFeatures(features), outputDigits, startingWeights);
            }

            double learningRate = DFNParams.INITIALLEARNRATEBACKPROP;/// perceptronInputCount;
            double secondLearningRate = 4.9 * learningRate;// / perceptronInputCount;
            double momentum = DFNParams.MOMENTUM;

            Dictionary<int, double[][]> featureCache = new Dictionary<int, double[][]>();

            int currentStartImageIx = 0;
            int batchImageSize = (int)Math.Floor((double)imageCount / (threads * imageBatches));

            double[][] testImages = DomainManager.Instance.getAllTestImages();

            double lastTrainingAccuracy = 0;

            //we pad all our inputs with 1 input for the bias
            int biasPadding = 1;

            List<int> imageOrder = new List<int>(testImages.Length);
            for (int i = 0; i < testImages.Length; i++)
                imageOrder.Add(i);
            Random rng = new Random();            

            Stopwatch sw = Stopwatch.StartNew();

            //now we can test in random order
            for (int t = 0; t < repeatTestSet; t++)
            {
                int[] wrongGuesses = new int[interestedDigits.Count];
                int correctCount = 0;
                currentStartImageIx = 0;

                Stopwatch testSW = Stopwatch.StartNew();

                // reshuffle the images
                imageOrder = imageOrder.OrderBy(a => rng.Next()).ToList();

                

                // create the worker threads and start them on a set of the images for this pass
                BackpropWorker[] workers = new BackpropWorker[threads - 1];
                Thread[] workerThreads = new Thread[threads - 1];               

                for (int batch = 0; batch < imageBatches; batch++)
                {
                    for (int i = 0; i < threads - 1; i++)
                    {
                        int iCount = Math.Min(batchImageSize, imageCount - currentStartImageIx);


                        workers[i] = new BackpropWorker(i, iCount, currentStartImageIx, imageOrder, digitLabels, workerbpps[i], testImages, interestedDigits, learningRate, secondLearningRate, momentum, true, new PassFinishedCallback(PassFinished), false);
                        workerThreads[i] = new Thread(new ThreadStart(workers[i].DoWork));
                        workerThreads[i].Start();

                        currentStartImageIx += iCount;
                    }

                    // now do part of the images with the main thread
                    int myCount = Math.Min(batchImageSize, imageCount - currentStartImageIx);
                    BackpropWorker bpw = new BackpropWorker(0, myCount, currentStartImageIx, imageOrder, digitLabels, bpp, testImages, interestedDigits, learningRate, secondLearningRate, momentum, true, null, false);
                    bpw.DoWork();
                    currentStartImageIx += myCount;

                    // now collect all the results from the worker threads 
                    for (int i = 0; i < threads - 1; i++)
                        workerThreads[i].Join();

                    correctCount += bpw.getCorrect();
                    for(int j = 0; j < wrongGuesses.Length; j++)
                        wrongGuesses[j] += bpw.getWrongGuesses()[j];

                    for (int i = 0; i < threads - 1; i++)
                    {
                        correctCount += workerCorrectCount[i];
                        for (int j = 0; j < wrongGuesses.Length; j++)
                            wrongGuesses[j] += workerWrongGuesses[i][j];
                    }

                    // merge the networks
                    bpp.mergeNetworks(workerbpps);

                    // copy the merged network to the networks used by the workers for the next round
                    bpp.copyNetwork(workerbpps);
                }
                
                Console.WriteLine("Single Test Run Time: {0} for {1} features and {2} images", testSW.ElapsedMilliseconds, features.Length, imageCount);

                double trainingAccuracy = ((double)correctCount) / imageCount;
                lastTrainingAccuracy = trainingAccuracy;

                // save the network?
                if ((t + 1) % DFNParams.NETWORKSAVEFREQUENCY == 0)
                {
                    saveNetwork(bpp, t + 1);
                }

                //we do this every time no matter what, it's fast enough now
                //if (t % DFNLibrary.DFNParams.TRAININGOUTPUTFREQ == 0)
                //{
                // output the current training/test stats
                if (trainingAccuracy >= DFNParams.TRAININGSTOPACCURACY)
                {
                    if (DFNParams.SAVEWRONGGUESSES)
                    {
                        // we're done with backprop, output the stats regarding which digits were wrong in training/testing, and output the final features                        
                        if(wrongGuesses.Length == 10)
                            System.IO.File.AppendAllText("aTrainingGuessesWrong.txt", String.Format("{0}:{1}:{2}:{3}:{4}:{5}:{6}:{7}:{8}:{9}", wrongGuesses[0], wrongGuesses[1], wrongGuesses[2], wrongGuesses[3], wrongGuesses[4], wrongGuesses[5], wrongGuesses[6], wrongGuesses[7], wrongGuesses[8], wrongGuesses[9]));
                        else
                            System.IO.File.AppendAllText("aTrainingGuessesWrong.txt", String.Format("{0}:{1}", wrongGuesses[0], wrongGuesses[1]));
                    }
                    if (DFNParams.RECORDFEATURES)
                    {
                        for (int k = 0; k < bpp.allFeatures.Length; k++)
                        {
                            WriteFeature(bpp.allFeatures[k].Weights, imageDirectory + "feature_" + k + "_POST.jpg");
                        }
                    }

                    double testingAccuracy = runBackpropNetworkOnValidationMulti(bpp, features.Length, biasPadding, po, DFNParams.SAVEWRONGGUESSES);
                    System.IO.File.AppendAllText("aPerceptronResults.txt", String.Format("{3}---Training: {0}, Testing: {1}, Training Pass: {2}", trainingAccuracy, testingAccuracy, t, System.DateTime.Now) + Environment.NewLine);
                    System.IO.File.AppendAllText("aTrainingAccuracy.txt", trainingAccuracy + Environment.NewLine);
                    System.IO.File.AppendAllText("aTestingAccuracy.txt", testingAccuracy + Environment.NewLine);
                    Environment.Exit(0);
                }
                else
                {
                    double testingAccuracy = runBackpropNetworkOnValidationMulti(bpp, features.Length, biasPadding, po);
                    System.IO.File.AppendAllText("aPerceptronResults.txt", String.Format("{3}---Training: {0}, Testing: {1}, Training Pass: {2}", trainingAccuracy, testingAccuracy, t, System.DateTime.Now) + Environment.NewLine);
                    System.IO.File.AppendAllText("aTrainingAccuracy.txt", trainingAccuracy + Environment.NewLine);
                    System.IO.File.AppendAllText("aTestingAccuracy.txt", testingAccuracy + Environment.NewLine);                    
                }

                learningRate *= (1 - DFNLibrary.DFNParams.LEARNRATEDECAY);
                //decay second layer rate at the same pace
                secondLearningRate *= (1 - DFNLibrary.DFNParams.LEARNRATEDECAY);

            }

            Console.WriteLine("Full Test Elapse Time: {0} for {1} features and {2} repetitions", sw.ElapsedMilliseconds, features.Length, repeatTestSet);

            double lastValidationAccuracy = runBackpropNetworkOnValidation(bpp, features.Length, biasPadding, po);
            //send back the network, and it's latest work            
            return new Tuple<BackPropPerceptron, double, double>(bpp, lastTrainingAccuracy, lastValidationAccuracy);
        }

        public static void saveNetwork(BackPropPerceptron bpp, int epoch)
        {
            string path = Environment.CurrentDirectory + "/networks";
            Directory.CreateDirectory(path);

            path += "/network_" + epoch + ".xml";

            DFNLibrary.NetworkSaver.saveBPPNetwork(bpp, path);
        }

        public static SingleOutputNetwork[] copyFeatures(SingleOutputNetwork[] features)
        {
            SingleOutputNetwork[] newFeatures = new SingleOutputNetwork[features.Length];
            for (int i = 0; i < features.Length; i++)
            {
                double[] weights = new double[features[i].Weights.Length];
                Array.Copy(features[i].Weights, weights, weights.Length);
                newFeatures[i] = new SingleOutputNetwork(weights, features[i].ActFun, features[i].AllowBackpropWeights);
            }

            return newFeatures;
        }

        public static void PassFinished(int workerNumber, BackPropPerceptron network, int correct, int[] wrong)
        {
            workerbpps[workerNumber] = network;
            workerCorrectCount[workerNumber] = correct;
            workerWrongGuesses[workerNumber] = wrong;
        }        

        #endregion



    }

    class BackpropWorker
    {
        private int workerNumber;
        private int iCount;
        private int currentStartImageIx;
        private List<int> imageOrder;
        private int[] digitLabels;
        private BackPropPerceptron bpp;
        private double[][] images;
        private List<int> interestedDigits;
        private double learningRate;
        private double secondLearningRate;
        private double momentum;
        private int correctCount;
        private int[] wrongGuesses;
        private bool updateNetwork;
        private PassFinishedCallback callback;
        private bool showWrongGuesses;

        public BackpropWorker(int workerNumber, int iCount, int currentStartImageIx, List<int> imageOrder, int[] digitLabels, BackPropPerceptron bpp, double[][] images, List<int> interestedDigits, double learningRate,
            double secondLearningRate, double momentum, bool updateNetwork, PassFinishedCallback callback, bool showWrongGuesses)
        {
            this.workerNumber = workerNumber;
            this.iCount = iCount;
            this.currentStartImageIx = currentStartImageIx;
            this.imageOrder = imageOrder;
            this.digitLabels = digitLabels;
            this.bpp = bpp;
            this.images = images;
            this.interestedDigits = interestedDigits;
            this.learningRate = learningRate;
            this.secondLearningRate = secondLearningRate;
            this.momentum = momentum;
            this.updateNetwork = updateNetwork;
            wrongGuesses = new int[interestedDigits.Count];
            this.callback = callback;
            this.showWrongGuesses = showWrongGuesses;
        }

        public void DoWork()
        {
            for (int image = 0; image < iCount; image++)
            {
                //to read the label, we need to grab the shuffled index (though our feature cache is in order)
                int shuffleIx = imageOrder[currentStartImageIx + image];
                int labelDigit = digitLabels[shuffleIx];

                //activate the network, then correct the weights in a parallel fashion
                bool isCorrect = bpp.ActivateFullNetwork(
                    images[shuffleIx],
                    interestedDigits.IndexOf(labelDigit), // what index should be the highest activated?
                    updateNetwork, //if it's wrong, should we fix it?                    
                    learningRate, //how much do we fix it  
                    secondLearningRate, //how much do we fix the next layer too?
                    momentum, //momentum is if your previous delta was in the same direction, we add part of it
                    DFNLibrary.DFNParams.USE_SPARSE_INITIALIZATION);

                // save this wrong guess to a file
                if (!isCorrect && showWrongGuesses)
                {
                    string outputDirectory = Environment.CurrentDirectory + "/wrongGuesses/";
                    if (!Directory.Exists(outputDirectory))
                        Directory.CreateDirectory(outputDirectory);
                    string outFile = outputDirectory + "A" + labelDigit + "-G" + bpp.LastChosenIx + "-" + shuffleIx + ".png";
                    FullFeatureValidation.WriteImage(images[shuffleIx], outFile, 0);
                }

                //if we got it right, incremenet our correct count
                if (isCorrect)
                    correctCount++;
                else
                    wrongGuesses[labelDigit]++;

                //if(image % 100 == 0)
                //    Console.WriteLine(workerNumber + ":" + image);
            }

            if(callback != null)
                callback(workerNumber, bpp, correctCount, wrongGuesses);
        }

        public int[] getWrongGuesses()
        {
            return wrongGuesses;
        }

        public int getCorrect()
        {
            return correctCount;
        }
    }

    public delegate void PassFinishedCallback(int workerNumber, BackPropPerceptron network, int correct, int[] wrongGuesses);
}
