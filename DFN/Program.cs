using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using SharpNeat.EvolutionAlgorithms;
using SharpNeat.Genomes.Neat;
using SharpNeat.Decoders;
using DFN.Evaluation;


using DFN.Validation;
using DFNLibrary;
using System.Reflection;
using System.IO;
using SharpNeat.Phenomes;
using SharpNeat.Network;
using SharpNeat.Novelty;

namespace DFN
{
    class Program
    {
        int MAXPARALLELISM = DFNLibrary.DFNParams.THREADS;        

        static FeatureNEAT experiment;
        static void Main(string[] args)
        {
            if (args.Length == 0)
            {
                return;
            }
            string mode = args[0];

            if (mode == "backprop")
            {
                // set the domain manager based on which domain we're using
                if (DFNLibrary.DFNParams.DOMAIN == DFNLibrary.Domain.MNIST)
                    DomainManager.Instance = MNISTManager.Instance;
                else if (DFNLibrary.DFNParams.DOMAIN == DFNLibrary.Domain.CIFAR)
                    DomainManager.Instance = CIFARManager.Instance;                

                string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                               
                string folderToLoad = currentAssemblyDirectoryName + "/" + DFNParams.DATA_TO_TRAIN_FOLDER;

                var dirs = Directory.GetDirectories(folderToLoad);
                if (dirs.Any(x => x.IndexOf("novelty") != -1))
                {
                    //we're in the folder before novelty
                    folderToLoad += "\\novelty";
                }
                //how parallel do we get, yo?
                ParallelOptions po = new ParallelOptions();
                po.MaxDegreeOfParallelism = DFNParams.THREADS;

                List<int> digitsInLoaded;

                //load all our features, please!
                var loadedGenomes = DFNSaver.Instance.loadUniquePopulationInChunks(folderToLoad, out digitsInLoaded);
                loadedGenomes = loadedGenomes.OrderBy(si => si.BirthGeneration).ToList(); // sort the features by their birth generation

                SingleOutputNetwork[] decodedSingleNetworks;
                if (DFNParams.TEST_RANDOM_BP_FEATURES)
                {
                    var featureCount = loadedGenomes.Count;

                    Random r = new Random();
                    double smallRandom = 4 * Math.Sqrt(6.0 / featureCount);

                    decodedSingleNetworks = new SingleOutputNetwork[featureCount];

                    for (int i = 0; i < featureCount; i++)
                    {
                        double[] rWeights = new double[DomainManager.Instance.getTotalInputs()];

                        if (DFNParams.USE_SPARSE_INITIALIZATION)
                        {
                            int connectedNeurons = 15;
                            smallRandom = 4 * Math.Sqrt(6.0 / connectedNeurons);
                            for (int j = 0; j < connectedNeurons; j++)
                            {
                                int index = r.Next(rWeights.Length);
                                rWeights[index] = 2 * (smallRandom * r.NextDouble()) - smallRandom;
                            }
                        }
                        else
                        {
                            double subInterval = 2 * smallRandom / featureCount;
                            double featureIntervalStart = subInterval * i - smallRandom;

                            for (int w = 0; w < rWeights.Length; w++)
                                //rWeights[w] = r.NextDouble() * subInterval + featureIntervalStart; // "clever" initialization - as described in "Interval Based Weight Initialization Method for Sigmoidal Feedforward Artificial Neural Networks"
                                rWeights[w] = 2 * (smallRandom * r.NextDouble()) - smallRandom;  // standard initialization - uniform distribution
                        }

                        decodedSingleNetworks[i] = new SingleOutputNetwork(rWeights, SteepenedSigmoid.__DefaultInstance, null);
                    }
                }
                else
                {
                    //now we need to convert our genomes into blackboxes -- then to single output networks

                    //decode our genomes in parallel
                    IBlackBox[] decodedGenomes = ExperimentLauncher.ParallelDecodeGenomes(loadedGenomes, po);

                    List<SingleOutputNetwork> allFeatures = new List<SingleOutputNetwork>();
                    //[] decodedSingleNetworks = new SingleOutputNetwork[decodedGenomes.Length];
                    for (int i = 0; i < decodedGenomes.Length; i++)
                        if (decodedGenomes[i] != null)
                            allFeatures.Add(decodedGenomes[i] as SingleOutputNetwork);

                    decodedSingleNetworks = allFeatures.ToArray();
                }

                //load the mnist files please! Needs to be the same mnist loading as our experiment
                ExperimentLauncher.LoadDomainFiles(null, false);
               
                FullFeatureValidation.BackpropCreateAndTrainNetworkMulti(DFNParams.PERCEPTRONITERATIONS, decodedSingleNetworks, DFNParams.IMAGEBATCHES, po);
            }
            else if (mode == "collection")
            {


                // set the domain manager based on which domain we're using
                if (DFNLibrary.DFNParams.DOMAIN == DFNLibrary.Domain.MNIST)
                    DomainManager.Instance = MNISTManager.Instance;
                else if (DFNLibrary.DFNParams.DOMAIN == DFNLibrary.Domain.CIFAR)
                    DomainManager.Instance = CIFARManager.Instance;

                //load up our images (providing a path for loading)
                DomainManager.LoadImages();

                //create where to save our information
                DFNSaver.Instance.createExperimentalDirectories();

                experiment = ExperimentLauncher.CreateNEATExperiment(EA_UpdateEvent_Threshold, EA_PausedEvent);

                //start the experiment here
                ExperimentLauncher.StartExperiment(experiment, false);
            }

        }

        static void currentDomain_FirstChanceException(object sender, System.Runtime.ExceptionServices.FirstChanceExceptionEventArgs e)
        {
            Console.WriteLine(e.Exception.StackTrace);
        }

        static void currentDomain_UnhandledException(object sender, UnhandledExceptionEventArgs e)
        {
            Exception ex = e.ExceptionObject as Exception;
            if(ex != null)
                Console.WriteLine(ex.StackTrace);
        }

        static void EA_PausedEvent(object sender, EventArgs e)
        {
            Console.WriteLine("EA Paused");
        }

        public delegate void PerceptronTrainingCall(FeatureNEAT experiment);

        static bool isRunningPerceptronTest = false;
      
        static void EA_UpdateEvent_Threshold(object sender, EventArgs e)
        {
            var exp = sender as NeatEvolutionAlgorithm<NeatGenome>;
            FeatureNEAT.EvolutionUpdate_CollectFeatures_Threshold(exp);
        }
    }
}
