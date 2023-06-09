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
using SharpNeat.Phenomes.NeuralNets;
using SharpNeat.Phenomes;
using SharpNeat.Genomes.HyperNeat;
using SharpNeat.Core;
using DFNLibrary;
using SharpNeat.Network;

namespace DFN
{
    public class ExperimentLauncher
    {
        static int MAXPARALLELISM = DFNLibrary.DFNParams.THREADS;

        //acyclic networks (feed forward only -- this will be useful in the future for speeding up things on GPUs)
        public static NetworkActivationScheme CPPNActivation = NetworkActivationScheme.CreateAcyclicScheme();
        public static NetworkActivationScheme SubstrateActivation = (DFNLibrary.DFNParams.HiddenNodeSubstrateWidth == 0 ? 
                NetworkActivationScheme.CreateSingleOutputScheme() : NetworkActivationScheme.CreateSingleOutputHiddenScheme());

        //is LEO used? Keep this as a static variable, because we need to use it for perceptron training
        public static bool mnistUseBiasLEO = DFNLibrary.DFNParams.USEBIASLEO;
        public static bool mnistUseLEO = DFNLibrary.DFNParams.USELEO;


        public static void StartExperiment(FeatureNEAT exp, bool onNewThread)
        {
            //lets start the evo algorithm --- on this thread or another
            if (onNewThread)
                exp.EA.StartContinue();
            else
                exp.EA.AlgorithmThreadMethod();
        }
        public static NeatGenomeParameters ExperimentGenomeParams()
        {
            NeatGenomeParameters ngp = new NeatGenomeParameters();
            ngp.ConnectionWeightRange = 3.0;

            //for the love of god, create all connections in our initial CPPN
            ngp.InitialInterconnectionsProportion = 1.0;
            ngp.FeedforwardOnly = true;
            return ngp;
        }

        public static FeatureNEAT CreateNEATExperiment(EventHandler updateEvent = null, EventHandler pauseEvent = null)
        {
            ParallelOptions po = new ParallelOptions();
            //max threads  == 8 -- quad core x 2
            po.MaxDegreeOfParallelism = MAXPARALLELISM;

            NeatEvolutionAlgorithmParameters expParams = new NeatEvolutionAlgorithmParameters();
            //kind of important for this experiment doy!
            expParams.UseNovelty = true;
            expParams.ElitismProportion = .1;

            var ngp = ExperimentGenomeParams();

            //send in our experiment params along with genome params and how activation wil be handled
            //additionally, parallel options are sent for regulating how the multithreading happens inside the EA and eval
            FeatureNEAT featureExperiment = new FeatureNEAT(expParams, ngp, CPPNActivation, SubstrateActivation, po);

            //normalize weights in the network? AYUP. Don't want feature saturation
            bool normalizeWeight = true;

            //usiong leo on outputs?
            bool useLEO = mnistUseLEO;

            //when creating bias node weights, do we use the LEO functionality?
            bool biasUsesLEO = mnistUseBiasLEO;

            //100 to start
            int populationSize = 100;

            IDomain domain = DomainManager.Instance;

            int inputs = domain.getCPPNInputs();
            int outputs = domain.getCPPNOutputs(biasUsesLEO, useLEO);

            //100 CPPNs with 2 inputs, and 3 outputs
            //this includes leo and whether or not the bias is using LEO too
            featureExperiment.InitializeEA(normalizeWeight, populationSize, inputs, outputs, biasUsesLEO, useLEO);

            //now that everything is initialized, we can start the experiment
            //first, we'd like to hook into some callbacks
            // Attach update event listener.
            if (updateEvent != null)
                featureExperiment.EA.UpdateEvent += updateEvent;
            else
                featureExperiment.EA.UpdateEvent += new EventHandler(EA_UpdateEvent);
           
            //register for paused event callbacks
            if (pauseEvent != null)
                featureExperiment.EA.PausedEvent += pauseEvent;
            else
                featureExperiment.EA.PausedEvent += new EventHandler(EA_PausedEvent);

            //this really is only called if you start on another thread
            return featureExperiment;
        }               

        public static IBlackBox[] ParallelDecodeGenomes(List<NeatGenome> evals, ParallelOptions po)
        {
            if (evals.Count == 0)
                return null;

            var ngp = ExperimentGenomeParams();
            CppnGenomeFactory madeUpFactory = new CppnGenomeFactory(evals[0].InputNodeCount, evals[0].OutputNodeCount, FeatureNEAT.CreateLibraryCppn(), ngp);

            //Grab our mnist info
            IDomain domain = DomainManager.Instance;

            //create a decoder for our genomes
            var genomeDecoder = DomainManager.CreateGenomeDecoder(true, domain.getPixelWidth(), domain.getPixelHeight(), CPPNActivation, SubstrateActivation, mnistUseBiasLEO, mnistUseLEO);

            //need to store our decoded genomes please
            var decodedGenomes = new IBlackBox[evals.Count];

            for (int i = 0; i < evals.Count; i++) // not really parallel!  parallel.for crashes here occasionally
            //Parallel.For(0, evals.Count, po, i =>
            {
                if (evals[i].EvaluationInfo == null || evals[i].EvaluationInfo.Behavior == null)
                {
                    //this will create an evaluation object
                    if(evals[i].GenomeFactory == null)                    
                        evals[i].GenomeFactory = madeUpFactory;

                    //then we can decode once we have a proper genome factory
                    decodedGenomes[i] = genomeDecoder.Decode(evals[i]);
                }
            }
            //);

            //all decoded!
            return decodedGenomes;
        }
        public static FitnessInfo[] ParallelGenomeEvaluations(
            IBlackBox[] decodedGenomes, 
            double[][] testImages, 
            int pixelCount, 
            Dictionary<int, Tuple<double,double>> minMaxDictionary,
            bool writeToDictionary,
            ParallelOptions po)
        {
            var allEvaluations = new FitnessInfo[decodedGenomes.Length];

            Parallel.For(0, decodedGenomes.Length, po, i =>
            {
                //if we don't have a behavior, let's set one!
                if (decodedGenomes[i] != null)
                {
                    //this will evaluate the genome -- creating the behavior
                    allEvaluations[i] = DomainManager.EvaluateImages(testImages, pixelCount, decodedGenomes[i], minMaxDictionary, writeToDictionary, i);
                }                
            });

            return allEvaluations;
        }
        
        static void EA_PausedEvent(object sender, EventArgs e)
        {
            Console.WriteLine("EA Paused");
        }
        static bool saveNextGen = false;

        //default update behavior (if not overridden)
        static void EA_UpdateEvent(object sender, EventArgs e)
        {
            var exp = sender as NeatEvolutionAlgorithm<NeatGenome>;

            if(exp == null)
                return;

            Console.WriteLine("EA Update Gen: {0}", exp.CurrentGeneration);

            //Save population and archive every 50 gens
            if((exp.CurrentGeneration + 1) % 50 == 0)
                saveNextGen = true;

            //do we save the population?
            if (saveNextGen)
            {
                //handle saving the novelty archive to file a little bit at a time
                DFNSaver.Instance.savePopulationAndArchive(exp.CurrentChampGenome, exp.GenomeList, exp.noveltyArchives[1].ArchiveList, (int)exp.CurrentGeneration);

                //no saving again till next time, thank you
                saveNextGen = false;
            }

        }

        #region Load and Run Perceptron from Folder
                
        //make sure we're all loaded up
        public static void LoadDomainFiles(string folder, bool createDirectories)
        {
            //only do this if the images aren't loaded
            if (DomainManager.Instance.getAllTestImages() == null)
            {
                DomainManager.Instance.LoadImages(folder);

                if(createDirectories)
                    DFNSaver.Instance.createExperimentalDirectories();
            }
        }              

        #endregion
    }
}
