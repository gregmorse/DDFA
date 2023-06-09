using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.EvolutionAlgorithms;
using SharpNeat.Genomes.Neat;
using SharpNeat.Core;
using SharpNeat.SpeciationStrategies;
using SharpNeat.DistanceMetrics;
using System.Threading.Tasks;
using SharpNeat.EvolutionAlgorithms.ComplexityRegulation;
using SharpNeat.Phenomes;
using SharpNeat.Decoders;
using SharpNeat.Decoders.Neat;
using DFN.Evaluation;
using SharpNeat.Novelty;
using SharpNeat.Network;
using SharpNeat.Genomes.HyperNeat;
using DFNLibrary;

namespace DFN
{
    public class FeatureNEAT
    {
        float PROBADDITIONTHRESHOLD = DFNLibrary.DFNParams.PROBADDITIONTHRESHOLD;
        float FEATUREADDITIONTHRESHOLD = DFNLibrary.DFNParams.FEATUREADDITIONTHRESHOLD;

        ParallelOptions _parallelOptions;
        NeatEvolutionAlgorithm<NeatGenome> _ea;
        NeatEvolutionAlgorithmParameters _eaParams;
        NeatGenomeParameters _neatGenomeParams = new NeatGenomeParameters();
        NetworkActivationScheme _cppnActivationScheme;
        NetworkActivationScheme _substrateActivationScheme;
        CppnGenomeFactory _genomeFactory;
        int _populationSize;

        IGenomeDecoder<NeatGenome, IBlackBox> _genomeDecoder;

        public FeatureNEAT(NeatEvolutionAlgorithmParameters evoParams, 
            NeatGenomeParameters genomeParams,
            NetworkActivationScheme cppnActivationScheme,
            NetworkActivationScheme substrateActivationScheme,
            ParallelOptions pOptions)
        {
            _parallelOptions = pOptions;
            _eaParams = evoParams;
            _neatGenomeParams = genomeParams;
            _cppnActivationScheme = cppnActivationScheme;
            _substrateActivationScheme = substrateActivationScheme;

            //no acyclic networks -- must match activation scheme
            _neatGenomeParams.FeedforwardOnly = _cppnActivationScheme.AcyclicNetwork;
        }

        public NeatEvolutionAlgorithm<NeatGenome> EA { get { return _ea; } }
        public NeatGenomeParameters NeatGenomeParams { get { return _neatGenomeParams; } }
        public IGenomeDecoder<NeatGenome, IBlackBox> GenomeDecoder { get { return _genomeDecoder; } } 

        //where is the step function inside our activation library? Needed for replacing LEO output with step function
        static int stepFuncionActivationID;
        static int bipolarActivationID;
        /// <summary>
        /// Create an IActivationFunctionLibrary for use with CPPNs.
        /// </summary>
        public static IActivationFunctionLibrary CreateLibraryCppn()
        {
            List<ActivationFunctionInfo> fnList = new List<ActivationFunctionInfo>(5);
            fnList.Add(new ActivationFunctionInfo(0, 0.25, Linear.__DefaultInstance));

            //have to set outputs for genomes manually or they go to LINEAR - wtf?
            bipolarActivationID = 1;

            fnList.Add(new ActivationFunctionInfo(bipolarActivationID, 0.25, BipolarSigmoid.__DefaultInstance));
            fnList.Add(new ActivationFunctionInfo(2, 0.25, Gaussian.__DefaultInstance));
            fnList.Add(new ActivationFunctionInfo(3, 0.25, Sine.__DefaultInstance));
            //then we add a step function to our list fo potential activations -- but with 0 probability
            stepFuncionActivationID = 4;
            fnList.Add(new ActivationFunctionInfo(stepFuncionActivationID, 0, StepFunction.__DefaultInstance));
            return new DefaultActivationFunctionLibrary(fnList);
        }

        public void InitializeEA(bool normalizeWeight, int popSize, int inputs, int outputs, bool biasUsesLEO, bool useLEO)
        {
            _genomeFactory = new CppnGenomeFactory(inputs, outputs, CreateLibraryCppn(), _neatGenomeParams);
            _populationSize = popSize;

            //huzzah -- make our initial pop
            List<NeatGenome> genomeList = CreateInitialPopulation(_genomeFactory, normalizeWeight, popSize, inputs, outputs, biasUsesLEO, useLEO);

            //creates the evaluator and what have you
            //initializes the evolutionary algorithm
            _ea = CreateEvolutionAlgorithm(_genomeFactory, genomeList, normalizeWeight, biasUsesLEO, useLEO);
        }

        public static List<NeatGenome> CreateInitialPopulation(CppnGenomeFactory genomeFactory, bool normalizeWeight, int populationSize, int inputs, int outputs, bool biasUsesLEO, bool useLEO)
        {
            // Create an initial population of randomly generated genomes.
            List<NeatGenome> genomeList = new List<NeatGenome>();

            for (var g = 0; g < populationSize; g++)
            {
                //reset beforehand or else it throws an exception
                genomeFactory.InnovationIdGenerator.Reset();

                //create a random generic seed in the 0th generation
                NeatGenome seed = genomeFactory.CreateGenome(0);

                //if we have LEO, we need to replace the random seed genome's LEO node with a step function
                if (useLEO || biasUsesLEO)
                {

                    //we call a helper function to tell us where all the LEO indexes are for our initial seed
                    //we may have many depending on the substrate setup (and the multi-modal setup)

                    IDomain domain = DomainManager.Instance;

                    List<int> leoIxs = domain.LeoIndexes(biasUsesLEO, useLEO);
                    string stepFunctionID = StepFunction.__DefaultInstance.FunctionId;

                    NeuronGeneList ngl = new NeuronGeneList(seed.NeuronGeneList.Capacity);
                    for (var i = 0; i < seed.NeuronGeneList.Count; i++)
                    {
                        var seedNeuron = seed.NeuronGeneList[i];

                        //for leo, we MUST use a step function 
                        //therefore, we copy a generic seed object from our factory
                        //until we get to the node that's the leo node -- there we replace the activation with the step function ID
                        if (!leoIxs.Contains(i - (inputs + 1)))
                        {
                            if (i < inputs + 1) //linear activation for inputs and bias -- not that it matters
                                ngl.Add(new NeuronGene(seedNeuron.Id, seedNeuron.NodeType, 0));
                            else
                                ngl.Add(new NeuronGene(seedNeuron.Id, seedNeuron.NodeType, bipolarActivationID));
                        }
                        else
                        {
                            ngl.Add(new NeuronGene(seedNeuron.Id, seedNeuron.NodeType, stepFuncionActivationID));
                        }

                        //need to copy source and target manually -- this is annoying because I can't change activation from outside the neurongene
                        //that's stupid
                        foreach (var x in seedNeuron.SourceNeurons)
                            ngl[i].SourceNeurons.Add(x);

                        foreach (var x in seedNeuron.TargetNeurons)
                            ngl[i].TargetNeurons.Add(x);
                    }

                    //now we have a generic LEO seed -- no specific seeded LEO structure -- just initialized like every other output
                    seed = genomeFactory.CreateGenome(seed.Id,
                                                    seed.BirthGeneration,
                                                        ngl,
                                                        seed.ConnectionGeneList,
                                                        seed.InputNeuronCount,
                                                        seed.OutputNeuronCount,
                                                        false);

                }

                //add the seed to our list, continue onwards!
                genomeList.Add(seed);
            }

            return genomeList;

         
        }


        /// <summary>
        /// Create and return a NeatEvolutionAlgorithm object ready for running the NEAT algorithm/search. Various sub-parts
        /// of the algorithm are also constructed and connected up.
        /// This overload accepts a pre-built genome population and their associated/parent genome factory.
        /// </summary>
        NeatEvolutionAlgorithm<NeatGenome> CreateEvolutionAlgorithm(CppnGenomeFactory genomeFactory, List<NeatGenome> genomeList, bool normalizeWeight, bool biasUsesLEO, bool useLEO)
        {
            // Create distance metric. Mismatched genes have a fixed distance of 10; for matched genes the distance is their weigth difference.
            IDistanceMetric distanceMetric = new ManhattanDistanceMetric(1.0, 0.0, 10.0);
            ISpeciationStrategy<NeatGenome> speciationStrategy = new ParallelKMeansClusteringStrategy<NeatGenome>(distanceMetric, _parallelOptions);

            // Create complexity regulation strategy. -- make it absolute and basically no level that will ever be reached.
            //complexification shall not be stopped! mwahahaha
            IComplexityRegulationStrategy complexityRegulationStrategy = new DefaultComplexityRegulationStrategy(ComplexityCeilingType.Absolute, 1000000);
                
            // Create the evolution algorithm.
            NeatEvolutionAlgorithm<NeatGenome> ea = new NeatEvolutionAlgorithm<NeatGenome>(_eaParams, speciationStrategy, complexityRegulationStrategy);

            //create our archives -- one for evolution, the other that accumulates features for us
            NoveltyArchive<NeatGenome> mainArchive = new NoveltyArchive<NeatGenome>(true, PROBADDITIONTHRESHOLD, _parallelOptions, DFNParams.GENERATE_RANDOM_CPPNS, DFNParams.RANDOM_CPPN_ADDITION_FREQUENCY);            
            NoveltyArchive<NeatGenome> featureCollection = new NoveltyArchive<NeatGenome>(false, FEATUREADDITIONTHRESHOLD, _parallelOptions, DFNParams.GENERATE_RANDOM_CPPNS, DFNParams.RANDOM_CPPN_ADDITION_FREQUENCY);

            //this is the order that the archives will be processed during evolution
            ea.AddNoveltyArchive(mainArchive);
            ea.AddNoveltyArchive(featureCollection);

            IDomain domain = DomainManager.Instance;

            // Create IBlackBox evaluator.
            IDomainEvaluator evaluator = domain.createEvaluator();

            //create our decoder
            _genomeDecoder = DomainManager.CreateGenomeDecoder(normalizeWeight, domain.getPixelWidth(), domain.getPixelHeight(),
                _cppnActivationScheme, _substrateActivationScheme, biasUsesLEO, useLEO);

            // Create a genome list evaluator. This packages up the genome decoder with the genome evaluator.
            IGenomeListEvaluator<NeatGenome> innerEvaluator = new ParallelGenomeListEvaluator<NeatGenome, IBlackBox>(_genomeDecoder, evaluator, _parallelOptions);

            // Wrap the list evaluator in a 'selective' evaulator that will only evaluate new genomes. That is, we skip re-evaluating any genomes
            // that were in the population in previous generations (elite genomes). This is determined by examining each genome's evaluation info object.
            IGenomeListEvaluator<NeatGenome> selectiveEvaluator = new SelectiveGenomeListEvaluator<NeatGenome>(
                                                                                    innerEvaluator,
                                                                                    SelectiveGenomeListEvaluator<NeatGenome>.CreatePredicate_OnceOnly());
            // Initialize the evolution algorithm.
            ea.Initialize(selectiveEvaluator, genomeFactory, genomeList);

            // Finished. Return the evolution algorithm
            return ea;
        }        

        static int updateCount = 0;
        static bool checkNextTime = false;
        static uint lastSaveGen = 0;

        //standard threshold based collection here
        public static void EvolutionUpdate_CollectFeatures_Threshold(NeatEvolutionAlgorithm<NeatGenome> exp)
        {
            Console.WriteLine("EA Update Gen: {0}", exp.CurrentGeneration);

            //okay, we're in update mode, let's grab our novelty archive and do some perceptron training on this thread or another!

            //Save population and archive every 50 gens
            //do we save the population?
            if ((exp.CurrentGeneration - lastSaveGen) > DFNParams.GENERATIONSPERSAVE)
            {
                //handle saving the novelty archive to file a little bit at a time
                DFNSaver.Instance.savePopulationAndArchive(exp.CurrentChampGenome, exp.GenomeList, exp.noveltyArchives[1].ArchiveList, (int)exp.CurrentGeneration);

                //no saving again till next time, thank you
                lastSaveGen = exp.CurrentGeneration;
            }

            updateCount++;
        }


        static Dictionary<uint, NeatGenome> allGenomes = new Dictionary<uint, NeatGenome>();
        static List<NeatGenome> newGenomes = new List<NeatGenome>();               
    }
}
