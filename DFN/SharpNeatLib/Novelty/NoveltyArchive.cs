using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Genomes.Neat;
using System.Threading.Tasks;
using SharpNeat.Core;

namespace SharpNeat.Novelty
{
    public class NoveltyArchive<TGenome> where TGenome : class, IGenome<TGenome>
    {
        public static bool GENERATE_RANDOM_CPPNS; 
        public static double RANDOM_CPPN_ADDITION_FREQUENCY; 

        private List<TGenome> archiveList;
        private NoveltyCalc<TGenome> noveltyCalc;
        private ParallelOptions parallelOptions;
        private HashSet<uint> addedGenomes = new HashSet<uint>();

        private ArchiveCorrelation corrArchive;

        // mainarchive = true means use probabilistic add and threshold specifies the % chance of adding a genome to the archive (0.05 = 5%)
        // mainarchive = false means use a threshold based addition.  use this for the feature collector
        public NoveltyArchive(bool _mainArchive, float _threshold, ParallelOptions _parallelOptions, bool generateRandomCPPNs = false, double additionFrequency = 0)
        {
            noveltyCalc = new NoveltyCalc<TGenome>(_mainArchive, _threshold, _parallelOptions);
            archiveList = new List<TGenome>();
            parallelOptions = _parallelOptions;
            NoveltyArchive<TGenome>.GENERATE_RANDOM_CPPNS = generateRandomCPPNs;
            NoveltyArchive<TGenome>.RANDOM_CPPN_ADDITION_FREQUENCY = additionFrequency;
        }

        public void evaluatePopulation(IList<TGenome> population, bool addToArchive)
        {            
            noveltyCalc.evaluatePopulation(population, this, addToArchive);
        }

        // adds a copy of the genome to the archive list
        public void addGenome(TGenome genome)
        {
            //if we already got you -- don't add this genome AGAIN to the archive
            if (addedGenomes.Contains(genome.Id))
                return;
            
            //if you're new, welcome friend :)            
            addedGenomes.Add(genome.Id);

            IGenome<NeatGenome> ng = (IGenome<NeatGenome>)genome;
            NeatGenome clone = new NeatGenome((NeatGenome)ng, ng.Id, ng.BirthGeneration);

            if (ng.EvaluationInfo != null)
            {
                //copy fitness for outside looking in stuff
                clone.EvaluationInfo.SetFitness(ng.EvaluationInfo.Fitness);

                //need to cache the behavior for later training!
                clone.EvaluationInfo.SetBehavior(ng.EvaluationInfo.Behavior);


                clone.EvaluationInfo.SetBinaryActivation(ng.EvaluationInfo.BinaryActivation);

                clone.EvaluationInfo.SetObjectives(ng.EvaluationInfo.Objectives);

                //clone the novelty info as well!
                if (ng.EvaluationInfo.NoveltyInformation != null)
                {
                    clone.EvaluationInfo.SetNoveltyInformation(new NoveltyEvaluationInfo<NeatGenome>
                        (ng.EvaluationInfo.NoveltyInformation.NoveltyScore, ng.EvaluationInfo.NoveltyInformation.NearestNeighbors));
                }
            }
          

            archiveList.Add(clone as TGenome);
        }
       
        public List<TGenome> ArchiveList
        {
            get { return archiveList; }
        }     
    }
}
