using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Threading;
using SharpNeat.Genomes.Neat;
using SharpNeat.Core;


namespace SharpNeat.Novelty
{
    //Paul - Sorry, this is truly an abomination, but I can't figure out a better way to inject these properties for now
    public class NoveltyAdditions
    {
        public static bool UseSetDistanceModifier;
        public static bool UseSetOverlapFeatureCollection;
        public static double SetOverlapFeatureThreshold;
        public static bool ModifyNoveltyWithHeatMap;
        public static bool ReplaceArchiveWithHeatMap;
        public static bool USEIMAGEWISENOVELTY;
        public static int IMAGEWISEK;
    }    

    public class NoveltyCalc<TGenome> where TGenome : class, IGenome<TGenome>
    {
        // number of nearest neighbors for the novelty calculation
        public const int KNEAREST = 20;        

        // minimum genome fitness
        private const float MINFITNESS = 0.000001f;        

        private ParallelOptions parallelOptions;

        // true = probabilistic add (standard archive for calculating fitness)
        // false = threshold based add (feature collection archive)
        private bool probAdd;

        // threshold for adding a genome to the archive
        // if probabilistic, number should be between 0 and 1 and represents probability of adding to the archive (0.05 = 5% chance)
        // if based on min distance, then threshold is the minimum value for adding a new member to the archive
        private float addThreshold;

        public float[] heatMapBehavior;
        public float[] heatMapSum;
        public HashSet<uint> includedHeatMapGenomes;

        // necessary for addition to a distance threshold based archive
        private Semaphore sem;

        // list of image outputs used as a cache for image-wise novelty calculation
        public List<float>[] imageOutputs;

        #region Get/Set Use Second Behavior
        /// <summary>
        /// Does the archive calculate two behavior vectors?
        /// </summary>
        public bool UseSecondBehavior
        {
            get;
            set;
        }

        #endregion
        public NoveltyCalc(bool _probAdd, float _addThreshold, ParallelOptions _parallelOptions)
        {
            probAdd = _probAdd;
            addThreshold = _addThreshold;
            parallelOptions = _parallelOptions;
            sem = new Semaphore(1, 1);
            imageOutputs = new List<float>[60000];
            for (int i = 0; i < 60000; i++)
                imageOutputs[i] = new List<float>();
        }

        void updateArchiveHeatMap(IList<TGenome> population, NoveltyArchive<TGenome> archive)
        {
            if (heatMapBehavior == null)
            {
                TGenome firstNotNull = archive.ArchiveList.Count == 0  ?
                    population.First(x => x != null && x.EvaluationInfo.Behavior != null) :
                    archive.ArchiveList.First(x => x != null && x.EvaluationInfo.Behavior != null);

                heatMapSum = new float[firstNotNull.EvaluationInfo.Behavior.Length];
                heatMapBehavior = new float[firstNotNull.EvaluationInfo.Behavior.Length];

                includedHeatMapGenomes = new HashSet<uint>();
            }

            if (archive.ArchiveList.Count == 0)
                return;

            bool addedNew = false;
            var aList = archive.ArchiveList;
            for (int i = 0; i < aList.Count; i++)
            {
                var tgenome = aList[i];
                if (tgenome != null && !includedHeatMapGenomes.Contains(tgenome.Id))
                {
                    //well, we should addd this genome
                    var behavior = tgenome.EvaluationInfo.Behavior;

                    //if we have a behavior, add it to our heat map
                    if (behavior != null)
                    {
                        //add to our sum object
                        for (int b = 0; b < behavior.Length; b++)
                            heatMapSum[b] += behavior[b];

                        addedNew = true;
                        includedHeatMapGenomes.Add(tgenome.Id);
                    }
                }
            }


            if (addedNew)
            {
                //grab the min/max of our sums
                float min = heatMapSum.Min();
                float max = heatMapSum.Max();

                float range = max - min;

                //if min = max then this will 0 out anyways when you remove the min
                if (range == 0)
                    range = 1;

                //now we calculate the range -- and we invert the whole thing -- since we're calculate the distance -- we want the opposite 
                //that is if you have high activation and a high heatmap -- we don't want you! we want low activation on high heat map, and high activation on low heat map
                for (int b = 0; b < heatMapBehavior.Length; b++)
                    heatMapBehavior[b] = 1.0f - ((heatMapSum[b] - min)/range);
            }
        }

        float[,] calculateFloatDistances(IList<TGenome> population, NoveltyArchive<TGenome> archive)
        {
            // calc distance between all population members and pop members vs archive
            float[,] distances = new float[population.Count, population.Count + archive.ArchiveList.Count];

            if(NoveltyAdditions.ModifyNoveltyWithHeatMap || NoveltyAdditions.ReplaceArchiveWithHeatMap)
            {
                updateArchiveHeatMap(population, archive);            
            }

            if (!NoveltyAdditions.UseSetOverlapFeatureCollection && !NoveltyAdditions.ModifyNoveltyWithHeatMap)
            {
                // run the distance calculations in parallel
                Parallel.For(0, population.Count, parallelOptions, i =>
                {
                    for (int j = 0; j < population.Count; j++)
                    {
                        if (i < j) // don't need to calc distance from a genome to itself, or for the "bottom left half" of the array
                            distances[i, j] = calcMinDistance(population[i].EvaluationInfo.Behavior, population[j].EvaluationInfo.Behavior);
                    }
                    for (int j = 0; j < archive.ArchiveList.Count; j++)
                    {
                        // must fill this entire array, there is no reuse possible in this portion
                        distances[i, j + population.Count] = calcMinDistance(population[i].EvaluationInfo.Behavior, archive.ArchiveList[j].EvaluationInfo.Behavior);
                    }
                });
            }
            else if ((NoveltyAdditions.ModifyNoveltyWithHeatMap || NoveltyAdditions.ReplaceArchiveWithHeatMap) && (!NoveltyAdditions.UseSetOverlapFeatureCollection))
            {
                // run the distance calculations in parallel -- this time with some binar information
                Parallel.For(0, population.Count, parallelOptions, i =>
                {
                    //if we don't modify, make this 0 - causing no adjustment to distances
                    float heatMapModifier = NoveltyAdditions.ModifyNoveltyWithHeatMap ? calcHeatMapDistanceFraction(population[i].EvaluationInfo.Behavior, heatMapBehavior) : 0.0f;

                    for (int j = 0; j < population.Count; j++)
                    {
                        if (i < j) // don't need to calc distance from a genome to itself, or for the "bottom left half" of the array
                        {
                            float dist = calcMinDistance(population[i].EvaluationInfo.Behavior, population[j].EvaluationInfo.Behavior);
                            dist *= (1 + heatMapModifier);
                            distances[i, j] = dist;
                        }
                    }

                    if (!NoveltyAdditions.ReplaceArchiveWithHeatMap)
                    {
                        for (int j = 0; j < archive.ArchiveList.Count; j++)
                        {
                            // must fill this entire array, there is no reuse possible in this portion
                            float dist = calcMinDistance(population[i].EvaluationInfo.Behavior, archive.ArchiveList[j].EvaluationInfo.Behavior);
                            dist *= (1 + heatMapModifier);
                            distances[i, j + population.Count] = dist;
                        }
                    }
                });
            }
            else if (NoveltyAdditions.UseSetOverlapFeatureCollection && !(NoveltyAdditions.ModifyNoveltyWithHeatMap || NoveltyAdditions.ReplaceArchiveWithHeatMap))
            {
                // run the distance calculations in parallel -- this time with some binar information
                Parallel.For(0, population.Count, parallelOptions, i =>
                {
                    for (int j = 0; j < population.Count; j++)
                    {
                        if (i < j) // don't need to calc distance from a genome to itself, or for the "bottom left half" of the array
                        {
                            float dist = calcMinDistance(population[i].EvaluationInfo.Behavior, population[j].EvaluationInfo.Behavior);
                            dist *= (1 + calcSetDifferences(population[i].EvaluationInfo.BinaryActivation, population[j].EvaluationInfo.BinaryActivation));
                            distances[i, j] = dist;
                        }
                    }
                    for (int j = 0; j < archive.ArchiveList.Count; j++)
                    {
                        // must fill this entire array, there is no reuse possible in this portion
                        float dist = calcMinDistance(population[i].EvaluationInfo.Behavior, archive.ArchiveList[j].EvaluationInfo.Behavior);
                        dist *= (1 + calcSetDifferences(population[i].EvaluationInfo.BinaryActivation, archive.ArchiveList[j].EvaluationInfo.BinaryActivation));
                        distances[i, j + population.Count] = dist;
                    }
                });
            }
            else
            {
                // run the distance calculations in parallel -- this time with some binar information
                Parallel.For(0, population.Count, parallelOptions, i =>
                {
                    //if we don't modify, make this 0 - causing no adjustment to distances
                    float heatMapModifier = NoveltyAdditions.ModifyNoveltyWithHeatMap ? calcHeatMapDistanceFraction(population[i].EvaluationInfo.Behavior, heatMapBehavior) : 0.0f;

                    for (int j = 0; j < population.Count; j++)
                    {
                        if (i < j) // don't need to calc distance from a genome to itself, or for the "bottom left half" of the array
                        {
                            float dist = calcMinDistance(population[i].EvaluationInfo.Behavior, population[j].EvaluationInfo.Behavior);
                            dist *= (1 + calcSetDifferences(population[i].EvaluationInfo.BinaryActivation, population[j].EvaluationInfo.BinaryActivation));
                            dist *= (1 + heatMapModifier);
                            distances[i, j] = dist;
                        }
                    }

                    if (!NoveltyAdditions.ReplaceArchiveWithHeatMap)
                    {
                        for (int j = 0; j < archive.ArchiveList.Count; j++)
                        {
                            // must fill this entire array, there is no reuse possible in this portion
                            float dist = calcMinDistance(population[i].EvaluationInfo.Behavior, archive.ArchiveList[j].EvaluationInfo.Behavior);
                            dist *= (1 + calcSetDifferences(population[i].EvaluationInfo.BinaryActivation, archive.ArchiveList[j].EvaluationInfo.BinaryActivation));
                            dist *= (1 + heatMapModifier);
                            distances[i, j + population.Count] = dist;
                        }
                    }
                });

            }
           
            return distances;
        }        

        public void evaluatePopulation(IList<TGenome> population, NoveltyArchive<TGenome> archive, bool addToArchive)
        {
            if (probAdd)
            {
                if (!NoveltyAdditions.USEIMAGEWISENOVELTY)
                {
                    //go ahead and calculate the distances in parallel -- some options may change
                    float[,] distances = calculateFloatDistances(population, archive);

                    // determine each member's novelty in parallel
                    Parallel.For(0, population.Count, parallelOptions, i =>
                        {
                            List<Tuple<int, float>> sortedDistances = new List<Tuple<int, float>>();
                            for (int j = 0; j < population.Count + archive.ArchiveList.Count; j++)
                            {
                                // if i = j, don't add the distance as it will be 0
                                // if i > j, we need to check j, i as only the upper right half of the array will have values
                                if (i < j)
                                    sortedDistances.Add(new Tuple<int, float>(j, distances[i, j]));
                                else if (i > j)
                                    sortedDistances.Add(new Tuple<int, float>(j, distances[j, i]));
                            }

                            // sort the resulting list of distances and take the sum of the k smallest values for the novelty score for this genome
                            sortedDistances.Sort((x, y) => x.Item2.CompareTo(y.Item2));
                            float noveltyScore = 0;

                            //can't have more neighbors than there are individuals
                            int nearestNeighbors = Math.Min(KNEAREST, sortedDistances.Count);

                            for (int j = 0; j < nearestNeighbors; j++)
                            {
                                noveltyScore += sortedDistances[j].Item2;
                            }

                            //in the event that you were an empty genome, and you never had a novelty behavior -- 
                            //you are total garbage, so you need to have 0 novelty -- eradicated!
                            if (population[i].EvaluationInfo.Behavior == null)
                                noveltyScore = 0;

                            population[i].EvaluationInfo.SetFitness(Math.Max(noveltyScore, MINFITNESS)); // DFN-TODO: Paul, is this the correct way to set the fitness?

                            List<TGenome> nearestIds = new List<TGenome>();
                            //save our novelty info from this
                            for (int g = 0; g < nearestNeighbors; g++)
                            {
                                int popIx = sortedDistances[g].Item1;
                                if (popIx < population.Count)
                                    nearestIds.Add(population[popIx]);
                                else
                                    nearestIds.Add(archive.ArchiveList[popIx - population.Count]);
                            }

                            //save our novlety info!
                            population[i].EvaluationInfo.SetNoveltyInformation(new NoveltyEvaluationInfo<TGenome>(noveltyScore, nearestIds));


                        });

                    if (UseSecondBehavior)
                    {
                        // calc distance between all population members and pop members vs archive
                        float[,] secondDistances = new float[population.Count, population.Count + archive.ArchiveList.Count];

                        // run the distance calculations in parallel
                        Parallel.For(0, population.Count, parallelOptions, i =>
                        {
                            for (int j = 0; j < population.Count; j++)
                            {
                                if (i < j) // don't need to calc distance from a genome to itself, or for the "bottom left half" of the array
                                    secondDistances[i, j] = calcMinDistance(population[i].EvaluationInfo.SecondBehavior, population[j].EvaluationInfo.SecondBehavior);
                            }
                            for (int j = 0; j < archive.ArchiveList.Count; j++)
                            {
                                // must fill this entire array, there is no reuse possible in this portion
                                secondDistances[i, j + population.Count] = calcMinDistance(population[i].EvaluationInfo.SecondBehavior, archive.ArchiveList[j].EvaluationInfo.SecondBehavior);
                            }
                        });

                        // determine each member's novelty in parallel
                        Parallel.For(0, population.Count, parallelOptions, i =>
                        {
                            List<Tuple<int, float>> sortedDistances = new List<Tuple<int, float>>();
                            for (int j = 0; j < population.Count + archive.ArchiveList.Count; j++)
                            {
                                // if i = j, don't add the distance as it will be 0
                                // if i > j, we need to check j, i as only the upper right half of the array will have values
                                if (i < j)
                                    sortedDistances.Add(new Tuple<int, float>(j, distances[i, j]));
                                else if (i > j)
                                    sortedDistances.Add(new Tuple<int, float>(j, distances[j, i]));
                            }

                            // sort the resulting list of distances and take the sum of the k smallest values for the novelty score for this genome
                            sortedDistances.Sort((x, y) => x.Item2.CompareTo(y.Item2));
                            float noveltyScore = 0;

                            //can't have more neighbors than there are individuals
                            int nearestNeighbors = Math.Min(KNEAREST, sortedDistances.Count);

                            for (int j = 0; j < nearestNeighbors; j++)
                            {
                                noveltyScore += sortedDistances[j].Item2;
                            }

                            //in the event that you were an empty genome, and you never had a novelty behavior -- 
                            //you are total garbage, so you need to have 0 novelty -- eradicated!
                            if (population[i].EvaluationInfo.SecondBehavior == null)
                                noveltyScore = 0;

                            var novelInfo = population[i].EvaluationInfo.NoveltyInformation;
                            //novel info shouldn't be null!
                            if (novelInfo != null)
                            {
                                novelInfo.AlternateNovelty = noveltyScore;

                                //save our novelty info from this
                                List<TGenome> nearestIds = new List<TGenome>();
                                for (int g = 0; g < nearestNeighbors; g++)
                                {
                                    int popIx = sortedDistances[g].Item1;
                                    if (popIx < population.Count)
                                        nearestIds.Add(population[popIx]);
                                    else
                                        nearestIds.Add(archive.ArchiveList[popIx - population.Count]);
                                }
                                novelInfo.AlternateNeighbors = nearestIds;

                            }

                            //double currentFitness = population[i].EvaluationInfo.Fitness;

                            //population[i].EvaluationInfo.SetFitness(Math.Max(MINFITNESS, noveltyScore));//currentFitness + noveltyScore));
                            //population[i].EvaluationInfo.SetFitness(Math.Max(MINFITNESS, currentFitness + 1000*noveltyScore));
                        });
                    }
                }
                else
                {                    
                    // use the image-wise novelty calculation method

                    // dummy nearest IDs, there's no sense of nearest neighbor with this algorithm
                    List<TGenome> nearestIds = new List<TGenome>();                    

                    //for(int i = 0; i < population.Count; i++)
                    Parallel.For(0, population.Count, parallelOptions, i =>
                        {
                            if (archive.ArchiveList.Count == 0)
                            {
                                // nobody in archive to compare against, set everyone to the same value for now
                                population[i].EvaluationInfo.SetNoveltyInformation(new NoveltyEvaluationInfo<TGenome>(1, nearestIds));
                            }
                            else
                            {                                
                                population[i].EvaluationInfo.SetNoveltyInformation(new NoveltyEvaluationInfo<TGenome>(Math.Max(calculateImageWiseNovelty(population[i].EvaluationInfo.Behavior), MINFITNESS), nearestIds));                                
                            }
                        });
                }                

                if (addToArchive)
                {
                    Random r = new Random();
                    // check the population for addition to the archive
                    // not using parallel here as adding to our list is not thread safe.  this piece of code should be quite trivial in terms of time complexity anyway
                    for (int i = 0; i < population.Count; i++)
                    {
                        if (r.NextDouble() < addThreshold)
                        {
                            archive.addGenome(population[i]);
                            if (NoveltyAdditions.USEIMAGEWISENOVELTY)
                            {
                                if(population[i].EvaluationInfo != null && population[i].EvaluationInfo.Behavior != null)
                                    for (int j = 0; j < 60000; j++)
                                        imageOutputs[j].Add(population[i].EvaluationInfo.Behavior[j]);
                            }
                        }
                    }
                }
            }
            else
            {
                Random r = new Random();
                double threshold = NoveltyArchive<TGenome>.RANDOM_CPPN_ADDITION_FREQUENCY / population.Count;
                
                if (NoveltyArchive<TGenome>.GENERATE_RANDOM_CPPNS)
                {
                    for (int i = 0; i < population.Count; i++)
                    {
                        if (r.NextDouble() < threshold)
                            archive.addGenome(population[i]);
                    }
                    if (archive.ArchiveList.Count > 3500)
                        System.Environment.Exit(0);
                }
                else
                {
                    if (addThreshold != 0)
                    {
                        // must be careful with this parallelization as each time an individual is added to the archive, it changes the calculation for all other individuals
                        Parallel.For(0, population.Count, parallelOptions, i =>
                            {
                                // if archive is empty, start it off with this guy
                                if (archive.ArchiveList.Count == 0)
                                {
                                    lock (archive.ArchiveList)
                                    {
                                        //was weirdly adding things that were nnull
                                        if (archive.ArchiveList.Count == 0 && population[i] != null && addToArchive)
                                            archive.addGenome(population[i]);
                                    }
                                }
                                else
                                {
                                    bool addSuccessful = false;
                                    while (!addSuccessful)
                                    {
                                        int archiveCount = archive.ArchiveList.Count;
                                        // find the nearest neighbor in the archive
                                        float minDistance = float.MaxValue;
                                        for (int j = 0; j < archiveCount; j++)
                                        {
                                            //it's possible that when we're reading the archive, some other thread is adding the object 
                                            //to fix this, we simply don't bother checking against in progress adds
                                            if (archive.ArchiveList[j] != null)
                                            {
                                                float distance = NoveltyAdditions.UseSetOverlapFeatureCollection ? //if we use set feature collection, grab our set overlap amount
                                                        calcSetDifferences(population[i].EvaluationInfo.BinaryActivation, archive.ArchiveList[j].EvaluationInfo.BinaryActivation)
                                                        :
                                                        calcMinDistance(population[i].EvaluationInfo.Behavior, archive.ArchiveList[j].EvaluationInfo.Behavior);
                                                if (distance < minDistance)
                                                    minDistance = distance;
                                            }
                                        }

                                        //no such thing as secong behavior + overlap right now
                                        if (UseSecondBehavior)
                                        {
                                            if (NoveltyAdditions.UseSetOverlapFeatureCollection)
                                                throw new NotImplementedException("no such thing as secong behavior + set overlap collection right now");

                                            // find the nearest neighbor in the archive
                                            float minSecondDistance = float.MaxValue;
                                            for (int j = 0; j < archiveCount; j++)
                                            {
                                                //it's possible that when we're reading the archive, some other thread is adding the object 
                                                //to fix this, we simply don't bother checking against in progress adds
                                                if (archive.ArchiveList[j] != null)
                                                {
                                                    float distance = calcMinDistance(population[i].EvaluationInfo.SecondBehavior, archive.ArchiveList[j].EvaluationInfo.SecondBehavior);
                                                    if (distance < minSecondDistance)
                                                        minSecondDistance = distance;
                                                }
                                            }

                                            //add the two for the sum of the min distances > threshold check
                                            minDistance += minSecondDistance;
                                        }

                                        bool overThreshold = NoveltyAdditions.UseSetOverlapFeatureCollection ?
                                            //minDistance represents the minimum percentage overlap among the archive -- therefore, 1- minDistance represents the largest different, if it's 
                                            //over X percent (the addThreshold), then we add th eobject
                                            minDistance > NoveltyAdditions.SetOverlapFeatureThreshold
                                            //otherwise, it's just normal feature collection, if you're over the threshold you're in!
                                            : minDistance > addThreshold;

                                        // if the nearest neighbor is further away than the threshold, add this guy
                                        if (overThreshold && addToArchive)
                                        {
                                            sem.WaitOne();
                                            // if the count in the archive is still the same, we're safe to add to it
                                            if (archiveCount == archive.ArchiveList.Count)
                                            {
                                                archive.addGenome(population[i]);
                                                addSuccessful = true;
                                            }
                                            sem.Release();
                                        }
                                        else
                                            addSuccessful = true; // we can exit loop as we didn't need to add this individual
                                    }
                                }
                            });
                    }
                    else // we're adding anyone and everyone, no marine gets left behind
                    {
                        for (int i = 0; i < population.Count; i++)
                        {
                            archive.addGenome(population[i]);
                        }
                    }
                }
            }
        }

        //Math.pow SUCKS for integers, we try to fix that here
        static double CustomPow(double num, int exp)
        {
            double result = 1.0;
            while (exp > 0)
            {
                if (exp % 2 == 1)
                    result *= num;
                exp >>= 1;
                num *= num;
            }

            return result;
        }
        #region [Heat Map Abandoned Code]

        //how far are we from the heat map?
        public static float calcHeatMapDistanceFraction(float[] behavior, float[] heatmap)
        {
            if (behavior == null)
                return 0.0f;

            float dist = 0;
            for (int b = 0; b < behavior.Length; b++)
                dist += Math.Abs(behavior[b] - heatmap[b]);

            return dist / heatmap.Length;
        }
        #endregion

        public static float calcSetDifferences(bool[] aBinary, bool[] bBinary)
        {
            int ab = 0;
            int anotb = 0;
            int bnota = 0;

            //don't bother, they're not being compared -- distance is wonky anyways
            if (aBinary == null || bBinary == null)
                return 0.0f;

            for (var i = 0; i < aBinary.Length; i++)
            {
                //xor our sets together
                bool a = aBinary[i];
                bool b = bBinary[i];
                bool nota = !a;
                bool notb = !b;

                //XOR = (A OR B) AND (NOT (A AND B))
                //if we xor together a,b -- this equal the amount of non-overlap the sets have
                if ((a || b) && (!(a && b)))
                    ab++;

                if ((a || notb) && (!(a && notb)))
                    anotb++;


                if ((nota || b) && (!(nota && b)))
                    bnota++;
            }

            float minXOR = (float)Math.Min(ab, Math.Min(anotb, bnota));

            //minXOR equals the minimum differences among the sets == % overlap of the sets 
            return (minXOR / aBinary.Length);

        }

        // returns the index of the individual in the candidate list that is maximally distant from the archive members
        public static int findFurthestGenome(NoveltyArchive<NeatGenome> archive, List<NeatGenome> candidateList, ParallelOptions po)
        {
            int index = 0;
            float bestDistance = 0;
            Object lockObj = new Object();
            Parallel.For(0, candidateList.Count, po, i =>
            //for (int i = 0; i < candidateList.Count; i++)
            {
                // find the distance to the nearest archive member for this candidate
                float minDistance = float.MaxValue;
                foreach (NeatGenome genome in archive.ArchiveList)
                {
                    float distance = calcMinDistance(genome.EvaluationInfo.Behavior, candidateList[i].EvaluationInfo.Behavior);
                    if (distance < minDistance)
                        minDistance = distance;
                }

                // if the distance is greater than the previous best, it's the new best
                if (minDistance > bestDistance)
                {
                    lock (lockObj)
                    {
                        // check again in case it was updated recently
                        if (minDistance > bestDistance)
                        {
                            bestDistance = minDistance;
                            index = i;
                        }
                    }
                }
            });

            return index;
        }

        public static float findDistanceToNearest(NeatGenome genome, List<NeatGenome> genomeList, ParallelOptions po)
        {
            float minDistance = float.MaxValue;
            Object lockObj = new Object();
            Parallel.For(0, genomeList.Count, po, i =>
            //foreach (NeatGenome g in genomeList)
            {
                if (genomeList[i] != genome)
                {
                    float distance = calcMinDistance(genome.EvaluationInfo.Behavior, genomeList[i].EvaluationInfo.Behavior);
                    if (distance < minDistance)
                    lock(lockObj)
                    {
                        if(distance < minDistance)
                            minDistance = distance;
                    }
                        
                }
            });

            return minDistance;
        }

        public static bool isOverThreshold(List<NeatGenome> genomes, NeatGenome candidate, float threshold, ParallelOptions po)
        {
            bool over = true;

            Parallel.For(0, genomes.Count, po, i =>
            //foreach (NeatGenome genome in genomes)
            {
                if (over)
                {
                    if (NoveltyCalc<NeatGenome>.calcMinDistance(genomes[i].EvaluationInfo.Behavior, candidate.EvaluationInfo.Behavior) < threshold)
                    {
                        over = false;
                    }
                }
            });

            return over;
        }

        // calculates the minimum distance between two behavior vectors - considers both the distance between A and B and also A and inverse(B)
        public static float calcMinDistance(float[] aBehavior, float[] bBehavior)
        {
            float distance = 0;
            float minusDistance = 0;
            int power = 2;

            if (NoveltyArchive<TGenome>.GENERATE_RANDOM_CPPNS)
            {
                Random r = new Random();
                return (float)r.NextDouble();
            }

            if (aBehavior == null && bBehavior == null)
            {
                //both objects are null -- there distance is equal -- they're empty behaviors
                return 0;
            }
            else if (aBehavior == null)
            {
                //take bBehavior as the distance
                float bSum = 0, bSumInv = 0;

                for (var i = 0; i < bBehavior.Length; i++)
                {
                    bSum += (float)CustomPow(bBehavior[i], power);
                    bSumInv += (float)CustomPow(1 - bBehavior[i], power);
                }
                return Math.Min(bSum, bSumInv);
            }
            else if (bBehavior == null)
            {
                //take aBehavior as the distance -- since the other object is an empty behavior
                float aSum = 0, aSumInv = 0;
                for (var i = 0; i < aBehavior.Length; i++)
                {
                    aSum += (float)CustomPow(aBehavior[i], power);
                    aSumInv += (float)CustomPow(1 - aBehavior[i], power);
                }
                return Math.Min(aSum, aSumInv);
            }

            for (int i = 0; i < aBehavior.Length; i++)
            {
                float a = aBehavior[i];
                float b = bBehavior[i];

                float delta = a - b;
                float minus = a - (1 - b);

                distance += (float)CustomPow(delta, power);
                minusDistance += (float)CustomPow(minus, power);
            }

            

            return Math.Min(distance, minusDistance);
        }

        // DEPRECATED FOR NORMAL USE
        public static float calcDistance(float[] aBehavior, float[] bBehavior)
        {
            float distance = 0;

            if (aBehavior == null && bBehavior == null)
            {
                //both objects are null -- there distance is equal -- they're empty behaviors
                return 0;
            }
            else if (aBehavior == null)
            {
                //take bBehavior as the distance
                float bSum = 0;
                for (var i = 0; i < bBehavior.Length; i++)
                    bSum += bBehavior[i] * bBehavior[i];
                return bSum;
            }
            else if (bBehavior == null)
            {
                //take aBehavior as the distance -- since the other object is an empty behavior
                float aSum = 0;
                for (var i = 0; i < aBehavior.Length; i++)
                    aSum += aBehavior[i] * aBehavior[i];
                return aSum;
            }
            //otherwise, both behaviors exist -- we're ready to sum the difference
            for (int i = 0; i < aBehavior.Length; i++)
            {
                float delta = aBehavior[i] - bBehavior[i];
                distance += delta * delta;
            }

            return distance;
        }

        public bool behaviorsOverlap(float[] aBehavior, float[] bBehavior)
        {
            return calcMinDistance(aBehavior, bBehavior) < addThreshold;
        }

        public double calculateImageWiseNovelty(float[] behavior)
        {
            double novelty = 0;

            if (behavior != null)
            {
                for (int i = 0; i < behavior.Length; i++)
                {
                    List<float> nearest = new List<float>();
                    //List<float> fullList = new List<float>();

                    // find the nearest members of the archive for this particular image
                    for (int j = 0; j < imageOutputs[i].Count; j++)
                    {                        
                        float distance = Math.Abs(imageOutputs[i][j] - behavior[i]);

                        //fullList.Add(distance);
                            
                        if (nearest.Count < NoveltyAdditions.IMAGEWISEK)
                        {
                            nearest.Add(distance);
                            if (nearest.Count == NoveltyAdditions.IMAGEWISEK)
                                nearest.Sort();
                        }
                        else
                        {
                            if (distance < nearest[NoveltyAdditions.IMAGEWISEK - 1])
                            {                                    
                                nearest.RemoveAt(NoveltyAdditions.IMAGEWISEK - 1);
                                for (int k = 0; k < NoveltyAdditions.IMAGEWISEK - 1; k++)
                                {
                                    if (nearest[k] > distance)
                                    {
                                        nearest.Insert(k, distance);
                                        break;
                                    }
                                }

                                if (nearest.Count < NoveltyAdditions.IMAGEWISEK) // we didn't add the individual yet, they should be placed on the end
                                    nearest.Add(distance);

                                /*
                                nearest.Add(distance);
                                nearest.Sort();
                                nearest.RemoveAt(NoveltyAdditions.IMAGEWISEK);
                                */
                            }
                        }                            
                    }

                    /*
                    fullList.Sort();
                    for (int j = 0; j < Math.Min(NoveltyAdditions.IMAGEWISEK, fullList.Count); j++)
                        nearest.Add(fullList[j]);
                    */
                    novelty += nearest.Sum() / nearest.Count;
                }
            }

            return novelty;
        }       
    }
}
