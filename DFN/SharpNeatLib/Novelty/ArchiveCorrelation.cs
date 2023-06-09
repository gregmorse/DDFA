using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Genomes.Neat;
using System.Threading.Tasks;
using System.Threading;
using SharpNeat.Core;

namespace SharpNeat.Novelty
{
    public class ArchiveCorrelation
    {
        private ParallelOptions po;
        private List<ImageCorrelation> corrList;
        private int maxMembers;
        private static bool useSimilarity;

        private float[][] behaviorCache; // cached behavior values for calculateCorrelations()

        public ArchiveCorrelation(int maxMembers, ParallelOptions po, bool useSimilarity)
        {           
            this.maxMembers = maxMembers;
            this.po = po;
            ArchiveCorrelation.useSimilarity = useSimilarity;
        }        

        public void calculateCorrelations(List<NeatGenome> archive, double[][] images)
        {
            int threads = po.MaxDegreeOfParallelism;
            behaviorCache = new float[archive.Count][];
            for (int i = 0; i < archive.Count; i++)
            {
                if (archive[i].EvaluationInfo.Behavior != null)
                {
                    behaviorCache[i] = archive[i].EvaluationInfo.Behavior;                    
                }
            }

            ArchiveCorrelationWorker[] workers = new ArchiveCorrelationWorker[threads - 1];
            Thread[] workerThreads = new Thread[threads - 1];

            // we only evaluate when i < j, which means we're filling up one "corner" (i.e. triangle) of the grid
            // we want each thread to have roughly the same amount of work, so we can't just use (i / threads) as a slice size, the slice size much change depending on where in the triangle we are
            int start = 0, end = 0;
            int imageCount = archive[0].EvaluationInfo.Behavior.Length;
            int totalEvals = (imageCount + 1) * imageCount / 2;
            int idealThreadEvals = totalEvals / threads;
            
            for (int i = 0; i < threads - 1; i++)
            {
                int sumEvals = 0;
                while (sumEvals < idealThreadEvals)
                {
                    sumEvals += imageCount - end++;                    
                }
                workers[i] = new ArchiveCorrelationWorker(archive, null, start, end, maxMembers, null, behaviorCache, images);
                workerThreads[i] = new Thread(new ThreadStart(workers[i].calculateCorrelationsJob));
                workerThreads[i].Start();
                start = end;
            }
            
            // now do part of the images with the main thread
            end = imageCount;
            ArchiveCorrelationWorker acw = new ArchiveCorrelationWorker(archive, null, start, end, maxMembers, null, behaviorCache, images);
            acw.calculateCorrelationsJob();
            
            // now collect all the results from the worker threads 
            for (int i = 0; i < threads - 1; i++)            
                workerThreads[i].Join();
            
            // build our master list from the best of each thread
            corrList = acw.getCorrelationList();
            for (int i = 0; i < threads - 1; i++)
                corrList = corrList.Concat(workers[i].getCorrelationList()).ToList();
            
            // trim back our list to the proper size
            corrList = corrList.OrderByDescending(si => si.CorrCoef).ToList();
            corrList.RemoveRange(maxMembers, corrList.Count - maxMembers);   
         
            // print off the highest 10 correlation average, and the highest 1 image pair
            float corrCoef = 0f;
            for (int i = 0; i < 10; i++)
            {
                corrCoef += corrList[i].CorrCoef;
            }
            corrCoef /= 10;
            System.IO.File.AppendAllText("correlationData.txt", System.DateTime.Now.ToString() + " : " + corrCoef + " : " + corrList[0].Image1 + " : " + corrList[0].Image2 + Environment.NewLine);
        }

        public void testCorrelationsList(List<NeatGenome> archive, double[][] images)
        {
            float corrCoefSum = 0f;
            for (int i = 0; i < 10; i++)
            {
                float[] set1 = new float[archive.Count];
                float[] set2 = new float[archive.Count];

                int setCounter = 0;
                for (int j = 0; j < archive.Count; j++)
                {
                    if (archive[j].EvaluationInfo.Behavior == null)
                    {
                        set1[setCounter] = 0;
                        set2[setCounter++] = 0;
                    }
                    else
                    {
                        set1[setCounter] = archive[j].EvaluationInfo.Behavior[corrList[i].Image1];
                        set2[setCounter++] = archive[j].EvaluationInfo.Behavior[corrList[i].Image2];
                    }
                }

                corrCoefSum += calculateCoefficient(set1, set2, images[corrList[i].Image1], images[corrList[i].Image2]);
            }
            corrCoefSum /= 10;
            System.IO.File.AppendAllText("correlationData.txt", System.DateTime.Now.ToString() + " -- End of phase " + corrCoefSum + Environment.NewLine);
        }

        public int findBestBreaker(List<NeatGenome> archive, FitnessInfo[] fitInfo, double[][] images)
        {
            int threads = po.MaxDegreeOfParallelism;
            int bestIndex = 0;
            float largestDifference = 0f;

            ArchiveCorrelationWorker[] workers = new ArchiveCorrelationWorker[threads - 1];
            Thread[] workerThreads = new Thread[threads - 1];
            
            int start = 0, end = 0;
            int idealThreadEvals = fitInfo.Length / threads;

            for (int i = 0; i < threads - 1; i++)
            {
                end = start + idealThreadEvals;
                workers[i] = new ArchiveCorrelationWorker(archive, fitInfo, start, end, maxMembers, corrList, null, images);
                workerThreads[i] = new Thread(new ThreadStart(workers[i].findBestBreakerJob));
                workerThreads[i].Start();
                start = end;
            }

            // now do part of the images with the main thread
            end = fitInfo.Length;
            ArchiveCorrelationWorker acw = new ArchiveCorrelationWorker(archive, fitInfo, start, end, maxMembers, corrList, null, images);
            acw.findBestBreakerJob();

            // now collect all the results from the worker threads 
            for (int i = 0; i < threads - 1; i++)
                workerThreads[i].Join();

            largestDifference = acw.LargestDifference;
            bestIndex = acw.BestIndex;
            for (int i = 0; i < threads - 1; i++)
                if (workers[i].LargestDifference > largestDifference)
                {
                    largestDifference = workers[i].LargestDifference;
                    bestIndex = workers[i].BestIndex;
                }

            return bestIndex;
        }

        public double calculateBreakerScore(List<NeatGenome> archive, float[] behavior, double[][] images)
        {
            return Math.Pow(ArchiveCorrelation.calculateCoefficientDifference(corrList, archive, behavior, images), 4);
        }

        public static float calculateCoefficient(float[] set1, float[] set2, double[] image1, double[] image2)
        {
            float sum = 0;
            // calculate average for each set
            for (int i = 0; i < set1.Length; i++)
                sum += set1[i];
            float avg1 = sum / set1.Length;

            sum = 0;
            for (int i = 0; i < set2.Length; i++)
                sum += set2[i];
            float avg2 = sum / set2.Length;

            float termA = 0;
            float termB = 0;
            float termC = 0;

            for (int i = 0; i < set1.Length; i++)
            {
                termA += (set1[i] - avg1) * (set2[i] - avg2);
                termB += (set1[i] - avg1) * (set1[i] - avg1);
                termC += (set2[i] - avg2) * (set2[i] - avg2);
            }

            float coeff = Math.Abs(termA / (float)Math.Sqrt(termB * termC));

            // scale coefficient by image similarity            

            if (useSimilarity)
            {
                coeff *= calculateImageDifference(image1, image2);
            }

            return coeff;
        }

        public static float calculateImageDifference(double[] image1, double[] image2)
        {
            double diff = 0f;

            for (int i = 0; i < image1.Length; i++)
                diff += (image1[i] - image2[i]) * (image1[i] - image2[i]);

            return (float)diff;
        }

        public static float calculateCoefficientDifference(List<ImageCorrelation> corrList, List<NeatGenome> archive, float[] behavior, double[][] images)
        {
            float sumDifference = 0f;
            foreach (ImageCorrelation corr in corrList)
            {
                int counter1 = 0, counter2 = 0;
                float[] signature1 = new float[archive.Count + 1];
                float[] signature2 = new float[archive.Count + 1];
                for (int k = 0; k < archive.Count; k++)
                {
                    if (archive[k].EvaluationInfo.Behavior != null)
                    {
                        signature1[counter1++] = archive[k].EvaluationInfo.Behavior[corr.Image1];
                        signature2[counter2++] = archive[k].EvaluationInfo.Behavior[corr.Image2];
                    }
                    else
                    {
                        signature1[counter1++] = 0;
                        signature2[counter2++] = 0;
                    }
                }
                if (behavior != null)
                {
                    signature1[counter1++] = behavior[corr.Image1];
                    signature2[counter2++] = behavior[corr.Image2];
                }
                else
                {
                    signature1[counter1++] = 0;
                    signature2[counter2++] = 0;
                }

                float newCorr = ArchiveCorrelation.calculateCoefficient(signature1, signature2, images[corr.Image1], images[corr.Image2]);

                //sumDifference += Math.Max(corr.CorrCoef - newCorr, 0); // ignore any that strengthen correlations
                sumDifference += corr.CorrCoef - newCorr; // consider both strengthening and weakening of correlations
            }

            return sumDifference;
        }
    }

    class ArchiveCorrelationWorker
    {
        private List<NeatGenome> archive;
        private FitnessInfo[] fitInfo;
        private int start, end, maxMembers;

        // calculateCorrelations return value
        private List<ImageCorrelation> correlationList;
        public List<ImageCorrelation> getCorrelationList() { return correlationList; }
        private int lowestIndex;
        private float lowestCorr;

        // findBestBreaker return values
        public int BestIndex { get; set; }
        public float LargestDifference { get; set; }

        private float[][] behaviorCache;
        private double[][] images;

        List<ImageCorrelation> corrList; // passed in for findBestBreaker

        public ArchiveCorrelationWorker(List<NeatGenome> archive, FitnessInfo[] fitInfo, int start, int end, int maxMembers, List<ImageCorrelation> corrList, float[][] behaviorCache, double[][] images)
        {
            this.archive = archive;
            this.fitInfo = fitInfo;
            this.start = start;
            this.end = end;
            this.maxMembers = maxMembers;
            this.corrList = corrList;
            this.behaviorCache = behaviorCache;
            this.images = images;
            correlationList = new List<ImageCorrelation>(maxMembers);            
        }

        public void calculateCorrelationsJob()
        {
            for (int i = start; i < end; i++)
            {
                int counter1 = 0;
                float[] signature1 = new float[archive.Count];
                for (int k = 0; k < archive.Count; k++)
                {
                    if (archive[k].EvaluationInfo != null && archive[k].EvaluationInfo.Behavior != null)
                        signature1[counter1++] = behaviorCache[k][i];
                    else
                        signature1[counter1++] = 0;
                }

                for (int j = 0; j < archive[0].EvaluationInfo.Behavior.Length; j++)
                {
                    if (i < j) // only need to compare image pairs once
                    {
                        int counter2 = 0;
                        float[] signature2 = new float[archive.Count];
                        for (int k = 0; k < archive.Count; k++)
                        {
                            if (archive[k].EvaluationInfo != null && archive[k].EvaluationInfo.Behavior != null)
                                signature2[counter2++] = behaviorCache[k][j];
                            else
                                signature2[counter2++] = 0;
                        }

                        ImageCorrelation corr = new ImageCorrelation(i, j, ArchiveCorrelation.calculateCoefficient(signature1, signature2, images[i], images[j]), ArchiveCorrelation.calculateImageDifference(images[i], images[j]));                        
                        addImageCorrelation(corr);                        
                    }
                }
            }
        }

        public void findBestBreakerJob()
        {
            float largestDifference = 0f;
            int bestIndex = 0;

            for (int i = start; i < end; i++)
            {
                if (fitInfo[i]._novelty != null)
                {
                    float sumDifference = ArchiveCorrelation.calculateCoefficientDifference(corrList, archive, fitInfo[i]._novelty, images);

                    if (sumDifference > largestDifference)
                    {
                        // check again before setting
                        if (sumDifference > largestDifference)
                        {
                            largestDifference = sumDifference;
                            bestIndex = i;
                        }                        
                    }
                }
            }

            this.BestIndex = bestIndex;
            this.LargestDifference = largestDifference;
        }

        private void addImageCorrelation(ImageCorrelation corr)
        {
            if (correlationList.Count < maxMembers)
            {
                correlationList.Add(corr);
                // sort if we reached max                
                if (correlationList.Count == maxMembers)
                    correlationList = correlationList.OrderByDescending(si => si.CorrCoef).ToList();
            }
            else
            {                
                
                // boot the lowest member if this guy is better
                if (correlationList[maxMembers - 1].CorrCoef < corr.CorrCoef)
                {                                        
                    correlationList.RemoveAt(maxMembers - 1);
                    int index = correlationList.BinarySearch(corr, new ImageCorrelationComparer());
                    if (index < 0)
                        index = ~index;
                    correlationList.Insert(index, corr);                                       
                }
                
            }
        }

        private void findLowestCorr()
        {
            lowestCorr = float.MaxValue;
            for (int i = 0; i < correlationList.Count; i++)
            {
                if (correlationList[i].CorrCoef < lowestCorr)
                {
                    lowestCorr = correlationList[i].CorrCoef;
                    lowestIndex = i;
                }
            }
        }
    }

    public class ImageCorrelationComparer : IComparer<ImageCorrelation>
    {
        public int Compare(ImageCorrelation a, ImageCorrelation b)
        {
            if (a.CorrCoef > b.CorrCoef)
                return -1;
            if (a.CorrCoef < b.CorrCoef)
                return 1;
            return 0;
        }
    }
}
