using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Core;

namespace SharpNeat.Novelty
{
    public class NoveltyEvaluationInfo<TGenome> where TGenome : class, IGenome<TGenome>
    {
        public NoveltyEvaluationInfo(double novelScore, List<TGenome> nearestNeighborGenomes)
        {
            NoveltyScore = novelScore;
            NearestNeighbors = nearestNeighborGenomes;
        }
       

        #region Public Props
        public List<TGenome> NearestNeighbors
        {
            get;
            set;
        }
        public double NoveltyScore
        {
            get;
            set;
        }
        public double AlternateNovelty
        {
            get;
            set;
        }
        public List<TGenome> AlternateNeighbors
        {
            get;
            set;
        }
        #endregion

    }
}
