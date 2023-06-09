/* ***************************************************************************
 * This file is part of SharpNEAT - Evolution of Neural Networks.
 * 
 * Copyright 2004-2006, 2009-2012 Colin Green (sharpneat@gmail.com)
 *
 * SharpNEAT is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SharpNEAT is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SharpNEAT.  If not, see <http://www.gnu.org/licenses/>.
 */
using System;
using SharpNeat.Utility;
using System.Collections.Generic;
using SharpNeat.Novelty;

namespace SharpNeat.Core
{
    /// <summary>
    /// Encapsulates genome evaluation data; specifically whether the genome has been evaluated, the genome's fitness,
    /// how many times the genome has been evaluated and if more than once then the mean fitness.
    /// 
    /// Mean fitness is sometimes used when evaluations are non-determininstic (where each evaluation with the 
    /// same parameters results in a different fitness, usually following some distribution curve) or else 
    /// successive evaluations use different evaluation parameters and therefore result in different fitnesses. 
    /// This approach is sometimes used in order to evaluate against a very computationally expensive fitness
    /// scheme, or indeed a computationally intractable scheme - i.e. we only execute against some subset of the
    /// problem space for each genome evaluation, and thus we obtain a more representative value for a genome's
    /// fitness by averaging over successive evaluations.
    /// </summary>
    public class EvaluationInfo<TGenome> where TGenome : class, IGenome<TGenome>
    {
        readonly DoubleCircularBufferWithStats _fitnessHistory;
        readonly int _fitnessHistoryLength;


        public static string FitnessObjective = "fitness";
        public static string NoveltyObjective = "novelty";
        public static string AlternateObjective = "secondNovelty";

        float[] _behavior;
        bool[] _binaryActivation;
        float[] _secondaryBehavior;
        double _actualFitnessObjective;

        NoveltyEvaluationInfo<TGenome> noveltyInfo;


        double _fitness;
        bool _isEvaluated;
        uint _evaluationCount;
        uint _evaluationPassCount;
        AuxFitnessInfo[] _auxFitnessArr;
        SortedDictionary<string, double> objectives = new SortedDictionary<string, double>();

        SortedDictionary<string, double> rawLocalObjectives = new SortedDictionary<string, double>();
        Dictionary<uint, double> geneticDistance = new Dictionary<uint, double>();

        #region Constructor

        /// <summary>
        /// Construct and allocate the specified amount of memory for fitness history. Use
        /// zero if you don't require fitness history - but note that no arithmetic mean will be available.
        /// </summary>
        public EvaluationInfo(int fitnessHistoryLength)
        {
            _fitnessHistoryLength = fitnessHistoryLength;
            if(0 != fitnessHistoryLength) {
                _fitnessHistory = new DoubleCircularBufferWithStats(fitnessHistoryLength);
            }
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the fitness value to be used by the evolution algorithm for selection of genomes
        /// (for reproduction) and species fitness sharing. If a fitness buffer is in use this will be the 
        /// average fitness for the last N fitness evaluations as held by the fitness history buffer,
        /// otherwise if no buffer is in use it is simply the fitness from the most recent evaluation.
        /// </summary>
        public double Fitness
        {
            get { return (null != _fitnessHistory) ? _fitnessHistory.Mean : _fitness; }
        }

        /// <summary>
        /// Returns the novelty information stores from a novelty evaluation!
        /// </summary>
        public NoveltyEvaluationInfo<TGenome> NoveltyInformation
        {
            get { return noveltyInfo; }
        }

        /// <summary>
        /// What is the thing we actually desire
        /// </summary>
        public double GlobalFitness
        {
            get { return _actualFitnessObjective; }
        }

        //DFN: Need to be able to access behavior for agiven genome evaluation
        public float[] Behavior
        {
            get { return _behavior; }
        }

        //DFN-Paul: we have a second behavior we keep track of as well -- for double objective novelty style -- yo
        public float[] SecondBehavior
        {
            get { return _secondaryBehavior; }
        }

        //DFN-Paul: we have a second behavior we keep track of as well -- for double objective novelty style -- yo
        public bool[] BinaryActivation
        {
            get { return _binaryActivation; }
        }
        /// <summary>
        /// Gets the fitness from the most recent evaluation. This may be different to the Fitness property if a fitness
        /// history buffer is in use that averages out the reported fitness over a number of evaluations.
        /// </summary>
        public double MostRecentFitness
        {
            get { return _fitness; }
        }

        /// <summary>
        /// Gets the arithmetic mean of the most recent fitnesses as stored in the fitness history buffer.
        /// Note that an exception will be thrown if the history buffer is not being used.
        /// </summary>
        public double MeanFitness
        {
            get 
            {   // Note. throws an exception if there is no fitness history. If you want a mean you must
                // store fitness history.
                if (_fitnessHistory == null)
                    return -1;
                return _fitnessHistory.Mean; 
            }
        }

        /// <summary>
        /// Gets or sets auxiliary fitness info, i.e. for evaluation metrics other than the
        /// primary fitness metric but that nonetheless we are interested in observing.
        /// </summary>
        public AuxFitnessInfo[] AuxFitnessArr
        {
            get { return _auxFitnessArr; }
            set { _auxFitnessArr = value; }
        }

        /// <summary>
        /// Gets a value indicating whether a genome has been evaluated at least once.
        /// </summary>
        public bool IsEvaluated
        {
            get { return _isEvaluated; }
            set { _isEvaluated = value; }
        }

        /// <summary>
        /// Gets the total number of times the genome has been evaluated. 
        /// </summary>
        public uint EvaluationCount
        {
            get { return _evaluationCount;  }
        }

        /// <summary>
        /// Gets or sets the total number of times the genome has skipped evaluation.
        /// Some evaluation schemes re-evaluate genomes that persist between generations (e.g. elite genomes)
        /// at each generation, whereas other schemes may chose to not re-evaulate or only re-evaluate every Nth
        /// generation/attempt. This counter tracks how many times the genome has been skipped to support such schemes.
        /// </summary>
        public uint EvaluationPassCount
        {
            get { return _evaluationPassCount;  }
            set { _evaluationPassCount = value;  }
        }

        /// <summary>
        /// Gets EvaluationCount + EvaluationPassCount.
        /// </summary>
        public uint TotalEvaluationCount
        {
            get { return _evaluationCount + _evaluationPassCount; }
        }

        /// <summary>
        /// Gets the capacity of the fitness history buffer in use. Zero if no history buffer is being used.
        /// </summary>
        public int FitnessHistoryLength
        {
            get { return _fitnessHistoryLength; }
        }

        public SortedDictionary<string, double> Objectives
        {
            get { return objectives; }
        }

        #endregion

        #region Public Methods

        /// <summary>
        /// Assign a fitness. If a fitness history buffer was created then the value will be
        /// enqueued in the buffer.
        /// </summary>
        public void SetFitness(double fitness)
        {
            if(fitness < 0.0 || double.IsNaN(fitness) || double.IsInfinity(fitness)) {
                throw new ArgumentOutOfRangeException("Negative fitness values are not valid.");
            }

            _isEvaluated = true;
            _evaluationCount++;
            _fitness = fitness;

            if(null != _fitnessHistory) {
                _fitnessHistory.Enqueue(fitness);
            }
        }
        /// <summary>
        /// This is your objective you're actually trying to solve, whether or not you're using it directly for search. Distance traveled, speed, etc. 
        /// </summary>
        /// <param name="fit"></param>
        public void SetGlobalFitness(double fit)
        {
            _actualFitnessObjective = fit;
        }
        public void SetBehavior(float[] behavior)
        {
            _behavior = behavior;
        }

        public void SetNoveltyInformation(NoveltyEvaluationInfo<TGenome> nv)
        {
            //save this novelty info -- DOES NOT contain behavior -- just info about novelty calcuations
            noveltyInfo = nv;
        }

        public void SetSecondBehavior(float[] behavior)
        {
            _secondaryBehavior = behavior;
        }
        public void SetBinaryActivation(bool[] act)
        {
            _binaryActivation = act;
        }

        //copy the objectives dictionary when setting
        public void SetObjectives(SortedDictionary<string, double> dic)
        {
            if(dic != null)
                objectives = new SortedDictionary<string, double>(dic);
            else
                objectives = new SortedDictionary<string, double>();
        }

        public void SetObjective(string objectiveName, double val)
        {
            if (!objectives.ContainsKey(objectiveName))
                objectives.Add(objectiveName, val);
            else
                objectives[objectiveName] = val;
        }

        /// <summary>
        /// Increments EvaluationPassCount.
        /// </summary>
        public void IncrEvaluationPassCount()
        {
            _evaluationPassCount++;
        }

        #endregion
    }
}
