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

namespace SharpNeat.Core
{
    /// <summary>
    /// Wrapper struct for fitness values.
    /// </summary>
    public struct FitnessInfo
    {
        /// <summary>
        /// Precosntructed FitnessInfo for commen case of representing zero fitness.
        /// </summary>
        public static FitnessInfo Zero = new FitnessInfo(0.0, null, 0.0, null);

        /// <summary>
        /// Fitness score.
        /// </summary>
        public double _fitness;

        //DFN: Needed to add novelty to the fitness info for doing blackbox evaluation on deep feature neat
        /// <summary>
        /// Novelty behavior storage
        /// </summary>
        public float[] _novelty;

        public double[] _noveltyD;

        public bool[] _binaryActivation;
        public float[] _secondBehavior;

        /// <summary>
        /// Auxiliary fitness info, i.e. for evaluation metrics other than the
        /// primary fitness metric but that nonetheless we are interested in observing.
        /// </summary>
        public AuxFitnessInfo[] _auxFitnessArr;

        /// <summary>
        /// Default constructor.
        /// </summary>
        public FitnessInfo(double fitness, float[] behavior, double alternativeFitness, float[] secondBehavior = null, bool[] binaryActivation = null, double[] noveltyD = null)
        {
            _fitness = fitness;
            _novelty = behavior;
            _auxFitnessArr = new AuxFitnessInfo[] {new AuxFitnessInfo("Alternative Fitness", alternativeFitness)};
            _secondBehavior = secondBehavior;
            _binaryActivation = binaryActivation;
            _noveltyD = noveltyD;
        }

        /// <summary>
        /// Construct with the provided fitness value and auxiliary fitness info.
        /// </summary>
        public FitnessInfo(double fitness, float[] behavior, AuxFitnessInfo[] auxFitnessArr, float[] secondBehavior = null, bool[] binaryActivation = null, double[] noveltyD = null)
        {
            _fitness = fitness;
            _novelty = behavior;
            _auxFitnessArr = auxFitnessArr;
            _secondBehavior = secondBehavior;
            _binaryActivation = binaryActivation;
            _noveltyD = noveltyD;
        }

    }
}