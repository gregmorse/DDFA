using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Phenomes.NeuralNets;
using SharpNeat.Network;
using System.Threading.Tasks;

namespace SharpNeat.Phenomes.NeuralNets
{
    //multiple output perceptron 

    /// <summary>
    /// WARNING: MultiPerceptron assumes that activation is a SteependSigmoid. Actually, perceptrons are Linear output activation by default. 
    /// </summary>
    public class MultiPerceptron : OpenAcyclicNetwork
    {
        public ParallelOptions po
        {
            get;
            set;
        }
        int[][] outputsToConnections;
        public MultiPerceptron(IActivationFunction[] nodeActivationFnArr,
                                  double[][] nodeAuxArgsArr,
                                  FastConnection[] connectionArr,
                                  LayerInfo[] layerInfoArr,
                                  int[] outputNodeIdxArr,
                                  int nodeCount,
                                  int inputNodeCount,
                                  int outputNodeCount)
            : base(nodeActivationFnArr, nodeAuxArgsArr, connectionArr, layerInfoArr, outputNodeIdxArr, nodeCount, inputNodeCount, outputNodeCount)
        {
            //here we actually track all connections going into the outputs
            //we have as many connections as there are outputs
            outputsToConnections = new int[outputNodeCount][];

            List<int> outputNodeIds = new List<int>();
            for (var i = 0; i < outputNodeIdxArr.Length; i++)
                outputNodeIds.Add(outputNodeIdxArr[i]);

            SortedList<int, List<int>> mappedConnections = new SortedList<int, List<int>>();
            for (var i = 0; i < _connectionArr.Length; i++)
            {
                var c = _connectionArr[i];

                var ixOf = outputNodeIds.IndexOf(c._tgtNeuronIdx);

                if (ixOf == -1)
                    throw new Exception("Perceptron network has a connection going to something other than an output node.");


                List<int> incoming;
                if (!mappedConnections.TryGetValue(ixOf, out incoming))
                {
                    incoming = new List<int>();
                    mappedConnections.Add(ixOf, incoming);
                }

                //now we have incoming array, add this connection
                incoming.Add(i);
            }

            //mapped connectiosn has a list of output nodes and their connections
            for (int i = 0; i < outputNodeCount; i++)
            {
                if (!mappedConnections.ContainsKey(i))
                    throw new Exception(string.Format("Perceptron output {0} has 0 connections (!)", i));

                //otherwise, we have incoming connections, so we set it inside our 2D array
                outputsToConnections[i] = mappedConnections[i].ToArray();
            }

            //now we have all connections for easy updating when shit goes down
        }

        public void MakeAllWeightCorrections(double[] allAtOnce)
        { 
            for(int i = 0; i < allAtOnce.Length; i++)
            {
                this.ConnectionList[i]._weight += allAtOnce[i];
            }                        
        }

        //Just send in the correct outputs, we'll ready the actual outputs, and punish poor behavior
        /// <summary>
        /// What was the correct maximum for our outputs? If it wasn't that, correct the behavior!
        /// </summary>
        /// <param name="properOutputs"></param>
        public double[] LearnFromCorrect(bool[] properOutputs, float _learningRate, bool adjustAllAtOnce)
        {
            double[] allAtOnce = (adjustAllAtOnce ? new double[ConnectionList.Length] : null);
            if (properOutputs.Length != _outputNodeCount)
                throw new Exception("Learning output array must be equal in size to Network output count");

            int maximumOutputIx = -1;

            if (_outputNodeCount > 1)
            {
                double max = double.MinValue;
                //check our outputs
                for (int i = 0; i < OutputCount; i++)
                {
                    double o = OutputSignalArray[i];
                    if (o > max)
                    {
                        max = o;
                        maximumOutputIx = i;
                    }
                }
            }
            else
            {
                //we just say if our only output is active or not
                if (OutputSignalArray[0] > .5)
                {
                    maximumOutputIx = 0; //we guessed it's active -- so we "select" output 0
                }
                //otherwise, we leave maximumoutput = -1, therefore 0 did not activate, so the outputguess == false
            }

            for (int i = 0; i < OutputCount; i++)
            {
                //if we're the maximum output node -- our guess == true
                bool outputGuess = (i == maximumOutputIx);
 
                if (properOutputs[i] == outputGuess)
                {
                    //correct! do nothing
                }
                else
                {
                    //wrong! You should be 1 or -1
                    double proper = properOutputs[i] ? 1.0 : -1.0;
                    double oOutput = this.OutputSignalArray[i];
                    
                    //grab the incoming connections for this output
                    int[] wrongIncoming = outputsToConnections[i];

                    //we need to adjust all the wrong incoming objects
                    Parallel.For(0, wrongIncoming.Length, po, c =>
                    //for(var c=0; c < wrongIncoming.Length; c++)
                    {
                        //connection
                        var connIx = wrongIncoming[c];
                        var conn = _connectionArr[connIx];

                        //grab the source directly (inputs wired to outputs in perceptron so source is input neuron
                        //according to: http://en.wikipedia.org/wiki/Perceptron under Steps
                        double weightAdjust = _learningRate * (proper - oOutput) * _activationArr[conn._srcNeuronIdx];

                        //now we can adjust the weight directly in our array!
                        if (adjustAllAtOnce)
                        {
                            allAtOnce[connIx] += weightAdjust;
                        }
                        else //just do it now
                            _connectionArr[connIx]._weight += weightAdjust;

                        //Console.WriteLine("Ix: {4}, Input: {6}, Actual {0}, desired {1}, current: {2}, adjustment: {3}, final: {5}", oOutput, proper, conn._weight, weightAdjust, connIx, _connectionArr[connIx]._weight, _activationArr[conn._srcNeuronIdx]);

                    });
                }
            }

            //most of the time null, unless specified
            return allAtOnce;
        }

    }

    //This is a network that has connections accessible -- and generally all the internals of the acyclic network are available
    public class OpenAcyclicNetwork : FastAcyclicNetwork
    {
        /// <summary>
        /// Inherits the normal fast acyclic network, except adds some members for messing with weights during the lifetime!
        /// </summary>
        /// <param name="nodeActivationFnArr"></param>
        /// <param name="nodeAuxArgsArr"></param>
        /// <param name="connectionArr"></param>
        /// <param name="layerInfoArr"></param>
        /// <param name="outputNodeIdxArr"></param>
        /// <param name="nodeCount"></param>
        /// <param name="inputNodeCount"></param>
        /// <param name="outputNodeCount"></param>
        public OpenAcyclicNetwork(IActivationFunction[] nodeActivationFnArr,
                                  double[][] nodeAuxArgsArr,
                                  FastConnection[] connectionArr,
                                  LayerInfo[] layerInfoArr,
                                  int[] outputNodeIdxArr,
                                  int nodeCount,
                                  int inputNodeCount,
                                  int outputNodeCount)
            : base(nodeActivationFnArr, nodeAuxArgsArr, connectionArr, layerInfoArr, outputNodeIdxArr, nodeCount, inputNodeCount, outputNodeCount)
        {
            _outputMapping = outputNodeIdxArr;
        }

        int[] _outputMapping;
        public int[] OutputMapping
        {
            get { return _outputMapping; }
        }

        public LayerInfo[] LayerInfo
        {
            get { return _layerInfoArr; }
        }
        public double[] ActivationArray
        {
            get { return _activationArr; }
        }

        //should be enough to mess with connections, mwahahaha
        public FastConnection[] ConnectionList
        {
            get
            {
                return _connectionArr;
            }
        }
    }
}
