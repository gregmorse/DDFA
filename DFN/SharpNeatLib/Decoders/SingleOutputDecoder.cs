/* ***************************************************************************
 * This file is part of SharpNEAT - Evolution of Neural Networks.
 * 
 * Copyright 2004-2006, 2009-2010 Colin Green (sharpneat@gmail.com)
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
using System.Collections.Generic;
using System.Diagnostics;
using SharpNeat.Network;
using SharpNeat.Phenomes.NeuralNets;

namespace SharpNeat.Decoders
{
    /// <summary>
    /// Static factory for creating AcyclicNetwork(s) from INetworkDefinition(s).
    /// </summary>
    public class SingleOutputNetworkFactory
    {
        public static bool shouldLimitWeightBackprop = false;

        #region Public Methods

        /// <summary>
        /// Creates a AcyclicNetwork from an INetworkDefinition.
        /// </summary>
        public static AcyclicByteNetwork CreateFastAcyclicNetwork(INetworkDefinition networkDef)
        {
            Debug.Assert(!CyclicNetworkTest.IsNetworkCyclic(networkDef), "Attempt to decode a cyclic network into a FastAcyclicNetwork.");

            return new AcyclicByteNetwork(networkDef.InputNodeCount, networkDef.OutputNodeCount, networkDef.ConnectionList, networkDef.NodeList, networkDef.ActivationFnLibrary);
        }

        /// <summary>
        /// Creates a AcyclicNetwork from an INetworkDefinition.
        /// </summary>
        public static SingleOutputNetwork CreateSingleOutputNetwork(INetworkDefinition networkDef)
        {
            Debug.Assert(!CyclicNetworkTest.IsNetworkCyclic(networkDef), "Attempt to decode a cyclic network into a FastAcyclicNetwork.");

            int tgtNodeId = -1;

            double[] weights = new double[networkDef.InputNodeCount + 1];

            //networkDef.ActivationFnLibrary.GetFunction(
            IActivationFunction activationOut = null;

            bool[] allowBackprop = new bool[networkDef.InputNodeCount + 1]; 

            for (int i = 0; i < networkDef.ConnectionList.Count; i++)
            {
                var conn = networkDef.ConnectionList[i];

                if (tgtNodeId == -1)
                {
                    tgtNodeId = (int)conn.TargetNodeId;
                    //get the activation for the output - yeah!
                    activationOut = networkDef.ActivationFnLibrary.GetFunction(networkDef.NodeList[tgtNodeId].ActivationFnId);
                }
                if(conn.TargetNodeId != tgtNodeId)
                    throw new Exception("Single Output Network can only have one output derr, this network has more!");

                //anywhere there is an actual wait, we allow backprop
                allowBackprop[conn.SourceNodeId] = true;

                //easy configuration, use sourceId as index into weights array, set the weights!
                weights[conn.SourceNodeId] = conn.Weight;
            }


            return new SingleOutputNetwork(weights, activationOut, shouldLimitWeightBackprop ? allowBackprop : null);
        }

        /// <summary>
        /// Creates a AcyclicNetwork from an INetworkDefinition.
        /// </summary>
        public static SingleOutputNetworkWithHidden CreateSingleOutputNetworkWithHidden(INetworkDefinition networkDef)
        {
            
            Debug.Assert(!CyclicNetworkTest.IsNetworkCyclic(networkDef), "Attempt to decode a cyclic network into a FastAcyclicNetwork.");

            //input nodes!
            HashSet<int> nodeIDList = new HashSet<int>();

            //Dictionary<int, List<INetworkConnection>> incomingConnections = new Dictionary<int, List<INetworkConnection>>();
            //Dictionary<int, List<INetworkConnection>> outgoingConnections = new Dictionary<int, List<INetworkConnection>>();

            //double[] weights = new double[networkDef.InputNodeCount + 1];


            int inputAndBiasNodeCount = networkDef.InputNodeCount + 1;
            int outputEnd = inputAndBiasNodeCount + networkDef.OutputNodeCount;

            //how many hidden???
            int hiddenCount = networkDef.NodeList.Count - outputEnd;

            //now we loop through nodes looking for the various types
            double[] inputsToOutputs = new double[inputAndBiasNodeCount];

            //create all our hidden to output weights 
            double[] hiddenToOutputs = new double[hiddenCount];

            //create all the input -> hidden layer weights
            double[][] hiddenWeights = new double[hiddenCount][];

            //store all the hidden weights 
            for (int i = 0; i < hiddenCount; i++)
                hiddenWeights[i] = new double[inputAndBiasNodeCount];

            //networkDef.ActivationFnLibrary.GetFunction(
            IActivationFunction activationOut = null;

            //grab actiation from an output node
            activationOut = networkDef.ActivationFnLibrary.GetFunction(networkDef.NodeList[networkDef.InputNodeCount + 1].ActivationFnId);

            for (int i = 0; i < networkDef.ConnectionList.Count; i++)
            {
                var conn = networkDef.ConnectionList[i];

                uint sourceNode = conn.SourceNodeId;
                uint targetNode = conn.TargetNodeId;

                //from input nodes directly to the output node
                if (sourceNode < inputAndBiasNodeCount && targetNode < outputEnd)
                {
                    //set it up!
                    inputsToOutputs[sourceNode] = conn.Weight;
                }
                else if (sourceNode < inputAndBiasNodeCount && targetNode >= outputEnd)
                {
                    //we are going from the inputs to a hidden node

                    //targetNode = hidden node to hook into, source node = input we're reading with this connection
                    hiddenWeights[targetNode - outputEnd][sourceNode] = conn.Weight;
                }
                else if (sourceNode >= outputEnd && targetNode < outputEnd)
                {
                    //we are going from a hidden node to the output node
                    //remember conn.soureid is >= outputend for hidden nodes, so we need to 0 index into our array, so we subtract outputend
                    hiddenToOutputs[sourceNode - outputEnd] = conn.Weight;
                }
                else
                {
                    throw new NotImplementedException("Didn't know a connection could exist in this state for hidden node feautres");
                }
            }

            return new SingleOutputNetworkWithHidden(inputsToOutputs, hiddenWeights, hiddenToOutputs, activationOut);
            //SingleOutputNetworkWithHidden soNetwork = new SingleOutputNetworkWithHidden(inputsToOutputs, hiddenWeights, hiddenToOutputs, activationOut);
            //testNetowrks(soNetwork, networkDef);
            //return soNetwork;
        }

        static void testNetowrks(SingleOutputNetworkWithHidden soNetwork, INetworkDefinition networkDef)
        {
           double[][] hiddenWeights =  soNetwork.InputsToHiddenWeights;
           int hiddenCount = hiddenWeights.Length;

            var fastAcyclic = FastAcyclicNetworkFactory.CreateFastAcyclicNetwork(networkDef, true);
            var cyclic = CyclicNetworkFactory.CreateCyclicNetwork(networkDef, NetworkActivationScheme.CreateCyclicFixedTimestepsScheme(2));

            var customFastAcyclic = CreateFastAcyclicNetwork(networkDef);

            var randomInput = new double[networkDef.InputNodeCount];

            Random r = new Random();
            double errorSum = 0;
            double cycleError = 0;
            double acError = 0;
            double customFast = 0;
            for (var t = 0; t < 100; t++)
            {
                for (int i = 0; i < randomInput.Length; i++)
                {
                    randomInput[i] = r.NextDouble();
                }
                //reset 
                fastAcyclic.ResetState();

                //copy from our random for activation
                fastAcyclic.InputSignalArray.CopyFrom(randomInput, 0);

                //activate
                fastAcyclic.Activate();

                double fastOutput = fastAcyclic.OutputSignalArray[0];


                //reset 
                cyclic.ResetState();

                //copy from our random for activation
                cyclic.InputSignalArray.CopyFrom(randomInput, 0);

                //activate
                cyclic.Activate();

                double cyclicOutput = cyclic.OutputSignalArray[0];

                //reset 
                customFastAcyclic.ResetState();

                //copy from our random for activation
                customFastAcyclic.InputSignalArray.CopyFrom(randomInput, 0);

                //activate
                customFastAcyclic.Activate();

                double customFastOutput = customFastAcyclic.OutputSignalArray[0];

                soNetwork.ActivateDotProduct(randomInput);

                //grab our other output
                double singleOutput = soNetwork.LastActivation;

                //compare!
                errorSum += Math.Abs(fastOutput - singleOutput);
                cycleError += Math.Abs(cyclicOutput - singleOutput);
                //any error betwen the two types?
                acError += Math.Abs(cyclicOutput - fastOutput);

                customFast += Math.Abs(cyclicOutput - customFastOutput);
            }


            Console.WriteLine("Network error sum: " + errorSum);

            if (errorSum > 0.0001 || customFast > .0001)
            {
                bool allZero = false;
                for (int i = 0; i < hiddenWeights.Length; i++)
                {
                    bool iz = true;
                    for (int j = 0; j < hiddenWeights[i].Length; j++)
                        iz = iz && hiddenWeights[i][j] == 0.0;

                    allZero = allZero || iz;
                }

                if (allZero)
                    Console.WriteLine("There exists an all zero weight pattern for hidden node");

                errorSum = 0;
                List<double> errorTerms = new List<double>();
                for (var t = 0; t < 100; t++)
                {
                    for (int i = 0; i < randomInput.Length; i++)
                    {
                        randomInput[i] = r.NextDouble();
                    }
                    //reset 
                    fastAcyclic.ResetState();

                    //copy from our random for activation
                    fastAcyclic.InputSignalArray.CopyFrom(randomInput, 0);

                    //activate
                    fastAcyclic.Activate();

                    var actArray = (fastAcyclic as OpenAcyclicNetwork).ActivationArray;

                    double hiddenSum = 0;
                    int lastIx = actArray.Length - 2;
                    for (int h = 0; h < hiddenCount; h++)
                    {
                        hiddenSum += actArray[lastIx - h];
                    }

                    double fastOutput = fastAcyclic.OutputSignalArray[0];

                    //reset 
                    cyclic.ResetState();

                    //copy from our random for activation
                    cyclic.InputSignalArray.CopyFrom(randomInput, 0);

                    //activate
                    cyclic.Activate();

                    double cyclicOutput = cyclic.OutputSignalArray[0];

                    //reset 
                    customFastAcyclic.ResetState();

                    //copy from our random for activation
                    customFastAcyclic.InputSignalArray.CopyFrom(randomInput, 0);

                    //activate
                    customFastAcyclic.Activate();

                    double customFastOutput = customFastAcyclic.OutputSignalArray[0];

                    soNetwork.ActivateDotProduct(randomInput);

                    //grab our other output
                    double singleOutput = soNetwork.LastActivation;

                    errorTerms.Add(Math.Abs(fastOutput - singleOutput));

                    //compare!
                    errorSum += Math.Abs(fastOutput - singleOutput);
                }
            }
            else
            {
                bool allZero = false;
                for (int i = 0; i < hiddenWeights.Length; i++)
                {
                    bool iz = true;
                    for (int j = 0; j < hiddenWeights[i].Length; j++)
                        iz = iz && hiddenWeights[i][j] == 0.0;

                    allZero = allZero || iz;
                }

                if (allZero)
                    Console.WriteLine("There exists an all zero weight pattern for hidden node");
            }


        }

       

        #endregion

       
    }
}
