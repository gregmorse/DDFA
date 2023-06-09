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
using System.Linq;
using SharpNeat.Network;
using SharpNeat.Phenomes;

namespace SharpNeat.Decoders.HyperNeat
{
    /// <summary>
    /// HyperNEAT substrate. Encapsulates substrate nodes in sets and connections. Connections can be defined explicitly or 
    /// by providing mapping functions that map (connect) between nodes in sets. Node sets can be arranged as layers, however
    /// there is no limitation on node positions within the substrate - nodes in a set can be distributed throughout the substrate
    /// with no restrictions based on e.g. where nodes in other sets are located.
    /// </summary>
    public class Substrate
    {
        /// <summary>
        /// The maximum number of substrate conenctions that we cache when using _nodeSetMappingList. If the number of 
        /// connections is less then this then we cache the susbstrate connections to avoid having to invoke the mapping 
        /// functions when creating/growing a network fromt the substrate.
        /// </summary>
        const int ConnectionCountCacheThreshold = 50000;
        /// <summary>
        /// Substrate nodes, represented as distinct sets of nodes. By convention the first and second sets in the
        /// list represent the input and output noes respectively. All other sets represent hidden nodes.
        /// </summary>
        readonly List<SubstrateNodeSet> _nodeSetList;
        /// <summary>
        /// The activation function library allocated to the networks that are 'grown' from the substrate.
        /// _activationFnId refers to a function in this library.
        /// </summary>
        readonly IActivationFunctionLibrary _activationFnLibrary;
        /// <summary>
        /// The activation function ID that is uniformly allocated to all nodes in the netorks that are 'grown' 
        /// from the substrate.
        /// </summary>
        readonly int _activationFnId;
        /// <summary>
        /// A list of mapping functions that provide a means of obtaining a list of substrate connections 
        /// from the _nodeSetList. 
        /// </summary>
        readonly List<NodeSetMapping> _nodeSetMappingList;
        /// <summary>
        /// Pre-built set of substrate connections.
        /// </summary>
        readonly List<SubstrateConnection> _connectionList;
        /// <summary>
        /// A hint to the method creating networks from substrate - approximate number of connections that can be 
        /// expected to be grown by the substrate and it's mapping functions.
        /// </summary>
        readonly int _connectionCountHint;
        /// <summary>
        /// The weight threshold below which substrate connections are not created in grown networks.
        /// </summary>
        readonly double _weightThreshold;
        /// <summary>
        /// Defines the weight range of grown connections (+-maxWeight).
        /// </summary>
        readonly double _maxWeight;
        /// <summary>
        /// Precalculated value for rescaling grown conenctions to the required weight range as described by _maxWeight.
        /// </summary>
        readonly double _weightRescalingCoeff;
        /// <summary>
        /// Pre-built node list for creating new concrete network instances. This can be prebuilt because
        /// the set of nodes remains the same for each network instantiation, only the connections differ between
        /// instantiations.
        /// </summary>
        readonly NodeList _netNodeList;
        /// <summary>
        /// The number of input nodes.
        /// </summary>
        readonly int _inputNodeCount;
        /// <summary>
        /// The number of output nodes.
        /// </summary>
        readonly int _outputNodeCount;
        /// <summary>
        /// Dimensionality of the substrate. The number of coordinate values in a node position; typically 2D or 3D.
        /// </summary>
        readonly int _dimensionality;

        //DFN: Added LEO to substrate construction
        readonly bool _useLEO;
        readonly bool _normalizeWeights;
        readonly bool _singleOutput;
        readonly int _LEOPosition;
        readonly int _inputLists;

        #region Constructor

        /// <summary>
        /// Construct a substrate with the provided node sets and a predetermined set of connections. 
        /// </summary>
        /// <param name="nodeSetList">Substrate nodes, represented as distinct sets of nodes. By convention the first and second
        /// sets in the list represent the input and output noes respectively. All other sets represent hidden nodes.</param>
        /// <param name="activationFnLibrary">The activation function library allocated to the networks that are 'grown' from the substrate.</param>
        /// <param name="activationFnId">The ID of an activation function function in activationFnLibrary. This is the activation function 
        /// ID assigned to all nodes in networks that are 'grown' from the substrate. </param>
        /// <param name="weightThreshold">The weight threshold below which substrate connections are not created in grown networks.</param>
        /// <param name="maxWeight">Defines the weight range of grown connections (+-maxWeight).</param>
        /// <param name="connectionList">A predetermined list of substrate connections.</param>
        public Substrate(List<SubstrateNodeSet> nodeSetList, 
                         IActivationFunctionLibrary activationFnLibrary, int activationFnId,
                         double weightThreshold, double maxWeight,
                         List<SubstrateConnection> connectionList, bool normalizeWeight, bool singleOutput, bool useLEO, int LEOPosition = -1, int inputCountOverride = 0, int inputLists = 1)
        {
            VaildateSubstrateNodes(nodeSetList);

            _nodeSetList = nodeSetList;
            _inputNodeCount =  _nodeSetList[0].NodeList.Count;
            if (inputCountOverride > 0)
                _inputNodeCount = inputCountOverride;
            _outputNodeCount = _nodeSetList[inputLists].NodeList.Count;
            _dimensionality = _nodeSetList[0].NodeList[0]._position.GetUpperBound(0) + 1;

            _activationFnLibrary = activationFnLibrary;
            _activationFnId = activationFnId;

            _weightThreshold = weightThreshold;
            _maxWeight = maxWeight;
            _weightRescalingCoeff = _maxWeight / (1.0 - _weightThreshold);

            // Set total connection count hint value (includes additional connections to a bias node).
            _connectionList = connectionList;
            _connectionCountHint = connectionList.Count + CalcBiasConnectionCountHint(nodeSetList);

            _inputLists = inputLists;
            // Pre-create the network definition node list. This is re-used each time a network is created from the substrate.
            _netNodeList = CreateNetworkNodeList();

            //DFN: Added LEO to substrate construction
            _useLEO = useLEO;
            _singleOutput = singleOutput;
            _normalizeWeights = normalizeWeight;
            _LEOPosition = LEOPosition;
            
        }

        /// <summary>
        /// Constructs with the provided substrate nodesets and mappings that describe how the nodesets are to be connected up.
        /// </summary>
        /// <param name="nodeSetList">Substrate nodes, represented as distinct sets of nodes. By convention the first and second
        /// sets in the list represent the input and output noes respectively. All other sets represent hidden nodes.</param>
        /// <param name="activationFnLibrary">The activation function library allocated to the networks that are 'grown' from the substrate.</param>
        /// <param name="activationFnId">The ID of an activation function function in activationFnLibrary. This is the activation function 
        /// ID assigned to all nodes in networks that are 'grown' from the substrate. </param>
        /// <param name="weightThreshold">The weight threshold below which substrate connections are not created in grown networks.</param>
        /// <param name="maxWeight">Defines the weight range of grown connections (+-maxWeight).</param>/// 
        /// <param name="nodeSetMappingList">A list of mappings between node sets that defines what connections to create between substrate nodes.</param>
        public Substrate(List<SubstrateNodeSet> nodeSetList, 
                         IActivationFunctionLibrary activationFnLibrary, int activationFnId,
                         double weightThreshold, double maxWeight,
                         List<NodeSetMapping> nodeSetMappingList, bool normalizeWeight, bool singleOutput, bool useLEO, int LEOPosition = -1, int inputCountOverride = 0, int inputLists = 1)
        {
            VaildateSubstrateNodes(nodeSetList);

            _nodeSetList = nodeSetList;
            _inputNodeCount =  _nodeSetList[0].NodeList.Count;
            if (inputCountOverride > 0)
                _inputNodeCount = inputCountOverride;
            _outputNodeCount = _nodeSetList[inputLists].NodeList.Count;
            _dimensionality = _nodeSetList[0].NodeList[0]._position.GetUpperBound(0) + 1;

            _activationFnLibrary = activationFnLibrary;
            _activationFnId = activationFnId;

            _weightThreshold = weightThreshold;
            _maxWeight = maxWeight;
            _weightRescalingCoeff = _maxWeight / (1.0 - _weightThreshold);

            _nodeSetMappingList = nodeSetMappingList;
            
            // Get an estimate for the number of connections defined by mappings.
            int nonBiasConnectionCountHint = 0;
            foreach(NodeSetMapping mapping in nodeSetMappingList) {
                nonBiasConnectionCountHint += mapping.GetConnectionCountHint(nodeSetList);
            }

            if(nonBiasConnectionCountHint <= ConnectionCountCacheThreshold)
            {   
                // Pre-generate the substrate connections and store them in a list.
                _connectionList = _connectionCountHint == 0 ? new List<SubstrateConnection>() : new List<SubstrateConnection>(nonBiasConnectionCountHint);
                foreach(NodeSetMapping mapping in nodeSetMappingList)
                {
                    IEnumerable<SubstrateConnection> connectionSequence = mapping.GenerateConnections(nodeSetList);
                    foreach(SubstrateConnection conn in connectionSequence) {
                        _connectionList.Add(conn);
                    }
                }
                _connectionList.TrimExcess();
            }

            // Set total connection count hint value (includes additional connections to a bias node).
            _connectionCountHint = nonBiasConnectionCountHint + CalcBiasConnectionCountHint(nodeSetList);

            _inputLists = inputLists;
            // Pre-create the network definition node list. This is re-used each time a network is created from the substrate.
            _netNodeList = CreateNetworkNodeList();

            //DFN: Added LEO to substrate construction
            _useLEO = useLEO;
            _singleOutput = singleOutput;
            _normalizeWeights = normalizeWeight;
            _LEOPosition = LEOPosition;
            
        }

      
        public bool DisableBias
        {
            get;
            set;
        }

        #endregion

        #region Properties

        /// <summary>
        /// Gets the list of substrate node sets. By convention the first nodeset describes the inputs nodes and the
        /// first node of that set describes the bias node. The last nodeset describes the output nodes.
        /// </summary>
        public List<SubstrateNodeSet> NodeSetList
        {
            get { return _nodeSetList; }
        }

        /// <summary>
        /// DFN: Does this substrate use LEO for weight expression
        /// </summary>
        public bool UseLEO
        {
            get { return _useLEO; }
        }
        #endregion

        #region Public Methods

        /// <summary>
        /// Create a network definition by querying the provided IBlackBox (typically a CPPN) with the 
        /// substrate connection endpoints.
        /// </summary>
        /// <param name="blackbox">The HyperNEAT CPPN that defines the strength of connections between nodes on the substrate.</param>
        /// <param name="lengthCppnInput">Optionally we provide a connection length input to the CPPN.</param>
        public INetworkDefinition CreateNetworkDefinition(IBlackBox blackbox, bool lengthCppnInput)
        {
            // Get the sequence of substrate connections. Either a pre-built list or a dynamically 
            // generated sequence.
            IEnumerable<SubstrateConnection> connectionSequence = _connectionList ?? GetConnectionSequence();

            // Iterate over substrate connections. Determine each connection's weight and create a list
            // of network definition connections.
            ISignalArray inputSignalArr = blackbox.InputSignalArray;
            ISignalArray outputSignalArr = blackbox.OutputSignalArray;
            ConnectionList networkConnList = new ConnectionList(_connectionCountHint);
            //we only have x,y as inputs (2 dimensional)
            int lengthInputIdx = _singleOutput ? _dimensionality :   _dimensionality + _dimensionality;

            //DFN: this is where we read the proper CPPN output for the weight for each substrate connection -- determined by what layer is connected to what
            //this is for multi-modal support in the future
            int cppnOutputIx;
            foreach(SubstrateConnection substrateConn in connectionSequence)
            {
                cppnOutputIx = substrateConn._cppnOutputIndex;

                // Assign the connection's endpoint position coords to the CPPN/blackbox inputs. Note that position dimensionality is not fixed.
                //DFN: If _singleOutput that implies there is a single output
                //therefore, we can skip this, and simply input our x,y of the 2D, and it knows the output anyways (there is only 1 possible)
                if (_singleOutput)
                {
                    inputSignalArr[0] = substrateConn._srcNode._position[0];
                    inputSignalArr[1] = substrateConn._srcNode._position[1];
                }
                else
                {
                    for (int i = 0; i < _dimensionality; i++)
                    {
                        inputSignalArr[i] = substrateConn._srcNode._position[i];
                        inputSignalArr[i + _dimensionality] = substrateConn._tgtNode._position[i];
                    }
                }

                // Optional connection length input.
                if(lengthCppnInput) {
                    inputSignalArr[lengthInputIdx] = CalculateConnectionLength(substrateConn._srcNode._position, substrateConn._tgtNode._position);
                }

                // Reset blackbox state and activate it.
                blackbox.ResetState();
                blackbox.Activate();

                // Read connection weight from output 0.
                double weight = outputSignalArr[cppnOutputIx];

                //DFN: LEO implementation
                if (_useLEO)
                {
                    //shoudl we connect or not? 
                    //only if our LEO node is > 0 -- LEO node is a step function
                    int leoPos;
                    if (_LEOPosition == -1)
                        leoPos = cppnOutputIx + 1;
                    else
                        leoPos = _LEOPosition;
                    
                    if (outputSignalArr[leoPos] > 0)
                    {
                        // For weights passed by LEO we re-scale into the range [-_maxWeight,_maxWeight],
                        weight = weight * _weightRescalingCoeff;

                        //Console.WriteLine("Src: ({0}, {1}), Tgt: ({2}, {3})",
                        //    substrateConn._srcNode._position[0], substrateConn._srcNode._position[1],
                        //    substrateConn._tgtNode._position[0], substrateConn._tgtNode._position[1]);

                        //waaaaaaaaaa
                        // Create network definition connection and add to list.
                        networkConnList.Add(new NetworkConnection(substrateConn._srcNode._id,
                                                                  substrateConn._tgtNode._id, weight));
                    }

                }
                else
                {
                    // Skip connections with a weight magnitude less than _weightThreshold.
                    double weightAbs = Math.Abs(weight);
                    if (weightAbs > _weightThreshold)
                    {
                        // For weights over the threshold we re-scale into the range [-_maxWeight,_maxWeight],
                        // assuming IBlackBox outputs are in the range [-1,1].
                        weight = (weightAbs - _weightThreshold) * _weightRescalingCoeff * Math.Sign(weight);

                        // Create network definition connection and add to list.
                        networkConnList.Add(new NetworkConnection(substrateConn._srcNode._id,
                                                                  substrateConn._tgtNode._id, weight));
                    }  
                }
            }

            if (!DisableBias)
            {
                // Additionally we create connections from each hidden and output node to a bias node that is not defined at any 
                // position on the substrate. The motivation here is that a each node's input bias is independent of any source
                // node (and associated source node position on the substrate). That we refer to a bias 'node' is a consequence of how input 
                // biases are handled in NEAT - with a specific bias node that other nodes can be connected to.
                int setCount = _nodeSetList.Count;
                for (int i = _inputLists; i < setCount; i++)
                {
                    SubstrateNodeSet nodeSet = _nodeSetList[i];
                    foreach (SubstrateNode node in nodeSet.NodeList)
                    {
                        // Assign the node's position coords to the blackbox inputs. The CPPN inputs for source node coords are set to zero when obtaining bias values.
                        //DFN: If _singleOutput that implies there is a single output
                        //therefore, we can skip this, and simply input our x,y of the 2D, and it knows the output anyways (there is only 1 possible)
                        if (_singleOutput)
                        {
                            inputSignalArr[0] = 0;
                            inputSignalArr[1] = 0;
                        }
                        else
                        {
                            for (int j = 0; j < _dimensionality; j++)
                            {
                                inputSignalArr[j] = 0.0;
                                inputSignalArr[j + _dimensionality] = node._position[j];
                            }
                        }

                        // Optional connection length input.
                        if (lengthCppnInput)
                        {
                            inputSignalArr[lengthInputIdx] = CalculateConnectionLength(node._position);
                        }

                        // Reset blackbox state and activate it.
                        blackbox.ResetState();
                        blackbox.Activate();

                        // Read bias connection weight from output 1.
                        double weight = outputSignalArr[0];

                        //DFN: LEO implementation
                        if (_useLEO)
                        {
                            //shoudl we connect or not? 
                            //only if our LEO node is > 0 -- LEO node is a step function
                            if (outputSignalArr[1] > 0)
                            {
                                // For weights passed by LEO we re-scale into the range [-_maxWeight,_maxWeight],
                                weight = weight * _weightRescalingCoeff;

                                //waaaaaaaaaa
                                // Create network definition connection and add to list.
                                networkConnList.Add(new NetworkConnection(0, node._id, weight));
                            }
                        }
                        else
                        {
                            // Skip connections with a weight magnitude less than _weightThreshold.
                            double weightAbs = Math.Abs(weight);
                            if (weightAbs > _weightThreshold)
                            {
                                // For weights over the threshold we re-scale into the range [-_maxWeight,_maxWeight],
                                // assuming IBlackBox outputs are in the range [-1,1].
                                weight = (weightAbs - _weightThreshold) * _weightRescalingCoeff * Math.Sign(weight);

                                // Create network definition connection and add to list. Bias node is always ID 0.
                                networkConnList.Add(new NetworkConnection(0, node._id, weight));
                            }
                        }



                    }
                }
            }

            // Check for no connections.
            // If no connections were generated then there is no point in further evaulating the network.
            // However, null is a valid response when decoding genomes to phenomes, therefore we do that here.
            if(networkConnList.Count == 0) {
                return null;
            }

            //DFN: Paul - Update -- removed bias from the normalization process 
            //it could cause weird issues if the bias is generated separately in weights, and it's the only - or + weight
            //then it wuold get shot to maximum weight -- defeating the purpose
            if (_normalizeWeights)
            {

                Dictionary<string, double> maxPositive = new Dictionary<string, double>();
                Dictionary<string, double> maxNegative = new Dictionary<string, double>();

                double weightRange = 3.0;

                for (int i = 0; i < networkConnList.Count; i++)
                {
                    INetworkConnection nc = networkConnList[i];

                    //don't mess with bias weights
                    if (nc.SourceNodeId == 0)
                        continue;
                    
                    //align by targets -- if you are teh same target, you get normalized for that target
                    string conn = "" + nc.TargetNodeId;

                    if (!maxPositive.ContainsKey(conn))
                    {
                        maxPositive.Add(conn, 0.0);
                        maxNegative.Add(conn, 0.0);
                    }

                    if (nc.Weight >= 0)
                        maxPositive[conn] += nc.Weight;
                    else
                        maxNegative[conn] += nc.Weight;
                }

                for (int i = 0; i < networkConnList.Count; i++)
                {
                    INetworkConnection nc = networkConnList[i];

                    //don't mess with bias weights
                    if (nc.SourceNodeId == 0)
                        continue;

                    //what conn are we, we need to index
                    string conn = "" + nc.TargetNodeId;

                    //we use divide to prevent NaN issues when max pos or max neg == 0 and we have 0/0
                    double divide;
                    if (nc.Weight >= 0)
                    {
                        divide = (maxPositive[conn] == 0.0 ? 1.0 : maxPositive[conn]);

                        networkConnList[i] = new NetworkConnection(nc.SourceNodeId, nc.TargetNodeId, weightRange * nc.Weight / divide);
                    }
                    else
                    {
                        divide = (maxNegative[conn] == 0.0 ? 1.0 : maxNegative[conn]);

                        networkConnList[i] = new NetworkConnection(nc.SourceNodeId, nc.TargetNodeId, -1.0 * weightRange * nc.Weight / divide);
                    }
                }
            }




            // Construct and return a network definition.
            NetworkDefinition networkDef = new NetworkDefinition(_inputNodeCount, _outputNodeCount,
                                                                 _activationFnLibrary, _netNodeList, networkConnList);

            // Check that the definition is valid and return it.
            Debug.Assert(networkDef.PerformIntegrityCheck());
            return networkDef;
        }

        #endregion  

        #region Private Methods
        
        private void VaildateSubstrateNodes(List<SubstrateNodeSet> nodeSetList)
        {
            // Baseline validation tests. There should be at least two nodesets (input and output sets), and each of those must have at least one node.
            if(nodeSetList.Count < 2) {
                throw new ArgumentException("Substrate requires a minimum of two NodeSets - one each for input and outut nodes.");
            }

            // Input nodes.
            if(nodeSetList[0].NodeList.Count == 0) {
                throw new ArgumentException("Substrate input nodeset must have at least one node.");
            }

            if(nodeSetList[1].NodeList.Count == 0) {
                throw new ArgumentException("Substrate output nodeset must have at least one node.");
            }

            // Check for duplicate IDs or ID zero (reserved for bias node).
            Dictionary<uint,object> idDict = new Dictionary<uint,object>();
            foreach(SubstrateNodeSet nodeSet in nodeSetList)
            {
                foreach(SubstrateNode node in nodeSet.NodeList)
                {
                    if(0u == node._id) {
                        throw new ArgumentException("Substrate node with invalid ID of 0 (reserved for bias node).");
                    }
                    if(idDict.ContainsKey(node._id)) {
                        throw new ArgumentException(string.Format("Substrate node with duplicate ID of [{0}]", node._id));
                    }
                    idDict.Add(node._id, null);
                }
            }

            // Check ID ordering.
            // Input node IDs should be contiguous and ordered sequentially after the bais node with ID 0.
            SubstrateNodeSet inputNodeSet = nodeSetList[0];
            int count = inputNodeSet.NodeList.Count;
            int expectedId = 1;
            for(int i=0; i<count; i++, expectedId++)
            {
                if(inputNodeSet.NodeList[i]._id != expectedId) {
                    throw new ArgumentException(string.Format("Substrate input node with unexpected ID of [{0}]. Ids should be contguous and starting from 1.", inputNodeSet.NodeList[i]._id));
                }
            }

            // Output node IDs should be contiguous and ordered sequentially after the last input ID.
            SubstrateNodeSet outputNodeSet = nodeSetList[1];
            count = outputNodeSet.NodeList.Count;
            for(int i=0; i<count; i++, expectedId++)
            {
                if(outputNodeSet.NodeList[i]._id != expectedId) {
                    throw new ArgumentException(string.Format("Substrate output node with unexpected ID of [{0}].", outputNodeSet.NodeList[i]._id));
                }
            }

            // Hidden node IDs don't have to be contiguous but must have IDs greater than all of the input and output IDs.
            int setCount = nodeSetList.Count;
            for(int i=2; i<setCount; i++)
            {
                SubstrateNodeSet hiddenNodeSet = nodeSetList[i];
                foreach(SubstrateNode node in hiddenNodeSet.NodeList)
                {
                    if(node._id < expectedId) {
                        throw new ArgumentException(string.Format("Substrate hidden node with unexpected ID of [{0}] (must be greater than the last output node ID [{1}].",
                                                                  hiddenNodeSet.NodeList[i]._id, expectedId-1));
                    }
                }
            }
        }

        /// <summary>
        /// Calculate the maximum number of possible bias connections. Input nodes don't have a bias therefore this value
        /// is the number of hidden and output nodes.
        /// </summary>
        private int CalcBiasConnectionCountHint(List<SubstrateNodeSet> nodeSetList)
        {
            // Count nodes in all nodesets except for the first (input) nodeset.
            int total = 0;
            int setCount = nodeSetList.Count;
            for(int i=1; i<setCount; i++)
            {
                total += nodeSetList[i].NodeList.Count;
            }
            return total;
        }

        /// <summary>
        /// Pre-build the network node list used for constructing new networks 'grown' on the substrate.
        /// This can be prebuilt because the set of nodes remains the same for each network instantiation,
        /// only the connections differ between instantiations.
        /// </summary>
        private NodeList CreateNetworkNodeList()
        {
            // Count the total number of nodes.
            int nodeCount = 0;
            foreach(SubstrateNodeSet set in _nodeSetList) {
                nodeCount += set.NodeList.Count;
            }
            // Count the additional bias node (not explicitly defined on the substrate).
            nodeCount++;

            // Allocate storage for the nodes.
            NodeList nodeList = new NodeList(nodeCount);

            // Create bias node.
            // Note. The nodes are created in the order of inputs, outputs and then hidden. This is the order required when constructing
            // instances of the NeatGenome and NetworkDefinition classes. The requirement comes about through internal implementation of 
            // those classes - see comments on those classes for more info.
            nodeList.Add(new NetworkNode(0u, NodeType.Bias, _activationFnId));

            SubstrateNodeSet nodeSet;
            int setNodeCount;
            // Create input nodes. By convention the first nodeset describes the input nodes (not including the bias).  _inputLists contains the number of lists used for inputs (for multispatial input spaces)
            for (int j = 0; j < _inputLists; j++)
            {
                nodeSet = _nodeSetList[j];
                setNodeCount = nodeSet.NodeList.Count;

                // Create input nodes.
                for (int i = 0; i < setNodeCount; i++)
                {
                    nodeList.Add(new NetworkNode(nodeSet.NodeList[i]._id, NodeType.Input, _activationFnId));
                }
            }
            // Create output nodes. By convention the second nodeset describes the output nodes.  If there is more than one input list, it will be offset to = the number of input lists
            nodeSet = _nodeSetList[_inputLists];
            setNodeCount = nodeSet.NodeList.Count;
            for(int i=0; i<setNodeCount; i++) {
                nodeList.Add(new NetworkNode(nodeSet.NodeList[i]._id, NodeType.Output, _activationFnId));
            }

            // Create hidden nodes (if any). All nodesets after the input and output nodesets define hidden nodes.
            int setCount = _nodeSetList.Count;
            for(int nodeSetIdx = 2 + (_inputLists - 1); nodeSetIdx<setCount; nodeSetIdx++)
            {
                nodeSet = _nodeSetList[nodeSetIdx];
                setNodeCount = nodeSet.NodeList.Count;
                for(int i=0; i<setNodeCount; i++) {
                    nodeList.Add(new NetworkNode(nodeSet.NodeList[i]._id, NodeType.Hidden, _activationFnId));
                }
            }

            return nodeList;
        }

        private IEnumerable<SubstrateConnection> GetConnectionSequence()
        {
            // If the connections are in a list then return the list.
            if(null != _connectionList) {
                return _connectionList;
            }

            // Generate the connections from the nodeset mappings.
            int count = _nodeSetMappingList.Count;
            if(0 == count)
            {   // No mappings.
                return null;
            }

            // Concatenate the IEnumerables from each mapping to produce one all-encompassing IEnumerable.
            IEnumerable<SubstrateConnection> enumerable = _nodeSetMappingList[0].GenerateConnections(_nodeSetList);
            for(int i=1; i<count; i++) {
                enumerable = Enumerable.Concat(enumerable, _nodeSetMappingList[i].GenerateConnections(_nodeSetList));
            }

            return enumerable;
        }

        /// <summary>
        /// Calculates the euclidean distance between two points in N dimensional space.
        /// </summary>
        private double CalculateConnectionLength(double[] a, double[] b)
        {
            double acc = 0.0;
            for(int i=0; i<a.Length; i++)  {
                acc += (a[i]-b[i]) * (a[i]-b[i]);
            }
            return Math.Sqrt(acc);
        }

        /// <summary>
        /// Calculates the euclidean distance between a point and the origin.
        /// </summary>
        private double CalculateConnectionLength(double[] a)
        {
            double acc = 0.0;
            for(int i=0; i<a.Length; i++)  {
                acc += a[i] * a[i];
            }
            return Math.Sqrt(acc);
        }

        #endregion
    }
}
