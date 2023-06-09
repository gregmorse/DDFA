using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Phenomes;

namespace SharpNeat.Network
{

     public enum NodeActivationType
    {   
         Gaussian = 0,
         Linear,
         Mult,
         Sigmoid,
         BipolarSigmoid,
         Sine,
         StepFunction
     }

    // public enum NodeActivationType
    //{   
    //     Gaussian = 0,
    //     Linear,
    //     Mult,
    //     Sigmoid,
    //     BipolarSigmoid,
    //     Sine,
    //     StepFunction
    // }

     public struct ByteConnection
     {
         public int source;
         public int target;
         public double weight;
         public int layer;
     }

    public class AcyclicByteNetwork : IBlackBox
    {
        #region Network Speedup Variables
        List<int[]> allOperations = new List<int[]>();
        double[] storedValues;
        Dictionary<INetworkNode, List<int>> inputLocations;
        Dictionary<int, INetworkNode> outputLocations;
        List<INetworkConnection> orderedConnections;
        Dictionary<int, int> nodeDepths;
        Dictionary<int, List<int>> nodesToActivateAtDepth = new Dictionary<int, List<int>>();
        int hiddenNodeCount = 0;
        int maxNodeDepth = 0;
        IActivationFunction[] nodeActivations;
        #endregion

        //int[] connectionToSource;
        //double[] connectionWeightList;

        double[] nodeRegisters;

        public AcyclicByteNetwork(int inputs, int outputs, IConnectionList iConnList, INodeList nodeList, IActivationFunctionLibrary library)
        {
            InputCount = inputs;
            OutputCount = outputs;

            Dictionary<int, List<INetworkConnection>> incomingConnections = new Dictionary<int, List<INetworkConnection>>();
            Dictionary<int, List<INetworkConnection>> outgoingConnections = new Dictionary<int, List<INetworkConnection>>();

            var nodeCopy = nodeList.ToList();

            //sort ascending id order
            nodeCopy.Sort((x, y) => x.Id.CompareTo(y.Id));

            Dictionary<uint, int> nodeRemap = new Dictionary<uint, int>();
            //now make a mapping from previous ids to new ids, we'll remap our 
            for (int i = 0; i < nodeCopy.Count; i++ )
            {
                var nc = nodeCopy[i];
                nodeRemap.Add(nc.Id, i);
            }

            var connectionList = iConnList.ToList();

            for (int i = 0; i < connectionList.Count; i++)
            {
                var conn = connectionList[i];
                int sourceNode = nodeRemap[conn.SourceNodeId];
                int targetNode = nodeRemap[conn.TargetNodeId];

                //quick replace the connection by making a remapped clone!
                //we must do this before adding to incoming or outgoing to avoid confusion
                conn = new NetworkConnection((uint)sourceNode, (uint)targetNode, conn.Weight);

                if (!incomingConnections.ContainsKey(targetNode))
                {
                    incomingConnections.Add(targetNode, new List<INetworkConnection>());
                }
                incomingConnections[targetNode].Add(conn);

                if (!outgoingConnections.ContainsKey(sourceNode))
                {
                    outgoingConnections.Add(sourceNode, new List<INetworkConnection>());
                }
                outgoingConnections[sourceNode].Add(conn);

                //replace that connection with a new connection with the remapped nodes --
                //we don't need to use node ids, just the rempaped ixs -- everything else remains the same
                connectionList[i] = conn;
            }

            //now we get node depth, let's sort our connections accordingly
            //set up our node depths
            nodeDepths = recursiveGetNodeDepth(nodeCopy, incomingConnections);

            //get the max depth of the network (for activation purposes)
            maxNodeDepth = nodeDepths.Max(x => x.Value);

            foreach(var depthPair in nodeDepths)
            {
                var nodeID = depthPair.Key;
                var depthLevel = depthPair.Value;
                
                if (!nodesToActivateAtDepth.ContainsKey(depthLevel))
                    nodesToActivateAtDepth.Add(depthLevel, new List<int>());

                //nodes being activated for a given depth
                //only hidden or output nodes can be activated
                if(nodeID >= InputCount + 1)
                    nodesToActivateAtDepth[depthLevel].Add(nodeID);
            }

            //going to sort the connection list now
            //sort ascending according to depth of the source node (i.e. where it's coming from)
            connectionList.Sort((x, y) => nodeDepths[(int)x.SourceNodeId] - nodeDepths[(int)y.SourceNodeId]);

            //save this ordered list
            orderedConnections = connectionList;

            //for storing all our node info
            nodeRegisters = new double[nodeCopy.Count];
            
            //always have bias please!
            nodeRegisters[0] = 1.0;

            nodeActivations = nodeCopy.Select(x => library.GetFunction(x.ActivationFnId)).ToArray();

            // Wrap a sub-range of the _activationArr that holds the activation values for the input nodes.
            // Offset is 1 to skip bias neuron (The value at index 1 is the first black box input).
            InputSignalArray = new SignalArray(nodeRegisters, 1, InputCount);

            // Wrap a sub-range of the _activationArr that holds the activation values for the output nodes.
            // Offset is 1 + input to skip bias neuron + input neurons (The value at index 1 is the first black box input).
            OutputSignalArray = new SignalArray(nodeRegisters, 1 + InputCount, OutputCount);

        }

        #region IBlackBox Members
        
        /// <summary>
        /// Gets the number of inputs to the blackbox. This is assumed to be fixed for the lifetime of the IBlackBox.
        /// </summary>
        public int InputCount { get; set; }

        /// <summary>
        /// Gets the number of outputs from the blackbox. This is assumed to be fixed for the lifetime of the IBlackBox.
        /// </summary>
        public int OutputCount { get; set; }


        /// <summary>
        /// Gets an array of input values that feed into the black box. 
        /// </summary>
        public ISignalArray InputSignalArray { get; set; }

        /// <summary>
        /// Gets an array of output values that feed out from the black box. 
        /// </summary>
        public ISignalArray OutputSignalArray { get; set; }

        /// <summary>
        /// Gets a value indicating whether the black box's internal state is valid. It may become invalid if e.g. we ask a recurrent
        /// neural network to relax and it is unable to do so.
        /// </summary>
        public bool IsStateValid { get; set; }

        /// <summary>
        /// Activate the black box. This is a request for the box to accept its inputs and produce output signals
        /// ready for reading from OutputSignalArray.
        /// </summary>
        public void Activate()
        {
            int lastDepth = 0;

            //reset nodes that aren't inputs
            for(var i=InputCount +1; i < nodeRegisters.Length; i++)
                nodeRegisters[i] = 0;

            //activate this first depth -- this would be for any dangling neurons 
            //how have no incoming connections, but some outgoing -- no inputs are activated
            List<int> nodesToActivate = nodesToActivateAtDepth[lastDepth];

            //activate all nodes at this final depth
            for (int nd = 0; nd < nodesToActivate.Count; nd++)
            {
                int nodeID = nodesToActivate[nd];
                nodeRegisters[nodeID] = nodeActivations[nodeID].Calculate(nodeRegisters[nodeID], null);
            }

            for (int i = 0; i < orderedConnections.Count; i++)
            {
                int depth = nodeDepths[(int)orderedConnections[i].SourceNodeId];
                if (depth != lastDepth)
                {
                    nodesToActivate = nodesToActivateAtDepth[depth];

                    //activate all nodes at this depth
                    for (int nd = 0; nd < nodesToActivate.Count; nd++)
                    {
                        int nodeID = nodesToActivate[nd];
                        nodeRegisters[nodeID] = nodeActivations[nodeID].Calculate(nodeRegisters[nodeID], null);
                    }
                    //okay, no more for this level
                    lastDepth = depth;
                }

                //now we're positive our node sources have the right info, just propogate info
                var conn = orderedConnections[i];
                nodeRegisters[conn.TargetNodeId] += nodeRegisters[conn.SourceNodeId] * conn.Weight;
            }

            //activate this last depth -- where there wouldn't be any source connections, please
            nodesToActivate = nodesToActivateAtDepth[maxNodeDepth];

            //activate all nodes at this final depth
            for (int nd = 0; nd < nodesToActivate.Count; nd++)
            {
                int nodeID = nodesToActivate[nd];
                nodeRegisters[nodeID] = nodeActivations[nodeID].Calculate(nodeRegisters[nodeID], null);
            }
        }

        /// <summary>
        /// Reset any internal state.
        /// </summary>
        public void ResetState()
        {
        }

        #endregion

        #region Recurse for depth

        Dictionary<int, int> recursiveGetNodeDepth(List<INetworkNode> nodeList, Dictionary<int, List<INetworkConnection>> incomingConnections)
        {
            Dictionary<int, int> depth = new Dictionary<int,int>();

            for(var i=0; i < nodeList.Count; i++)
            {
                //recurse on every node -- some easier than others
                recurseOnNode(i, incomingConnections, depth);
            }

            return depth;
        }

        void recurseOnNode(int nodeID, Dictionary<int, List<INetworkConnection>> incomingConnections, Dictionary<int, int> depth)
        {
            //don't bother recursing on something you've alreayd finished
            if(depth.ContainsKey(nodeID))
            {
                return;
            }

            //node has zero incoming -- no depth for this node
            if(!incomingConnections.ContainsKey(nodeID))
            {
                //this node doesn't have any incoming -- it's free and clear to go
                depth.Add(nodeID, 0);
                return;
            }

            //otherwise, we need to loop through the incoming connections and process the parents
            var connList = incomingConnections[nodeID];

            var parentNodeIDs = connList.Select(x=>(int)x.SourceNodeId);

            foreach(var parentID in parentNodeIDs)
            {
                //make sure we know it's depth
                recurseOnNode(parentID, incomingConnections, depth);
            }

            //take the max of the parent depths + 1
            int nodeDepth = parentNodeIDs.Max(x=> depth[x]) + 1;

            //set our node depth, then we're done
            depth[nodeID] = nodeDepth;
        }

        #endregion


        //#region Network Speedup Code

        //void initializeByteCodeNetwork()
        //{
        //    //connectionlist must be topologically sorted before calling this function
        //    Dictionary<INetworkNode, List<INetworkConnection>> aggregatedNetworkCons = new Dictionary<INetworkNode, List<INetworkConnection>>();

        //    List<INetworkNode> nodeOrder = new List<INetworkNode>();
        //    Dictionary<INetworkNode, List<int[]>> operationsForNodes = new Dictionary<INetworkNode, List<int[]>>();
        //    inputLocations = new Dictionary<INetworkNode, List<int>>();
         

        //    NetworkNode lastTarget;
        //    List<int> locations;

        //    int opParams = 4;
        //    //first we gather our connections in order 

        //    lastTarget = orderedConnections[0].TargetNodeId;
        //    Dictionary<INetworkNode, int> nodeStorageIndex = new Dictionary<INetworkNode, int>();
        //    foreach (INetworkConnection netCon in orderedConnections)
        //    {
        //        //we want to keep the correct order of node activation, so we keep a list
        //        //Since the connections are topologically sorted, this guarantees nodes activated in this order will be accurate
        //        if (!nodeOrder.Contains(netCon.Target))
        //        {
        //            nodeOrder.Add(netCon.Target);
        //            //we handle initialization here, so we can make code clearer later
        //            nodeStorageIndex.Add(netCon.Target, 0);
        //        }
        //        else
        //        //we've seen this node before, but there was a break between the last target node and this target node
        //        {

        //            //if (netCon.Target != lastTarget)
        //            //    Console.WriteLine("Dunno if this is an issue, but the connections are kind of out of netowkr node order");

        //        }

        //        if (!aggregatedNetworkCons.ContainsKey(netCon.Target))
        //            aggregatedNetworkCons.Add(netCon.Target, new List<NetworkConnection>());

        //        aggregatedNetworkCons[netCon.Target].Add(netCon);

        //        lastTarget = netCon.Target;

        //    }



        //    List<NetworkNode> visited = new List<NetworkNode>();
        //    int previousStorage;

        //    //List<int[]> allOperations = new List<int[]>();

        //    //the number of registers we'll need is = to 3* number of connections minus hidden nodes 
        //    //-- you need a place for the input value, the weight, and the storage
        //    //however, if you have a hidden node, then you already have the input calculated (still need the weight and the storage) -- so minus 1 per hidden node


        //    storedValues = new double[3 * this.orderedConnections.Count];

        //    int opCode = 0;
        //    int param1 = 1;
        //    int param2 = 2;
        //    int register = 3;

        //    outputLocations = new Dictionary<int, INetworkNode>();

        //    int arrayIx = 0;

        //    int opIx = 0;
        //    NetworkConnection netConnection;
        //    foreach (NetworkNode nn in nodeOrder)
        //    {

        //        int[] operation;

        //        List<NetworkConnection> aggCons = aggregatedNetworkCons[nn];
        //        List<int> registers = new List<int>();

        //        //foreach (NetworkConnection ni in aggCons)
        //        //{
        //        //    ;
        //        //}

        //        for (int i = 0; i < aggCons.Count; i++)
        //        {
        //            netConnection = aggCons[i];



        //            //our first operation, multiply weight times 
        //            operation = new int[opParams];
        //            //we do the normal operations here, this is weight * activation -- constants where possible, and references to previously calculated objects

        //            operation[opCode] = (int)Operations.Multi;

        //            //first thing we do, check to see if our source node has been visited, 
        //            if (visited.Contains(netConnection.Source))
        //            {
        //                //If so, we grab the stored value for our first parameter
        //                operation[param1] = nodeStorageIndex[netConnection.Source];
        //                //we don't iuncrement opIx since we don't need to have another storage space

        //                operation[param2] = opIx;
        //                storedValues[opIx, 0] = netConnection.Weight;
        //                storedValues[opIx, 1] = netConnection.Target.NodeID;
        //                storedValues[opIx, 2] = netConnection.Source.NodeID;
        //                storedValues[opIx, 3] = netConnection.SourceNode.OutputLevel;
        //                storedValues[opIx, 4] = 0;
        //                opIx++;

        //            }
        //            else
        //            {
        //                //this is the input from the node -- which is either from an input or from a hidden node (i.e. calculated before)
        //                //We just need an index to reach, we don't mess with stored value
        //                operation[param1] = opIx;

        //                //if you're an input, keep track of the array location -- you will set these later
        //                if (netConnection.Source.Type == NetworkNodeType.Sensor)
        //                {
        //                    if (!inputLocations.ContainsKey(netConnection.Source))
        //                        inputLocations.Add(netConnection.Source, new List<int>());

        //                    inputLocations[netConnection.Source].Add(opIx);
        //                }

        //                opIx++;

        //                //we store the index to act on for multiplication
        //                operation[param2] = opIx;
        //                //now we store in the weight for this operation
        //                storedValues[opIx, 0] = netConnection.Weight;
        //                storedValues[opIx, 1] = netConnection.Target.NodeID;
        //                storedValues[opIx, 2] = netConnection.Source.NodeID;
        //                storedValues[opIx, 3] = netConnection.SourceNode.OutputLevel;
        //                storedValues[opIx, 4] = 0;

        //                opIx++;
        //            }

        //            //then we have to store the results of this multiplication somewhere
        //            operation[register] = opIx;
        //            registers.Add(opIx);

        //            //I KNEW I'D FORGET THIS AND I DID
        //            //we have to keep track of where this node places it's calculations (moving each time there is a new register),
        //            //so that later references will pick it up from the correct index
        //            nodeStorageIndex[nn] = registers.Last();

        //            opIx++;

        //            //Push this operation, we're done with it
        //            allOperations.Add(operation);

        //            //Now-- if you're the last connection, we must do the summation and activation function
        //            if (i == aggCons.Count - 1)
        //            {
        //                //perform the summation
        //                int nodeCalcLocation = registers.Last();
        //                for (int p = 0; p < registers.Count - 1; p++)
        //                {
        //                    operation = new int[opParams];
        //                    //special case activation function
        //                    //if it's multiplying then, isntead of adding everything we just multiply it, easy peazy 
        //                    operation[opCode] = (netConnection.Target.ActivationType == NodeActivationType.Mult) ? (int)Operations.Multi : (int)Operations.Addition;
        //                    operation[param1] = registers[p];
        //                    operation[param2] = nodeCalcLocation;
        //                    operation[register] = nodeCalcLocation;

        //                    //slamma jamma, add the operation
        //                    allOperations.Add(operation);
        //                }

        //                //now perform the activation, unless it's multiplication, then we've already done the correct multiplication

        //                if (netConnection.Target.ActivationType != NodeActivationType.Mult)
        //                {
        //                    operation = new int[opParams];
        //                    operation[opCode] = codeForActivationFunction(netConnection.Target.ActivationType);
        //                    operation[param1] = nodeCalcLocation;
        //                    //param two isn't needed, just the activation function, and the node
        //                    //we default to the same though
        //                    operation[param2] = nodeCalcLocation;
        //                    operation[register] = nodeCalcLocation;

        //                    //add our operation, and we're good
        //                    allOperations.Add(operation);

        //                }

        //                //you never see a node more than once, so we can add it inside here without checking
        //                if (nn.Type == NetworkNodeType.Output)
        //                {
        //                    outputLocations.Add(nodeCalcLocation, nn);
        //                }



        //            }

        //        }

        //        if (!visited.Contains(nn))
        //        {
        //            visited.Add(nn);
        //        }




        //    }
        //}

        //private int codeForActivationFunction(NodeActivationType nodeActivationType)
        //{
        //    switch (nodeActivationType)
        //    {
        //        case NodeActivationType.Gaussian:
        //            return (int)NodeActivationType.Gaussian;
        //        case NodeActivationType.Linear:
        //            return (int)NodeActivationType.Linear;
        //        case NodeActivationType.Mult:
        //            return (int)NodeActivationType.Mult;
        //        case NodeActivationType.Sigmoid:
        //            return (int)NodeActivationType.Sigmoid;
        //        case NodeActivationType.BipolarSigmoid:
        //            return (int)NodeActivationType.BipolarSigmoid;
        //        case NodeActivationType.Sine:
        //            return (int)NodeActivationType.Sine;
        //        case NodeActivationType.StepFunction:
        //            return (int)NodeActivationType.StepFunction;
        //        default:
        //            throw new NotImplementedException("Never seen that activation type before, error in ANN operations translation");

        //    }
        //}

        //public void ActivateOperations(double sigmoidSlope1)
        //{
        //    //we have to set the inputs here
        //    //This is pretty inefficient -- i don't like it
        //    //Will have to think of a better way to do this
        //    foreach (var nnPair in inputLocations)
        //    {
        //        for (int i = 0; i < nnPair.Value.Count; i++)
        //        {
        //            storedValues[nnPair.Value[i], 0] = nnPair.Key.OutputLevel;
        //        }
        //    }


        //    int count = 0;
        //    for (int i = 0; i < storedValues.GetLength(0); i++)
        //    {
        //        //Connection Weight Stored in the third value
        //    }


        //    //in this particular case, we run the operations in order -- we'll make all operations a single int array that we increment
        //    //until finished to further optimize
        //    for (int i = 0; i < allOperations.Count; i++)
        //    {
        //        performOp(allOperations[i], 0, ref storedValues, sigmoidSlope1);
        //    }

        //    //and now all we need to do is move our outputs to the right place
        //    foreach (var ixAndOutPair in outputLocations)
        //    {
        //        ixAndOutPair.Value.OutputLevel = storedValues[ixAndOutPair.Key, 0];
        //    }

        //    //and we're done

        //}

        //bool Contains(double[,] myArray, int ID)
        //{

        //    for (int i = 0; i < myArray.GetLength(0); i++)
        //    {
        //        if (myArray[i, 0] == ID)
        //        {
        //            return true;
        //        }
        //    }

        //    return false;
        //}

        //static void performOp(int[] operation, int startIndex, ref double[,] storedValues, double sigmoidSlope)
        //{

        //    int p1Ix = operation[startIndex + 1];
        //    int p2Ix = operation[startIndex + 2];
        //    int storeIx = operation[startIndex + 3];
        //    double param1, param2;
        //    param1 = storedValues[p1Ix, 0];
        //    param2 = storedValues[p2Ix, 0];
        //    switch ((StepFunction)operation[startIndex])
        //    {
        //        //multiplication and addition are the most common, so they go first
        //        case Operations.Multi:

        //            storedValues[storeIx, 0] = param1 * param2;

        //            break;
        //        case Operations.Addition:
        //            storedValues[storeIx, 0] = param1 + param2;
        //            break;

        //        //These are our activation functions
        //        //The first part of the operation is what defines the activation function
        //        //the second part is the stored double to manimpulate
        //        //everything is stored in place

        //        //we don't do anything, linear is linear
        //        case Operations.Linear:
        //        //we don't do anything for multiplication since it's handled elsewhere in code
        //        case Operations.ActMult:
        //            break;

        //        //sigmoid are generally the outputs

        //        // param1 = storedValues[p1Ix, 0];
        //        case Operations.Sigmoid:
        //            storedValues[p1Ix, 0] = sigmoidActivationFunction(param1, sigmoidSlope);
        //            break;
        //        case Operations.Gaussian:
        //            storedValues[p1Ix, 0] = gaussianActivationFunction(param1);
        //            break;

        //        case Operations.Sine:
        //            storedValues[p1Ix, 0] = sineActivationFunction(param1);
        //            break;
        //        case Operations.Threshhold:
        //            storedValues[p1Ix, 0] = threshholdActivationFunction(param1, NetworkNode.DefaultThreshold);
        //            break;

        //        //case Operations.SigmoidPitched:
        //        //storedValues[p1Ix] = sigmoidActivationFunction(param1);
        //        //break;
        //        //throw new NotImplementedException("Haven't implemented SigmoidPitched -- don't know the difference between normal signmoid and pitched???");
        //        default:
        //            throw new NotImplementedException("Operation code not defined in performOp inside network code");


        //    }



        //}
        //#endregion

//        #region Construction

////        public bool verifiedRecurrent = false;
////        public Network(GenotypeNetwork gn)
////        {
////            Dictionary<NetworkNode, List<NetworkConnection>> nodesAndSourceConnections = new Dictionary<NetworkNode, List<NetworkConnection>>();
////            Dictionary<NetworkNode, List<NetworkConnection>> nodesAndOutConnections = new Dictionary<NetworkNode, List<NetworkConnection>>();
		
////            nodeList = new List<NetworkNode>();
////                connectionList = new List<NetworkConnection>();
////                baseGenotypeNetwork = gn;

////                Hashtable hash = new Hashtable();
////                //gn.NodeList = new List<NetworkNode>();
////                //gn.ConnectionList = new List<NetworkConnection>();

////                // Add all nodes...
////                foreach (NodeGene node in gn.NodesList)
////                {
////                    NetworkNode netNode = new NetworkNode(node.NodeType, node.ID);
////                    netNode.BaseNodeGene = node;
////                    netNode.ActivationType = node.ActivationType;
////                    nodeList.Add(netNode);
                    
////                    if (node.NodeType == NetworkNodeType.Hidden)
////                        hiddenNodeCount++;

////                    nodesAndSourceConnections.Add(netNode, new List<NetworkConnection>());
////                    nodesAndOutConnections.Add(netNode, new List<NetworkConnection>());

////                    if (hash.ContainsKey(node.GetHashCode()))
////                    {
////                    }
////                    else
////                    {
////                        hash.Add(node.GetHashCode(), netNode);
////                    }
////                }

////                // Add all connections ...
////                foreach (ConnectGene con in gn.ConnectList)
////                {
////                    if (con.IsEnabled)
////                    {
////                        NetworkNode inNode = (NetworkNode)hash[con.InNode.GetHashCode()];
////                        NetworkNode outNode = (NetworkNode)hash[con.OutNode.GetHashCode()];

////                        //if (outNode.BaseNodeGene.NodeType == NetworkNodeType.Hidden)
////                        //    Console.WriteLine("der");

////                        NetworkConnection netCon = new NetworkConnection(inNode, outNode, con.Weight, con.IsEnabled, con.InnovID);

////                        nodesAndSourceConnections[outNode].Add(netCon);
////                        nodesAndOutConnections[inNode].Add(netCon);
                        
////                        netCon.BaseGene = con;
////                        connectionList.Add(netCon);
////                    }
////                }

////                Dictionary<int, List<NetworkNode>> layerNodes = new Dictionary<int, List<NetworkNode>>();
////                //I'm almost certain this requires the sensors to go first, 
////                if (!this.checkRecurrent(layerNodes, nodesAndOutConnections)) {//this.isRecurrent()){
////                    connectionList = topologicallySortedConnections(nodesAndSourceConnections);
////                    initializeByteCodeNetwork();
////                }
////                else
////                    verifiedRecurrent = true;
////        }

        
////        private bool checkRecurrent(Dictionary<int, List<NetworkNode>> layerNodes, Dictionary<NetworkNode, List<NetworkConnection>> nodesAndTargets)
////        {
////            List<NetworkNode> inputs = nodesAndTargets.Keys.Where(x => x.Type == NetworkNodeType.Sensor).ToList();

////            //Dictionary<int, List<NetworkNode>> layerNodes = new Dictionary<int, List<NetworkNode>>();
////            layerNodes.Add(0, new List<NetworkNode>(inputs));


////            List<NetworkConnection> outgoingConnections = inputs.SelectMany(x => nodesAndTargets[x]).Distinct().ToList();

////            List<NetworkConnection> addLater = new List<NetworkConnection>();
////            List<NetworkConnection> investigatedConnections = new List<NetworkConnection>();
////            bool isrec = rCheckRecurrent(1, outgoingConnections, nodesAndTargets, layerNodes, addLater, investigatedConnections);

////            if (isrec)
////                return true;

////            int nextLayer = layerNodes.Max(x => x.Key) + 1;
////            layerNodes.Add(nextLayer, new List<NetworkNode>());

////            foreach (NetworkConnection nc in addLater)
////            {
////                if (!layerNodes[nextLayer].Contains(nc.Target))
////                    layerNodes[nextLayer].Add(nc.Target);
////            }

////            isrec = false;
////            foreach (var layer in layerNodes)
////            {

////                //Console.WriteLine("Layer : " + layer.Key);
////                foreach (var node in layer.Value)
////                {
////                    //Console.WriteLine("Node: " + node.NodeID + " type: " + node.Type);

////                    List<NetworkConnection> targets = nodesAndTargets[node];
////                    var previousLayers = layerNodes.Where(x => x.Key <= layer.Key);
////                    foreach (NetworkConnection nc in targets)
////                    {
////                        if (previousLayers.Any(x => x.Value.Contains(nc.Target)))
////                            return true;

////                    }



////                }

////            }

////            return isrec;
////        }

////        private bool rCheckRecurrent(int currentLayer, List<NetworkConnection> layerConnections,
////            Dictionary<NetworkNode, List<NetworkConnection>> nodesAndTargets, Dictionary<int, List<NetworkNode>> layerNodes,
////            List<NetworkConnection> addConnections, List<NetworkConnection> investigatedConnections)
////        {
////            bool isRecurrent = false;
////            List<NetworkConnection> connectionsWithTargets = layerConnections.Where(x => nodesAndTargets[x.Target].Count > 0).ToList();
////            List<NetworkConnection> addLater = layerConnections.Where(x => nodesAndTargets[x.Target].Count == 0).ToList();
////            addConnections.AddRange(addLater);

////            if (connectionsWithTargets.Count > 0 && !layerNodes.ContainsKey(currentLayer))
////                layerNodes.Add(currentLayer, new List<NetworkNode>());


////            //Console.WriteLine("Investigating connections: ");
////            //connectionsWithTargets.ForEach(x => Console.WriteLine("Con: " + x));
////            //Console.WriteLine("Leaving: ");
////            //addLater.ForEach(x => Console.WriteLine("conl: " + x));


////            //check the previous layer
////            List<NetworkNode> previousNodes = layerNodes[currentLayer - 1];
////            foreach (NetworkConnection nc in connectionsWithTargets)
////            {
////                //if (previousNodes.Contains(nc.Target))
////                //{
////                //    previousNodes.Remove(nc.Target);
////                //}
////                if (!layerNodes[currentLayer].Contains(nc.Target))
////                    layerNodes[currentLayer].Add(nc.Target);
////            }



////            //if any of the connections have targets that themselves have targets
////            if (layerConnections.Any(x => nodesAndTargets[x.Target].Count > 0))
////            {
////                List<NetworkConnection> targetCons = new List<NetworkConnection>();
////                foreach (NetworkConnection netCon in connectionsWithTargets)
////                {
////                    targetCons.AddRange(nodesAndTargets[netCon.Target]);
////                }
////                targetCons = targetCons.Distinct().ToList();

////                //List<NetworkConnection> innerLayerCons = targetCons.Where(x => targetCons.Any(tc => tc.Source == x.Target)).ToList();
////                //if (innerLayerCons.Count > 0)
////                //    targetCons = innerLayerCons;


////                //this gets a little convoluted for the algorithm, but I believe it works
////                int totalCons = targetCons.Count;

////                List<NetworkConnection> innerLayerCons = targetCons.Where(x => targetCons.Any(tc => tc.Source == x.Target)).ToList();
////                targetCons = targetCons.Where(x => !targetCons.Any(tc => tc.Source == x.Target)).ToList();

////                if (targetCons.Count != totalCons)
////                {
////                    //remove any nodes added that were targets in this current layer, and then investiagte them at the next layer
////                    //what this means is that everything that has a target that equals a source, we add to this layer, and look at the rest next layer
////                    //layerNodes.Add(currentLayer + 1, new List<NetworkNode>());



////                    innerLayerCons.ForEach(x =>
////                    {
////                        //first we eject it from the current layer, we'll look at it next time
////                        layerNodes[currentLayer].Remove(x.Target);

////                        //we don't have to do this to process, since it gets processed next round
////                        //layerNodes[currentLayer + 1].Add(x.Target);

////                        //we remove any instance of the node as a source, since it's not actually part of this layer
////                        targetCons.RemoveAll(cons => cons.Source == x.Target);

////                        //but you see we must investigate this inner connection, since it is a target connection of the current node layer
////                        targetCons.Add(x);

////                    });
////                }
////                //  altCheckRecurrent(layerNodes.Max(x => x.Key) + 1, innerLayerCons,
////                //      nodesAndTargets, layerNodes, addLater);

////                bool recCheck = false;
////                targetCons.ForEach(x =>
////                {
////                    if (investigatedConnections.Contains(x))
////                        recCheck = true;
////                    else
////                        investigatedConnections.Add(x);
////                });

////                if (recCheck)
////                    return true;

////                int layer = layerNodes[currentLayer].Count > 0 ? currentLayer + 1 : currentLayer;
////                rCheckRecurrent(layer, targetCons,
////                      nodesAndTargets, layerNodes, addLater, investigatedConnections);

////            }


////            return isRecurrent;
////        }

//    }
//    #endregion

    }
}
