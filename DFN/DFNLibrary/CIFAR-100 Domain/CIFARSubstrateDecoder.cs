using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Core;
using SharpNeat.Genomes.Neat;
using SharpNeat.Phenomes;
using SharpNeat.Decoders.HyperNeat;
using SharpNeat.Network;
using SharpNeat.Decoders;
using DFNLibrary;

namespace DFN.Evaluation
{
    public class CIFARSubstrateDecoder
    {
        private static void AddNode(SubstrateNodeSet nodeSet, uint id, double x, double y)
        {
            nodeSet.NodeList.Add(new SubstrateNode(id, new double[] { x, y }));
        }

        public static List<int> LeoIndexes(bool biasUsesLEO, bool useLEO)
        {
            List<NodeSetMapping> maps = CIFARSubstrateLayerMapping(biasUsesLEO, useLEO);

            List<int> leoIxs = new List<int>();

            //if we have a bias LEO, then the bias LEO is at index 1
            if (biasUsesLEO)
                leoIxs.Add(1);

            //this is the location of all the LEO ixs for the experiment substrate
            //for (int i = 0; i < maps.Count; i++)
            //    leoIxs.Add(maps[i].OutputNeuronIx + 1);
            if (DFNParams.HiddenNodeSubstrateWidth > 0)
            {
                throw new NotImplementedException();
            }
            else
                leoIxs.Add(5); // hardcoded for now, need to change this if we add hidden layer(s)
        



            return leoIxs;
        }

        //square mapping --
        //we mpa from inputs to hidden using a cppn output
        //then we map from hidden to outputs using an entirely separate map!
        public static List<NodeSetMapping> CIFARSubstrateLayerMapping(bool biasUsesLEO, bool useLEO)
        {
            //we setup layer connections. Everything wires directly
            List<NodeSetMapping> nodeSetMappingList = new List<NodeSetMapping>(1);

            //just pull from the 0th node (after bias and bias LEO)
            var cppnOutput = 0;

            //0 is the ix of the cppn to read from, the second 0 is the id of the source set in the nodeSetList array above, and the 2 is the tgt
            nodeSetMappingList.Add(NodeSetMapping.Create(cppnOutput, 0, 3, (double?)null, biasUsesLEO));
            nodeSetMappingList.Add(NodeSetMapping.Create(1, 1, 3, (double?)null, biasUsesLEO));
            nodeSetMappingList.Add(NodeSetMapping.Create(2, 2, 3, (double?)null, biasUsesLEO));

            //now we need to add connections from input -> hidden, hidden -> output
            if (DFNParams.HiddenNodeSubstrateWidth > 0)
            {
                throw new NotImplementedException();
            }


            return nodeSetMappingList;

        }

        public static IGenomeDecoder<NeatGenome, IBlackBox> CreateGenomeDecoder(bool normalizeWeight, int pixelWidth, int pixelHeight,
            NetworkActivationScheme cppnSchem, NetworkActivationScheme substrateScheme, bool biasUsesLEO, bool useLEO)
        {
            // Create HyperNEAT network substrate.

            //-- Create input layer nodes.
            SubstrateNodeSet inputLayer = new SubstrateNodeSet(13);

            double widthDx = 2.0 / (pixelWidth - 1);
            double heightDx = 2.0 / (pixelHeight - 1);

            double x = -1;
            double y = -1;

            uint ix = 1;

            for (var h = 0; h < pixelHeight; h++)
            {
                //we go by rows 
                //so for each height pixel, we sweep across all X
                for (var w = 0; w < pixelWidth; w++)
                {
                    AddNode(inputLayer, ix++, x, y);
                    x += widthDx;
                }

                //add chunk of width, do it agains
                x = -1;
                y += heightDx;
            }            

            // input layer 2
            SubstrateNodeSet inputLayer2 = new SubstrateNodeSet(13);         

            x = -1;
            y = -1;

            for (var h = 0; h < pixelHeight; h++)
            {
                //we go by rows 
                //so for each height pixel, we sweep across all X
                for (var w = 0; w < pixelWidth; w++)
                {
                    AddNode(inputLayer2, ix++, x, y);
                    x += widthDx;
                }

                //add chunk of width, do it agains
                x = -1;
                y += heightDx;
            }

            SubstrateNodeSet inputLayer3 = new SubstrateNodeSet(13);

            x = -1;
            y = -1;

            for (var h = 0; h < pixelHeight; h++)
            {
                //we go by rows 
                //so for each height pixel, we sweep across all X
                for (var w = 0; w < pixelWidth; w++)
                {
                    AddNode(inputLayer3, ix++, x, y);
                    x += widthDx;
                }

                //add chunk of width, do it agains
                x = -1;
                y += heightDx;
            }

            //-- Output layer nodes. Just 2 for this experiment. (x coordinate guess of input pixel, y coordinate guess of input pixel)
            //ideally, these would not be on the same plane -- we'd use Justin's method
            SubstrateNodeSet outputLayer = new SubstrateNodeSet(2);
            AddNode(outputLayer, ix++, 0.0, 0.0); // Output the x --  left of the layer

            //do we have a single output node? Yup! Then we don't need the (x2, y2) part of the cppn, just the (x1, y1) -- i.e. half the inputs
            bool singleOut = (outputLayer.NodeList.Count == 1 && DFNParams.HiddenNodeSubstrateWidth == 0);

            //we can't use single output if we have a hidden layer -- we need 4 dimensions (x1, y1) -> (x2, y2) in the output

            // Connect up layers.
            List<SubstrateNodeSet> nodeSetList = new List<SubstrateNodeSet>(4);
            nodeSetList.Add(inputLayer);            
            nodeSetList.Add(inputLayer2);
            nodeSetList.Add(inputLayer3);
            nodeSetList.Add(outputLayer);
            

            if (DFNParams.HiddenNodeSubstrateWidth > 0)
            {
                throw new NotImplementedException();
            }


            //we setup layer connections. Everything wires directly --- it's simple
            List<NodeSetMapping> nodeSetMappingList = CIFARSubstrateLayerMapping(biasUsesLEO, useLEO);

            //Paul: need plain sigmoid for our backprop code -- also, why use steepened sigmoid for our feature activation? Sucha  steep slope...
            IActivationFunction actFun = DFNParams.PLAIN_SIGMOID_ACTIVATION ? PlainSigmoid.__DefaultInstance : SteepenedSigmoid.__DefaultInstance;

            // Construct substrate.
            Substrate substrate = new Substrate(nodeSetList, DefaultActivationFunctionLibrary.CreateLibraryNeat(actFun),
                0, 0.2, 3, nodeSetMappingList, normalizeWeight, singleOut, useLEO, 5, 3072, 3);

            // Create genome decoder. Decodes to a neural network packaged with an activation scheme that defines a fixed number of activations per evaluation.
            IGenomeDecoder<NeatGenome, IBlackBox> genomeDecoder = new HyperNeatDecoder(substrate, cppnSchem, substrateScheme);
            return genomeDecoder;
        }
    }
}
