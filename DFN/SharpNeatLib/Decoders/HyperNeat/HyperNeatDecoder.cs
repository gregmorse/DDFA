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
using SharpNeat.Core;
using SharpNeat.Genomes.Neat;
using SharpNeat.Network;
using SharpNeat.Phenomes;
using SharpNeat.Phenomes.NeuralNets;
using System;
using System.Drawing;

namespace SharpNeat.Decoders.HyperNeat
{
    /// <summary>
    /// Decodes CPPN NeatGenome's into concrete network instances.
    /// This decoder uses a HyperNEAT substrate and queries a CPPN NEAT network to
    /// generate/grow a network from the substrate.
    /// </summary>
    public class HyperNeatDecoder : IGenomeDecoder<NeatGenome,IBlackBox>
    {
        readonly Substrate _substrate;
        readonly NetworkActivationScheme _activationSchemeCppn;
        readonly NetworkActivationScheme _activationSchemeSubstrate;
        readonly bool _lengthCppnInput;

        delegate IBlackBox DecodeCppnGenome(NeatGenome genome);
        readonly DecodeCppnGenome _decodeCppnMethod;

        delegate IBlackBox CreateSubstrateNetwork(INetworkDefinition networkDef);
        readonly CreateSubstrateNetwork _createSubstrateNetworkMethod;

        #region Constructors

        /// <summary>
        /// Constructs with the provided substrate, CPPN activation scheme and substrate 
        /// network activation scheme.
        /// </summary>
        public HyperNeatDecoder(Substrate substrate,
                                NetworkActivationScheme activationSchemeCppn,
                                NetworkActivationScheme activationSchemeSubstrate)
        {
            _substrate = substrate;
            _activationSchemeCppn = activationSchemeCppn;
            _activationSchemeSubstrate = activationSchemeSubstrate;
            _decodeCppnMethod = GetDecodeCppnMethod(_activationSchemeCppn);
            _createSubstrateNetworkMethod = GetCreateSubstrateNetworkMethod(activationSchemeSubstrate);
        }

        /// <summary>
        /// Constructs with the provided substrate, CPPN activation scheme and substrate 
        /// network activation scheme.
        /// </summary>
        public HyperNeatDecoder(Substrate substrate,
                                NetworkActivationScheme activationSchemeCppn,
                                NetworkActivationScheme activationSchemeSubstrate,
                                bool lengthCppnInput)
        {
            _substrate = substrate;
            _activationSchemeCppn = activationSchemeCppn;
            _activationSchemeSubstrate = activationSchemeSubstrate;
            _decodeCppnMethod = GetDecodeCppnMethod(_activationSchemeCppn);
            _createSubstrateNetworkMethod = GetCreateSubstrateNetworkMethod(_activationSchemeSubstrate);
            _lengthCppnInput = lengthCppnInput;
        }

        #endregion

        #region IGenomeDecoder Members

        /// <summary>
        /// Decodes a CPPN NeatGenome to a concrete network instance via a HyperNEAT substrate.
        /// </summary>
        public IBlackBox Decode(NeatGenome genome)
        {
            // Decode the CPPN.
            IBlackBox cppn = _decodeCppnMethod(genome);

            // Generate a network definition from the CPPN and Substrate.
            INetworkDefinition substrateNetworkDef = _substrate.CreateNetworkDefinition(cppn, _lengthCppnInput);

            // print this network to an image file
            /*
            Bitmap bmp = new Bitmap(56, 56);
            for(int i = 0; i < 56; i++)
                for(int j = 0; j < 56; j++)
                    bmp.SetPixel(i, j, Color.FromArgb(128, Color.Black));
            for (int i = 0; i < substrateNetworkDef.ConnectionList.Count; i++)
            {
                if (substrateNetworkDef.ConnectionList[i].SourceNodeId > 0)
                {          
                    // color is normalized and a sharpening factor (2000) is applied to be able to distinguish fine differences more easily
                    int alpha;
                    alpha = Math.Max(Math.Min((int)Math.Floor((substrateNetworkDef.ConnectionList[i].Weight) * 2000) + 128, 255), 0);
                    Color c = Color.FromArgb(alpha, alpha, alpha);

                    // for each weight in the network, generate 4 pixels to enlarge the image
                    bmp.SetPixel(((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) / 28 * 2, ((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) % 28 * 2, c);
                    bmp.SetPixel(((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) / 28 * 2 + 1, ((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) % 28 * 2, c);
                    bmp.SetPixel(((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) / 28 * 2, ((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) % 28 * 2 + 1, c);
                    bmp.SetPixel(((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) / 28 * 2 + 1, ((int)substrateNetworkDef.ConnectionList[i].SourceNodeId - 1) % 28 * 2 + 1, c);
                }
            }
            bool fileExists = true;
            string fileName = "";
            int counter = 0;
            while (fileExists)
            {
                fileName = Environment.CurrentDirectory + "\\featureSamples\\featureSample" + counter++ + ".jpg";
                if (!System.IO.File.Exists(fileName))
                    fileExists = false;
            }
            bmp.Save(fileName, System.Drawing.Imaging.ImageFormat.Jpeg);
            */

            // Check for null network. This can happen if, e.g. querying the substrate did not result in any connections.
            if(null == substrateNetworkDef) {
                return null;
            }

            // Create a network from the substrate network definition, and return it.
            return _createSubstrateNetworkMethod(substrateNetworkDef);
        }

        #endregion

        #region Private Methods [CPPN Decoding]

        /// <summary>
        /// Method that determines which CPPN decode routine to use.
        /// </summary>
        private DecodeCppnGenome GetDecodeCppnMethod(NetworkActivationScheme activationScheme)
        {
            if (activationScheme.OpenAcyclicNetwork)
            {
                if (activationScheme.AcyclicNetwork == false)
                    throw new Exception("Activation scheme for perceptrons must also be acyclic!");

                return DecodeToOpenAcyclicNetwork;
            }
            if(activationScheme.AcyclicNetwork)
            {
                return DecodeToFastAcyclicNetwork;
            }

            if(activationScheme.FastFlag)
            {
                return DecodeToFastCyclicNetwork;
            }
            return DecodeToCyclicNetwork;
        }

        private OpenAcyclicNetwork DecodeToOpenAcyclicNetwork(NeatGenome genome)
        {
            //same call for acyclic, but with a mulit
            return MultiPerceptronNetworkFactory.CreateAcyclicOpenNetwork(genome);
        }

        private AcyclicByteNetwork DecodeToFastAcyclicNetwork(NeatGenome genome)
        {
            return SingleOutputNetworkFactory.CreateFastAcyclicNetwork(genome);
        }

        private CyclicNetwork DecodeToCyclicNetwork(NeatGenome genome)
        {
            return CyclicNetworkFactory.CreateCyclicNetwork(genome, _activationSchemeCppn);
        }

        private FastCyclicNetwork DecodeToFastCyclicNetwork(NeatGenome genome)
        {
            return FastCyclicNetworkFactory.CreateFastCyclicNetwork(genome, _activationSchemeCppn);
        }

        #endregion

        #region Private Methods [Substrate Network Creation]

        /// <summary>
        /// Method that determines which substrate instance creation routine to use.
        /// </summary>
        private CreateSubstrateNetwork GetCreateSubstrateNetworkMethod(NetworkActivationScheme activationScheme)
        {
            //we create substrates that activate quite simply! Just dot product inputs X weights
            if (activationScheme.SingleOutputNetwork)
            {
                return CreateSubstrateNetwork_SingleOutputNetwork;
            }

            if (activationScheme.SingleOuputHiddenNetwork)
            {
                return CreateSubstrateNetwork_SingleOutputNetworkWithHidden;
            }

            if (activationScheme.OpenAcyclicNetwork)
            {
                if (activationScheme.AcyclicNetwork == false)
                    throw new Exception("Activation scheme for perceptrons must also be acyclic!");

                return CreateSubstrateNetwork_OpenAcyclicNetwork;
            }

            if(activationScheme.AcyclicNetwork)
            {
                return CreateSubstrateNetwork_AcyclicNetwork;
            }

            if(activationScheme.FastFlag)
            {
                return CreateSubstrateNetwork_FastCyclicNetwork;
            }
            return CreateSubstrateNetwork_CyclicNetwork;
        }
        private SingleOutputNetwork CreateSubstrateNetwork_SingleOutputNetwork(INetworkDefinition networkDef)
        {
            //same call for acyclic, but with a mulit
            return SingleOutputNetworkFactory.CreateSingleOutputNetwork(networkDef);
        }
        private SingleOutputNetworkWithHidden CreateSubstrateNetwork_SingleOutputNetworkWithHidden(INetworkDefinition networkDef)
        {
            //same call for acyclic, but with a mulit
            return SingleOutputNetworkFactory.CreateSingleOutputNetworkWithHidden(networkDef);
        }
        private OpenAcyclicNetwork CreateSubstrateNetwork_OpenAcyclicNetwork(INetworkDefinition networkDef)
        {
            //same call for acyclic, but with a mulit
            return MultiPerceptronNetworkFactory.CreateAcyclicOpenNetwork(networkDef);
        }
        private AcyclicByteNetwork CreateSubstrateNetwork_AcyclicNetwork(INetworkDefinition networkDef)
        {
            return SingleOutputNetworkFactory.CreateFastAcyclicNetwork(networkDef);
        }

        private CyclicNetwork CreateSubstrateNetwork_CyclicNetwork(INetworkDefinition networkDef)
        {
            return CyclicNetworkFactory.CreateCyclicNetwork(networkDef, _activationSchemeCppn);
        }

        private FastCyclicNetwork CreateSubstrateNetwork_FastCyclicNetwork(INetworkDefinition networkDef)
        {
            return FastCyclicNetworkFactory.CreateFastCyclicNetwork(networkDef, _activationSchemeSubstrate);
        }

        #endregion
    }
}
