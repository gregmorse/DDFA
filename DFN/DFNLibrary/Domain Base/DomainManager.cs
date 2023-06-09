using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Core;
using SharpNeat.Genomes.Neat;
using SharpNeat.Phenomes;
using SharpNeat.Decoders;

namespace DFN.Evaluation
{
    public class DomainManager
    {              
        public static IDomain Instance
        {
            get;
            set;
        }

        public static void LoadImages(string jsonFolder = @"")
        {
            Instance.LoadImages(jsonFolder);
        }

        public static IGenomeDecoder<NeatGenome, IBlackBox> CreateGenomeDecoder(bool normalizeWeight, int pixelWidth, int pixelHeight,
            NetworkActivationScheme cppnSchem, NetworkActivationScheme substrateScheme, bool biasUsesLEO, bool useLEO)
        {
            return Instance.CreateGenomeDecoder(normalizeWeight, pixelWidth, pixelHeight, cppnSchem, substrateScheme, biasUsesLEO, useLEO);
        }

        public static FitnessInfo EvaluateImages(
            double[][] testImages,
            int pixelCount,
            IBlackBox box,
            Dictionary<int, Tuple<double, double>> minMaxDictionary,
            bool writeToDictionary,
            int boxIx
            )
        {
            return Instance.EvaluateImages(testImages, pixelCount, box, minMaxDictionary, writeToDictionary, boxIx);
        }

        public static int getCPPNInputs()
        {
            return Instance.getCPPNInputs();
        }

        public static int getCPPNOutputs(bool biasUsesLEO, bool useLEO)
        {
            return Instance.getCPPNOutputs(biasUsesLEO, useLEO);
        }

        public static List<int> getInterestedDigits()
        {
            return Instance.getInterestedDigits();
        }
    }
}
