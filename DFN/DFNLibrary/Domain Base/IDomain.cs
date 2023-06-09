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
    public interface IDomain
    {
        void LoadImages(string jsonFolder = @"");        

        int getPixelWidth();
        int getPixelHeight();
        int getTotalInputs();

        IGenomeDecoder<NeatGenome, IBlackBox> CreateGenomeDecoder(bool normalizeWeight, int pixelWidth, int pixelHeight,
            NetworkActivationScheme cppnSchem, NetworkActivationScheme substrateScheme, bool biasUsesLEO, bool useLEO);

        FitnessInfo EvaluateImages(
            double[][] testImages,
            int pixelCount,
            IBlackBox box,
            Dictionary<int, Tuple<double, double>> minMaxDictionary,
            bool writeToDictionary,
            int boxIx
            );

        IDomainEvaluator createEvaluator();

        List<int> LeoIndexes(bool biasUsesLEO, bool useLEO);

        int getCPPNInputs();

        int getCPPNOutputs(bool biasUsesLEO, bool useLEO);

        List<int> getInterestedDigits();

        double[][] getAllTestImages();

        double[][] getAllValidationImages();

        int[] getAllTestLabels();

        int getTotalTestImageCount();

        List<int> getDigitsForString();

        int[] getAllValidationLabels();
    }
}
