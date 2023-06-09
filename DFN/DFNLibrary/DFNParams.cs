using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Novelty;
using SharpNeat.Decoders;

namespace DFNLibrary
{
    public enum DFNCollectorType
    {
        Threshold
    }

    public enum Domain
    {
        MNIST,
        CIFAR // color CIFAR (HSV channels)
    }

    public class DFNParams
    {
        public static Domain DOMAIN = Domain.MNIST;
        public static int GENERATIONSPERSAVE = 9;
        public static float PROBADDITIONTHRESHOLD = 0.015f; // probability of adding genome to novelty archive (used to measure novelty and guide evolution)
        public static float FEATUREADDITIONTHRESHOLD = 2000f; // threshold for adding genome to feature collector (final features for use in aggregate network)
        public static int THREADS = 6;

        public static int TRAININGOUTPUTFREQ = 1;
        public static bool RANDOMIZESTARTINGWEIGHTS = false; //do we randomize the starting input weights? Keep them small but random
        public static bool USESLIMPERCEPTRON = true;

        // backprop/perceptron params
        public static int PERCEPTRONITERATIONS = 10000; // how many training passes for the perceptron?
        public static float INITIALLEARNRATEPERCEPTRON = 1.0f;
        public static float INITIALLEARNRATEBACKPROP = 0.01f; // standard = .01
        public static float LEARNRATEDECAY = 0.001f; // 0.01 = 1% decay per pass - standard = .001
        public static float MOMENTUM = .1f; //the momentum inside backprop -- decrease training times - standard = .1
        public static int IMAGEBATCHES = 10;
        public static bool PLAIN_SIGMOID_ACTIVATION = false; //what activation to use for the features        
        public static int PERCTRAINFREQ = 10; // how many generations between perceptron training events?
        public static bool USELEO = true;
        public static bool USEBIASLEO = true;

        public static float TRAININGSTOPACCURACY = 1f; // when to stop training for backprop, set to 1 to train until it is "perfect"        
        public static float PERCEPTRONTRAININGSTOPACCURACY = 1f; // when to stop training for perceptron
        public static bool RECORDFEATURES = false; // set to true to write the features to image files before and after backprop has had its way with them
        public static bool SAVEWRONGGUESSES = false; // set to true to save the frequency of misclassified digits at the end of backprop training, as well as images of the misclassified images
        public static bool SAVEPOPULATION = false; // save population at each save point?
        public static int NETWORKSAVEFREQUENCY = 200; // how often should we save the neural network to a file during backprop? set to int.maxvalue to effectively disable this

        //This is the width in number of nodes of the hidden node substrate, 0 will deactivate hidden nodes in the features
        public static int HiddenNodeSubstrateWidth = 0;

        public static bool CREATE_MNIST_VALIDATION = false; // hold back 10k images from the backprop run to be used for testing (i.e. validation set)
        public static bool GENERATE_RANDOM_CPPNS = false; // used to generate random CPPNs by setting the feature collector to a probabilistic archive and the novelty calculation to a random score
        public static double RANDOM_CPPN_ADDITION_FREQUENCY = 5; // if generating random CPPNs, how many random CPPNs to add to the archive each generation?
        public static bool TEST_RANDOM_BP_FEATURES = false; // use completely random features at the start of backprop             
        public static bool USE_SPARSE_INITIALIZATION = false; // if using random backprop features, initialize them using sparse initialization, as described in "Deep learning via Hessian-free optimization" (Martens)

        public static string DATA_TO_TRAIN_FOLDER = "archive";
    }      
}
