using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using SharpNeat.Phenomes.NeuralNets;
using System.Diagnostics;
using System.Threading.Tasks;
using SharpNeat.Phenomes;

namespace SharpNeat.Network
{
    /// <summary>
    /// A single layer perceptron that's quite slim and runs on arrays as opposed to connection objects like other networks.
    /// </summary>
    public class SingleOutputNetwork : IBlackBox 
    {
        //only need two things to define us here!
        IActivationFunction activation;
        double[] weights;
        bool[] backPropAllowed;

        /// <summary>
        /// Send in the connections you would normally create, and then also the intended feature count for verification of the proper connections
        /// </summary>
        /// <param name="inputFeatureCount"></param>
        /// <param name="connections"></param>
        public SingleOutputNetwork(double[] weights, IActivationFunction actFun, bool[] backPropWeights)
        {
            this.weights = weights;
            this.activation = actFun;
            this.backPropAllowed = backPropWeights;

            Debug.Assert(actFun != null, "Single Output must have non-null activation!");
        }
        
        double lastActivation;
        public double LastActivation
        {
            get { return lastActivation; }
        }
        public double[] Weights
        {
            get { return weights; }
            set { weights = value; }
        }
        public bool[] AllowBackpropWeights
        {
            get { return backPropAllowed; }
        }
        public IActivationFunction ActFun
        {
            get { return activation; }
        }
        public void Activate()
        {
            throw new NotImplementedException("Doesn't implement activate. Use the proper Dot Product Activation with inputs");
        }

        double[] propDeltas;
        public void PropogateError(double[] inputs, double propError, double learningRate, double momentum)
        {
            if (propDeltas == null)
                propDeltas = new double[weights.Length];
            bool limitedBackprop = (backPropAllowed != null);
            //now propogate based on feature inputs
            for (int w = 0; w < weights.Length; w++)
            {
                //skip error propogation on weights that didn't have connections
                if (limitedBackprop && !backPropAllowed[w])
                    continue;

                //original input -- if it's bias then 1.0
                //otherwise w-1
                double inputToWeight = (w==0 ? 1.0 : inputs[w-1]);

                //original weight, will modify based on error
                double oWeight = weights[w];

                //calculate the error caused by this weight for this output node and feature combination (later to be summed)
                double delta = (learningRate * propError * inputToWeight);

                oWeight += delta;
                oWeight += momentum * propDeltas[w];
                propDeltas[w] = delta;
                //weight change = 
                //w_new = w_old + learningRate x error x inActivation
                weights[w] = oWeight;
            }

            //all weights changed based on error rate
        }

        public double[] getErrorDeltas(double[] inputs, double propError, double learningRate)
        {
            double[] deltas = new double[weights.Length];

            if (propDeltas == null)
                propDeltas = new double[weights.Length];

            //now propogate based on feature inputs
            for (int w = 0; w < weights.Length; w++)
            {
                //original input -- if it's bias then 1.0
                //otherwise w-1
                double inputToWeight = (w == 0 ? 1.0 : inputs[w - 1]);

                //original weight, will modify based on error
                double oWeight = weights[w];

                //calculate the error caused by this weight for this output node and feature combination (later to be summed)
                double delta = (learningRate * propError * inputToWeight);

                deltas[w] = delta;

                //oWeight += delta;
                //oWeight += momentum * propDeltas[w];
                //propDeltas[w] = delta;
                //weight change = 
                //w_new = w_old + learningRate x error x inActivation
                //weights[w] = oWeight;
            }

            return deltas;
        }

        public void updateFeatureFromDeltas(double[] deltas)
        {
            for (int i = 0; i < deltas.Length; i++)
                weights[i] += deltas[i];
        }

        //give the inputs, and what the correct guess should be
        public double ActivateDotProduct(double[] inputs, bool biasIsZero = false)
        {
            double sum = 0;

            if (!biasIsZero)
            {
                //handle bias weight
                //always 1 bias please
                sum += weights[0] * 1.0;
            }

            //offset by one because of the bias node 
            for (int i = 0; i < inputs.Length; i++)
                sum += inputs[i] * weights[i+1];

            //calculate the activation from the sum, please
            lastActivation = activation.Calculate(sum, null);

            return lastActivation;
        }

        #region Satisfy blackbox [barely]

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
        /// Reset any internal state.
        /// </summary>
        public void ResetState() { }

        #endregion
    }

    /// <summary>
    /// A single layer perceptron that's quite slim and runs on arrays as opposed to connection objects like other networks.
    /// </summary>
    public class SingleOutputNetworkWithHidden : IBlackBox
    {
        //only need two things to define us here!
        IActivationFunction activation;

        int hiddenNodeCount;
        double[] inToOutWeights;
        double[][] hiddenWeights;
        double[] hiddenToOutWeights;

        double[] intermediateActivation;

        /// <summary>
        /// Send in the connections you would normally create, and then also the intended feature count for verification of the proper connections
        /// </summary>
        /// <param name="inputFeatureCount"></param>
        /// <param name="connections"></param>
        public SingleOutputNetworkWithHidden(
            double[] in2Out, 
            double[][] in2Hidden, 
            double[] hidden2Out, 
            IActivationFunction actFun)
        {
            this.inToOutWeights = in2Out;
            this.hiddenWeights = in2Hidden;
            this.hiddenToOutWeights = hidden2Out;

            this.hiddenNodeCount = in2Hidden.Length;
            this.intermediateActivation = new double[hiddenNodeCount];
            this.activation = actFun;

            Debug.Assert(actFun != null, "Single Output must have non-null activation!");
        }

        double lastActivation;
        public double LastActivation
        {
            get { return lastActivation; }
        }
        public double[] InputsToOutputWeights
        {
            get { return inToOutWeights; }
        }
        public double[][] InputsToHiddenWeights
        {
            get { return hiddenWeights; }
        }
        public double[] HiddenToOutputWeights
        {
            get { return hiddenToOutWeights; }
        }

        public void Activate()
        {
            throw new NotImplementedException("Doesn't implement activate. Use the proper Dot Product Activation with inputs");
        }

        //give the inputs, and what the correct guess should be
        public double ActivateDotProduct(double[] inputs)
        {

            for (int i = 0; i < hiddenNodeCount; i++)
            {
                double[] weights = hiddenWeights[i];

                double sum = 0;

                //handle bias weight
                //always 1 bias please
                sum += weights[0];

                 //offset by one because of the bias node 
                for (int c = 0; c < inputs.Length; c++)
                    sum += inputs[c] * weights[c + 1];

                //set that activation inside the intermediate holder
                intermediateActivation[i] = activation.Calculate(sum, null);
            }

             double finalSum = 0;

            //handle bias for inputs to final ouput weights
             finalSum += inToOutWeights[0];

            //now handle the connections from our hidden nodes to the output node
             for (int i = 0; i < hiddenNodeCount; i++)
             {
                 //multiply and add this hidden layer activation * weight for that node
                 finalSum += intermediateActivation[i] * hiddenToOutWeights[i];
             }

            //finally, loop through all the input to output connection objects doyyyy
             for (int i = 0; i < inputs.Length; i++)
             {
                 finalSum += inputs[i] * inToOutWeights[i + 1];
             }

            //calculate the activation from the sum, please
             lastActivation = activation.Calculate(finalSum, null);

            return lastActivation;
        }

        #region Satisfy blackbox [barely]

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
        /// Reset any internal state.
        /// </summary>
        public void ResetState() { }

        #endregion


    }


}
