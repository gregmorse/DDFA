using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Xml;
using SharpNeat.Network;

namespace DFNLibrary
{
    public class NetworkSaver
    {
        public static void saveBPPNetwork(BackPropPerceptron bpp, string path)
        {
            int featureCounter = 0;

            XmlWriterSettings settings = new XmlWriterSettings();
            settings.Indent = true;

            XmlWriter writer = XmlWriter.Create(path, settings);
            writer.WriteStartDocument();
            writer.WriteStartElement("Network");

            writer.WriteAttributeString("OutputCount", bpp._weights.Length.ToString());
            writer.WriteAttributeString("FeatureCount", bpp.allFeatures.Length.ToString());
            writer.WriteAttributeString("WeightsPerFeature", bpp.allFeatures[0].Weights.Length.ToString());


            writer.WriteStartElement("TopLayer");
            foreach (double[] weights in bpp._weights)
            {
                writer.WriteStartElement("Output-" + featureCounter++);
                int weightCounter = 0;
                foreach (double weight in weights)
                {
                    writer.WriteAttributeString("F-" + weightCounter++.ToString(), weight.ToString());
                }
                writer.WriteEndElement();
            }
            writer.WriteEndElement();

            featureCounter = 0;
            writer.WriteStartElement("FeatureLayer");            
            foreach (SingleOutputNetwork son in bpp.allFeatures)
            {
                writer.WriteStartElement("Feature-" + featureCounter++);
                int weightCounter = 0;
                bool[] backAllowed = son.AllowBackpropWeights;
                writer.WriteAttributeString("BPLimited", (backAllowed != null).ToString());

                foreach (double weight in son.Weights)
                {
                    // only write non-zero weights
                    if(weight != 0)
                        writer.WriteAttributeString("W-" + weightCounter.ToString(), weight.ToString());

                    //are we doing back allowed?
                    if(backAllowed != null && backAllowed[weightCounter])
                        writer.WriteAttributeString("WA-" + weightCounter.ToString(), backAllowed[weightCounter].ToString());

                    //we increment weight ix regardless of if it was zero or not
                    //when we read back in, the weights have to be from the correct location referenced by weightcounter
                    weightCounter++;
                }
                writer.WriteEndElement();
            }
            writer.WriteEndElement();

            writer.WriteEndElement();
            writer.WriteEndDocument();
            writer.Flush();
            writer.Close();
        }

        
        public static BackPropPerceptron loadBPPNetwork(string path)
        {
            BackPropPerceptron bpp;
            int outputCount = 0, featureCount = 0, weightsPerFeature = 0;
            double[][] startingWeights;
            SingleOutputNetwork[] features;

            XmlReader reader = XmlReader.Create(path);
            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.Element && reader.Name == "Network")
                {
                    outputCount = Int32.Parse(reader.GetAttribute("OutputCount"));
                    featureCount = Int32.Parse(reader.GetAttribute("FeatureCount"));
                    weightsPerFeature = Int32.Parse(reader.GetAttribute("WeightsPerFeature"));

                    break;
                }
            }

            startingWeights = new double[outputCount][];
            for (int i = 0; i < outputCount; i++)
            {
                startingWeights[i] = new double[featureCount + 1];
            }

            features = new SingleOutputNetwork[featureCount];

            int outputCounter = 0, featureCounter = 0;

            while (reader.Read())
            {
                if (reader.NodeType == XmlNodeType.Element && reader.Name.Substring(0, 6) == "Output")
                {
                    for (int i = 0; i < reader.AttributeCount; i++)
                    {
                        startingWeights[outputCounter][i] = double.Parse(reader.GetAttribute(i));
                    }
                    outputCounter++;
                }
                if (reader.NodeType == XmlNodeType.Element && reader.Name.Substring(0, 8) == "Feature-")
                {
                    double[] weights = new double[weightsPerFeature];
                    double weight;
                    bool bpLimited;
                    bool[] backAllowed = null;

                    //if we're only limited to the weights that the CPPN created for backprop
                    if(bool.TryParse(reader.GetAttribute("BPLimited"), out bpLimited))
                    {
                        if (bpLimited)
                            backAllowed = new bool[weightsPerFeature];
                    }

                    for (int i = 0; i < weightsPerFeature; i++)
                    {
                        if (double.TryParse(reader.GetAttribute("W-" + i), out weight))
                            weights[i] = weight;     
                        if(bpLimited)
                        {
                            bool shouldProp;
                            if (bool.TryParse(reader.GetAttribute("WA-" + i), out shouldProp))
                                backAllowed[i] = shouldProp;
                        }
                    }
                    features[featureCounter++] = new SingleOutputNetwork(weights, SteepenedSigmoid.__DefaultInstance, backAllowed);                    
                }
            }

            bpp = new BackPropPerceptron(features, outputCount, startingWeights);
            return bpp;
        }
        
    }
}
