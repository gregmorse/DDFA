using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;
using System.IO;
using Newtonsoft.Json.Linq;
using Newtonsoft.Json;

namespace DataLibrary
{
    public class MNISTDataLoader
    {
        public static SortedList<int, List<double[]>> loadFullPicturesInPieces(int pieceCount, string modifier, string jsonFolder = @"")
        {
             SortedList<int, List<double[]>> picturesFromJSON = new SortedList<int, List<double[]>>();

            //now let's load it uppppp
             if (jsonFolder == "")
            {
                //no path supplied, we shall build our own to the first chunk
                string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                jsonFolder = currentAssemblyDirectoryName + "/../../../../data/MNIST/";
            }


            //pull everything in pieces
             for (int i = 0; i < pieceCount; i++)
             {
                 string fileName = jsonFolder + modifier + "_" + i + ".json";

                 SortedList<int, List<double[]>> dictionaryPiece = loadPicturesAsDoubles(fileName);
                 
                 foreach(var kvp in dictionaryPiece)
                 {
                     //if we've never seen this digit before, store everything!
                     if (!picturesFromJSON.ContainsKey(kvp.Key))
                         picturesFromJSON.Add(kvp.Key, kvp.Value);
                     else
                     {
                         //otehrwise, we merge the two
                         picturesFromJSON[kvp.Key].AddRange(kvp.Value);
                     }
                 }
             }

            //all merged up now!
             return picturesFromJSON;
        }

        //This does the main legwork of pulling in files
        //need to pull in data from json files at a specified path
        public static SortedList<int, List<double[]>> loadPicturesAsDoubles(string jsonPath = @"")
        {
            SortedList<int, List<double[]>> picturesFromJSON = new SortedList<int, List<double[]>>();

            //now let's load it uppppp
            if (jsonPath == "")
            {
                //no path supplied, we shall build our own to the first chunk
                string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
                jsonPath = currentAssemblyDirectoryName + "/../../../../data/MNIST/train.json";
            }            
            
            // alternate way of loading data that uses less memory
            using (var reader = new JsonTextReader(File.OpenText(jsonPath)))
            {
                int currentDigit = 0;
                List<double> pixels = new List<double>();
                while (reader.Read())
                {
                    if (reader.TokenType == JsonToken.PropertyName) // we have a new digit to work with
                    {
                        currentDigit = Int32.Parse(reader.Value.ToString());
                        picturesFromJSON[currentDigit] = new List<double[]>();
                    }
                    else if (reader.TokenType == JsonToken.Integer) // add this to the pixel list, normalizing to [0,1]
                    {
                        pixels.Add(Double.Parse(reader.Value.ToString()) / 255);
                    }
                    else if (reader.TokenType == JsonToken.EndArray) // if pixel list is not blank, copy it to an array and add it to the picture list
                    {
                        if (pixels.Count > 0)
                        {
                            picturesFromJSON[currentDigit].Add(pixels.ToArray());
                            pixels.Clear();
                        }
                    }
                }
            }
            
            //all built up! Just need to send back the pixel info
            return picturesFromJSON;
        }


        public static SortedList<int, List<float[]>> loadPictures(string jsonPath = @"")
        {
            SortedList<int, List<double[]>> picturesFromJSON = loadPicturesAsDoubles(jsonPath);
            SortedList<int, List<float[]>> floatConversions = new SortedList<int, List<float[]>>();

            foreach (var kvp in picturesFromJSON)
            {
                floatConversions[kvp.Key] = new List<float[]>();
                foreach (var floatArray in kvp.Value)
                {
                    float[] fpixels = new float[floatArray.Length];
                    for (var i = 0; i < floatArray.Length; i++)
                    {
                        //return to 0, 255
                        fpixels[i] = (float)floatArray[i];
                    }
                    //now add the new byte converted array to our list
                    floatConversions[kvp.Key].Add(fpixels);
                }
            }
            return floatConversions;
        }

        public static byte[] ConvertDoubleToByteImage(double[] image)
        {
            byte[] bPixels = new byte[3 * image.Length];

            for (int i = 0; i < image.Length; i++)
            {
                byte bPixel = (byte)Math.Round(image[i] * 255.0f);
                
                //return to 0, 255
                bPixels[3 * i] = bPixel;
                bPixels[3 * i + 1] = bPixel;
                bPixels[3 * i + 2] = bPixel;
            }

            return bPixels;
        }

        //if we want the returned images as bytes (for bitmap display)
        //this is our helper -- just calls loadImages that returns a float collection and converts to bytes
        //more expensive -- but just for testing purposes
        public static SortedList<int, List<byte[]>> loadPicturesAsBytes(string jsonPath = @"")
        {
            SortedList<int, List<float[]>> picturesFromJSON = loadPictures(jsonPath);
            SortedList<int, List<byte[]>> byteConversions = new SortedList<int, List<byte[]>>();

            foreach (var kvp in picturesFromJSON)
            {
                byteConversions[kvp.Key] = new List<byte[]>();
                foreach (var floatArray in kvp.Value)
                {
                    byte[] bpixels = new byte[3*floatArray.Length];
                    for (var i = 0; i < floatArray.Length; i++)
                    {
                        byte bPixel = (byte)Math.Round(floatArray[i] * 255.0f);
                        //return to 0, 255
                        bpixels[3 * i] = bPixel;
                        bpixels[3*i+1] = bPixel;
                        bpixels[3*i+2] = bPixel;
                    }
                    //now add the new byte converted array to our list
                    byteConversions[kvp.Key].Add(bpixels);
                }
            }
            return byteConversions;
        }
    }
}
