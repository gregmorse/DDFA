using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.IO;

namespace DataLibrary
{
    public class CIFARDataLoader
    {
        public static SortedList<int, List<double[]>> loadPicturesAsDoubles(string path)
        {
            SortedList<int, List<double[]>> pictures = new SortedList<int, List<double[]>>();
            
            // alternate way of loading data that uses less memory
            using (var reader = new BinaryReader(File.Open(path, FileMode.Open)))
            {
                int currentDigit = 0;
                List<double> pixels = new List<double>();
                
                for(int i = 0; i < 100; i++)
                    pictures[i] = new List<double[]>();

                while (reader.PeekChar() != -1)
                {
                    currentDigit = reader.ReadByte(); // coarse label, read right past it
                    currentDigit = reader.ReadByte();                    
                    
                    for(int i = 0; i < 3072; i++)
                    {
                        pixels.Add((double)reader.ReadByte() / 255);
                    }
                    
                    double[] hsbPixels = pixelsToHSB(pixels.ToArray());

                    pictures[currentDigit].Add(hsbPixels);
                    pixels.Clear();                    
                }
            }

            //all built up! Just need to send back the pixel info
            return pictures;
        }

        private static double[] pixelsToHSB(double[] rgbPixels)
        {
            double[] hsbPixels = new double[rgbPixels.Length];

            for (int i = 0; i < rgbPixels.Length / 3; i++)
            {
                double r = rgbPixels[i];
                double g = rgbPixels[i + rgbPixels.Length / 3];
                double b = rgbPixels[i + 2 * rgbPixels.Length / 3];
                double max = Math.Max(Math.Max(r, g), b);
                double min = Math.Min(Math.Min(r, g), b);
                double delta = max - min;
                if (delta != 0)
                {
                    double hue;
                    if (r == max)
                    {
                        hue = (g - b) / delta;
                    }
                    else
                    {
                        if (g == max)
                        {
                            hue = 2 + (b - r) / delta;
                        }
                        else
                        {
                            hue = 4 + (r - g) / delta;
                        }
                    }
                    hue *= 60;
                    if (hue < 0) hue += 360;
                    hsbPixels[i] = hue;
                }
                else
                {
                    hsbPixels[i] = 0;
                }
                hsbPixels[i] /= 360; // hue will be [0, 360] - scale to [0, 1]
                hsbPixels[i + rgbPixels.Length / 3] = max == 0 ? 0 : (max - min) / max;
                hsbPixels[i + 2 * rgbPixels.Length / 3] = max;
            }

            return hsbPixels;
        }
    }
}
