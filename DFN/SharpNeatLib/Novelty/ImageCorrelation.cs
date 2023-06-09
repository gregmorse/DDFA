using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace SharpNeat.Novelty
{
    public class ImageCorrelation
    {
        public int Image1 { get; set; }
        public int Image2 { get; set; }
        public float CorrCoef { get; set; }
        public float ImageDiff { get; set; }

        public ImageCorrelation(int image1, int image2, float corrCoef, float imageDiff)
        {
            this.Image1 = image1;
            this.Image2 = image2;
            this.CorrCoef = corrCoef;
            this.ImageDiff = imageDiff;
        }
    }
}
