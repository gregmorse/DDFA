using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
//using System.Windows.Shapes;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;
using MNISTDataLibrary;
using System.Reflection;
using System.IO;

namespace MNISTLoader
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();


            //var mnistLoader = new DataLoader();

            string currentAssemblyDirectoryName = Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location);
            string jsonPath = currentAssemblyDirectoryName + "/../../../../../../data/chunks/full.json";

            //going to pull in the full file and convert to byte pictures
            var allDigits = DataLoader.loadPicturesAsBytes(jsonPath);

            foreach (var kvp in allDigits)
            {
                //we will use digit later for labeling -- don't do anything for now
                int digit = kvp.Key;
                List<byte[]> pictures = kvp.Value;

                //we create a wrap panel to handle the display of digits
                WrapPanel wrap = new WrapPanel();
                wrap.Orientation = Orientation.Horizontal;
                

                //go through each picture (represented as byte array)
                foreach (byte[] pixelArray in pictures)
                {
                    //create an image that will hold our bitmap from bytes
                    System.Windows.Controls.Image bmp = new System.Windows.Controls.Image();
                    bmp.Width = 28;
                    bmp.Height = 28;
                    //jesus. have to do some work to let wpf display bitmaps
                    //first, we take our pixel array and convert to a new System.Drawing.Bitmap
                    //loadbitmap handles converting from bitmap to bitmap source necessary for wpf display
                    bmp.Source = loadBitmap(createBitmapFromBytes(28, 28, pixelArray));

                    //now take our image and add it to the wrap panel
                    wrap.Children.Add(bmp);
                }

                //now add our wrap to the stack panels
                imageStack.Children.Add(wrap);
            }
        }


        [DllImport("gdi32")]
        static extern int DeleteObject(IntPtr o);

        public static BitmapSource loadBitmap(System.Drawing.Bitmap source)
        {
            IntPtr ip = source.GetHbitmap();
            BitmapSource bs = null;
            try
            {
                bs = System.Windows.Interop.Imaging.CreateBitmapSourceFromHBitmap(ip,
                   IntPtr.Zero, Int32Rect.Empty,
                   System.Windows.Media.Imaging.BitmapSizeOptions.FromEmptyOptions());
            }
            finally
            {
                DeleteObject(ip);
            }

            return bs;
        }

        public Bitmap createBitmapFromBytes(int width, int height, byte[] pixels)
        {
            var b = new Bitmap(width, height, System.Drawing.Imaging.PixelFormat.Format24bppRgb);

            //ColorPalette ncp = b.Palette;
            //for (int i = 0; i < 256; i++)
            //    ncp.Entries[i] = Color.FromArgb(255, i, i, i);
            //b.Palette = ncp;

            //create a known rectange bounds for this
            var BoundsRect = new System.Drawing.Rectangle(0, 0, width, height);

            //grab the whole bitmap data (bounds can be used to select only a portion)
            BitmapData bmpData = b.LockBits(BoundsRect,
                                            ImageLockMode.WriteOnly,
                                            b.PixelFormat);

            //grab pointer to start data location
            IntPtr ptr = bmpData.Scan0;

            //how many bytes do we need
            int bytes = bmpData.Stride * b.Height;

            //test that the desired bytes is equal to the number of bytes sent in (i.e. width/height and pixel count match)
            if (bytes != pixels.Length)
                throw new Exception("Improper byte length -- should equal : " + bytes + " but " + pixels.Length + " were sent in.");

            //copy over the pixels into the bitmap info
            Marshal.Copy(pixels, 0, ptr, bytes);

            //unlock, read for bitmap display
            b.UnlockBits(bmpData);
            
            //return the bitmap!
            return b;

        }


    }
}
