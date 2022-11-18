using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace ConsoleApp2
{
    class Program
    {
        const int Width = 28;
        const int Height = 28;

        static void Main(string[] args)
        {


            string imagePath = @Console.ReadLine();
            //var image = Process(Image.Load<Rgb24>(imagePath));
            var image = PreprocessTestImageOriginal(imagePath);
            const string modelPath = @"keras_mnist.onnx";
            float[] inputfilearray = Predict(modelPath, image);
            int prediction = inputfilearray.ToList().IndexOf(inputfilearray.Max());
            Console.WriteLine($" 예측 : {prediction}");
             string ds=   Console.ReadLine();
        }
        private static float[][] PreprocessTestImageOriginal(string path)
        {
            var img = new System.Drawing.Bitmap(path);

            int width = Width ; 
            int height = Height ;
            System.Drawing.Size resize = new System.Drawing.Size(width, height);
            System.Drawing.Bitmap resizeImage = new System.Drawing.Bitmap(img, resize);
            

            var result = new float[resizeImage.Height][];

            for (int i = 0; i < resizeImage.Height; i++)
            {
                result[i] = new float[resizeImage.Width];
                for (int j = 0; j < resizeImage.Width; j++)
                {
                    var pixel = resizeImage.GetPixel(j, i);

                    var gray = RgbToGray(pixel);

                    var normalized = 0<gray ?1:0 ;

                    result[i][j] = gray;

                }
                Console.WriteLine();
            }
            return result;
        }
        private static float RgbToGray(System.Drawing.Color pixel) => 0.299f * pixel.R + 0.587f * pixel.G + 0.114f * pixel.B;

        public static float[] Process(Image<Rgb24> original)
        {
            var floats = new float[Width * Height*3];
            var image = original.Clone(ctx =>
            {
                ctx.Resize(new ResizeOptions
                {
                    Size = new Size(Width, Height),
                    Mode = ResizeMode.Crop
                });// .BinaryThreshold( 0.5f );//이진화
            });
            for (var x = 0; x < image.Width; x++)
            {
                for (var y = 0; y < image.Height; y++)
                {
                    floats[x + y * image.Width] = (image[x, y].R == 255) ? 0 : 1;
                }
            }
            return floats;
        }
        private static float[] Predict(string modelPath, float[][] image)
        {

            using (var session = new InferenceSession(modelPath))
            {
                var modelInputLayerName = session.InputMetadata.Keys.Single();
                var innodedims = new int[] {1, 28, 28 ,1}; 
                var inputTensor = new DenseTensor<float>(image.SelectMany(x => x).ToArray(), innodedims);
                var modelInput = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(modelInputLayerName, inputTensor)
            };
                var result = session.Run(modelInput);
                return ((DenseTensor<float>)result.Single().Value).ToArray();
            }

        }
    }
}
