using OpenCvSharp;

namespace ONNX.FastNeuralStyleTransfer;

public class Program
{
    static void Main()
    {
        var inputImage = new Mat(@"input\thiscatdoesnotexist_01.jpg");

        var model = new FastNeuralStyleTransfer_Model(@"model\mosaic.onnx");
        var outputImage = model.RunInference(inputImage);

        Directory.CreateDirectory("output");
        outputImage.SaveImage(@"output\thiscatdoesnotexist_01_mosaic.jpg");
    }
}
