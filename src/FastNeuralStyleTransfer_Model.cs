namespace ONNX.FastNeuralStyleTransfer;

using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using OpenCvSharp;

public class FastNeuralStyleTransfer_Model : IDisposable
{
    private InferenceSession? _inferenceSession;

    public string ModelFilename { get; }

    /// <summary>
    /// Model image number of channels
    /// </summary>
    public int Image_Channels { get; set; } = 3;
    /// <summary>
    /// Model image width
    /// </summary>
    public int Image_Width { get; set; } = 224;
    /// <summary>
    /// Model image height
    /// </summary>
    public int Image_Height { get; set; } = 224;
    public bool Input_Normalization { get; set; }
    public float[] Input_Normalization_Mean { get; set; } = new float[] { 0, 0, 0 };
    public float[] Input_Normalization_Std { get; set; } = new float[] { 1, 1, 1 };

    public FastNeuralStyleTransfer_Model(string modelFilename)
    {
        ModelFilename = modelFilename;
    }

    public Mat RunInference(Mat image)
    {
        // Resize image is needed
        using var resizedImage = new Mat();
        Cv2.Resize(image, resizedImage, new Size(Image_Width, Image_Height));
        Cv2.CvtColor(resizedImage, resizedImage, ColorConversionCodes.BGR2RGB);

        // Create tensor
        var input = ONNXUtils.CreateInputImageTensor(resizedImage, Input_Normalization, Input_Normalization_Mean, Input_Normalization_Std);

        // Setup inputs and outputs
        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("input1", input)
            };

        // Run inference
        if (_inferenceSession is null)
        {
            _inferenceSession = new InferenceSession(ModelFilename);
        }
        using var results = _inferenceSession.Run(inputs);

        var outputTensor = results.First().AsTensor<float>();
        return ONNXUtils.CreateMatFromDenseTensor(outputTensor, Image_Width, Image_Height, MatType.CV_8UC3);
    }

    public void Dispose()
    {
        Dispose(true);
        GC.SuppressFinalize(this);
    }

    protected virtual void Dispose(bool disposing)
    {
        if (disposing)
        {
            if (_inferenceSession is not null)
            {
                _inferenceSession.Dispose();
                _inferenceSession = null;
            }
        }
    }
}
