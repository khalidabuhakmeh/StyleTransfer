namespace ONNX.FastNeuralStyleTransfer;

using System;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;

public static class ONNXUtils
{
    /// <summary>
    /// Create DenseTensor for ONNX image input.
    /// Image must already be in the Model Input desired size and number of channels.
    /// This method modifies image received in arguments so make a copy before if you wan't to avoid this.
    /// </summary>
    public static DenseTensor<float> CreateInputImageTensor(Mat image) => CreateInputImageTensor(image, normalize: false);

    /// <summary>
    /// Create DenseTensor for ONNX image input.
    /// Image must already be in the Model Input desired size and number of channels.
    /// This method modifies image received in arguments so make a copy before if you wan't to avoid this.
    /// </summary>
    public static DenseTensor<float> CreateInputImageTensor(Mat image, bool normalize, float[]? normalize_mean = null, float[]? normalize_std = null)
    {
        var imageChannels = image.Channels();

        // Check arguments
        if (normalize)
        {
            if (normalize_mean is null) throw new ArgumentNullException(nameof(normalize_mean));
            if (normalize_std is null) throw new ArgumentNullException(nameof(normalize_std));

            if (normalize_mean.Length != imageChannels) throw new ArgumentException($"{nameof(normalize_mean)} argument array lenght should match image number of channels.");
            if (normalize_std.Length != imageChannels) throw new ArgumentException($"{nameof(normalize_std)} argument array lenght should match image number of channels.");
        }

        var input = new DenseTensor<float>(new[] { 1, imageChannels, image.Height, image.Width });
        var mat3 = new Mat<Vec3b>(image);
        var indexer = mat3.GetIndexer();

        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var color = indexer[y, x];
                if (normalize)
                {
                    input[0, 0, y, x] = (color.Item0 - normalize_mean![0]) / normalize_std![0];
                    input[0, 1, y, x] = (color.Item1 - normalize_mean[1]) / normalize_std[1];
                    input[0, 2, y, x] = (color.Item2 - normalize_mean[2]) / normalize_std[2];
                }
                else
                {
                    input[0, 0, y, x] = color.Item0;
                    input[0, 1, y, x] = color.Item1;
                    input[0, 2, y, x] = color.Item2;
                }
            }
        }

        return input;
    }

    /// <summary>
    /// Create a Mat image from ONNX Tensor output.
    /// </summary>
    public static Mat CreateMatFromDenseTensor(Tensor<float> imageTensor, int imageWidth, int imageHeight, MatType imageType)
    {
        // TODO: Find more efficient way to create Mat from Tensor float values
        var image = new Mat(imageHeight, imageWidth, imageType);
        for (var y = 0; y < image.Height; y++)
        {
            for (var x = 0; x < image.Width; x++)
            {
                var color = new Vec3b(FloatPixelValueToByte(imageTensor[0, 0, y, x]), FloatPixelValueToByte(imageTensor[0, 1, y, x]), FloatPixelValueToByte(imageTensor[0, 2, y, x]));
                image.Set(y, x, color);
            }
        }

        return image;

        byte FloatPixelValueToByte(float pixelValue)
        {
            if (pixelValue < 0)
            {
                return 0;
            }
            else if (pixelValue > 255)
            {
                return 255;
            }
            else
            {
                return Convert.ToByte(pixelValue);
            }
        }
    }
}
