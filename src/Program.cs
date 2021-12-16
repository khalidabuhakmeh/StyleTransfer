using ONNX.FastNeuralStyleTransfer;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Formats;
using SixLabors.ImageSharp.Formats.Jpeg;
using SixLabors.ImageSharp.Formats.Png;
using SixLabors.ImageSharp.PixelFormats;

const string filename = "cat.png";
const string imageFilePath = @$"./input/{filename}";
Directory.CreateDirectory("output");

var styles = new[]
{
    (name: "mosaic", model: "mosaic.onnx"),
    (name: "candy", model: "candy-9.onnx"),
    (name: "pointilism", model: "pointilism-9.onnx"),
    (name: "rain-princess", model: "rain-princess-9.onnx"),
    (name: "udnie", model: "udnie-9.onnx"),
};

var image = Image.Load<Rgb24>(imageFilePath);
foreach (var (name, model) in styles) {
    var result = StyleTransfer.Process(image, model);    
    
    IImageEncoder encoder = Path.GetExtension(imageFilePath) switch {
        ".png" => new PngEncoder(),
        ".jpg" => new JpegEncoder(),
        _ => throw new Exception()
    };
    
    result.Save($@"./output/{name}_{filename}", encoder);
}









