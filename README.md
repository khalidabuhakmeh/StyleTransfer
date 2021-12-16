# ONNX.FastNeuralStyleTransfer-Demo

Sample code to do style transfer using ONNX Model
Zoo models.

# Models used

https://github.com/onnx/models/tree/master/vision/style_transfer/fast_neural_style

# Sample input/output
![](src/input/thiscatdoesnotexist_01.jpg)

![](src/output/expected_thiscatdoesnotexist_01_mosaic.jpg)

## Special Thanks

- Gerardo Lijs for the initial code sample
- ImageSharp for making great image manipulation APIS.

## Note to macOS Users

The `Microsoft.ML.OnnxRuntime` package has issues. To run this sample on macOS you'll need to `brew install onnxruntime` and then it will work.