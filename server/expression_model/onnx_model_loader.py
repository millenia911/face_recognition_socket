import onnxruntime as rt
import os
dir_here = os.path.dirname(os.path.realpath(__file__))

# TODO: IMPORTANT! Refactor model loading scheme
class ONNXModel():
    def __init__(self) -> None:
        self.model = self.init_model()

    def init_model(self):
        providers = ['CPUExecutionProvider']
        m = rt.InferenceSession(os.path.join(dir_here,"model_1.onnx"), providers=providers)
        self.outputs = m.get_outputs()[0].name
        return m

    def predict(self, input_img):
        onnx_pred = self.model.run([self.outputs], {"input": input_img})
        return onnx_pred