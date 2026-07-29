import onnxruntime as ort
import torch
import numpy as np

dummy_input = torch.randn(1, 3, 224, 224)  # Example input

session = ort.InferenceSession("tuned_effnet.onnx")
# session = ort.InferenceSession("tuned_regnet.onnx")
inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
outputs = session.run(None, inputs)
_, predicted = torch.max(torch.tensor(np.asarray(outputs[0])), 1)
print(predicted.tolist()[0])