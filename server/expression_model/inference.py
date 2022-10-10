import numpy as np
import cv2, time
from .get_model import arch_01_ONNX as onnx
import onnxruntime
onnx_instance = onnxruntime.capi.onnxruntime_inference_collection.InferenceSession

def predict(img, model, classes_names):
  # return classes_names[3], .777
  start = time.perf_counter()
  img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
  # x = image.img_to_array(img)  
  x = np.expand_dims(img, axis=0)   
  images = np.vstack([x])
  if isinstance(model, onnxruntime.capi.onnxruntime_inference_collection.InferenceSession):
    images = images.astype(np.float32)
    classes = onnx.predict(images)
  else:
    raise(ValueError("MODEL IS NOT ONNX")) 
    # classes = model.predict(images)  

  confidence = (np.amax(classes)/1)*100
  predicted_class_index = np.argmax(classes)
  predicted_class = classes_names[predicted_class_index]
  print(f"EXP LATENCY = {time.perf_counter() - start}")
  return predicted_class, round(confidence, 3)
 