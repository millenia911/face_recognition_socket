import numpy as np
import cv2

def predict(img, model, classes_names):
  img = cv2.resize(img, dsize=(64,64), interpolation=cv2.INTER_CUBIC)
  # x = image.img_to_array(img)  
  x = np.expand_dims(img, axis=0)   
  images = np.vstack([x])

  classes = model.predict(images)  
  confidence = (np.amax(classes)/1)*100
  predicted_class_index = np.argmax(classes)
  predicted_class = classes_names[predicted_class_index]

  return predicted_class, round(confidence, 3)