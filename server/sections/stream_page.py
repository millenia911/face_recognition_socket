from flask import copy_current_request_context
from flask_socketio import Namespace, emit, disconnect
from core_recognition.face_recognition import main_recognition
import cv2, numpy as np

class StreamPage(Namespace):
    def __init__(self, namespace="/"):
        super().__init__(namespace)
    
    def is_valid_data(self, data):
        for d in data:
            if d is None or len(d) == 0:
                return False
        return True

    def emit_admin_event_message(self, status=None, msg=None):
        if status is not None or msg is not None:
            emit("admin_event_message", {"status":status, 
                                         "message":msg})
        else:
            emit("admin_event_message", {"status":"failed",
                                         "message":"sorry, something is wrong"})
    def on_disconnect_request(data=None):
        @copy_current_request_context
        def dc():
            disconnect()
        dc()

    def on_transmit_img(self, image_bytes, with_image=1):
        if not self.is_valid_data([image_bytes]):
            print("failed to process data")
            self.emit_admin_event_message(status="failed", 
                                            msg="Image cannot be none or zero lenght")
            return
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        img_out, data = None, None
        img_out, data = main_recognition(img)
        # print("--------Recognition Failed--------")
        self.inference_result_event(img_out, data, with_image)
    
    def inference_result_event(self, img, data, with_image=1):
        if not self.is_valid_data([img, data]):
            print("Img or data return empty or None")
            self.emit_admin_event_message(status="failed",
                                          msg="Failed to process face recognition")
            raise ValueError("Img or data return empty or None")
        
        if with_image:
            _, img = cv2.imencode(".jpg", img)
            emit("inference_result", (data, img.tobytes()))
        else:
            emit("inference_result", (data, 0))
