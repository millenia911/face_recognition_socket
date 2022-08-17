from flask import copy_current_request_context
from flask_socketio import Namespace, emit, disconnect
import os, random, shutil, cv2, uuid
import numpy as np

class AdminPage(Namespace):
    def __init__(self, namespace, im_path="./people"):
        super().__init__(namespace)
        self.im_path = im_path

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
    def on_submit_picture(self, name, image_bytes):
        if not self.is_valid_data([name, image_bytes]):
            print("failed to process data")
            self.emit_admin_event_message(status="failed", 
                                            msg="Name or image cannot be none or zero lenght")
            return
        
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pic_path = os.path.join(self.im_path, name)
        os.makedirs(pic_path, exist_ok=True)
        _ = cv2.imwrite(os.path.join(pic_path, f"{name}_{random.randint(1, 100000)}.jpg"), img)

    def on_delete_person(self, name):
        # TODO: TEST THIS, DONT FORGET
        # TODO: delete on embedding pickle and restart emb dataframe on inference
        if not self.is_valid_data([name]):
            self.emit_admin_event_message(status="failed", 
                                          msg="Name or image cannot be none or zero lenght")
            return
        path = os.path.join(self.im_path, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
        else:
            self.emit_admin_event_message(status="failed",
                                          msg="Name is not found")
    def on_disconnect_request(data=None):
        @copy_current_request_context
        def dc():
            disconnect()
        dc()
