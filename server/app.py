from flask import Flask, render_template, session, request
from flask_socketio import SocketIO, emit, disconnect, Namespace
import numpy as np
import cv2
import os
import shutil
import uuid

app = Flask(__name__,)
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.debug = True
socketio = SocketIO(app, async_mode=None)

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
    @socketio.event
    def on_submit_picture(self, name, image_bytes):
        if not self.is_valid_data([name, image_bytes]):
            self.emit_admin_event_message(status="failed", 
                                          msg="Name or image cannot be none or zero lenght")
            return
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pic_path = os.path.join(self.im_path)
        os.makedirs(pic_path, exist_ok=True)
        _ = cv2.imwrite(pic_path, img)

    @socketio.event
    def on_delete_person(self, name):
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

socketio.on_namespace(AdminPage("/api/admin"))
        
if __name__ == "__main__":
    socketio.run(app, port=5000)