from flask import Flask, render_template, session, request, copy_current_request_context
from flask_socketio import SocketIO, emit, disconnect, Namespace
import multiprocessing as mp
import numpy as np
import cv2
import os
import shutil
import uuid
import random
import asyncio

app = Flask(__name__)
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.debug = True
socketio = SocketIO(app, async_mode="eventlet", logger=True, 
                    engineio_logger=True, max_http_buffer_size=100000000)

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
    # @socketio.event
    def on_submit_picture(self, data):
        name = data["name"]
        image_bytes = data["image_bytes"]
        print(name, "is retrieved")
        print(type(image_bytes))
        if not self.is_valid_data([name, image_bytes]):
            print("failed?")
            self.emit_admin_event_message(status="failed", 
                                            msg="Name or image cannot be none or zero lenght")
            return
        print("here")
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pic_path = os.path.join(self.im_path, name)
        os.makedirs(pic_path, exist_ok=True)
        _ = cv2.imwrite(os.path.join(pic_path, f"{name}_{random.randint(1, 100000)}.jpg"), img)

    # @socketio.event
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
    def on_disconnect_request():
        @copy_current_request_context
        def dc():
            disconnect()
        dc()
    


socketio.on_namespace(AdminPage("/admin"))

@socketio.event
def disconnect_request():
    @copy_current_request_context
    def dc():
        disconnect()
    dc()
    
@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected', request.sid)

if __name__ == "__main__":
    socketio.run(app, port=5000)