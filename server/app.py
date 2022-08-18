from flask import Flask, request, copy_current_request_context, Response
from flask_socketio import SocketIO, disconnect
from sections.admin_page import AdminPage
from sections.stream_page import StreamPage
import uuid, cv2, os, random, json, base64
from io import BytesIO
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.debug = True
socketio = SocketIO(app, async_mode="eventlet", logger=True, 
                    engineio_logger=True, max_http_buffer_size=100000000)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

@app.route("/api/admin", methods=["POST"])
def submit_picture_request():
    try:
        req = request.json
        req = json.loads(req)
        name = req["name"]
        image_bytes = req["image_bytes"]
        image_bytes = list(image_bytes.values())[0]
        image_bytes = base64.b64decode(image_bytes)
        im_path = "./people"
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        pic_path = os.path.join(im_path, name)
        os.makedirs(pic_path, exist_ok=True)
        _ = cv2.imwrite(os.path.join(pic_path, f"{name}_{random.randint(1, 100000)}.jpg"), img)
        return Response(response="success", status=200, mimetype="text/plain")
    except Exception as e:
        return Response(response=str(e), status=400, mimetype="text/plain")

socketio.on_namespace(AdminPage("/admin"))
socketio.on_namespace(StreamPage("/stream"))

@socketio.event
def disconnect_request(data=None):
    @copy_current_request_context
    def dc():
        disconnect()
    dc()
    
@socketio.on('disconnect')
def disconnect_response():
    print('Client disconnected', request.sid)


if __name__ == "__main__":
    socketio.run(app, port=8888)