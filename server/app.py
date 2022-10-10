from flask import Flask, request, copy_current_request_context, Response
# from flask_sqlalchemy import SQLAlchemy
from flask_socketio import SocketIO, disconnect
from sections.admin_page import AdminPage
from sections.stream_page import StreamPage
from datetime import datetime
import uuid, cv2, os, random, json, base64
import numpy as np

app = Flask(__name__)
app.config["SECRET_KEY"] = str(uuid.uuid4())
# app.config["SQLALCHEMY_DATABASE_URI"] = 'sqlite:///detectionhistory.db'
app.debug = True
# db = SQLAlchemy(app)
# TODO: finish database feature
# class History(db.Model):
#     date = db.Column(db.DateTime, default=datetime.now)
#     status = db.Column(db.String(20))
#     name = db.Column(db.String(80))
#     distance = db.Column(db.Float)
#     box = None
#     expression = db.Column(db.String(20))
#     exp_score = db.Column(db.Float)

# TODO: fix 'async_mode', currently failed for gevent, eventlet, wsgi
socketio = SocketIO(app, logger=True, 
                    engineio_logger=True, max_http_buffer_size=100000000,
                    cors_allowed_origins="*")

@app.route("/")
def hello_world():
    return "<p>Hello!</p>"

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