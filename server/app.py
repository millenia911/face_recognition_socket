from flask import Flask, request, copy_current_request_context
from flask_socketio import SocketIO, disconnect
from sections.admin_page import AdminPage
from sections.stream_page import StreamPage
import uuid

app = Flask(__name__)
app.config["SECRET_KEY"] = str(uuid.uuid4())
app.debug = True
socketio = SocketIO(app, async_mode="eventlet", logger=True, 
                    engineio_logger=True, max_http_buffer_size=100000000)


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
    socketio.run(app, port=5000)