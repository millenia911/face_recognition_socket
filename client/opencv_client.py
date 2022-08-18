import cv2, os, time, random
import numpy as np
import socketio

sio = socketio.Client()
ret_img = None
# blank_image = np.zeros(shape=[480, 640, 3], dtype=np.uint8)
# cv2.imshow("frame", blank_image)


def convert_picture(path):
    if os.path.isfile(path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        _, img = cv2.imencode('.jpg', img)
    elif isinstance(path, np.ndarray):
        _, img = cv2.imencode('.jpg', path)
        return img.tobytes()
    else: raise TypeError("Image type is not supported")
    return img.tobytes()    

@sio.event
def connect():
    print('connection established')

@sio.event
def disconnect():
    print('disconnected from server')
    sio.disconnect()

def stream_transmit_event(img, namespace="/stream"):
    img_bytes = convert_picture(img)
    sio.emit("transmit_img", (img_bytes, 1), namespace=namespace)

@sio.on("inference_result", namespace="/stream")
def get_result(data, image_byte):
    global ret_img, state
    nparr = np.frombuffer(image_byte, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    ret_img = img
    state = "response_recieved"
    # cv2.imshow('frame', ret_img)
    # cv2.imwrite('frame.jpg', ret_img)
rand_num = random.randint(1, 100000)
sio.connect("http://127.0.0.1:8888", wait_timeout=200, namespaces=["/stream"])
print("SID is ", sio.sid)
vid = cv2.VideoCapture(0)
state = "response_recieved"

while(vid.isOpened()):
    toc = time.perf_counter()
    if state == "response_recieved":
        _, frame = vid.read()
        state = "transmitting"
        stream_transmit_event(frame)
    # cv2.imshow('frame', frame)
    cv2.imshow(f'frame{rand_num}', frame)
    while(state == "transmitting"): pass
    tic = time.perf_counter()
    if ret_img is not None:
        if len(ret_img) > 0:
            cv2.imshow(f'frame_res{rand_num}', ret_img)
    print("FPS = ", round(1/(tic-toc), 3))

    # time.sleep(0.9)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
  
vid.release()
cv2.destroyAllWindows()
sio.disconnect()