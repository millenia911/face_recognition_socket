import socketio
import cv2
import os
import tqdm
from multiprocessing import Process

sio = socketio.Client()

def convert_picture(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    _, img = cv2.imencode('.jpg', img)
    return img.tobytes()    
@sio.event
def connect():
    print('connection established')
@sio.event
def disconnect():
    print('disconnected from server')
    exit()
    
def emit_picture_submit(name, path):
    img_byte = convert_picture(path)
    sio.emit("submit_picture", {"name":name, "image_bytes": img_byte}, namespace="/admin")

def main_process():
    sio.connect("http://127.0.0.1:5000", wait_timeout=200, namespaces=["/", "/admin"])
    print("SID is ", sio.sid)
    for f in os.listdir("./people"):
        pth = os.path.join("./people", f)
        if os.path.isdir(pth):
            pbar = tqdm.tqdm(total=len(os.listdir(pth)))
            print(f"Sumbitting images for {f}")
            for p in os.listdir(pth):
                emit_picture_submit(name=f, path=os.path.join(pth, p))
                pbar.update(1)
            pbar.close()
    sio.emit("disconnect_request", namespace="/")  
    sio.emit("disconnect_request", namespace="/admin")

p1 = Process(target=main_process)
p1.start()
p1.join()

