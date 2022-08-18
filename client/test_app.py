import socketio
import cv2, os, tqdm
import numpy as np
# from multiprocessing import Process
import time

sio = socketio.Client()
tic = 0
toc = 0
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
    sio.disconnect()

def emit_picture_submit(name, path):
    img_bytes = convert_picture(path)
    sio.emit("submit_picture", (name, img_bytes), namespace="/admin")

def stream_transmit_event(img, namespace="/stream"):
    img_bytes = convert_picture(img)
    sio.emit("transmit_img", (img_bytes), namespace=namespace)

def main_process():
    sio.connect("http://127.0.0.1:5000", wait_timeout=200, namespaces=["/", "/admin", "/stream"])
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

@sio.on("inference_result", namespace="/stream")
def get_result(data, image_byte):
    nparr = np.frombuffer(image_byte, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    pic_path = "./results"
    toc = time.time()
    print("time elapsed= ", toc-tic)
    # print(data)
    os.makedirs(pic_path, exist_ok=True)
    #_ = cv2.imwrite(os.path.join(pic_path, f"result_{random.randint(1, 100000)}.jpg"), img)
    # sio.emit("disconnect_request", namespace="/stream")  

sio.connect("https://0aca-114-4-215-245.ap.ngrok.io/", wait_timeout=200, namespaces=["/stream"])
print("SID is ", sio.sid)
def main_process2():
    stream_transmit_event("/home/millenia/code/face_rek/client/tom_lizzie.jpg")


# p1 = Process(target=main_process2)
# p1.start()
# p1.join()
for i in range(20):
    tic = time.time()
    main_process2()
