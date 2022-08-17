from deepface import DeepFace
from deepface.detectors import FaceDetector
from deepface.commons import functions, distance as dst
from expression_model.get_model import arch_01, class_names
from expression_model.inference import predict
import bbox_visualizer as bbv
import pandas as pd
import os
import cv2
import tqdm
import numpy as np

def build_models(model_name="dlib", 
                 detector_model="opencv", 
                 expression_model=arch_01):
    # step 0, initiate model
    model = DeepFace.build_model(model_name)
    face_detector = FaceDetector.build_model(detector_model)
    if expression_model is not None:
        exp_model = expression_model()
    return model, face_detector, exp_model

def predic_expression(imgs_input, exp_model, class_list):
    result = []
    for face, (x,y,w,h) in imgs_input:
        exp_class, confidence = predict(face, exp_model, class_list)
        result.append([exp_class, confidence])
    return result

def det_face(img, face_detector, detector_backend="opencv"):
    faces = FaceDetector.detect_faces(face_detector, detector_backend=detector_backend, 
                                      img=img, align = False)
    return faces # [face, xywh]

def encode_representation(img, model):
    preprocessed_img = functions.preprocess_face(img = img, 
                                                 target_size = (input_shape_y, input_shape_x), 
                                                 enforce_detection = False, 
                                                 detector_backend = 'opencv', 
                                                 align=True)

    encode = model.predict(preprocessed_img)[0,:]
    return encode

def create_representation_dataframe(model, data_dir="./", faces_dir="./faces"):
    names = os.listdir(faces_dir)
    saved_emb = None
    filenames = []
    if "embedding_data.pkl" in os.listdir(data_dir):
        saved_emb = pd.read_pickle(os.path.join(data_dir, "embedding_data.pkl"))
        filenames = saved_emb["filename"].to_list()

    embeddings = []
    for name in names:
        face_pics = os.listdir(os.path.join(faces_dir, name))
        print(f"Generating representations for {name}")
        pbar = tqdm.tqdm(total=len(face_pics))
        for fp in face_pics:
            # if fp is in filename, means it was encoded before, so we can skip it.
            if fp in filenames:
                pbar.update(1)
                continue
            
            emb_list = []
            if fp.split(".")[-1] in ["jpg", "png", "jpeg"]:
                fp_path = os.path.join(faces_dir, name, fp)
                im = cv2.imread(fp_path)
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                h, w, _ = im.shape

                if h > 200 or w > 200:
                    if w > h :
                        h = int((200/w)*h)
                        im = cv2.resize(im, (200, h), interpolation=cv2.INTER_AREA)
                    elif h > w :
                        w = int((200/h)*w)
                        im = cv2.resize(im, (w, 200), interpolation=cv2.INTER_AREA)

                emb = encode_representation(im, model)
                emb_list = [name, fp, emb]
                embeddings.append(emb_list)
            pbar.update(1)
        pbar.close()
    embeddings = pd.DataFrame(embeddings, columns=["name", "filename", "embedding"])

    if saved_emb is not None:
        embedding_dataframe = pd.concat([saved_emb, embeddings], ignore_index=True)
        embedding_dataframe.reset_index()
        embedding_dataframe.to_pickle(os.path.join(data_dir, "embedding_data.pkl"))
        return embedding_dataframe
    else: 
        embeddings.to_pickle(os.path.join(data_dir, "embedding_data.pkl"))
        return embeddings
        

def recog_face(data_frame, faces, model, distance_metric="cosine", max_distance = 0.07):
    """return: faces and their name for one picture"""
    def compare_faces(img, data_frame=data_frame):
        preprocessed_img = functions.preprocess_face(img=img, 
                                                     target_size = (input_shape_y, input_shape_x), 
                                                     enforce_detection = False, 
                                                     detector_backend = 'opencv')

        if preprocessed_img.shape[1:3] == input_shape:
                img1_representation = model.predict(preprocessed_img)[0,:]
                def findDistance(row):
                    img2_representation = row['embedding']

                    distance = 1000 #initialize very large value
                    if distance_metric == 'cosine':
                        distance = dst.findCosineDistance(img1_representation, img2_representation)
                    elif distance_metric == 'euclidean':
                        distance = dst.findEuclideanDistance(img1_representation, img2_representation)
                    elif distance_metric == 'euclidean_l2':
                        distance = dst.findEuclideanDistance(dst.l2_normalize(img1_representation), dst.l2_normalize(img2_representation))

                    return distance

                data_frame['distance'] = data_frame.apply(findDistance, axis = 1)
                data_frame = data_frame.sort_values(by = ["distance"])

                candidate = data_frame.iloc[0]
                face_name = candidate['name']
                best_distance = candidate['distance']
                return face_name, best_distance

    if data_frame.shape[0] <= 0:
        print("There's no face to verify")
        return None, "Warning!, there's no face to verify"
    
    result= []
    for face, (x,y,w,h) in faces:
        xyxy = xywh_to_xyxy(x,y,w,h)
        face_name, distance = compare_faces(face)
        if distance < max_distance:
            # [status, name, distance]
            result.append(["known", face_name, distance, np.array(xyxy)])
        else: result.append(["unknown", face_name, distance, np.array(xyxy)])
    return result

input_shape = (150,150)
input_shape_x, input_shape_y = input_shape

def get_image(im_src, max_w_or_h=200):
    if os.path.isfile(im_src):
        im = cv2.imread(im_src)
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    elif isinstance(im_src, np.ndarray):
        im = im_src
    else: raise TypeError("Image type is not supported")

    if max_w_or_h is None:
        return im
    down_scale=1
    im_org = im.copy()
    h, w, _ = im.shape
    if h > max_w_or_h or w > max_w_or_h:
        if w > h :
            down_scale = max_w_or_h/w
            h = int(down_scale*h)
            im = cv2.resize(im, (max_w_or_h, h), interpolation=cv2.INTER_AREA)
        elif h > w :
            down_scale = max_w_or_h/h
            w = int(down_scale*w)
            im = cv2.resize(im, (w, max_w_or_h), interpolation=cv2.INTER_AREA)

        return im, down_scale, im_org

def save_im(im, im_name):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_name, im)

def xywh_to_xyxy(x,y,w,h):
    return (x), (y), (x+w), (y+h)

def draw_boxes(img, res_data: pd.DataFrame, label_col="name", color=(255,255,255), upscale=1):
    up_scaled_bbox = (np.array(res_data["box"].to_list()))*((upscale))
    up_scaled_bbox = up_scaled_bbox.astype(int)
    img = bbv.draw_multiple_rectangles(img, up_scaled_bbox, bbox_color=color)
    img = bbv.add_multiple_labels(img, res_data[label_col].tolist(), up_scaled_bbox, text_bg_color=color)
    img = bbv.add_multiple_T_labels(img, res_data["expression"].tolist(), up_scaled_bbox)

    return img

settings = {
    "model": "Dlib",
    "detector_model": "opencv",
    "distance_metric": "consine" 
}

model_name = settings["model"]
detector_model = settings["detector_model"]
model, face_det_model, exp_rec_model = build_models(model_name, detector_model)

emb_data = create_representation_dataframe(model, faces_dir="./people")

def main_recognition(img):    
    # start detection
    im, dw_scale, im_original = get_image(img)
    detected_faces = det_face(im, face_det_model, detector_backend=detector_model)
    res = recog_face(data_frame=emb_data, faces=detected_faces, model=model)
    exp = predic_expression(detected_faces, exp_rec_model, class_list=class_names)

    if res[0] is not None:
        res = pd.DataFrame(res, columns=["status", "name", "distance", "box"])
        exp = pd.DataFrame(exp, columns=["expression", "exp_score"])
        res = pd.concat([res,exp], axis=1)
        res_known = res.loc[res["status"] == "known"]
        res_unknown = res.loc[res["status"] == "unknown"]
        im = draw_boxes(im_original, res_known, upscale=1/dw_scale)
        im = draw_boxes(im, res_unknown, label_col="status", color=(255,50,10), upscale=1/dw_scale)
        # save_im(im, "./detection_result.jpg")
        return im, res.to_json()
    else: 
        print(res[1])
        return res