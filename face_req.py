from deepface import DeepFace
from deepface.detectors import FaceDetector
from deepface.commons import functions, distance as dst
import bbox_visualizer as bbv
import pandas as pd
import os
import cv2
import tqdm

def build_models(model_name="dlib", detector_model="opencv"):
    # step 0, initiate model
    model = DeepFace.build_model(model_name)
    print(model_name," is built")
    face_detector = FaceDetector.build_model(detector_model)

    return model, face_detector

def det_face(img, face_detector):
    faces = FaceDetector.detect_faces(face_detector, detector_backend="opencv", 
                                      img=img, align = False)
    return faces # face and xywh

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

    # if data_frame.shape[0] > 0:
    #     pass
    # else: print("There's no face to verify")
    
    result= []
    for face, (x,y,w,h) in faces:
        xyxy = xywh_to_xyxy(x,y,w,h)
        face_name, distance = compare_faces(face)
        if distance < max_distance:
            # [status, name, distance]
            result.append(["known", face_name, distance, list(xyxy)])
        else: result.append(["unknown", face_name, distance, list(xyxy)])
    return result

input_shape = (150,150)
input_shape_x, input_shape_y = input_shape

settings = {
    "model": "Dlib",
    "detector_model": "opencv",
    "distance_metric": "consine" 
}

def get_image(im_path):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    h, w, _ = im.shape
    # if h > 200 or w > 200:
    #     if w > h :
    #         h = int((200/w)*h)
    #         im = cv2.resize(im, (200, h), interpolation=cv2.INTER_AREA)
    #     elif h > w :
    #         w = int((200/h)*w)
    #         im = cv2.resize(im, (w, 200), interpolation=cv2.INTER_AREA)

    return im

def save_im(im, im_name):
    im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)
    cv2.imwrite(im_name, im)

def xywh_to_xyxy(x,y,w,h):
    return int(x), int(y), int(x+w), int(y+h)

if __name__=="__main__":    
    model_name = "Dlib"
    detector_model = "opencv"
    model, face_det_model = build_models(model_name, detector_model)

    emb_data = create_representation_dataframe(model, faces_dir="./people")
    
    # start detection
    im = get_image("./unknown_group.jpg")
    detected_faces = det_face(im, face_det_model)
    res = recog_face(data_frame=emb_data, faces=detected_faces, model=model)

    res = pd.DataFrame(res, columns=["status", "name", "distance", "box"])

    im = bbv.draw_multiple_rectangles(im, res["box"].tolist())
    im = bbv.add_multiple_labels(im, res["name"].tolist(), res["box"].tolist())
    print(res["box"])
    save_im(im, "./detection_result.jpg")
