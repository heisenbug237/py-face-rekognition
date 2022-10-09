import numpy as np
import os, sys, time
import imutils
import cv2
from imutils.video import VideoStream
from imutils import paths
from sklearn.preprocessing import LabelEncoder


MODEL_PATH = "deploy.prototxt.txt"
WEIGHTS_PATH = "res10_300x300_ssd_iter_140000.caffemodel"
TRAIN_DATA_PATH = './data/'

BLUE = (247, 173, 62) 
WHITE = (255, 255, 255)      
FONT = cv2.FONT_HERSHEY_SIMPLEX  


def drawRectangle(image, color, faces, thickness=2):
    (x, y, x1, y1) = faces
    h = y1 - y
    w = x1 - x
    barLength = int(h / 8)
    cv2.rectangle(image, (x, y-barLength), (x+w, y), color, -1)
    cv2.rectangle(image, (x, y-barLength), (x+w, y), color, thickness)
    cv2.rectangle(image, (x, y), (x1, y1), color, thickness)
    return image

def changeFontScale(h, fontScale):
    fontScale = h/108 * fontScale
    return fontScale

def detectFaces(image, model, return_boxes=False, recognizer=None, label_encoder=None, conf=0.3):
    h, w, _ = image.shape
    resizedImage = cv2.resize(image, (300, 300))
    blob = cv2.dnn.blobFromImage(resizedImage, 1.0, (300, 300), (104.0, 177.0, 124.0))
    model.setInput(blob)
    faces = model.forward()
    boxes = []
    for i in range(0, faces.shape[2]):
        confidence = faces[0, 0, i, 2]
        if confidence > conf:
            box = faces[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x, y, x1, y1) = box.astype("int")
            boxes.append((x, y, x1, y1))
            face_name = "unknown"
            if recognizer is not None:
                faceROI = image[y:y1, x:x1]
                faceROI = cv2.resize(faceROI, (48, 64))
                faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
                (pred, p_conf) = recognizer.predict(faceROI)
                face_name = label_encoder.inverse_transform([pred])[0]
            fontScale = changeFontScale(y1-y, 0.4)
            image = drawRectangle(image, BLUE, (x, y, x1, y1))
            text = "{} {:0.2f}%".format(face_name, confidence * 100)
            textY = y - 2
            if (textY - 2 < 20): textY = y + 20 
            cv2.putText(image, text, (x, textY), FONT, fontScale, WHITE, 1)
    if return_boxes: return boxes
    return image

def loadTrainData(data_path, model, min_samples=3):
    image_paths = list(paths.list_images(data_path))
    names = [p.split(os.path.sep)[-2] for p in image_paths]
    (names, counts) = np.unique(names, return_counts=True)
    names = names.tolist()
    print(names)
    faces = []
    labels = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        name = image_path.split(os.path.sep)[-2]
        if counts[names.index(name)] < min_samples:
            continue
        bboxes = detectFaces(image, model, return_boxes=True)
        for (sX, sY, eX, eY) in bboxes:
            if(sY==eY or sX==eX): continue
            faceROI = image[sY:eY, sX:eX]
            faceROI = cv2.resize(faceROI, (48, 64))
            faceROI = cv2.cvtColor(faceROI, cv2.COLOR_BGR2GRAY)
            faces.append(faceROI)
            labels.append(name)
    faces = np.array(faces)
    labels = np.array(labels)
    return (faces, labels)

def trainRecognizer(label_encoder, model):
    print('loading data...')
    (faces, labels) = loadTrainData(TRAIN_DATA_PATH, model, min_samples=3)
    labels = label_encoder.fit_transform(labels)
    print('training face recognizer...')
    recognizer = cv2.face.LBPHFaceRecognizer_create(radius=2, neighbors=16, grid_x=8, grid_y=8)
    start_time = time.time()
    recognizer.train(faces, labels)
    end_time = time.time()
    print('training completed in {:.2f} seconds'.format(end_time - start_time))
    return recognizer

def useWebcam(model, label_encoder=None, recognizer=None):
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        frame = detectFaces(frame, model, label_encoder=label_encoder, recognizer=recognizer)
        cv2.imshow("Face Detection", frame)
        if cv2.waitKey(1) > 0:
            break
    cv2.destroyAllWindows()
    vs.stop()

def useImage(model, label_encoder=None, recognizer=None):
    image = cv2.imread(sys.argv[1]) 
    image = detectFaces(image, model, label_encoder=label_encoder, recognizer=recognizer)
    cv2.imwrite("faces.jpg", image)
    cv2.imshow("Face Detection", image)
    cv2.waitKey(0)


def main():
    net = cv2.dnn.readNetFromCaffe(MODEL_PATH, WEIGHTS_PATH)
    le = LabelEncoder()
    recog = trainRecognizer(le, net)
    if len(sys.argv)==1:
        useWebcam(net, le, recog)
    elif len(sys.argv)==2:
        useImage(net, le, recog)
    else:
        print("Usage: python face-detect-dnn.py [optional.jpg]")
        exit()

if __name__ == "__main__":
    main()
