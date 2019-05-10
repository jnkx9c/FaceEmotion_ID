from keras.preprocessing.image import img_to_array
#import imutils
import cv2
from keras.models import load_model
import numpy as np

# parameters for loading data and images
detection_model_path = 'haarcascade_files/haarcascade_frontalface_default.xml'
FER_emotion_model_path = 'models/_mini_XCEPTION.106-0.65.hdf5'
emotion_model_path = 'models/moods_win_dataaug1.h5'

# hyper-parameters for bounding boxes shape
# loading models
face_detection = cv2.CascadeClassifier(detection_model_path)
FER_emotion_classifier = load_model(FER_emotion_model_path, compile=False)
emotion_classifier = load_model(emotion_model_path, compile=False)

FER_EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised", "neutral"]
PROJ_EMOTIONS = ["cry","laugh","neutral","sad","smile"]


# starting video streaming
cv2.namedWindow('FER Benchmark')
camera = cv2.VideoCapture(0)
FER_preds = []
PROJ_preds = []
while True:
    frame = camera.read()[1]
    #reading the frame
    #frame = imutils.resize(frame,width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    FER_canvas = np.zeros((250, 300, 3), dtype="uint8")
    PROJ_canvas = np.zeros((250, 300, 3), dtype="uint8")
    FER_frameClone = frame.copy()
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces
                    # Extract the ROI of the face from the grayscale image, resize it to a fixed 48x48 pixels, and then prepare
            # the ROI for classification via the CNN
        FER_roi = gray[fY:fY + fH, fX:fX + fW]
        FER_roi = cv2.resize(FER_roi, (48, 48))
        FER_roi = FER_roi.astype("float") / 255.0
        FER_roi = img_to_array(FER_roi)
        FER_roi = np.expand_dims(FER_roi, axis=0)        
        
        FER_preds = FER_emotion_classifier.predict(FER_roi)[0]
        FER_emotion_probability = np.max(FER_preds)
        FER_label = FER_EMOTIONS[FER_preds.argmax()]


        PROJ_roi = gray[fY:fY + fH, fX:fX + fW]
        PROJ_roi = cv2.resize(PROJ_roi, (256, 256))
        PROJ_roi = PROJ_roi.astype("float") / 255.0
        PROJ_roi = img_to_array(PROJ_roi)
        PROJ_roi = np.expand_dims(PROJ_roi, axis=0)

        PROJ_preds = emotion_classifier.predict(PROJ_roi)[0]
        PROJ_emotion_probability = np.max(PROJ_preds)
        PROJ_label = PROJ_EMOTIONS[PROJ_preds.argmax()]

        
    if(len(FER_preds)>0):
        for (i, (emotion, prob)) in enumerate(zip(FER_EMOTIONS, FER_preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(FER_canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 0, 255), -1)
                    cv2.putText(FER_canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
                    cv2.putText(FER_frameClone, FER_label, (fX, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.rectangle(FER_frameClone, (fX, fY), (fX + fW, fY + fH),
                                (255, 255, 255), 2)

    if(len(PROJ_preds)>0):
        for (i, (emotion, prob)) in enumerate(zip(PROJ_EMOTIONS, PROJ_preds)):
                    # construct the label text
                    text = "{}: {:.2f}%".format(emotion, prob * 100)
                    w = int(prob * 300)
                    cv2.rectangle(PROJ_canvas, (7, (i * 35) + 5),
                    (w, (i * 35) + 35), (0, 255, 0), -1)
                    cv2.putText(PROJ_canvas, text, (10, (i * 35) + 23),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)
                    cv2.putText(FER_frameClone, PROJ_label, (fX + fW, fY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    #cv2.rectangle(FER_frameClone, (fX, fY), (fX + fW, fY + fH),  (0, 0, 255), 2)                            

    cv2.imshow('Mood Dectection Benchmark', FER_frameClone)
    cv2.imshow("FER Benchmark Probabilities", FER_canvas)
    cv2.imshow("PROJECT Benchmark Probabilities", PROJ_canvas)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
