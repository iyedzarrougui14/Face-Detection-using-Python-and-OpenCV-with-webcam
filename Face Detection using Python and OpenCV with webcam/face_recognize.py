import cv2
import os
import numpy as np

size = 4
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'

# Create LBPH face recognizer
print("Recognizing face, please be in sufficient light.")

# Create a list of images and a list of corresponding names
(images, labels, names, id) = ([], [], {}, 0)
for (subdirs, dirs, files) in os.walk(datasets):
    for subdir in dirs:
        names[id] = subdir
        subjectpath = os.path.join(datasets, subdir)
        for filename in os.listdir(subjectpath):
            path = os.path.join(subjectpath, filename)
            label = id
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
        id += 1
(width, height) = (130, 100)

# Creating a numpy array from the two lists above
(images, labels) = [np.array(lis) for lis in [images, labels]]

# OpenCV trains a model on the images
model = cv2.face.LBPHFaceRecognizer_create()
model.train(images, labels)

# Use the recognizer on the camera stream
face_cascade = cv2.CascadeClassifier(haar_file)
webcam = cv2.VideoCapture(0)

while True:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))
        
        # Try to recognize the face
        prediction = model.predict(face_resize)
        cv2.rectangle(im, (x, y), (x + w, y + h), (255, 0, 0), 3)
        
        if prediction[1] < 500:
            cv2.putText(im, '% s - %.0f' % 
                       (names[prediction[0]], prediction[1]), 
                       (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255)) 
        else:
            cv2.putText(im, 'not recognized',  
                       (x-10, y-10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255)) 
        
        cv2.imshow('Face Recognizer', im)
      
    key = cv2.waitKey(10) 
    if key == ord('q'): 
        break

webcam.release()
cv2.destroyAllWindows()
