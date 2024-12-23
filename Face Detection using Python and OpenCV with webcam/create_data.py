#Creating the database
#It captures Images and Saves them into Dataset
#Folder under the folder name is sub_data

import cv2,  os
haad_file="haarcascade_frontalface_default.xml"


#All faces data will be present in this folder
datasets = 'datasets'


#There are also other sub_datasets
#I've used my name here "iyed" 
#You can change the label here
sub_data = 'iyed'

path = os.path.join(datasets, sub_data)
if not os.path.isdir(path):
    os.mkdir(path)
    
    
#Defining the size of the image
(width, height) = (100, 130)


# '0' is used for my personal webcam
# If you've another webcam
# Just attach the number '1'
face_cascade = cv2.CascadeClassifier(haad_file)
webcam = cv2.VideoCapture(0)


#The program loops until it collects 30 images of faces
count = 1
while count < 10:
    (_, im) = webcam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 4)
    
    
    for(x, y, w, h) in faces:
        cv2.rectangle(im, (x, y), (x+w, y+h), (255, 0, 0), 2)
        face = gray[y: y+h, x: x+w]
        face_resize = cv2.resize(face, (width, height))
        cv2.imwrite('% s/% s.png' % (path, count), face_resize) 
    count+= 1
    
    cv2.imshow('Face Detection', im)
    key = cv2.waitKey(10)
    if key == 'q':
        break
    