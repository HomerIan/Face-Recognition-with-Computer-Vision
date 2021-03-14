import cv2, imutils, numpy, os, requests
#face detection algorithm
haar_file = 'haarcascade_frontalface_default.xml'
datasets = 'datasets'
(images, labels, names, id) = ([], [], {}, 0)

#locate datasets folder
for(subdirs, dirs, files) in os.walk(datasets):
    #folder and subfolder
    for subdir in dirs:
        #iteration to get every subfolder
        names[id] = subdir
        #creating subject path of every subfolder in folder
        subjectpath = os.path.join(datasets, subdir)
        #iteration read the files in subfolder
        for filename in os.listdir(subjectpath):
            #file path 
            path = subjectpath + '/' + filename
            label = id
            #taking each images
            images.append(cv2.imread(path, 0))
            labels.append(int(label))
            #print(labels)
        id+=1
#size for crop image
(width, height) = (140, 110)

(images, labels) = [numpy.array(lis) for lis in [images, labels]]
#print(images, labels)

#load the algorithm
model = cv2.face.LBPHFaceRecognizer_create()
#model  = cv2.face.FisherFaceRecognizer_create()

model.train(images, labels)
print("training completed")

#face detection
face_cascade = cv2.CascadeClassifier(haar_file)
cam = cv2.VideoCapture(1)
'''
address = ''
cam.open(address)
'''
cnt=0
while True:
    _,frame = cam.read()
    frame = imutils.resize(frame, width = 600)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayFrame, 1.5, 4)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        face = grayFrame[y:y+h, x:x+w]
        face_resize = cv2.resize(face, (width, height))

        #prediction
        #value(names, accuracy)
        prediction = model.predict(face_resize)
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        #check prediction < accuracy
        if prediction[1] < 90:
            cv2.putText(frame, '%s - %.0f' % (names[prediction[0]],
                                                    prediction[1]),
                                                    (x-10, y-10),
                                                    cv2.FONT_HERSHEY_PLAIN,
                                                    3,
                                                    (0,255,0),2)
            print(names[prediction[0]])
            cnt=0
        #if unknown person
        else:
            cnt+=1
            cv2.putText(frame, 'Unknown', (x-10, y-10),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,
                        (0,255,0),2)
            #check if it still unkown till 100 times capture image
            if(cnt>100):
                print("Unknown Person")
                cv2.imwrite("unkown.jpg", frame)
                cnt=0
                
    cv2.imshow("Face Recognition", frame)
    key = cv2.waitKey(0)
    if key == 27:
        break;
#cam.release()
cv2.destroyAllWindows()


