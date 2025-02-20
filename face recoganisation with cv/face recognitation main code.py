import cv2
import numpy as np
import os

haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'datasets'
print('training....')
(images,lables,names,id) = ( [],[],{},0)

for (subdirs,dirs,files) in os.walk(dataset):
    for subdir in dirs :
        names[id] = subdir
        subjectpath = os.path.join(dataset,subdir)
        for filename in os.listdir(subjectpath):
            path = subjectpath + '/' + filename
            lable = id
            images.append(cv2.imread(path,0))
            lables.append(int(lable))
        id += 1
(width,height) = (130,100)

(images,lables) = [np.array(lis) for lis in [images,lables]]

model = cv2.face.FisherFaceRecognizer_create()
model = cv2.face.LBPHFaceRecognizer_create()

model.train(images,lables)

face_cascade = cv2.CascadeClassifier(haar_file)

cam = cv2.VideoCapture(0)

cnt = 0

while True :
    _,img = cam.read()
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.4,5)

    for (x,y,w,h) in faces:
        face = gray[y:y+h,x:x+w]
        face_resizes = cv2.resize(face,(width,height))

        prediction = model.predict(face_resizes)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),4)

        if prediction[1]<1500:
            cv2.putText(img,'%s - %.0f' % (names[prediction[0]],prediction[1]),(x-10,y-10),cv2.FONT_HERSHEY_COMPLEX,1,(51,250,250))
            print(names[prediction[0]])
            cnt=0
        else:
            cnt +=1
            cv2.putText(img, 'unknown', (x-10, y-10), cv2.FONT_HERSHEY_COMPLEX, 1, (100, 200, 200), 2)
            if(cnt>100):
                print('unknown image')
                cv2.imwrite('inputimg.png',img)
                cnt=0
    cv2.imshow('opencam',img)
    key = cv2.waitKey(10)
    if key == ord('p'):
        break

cam.release()
cv2.destroyAllWindows()
        
        
