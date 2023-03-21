import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import firestore

cred = credentials.Certificate('cognida-3a5de-firebase-adminsdk-uj31d-4abe3837e2.json')
app = firebase_admin.initialize_app(cred)
db = firestore.client()

processStart = False

path = './images'
teachersPath = './teachers'
images = []
teacherImages = []
personNames = []
teacherNames = []
myList = os.listdir(path)
teacherImagesPath = os.listdir(teachersPath)
print(myList)
for teacher in teacherImagesPath:
    current_Img = cv2.imread(f'{teachersPath}/{teacher}')
    teacherImages.append(current_Img)
    teacherNames.append(os.path.splitext(teacher)[0])
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personNames.append(os.path.splitext(cu_img)[0])
print(personNames)
print(teacherNames)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


a = '\downloads\Face_Recognition_Project-main\Attendance.csv'
print('Hello ',os.getcwd() + a)
def Attendance(name):
    doc_ref = db.collection('attendance').document(name)
    getData = doc_ref.get()
    time_now = datetime.now()
    tStr = time_now.strftime('%H:%M:%S')
    dStr = time_now.strftime('%d/%m/%Y')
    if getData.exists:
        temp = getData.to_dict()
        tempDate = temp["date"]
        tempTime = temp["time"]
        if tempDate[len(tempDate) - 1] != dStr:
            tempDate.insert(len(tempDate), dStr)
            tempTime.insert(len(tempTime), tStr)
            print(tempDate)
            doc_ref.update({
                'date': tempDate,
                'time': tempTime,
                # 'born': tempDate
            })
    else:
        tempDate = []
        tempTime = []
        tempDate.insert(len(tempDate), dStr)
        tempTime.insert(len(tempTime), tStr)
        doc_ref.set({
            'date': tempDate,
            'time': tempTime,
        })


    # with open(os.getcwd() + a, 'r+') as f:
    #     myDataList = f.readlines()
    #     nameList = []
    #     for line in myDataList:
    #         entry = line.split(',')
    #         nameList.append(entry[0])
    #     if name not in nameList:
    #         time_now = datetime.now()
    #         tStr = time_now.strftime('%H:%M:%S')
    #         dStr = time_now.strftime('%d/%m/%Y')
    #         f.writelines(f'\n{name},{tStr},{dStr}')


encodeListKnown = faceEncodings(images)
teacherEncodingList = faceEncodings(teacherImages)
print('All Encodings Complete!!!')

cap = cv2.VideoCapture(1)

while True:
    ret, frame = cap.read()
    # print(frame)
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25) 
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        teacherMatches = face_recognition.compare_faces(teacherEncodingList, encodeFace)
        teacherFaceDis = face_recognition.face_distance(teacherEncodingList, encodeFace)
        print(faceDis)
        matchIndex = np.argmin(faceDis) 
        teacherMatchIndex = np.argmin(teacherFaceDis)
           
    # ret, frame = cap.read()
    # print(frame)
    # faces = cv2.resize(frame,(0,0),None,0.25,0.25)
    # faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)

    # facesCurrentFrame = face_recognition.face_locations(faces)
    # encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)

    # for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
    #     matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
    #     faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
    #     print(faceDis)
    #     matchIndex = np.argmin(faceDis)
        if processStart:
            if matches[matchIndex]:
                name = personNames[matchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                Attendance(name)
            else:
                name = 'Unkown'
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        else:
            if teacherMatches[teacherMatchIndex]:
                name = teacherNames[teacherMatchIndex].upper()
                print(name)
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                processStart = True
                print('Process start')
            else:
                name = 'Unkown'
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)


    cv2.imshow('Webcam', frame)
    if cv2.waitKey(1) == 13:
        break

cap.release()
cv2.destroyAllWindows()

