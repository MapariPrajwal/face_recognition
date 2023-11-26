import cv2
import os
import face_recognition as FR
import pickle

kimageDIR='C://Users/prajw/Python/DemoImage/known'
knownEncodings=[]
names=[]
font=cv2.FONT_HERSHEY_SIMPLEX
for root,dirs,files in os.walk(kimageDIR):
    for file in files:
        kimgPath=os.path.join(root,file)
        kimgName=os.path.splitext(file)[0]
        
        face=FR.load_image_file(kimgPath)
        faceLoc=FR.face_locations(face)
        faceEncode=FR.face_encodings(face)[0]

        knownEncodings.append(faceEncode)
        names.append(kimgName)
    print(names)

with open('train.pkl','wb') as f:
    pickle.dump(names,f)
    pickle.dump(knownEncodings,f)