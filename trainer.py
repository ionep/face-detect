import os
import cv2
import numpy as np
from PIL import Image

recognizer=cv2.createLBPHFaceRecognizer()
path="datasheet"
faces=[]
Ids=[]

def getImagesWithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    Faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L');
        faceNp=np.array(faceImg,'uint8')
        ID=int(os.path.split(imagePath)[-1].split('.')[1])
        Faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Training",faceNp)
        cv2.waitKey(10)
    return np.array(IDs),Faces

Ids,faces=getImagesWithId(path)
recognizer.train(faces,Ids)
recognizer.save("recognizer/trainingdata.yml")
cv2.destroyAllWindows()
