import cv2
import numpy as np
import sqlite3

def insertOrUpdate(id,name,age,gender):
    conn=sqlite3.connect("facerec.db");
    query="SELECT * FROM People WHERE id="+str(id);
    cursor=conn.execute(query)
    clone=0
    for row in cursor:
        clone=1
    if(clone==1):
        query="UPDATE People SET name="+str(name)+",age="+str(age)+",gender="+str(gender)+" WHERE id="+str(id);
    else:
        query="INSERT INTO People(id,name,age,gender) VALUES ("+str(id)+","+str(name)+","+str(age)+","+str(gender)+")";
    conn.execute(query)
    conn.commit()
    conn.close()
    
faceDetect=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam=cv2.VideoCapture(0)
id=raw_input('Enter an id:')
name=raw_input("Enter your name:")
age=raw_input('Enter your age:')
gender=raw_input("Enter your gender:")
sampnum=0
insertOrUpdate(id,name,age,gender)

while(True):
    ret,img=cam.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=faceDetect.detectMultiScale(gray,1.3,5);
    for(x,y,w,h) in faces:
        sampnum=sampnum+1
        cv2.imwrite("datasheet/User."+str(id)+"."+str(sampnum)+".jpg",gray[y:y+h,x:x+w])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.waitKey(100);
    cv2.imshow("Face",img)
    cv2.waitKey(1)
    if(sampnum>200):
        break;
cam.release();
cv2.destroyAllWindows()
