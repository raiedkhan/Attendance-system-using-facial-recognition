import cv2,os
import shutil
import csv
import numpy as np
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter import messagebox
import pandas as pd
import datetime
import time
import pymysql

window =Tk()
window.geometry('600x600')
window.resizable(width=False, height=False)
window.title("My Attendance Portal")
window.configure(background='#D0D3D4')

image=PIL.Image.open("Images/logo.png")
photo=PIL.ImageTk.PhotoImage(image)
lab=Label(image=photo,bg='#D0D3D4')
lab.pack()

fn=StringVar()
ln=StringVar()
dn=StringVar()
v=StringVar()

label2=Label(window,text="New User",fg='#717D7E',bg='#D0D3D4',font=("roboto",20,"bold")).place(x=20,y=200)

label3=Label(window,text="Enter Name :",fg='black',bg='#D0D3D4',font=("roboto",15)).place(x=20,y=250)

label4=Label(window,text="Enter Roll Number :",fg='black',bg='#D0D3D4',font=("roboto",15)).place(x=275,y=252)

label5=Label(window,text="Note : To exit the frame window press 'q'",fg='red',bg='#D0D3D4',font=("roboto",15)).place(x=20,y=100)

status=Label(window,textvariable=v,fg='red',bg='#D0D3D4',font=("roboto",15,"italic")).place(x=20,y=150)

label6=Label(window,text="Already a User ?",fg='#717D7E',bg='#D0D3D4',font=("roboto",20,"bold")).place(x=20,y=350)

label7=Label(window,text="Delete a users information",fg='#717D7E',bg='#D0D3D4',font=("roboto",20,"bold")).place(x=20,y=450)

label8=Label(window,text="Enter Id :",fg='black',bg='#D0D3D4',font=("roboto",15)).place(x=20,y=500)

entry_name=Entry(window,textvar=fn)
entry_name.place(x=150,y=257)

entry_id=Entry(window,textvar=ln)
entry_id.place(x=455,y=257)

entry_name_del=Entry(window,textvar=dn)
entry_name_del.place(x=150,y=507)


def insert_user():
    Id=ln.get()
    name=fn.get()
    if(Id.isnumeric() and name.isalpha()):
        df=pd.read_csv("StudentDetails\StudentDetails.csv")
        if(df['Id'].astype(str).str.contains(str(Id)).any()==True):
            v.set("User with same Roll No. Already exists")
            messagebox.showinfo('My Attendance Portal','User with same Roll No. Already exists')
        else:
            cam = cv2.VideoCapture(0)
            harcascadePath = "haarcascade_frontalface_default.xml"
            detector=cv2.CascadeClassifier(harcascadePath)
            sampleNum=0
            while(True):
                ret, img = cam.read()
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                    #Incrementing sample number 
                    sampleNum=sampleNum+1
                    #Saving the captured face in the dataset folder TrainingImage
                    cv2.imwrite("TrainingImage\ "+name.lower() +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                    #Display the frame
                    cv2.imshow('frame',img)
                #wait for 100 miliseconds 
                if cv2.waitKey(100) & 0xFF == ord('q'):
                    break
                # Break if the sample number is morethan 100
                elif sampleNum>60:
                    break
            cam.release()
            cv2.destroyAllWindows() 
            row = [Id , name]
            with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
                writer = csv.writer(csvFile)
                writer.writerow(row)
            csvFile.close()
            name_saved=" ID : "+str(Id)+ " with NAME : "+ name +" Saved"
            v.set(name_saved)
    else:
        if(Id.isnumeric()==False):
            v.set("Please enter Correct format of ROll No.")
            messagebox.showinfo('My Attendance Portal','Please enter Correct format of ROll No.')
        if(name.isalpha()==False):
            messagebox.showinfo('My Attendance Portal','Please enter Correct format of USERNAME')
            v.set("Please enter Correct format of USERNAME")

#Train images #

def train_image():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector =cv2.CascadeClassifier(harcascadePath)
    faces,Id = ImagesAndNames("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("recognizers/Trainner.yml")
    v.set("Images Trained")
    messagebox.showinfo('My Attendance Portal','Images Trained Succesfully')

def ImagesAndNames(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    #create empty face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #Loading the images in Training images and converting it to gray scale
        g_image=PIL.Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        image_ar=np.array(g_image,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(image_ar)
        Ids.append(Id)        
    return faces,Ids

def track_user():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("recognizers/Trainner.yml")
    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector=cv2.CascadeClassifier(harcascadePath)
    font=cv2.FONT_HERSHEY_SIMPLEX
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    col_names =  ['ID','Date','Time']
    attendance = pd.DataFrame(columns = col_names)  
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
            print("Confidence is "+str(conf)+"Id is "+str(Id))
            name=df.loc[df['Id'] == Id]['Name'].values
            name_get=str(Id)+"-"+name
            time_s = time.time()      
            date = str(datetime.datetime.fromtimestamp(time_s).strftime('%Y-%m-%d'))
            timeStamp = datetime.datetime.fromtimestamp(time_s).strftime('%H:%M:%S')
            attendance.loc[len(attendance)] = [Id,date,timeStamp]
            if(conf>75):
                noOfFile=len(os.listdir("ImagesUnknown"))+1
                cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", img[y:y+h,x:x+w])
                Id='Unknown'
                name_get=str(Id)
            cv2.putText(img,str(name_get),(x+w,y+h),font,0.5,(0,255,255),2,cv2.LINE_AA)
            attendance=attendance.drop_duplicates(keep='first',subset=['ID'])
            fileName="Attendance/in.json"
            attendance.to_json(fileName,orient="index")
        cv2.imshow('img',img)
        if (cv2.waitKey(1)==ord('q')):
            MsgBox = messagebox.askquestion ('My Attendance Portal','Are you sure you want to exit the application',icon = 'warning')
            if MsgBox == 'yes':
                break
    cam.release()
    cv2.destroyAllWindows()
    v.set("Images tracked ")

def update_att():
    conn=pymysql.connect(host="localhost",user="root",passwd="",db="map")
    myCursor=conn.cursor()
    myCursor.execute("SELECT * FROM attendance;")
    records=myCursor.fetchall()
    length_db=len(records)
    if(length_db==0):
        df=pd.read_json("Attendance/classtest.json")
        length_df=len(df.columns)
        for i in range (length_df):
            id=(df[i].ID).item()
            date=(df[i].Date).item()
            time=(df[i].Time).item()
            myCursor.execute(""" INSERT INTO attendance(id,date1,time1,att,totclass) VALUES (%s,%s,%s,%s,%s)""",(id,date,time,0,0))
        v.set("Attendance Inserted for the first time")
    else:
        df=pd.read_json("Attendance/in.json")
        length_df=len(df.columns)
        date_json=(df[0].Date)
        check=0
        for row in records:
            date_db=row[1]
            if(date_db==date_json):
                check=1
                break
            else:
                check=0
        if(check==1):
            v.set("Total classes didn't updated")
        else:
            myCursor.execute("UPDATE attendance SET totclass=totclass+1")
        for i in range(length_df):
            id_json=(df[i].ID)
            date_json=(df[i].Date)
            time_json=(df[i].Time)
            date=str(datetime.date.today())
            for row in records:
                id_db=row[0]
                date_db=row[1]
                if(id_json==id_db and date_db!=date_json):
                    sql=" UPDATE attendance SET date1=%s,time1=%s,att=att+1 WHERE id=%s"
                    val=(date_json,time_json,id_json)
                    myCursor.execute(sql,val)
                    v.set("Attendance Updated")
    conn.commit()
    conn.close()

def del_user():
    roll_del=int(dn.get())
    src="TrainingImage"
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    for roll in df['Id']:
        if(roll==roll_del):
            for image_file_name in os.listdir(src):
                roll_str=str(roll)
                if(roll_str in image_file_name):
                    v.set("Deleting the Given user names info...")
                    os.remove(src+"/"+image_file_name)
                    df.drop(df.loc[df['Id']==roll_del].index, inplace=True)
                    df.to_csv("StudentDetails\StudentDetails.csv", index=False, encoding='utf8')
                    v.set(roll_str+" Deleted From Database")
        else:
            v.set("User with given roll number not present")

def ExitApplication():
    MsgBox = messagebox.askquestion ('My Attendance Portal','Are you sure you want to exit the application',icon = 'warning')
    if MsgBox == 'yes':
       window.destroy()
    else:
        messagebox.showinfo('My Attendance Portal','You will now return to the application screen')

def update():
    MsgBox = messagebox.askquestion ('My Attendance Portal','Are you sure you want to update the attendance',icon = 'warning')
    if MsgBox == 'yes':
       update_att()
    else:
        messagebox.showinfo('My Attendance Portal','Attendance has been succesfully updated')

def delete():
    MsgBox = messagebox.askquestion ('My Attendance Portal','Are you sure you want to delete Users Information',icon = 'warning')
    if MsgBox == 'yes':
       del_user()
    else:
        messagebox.showinfo('My Attendance Portal','User information is not deleted')

submitb=PhotoImage(file = r"Images\upload.png")
submitimage = submitb.subsample(3, 3)

trainb=PhotoImage(file = r"Images\train.png")
trainimage = trainb.subsample(2, 2)

trackb=PhotoImage(file = r"Images\face.png")
trackimage = trackb.subsample(1, 1)

updateb=PhotoImage(file = r"Images\update.png")
updateimage = updateb.subsample(2, 2)

deleteb=PhotoImage(file = r"Images\delete.png")
deleteimage = deleteb.subsample(2, 2)

exitb=PhotoImage(file = r"Images\exit.png")
exitbimage = exitb.subsample(2, 2)

#BUTTONS PRESENT HERE

button1=Button(window,text="Exit",image=exitbimage,compound=LEFT,width=70,fg='#fff',bg='red',relief=RAISED,font=("roboto",15,"bold"),command=ExitApplication)
button1.place(x=500,y=550)

button2=Button(window,text="Submit",image=submitimage,compound=LEFT,fg='#fff',bg='#27AE60',relief=RAISED,font=("roboto",15,"bold"),command=insert_user)
button2.place(x=20,y=300)

button3=Button(window,text="Train Images",image=trainimage,compound=LEFT,fg='#fff',bg='#5DADE2',relief=RAISED,font=("roboto",15,"bold"),command=train_image)
button3.place(x=130,y=300)

button4=Button(window,text="Track User",image=trackimage,compound=LEFT,fg='#fff',bg='#2ECC71',relief=RAISED,font=("roboto",15,"bold"),command=track_user)
button4.place(x=20,y=400)

button4=Button(window,text="Update Attendance",image=updateimage,compound=LEFT,fg='#fff',bg='#3498DB',relief=RAISED,font=("roboto",15,"bold"),command=update)
button4.place(x=180,y=400)

button6=Button(window,text="Delete User",image=deleteimage,compound=LEFT,fg='#fff',bg='red',relief=RAISED,font=("roboto",15,"bold"),command=delete)
button6.place(x=20,y=550)

window.mainloop()