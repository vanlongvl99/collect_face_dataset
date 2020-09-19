import tkinter as tk
from tkinter import Message, Text, messagebox
import cv2
import os
import time
import numpy as np
from sklearn.svm import SVC
# import joblib
from numpy import expand_dims
from sklearn.preprocessing import Normalizer
import datetime
from numpy import load
from os import listdir
import os.path


num_of_images = 10

#### Make GUI ####
window = tk.Tk()
window.title("Face_Recogniser")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Face-Recognition-System-mtcnn-faiss",
    bg ="green", fg = "white", width = 50,
    height = 3, font = ('times', 30, 'bold'))
message.place(x = 200, y = 20)
lbl2 = tk.Label(window, text ="Name of new person",
width = 20, fg ="green", bg ="white",
height = 2, font =('times', 15, ' bold '))
lbl2.place(x = 400, y = 200)
txt2 = tk.Entry(window, width = 20,
bg ="white", fg ="green",
font = ('times', 15, ' bold '))
txt2.place(x = 700, y = 215)
############         #################



# Take train image of new user nhìn thẳng.
def TakeImages():
    goc_nhin = "nhin_thang"
    name =(txt2.get())

    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    print("start take image")
    try:
        print("name", name)
        print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video_nhin_thang.avi", fourcc, 30.0, (640,480))
    
    cam = cv2.VideoCapture(0)
    count = 0
    while(True):
        ret, frame = cam.read()
        frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        frame_resize = cv2.flip(frame_resize, 1)
        cv2.imshow('frame_resize', frame_resize)
        cv2.imwrite("./dataset/" + name + "/" + goc_nhin + "_" + str(count) + ".jpg", frame)
        videoWriter.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 150
        elif count == num_of_images:
            break
        count += 1
    cam.release()
    videoWriter.release()

    cv2.destroyAllWindows()


# Take train image of new user look up.
def TakeImages_look_up():
    goc_nhin = "look_up"
    print("start take image")
    name =(txt2.get())
    try:
        print("name", name)
        print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video_nhin_thang.avi", fourcc, 30.0, (640,480))

    count = 0
    while(True):
        ret, frame = cam.read()
        frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        frame_resize = cv2.flip(frame_resize, 1)
        cv2.imshow('frame_resize', frame_resize)
        cv2.imwrite("./dataset/" + name + "/" + goc_nhin + "_" + str(count) + ".jpg", frame)
        videoWriter.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 150
        elif count == num_of_images:
            break
        count += 1

    cam.release()
    videoWriter.release()

    cv2.destroyAllWindows()

# Take train image of new user look_down.
def TakeImages_look_down():
    goc_nhin = "look_down"
    print("start take image")
    name =(txt2.get())
    try:
        print("name", name)
        print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video_nhin_thang.avi", fourcc, 30.0, (640,480))

    count = 0
    while(True):
        ret, frame = cam.read()
        frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        frame_resize = cv2.flip(frame_resize, 1)
        cv2.imshow('frame_resize', frame_resize)
        cv2.imwrite("./dataset/" + name + "/" + goc_nhin + "_" + str(count) + ".jpg", frame)
        videoWriter.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 150
        elif count == num_of_images:
            break
        count += 1

    cam.release()
    videoWriter.release()

    cv2.destroyAllWindows()



# Take train image of new user look_right.
def TakeImages_look_right():
    goc_nhin = "look_right"
    print("start take image")
    name =(txt2.get())
    try:
        print("name", name)
        print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video_nhin_thang.avi", fourcc, 30.0, (640,480))

    count = 0
    while(True):
        ret, frame = cam.read()
        frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        frame_resize = cv2.flip(frame_resize, 1)
        cv2.imshow('frame_resize', frame_resize)
        cv2.imwrite("./dataset/" + name + "/" + goc_nhin + "_" + str(count) + ".jpg", frame)
        videoWriter.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 150
        elif count == num_of_images:
            break
        count += 1

    cam.release()
    videoWriter.release()

    cv2.destroyAllWindows()
 
 # Take train image of new user look_left.
def TakeImages_look_left():
    goc_nhin = "look_left"
    print("start take image")
    name =(txt2.get())
    try:
        print("name", name)
        print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video_nhin_thang.avi", fourcc, 30.0, (640,480))

    count = 0
    while(True):
        ret, frame = cam.read()
        frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        frame_resize = cv2.flip(frame_resize, 1)
        cv2.imshow('frame_resize', frame_resize)
        cv2.imwrite("./dataset/" + name + "/" + goc_nhin + "_" + str(count) + ".jpg", frame)
        videoWriter.write(frame)

        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 150
        elif count == num_of_images:
            break
        count += 1

    cam.release()
    videoWriter.release()

    cv2.destroyAllWindows()

 # Take train image of new user smile.
def TakeImages_smile():
    goc_nhin = "smile"
    print("start take image")
    name =(txt2.get())
    try:
        print("name", name)
        print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    cam = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video_nhin_thang.avi", fourcc, 30.0, (640,480))

    count = 0
    while(True):
        ret, frame = cam.read()
        frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        frame_resize = cv2.flip(frame_resize, 1)
        cv2.imshow('frame_resize', frame_resize)
        cv2.imwrite("./dataset/" + name + "/" + goc_nhin + "_" + str(count) + ".jpg", frame)
        videoWriter.write(frame)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        # break if the sample number is more than 150
        elif count == num_of_images:
            break
        count += 1

    cam.release()
    videoWriter.release()

    cv2.destroyAllWindows()






#Take image from camera nhìn thẳng
takeImg = tk.Button(window, text ="Nhìn thẳng",      
command = TakeImages, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 150, y = 400) 

#Take image from camera nhìn len
takeImg = tk.Button(window, text ="Nhìn lên",      
command = TakeImages_look_up, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 300, y = 400) 


#Take image from camera nhìn len
takeImg = tk.Button(window, text ="Nhìn xuống",      
command = TakeImages_look_down, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 450, y = 400) 

#Take image from camera nhìn trai
takeImg = tk.Button(window, text ="Nhìn trái",      
command = TakeImages_look_left, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 600, y = 400) 

#Take image from camera nhìn phải
takeImg = tk.Button(window, text ="Nhìn phải",      
command = TakeImages_look_right, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 750, y = 400) 

#Take image from camera nhìn phải
takeImg = tk.Button(window, text ="Cười",      
command = TakeImages_smile, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 900, y = 400) 


# Quit GUI
quitWindow = tk.Button(window, text ="Quit",  
command = window.destroy, fg ="white", bg ="green",  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
quitWindow.place(x = 1050, y = 400) 
window.mainloop()