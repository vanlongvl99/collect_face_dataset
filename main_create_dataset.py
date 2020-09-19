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
from datetime import datetime


num_of_images = 20




#### Make GUI ####
window = tk.Tk()
window.title("Face_Recogniser")
window.configure(background ='white')
window.grid_rowconfigure(0, weight = 1)
window.grid_columnconfigure(0, weight = 1)
message = tk.Label(
    window, text ="Create dataset system",
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
    # print("start take image")
    try:
        # print("name", name)
        # print("make forder")
        os.mkdir("./dataset/" + name)
    except:
        print("pass")
        pass
    fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
    videoWriter = cv2.VideoWriter("dataset/" + name + "/" + "video.avi", fourcc, 30.0, (640,480))
    
    cam = cv2.VideoCapture(2)
    count = 0
    flag = -1
    while(True):
        # print(datetime.now())
        ret, frame = cam.read()
        # print(frame)
        # frame_resize = cv2.resize(frame,(480,int(frame.shape[0]/frame.shape[1]*480)))
        cv2.imshow('frame_resize', frame)
        videoWriter.write(frame)
        key = cv2.waitKey(20)
        # nhấn phím 1 nhìn thẳng
        if key  == ord('1'):
            flag = 1
        # nhấn phím 2 nhìn lên
        elif key  == ord('2'):
            flag = 2
        # nhấn phím 3 nhìn xuống
        elif key  == ord('3'):
            flag = 3
        # nhấn phím 4 nhìn trái
        elif key  == ord('4'):
            flag = 4
        # nhấn phím 5 nhìn phải
        elif key  == ord('5'):
            flag = 5
        # cười
        elif key  == ord('6'):
            flag = 6
        # Xoay vong tron
        elif key  == ord('7'):
            flag = 7
        # Nhắm mắt trái
        elif key  == ord('8'):
            flag = 8
        #nhắm mắt phải
        elif key  == ord('9'):
            flag = 9
    
    
    
        # break if the sample number is more than 20

        if flag == 1:
            cv2.imwrite("./dataset/" + name + "/" + "nhin_thang_" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nhin_thang_" + str(count) + ".jpg")
        elif flag == 2:
            cv2.imwrite("./dataset/" + name + "/" + "nhin_len_" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nhin_len_" + str(count) + ".jpg")

        elif flag == 3:
            cv2.imwrite("./dataset/" + name + "/" + "nhin_xuong_" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nhin_xuong_" + str(count) + ".jpg")
        elif flag == 4:
            cv2.imwrite("./dataset/" + name + "/" + "nhin_trai_" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nhin_trai_" + str(count) + ".jpg")
        elif flag == 5:
            cv2.imwrite("./dataset/" + name + "/" + "nhin_phai_" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nhin_phai_" + str(count) + ".jpg")
        elif flag == 6:
            cv2.imwrite("./dataset/" + name + "/" + "cuoi_" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "cuoi_" + str(count) + ".jpg")
        elif flag == 7:
            cv2.imwrite("./dataset/" + name + "/" + "xoay_vong_tron" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "xoay_vong_tron" + str(count) + ".jpg")
        elif flag == 8:
            cv2.imwrite("./dataset/" + name + "/" + "nham_mat_trai" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nham_mat_trai" + str(count) + ".jpg")
        elif flag == 9:
            cv2.imwrite("./dataset/" + name + "/" + "nham_mat_phai" + str(count) + ".jpg", frame)
            count += 1
            print("./dataset/" + name + "/" + "nham_mat_phai" + str(count) + ".jpg")


        if count > num_of_images:
            flag = -1
            count = 0
        if key  == ord('q'):
            break

        # print(datetime.now())

    cam.release()
    videoWriter.release()
    cv2.destroyAllWindows()





#Take image from camera nhìn thẳng
takeImg = tk.Button(window, text ="Take images",      
command = TakeImages, fg ="white", bg ="green",
# kích thước của các button  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
# Tọa độ của các button
takeImg.place(x = 150, y = 400) 


# Quit GUI
quitWindow = tk.Button(window, text ="Quit",  
command = window.destroy, fg ="white", bg ="green",  
width = 10, height = 3, activebackground = "Red",  
font =('times', 15, ' bold ')) 
quitWindow.place(x = 1050, y = 400) 
window.mainloop()