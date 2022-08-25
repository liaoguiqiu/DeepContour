# this file is modified ffom the correc ruller which will read the image and estimatd the contour and the peak value
#

# import rospy
# from std_msgs.msg import String

import cv2
import numpy as np
import scipy.signal as signal
# import pandas as pd
from deploy.DeepAutoJson_oct import Auto_json_label
import matplotlib
import time

from scipy.ndimage import gaussian_filter1d
from dataset_sheath import Resample_size
import zmq  # Client Program
from  stabilization2.De_NURD_with_generator_deep_needle.De_NURD.Correct_sequence_integral import OCTStab

# stabilization 2
# import packs about hte OCT stream and socket
Stabilization_flag = True
matplotlib.use('TkAgg')
# rospy.init_node('talker', anonymous=True)
# pub = rospy.Publisher('chatter', String, queue_size=10)
#
#
# def talker():
#     pub = rospy.Publisher('chatter', String, queue_size=10)
#     rospy.init_node('talker', anonymous=True)
#     rate = rospy.Rate(10) # 10hz
#     while not rospy.is_shutdown():
#         hello_str = "hello world %s" % rospy.get_time()
#         rospy.loginfo(hello_str)
#         pub.publish(hello_str)
#         rate.sleep()
# def talker2():
#
#
#     hello_str = "hello world %s" % rospy.get_time()
#     rospy.loginfo(hello_str)
#     pub.publish(hello_str)

def draw_coordinates_color(img1, vy, color):
    if color == 0:
        painter = [254, 0, 0]
    elif color == 1:
        painter = [0, 254, 0]
    elif color == 2:
        painter = [0, 0, 254]
    else:
        painter = [0, 0, 0]
        # path0  = signal.resample(path0, W)
    H, W, _ = img1.shape
    for j in range(W):
        # path0l[path0x[j]]
        dy = np.clip(vy[j], 2, H - 2)

        img1[int(dy) + 1, j, :] = img1[int(dy), j, :] = painter

        # img1[int(dy)+1,dx,:]=img1[int(dy)-1,dx,:]=img1[int(dy),dx,:]=painter

    return img1


class Find_shadow(object):
    def __init__(self):
        self.folder_num = 0
        self.database_root = " "
        # self.save_root  = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairB/with ruler/correct2/"
        # self.save_root  = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2_correct/"
        self.save_root = "/home/icube/OCT_projects/Contour_project/pairD/ref_correct/"
        # self.save_root  = "../pairD/ref_correct/"

        # self.database_root = "D:/Deep learning/dataset/original/animal_tissue/1/"
        # self.database_root = "D:/Deep learning/dataset/original/IVOCT/1/"
        self.auto_label = Auto_json_label()

        self.f_downsample_factor = 93
        # self.all_dir = "D:/PhD/trying/tradition_method/OCT/sheath registration/pairC/ruler/2_/"
        self.all_dir = "/home/icube/OCT_projects/Contour_project/pairD/ref_/"
        # self.all_dir = "../pairD/ref_/"

        self.image_dir = self.database_root + "pic/"
        self.json_dir = self.database_root + "label/"  # for this class sthis dir ist save the modified json
        self.json_save_dir = self.database_root + "label_generate/"
        self.img_num = 0
        self.last_est = 0
        # stablization nets
        self.stabnet = OCTStab( )


        # socket
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.port = "5555"
        self.socket.connect("tcp://192.168.12.104:%s" % self.port)
        # print("socket state :" + str(socket.connected() ) )

    def find_the_peak(self, c2, gray):  # crop the ROI based on the contour
        shift = 10
        rate0 = 0.1
        rate = 0.3
        y1 = np.array(c2)  # transfer list to array
        y = y1[:, 1]
        y = signal.resample(y, Resample_size)
        y = gaussian_filter1d(y, 5)  # smooth the path

        max_idex = np.argmin(y)
        y = y.astype(int)

        return max_idex, y

    def deal_pic(self, img, num,Gray=0):
        ref_position = 252  # this is calculated with img 5, so all img will be shift to this posion



        original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H_i, W_i = original.shape
        img = cv2.resize (img, (Resample_size, Resample_size))
        # cv2.imshow('real', img)

        # img  = cv2.resize(img, ( Resample_size,Resample_size), interpolation=cv2.INTER_AREA)
        # cv2.waitKey(10)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        H, W = gray.shape
        coordinates_all,_ = self.auto_label.predict_contour(gray, Resample_size, Resample_size, points=300)
        coordinates1 = coordinates_all[0]
        coordinates2 = coordinates_all[1]

        max_idex, y = self.find_the_peak(coordinates2, gray)
        max_idex1, y1 = self.find_the_peak(coordinates1, gray)


        draw_coordinates_color(img, y, 1)
        draw_coordinates_color(img, y1, 0)

        shift = (ref_position - max_idex) * W_i / Resample_size
        # shift = 0.3*shift + 0.7*self.last_est
        self.last_est = shift

        New = np.roll(original, int(shift), axis=1)
        # save this to new folder
        # cv2.imwrite(self.save_root + str(num) + ".jpg", New)

        # cv2.imshow('seg', img)
        #
        # cv2.waitKey(2)

        pass
        return y1,y, img

    def deal_folder(self):
        # for i in os.listdir(self.image_dir): # star from the image folder
        # for i in os.listdir(self.all_dir): # star from the image folder
        for a in range(166, 894):
            # for i in os.listdir("E:\\estimagine\\vs_project\\PythonApplication_data_au\\pic\\"):
            # separath  the name of json
            # a, b = os.path.splitext(i)
            # if it is a img it will have corresponding image
            # if b == ".jpg" :
            img_path = self.all_dir + str(a) + ".jpg"
            # jason_path  = self.json_dir + a + ".json"
            img1 = cv2.imread(img_path)

            if img1 is None:
                print("no_img")
            else:
                y1,y2, img = self.deal_pic(img1, a)


        return 0
    def OCT_stream2(self):

        self.socket.send_string("Hello")
        print("send hello ")
        message = self.socket.recv()  # receive the data from the sever C++
        print("receive ")
        ReceivedImage = message
        ReceivedImage = np.fromstring(message, dtype=np.uint8)
        img_decode = cv2.imdecode(ReceivedImage, flags=cv2.IMREAD_UNCHANGED)
        frame = img_decode
        # frame = ReceivedImage

        return frame
    # def OCT_stream(self):
    #     context = zmq.Context()
    #     socket = context.socket(zmq.REQ)
    #     port = "5555"
    #     socket.connect("tcp://192.168.12.83:%s" % port)
    #     # print("socket state :" + str(socket.connected() ) )
    #     socket.send_string("Hello")
    #     print("send hello ")
    #     message = socket.recv()  # receive the data from the sever C++
    #     print("receive ")
    #     ReceivedImage = message
    #     ReceivedImage = np.fromstring(message, dtype=np.uint8)
    #     img_decode = cv2.imdecode(ReceivedImage, flags=cv2.IMREAD_UNCHANGED)
    #     frame = img_decode
    #     return frame
    # thsi is adapted from the socket read y
    def deal_stream(self):
        last_time = 0
        this_time = 0
        while(1):
            frame = self.OCT_stream2()
            # cv2.imshow("Input", frame)
            this_time   = time.time()
            print("update time" + str(this_time - last_time))
            last_time   = this_time
            if frame is None:
                print("no_img")
            else:
                process_start = time.time()

                frame = cv2.resize(frame, (256, 256))

                frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                if Stabilization_flag == True:
                    frame = self.stabnet.correct(frame)
                    frame = cv2.cvtColor(frame.astype(np.uint8), cv2.COLOR_GRAY2RGB)
                process_end= time.time()

                y1,y2, img = self.deal_pic(frame, 0)
                cv2.imshow('seg', img)

                cv2.waitKey(10)

            print("PROCESS TIME" + str(process_end - process_start))
            # talker2()

            # cv2.waitKey(10)
            pass



if __name__ == '__main__':
    cheker = Find_shadow()
    img_path = cheker.all_dir + "166" + ".jpg"
    img1 = cv2.imread(img_path)

    # jason_path  = self.json_dir + a + ".json"
    cheker.deal_pic(img1, 0)
    cheker.deal_stream()
    # img1 = cv2.imread(img_path)
    # cheker.deal_folder()




