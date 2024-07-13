
from __future__ import print_function, division
import argparse
import os

import torch
print(torch.cuda.is_available())
import random
import torch.nn as nn
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import time
import sys
import math
import cv2
from PIL import Image
from models.gwcnet import GwcNet




import os
import glob
import torch
import cv2
import argparse
import util.io
from torchvision.transforms import Compose
from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet




from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from ui_demo import Ui_MainWindow
import cv2 as cv
import numpy as np
import sys


# ---------------------------------------------------------------------------



import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data
from datasets import __datasets__
import gc
import skimage
import skimage.io
import skimage.transform
from datasets.data_io import get_transform
from models.bgnet import BGNet
from models.bgnet_plus import BGNet_Plus

modelb = BGNet_Plus().cuda()
checkpoint = torch.load('models/Sceneflow-IRS-BGNet-Plus.pth',map_location=lambda storage, loc: storage)
modelb.load_state_dict(checkpoint) 
modelb.eval()








def calibration(frame1, frame2, w, h): 
    left_camera_matrix = np.array([[824.93564, 0., 251.64723],
                                [0., 825.93598, 286.58058],
                                [0., 0., 1.]])
    left_distortion = np.array([[0.23233, -0.99375, 0.00160, 0.00145, 0.00000]])

    right_camera_matrix = np.array([[853.66485, 0., 217.00856],
                                    [0., 852.95574, 269.37140],
                                    [0., 0., 1.]])
    right_distortion = np.array([[0.30829, -1.61541, 0.01495, -0.00758, 0.00000]])

    # 旋转关系向量
    om = np.array([0.01911, 0.03125, -0.00960])
    # 使用Rodrigues变换将om变换为R
    R = cv2.Rodrigues(om)[0]
    # 平移关系向量
    T = np.array([-70.59612, -2.60704, 18.87635])

    # size = (640, 480) # 图像尺寸
    size = (w, h) # 图像尺寸

    # 进行立体更正
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(left_camera_matrix, left_distortion, right_camera_matrix, right_distortion, size, R, T)

    # 计算更正map
    left_map1, left_map2 = cv2.initUndistortRectifyMap(left_camera_matrix, left_distortion, R1, P1, size, cv2.CV_16SC2)
    right_map1, right_map2 = cv2.initUndistortRectifyMap(right_camera_matrix, right_distortion, R2, P2, size, cv2.CV_16SC2)

    # 根据更正map对图片进行重构
    img1_rectified = cv2.remap(frame1, left_map1, left_map2, cv2.INTER_LINEAR)
    img2_rectified = cv2.remap(frame2, right_map1, right_map2, cv2.INTER_LINEAR)
    return img1_rectified, img2_rectified


def trans_color(gray):
    gray = gray.reshape(gray.shape[0], gray.shape[1], 1).astype('uint8')
    out = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    out = cv2.applyColorMap(out, cv2.COLORMAP_JET)
    return out


def mypred(img1, img2):
    left_img = Image.open(img1).convert('L')
    right_img = Image.open(img2).convert('L')
    w, h = left_img.size
    h1 = h % 64
    w1 = w % 64
    h1 = h  - h1
    w1 =  w - w1
    h1 = int(h1)
    w1 = int(w1)
    left_img = left_img.resize((w1, h1),Image.ANTIALIAS)
    right_img = right_img.resize((w1, h1),Image.ANTIALIAS)
    left_img = np.ascontiguousarray(left_img, dtype=np.float32)
    right_img = np.ascontiguousarray(right_img, dtype=np.float32)
    preprocess = get_transform()
    left_img = preprocess(left_img)
    right_img = preprocess(right_img)
    pred,_ = modelb(left_img.unsqueeze(0).cuda(), right_img.unsqueeze(0).cuda()) 
    pred = pred[0].data.cpu().numpy()    
    # skimage.io.imsave('sample_disp.png',pred.astype('uint16'))


    img = (pred).astype('uint8')
    img = Image.fromarray(img)
    img.save('./images/out1/1.png' )

    output0 = cv2.convertScaleAbs(pred, alpha=3.0, beta=5)
    out0 = trans_color(output0)
    cv2.imwrite('./images/out2/1.png', out0)
    return pred


################################################------------------------------------##############################

class my_form(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        super().setupUi(self)
        self.initui()
        # 定义类内变量
        self.img1 = np.array([])   # 输入图像1
        self.img2 = np.array([])    # 输入图像2
        self.flag = 0   # 记录步骤
        self.flag_binary = 0
        self.flag_gaussianblur = 0

    def initui(self):
        # 定义点击逻辑
        self.pushButton_img1_input.clicked.connect(self.choose_img1)
        self.pushButton_img2_input.clicked.connect(self.choose_img2)
        self.pushButton_binary.clicked.connect(self.img_binary)
        self.pushButton_6.clicked.connect(self.realtime)
        self.pushButton_7.clicked.connect(self.SGM)
        self.pushButton_8.clicked.connect(self.realtime2)
        self.pushButton_gaussian_filter.clicked.connect(self.gaussian_filter)
        self.pushButton_clearall.clicked.connect(self.clear_all)

    def choose_img1(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片1', './images/', 'Image Files(*.png *.jpg *.bmp)')
        if fname[0]:
            img = cv.imread(fname[0])
            self.img1 = img
            self.img1_ = fname[0]
            # 转化格式后于label上显示图像
            x = img.shape[1]
            y = img.shape[0]
            z = img.shape[2]
            frame = QImage(img, x, y, z * x, QImage.Format_BGR888)

            self.label_img_in1.setPixmap(QPixmap(frame))
            self.label_img_in1.setScaledContents(True)

    def choose_img2(self):
        fname = QFileDialog.getOpenFileName(self, '选择图片2', './images/', 'Image Files(*.png *.jpg *.bmp)')
        if fname[0]:
            img = cv.imread(fname[0])
            self.img2 = img
            self.img2_ = fname[0]
            # 转化格式后于label上显示图像
            x = img.shape[1]
            y = img.shape[0]
            z = img.shape[2]
            frame = QImage(img, x, y, z * x, QImage.Format_BGR888)

            self.label_img_in2.setPixmap(QPixmap(frame))
            self.label_img_in2.setScaledContents(True)

    def img_binary(self):
        if self.img1.size:
            if self.flag_binary == 0:
                img1 = self.img1_
                img2 = self.img2_
                
                #################### 转化为三通道################
                mypred(img1, img2)
                img_gray = cv.imread('./images/out1/1.png', cv.IMREAD_COLOR)
                
               
                ######################转换完成#################
                x = img_gray.shape[1]
                y = img_gray.shape[0]
                z = img_gray.shape[2]
                frame = QImage(img_gray, x, y, z * x, QImage.Format_BGR888)
                self.label_img_out.setPixmap(QPixmap(frame))
                self.label_img_out.setScaledContents(True)
                # self.img1 = img_gray
                self.flag += 1
                self.flag_binary += 1
            else:
                QMessageBox.warning(self, "警告", "已进行过处理1！", QMessageBox.Ok, QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "警告", "请先选择输入图像！", QMessageBox.Ok, QMessageBox.Ok)

    def gaussian_filter(self):
        if self.img1.size:
            if self.flag_gaussianblur == 0:
                
                ######################--------------三通道---------------------####################
                img_blur = cv.imread('./images/out2/1.png', cv.IMREAD_COLOR)
                ##################----------------------------------------------#####################


                x = img_blur.shape[1]
                y = img_blur.shape[0]
                z = img_blur.shape[2]
                frame = QImage(img_blur, x, y, z * x, QImage.Format_BGR888)

                self.label_img_out_2.setPixmap(QPixmap(frame))
                self.label_img_out_2.setScaledContents(True)
                # self.img1 = img_blur
                self.flag += 1
                self.flag_gaussianblur += 1
            else:
                QMessageBox.warning(self, "警告", "已进行过处理2！", QMessageBox.Ok, QMessageBox.Ok)
        else:
            QMessageBox.warning(self, "警告", "请先选择输入图像！", QMessageBox.Ok, QMessageBox.Ok)


    def realtime(self):
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-i", "--input_path", default="./images/in_r/", help="folder with input images"
        )

        parser.add_argument(
            "-o",
            "--output_path",
            default="./images/out_r",
            help="folder for output images",
        )

        parser.add_argument(
            "-m", "--model_weights", default=None, help="path to model weights"
        )

        parser.add_argument(
            "-t",
            "--model_type",
            default="dpt_hybrid",
            help="model type [dpt_large|dpt_hybrid|midas_v21]",
        )

        # parser.add_argument("--kitti_crop", dest="kitti_crop", action="store_true")
        parser.add_argument("--optimize", dest="optimize", action="store_true")
        parser.add_argument("--no-optimize", dest="optimize", action="store_false")

        parser.set_defaults(optimize=True)

        args = parser.parse_args()

        default_models = {
            "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
            "dpt_large": "dpt/weights/dpt_large-midas-2f21e586.pt",
        }

        if args.model_weights is None:
            args.model_weights = default_models[args.model_type]

        # # set torch options
        # torch.backends.cudnn.enabled = True
        # torch.backends.cudnn.benchmark = True


        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # load network
        if args.model_type == "dpt_large":  # DPT-Large
            net_w = net_h = 384
            model = DPTDepthModel(
                path="./dpt/weights/dpt_large-midas-2f21e586.pt",
                backbone="vitl16_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        elif args.model_type == "dpt_hybrid": 
            net_w = net_h = 384
            model = DPTDepthModel(
                path= "./dpt/weights/dpt_hybrid-midas-501f0c75.pt",
                backbone="vitb_rn50_384",
                non_negative=True,
                enable_attention_hooks=False,
            )
            normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

        transform = Compose(
            [
                Resize(
                    net_w,
                    net_h,
                    resize_target=None,
                    keep_aspect_ratio=True,
                    ensure_multiple_of=32,
                    resize_method="minimal",
                    image_interpolation_method=cv2.INTER_CUBIC,
                ),
                normalization,
                PrepareForNet(),
            ]
        )

        model.eval()
        model = model.to(memory_format=torch.channels_last)
        model = model.half()
        model.to(device)


        if True:    
            AUTO = True  # 自动拍照，或手动按s键拍照
            INTERVAL = 0.01  # 自动拍照间隔（单位s）


            camera = cv2.VideoCapture(0)

            # 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)


            utc = time.time()
            folder = "./images/in_r/"  # 拍照文件目录

            def shot( frame):
                path = folder + "0.png"
                cv2.imwrite(path, frame)
         
            G, all = 0, 1000
            while (G<all):
                G=G+1
                ret, frame = camera.read()
                now = time.time()
                if AUTO and now - utc >= INTERVAL:
                    shot( frame)
                    utc = now
                
                ########################--------------------------########################
                
                img_names = glob.glob(os.path.join(args.input_path, "*"))
                num_images = len(img_names)
                # print("start processing")
                for ind, img_name in enumerate(img_names):
                    if os.path.isdir(img_name):
                        continue

                    print("  processing {} ({}/{})".format(img_name, G, all))

                    img = util.io.read_image(img_name)

                    # if args.kitti_crop is True:
                    #     height, width, _ = img.shape
                    #     top = height - 352
                    #     left = (width - 1216) // 2
                    #     img = img[top : top + 352, left : left + 1216, :]

                    img_input = transform({"image": img})["image"]
                    # compute
                    with torch.no_grad():
                        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)


                        if args.optimize == True and device == torch.device("cuda"):
                            sample = sample.to(memory_format=torch.channels_last)
                            sample = sample.half()

                        prediction = model.forward(sample)
                        prediction = (
                            torch.nn.functional.interpolate(
                                prediction.unsqueeze(1),
                                size=img.shape[:2],
                                mode="bicubic",
                                align_corners=False,
                            ).squeeze().cpu().numpy()
                        )

                        depth= prediction[0][0]
                        str = '%.1f' % depth
                        # self.label.setText(str)

                    filename = os.path.join(
                        args.output_path, os.path.splitext(os.path.basename(img_name))[0]
                    )
                    util.io.write_depth(filename, prediction, bits=2)



                ######################转换完成#################
                img_ = cv.imread('./images/in_r/0.png', cv.IMREAD_COLOR)
                x1 = img_.shape[1]
                y1 = img_.shape[0]
                z1 = img_.shape[2]
                frame1 = QImage(img_, x1, y1, z1 * x1, QImage.Format_BGR888)
                self.label_img_in1.setPixmap(QPixmap(frame1))
                self.label_img_in1.setScaledContents(True)
                self.label_img_in2.setPixmap(QPixmap(frame1))
                self.label_img_in2.setScaledContents(True)

    
                
                img_gray = cv.imread('./images/out_r/0.png', cv.IMREAD_COLOR)
                x1 = img_gray.shape[1]
                y1 = img_gray.shape[0]
                z1 = img_gray.shape[2]
                frame1 = QImage(img_gray, x1, y1, z1 * x1, QImage.Format_BGR888)
                self.label_img_out.setPixmap(QPixmap(frame1))
                self.label_img_out.setScaledContents(True)

                img_blur = cv2.imread('./images/out_r/0.png', 0)
                img_blur = cv2.convertScaleAbs(img_blur, alpha=1.5, beta=10)
                img_blur = trans_color(img_blur)
                x = img_blur.shape[1]
                y = img_blur.shape[0]
                z = img_blur.shape[2]
                frame = QImage(img_blur, x, y, z * x, QImage.Format_BGR888)
                self.label_img_out_2.setPixmap(QPixmap(frame))
                self.label_img_out_2.setScaledContents(True)
                #########################---------------------------######################
                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            camera.release()
            self.label_img_out.clear()
            self.label_img_out_2.clear()    
       


    def realtime2(self):
        if True:
            ########################--------------------------#########################
            AUTO = True  # 自动拍照，或手动按s键拍照
            INTERVAL = 0.01 # 自动拍照间隔（单位s）
            camera = cv2.VideoCapture(1)
            # 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
            camera.set(cv2.CAP_PROP_FRAME_WIDTH,2560)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
            
        
            utc = time.time()
            folder = "./images/videoTest/" # 拍照文件目录


            def shot(pos, frame, w, h):
                path = folder + pos +  "00.png"      
                frame = cv2.resize(frame,(w, h))
                cv2.imwrite(path, frame)
                print("snapshot saved into: " + path)
            
            G, all = 0, 1000
            while (G<all):
                G=G+1
                ret, frame = camera.read()
                # print("ret:",ret)
                # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
                left_frame = frame[0:720, 0:1280]
                right_frame = frame[0:720, 1280:2560]

                w, h=640, 480
                now = time.time()
                if AUTO and now - utc >= INTERVAL:
                    shot("/L/", left_frame, w, h )
                    shot("/R/", right_frame, w, h )
                    utc = now
            
                
                
                ll, rr = "./images/videoTest/L/00.png" ,"./images/videoTest/R/00.png"

                pred = mypred(ll, rr)
                depth =  pred[0][0]
                str = '%.1f' % depth
                self.label.setText(str)


                
                print("  processing  ({}/{})".format( G, all))
                ######################转换完成#################
                imgL= cv.imread("./images/out1/1.png", cv.IMREAD_COLOR)
                x2 = imgL.shape[1]
                y2 = imgL.shape[0]
                z2 = imgL.shape[2]
                frame2 = QImage(imgL, x2, y2, z2 * x2, QImage.Format_BGR888)
                self.label_img_out.setPixmap(QPixmap(frame2))
                self.label_img_out.setScaledContents(True)
                
                imgR= cv.imread("./images/out2/1.png", cv.IMREAD_COLOR)
                x3 = imgR.shape[1]
                y3 = imgR.shape[0]
                z3 = imgR.shape[2]
                frame3 = QImage(imgR, x3, y3, z3 * x3, QImage.Format_BGR888)
                self.label_img_out_2.setPixmap(QPixmap(frame3))
                self.label_img_out_2.setScaledContents(True)


                img_gray = cv.imread('./images/videoTest//L/00.png', cv.IMREAD_COLOR)
                x1 = img_gray.shape[1]
                y1 = img_gray.shape[0]
                z1 = img_gray.shape[2]
                frame1 = QImage(img_gray, x1, y1, z1 * x1, QImage.Format_BGR888)
                self.label_img_in1.setPixmap(QPixmap(frame1))
                self.label_img_in1.setScaledContents(True)

                img_blur = cv.imread('./images/videoTest//R/00.png', cv.IMREAD_COLOR)
                x = img_blur.shape[1]
                y = img_blur.shape[0]
                z = img_blur.shape[2]
                frame = QImage(img_blur, x, y, z * x, QImage.Format_BGR888)
                self.label_img_in2.setPixmap(QPixmap(frame))
                self.label_img_in2.setScaledContents(True)
                #########################---------------------------######################

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            camera.release()
            self.label_img_in1.clear()
            self.label_img_in2.clear()
            self.label_img_out.clear()
            self.label_img_out_2.clear()    

    def SGM(self):
        from matplotlib import pyplot as plt
        if True:
            ########################--------------------------#########################
            AUTO = True  # 自动拍照，或手动按s键拍照
            INTERVAL = 0.01  # 自动拍照间隔（单位s）
            camera = cv2.VideoCapture(1)
            # 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

            utc = time.time()
            folder = "./images/videoTest/"  # 拍照文件目录

            def shot(pos, frame, w, h):
                path = folder + pos + "00.png"
                frame = cv2.resize(frame, (w, h))
                cv2.imwrite(path, frame)
                print("snapshot saved into: " + path)

            G, all = 0, 1000
            while (G < all):
                G = G + 1
                ret, frame = camera.read()
                # print("ret:",ret)
                # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
                left_frame = frame[0:720, 0:1280]
                right_frame = frame[0:720, 1280:2560]

                w, h = 640, 480
                now = time.time()
                if AUTO and now - utc >= INTERVAL:
                    shot("/L/", left_frame, w, h)
                    shot("/R/", right_frame, w, h)
                    utc = now


                ll, rr = "./images/videoTest/L/00.png", "./images/videoTest/R/00.png"

                imgL = cv.imread(ll, 0)
                imgR = cv.imread(rr, 0)
                stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
                disparity = stereo.compute(imgL, imgR)
                pred = disparity.data.cpu().numpy()
                img = (pred).astype('uint8')
                img = Image.fromarray(img)
                img.save('./images/out1/1.png')
                output0 = cv2.convertScaleAbs(pred, alpha=3.0, beta=5)
                out0 = trans_color(output0)
                cv2.imwrite('./images/out2/1.png', out0)

                depth = pred[0][0]
                str = '%.1f' % depth
                self.label.setText(str)

                print("  processing  ({}/{})".format(G, all))
                ######################转换完成#################
                imgL = cv.imread("./images/out1/1.png", cv.IMREAD_COLOR)
                x2 = imgL.shape[1]
                y2 = imgL.shape[0]
                z2 = imgL.shape[2]
                frame2 = QImage(imgL, x2, y2, z2 * x2, QImage.Format_BGR888)
                self.label_img_out.setPixmap(QPixmap(frame2))
                self.label_img_out.setScaledContents(True)

                imgR = cv.imread("./images/out2/1.png", cv.IMREAD_COLOR)
                x3 = imgR.shape[1]
                y3 = imgR.shape[0]
                z3 = imgR.shape[2]
                frame3 = QImage(imgR, x3, y3, z3 * x3, QImage.Format_BGR888)
                self.label_img_out_2.setPixmap(QPixmap(frame3))
                self.label_img_out_2.setScaledContents(True)

                img_gray = cv.imread('./images/videoTest//L/00.png', cv.IMREAD_COLOR)
                x1 = img_gray.shape[1]
                y1 = img_gray.shape[0]
                z1 = img_gray.shape[2]
                frame1 = QImage(img_gray, x1, y1, z1 * x1, QImage.Format_BGR888)
                self.label_img_in1.setPixmap(QPixmap(frame1))
                self.label_img_in1.setScaledContents(True)

                img_blur = cv.imread('./images/videoTest//R/00.png', cv.IMREAD_COLOR)
                x = img_blur.shape[1]
                y = img_blur.shape[0]
                z = img_blur.shape[2]
                frame = QImage(img_blur, x, y, z * x, QImage.Format_BGR888)
                self.label_img_in2.setPixmap(QPixmap(frame))
                self.label_img_in2.setScaledContents(True)
                #########################---------------------------######################

                key = cv2.waitKey(1)
                if key == ord("q"):
                    break
            camera.release()
            self.label_img_in1.clear()
            self.label_img_in2.clear()
            self.label_img_out.clear()
            self.label_img_out_2.clear()



        imgL = cv.imread('tsukuba_l.png', 0)
        imgR = cv.imread('tsukuba_r.png', 0)
        stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
        disparity = stereo.compute(imgL, imgR)
        plt.imshow(disparity, 'gray')
        plt.show()





    def clear_all(self):
        self.img1 = np.array([])
        self.img2 = np.array([])
        self.flag = 0
        self.flag_binary = 0
        self.flag_gaussianblur = 0
        self.label_img_in1.clear()
        self.label_img_in2.clear()
        self.label_img_out.clear()
        self.label_img_out_2.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    my_uiform = my_form()
    my_uiform.show()
    sys.exit(app.exec_())
    
    
    
