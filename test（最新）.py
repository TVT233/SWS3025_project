import cv2
import dlib
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import sys
import os
from scipy import signal
from scipy.fftpack import fft, ifft
import matplotlib.mlab as mlab
from sklearn import preprocessing
from sklearn import impute
import pandas as pd

predictorPath = r"shape_predictor_68_face_landmarks.dat"  # 模型位置 此模型基于HOG特征
predictorIdx = [[1, 2, 3, 4, 31, 36, 48], [12, 13, 14, 15, 35, 45, 54]]  # 这是68个特征点里边的左右脸颊部分的特征点编号
medIdx = [27, 28, 29, 30]  # 大概是中轴线的坐标（？）


def rect_to_bb(rect):
    """ Transform a rectangle into a bounding box
    Args:
        rect: an instance of dlib.rectangle
    Returns:
        [x, y, w, h]: coordinates of the upper-left corner
            and the width and height of the box
    """
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return [x, y, w, h]


def shape_to_np(shape, dtype="int"):
    """ Transform the detection results into points
    Args:
        shape: an instance of dlib.full_object_detection
    Returns:
        coords: an array of point coordinates
            columns - x; y
    """
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def np_to_bb(coords, ratio=4, dtype="int"):
    """ Chooose ROI based on points and ratio
    Args:
        coords: an array of point coordinates
            columns - x; y
        ratio: the ratio of the length of the bounding box in each direction
            to the distance between ROI and the bounding box 每个方向的边界框长度与ROI与边界框距离的比值
        dtype: optional variable, type of the coordinates
    Returns:
        coordinates of the upper-left and bottom-right corner
    """
    roi = cv2.fitEllipse(coords)  # 轮廓椭圆拟合
    m_roi = list(map(int, [roi[0][0], roi[0][1], roi[1][0] / (ratio + 1), roi[1][1] / (ratio + 1), roi[
        2]]))  # 全取int（？） map()接收一个函数 f（） 和一个 list，并通过把函数 f 依次作用在 list 的每个元素上，得到一个新的 list 并返回（不改变原来的list）
    return m_roi

def resize(image, width=1200):
    """ Resize the image with width
    Args:
        image: an instance of numpy.ndarray, the image
        width: the width of the resized image
    Returns:
        resized: the resized image
        size: size of the resized image
    """
    r = width * 1.0 / image.shape[1]
    size = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, size, interpolation=cv2.INTER_AREA)  # 调整图像的高度（宽度），高度的新大小=原高度*1200*1.0/长度
    return resized, size


def coordTrans(imShape, oriSize, rect):
    """Transform the coordinates into the original image 将坐标转换为原始图像
    Args:
        imShape: shape of the detected image 被检测图像的形状
        oriSize: size of the original image 原始图像的大小
        rect: an instance of dlib.rectangle, the face region
    Returns:
        the rect in the original image 原图中的脸部区域
    """

    left = int(rect.left() / oriSize[0] * imShape[1])
    right = int(rect.right() / oriSize[0] * imShape[1])
    top = int(rect.top() / oriSize[1] * imShape[0])
    bottom = int(rect.bottom() / oriSize[1] * imShape[0])

    left = int(round(rect.left() / oriSize[0] * imShape[1]))
    right = int(round(rect.right() / oriSize[0] * imShape[1]))
    top = int(round(rect.top() / oriSize[1] * imShape[0]))
    bottom = int(round(rect.bottom() / oriSize[1] * imShape[0]))
    # 四舍五入又做了一遍？

    return dlib.rectangle(left, top, right, bottom)


def dataplot(T, data, fn):
    plt.figure(fn)  # 创建画板？
    clr = ['b', 'g', 'r']  # color
    for i in range(0, 3):
        plt.subplot(3, 2, 2 * i + 1)  # 指一个3行2列的图中从左到右从上到下的第1,3,5的位置
        plt.plot(T, data[i], clr[i])  # 由点连成折线图
        plt.grid()  # 生成网格
        plt.subplot(3, 2, 2 * i + 2)  # 指一个3行2列的图中从左到右从上到下的第2,4的位置
        plt.psd(data[i], NFFT=256, Fs=25, window=mlab.window_none,
                scale_by_freq=True)
        # plt.psd：绘制功率谱密度，对于数据data[i]
        # NFFT：在每个块中用于FFT的数据点的数目
        # Fs：采样频率为256（每个时间单位的采样点数，用于计算傅立叶频率，频率，以周期/时间单位）
        # window=mlab.window_none表示窗口版本（？）


class Detector:
    """ Detect and calculate ppg signal 检测并计算ppg信号
    roiRatio: a positive number, the roi gets bigger as it increases 一个正数，ROI随着它增大而增大
    smoothRatio: a real number between 0 and 1, 在0到1之间的一个平滑值，越大界限越明显
         the landmarks get stabler as it increases
    """
    detectSize = 480
    clipSize = 540
    roiRatio = 2
    markSmoothRatio = 0.95
    Smthre = 0.9
    alpha = round(math.log(1 / Smthre) / 20, 4)  # 以4为底 (1 / Smthre) / 20 的对数

    # _foo 保护变量 __foo__ 私有成员
    def __init__(self, detectorPath=None, predictorPath=None, predictorIdx=None):
        # 两个下划线开头的函数是声明该属性为私有，不能在类的外部被使用或访问（？）
        """ Initialize the instance of Detector 初始化

        detector: dlib.fhog_object_detector # 在给定数据集上，使用hog+svm方法建立一个detector做目标检测（？）
        predictor: dlib.shape_predictor #关键点预测器，用于标记人脸关键点
        rect: dlib.rectangle, face region in the last frame 最后一帧的面部区域
        landmarks: numpy.ndarray, coordinates of face landmarks in the last frame 最后一帧的面部坐标
                columns - x; y

        Args:
            detectorPath: path of the face detector
            predictorPath: path of the shape predictor
        """
        self.detector = dlib.get_frontal_face_detector()
        # get_frontal_face_detector 返回值是一个矩形 是包含全部人脸的一个矩形凸包
        # (PythonFunction, in Classes) in Classes是采样次数
        # 通过self间接调用被封装的内容
        self.predictor = dlib.shape_predictor(predictorPath)  # predictorPath是68个关键点模型地址，在开头定义过，这里载入预测器（模型）
        self.idx = predictorIdx
        self.rect = None
        self.face = None
        self.landmarks = None
        self.rois = None

    def __call__(self, image, t):
        """ Detect the face region and returns the ROI value 检测人脸区域并返回ROI值

        Face detection is the slowest part. 人脸检测是最慢的部分

        Args:
            image: an instance of numpy.ndarray, the image
        Return:
            val: an array of ROI value in each color channel 每个颜色通道的ROI值数组
        """
        val = [0, 0, 0]
        # Resize the image to limit the calculation 重置（调整）图像大小以限制计算
        resized, detectionSize = resize(image, self.detectSize)  # 把图像按比例裁了方便计算
        # Perform face detection on a grayscale image
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)  # 把裁下来的区域变成灰度图
        # No need for upsample, because its effect is the same as resize 不需要upsample，因为它的效果和resize一样
        rects = self.detector(gray, upsample_num_times=0)  # 对灰度图进行面部识别 返回值是一个矩形 是包含全部人脸的一个矩形凸包（*是什么格式呢？）
        num = len(rects)  # there should be one face 脸的个数
        if num == 0:  # 如果没有识别到用户的脸 就沿用上一张图像并返回没有识别
            print("Time: ", '{:.3f}'.format(t), " No face in the frame!")  # {:.3f}表示保留小数点后三位
            if isinstance(self.landmarks, type(None)):
                return val
            else:
                landmarks = self.landmarks
                # Perfom landmarks smoothing
                if (self.rect != None):
                    distFM = self.distForMarks(self.landmarks[medIdx], landmarks[medIdx])
                    # print('\ndistFM:'+'{:.2f}'.format(distFM)+'\n') {:.2f}表示保留小数点后两位
                    # print(self.distForMarks(self.landmarks[medIdx], landmarks[medIdx]))
                    landmarks = self.smoothMarks(self.landmarks, landmarks, distFM)

                # ROI value，idx是脸颊部分的编号，这里根据这些特征点和比例（上面定义的是2）选择ROI区域，得到roi再记到rois里
                rois = [np_to_bb(landmarks[idx], self.roiRatio) for idx in self.idx]
                mask = np.zeros([2, image.shape[0], image.shape[1]], dtype=int)
                # image.shape[0], 图片垂直尺寸；image.shape[1], 图片水平尺寸。
                # 制作mask掩码，生成2个多维数组，每一个的行数=图片的垂直尺寸；列数=图片水平尺寸。每个值都是0.

                val = [[], []]
                for i in range(0, len(rois)):
                    # print(i)
                    roi = rois[i]
                    cv2.ellipse(mask[i], (roi[0], roi[1]), (roi[2], roi[3]), roi[4], -180, 180, 255, -1)
                    # 绘制椭圆，在mask[i]上，以(roi[0], roi[1])为中心，(roi[2], roi[3])为长短轴（？），
                    # 旋转roi[4]度（顺时针方向），绘制的起始角度为-180（顺时针方向）
                    # 绘制的终止角度为180（例如，绘制整个椭圆是0,360，绘制下半椭圆就是0,180），此处即从-180到180
                    # 颜色为255（白色）
                    # 线性：-1表示填充（？）
                    # 参考https://blog.csdn.net/qq_14997473/article/details/80819337

                    mmsk = np.mean(image[mask[i] > 0], 0)  # 返回条件成立的占比（？还是求均值）
                    val[i].append(mmsk)  # 在每个val[i]的最后插入mmsk
                height = image.shape[0]  # 垂直尺度
                width = image.shape[1]  # 水平尺度
                iht = int(height / 4)
                iwt = int(width / 16)
                # 下两行是在算啥（？）
                lbg = np.mean(np.mean(image[iht:height - iht, iwt:iwt * 2], 0), 0)
                rbg = np.mean(np.mean(image[iht:height - iht, width - 2 * iwt:width - iwt], 0), 0)
                return val, rois, [lbg, rbg]
        if num >= 2:
            print("More than one face!")
            return val
        rect = rects[0]
        # Perform landmark prediction on the face region 对面区域进行地标性预测（？）
        face = coordTrans(image.shape, detectionSize, rect)  # rect是识别的人脸坐标 这里把图像截出来
        # print(face)
        shape = self.predictor(image, face)  # 作出68个特征点
        landmarks = shape_to_np(shape)  # 将shape的检测结果转换为点坐标
        # Perfom landmarks smoothing 大概是平滑化处理吧 原理没看懂（？）
        if (self.rect != None):
            distFM = self.distForMarks(self.landmarks[medIdx], landmarks[medIdx])
            # print('\ndistFM:'+'{:.2f}'.format(distFM)+'\n')
            # print(self.distForMarks(self.landmarks[medIdx], landmarks[medIdx]))
            landmarks = self.smoothMarks(self.landmarks, landmarks, distFM)

        # ROI value 这部分和上面那部分一样 求ROI
        rois = [np_to_bb(landmarks[idx], self.roiRatio) for idx in self.idx]
        mask = np.zeros([2, image.shape[0], image.shape[1]], dtype=int)
        val = [[], []]
        for i in range(0, len(rois)):
            # print(i)
            roi = rois[i]
            cv2.ellipse(mask[i], (roi[0], roi[1]), (roi[2], roi[3]), roi[4], -180, 180, 255, -1)
            mmsk = np.mean(image[mask[i] > 0], 0)
            val[i].append(mmsk)
        # print(val)
        # sys.exit(0)
        # plt.figure(1)
        # plt.subplot(1, 2, i + 1)
        # plt.imshow(mask[i])

        # plt.show()
        # cv2.waitKey()

        # vals = [np.mean(np.mean(image[roi[1]:roi[3], roi[0]:roi[2]], 0), 0) for roi in rois]
        # val = np.mean(vals, 0)
        # image 2160x3840,3
        height = image.shape[0]
        width = image.shape[1]
        iht = int(height / 4)
        iwt = int(width / 16)
        lbg = np.mean(np.mean(image[iht:height - iht, iwt:iwt * 2], 0), 0)
        rbg = np.mean(np.mean(image[iht:height - iht, width - 2 * iwt:width - iwt], 0), 0)

        self.rect = rect
        self.landmarks = landmarks
        self.face = face
        return val, rois, [lbg, rbg]

    def smoothMarks(self, landmarks1, landmarks2, distFM):
        smoothRatio = math.exp(-self.alpha * distFM)  # 返回e（欧拉常数）的x次方，大概是平滑的一种计算（？）
        landmarks = smoothRatio * landmarks1 \
                    + (1 - smoothRatio) * landmarks2
        landmarks = np.array([[round(pair[0]), round(pair[1])]
                              for pair in landmarks])
        landmarks = landmarks.astype(int)
        return landmarks

    def distForMarks(self, mask1, mask2):
        """Calculate the distance between two rectangles for rectangle smoothing 计算两个矩形之间的距离，用于矩形平滑
        Arg:
            rect1, rect2: dlib.rectangle
        Return:
            distance between rectangles
        """
        dist = mask1 - mask2
        dist = np.sum(np.sqrt(np.sum(np.multiply(dist, dist), 1)))
        return dist


def VideoToTxt(videopath, PPGpath, subject, filename, startTime):
    # Initialization
    # TXT数据存下（之后读取就读只读TXT数据就行）
    print('File ' + videopath + ' Extracting...')  # Extracting：提取
    detect = Detector(predictorPath=predictorPath, predictorIdx=predictorIdx)
    times = []
    data = [[], [], []]
    data2 = [[], [], []]
    cbg = []
    rois_clt = []
    ED = []

    # 302-335行 获取PPG信号
    if os.path.exists(PPGpath):
        # 判断PPGpath是否存在于代码的同一目录下（这里也可以写绝对路径，也可以检测文件夹）
        # 参考https://blog.csdn.net/u012424313/article/details/82216092
        with open(PPGpath, 'r') as f:
            # 从文件中读取数据，然后关闭文件句柄；
            # 参考https://www.jb51.net/article/135285.htm
            lines = f.readlines()
            ED = list(map(float, lines[1:]))  # 全转换成浮点型

    # 关于cv2的函数 https://www.jianshu.com/p/c8414aac8b04
    video = cv2.VideoCapture(videopath)  # 写明本地视频路径
    #    video = cv2.VideoCapture(0)
    fps = video.get(cv2.CAP_PROP_FPS)  # 得到视频的帧率
    video.set(cv2.CAP_PROP_POS_FRAMES, startTime * fps)  # 设定从视频的第（startTime * fps）帧开始读取
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # 得到水平长度
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 得到垂直长度
    # Handle frame one by one （*Handle frame啥意思？）
    t = 0.0
    t_off = 1
    ret, frame = video.read()  # 读取视频 ret为True或者False,代表有没有读取到图片；frame表示截取到一帧的图片
    init_t = time.time()  # 该函数返回当前时间到时间戳，也就是距离1970年1月1日00:00:00的差值

    # c=1
    # if video.isOpened():
    #     ret, frame = video.read()
    # else:
    #     ret=False
    # timeF=30
    # while ret:
    #     if(c%timeF==0):
    #
    #         val = detect(frame, t)

    c=0
    while (ret):  # 当视频（摄像头）初始化成功
        # cv2.imshow('frame',frame)
        # cv2.waitKey(int(1000/fps))
        c=c+1
        t += 1.0 / fps  # 每次增加一帧的时间
        # detect
        val = detect(frame, t)  # detect = Detector(predictorPath=predictorPath, predictorIdx=predictorIdx) （？）
        # print(type(val))
        if type(val) != list:  # （？）
            value = val[0][0][0]
            value2 = val[0][1][0]
            rois = val[1]
            tc = val[2]

            # show result
            times.append(t)  # 在times的最后插入t

            # 以下的PPG计算没看懂原理
            if type(value) != int:
                # print(value)
                for i in range(3):
                    data[i].append(value[i])
                    data2[i].append(value2[i])
                cbg.append(tc)
                rois_clt.append(rois)
            elif len(data[0]) > 1:
                for i in range(3):
                    data[i].append(data[i][-1])
                    data2[i].append(data2[i][-1])
                cbg.append(cbg[-1])
            # print(t,'Save mode ON:',savemode)
        # check stop or quit 停止或退出
        ret, frame = video.read()
        # if c==5:
        #     break
        if cv2.waitKey(1) & 0xFF == ord('q') or not ret:  # or t > t_off:
            break

    # release memory and destroy windows 释放内存并销毁窗口
    video.release()
    print(c)
    data = np.array(data)
    data2 = np.array(data2)
    nbg = np.array(cbg)
    #
    # if ED:
    #     yed = np.array(ED)
    #     np.savetxt(cutoutpath + '_ed.txt', yed, fmt='%f')
    nprc = np.array(rois_clt)
    return [data, data2, nbg[:, 0, :].T, nbg[:, 1, :].T, nprc[:, 0, :], nprc[:, 1, :]]


def ReadOut(subject):
    cofolder = ROUT + subject
    cutoutpath = cofolder + '/' + subject
    print(cutoutpath + '\nLoading file...\n')
    # data-3(BGR)*N
    data1 = np.loadtxt(cutoutpath + '_data1' + '.txt')
    data2 = np.loadtxt(cutoutpath + '_data2' + '.txt')
    # # bg-3(BGR)*N
    bg1 = np.loadtxt(cutoutpath + '_bg1' + '.txt')
    bg2 = np.loadtxt(cutoutpath + '_bg2' + '.txt')
    # # rc-N*5((x,y),(2b,2a),theta)
    rc1 = np.loadtxt(cutoutpath + '_roi1' + '.txt')
    rc2 = np.loadtxt(cutoutpath + '_roi2' + '.txt')
    # 8*N-1
    ed = []
    if os.path.exists(cutoutpath + '_ed.txt'):
        ed = np.loadtxt(cutoutpath + '_ed.txt')
    # return [data1, rc1, bg1], [data2, rc2, bg2], ed
    return data1, data2, bg1, bg2, rc1, rc2


def EDARead(subject):
    cofolder = './data_edav100/EDA/' + subject
    eda_data = np.loadtxt(cofolder + '_marked' + '.txt')
    return eda_data


if __name__ == "__main__":
    videofolder = 'C:\\Users\\rosie\\Desktop\\ppg\\video'  # 读取视频文件夹所在位置
    #video = cv2.VideoCapture('videofolder')  # 写明本地视频路径
    #    video = cv2.VideoCapture(0)
    #videofolder = r'C:\\Users\Administrator\Desktop\ppg\\video\\'  # 读取视频文件夹所在位置
    ExpDate = ['0509']  # ExpDate 存储办公室名称
    ExpN = [1]  # ExpN 存储每个办公室几个人
    MVIN = 15
    ROUT = './data_edav100/'
    BOUT = './beat_split_edav64/'
    VIDEO_FIlE = []
    subjects = []
    for i in range(0, len(ExpDate)):  # 读取ExpDate、ExpN中的名称，做子文件视频的命名（为视频做标注），方便查找视频
        for j in range(0, ExpN[i]):
            sub_file = [ExpDate[i] + '_' + '{:02d}'.format(j + 1) + '_' + '{:d}'.format(mvn + 1) + '.mp4' for mvn in
                        range(0, MVIN)]
            # subjects.append(ExpDate[i]+'_'+'{:02d}'.format(j+1))
            VIDEO_FIlE.append(sub_file)  # 在VIDEO_FIlE中添加上面的子文件（相当于做个视频名称目录）
   # print('VIDEO_FIlE',VIDEO_FIlE)
    Video_Read = True  # 设置状态（如视频读取是否）
    Beats_Split = False
    Data_label = True
    Data_ob = True
    # 创建子文件夹（？）

    # 3.29 这个if Video_Read == False: 可以正常执行，输出视频文件的txt文件，共6个
    #
    if Video_Read == False:  # 如果视频读取失败
        for SF in VIDEO_FIlE:  # 遍历VIDEO_FIlE中的子文件视频名称，查看该视频是否在videofolder中
            v_data1 = []  # 预先为参数v_data1提供一个数组，方便后面将相关参数存到相应数组中
            v_data2 = []  # 预先为参数v_data2提供一个数组，方便后面将相关参数存到相应数组中
            v_bkgd1 = []  # 同理
            v_bkgd2 = []
            v_roi1 = []
            v_roi2 = []
            # 数据存下（？）
            for vi in SF:
                fe = os.path.exists(videofolder + vi)  # 判断判断文件是否存在
                print('File\n', videofolder + vi, '\nexisting: ', fe)  # 输出文件存在状态
                subs = vi.split('_')  # 433-440，将视频名称拆分，看看具体是哪个视频
                subject = subs[0] + '_' + subs[1]  # 拆分成前后两个部分   （文件夹名称+子视频名称）
                print('subject: ', subject)  # 输出名称
                # All_data: 3xn (RGBxframes), ROI:nx5
                if subs[2] == 1:
                    st = 60
                else:
                    st = 0
                # print(st)

                # 当Video_Read = False，VideoToTxt()程序死在这
                [data1, data2, bg1, bg2, roi1, roi2] = VideoToTxt(videofolder + vi, '', subject, subject,st)  # 将视频转换成TXT格式（之后读取就读只读TXT数据就行）；这里的st就是startTime，视频是从st * fps帧开始读取的
                v_data1.append(data1)  # 添加PPG信号中的返回的标注信息data1
                v_data2.append(data2)  # 添加PPG信号中的返回的标注信息data2
                v_bkgd1.append(bg1)  # 以下同理
                v_bkgd2.append(bg2)
                v_roi1.append(roi1)
                v_roi2.append(roi2)
                print('File ', vi, ' Loading Finished')
            if v_data1 == []:
                continue

            av_data1 = np.concatenate(v_data1, 1)  # 将PPG标注信息v_data1与1相拼接（标注信息，名称命名）
            av_data2 = np.concatenate(v_data2, 1)  # 将PPG标注信息v_data2与1相拼接（标注信息，名称命名）
            av_bg1 = np.concatenate(v_bkgd1, 1)  # 将PPG标注信息v_bkgd1与1相拼接（标注信息，名称命名）
            av_bg2 = np.concatenate(v_bkgd2, 1)  # 将PPG标注信息v_bkgd2与1相拼接（标注信息，名称命名）
            av_roi1 = np.concatenate(v_roi1, 0)  # 将PPG标注信息v_roi1与0相拼接（标注信息，名称命名）
            av_roi2 = np.concatenate(v_roi2, 0)  # 将PPG标注信息v_roi2与0相拼接（标注信息，名称命名）
            cofolder = ROUT + subject
            cutoutpath = cofolder + '/' + subject
            if not os.path.exists(cofolder):  # 如果cofolder不存在
                os.makedirs(cofolder)  # 创建 cofolder

            np.savetxt(cutoutpath + '_data1' + '.txt', av_data1, fmt='%.4f')  # 保存命名过的.txt文件（data1）
            np.savetxt(cutoutpath + '_data2' + '.txt', av_data2, fmt='%.4f')  # 保存命名过的.txt文件（data2）
            # 以下同理
            np.savetxt(cutoutpath + '_bg1' + '.txt', av_bg1, fmt='%.4f')
            np.savetxt(cutoutpath + '_bg2' + '.txt', av_bg2, fmt='%.4f')

            np.savetxt(cutoutpath + '_roi1' + '.txt', av_roi1, fmt='%.4f')
            np.savetxt(cutoutpath + '_roi2' + '.txt', av_roi2, fmt='%.4f')

            print(SF, '\nFile Done.')
    if Video_Read == True and Beats_Split == False:  # 如果视频读取成功，但 Beats_Split失败
        # subjects = ['0501_01', '0501_02', '0501_03', '0502_01', '0502_03', '0502_04', '0502_05', '0502_06', '0502_07']
        #subjects = ['0509_01', '0509_02', '0509_03', '0509_04',  # 科目： 猜测，应该是509，510办公室人员 原始数据的集合
                   # '0510_01', '0510_02', '0510_03', '0510_04', '0510_05', '0510_06', '0510_07']
        subjects = ['0509_01']
        # subjects = ['0501_02', '0501_03', '0502_02']
        fps = 50  # 设置参数fps数值
        tt = 164  # -60 if it's 0509-0510               #设置参数tt数值
        ttf = tt * fps  # 计算ttf
        #std表示的是？
        std = np.array([[164, 284], [298, 418], [436, 556], [570, 690], [708, 828], [840, 960]]) - tt
        cutoff1 = [0.5, 5]
        eda_ch = False
        # 排除NA数据（？）
        for subject in subjects:  # 排除 '0502_03'中的数据
            # if subject != '0502_03':
            #     continue
            sdata1, sdata2, sbg1, sbg2, src1, src2 = ReadOut(subject)  
            # 读取.txt文件中的PPG信号标注信息sdata1, sdata2, sbg1, sbg2, src1, src2
            # data - PDA[0,1][0] 3(BGR)*N
            # rc - PDA[0,1][1] N*5 para((x,y),(2b,2a),theta)
            # bg - PDA[0,1][2] 3(BGR)*N
            # ed - PDA[2] (8*N-1,)

            # butter、firwin2构造滤波器（？）
            if eda_ch == True:
                eda_data = EDARead(subject)  # 从subject中读取eda_data
                eda_time = eda_data[:, 0]  # 读取eda_data的第0列，称作eda_time
                eda_conduct = eda_data[:, 1]  # 读取eda_data的第1列，称作eda_conduct
                eda_tonic = eda_data[:, 2]  # 读取eda_data的第2列，称作eda_tonic
                eda_tdata_sel = []  # 建立eda_tdata_sel数组，方便后面往里面存数据
                eda_cdata_sel = []  # 建立eda_cdata_sel数组，方便后面往里面存数据
                for sti in std:  # 从这开始到535的roi2 = o_roi2[0:filt_data.shape[0], :]是滤波效果（？）      #应该是滤波效果  filt_data
                    ti_eda = (eda_time > sti[0]) & (
                                eda_time < sti[1])  # 取 eda_time中  sti[0]< eda_time < sti[1]的这段，赋值给ti_eda
                    tempt = eda_tonic[ti_eda]
                    tempc = eda_conduct[ti_eda]
                    eda_tdata_sel.append(tempt)  # 往eda_tdata_sel中添加tempt
                    eda_cdata_sel.append(tempc)  # 往eda_cdata_sel中添加tempc
                eda_tscale = (eda_tonic - np.min(eda_tdata_sel)) / (
                            np.max(eda_tdata_sel) - np.min(eda_tdata_sel))  # 归一化公式  归一化 eda_tscale
                eda_cscale = (eda_conduct - np.min(eda_cdata_sel)) / (
                            np.max(eda_cdata_sel) - np.min(eda_cdata_sel))  # 归一化 eda_cscale
                eda_stage_scale = []  # 建立eda_stage_scale数组，方便后面往里面存数据

            # 谐波频率0.5-5Hz
            # fir有限冲击响应，效果没有butter好但稳定
            # butter迭代，上一点，有故障可能

            # 开始滤波1（？） 不确定是1还是2    #滤波 data2       #519-645 这后面是用PPG信号的标注信息做科研学术，提取信号预处理，提取特征（公式计算）
            o_data2 = sdata2[1][:]
            my_imputer = impute.SimpleImputer()  # 对数据中的缺失值进行插补
            data_imputed = my_imputer.fit_transform(o_data2.reshape(1,-1))  # 先将o_data2转成一列，对o_data2先拟合fit，找到该part的整体指标，如均值、方差、最大值最小值等等（根据具体转换的目的），然后对该o_data2进行转换transform，从而实现数据的标准化、归一化等等。
            data2 = data_imputed.reshape(-1, order='C')  # reshape(-1, order='C') 以二维数组为例，简单来讲就是横着读，横着写，优先读/写一行。
            bg2 = sbg2[1][:]
            o_roi2 = src2[:, :]
            # data2 = preprocessing.robust_scale(o_data2)

            # 开始滤波2（？）
            times = np.array(range(0, data2.shape[0])) / fps  # 计算 times
            b_but, a_but = signal.butter(6, [2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps], 'bandpass')  # 计算 b_but, a_but
            b_fir = signal.firwin2(512, [0, 2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps, 1], [0, 1, 1, 0])  # 计算 b_fir
            cutoff2 = [0.7, 2]
            times = np.array(range(0, data2.shape[0])) / fps  # 计算  times
            # ffd1 = signal.filtfilt(b_fir, 1, data2)
            fd2 = signal.filtfilt(b_but, a_but, data2)  # filtfilt()    开始滤波data2得到 fd2
            filt_data = signal.filtfilt(b_fir, 1, fd2)  # 开始滤波fd2得到 filt_data
            # smooth_filt_data = np.array(savgol(filt_data.tolist(), 7, 2))
            roi2 = o_roi2[0:filt_data.shape[0], :]

            fig2 = plt.figure()
            ax1 = fig2.add_subplot(221)
            fb2 = signal.filtfilt(b_but, a_but, bg2)
            fb2 = signal.filtfilt(b_fir, 1, fb2)
            roi2_fbut = signal.filtfilt(b_but, a_but, roi2.T)
            roi2_fbf = signal.filtfilt(b_fir, 1, roi2_fbut).T
            ax1.plot(times, filt_data - np.mean(filt_data) + 2, label='face green channel', color='green')
            ax1.plot(times, fb2 - np.mean(fb2), label='backgroud', color='blue')
            ax1.grid()
            ax1.legend()
            ax3 = fig2.add_subplot(223)

            ax3.plot(times, data2)
            ax3.grid()
            ax4 = fig2.add_subplot(224)
            ax4.psd(filt_data, NFFT=512, Fs=fps, window=mlab.window_none,
                    scale_by_freq=True)
            ax4.set_xlim([0, 5])
            ax2 = fig2.add_subplot(222)
            ax2.plot(times, roi2_fbf[:, 0] - np.mean(roi2_fbf[:, 0]) + 8, label='x')
            ax2.plot(times, roi2_fbf[:, 1] - np.mean(roi2_fbf[:, 1]) + 6, label='y')
            ax2.plot(times, roi2_fbf[:, 2] - np.mean(roi2_fbf[:, 2]) + 8, label='2b')
            ax2.plot(times, roi2_fbf[:, 3] - np.mean(roi2_fbf[:, 3]) + 6, label='2a')
            ax2.plot(times, roi2_fbf[:, 4] - np.mean(roi2_fbf[:, 4]) + 6, label='th')
            ax2.grid()
            ax2.legend()
            # sss
            # stage_beats_split(data2, subject, pfn)
            # sss
            fd_scale = []  # 建立fd_scale数组，方便后面往里面存数据

            for sti in std:
                #print(sti)
                temp_data = filt_data[sti[0] * fps:sti[1] * fps]

                if eda_ch == True:
                    ti_eda = (eda_time > sti[0]) & (
                                eda_time < sti[1])  # 取 eda_time中  sti[0]< eda_time < sti[1]的这段，赋值给ti_eda
                    tempt = eda_tscale[ti_eda]
                    tempc = eda_cscale[ti_eda]
                    eda_stage_scale.append(tempt)  # 往eda_stage_scale中添加tempt
                fd_scale.append(temp_data)  # 往fd_scale中添加temp_data
            # fd_scale = preprocessing.robust_scale(filt_data.reshape(-1, 1))[:].T[0]
            data_stage_scale = preprocessing.robust_scale(list(map(list, zip(*fd_scale))))
            if eda_ch == True:
                eda_stage_scale = np.array(eda_stage_scale).T  # 将eda_stage_scale矩阵转置
            b_c1, a_c1 = signal.butter(6, [2 * cutoff2[0] / fps, 2 * cutoff2[1] / fps], 'bandpass')  # 计算得到 b_c1, a_c1
            for k in range(0, data_stage_scale.shape[1]):
                data_temp = data_stage_scale[:, k]  # 将data_stage_scale 中的第K列赋值给  data_temp
                if eda_ch == True:
                    eda_temp = eda_stage_scale[:, k]  # 将eda_stage_scale 中的第K列赋值给  eda_temp
                ffds = signal.filtfilt(b_c1, a_c1, data_temp)  # 滤波 data_temp
                fpeak = signal.argrelextrema(ffds, np.greater)[0]  # 寻找滤波后的信号ffds的极值点
                data = np.array(data_temp)
                for i in range(0, len(fpeak)):  # 以下程序判断极值点
                    pkn = 3
                    if fpeak[i] + pkn >= len(data) - pkn or fpeak[i] - pkn < pkn:
                        continue
                    for j in range(0, pkn):
                        while data[fpeak[i] + pkn - j] > data[fpeak[i]]:
                            if fpeak[i] + pkn - j >= len(data) - pkn:
                                break
                            fpeak[i] = fpeak[i] + pkn - j
                        while data[fpeak[i] - pkn + j] > data[fpeak[i]]:
                            if fpeak[i] - pkn + j < pkn:
                                break
                            fpeak[i] = fpeak[i] - pkn + j
                rec_data = np.copy(data)
                normalized_para_A = np.zeros([len(fpeak) - 1, 1])  # np.zeros 返回来一个给定形状和类型的用0填充的数组；\
                normalized_para_T = np.zeros([len(fpeak) - 1, 1])
                if eda_ch == True:
                    normalized_EDA = np.zeros([len(fpeak) - 1, 1])
                availability = np.ones([len(fpeak) - 1, 1], dtype=int)  # np.ones() 返回来一个给定形状和类型的用1填充的数组；
                normalized_para_beats = np.zeros([len(fpeak) - 1, 64])
                for nt in range(1, len(fpeak)):
                    temp = np.array(rec_data[fpeak[nt - 1]:(fpeak[nt] + 1)])
                    if len(temp) < 30 or len(temp) > 75:
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = 0
                        availability[nt - 1] = -1
                        print('????')
                        continue
                    t0 = temp[0]
                    ln = fpeak[nt] + 1 - fpeak[nt - 1]
                    normalized_para_T[nt - 1] = ln / fps
                    if eda_ch == True:
                        normalized_EDA[nt - 1] = np.mean(
                            eda_temp[fpeak[nt - 1] * 2:fpeak[nt] * 2])  # 求平均，得到归一化的 normalized_EDA
                    dy = (temp[-1] - t0) / (ln - 1)
                    for nj in range(0, ln):
                        temp[nj] = temp[nj] - dy * nj - t0
                    temp_peak_x = signal.argrelextrema(temp, np.greater)[0]  # 找极值
                    temp = -temp
                    if len(temp_peak_x) == 0:
                        temp_sth = temp.max()
                    else:
                        temp_peak_y = temp[temp_peak_x]
                        temp_sth = np.min(temp_peak_y)  # 求极小值
                    if temp.min() < -0.05 or temp.max() > 6 or temp.max() < 0.4 or (
                            temp_sth < 0.2 * temp.max() and len(temp_peak_x) > 1):
                        availability[nt - 1] = 0
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp.min()
                        normalized_para_A[nt - 1] = temp.max()
                    else:
                        normalized_para_A[nt - 1] = temp.max()
                        temp = temp / normalized_para_A[nt - 1]
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp[0:-1]
                        xnew = np.linspace(fpeak[nt - 1], fpeak[nt], 64)  # np.linspace() 主要用来创建等差数列
                        f = interpolate.interp1d(range(fpeak[nt - 1], fpeak[nt] + 1), temp,
                                                 kind="slinear")  # interpolate.interp1d() 用于完成一维数据的插值运算。
                        ynew = f(xnew)
                        normalized_para_beats[nt - 1] = ynew

                plt.figure(1)
                plt.subplot(2, 1, 1)
                xt = np.array(range(0, rec_data.shape[0]))
                xt = xt / 50
                plt.plot(xt, rec_data, label='raw')
                plt.plot(xt, data - np.mean(data) - 2, label='fdsc')
                # plt.plot(xt, ffds - np.mean(ffds) + 4, label='filt')
                avl = np.where(availability == 1)[0]
                # plt.plot(fpeak / 50, rec_data[fpeak], linestyle=':', marker='.', label='pk-s')
                avp = avl+1
                plt.plot(fpeak[avp]/50, rec_data[fpeak[avp]], linestyle=':',marker='.', label='pk-e')
                plt.legend()
                plt.grid()
                plt.subplot(2, 1, 2)
                plt.psd(rec_data, NFFT=256, Fs=fps, window=mlab.window_none,
                        scale_by_freq=True)
                plt.xlim([0, 5])
                plt.figure(2)
                plt.plot(availability, label='A')
                plt.show()
                # sss
                filename = subject  # 文件命名
                cofolder = BOUT + subject
                cutoutpath = cofolder + '/' + filename
                if not os.path.exists(cofolder):  # 如果不存在cofolder
                    os.makedirs(cofolder)  # 创建cofolder
                # split_path =
                print(filename + ' Saving Data_' + '{:02d}'.format(k) + '... ')  # 输出文件名
                np.savetxt(cutoutpath + '_rec_data_' + '{:02d}'.format(k) + '.txt', rec_data,
                           fmt='%.4f')  # 保存.txt文件（rec_data）
                np.savetxt(cutoutpath + '_para_A_' + '{:02d}'.format(k) + '.txt', normalized_para_A, fmt='%.4f')  # 以下同理
                np.savetxt(cutoutpath + '_para_T_' + '{:02d}'.format(k) + '.txt', normalized_para_T, fmt='%.4f')
                if eda_ch == True:
                    np.savetxt(cutoutpath + '_EDA_' + '{:02d}'.format(k) + '.txt', normalized_EDA, fmt='%.4f')
                np.savetxt(cutoutpath + '_para_beats_' + '{:02d}'.format(k) + '.txt', normalized_para_beats, fmt='%.4f')
                np.savetxt(cutoutpath + '_availability_' + '{:02d}'.format(k) + '.txt', availability, fmt='%d')
                np.savetxt(cutoutpath + '_fpeak_' + '{:02d}'.format(k) + '.txt', fpeak, fmt='%d')
    if Video_Read == True and Beats_Split == True and Data_label == False:
        # subjects = ['0501_01', '0501_02', '0501_03', '0502_01', '0502_02', '0502_03', '0502_04', '0502_05', '0502_06', '0502_07']
        # subjects = ['0501_01', '0501_02', '0501_03',
        #             '0502_01', '0502_02', '0502_03', '0502_04', '0502_05', '0502_06', '0502_07',
        #             '0509_01', '0509_02', '0509_03', '0509_04',
        #             '0510_01', '0510_02', '0510_03', '0510_04', '0510_06', '0510_07']
        subjects = ['0509_01']
        # subjects = ['0501_02', '0501_03', '0502_02']
        # labels = np.array([[3, 1, 3, 2, 3, 0],
        #                    [3, 0, 3, 2, 3, 1],
        #                    ])
        labels = np.array([[3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 2, 5, 0],
                           [3, 0, 4, 2, 5, 1],

                           [3, 0, 4, 2, 5, 1],
                           [3, 1, 4, 0, 5, 2],
                           [3, 0, 4, 2, 5, 1],
                           [3, 0, 4, 2, 5, 1],
                           [3, 2, 4, 0, 5, 1],
                           [3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 2, 5, 0],

                           [3, 0, 4, 2, 5, 1],
                           [3, 1, 4, 2, 5, 0],
                           [3, 0, 4, 2, 5, 1],
                           [3, 0, 4, 2, 5, 1],

                           [3, 2, 4, 0, 5, 1],
                           [3, 1, 4, 2, 5, 0],
                           [3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 2, 5, 0],
                           [3, 1, 4, 0, 5, 2],
                           [3, 1, 4, 0, 5, 2]])
        si = -1
        for subject in subjects:
            # if subject != subjects[1]:
            #     continue
            si = si + 1
            data = []
            fi = 0
            eda_ch = False
            cofolder = BOUT + subject
            cutoutpath = cofolder + '/' + subject
            while (True):
                rec_file = cutoutpath + '_rec_data_' + '{:02d}'.format(fi) + '.txt'
                para_A_file = cutoutpath + '_para_A_' + '{:02d}'.format(fi) + '.txt'
                para_T_file = cutoutpath + '_para_T_' + '{:02d}'.format(fi) + '.txt'
                if eda_ch == True:
                    EDA_file = cutoutpath + '_EDA_' + '{:02d}'.format(fi) + '.txt'
                para_beats_file = cutoutpath + '_para_beats_' + '{:02d}'.format(fi) + '.txt'
                avail_file = cutoutpath + '_availability_' + '{:02d}'.format(fi) + '.txt'
                fpeak_file = cutoutpath + '_fpeak_' + '{:02d}'.format(fi) + '.txt'

                if not os.path.exists(rec_file):
                    break
                else:
                    print(subject + ' loading Data_' + '{:02d}'.format(fi) + '... ')
                    rec_data = np.loadtxt(rec_file)  # 6000,
                    para_A = np.loadtxt(para_A_file)  # 147,
                    para_T = np.loadtxt(para_T_file)  # 147,
                    if eda_ch == True:
                        EDA = np.loadtxt(EDA_file)  # 147,
                    para_beats = np.loadtxt(para_beats_file)  # 147x64
                    availability = np.loadtxt(avail_file)
                    fpeak = np.loadtxt(fpeak_file)
                    avl = np.where(availability == 1)[0]
                    m_A = para_A[avl]  # signal.savgol_filter(para_A[avl],5,2)
                    m_T = para_T[avl]  # signal.savgol_filter(para_T[avl],5,2)
                    if eda_ch == True:
                        m_eda = EDA[avl]  #
                    nback = np.zeros(m_A.shape) + labels[si][fi]
                    if eda_ch == True:
                        m_data = np.vstack([m_A, m_T, para_beats[avl].T, m_eda, nback])  # 68x294
                    else:
                        m_data = np.vstack([m_A, m_T, para_beats[avl].T, nback])
                    m_data = m_data[:, 3:-3]
                    data.append(m_data)
                    print('data lenght: ' + str(len(data)))
                fi = fi + 1
            # rst = random.sample(range(10, len(fpeak) - 10), 30)
            # wn = 7
            # for ri in rst:
            #     while sum(availability[ri - wn:ri + wn]) < wn + 1:
            #         ri = random.randint(10, len(fpeak) - 10)
            #     temp_beats = np.array(para_beats[ri - wn:ri + wn, :])
            #     rbs = temp_beats[np.where(availability[ri - wn:ri + wn] == 1)[0], :].T
            #
            # plt.figure(1)
            # plt.subplot(2,2,1)
            # plt.plot(para_A[avl])
            # plt.subplot(2, 2, 2)
            # plt.plot(m_A)
            # plt.subplot(2, 2, 3)
            # plt.plot(para_T[avl])
            # plt.subplot(2, 2, 4)
            # plt.plot(m_T)
            # plt.figure(2)
            # plt.plot(para_beats[avl, :].T)
            # # plot_acf(rec_data).show()
            # # plot_pacf(rec_data).show()
            # rx = np.array(range(0,1000))
            # recy = np.sin(rx)+np.sin(2*rx)/2+np.sin(4*rx)/4
            # plot_acf(recy).show()
            # plot_pacf(recy).show()
            # sss

            # plt.figure()
            # plt.subplot(2,2,1)
            # plt.plot(m_A)
            # plt.grid()
            # plt.title('para_A '+subject+' '+vc)
            # plt.subplot(2,2,2)
            # plt.plot(m_T)
            # plt.grid()
            # plt.title('para_T '+subject+' '+vc)
            # plt.subplot(2,2,3)
            # plt.plot(p1_rbs)
            # plt.grid()
            # plt.title('base-beat '+subject+' '+vc)
            # plt.subplot(2,2,4)
            # plt.plot(m_data[2:,:])
            # plt.grid()
            # plt.title('pca_data '+subject+' '+vc)
            s_data = np.concatenate(data, 1)
            ldfolder = './data_ATB64_back'
            ldpath = ldfolder + '/data_' + subject
            if not os.path.exists(ldfolder):
                os.makedirs(ldfolder)
            print('subject ' + subject + ' Saving Data... ')
            np.savetxt(ldpath + '.txt', s_data, fmt='%.4f')
            print('subject ' + subject + ' Done... ')
    if Data_ob == False:  # 一般情况下开始的地方
        # subjects = ['0501_01', '0501_02', '0501_03', '0502_01', '0502_02', '0502_03', '0502_04', '0502_05', '0502_06','0502_07']
        subjects = ['0509_01']
        fps = 50
        tt = 164
        ttf = tt * fps
        std = np.array([[164, 284], [298, 418], [436, 556], [570, 690], [708, 828], [840, 960]]) - tt
        cutoff1 = [0.5, 5]
        for subject in subjects:
            if subject != '0509_01':  # [3, 1, 4, 2, 5, 0],
                continue
            sdata1, sdata2, sbg1, sbg2, src1, src2 = ReadOut(subject)
            # data - PDA[0,1][0] 3(BGR)*N
            # rc - PDA[0,1][1] N*5 para((x,y),(2b,2a),theta)
            # bg - PDA[0,1][2] 3(BGR)*N
            # ed - PDA[2] (8*N-1,)

            #eda_data = EDARead(subject)

            # 863 行与源代码不一样，有注释掉
            # 以下几行是武学长说要删掉的 这个数据视频里边没有
            # eda_time = eda_data[:, 0]
            # eda_tdata = eda_data[:, 1]
            # eda_cdata = eda_data[:, 2]

            data2 = sdata2[1][:]
            bg2 = sbg2[1][:]
            roi2 = src2[:, :]
            times = np.array(range(0, data2.shape[0])) / fps

            b_but, a_but = signal.butter(6, [2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps], 'bandpass')
            b_fir = signal.firwin2(256, [0, 2 * cutoff1[0] / fps, 2 * cutoff1[1] / fps, 1], [0, 1, 1, 0])
            cutoff2 = [0.7, 2]

            #与源代码不一样，有注释掉
            # 这块也是3.1注释掉的
            # times = np.array(range(0, data2.shape[0])) / fps
            fd2 = signal.filtfilt(b_but, a_but, data2)
            filt_data = signal.filtfilt(b_fir, 1, fd2)

            # smooth_filt_data = np.array(savgol(filt_data.tolist(), 7, 2))

            fig2 = plt.figure()
            ax1 = fig2.add_subplot(221)
            fd2 = signal.filtfilt(b_but, a_but, data2)
            fd2 = signal.filtfilt(b_fir, 1, fd2)
            fb2 = signal.filtfilt(b_but, a_but, bg2)
            fb2 = signal.filtfilt(b_fir, 1, fb2)
            roi2_fbut = signal.filtfilt(b_but, a_but, roi2.T)
            roi2_fbf = signal.filtfilt(b_fir, 1, roi2_fbut).T
            ax1.plot(times, fd2 - np.mean(fd2) + 2, label='face green channel', color='green')
            ax1.plot(times, fb2 - np.mean(fb2), label='backgroud', color='blue')
            ax1.grid()
            ax1.legend()
            ax3 = fig2.add_subplot(223)
            logfd2 = np.log(fd2 + 10)
            b_fir2 = signal.firwin2(512, [0, 2 * 0.5 / fps, 2 * 1 / fps, 1], [1, 1, 0, 0])
            lfd2 = signal.filtfilt(b_fir2, 1, data2)
            ax3.plot(times, data2)
            ax3.plot(times, lfd2)
            ax3.grid()
            ax4 = fig2.add_subplot(224)

            ax4.psd(fd2, NFFT=512, Fs=fps, window=mlab.window_none,
                    scale_by_freq=True)
            ax4.set_xlim([0, 5])
            ax2 = fig2.add_subplot(222)
            ax2.plot(times, roi2_fbf[:, 0] - np.mean(roi2_fbf[:, 0]) + 8, label='x')
            ax2.plot(times, roi2_fbf[:, 1] - np.mean(roi2_fbf[:, 1]) + 6, label='y')
            ax2.plot(times, roi2_fbf[:, 2] - np.mean(roi2_fbf[:, 2]) + 8, label='2b')
            ax2.plot(times, roi2_fbf[:, 3] - np.mean(roi2_fbf[:, 3]) + 6, label='2a')
            ax2.plot(times, roi2_fbf[:, 4] - np.mean(roi2_fbf[:, 4]) + 6, label='th')
            ax2.grid()
            ax2.legend()
            # sss
            #stage_beats_split(data2, subject, pfn)
            # sss
            fd_scale = []
            eda_tdata_sel = []
            eda_cdata_sel = []
            # 这块也是3.1注释掉的
            # for sti in std:
            # ti_eda = (eda_time > sti[0]) & (eda_time < sti[1])
            # tempt = eda_tdata[ti_eda]
            # tempc = eda_cdata[ti_eda]
            # eda_tdata_sel.append(tempt)
            # eda_cdata_sel.append(tempc)
            # eda_tscale = (eda_tdata - np.min(eda_tdata_sel)) / (np.max(eda_tdata_sel) - np.min(eda_tdata_sel))
            # eda_cscale = (eda_cdata - np.min(eda_cdata_sel)) / (np.max(eda_cdata_sel) - np.min(eda_cdata_sel))
            # eda_stage_scale = []
            for sti in std:
                #print(sti)
                temp_data = filt_data[sti[0] * fps:sti[1] * fps]
                # ti_eda = (eda_time > sti[0]) & (eda_time < sti[1])
                # tempt = eda_tscale[ti_eda]
                # tempc = eda_cscale[ti_eda]
                # eda_stage_scale.append(tempt)
                fd_scale.append(temp_data)
            #fd_scale = preprocessing.robust_scale(filt_data.reshape(-1, 1))[:].T[0]
            data_stage_scale = preprocessing.robust_scale(list(map(list, zip(*fd_scale))))
            # 以下这些行（到974）是因为缺少eda_stage_scale，上边应该注释掉了 所以这块也注释掉 3.15
            # eda_stage_scale = np.array(eda_stage_scale).T
            b_c1, a_c1 = signal.butter(6, [2 * cutoff2[0] / fps, 2 * cutoff2[1] / fps], 'bandpass')
            for k in range(0, data_stage_scale.shape[1]):
                if k != 3:
                    continue
                data_temp = data_stage_scale[:, k]
            # eda_temp = eda_stage_scale[:, k]
            # 这是源代码的注释 f ,t ,Sxx = signal.spectrogram(data_temp, fs=50, window='', nperseg=1024, noverlap=512)
            # 这是源代码的注释 ssss
                ffds = signal.filtfilt(b_c1, a_c1, data_temp)
                fpeak = signal.argrelextrema(ffds, np.greater)[0]
                data = np.array(data_temp)

                for i in range(0, len(fpeak)):
                    pkn = 3
                    if fpeak[i] + pkn >= len(data) - pkn or fpeak[i] - pkn < pkn:
                        continue
                    for j in range(0, pkn):
                        while data[fpeak[i] + pkn - j] > data[fpeak[i]]:
                            if fpeak[i] + pkn - j >= len(data) - pkn:
                                break
                            fpeak[i] = fpeak[i] + pkn - j
                        while data[fpeak[i] - pkn + j] > data[fpeak[i]]:
                            if fpeak[i] - pkn + j < pkn:
                                break
                            fpeak[i] = fpeak[i] - pkn + j
#修改到此 4.2
                rec_data = np.copy(data)
                normalized_para_A = np.zeros([len(fpeak) - 1, 1])
                normalized_para_T = np.zeros([len(fpeak) - 1, 1])
                normalized_EDA = np.zeros([len(fpeak) - 1, 1])
                availability = np.ones([len(fpeak) - 1, 1], dtype=int)
                normalized_para_beats = np.zeros([len(fpeak) - 1, 64])
                for nt in range(1, len(fpeak)):
                    temp = np.array(rec_data[fpeak[nt - 1]:(fpeak[nt] + 1)])
                    if len(temp) < 20 or len(temp) > 80:
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = 0
                        availability[nt - 1] = -1
                # 源代码注释 print('{%d,%d}: ????' % (k, nt))
                        continue
                    t0 = temp[0]
                    ln = fpeak[nt] + 1 - fpeak[nt - 1]
                    normalized_para_T[nt - 1] = ln / fps
                    # normalized_EDA[nt - 1] = np.mean(eda_temp[fpeak[nt - 1] * 2:fpeak[nt] * 2])
                    dy = (temp[-1] - t0) / (ln - 1)
                    for nj in range(0, ln):
                        temp[nj] = temp[nj] - dy * nj - t0
                    temp_peak_x = signal.argrelextrema(temp, np.greater)[0]
                    temp = -temp
                    if len(temp_peak_x) == 0:
                        temp_sth = temp.max()
                    else:
                        temp_peak_y = temp[temp_peak_x]
                        temp_sth = np.min(temp_peak_y)
                    if temp.min() < -0.05 or temp.max() > 4 or temp.max() < 0.4 or (
                            temp_sth < 0.3 * temp.max() and len(temp_peak_x) > 1):
                        availability[nt - 1] = 0
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp.min()
                        normalized_para_A[nt - 1] = temp.max()
                        print('{%.4f,%.4f, %.4f}: ????' % (temp[-1], temp.max(), temp.min()))
                    else:
                        normalized_para_A[nt - 1] = temp.max()
                        temp = temp / normalized_para_A[nt - 1]
                        rec_data[fpeak[nt - 1]:fpeak[nt]] = temp[0:-1]
                        xnew = np.linspace(fpeak[nt - 1], fpeak[nt], 64)
                        f = interpolate.interp1d(range(fpeak[nt - 1], fpeak[nt] + 1), temp, kind="slinear")
                        ynew = f(xnew)
                        normalized_para_beats[nt - 1] = ynew

            # 下边一直到结尾 都往前删了一行Tab 因为上边注释掉的之后 这个有缩进问题 3.15
            # 又因为缺少rec_data 和 availability底下相关的都注释掉了
                plt.figure(1)
                plt.subplot(2, 1, 1)
                xt = np.array(range(0, rec_data.shape[0]))
                xt = xt / 50
                # plt.plot(xt, rec_data, label='rec')
                plt.plot(xt, - data + np.mean(data), label='FiltData')
                # plt.plot(xt, ffds - np.mean(ffds) + 1, label='filt')
                avl = np.where(availability == 1)[0]
                # plt.plot(fpeak / 50, rec_data[fpeak], linestyle=':', marker='.', label='pk-s')
                avp = avl + 1
                # plt.plot(fpeak[avp] / 50, rec_data[fpeak[avp]], linestyle=':', marker='.', label='pk-e')
                plt.legend()
                plt.grid()
                plt.subplot(2, 1, 2)
                plt.psd(rec_data, NFFT=256, Fs=fps, window=mlab.window_none,
                scale_by_freq=True)
                plt.xlim([0, 5])
                plt.figure(2)
                plt.plot(availability, label='A')
                plt.show()

