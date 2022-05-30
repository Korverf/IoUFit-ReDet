import cv2
import os
import numpy as np
import random
import albumentations as A
import matplotlib.pyplot as plt


def verticalFlip(img, bbox):
    aug_ver = A.VerticalFlip(p=1)
    aug_img = aug_ver(image=img)["image"]
    for i in range(bbox.shape[0]):
        for j in range(bbox.shape[1]):
            bbox[i][j][1] = height - bbox[i][j][1]

    return aug_img, bbox


def horizontalFlip(img, bbox):
    aug_hor = A.HorizontalFlip(p=1)
    aug_img = aug_hor(image=img)["image"]
    for i in range(bbox.shape[0]):
        for j in range(bbox.shape[1]):
            bbox[i][j][0] = width - bbox[i][j][0]

    return aug_img, bbox


def warpAffine(img, bbox):
    aug_img = cv2.warpAffine(img, m, dsize=(width, height), borderValue=(0, 0, 0))
    for num in range(bbox.shape[0]):
        a = np.append(bbox[num].T, [1, 1, 1, 1])
        a = a.reshape((3, 4))
        bbox[num] = np.dot(m, a).transpose((1, 0))

    return aug_img, bbox


def simpleResize(img, bbox, scale):
    aug_img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    bbox = bbox * scale

    return aug_img, bbox


def Blur(img):
    aug_blur = A.Blur(blur_limit=7, p=1)
    aug_img = aug_blur(image=img)["image"]

    return aug_img


def hueSaturationValue(img):
    aug_hsv = A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=20, val_shift_limit=20, p=1)
    aug_img = aug_hsv(image=img)["image"]

    return aug_img


def randomBrightnessContrast(img):
    aug_BC = A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1)
    aug_img = aug_BC(image=img)["image"]

    return aug_img


def gaussianBlur(img):
    aug_gBlur = A.GaussianBlur(blur_limit=(7, 7), sigma_limit=5, p=1)
    aug_img = aug_gBlur(image=img)["image"]

    return aug_img


def randomFog(img):
    aug_fog = A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1)
    aug_img = aug_fog(image=img)["image"]

    return aug_img


def visualize(image, bboxes):
    img = image.copy()
    cv2.polylines(img, np.int32(bboxes), True, BOX_COLOR, thickness=1)
    cv2.imshow("data_augmentation", img)
    cv2.waitKey(0)
    return img


def saveData(img, bbox, type, imgOutPath, labelOutPath, name, num):
    cv2.imwrite(imgOutPath + str(num) + "0" + name + "_new.jpg", img)
    n = bbox.shape[0] * bbox.shape[1] * bbox.shape[2]
    bbox = bbox.flatten().reshape(int(n / 8), 8)
    with open(labelOutPath + "1_" + name + "_new.txt", "w") as f:
        for i in range(bbox.shape[0]):
            x1 = bbox[i][0]
            y1 = bbox[i][1]
            x2 = bbox[i][2]
            y2 = bbox[i][3]
            x3 = bbox[i][4]
            y3 = bbox[i][5]
            x4 = bbox[i][6]
            y4 = bbox[i][7]
            f.write("{} {} {} {} {} {} {} {} {}\n".format(x1, y1, x2, y2, x3, y3, x4, y4, type))


if __name__ == "__main__":

    imgPath = "D:/pythonWorkplace/test/toolbox/smallTestDataset/1024Image/"
    labelPath = "D:/pythonWorkplace/test/toolbox/smallTestDataset/1024Label/"
    imgOutPath = "D:/pythonWorkplace/test/toolbox/smallTestDataset/1024NewImage/"
    labelOutPath = "D:/pythonWorkplace/test/toolbox/smallTestDataset/1024NewLabel/"
    name_list = os.listdir(imgPath)

    # 扩充次数
    expand_times = 1
    # 变化参数
    BOX_COLOR = (0, 0, 255)
    TEXT_COLOR = (255, 255, 255)
    width = 1024
    height = 1024
    # 仿射矩阵
    m = cv2.getRotationMatrix2D(center=(width // 2, height // 2), angle=-30, scale=0.5)
    # 尺度变换
    scale = 0.3
    # 是否显示变化后图片
    isVisualize = True
    # 是否保存
    isSave = False

    # aug_dictionary = {
    #     "垂直翻转" : verticalFlip,
    #     "水平翻转" : horizontalFlip,
    #     "仿射变换" : warpAffine,
    #     "尺度变换" : resize，
    #     "随机模糊" : blur,
    #     "随机hsv变化" : hueSaturationValue,
    #     "随机亮度对比度" : randomBrightnessContrast,
    #     "随机高斯模糊" : gaussianBlur,
    #     "随机雾" : randomFog
    # }
    # aug_list = ["verticalFlip", "horizontalFlip","randomFog"]

    # 增强方式
    # list没有加仿射变换，因为对global会有影响
    aug_lists = ["verticalFlip", "horizontalFlip", "resize", "blur", "hueSaturationValue",
                 "randomBrightnessContrast", "gaussianBlur", "randomFog"]
    # 每次扩充选择增强种类
    aug_num = 3

    print("初始数据集大小:{}".format(len(name_list)))
    for num in range(expand_times):
        aug_list = random.sample(aug_lists, aug_num)
        print("已选择增强方式：{}".format(aug_list))
        for i in name_list:
            name = i[:-4]
            txtPath = labelPath + name + '.txt'
            img = cv2.imread(imgPath + i)
            with open(txtPath, 'r') as fread:
                lines = fread.readlines()
                nplines = []
                # read lines
                for line in lines:
                    line = line.split()
                    fourPoints = np.array(line[:8], dtype=np.float32)
                    type = line[-1]
                    nplines.append(fourPoints[np.newaxis])
                nplines = np.concatenate(nplines, 0).reshape(-1, 4, 2)

            # 垂直翻转
            if "verticalFlip" in aug_list:
                img, nplines = verticalFlip(img, nplines)
            # 水平翻转
            if "horizontalFlip" in aug_list:
                img, nplines = horizontalFlip(img, nplines)
            # 仿射变换
            if "warpAffine" in aug_list:
                img, nplines = warpAffine(img, nplines)
            # 尺度变换
            if "resize" in aug_list:
                img, nplines = simpleResize(img, nplines, scale)
            # 随机模糊
            if "blur" in aug_list:
                img = Blur(img)
            # 随机hsv变化
            if "hueSaturationValue" in aug_list:
                img = hueSaturationValue(img)
            # 随机亮度
            if "randomBrightnessContrast" in aug_list:
                img = randomBrightnessContrast(img)
            # 随机高斯模糊
            if "gaussianBlur" in aug_list:
                img = gaussianBlur(img)
            # 随机雾
            if "randomFog" in aug_list:
                img = randomFog(img)

            if isSave:
                saveData(img, nplines, type,imgOutPath,labelOutPath,name,num)
            if isVisualize:
                visualize(img, nplines)

    print("扩充的数据集大小：{}".format(len([lists for lists in os.listdir(imgOutPath)])))
