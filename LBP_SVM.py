# -*- coding=utf-8 -*-
import glob
import platform
import time
from PIL import Image
from skimage.feature import hog
import numpy as np
import os
import joblib
from sklearn.svm import LinearSVC, SVC
import shutil
import sys
import cv2
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, precision_recall_curve, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from skimage.feature import *

# 第一个是你的类别   第二个是类别对应的名称   输出结果的时候方便查看
label_map = {0: 'background',
             1: 'foreground',
             }
target_name = ['background','foreground']
# 训练集图片的位置
train_image_path = r'G:\paddle\HOG_SVM-master'
# 测试集图片的位置
test_image_path = r'G:\paddle\HOG_SVM-master'

labels_file = 'labels.txt'

# 训练集标签的位置
# train_label_path = os.path.join('image128','train.txt')
train_label_path = 'train_list.txt'
# 测试集标签的位置
# test_label_path = os.path.join('image128','train.txt')
test_label_path = 'val_list.txt'

image_height = 224
image_width = 224

train_feat_path = 'LBP_train/'
test_feat_path = 'LBP_test1/'
model_path = 'LBP_model/'


# 获得图片列表
def get_image_list(filePath, nameList):
    print('read image from ', filePath)
    img_list = []
    for name in nameList:
        temp = Image.open(os.path.join(filePath, name))
        img_list.append(temp.copy())
        temp.close()
    return img_list


# 获得图片列表
def get_image_list_from_label_file(image_path, label_file_path):
    imgs_lists = []
    gt_labels = []
    img_name_lists = []
    with open(label_file_path, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            image_name, label = line.strip("\n").split()
            label = int(label)
            imgs_lists.append(os.path.join(image_path, image_name))
            gt_labels.append(int(label))

    return imgs_lists, gt_labels


# 提取特征并保存
def get_feat(image_list, label_list, save_Path):
    # i = 0
    esp = 1e-7
    n_points = 8
    radius = 2
    # file1 = os.listdir(image_list)
    # for image,label in zip(image_list,label_list):
    for index, img in enumerate(image_list):
        # try:
        #     # 如果是灰度图片  把3改为-1
        #     image = np.reshape(image, (image_height, image_width, 3))
        # except:
        #     print('发送了异常，图片大小size不满足要求：',name_list[i])
        #     continue
        # gray = rgb2gray(image) / 255.0
        image_name = os.path.split(img)[1].split('.')[0]
        # image_name = img_name[1].split('.')[0]
        image = cv2.imread(img)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 这句话根据你的尺寸改改
        lbp = local_binary_pattern(gray, n_points, radius)
        # (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        # hist = hist.astype("float")
        # hist /= (hist.sum() + esp)
        max_bins = int(lbp.max() + 1)
        # hist size:256
        hist, _ = np.histogram(lbp, normed=True, bins=max_bins, range=(0, max_bins))
        # fd = hog(gray, orientations=9, block_norm='L1', pixels_per_cell=[32, 32], cells_per_block=[4, 4],
        #          visualize=False,
        #          transform_sqrt=True)
        # fd = np.concatenate((fd, [label_list[i]]))
        fd = np.concatenate((hist, [label_list[index]]))
        # fd_name = name_list[i] + '.feat'
        fd_name = image_name + '_' + str(index) + '.feat'
        fd_path = os.path.join(save_Path, fd_name)
        joblib.dump(fd, fd_path)
        # i += 1
    print("Test features are extracted and saved.")


# 变成灰度图片
def rgb2gray(im):
    gray = im[:, :, 0] * 0.2989 + im[:, :, 1] * 0.5870 + im[:, :, 2] * 0.1140
    return gray


# 获得图片名称与对应的类别
def get_name_label(file_path):
    print("read label from ", file_path)
    name_list = []
    label_list = []
    with open(file_path) as f:
        for line in f.readlines():
            # 一般是name label  三部分，所以至少长度为3  所以可以通过这个忽略空白行
            if len(line) >= 3:
                name_list.append(line.split(' ')[0])
                label_list.append(line.split(' ')[1].replace('\n', '').replace('\r', ''))
                if not str(label_list[-1]).isdigit():
                    print("label必须为数字，得到的是：", label_list[-1], "程序终止，请检查文件")
                    exit(1)
    return name_list, label_list


# 提取特征
def extra_feat():
    train_name, train_label = get_name_label(labels_file)
    test_name, test_label = get_name_label(labels_file)

    # train_image = get_image_list(train_image_path, train_name)
    # test_image = get_image_list(test_image_path, test_name)
    train_image, train_gt_labels = get_image_list_from_label_file(train_image_path, train_label_path)
    test_image, test_gt_labels = get_image_list_from_label_file(test_image_path, test_label_path)
    get_feat(train_image, train_gt_labels, train_feat_path)
    get_feat(test_image, test_gt_labels, test_feat_path)


# 创建存放特征的文件夹
def mkdir():
    if not os.path.exists(train_feat_path):
        os.mkdir(train_feat_path)
    if not os.path.exists(test_feat_path):
        os.mkdir(test_feat_path)


# 训练和测试
# def train_and_test():
#     t0 = time.time()
#     features = []
#     labels = []
#     correct_number = 0
#     total = 0
#     for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
#         data = joblib.load(feat_path)
#         features.append(data[:-1])
#         labels.append(data[-1])
#     print("Training a Linear LinearSVM Classifier.")
#     clf = LinearSVC()
#     clf.fit(features, labels)
#     # 下面的代码是保存模型的
#     if not os.path.exists(model_path):
#         os.makedirs(model_path)
#     joblib.dump(clf, model_path + 'model')
#     # 下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
#     # clf = joblib.load(model_path+'model')
#     print("训练之后的模型存放在model文件夹中")
#     # exit()
#     result_list = []
#     for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
#         total += 1
#         if platform.system() == 'Windows':
#             symbol = '\\'
#         else:
#             symbol = '/'
#         image_name = feat_path.split(symbol)[1].split('.feat')[0]
#         data_test = joblib.load(feat_path)
#         data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
#         result = clf.predict(data_test_feat)
#         result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
#         if int(result[0]) == int(data_test[-1]):
#             correct_number += 1
#     write_to_txt(result_list)
#     rate = float(correct_number) / total
#     t1 = time.time()
#     print('准确率是： %f' % rate)
#     print('耗时是 : %f' % (t1 - t0))

def train():
    # t0 = time.time()
    features = []
    labels = []
    # correct_number = 0
    # total = 0
    for feat_path in glob.glob(os.path.join(train_feat_path, '*.feat')):
        data = joblib.load(feat_path)
        features.append(data[:-1])
        labels.append(data[-1])
    print("Training a Linear LinearSVM Classifier.")
    clf = LinearSVC()
    clf.fit(features, labels)
    # 下面的代码是保存模型的
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    joblib.dump(clf, model_path + 'model')
    # 下面的代码是加载模型  可以注释上面的代码   直接进行加载模型  不进行训练
    # clf = joblib.load(model_path+'model')
    print("训练之后的模型存放在model文件夹中")
    # exit()
    # result_list = []
    # for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
    #     total += 1
    #     if platform.system() == 'Windows':
    #         symbol = '\\'
    #     else:
    #         symbol = '/'
    #     image_name = feat_path.split(symbol)[1].split('.feat')[0]
    #     data_test = joblib.load(feat_path)
    #     data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
    #     result = clf.predict(data_test_feat)
    #     result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
    #     if int(result[0]) == int(data_test[-1]):
    #         correct_number += 1
    # write_to_txt(result_list)
    # rate = float(correct_number) / total
    # t1 = time.time()
    # print('准确率是： %f' % rate)
    # print('耗时是 : %f' % (t1 - t0))


def test():
    test_feat_list = []
    test_gt_labels = []
    correct_number = 0
    total = 0
    clf = joblib.load(model_path + 'model')
    # exit()
    result_list = []
    test_res = []
    for feat_path in glob.glob(os.path.join(test_feat_path, '*.feat')):
        total += 1
        if platform.system() == 'Windows':
            symbol = '\\'
        else:
            symbol = '/'
        image_name = feat_path.split(symbol)[1].split('.feat')[0]
        data_test = joblib.load(feat_path)
        test_gt_labels.append(int(data_test[-1]))
        data_test_feat = data_test[:-1].reshape((1, -1)).astype(np.float64)
        test_feat_list.append(data_test_feat)
        result = clf.predict(data_test_feat)
        test_res.append(int(result[0]))
    test_gt_labels_np = np.array(test_gt_labels)
    test_res_np = np.array(test_res)
        # result_list.append(image_name + ' ' + label_map[int(result[0])] + '\n')
    #     if int(result[0]) == int(data_test[-1]):
    #         correct_number += 1
    # print('预测正确数量为:', correct_number)
    # write_to_txt(result_list)
    # rate = float(correct_number) / total


    # tn, fp, fn, tp = confusion_matrix(test_gt_labels_np, test_res_np).ravel()
    # C2 = confusion_matrix(test_gt_labels_np, test_res_np)
    # print('使用SVM预测的准确率为：', C2)
    # print('使用SVM预测的准确率为：', accuracy_score(test_gt_labels, test_res))
    # print('使用SVM预测的精确率为：',
    #       precision_score(test_gt_labels, test_res, average=None))
    # print('使用SVM预测的召回率为：',
    #       recall_score(test_gt_labels, test_res, average=None))
    # print('使用SVM预的F1值为：',
    #       f1_score(test_gt_labels, test_res, average=None))
    # print('使用SVM预测的分类报告为：', '\n',
    #       classification_report(test_gt_labels, test_res, target_names=target_name, digits=4))
    t1 = time.time()
    # print('精确率是： %f' % rate)
    print('耗时是 : %f' % (t1 - t0))
    # test_feat = np.array(test_feat_list).reshape((1562,2304))
    #
    # y_pro = clf.decision_function(test_feat)
    # test_res = np.array(test_res)
    # fpr, tpr, thresholds = roc_curve(test_res, y_pro)
    # plt.figure(figsize=(10, 6))
    # plt.xlim(0, 1)  ##设定x轴的范围
    # plt.ylim(0.0, 1.1)  ## 设定y轴的范围
    # plt.xlabel('False Postive Rate')
    # plt.ylabel('True Postive Rate')
    # plt.plot(fpr, tpr, linewidth=2, linestyle="-", color='red')
    # plt.show()

    # plot_PR(test_feat,test_res)


def write_to_txt(list):
    with open('result.txt', 'w') as f:
        f.writelines(list)
    print('每张图片的识别结果存放在result.txt里面')


def plot_PR(x_test, y_test):
    clf = joblib.load(model_path + 'model')
    # y_pro = clf._predict_proba_lr(x_test)
    y_pro = clf.decision_function(x_test)
    precision, recall, thresholds = precision_recall_curve(y_test, y_pro, pos_label=1)
    # average_precision = average_precision_score(y_test, y_pro[:, 1])
    # ax2 = plt.subplot(224)
    # ax2.set_title("Precision_Recall Curve AP=%0.2f" % average_precision, verticalalignment='center')
    # plt.step(precision, recall, where='post', alpha=0.2, color='r')
    plt.fill_between(recall, precision, step='post', alpha=0.2, color='b')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.plot(recall, precision)
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()




if __name__ == '__main__':
    # mkdir()  # 不存在文件夹就创建
    # # need_input = input('是否手动输入各个信息？y/n\n')
    #
    # # if need_input == 'y':
    # #     train_image_path = input('请输入训练图片文件夹的位置,如 /home/icelee/image\n')y
    # #     test_image_path = input('请输入测试图片文件夹的位置,如 /home/icelee/image\n')
    # #     train_label_path = input('请输入训练集合标签的位置,如 /home/icelee/train.txt\n')
    # #     test_label_path = input('请输入测试集合标签的位置,如 /home/icelee/test1.txt\n')
    # #     size = int(input('请输入您图片的大小：如64x64，则输入64\n'))
    # if sys.version_info < (3,):
    #     need_extra_feat = raw_input('是否需要重新获取特征？y/n\n')
    # else:
    #     need_extra_feat = input('是否需要重新获取特征？y/n\n')
    # if need_extra_feat == 'y':
    #     # shutil.rmtree(train_feat_path)
    #     # shutil.rmtree(test_feat_path)
    mkdir()
    #     extra_feat()  # 获取特征并保存在文件夹
    #     train()  # 训练并预测
    # else:
    #     train()
    t0 = time.time()
    test_image, test_gt_labels = get_image_list_from_label_file(test_image_path, test_label_path)
    get_feat(test_image, test_gt_labels, test_feat_path)
    test()
