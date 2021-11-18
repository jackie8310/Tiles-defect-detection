import os
import sys
import time

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.abspath(os.path.join(__dir__, '../')))

import cv2
import numpy as np
import os.path as osp

from utils import logger
from utils import config
from utils.predictor import Predictor
from utils.get_image_list import get_image_list
from utils.preprocess import create_operators
from scipy.spatial.distance import cosine


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class RecPredictor(Predictor):
    def __init__(self, config):
        super().__init__(config["Global"],
                         config["Global"]["rec_inference_model_dir"])
        self.preprocess_ops = create_operators(config["RecPreProcess"][
                                                   "transform_ops"])

    def predict(self, images, feature_normalize=True):
        input_names = self.paddle_predictor.get_input_names()
        input_tensor = self.paddle_predictor.get_input_handle(input_names[0])

        output_names = self.paddle_predictor.get_output_names()
        output_tensor = self.paddle_predictor.get_output_handle(output_names[0])

        if not isinstance(images, (list,)):
            images = [images]
        for idx in range(len(images)):
            for ops in self.preprocess_ops:
                images[idx] = ops(images[idx])
        image = np.array(images)

        input_tensor.copy_from_cpu(image)
        self.paddle_predictor.run()
        batch_output = output_tensor.copy_to_cpu()

        if feature_normalize:
            feas_norm = np.sqrt(
                np.sum(np.square(batch_output), axis=1, keepdims=True))
            batch_output = np.divide(batch_output, feas_norm)

        return batch_output




def main(config):
    t0 = time.time()
    rec_predictor = RecPredictor(config)
    # image_list = get_image_list(config["Global"]["infer_imgs"])
    output_des = []
    output_tem = []
    des_list = []
    tem_list = []
    file1 = os.listdir(despath)
    file2 = os.listdir(tempath)
    threshold = 0.5
    pt = 0

    for fl1, fl2 in zip(file1, file2):
        des_img = cv2.imread(osp.join(despath, fl1))[:, :, ::-1]
        tmp_img = cv2.imread(osp.join(tempath, fl2))[:, :, ::-1]
        output1 = rec_predictor.predict(des_img)
        output2 = rec_predictor.predict(tmp_img)
        op3 = cosine(output1, output2)
    #     if op3 < threshold:
    #         name1 = fl1.split('.')[0]
    #         name2 = fl2.split('.')[0]
    #         pt += 1
    #         # print(op3)
    #         print('{}->{} 的余弦距离为:{}'.format(name1, name2, op3))
    # print('误检数量为:{}'.format(pt))
    t1 = time.time()

    print('耗时是 : %f' % (t1 - t0))


    # for file1 in os.listdir(despath):
    #     des_img = cv2.imread(osp.join(despath, file1))
    #     output1 = rec_predictor.predict(des_img)
    #     output_des.append(output1)
    #     des_list.append(file1.split('.')[0])
    # for file2 in os.listdir(tempath):
    #     des_img = cv2.imread(osp.join(tempath, file2))
    #     output2 = rec_predictor.predict(des_img)
    #     output_tem.append(output2)
    #     tem_list.append(file2.split('.')[0])
    # if len(output_des) == len(output_tem):
    #     for index in range(len(output_tem)):
    #         vector1 = output_des[index]
    #         vector2 = output_tem[index]
    #         # op2 = np.linalg.norm(vector1 - vector2)
    #         op3 = cosine(vector1, vector2)
    #         # print("{} 的欧式距离为:{}".format(des_list[index], op2))
    #
    #         print('{} 的余弦距离为:{}'.format(des_list[index], op3))
    # else:
    #     print('不能比较！')



if __name__ == "__main__":
    despath = r'G:\Dataset\template_dateset\datasets_tem\1'
    tempath = r'G:\Dataset\template_dateset\datasets_tem\5'
    args = config.parse_args()
    config = config.get_config(args.config, overrides=args.override, show=False)
    main(config)
