import paddle
from infer import Detector, predict_image, Config, visualize
import os
import os.path as osp
import cv2
import numpy as np
import sys
import PIL.Image as Image
import time


# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass

if __name__ == "__main__":
    model_dir = r'G:\paddle\detection0.5bate\output\0419amodel_18000'
    output_dir = r'C:\Users\Administrator\Desktop\2021-10-01\src\res'
    data_dir = r'C:\Users\Administrator\Desktop\2021-10-01\src'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    config = Config(model_dir)
    labels = config.labels
    threshold = 0.6
    detector = Detector(config=config, model_dir=model_dir, use_gpu=True, threshold=threshold)

    for im_file in os.listdir(data_dir):
        if not im_file.endswith('.jpg'):
            continue
        im_file = im_file.strip()
        # sys.stdout = Logger(r'F:\paddle\detection0.5bate\output\a.txt')
        print(im_file + '\n')
        im_file = osp.join(data_dir, im_file)
        # rgb_img = Image.open(im_file).convert('RGB')
        # cv_img = np.asarray(rgb_img)
        bgr_img = cv2.imread(im_file)
        cv_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
        h, w, c = cv_img.shape
        if h < 2048 or w < 2448:
            continue
        result = []
        img1 = cv_img[0:1024, 0:1224, :]
        result1 = detector.predict(img1, run_benchmark=False, threshold=threshold)
        boxes1 = result1['boxes']
        if len(boxes1) > 0:
            for box in boxes1:
                # print('class_id:{:s}, confidence:{:.4f},'
                #       'left_top:[{:.2f},{:.2f}],'
                #       ' right_bottom:[{:.2f},{:.2f}]'.format(
                #     labels[int(box[0])], box[1], box[2], box[3], box[4], box[5]))
                print(labels[int(box[0])], box[2], box[3], box[4], box[5])
                result.append(box)
        img2 = cv_img[1024:, 0:1224, :]
        result2 = detector.predict(img2, run_benchmark=False, threshold=threshold)
        boxes2 = result2['boxes']
        if len(boxes2) > 0:
            boxes2[:, 3] += 1024
            boxes2[:, 5] += 1024
            for box in boxes2:
                # print('class_id:{:s}, confidence:{:.4f},'
                #       'left_top:[{:.2f},{:.2f}],'
                #       ' right_bottom:[{:.2f},{:.2f}]'.format(
                #     labels[int(box[0])], box[1], box[2], box[3], box[4], box[5]))
                # print(labels[int(box[0])], box[2], box[3], box[4], box[5])
                result.append(box)
        img3 = cv_img[0:1024, 1224:, :]
        result3 = detector.predict(img3, run_benchmark=False, threshold=threshold)
        boxes3 = result3['boxes']
        if len(boxes3) > 0:
            boxes3[:, 2] += 1224
            boxes3[:, 4] += 1224
            for box in boxes3:
                # print('class_id:{:s}, confidence:{:.4f},'
                #       'left_top:[{:.2f},{:.2f}],'
                #       ' right_bottom:[{:.2f},{:.2f}]'.format(
                #     labels[int(box[0])], box[1], box[2], box[3], box[4], box[5]))
                # print(labels[int(box[0])], box[2], box[3], box[4], box[5])
                result.append(box)
        img4 = cv_img[1024:, 1224:, :]
        result4 = detector.predict(img4, run_benchmark=False, threshold=threshold)
        boxes4 = result4['boxes']
        if len(boxes4) > 0:
            boxes4[:, 2] += 1224
            boxes4[:, 3] += 1024
            boxes4[:, 4] += 1224
            boxes4[:, 5] += 1024
            for box in boxes4:
                # print('class_id:{:s}, confidence:{:.4f},'
                #       'left_top:[{:.2f},{:.2f}],'
                #       ' right_bottom:[{:.2f},{:.2f}]'.format(
                #     labels[int(box[0])], box[1], box[2], box[3], box[4], box[5]))
                # print(labels[int(box[0])], box[2], box[3], box[4], box[5])
                result.append(box)
        drw_result = {'boxes': np.array([])}
        if len(result) > 0:
            drw_result['boxes'] = np.array(result)
            added_image = visualize(im_file, drw_result, threshold=threshold, labels=detector.config.labels,
                                    output_dir=output_dir)
        # result = detector.predict(im_file, run_benchmark=False, threshold=threshold)

        # if len(result['boxes']) > 0:
        # print(result)
        # save added image
        # added_image = visualize(im_file, drw_result, threshold=threshold, labels=detector.config.labels,
        #                         output_dir=output_dir)