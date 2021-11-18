import os
import time

import numpy as np
import cv2
import paddle
from scipy.spatial.distance import cosine
from src.PaddleInference import Detector, RectBoxPredictor
from src.TileTemplateRepository import TemplateRepository
from src.TileDefectsFilter import TileEdgeFilter





class TileDetectorObject(QObject):
    signal_output_result = pyqtSignal(object, object, int, object)  

    def __init__(self):
        super(TileDetectorObject, self).__init__()

        self.debug_mode = False

    def detector_init(self, model_dir, cfg_dir):
        self.model_dir = model_dir
        self.detector = Detector(model_dir)
        self.labels_id = self.detector.labels
        self.detector_local_threshold_data_1 = self.detector.local_threshold_dict
        self.detector_global_threshold_1 = self.detector.global_threshold

        self.detector_local_threshold_data_2 = {}
        self.detector_global_threshold_2 = float(self.detector_global_threshold_1) / 4
        for key in self.detector_local_threshold_data_1.keys():
            self.detector_local_threshold_data_2[key] = float(self.detector_local_threshold_data_1[key]) / 4


        pre_detect_img = cv2.imread(os.path.join(cfg_dir, 'pre_detect.jpg'))
        self.detector.predict(pre_detect_img)

        self.rect_box_detector = RectBoxPredictor(os.path.join(model_dir, 'MobileNetV1_infer'))
        self.rect_box_detector.predict(pre_detect_img)

        self.detect_stop = True


    def filter_init(self, config_data):
        self.config_data = config_data['filter']

        self.template_repository = TemplateRepository(self.model_dir, self.config_data)

        self.mask_img_list = {}
        for label in self.config_data['mask_label_list']:
            self.mask_img_list[label] = np.zeros((2048, 2448), dtype=np.uint8)

        self.mask_extend_size_label = self.config_data['mask_label_extend_size']

        self.edge_filter_obj = TileEdgeFilter(config_data['calibrate'])

    def filter_init_test(self, config_data):
        self.config_data = config_data['filter']

        self.template_repository = TemplateRepository(self.model_dir, self.config_data)

        self.mask_img_list = {}
        for label in self.config_data['mask_label_list']:
            self.mask_img_list[label] = np.zeros((2048, 2448), dtype=np.uint8)

        self.mask_extend_size_label = self.config_data['mask_label_extend_size']

    def update_config_data(self, config_data_dict):
        self.config_data = config_data_dict['filter']
        self.mask_extend_size_label = self.config_data['mask_label_extend_size']

        for key in self.config_data['mask_label_list']:
            if key not in self.mask_img_list.keys():
                self.mask_img_list[key] = np.zeros((2048, 2448), dtype=np.uint8)
        keys = []
        for key in self.mask_img_list.keys():
            if key not in self.config_data['mask_label_list']:
                keys.append(key)

        for key in keys:
            self.mask_img_list.pop(key)

        self.detector_local_threshold_data_1 = config_data_dict['threshold']['threshold_list']
        self.detector_global_threshold_1 = float(config_data_dict['threshold']['global'])

        self.detector_global_threshold_2 = self.detector_global_threshold_1 / 4
        for key in self.detector_local_threshold_data_1.keys():
            self.detector_local_threshold_data_2[key] = float(self.detector_local_threshold_data_1[key]) / 4

        self.template_repository.update_config_data(self.config_data)

        self.edge_filter_obj.update_edge_config(config_data_dict['calibrate'])

    def draw_mask(self, label, points):

        if label in self.mask_img_list.keys():
            self.mask_img_list[label][points[1]:points[3], points[0]:points[2]] = 255

    def label_rect_compare_2(self, srcImg, targetImg, label):

        src_gray_img = cv2.cvtColor(srcImg, cv2.COLOR_RGB2GRAY)
        target_gray_img = cv2.cvtColor(targetImg, cv2.COLOR_RGB2GRAY)

        res = cv2.matchTemplate(src_gray_img, target_gray_img, cv2.TM_CCORR_NORMED)

        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        h, w, c = targetImg.shape
        loc = max_loc

        if label in ['LZ','DM']:
            src_roi_rgb = srcImg[loc[1]:loc[1] + h, loc[0]:loc[0] + w, :]
            det_hsv = cv2.cvtColor(targetImg, cv2.COLOR_RGB2HSV)
            tmp_hsv = cv2.cvtColor(src_roi_rgb, cv2.COLOR_RGB2HSV)
            det_hist = cv2.calcHist([det_hsv], [2], None, [255], [0, 256])
            tmp_hist = cv2.calcHist([tmp_hsv], [2], None, [255], [0, 256])
            th = np.argmax(tmp_hist) / 2
            det_hist = det_hist.flatten()
            tmp_hist = tmp_hist.flatten()
            # det_hist[:int(th)]
            n1 = np.sum(det_hist[:int(th)] > 0)
            n2 = np.sum(tmp_hist[:int(th)] > 0)

            if n1 >= n2 * 1.5:
                return True
            else:
                return False
        elif label in ['LX', 'HH']:
            det_roi = target_gray_img
            tmp_roi = src_gray_img[loc[1]:loc[1] + h, loc[0]:loc[0] + w]
            # err = cv2.absdiff(det_roi, tmp_roi)
            h, w = det_roi.shape
            if h / w <= 1:
                sobelx1 = cv2.Sobel(det_roi, cv2.CV_64F, 0, 1, ksize=3)
                sobelx2 = cv2.Sobel(tmp_roi, cv2.CV_64F, 0, 1, ksize=3)
                # sobeldiff = cv2.Sobel(err, cv2.CV_64F, 0, 1, ksize=3)
            else:
                sobelx1 = cv2.Sobel(det_roi, cv2.CV_64F, 1, 0, ksize=3)
                sobelx2 = cv2.Sobel(tmp_roi, cv2.CV_64F, 1, 0, ksize=3)
                # sobeldiff = cv2.Sobel(err, cv2.CV_64F, 1, 0, ksize=3)

            sobelx1 = cv2.convertScaleAbs(sobelx1)
            sobelx2 = cv2.convertScaleAbs(sobelx2)
            # sobeldiff = cv2.convertScaleAbs(sobeldiff)
            th1, thresh1 = cv2.threshold(sobelx1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            th2, thresh2 = cv2.threshold(sobelx2, th1, 255, cv2.THRESH_BINARY)
            # th3, thresh3 = cv2.threshold(sobeldiff, th1, 255, cv2.THRESH_BINARY)
            sum1 = np.sum(thresh1.flatten() > 0)
            sum2 = np.sum(thresh2.flatten() > 0)
            # sum3 = np.sum(thresh3.flatten() > 0)
            score = abs(sum2 / (h * w) - sum1 / (h * w)) / (sum1 / (h * w))
            # print('name: {},score: {:.4f}'.format(img_name, score * 100))
            if score > 0.05:
                return True
            else:
                return False
        elif label in ['QJ', 'QB', 'BM']:
            if max_val < 0.98:
                return True
            else:
                return False
        else:
            src_roi_rgb = srcImg[loc[1]:loc[1] + h, loc[0]:loc[0] + w, :]
            det_hsv = cv2.cvtColor(targetImg, cv2.COLOR_RGB2HSV)
            tmp_hsv = cv2.cvtColor(src_roi_rgb, cv2.COLOR_RGB2HSV)
            det_hist = cv2.calcHist([det_hsv], [2], None, [255], [0, 256])
            tmp_hist = cv2.calcHist([tmp_hsv], [2], None, [255], [0, 256])
            tIdxes = np.argwhere(tmp_hist.flatten() > 0)
            threshold1 = np.max(tIdxes) - np.min(tIdxes)
            tIdxes = np.argwhere(det_hist.flatten() > 0)
            threshold2 = np.max(tIdxes) - np.min(tIdxes)

            if threshold1 * 1.8 < threshold2:
                return True

            else:
                return False

    def clear_template_repository(self):
        self.template_repository.clear_repository()
        for key in self.mask_img_list.keys():
            self.mask_img_list[key][:, :] = 0

    def extend_points(self, points, label):
        xmin, ymin, xmax, ymax = points[:]
        d_y = (ymax - ymin) / 2
        d_x = (xmax - xmin) / 2
        if label in ['LX', 'HH']:  
            w = xmax - xmin
            h = ymax - ymin
            if w > h:
                d_x = 20
            else:
                d_y = 20

        if xmin - d_x < 0:
            xmin = 0
        else:
            xmin -= d_x
        if xmax + d_x > 2448:
            xmax = 2448
        else:
            xmax += d_x
        if ymin - d_y < 0:
            ymin = 0
        else:
            ymin -= d_y
        if ymax + d_y > 2048:
            ymax = 2048
        else:
            ymax += d_y

        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def cal_points(self, points, loc_id, isExtend):
        xmin, ymin, xmax, ymax = points
        if loc_id == 11:
            pass
        elif loc_id == 12:
            ymin += 1024
            ymax += 1024
        elif loc_id == 21:
            xmin += 1224
            xmax += 1224
        else:
            xmin += 1224
            xmax += 1224
            ymin += 1024
            ymax += 1024

        if isExtend:
            d_y = (ymax - ymin) / 2
            d_x = (xmax - xmin) / 2
            if xmin - d_x < 0:
                xmin = 0
            else:
                xmin -= d_x
            if xmax + d_x > 2448:
                xmax = 2448
            else:
                xmax += d_x
            if ymin - d_y < 0:
                ymin = 0
            else:
                ymin -= d_y
            if ymax + d_y > 2048:
                ymax = 2048
            else:
                ymax += d_y

        return [int(xmin), int(ymin), int(xmax), int(ymax)]

    def mask_filter(self, np_boxes, loc_id):
        # TODO
        # return np_boxes
        expect_boxes = []
        if len(np_boxes) > 0:
            for dt in np_boxes:
                clsid = int(dt[0])
                label = self.labels_id[clsid]
                xmin, ymin, xmax, ymax = self.cal_points(dt[2:].astype(np.int), loc_id, False)

                if label in self.mask_img_list.keys():
                    masker = self.mask_img_list[label]

                    if np.count_nonzero(masker[ymin:ymax, xmin:xmax]) > 0:
                        expect_boxes.append(False)
                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                    else:
                        expect_boxes.append(True)
                else:
                    expect_boxes.append(True)

            return np_boxes[expect_boxes, :]
        else:
            return np_boxes

    def edge_filter(self, np_boxes, loc_id):
        return self.edge_filter_obj.filter(np_boxes, loc_id)

    def detector_cascade_detect_test_2(self, img, loc_id):
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)
        det_results1 = self.detector.predict(img)

        temp_img = self.template_repository.find_template(img, loc_id)
        if len(det_results1['boxes']) > 0:

            np_boxes = self.mask_filter(det_results1['boxes'], loc_id)
            det_results1['boxes'] = np_boxes
            if len(np_boxes) > 0:
                if temp_img is not None:

                    self.detector.threshold_adjust(self.detector_global_threshold_2,
                                                   self.detector_local_threshold_data_2)
                    det_results2 = self.detector.predict(temp_img)
                    if len(det_results2['boxes']) > 0:

                        np_boxes2 = det_results2['boxes']
                        expect_boxes = []
                        for dt in np_boxes:
                            clsid = int(dt[0])
                            label = self.labels_id[clsid]
                            xmin1, ymin1, xmax1, ymax1 = dt[2:]


                            if clsid in np_boxes2[:, 0].astype(np.int):
                         
                                tt_box = np_boxes2[np.where(np_boxes2[:, 0].astype(np.int) == clsid)]
                                flag_list = []
                                for dt2 in tt_box:
                                    xmin2, ymin2, xmax2, ymax2 = dt2[2:]
                            
                                    radius_pixel = int(self.config_data['u_radius_pixel'])
                                    size_rate = float(self.config_data['f_size_rate'])
                                    x1, y1 = (xmax1 - xmin1) / 2 + xmin1, (ymax1 - ymin1) / 2 + ymin1
                                    x2, y2 = (xmax2 - xmin2) / 2 + xmin2, (ymax2 - ymin2) / 2 + ymin2
                                    dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                                    rate = min(area1, area2) / max(area1, area2)
                                    print(
                                        'label: {}, dis: {:.4f},rate: {:.4f}'.format(self.labels_id[clsid], dis, rate))
                                    if label in self.config_data['only_size_labels']:  
                                        if rate >= size_rate:
                                            flag_list.append(1)
                                        else:
                                            flag_list.append(0)
                                    else:
                                 
                                        if dis <= radius_pixel and rate >= size_rate:
                                         
                                            if label in self.mask_extend_size_label:
                                                xmin, ymin, xmax, ymax = self.cal_points(dt2[2:], loc_id, True)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                xmin, ymin, xmax, ymax = self.cal_points(dt[2:], loc_id, True)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                            else:
                                                xmin, ymin, xmax, ymax = self.cal_points(dt2[2:], loc_id, False)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                xmin, ymin, xmax, ymax = self.cal_points(dt[2:], loc_id, False)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                            flag_list.append(1)
                                        else:
                                           
                                            d_y = (ymax1 - ymin1) / 2
                                            d_x = (xmax1 - xmin1) / 2
                                            n_xmin1 = 0
                                            n_xmax1 = 0
                                            n_ymin1 = 0
                                            n_ymax1 = 0
                                            if xmin1 - d_x < 0:
                                                n_xmin1 = 0
                                            else:
                                                n_xmin1 = xmin1 - d_x
                                            if xmax1 + d_x > 1224:
                                                n_xmax1 = 1224
                                            else:
                                                n_xmax1 = d_x + xmax1
                                            if ymin1 - d_y < 0:
                                                n_ymin1 = 0
                                            else:
                                                n_ymin1 = ymin1 - d_y
                                            if ymax1 + d_y > 1024:
                                                n_ymax1 = 1024
                                            else:
                                                n_ymax1 = d_y + ymax1

                                            
                                            if self.label_rect_compare_2(
                                                    temp_img[int(n_ymin1):int(n_ymax1), int(n_xmin1):int(n_xmax1), :],
                                                    img[int(ymin1):int(ymax1), int(xmin1):int(xmax1), :],
                                                    label):
                                             
                                                flag_list.append(0)
                                            else:
                                                
                                                if label in self.mask_extend_size_label:
                                                    xmin1, ymin1, xmax1, ymax1 = self.cal_points(
                                                        [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                                self.draw_mask(label, [int(xmin1), int(ymin1), int(xmax1), int(ymax1)])
                                                flag_list.append(1)

                                if sum(flag_list) > 0:
                                    expect_boxes.append(False)
                                    print('模板过滤: [{}] 已过滤'.format(label))
                                else:
                                    expect_boxes.append(True)
                                    print('模板过滤: [{}] 未过滤'.format(label))
                            else:
                                expect_boxes.append(True)

                        det_results1['boxes'] = np_boxes[expect_boxes, :]
                    else:
                        print('模板已找到,检测图: 有框，模板图: 无框.')
                        print('直接进行模板搜索的方式比较标签.')
                        expect_boxes = []
                        for dt in np_boxes:
                            clsid = int(dt[0])
                            label = self.labels_id[clsid]
                            
                            xmin, ymin, xmax, ymax = dt[2:]

                           
                            d_y = (ymax - ymin) / 2
                            d_x = (xmax - xmin) / 2
                            n_xmin = 0
                            n_xmax = 0
                            n_ymin = 0
                            n_ymax = 0
                            if xmin - d_x < 0:
                                n_xmin = 0
                            else:
                                n_xmin = xmin - d_x
                            if xmax + d_x > 1224:
                                n_xmax = 1224
                            else:
                                n_xmax = d_x + xmax
                            if ymin - d_y < 0:
                                n_ymin = 0
                            else:
                                n_ymin = ymin - d_y
                            if ymax + d_y > 1024:
                                n_ymax = 1024
                            else:
                                n_ymax = d_y + ymax

                            if self.label_rect_compare_2(temp_img[int(n_ymin):int(n_ymax), int(n_xmin):int(n_xmax), :],
                                                         img[int(ymin):int(ymax), int(xmin):int(xmax), :], label):
                            
                                expect_boxes.append(True)
                                print('模板搜索: [{}] 未过滤.'.format(label))
                            else:

                                if label in self.mask_extend_size_label:
                                    xmin, ymin, xmax, ymax = self.cal_points([xmin, ymin, xmax, ymax], loc_id, True)
                                self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                expect_boxes.append(False)
                                print('模板搜索: [{}] 已过滤.'.format(label))

                        det_results1['boxes'] = np_boxes[expect_boxes, :]
                    
                else:
                    print('模板未找到,检测图: 有框', np_boxes)
            else:
                print('mask 已全部过滤掉.')
        else:
            print('检测图: 无框.')
        return det_results1, temp_img

    def detector_cascade_detect_test_3(self, img, loc_id):
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)

        det_results1 = self.detector.predict(img)

        temp_img = self.template_repository.find_template(img, loc_id)

        if len(det_results1['boxes']) > 0:
     
            np_boxes = self.mask_filter(det_results1['boxes'], loc_id)
            det_results1['boxes'] = np_boxes
            if len(np_boxes) > 0:
                if temp_img is not None:
             
                    self.detector.threshold_adjust(self.detector_global_threshold_2,
                                                   self.detector_local_threshold_data_2)
                    det_results2 = self.detector.predict(temp_img)
                    if len(det_results2['boxes']) > 0:

                        np_boxes2 = det_results2['boxes']
                        expect_boxes = []
                        for dt in np_boxes:
                            clsid = int(dt[0])
                            label = self.labels_id[clsid]
                            xmin1, ymin1, xmax1, ymax1 = dt[2:]

                        
                            if clsid in np_boxes2[:, 0].astype(np.int):
                         
                                tt_box = np_boxes2[np.where(np_boxes2[:, 0].astype(np.int) == clsid)]
                                flag_list = []
                                for dt2 in tt_box:
                                    xmin2, ymin2, xmax2, ymax2 = dt2[2:]
                              
                                    radius_pixel = int(self.config_data['u_radius_pixel'])
                                    size_rate = float(self.config_data['f_size_rate'])
                                    x1, y1 = (xmax1 - xmin1) / 2 + xmin1, (ymax1 - ymin1) / 2 + ymin1
                                    x2, y2 = (xmax2 - xmin2) / 2 + xmin2, (ymax2 - ymin2) / 2 + ymin2
                                    dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                                    rate = min(area1, area2) / max(area1, area2)
                                    # print('label: {}, dis: {:.4f},rate: {:.4f}'.format(self.labels_id[clsid], dis, rate))
                                    if label in self.config_data['only_size_labels']:  
                                        if rate >= size_rate:
                                            flag_list.append(1)
                                        else:
                                            flag_list.append(0)
                                    else:
                                  
                                        if dis <= radius_pixel and rate >= size_rate:
                                        
                                            if label in self.mask_extend_size_label:
                                                xmin, ymin, xmax, ymax = self.cal_points(dt2[2:], loc_id, True)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                xmin, ymin, xmax, ymax = self.cal_points(dt[2:], loc_id, True)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                            else:
                                                xmin, ymin, xmax, ymax = self.cal_points(dt2[2:], loc_id, False)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                xmin, ymin, xmax, ymax = self.cal_points(dt[2:], loc_id, False)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                            flag_list.append(1)
                                        else:
                                            flag_list.append(0)
                                if sum(flag_list) > 0:
                                    expect_boxes.append(False)
 
                                else:
                                    expect_boxes.append(True)
       
                            else:
 
                                d_y = (ymax1 - ymin1) / 2
                                d_x = (xmax1 - xmin1) / 2
                                n_xmin1 = 0
                                n_xmax1 = 0
                                n_ymin1 = 0
                                n_ymax1 = 0
                                if xmin1 - d_x < 0:
                                    n_xmin1 = 0
                                else:
                                    n_xmin1 = xmin1 - d_x
                                if xmax1 + d_x > 1224:
                                    n_xmax1 = 1224
                                else:
                                    n_xmax1 = d_x + xmax1
                                if ymin1 - d_y < 0:
                                    n_ymin1 = 0
                                else:
                                    n_ymin1 = ymin1 - d_y
                                if ymax1 + d_y > 1024:
                                    n_ymax1 = 1024
                                else:
                                    n_ymax1 = d_y + ymax1

                                # xmin, ymin, xmax, ymax = self.cal_points(dt[2:], loc_id, True)
                                if self.label_rect_compare_2(temp_img[int(n_ymin1):int(n_ymax1),
                                                             int(n_xmin1):int(n_xmax1), :],
                                                             img[int(ymin1):int(ymax1),
                                                             int(xmin1):int(xmax1), :], label):
                             
                                    expect_boxes.append(True)
                                else:
                                  
                                    if label in self.mask_extend_size_label:
                                        xmin1, ymin1, xmax1, ymax1 = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                    else:
                                        xmin1, ymin1, xmax1, ymax1 = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                    self.draw_mask(label, [int(xmin1), int(ymin1), int(xmax1), int(ymax1)])
                                    expect_boxes.append(False)

                        det_results1['boxes'] = np_boxes[expect_boxes, :]
                    else:
                        print('模板已找到,检测图: 有框，模板图: 无框.')
                        print('直接进行模板搜索的方式比较标签.')
                        expect_boxes = []
                        for dt in np_boxes:
                            clsid = int(dt[0])
                            label = self.labels_id[clsid]
                            # if label in self.mask_extend_size_label:
                            xmin, ymin, xmax, ymax = dt[2:]
                            d_y = (ymax - ymin) / 2
                            d_x = (xmax - xmin) / 2
                            n_xmin = 0
                            n_xmax = 0
                            n_ymin = 0
                            n_ymax = 0
                            if xmin - d_x < 0:
                                n_xmin = 0
                            else:
                                n_xmin = xmin - d_x
                            if xmax + d_x > 1224:
                                n_xmax = 1224
                            else:
                                n_xmax = d_x + xmax
                            if ymin - d_y < 0:
                                n_ymin = 0
                            else:
                                n_ymin = ymin - d_y
                            if ymax + d_y > 1024:
                                n_ymax = 1024
                            else:
                                n_ymax = d_y + ymax

                            if self.label_rect_compare_2(
                                    temp_img[int(n_ymin):int(n_ymax), int(n_xmin):int(n_xmax), :],
                                    img[int(ymin):int(ymax), int(xmin):int(xmax), :], label):
                        
                                expect_boxes.append(True)
                         
                            else:
                           
                                
                                if label in self.mask_extend_size_label:
                                    xmin, ymin, xmax, ymax = self.cal_points([xmin, ymin, xmax, ymax], loc_id, True)
                                else:
                                    xmin, ymin, xmax, ymax = self.cal_points([xmin, ymin, xmax, ymax], loc_id, False)
                                self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                expect_boxes.append(False)

                            det_results1['boxes'] = np_boxes[expect_boxes, :]

                       
                else:
                    det_results1['boxes'] = np_boxes
            else:
                print('mask 已全部过滤掉.')
        else:
            print('检测图: 无框.')
        return det_results1, temp_img

    def detector_cascade_detect_test_4(self, img, loc_id):
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)
        det_results1 = self.detector.predict(img)
        temp_img = self.template_repository.find_template(img, loc_id)

        if temp_img is not None:
            if len(det_results1['boxes']) > 0:
           
                np_boxes = self.mask_filter(det_results1['boxes'], loc_id)
                det_results1['boxes'] = np_boxes
                radius_pixel = int(self.config_data['u_radius_pixel'])
                size_rate = float(self.config_data['f_size_rate'])
                if len(np_boxes) > 0:
                    print('模板图: 有框.')
               
                    self.detector.threshold_adjust(self.detector_global_threshold_2,
                                                   self.detector_local_threshold_data_2)
                    det_results2 = self.detector.predict(temp_img)
                    if len(det_results2['boxes']) > 0:
                        np_boxes2 = det_results2['boxes']
                        expect_boxes = []
                        for dt in np_boxes:
                            clsid = int(dt[0])
                            label = self.labels_id[clsid]
                            xmin1, ymin1, xmax1, ymax1 = dt[2:]
                            if clsid in np_boxes2[:, 0].astype(np.int):
                                print('\t模板图有相同标签.')
                           
                                tt_box = np_boxes2[np.where(np_boxes2[:, 0].astype(np.int) == clsid)]
                                flag_list = []
                                for dt2 in tt_box:
                                    xmin2, ymin2, xmax2, ymax2 = dt2[2:]
                           
                                    x1, y1 = (xmax1 - xmin1) / 2 + xmin1, (ymax1 - ymin1) / 2 + ymin1
                                    x2, y2 = (xmax2 - xmin2) / 2 + xmin2, (ymax2 - ymin2) / 2 + ymin2
                                    dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                                    rate = min(area1, area2) / max(area1, area2)
                                    print('\t\tlabel: {}, dis: {:.4f},rate: {:.4f}'.format(self.labels_id[clsid], dis,
                                                                                           rate))
                                    if label in self.config_data['only_size_labels']:  
                                        if rate >= size_rate:
                                            flag_list.append(1)
                                        else:
                                            flag_list.append(0)
                                    else:
                                    
                                        if dis <= radius_pixel and rate >= size_rate:
                                     
                                            if label in self.mask_extend_size_label:
                                                xmin, ymin, xmax, ymax = self.cal_points([xmin2, ymin2, xmax2, ymax2],
                                                                                         loc_id, True)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                xmin, ymin, xmax, ymax = self.cal_points([xmin1, ymin1, xmax1, ymax1],
                                                                                         loc_id, True)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                            else:
                                                xmin, ymin, xmax, ymax = self.cal_points([xmin2, ymin2, xmax2, ymax2],
                                                                                         loc_id, False)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                xmin, ymin, xmax, ymax = self.cal_points([xmin1, ymin1, xmax1, ymax1],
                                                                                         loc_id, False)
                                                self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                            flag_list.append(1)
                                        else:
                                            print('\t\t位置尺度不满足条件,进行模板搜索标签过滤.')
                                            new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1])
                                            if self.label_rect_compare_2(
                                                    temp_img[new_pts[1]:new_pts[3], new_pts[0]:new_pts[2], :],
                                                    img[int(ymin1):int(ymax1), int(xmin1):int(xmax1), :], label):
                                            
                                                flag_list.append(0)
                                                print('\t\t标签比较: [{}] 真标签'.format(label))
                                            else:
                                                print('\t\t标签比较: [{}] 假标签'.format(label))
                                                
                                                if label in self.mask_extend_size_label:
                                                    xmin, ymin, xmax, ymax = self.cal_points(
                                                        [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                                else:
                                                    xmin, ymin, xmax, ymax = self.cal_points(
                                                        [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                                self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                                flag_list.append(1)

                                if sum(flag_list) > 0:
                                    expect_boxes.append(False)
                                    print('\t模板过滤: [{}] 已过滤'.format(label))
                                else:
                                    expect_boxes.append(True)
                                    print('\t模板过滤: [{}] 未过滤'.format(label))
                            else:
                                print('\t模板图没有相同标签.')
                                print('\t进行模板搜索标签过滤.')
                                new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1])
                                if self.label_rect_compare_2(temp_img[new_pts[1]:new_pts[3],
                                                             new_pts[0]:new_pts[2], :],
                                                             img[int(ymin1):int(ymax1),
                                                             int(xmin1):int(xmax1), :], label):
                         
                                    expect_boxes.append(True)
                                    print('\t\t标签比较: [{}] 真标签'.format(label))
                                else:
                                    print('\t\t标签比较: [{}] 假标签'.format(label))
                                   
                                    if label in self.mask_extend_size_label:
                                        xmin, ymin, xmax, ymax = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                    else:
                                        xmin, ymin, xmax, ymax = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                    self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                    expect_boxes.append(False)
                        det_results1['boxes'] = np_boxes[expect_boxes, :]
                    else:
                        print('模板图: 无框.')
                        print('\t进行模板搜索标签过滤.')
                        expect_boxes = []
                        for dt in np_boxes:
                            xmin1, ymin1, xmax1, ymax1 = dt[2:]
                            clsid = int(dt[0])
                            label = self.labels_id[clsid]
                            new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1])
                            if self.label_rect_compare_2(temp_img[new_pts[1]:new_pts[3],
                                                         new_pts[0]:new_pts[2], :],
                                                         img[int(ymin1):int(ymax1),
                                                         int(xmin1):int(xmax1), :], label):
                               
                                expect_boxes.append(True)
                                print('\t\t标签比较: [{}] 真标签'.format(label))
                            else:
                                print('\t\t标签比较: [{}] 假标签'.format(label))
                                
                                if label in self.mask_extend_size_label:
                                    xmin, ymin, xmax, ymax = self.cal_points(
                                        [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                else:
                                    xmin, ymin, xmax, ymax = self.cal_points(
                                        [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                expect_boxes.append(False)

                        det_results1['boxes'] = np_boxes[expect_boxes, :]
                else:
                    print('mask过滤: 无标签.')
            else:
                print('检测图无框.')
        else:
            print('模板未找到.')

        return det_results1, temp_img

    def detector_start_detect(self):
        self.detect_stop = False

    def detector_stop_detect(self):
        self.detect_stop = True


    def rect_boxes_compare(self, np_boxes1, np_boxes2):
        error_boxes = [] 
        if len(np_boxes2) == 0:
            return np_boxes1, np.array([])


        masker_img2 = np.zeros((1024, 1224), dtype=np.int8)
        for dt in np_boxes2:
            pts = dt[2:].astype(int)
            masker_img2[pts[1]:pts[3], pts[0]:pts[2]] = 255

        expect_boxes = []
        for dt1 in np_boxes1:
            pts = dt1[2:].astype(int)
     
            masker_img1 = np.zeros((1024, 1224), dtype=np.int8)
            masker_img1[pts[1]:pts[3], pts[0]:pts[2]] = 255  # 画框
      
            masker_dst = cv2.bitwise_and(masker_img1, masker_img2)
            area = np.count_nonzero(masker_dst)
            xmin1, ymin1, xmax1, ymax1 = dt1[2:]
            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
            if (area / area1) >= 0.2:
            
                expect_boxes.append(False)
                error_boxes.append(True)
            
                clsid1 = int(dt1[0])
                label1 = self.labels_id[clsid1]
                self.draw_mask(label1, pts)
            else:
            
                expect_boxes.append(True)
                error_boxes.append(False)

        return np_boxes1[expect_boxes, :], np_boxes1[error_boxes, :]


    def rect_box_vector_comp(self, img1, img2):

        vector1 = self.rect_box_detector.predict(img1)
        vector2 = self.rect_box_detector.predict(img2)
        op2 = cosine(vector1, vector2)
        return abs(op2)


    def boxes_cnn_compare(self, np_boxes, srcImg, tempImg, loc_id):


        if len(np_boxes) == 0:
            return np_boxes

        
        expect_boxes = []
        ret_boxes = np_boxes.copy()
        for dt in np_boxes:
            clsid = int(dt[0])
            label = self.labels_id[clsid]
            pts = dt[2:].astype(int)


            xmin1, xmax1 = pts[0] - (pts[2] - pts[0])*2, (pts[2] - pts[0])*2 + pts[2]
            ymin1, ymax1 = pts[1] - (pts[3] - pts[1])*2, (pts[3] - pts[1])*2 + pts[3]
            if xmin1 < 0:
                xmin1 = 0
            if ymin1 < 0:
                ymin1 = 0
            if xmax1 > 1224:
                xmax1 = 1224
            if ymax1 > 1024:
                ymax1 = 1024
           
            src_rect = srcImg[pts[1]:pts[3], pts[0]:pts[2], :]
            temp_rect = tempImg[ymin1:ymax1, xmin1:xmax1, :]
           
            im_encode = cv2.imencode('.jpg', src_rect, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]
            src_rect = cv2.imdecode(im_encode, 1)



            src_gray_img = cv2.cvtColor(src_rect, cv2.COLOR_RGB2GRAY)
            temp_gray_img = cv2.cvtColor(temp_rect, cv2.COLOR_RGB2GRAY)
            res = cv2.matchTemplate(temp_gray_img, src_gray_img, cv2.TM_CCORR_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            h, w, c = src_rect.shape
            loc = max_loc
            temp_rect = temp_rect[loc[1]:loc[1] + h, loc[0]:loc[0] + w, :]

           
            score = self.rect_box_vector_comp(src_rect, temp_rect)

            if label in ['QJ','QB','BM']:
                if max_val < 0.98:
                    expect_boxes.append(True)
                else:
            
                    new_pts = self.cal_points(pts, loc_id, False)
                    self.draw_mask(label, new_pts)
                    expect_boxes.append(False)
            else:
                if score >= 0.5:
                    expect_boxes.append(True)
                else:
       
                    if label in ['BB']:
                        if h >= 500 or w >= 500:
                            pass
                        else:
                            new_pts = self.cal_points(pts, loc_id, False)
                            self.draw_mask(label, new_pts)
                    expect_boxes.append(False)

           

        return ret_boxes[expect_boxes, :]

    def detector_detect(self, srcImg, loc_id):

        if self.detect_stop:
            return
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)
        # TODO
        im_encode = cv2.imencode('.jpg', srcImg, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]
        img = cv2.imdecode(im_encode, 1)

        det_results1 = self.detector.predict(img)

        temp_img = self.template_repository.find_template(img, loc_id)
       

        if temp_img is not None:
            if len(det_results1['boxes']) > 0:
              
                edge_np_boxes = self.edge_filter(det_results1['boxes'], loc_id)
                det_results1['boxes'] = edge_np_boxes
                if len(edge_np_boxes) > 0:
              
                    mask_np_boxes = self.mask_filter(edge_np_boxes, loc_id)

                    det_results1['boxes'] = mask_np_boxes
                    if len(mask_np_boxes) > 0:
                        det_results1['boxes'] = self.boxes_cnn_compare(mask_np_boxes, srcImg, temp_img, loc_id)

        self.signal_output_result.emit(srcImg, det_results1, loc_id, temp_img)

   
    def detector_detect_test(self, srcImg, loc_id):
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)
        img = srcImg

        det_results1 = self.detector.predict(img)
        temp_img = self.template_repository.find_template(img, loc_id)

        if temp_img is not None:
            if len(det_results1['boxes']) > 0:
                edge_np_boxes = self.edge_filter(det_results1['boxes'], loc_id)
                det_results1['boxes'] = edge_np_boxes
                if len(edge_np_boxes) > 0:
                
                    np_boxes = self.mask_filter(edge_np_boxes, loc_id)
                    det_results1['boxes'] = np_boxes
                    radius_pixel = int(self.config_data['u_radius_pixel'])
                    size_rate = float(self.config_data['f_size_rate'])
                    if len(np_boxes) > 0:

                        self.detector.threshold_adjust(self.detector_global_threshold_2,
                                                       self.detector_local_threshold_data_2)
                        det_results2 = self.detector.predict(temp_img)
                        edge_np_boxes = self.edge_filter(det_results2['boxes'], loc_id)
                        if len(edge_np_boxes) > 0:
   
                            det_results2['boxes'] = self.mask_filter(edge_np_boxes, loc_id)

                            if len(det_results2['boxes']) > 0:
                                np_boxes2 = det_results2['boxes']
                                expect_boxes = []
                                for dt in np_boxes:
                                    clsid = int(dt[0])
                                    label = self.labels_id[clsid]
                                    xmin1, ymin1, xmax1, ymax1 = dt[2:]
                                    if clsid in np_boxes2[:, 0].astype(np.int):

                                        tt_box = np_boxes2[np.where(np_boxes2[:, 0].astype(np.int) == clsid)]
                                        flag_list = []
                                        for dt2 in tt_box:
                                            xmin2, ymin2, xmax2, ymax2 = dt2[2:]
                                     
                                            x1, y1 = (xmax1 - xmin1) / 2 + xmin1, (ymax1 - ymin1) / 2 + ymin1
                                            x2, y2 = (xmax2 - xmin2) / 2 + xmin2, (ymax2 - ymin2) / 2 + ymin2
                                            dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                            area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                                            rate = min(area1, area2) / max(area1, area2)
                                            
                                            if label in self.config_data['only_size_labels']: 
                                                if rate >= size_rate:
                                                    flag_list.append(1)
                                                else:
                                                    flag_list.append(0)
                                            else:
                                             
                                                if dis <= radius_pixel and rate >= size_rate:
                                                    
                                                    if label in self.mask_extend_size_label:
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin2, ymin2, xmax2, ymax2], loc_id, True)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                    else:
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin2, ymin2, xmax2, ymax2], loc_id, False)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                    flag_list.append(1)
                                                else:
                                                    
                                                    new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1], label)
                                                    if self.label_rect_compare_2(
                                                            temp_img[new_pts[1]:new_pts[3], new_pts[0]:new_pts[2], :],
                                                            img[int(ymin1):int(ymax1), int(xmin1):int(xmax1), :],
                                                            label):
                                                      
                                                        flag_list.append(0)
                                                        
                                                    else:
                                                                                                                                        
                                                        if label in self.mask_extend_size_label:
                                                            xmin, ymin, xmax, ymax = self.cal_points(
                                                                [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                                        else:
                                                            xmin, ymin, xmax, ymax = self.cal_points(
                                                                [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                                        self.draw_mask(label,
                                                                       [int(xmin), int(ymin), int(xmax), int(ymax)])
                                                        flag_list.append(1)

                                        if sum(flag_list) > 0:
                                            expect_boxes.append(False)
                                            
                                        else:
                                            expect_boxes.append(True)
                                            
                                    else:
                                        
                                        new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1], label)
                                        if self.label_rect_compare_2(temp_img[new_pts[1]:new_pts[3],
                                                                     new_pts[0]:new_pts[2], :],
                                                                     img[int(ymin1):int(ymax1),
                                                                     int(xmin1):int(xmax1), :], label):
                                          
                                            expect_boxes.append(True)
                                          
                                        else:
             
                                            if label in self.mask_extend_size_label:
                                                xmin, ymin, xmax, ymax = self.cal_points(
                                                    [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                            else:
                                                xmin, ymin, xmax, ymax = self.cal_points(
                                                    [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                            self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                            expect_boxes.append(False)
                                det_results1['boxes'] = np_boxes[expect_boxes, :]
                        else:
             
                            expect_boxes = []
                            for dt in np_boxes:
                                xmin1, ymin1, xmax1, ymax1 = dt[2:]
                                clsid = int(dt[0])
                                label = self.labels_id[clsid]
                                new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1], label)
                                if self.label_rect_compare_2(temp_img[new_pts[1]:new_pts[3],
                                                             new_pts[0]:new_pts[2], :],
                                                             img[int(ymin1):int(ymax1),
                                                             int(xmin1):int(xmax1), :], label):
                             
                                    expect_boxes.append(True)
                      
                                else:
                
                                    if label in self.mask_extend_size_label:
                                        xmin, ymin, xmax, ymax = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                    else:
                                        xmin, ymin, xmax, ymax = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                    self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                    expect_boxes.append(False)

                            det_results1['boxes'] = np_boxes[expect_boxes, :]

        return det_results1, temp_img

    def detector_detect_test_2(self, srcImg, loc_id):
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)
        img = srcImg

        det_results1 = self.detector.predict(img)
        temp_img = self.template_repository.find_template(img, loc_id)

        t_imgs = []
        s_imgs = []
        labels = []

        if temp_img is not None:
            if len(det_results1['boxes']) > 0:
                np_boxes = self.boxes_cnn_compare(det_results1['boxes'], srcImg, temp_img)  # CNN比较
                
                det_results1['boxes'] = np_boxes
                for dt in np_boxes:
                    clsid = int(dt[0])
                    label = self.labels_id[clsid]
                    xmin1, ymin1, xmax1, ymax1 = dt[2:]

                    t_label_img = temp_img[int(ymin1):int(ymax1), int(xmin1):int(xmax1), :]
                    s_label_img = img[int(ymin1):int(ymax1), int(xmin1):int(xmax1), :]

                    t_imgs.append(t_label_img)
                    s_imgs.append(s_label_img)
                    labels.append(label)

        return det_results1, temp_img

    def detector_detect_test_3(self, srcImg, loc_id):
        self.detector.threshold_adjust(self.detector_global_threshold_1, self.detector_local_threshold_data_1)
        img = srcImg

        det_results1 = self.detector.predict(img)
        temp_img = self.template_repository.find_template(img, loc_id)

        if temp_img is not None:
            if len(det_results1['boxes']) > 0:
                edge_np_boxes = self.edge_filter(det_results1['boxes'], loc_id)
                det_results1['boxes'] = edge_np_boxes
                if len(edge_np_boxes) > 0:
                
                    np_boxes = self.mask_filter(edge_np_boxes, loc_id)
                    det_results1['boxes'] = np_boxes
                    radius_pixel = int(self.config_data['u_radius_pixel'])
                    size_rate = float(self.config_data['f_size_rate'])
                    if len(np_boxes) > 0:
           
                        self.detector.threshold_adjust(self.detector_global_threshold_2,
                                                       self.detector_local_threshold_data_2)
                        det_results2 = self.detector.predict(temp_img)
                        edge_np_boxes = self.edge_filter(det_results2['boxes'], loc_id)
                        if len(edge_np_boxes) > 0:
                        
                            det_results2['boxes'] = self.mask_filter(edge_np_boxes, loc_id)

                            if len(det_results2['boxes']) > 0:
                                np_boxes2 = det_results2['boxes']
                                expect_boxes = []
                                for dt in np_boxes:
                                    clsid = int(dt[0])
                                    label = self.labels_id[clsid]
                                    xmin1, ymin1, xmax1, ymax1 = dt[2:]
                                    if clsid in np_boxes2[:, 0].astype(np.int):
                                 
                                        tt_box = np_boxes2[np.where(np_boxes2[:, 0].astype(np.int) == clsid)]
                                        flag_list = []
                                        for dt2 in tt_box:
                                            xmin2, ymin2, xmax2, ymax2 = dt2[2:]
                                        
                                            x1, y1 = (xmax1 - xmin1) / 2 + xmin1, (ymax1 - ymin1) / 2 + ymin1
                                            x2, y2 = (xmax2 - xmin2) / 2 + xmin2, (ymax2 - ymin2) / 2 + ymin2
                                            dis = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                                            area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
                                            area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
                                            rate = min(area1, area2) / max(area1, area2)
                                            
                                            if label in self.config_data['only_size_labels']:  
                                                if rate >= size_rate:
                                                    flag_list.append(1)
                                                else:
                                                    flag_list.append(0)
                                            else:
                                              
                                                if dis <= radius_pixel and rate >= size_rate:
                                                
                                                    if label in self.mask_extend_size_label:
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin2, ymin2, xmax2, ymax2], loc_id, True)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                    else:
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin2, ymin2, xmax2, ymax2], loc_id, False)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                        xmin, ymin, xmax, ymax = self.cal_points(
                                                            [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                                        self.draw_mask(label, [xmin, ymin, xmax, ymax])
                                                    flag_list.append(1)
                                                else:
                                            
                                                    new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1], label)
                                                    if self.label_rect_compare_2(
                                                            temp_img[new_pts[1]:new_pts[3], new_pts[0]:new_pts[2], :],
                                                            img[int(ymin1):int(ymax1), int(xmin1):int(xmax1), :],
                                                            label):
                                                
                                                        flag_list.append(0)
                                                
                                                    else:
                                                        
                                                        if label in self.mask_extend_size_label:
                                                            xmin, ymin, xmax, ymax = self.cal_points(
                                                                [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                                        else:
                                                            xmin, ymin, xmax, ymax = self.cal_points(
                                                                [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                                        self.draw_mask(label,
                                                                       [int(xmin), int(ymin), int(xmax), int(ymax)])
                                                        flag_list.append(1)

                                        if sum(flag_list) > 0:
                                            expect_boxes.append(False)
                                     
                                        else:
                                            expect_boxes.append(True)
                                          
                                    else:
                             
                                        new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1], label)
                                        if self.label_rect_compare_2(temp_img[new_pts[1]:new_pts[3],
                                                                     new_pts[0]:new_pts[2], :],
                                                                     img[int(ymin1):int(ymax1),
                                                                     int(xmin1):int(xmax1), :], label):
                                          
                                            expect_boxes.append(True)
                                            
                                        else:
                                            
                                            if label in self.mask_extend_size_label:
                                                xmin, ymin, xmax, ymax = self.cal_points(
                                                    [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                            else:
                                                xmin, ymin, xmax, ymax = self.cal_points(
                                                    [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                            self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                            expect_boxes.append(False)
                                det_results1['boxes'] = np_boxes[expect_boxes, :]
                        else:
                            
                            expect_boxes = []
                            for dt in np_boxes:
                                xmin1, ymin1, xmax1, ymax1 = dt[2:]
                                clsid = int(dt[0])
                                label = self.labels_id[clsid]
                                new_pts = self.extend_points([xmin1, ymin1, xmax1, ymax1], label)
                                if self.label_rect_compare_2(temp_img[new_pts[1]:new_pts[3],
                                                             new_pts[0]:new_pts[2], :],
                                                             img[int(ymin1):int(ymax1),
                                                             int(xmin1):int(xmax1), :], label):
                                    
                                    expect_boxes.append(True)
                                   
                                else:
                                    
                                    if label in self.mask_extend_size_label:
                                        xmin, ymin, xmax, ymax = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, True)
                                    else:
                                        xmin, ymin, xmax, ymax = self.cal_points(
                                            [xmin1, ymin1, xmax1, ymax1], loc_id, False)
                                    self.draw_mask(label, [int(xmin), int(ymin), int(xmax), int(ymax)])
                                    expect_boxes.append(False)

                            det_results1['boxes'] = np_boxes[expect_boxes, :]

        return det_results1, temp_img
