import numpy as np
import cv2
from src.FeatureCNN import FeatureCnn

class TileTemplateFilter(object):
    def __init__(self, cfg_data_dict):

        # self.config_data = {'fFilterThresh':0.95,'uTempNumber':20}
        self.config_data = cfg_data_dict
        self.temp_img_list_21 = []
        self.temp_hash_list_21 = []
        self.temp_img_list_22 = []
        self.temp_hash_list_22 = []

        self.temp_img_list_11 = []
        self.temp_hash_list_11 = []
        self.temp_img_list_12 = []
        self.temp_hash_list_12 = []

        self.cnt = 1
        self.pass_num = 0


        self.model = FeatureCnn()

    def cal_hash(self, img, scale=2):
        img = cv2.resize(img, (img.shape[1] // scale, img.shape[0] // scale))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        average = np.mean(gray)
        np_hash = np.where(gray > average, 1, 0)
        return np_hash

    def compare_hash(self, hash1, hash2):
        t_hash = np.logical_xor(hash1, hash2)
        return np.sum(t_hash)

    def update_config_data(self, cfg_data_dict):
        nums = int(cfg_data_dict['u_temp_num'])

        if int(self.config_data['u_temp_num']) > nums:    
            deleta = int(self.config_data['u_temp_num']) - nums
            for i in range(deleta): 
                self.temp_img_list_11.pop(0)
                self.temp_hash_list_11.pop(0)
                self.temp_img_list_12.pop(0)
                self.temp_hash_list_12.pop(0)
                self.temp_img_list_21.pop(0)
                self.temp_hash_list_21.pop(0)
                self.temp_img_list_22.pop(0)
                self.temp_hash_list_22.pop(0)

        self.config_data = cfg_data_dict


    def compare_code_cnn(self,a,b):
        return self.model.feature_compare(a,b)

    def cal_code_cnn(self,img):
        return self.model.get_feature_code(img)

    def create_template_cnn(self,img,loc_id):
        if loc_id == 11:
            if len(self.temp_img_list_11) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_11.append(img)

                    hv = self.cal_code_cnn(img)
                    self.temp_hash_list_11.append(hv)


                    r_img = np.rot90(img, 2)
                    self.temp_img_list_22.append(r_img)
                    r_hv = self.cal_code_cnn(r_img)
                    self.temp_hash_list_22.append(r_hv)
                else:
                    self.pass_num += 1
        elif loc_id == 12:
            if len(self.temp_img_list_12) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_12.append(img)

                    hv = self.cal_code_cnn(img)
                    self.temp_hash_list_12.append(hv)


                    r_img = np.rot90(img, 2)
                    self.temp_img_list_21.append(r_img)
                    r_hv = self.cal_code_cnn(r_img)
                    self.temp_hash_list_21.append(r_hv)

                else:
                    self.pass_num += 1
        elif loc_id == 21:
            if len(self.temp_img_list_21) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_21.append(img)

                    hv = self.cal_code_cnn(img)
                    self.temp_hash_list_21.append(hv)


                    r_img = np.rot90(img, 2)
                    self.temp_img_list_12.append(r_img)
                    r_hv = self.cal_code_cnn(r_img)
                    self.temp_hash_list_12.append(r_hv)

                else:
                    self.pass_num += 1
        else:
            if len(self.temp_img_list_22) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_22.append(img)
                    hv = self.cal_code_cnn(img)
                    self.temp_hash_list_22.append(hv)

                    r_img = np.rot90(img, 2)
                    self.temp_img_list_11.append(r_img)
                    r_hv = self.cal_code_cnn(r_img)
                    self.temp_hash_list_11.append(r_hv)

                else:
                    self.pass_num += 1

        return True

    def create_template(self, img, loc_id):

        if loc_id == 11:
            if len(self.temp_img_list_11) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_11.append(img)
                    hv = self.cal_hash(img)
                    self.temp_hash_list_11.append(hv)

                    r_img = np.rot90(img,2)
                    self.temp_img_list_22.append(r_img)
                    r_hv = self.cal_hash(r_img)
                    self.temp_hash_list_22.append(r_hv)
                else:
                    self.pass_num += 1
        elif loc_id == 12:
            if len(self.temp_img_list_12) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_12.append(img)

                    hv = self.cal_hash(img)
                    self.temp_hash_list_12.append(hv)

                    r_img = np.rot90(img,2)
                    self.temp_img_list_21.append(r_img)
                    r_hv = self.cal_hash(r_img)
                    self.temp_hash_list_21.append(r_hv)

                else:
                    self.pass_num += 1
        elif loc_id == 21:
            if len(self.temp_img_list_21) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_21.append(img)

                    hv = self.cal_hash(img)
                    self.temp_hash_list_21.append(hv)


                    r_img = np.rot90(img,2)
                    self.temp_img_list_12.append(r_img)
                    r_hv = self.cal_hash(r_img)
                    self.temp_hash_list_12.append(r_hv)

                else:
                    self.pass_num += 1
        else:
            if len(self.temp_img_list_22) >= int(self.config_data['u_temp_num']) * 2:
                return False
            else:
                if self.pass_num >= int(self.config_data['u_pass_num']):
                    self.temp_img_list_22.append(img)

                    hv = self.cal_hash(img)
                    self.temp_hash_list_22.append(hv)


                    r_img = np.rot90(img,2)
                    self.temp_img_list_11.append(r_img)
                    r_hv = self.cal_hash(r_img)
                    self.temp_hash_list_11.append(r_hv)

                else:
                    self.pass_num += 1
        return True

    def get_template_cnn(self,img,loc_id):
        hash1 = self.cal_code_cnn(img)
        compare_result = []
        idx = 0
        if loc_id == 11:
            for hash2 in self.temp_hash_list_11:
                compare_result.append(self.compare_code_cnn(hash1, hash2))
        elif loc_id == 12:
            for hash2 in self.temp_hash_list_12:
                compare_result.append(self.compare_code_cnn(hash1, hash2))
        elif loc_id == 21:
            for hash2 in self.temp_hash_list_21:
                compare_result.append(self.compare_code_cnn(hash1, hash2))

        else:
            for hash2 in self.temp_hash_list_22:
                compare_result.append(self.compare_code_cnn(hash1, hash2))

        if max(compare_result) < 0.9:
            idx = -1

            self.update_template(img,loc_id)
        else:
            idx = compare_result.index(max(compare_result))

        return idx

    def get_template(self, img, loc_id):
        hash1 = self.cal_hash(img)
        compare_result = []
        idx = 0
        if loc_id == 11:
            for hash2 in self.temp_hash_list_11:
                compare_result.append(self.compare_hash(hash1, hash2))

            idx = compare_result.index(min(compare_result))
        elif loc_id == 12:
            for hash2 in self.temp_hash_list_12:
                compare_result.append(self.compare_hash(hash1, hash2))

            idx = compare_result.index(min(compare_result))
        elif loc_id == 21:
            for hash2 in self.temp_hash_list_21:
                compare_result.append(self.compare_hash(hash1, hash2))

            idx = compare_result.index(min(compare_result))
        else:
            for hash2 in self.temp_hash_list_22:
                compare_result.append(self.compare_hash(hash1, hash2))

            idx = compare_result.index(min(compare_result))

        return idx

    def update_template(self, img, loc_id):

        if loc_id == 11:
            self.temp_img_list_11.append(img)
            hv = self.cal_code_cnn(img)
            self.temp_hash_list_11.append(hv)
        elif loc_id == 12:

            self.temp_img_list_12.append(img)

            hv = self.cal_code_cnn(img)
            self.temp_hash_list_12.append(hv)
        elif loc_id == 21:
 
            self.temp_img_list_21.append(img)

            hv = self.cal_code_cnn(img)
            self.temp_hash_list_21.append(hv)
        else:

            self.temp_img_list_22.append(img)

            hv = self.cal_code_cnn(img)
            self.temp_hash_list_22.append(hv)

    def clear_template(self):
        self.temp_img_list_11.clear()
        self.temp_hash_list_11.clear()
        self.temp_img_list_12.clear()
        self.temp_hash_list_12.clear()
        self.temp_img_list_21.clear()
        self.temp_hash_list_21.clear()
        self.temp_img_list_22.clear()
        self.temp_hash_list_22.clear()


    def rotate_align_img(self, temp_img_gray, target_img_gray, loc_id):

        MAX_FEATURES = 500
        GOOD_MATCH_PERCENT = 0.5
        orb = cv2.ORB_create(MAX_FEATURES)

        keypoints1, descriptors1 = orb.detectAndCompute(temp_img_gray, None)
        keypoints2, descriptors2 = orb.detectAndCompute(target_img_gray, None)

        try:
            matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
            matches = matcher.match(descriptors1, descriptors2, None)

            # Sort matches by score
            if len(matches) < 4:
 
                return None
            matches.sort(key=lambda x: x.distance, reverse=False)

            # Remove not so good matches
            numGoodMatches = int(len(matches) * GOOD_MATCH_PERCENT)
            matches = matches[:numGoodMatches]
            if len(matches) < 4:
                # print('orb: len(matches) <= 0')
                return None
            # Draw top matches
            # imMatches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
            # cv2.imwrite("matches.jpg", imMatches)

            # Extract location of good matches
            points1 = np.zeros((len(matches), 2), dtype=np.float32)
            points2 = np.zeros((len(matches), 2), dtype=np.float32)

            for i, match in enumerate(matches):
                points1[i, :] = keypoints1[match.queryIdx].pt
                points2[i, :] = keypoints2[match.trainIdx].pt

            h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)

            # Use homography
            height, width = temp_img_gray.shape
            im1Reg = cv2.warpPerspective(temp_img_gray, h, (width, height))


            return im1Reg

        except Exception as e:
            # print('orb: ', str(e))
            pass

        return None


    def compare_label(self, img1, img2):
        
         md = cv2.TM_CCORR_NORMED  # cv2.TM_SQDIFF_NORMED, cv2.TM_CCORR_NORMED, cv2.TM_CCOEFF_NORMED
        res = cv2.matchTemplate(img1, img2, cv2.TM_CCORR_NORMED)
        loc = None
        if cv2.TM_SQDIFF_NORMED == md:
            t = 1 - 0.98
            loc = np.where(res < t)
        else:
            t = 0.98
            loc = np.where(res > t)

        pts = None
        th, tw = img2.shape[:2]
        for pt in zip(*loc[::-1]):
            # cv2.rectangle(img1, pt, (pt[0] + tw, pt[1] + th), (0, 0, 255))
            pts = [pt[0], pt[1], tw, th]
        if pts:
            return False
        else:
            return True

    def filter(self, template_id, img, np_boxes, loc_id):

        pass_boxes = np_boxes  
        unpass_boxes = np.array([])  
        template_img = None
        if template_id >= 0:
            if loc_id == 11:
                template_img = self.temp_img_list_11[template_id]
            elif loc_id == 12:
                template_img = self.temp_img_list_12[template_id]
            elif loc_id == 21:
                template_img = self.temp_img_list_21[template_id]
            else:
                template_img = self.temp_img_list_22[template_id]

        if template_img is not None:
  
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            template_img_gray = cv2.cvtColor(template_img, cv2.COLOR_RGB2GRAY)
    
            alignedImg_gray = self.rotate_align_img(template_img_gray, img_gray, loc_id)

            if alignedImg_gray is not None:  
                expect_boxes = []

                for box in np_boxes:
                    xmin, ymin, xmax, ymax = box[2:].astype(np.int)
          
                    ty = int((ymax - ymin))
                    tx = int((xmax - xmin))
                    nx1 = int(xmin) - tx
                    ny1 = int(ymin) - ty
                    nx2 = int(xmax) + tx
                    ny2 = int(ymax) + ty
                    if nx1 < 0:
                        nx1 = 0
                    if ny1 < 0:
                        ny1 = 0
                    if nx2 > 1224:
                        nx2 = 1224
                    if ny2 > 1024:
                        ny2 = 1024
                    img1 = alignedImg_gray[ny1:ny2, nx1:nx2]

                    img2 = img_gray[ymin:ymax, xmin:xmax]


                    if self.compare_label(img1, img2):
                        expect_boxes.append(True)
                    else:
                        expect_boxes.append(False)

                pass_boxes = np_boxes[expect_boxes, :]
                unpass_boxes = np_boxes[~np.array(expect_boxes), :]

        return [pass_boxes, unpass_boxes]


class TileMaskFilter(object):
    def __init__(self, cfg_data_dict, labels_list):
        self.config_data = cfg_data_dict
        self.labels = labels_list
        self.img_mask_list_11 = {}
        self.img_mask_list_12 = {}

        self.img_mask_list_21 = {}
        self.img_mask_list_22 = {}

        self.create_mask()

    def create_mask(self):

        for label in self.config_data['mask_label']:
            self.img_mask_list_11[label] = np.zeros((1024, 1224), np.uint8)
            self.img_mask_list_12[label] = np.zeros((1024, 1224), np.uint8)
            self.img_mask_list_21[label] = np.zeros((1024, 1224), np.uint8)
            self.img_mask_list_22[label] = np.zeros((1024, 1224), np.uint8)

    def update_config_data(self, cfg_data_dict):
        self.config_data = cfg_data_dict


        for key in self.img_mask_list_11.keys():
            if key not in cfg_data_dict['mask_label']:
                self.img_mask_list_11.pop(key)
                self.img_mask_list_12.pop(key)
                self.img_mask_list_21.pop(key)
                self.img_mask_list_22.pop(key)


        for label in self.config_data['mask_label']:
            if label not in self.img_mask_list_11:
                self.img_mask_list_11[label] = np.zeros((1024, 1224), np.uint8)
                self.img_mask_list_12[label] = np.zeros((1024, 1224), np.uint8)
                self.img_mask_list_21[label] = np.zeros((1024, 1224), np.uint8)
                self.img_mask_list_22[label] = np.zeros((1024, 1224), np.uint8)

    def draw_mask(self, np_boxes, loc_id):

        if len(np_boxes) > 0:
            for box in np_boxes:
                label = self.labels[int(box[0])]
                xmin, ymin, xmax, ymax = box[2:].astype(np.int)

                if loc_id == 11:
                    if label in self.img_mask_list_11:
                        # cv2.rectangle(self.img_mask_list_1[label], (xmin, ymin), (xmin + w, ymin + h), (255), -1)
                        self.img_mask_list_11[label][ymin:ymax, xmin:xmax] = 255
                elif loc_id == 12:
                    if label in self.img_mask_list_12:
                        # cv2.rectangle(self.img_mask_list_1[label], (xmin, ymin), (xmin + w, ymin + h), (255), -1)
                        self.img_mask_list_12[label][ymin:ymax, xmin:xmax] = 255
                elif loc_id == 21:
                    if label in self.img_mask_list_21:
                        # cv2.rectangle(self.img_mask_list_1[label], (xmin, ymin), (xmin + w, ymin + h), (255), -1)
                        self.img_mask_list_21[label][ymin:ymax, xmin:xmax] = 255
                else:
                    if label in self.img_mask_list_22:
                        # cv2.rectangle(self.img_mask_list_1[label], (xmin, ymin), (xmin + w, ymin + h), (255), -1)
                        self.img_mask_list_22[label][ymin:ymax, xmin:xmax] = 255

    def clear_mask(self):

        self.img_mask_list_11.clear()
        self.img_mask_list_12.clear()
        self.img_mask_list_21.clear()
        self.img_mask_list_22.clear()
        self.create_mask()


    def filter(self, np_boxes, loc_id):

        pass_boxes = np.array([])  
        unpass_boxes = np.array([]) 

        if len(np_boxes) > 0:
            expect_boxes = []
            for box in np_boxes:
                label = self.labels[int(box[0])]
                xmin, ymin, xmax, ymax = box[2:].astype(np.int)
                if loc_id == 11:
                    if label in self.img_mask_list_11:
                        if np.count_nonzero(self.img_mask_list_11[label][ymin:ymax, xmin:xmax]) > 0:  
                            expect_boxes.append(False)
                        else:
                            expect_boxes.append(True)
                    else:
                        expect_boxes.append(True)
                elif loc_id == 12:
                    if label in self.img_mask_list_12:
                        if np.count_nonzero(self.img_mask_list_12[label][ymin:ymax, xmin:xmax]) > 0:  
                            expect_boxes.append(False)
                        else:
                            expect_boxes.append(True)
                    else:
                        expect_boxes.append(True)
                elif loc_id == 21:
                    if label in self.img_mask_list_21:
                        if np.count_nonzero(self.img_mask_list_21[label][ymin:ymax, xmin:xmax]) > 0:  
                            expect_boxes.append(False)
                        else:
                            expect_boxes.append(True)
                    else:
                        expect_boxes.append(True)
                else:
                    if label in self.img_mask_list_22:
                        if np.count_nonzero(self.img_mask_list_22[label][ymin:ymax, xmin:xmax]) > 0:  
                            expect_boxes.append(False)
                        else:
                            expect_boxes.append(True)
                    else:
                        expect_boxes.append(True)

            pass_boxes = np_boxes[expect_boxes, :]

            expect_boxes = ~np.array(expect_boxes)
            unpass_boxes = np_boxes[~np.array(expect_boxes), :]

            if len(unpass_boxes) > 0:
                self.draw_mask(unpass_boxes,loc_id)

        return pass_boxes #[pass_boxes, unpass_boxes]


class TileEdgeFilter(object):
    def __init__(self, config_data_list=None):
        self.config_data = config_data_list

        self.mask_img = np.zeros((2048, 2448), dtype=np.uint8)

        line1_pos = int(config_data_list[0])
        line2_pos = int(config_data_list[1])
        line3_pos = int(config_data_list[2])
        line4_pos = int(config_data_list[3])
        line5_pos = int(config_data_list[4])
        line6_pos = int(config_data_list[5])
        line7_pos = int(config_data_list[6])
        line8_pos = int(config_data_list[7])

        self.mask_img[:, 0:line1_pos] = 255
        self.mask_img[:, line2_pos:line3_pos] = 255
        self.mask_img[:, line4_pos:] = 255
        self.mask_img[0:line5_pos, :] = 255
        self.mask_img[line6_pos:line7_pos, :] = 255
        self.mask_img[line8_pos:, :] = 255

    def update_edge_config(self,new_list):
        self.config_data = new_list
        self.update_edge_mask()

    def update_edge_mask(self):
        self.mask_img[:, :] = 0
        line1_pos = int(self.config_data[0])
        line2_pos = int(self.config_data[1])
        line3_pos = int(self.config_data[2])
        line4_pos = int(self.config_data[3])
        line5_pos = int(self.config_data[4])
        line6_pos = int(self.config_data[5])
        line7_pos = int(self.config_data[6])
        line8_pos = int(self.config_data[7])

        self.mask_img[:, 0:line1_pos] = 255
        self.mask_img[:, line2_pos:line3_pos] = 255
        self.mask_img[:, line4_pos:] = 255
        self.mask_img[0:line5_pos, :] = 255
        self.mask_img[line6_pos:line7_pos, :] = 255
        self.mask_img[line8_pos:, :] = 255

    def filter2(self,np_boxes,loc_id):
        if len(np_boxes) > 0:
            ret_boxes = np_boxes.copy()

            per_cnt = 0.7
            if loc_id == 21 or loc_id == 22:
                expect_boxes = ((ret_boxes[:, 5] - ret_boxes[:, 3]) / 2 + ret_boxes[:, 3]) >= (int(self.config_data[7]) - int(self.config_data[6]))

                # expect_boxes = pass_boxes[:,5] >= (int(self.config_data[7]) - int(self.config_data[6]))

                pass_boxes = ret_boxes[expect_boxes, :]

            if len(pass_boxes) > 0:
                if loc_id == 11:
                    pass
                elif loc_id == 12:
                    pass_boxes[:, 3] += 1024
                    pass_boxes[:, 5] += 1024
                elif loc_id == 21:
                    pass_boxes[:, 2] += 1224
                    pass_boxes[:, 4] += 1224
                else:
                    ret_boxes[:, 2] += 1224
                    ret_boxes[:, 3] += 1024
                    ret_boxes[:, 4] += 1224
                    ret_boxes[:, 5] += 1024

            if loc_id == 11:
                pass
            elif loc_id == 12:
                ret_boxes[:, 3] += 1024
                ret_boxes[:, 5] += 1024
            elif loc_id == 21:
                ret_boxes[:, 2] += 1224
                ret_boxes[:, 4] += 1224
            else:
                ret_boxes[:, 2] += 1224
                ret_boxes[:, 3] += 1024
                ret_boxes[:, 4] += 1224
                ret_boxes[:, 5] += 1024

            expect_boxes = []
            for box in ret_boxes:
                xmin, ymin, xmax, ymax = box[2:].astype(np.int)
                w = xmax - xmin
                h = ymax - ymin
                if np.count_nonzero(self.mask_img[ymin:ymax, xmin:xmax]) >= (w * h * per_cnt):
                    expect_boxes.append(False)
                else:
                    expect_boxes.append(True)

            pass_boxes = np_boxes[expect_boxes, :]
            if len(pass_boxes):
                if loc_id == 21 or loc_id == 22:
                    expect_boxes = ((pass_boxes[:,5] - pass_boxes[:,3]) / 2 + pass_boxes[:,3]) >= (int(self.config_data[7]) - int(self.config_data[6]))

                    # expect_boxes = pass_boxes[:,5] >= (int(self.config_data[7]) - int(self.config_data[6]))

                    pass_boxes = pass_boxes[expect_boxes, :]

            return pass_boxes
        else:
            return np_boxes

    def filter(self, np_boxes, loc_id):
        # TODO
        # return np_boxes
        if len(np_boxes) > 0:
            ret_boxes = np_boxes.copy()
            per_cnt = 0.7
            if loc_id == 11:
                pass
            elif loc_id == 12:
                ret_boxes[:, 3] += 1024
                ret_boxes[:, 5] += 1024
            elif loc_id == 21:
                ret_boxes[:, 2] += 1224
                ret_boxes[:, 4] += 1224
            else:
                ret_boxes[:, 2] += 1224
                ret_boxes[:, 3] += 1024
                ret_boxes[:, 4] += 1224
                ret_boxes[:, 5] += 1024

            expect_boxes = []
            for box in ret_boxes:
                xmin, ymin, xmax, ymax = box[2:].astype(np.int)
                w = xmax - xmin
                h = ymax - ymin

                center_x = w // 2 + xmin
                center_y = h // 2 + ymin

                if self.mask_img[center_y][center_x] > 0:
                    expect_boxes.append(False)
                else:
                    expect_boxes.append(True)

            pass_boxes = np_boxes[expect_boxes, :]
            return pass_boxes
        else:
            return np_boxes


class TileDefectsFilter(QObject):
    signal_output_filter_result = pyqtSignal(object,dict,int)  

    signal_clear_filter = pyqtSignal()  


    def __init__(self,temp_mask_filter_cfg_dict,edge_filter_cfg_list,label_list):
        super(TileDefectsFilter, self).__init__()

        self.temp_filter_obj = TileTemplateFilter(temp_mask_filter_cfg_dict)
        self.mask_filter_obj = TileMaskFilter(temp_mask_filter_cfg_dict,label_list)
        self.edge_filter_obj = TileEdgeFilter(edge_filter_cfg_list)

        self.logger = getSysLogger()

        self.clear_flag = False
        self.signal_clear_filter.connect(self.slot_clear_filter)

    def update_temp_config(self,new_dict):
        self.temp_filter_obj.update_config_data(new_dict)

    def update_mask_config(self,new_dict):
        self.mask_filter_obj.update_config_data(new_dict)

    def update_edge_config(self,new_list):
        self.edge_filter_obj.update_edge_config(new_list)

    def slot_clear_filter(self):
        self.clear_flag = True

    def clear_filter(self):
        self.temp_filter_obj.clear_template()
        self.mask_filter_obj.clear_mask()


    def slot_find_template_test(self,img,det_results,loc_id):
        tid = -1
        if self.temp_filter_obj.create_template_cnn(img, loc_id):

        else:
            tid = self.temp_filter_obj.get_template_cnn(img, loc_id)  

        temp_img = None
        if tid >= 0:
            if loc_id == 11:
                temp_img = self.temp_filter_obj.temp_img_list_11[tid]
            elif loc_id == 12:
                temp_img = self.temp_filter_obj.temp_img_list_12[tid]
            elif loc_id == 21:
                temp_img = self.temp_filter_obj.temp_img_list_21[tid]
            else:
                temp_img = self.temp_filter_obj.temp_img_list_22[tid]
        else:
            print('模板未找到.')


        return det_results, temp_img

    def slot_defects_filter_test(self,img,det_results,loc_id):
        tid = -1
        if self.temp_filter_obj.create_template_cnn(img, loc_id):

            print('小图模板正在建立.')
        else:

            np_boxes = det_results['boxes']
            if len(np_boxes):

                np_boxes = self.edge_filter_obj.filter(np_boxes,loc_id)
                mask_pass_boxes = self.mask_filter_obj.filter(np_boxes, loc_id)  
                if len(mask_pass_boxes):

                    tid = self.temp_filter_obj.get_template_cnn(img, loc_id)  
                  
                    temp_pass_boxes, temp_unpass_boxes = self.temp_filter_obj.filter(tid, img, mask_pass_boxes, loc_id)  

 
                    if len(temp_unpass_boxes):
                        expect_boxes = (temp_unpass_boxes[:, 1] < float(self.mask_filter_obj.config_data['f_filter_thresh']))  
                        t_np_boxes = temp_unpass_boxes[expect_boxes, :]
                        if len(t_np_boxes):
                            self.mask_filter_obj.draw_mask(t_np_boxes, loc_id)

                    if len(temp_pass_boxes):
                        det_results['boxes'] = temp_pass_boxes
                    else:
                        det_results['boxes'] = np.array([])

                else:
                    det_results['boxes'] = np.array([])

            else:
         
                pass

        temp_img = None
        if tid >= 0:
            if loc_id == 11:
                temp_img = self.temp_filter_obj.temp_img_list_11[tid]
            elif loc_id == 12:
                temp_img = self.temp_filter_obj.temp_img_list_12[tid]
            elif loc_id == 21:
                temp_img = self.temp_filter_obj.temp_img_list_21[tid]
            else:
                temp_img = self.temp_filter_obj.temp_img_list_22[tid]


        return det_results, temp_img

    def slot_defects_filter(self,img,det_results,loc_id):
        if self.temp_filter_obj.create_template_cnn(img, loc_id):

            pass
    
        else:

            np_boxes = det_results['boxes']
            if len(np_boxes):

                np_boxes = self.edge_filter_obj.filter(np_boxes,loc_id)
 
                mask_pass_boxes = self.mask_filter_obj.filter(np_boxes, loc_id)  
               
                if len(mask_pass_boxes):

                    tid = self.temp_filter_obj.get_template_cnn(img, loc_id)  
                    if tid >= 0:
                       
                        temp_pass_boxes, temp_unpass_boxes = self.temp_filter_obj.filter(tid, img, mask_pass_boxes, loc_id)  # 模板过滤,返回过滤的box[真Box,假box]


                        if len(temp_unpass_boxes):
                            expect_boxes = (temp_unpass_boxes[:, 1] < float(self.mask_filter_obj.config_data['f_filter_thresh']))  # 小于阈值的标签需要过滤
                            t_np_boxes = temp_unpass_boxes[expect_boxes, :]
                            if len(t_np_boxes):
                                self.mask_filter_obj.draw_mask(t_np_boxes, loc_id)

                        det_results['boxes'] = temp_pass_boxes
                    else:
                        det_results['boxes'] = mask_pass_boxes
                else:
                    det_results['boxes'] = mask_pass_boxes


    
        self.signal_output_filter_result.emit(img, det_results, loc_id)

        if self.clear_flag:
            self.clear_filter()
            self.clear_flag = False


