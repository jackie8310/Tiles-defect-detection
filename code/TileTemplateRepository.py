import copy

from src.PaddleInference import RectBoxPredictor

from scipy.spatial.distance import cosine

import os

class TemplateRepository(object):
    def __init__(self, model_dir, config_data_dict):
        self.config_data = config_data_dict
        self.image_pass_num = 0  
        self.feature_repository = {11: [], 12: [], 21: [], 22: []}
        self.temp_img_repository = {11: [], 12: [], 21: [], 22: []}
        self.defects_flag_records = {11: [], 12: [], 21: [], 22: []}

        self.feature_extractor = RectBoxPredictor(os.path.join(model_dir, 'TempModel'))
        self.img = None


    def feature_compare(self,f1,f2):
        op2 = cosine(f1, f2)

        return abs(op2)

    def get_features(self, img):
        return self.feature_extractor.predict(img)

    def features_compare(self, f1, f2):
        return self.feature_compare(f1, f2)


    def update_template(self, img, loc_id):
        self.temp_img_repository[loc_id].pop(0)
        self.feature_repository[loc_id].pop(0)
        self.add_template(img, loc_id)

    def update_config_data(self, config_data_dict):
        self.config_data = config_data_dict


    def add_template(self, img, loc_id):
        self.feature_repository[loc_id].append(self.get_features(img))
        self.temp_img_repository[loc_id].append(img)

    def clear_repository(self):
        self.temp_img_repository[11].clear()
        self.temp_img_repository[12].clear()
        self.temp_img_repository[21].clear()
        self.temp_img_repository[22].clear()

        self.feature_repository[11].clear()
        self.feature_repository[12].clear()
        self.feature_repository[21].clear()
        self.feature_repository[22].clear()


    def create_template(self, img, loc_id):
        if len(self.temp_img_repository[loc_id]) >= int(self.config_data['u_template_num']) * 2:
            self.update_template(img, loc_id)
        else:
            if self.image_pass_num >= int(self.config_data['u_image_pass_num']):
                self.add_template(img, loc_id)
            else:
                self.image_pass_num += 1


    def get_template(self, img, loc_id):
        temp_img = None
        try:
            target_code = self.get_features(img)
            compare_result = []
            for temp_code in self.feature_repository[loc_id]:
                compare_result.append(self.features_compare(temp_code, target_code))

            idx = compare_result.index(min(compare_result))
            temp_img = self.temp_img_repository[loc_id][idx]
            if len(compare_result) >= int(self.config_data['u_template_num']):

                tt_compare_result = copy.deepcopy(compare_result)
                tt_compare_result.pop(idx)
                idx = compare_result.index(min(tt_compare_result))

                self.temp_img_repository[loc_id].pop(idx)
                self.feature_repository[loc_id].pop(idx)

            self.add_template(img,loc_id)


        except Exception as e:
            pass
        return temp_img


    def find_template(self, img, loc_id):
        template_img = None
        if len(self.temp_img_repository[loc_id]) > 0:
            template_img = self.get_template(img, loc_id)
        else:
            self.add_template(img,loc_id)
 

        return template_img
