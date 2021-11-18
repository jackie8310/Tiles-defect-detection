import os
import cv2
import numpy as np
import paddle


class FeatureCnn(object):
    def __init__(self,model_dir):
        
        self.model = paddle.vision.models.vgg16(pretrained=False, batch_norm=False, num_classes=0)
        para_state_dict = paddle.load(os.path.join(model_dir,'vgg16.pdparams'))
        self.model.load_dict(para_state_dict)

 

    def get_feature_code(self,img):
        img2 = cv2.resize(img, (224, 224))
        img2 = img2 / 255. * 2. - 1.
        img2 = np.transpose(img2, (2, 0, 1)).astype('float32')

        
        tensor = paddle.to_tensor(np.expand_dims(img2, 0))

        feature = self.model(tensor).numpy().flatten()
        return np.mat(feature)

    def feature_compare(self,a,b):
        s1 = np.linalg.norm(a)
        s2 = np.linalg.norm(b)
        return float(a * b.T) / (s1 * s2)


