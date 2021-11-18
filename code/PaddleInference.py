import copy
import importlib
import time

import numpy as np
import cv2
import os
import yaml
import paddle
from paddle.inference import Config, create_predictor
from functools import reduce
from PIL import Image



RESIZE_SCALE_SET = {
    'RCNN',
    'RetinaNet',
    'FCOS',
    'SOLOv2',
}
def decode_image(im_file, im_info):
    """read rgb image
    Args:
        im_file (str/np.ndarray): path of image/ np.ndarray read by cv2
        im_info (dict): info of image
    Returns:
        im (np.ndarray):  processed image (np.ndarray)
        im_info (dict): info of processed image
    """
    if isinstance(im_file, str):
        with open(im_file, 'rb') as f:
            im_read = f.read()
        data = np.frombuffer(im_read, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        im_info['origin_shape'] = im.shape[:2]
        im_info['resize_shape'] = im.shape[:2]
    else:
        im = im_file
        im_info['origin_shape'] = im.shape[:2]
        im_info['resize_shape'] = im.shape[:2]
    return im, im_info

class Resize(object):
    """resize image by target_size and max_size
    Args:
        arch (str): model type
        target_size (int): the target size of image
        max_size (int): the max size of image
        use_cv2 (bool): whether us cv2
        image_shape (list): input shape of model
        interp (int): method of resize
    """

    def __init__(self,
                 arch,
                 target_size,
                 max_size,
                 use_cv2=True,
                 image_shape=None,
                 interp=cv2.INTER_LINEAR,
                 resize_box=False):
        self.target_size = target_size
        self.max_size = max_size
        self.image_shape = image_shape
        self.arch = arch
        self.use_cv2 = use_cv2
        self.interp = interp

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im_channel = im.shape[2]
        im_scale_x, im_scale_y = self.generate_scale(im)
        if self.use_cv2:
            im = cv2.resize(
                im,
                None,
                None,
                fx=im_scale_x,
                fy=im_scale_y,
                interpolation=self.interp)
        else:
            resize_w = int(im_scale_x * float(im.shape[1]))
            resize_h = int(im_scale_y * float(im.shape[0]))
            if self.max_size != 0:
                raise TypeError(
                    'If you set max_size to cap the maximum size of image,'
                    'please set use_cv2 to True to resize the image.')
            im = im.astype('uint8')
            im = Image.fromarray(im)
            im = im.resize((int(resize_w), int(resize_h)), self.interp)
            im = np.array(im)

        # padding im when image_shape fixed by infer_cfg.yml
        if self.max_size != 0 and self.image_shape is not None:
            padding_im = np.zeros(
                (self.max_size, self.max_size, im_channel), dtype=np.float32)
            im_h, im_w = im.shape[:2]
            padding_im[:im_h, :im_w, :] = im
            im = padding_im

        im_info['scale'] = [im_scale_x, im_scale_y]
        im_info['resize_shape'] = im.shape[:2]
        return im, im_info

    def generate_scale(self, im):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
        origin_shape = im.shape[:2]
        im_c = im.shape[2]
        if self.max_size != 0 and self.arch in RESIZE_SCALE_SET:
            im_size_min = np.min(origin_shape[0:2])
            im_size_max = np.max(origin_shape[0:2])
            im_scale = float(self.target_size) / float(im_size_min)
            if np.round(im_scale * im_size_max) > self.max_size:
                im_scale = float(self.max_size) / float(im_size_max)
            im_scale_x = im_scale
            im_scale_y = im_scale
        else:
            im_scale_x = float(self.target_size) / float(origin_shape[1])
            im_scale_y = float(self.target_size) / float(origin_shape[0])
        return im_scale_x, im_scale_y

class Normalize(object):
    """normalize image
    Args:
        mean (list): im - mean
        std (list): im / std
        is_scale (bool): whether need im / 255
        is_channel_first (bool): if True: image shape is CHW, else: HWC
    """

    def __init__(self, mean, std, is_scale=True, is_channel_first=False):
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        im = im.astype(np.float32, copy=False)
        if self.is_channel_first:
            mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
            std = np.array(self.std)[:, np.newaxis, np.newaxis]
        else:
            mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
            std = np.array(self.std)[np.newaxis, np.newaxis, :]
        if self.is_scale:
            im = im / 255.0
        im -= mean
        im /= std
        return im, im_info

class Permute(object):
    """permute image
    Args:
        to_bgr (bool): whether convert RGB to BGR
        channel_first (bool): whether convert HWC to CHW
    """

    def __init__(self, to_bgr=False, channel_first=True):
        self.to_bgr = to_bgr
        self.channel_first = channel_first

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        if self.channel_first:
            im = im.transpose((2, 0, 1)).copy()
        if self.to_bgr:
            im = im[[2, 1, 0], :, :]
        return im, im_info

class PadStride(object):
    """ padding image for model with FPN
    Args:
        stride (bool): model with FPN need image shape % stride == 0
    """

    def __init__(self, stride=0):
        self.coarsest_stride = stride

    def __call__(self, im, im_info):
        """
        Args:
            im (np.ndarray): image (np.ndarray)
            im_info (dict): info of image
        Returns:
            im (np.ndarray):  processed image (np.ndarray)
            im_info (dict): info of processed image
        """
        coarsest_stride = self.coarsest_stride
        if coarsest_stride == 0:
            return im
        im_c, im_h, im_w = im.shape
        pad_h = int(np.ceil(float(im_h) / coarsest_stride) * coarsest_stride)
        pad_w = int(np.ceil(float(im_w) / coarsest_stride) * coarsest_stride)
        padding_im = np.zeros((im_c, pad_h, pad_w), dtype=np.float32)
        padding_im[:, :im_h, :im_w] = im
        im_info['pad_shape'] = padding_im.shape[1:]
        return padding_im, im_info

def preprocess(im, preprocess_ops):
    # process image by preprocess_ops
    im_info = {
        'scale': [1., 1.],
        'origin_shape': None,
        'resize_shape': None,
        'pad_shape': None,
    }
    im, im_info = decode_image(im, im_info)
    for operator in preprocess_ops:
        im, im_info = operator(im, im_info)
    im = np.array((im,)).astype('float32')
    return im, im_info

class LoadConfig():
    """set config of preprocess, postprocess and visualize
    Args:
        model_dir (str): root path of model.yml
    """

    def __init__(self, model_dir):
        # parsing Yaml config for Preprocess
        deploy_file = os.path.join(model_dir, 'infer_cfg.yml')
        with open(deploy_file) as f:
            yml_conf = yaml.safe_load(f)

        self.arch = yml_conf['arch']
        self.preprocess_infos = yml_conf['Preprocess']
        self.use_python_inference = yml_conf['use_python_inference']
        self.min_subgraph_size = yml_conf['min_subgraph_size']
        self.labels = yml_conf['label_list']
        self.labels_threshold_dict = yml_conf['threshold_list']
        # print(self.labels_threshold)
        self.threshold = yml_conf['draw_threshold']
        self.mask_resolution = None
        if 'mask_resolution' in yml_conf:
            self.mask_resolution = yml_conf['mask_resolution']

class Detector(object):
    def __init__(self,model_dir):
        self.model_dir = model_dir
        self.config = LoadConfig(model_dir)
        self.predictor = self._init_predictor()

        self.labels = self.config.labels

        self.global_threshold = self.config.threshold
        self.local_threshold_dict = self.config.labels_threshold_dict

    def threshold_adjust(self,new_threshold,new_value_dict):
        self.local_threshold_dict = new_value_dict
        self.global_threshold = new_threshold

    def _init_predictor(self):
        config = Config()
        # config.set_model(model_dir)
        config.set_prog_file(os.path.join(self.model_dir, '__model__'))
        config.set_params_file(os.path.join(self.model_dir, '__params__'))
        # config = Config(model_dir)
        # config.disable_gpu()
        config.enable_use_gpu(1024, 0)
        config.switch_ir_optim(True)
        config.enable_memory_optim()

        # config.enable_tensorrt_engine(precision_mode=PrecisionType.Float32,
        #                               use_calib_mode=True)
        # config.enable_mkldnn()

        config.disable_glog_info()
        config.switch_use_feed_fetch_ops(False)

        return create_predictor(config)

    def preprocess(self, im):
        # preprocess_ops = []
        # for op_info in self.config.preprocess_infos:
        #     new_op_info = op_info.copy()
        #     op_type = new_op_info.pop('type')
        #     if op_type == 'Resize':
        #         new_op_info['arch'] = self.config.arch
        #     preprocess_ops.append(eval(op_type)(**new_op_info))
        # im, im_info = preprocess(im, preprocess_ops)
        # inputs = create_inputs(im, im_info, self.config.arch)
        # return inputs, im_info

        preprocess_ops = []
        for op_info in self.config.preprocess_infos:
            new_op_info = op_info.copy()
            op_type = new_op_info.pop('type')
            if op_type == 'Resize':
                new_op_info['arch'] = self.config.arch
            preprocess_ops.append(eval(op_type)(**new_op_info))
        im, im_info = preprocess(im, preprocess_ops)
        # inputs = create_inputs(im, im_info, self.config.arch)

        inputs = {}
        inputs['image'] = im
        origin_shape = list(im_info['origin_shape'])
        resize_shape = list(im_info['resize_shape'])
        pad_shape = list(im_info['pad_shape']) if im_info['pad_shape'] is not None else resize_shape
        scale_x, scale_y = im_info['scale']
        inputs['im_info'] = np.array([pad_shape + [scale_x]]).astype('float32')
        inputs['im_shape'] = np.array([origin_shape + [1.]]).astype('float32')

        return inputs, im_info

    def postprocess(self,np_boxes):
        results = {}
        expect_boxes = (np_boxes[:, 1] > float(self.global_threshold)) & (np_boxes[:, 0] > -1)
        np_boxes2 = np_boxes[expect_boxes, :]
        if len(np_boxes2):
            expect_boxes = []
            for index, label_id in enumerate(np_boxes2[:, 0]):
                label = self.labels[int(label_id)]
                if np_boxes2[index][1] >= self.local_threshold_dict[label] and label != 'SD':
                    expect_boxes.append(True)
                else:
                    expect_boxes.append(False)

            results['boxes'] = np_boxes2[expect_boxes, :]
        else:
            results['boxes'] = np_boxes2
        # results['boxes'] = np_boxes2

        return results

    def predict(self,image):
        '''
        Args:
            image (str/np.ndarray): path of image/ np.ndarray read by cv2
            threshold (float): threshold of predicted box' score
        Returns:
            results (dict): include 'boxes': np.ndarray: shape:[N,6], N: number of box,
                            matix element:[class, score, x_min, y_min, x_max, y_max]
                            MaskRCNN's results include 'masks': np.ndarray:
                            shape:[N, class_num, mask_resolution, mask_resolution]
        '''
        # t1 = time.time()
        inputs, im_info = self.preprocess(image)

        input_names = self.predictor.get_input_names()

        for i in range(len(input_names)):
            input_tensor = self.predictor.get_input_handle(input_names[i])
            input_tensor.copy_from_cpu(inputs[input_names[i]])

        self.predictor.run()
        output_names = self.predictor.get_output_names()
        boxes_tensor = self.predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()


        results = []
        if reduce(lambda x, y: x * y, np_boxes.shape) < 6:

            results = {'boxes': np.array([])}
        else:
            results = self.postprocess(np_boxes)


        return results


class AttrDict(dict):
    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, value):
        if key in self.__dict__:
            self.__dict__[key] = value
        else:
            self[key] = value

    def __deepcopy__(self, content):
        return copy.deepcopy(dict(self))


def create_attr_dict(yaml_config):
    from ast import literal_eval
    for key, value in yaml_config.items():
        if type(value) is dict:
            yaml_config[key] = value = AttrDict(value)
        if isinstance(value, str):
            try:
                value = literal_eval(value)
            except BaseException:
                pass
        if isinstance(value, AttrDict):
            create_attr_dict(yaml_config[key])
        else:
            yaml_config[key] = value

class NormalizeImage(object):
    """ normalize image such as substract mean, divide std
    """

    def __init__(self,
                 scale=None,
                 mean=None,
                 std=None,
                 order='chw',
                 output_fp16=False,
                 channel_num=3):
        if isinstance(scale, str):
            scale = eval(scale)
        assert channel_num in [
            3, 4
        ], "channel number of input image should be set to 3 or 4."
        self.channel_num = channel_num
        self.output_dtype = 'float16' if output_fp16 else 'float32'
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        self.order = order
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if self.order == 'chw' else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype('float32')
        self.std = np.array(std).reshape(shape).astype('float32')

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        assert isinstance(img,
                          np.ndarray), "invalid input 'img' in NormalizeImage"

        img = (img.astype('float32') * self.scale - self.mean) / self.std

        if self.channel_num == 4:
            img_h = img.shape[1] if self.order == 'chw' else img.shape[0]
            img_w = img.shape[2] if self.order == 'chw' else img.shape[1]
            pad_zeros = np.zeros(
                (1, img_h, img_w)) if self.order == 'chw' else np.zeros(
                    (img_h, img_w, 1))
            img = (np.concatenate(
                (img, pad_zeros), axis=0)
                   if self.order == 'chw' else np.concatenate(
                       (img, pad_zeros), axis=2))
        return img.astype(self.output_dtype)

class ToCHWImage(object):
    """ convert hwc image to chw image
    """

    def __init__(self):
        pass

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = np.array(img)

        return img.transpose((2, 0, 1))

class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass

class ResizeImage(object):
    """ resize image """

    def __init__(self, size=None, resize_short=None, interpolation=-1):
        self.interpolation = interpolation if interpolation >= 0 else None
        if resize_short is not None and resize_short > 0:
            self.resize_short = resize_short
            self.w = None
            self.h = None
        elif size is not None:
            self.resize_short = None
            self.w = size if type(size) is int else size[0]
            self.h = size if type(size) is int else size[1]
        else:
            raise OperatorParamError("invalid params for ReisizeImage for '\
                'both 'size' and 'resize_short' are None")

    def __call__(self, img):
        img_h, img_w = img.shape[:2]
        if self.resize_short is not None:
            percent = float(self.resize_short) / min(img_w, img_h)
            w = int(round(img_w * percent))
            h = int(round(img_h * percent))
        else:
            w = self.w
            h = self.h
        if self.interpolation is None:
            return cv2.resize(img, (w, h))
        else:
            return cv2.resize(img, (w, h), interpolation=self.interpolation)


class RectBoxPredictor(object):
    def __init__(self,model_dir):
        self.model_dir = model_dir
        self.cfg_args = self.parse_args()
        # config = config.get_config(args.config, overrides=args.override, show=True)

        self.paddle_predictor = self.create_paddle_predictor()
        self.preprocess_ops = self.create_operators(self.cfg_args["RecPreProcess"]["transform_ops"])

    def create_paddle_predictor(self):
        params_file = os.path.join(self.model_dir, "inference.pdiparams")
        model_file = os.path.join(self.model_dir, "inference.pdmodel")
        config = Config(model_file, params_file)

        args = self.cfg_args['Global']

        if args.use_gpu:
            config.enable_use_gpu(args.gpu_mem, 0)
        else:
            config.disable_gpu()
            if args.enable_mkldnn:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
        config.set_cpu_math_library_num_threads(args.cpu_num_threads)

        if args.enable_profile:
            config.enable_profile()
        config.disable_glog_info()
        config.switch_ir_optim(args.ir_optim)  # default true
        if args.use_tensorrt:
            config.enable_tensorrt_engine(
                precision_mode=Config.Precision.Half
                if args.use_fp16 else Config.Precision.Float32,
                max_batch_size=args.batch_size)

        config.enable_memory_optim()
        # use zero copy
        config.switch_use_feed_fetch_ops(False)
        predictor = create_predictor(config)

        return predictor

    def parse_args(self):
        deploy_file = os.path.join(self.model_dir, 'inference_rec.yaml')
        with open(deploy_file, 'r') as fopen:
            yaml_config = AttrDict(yaml.load(fopen, Loader=yaml.SafeLoader))
        create_attr_dict(yaml_config)
        return yaml_config

    def create_operators(self,params):
        """
        create operators based on the config

        Args:
            params(list): a dict list, used to create some operators
        """
        assert isinstance(params, list), ('operator config should be a list')
        mod = importlib.import_module(__name__)
        ops = []
        for operator in params:
            assert isinstance(operator,
                              dict) and len(operator) == 1, "yaml format error"
            op_name = list(operator)[0]
            param = {} if operator[op_name] is None else operator[op_name]
            op = getattr(mod, op_name)(**param)
            ops.append(op)

        return ops

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

if __name__ == '__main__':
    model_dir = r'F:\Project\Tile\PythonProject\TileDefectDetect_v13\model\MobileNetV1_infer'
    img = cv2.imread(r'F:\Project\Tile\PythonProject\TileDefectDetect_v13\config\pre_detect.jpg')

    rbp = RectBoxPredictor(model_dir)
    print(rbp.predict(img))





