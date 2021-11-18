"""
# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import math
import random
import cv2
import numpy as np
import importlib
from PIL import Image, ImageEnhance

# from python.det_preprocess import DetNormalizeImage, DetPadStride, DetPermute, DetResize


def create_operators(params):
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


class OperatorParamError(ValueError):
    """ OperatorParamError
    """
    pass


class DecodeImage(object):
    """ decode image """

    def __init__(self, to_rgb=True, to_np=False, channel_first=False):
        self.to_rgb = to_rgb
        self.to_np = to_np  # to numpy
        self.channel_first = channel_first  # only enabled when to_np is True

    def __call__(self, img):
        if six.PY2:
            assert type(img) is str and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        else:
            assert type(img) is bytes and len(
                img) > 0, "invalid input 'img' in DecodeImage"
        data = np.frombuffer(img, dtype='uint8')
        img = cv2.imdecode(data, 1)
        if self.to_rgb:
            assert img.shape[2] == 3, 'invalid shape of image[%s]' % (
                img.shape)
            img = img[:, :, ::-1]

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        return img


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


class CropImage(object):
    """ crop image """

    def __init__(self, size):
        if type(size) is int:
            self.size = (size, size)
        else:
            self.size = size  # (h, w)

    def __call__(self, img):
        w, h = self.size
        img_h, img_w = img.shape[:2]

        if img_h < h or img_w < w:
            raise Exception(
                f"The size({h}, {w}) of CropImage must be greater than size({img_h}, {img_w}) of image. Please check image original size and size of ResizeImage if used."
            )

        w_start = (img_w - w) // 2
        h_start = (img_h - h) // 2

        w_end = w_start + w
        h_end = h_start + h
        return img[h_start:h_end, w_start:w_end, :]


class RandCropImage(object):
    """ random crop image """

    def __init__(self, size, scale=None, ratio=None, interpolation=-1):

        self.interpolation = interpolation if interpolation >= 0 else None
        if type(size) is int:
            self.size = (size, size)  # (h, w)
        else:
            self.size = size

        self.scale = [0.08, 1.0] if scale is None else scale
        self.ratio = [3. / 4., 4. / 3.] if ratio is None else ratio

    def __call__(self, img):
        size = self.size
        scale = self.scale
        ratio = self.ratio

        aspect_ratio = math.sqrt(random.uniform(*ratio))
        w = 1. * aspect_ratio
        h = 1. / aspect_ratio

        img_h, img_w = img.shape[:2]

        bound = min((float(img_w) / img_h) / (w**2),
                    (float(img_h) / img_w) / (h**2))
        scale_max = min(scale[1], bound)
        scale_min = min(scale[0], bound)

        target_area = img_w * img_h * random.uniform(scale_min, scale_max)
        target_size = math.sqrt(target_area)
        w = int(target_size * w)
        h = int(target_size * h)

        i = random.randint(0, img_w - w)
        j = random.randint(0, img_h - h)

        img = img[j:j + h, i:i + w, :]
        if self.interpolation is None:
            return cv2.resize(img, size)
        else:
            return cv2.resize(img, size, interpolation=self.interpolation)


class RandFlipImage(object):
    """ random flip image
        flip_code:
            1: Flipped Horizontally
            0: Flipped Vertically
            -1: Flipped Horizontally & Vertically
    """

    def __init__(self, flip_code=1):
        assert flip_code in [-1, 0, 1
                             ], "flip_code should be a value in [-1, 0, 1]"
        self.flip_code = flip_code

    def __call__(self, img):
        if random.randint(0, 1) == 1:
            return cv2.flip(img, self.flip_code)
        else:
            return img



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


class Contrast(object):
    def __init__(self,factor):
        self.factor = factor

    def __call__(self, img):
        img = ImageEnhance.Contrast(Image.fromarray(img)).enhance(self.factor)
        return np.array(img)
        # mean = np.uint8(cv2.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))[0])
        # img_deg = np.ones_like(img) * mean
        # return cv2.addWeighted(img, self.factor, img_deg, 1 - self.factor, 0.0)

class Sharpness(object):
    def __init__(self,factor):
        self.factor = factor

    def __call__(self, img):
        img = ImageEnhance.Sharpness(Image.fromarray(img)).enhance(self.factor)
        return np.array(img)
    # """Implements Sharpness function from PIL."""
    # def __init__(self,factor):
    #     self.factor = factor
    # def blend(self,image1, image2, factor):
    #     """Blend image1 and image2 using 'factor'.
    #
    #     Factor can be above 0.0.    A value of 0.0 means only image1 is used.
    #     A value of 1.0 means only image2 is used.    A value between 0.0 and
    #     1.0 means we linearly interpolate the pixel values between the two
    #     images.    A value greater than 1.0 "extrapolates" the difference
    #     between the two pixel values, and we clip the results to values
    #     between 0 and 255.
    #
    #     Args:
    #         image1: An image Tensor of type uint8.
    #         image2: An image Tensor of type uint8.
    #         factor: A floating point value above 0.0.
    #
    #     Returns:
    #         A blended image Tensor of type uint8.
    #     """
    #     if factor == 0.0:
    #         return image1
    #     if factor == 1.0:
    #         return image2
    #
    #     image1 = image1.astype(np.float32)
    #     image2 = image2.astype(np.float32)
    #
    #     difference = image2 - image1
    #     scaled = factor * difference
    #
    #     # Do addition in float.
    #     temp = image1 + scaled
    #
    #     # Interpolate
    #     if factor > 0.0 and factor < 1.0:
    #         # Interpolation means we always stay within 0 and 255.
    #         return temp.astype(np.uint8)
    #
    #     # Extrapolate:
    #     #
    #     # We need to clip and then cast.
    #     return np.clip(temp, a_min=0, a_max=255).astype(np.uint8)
    # def __call__(self, img):
    #     orig_image = img
    #     image = img.astype(np.float32)
    #     # Make image 4D for conv operation.
    #     # SMOOTH PIL Kernel.
    #     kernel = np.array([[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=np.float32) / 13.
    #     result = cv2.filter2D(image, -1, kernel).astype(np.uint8)
    #
    #     # Blend the final result.
    #     return self.blend(result, orig_image, self.factor)

class Autocontrast(object):
    """Implements Autocontrast function from PIL.

    Args:
        image: A 3D uint8 tensor.

    Returns:
        The image after it has had autocontrast applied to it and will be of type
        uint8.
    """
    def __init__(self):
        pass

    def scale_channel(self,image):
        """Scale the 2D image using the autocontrast rule."""
        # A possibly cheaper version can be done using cumsum/unique_with_counts
        # over the histogram values, rather than iterating over the entire image.
        # to compute mins and maxes.
        lo = float(np.min(image))
        hi = float(np.max(image))

        # Scale the image, making the lowest value 0 and the highest value 255.
        def scale_values(im):
            scale = 255.0 / (hi - lo)
            offset = -lo * scale
            im = im.astype(np.float32) * scale + offset
            img = np.clip(im, a_min=0, a_max=255.0)
            return im.astype(np.uint8)

        result = scale_values(image) if hi > lo else image
        return result
    def __call__(self, img):
    # Assumes RGB for now.    Scales each channel independently
    # and then stacks the result.
        s1 = self.scale_channel(img[:, :, 0])
        s2 = self.scale_channel(img[:, :, 1])
        s3 = self.scale_channel(img[:, :, 2])
        image = np.stack([s1, s2, s3], 2)
        return image