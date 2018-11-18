from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import types


class JointScale(object):
    """Rescales the input PIL.Image to the given 'size'.
    'size' will be the size of the smaller edge.
    For example, if height > width, then image will be
    rescaled to (size * height / width, size)
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        w, h = imgs[0].size
        if (w <= h and w == self.size) or (h <= w and h == self.size):
            return imgs
        if w < h:
            ow = self.size
            oh = int(self.size * h / w)
            return [img.resize((ow, oh), self.interpolation) for img in imgs]
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return [img.resize((ow, oh), self.interpolation) for img in imgs]


class JointResize(object):
    """Resize the input PIL.Image to the given 'size'.
    size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        return [img.resize((self.size, self.size), self.interpolation) for img in imgs]


class MaskResize(object):
    """Resize the input PIL.Image to the given 'size'.
    size should be an integer, in which case the target will be of a square shape (size, size)
    interpolation: Default: interpolation=Image.NEAREST
    """

    def __init__(self, size, interpolation=Image.NEAREST):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        return [img.resize((self.size, self.size), self.interpolation) for img in imgs]



class JointCenterCrop(object):
    """Crops the given PIL.Image at the center to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, imgs):
        w, h = imgs[0].size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointPad(object):
    """Pads the given PIL.Image on all sides with the given "pad" value"""

    def __init__(self, padding, fill=0):
        assert isinstance(padding, numbers.Number)
        assert isinstance(fill, numbers.Number) or isinstance(fill, str) or isinstance(fill, tuple)
        self.padding = padding
        self.fill = fill

    def __call__(self, imgs):
        return [ImageOps.expand(img, border=self.padding, fill=self.fill) for img in imgs]


class JointLambda(object):
    """Applies a lambda as a transform."""

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, imgs):
        return [self.lambd(img) for img in imgs]


class JointRandomCrop(object):
    """Crops the given list of PIL.Image at a random location to have a region of
    the given size. size can be a tuple (target_height, target_width)
    or an integer, in which case the target will be of a square shape (size, size)
    """

    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, imgs):
        if self.padding > 0:
            imgs = [ImageOps.expand(img, border=self.padding, fill=0) for img in imgs]

        w, h = imgs[0].size
        th, tw = self.size
        if w == tw and h == th:
            return imgs

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return [img.crop((x1, y1, x1 + tw, y1 + th)) for img in imgs]


class JointRandomHorizontalFlip(object):
    """Randomly horizontally flips the given list of PIL.Image with a probability of 0.5
    """

    def __call__(self, imgs):
        if random.random() < 0.5:
            return [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        return imgs


class JointRandomSizedCrop(object):
    """Random crop the given list of PIL.Image to a random size of (0.08 to 1.0) of the original size
    and and a random aspect ratio of 3/4 to 4/3 of the original aspect ratio
    This is popularly used to train the Inception networks
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        for attempt in range(10):
            area = imgs[0].size[0] * imgs[0].size[1]
            target_area = random.uniform(0.08, 1.0) * area
            aspect_ratio = random.uniform(3. / 4, 4. / 3)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= imgs[0].size[0] and h <= imgs[0].size[1]:
                x1 = random.randint(0, imgs[0].size[0] - w)
                y1 = random.randint(0, imgs[0].size[1] - h)

                imgs = [img.crop((x1, y1, x1 + w, y1 + h)) for img in imgs]
                assert(imgs[0].size == (w, h))

                return [img.resize((self.size, self.size), self.interpolation) for img in imgs]

        # Fallback
        scale = JointScale(self.size, interpolation=self.interpolation)
        crop = JointCenterCrop(self.size)
        return crop(scale(imgs))
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[125, 123, 113]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):

            img_np = np.asarray(img, dtype=np.uint8)
            area = img_np.shape[0] * img_np.shape[1]
            #area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_np.shape[1] and h < img_np.shape[0]:
                x1 = random.randint(0, img_np.shape[0] - h)
                y1 = random.randint(0, img_np.shape[1] - w)
                if img_np.shape[2] == 3:
                    img_np.flags.writeable = True ##
                    img_np[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img_np[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img_np[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]
                else:
                    img_np[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                img = Image.fromarray(img_np) ##rgb???
                return img
        return img

class RandomErasing_random(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al.
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=[125, 123, 113]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) > self.probability:
            return img

        for attempt in range(100):

            img_np = np.asarray(img, dtype=np.uint8)
            area = img_np.shape[0] * img_np.shape[1]
            #area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img_np.shape[1] and h < img_np.shape[0]:
                x1 = random.randint(0, img_np.shape[0] - h)
                y1 = random.randint(0, img_np.shape[1] - w)
                if img_np.shape[2] == 3:
                    img_np.flags.writeable = True ##
                    img_np[x1:x1 + h, y1:y1 + w, :] = 255 * np.random.rand(h,w,3)
                else:
                    img_np[x1:x1 + h, y1:y1 + w, 0] = 255 * np.random.rand(h,w,1)
                img = Image.fromarray(img_np) ##rgb???
                return img
        return img
