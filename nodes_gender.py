import cv2
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw
import time
import os

import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
import numpy as np
from PIL import Image
import comfy

portrait_enhancement = None

def get_portrait_enhancement():
    global portrait_enhancement
    if portrait_enhancement is None:
        portrait_enhancement = pipeline(Tasks.image_portrait_enhancement, model='damo/cv_gpen_image-portrait-enhancement', model_revision='v1.0.0')
    return portrait_enhancement

def tensorimg_to_cv2img(tensor_img):
    numpy_image = tensor_img.numpy()
    numpy_image = numpy_image * 255.0
    numpy_image = numpy_image.astype('uint8')
    rgb_image = cv2.cvtColor(numpy_image, cv2.COLOR_BGR2RGB)
    return rgb_image

def cv2img_to_tensorimg(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    numpy_image = np.array(img_rgb)
    numpy_image = numpy_image / 255.0
    tensor_img = torch.from_numpy(numpy_image)
    return tensor_img

def pilimg_to_tensorimg(pil_img):
    numpy_image = np.array(pil_img)
    tensor_img = torch.tensor(numpy_image, dtype=torch.float32) / 255.0
    return tensor_img

def tensorimg_to_pilimg(tensor_img):
    numpy_image = (tensor_img * 255).byte().numpy()
    
    numpy_image = np.clip(numpy_image, 0, 255).astype(np.uint8)
    
    pil_img = Image.fromarray(numpy_image)
    
    return pil_img


class GenderDetectionNode:
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("TEXT", )
    RETURN_NAMES = (
        "text",
    )

    FUNCTION = "run"
    CATEGORY = "GPEN"

    def run(self, images):

        image_out = []
        idx = 0
        pbar = comfy.utils.ProgressBar(len(images))
        for image in images:
            
            start_time = time.time()

            image = tensorimg_to_cv2img(image)
            fair_face_attribute_func = pipeline(Tasks.face_attribute_recognition, 'damo/cv_resnet34_face-attribute-recognition_fairface')

            rc_img_path = image
            raw_result = fair_face_attribute_func(src_img_path)

        return (
                raw_result, 
                )


NODE_CLASS_MAPPINGS = {

    "GenderDetectionNode": GenderDetectionNode,
}

# A dictionary that contains the friendly/humanly readable titles for the nodes
NODE_DISPLAY_NAME_MAPPINGS = {
    "GenderDetectionNode": "GenderDetectionNode",
}
