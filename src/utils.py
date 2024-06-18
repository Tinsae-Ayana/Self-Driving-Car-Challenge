import torch
from torch import nn
import tensorflow as tf
from torchvision.models.segmentation import deeplabv3_resnet50 as deeplab, DeepLabV3_ResNet50_Weights as weights
from torchvision.transforms import Resize, ToTensor, Compose
import numpy as np
from cam2bev import uNetXST
import xml.etree.ElementTree as xmlET
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

"""
   The output of segmenation number is single channel with elements corresponde to the class number. 
   The color map is given below to each class
"""
color_map_segmentation = {
                            0: (0, 0, 0),
                            1: (0, 0, 0),
                            2: (0, 0, 0),
                            3: (0, 0, 0),
                            4: (0, 0, 0),
                            5: (111, 74, 0),
                            6: (81, 0, 81),
                            7: (128, 64, 128),
                            8: (244, 35, 232),
                            9: (250, 170, 160),
                            10: (230, 150, 140),
                            11: (70, 70, 70),
                            12: (102, 102, 156),
                            13: (190, 153, 153),
                            14: (180, 165, 180),
                            15: (150, 100, 100),
                            16: (150, 120, 90),
                            17: (153, 153, 153),
                            18: (153, 153, 153),
                            19: (250, 170, 30),
                            20: (220, 220, 0),
                            21: (107, 142, 35),
                            22: (152, 251, 152),
                            23: (70, 130, 180),
                            24: (220, 20, 60),
                            25: (255, 0, 0),
                            26: (0, 0, 142),
                            27: (0, 0, 70),
                            28: (0, 60, 100),
                            29: (0, 0, 90),
                            30: (0, 0, 110),
                            31: (0, 80, 100),
                            32: (0, 0, 230),
                            33: (119, 11, 32),
                            -1: (0, 0, 142)
                        }


# homograph matrix for front camera
H = [
     np.array([[0.03506686613905922, 27.971438297785962, -0.17694724954191404], 
            [0.3821882391578238, 9.481642330993019e-17, 5.46222110929461], 
            [25.000001047737943, 6.202207287472715e-15, 27.000001047737943]]) 
    ]

# semantic segmentation model
def create_seg_model(numclass=3):
    model = deeplab(weights=weights.DEFAULT)
    model.classifier[4] = torch.nn.Conv2d(256, numclass, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, numclass, kernel_size=(1,1), stride=(1,1))
    return model

# load semantic segmentation model
def load_seg_model():
    model = torch.load("weights/Deeplabv3.pth")
    return model.to("cpu")

# bird eye view model
def load_bev_model():
    model = uNetXST.get_network(input_shape=(256, 512, 10), n_output_channels=4, n_inputs= 1, thetas=H)
    model.load_weights("weights/best_weights.hdf5")
    return model

# transform the outof segmenation model to colored image
def apply_color_map_seg(output_tensor):
    # Create an empty tensor to hold the RGB image
    height, width  = output_tensor.shape
    colored_tensor = torch.zeros((height, width, 3), dtype=torch.uint8)
    
    for label_id, color in color_map_segmentation.items():
        mask = output_tensor == label_id
        for i in range(3): 
            colored_tensor[:, :, i][mask] = color[i]
    return colored_tensor

# hot encode the input segmented image into input of bev model
def one_hot_encode_image_op(image):       # image should be 

    palette = "src/one_hot_encoding/convert_10.xml"
    palette = parse_convert_xml(palette)
    one_hot_map = []

    for class_colors in palette:

        class_map = tf.zeros(image.shape[0:2], dtype=tf.int32)

        for color in class_colors:
            # find instances of color and append layer to one-hot-map
            class_map = tf.bitwise.bitwise_or(class_map, tf.cast(tf.reduce_all(tf.equal(image, color), axis=-1), tf.int32))
        one_hot_map.append(class_map)

    # finalize one-hot-map
    one_hot_map = tf.stack(one_hot_map, axis=-1)
    one_hot_map = tf.cast(one_hot_map, tf.float32)

    return one_hot_map

# decode the output of the bev model
def one_hot_decode_image(one_hot_image):   # one_hot_image should be the output of the bev model

    palette = "src/one_hot_encoding/convert_3+occl.xml"
    palette = parse_convert_xml(palette)

    # create empty image with correct dimensions
    height, width = one_hot_image.shape[0:2]
    depth = palette[0][0].size
    image = np.zeros([height, width, depth])

    # reduce all layers of one-hot-encoding to one layer with indices of the classes
    map_of_classes = one_hot_image.argmax(2)

    for idx, class_colors in enumerate(palette):
        # fill image with corresponding class colors
        image[np.where(map_of_classes == idx)] = class_colors[0]

    image = image.astype(np.uint8)

    return image

# change xml file to color map
def parse_convert_xml(conversion_file_path):

    defRoot = xmlET.parse(conversion_file_path).getroot()

    one_hot_palette = []
    class_list = []
    for idx, defElement in enumerate(defRoot.findall("SLabel")):
        from_color = np.fromstring(defElement.get("fromColour"), dtype=int, sep=" ")
        to_class = np.fromstring(defElement.get("toValue"), dtype=int, sep=" ")
        if to_class in class_list:
             one_hot_palette[class_list.index(to_class)].append(from_color)
        else:
            one_hot_palette.append([from_color])
            class_list.append(to_class)

    return one_hot_palette

# load image from directory 
def load_image(file_location):
    img = Image.open(file_location).convert("RGB")
    img_transform = Compose([
        Resize((256, 512)),
        ToTensor()
    ])
    return img_transform(img).squeeze(0)

# visualize image
def visualize_image(image1, image2, image3):
    plt.subplot(1,3,1)
    plt.axis("off")
    plt.imshow(image1)
    plt.subplot(1,3,2)
    plt.axis("off")
    plt.imshow(image2)
    plt.subplot(1,3,3)
    plt.axis("off")
    plt.imshow(image3)
    plt.show()