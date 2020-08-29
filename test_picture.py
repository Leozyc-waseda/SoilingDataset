
 
#!--*-- coding:utf-8 --*--
 
# Deeplab Demo 
#  This file calculates the final result and outputs the soling coverage
import os
import tarfile
 
from matplotlib import gridspec
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import tempfile
from six.moves import urllib
 
import tensorflow as tf
import cv2

 
class DeepLabModel(object):
    """
    加载 DeepLab 模型；
    推断 Inference.
    """
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513
    FROZEN_GRAPH_NAME = 'frozen_inference_graph'
 
    def __init__(self, tarball_path):
        """
        Creates and loads pretrained deeplab model.
        """
        self.graph = tf.Graph()
 
        graph_def = None
        # Extract frozen graph from tar archive.
        tar_file = tarfile.open(tarball_path)
        for tar_info in tar_file.getmembers():
            if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
                file_handle = tar_file.extractfile(tar_info)
                graph_def = tf.GraphDef.FromString(file_handle.read())
                break
 
        tar_file.close()
 
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
 
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
 
        self.sess = tf.Session(graph=self.graph)
 
 
    def run(self, image):
        """
        Runs inference on a single image.
        Args:
        image: A PIL.Image object, raw input image.
        Returns:
        resized_image: RGB image resized from original input image.
        seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(self.OUTPUT_TENSOR_NAME,
                                      feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map
 
 
def create_pascal_label_colormap():
    """
    Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
        A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
 
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3
 
    return colormap
 
 
def label_to_color_image(label):
    """
    Adds color defined by the dataset colormap to the label.
    Args:
        label: A 2D array with integer type, storing the segmentation label.
    Returns:
        result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.
    Raises:
        ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')
 
    colormap = create_pascal_label_colormap()
 
    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')
 
    return colormap[label]
 
 
def vis_segmentation(image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(15, 5))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])
 

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')
 
    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    Soiling_image = calculate_soiling_coverage(seg_image)  
    plt.imshow(Soiling_image)
    plt.axis('off')
#    plt.savefig('/home/ogai/Desktop/result.png')
    plt.title('segmentation map')
 
    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(Soiling_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')
 
    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
   # plt.grid('off')
    plt.show()


 
 
## 
LABEL_NAMES = np.asarray(
    ['clean', 'opaque', 'transparent' ])
 
FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)
 
 
## Tensorflow 提供的模型下载
MODEL_NAME = 'xception_coco_voctrainval'
# ['mobilenetv2_coco_voctrainaug', 'mobilenetv2_coco_voctrainval', 'xception_coco_voctrainaug', 'xception_coco_voctrainval']
 
_DOWNLOAD_URL_PREFIX = 'http://download.tensorflow.org/models/'
_MODEL_URLS = {'mobilenetv2_coco_voctrainaug': 'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
               'mobilenetv2_coco_voctrainval': 'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
               'xception_coco_voctrainaug': 'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
               'xception_coco_voctrainval': 'deeplabv3_pascal_trainval_2018_01_04.tar.gz', }
 
# _TARBALL_NAME = 'deeplab_model.tar.gz'
 
# model_dir = tempfile.mkdtemp()
# tf.gfile.MakeDirs(model_dir)
#
# download_path = os.path.join(model_dir, _TARBALL_NAME)
# print('downloading model, this might take a while...')
# urllib.request.urlretrieve(_DOWNLOAD_URL_PREFIX + _MODEL_URLS[MODEL_NAME], download_path)
# print('download completed! loading DeepLab model...')
 
 
 
download_path = '/home/ogai/Desktop/labeled_ubuntu/output_model/pb5_7w_5_25.tar.gz'  #模型所在位置
 
MODEL = DeepLabModel(download_path)
print('model loaded successfully!')
 
 
## 
def run_visualization(imagefile):
    """
    DeepLab 语义分割，并可视化结果.
    """
    orignal_im = Image.open(imagefile)
    print('running deeplab on image %s...' % imagefile)
    resized_im, seg_map = MODEL.run(orignal_im)
 
    vis_segmentation(resized_im, seg_map)


def calculate_soiling_coverage(image):
    Soiling_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.imshow('Soiling_image_test',Soiling_image)
#    image = cv2.imread('image')
    gray = cv2.cvtColor(Soiling_image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    total = 0

    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        print('w=',w,'h=',h)
    
        mask = np.zeros(Soiling_image.shape, dtype=np.uint8)
        cv2.fillPoly(mask, [c], [255,255,255])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        pixels = cv2.countNonZero(mask)
        total += pixels
        cv2.putText(Soiling_image, format(pixels), (x + w//3 ,y +  h//3 ), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    print('mask pixel',total)
    image_area = Soiling_image.shape[0] * Soiling_image.shape[1]
    area_ratio = (pixels / image_area) * 100
    coverage = round((total / image_area),3)*100

    cv2.putText(Soiling_image, ' soiling coverage = '+'{}'.format(coverage) +'%' , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    print('coverage of soiling part ',coverage)
    
    cv2.namedWindow("thresh",0)
    cv2.resizeWindow("thresh", 1600, 1200)
    cv2.imshow("thresh",thresh)

   
    cv2.namedWindow("Soiling_image",0)
    cv2.resizeWindow("Soiling_image", 1600, 1200)
    cv2.imshow("Soiling_image",Soiling_image)
    cv2.waitKey(0)

    Soiling_image = cv2.cvtColor(Soiling_image, cv2.COLOR_BGR2RGB)
    return Soiling_image

 
images_dir = '/home/ogai/Desktop/Soiling_dataset/test'  #测试图片目录所在位置
images = sorted(os.listdir(images_dir))
for imgfile in images:
    run_visualization(os.path.join(images_dir, imgfile))
print('Done.')