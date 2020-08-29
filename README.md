# SoilingDataset 
Due to the postponement of the release of the woodscape dataset, I plan to create a soiling dataset myself for research.

# Deeplabv3+ train your own dataset

#  1 数据集
## 1.1 取得数据
  - 将摄像头装到toyota coms车的后方位置，四个摄像头并排，一起取得图像。其中3个脏的摄像头，一个干净的摄像头（干净的镜头是为了方便以后desoiling，或者用CycleGAN来生成假数据的情况下，需要一个完全干净的图片） 
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/IMG_4905.jpeg)
  - 脏污是由我自己手动弄脏并且涂抹在摄像头的镜头上，用泥土+水+牙刷去制造Soiling的数据 
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/IMG_4912.jpeg)
  - 录制的软件都是由MacBook Pro(2020)完成，其中使用了两个type-c hub防止供电不足导致不能录影，录影软件使用的是Camera Record-Camera Recorder 付费在AppStore里面下载，非常小巧以及稳定，同时录制4个视频画面没有卡顿
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/%E6%88%AA%E5%B1%8F2020-07-12%20%E4%B8%8B%E5%8D%886.11.18.png)
  -  录影规格：30FPS，分辨率是1920，1080
  -  分三天不同的天气和时间点，驾驶toyota coms，时速大约30km/h; coms绕学校附近一圈，起点：北九工大车库，终点：北九工大车库；每段时间时间约为6分钟左右（绕一圈的时间），每天两次，每次4个视频，最后得到32个视频，每个视频6分钟
 
## 1.2 处理数据

  - 首先将video转换成一帧一帧的图片，用Video2Picture.py这个文件，修改输入和输出的文件夹即可使用
  - 删除开始和最后的相关图片（因为都是稳定的不动的照片），只选择每隔60帧的图片（增加多样性），汇集能用的图片，批量将他们命名
  ```sh
  #批量处理文件，将该文件目录下的所有文件按顺序排序，命名,linux
 c=9;for i in *.png;do mv -f $i $((c+=1)).png;done
 
 ＃将２个连续数字的文件名字，更换成soiling1_ｘ，ｘ和原文件一一对应
 rename -v 's/(\d{2})\.png/soiling1_$1.png/' *.png
  ```
  - 其实最后导出来的图片挺多的，只是手动标注太费时间和精力了，最后dataset里面只有184张照片

## 1.3 标注数据
labelme我是在[anaconda][df3]上面运行的，添加了一个虚拟环境（强烈建议用anaconda，下载各种包，备份工作环境都很方便）
```sh
#在anaconda下建立一个labelme的环境
# python2
conda create --name=labelme python=2.7
source activate labelme
# conda install -c conda-forge pyside2
conda install pyqt
pip install labelme
# if you'd like to use the latest version. run below:
# pip install git+https://github.com/wkentaro/labelme.git

# python3
conda create --name=labelme python=3.6
source activate labelme
# conda install -c conda-forge pyside2
# conda install pyqt
# pip install pyqt5  # pyqt5 can be installed via pip on python3
pip install labelme
# or you can install everything by conda command
# conda install labelme -c conda-forge
```
标注数据是用的是[Github labelme][df1]，我感觉非常好用挺适合小白的,PS：日立也出了个标注工具[Hitachi-semantic-segmentation-editor][df2],感觉很适合标语义分割(新手不推荐)，但是Mac上用的不方便我就用[labelme][df1]了
```sh
# macOS Sierra
brew install pyqt  # maybe pyqt5
pip install labelme  # both python2/3 should work

# or install standalone executable/app from:
# https://github.com/wkentaro/labelme/releases
```



  - 标注主要分了两个类，一个是opaque，一个是transparent，background是clean
  - 标注异常费力，几乎每张图片都需要花费我几乎1个小时去标注，细分透明和不透明的部分，最终得到了184张图片
  ![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/labelme.png)
  - 其实有大牛已经在研究这个领域，并在去年的ICCV上发表了paper介绍了自己的数据集，只是由于新冠病毒的影响，以及欧洲对数据的新规定（公共数据集必须给人脸和车牌马赛克，不然不允许发布），所以推迟了一年，虽然目前为止2020年8月13日都还没有发布。我也问过作者他说他将会在今年Q3，Q4的时候发布，但是现在看来情势也不容乐观。最迟推断大概明天就会有soiling相关的dataset，大约接近5000张的数据，那么就不用辛苦的做数据集了
  - 由于数据集的稀少和环境相似，其实数据集多样性和鲁棒性都不是很好，测试集和训练集也是高度相似，这点如何解决我还没有任何头绪
  - 在Deep Learning中，数据集起码要有3000~5000张，才会有不错的效果，所以语义分割的数据集是非常昂贵和难以获取的。我也想去淘宝上花钱雇人帮我标注，只是15RMB一张的图片让我望而却步，其他研究者都是委托专门的数据公司进行标注，所以我的小研究就靠184张图片来搞吧
  - 未来可能会用CycleGAN之类的技术去建立假数据，扩充我的数据集，增加多样性和鲁棒性

 

 [df1]: <https://github.com/wkentaro/labelme>
 [df2]: <https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor>
 [df3]: <https://www.anaconda.com/products/individual>

#  2  用[DeeplabV3+][df4]训练自己的数据集

> 目前在语义分割领域成功的Semantic Segmentation的models有很多，之所以选择[Deeplabv3+][df4]只是他比较符合我的需求
> 而且以后做成Video的版本也会比较方便（网上代码多~）

未来我也打算尝试一下Mask R-CNN什么的

 [df4]: <https://github.com/tensorflow/models/tree/master/research/deeplab>

## 2.1 训练[Deeplabv3+][df4]之前环境搭建，基于Ubuntu16.04系统

下面给出配置顺序:

* [Nvidia驱动] - Nvidia driver Downloads，选择自己对应的GPU显卡（例如我的卡是GeForce GTX 1070Ti）,下载后会得到一个NVIDIA-Linux-x86_64-450.run的文件（显卡不同450数字也会变为其他数字），好好保存这个文件，记住他的路径，因为未来显卡崩了重装是很常见的事情 T T
```sh
#Linux显卡崩了修复步骤
#1. control + alt + F1
#2. 切换到刚才下载.run后缀的文件夹里
$ cd /home/user/Downloads
#3. 关闭图形显表
$ sudo service lightdm stop
#4. 运行.run文件
$ sudo ./NVIDIA-Linux-x86_64-450.run
#5. 重启
$ sudo reboot
```
若是安装成功，通过下面的命令就能打开GUI的显卡操作界面
```sh
#GUI查看
$ nvidia-settings
#查看显卡
$ nvidia-smi
```
* [安装CUDA] - ``` nvcc  -V ```查看cuda版本
* [安装CuDNN] - 网上很多详细解释，找找就行
* [安装Anaconda] - Anaconda 是一个用于科学计算的 Python 发行版，支持 Linux, Mac, Windows, 包含了众多流行的科学计算、数据分析的 Python 包 。
安装完后``` anaconda-navigator```的命名打开anaconda的GUI界面
1. 在Anaconda下面装tensorflow
```sh
#这步也可以在anaconda-navigator打开后点击鼠标创建
$ conda create -n tfgpu python=3.7
```
2. 激活tfgpu的环境
```sh
$ source activate tfgpu
```
3. 下载各种包，具体可以google ```Tensorflow gpu```,也可以在anaconda-navigator上面点击鼠标安装
```sh
$ pip3 install tensorflow==1.15
$ pip3 install tensorflow_gpu==1.15
```
4. 测试安装环境是否完成
打开[DeeplabV3+][df4]
```sh
#clone一下deeplab tensorflow的github
$ git clone https://github.com/tensorflow/models.git
#deeplab的文件夹在 home/models/research/deeplab
$ cd ~/home/models/research/deeplab
#有一个叫model_test.py的文件，这个脚本是测试你的电脑是否安装好了所有deeplab训练所必须的环境，输出Ok代表没问题，若是缺什么包看命名行报错再继续更改
$ python3 model_test.py
# ps ：github里面作者也有些，这段话也得加上，不然会有很多错误
export PYTHONPATH=/home/bai/models/research/slim:/home/bai/models/research:$PYTHONPATH
```
到现在为止为了训练deeplabv3+的环境就设置完成了，运行model_test.py后没有输出Ok的情况下一定要修复到出现Ok在进行下一步。


 [Nvidia驱动]: <https://www.nvidia.com/Download/index.aspx?lang=en-us>

 [安装CUDA]: <https://developer.nvidia.com/cuda-toolkit>

 [安装CuDNN]: <https://developer.nvidia.com/rdp/cudnn-download>
 
 [安装Anaconda]: <https://www.anaconda.com/products/individual>


## 2.2 自己做好的数据集的图像处理

### 2.2.1 labelme标注后文件的处理
使用labelme标注后的文件都是.json后缀，里面记录了你标注的信息，我们要使用它作为网络的输入数据的话还需要做一些处理
* json文件需要解析之后才能得到标签图片。这里使用```labelme2voc.py```生成```voc```数据集格式的数据集，其中产生的label图可以保证每一类的编号都是一致的。 
* 该脚本转换完成的label图是```8-bit```的```RGB```图，需要再转换成灰度图。

转换步骤:
(1)  训练数据集生成标签图
建立文件夹```~/Desktop/Soiling_dataset/dataset_train```，并在```dataset_train```下建立子文件夹```data_annotated```。
把训练数据集图像被labelme标注的```json```文件放置到```dataset_train/data_annotated```目录下。
在```~/Desktop/Soiling_dataset/```目录下执行:
```sh
$ python3 labelme2voc.py dataset_train/data_annotated dataset_train/data_dataset_voc --labels labels.txt
```
其中，labels.txt中是需要分割的物体的类别。包括:
```sh
__ignore__
clean
opaque
transparent
```
执行后生成:
* data_dataset_voc/JPEGImages 
* data_dataset_voc/SegmentationClass 
* data_dataset_voc/SegmentationClassVisualization

(2)  测试集的数据集生成标签图
* 同理，建一个dataset_val的文件，修正一下代码就好

(3) mask灰度值的转换
1. 对训练集图片去除```colormap```:
deeplab的官方给出了去掉```colormap```的程序，在```~/models/research/deeplab/datasets/remove_gt_colormap.py```里面
```sh
#在自己的dataset文件下
$ cd `~/Desktop/Soiling_dataset/
#SegmentationClassPNG-raw文件夹下面就是我们训练需要用上的灰度图
$ cd ~/Desktop/labeled_ubuntu/dataset_train
1
python3 labelme2voc.py train_annotated train_dataset_voc --labels labels.txt

2
python3 labelme2voc.py val_annotated val_dataset_voc --labels labels.txt

3
python3 labelme2voc.py test_annotated test_dataset_voc --labels labels.txt

python3 ~/models/research/deeplab/datasets/remove_gt_colormap.py --original_gt_folder train_dataset_voc/SegmentationClassPNG --output_dir train_dataset_voc/SegmentationClassPNG-raw

python3 ~/models/research/deeplab/datasets/remove_gt_colormap.py --original_gt_folder val_dataset_voc/SegmentationClassPNG --output_dir val_dataset_voc/SegmentationClassPNG-raw

python3 ~/models/research/deeplab/datasets/remove_gt_colormap.py --original_gt_folder test_dataset_voc/SegmentationClassPNG --output_dir test_dataset_voc/SegmentationClassPNG-raw
```
2. 同理对测试集的图片也去除```colormap```

### 2.2.2 汇总制作tfrecord文件
数据集的文件夹结构为:

├── test

├── testannot

├── train

├── trainannot

├── val

└── valannot 由上到下分别是测试集、测试集标签、训练集、训练集标签、验证集、验证集标签。

制作tfrecord的文件
TF 提供了一种统一输入数据的格式——``` TFRecord ```它有两个优点:
1. 可以将一个样本的所有信息统一起来存储，这些信息可以是不同的数据类型。其内部使用“Protocol Buffer”二进制数据编码方案。
2. 利用文件队列的多线程操作，使得数据的读取和批量处理更加方便快捷。

在制作TFRecord之前，需要有文件指引将数据集分类成训练集、测试集、验证集，故需要创建指引文件。 将所有图片和mask分在两个文件夹下，设置如下:
```~/Desktop/Soiling_dataset/image```: 存放所有的输入图片，共有184张，这其中包括训练集、测试集、 验证集的图片。
```~/Desktop/Soiling_dataset/mask```:存放所有的标签图片，共有184张，和image文件夹下的图片是一一对应的。
对于CamVid数据集，创建了一个目录```~/Desktop/Soiling_dataset/index```，该目录下包含三个.txt文件:
train.txt:所有训练集的文件名称 trainval.txt:所有验证集的文件名称 val.txt:所有测试集的文件名称

同样deeplab官方也提供了```build_voc2012_data.py```的脚步，让我们转换数据
```sh
#tfrecord是最后得到的文件夹
$ python3 ~/models/research/deeplab/datasets/build_voc2012_data.py \ --image_folder="~/Desktop/Soiling_dataset/image" \ --semantic_segmentation_folder="~/Desktop/Soiling_dataset/mask" \ --list_folder="~/Desktop/Soiling_dataset/index" \ --image_format="jpg" \ --output_dir="~/Desktop/Soiling_dataset/tfrecord"
```


### 2.2.3 修改训练脚本

* 在DeepLabv3+模型的基础上，主要需要修改以下两个文件
* ```data_generator.py``` 文件``` train_utils.py```

1. 添加数据集描述
在datasets/data_generator.py文件中，添加自己的数据集描述:
```sh
_MYDATA_INFORMATION = DatasetDescriptor(
    splits_to_sizes={
        'train': 128,  # num of samples in images/training
        'val': 36,  # num of samples in images/validation
    },
    num_classes=4, #为4的原因：ignore_lable + clean(background) + opaque(class1) +transparent(class2)
    ignore_label=255, # 忽略像素点为(255，255，255)既白色
)
```

2. 注册数据集
同时在datasets/data_generator.py文件，添加对应数据集的名称:
```sh
_DATASETS_INFORMATION = {
'cityscapes': _CITYSCAPES_INFORMATION, 
'pascal_voc_seg': _PASCAL_VOC_SEG_INFORMATION, 
'ade20k': _ADE20K_INFORMATION, 
'soiling_dataset':_SOILING_DATASET #自己的数据集
}
```

3. 修改train_utils.py
对应的utils/train_utils.py中，将159行关于 exclude_list 的设置修改，作用是在使用预训练权重时
候，不加载该 logit 层:
```sh
exclude_list = ['global_step','logits'] if not initialize_last_layer: exclude_list.extend(last_layers)
```

4. 训练和可视化
如果想在DeepLab的基础上fine-tune其他数据集， 可在```deeplab/train.py```中修改输入参数。 
一些选项:
* 使用预训练的所有权重，设置```initialize_last_layer=True ```
* 只使用网络的backbone，设置```initialize_last_layer=Fals```e和``` last_layers_contain_logits_only=False ```
* 使用所有的预训练权重，除了logits。因为如果是自己的数据集，对应的classes不同(这个我们前 面已经设置不加载logits),可设置```initialize_last_layer=False```和``` last_layers_contain_logits_only=True```

这里使用的设置是: 
```initialize_last_layer=False``` #157行 
``` last_layers_contain_logits_only=True ```#160行 


## 2.3 训练网络

### 2.3.1 模型的下载
在model_zoo上下载预训练模型: 下载地址:https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zo
o.md
由于我想用的backbone是```xception```我下载的训练模型是```xception_cityscapes_trainfine```

下载到deeplab目录下，然后解压:
```tar -zxvf deeplabv3_cityscapes_train_2018_02_06.tar.gz ```
需要注意对应的解压文件目录为:
```/home/user/models/research/deeplab/deeplabv3_cityscapes_train```

### 2.3.2 Imbalance的调整与修正

1. ENet_weighing修正idea 来自[ENet paper](https://arxiv.org/abs/1606.02147)
```sh
def ENet_weighing(image_files=image_files, num_classes=12):
    '''
    The custom class weighing function as seen in the ENet paper.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    #initialize dictionary with all 0
    label_to_frequency = {}
    for i in range(num_classes):
        label_to_frequency[i] = 0

    for n in range(len(image_files)):
        image = imread(image_files[n])

        #For each label in each image, sum up the frequency of the label and add it to label_to_frequency dict
        for i in range(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            label_to_frequency[i] += class_frequency

    #perform the weighing function label-wise and append the label's class weights to class_weights
    class_weights = []
    total_frequency = sum(label_to_frequency.values())
    for label, frequency in label_to_frequency.items():
        class_weight = 1 / np.log(1.02 + (frequency / total_frequency))
        class_weights.append(class_weight)

    #Set the last class_weight to 0.0
    class_weights[-1] = 0.0

    return class_weights
```
#自己的笔记
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/UNet1.png)
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/UNet2.png)

2. Median frequency balancing 计算idea，来自[SegNet paper](https://arxiv.org/abs/1511.00561)

```sh
def median_frequency_balancing(image_files=image_files, num_classes=12):
    '''
    Perform median frequency balancing on the image files, given by the formula:
    f = Median_freq_c / total_freq_c
    where median_freq_c is the median frequency of the class for all pixels of C that appeared in images
    and total_freq_c is the total number of pixels of c in the total pixels of the images where c appeared.
    INPUTS:
    - image_files(list): a list of image_filenames which element can be read immediately
    - num_classes(int): the number of classes of pixels in all images
    OUTPUTS:
    - class_weights(list): a list of class weights where each index represents each class label and the element is the class weight for that label.
    '''
    #Initialize all the labels key with a list value
    label_to_frequency_dict = {}
    for i in range(num_classes):
        label_to_frequency_dict[i] = []

    for n in range(len(image_files)):
        image = imread(image_files[n])

        #For each image sum up the frequency of each label in that image and append to the dictionary if frequency is positive.
        for i in range(num_classes):
            class_mask = np.equal(image, i)
            class_mask = class_mask.astype(np.float32)
            class_frequency = np.sum(class_mask)

            if class_frequency != 0.0:
                label_to_frequency_dict[i].append(class_frequency)

    class_weights = []

    #Get the total pixels to calculate total_frequency later
    total_pixels = 0
    for frequencies in label_to_frequency_dict.values():
        total_pixels += sum(frequencies)

    for i, j in label_to_frequency_dict.items():
        j = sorted(j) #To obtain the median, we got to sort the frequencies

        median_frequency = np.median(j) / sum(j)
        total_frequency = sum(j) / total_pixels
        median_frequency_balanced = median_frequency / total_frequency
        class_weights.append(median_frequency_balanced)

    #Set the last class_weight to 0.0 as it's the background class
    class_weights[-1] = 0.0

    return class_weights

if __name__ == "__main__":
#改一下图片输入地址就好
    median_frequency_balancing(image_files, num_classes=4)
    ENet_weighing(image_files, num_classes=4)
```
#自己的笔记
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/segnet_media.png)

3. 调整class weights

最后```media_frequency_balancing class_weights:[0.012739 , 0.0207334,, 0.0541833 ]```
```ENet_weighting class_weights:[1.865889 , 4.895958 , 8.532353 , 50.498]```

在```train_untils.py```下面修改权重
```sh
#在这个程序后面添加
scaled_lables = tf.reshape(scaled_labels,shape=[-1])
ignore_weight = 0
label0_weight = 1 # background clean
label1_weight =  5 # class 1 opaque 
label2_weight = 30 # class 2 transparent
not_ignore_mask = tf.to_float(tf.equal(scaled_labeles,0)) * label0_weight + tf.to_float(tf.equal(scaled_labels,1)) * label1_weight +tf.to_float(tf.equal(scaled_labels,2)) * label2_weight + tf.to_float(tf.equal(scaled_labels,ignore_label))

#或者修改loss_weights=1.0这行代码
loss_weights=[0 , 1 , 5 ,30]
```

### 2.3.3 开始训练

注意如下几个参数:
* ```tf_initial_checkpoint```:预训练的权重，因为数据集都和CityScapes类似，所以使用的是CityScapes的预训练权重
* ```train_logdir```: 训练产生的文件存放位置
* ```dataset_dir```: 数据集的TFRecord文件 
* ```dataset```:设置为在data_generator.py文件设置的数据集名称

在自己的数据集上的训练指令如下:
在目录 ```~/models/research/deeplab```下执行

```sh 
python train.py \
--logtostderr \
--training_number_of_steps=3000 \ #可以继续增加到30W
--train_split="train" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \ 
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--train_crop_size=513,513 \  #如果要增加batch_size，然而GPU带不动的话，可以修改train_crop_size 到321,321
--train_batch_size=4 \ #可以继续增加，如果GPU带得动的话
--dataset="soiling_dataset" \
--tf_initial_checkpoint='/home/user/models/research/deeplab/deeplabv3_cityscapes_train/model.ckpt' \ #deeplab给的checkpoint
--train_logdir='/home/user/models/research/deeplab/exp/soiling_train/train' \
--dataset_dir='~/Desktop/Soiling_dataset/tfrecord' #tfrecord，之前用build_voc2012_data.py打包的
```

Q: 如何设置train_crop_size的值?
A: output_stride * k + 1, where k is an integer. For example, we have 321x321，513x513 


## 2.4 训练结果可视化

```sh
python3 ~/models/research/deeplab/vis.py \ 
--logtostderr \
--vis_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--vis_crop_size=1080,1920\
--dataset="soiling_dataset" \
--colormap_type="pascal" \ 
--checkpoint_dir='/home/user/models/research/deeplab/exp/soiling_train/train'\
--vis_logdir='/home/user/models/research/deeplab/exp/soiling_train/vis' \
--dataset_dir='/home/user/dataset/Soiling_dataset/tfrecord'
```
* vis_split:设置为测试集 
* vis_crop_size:设置1080,1920为图片的大小 
* dataset:设置为我们在
* data_generator.py文件设置的数据集名称 
* dataset_dir:设置为创建的TFRecord 
* colormap_type:可视化标注的颜色可到目录deeplab/exp/soiling_train/vis下查看可视化结果

![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/000032_image.png)

![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/000032_prediction.png)

## 2.5 性能评估

```sh
python3 ~/models/research/deeplab/eval.py \ 
--logtostderr \
--eval_split="val" \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=1080,1920\
--dataset="soiling_dataset" \ 
--checkpoint_dir='/home/user/models/research/deeplab/exp/soiling_train/train'\
--eval_logdir='/home/user/models/research/deeplab/exp/soiling_train/eval' \
--dataset_dir='/home/user/Desktop/Soiling_dataset/tfrecord' \ 
--max_number_of_evaluations=1
```

* eval_split:设置为测试集 
* crop_size:同样设置为1080和1920 
* dataset:设置为soiling_dataset
* dataset_dir:设置为我们创建的数据集

查看mIoU值:
```sh
tensorboard --logdir /home/user/models/research/deeplab/exp/soiling_train/eval --host=127.0.0.1 #解决打开6006的问题
```

![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/miou_5;25_28w.png)


查看训练过程的loss:
```sh
tensorboard --logdir /home/user/models/research/deeplab/exp/soiling_train/train
```

![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/loss.png)
# 3 导出网络

```sh
python3 ~/models/research/deeplab/export_model.py \ 
--checkpoint_path='/home/user/models/research/deeplab/exp/soiling_train/model.ckpt-200967' \ #选择训练后的ckpt文件，后面的数字代码训练次数
--export_path='/home/user/Desktop/labeled_ubuntu/output_model/frozen_inference_graph_20w_5_30.pb' \
--model_variant="xception_65" \
--atrous_rates=6 \
--atrous_rates=12 \
--atrous_rates=18 \
--output_stride=16 \
--decoder_output_stride=4 \
--eval_crop_size=1080,1920\
--dataset="soiling_dataset" \ 
--inference_scales=1.0
```
 
* 导出后在那个文件下压缩一下
```
tar -zvcf pb2.tar.gz frozen_inference_graph_20w_5_30.pb
```

* 测试图片
``` python3 '/home/user/Desktop/labeled_ubuntu/dataset_train/test_picture.py'```

# 4 结果

![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/results.png)
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/results2.png)

# 5 计算遮挡率

用```opencv_python.py```可以使用python计算灰度图像中的蒙版区域（以像素为单位）
```sh
import cv2
import numpy as np

image = cv2.imread('/home/ogai/models/research/deeplab/exp/soiling_train/vis/vis/soiling1_10.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
total = 0

for c in cnts:
    x,y,w,h = cv2.boundingRect(c)
    mask = np.zeros(image.shape, dtype=np.uint8)
    cv2.fillPoly(mask, [c], [255,255,255])
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    pixels = cv2.countNonZero(mask)
    total += pixels
    cv2.putText(image, '{}'.format(pixels), (x ,y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

print('mask pixel',total)
total_pixel = 2073600
coverage = round((total / total_pixel),2)*100

cv2.putText(image, 'coverage of soiling part = '+'{}'.format(coverage) +'%' , (20,20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

print('coverage of soiling part ',coverage)

cv2.namedWindow("thresh",0)
cv2.resizeWindow("thresh", 1440, 900)
cv2.imshow("thresh",thresh)


#cv2.imshow('thresh', thresh)
#cv2.imshow('image', image)
#cv2.waitKey(0)

cv2.namedWindow("image",0)
cv2.resizeWindow("image", 1440, 900)
cv2.imshow("image",image)
cv2.waitKey(0)


```
测试图片
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/soiling1_10.png)

先将图片转换成灰度图片
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/mask_processed.png)

最后将统计到的像素除以总的像素即得到遮挡率
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/coverage_of_soiling_part.png)

结合展示
![Image text](https://github.com/Leozyc-waseda/SoilingDataset/blob/master/github_images/results_draft1.png)
