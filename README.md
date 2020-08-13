# SoilingDataset
Due to the postponement of the release of the woodscape dataset, I plan to create a soiling dataset myself for research.

#  1 数据集
## 1.1 取得数据
  - 将摄像头装到toyota coms车的后方位置，四个摄像头并排，一起取得图像。其中3个脏的摄像头，一个干净的摄像头（干净的镜头是为了方便以后desoiling，或者用CycleGAN来生成假数据的情况下，需要一个完全干净的图片） 
  - 脏污是由我自己手动弄脏并且涂抹在摄像头的镜头上，用泥土+水+牙刷去制造Soiling的数据 
  - 录制的软件都是由MacBook Pro(2020)完成，其中使用了两个type-c hub防止供电不足导致不能录影，录影软件使用的是Camera Record-Camera Recorder 付费在AppStore里面下载，非常小巧以及稳定，同时录制4个视频画面没有卡顿
  -  录影规格：30FPS，分辨率是1920，1080
  -  分三天不同的天气和时间点，驾驶toyota coms，时速大约30km/h; coms饶学校附近一圈，起点：北九工大车库，终点：北九工大车库；每段时间时间约为6分钟左右（绕一圈的时间），每天两次，每次4个视频，最后得到32个视频，每个视频6分钟
 
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
labelme我是在[anaconda][df3]上面运行的，添加了一个虚拟环境（做deep learning的人最方便就是用anaconda，下载各种包，备份工作环境都很方便）
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
  - 标注异常费力，几乎每张图片都需要花费我解决1个小时去标注，细分透明和不透明的部分，最终得到了184张图片
  - 其实有大牛已经在研究这个领域，并在去年的ICCV上发表了paper介绍了自己的数据集，只是由于新冠病毒的影响，以及欧洲对数据的新规定（公共数据集必须给人脸和车牌马赛克，不然不允许发布），所以推迟了一年，虽然目前为止2020年8月13日都还没有发布。我也问过作者他说他将会在今年Q3，Q4的时候发布，但是现在看来情势也不容乐观。最迟推断大概明天就会有soiling相关的dataset，大约接近5000张的数据，那么就不用辛苦的做数据集了
  - 由于数据集的稀少和环境相似，其实数据集多样性和鲁棒性都不是很好，测试集和训练集也是高度相似，这点如何解决我还没有任何头绪
  - 在Deep Learning中，数据集起码要有3000~5000张，才会有不错的效果，所以语义分割的数据集是非常昂贵和难以获取的。我也想去淘宝上花钱雇人帮我标注，只是15RMB一张的图片让我望而却步，其他研究者都是委托专门的数据公司进行标注，所以我的小研究就靠184张图片来搞吧
  - 未来可能会用CycleGAN之类的技术去建立假数据，扩充我的数据集，增加多样性和鲁棒性

 

 [df1]: <https://github.com/wkentaro/labelme>
 [df2]: <https://github.com/Hitachi-Automotive-And-Industry-Lab/semantic-segmentation-editor>
 [df3]: <https://www.anaconda.com/products/individual>
