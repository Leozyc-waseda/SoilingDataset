import os,shutil
from PIL import Image
 


train_path = '/home/ogai/Desktop/Soiling_dataset/train/'
filelist_train = sorted(os.listdir(train_path))

val_path = '/home/ogai/Desktop/Soiling_dataset/val'
filelist_val = sorted(os.listdir(val_path))

test_path = '/home/ogai/Desktop/Soiling_dataset/test'
filelist_test = sorted(os.listdir(test_path))
index_path = '/home/ogai/Desktop/Soiling_dataset/index'

VOC_file_dir = index_path


VOC_train_file = open(os.path.join(VOC_file_dir, "train.txt"), 'w')
VOC_val_file = open(os.path.join(VOC_file_dir, "val.txt"), 'w')
VOC_test_file = open(os.path.join(VOC_file_dir, "test.txt"), 'w')
VOC_train_file.close()
VOC_val_file.close()
VOC_test_file.close()

VOC_train_file = open(os.path.join(VOC_file_dir, "train.txt"), 'a')
VOC_val_file = open(os.path.join(VOC_file_dir, "val.txt"), 'a')
VOC_test_file = open(os.path.join(VOC_file_dir, "test.txt"), 'a')

for eachfile in filelist_train:
    (temp_name,temp_extention) = os.path.splitext(eachfile)
    img_name = temp_name
    VOC_train_file.write(img_name + '\n')

for eachfile in filelist_val:
    (temp_name, temp_extention) = os.path.splitext(eachfile)
    img_name = temp_name
    VOC_val_file.write(img_name + '\n')

for eachfile in filelist_test:
    (temp_name, temp_extention) = os.path.splitext(eachfile)
    img_name = temp_name
    VOC_test_file.write(img_name + '\n')

VOC_train_file.close()
VOC_val_file.close()
VOC_test_file.close()



