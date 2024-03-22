import cv2
from distutils.dir_util import copy_tree
import os

FOLD_TYPE = ['1-fold', '1-fold']
DB_NAME = 'ACL-GAN'
DBs = ['A', 'B']


for fold in FOLD_TYPE:
    print(f'--------------------------------NOW FOLD : [{fold}]--------------------------------')

    iris_path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/A/iris/live'
    iris_upper_path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/A/iris_upper/live'
    iris_lower_path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/A/iris_lower/live'
    #
    copy_iris_path = f'Z:/1st/colab_backup/FastGAN/Warsaw/ROI/{fold}/A/iris/live'
    copy_iris_upper_path = f'Z:/1st/colab_backup/FastGAN/Warsaw/ROI/{fold}/A/iris_upper/live'
    copy_iris_lower_path = f'Z:/1st/colab_backup/FastGAN/Warsaw/ROI/{fold}/A/iris_lower/live'
    #
    copy_tree(iris_path, copy_iris_path)
    print('clear A iris')
    copy_tree(iris_upper_path, copy_iris_upper_path)
    print('clear A iris_upper')
    copy_tree(iris_lower_path, copy_iris_lower_path)
    print('clear A iris_lower')

    iris_path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/B/iris/live'
    iris_upper_path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/B/iris_upper/live'
    iris_lower_path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Proposed/1-fold/B/iris_lower/live'

    copy_iris_path = f'Z:/1st/colab_backup/FastGAN/Warsaw/ROI/{fold}/B/iris/live'
    copy_iris_upper_path = f'Z:/1st/colab_backup/FastGAN/Warsaw/ROI/{fold}/B/iris_upper/live'
    copy_iris_lower_path = f'Z:/1st/colab_backup/FastGAN/Warsaw/ROI/{fold}/B/iris_lower/live'
    #
    copy_tree(iris_path, copy_iris_path)
    print('clear B iris')
    copy_tree(iris_upper_path, copy_iris_upper_path)
    print('clear B iris_upper')
    copy_tree(iris_lower_path, copy_iris_lower_path)
    print('clear B iris_lower')