import cv2
from distutils.dir_util import copy_tree
import os

checkpoint_dict = {'NestedUVC_LocalAttention': ['250', '265'], 'NestedUVC_ReAttention': ['270', '299'],
                   'NestedUVC_MSA': ['284', '293'], 'NestedUVC_DualAttention_Parallel': ['274', '276'],
                   'NestedUVC_DualAttention_Parallel_Fourier_MSE': ['280', '262'],
                   'NestedUVC_DualAttention_Parallel_Fourier_MSE_woSu': ['250', '259']}
# checkpoint_dict = {'NestedUVC_DualAttention_Parallel_Fourier_MSE': ['252', '268']}
FOLD_TYPE = ['1-fold']
DB_NAMES = ['NestedUVC_DualAttention_Parallel_Fourier_MSE']

for DB_NAME in DB_NAMES:
    for fold in FOLD_TYPE:
        if fold == '1-fold':
            folder = 'B'
        else:
            folder = 'A'
        print(f'--------------------------------NOW FOLD : [{fold}]--------------------------------')
        iris_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/original/{fold}/{folder}/iris/live'
        iris_upper_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/original/{fold}/{folder}/iris_upper/live'
        iris_lower_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/original/{fold}/{folder}/iris_lower/live'
        #
        # iris_path = f'M:/2nd/dataset/ND/Warsaw/Ablation/NestUVC_NoSupervision/{fold}/{folder}/iris/live'
        # iris_upper_path = f'M:/2nd/dataset/ND/Warsaw/Ablation/NestUVC_NoSupervision/{fold}/{folder}/iris_upper/live'
        # iris_lower_path = f'M:/2nd/dataset/ND/Warsaw/Ablation/NestUVC_NoSupervision/{fold}/{folder}/iris_lower/live'
        #
        copy_iris_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}/{fold}/{folder}/iris/live'
        copy_iris_upper_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}/{fold}/{folder}/iris_upper/live'
        copy_iris_lower_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}/{fold}/{folder}/iris_lower/live'
        # #
        copy_tree(iris_path, copy_iris_path)
        print(f'clear {folder} iris')
        copy_tree(iris_upper_path, copy_iris_upper_path)
        print(f'clear {folder} copy_iris_upper_path')
        copy_tree(iris_lower_path, copy_iris_lower_path)
        print(f'clear {folder} copy_iris_lower_path')
