"""
First Step Conver BMP
Original ROI code --> only Decode BMP IMAGE
"""

import cv2
import glob
import re
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Notre-Dame
checkpoint_dict = {'NestedUVC_LocalAttention': ['279', '277'], 'NestedUVC_ReAttention': ['270', '275'],
                   'NestedUVC_MSA': ['253', '254'], 'NestedUVC_DualAttention_Parallel': ['264', '277'],
                   'NestedUVC_DualAttention_Parallel_Fourier_MSE': ['269', '262'],
                   'NestedUVC_DualAttention_Parallel_Fourier_MSE_2': ['276', '276'],
                   'NestedUVC_DualAttention_Parallel_Fourier_MSE_woSu': ['296', '293'],
                   'CycleGAN': ['192', '178']}

# checkpoint_dict = {'NestedUVC_LocalAttention': ['250', '265'], 'NestedUVC_ReAttention': ['270', '299'],
#                    'NestedUVC_MSA': ['284', '293'], 'NestedUVC_DualAttention_Parallel': ['274', '276'],
#                    'NestedUVC_DualAttention_Parallel_Fourier_MSE': ['259', '268'],
#                    'NestedUVC_DualAttention_Parallel_Fourier_MSE_woSu': ['250', '259']}
FOLD_TYPE = ['1-fold', '2-fold']
DB_NAMES = ['CycleGAN']

for DB_NAME in DB_NAMES:
    print(f'-------------------------NOW: {DB_NAME}-------------------------')
    checkpoint = checkpoint_dict[DB_NAME]
    for full_count in range(len(FOLD_TYPE)):
        if FOLD_TYPE[full_count] == '1-fold':
            DB = 'B'
        else:
            DB = 'A'

        # path = f'M:/2nd/backup/Experiment/GANs/Ablation/ND/NEW/{DB_NAME}/{FOLD_TYPE[full_count]}/test'
        # image_path = f'/{DB}/A2B/{checkpoint[full_count]}'
        # path = 'M:/2nd/dataset/ND'
        path = f'Z:/1st/Iris_dataset/nd_labeling_iris_data/Cycle_ROI/{FOLD_TYPE[full_count]}/'
        image_path = f'/{DB}/fake/'
        # path = f'M:/2nd/backup/Experiment/GANs/Compare/ND/{DB_NAME}/{FOLD_TYPE[full_count]}/test'
        # image_path = f'/{DB}/A2B/{checkpoint[full_count]}/'
        imgs = os.listdir(f'{path}{image_path}')

        iris_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris/fake'
        iris_upper_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_upper/fake'
        iris_lower_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_lower/fake'
        # iris_circle_path = f'M:/2nd/dataset/ND/ROI/Ablation/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_circle/fake'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)
        # os.makedirs(iris_circle_path, exist_ok=True)

        postion_info_files = os.listdir(f'ND/Position/old/{DB}')
        print(postion_info_files)

        for i in range(len(postion_info_files)):
            print(f'ND/Position/old/{DB}/position_try_{i + 1}.csv')
            info_list = pd.read_csv(f'ND/Position/old/{DB}/position_try_{i + 1}.csv')
            info_list = np.array(info_list)
            # print(len(info_list))
            # print(info_list)
            for info in info_list:
                try:
                    img_path = info[1]
                    # img_path = re.compile(f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Cycle_ROI/{DB}/live/').sub(
                    #     f'M:/2nd/backup/Experiment/GANs/Ablation/Warsaw/{DB_NAME}/{FOLD_TYPE[full_count]}/test/{DB}/A2B/{checkpoint[full_count]}/',
                    #     img_path)
                    # img_path = re.compile(f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Cycle_ROI/{DB}/live/').sub(
                    #     f'{path}{image_path}/',
                    #     img_path)
                    img_path = re.compile(f'Z:/2nd_paper/backup/NestedUVC_GAN_Attention/1-fold/test/{DB}/A2B/250/').sub(
                        f'{path}{image_path}',
                        img_path)
                    # img_path = re.compile('/backup/NestedUVC_GAN_Attention/1-fold/test/A/A2B/250/').sub('/dataset/ND/B_live/', img_path)
                    # print(img_path)
                    # img_path = re.compile('.bmp').sub('.png', img_path)
                    # img_path = re.compile('_A2B.bmp').sub('.png', img_path)
                    # img_path = re.compile('.png').sub('_A2B.png', img_path)
                    # print(img_path)
                    img_path = re.compile('.png').sub('_A2B.bmp', img_path)
                    positons = info[0]
                    positons = re.sub('[[\]]', '', positons)
                    positons = positons.split(' ')
                    # print(positons)

                    new_positions = []
                    for i in positons:
                        if i == '':
                            pass
                        else:
                            new_positions.append(i)

                    # imgpath = f'{path}/{image_path}/{imgName}'
                    img = cv2.imread(img_path)
                    # img = cv2.resize(img, dsize=(224, 224))
                    img = cv2.resize(img, dsize=(440, 280))
                    # img = cv2.resize(img, dsize=(580, 420))

                    x = int(float(new_positions[0]))
                    y = int(float(new_positions[1]))
                    r = int(float(new_positions[2]))

                    img2 = img.copy()

                    cv2.circle(img2, (x, y), r, (0, 255, 0), 2)

                    iris = img[y - r:y + r, x - r:x + r]
                    iris_up = img[y - 130:y - 20, 0:440]
                    iris_down = img[y + 20: y + 130, 0:440]

                    if (y + r) > 280:
                        iris = img[y - r:280, x - r:x + r]
                        iris_down = iris_up
                        iris_down = cv2.flip(iris_down, 0)

                    elif (y + 130) > 280:
                        iris_down = iris_up
                        iris_down = cv2.flip(iris_down, 0)

                    if (y - r) < 0:
                        iris = img[0:y + r, x - r:x + r]
                        iris_up = iris_down
                        iris_up = cv2.flip(iris_up, 0)

                    elif (y - 130) < 0:
                        iris_up = iris_down
                        iris_up = cv2.flip(iris_up, 0)

                    imgName = img_path.split('/')[-1]
                    imgName = re.compile(".bmp").sub('.png', imgName)
                    # cv2.imwrite(f'{iris_circle_path}/{imgName}',img2)
                    cv2.imwrite(f'{iris_path}/{imgName}', iris)
                    cv2.imwrite(f'{iris_upper_path}/{imgName}',iris_up)
                    cv2.imwrite(f'{iris_lower_path}/{imgName}',iris_down)

                except:
                    print("ERROR PATH !!!!!")
                    print(f'{img_path}')
                    pass

        print(f'CLEAR {FOLD_TYPE[full_count]} {DB} !!!!!!!!!')