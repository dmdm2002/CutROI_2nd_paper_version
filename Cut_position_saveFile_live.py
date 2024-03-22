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

FOLD_TYPE = ['1-fold', '2-fold']
DB_NAMES = ['original']

for DB_NAME in DB_NAMES:
    for full_count in range(len(FOLD_TYPE)):
        if FOLD_TYPE[full_count] == '1-fold':
            DB = 'B'
        else:
            DB = 'A'

        path = f'M:/2nd/dataset/Warsaw/{DB_NAME}'
        image_path = f'/{DB}/'
        imgs = os.listdir(f'{path}{image_path}')

        iris_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}_resize/{FOLD_TYPE[full_count]}/{DB}/iris/live'
        iris_upper_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}_resize/{FOLD_TYPE[full_count]}/{DB}/iris_upper/live'
        iris_lower_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}_resize/{FOLD_TYPE[full_count]}/{DB}/iris_lower/live'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)
        # os.makedirs(iris_circle_path, exist_ok=True)

        postion_info_files = os.listdir(f'Warsaw/Position/old/{DB}')
        print(postion_info_files)

        for i in range(len(postion_info_files)):
            print(f'Warsaw/Position/old/{DB}/position_try_{i + 1}.csv')
            info_list = pd.read_csv(f'Warsaw/Position/old/{DB}/position_try_{i + 1}.csv')
            info_list = np.array(info_list)
            # print(len(info_list))
            # print(info_list)
            for info in info_list:
                try:
                    img_path = info[1]
                    folder = img_path.split('/')[-1].split('_')[0]
                    # folder = img_path.split('/')[-1].split('d')[0]
                    # img_path = re.compile(f'Z:/2nd_paper/backup/NestedUVC_GAN_Attention/1-fold/test/{DB}/A2B/250/').sub(
                    #     f'M:/2nd/dataset/ND/original/{DB}/{folder}/', img_path
                    # )
                    img_path = re.compile(f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/Cycle_ROI/{DB}/live/').sub(
                        f'M:/2nd/dataset/Warsaw/original/{DB}/{folder}/',
                        img_path)

                    img_path = re.compile('.bmp').sub('.png', img_path)
                    positons = info[0]
                    positons = re.sub('[[\]]', '', positons)
                    positons = positons.split(' ')

                    new_positions = []
                    for i in positons:
                        if i == '':
                            pass
                        else:
                            new_positions.append(i)

                    # imgpath = f'{path}/{image_path}/{imgName}'
                    img = cv2.imread(img_path)
                    img = cv2.resize(img, dsize=(224, 224))
                    # img = cv2.resize(img, dsize=(440, 280))
                    img = cv2.resize(img, dsize=(580, 420))

                    x = int(float(new_positions[0]))
                    y = int(float(new_positions[1]))
                    r = int(float(new_positions[2]))

                    img2 = img.copy()

                    cv2.circle(img2, (x, y), r, (0, 255, 0), 2)

                    iris = img[y - r:y + r, x - r:x + r]
                    iris_up = img[y - 130:y - 20, 0:580]
                    iris_down = img[y + 20: y + 130, 0:580]

                    if (y + r) > 420:
                        iris = img[y - r:420, x - r:x + r]
                        iris_down = iris_up
                        iris_down = cv2.flip(iris_down, 0)

                    elif (y + 130) > 420:
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
                    cv2.imwrite(f'{iris_upper_path}/{imgName}', iris_up)
                    cv2.imwrite(f'{iris_lower_path}/{imgName}', iris_down)

                except:
                    print("ERROR PATH !!!!!")
                    print(f'{img_path}')
                    pass

        print(f'CLEAR {FOLD_TYPE[full_count]} {DB} !!!!!!!!!')