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
DB_NAMES = ['FastGAN']

for DB_NAME in DB_NAMES:
    print(f'-------------------------NOW: {DB_NAME}-------------------------')
    for full_count in range(len(FOLD_TYPE)):
        if FOLD_TYPE[full_count] == '1-fold':
            DB = 'B'
        else:
            DB = 'A'

        path = f'Z:1st/colab_backup/{DB_NAME}/ND/{FOLD_TYPE[full_count]}'
        image_path = f'/{DB}/'
        imgs = os.listdir(f'{path}{image_path}')

        iris_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris/fake'
        iris_upper_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris_upper/fake'
        iris_lower_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris_lower/fake'
        iris_circle_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris_circle/fake'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)
        # os.makedirs(iris_circle_path, exist_ok=True)

        postion_info_files = os.listdir(f'FastGAN/ND/Position/old/{DB}/{FOLD_TYPE[full_count]}')
        print(postion_info_files)

        for i in range(len(postion_info_files)):
            print(f'FastGAN/ND/Position/old/{DB}/{FOLD_TYPE[full_count]}/position_try_{i + 1}.csv')
            info_list = pd.read_csv(f'FastGAN/ND/Position/old/{DB}/{FOLD_TYPE[full_count]}/position_try_{i + 1}.csv')
            info_list = np.array(info_list)
            # print(len(info_list))
            # print(info_list)
            for info in info_list:
                try:
                    img_path = info[1]
                    img_path = re.compile('.bmp').sub('.png', img_path)
                    img_path = re.compile(f'Z:/1st/colab_backup/FastGAN/{FOLD_TYPE[full_count]}/{DB}/').sub(f'{path}{image_path}/', img_path)
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
                    # img = cv2.resize(img, dsize=(512, 512))
                    img = cv2.resize(img, dsize=(224, 224))
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