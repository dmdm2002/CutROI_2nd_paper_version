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

FOLD_TYPE = ['1-fold', '1-fold']
# checkpoint = ['192', '178']
DB_NAME = 'ACL-GAN'
DBs = ['A', 'B']


for full_count in range(len(FOLD_TYPE)):
    for DB in DBs:
        path = f'Z:/2nd_paper/backup/Compare/Other_GANs/{DB_NAME}/{FOLD_TYPE[full_count]}/test'
        image_path = f'/{DB}/fake'
        # path = 'Z:/2nd_paper/dataset/ND'
        # image_path = '/B_live'
        imgs = os.listdir(f'{path}{image_path}')

        iris_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris/fake'
        iris_upper_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_upper/fake'
        iris_lower_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_lower/fake'
        iris_circle_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_circle/fake'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)
        os.makedirs(iris_circle_path, exist_ok=True)

        postion_info_files = os.listdir(f'../ND/Position/old/{DB}')
        print(postion_info_files)

        for i in range(len(postion_info_files)):
            print(f'../ND/Position/old/{DB}/position_try_{i+1}.csv')
            info_list = pd.read_csv(f'../ND/Position/old/{DB}/position_try_{i + 1}.csv')
            info_list = np.array(info_list)
            # print(len(info_list))
            # print(info_list)
            for info in info_list:
                try:
                    img_path = info[1]
                    img_path = re.compile(
                        f'/backup/NestedUVC_GAN_Attention/1-fold/test/{DB}/A2B/250/'
                    ).sub(
                        f'/backup/Compare/Other_GANs/{DB_NAME}/{FOLD_TYPE[full_count]}/test/{DB}/fake/', img_path
                    )
                    # img_path = re.compile('/backup/NestedUVC_GAN_Attention/1-fold/test/A/A2B/250/').sub('/dataset/ND/B_live/', img_path)
                    # print(img_path)
                    img_path = re.compile('.bmp').sub('.png', img_path)
                    img_path = re.compile('.png').sub('_output0.png',img_path)
                    # print(img_path)
                    # img_path = re.compile('.png').sub('_A2B.bmp', img_path)
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
                    img = cv2.resize(img, dsize=(224, 224))
                    # img = cv2.GaussianBlur(img, (3, 3), 1.0)
                    img = cv2.resize(img, dsize=(440, 280))

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
                    imgName = re.compile("_A2B.bmp").sub('.png', imgName)
                    imgName = re.compile('_output0.png').sub('.png', imgName)
                    cv2.imwrite(f'{iris_circle_path}/{imgName}',img2)
                    cv2.imwrite(f'{iris_path}/{imgName}', iris)
                    cv2.imwrite(f'{iris_upper_path}/{imgName}',iris_up)
                    cv2.imwrite(f'{iris_lower_path}/{imgName}',iris_down)

                except:
                    print("ERROR PATH !!!!!")
                    pass

        print(f'CLEAR {FOLD_TYPE[full_count]} {DB} !!!!!!!!!')