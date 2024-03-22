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

FOLD_TYPE = ['1-fold', '1-fold']
# checkpoint = ['192', '178']
DB_NAME = 'ACL-GAN'
DBs = ['A', 'B']

for full_count in range(len(FOLD_TYPE)):
    for DB in DBs:
        path = f'Z:/2nd_paper/backup/Compare/Other_GANs/{DB_NAME}/{FOLD_TYPE[full_count]}/test'
        image_path = f'/{DB}/fake'
        # path = 'Z:/2nd_paper/dataset/ND'
        imgs = os.listdir(f'{path}{image_path}')
        # image_path = '/B_live'

        f = open(f'../ND/Position/old/first_step_position/{DB}.txt', "r", encoding='utf-8')

        position_info = []
        for line in f.readlines():
            line = re.compile('\n').sub(' ', line)
            line = re.split(', ', line)

            info = [line[0], line[1], line[2], line[3]]
            position_info.append(info)

        print(position_info[0])

        iris_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris/fake'
        iris_upper_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_upper/fake'
        iris_lower_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_lower/fake'
        iris_circle_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_circle/fake'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)
        os.makedirs(iris_circle_path, exist_ok=True)

        for imgName in imgs:
            #     print(f'{train_live_path}/{imgName}')
            for position in position_info:
                position[3] = re.compile(' ').sub('', position[3])
                position[3] = re.compile('_A2B.bmp').sub('.png', position[3])
                position[3] = re.compile('_output0.png').sub('.png', position[3])

                temp = re.compile('_output0.png').sub('.png', imgName)

                if f'Cycle_ROI/{DB}/fake/{temp}' == position[3]:
                    # print(f'{path}/{image_path}/{imgName}')
                    imgpath = f'{path}/{image_path}/{imgName}'

                    # print(imgpath)

                    img = cv2.imread(imgpath)
                    # img = cv2.resize(img, dsize=(224, 224))
                    img = cv2.resize(img, dsize=(224, 224))
                    img = cv2.resize(img, dsize=(440, 280))

                    x = int(position[0])
                    y = int(position[1])
                    r = int(position[2])

                    # img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    # img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    img2 = img.copy()

                    cv2.circle(img2, (x, y), r, (0, 255, 0), 2)

                    iris = img[y - r:y + r, x - r:x + r]
                    iris_up = img[y - 130:y - 20, 0:440]
                    iris_down = img[y + 20: y + 130, 0:440]

                    # 280 -> 224
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

                    imgName = re.compile("_A2B.bmp").sub('.png', imgName)
                    cv2.imwrite(f'{iris_circle_path}/{temp}',img2)
                    cv2.imwrite(f'{iris_path}/{temp}', iris)
                    cv2.imwrite(f'{iris_upper_path}/{temp}',iris_up)
                    cv2.imwrite(f'{iris_lower_path}/{temp}',iris_down)

                    # 이미지 만나서 처리하면 해당 루프 종료하고 위로 올라가기
                    continue

                else:
                    pass

        print(f'CLEAR {FOLD_TYPE[full_count]} {DB} !!!!!!!!!')