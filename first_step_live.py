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
        # path = 'Z:/2nd_paper/dataset/ND'
        imgs = glob.glob(f'{path}{image_path}*/*')
        # image_path = '/B_live'

        f = open(f'Warsaw/Position/old/first_step_position/{DB}.txt', "r", encoding='utf-8')

        position_info = []
        for line in f.readlines():
            line = re.compile('\n').sub(' ', line)
            line = re.split(', ', line)

            info = [line[0], line[1], line[2], line[3]]
            position_info.append(info)

        print(position_info[0])

        iris_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}_resize/{FOLD_TYPE[full_count]}/{DB}/iris/live'
        iris_upper_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}_resize/{FOLD_TYPE[full_count]}/{DB}/iris_upper/live'
        iris_lower_path = f'M:/2nd/dataset/Warsaw/ROI/Ablation/{DB_NAME}_resize/{FOLD_TYPE[full_count]}/{DB}/iris_lower/live'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)

        for imgName in imgs:
            split_Name = imgName.split('\\')
            folder = split_Name[-2]
            imgName = split_Name[-1]

            for position in position_info:
                position[3] = re.compile(' ').sub('', position[3])
                position[3] = re.compile('_A2B.bmp').sub('.png', position[3])
                position[3] = re.compile('.bmp').sub('.png', position[3])
                # position[3] = re.compile('live').sub('fake', position[3])
                # position[3] = re.compile('_A2B').sub('', position[3])

                if f'Cycle_ROI/{DB}/live/{imgName}' == position[3]:
                    imgpath = f'{path}/{image_path}/{folder}/{imgName}'

                    print(imgpath)
                    img = cv2.imread(imgpath)
                    img = cv2.resize(img, dsize=(224, 224))
                    # img = cv2.resize(img, dsize=(440, 280))
                    img = cv2.resize(img, dsize=(580, 420))

                    x = int(position[0])
                    y = int(position[1])
                    r = int(position[2])

                    img2 = img.copy()

                    cv2.circle(img2, (x, y), r, (0, 255, 0), 2)

                    iris = img[y - r:y + r, x - r:x + r]
                    iris_up = img[y - 130:y - 20, 0:580]
                    iris_down = img[y + 20: y + 130, 0:580]
                    #
                    # # 280 -> 224
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

                    try:
                        imgName = re.compile("_A2B.bmp").sub('.png', imgName)
                        # cv2.imwrite(f'{iris_circle_path}/{imgName}',img2)
                        cv2.imwrite(f'{iris_path}/{imgName}', iris)
                        cv2.imwrite(f'{iris_upper_path}/{imgName}', iris_up)
                        cv2.imwrite(f'{iris_lower_path}/{imgName}', iris_down)

                        # 이미지 만나서 처리하면 해당 루프 종료하고 위로 올라가기
                        continue
                    except:
                        print(img)
                        print(f'{x} | {y} | {r}')
                        print(f'{y - r}:{y + r}, {x - r}:{x + r}')
                        print(f'{imgName}')
                        cv2.imwrite(f'{iris_path}/{imgName}', img2)
                        print(iris)
                else:
                    pass

        print(f'CLEAR {FOLD_TYPE[full_count]} {DB} !!!!!!!!!')