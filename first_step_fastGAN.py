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
DB_NAME = 'FastGAN'
DBs = ['A', 'B']
# IMG_SZ = [580, 420]
IMG_SZ = [440, 280]


for full_count in range(len(FOLD_TYPE)):
    if FOLD_TYPE[full_count] == '1-fold':
        DB = 'B'
    else:
        DB = 'A'

    path = f'Z:1st/colab_backup/{DB_NAME}/ND/{FOLD_TYPE[full_count]}'
    image_path = f'/{DB}/'
    # path = f'Z:1st/colab_backup/{DB_NAME}/ND/BMP/{FOLD_TYPE[full_count]}'
    # image_path = f'/{DB}/fake'
    # path = 'Z:/2nd_paper/dataset/ND'
    imgs = os.listdir(f'{path}{image_path}')
    # image_path = '/B_live'

    f = open(f'Z:/1st/colab_backup/{DB_NAME}/OLD/ND/BMP/{FOLD_TYPE[full_count]}/{DB}_myfile.txt', "r", encoding='utf-8')

    position_info = []
    for line in f.readlines():
        line = re.compile('\n').sub(' ', line)
        line = re.split(', ', line)

        info = [line[0], line[1], line[2], line[3]]
        position_info.append(info)

    print(position_info[0])

    iris_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris/fake'
    iris_upper_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris_upper/fake'
    iris_lower_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris_lower/fake'
    iris_circle_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}_new/{FOLD_TYPE[full_count]}/{DB}/iris_circle/fake'

    os.makedirs(iris_path, exist_ok=True)
    os.makedirs(iris_upper_path, exist_ok=True)
    os.makedirs(iris_lower_path, exist_ok=True)
    os.makedirs(iris_circle_path, exist_ok=True)

    for imgName in imgs:
        #     print(f'{train_live_path}/{imgName}')
        for position in position_info:
            position[3] = re.compile(' ').sub('', position[3])
            position[3] = re.compile('.bmp').sub('.png', position[3])
            # print(position[3])
            # position[3] = re.compile('BMP').sub('', position[3])
            # # print(position[3])
            # position[3] = re.compile('A/fake').sub('test/A', position[3])
            # position[3] = re.compile('B/fake').sub('test/B', position[3])
            # position[3] = re.compile('//').sub('/', position[3])

            # print(position[3])
            # if FOLD_TYPE[full_count] == "1-fold":
            #     imgName = imgName.split('.')[0]
            #     imgName = (int(imgName) + 2277)

            if f'Z:/1st/colab_backup/FastGAN/ND/BMP/{FOLD_TYPE[full_count]}/{DB}/fake/{imgName}' == position[3]:
                # print(f'{path}/{image_path}/{imgName}')
                # print('check')
                print(f'Z:/1st/colab_backup/FastGAN/ND/BMP/{FOLD_TYPE[full_count]}/{DB}/fake/{imgName} | {position[3]}')
                imgpath = f'{path}{image_path}/{imgName}'

                # print(imgpath)

                img = cv2.imread(imgpath)
                # img = cv2.resize(img, dsize=(224, 224))
                # img = cv2.resize(img, dsize=(512, 512))
                # img = cv2.resize(img, dsize=(224, 224))
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
                # cv2.imwrite(f'{iris_circle_path}/{imgName}',img2)
                cv2.imwrite(f'{iris_path}/{imgName}', iris)
                cv2.imwrite(f'{iris_upper_path}/{imgName}',iris_up)
                cv2.imwrite(f'{iris_lower_path}/{imgName}',iris_down)

                # 이미지 만나서 처리하면 해당 루프 종료하고 위로 올라가기
                continue

            else:
                pass
                # print(f'real: Z:/2nd_paper/backup/Compare/Other_GANs/PGGAN/{FOLD_TYPE[full_count]}/test/{DB}/{imgName}')
                # print(position[3])

    print(f'CLEAR {FOLD_TYPE[full_count]} {DB} !!!!!!!!!')