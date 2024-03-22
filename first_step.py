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

DBs = ['A', 'B']
for DB_NAME in DB_NAMES:
    print(f'-------------------------NOW: {DB_NAME}-------------------------')
    checkpoint = checkpoint_dict[DB_NAME]
    for full_count in range(len(FOLD_TYPE)):
        # for DB in DBs:
        if FOLD_TYPE[full_count] == '1-fold':
            DB = 'B'
        else:
            DB = 'A'
        # Z:\1st\Iris_dataset\nd_labeling_iris_data\CycleGAN\1-fold\A\fake
        # path = f'Z:/1st/Iris_dataset/Warsaw_labeling_iris_data/innerclass/{DB_NAME}/{FOLD_TYPE[full_count]}'
        path = f'Z:/1st/Iris_dataset/nd_labeling_iris_data/Cycle_ROI/{FOLD_TYPE[full_count]}/'
        image_path = f'/{DB}/fake'
        # image_path = f'/{DB}/A2B/{checkpoint[full_count]}'
        # path = 'M:/2nd/dataset/ND'
        imgs = os.listdir(f'{path}{image_path}')
        # image_path = '/B_live'

        f = open(f'ND/Position/old/first_step_position/{DB}.txt', "r", encoding='utf-8')

        position_info = []
        for line in f.readlines():
            line = re.compile('\n').sub(' ', line)
            line = re.split(', ', line)

            info = [line[0], line[1], line[2], line[3]]
            position_info.append(info)

        # print(position_info[0])

        iris_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris/fake'
        iris_upper_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_upper/fake'
        iris_lower_path = f'M:/2nd/dataset/ND/ROI/Compare/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_lower/fake'
        # iris_circle_path = f'M:/2nd/dataset/ND/ROI/Ablation/{DB_NAME}/{FOLD_TYPE[full_count]}/{DB}/iris_circle/fake'

        os.makedirs(iris_path, exist_ok=True)
        os.makedirs(iris_upper_path, exist_ok=True)
        os.makedirs(iris_lower_path, exist_ok=True)
        # os.makedirs(iris_circle_path, exist_ok=True)

        for imgName in imgs:
            # temp = re.compile('_A2B').sub('', imgName)
            #     print(f'{train_live_path}/{imgName}')
            for position in position_info:
                position[3] = re.compile(' ').sub('', position[3])
                # position[3] = re.compile('_A2B').sub('', position[3])
                # position[3] = re.compile('.bmp').sub('_A2B.bmp', position[3])
                position[3] = re.compile('live').sub('fake', position[3])

                # print(f'{position[3]} | Cycle_ROI/{DB}/fake/{imgName}')
                # M:\2nd\backup\Experiment\GANs\Compare\Warsaw\CycleGAN\1-fold\B\fake
                # print(position[3])
                if f'Cycle_ROI/{DB}/fake/{imgName}' == position[3]:
                    print(f'{position[3]} | Cycle_ROI/{DB}/fake/{imgName}')
                    # print(f'{path}/{image_path}/{imgName}')
                    imgpath = f'{path}/{image_path}/{imgName}'

                    # print(imgpath)

                    img = cv2.imread(imgpath)
                    # img = cv2.resize(img, dsize=(224, 224))
                    # img = cv2.resize(img, dsize=(224, 224))
                    img = cv2.resize(img, dsize=(440, 280))
                    # img = cv2.resize(img, dsize=(580, 420))

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
                    #
                    # # 280 -> 224
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

                    try:
                        imgName = re.compile(".bmp").sub('.png', imgName)
                        # cv2.imwrite(f'{iris_circle_path}/{imgName}',img2)
                        cv2.imwrite(f'{iris_path}/{imgName}', iris)
                        cv2.imwrite(f'{iris_upper_path}/{imgName}',iris_up)
                        cv2.imwrite(f'{iris_lower_path}/{imgName}',iris_down)

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