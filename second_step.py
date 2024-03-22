import cv2
import glob
import re
import os
import matplotlib.pyplot as plt
import pandas as pd


IMG_SZ = [440, 280]

iris_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/PGGAN/2-fold/iris/B/fake'
iris_circle_path = f'Z:/2nd_paper/dataset/ND/ROI/Compare/PGGAN/2-fold/B/iris_circle/fake'

"""
third step Other image Cut RoI
"""
path = f'Z:/2nd_paper/backup/Compare/Other_GANs/PGGAN/2-fold/test'
image_path = f'/B'
imgs = os.listdir(f'{path}{image_path}')

finish_images = os.listdir(f'{iris_circle_path}')

temp_iris = f'Z:/2nd_paper/dataset/ND/ROI/Compare/PGGAN/2-fold/temp/B/iris/fake'
temp_iris_circle = f'Z:/2nd_paper/dataset/ND/ROI/Compare/PGGAN/2-fold/temp/B/iris_circle/fake'

os.makedirs(temp_iris, exist_ok=True)
os.makedirs(temp_iris_circle, exist_ok=True)

kernel = 7
minDist = 100
param1 = 70
param2 = 30
# minRadius = 20
# maxRadius = 80
minRadius = 110
maxRadius = 150


position_info = []
def cut_func():
    finish = 0
    for imgName in imgs:
        same_count = 0
        for i in finish_images:
            matching = re.compile('.bmp').sub('.png', i)
            # print(f'matching: {matching}')
            # print(f'Name: {imgName}')
            if matching == imgName:
                same_count += 1
        if same_count == 0:
            print(imgName)
            # root = 'D:/Research/datasets/nd_labeling_iris_data/'
            # image_path = 'RaSGAN/BMP/1-fold/B/fake'

            img = cv2.imread(f'{path}{image_path}/{imgName}', 0)
            img = cv2.resize(img, dsize=(IMG_SZ[0], IMG_SZ[1]))

            img2 = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            mBlurimg = cv2.medianBlur(img, 3)
            # mBlurimg = cv2.medianBlur(mBlurimg, 3)
            tmp = mBlurimg.copy()
                    # print(f'{path}{train_live_path}/{imgName}')
            # for i in range(0, len(tmp)):
            #     for j in range(0, len(tmp[i])):
            #         # A Live : tmp[i][j]> 93
            #         if tmp[i][j] > 93:
            #             tmp[i][j] = 225
            tmp = cv2.equalizeHist(tmp)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3,3))
            # tmp = clahe.apply(tmp)
            # # mBlurimg = cv2.medianBlur(tmp, 7)
            ret,tmp = cv2.threshold(tmp, 70, 255,cv2.THRESH_BINARY)
            # tmp = cv2.Canny(tmp, 80, 255)
            #         plt.imshow(canny)
            #         plt.show()
            #         mBlurimg = cv2.medianBlur(mBlurimg, 7)

            circles = cv2.HoughCircles(tmp, cv2.HOUGH_GRADIENT, 1, minDist, param1=70, param2=10,
                                       minRadius=90, maxRadius=150)
            if circles is None:
                circles = cv2.HoughCircles(tmp, cv2.HOUGH_GRADIENT, 1, minDist, param1=100, param2=10,
                                           minRadius=150, maxRadius=180)
            #         print(circles)

            print(circles)
            position_info.append([circles[0][0], f'{path}{image_path}/{imgName}'])

            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            r = int(circles[0][0][2])
            half_r = r // 2

            img2 = cv2.circle(img2, (x, y), r, (0, 255, 0), 2)

            iris = img[y - r:y + r, x - r:x + r]
            iris_up = img[y - 130:y - 20, 0:IMG_SZ[0]]
            iris_down = img[y + 20: y + 130, 0:IMG_SZ[0]]

            if (y + r) > IMG_SZ[1]:
                iris = img[y - r:IMG_SZ[1], x - r:x + r]
                iris_down = iris_up
                iris_down = cv2.flip(iris_down, 0)

            elif (y + 130) > IMG_SZ[1]:
                iris_down = iris_up
                iris_down = cv2.flip(iris_down, 0)

            if (y - r) < 0:
                iris = img[0:y + r, x - r:x + r]
                iris_up = iris_down
                iris_up = cv2.flip(iris_up, 0)

            elif (y - 130) < 0:
                iris_up = iris_down
                iris_up = cv2.flip(iris_up, 0)

            imgName = re.compile(".bmp").sub('.png', imgName)
            cv2.imwrite(f'{temp_iris_circle}/{imgName}', img2)
            cv2.imwrite(f'{temp_iris}/{imgName}', iris)


if len(finish_images) == len(imgs):
    print('finish cut')
else:
    cut_num = 3
    print('start Cut ROI')
    try:
        cut_func()
        print('End Cut ROI')
        df = pd.DataFrame(position_info)
        os.makedirs('PGGAN/ND/Position/B/2-fold', exist_ok=True)
        df.to_csv(f'PGGAN/ND/Position/B/2-fold/position_try_{cut_num}.csv', index=False)
        print(df)
    except Exception as e:
        print(f'Error : {e}')
        print('Error End Cut ROI')
        df = pd.DataFrame(position_info)
        os.makedirs('PGGAN/ND/Position/B/2-fold', exist_ok=True)
        df.to_csv(f'PGGAN/ND/Position/B/2-fold/position_try_{cut_num}.csv', index=False)
        print(df)