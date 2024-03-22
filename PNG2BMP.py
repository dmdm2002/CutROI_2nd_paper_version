import cv2
import os
import re
import glob
import multiprocessing
import pandas as pd

cls = ['A', 'B']
PATH = 'Z:/2nd_paper/backup/Compare/Other_GANs/PGGAN/'
OUTPUT_ROOT = 'Z:/2nd_paper/backup/Compare/Other_GANs/PGGAN/BMP/'
FOLD = ['1-fold', '1-fold']


def convert_img(cls):
    for fold in FOLD:
        for cl in cls:
            imgs = glob.glob(f'{PATH}/{fold}/test/{cl}/*')
            print(f'{PATH}/{fold}/{cl}/*')
            print(imgs)
            OUTPUT_PATH = f'{OUTPUT_ROOT}/{fold}/{cl}/fake'

            c_plus_path = []

            """ BMP로 변환된 이미지가 저장될 OUTPUT FOLDER 생성"""
            os.makedirs(OUTPUT_PATH, exist_ok=True)

            for img in imgs:
                temp = cv2.imread(img)
                temp = cv2.resize(temp, dsize=(440, 280))

                name = img.split('\\')[-1]
                name = re.compile('.png').sub('.bmp', name)

                cv2.imwrite(f'{OUTPUT_PATH}/{name}', temp)
                c_plus_path.append(f'{OUTPUT_PATH}/{name}')

            df = pd.DataFrame(c_plus_path)
            df.to_csv(f'{OUTPUT_ROOT}/{fold}/{cl}.txt', index=False)


if __name__ == '__main__':
    pool = multiprocessing.Pool(processes=2)
    pool.map(convert_img, cls)
    pool.close()
    pool.join()