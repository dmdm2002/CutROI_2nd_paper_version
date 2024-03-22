import cv2
import os


def get_image(path, output_path, label):
    imglist = os.listdir(f'{path}/{label}')
    for imgName in imglist:
        output_folder = f'{output_path}/{label}'
        os.makedirs(output_folder, exist_ok=True)

        imgPath = f'{path}/{label}/{imgName}'
        img = cv2.imread(imgPath)

        if label == 'fake':
            img = cv2.GaussianBlur(img, (3, 3), 1.0)

        output = f'{output_folder}/{imgName}'
        cv2.imwrite(output, img)

    return 0

basePath = f'Z:/2nd_paper/dataset/ND/ROI/CycleGAN'

# # input image dir path
db = ['A', 'B']

for d in db:
    if d == 'A':
        fold = '1-fold'
    else:
        fold = '1-fold'
    iris_B = f'{basePath}/{fold}/{d}'
    iris_B_folders = os.listdir(iris_B)
    print(f'input folder : {iris_B_folders}')

    iris_B_output = f'{basePath}/Attack/Gaussian/{d}/blur_33'
    os.makedirs(iris_B_output, exist_ok=True)

    os.makedirs(f'{iris_B_output}/iris', exist_ok=True)
    os.makedirs(f'{iris_B_output}/iris_upper', exist_ok=True)
    os.makedirs(f'{iris_B_output}/iris_lower', exist_ok=True)
    iris_B_output_folders = os.listdir(iris_B_output)
    print(f'output folder : {iris_B_output_folders}')

    classes = ['live', 'fake']
    for input_folder in iris_B_folders:
        for output_folder in iris_B_output_folders:
            if input_folder == output_folder:
                img_path = f'{iris_B}/{input_folder}'
                print(f'now img path : {img_path}')
                output_dir = f'{iris_B_output}/{input_folder}'
                for i in range(0, len(classes)):
                    get_image(img_path, output_dir, classes[i])

                    print(f'finish {input_folder} {classes[i]} !!')

    print(f'------clear {d}------')