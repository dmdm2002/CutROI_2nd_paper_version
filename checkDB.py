import os
import re

path = 'Z:/2nd_paper/dataset/ND/ROI/CycleGAN/A/iris'

fake_list = os.listdir(f'{path}/fake')
live_list = os.listdir(f'{path}/live')


for f_name in fake_list:
    cnt = 0
    f_name = re.compile('_A2B').sub('', f_name)
    for l_name in live_list:
        if f_name == l_name:
           cnt = cnt + 1

    if cnt == 0:
        print(f'not matching name : {f_name}')

    if cnt > 1:
        print(f'over matching name : {f_name}')