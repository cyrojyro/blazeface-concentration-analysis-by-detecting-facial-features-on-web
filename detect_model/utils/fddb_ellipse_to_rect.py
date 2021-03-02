# modified OSS Code
# https://github.com/hualitlc/MTCNN-on-FDDB-Dataset/blob/master/convertEllipseToRectangle.py

from PIL import Image
import numpy as np
from math import *
import os

DATASET_DIR = '/home/cyrojyro/hddrive/WIDER_train/images/'
ELIP_GT_DIR = '/home/cyrojyro/hddrive/wider_face_split/FDDB-folds/'
OUTPUT_PATH = './fddb_gt.txt'


def filterCoordinate(c, m):
    if c < 0:
        return 0
    elif c > m:
        return m
    else:
        return c


def convertEllipseToRect(ellipseFilename, output_file):

    with open(ellipseFilename) as f:
        lines = [line.rstrip('\n') for line in f]

    i = 0
    while i < len(lines):
        img_file = DATASET_DIR + lines[i] + '.jpg'
        img = Image.open(img_file)
        w = img.size[0]
        h = img.size[1]
        num_faces = int(lines[i+1])

        output_file.write(lines[i] + '.jpg' + '\n')
        output_file.write(str(num_faces) + '\n')

        for j in range(num_faces):
            ellipse = lines[i+2+j].split()[0:5]
            a = float(ellipse[0])
            b = float(ellipse[1])
            angle = float(ellipse[2])
            centre_x = float(ellipse[3])
            centre_y = float(ellipse[4])

            tan_t = -(b/a)*tan(angle)
            t = atan(tan_t)
            x1 = centre_x + (a*cos(t)*cos(angle) - b*sin(t)*sin(angle))
            x2 = centre_x + (a*cos(t+pi)*cos(angle) - b*sin(t+pi)*sin(angle))
            x_max = filterCoordinate(max(x1, x2), w)
            x_min = filterCoordinate(min(x1, x2), w)

            if tan(angle) != 0:
                tan_t = (b/a)*(1/tan(angle))
            else:
                tan_t = (b/a)*(1/(tan(angle)+0.0001))

            t = atan(tan_t)
            y1 = centre_y + (b*sin(t)*cos(angle) + a*cos(t)*sin(angle))
            y2 = centre_y + (b*sin(t+pi)*cos(angle) + a*cos(t+pi)*sin(angle))

            y_max = filterCoordinate(max(y1, y2), h)
            y_min = filterCoordinate(min(y1, y2), h)
            y_len_p = abs(y_max - y_min) / 10
            y_max = filterCoordinate(max(y1 + y_len_p, y2 + y_len_p), h)
            y_min = filterCoordinate(min(y1 + y_len_p, y2 + y_len_p), h)

            text = str(int(x_min)) + ' ' + str(int(y_min)) + ' ' + \
                str(int(abs(x_max-x_min))) + ' ' + \
                str(int(abs(y_max-y_min) * 9 / 10)) + ' 0 0 0 0 0 0\n'
            output_file.write(text)

        i = i + num_faces + 2


if __name__ == '__main__':
    output_file = open(OUTPUT_PATH, 'w')
    for i in range(1, 11):
        fileElliName = "FDDB-fold-%02d-ellipseList.txt" % i
        ellipseFilename = ELIP_GT_DIR + fileElliName

        convertEllipseToRect(ellipseFilename, output_file)
    output_file.close()
