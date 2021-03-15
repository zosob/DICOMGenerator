#Resizing the dataset

import os
import cv2

src = "./LeftDataset" #Source image with first slice of each image
dest= "./ResizedData"

for each in os.listdir(src):
    img = cv2.imread(os.path.join(src, each))
    img =cv2.resize(img,(256,256))
    cv2.imwrite(os.path.join(dest,each), img)
