import mmcv
import cv2
import os
import numpy as np
import mahotas
from tqdm import tqdm

def main():
    img_root = '/home/chenriquan/Datasets/CC/image'
    anno_path = '/home/chenriquan/Datasets/CC/train_anno.pickle'
    texture_root = 'texture'

    annos = mmcv.load(anno_path)

    img_id = 0
    for a in tqdm(annos):
        img = mmcv.imread(os.path.join(img_root, a['filename']))

        grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        binary_T = mahotas.thresholding.rc(grayimg)
        if np.median(grayimg) - binary_T <= 40:
            # low constrast img
            q = a['ann']['quadrilateral'].reshape(-1, 2)
            min_x = int(q[:, 1].min())
            # max_x = int(q[:, 0].max())

            if min_x < 32:
                continue
            else:
                texture = img[0:min_x, :, :]

                if np.random.rand() >= 0.5:
                    texture = texture.transpose(1, 0, 2)
                mmcv.imwrite(texture, os.path.join(texture_root, "%05d.jpg"%img_id))
                img_id += 1




if __name__ == '__main__':
    main()
