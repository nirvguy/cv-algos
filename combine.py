import argparse
import cv2
import os
import numpy as np

def fuse(files, w=np.ones(3)):
    final_img = None
    for f in files:
        im = cv2.imread(f)
        im = im/255
        if final_img is None:
            final_img = im
        else:
            final_img += im
    final_img /= len(files)
    final_img = cv2.normalize(final_img,None,0,255,cv2.NORM_MINMAX)

    return final_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('file', type=str, nargs='+')
    args = parser.parse_args()
    final_img = fuse(args.file)
    cv2.imwrite(args.output, final_img)
    return

if __name__ == '__main__':
    main()
