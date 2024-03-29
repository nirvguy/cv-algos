import argparse
import cv2
import os
import math
import numpy as np
from itertools import islice

def normal(x, mu=0, sigma=1):
    return np.exp(-(x-mu)**2/(2*sigma**2))

def it_gaussian_pyramid(im, l_max=None):
    if l_max is None:
        l_max = math.floor(math.log2(np.min(im.shape[0:1])))
    yield im
    for l in range(0, l_max):
        im = cv2.pyrDown(im)
        yield im

def it_laplacian_pyramid(im, l_max=None):
    for i, down in enumerate(islice(it_gaussian_pyramid(im, l_max=l_max), 1, None)):
        target_shape = (im.shape[1], im.shape[0])
        im2 = im-cv2.pyrUp(down, dstsize=target_shape)
        yield im2
        im = down
    yield down

def compute_weights(imgs, w=np.ones(3)):
    masks = []
    for i, im in enumerate(imgs):
        gray_1 = im.mean(axis=-1)
        mask_c = np.absolute(cv2.Laplacian(gray_1,cv2.CV_64F))
        mask_s = (im - gray_1[:,:,None]).std(axis=-1)
        mask_e = normal(im, mu=0.5, sigma=0.2).prod(axis=-1)
        mask = (mask_c**w[0])*(mask_s**w[1])*(mask_e**w[2])
        masks.append(mask)
        del gray_1
        del mask_c
        del mask_s
        del mask_e

    masks = np.stack(masks)+1e-12
    masks = masks/masks.sum(axis=0)
    return masks

def get_final_pyramid(imgs, masks, l=None):
    l_max = math.floor(math.log2(np.min(masks[0].shape)))
    if l is not None and (l > l_max or l < 1):
        raise Exception("Invalid l_max parameter. Must lay between [1, {}]. Got: {}".format(l_max, l))
    elif l is not None:
        l_max = l
    levels = [None]*(l_max+1)

    for i, (im, mask) in enumerate(zip(imgs, masks)):
        weights_pyramid = list(it_gaussian_pyramid(mask, l_max))
        im_pyramid = list(it_laplacian_pyramid(im, l_max))
        for j, (w_l, im_l) in enumerate(zip(weights_pyramid, im_pyramid)):
            l = w_l[:,:,None] * im_l
            if levels[j] is None:
                levels[j] = l
            else:
                levels[j] += l
    return levels

def melt_pyramid(levels):
    final_img = levels[-1]
    for level in reversed(levels[:-1]):
        target_shape = (level.shape[1], level.shape[0])
        final_img = level + cv2.pyrUp(final_img, dstsize=target_shape)
    return final_img

def robust_normalization(img, s_white=0.1):
    ch_max = np.max(img, axis=2)
    ch_min = np.min(img, axis=2)
    n = ch_max.size
    s_max = np.sort(ch_max.flatten())
    s_min = np.sort(ch_min.flatten())
    v_max = s_max[math.ceil((1-s_white) * n - 1)]
    v_min = s_min[math.floor(s_white * n)]
    return (np.clip((img - v_min)/(v_max - v_min), 0, 1) * 255).astype(np.uint8)

def fuse(files, w=np.ones(3), l_max=None):
    def iter_imgs():
        for f in files:
            im = cv2.imread(f)
            im = im/255
            yield im

    masks = compute_weights(iter_imgs(), w=w)
    pyramid = get_final_pyramid(iter_imgs(), masks, l=l_max)
    final_img = melt_pyramid(pyramid)
    return final_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--w-c', type=float, default=1)
    parser.add_argument('--w-s', type=float, default=1)
    parser.add_argument('--w-e', type=float, default=1)
    parser.add_argument('--s-white', type=float, default=0.08)
    parser.add_argument('--l-max', type=int)
    parser.add_argument('--normalization', type=str, default='robust', choices=('robust', 'clipping'))
    parser.add_argument('file', type=str, nargs='+')
    args = parser.parse_args()
    final_img = fuse(args.file, w=(args.w_c, args.w_s, args.w_e), l_max=args.l_max)
    if args.normalization == 'robust':
        final_img = robust_normalization(final_img, s_white=args.s_white)
    else:
        final_img = final_img.clip(0, 1) * 255
    cv2.imwrite(args.output, final_img)
    return

if __name__ == '__main__':
    main()
