'''
Center crop implementation
'''
import random
import logging
from math import inf

import numpy as np
import cv2 as cv
import sparse as sparse3d
from scipy.sparse import dok_matrix

from transforms import Transform


class CenterCrop(Transform):
    '''
    Center crops sample and image (should be ndarrays)
    Its never called on patches
    '''
    def __init__(self, cropx, cropy, cropz=None, segmentation=True, assert_big_enough=False):
        '''
        cropx: crop width
        cropy: crop height
        cropz: crop depth, if not None will consider input a 3D numpy volume
        segmentation: will there be a segmentation as a target?
        assert_big_enough: should i assert that the inputs can be center cropped? (shape > (cropx, cropy...))
        '''
        self.cropx = cropx
        self.cropy = cropy
        self.cropz = cropz
        if cropz is not None:
            self.volumetric = True
        else:
            self.volumetric = False
        self.segmentation = segmentation
        self.assert_big_enough = assert_big_enough

        # TODO suppressed for BTRSeg
        # logging.warning("WARNING: CenterCrop is bugged for non-square/cubic shapes and assert big enough.")

    def __call__(self, img, tgt):
        '''
        img: 2D or 3D numpy array if using cropz
        mask: 2D or 3D numpy array if using cropz
        '''
        cropx = self.cropx
        cropy = self.cropy
        cropz = self.cropz

        if self.assert_big_enough:
            npshape = np.array(img.shape)[-1:-4:-1]
            if self.volumetric:
                assert (npshape < np.array([cropx, cropy, cropz])).sum() == 0, "This image is too small for center crop"
            else:
                assert (npshape < np.array([cropx, cropy])).sum() == 0, "This image is too small for center crop"

        if self.volumetric:
            if img.ndim > 3:
                c, z, y, x = img.shape
            else:
                z, y, x = img.shape
        else:
            if img.ndim > 2:
                c, y, x = img.shape
            else:
                y, x = img.shape

        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        if self.volumetric:
            startz = z//2-(cropz//2)
            if img.ndim > 3:
                rimg = img[:, startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]
            else:
                rimg = img[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]

            if self.segmentation:
                if tgt.ndim > 3:
                    rtgt = tgt[:, startz:startz+cropz, starty:starty + cropy, startx:startx+cropx]
                else:
                    rtgt = tgt[startz:startz+cropz, starty:starty + cropy, startx:startx+cropx]
            else:
                rtgt = tgt

            return rimg, rtgt
        else:  # considers multi channels slices (ndim > 2)
            if img.ndim > 2:
                rimg = img[:, starty:starty + cropy, startx:startx+cropx]
            else:
                rimg = img[starty:starty + cropy, startx:startx + cropx]

            if self.segmentation:
                if tgt.ndim > 2:
                    rtgt = tgt[:, starty:starty + cropy, startx:startx+cropx]
                else:
                    rtgt = tgt[starty:starty + cropy, startx:startx + cropx]
            else:
                rtgt = tgt

            return rimg, rtgt

    def __str__(self):
        return "CenterCrop, patch size: {}x{}x{} volumetric {} segmentation {}".format(self.cropx, self.cropy, self.cropz,
                                                                                       self.volumetric, self.segmentation)


class ReturnPatch(object):
    '''
    Random patch centered around hippocampus
    If no hippocampus present, random patch
    Ppositive is chance of returning a patch around the hippocampus
    Kernel shape is shape of kernel for boundary extraction

    In current state, Multitask selects a random patch
    '''
    def __init__(self, ppositive=0.8, patch_size=(32, 32), kernel_shape=(3, 3), fullrandom=False, anyborder=False, debug=False,
                 segmentation=True, reset_seed=True):
        '''
        Sets desired patchsize (width, height)
        '''
        self.reset_seed = reset_seed
        self.psize = patch_size
        self.ppositive = ppositive
        self.kernel = np.ones(kernel_shape, np.uint8)
        self.ks = kernel_shape
        self.fullrandom = fullrandom
        self.anyborder = anyborder
        self.debug = debug
        dim = len(patch_size)
        assert dim in (2, 3), "only support 2D or 3D patch"
        if dim == 3:
            self.volumetric = True
        elif dim == 2:
            self.volumetric = False
        self.segmentation = segmentation

    def random_choice_3d(self, keylist):
        '''
        Returns random point in 3D sparse COO object
        '''
        lens = [len(keylist[x]) for x in range(3)]
        assert lens[0] == lens[1] and lens[0] == lens[2] and lens[1] == lens[2], "error in random_choice_3d sparse matrix"
        position = random.choice(range(len(keylist[0])))
        point = [keylist[x][position] for x in range(3)]
        return point

    def __call__(self, image, mask, debug=False):
        '''
        Returns patch of image and mask
        '''
        debug = self.debug
        if self.reset_seed:
            random.seed()
        # Get list of candidates for patch center
        e2d = False
        shape = image.shape
        if not self.volumetric and len(shape) == 3:
            shape = (shape[1], shape[2])
            e2d = True

        if not self.fullrandom:
            if self.volumetric:
                borders = np.zeros(shape, dtype=mask.dtype)
                for i in range(shape[0]):
                    uintmask = (mask[i]*255).astype(np.uint8)
                    borders[i] = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(mask.dtype)
                sparse = sparse3d.COO.from_numpy(borders)
                keylist = sparse.nonzero()
            else:
                if mask.ndim > 2:
                    # hmask is now everything, hip only deprecated in multitask
                    if self.anyborder:
                        hmask = mask.sum(axis=0)
                    else:
                        try:
                            hmask = mask[11] + mask[12]
                        except IndexError:  # half labels
                            hmask = mask[6]
                else:
                    hmask = mask
                # Get border of mask
                uintmask = (hmask*255).astype(np.uint8)
                borders = ((uintmask - cv.erode(uintmask, self.kernel, iterations=1))/255).astype(hmask.dtype)
                sparse = dok_matrix(borders)
                keylist = list(sparse.keys())
                if debug:
                    print("Candidates {}".format(keylist))

        # Get top left and bottom right of patch centered on mask border
        four_d_volume = int(image.ndim == 4)
        if self.segmentation:
            four_d_mask = int(mask.ndim == 4)

        tl_row_limit = shape[0 + four_d_volume] - self.psize[0]
        tl_col_limit = shape[1 + four_d_volume] - self.psize[1]
        if self.volumetric:
            tl_depth_limit = shape[2 + four_d_volume] - self.psize[2]
            tl_rdepth = inf
        tl_rrow = inf
        tl_rcol = inf

        if self.fullrandom:
            if self.volumetric:
                tl_rrow, tl_rcol, tl_rdepth = (random.randint(0, tl_row_limit), random.randint(0, tl_col_limit),
                                               random.randint(0, tl_depth_limit))
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)
        elif len(keylist[0]) > 0 and random.random() < self.ppositive:
            if self.volumetric:
                while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit or tl_rdepth > tl_depth_limit:
                    tl_rrow, tl_rcol, tl_rdepth = self.random_choice_3d(keylist)
                    tl_rrow -= self.psize[0]//2
                    tl_rcol -= self.psize[1]//2
                    tl_rdepth -= self.psize[2]//2
            else:
                while tl_rrow > tl_row_limit or tl_rcol > tl_col_limit:
                    tl_rrow, tl_rcol = random.choice(list(sparse.keys()))
                    tl_rrow -= self.psize[0]//2
                    tl_rcol -= self.psize[1]//2
        else:
            if self.volumetric:
                tl_rrow, tl_rcol, tl_rdepth = (random.randint(0, tl_row_limit), random.randint(0, tl_col_limit),
                                               random.randint(0, tl_depth_limit))
            else:
                tl_rrow, tl_rcol = random.randint(0, tl_row_limit), random.randint(0, tl_col_limit)

        if tl_rrow < 0:
            tl_rrow = 0
        if tl_rcol < 0:
            tl_rcol = 0
        if self.volumetric:
            if tl_rdepth < 0:
                tl_rdepth = 0

        if debug:
            print("Patch top left(row, col): {} {}".format(tl_rrow, tl_rcol))

        if self.volumetric:
            if four_d_volume:
                rimage = image[:, tl_rrow:tl_rrow + self.psize[0],
                               tl_rcol:tl_rcol + self.psize[1],
                               tl_rdepth:tl_rdepth + self.psize[2]]
            else:
                rimage = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1], tl_rdepth:tl_rdepth + self.psize[2]]

            if self.segmentation:
                if four_d_mask:
                    rmask = mask[:, tl_rrow:tl_rrow + self.psize[0],
                                 tl_rcol:tl_rcol + self.psize[1],
                                 tl_rdepth:tl_rdepth + self.psize[2]]
                else:
                    rmask = mask[tl_rrow:tl_rrow + self.psize[0],
                                 tl_rcol:tl_rcol + self.psize[1],
                                 tl_rdepth:tl_rdepth + self.psize[2]]
            else:
                rmask = mask
        else:
            if e2d:
                rimage = image[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                rimage = image[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]

            if len(mask.shape) > 2:
                rmask = mask[:, tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]
            else:
                rmask = mask[tl_rrow:tl_rrow + self.psize[0], tl_rcol:tl_rcol + self.psize[1]]

        if debug:
            print(rimage.shape, rmask.shape)
            from matplotlib import pyplot as plt
            fulldisp = image[1] + mask
            fulldisp[fulldisp > 1] = 1
            fulldisp[fulldisp < 0] = 0
            disp = rimage[1] + rmask
            disp[disp > 1] = 1
            disp[disp < 0] = 0
            plt.figure(num="overlap")
            plt.imshow(fulldisp, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Patch overlap")
            plt.imshow(disp, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Mask Patch")
            plt.imshow(rmask, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Brain Patch")
            plt.imshow(rimage[1], cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Borders")
            plt.imshow(borders, cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Brain")
            plt.imshow(image[1], cmap='gray', vmin=0, vmax=1)
            plt.figure(num="Mask")
            plt.imshow(mask, cmap='gray', vmin=0, vmax=1)

        return rimage, rmask

    def __str__(self):
        return ("ReturnPatch: ppositive {} patch_size {}, kernel_shape {}, volumetric {}, "
                "anyborder {}".format(self.ppositive, self.psize, self.ks, self.volumetric, self.anyborder))


def test_patch(display=False, long_test=False):
    from transforms import Compose
    from transforms.to_tensor import ToTensor
    from transforms.intensity import RandomIntensity
    from visualization.multiview import MultiViewer, brats_preparation
    from utils.reproducible import deterministic_run
    from datasets.brats import BRATS
    from tqdm import tqdm
    from matplotlib import pyplot as plt

    deterministic_run()

    transform = Compose([ReturnPatch(patch_size=(128, 128, 128), segmentation=True, fullrandom=True, reset_seed=False),
                         # CenterCrop(128, 128, 128),
                         RandomIntensity(reset_seed=False),
                         ToTensor(volumetric=True, classify=False)])

    dataloader = BRATS(year="2020", release="default", group="all", mode="all",
                       transform=transform, verbose=True,
                       convert_to_eval_format=True).get_dataloader(batch_size=1, shuffle=True,
                                                                   num_workers=12 if long_test else 1)

    iterator = iter(dataloader)
    x, y, tumor, age, survival = next(iterator)
    logging.info(f"{x.shape}, {y.shape}, {BRATS.CLASSES[tumor]}, {age.item()}, {survival.item()}")

    if long_test:
        ysums = []
        ages = []
        survivals = []
        for i, batch in tqdm(enumerate(iterator), desc="Long load test...", total=len(iterator)):
            x, y, tumor, age, survival = batch
            x = x.squeeze()
            xsums = set([x[i].sum() for i in range(4)])
            assert len(xsums) == 4  # asserts all channels are different
            ysums.append(y.sum())
            ages.append(age.item())
            survivals.append(survival.item())
        if display:
            plt.figure()
            plt.xlabel('batch')
            plt.ylabel('target sums')
            plt.plot(range(i+1), ysums)
            plt.figure()
            plt.xlabel('batch')
            plt.ylabel('ages')
            plt.plot(range(i+1), ages)
            plt.figure()
            plt.xlabel('batch')
            plt.ylabel('survivals')
            plt.plot(range(i+1), survivals)
            plt.show()

    ret = "ENTER"
    if display:
        iterator = iter(dataloader)
        while ret == "ENTER":
            logging.info("Press ENTER to look-up the next volume, ESC to exit.")
            x, y, tumor, age, survival = next(iterator)
            x = x.squeeze()
            y = y.squeeze()
            logging.info(f"{x.shape}, {y.shape}, {BRATS.CLASSES[tumor]}, {age.item()}, {survival.item()}")
            x, y = brats_preparation({"data": x.numpy(), "target": y.numpy()})
            ret = MultiViewer(x, y, cube_side=128).display(channel_select=0)
