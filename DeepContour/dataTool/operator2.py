import cv2
import math
import numpy as np
import os
import random
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from dataTool.operater import Basic_Operator
from scipy import fftpack, ndimage


class Basic_Operator2:

    def add_original_back(new, back, mask):
        mask = np.clip(mask, 0, 1)
        new = new + back * mask
        # create mask1
        # keep thebakcground part

        return new

    def pure_background(img, contourx, contoury, H, W):

        def flip_source_to_full(sourimg, H, W):
            # this is to filip the source and duplicate on patch to a full image

            # first flip through Horizontal
            sr_H, sr_W = sourimg.shape
            pend_cnt = int(W / sr_W) + 1
            pender = cv2.flip(sourimg, 1)
            new = sourimg
            for i in range(pend_cnt):
                new = np.append(new, pender, axis=1)  # cascade
                pender = cv2.flip(pender, 1)

            # then flip through vertial
            sourimg = new
            sr_H, sr_W = sourimg.shape
            pend_cnt = int(H / sr_H) + 1
            pender = cv2.flip(sourimg, 0)
            new = sourimg
            for i in range(pend_cnt):
                new = np.append(new, pender, axis=0)  # cascade
                pender = cv2.flip(pender, 0)

            out = new[0:H, 0:W]
            return out
            # pass

        # use different strategies:
        contoury = contoury.astype(int)
        contourx = contourx.astype(int)

        ori_H, ori_W = img.shape
        points = len(contourx[1])
        new = np.zeros((H, W))
        c_len = len(contourx[1])  # use the send
        # c_len0 = len(contourx[0]) # use the send

        # Dice = int( np.random.random_sample()*10)
        # method 1 just use the left and right side of the imag to raasampel

        min_b = int(np.max(contoury[0]))
        max_b = int(np.min(contoury[1]))
        if c_len < 0.8 * ori_W:
            # TODO: it will never get in this condition --- we make sure that all a-lines have a contour value,
            #  even if it is the image height. So why is this here?
            min_b = int(np.max(contoury[0]))

            sourimg1 = img[min_b:ori_H, 0:contourx[1][0]]
            sourimg2 = img[min_b:ori_H, contourx[1][c_len - 2]: ori_W]
            sourimg = np.append(sourimg2, sourimg1, axis=1)  # the right sequence
            # out = flip_source_to_full(sourimg,H,W)
            # sr_H,sr_W  = sourimg.shape
            # pend_cnt  = int(W/sr_W)+1
            # pender  =   cv2.flip(sourimg, 1)
            # new  = sourimg
            # for i in range(pend_cnt):
            #    new  = np.append(new,pender, axis=1) # cascade
            #    pender  =   cv2.flip(pender, 1)
            # out  = new[:,0:W] # crop out the sheth
            # out  = cv2.resize(out, (W,H), interpolation=cv2.INTER_LINEAR )
        else:
            if (
                    max_b - min_b) > 200:  # TODO: create condition that works for IVUS and OCT. Currently it is user-defined
                # method 2 the line is generated with the line above the the contour
                # generate line by line
                # min_b  = int(np.max(contoury[0]))
                # max_b  = int(np.min(contoury[1]))
                # if (max_b -min_b)>200:
                sourimg = img[min_b:max_b, :]
                # out = flip_source_to_full(sourimg,H,W)

                # sr_H,sr_W  = sourimg.shape
                # pend_cnt  = int(H/sr_H)+1
                # pender  =   cv2.flip(sourimg, 0)
                # new  = sourimg
                # for i in range(pend_cnt):
                #    new  = np.append(new,pender, axis=0) # cascade
                #    pender  =   cv2.flip(pender, 0)

                # out  = new
                # out  = cv2.resize(out, (W,H), interpolation=cv2.INTER_LINEAR )
            else:  # del with this special condition when full sorround contour 
                # left_a = np.max([contourx[0][0],contourx[1][0]])
                # right_a = np.max([contourx[0][0],contourx[c_len][0]])
                index = 0
                source_i = 0
                sourimg = np.zeros((ori_H, 50))
                # calculate the with between 2 bondaries
                py1_py2 = contoury[1] - contoury[0]
                max_d = int(0.5 * np.max(py1_py2))
                sourimg = np.zeros((max_d, W))  # create a block based on the area

                while (1):
                    if (contoury[1][index] - contoury[0][index]) > max_d :
                        sourimg[:, source_i] = img[int(contoury[0][index]):int(contoury[0][index]) + max_d,
                                               contourx[1][index]]
                        # TODO: Ask GUIQIU! Why using catheter contour for the rows and lumen contour for the columns?
                        source_i += 1
                        if source_i >= W:
                            break
                    index += 1
                    if index >= len(contoury[1]):
                        index = 0

                # sr_H,sr_W  = sourimg.shape
                # pend_cnt  = int(W/sr_W)+1 # pend through horizontal
                # pender  =   cv2.flip(sourimg, 1)
                # new  = sourimg
                # for i in range(pend_cnt):
                #    new  = np.append(new,pender, axis=1) # cascade
                #    pender  =   cv2.flip(pender, 1)
                ##min_b  = int(np.max(contoury[0]))   
                # out  = new[:,0:W] # crop out the sheth
                # out  = cv2.resize(out, (W,H), interpolation=cv2.INTER_LINEAR )
        out = flip_source_to_full(sourimg, H, W)

        return out

    #### Extract a mask for the IVUS background from the original contours to pass to the background blood generator
    @staticmethod
    def set_background_mask(h, top_contoury, base_contoury):

        # to remove pixels too close to the contours and just grab background
        top_contoury = np.around(top_contoury) + 3
        base_contoury = np.around(base_contoury) - 3

        # if the contour is not defined, set its value to zero so it does not get used as background (mask out)
        base_contoury[base_contoury >= (h - 4)] = 0

        # for all the pixels between the two contours, we set the mask at 1, which is where the background is
        back_mask = np.logical_and(np.array(range(h))[:, None] < base_contoury,
                                   np.array(range(h))[:, None] > top_contoury) * 1

        return back_mask

    # fill wire area (below the wire contour) with original data for IVUS images
    @staticmethod
    def fill_wire_ivus(img1, H, wire_idx, contour0y, new_contoury, new, mask):

        points = len(wire_idx)

        for i in range(points - 1):

            source_line = img1[:, wire_idx[i]]
            newy = int(new_contoury[wire_idx[i]])
            iniy = int(contour0y[wire_idx[i]]) - 3  # - 3 to give more highlight to the boundary
            shift = int(newy - iniy)

            if shift > 0:
                new[newy:H, wire_idx[i]] = source_line[iniy:H - shift]
                mask[newy:H, wire_idx[i]] = 0
            else:
                new[newy:H + shift, wire_idx[i]] = source_line[iniy:H]
                mask[newy:H + shift, wire_idx[i]] = 0

                # fill the bottom section with 0s
                new[H + shift:H, wire_idx[i]] = 0
                mask[H + shift:H, wire_idx[i]] = 0

        return new, mask

    @staticmethod
    def generate_ivus_blood_background(img, back_mask, h, w):

        # Create grid to fill with patch of background
        # grid = np.zeros((h, w))

        # Split img mask in 4 quadrants: to be used as 4 grids to randomly fill the background faster
        # since the pixels to fill are randomly chosen
        quad_ii, quad_iii = np.vsplit(back_mask, 2)
        quad_ii, quad_i = np.hsplit(quad_ii, 2)
        quad_iii, quad_iv = np.hsplit(quad_iii, 2)

        mask_h, mask_w = quad_i.shape  # same size for all

        # back_mask = np.clip(back_mask, 0, 1)  # TODO: remove after making sure that it gets 0s and 1s from get_background_mask()

        # determine approximately the background region size that will be used in the generated background
        back_region_idx = np.array(np.where(back_mask == 1))  # check rows that are filled with 1s

        # crop original background from image: both the mask and the image itself
        # Mask
        crop_mask = back_mask[min(back_region_idx[0]):max(back_region_idx[0]),
                              min(back_region_idx[1]):max(back_region_idx[1])]

        # Background image (only)
        back_region = back_mask * img
        crop_region = back_region[min(back_region_idx[0]):max(back_region_idx[0]),
                                  min(back_region_idx[1]):max(back_region_idx[1])]

        # find empty full columns probably due to the wire(s)
        empty_columns = np.where(~crop_mask.any(axis=0))[0]
        crop_region = np.delete(crop_region, empty_columns, axis=1)
        crop_mask = np.delete(crop_mask, empty_columns, axis=1)

        flipped_to_fill_mask = (crop_mask == np.zeros(crop_mask.shape))
        flipped_once = flipped_to_fill_mask * cv2.flip(crop_region, 0) + crop_region

        # Update mask and flip again
        flipped_to_fill_mask = (flipped_to_fill_mask * cv2.flip(crop_mask, 0) + crop_mask) == np.zeros(
            flipped_to_fill_mask.shape)

        flipped_twice = flipped_to_fill_mask * cv2.flip(crop_region, 1) + flipped_once

        # In this last step the flipped_to_fill_mask is 0 when the pixel is not filled w.r.t. to the above times
        flipped_to_fill_mask = flipped_to_fill_mask * cv2.flip(crop_mask, 1) + (1 - (flipped_to_fill_mask * 1))

        # find rows that are not fully filled and remove them
        remove_rows = np.where(~flipped_to_fill_mask.all(axis=1))[0]
        crop_flipped = np.delete(flipped_twice, remove_rows, axis=0)
        # crop_mask_flipped = np.delete(flipped_to_fill_mask, remove_rows, axis=0)

        flipped_full_x = crop_flipped
        for _ in range(flipped_twice.shape[1] // crop_flipped.shape[1] + 1):
            flipped_full_x = np.append(flipped_full_x, crop_flipped, axis=1)

        flipped_full_xy = flipped_full_x
        for i_y in range(flipped_twice.shape[0] // crop_flipped.shape[0] + 1):
            flipped_full_xy = np.append(flipped_full_xy, flipped_full_x, axis=0)

        flipped_full_xy = flipped_full_xy[0:flipped_twice.shape[0], 0:flipped_twice.shape[1]]

        full_crop_region = (1-flipped_to_fill_mask) * flipped_full_xy + flipped_twice
        full_crop_mask = np.ones(full_crop_region.shape)

        # Split back_region in 4 quadrants. The grids are already full with the original contour in the original place
        # this function will cover the remaining sections that will need to show background due to the new contours
        #  II  |  I
        # -----------
        #  III |  IV
        quad_ii_region, quad_iii_region = np.vsplit(back_region, 2)
        quad_ii_region, quad_i_region = np.hsplit(quad_ii_region, 2)
        quad_iii_region, quad_iv_region = np.hsplit(quad_iii_region, 2)

        # find indexes of elements still not filled (for index i, mask == 0) - before iterating to fill them
        # all to be filled at this point
        fill_quad_i = np.array(np.where(quad_i == 0))
        fill_quad_ii = np.array(np.where(quad_ii == 0))
        fill_quad_iii = np.array(np.where(quad_iii == 0))
        fill_quad_iv = np.array(np.where(quad_iv == 0))

        # fill each quadrant of the grid as long as there are empty pixels
        while fill_quad_i.size != 0 or fill_quad_ii.size != 0 or fill_quad_iii.size != 0 or fill_quad_iv.size != 0:

            # # initialize the centers (helps in case the quadrant is already full)
            # center_quad_i = []
            # center_quad_ii = []
            # center_quad_iii = []
            # center_quad_iv = []

            angle = random.randint(-180, 180)  # small angle in degrees
            # angle = 0
            rotated_back_mask = np.around(ndimage.rotate(full_crop_mask, angle, order=1))
            rotated_back_mask = np.clip(rotated_back_mask, 0, 1)  # just to make sure
            rotated_back_region = ndimage.rotate(full_crop_region, angle, order=1)

            starty = rotated_back_mask.shape[0] // 2 - mask_h // 2
            startx = rotated_back_mask.shape[1] // 2 - mask_w // 2
            # if it is negative, we need to do padding
            if startx < 0 or starty < 0:
                rotated_back_region = np.pad(rotated_back_region, [(np.clip(-starty, 0, mask_h),
                                                                    np.clip(-starty, 0, mask_h)),
                                                                   (np.clip(-startx, 0, mask_w),
                                                                    np.clip(-startx, 0, mask_w))],
                                             mode='constant', constant_values=0)

                rotated_back_mask = np.pad(rotated_back_mask, [(np.clip(-starty, 0, mask_h),
                                                                np.clip(-starty, 0, mask_h)),
                                                               (np.clip(-startx, 0, mask_w),
                                                                np.clip(-startx, 0, mask_w))],
                                           mode='constant', constant_values=0)

                # compute again for the padded arrays
                starty = rotated_back_mask.shape[0] // 2 - mask_h // 2
                startx = rotated_back_mask.shape[1] // 2 - mask_w // 2

            rotated_back_mask = rotated_back_mask[starty:starty + mask_h, startx:startx + mask_w]
            rotated_back_region = rotated_back_region[starty:starty + mask_h, startx:startx + mask_w]

            # get 1 random center for the background region to fill if the whole quadrant still has empty pixels
            if fill_quad_i.size != 0:
                center_quad_i = fill_quad_i[:, random.sample(range(0, len(fill_quad_i[0])), 1)]
                # TODO: Shifting the center to the top of the image corner
                # center_quad_i[0] = center_quad_i[0] - full_crop_region.shape[0] // 2  # y
                # center_quad_i[1] = center_quad_i[1] - full_crop_region.shape[1] // 2  # x

                # if random center is on the mask center right, the difference will be positive and vice-versa
                pad_x_i = [2*int(np.clip(center_quad_i[1] - rotated_back_mask.shape[1] // 2, 0, mask_w)),
                           2*int(abs(np.clip(center_quad_i[1] - rotated_back_mask.shape[1] // 2, -mask_w, 0)))]

                # if random center is below the mask center, the difference will be positive and vice-versa
                pad_y_i = [2*int(np.clip(center_quad_i[0] - rotated_back_mask.shape[0] // 2, 0, mask_h)),
                           2*int(abs(np.clip(center_quad_i[0] - rotated_back_mask.shape[0] // 2, -mask_h, 0)))]

                quad_i_shift_mask = np.pad(rotated_back_mask, [pad_y_i, pad_x_i],
                                           mode='constant', constant_values=0)
                quad_i_shift_region = np.pad(rotated_back_region, [pad_y_i, pad_x_i],
                                             mode='constant', constant_values=0)

                # crop at the middle again
                starty_i = quad_i_shift_mask.shape[0] // 2 - mask_h // 2
                startx_i = quad_i_shift_mask.shape[1] // 2 - mask_w // 2

                quad_i_shift_mask = quad_i_shift_mask[starty_i:starty_i + mask_h, startx_i:startx_i + mask_w]
                quad_i_shift_region = quad_i_shift_region[starty_i:starty_i + mask_h, startx_i:startx_i + mask_w]

                comb_mask_i = (quad_i < quad_i_shift_mask).astype(int)
                quad_i = comb_mask_i * quad_i_shift_mask + quad_i
                quad_i_region = comb_mask_i * quad_i_shift_region + quad_i_region

            if fill_quad_ii.size != 0:
                center_quad_ii = fill_quad_ii[:, random.sample(range(0, len(fill_quad_ii[0])), 1)]

                # if random center is on the mask center right, the difference will be positive and vice-versa
                pad_x_ii = [2*int(np.clip(center_quad_ii[1] - rotated_back_mask.shape[1] // 2, 0, mask_w)),
                            2*int(abs(np.clip(center_quad_ii[1] - rotated_back_mask.shape[1] // 2, -mask_w, 0)))]

                # if random center is below the mask center, the difference will be positive and vice-versa
                pad_y_ii = [2*int(np.clip(center_quad_ii[0] - rotated_back_mask.shape[0] // 2, 0, mask_h)),
                            2*int(abs(np.clip(center_quad_ii[0] - rotated_back_mask.shape[0] // 2, -mask_h, 0)))]

                quad_ii_shift_mask = np.pad(rotated_back_mask, [pad_y_ii, pad_x_ii],
                                            mode='constant', constant_values=0)
                quad_ii_shift_region = np.pad(rotated_back_region, [pad_y_ii, pad_x_ii],
                                              mode='constant', constant_values=0)

                # crop at the middle again
                starty_ii = quad_ii_shift_mask.shape[0] // 2 - mask_h // 2
                startx_ii = quad_ii_shift_mask.shape[1] // 2 - mask_w // 2

                quad_ii_shift_mask = quad_ii_shift_mask[starty_ii:starty_ii + mask_h, startx_ii:startx_ii + mask_w]
                quad_ii_shift_region = quad_ii_shift_region[starty_ii:starty_ii + mask_h, startx_ii:startx_ii + mask_w]

                comb_mask_ii = (quad_ii < quad_ii_shift_mask).astype(int)
                quad_ii = comb_mask_ii * quad_ii_shift_mask + quad_ii
                quad_ii_region = comb_mask_ii * quad_ii_shift_region + quad_ii_region

            if fill_quad_iii.size != 0:
                # center_quad_iii = fill_quad_iii[:, random.sample(range(0, len(fill_quad_iii[0])),
                #                                                  np.clip(len(fill_quad_iii[0]), 0, 10))]

                center_quad_iii = fill_quad_iii[:, random.sample(range(0, len(fill_quad_iii[0])), 1)]

                # if random center is on the mask center right, the difference will be positive and vice-versa
                pad_x_iii = [2*int(np.clip(center_quad_iii[1] - rotated_back_mask.shape[1] // 2, 0, mask_w)),
                             2*int(abs(np.clip(center_quad_iii[1] - rotated_back_mask.shape[1] // 2, -mask_w, 0)))]

                # if random center is below the mask center, the difference will be positive and vice-versa
                pad_y_iii = [2*int(np.clip(center_quad_iii[0] - rotated_back_mask.shape[0] // 2, 0, mask_h)),
                             2*int(abs(np.clip(center_quad_iii[0] - rotated_back_mask.shape[0] // 2, -mask_h, 0)))]

                quad_iii_shift_mask = np.pad(rotated_back_mask, [pad_y_iii, pad_x_iii],
                                             mode='constant', constant_values=0)
                quad_iii_shift_region = np.pad(rotated_back_region, [pad_y_iii, pad_x_iii],
                                               mode='constant', constant_values=0)

                # crop at the middle again
                starty_iii = quad_iii_shift_mask.shape[0] // 2 - mask_h // 2
                startx_iii = quad_iii_shift_mask.shape[1] // 2 - mask_w // 2

                quad_iii_shift_mask = quad_iii_shift_mask[starty_iii:starty_iii + mask_h,
                                                          startx_iii:startx_iii + mask_w]
                quad_iii_shift_region = quad_iii_shift_region[starty_iii:starty_iii + mask_h,
                                                              startx_iii:startx_iii + mask_w]

                comb_mask_iii = (quad_iii < quad_iii_shift_mask).astype(int)
                quad_iii = comb_mask_iii * quad_iii_shift_mask + quad_iii
                quad_iii_region = comb_mask_iii * quad_iii_shift_region + quad_iii_region

            if fill_quad_iv.size != 0:
                center_quad_iv = fill_quad_iv[:, random.sample(range(0, len(fill_quad_iv[0])), 1)]

                # if random center is on the mask center right, the difference will be positive and vice-versa
                pad_x_iv = [2*int(np.clip(center_quad_iv[1] - rotated_back_mask.shape[1] // 2, 0, mask_w)),
                            2*int(abs(np.clip(center_quad_iv[1] - rotated_back_mask.shape[1] // 2, -mask_w, 0)))]

                # if random center is below the mask center, the difference will be positive and vice-versa
                pad_y_iv = [2*int(np.clip(center_quad_iv[0] - rotated_back_mask.shape[0] // 2, 0, mask_h)),
                            2*int(abs(np.clip(center_quad_iv[0] - rotated_back_mask.shape[0] // 2, -mask_h, 0)))]

                quad_iv_shift_mask = np.pad(rotated_back_mask, [pad_y_iv, pad_x_iv],
                                            mode='constant', constant_values=0)
                quad_iv_shift_region = np.pad(rotated_back_region, [pad_y_iv, pad_x_iv],
                                              mode='constant', constant_values=0)

                # crop at the middle again
                starty_iv = quad_iv_shift_mask.shape[0] // 2 - mask_h // 2
                startx_iv = quad_iv_shift_mask.shape[1] // 2 - mask_w // 2

                quad_iv_shift_mask = quad_iv_shift_mask[starty_iv:starty_iv + mask_h, startx_iv:startx_iv + mask_w]
                quad_iv_shift_region = quad_iv_shift_region[starty_iv:starty_iv + mask_h, startx_iv:startx_iv + mask_w]

                comb_mask_iv = (quad_iv < quad_iv_shift_mask).astype(int)
                quad_iv = comb_mask_iv * quad_iv_shift_mask + quad_iv
                quad_iv_region = comb_mask_iv * quad_iv_shift_region + quad_iv_region

            # check again after each iteration
            fill_quad_i = np.array(np.where(quad_i == 0))
            fill_quad_ii = np.array(np.where(quad_ii == 0))
            fill_quad_iii = np.array(np.where(quad_iii == 0))
            fill_quad_iv = np.array(np.where(quad_iv == 0))

        # put all quadrants together to create a full image with background with the same size as img
        # Remember the order:
        #  II  |  I
        # -----------
        #  III |  IV
        back_img = np.append(np.append(quad_ii_region, quad_i_region, axis=1),
                             np.append(quad_iii_region, quad_iv_region, axis=1),
                             axis=0)

        return back_img

    # use the H and W of origina to confine , and generate a random reseanable signal in the window
    def upsample_background(img, H_new, W_new):
        # use fft to upsample 
        H, W = img.shape
        im_fft = fftpack.fft2(img)
        im_fft2 = im_fft.copy()
        H, W = img.shape

        LR = np.zeros((H, int((W_new - W) / 2)))
        new = np.append(LR, im_fft2, axis=1)  # cascade
        new = np.append(new, LR, axis=1)  # cascade
        H, W = new.shape
        TB = np.zeros(((int((H_new - H) / 2)), W))
        new = np.append(TB, new, axis=0)  # cascade
        new = np.append(new, TB, axis=0)  # cascade
        new_img = fftpack.ifft2(im_fft2).real
        new_img = cv2.resize(img, (W_new, H_new), interpolation=cv2.INTER_AREA)

        return new_img

    def random_sheath_contour(H, W, x, y):
        # first need to determine whether use the original countour to shift

        # random rol the sheath 

        np.roll(y, int(np.random.random_sample() * len(y) - 1))

        dc1 = np.random.random_sample() * 10
        dc1 = int(dc1) % 1
        if dc1 == 0:  # not use the original signal
            # inital ramdon width and height

            # should mainly based o the sheath orginal contor
            newy = signal.resample(y, W)
            newx = np.arange(0, W)
            r_vector = np.random.sample(20) * 10
            r_vector = signal.resample(r_vector, W)
            r_vector = gaussian_filter1d(r_vector, 10)

            randomshift = np.random.random_sample() * 100 - 50
            newy = newy + r_vector + randomshift

            newy = np.clip(newy, 50, H - 1)
        else:
            newy = signal.resample(y, W)
            newx = np.arange(0, W)
        # width  = 30% - % 50

        # sample = np.arange(width)
        # r_vector   = np.random.sample(20)*20
        # r_vector = gaussian_filter1d (r_vector ,10)
        # newy = np.sin( 1*np.pi/width * sample)
        # newy = -new_contoury*(dy2-dy1)+dy2
        # newy=new_contoury+r_vector
        # newx = np.arange(dx1, dx2)
        return newx, newy

    def random_sheath_contour_ivus(H, W, x, y):
        # first need to determine whether use the origina lcountour to shift

        # random rol the sheath 

        np.roll(y, int(np.random.random_sample() * len(y) - 1))

        dc1 = np.random.random_sample() * 10
        dc1 = int(dc1) % 2
        if dc1 == 0:  # not use the original signal
            # inital ramdon width and height

            # should mainly based o the sheath orginal contor
            newy = signal.resample(y, W)
            newx = np.arange(0, W)
            r_vector = np.random.sample(20) * 10
            r_vector = signal.resample(r_vector, W)
            r_vector = gaussian_filter1d(r_vector, 10)

            randomshift = np.random.random_sample() * 20 - 10
            newy = newy + r_vector + randomshift

        else:
            newy = signal.resample(y, W)
            newx = np.arange(0, W)
        # width  = 30% - % 50

        # sample = np.arange(width)
        # r_vector   = np.random.sample(20)*20
        # r_vector = gaussian_filter1d (r_vector ,10)
        # newy = np.sin( 1*np.pi/width * sample)
        # newy = -new_contoury*(dy2-dy1)+dy2
        # newy=new_contoury+r_vector
        # newx = np.arange(dx1, dx2)
        # Flag for the different modalities
        if H < 800:  # user-defined value based on IVUS images being 512x512 and OCT 1024x1024, typically
            newy = np.clip(newy, 35, H - 1)  # appropriate for OCT since image is 1024. IVUS is 512. Use 35 for IVUS
        else:
            newy = np.clip(newy, 50, H - 1)

        return newx, newy

    def random_shape_contour3(H_ini, W_ini, H, W, sx, sy, x, y):
        # simple version, just move up and down
        dc1 = np.random.random_sample() * 100
        leny = len(y)
        mask = y < (H_ini - 20)
        r_vector = np.random.sample(20) * 50
        r_vector = signal.resample(r_vector, leny)
        r_vector = gaussian_filter1d(r_vector, 3)
        shift = np.random.random_sample() * H - H / 1.2
        newy = y + mask * (r_vector + shift)

        newx = x

        for i in range(len(newy)):
            newy[i] = np.clip(newy[i].astype(int), sy[newx[i].astype(int) ]- 1, H - 1)  # allow it to merge int o 1 pix

        # width  = 30% - % 50
        newy = np.clip(newy, 0, H - 1)
        return newx, newy

    # this one will not regard the original width of contour
    def random_shape_contour2(H_ini, W_ini, H, W, sx, sy, x, y):
        dc1 = np.random.random_sample() * 100

        if int(dc1) % 2 != 0:  # use the original signal
            # inital ramdon width and height

            width = int((0.05 + 0.91 * np.random.random_sample()) * W)
            dx1 = int(np.random.random_sample() * (W - width))
            dx2 = dx1 + width
            dy1 = int(np.random.random_sample() * H * 1.5 - 0.25 * H)
            dy2 = int(np.random.random_sample() * (H * 1.5 - dy1)) + dy1

            height = dy2 - dy1
            # star and end
            # new x
            newx = np.arange(dx1, dx2)
            # new y based on a given original y
            newy = signal.resample(y, width)
            r_vector = np.random.sample(20) * 200
            r_vector = signal.resample(r_vector, width)
            r_vector = gaussian_filter1d(r_vector, 3)
            newy = newy + r_vector
            miny = min(newy)
            height0 = max(newy) - miny
            newy = (newy - miny) * height / height0 + dy1
        else:

            if int(dc1) % 4 != 0:
                newy = y + np.random.random_sample() * H
                newx = x
            else:
                newy = H - y
                newy = np.clip(newy, 20, H - 1) - 20
                newy = newy * np.random.random_sample() * 2
                newy = H - newy
                newx = x

        for i in range(len(newy)):
            newy[i] = np.clip(newy[i], sy[newx[i]] - 1, H - 1)  # allow it to merge int o 1 pix

        # width  = 30% - % 50
        newy = np.clip(newy, 0, H - 1)

        return newx, newy

    # draw color contour
    def random_shape_contour(H_ini, W_ini, H, W, sx, sy, x, y):
        # determine the tissue contour based o hte determined sheath contour
        dc1 = np.random.random_sample() * 100
        dc1 = int(dc1) % 10
        if dc1 != 0:  # use the original signal
            # inital ramdon width and height

            width = int((0.05 + 0.91 * np.random.random_sample()) * W)
            dx1 = int(np.random.random_sample() * (W - width))
            dx2 = dx1 + width
            dy1 = int(np.random.random_sample() * H * 1.5 - 0.25 * H)
            dy2 = int(np.random.random_sample() * (H * 1.5 - dy1)) + dy1

            height = dy2 - dy1
            # star and end
            # new x
            newx = np.arange(dx1, dx2)
            # new y based on a given original y
            newy = signal.resample(y, width)
            r_vector = np.random.sample(20) * 50
            r_vector = signal.resample(r_vector, width)
            r_vector = gaussian_filter1d(r_vector, 10)
            newy = newy + r_vector
            miny = min(newy)
            height0 = max(newy) - miny
            newy = (newy - miny) * height / height0 + dy1
        else:
            newy = y
            newx = x
        if len(x) > 0.96 * W_ini:  # consider the special condition of full and gapin middel

            width = W
            dx1 = 0
            dx2 = W
            dy1 = int(np.random.random_sample() * H * 1.5 - 0.25 * H)
            dy2 = int(np.random.random_sample() * (H * 1.5 - dy1)) + dy1

            height = dy2 - dy1
            # star and end
            # new x
            newx = np.arange(dx1, dx2)
            # new y based on a given original y
            newy = signal.resample(y, width)
            r_vector = np.random.sample(20) * 50
            r_vector = signal.resample(r_vector, width)
            r_vector = gaussian_filter1d(r_vector, 10)
            newy = newy + r_vector
            miny = min(newy)
            height0 = max(newy) - miny
            newy = (newy - miny) * height / height0 + dy1

            # rememver to add resacle later
            # newy = signal.resample(y, W)
            # newx = np.arange(0, W)
            ##np.roll(y, int(np.random.random_sample()*len(y)-1)) 
            # newy  = newy +  np.random.random_sample() *H/2

            # and also deal with black area
            for i in range(len(y)):
                if y[i] >= (H_ini - 5):
                    newy[i] = H - 1  # allow it to merge int o 1 pix

        # limit by the bondary of the sheath
        for i in range(len(newy)):
            newy[i] = np.clip(newy[i], sy[newx[i]] - 1, H - 1)  # allow it to merge int o 1 pix

        # width  = 30% - % 50
        newy = np.clip(newy, 0, H - 1)

        return newx, newy

    # draw color contour
    def random_shape_contour_ivus(H_ini, W_ini, H, W, sx, sy, x, y):
        # simple version, just move up and down
        dc1 = np.random.random_sample() * 100
        leny = len(y)
        mask = y < (H_ini - 20)
        r_vector = np.random.sample(20) * 50
        r_vector = signal.resample(r_vector, leny)
        r_vector = gaussian_filter1d(r_vector, 3)
        shift = np.random.random_sample() * H / 2 - H / 4
        newy = y + mask * (r_vector + shift)

        newx = x

        for i in range(len(newy)):
            newy[i] = np.clip(newy[i], sy[newx[i]] + 1, H - 1)  # allow it to merge int o 1 pix

        # width  = 30% - % 50
        newy = np.clip(newy, 0, H - 1)
        return newx, newy

    @staticmethod
    def random_shape_contour_ivus_multi(H, top_contoury, contourx, contoury):

        leny = len(contoury[0])

        # TODO: adapt these values to less user-defined and more related with proportions in image? std and [0, 50)?
        # compute smooth random distorted signal over the contour width and a random shift value within [- H/4, H/4)
        r_vector = np.random.sample(20) * 50
        r_vector = signal.resample(r_vector, leny)
        r_vector = gaussian_filter1d(r_vector, 3)
        shift = np.random.random_sample() * H / 2 - H / 4

        # newy = [None] * len(contoury[:, 0])
        newx = contourx
        newy = np.ones([len(contoury[:, 0]), len(contourx[0, :])]) * (H - 1)

        # generate contours with the same shift and random 'distortion' for all the contours
        for idx in range(len(newy)):
            mask = contoury[idx] < (0.98 * H)
            newy[idx] = contoury[idx] + mask * (r_vector + shift)

            for i in range(len(newy[idx])):
                # allow it to merge int o 1 pix
                newy[idx, i] = np.clip(newy[idx, i], top_contoury[newx[idx, i]] + 1, H - 1)

            newy[idx] = np.clip(newy[idx], 0, H - 1)  # extra safety to make sure that no contour is outside the img

        return newx, newy

    # generate contour for guide-wire in IVUS. Specific operation.
    @staticmethod
    def random_wire_contour_ivus(H, x, y, sy_ini, wire_idx):

        len_y = len(y)  # number of points
        len_sy = len(sy_ini)  # number of wires (and associated contours)
        sy = np.zeros(len_y)

        mask = y < (H * 0.98)
        new_y = y.copy()
        new_x = x

        if len_sy != 0:  # if 0 -> no top contour
            for i in range(len_sy):  # assign the top contours to the respective indexes of sy
                sy[wire_idx[i]] = sy_ini[i, wire_idx[i]]

                shift = np.random.random_sample() * H / 4 - H / 8  # a different shift per wire
                new_y[wire_idx[i]] = y[wire_idx[i]] + mask[wire_idx[i]] * shift

        else:  # -> consider image boundaries
            shift = np.random.random_sample() * H / 4 - H / 8
            new_y = y + mask * shift

        for i in range(len(new_y)):
            new_y[i] = np.clip(new_y[i], sy[i] + 1, H - 1)  # allow it to merge int o 1 pix
            # assumes that sy is above the y

        new_y = np.clip(new_y, 0, H - 1)

        return new_x, new_y

    # this one will not regard the original width of contour
    def fill_sheath_with_contour_ivus(img1, H_new, W_new, contour0x, contour0y,
                                      new_contourx, new_contoury):
        H, W = img1.shape
        img1 = cv2.resize(img1, (W_new, H_new), interpolation=cv2.INTER_AREA)
        contour0y = contour0y * H_new / H
        points = len(contour0x)
        points_new = len(new_contoury)
        # W_new  = points_new
        new = np.zeros((H_new, W_new))
        mask = new + 255
        contour0y = signal.resample(contour0y, W_new)
        contour0x = np.arange(0, W_new)
        # use a dice to determin wheterh follw orgina sequnce 
        Dice = int(np.random.random_sample() * 10)

        for i in range(W_new):

            line_it = i  # TODO: Why doing this and not using just i all the way?

            source_line = img1[:, contour0x[line_it]]
            # directly_fill in new
            newy = int(new_contoury[i])
            iniy = int(contour0y[line_it])  # add 5 to give more high light bondaries
            shift = int(newy - iniy)
            if shift < 0:
                new[0:newy, i] = source_line[-shift:iniy]
                mask[0:newy, i] = 0
            else:
                new[shift:newy, i] = source_line[0:iniy]
                mask[shift:newy, i] = 0

            pass
        pass
        return new, mask

    def fill_sheath_with_contour(img1, H_new, W_new, contour0x, contour0y,
                                 new_contourx, new_contoury):
        H, W = img1.shape
        img1 = cv2.resize(img1, (W_new, H_new), interpolation=cv2.INTER_AREA)
        contour0y = contour0y * H_new / H
        points = len(contour0x)
        points_new = len(new_contoury)
        # W_new  = points_new
        new = np.zeros((H_new, W_new))
        mask = new + 255
        contour0y = signal.resample(contour0y, W_new)
        contour0x = np.arange(0, W_new)
        # use a dice to determin wheterh follw orgina sequnce 
        Dice = int(np.random.random_sample() * 10)

        for i in range(W_new):

            if Dice % 10 == 0:  # less possibility to random select the A line s
                line_it = int(np.random.random_sample() * points)
                line_it = np.clip(line_it, 0, points - 1)

            else:
                line_it = i
            # line_it   =  i

            source_line = img1[:, contour0x[line_it]]
            # directly_fillinnew
            newy = int(new_contoury[i])
            if (newy > 0.96 *H):
                new[0:newy, i] = img1[0:newy, i]
                continue

            iniy = int(contour0y[line_it])   # add 5 to give more high light bondaries
            shift = int(newy - iniy)
            if shift < 0:
                new[0:newy, i] = source_line[-shift:iniy]
                mask[0:newy, i] = 0
            else:
                new[shift:newy, i] = source_line[0:iniy]
                mask[shift:newy, i] = 0

            pass
        pass
        return new, mask

    # this new converting function will use the orginal conection between the back and tehe tissue, also need the sheath
    def fill_patch_base_origin3_soft_edge(img1, H_new, contoury_s, contour0x, contour0y,
                                          new_contourx, new_contoury, new, mask):
        H, W = img1.shape
        contour0y = contour0y * H_new / H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new = points_new
        # resize the patch has countour to the target contour size
        # original_patch  =  img1[:,contour0x[0]:contour0x[points-1]]
        # original_patch  =  cv2.resize(original_patch, (points_new,H_new), interpolation=cv2.INTER_AREA)
        # contour0y=signal.resample(contour0y, points_new)
        img1 = cv2.resize(img1, (W, H_new), interpolation=cv2.INTER_AREA)
        line_it = 0
        i = 0
        # contourH = contour0y - contoury_s
        # maxh = np.min(contourH)
        contoury_s = contoury_s.astype(int)
        while (1):

            if contour0y[line_it] <= (H_new - 10):
                # TODO: remove ambiguity of i and line_it.
                #  If the line has been passed because there is no contour, we should not fill it here!
                # source_line = img1[:,contour0x[line_it]]
                # newy   = int(new_contoury[i] )
                # iniy   =  int (contour0y[line_it]) - 3   # add 5 to give more high light bondaries
                # shift  =  int(newy - iniy)
                #
                # # this is  the key difference of the connection
                # if shift > 0:
                #     start = contoury_s[i] + shift
                #     new[start:H_new,new_contourx[i]] = source_line[contoury_s[i]:H_new-shift]
                #     mask[start:H_new,new_contourx[i]] = 0
                # else :
                #     new[contoury_s[i]:H_new+shift,new_contourx[i]] = source_line[contoury_s[i]-shift:H_new]
                #     mask[contoury_s[i]:H_new+shift,new_contourx[i]]  = 0

                # New code! # TODO: CREATE NEW FUNCTION INSTEAD OF CHANGING HIS CODE!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                source_line = img1[:, contour0x[line_it]]
                newy = int(new_contoury[line_it])
                iniy = int(contour0y[line_it]) - 3  # add 5 to give more high light bondaries
                shift = int(newy - iniy)

                # this is  the key difference of the connection
                if shift > 0:
                    start = contoury_s[line_it] + shift
                    new[start:H_new, new_contourx[line_it]] = source_line[contoury_s[line_it]:H_new - shift]
                    mask[start:H_new, new_contourx[line_it]] = 0
                else:
                    new[contoury_s[line_it]:H_new + shift, new_contourx[line_it]] = source_line[
                                                                                    contoury_s[line_it] - shift:H_new]
                    mask[contoury_s[line_it]:H_new + shift, new_contourx[line_it]] = 0
                # i+=1
                # if i> (points_new-1):
                #     break
            line_it += 1
            if line_it > (points - 1):
                # line_it =0
                break

        return new, mask

    # fill_patch_base_origin2 adapted to multiple contours to handle IVUS images
    @staticmethod
    def fill_patch_base_multi(img1, H, sharp, contoury_s, contour0x, contour0y,
                              new_contourx, new_contoury, new, mask):

        points = len(contour0x)

        # if contoury_s is not just zeros and use soft edge is true (double condition as a precaution)
        if np.any(contoury_s) and not sharp:
            contoury_s = contoury_s.astype(int)

        for i in range(points - 1):

            if contour0y[i] <= (0.98 * H):

                source_line = img1[:, contour0x[i]]
                newy = int(new_contoury[i])
                iniy = int(contour0y[i]) - 3  # - 3 to give more highlight to the boundary
                shift = int(newy - iniy)

                if sharp:  # Sharp edge
                    if shift > 0:
                        new[newy:H, new_contourx[i]] = source_line[iniy:H - shift]
                        mask[newy:H, new_contourx[i]] = 0
                    else:
                        new[newy:H + shift, new_contourx[i]] = source_line[iniy:H]
                        mask[newy:H + shift, new_contourx[i]] = 0

                        # fill the bottom section with 0s
                        new[H + shift:H, new_contourx[i]] = 0
                        mask[H + shift:H, new_contourx[i]] = 0

                else:  # Soft edge
                    # this is the key difference to handle the different edges
                    if shift > 0:
                        start = contoury_s[i] + shift
                        new[start:H, new_contourx[i]] = source_line[contoury_s[i]:H - shift]
                        mask[start:H, new_contourx[i]] = 0
                    else:
                        new[contoury_s[i]:H + shift, new_contourx[i]] = source_line[contoury_s[i] - shift:H]
                        mask[contoury_s[i]:H + shift, new_contourx[i]] = 0

                        # fill the bottom section with 0s
                        new[H + shift:H, new_contourx[i]] = 0
                        mask[H + shift:H, new_contourx[i]] = 0

        return new, mask

    # this one will not regard the images contour pision, just take A-line is long enough
    def fill_patch_base_origin2(img1, H_new, contour0x, contour0y,
                                new_contourx, new_contoury, new, mask):
        H, W = img1.shape
        contour0y = contour0y * H_new / H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new = points_new
        # resize the patch has countour to the target contour size
        # original_patch  =  img1[:,contour0x[0]:contour0x[points-1]]
        # original_patch  =  cv2.resize(original_patch, (points_new,H_new), interpolation=cv2.INTER_AREA)
        # contour0y=signal.resample(contour0y, points_new)
        img1 = cv2.resize(img1, (W, H_new), interpolation=cv2.INTER_AREA)
        line_it = 0
        i = 0
        # contourH = H_new - contour0y
        # maxh = np.max(contourH)
        while 1:

            if contour0y[line_it] <= (H_new - 10):
                source_line = img1[:, contour0x[line_it].astype(int)]
                newy = int(new_contoury[i])
                iniy = int(contour0y[line_it]) - 3  # add 5 to give more high light boundaries
                shift = int(newy - iniy)
                if shift > 0:
                    new[newy:H_new, new_contourx[i].astype(int)] = source_line[iniy:H_new - shift]
                    mask[newy:H_new, new_contourx[i].astype(int)] = 0
                else:
                    new[newy:H_new + shift, new_contourx[i].astype(int)] = source_line[iniy:H_new]
                    mask[newy:H_new + shift, new_contourx[i].astype(int)] = 0

                i += 1
                if i > (points_new - 1):
                    break
            line_it += 1
            if line_it > (points - 1):
                line_it = 0

        return new, mask

    # deal with non full connected path, transfer these blank area
    def fill_patch_base_origin(img1, H_new, contour0x, contour0y,
                               new_contourx, new_contoury, new, mask):
        H, W = img1.shape
        contour0y = contour0y * H_new / H
        points = len(contour0x)
        points_new = len(new_contoury)
        W_new = points_new
        # resize the patch has countour to the target contour size
        original_patch = img1[:, contour0x[0]:contour0x[points - 1]]
        original_patch = cv2.resize(original_patch, (points_new, H_new), interpolation=cv2.INTER_AREA)
        contour0y = signal.resample(contour0y, points_new)
        img1 = cv2.resize(img1, (W, H_new), interpolation=cv2.INTER_AREA)

        # new  = np.zeros((H_new,W_new))
        for i in range(points_new):
            # line_it = int( np.random.random_sample()*points)
            # line_it = np.clip(line_it,0,points-1)
            line_it = i
            source_line = original_patch[:, line_it]
            # new[:,i] = ba.warp_padding_line1(source_line, contour0y[line_it],new_contoury[i])
            # new[:,i] = Basic_Operator .warp_padding_line2(source_line, contour0y[i],new_contoury[i])
            # random select a source
            newy = int(new_contoury[i])
            iniy = int(contour0y[line_it]) - 3  # add 5 to give more high light bondaries
            shift = int(newy - iniy)
            if shift > 0:
                new[newy:H_new, new_contourx[i]] = source_line[iniy:H_new - shift]
                mask[newy:H_new, new_contourx[i]] = 0
            else:
                new[newy:H_new + shift, new_contourx[i]] = source_line[iniy:H_new]
                mask[newy:H_new + shift, new_contourx[i]] = 0

        return new, mask

    # deal with non full connected path, transfer these blank area
    def re_fresh_path(px, py, H, W):
        # this function input the original coordinates of contour x and y, orginal image size and out put size

        if len(px) > 0.96 * W:  # first consider the special condition of full and gapin middel
            # rememver to add resacle later
            new_y = signal.resample(py, W)
            new_x = np.arange(0, W)
            return new_x, new_y

            ##np.roll(y, int(np.random.random_sample()*len(y)-1)) 
            # newy  = newy +  np.random.random_sample() *H/2

        clen = len(px)
        # img_piece = this_gray[:,this_pathx[0]:this_pathx[clen-1]]
        # no crop blank version
        # factor=W2/W

        this_pathy = py
        # resample

        # first determine the lef piece
        pathl = np.zeros(int(px[0])) + H - 1
        len1 = len(this_pathy)
        len2 = len(pathl)
        pathr = np.zeros(W - len1 - len2) + H - 1
        path_piece = np.append(pathl, this_pathy, axis=0)
        path_piece = np.append(path_piece, pathr, axis=0)
        new_y = signal.resample(path_piece, W)

        # in the down sample pathy function the dot with no label will be add a number of Height,
        # however because the label software can not label the leftmost and the rightmost points,
        # so it will be given a max value,  I crop the edge of the label, remember to crop the image correspondingly .
        new_x = np.arange(0, W)
        # path_piece = signal.resample(path_piece[3:W2-3], W2)
        return new_x, new_y
