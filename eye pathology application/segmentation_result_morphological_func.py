import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import math
from scipy.ndimage import measurements, morphology
from PIL import Image
from matplotlib import cm




# MAIN #
def segmentation_post_proccess(img,mask,target_dir,img_name):
    #Support functions
    def circular_filter(radius):
        radius = np.int(radius)
        kernel = np.zeros((2 * radius + 1, 2 * radius + 1), np.uint8)
        y, x = np.ogrid[-radius:radius + 1, -radius:radius + 1]
        mask = x ** 2 + y ** 2 <= radius ** 2
        kernel[mask] = 1
        return kernel


    def image_preprocess(img):
        img_pil = Image.fromarray(np.uint8(img*255)).resize((512, 512))
        img = np.resize(img, (512, 512, 3))
        return img_pil



    def mask_cropper(img, mask):
        # Crop image with 2D mask
        big_mask_inter = np.zeros((512, 512, 3))
        big_mask_inter[:, :, 0] = mask
        big_mask_inter[:, :, 1] = mask
        big_mask_inter[:, :, 2] = mask
        im_array = np.asarray(img)
        img_rgb = Image.fromarray(im_array.astype('uint8'), 'RGB')
        cropped_img_array = img_rgb * np.uint8(big_mask_inter)
        cropped_img = Image.fromarray(cropped_img_array.astype('uint8'), 'RGB')
        return cropped_img

    def morphological_algorithm(mask, image, flag='segmentation succeed'):
        ret, thresh_mask = cv2.threshold(mask, 127, 1, cv2.THRESH_BINARY)

        num_labels_orig, labels_im_orig = cv2.connectedComponents(thresh_mask)
        (unique, counts) = np.unique(labels_im_orig, return_counts=True)

        frequencies = np.asarray((unique[1:], counts[1:])).T  # Return [unique1,numbers;unique2,numbers..]
        max_pixels = np.max(frequencies[:, 1], axis=0)
        label_radius = np.floor((((max_pixels) / 8) / (math.pi)) ** 0.5)
        kernel_8 = circular_filter(label_radius)
        opening = cv2.morphologyEx(thresh_mask, cv2.MORPH_OPEN, kernel_8)
        largest_label = np.argmax(frequencies[:, 1], axis=0) + 1
        largest_component = labels_im_orig == largest_label
        x = opening + largest_component
        x1 = x == 1
        x2 = x == 2
        opening_and_component = np.uint8(x1 + x2)
        closing = cv2.morphologyEx(opening_and_component, cv2.MORPH_CLOSE, kernel_8)
        num_labels_clos, labels_im_clos = cv2.connectedComponents(closing)
        list_of_centers = []
        for i in range(1, num_labels_clos):
            center_i = measurements.center_of_mass(labels_im_clos == i)
            list_of_centers.append([np.floor(center_i[1]), np.floor(center_i[0])])
        euclidean_pwr = (list_of_centers - np.array([256, 256])) ** 2
        euclidean_dist = (euclidean_pwr[:, 0] + euclidean_pwr[:, 1]) ** 0.5
        most_centered_label = np.argmin(euclidean_dist) + 1
        binary_mask = labels_im_clos == most_centered_label
        binary_mask_filled = morphology.binary_fill_holes(binary_mask, structure=None, output=None, origin=0)

        kernel = np.ones((3, 3), np.uint8)
        erosion = cv2.erode(binary_mask_filled.astype('uint8'), kernel, iterations=1)
        edges = binary_mask_filled - erosion
        cropped_image = mask_cropper(image, binary_mask_filled)
        num_labels_edges, labels_edges = cv2.connectedComponents(edges)
        if num_labels_edges > 2:
            flag = 'segmentation failed'
        return binary_mask, cropped_image, edges, flag




    #Main function

    image_pil = image_preprocess(img)
    binary_mask, cropped_image, edges, flag = morphological_algorithm(mask, image_pil)
    Cropped_img_loc =os.path.join(target_dir, 'Cropped_Img')
    if not os.path.isdir(Cropped_img_loc):
        os.mkdir(Cropped_img_loc)
    cropped_image.save(os.path.join(Cropped_img_loc, img_name))
    binary_mask = Image.fromarray(binary_mask)
    binary_mask.save(os.path.join(target_dir, 'binary_mask_' + img_name))
    binary_edges = Image.fromarray(edges*255)
    binary_edges.save(os.path.join(target_dir, 'binary_edges_' + img_name))
    return binary_mask, cropped_image, edges, flag
