from PIL import Image
import numpy as np
import os
import scipy.misc as misc
import scipy.io as scio
import cv2
import matplotlib.pyplot as plt

disc_threshold = 40
cup_threshold = 250

origin_image_folder = "E:\ISBI_code\Source_Data\\650\\650image"
lesion_folder = "data\lesions"
different_threshold_folder = "data\different_threshold"
ground_truth_folder = "E:\ISBI_code\Source_Data\\650\\650mask"
show_images_folder = "data\show_images"

all_test_files = os.listdir(lesion_folder)


def read_mat_file(image_path):
    data = scio.loadmat(image_path)
    # print(type(data))
    # print(data['maskFull'])
    # print(data.keys())

    original_data = data['maskFull']
    resized_data = misc.imresize(original_data, (1024, 1024))

    return resized_data


def extract_disc_contour(image):
    ret, binary = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours


def extract_cup_contour(image):
    ret, binary = cv2.threshold(image, 1.5, 1, cv2.THRESH_BINARY)
    binary = binary.astype(np.uint8)
    _, contours, hierarchy = cv2.findContours(binary, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)
    return contours


total_m1 = 0
total_m2 = 0
count = 130

for test_image in all_test_files[:count]:
    image_path = os.path.join(lesion_folder, test_image)
    ground_truth_path = os.path.join(ground_truth_folder, test_image.split('.')[0] + '.mat')
    ground_truth = read_mat_file(ground_truth_path)
    ground_truth_image = np.array(ground_truth)
    origin_image_path = os.path.join(origin_image_folder, test_image)
    origin_image = Image.open(origin_image_path)
    origin_image = misc.imresize(origin_image, (1024, 1024))


    im = Image.open(image_path)
    image = np.array(im)

    assert np.shape(image) == np.shape(ground_truth_image), "The shape of image is not equality to the shape of gt"
    base = np.zeros(shape=np.shape(image))
    # base[image>disc_threshold] = 1
    base[image>cup_threshold] =2
    misc.imsave(os.path.join(different_threshold_folder, test_image), base)


    # draw the contours based the prediction
    lesion_disc_contour = extract_disc_contour(base.astype(np.uint8))
    lesion_cup_contour = extract_cup_contour(base.astype(np.uint8))
    disc_labeled_image = cv2.drawContours(origin_image, lesion_disc_contour, -1, (0, 255, 0), 1)
    labeled_image = cv2.drawContours(disc_labeled_image, lesion_cup_contour, -1, (0, 255, 0), 1)
    # plt.imshow(labeled_image)
    # plt.show()
    misc.imsave(os.path.join(show_images_folder, test_image), labeled_image)

    # draw the contours based the ground truth
    lesion_disc_contour = extract_disc_contour(ground_truth_image.astype(np.uint8))
    lesion_cup_contour = extract_cup_contour(ground_truth_image.astype(np.uint8))
    disc_labeled_image = cv2.drawContours(origin_image, lesion_disc_contour, -1, (255, 0, 255), 1)
    labeled_image = cv2.drawContours(disc_labeled_image, lesion_cup_contour, -1, (255, 0, 255), 1)
    # plt.imshow(labeled_image)
    # plt.show()
    misc.imsave(os.path.join(show_images_folder, test_image), labeled_image)

    gt_image = np.zeros(shape=np.shape(image))
    gt_image[ground_truth_image>1] = 1
    print(np.max(gt_image), np.min(gt_image))
    over_lap = gt_image + base
    # the Union set
    image_union = np.zeros(shape=np.shape(image))
    image_union[over_lap>0] = 1

    # the intersection set
    image_inter = np.zeros(shape=np.shape(image))
    image_inter[over_lap>1] = 1

    # the number of union set and intersection set
    number_union = np.sum(image_union)
    number_inter = np.sum(image_inter)

    # the number of image and ground_truth_image
    number_test = np.sum(base)
    print(number_test)
    number_gt = np.sum(gt_image)
    print(number_gt)

    m_1 = (1 - number_inter/number_union)
    m_2 = (np.abs(number_test - number_gt))/number_gt



    total_m1 += m_1
    total_m2 += m_2

print(total_m1/count, total_m2/count)



    # print(base)
    # print(np.count_nonzero(base))
    # print(np.max(base))






