import cv2
import numpy as np
import os



image_folder = 'diaretdb1_v_1_1/resources/images/process_data1'
ground_truth_folder = 'diaretdb1_v_1_1/resources/images/ddb1_groundtruth'
hardexudates = os.path.join(ground_truth_folder, 'hardexudates')
hemorrhages = os.path.join(ground_truth_folder, 'hemorrhages')
redsmalldots = os.path.join(ground_truth_folder, 'redsmalldots')
softexudates = os.path.join(ground_truth_folder, 'softexudates')

train_files = []
test_files = []
with open('trainset.txt', 'r') as file:
    for line in file.readlines():
        line = line.rstrip()
        print('train = {}'.format(line))
        train_files.append(line)

with open('testset.txt', 'r') as file:
    for line in file.readlines():
        line = line.rstrip()
        print('test = {}'.format(line))
        test_files.append(file)

assert len(train_files) + len(test_files) == 89, 'the number of total set is not 89'

total_train_images = train_files
total_train_length = len(total_train_images)
total_train_index = 0
train_epoch = 0

total_test_images = train_files
total_test_length = len(total_test_images)
total_test_index = 0
test_epoch = 0


def get_images_labels(batch_size):
    global total_train_index, total_train_length, total_train_images, train_epoch
    # sample the (image path, label)
    image_paths = []

    start = total_train_index
    end = total_train_index + batch_size
    for i in range(start, end):
        if i == total_train_length:
            np.random.shuffle(total_train_images)
            train_epoch += 1

        idx = i % total_train_length
        image_name = total_train_images[idx]
        image_path = os.path.join(image_folder, image_name)
        hardexudates_path = os.path.join(hardexudates, image_name)
        hemorrhages_path = os.path.join(hemorrhages, image_name)
        redsmalldots_path = os.path.join(redsmalldots, image_name)
        softexudates_path = os.path.join(softexudates, image_name)
        image_paths.append((image_path, hardexudates_path, hemorrhages_path, redsmalldots_path, softexudates_path))

    total_train_index = end % total_train_length

    # for image_path, hardexudates_path, hemorrhages_path, redsmalldots_path, softexudates_path in image_paths:
    #     print(image_path)
    #     print(hardexudates_path)
    #     print(hemorrhages_path)
    #     print(redsmalldots_path)
    #     print(softexudates_path)
    # print('{} / {}'.format(total_index, total_length))
    return image_paths


def get_test_images_labels(batch_size):
    global total_test_index, total_test_length, total_test_images, test_epoch
    # sample the (image path, label)
    image_paths = []

    start = total_test_index
    end = total_test_index + batch_size
    for i in range(start, end):
        if i == total_test_length:
            np.random.shuffle(total_test_images)
            test_epoch += 1

        idx = i % total_test_length
        image_name = total_test_images[idx]
        image_path = os.path.join(image_folder, image_name)
        hardexudates_path = os.path.join(hardexudates, image_name)
        hemorrhages_path = os.path.join(hemorrhages, image_name)
        redsmalldots_path = os.path.join(redsmalldots, image_name)
        softexudates_path = os.path.join(softexudates, image_name)
        image_paths.append((image_path, hardexudates_path, hemorrhages_path, redsmalldots_path, softexudates_path))

    total_test_index = end % total_test_length

    # for image_path, hardexudates_path, hemorrhages_path, redsmalldots_path, softexudates_path in image_paths:
    #     print(image_path)
    #     print(hardexudates_path)
    #     print(hemorrhages_path)
    #     print(redsmalldots_path)
    #     print(softexudates_path)
    # print('{} / {}'.format(total_index, total_length))
    return image_paths


def pre_processing(image_path, is_color=True, flip=False, rotate=0):
    if is_color:
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    else:
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    image = cv2.resize(image, (1024, 1024))
    if flip:
        image = np.fliplr(image)
    # image = np.rot90(image, rotate)
    image = image.astype(dtype=np.float32)
    return image


def load_batch(batch_size=1, path_func=get_images_labels):
    global train_epoch

    _flip = (train_epoch % 2 == 0)

    data = np.empty(shape=[batch_size, 1024, 1024, 3], dtype=np.float32)
    labels = np.empty(shape=[batch_size, 1024, 1024, 1], dtype=np.float32)

    image_paths = path_func(batch_size)

    for idx, image_path_tuple in enumerate(image_paths):
        # read image
        image_path = image_path_tuple[0]
        data[idx] = pre_processing(image_path, is_color=True, flip=_flip, rotate=0)

        # read lesion maps
        lesion_map = []
        for lesion_path in image_path_tuple[1:]:
            lesion_map.append(pre_processing(lesion_path, is_color=False, flip=_flip, rotate=0))

        # 1024 x 1024 x 4
        lesion_map = np.stack(lesion_map, axis=2)
        labels[idx] = np.max(lesion_map, axis=2, keepdims=True)

    return data, labels


if __name__ == '__main__':
    data, labels = load_batch(10)
    for i in range(89):
        data, labels = load_batch(10)