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
        # print('train = {}'.format(line))
        train_files.append(line)

with open('testset.txt', 'r') as file:
    for line in file.readlines():
        line = line.rstrip()
        # print('test = {}'.format(line))
        test_files.append(file)

assert len(train_files) + len(test_files) == 89, 'the number of total set is not 89'

total_images = os.listdir(image_folder)
total_length = len(total_images)
total_index = 0
epoch = 0


def get_images_labels(batch_size):
    global total_index, total_length, total_images, epoch
    # sample the (image path, label)
    image_paths = []

    start = total_index
    end = total_index + batch_size
    for i in range(start, end):
        if i == total_length:
            np.random.shuffle(total_images)
            epoch += 1

        idx = i % total_length
        image_name = total_images[idx]
        image_path = os.path.join(image_folder, image_name)
        hardexudates_path = os.path.join(hardexudates, image_name)
        hemorrhages_path = os.path.join(hemorrhages, image_name)
        redsmalldots_path = os.path.join(redsmalldots, image_name)
        softexudates_path = os.path.join(softexudates, image_name)
        image_paths.append((image_path, hardexudates_path, hemorrhages_path, redsmalldots_path, softexudates_path))

    total_index = end % total_length

    # for image_path, hardexudates_path, hemorrhages_path, redsmalldots_path, softexudates_path in image_paths:
    #     print(image_path)
    #     print(hardexudates_path)
    #     print(hemorrhages_path)
    #     print(redsmalldots_path)
    #     print(softexudates_path)
    # print('{} / {}'.format(total_index, total_length))
    return image_paths


def normalize_frames(frames):
    """
    Convert frames from int8 [0, 255] to float32 [-1, 1].

    @param frames: A numpy array. The frames to be converted.

    @return: The normalized frames.
    """
    new_frames = frames.astype(np.float32)
    # new_frames /= (255 / 2)
    # new_frames -= 1
    new_frames -= 128.0

    return new_frames


def denormalize_frames(frames):
    """
    Performs the inverse operation of normalize_frames.

    @param frames: A numpy array. The frames to be converted.

    @return: The denormalized frames.
    """
    # new_frames = frames + 1
    # new_frames *= (255 / 2)
    # noinspection PyUnresolvedReferences
    new_frames = frames + 128
    new_frames = new_frames.astype(np.uint8)

    return new_frames


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


def load_batch(batch_size=1):
    global epoch, flip_rotate_args

    # _flip = (epoch % 2 == 0)
    _flip = False

    data = np.empty(shape=[batch_size, 1024, 1024, 3], dtype=np.float32)
    labels = np.empty(shape=[batch_size, 1024, 1024, 1], dtype=np.float32)

    image_paths = get_images_labels(batch_size)

    for idx, image_path_tuple in enumerate(image_paths):
        # read image
        image_path = image_path_tuple[0]
        data[idx] = pre_processing(image_path, is_color=True, flip=_flip, rotate=0)

        # read lesion maps
        lesion_map = []
        for lesion_path in image_path_tuple[1:]:
            lesion_map.append(pre_processing(lesion_path, is_color=False, flip=_flip, rotate=0))

        # 1024 x 1024 x 4
        # labels[idx] = np.stack(lesion_map, axis=2)
        lesion_map = np.stack(lesion_map, axis=2)
        labels[idx] = np.max(lesion_map, axis=2, keepdims=True)
        print(image_path)
        cv2.imwrite(os.path.join('data/gt', os.path.split(image_path)[-1]), labels[idx])

    # data = normalize_frames(data)
    # labels = normalize_frames(labels)
    return data, labels


if __name__ == '__main__':
    # for i in range(89):
    #     data, labels = load_batch(1)
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
