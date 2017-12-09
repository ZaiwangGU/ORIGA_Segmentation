import scipy.misc as misc
import scipy.io as scio
import numpy as np
import os
origin_data_folder = "E:\ISBI_code\Source_Data\\650\\650image"
ground_truth_folder = "E:\ISBI_code\Source_Data\\650\\650mask"
train_file_path = "train_filename.txt"
test_file_path = "test_filename.txt"
label_path = "E:\ISBI_code\Source_Data\labels.xlsl"

train_files = []
test_files = []
with open(train_file_path) as file:
    for line in file.readlines():
        root_name = line.split('.')[0]
        train_files.append(root_name)
with open(test_file_path) as file:
    for line in file.readlines():
        root_name = line.split('.')[0]
        test_files.append(root_name)
print(len(train_files))
print(len(test_files))
assert len(train_files) + len(test_files) == 650, "the number of total set is not 650"

total_train_images = train_files
total_train_length = len(total_train_images)
total_train_index = 0
train_epoch = 0

total_test_images = test_files
total_test_length = len(total_test_images)
total_test_index = 0
test_epoch = 0


def preprocessed_mat(image_path):
    data = scio.loadmat(image_path)
    # print(type(data))
    # print(data['maskFull'])
    # print(data.keys())
    original_data = data['maskFull']
    resized_data = misc.imresize(original_data, (1024, 1024))
    resized_data = resized_data / (np.max(resized_data) - np.min(resized_data)) * 255.
    return resized_data


def preprocessed_orginal_image(image_path, flip=False):
    original_data = misc.imread(image_path, mode='RGB')
    resized_data = misc.imresize(original_data, (1024, 1024))

    if flip:
        resized_data = np.fliplr(resized_data)
    # resized_data = resized_data / (np.max(resized_data) - np.min(resized_data)) * 255.

    # image = np.rot90(image, rotate)
    image = resized_data.astype(dtype=np.float32)
    return image

# all_files = []
# all_files_name = os.listdir(origin_data_path)
# print(all_files_name[:10])
#
# fp = open("E:\\battleNet10_1\\filename.txt", "w")
#
# for file_name in all_files_name:
#     fp.write(file_name+"\n")
# fp.close()


def get_images_labels(batch_size):
    global total_train_images, total_train_index, total_train_length, train_epoch
    image_paths = []

    start = total_train_index
    end = total_train_index + batch_size
    for i in range(start, end):
        if i == total_train_length:
            np.random.shuffle(total_train_images)
            train_epoch +=1

        idx = i % total_train_length
        image_name = total_train_images[idx]
        image_path = os.path.join(origin_data_folder, image_name+'.jpg')
        ground_truth_path = os.path.join(ground_truth_folder, image_name+'.mat')
        image_paths.append((image_path, ground_truth_path))

    total_train_index = end % total_train_length

    return image_paths


def get_test_images_labels(batch_size):
    global total_test_images, total_test_index, total_test_length, test_epoch
    image_paths = []

    start = total_test_index
    end = total_test_index + batch_size
    for i in range(start, end):
        if i == total_test_length:
            np.random.shuffle(total_test_images)
            test_epoch += 1

        idx = i % total_test_length
        image_name = total_test_images[idx]
        image_path = os.path.join(origin_data_folder, image_name + '.jpg')
        ground_truth_path = os.path.join(ground_truth_folder, image_name + '.mat')
        image_paths.append((image_path, ground_truth_path))

    total_test_index = end % total_test_length

    return image_paths

def load_batch(batch_size=1, path_func = get_images_labels):
    global train_epoch

    _flip = (train_epoch % 2 ==0)

    data = np.empty(shape=[batch_size, 1024, 1024, 3], dtype=np.float32)
    ground_truth = np.empty(shape=[batch_size, 1024, 1024, 1], dtype=np.float32)

    image_paths = path_func(batch_size)

    for idx, image_path_tuple in enumerate(image_paths):
        # read image
        image_path = image_path_tuple[0]
        data[idx] = preprocessed_orginal_image(image_path)

        # read ground truth
        ground_truth_path = image_path_tuple[1]
        ground_truth[idx] = np.reshape(preprocessed_mat(ground_truth_path), (1024, 1024, 1))
        #ground_truth[idx] = preprocessed_mat(ground_truth_path)

    return data, ground_truth



if __name__ == '__main__':
    data, ground_truth = load_batch(10)
    for i in range(650):
        data, ground_truth = load_batch(10)