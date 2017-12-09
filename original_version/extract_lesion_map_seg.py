from __future__ import print_function, division, absolute_import
from lesion_model1 import LesionNet
from utils_3 import preprocessed_orginal_image
import tensorflow as tf
import os
import numpy as np
import scipy.misc as misc



os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model_load_path = 'Models/model.ckpt-7500'

# source_folder = '/home/liuwen/ssd/Messidor/original/validate_crop_scale_enhance_rotate_1'
# des_folder = '/home/liuwen/ssd/Messidor/original/lesion_maps/validate'

source_folder = "E:\ISBI_code\Source_Data\\650\\650image"
test_file_path = "test_filename.txt"
des_folder = 'data/lesions'

input_data = np.zeros((1, 1024, 1024, 3), dtype=np.float32)

with tf.Session(config=config) as sess:
    print('Init models...')
    lesion_net = LesionNet(sess)

    print('Define graphs...')
    lesion_net.define_graph(short_cut=True)

    print('Init variables...')
    saver = tf.train.Saver(max_to_keep=None)
    sess.run(tf.global_variables_initializer())

    # if load path specified, load a saved model
    if model_load_path:
        saver.restore(sess, model_load_path)
        print('Model restored from ' + model_load_path)

    print('Init successfully!')

    # Extract lesion map
    f = open(test_file_path)
    all_test_files = f.readlines()

    for image_name in all_test_files:
        source_image_path = os.path.join(source_folder, image_name.split('.')[0] + '.jpg')
        # des_image_path = os.path.join(des_folder, image_name.split('.')[0] + '.png')
        des_image_path = os.path.join(des_folder, image_name.split('.')[0] + '.jpg')

        input_data[0, ...] = preprocessed_orginal_image(source_image_path)
        lesion_map = lesion_net.inference(input_data)
        # lesion_map = np.squeeze(lesion_map)
        lesion_map = np.reshape(lesion_map, (1024, 1024))

        # cv2.imshow('lesion_map', lesion_map / 255)
        # cv2.waitKey(50)
        print(des_image_path, lesion_map.shape)
        misc.imsave(des_image_path, lesion_map)



