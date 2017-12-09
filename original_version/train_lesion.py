import tensorflow as tf
import scipy.misc as misc
import os
import numpy as np
from utils_3 import load_batch, get_images_labels, get_test_images_labels
from lesion_model import LesionNet
import constants as c

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

model_load_path = ''
batch_size = 10
num_epochs = 50000

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

    global_step = 0
    for i in range(num_epochs):
        data, labels = load_batch(batch_size=batch_size, path_func=get_images_labels)
        global_step = lesion_net.train_step(batch_data=data, batch_labels=labels)

        if global_step % 2500 == 0:
            print('-' * 30)
            print('Saving models...')
            saver.save(sess, os.path.join('Models', 'model.ckpt'), global_step=global_step)
            print('Saved models!')
            print('-' * 30)
        if global_step % 500 == 0:
            print('Saving train images...')
            sources, ground_truths, lesion_maps = sess.run(
                [lesion_net.data, lesion_net.ground_truth, lesion_net.end_points['lesion_map']], feed_dict={
                    lesion_net.data: data,
                    lesion_net.ground_truth: labels
                }
            )

            print('sources = {} / {}'.format(sources.min(), sources.max()))
            print('lesion map = {} / {}'.format(lesion_maps.min(), lesion_maps.max()))

            for idx, (source, ground_truth, lesion_map) in enumerate(zip(sources, ground_truths, lesion_maps)):
                source = np.reshape(source, (1024, 1024, 3))
                ground_truth = np.reshape(ground_truth, (1024, 1024))
                lesion_map = np.reshape(lesion_map, (1024, 1024))
                '''
                misc.imsave(os.path.join(c.train_images_saver,
                                         'source_' + str(global_step) + '_' + str(idx) + '.jpg'), source)


                misc.imsave(os.path.join(c.train_images_saver,
                                         'label_' + str(global_step) + '_' + str(idx) + '.jpg'), ground_truth)
                '''
                misc.imsave(os.path.join(c.train_images_saver,
                                         'pred_' + str(global_step) + '_' + str(idx) + '.jpg'), lesion_map)
        if global_step % 2500 == 0:
            print('Saving test images...')
            data, labels = load_batch(batch_size=batch_size, path_func=get_test_images_labels)
            sources, ground_truths, lesion_maps = sess.run(
                [lesion_net.data, lesion_net.ground_truth, lesion_net.end_points['lesion_map']], feed_dict={
                    lesion_net.data: data,
                    lesion_net.ground_truth: labels
                }
            )

            print('sources = {} / {}'.format(sources.min(), sources.max()))
            print('lesion map = {} / {}'.format(lesion_maps.min(), lesion_maps.max()))

            for idx, (source, ground_truth, lesion_map) in enumerate(zip(sources, ground_truths, lesion_maps)):
                source = np.reshape(source, (1024, 1024, 3))
                ground_truth = np.reshape(ground_truth, (1024, 1024))
                lesion_map = np.reshape(lesion_map, (1024, 1024))
                misc.imsave(os.path.join(c.test_images_saver,
                                         'source_' + str(global_step) + '_' + str(idx) + '.jpg'), source)

                misc.imsave(os.path.join(c.test_images_saver,
                                         'label_' + str(global_step) + '_' + str(idx) + '.jpg'), ground_truth)
                misc.imsave(os.path.join(c.test_images_saver,
                                         'pred_' + str(global_step) + '_' + str(idx) + '.jpg'), lesion_map)



