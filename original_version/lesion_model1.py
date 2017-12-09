import tensorflow as tf
# import tensorflow.contrib.layers as tf_layers
slim = tf.contrib.slim
import numpy as np

class LesionNet(object):

    def __init__(self, sess):
        self.sess = sess
        self.end_points = dict()
        self.data = None
        self.ground_truth = None

    def define_graph(self, short_cut=False):

        def _collect_end_points(name, tensor):
            self.end_points[name] = tensor

        def count_mse(image,resize, endpoint):
            ground_truth = tf.image.resize_area(image, [resize, resize])

            return tf.reduce_mean(tf.square(ground_truth - self.end_points[endpoint]))

        with tf.variable_scope(name_or_scope='Lesion'):
            with tf.name_scope('Data'):
                self.data = tf.placeholder(shape=[None, 1024, 1024, 3], dtype=tf.float32)
                self.ground_truth = tf.placeholder(shape=[None, 1024, 1024, 1], dtype=tf.float32)
                # self.label = tf.placeholder(shape=[None, 2], dtype=tf.float32)

            with tf.variable_scope(name_or_scope='Encoder'):
                # 360 x 360 x 32
                conv1 = slim.conv2d(
                    inputs=self.data,
                    num_outputs=32,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv1'
                )
                conv1_relu = tf.nn.relu(conv1, name='conv1_relu')
                _collect_end_points('conv1', conv1)
                _collect_end_points('conv1_relu', conv1_relu)

                # 180 x 180 x 64
                conv2 = slim.conv2d(
                    inputs=conv1_relu,
                    num_outputs=64,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv2'
                )
                conv2_relu = tf.nn.relu(conv2, name='conv2_relu')
                _collect_end_points('conv2', conv2)
                _collect_end_points('conv2_relu', conv2_relu)

                # 90 x 90 x 128
                conv3 = slim.conv2d(
                    inputs=conv2_relu,
                    num_outputs=128,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv3'
                )
                conv3_relu = tf.nn.relu(conv3, name='conv3_relu')
                _collect_end_points('conv3', conv3)
                _collect_end_points('conv3_relu', conv3_relu)

                # 45 x 45 x 256
                conv4 = slim.conv2d(
                    inputs=conv3_relu,
                    num_outputs=256,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv4'
                )
                conv4_relu = tf.nn.relu(conv4, name='conv4_relu')
                _collect_end_points('conv4', conv4)
                _collect_end_points('conv4_relu', conv4_relu)
                # 22 x 22 x 512
                conv5 = slim.conv2d(
                    inputs=conv4_relu,
                    num_outputs=512,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv5'
                )
                conv5_relu = tf.nn.relu(conv5, name='conv5_relu')
                _collect_end_points('conv5', conv5)
                _collect_end_points('conv5_relu', conv5_relu)

                # 11 x 11 x 1024
                conv6 = slim.conv2d(
                    inputs=conv5_relu,
                    num_outputs=1024,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv6'
                )
                conv6_relu = tf.nn.relu(conv6, name='conv6_relu')
                _collect_end_points('conv6', conv6)
                _collect_end_points('conv6_relu', conv6_relu)

                conv7 = slim.conv2d(
                    inputs=conv6_relu,
                    num_outputs=1024,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv7'
                )
                conv7_relu = tf.nn.relu(conv7, name='conv7_relu')
                _collect_end_points('conv7', conv7)
                _collect_end_points('conv7_relu', conv7_relu)
                # print(conv7)

                conv7_pool = tf.nn.max_pool(value=conv7_relu,
                                            strides=[1, 2, 2, 1],
                                            ksize=[1, 2, 2, 1],
                                            padding='SAME',
                                            name='conv7_pool',
                                            )
                _collect_end_points('conv7_pool', conv7_pool)
                # print(conv7_pool)


                conv8 = slim.conv2d(
                    inputs=conv7_pool,
                    num_outputs=2048,
                    kernel_size=3,
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    scope='conv8'
                )
                conv8_relu = tf.nn.relu(conv8, name='conv8_relu')
                _collect_end_points('conv8', conv8)
                _collect_end_points('conv8_relu', conv8_relu)
                # print(conv8_relu)

                conv8_pool = tf.nn.avg_pool(
                    value=conv8,
                    ksize=[1, 4, 4, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name='conv8_pool'
                )
                _collect_end_points('conv8_pool', conv8_pool)
                # print(conv8_pool)

                net = slim.conv2d(conv8_pool, 4096, [1, 1], scope='fc7')
                # print(net)
                net = slim.dropout(
                    net, is_training=True, scope='dropout7')
                net = slim.conv2d(
                    net,
                    2, [1, 1],
                    activation_fn=None,
                    normalizer_fn=None,
                    scope='fc8')
                net = tf.squeeze(net, [1, 2])
                # print(net)
                # 11 x 11 x 1024
                encoder = slim.conv2d(
                    inputs=conv6_relu,
                    num_outputs=1024,
                    kernel_size=3,
                    stride=1,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='encoder_final'
                )
                _collect_end_points('encoder_final', encoder)

            with tf.variable_scope(name_or_scope='Decoder'):
                # 22 x 22 x 512
                deconv1 = slim.conv2d_transpose(
                    inputs=encoder,
                    num_outputs=512,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv1'
                )
                if short_cut:
                    concat1 = tf.concat([deconv1, conv5], axis=-1)
                    deconv1_relu = tf.nn.relu(concat1, name='deconv1_relu')
                else:
                    deconv1_relu = tf.nn.relu(deconv1, name='deconv1_relu')

                deconv1_relu_1 = tf.nn.relu(deconv1, name='deconv1_relu_1')
                deconv1_output = slim.conv2d(
                    inputs=deconv1_relu_1,
                    num_outputs=1,
                    kernel_size=1,
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv1_output'
                )
                deconv1_output=tf.clip_by_value(deconv1_output, 0, 255)


                _collect_end_points('deconv1', deconv1)
                _collect_end_points('deconv1_relu', deconv1_relu)
                _collect_end_points('deconv1_output', deconv1_output)

                # 44 x 44 x 256
                deconv2 = slim.conv2d_transpose(
                    inputs=deconv1_relu,
                    num_outputs=256,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv2'
                )
                if short_cut:
                    concat2 = tf.concat([deconv2, conv4], axis=-1)
                    deconv2_relu = tf.nn.relu(concat2, name='deconv2_relu')
                else:
                    deconv2_relu = tf.nn.relu(deconv2, name='deconv2_relu')

                deconv2_relu_1 = tf.nn.relu(deconv2, name='deconv1_relu_1')
                deconv2_output = slim.conv2d(
                    inputs=deconv1_relu_1,
                    num_outputs=1,
                    kernel_size=1,
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv2_output'
                )
                deconv2_output = tf.clip_by_value(deconv2_output, 0, 255)
                _collect_end_points('deconv2', deconv2)
                _collect_end_points('deconv2_relu', deconv2_relu)
                _collect_end_points('deconv2_output', deconv2_output)

                # 128 x 128 x 128
                deconv3 = slim.conv2d_transpose(
                    inputs=deconv2_relu,
                    num_outputs=128,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv3'
                )
                if short_cut:
                    concat3 = tf.concat([deconv3, conv3], axis=-1)
                    deconv3_relu = tf.nn.relu(concat3, name='deconv3_relu')
                else:
                    deconv3_relu = tf.nn.relu(deconv3, name='deconv3_relu')

                deconv3_relu_1 = tf.nn.relu(deconv3, name='deconv1_relu_1')
                deconv3_output = slim.conv2d(
                    inputs=deconv3_relu_1,
                    num_outputs=1,
                    kernel_size=1,
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv3_output'
                )
                deconv3_output = tf.clip_by_value(deconv3_output, 0, 255)
                _collect_end_points('deconv3', deconv3)
                _collect_end_points('deconv3_relu', deconv3_relu)
                _collect_end_points('deconv3_output', deconv3_output)

                # 256 x 256 x 64
                deconv4 = slim.conv2d_transpose(
                    inputs=deconv3_relu,
                    num_outputs=64,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv4'
                )
                if short_cut:
                    concat4 = tf.concat([deconv4, conv2], axis=-1)
                    deconv4_relu = tf.nn.relu(concat4, name='deconv4_relu')
                else:
                    deconv4_relu = tf.nn.relu(deconv4, name='deconv4_relu')

                deconv4_relu_1 = tf.nn.relu(deconv4, name='deconv1_relu_1')
                deconv4_output = slim.conv2d(
                    inputs=deconv4_relu_1,
                    num_outputs=1,
                    kernel_size=1,
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv4_output'
                )
                deconv4_output = tf.clip_by_value(deconv4_output, 0, 255)
                _collect_end_points('deconv4', deconv4)
                _collect_end_points('deconv4_relu', deconv4_relu)
                _collect_end_points('deconv4_output', deconv4_output)

                # 512 x 512 x 32
                deconv5 = slim.conv2d_transpose(
                    inputs=deconv4_relu,
                    num_outputs=32,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv5'
                )
                if short_cut:
                    concat5 = tf.concat([deconv5, conv1], axis=-1)
                    deconv5_relu = tf.nn.relu(concat5, name='deconv5_relu')
                else:
                    deconv5_relu = tf.nn.relu(deconv5, name='deconv5_relu')

                deconv5_relu_1 = tf.nn.relu(deconv5, name='deconv5_relu_1')
                deconv5_output = slim.conv2d(
                    inputs=deconv5_relu_1,
                    num_outputs=1,
                    kernel_size=1,
                    stride=1,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv5_output'
                )
                deconv5_output = tf.clip_by_value(deconv5_output, 0, 255)
                _collect_end_points('deconv5', deconv5)
                _collect_end_points('deconv5_relu', deconv5_relu)
                _collect_end_points('deconv5_output', deconv5_output)

                # 1024 x 1024 x 4
                lesion_map = slim.conv2d_transpose(
                    inputs=deconv5_relu,
                    num_outputs=1,
                    kernel_size=3,
                    stride=2,
                    padding='SAME',
                    activation_fn=tf.nn.relu,
                    scope='lesion_map'
                )
                lesion_map = tf.clip_by_value(lesion_map, 0.0, 255.0)
                _collect_end_points('lesion_map', lesion_map)

        with tf.name_scope('LesionTrain'):
            # self.mse_1 = tf.reduce_mean(tf.square(tf.image.resize_area(self.ground_truth, [32, 32]) - self.end_points['deconv1_output']))
            self.mse_1 = count_mse(self.ground_truth, 32, 'deconv1_output')
            print(self.mse_1)

            self.mse_2 = count_mse(self.ground_truth, 64, 'deconv2_out_put')
            self.mse_3 = count_mse(self.ground_truth, 128, 'deconv3_out_put')
            self.mse_4 = count_mse(self.ground_truth, 256, 'deconv4_out_put')
            self.mse_5 = count_mse(self.ground_truth, 512, 'deconv5_out_put')
            self.mse_lesion = tf.reduce_mean(tf.square(lesion_map - self.ground_truth))
            self.mse = self.mse_1+self.mse_2+self.mse_3+self.mse_4+self.mse_5+self.mse_lesion
            regularization_losses = tf.losses.get_regularization_losses('Lesion')
            self.global_loss = tf.add_n([self.mse] + regularization_losses, 'global_loss')

            # only trainable variables
            self.train_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lesion')
            # for train_var in self.train_vars:
            #     print(train_var)
            self.global_step = tf.Variable(0, trainable=False)

            # 31,000 - 49,500
            learning_rate = 0.0001
            optimizer = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            self.train_op = optimizer.minimize(var_list=self.train_vars,
                                               loss=self.mse, global_step=self.global_step)

            # add summaries
            tf.summary.scalar('learning_rate', learning_rate)
            tf.summary.scalar('mse', self.mse)
            tf.summary.scalar('loss', self.global_loss)

        # merge all summaries and write to file
        self.train_summaries_merge = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('Summaries', graph=self.sess.graph)

    def train_step(self, batch_data, batch_labels):
        feed_dict = {
            self.data: batch_data,
            self.ground_truth: batch_labels
        }
        _, loss, global_loss, global_step, summaries = \
            self.sess.run([self.train_op, self.mse, self.global_loss,
                           self.global_step, self.train_summaries_merge], feed_dict=feed_dict)
        ##
        # User output
        ##
        if global_step % 1 == 0:
            print('Iteration = {}, loss = {:.6f}, global loss = {:.6f}'.format(global_step, loss, global_loss))
        if global_step % 250 == 0:
            self.summary_writer.add_summary(summaries, global_step)
            print('Summaries saved at step = {}.'.format(global_step))
        return global_step

    def inference(self, batch_data):
        feed_dict = {
            self.data: batch_data
        }

        lesion_map = self.sess.run(self.end_points['lesion_map'], feed_dict=feed_dict)

        return lesion_map
