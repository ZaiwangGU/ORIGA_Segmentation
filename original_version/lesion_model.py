import tensorflow as tf
# import tensorflow.contrib.layers as tf_layers
slim = tf.contrib.slim


class LesionNet(object):

    def __init__(self, sess):
        self.sess = sess
        self.end_points = dict()
        self.data = None
        self.ground_truth = None

    def define_graph(self, short_cut=False):

        def _collect_end_points(name, tensor):
            self.end_points[name] = tensor

        with tf.variable_scope(name_or_scope='Lesion'):
            with tf.name_scope('Data'):
                self.data = tf.placeholder(shape=[None, 1024, 1024, 3], dtype=tf.float32)
                self.ground_truth = tf.placeholder(shape=[None, 1024, 1024, 1], dtype=tf.float32)

            with tf.variable_scope(name_or_scope='Encoder'):
                # 512 x 512 x 32
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

                # 256 x 256 x 64
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

                # 128 x 128 x 128
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

                # 64 x 64 x 256
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
                # 32 x 32 x 512
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

                # 16 x 16 x 1024
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

                # 16 x 16 x 1024
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
                # 32 x 32 x 512
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
                _collect_end_points('deconv1', deconv1)
                _collect_end_points('deconv1_relu', deconv1_relu)

                # 64 x 64 x 256
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
                _collect_end_points('deconv2', deconv2)
                _collect_end_points('deconv2_relu', deconv2_relu)

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
                _collect_end_points('deconv3', deconv3)
                _collect_end_points('deconv3_relu', deconv3_relu)

                # 128 x 128 x 128

                deconv3_output_1 = slim.conv2d(
                    inputs=deconv3_relu,
                    num_outputs=64,
                    stride=1,
                    kernel_size=3,
                    activation_fn=None,
                    padding='SAME',
                    scope='deconv3_output_1'
                )
                deconv3_output_2 = slim.conv2d(
                    inputs=deconv3_output_1,
                    num_outputs=1,
                    stride=1,
                    kernel_size=3,
                    activation_fn=None,
                    padding='SAME',
                    scope='deconv3_output_2'
                )

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
                _collect_end_points('deconv4', deconv4)
                _collect_end_points('deconv4_relu', deconv4_relu)

                # 256 x256 x 64

                deconv4_output_1 = slim.conv2d(
                    inputs=deconv4_relu,
                    num_outputs=32,
                    stride=1,
                    kernel_size=3,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv4_output_1'
                )
                deconv4_output_2 = slim.conv2d(
                    inputs=deconv4_output_1,
                    num_outputs=1,
                    stride=1,
                    kernel_size=3,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv4_output_2'
                    )

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
                _collect_end_points('deconv5', deconv5)
                _collect_end_points('deconv5_relu', deconv5_relu)

                # 512 x 512 x 1
                deconv5_output = slim.conv2d(
                    inputs=deconv5_relu,
                    num_outputs=1,
                    stride=1,
                    kernel_size=3,
                    padding='SAME',
                    activation_fn=None,
                    scope='deconv5_output'
                )

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

            def get_train_names():
                names = ['Lesion/Decoder/deconv3_output_1/',
                         'Lesion/Decoder/deconv3_output_2/',
                         'Lesion/Decoder/deconv4_output_1/',
                         'Lesion/Decoder/deconv4_output_1/',
                         'Lesion/Decoder/deconv5_output/']
                return names

            def get_fine_tuning_names():
                names = ['Lesion/Encoder/',
                         'Lesion/Decoder/deconv1/',
                         'Lesion/Decoder/deconv2/',
                         'Lesion/Decoder/deconv3/',
                         'Lesion/Decoder/deconv4/',
                         'Lesion/Decoder/deconv5/',
                         'Lesion/Decoder/lesion_map']

            training_vars = get_train_names()
            training_variables = []
            for name in training_vars:
                training_variables += slim.get_variables(name)

            fine_tuning_vars = get_fine_tuning_names()
            finu_tuning_variables =[]
            for name in fine_tuning_vars:
                finu_tuning_variables += slim.get_variables(name)

            self.mse = tf.reduce_mean(tf.square(lesion_map - self.ground_truth))
            regularization_losses = tf.losses.get_regularization_losses('Lesion')
            self.global_loss = tf.add_n([self.mse] + regularization_losses, 'global_loss')

            # only trainable variables
            self.train_vars = tf.get_collection(key=tf.GraphKeys.TRAINABLE_VARIABLES, scope='Lesion')

            assert len(self.train_vars) == len(training_variables) + len(finu_tuning_variables), "The length is not same"

            # for train_var in self.train_vars:
            #     print(train_var)
            self.global_step = tf.Variable(0, trainable=False)

            # 31,000 - 49,500
            learning_rate = 0.001
            finu_tuning_rate = 0.0001
            optimizer1 = tf.train.AdagradOptimizer(learning_rate=learning_rate)
            optimizer2 = tf.train.AdagradOptimizer(learning_rate=learning_rate)

            self.train_op1 = optimizer1.minimize(var_list=training_variables,
                                                loss=self.mse, global_step=self.global_step)
            self.train_op2 = optimizer2.minimize(var_list=finu_tuning_variables,
                                                 loss=self.mse, global_step=self.global_step)
            self.train_op = tf.group(self.train_op1, self.train_op2, name='train_op')
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
