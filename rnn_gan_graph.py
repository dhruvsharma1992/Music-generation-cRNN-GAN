"""Provide function to build an RNN-GAN model's graph. """

import tensorflow as tf


def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=True,
                  reuse=False):
  """Makes a RNN cell from the given hyperparameters.

  Args:
    rnn_layer_sizes: A list of integer sizes (in units) for each layer of the RNN.
    dropout_keep_prob: The float probability to keep the output of any given sub-cell.
    attn_length: The size of the attention vector.
    base_cell: The base tf.contrib.rnn.RNNCell to use for sub-cells.
    state_is_tuple: A boolean specifying whether to use tuple of hidden matrix
        and cell matrix as a state instead of a concatenated matrix.

  Returns:
      A tf.contrib.rnn.MultiRNNCell based on the given hyperparameters.
  """
  cells = []
  for num_units in rnn_layer_sizes:
    cell = base_cell(num_units, state_is_tuple=state_is_tuple, reuse=reuse)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple, reuse=reuse)

  return cell


def data_type():
    return tf.float32  # this can be replaced with tf.float32


def discriminator(config, inputs, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()
    fw_cell_d = make_rnn_cell(config.d_rnn_layers, reuse=reuse)
    bw_cell_d = make_rnn_cell(config.d_rnn_layers, reuse=reuse)
    init_state_fw_d = fw_cell_d.zero_state(config.batch_size, data_type())
    init_state_bw_d = bw_cell_d.zero_state(config.batch_size, data_type())
    outputs, final_state_fw, final_state_bw = \
        tf.contrib.rnn.static_bidirectional_rnn(fw_cell_d, bw_cell_d, inputs,
                                                initial_state_fw=init_state_fw_d,
                                                initial_state_bw=init_state_bw_d, scope='bidirection_rnn')
    decisions = tf.contrib.layers.fully_connected(outputs, 1, scope='decision')
    decisions = tf.transpose(decisions, perm=[1, 0, 2])
    decision = tf.reduce_mean(decisions, reduction_indices=[1, 2])
    return decision


def build_graph(config):
    print("config\n",config)
    batch_size = config.batch_size
    song_length = config.song_length
    num_song_features = config.num_song_features
    with tf.Graph().as_default() as graph:
        # input_melody is a seed with the shape of [batch_size, note_length, num_song_features] to generate a melody.
        input_melody = tf.placeholder(dtype=tf.int32, shape=[batch_size, song_length, config.num_song_features])

        global_step = tf.Variable(1, trainable=False, name='global_step')

        tf.add_to_collection('input_melody', input_melody)
        tf.add_to_collection('global_step', global_step)

        with tf.variable_scope('G') as scopeG:
            cell_g = make_rnn_cell(config.g_rnn_layers)  # set to [300, 300]
            init_state_g = cell_g.zero_state(batch_size, data_type())

            # PreTraining
            # input a note and the output note should be the next note in the input melody

            input_melodys = [tf.squeeze(input_, [1]) for input_ in tf.split(tf.to_float(input_melody), song_length, 1)]

            pre_output_melody = []
            pre_output_melody.append(input_melodys[0])
            state_g = init_state_g

            for i in range(song_length-1):
                if i > 0:
                    scopeG.reuse_variables()

                inputs = tf.nn.relu(tf.contrib.layers.fully_connected(input_melodys[i], config.g_rnn_layers[0],
                                                           scope='note_to_input'))
                outputs, state_g = cell_g(inputs, state_g)

                g_output_note = tf.contrib.layers.fully_connected(outputs, config.num_song_features,
                                                                  scope='output_to_note')
                pre_output_melody.append(g_output_note)

            pre_output_melody_tf = tf.transpose(pre_output_melody, perm=[1, 0, 2])
            tf.add_to_collection('pre_output_melody', pre_output_melody_tf)

            weight_ticks = tf.constant([config.melody_params.ticks_weight, config.melody_params.length_weight,
                                        config.melody_params.pitch_weight, config.melody_params.velocity_weight],
                                       dtype=data_type())

            pre_loss_g = tf.reduce_mean(tf.squared_difference(tf.multiply(pre_output_melody_tf, weight_ticks),
                                                             tf.multiply(tf.to_float(input_melody), weight_ticks)))

            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_loss = config.reg_constant * sum(reg_losses)

            pre_loss_g = pre_loss_g + reg_loss
            tf.add_to_collection('pre_loss_g', pre_loss_g)

            pre_g_opt = tf.train.GradientDescentOptimizer(config.initial_g_learning_rate)
            g_params = [v for v in tf.trainable_variables() if v.name.startswith('G/')]
            pre_g_gradients = tf.gradients(pre_loss_g, g_params)
            clipped_gradients, _ = tf.clip_by_global_norm(pre_g_gradients, config.clip_norm)
            g_pre_train_op = pre_g_opt.apply_gradients(zip(clipped_gradients, g_params), global_step)
            # tf.add_to_collection('g_learning_rate', g_learning_rate)
            tf.add_to_collection('g_pre_train_op', g_pre_train_op)

            # Train
            # generate a random note as a seed with the shape of [batch_size, num_song_features] to
            # generate a piece of melody with the length of notes_length
            # config.melody_params.nor_ticks = int(config.melody_params.nor_ticks)
            print("ticks:",config.melody_params.nor_ticks)
            
            random_ticks = tf.random_uniform(shape=[batch_size, 1], minval=0,
                                             maxval=config.melody_params.nor_ticks, dtype=tf.int32)
            random_length = tf.random_uniform(shape=[batch_size, 1], minval=0,
                                              maxval=config.melody_params.nor_length, dtype=tf.int32)
            random_pitch = tf.random_uniform(shape=[batch_size, 1], minval=0,
                                             maxval=config.melody_params.nor_pitch, dtype=tf.int32)
            random_velocity = tf.random_uniform(shape=[batch_size, 1], minval=0,
                                                maxval=config.melody_params.nor_velocity, dtype=tf.int32)
            # random_rnn_input' shape is [batch_size, num_song_features]
            random_rnn_input = tf.to_float(tf.concat([random_ticks, random_length, random_pitch, random_velocity], 1))

            output_melody = []
            generated_note = random_rnn_input

            for i in range(song_length):
                if i > 0:
                    scopeG.reuse_variables()

                inputs = tf.nn.relu(tf.contrib.layers.fully_connected(generated_note, config.g_rnn_layers[0],
                                                                      scope='note_to_input'))
                outputs, state_g = cell_g(inputs, state_g)

                g_output_note = tf.contrib.layers.fully_connected(outputs, config.num_song_features,
                                                                  scope='output_to_note')
                random_offset = tf.truncated_normal(shape=[batch_size, 4], mean=0.0, stddev=27.0)
                generated_note = tf.add(g_output_note, random_offset)
                output_melody.append(g_output_note)

            output_melody_tf = tf.transpose(output_melody, perm=[1, 0, 2])
            tf.add_to_collection('output_melody', output_melody_tf)

        with tf.variable_scope('D') as scopeD:
            inputs_d = [tf.to_float(tf.squeeze(input_d)) for input_d in tf.split(input_melody, song_length, 1)]
            real_d = discriminator(config, inputs_d)
            generated_d = discriminator(config, output_melody, reuse=True)
            if config.wgan:
                loss_d = -tf.reduce_mean(real_d) + tf.reduce_mean(generated_d)
                loss_g = -tf.reduce_mean(generated_d)
            else:
                loss_d = tf.reduce_mean(-tf.log(tf.clip_by_value(real_d, 1e-1000000, 1.0)) \
                               - tf.log(1 - tf.clip_by_value(generated_d, 0.0, 1.0 - 1e-1000000)))
                loss_g = tf.reduce_mean(-tf.log(tf.clip_by_value(generated_d, 1e-1000000, 1.0)))

        d_params = [v for v in tf.trainable_variables() if v.name.startswith('D/')]

        loss_d = loss_d + reg_loss
        loss_g = loss_g + reg_loss
        optimizer_d = tf.train.RMSPropOptimizer(learning_rate=config.initial_learning_rate)
        optimizer_g = tf.train.RMSPropOptimizer(learning_rate=config.initial_learning_rate)

        d_grads, _ = tf.clip_by_global_norm(
            tf.gradients(loss_d, d_params, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N),
            config.max_grad_norm)
        g_grads, _ = tf.clip_by_global_norm(tf.gradients(loss_g, g_params, aggregation_method=
                                                tf.AggregationMethod.EXPERIMENTAL_ACCUMULATE_N), config.max_grad_norm)
        train_d_op = optimizer_d.apply_gradients(zip(d_grads, d_params))
        clip_d_op = [w.assign(tf.clip_by_value(w, -config.clip_w_norm, config.clip_w_norm)) for w in d_params]

        train_g_op = optimizer_g.apply_gradients(zip(g_grads, g_params), global_step)
        tf.add_to_collection('loss_d', loss_d)
        tf.add_to_collection('loss_g', loss_g)
        tf.add_to_collection('train_d_op', train_d_op)
        tf.add_to_collection('clip_d_op', clip_d_op)
        tf.add_to_collection('train_g_op', train_g_op)
    return graph


class RnnGanConfig:
    def __init__(self, melody_param=None):
        self.batch_size = 10
        self.song_length = 500
        self.num_song_features = 4
        self.g_rnn_layers = [300, 300]
        self.d_rnn_layers = [300, 300]
        self.clip_norm = 5
        self.initial_g_learning_rate = 0.005
        self.initial_learning_rate = 0.0001
        self.decay_steps = 1000
        self.decay_rate = 0.95

        self.reg_constant = 0.1  # for regularization, choose a appropriate one
        self.max_grad_norm = 5
        self.clip_w_norm = 0.02

        self.wgan = True

        self.melody_params = melody_param


def main(_):
    config = RnnGanConfig()
    build_graph(config)


if __name__ == "__main__":
    tf.app.run()





