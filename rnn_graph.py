"""Provide function to build an RNN model's graph. """

import tensorflow as tf

import properties

def make_rnn_cell(rnn_layer_sizes,
                  dropout_keep_prob=1.0,
                  attn_length=0,
                  base_cell=tf.contrib.rnn.BasicLSTMCell,
                  state_is_tuple=False):
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
    cell = base_cell(num_units, state_is_tuple=state_is_tuple)
    cell = tf.contrib.rnn.DropoutWrapper(
        cell, output_keep_prob=dropout_keep_prob)
    cells.append(cell)

  cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=state_is_tuple)
  if attn_length:
    cell = tf.contrib.rnn.AttentionCellWrapper(
        cell, attn_length, state_is_tuple=state_is_tuple)

  return cell


def data_type():
    return tf.float32


def build_graph(config):
    batch_size = config.batch_size
    # song_length = config.song_length
    num_song_features = config.num_song_features
    with tf.Graph().as_default() as graph:
        # this graph use input_melody to generate a note following input_melody. the generated noted is
        # in a form of probabilities of every song feature.
        # In training, labels shows the correct label following input_melody
        # batch_size means training some
        input_labels = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])
        input_melody = tf.placeholder(dtype=tf.int32, shape=[batch_size, None])

        global_step = tf.Variable(0, trainable=False, name='global_step')

        tf.add_to_collection('input_melody', input_melody)
        tf.add_to_collection('input_labels', input_labels)
        tf.add_to_collection('global_step', global_step)

        with tf.variable_scope('G') as scopeG:
            embedding = tf.get_variable("embedding", [config.melody_params.nor_pitch + 2, config.g_rnn_layers[0]])
            inputs = tf.nn.embedding_lookup(embedding, input_melody)

            cell_g = make_rnn_cell(config.g_rnn_layers)
            init_state_g = cell_g.zero_state(batch_size, data_type())

            norm = tf.random_normal_initializer(stddev=1.0, dtype=data_type())
            outputs, init_state_g = tf.nn.dynamic_rnn(
                cell_g, inputs, initial_state=init_state_g, parallel_iterations=1,
                swap_memory=True)
            outputs_flat = tf.reshape(outputs, [-1, cell_g.output_size])

            pitch_logits_flat = tf.contrib.layers.fully_connected(
                outputs_flat, config.melody_params.nor_pitch + 2, scope='output_to_pitch', weights_initializer=norm)

            tf.add_to_collection('inputs_f', inputs)
            tf.add_to_collection('outputs_f', outputs)

            tf.add_to_collection('pitch_logits_flat', pitch_logits_flat)

            # Training
            # get softmax cross entropy of every song features
            # softmax for every features shows the probable value of one step
            labels = tf.reshape(input_labels, [-1])

            pitch_softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=pitch_logits_flat)
            loss = tf.reduce_mean(pitch_softmax_cross_entropy)

            tf.add_to_collection('pitch_softmax_cross_entropy', pitch_softmax_cross_entropy)
            tf.add_to_collection('loss', loss)
            learning_rate = tf.train.exponential_decay(
                config.initial_learning_rate, global_step, config.decay_steps,
                config.decay_rate, staircase=True, name='learning_rate')

            opt = tf.train.AdamOptimizer(learning_rate)
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, _ = tf.clip_by_global_norm(gradients, config.clip_norm)
            train_op = opt.apply_gradients(zip(clipped_gradients, params), global_step)
            tf.add_to_collection('learning_rate', learning_rate)
            tf.add_to_collection('train_op', train_op)

            # Generate a step for generating. Returned softmaxs contain probabilities of every one-hot features

            temperature = tf.placeholder(data_type(), [])
            pitch_softmax_flat = tf.nn.softmax(
                tf.div(pitch_logits_flat, tf.fill([config.melody_params.nor_pitch + 2], temperature)))
            pitch_softmax = tf.reshape(pitch_softmax_flat, [batch_size, -1, config.melody_params.nor_pitch + 2])

            tf.add_to_collection('temperature', temperature)
            tf.add_to_collection('pitch_softmax', pitch_softmax)
            tf.add_to_collection('initial_state', init_state_g)
    return graph



def main(_):
    melody_param = properties.MelodyParam()
    config = RnnConfig(melody_param=melody_param)
    build_graph(config)


class RnnConfig:
    def __init__(self, melody_param=None):
        self.batch_size = 2
        self.song_length = 30
        self.generated_song_length = 128
        self.num_song_features = 4
        self.g_rnn_layers = [100, 100]
        self.clip_norm = 5
        self.initial_learning_rate = 0.005
        self.decay_steps = 1000
        self.decay_rate = 0.95
        self.reg_constant = 0.01  # for regularization, choose a appropriate one

        self.melody_params = melody_param


if __name__ == '__main__':
    tf.app.run()





