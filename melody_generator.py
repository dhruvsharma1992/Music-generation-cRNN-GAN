"""" 
RNN-GAN 
to run:
python melody_generator.py --network gan --datadir data/examples/ --traindir data/traindir2/
"""
import os, datetime

import tensorflow as tf
import numpy as np

import rnn_gan_graph
import rnn_graph
import properties
import melody_utils
import rnn_melody_utils
# import seqGan.seqgan

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string("datadir", None,
                           "Directory to save and load midi music files.")
tf.app.flags.DEFINE_string("traindir", None,
                           "Directory to save checkpoints.")
tf.app.flags.DEFINE_string("generated_data_dir", None,
                           "Directory to save midi files.")
tf.app.flags.DEFINE_string("network", 'rnn',
                           "Select a network to use.")
tf.app.flags.DEFINE_integer("num_max_epochs_1", 100000,
                            "Select max epoch of rnn gan.")
tf.app.flags.DEFINE_integer("num_max_epochs_2", 1000,
                            "Select max epoch of rnn.")


def rnn_gan_train(graph, loader, config):
    global_step = graph.get_collection('global_step')[0]
    input_melody = graph.get_collection('input_melody')[0]
    loss_d = graph.get_collection('loss_d')[0]
    loss_g = graph.get_collection('loss_g')[0]
    clip_d_op = graph.get_collection('clip_d_op')[0]
    train_g_op = graph.get_collection('train_g_op')[0]
    pre_output_melody = graph.get_collection('pre_output_melody')[0]
    output_melody = graph.get_collection('output_melody')[0]
    g_pre_train_op = graph.get_collection('g_pre_train_op')[0]
    pre_loss_g = graph.get_collection('pre_loss_g')[0]

    sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.traindir, save_model_secs=12000, global_step=global_step)

    with sv.managed_session() as session:
        global_step_ = session.run(global_step)
        tf.logging.info('Starting training loop for RNN-GAN...')
        print('Begin Run Net. Print Every 100 epochs ')
        while global_step_ < FLAGS.num_max_epochs_1:
            if sv.should_stop():
                break
            if global_step_ < 1500:
                # Pre-Training
                batch_songs = loader.get_batch(config.batch_size, config.song_length)
                output_melody_, pre_loss_g_, global_step_, _ = session.run(
                    [pre_output_melody, pre_loss_g, global_step,
                     g_pre_train_op], {input_melody: batch_songs})
                if global_step_ % 100 == 0:
                    print('Epoch: %d  pre_loss: %f' % (global_step_, pre_loss_g_))

            else:
                # Training
                d_iters = 4
                for _ in range(d_iters):
                    batch_songs = loader.get_batch(config.batch_size, config.song_length)
                    _ = session.run([clip_d_op], {input_melody: batch_songs})

                batch_songs = loader.get_batch(config.batch_size, config.song_length)
                g_op, global_step_, g_loss, d_loss, output_melody_ = session.run(
                    [train_g_op, global_step, loss_g, loss_d, output_melody], {input_melody: batch_songs})

                if global_step_ % 50 == 0:
                    print('Global_step: %d    loss_d: %f    loss_g: %f' % (global_step_, d_loss, g_loss))

            if (1000 > global_step_ > 500 or global_step_ > 1500) and global_step_ % 200 == 0:
                filename = os.path.join(FLAGS.generated_data_dir, 'global_step-{}-{}.midi'
                                        .format(global_step_, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
                print('save file: %s' % filename)
                loader.data_to_song(filename, output_melody_[0])
                print(output_melody_[0][0:30])


def rnn_gan(melody_param):
    config = rnn_gan_graph.RnnGanConfig(melody_param=melody_param)
    print('Begin Load Data....')
    loader = melody_utils.MusicDataLoader(FLAGS.datadir, config)
    loader.get_batch(config.batch_size, config.song_length)
    print('Begin Create Graph for RNN-GAN ....')
    graph = rnn_gan_graph.build_graph(config)

    rnn_gan_train(graph, loader, config)


def rnn_train(graph, loader, config):
    global_step = graph.get_collection('global_step')[0]
    temperature = graph.get_collection('temperature')[0]
    input_melody = graph.get_collection('input_melody')[0]
    input_labels = graph.get_collection('input_labels')[0]
    loss = graph.get_collection('loss')[0]
    train_op = graph.get_collection('train_op')[0]

    pitch_logits_flat = graph.get_collection('pitch_logits_flat')[0]
    pitch_softmax_cross_entropy = graph.get_collection('pitch_softmax_cross_entropy')[0]
    pitch_softmax = graph.get_collection('pitch_softmax')[0]

    inputs_f = graph.get_collection('inputs_f')[0]
    outputs_f = graph.get_collection('outputs_f')[0]

    temperature_ = 1

    sv = tf.train.Supervisor(graph=graph, logdir=FLAGS.traindir, save_model_secs=12000, global_step=global_step)

    with sv.managed_session() as session:
        global_step_ = session.run(global_step)
        tf.logging.info('Starting training loop...')
        print('Begin Run Net. Print Every 100 epochs ')
        while global_step_ < FLAGS.num_max_epochs_2:
            if sv.should_stop():
                break
            inputs, outputs = loader.get_batch_rnn(config.batch_size, config.song_length)
            loss_, global_step_, _, pitch_logits_flat_, pitch_softmax_cross_entropy_, inputs_f_, outputs_f_= \
                session.run([loss, global_step, train_op, pitch_logits_flat, pitch_softmax_cross_entropy,
                             inputs_f, outputs_f], {input_melody: inputs, input_labels: outputs})

            if global_step_ % 10 == 0:
                print('Global_step: %d    loss: %f' % (global_step_, loss_))
            if global_step_ > 300 and global_step_ % 50 == 0:
                generated_melody = []
                inputs, outputs = loader.get_batch_rnn(config.batch_size, config.song_length + 1)
                notes = [[i] for i in inputs[:, -1]]
                for i in range(config.generated_song_length + 1):
                    if i == 0:
                        outputs_f_ = session.run([outputs_f], {input_melody: inputs,
                                                          temperature: temperature_})
                    else:
                        pitch_softmax_ = session.run([pitch_softmax], {input_melody: notes, temperature: temperature_})
                        notes = retrieve_note(pitch_softmax_, config.batch_size, config.melody_params)
                        generated_melody.append(notes[0][0])
                filename = os.path.join(FLAGS.generated_data_dir, 'global_step-{}-{}.midi'
                                        .format(global_step_, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
                print('save file: %s' % filename)
                print(generated_melody)
                loader.data_to_song(filename, generated_melody)


def retrieve_note(pitch_softmax, batch_size, melody_param):
    pitchs = []
    for i in range(batch_size):
        pitchs.append([np.random.choice(melody_param.nor_pitch + 2, p=pitch_softmax[0][i][0])])
    return pitchs


def rnn(melody_param):
    config = rnn_graph.RnnConfig(melody_param)
    print('Begin Create Graph....')
    graph = rnn_graph.build_graph(config)
    print('Begin Load Data....')
    loader = rnn_melody_utils.MusicDataLoader(FLAGS.datadir, config)
    rnn_train(graph, loader, config)


# def seq_gan(melody_param):
#     config = seqGan.seqgan.SeqGanConfig(melody_param)
#     loader = rnn_melody_utils.MusicDataLoader(FLAGS.datadir, config)
#     loader.genedir = FLAGS.generated_data_dir
#     seqGan.seqgan.run_seq_gan(config, loader)
# 

def main(_):
    print(tf.__version__)
    if not FLAGS.datadir or not os.path.exists(FLAGS.datadir):
        raise ValueError("Must set --datadir to midi music dir.")
    if not FLAGS.traindir:
        raise ValueError("Must set --traindir to dir where I can save model and plots.")


    if not os.path.exists(FLAGS.traindir):
        try:
            os.makedirs(FLAGS.traindir)
        except:
            raise IOError

    FLAGS.generated_data_dir = os.path.join(FLAGS.traindir, 'generated_data')
    if not os.path.exists(FLAGS.generated_data_dir):
        try:
            os.makedirs(FLAGS.generated_data_dir)
        except:
            raise IOError

    print('Train dir: %s' % FLAGS.traindir)

    melody_param = properties.MelodyParam()

    if FLAGS.network == 'rnn_gan':
        rnn_gan(melody_param)
    elif FLAGS.network == 'rnn':
        rnn(melody_param)
    # elif FLAGS.network == 'seq_gan':
    #     seq_gan(melody_param)


if __name__ == '__main__':
    tf.app.run()