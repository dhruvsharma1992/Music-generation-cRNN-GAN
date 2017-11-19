import os, datetime

import numpy as np
import tensorflow as tf
import random

from generator import Generator
from discriminator import Discriminator
from rollout import ROLLOUT


class SeqGanConfig:
    def __init__(self, melody_params):
        self.emb_dim = 32  # embedding dimension
        self.hidden_dim = 32  # hidden state dimension of lstm cell
        self.seq_length = 50  # sequence length
        self.start_token = 0
        self.pre_epoch_num = 60  # supervise (maximum likelihood estimation) epochs
        self.seed = 88
        self.batch_size = 64

        self.dis_embedding_dim = 64
        self.dis_filter_sizes = [1, 2, 3, 4, 5]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.dis_num_filters = [100, 200, 200, 200, 200]  # [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]
        self.dis_dropout_keep_prob = 0.75
        self.dis_l2_reg_lambda = 0.2
        self.dis_batch_size = 64

        self.total_batch = 200
        self.generated_num = 400
        self.melody_params = melody_params

        self.num_batch = 20


def generate_samples(sess, trainable_model, batch_size, generated_num, loader):
    # Generate Samples
    generated_samples = []
    for _ in range(int(generated_num / batch_size)):
        generated_samples.extend(trainable_model.generate(sess))
    loader.store_negative_data(generated_samples)


def pre_train_epoch(sess, trainable_model, loader, config):
    # Pre-train the generator using MLE for one epoch
    supervised_g_losses = []

    for it in xrange(config.num_batch):
        batch, _ = loader.get_batch_rnn(config.batch_size, config.seq_length)
        _, g_loss = trainable_model.pretrain_step(sess, batch)
        supervised_g_losses.append(g_loss)

    return np.mean(supervised_g_losses)


def run_seq_gan(config, loader):
    random.seed(config.seed)
    np.random.seed(config.seed)
    assert config.start_token == 0

    vocab_size = config.melody_params.nor_pitch + 2

    generator = Generator(vocab_size, config.batch_size, config.emb_dim, config.hidden_dim, config.seq_length,
                          config.start_token)

    discriminator = Discriminator(sequence_length=config.seq_length, num_classes=2, vocab_size=vocab_size,
                                  embedding_size=config.dis_embedding_dim,
                                  filter_sizes=config.dis_filter_sizes, num_filters=config.dis_num_filters,
                                  l2_reg_lambda=config.dis_l2_reg_lambda)

    config_t = tf.ConfigProto()
    config_t.gpu_options.allow_growth = True
    sess = tf.Session(config=config_t)
    sess.run(tf.global_variables_initializer())

    #  pre-train generator
    print ('Start pre-training...')
    for epoch in xrange(config.pre_epoch_num):
        loss = pre_train_epoch(sess, generator, loader, config)
        if epoch % 5 == 4:
            print ('pre-train epoch ', epoch, 'test_loss ', loss)

    print ('Start pre-training discriminator...')
    # Train 3 epoch on the generated data and do this for 50 times
    for _ in range(30):
        generate_samples(sess, generator, config.batch_size, config.generated_num, loader)
        for _ in range(3):
            for it in xrange(config.num_batch):
                x_batch, y_batch = loader.seq_train_data(config.batch_size, config.seq_length)
                feed = {
                    discriminator.input_x: x_batch,
                    discriminator.input_y: y_batch,
                    discriminator.dropout_keep_prob: config.dis_dropout_keep_prob
                }
                _ = sess.run(discriminator.train_op, feed)

    rollout = ROLLOUT(generator, 0.8)

    print ('#########################################################################')
    print ('Start Adversarial Training...')
    for total_batch in range(config.total_batch):
        # Train the generator for one step
        for it in range(1):
            samples = generator.generate(sess)
            rewards = rollout.get_reward(sess, samples, 16, discriminator, config.seq_length)
            feed = {generator.x: samples, generator.rewards: rewards}
            _ = sess.run(generator.g_updates, feed_dict=feed)

        # Update roll-out parameters
        rollout.update_params()

        if total_batch % 5 == 4:
            filename = os.path.join(loader.genedir, 'global_step-{}-{}.midi'
                                    .format(total_batch, datetime.datetime.today().strftime('%Y-%m-%d-%H-%M-%S')))
            print('save file: %s' % filename)
            print(samples[0])
            loader.data_to_song(filename, samples[0])

        # Train the discriminator
        for _ in range(5):
            generate_samples(sess, generator, config.batch_size, config.generated_num, loader)

            for _ in range(3):
                for it in xrange(config.num_batch):
                    x_batch, y_batch = loader.seq_train_data(config.batch_size, config.seq_length)
                    feed = {
                        discriminator.input_x: x_batch,
                        discriminator.input_y: y_batch,
                        discriminator.dropout_keep_prob: config.dis_dropout_keep_prob
                    }
                    _ = sess.run(discriminator.train_op, feed)