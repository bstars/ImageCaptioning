import sys
sys.path.append('..')

import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf
import numpy as np
import os


from models.vgg import vgg16
from utils.data_utils import COCODataset
from utils.config import CONFIG



class Captioning(object):
    def __init__(self, rnn_dim, embed_dim, embed_matrix:np.array=None, vgg_layer='fc2'):
        self.coco = COCODataset()
        self.vgg = vgg16(path=os.path.join(CONFIG.COCO_PATH, 'vgg16_weights.npz'), from_tf_check_point=False)
        self.vgg_feature_name = vgg_layer
        self.cnn_dim = self.vgg.get_layer_shape(self.vgg_feature_name).as_list()[-1]

        # self.cnn_dim = 512


        self.word_to_idx = self.coco.word_to_idx
        self.idx_to_word = self.coco.idx_to_word
        self.vocab_size = CONFIG.VOCAB_SIZE

        self.null_idx = self.word_to_idx['<NULL>']
        self.start_idx = self.word_to_idx['<START>']
        self.end_idx = self.word_to_idx['<END>']

        self.embed_dim = embed_dim
        self.rnn_dim = rnn_dim
        self.embed_matrix = embed_matrix

        # Suppose the caption is :
        #       <START> i play basketball ... as Klay <END>    # length 17
        # Then the input is
        #       <START> i play basketball ... as Klay           # length 16
        # And the desired output is
        #       i play basketball ... as Klay <END>             # length
        # If the caption is padded with <NULL>, then <NULL> is not accounted in loss function,
        # thus we need a mask too indicate the <NULL> token

        self.time_span = CONFIG.TIME_SPAN - 1
        self.params = {}

        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with tf.name_scope('Captioning'):
                self.setup()
                self.build_graph()

    def setup(self):
        if self.embed_matrix is not None:
            embed_init = tf.constant_initializer(value=self.embed_matrix)
        else:
            embed_init = tf.truncated_normal_initializer(stddev=0.1)

        self.params['W_word_embed'] = tf.get_variable(
            name='W_word_embed',
            dtype=tf.float32,
            shape=[CONFIG.VOCAB_SIZE, self.embed_dim],
            initializer=embed_init,
            trainable=True
        )

        # weight and bias that transform cnn feature to rnn initial state
        self.params['W_proj'] = tf.get_variable(
            name='W_proj',
            shape=[self.cnn_dim, self.rnn_dim],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(
                stddev=0.1)
        ) / tf.sqrt(float(self.cnn_dim))
        self.params['b_proj'] = tf.get_variable(
            name='b_proj',
            shape=[self.rnn_dim],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )


        # weight and bias that transform rnn hidden states to output
        self.params['W_output'] = tf.get_variable(
            name='W_output',
            shape=[self.rnn_dim, self.vocab_size],
            dtype=tf.float32,
            initializer=tf.random_normal_initializer(stddev=0.1)
        ) / tf.sqrt(float(self.rnn_dim))

        self.params['b_output'] = tf.get_variable(
            name='b_output',
            shape=[self.vocab_size],
            dtype=tf.float32,
            initializer=tf.zeros_initializer()
        )


        self.rnn_cell1 = tf.nn.rnn_cell.LSTMCell(num_units=self.rnn_dim, name='lstm_cell1', state_is_tuple=True)

    def build_graph(self):
        # image feature produced by CNN
        self.feature_holder = tf.placeholder(dtype=tf.float32, shape=[None, self.cnn_dim], name='feature_holder')

        # mask for output, if the label is <NULL>, this word will not be counted since <NULL> is determined once
        # the model prediction gives <END>
        self.mask_holder = tf.placeholder(dtype=tf.float32, shape=[None, self.time_span], name='mask_holder')

        # input and output placeholder, each element is the index of word in vocabulary
        self.input_holder = tf.placeholder(dtype=tf.int32, shape=[None, self.time_span], name='inputs_holder')
        self.output_holder = tf.placeholder(dtype=tf.int32, shape=[None, self.time_span], name='output_holder')

        # placeholder for LSTM cell, will always feed zeros to it
        self.c0_holder = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_dim], name='c0_holder')

        self.initial_state = tf.nn.elu(
            tf.matmul(self.feature_holder, self.params['W_proj']) + self.params['b_proj']
        )

        '''training stage graph'''
        # the none trivial part of the sentence(i.e. not <NULL>) is known
        word_embed = tf.nn.embedding_lookup(self.params['W_word_embed'], self.input_holder)  # [none, 16, embed_dim]
        rnn1_output, _ = tf.nn.dynamic_rnn(self.rnn_cell1, inputs=word_embed, initial_state=tf.nn.rnn_cell.LSTMStateTuple(self.initial_state, self.initial_state))


        # [none, time_span, hidden_dim]

        output = tf.matmul(rnn1_output, self.params['W_output']) + self.params['b_output']
        # [none, time_span, vocab_size]

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.output_holder, logits=output) * self.mask_holder
        )


        '''inference stage graph'''
        output_infer_time = []
        self.start_holder_infer_time = tf.placeholder(dtype=tf.int32, shape=[None], name='start_holder_infer_time')
        state = tf.nn.rnn_cell.LSTMStateTuple(self.initial_state, self.initial_state)

        out_prev = self.start_holder_infer_time  # use the word generated at previous step as the input of next step
        for i in range(self.time_span):
            embed = tf.nn.embedding_lookup(self.params['W_word_embed'], out_prev)
            rnn1_output_step_infer_time, state = self.rnn_cell1.call(embed, state)

            logits = tf.matmul(rnn1_output_step_infer_time, self.params['W_output']) + self.params['b_output']   # [none, vocab_size]
            out_prev = tf.argmax(logits, axis=1) # [None]
            output_infer_time.append(out_prev)

        output_infer_time = tf.stack(output_infer_time) # [time_span, none]
        self.output_infer_time = tf.transpose(output_infer_time)


if __name__ == '__main__':
    model = Captioning(CONFIG.RNN_DIM, embed_dim=CONFIG.WORD2VEC_EMBED_DIM)
