import sys
sys.path.append('..')

import tensorflow
tensorflow.compat.v1.disable_v2_behavior()

import tensorflow.compat.v1 as tf
import numpy as np
from scipy.io import savemat

from utils.data_utils import COCODataset
from utils.config import CONFIG

def sample_dataset(data:COCODataset, size:int, type='train'):
    captions = data.train_captions
    # captions = data[type + '_captions']
    END_IDX = data.word_to_idx['<END>']
    idx_to_word = data.idx_to_word

    ret = np.zeros(shape=[size, 2])

    idxes = np.random.randint(0, len(captions), size)

    for i in range(size):
        idx = idxes[i]
        sentence = captions[idx]
        sentence_length = np.where(sentence==END_IDX)[0][0]

        # Each sentence contains 1 <START>, 1 <END> and 1 <NULL>
        sentence = sentence[:sentence_length+2]
        sentence_length = len(sentence)

        context_idx = np.random.randint(0, sentence_length)

        shift = np.random.randint(-CONFIG.WORD2VEC_WINDOW_SIZE, CONFIG.WORD2VEC_WINDOW_SIZE, 1)[0]
        while shift == 0:
            shift = np.random.randint(-CONFIG.WORD2VEC_WINDOW_SIZE, CONFIG.WORD2VEC_WINDOW_SIZE, 1)[0]

        target_idx = context_idx + shift
        target_idx = max(0, target_idx)
        target_idx = min(sentence_length-1, target_idx)

        # print(decode_captions(sentence, idx_to_word))
        # print(idx_to_word[sentence[context_idx]], idx_to_word[sentence[target_idx]])
        # print()

        ret[i] = [sentence[context_idx], sentence[target_idx]]
    return ret

class Word2Vec(object):
    def __init__(self, embed_dim, vocab_size):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.params = {}
        self.graph = tf.Graph()
        self.sess = tf.Session(graph=self.graph)

        with self.graph.as_default():
            with tf.name_scope('Word2Vec'):
                self.setup()
                self.build_graph()


    def setup(self):
        self.XHolder = tf.placeholder(dtype=tf.int32, shape=[None])
        self.YHolder = tf.placeholder(dtype=tf.int32, shape=[None])

        self.params['W_embed'] = tf.get_variable(name='W_embed',
                                                 shape=[self.vocab_size, self.embed_dim],
                                                 dtype=tf.float32,
                                                 initializer=tf.truncated_normal_initializer(stddev=0.1))
        # self.params['b_embed'] = tf.get_variable('b_embed',
        #                                          shape=[self.embed_dim],
        #                                          dtype=tf.float32,
        #                                          initializer=tf.zeros_initializer())

        self.params['W_nce'] = tf.get_variable(name='W_nce',
                                               shape=[self.vocab_size, self.embed_dim],
                                               dtype=tf.float32,
                                               initializer=tf.truncated_normal_initializer(stddev=0.1))
        self.params['b_nce'] = tf.get_variable('b_nce',
                                               shape=[self.vocab_size],
                                               dtype=tf.float32,
                                               initializer=tf.zeros_initializer())

    def build_graph(self):
        embed = tf.nn.embedding_lookup(self.params['W_embed'], self.XHolder)
        YHolder_expand = tf.expand_dims(self.YHolder, axis=-1)

        self.loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights=self.params['W_nce'], biases=self.params['b_nce'], labels=YHolder_expand, inputs=embed,
                num_sampled=1, num_classes=self.vocab_size
            )
        )
        print(self.loss.shape)

    def train(self, data):
        test_data = sample_dataset(data, 100, type='val')

        with self.graph.as_default():
            step = tf.train.AdamOptimizer(CONFIG.WORD2VEC_LEARNING_RATE).minimize(self.loss)
            self.sess.run(tf.global_variables_initializer())

        try:
            i = 0
            while i<100000:

                i += 1
                train_data = sample_dataset(data, CONFIG.WORD2VEC_BATCH_SIZE, type='train')
                feed_dict = {
                    self.XHolder:train_data[:, 0],
                    self.YHolder:train_data[:, 1]
                }
                l_train, _ = self.sess.run([self.loss, step], feed_dict=feed_dict)

                feed_dict = {
                    self.XHolder:test_data[:, 0],
                    self.YHolder:test_data[:, 1]
                }
                l_test = self.sess.run(self.loss, feed_dict=feed_dict)
                print('iter %d, training loss: %.2f, testing loss: %.2f' %
                      (i, l_train, l_test)
                )

        except KeyboardInterrupt:
            embed_mat = self.sess.run(self.params['W_embed'])
            savemat('../data/embed_mat.mat', {'embed':embed_mat})

if __name__ == '__main__':
    coco = COCODataset()
    model = Word2Vec(CONFIG.WORD2VEC_EMBED_DIM, CONFIG.VOCAB_SIZE)
    model.train(coco)