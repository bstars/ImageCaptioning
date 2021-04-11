import sys
sys.path.append('..')


import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

from utils.config import CONFIG
from utils.data_utils import COCODataset
from models.Captioning import Captioning

class Solver():
    def __init__(self, model:Captioning, dataset:COCODataset, ckpt_path=None):
        self.model = model
        self.coco = dataset

        self.null_idx = self.coco.word_to_idx['<NULL>']
        self.unk_idx = self.coco.word_to_idx['<UNK>']
        self.start_idx = self.coco.word_to_idx['<START>']
        self.learning_rate = CONFIG.LEARNING_RATE

        self.sess = tf.Session(graph=model.graph)
        with model.graph.as_default():
            self.learning_rate_holder = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate_holder')
            self.optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate_holder)

            self.step = self.optimizer.minimize(self.model.loss)
            self.sess.run(tf.global_variables_initializer())

            if ckpt_path is not None:
                saver = tf.train.Saver()
                saver.restore(self.sess, ckpt_path)
            # self.sess.run(tf.variables_initializer(self.optimizer.variables()))

    def get_training_feed_dict(self, imgs, captions, lr):
        captions_in, captions_out = captions[:, :-1], captions[:, 1:]

        mask = np.array(captions_out!=self.null_idx).astype(np.float)

        for k,v in CONFIG.MASK_WEIGHT.items():
            idx = np.where(captions_out == v)
            mask[idx] = v

        features = self.model.vgg.get_feature(imgs, layername=self.model.vgg_feature_name)
        c0 = np.zeros(shape=[len(imgs), self.model.rnn_dim])

        feed_dict = {
            self.model.feature_holder : features,
            self.model.c0_holder : c0,
            self.model.output_holder : captions_out,
            self.model.input_holder : captions_in,
            self.model.mask_holder : mask,
            self.learning_rate_holder : lr
        }

        return feed_dict

    def get_inference_feed_dict(self, imgs):
        features_test = self.model.vgg.get_feature(imgs, layername=self.model.vgg_feature_name)
        c0 = np.zeros(shape=[len(imgs), self.model.rnn_dim])

        feed_dict = {
            self.model.feature_holder:features_test,
            self.model.c0_holder:c0,
            self.model.start_holder_infer_time:np.ones(shape=(len(imgs))) * self.start_idx
        }

        return feed_dict

    def fit_small_dataset(self, num):
        imgs_test, captions_test = self.coco.load_batch(batch_size=num, kind='val')
        feed_dict_test = self.get_training_feed_dict(imgs_test, captions_test, self.learning_rate)
        for i in range(300):
            l_test, _ = self.sess.run([self.model.loss, self.step], feed_dict=feed_dict_test)
            print(i, l_test)
        feed_dict_inf = self.get_inference_feed_dict(imgs_test)
        captions = self.sess.run(self.model.output_infer_time, feed_dict=feed_dict_inf)
        return imgs_test, captions

    def train(self):
        imgs_test, captions_test = self.coco.load_batch(batch_size=5, kind='val')
        feed_dict_test = self.get_training_feed_dict(imgs_test, captions_test, 0.1)

        num_iter = 0
        num_save = 1
        try:
            while True:
                num_iter += 1
                if num_iter % 100 == 0:
                    self.learning_rate = self.learning_rate * CONFIG.LEARNING_RATE_DECAY
                imgs_train, captions_train = self.coco.load_batch(batch_size=CONFIG.BATCH_SIZE)
                feed_dict_train = self.get_training_feed_dict(imgs_train, captions_train, self.learning_rate)

                l_train, _ = self.sess.run([self.model.loss, self.step], feed_dict=feed_dict_train)
                l_test = self.sess.run(self.model.loss, feed_dict=feed_dict_test)

                print(
                    'iter %d, epoch %.3f, training loss: %.7f, testing loss: %.7f' %
                    (num_iter, self.coco.get_num_epoch(), l_train, l_test)
                )

                if num_iter % 1500==0:
                    os.mkdir('ckpt%d'%(num_save))
                    with self.model.graph.as_default():
                        saver = tf.train.Saver()
                        saver.save(self.sess, "./ckpt%d/checkpoint1" % (num_save))
                    num_save += 1

        except:
            with self.model.graph.as_default():
                saver = tf.train.Saver()
                saver.save(self.sess, "./checkpoint1")


if __name__ == "__main__":
    # embed_mat = loadmat('../embed_mat.mat')['embed']
    embed_mat = None
    model = Captioning(CONFIG.RNN_DIM, embed_dim=CONFIG.WORD2VEC_EMBED_DIM, embed_matrix=embed_mat)
    ds = COCODataset()

    solver = Solver(model, ds)
    imgs, captions = solver.fit_small_dataset(2)
    captions = ds.translate_captions(captions)

    for i in range(len(imgs)):
        plt.imshow(imgs[i])
        plt.title(captions[i])
        plt.show()