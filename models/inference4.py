import sys

sys.path.append('..')

import os
import tensorflow

tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt
import cv2
import numpy as np

from models.Captioning import Captioning
from utils.config import CONFIG


class Inference():
    def __init__(self, model:Captioning, ckpt_path):
        self.model = model
        self.sess = tf.Session(graph=model.graph)
        with model.graph.as_default():
            self.inference_input_holder = tf.placeholder(dtype=tf.int32, shape=[None])
            self.inference_c_holder = tf.placeholder(dtype=tf.float32, shape=[None, model.rnn_dim])
            self.inference_h_holder = tf.placeholder(dtype=tf.float32, shape=[None, model.rnn_dim])

            state = tf.nn.rnn_cell.LSTMStateTuple(self.inference_c_holder, self.inference_h_holder)
            inference_embed = tf.nn.embedding_lookup(model.params['W_word_embed'], self.inference_input_holder)
            inference_rnn_output, (self.next_c, self.next_h) = model.rnn_cell1.call(inference_embed, state)

            self.inference_output = tf.nn.softmax(
                tf.matmul(inference_rnn_output, model.params['W_output']) + model.params['b_output'], axis=1
            )

            #self.sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_path)

    def predict(self, images, beam_width):
        features = self.model.vgg.get_feature(images, self.model.vgg_feature_name)
        initial_states = self.sess.run(self.model.initial_state, feed_dict={self.model.feature_holder:features})
        # [none, 512]
        rets = []

        for initial_state in initial_states:
            captions = self.beam_search(np.array([initial_state]), beam_width)
            rets.extend(self.model.coco.translate_captions(captions))
        return rets

    def get_dict_dumb(self, inputs, c, h):
        feed_dict = {
            self.inference_input_holder:np.copy(inputs),
            self.inference_c_holder:np.copy(c),
            self.inference_h_holder:np.copy(h)
        }
        return feed_dict

    def beam_search(self, state, beam_width=10):
        inputs = np.array([self.model.start_idx])

        top_outputs = [([self.model.start_idx], 0, state, state)]  # (word_list, log_probability, c, h)

        c = state
        h = state

        while True:
            feed_dict = self.get_dict_dumb(inputs, c, h)
            outputs, c, h = self.sess.run(
                [self.inference_output, self.next_c, self.next_h],
                feed_dict=feed_dict)

            print(outputs.shape, c.shape, h.shape)

            for i in range(len(top_outputs)):
                outputs[i, :] = np.log(outputs[i, :]) + top_outputs[i][1]

            ridx, cidx = np.unravel_index(np.argsort(outputs, axis=None), outputs.shape)

            next_top_inputs = []
            new_inputs = []
            new_h = []
            new_c = []

            for i in range(beam_width):
                if cidx[-i] == self.model.end_idx:
                    return [top_outputs[ridx[-i]][0] + [cidx[-i]]]
                next_top_inputs.append(
                    (top_outputs[ridx[-i]][0] + [cidx[-i]], outputs[ridx[-i], cidx[-i]], c[ridx[-i]], h[ridx[-i]])
                )
                new_inputs.append(cidx[-i])
                new_c.append(c[ridx[-i]])
                new_h.append(h[ridx[-i]])

            inputs = np.array(new_inputs)
            top_outputs = next_top_inputs
            c = np.stack(new_c)
            h = np.stack(new_h)


if __name__ == '__main__':
    fnames = [
        '../test_img/img_479.jpg',
        '../test_img/img_703.jpg',
        '../test_img/car.jpg',
        '../test_img/bike.jpg',
        '../test_img/eat.jpg',
    ]

    imgs = []
    for fname in fnames:
        img = plt.imread(fname)
        img = cv2.resize(img, (CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH))
        print(img.shape)
        imgs.append(img)
    imgs = np.array(imgs)


    model = Captioning(CONFIG.RNN_DIM, CONFIG.WORD2VEC_EMBED_DIM, None)
    predictor = Inference(model, '../ckptyao/checkpoint1')
    captions = predictor.predict(imgs, 30)

    for i in range(len(imgs)):
        plt.imshow(imgs[i])
        plt.title(captions[i])
        plt.show()

