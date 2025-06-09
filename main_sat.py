import os
import argparse
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

from config import Config
from model import CaptionGenerator
from dataset_new_SAT import prepare_train_data, prepare_eval_data, prepare_test_data

def main(args):
    config = Config()
    config.phase = args.phase
    config.train_cnn = args.train_cnn
    config.beam_size = args.beam_size

    with tf.Session() as sess:
        if args.phase == 'train':
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if args.load:
                model.load(sess, args.model_file)
            if args.load_cnn:
                model.load_cnn(sess, args.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif args.phase == 'eval':
            data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, args.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, data, vocabulary)

        else:
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, args.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--phase', type=str, default='train', choices=['train', 'eval', 'test'])
    parser.add_argument('--load', action='store_true', help='Load pretrained model')
    parser.add_argument('--model_file', type=str, default=None)
    parser.add_argument('--load_cnn', action='store_true', help='Load pretrained CNN')
    parser.add_argument('--cnn_model_file', type=str, default='./vgg16_no_fc.npy')
    parser.add_argument('--train_cnn', action='store_true', help='Train CNN along with RNN')
    parser.add_argument('--beam_size', type=int, default=3)

    args = parser.parse_args()
    main(args)
