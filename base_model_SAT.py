import os
import copy
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
from tqdm import tqdm

from utils.nn import NN
from utils.coco.coco import COCO
from utils.coco.pycocoevalcap.eval import COCOEvalCap
from utils.misc import ImageLoader, CaptionData, TopN


class BaseModel(object):
    def __init__(self, config):
        self.config = config
        self.is_train = config.phase == 'train'
        self.train_cnn = self.is_train and config.train_cnn
        self.image_loader = ImageLoader('./utils/ilsvrc_2012_mean.npy')
        self.image_shape = [224, 224, 3]
        self.nn = NN(config)
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.build()

    def build(self):
        raise NotImplementedError()

    def train(self, sess, train_data):
        print("Training the model...")
        config = self.config

        if not os.path.exists(config.summary_dir):
            os.makedirs(config.summary_dir)
        train_writer = tf.summary.FileWriter(config.summary_dir, sess.graph)

        for epoch in range(config.num_epochs):
            train_data.reset()
            for batch_i in range(train_data.num_batches):
                batch = train_data.next_batch()
                image_files, sentences, masks = batch
                images = self.image_loader.load_images(image_files)
                feed_dict = {
                    self.images: images,
                    self.sentences: sentences,
                    self.masks: masks
                }
                _, summary, global_step = sess.run(
                    [self.opt_op, self.summary, self.global_step],
                    feed_dict=feed_dict)
                train_writer.add_summary(summary, global_step)
                if (global_step + 1) % config.save_period == 0:
                    self.save(sess)

        self.save(sess)
        train_writer.close()
        print("Training complete.")

    def eval(self, sess, eval_gt_coco, eval_data, vocabulary):
        print("Evaluating the model ...")
        config = self.config
        results = []

        if not os.path.exists(config.eval_result_dir):
            os.makedirs(config.eval_result_dir)

        idx = 0
        for _ in tqdm(range(eval_data.num_batches), desc='eval_batches'):
            batch = eval_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if eval_data.current_idx < eval_data.count else eval_data.fake_count
            for l in range(config.batch_size - fake_cnt):
                word_idxs = caption_data[l][0].sentence
                caption = vocabulary.get_sentence(word_idxs)
                results.append({'image_id': eval_data.image_ids[idx], 'caption': caption})
                idx += 1

                if config.save_eval_result_as_image:
                    image_file = batch[l]
                    image_name = os.path.splitext(os.path.basename(image_file))[0]
                    img = plt.imread(image_file)
                    plt.imshow(img)
                    plt.axis('off')
                    plt.title(caption)
                    plt.savefig(os.path.join(config.eval_result_dir, image_name + '_result.jpg'))
                    plt.close()

        with open(config.eval_result_file, 'w') as fp:
            json.dump(results, fp)

        eval_result_coco = eval_gt_coco.loadRes(config.eval_result_file)
        scorer = COCOEvalCap(eval_gt_coco, eval_result_coco)
        scorer.evaluate()
        print("Evaluation complete.")

    def test(self, sess, test_data, vocabulary):
        print("Testing the model ...")
        config = self.config

        if not os.path.exists(config.test_result_dir):
            os.makedirs(config.test_result_dir)

        captions = []
        scores = []

        for _ in tqdm(range(test_data.num_batches), desc='test_batches'):
            batch = test_data.next_batch()
            caption_data = self.beam_search(sess, batch, vocabulary)

            fake_cnt = 0 if test_data.current_idx < test_data.count else test_data.fake_count
            for l in range(config.batch_size - fake_cnt):
                word_idxs = caption_data[l][0].sentence
                score = caption_data[l][0].score
                caption = vocabulary.get_sentence(word_idxs)
                captions.append(caption)
                scores.append(score)

                image_file = batch[l]
                image_name = os.path.splitext(os.path.basename(image_file))[0]
                img = plt.imread(image_file)
                plt.imshow(img)
                plt.axis('off')
                plt.title(caption)
                plt.savefig(os.path.join(config.test_result_dir, image_name + '_result.jpg'))
                plt.close()

        import pandas as pd
        results = pd.DataFrame({'image_files': test_data.image_files,
                                'caption': captions,
                                'prob': scores})
        results.to_csv(config.test_result_file)
        print("Testing complete.")

    def beam_search(self, sess, image_files, vocabulary):
        config = self.config
        images = self.image_loader.load_images(image_files)

        contexts, initial_memory, initial_output = sess.run(
            [self.conv_feats, self.initial_memory, self.initial_output],
            feed_dict={self.images: images})

        partial_caption_data = []
        complete_caption_data = []
        for k in range(config.batch_size):
            initial_beam = CaptionData(sentence=[],
                                       memory=initial_memory[k],
                                       output=initial_output[k],
                                       score=1.0)
            partial_caption_data.append(TopN(config.beam_size))
            partial_caption_data[-1].push(initial_beam)
            complete_caption_data.append(TopN(config.beam_size))

        for idx in range(config.max_caption_length):
            partial_caption_data_lists = []
            for k in range(config.batch_size):
                data = partial_caption_data[k].extract()
                partial_caption_data_lists.append(data)
                partial_caption_data[k].reset()

            num_steps = 1 if idx == 0 else config.beam_size
            for b in range(num_steps):
                if idx == 0:
                    last_word = np.zeros((config.batch_size), np.int32)
                else:
                    last_word = np.array([pcl[b].sentence[-1] for pcl in partial_caption_data_lists], np.int32)

                last_memory = np.array([pcl[b].memory for pcl in partial_caption_data_lists], np.float32)
                last_output = np.array([pcl[b].output for pcl in partial_caption_data_lists], np.float32)

                memory, output, scores = sess.run(
                    [self.memory, self.output, self.probs],
                    feed_dict={
                        self.contexts: contexts,
                        self.last_word: last_word,
                        self.last_memory: last_memory,
                        self.last_output: last_output})

                for k in range(config.batch_size):
                    caption_data = partial_caption_data_lists[k][b]
                    words_and_scores = list(enumerate(scores[k]))
                    words_and_scores.sort(key=lambda x: -x[1])
                    words_and_scores = words_and_scores[:config.beam_size + 1]

                    for w, s in words_and_scores:
                        sentence = caption_data.sentence + [w]
                        score = caption_data.score * s
                        beam = CaptionData(sentence, memory[k], output[k], score)
                        if vocabulary.words[w] == '.':
                            complete_caption_data[k].push(beam)
                        else:
                            partial_caption_data[k].push(beam)

        results = []
        for k in range(config.batch_size):
            if complete_caption_data[k].size() == 0:
                complete_caption_data[k] = partial_caption_data[k]
            results.append(complete_caption_data[k].extract(sort=True))

        return results

    def save(self, sess):
        config = self.config
        data = {}
        for v in tf.global_variables():
            data[v.name] = sess.run(v)
        save_path = os.path.join(config.save_dir, str(sess.run(self.global_step)))
        print(f"Saving the model to {save_path}.npy ...")
        np.save(save_path, data)

        with open(os.path.join(config.save_dir, "config.pickle"), "wb") as info_file:
            config_ = copy.copy(config)
            config_.global_step = sess.run(self.global_step)
            pickle.dump(config_, info_file)

        print("Model saved.")

    def load(self, sess, model_file=None):
        config = self.config
        if model_file is not None:
            save_path = model_file
        else:
            info_path = os.path.join(config.save_dir, "config.pickle")
            with open(info_path, "rb") as info_file:
                config = pickle.load(info_file)
                global_step = config.global_step
            save_path = os.path.join(config.save_dir, f"{global_step}.npy")

        print(f"Loading the model from {save_path} ...")
        data_dict = np.load(save_path, allow_pickle=True).item()
        count = 0
        for v in tqdm(tf.global_variables()):
            if v.name in data_dict:
                sess.run(v.assign(data_dict[v.name]))
                count += 1
        print(f"{count} tensors loaded.")

    def load_cnn(self, sess, data_path, ignore_missing=True):
        print(f"Loading the CNN from {data_path} ...")
        data_dict = np.load(data_path, allow_pickle=True).item()
        count = 0
        for op_name in tqdm(data_dict):
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in data_dict[op_name].items():
                    try:
                        var = tf.get_variable(param_name)
                        sess.run(var.assign(data))
                        count += 1
                    except ValueError:
                        if not ignore_missing:
                            raise
        print(f"{count} tensors loaded.")
