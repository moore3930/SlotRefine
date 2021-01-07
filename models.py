import tensorflow as tf
import os
import argparse
import io
import time
import numpy as np
import utils as local_utils
import layers
import random
import thumt.layers as layers
import thumt.losses as losses
from thumt.layers.nn import linear
from thumt.models.transformer import transformer_encoder, _ffn_layer
from utils import get_logger
from utils import get_uncoordinated_chunking_nums
from tensorflow.contrib.layers import xavier_initializer


class Model(object):
    """Abstracts a Tensorflow graph for a learning task.
    We use various Model classes as usual abstractions to encapsulate tensorflow
    computational graphs. Each algorithm you will construct in this homework will
    inherit from a Model object.
    """

    def __init__(self):
        self.input_data = None

    def load_data(self):
        """Loads data from disk and stores it in memory.
        Feel free to add instance variables to Model object that store loaded data.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_placeholders(self):
        """Adds placeholder variables to tensorflow computational graph.
        Tensorflow uses placeholder variables to represent locations in a
        computational graph where data is inserted.  These placeholders are used as
        inputs by the rest of the model building code and will be fed data during
        training.
        See for more information:
        https://www.tensorflow.org/versions/r0.7/api_docs/python/io_ops.html#placeholders
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def create_feed_dict(self, input_batch, label_batch):
        """Creates the feed_dict for training the given step.
        A feed_dict takes the form of:
        feed_dict = {
                <placeholder>: <tensor of values to be passed for placeholder>,
                ....
        }

        If label_batch is None, then no labels are added to feed_dict.
        Hint: The keys for the feed_dict should be a subset of the placeholder
                    tensors created in add_placeholders.

        self.args:
            input_batch: A batch of input data.
            label_batch: A batch of label data.
        Returns:
            feed_dict: The feed dictionary mapping from placeholders to values.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_embedding(self):
        """Add embedding layer. that maps from vocabulary to vectors.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_model(self, input_data):
        """Implements core of model that transforms input_data into predictions.
        The core transformation for this model which transforms a batch of input
        data into a batch of predictions.
        self.args:
            input_data: A tensor of shape (batch_size, n_features).
        Returns:
            out: A tensor of shape (batch_size, n_classes)
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        """Adds ops for loss to the computational graph.
        self.args:
            pred: A tensor of shape (batch_size, n_classes)
        Returns:
            loss: A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_epoch(self, sess, input_data, input_labels):
        """Runs an epoch of training.
        Trains the model for one-epoch.

        self.args:
            sess: tf.Session() object
            input_data: np.ndarray of shape (n_samples, n_features)
            input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: scalar. Average minibatch loss of model on epoch.
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def fit(self, sess, input_data, input_labels):
        """Fit model on provided data.
        self.args:
            sess: tf.Session()
            input_data: np.ndarray of shape (n_samples, n_features)
            input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            losses: list of loss per epoch
        """
        raise NotImplementedError("Each Model must re-implement this method.")

    def predict(self, sess, input_data, input_labels=None):
        """Make predictions from the provided model.
        self.args:
            sess: tf.Session()
            input_data: np.ndarray of shape (n_samples, n_features)
            input_labels: np.ndarray of shape (n_samples, n_classes)
        Returns:
            average_loss: Average loss of model.
            predictions: Predictions of model on input_data
        """
        return None, None
        raise NotImplementedError("Each Model must re-implement this method.")


class NatSLU(Model):
    def __init__(self, args):

        self.arg = args

        # Print self.arguments
        print('=============== Args are as below ===============')
        for k, v in sorted(vars(self.arg).items()):
            print(k, "=", v)

        # full path to data will be: ./data + dataset + train/test/valid
        if self.arg.dataset is None:
            print("name of dataset can not be None")
            exit(1)
        elif self.arg.dataset == "snips":
            print("use snips dataset")
        elif self.arg.dataset == "atis":
            print("use atis dataset")
        else:
            print("use own dataset: ", self.arg.dataset)

        # add logger
        self.logger = get_logger(
            self.arg.name,
            self.arg.log_dir,
            self.arg.config_dir)

        # init data paths
        self.full_train_path = os.path.join("./data", self.arg.dataset, self.arg.train_data_path)
        self.full_test_path = os.path.join("./data", self.arg.dataset, self.arg.test_data_path)
        self.full_valid_path = os.path.join("./data", self.arg.dataset, self.arg.valid_data_path)

        # create tokenizer
        self.create_tokenizer()

        # get index of O-tag
        self.o_idx = 0
        for word, idx in self.seq_out_tokenizer.word_index.items():
            if word == 'o':
                self.o_idx = idx
                print("o_idx is: ".format(self.o_idx))
                break

        # global step
        self.global_step = tf.Variable(0, trainable=False, name='global_step')

        # create placeholders
        self.create_placeholder()

        # create train graph
        self.create_train_graph()

        # create eval graph
        self.create_eval_graph()

        # create inference graph
        self.create_test_graph()

    def add_optimizer(self, loss, global_step, isAdam=True):
        """
        Add optimizer for training variables

        Parameters
        ----------
        loss:		Computed loss

        Returns
        -------
        train_op:	Training optimizer
        """
        learning_rate = tf.train.exponential_decay(self.arg.lr, global_step, self.arg.decay_steps,
                                                   self.arg.decay_rate, staircase=False)

        with tf.name_scope('Optimizer'):
            if isAdam and self.arg.learning_rate_decay:
                optimizer = tf.train.AdamOptimizer(learning_rate)
            elif isAdam:
                optimizer = tf.train.AdamOptimizer(self.arg.lr)
            else:
                optimizer = tf.train.GradientDescentOptimizer(self.arg.lr)
            params = tf.trainable_variables()
            gradients = tf.gradients(loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, 5.0)
            train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=global_step)

        return train_op

    def create_tokenizer(self):
        self.seq_in_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token='<unk>',
                                                                      split=self.arg.split)
        self.seq_out_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', split=self.arg.split,
                                                                       oov_token='<unk>')
        self.label_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', split=self.arg.split,
                                                                     oov_token='<unk>')

        # with open(os.path.join(self.full_train_path, self.arg.input_file)) as fin:
        with open(os.path.join(self.full_train_path, self.arg.input_file), encoding=self.arg.encode_mode) as fin:
            lines = fin.readlines()
            lines = [line.strip().lower().split('\t') for line in lines]
            try:
                seq_in, seq_out, intent = zip(*lines)
                # remove digiter
                if self.arg.rm_nums:
                    seq_in = [local_utils.remove_digital_sentence_processer(line) for line in seq_in]
            except:
                print(lines)
                print('input data is unvalid!')
            self.seq_in_tokenizer.fit_on_texts(seq_in)
            self.seq_out_tokenizer.fit_on_texts(seq_out)
            self.label_tokenizer.fit_on_texts(intent)

        print("size of seq_in_tokenizer is {}".format(len(self.seq_in_tokenizer.word_index)))
        print("size of seq_out_tokenizer is {}".format(len(self.seq_out_tokenizer.word_index)))
        print("size of label_tokenizer is {}".format(len(self.label_tokenizer.word_index)))

    def batch_process(self, lines):
        # lines = [line.decode(self.arg.encode_mode) for line in lines]
        lines = [line.strip().lower().split('\t') for line in lines]

        try:
            # for dirty samples, replace multi-space with one
            seq_in, seq_out, label = zip(*lines)
            seq_in = [' '.join(line.split()) for line in seq_in]
        except:
            print(lines)
            print('input data is unvalid!')

        # remove digiter
        if self.arg.rm_nums:
            seq_in = [local_utils.remove_digital_sentence_processer(line) for line in seq_in]

        seq_in_ids = self.seq_in_tokenizer.texts_to_sequences(seq_in)
        seq_out_ids = self.seq_out_tokenizer.texts_to_sequences(seq_out)
        label_ids = self.label_tokenizer.texts_to_sequences(label)

        label_ids = np.array(label_ids).astype(np.int32)
        label_ids = label_ids.squeeze()

        seq_in_ids = tf.keras.preprocessing.sequence.pad_sequences(seq_in_ids, padding='post', truncating='post')
        temp = seq_in_ids > 0
        sequence_length = temp.sum(-1)
        sequence_length = sequence_length.astype(np.int32)

        seq_out_ids = tf.keras.preprocessing.sequence.pad_sequences(seq_out_ids, padding='post', truncating='post')

        seq_out_weights = seq_out_ids > 0
        seq_out_weights = seq_out_weights.astype(np.float32)

        return seq_in_ids, sequence_length, seq_out_ids, seq_out_weights, label_ids

    def get_batch(self, path, batch_size, is_train=False):
        dataset = tf.data.TextLineDataset([path])
        if is_train:
            dataset = dataset.shuffle(1000)
        dataset = dataset.batch(batch_size)
        iter = dataset.make_initializable_iterator()
        batch = iter.get_next()  # Tensor("IteratorGetNext:0", shape=(?,), dtype=string)

        input_data, sequence_length, slots, slot_weights, intent = \
            tf.py_func(self.batch_process, [batch], [tf.int32, tf.int32, tf.int32, tf.float32, tf.int32])

        return input_data, sequence_length, slots, slot_weights, intent, iter

    def get_batch_np_iter(self, path):
        data = []
        with open(path, 'r') as fin:
            for line in fin:
                line = line.strip()
                data.append(line)

        random.shuffle(data)
        for line in data:
            yield line

    def get_batch_np(self, iterator, path, batch_size, is_train=False):
        cnt = 0
        batch = []
        is_last_batch = False
        while True:
            try:
                line = next(iterator)
                batch.append(line)
                cnt += 1

                if batch_size == cnt:
                    break
            except StopIteration:
                iterator = self.get_batch_np_iter(path)
                is_last_batch = True
        return batch, iterator, is_last_batch

    def infer_batch_process(self, lines):
        pass

    def get_infer_bach(self, path, batch_size):
        pass

    def get_start_tags(self, slot_outputs):
        # pred_slot = slot_outputs.reshape((slot_outputs.shape[0], slot_outputs.shape[1], -1))
        pred_slot = slot_outputs[:, :, 2:].argmax(-1) + 2
        start_ids = []
        for word, idx in self.seq_out_tokenizer.word_index.items():
            if word.startswith('b-', 0, len(word)):
                start_ids.append(idx)
        start_tags = [[i if i in start_ids else 0 for i in line] for line in pred_slot]
        return start_tags

    def create_model(
            self,
            input_data,
            input_tags,
            input_size,
            sequence_length,
            slot_size,
            intent_size,
            hidden_size=128,
            is_training=True,
            model_name="SlotRefine"
    ):
        with tf.variable_scope(name_or_scope=model_name, reuse=tf.AUTO_REUSE):
            dtype = tf.get_variable_scope().dtype
            print("slot_size is {}".format(slot_size))
            print("intent_size is {}".format(intent_size))

            word_embedding = tf.get_variable("word_embedding", [input_size, hidden_size],
                                             initializer=xavier_initializer())
            inputs_emb = tf.nn.embedding_lookup(word_embedding, input_data)  # [batch, len_q, hidden_size]
            tag_embedding = tf.get_variable("tag_embedding", [slot_size, hidden_size], initializer=xavier_initializer())
            tags_emb = tf.nn.embedding_lookup(tag_embedding, input_tags)  # [batch, len_q, hidden_size]

            inputs = inputs_emb + tags_emb

            # insert CLS as the first token
            cls = tf.get_variable("cls", [hidden_size], trainable=True, initializer=xavier_initializer())
            cls = tf.reshape(cls, [1, 1, -1])
            cls = tf.tile(cls, [tf.shape(inputs)[0], 1, 1])
            inputs = tf.concat([cls, inputs], 1)

            src_mask = tf.sequence_mask(sequence_length + 1, maxlen=tf.shape(inputs)[1],
                                        dtype=dtype or tf.float32)  # [batch, len_q]
            src_mask.set_shape((None, None))

            print(src_mask.shape)

            if self.arg.multiply_embedding_mode == "sqrt_depth":
                inputs = inputs * (hidden_size ** -0.5)

            inputs = inputs * tf.expand_dims(src_mask, -1)
            bias = tf.get_variable("bias", [hidden_size])
            encoder_input = tf.nn.bias_add(inputs, bias)
            enc_attn_bias = layers.attention.attention_bias(src_mask, "masking", dtype=dtype)

            if self.arg.residual_dropout:
                if is_training:
                    keep_prob = 1.0 - self.arg.residual_dropout
                else:
                    keep_prob = 1.0
                encoder_input = tf.nn.dropout(encoder_input, keep_prob)

            # Feed into Transformer
            att_dropout = self.arg.attention_dropout
            res_dropout = self.arg.residual_dropout
            if not is_training:
                self.arg.attention_dropout = 0.0
                self.arg.residual_dropout = 0.0
            outputs = transformer_encoder(encoder_input, enc_attn_bias, self.arg)  # [batch, len_q + 1, out_size]
            self.arg.attention_dropout = att_dropout
            self.arg.residual_dropout = res_dropout

            intent_output, slot_output = tf.split(outputs, [1, tf.shape(outputs)[1] - 1], 1)

            with tf.variable_scope("intent_proj"):
                intent_state = intent_output
                intent_output = _ffn_layer(intent_output, self.arg.hidden_size, intent_size, scope="intent")

                # mask first token of intent_output forcing that no padding label be predicted.
                mask_values = tf.ones(tf.shape(intent_output)) * -1e10
                mask_true = tf.ones(tf.shape(intent_output), dtype=bool)
                mask_false = tf.zeros(tf.shape(intent_output), dtype=bool)
                intent_output_mask = tf.concat([mask_true[:, :, :2], mask_false[:, :, 2:]], -1)
                intent_output = tf.where(intent_output_mask, mask_values, intent_output)

            with tf.variable_scope("slot_proj"):

                slot_output = tf.concat([slot_output, tf.tile(intent_state, [1, tf.shape(slot_output)[1], 1])], 2)
                # slot_output = linear(slot_output, slot_size, True, True, scope="slot")  # [?, ?, slot_size]
                slot_output = _ffn_layer(slot_output, self.arg.hidden_size, slot_size, scope='slot')

                # mask first two tokens (_PAD_, _UNK_) of slot_outputs forcing that no padding label be predicted.
                mask_values = tf.ones(tf.shape(slot_output)) * -1e10
                mask_true = tf.ones(tf.shape(slot_output), dtype=bool)
                mask_false = tf.zeros(tf.shape(slot_output), dtype=bool)
                slot_outputs_mask = tf.concat([mask_true[:, :, :2], mask_false[:, :, 2:]], -1)
                slot_output = tf.where(slot_outputs_mask, mask_values, slot_output)

            outputs = [slot_output, intent_output]
        return outputs

    def create_loss(self, training_outputs, slots, slot_weights, intent):
        slots_shape = tf.shape(slots)
        slots_reshape = tf.reshape(slots, [-1])
        slot_outputs = training_outputs[0]
        slot_outputs = tf.reshape(slot_outputs, [tf.shape(slots_reshape)[0], -1])

        with tf.variable_scope("slot_loss"):
            print("==== create loss ====")
            print(slots_reshape.shape)
            print(slot_outputs.shape)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=slots_reshape, logits=slot_outputs)
            # crossent = tf.compat.v1.losses.softmax_cross_entropy
            crossent = tf.reshape(crossent, slots_shape)
            slot_loss = tf.reduce_sum(crossent * slot_weights, 1)
            total_size = tf.reduce_sum(slot_weights, 1)
            total_size += 1e-12
            slot_loss = slot_loss / total_size

        intent_output = training_outputs[1]
        intent_output = tf.reshape(intent_output, [tf.shape(intent)[0], -1])

        with tf.variable_scope("intent_loss"):
            print("==== intent loss ====")
            print(intent.shape)
            print(intent_output.shape)
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=intent, logits=intent_output)
            intent_loss = tf.reduce_mean(crossent)

        return slot_loss, intent_loss

    def create_placeholder(self):
        self.input_data = tf.placeholder(tf.int32, [None, None], name='seq_input')
        self.input_tags = tf.placeholder(tf.int32, [None, None], name='input_label')
        self.sequence_length = tf.placeholder(tf.int32, None, name='seq_length')
        self.slots = tf.placeholder(tf.int32, [None, None], name='slots')
        self.slot_weights = tf.placeholder(tf.float32, [None, None], name='slot_weights')
        self.intent = tf.placeholder(tf.int32, None, name='intent')

    def create_train_graph(self):
        print('=== create_train_graph ===')
        print(self.input_data.shape)
        print(self.sequence_length.shape)

        # feed into model
        self.train_outputs = self.create_model(self.input_data, self.input_tags,
                                               len(self.seq_in_tokenizer.word_index) + 1,
                                               self.sequence_length,
                                               len(self.seq_out_tokenizer.word_index) + 1,
                                               len(self.label_tokenizer.word_index) + 1,
                                               hidden_size=self.arg.hidden_size)

        # get loss
        self.slot_loss, self.intent_loss = self.create_loss(self.train_outputs, self.slots,
                                                            self.slot_weights, self.intent)
        self.loss = self.arg.alpha * self.slot_loss + (1 - self.arg.alpha) * self.intent_loss

        if self.arg.opt == 'adam':
            self.train_op = self.add_optimizer(self.loss, self.global_step)
        else:
            self.train_op = self.add_optimizer(self.loss, self.global_step, isAdam=False)

        self.merged_summ = tf.summary.merge_all()

    def create_eval_graph(self):
        # reuse and feed into model
        self.eval_outputs = self.create_model(self.input_data, self.input_tags,
                                              len(self.seq_in_tokenizer.word_index) + 1,
                                              self.sequence_length,
                                              len(self.seq_out_tokenizer.word_index) + 1,
                                              len(self.label_tokenizer.word_index) + 1,
                                              hidden_size=self.arg.hidden_size,
                                              is_training=False)
        self.eval_outputs.append(self.slots)
        self.eval_outputs.append(self.intent)
        self.eval_outputs.append(self.sequence_length)

    def create_test_graph(self):
        # reuse and feed into model
        self.test_outputs = self.create_model(self.input_data, self.input_tags,
                                              len(self.seq_in_tokenizer.word_index) + 1,
                                              self.sequence_length,
                                              len(self.seq_out_tokenizer.word_index) + 1,
                                              len(self.label_tokenizer.word_index) + 1,
                                              hidden_size=self.arg.hidden_size,
                                              is_training=False)

        self.test_outputs.append(self.slots)
        self.test_outputs.append(self.intent)
        self.test_outputs.append(self.sequence_length)
        self.test_outputs.append(self.input_data)

    def train_one_epoch(self, sess, epoch, shuffle=True):
        """Run one training epoch"""
        losses = []
        slot_losses = []
        intent_losses = []
        cnt = 0
        step = 0
        train_path = os.path.join(self.full_train_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(train_path)

        while 1:
            step = step + 1

            batch, iterator, last_batch = self.get_batch_np(batch_iter, train_path, self.arg.batch_size)
            batch_iter = iterator
            seq_in_ids, sequence_length, seq_out_ids, seq_out_weights, label_ids = self.batch_process(batch)

            first_pass_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # first pass
                train_ouput, loss, slot_loss, intent_loss, _ = \
                    sess.run([self.train_outputs, self.loss, self.slot_loss, self.intent_loss, self.train_op],
                             feed_dict={self.input_data: seq_in_ids,
                                        self.input_tags: first_pass_in_tags,
                                        self.sequence_length: sequence_length,
                                        self.slots: seq_out_ids,
                                        self.slot_weights: seq_out_weights,
                                        self.intent: label_ids})
                losses.append(loss)
                slot_losses.append(slot_loss)
                intent_losses.append(intent_loss)

                # second pass
                slot = train_ouput[0]
                second_pass_in_tags = self.get_start_tags(slot)
                train_ouput, loss, slot_loss, intent_loss, _ = \
                    sess.run([self.train_outputs, self.loss, self.slot_loss, self.intent_loss, self.train_op],
                             feed_dict={self.input_data: seq_in_ids,
                                        self.input_tags: second_pass_in_tags,
                                        self.sequence_length: sequence_length,
                                        self.slots: seq_out_ids,
                                        self.slot_weights: seq_out_weights,
                                        self.intent: label_ids})
            except:
                print("Runtime Error in train_one_epoch")
                break

            cnt += self.arg.batch_size
            if step % 20 == 0:
                self.logger.info(
                    "E:{} Sents: {}: Train Loss \t{:.5}\t{:.5}\t{:.5}".format(
                        epoch,
                        cnt,
                        np.mean(loss),
                        np.mean(slot_loss),
                        np.mean(intent_loss)
                    )
                )
                losses = []
                slot_losses = []
                intent_losses = []

            if last_batch:
                break

    def evaluation(self, sess):
        """Do Evaluation"""

        def valid(eval_outputs):

            # print(type(eval_outputs))
            # print(eval_outputs[0].shape)    # pred_slots
            # print(eval_outputs[1].shape)    # pred_intents
            # print(eval_outputs[2].shape)    # correct_slots
            # print(eval_outputs[3].shape)    # correct_intent
            # print(eval_outputs[4].shape)    # sequence_length

            # intent
            # pred_intent = eval_outputs[1].argmax(-1).reshape(-1)
            pred_intent = eval_outputs[1][:, :, 2:].argmax(-1).reshape(-1) + 2
            correct_intent = eval_outputs[3]
            intent_acc_sample_wise = correct_intent == pred_intent
            intent_acc = intent_acc_sample_wise.astype(np.float)
            intent_acc = np.mean(intent_acc) * 100.0
            # print("intent acc is {}".format(intent_acc))

            # slot acc
            sequence_length = eval_outputs[4]
            correct_slot = eval_outputs[2]
            pred_slot = eval_outputs[0].reshape((correct_slot.shape[0], correct_slot.shape[1], -1))
            pred_slot = pred_slot[:, :, 2:].argmax(-1) + 2

            slot_acc_sample_wise = correct_slot == pred_slot  # [batch_size, max_len]
            a = np.arange(correct_slot.shape[1])
            mask = np.tile(np.expand_dims(a, 0), [correct_slot.shape[0], 1]) >= np.expand_dims(sequence_length, -1)

            slot_acc_sample_wise = np.logical_or(mask, slot_acc_sample_wise)
            slot_acc_sample_wise = np.logical_and.reduce(slot_acc_sample_wise, -1)
            slot_acc_sample_wise = slot_acc_sample_wise.astype(np.float)
            slot_acc = np.mean(slot_acc_sample_wise) * 100.0

            # sent acc
            sent_acc_sampel_wise = np.logical_and(intent_acc_sample_wise, slot_acc_sample_wise)
            sent_acc = np.mean(sent_acc_sampel_wise.astype(np.float)) * 100.0

            # calculate slot F1
            pred_slot_label = []
            correct_slot_label = []

            for pred_line, correct_line, length in zip(pred_slot, correct_slot, sequence_length):
                pred_temp = []
                correct_temp = []
                for i in range(length):
                    pred_temp.append(self.seq_out_tokenizer.index_word[pred_line[i]])
                    correct_temp.append(self.seq_out_tokenizer.index_word[correct_line[i]])
                pred_slot_label.append(pred_temp)
                correct_slot_label.append(correct_temp)

            f1, precision, recall = local_utils.computeF1Score(correct_slot_label, pred_slot_label)
            # print("F1: {}, precision: {}, recall: {}".format(f1, precision, recall))

            return f1, slot_acc, intent_acc, sent_acc

        step = 0
        f1 = 0
        slot_acc = 0
        intent_acc = 0
        sent_acc = 0
        sample_cnt = 0

        valid_path = os.path.join(self.full_valid_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(valid_path)

        while 1:
            step = step + 1

            batch, iterator, last_batch = self.get_batch_np(batch_iter, valid_path, 1000)
            batch_iter = iterator
            seq_in_ids, sequence_length, seq_out_ids, _, label_ids = self.batch_process(batch)
            first_pass_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # first pass
                eval_outputs = sess.run(self.eval_outputs, feed_dict={self.input_data: seq_in_ids,
                                                                      self.input_tags: first_pass_in_tags,
                                                                      self.sequence_length: sequence_length,
                                                                      self.slots: seq_out_ids,
                                                                      self.intent: label_ids})

                # second pass
                slot = eval_outputs[0]
                second_pass_in_tags = self.get_start_tags(slot)
                eval_outputs = sess.run(self.eval_outputs, feed_dict={self.input_data: seq_in_ids,
                                                                      self.input_tags: second_pass_in_tags,
                                                                      self.sequence_length: sequence_length,
                                                                      self.slots: seq_out_ids,
                                                                      self.intent: label_ids})
            except:
                print("Runtime Error in evaluation")
                break

            f1_batch, slot_acc_batch, intent_acc_batch, sent_acc_batch = valid(eval_outputs)

            f1 = (f1 * sample_cnt + f1_batch * len(eval_outputs[0])) \
                 / (sample_cnt + len(eval_outputs[0]))
            slot_acc = (slot_acc * sample_cnt + slot_acc_batch * len(eval_outputs[0])) \
                       / (sample_cnt + len(eval_outputs[0]))
            intent_acc = (intent_acc * sample_cnt + intent_acc_batch * len(eval_outputs[0])) \
                         / (sample_cnt + len(eval_outputs[0]))
            sent_acc = (sent_acc * sample_cnt + sent_acc_batch * len(eval_outputs[0])) \
                       / (sample_cnt + len(eval_outputs[0]))
            sample_cnt += len(eval_outputs[0])

            if last_batch:
                break

        print("Eval Results: F1: {}, intent_acc: {}, slot_acc: {}, sent_acc: {}".format(f1, intent_acc,
                                                                                        slot_acc, sent_acc))
        print("Running Params: {}-{}-{}-{}-{}-{}-{}-{}".format(self.arg.batch_size, self.arg.lr, self.arg.hidden_size,
                                                               self.arg.filter_size, self.arg.num_heads,
                                                               self.arg.num_encoder_layers,
                                                               self.arg.attention_dropout, self.arg.residual_dropout))

        return f1, slot_acc, intent_acc, sent_acc

    def inference(self, sess, epoch, diff, dump):
        """Do Inferance"""

        def post_process(outputs):
            # intent
            # pred_intent = outputs[1].argmax(-1).reshape(-1)     # [batch_size]
            pred_intent = outputs[1][:, :, 2:].argmax(-1).reshape(-1) + 2
            correct_intent = outputs[3]  # [batch_size]

            # slot
            sequence_length = outputs[4]  # [batch_size, len, size]
            correct_slot = outputs[2]  # [batch_size, len]
            pred_slot = outputs[0].reshape((correct_slot.shape[0], correct_slot.shape[1], -1))
            # pred_slot = np.argmax(pred_slot, 2)     # [batch_size, len]
            pred_slot = pred_slot[:, :, 2:].argmax(-1) + 2

            # input sentence
            input_data = outputs[5]  # [batch_size, len]

            ref = []
            pred = []

            for words, c_i, p_i, seq_len, c_slot, p_slot in zip(input_data, correct_intent, pred_intent,
                                                                sequence_length, correct_slot, pred_slot):
                words_output = ' '.join(
                    [self.seq_in_tokenizer.index_word[idx] for idx, _ in zip(words, range(seq_len))])
                c_i_output = self.label_tokenizer.index_word[c_i]
                c_slot_output = ' '.join(
                    [self.seq_out_tokenizer.index_word[idx] for idx, _ in zip(c_slot, range(seq_len))])
                p_i_output = self.label_tokenizer.index_word[p_i]
                p_slot_output = ' '.join(
                    [self.seq_out_tokenizer.index_word[idx] for idx, _ in zip(p_slot, range(seq_len))])
                ref.append('\t'.join([words_output, c_i_output, c_slot_output]))
                pred.append('\t'.join([words_output, p_i_output, p_slot_output]))
            return ref, pred

        step = 0
        if dump:
            fout = open(os.path.join(self.full_test_path, '{}_{}'.format(self.arg.infer_file, epoch)), 'w')

        test_path = os.path.join(self.full_test_path, self.arg.input_file)
        batch_iter = self.get_batch_np_iter(test_path)

        cnt = 0
        while 1:
            step = step + 1

            batch, iterator, last_batch = self.get_batch_np(batch_iter, test_path, self.arg.batch_size)
            batch_iter = iterator
            seq_in_ids, sequence_length, seq_out_ids, _, label_ids = self.batch_process(batch)
            first_pass_in_tags = np.ones(seq_in_ids.shape, dtype=np.int32) * self.o_idx

            try:
                # first pass
                infer_outputs = sess.run(self.test_outputs, feed_dict={self.input_data: seq_in_ids,
                                                                       self.input_tags: first_pass_in_tags,
                                                                       self.sequence_length: sequence_length,
                                                                       self.slots: seq_out_ids,
                                                                       self.intent: label_ids})

                # second pass
                slot = infer_outputs[0]
                second_pass_in_tags = self.get_start_tags(slot)
                infer_outputs = sess.run(self.test_outputs, feed_dict={self.input_data: seq_in_ids,
                                                                       self.input_tags: second_pass_in_tags,
                                                                       self.sequence_length: sequence_length,
                                                                       self.slots: seq_out_ids,
                                                                       self.intent: label_ids})
            except:
                print("Runtime Error in inference")
                break

            # output
            cnt += self.arg.batch_size
            if dump:
                ref_batch, pred_batch = post_process(infer_outputs)
                for ref_line, pred_line in zip(ref_batch, pred_batch):
                    # if diff and ref_line == pred_line:
                    #     continue
                    fout.write(ref_line + '\n')
                    fout.write(pred_line + '\n')

            if last_batch:
                break

        if dump:
            fout.flush()
            fout.close()

            # calculate uncoordinated chunk nums
            diff_file = os.path.join(self.full_test_path, '{}_{}'.format(self.arg.infer_file, epoch))
            uncoordinated_nums = get_uncoordinated_chunking_nums(diff_file)
            print("uncoordinated nums : {}".format(uncoordinated_nums))

    def fit(self, sess):
        """Train and Evaluate"""
        self.saver = tf.train.Saver()
        save_dir = 'model/' + self.arg.name + '/'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        self.save_path = os.path.join(save_dir, 'best_int_avg')

        if self.arg.restore:
            self.saver.restore(sess, self.save_path)

        for epoch in range(self.arg.max_epochs):
            self.logger.info('Epoch: {}'.format(epoch))

            self.train_one_epoch(sess, epoch)

            self.evaluation(sess)

            if self.arg.dump:
                if epoch % 20 == 0:
                    self.inference(sess, epoch, self.arg.remain_diff, self.arg.dump)
            else:
                print('dump is False')
                self.inference(sess, epoch, self.arg.remain_diff, self.arg.dump)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # fmt: off
    parser.add_argument('-name', dest="name", default='default-SLU', help='Name of the run')
    parser.add_argument("--encode_mode", type=str, default='gb18030', help="encode mode")
    parser.add_argument("--split", type=str, default='\x01', help="split str")
    parser.add_argument('-restore', dest="restore", action='store_true',
                        help='Restore from the previous best saved model')
    parser.add_argument('--dump', type=bool, default=False, help="is dump")
    parser.add_argument("--rm_nums", type=bool, default=False, help="rm nums")
    parser.add_argument("--remain_diff", type=bool, default=True, help="just remain diff")

    # Transformer
    parser.add_argument("--hidden_size", type=int, default=32, help="hidden_size")
    parser.add_argument("--filter_size", type=int, default=32, help="filter_size")
    parser.add_argument("--num_heads", type=int, default=8, help="num_heads")
    parser.add_argument("--num_encoder_layers", type=int, default=2, help="num_encoder_layers")
    parser.add_argument('--attention_dropout', default=0.0, type=float, help='att_dropout')
    parser.add_argument('--residual_dropout', default=0.1, type=float, help='res_dropout')
    parser.add_argument('--relu_dropout', dest="relu_dropout", default=0.0, type=float, help='relu_dropout')
    parser.add_argument('--label_smoothing', dest="label_smoothing", default=0.1, type=float, help='label_smoothing')
    parser.add_argument('--attention_key_channels', dest="attention_key_channels",
                        default=0, type=int, help='attention_key_channels')
    parser.add_argument('--attention_value_channels', dest="attention_value_channels",
                        default=0, type=int, help='attention_value_channels')
    parser.add_argument("--layer_preprocess", type=str, default='none', help="layer_preprocess")
    parser.add_argument("--layer_postprocess", type=str, default='layer_norm', help="layer_postprocess")
    parser.add_argument("--multiply_embedding_mode", type=str, default='sqrt_depth', help="multiply_embedding_mode")
    parser.add_argument("--shared_embedding_and_softmax_weights", type=bool,
                        default=False, help="shared_embedding_and_softmax_weights.")
    parser.add_argument("--shared_source_target_embedding", type=bool,
                        default=False, help="shared_source_target_embedding.")
    parser.add_argument("--position_info_type", type=str, default='relative', help="position_info_type")
    parser.add_argument("--max_relative_dis", type=int, default=16, help="max_relative_dis")

    # Training Environment
    parser.add_argument("--batch_size", type=int, default=512, help="Batch size.")
    parser.add_argument("--max_epochs", type=int, default=20, help="Max epochs to train.")
    parser.add_argument("--no_early_stop", action='store_false', dest='early_stop',
                        help="Disable early stop, which is based on sentence level accuracy.")
    parser.add_argument("--patience", type=int, default=5, help="Patience to wait before stop.")
    parser.add_argument('--lr', dest="lr", default=0.01, type=float, help='Learning rate')
    parser.add_argument('-opt', dest="opt", default='adam', help='Optimizer to use for training')
    parser.add_argument("--alpha", type=float, default=0.5, help="balance the intent & slot filling task")
    parser.add_argument("--learning_rate_decay", type=bool, default=True, help="learning_rate_decay")
    parser.add_argument("--decay_steps", type=int, default=300 * 4, help="decay_steps.")
    parser.add_argument("--decay_rate", type=float, default=0.9, help="decay_rate.")

    # Model and Vocab
    parser.add_argument("--dataset", type=str, default='duer-os',
                        help="""Type 'atis' or 'snips' to use dataset provided by us or enter what ever you named your own dataset.
                    Note, if you don't want to use this part, enter --dataset=''. It can not be None""")
    parser.add_argument("--model_path", type=str, default='./model', help="Path to save model.")
    parser.add_argument("--vocab_path", type=str, default='./vocab', help="Path to vocabulary files.")

    # Data
    parser.add_argument("--train_data_path", type=str, default='train', help="Path to training data files.")
    parser.add_argument("--test_data_path", type=str, default='test', help="Path to testing data files.")
    parser.add_argument("--valid_data_path", type=str, default='test', help="Path to validation data files.")
    parser.add_argument("--input_file", type=str, default='data', help="Input file name.")
    parser.add_argument("--infer_file", type=str, default='infer', help="Infer file name")

    # parser.add_argument("--input_file", type=str, default='seq.in', help="Input file name.")
    # parser.add_argument("--slot_file", type=str, default='seq.out', help="Slot file name.")
    # parser.add_argument("--intent_file", type=str, default='label', help="Intent file name.")

    # Others
    parser.add_argument('-logdir', dest="log_dir", default='./log/', help='Log directory')
    parser.add_argument('-config', dest="config_dir", default='./config/', help='Config directory')
    # fmt: on

    args = parser.parse_args()

    model = NatSLU(args)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        model.fit(sess)

    print('Model Trained Successfully!!')
