import os
import numpy as np
import pandas as pd
import tensorflow as tf
from test_tube import HyperOptArgumentParser

from models.dual_encoder_dense.model_dual_encoder_dense import dot_product_scoring


def test_main(hparams):
    from dataset.ubuntu_dialogue_corpus import Tokenizer
    print(hparams.vocab_path)
    tokenizer = Tokenizer(hparams.vocab_path, hparams.max_seq_len)

    def create_prediction_data(text, labels, tokenizer, mode='encode'):
        print('creating encoding for input...')
        print('printing input: ', text)
        return (np.array([tokenizer.text_to_sequence(text)]),
                tokenizer.texts_to_sequences(labels),
                tokenizer.vocab_size())

    text = "This is a test input."

    data = pd.read_csv(hparams.dataset_train_path)
    labels = data['Utterance'].values
    text_enc, labels_enc, vocab_size = create_prediction_data(
        text, labels, tokenizer, mode='encode')
    x_len = text_enc.shape[0]
    y_len = labels_enc.shape[0]

    input_x = tf.placeholder(dtype=tf.int32, shape=[1, None], name='input_x')
    input_y = tf.placeholder(dtype=tf.int32, shape=[y_len, None], name='input_y')

    embedding = tf.get_variable('embedding',
                                [vocab_size, hparams.embedding_dim])

    # Lookup the embeddings.
    embedding_x = tf.nn.embedding_lookup(embedding, input_x)
    embedding_y = tf.nn.embedding_lookup(embedding, input_y)

    # Sum the embeddings for each instance.
    x = tf.reduce_sum(embedding_x, axis=1)
    y = tf.reduce_sum(embedding_y, axis=1)

    S = dot_product_scoring(x, y, is_training=False)

    value = tf.argmax(S, axis=1, name='value')
    score = tf.reduce_max(S, axis=1, name='score')

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

    saver_all = tf.train.Saver(tf.global_variables(), max_to_keep=100)
    checkpoint_dir = os.path.join(hparams.model_save_dir, hparams.exp_name, 'epoch_{}/step_{}'.format(hparams.epoch,
                                                                                                      hparams.step))
    print('checkpoint_dir %s', checkpoint_dir)
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    saver_all.restore(sess, ckpt.model_checkpoint_path)
    sess.run(tf.global_variables_initializer())

    """Print the number of parameters."""
    total_parameters = 0
    print("Parameters:")
    for variable in tf.trainable_variables():
        shape = variable.get_shape()
        variable_parameters = 1
        for dim in shape:
            variable_parameters *= dim.value
        print("\t%d\t%s" % (variable_parameters, variable.name))
        total_parameters += variable_parameters
    print("Total number of parameters: %d." % total_parameters)

    feed_dict = {input_x: text_enc, input_y: labels_enc}
    out_value, out_score = sess.run([value, score], feed_dict=feed_dict)
    print('label: ', labels[out_value])
    print('score: ', out_score)


if __name__ == '__main__':
    parser = HyperOptArgumentParser(strategy='random_search')
    # path params
    parser.add_opt_argument_list('--lr_1', default=0.0001, options=[0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002],
                                 type=float,
                                 tunnable=True)
    parser.add_opt_argument_list('--batch_size', default=10, options=[20, 30, 40, 50], type=int, tunnable=True)
    parser.add_opt_argument_list('--embedding_dim', default=320, options=[100, 200, 320, 400], type=int, tunnable=True)

    parser.add_argument('--root_dir', default='./data')
    parser.add_argument('--dataset_train_path', default='./data/train.csv')
    parser.add_argument('--dataset_test_path', default='./data/test.csv')
    parser.add_argument('--dataset_val_path', default='./data/val.csv')
    parser.add_argument('--vocab_path', default='./data/vocabulary.txt')
    parser.add_opt_argument_list('--max_seq_len', default=50, options=[50, 70, 90, 110], type=int, tunnable=True)

    parser.add_argument('--epoch', default=0)
    parser.add_argument('--step', default='FINAL')
    parser.add_argument('--model_save_dir', default='./experiments')
    parser.add_argument('--test_tube_dir', default='')

    # experiment params
    parser.add_argument('--exp_name', default='dual_conv_dense')
    hparams = parser.parse_args()
    test_main(hparams=hparams)