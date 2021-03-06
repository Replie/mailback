import os

from models import dual_encoder_trainer
from test_tube import HyperOptArgumentParser

parser = HyperOptArgumentParser(strategy='random_search')

# --------------------------
# build program arguments
# --------------------------

parser.add_opt_argument_list('--lr_1', default=0.0001, options=[0.0001, 0.0002, 0.0004, 0.0008, 0.001, 0.002], type=float,
                             tunnable=True)
parser.add_opt_argument_list('--batch_size', default=10, options=[20, 30, 40, 50], type=int, tunnable=True)
parser.add_opt_argument_list('--embedding_dim', default=320, options=[100, 200, 320, 400], type=int, tunnable=True)
parser.add_opt_argument_list('--max_seq_len', default=50, options=[50, 70, 90, 110], type=int, tunnable=True)

# training params
parser.add_argument('--nb_epochs', default=1000, type=int)
parser.add_argument('--optimizer_name', default='adam')
parser.add_argument('--eval_every_n_batches', default=200, type=int)
parser.add_argument('--save_every_n_batches', default=10000, type=int)
parser.add_argument('--train_mode', default='train')

# model params
parser.add_argument('--nb_grams', default=2, type=int)

# path params
parser.add_argument('--root_dir', default='./data')
parser.add_argument('--dataset_train_path', default='./data/train.csv')
parser.add_argument('--dataset_test_path', default='./data/test.csv')
parser.add_argument('--dataset_val_path', default='./data/val.csv')
parser.add_argument('--vocab_path', default='./data/vocabulary.txt')

parser.add_argument('--model_save_dir', default='./experiments')
parser.add_argument('--test_tube_dir', default='')

# experiment params
parser.add_argument('--exp_name', default='dual_conv_dense')
parser.add_argument('--exp_desc', default='Dual dense + dot product loss. Base model')
parser.add_argument('--debug', default=False)

parser.add_argument('--test', default=False)
parser.add_argument('--epoch', default=0)
parser.add_argument('--step', default=150)

# tf params
parser.add_argument('--gpus', default='1')
parser.add_json_config_argument('-c', '--config', type=str)
# --------------------------
# --------------------------

# parse params
hparams = parser.parse_args()

# --------------------------
# TRAIN *****************
# --------------------------
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = hparams.gpus

if hparams.test:
    dual_encoder_trainer.test_main(hparams)
else:
    dual_encoder_trainer.train_main(hparams)

