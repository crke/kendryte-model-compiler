import os
import tempfile
import tensorflow as tf

from model_loader import darknet
from model_loader.darknet.D2T_lib import darknet_tool, tensorflow_tool

cfg_file = '/home/incubator/Work/kendryte-model-compiler/cfg/tiny_relu.cfg'
weights_file = '/home/incubator/Work/kendryte-model-compiler/weights/tiny.weights'
output_dir = '/home/incubator/Work/kendryte-model-compiler-git/output_dir/'


def decode_darknet(cfg_file, weights_file, output_dir):
    net1 = darknet_tool.darknet_network('network',
                                        cfg_file=cfg_file,
                                        weights_file=weights_file,
                                        dtype='float32')

    net1.net.statistcs_size(print_out=True)

    if os.path.exists(output_dir) and False:
        print('For security, I won\'t overwrite the existed directory.')
    else:
        tensorflow_tool.darknet_to_tf_module(net1, out_dir=output_dir)

decode_darknet(cfg_file, weights_file, output_dir)