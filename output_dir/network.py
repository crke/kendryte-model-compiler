"""
This .py module is compiled by Dark2T ver. 0.0.1
you use this .py code as a module to import, and embed it into your own tensorflow code
"""

# for storing weights
data_lib = dict()
end_node = dict()

# imports
# from d2t.tf_functions import *
from model_loader.darknet.D2T_lib.static_lib.tf_functions import *
import tensorflow as tf
from re import split

_mod_path_ = ''.join('%s/'%_s for _s in split(r'[/\\]', __file__)[:-1])
# ----------------------------------------- #
# global data type
_data_type_ = 'float32'

# read weights from .tfw file
_data_bytes_ = bytes_from_TFW(_mod_path_ + 'network.tfw')


def load_data():
	global data_lib
	# load pre-trained data here
	with tf.variable_scope('network','network_var'):
		data_lib['0_conv_b'] = var_from_bytes(_data_bytes_,0,64,
												None,False,'0_conv_b',_data_type_)
		data_lib['0_conv_s'] = var_from_bytes(_data_bytes_,64,128,
												None,False,'0_conv_s',_data_type_)
		data_lib['0_conv_m'] = var_from_bytes(_data_bytes_,128,192,
												None,False,'0_conv_m',_data_type_)
		data_lib['0_conv_v'] = var_from_bytes(_data_bytes_,192,256,
												None,False,'0_conv_v',_data_type_)
		data_lib['0_conv_w'] = var_from_bytes(_data_bytes_,256,1984,
												[3,3,3,16],False,'0_conv_w',_data_type_)
		
		
		data_lib['2_conv_b'] = var_from_bytes(_data_bytes_,1984,2112,
												None,False,'2_conv_b',_data_type_)
		data_lib['2_conv_s'] = var_from_bytes(_data_bytes_,2112,2240,
												None,False,'2_conv_s',_data_type_)
		data_lib['2_conv_m'] = var_from_bytes(_data_bytes_,2240,2368,
												None,False,'2_conv_m',_data_type_)
		data_lib['2_conv_v'] = var_from_bytes(_data_bytes_,2368,2496,
												None,False,'2_conv_v',_data_type_)
		data_lib['2_conv_w'] = var_from_bytes(_data_bytes_,2496,20928,
												[3,3,16,32],False,'2_conv_w',_data_type_)
		
		
		data_lib['4_conv_b'] = var_from_bytes(_data_bytes_,20928,20992,
												None,False,'4_conv_b',_data_type_)
		data_lib['4_conv_s'] = var_from_bytes(_data_bytes_,20992,21056,
												None,False,'4_conv_s',_data_type_)
		data_lib['4_conv_m'] = var_from_bytes(_data_bytes_,21056,21120,
												None,False,'4_conv_m',_data_type_)
		data_lib['4_conv_v'] = var_from_bytes(_data_bytes_,21120,21184,
												None,False,'4_conv_v',_data_type_)
		data_lib['4_conv_w'] = var_from_bytes(_data_bytes_,21184,23232,
												[1,1,32,16],False,'4_conv_w',_data_type_)
		
		
		data_lib['5_conv_b'] = var_from_bytes(_data_bytes_,23232,23744,
												None,False,'5_conv_b',_data_type_)
		data_lib['5_conv_s'] = var_from_bytes(_data_bytes_,23744,24256,
												None,False,'5_conv_s',_data_type_)
		data_lib['5_conv_m'] = var_from_bytes(_data_bytes_,24256,24768,
												None,False,'5_conv_m',_data_type_)
		data_lib['5_conv_v'] = var_from_bytes(_data_bytes_,24768,25280,
												None,False,'5_conv_v',_data_type_)
		data_lib['5_conv_w'] = var_from_bytes(_data_bytes_,25280,99008,
												[3,3,16,128],False,'5_conv_w',_data_type_)
		
		
		data_lib['6_conv_b'] = var_from_bytes(_data_bytes_,99008,99072,
												None,False,'6_conv_b',_data_type_)
		data_lib['6_conv_s'] = var_from_bytes(_data_bytes_,99072,99136,
												None,False,'6_conv_s',_data_type_)
		data_lib['6_conv_m'] = var_from_bytes(_data_bytes_,99136,99200,
												None,False,'6_conv_m',_data_type_)
		data_lib['6_conv_v'] = var_from_bytes(_data_bytes_,99200,99264,
												None,False,'6_conv_v',_data_type_)
		data_lib['6_conv_w'] = var_from_bytes(_data_bytes_,99264,107456,
												[1,1,128,16],False,'6_conv_w',_data_type_)
		
		
		data_lib['7_conv_b'] = var_from_bytes(_data_bytes_,107456,107968,
												None,False,'7_conv_b',_data_type_)
		data_lib['7_conv_s'] = var_from_bytes(_data_bytes_,107968,108480,
												None,False,'7_conv_s',_data_type_)
		data_lib['7_conv_m'] = var_from_bytes(_data_bytes_,108480,108992,
												None,False,'7_conv_m',_data_type_)
		data_lib['7_conv_v'] = var_from_bytes(_data_bytes_,108992,109504,
												None,False,'7_conv_v',_data_type_)
		data_lib['7_conv_w'] = var_from_bytes(_data_bytes_,109504,183232,
												[3,3,16,128],False,'7_conv_w',_data_type_)
		
		
		data_lib['9_conv_b'] = var_from_bytes(_data_bytes_,183232,183360,
												None,False,'9_conv_b',_data_type_)
		data_lib['9_conv_s'] = var_from_bytes(_data_bytes_,183360,183488,
												None,False,'9_conv_s',_data_type_)
		data_lib['9_conv_m'] = var_from_bytes(_data_bytes_,183488,183616,
												None,False,'9_conv_m',_data_type_)
		data_lib['9_conv_v'] = var_from_bytes(_data_bytes_,183616,183744,
												None,False,'9_conv_v',_data_type_)
		data_lib['9_conv_w'] = var_from_bytes(_data_bytes_,183744,200128,
												[1,1,128,32],False,'9_conv_w',_data_type_)
		
		
		data_lib['10_conv_b'] = var_from_bytes(_data_bytes_,200128,201152,
												None,False,'10_conv_b',_data_type_)
		data_lib['10_conv_s'] = var_from_bytes(_data_bytes_,201152,202176,
												None,False,'10_conv_s',_data_type_)
		data_lib['10_conv_m'] = var_from_bytes(_data_bytes_,202176,203200,
												None,False,'10_conv_m',_data_type_)
		data_lib['10_conv_v'] = var_from_bytes(_data_bytes_,203200,204224,
												None,False,'10_conv_v',_data_type_)
		data_lib['10_conv_w'] = var_from_bytes(_data_bytes_,204224,499136,
												[3,3,32,256],False,'10_conv_w',_data_type_)
		
		
		data_lib['11_conv_b'] = var_from_bytes(_data_bytes_,499136,499264,
												None,False,'11_conv_b',_data_type_)
		data_lib['11_conv_s'] = var_from_bytes(_data_bytes_,499264,499392,
												None,False,'11_conv_s',_data_type_)
		data_lib['11_conv_m'] = var_from_bytes(_data_bytes_,499392,499520,
												None,False,'11_conv_m',_data_type_)
		data_lib['11_conv_v'] = var_from_bytes(_data_bytes_,499520,499648,
												None,False,'11_conv_v',_data_type_)
		data_lib['11_conv_w'] = var_from_bytes(_data_bytes_,499648,532416,
												[1,1,256,32],False,'11_conv_w',_data_type_)
		
		
		data_lib['12_conv_b'] = var_from_bytes(_data_bytes_,532416,533440,
												None,False,'12_conv_b',_data_type_)
		data_lib['12_conv_s'] = var_from_bytes(_data_bytes_,533440,534464,
												None,False,'12_conv_s',_data_type_)
		data_lib['12_conv_m'] = var_from_bytes(_data_bytes_,534464,535488,
												None,False,'12_conv_m',_data_type_)
		data_lib['12_conv_v'] = var_from_bytes(_data_bytes_,535488,536512,
												None,False,'12_conv_v',_data_type_)
		data_lib['12_conv_w'] = var_from_bytes(_data_bytes_,536512,831424,
												[3,3,32,256],False,'12_conv_w',_data_type_)
		
		
		data_lib['14_conv_b'] = var_from_bytes(_data_bytes_,831424,831680,
												None,False,'14_conv_b',_data_type_)
		data_lib['14_conv_s'] = var_from_bytes(_data_bytes_,831680,831936,
												None,False,'14_conv_s',_data_type_)
		data_lib['14_conv_m'] = var_from_bytes(_data_bytes_,831936,832192,
												None,False,'14_conv_m',_data_type_)
		data_lib['14_conv_v'] = var_from_bytes(_data_bytes_,832192,832448,
												None,False,'14_conv_v',_data_type_)
		data_lib['14_conv_w'] = var_from_bytes(_data_bytes_,832448,897984,
												[1,1,256,64],False,'14_conv_w',_data_type_)
		
		
		data_lib['15_conv_b'] = var_from_bytes(_data_bytes_,897984,900032,
												None,False,'15_conv_b',_data_type_)
		data_lib['15_conv_s'] = var_from_bytes(_data_bytes_,900032,902080,
												None,False,'15_conv_s',_data_type_)
		data_lib['15_conv_m'] = var_from_bytes(_data_bytes_,902080,904128,
												None,False,'15_conv_m',_data_type_)
		data_lib['15_conv_v'] = var_from_bytes(_data_bytes_,904128,906176,
												None,False,'15_conv_v',_data_type_)
		data_lib['15_conv_w'] = var_from_bytes(_data_bytes_,906176,2085824,
												[3,3,64,512],False,'15_conv_w',_data_type_)
		
		
		data_lib['16_conv_b'] = var_from_bytes(_data_bytes_,2085824,2086080,
												None,False,'16_conv_b',_data_type_)
		data_lib['16_conv_s'] = var_from_bytes(_data_bytes_,2086080,2086336,
												None,False,'16_conv_s',_data_type_)
		data_lib['16_conv_m'] = var_from_bytes(_data_bytes_,2086336,2086592,
												None,False,'16_conv_m',_data_type_)
		data_lib['16_conv_v'] = var_from_bytes(_data_bytes_,2086592,2086848,
												None,False,'16_conv_v',_data_type_)
		data_lib['16_conv_w'] = var_from_bytes(_data_bytes_,2086848,2217920,
												[1,1,512,64],False,'16_conv_w',_data_type_)
		
		
		data_lib['17_conv_b'] = var_from_bytes(_data_bytes_,2217920,2219968,
												None,False,'17_conv_b',_data_type_)
		data_lib['17_conv_s'] = var_from_bytes(_data_bytes_,2219968,2222016,
												None,False,'17_conv_s',_data_type_)
		data_lib['17_conv_m'] = var_from_bytes(_data_bytes_,2222016,2224064,
												None,False,'17_conv_m',_data_type_)
		data_lib['17_conv_v'] = var_from_bytes(_data_bytes_,2224064,2226112,
												None,False,'17_conv_v',_data_type_)
		data_lib['17_conv_w'] = var_from_bytes(_data_bytes_,2226112,3405760,
												[3,3,64,512],False,'17_conv_w',_data_type_)
		
		
		data_lib['18_conv_b'] = var_from_bytes(_data_bytes_,3405760,3406272,
												None,False,'18_conv_b',_data_type_)
		data_lib['18_conv_s'] = var_from_bytes(_data_bytes_,3406272,3406784,
												None,False,'18_conv_s',_data_type_)
		data_lib['18_conv_m'] = var_from_bytes(_data_bytes_,3406784,3407296,
												None,False,'18_conv_m',_data_type_)
		data_lib['18_conv_v'] = var_from_bytes(_data_bytes_,3407296,3407808,
												None,False,'18_conv_v',_data_type_)
		data_lib['18_conv_w'] = var_from_bytes(_data_bytes_,3407808,3669952,
												[1,1,512,128],False,'18_conv_w',_data_type_)
		
		
def network_forward(input):
	global data_lib
	net = input

	net = convolutional(net, weights=data_lib['0_conv_w'], biases=data_lib['0_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['0_conv_s'],
	                  bn_mean = data_lib['0_conv_m'],
	                  bn_variance = data_lib['0_conv_v'],
	                  parent_scope = 'None', my_scope = '0_conv',
	                  reuse = False)
	end_node['0_conv'] = net # save node
	                  
	net = max_pool(net, ksize=2, strides=2, padding='SAME', scope='None', name = '1_maxpool')
	end_node['1_maxpool'] = net # save node
	
	net = convolutional(net, weights=data_lib['2_conv_w'], biases=data_lib['2_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['2_conv_s'],
	                  bn_mean = data_lib['2_conv_m'],
	                  bn_variance = data_lib['2_conv_v'],
	                  parent_scope = 'None', my_scope = '2_conv',
	                  reuse = False)
	end_node['2_conv'] = net # save node
	                  
	net = max_pool(net, ksize=2, strides=2, padding='SAME', scope='None', name = '3_maxpool')
	end_node['3_maxpool'] = net # save node
	
	net = convolutional(net, weights=data_lib['4_conv_w'], biases=data_lib['4_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['4_conv_s'],
	                  bn_mean = data_lib['4_conv_m'],
	                  bn_variance = data_lib['4_conv_v'],
	                  parent_scope = 'None', my_scope = '4_conv',
	                  reuse = False)
	end_node['4_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['5_conv_w'], biases=data_lib['5_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['5_conv_s'],
	                  bn_mean = data_lib['5_conv_m'],
	                  bn_variance = data_lib['5_conv_v'],
	                  parent_scope = 'None', my_scope = '5_conv',
	                  reuse = False)
	end_node['5_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['6_conv_w'], biases=data_lib['6_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['6_conv_s'],
	                  bn_mean = data_lib['6_conv_m'],
	                  bn_variance = data_lib['6_conv_v'],
	                  parent_scope = 'None', my_scope = '6_conv',
	                  reuse = False)
	end_node['6_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['7_conv_w'], biases=data_lib['7_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['7_conv_s'],
	                  bn_mean = data_lib['7_conv_m'],
	                  bn_variance = data_lib['7_conv_v'],
	                  parent_scope = 'None', my_scope = '7_conv',
	                  reuse = False)
	end_node['7_conv'] = net # save node
	                  
	net = max_pool(net, ksize=2, strides=2, padding='SAME', scope='None', name = '8_maxpool')
	end_node['8_maxpool'] = net # save node
	
	net = convolutional(net, weights=data_lib['9_conv_w'], biases=data_lib['9_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['9_conv_s'],
	                  bn_mean = data_lib['9_conv_m'],
	                  bn_variance = data_lib['9_conv_v'],
	                  parent_scope = 'None', my_scope = '9_conv',
	                  reuse = False)
	end_node['9_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['10_conv_w'], biases=data_lib['10_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['10_conv_s'],
	                  bn_mean = data_lib['10_conv_m'],
	                  bn_variance = data_lib['10_conv_v'],
	                  parent_scope = 'None', my_scope = '10_conv',
	                  reuse = False)
	end_node['10_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['11_conv_w'], biases=data_lib['11_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['11_conv_s'],
	                  bn_mean = data_lib['11_conv_m'],
	                  bn_variance = data_lib['11_conv_v'],
	                  parent_scope = 'None', my_scope = '11_conv',
	                  reuse = False)
	end_node['11_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['12_conv_w'], biases=data_lib['12_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['12_conv_s'],
	                  bn_mean = data_lib['12_conv_m'],
	                  bn_variance = data_lib['12_conv_v'],
	                  parent_scope = 'None', my_scope = '12_conv',
	                  reuse = False)
	end_node['12_conv'] = net # save node
	                  
	net = max_pool(net, ksize=2, strides=2, padding='SAME', scope='None', name = '13_maxpool')
	end_node['13_maxpool'] = net # save node
	
	net = convolutional(net, weights=data_lib['14_conv_w'], biases=data_lib['14_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['14_conv_s'],
	                  bn_mean = data_lib['14_conv_m'],
	                  bn_variance = data_lib['14_conv_v'],
	                  parent_scope = 'None', my_scope = '14_conv',
	                  reuse = False)
	end_node['14_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['15_conv_w'], biases=data_lib['15_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['15_conv_s'],
	                  bn_mean = data_lib['15_conv_m'],
	                  bn_variance = data_lib['15_conv_v'],
	                  parent_scope = 'None', my_scope = '15_conv',
	                  reuse = False)
	end_node['15_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['16_conv_w'], biases=data_lib['16_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['16_conv_s'],
	                  bn_mean = data_lib['16_conv_m'],
	                  bn_variance = data_lib['16_conv_v'],
	                  parent_scope = 'None', my_scope = '16_conv',
	                  reuse = False)
	end_node['16_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['17_conv_w'], biases=data_lib['17_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['17_conv_s'],
	                  bn_mean = data_lib['17_conv_m'],
	                  bn_variance = data_lib['17_conv_v'],
	                  parent_scope = 'None', my_scope = '17_conv',
	                  reuse = False)
	end_node['17_conv'] = net # save node
	                  
	net = convolutional(net, weights=data_lib['18_conv_w'], biases=data_lib['18_conv_b'],
	                  strides=1, padding='SAME',
	                  activation_fn = tf.nn.relu,
	                  batch_normalize = True,
	                  bn_scale = data_lib['18_conv_s'],
	                  bn_mean = data_lib['18_conv_m'],
	                  bn_variance = data_lib['18_conv_v'],
	                  parent_scope = 'None', my_scope = '18_conv',
	                  reuse = False)
	end_node['18_conv'] = net # save node
	                  
	return net
