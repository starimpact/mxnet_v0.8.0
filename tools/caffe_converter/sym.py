import mxnet as mx
data = mx.symbol.Variable(name='data')
slice_data = mx.symbol.SliceChannel(name='slice_data', data=data , num_outputs=6)
data_patch0 = slice_data[0]
data_patch1 = slice_data[1]
data_patch2 = slice_data[2]
data_patch3 = slice_data[3]
data_patch6 = slice_data[4]
data_patch7 = slice_data[5]
conv1_patch0 = mx.symbol.Convolution(name='conv1_patch0', data=data_patch0 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=False)
conv1_bn_patch0 = mx.symbol.BatchNorm(name='conv1_bn_patch0', data=conv1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_scale_patch0 = conv1_bn_patch0
conv1_relu_patch0 = mx.symbol.Activation(name='conv1_relu_patch0', data=conv1_scale_patch0 , act_type='relu')
pool1_patch0 = mx.symbol.Pooling(name='pool1_patch0', data=conv1_relu_patch0 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch0 = mx.symbol.Convolution(name='res2a_branch1_patch0', data=pool1_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch1_bn_patch0 = mx.symbol.BatchNorm(name='res2a_branch1_bn_patch0', data=res2a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch1_scale_patch0 = res2a_branch1_bn_patch0
res2a_branch2a_patch0 = mx.symbol.Convolution(name='res2a_branch2a_patch0', data=pool1_patch0 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res2a_branch2a_bn_patch0', data=res2a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2a_scale_patch0 = res2a_branch2a_bn_patch0
res2a_branch2a_relu_patch0 = mx.symbol.Activation(name='res2a_branch2a_relu_patch0', data=res2a_branch2a_scale_patch0 , act_type='relu')
res2a_branch2b_patch0 = mx.symbol.Convolution(name='res2a_branch2b_patch0', data=res2a_branch2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2a_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res2a_branch2b_bn_patch0', data=res2a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2b_scale_patch0 = res2a_branch2b_bn_patch0
res2a_branch2b_relu_patch0 = mx.symbol.Activation(name='res2a_branch2b_relu_patch0', data=res2a_branch2b_scale_patch0 , act_type='relu')
res2a_branch2c_patch0 = mx.symbol.Convolution(name='res2a_branch2c_patch0', data=res2a_branch2b_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res2a_branch2c_bn_patch0', data=res2a_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2c_scale_patch0 = res2a_branch2c_bn_patch0
res2a_patch0 = mx.symbol.broadcast_plus(name='res2a_patch0', *[res2a_branch1_scale_patch0,res2a_branch2c_scale_patch0] )
res2a_relu_patch0 = mx.symbol.Activation(name='res2a_relu_patch0', data=res2a_patch0 , act_type='relu')
res2b1_branch2a_patch0 = mx.symbol.Convolution(name='res2b1_branch2a_patch0', data=res2a_relu_patch0 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res2b1_branch2a_bn_patch0', data=res2b1_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2a_scale_patch0 = res2b1_branch2a_bn_patch0
res2b1_branch2a_relu_patch0 = mx.symbol.Activation(name='res2b1_branch2a_relu_patch0', data=res2b1_branch2a_scale_patch0 , act_type='relu')
res2b1_branch2b_patch0 = mx.symbol.Convolution(name='res2b1_branch2b_patch0', data=res2b1_branch2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b1_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res2b1_branch2b_bn_patch0', data=res2b1_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2b_scale_patch0 = res2b1_branch2b_bn_patch0
res2b1_branch2b_relu_patch0 = mx.symbol.Activation(name='res2b1_branch2b_relu_patch0', data=res2b1_branch2b_scale_patch0 , act_type='relu')
res2b1_branch2c_patch0 = mx.symbol.Convolution(name='res2b1_branch2c_patch0', data=res2b1_branch2b_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res2b1_branch2c_bn_patch0', data=res2b1_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2c_scale_patch0 = res2b1_branch2c_bn_patch0
res2b1_patch0 = mx.symbol.broadcast_plus(name='res2b1_patch0', *[res2a_relu_patch0,res2b1_branch2c_scale_patch0] )
res2b1_relu_patch0 = mx.symbol.Activation(name='res2b1_relu_patch0', data=res2b1_patch0 , act_type='relu')
res2b2_branch2a_patch0 = mx.symbol.Convolution(name='res2b2_branch2a_patch0', data=res2b1_relu_patch0 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res2b2_branch2a_bn_patch0', data=res2b2_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2a_scale_patch0 = res2b2_branch2a_bn_patch0
res2b2_branch2a_relu_patch0 = mx.symbol.Activation(name='res2b2_branch2a_relu_patch0', data=res2b2_branch2a_scale_patch0 , act_type='relu')
res2b2_branch2b_patch0 = mx.symbol.Convolution(name='res2b2_branch2b_patch0', data=res2b2_branch2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b2_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res2b2_branch2b_bn_patch0', data=res2b2_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2b_scale_patch0 = res2b2_branch2b_bn_patch0
res2b2_branch2b_relu_patch0 = mx.symbol.Activation(name='res2b2_branch2b_relu_patch0', data=res2b2_branch2b_scale_patch0 , act_type='relu')
res2b2_branch2c_patch0 = mx.symbol.Convolution(name='res2b2_branch2c_patch0', data=res2b2_branch2b_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res2b2_branch2c_bn_patch0', data=res2b2_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2c_scale_patch0 = res2b2_branch2c_bn_patch0
res2b2_patch0 = mx.symbol.broadcast_plus(name='res2b2_patch0', *[res2b1_relu_patch0,res2b2_branch2c_scale_patch0] )
res2b2_relu_patch0 = mx.symbol.Activation(name='res2b2_relu_patch0', data=res2b2_patch0 , act_type='relu')
res3a_branch1_patch0 = mx.symbol.Convolution(name='res3a_branch1_patch0', data=res2b2_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch1_bn_patch0 = mx.symbol.BatchNorm(name='res3a_branch1_bn_patch0', data=res3a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch1_scale_patch0 = res3a_branch1_bn_patch0
res3a_branch2a_patch0 = mx.symbol.Convolution(name='res3a_branch2a_patch0', data=res2b2_relu_patch0 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res3a_branch2a_bn_patch0', data=res3a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2a_scale_patch0 = res3a_branch2a_bn_patch0
res3a_branch2a_relu_patch0 = mx.symbol.Activation(name='res3a_branch2a_relu_patch0', data=res3a_branch2a_scale_patch0 , act_type='relu')
res3a_branch2b_patch0 = mx.symbol.Convolution(name='res3a_branch2b_patch0', data=res3a_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3a_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res3a_branch2b_bn_patch0', data=res3a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2b_scale_patch0 = res3a_branch2b_bn_patch0
res3a_branch2b_relu_patch0 = mx.symbol.Activation(name='res3a_branch2b_relu_patch0', data=res3a_branch2b_scale_patch0 , act_type='relu')
res3a_branch2c_patch0 = mx.symbol.Convolution(name='res3a_branch2c_patch0', data=res3a_branch2b_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3a_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res3a_branch2c_bn_patch0', data=res3a_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2c_scale_patch0 = res3a_branch2c_bn_patch0
res3a_patch0 = mx.symbol.broadcast_plus(name='res3a_patch0', *[res3a_branch1_scale_patch0,res3a_branch2c_scale_patch0] )
res3a_relu_patch0 = mx.symbol.Activation(name='res3a_relu_patch0', data=res3a_patch0 , act_type='relu')
res3b1_branch2a_patch0 = mx.symbol.Convolution(name='res3b1_branch2a_patch0', data=res3a_relu_patch0 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res3b1_branch2a_bn_patch0', data=res3b1_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2a_scale_patch0 = res3b1_branch2a_bn_patch0
res3b1_branch2a_relu_patch0 = mx.symbol.Activation(name='res3b1_branch2a_relu_patch0', data=res3b1_branch2a_scale_patch0 , act_type='relu')
res3b1_branch2b_patch0 = mx.symbol.Convolution(name='res3b1_branch2b_patch0', data=res3b1_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b1_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res3b1_branch2b_bn_patch0', data=res3b1_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2b_scale_patch0 = res3b1_branch2b_bn_patch0
res3b1_branch2b_relu_patch0 = mx.symbol.Activation(name='res3b1_branch2b_relu_patch0', data=res3b1_branch2b_scale_patch0 , act_type='relu')
res3b1_branch2c_patch0 = mx.symbol.Convolution(name='res3b1_branch2c_patch0', data=res3b1_branch2b_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res3b1_branch2c_bn_patch0', data=res3b1_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2c_scale_patch0 = res3b1_branch2c_bn_patch0
res3b1_patch0 = mx.symbol.broadcast_plus(name='res3b1_patch0', *[res3a_relu_patch0,res3b1_branch2c_scale_patch0] )
res3b1_relu_patch0 = mx.symbol.Activation(name='res3b1_relu_patch0', data=res3b1_patch0 , act_type='relu')
res3b2_branch2a_patch0 = mx.symbol.Convolution(name='res3b2_branch2a_patch0', data=res3b1_relu_patch0 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res3b2_branch2a_bn_patch0', data=res3b2_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2a_scale_patch0 = res3b2_branch2a_bn_patch0
res3b2_branch2a_relu_patch0 = mx.symbol.Activation(name='res3b2_branch2a_relu_patch0', data=res3b2_branch2a_scale_patch0 , act_type='relu')
res3b2_branch2b_patch0 = mx.symbol.Convolution(name='res3b2_branch2b_patch0', data=res3b2_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b2_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res3b2_branch2b_bn_patch0', data=res3b2_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2b_scale_patch0 = res3b2_branch2b_bn_patch0
res3b2_branch2b_relu_patch0 = mx.symbol.Activation(name='res3b2_branch2b_relu_patch0', data=res3b2_branch2b_scale_patch0 , act_type='relu')
res3b2_branch2c_patch0 = mx.symbol.Convolution(name='res3b2_branch2c_patch0', data=res3b2_branch2b_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res3b2_branch2c_bn_patch0', data=res3b2_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2c_scale_patch0 = res3b2_branch2c_bn_patch0
res3b2_patch0 = mx.symbol.broadcast_plus(name='res3b2_patch0', *[res3b1_relu_patch0,res3b2_branch2c_scale_patch0] )
res3b2_relu_patch0 = mx.symbol.Activation(name='res3b2_relu_patch0', data=res3b2_patch0 , act_type='relu')
res3b3_branch2a_patch0 = mx.symbol.Convolution(name='res3b3_branch2a_patch0', data=res3b2_relu_patch0 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res3b3_branch2a_bn_patch0', data=res3b3_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2a_scale_patch0 = res3b3_branch2a_bn_patch0
res3b3_branch2a_relu_patch0 = mx.symbol.Activation(name='res3b3_branch2a_relu_patch0', data=res3b3_branch2a_scale_patch0 , act_type='relu')
res3b3_branch2b_patch0 = mx.symbol.Convolution(name='res3b3_branch2b_patch0', data=res3b3_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b3_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res3b3_branch2b_bn_patch0', data=res3b3_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2b_scale_patch0 = res3b3_branch2b_bn_patch0
res3b3_branch2b_relu_patch0 = mx.symbol.Activation(name='res3b3_branch2b_relu_patch0', data=res3b3_branch2b_scale_patch0 , act_type='relu')
res3b3_branch2c_patch0 = mx.symbol.Convolution(name='res3b3_branch2c_patch0', data=res3b3_branch2b_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res3b3_branch2c_bn_patch0', data=res3b3_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2c_scale_patch0 = res3b3_branch2c_bn_patch0
res3b3_patch0 = mx.symbol.broadcast_plus(name='res3b3_patch0', *[res3b2_relu_patch0,res3b3_branch2c_scale_patch0] )
res3b3_relu_patch0 = mx.symbol.Activation(name='res3b3_relu_patch0', data=res3b3_patch0 , act_type='relu')
res4a_branch1_patch0 = mx.symbol.Convolution(name='res4a_branch1_patch0', data=res3b3_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch1_bn_patch0 = mx.symbol.BatchNorm(name='res4a_branch1_bn_patch0', data=res4a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch1_scale_patch0 = res4a_branch1_bn_patch0
res4a_branch2a_patch0 = mx.symbol.Convolution(name='res4a_branch2a_patch0', data=res3b3_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res4a_branch2a_bn_patch0', data=res4a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2a_scale_patch0 = res4a_branch2a_bn_patch0
res4a_branch2a_relu_patch0 = mx.symbol.Activation(name='res4a_branch2a_relu_patch0', data=res4a_branch2a_scale_patch0 , act_type='relu')
res4a_branch2b_patch0 = mx.symbol.Convolution(name='res4a_branch2b_patch0', data=res4a_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4a_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res4a_branch2b_bn_patch0', data=res4a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2b_scale_patch0 = res4a_branch2b_bn_patch0
res4a_branch2b_relu_patch0 = mx.symbol.Activation(name='res4a_branch2b_relu_patch0', data=res4a_branch2b_scale_patch0 , act_type='relu')
res4a_branch2c_patch0 = mx.symbol.Convolution(name='res4a_branch2c_patch0', data=res4a_branch2b_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4a_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res4a_branch2c_bn_patch0', data=res4a_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2c_scale_patch0 = res4a_branch2c_bn_patch0
res4a_patch0 = mx.symbol.broadcast_plus(name='res4a_patch0', *[res4a_branch1_scale_patch0,res4a_branch2c_scale_patch0] )
res4a_relu_patch0 = mx.symbol.Activation(name='res4a_relu_patch0', data=res4a_patch0 , act_type='relu')
res4b1_branch2a_patch0 = mx.symbol.Convolution(name='res4b1_branch2a_patch0', data=res4a_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res4b1_branch2a_bn_patch0', data=res4b1_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2a_scale_patch0 = res4b1_branch2a_bn_patch0
res4b1_branch2a_relu_patch0 = mx.symbol.Activation(name='res4b1_branch2a_relu_patch0', data=res4b1_branch2a_scale_patch0 , act_type='relu')
res4b1_branch2b_patch0 = mx.symbol.Convolution(name='res4b1_branch2b_patch0', data=res4b1_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b1_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res4b1_branch2b_bn_patch0', data=res4b1_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2b_scale_patch0 = res4b1_branch2b_bn_patch0
res4b1_branch2b_relu_patch0 = mx.symbol.Activation(name='res4b1_branch2b_relu_patch0', data=res4b1_branch2b_scale_patch0 , act_type='relu')
res4b1_branch2c_patch0 = mx.symbol.Convolution(name='res4b1_branch2c_patch0', data=res4b1_branch2b_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res4b1_branch2c_bn_patch0', data=res4b1_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2c_scale_patch0 = res4b1_branch2c_bn_patch0
res4b1_patch0 = mx.symbol.broadcast_plus(name='res4b1_patch0', *[res4a_relu_patch0,res4b1_branch2c_scale_patch0] )
res4b1_relu_patch0 = mx.symbol.Activation(name='res4b1_relu_patch0', data=res4b1_patch0 , act_type='relu')
res4b2_branch2a_patch0 = mx.symbol.Convolution(name='res4b2_branch2a_patch0', data=res4b1_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res4b2_branch2a_bn_patch0', data=res4b2_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2a_scale_patch0 = res4b2_branch2a_bn_patch0
res4b2_branch2a_relu_patch0 = mx.symbol.Activation(name='res4b2_branch2a_relu_patch0', data=res4b2_branch2a_scale_patch0 , act_type='relu')
res4b2_branch2b_patch0 = mx.symbol.Convolution(name='res4b2_branch2b_patch0', data=res4b2_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b2_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res4b2_branch2b_bn_patch0', data=res4b2_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2b_scale_patch0 = res4b2_branch2b_bn_patch0
res4b2_branch2b_relu_patch0 = mx.symbol.Activation(name='res4b2_branch2b_relu_patch0', data=res4b2_branch2b_scale_patch0 , act_type='relu')
res4b2_branch2c_patch0 = mx.symbol.Convolution(name='res4b2_branch2c_patch0', data=res4b2_branch2b_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res4b2_branch2c_bn_patch0', data=res4b2_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2c_scale_patch0 = res4b2_branch2c_bn_patch0
res4b2_patch0 = mx.symbol.broadcast_plus(name='res4b2_patch0', *[res4b1_relu_patch0,res4b2_branch2c_scale_patch0] )
res4b2_relu_patch0 = mx.symbol.Activation(name='res4b2_relu_patch0', data=res4b2_patch0 , act_type='relu')
res4b3_branch2a_patch0 = mx.symbol.Convolution(name='res4b3_branch2a_patch0', data=res4b2_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res4b3_branch2a_bn_patch0', data=res4b3_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2a_scale_patch0 = res4b3_branch2a_bn_patch0
res4b3_branch2a_relu_patch0 = mx.symbol.Activation(name='res4b3_branch2a_relu_patch0', data=res4b3_branch2a_scale_patch0 , act_type='relu')
res4b3_branch2b_patch0 = mx.symbol.Convolution(name='res4b3_branch2b_patch0', data=res4b3_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b3_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res4b3_branch2b_bn_patch0', data=res4b3_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2b_scale_patch0 = res4b3_branch2b_bn_patch0
res4b3_branch2b_relu_patch0 = mx.symbol.Activation(name='res4b3_branch2b_relu_patch0', data=res4b3_branch2b_scale_patch0 , act_type='relu')
res4b3_branch2c_patch0 = mx.symbol.Convolution(name='res4b3_branch2c_patch0', data=res4b3_branch2b_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res4b3_branch2c_bn_patch0', data=res4b3_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2c_scale_patch0 = res4b3_branch2c_bn_patch0
res4b3_patch0 = mx.symbol.broadcast_plus(name='res4b3_patch0', *[res4b2_relu_patch0,res4b3_branch2c_scale_patch0] )
res4b3_relu_patch0 = mx.symbol.Activation(name='res4b3_relu_patch0', data=res4b3_patch0 , act_type='relu')
res4b4_branch2a_patch0 = mx.symbol.Convolution(name='res4b4_branch2a_patch0', data=res4b3_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res4b4_branch2a_bn_patch0', data=res4b4_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2a_scale_patch0 = res4b4_branch2a_bn_patch0
res4b4_branch2a_relu_patch0 = mx.symbol.Activation(name='res4b4_branch2a_relu_patch0', data=res4b4_branch2a_scale_patch0 , act_type='relu')
res4b4_branch2b_patch0 = mx.symbol.Convolution(name='res4b4_branch2b_patch0', data=res4b4_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b4_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res4b4_branch2b_bn_patch0', data=res4b4_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2b_scale_patch0 = res4b4_branch2b_bn_patch0
res4b4_branch2b_relu_patch0 = mx.symbol.Activation(name='res4b4_branch2b_relu_patch0', data=res4b4_branch2b_scale_patch0 , act_type='relu')
res4b4_branch2c_patch0 = mx.symbol.Convolution(name='res4b4_branch2c_patch0', data=res4b4_branch2b_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res4b4_branch2c_bn_patch0', data=res4b4_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2c_scale_patch0 = res4b4_branch2c_bn_patch0
res4b4_patch0 = mx.symbol.broadcast_plus(name='res4b4_patch0', *[res4b3_relu_patch0,res4b4_branch2c_scale_patch0] )
res4b4_relu_patch0 = mx.symbol.Activation(name='res4b4_relu_patch0', data=res4b4_patch0 , act_type='relu')
res4b5_branch2a_patch0 = mx.symbol.Convolution(name='res4b5_branch2a_patch0', data=res4b4_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res4b5_branch2a_bn_patch0', data=res4b5_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2a_scale_patch0 = res4b5_branch2a_bn_patch0
res4b5_branch2a_relu_patch0 = mx.symbol.Activation(name='res4b5_branch2a_relu_patch0', data=res4b5_branch2a_scale_patch0 , act_type='relu')
res4b5_branch2b_patch0 = mx.symbol.Convolution(name='res4b5_branch2b_patch0', data=res4b5_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b5_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res4b5_branch2b_bn_patch0', data=res4b5_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2b_scale_patch0 = res4b5_branch2b_bn_patch0
res4b5_branch2b_relu_patch0 = mx.symbol.Activation(name='res4b5_branch2b_relu_patch0', data=res4b5_branch2b_scale_patch0 , act_type='relu')
res4b5_branch2c_patch0 = mx.symbol.Convolution(name='res4b5_branch2c_patch0', data=res4b5_branch2b_relu_patch0 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res4b5_branch2c_bn_patch0', data=res4b5_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2c_scale_patch0 = res4b5_branch2c_bn_patch0
res4b5_patch0 = mx.symbol.broadcast_plus(name='res4b5_patch0', *[res4b4_relu_patch0,res4b5_branch2c_scale_patch0] )
res4b5_relu_patch0 = mx.symbol.Activation(name='res4b5_relu_patch0', data=res4b5_patch0 , act_type='relu')
res5a_branch1_patch0 = mx.symbol.Convolution(name='res5a_branch1_patch0', data=res4b5_relu_patch0 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch1_bn_patch0 = mx.symbol.BatchNorm(name='res5a_branch1_bn_patch0', data=res5a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch1_scale_patch0 = res5a_branch1_bn_patch0
res5a_branch2a_patch0 = mx.symbol.Convolution(name='res5a_branch2a_patch0', data=res4b5_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res5a_branch2a_bn_patch0', data=res5a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2a_scale_patch0 = res5a_branch2a_bn_patch0
res5a_branch2a_relu_patch0 = mx.symbol.Activation(name='res5a_branch2a_relu_patch0', data=res5a_branch2a_scale_patch0 , act_type='relu')
res5a_branch2b_patch0 = mx.symbol.Convolution(name='res5a_branch2b_patch0', data=res5a_branch2a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5a_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res5a_branch2b_bn_patch0', data=res5a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2b_scale_patch0 = res5a_branch2b_bn_patch0
res5a_branch2b_relu_patch0 = mx.symbol.Activation(name='res5a_branch2b_relu_patch0', data=res5a_branch2b_scale_patch0 , act_type='relu')
res5a_branch2c_patch0 = mx.symbol.Convolution(name='res5a_branch2c_patch0', data=res5a_branch2b_relu_patch0 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5a_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res5a_branch2c_bn_patch0', data=res5a_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2c_scale_patch0 = res5a_branch2c_bn_patch0
res5a_patch0 = mx.symbol.broadcast_plus(name='res5a_patch0', *[res5a_branch1_scale_patch0,res5a_branch2c_scale_patch0] )
res5a_relu_patch0 = mx.symbol.Activation(name='res5a_relu_patch0', data=res5a_patch0 , act_type='relu')
res5b1_branch2a_patch0 = mx.symbol.Convolution(name='res5b1_branch2a_patch0', data=res5a_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res5b1_branch2a_bn_patch0', data=res5b1_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2a_scale_patch0 = res5b1_branch2a_bn_patch0
res5b1_branch2a_relu_patch0 = mx.symbol.Activation(name='res5b1_branch2a_relu_patch0', data=res5b1_branch2a_scale_patch0 , act_type='relu')
res5b1_branch2b_patch0 = mx.symbol.Convolution(name='res5b1_branch2b_patch0', data=res5b1_branch2a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b1_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res5b1_branch2b_bn_patch0', data=res5b1_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2b_scale_patch0 = res5b1_branch2b_bn_patch0
res5b1_branch2b_relu_patch0 = mx.symbol.Activation(name='res5b1_branch2b_relu_patch0', data=res5b1_branch2b_scale_patch0 , act_type='relu')
res5b1_branch2c_patch0 = mx.symbol.Convolution(name='res5b1_branch2c_patch0', data=res5b1_branch2b_relu_patch0 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res5b1_branch2c_bn_patch0', data=res5b1_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2c_scale_patch0 = res5b1_branch2c_bn_patch0
res5b1_patch0 = mx.symbol.broadcast_plus(name='res5b1_patch0', *[res5a_relu_patch0,res5b1_branch2c_scale_patch0] )
res5b1_relu_patch0 = mx.symbol.Activation(name='res5b1_relu_patch0', data=res5b1_patch0 , act_type='relu')
res5b2_branch2a_patch0 = mx.symbol.Convolution(name='res5b2_branch2a_patch0', data=res5b1_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2a_bn_patch0 = mx.symbol.BatchNorm(name='res5b2_branch2a_bn_patch0', data=res5b2_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2a_scale_patch0 = res5b2_branch2a_bn_patch0
res5b2_branch2a_relu_patch0 = mx.symbol.Activation(name='res5b2_branch2a_relu_patch0', data=res5b2_branch2a_scale_patch0 , act_type='relu')
res5b2_branch2b_patch0 = mx.symbol.Convolution(name='res5b2_branch2b_patch0', data=res5b2_branch2a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b2_branch2b_bn_patch0 = mx.symbol.BatchNorm(name='res5b2_branch2b_bn_patch0', data=res5b2_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2b_scale_patch0 = res5b2_branch2b_bn_patch0
res5b2_branch2b_relu_patch0 = mx.symbol.Activation(name='res5b2_branch2b_relu_patch0', data=res5b2_branch2b_scale_patch0 , act_type='relu')
res5b2_branch2c_patch0 = mx.symbol.Convolution(name='res5b2_branch2c_patch0', data=res5b2_branch2b_relu_patch0 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2c_bn_patch0 = mx.symbol.BatchNorm(name='res5b2_branch2c_bn_patch0', data=res5b2_branch2c_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2c_scale_patch0 = res5b2_branch2c_bn_patch0
res5b2_patch0 = mx.symbol.broadcast_plus(name='res5b2_patch0', *[res5b1_relu_patch0,res5b2_branch2c_scale_patch0] )
res5b2_relu_patch0 = mx.symbol.Activation(name='res5b2_relu_patch0', data=res5b2_patch0 , act_type='relu')
reduce_conv_patch0 = mx.symbol.Convolution(name='reduce_conv_patch0', data=res5b2_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduce_conv_bn_patch0 = mx.symbol.BatchNorm(name='reduce_conv_bn_patch0', data=reduce_conv_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
reduce_conv_scale_patch0 = reduce_conv_bn_patch0
pool5_patch0 = mx.symbol.Pooling(name='pool5_patch0', data=reduce_conv_scale_patch0 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch1 = mx.symbol.Convolution(name='conv1_patch1', data=data_patch1 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=False)
conv1_bn_patch1 = mx.symbol.BatchNorm(name='conv1_bn_patch1', data=conv1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_scale_patch1 = conv1_bn_patch1
conv1_relu_patch1 = mx.symbol.Activation(name='conv1_relu_patch1', data=conv1_scale_patch1 , act_type='relu')
pool1_patch1 = mx.symbol.Pooling(name='pool1_patch1', data=conv1_relu_patch1 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch1 = mx.symbol.Convolution(name='res2a_branch1_patch1', data=pool1_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch1_bn_patch1 = mx.symbol.BatchNorm(name='res2a_branch1_bn_patch1', data=res2a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch1_scale_patch1 = res2a_branch1_bn_patch1
res2a_branch2a_patch1 = mx.symbol.Convolution(name='res2a_branch2a_patch1', data=pool1_patch1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res2a_branch2a_bn_patch1', data=res2a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2a_scale_patch1 = res2a_branch2a_bn_patch1
res2a_branch2a_relu_patch1 = mx.symbol.Activation(name='res2a_branch2a_relu_patch1', data=res2a_branch2a_scale_patch1 , act_type='relu')
res2a_branch2b_patch1 = mx.symbol.Convolution(name='res2a_branch2b_patch1', data=res2a_branch2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2a_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res2a_branch2b_bn_patch1', data=res2a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2b_scale_patch1 = res2a_branch2b_bn_patch1
res2a_branch2b_relu_patch1 = mx.symbol.Activation(name='res2a_branch2b_relu_patch1', data=res2a_branch2b_scale_patch1 , act_type='relu')
res2a_branch2c_patch1 = mx.symbol.Convolution(name='res2a_branch2c_patch1', data=res2a_branch2b_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res2a_branch2c_bn_patch1', data=res2a_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2c_scale_patch1 = res2a_branch2c_bn_patch1
res2a_patch1 = mx.symbol.broadcast_plus(name='res2a_patch1', *[res2a_branch1_scale_patch1,res2a_branch2c_scale_patch1] )
res2a_relu_patch1 = mx.symbol.Activation(name='res2a_relu_patch1', data=res2a_patch1 , act_type='relu')
res2b1_branch2a_patch1 = mx.symbol.Convolution(name='res2b1_branch2a_patch1', data=res2a_relu_patch1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res2b1_branch2a_bn_patch1', data=res2b1_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2a_scale_patch1 = res2b1_branch2a_bn_patch1
res2b1_branch2a_relu_patch1 = mx.symbol.Activation(name='res2b1_branch2a_relu_patch1', data=res2b1_branch2a_scale_patch1 , act_type='relu')
res2b1_branch2b_patch1 = mx.symbol.Convolution(name='res2b1_branch2b_patch1', data=res2b1_branch2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b1_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res2b1_branch2b_bn_patch1', data=res2b1_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2b_scale_patch1 = res2b1_branch2b_bn_patch1
res2b1_branch2b_relu_patch1 = mx.symbol.Activation(name='res2b1_branch2b_relu_patch1', data=res2b1_branch2b_scale_patch1 , act_type='relu')
res2b1_branch2c_patch1 = mx.symbol.Convolution(name='res2b1_branch2c_patch1', data=res2b1_branch2b_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res2b1_branch2c_bn_patch1', data=res2b1_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2c_scale_patch1 = res2b1_branch2c_bn_patch1
res2b1_patch1 = mx.symbol.broadcast_plus(name='res2b1_patch1', *[res2a_relu_patch1,res2b1_branch2c_scale_patch1] )
res2b1_relu_patch1 = mx.symbol.Activation(name='res2b1_relu_patch1', data=res2b1_patch1 , act_type='relu')
res2b2_branch2a_patch1 = mx.symbol.Convolution(name='res2b2_branch2a_patch1', data=res2b1_relu_patch1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res2b2_branch2a_bn_patch1', data=res2b2_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2a_scale_patch1 = res2b2_branch2a_bn_patch1
res2b2_branch2a_relu_patch1 = mx.symbol.Activation(name='res2b2_branch2a_relu_patch1', data=res2b2_branch2a_scale_patch1 , act_type='relu')
res2b2_branch2b_patch1 = mx.symbol.Convolution(name='res2b2_branch2b_patch1', data=res2b2_branch2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b2_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res2b2_branch2b_bn_patch1', data=res2b2_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2b_scale_patch1 = res2b2_branch2b_bn_patch1
res2b2_branch2b_relu_patch1 = mx.symbol.Activation(name='res2b2_branch2b_relu_patch1', data=res2b2_branch2b_scale_patch1 , act_type='relu')
res2b2_branch2c_patch1 = mx.symbol.Convolution(name='res2b2_branch2c_patch1', data=res2b2_branch2b_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res2b2_branch2c_bn_patch1', data=res2b2_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2c_scale_patch1 = res2b2_branch2c_bn_patch1
res2b2_patch1 = mx.symbol.broadcast_plus(name='res2b2_patch1', *[res2b1_relu_patch1,res2b2_branch2c_scale_patch1] )
res2b2_relu_patch1 = mx.symbol.Activation(name='res2b2_relu_patch1', data=res2b2_patch1 , act_type='relu')
res3a_branch1_patch1 = mx.symbol.Convolution(name='res3a_branch1_patch1', data=res2b2_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch1_bn_patch1 = mx.symbol.BatchNorm(name='res3a_branch1_bn_patch1', data=res3a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch1_scale_patch1 = res3a_branch1_bn_patch1
res3a_branch2a_patch1 = mx.symbol.Convolution(name='res3a_branch2a_patch1', data=res2b2_relu_patch1 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res3a_branch2a_bn_patch1', data=res3a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2a_scale_patch1 = res3a_branch2a_bn_patch1
res3a_branch2a_relu_patch1 = mx.symbol.Activation(name='res3a_branch2a_relu_patch1', data=res3a_branch2a_scale_patch1 , act_type='relu')
res3a_branch2b_patch1 = mx.symbol.Convolution(name='res3a_branch2b_patch1', data=res3a_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3a_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res3a_branch2b_bn_patch1', data=res3a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2b_scale_patch1 = res3a_branch2b_bn_patch1
res3a_branch2b_relu_patch1 = mx.symbol.Activation(name='res3a_branch2b_relu_patch1', data=res3a_branch2b_scale_patch1 , act_type='relu')
res3a_branch2c_patch1 = mx.symbol.Convolution(name='res3a_branch2c_patch1', data=res3a_branch2b_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3a_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res3a_branch2c_bn_patch1', data=res3a_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2c_scale_patch1 = res3a_branch2c_bn_patch1
res3a_patch1 = mx.symbol.broadcast_plus(name='res3a_patch1', *[res3a_branch1_scale_patch1,res3a_branch2c_scale_patch1] )
res3a_relu_patch1 = mx.symbol.Activation(name='res3a_relu_patch1', data=res3a_patch1 , act_type='relu')
res3b1_branch2a_patch1 = mx.symbol.Convolution(name='res3b1_branch2a_patch1', data=res3a_relu_patch1 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res3b1_branch2a_bn_patch1', data=res3b1_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2a_scale_patch1 = res3b1_branch2a_bn_patch1
res3b1_branch2a_relu_patch1 = mx.symbol.Activation(name='res3b1_branch2a_relu_patch1', data=res3b1_branch2a_scale_patch1 , act_type='relu')
res3b1_branch2b_patch1 = mx.symbol.Convolution(name='res3b1_branch2b_patch1', data=res3b1_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b1_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res3b1_branch2b_bn_patch1', data=res3b1_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2b_scale_patch1 = res3b1_branch2b_bn_patch1
res3b1_branch2b_relu_patch1 = mx.symbol.Activation(name='res3b1_branch2b_relu_patch1', data=res3b1_branch2b_scale_patch1 , act_type='relu')
res3b1_branch2c_patch1 = mx.symbol.Convolution(name='res3b1_branch2c_patch1', data=res3b1_branch2b_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res3b1_branch2c_bn_patch1', data=res3b1_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2c_scale_patch1 = res3b1_branch2c_bn_patch1
res3b1_patch1 = mx.symbol.broadcast_plus(name='res3b1_patch1', *[res3a_relu_patch1,res3b1_branch2c_scale_patch1] )
res3b1_relu_patch1 = mx.symbol.Activation(name='res3b1_relu_patch1', data=res3b1_patch1 , act_type='relu')
res3b2_branch2a_patch1 = mx.symbol.Convolution(name='res3b2_branch2a_patch1', data=res3b1_relu_patch1 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res3b2_branch2a_bn_patch1', data=res3b2_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2a_scale_patch1 = res3b2_branch2a_bn_patch1
res3b2_branch2a_relu_patch1 = mx.symbol.Activation(name='res3b2_branch2a_relu_patch1', data=res3b2_branch2a_scale_patch1 , act_type='relu')
res3b2_branch2b_patch1 = mx.symbol.Convolution(name='res3b2_branch2b_patch1', data=res3b2_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b2_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res3b2_branch2b_bn_patch1', data=res3b2_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2b_scale_patch1 = res3b2_branch2b_bn_patch1
res3b2_branch2b_relu_patch1 = mx.symbol.Activation(name='res3b2_branch2b_relu_patch1', data=res3b2_branch2b_scale_patch1 , act_type='relu')
res3b2_branch2c_patch1 = mx.symbol.Convolution(name='res3b2_branch2c_patch1', data=res3b2_branch2b_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res3b2_branch2c_bn_patch1', data=res3b2_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2c_scale_patch1 = res3b2_branch2c_bn_patch1
res3b2_patch1 = mx.symbol.broadcast_plus(name='res3b2_patch1', *[res3b1_relu_patch1,res3b2_branch2c_scale_patch1] )
res3b2_relu_patch1 = mx.symbol.Activation(name='res3b2_relu_patch1', data=res3b2_patch1 , act_type='relu')
res3b3_branch2a_patch1 = mx.symbol.Convolution(name='res3b3_branch2a_patch1', data=res3b2_relu_patch1 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res3b3_branch2a_bn_patch1', data=res3b3_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2a_scale_patch1 = res3b3_branch2a_bn_patch1
res3b3_branch2a_relu_patch1 = mx.symbol.Activation(name='res3b3_branch2a_relu_patch1', data=res3b3_branch2a_scale_patch1 , act_type='relu')
res3b3_branch2b_patch1 = mx.symbol.Convolution(name='res3b3_branch2b_patch1', data=res3b3_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b3_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res3b3_branch2b_bn_patch1', data=res3b3_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2b_scale_patch1 = res3b3_branch2b_bn_patch1
res3b3_branch2b_relu_patch1 = mx.symbol.Activation(name='res3b3_branch2b_relu_patch1', data=res3b3_branch2b_scale_patch1 , act_type='relu')
res3b3_branch2c_patch1 = mx.symbol.Convolution(name='res3b3_branch2c_patch1', data=res3b3_branch2b_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res3b3_branch2c_bn_patch1', data=res3b3_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2c_scale_patch1 = res3b3_branch2c_bn_patch1
res3b3_patch1 = mx.symbol.broadcast_plus(name='res3b3_patch1', *[res3b2_relu_patch1,res3b3_branch2c_scale_patch1] )
res3b3_relu_patch1 = mx.symbol.Activation(name='res3b3_relu_patch1', data=res3b3_patch1 , act_type='relu')
res4a_branch1_patch1 = mx.symbol.Convolution(name='res4a_branch1_patch1', data=res3b3_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch1_bn_patch1 = mx.symbol.BatchNorm(name='res4a_branch1_bn_patch1', data=res4a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch1_scale_patch1 = res4a_branch1_bn_patch1
res4a_branch2a_patch1 = mx.symbol.Convolution(name='res4a_branch2a_patch1', data=res3b3_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res4a_branch2a_bn_patch1', data=res4a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2a_scale_patch1 = res4a_branch2a_bn_patch1
res4a_branch2a_relu_patch1 = mx.symbol.Activation(name='res4a_branch2a_relu_patch1', data=res4a_branch2a_scale_patch1 , act_type='relu')
res4a_branch2b_patch1 = mx.symbol.Convolution(name='res4a_branch2b_patch1', data=res4a_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4a_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res4a_branch2b_bn_patch1', data=res4a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2b_scale_patch1 = res4a_branch2b_bn_patch1
res4a_branch2b_relu_patch1 = mx.symbol.Activation(name='res4a_branch2b_relu_patch1', data=res4a_branch2b_scale_patch1 , act_type='relu')
res4a_branch2c_patch1 = mx.symbol.Convolution(name='res4a_branch2c_patch1', data=res4a_branch2b_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4a_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res4a_branch2c_bn_patch1', data=res4a_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2c_scale_patch1 = res4a_branch2c_bn_patch1
res4a_patch1 = mx.symbol.broadcast_plus(name='res4a_patch1', *[res4a_branch1_scale_patch1,res4a_branch2c_scale_patch1] )
res4a_relu_patch1 = mx.symbol.Activation(name='res4a_relu_patch1', data=res4a_patch1 , act_type='relu')
res4b1_branch2a_patch1 = mx.symbol.Convolution(name='res4b1_branch2a_patch1', data=res4a_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res4b1_branch2a_bn_patch1', data=res4b1_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2a_scale_patch1 = res4b1_branch2a_bn_patch1
res4b1_branch2a_relu_patch1 = mx.symbol.Activation(name='res4b1_branch2a_relu_patch1', data=res4b1_branch2a_scale_patch1 , act_type='relu')
res4b1_branch2b_patch1 = mx.symbol.Convolution(name='res4b1_branch2b_patch1', data=res4b1_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b1_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res4b1_branch2b_bn_patch1', data=res4b1_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2b_scale_patch1 = res4b1_branch2b_bn_patch1
res4b1_branch2b_relu_patch1 = mx.symbol.Activation(name='res4b1_branch2b_relu_patch1', data=res4b1_branch2b_scale_patch1 , act_type='relu')
res4b1_branch2c_patch1 = mx.symbol.Convolution(name='res4b1_branch2c_patch1', data=res4b1_branch2b_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res4b1_branch2c_bn_patch1', data=res4b1_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2c_scale_patch1 = res4b1_branch2c_bn_patch1
res4b1_patch1 = mx.symbol.broadcast_plus(name='res4b1_patch1', *[res4a_relu_patch1,res4b1_branch2c_scale_patch1] )
res4b1_relu_patch1 = mx.symbol.Activation(name='res4b1_relu_patch1', data=res4b1_patch1 , act_type='relu')
res4b2_branch2a_patch1 = mx.symbol.Convolution(name='res4b2_branch2a_patch1', data=res4b1_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res4b2_branch2a_bn_patch1', data=res4b2_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2a_scale_patch1 = res4b2_branch2a_bn_patch1
res4b2_branch2a_relu_patch1 = mx.symbol.Activation(name='res4b2_branch2a_relu_patch1', data=res4b2_branch2a_scale_patch1 , act_type='relu')
res4b2_branch2b_patch1 = mx.symbol.Convolution(name='res4b2_branch2b_patch1', data=res4b2_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b2_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res4b2_branch2b_bn_patch1', data=res4b2_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2b_scale_patch1 = res4b2_branch2b_bn_patch1
res4b2_branch2b_relu_patch1 = mx.symbol.Activation(name='res4b2_branch2b_relu_patch1', data=res4b2_branch2b_scale_patch1 , act_type='relu')
res4b2_branch2c_patch1 = mx.symbol.Convolution(name='res4b2_branch2c_patch1', data=res4b2_branch2b_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res4b2_branch2c_bn_patch1', data=res4b2_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2c_scale_patch1 = res4b2_branch2c_bn_patch1
res4b2_patch1 = mx.symbol.broadcast_plus(name='res4b2_patch1', *[res4b1_relu_patch1,res4b2_branch2c_scale_patch1] )
res4b2_relu_patch1 = mx.symbol.Activation(name='res4b2_relu_patch1', data=res4b2_patch1 , act_type='relu')
res4b3_branch2a_patch1 = mx.symbol.Convolution(name='res4b3_branch2a_patch1', data=res4b2_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res4b3_branch2a_bn_patch1', data=res4b3_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2a_scale_patch1 = res4b3_branch2a_bn_patch1
res4b3_branch2a_relu_patch1 = mx.symbol.Activation(name='res4b3_branch2a_relu_patch1', data=res4b3_branch2a_scale_patch1 , act_type='relu')
res4b3_branch2b_patch1 = mx.symbol.Convolution(name='res4b3_branch2b_patch1', data=res4b3_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b3_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res4b3_branch2b_bn_patch1', data=res4b3_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2b_scale_patch1 = res4b3_branch2b_bn_patch1
res4b3_branch2b_relu_patch1 = mx.symbol.Activation(name='res4b3_branch2b_relu_patch1', data=res4b3_branch2b_scale_patch1 , act_type='relu')
res4b3_branch2c_patch1 = mx.symbol.Convolution(name='res4b3_branch2c_patch1', data=res4b3_branch2b_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res4b3_branch2c_bn_patch1', data=res4b3_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2c_scale_patch1 = res4b3_branch2c_bn_patch1
res4b3_patch1 = mx.symbol.broadcast_plus(name='res4b3_patch1', *[res4b2_relu_patch1,res4b3_branch2c_scale_patch1] )
res4b3_relu_patch1 = mx.symbol.Activation(name='res4b3_relu_patch1', data=res4b3_patch1 , act_type='relu')
res4b4_branch2a_patch1 = mx.symbol.Convolution(name='res4b4_branch2a_patch1', data=res4b3_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res4b4_branch2a_bn_patch1', data=res4b4_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2a_scale_patch1 = res4b4_branch2a_bn_patch1
res4b4_branch2a_relu_patch1 = mx.symbol.Activation(name='res4b4_branch2a_relu_patch1', data=res4b4_branch2a_scale_patch1 , act_type='relu')
res4b4_branch2b_patch1 = mx.symbol.Convolution(name='res4b4_branch2b_patch1', data=res4b4_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b4_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res4b4_branch2b_bn_patch1', data=res4b4_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2b_scale_patch1 = res4b4_branch2b_bn_patch1
res4b4_branch2b_relu_patch1 = mx.symbol.Activation(name='res4b4_branch2b_relu_patch1', data=res4b4_branch2b_scale_patch1 , act_type='relu')
res4b4_branch2c_patch1 = mx.symbol.Convolution(name='res4b4_branch2c_patch1', data=res4b4_branch2b_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res4b4_branch2c_bn_patch1', data=res4b4_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2c_scale_patch1 = res4b4_branch2c_bn_patch1
res4b4_patch1 = mx.symbol.broadcast_plus(name='res4b4_patch1', *[res4b3_relu_patch1,res4b4_branch2c_scale_patch1] )
res4b4_relu_patch1 = mx.symbol.Activation(name='res4b4_relu_patch1', data=res4b4_patch1 , act_type='relu')
res4b5_branch2a_patch1 = mx.symbol.Convolution(name='res4b5_branch2a_patch1', data=res4b4_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res4b5_branch2a_bn_patch1', data=res4b5_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2a_scale_patch1 = res4b5_branch2a_bn_patch1
res4b5_branch2a_relu_patch1 = mx.symbol.Activation(name='res4b5_branch2a_relu_patch1', data=res4b5_branch2a_scale_patch1 , act_type='relu')
res4b5_branch2b_patch1 = mx.symbol.Convolution(name='res4b5_branch2b_patch1', data=res4b5_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b5_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res4b5_branch2b_bn_patch1', data=res4b5_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2b_scale_patch1 = res4b5_branch2b_bn_patch1
res4b5_branch2b_relu_patch1 = mx.symbol.Activation(name='res4b5_branch2b_relu_patch1', data=res4b5_branch2b_scale_patch1 , act_type='relu')
res4b5_branch2c_patch1 = mx.symbol.Convolution(name='res4b5_branch2c_patch1', data=res4b5_branch2b_relu_patch1 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res4b5_branch2c_bn_patch1', data=res4b5_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2c_scale_patch1 = res4b5_branch2c_bn_patch1
res4b5_patch1 = mx.symbol.broadcast_plus(name='res4b5_patch1', *[res4b4_relu_patch1,res4b5_branch2c_scale_patch1] )
res4b5_relu_patch1 = mx.symbol.Activation(name='res4b5_relu_patch1', data=res4b5_patch1 , act_type='relu')
res5a_branch1_patch1 = mx.symbol.Convolution(name='res5a_branch1_patch1', data=res4b5_relu_patch1 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch1_bn_patch1 = mx.symbol.BatchNorm(name='res5a_branch1_bn_patch1', data=res5a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch1_scale_patch1 = res5a_branch1_bn_patch1
res5a_branch2a_patch1 = mx.symbol.Convolution(name='res5a_branch2a_patch1', data=res4b5_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res5a_branch2a_bn_patch1', data=res5a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2a_scale_patch1 = res5a_branch2a_bn_patch1
res5a_branch2a_relu_patch1 = mx.symbol.Activation(name='res5a_branch2a_relu_patch1', data=res5a_branch2a_scale_patch1 , act_type='relu')
res5a_branch2b_patch1 = mx.symbol.Convolution(name='res5a_branch2b_patch1', data=res5a_branch2a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5a_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res5a_branch2b_bn_patch1', data=res5a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2b_scale_patch1 = res5a_branch2b_bn_patch1
res5a_branch2b_relu_patch1 = mx.symbol.Activation(name='res5a_branch2b_relu_patch1', data=res5a_branch2b_scale_patch1 , act_type='relu')
res5a_branch2c_patch1 = mx.symbol.Convolution(name='res5a_branch2c_patch1', data=res5a_branch2b_relu_patch1 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5a_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res5a_branch2c_bn_patch1', data=res5a_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2c_scale_patch1 = res5a_branch2c_bn_patch1
res5a_patch1 = mx.symbol.broadcast_plus(name='res5a_patch1', *[res5a_branch1_scale_patch1,res5a_branch2c_scale_patch1] )
res5a_relu_patch1 = mx.symbol.Activation(name='res5a_relu_patch1', data=res5a_patch1 , act_type='relu')
res5b1_branch2a_patch1 = mx.symbol.Convolution(name='res5b1_branch2a_patch1', data=res5a_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res5b1_branch2a_bn_patch1', data=res5b1_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2a_scale_patch1 = res5b1_branch2a_bn_patch1
res5b1_branch2a_relu_patch1 = mx.symbol.Activation(name='res5b1_branch2a_relu_patch1', data=res5b1_branch2a_scale_patch1 , act_type='relu')
res5b1_branch2b_patch1 = mx.symbol.Convolution(name='res5b1_branch2b_patch1', data=res5b1_branch2a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b1_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res5b1_branch2b_bn_patch1', data=res5b1_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2b_scale_patch1 = res5b1_branch2b_bn_patch1
res5b1_branch2b_relu_patch1 = mx.symbol.Activation(name='res5b1_branch2b_relu_patch1', data=res5b1_branch2b_scale_patch1 , act_type='relu')
res5b1_branch2c_patch1 = mx.symbol.Convolution(name='res5b1_branch2c_patch1', data=res5b1_branch2b_relu_patch1 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res5b1_branch2c_bn_patch1', data=res5b1_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2c_scale_patch1 = res5b1_branch2c_bn_patch1
res5b1_patch1 = mx.symbol.broadcast_plus(name='res5b1_patch1', *[res5a_relu_patch1,res5b1_branch2c_scale_patch1] )
res5b1_relu_patch1 = mx.symbol.Activation(name='res5b1_relu_patch1', data=res5b1_patch1 , act_type='relu')
res5b2_branch2a_patch1 = mx.symbol.Convolution(name='res5b2_branch2a_patch1', data=res5b1_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2a_bn_patch1 = mx.symbol.BatchNorm(name='res5b2_branch2a_bn_patch1', data=res5b2_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2a_scale_patch1 = res5b2_branch2a_bn_patch1
res5b2_branch2a_relu_patch1 = mx.symbol.Activation(name='res5b2_branch2a_relu_patch1', data=res5b2_branch2a_scale_patch1 , act_type='relu')
res5b2_branch2b_patch1 = mx.symbol.Convolution(name='res5b2_branch2b_patch1', data=res5b2_branch2a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b2_branch2b_bn_patch1 = mx.symbol.BatchNorm(name='res5b2_branch2b_bn_patch1', data=res5b2_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2b_scale_patch1 = res5b2_branch2b_bn_patch1
res5b2_branch2b_relu_patch1 = mx.symbol.Activation(name='res5b2_branch2b_relu_patch1', data=res5b2_branch2b_scale_patch1 , act_type='relu')
res5b2_branch2c_patch1 = mx.symbol.Convolution(name='res5b2_branch2c_patch1', data=res5b2_branch2b_relu_patch1 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2c_bn_patch1 = mx.symbol.BatchNorm(name='res5b2_branch2c_bn_patch1', data=res5b2_branch2c_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2c_scale_patch1 = res5b2_branch2c_bn_patch1
res5b2_patch1 = mx.symbol.broadcast_plus(name='res5b2_patch1', *[res5b1_relu_patch1,res5b2_branch2c_scale_patch1] )
res5b2_relu_patch1 = mx.symbol.Activation(name='res5b2_relu_patch1', data=res5b2_patch1 , act_type='relu')
reduce_conv_patch1 = mx.symbol.Convolution(name='reduce_conv_patch1', data=res5b2_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduce_conv_bn_patch1 = mx.symbol.BatchNorm(name='reduce_conv_bn_patch1', data=reduce_conv_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
reduce_conv_scale_patch1 = reduce_conv_bn_patch1
pool5_patch1 = mx.symbol.Pooling(name='pool5_patch1', data=reduce_conv_scale_patch1 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch2 = mx.symbol.Convolution(name='conv1_patch2', data=data_patch2 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=False)
conv1_bn_patch2 = mx.symbol.BatchNorm(name='conv1_bn_patch2', data=conv1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_scale_patch2 = conv1_bn_patch2
conv1_relu_patch2 = mx.symbol.Activation(name='conv1_relu_patch2', data=conv1_scale_patch2 , act_type='relu')
pool1_patch2 = mx.symbol.Pooling(name='pool1_patch2', data=conv1_relu_patch2 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch2 = mx.symbol.Convolution(name='res2a_branch1_patch2', data=pool1_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch1_bn_patch2 = mx.symbol.BatchNorm(name='res2a_branch1_bn_patch2', data=res2a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch1_scale_patch2 = res2a_branch1_bn_patch2
res2a_branch2a_patch2 = mx.symbol.Convolution(name='res2a_branch2a_patch2', data=pool1_patch2 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res2a_branch2a_bn_patch2', data=res2a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2a_scale_patch2 = res2a_branch2a_bn_patch2
res2a_branch2a_relu_patch2 = mx.symbol.Activation(name='res2a_branch2a_relu_patch2', data=res2a_branch2a_scale_patch2 , act_type='relu')
res2a_branch2b_patch2 = mx.symbol.Convolution(name='res2a_branch2b_patch2', data=res2a_branch2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2a_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res2a_branch2b_bn_patch2', data=res2a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2b_scale_patch2 = res2a_branch2b_bn_patch2
res2a_branch2b_relu_patch2 = mx.symbol.Activation(name='res2a_branch2b_relu_patch2', data=res2a_branch2b_scale_patch2 , act_type='relu')
res2a_branch2c_patch2 = mx.symbol.Convolution(name='res2a_branch2c_patch2', data=res2a_branch2b_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res2a_branch2c_bn_patch2', data=res2a_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2c_scale_patch2 = res2a_branch2c_bn_patch2
res2a_patch2 = mx.symbol.broadcast_plus(name='res2a_patch2', *[res2a_branch1_scale_patch2,res2a_branch2c_scale_patch2] )
res2a_relu_patch2 = mx.symbol.Activation(name='res2a_relu_patch2', data=res2a_patch2 , act_type='relu')
res2b1_branch2a_patch2 = mx.symbol.Convolution(name='res2b1_branch2a_patch2', data=res2a_relu_patch2 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res2b1_branch2a_bn_patch2', data=res2b1_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2a_scale_patch2 = res2b1_branch2a_bn_patch2
res2b1_branch2a_relu_patch2 = mx.symbol.Activation(name='res2b1_branch2a_relu_patch2', data=res2b1_branch2a_scale_patch2 , act_type='relu')
res2b1_branch2b_patch2 = mx.symbol.Convolution(name='res2b1_branch2b_patch2', data=res2b1_branch2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b1_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res2b1_branch2b_bn_patch2', data=res2b1_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2b_scale_patch2 = res2b1_branch2b_bn_patch2
res2b1_branch2b_relu_patch2 = mx.symbol.Activation(name='res2b1_branch2b_relu_patch2', data=res2b1_branch2b_scale_patch2 , act_type='relu')
res2b1_branch2c_patch2 = mx.symbol.Convolution(name='res2b1_branch2c_patch2', data=res2b1_branch2b_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res2b1_branch2c_bn_patch2', data=res2b1_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2c_scale_patch2 = res2b1_branch2c_bn_patch2
res2b1_patch2 = mx.symbol.broadcast_plus(name='res2b1_patch2', *[res2a_relu_patch2,res2b1_branch2c_scale_patch2] )
res2b1_relu_patch2 = mx.symbol.Activation(name='res2b1_relu_patch2', data=res2b1_patch2 , act_type='relu')
res2b2_branch2a_patch2 = mx.symbol.Convolution(name='res2b2_branch2a_patch2', data=res2b1_relu_patch2 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res2b2_branch2a_bn_patch2', data=res2b2_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2a_scale_patch2 = res2b2_branch2a_bn_patch2
res2b2_branch2a_relu_patch2 = mx.symbol.Activation(name='res2b2_branch2a_relu_patch2', data=res2b2_branch2a_scale_patch2 , act_type='relu')
res2b2_branch2b_patch2 = mx.symbol.Convolution(name='res2b2_branch2b_patch2', data=res2b2_branch2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b2_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res2b2_branch2b_bn_patch2', data=res2b2_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2b_scale_patch2 = res2b2_branch2b_bn_patch2
res2b2_branch2b_relu_patch2 = mx.symbol.Activation(name='res2b2_branch2b_relu_patch2', data=res2b2_branch2b_scale_patch2 , act_type='relu')
res2b2_branch2c_patch2 = mx.symbol.Convolution(name='res2b2_branch2c_patch2', data=res2b2_branch2b_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res2b2_branch2c_bn_patch2', data=res2b2_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2c_scale_patch2 = res2b2_branch2c_bn_patch2
res2b2_patch2 = mx.symbol.broadcast_plus(name='res2b2_patch2', *[res2b1_relu_patch2,res2b2_branch2c_scale_patch2] )
res2b2_relu_patch2 = mx.symbol.Activation(name='res2b2_relu_patch2', data=res2b2_patch2 , act_type='relu')
res3a_branch1_patch2 = mx.symbol.Convolution(name='res3a_branch1_patch2', data=res2b2_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch1_bn_patch2 = mx.symbol.BatchNorm(name='res3a_branch1_bn_patch2', data=res3a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch1_scale_patch2 = res3a_branch1_bn_patch2
res3a_branch2a_patch2 = mx.symbol.Convolution(name='res3a_branch2a_patch2', data=res2b2_relu_patch2 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res3a_branch2a_bn_patch2', data=res3a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2a_scale_patch2 = res3a_branch2a_bn_patch2
res3a_branch2a_relu_patch2 = mx.symbol.Activation(name='res3a_branch2a_relu_patch2', data=res3a_branch2a_scale_patch2 , act_type='relu')
res3a_branch2b_patch2 = mx.symbol.Convolution(name='res3a_branch2b_patch2', data=res3a_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3a_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res3a_branch2b_bn_patch2', data=res3a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2b_scale_patch2 = res3a_branch2b_bn_patch2
res3a_branch2b_relu_patch2 = mx.symbol.Activation(name='res3a_branch2b_relu_patch2', data=res3a_branch2b_scale_patch2 , act_type='relu')
res3a_branch2c_patch2 = mx.symbol.Convolution(name='res3a_branch2c_patch2', data=res3a_branch2b_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3a_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res3a_branch2c_bn_patch2', data=res3a_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2c_scale_patch2 = res3a_branch2c_bn_patch2
res3a_patch2 = mx.symbol.broadcast_plus(name='res3a_patch2', *[res3a_branch1_scale_patch2,res3a_branch2c_scale_patch2] )
res3a_relu_patch2 = mx.symbol.Activation(name='res3a_relu_patch2', data=res3a_patch2 , act_type='relu')
res3b1_branch2a_patch2 = mx.symbol.Convolution(name='res3b1_branch2a_patch2', data=res3a_relu_patch2 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res3b1_branch2a_bn_patch2', data=res3b1_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2a_scale_patch2 = res3b1_branch2a_bn_patch2
res3b1_branch2a_relu_patch2 = mx.symbol.Activation(name='res3b1_branch2a_relu_patch2', data=res3b1_branch2a_scale_patch2 , act_type='relu')
res3b1_branch2b_patch2 = mx.symbol.Convolution(name='res3b1_branch2b_patch2', data=res3b1_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b1_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res3b1_branch2b_bn_patch2', data=res3b1_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2b_scale_patch2 = res3b1_branch2b_bn_patch2
res3b1_branch2b_relu_patch2 = mx.symbol.Activation(name='res3b1_branch2b_relu_patch2', data=res3b1_branch2b_scale_patch2 , act_type='relu')
res3b1_branch2c_patch2 = mx.symbol.Convolution(name='res3b1_branch2c_patch2', data=res3b1_branch2b_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res3b1_branch2c_bn_patch2', data=res3b1_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2c_scale_patch2 = res3b1_branch2c_bn_patch2
res3b1_patch2 = mx.symbol.broadcast_plus(name='res3b1_patch2', *[res3a_relu_patch2,res3b1_branch2c_scale_patch2] )
res3b1_relu_patch2 = mx.symbol.Activation(name='res3b1_relu_patch2', data=res3b1_patch2 , act_type='relu')
res3b2_branch2a_patch2 = mx.symbol.Convolution(name='res3b2_branch2a_patch2', data=res3b1_relu_patch2 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res3b2_branch2a_bn_patch2', data=res3b2_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2a_scale_patch2 = res3b2_branch2a_bn_patch2
res3b2_branch2a_relu_patch2 = mx.symbol.Activation(name='res3b2_branch2a_relu_patch2', data=res3b2_branch2a_scale_patch2 , act_type='relu')
res3b2_branch2b_patch2 = mx.symbol.Convolution(name='res3b2_branch2b_patch2', data=res3b2_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b2_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res3b2_branch2b_bn_patch2', data=res3b2_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2b_scale_patch2 = res3b2_branch2b_bn_patch2
res3b2_branch2b_relu_patch2 = mx.symbol.Activation(name='res3b2_branch2b_relu_patch2', data=res3b2_branch2b_scale_patch2 , act_type='relu')
res3b2_branch2c_patch2 = mx.symbol.Convolution(name='res3b2_branch2c_patch2', data=res3b2_branch2b_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res3b2_branch2c_bn_patch2', data=res3b2_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2c_scale_patch2 = res3b2_branch2c_bn_patch2
res3b2_patch2 = mx.symbol.broadcast_plus(name='res3b2_patch2', *[res3b1_relu_patch2,res3b2_branch2c_scale_patch2] )
res3b2_relu_patch2 = mx.symbol.Activation(name='res3b2_relu_patch2', data=res3b2_patch2 , act_type='relu')
res3b3_branch2a_patch2 = mx.symbol.Convolution(name='res3b3_branch2a_patch2', data=res3b2_relu_patch2 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res3b3_branch2a_bn_patch2', data=res3b3_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2a_scale_patch2 = res3b3_branch2a_bn_patch2
res3b3_branch2a_relu_patch2 = mx.symbol.Activation(name='res3b3_branch2a_relu_patch2', data=res3b3_branch2a_scale_patch2 , act_type='relu')
res3b3_branch2b_patch2 = mx.symbol.Convolution(name='res3b3_branch2b_patch2', data=res3b3_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b3_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res3b3_branch2b_bn_patch2', data=res3b3_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2b_scale_patch2 = res3b3_branch2b_bn_patch2
res3b3_branch2b_relu_patch2 = mx.symbol.Activation(name='res3b3_branch2b_relu_patch2', data=res3b3_branch2b_scale_patch2 , act_type='relu')
res3b3_branch2c_patch2 = mx.symbol.Convolution(name='res3b3_branch2c_patch2', data=res3b3_branch2b_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res3b3_branch2c_bn_patch2', data=res3b3_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2c_scale_patch2 = res3b3_branch2c_bn_patch2
res3b3_patch2 = mx.symbol.broadcast_plus(name='res3b3_patch2', *[res3b2_relu_patch2,res3b3_branch2c_scale_patch2] )
res3b3_relu_patch2 = mx.symbol.Activation(name='res3b3_relu_patch2', data=res3b3_patch2 , act_type='relu')
res4a_branch1_patch2 = mx.symbol.Convolution(name='res4a_branch1_patch2', data=res3b3_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch1_bn_patch2 = mx.symbol.BatchNorm(name='res4a_branch1_bn_patch2', data=res4a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch1_scale_patch2 = res4a_branch1_bn_patch2
res4a_branch2a_patch2 = mx.symbol.Convolution(name='res4a_branch2a_patch2', data=res3b3_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res4a_branch2a_bn_patch2', data=res4a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2a_scale_patch2 = res4a_branch2a_bn_patch2
res4a_branch2a_relu_patch2 = mx.symbol.Activation(name='res4a_branch2a_relu_patch2', data=res4a_branch2a_scale_patch2 , act_type='relu')
res4a_branch2b_patch2 = mx.symbol.Convolution(name='res4a_branch2b_patch2', data=res4a_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4a_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res4a_branch2b_bn_patch2', data=res4a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2b_scale_patch2 = res4a_branch2b_bn_patch2
res4a_branch2b_relu_patch2 = mx.symbol.Activation(name='res4a_branch2b_relu_patch2', data=res4a_branch2b_scale_patch2 , act_type='relu')
res4a_branch2c_patch2 = mx.symbol.Convolution(name='res4a_branch2c_patch2', data=res4a_branch2b_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4a_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res4a_branch2c_bn_patch2', data=res4a_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2c_scale_patch2 = res4a_branch2c_bn_patch2
res4a_patch2 = mx.symbol.broadcast_plus(name='res4a_patch2', *[res4a_branch1_scale_patch2,res4a_branch2c_scale_patch2] )
res4a_relu_patch2 = mx.symbol.Activation(name='res4a_relu_patch2', data=res4a_patch2 , act_type='relu')
res4b1_branch2a_patch2 = mx.symbol.Convolution(name='res4b1_branch2a_patch2', data=res4a_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res4b1_branch2a_bn_patch2', data=res4b1_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2a_scale_patch2 = res4b1_branch2a_bn_patch2
res4b1_branch2a_relu_patch2 = mx.symbol.Activation(name='res4b1_branch2a_relu_patch2', data=res4b1_branch2a_scale_patch2 , act_type='relu')
res4b1_branch2b_patch2 = mx.symbol.Convolution(name='res4b1_branch2b_patch2', data=res4b1_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b1_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res4b1_branch2b_bn_patch2', data=res4b1_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2b_scale_patch2 = res4b1_branch2b_bn_patch2
res4b1_branch2b_relu_patch2 = mx.symbol.Activation(name='res4b1_branch2b_relu_patch2', data=res4b1_branch2b_scale_patch2 , act_type='relu')
res4b1_branch2c_patch2 = mx.symbol.Convolution(name='res4b1_branch2c_patch2', data=res4b1_branch2b_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res4b1_branch2c_bn_patch2', data=res4b1_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2c_scale_patch2 = res4b1_branch2c_bn_patch2
res4b1_patch2 = mx.symbol.broadcast_plus(name='res4b1_patch2', *[res4a_relu_patch2,res4b1_branch2c_scale_patch2] )
res4b1_relu_patch2 = mx.symbol.Activation(name='res4b1_relu_patch2', data=res4b1_patch2 , act_type='relu')
res4b2_branch2a_patch2 = mx.symbol.Convolution(name='res4b2_branch2a_patch2', data=res4b1_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res4b2_branch2a_bn_patch2', data=res4b2_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2a_scale_patch2 = res4b2_branch2a_bn_patch2
res4b2_branch2a_relu_patch2 = mx.symbol.Activation(name='res4b2_branch2a_relu_patch2', data=res4b2_branch2a_scale_patch2 , act_type='relu')
res4b2_branch2b_patch2 = mx.symbol.Convolution(name='res4b2_branch2b_patch2', data=res4b2_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b2_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res4b2_branch2b_bn_patch2', data=res4b2_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2b_scale_patch2 = res4b2_branch2b_bn_patch2
res4b2_branch2b_relu_patch2 = mx.symbol.Activation(name='res4b2_branch2b_relu_patch2', data=res4b2_branch2b_scale_patch2 , act_type='relu')
res4b2_branch2c_patch2 = mx.symbol.Convolution(name='res4b2_branch2c_patch2', data=res4b2_branch2b_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res4b2_branch2c_bn_patch2', data=res4b2_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2c_scale_patch2 = res4b2_branch2c_bn_patch2
res4b2_patch2 = mx.symbol.broadcast_plus(name='res4b2_patch2', *[res4b1_relu_patch2,res4b2_branch2c_scale_patch2] )
res4b2_relu_patch2 = mx.symbol.Activation(name='res4b2_relu_patch2', data=res4b2_patch2 , act_type='relu')
res4b3_branch2a_patch2 = mx.symbol.Convolution(name='res4b3_branch2a_patch2', data=res4b2_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res4b3_branch2a_bn_patch2', data=res4b3_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2a_scale_patch2 = res4b3_branch2a_bn_patch2
res4b3_branch2a_relu_patch2 = mx.symbol.Activation(name='res4b3_branch2a_relu_patch2', data=res4b3_branch2a_scale_patch2 , act_type='relu')
res4b3_branch2b_patch2 = mx.symbol.Convolution(name='res4b3_branch2b_patch2', data=res4b3_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b3_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res4b3_branch2b_bn_patch2', data=res4b3_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2b_scale_patch2 = res4b3_branch2b_bn_patch2
res4b3_branch2b_relu_patch2 = mx.symbol.Activation(name='res4b3_branch2b_relu_patch2', data=res4b3_branch2b_scale_patch2 , act_type='relu')
res4b3_branch2c_patch2 = mx.symbol.Convolution(name='res4b3_branch2c_patch2', data=res4b3_branch2b_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res4b3_branch2c_bn_patch2', data=res4b3_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2c_scale_patch2 = res4b3_branch2c_bn_patch2
res4b3_patch2 = mx.symbol.broadcast_plus(name='res4b3_patch2', *[res4b2_relu_patch2,res4b3_branch2c_scale_patch2] )
res4b3_relu_patch2 = mx.symbol.Activation(name='res4b3_relu_patch2', data=res4b3_patch2 , act_type='relu')
res4b4_branch2a_patch2 = mx.symbol.Convolution(name='res4b4_branch2a_patch2', data=res4b3_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res4b4_branch2a_bn_patch2', data=res4b4_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2a_scale_patch2 = res4b4_branch2a_bn_patch2
res4b4_branch2a_relu_patch2 = mx.symbol.Activation(name='res4b4_branch2a_relu_patch2', data=res4b4_branch2a_scale_patch2 , act_type='relu')
res4b4_branch2b_patch2 = mx.symbol.Convolution(name='res4b4_branch2b_patch2', data=res4b4_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b4_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res4b4_branch2b_bn_patch2', data=res4b4_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2b_scale_patch2 = res4b4_branch2b_bn_patch2
res4b4_branch2b_relu_patch2 = mx.symbol.Activation(name='res4b4_branch2b_relu_patch2', data=res4b4_branch2b_scale_patch2 , act_type='relu')
res4b4_branch2c_patch2 = mx.symbol.Convolution(name='res4b4_branch2c_patch2', data=res4b4_branch2b_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res4b4_branch2c_bn_patch2', data=res4b4_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2c_scale_patch2 = res4b4_branch2c_bn_patch2
res4b4_patch2 = mx.symbol.broadcast_plus(name='res4b4_patch2', *[res4b3_relu_patch2,res4b4_branch2c_scale_patch2] )
res4b4_relu_patch2 = mx.symbol.Activation(name='res4b4_relu_patch2', data=res4b4_patch2 , act_type='relu')
res4b5_branch2a_patch2 = mx.symbol.Convolution(name='res4b5_branch2a_patch2', data=res4b4_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res4b5_branch2a_bn_patch2', data=res4b5_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2a_scale_patch2 = res4b5_branch2a_bn_patch2
res4b5_branch2a_relu_patch2 = mx.symbol.Activation(name='res4b5_branch2a_relu_patch2', data=res4b5_branch2a_scale_patch2 , act_type='relu')
res4b5_branch2b_patch2 = mx.symbol.Convolution(name='res4b5_branch2b_patch2', data=res4b5_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b5_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res4b5_branch2b_bn_patch2', data=res4b5_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2b_scale_patch2 = res4b5_branch2b_bn_patch2
res4b5_branch2b_relu_patch2 = mx.symbol.Activation(name='res4b5_branch2b_relu_patch2', data=res4b5_branch2b_scale_patch2 , act_type='relu')
res4b5_branch2c_patch2 = mx.symbol.Convolution(name='res4b5_branch2c_patch2', data=res4b5_branch2b_relu_patch2 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res4b5_branch2c_bn_patch2', data=res4b5_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2c_scale_patch2 = res4b5_branch2c_bn_patch2
res4b5_patch2 = mx.symbol.broadcast_plus(name='res4b5_patch2', *[res4b4_relu_patch2,res4b5_branch2c_scale_patch2] )
res4b5_relu_patch2 = mx.symbol.Activation(name='res4b5_relu_patch2', data=res4b5_patch2 , act_type='relu')
res5a_branch1_patch2 = mx.symbol.Convolution(name='res5a_branch1_patch2', data=res4b5_relu_patch2 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch1_bn_patch2 = mx.symbol.BatchNorm(name='res5a_branch1_bn_patch2', data=res5a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch1_scale_patch2 = res5a_branch1_bn_patch2
res5a_branch2a_patch2 = mx.symbol.Convolution(name='res5a_branch2a_patch2', data=res4b5_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res5a_branch2a_bn_patch2', data=res5a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2a_scale_patch2 = res5a_branch2a_bn_patch2
res5a_branch2a_relu_patch2 = mx.symbol.Activation(name='res5a_branch2a_relu_patch2', data=res5a_branch2a_scale_patch2 , act_type='relu')
res5a_branch2b_patch2 = mx.symbol.Convolution(name='res5a_branch2b_patch2', data=res5a_branch2a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5a_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res5a_branch2b_bn_patch2', data=res5a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2b_scale_patch2 = res5a_branch2b_bn_patch2
res5a_branch2b_relu_patch2 = mx.symbol.Activation(name='res5a_branch2b_relu_patch2', data=res5a_branch2b_scale_patch2 , act_type='relu')
res5a_branch2c_patch2 = mx.symbol.Convolution(name='res5a_branch2c_patch2', data=res5a_branch2b_relu_patch2 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5a_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res5a_branch2c_bn_patch2', data=res5a_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2c_scale_patch2 = res5a_branch2c_bn_patch2
res5a_patch2 = mx.symbol.broadcast_plus(name='res5a_patch2', *[res5a_branch1_scale_patch2,res5a_branch2c_scale_patch2] )
res5a_relu_patch2 = mx.symbol.Activation(name='res5a_relu_patch2', data=res5a_patch2 , act_type='relu')
res5b1_branch2a_patch2 = mx.symbol.Convolution(name='res5b1_branch2a_patch2', data=res5a_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res5b1_branch2a_bn_patch2', data=res5b1_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2a_scale_patch2 = res5b1_branch2a_bn_patch2
res5b1_branch2a_relu_patch2 = mx.symbol.Activation(name='res5b1_branch2a_relu_patch2', data=res5b1_branch2a_scale_patch2 , act_type='relu')
res5b1_branch2b_patch2 = mx.symbol.Convolution(name='res5b1_branch2b_patch2', data=res5b1_branch2a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b1_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res5b1_branch2b_bn_patch2', data=res5b1_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2b_scale_patch2 = res5b1_branch2b_bn_patch2
res5b1_branch2b_relu_patch2 = mx.symbol.Activation(name='res5b1_branch2b_relu_patch2', data=res5b1_branch2b_scale_patch2 , act_type='relu')
res5b1_branch2c_patch2 = mx.symbol.Convolution(name='res5b1_branch2c_patch2', data=res5b1_branch2b_relu_patch2 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res5b1_branch2c_bn_patch2', data=res5b1_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2c_scale_patch2 = res5b1_branch2c_bn_patch2
res5b1_patch2 = mx.symbol.broadcast_plus(name='res5b1_patch2', *[res5a_relu_patch2,res5b1_branch2c_scale_patch2] )
res5b1_relu_patch2 = mx.symbol.Activation(name='res5b1_relu_patch2', data=res5b1_patch2 , act_type='relu')
res5b2_branch2a_patch2 = mx.symbol.Convolution(name='res5b2_branch2a_patch2', data=res5b1_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2a_bn_patch2 = mx.symbol.BatchNorm(name='res5b2_branch2a_bn_patch2', data=res5b2_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2a_scale_patch2 = res5b2_branch2a_bn_patch2
res5b2_branch2a_relu_patch2 = mx.symbol.Activation(name='res5b2_branch2a_relu_patch2', data=res5b2_branch2a_scale_patch2 , act_type='relu')
res5b2_branch2b_patch2 = mx.symbol.Convolution(name='res5b2_branch2b_patch2', data=res5b2_branch2a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b2_branch2b_bn_patch2 = mx.symbol.BatchNorm(name='res5b2_branch2b_bn_patch2', data=res5b2_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2b_scale_patch2 = res5b2_branch2b_bn_patch2
res5b2_branch2b_relu_patch2 = mx.symbol.Activation(name='res5b2_branch2b_relu_patch2', data=res5b2_branch2b_scale_patch2 , act_type='relu')
res5b2_branch2c_patch2 = mx.symbol.Convolution(name='res5b2_branch2c_patch2', data=res5b2_branch2b_relu_patch2 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2c_bn_patch2 = mx.symbol.BatchNorm(name='res5b2_branch2c_bn_patch2', data=res5b2_branch2c_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2c_scale_patch2 = res5b2_branch2c_bn_patch2
res5b2_patch2 = mx.symbol.broadcast_plus(name='res5b2_patch2', *[res5b1_relu_patch2,res5b2_branch2c_scale_patch2] )
res5b2_relu_patch2 = mx.symbol.Activation(name='res5b2_relu_patch2', data=res5b2_patch2 , act_type='relu')
reduce_conv_patch2 = mx.symbol.Convolution(name='reduce_conv_patch2', data=res5b2_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduce_conv_bn_patch2 = mx.symbol.BatchNorm(name='reduce_conv_bn_patch2', data=reduce_conv_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
reduce_conv_scale_patch2 = reduce_conv_bn_patch2
pool5_patch2 = mx.symbol.Pooling(name='pool5_patch2', data=reduce_conv_scale_patch2 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch3 = mx.symbol.Convolution(name='conv1_patch3', data=data_patch3 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=False)
conv1_bn_patch3 = mx.symbol.BatchNorm(name='conv1_bn_patch3', data=conv1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_scale_patch3 = conv1_bn_patch3
conv1_relu_patch3 = mx.symbol.Activation(name='conv1_relu_patch3', data=conv1_scale_patch3 , act_type='relu')
pool1_patch3 = mx.symbol.Pooling(name='pool1_patch3', data=conv1_relu_patch3 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch3 = mx.symbol.Convolution(name='res2a_branch1_patch3', data=pool1_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch1_bn_patch3 = mx.symbol.BatchNorm(name='res2a_branch1_bn_patch3', data=res2a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch1_scale_patch3 = res2a_branch1_bn_patch3
res2a_branch2a_patch3 = mx.symbol.Convolution(name='res2a_branch2a_patch3', data=pool1_patch3 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res2a_branch2a_bn_patch3', data=res2a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2a_scale_patch3 = res2a_branch2a_bn_patch3
res2a_branch2a_relu_patch3 = mx.symbol.Activation(name='res2a_branch2a_relu_patch3', data=res2a_branch2a_scale_patch3 , act_type='relu')
res2a_branch2b_patch3 = mx.symbol.Convolution(name='res2a_branch2b_patch3', data=res2a_branch2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2a_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res2a_branch2b_bn_patch3', data=res2a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2b_scale_patch3 = res2a_branch2b_bn_patch3
res2a_branch2b_relu_patch3 = mx.symbol.Activation(name='res2a_branch2b_relu_patch3', data=res2a_branch2b_scale_patch3 , act_type='relu')
res2a_branch2c_patch3 = mx.symbol.Convolution(name='res2a_branch2c_patch3', data=res2a_branch2b_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res2a_branch2c_bn_patch3', data=res2a_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2c_scale_patch3 = res2a_branch2c_bn_patch3
res2a_patch3 = mx.symbol.broadcast_plus(name='res2a_patch3', *[res2a_branch1_scale_patch3,res2a_branch2c_scale_patch3] )
res2a_relu_patch3 = mx.symbol.Activation(name='res2a_relu_patch3', data=res2a_patch3 , act_type='relu')
res2b1_branch2a_patch3 = mx.symbol.Convolution(name='res2b1_branch2a_patch3', data=res2a_relu_patch3 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res2b1_branch2a_bn_patch3', data=res2b1_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2a_scale_patch3 = res2b1_branch2a_bn_patch3
res2b1_branch2a_relu_patch3 = mx.symbol.Activation(name='res2b1_branch2a_relu_patch3', data=res2b1_branch2a_scale_patch3 , act_type='relu')
res2b1_branch2b_patch3 = mx.symbol.Convolution(name='res2b1_branch2b_patch3', data=res2b1_branch2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b1_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res2b1_branch2b_bn_patch3', data=res2b1_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2b_scale_patch3 = res2b1_branch2b_bn_patch3
res2b1_branch2b_relu_patch3 = mx.symbol.Activation(name='res2b1_branch2b_relu_patch3', data=res2b1_branch2b_scale_patch3 , act_type='relu')
res2b1_branch2c_patch3 = mx.symbol.Convolution(name='res2b1_branch2c_patch3', data=res2b1_branch2b_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res2b1_branch2c_bn_patch3', data=res2b1_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2c_scale_patch3 = res2b1_branch2c_bn_patch3
res2b1_patch3 = mx.symbol.broadcast_plus(name='res2b1_patch3', *[res2a_relu_patch3,res2b1_branch2c_scale_patch3] )
res2b1_relu_patch3 = mx.symbol.Activation(name='res2b1_relu_patch3', data=res2b1_patch3 , act_type='relu')
res2b2_branch2a_patch3 = mx.symbol.Convolution(name='res2b2_branch2a_patch3', data=res2b1_relu_patch3 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res2b2_branch2a_bn_patch3', data=res2b2_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2a_scale_patch3 = res2b2_branch2a_bn_patch3
res2b2_branch2a_relu_patch3 = mx.symbol.Activation(name='res2b2_branch2a_relu_patch3', data=res2b2_branch2a_scale_patch3 , act_type='relu')
res2b2_branch2b_patch3 = mx.symbol.Convolution(name='res2b2_branch2b_patch3', data=res2b2_branch2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b2_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res2b2_branch2b_bn_patch3', data=res2b2_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2b_scale_patch3 = res2b2_branch2b_bn_patch3
res2b2_branch2b_relu_patch3 = mx.symbol.Activation(name='res2b2_branch2b_relu_patch3', data=res2b2_branch2b_scale_patch3 , act_type='relu')
res2b2_branch2c_patch3 = mx.symbol.Convolution(name='res2b2_branch2c_patch3', data=res2b2_branch2b_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res2b2_branch2c_bn_patch3', data=res2b2_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2c_scale_patch3 = res2b2_branch2c_bn_patch3
res2b2_patch3 = mx.symbol.broadcast_plus(name='res2b2_patch3', *[res2b1_relu_patch3,res2b2_branch2c_scale_patch3] )
res2b2_relu_patch3 = mx.symbol.Activation(name='res2b2_relu_patch3', data=res2b2_patch3 , act_type='relu')
res3a_branch1_patch3 = mx.symbol.Convolution(name='res3a_branch1_patch3', data=res2b2_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch1_bn_patch3 = mx.symbol.BatchNorm(name='res3a_branch1_bn_patch3', data=res3a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch1_scale_patch3 = res3a_branch1_bn_patch3
res3a_branch2a_patch3 = mx.symbol.Convolution(name='res3a_branch2a_patch3', data=res2b2_relu_patch3 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res3a_branch2a_bn_patch3', data=res3a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2a_scale_patch3 = res3a_branch2a_bn_patch3
res3a_branch2a_relu_patch3 = mx.symbol.Activation(name='res3a_branch2a_relu_patch3', data=res3a_branch2a_scale_patch3 , act_type='relu')
res3a_branch2b_patch3 = mx.symbol.Convolution(name='res3a_branch2b_patch3', data=res3a_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3a_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res3a_branch2b_bn_patch3', data=res3a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2b_scale_patch3 = res3a_branch2b_bn_patch3
res3a_branch2b_relu_patch3 = mx.symbol.Activation(name='res3a_branch2b_relu_patch3', data=res3a_branch2b_scale_patch3 , act_type='relu')
res3a_branch2c_patch3 = mx.symbol.Convolution(name='res3a_branch2c_patch3', data=res3a_branch2b_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3a_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res3a_branch2c_bn_patch3', data=res3a_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2c_scale_patch3 = res3a_branch2c_bn_patch3
res3a_patch3 = mx.symbol.broadcast_plus(name='res3a_patch3', *[res3a_branch1_scale_patch3,res3a_branch2c_scale_patch3] )
res3a_relu_patch3 = mx.symbol.Activation(name='res3a_relu_patch3', data=res3a_patch3 , act_type='relu')
res3b1_branch2a_patch3 = mx.symbol.Convolution(name='res3b1_branch2a_patch3', data=res3a_relu_patch3 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res3b1_branch2a_bn_patch3', data=res3b1_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2a_scale_patch3 = res3b1_branch2a_bn_patch3
res3b1_branch2a_relu_patch3 = mx.symbol.Activation(name='res3b1_branch2a_relu_patch3', data=res3b1_branch2a_scale_patch3 , act_type='relu')
res3b1_branch2b_patch3 = mx.symbol.Convolution(name='res3b1_branch2b_patch3', data=res3b1_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b1_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res3b1_branch2b_bn_patch3', data=res3b1_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2b_scale_patch3 = res3b1_branch2b_bn_patch3
res3b1_branch2b_relu_patch3 = mx.symbol.Activation(name='res3b1_branch2b_relu_patch3', data=res3b1_branch2b_scale_patch3 , act_type='relu')
res3b1_branch2c_patch3 = mx.symbol.Convolution(name='res3b1_branch2c_patch3', data=res3b1_branch2b_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res3b1_branch2c_bn_patch3', data=res3b1_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2c_scale_patch3 = res3b1_branch2c_bn_patch3
res3b1_patch3 = mx.symbol.broadcast_plus(name='res3b1_patch3', *[res3a_relu_patch3,res3b1_branch2c_scale_patch3] )
res3b1_relu_patch3 = mx.symbol.Activation(name='res3b1_relu_patch3', data=res3b1_patch3 , act_type='relu')
res3b2_branch2a_patch3 = mx.symbol.Convolution(name='res3b2_branch2a_patch3', data=res3b1_relu_patch3 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res3b2_branch2a_bn_patch3', data=res3b2_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2a_scale_patch3 = res3b2_branch2a_bn_patch3
res3b2_branch2a_relu_patch3 = mx.symbol.Activation(name='res3b2_branch2a_relu_patch3', data=res3b2_branch2a_scale_patch3 , act_type='relu')
res3b2_branch2b_patch3 = mx.symbol.Convolution(name='res3b2_branch2b_patch3', data=res3b2_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b2_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res3b2_branch2b_bn_patch3', data=res3b2_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2b_scale_patch3 = res3b2_branch2b_bn_patch3
res3b2_branch2b_relu_patch3 = mx.symbol.Activation(name='res3b2_branch2b_relu_patch3', data=res3b2_branch2b_scale_patch3 , act_type='relu')
res3b2_branch2c_patch3 = mx.symbol.Convolution(name='res3b2_branch2c_patch3', data=res3b2_branch2b_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res3b2_branch2c_bn_patch3', data=res3b2_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2c_scale_patch3 = res3b2_branch2c_bn_patch3
res3b2_patch3 = mx.symbol.broadcast_plus(name='res3b2_patch3', *[res3b1_relu_patch3,res3b2_branch2c_scale_patch3] )
res3b2_relu_patch3 = mx.symbol.Activation(name='res3b2_relu_patch3', data=res3b2_patch3 , act_type='relu')
res3b3_branch2a_patch3 = mx.symbol.Convolution(name='res3b3_branch2a_patch3', data=res3b2_relu_patch3 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res3b3_branch2a_bn_patch3', data=res3b3_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2a_scale_patch3 = res3b3_branch2a_bn_patch3
res3b3_branch2a_relu_patch3 = mx.symbol.Activation(name='res3b3_branch2a_relu_patch3', data=res3b3_branch2a_scale_patch3 , act_type='relu')
res3b3_branch2b_patch3 = mx.symbol.Convolution(name='res3b3_branch2b_patch3', data=res3b3_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b3_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res3b3_branch2b_bn_patch3', data=res3b3_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2b_scale_patch3 = res3b3_branch2b_bn_patch3
res3b3_branch2b_relu_patch3 = mx.symbol.Activation(name='res3b3_branch2b_relu_patch3', data=res3b3_branch2b_scale_patch3 , act_type='relu')
res3b3_branch2c_patch3 = mx.symbol.Convolution(name='res3b3_branch2c_patch3', data=res3b3_branch2b_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res3b3_branch2c_bn_patch3', data=res3b3_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2c_scale_patch3 = res3b3_branch2c_bn_patch3
res3b3_patch3 = mx.symbol.broadcast_plus(name='res3b3_patch3', *[res3b2_relu_patch3,res3b3_branch2c_scale_patch3] )
res3b3_relu_patch3 = mx.symbol.Activation(name='res3b3_relu_patch3', data=res3b3_patch3 , act_type='relu')
res4a_branch1_patch3 = mx.symbol.Convolution(name='res4a_branch1_patch3', data=res3b3_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch1_bn_patch3 = mx.symbol.BatchNorm(name='res4a_branch1_bn_patch3', data=res4a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch1_scale_patch3 = res4a_branch1_bn_patch3
res4a_branch2a_patch3 = mx.symbol.Convolution(name='res4a_branch2a_patch3', data=res3b3_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res4a_branch2a_bn_patch3', data=res4a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2a_scale_patch3 = res4a_branch2a_bn_patch3
res4a_branch2a_relu_patch3 = mx.symbol.Activation(name='res4a_branch2a_relu_patch3', data=res4a_branch2a_scale_patch3 , act_type='relu')
res4a_branch2b_patch3 = mx.symbol.Convolution(name='res4a_branch2b_patch3', data=res4a_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4a_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res4a_branch2b_bn_patch3', data=res4a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2b_scale_patch3 = res4a_branch2b_bn_patch3
res4a_branch2b_relu_patch3 = mx.symbol.Activation(name='res4a_branch2b_relu_patch3', data=res4a_branch2b_scale_patch3 , act_type='relu')
res4a_branch2c_patch3 = mx.symbol.Convolution(name='res4a_branch2c_patch3', data=res4a_branch2b_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4a_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res4a_branch2c_bn_patch3', data=res4a_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2c_scale_patch3 = res4a_branch2c_bn_patch3
res4a_patch3 = mx.symbol.broadcast_plus(name='res4a_patch3', *[res4a_branch1_scale_patch3,res4a_branch2c_scale_patch3] )
res4a_relu_patch3 = mx.symbol.Activation(name='res4a_relu_patch3', data=res4a_patch3 , act_type='relu')
res4b1_branch2a_patch3 = mx.symbol.Convolution(name='res4b1_branch2a_patch3', data=res4a_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res4b1_branch2a_bn_patch3', data=res4b1_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2a_scale_patch3 = res4b1_branch2a_bn_patch3
res4b1_branch2a_relu_patch3 = mx.symbol.Activation(name='res4b1_branch2a_relu_patch3', data=res4b1_branch2a_scale_patch3 , act_type='relu')
res4b1_branch2b_patch3 = mx.symbol.Convolution(name='res4b1_branch2b_patch3', data=res4b1_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b1_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res4b1_branch2b_bn_patch3', data=res4b1_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2b_scale_patch3 = res4b1_branch2b_bn_patch3
res4b1_branch2b_relu_patch3 = mx.symbol.Activation(name='res4b1_branch2b_relu_patch3', data=res4b1_branch2b_scale_patch3 , act_type='relu')
res4b1_branch2c_patch3 = mx.symbol.Convolution(name='res4b1_branch2c_patch3', data=res4b1_branch2b_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res4b1_branch2c_bn_patch3', data=res4b1_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2c_scale_patch3 = res4b1_branch2c_bn_patch3
res4b1_patch3 = mx.symbol.broadcast_plus(name='res4b1_patch3', *[res4a_relu_patch3,res4b1_branch2c_scale_patch3] )
res4b1_relu_patch3 = mx.symbol.Activation(name='res4b1_relu_patch3', data=res4b1_patch3 , act_type='relu')
res4b2_branch2a_patch3 = mx.symbol.Convolution(name='res4b2_branch2a_patch3', data=res4b1_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res4b2_branch2a_bn_patch3', data=res4b2_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2a_scale_patch3 = res4b2_branch2a_bn_patch3
res4b2_branch2a_relu_patch3 = mx.symbol.Activation(name='res4b2_branch2a_relu_patch3', data=res4b2_branch2a_scale_patch3 , act_type='relu')
res4b2_branch2b_patch3 = mx.symbol.Convolution(name='res4b2_branch2b_patch3', data=res4b2_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b2_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res4b2_branch2b_bn_patch3', data=res4b2_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2b_scale_patch3 = res4b2_branch2b_bn_patch3
res4b2_branch2b_relu_patch3 = mx.symbol.Activation(name='res4b2_branch2b_relu_patch3', data=res4b2_branch2b_scale_patch3 , act_type='relu')
res4b2_branch2c_patch3 = mx.symbol.Convolution(name='res4b2_branch2c_patch3', data=res4b2_branch2b_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res4b2_branch2c_bn_patch3', data=res4b2_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2c_scale_patch3 = res4b2_branch2c_bn_patch3
res4b2_patch3 = mx.symbol.broadcast_plus(name='res4b2_patch3', *[res4b1_relu_patch3,res4b2_branch2c_scale_patch3] )
res4b2_relu_patch3 = mx.symbol.Activation(name='res4b2_relu_patch3', data=res4b2_patch3 , act_type='relu')
res4b3_branch2a_patch3 = mx.symbol.Convolution(name='res4b3_branch2a_patch3', data=res4b2_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res4b3_branch2a_bn_patch3', data=res4b3_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2a_scale_patch3 = res4b3_branch2a_bn_patch3
res4b3_branch2a_relu_patch3 = mx.symbol.Activation(name='res4b3_branch2a_relu_patch3', data=res4b3_branch2a_scale_patch3 , act_type='relu')
res4b3_branch2b_patch3 = mx.symbol.Convolution(name='res4b3_branch2b_patch3', data=res4b3_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b3_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res4b3_branch2b_bn_patch3', data=res4b3_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2b_scale_patch3 = res4b3_branch2b_bn_patch3
res4b3_branch2b_relu_patch3 = mx.symbol.Activation(name='res4b3_branch2b_relu_patch3', data=res4b3_branch2b_scale_patch3 , act_type='relu')
res4b3_branch2c_patch3 = mx.symbol.Convolution(name='res4b3_branch2c_patch3', data=res4b3_branch2b_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res4b3_branch2c_bn_patch3', data=res4b3_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2c_scale_patch3 = res4b3_branch2c_bn_patch3
res4b3_patch3 = mx.symbol.broadcast_plus(name='res4b3_patch3', *[res4b2_relu_patch3,res4b3_branch2c_scale_patch3] )
res4b3_relu_patch3 = mx.symbol.Activation(name='res4b3_relu_patch3', data=res4b3_patch3 , act_type='relu')
res4b4_branch2a_patch3 = mx.symbol.Convolution(name='res4b4_branch2a_patch3', data=res4b3_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res4b4_branch2a_bn_patch3', data=res4b4_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2a_scale_patch3 = res4b4_branch2a_bn_patch3
res4b4_branch2a_relu_patch3 = mx.symbol.Activation(name='res4b4_branch2a_relu_patch3', data=res4b4_branch2a_scale_patch3 , act_type='relu')
res4b4_branch2b_patch3 = mx.symbol.Convolution(name='res4b4_branch2b_patch3', data=res4b4_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b4_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res4b4_branch2b_bn_patch3', data=res4b4_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2b_scale_patch3 = res4b4_branch2b_bn_patch3
res4b4_branch2b_relu_patch3 = mx.symbol.Activation(name='res4b4_branch2b_relu_patch3', data=res4b4_branch2b_scale_patch3 , act_type='relu')
res4b4_branch2c_patch3 = mx.symbol.Convolution(name='res4b4_branch2c_patch3', data=res4b4_branch2b_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res4b4_branch2c_bn_patch3', data=res4b4_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2c_scale_patch3 = res4b4_branch2c_bn_patch3
res4b4_patch3 = mx.symbol.broadcast_plus(name='res4b4_patch3', *[res4b3_relu_patch3,res4b4_branch2c_scale_patch3] )
res4b4_relu_patch3 = mx.symbol.Activation(name='res4b4_relu_patch3', data=res4b4_patch3 , act_type='relu')
res4b5_branch2a_patch3 = mx.symbol.Convolution(name='res4b5_branch2a_patch3', data=res4b4_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res4b5_branch2a_bn_patch3', data=res4b5_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2a_scale_patch3 = res4b5_branch2a_bn_patch3
res4b5_branch2a_relu_patch3 = mx.symbol.Activation(name='res4b5_branch2a_relu_patch3', data=res4b5_branch2a_scale_patch3 , act_type='relu')
res4b5_branch2b_patch3 = mx.symbol.Convolution(name='res4b5_branch2b_patch3', data=res4b5_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b5_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res4b5_branch2b_bn_patch3', data=res4b5_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2b_scale_patch3 = res4b5_branch2b_bn_patch3
res4b5_branch2b_relu_patch3 = mx.symbol.Activation(name='res4b5_branch2b_relu_patch3', data=res4b5_branch2b_scale_patch3 , act_type='relu')
res4b5_branch2c_patch3 = mx.symbol.Convolution(name='res4b5_branch2c_patch3', data=res4b5_branch2b_relu_patch3 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res4b5_branch2c_bn_patch3', data=res4b5_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2c_scale_patch3 = res4b5_branch2c_bn_patch3
res4b5_patch3 = mx.symbol.broadcast_plus(name='res4b5_patch3', *[res4b4_relu_patch3,res4b5_branch2c_scale_patch3] )
res4b5_relu_patch3 = mx.symbol.Activation(name='res4b5_relu_patch3', data=res4b5_patch3 , act_type='relu')
res5a_branch1_patch3 = mx.symbol.Convolution(name='res5a_branch1_patch3', data=res4b5_relu_patch3 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch1_bn_patch3 = mx.symbol.BatchNorm(name='res5a_branch1_bn_patch3', data=res5a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch1_scale_patch3 = res5a_branch1_bn_patch3
res5a_branch2a_patch3 = mx.symbol.Convolution(name='res5a_branch2a_patch3', data=res4b5_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res5a_branch2a_bn_patch3', data=res5a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2a_scale_patch3 = res5a_branch2a_bn_patch3
res5a_branch2a_relu_patch3 = mx.symbol.Activation(name='res5a_branch2a_relu_patch3', data=res5a_branch2a_scale_patch3 , act_type='relu')
res5a_branch2b_patch3 = mx.symbol.Convolution(name='res5a_branch2b_patch3', data=res5a_branch2a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5a_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res5a_branch2b_bn_patch3', data=res5a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2b_scale_patch3 = res5a_branch2b_bn_patch3
res5a_branch2b_relu_patch3 = mx.symbol.Activation(name='res5a_branch2b_relu_patch3', data=res5a_branch2b_scale_patch3 , act_type='relu')
res5a_branch2c_patch3 = mx.symbol.Convolution(name='res5a_branch2c_patch3', data=res5a_branch2b_relu_patch3 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5a_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res5a_branch2c_bn_patch3', data=res5a_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2c_scale_patch3 = res5a_branch2c_bn_patch3
res5a_patch3 = mx.symbol.broadcast_plus(name='res5a_patch3', *[res5a_branch1_scale_patch3,res5a_branch2c_scale_patch3] )
res5a_relu_patch3 = mx.symbol.Activation(name='res5a_relu_patch3', data=res5a_patch3 , act_type='relu')
res5b1_branch2a_patch3 = mx.symbol.Convolution(name='res5b1_branch2a_patch3', data=res5a_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res5b1_branch2a_bn_patch3', data=res5b1_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2a_scale_patch3 = res5b1_branch2a_bn_patch3
res5b1_branch2a_relu_patch3 = mx.symbol.Activation(name='res5b1_branch2a_relu_patch3', data=res5b1_branch2a_scale_patch3 , act_type='relu')
res5b1_branch2b_patch3 = mx.symbol.Convolution(name='res5b1_branch2b_patch3', data=res5b1_branch2a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b1_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res5b1_branch2b_bn_patch3', data=res5b1_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2b_scale_patch3 = res5b1_branch2b_bn_patch3
res5b1_branch2b_relu_patch3 = mx.symbol.Activation(name='res5b1_branch2b_relu_patch3', data=res5b1_branch2b_scale_patch3 , act_type='relu')
res5b1_branch2c_patch3 = mx.symbol.Convolution(name='res5b1_branch2c_patch3', data=res5b1_branch2b_relu_patch3 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res5b1_branch2c_bn_patch3', data=res5b1_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2c_scale_patch3 = res5b1_branch2c_bn_patch3
res5b1_patch3 = mx.symbol.broadcast_plus(name='res5b1_patch3', *[res5a_relu_patch3,res5b1_branch2c_scale_patch3] )
res5b1_relu_patch3 = mx.symbol.Activation(name='res5b1_relu_patch3', data=res5b1_patch3 , act_type='relu')
res5b2_branch2a_patch3 = mx.symbol.Convolution(name='res5b2_branch2a_patch3', data=res5b1_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2a_bn_patch3 = mx.symbol.BatchNorm(name='res5b2_branch2a_bn_patch3', data=res5b2_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2a_scale_patch3 = res5b2_branch2a_bn_patch3
res5b2_branch2a_relu_patch3 = mx.symbol.Activation(name='res5b2_branch2a_relu_patch3', data=res5b2_branch2a_scale_patch3 , act_type='relu')
res5b2_branch2b_patch3 = mx.symbol.Convolution(name='res5b2_branch2b_patch3', data=res5b2_branch2a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b2_branch2b_bn_patch3 = mx.symbol.BatchNorm(name='res5b2_branch2b_bn_patch3', data=res5b2_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2b_scale_patch3 = res5b2_branch2b_bn_patch3
res5b2_branch2b_relu_patch3 = mx.symbol.Activation(name='res5b2_branch2b_relu_patch3', data=res5b2_branch2b_scale_patch3 , act_type='relu')
res5b2_branch2c_patch3 = mx.symbol.Convolution(name='res5b2_branch2c_patch3', data=res5b2_branch2b_relu_patch3 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2c_bn_patch3 = mx.symbol.BatchNorm(name='res5b2_branch2c_bn_patch3', data=res5b2_branch2c_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2c_scale_patch3 = res5b2_branch2c_bn_patch3
res5b2_patch3 = mx.symbol.broadcast_plus(name='res5b2_patch3', *[res5b1_relu_patch3,res5b2_branch2c_scale_patch3] )
res5b2_relu_patch3 = mx.symbol.Activation(name='res5b2_relu_patch3', data=res5b2_patch3 , act_type='relu')
reduce_conv_patch3 = mx.symbol.Convolution(name='reduce_conv_patch3', data=res5b2_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduce_conv_bn_patch3 = mx.symbol.BatchNorm(name='reduce_conv_bn_patch3', data=reduce_conv_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
reduce_conv_scale_patch3 = reduce_conv_bn_patch3
pool5_patch3 = mx.symbol.Pooling(name='pool5_patch3', data=reduce_conv_scale_patch3 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch6 = mx.symbol.Convolution(name='conv1_patch6', data=data_patch6 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=False)
conv1_bn_patch6 = mx.symbol.BatchNorm(name='conv1_bn_patch6', data=conv1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_scale_patch6 = conv1_bn_patch6
conv1_relu_patch6 = mx.symbol.Activation(name='conv1_relu_patch6', data=conv1_scale_patch6 , act_type='relu')
pool1_patch6 = mx.symbol.Pooling(name='pool1_patch6', data=conv1_relu_patch6 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch6 = mx.symbol.Convolution(name='res2a_branch1_patch6', data=pool1_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch1_bn_patch6 = mx.symbol.BatchNorm(name='res2a_branch1_bn_patch6', data=res2a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch1_scale_patch6 = res2a_branch1_bn_patch6
res2a_branch2a_patch6 = mx.symbol.Convolution(name='res2a_branch2a_patch6', data=pool1_patch6 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res2a_branch2a_bn_patch6', data=res2a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2a_scale_patch6 = res2a_branch2a_bn_patch6
res2a_branch2a_relu_patch6 = mx.symbol.Activation(name='res2a_branch2a_relu_patch6', data=res2a_branch2a_scale_patch6 , act_type='relu')
res2a_branch2b_patch6 = mx.symbol.Convolution(name='res2a_branch2b_patch6', data=res2a_branch2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2a_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res2a_branch2b_bn_patch6', data=res2a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2b_scale_patch6 = res2a_branch2b_bn_patch6
res2a_branch2b_relu_patch6 = mx.symbol.Activation(name='res2a_branch2b_relu_patch6', data=res2a_branch2b_scale_patch6 , act_type='relu')
res2a_branch2c_patch6 = mx.symbol.Convolution(name='res2a_branch2c_patch6', data=res2a_branch2b_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res2a_branch2c_bn_patch6', data=res2a_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2c_scale_patch6 = res2a_branch2c_bn_patch6
res2a_patch6 = mx.symbol.broadcast_plus(name='res2a_patch6', *[res2a_branch1_scale_patch6,res2a_branch2c_scale_patch6] )
res2a_relu_patch6 = mx.symbol.Activation(name='res2a_relu_patch6', data=res2a_patch6 , act_type='relu')
res2b1_branch2a_patch6 = mx.symbol.Convolution(name='res2b1_branch2a_patch6', data=res2a_relu_patch6 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res2b1_branch2a_bn_patch6', data=res2b1_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2a_scale_patch6 = res2b1_branch2a_bn_patch6
res2b1_branch2a_relu_patch6 = mx.symbol.Activation(name='res2b1_branch2a_relu_patch6', data=res2b1_branch2a_scale_patch6 , act_type='relu')
res2b1_branch2b_patch6 = mx.symbol.Convolution(name='res2b1_branch2b_patch6', data=res2b1_branch2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b1_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res2b1_branch2b_bn_patch6', data=res2b1_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2b_scale_patch6 = res2b1_branch2b_bn_patch6
res2b1_branch2b_relu_patch6 = mx.symbol.Activation(name='res2b1_branch2b_relu_patch6', data=res2b1_branch2b_scale_patch6 , act_type='relu')
res2b1_branch2c_patch6 = mx.symbol.Convolution(name='res2b1_branch2c_patch6', data=res2b1_branch2b_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res2b1_branch2c_bn_patch6', data=res2b1_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2c_scale_patch6 = res2b1_branch2c_bn_patch6
res2b1_patch6 = mx.symbol.broadcast_plus(name='res2b1_patch6', *[res2a_relu_patch6,res2b1_branch2c_scale_patch6] )
res2b1_relu_patch6 = mx.symbol.Activation(name='res2b1_relu_patch6', data=res2b1_patch6 , act_type='relu')
res2b2_branch2a_patch6 = mx.symbol.Convolution(name='res2b2_branch2a_patch6', data=res2b1_relu_patch6 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res2b2_branch2a_bn_patch6', data=res2b2_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2a_scale_patch6 = res2b2_branch2a_bn_patch6
res2b2_branch2a_relu_patch6 = mx.symbol.Activation(name='res2b2_branch2a_relu_patch6', data=res2b2_branch2a_scale_patch6 , act_type='relu')
res2b2_branch2b_patch6 = mx.symbol.Convolution(name='res2b2_branch2b_patch6', data=res2b2_branch2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b2_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res2b2_branch2b_bn_patch6', data=res2b2_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2b_scale_patch6 = res2b2_branch2b_bn_patch6
res2b2_branch2b_relu_patch6 = mx.symbol.Activation(name='res2b2_branch2b_relu_patch6', data=res2b2_branch2b_scale_patch6 , act_type='relu')
res2b2_branch2c_patch6 = mx.symbol.Convolution(name='res2b2_branch2c_patch6', data=res2b2_branch2b_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res2b2_branch2c_bn_patch6', data=res2b2_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2c_scale_patch6 = res2b2_branch2c_bn_patch6
res2b2_patch6 = mx.symbol.broadcast_plus(name='res2b2_patch6', *[res2b1_relu_patch6,res2b2_branch2c_scale_patch6] )
res2b2_relu_patch6 = mx.symbol.Activation(name='res2b2_relu_patch6', data=res2b2_patch6 , act_type='relu')
res3a_branch1_patch6 = mx.symbol.Convolution(name='res3a_branch1_patch6', data=res2b2_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch1_bn_patch6 = mx.symbol.BatchNorm(name='res3a_branch1_bn_patch6', data=res3a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch1_scale_patch6 = res3a_branch1_bn_patch6
res3a_branch2a_patch6 = mx.symbol.Convolution(name='res3a_branch2a_patch6', data=res2b2_relu_patch6 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res3a_branch2a_bn_patch6', data=res3a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2a_scale_patch6 = res3a_branch2a_bn_patch6
res3a_branch2a_relu_patch6 = mx.symbol.Activation(name='res3a_branch2a_relu_patch6', data=res3a_branch2a_scale_patch6 , act_type='relu')
res3a_branch2b_patch6 = mx.symbol.Convolution(name='res3a_branch2b_patch6', data=res3a_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3a_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res3a_branch2b_bn_patch6', data=res3a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2b_scale_patch6 = res3a_branch2b_bn_patch6
res3a_branch2b_relu_patch6 = mx.symbol.Activation(name='res3a_branch2b_relu_patch6', data=res3a_branch2b_scale_patch6 , act_type='relu')
res3a_branch2c_patch6 = mx.symbol.Convolution(name='res3a_branch2c_patch6', data=res3a_branch2b_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3a_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res3a_branch2c_bn_patch6', data=res3a_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2c_scale_patch6 = res3a_branch2c_bn_patch6
res3a_patch6 = mx.symbol.broadcast_plus(name='res3a_patch6', *[res3a_branch1_scale_patch6,res3a_branch2c_scale_patch6] )
res3a_relu_patch6 = mx.symbol.Activation(name='res3a_relu_patch6', data=res3a_patch6 , act_type='relu')
res3b1_branch2a_patch6 = mx.symbol.Convolution(name='res3b1_branch2a_patch6', data=res3a_relu_patch6 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res3b1_branch2a_bn_patch6', data=res3b1_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2a_scale_patch6 = res3b1_branch2a_bn_patch6
res3b1_branch2a_relu_patch6 = mx.symbol.Activation(name='res3b1_branch2a_relu_patch6', data=res3b1_branch2a_scale_patch6 , act_type='relu')
res3b1_branch2b_patch6 = mx.symbol.Convolution(name='res3b1_branch2b_patch6', data=res3b1_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b1_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res3b1_branch2b_bn_patch6', data=res3b1_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2b_scale_patch6 = res3b1_branch2b_bn_patch6
res3b1_branch2b_relu_patch6 = mx.symbol.Activation(name='res3b1_branch2b_relu_patch6', data=res3b1_branch2b_scale_patch6 , act_type='relu')
res3b1_branch2c_patch6 = mx.symbol.Convolution(name='res3b1_branch2c_patch6', data=res3b1_branch2b_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res3b1_branch2c_bn_patch6', data=res3b1_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2c_scale_patch6 = res3b1_branch2c_bn_patch6
res3b1_patch6 = mx.symbol.broadcast_plus(name='res3b1_patch6', *[res3a_relu_patch6,res3b1_branch2c_scale_patch6] )
res3b1_relu_patch6 = mx.symbol.Activation(name='res3b1_relu_patch6', data=res3b1_patch6 , act_type='relu')
res3b2_branch2a_patch6 = mx.symbol.Convolution(name='res3b2_branch2a_patch6', data=res3b1_relu_patch6 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res3b2_branch2a_bn_patch6', data=res3b2_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2a_scale_patch6 = res3b2_branch2a_bn_patch6
res3b2_branch2a_relu_patch6 = mx.symbol.Activation(name='res3b2_branch2a_relu_patch6', data=res3b2_branch2a_scale_patch6 , act_type='relu')
res3b2_branch2b_patch6 = mx.symbol.Convolution(name='res3b2_branch2b_patch6', data=res3b2_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b2_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res3b2_branch2b_bn_patch6', data=res3b2_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2b_scale_patch6 = res3b2_branch2b_bn_patch6
res3b2_branch2b_relu_patch6 = mx.symbol.Activation(name='res3b2_branch2b_relu_patch6', data=res3b2_branch2b_scale_patch6 , act_type='relu')
res3b2_branch2c_patch6 = mx.symbol.Convolution(name='res3b2_branch2c_patch6', data=res3b2_branch2b_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res3b2_branch2c_bn_patch6', data=res3b2_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2c_scale_patch6 = res3b2_branch2c_bn_patch6
res3b2_patch6 = mx.symbol.broadcast_plus(name='res3b2_patch6', *[res3b1_relu_patch6,res3b2_branch2c_scale_patch6] )
res3b2_relu_patch6 = mx.symbol.Activation(name='res3b2_relu_patch6', data=res3b2_patch6 , act_type='relu')
res3b3_branch2a_patch6 = mx.symbol.Convolution(name='res3b3_branch2a_patch6', data=res3b2_relu_patch6 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res3b3_branch2a_bn_patch6', data=res3b3_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2a_scale_patch6 = res3b3_branch2a_bn_patch6
res3b3_branch2a_relu_patch6 = mx.symbol.Activation(name='res3b3_branch2a_relu_patch6', data=res3b3_branch2a_scale_patch6 , act_type='relu')
res3b3_branch2b_patch6 = mx.symbol.Convolution(name='res3b3_branch2b_patch6', data=res3b3_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b3_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res3b3_branch2b_bn_patch6', data=res3b3_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2b_scale_patch6 = res3b3_branch2b_bn_patch6
res3b3_branch2b_relu_patch6 = mx.symbol.Activation(name='res3b3_branch2b_relu_patch6', data=res3b3_branch2b_scale_patch6 , act_type='relu')
res3b3_branch2c_patch6 = mx.symbol.Convolution(name='res3b3_branch2c_patch6', data=res3b3_branch2b_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res3b3_branch2c_bn_patch6', data=res3b3_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2c_scale_patch6 = res3b3_branch2c_bn_patch6
res3b3_patch6 = mx.symbol.broadcast_plus(name='res3b3_patch6', *[res3b2_relu_patch6,res3b3_branch2c_scale_patch6] )
res3b3_relu_patch6 = mx.symbol.Activation(name='res3b3_relu_patch6', data=res3b3_patch6 , act_type='relu')
res4a_branch1_patch6 = mx.symbol.Convolution(name='res4a_branch1_patch6', data=res3b3_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch1_bn_patch6 = mx.symbol.BatchNorm(name='res4a_branch1_bn_patch6', data=res4a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch1_scale_patch6 = res4a_branch1_bn_patch6
res4a_branch2a_patch6 = mx.symbol.Convolution(name='res4a_branch2a_patch6', data=res3b3_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res4a_branch2a_bn_patch6', data=res4a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2a_scale_patch6 = res4a_branch2a_bn_patch6
res4a_branch2a_relu_patch6 = mx.symbol.Activation(name='res4a_branch2a_relu_patch6', data=res4a_branch2a_scale_patch6 , act_type='relu')
res4a_branch2b_patch6 = mx.symbol.Convolution(name='res4a_branch2b_patch6', data=res4a_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4a_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res4a_branch2b_bn_patch6', data=res4a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2b_scale_patch6 = res4a_branch2b_bn_patch6
res4a_branch2b_relu_patch6 = mx.symbol.Activation(name='res4a_branch2b_relu_patch6', data=res4a_branch2b_scale_patch6 , act_type='relu')
res4a_branch2c_patch6 = mx.symbol.Convolution(name='res4a_branch2c_patch6', data=res4a_branch2b_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4a_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res4a_branch2c_bn_patch6', data=res4a_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2c_scale_patch6 = res4a_branch2c_bn_patch6
res4a_patch6 = mx.symbol.broadcast_plus(name='res4a_patch6', *[res4a_branch1_scale_patch6,res4a_branch2c_scale_patch6] )
res4a_relu_patch6 = mx.symbol.Activation(name='res4a_relu_patch6', data=res4a_patch6 , act_type='relu')
res4b1_branch2a_patch6 = mx.symbol.Convolution(name='res4b1_branch2a_patch6', data=res4a_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res4b1_branch2a_bn_patch6', data=res4b1_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2a_scale_patch6 = res4b1_branch2a_bn_patch6
res4b1_branch2a_relu_patch6 = mx.symbol.Activation(name='res4b1_branch2a_relu_patch6', data=res4b1_branch2a_scale_patch6 , act_type='relu')
res4b1_branch2b_patch6 = mx.symbol.Convolution(name='res4b1_branch2b_patch6', data=res4b1_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b1_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res4b1_branch2b_bn_patch6', data=res4b1_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2b_scale_patch6 = res4b1_branch2b_bn_patch6
res4b1_branch2b_relu_patch6 = mx.symbol.Activation(name='res4b1_branch2b_relu_patch6', data=res4b1_branch2b_scale_patch6 , act_type='relu')
res4b1_branch2c_patch6 = mx.symbol.Convolution(name='res4b1_branch2c_patch6', data=res4b1_branch2b_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res4b1_branch2c_bn_patch6', data=res4b1_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2c_scale_patch6 = res4b1_branch2c_bn_patch6
res4b1_patch6 = mx.symbol.broadcast_plus(name='res4b1_patch6', *[res4a_relu_patch6,res4b1_branch2c_scale_patch6] )
res4b1_relu_patch6 = mx.symbol.Activation(name='res4b1_relu_patch6', data=res4b1_patch6 , act_type='relu')
res4b2_branch2a_patch6 = mx.symbol.Convolution(name='res4b2_branch2a_patch6', data=res4b1_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res4b2_branch2a_bn_patch6', data=res4b2_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2a_scale_patch6 = res4b2_branch2a_bn_patch6
res4b2_branch2a_relu_patch6 = mx.symbol.Activation(name='res4b2_branch2a_relu_patch6', data=res4b2_branch2a_scale_patch6 , act_type='relu')
res4b2_branch2b_patch6 = mx.symbol.Convolution(name='res4b2_branch2b_patch6', data=res4b2_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b2_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res4b2_branch2b_bn_patch6', data=res4b2_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2b_scale_patch6 = res4b2_branch2b_bn_patch6
res4b2_branch2b_relu_patch6 = mx.symbol.Activation(name='res4b2_branch2b_relu_patch6', data=res4b2_branch2b_scale_patch6 , act_type='relu')
res4b2_branch2c_patch6 = mx.symbol.Convolution(name='res4b2_branch2c_patch6', data=res4b2_branch2b_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res4b2_branch2c_bn_patch6', data=res4b2_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2c_scale_patch6 = res4b2_branch2c_bn_patch6
res4b2_patch6 = mx.symbol.broadcast_plus(name='res4b2_patch6', *[res4b1_relu_patch6,res4b2_branch2c_scale_patch6] )
res4b2_relu_patch6 = mx.symbol.Activation(name='res4b2_relu_patch6', data=res4b2_patch6 , act_type='relu')
res4b3_branch2a_patch6 = mx.symbol.Convolution(name='res4b3_branch2a_patch6', data=res4b2_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res4b3_branch2a_bn_patch6', data=res4b3_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2a_scale_patch6 = res4b3_branch2a_bn_patch6
res4b3_branch2a_relu_patch6 = mx.symbol.Activation(name='res4b3_branch2a_relu_patch6', data=res4b3_branch2a_scale_patch6 , act_type='relu')
res4b3_branch2b_patch6 = mx.symbol.Convolution(name='res4b3_branch2b_patch6', data=res4b3_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b3_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res4b3_branch2b_bn_patch6', data=res4b3_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2b_scale_patch6 = res4b3_branch2b_bn_patch6
res4b3_branch2b_relu_patch6 = mx.symbol.Activation(name='res4b3_branch2b_relu_patch6', data=res4b3_branch2b_scale_patch6 , act_type='relu')
res4b3_branch2c_patch6 = mx.symbol.Convolution(name='res4b3_branch2c_patch6', data=res4b3_branch2b_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res4b3_branch2c_bn_patch6', data=res4b3_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2c_scale_patch6 = res4b3_branch2c_bn_patch6
res4b3_patch6 = mx.symbol.broadcast_plus(name='res4b3_patch6', *[res4b2_relu_patch6,res4b3_branch2c_scale_patch6] )
res4b3_relu_patch6 = mx.symbol.Activation(name='res4b3_relu_patch6', data=res4b3_patch6 , act_type='relu')
res4b4_branch2a_patch6 = mx.symbol.Convolution(name='res4b4_branch2a_patch6', data=res4b3_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res4b4_branch2a_bn_patch6', data=res4b4_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2a_scale_patch6 = res4b4_branch2a_bn_patch6
res4b4_branch2a_relu_patch6 = mx.symbol.Activation(name='res4b4_branch2a_relu_patch6', data=res4b4_branch2a_scale_patch6 , act_type='relu')
res4b4_branch2b_patch6 = mx.symbol.Convolution(name='res4b4_branch2b_patch6', data=res4b4_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b4_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res4b4_branch2b_bn_patch6', data=res4b4_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2b_scale_patch6 = res4b4_branch2b_bn_patch6
res4b4_branch2b_relu_patch6 = mx.symbol.Activation(name='res4b4_branch2b_relu_patch6', data=res4b4_branch2b_scale_patch6 , act_type='relu')
res4b4_branch2c_patch6 = mx.symbol.Convolution(name='res4b4_branch2c_patch6', data=res4b4_branch2b_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res4b4_branch2c_bn_patch6', data=res4b4_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2c_scale_patch6 = res4b4_branch2c_bn_patch6
res4b4_patch6 = mx.symbol.broadcast_plus(name='res4b4_patch6', *[res4b3_relu_patch6,res4b4_branch2c_scale_patch6] )
res4b4_relu_patch6 = mx.symbol.Activation(name='res4b4_relu_patch6', data=res4b4_patch6 , act_type='relu')
res4b5_branch2a_patch6 = mx.symbol.Convolution(name='res4b5_branch2a_patch6', data=res4b4_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res4b5_branch2a_bn_patch6', data=res4b5_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2a_scale_patch6 = res4b5_branch2a_bn_patch6
res4b5_branch2a_relu_patch6 = mx.symbol.Activation(name='res4b5_branch2a_relu_patch6', data=res4b5_branch2a_scale_patch6 , act_type='relu')
res4b5_branch2b_patch6 = mx.symbol.Convolution(name='res4b5_branch2b_patch6', data=res4b5_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b5_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res4b5_branch2b_bn_patch6', data=res4b5_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2b_scale_patch6 = res4b5_branch2b_bn_patch6
res4b5_branch2b_relu_patch6 = mx.symbol.Activation(name='res4b5_branch2b_relu_patch6', data=res4b5_branch2b_scale_patch6 , act_type='relu')
res4b5_branch2c_patch6 = mx.symbol.Convolution(name='res4b5_branch2c_patch6', data=res4b5_branch2b_relu_patch6 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res4b5_branch2c_bn_patch6', data=res4b5_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2c_scale_patch6 = res4b5_branch2c_bn_patch6
res4b5_patch6 = mx.symbol.broadcast_plus(name='res4b5_patch6', *[res4b4_relu_patch6,res4b5_branch2c_scale_patch6] )
res4b5_relu_patch6 = mx.symbol.Activation(name='res4b5_relu_patch6', data=res4b5_patch6 , act_type='relu')
res5a_branch1_patch6 = mx.symbol.Convolution(name='res5a_branch1_patch6', data=res4b5_relu_patch6 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch1_bn_patch6 = mx.symbol.BatchNorm(name='res5a_branch1_bn_patch6', data=res5a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch1_scale_patch6 = res5a_branch1_bn_patch6
res5a_branch2a_patch6 = mx.symbol.Convolution(name='res5a_branch2a_patch6', data=res4b5_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res5a_branch2a_bn_patch6', data=res5a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2a_scale_patch6 = res5a_branch2a_bn_patch6
res5a_branch2a_relu_patch6 = mx.symbol.Activation(name='res5a_branch2a_relu_patch6', data=res5a_branch2a_scale_patch6 , act_type='relu')
res5a_branch2b_patch6 = mx.symbol.Convolution(name='res5a_branch2b_patch6', data=res5a_branch2a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5a_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res5a_branch2b_bn_patch6', data=res5a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2b_scale_patch6 = res5a_branch2b_bn_patch6
res5a_branch2b_relu_patch6 = mx.symbol.Activation(name='res5a_branch2b_relu_patch6', data=res5a_branch2b_scale_patch6 , act_type='relu')
res5a_branch2c_patch6 = mx.symbol.Convolution(name='res5a_branch2c_patch6', data=res5a_branch2b_relu_patch6 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5a_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res5a_branch2c_bn_patch6', data=res5a_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2c_scale_patch6 = res5a_branch2c_bn_patch6
res5a_patch6 = mx.symbol.broadcast_plus(name='res5a_patch6', *[res5a_branch1_scale_patch6,res5a_branch2c_scale_patch6] )
res5a_relu_patch6 = mx.symbol.Activation(name='res5a_relu_patch6', data=res5a_patch6 , act_type='relu')
res5b1_branch2a_patch6 = mx.symbol.Convolution(name='res5b1_branch2a_patch6', data=res5a_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res5b1_branch2a_bn_patch6', data=res5b1_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2a_scale_patch6 = res5b1_branch2a_bn_patch6
res5b1_branch2a_relu_patch6 = mx.symbol.Activation(name='res5b1_branch2a_relu_patch6', data=res5b1_branch2a_scale_patch6 , act_type='relu')
res5b1_branch2b_patch6 = mx.symbol.Convolution(name='res5b1_branch2b_patch6', data=res5b1_branch2a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b1_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res5b1_branch2b_bn_patch6', data=res5b1_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2b_scale_patch6 = res5b1_branch2b_bn_patch6
res5b1_branch2b_relu_patch6 = mx.symbol.Activation(name='res5b1_branch2b_relu_patch6', data=res5b1_branch2b_scale_patch6 , act_type='relu')
res5b1_branch2c_patch6 = mx.symbol.Convolution(name='res5b1_branch2c_patch6', data=res5b1_branch2b_relu_patch6 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res5b1_branch2c_bn_patch6', data=res5b1_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2c_scale_patch6 = res5b1_branch2c_bn_patch6
res5b1_patch6 = mx.symbol.broadcast_plus(name='res5b1_patch6', *[res5a_relu_patch6,res5b1_branch2c_scale_patch6] )
res5b1_relu_patch6 = mx.symbol.Activation(name='res5b1_relu_patch6', data=res5b1_patch6 , act_type='relu')
res5b2_branch2a_patch6 = mx.symbol.Convolution(name='res5b2_branch2a_patch6', data=res5b1_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2a_bn_patch6 = mx.symbol.BatchNorm(name='res5b2_branch2a_bn_patch6', data=res5b2_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2a_scale_patch6 = res5b2_branch2a_bn_patch6
res5b2_branch2a_relu_patch6 = mx.symbol.Activation(name='res5b2_branch2a_relu_patch6', data=res5b2_branch2a_scale_patch6 , act_type='relu')
res5b2_branch2b_patch6 = mx.symbol.Convolution(name='res5b2_branch2b_patch6', data=res5b2_branch2a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b2_branch2b_bn_patch6 = mx.symbol.BatchNorm(name='res5b2_branch2b_bn_patch6', data=res5b2_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2b_scale_patch6 = res5b2_branch2b_bn_patch6
res5b2_branch2b_relu_patch6 = mx.symbol.Activation(name='res5b2_branch2b_relu_patch6', data=res5b2_branch2b_scale_patch6 , act_type='relu')
res5b2_branch2c_patch6 = mx.symbol.Convolution(name='res5b2_branch2c_patch6', data=res5b2_branch2b_relu_patch6 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2c_bn_patch6 = mx.symbol.BatchNorm(name='res5b2_branch2c_bn_patch6', data=res5b2_branch2c_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2c_scale_patch6 = res5b2_branch2c_bn_patch6
res5b2_patch6 = mx.symbol.broadcast_plus(name='res5b2_patch6', *[res5b1_relu_patch6,res5b2_branch2c_scale_patch6] )
res5b2_relu_patch6 = mx.symbol.Activation(name='res5b2_relu_patch6', data=res5b2_patch6 , act_type='relu')
reduce_conv_patch6 = mx.symbol.Convolution(name='reduce_conv_patch6', data=res5b2_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduce_conv_bn_patch6 = mx.symbol.BatchNorm(name='reduce_conv_bn_patch6', data=reduce_conv_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
reduce_conv_scale_patch6 = reduce_conv_bn_patch6
pool5_patch6 = mx.symbol.Pooling(name='pool5_patch6', data=reduce_conv_scale_patch6 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch7 = mx.symbol.Convolution(name='conv1_patch7', data=data_patch7 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=False)
conv1_bn_patch7 = mx.symbol.BatchNorm(name='conv1_bn_patch7', data=conv1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
conv1_scale_patch7 = conv1_bn_patch7
conv1_relu_patch7 = mx.symbol.Activation(name='conv1_relu_patch7', data=conv1_scale_patch7 , act_type='relu')
pool1_patch7 = mx.symbol.Pooling(name='pool1_patch7', data=conv1_relu_patch7 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch7 = mx.symbol.Convolution(name='res2a_branch1_patch7', data=pool1_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch1_bn_patch7 = mx.symbol.BatchNorm(name='res2a_branch1_bn_patch7', data=res2a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch1_scale_patch7 = res2a_branch1_bn_patch7
res2a_branch2a_patch7 = mx.symbol.Convolution(name='res2a_branch2a_patch7', data=pool1_patch7 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res2a_branch2a_bn_patch7', data=res2a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2a_scale_patch7 = res2a_branch2a_bn_patch7
res2a_branch2a_relu_patch7 = mx.symbol.Activation(name='res2a_branch2a_relu_patch7', data=res2a_branch2a_scale_patch7 , act_type='relu')
res2a_branch2b_patch7 = mx.symbol.Convolution(name='res2a_branch2b_patch7', data=res2a_branch2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2a_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res2a_branch2b_bn_patch7', data=res2a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2b_scale_patch7 = res2a_branch2b_bn_patch7
res2a_branch2b_relu_patch7 = mx.symbol.Activation(name='res2a_branch2b_relu_patch7', data=res2a_branch2b_scale_patch7 , act_type='relu')
res2a_branch2c_patch7 = mx.symbol.Convolution(name='res2a_branch2c_patch7', data=res2a_branch2b_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2a_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res2a_branch2c_bn_patch7', data=res2a_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2a_branch2c_scale_patch7 = res2a_branch2c_bn_patch7
res2a_patch7 = mx.symbol.broadcast_plus(name='res2a_patch7', *[res2a_branch1_scale_patch7,res2a_branch2c_scale_patch7] )
res2a_relu_patch7 = mx.symbol.Activation(name='res2a_relu_patch7', data=res2a_patch7 , act_type='relu')
res2b1_branch2a_patch7 = mx.symbol.Convolution(name='res2b1_branch2a_patch7', data=res2a_relu_patch7 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res2b1_branch2a_bn_patch7', data=res2b1_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2a_scale_patch7 = res2b1_branch2a_bn_patch7
res2b1_branch2a_relu_patch7 = mx.symbol.Activation(name='res2b1_branch2a_relu_patch7', data=res2b1_branch2a_scale_patch7 , act_type='relu')
res2b1_branch2b_patch7 = mx.symbol.Convolution(name='res2b1_branch2b_patch7', data=res2b1_branch2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b1_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res2b1_branch2b_bn_patch7', data=res2b1_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2b_scale_patch7 = res2b1_branch2b_bn_patch7
res2b1_branch2b_relu_patch7 = mx.symbol.Activation(name='res2b1_branch2b_relu_patch7', data=res2b1_branch2b_scale_patch7 , act_type='relu')
res2b1_branch2c_patch7 = mx.symbol.Convolution(name='res2b1_branch2c_patch7', data=res2b1_branch2b_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b1_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res2b1_branch2c_bn_patch7', data=res2b1_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b1_branch2c_scale_patch7 = res2b1_branch2c_bn_patch7
res2b1_patch7 = mx.symbol.broadcast_plus(name='res2b1_patch7', *[res2a_relu_patch7,res2b1_branch2c_scale_patch7] )
res2b1_relu_patch7 = mx.symbol.Activation(name='res2b1_relu_patch7', data=res2b1_patch7 , act_type='relu')
res2b2_branch2a_patch7 = mx.symbol.Convolution(name='res2b2_branch2a_patch7', data=res2b1_relu_patch7 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res2b2_branch2a_bn_patch7', data=res2b2_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2a_scale_patch7 = res2b2_branch2a_bn_patch7
res2b2_branch2a_relu_patch7 = mx.symbol.Activation(name='res2b2_branch2a_relu_patch7', data=res2b2_branch2a_scale_patch7 , act_type='relu')
res2b2_branch2b_patch7 = mx.symbol.Convolution(name='res2b2_branch2b_patch7', data=res2b2_branch2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res2b2_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res2b2_branch2b_bn_patch7', data=res2b2_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2b_scale_patch7 = res2b2_branch2b_bn_patch7
res2b2_branch2b_relu_patch7 = mx.symbol.Activation(name='res2b2_branch2b_relu_patch7', data=res2b2_branch2b_scale_patch7 , act_type='relu')
res2b2_branch2c_patch7 = mx.symbol.Convolution(name='res2b2_branch2c_patch7', data=res2b2_branch2b_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res2b2_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res2b2_branch2c_bn_patch7', data=res2b2_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res2b2_branch2c_scale_patch7 = res2b2_branch2c_bn_patch7
res2b2_patch7 = mx.symbol.broadcast_plus(name='res2b2_patch7', *[res2b1_relu_patch7,res2b2_branch2c_scale_patch7] )
res2b2_relu_patch7 = mx.symbol.Activation(name='res2b2_relu_patch7', data=res2b2_patch7 , act_type='relu')
res3a_branch1_patch7 = mx.symbol.Convolution(name='res3a_branch1_patch7', data=res2b2_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch1_bn_patch7 = mx.symbol.BatchNorm(name='res3a_branch1_bn_patch7', data=res3a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch1_scale_patch7 = res3a_branch1_bn_patch7
res3a_branch2a_patch7 = mx.symbol.Convolution(name='res3a_branch2a_patch7', data=res2b2_relu_patch7 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res3a_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res3a_branch2a_bn_patch7', data=res3a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2a_scale_patch7 = res3a_branch2a_bn_patch7
res3a_branch2a_relu_patch7 = mx.symbol.Activation(name='res3a_branch2a_relu_patch7', data=res3a_branch2a_scale_patch7 , act_type='relu')
res3a_branch2b_patch7 = mx.symbol.Convolution(name='res3a_branch2b_patch7', data=res3a_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3a_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res3a_branch2b_bn_patch7', data=res3a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2b_scale_patch7 = res3a_branch2b_bn_patch7
res3a_branch2b_relu_patch7 = mx.symbol.Activation(name='res3a_branch2b_relu_patch7', data=res3a_branch2b_scale_patch7 , act_type='relu')
res3a_branch2c_patch7 = mx.symbol.Convolution(name='res3a_branch2c_patch7', data=res3a_branch2b_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3a_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res3a_branch2c_bn_patch7', data=res3a_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3a_branch2c_scale_patch7 = res3a_branch2c_bn_patch7
res3a_patch7 = mx.symbol.broadcast_plus(name='res3a_patch7', *[res3a_branch1_scale_patch7,res3a_branch2c_scale_patch7] )
res3a_relu_patch7 = mx.symbol.Activation(name='res3a_relu_patch7', data=res3a_patch7 , act_type='relu')
res3b1_branch2a_patch7 = mx.symbol.Convolution(name='res3b1_branch2a_patch7', data=res3a_relu_patch7 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res3b1_branch2a_bn_patch7', data=res3b1_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2a_scale_patch7 = res3b1_branch2a_bn_patch7
res3b1_branch2a_relu_patch7 = mx.symbol.Activation(name='res3b1_branch2a_relu_patch7', data=res3b1_branch2a_scale_patch7 , act_type='relu')
res3b1_branch2b_patch7 = mx.symbol.Convolution(name='res3b1_branch2b_patch7', data=res3b1_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b1_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res3b1_branch2b_bn_patch7', data=res3b1_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2b_scale_patch7 = res3b1_branch2b_bn_patch7
res3b1_branch2b_relu_patch7 = mx.symbol.Activation(name='res3b1_branch2b_relu_patch7', data=res3b1_branch2b_scale_patch7 , act_type='relu')
res3b1_branch2c_patch7 = mx.symbol.Convolution(name='res3b1_branch2c_patch7', data=res3b1_branch2b_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b1_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res3b1_branch2c_bn_patch7', data=res3b1_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b1_branch2c_scale_patch7 = res3b1_branch2c_bn_patch7
res3b1_patch7 = mx.symbol.broadcast_plus(name='res3b1_patch7', *[res3a_relu_patch7,res3b1_branch2c_scale_patch7] )
res3b1_relu_patch7 = mx.symbol.Activation(name='res3b1_relu_patch7', data=res3b1_patch7 , act_type='relu')
res3b2_branch2a_patch7 = mx.symbol.Convolution(name='res3b2_branch2a_patch7', data=res3b1_relu_patch7 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res3b2_branch2a_bn_patch7', data=res3b2_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2a_scale_patch7 = res3b2_branch2a_bn_patch7
res3b2_branch2a_relu_patch7 = mx.symbol.Activation(name='res3b2_branch2a_relu_patch7', data=res3b2_branch2a_scale_patch7 , act_type='relu')
res3b2_branch2b_patch7 = mx.symbol.Convolution(name='res3b2_branch2b_patch7', data=res3b2_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b2_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res3b2_branch2b_bn_patch7', data=res3b2_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2b_scale_patch7 = res3b2_branch2b_bn_patch7
res3b2_branch2b_relu_patch7 = mx.symbol.Activation(name='res3b2_branch2b_relu_patch7', data=res3b2_branch2b_scale_patch7 , act_type='relu')
res3b2_branch2c_patch7 = mx.symbol.Convolution(name='res3b2_branch2c_patch7', data=res3b2_branch2b_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b2_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res3b2_branch2c_bn_patch7', data=res3b2_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b2_branch2c_scale_patch7 = res3b2_branch2c_bn_patch7
res3b2_patch7 = mx.symbol.broadcast_plus(name='res3b2_patch7', *[res3b1_relu_patch7,res3b2_branch2c_scale_patch7] )
res3b2_relu_patch7 = mx.symbol.Activation(name='res3b2_relu_patch7', data=res3b2_patch7 , act_type='relu')
res3b3_branch2a_patch7 = mx.symbol.Convolution(name='res3b3_branch2a_patch7', data=res3b2_relu_patch7 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res3b3_branch2a_bn_patch7', data=res3b3_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2a_scale_patch7 = res3b3_branch2a_bn_patch7
res3b3_branch2a_relu_patch7 = mx.symbol.Activation(name='res3b3_branch2a_relu_patch7', data=res3b3_branch2a_scale_patch7 , act_type='relu')
res3b3_branch2b_patch7 = mx.symbol.Convolution(name='res3b3_branch2b_patch7', data=res3b3_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res3b3_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res3b3_branch2b_bn_patch7', data=res3b3_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2b_scale_patch7 = res3b3_branch2b_bn_patch7
res3b3_branch2b_relu_patch7 = mx.symbol.Activation(name='res3b3_branch2b_relu_patch7', data=res3b3_branch2b_scale_patch7 , act_type='relu')
res3b3_branch2c_patch7 = mx.symbol.Convolution(name='res3b3_branch2c_patch7', data=res3b3_branch2b_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res3b3_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res3b3_branch2c_bn_patch7', data=res3b3_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res3b3_branch2c_scale_patch7 = res3b3_branch2c_bn_patch7
res3b3_patch7 = mx.symbol.broadcast_plus(name='res3b3_patch7', *[res3b2_relu_patch7,res3b3_branch2c_scale_patch7] )
res3b3_relu_patch7 = mx.symbol.Activation(name='res3b3_relu_patch7', data=res3b3_patch7 , act_type='relu')
res4a_branch1_patch7 = mx.symbol.Convolution(name='res4a_branch1_patch7', data=res3b3_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch1_bn_patch7 = mx.symbol.BatchNorm(name='res4a_branch1_bn_patch7', data=res4a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch1_scale_patch7 = res4a_branch1_bn_patch7
res4a_branch2a_patch7 = mx.symbol.Convolution(name='res4a_branch2a_patch7', data=res3b3_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res4a_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res4a_branch2a_bn_patch7', data=res4a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2a_scale_patch7 = res4a_branch2a_bn_patch7
res4a_branch2a_relu_patch7 = mx.symbol.Activation(name='res4a_branch2a_relu_patch7', data=res4a_branch2a_scale_patch7 , act_type='relu')
res4a_branch2b_patch7 = mx.symbol.Convolution(name='res4a_branch2b_patch7', data=res4a_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4a_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res4a_branch2b_bn_patch7', data=res4a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2b_scale_patch7 = res4a_branch2b_bn_patch7
res4a_branch2b_relu_patch7 = mx.symbol.Activation(name='res4a_branch2b_relu_patch7', data=res4a_branch2b_scale_patch7 , act_type='relu')
res4a_branch2c_patch7 = mx.symbol.Convolution(name='res4a_branch2c_patch7', data=res4a_branch2b_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4a_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res4a_branch2c_bn_patch7', data=res4a_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4a_branch2c_scale_patch7 = res4a_branch2c_bn_patch7
res4a_patch7 = mx.symbol.broadcast_plus(name='res4a_patch7', *[res4a_branch1_scale_patch7,res4a_branch2c_scale_patch7] )
res4a_relu_patch7 = mx.symbol.Activation(name='res4a_relu_patch7', data=res4a_patch7 , act_type='relu')
res4b1_branch2a_patch7 = mx.symbol.Convolution(name='res4b1_branch2a_patch7', data=res4a_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res4b1_branch2a_bn_patch7', data=res4b1_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2a_scale_patch7 = res4b1_branch2a_bn_patch7
res4b1_branch2a_relu_patch7 = mx.symbol.Activation(name='res4b1_branch2a_relu_patch7', data=res4b1_branch2a_scale_patch7 , act_type='relu')
res4b1_branch2b_patch7 = mx.symbol.Convolution(name='res4b1_branch2b_patch7', data=res4b1_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b1_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res4b1_branch2b_bn_patch7', data=res4b1_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2b_scale_patch7 = res4b1_branch2b_bn_patch7
res4b1_branch2b_relu_patch7 = mx.symbol.Activation(name='res4b1_branch2b_relu_patch7', data=res4b1_branch2b_scale_patch7 , act_type='relu')
res4b1_branch2c_patch7 = mx.symbol.Convolution(name='res4b1_branch2c_patch7', data=res4b1_branch2b_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b1_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res4b1_branch2c_bn_patch7', data=res4b1_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b1_branch2c_scale_patch7 = res4b1_branch2c_bn_patch7
res4b1_patch7 = mx.symbol.broadcast_plus(name='res4b1_patch7', *[res4a_relu_patch7,res4b1_branch2c_scale_patch7] )
res4b1_relu_patch7 = mx.symbol.Activation(name='res4b1_relu_patch7', data=res4b1_patch7 , act_type='relu')
res4b2_branch2a_patch7 = mx.symbol.Convolution(name='res4b2_branch2a_patch7', data=res4b1_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res4b2_branch2a_bn_patch7', data=res4b2_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2a_scale_patch7 = res4b2_branch2a_bn_patch7
res4b2_branch2a_relu_patch7 = mx.symbol.Activation(name='res4b2_branch2a_relu_patch7', data=res4b2_branch2a_scale_patch7 , act_type='relu')
res4b2_branch2b_patch7 = mx.symbol.Convolution(name='res4b2_branch2b_patch7', data=res4b2_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b2_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res4b2_branch2b_bn_patch7', data=res4b2_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2b_scale_patch7 = res4b2_branch2b_bn_patch7
res4b2_branch2b_relu_patch7 = mx.symbol.Activation(name='res4b2_branch2b_relu_patch7', data=res4b2_branch2b_scale_patch7 , act_type='relu')
res4b2_branch2c_patch7 = mx.symbol.Convolution(name='res4b2_branch2c_patch7', data=res4b2_branch2b_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b2_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res4b2_branch2c_bn_patch7', data=res4b2_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b2_branch2c_scale_patch7 = res4b2_branch2c_bn_patch7
res4b2_patch7 = mx.symbol.broadcast_plus(name='res4b2_patch7', *[res4b1_relu_patch7,res4b2_branch2c_scale_patch7] )
res4b2_relu_patch7 = mx.symbol.Activation(name='res4b2_relu_patch7', data=res4b2_patch7 , act_type='relu')
res4b3_branch2a_patch7 = mx.symbol.Convolution(name='res4b3_branch2a_patch7', data=res4b2_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res4b3_branch2a_bn_patch7', data=res4b3_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2a_scale_patch7 = res4b3_branch2a_bn_patch7
res4b3_branch2a_relu_patch7 = mx.symbol.Activation(name='res4b3_branch2a_relu_patch7', data=res4b3_branch2a_scale_patch7 , act_type='relu')
res4b3_branch2b_patch7 = mx.symbol.Convolution(name='res4b3_branch2b_patch7', data=res4b3_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b3_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res4b3_branch2b_bn_patch7', data=res4b3_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2b_scale_patch7 = res4b3_branch2b_bn_patch7
res4b3_branch2b_relu_patch7 = mx.symbol.Activation(name='res4b3_branch2b_relu_patch7', data=res4b3_branch2b_scale_patch7 , act_type='relu')
res4b3_branch2c_patch7 = mx.symbol.Convolution(name='res4b3_branch2c_patch7', data=res4b3_branch2b_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b3_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res4b3_branch2c_bn_patch7', data=res4b3_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b3_branch2c_scale_patch7 = res4b3_branch2c_bn_patch7
res4b3_patch7 = mx.symbol.broadcast_plus(name='res4b3_patch7', *[res4b2_relu_patch7,res4b3_branch2c_scale_patch7] )
res4b3_relu_patch7 = mx.symbol.Activation(name='res4b3_relu_patch7', data=res4b3_patch7 , act_type='relu')
res4b4_branch2a_patch7 = mx.symbol.Convolution(name='res4b4_branch2a_patch7', data=res4b3_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res4b4_branch2a_bn_patch7', data=res4b4_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2a_scale_patch7 = res4b4_branch2a_bn_patch7
res4b4_branch2a_relu_patch7 = mx.symbol.Activation(name='res4b4_branch2a_relu_patch7', data=res4b4_branch2a_scale_patch7 , act_type='relu')
res4b4_branch2b_patch7 = mx.symbol.Convolution(name='res4b4_branch2b_patch7', data=res4b4_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b4_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res4b4_branch2b_bn_patch7', data=res4b4_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2b_scale_patch7 = res4b4_branch2b_bn_patch7
res4b4_branch2b_relu_patch7 = mx.symbol.Activation(name='res4b4_branch2b_relu_patch7', data=res4b4_branch2b_scale_patch7 , act_type='relu')
res4b4_branch2c_patch7 = mx.symbol.Convolution(name='res4b4_branch2c_patch7', data=res4b4_branch2b_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b4_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res4b4_branch2c_bn_patch7', data=res4b4_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b4_branch2c_scale_patch7 = res4b4_branch2c_bn_patch7
res4b4_patch7 = mx.symbol.broadcast_plus(name='res4b4_patch7', *[res4b3_relu_patch7,res4b4_branch2c_scale_patch7] )
res4b4_relu_patch7 = mx.symbol.Activation(name='res4b4_relu_patch7', data=res4b4_patch7 , act_type='relu')
res4b5_branch2a_patch7 = mx.symbol.Convolution(name='res4b5_branch2a_patch7', data=res4b4_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res4b5_branch2a_bn_patch7', data=res4b5_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2a_scale_patch7 = res4b5_branch2a_bn_patch7
res4b5_branch2a_relu_patch7 = mx.symbol.Activation(name='res4b5_branch2a_relu_patch7', data=res4b5_branch2a_scale_patch7 , act_type='relu')
res4b5_branch2b_patch7 = mx.symbol.Convolution(name='res4b5_branch2b_patch7', data=res4b5_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res4b5_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res4b5_branch2b_bn_patch7', data=res4b5_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2b_scale_patch7 = res4b5_branch2b_bn_patch7
res4b5_branch2b_relu_patch7 = mx.symbol.Activation(name='res4b5_branch2b_relu_patch7', data=res4b5_branch2b_scale_patch7 , act_type='relu')
res4b5_branch2c_patch7 = mx.symbol.Convolution(name='res4b5_branch2c_patch7', data=res4b5_branch2b_relu_patch7 , num_filter=1024, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res4b5_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res4b5_branch2c_bn_patch7', data=res4b5_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res4b5_branch2c_scale_patch7 = res4b5_branch2c_bn_patch7
res4b5_patch7 = mx.symbol.broadcast_plus(name='res4b5_patch7', *[res4b4_relu_patch7,res4b5_branch2c_scale_patch7] )
res4b5_relu_patch7 = mx.symbol.Activation(name='res4b5_relu_patch7', data=res4b5_patch7 , act_type='relu')
res5a_branch1_patch7 = mx.symbol.Convolution(name='res5a_branch1_patch7', data=res4b5_relu_patch7 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch1_bn_patch7 = mx.symbol.BatchNorm(name='res5a_branch1_bn_patch7', data=res5a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch1_scale_patch7 = res5a_branch1_bn_patch7
res5a_branch2a_patch7 = mx.symbol.Convolution(name='res5a_branch2a_patch7', data=res4b5_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
res5a_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res5a_branch2a_bn_patch7', data=res5a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2a_scale_patch7 = res5a_branch2a_bn_patch7
res5a_branch2a_relu_patch7 = mx.symbol.Activation(name='res5a_branch2a_relu_patch7', data=res5a_branch2a_scale_patch7 , act_type='relu')
res5a_branch2b_patch7 = mx.symbol.Convolution(name='res5a_branch2b_patch7', data=res5a_branch2a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5a_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res5a_branch2b_bn_patch7', data=res5a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2b_scale_patch7 = res5a_branch2b_bn_patch7
res5a_branch2b_relu_patch7 = mx.symbol.Activation(name='res5a_branch2b_relu_patch7', data=res5a_branch2b_scale_patch7 , act_type='relu')
res5a_branch2c_patch7 = mx.symbol.Convolution(name='res5a_branch2c_patch7', data=res5a_branch2b_relu_patch7 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5a_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res5a_branch2c_bn_patch7', data=res5a_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5a_branch2c_scale_patch7 = res5a_branch2c_bn_patch7
res5a_patch7 = mx.symbol.broadcast_plus(name='res5a_patch7', *[res5a_branch1_scale_patch7,res5a_branch2c_scale_patch7] )
res5a_relu_patch7 = mx.symbol.Activation(name='res5a_relu_patch7', data=res5a_patch7 , act_type='relu')
res5b1_branch2a_patch7 = mx.symbol.Convolution(name='res5b1_branch2a_patch7', data=res5a_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res5b1_branch2a_bn_patch7', data=res5b1_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2a_scale_patch7 = res5b1_branch2a_bn_patch7
res5b1_branch2a_relu_patch7 = mx.symbol.Activation(name='res5b1_branch2a_relu_patch7', data=res5b1_branch2a_scale_patch7 , act_type='relu')
res5b1_branch2b_patch7 = mx.symbol.Convolution(name='res5b1_branch2b_patch7', data=res5b1_branch2a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b1_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res5b1_branch2b_bn_patch7', data=res5b1_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2b_scale_patch7 = res5b1_branch2b_bn_patch7
res5b1_branch2b_relu_patch7 = mx.symbol.Activation(name='res5b1_branch2b_relu_patch7', data=res5b1_branch2b_scale_patch7 , act_type='relu')
res5b1_branch2c_patch7 = mx.symbol.Convolution(name='res5b1_branch2c_patch7', data=res5b1_branch2b_relu_patch7 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b1_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res5b1_branch2c_bn_patch7', data=res5b1_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b1_branch2c_scale_patch7 = res5b1_branch2c_bn_patch7
res5b1_patch7 = mx.symbol.broadcast_plus(name='res5b1_patch7', *[res5a_relu_patch7,res5b1_branch2c_scale_patch7] )
res5b1_relu_patch7 = mx.symbol.Activation(name='res5b1_relu_patch7', data=res5b1_patch7 , act_type='relu')
res5b2_branch2a_patch7 = mx.symbol.Convolution(name='res5b2_branch2a_patch7', data=res5b1_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2a_bn_patch7 = mx.symbol.BatchNorm(name='res5b2_branch2a_bn_patch7', data=res5b2_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2a_scale_patch7 = res5b2_branch2a_bn_patch7
res5b2_branch2a_relu_patch7 = mx.symbol.Activation(name='res5b2_branch2a_relu_patch7', data=res5b2_branch2a_scale_patch7 , act_type='relu')
res5b2_branch2b_patch7 = mx.symbol.Convolution(name='res5b2_branch2b_patch7', data=res5b2_branch2a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
res5b2_branch2b_bn_patch7 = mx.symbol.BatchNorm(name='res5b2_branch2b_bn_patch7', data=res5b2_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2b_scale_patch7 = res5b2_branch2b_bn_patch7
res5b2_branch2b_relu_patch7 = mx.symbol.Activation(name='res5b2_branch2b_relu_patch7', data=res5b2_branch2b_scale_patch7 , act_type='relu')
res5b2_branch2c_patch7 = mx.symbol.Convolution(name='res5b2_branch2c_patch7', data=res5b2_branch2b_relu_patch7 , num_filter=2048, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
res5b2_branch2c_bn_patch7 = mx.symbol.BatchNorm(name='res5b2_branch2c_bn_patch7', data=res5b2_branch2c_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
res5b2_branch2c_scale_patch7 = res5b2_branch2c_bn_patch7
res5b2_patch7 = mx.symbol.broadcast_plus(name='res5b2_patch7', *[res5b1_relu_patch7,res5b2_branch2c_scale_patch7] )
res5b2_relu_patch7 = mx.symbol.Activation(name='res5b2_relu_patch7', data=res5b2_patch7 , act_type='relu')
reduce_conv_patch7 = mx.symbol.Convolution(name='reduce_conv_patch7', data=res5b2_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
reduce_conv_bn_patch7 = mx.symbol.BatchNorm(name='reduce_conv_bn_patch7', data=reduce_conv_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
reduce_conv_scale_patch7 = reduce_conv_bn_patch7
pool5_patch7 = mx.symbol.Pooling(name='pool5_patch7', data=reduce_conv_scale_patch7 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
concat_0 = mx.symbol.Concat(name='concat_0', *[pool5_patch0,pool5_patch1,pool5_patch2,pool5_patch3,pool5_patch6,pool5_patch7] )
