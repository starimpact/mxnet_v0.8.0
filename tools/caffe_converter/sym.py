import mxnet as mx
data = mx.symbol.Variable(name='data')
slice_data = mx.symbol.SliceChannel(name='slice_data', data=data , num_outputs=6)
data_patch0 = slice_data[0]
data_patch1 = slice_data[1]
data_patch2 = slice_data[2]
data_patch3 = slice_data[3]
data_patch6 = slice_data[4]
data_patch7 = slice_data[5]
conv1_patch0 = mx.symbol.Convolution(name='conv1_patch0', data=data_patch0 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
bn_conv1_patch0 = mx.symbol.BatchNorm(name='bn_conv1_patch0', data=conv1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale_conv1_patch0 = bn_conv1_patch0
conv1_relu_patch0 = mx.symbol.Activation(name='conv1_relu_patch0', data=scale_conv1_patch0 , act_type='relu')
pool1_patch0 = mx.symbol.Pooling(name='pool1_patch0', data=conv1_relu_patch0 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch0 = mx.symbol.Convolution(name='res2a_branch1_patch0', data=pool1_patch0 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1_patch0 = mx.symbol.BatchNorm(name='bn2a_branch1_patch0', data=res2a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch1_patch0 = bn2a_branch1_patch0
res2a_branch2a_patch0 = mx.symbol.Convolution(name='res2a_branch2a_patch0', data=pool1_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2a_patch0 = mx.symbol.BatchNorm(name='bn2a_branch2a_patch0', data=res2a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2a_patch0 = bn2a_branch2a_patch0
res2a_branch2a_relu_patch0 = mx.symbol.Activation(name='res2a_branch2a_relu_patch0', data=scale2a_branch2a_patch0 , act_type='relu')
res2a_branch2b_patch0 = mx.symbol.Convolution(name='res2a_branch2b_patch0', data=res2a_branch2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b_patch0 = mx.symbol.BatchNorm(name='bn2a_branch2b_patch0', data=res2a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2b_patch0 = bn2a_branch2b_patch0
res2a_patch0 = mx.symbol.broadcast_plus(name='res2a_patch0', *[scale2a_branch1_patch0,scale2a_branch2b_patch0] )
res2a_relu_patch0 = mx.symbol.Activation(name='res2a_relu_patch0', data=res2a_patch0 , act_type='relu')
res2b_branch2a_patch0 = mx.symbol.Convolution(name='res2b_branch2a_patch0', data=res2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2a_patch0 = mx.symbol.BatchNorm(name='bn2b_branch2a_patch0', data=res2b_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2a_patch0 = bn2b_branch2a_patch0
res2b_branch2a_relu_patch0 = mx.symbol.Activation(name='res2b_branch2a_relu_patch0', data=scale2b_branch2a_patch0 , act_type='relu')
res2b_branch2b_patch0 = mx.symbol.Convolution(name='res2b_branch2b_patch0', data=res2b_branch2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b_patch0 = mx.symbol.BatchNorm(name='bn2b_branch2b_patch0', data=res2b_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2b_patch0 = bn2b_branch2b_patch0
res2b_patch0 = mx.symbol.broadcast_plus(name='res2b_patch0', *[res2a_relu_patch0,scale2b_branch2b_patch0] )
res2b_relu_patch0 = mx.symbol.Activation(name='res2b_relu_patch0', data=res2b_patch0 , act_type='relu')
res2c_branch2a_patch0 = mx.symbol.Convolution(name='res2c_branch2a_patch0', data=res2b_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2a_patch0 = mx.symbol.BatchNorm(name='bn2c_branch2a_patch0', data=res2c_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2a_patch0 = bn2c_branch2a_patch0
res2c_branch2a_relu_patch0 = mx.symbol.Activation(name='res2c_branch2a_relu_patch0', data=scale2c_branch2a_patch0 , act_type='relu')
res2c_branch2b_patch0 = mx.symbol.Convolution(name='res2c_branch2b_patch0', data=res2c_branch2a_relu_patch0 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b_patch0 = mx.symbol.BatchNorm(name='bn2c_branch2b_patch0', data=res2c_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2b_patch0 = bn2c_branch2b_patch0
res2c_patch0 = mx.symbol.broadcast_plus(name='res2c_patch0', *[res2b_relu_patch0,scale2c_branch2b_patch0] )
res2c_relu_patch0 = mx.symbol.Activation(name='res2c_relu_patch0', data=res2c_patch0 , act_type='relu')
res3a_branch1_patch0 = mx.symbol.Convolution(name='res3a_branch1_patch0', data=res2c_relu_patch0 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1_patch0 = mx.symbol.BatchNorm(name='bn3a_branch1_patch0', data=res3a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch1_patch0 = bn3a_branch1_patch0
res3a_branch2a_patch0 = mx.symbol.Convolution(name='res3a_branch2a_patch0', data=res2c_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn3a_branch2a_patch0 = mx.symbol.BatchNorm(name='bn3a_branch2a_patch0', data=res3a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2a_patch0 = bn3a_branch2a_patch0
res3a_branch2a_relu_patch0 = mx.symbol.Activation(name='res3a_branch2a_relu_patch0', data=scale3a_branch2a_patch0 , act_type='relu')
res3a_branch2b_patch0 = mx.symbol.Convolution(name='res3a_branch2b_patch0', data=res3a_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b_patch0 = mx.symbol.BatchNorm(name='bn3a_branch2b_patch0', data=res3a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2b_patch0 = bn3a_branch2b_patch0
res3a_patch0 = mx.symbol.broadcast_plus(name='res3a_patch0', *[scale3a_branch1_patch0,scale3a_branch2b_patch0] )
res3a_relu_patch0 = mx.symbol.Activation(name='res3a_relu_patch0', data=res3a_patch0 , act_type='relu')
res3b_branch2a_patch0 = mx.symbol.Convolution(name='res3b_branch2a_patch0', data=res3a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2a_patch0 = mx.symbol.BatchNorm(name='bn3b_branch2a_patch0', data=res3b_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2a_patch0 = bn3b_branch2a_patch0
res3b_branch2a_relu_patch0 = mx.symbol.Activation(name='res3b_branch2a_relu_patch0', data=scale3b_branch2a_patch0 , act_type='relu')
res3b_branch2b_patch0 = mx.symbol.Convolution(name='res3b_branch2b_patch0', data=res3b_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2b_patch0 = mx.symbol.BatchNorm(name='bn3b_branch2b_patch0', data=res3b_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2b_patch0 = bn3b_branch2b_patch0
res3b_patch0 = mx.symbol.broadcast_plus(name='res3b_patch0', *[res3a_relu_patch0,scale3b_branch2b_patch0] )
res3b_relu_patch0 = mx.symbol.Activation(name='res3b_relu_patch0', data=res3b_patch0 , act_type='relu')
res3c_branch2a_patch0 = mx.symbol.Convolution(name='res3c_branch2a_patch0', data=res3b_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2a_patch0 = mx.symbol.BatchNorm(name='bn3c_branch2a_patch0', data=res3c_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2a_patch0 = bn3c_branch2a_patch0
res3c_branch2a_relu_patch0 = mx.symbol.Activation(name='res3c_branch2a_relu_patch0', data=scale3c_branch2a_patch0 , act_type='relu')
res3c_branch2b_patch0 = mx.symbol.Convolution(name='res3c_branch2b_patch0', data=res3c_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2b_patch0 = mx.symbol.BatchNorm(name='bn3c_branch2b_patch0', data=res3c_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2b_patch0 = bn3c_branch2b_patch0
res3c_patch0 = mx.symbol.broadcast_plus(name='res3c_patch0', *[res3b_relu_patch0,scale3c_branch2b_patch0] )
res3c_relu_patch0 = mx.symbol.Activation(name='res3c_relu_patch0', data=res3c_patch0 , act_type='relu')
res3d_branch2a_patch0 = mx.symbol.Convolution(name='res3d_branch2a_patch0', data=res3c_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2a_patch0 = mx.symbol.BatchNorm(name='bn3d_branch2a_patch0', data=res3d_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2a_patch0 = bn3d_branch2a_patch0
res3d_branch2a_relu_patch0 = mx.symbol.Activation(name='res3d_branch2a_relu_patch0', data=scale3d_branch2a_patch0 , act_type='relu')
res3d_branch2b_patch0 = mx.symbol.Convolution(name='res3d_branch2b_patch0', data=res3d_branch2a_relu_patch0 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2b_patch0 = mx.symbol.BatchNorm(name='bn3d_branch2b_patch0', data=res3d_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2b_patch0 = bn3d_branch2b_patch0
res3d_patch0 = mx.symbol.broadcast_plus(name='res3d_patch0', *[res3c_relu_patch0,scale3d_branch2b_patch0] )
res3d_relu_patch0 = mx.symbol.Activation(name='res3d_relu_patch0', data=res3d_patch0 , act_type='relu')
res4a_branch1_patch0 = mx.symbol.Convolution(name='res4a_branch1_patch0', data=res3d_relu_patch0 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1_patch0 = mx.symbol.BatchNorm(name='bn4a_branch1_patch0', data=res4a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch1_patch0 = bn4a_branch1_patch0
res4a_branch2a_patch0 = mx.symbol.Convolution(name='res4a_branch2a_patch0', data=res3d_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn4a_branch2a_patch0 = mx.symbol.BatchNorm(name='bn4a_branch2a_patch0', data=res4a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2a_patch0 = bn4a_branch2a_patch0
res4a_branch2a_relu_patch0 = mx.symbol.Activation(name='res4a_branch2a_relu_patch0', data=scale4a_branch2a_patch0 , act_type='relu')
res4a_branch2b_patch0 = mx.symbol.Convolution(name='res4a_branch2b_patch0', data=res4a_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b_patch0 = mx.symbol.BatchNorm(name='bn4a_branch2b_patch0', data=res4a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2b_patch0 = bn4a_branch2b_patch0
res4a_patch0 = mx.symbol.broadcast_plus(name='res4a_patch0', *[scale4a_branch1_patch0,scale4a_branch2b_patch0] )
res4a_relu_patch0 = mx.symbol.Activation(name='res4a_relu_patch0', data=res4a_patch0 , act_type='relu')
res4b_branch2a_patch0 = mx.symbol.Convolution(name='res4b_branch2a_patch0', data=res4a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2a_patch0 = mx.symbol.BatchNorm(name='bn4b_branch2a_patch0', data=res4b_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2a_patch0 = bn4b_branch2a_patch0
res4b_branch2a_relu_patch0 = mx.symbol.Activation(name='res4b_branch2a_relu_patch0', data=scale4b_branch2a_patch0 , act_type='relu')
res4b_branch2b_patch0 = mx.symbol.Convolution(name='res4b_branch2b_patch0', data=res4b_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2b_patch0 = mx.symbol.BatchNorm(name='bn4b_branch2b_patch0', data=res4b_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2b_patch0 = bn4b_branch2b_patch0
res4b_patch0 = mx.symbol.broadcast_plus(name='res4b_patch0', *[res4a_relu_patch0,scale4b_branch2b_patch0] )
res4b_relu_patch0 = mx.symbol.Activation(name='res4b_relu_patch0', data=res4b_patch0 , act_type='relu')
res4c_branch2a_patch0 = mx.symbol.Convolution(name='res4c_branch2a_patch0', data=res4b_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2a_patch0 = mx.symbol.BatchNorm(name='bn4c_branch2a_patch0', data=res4c_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2a_patch0 = bn4c_branch2a_patch0
res4c_branch2a_relu_patch0 = mx.symbol.Activation(name='res4c_branch2a_relu_patch0', data=scale4c_branch2a_patch0 , act_type='relu')
res4c_branch2b_patch0 = mx.symbol.Convolution(name='res4c_branch2b_patch0', data=res4c_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2b_patch0 = mx.symbol.BatchNorm(name='bn4c_branch2b_patch0', data=res4c_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2b_patch0 = bn4c_branch2b_patch0
res4c_patch0 = mx.symbol.broadcast_plus(name='res4c_patch0', *[res4b_relu_patch0,scale4c_branch2b_patch0] )
res4c_relu_patch0 = mx.symbol.Activation(name='res4c_relu_patch0', data=res4c_patch0 , act_type='relu')
res4d_branch2a_patch0 = mx.symbol.Convolution(name='res4d_branch2a_patch0', data=res4c_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2a_patch0 = mx.symbol.BatchNorm(name='bn4d_branch2a_patch0', data=res4d_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2a_patch0 = bn4d_branch2a_patch0
res4d_branch2a_relu_patch0 = mx.symbol.Activation(name='res4d_branch2a_relu_patch0', data=scale4d_branch2a_patch0 , act_type='relu')
res4d_branch2b_patch0 = mx.symbol.Convolution(name='res4d_branch2b_patch0', data=res4d_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2b_patch0 = mx.symbol.BatchNorm(name='bn4d_branch2b_patch0', data=res4d_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2b_patch0 = bn4d_branch2b_patch0
res4d_patch0 = mx.symbol.broadcast_plus(name='res4d_patch0', *[res4c_relu_patch0,scale4d_branch2b_patch0] )
res4d_relu_patch0 = mx.symbol.Activation(name='res4d_relu_patch0', data=res4d_patch0 , act_type='relu')
res4e_branch2a_patch0 = mx.symbol.Convolution(name='res4e_branch2a_patch0', data=res4d_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2a_patch0 = mx.symbol.BatchNorm(name='bn4e_branch2a_patch0', data=res4e_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2a_patch0 = bn4e_branch2a_patch0
res4e_branch2a_relu_patch0 = mx.symbol.Activation(name='res4e_branch2a_relu_patch0', data=scale4e_branch2a_patch0 , act_type='relu')
res4e_branch2b_patch0 = mx.symbol.Convolution(name='res4e_branch2b_patch0', data=res4e_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2b_patch0 = mx.symbol.BatchNorm(name='bn4e_branch2b_patch0', data=res4e_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2b_patch0 = bn4e_branch2b_patch0
res4e_patch0 = mx.symbol.broadcast_plus(name='res4e_patch0', *[res4d_relu_patch0,scale4e_branch2b_patch0] )
res4e_relu_patch0 = mx.symbol.Activation(name='res4e_relu_patch0', data=res4e_patch0 , act_type='relu')
res4f_branch2a_patch0 = mx.symbol.Convolution(name='res4f_branch2a_patch0', data=res4e_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2a_patch0 = mx.symbol.BatchNorm(name='bn4f_branch2a_patch0', data=res4f_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2a_patch0 = bn4f_branch2a_patch0
res4f_branch2a_relu_patch0 = mx.symbol.Activation(name='res4f_branch2a_relu_patch0', data=scale4f_branch2a_patch0 , act_type='relu')
res4f_branch2b_patch0 = mx.symbol.Convolution(name='res4f_branch2b_patch0', data=res4f_branch2a_relu_patch0 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2b_patch0 = mx.symbol.BatchNorm(name='bn4f_branch2b_patch0', data=res4f_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2b_patch0 = bn4f_branch2b_patch0
res4f_patch0 = mx.symbol.broadcast_plus(name='res4f_patch0', *[res4e_relu_patch0,scale4f_branch2b_patch0] )
res4f_relu_patch0 = mx.symbol.Activation(name='res4f_relu_patch0', data=res4f_patch0 , act_type='relu')
res5a_branch1_patch0 = mx.symbol.Convolution(name='res5a_branch1_patch0', data=res4f_relu_patch0 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn5a_branch1_patch0 = mx.symbol.BatchNorm(name='bn5a_branch1_patch0', data=res5a_branch1_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch1_patch0 = bn5a_branch1_patch0
res5a_branch2a_patch0 = mx.symbol.Convolution(name='res5a_branch2a_patch0', data=res4f_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn5a_branch2a_patch0 = mx.symbol.BatchNorm(name='bn5a_branch2a_patch0', data=res5a_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2a_patch0 = bn5a_branch2a_patch0
res5a_branch2a_relu_patch0 = mx.symbol.Activation(name='res5a_branch2a_relu_patch0', data=scale5a_branch2a_patch0 , act_type='relu')
res5a_branch2b_patch0 = mx.symbol.Convolution(name='res5a_branch2b_patch0', data=res5a_branch2a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5a_branch2b_patch0 = mx.symbol.BatchNorm(name='bn5a_branch2b_patch0', data=res5a_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2b_patch0 = bn5a_branch2b_patch0
res5a_patch0 = mx.symbol.broadcast_plus(name='res5a_patch0', *[scale5a_branch1_patch0,scale5a_branch2b_patch0] )
res5a_relu_patch0 = mx.symbol.Activation(name='res5a_relu_patch0', data=res5a_patch0 , act_type='relu')
res5b_branch2a_patch0 = mx.symbol.Convolution(name='res5b_branch2a_patch0', data=res5a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2a_patch0 = mx.symbol.BatchNorm(name='bn5b_branch2a_patch0', data=res5b_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2a_patch0 = bn5b_branch2a_patch0
res5b_branch2a_relu_patch0 = mx.symbol.Activation(name='res5b_branch2a_relu_patch0', data=scale5b_branch2a_patch0 , act_type='relu')
res5b_branch2b_patch0 = mx.symbol.Convolution(name='res5b_branch2b_patch0', data=res5b_branch2a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2b_patch0 = mx.symbol.BatchNorm(name='bn5b_branch2b_patch0', data=res5b_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2b_patch0 = bn5b_branch2b_patch0
res5b_patch0 = mx.symbol.broadcast_plus(name='res5b_patch0', *[res5a_relu_patch0,scale5b_branch2b_patch0] )
res5b_relu_patch0 = mx.symbol.Activation(name='res5b_relu_patch0', data=res5b_patch0 , act_type='relu')
res5c_branch2a_patch0 = mx.symbol.Convolution(name='res5c_branch2a_patch0', data=res5b_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2a_patch0 = mx.symbol.BatchNorm(name='bn5c_branch2a_patch0', data=res5c_branch2a_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2a_patch0 = bn5c_branch2a_patch0
res5c_branch2a_relu_patch0 = mx.symbol.Activation(name='res5c_branch2a_relu_patch0', data=scale5c_branch2a_patch0 , act_type='relu')
res5c_branch2b_patch0 = mx.symbol.Convolution(name='res5c_branch2b_patch0', data=res5c_branch2a_relu_patch0 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2b_patch0 = mx.symbol.BatchNorm(name='bn5c_branch2b_patch0', data=res5c_branch2b_patch0 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2b_patch0 = bn5c_branch2b_patch0
res5c_patch0 = mx.symbol.broadcast_plus(name='res5c_patch0', *[res5b_relu_patch0,scale5c_branch2b_patch0] )
res5c_relu_patch0 = mx.symbol.Activation(name='res5c_relu_patch0', data=res5c_patch0 , act_type='relu')
pool5_patch0 = mx.symbol.Pooling(name='pool5_patch0', data=res5c_relu_patch0 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch1 = mx.symbol.Convolution(name='conv1_patch1', data=data_patch1 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
bn_conv1_patch1 = mx.symbol.BatchNorm(name='bn_conv1_patch1', data=conv1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale_conv1_patch1 = bn_conv1_patch1
conv1_relu_patch1 = mx.symbol.Activation(name='conv1_relu_patch1', data=scale_conv1_patch1 , act_type='relu')
pool1_patch1 = mx.symbol.Pooling(name='pool1_patch1', data=conv1_relu_patch1 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch1 = mx.symbol.Convolution(name='res2a_branch1_patch1', data=pool1_patch1 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1_patch1 = mx.symbol.BatchNorm(name='bn2a_branch1_patch1', data=res2a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch1_patch1 = bn2a_branch1_patch1
res2a_branch2a_patch1 = mx.symbol.Convolution(name='res2a_branch2a_patch1', data=pool1_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2a_patch1 = mx.symbol.BatchNorm(name='bn2a_branch2a_patch1', data=res2a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2a_patch1 = bn2a_branch2a_patch1
res2a_branch2a_relu_patch1 = mx.symbol.Activation(name='res2a_branch2a_relu_patch1', data=scale2a_branch2a_patch1 , act_type='relu')
res2a_branch2b_patch1 = mx.symbol.Convolution(name='res2a_branch2b_patch1', data=res2a_branch2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b_patch1 = mx.symbol.BatchNorm(name='bn2a_branch2b_patch1', data=res2a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2b_patch1 = bn2a_branch2b_patch1
res2a_patch1 = mx.symbol.broadcast_plus(name='res2a_patch1', *[scale2a_branch1_patch1,scale2a_branch2b_patch1] )
res2a_relu_patch1 = mx.symbol.Activation(name='res2a_relu_patch1', data=res2a_patch1 , act_type='relu')
res2b_branch2a_patch1 = mx.symbol.Convolution(name='res2b_branch2a_patch1', data=res2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2a_patch1 = mx.symbol.BatchNorm(name='bn2b_branch2a_patch1', data=res2b_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2a_patch1 = bn2b_branch2a_patch1
res2b_branch2a_relu_patch1 = mx.symbol.Activation(name='res2b_branch2a_relu_patch1', data=scale2b_branch2a_patch1 , act_type='relu')
res2b_branch2b_patch1 = mx.symbol.Convolution(name='res2b_branch2b_patch1', data=res2b_branch2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b_patch1 = mx.symbol.BatchNorm(name='bn2b_branch2b_patch1', data=res2b_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2b_patch1 = bn2b_branch2b_patch1
res2b_patch1 = mx.symbol.broadcast_plus(name='res2b_patch1', *[res2a_relu_patch1,scale2b_branch2b_patch1] )
res2b_relu_patch1 = mx.symbol.Activation(name='res2b_relu_patch1', data=res2b_patch1 , act_type='relu')
res2c_branch2a_patch1 = mx.symbol.Convolution(name='res2c_branch2a_patch1', data=res2b_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2a_patch1 = mx.symbol.BatchNorm(name='bn2c_branch2a_patch1', data=res2c_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2a_patch1 = bn2c_branch2a_patch1
res2c_branch2a_relu_patch1 = mx.symbol.Activation(name='res2c_branch2a_relu_patch1', data=scale2c_branch2a_patch1 , act_type='relu')
res2c_branch2b_patch1 = mx.symbol.Convolution(name='res2c_branch2b_patch1', data=res2c_branch2a_relu_patch1 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b_patch1 = mx.symbol.BatchNorm(name='bn2c_branch2b_patch1', data=res2c_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2b_patch1 = bn2c_branch2b_patch1
res2c_patch1 = mx.symbol.broadcast_plus(name='res2c_patch1', *[res2b_relu_patch1,scale2c_branch2b_patch1] )
res2c_relu_patch1 = mx.symbol.Activation(name='res2c_relu_patch1', data=res2c_patch1 , act_type='relu')
res3a_branch1_patch1 = mx.symbol.Convolution(name='res3a_branch1_patch1', data=res2c_relu_patch1 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1_patch1 = mx.symbol.BatchNorm(name='bn3a_branch1_patch1', data=res3a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch1_patch1 = bn3a_branch1_patch1
res3a_branch2a_patch1 = mx.symbol.Convolution(name='res3a_branch2a_patch1', data=res2c_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn3a_branch2a_patch1 = mx.symbol.BatchNorm(name='bn3a_branch2a_patch1', data=res3a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2a_patch1 = bn3a_branch2a_patch1
res3a_branch2a_relu_patch1 = mx.symbol.Activation(name='res3a_branch2a_relu_patch1', data=scale3a_branch2a_patch1 , act_type='relu')
res3a_branch2b_patch1 = mx.symbol.Convolution(name='res3a_branch2b_patch1', data=res3a_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b_patch1 = mx.symbol.BatchNorm(name='bn3a_branch2b_patch1', data=res3a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2b_patch1 = bn3a_branch2b_patch1
res3a_patch1 = mx.symbol.broadcast_plus(name='res3a_patch1', *[scale3a_branch1_patch1,scale3a_branch2b_patch1] )
res3a_relu_patch1 = mx.symbol.Activation(name='res3a_relu_patch1', data=res3a_patch1 , act_type='relu')
res3b_branch2a_patch1 = mx.symbol.Convolution(name='res3b_branch2a_patch1', data=res3a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2a_patch1 = mx.symbol.BatchNorm(name='bn3b_branch2a_patch1', data=res3b_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2a_patch1 = bn3b_branch2a_patch1
res3b_branch2a_relu_patch1 = mx.symbol.Activation(name='res3b_branch2a_relu_patch1', data=scale3b_branch2a_patch1 , act_type='relu')
res3b_branch2b_patch1 = mx.symbol.Convolution(name='res3b_branch2b_patch1', data=res3b_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2b_patch1 = mx.symbol.BatchNorm(name='bn3b_branch2b_patch1', data=res3b_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2b_patch1 = bn3b_branch2b_patch1
res3b_patch1 = mx.symbol.broadcast_plus(name='res3b_patch1', *[res3a_relu_patch1,scale3b_branch2b_patch1] )
res3b_relu_patch1 = mx.symbol.Activation(name='res3b_relu_patch1', data=res3b_patch1 , act_type='relu')
res3c_branch2a_patch1 = mx.symbol.Convolution(name='res3c_branch2a_patch1', data=res3b_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2a_patch1 = mx.symbol.BatchNorm(name='bn3c_branch2a_patch1', data=res3c_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2a_patch1 = bn3c_branch2a_patch1
res3c_branch2a_relu_patch1 = mx.symbol.Activation(name='res3c_branch2a_relu_patch1', data=scale3c_branch2a_patch1 , act_type='relu')
res3c_branch2b_patch1 = mx.symbol.Convolution(name='res3c_branch2b_patch1', data=res3c_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2b_patch1 = mx.symbol.BatchNorm(name='bn3c_branch2b_patch1', data=res3c_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2b_patch1 = bn3c_branch2b_patch1
res3c_patch1 = mx.symbol.broadcast_plus(name='res3c_patch1', *[res3b_relu_patch1,scale3c_branch2b_patch1] )
res3c_relu_patch1 = mx.symbol.Activation(name='res3c_relu_patch1', data=res3c_patch1 , act_type='relu')
res3d_branch2a_patch1 = mx.symbol.Convolution(name='res3d_branch2a_patch1', data=res3c_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2a_patch1 = mx.symbol.BatchNorm(name='bn3d_branch2a_patch1', data=res3d_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2a_patch1 = bn3d_branch2a_patch1
res3d_branch2a_relu_patch1 = mx.symbol.Activation(name='res3d_branch2a_relu_patch1', data=scale3d_branch2a_patch1 , act_type='relu')
res3d_branch2b_patch1 = mx.symbol.Convolution(name='res3d_branch2b_patch1', data=res3d_branch2a_relu_patch1 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2b_patch1 = mx.symbol.BatchNorm(name='bn3d_branch2b_patch1', data=res3d_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2b_patch1 = bn3d_branch2b_patch1
res3d_patch1 = mx.symbol.broadcast_plus(name='res3d_patch1', *[res3c_relu_patch1,scale3d_branch2b_patch1] )
res3d_relu_patch1 = mx.symbol.Activation(name='res3d_relu_patch1', data=res3d_patch1 , act_type='relu')
res4a_branch1_patch1 = mx.symbol.Convolution(name='res4a_branch1_patch1', data=res3d_relu_patch1 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1_patch1 = mx.symbol.BatchNorm(name='bn4a_branch1_patch1', data=res4a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch1_patch1 = bn4a_branch1_patch1
res4a_branch2a_patch1 = mx.symbol.Convolution(name='res4a_branch2a_patch1', data=res3d_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn4a_branch2a_patch1 = mx.symbol.BatchNorm(name='bn4a_branch2a_patch1', data=res4a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2a_patch1 = bn4a_branch2a_patch1
res4a_branch2a_relu_patch1 = mx.symbol.Activation(name='res4a_branch2a_relu_patch1', data=scale4a_branch2a_patch1 , act_type='relu')
res4a_branch2b_patch1 = mx.symbol.Convolution(name='res4a_branch2b_patch1', data=res4a_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b_patch1 = mx.symbol.BatchNorm(name='bn4a_branch2b_patch1', data=res4a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2b_patch1 = bn4a_branch2b_patch1
res4a_patch1 = mx.symbol.broadcast_plus(name='res4a_patch1', *[scale4a_branch1_patch1,scale4a_branch2b_patch1] )
res4a_relu_patch1 = mx.symbol.Activation(name='res4a_relu_patch1', data=res4a_patch1 , act_type='relu')
res4b_branch2a_patch1 = mx.symbol.Convolution(name='res4b_branch2a_patch1', data=res4a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2a_patch1 = mx.symbol.BatchNorm(name='bn4b_branch2a_patch1', data=res4b_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2a_patch1 = bn4b_branch2a_patch1
res4b_branch2a_relu_patch1 = mx.symbol.Activation(name='res4b_branch2a_relu_patch1', data=scale4b_branch2a_patch1 , act_type='relu')
res4b_branch2b_patch1 = mx.symbol.Convolution(name='res4b_branch2b_patch1', data=res4b_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2b_patch1 = mx.symbol.BatchNorm(name='bn4b_branch2b_patch1', data=res4b_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2b_patch1 = bn4b_branch2b_patch1
res4b_patch1 = mx.symbol.broadcast_plus(name='res4b_patch1', *[res4a_relu_patch1,scale4b_branch2b_patch1] )
res4b_relu_patch1 = mx.symbol.Activation(name='res4b_relu_patch1', data=res4b_patch1 , act_type='relu')
res4c_branch2a_patch1 = mx.symbol.Convolution(name='res4c_branch2a_patch1', data=res4b_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2a_patch1 = mx.symbol.BatchNorm(name='bn4c_branch2a_patch1', data=res4c_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2a_patch1 = bn4c_branch2a_patch1
res4c_branch2a_relu_patch1 = mx.symbol.Activation(name='res4c_branch2a_relu_patch1', data=scale4c_branch2a_patch1 , act_type='relu')
res4c_branch2b_patch1 = mx.symbol.Convolution(name='res4c_branch2b_patch1', data=res4c_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2b_patch1 = mx.symbol.BatchNorm(name='bn4c_branch2b_patch1', data=res4c_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2b_patch1 = bn4c_branch2b_patch1
res4c_patch1 = mx.symbol.broadcast_plus(name='res4c_patch1', *[res4b_relu_patch1,scale4c_branch2b_patch1] )
res4c_relu_patch1 = mx.symbol.Activation(name='res4c_relu_patch1', data=res4c_patch1 , act_type='relu')
res4d_branch2a_patch1 = mx.symbol.Convolution(name='res4d_branch2a_patch1', data=res4c_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2a_patch1 = mx.symbol.BatchNorm(name='bn4d_branch2a_patch1', data=res4d_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2a_patch1 = bn4d_branch2a_patch1
res4d_branch2a_relu_patch1 = mx.symbol.Activation(name='res4d_branch2a_relu_patch1', data=scale4d_branch2a_patch1 , act_type='relu')
res4d_branch2b_patch1 = mx.symbol.Convolution(name='res4d_branch2b_patch1', data=res4d_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2b_patch1 = mx.symbol.BatchNorm(name='bn4d_branch2b_patch1', data=res4d_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2b_patch1 = bn4d_branch2b_patch1
res4d_patch1 = mx.symbol.broadcast_plus(name='res4d_patch1', *[res4c_relu_patch1,scale4d_branch2b_patch1] )
res4d_relu_patch1 = mx.symbol.Activation(name='res4d_relu_patch1', data=res4d_patch1 , act_type='relu')
res4e_branch2a_patch1 = mx.symbol.Convolution(name='res4e_branch2a_patch1', data=res4d_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2a_patch1 = mx.symbol.BatchNorm(name='bn4e_branch2a_patch1', data=res4e_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2a_patch1 = bn4e_branch2a_patch1
res4e_branch2a_relu_patch1 = mx.symbol.Activation(name='res4e_branch2a_relu_patch1', data=scale4e_branch2a_patch1 , act_type='relu')
res4e_branch2b_patch1 = mx.symbol.Convolution(name='res4e_branch2b_patch1', data=res4e_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2b_patch1 = mx.symbol.BatchNorm(name='bn4e_branch2b_patch1', data=res4e_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2b_patch1 = bn4e_branch2b_patch1
res4e_patch1 = mx.symbol.broadcast_plus(name='res4e_patch1', *[res4d_relu_patch1,scale4e_branch2b_patch1] )
res4e_relu_patch1 = mx.symbol.Activation(name='res4e_relu_patch1', data=res4e_patch1 , act_type='relu')
res4f_branch2a_patch1 = mx.symbol.Convolution(name='res4f_branch2a_patch1', data=res4e_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2a_patch1 = mx.symbol.BatchNorm(name='bn4f_branch2a_patch1', data=res4f_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2a_patch1 = bn4f_branch2a_patch1
res4f_branch2a_relu_patch1 = mx.symbol.Activation(name='res4f_branch2a_relu_patch1', data=scale4f_branch2a_patch1 , act_type='relu')
res4f_branch2b_patch1 = mx.symbol.Convolution(name='res4f_branch2b_patch1', data=res4f_branch2a_relu_patch1 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2b_patch1 = mx.symbol.BatchNorm(name='bn4f_branch2b_patch1', data=res4f_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2b_patch1 = bn4f_branch2b_patch1
res4f_patch1 = mx.symbol.broadcast_plus(name='res4f_patch1', *[res4e_relu_patch1,scale4f_branch2b_patch1] )
res4f_relu_patch1 = mx.symbol.Activation(name='res4f_relu_patch1', data=res4f_patch1 , act_type='relu')
res5a_branch1_patch1 = mx.symbol.Convolution(name='res5a_branch1_patch1', data=res4f_relu_patch1 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn5a_branch1_patch1 = mx.symbol.BatchNorm(name='bn5a_branch1_patch1', data=res5a_branch1_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch1_patch1 = bn5a_branch1_patch1
res5a_branch2a_patch1 = mx.symbol.Convolution(name='res5a_branch2a_patch1', data=res4f_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn5a_branch2a_patch1 = mx.symbol.BatchNorm(name='bn5a_branch2a_patch1', data=res5a_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2a_patch1 = bn5a_branch2a_patch1
res5a_branch2a_relu_patch1 = mx.symbol.Activation(name='res5a_branch2a_relu_patch1', data=scale5a_branch2a_patch1 , act_type='relu')
res5a_branch2b_patch1 = mx.symbol.Convolution(name='res5a_branch2b_patch1', data=res5a_branch2a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5a_branch2b_patch1 = mx.symbol.BatchNorm(name='bn5a_branch2b_patch1', data=res5a_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2b_patch1 = bn5a_branch2b_patch1
res5a_patch1 = mx.symbol.broadcast_plus(name='res5a_patch1', *[scale5a_branch1_patch1,scale5a_branch2b_patch1] )
res5a_relu_patch1 = mx.symbol.Activation(name='res5a_relu_patch1', data=res5a_patch1 , act_type='relu')
res5b_branch2a_patch1 = mx.symbol.Convolution(name='res5b_branch2a_patch1', data=res5a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2a_patch1 = mx.symbol.BatchNorm(name='bn5b_branch2a_patch1', data=res5b_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2a_patch1 = bn5b_branch2a_patch1
res5b_branch2a_relu_patch1 = mx.symbol.Activation(name='res5b_branch2a_relu_patch1', data=scale5b_branch2a_patch1 , act_type='relu')
res5b_branch2b_patch1 = mx.symbol.Convolution(name='res5b_branch2b_patch1', data=res5b_branch2a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2b_patch1 = mx.symbol.BatchNorm(name='bn5b_branch2b_patch1', data=res5b_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2b_patch1 = bn5b_branch2b_patch1
res5b_patch1 = mx.symbol.broadcast_plus(name='res5b_patch1', *[res5a_relu_patch1,scale5b_branch2b_patch1] )
res5b_relu_patch1 = mx.symbol.Activation(name='res5b_relu_patch1', data=res5b_patch1 , act_type='relu')
res5c_branch2a_patch1 = mx.symbol.Convolution(name='res5c_branch2a_patch1', data=res5b_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2a_patch1 = mx.symbol.BatchNorm(name='bn5c_branch2a_patch1', data=res5c_branch2a_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2a_patch1 = bn5c_branch2a_patch1
res5c_branch2a_relu_patch1 = mx.symbol.Activation(name='res5c_branch2a_relu_patch1', data=scale5c_branch2a_patch1 , act_type='relu')
res5c_branch2b_patch1 = mx.symbol.Convolution(name='res5c_branch2b_patch1', data=res5c_branch2a_relu_patch1 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2b_patch1 = mx.symbol.BatchNorm(name='bn5c_branch2b_patch1', data=res5c_branch2b_patch1 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2b_patch1 = bn5c_branch2b_patch1
res5c_patch1 = mx.symbol.broadcast_plus(name='res5c_patch1', *[res5b_relu_patch1,scale5c_branch2b_patch1] )
res5c_relu_patch1 = mx.symbol.Activation(name='res5c_relu_patch1', data=res5c_patch1 , act_type='relu')
pool5_patch1 = mx.symbol.Pooling(name='pool5_patch1', data=res5c_relu_patch1 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch2 = mx.symbol.Convolution(name='conv1_patch2', data=data_patch2 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
bn_conv1_patch2 = mx.symbol.BatchNorm(name='bn_conv1_patch2', data=conv1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale_conv1_patch2 = bn_conv1_patch2
conv1_relu_patch2 = mx.symbol.Activation(name='conv1_relu_patch2', data=scale_conv1_patch2 , act_type='relu')
pool1_patch2 = mx.symbol.Pooling(name='pool1_patch2', data=conv1_relu_patch2 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch2 = mx.symbol.Convolution(name='res2a_branch1_patch2', data=pool1_patch2 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1_patch2 = mx.symbol.BatchNorm(name='bn2a_branch1_patch2', data=res2a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch1_patch2 = bn2a_branch1_patch2
res2a_branch2a_patch2 = mx.symbol.Convolution(name='res2a_branch2a_patch2', data=pool1_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2a_patch2 = mx.symbol.BatchNorm(name='bn2a_branch2a_patch2', data=res2a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2a_patch2 = bn2a_branch2a_patch2
res2a_branch2a_relu_patch2 = mx.symbol.Activation(name='res2a_branch2a_relu_patch2', data=scale2a_branch2a_patch2 , act_type='relu')
res2a_branch2b_patch2 = mx.symbol.Convolution(name='res2a_branch2b_patch2', data=res2a_branch2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b_patch2 = mx.symbol.BatchNorm(name='bn2a_branch2b_patch2', data=res2a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2b_patch2 = bn2a_branch2b_patch2
res2a_patch2 = mx.symbol.broadcast_plus(name='res2a_patch2', *[scale2a_branch1_patch2,scale2a_branch2b_patch2] )
res2a_relu_patch2 = mx.symbol.Activation(name='res2a_relu_patch2', data=res2a_patch2 , act_type='relu')
res2b_branch2a_patch2 = mx.symbol.Convolution(name='res2b_branch2a_patch2', data=res2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2a_patch2 = mx.symbol.BatchNorm(name='bn2b_branch2a_patch2', data=res2b_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2a_patch2 = bn2b_branch2a_patch2
res2b_branch2a_relu_patch2 = mx.symbol.Activation(name='res2b_branch2a_relu_patch2', data=scale2b_branch2a_patch2 , act_type='relu')
res2b_branch2b_patch2 = mx.symbol.Convolution(name='res2b_branch2b_patch2', data=res2b_branch2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b_patch2 = mx.symbol.BatchNorm(name='bn2b_branch2b_patch2', data=res2b_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2b_patch2 = bn2b_branch2b_patch2
res2b_patch2 = mx.symbol.broadcast_plus(name='res2b_patch2', *[res2a_relu_patch2,scale2b_branch2b_patch2] )
res2b_relu_patch2 = mx.symbol.Activation(name='res2b_relu_patch2', data=res2b_patch2 , act_type='relu')
res2c_branch2a_patch2 = mx.symbol.Convolution(name='res2c_branch2a_patch2', data=res2b_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2a_patch2 = mx.symbol.BatchNorm(name='bn2c_branch2a_patch2', data=res2c_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2a_patch2 = bn2c_branch2a_patch2
res2c_branch2a_relu_patch2 = mx.symbol.Activation(name='res2c_branch2a_relu_patch2', data=scale2c_branch2a_patch2 , act_type='relu')
res2c_branch2b_patch2 = mx.symbol.Convolution(name='res2c_branch2b_patch2', data=res2c_branch2a_relu_patch2 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b_patch2 = mx.symbol.BatchNorm(name='bn2c_branch2b_patch2', data=res2c_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2b_patch2 = bn2c_branch2b_patch2
res2c_patch2 = mx.symbol.broadcast_plus(name='res2c_patch2', *[res2b_relu_patch2,scale2c_branch2b_patch2] )
res2c_relu_patch2 = mx.symbol.Activation(name='res2c_relu_patch2', data=res2c_patch2 , act_type='relu')
res3a_branch1_patch2 = mx.symbol.Convolution(name='res3a_branch1_patch2', data=res2c_relu_patch2 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1_patch2 = mx.symbol.BatchNorm(name='bn3a_branch1_patch2', data=res3a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch1_patch2 = bn3a_branch1_patch2
res3a_branch2a_patch2 = mx.symbol.Convolution(name='res3a_branch2a_patch2', data=res2c_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn3a_branch2a_patch2 = mx.symbol.BatchNorm(name='bn3a_branch2a_patch2', data=res3a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2a_patch2 = bn3a_branch2a_patch2
res3a_branch2a_relu_patch2 = mx.symbol.Activation(name='res3a_branch2a_relu_patch2', data=scale3a_branch2a_patch2 , act_type='relu')
res3a_branch2b_patch2 = mx.symbol.Convolution(name='res3a_branch2b_patch2', data=res3a_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b_patch2 = mx.symbol.BatchNorm(name='bn3a_branch2b_patch2', data=res3a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2b_patch2 = bn3a_branch2b_patch2
res3a_patch2 = mx.symbol.broadcast_plus(name='res3a_patch2', *[scale3a_branch1_patch2,scale3a_branch2b_patch2] )
res3a_relu_patch2 = mx.symbol.Activation(name='res3a_relu_patch2', data=res3a_patch2 , act_type='relu')
res3b_branch2a_patch2 = mx.symbol.Convolution(name='res3b_branch2a_patch2', data=res3a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2a_patch2 = mx.symbol.BatchNorm(name='bn3b_branch2a_patch2', data=res3b_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2a_patch2 = bn3b_branch2a_patch2
res3b_branch2a_relu_patch2 = mx.symbol.Activation(name='res3b_branch2a_relu_patch2', data=scale3b_branch2a_patch2 , act_type='relu')
res3b_branch2b_patch2 = mx.symbol.Convolution(name='res3b_branch2b_patch2', data=res3b_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2b_patch2 = mx.symbol.BatchNorm(name='bn3b_branch2b_patch2', data=res3b_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2b_patch2 = bn3b_branch2b_patch2
res3b_patch2 = mx.symbol.broadcast_plus(name='res3b_patch2', *[res3a_relu_patch2,scale3b_branch2b_patch2] )
res3b_relu_patch2 = mx.symbol.Activation(name='res3b_relu_patch2', data=res3b_patch2 , act_type='relu')
res3c_branch2a_patch2 = mx.symbol.Convolution(name='res3c_branch2a_patch2', data=res3b_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2a_patch2 = mx.symbol.BatchNorm(name='bn3c_branch2a_patch2', data=res3c_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2a_patch2 = bn3c_branch2a_patch2
res3c_branch2a_relu_patch2 = mx.symbol.Activation(name='res3c_branch2a_relu_patch2', data=scale3c_branch2a_patch2 , act_type='relu')
res3c_branch2b_patch2 = mx.symbol.Convolution(name='res3c_branch2b_patch2', data=res3c_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2b_patch2 = mx.symbol.BatchNorm(name='bn3c_branch2b_patch2', data=res3c_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2b_patch2 = bn3c_branch2b_patch2
res3c_patch2 = mx.symbol.broadcast_plus(name='res3c_patch2', *[res3b_relu_patch2,scale3c_branch2b_patch2] )
res3c_relu_patch2 = mx.symbol.Activation(name='res3c_relu_patch2', data=res3c_patch2 , act_type='relu')
res3d_branch2a_patch2 = mx.symbol.Convolution(name='res3d_branch2a_patch2', data=res3c_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2a_patch2 = mx.symbol.BatchNorm(name='bn3d_branch2a_patch2', data=res3d_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2a_patch2 = bn3d_branch2a_patch2
res3d_branch2a_relu_patch2 = mx.symbol.Activation(name='res3d_branch2a_relu_patch2', data=scale3d_branch2a_patch2 , act_type='relu')
res3d_branch2b_patch2 = mx.symbol.Convolution(name='res3d_branch2b_patch2', data=res3d_branch2a_relu_patch2 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2b_patch2 = mx.symbol.BatchNorm(name='bn3d_branch2b_patch2', data=res3d_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2b_patch2 = bn3d_branch2b_patch2
res3d_patch2 = mx.symbol.broadcast_plus(name='res3d_patch2', *[res3c_relu_patch2,scale3d_branch2b_patch2] )
res3d_relu_patch2 = mx.symbol.Activation(name='res3d_relu_patch2', data=res3d_patch2 , act_type='relu')
res4a_branch1_patch2 = mx.symbol.Convolution(name='res4a_branch1_patch2', data=res3d_relu_patch2 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1_patch2 = mx.symbol.BatchNorm(name='bn4a_branch1_patch2', data=res4a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch1_patch2 = bn4a_branch1_patch2
res4a_branch2a_patch2 = mx.symbol.Convolution(name='res4a_branch2a_patch2', data=res3d_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn4a_branch2a_patch2 = mx.symbol.BatchNorm(name='bn4a_branch2a_patch2', data=res4a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2a_patch2 = bn4a_branch2a_patch2
res4a_branch2a_relu_patch2 = mx.symbol.Activation(name='res4a_branch2a_relu_patch2', data=scale4a_branch2a_patch2 , act_type='relu')
res4a_branch2b_patch2 = mx.symbol.Convolution(name='res4a_branch2b_patch2', data=res4a_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b_patch2 = mx.symbol.BatchNorm(name='bn4a_branch2b_patch2', data=res4a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2b_patch2 = bn4a_branch2b_patch2
res4a_patch2 = mx.symbol.broadcast_plus(name='res4a_patch2', *[scale4a_branch1_patch2,scale4a_branch2b_patch2] )
res4a_relu_patch2 = mx.symbol.Activation(name='res4a_relu_patch2', data=res4a_patch2 , act_type='relu')
res4b_branch2a_patch2 = mx.symbol.Convolution(name='res4b_branch2a_patch2', data=res4a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2a_patch2 = mx.symbol.BatchNorm(name='bn4b_branch2a_patch2', data=res4b_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2a_patch2 = bn4b_branch2a_patch2
res4b_branch2a_relu_patch2 = mx.symbol.Activation(name='res4b_branch2a_relu_patch2', data=scale4b_branch2a_patch2 , act_type='relu')
res4b_branch2b_patch2 = mx.symbol.Convolution(name='res4b_branch2b_patch2', data=res4b_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2b_patch2 = mx.symbol.BatchNorm(name='bn4b_branch2b_patch2', data=res4b_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2b_patch2 = bn4b_branch2b_patch2
res4b_patch2 = mx.symbol.broadcast_plus(name='res4b_patch2', *[res4a_relu_patch2,scale4b_branch2b_patch2] )
res4b_relu_patch2 = mx.symbol.Activation(name='res4b_relu_patch2', data=res4b_patch2 , act_type='relu')
res4c_branch2a_patch2 = mx.symbol.Convolution(name='res4c_branch2a_patch2', data=res4b_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2a_patch2 = mx.symbol.BatchNorm(name='bn4c_branch2a_patch2', data=res4c_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2a_patch2 = bn4c_branch2a_patch2
res4c_branch2a_relu_patch2 = mx.symbol.Activation(name='res4c_branch2a_relu_patch2', data=scale4c_branch2a_patch2 , act_type='relu')
res4c_branch2b_patch2 = mx.symbol.Convolution(name='res4c_branch2b_patch2', data=res4c_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2b_patch2 = mx.symbol.BatchNorm(name='bn4c_branch2b_patch2', data=res4c_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2b_patch2 = bn4c_branch2b_patch2
res4c_patch2 = mx.symbol.broadcast_plus(name='res4c_patch2', *[res4b_relu_patch2,scale4c_branch2b_patch2] )
res4c_relu_patch2 = mx.symbol.Activation(name='res4c_relu_patch2', data=res4c_patch2 , act_type='relu')
res4d_branch2a_patch2 = mx.symbol.Convolution(name='res4d_branch2a_patch2', data=res4c_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2a_patch2 = mx.symbol.BatchNorm(name='bn4d_branch2a_patch2', data=res4d_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2a_patch2 = bn4d_branch2a_patch2
res4d_branch2a_relu_patch2 = mx.symbol.Activation(name='res4d_branch2a_relu_patch2', data=scale4d_branch2a_patch2 , act_type='relu')
res4d_branch2b_patch2 = mx.symbol.Convolution(name='res4d_branch2b_patch2', data=res4d_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2b_patch2 = mx.symbol.BatchNorm(name='bn4d_branch2b_patch2', data=res4d_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2b_patch2 = bn4d_branch2b_patch2
res4d_patch2 = mx.symbol.broadcast_plus(name='res4d_patch2', *[res4c_relu_patch2,scale4d_branch2b_patch2] )
res4d_relu_patch2 = mx.symbol.Activation(name='res4d_relu_patch2', data=res4d_patch2 , act_type='relu')
res4e_branch2a_patch2 = mx.symbol.Convolution(name='res4e_branch2a_patch2', data=res4d_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2a_patch2 = mx.symbol.BatchNorm(name='bn4e_branch2a_patch2', data=res4e_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2a_patch2 = bn4e_branch2a_patch2
res4e_branch2a_relu_patch2 = mx.symbol.Activation(name='res4e_branch2a_relu_patch2', data=scale4e_branch2a_patch2 , act_type='relu')
res4e_branch2b_patch2 = mx.symbol.Convolution(name='res4e_branch2b_patch2', data=res4e_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2b_patch2 = mx.symbol.BatchNorm(name='bn4e_branch2b_patch2', data=res4e_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2b_patch2 = bn4e_branch2b_patch2
res4e_patch2 = mx.symbol.broadcast_plus(name='res4e_patch2', *[res4d_relu_patch2,scale4e_branch2b_patch2] )
res4e_relu_patch2 = mx.symbol.Activation(name='res4e_relu_patch2', data=res4e_patch2 , act_type='relu')
res4f_branch2a_patch2 = mx.symbol.Convolution(name='res4f_branch2a_patch2', data=res4e_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2a_patch2 = mx.symbol.BatchNorm(name='bn4f_branch2a_patch2', data=res4f_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2a_patch2 = bn4f_branch2a_patch2
res4f_branch2a_relu_patch2 = mx.symbol.Activation(name='res4f_branch2a_relu_patch2', data=scale4f_branch2a_patch2 , act_type='relu')
res4f_branch2b_patch2 = mx.symbol.Convolution(name='res4f_branch2b_patch2', data=res4f_branch2a_relu_patch2 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2b_patch2 = mx.symbol.BatchNorm(name='bn4f_branch2b_patch2', data=res4f_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2b_patch2 = bn4f_branch2b_patch2
res4f_patch2 = mx.symbol.broadcast_plus(name='res4f_patch2', *[res4e_relu_patch2,scale4f_branch2b_patch2] )
res4f_relu_patch2 = mx.symbol.Activation(name='res4f_relu_patch2', data=res4f_patch2 , act_type='relu')
res5a_branch1_patch2 = mx.symbol.Convolution(name='res5a_branch1_patch2', data=res4f_relu_patch2 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn5a_branch1_patch2 = mx.symbol.BatchNorm(name='bn5a_branch1_patch2', data=res5a_branch1_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch1_patch2 = bn5a_branch1_patch2
res5a_branch2a_patch2 = mx.symbol.Convolution(name='res5a_branch2a_patch2', data=res4f_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn5a_branch2a_patch2 = mx.symbol.BatchNorm(name='bn5a_branch2a_patch2', data=res5a_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2a_patch2 = bn5a_branch2a_patch2
res5a_branch2a_relu_patch2 = mx.symbol.Activation(name='res5a_branch2a_relu_patch2', data=scale5a_branch2a_patch2 , act_type='relu')
res5a_branch2b_patch2 = mx.symbol.Convolution(name='res5a_branch2b_patch2', data=res5a_branch2a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5a_branch2b_patch2 = mx.symbol.BatchNorm(name='bn5a_branch2b_patch2', data=res5a_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2b_patch2 = bn5a_branch2b_patch2
res5a_patch2 = mx.symbol.broadcast_plus(name='res5a_patch2', *[scale5a_branch1_patch2,scale5a_branch2b_patch2] )
res5a_relu_patch2 = mx.symbol.Activation(name='res5a_relu_patch2', data=res5a_patch2 , act_type='relu')
res5b_branch2a_patch2 = mx.symbol.Convolution(name='res5b_branch2a_patch2', data=res5a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2a_patch2 = mx.symbol.BatchNorm(name='bn5b_branch2a_patch2', data=res5b_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2a_patch2 = bn5b_branch2a_patch2
res5b_branch2a_relu_patch2 = mx.symbol.Activation(name='res5b_branch2a_relu_patch2', data=scale5b_branch2a_patch2 , act_type='relu')
res5b_branch2b_patch2 = mx.symbol.Convolution(name='res5b_branch2b_patch2', data=res5b_branch2a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2b_patch2 = mx.symbol.BatchNorm(name='bn5b_branch2b_patch2', data=res5b_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2b_patch2 = bn5b_branch2b_patch2
res5b_patch2 = mx.symbol.broadcast_plus(name='res5b_patch2', *[res5a_relu_patch2,scale5b_branch2b_patch2] )
res5b_relu_patch2 = mx.symbol.Activation(name='res5b_relu_patch2', data=res5b_patch2 , act_type='relu')
res5c_branch2a_patch2 = mx.symbol.Convolution(name='res5c_branch2a_patch2', data=res5b_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2a_patch2 = mx.symbol.BatchNorm(name='bn5c_branch2a_patch2', data=res5c_branch2a_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2a_patch2 = bn5c_branch2a_patch2
res5c_branch2a_relu_patch2 = mx.symbol.Activation(name='res5c_branch2a_relu_patch2', data=scale5c_branch2a_patch2 , act_type='relu')
res5c_branch2b_patch2 = mx.symbol.Convolution(name='res5c_branch2b_patch2', data=res5c_branch2a_relu_patch2 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2b_patch2 = mx.symbol.BatchNorm(name='bn5c_branch2b_patch2', data=res5c_branch2b_patch2 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2b_patch2 = bn5c_branch2b_patch2
res5c_patch2 = mx.symbol.broadcast_plus(name='res5c_patch2', *[res5b_relu_patch2,scale5c_branch2b_patch2] )
res5c_relu_patch2 = mx.symbol.Activation(name='res5c_relu_patch2', data=res5c_patch2 , act_type='relu')
pool5_patch2 = mx.symbol.Pooling(name='pool5_patch2', data=res5c_relu_patch2 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch3 = mx.symbol.Convolution(name='conv1_patch3', data=data_patch3 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
bn_conv1_patch3 = mx.symbol.BatchNorm(name='bn_conv1_patch3', data=conv1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale_conv1_patch3 = bn_conv1_patch3
conv1_relu_patch3 = mx.symbol.Activation(name='conv1_relu_patch3', data=scale_conv1_patch3 , act_type='relu')
pool1_patch3 = mx.symbol.Pooling(name='pool1_patch3', data=conv1_relu_patch3 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch3 = mx.symbol.Convolution(name='res2a_branch1_patch3', data=pool1_patch3 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1_patch3 = mx.symbol.BatchNorm(name='bn2a_branch1_patch3', data=res2a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch1_patch3 = bn2a_branch1_patch3
res2a_branch2a_patch3 = mx.symbol.Convolution(name='res2a_branch2a_patch3', data=pool1_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2a_patch3 = mx.symbol.BatchNorm(name='bn2a_branch2a_patch3', data=res2a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2a_patch3 = bn2a_branch2a_patch3
res2a_branch2a_relu_patch3 = mx.symbol.Activation(name='res2a_branch2a_relu_patch3', data=scale2a_branch2a_patch3 , act_type='relu')
res2a_branch2b_patch3 = mx.symbol.Convolution(name='res2a_branch2b_patch3', data=res2a_branch2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b_patch3 = mx.symbol.BatchNorm(name='bn2a_branch2b_patch3', data=res2a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2b_patch3 = bn2a_branch2b_patch3
res2a_patch3 = mx.symbol.broadcast_plus(name='res2a_patch3', *[scale2a_branch1_patch3,scale2a_branch2b_patch3] )
res2a_relu_patch3 = mx.symbol.Activation(name='res2a_relu_patch3', data=res2a_patch3 , act_type='relu')
res2b_branch2a_patch3 = mx.symbol.Convolution(name='res2b_branch2a_patch3', data=res2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2a_patch3 = mx.symbol.BatchNorm(name='bn2b_branch2a_patch3', data=res2b_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2a_patch3 = bn2b_branch2a_patch3
res2b_branch2a_relu_patch3 = mx.symbol.Activation(name='res2b_branch2a_relu_patch3', data=scale2b_branch2a_patch3 , act_type='relu')
res2b_branch2b_patch3 = mx.symbol.Convolution(name='res2b_branch2b_patch3', data=res2b_branch2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b_patch3 = mx.symbol.BatchNorm(name='bn2b_branch2b_patch3', data=res2b_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2b_patch3 = bn2b_branch2b_patch3
res2b_patch3 = mx.symbol.broadcast_plus(name='res2b_patch3', *[res2a_relu_patch3,scale2b_branch2b_patch3] )
res2b_relu_patch3 = mx.symbol.Activation(name='res2b_relu_patch3', data=res2b_patch3 , act_type='relu')
res2c_branch2a_patch3 = mx.symbol.Convolution(name='res2c_branch2a_patch3', data=res2b_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2a_patch3 = mx.symbol.BatchNorm(name='bn2c_branch2a_patch3', data=res2c_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2a_patch3 = bn2c_branch2a_patch3
res2c_branch2a_relu_patch3 = mx.symbol.Activation(name='res2c_branch2a_relu_patch3', data=scale2c_branch2a_patch3 , act_type='relu')
res2c_branch2b_patch3 = mx.symbol.Convolution(name='res2c_branch2b_patch3', data=res2c_branch2a_relu_patch3 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b_patch3 = mx.symbol.BatchNorm(name='bn2c_branch2b_patch3', data=res2c_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2b_patch3 = bn2c_branch2b_patch3
res2c_patch3 = mx.symbol.broadcast_plus(name='res2c_patch3', *[res2b_relu_patch3,scale2c_branch2b_patch3] )
res2c_relu_patch3 = mx.symbol.Activation(name='res2c_relu_patch3', data=res2c_patch3 , act_type='relu')
res3a_branch1_patch3 = mx.symbol.Convolution(name='res3a_branch1_patch3', data=res2c_relu_patch3 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1_patch3 = mx.symbol.BatchNorm(name='bn3a_branch1_patch3', data=res3a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch1_patch3 = bn3a_branch1_patch3
res3a_branch2a_patch3 = mx.symbol.Convolution(name='res3a_branch2a_patch3', data=res2c_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn3a_branch2a_patch3 = mx.symbol.BatchNorm(name='bn3a_branch2a_patch3', data=res3a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2a_patch3 = bn3a_branch2a_patch3
res3a_branch2a_relu_patch3 = mx.symbol.Activation(name='res3a_branch2a_relu_patch3', data=scale3a_branch2a_patch3 , act_type='relu')
res3a_branch2b_patch3 = mx.symbol.Convolution(name='res3a_branch2b_patch3', data=res3a_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b_patch3 = mx.symbol.BatchNorm(name='bn3a_branch2b_patch3', data=res3a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2b_patch3 = bn3a_branch2b_patch3
res3a_patch3 = mx.symbol.broadcast_plus(name='res3a_patch3', *[scale3a_branch1_patch3,scale3a_branch2b_patch3] )
res3a_relu_patch3 = mx.symbol.Activation(name='res3a_relu_patch3', data=res3a_patch3 , act_type='relu')
res3b_branch2a_patch3 = mx.symbol.Convolution(name='res3b_branch2a_patch3', data=res3a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2a_patch3 = mx.symbol.BatchNorm(name='bn3b_branch2a_patch3', data=res3b_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2a_patch3 = bn3b_branch2a_patch3
res3b_branch2a_relu_patch3 = mx.symbol.Activation(name='res3b_branch2a_relu_patch3', data=scale3b_branch2a_patch3 , act_type='relu')
res3b_branch2b_patch3 = mx.symbol.Convolution(name='res3b_branch2b_patch3', data=res3b_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2b_patch3 = mx.symbol.BatchNorm(name='bn3b_branch2b_patch3', data=res3b_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2b_patch3 = bn3b_branch2b_patch3
res3b_patch3 = mx.symbol.broadcast_plus(name='res3b_patch3', *[res3a_relu_patch3,scale3b_branch2b_patch3] )
res3b_relu_patch3 = mx.symbol.Activation(name='res3b_relu_patch3', data=res3b_patch3 , act_type='relu')
res3c_branch2a_patch3 = mx.symbol.Convolution(name='res3c_branch2a_patch3', data=res3b_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2a_patch3 = mx.symbol.BatchNorm(name='bn3c_branch2a_patch3', data=res3c_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2a_patch3 = bn3c_branch2a_patch3
res3c_branch2a_relu_patch3 = mx.symbol.Activation(name='res3c_branch2a_relu_patch3', data=scale3c_branch2a_patch3 , act_type='relu')
res3c_branch2b_patch3 = mx.symbol.Convolution(name='res3c_branch2b_patch3', data=res3c_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2b_patch3 = mx.symbol.BatchNorm(name='bn3c_branch2b_patch3', data=res3c_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2b_patch3 = bn3c_branch2b_patch3
res3c_patch3 = mx.symbol.broadcast_plus(name='res3c_patch3', *[res3b_relu_patch3,scale3c_branch2b_patch3] )
res3c_relu_patch3 = mx.symbol.Activation(name='res3c_relu_patch3', data=res3c_patch3 , act_type='relu')
res3d_branch2a_patch3 = mx.symbol.Convolution(name='res3d_branch2a_patch3', data=res3c_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2a_patch3 = mx.symbol.BatchNorm(name='bn3d_branch2a_patch3', data=res3d_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2a_patch3 = bn3d_branch2a_patch3
res3d_branch2a_relu_patch3 = mx.symbol.Activation(name='res3d_branch2a_relu_patch3', data=scale3d_branch2a_patch3 , act_type='relu')
res3d_branch2b_patch3 = mx.symbol.Convolution(name='res3d_branch2b_patch3', data=res3d_branch2a_relu_patch3 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2b_patch3 = mx.symbol.BatchNorm(name='bn3d_branch2b_patch3', data=res3d_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2b_patch3 = bn3d_branch2b_patch3
res3d_patch3 = mx.symbol.broadcast_plus(name='res3d_patch3', *[res3c_relu_patch3,scale3d_branch2b_patch3] )
res3d_relu_patch3 = mx.symbol.Activation(name='res3d_relu_patch3', data=res3d_patch3 , act_type='relu')
res4a_branch1_patch3 = mx.symbol.Convolution(name='res4a_branch1_patch3', data=res3d_relu_patch3 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1_patch3 = mx.symbol.BatchNorm(name='bn4a_branch1_patch3', data=res4a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch1_patch3 = bn4a_branch1_patch3
res4a_branch2a_patch3 = mx.symbol.Convolution(name='res4a_branch2a_patch3', data=res3d_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn4a_branch2a_patch3 = mx.symbol.BatchNorm(name='bn4a_branch2a_patch3', data=res4a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2a_patch3 = bn4a_branch2a_patch3
res4a_branch2a_relu_patch3 = mx.symbol.Activation(name='res4a_branch2a_relu_patch3', data=scale4a_branch2a_patch3 , act_type='relu')
res4a_branch2b_patch3 = mx.symbol.Convolution(name='res4a_branch2b_patch3', data=res4a_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b_patch3 = mx.symbol.BatchNorm(name='bn4a_branch2b_patch3', data=res4a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2b_patch3 = bn4a_branch2b_patch3
res4a_patch3 = mx.symbol.broadcast_plus(name='res4a_patch3', *[scale4a_branch1_patch3,scale4a_branch2b_patch3] )
res4a_relu_patch3 = mx.symbol.Activation(name='res4a_relu_patch3', data=res4a_patch3 , act_type='relu')
res4b_branch2a_patch3 = mx.symbol.Convolution(name='res4b_branch2a_patch3', data=res4a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2a_patch3 = mx.symbol.BatchNorm(name='bn4b_branch2a_patch3', data=res4b_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2a_patch3 = bn4b_branch2a_patch3
res4b_branch2a_relu_patch3 = mx.symbol.Activation(name='res4b_branch2a_relu_patch3', data=scale4b_branch2a_patch3 , act_type='relu')
res4b_branch2b_patch3 = mx.symbol.Convolution(name='res4b_branch2b_patch3', data=res4b_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2b_patch3 = mx.symbol.BatchNorm(name='bn4b_branch2b_patch3', data=res4b_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2b_patch3 = bn4b_branch2b_patch3
res4b_patch3 = mx.symbol.broadcast_plus(name='res4b_patch3', *[res4a_relu_patch3,scale4b_branch2b_patch3] )
res4b_relu_patch3 = mx.symbol.Activation(name='res4b_relu_patch3', data=res4b_patch3 , act_type='relu')
res4c_branch2a_patch3 = mx.symbol.Convolution(name='res4c_branch2a_patch3', data=res4b_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2a_patch3 = mx.symbol.BatchNorm(name='bn4c_branch2a_patch3', data=res4c_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2a_patch3 = bn4c_branch2a_patch3
res4c_branch2a_relu_patch3 = mx.symbol.Activation(name='res4c_branch2a_relu_patch3', data=scale4c_branch2a_patch3 , act_type='relu')
res4c_branch2b_patch3 = mx.symbol.Convolution(name='res4c_branch2b_patch3', data=res4c_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2b_patch3 = mx.symbol.BatchNorm(name='bn4c_branch2b_patch3', data=res4c_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2b_patch3 = bn4c_branch2b_patch3
res4c_patch3 = mx.symbol.broadcast_plus(name='res4c_patch3', *[res4b_relu_patch3,scale4c_branch2b_patch3] )
res4c_relu_patch3 = mx.symbol.Activation(name='res4c_relu_patch3', data=res4c_patch3 , act_type='relu')
res4d_branch2a_patch3 = mx.symbol.Convolution(name='res4d_branch2a_patch3', data=res4c_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2a_patch3 = mx.symbol.BatchNorm(name='bn4d_branch2a_patch3', data=res4d_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2a_patch3 = bn4d_branch2a_patch3
res4d_branch2a_relu_patch3 = mx.symbol.Activation(name='res4d_branch2a_relu_patch3', data=scale4d_branch2a_patch3 , act_type='relu')
res4d_branch2b_patch3 = mx.symbol.Convolution(name='res4d_branch2b_patch3', data=res4d_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2b_patch3 = mx.symbol.BatchNorm(name='bn4d_branch2b_patch3', data=res4d_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2b_patch3 = bn4d_branch2b_patch3
res4d_patch3 = mx.symbol.broadcast_plus(name='res4d_patch3', *[res4c_relu_patch3,scale4d_branch2b_patch3] )
res4d_relu_patch3 = mx.symbol.Activation(name='res4d_relu_patch3', data=res4d_patch3 , act_type='relu')
res4e_branch2a_patch3 = mx.symbol.Convolution(name='res4e_branch2a_patch3', data=res4d_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2a_patch3 = mx.symbol.BatchNorm(name='bn4e_branch2a_patch3', data=res4e_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2a_patch3 = bn4e_branch2a_patch3
res4e_branch2a_relu_patch3 = mx.symbol.Activation(name='res4e_branch2a_relu_patch3', data=scale4e_branch2a_patch3 , act_type='relu')
res4e_branch2b_patch3 = mx.symbol.Convolution(name='res4e_branch2b_patch3', data=res4e_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2b_patch3 = mx.symbol.BatchNorm(name='bn4e_branch2b_patch3', data=res4e_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2b_patch3 = bn4e_branch2b_patch3
res4e_patch3 = mx.symbol.broadcast_plus(name='res4e_patch3', *[res4d_relu_patch3,scale4e_branch2b_patch3] )
res4e_relu_patch3 = mx.symbol.Activation(name='res4e_relu_patch3', data=res4e_patch3 , act_type='relu')
res4f_branch2a_patch3 = mx.symbol.Convolution(name='res4f_branch2a_patch3', data=res4e_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2a_patch3 = mx.symbol.BatchNorm(name='bn4f_branch2a_patch3', data=res4f_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2a_patch3 = bn4f_branch2a_patch3
res4f_branch2a_relu_patch3 = mx.symbol.Activation(name='res4f_branch2a_relu_patch3', data=scale4f_branch2a_patch3 , act_type='relu')
res4f_branch2b_patch3 = mx.symbol.Convolution(name='res4f_branch2b_patch3', data=res4f_branch2a_relu_patch3 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2b_patch3 = mx.symbol.BatchNorm(name='bn4f_branch2b_patch3', data=res4f_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2b_patch3 = bn4f_branch2b_patch3
res4f_patch3 = mx.symbol.broadcast_plus(name='res4f_patch3', *[res4e_relu_patch3,scale4f_branch2b_patch3] )
res4f_relu_patch3 = mx.symbol.Activation(name='res4f_relu_patch3', data=res4f_patch3 , act_type='relu')
res5a_branch1_patch3 = mx.symbol.Convolution(name='res5a_branch1_patch3', data=res4f_relu_patch3 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn5a_branch1_patch3 = mx.symbol.BatchNorm(name='bn5a_branch1_patch3', data=res5a_branch1_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch1_patch3 = bn5a_branch1_patch3
res5a_branch2a_patch3 = mx.symbol.Convolution(name='res5a_branch2a_patch3', data=res4f_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn5a_branch2a_patch3 = mx.symbol.BatchNorm(name='bn5a_branch2a_patch3', data=res5a_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2a_patch3 = bn5a_branch2a_patch3
res5a_branch2a_relu_patch3 = mx.symbol.Activation(name='res5a_branch2a_relu_patch3', data=scale5a_branch2a_patch3 , act_type='relu')
res5a_branch2b_patch3 = mx.symbol.Convolution(name='res5a_branch2b_patch3', data=res5a_branch2a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5a_branch2b_patch3 = mx.symbol.BatchNorm(name='bn5a_branch2b_patch3', data=res5a_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2b_patch3 = bn5a_branch2b_patch3
res5a_patch3 = mx.symbol.broadcast_plus(name='res5a_patch3', *[scale5a_branch1_patch3,scale5a_branch2b_patch3] )
res5a_relu_patch3 = mx.symbol.Activation(name='res5a_relu_patch3', data=res5a_patch3 , act_type='relu')
res5b_branch2a_patch3 = mx.symbol.Convolution(name='res5b_branch2a_patch3', data=res5a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2a_patch3 = mx.symbol.BatchNorm(name='bn5b_branch2a_patch3', data=res5b_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2a_patch3 = bn5b_branch2a_patch3
res5b_branch2a_relu_patch3 = mx.symbol.Activation(name='res5b_branch2a_relu_patch3', data=scale5b_branch2a_patch3 , act_type='relu')
res5b_branch2b_patch3 = mx.symbol.Convolution(name='res5b_branch2b_patch3', data=res5b_branch2a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2b_patch3 = mx.symbol.BatchNorm(name='bn5b_branch2b_patch3', data=res5b_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2b_patch3 = bn5b_branch2b_patch3
res5b_patch3 = mx.symbol.broadcast_plus(name='res5b_patch3', *[res5a_relu_patch3,scale5b_branch2b_patch3] )
res5b_relu_patch3 = mx.symbol.Activation(name='res5b_relu_patch3', data=res5b_patch3 , act_type='relu')
res5c_branch2a_patch3 = mx.symbol.Convolution(name='res5c_branch2a_patch3', data=res5b_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2a_patch3 = mx.symbol.BatchNorm(name='bn5c_branch2a_patch3', data=res5c_branch2a_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2a_patch3 = bn5c_branch2a_patch3
res5c_branch2a_relu_patch3 = mx.symbol.Activation(name='res5c_branch2a_relu_patch3', data=scale5c_branch2a_patch3 , act_type='relu')
res5c_branch2b_patch3 = mx.symbol.Convolution(name='res5c_branch2b_patch3', data=res5c_branch2a_relu_patch3 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2b_patch3 = mx.symbol.BatchNorm(name='bn5c_branch2b_patch3', data=res5c_branch2b_patch3 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2b_patch3 = bn5c_branch2b_patch3
res5c_patch3 = mx.symbol.broadcast_plus(name='res5c_patch3', *[res5b_relu_patch3,scale5c_branch2b_patch3] )
res5c_relu_patch3 = mx.symbol.Activation(name='res5c_relu_patch3', data=res5c_patch3 , act_type='relu')
pool5_patch3 = mx.symbol.Pooling(name='pool5_patch3', data=res5c_relu_patch3 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch6 = mx.symbol.Convolution(name='conv1_patch6', data=data_patch6 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
bn_conv1_patch6 = mx.symbol.BatchNorm(name='bn_conv1_patch6', data=conv1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale_conv1_patch6 = bn_conv1_patch6
conv1_relu_patch6 = mx.symbol.Activation(name='conv1_relu_patch6', data=scale_conv1_patch6 , act_type='relu')
pool1_patch6 = mx.symbol.Pooling(name='pool1_patch6', data=conv1_relu_patch6 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch6 = mx.symbol.Convolution(name='res2a_branch1_patch6', data=pool1_patch6 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1_patch6 = mx.symbol.BatchNorm(name='bn2a_branch1_patch6', data=res2a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch1_patch6 = bn2a_branch1_patch6
res2a_branch2a_patch6 = mx.symbol.Convolution(name='res2a_branch2a_patch6', data=pool1_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2a_patch6 = mx.symbol.BatchNorm(name='bn2a_branch2a_patch6', data=res2a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2a_patch6 = bn2a_branch2a_patch6
res2a_branch2a_relu_patch6 = mx.symbol.Activation(name='res2a_branch2a_relu_patch6', data=scale2a_branch2a_patch6 , act_type='relu')
res2a_branch2b_patch6 = mx.symbol.Convolution(name='res2a_branch2b_patch6', data=res2a_branch2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b_patch6 = mx.symbol.BatchNorm(name='bn2a_branch2b_patch6', data=res2a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2b_patch6 = bn2a_branch2b_patch6
res2a_patch6 = mx.symbol.broadcast_plus(name='res2a_patch6', *[scale2a_branch1_patch6,scale2a_branch2b_patch6] )
res2a_relu_patch6 = mx.symbol.Activation(name='res2a_relu_patch6', data=res2a_patch6 , act_type='relu')
res2b_branch2a_patch6 = mx.symbol.Convolution(name='res2b_branch2a_patch6', data=res2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2a_patch6 = mx.symbol.BatchNorm(name='bn2b_branch2a_patch6', data=res2b_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2a_patch6 = bn2b_branch2a_patch6
res2b_branch2a_relu_patch6 = mx.symbol.Activation(name='res2b_branch2a_relu_patch6', data=scale2b_branch2a_patch6 , act_type='relu')
res2b_branch2b_patch6 = mx.symbol.Convolution(name='res2b_branch2b_patch6', data=res2b_branch2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b_patch6 = mx.symbol.BatchNorm(name='bn2b_branch2b_patch6', data=res2b_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2b_patch6 = bn2b_branch2b_patch6
res2b_patch6 = mx.symbol.broadcast_plus(name='res2b_patch6', *[res2a_relu_patch6,scale2b_branch2b_patch6] )
res2b_relu_patch6 = mx.symbol.Activation(name='res2b_relu_patch6', data=res2b_patch6 , act_type='relu')
res2c_branch2a_patch6 = mx.symbol.Convolution(name='res2c_branch2a_patch6', data=res2b_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2a_patch6 = mx.symbol.BatchNorm(name='bn2c_branch2a_patch6', data=res2c_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2a_patch6 = bn2c_branch2a_patch6
res2c_branch2a_relu_patch6 = mx.symbol.Activation(name='res2c_branch2a_relu_patch6', data=scale2c_branch2a_patch6 , act_type='relu')
res2c_branch2b_patch6 = mx.symbol.Convolution(name='res2c_branch2b_patch6', data=res2c_branch2a_relu_patch6 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b_patch6 = mx.symbol.BatchNorm(name='bn2c_branch2b_patch6', data=res2c_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2b_patch6 = bn2c_branch2b_patch6
res2c_patch6 = mx.symbol.broadcast_plus(name='res2c_patch6', *[res2b_relu_patch6,scale2c_branch2b_patch6] )
res2c_relu_patch6 = mx.symbol.Activation(name='res2c_relu_patch6', data=res2c_patch6 , act_type='relu')
res3a_branch1_patch6 = mx.symbol.Convolution(name='res3a_branch1_patch6', data=res2c_relu_patch6 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1_patch6 = mx.symbol.BatchNorm(name='bn3a_branch1_patch6', data=res3a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch1_patch6 = bn3a_branch1_patch6
res3a_branch2a_patch6 = mx.symbol.Convolution(name='res3a_branch2a_patch6', data=res2c_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn3a_branch2a_patch6 = mx.symbol.BatchNorm(name='bn3a_branch2a_patch6', data=res3a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2a_patch6 = bn3a_branch2a_patch6
res3a_branch2a_relu_patch6 = mx.symbol.Activation(name='res3a_branch2a_relu_patch6', data=scale3a_branch2a_patch6 , act_type='relu')
res3a_branch2b_patch6 = mx.symbol.Convolution(name='res3a_branch2b_patch6', data=res3a_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b_patch6 = mx.symbol.BatchNorm(name='bn3a_branch2b_patch6', data=res3a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2b_patch6 = bn3a_branch2b_patch6
res3a_patch6 = mx.symbol.broadcast_plus(name='res3a_patch6', *[scale3a_branch1_patch6,scale3a_branch2b_patch6] )
res3a_relu_patch6 = mx.symbol.Activation(name='res3a_relu_patch6', data=res3a_patch6 , act_type='relu')
res3b_branch2a_patch6 = mx.symbol.Convolution(name='res3b_branch2a_patch6', data=res3a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2a_patch6 = mx.symbol.BatchNorm(name='bn3b_branch2a_patch6', data=res3b_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2a_patch6 = bn3b_branch2a_patch6
res3b_branch2a_relu_patch6 = mx.symbol.Activation(name='res3b_branch2a_relu_patch6', data=scale3b_branch2a_patch6 , act_type='relu')
res3b_branch2b_patch6 = mx.symbol.Convolution(name='res3b_branch2b_patch6', data=res3b_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2b_patch6 = mx.symbol.BatchNorm(name='bn3b_branch2b_patch6', data=res3b_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2b_patch6 = bn3b_branch2b_patch6
res3b_patch6 = mx.symbol.broadcast_plus(name='res3b_patch6', *[res3a_relu_patch6,scale3b_branch2b_patch6] )
res3b_relu_patch6 = mx.symbol.Activation(name='res3b_relu_patch6', data=res3b_patch6 , act_type='relu')
res3c_branch2a_patch6 = mx.symbol.Convolution(name='res3c_branch2a_patch6', data=res3b_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2a_patch6 = mx.symbol.BatchNorm(name='bn3c_branch2a_patch6', data=res3c_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2a_patch6 = bn3c_branch2a_patch6
res3c_branch2a_relu_patch6 = mx.symbol.Activation(name='res3c_branch2a_relu_patch6', data=scale3c_branch2a_patch6 , act_type='relu')
res3c_branch2b_patch6 = mx.symbol.Convolution(name='res3c_branch2b_patch6', data=res3c_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2b_patch6 = mx.symbol.BatchNorm(name='bn3c_branch2b_patch6', data=res3c_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2b_patch6 = bn3c_branch2b_patch6
res3c_patch6 = mx.symbol.broadcast_plus(name='res3c_patch6', *[res3b_relu_patch6,scale3c_branch2b_patch6] )
res3c_relu_patch6 = mx.symbol.Activation(name='res3c_relu_patch6', data=res3c_patch6 , act_type='relu')
res3d_branch2a_patch6 = mx.symbol.Convolution(name='res3d_branch2a_patch6', data=res3c_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2a_patch6 = mx.symbol.BatchNorm(name='bn3d_branch2a_patch6', data=res3d_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2a_patch6 = bn3d_branch2a_patch6
res3d_branch2a_relu_patch6 = mx.symbol.Activation(name='res3d_branch2a_relu_patch6', data=scale3d_branch2a_patch6 , act_type='relu')
res3d_branch2b_patch6 = mx.symbol.Convolution(name='res3d_branch2b_patch6', data=res3d_branch2a_relu_patch6 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2b_patch6 = mx.symbol.BatchNorm(name='bn3d_branch2b_patch6', data=res3d_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2b_patch6 = bn3d_branch2b_patch6
res3d_patch6 = mx.symbol.broadcast_plus(name='res3d_patch6', *[res3c_relu_patch6,scale3d_branch2b_patch6] )
res3d_relu_patch6 = mx.symbol.Activation(name='res3d_relu_patch6', data=res3d_patch6 , act_type='relu')
res4a_branch1_patch6 = mx.symbol.Convolution(name='res4a_branch1_patch6', data=res3d_relu_patch6 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1_patch6 = mx.symbol.BatchNorm(name='bn4a_branch1_patch6', data=res4a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch1_patch6 = bn4a_branch1_patch6
res4a_branch2a_patch6 = mx.symbol.Convolution(name='res4a_branch2a_patch6', data=res3d_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn4a_branch2a_patch6 = mx.symbol.BatchNorm(name='bn4a_branch2a_patch6', data=res4a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2a_patch6 = bn4a_branch2a_patch6
res4a_branch2a_relu_patch6 = mx.symbol.Activation(name='res4a_branch2a_relu_patch6', data=scale4a_branch2a_patch6 , act_type='relu')
res4a_branch2b_patch6 = mx.symbol.Convolution(name='res4a_branch2b_patch6', data=res4a_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b_patch6 = mx.symbol.BatchNorm(name='bn4a_branch2b_patch6', data=res4a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2b_patch6 = bn4a_branch2b_patch6
res4a_patch6 = mx.symbol.broadcast_plus(name='res4a_patch6', *[scale4a_branch1_patch6,scale4a_branch2b_patch6] )
res4a_relu_patch6 = mx.symbol.Activation(name='res4a_relu_patch6', data=res4a_patch6 , act_type='relu')
res4b_branch2a_patch6 = mx.symbol.Convolution(name='res4b_branch2a_patch6', data=res4a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2a_patch6 = mx.symbol.BatchNorm(name='bn4b_branch2a_patch6', data=res4b_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2a_patch6 = bn4b_branch2a_patch6
res4b_branch2a_relu_patch6 = mx.symbol.Activation(name='res4b_branch2a_relu_patch6', data=scale4b_branch2a_patch6 , act_type='relu')
res4b_branch2b_patch6 = mx.symbol.Convolution(name='res4b_branch2b_patch6', data=res4b_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2b_patch6 = mx.symbol.BatchNorm(name='bn4b_branch2b_patch6', data=res4b_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2b_patch6 = bn4b_branch2b_patch6
res4b_patch6 = mx.symbol.broadcast_plus(name='res4b_patch6', *[res4a_relu_patch6,scale4b_branch2b_patch6] )
res4b_relu_patch6 = mx.symbol.Activation(name='res4b_relu_patch6', data=res4b_patch6 , act_type='relu')
res4c_branch2a_patch6 = mx.symbol.Convolution(name='res4c_branch2a_patch6', data=res4b_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2a_patch6 = mx.symbol.BatchNorm(name='bn4c_branch2a_patch6', data=res4c_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2a_patch6 = bn4c_branch2a_patch6
res4c_branch2a_relu_patch6 = mx.symbol.Activation(name='res4c_branch2a_relu_patch6', data=scale4c_branch2a_patch6 , act_type='relu')
res4c_branch2b_patch6 = mx.symbol.Convolution(name='res4c_branch2b_patch6', data=res4c_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2b_patch6 = mx.symbol.BatchNorm(name='bn4c_branch2b_patch6', data=res4c_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2b_patch6 = bn4c_branch2b_patch6
res4c_patch6 = mx.symbol.broadcast_plus(name='res4c_patch6', *[res4b_relu_patch6,scale4c_branch2b_patch6] )
res4c_relu_patch6 = mx.symbol.Activation(name='res4c_relu_patch6', data=res4c_patch6 , act_type='relu')
res4d_branch2a_patch6 = mx.symbol.Convolution(name='res4d_branch2a_patch6', data=res4c_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2a_patch6 = mx.symbol.BatchNorm(name='bn4d_branch2a_patch6', data=res4d_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2a_patch6 = bn4d_branch2a_patch6
res4d_branch2a_relu_patch6 = mx.symbol.Activation(name='res4d_branch2a_relu_patch6', data=scale4d_branch2a_patch6 , act_type='relu')
res4d_branch2b_patch6 = mx.symbol.Convolution(name='res4d_branch2b_patch6', data=res4d_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2b_patch6 = mx.symbol.BatchNorm(name='bn4d_branch2b_patch6', data=res4d_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2b_patch6 = bn4d_branch2b_patch6
res4d_patch6 = mx.symbol.broadcast_plus(name='res4d_patch6', *[res4c_relu_patch6,scale4d_branch2b_patch6] )
res4d_relu_patch6 = mx.symbol.Activation(name='res4d_relu_patch6', data=res4d_patch6 , act_type='relu')
res4e_branch2a_patch6 = mx.symbol.Convolution(name='res4e_branch2a_patch6', data=res4d_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2a_patch6 = mx.symbol.BatchNorm(name='bn4e_branch2a_patch6', data=res4e_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2a_patch6 = bn4e_branch2a_patch6
res4e_branch2a_relu_patch6 = mx.symbol.Activation(name='res4e_branch2a_relu_patch6', data=scale4e_branch2a_patch6 , act_type='relu')
res4e_branch2b_patch6 = mx.symbol.Convolution(name='res4e_branch2b_patch6', data=res4e_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2b_patch6 = mx.symbol.BatchNorm(name='bn4e_branch2b_patch6', data=res4e_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2b_patch6 = bn4e_branch2b_patch6
res4e_patch6 = mx.symbol.broadcast_plus(name='res4e_patch6', *[res4d_relu_patch6,scale4e_branch2b_patch6] )
res4e_relu_patch6 = mx.symbol.Activation(name='res4e_relu_patch6', data=res4e_patch6 , act_type='relu')
res4f_branch2a_patch6 = mx.symbol.Convolution(name='res4f_branch2a_patch6', data=res4e_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2a_patch6 = mx.symbol.BatchNorm(name='bn4f_branch2a_patch6', data=res4f_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2a_patch6 = bn4f_branch2a_patch6
res4f_branch2a_relu_patch6 = mx.symbol.Activation(name='res4f_branch2a_relu_patch6', data=scale4f_branch2a_patch6 , act_type='relu')
res4f_branch2b_patch6 = mx.symbol.Convolution(name='res4f_branch2b_patch6', data=res4f_branch2a_relu_patch6 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2b_patch6 = mx.symbol.BatchNorm(name='bn4f_branch2b_patch6', data=res4f_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2b_patch6 = bn4f_branch2b_patch6
res4f_patch6 = mx.symbol.broadcast_plus(name='res4f_patch6', *[res4e_relu_patch6,scale4f_branch2b_patch6] )
res4f_relu_patch6 = mx.symbol.Activation(name='res4f_relu_patch6', data=res4f_patch6 , act_type='relu')
res5a_branch1_patch6 = mx.symbol.Convolution(name='res5a_branch1_patch6', data=res4f_relu_patch6 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn5a_branch1_patch6 = mx.symbol.BatchNorm(name='bn5a_branch1_patch6', data=res5a_branch1_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch1_patch6 = bn5a_branch1_patch6
res5a_branch2a_patch6 = mx.symbol.Convolution(name='res5a_branch2a_patch6', data=res4f_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn5a_branch2a_patch6 = mx.symbol.BatchNorm(name='bn5a_branch2a_patch6', data=res5a_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2a_patch6 = bn5a_branch2a_patch6
res5a_branch2a_relu_patch6 = mx.symbol.Activation(name='res5a_branch2a_relu_patch6', data=scale5a_branch2a_patch6 , act_type='relu')
res5a_branch2b_patch6 = mx.symbol.Convolution(name='res5a_branch2b_patch6', data=res5a_branch2a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5a_branch2b_patch6 = mx.symbol.BatchNorm(name='bn5a_branch2b_patch6', data=res5a_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2b_patch6 = bn5a_branch2b_patch6
res5a_patch6 = mx.symbol.broadcast_plus(name='res5a_patch6', *[scale5a_branch1_patch6,scale5a_branch2b_patch6] )
res5a_relu_patch6 = mx.symbol.Activation(name='res5a_relu_patch6', data=res5a_patch6 , act_type='relu')
res5b_branch2a_patch6 = mx.symbol.Convolution(name='res5b_branch2a_patch6', data=res5a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2a_patch6 = mx.symbol.BatchNorm(name='bn5b_branch2a_patch6', data=res5b_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2a_patch6 = bn5b_branch2a_patch6
res5b_branch2a_relu_patch6 = mx.symbol.Activation(name='res5b_branch2a_relu_patch6', data=scale5b_branch2a_patch6 , act_type='relu')
res5b_branch2b_patch6 = mx.symbol.Convolution(name='res5b_branch2b_patch6', data=res5b_branch2a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2b_patch6 = mx.symbol.BatchNorm(name='bn5b_branch2b_patch6', data=res5b_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2b_patch6 = bn5b_branch2b_patch6
res5b_patch6 = mx.symbol.broadcast_plus(name='res5b_patch6', *[res5a_relu_patch6,scale5b_branch2b_patch6] )
res5b_relu_patch6 = mx.symbol.Activation(name='res5b_relu_patch6', data=res5b_patch6 , act_type='relu')
res5c_branch2a_patch6 = mx.symbol.Convolution(name='res5c_branch2a_patch6', data=res5b_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2a_patch6 = mx.symbol.BatchNorm(name='bn5c_branch2a_patch6', data=res5c_branch2a_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2a_patch6 = bn5c_branch2a_patch6
res5c_branch2a_relu_patch6 = mx.symbol.Activation(name='res5c_branch2a_relu_patch6', data=scale5c_branch2a_patch6 , act_type='relu')
res5c_branch2b_patch6 = mx.symbol.Convolution(name='res5c_branch2b_patch6', data=res5c_branch2a_relu_patch6 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2b_patch6 = mx.symbol.BatchNorm(name='bn5c_branch2b_patch6', data=res5c_branch2b_patch6 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2b_patch6 = bn5c_branch2b_patch6
res5c_patch6 = mx.symbol.broadcast_plus(name='res5c_patch6', *[res5b_relu_patch6,scale5c_branch2b_patch6] )
res5c_relu_patch6 = mx.symbol.Activation(name='res5c_relu_patch6', data=res5c_patch6 , act_type='relu')
pool5_patch6 = mx.symbol.Pooling(name='pool5_patch6', data=res5c_relu_patch6 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
conv1_patch7 = mx.symbol.Convolution(name='conv1_patch7', data=data_patch7 , num_filter=64, pad=(3, 3), kernel=(7,7), stride=(2,2), no_bias=True)
bn_conv1_patch7 = mx.symbol.BatchNorm(name='bn_conv1_patch7', data=conv1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale_conv1_patch7 = bn_conv1_patch7
conv1_relu_patch7 = mx.symbol.Activation(name='conv1_relu_patch7', data=scale_conv1_patch7 , act_type='relu')
pool1_patch7 = mx.symbol.Pooling(name='pool1_patch7', data=conv1_relu_patch7 , pooling_convention='full', pad=(0,0), kernel=(3,3), stride=(2,2), pool_type='max')
res2a_branch1_patch7 = mx.symbol.Convolution(name='res2a_branch1_patch7', data=pool1_patch7 , num_filter=64, pad=(0, 0), kernel=(1,1), stride=(1,1), no_bias=True)
bn2a_branch1_patch7 = mx.symbol.BatchNorm(name='bn2a_branch1_patch7', data=res2a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch1_patch7 = bn2a_branch1_patch7
res2a_branch2a_patch7 = mx.symbol.Convolution(name='res2a_branch2a_patch7', data=pool1_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2a_patch7 = mx.symbol.BatchNorm(name='bn2a_branch2a_patch7', data=res2a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2a_patch7 = bn2a_branch2a_patch7
res2a_branch2a_relu_patch7 = mx.symbol.Activation(name='res2a_branch2a_relu_patch7', data=scale2a_branch2a_patch7 , act_type='relu')
res2a_branch2b_patch7 = mx.symbol.Convolution(name='res2a_branch2b_patch7', data=res2a_branch2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2a_branch2b_patch7 = mx.symbol.BatchNorm(name='bn2a_branch2b_patch7', data=res2a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2a_branch2b_patch7 = bn2a_branch2b_patch7
res2a_patch7 = mx.symbol.broadcast_plus(name='res2a_patch7', *[scale2a_branch1_patch7,scale2a_branch2b_patch7] )
res2a_relu_patch7 = mx.symbol.Activation(name='res2a_relu_patch7', data=res2a_patch7 , act_type='relu')
res2b_branch2a_patch7 = mx.symbol.Convolution(name='res2b_branch2a_patch7', data=res2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2a_patch7 = mx.symbol.BatchNorm(name='bn2b_branch2a_patch7', data=res2b_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2a_patch7 = bn2b_branch2a_patch7
res2b_branch2a_relu_patch7 = mx.symbol.Activation(name='res2b_branch2a_relu_patch7', data=scale2b_branch2a_patch7 , act_type='relu')
res2b_branch2b_patch7 = mx.symbol.Convolution(name='res2b_branch2b_patch7', data=res2b_branch2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2b_branch2b_patch7 = mx.symbol.BatchNorm(name='bn2b_branch2b_patch7', data=res2b_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2b_branch2b_patch7 = bn2b_branch2b_patch7
res2b_patch7 = mx.symbol.broadcast_plus(name='res2b_patch7', *[res2a_relu_patch7,scale2b_branch2b_patch7] )
res2b_relu_patch7 = mx.symbol.Activation(name='res2b_relu_patch7', data=res2b_patch7 , act_type='relu')
res2c_branch2a_patch7 = mx.symbol.Convolution(name='res2c_branch2a_patch7', data=res2b_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2a_patch7 = mx.symbol.BatchNorm(name='bn2c_branch2a_patch7', data=res2c_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2a_patch7 = bn2c_branch2a_patch7
res2c_branch2a_relu_patch7 = mx.symbol.Activation(name='res2c_branch2a_relu_patch7', data=scale2c_branch2a_patch7 , act_type='relu')
res2c_branch2b_patch7 = mx.symbol.Convolution(name='res2c_branch2b_patch7', data=res2c_branch2a_relu_patch7 , num_filter=64, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn2c_branch2b_patch7 = mx.symbol.BatchNorm(name='bn2c_branch2b_patch7', data=res2c_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale2c_branch2b_patch7 = bn2c_branch2b_patch7
res2c_patch7 = mx.symbol.broadcast_plus(name='res2c_patch7', *[res2b_relu_patch7,scale2c_branch2b_patch7] )
res2c_relu_patch7 = mx.symbol.Activation(name='res2c_relu_patch7', data=res2c_patch7 , act_type='relu')
res3a_branch1_patch7 = mx.symbol.Convolution(name='res3a_branch1_patch7', data=res2c_relu_patch7 , num_filter=128, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn3a_branch1_patch7 = mx.symbol.BatchNorm(name='bn3a_branch1_patch7', data=res3a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch1_patch7 = bn3a_branch1_patch7
res3a_branch2a_patch7 = mx.symbol.Convolution(name='res3a_branch2a_patch7', data=res2c_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn3a_branch2a_patch7 = mx.symbol.BatchNorm(name='bn3a_branch2a_patch7', data=res3a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2a_patch7 = bn3a_branch2a_patch7
res3a_branch2a_relu_patch7 = mx.symbol.Activation(name='res3a_branch2a_relu_patch7', data=scale3a_branch2a_patch7 , act_type='relu')
res3a_branch2b_patch7 = mx.symbol.Convolution(name='res3a_branch2b_patch7', data=res3a_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3a_branch2b_patch7 = mx.symbol.BatchNorm(name='bn3a_branch2b_patch7', data=res3a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3a_branch2b_patch7 = bn3a_branch2b_patch7
res3a_patch7 = mx.symbol.broadcast_plus(name='res3a_patch7', *[scale3a_branch1_patch7,scale3a_branch2b_patch7] )
res3a_relu_patch7 = mx.symbol.Activation(name='res3a_relu_patch7', data=res3a_patch7 , act_type='relu')
res3b_branch2a_patch7 = mx.symbol.Convolution(name='res3b_branch2a_patch7', data=res3a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2a_patch7 = mx.symbol.BatchNorm(name='bn3b_branch2a_patch7', data=res3b_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2a_patch7 = bn3b_branch2a_patch7
res3b_branch2a_relu_patch7 = mx.symbol.Activation(name='res3b_branch2a_relu_patch7', data=scale3b_branch2a_patch7 , act_type='relu')
res3b_branch2b_patch7 = mx.symbol.Convolution(name='res3b_branch2b_patch7', data=res3b_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3b_branch2b_patch7 = mx.symbol.BatchNorm(name='bn3b_branch2b_patch7', data=res3b_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3b_branch2b_patch7 = bn3b_branch2b_patch7
res3b_patch7 = mx.symbol.broadcast_plus(name='res3b_patch7', *[res3a_relu_patch7,scale3b_branch2b_patch7] )
res3b_relu_patch7 = mx.symbol.Activation(name='res3b_relu_patch7', data=res3b_patch7 , act_type='relu')
res3c_branch2a_patch7 = mx.symbol.Convolution(name='res3c_branch2a_patch7', data=res3b_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2a_patch7 = mx.symbol.BatchNorm(name='bn3c_branch2a_patch7', data=res3c_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2a_patch7 = bn3c_branch2a_patch7
res3c_branch2a_relu_patch7 = mx.symbol.Activation(name='res3c_branch2a_relu_patch7', data=scale3c_branch2a_patch7 , act_type='relu')
res3c_branch2b_patch7 = mx.symbol.Convolution(name='res3c_branch2b_patch7', data=res3c_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3c_branch2b_patch7 = mx.symbol.BatchNorm(name='bn3c_branch2b_patch7', data=res3c_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3c_branch2b_patch7 = bn3c_branch2b_patch7
res3c_patch7 = mx.symbol.broadcast_plus(name='res3c_patch7', *[res3b_relu_patch7,scale3c_branch2b_patch7] )
res3c_relu_patch7 = mx.symbol.Activation(name='res3c_relu_patch7', data=res3c_patch7 , act_type='relu')
res3d_branch2a_patch7 = mx.symbol.Convolution(name='res3d_branch2a_patch7', data=res3c_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2a_patch7 = mx.symbol.BatchNorm(name='bn3d_branch2a_patch7', data=res3d_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2a_patch7 = bn3d_branch2a_patch7
res3d_branch2a_relu_patch7 = mx.symbol.Activation(name='res3d_branch2a_relu_patch7', data=scale3d_branch2a_patch7 , act_type='relu')
res3d_branch2b_patch7 = mx.symbol.Convolution(name='res3d_branch2b_patch7', data=res3d_branch2a_relu_patch7 , num_filter=128, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn3d_branch2b_patch7 = mx.symbol.BatchNorm(name='bn3d_branch2b_patch7', data=res3d_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale3d_branch2b_patch7 = bn3d_branch2b_patch7
res3d_patch7 = mx.symbol.broadcast_plus(name='res3d_patch7', *[res3c_relu_patch7,scale3d_branch2b_patch7] )
res3d_relu_patch7 = mx.symbol.Activation(name='res3d_relu_patch7', data=res3d_patch7 , act_type='relu')
res4a_branch1_patch7 = mx.symbol.Convolution(name='res4a_branch1_patch7', data=res3d_relu_patch7 , num_filter=256, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn4a_branch1_patch7 = mx.symbol.BatchNorm(name='bn4a_branch1_patch7', data=res4a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch1_patch7 = bn4a_branch1_patch7
res4a_branch2a_patch7 = mx.symbol.Convolution(name='res4a_branch2a_patch7', data=res3d_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn4a_branch2a_patch7 = mx.symbol.BatchNorm(name='bn4a_branch2a_patch7', data=res4a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2a_patch7 = bn4a_branch2a_patch7
res4a_branch2a_relu_patch7 = mx.symbol.Activation(name='res4a_branch2a_relu_patch7', data=scale4a_branch2a_patch7 , act_type='relu')
res4a_branch2b_patch7 = mx.symbol.Convolution(name='res4a_branch2b_patch7', data=res4a_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4a_branch2b_patch7 = mx.symbol.BatchNorm(name='bn4a_branch2b_patch7', data=res4a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4a_branch2b_patch7 = bn4a_branch2b_patch7
res4a_patch7 = mx.symbol.broadcast_plus(name='res4a_patch7', *[scale4a_branch1_patch7,scale4a_branch2b_patch7] )
res4a_relu_patch7 = mx.symbol.Activation(name='res4a_relu_patch7', data=res4a_patch7 , act_type='relu')
res4b_branch2a_patch7 = mx.symbol.Convolution(name='res4b_branch2a_patch7', data=res4a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2a_patch7 = mx.symbol.BatchNorm(name='bn4b_branch2a_patch7', data=res4b_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2a_patch7 = bn4b_branch2a_patch7
res4b_branch2a_relu_patch7 = mx.symbol.Activation(name='res4b_branch2a_relu_patch7', data=scale4b_branch2a_patch7 , act_type='relu')
res4b_branch2b_patch7 = mx.symbol.Convolution(name='res4b_branch2b_patch7', data=res4b_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4b_branch2b_patch7 = mx.symbol.BatchNorm(name='bn4b_branch2b_patch7', data=res4b_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4b_branch2b_patch7 = bn4b_branch2b_patch7
res4b_patch7 = mx.symbol.broadcast_plus(name='res4b_patch7', *[res4a_relu_patch7,scale4b_branch2b_patch7] )
res4b_relu_patch7 = mx.symbol.Activation(name='res4b_relu_patch7', data=res4b_patch7 , act_type='relu')
res4c_branch2a_patch7 = mx.symbol.Convolution(name='res4c_branch2a_patch7', data=res4b_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2a_patch7 = mx.symbol.BatchNorm(name='bn4c_branch2a_patch7', data=res4c_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2a_patch7 = bn4c_branch2a_patch7
res4c_branch2a_relu_patch7 = mx.symbol.Activation(name='res4c_branch2a_relu_patch7', data=scale4c_branch2a_patch7 , act_type='relu')
res4c_branch2b_patch7 = mx.symbol.Convolution(name='res4c_branch2b_patch7', data=res4c_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4c_branch2b_patch7 = mx.symbol.BatchNorm(name='bn4c_branch2b_patch7', data=res4c_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4c_branch2b_patch7 = bn4c_branch2b_patch7
res4c_patch7 = mx.symbol.broadcast_plus(name='res4c_patch7', *[res4b_relu_patch7,scale4c_branch2b_patch7] )
res4c_relu_patch7 = mx.symbol.Activation(name='res4c_relu_patch7', data=res4c_patch7 , act_type='relu')
res4d_branch2a_patch7 = mx.symbol.Convolution(name='res4d_branch2a_patch7', data=res4c_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2a_patch7 = mx.symbol.BatchNorm(name='bn4d_branch2a_patch7', data=res4d_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2a_patch7 = bn4d_branch2a_patch7
res4d_branch2a_relu_patch7 = mx.symbol.Activation(name='res4d_branch2a_relu_patch7', data=scale4d_branch2a_patch7 , act_type='relu')
res4d_branch2b_patch7 = mx.symbol.Convolution(name='res4d_branch2b_patch7', data=res4d_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4d_branch2b_patch7 = mx.symbol.BatchNorm(name='bn4d_branch2b_patch7', data=res4d_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4d_branch2b_patch7 = bn4d_branch2b_patch7
res4d_patch7 = mx.symbol.broadcast_plus(name='res4d_patch7', *[res4c_relu_patch7,scale4d_branch2b_patch7] )
res4d_relu_patch7 = mx.symbol.Activation(name='res4d_relu_patch7', data=res4d_patch7 , act_type='relu')
res4e_branch2a_patch7 = mx.symbol.Convolution(name='res4e_branch2a_patch7', data=res4d_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2a_patch7 = mx.symbol.BatchNorm(name='bn4e_branch2a_patch7', data=res4e_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2a_patch7 = bn4e_branch2a_patch7
res4e_branch2a_relu_patch7 = mx.symbol.Activation(name='res4e_branch2a_relu_patch7', data=scale4e_branch2a_patch7 , act_type='relu')
res4e_branch2b_patch7 = mx.symbol.Convolution(name='res4e_branch2b_patch7', data=res4e_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4e_branch2b_patch7 = mx.symbol.BatchNorm(name='bn4e_branch2b_patch7', data=res4e_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4e_branch2b_patch7 = bn4e_branch2b_patch7
res4e_patch7 = mx.symbol.broadcast_plus(name='res4e_patch7', *[res4d_relu_patch7,scale4e_branch2b_patch7] )
res4e_relu_patch7 = mx.symbol.Activation(name='res4e_relu_patch7', data=res4e_patch7 , act_type='relu')
res4f_branch2a_patch7 = mx.symbol.Convolution(name='res4f_branch2a_patch7', data=res4e_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2a_patch7 = mx.symbol.BatchNorm(name='bn4f_branch2a_patch7', data=res4f_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2a_patch7 = bn4f_branch2a_patch7
res4f_branch2a_relu_patch7 = mx.symbol.Activation(name='res4f_branch2a_relu_patch7', data=scale4f_branch2a_patch7 , act_type='relu')
res4f_branch2b_patch7 = mx.symbol.Convolution(name='res4f_branch2b_patch7', data=res4f_branch2a_relu_patch7 , num_filter=256, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn4f_branch2b_patch7 = mx.symbol.BatchNorm(name='bn4f_branch2b_patch7', data=res4f_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale4f_branch2b_patch7 = bn4f_branch2b_patch7
res4f_patch7 = mx.symbol.broadcast_plus(name='res4f_patch7', *[res4e_relu_patch7,scale4f_branch2b_patch7] )
res4f_relu_patch7 = mx.symbol.Activation(name='res4f_relu_patch7', data=res4f_patch7 , act_type='relu')
res5a_branch1_patch7 = mx.symbol.Convolution(name='res5a_branch1_patch7', data=res4f_relu_patch7 , num_filter=512, pad=(0, 0), kernel=(1,1), stride=(2,2), no_bias=True)
bn5a_branch1_patch7 = mx.symbol.BatchNorm(name='bn5a_branch1_patch7', data=res5a_branch1_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch1_patch7 = bn5a_branch1_patch7
res5a_branch2a_patch7 = mx.symbol.Convolution(name='res5a_branch2a_patch7', data=res4f_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(2,2), no_bias=True)
bn5a_branch2a_patch7 = mx.symbol.BatchNorm(name='bn5a_branch2a_patch7', data=res5a_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2a_patch7 = bn5a_branch2a_patch7
res5a_branch2a_relu_patch7 = mx.symbol.Activation(name='res5a_branch2a_relu_patch7', data=scale5a_branch2a_patch7 , act_type='relu')
res5a_branch2b_patch7 = mx.symbol.Convolution(name='res5a_branch2b_patch7', data=res5a_branch2a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5a_branch2b_patch7 = mx.symbol.BatchNorm(name='bn5a_branch2b_patch7', data=res5a_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5a_branch2b_patch7 = bn5a_branch2b_patch7
res5a_patch7 = mx.symbol.broadcast_plus(name='res5a_patch7', *[scale5a_branch1_patch7,scale5a_branch2b_patch7] )
res5a_relu_patch7 = mx.symbol.Activation(name='res5a_relu_patch7', data=res5a_patch7 , act_type='relu')
res5b_branch2a_patch7 = mx.symbol.Convolution(name='res5b_branch2a_patch7', data=res5a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2a_patch7 = mx.symbol.BatchNorm(name='bn5b_branch2a_patch7', data=res5b_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2a_patch7 = bn5b_branch2a_patch7
res5b_branch2a_relu_patch7 = mx.symbol.Activation(name='res5b_branch2a_relu_patch7', data=scale5b_branch2a_patch7 , act_type='relu')
res5b_branch2b_patch7 = mx.symbol.Convolution(name='res5b_branch2b_patch7', data=res5b_branch2a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5b_branch2b_patch7 = mx.symbol.BatchNorm(name='bn5b_branch2b_patch7', data=res5b_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5b_branch2b_patch7 = bn5b_branch2b_patch7
res5b_patch7 = mx.symbol.broadcast_plus(name='res5b_patch7', *[res5a_relu_patch7,scale5b_branch2b_patch7] )
res5b_relu_patch7 = mx.symbol.Activation(name='res5b_relu_patch7', data=res5b_patch7 , act_type='relu')
res5c_branch2a_patch7 = mx.symbol.Convolution(name='res5c_branch2a_patch7', data=res5b_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2a_patch7 = mx.symbol.BatchNorm(name='bn5c_branch2a_patch7', data=res5c_branch2a_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2a_patch7 = bn5c_branch2a_patch7
res5c_branch2a_relu_patch7 = mx.symbol.Activation(name='res5c_branch2a_relu_patch7', data=scale5c_branch2a_patch7 , act_type='relu')
res5c_branch2b_patch7 = mx.symbol.Convolution(name='res5c_branch2b_patch7', data=res5c_branch2a_relu_patch7 , num_filter=512, pad=(1, 1), kernel=(3,3), stride=(1,1), no_bias=True)
bn5c_branch2b_patch7 = mx.symbol.BatchNorm(name='bn5c_branch2b_patch7', data=res5c_branch2b_patch7 , use_global_stats=True, fix_gamma=False, eps=0.000100)
scale5c_branch2b_patch7 = bn5c_branch2b_patch7
res5c_patch7 = mx.symbol.broadcast_plus(name='res5c_patch7', *[res5b_relu_patch7,scale5c_branch2b_patch7] )
res5c_relu_patch7 = mx.symbol.Activation(name='res5c_relu_patch7', data=res5c_patch7 , act_type='relu')
pool5_patch7 = mx.symbol.Pooling(name='pool5_patch7', data=res5c_relu_patch7 , pooling_convention='full', pad=(0,0), kernel=(7,7), stride=(1,1), pool_type='avg')
concat_0 = mx.symbol.Concat(name='concat_0', *[pool5_patch0,pool5_patch1,pool5_patch2,pool5_patch3,pool5_patch6,pool5_patch7] )
