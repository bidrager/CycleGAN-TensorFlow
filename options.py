import  tensorflow as tf

def _inti_w(shape,name,wd=1e-4,stddev=0.01):

    initializer = tf.truncated_normal_initializer(stddev=stddev)
    w = tf.get_variable(shape=shape,initializer=initializer,dtype=tf.float32,name=name)
    # collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    # weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='weight_loss')
    # tf.add_to_collection(collection_name,weight_decay)
    return w

def _init_b(shape,name, constant=0.0,db=1e-4):
    initializer = tf.constant_initializer(constant)
    b = tf.get_variable( shape=shape, initializer=initializer,name=name)
    # collection_name=tf.GraphKeys.REGULARIZATION_LOSSES
    # b_decay=tf.multiply(tf.nn.l2_loss(b),db,name='b_loss')
    # tf.add_to_collection(collection_name,b_decay)
    return b

def _conv2d(input,w_shape,b_shape,stride,name,name_w,name_b,padding='SAME'):
    with tf.name_scope(name):
        w=_inti_w(w_shape,name_w)
        b=_init_b(b_shape,name_b)
        conv=tf.nn.conv2d(input,w,stride,padding)
        conv=tf.nn.bias_add(conv,b)
    return conv

def _deconv2d(input,w_shape,b_shape,output_shape,stride,name,name_w,name_b,padding='SAME'):
    with tf.name_scope(name):
        w = _inti_w(w_shape,name_w)
        b = _init_b(b_shape,name_b)
        deconv=tf.nn.conv2d_transpose(input,w,output_shape,stride,padding)
        deconv=tf.nn.bias_add(deconv,b)
        deconv=tf.nn.relu(deconv)
    return deconv

def resnet_block(input,w_shape,b_shape,stride,name,name_w,name_b,padding='SAME'):
    with tf.name_scope(name):
        conv1=_conv2d(input,w_shape,b_shape,stride,'conv_1','conv_1_'+name_w,'conv_1_'+name_b,padding,)
        conv1=tf.nn.relu(conv1)
        conv2 = _conv2d(conv1, w_shape, b_shape, stride, 'conv_2','conv_2_'+name_w,'conv_2_'+name_b,padding)
        output=input+conv2
    return tf.nn.relu(output)

