import tensorflow as tf
import options

class Generator():
    def __init__(self,name):
        self.name=name
    def __call__(self, input):
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as scope:
            conv_1 = options._conv2d(input, [7, 7, 3, 64], [64], stride=[1, 1, 1, 1], name='conv_1',name_w='conv_1_w',name_b='conv_1_b')
            conv_1 = tf.nn.relu(conv_1)

            conv_2 = options._conv2d(conv_1, [3, 3, 64, 128], [128], stride=[1,2 ,2 ,1], name='conv_2',name_w='conv_2_w',name_b='conv_2_b')
            conv_2 = tf.nn.relu(conv_2)

            conv_3 = options._conv2d(conv_2, [3,3, 128, 256], [256], stride=[1, 2, 2, 1], name='conv_3',name_w='conv_3_w',name_b='conv_3_b')
            conv_3 = tf.nn.relu(conv_3)

            res_1 = options.resnet_block(conv_3,[3,3,256,256],[256],[1,1,1,1],'res_1',name_w='res_1_w',name_b='res_1_b')
            res_2 = options.resnet_block(res_1, [3, 3, 256, 256], [256], [1, 1, 1, 1], 'res_2',name_w='res_2_w',name_b='res_2_b')
            res_3 = options.resnet_block(res_2, [3, 3, 256, 256], [256], [1, 1, 1, 1], 'res_3',name_w='res_3_w',name_b='res_3_b')
            res_4 = options.resnet_block(res_3, [3, 3, 256, 256], [256], [1, 1, 1, 1], 'res_4',name_w='res_4_w',name_b='res_4_b')
            res_5 = options.resnet_block(res_4, [3, 3, 256, 256], [256], [1, 1, 1, 1], 'res_5',name_w='res_5_w',name_b='res_5_b')
            res_6 = options.resnet_block(res_5, [3, 3, 256, 256], [256], [1, 1, 1, 1], 'res_6',name_w='res_6_w',name_b='res_6_b')

            deconv_1=options._deconv2d(res_6,[3,3,128,256],[128],tf.shape(conv_2),[1,2,2,1],'deconv_1','deconv_1_w','deconv_1_b')
            deconv_2 = options._deconv2d(deconv_1, [3, 3, 64, 128], [64], tf.shape(conv_1),[1, 2, 2, 1],
                                         'deconv_2','deconv_2_w','deconv_2_b')

            output=options._conv2d(deconv_2,[7,7,64,3],[3],stride=[1,1,1,1],name='output',name_w='conv_out_w',name_b='conv_out_b')
            output=tf.nn.sigmoid(output)
        return output

class Descrimator():
    def __init__(self,name):
        self.name=name
    def __call__(self,input):
        with tf.variable_scope(self.name,reuse=tf.AUTO_REUSE) as scope:
            conv_1 = options._conv2d(input, [5, 5, 3, 64], [64], stride=[1, 1, 1, 1] ,name='conv_1',name_w='conv_1_w',name_b='conv_1_b')
            conv_1 = tf.nn.relu(conv_1)
            conv_1=tf.nn.max_pool(conv_1,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            conv_2 = options._conv2d(conv_1, [3, 3, 64, 128], [128], stride=[1, 2, 2, 1], name='conv_2',name_w='conv_2_w',name_b='conv_2_b')
            conv_2 = tf.nn.relu(conv_2)

            conv_3 = options._conv2d(conv_2, [3, 3, 128, 256], [256], stride=[1, 2, 2, 1], name='conv_3',name_w='conv_3_w',name_b='conv_3_b')
            conv_3 = tf.nn.relu(conv_3)
            conv_3= tf.nn.max_pool(conv_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

            conv_4 = options._conv2d(conv_3, [3, 3, 256, 256], [256], stride=[1, 2, 2, 1], name='conv_4',name_w='conv_4_w',name_b='conv_4_b')
            conv_4 = tf.nn.relu(conv_4)

            conv_5 = options._conv2d(conv_4, [3, 3, 256, 512], [512], stride=[1, 2, 2, 1], name='conv_5',name_w='conv_5_w',name_b='conv_5_b')
            conv_5 = tf.nn.relu(conv_5)
            conv_5 = tf.nn.max_pool(conv_5,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
            logit=options._conv2d(conv_5,[2,2,512,1],[1],stride=[1,1,1,1],name='logit',name_w='conv_logit_w',name_b='conv_logit_b')
            logit=tf.reshape(logit,[-1,1])
        return tf.nn.sigmoid(logit)




