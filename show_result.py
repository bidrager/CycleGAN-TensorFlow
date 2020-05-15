import numpy as np
import tensorflow as tf
from cycle_gan import Generator,Descrimator
import cv2



generator_a=Generator('generator_A')
generator_b=Generator('generator_B')

descrimator_a=Descrimator('descrimator_A')
descrimator_b=Descrimator('descrimator_B')

g_man=tf.placeholder(dtype=tf.float32,shape=[None,256,256,3])
g_woman=tf.placeholder(dtype=tf.float32,shape=[None,256,256,3])

d_man=tf.placeholder(dtype=tf.float32,shape=[None,256,256,3])
d_woman=tf.placeholder(dtype=tf.float32,shape=[None,256,256,3])

image_man=generator_a(g_man)
image_woman=generator_b(g_woman)


sess = tf.Session()
saver=tf.train.Saver(max_to_keep=2)
summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess.run(init)


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


checkpoint = tf.train.get_checkpoint_state('model')
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print('successed')
else:
    print('could not find old network weights')
man_image=['real_image/man/000064.jpg','real_image/man/000134.jpg']
woman_image=['real_image/woman/000024.jpg','real_image/woman/000085.jpg']
man_images=[]
for img in man_image:
    image=cv2.imread(img)
    image=image/255
    man_images.append(image)
man_images=np.array(man_images,dtype=np.float32)
woman_images=[]
for img in woman_image:
    image=cv2.imread(img)
    image=image/255
    woman_images.append(image)
woman_image=np.array(woman_images,dtype=np.float32)
image_w=sess.run(image_man,feed_dict={g_man:man_images})
cv2.imwrite('mantowoman/0.jpg',image_w[0]*255)
cv2.imwrite('mantowoman/1.jpg',image_w[1]*255)
image_w=sess.run(image_woman,feed_dict={g_woman:woman_image})
cv2.imwrite('womantoman/0.jpg',image_w[0]*255)
cv2.imwrite('womantoman/1.jpg',image_w[1]*255)



