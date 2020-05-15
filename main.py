import tensorflow as tf
import  read_tfrecord
from cycle_gan import Generator,Descrimator
import config
import cv2

log_dir='log'
man_train_tfrecord = 'data/man_train_tfrecord.tfrecords'
man_test_tfrecord = 'data/man_test_tfrecord.tfrecords'
woman_train_tfrecord = 'data/woman_train_tfrecord.tfrecords'
woman_test_tfrecord = 'data/woman_test_tfrecord.tfrecords'

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

image_man_loss=tf.reduce_mean(tf.abs(generator_b(generator_a(g_man))-g_man))
image_woman_loss=tf.reduce_mean(tf.abs(generator_a(generator_b(g_woman))-g_woman))
image_loss=image_man_loss+image_woman_loss

tf.summary.scalar( 'image_man_loss', image_man_loss)
tf.summary.scalar( 'iamge_woman_loss', image_woman_loss)

g_a_loss= 8*image_loss + tf.reduce_mean(tf.squared_difference(descrimator_b(generator_a(g_man)),1))\
          +tf.reduce_sum(tf.get_collection('generator_a_loss'))
g_b_loss=8*image_loss + tf.reduce_mean(tf.squared_difference(descrimator_a(generator_b(g_woman)),1))\
         +tf.reduce_sum(tf.get_collection('generator_b_loss'))
d_a_loss=tf.reduce_mean(tf.squared_difference(descrimator_a(d_man),1))+tf.reduce_mean(tf.squared_difference(descrimator_a(generator_b(g_woman)),0))\
         +tf.reduce_sum(tf.get_collection('descrimator_a_loss'))
d_b_loss=tf.reduce_mean(tf.squared_difference(descrimator_b(d_woman),1))+tf.reduce_mean(tf.squared_difference(descrimator_b(generator_a(g_man)),0))\
         +tf.reduce_sum(tf.get_collection('descrimator_b_loss'))


t_vars = tf.trainable_variables()
generator_a_var = [var for var in t_vars if "generator_A" in var.name]
for w in generator_a_var:
    weight_decay = tf.multiply(tf.nn.l2_loss(w), config.wd)
    tf.add_to_collection("generator_a_loss", weight_decay)

generator_b_var = [var for var in t_vars if "generator_B" in var.name]
for w in generator_b_var:
    weight_decay = tf.multiply(tf.nn.l2_loss(w), config.wd)
    tf.add_to_collection("generator_b_loss", weight_decay)

descrimator_a_var = [var for var in t_vars if "descrimator_A" in var.name]
for w in descrimator_a_var:
    weight_decay = tf.multiply(tf.nn.l2_loss(w), config.wd)
    tf.add_to_collection("descrimator_a_loss", weight_decay)

descrimator_b_var = [var for var in t_vars if "descrimator_B" in var.name]
for w in descrimator_b_var:
    weight_decay = tf.multiply(tf.nn.l2_loss(w), config.wd)
    tf.add_to_collection("descrimator_b_loss", weight_decay)

generator_a_loss=g_a_loss+tf.add_n(tf.get_collection("generator_a_loss"))
generator_b_loss=g_b_loss+tf.add_n(tf.get_collection("generator_b_loss"))
descrimator_a_loss=d_a_loss+tf.add_n(tf.get_collection("descrimator_a_loss"))
descrimator_b_loss=d_b_loss+tf.add_n(tf.get_collection("descrimator_b_loss"))

tf.summary.scalar( 'g_a_loss', g_a_loss)
tf.summary.scalar( 'g_b_loss', g_b_loss)
tf.summary.scalar( 'd_a_loss', d_a_loss)
tf.summary.scalar( 'd_b_loss', d_b_loss)

tf.summary.scalar( 'generator_a_loss', generator_a_loss)
tf.summary.scalar( 'generator_b_loss', generator_b_loss)
tf.summary.scalar( 'descrimator_a_loss', descrimator_a_loss)
tf.summary.scalar( 'descrimator_b_loss', descrimator_b_loss)

optimizor_ga=tf.train.AdamOptimizer(learning_rate=config.learning_rate)
optimizor_gb=tf.train.AdamOptimizer(learning_rate=config.learning_rate)
optimizor_da=tf.train.AdamOptimizer(learning_rate=config.d_learning_rate)
optimizor_db=tf.train.AdamOptimizer(learning_rate=config.d_learning_rate)
train_ga=optimizor_ga.minimize(generator_a_loss,var_list=generator_a_var)
train_gb=optimizor_gb.minimize(generator_b_loss,var_list=generator_b_var)
train_da=optimizor_da.minimize(descrimator_a_loss,var_list=descrimator_a_var)
train_db=optimizor_db.minimize(descrimator_b_loss,var_list=descrimator_b_var)

sess = tf.Session()
saver=tf.train.Saver(max_to_keep=2)
writer = tf.summary.FileWriter(log_dir,sess.graph)
summary_op = tf.summary.merge_all()
init = tf.global_variables_initializer()
sess.run(init)
g_man_image=read_tfrecord.read_batch_image(man_train_tfrecord,batch_size=config.batch_size)
g_woman_image=read_tfrecord.read_batch_image(woman_train_tfrecord,batch_size=config.batch_size)
d_man_image=read_tfrecord.read_batch_image(man_train_tfrecord,batch_size=config.batch_size)
d_woman_image=read_tfrecord.read_batch_image(woman_train_tfrecord,batch_size=config.batch_size)

g_man_image_test=read_tfrecord.read_batch_image(man_test_tfrecord,batch_size=1)
g_woman_image_test=read_tfrecord.read_batch_image(woman_test_tfrecord,batch_size=1)


coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)


checkpoint = tf.train.get_checkpoint_state('model')
if checkpoint and checkpoint.model_checkpoint_path:
    saver.restore(sess, checkpoint.model_checkpoint_path)
    print('successed')
else:
    print('could not find old network weights')

for i in range(0,80001):
    g_man_images, g_woman_images, d_man_images, d_woman_images = sess.run([g_man_image, g_woman_image, d_man_image, d_woman_image])
    sess.run([train_ga,train_gb,train_da,train_db],feed_dict={g_man:g_man_images,g_woman:g_woman_images,d_man:d_man_images,d_woman:d_woman_images})
    if i%100==0:
        image_man_los,image_woman_los,g_a_los, g_b_los, d_a_los, d_b_los,generator_a_los,generator_b_los,descrimator_a_los,descrimator_b_los,summary,= \
            sess.run( [image_man_loss,image_woman_loss,g_a_loss, g_b_loss, d_a_loss, d_b_loss,generator_a_loss,generator_b_loss,descrimator_a_loss,
                       descrimator_b_loss,summary_op],
                    feed_dict={g_man:g_man_images,g_woman:g_woman_images,d_man:d_man_images,d_woman:d_woman_images})
        writer.add_summary(summary, global_step=i)
        print(i,'image_man_los:  ',image_man_los,'  image_woman_los:  ',image_woman_los,'  g_a_los: ',
              g_a_los,'  g_b_los:  ',g_b_los,'  d_a_los:  ',d_a_los, '  d_b_los:  ',d_b_los,
              '  generator_a_los:  ', generator_a_los,'  generator_b_los:  ',generator_b_los,
              '  descrimator_a_los: ', descrimator_a_los, '  descrimator_b_los:  ',descrimator_b_los)
    if i%10000==0 and i>0:
        saver.save(sess,'model/cycle_gan_%d.ckpt'%(i),global_step=i)
        config.learning_rate*=0.5
        config.d_learning_rate*=0.5

    if i % 4000==0:
        g_man_images_t, g_woman_images_t = sess.run([g_man_image_test, g_woman_image_test])
        cv2.imwrite('real_image/%05d_woman_by_man.jpg' % i, g_man_images_t[0] * 255)
        cv2.imwrite('real_image/%05d_man_by_woman.jpg' % i, g_woman_images_t[0] * 255)

        image_w,image_m=sess.run([image_man,image_woman],feed_dict={g_man:g_man_images_t,g_woman:g_woman_images_t})
        cv2.imwrite('generate_image/%05d_woman_by_man.jpg'%i,image_w[0]*255)
        cv2.imwrite('generate_image/%05d_man_by_woman.jpg'%i, image_m[0]*255)


