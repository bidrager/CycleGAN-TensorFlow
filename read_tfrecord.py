import tensorflow as tf



def read_batch_image(tfrecord_dir,batch_size=1,num_threads=2):
    filename_queue=tf.train.string_input_producer([tfrecord_dir])
    reader=tf.TFRecordReader()
    _,example=reader.read(filename_queue)
    features=tf.parse_single_example(example,
                                     features={
                                         'image_name': tf.FixedLenFeature([],tf.string),
                                         'image': tf.FixedLenFeature([],tf.string)
                                     })

    image = tf.decode_raw(features['image'],tf.uint8)
    image = tf.reshape(image, [256, 256,3])
    image=image/255
    images = tf.train.shuffle_batch(
        [image], batch_size=batch_size, num_threads=num_threads,
        capacity=10*batch_size,min_after_dequeue=3)
    return images




