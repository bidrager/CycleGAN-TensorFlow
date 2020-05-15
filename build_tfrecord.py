import tensorflow as tf
import  os
import random
import cv2

# def _int64_feature(value):
#     if not isinstance(value,list):
#         value=[value]
#     return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def _byte_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _convet_to_example(file_name,image):
    image=image.tobytes()
    example=tf.train.Example(features=tf.train.Features(feature={
        'image_name':_byte_feature(tf.compat.as_bytes(file_name)),
        'image':_byte_feature(image)
         }))
    return example

def get_name_list(source_path):
    name_list=os.listdir(source_path)
    random.seed(1)
    random.shuffle(name_list)
    return name_list[0:-100],name_list[-100:]


def write_data_tfreocord(source_path,name_list,tfrecord_dir):
    writer=tf.python_io.TFRecordWriter(tfrecord_dir)
    for image_name in  name_list:
        image_path=os.path.join(source_path,image_name)
        image=cv2.imread(image_path)
        example=_convet_to_example(image_name,image)
        writer.write(example.SerializeToString())

if __name__=='__main__':
    man_data_dir='../data/cycle_gan_data/a_resized'
    woman_data_dir = '../data/cycle_gan_data/b_resized'
    man_train_tfrecord='data/man_train_tfrecord.tfrecords'
    man_test_tfrecord = 'data/man_test_tfrecord.tfrecords'
    woman_train_tfrecord = 'data/woman_train_tfrecord.tfrecords'
    woman_test_tfrecord = 'data/woman_test_tfrecord.tfrecords'
    man_train,man_test=get_name_list(man_data_dir)
    print(len(man_train),len(man_test))
    woman_train, woman_test = get_name_list(woman_data_dir)
    print(len(woman_train), len(woman_test))
    print('write man train')
    write_data_tfreocord(source_path=man_data_dir,name_list=man_train,tfrecord_dir=man_train_tfrecord)
    print('write man test')
    write_data_tfreocord(source_path=man_data_dir,name_list=man_test,tfrecord_dir=man_test_tfrecord)
    print('write woman train')
    write_data_tfreocord(source_path=woman_data_dir,name_list=woman_train,tfrecord_dir=woman_train_tfrecord)
    print('write woman test')
    write_data_tfreocord(source_path=woman_data_dir,name_list=woman_test,tfrecord_dir=woman_test_tfrecord)
    print('done')