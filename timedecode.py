import numpy as np
from PIL import Image
from zfpy import decompress_numpy
from os.path import join
import time 
from os import listdir


data_dir = './data/profiling'

d = join(data_dir, 'npy')
start_time  = time.time()
for partition in listdir(d):
    partition_f = join(d, partition)
    for c in listdir(partition_f):
        class_f = join(partition_f, c)
        for img in listdir(class_f):
            a = np.load(join(class_f, img))
print(f'Numpy (Raw) decode time: {time.time() - start_time} seconds')


d = join(data_dir, 'jpg')
start_time  = time.time()
for partition in listdir(d):
    partition_f = join(d, partition)
    for c in listdir(partition_f):
        class_f = join(partition_f, c)
        for img in listdir(class_f):
            a = np.asarray(Image.open(join(class_f, img)))
print(f'JPEG (Pillow) decode time: {time.time() - start_time} seconds')


d = join(data_dir, 'zfp')
start_time  = time.time()
for partition in listdir(d):
    partition_f = join(d, partition)
    for c in listdir(partition_f):
        class_f = join(partition_f, c)
        for img in listdir(class_f):
            with open(join(class_f, img), 'rb') as fp:
                b = fp.read()
            a = decompress_numpy(b)
print(f'ZFP (Non-CUDA) decode time: {time.time() - start_time} seconds')

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

batch_size = 1
class SimplePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, image_dir, device_id):
        super(SimplePipeline, self).__init__(batch_size, num_threads, device_id, seed = 12)
        self.input = ops.FileReader(file_root = image_dir)
        # instead of path to file directory file with pairs image_name image_label_value can be provided
        # self.input = ops.FileReader(file_root = image_dir, file_list = image_dir + '/file_list.txt')
        self.decode = ops.ImageDecoder(device = 'cpu', output_type = types.RGB)

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return (images, labels)

start = time.time()
train_pipe = SimplePipeline(batch_size, 1, './data/profiling/jpg/train', 0)
test_pipe =SimplePipeline(batch_size, 1, './data/profiling/jpg/test', 0)
train_pipe.build()
test_pipe.build()
train_out = train_pipe.run()
test_out = test_pipe.run()
print(f'DALI (CPU) Time: {time.time() - start} seconds')