import zfpy
import numpy as np
import torch 

class zfpycoefficient(object):

    def __init__(self, **kwargs):
        self.zfpy_args = kwargs

    def __call__(self, tensor):
        """
        Expects a torch tensor and returns a torch tensor with zfpy coefficients
        """
        if self.zfpy_args['precision'] == -1:
            return tensor
        img_np = tensor.numpy()
        return torch.from_numpy(zfpy.decompress_numpy(zfpy.compress_numpy(img_np, **self.zfpy_args)))

    def __repr__(self):
        s = 'ZFP coefficient extractor with params:\n'
        for k in self.zfpy_args.keys():
            s += str(k) + ' = ' + str(self.zfpy_args[k])

        return s
