from PIL import Image
import subprocess
import numpy as np
import torch 
import tempfile

class zfpycoefficient(object):

    def __init__(self, precision=-1, rate=-1):
        self.precision = precision
        self.rate = rate
        assert not (self.rate > 0 and self.precision >
                0), 'Only precision OR rate may be specified'


    def __call__(self, tensor):
        """
        Expects a torch tensor and returns a torch tensor with zfpy coefficients
        """

        if self.precision > 0:
            cmd = f'./ppm -{self.precision}'
        else:
            cmd = f'./ppm {self.rate}'
        img = (np.transpose(tensor.numpy(), (1, 2, 0)) + 3) / 6
        img = (img * 255).astype(np.uint8)
        img = Image.fromarray(img, mode='RGB')
        fp = tempfile.TemporaryFile()
        img.save(fp)
        fp.seek(0)
        proccess = subprocess.Popen(
            cmd.split(), stdin=fp, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        compressed_img, error = proccess.communicate()
        fp.close()
        
        #TODO REPLACE THIS
        reconstructed_img = np.asarray(Image.frombuffer(
            'RGB', (img.shape[0], img.shape[1]), compressed_img)).astype(np.float) / 255
        reconstructed_img = np.transpose((reconstructed_img * 6) - 3, (2, 0, 1))
        #TODO REPLACE THIS
        
        return reconstructed_img

    def __repr__(self):
        s = 'ZFP coefficient extractor with params:\n'
        for k in self.zfpy_args.keys():
            s += str(k) + ' = ' + str(self.zfpy_args[k])

        return s
