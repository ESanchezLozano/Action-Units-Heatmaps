import dlib
import cv2
import torch
import numpy as np
from hourglass import FullNetwork as fn
import scipy.io as sio
import os
try:
    import urllib.request as request_file
except BaseException:
    import urllib as request_file

class AUdetector:

    def __init__(self, path_to_predictor='shape_predictor_68_face_landmarks.dat', enable_cuda=True):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(path_to_predictor)
        # Initialise AU detector
        self.AUdetector = fn(1)
        self.enable_cuda = enable_cuda
        if not os.path.exists('model'):
            os.mkdir('model')

        if not os.path.isfile('model/AUdetector.pth.tar'):
            request_file.urlretrieve(
                    "https://esanchezlozano.github.io/files/AUdetector.pth.tar",
                    'model/AUdetector.pth.tar')

        #net_weigths = torch.load('model/AUdetector.pth.tar') # FIXED BUG IN CPU MODEL
        net_weigths = torch.load('model/AUdetector.pth.tar', map_location=lambda storage, loc: storage)
        net_dict = {k.replace('module.',''): v for k, v in net_weigths['state_dict'].items()}
        
        self.AUdetector.load_state_dict(net_dict)
        if self.enable_cuda:
            self.AUdetector = self.AUdetector.cuda()
        self.AUdetector.eval()

    def detectAU(self,image):
        if isinstance(image, str):
            image = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2RGB)
        dets = self.detector(image)
        shape = self.predictor(image,dets[0])
        coords = np.zeros((68, 2), dtype='float')
        for i in range(0,68):
            coords[i] = (float(shape.part(i).x),float(shape.part(i).y))
        frame = self.transform(image, coords)
        image = frame.swapaxes(2,1).swapaxes(1,0)/255.0
        input = torch.from_numpy(image).float().unsqueeze(0)
        if self.enable_cuda:
            input = input.cuda()

        input_var = torch.autograd.Variable(input)
        outputs = self.AUdetector(input_var)
        pred = np.zeros(5)
        out_tmp = outputs[-1][0,:,:,:]
        for k in range(0,5):
            tmp = out_tmp[k,:,:].data.max()
            if tmp < 0:
                tmp = 0
            elif tmp > 5:
                tmp = 5
            pred[k] = tmp

        if self.enable_cuda:
            maps = out_tmp.cpu()
        else:
            maps = out_tmp
        return pred, maps, frame        

    def transform(self,image,landmarks,s0=None):
        """ s0 is the points registered in the centre of the 256x256 image """
        if s0 is None:
            s0 = np.array([[127.6475, 227.8161], [79.1608, 87.0376], [176.8392, 87.0376]], np.float32)
        idx = [8,36,45] #"""Anchor points"""
        pts = np.float32(landmarks[idx,:])
        M = cv2.getAffineTransform(pts,s0)
        dst = cv2.warpAffine(image, M, (256,256))
        return dst











