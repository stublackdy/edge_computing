import torch
import os
import cv2
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import time
class client_model(torch.nn.Module):
    def __init__(self,ip):
        super(client_model, self).__init__()
        
        self.url=ip
        self.input=[]
        self.output=[]
        self.stx=234
        self.sty=206
        self.mult=6
        self.sz=self.mult*256
        self.triton_client = httpclient.InferenceServerClient(url=self.url)



    def inference(self,pic):

        source_dir = './data/'
        results_dir = os.path.abspath(source_dir) + '_res'
        if not os.path.isdir(results_dir):
            os.makedirs(results_dir)


        capture_real = cv2.imread(os.path.join(source_dir, pic))[:,:,0].astype('float')
        capture_crop = capture_real[self.sty:self.sty+self.sz,self.stx:self.stx+self.sz][::-1,::-1].copy()

        input = torch.Tensor(capture_crop)/255*2
        input=input.unsqueeze(0).numpy()

        inputs = []
        inputs.append(httpclient.InferInput('input_0', input.shape, "FP32"))
        inputs[0].set_data_from_numpy(input, binary_data=False)     
        outputs = []

        outputs.append(httpclient.InferRequestedOutput('output_0', binary_data=False))  

        print("inf begin",time.time())
        results = self.triton_client.infer('MY_RT', inputs=inputs, outputs=outputs)
        print("inf end",time.time())
        out=results.as_numpy('output_0')
    
        out=np.array(out)
        out=np.squeeze(out)

        im = []
        for k in range(out.shape[0]):
            temp = out[k,:,:]*255*4
            im.append(Image.fromarray(temp[::-1,::-1]).convert('L'))
        im[0].save(os.path.join(results_dir, pic+'.gif'), save_all=True, append_images=im[1:], duration=150, loop=0x7fff * 2 + 1)