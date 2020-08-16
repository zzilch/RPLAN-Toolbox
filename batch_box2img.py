import os
from multiprocessing import Pool,Manager
import numpy as np
from skimage import io
from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.utils import get_edges,savepkl,savemat,loadpkl

from tqdm.auto import tqdm

img_dir = './data'
img_aligned_dir = './data_aligned'

def boxes2img(boxes, types, order):
    img = np.full((256,256),13,dtype=np.uint8)
    boxes = boxes[order]
    types = types[order]
    for (x0,y0,x1,y1),t in zip(boxes,types):
        img[y0:y1,x0:x1]=t
    return img

def func(data):
    inside = io.imread(f'{img_dir}/{data["name"]}.png')[...,-1]
    img = boxes2img(data['boxes_aligned'],data['types'],data['order'])
    img[inside==0]=13
    io.imsave(f'{img_aligned_dir}/{data["name"]}.png',img)
    return 1
    
class Callback():
    def __init__(self,length):
        self.bar = tqdm(total=length)
        self.output = []

    def update(self, ret):
        self.bar.update(1)
    
    def close(self):
        self.bar.close()
    
if __name__ == '__main__': 
    data_pkl = loadpkl('./output/data_aligned.pkl')
    print(len(data_pkl))
    with Manager() as manager:
        with Pool(32) as pool:
            cb = Callback(len(data_pkl))
            [pool.apply_async(func,args=(data,),callback=cb.update) for data in data_pkl]
            #[cb.update(func(i)) for i in ids]
            
            pool.close()
            pool.join()
            cb.close()
