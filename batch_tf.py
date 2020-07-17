import os
import numpy as np
from tqdm.auto import tqdm
from rplan.floorplan import Floorplan
from rplan.measure import compute_tf,sample_tf
from rplan.utils import get_edges,savepkl,savemat
from multiprocessing import Pool,Manager

def func(file_path):
    fp = Floorplan(file_path)
    data = fp.to_dict(dtype=int)
    x,y = compute_tf(data['boundary'])
    tf_piecewise = np.array([x,y]).astype(np.float32)
    tf_discrete = sample_tf(x,y).astype(np.float32)
    return fp.name,tf_piecewise,tf_discrete

    
class Callback():
    def __init__(self,length):
        self.bar = tqdm(total=length)
        self.tf_piecewise = {}
        self.tf_discrete = {}

    def update(self, ret):
        name,tf_piecewise,tf_discrete = ret
        self.tf_piecewise[name] = tf_piecewise
        self.tf_discrete[name] = tf_discrete
        self.bar.update(1)
    
    def close(self):
        self.bar.close()
    
if __name__ == '__main__': 
    img_dir = './data'
    ids = os.listdir(img_dir)
    ids = [f'{img_dir}/{i}' for i in ids]
    print(len(ids))
    with Manager() as manager:
        with Pool(32) as pool:
            cb = Callback(len(ids))
            [pool.apply_async(func,args=(i,),callback=cb.update) for i in ids]
            pool.close()
            pool.join()
            cb.close()
            savepkl('./output/tf_piecewise.pkl',cb.tf_piecewise)
            savepkl('./output/tf_discrete.pkl',cb.tf_discrete)
            # for matlab

        
