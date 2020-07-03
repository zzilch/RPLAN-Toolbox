import os
import numpy as np
from tqdm.auto import tqdm
from rplan.floorplan import Floorplan
from rplan.align import align_fp_gt
from rplan.decorate import get_dw
from rplan.utils import get_edges,savepkl,savemat
from multiprocessing import Pool,Manager

def func(file_path):
    fp = Floorplan(file_path)
    data = fp.to_dict(dtype=np.uint8)
    boxes_aligned, order, room_boundaries = align_fp_gt(data['boundary'],data['boxes'],data['types'],data['edges'],dtype=np.uint8)
    data['boxes_aligned'] = boxes_aligned
    data['edges_aligned'] = np.array(get_edges(boxes_aligned),dtype=np.uint8)
    data['order'] = order
    data['room_boundaries'] = room_boundaries
    doors,windows = get_dw(data)
    data['doors'] = doors
    data['windows'] = windows
    return data

    
class Callback():
    def __init__(self,length):
        self.bar = tqdm(total=length)
        self.output = []

    def update(self, ret):
        self.output.append(ret)
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
            savepkl('./output/data_decorated.pkl',cb.output)
            # for matlab
            # savemat('./output/data_decorated.mat',{'data':cb.output})

        
