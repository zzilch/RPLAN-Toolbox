from skimage import io
from skimage import morphology,feature,transform,measure
from pathlib import Path
from scipy import stats
from scipy import ndimage
from shapely import geometry
import numpy as np

from .utils import collide2d,point_box_relation,door_room_relation

class Floorplan():

    @property
    def boundary(self): return self.image[...,0]
    
    @property
    def category(self): return self.image[...,1]

    @property
    def instance(self): return self.image[...,2]
    
    @property
    def inside(self): return self.image[...,3]

    def __init__(self,file_path):
        self.path = file_path
        self.name = Path(self.path).stem
        self.image = io.imread(self.path)
        self.h,self.w,self.c = self.image.shape
        
        self.front_door = None
        self.exterior_boundary = None
        self.rooms = None
        self.edges = None

        self.archs = None
        self.graph = None

        self._get_front_door()
        self._get_exterior_boundary()
        self._get_rooms()
        self._get_edges()
        
    def __repr__(self): 
        return f'{self.name},({self.h},{self.w},{self.c})'

    def _get_front_door(self):
        front_door_mask = self.boundary==255
        # fast bbox
        # min_h,max_h = np.where(np.any(front_door_mask,axis=1))[0][[0,-1]]
        # min_w,max_w = np.where(np.any(front_door_mask,axis=0))[0][[0,-1]]  
        # self.front_door = np.array([min_h,min_w,max_h,max_w],dtype=int)
        region = measure.regionprops(front_door_mask.astype(int))[0]
        self.front_door = np.array(region.bbox,dtype=int)

    def _get_exterior_boundary(self):
        if self.front_door is None: self._get_front_door()
        self.exterior_boundary = []

        min_h,max_h = np.where(np.any(self.boundary,axis=1))[0][[0,-1]]
        min_w,max_w = np.where(np.any(self.boundary,axis=0))[0][[0,-1]]
        min_h = max(min_h-10,0)
        min_w = max(min_w-10,0)
        max_h = min(max_h+10,self.h)
        max_w = min(max_w+10,self.w)

        # src: http://staff.ustc.edu.cn/~fuxm/projects/DeepLayout/index.html
        # search direction:0(right)/1(down)/2(left)/3(up)
        # find the left-top point
        flag = False
        for h in range(min_h, max_h):
            for w in range(min_w, max_w):
                if self.inside[h, w] == 255:
                    self.exterior_boundary.append((h, w, 0))
                    flag = True
                    break
            if flag:
                break
        
        # left/top edge: inside
        # right/bottom edge: outside
        while(flag):
            if self.exterior_boundary[-1][2] == 0:
                for w in range(self.exterior_boundary[-1][1]+1, max_w):
                    corner_sum = 0
                    if self.inside[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
            
            if self.exterior_boundary[-1][2] == 1:      
                for h in range(self.exterior_boundary[-1][0]+1, max_h): 
                    corner_sum = 0                
                    if self.inside[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break

            if self.exterior_boundary[-1][2] == 2:   
                for w in range(self.exterior_boundary[-1][1]-1, min_w, -1):
                    corner_sum = 0                     
                    if self.inside[self.exterior_boundary[-1][0], w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0], w-1] == 255:
                        corner_sum += 1
                    if self.inside[self.exterior_boundary[-1][0]-1, w-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (self.exterior_boundary[-1][0], w, 3)
                        break
                    if corner_sum == 3:
                        new_point = (self.exterior_boundary[-1][0], w, 1)
                        break

            if self.exterior_boundary[-1][2] == 3:       
                for h in range(self.exterior_boundary[-1][0]-1, min_h, -1):
                    corner_sum = 0                
                    if self.inside[h, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]] == 255:
                        corner_sum += 1
                    if self.inside[h, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if self.inside[h-1, self.exterior_boundary[-1][1]-1] == 255:
                        corner_sum += 1
                    if corner_sum == 1:
                        new_point = (h, self.exterior_boundary[-1][1], 0)
                        break
                    if corner_sum == 3:
                        new_point = (h, self.exterior_boundary[-1][1], 2)
                        break

            if new_point != self.exterior_boundary[0]:
                self.exterior_boundary.append(new_point)
            else:
                flag = False
        self.exterior_boundary = [[r,c,d,0] for r,c,d in self.exterior_boundary]
        
        door_y1,door_x1,door_y2,door_x2 = self.front_door
        door_h,door_w = door_y2-door_y1,door_x2-door_x1
        is_vertical = door_h>door_w or door_h==1 # 

        insert_index = None
        door_index = None
        new_p = []
        th = 3
        for i in range(len(self.exterior_boundary)):
            y1,x1,d,_ = self.exterior_boundary[i]
            y2,x2,_,_ = self.exterior_boundary[(i+1)%len(self.exterior_boundary)] 
            if is_vertical!=d%2: continue
            if is_vertical and (x1-th<door_x1<x1+th or x1-th<door_x2<x1+th): # 1:down 3:up
                l1 = geometry.LineString([[y1,x1],[y2,x2]])    
                l2 = geometry.LineString([[door_y1,x1],[door_y2,x1]])  
                l12 = l1.intersection(l2)
                if l12.length>0:
                    dy1,dy2 = l12.xy[0] # (y1>y2)==(dy1>dy2)
                    insert_index = i
                    door_index = i+(y1!=dy1)
                    if y1!=dy1: new_p.append([dy1,x1,d,1])
                    if y2!=dy2: new_p.append([dy2,x1,d,1])
            elif not is_vertical and (y1-th<door_y1<y1+th or y1-th<door_y2<y1+th):
                l1 = geometry.LineString([[y1,x1],[y2,x2]])    
                l2 = geometry.LineString([[y1,door_x1],[y1,door_x2]])  
                l12 = l1.intersection(l2)
                if l12.length>0:
                    dx1,dx2 = l12.xy[1] # (x1>x2)==(dx1>dx2)
                    insert_index = i
                    door_index = i+(x1!=dx1)
                    if x1!=dx1: new_p.append([y1,dx1,d,1])
                    if x2!=dx2: new_p.append([y1,dx2,d,1])                

        if len(new_p)>0:
            self.exterior_boundary = self.exterior_boundary[:insert_index+1]+new_p+self.exterior_boundary[insert_index+1:]
        self.exterior_boundary = self.exterior_boundary[door_index:]+self.exterior_boundary[:door_index]

        self.exterior_boundary = np.array(self.exterior_boundary,dtype=int)

    def _get_rooms(self):
        rooms = []
        regions = measure.regionprops(self.instance)
        for region in regions:
            c = stats.mode(self.category[region.coords[:,0],region.coords[:,1]])[0][0]
            y0,x0,y1,x1 = np.array(region.bbox) 
            rooms.append([y0,x0,y1,x1,c])
        self.rooms = np.array(rooms,dtype=int)

    def _get_edges(self,th=9):
        if self.rooms is None: self._get_rooms()
        edges = []
        for u in range(len(self.rooms)):
            for v in range(u+1,len(self.rooms)):
                if not collide2d(self.rooms[u,:4],self.rooms[v,:4],th=th): continue
                uy0, ux0, uy1, ux1, c1 = self.rooms[u]
                vy0, vx0, vy1, vx1, c2 = self.rooms[v]
                uc = (uy0+uy1)/2,(ux0+ux1)/2
                vc = (vy0+vy1)/2,(vx0+vx1)/2
                if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                    relation = 5 #'surrounding'
                elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                    relation = 4 #'inside'
                else:
                    relation = point_box_relation(uc,self.rooms[v,:4])
                edges.append([u,v,relation])
                
        self.edges = np.array(edges,dtype=int)

    def _get_archs(self):
        '''
        Interior doors
        '''
        archs = []
        
        # treat archs as instances
        # index = len(self.rooms)+1

        # for category in range(num_category,len(room_label)):
        for category in [17]: # only get doors for building graphs
            mask = (self.category==category).astype(np.uint8)

            # distance transform -> threshold -> corner detection -> remove corner -> watershed -> label region
            #distance = cv2.distanceTransform(mask,cv2.DIST_C,3)
            distance = ndimage.morphology.distance_transform_cdt(mask)

            # local_maxi = feature.peak_local_max(distance, indices=False) # line with one pixel
            local_maxi = (distance>1).astype(np.uint8) 

            # corner_measurement = feature.corner_shi_tomasi(local_maxi) # short lines will be removed
            corner_measurement = feature.corner_harris(local_maxi) 

            local_maxi[corner_measurement>0] = 0

            markers = measure.label(local_maxi)

            labels = morphology.watershed(-distance, markers, mask=mask, connectivity=8)
            regions = measure.regionprops(labels)

            for region in regions:
                y0,x0,y1,x1 = np.array(region.bbox) 
                archs.append([y0,x0,y1,x1,category])

        self.archs = np.array(archs,dtype=int)

    def _get_graph(self,th=9):
        '''
        More detail graph
        '''
        if self.rooms is None: self._get_rooms()
        if self.archs is None: self._get_archs()
        graph = []
        door_pos = [[None,0] for i in range(len(self.rooms))]
        edge_set = set()

        # add accessible edges
        doors = self.archs[self.archs[:,-1]==17]
        for i in range(len(doors)):
            bbox = doors[i,:4]
            
            # left <-> right
            for row in range(bbox[0],bbox[2]):

                u = self.instance[row,bbox[1]-1]-1
                v = self.instance[row,bbox[3]+1]-1
                if (u,v) in edge_set or (v,u) in edge_set:continue
                if u>=0 and v>=0 and (u,v):
                    edge_set.add((u,v))
                    graph.append([u,v,None,1,i])

            # up <-> down
            for col in range(bbox[1],bbox[3]):
                u = self.instance[bbox[0]-1,col]-1
                v = self.instance[bbox[2]+1,col]-1
                if (u,v) in edge_set or (v,u) in edge_set:continue
                if u>=0 and v>=0 and (u,v):
                    edge_set.add((u,v))
                    graph.append([u,v,None,1,i])

        # add adjacent edges
        for u in range(len(self.rooms)):
            for v in range(u+1,len(self.rooms)):
                if (u,v) in edge_set or (v,u) in edge_set: continue

                # collision detection
                if collide2d(self.rooms[u,:4],self.rooms[v,:4],th=th):
                    edge_set.add((u,v))
                    graph.append([u,v,None,0,None])

        # add edge relation
        for i in range(len(graph)):
            u,v,e,t,d = graph[i]
            uy0, ux0, uy1, ux1 = self.rooms[u,:4]
            vy0, vx0, vy1, vx1 = self.rooms[v,:4]
            uc = (uy0+uy1)/2,(ux0+ux1)/2
            vc = (vy0+vy1)/2,(vx0+vx1)/2

            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 5 #'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 4 #'inside'
            else:
                relation = point_box_relation(uc,self.rooms[v,:4])
            
            graph[i][2] = relation


            if d is not None:
                c_u = self.rooms[u,-1]
                c_v = self.rooms[v,-1]

                if c_u > c_v and door_pos[u][0] is None:
                    room = u
                else:
                    room = v
                door_pos[room][0]=d
                

                d_center = self.archs[d,:4]
                d_center = (d_center[:2]+d_center[2:])/2.0

                dpos =  door_room_relation(d_center,self.rooms[room,:4])
                if dpos!=0: door_pos[room][1] = dpos
        
        self.graph = graph
        self.door_pos = door_pos

    def to_dict(self,xyxy=True,dtype=int):
        '''
        Compress data, notice:
        !!! int->uint8: a(uint8)+b(uint8) may overflow !!!
        '''
        return {
            'name'      :self.name,
            'types'     :self.rooms[:,-1].astype(dtype),
            'boxes'     :(self.rooms[:,[1,0,3,2]]).astype(dtype) 
            if xyxy else self.rooms[:,:4].astype(dtype),
            'boundary'  :self.exterior_boundary[:,[1,0,2,3]].astype(dtype)
            if xyxy else self.exterior_boundary.astype(dtype),
            'edges'     :self.edges.astype(dtype)
        }

if __name__ == "__main__":
    pass