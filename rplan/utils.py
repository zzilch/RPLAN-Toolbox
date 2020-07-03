import numpy as np
import scipy.io as sio
import pickle

room_label = [(0, 'LivingRoom', 1, "PublicArea"),
            (1, 'MasterRoom', 0, "Bedroom"),
            (2, 'Kitchen', 1, "FunctionArea"),
            (3, 'Bathroom', 0, "FunctionArea"),
            (4, 'DiningRoom', 1, "FunctionArea"),
            (5, 'ChildRoom', 0, "Bedroom"),
            (6, 'StudyRoom', 0, "Bedroom"),
            (7, 'SecondRoom', 0, "Bedroom"),
            (8, 'GuestRoom', 0, "Bedroom"),
            (9, 'Balcony', 1, "PublicArea"),
            (10, 'Entrance', 1, "PublicArea"),
            (11, 'Storage', 0, "PublicArea"),
            (12, 'Wall-in', 0, "PublicArea"),
            (13, 'External', 0, "External"),
            (14, 'ExteriorWall', 0, "ExteriorWall"),
            (15, 'FrontDoor', 0, "FrontDoor"),
            (16, 'InteriorWall', 0, "InteriorWall"),
            (17, 'InteriorDoor', 0, "InteriorDoor")]
    
def savemat(file_path,data):
    sio.savemat(file_path,data)

def loadmat(file_path):
    return sio.loadmat(file_path, squeeze_me=True, struct_as_record=False)

def savepkl(file_path,data):
    pickle.dump(data,open(file_path,'wb'))

def loadpkl(file_path):
    return pickle.load(open(file_path,'rb'))

def get_color_map():
    color = np.array([
        [244,242,229], # living room
        [253,244,171], # bedroom
        [234,216,214], # kitchen
        [205,233,252], # bathroom
        [208,216,135], # balcony
        [185,231,168], # balcony
        [249,222,189], # Storage
        [ 79, 79, 79], # exterior wall
        [255,225, 25], # FrontDoor
        [128,128,128], # interior wall
        [255,255,255]
    ],dtype=np.int64)
    cIdx  = np.array([1,2,3,4,1,2,2,2,2,5,1,6,1,10,7,8,9,10])-1
    return color[cIdx]

def collide2d(bbox1, bbox2, th=0):
    return not(
        (bbox1[0]-th > bbox2[2]) or
        (bbox1[2]+th < bbox2[0]) or
        (bbox1[1]-th > bbox2[3]) or
        (bbox1[3]+th < bbox2[1])
    )

edge_type = ['left-above',
    'left-below',
    'left-of',
    'above',
    'inside',
    'surrounding',
    'below',
    'right-of',
    'right-above',
    'right-below']

def point_box_relation(u,vbox):
    uy,ux = u
    vy0, vx0, vy1, vx1 = vbox
    if (ux<vx0 and uy<=vy0) or (ux==vx0 and uy==vy0):
        relation = 0 # 'left-above'
    elif (vx0<=ux<vx1 and uy<=vy0):
        relation = 3 # 'above'
    elif (vx1<=ux and uy<vy0) or (ux==vx1 and uy==vy0):
        relation = 8 # 'right-above'
    elif (vx1<=ux and vy0<=uy<vy1):
        relation = 7 # 'right-of'
    elif (vx1<ux and vy1<=uy) or (ux==vx1 and uy==vy1):
        relation = 9 # 'right-below'
    elif (vx0<ux<=vx1 and vy1<=uy):
        relation = 6 # 'below'
    elif (ux<=vx0 and vy1<uy) or (ux==vx0 and uy==vy1):
        relation = 1 # 'left-below'
    elif(ux<=vx0 and vy0<uy<=vy1):
        relation = 2 # 'left-of'
    elif(vx0<ux<vx1 and vy0<uy<vy1):
        relation = 4 # 'inside'

    return relation

def get_edges(boxes,th=9):
    edges = []
    for u in range(len(boxes)):
        for v in range(u+1,len(boxes)):
            if not collide2d(boxes[u,:4],boxes[v,:4],th=th): continue
            uy0, ux0, uy1, ux1 = boxes[u,:4].astype(int)
            vy0, vx0, vy1, vx1 = boxes[v,:4].astype(int)
            uc = (uy0+uy1)/2,(ux0+ux1)/2
            vc = (vy0+vy1)/2,(vx0+vx1)/2
            if ux0 < vx0 and ux1 > vx1 and uy0 < vy0 and uy1 > vy1:
                relation = 5 #'surrounding'
            elif ux0 >= vx0 and ux1 <= vx1 and uy0 >= vy0 and uy1 <= vy1:
                relation = 4 #'inside'
            else:
                relation = point_box_relation(uc,boxes[v,:4])
            edges.append([u,v,relation])
            
    edges = np.array(edges,dtype=int)
    return edges

door_pos = [
    'nan',
    'bottom',
    'bottom-right','right-bottom',
    'right',
    'right-top','top-right',
    'top',
    'top-left','left-top',
    'left',
    'left-bottom','bottom-left'
]

def door_room_relation(d_center,r_box):
    y0,x0,y1,x1 = r_box
    yc,xc = (y1+y0)/2, (x0+x1)/2
    y,x = d_center
    
    if x==xc and y<yc:return 7
    elif x==xc and y>yc:return 1
    elif y==yc and x<xc:return 10
    elif y==yc and x>xc:return 4
    elif x0<x<xc:
        if y<yc:return 8
        else:return 12
    elif xc<x<x1:
        if y<yc:return 6
        else:return 2
    elif y0<y<yc:
        if x<xc:return 9
        else:return 5
    elif yc<y<y1:
        if x<xc:return 11
        else:return 3
    else:return 0