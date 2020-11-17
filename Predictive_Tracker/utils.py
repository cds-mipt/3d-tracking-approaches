labels = [0, 1, 3, 4, 6, 7, 8]

label_to_str = {
    0: 'car',
    1: 'truck',
    3: 'bus',
    4: 'trailer',
    6: 'motorcycle',
    7: 'bicycle',
    8: 'pedestrian'
}

def label_to_box(l, id):
    return [
        l.box.center_x, 
        l.box.center_y, 
        l.box.center_z, 
        l.box.width, 
        l.box.length, 
        l.box.height, 
        l.box.heading,
        l.type,
        id,
    ]
    
