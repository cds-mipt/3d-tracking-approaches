import torch

device = 'cpu'

def _get_min_range(a, b):
    a_ = a.repeat(1, b.size(0)).reshape(a.size(0), b.size(0), 2)
    b_ = b.repeat(a.size(0), 1).reshape(a.size(0), b.size(0), 2)

    c = a[:,0,None].le(b[:,0]).unsqueeze(2) # (N, M, 1)
    d = torch.where(c, a_, b_)
    return d

def _get_max_range(a, b):
    a_ = a.repeat(1, b.size(0)).reshape(a.size(0), b.size(0), 2)
    b_ = b.repeat(a.size(0), 1).reshape(a.size(0), b.size(0), 2)

    c = a[:,0,None].ge(b[:,0]).unsqueeze(2) # (N, M, 1)
    d = torch.where(c, a_, b_)
    return d

def _range_intersection(range1, range2):
    min_range = _get_min_range(range1, range2) # (N, M, 2)
    max_range = _get_max_range(range1, range2) # (N, M, 2)
    zeros = torch.zeros(range1.size(0), range2.size(0)).to(device)
    real = torch.where(min_range[:,:,1] < max_range[:,:,1], min_range[:,:,1], max_range[:,:,1]) - max_range[:,:,0]
    out = torch.where(min_range[:,:,1] < max_range[:,:,0], zeros, real)
    return out # (N, M)

def _get_z_intersection(set1, set2):
    set1_zrange = torch.stack([set1[:,2] - set1[:,5] / 2, set1[:,2] + set1[:,5] / 2], dim=1) # (N, 2)
    set2_zrange = torch.stack([set2[:,2] - set2[:,5] / 2, set2[:,2] + set2[:,5] / 2], dim=1) # (M, 2)
    return _range_intersection(set1_zrange, set2_zrange) # (N, M)

def _rotation_transform(rotation_center, angle):
    set_size = angle.size(0)
    ones = torch.ones(set_size).to(device)
    zeros = torch.zeros(set_size).to(device)
    
    t1_mat = torch.stack([
        torch.stack([ones, zeros, -rotation_center[:,0]]).transpose(1, 0),
        torch.stack([zeros, ones, -rotation_center[:,1]]).transpose(1, 0),
        torch.stack([zeros, zeros, ones]).transpose(1, 0),
    ]).transpose(1, 0).to(device)

    t2_mat = torch.stack([
        torch.stack([ones, zeros, rotation_center[:,0]]).transpose(1, 0),
        torch.stack([zeros, ones, rotation_center[:,1]]).transpose(1, 0),
        torch.stack([zeros, zeros, ones]).transpose(1, 0),
    ]).transpose(1, 0).to(device)

    r_mat = torch.stack([
        torch.stack([torch.cos(angle), -torch.sin(angle), zeros]).transpose(1, 0),
        torch.stack([torch.sin(angle), torch.cos(angle), zeros]).transpose(1, 0),
        torch.stack([zeros, zeros, ones]).transpose(1, 0),
    ]).transpose(1, 0).to(device)
    
    return t2_mat.matmul(r_mat.matmul(t1_mat))

def _corner_points(s):
    """
    s: Tensor (N, 7)
    """
    ones_tensor = torch.ones(s.size(0)).to(device)
    
    return torch.stack([
        torch.stack([s[:,0] - s[:,3] / 2, s[:,1] + s[:,4] / 2, ones_tensor]).transpose(1, 0),
        torch.stack([s[:,0] + s[:,3] / 2, s[:,1] + s[:,4] / 2, ones_tensor]).transpose(1, 0),
        torch.stack([s[:,0] - s[:,3] / 2, s[:,1] - s[:,4] / 2, ones_tensor]).transpose(1, 0),
        torch.stack([s[:,0] + s[:,3] / 2, s[:,1] - s[:,4] / 2, ones_tensor]).transpose(1, 0),
    ]).transpose(1, 0) # (N, 4, 3)

def iou3d(set1, set2):
    """
    set1: Tensor (N, 7) (cx, cy, cz, w, l, h, theta)
    set2: Tensor (M, 7) (cx, cy, cz, w, l, h, theta)
    """
    z_inter = _get_z_intersection(set1, set2)  # (N, M)
    
    points1 = _corner_points(set1) # (N, 4, 3)
    points2 = _corner_points(set2) # (M, 4, 3)
    transform_mat1 = _rotation_transform(set1[:,0:2], set1[:,6]) # (N, 3, 3)
    transform_mat2 = _rotation_transform(set2[:,0:2], set2[:,6]) # (M, 3, 3)
    
    points1 = torch.einsum('kji,kbi->kbj', transform_mat1, points1)[:,:,:2] # (N, 4, 2)
    points2 = torch.einsum('kji,kbi->kbj', transform_mat2, points2)[:,:,:2] # (M, 4, 2)
    
    x_range1 = torch.stack([points1[:,:,0].min(1)[0], points1[:,:,0].max(1)[0]]).transpose(1, 0) # (N, 2)
    x_range2 = torch.stack([points2[:,:,0].min(1)[0], points2[:,:,0].max(1)[0]]).transpose(1, 0) # (M, 2)
    x_inter = _range_intersection(x_range1, x_range2) # (N, M)
    
    y_range1 = torch.stack([points1[:,:,1].min(1)[0], points1[:,:,1].max(1)[0]]).transpose(1, 0) # (N, 2)
    y_range2 = torch.stack([points2[:,:,1].min(1)[0], points2[:,:,1].max(1)[0]]).transpose(1, 0) # (M, 2)
    y_inter = _range_intersection(y_range1, y_range2) # (N, M)
    
    intersection_area = x_inter * y_inter # (N, M)
    set1_area = set1[:,3] * set1[:,4] # (N)
    set2_area = set2[:,3] * set2[:,4] # (M)
    
    union_area = (set1_area[:, None] + set2_area) - intersection_area
    set1_volume = set1_area * set1[:,5]
    set2_volume = set2_area * set2[:,5]

    intersection_volume = intersection_area * z_inter
    union_volume = (set1_volume[:, None] + set2_volume) - intersection_volume
    ones = torch.ones(set1.size(0), set2.size(0)).to(device)
    zeros = torch.zeros(set1.size(0), set2.size(0)).to(device)
    iou = torch.where(union_volume == zeros, ones, intersection_volume / union_volume)
    
    return iou.abs() # (N, M)
