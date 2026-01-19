import torch
import torch.nn.functional as F
import math

#########
# Helpers
#########

def xyz2uv(xyz, up_axis=2):
    """
    xyz: (..., 3) tensor
    returns (..., 2) tensor with (lon, lat) in radians
    """
    norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    xyz_norm = xyz / norm
    other_axes = [i for i in range(0,3) if i!=up_axis]
    x = xyz_norm[..., other_axes[0]]
    y = xyz_norm[..., other_axes[1]]
    z = xyz_norm[..., up_axis]
    
    lon = torch.atan2(x, y)  # range [-pi, pi]
    lat = torch.asin(z)      # range [-pi/2, pi/2]
    u = lon / math.pi
    v = lat / (math.pi/2)

    return torch.stack([u,v], dim=-1)

def equirectangular_sampling(directions, image):
    """
    directions: B, ..., 3
    image: B, H, W, C
    """
    dir_shape = directions.shape
    B = directions.shape[0]

    directions = directions.reshape(B,1,-1,3)
    uv = xyz2uv(directions)

    img_t = image.permute(0, 3, 1, 2) # B, C, H, W

    samples = F.grid_sample(img_t, uv, padding_mode='reflection') # B, C, H, W
    
    samples = samples.permute(0,2,3,1)

    return samples


#sunflower/Fermat–spiral layout with the golden angle φ=π(3−5)
# Vogel, H. (1979). "A better way to construct the sunflower head". Mathematical Biosciences. 44 (3–4): 179–189. doi:10.1016/0025-5564(79)90080-4.
# https://en.wikipedia.org/wiki/Fermat's_spiral
def sunflower_disk(N, device=None):
    N = torch.tensor(N, device=device)
    phi = math.pi * (3 - math.sqrt(5))  # golden angle
    k = torch.arange(N, device=device)
    r = torch.sqrt((k + 0.5) / N)
    theta = k * phi
    x = r * torch.cos(theta)
    y = r * torch.sin(theta)
    return torch.column_stack([x, y])

def project_on_sphere(tensor_to_project, fov, up_axis=2, look_at_axis=1):
    device = tensor_to_project.device
    f = 0.5 * 2 / math.tan(0.5 * math.radians(fov))
    K = torch.tensor([
        [f, 0., 0.],
        [0., f, 0.],
        [0., 0., 1.],
    ], device=device) 
    K_inv = torch.linalg.inv(K)   
    # y is up z is look at now.
    xy = tensor_to_project
    z = torch.ones(xy.shape[0],1,device=device)
    xyz = torch.cat((xy,z), dim=-1)
    xyz = xyz @ K_inv.T  # (H,W,3)
    # reshuffle dims.
    remaining = 3-(up_axis+look_at_axis)
    xyz = xyz[...,[remaining, up_axis, look_at_axis]]
    return xyz

def pitch_yaw_to_rot(pitch, yaw, device=None, pitch_axis=1, yaw_axis=2):
    """
    pitch: B
    yaw: B
    """
    B = pitch.shape[0]
    device = pitch.device
    assert pitch.shape[0] == yaw.shape[0], "pitch and yaw mus have same batch dim"
    cos_y = torch.cos(pitch)
    sin_y = torch.sin(pitch)
    cos_z = torch.cos(yaw)
    sin_z = torch.sin(yaw)
    
    # Rotation arround Y (1)
    R_y = torch.stack([
        torch.stack([cos_y, torch.zeros(B, device=device),  sin_y], dim=-1),
        torch.stack([torch.zeros(B, device=device), torch.ones(B, device=device), torch.zeros(B, device=device)], dim=-1),
        torch.stack([-sin_y, torch.zeros(B, device=device), cos_y], dim=-1),
    ], dim=-2)  # shape (B, 3, 3)
    
    # Rotation arround Z (2)
    R_z = torch.stack([
        torch.stack([cos_z, -sin_z, torch.zeros(B, device=device)], dim=-1),
        torch.stack([sin_z,  cos_z, torch.zeros(B, device=device)], dim=-1),
        torch.stack([torch.zeros(B, device=device), torch.zeros(B, device=device), torch.ones(B, device=device)], dim=-1),
    ], dim=-2)  # shape (B, 3, 3)

    # first rotate arround pitch max 180° (y), then rotate arround yaw max 360° (z)
    R = torch.bmm(R_z, R_y)  # shape (B, 3, 3)
    remaining_axis = 3-(pitch_axis+yaw_axis)
    R[...,:,:] = R[...,[remaining_axis,pitch_axis, yaw_axis],:]
    R[...,:,:] = R[...,:,[remaining_axis,pitch_axis, yaw_axis]]
    return R


    
    R = torch.stack([
        torch.stack([cos_y, sin_y * sin_p, sin_y * cos_p], dim=-1),
        torch.stack([torch.zeros(B, device=device), cos_p, -sin_p], dim=-1),
        torch.stack([-sin_y, cos_y * sin_p, cos_y * cos_p], dim=-1),
    ], dim=-2)  # shape (B, 3, 3)
    return R

def xyz_to_rotation(xyz, up_axis=2, yaw_axis=0):
    uv = xyz2uv(xyz, up_axis=up_axis)
    u,v = uv[...,0], uv[...,1]
    size = xyz.shape[:-1]
    yaw = u*math.pi
    pitch = v*(math.pi/2)
    R = pitch_yaw_to_rot(pitch, yaw, pitch_axis=up_axis, yaw_axis=yaw_axis)
    R = R.reshape(*size,3,3)
    return R

def uv2xyz(uv):
    theta = math.pi * uv[...,0]
    phi = (uv[...,1] + 1.0) * (math.pi / 2)
    
    x = torch.sin(phi) * torch.cos(theta)
    y = torch.sin(phi) * torch.sin(theta)
    z = torch.cos(phi)
    return torch.stack((x, y, z), dim=-1)

########
# Samplers
########


def uniform_directions(shape, device=None):
    """
    Samples direction vectors uniformly
    shape can be arbitrary. but last rank is dimension
    """

    directions = torch.randn(shape, device=device) 
    norm = torch.linalg.norm(directions, dim=-1, keepdim=True)
    directions = directions/norm
    return directions

def sunflower_3d(N, device=None):
    """
    Does direction Sampling on the Sphere with Sunflower sampling
    (Fibonachi sphere, Gonzales, A. (2009))
    """
    N = torch.tensor(N)
    indices = torch.arange(N, device=device) + 0.5
    phi = torch.arccos(1 - 2*indices/N)
    theta = math.pi * (1 + math.sqrt(5)) * indices
    x = torch.cos(theta) * torch.sin(phi)
    y = torch.sin(theta) * torch.sin(phi)
    z = torch.cos(phi)
    return torch.column_stack([x, y, z])


def random_rotation_matrix(*size, pitch_axis=0, yaw_axis=2, device=None):
    """
    Generate a random rotation matrix by sampling pitch and yaw
    so they are uniform on a sphere.
    This way the up direction is preserved.
    """
    yaw = torch.rand(size, device=device)*2*math.pi # [0,2pi]
    pitch = torch.arcsin(torch.rand(size, device=device)*2-1) # [-0.5pi,0.5pi]
    pitch = pitch.reshape(-1)
    yaw = yaw.reshape(-1)
    R = pitch_yaw_to_rot(pitch, yaw, pitch_axis=pitch_axis, yaw_axis=yaw_axis)
    R = R.reshape(*size,3,3)
    return R

def uniform_sunflower_patches(B,N_patches,N_sunflower,fov, device=None):
    """
    Produces Patches in the shape of a sunflower/Fermat–spiral (Vogel, H. (1979))
    which are sampled randomly on the sphere but up is preserved.
    """
    xyz = project_on_sphere(sunflower_disk(N_sunflower, device=device),fov, up_axis=2, look_at_axis=1)
    #xyz = project_on_sphere(sunflower_disk(N),fov, up_axis=2, look_at_axis=1)
    R = random_rotation_matrix(B,N_patches, pitch_axis=0, yaw_axis=2, device=device)
    xyz = torch.einsum('...ij,nj->...ni', R, xyz)  # (B, 3)
    return xyz

def double_sunflower(B,N_patches,N_sunflower,fov, device=None):
    """
    Does Sampling on the Sphere with Sunflower sampling: (Fibonachi sphere, Gonzales, A. (2009))
    And uses Patches in the shape of a sunflower/Fermat–spiral (Vogel, H. (1979))
    """
    xyz = project_on_sphere(sunflower_disk(N_sunflower, device=device),fov, up_axis=2, look_at_axis=1)
    #xyz = project_on_sphere(sunflower_disk(N),fov, up_axis=2, look_at_axis=1)
    xyz_sphere = sunflower_3d(N_patches, device=device)
    R = xyz_to_rotation(xyz_sphere, up_axis=0, yaw_axis=2)
    xyz = torch.einsum('...ij,nj->...ni', R, xyz) 
    xyz = xyz.unsqueeze(0).expand(B,-1,-1,-1)
    return xyz

def line_sampling(B,N_patches,N, device=None):
    height = math.sqrt(N_patches*N*0.5)
    width = height*2
    rows_per_patch = width/N_patches
    height = int(height)
    width = int(width)
    rows = int(rows_per_patch)
    assert height*width == N_patches*N, "N must be a multiple x of N_patches such that sqrt(x*0.5) is a natural Number so x\in{0.5,2,8,18,32,...}"
    
    u = torch.linspace(-1,1,width,device=device).reshape(int(width/rows),rows,1).repeat(1,1,height)
    # torch.allclose(torch.arcsin(torch.linspace(-1,1,100))/math.pi*2, torch.arcsin(torch.linspace(-1,1,100))/math.pi*2)
    v = (1-torch.arccos(torch.linspace(-1,1,height,device=device))/math.pi*2).reshape(1,1,height).repeat(int(width/rows),rows,1)
    uv = torch.stack([u,v], dim=-1).reshape(N_patches,N,2)
    xyz = uv2xyz(uv)
    xyz = xyz.unsqueeze(0).expand(B,-1,-1,-1)
    return xyz

