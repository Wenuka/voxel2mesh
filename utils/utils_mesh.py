import numpy as np

import numpy as np
import ctypes
from ctypes import *
import torch

from skimage import measure
import torch.nn.functional as F
import time
# from IPython import embed
from scipy.io import savemat
from utils.utils_common import volume_suffix
# http://bikulov.org/blog/2013/10/01/using-cuda-c-plus-plus-functions-in-python-via-star-dot-so-and-ctypes/
# nvcc -ccbin=/usr/bin/gcc-4.8 -Xcompiler -fPIC -shared -o kernel.so kernel.cu
# for cmake -DCMAKE_C_COMPILER=/usr/bin/gcc-4.8

# extract cuda_sum function pointer in the shared object cuda_sum.so
def cuda_get_rasterize():
    dll = ctypes.CDLL('/cvlabdata2'+volume_suffix+'/home/wickrama/projects/U-Net/Experiments/meshnet/mnet/kernel.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_rasterize
    func.argtypes = [POINTER(c_int), POINTER(c_float), POINTER(c_int), POINTER(c_float), c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]

    ctypes._reset_cache()
    return func

# create __cuda_sum function with get_cuda_sum()
__cuda_rasterize = cuda_get_rasterize()

# convenient python wrapper for __cuda_sum
# it does all job with types convertation
# from python ones to C++ ones
def cuda_rasterize(grid, vertices, faces, D, H, W, N_vertices, N_faces, debug):
    grid_p = grid.ctypes.data_as(POINTER(c_int))
    vertices_p = vertices.ctypes.data_as(POINTER(c_float))
    faces_p = faces.ctypes.data_as(POINTER(c_int))
    debug_p = debug.ctypes.data_as(POINTER(c_float))

    __cuda_rasterize(grid_p, vertices_p, faces_p, debug_p, D, H, W, N_vertices, N_faces)


def cuda_rasterize2(grid, vertices, faces, D, H, W, N_vertices, N_faces, debug):
    grid_p = grid.ctypes.data_as(POINTER(c_int))
    vertices_p = vertices.ctypes.data_as(POINTER(c_float))
    faces_p = faces.ctypes.data_as(POINTER(c_int))
    debug_p = debug.ctypes.data_as(POINTER(c_float))

    dll = ctypes.CDLL('/cvlabdata2'+volume_suffix+'/home/wickrama/projects/U-Net/Experiments/meshnet/mnet/kernel.so', mode=ctypes.RTLD_GLOBAL)
    func = dll.cuda_rasterize
    func.argtypes = [POINTER(c_int), POINTER(c_float), POINTER(c_int), POINTER(c_float), c_size_t, c_size_t, c_size_t, c_size_t, c_size_t]
    func(grid_p, vertices_p, faces_p, debug_p, D, H, W, N_vertices, N_faces)

    del dll



# testing, sum of two arrays of ones and output head part of resulting array
def rasterize_gpu(vertices, faces, grid_size):

    D, H, W = grid_size
    N_vertices = len(vertices)
    N_faces = len(faces)
    volume = np.zeros(grid_size).astype('int32')
    debug = np.zeros(grid_size).astype('float32')
    vertices = vertices.astype('float32')
    faces = faces.astype('int32')

    cuda_rasterize(volume, vertices, faces, D, H, W, N_vertices, N_faces, debug)


    return volume, debug
 
def read_obj(filepath):
    vertices = []
    faces = [] 
    normals = []   
    with open(filepath) as fp:
        line = fp.readline() 
        cnt = 1 
        while line: 
            if line[0] is not '#': 
                cnt = cnt + 1 
                values = [float(x) for x in line.split('\n')[0].split(' ')[1:]] 
                if line[:2] == 'vn':  
                    normals.append(values)
                elif line[0] == 'v':
                    vertices.append(values)
                elif line[0] == 'f':
                    faces.append(values) 
            line = fp.readline()
        vertices = np.array(vertices)
        normals = np.array(normals)
        faces = np.array(faces)
        faces = np.int64(faces) - 1
        if len(normals) > 0:
            return vertices, faces, normals
        else:
            return vertices, faces

def run_rasterize(vertices, faces_, grid_size):
    v = [vertices[faces_[:, i], :] for i in range(3)]
    face_areas = np.abs(np.cross(v[2] - v[0], v[1] - v[0]) / 2)
    face_areas = np.linalg.norm(face_areas, axis=1)
    faces = faces_[face_areas > 0]

    labels, _ = rasterize_gpu(vertices, faces, grid_size=grid_size)

    return labels


def save_to_obj(filepath, points, faces, normals=None): 
    with open(filepath, 'w') as file:
        vals = ''
        for i, point in enumerate(points[0]):
            point = point.data.cpu().numpy()
            vals += 'v ' + ' '.join([str(val) for val in point]) + '\n'
        if normals is not None:
            for i, normal in enumerate(normals[0]):
                normal = normal.data.cpu().numpy()
                vals += 'vn ' + ' '.join([str(val) for val in normal]) + '\n'
        if len(faces) > 0:
            for i, face in enumerate(faces[0]):
                face = face.data.cpu().numpy()
                vals += 'f ' + ' '.join([str(val+1) for val in face]) + '\n'
        file.write(vals)


    # if normals is not None:
    #     savemat(filepath.replace('obj','mat'),mdict={'vertices':points[0].data.cpu().numpy(),'faces':faces[0].data.cpu().numpy(),'normals':normals[0].data.cpu().numpy()})
    # else:
    #     savemat(filepath.replace('obj','mat'),mdict={'vertices':points[0].data.cpu().numpy(),'faces':faces[0].data.cpu().numpy()})
 
def save_to_obj2(filepath, points, faces, normals):
    # write new data
    with open(filepath, 'w') as file:
        vals = ''
        for i, point in enumerate(points):
            point = point.data.cpu().numpy()
            vals += 'v ' + ' '.join([str(val) for val in point]) + '\n'

        if normals is not None:
            for i, normal in enumerate(normals):
                normal = normal.data.cpu().numpy()
                vals += 'vn ' + ' '.join([str(val) for val in normal]) + '\n'

        if len(faces) > 0:
            for i, face in enumerate(faces['face-3']):
                face = face.data.cpu().numpy()
                vals += 'f ' + ' '.join([str(val+1) for val in face]) + '\n'

            for i, face in enumerate(faces['face-4']):
                if not torch.all(faces['face-4'] == -1):
                    face = face.data.cpu().numpy()
                    vals += 'f ' + ' '.join([str(val+1) for val in face]) + '\n'
        file.write(vals)

def save_to_ply(filepath, points, point_colors):
    # write new data
    # point_colors = colors
    # points = sample_points_
    with open(filepath, 'w') as file:
        vals = 'ply\n' \
               'format ascii 1.0\n'

        vals += 'element vertex {}\n'.format(len(points[0]))
        vals += 'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'end_header\n'

        points = points.data.cpu().numpy()
        point_colors = point_colors.data.cpu().numpy()
        for i, (point, color) in enumerate(zip(points[0],point_colors[0])):

            vals += '' + ' '.join([str(val) for val in point])
            vals += ' ' + ' '.join([str(np.uint8(255*val)) for val in color]) + '\n'
        vals += ' '
        # if len(faces) > 0:
        #     for i, face in enumerate(faces[0]):
        #         face = face.data.cpu().numpy()
        #         vals += 'f ' + ' '.join([str(val+1) for val in face]) + '\n'
        file.write(vals)

def plotvertices(fig, points):
    from mpl_toolkits.mplot3d import Axes3D
    ax = Axes3D(fig)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2])

def clean_border_pixels(image, gap):
    '''
    :param image:
    :param gap:
    :return:
    '''
    assert len(image.shape) == 3, "input should be 3 dim"

    D, H, W = image.shape
    y_ = image.clone()
    y_[:gap] = 0;
    y_[:, :gap] = 0;
    y_[:, :, :gap] = 0;
    y_[D - gap:] = 0;
    y_[:, H - gap] = 0;
    y_[:, :, W - gap] = 0;

    return y_

def voxel2mesh(volume, gap, shape):
    '''
    :param volume:
    :param gap:
    :param shape:
    :return:
    '''
    vertices_mc, faces_mc, _, _ = measure.marching_cubes_lewiner(volume.cpu().data.numpy(), 0, step_size=gap, allow_degenerate=False)
    vertices_mc = torch.flip(torch.from_numpy(vertices_mc), dims=[1]).float()  # convert z,y,x -> x, y, z
    vertices_mc = normalize_vertices(vertices_mc, shape)
    faces_mc = torch.from_numpy(faces_mc).long()
    return vertices_mc, faces_mc

def get_extremity_landmarks(surface_points):

    if isinstance(surface_points, torch.Tensor):
        if len(surface_points.shape) == 3:
            low_points = torch.gather(surface_points, dim=1, index=torch.argmin(surface_points, dim=1)[:, :, None].repeat(1, 1, 3))
            high_points = torch.gather(surface_points, dim=1, index=torch.argmax(surface_points, dim=1)[:, :, None].repeat(1, 1, 3))
            extreamities = torch.cat([low_points, high_points], dim=1)
        elif len(surface_points.shape) == 2:
            low_points = surface_points[torch.argmin(surface_points, dim=0)]
            high_points = surface_points[torch.argmax(surface_points, dim=0)]
            extreamities = torch.cat([low_points, high_points], dim=0)
        else:
            raise Exception('unsupported data dimension')
    else:
        raise Exception('unsupported data type')
    return extreamities


# def normalize_vertices(vertices, shape):
#     assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
#     assert shape.shape[0] == 1, "first dim of shape should be length 1"

#     return 2*(vertices/torch.max(shape) - 0.5)

def normalize_vertices(vertices, shape):
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2*(vertices/(torch.max(shape)-1) - 0.5)


def sample_outer_surface(volume, shape):
    # surface = F.max_pool3d(volume[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] - volume.float() # outer surface
    # surface = F.max_pool3d(-volume[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] + volume.float() # inner surface

    # inner surface
    # a = F.max_pool3d(-volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    # b = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    # c = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0] 
    # border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0) 
    # surface = border + volume.float() 

    # outer surface
    a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0] 
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0) 
    surface = border - volume.float()
 
    surface_points = torch.nonzero(surface)
    surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
    surface_points = normalize_vertices(surface_points, shape) 

    return surface_points

def sample_outer_surface_in_voxel(volume):
    # surface = F.max_pool3d(volume[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] - volume.float() # outer surface
    # surface = F.max_pool3d(-volume[None, None].float(), kernel_size=3, stride=1, padding=1)[0, 0] + volume.float() # inner surface

    # inner surface
    # a = F.max_pool3d(-volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    # b = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    # c = F.max_pool3d(-volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0] 
    # border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0) 
    # surface = border + volume.float() 

    # outer surface
    a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0] 
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0) 
    surface = border - volume.float()
    return surface.long()

# https://sites.google.com/site/pointcloudreconstruction/marching-cubes
# https://github.com/mmolero/pypoisson
# import numpy as np
# from pypoisson import poisson_reconstruction
# from utils.utils_mesh import read_obj
# filename = "/cvlabdata2/home/wickrama/projects/pypoisson/example/horse_with_normals.xyz"
# output_file = "/cvlabdata2/home/wickrama/projects/pypoisson/example/horse_reconstruction.ply"
# sphere_path='/cvlabsrc1/cvlab/datasets_udaranga/datasets/3d/mesh_templates/spheres/icosahedron_{}.obj'.format(162)
# sphere_path='/cvlabdata2/cvlab/datasets_udaranga/experiments/vmnet/Experiment_612/trial_1/best_performance3/mesh_162/testing_pred_0_part_0.obj' 
# points, sphere_faces, normals = read_obj(sphere_path)
# normals = points
# import time

# def points_normals_from(filename):
#     array = np.genfromtxt(filename)
#     return array[:,0:3], array[:,3:6]

# def ply_from_array(points, faces, output_file): 
#     num_points = len(points)
#     num_triangles = len(faces) 
#     header = 'ply'

# import time
# start = time.time()
# faces, vertices = poisson_reconstruction(points, normals, depth=10)
# end = time.time() - start




