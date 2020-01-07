import numpy as np

import sys
import os

import ellipsoid_intersection as ei
import bezier_curves as bzc


### Position is given in polar coordinates: Radius, Theta and Phi
CA_positions_mean = {
'L_Ostium':( 0.0,   0.0,  0.0),
    'LMm':( 0.7,  80.0,   4.0),
    'L1p':( 1.6+0.0*0.7,  70.0-1.5*12.0,  -3.0+0.0*12.0),
    'L1m':( 2.3+1.0*0.8,  63.0-1.5*10.0,  -5.0-0.5*11.0),
    'L2m':( 5.3,  44.0, -12.0),
    'L3m':(10.1,  30.0, -32.0),
    'L4p':(12.2,  32.0, -46.0),
    'L4m':(11.8,  33.0, -49.0),
    'D1o':( 3.2-1.0*0.8,  61.0+1.0*11.0,  -8.0-0.0*10.0),
    'D1m':( 6.2+0.0*1.0,  61.0+1.0*10.0, -23.0+0.0*11.0),
    'D2o':( 4.9+1.0*1.0,  47.0+1.0*10.0, -12.0+0.0* 7.0),
    'D2m':( 7.7+0.0*0.7,  49.0+1.0*10.0, -24.0+0.0* 8.0),
    'D3o':( 6.6+0.0*1.1,  41.0+1.0* 9.0, -17.0+0.0* 7.0),
    'D3m':( 9.0+0.0*0.9,  42.0+1.0*10.0, -26.0+0.0* 7.0),    
    'S1o':( 3.3-1.0*1.1,  57.0-2.0*11.0,  -9.0-1.0*10.0),
    'S1m':( 4.3-1.0*1.1,  35.0-3.0*12.0, -31.0-2.0*10.0),
    'S2o':( 5.2-0.1*1.1,  44.0-1.0*10.0, -12.0-0.0*10.0),
    'S2m':( 5.8+0.0*0.9,  31.0-1.0* 7.0, -29.0+1.5*10.0),
    'S3o':( 7.0+0.0*0.8,  37.0+0.0* 8.0, -19.0+0.0* 7.0),
    'S3m':( 7.3+0.0*0.8,  30.0+0.0* 7.0, -31.0+0.0* 8.0), 
    'C1p':( 1.4,  86.0, -12.0),
    'C1m':( 1.8,  94.0, -24.0),
    'C2m':( 3.1, 116.0, -37.0),
    'C3m':( 4.8, 135.0, -51.0),
    'C4m':( 5.9, 166.0, -71.0),
    'MRo':( 1.2+0.0*0.5,  85.0+0.0*18.0, -10.0+0.0*17.0),
    'MRm':( 5.6+0.0*1.6,  80.0+0.0*14.0, -34.0+0.0*13.0),
    'OMo':( 2.5, 104.0, -28.0),
    'OMb':( 4.4, 107.0, -37.0),
    'OMa':( 5.3, 102.0, -43.0),
    'OMp':( 4.9, 114.0, -43.0),    
    'M1o':( 2.2+0.0*0.7, 102.0+0.0*15.0, -31.0+0.0*10.0),
    'M1m':( 5.6-1.5*1.4,  91.0+0.7*15.0, -38.0+0.1* 9.0),
    'M2o':( 3.4+3.0*0.6, 118.0-0.5*18.0, -41.0-0.0* 8.0),
    'M2m':( 6.0+0.0*0.8, 109.0-0.0*20.0, -51.0+0.0* 8.0),
    'M3o':( 4.7+1.0*1.0, 131.0+1.0*20.0, -49.0+1.0* 8.0),
    'M3m':( 6.5+1.0*1.1, 118.0+1.0*27.0, -58.0+1.0*10.0),
    'R_Ostium':( 3.7, -25.0, -23.0),
    'R1p':( 0.5, -71.0, -22.0),
    'R1m':( 1.7, -68.0, -24.0),
    'R2m':( 4.6, -72.0, -48.0),
    'R3m':( 6.4, -91.0, -69.0),
    'R4m':( 6.2, 180.0, -66.0),
    'RDm':( 7.6,  80.0, -76.0),
    'RIm':( 7.6, 143.0, -68.0),
    'RPm':( 8.0, 145.0, -56.0)
}
            
CA_references = {
    'L_Ostium':'L_Ostium',
    'LMm':'L_Ostium',
    'L1p':'L_Ostium',
    'L1m':'L_Ostium',
    'L2m':'L_Ostium',
    'L3m':'L_Ostium',
    'L4p':'L_Ostium',
    'L4m':'L_Ostium',
    'D1o':'L_Ostium',
    'D1m':'L_Ostium',
    'D2o':'L_Ostium',
    'D2m':'L_Ostium',
    'D3o':'L_Ostium',
    'D3m':'L_Ostium',
    'S1o':'L_Ostium',
    'S1m':'L_Ostium',
    'S2o':'L_Ostium',
    'S2m':'L_Ostium',
    'S3o':'L_Ostium',
    'S3m':'L_Ostium',
    'C1p':'L_Ostium',
    'C1m':'L_Ostium',
    'C2m':'L_Ostium',
    'C3m':'L_Ostium',
    'C4m':'L_Ostium',
    'MRo':'L_Ostium',
    'MRm':'L_Ostium',
    'OMo':'L_Ostium',
    'OMb':'L_Ostium',
    'OMa':'L_Ostium',
    'OMp':'L_Ostium',
    'M1o':'L_Ostium',
    'M1m':'L_Ostium',
    'M2o':'L_Ostium',
    'M2m':'L_Ostium',
    'M3o':'L_Ostium',
    'M3m':'L_Ostium',
    'R_Ostium':'L_Ostium',
    'R1p':'R_Ostium',
    'R1m':'R_Ostium',
    'R2m':'R_Ostium',
    'R3m':'R_Ostium',
    'R4m':'R_Ostium',
    'RDm':'R_Ostium',
    'RIm':'R_Ostium',
    'RPm':'R_Ostium'
}


            
CA_positions_stddev ={
    'L_Ostium':(0.0, 0.0, 0.0),
    'LMm':(0.4,20.0,15.0),
    'L1p':(0.7,12.0,12.0),
    'L1m':(0.8,10.0,11.0),
    'L2m':(0.8, 7.0, 9.0),
    'L3m':(0.9, 8.0, 7.0),
    'L4p':(1.0, 9.0, 7.0),
    'L4m':(0.9,10.0, 7.0),
    'D1o':(0.8,11.0,10.0),
    'D1m':(1.0, 9.0,11.0),
    'D2o':(1.0,10.0, 7.0),
    'D2m':(0.7,10.0, 8.0),
    'D3o':(1.1, 9.0, 7.0),
    'D3m':(0.9,10.0, 7.0),
    'S1o':(1.1,11.0,10.0),
    'S1m':(1.1,12.0,10.0),
    'S2o':(1.1,10.0,10.0),
    'S2m':(0.9, 7.0,10.0),
    'S3o':(0.8, 8.0, 7.0),
    'S3m':(0.8, 7.0, 8.0),
    'C1p':(0.7,15.0,13.0),
    'C1m':(0.8,17.0,12.0),
    'C2m':(0.9,18.0, 9.0),
    'C3m':(1.0,21.0, 8.0),
    'C4m':(0.9,39.0, 9.0),
    'MRo':(0.5,18.0,17.0),
    'MRm':(1.6,14.0,13.0),
    'OMo':(1.0,11.0, 9.0),
    'OMb':(0.9, 6.0, 9.0),
    'OMa':(0.6, 6.0, 8.0),
    'OMp':(0.7, 5.0, 7.0),
    'M1o':(0.7,15.0,10.0),
    'M1m':(1.4,15.0, 9.0),
    'M2o':(0.6,18.0, 8.0),
    'M2m':(0.8,20.0, 8.0),
    'M3o':(1.0,20.0, 8.0),
    'M3m':(1.1,27.0,10.0),
    'R_Ostium':(0.0, 0.0, 0.0),
    'R1p':(0.1,19.0,17.0),
    'R1m':(0.4,14.0,11.0),
    'R2m':(0.7,11.0, 8.0),
    'R3m':(1.1,64.0, 5.0),
    'R4m':(0.8,19.0, 8.0),
    'RDm':(1.1,60.0, 8.0),
    'RIm':(1.0,25.0, 9.0),
    'RPm':(0.8,21.0, 9.0),
}
            
CA_diameter_mean ={
    'L_Ostium':4.5,
    'LMm':4.5,
    'L1p':3.7,
    'L1m':3.6,
    'L2m':2.5,
    'L3m':1.7,
    'L4p':1.4,
    'L4m':1.1,
    'D1o':2.1,
    'D1m':1.5,
    'D2o':1.9,
    'D2m':1.4,
    'D3o':1.7,
    'D3m':1.3,
    'S1o':1.4,
    'S1m':1.0,
    'S2o':1.1,
    'S2m':0.9,
    'S3o':1.1,
    'S3m':0.9,
    'C1p':2.9,
    'C1m':2.9,
    'C2m':3.1,
    'C3m':1.4,
    'C4m':0.0,
    'MRo':1.8,
    'MRm':1.3,
    'OMo':0.0,
    'OMb':0.0,
    'OMa':0.0,
    'OMp':3.3,
    'M1o':2.1,
    'M1m':1.5,
    'M2o':2.0,
    'M2m':1.4,
    'M3o':1.7,
    'M3m':1.4,
    'R_Ostium':4.0,
    'R1p':4.0,
    'R1m':3.9,
    'R2m':3.4,
    'R3m':3.1,
    'R4m':2.2,
    'RDm':2.0,
    'RIm':1.4,
    'RPm':1.4,
}
            
CA_radius_stddev ={
    'LMm':0.5,
    'L1p':0.5,
    'L1m':0.5,
    'L2m':0.5,
    'L3m':0.5,
    'L4p':0.5,
    'L4m':0.4,
    'D1o':0.5,
    'D1m':0.3,
    'D2o':0.4,
    'D2m':0.2,
    'D3o':0.3,
    'D3m':0.2,
    'S1o':0.2,
    'S1m':0.2,
    'S2o':0.3,
    'S2m':0.2,
    'S3o':0.3,
    'S3m':0.2,
    'C1p':0.5,
    'C1m':0.5,
    'C2m':0.6,
    'C3m':0.6,
    'C4m':0.0,
    'MRo':0.2,
    'MRm':0.2,
    'OMo':0.0,
    'OMb':0.0,
    'OMa':0.0,
    'OMp':0.2,
    'M1o':0.5,
    'M1m':0.4,
    'M2o':0.4,
    'M2m':0.3,
    'M3o':0.2,
    'M3m':0.3,
    'R1p':0.6,
    'R1m':0.6,
    'R2m':0.5,
    'R3m':0.5,
    'R4m':0.5,
    'RDm':0.3,
    'RIm':0.4,
    'RPm':0.2,
}

### Right coronary artery dominant
ca_relations_RD = {
    'L_Ostium':['R_Ostium', 'LMm'],
    'LMm':['L1p', 'C1p', 'D1o'],
    'L1p':['L1m', 'S1o'],
    'L1m':['L2m', 'D2o'],
    'D1o':['D1m'],
    'D1m':[],
    'S1o':['S1m'],
    'S1m':[],
    'L2m':['D3o','S2o', 'L3m'],
    'S2o':['S2m'],
    'S2m':[],
    'D2o':['D2m'],
    'D2m':[],
    'D3o':['D3m'],
    'D3m':[],
    'L3m':['L4p'],
    'L4p':['L4m'],
    'L4m':[],
    'C1p':['C1m', 'MRo'],
    'C1m':['C2m', 'M1o'],
    'MRo':['MRm'],
    'MRm':[],
    'C2m':['C3m'],
    'C3m':[],
    'M1o':['M1m'],
    'M1m':['M2o'],
    'M2o':['M2m'],
    'M2m':[],
    'R_Ostium':['R1p'],
    'R1p':['R1m'],
    'R1m':['R2m'],
    'R2m':['R3m'],
    'R3m':['RDm', 'R4m'],
    'RDm':[],
    'R4m':['RIm', 'RPm'],
    'RIm':[],
    'RPm':[]
}

ca_left_artery_RD = ['L_Ostium', 'LMm','L1p','L1m','D1o','D1m','S1o','S1m','L2m','S2o','S2m','D2o','D2m','D3o','D3m','L3m','L4p','L4m','C1p','C1m','MRo','MRm','C2m','C3m','M1o','M1m','M2o','M2m']
ca_right_artery_RD = ['R_Ostium', 'R1p', 'R1m', 'R2m', 'R3m', 'RDm', 'R4m', 'RIm', 'RPm']

ca_circunflex_RD = [
    'LMm', 'C1p', 'C1m', 'C2m', 'C3m',
    'R1p', 'R1m', 'R2m', 'R3m', 'R4m',
]

ca_distal_RD = 'C3m'

def allignation_mat(src, trg):
    src = src / np.linalg.norm(src)
    trg = trg / np.linalg.norm(trg)
    
    c_theta = np.dot(src, trg)
    if np.abs(c_theta + 1.0) < 1e-6:
        return -np.eye(3)
    
    v = np.cross(src, trg)
    skew_mat = np.array([[0.0, -v[2], v[1]],[v[2], 0.0, -v[0]],[-v[1], v[0], 0.0]])
    
    return np.eye(3) + skew_mat + 1.0/(1.0+c_theta) * np.matmul(skew_mat,skew_mat)


def ang2pos(angles, origin = (0.0, 0.0, 0.0)):
    radius, theta, phi = angles
    theta = np.pi * theta / 180.0
    phi = np.pi * phi / 180.0
    
    position = radius * np.array((np.cos(phi)*np.sin(theta), np.sin(phi), np.cos(phi)*np.cos(theta)))
    
    return position + origin


def project_segments(position, ellipsoid_model_pars):
    ### Projection of the position to the ellipsoid surface:
    t = np.sqrt(1./np.sum(position**2 / (ellipsoid_model_pars[0]**2, ellipsoid_model_pars[0]**2, ellipsoid_model_pars[1]**2)))
    return t*position


class ArterySegment():
    def __init__(self, segment_label, position, ellipsoid_model_pars, radius,
            parent_position=None, parent_radius=None,
            segment_points_density = 500, radial_resolution = 10,
            rotate_matrix=None, center_vector=None):
        self._segment_label = segment_label  
        self._parent_radius = parent_radius
        self._radius = radius
        self._position = position

        self._intersection_plane = None
        self._ellipsoid_pars = ellipsoid_model_pars
        self._ellipsoid = ei.EllipsoidIntersection(ellipsoid_model_pars[0], ellipsoid_model_pars[1], segment_points_density = segment_points_density)
        
        self._segment_arc_radius = None
        self._segment_arc_circunferences = self._position[np.newaxis,...]
        self._segment_arc_relations = None
        self._segment_arc_positions = self._position[np.newaxis,...]
        self._segment_arc_normals = None

        self._projected_points = None
        self._visible_relations = None
        self._mapping_visible_idx = None
        self._visible_points_idx = None

        self._radial_resolution = radial_resolution
        self._radius_scale = 1.0
        self._diam_narrowing_model = None

        self._boundary_box = None
        self._boundary_box_relations = None
        
        self._children = []

        if parent_position is not None:
            self._compute_ellipsoid_intersection(parent_position)
            self._compute_arc_circunferences()
            self._compute_arc_relations()
            self._compute_boundary_box()

            if rotate_matrix is not None:            
                self._rotate_points(rotate_matrix)
            
            if center_vector is not None:
                self._center_points(center_vector)
        
        else:
            self._segment_arc_relations = (0,0,0,0)
        
    
    def _rotate_points(self, rotate_matrix):
        self._segment_arc_circunferences = np.matmul(self._segment_arc_circunferences, rotate_matrix)
        self._segment_arc_positions = np.matmul(self._segment_arc_positions, rotate_matrix)
        self._segment_arc_normals = np.matmul(self._segment_arc_normals, rotate_matrix)


    def _center_points(self, center_vector):
        self._segment_arc_circunferences = self._segment_arc_circunferences + center_vector
        self._segment_arc_positions = self._segment_arc_positions + center_vector


    def _compute_ellipsoid_intersection(self, parent_position):
        self._intersection_plane = np.cross(self._position, parent_position)
        self._intersection_plane = self._intersection_plane / np.linalg.norm(self._intersection_plane)
        d = np.dot(self._intersection_plane, self._position)
        
        self._ellipsoid.compute_intersection(self._intersection_plane, d)
        self._ellipsoid.compute_plane_arc(self._position, parent_position)
        self._segment_arc_positions = self._ellipsoid.get_arc()
        self._n_arc_points = self._segment_arc_positions.shape[0]

        ### Assign to each point of the arc a radius:
        self._segment_arc_radius = np.linspace(self._radius, self._parent_radius, self._n_arc_points)
        self._radius_scale = np.ones(self._n_arc_points)
        self._diam_narrowing_model = bzc.BSpline(knots=int(np.min((5.0, self._n_arc_points))))


    def _compute_arc_relations(self):
        ### Compute the relation that defines the polygons (quads) of the segment
        self._segment_arc_relations = []
        self._segment_arc_normals = []
   
        quad_relations_base = np.zeros([self._radial_resolution,4], dtype=np.int64)
        for i in range(self._radial_resolution):
            quad_relations_base[i,:] = np.array((self._radial_resolution+i, self._radial_resolution+(i-1)%self._radial_resolution, (i-1)%self._radial_resolution, i), dtype=np.int64)
        
        increment_idx = np.array([[[i, i, i, i] for j in range(self._radial_resolution)] for i in range(self._n_arc_points-1)]).reshape(-1, 4)
        increment_idx = increment_idx * self._radial_resolution + 1
        self._segment_arc_relations = np.tile(quad_relations_base, (self._n_arc_points-1,1)) + increment_idx

        ### Compute the face normal vectors:
        v1_temp = self._segment_arc_circunferences[self._segment_arc_relations[:,0],...] - self._segment_arc_circunferences[self._segment_arc_relations[:,3],...]
        v2_temp = self._segment_arc_circunferences[self._segment_arc_relations[:,1],...] - self._segment_arc_circunferences[self._segment_arc_relations[:,3],...]
        self._segment_arc_normals = np.cross(v1_temp, v2_temp)
        self._segment_arc_normals = self._segment_arc_normals / np.linalg.norm(self._segment_arc_normals, axis=1)[...,np.newaxis]

        for i in range(self._radial_resolution):
            ### Staring face
            triangle_relation = np.array((i, -1, (i-1)%self._radial_resolution, i), dtype=np.int64) + 1
            v1_temp = self._segment_arc_circunferences[triangle_relation[0],...] - self._segment_arc_circunferences[triangle_relation[1],...]
            v2_temp = self._segment_arc_circunferences[triangle_relation[2],...] - self._segment_arc_circunferences[triangle_relation[1],...]
            v3_temp = self._segment_arc_circunferences[triangle_relation[2],...] - self._segment_arc_circunferences[triangle_relation[0],...]
            
            triangle_normal = np.cross(v1_temp, v2_temp)
            triangle_normal_length = np.linalg.norm(triangle_normal)
            if triangle_normal_length < 1e-4:
                triangle_normal = np.cross(v1_temp, v3_temp)
                triangle_normal_length = np.linalg.norm(triangle_normal)
                print(self._segment_label, triangle_relation)
                print(self._segment_arc_circunferences[triangle_relation[0]])
                print(self._segment_arc_circunferences[triangle_relation[1]])
                print(self._segment_arc_circunferences[triangle_relation[2]])
                print('Coplanar triange, fixed has norm:', triangle_normal_length)
                print('V1', v1_temp)
                print('V2', v2_temp)
                print('V3', v3_temp)
            
            triangle_normal = triangle_normal / triangle_normal_length

            self._segment_arc_normals = np.vstack((self._segment_arc_normals, triangle_normal))
            self._segment_arc_relations = np.vstack((self._segment_arc_relations, triangle_relation))

            ### Ending face
            triangle_relation = np.array((i, self._radial_resolution, (i-1)%self._radial_resolution, i), dtype=np.int64) + self._segment_arc_circunferences.shape[0] - 1 - self._radial_resolution
            v1_temp = self._segment_arc_circunferences[triangle_relation[0],...] - self._segment_arc_circunferences[triangle_relation[1],...]
            v2_temp = self._segment_arc_circunferences[triangle_relation[2],...] - self._segment_arc_circunferences[triangle_relation[1],...]
            triangle_normal = np.cross(v2_temp, v1_temp)
            triangle_normal = triangle_normal / np.linalg.norm(triangle_normal)
            self._segment_arc_normals = np.vstack((self._segment_arc_normals, triangle_normal))
            self._segment_arc_relations = np.vstack((self._segment_arc_relations, triangle_relation))


    def _compute_boundary_box(self):
        ### Compute the boundary box, intended to speed up the rendering process
        center_point = (np.max(self._segment_arc_circunferences) + np.min(self._segment_arc_circunferences))/2.
        centered_points = self._segment_arc_circunferences - center_point
        
        A_mat = np.matmul(centered_points.T, centered_points)
        l, w = np.linalg.eig(A_mat)
        egv_order = np.argsort(l)

        ### Project the points into each vector
        P = np.matmul(centered_points, w)
        P_max = P.max(axis=0)
        P_min = P.min(axis=0)

        self._boundary_box = [
            center_point + P_max[0]*w[:,0] + P_max[1]*w[:,1] + P_max[2]*w[:,2],
            center_point + P_max[0]*w[:,0] + P_max[1]*w[:,1] + P_min[2]*w[:,2], 
            center_point + P_max[0]*w[:,0] + P_min[1]*w[:,1] + P_max[2]*w[:,2],    
            center_point + P_min[0]*w[:,0] + P_max[1]*w[:,1] + P_max[2]*w[:,2],    
            center_point + P_max[0]*w[:,0] + P_min[1]*w[:,1] + P_min[2]*w[:,2],   
            center_point + P_min[0]*w[:,0] + P_min[1]*w[:,1] + P_max[2]*w[:,2],    
            center_point + P_min[0]*w[:,0] + P_max[1]*w[:,1] + P_min[2]*w[:,2],    
            center_point + P_min[0]*w[:,0] + P_min[1]*w[:,1] + P_min[2]*w[:,2], 
        ]

        self._boundary_box_relations = [
            (4,1,0,2), (0,2,5,3), (5,3,6,7),
            (6,7,4,1), (2,4,7,5), (3,0,1,6),
        ]


    def _compute_arc_circunferences(self):
        ### Compute a perpendicular direction to the tangent evaluated in each position
        pos_dir = self._segment_arc_positions / np.array((self._ellipsoid_pars[0]**2, self._ellipsoid_pars[0]**2, self._ellipsoid_pars[1]**2))
        pos_dir = pos_dir / np.linalg.norm(pos_dir, axis=1)[...,np.newaxis]

        ort_dir = np.cross(pos_dir[0,:], pos_dir[1,:])
        ort_dir =  ort_dir / np.linalg.norm(ort_dir)

        theta = np.arange(0.0, 2.0*np.pi, 2.0*np.pi/self._radial_resolution)
        ctheta = np.cos(theta)
        stheta = np.sin(theta)

        self._segment_arc_circunferences = self._segment_arc_positions[0]
        for m in range(self._n_arc_points):
            current_circunference = self._segment_arc_radius[m] * self._radius_scale[m] * (np.outer(ctheta, ort_dir) + np.outer(stheta, pos_dir[m,:])) + self._segment_arc_positions[m]
            self._segment_arc_circunferences = np.vstack((self._segment_arc_circunferences, current_circunference))

        ### add the fisrt and las segment position, it allows to construct faces that close the segment surfce 
        self._segment_arc_circunferences = np.vstack((self._segment_arc_circunferences, self._segment_arc_positions[-1]))


    def _check_occlusions(self, projection_plane_N):
        cos_angle = np.dot(self._segment_arc_normals, projection_plane_N)
        self._visible_relations = cos_angle > 0.0

        self._mapping_visible_idx = np.arange(self._segment_arc_circunferences.shape[0])
        
        ### Compute the indices of the points that are visible, and 
        self._visible_points_idx = np.unique(self._segment_arc_relations[self._visible_relations])
        del_idx = np.setdiff1d(self._mapping_visible_idx, self._visible_points_idx)
        adj_idx = np.zeros(self._mapping_visible_idx.size, dtype=np.int64)

        adj_idx[del_idx] = 1
        adj_idx = np.cumsum(adj_idx)
        self._mapping_visible_idx = self._mapping_visible_idx - adj_idx


    def get_label(self):
        return self._segment_label
        
    
    def get_positions(self):
        return self._position
        
    
    def get_arc(self):
        return self._segment_arc_circunferences, self._segment_arc_relations, self._segment_arc_normals


    def get_boundary_box(self):
        return self._boundary_box, self._boundary_box_relations


    def get_segment_children(self):
        return self._children


    def assign_child(self, child_label):
        self._children.append(child_label)
        

    def project_arc_circunferences(self, projection_plane_N, projection_plane_d, projection_plane_reference):
        self._check_occlusions(projection_plane_N)
        
        projection_relations = self._mapping_visible_idx[self._segment_arc_relations[self._visible_relations]]
        projected_points = np.zeros([self._visible_points_idx.size, 3])

        for i, point_3d in enumerate(self._segment_arc_circunferences[self._visible_points_idx]):
            vector2plane = point_3d - projection_plane_reference
            point_2d = np.dot(projection_plane_N, vector2plane)*projection_plane_N
            point_2d = vector2plane - point_2d
            projected_points[i] = point_2d

        return projected_points + projection_plane_reference, projection_relations


    def narrow_diameter(self, position, length, narrowing_percentage):
        x = np.linspace(0., 1., self._n_arc_points)
        y = np.zeros(self._n_arc_points)

        y[position:(position+length+1)] = 1.0
        narrowing_sim = self._diam_narrowing_model.fit(np.column_stack((x, y)), self._n_arc_points)

        # The scale is adjusted between 0 and 1
        narrowing_sim[:,1] = narrowing_sim[:,1] / np.max(narrowing_sim[:,1])
        self._radius_scale = 1. - narrowing_sim[:,1] * narrowing_percentage

        ### Recompute the segment circunferences with the narrowed diameter
        self._compute_arc_circunferences()
        print('narrowing scaling:', self._radius_scale)



class ArteryModel():
    def __init__(self, segment_points_density=500, radial_resolution=10, random_seed=None, random_positions=False):
        self._ellipsoid = None
        self._circunflex_plane_model = None
        self._ellipsoid_model = None
        self._artery_origin_refs = None
        self._artery_tree_dict = {}

        self._random_seed = random_seed
        self._random_positions = random_positions

        ### Initalization methods
        self._compute_circunflex_plane()
        self._compute_ellipsoid_model()
        self._compute_artery_tree(segment_points_density, radial_resolution)


    
    ### Compute plane where the circunflex plane lays
    def _compute_circunflex_plane(self):
        l_ostium_pos = (0.0, 0.0, 0.0)
        r_ostium_pos = ang2pos(CA_positions_mean['R_Ostium'], l_ostium_pos)
        
        self._artery_origin_refs = {'L_Ostium':l_ostium_pos, 'R_Ostium':r_ostium_pos}
        
        cfx_points = np.zeros([len(ca_circunflex_RD), 3])
        for cfx_i, cfx_label in enumerate(ca_circunflex_RD):
            angles = np.array(CA_positions_mean[cfx_label])
            cfx_points[cfx_i, :] = ang2pos(angles, origin=self._artery_origin_refs[CA_references[cfx_label]])
        
        ### Find the center point considering all points (not averaging)
        center_point = (np.max(cfx_points,axis=0) + np.min(cfx_points,axis=0))/2.0
        
        centered_points = cfx_points - center_point

        radius = np.mean(np.sqrt(np.sum(centered_points*centered_points, axis=1)))

        A_mat = np.matmul(centered_points.T, centered_points)
        l, w = np.linalg.eig(A_mat)
        l_min = np.argmin(l)
        d = np.dot(w[:,l_min], center_point)
        
        ### Ellipsoid heart model values were taken from: Gupta et al., A morphometric study of measurements of heart in adults and its relation with age and height of the individual: A post-mortem study. 2014
        reference_point = w[:,l_min] * 8.7 * (radius / 4.32)
        
        self._circunflex_plane_model = (w[:,l_min], d, center_point, reference_point, radius)
        
    
    def _compute_ellipsoid_model(self):
        N, d, circunflex_center, reference_point, e_a = self._circunflex_plane_model
        
        ### Rotate the plane normal according to the origin cannonic direction
        reference_origin = (0.0, 0.0, 1.0)
        A_mat = allignation_mat(N, reference_origin)
        reference_point = np.matmul(A_mat, reference_point)

        e_c = np.sqrt(reference_point[2]**2/(1.0 - np.sum(reference_point[:2]**2)/e_a**2))

        self._ellipsoid = ei.EllipsoidIntersection(e_a, e_c, segment_points_density=1)

        ### Radius is taken as parameter a for the ellipsoid model
        self._ellipsoid_model = (e_a, e_c, A_mat)
        
    
    ### Add segments recursively
    def _add_segment(self, current_label, parent_position, parent_radius, segment_points_density, radial_resolution):
        ### Compute the new segment position
        angles = np.array(CA_positions_mean[current_label]) + 0.0*np.array(CA_positions_stddev[current_label])
        position = ang2pos(angles, origin=self._artery_origin_refs[CA_references[current_label]])
        
        ### Center points at origin
        position = position - self._circunflex_plane_model[2]
        
        ### Allign the points to the reference vector
        position = np.dot(self._ellipsoid_model[2], position)
        
        ### Project the point to the surface of the ellipsoid
        position = project_segments(position, self._ellipsoid_model)
        
        if self._random_positions:
            theta, phi = self._ellipsoid.compute_angles(position)
            new_theta = theta + (np.random.random() * 2.0 - 1.0) * np.pi * 3e-2
            new_phi = phi + (np.random.random() * 2.0 - 1.0) * np.pi * 1e-3
            position = self._ellipsoid.get_position_ellipsoid(new_theta, new_phi)

        ### Generate a new artery segment with all the defined points
        current_radius = CA_diameter_mean[current_label] / 2.0 * 0.1 # To convert to cm from mm
        self._artery_tree_dict[current_label] = ArterySegment(current_label,
            position, self._ellipsoid_model, radius=current_radius,
            parent_position=parent_position, parent_radius=parent_radius,
            segment_points_density=segment_points_density,
            radial_resolution=radial_resolution, rotate_matrix=self._ellipsoid_model[2], center_vector=self._circunflex_plane_model[2])
        
        for child_label in ca_relations_RD[current_label]:
            self._add_segment(child_label, position, current_radius, segment_points_density, radial_resolution)
            self._artery_tree_dict[current_label].assign_child(child_label)

        
    def _compute_artery_tree(self, segment_points_density, radial_resolution):
        np.random.seed(self._random_seed)
        self._add_segment('L_Ostium', None, None, segment_points_density, radial_resolution)
    
    
    def get_artery_segment(self, segment_label):
        return self._artery_tree_dict[segment_label]


    def get_artery_tree(self):
        return self._artery_tree_dict


    def get_ellipsoid_model_pars(self):
        return self._ellipsoid_model


    def get_circunflex_model(self):
        return self._circunflex_plane_model