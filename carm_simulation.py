import numpy as np
import math
import cv2

import sys
import os

from synthetic_artery import ca_left_artery_RD, ca_right_artery_RD

class CArm():
    def __init__(self, resolution_x = 512, resolution_y = 512):
        self._distance_source_patient = 1.0
        self._distance_source_detector = 2.0

        self._lao_rao_angle = 0.0
        self._cra_cau_angle = 0.0
        self._displacement_vector = np.array((0.0, 0.0, 0.0))
        self._rotation_mat = None

        self._projection_plane_normal = np.array((0.0, 1.0, 0.0))
        self._projection_plane_d = 0.0

        self._source_ref = np.array((0.0, 0.0, 0.0))
        self._detector_ref = np.array((0.0, 0.0, 0.0))
        
        self._pix_x = 0.0308
        self._pix_y = 0.0308

        self._projection_image = np.zeros([resolution_x, resolution_y], dtype=np.int32)
        
        self._update_scene()

    def _update_scene(self):
        clr = np.cos(self._lao_rao_angle)
        slr = np.sin(self._lao_rao_angle)

        ccc = np.cos(self._cra_cau_angle)
        scc = np.sin(self._cra_cau_angle)

        rot_mat_laorao = np.array([[clr, slr, 0.0],[-slr, clr, 0.0],[0.0, 0.0, 1.0]])
        rot_mat_cracau = np.array([[1.0, 0.0, 0.0], [0.0, ccc, scc],[ 0.0,-scc, ccc]])

        self._rotation_mat = np.matmul(rot_mat_laorao, rot_mat_cracau)
        self._projection_plane_normal = np.dot(np.array((0.0, 1.0, 0.0)), self._rotation_mat)

        self._source_ref = self._distance_source_patient * np.dot(np.array((0.0, 1.0, 0.0)), self._rotation_mat)
        self._source_ref = self._source_ref + self._displacement_vector

        self._detector_ref = self._source_ref - self._distance_source_detector * self._projection_plane_normal
        self._projection_plane_d = np.dot(self._detector_ref, self._projection_plane_normal)
        print('Projection plane:', self._projection_plane_normal, ', d:', self._projection_plane_d)

    def set_pixdim(self, pix_x, pix_y):
        self._pix_x = pix_x
        self._pix_y = pix_y

    def set_src2det(self, dst_src2det):
        self._distance_source_detector = dst_src2det
        self._update_scene()

    def set_src2pat(self, dst_src2pat):
        self._distance_source_patient = dst_src2pat
        self._update_scene()

    def rot_laorao(self, laorao_angle):
        self._lao_rao_angle = laorao_angle
        self._update_scene()

    def rot_cracau(self, cracau_angle):
        self._cra_cau_angle = cracau_angle
        self._update_scene()

    def move(self, x=0.0, y=0.0, z=0.0):
        self._displacement_vector = np.array((x, y, z))
        self._update_scene()

    def get_projection_plane(self):
        return self._projection_plane_normal, self._projection_plane_d, self._detector_ref

    def get_rotation_matrix(self):
        return self._rotation_mat

    def get_source_position(self):
        return self._source_ref

    def project_artery(self, artery_model, artery_sub_tree='both'):
        ### Define the artery sub tree (left or right) to be displayed
        if artery_sub_tree=='left':
            avoid_sub_tree = ca_right_artery_RD

        elif artery_sub_tree=='right':
            avoid_sub_tree = ca_left_artery_RD

        else:
            avoid_sub_tree = []
        
        rows, cols = self._projection_image.shape
        self._projection_image[:] = 0
        scale_pixel = 1./np.array((self._pix_x, 1.0, self._pix_y))
        center_point = np.array((cols/2.0, 0.0, rows/2.0))
        artery_tree_dict = artery_model.get_artery_tree()
        for current_label in artery_tree_dict.keys():
            if current_label == 'L_Ostium': continue
            if current_label in avoid_sub_tree: continue

            current_segment = artery_tree_dict[current_label]
            projected_points, projected_relations = current_segment.project_arc_circunferences(self._projection_plane_normal, self._projection_plane_d, self._detector_ref)
            ### Scale projected points according to the size of a pixel:
            projected_points = np.dot(projected_points, self._rotation_mat.T)
            projected_points = projected_points * scale_pixel
            projected_points = projected_points + center_point[np.newaxis,...]
            
            projected_points[:,1] = projected_points[:,2]

            vertices = np.floor(projected_points[projected_relations,:2]).astype(np.int32)

            for rect in vertices:
                cv2.fillConvexPoly(self._projection_image, rect, 255)
    
        return self._projection_image