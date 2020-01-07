import numpy as np
import matplotlib.pyplot as plt
import vtk

import synthetic_artery as syn
import carm_simulation as carm

import sys
import os
import argparse

def draw_sphere(center, renderer, radii=0.05, color='Tomato', renderer_colors=None):
    # create a Sphere"
    sphereSource = vtk.vtkSphereSource()
    sphereSource.SetCenter(center)
    sphereSource.SetRadius(radii)

    # create a mapper
    sphereMapper = vtk.vtkPolyDataMapper()
    sphereMapper.SetInputConnection(sphereSource.GetOutputPort())

    # create an actor
    sphereActor = vtk.vtkActor()
    sphereActor.SetMapper(sphereMapper)
    if renderer_colors is not None:
        sphereActor.GetProperty().SetColor(renderer_colors.GetColor3d(color))

    # add the actors to the scene
    renderer.AddActor(sphereActor)

def render_carm(carm_model, renderer=None, display_render=True):
    projection_plane_normal, projection_plane_d, detector_ref = carm_model.get_projection_plane()
    rot_mat = carm_model.get_rotation_matrix()
    src_position = carm_model.get_source_position()

    print('C-Arm model:', src_position, ',', detector_ref)
    colors = vtk.vtkNamedColors()

    if renderer is None:
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95,0.95,1.0)
    
    draw_sphere(detector_ref, renderer, radii=1.0, color='Tomato', renderer_colors=colors)
    draw_sphere(src_position, renderer, radii=1.0, color='Banana', renderer_colors=colors)

    if display_render:
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName('3D ellipse')
        renderWindow.AddRenderer(renderer)
        
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderer.ResetCamera()
        renderWindow.Render()
        
        renderWindowInteractor.Start()

    return renderer

def render_artery(artery_model, renderer=None, display_render=True):
    colors = vtk.vtkNamedColors()

    if renderer is None:
        renderer = vtk.vtkRenderer()
        renderer.SetBackground(0.95,0.95,1.0)

    pointsArtery = vtk.vtkPoints()
    ugridArtery = vtk.vtkUnstructuredGrid()
    ugridArtery.Allocate(500)

    artery_tree = artery_model.get_artery_tree()
    t_i=0
    for segment_label in artery_tree.keys():
        if (segment_label == 'L_Ostium'): continue
        segment = artery_model.get_artery_segment(segment_label)
        segment_arc_pos, segment_arc_rel, _ = segment.get_arc()
        segment_arc_rel = np.array(segment_arc_rel,dtype=np.int64) + t_i

        for relation in segment_arc_rel:
            ugridArtery.InsertNextCell(vtk.VTK_QUAD, 4, relation)

        for curr_position in segment_arc_pos:
            pointsArtery.InsertPoint(t_i, curr_position)
            t_i+=1
        
    ugridArtery.SetPoints(pointsArtery)
    ugridMapperArtery = vtk.vtkDataSetMapper()
    ugridMapperArtery.SetInputData(ugridArtery)

    ugridActorArtery = vtk.vtkActor()
    ugridActorArtery.SetMapper(ugridMapperArtery)
    ugridActorArtery.GetProperty().SetColor(colors.GetColor3d('Red'))
    ugridActorArtery.GetProperty().EdgeVisibilityOn()
    
    renderer.AddActor(ugridActorArtery)

    if display_render:
        renderWindow = vtk.vtkRenderWindow()
        renderWindow.SetWindowName('3D ellipse')
        renderWindow.AddRenderer(renderer)
        
        renderWindowInteractor = vtk.vtkRenderWindowInteractor()
        renderWindowInteractor.SetRenderWindow(renderWindow)

        renderer.ResetCamera()
        renderWindow.Render()
        
        renderWindowInteractor.Start()

    return renderer



def main():
    carm_model = carm.CArm(resolution_x = 512, resolution_y = 512)
    carm_model.set_src2det(111.2)
    carm_model.set_src2pat(87.61815646)
    carm_model.rot_laorao(44.9*np.pi/180.0)
    carm_model.rot_cracau(-0.1*np.pi/180.0)


    artery_model = syn.ArteryModel(segment_points_density=500, radial_resolution=10, random_seed=None, random_positions=True)
    artery_tree = artery_model.get_artery_tree()

    artery_renderer = render_artery(artery_model, display_render=False)
    render_carm(carm_model, artery_renderer)
   
    plt.subplot(1,2,1)
    projection_img = carm_model.project_artery(artery_model, artery_sub_tree='left')
    plt.imshow(projection_img, 'gray')
    plt.subplot(1,2,2)
    projection_img = carm_model.project_artery(artery_model, artery_sub_tree='right')
    plt.imshow(projection_img, 'gray')
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Synthetic angiogram generator.')
    main()