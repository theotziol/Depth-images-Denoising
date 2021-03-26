import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy as dc

#removes points that have few neighbors in a given sphere around them
path1 = 'D:\Documents\Master Thesis\data\scan_7_solderinh_2\debug_calibration_laser_cloud.ply'
path = o3d.io.read_point_cloud(path1)

def radius_filter(cloud, points=3, radius=4, ):
    cl, ind = cloud.remove_radius_outlier(nb_points=points, radius=radius)
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud], width=1200, height=800)
    print(np.asarray(outlier_cloud.points), "\n")
    return inlier_cloud

def filter(cloud, distance_threshold =1,ransac_n=30,num_iterations=1000):
    #filter negative z axis values 
    pcd = dc(cloud)
    plane_model, inliers = pcd.segment_plane(distance_threshold,
                                             ransac_n,
                                             num_iterations)
    [a, b, c, d] = plane_model
    print(f"Plane equation: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
    
    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])
    outlier_cloud = pcd.select_by_index(inliers, invert=True)
    array = np.asarray(inlier_cloud.points)
    zmax = np.max(array[:,2])
    zmin = np.min(array[:,2])
    average = np.average(array[:,2])
    limit = (zmax + average)/2
    print("average =", average)
    print(array, zmax, zmin)
    o3d.visualization.draw_geometries([inlier_cloud,outlier_cloud],width=1280,height=720)
    array2 = dc(np.asarray(pcd.points))
    for i in range(len(array2)):
        if array2[i,2] < limit:
            array2[i,2] = limit
    pcloud = o3d.geometry.PointCloud()
    pcloud.points = o3d.utility.Vector3dVector(array2)
    o3d.visualization.draw_geometries([pcloud],width=1280,height=720)
    return pcloud
    


#with open(path1) as f:
    #i=0
    #while i <2000:
        #i += 1
        #print(f.readline())
        

#cld = filter(path)
#out = radius_filter(path)
