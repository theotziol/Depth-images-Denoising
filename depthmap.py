import numpy as np
from matplotlib import pyplot as plt
import open3d as o3d
import matplotlib.image as img
from copy import deepcopy as dc
from PIL import Image
#np.set_printoptions(threshold=np.inf)

class pointcloud_to_depth():

    def __init__(self, cloud):
        self.cloud = o3d.io.read_point_cloud(cloud)
        self.array3d = np.asarray(self.cloud.points)
        print(self.array3d.shape)
        self.x_values = self.array3d[:,0]
        self.y_values = self.array3d[:,1]
        self.z_values = self.array3d[:,2]

        self.x_axis_rounded = np.round(self.x_values,1)
        self.y_axis_rounded = np.round(self.y_values,1)
        self.z_axis_rounded = np.round(self.z_values,1)

        self.x_min = np.min(self.x_axis_rounded)
        self.x_max = np.max(self.x_axis_rounded)

        self.y_min = np.min(self.y_axis_rounded)
        self.y_max = np.max(self.y_axis_rounded)

        self.z_min = np.min(self.z_axis_rounded)
        self.z_max = np.max(self.z_axis_rounded)
        print(self.x_max, self.y_max)


        self.x_axis = np.unique(self.x_axis_rounded)
        self.y_axis = np.unique(self.y_axis_rounded)
        self.z_axis = np.unique(self.z_axis_rounded)
        self.width = 60#np.rint(self.x_max-self.x_min).astype(np.int16)
        self.length =  40#np.rint(self.y_max-self.y_min).astype(np.int16)
        self.depth_length = len(self.z_axis)

        print("Width of point cloud to image = {}".format(self.width),
              "length of point cloud to image = {}".format(self.length),sep = '\n')

    def to_map(self, axis:str): #all variables must be iterative for map() function
        if axis == 'x':
            x = self.x_axis_rounded
            in_min = np.full_like(x, self.x_min)
            in_max = np.full_like(x, self.x_max)
            out_min = np.zeros_like(x)
            out_max = np.full_like(x, self.width)
        elif axis == 'y':
            x = self.y_axis_rounded
            in_min = np.full_like(x, self.y_min)
            in_max = np.full_like(x, self.y_max)
            out_min = np.zeros_like(x)
            out_max = np.full_like(x,self.length)
        elif axis =='z':
            x = self.z_axis_rounded
            in_min = np.full_like(x,self.z_min)
            in_max = np.full_like(x,self.z_max)
            out_min = np.zeros_like(x)
            if self.depth_length <= 255:
                out_max = np.full_like(x,self.depth_length)
            else: out_max = np.full_like(x,255)
        else: print('wrong axis, give x,y, or z for axis')

        return np.asarray(list(map(self.linear_map, x, in_min, 
                    in_max, out_min , out_max)))
     

    def linear_map(self,x, in_min, in_max, out_min, out_max):
        return (x - in_min) * (out_max - out_min) // (in_max - in_min) + out_min

    def mapped(self):
        x = self.to_map('x')
        x = np.rint(x).astype(np.int16)
        y = self.to_map('y')
        y = np.rint(y).astype(np.int16)
        z = self.to_map('z')
        print(x.shape)
        print(y.shape)
        print(z.shape)

        xyz = np.stack((x,y,z), axis = 1)
        #print(xyz)
        canvas = np.ones((self.width+1,self.length+1))
        canvas[x, y] = z
        print(canvas)
        plt.imshow(canvas, cmap="Greys")
        plt.show()
        o3d.visualization.draw_geometries([self.cloud],width=1200, height=800)

def visualize(basecloud, movingcloud):
    cloud1 = o3d.io.read_point_cloud(basecloud)
    cloud2 = o3d.io.read_point_cloud(movingcloud)
    cloud1.paint_uniform_color([0.5, 0.5, 0.2])
    cloud2.paint_uniform_color([1, 0.5, 0.2]) 
    o3d.visualization.draw_geometries([cloud1, cloud2],width=1200, height=800)












test = 'D:\Documents\Master Thesis\processed_data\scan14\scan_14_pensa_1_cropped_rs_cloud.ply'
test2 = 'D:\Documents\Master Thesis\processed_data\scan14\scan_14_pensa_1_filtered_laser_cloud.ply'
visualize(test,test2)
test_cloud = pointcloud_to_depth(test2)
test_cloud.mapped()




image = 'D:\Documents\Master Thesis\depthmaps\depthmap_95_hr.png'
depth = img.imread(image)
#save depthmap to folder with Image.save()