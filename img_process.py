''' Preprocess 2d images for the depthmap Super Rsolution
Original depthmaps have shape = (63,96), modified to shape = 62,96 for cropping
desired cropped shape = (32,32)
Created by theotziol'''

import numpy as np
import os
import glob #for find pictures by text
import shutil #for copy
import PIL
from PIL import Image
from copy import deepcopy as dc
from matplotlib import pyplot as plt
import re

depthmaps = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\'
raw = depthmaps+'raw\\'
train = depthmaps +'train\\'
train_hr = train +'hr\\'
train_lr = train + 'lr\\' 

test = depthmaps + 'test\\'
test_hr = test + 'hr\\'
test_lr = test + 'lr\\'
evaluation = test_hr + 'cropped\\'
to_test = test_lr + 'to_test\\'

def text(txt): #usefull for saving new images using old_name
    x = txt.split('.')
    return x[0]

def img_generator(path_from_img:str, func, path_to_save = None): #the func to generate images
    os.chdir(path_from_img)
    images = os.listdir(path_from_img)
    if path_to_save == None:
        list(map(func,images))
    else:
        paths = [path_to_save for i in range(len(images))]
        list(map(func,images,paths))

def reshape(name:str,path_to_save, path = depthmaps):
    os.chdir(path)
    print('Copying images from {}'.format(os.getcwd()), 'with name {}'.format(name),'to {}'.format(path_to_save))
    for file in glob.glob(name): #'*lr.png' or '*hr.png'
        if file not in os.listdir(path_to_save):
            print(file)
            img = np.asarray(Image.open(file))
            img2 = img[:-1,:]
            new_image = Image.fromarray(img2)
            os.chdir(path_to_save)
            new_image.save(file)
            os.chdir(path)


def flip(image_path:str): #modified, added results as rotated used with img_generator
    if image_path.rfind('.png') != -1:

        img = np.asarray(Image.open(image_path))
        mirror_ho = dc(np.flip(img,1))
        mirror_ve = dc(np.flip(img,0))
        mirror_ho_to_ve = dc(np.flip(mirror_ho,0))

        new_image = Image.fromarray(mirror_ho)
        new_image2 = Image.fromarray(mirror_ve)
        new_image3 = Image.fromarray(mirror_ho_to_ve)

        new_image.save('{}_flipped.png'.format(text(image_path)))
        new_image2.save('{}_flipped_rotated.png'.format(text(image_path)))
        new_image3.save('{}_rotated.png'.format(text(image_path)))
    else: print(image_path)

def rotated(image_path,path_to_save): # can be implemented on flip (modified) used here only for 90degrees
    local = os.getcwd()
    if image_path.rfind('.png') != -1:
        img = Image.open(image_path)
        new = img.rotate(90, expand = 1) #180
        os.chdir(path_to_save)
        new.save('{}_rotated_vert.png'.format(text(image_path)))
    else: print(image_path)
    os.chdir(local)

def cropped(image_path:str, path_to_save):
    new_img_w = 32
    new_img_h = 32
    stride_w = 8
    stride_h = 6
    style = 'hrzntl'
    count = 0
    local = os.getcwd()
    if image_path.rfind('.png') != -1: 
        img = dc(np.asarray(Image.open(image_path)))
        img_height,img_width = img.shape
        if img_height == 96:
            stride_w = 5
            stride_h = 8
            style = 'vrtcl'
        for h in range(0,img_height - new_img_h +1, stride_h):
            for w in range(0,img_width - new_img_w +1, stride_w):
                sub_img = img[h : h + new_img_h, w : w + new_img_w]
                new_image = Image.fromarray(sub_img)
                os.chdir(path_to_save)
                new_image.save('{name}_crpd_{s}_h_{height}_and_w_{width}.png'.format(name = text(image_path),
                                                                                        s = style,
                                                                                        height = h, width = w))
                count += 1
        
        print('Generated {i} cropped images, from {p} to {p2}'.format(i = count, p = image_path,
                                                                     p2 =path_to_save))
    else: print('no image',image_path)
    os.chdir(local)



def cropped_test(image_path:str, path_to_save):
    new_img_w = 32
    new_img_h = 32
    stride_w = 32
    stride_h = 30
    count = 0
    local = os.getcwd()
    if image_path.rfind('.png') != -1: 
        img = dc(np.asarray(Image.open(image_path)))
        img_height,img_width = img.shape
        for h in range(0,img_height - new_img_h +1, stride_h):
            for w in range(0,img_width - new_img_w +1, stride_w):
                sub_img = img[h : h + new_img_h, w : w + new_img_w]
                new_image = Image.fromarray(sub_img)
                os.chdir(path_to_save)
                new_image.save('{name}_crpd_h_{height}_and_w_{width}.png'.format(name = text(image_path),
                                                                                        height = h, width = w))
                count += 1
        
        print('Generated {i} cropped images, from {p} to {p2}'.format(i = count, p = image_path,
                                                                     p2 =path_to_save))
    else: print('no image',image_path)
    os.chdir(local)
#img_generator(test_hr,cropped_test,evaluation)
#img_generator(test_lr,cropped_test, to_test)

def sort_nicely( l ):
# Sort the given list in the way that humans expect.
#  https://blog.codinghorror.com/sorting-for-humans-natural-sort-order/

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l

def pred_recon(image_path):
    os.chdir(image_path)
    tex = image_path.split('\\')
    imgs = dc(sort_nicely(os.listdir(image_path)))
    stride_w = 32
    stride_h = 30
    original_h = 62
    original_w = 96
    cropped = 6
    images = [i for i in range( int( len(imgs) // cropped ))]
    new_img = np.zeros(shape = (original_h, original_w))

    start = 0
    end = 5
    for i in range(len(images)):
        
        images[i] = [np.asarray(Image.open(imgs[x])) for x in range(start,end+1)]
        start += 6 
        end += 6
    
    for i in range(len(images)):
        row1 = np.hstack((images[i][0],images[i][1],images[i][2]))
        row2 = np.hstack((images[i][3],images[i][4],images[i][5]))
        new_image = np.vstack((row1[:-2,:],row2))
        x = Image.fromarray(new_image.astype(np.uint8))
        x.save('{}_predicted_recon_{}.png'.format(i,tex[-1]))




