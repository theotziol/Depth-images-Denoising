import numpy as np
import cv2
from PIL import Image
from PIL.ImageFilter import SHARPEN 
import os
from SSIM_PIL import compare_ssim 
import matplotlib.pyplot as plt
from img_process import sort_nicely as sr
import pandas as pd


def psnr(y_true, y_pred):

    assert y_true.shape == y_pred.shape, "Cannot calculate PSNR. Input shapes not same." \
                                         " y_true shape = %s, y_pred shape = %s" % (str(y_true.shape),
                                                                                   str(y_pred.shape))

    return (20*np.log10(255)) -10. * np.log10(np.mean(np.square(y_pred - y_true)))



def ssim(a,b):
    a = Image.open(a)
    b = Image.open(b)
    print(np.round(compare_ssim(a,b), 4))
    return  np.round(compare_ssim(a,b), 4)

def ssim_PIL(a,b):
    b = Image.open(b)
    return  np.round(compare_ssim(a,b), 4)

def sharp(lr_image):
    im1 = Image.open(lr_image)
    im1 = im1.filter(SHARPEN)
    return im1


def compare(img1, img2):
    ssi = np.round(ssim(img1, img2),4)
    img1 = np.asarray(Image.open(img1))
    img2 = np.asarray(Image.open(img2))
    psn = np.round(psnr(img1,img2),3)
    tex = ' PSNR = {} , SSIM = {}'.format(psn,ssi)
    return tex, psn, ssi

def compare_PIL(img1,img2):
    ssi = ssim_PIL(img1, img2)
    img1 = np.asarray(img1)
    img2 = np.asarray(Image.open(img2))
    psn = np.round(psnr(img1,img2),3)
    tex = ' PSNR = {} , SSIM = {}'.format(psn,ssi)
    return tex, psn, ssi


def plot(im1,im2,im3, im4, lbl1, lbl2, lbl3, title):
    im1 = Image.open(im1)
    #im2 = Image.open(im2)
    im3 = Image.open(im3)
    im4 = Image.open(im4)
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6,6),
                             sharex=True, sharey=True)
    ax = axes.ravel()
    fig.suptitle('Αποτελέσματα {} σε PSNR και SSIM'.format(title))
    ax[0].imshow(im1, cmap=plt.cm.gray)
    ax[0].set_title('Input')
    ax[0].set_xlabel(lbl1)

    ax[1].imshow(im2, cmap=plt.cm.gray)
    ax[1].set_title('Input with sharpening filter')
    ax[1].set_xlabel(lbl2)

    ax[2].imshow(im3, cmap=plt.cm.gray)
    ax[2].set_title('predicted')
    ax[2].set_xlabel(lbl3)

    ax[3].imshow(im4, cmap=plt.cm.gray)
    ax[3].set_title('Ground Truth')
    ax[3].set_xlabel('label')
    plt.tight_layout()
    plt.show()


pthlr =  'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\lr\\to_test\\'  
pthhr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\hr\\cropped\\'
pthpr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\output\\256_128_64_1\\'

#pthlr =  'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\lr\\'
#pthhr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\hr\\'
#pthpr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\output\\128_64_32_1\\recon\\'
hr_path = os.listdir(pthhr)
lr_path = os.listdir(pthlr)
pred = os.listdir(pthpr)


hr = sr([pthhr+x for x in hr_path if x.endswith('.png')])
lr = sr([pthlr+x for x in lr_path if x.endswith('.png')])
pr = sr([pthpr+x for x in pred if x.endswith('.png')])
#print(lr)
shrpn_lis = []

for i in range(len(lr)):
    x = sharp(lr[i])
    shrpn_lis.append(x)
lis = []
lis_s = []
for i in range(len(lr)):
    lr_hr ,p1, s1 = compare(lr[i], hr[i])
    sharp_hr,p2,s2 = compare_PIL(shrpn_lis[i], hr[i])
    pred_hr,p3,s3 = compare(pr[i], hr[i])
    #plot(lr[i], shrpn_lis [i], pr[i], hr[i], lr_hr, sharp_hr, pred_hr,'Αρχιτεκτονικής 256-128-1')
    if p1 > p3 or p2 > p3:
        psn = 0
        dif_p = p3-p1
    else:
        psn = 1
        dif_p = p3-p1
    if s1 > s3 or s2 > s3:
        ss = 0
        dif_s = s3-s1
    else:
        ss = 1
        dif_s = s3-s1
    p = list((p1,p2,p3,dif_p,psn))
    s = list((s1,s2,s3,dif_s,ss))
    lis.append(p)
    lis_s.append(s)
x = np.sum(np.asarray(lis)[:,4])
y = np.sum(np.asarray(lis_s)[:,4])

dtf = pd.DataFrame(lis, columns = (['input','input+ sharp','predicted','difference psnr','result']))
dtf1 = pd.DataFrame(lis_s, columns = (['input','input+ sharp','predicted', 'difference ssim', 'result']))

print('\n'*3, 'PSNR = ',x ,'ssim = ', y)

os.chdir(pthpr)
ask = input('to_save? ')
t = ask.isdigit()

if not t:
    name = input('give name to dataframe of architecture: ')
    dtf.to_csv('PSNR_{}.csv'.format(name))
    dtf1.to_csv('SSIM_{}.csv'.format(name))
    le1 = len(lis)
    le2 = len(lis_s)
    assert le1 ==le2
    f = open('sum.txt','w')
    f.write('To PSNR βελτιώθηκε σε {} από τις {} εικόνες \n'.format(x, le1))
    f.write('To SSIM βελτιώθηκε σε {} από τις {} εικόνες \n'.format(y, le2))
    f.write('To PSNR βελτιώθηκε σε ποσοστό {} % \n'.format(x/le1 *100))
    f.write('To SSIM βελτιώθηκε σε ποσοστό {} % \n'.format(y/le2 *100))
    
    f.close()







'''The convolution matrix used for sharpening is,
(-2, -2, -2,

-2, 32, -2,

-2, -2, -2)

a 3x3 matrix.'''