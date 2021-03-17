import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, Add
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import timeit
import os
from PIL import Image
from tensorflow.keras import utils
from copy import deepcopy as dc
from random import randint
from img_data_process import sort_nicely
import pydot
import pydotplus
tf.config.run_functions_eagerly(True)
tf.config.set_visible_devices([], 'GPU')

tf.keras.backend.set_floatx('float32')

x_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\train\\lr\\cropped' 
label_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\train\\hr\\cropped' 
test_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\lr\\to_test' 
test_hr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\hr\\cropped'
save_predictions = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\output'
save_model = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\models\\'
def psnr(a,b):
    return  tf.image.psnr(a, b, max_val=1.0)
def vgg_loss( y_true, y_pred):
    #From deepak https://gist.github.com/deepak112/c76ed1dbbfa3eff1249493eadfd2b9b5
    vgg19 = VGG19(include_top=False, weights='imagenet', input_shape=(32,32,3))
    vgg19.trainable = False
    for l in vgg19.layers:
        l.trainable = False
    model = Model(inputs=vgg19.input, outputs=vgg19.get_layer('block5_conv4').output)
    model.trainable = False
    return tf.math.reduce_mean(tf.math.square(model(y_true) - model(y_pred)))

def ssim(y_true, y_pred):
  return tf.reduce_mean(tf.image.ssim(y_true, y_pred, 2.0))

x_data = os.listdir(x_path)
y_data = os.listdir(label_path)

for i in range(3):
    k = randint(100, 50000)
    print(x_data[k], y_data[k], '\n', 'testing data and labels order')


def data(path,datas):
    os.chdir(path)
    x = []
    for img in datas:
        if img.rfind('.png') != -1: 
            i = np.asarray(Image.open(img))
            thr = np.stack((i,)*3, axis =-1)
            assert thr.shape == (32,32,3)
            x.append(thr)
            #x.append(np.asarray(Image.open(img)))
        else:print('not image the file: ',img)
    print("Array with len",len(x))
    for i in range(len(x)):
        x[i] = x[i]/255
    return dc(np.asarray(x))

y = data(label_path, y_data)
x = data(x_path,x_data)

img_height = 32 #42 #63
img_width = 32 #64 #96
channels = 3 #1
padding = "same"
optimizer = optimizers.Adam()
batch = 8


input = Input(shape = (img_height, img_width, channels))

f1 = 9
f2 = 1
f3 = 5


n1 = 64
n2 = 32
n3 = channels


x1 = Conv2D(n1, (f1, f1), padding = padding, activation='relu',)(input)
x2 = Conv2D(n2, (f2, f2),padding = padding, activation='relu',)(x1)
#x3 = Add()([input,x2])
x4 = Conv2D(n3, (f3, f3),padding = padding,)(x2)


m = Model(input,x4, name = '{filter0}_{filter1}_{filter2}_in_{a}_{b}_{c}_vggloss'.format(filter0 = n1,filter1 = n2,filter2 = n3, a=f1,b=f2,c = f3))
m.summary()

m.compile(optimizer=optimizer, loss= vgg_loss, metrics=[psnr])

os.chdir(save_model)
tf.keras.utils.plot_model(model = m,to_file= 'vgg_loss.png'.format(n1), show_shapes = True,show_dtype = True)
early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10,verbose = 2, restore_best_weights= True)

start = timeit.default_timer()
try:
    history = m.fit(x,y, batch_size= batch, epochs = 60,validation_split =0.1, verbose = 2, callbacks = early_stop, shuffle = True )
except KeyboardInterrupt:
    m.save('{filter0}_{filter1}_{fi3}_{a}_{b}_{c}_vgg_loss'.format(filter0 = n1,filter1 = n2,fi3 = n3,a=f1,b=f2,c = f3))
stop = timeit.default_timer()
print('Time: ', stop - start, 'Time minutes:' ,(stop - start)//60)  

print(history.history.keys())

plt.plot(history.history['psnr'])
plt.plot(history.history['val_psnr'])
plt.title('model PSNR')
plt.ylabel('PSNR')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()

test_data = sort_nicely(os.listdir(test_path))
evltion = sort_nicely(os.listdir(test_hr))
test = data(test_path,test_data)
evaluation = data(test_hr, evltion)



e = m.evaluate(x=test,y=evaluation,verbose = 1, batch_size = len(test))
print('evaluation loss = {}, evaluation PSNR = {} '.format(e[0],e[1]))
predictions = m.predict(test,verbose = 1)


min_loss = np.min(np.asarray(history.history['val_loss']))
max_loss = np.max(np.asarray(history.history['val_psnr']))

os.chdir(save_model)
f = open('vgg_loss_{filter0}_{filter1}_{filter2}_{a}_{b}_{c}_mae.txt'.format(filter0 = n1,filter1 = n2,filter2 = n3 ,a=f1,b=f2,c = f3),'w')
f.write('Αρχιτεκτονική δικτύου {a}-{b}-{c}, με channels {d}-{e}-{f} \n'.format(a=f1,b=f2,c= f3,d=n1,e=n2,f=n3))

f.write('Ελάχιστο Mean Square Error στο validation set: {} \n'.format(min_loss))
f.write('Μέγιστο PSNR στο validation set: {} \n'.format(max_loss))


f.write('Testing loss {loss}, PSNR {ps} \n'.format(loss = e[0],ps = e[1]))
f.write('Χρόνος εκτέλεσης: {m} λεπτά, και {s} δευτερόλεπτα \n'.format(m = int((stop - start)//60), s = int(np.round((stop - start)%60))))
f.close()



count = 0
for img in predictions:

    grey =  np.mean(img, -1)

    if grey.shape == (32,32):
        grey = grey.astype(np.uint8)
        i = Image.fromarray(grey)
        os.chdir(save_predictions)
    else:
        img = np.squeeze(grey, axis=2) * 255
        img = img.astype(np.uint8)
        i = Image.fromarray(img)
        os.chdir(save_predictions)
        i.save('{}_predicted.png'.format(count))
        count += 1



    


