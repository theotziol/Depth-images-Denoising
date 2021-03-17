import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Cropping2D, Add, BatchNormalization, Conv2DTranspose, Activation
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import timeit
import os
from PIL import Image
from tensorflow.keras import utils
from copy import deepcopy as dc
from random import randint
from img_data_process import sort_nicely




x_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\train\\lr\\cropped' 
label_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\train\\hr\\cropped' 
test_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\lr\\to_test' 
test_hr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\hr\\cropped'
save_predictions = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\test\\output'
save_model = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset2\\models\\'
def psnr(a,b):
    return  tf.image.psnr(a, b, max_val=1.0)



x_data = os.listdir(x_path)
y_data = os.listdir(label_path)
test_data = sort_nicely(os.listdir(test_path))
evltion = sort_nicely(os.listdir(test_hr))
for i in range(2):
    k = randint(100, 30000)
    print(x_data[k], y_data[k], '\n', 'testing data and labels order')



def data(path,datas):
    os.chdir(path)
    x = []
    
    for img in datas:
        if img.rfind('.png') != -1: 
            x.append(np.asarray(Image.open(img)))
        else:print('not image the file: ',img)
    print("Array with len",len(x))
    for i in range(len(x)):
        x[i] = x[i]/255
    return dc(np.asarray(x))

y = data(label_path, y_data)
x = data(x_path,x_data)
test = data(test_path,test_data)
evaluation = data(test_hr, evltion)


img_height = 32 #42 #63
img_width = 32 #64 #96
channels = 1
padding = "same"
optimizer = optimizers.Adam()
batch = 128 #256 #48 #42 



f1 = 9
f2 = 5
f3 = 3
f4 = 1
f5= 3

n0 = 128
n1 = 64
n2 = 32
n3 = 1

input = Input(shape = (img_height, img_width, channels))


x1 = Conv2D(n0, (f1, f1), padding = padding, activation= 'relu')(input)
b0 = Add()([input,x1])
x2 = Conv2D(n0, (f1, f1), padding = padding, activation='relu',)(b0)
b1 = Add()([input,x2])
x3 = Conv2D(n0, (f3, f3), padding = padding, activation='relu',)(b1)
b2 = Add()([input,x3])
x4 = Conv2D(n0, (f3, f3), padding = padding, activation='relu',)(b1)
b3 = Add()([input,x4])
x11 = Conv2D(n0, (f3, f3), padding = padding, activation='relu',)(b3)
b10 = Add()([input,x11])
x12 = Conv2D(n0, (f2, f2), padding = padding, activation='relu',)(b10)
b11 = Add()([input,x12])
x13 = Conv2D(n0, (f2, f2), padding = padding, activation='relu',)(b11)
b12 = Add()([input,x13])
x14 = Conv2D(1, (f4, f4),padding = padding)(b12)


os.chdir(save_model)
m = Model(input,x14, name = 'deep8')
m.summary()
tf.keras.utils.plot_model(model = m,to_file= 'deep8.png', show_shapes = True,show_dtype = True)

m.compile(optimizer=optimizer, loss='mse', metrics=[psnr])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=7, restore_best_weights= True)

start = timeit.default_timer()
history = m.fit(x,y, batch_size= batch, epochs =60,validation_split =0.1, verbose = 2, callbacks = early_stop, shuffle = True )
stop = timeit.default_timer()
print('Time: ', stop - start, 'Time minutes:' ,(stop - start)//60)  
os.chdir(save_model)
m.save('deep8')
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
predictions = m.predict(test,verbose = 2)

e = m.evaluate(x=test,y=evaluation,verbose = 2)

min_loss = np.min(np.asarray(history.history['val_loss']))
max_loss = np.max(np.asarray(history.history['val_psnr']))

#f = open('deep8.txt','w')
#f.write('Αρχιτεκτονική δικτύου 14 στρωμάτων με 64 φίλτρα και μέγεθος kernel 3x3 \n')


'''f.write('Ελάχιστο Training loss στο validation set: {} \n'.format(min_loss))
f.write('Μέγιστο PSNR στο validation set: {} \n'.format(max_loss))


f.write('Testing loss {loss}, PSNR {ps} \n'.format(loss = e[0],ps = e[1]))
f.write('Χρόνος εκτέλεσης: {m} λεπτά, και {s} δευτερόλεπτα \n'.format(m = int((stop - start)//60), s = int(np.round((stop - start)%60))))
f.close()'''



count = 0
for img in predictions:
    img = np.squeeze(img, axis=2) * 255
    img = img.astype(np.uint8)
    i = Image.fromarray(img)
    os.chdir(save_predictions)
    i.save('{}_predicted.png'.format(count))
    count += 1





