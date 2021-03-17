import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, Cropping2D
from tensorflow.keras.applications.vgg19 import VGG19
import matplotlib.pyplot as plt
import timeit
import os
from PIL import Image
from tensorflow.keras import utils
from copy import deepcopy as dc
from random import randint
from img_data_process import sort_nicely



x_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset\\train\\lr\\cropped' 
label_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset\\train\\hr\\cropped' 
test_path = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset\\test\\lr\\to_test' 
test_hr = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset\\test\\hr\\cropped'
save_predictions = 'D:\\Documents\\Master Thesis\\depthmaps\\final_dataset\\test\\output'
def psnr(a,b):
    return  tf.image.psnr(a, b, max_val=1.0)



x_data = os.listdir(x_path)
y_data = os.listdir(label_path)
test_data = sort_nicely(os.listdir(test_path))
evltion = sort_nicely(os.listdir(test_hr))
for i in range(5):
    k = randint(100, 50000)
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


img_height = 21 #42 #63
img_width = 32 #64 #96
channels = 1
padding = "same"
optimizer = optimizers.Adam()
batch = 484 #256 #48 #42 



f1 = 5
f2 = 3
f3 = 9
f4 = 1
f5= 3

n0 = 128
n1 = 64
n2 = 32
n3 = 1

input = Input(shape = (img_height, img_width, channels))
x1 = Conv2D(n0, (f1, f1), padding = padding, activation='relu',name = 'encoder_start')(input)
x2 = MaxPooling2D((2, 2), padding='same')(x1)
x3 = Conv2D(n1, (f2, f2), padding = padding, activation='relu')(x2)
encoded = MaxPooling2D((2, 2), padding='same')(x3)

x4 = Conv2D(n1, (f2, f2), padding = padding, activation='relu', name = 'decoder_start')(encoded)
x5 = UpSampling2D((2, 2))(x4)
x6 = Conv2D(n0, (f1, f1),padding = padding, activation='relu')(x5)
x7 = Cropping2D(cropping = ((1,0),(0,0)))(x6)
x8 = UpSampling2D((2, 2))(x7)
x9 = Conv2D(1, (f5, f5),padding = padding, name = 'decoder_end')(x8)
decoded = Cropping2D(cropping = ((0,1),(0,0)))(x9)

x10 = Conv2D(n1, (f3, f3), padding = padding, activation='relu',)(decoded)
x11 = Conv2D(n2, (f4, f4),padding = padding, activation='relu',)(x10)
x12 = Conv2D(n3, (f5, f5),padding = padding,)(x11) 

m = Model(input,x12, name = 'auto_encoder_SRCNN')
m.summary()

m.compile(optimizer=optimizer, loss='mse', metrics=[psnr])

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", patience=10, restore_best_weights= True)

start = timeit.default_timer()
history = m.fit(x,y, batch_size= batch, epochs = 110,validation_split =0.05, verbose = 2, callbacks = early_stop, shuffle = True )
stop = timeit.default_timer()
print('Time: ', stop - start, 'Time minutes:' ,(stop - start)//60)  
m.save('encodersrcnn')

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

f = open('encodersrcnn.txt','w')
f.write('Αρχιτεκτονική δικτύου encoder με SRCNN 64-32-1,9-1-5 \n')

f.write('Ελάχιστο Training loss στο validation set: {} \n'.format(min_loss))
f.write('Μέγιστο PSNR στο validation set: {} \n'.format(max_loss))


f.write('Testing loss {loss}, PSNR {ps} \n'.format(loss = e[0],ps = e[1]))
f.write('Χρόνος εκτέλεσης: {m} λεπτά, και {s} δευτερόλεπτα \n'.format(m = int((stop - start)//60), s = int(np.round((stop - start)%60))))
f.close()



count = 0
for img in predictions:
    img = np.squeeze(img, axis=2) * 255
    img = img.astype(np.uint8)
    i = Image.fromarray(img)
    os.chdir(save_predictions)
    i.save('{}_predicted.png'.format(count))
    count += 1





