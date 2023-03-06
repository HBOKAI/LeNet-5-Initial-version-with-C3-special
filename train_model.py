import tensorflow as tf
from tensorflow.keras import Sequential, layers, losses, optimizers, datasets
import numpy as np
import matplotlib.pyplot as plt
import os

    
class Con_sp(tf.keras.layers.Layer):

    def __init__(self, filter_num, filter_size, **kwargs):
        super(Con_sp, self).__init__(**kwargs)
        self.filter_num = filter_num
        self.filter_size = filter_size

    def build(self, input_shape): 
        self.weights3 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,3,6]),trainable=True,name='Weight3')
        self.weights4 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,4,6]),trainable=True,name='Weight4')
        self.weights4_4 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,4,3]),trainable=True,name='Weight4_4')
        self.weights6 = tf.Variable(initial_value=tf.random.normal([self.filter_size,self.filter_size,6,1]),trainable=True,name='Weight6')
        self.bias1 = tf.Variable( initial_value=tf.random.normal([self.filter_num]),trainable=True,name='Bias11')
        # self.weights1 = self.add_weight(shape=(self.filter_size,self.filter_size,6,self.filter_num),initializer='random_normal',trainable=True,name='ww11')
        # self.bias1 = self.add_weight(shape=(self.filter_num),initializer='random_normal',trainable=True,name='bb11')
        # self.shape1 = input_shape
        #相当于设置self.built = True
        super(Con_sp,self).build(input_shape)

    def call(self, inputs):
        for i in range(16):
            if i < 1:
                j=i
                basic_out = tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,0:1,0:1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                basic_out += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,1:2,0:0+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                basic_out += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,2:3,0:0+1],[1,1,1,1],'VALID',)
                basic_out = tf.nn.bias_add(basic_out, self.bias1[i:i+1])
                # print(basic_out.shape)
            if i>=1 and i < 6:
                j=i
                k=i
                output11 = tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,0:1,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,1:2,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1: 
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights3[...,2:3,k:k+1],[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                # print(output11)
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
            if i >= 6 and i < 12 :
                j=i-6
                k=i-6
                output11 = tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,0:1,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,1:2,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,2:3,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4[...,3:4,k:k+1],[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                # print(output11)
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
            if i >= 12 and i < 15 :
                j=i-12
                k=i-12
                output11 = tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,0:1,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,1:2,k:k+1],[1,1,1,1],'VALID')
                j+=2
                if (j/6) >= 1:
                    j-=6
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,2:3,k:k+1],[1,1,1,1],'VALID')
                j+=1
                if (j/6) == 1:
                    j=0
                output11 += tf.nn.conv2d(inputs[...,j:j+1],self.weights4_4[...,3:4,k:k+1],[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                # print(output11)
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
            if i == 15 :
                output11 = tf.nn.conv2d(inputs,self.weights6,[1,1,1,1],'VALID')
                output11 = tf.nn.bias_add(output11, self.bias1[i:i+1])
                basic_out = tf.concat(axis=3,values=[basic_out,output11])
                # print(i,"和: ",basic_out.shape)
        return basic_out
    # def get_config(self):
    #     # base_config = super(Con_sp, self).get_config
    #     # config1 = {
    #     #     "filter_num":self.filter_num ,
    #     #     "filter_size":self.filter_size
    #     #     }
    #         # dict(list(base_config.items()) + list(config1.items()))
    #     return {"filter_num":self.filter_num , "filter_size":self.filter_size}
    def get_config(self):
        config = super(Con_sp, self).get_config()
        config.update({
            "filter_num":self.filter_num,
            "filter_size":self.filter_size
        })
        return config   

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

(train_x, train_y),(test_x, test_y) = tf.keras.datasets.mnist.load_data()

train_x = 2*tf.convert_to_tensor(train_x,dtype=tf.float32)
train_x = tf.pad(train_x,[[0,0],[2,2],[2,2]],"CONSTANT",0) # 外圍填充
train_x = train_x / 255 # 圖像歸一化 0~1
train_x = tf.expand_dims(train_x,-1)
train_y = tf.one_hot(train_y, depth=10)

test_x = 2*tf.convert_to_tensor(test_x,dtype=tf.float32)
test_x = tf.pad(test_x,[[0,0],[2,2],[2,2]],"CONSTANT",0) # 外圍填充
test_x = test_x / 255 # 圖像歸一化 0~1
test_x = tf.expand_dims(test_x,-1)
test_y = tf.one_hot(test_y, depth=10)

model_input = layers.Input(shape=(32,32,1))
x = layers.Conv2D(6,kernel_size=5,strides=1)(model_input)
x = layers.Activation('relu')(x)
x = layers.AveragePooling2D(pool_size=2,strides=2)(x)
x = Con_sp(filter_num=16,filter_size=5)(x)
x = layers.Activation('relu')(x)
x = layers.AveragePooling2D(pool_size=2,strides=2)(x)
x = layers.Flatten()(x)
x = layers.Dense(120,activation='relu')(x)
x = layers.Dense(84,activation='relu')(x)
model_output = layers.Dense(10,activation='softmax')(x)

model = tf.keras.Model(inputs=model_input,outputs=model_output)
model.build(input_shape=(None,32,32,1))
model.summary() # 看建立的架構
tf.keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
model.compile(loss="categorical_crossentropy",optimizer=tf.keras.optimizers.Adam(),metrics=["accuracy"]) # 定義所要採用的loss funtion, optimizer, metrics
history = model.fit(x=train_x,y=train_y,batch_size=32,epochs=16,verbose=1,validation_split=0.1) # 設定 batch(批), epochs(跌代), verbose, validation(驗證，功能還不太確定)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
# plt.ylim([0.5, 1])
plt.legend(loc='lower right')

score = model.evaluate(test_x,test_y) #評估誤差
print("Test Loss: " ,score[0])
print("Test Accuracy: ",score[1])

result = model.predict(test_x[0:9])
print('前9筆預測結果: ',np.argmax(result, axis=-1),'\n')
print('前9筆實際值: ',np.argmax(test_y[0:9],axis=-1),'\n')

model.save('./CNN_MODEL')
model.save('./CNN_MODEL.h5')
plt.show()

