import tensorflow as tf
from PIL import Image
import numpy as np
import cv2
import time
np.set_printoptions(threshold=np.inf)# np.inf = 無窮大的浮點數，若矩陣數量大於threshold部分數值會以...代替
np.set_printoptions(suppress=True)#抑制顯示小數位數
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
 

def show_xy(event,x,y,flags,param):
    global dots, draw,img_gray                    # 定義全域變數
    if flags == 1:
        if event == 1:
            dots.append([x,y])            # 如果拖曳滑鼠剛開始，記錄第一點座標
        if event == 4:
            dots = []                     # 如果放開滑鼠，清空串列內容
        if event == 0 or event == 4:
            dots.append([x,y])            # 拖曳滑鼠時，不斷記錄座標
            x1 = dots[len(dots)-2][0]     # 取得倒數第二個點的 x 座標
            y1 = dots[len(dots)-2][1]     # 取得倒數第二個點的 y 座標
            x2 = dots[len(dots)-1][0]     # 取得倒數第一個點的 x 座標
            y2 = dots[len(dots)-1][1]     # 取得倒數第一個點的 y 座標
            cv2.line(draw,(x1,y1),(x2,y2),(255,255,255),20)  # 畫直線
        cv2.imshow('img', draw)#draw

# _custom_objects = {
#                 'CON_SP':Con_sp,
#                 }
model = tf.keras.models.load_model("./CNN_MODEL") 
# model = tf.keras.models.load_model("./CNN_MODEL.h5",custom_objects=_custom_objects) #無法使用.h5檔案讀取

# img = Image.open("D:/Desktop/學校/實驗室/程式/TENSORFLOW/MNIST/test_images/7/7.860.jpg")
# img = Image.open("D:/Desktop/學校/實驗室/程式/gray.png")


dots = []   # 建立空陣列記錄座標
w = 320
h = 320
draw = np.zeros((h,w,3), dtype='uint8')   # 建立 420x240 的 RGBA 黑色畫布
while True:
    cv2.imshow('img', draw)
    cv2.setMouseCallback('img', show_xy)
    keyboard = cv2.waitKey(5)                    # 每 5 毫秒偵測一次鍵盤事件
    if keyboard == ord('q'):
        break                                    # 按下 q 就跳出

    if keyboard == ord('n'):
        img_gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)   # 轉為灰度圖
        img = cv2.resize(img_gray,(32,32))                          # 變更圖片尺寸
        cv2.imwrite("./images/gray.png",img)
        # img = np.array(img)
        # img = np.pad(img,pad_width=((2,2),(2,2)),mode='constant',constant_values=0)
        img = img/255
        img = np.expand_dims(img,0)
        np.savetxt("show_data.txt",img[0],fmt='%.01f')
        # print(img)
        # print(img.shape)
        # model.summary()
        start = time.time()
        predict = model.predict(img)
        end = time.time()
        print('預測結果: ',np.argmax(predict, axis=-1))
        print('預測時間: ',end-start)
        draw = np.zeros((h,w,3), dtype='uint8')
    if keyboard == ord('r'):
        draw = np.zeros((h,w,3), dtype='uint8')  # 按下 r 就變成原本全黑的畫布
        cv2.imshow('img', draw)


