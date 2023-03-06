import cv2
import numpy as np

dots = []   # 建立空串列記錄座標
w = 320
h = 320
draw = np.zeros((h,w,3), dtype='uint8')   # 建立 420x240 的 RGBA 黑色畫布
print(draw)
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
        
        img_gray = cv2.cvtColor(draw, cv2.COLOR_BGR2GRAY)
        cv2.imshow('oxxostudio', img_gray)#draw
def img_resize(input_img):
    # new_height = 32
    # new_width = 32
    # height = input_img.shape[0]
    # width = input_img.shape[1]
    # if width/height > new_width/new_height:
    new_img = cv2.resize(input_img,(32,32))
    return new_img
cv2.imshow('oxxostudio', draw)
cv2.setMouseCallback('oxxostudio', show_xy)

while True:
    keyboard = cv2.waitKey(5)                    # 每 5 毫秒偵測一次鍵盤事件
    if keyboard == ord('q'):
        break                                    # 如果按下 q 就跳出
    if keyboard == ord('r'):
        draw = np.zeros((h,w,3), dtype='uint8')  # 如果按下 r 就變成原本全黑的畫布
        cv2.imshow('oxxostudio', draw)
save_gray = img_resize(img_gray) 
cv2.imwrite("gray.png",save_gray)
cv2.destroyAllWindows()
