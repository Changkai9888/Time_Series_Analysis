import cv2,数据读取
import numpy as np,pandas as pd
from matplotlib import pyplot as plt
png_name=1
def plot_double(img1,img2):
    global png_name
    fig = plt.figure(figsize=(10,5))
    plt.subplot(121), plt.imshow(img1, cmap='gray')
    plt.title('Input Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2, cmap='gray',alpha=0.9);plt.imshow(img1, cmap='gray',alpha=0.3)
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.savefig('./gif/'+'{:03d}'.format(png_name)+'.png');png_name+=1;
    #plt.show()
    plt.close()
def filter(f):
    '''滤波器'''
    #基本滤波
    f[0][0]=0;#f[1][0]=0;f[-1][0]=0
    #全遮滤镜
    filt=np.zeros(f.shape)
    #刮除遮挡
    for i,k in [(i,k) for i in range(10) for k in range(10)]:
        if np.sqrt(i**2+k**2)<7 and np.sqrt(i**2+k**2)>0 and i>3 and k<5:#8055
            filt[i,k]=1;filt[i,-k]=1;filt[-i,-k]=1;filt[-i,k]=1
    #plt.imshow(np.fft.fftshift(filt), cmap='gray');plt.show()
    f=f* filt
    return  f
def FFD_2d(img,not_plot=0):
    #使用傅里叶变换得到频率域图像
    f0 = np.fft.fft2(img)
    f=filter(f0)
    #f[20:-20,:]=0;#abs(np.max(f));
    for i in range(0,len(f)//2,33):
        #f[i][-int(i*1)]=abs(np.max(f));
        #f[-i][int(i*1)]=abs(np.max(f));
        1==1
    fshift = np.fft.fftshift(f)
    #magnitude_spectrum = 10 * np.log10(np.abs(fshift))
    magnitude_spectrum =np.abs(fshift )
    # 将频率域图像可视化
    #plot_double(img,magnitude_spectrum)
    imgf= np.fft.fft2(f )
    f= np.fft.fft2(imgf)
    imgf= np.fft.fft2(f)
    if not_plot==0:
        plot_double(img,abs(imgf))
    return imgf
####
# 读入二值图像
#img = cv2.imread('Figure_5.png', 0)
df = pd.read_csv('.\data\c_min.csv')
c=df['close'].to_numpy()
def img_get_1(c,step=0):
    '''价格时间方块，色块深度表示时间段内的价格频数'''
    len_img=int((max(c)-min(c))//10*10)
    if step==0:
        step=max(1,len(c)//len_img//15*10)
    img = np.zeros((len_img,len(c)//step+1))
    avg=(max(c)+min(c))/2-len(img)//2
    for i,k in enumerate(c):
        if 0<int(k-avg)<=len(img):
            img[int(avg-k)][i//step]+=255//step
    return img
def img_get_2(c):
    '''无时间压缩，价格点被上下拉长'''
    len_img=int(max(c)-min(c))
    repet=len(c)//len_img
    img = np.zeros((len_img*repet,len(c)))
    avg=(max(c)+min(c))/2-len_img//2
    for i,k in enumerate(c):
        if 0>int(avg-k)*repet>int(avg-k-1)*repet>=-len(img)-1:
            img[(int(avg-k)-1)*repet:int(avg-k)*repet,i]+=255
    return img
def img_get_3(c):
    '''以尾价格为基准，固定价格空间高度'''
    len_img=int(100)
    repet=len(c)//len_img
    img = np.zeros((len_img*repet,len(c)))
    avg=c[-1]-len_img//2
    for i,k in enumerate(c):
        if 0>int(avg-k)*repet>int(avg-k-1)*repet>=-len(img)-1:
            img[(int(avg-k)-1)*repet:int(avg-k)*repet,i]+=255
    return img
def main_0(num):
    import fc
    #生成结果，保存图片
    for i in range(min(num,len(c)//10)):
        img=img_get_3(c[i*10:(i+100)*10])
        imgf=FFD_2d(img)
        imgf=FFD_2d(img)
        #fc.plot([img[:,-1],np.real(imgf[:,-1])],k=1,zoom='auto')
    return
def FFD_2d_sect_save_to_csv():
    #保存数据到csv文件
    from joblib import parallel_backend
    FFD_2d_sect_mean=np.zeros(len(c))
    FFD_2d_sect_diff=np.zeros(len(c))
    with parallel_backend('threading', n_jobs=14):
        for i in range(1000,len(c)):
            img=img_get_3(c[i-1000:i])#k线图
            imgf=FFD_2d(img,not_plot=1)#滤波后的k线图
            imgf_std=np.real(imgf[:,-1])
            imgf_std = (imgf_std - np.mean(imgf_std)) / np.std(imgf_std)
            d_sect=imgf_std*img[:,-1]/np.max(img[:,-1])#小切片
            FFD_2d_sect_mean[i]=np.mean(d_sect[d_sect != 0])
            FFD_2d_sect_diff[i]=(d_sect[d_sect != 0][-1]-d_sect[d_sect != 0][0])/len(d_sect[d_sect != 0])
            if i%1000==0:
                print(i)
    df['FFD_2d_sect_mean_7035'] = FFD_2d_sect_mean
    df['FFD_2d_sect_diff_7035'] = FFD_2d_sect_diff
    df.to_csv('.\data\c_min.csv', index=False)
    return
