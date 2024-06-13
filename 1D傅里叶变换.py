import 数据读取,尾数统计
import fc
c=数据读取.get_close()
c=c[-3000:]
p100=尾数统计.tail(c,300)[0]
fc.FFT(p100)

c_=[c[i]-i*0.01 for i in range(len(c))]
fc.plot(c_)
fc.FFT(尾数统计.tail(c_,100)[0])
