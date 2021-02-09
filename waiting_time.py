import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def p_wait_time(LAMDA = 0.5, READ_COL = 2, t = 0):    
    T = int(math.floor(t/READ_COL))
#    print T
    sum = 0
    for v in range(T+1):
        sum += (((LAMDA*(v*READ_COL - t))**v)/math.factorial(v))*math.exp(-LAMDA*(v*READ_COL - t))
#        print v, sum
    p = (1 - LAMDA*READ_COL)*sum
    return p
    
if __name__ == "__main__":
    HIT_RATE_CM = 2.5*100*(10**6)
    COL_AREA = 36.0*40.0*512*2
    READ_COL = 4

    
    col_hit_rate_bx = ((float(COL_AREA)/(10000*10000)) * HIT_RATE_CM )/ (40*(10**6))
    print col_hit_rate_bx
    
    p_cum = np.zeros((60), dtype = np.float64)
    p = np.zeros((60), dtype = np.float64)
    wait_time = np.arange(0,60,1.0)
#     print p_cum, wait_time*READ_COL
    for i,t in enumerate(wait_time):
#        print i, t
        p_cum[i] = p_wait_time(LAMDA = col_hit_rate_bx, READ_COL = READ_COL, t = t)
        if i == 0:
            p[i] = p_cum[i]
        else: p[i] = p_cum[i] - p_cum[i-1]
#         print p_cum[i], p[i]
# 	print np.mean(p)
    mean_dead_time = np.average((wait_time + 3), weights=p)
    print wait_time, p
    print mean_dead_time
    plt.bar(wait_time+3, p)
    plt.yscale('log')
    plt.show()
