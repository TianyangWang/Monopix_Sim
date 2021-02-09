import numpy as np
import math
import matplotlib.pyplot as plt

def ana_pileup(PIXEL_SIZE = 36.0*40.0, HIT_RATE = 1, DEAD_TIME = 400, PARALYZABLE = False):

    ################################################
    # PIXEL_SIZE        um2                        #
    # HIT_RATE          MHz/cm2                    #
    # DEAD_TIME         ns                         #
    ################################################

    hit_rate_per_pix = HIT_RATE*PIXEL_SIZE  # in Hz
    if PARALYZABLE == True:
        data_loss = (1 - math.exp(-hit_rate_per_pix*DEAD_TIME/1000000000)) # data loss for non-paralyzable system# data loss for non-paralyzable system
    else: data_loss = (1 - 1/(1+hit_rate_per_pix*DEAD_TIME/1000000000)) # data loss for paralyzable system
    return data_loss



if __name__ == "__main__":





    sim_digtalpileup = [0.0093358, 0.021119, 0.02626315, 0.03635547, 0.04915524, 0.06508328, 0.0838159, 0.11404344]
    dead_time=[ 25*4.35591436,  25*4.58679065,  25*4.8640348,   25*5.20141387,  25*5.63218055,  25*6.21720143,  25*6.97629024,  25*8.09351997]
#     dead_time = np.arange(100,101,1)
    hit_rate = np.arange(1, 5, 0.5)
    data_loss = np.empty((1,len(hit_rate)), dtype = np.float)
#     data_loss = np.empty((len(dead_time),len(hit_rate)), dtype = np.float)
    print data_loss
    for i in range(1):
        for j in range(len(hit_rate)):
            data_loss[i][j] = ana_pileup(HIT_RATE=hit_rate[j], DEAD_TIME=dead_time[j])
        plt.plot(hit_rate, data_loss[i]*100, 'x-', label = 'theory')
#    for i in range(len(dead_time)):
#         for j in range(len(hit_rate)):
#             data_loss[i][j] = ana_pileup(HIT_RATE=hit_rate[j], DEAD_TIME=dead_time[i])
#         plt.plot(hit_rate, data_loss[i]*100, 'x-', label = 'dead time: %f ns'%dead_time[i])
    plt.plot(hit_rate, sim_digtalpileup, 'gs-', label = 'simulated')
    plt.ylabel('%')
    plt.xlabel('hit rate (100 MHz/cm2)')
#     plt.title('Analog pileup')
    plt.legend()
    plt.show()