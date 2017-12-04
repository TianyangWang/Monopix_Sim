import numpy as np
#import random
#import numba
import matplotlib.pyplot as plt
#import yaml
import pickle
import time
import arch_sim
from analog_pileup import ana_pileup
from numba.tests.npyufunc.test_ufunc import dtype
from multiprocessing import Pool
from matplotlib.legend import Legend

#with open('data.yml', 'w') as outfile:
#    yaml.dump((analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, to_read_mon, col_ro_delay_hist), outfile)
#print 'finish writing data.yml'

#with open('datapickle', 'w') as outfile:
#    pickle.dump((analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, to_read_mon, col_ro_delay_hist), outfile)
#print 'finish writing to datapickle'

#with open('data.yml', 'r') as infile:
#    analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, to_read_mon, col_ro_delay_hist = yaml.load(infile)

#with open('datapickle', 'r') as infile:
#    analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, to_read_mon, col_ro_delay_hist = pickle.load(infile)

def func_arg(arg):
    return arch_sim.monopix_sim(**arg)

bf = time.time()

SIM_TIME = 50000

start_hit_rate = 1# *100MHz/cm2
top_hit_rate =  5# *100MHz/cm2
hit_rate_step = 0.5
hit_rate_ar = np.arange(start_hit_rate,(top_hit_rate + hit_rate_step), hit_rate_step)
print hit_rate_ar
len_hit_rate_ar = len(hit_rate_ar)

pileup_a = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
pileup_d = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
pileup_trig_mem = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
late_copy_loss = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
total_loss = np.zeros((len_hit_rate_ar), dtype=np.float64)

pool = Pool()
kw = []


for hit_rate in hit_rate_ar:
    kw.append({'SIM_TIME': SIM_TIME,
       'LATENCY': 400,
       'TRIGGER_RATE': 4.0/40,
       'PIXEL_AREA': 36.0*40.0, #36.0*40.0,
       'READ_COL': 4,
       'LOGIC_COLUMNS': 512/2, #512/4,
       'PIXEL_NO': 512*2, #512*4,
       'HIT_RATE_CM': hit_rate*100*(10**6),
       'MEAN_TOT': 15,
       'READ_TRIG_MEM': 32,
       'TRIG_MEM_SIZE': 256,
       'READ_OUT_FIFO': 32,
       'OUT_FIFO_SIZE': 128
       })

if 1:
    ret = pool.map(func_arg, kw)

    print 'finish..'

    with open('data.pickle', 'w') as outfile:
        pickle.dump(ret, outfile)

print 'pickle..'

with open('data.pickle', 'r') as infile:
    ret = pickle.load(infile)


for i,hit_rate  in enumerate(hit_rate_ar):
    print 'run%d for hit rate %f'%(i, hit_rate)

    #analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, to_read_mon, col_ro_delay_hist = arch_sim.monopix_sim(**kwa)

    analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, out_fifo_fill_mon, to_read_mon, col_ro_delay_hist, hits_per_bx = ret[i]

    print 'analog pileup @ %0.1f*100MHz/cm2:   %0.4f %s' % (hit_rate, (100*analog_pileup/total_hits), '%')
    print 'digital pileup @ %0.1f*100MHz/cm2:  %0.4f %s' % (hit_rate, (100*digital_pileup/total_hits), '%')
    print 'late copy @ %0.1f*100MHz/cm2:       %0.4f %s' % (hit_rate, (100*late_copy/total_hits), '%')
    print 'trig mem pileup @ %0.1f*100MHz/cm2: %0.4f %s' % (hit_rate, (100*trig_mem_pileup/total_hits), '%')
    print 'total hits @ %0.1f*100MHz/cm2:      '%hit_rate, total_hits
    print 'trig count @ %0.1f*100MHz/cm2:      '%hit_rate, trig_count
#     print 'Output FIFO histo: ', out_fifo_hist
    pileup_a[i] = 100*analog_pileup/total_hits
    pileup_d[i] = 100*digital_pileup/total_hits
    late_copy_loss[i] = 100*late_copy/total_hits
    pileup_trig_mem[i] = 100*trig_mem_pileup/total_hits
    total_loss[i] = 100*(analog_pileup + digital_pileup + late_copy + trig_mem_pileup)/total_hits
#    print hist_col_ro_delay[hit_rate-1]
#    print 'histogram trigger mem: ', trig_mem_hist
    print 'finish run %d for hit rate %f MHz/cm2'%(i, hit_rate)
    print '-------------------------------------------'

print 'plotting...'
print len(ret[0][11])

print 'digital pileup', digital_pileup





# data_loss = np.empty((len(hit_rate_ar)), dtype = np.float)
# dead_time = 375
# for i in range(len(hit_rate_ar)):
#     data_loss[i] = ana_pileup(HIT_RATE=hit_rate_ar[i], DEAD_TIME=dead_time)
# plt.plot(hit_rate_ar, data_loss*100, 'x-', label = 'Theoretical')



# plt.subplot(121)
# plt.plot(hit_rate_ar, pileup_a, 'r*-', label = 'Simulated')
# plt.plot(hit_rate_ar, pileup_d, 'gs-', label = 'Digital pileup')
# plt.plot(hit_rate_ar, late_copy_loss, 'bx-', label = 'Data loss due to late copy')
# plt.plot(hit_rate_ar, pileup_trig_mem, 'y+-', label = 'Trig memory pileup')
# plt.plot(hit_rate_ar, total_loss, 'k<-', label = 'Total data loss')
# plt.legend()
# plt.title('Data loss')
# plt.ylabel('%')
# plt.xlabel('Hit rate (100 MHz/cm2)')
# 
# #plt.clf()
# #
# plt.subplot(122)
# for i in range(len(hit_rate_ar)-1):
#     plt.bar(range(len(ret[i][6])),ret[i][6], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
# plt.legend()
# plt.title("Trigger memory occupancy")
# plt.ylabel('#')
# plt.xlabel('Occupied memeory')
# #

mean_dead_time = np.zeros((len(hit_rate_ar)), dtype=np.float)
plt.subplot(223)
for i in range(len(hit_rate_ar)):
    plt.bar(range(len(ret[i][11])-15),ret[i][11][15:]/np.sum(ret[0][11], dtype=np.float), label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
    print ret[i][11][15:]
    wait_time = range(len(ret[i][11])-15)
    print wait_time
    mean_dead_time[i] = np.average((wait_time), weights=ret[i][11][15:])
print 'mean dead time', mean_dead_time
plt.legend()
plt.title("Column RO delay")
plt.ylabel('#')
plt.yscale('log')
plt.xlabel('BX')


# 
# plt.subplot(222)
# for i in range(len(hit_rate_ar)-1):
#     plt.hist(ret[i][12], bins = 78, range = [1, 80], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
# plt.legend()
# plt.title("Hits per BX")
# plt.ylabel('#')
# plt.xlabel('BX')

# for i in range(len(hit_rate_ar)):
#     plt.plot(range(SIM_TIME),ret[i][8][0], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
# plt.legend()
# plt.title("Trigger memory fill per BX")
# plt.ylabel('#')
# plt.xlabel('BX')

# plt.subplot(224)
# for i in range(len(hit_rate_ar)):
#     plt.plot(range(SIM_TIME),ret[i][9], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
# plt.legend()
# plt.title("Output FIFO monitor")
# plt.ylabel('#')
# plt.xlabel('BX')

#plt.subplot(325)
#for i in range(len(hit_rate_ar)):
#    plt.plot(range(64),ret[i][11], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
# plt.legend()
#plt.title("Pixel readout delay (late copy not excluded)")
#plt.ylabel('#')
#plt.xlabel('BX')

#plt.subplot(325)
#for i in range(len(hit_rate_ar)):
#    plt.bar(range(len(ret[i][5])),ret[i][5], label = '%0.1f*100MHits/s/cm2'%(1+i*hit_rate_step))
#plt.legend()
#plt.title("Output FIFO occupancy")
#plt.ylabel('#')
#plt.xlabel('Occupied FIFO')

# plt.subplots_adjust(hspace = 0.4)

print 'Run Time %0.9fm' % ((time.time()-bf)/60)

plt.show()
