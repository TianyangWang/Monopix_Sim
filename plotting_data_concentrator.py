import numpy as np
#import random
#import numba
import matplotlib.pyplot as plt
#import yaml
import pickle
import time
import arch_sim_data_concentrator
# from analog_pileup import ana_pileup
from numba.tests.npyufunc.test_ufunc import dtype
from multiprocessing import Pool
from matplotlib.legend import Legend

#with open('data.yml', 'w') as outfile:
#    yaml.dump((analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, out_fifo_hist, eoc_fifo_hist, trig_count, eoc_fifo_fill_mon, to_read_mon, col_ro_delay_hist), outfile)
#print 'finish writing data.yml'

#with open('datapickle', 'w') as outfile:
#    pickle.dump((analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, out_fifo_hist, eoc_fifo_hist, trig_count, eoc_fifo_fill_mon, to_read_mon, col_ro_delay_hist), outfile)
#print 'finish writing to datapickle'

#with open('data.yml', 'r') as infile:
#    analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, out_fifo_hist, eoc_fifo_hist, trig_count, eoc_fifo_fill_mon, to_read_mon, col_ro_delay_hist = yaml.load(infile)

#with open('datapickle', 'r') as infile:
#    analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, out_fifo_hist, eoc_fifo_hist, trig_count, eoc_fifo_fill_mon, to_read_mon, col_ro_delay_hist = pickle.load(infile)

def func_arg(arg):
    return arch_sim_data_concentrator.monopix_sim(**arg)

bf = time.time()

SIM_TIME = 50000

start_hit_rate = 1# *100MHz/cm2
top_hit_rate =  2# *100MHz/cm2
hit_rate_step = 0.5
hit_rate_ar = np.arange(start_hit_rate,(top_hit_rate + hit_rate_step), hit_rate_step)
print hit_rate_ar
len_hit_rate_ar = len(hit_rate_ar)

pileup_a = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
pileup_d = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
pileup_eoc_fifo = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
late_copy_loss = np.zeros((len_hit_rate_ar) ,dtype=np.float64)
total_loss = np.zeros((len_hit_rate_ar), dtype=np.float64)

pool = Pool()
kw = []


for hit_rate in hit_rate_ar:
    kw.append({'SIM_TIME': SIM_TIME,
        'EOC_FIFO_LATENCY': 63,
        'LATENCY': 400,
        'TRIGGER_RATE': 4.0/40,
        'PIXEL_AREA': 36.0*36.0,
        'ROW': 512,
        'COL': 512,
        'LOGIC_COL': 512/2,
        'COL_REGION': 32,
        'HIT_RATE_CM': hit_rate*100*(10**6),
        'MEAN_TOT': 15,
        'READ_COL': 2,
        'READ_EOC_FIFO': 1,
        'EOC_FIFO_SIZE': 8,
        'READ_TRIG_MEM': 4,
        'TRIG_MEM_SIZE': 1024,
        'READ_OUT_FIFO': 4,
        'OUT_FIFO_SIZE': 96
        })

if 1:
    ret = pool.map(func_arg, kw)

    print 'finish..'

    with open('data.pickle', 'w') as outfile:
        pickle.dump(ret, outfile)

print 'pickle..'

with open('data.pickle', 'r') as infile:
    ret = pickle.load(infile)

sim_out = []
for i,hit_rate  in enumerate(hit_rate_ar):
    print 'run%d for hit rate %f'%(i, hit_rate)

    sim_out.append({'ANALOG_PILEUP': ret[i][0],
        'DIGITAL_PILEUP': ret[i][1],
        'LATE_COPY': ret[i][2],
        'EOC_FIFO_PILEUP': ret[i][3],
        'TOTAL_HITS': ret[i][4],
        'HITS_PER_BX': ret[i][5],
        'TRIG_COUNT': ret[i][6],
        'EOC_FIFO_HIST': ret[i][7],
        'TRIG_MEM_HIST': ret[i][8],
        'OUT_FIFO_HIST': ret[i][9],
        'COL_RO_DELAY': ret[i][10],
        'EOC_FIFO_FILL_MON': ret[i][11],
        'TRIG_MEM_FILL_MON': ret[i][12],
        'OUT_FIFO_FILL_MON': ret[i][13],
        'TO_READ_EOC_FIFO_MON': ret[i][14],
        'TO_READ_TRIG_MEM_MON': ret[i][15]
        })

    analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, hits_per_bx, trig_count, eoc_fifo_hist, trig_mem_hist, out_fifo_hist, col_ro_delay_hist, eoc_fifo_fill_mon, trig_mem_fill_mon, out_fifo_fill_mon, to_read_eoc_fifo_mon, to_read_trig_mem_mon = ret[i]

    print 'analog pileup @ %0.1f*100MHz/cm2:   %0.4f %s' % (hit_rate, (100*analog_pileup/total_hits), '%')
    print 'digital pileup @ %0.1f*100MHz/cm2:  %0.4f %s' % (hit_rate, (100*digital_pileup/total_hits), '%')
    print 'late copy @ %0.1f*100MHz/cm2:       %0.4f %s' % (hit_rate, (100*late_copy/total_hits), '%')
    print 'trig mem pileup @ %0.1f*100MHz/cm2: %0.4f %s' % (hit_rate, (100*eoc_fifo_pileup/total_hits), '%')
    print 'total hits @ %0.1f*100MHz/cm2:      '%hit_rate, total_hits
    print 'trig count @ %0.1f*100MHz/cm2:      '%hit_rate, trig_count

    pileup_a[i] = 100*analog_pileup/total_hits
    pileup_d[i] = 100*digital_pileup/total_hits
    late_copy_loss[i] = 100*late_copy/total_hits
    pileup_eoc_fifo[i] = 100*eoc_fifo_pileup/total_hits
    total_loss[i] = 100*(analog_pileup + digital_pileup + late_copy + eoc_fifo_pileup)/total_hits
    print 'finish run %d for hit rate %f MHz/cm2'%(i, hit_rate)
    print '-------------------------------------------'

print 'plotting...'


############### Theoretical pileup ###################
# data_loss = np.empty((len(hit_rate_ar)), dtype = np.float)
# dead_time = 375
# for i in range(len(hit_rate_ar)):
#     data_loss[i] = ana_pileup(HIT_RATE=hit_rate_ar[i], DEAD_TIME=dead_time)
# plt.plot(hit_rate_ar, data_loss*100, 'x-', label = 'Theoretical')



plt.subplot(221)
plt.plot(hit_rate_ar, pileup_a, 'r*', label = 'analog pileup')
plt.plot(hit_rate_ar, pileup_d, 'gs', label = 'Digital pileup')
plt.plot(hit_rate_ar, late_copy_loss, 'bx', label = 'Data loss due to late copy')
plt.plot(hit_rate_ar, pileup_eoc_fifo, 'y+', label = 'Eoc fifo pileup')
plt.plot(hit_rate_ar, total_loss, 'k<', label = 'Total data loss')
plt.legend()
plt.title('Data loss')
plt.ylabel('%')
plt.xlabel('Hit rate (100 MHz/cm2)')
#
# #plt.clf()
# #
plt.subplot(222)
for i in range(len(hit_rate_ar)):
    plt.step(range(len(sim_out[i]['EOC_FIFO_HIST'])),sim_out[i]['EOC_FIFO_HIST'], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
plt.legend()
plt.title("Eoc fifo occupancy")
plt.ylabel('#')
plt.xlabel('Occupied eoc fifo')
#
#
# mean_dead_time = np.zeros((len(hit_rate_ar)), dtype=np.float)
# plt.subplot(223)
# for i in range(len(hit_rate_ar)):
#     plt.bar(range(len(ret[i][11])-15),ret[i][11][15:]/np.sum(ret[0][11], dtype=np.float), label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
#     print ret[i][11][15:]
#     wait_time = range(len(ret[i][11])-15)
#     print wait_time
#     mean_dead_time[i] = np.average((wait_time), weights=ret[i][11][15:])
# print 'mean dead time', mean_dead_time
# plt.legend()
# plt.title("Column RO delay")
# plt.ylabel('#')
# plt.yscale('log')
# plt.xlabel('BX')


#
plt.subplot(223)
for i in range(len(hit_rate_ar)):
    plt.step(range(len(sim_out[i]['COL_RO_DELAY'])), sim_out[i]['COL_RO_DELAY'], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
plt.legend()
plt.title("Column readout delay")
plt.ylabel('#')
plt.xlabel('BX')

plt.subplot(224)
for i in range(len(hit_rate_ar)):
    plt.step(range(len(sim_out[i]['TRIG_MEM_HIST'])),sim_out[i]['TRIG_MEM_HIST'], label = '%0.1f*100MHits/s/cm2'%(start_hit_rate+i*hit_rate_step))
plt.legend()
plt.title('Trigger memory occupancy')
plt.ylabel('#')
plt.xlabel('Occupied trigger memory')

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

plt.subplots_adjust(hspace = 0.4)

print 'Run Time %0.9fm' % ((time.time()-bf)/60)

plt.show()
