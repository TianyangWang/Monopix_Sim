import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from arch_sim_MC import monopix_sim
from numba.tests.npyufunc.test_ufunc import dtype
from matplotlib.legend import Legend


bf = time.time()


#print (36.4*36.4*512*2)/(1000*1000)
mc_hits = np.load('sim_mc_pu200_36x36.npy')
print('MC data type: {}'.format(mc_hits.dtype))
total_event = np.amax(mc_hits['bcid'])
print 'total event: {}'.format(total_event)
mc_trigger = np.zeros((total_event + 1), dtype = bool)

# module 1 - 8 are quad, module 9 - 21 are double (inclined)
module_num = 1
module_hits = mc_hits[mc_hits['eta_module'] == module_num]

#eta_index = col, phi_index = row
row = 512
col = 512
chip_hits = module_hits[(module_hits['eta_index'] < col) & (module_hits['phi_index'] < row)]

print 'initializing hits...'
hit_dtype = np.dtype([('bcid', np.uint64),
                      ('eta_index', np.int16),
                      ('phi_index', np.int16),
                      ('charge', np.float64)
                      ])
hits = np.empty((chip_hits.size,), dtype=hit_dtype)
hits['bcid'] = chip_hits['bcid']
hits['eta_index'] = chip_hits['eta_index']
hits['phi_index'] = chip_hits['phi_index']
hits['charge'] = chip_hits['charge']
print 'hits data type', hits.dtype

SIM_TIME = np.amax(hits['bcid'])

with open('mc_trigger.pickle', 'r') as infile:
    mc_trigger = pickle.load(infile)



kwa = {'CHIP_HITS': hits,
       'TRIGGER': mc_trigger,
       'LATENCY': 400,
       'TRIGGER_RATE': 4.0/40,
       'PIXEL_AREA': 36.0*40.0,
       'READ_COL': 2,
       'ROW': row,
       'COL': col,
       'LOGIC_COL': col/2,
       'READ_TRIG_MEM': 4,
       'TRIG_MEM_SIZE': 128,
       'READ_OUT_FIFO': 4,
       'OUT_FIFO_SIZE': 8
       }
sim_out_kwlist = ['analog_pileup',
                'digital_pileup',
                'late_copy',
                'trig_mem_pileup',
                'total_hits',
                'out_fifo_hist',
                'trig_mem_hist',
                'tot_hist',
                'trig_count',
                'trig_mem_fill_mon',
                'out_fifo_fill_mon',
                'to_read_trig_mem_mon',
                'col_ro_delay_hist',
                'hits_per_bx',
                'hit_rate']
sim_out = {kw: None for kw in sim_out_kwlist}

if 1:
    sim_out = monopix_sim(**kwa)
    print 'saving data...'
    with open('data_MC.pickle', 'w') as outfile:
        pickle.dump(sim_out, outfile)

print 'pickle...'
with open('data_MC.pickle', 'r') as infile:
    sim_out = pickle.load(infile)

print '-------------------------------------------'
# print sim_out
print ('analog pileup:     {}%'.format(100*sim_out['analog_pileup']/sim_out['total_hits']))
print ('digital pileup:    {}%'.format(100*sim_out['digital_pileup']/sim_out['total_hits']))
print ('late copy:         {}%'.format(100*sim_out['late_copy']/sim_out['total_hits']))
print ('trig mem pileup:   {}%'.format(100*sim_out['trig_mem_pileup']/sim_out['total_hits']))
print ('total hits:        {}'.format(sim_out['total_hits']))
print ('trig count:        {}'.format(sim_out['trig_count']))
print ('hit rate:          {}MHz/cm2').format(sim_out['hit_rate'])
print '-------------------------------------------'

print 'plotting...'

#plt.subplot(211)
#plt.plot(hit_rate_ar, pileup_a, 'r*-', label = 'Analog pileup')
#plt.plot(hit_rate_ar, pileup_d, 'gs-', label = 'Digital pileup')
#plt.plot(hit_rate_ar, late_copy_loss, 'bx-', label = 'Data loss due to late copy')
#plt.plot(hit_rate_ar, pileup_trig_mem, 'y+-', label = 'Trig memory pileup')
#plt.plot(hit_rate_ar, total_loss, 'k', label = 'Total data loss')
#plt.legend()
#plt.title('Data loss')
#plt.ylabel('%')
#plt.xlabel('Hit rate (100 MHz/cm2)')

#plt.clf()
#print 'cc\n', sim_out[0][6]
plt.subplot(221)
plt.plot(sim_out['hit_rate'], 100*sim_out['analog_pileup']/sim_out['total_hits'], 'r*-', label = 'Analog pileup')
plt.plot(sim_out['hit_rate'], 100*sim_out['digital_pileup']/sim_out['total_hits'], 'gs-', label = 'Digital pileup')
plt.plot(sim_out['hit_rate'], 100*sim_out['late_copy']/sim_out['total_hits'], 'bx-', label = 'Data loss due to late copy')
plt.plot(sim_out['hit_rate'], 100*sim_out['trig_mem_pileup']/sim_out['total_hits'], 'y+-', label = 'Trig memory pileup')
plt.plot(sim_out['hit_rate'], 100*(sim_out['analog_pileup'] + sim_out['digital_pileup'] + sim_out['late_copy'] + sim_out['trig_mem_pileup'])/sim_out['total_hits'], 'k<-', label = 'Total data loss')
plt.legend()
plt.title('Data loss')
plt.ylabel('%')
plt.xlabel('Hit rate (100 MHz/cm2)')

plt.subplot(222)
plt.bar(range(len(sim_out['trig_mem_hist'])),sim_out['trig_mem_hist'], label = 'MC data')
# plt.legend()
plt.title("Trigger memory occupancy")
plt.ylabel('#')
plt.xlabel('Occupied memeory')

plt.subplot(223)
plt.bar(range(len(sim_out['col_ro_delay_hist'])), sim_out['col_ro_delay_hist'], label = 'MC data')
plt.ylabel('#')
plt.title('col ro delay')


plt.subplot(224)
plt.hist(sim_out['hits_per_bx'], bins = 78, range = [1, 80], label = 'MC data')
plt.title("Hits per BX")
plt.xlabel('Hits')
plt.ylabel('#')
# plt.plot(range(SIM_TIME),sim_out['trig_mem_fill_mon'][0], label = 'MC data')
# #plt.legend()
# plt.title("Trigger memory fill per BX")
# plt.ylabel('#')
# plt.xlabel('BX')

# plt.subplot(325)
# plt.plot(range(SIM_TIME),sim_out['out_fifo_fill_mon'], label = 'MC data')
# #plt.legend()
# plt.title("Output FIFO monitor")
# plt.ylabel('#')
# plt.xlabel('BX')

# plt.subplot(326)
# plt.bar(range(len(sim_out['tot_hist'])), sim_out['tot_hist'], label = 'MC data')
# plt.title('ToT histogram')
#plt.subplot(325)
#for i in range(len(hit_rate_ar)):
#    plt.plot(range(64),sim_out[i][11], label = '%0.1f*100MHits/s/cm2'%(1+i*hit_rate_step))
plt.subplots_adjust(hspace = 0.4)
plt.legend()
#plt.title("Pixel readout delay (late copy not excluded)")
#plt.ylabel('#')
#plt.xlabel('BX')



#plt.subplot(325)
#for i in range(len(hit_rate_ar)):
#    plt.bar(range(len(sim_out[i][5])),sim_out[i][5], label = '%0.1f*100MHits/s/cm2'%(1+i*hit_rate_step))
#plt.legend()
#plt.title("Output FIFO occupancy")
#plt.ylabel('#')
#plt.xlabel('Occupied FIFO')

print 'Run Time %0.9fm' % ((time.time()-bf)/60)

plt.show()

#for i in range(top_hit_rate):
#    plt.bar(hit_rate, hist_col_ro_delay[i], label = 'rate=%d'%i)
#leg = plt.legend(loc='best', ncol=10, mode="expand", shadow=True, fancybox=True)
#plt.title('TRIG_MEM_FILL')
#plt.ylabel('FILLED MEM')
#plt.xlabel('BX')
#plt.show()
