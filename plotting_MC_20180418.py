import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
from arch_sim_MC_20180418 import monopix_sim
from numba.tests.npyufunc.test_ufunc import dtype
from matplotlib.legend import Legend
from matplotlib.pyplot import hist


bf = time.time()


#print (36.4*36.4*512*2)/(1000*1000)
mc_hits = np.load('pixel_hits-36x36x25.npy')
print('MC data type: {}'.format(mc_hits.dtype))
total_event = np.amax(mc_hits['bcid'])
print 'total event: {}'.format(total_event)
mc_trigger = np.zeros((total_event + 1), dtype = bool)

# module 1 - 8 are quad, module 9 - 21 are double (inclined)
for module_num in range(26, 27):
    module_hits = mc_hits[mc_hits['eta_module'] == module_num]

    #eta_index = col, phi_index = row
    row = 512
    col = 512
    chip_hits = module_hits[(module_hits['eta_index'] < col) & (module_hits['phi_index'] < row)]

    threshold = 300
    noise_scale = 10
    sel_chip_hits = chip_hits[np.where((chip_hits['charge_track'] + chip_hits['charge_noise']/noise_scale) > threshold)] # select hits over threshold
    #

    print 'initializing hits...'
    hit_dtype = np.dtype([('bcid', np.uint64),
                          ('eta_index', np.int16),
                          ('phi_index', np.int16),
                          ('charge', np.float64)
                          ])
    hits = np.empty((sel_chip_hits.size,), dtype=hit_dtype)
    hits['bcid'] = sel_chip_hits['bcid']
    hits['eta_index'] = sel_chip_hits['eta_index']
    hits['phi_index'] = sel_chip_hits['phi_index']
    hits['charge'] = sel_chip_hits['charge_track'] + sel_chip_hits['charge_noise']/noise_scale


    print 'hits data type', hits.dtype

    SIM_TIME = np.amax(hits['bcid'])

    with open('mc_trigger.pickle', 'r') as infile:
        mc_trigger = pickle.load(infile)



    kwa = {'CHIP_HITS': hits,
           'TRIGGER': mc_trigger,
           'LATENCY': 400,
           'TRIGGER_RATE': 4.0/40,
           'PIXEL_AREA': 36.0*36.0,
           'READ_COL': 2,
           'ROW': row,
           'COL': col,
           'LOGIC_COL': col/2,
           'READ_TRIG_MEM': 2,
           'TRIG_MEM_SIZE': 128,
           'READ_OUT_FIFO': 4,
           'OUT_FIFO_SIZE': 32
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
                    'hit_rate',
                    'pix_hit_map']
    sim_out = {kw: None for kw in sim_out_kwlist}

    if 1:
        sim_out = monopix_sim(**kwa)
        print 'saving data...'
        with open('data_no_tot_limit_module_%s.pickle' %module_num, 'w') as outfile:
            pickle.dump(sim_out, outfile)

    print 'pickle...'
    with open('data_no_tot_limit_module_%s.pickle' %module_num, 'r') as infile:
        sim_out = pickle.load(infile)

    print '-------------------------------------------'
    # print sim_out
    analog_pileup = 100*sim_out['analog_pileup']/sim_out['total_hits']
    digital_pileup = 100*sim_out['digital_pileup']/sim_out['total_hits']
    late_copy = 100*sim_out['late_copy']/sim_out['total_hits']
    trig_mem_pileup = 100*sim_out['trig_mem_pileup']/sim_out['total_hits']
    total_loss = analog_pileup + digital_pileup + late_copy + trig_mem_pileup
    print ('analog pileup for module {}:     {}%'.format(module_num, analog_pileup))
    print ('digital pileup for module {}:    {}%'.format(module_num, digital_pileup))
    print ('late copy for module {}:         {}%'.format(module_num, late_copy))
    print ('trig mem pileup for module {}:   {}%'.format(module_num, trig_mem_pileup))
    print ('total data loss for module {}:   {}%'.format(module_num, total_loss))
    print ('total hits for module {}:        {}'.format(module_num, sim_out['total_hits']))
    print ('trig count for module {}:        {}'.format(module_num, sim_out['trig_count']))
    print ('hit rate for module {}:          {}MHz/cm2').format(module_num, sim_out['hit_rate'])

    total_tot = 0
    for i in range(len(sim_out['tot_hist'])):
        total_tot += i*sim_out['tot_hist'][i]
    mean_tot = total_tot/sim_out['total_hits']
    print ('mean ToT for module {}:          {}').format(module_num, mean_tot)
    print '-------------------------------------------'

print 'plotting...'

# plt.subplot(211)
# plt.plot(sim_out['hit_rate'], analog_pileup, 'r*-', label = 'Analog pileup')
# plt.plot(sim_out['hit_rate'], digital_pileup, 'gs-', label = 'Digital pileup')
# plt.plot(sim_out['hit_rate'], late_copy, 'bx-', label = 'Data loss due to late copy')
# plt.plot(sim_out['hit_rate'], trig_mem_pileup, 'y+-', label = 'Trig memory pileup')
# plt.plot(sim_out['hit_rate'], total_loss, 'k', label = 'Total data loss')
# plt.legend()
# plt.title('Data loss')
# plt.ylabel('%')
# plt.xlabel('Hit rate (100 MHz/cm2)')

#plt.clf()
#print 'cc\n', sim_out[0][6]
# plt.subplot(221)
# plt.plot(sim_out['hit_rate'], 100*sim_out['analog_pileup']/sim_out['total_hits'], 'r*-', label = 'Analog pileup')
# plt.plot(sim_out['hit_rate'], 100*sim_out['digital_pileup']/sim_out['total_hits'], 'gs-', label = 'Digital pileup')
# plt.plot(sim_out['hit_rate'], 100*sim_out['late_copy']/sim_out['total_hits'], 'bx-', label = 'Data loss due to late copy')
# plt.plot(sim_out['hit_rate'], 100*sim_out['trig_mem_pileup']/sim_out['total_hits'], 'y+-', label = 'Trig memory pileup')
# plt.plot(sim_out['hit_rate'], 100*(sim_out['analog_pileup'] + sim_out['digital_pileup'] + sim_out['late_copy'] + sim_out['trig_mem_pileup'])/sim_out['total_hits'], 'k<-', label = 'Total data loss')
# plt.legend()
# plt.title('Data loss')
# plt.ylabel('%')
# plt.xlabel('Hit rate (100 MHz/cm2)')
#
# plt.subplot(222)
# plt.bar(range(len(sim_out['trig_mem_hist'])),sim_out['trig_mem_hist'], label = 'MC data')
# plt.legend()
# plt.title("Trigger memory occupancy")
# plt.ylabel('#')
# plt.xlabel('Occupied memeory')
#
# plt.subplot(223)
# plt.bar(range(len(sim_out['col_ro_delay_hist'])), sim_out['col_ro_delay_hist'], label = 'MC data')
# plt.ylabel('#')
# plt.title('col ro delay')
#
# plt.subplot(224)
# plt.bar(range(len(sim_out['tot_hist'])), sim_out['tot_hist'], label = 'MC data')
# plt.title('ToT histogram')
#
# plt.hist(sim_out['hits_per_bx'], bins = 199, range = [1, 200], histtype = 'step', label = 'MC data')
# plt.yscale('log', nonposy='clip')
# plt.title('Hits per BX')

# plt.hist(sim_out['hits_per_bx'], bins = 78, range = [1, 80], label = 'MC data')
# plt.title("Hits per BX")
# plt.xlabel('Hits')
# plt.ylabel('#')

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


# plt.subplots_adjust(hspace = 0.4)
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

# plt.show()

#for i in range(top_hit_rate):
#    plt.bar(hit_rate, hist_col_ro_delay[i], label = 'rate=%d'%i)
#leg = plt.legend(loc='best', ncol=10, mode="expand", shadow=True, fancybox=True)
#plt.title('TRIG_MEM_FILL')
#plt.ylabel('FILLED MEM')
#plt.xlabel('BX')
#plt.show()
