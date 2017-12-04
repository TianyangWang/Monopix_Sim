#from __future__ import print_function
import numpy as np
import random
import numba
import pickle
import matplotlib.pyplot as plt
# import logging
# import logging.handlers
# import sys


# consoleHandler = logging.StreamHandler(sys.stdout)
# consoleHandler.setFormatter(logFormatter)
# rootLogger.addHandler(consoleHandler)

#from keyword import kwlist
#import matplotlib.pyplot as plt
#import yaml
#import pickle
#import logging

#logging.basicConfig(level=logging.INFO, format='%(message)s')
#logger = logging.getLogger()
#logger.addHandler(logging.FileHandler('test.log', 'a'))
#print = logger.info

@numba.jit
def monopix_sim(CHIP_HITS, TRIGGER, LATENCY = 400, TRIGGER_RATE = 4.0/40, PIXEL_AREA = 36.0*40.0, READ_COL = 4, ROW = 512, COL = 512, LOGIC_COL = 512/4, READ_TRIG_MEM = 1, TRIG_MEM_SIZE = 96, READ_OUT_FIFO = 1, OUT_FIFO_SIZE = 128):
    # Average hit rate L4: 0.021/mm2/BC
    ##############################################################
    #                      PAREMETERS                            #
    # LATENCY = 400              10us                            #
    # TRIGGER_RATE = 4.0/40      4MHz                            #
    # ROW = 2*512           number of pixle per r.o. unit   #
    # READ_COL = 4               10MHz column bus                #
    # READ_TRIG_MEM = 1          1 - 40MHz / 4 - 160MHz          #
    # READ_OUT_FIFO = 1          1 - 40MHz @ 32bit               #
    # CHIP_HITS                  dtype([('bcid', '<u4'),         #
    #                              ('eta_module', 'u1'),         #
    #                              ('phi_module', 'u1'),         #
    #                              ('eta_index', '<u2'),         #
    #                              ('phi_index', '<u2'),         #
    #                              ('charge', '<f8'),            #
    #                              ('trigger', '?')])            #
    ##############################################################

#     logFormatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]: %(message)s')
#     rootLogger = logging.getLogger()
#     rootLogger.setLevel(logging.DEBUG)
#     fileHandler = logging.handlers.RotatingFileHandler('log_MC.log', maxBytes=2000000, backupCount=3)
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)
    SIM_TIME = np.amax(CHIP_HITS['bcid'])

#     logging.info('Total events: {}'.format(SIM_TIME))

    TOT_SLOPE = 1.0/220               # ToT/e-
    MAX_TOT = int(np.amax(CHIP_HITS['charge'])*TOT_SLOPE)
    MEAN_TOT = 15

#     logging.info('Max ToT: {}'.format(MAX_TOT))

    PIXEL_MAX_LATENCY = 63              # 6-bit ToT
    WAIT_FOR_TRIG_MEM = False           # wait with in the array or read and losse

    active_area = float(PIXEL_AREA)*ROW*COL/(10000*10000)

    hit_rate = (float(CHIP_HITS.size)/np.amax(CHIP_HITS['bcid'])/active_area)*40   # MHz/cm2

    analog_pileup = 0.0
    digital_pileup = 0.0
    late_copy = 0.0
    trig_mem_pileup = 0.0
    total_hits = 0.0
    trig_count = 0

    pix_mem_tot = np.zeros((COL, ROW) ,dtype=np.int32)
    pix_mem_bx = np.zeros((COL, ROW) ,dtype=np.int32)
    trig_mem = np.zeros((LOGIC_COL, TRIG_MEM_SIZE) ,dtype=np.int32)
    out_fifo = np.zeros((OUT_FIFO_SIZE) ,dtype=np.int32)

    out_fifo_hist = np.zeros((OUT_FIFO_SIZE+1) ,dtype=np.uint)
    trig_mem_hist = np.zeros((TRIG_MEM_SIZE+1) ,dtype=np.uint)

    tot_hist = np.zeros((MAX_TOT+1) ,dtype=np.uint)
    trig_mem_fill_mon = np.zeros((LOGIC_COL, SIM_TIME) ,dtype=np.uint)
    out_fifo_fill_mon = np.zeros((SIM_TIME) ,dtype=np.uint)
    to_read_trig_mem_mon = np.zeros((SIM_TIME) ,dtype=np.uint)
    col_ro_delay_hist = np.zeros((PIXEL_MAX_LATENCY+1+MEAN_TOT), dtype=np.uint)

    read_col_cnt = np.zeros((LOGIC_COL) ,dtype=np.uint)
    read_fifo_cnt = 0

    large_tot = 0
    large_delay = 0

    hits_per_bx = np.zeros((SIM_TIME) ,dtype=np.uint)

    for bx in range(SIM_TIME):

        hits_bx = CHIP_HITS[CHIP_HITS['bcid'] == bx]
        hits_per_bx[bx] = hits_bx.size

        print 'bx number: {}'.format(bx)
#         logging.info('hits info: {}'.format(hits_bx))
#         logging.info('total hits for bx {}: {}'.format(bx, hits_per_bx[bx]))

        ###################################################################################
        #                                Output FIFO read                                 #
        ###################################################################################

        #histogram
        out_fifo_hist[len(np.nonzero(out_fifo)[0])] += 1

        len_out_fifo_occ = len(np.where(out_fifo==True)[0]) # occupied output fifo size
#         logging.info('fifo status at bx {}: {} {}'.format(bx, len_out_fifo_occ, np.where(out_fifo==True)[0]))
        out_fifo_fill_mon[bx] += len_out_fifo_occ

        if READ_OUT_FIFO >= 1:
            for read_output_fifo in range(len_out_fifo_occ):
                if read_output_fifo >= READ_OUT_FIFO:
                    break
                out_fifo_occ = np.where(out_fifo==True)[0]
                if len(out_fifo_occ):
                    out_fifo[out_fifo_occ[:1]] = 0
        else:
            if read_fifo_cnt >= (1/READ_OUT_FIFO)-1:
                out_fifo_occ = np.where(out_fifo==True)[0]
                if len(out_fifo_occ):
                    out_fifo[out_fifo_occ[:1]] = 0
                    read_fifo_cnt = 0
            else:
                read_fifo_cnt += 1
        #if read_fifo_cnt >= READ_OUT_FIFO):
        #    out_fifo_occ = np.where(out_fifo==True)[0] #occupied output fifo
        #    if len(out_fifo_occ):
        #        out_fifo[out_fifo_occ[:1]] = 0 #clear fist occupied output fifo
        #        read_fifo_cnt = 0
        #else:
        #    read_fifo_cnt += 1

        ##################################################################################
        #                             trig mem to out fifo                               #
        ##################################################################################
        max_trig_latency = np.max(trig_mem)
#         logging.info('max trig mem latency: {}'.format(max_trig_latency))
#         logging.info('trig mem: {}'.format(trig_mem))
        to_read_trig_mem = np.where((trig_mem > LATENCY) & (trig_mem == max_trig_latency))
        to_read_trig_mem_mon[bx] = len(to_read_trig_mem[0])
#         logging.info('length of trig mem to read: {}'.format(len(to_read_trig_mem[0])))
#         logging.info('trig mem to read: {}'.format(to_read_trig_mem))

        for read_trigger_mem in range(len(to_read_trig_mem[0])):
            if read_trigger_mem >= READ_TRIG_MEM:
                break
            empty_out_fifo = np.where(out_fifo==0)[0]
#             logging.info('empty fifo: {}'.format(empty_out_fifo))
            if len(empty_out_fifo) > 0:
                trig_mem[to_read_trig_mem[0][read_trigger_mem],to_read_trig_mem[1][read_trigger_mem]] = 0 #remove triggered data
#                 logging.info('read trig mem {}{}'.format(to_read_trig_mem[0][read_trigger_mem], to_read_trig_mem[1][read_trigger_mem]))
                out_fifo[empty_out_fifo[:1]] = 1

#         logging.info('occupied fifo after trige mem r.o.: {}'.format(np.where(out_fifo==True)[0]))

#         no_trigger = random.random() > TRIGGER_RATE
#         if no_trigger == False: trig_count += 1
        if TRIGGER[bx] == True: trig_count += 1


        ###################################################################################
        #                          Column Processing                                      #
        ###################################################################################
        for logic_col in range(LOGIC_COL):
            col_start = logic_col * (COL/LOGIC_COL)
            col_end = col_start + (COL/LOGIC_COL)
#             logging.info('col index: {} - {}'.format(col_start, col_end))

            #hit counter/tot
            pix_mem_tot[col_start:col_end][pix_mem_tot[col_start:col_end] > 0] -= 1
#             logging.info('pix tot: {}'.format(pix_mem_tot[col_start:col_end]))
#             logging.info('pix tot>0 len: {}'.format(len(pix_mem_tot[col_start:col_end][pix_mem_tot[col_start:col_end] > 0])))
#             logging.info('pix tot>0: {}'.format(pix_mem_tot[col_start:col_end][pix_mem_tot[col_start:col_end] > 0]))

            #bx counter
            pix_mem_bx[col_start:col_end][pix_mem_bx[col_start:col_end] > 0] += 1
#             logging.info('pix bx: {}'.format(pix_mem_bx[col_start:col_end]))
#             logging.info('pix bx>0 len: {}'.format(len(pix_mem_bx[col_start:col_end][pix_mem_bx[col_start:col_end] > 0])))
#             logging.info('pix bx>0: {}'.format(pix_mem_bx[col_start:col_end][pix_mem_bx[col_start:col_end] > 0]))

            #Late copy
            late_copy += len(np.where(pix_mem_bx[col_start:col_end] == (PIXEL_MAX_LATENCY + 1 + MEAN_TOT))[0])

            if TRIGGER[bx] == False:
                trig_mem[logic_col][trig_mem[logic_col] == LATENCY] = 0

            trig_mem[logic_col][trig_mem[logic_col] > 0] += 1

            #histogram
            len_trig_mem_occ = len(np.nonzero(trig_mem[logic_col])[0])
            trig_mem_hist[len_trig_mem_occ] += 1
#             logging.info('trig mem status: {}'.format(np.nonzero(trig_mem[logic_col])[0]))
            trig_mem_fill_mon[logic_col][bx] += len_trig_mem_occ

            #process eoc
#             logging.info('read_col_cnt[{}]: {}'.format(logic_col, read_col_cnt[logic_col]))
            if read_col_cnt[logic_col] >= READ_COL-1:
                max_bx_latency = np.max(pix_mem_bx[col_start:col_end][np.where(pix_mem_tot[col_start:col_end] == 0)])
#                 logging.info('max bx latency for pix to read: {}'.format(max_bx_latency))
                to_read_pix = np.where((pix_mem_bx[col_start:col_end] > 0) & (pix_mem_tot[col_start:col_end]==0) & (pix_mem_bx[col_start:col_end]==max_bx_latency))
#                 to_read_pix = np.where((pix_mem_bx[col_start:col_end] > 0) & (pix_mem_tot[col_start:col_end]==0))

                to_read_pix[0][:] += col_start
#                 logging.info('to read pix: {}'.format(to_read_pix))
                for read_pix in range(len(to_read_pix[0])): #something to read
#                     logging.info('pix to read {} - {}'.format(to_read_pix[0][read_pix], to_read_pix[1][read_pix]))
                    #TODO:Read regions?
                    empty_mems = np.where(trig_mem[logic_col] == 0)[0]
#                     logging.info('empty trig mem: {}'.format(empty_mems))
                    if len(empty_mems):
                        mem_loc = empty_mems[0]
                        if pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] <= (PIXEL_MAX_LATENCY + MEAN_TOT):
                            col_ro_delay_hist[pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]]] += 1
                            val = pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]]
#                             logging.info('col ro delay: {}'.format(pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]]))
                        else:
                            large_delay += 1
                            val = pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] % (PIXEL_MAX_LATENCY +1)
                        trig_mem[logic_col][mem_loc] = val
                        if WAIT_FOR_TRIG_MEM == True:
                            pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] = 0 #clear this pixel
                            read_col_cnt[logic_col] = 0
                    else:
                        trig_mem_pileup += 1

                    if WAIT_FOR_TRIG_MEM == False:
                        pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] = 0 #clear this pixel
                        read_col_cnt[logic_col] = 0

                    break
            elif np.max(pix_mem_bx[col_start:col_end][np.where(pix_mem_tot[col_start:col_end] == 0)]) > 0:
                read_col_cnt[logic_col] += 1

#             logging.info('occupied trig mem after matrix r.o.: {}'.format(np.nonzero(trig_mem[logic_col])[0]))
        # process hits
        for pix in hits_bx:
            total_hits += 1
#             logging.info('hit pix {} - {}'.format(pix['eta_index'], pix['phi_index']))
            if pix_mem_tot[pix['eta_index']][pix['phi_index']] > 0:
                analog_pileup += 1 #Analog pielup
            else:
                if pix_mem_bx[pix['eta_index']][pix['phi_index']] > 0:
                    digital_pileup += 1 #Digital pielup
                elif pix_mem_bx[pix['eta_index']][pix['phi_index']] == 0: #start bx_cnt
                    pix_mem_bx[pix['eta_index']][pix['phi_index']] = 1
#             print 'charge for hit ({} {}): {}'.format(pix['eta_index'], pix['phi_index'], pix['charge'])
            if int(TOT_SLOPE*pix['charge']) >= MEAN_TOT:
                hits_tot = MEAN_TOT
            else:
                hits_tot = int(TOT_SLOPE*pix['charge'])
            tot_hist[hits_tot] += 1
            if int(TOT_SLOPE*pix['charge']) > PIXEL_MAX_LATENCY:
                large_tot += 1
#             print 'MP ToT:', np.amax(tot_hist)
#             logging.info('ToT is: {}'.format(hits_tot))
            pix_mem_tot[pix['eta_index']][pix['phi_index']] += MEAN_TOT#hits_tot #paralyzable deadtime
#             logging.info('tot assigned to pix ({} {}): {}'.format(pix['eta_index'], pix['phi_index'], pix_mem_tot[pix['eta_index']][pix['phi_index']]))

    print 'large tot portion:  {}%'.format(100*large_tot/total_hits)
    print 'large col ro delay: {}%'.format(100*large_delay/total_hits)
    sim_out = {'analog_pileup': analog_pileup,
               'digital_pileup': digital_pileup,
               'late_copy': late_copy,
               'trig_mem_pileup': trig_mem_pileup,
               'total_hits': total_hits,
               'out_fifo_hist': out_fifo_hist,
               'trig_mem_hist': trig_mem_hist,
               'tot_hist': tot_hist,
               'trig_count': trig_count,
               'trig_mem_fill_mon': trig_mem_fill_mon,
               'out_fifo_fill_mon': out_fifo_fill_mon,
               'to_read_trig_mem_mon': to_read_trig_mem_mon,
               'col_ro_delay_hist': col_ro_delay_hist,
               'hits_per_bx': hits_per_bx,
               'hit_rate': hit_rate
               }
    return sim_out









####################################################################################################################
####################################################################################################################
#                                              Run                                                                 #
####################################################################################################################
####################################################################################################################
if __name__ == "__main__":
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
    col = 2
    chip_hits = module_hits[(module_hits['eta_index'] < col) & (module_hits['phi_index'] < row)]

    hit_dtype = np.dtype([('bcid', np.uint64),
                      ('eta_index', np.int32),
                      ('phi_index', np.int32),
                      ('charge', np.float64)
                      ])
    hits = np.empty((chip_hits.size,), dtype=hit_dtype)
    hits['bcid'] = chip_hits['bcid']
    hits['eta_index'] = chip_hits['eta_index']
    hits['phi_index'] = chip_hits['phi_index']
    hits['charge'] = chip_hits['charge']
    print 'hits data type', hits.dtype

    with open('mc_trigger.pickle', 'r') as infile:
        mc_trigger = pickle.load(infile)
    print mc_trigger
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
                'hits_per_bx']
    sim_out = {kw: None for kw in sim_out_kwlist}
    print sim_out
    sim_out = monopix_sim(**kwa)

#     print sim_out
    print ('hit rate:          {}'.format(sim_out['hit_rate']))
    print ('analog pileup:     {}%'.format(100*sim_out['analog_pileup']/sim_out['total_hits']))
    print ('digital pileup:    {}%'.format(100*sim_out['digital_pileup']/sim_out['total_hits']))
    print ('late copy:         {}%'.format(100*sim_out['late_copy']/sim_out['total_hits']))
    print ('trig mem pileup:   {}%'.format(100*sim_out['trig_mem_pileup']/sim_out['total_hits']))
    print ('total hits:        {}'.format(sim_out['total_hits']))
    print ('trig count:        {}'.format(sim_out['trig_count']))

    plt.subplot(221)
    plt.bar(range(len(sim_out['tot_hist'])), sim_out['tot_hist'], label = 'MC data')
    plt.title('ToT histogram')
    plt.subplot(222)
    plt.hist(sim_out['hits_per_bx'], bins = 19, range = [0, 20], label = 'MC data')
    plt.title('Hits per BX')
    plt.subplot(223)
    plt.bar(range(len(sim_out['trig_mem_hist'])),sim_out['trig_mem_hist'], label = 'MC data')
    plt.title('Trig. mem. occupancy')
    plt.show()


#    print 'digital pileup @ %d*100MHz/cm2: '%hit_rate, (100*digital_pileup/total_hits)
#    print 'late copy @ %d*100MHz/cm2: '%hit_rate, (100*late_copy/total_hits)
#    print 'trig mem pileup @ %d*100MHz/cm2: '%hit_rate, (100*trig_mem_pileup/total_hits)
#    print 'total hits @ %d*100MHz/cm2: '%hit_rate, total_hits
#    print 'trig count @ %d*100MHz/cm2: '%hit_rate, trig_count
