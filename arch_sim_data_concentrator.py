#from __future__ import print_function
import numpy as np
import random
import numba
import matplotlib.pyplot as plt
import pickle
#import matplotlib.pyplot as plt
#import yaml
#import pickle
# import logging
# import logging.handlers

#logging.basicConfig(level=logging.INFO, format='%(message)s')
#logger = logging.getLogger()
#logger.addHandler(logging.FileHandler('test.log', 'a'))
#print = logger.info

# @numba.njit
def monopix_sim(SIM_TIME = 100000, EOC_FIFO_LATENCY = 63, LATENCY = 400, TRIGGER_RATE = 4.0/40, PIXEL_AREA = 36.0*36.0, ROW = 512, COL = 512, LOGIC_COL = 512/2, COL_REGION = 32, HIT_RATE_CM = 100*(10**6), MEAN_TOT = 15, READ_COL = 2, READ_EOC_FIFO = 1, EOC_FIFO_SIZE = 8, READ_TRIG_MEM = 1, TRIG_MEM_SIZE = 1024, READ_OUT_FIFO = 1, OUT_FIFO_SIZE = 128):

#     logFormatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]: %(message)s')
#     rootLogger = logging.getLogger()
#     rootLogger.setLevel(logging.DEBUG)
#     fileHandler = logging.handlers.RotatingFileHandler('log.log', maxBytes=2000000, backupCount=3)
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)

    # Average hit rate L4: 0.021/mm2/BC
    PIXEL_MAX_LATENCY = 63 #64?      #6-bit ToT
    WAIT_FOR_EOC_FIFO = False #wait with in the array or read and losse

    analog_pileup = 0.0
    digital_pileup = 0.0
    late_copy = 0.0
    eoc_fifo_pileup = 0.0

    total_hits = 0.0
    trig_count = 0

    pixel_hit_rate_bx = ((float(PIXEL_AREA)/(10000*10000)) * HIT_RATE_CM )/ (40*(10**6))

    pix_mem_tot = np.zeros((COL, ROW) ,dtype=np.int32)
    pix_mem_bx = np.zeros((COL, ROW) ,dtype=np.int32)
    eoc_fifo = np.zeros((LOGIC_COL, EOC_FIFO_SIZE) ,dtype=np.int32)
    trig_mem = np.zeros((COL_REGION, TRIG_MEM_SIZE) ,dtype=np.int32)
    out_fifo = np.zeros((OUT_FIFO_SIZE) ,dtype=np.int32)

    eoc_fifo_hist = np.zeros((EOC_FIFO_SIZE+1) ,dtype=np.uint)
    trig_mem_hist = np.zeros((TRIG_MEM_SIZE+1) ,dtype=np.uint)
    out_fifo_hist = np.zeros((OUT_FIFO_SIZE+1) ,dtype=np.uint)
    col_ro_delay_hist = np.zeros((PIXEL_MAX_LATENCY+1+MEAN_TOT), dtype=np.uint)

    eoc_fifo_fill_mon = np.zeros((LOGIC_COL, SIM_TIME) ,dtype=np.uint)
    trig_mem_fill_mon = np.zeros((COL_REGION, SIM_TIME) ,dtype=np.uint)
    out_fifo_fill_mon = np.zeros((SIM_TIME) ,dtype=np.uint)

    to_read_eoc_fifo_mon = np.zeros((SIM_TIME) ,dtype=np.uint)
    to_read_trig_mem_mon = np.zeros((SIM_TIME) ,dtype=np.uint)

    read_col_cnt = np.zeros((LOGIC_COL) ,dtype=np.uint)

    large_delay = 0

    hits_per_bx = np.zeros((SIM_TIME) ,dtype=np.uint)

    for bx in range(SIM_TIME):
        if bx%1000==1:
            print bx
        ###################################################################################
        #                                Read output FIFO                                 #
        ###################################################################################
        out_fifo_hist[len(np.nonzero(out_fifo)[0])] += 1

        len_out_fifo_occ = len(np.where(out_fifo==True)[0]) # occupied output fifo size
        out_fifo_fill_mon[bx] += len_out_fifo_occ

        for read_output_fifo in range(len_out_fifo_occ):
            if read_output_fifo >= READ_OUT_FIFO:
                break
            out_fifo_occ = np.where(out_fifo==True)[0]
            if len(out_fifo_occ):
                out_fifo[out_fifo_occ[:1]] = 0


        ####################################################################################
        #                                 Read trigger memory                              #
        ####################################################################################
        max_trig_latency = np.max(trig_mem)
        to_read_trig_mem = np.where((trig_mem > LATENCY) & (trig_mem == max_trig_latency))
        to_read_trig_mem_mon[bx] = len(to_read_trig_mem[0])

        for read_trigger_mem in range(len(to_read_trig_mem[0])):
            if read_trigger_mem >= READ_TRIG_MEM:
                break
            empty_out_fifo = np.where(out_fifo==0)[0]
            if len(empty_out_fifo) > 0:
                trig_mem[to_read_trig_mem[0][read_trigger_mem],to_read_trig_mem[1][read_trigger_mem]] = 0 #remove triggered data
                out_fifo[empty_out_fifo[:1]] = 1


        no_trigger = random.random() > TRIGGER_RATE
        if no_trigger == False: trig_count += 1

        #####################################################################################
        #                                Matrix Processing                                  #
        #####################################################################################

        for col_region in range(COL_REGION):
            logic_col_start = col_region*(LOGIC_COL/COL_REGION)
            logic_col_end = logic_col_start+LOGIC_COL/COL_REGION

            if no_trigger:
                trig_mem[col_region][trig_mem[col_region] == LATENCY] = 0
            trig_mem[col_region][trig_mem[col_region] > 0] += 1

            ############################### Write trigger memory #############################
            max_eoc_fifo_latency = np.amax(eoc_fifo[logic_col_start:logic_col_end])

            to_read_eoc_fifo = np.where((eoc_fifo[logic_col_start:logic_col_end] >= EOC_FIFO_LATENCY) & (eoc_fifo[logic_col_start:logic_col_end] == max_eoc_fifo_latency))
            to_read_eoc_fifo_mon[bx] = len(to_read_eoc_fifo[0])
            to_read_eoc_fifo[0][:] += logic_col_start
#             print '--------------------------------------------------------------'
#             print eoc_fifo[logic_col_start:logic_col_end]
#             print 'eoc fifo having large latency'
#             print np.where((eoc_fifo[logic_col_start:logic_col_end] >= EOC_FIFO_LATENCY) & (eoc_fifo[logic_col_start:logic_col_end] == max_eoc_fifo_latency))
#             print 'maximumn eoc fifo latency:     {}'.format(max_eoc_fifo_latency)
#             print 'eoc fifo to read'
#             print to_read_eoc_fifo
            for read_eoc_fifo in range(len(to_read_eoc_fifo[0])):
                if read_eoc_fifo >= READ_EOC_FIFO:
                    break
                empty_trig_mem = np.where(trig_mem[col_region]==0)[0]
                if len(empty_trig_mem) > 0:
#                     print '--------------------------------------------------------------'
#                     print 'writing trig memory column region {}'.format(col_region)
#                     print 'before writing: {}'.format(trig_mem[col_region])
                    trig_mem[col_region][empty_trig_mem[:1]] = eoc_fifo[to_read_eoc_fifo[0][read_eoc_fifo],to_read_eoc_fifo[1][read_eoc_fifo]]
#                     print 'after writing: {}'.format(trig_mem[col_region])
                    eoc_fifo[to_read_eoc_fifo[0][read_eoc_fifo],to_read_eoc_fifo[1][read_eoc_fifo]] = 0

            trig_mem_occ = len(np.nonzero(trig_mem[col_region])[0])
            trig_mem_hist[trig_mem_occ] += 1
            trig_mem_fill_mon[col_region][bx] += trig_mem_occ

            ############################### Read column region ###############################
            for logic_col_inc in range(LOGIC_COL/COL_REGION):
                col_start = col_region* (COL/COL_REGION) + logic_col_inc * (COL/LOGIC_COL)
                col_end = col_start + (COL/LOGIC_COL)

                eoc_fifo[logic_col_start + logic_col_inc][eoc_fifo[logic_col_start + logic_col_inc] > 0] += 1

                pix_mem_bx[col_start:col_end][np.where(pix_mem_tot[col_start:col_end] > 0)] += 1
                late_copy += len(np.where(pix_mem_bx[col_start:col_end] == (PIXEL_MAX_LATENCY + 1))[0])

                if read_col_cnt[logic_col_start + logic_col_inc] >= READ_COL-1:
                    max_bx_latency = np.amax(pix_mem_bx[col_start:col_end] - pix_mem_tot[col_start:col_end])
                    to_read_pix = np.where(((pix_mem_bx[col_start:col_end] - pix_mem_tot[col_start:col_end]) > 0) & ((pix_mem_bx[col_start:col_end] - pix_mem_tot[col_start:col_end]) == max_bx_latency))
                    to_read_pix[0][:] += col_start

                    for read_pix in range(len(to_read_pix[0])):
                        empty_eoc_fifo = np.where(eoc_fifo[logic_col_start + logic_col_inc] == 0)[0]
                        if len(empty_eoc_fifo):
                            eoc_fifo_loc = empty_eoc_fifo[0]
                            if pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] < (PIXEL_MAX_LATENCY + 1):
                                col_ro_delay_hist[pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]]] += 1
                                val = pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]]
                            else:
#                                 print '--------------------------------------------------------------'
#                                 print 'Pixel to read with large delay'
#                                 print pix_mem_bx[col_start:col_end][pix_mem_bx[col_start:col_end] > 0]
                                large_delay += 1
                                val = pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] % (PIXEL_MAX_LATENCY +1)
#                             print 'writing eoc fifo logic column {}'.format(logic_col_start + logic_col_inc)
#                             print 'before writing: {}'.format(eoc_fifo[logic_col_start + logic_col_inc])
                            eoc_fifo[logic_col_start + logic_col_inc][eoc_fifo_loc] = val
#                             print 'after writing:  {}'.format(eoc_fifo[logic_col_start + logic_col_inc])
                            if WAIT_FOR_EOC_FIFO == True:
                                pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] = 0
                                pix_mem_tot[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] = 0
                                read_col_cnt[logic_col_start + logic_col_inc] = 0
                        else:
                            eoc_fifo_pileup += 1

                        if WAIT_FOR_EOC_FIFO == False:
                            pix_mem_bx[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] = 0 #clear this pixel
                            pix_mem_tot[to_read_pix[0][read_pix]][to_read_pix[1][read_pix]] = 0
                            read_col_cnt[logic_col_start + logic_col_inc] = 0
                        break

                elif np.amax(pix_mem_bx[col_start:col_end] - pix_mem_tot[col_start:col_end]) > 0:
                    read_col_cnt[logic_col_start + logic_col_inc] += 1

                eoc_fifo_occ = len(np.nonzero(eoc_fifo[logic_col_start + logic_col_inc])[0])
                eoc_fifo_hist[eoc_fifo_occ] += 1
                eoc_fifo_fill_mon[logic_col_start + logic_col_inc][bx] += eoc_fifo_occ

        ###################################################################################
        #                                Assign hits                                      #
        ###################################################################################
        hits_this_bx = np.where(np.random.rand(COL, ROW) < pixel_hit_rate_bx)
        total_hits += len(hits_this_bx[0])
        hits_per_bx[bx] += len(hits_this_bx[0])

        for pix in range(len(hits_this_bx[0])):

            col_index = hits_this_bx[0][pix]
            row_index = hits_this_bx[1][pix]
            if (pix_mem_tot[col_index][row_index] - pix_mem_bx[col_index][row_index] >= 0) & (pix_mem_bx[col_index][row_index] > 0):
                analog_pileup += 1
            else:
                if pix_mem_bx[col_index][row_index] > 0:
                    digital_pileup += 1
                elif pix_mem_bx[col_index][row_index] == 0:
                    pix_mem_bx[col_index][row_index] = 1

            pix_mem_tot[col_index][row_index] += MEAN_TOT #paralyzable deadtime

    return analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, hits_per_bx, trig_count, eoc_fifo_hist, trig_mem_hist, out_fifo_hist, col_ro_delay_hist, eoc_fifo_fill_mon, trig_mem_fill_mon, out_fifo_fill_mon, to_read_eoc_fifo_mon, to_read_trig_mem_mon


if __name__ == "__main__":
    #print (36.4*36.4*512*2)/(1000*1000)
    hit_rate = 2.0

    kwa = {'SIM_TIME': 10000,
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
           'EOC_FIFO_SIZE': 16,
           'READ_TRIG_MEM': 4,
           'TRIG_MEM_SIZE': 1024,
           'READ_OUT_FIFO': 4,
           'OUT_FIFO_SIZE': 96
           }

    if 1:
    #     analog_pileup, digital_pileup, late_copy, eoc_fifo_pileup, total_hits, hits_per_bx, trig_count, eoc_fifo_hist, trig_mem_hist, out_fifo_hist, col_ro_delay_hist, eoc_fifo_fill_mon, trig_mem_fill_mon, out_fifo_fill_mon, to_read_eoc_fifo_mon, to_read_trig_mem_mon = monopix_sim(**kwa)
        ret = monopix_sim(**kwa)

        print 'finish..'

        with open('data.pickle', 'w') as outfile:
            pickle.dump(ret, outfile)

    print 'pickle..'

    with open('data.pickle', 'r') as infile:
        sim_out = pickle.load(infile)


    print ('analog pileup @ {}*100MHz/cm2:     {}%'.format(hit_rate, (100*sim_out[0]/sim_out[4])))
    print ('digital pileup @ {}*100MHz/cm2:    {}%'.format(hit_rate, (100*sim_out[1]/sim_out[4])))
    print ('late copy @ {}*100MHz/cm2:         {}%'.format(hit_rate, (100*sim_out[2]/sim_out[4])))
    print ('eoc fifo pileup @ {}*100MHz/cm2:   {}%'.format(hit_rate, (100*sim_out[3]/sim_out[4])))
    print ('total hits @ {}*100MHz/cm2:        {}'.format(hit_rate, sim_out[4]))
    print ('trig count @ {}*100MHz/cm2:        {}'.format(hit_rate, sim_out[6]))

    plt.subplot(221)
    plt.bar(range(np.size(sim_out[7])), sim_out[7])
    plt.title('eoc fifo hist')

    plt.subplot(222)
    plt.bar(range(np.size(sim_out[8])), sim_out[8])
    plt.title('trig mem hist')

    plt.subplot(223)
    plt.bar(range(np.size(sim_out[9])), sim_out[9])
    plt.title('out fifo hist')

    plt.subplot(224)
    plt.bar(range(np.size(sim_out[10])), sim_out[10])
    plt.title('readout delay hist')
#    print 'digital pileup @ %d*100MHz/cm2: '%hit_rate, (100*digital_pileup/total_hits)
#    print 'late copy @ %d*100MHz/cm2: '%hit_rate, (100*late_copy/total_hits)
#    print 'trig mem pileup @ %d*100MHz/cm2: '%hit_rate, (100*eoc_fifo_pileup/total_hits)
#    print 'total hits @ %d*100MHz/cm2: '%hit_rate, total_hits
#    print 'trig count @ %d*100MHz/cm2: '%hit_rate, trig_count
    plt.show()