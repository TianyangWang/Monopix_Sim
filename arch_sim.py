#from __future__ import print_function
import numpy as np
import random
import numba
#import matplotlib.pyplot as plt
#import yaml
#import pickle
# import logging
# import logging.handlers

#logging.basicConfig(level=logging.INFO, format='%(message)s')
#logger = logging.getLogger()
#logger.addHandler(logging.FileHandler('test.log', 'a'))
#print = logger.info

@numba.njit
def monopix_sim(SIM_TIME = 100000, LATENCY = 400, TRIGGER_RATE = 4.0/40, PIXEL_AREA = 36.0*40.0, READ_COL = 4, LOGIC_COLUMNS = 512/4, PIXEL_NO = 4*512, HIT_RATE_CM = 100*(10**6), MEAN_TOT = 15, READ_TRIG_MEM = 1, TRIG_MEM_SIZE = 96, READ_OUT_FIFO = 1, OUT_FIFO_SIZE = 128):

#     logFormatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]: %(message)s')
#     rootLogger = logging.getLogger()
#     rootLogger.setLevel(logging.DEBUG)
#     fileHandler = logging.handlers.RotatingFileHandler('log.log', maxBytes=2000000, backupCount=3)
#     fileHandler.setFormatter(logFormatter)
#     rootLogger.addHandler(fileHandler)

    # Average hit rate L4: 0.021/mm2/BC
    PIXEL_MAX_LATENCY = 63 #63      #6-bit ToT
    #LATENCY = 400              10us
    #TRIGGER_RATE = 4.0/40      4MHz


    #PIXEL_NO = 2*512 #512*512/LOGIC_COLUMNS #number of pixle per readout unit

    # READ_COL = 4      # No. of bx
    # READ_TRIG_MEM = 1 # 1- 40 / 4 - 160MHz
    # READ_OUT_FIFO = 1 # *40MHz @ 32bit

    WAIT_FOR_TRIG_MEM = False #wait with in the array or read and losse

    analog_pileup = 0.0
    digital_pileup = 0.0
    late_copy = 0.0
    trig_mem_pileup = 0.0
    total_hits = 0.0
    trig_count = 0

    pixel_hit_rate_bx = ((float(PIXEL_AREA)/(10000*10000)) * HIT_RATE_CM )/ (40*(10**6))

    pix_mem_tot = np.zeros((LOGIC_COLUMNS, PIXEL_NO) ,dtype=np.int32)
    pix_mem_bx = np.zeros((LOGIC_COLUMNS, PIXEL_NO) ,dtype=np.int32)
    trig_mem = np.zeros((LOGIC_COLUMNS, TRIG_MEM_SIZE) ,dtype=np.int32)
    out_fifo = np.zeros((OUT_FIFO_SIZE) ,dtype=np.int32)

    out_fifo_hist = np.zeros((OUT_FIFO_SIZE+1) ,dtype=np.uint)
    trig_mem_hist = np.zeros((TRIG_MEM_SIZE+1) ,dtype=np.uint)

    trig_mem_fill_mon = np.zeros((LOGIC_COLUMNS, SIM_TIME) ,dtype=np.uint)
    out_fifo_fill_mon = np.zeros((SIM_TIME) ,dtype=np.uint)
    to_read_trig_mem_mon = np.zeros((SIM_TIME) ,dtype=np.uint)
    col_ro_delay_hist = np.zeros((PIXEL_MAX_LATENCY+1+MEAN_TOT), dtype=np.uint)

    read_col_cnt = np.zeros((LOGIC_COLUMNS) ,dtype=np.uint)
    read_fifo_cnt = 0

    hits_per_bx = np.zeros((SIM_TIME) ,dtype=np.uint)

    for bx in range(SIM_TIME):

#         print '########################'
#         print 'bx: ', bx
        ###################################################################################
        #                                Output FIFO read                                 #
        ###################################################################################
        #histogram
        out_fifo_hist[len(np.nonzero(out_fifo)[0])] += 1

        len_out_fifo_occ = len(np.where(out_fifo==True)[0]) # occupied output fifo size
        #print(bx, len_out_fifo_occ, np.where(out_fifo==True)[0])
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
                if len(out_fifo_occ):
                    out_fifo[out_fifo_occ[:1]] = 0
                    read_cnt_out = 0
            else:
                read_cnt_out += 1
        #if read_fifo_cnt >= READ_OUT_FIFO):
        #    out_fifo_occ = np.where(out_fifo==True)[0] #occupied output fifo
        #    if len(out_fifo_occ):
        #        out_fifo[out_fifo_occ[:1]] = 0 #clear fist occupied output fifo
        #        read_cnt_out = 0
        #else:
        #    read_cnt_out += 1

        ##################################################################################
        #                                 OUT_MEM_FILL                                   #
        ##################################################################################
        max_trig_latency = np.max(trig_mem)

        to_read_trig_mem = np.where((trig_mem > LATENCY) & (trig_mem == max_trig_latency))
        to_read_trig_mem_mon[bx] = len(to_read_trig_mem[0])
        #print (len(to_read_trig_mem[0]), to_read_trig_mem)

        for read_trigger_mem in range(len(to_read_trig_mem[0])):
            if read_trigger_mem >= READ_TRIG_MEM:
                break

            empty_out_mem = np.where(out_fifo==0)[0]
            #print (bx, empty_out_mem)

            if len(empty_out_mem) > 0:
                trig_mem[to_read_trig_mem[0][read_trigger_mem],to_read_trig_mem[1][read_trigger_mem]] = 0 #remove triggered data
                out_fifo[empty_out_mem[:1]] = 1
                #print out_fifo
        no_trigger = random.random() > TRIGGER_RATE
        if no_trigger == False: trig_count += 1


        ###################################################################################
        #                          Column Processing                                      #
        ###################################################################################
        for col in range(LOGIC_COLUMNS):

            #hit counter/tot
            pix_mem_tot[col][pix_mem_tot[col]>0] -= 1

            #bx counter
            pix_mem_bx[col][pix_mem_bx[col] > 0] += 1

            #Late copy
            late_copy += len(np.where(pix_mem_bx[col] == (PIXEL_MAX_LATENCY + 1 + MEAN_TOT))[0]) # add mean tot if using TE as time reference
            #h = np.where(pix_mem_bx[col] == PIXEL_MAX_LATENCY)
            #c = np.where(pix_mem_bx[col]>0)

            #print (bx, col, late_copy, np.where(pix_mem_bx[col] == PIXEL_MAX_LATENCY)[0])
            #print (bx, col, late_copy, c, pix_mem_bx[col][c[0]])

            #remove hits after latency if no trigger
            if no_trigger:
                trig_mem[col][trig_mem[col] == LATENCY] = 0

            trig_mem[col][trig_mem[col] > 0] += 1

            #histogram
            trig_mem_occ = len(np.nonzero(trig_mem[col])[0])
            trig_mem_hist[trig_mem_occ] += 1
            trig_mem_fill_mon[col][bx] += trig_mem_occ

            #process eoc
            if read_col_cnt[col] >= READ_COL-1:
                max_bx_latency = np.max(pix_mem_bx[col]) # should have pre-condition that pix_mem_tot = 0
                for read_pix in np.where((pix_mem_bx[col] > 0) & (pix_mem_tot[col]==0) & (pix_mem_bx[col]==max_bx_latency) )[0]: #something to read

                    #TODO:Read regions?
                    empty_mems = np.where(trig_mem[col] == 0)[0]
                    if len(empty_mems):
                        mem_loc = empty_mems[0]

                        if pix_mem_bx[col][read_pix] <= (PIXEL_MAX_LATENCY + MEAN_TOT):
                            col_ro_delay_hist[pix_mem_bx[col][read_pix]] += 1
                            val = pix_mem_bx[col][read_pix]
                        else: val = pix_mem_bx[col][read_pix] % (PIXEL_MAX_LATENCY +1)
                        trig_mem[col][mem_loc] = val

                        if WAIT_FOR_TRIG_MEM == True:
                            pix_mem_bx[col][read_pix] = 0 #clear this pixel
                            read_col_cnt[col] = 0
                    else:
                        trig_mem_pileup += 1

                    if WAIT_FOR_TRIG_MEM == False:
                        pix_mem_bx[col][read_pix] = 0 #clear this pixel
                        read_col_cnt[col] = 0

                    break
            elif np.max(pix_mem_bx[col][np.where(pix_mem_tot[col]==0)]) > 0:
#                 print pix_mem_bx[col][np.where(pix_mem_tot[col]==0)]
#                 logging.info('latency of next pix to read: {}'.format(np.max(pix_mem_bx[col][np.where(pix_mem_tot[col]==0)])))
                read_col_cnt[col] += 1
#                 print 'read_col_cnt', read_col_cnt

            #process hits
            for pix in  np.where(np.random.rand(pix_mem_bx[col].size) < pixel_hit_rate_bx)[0]:
                total_hits += 1
                hits_per_bx[bx] += 1
                if pix_mem_tot[col][pix] > 0:
                    analog_pileup += 1 #Analog pielup
                else:
                    if pix_mem_bx[col][pix] > 0:
                        digital_pileup += 1 #Digital pielup
                    elif pix_mem_bx[col][pix] == 0: #start bx_cnt
                        pix_mem_bx[col][pix] = 1

                pix_mem_tot[col][pix] += MEAN_TOT #paralyzable deadtime

    return analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, out_fifo_fill_mon, to_read_trig_mem_mon, col_ro_delay_hist, hits_per_bx


if __name__ == "__main__":
    #print (36.4*36.4*512*2)/(1000*1000)
    hit_rate = 4.0

    kwa = {'SIM_TIME': 10000,
           'LATENCY': 400,
           'TRIGGER_RATE': 4.0/40,
           'PIXEL_AREA': 36.0*40.0,
           'READ_COL': 4,
           'LOGIC_COLUMNS': 512/16,
           'PIXEL_NO': 2*512,
           'HIT_RATE_CM': hit_rate*100*(10**6),
           'MEAN_TOT': 15,
           'READ_TRIG_MEM': 4,
           'TRIG_MEM_SIZE': 96,
           'READ_OUT_FIFO': 2,
           'OUT_FIFO_SIZE': 2048
           }

    analog_pileup, digital_pileup, late_copy, trig_mem_pileup, total_hits, out_fifo_hist, trig_mem_hist, trig_count, trig_mem_fill_mon, out_fifo_fill_mon, to_read_trig_mem_mon, col_ro_delay_hist, hits_per_bx = monopix_sim(**kwa)

    print ('analog pileup @ {}*100MHz/cm2:     {}%'.format(hit_rate, (100*analog_pileup/total_hits)))
    print ('digital pileup @ {}*100MHz/cm2:    {}%'.format(hit_rate, (100*digital_pileup/total_hits)))
    print ('late copy @ {}*100MHz/cm2:         {}%'.format(hit_rate, (100*late_copy/total_hits)))
    print ('trig mem pileup @ {}*100MHz/cm2:   {}%'.format(hit_rate, (100*trig_mem_pileup/total_hits)))
    print ('total hits @ {}*100MHz/cm2:        {}'.format(hit_rate, total_hits))
    print ('trig count @ {}*100MHz/cm2:        {}'.format(hit_rate, trig_count))

#    print 'digital pileup @ %d*100MHz/cm2: '%hit_rate, (100*digital_pileup/total_hits)
#    print 'late copy @ %d*100MHz/cm2: '%hit_rate, (100*late_copy/total_hits)
#    print 'trig mem pileup @ %d*100MHz/cm2: '%hit_rate, (100*trig_mem_pileup/total_hits)
#    print 'total hits @ %d*100MHz/cm2: '%hit_rate, total_hits
#    print 'trig count @ %d*100MHz/cm2: '%hit_rate, trig_count
