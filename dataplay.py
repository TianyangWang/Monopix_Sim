import numpy as np
import matplotlib.pyplot as plt
import pickle

###############################################################################################
# PLOT ACCUMULATED HIT MAP
###############################################################################################
# module_num = 22
# with open('data_0422_module_%s.pickle' %module_num, 'r') as infile:
#         sim_out = pickle.load(infile)
#
# img = plt.imshow(sim_out['pix_hit_map'], interpolation='nearest')
# img.set_cmap('hot')
# plt.colorbar()
# plt.clim(0,3)
# plt.ylabel('Eta')
# plt.xlabel('Phi')
# plt.title('module %s' %module_num)
# plt.show()
#
# pix_no = (512*512)
#
# print pix_no, float(np.sum(sim_out['pix_hit_map']))/pix_no
#
# print np.bincount(sim_out['pix_hit_map'].flatten())

###############################################################################################

##############################################################################################
data_loss = [None]*27
hit_rate = [None]*27

pileup_a = [None]*27
pileup_d = [None]*27
pileup_trig_mem = [None]*27
late_copy_loss = [None]*27

for module_num in range(1, 27):
    with open('trigmem48_2_4_4_module_%s.pickle' %module_num, 'r') as infile:
        sim_out = pickle.load(infile)
#         plt.step(range(len(sim_out['trig_mem_hist'])),sim_out['trig_mem_hist'], label = 'Module%s' %module_num)
        pileup_a[module_num] = 100*sim_out['analog_pileup']/sim_out['total_hits']
        pileup_d[module_num] = 100*sim_out['digital_pileup']/sim_out['total_hits']
        late_copy_loss[module_num] = 100*sim_out['late_copy']/sim_out['total_hits']
        pileup_trig_mem[module_num] = 100*sim_out['trig_mem_pileup']/sim_out['total_hits']
        data_loss[module_num] = 100*(sim_out['analog_pileup'] + sim_out['digital_pileup'] + sim_out['late_copy'] + sim_out['trig_mem_pileup'])/sim_out['total_hits']
        hit_rate[module_num] = sim_out['hit_rate']
        i = range(len(sim_out['hits_per_bx']))
        plt.plot(i, sim_out['hits_per_bx'])
        plt.ylim(ymax=300)
print data_loss
print hit_rate
plt.subplot(211)
plt.plot(range(27), hit_rate, 'bo')
plt.ylabel('MHz/cm2')
plt.xlabel('Module')
plt.title('Hit rate')

plt.subplot(212)
plt.plot(range(27), pileup_a, 'r*', label = 'Analog pileup')
plt.plot(range(27), pileup_d, 'gs', label = 'Digital pileup')
plt.plot(range(27), late_copy_loss, 'bx', label = 'Data loss due to late copy')
plt.plot(range(27), pileup_trig_mem, 'y+', label = 'Trig memory pileup')
plt.plot(range(27), data_loss, 'kD', label = 'Total data loss')
plt.ylabel('%')
plt.xlabel('Module')
plt.legend()
plt.title('Data loss')

plt.subplots_adjust(hspace = 0.8)

plt.show()

###############################################################################################
#             CLUSTER INFORMATION
###############################################################################################

# clusters_1 = np.load('clusters_module1.npy')
# clusters_2 = np.load('clusters_module2.npy')
# clusters_12 = np.load('clusters_module12.npy')
# clusters_13 = np.load('clusters_module13.npy')
# clusters_26 = np.load('clusters_module26.npy')
#
# bin_1 = np.bincount(clusters_1['n_hits'])
# bin_2 = np.bincount(clusters_2['n_hits'])
# bin_12 = np.bincount(clusters_12['n_hits'])
# bin_13 = np.bincount(clusters_13['n_hits'])
# bin_26 = np.bincount(clusters_26['n_hits'])
#
# print bin_1.size, bin_2.size, bin_12.size, bin_13.size, bin_26.size
# max_bin_size = max(bin_1.size, bin_2.size, bin_12.size, bin_13.size, bin_26.size)
# print 'maximum cluster size: {}'.format(max_bin_size)
# bin_1 = np.pad(bin_1, (0, (max_bin_size - bin_1.size)), 'constant', constant_values=(0, 0))
# bin_2 = np.pad(bin_2, (0, (max_bin_size - bin_2.size)), 'constant', constant_values=(0, 0))
# bin_12 = np.pad(bin_2, (0, (max_bin_size - bin_12.size)), 'constant', constant_values=(0, 0))
# bin_13 = np.pad(bin_13, (0, (max_bin_size - bin_13.size)), 'constant', constant_values=(0, 0))
# bin_26 = np.pad(bin_26, (0, (max_bin_size - bin_26.size)), 'constant', constant_values=(0, 0))
# print bin_1.size, bin_2.size, bin_12.size, bin_13.size, bin_26.size
#
# total_size_1 = 0
# total_size_2 = 0
# total_size_12 = 0
# total_size_13 = 0
# total_size_26 = 0
# for i in range(200):
#     total_size_1 += i*bin_1[i]
#     total_size_2 += i*bin_2[i]
#     total_size_12 += i*bin_12[i]
#     total_size_13 += i*bin_13[i]
#     total_size_26 += i*bin_26[i]
#
# print clusters_1.size, total_size_1
# mean_size_1 = float(total_size_1)/clusters_1.size
# mean_size_2 = float(total_size_2)/clusters_2.size
# mean_size_12 = float(total_size_2)/clusters_12.size
# mean_size_13 = float(total_size_13)/clusters_13.size
# mean_size_26 = float(total_size_26)/clusters_26.size

# print mean_size_1, mean_size_2, mean_size_12, mean_size_13, mean_size_26
# plt.hist(clusters_1['charge'], bins = 500, range = [1, 20000], histtype = 'step', color = 'k', label = 'Module 1')
# plt.hist(clusters_2['charge'], bins = 49, range = [0, 50], histtype = 'step', color = 'y', label = 'Module 2')
# plt.hist(clusters_12['charge'], bins = 49, range = [0, 50], histtype = 'step', color = 'g', label = 'Module 12')
# plt.hist(clusters_13['charge'], bins = 500, range = [1, 20000], histtype = 'step', color = 'r', label = 'Module 13')
# plt.hist(clusters_26['charge'], bins = 500, range = [1, 20000], histtype = 'step', color = 'b', label = 'Module 26')
# plt.yscale('log', nonposy='clip')
# plt.title('Cluster size')
# plt.xlabel('Size')
# plt.legend()
# plt.show()

###############################################################################################
# LOAD MC DATA AND SET PARAMETER
###############################################################################################
# mc_hits = np.load('pixel_hits-36x36-5000.npy')
# print "finishing load data ..."
# total_event = np.amax(mc_hits['bcid'])+1
# print 'total event:                                {}'.format(total_event)
# total_module = np.amax(mc_hits['eta_module'])
# print 'module number:                              {}'.format(total_module)
# col_quad_module = np.amax(mc_hits[np.where(mc_hits['eta_module'] == 1)]['eta_index'])+1
# print 'total pixel columns per quad module:        {}'.format(col_quad_module)
# row_quad_module = np.amax(mc_hits[np.where(mc_hits['eta_module'] == 1)]['phi_index'])+1
# print 'total pixel rows per quad module:           {}'.format(row_quad_module)
# col_2_module = np.amax(mc_hits[np.where(mc_hits['eta_module'] == 26)]['eta_index'])+1
# print 'total pixel columns per double chip module: {}'.format(col_2_module)
# row_2_module = np.amax(mc_hits[np.where(mc_hits['eta_module'] == 26)]['phi_index'])+1
# print 'total pixel rows per double chip module:    {}'.format(row_2_module)
# 
# row = 512
# col = 512
# 
# module_area_quad = float(36*36*1112*1066)/100000000
# module_area_double = float(36*36*1112*533)/100000000
# print "quad module area:   {}cm2".format(module_area_quad)
# print "double module area: {}cm2".format(module_area_double)
# threshold = 150
# noise_scale = 10
# eta_index = col, phi_index = row


###############################################################################################

###############################################################################################


# sel_hits = mc_hits[np.where((mc_hits['charge_track'] + mc_hits['charge_noise']/noise_scale) > threshold)]
#
# hit_dtype = np.dtype([('bcid', np.uint64),
#                       ('eta_index', np.int16),
#                       ('phi_index', np.int16),
#                       ('charge', np.float64)
#                       ])
# hits = np.empty((sel_hits.size,), dtype=hit_dtype)
# hits['bcid'] = sel_hits['bcid']
# hits['eta_index'] = sel_hits['eta_index']
# hits['phi_index'] = sel_hits['phi_index']
# hits['charge'] = sel_hits['charge_track'] + sel_hits['charge_noise']/noise_scale

# print hits
#
# plt.hist(hits['charge'], bins = 500, range = [0, 40000], color = 'k', histtype = 'step')
# plt.yscale('log', nonposy='clip')
# plt.title('Hit charge')
# plt.show()
###############################################################################################
###############################################################################################
# pix_hit_map_quad = np.zeros((1112, 1066) ,dtype=np.int32)
# pix_hit_map_double = np.zeros((533, 1112) ,dtype=np.int32)
#
#
# for module_num in range(1,2):
#
#     module_hits = mc_hits[mc_hits['eta_module'] == module_num]
#
#     print "Assign chip hits..."
#     chip_hits = module_hits[(module_hits['eta_index'] < col) & (module_hits['phi_index'] < row)]
#
#     print "Applying threshold..."
#     sel_module_hits = module_hits[np.where((module_hits['charge_track'] + module_hits['charge_noise']/noise_scale) > threshold)]
#
#     sel_chip_hits = chip_hits[np.where((chip_hits['charge_track'] + chip_hits['charge_noise']/noise_scale) > threshold)]
#
#     if module_num <= 13:
#         hit_rate = 40*(sel_module_hits.size/total_event/module_area_quad)
#     else:
#         hit_rate = 40*(sel_module_hits.size/total_event/module_area_double)
#     print 'total hits for module {}:   {}'.format(module_num, module_hits.size)
#     print 'hit rate for module {}:     {}MHz/cm2'.format(module_num, hit_rate)

    ################################
    # SAVE AND PLOT HIGH OCCUPANCY EVENT
    ################################

#     high_occu_bx = 0
#
#     for bx in range(total_event+1):
#         hits_bx = sel_module_hits[sel_module_hits['bcid'] == bx]
#         if len(hits_bx) > 0:
#             high_occu_bx += 1
#             print bx
    #         np.save('sel_chip_hits_2000_1_bx%s' %bx ,sel_chip_hits_13[sel_chip_hits_13['bcid'] == bx])
#             for hit in hits_bx:
#                 if module_num <= 13:
#                     pix_hit_map_quad[hit['eta_index']][hit['phi_index']] += 1
#                 else:
#                     pix_hit_map_double[hit['eta_index']][hit['phi_index']] += 1
    #             print 'Column {} Row {}'.format(hit['eta_index'], hit['phi_index'])

# print 'total high occupany events with hits > 100: {}'.format(high_occu_bx)
# plt.subplot(211)
# img = plt.imshow(pix_hit_map_quad, interpolation='nearest')
# img.set_cmap('hot')
# plt.colorbar()
# plt.clim(0,3)
# plt.ylabel('Col')
# plt.xlabel('Row')

# plt.subplot(212)
# img = plt.imshow(pix_hit_map_double, interpolation='nearest')
# img.set_cmap('hot')
# plt.colorbar()
# plt.clim(0,10)
# plt.ylabel('Col')
# plt.xlabel('Row')

# plt.show()

# load_hits = np.load('sel_chip_hits_13_bx6033.npy')
# bx = 1879
# for hit in sel_module_hits_1[sel_module_hits_1['bcid'] == bx]:
#     pix_hit_map[hit['eta_index']][hit['phi_index']] += 1
#     print 'Column {} Row {}'.format(hit['eta_index'], hit['phi_index'])
# img = plt.imshow(pix_hit_map, interpolation='nearest')
# img.set_cmap('hot')
# # plt.colorbar()
# plt.clim(0,3)
# plt.ylabel('Col')
# plt.xlabel('Row')
# plt.title('bx %s' %bx)
# plt.show()
###############################################################################################
# PLOT OCCUPANCY OR CHARGE FOR SELETED MODULES
###############################################################################################
# for module_num in [1, 13, 26]:
#     module_hits = mc_hits[mc_hits['eta_module'] == module_num]
#     chip_hits = module_hits[(module_hits['eta_index'] < col) & (module_hits['phi_index'] < row)]
#     sel_chip_hits = chip_hits[np.where((chip_hits['charge_track'] + chip_hits['charge_noise']/noise_scale) > threshold)]
#     sel_module_hits = module_hits[np.where((module_hits['charge_track'] + module_hits['charge_noise']/noise_scale) > threshold)]
#     hits_per_bx = np.bincount(sel_chip_hits['bcid'])
#     average_rate = float(sel_chip_hits.size)/total_event
#     print "Average hit rate for module {}:   {}/hit/BX".format(module_num, average_rate)
#     plt.hist(hits_per_bx, bins = 199, range = [1, 200], histtype = 'step', label = 'Module %s' %module_num)
# #     plt.hist((sel_module_hits['charge_track'] + sel_module_hits['charge_noise']/noise_scale), bins = 200, range = [0, 20000], histtype = 'step', 'Module %s' %module_num)
#     plt.yscale('log', nonposy='clip')
#
# print 'plotting...'
# plt.legend()
# plt.title('Occupancy/chip/BX')
# # plt.title('Hit charge')
# plt.show()

###############################################################################################
# PLOT CHARGE
###############################################################################################
#
# sel_hits_200 = mc_hits[np.where(mc_hits['charge'] > 200)]
# sel_hits_500 = mc_hits[np.where(mc_hits['charge'] > 500)]
# sel_hits_1000 = mc_hits[np.where(mc_hits['charge'] > 1000)]
#
# plt.subplot(221)
# plt.hist(mc_hits['charge_track'], bins = 2999, range = [1, 3000], color = 'b', histtype = 'step', label = 'no threhold')
# plt.hist(sel_hits_200['charge_track'], bins = 2999, range = [1, 3000], color = 'r', histtype = 'step', label = '200e- threhold')
# plt.hist(sel_hits_500['charge_track'], bins = 2999, range = [1, 3000], color = 'g', histtype = 'step', label = '500e- threhold')
# plt.hist(sel_hits_1000['charge_track'], bins = 2999, range = [1, 3000], color = 'm', histtype = 'step', label = '1000e- threhold')
# plt.yscale('log', nonposy='clip')
# plt.title('Charge track')
# plt.legend()
#
#
# plt.subplot(222)
# plt.hist(mc_hits['charge_noise'], bins = 500, range = [-1000, 1000], color = 'b', histtype = 'step', label = 'no threhold')
# plt.hist(sel_hits_200['charge_noise'], bins = 500, range = [-1000, 1000], color = 'r', histtype = 'step', label = '200e- threhold')
# plt.hist(sel_hits_500['charge_noise'], bins = 500, range = [-1000, 1000], color = 'g', histtype = 'step', label = '500e- threhold')
# plt.hist(sel_hits_1000['charge_noise'], bins = 500, range = [-1000, 1000], color = 'm', histtype = 'step', label = '1000e- threhold')
# plt.yscale('log', nonposy='clip')
# plt.title('Charge noise')
# plt.legend()
#
# plt.subplot(223)
# plt.hist(mc_hits['charge_xtalk'], bins = 100, range = [0, 100], color = 'b', histtype = 'step', label = 'no threhold')
# plt.hist(sel_hits_200['charge_xtalk'], bins = 100, range = [0, 100], color = 'r', histtype = 'step', label = '200e- threhold')
# plt.hist(sel_hits_500['charge_xtalk'], bins = 100, range = [0, 100], color = 'g', histtype = 'step', label = '500e- threhold')
# plt.hist(sel_hits_1000['charge_xtalk'], bins = 100, range = [0, 100], color = 'm', histtype = 'step', label = '1000e- threhold')
# plt.yscale('log', nonposy='clip')
# plt.title('Charge crosstalk')
# plt.legend()

# plt.subplot(224)
# plt.hist(mc_hits['charge'], bins = 4000, range = [-1000, 3000], color = 'b', histtype = 'step', label = 'no threhold')
# plt.hist(mc_hits[np.where(mc_hits['charge']>200)]['charge'], bins = 4000, range = [-1000, 3000], color = 'r', histtype = 'step', label = '200e- threhold')
# plt.hist(mc_hits[np.where(mc_hits['charge']>500)]['charge'], bins = 4000, range = [-1000, 3000], color = 'g', histtype = 'step', label = '500e- threhold')
# plt.hist(mc_hits[np.where(mc_hits['charge']>1000)]['charge'], bins = 4000, range = [-1000, 3000], color = 'm', histtype = 'step', label = '1000e- threhold')
# plt.yscale('log', nonposy='clip')
# plt.title('Charge total')
# plt.legend()
#
# plt.show()

