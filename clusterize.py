from pixel_clusterizer import clusterizer
import numpy as np
import random
import matplotlib.pyplot as plt
import logging
import sys
from matplotlib.scale import scale_factory

logFormatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s]: %(message)s')
rootLogger = logging.getLogger()
rootLogger.setLevel(logging.DEBUG)
fileHandler = logging.FileHandler('log.log')
fileHandler.setFormatter(logFormatter)
rootLogger.addHandler(fileHandler)
consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)

# logging.basicConfig(filename='log.log', format='%(levelname)s:%(message)s', level=logging.DEBUG)

# root = logging.getLogger()
# root.setLevel(logging.INFO)

# ch = logging.StreamHandler(sys.stdout)
# ch.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# ch.setFormatter(formatter)
# root.addHandler(ch)


logging.info('loading MC hits data...')

mc_hits = np.load('/faust/user/tianyang/ownCloud/pixel_hits-150x50x150-1120.npy')

mc_hits_dtype = mc_hits.dtype
logging.info('MC hits data type: {}'.format(mc_hits_dtype))

threshold = 1500
noise_scale = 1
module_num = 1
module_hits = mc_hits[mc_hits['eta_module']==module_num]
sel_hits = module_hits[np.where((module_hits['charge_track'] + module_hits['charge_noise']/noise_scale) > threshold)]
# sel_hits = module_hits[np.where(module_hits['charge_track'] > threshold)]


hit_dtype = np.dtype([('event_number', np.uint32),
                      ('frame', '<u1'),
                      ('column', np.uint16),
                      ('row', np.uint16),
                      ('charge', '<u4'),
                     ])

hits = np.empty((sel_hits.size,), dtype=hit_dtype)
hits['event_number'] = sel_hits['bcid']
hits['frame'] = 0
hits['column'] = sel_hits['eta_index']
hits['row'] = sel_hits['phi_index']
hits['charge'] = sel_hits['charge_track'] + sel_hits['charge_noise']/noise_scale
# hits['charge'] = sel_hits['charge_track']


# Initialize clusterizer object
clusterizer = clusterizer.HitClusterizer()

# All cluster settings are listed here with their std. values
clusterizer.set_column_cluster_distance(3)  # cluster distance in columns
clusterizer.set_row_cluster_distance(3)  # cluster distance in rows
#clusterizer.set_frame_cluster_distance(4)   # cluster distance in time frames
#clusterizer.set_max_hit_charge(13)  # only add hits with charge <= 29
clusterizer.ignore_same_hits(True)  # Ignore same hits in an event for clustering
clusterizer.set_hit_dtype(hit_dtype)  # Set the data type of the hits (parameter data types and names)
#clusterizer.set_hit_fields({'event_number': 'event_number',  # Set the mapping of the hit names to the internal names (here there is no mapping done, this is the std. setting)
#                            'column': 'column',
#                            'row': 'row',
#                            'charge': 'charge',
#                            'frame': 'frame'
#                            })





# Main functions
cluster_hits, clusters = clusterizer.cluster_hits(hits)  # cluster hits

np.save('clusters_module%s' %module_num, clusters)


# Print input / output histograms
logging.info('INPUT selected MC hits:         {}'.format(hits))
logging.info('Cluster hits data type:         {}'.format(clusterizer._cluster_hits.dtype.names))
logging.info('OUTPUT hits with clusters info: {}'.format(cluster_hits))
logging.info('OUTPUT cluster data type:       {}'.format(clusters.dtype))

# print(clusters[clusters['n_hits']==3])





hit_per_module = float(np.size(hits))
logging.info('Total hits for module {}:       {}'.format(module_num, hit_per_module))

hit_num_from_cluster = 0
for i in range(np.max(clusters['n_hits'])):
    hit_num_from_cluster = hit_num_from_cluster + (i + 1) * np.size(clusters[np.where(clusters['n_hits'] == i+1)])

logging.info('Hit count from clusters: {}'.format(hit_num_from_cluster))

bcid_max = float(np.max(hits['event_number']))
pix_size = float(50*150)/1000000   # in mm2
module_size = float(np.max(hits['column']))*float(np.max(hits['row']))*pix_size/100
hit_rate = 40*hit_per_module/bcid_max/module_size

logging.info('Maximum bcid:            {}'.format(bcid_max))
logging.info('Pixel size:              {}mm2'.format(pix_size))
logging.info('Columns per module:      {}'.format(np.max(hits['column'])))
logging.info('Rows per module:         {}'.format(np.max(hits['row'])))
logging.info('Module size:             {}cm2'.format(module_size))
logging.info('Hit rate:                {}MHz/cm2'.format(hit_rate))




# plt.subplot(3,3,1)
# plt.hist(clusters['charge'], bins = 1000, range = [0, 4000], label = 'Cluster charge')
# plt.legend()
#  
# cluster_single_hit = clusters[np.where(clusters['n_hits'] == 1)]
# plt.subplot(3,3,2)
# plt.hist(cluster_single_hit['charge'], bins = 1000, range = [0, 4000], label = 'Single hit cluster charge')
# plt.legend()
#  
# cluster_2_hit = clusters[np.where(clusters['n_hits'] == 2)]
# plt.subplot(3,3,3)
# plt.hist(cluster_2_hit['charge'], bins = 1000, range = [0, 4000], label = '2 hit cluster charge')
# plt.legend()
#  
# cluster_3_hit = clusters[np.where(clusters['n_hits'] == 3)]
# plt.subplot(3,3,4)
# plt.hist(cluster_3_hit['charge'], bins = 1000, range = [0, 4000], label = '3 hit cluster charge')
# plt.legend()
#  
# cluster_4_hit = clusters[np.where(clusters['n_hits'] == 4)]
# plt.subplot(3,3,5)
# plt.hist(cluster_4_hit['charge'], bins = 1000, range = [0, 4000], label = '4 hit cluster charge')
# plt.legend()
#  
# cluster_5_hit = clusters[np.where(clusters['n_hits'] == 5)]
# plt.subplot(3,3,6)
# plt.hist(cluster_5_hit['charge'], bins = 1000, range = [0, 4000], label = '5 hit cluster charge')
# plt.legend()
#  
# cluster_6_hit = clusters[np.where(clusters['n_hits'] == 6)]
# plt.subplot(3,3,7)
# plt.hist(cluster_6_hit['charge'], bins = 1000, range = [0, 4000], label = '6 hit cluster charge')
# plt.legend()
#  
# cluster_7_hit = clusters[np.where(clusters['n_hits'] == 7)]
# plt.subplot(3,3,8)
# plt.hist(cluster_7_hit['charge'], bins = 1000, range = [0, 4000], label = '7 hit cluster charge')
# plt.legend()
# 
# cluster_8_hit = clusters[np.where(clusters['n_hits'] == 8)]
# plt.subplot(3,3,9)
# plt.hist(cluster_8_hit['charge'], bins = 1000, range = [0, 4000], label = '8 hit cluster charge')
# plt.legend()

# plt.subplot(3,3,9)
# plt.hist(cluster_hits['charge'], bins = 1000, range = [0, 40000], label = 'hit charge module 20')
# plt.legend()

# plt.subplot(2,3,6)
# seed_hits = cluster_hits[np.where(cluster_hits['is_seed'] == 1)]
# plt.hist(seed_hits['charge'], bins = 1000, range = [0, 40000], label = 'Seed hit charge')
# plt.legend()
# plt.show()




np.bincount(clusters['n_hits'])

plt.subplot(221)
plt.hist(hits['charge'], bins = 200, range = [0, 40000])
plt.yscale('log', nonposy='clip')
plt.title('Hit charge')

plt.subplot(222)
plt.hist(clusters['charge'], bins = 2000, range = [0, 40000])
plt.yscale('log', nonposy='clip')
plt.xlabel('Charge')
plt.title('Cluster charge')

claster_size_hist = np.bincount(clusters['n_hits'])[:40]
plt.subplot(223)
plt.bar(range(claster_size_hist.size), claster_size_hist)
plt.title('Cluster size')
plt.show()


