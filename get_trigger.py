import pickle
import numpy as np

mc_hits = np.load('pixel_hits-36x36x25.npy')

total_event = np.amax(mc_hits['bcid'])
print('total events: {}'.format(total_event))

mc_trigger = np.zeros((total_event + 1), dtype = bool)


for i in range(total_event + 1):
    mc_trigger[i] = mc_hits[mc_hits['bcid'] == i]['trigger'][0]
    if i%200 == 0:
        print 'processing bx {}...'.format(i)
        print 'trigger counts: {}'.format(len(mc_trigger[np.where(mc_trigger == True)]))
print 'total trigger counts: {}'.format(len(mc_trigger[np.where(mc_trigger == True)]))

with open('mc_trigger.pickle', 'w') as outfile:
    pickle.dump(mc_trigger, outfile)