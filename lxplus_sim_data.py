#To get ROOT on lxplus:
#cd /afs/cern.ch/sw/lcg/app/releases/ROOT/5.34.00/x86_64-slc5-gcc43-opt/root
#source /afs/cern.ch/sw/lcg/contrib/gcc/4.3/x86_64-slc5/setup.sh
#source bin/thisroot.sh
#cd -

import ROOT
import numpy as np

f =  ROOT.TFile.Open('sim.root')
t = f.Get("hitstree")
size = t.GetEntries()

#for i, e in enumerate(t):
#    print (e.bcid, e.eta_module, e.phi_module, e.eta_index, e.phi_index, e.charge, e.trigger)
#    if i > 10 : break

arr = np.recarray((size,), dtype=[('bcid', np.uint32), ('eta_module', np.uint8), ('phi_module', np.uint8), ('eta_index', np.uint16), ('phi_index', np.uint16), ('charge', np.float), ('trigger', np.bool) ])

for i, e in enumerate(t):
    arr[i] =  (e.bcid, ord(e.eta_module), ord(e.phi_module), e.eta_index, e.phi_index, e.charge, e.trigger)
    if i%10000 == 1:
        print (float(i)/size)*100, '%'


np.save('sim_mc_pu200_36x36', arr)

