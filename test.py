import os
import sys
sys.path.insert(0, 'python')
import hmm

id1 = hmm.Fenon.cvtname2id('AA')
id2 = hmm.Fenon.cvtname2id('JV')
lbl = os.path.join('data', 'clsp.lblnames')
scr = os.path.join('data', 'clsp.trnscr')
pts = os.path.join('data', 'clsp.endpts')
lbl = os.path.join('data', 'clsp.trnlbls')

trainer = hmm.Trainer()
trainer.read_fenones(lbl)
trainer.pick_fenonic_baseforms(scr, pts, lbl)
trainer.read_training_data(scr, lbl)
# print trainer.training_data
trainer.build_baseforms()
trainer.init_training_trellis()
trainer.forward()
