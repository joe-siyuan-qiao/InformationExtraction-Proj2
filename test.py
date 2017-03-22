import os
import sys
import time
sys.path.insert(0, 'python')
import hmm

class Timer(object):
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.tstart = time.time()

    def __exit__(self, type, value, traceback):
        if self.name:
            print '[%s]' % self.name,
        print 'Elapsed: %s' % (time.time() - self.tstart)

id1 = hmm.Fenon.cvtname2id('AA')
id2 = hmm.Fenon.cvtname2id('JV')
lblnames = os.path.join('data', 'clsp.lblnames')
scr = os.path.join('data', 'clsp.trnscr')
pts = os.path.join('data', 'clsp.endpts')
lbl = os.path.join('data', 'clsp.trnlbls')

trainer = hmm.Trainer()
trainer.read_fenones(lblnames)
trainer.pick_fenonic_baseforms(scr, pts, lbl)
trainer.read_training_data(scr, lbl)
# print trainer.training_data
with Timer('build_baseforms'):
    trainer.build_baseforms()
with Timer('init_training_trellis'):
    trainer.init_training_trellis()
print trainer.modelpool[0].trans
print trainer.modelpool[0].emiss
with Timer('forward'):
    trainer.forward()
# with Timer('backward'):
#     trainer.backward()
# with Timer('update'):
#     trainer.update()
# print trainer.modelpool[0].trans
# print trainer.modelpool[0].emiss
