import os
import sys
import time
sys.path.insert(0, 'python')
import hmm
import numpy as np

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
training_data = trainer.training_data
val_data = trainer.training_data[-198:]
trainer.training_data = training_data[:600]
# print trainer.training_data
with Timer('build_baseforms'):
    trainer.build_baseforms()
with Timer('init_training_trellis'):
    trainer.init_training_trellis()
with Timer('init_modelpool'):
    trainer.init_modelpool()
with Timer('update trellis'):
    trainer.update_trellis()
for i in range(10):
    trainer.forward()
    alp = trainer.getalp()
    print '{:.3f}@{:03d}'.format(np.mean(alp), i)
    trainer.backward()
    trainer.update_modelpool()
    trainer.update_trellis()
acc = 0.0
for i in range(len(val_data)):
    word_list, score_list = trainer.infer(val_data[i][1])
    inferred = word_list[np.argmax(np.array(score_list))]
    if inferred == val_data[i][0]:
        acc += 1
    print acc / (i + 1)
print acc / len(val_data)
# print trainer.modelpool[0].trans
# print trainer.modelpool[0].emiss
