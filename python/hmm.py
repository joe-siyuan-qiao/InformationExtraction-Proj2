"""
The implementation of the fenonic baseforms for isolated word recognition
"""

from copy import deepcopy
import numpy as np
import sys
import math
import pickle


class Fenon:
    """
    The class of fenon

    Members:
    -------
    name: string
        the code name for the fenon
    id: int
        the id of the fenon: 0 ~ 255
    trans: [] of size 3
        the transition probabilities: t1, t2, t3
    emiss: [[] of size num_outputs specified in __init__] of size 3
        the emission probabilities for the outputs
    alpha: [] of size 2
        the alpha value used in forwarding
    beta: [] of size 2
        the beta value used in backwarding
    """

    def __init__(self, name):
        self.name = name
        self.id = Fenon.cvtname2id(name)
        self.alpha = [0.0] * 2
        self.beta = [0.0] * 2

    def init_prob(self):
        self.trans = [0.8, 0.1, 0.1]
        self.emiss = []
        for i in range(3):
            self.emiss.append([0.5 / 255] * 256)
        for i in range(3):
            self.emiss[i][self.id] = 0.5

    def forward(self, prev, alpha, output):
        self.alpha[0] = alpha + prev.alpha[0] * \
            self.trans[1] * self.emiss[1][output]
        self.alpha[1] = prev.alpha[0] * self.trans[0] * \
            self.emiss[0][output] + self.alpha[0] * self.trans[2]
        return self.alpha[1]

    def backward(self, later, beta, output):
        self.beta[1] = beta
        self.beta[0] = self.beta[1] * self.trans[2] + later.beta[0] * \
            self.trans[1] * self.emiss[1][output] + later.beta[1] * \
            self.trans[0] * self.emiss[0][output]
        return self.beta[0]

    def alphasum(self):
        return self.alpha[0] + self.alpha[1]

    def normalpha(self, norm):
        self.alpha[0] /= norm
        self.alpha[1] /= norm

    def normbeta(self, norm):
        self.beta[0] /= norm
        self.beta[1] /= norm

    def zeroalpha(self):
        self.alpha[0] = 0.0
        self.alpha[1] = 0.0

    def pass_accmodel(self, accmodel, prev, output, norm):
        pt = []
        pt.append(prev.alpha[0] * self.trans[0] *
                  self.emiss[0][output] * self.beta[1])
        pt.append(prev.alpha[0] * self.trans[1] *
                  self.emiss[1][output] * self.beta[0])
        pt.append(self.alpha[0] * self.trans[2] *
                  self.beta[1] * norm)
        for i in range(len(accmodel[self.id].trans)):
            accmodel[self.id].trans[i] += pt[i]
            accmodel[self.id].emiss[i][output] += pt[i]

    def update(self, modelpool):
        self.trans = modelpool[self.id].trans
        self.emiss = modelpool[self.id].emiss

    @staticmethod
    def cvtname2id(name):
        s, e = name[0:2]
        return (ord(s) - 65) * 26 + ord(e) - 65


class Silence:
    """
    The class of silence

    Members:
    -------
    name: string
        sil
    id: int
        256
    trans: [] of size 12
        the transition probabilities: t1, t2, ..., t12
    emiss: [[] of size num_outputs specified in __init__] of size 12
        the emission probabilities for the outputs
    """

    def __init__(self):
        self.name = 'sil'
        self.id = 256
        self.alpha = [0.0] * 7
        self.beta = [0.0] * 7

    def init_prob(self):
        self.trans = [0.5] * 12
        self.emiss = []
        for i in range(12):
            self.emiss.append([1. / 256] * 256)

    def forward(self, prev, alpha, output):
        self.alpha[0] = alpha
        self.alpha[1] = prev.alpha[0] * self.trans[0] * self.emiss[0][output] + \
            prev.alpha[1] * self.trans[1] * self.emiss[1][output]
        self.alpha[2] = prev.alpha[1] * self.trans[2] * self.emiss[2][output] + \
            prev.alpha[2] * self.trans[3] * self.emiss[3][output]
        self.alpha[3] = prev.alpha[0] * self.trans[5] * self.emiss[5][output]
        self.alpha[4] = prev.alpha[3] * self.trans[7] * self.emiss[7][output]
        self.alpha[5] = prev.alpha[4] * self.trans[9] * self.emiss[9][output]
        self.alpha[6] = prev.alpha[5] * self.trans[11] * \
            self.emiss[11][output] + prev.alpha[2] * self.trans[4] * \
            self.emiss[4][output] + self.alpha[3] * self.trans[6] + \
            self.alpha[4] * self.trans[8] + self.alpha[5] * self.trans[10]
        return self.alpha[6]

    def backward(self, later, beta, output):
        self.beta[6] = beta
        self.beta[5] = self.beta[6] * self.trans[10] + \
            later.beta[6] * self.trans[11] * self.emiss[11][output]
        self.beta[4] = self.beta[6] * self.trans[8] + \
            later.beta[5] * self.trans[9] * self.emiss[9][output]
        self.beta[3] = self.beta[6] * self.trans[6] + \
            later.beta[4] * self.trans[7] * self.emiss[7][output]
        self.beta[2] = later.beta[6] * self.trans[4] * self.emiss[4][output] + \
            later.beta[2] * self.trans[3] * self.emiss[3][output]
        self.beta[1] = later.beta[2] * self.trans[2] * self.emiss[2][output] + \
            later.beta[1] * self.trans[1] * self.emiss[1][output]
        self.beta[0] = later.beta[1] * self.trans[0] * self.emiss[0][output] + \
            later.beta[3] * self.trans[5] * self.emiss[5][output]
        return self.beta[0]

    def alphasum(self):
        retsum = 0.0
        for i in range(7):
            retsum += self.alpha[i]
        return retsum

    def normalpha(self, norm):
        for i in range(7):
            self.alpha[i] /= norm

    def normbeta(self, norm):
        for i in range(7):
            self.beta[i] /= norm

    def zeroalpha(self):
        for i in range(7):
            self.alpha[i] = 0.0

    def pass_accmodel(self, accmodel, prev, output, norm):
        pt = []
        pt.append(prev.alpha[0] * self.trans[0] *
                  self.emiss[0][output] * self.beta[1])
        pt.append(prev.alpha[1] * self.trans[1] *
                  self.emiss[1][output] * self.beta[1])
        pt.append(prev.alpha[1] * self.trans[2] *
                  self.emiss[2][output] * self.beta[2])
        pt.append(prev.alpha[2] * self.trans[3] *
                  self.emiss[3][output] * self.beta[2])
        pt.append(prev.alpha[2] * self.trans[4] *
                  self.emiss[4][output] * self.beta[6])
        pt.append(prev.alpha[0] * self.trans[5] *
                  self.emiss[5][output] * self.beta[3])
        pt.append(self.alpha[3] * self.trans[6] * self.beta[6] * norm)
        pt.append(prev.alpha[3] * self.trans[7] *
                  self.emiss[7][output] * self.beta[4])
        pt.append(self.alpha[4] * self.trans[8] * self.beta[6] * norm)
        pt.append(prev.alpha[4] * self.trans[9] *
                  self.emiss[9][output] * self.beta[5])
        pt.append(self.alpha[5] * self.trans[10] * self.beta[6] * norm)
        pt.append(prev.alpha[5] * self.trans[11] *
                  self.emiss[11][output] * self.beta[6])
        for i in range(len(accmodel[self.id].trans)):
            accmodel[self.id].trans[i] += pt[i]
            accmodel[self.id].emiss[i][output] += pt[i]

    def update(self, modelpool):
        self.trans = modelpool[self.id].trans
        self.emiss = modelpool[self.id].emiss


class Baseform:
    """
    The class of fenonic baseforms

    Members:
    -------
    name: string
        the name of the word
    model: [] of Fenon/Silence
        the fenonic baseform of the word name
    """

    def __init__(self, name):
        self.name = name
        self.norm = 1.0

    def build(self, fenones, modelpool):
        self.model = [deepcopy(modelpool[256]), ]
        for fenon in fenones:
            self.model.append(deepcopy(modelpool[Fenon.cvtname2id(fenon)]))
        self.model.append(deepcopy(modelpool[256]))

    def forward(self, prev, output):
        """
        Parameters:
        ----------
        prev: Baseform
            the previous baseform
        """
        alpha = 0.0
        for i in range(len(self.model)):
            alpha = self.model[i].forward(prev.model[i], alpha, output)

    def backward(self, later, output):
        beta = 0.0
        for i in range(len(self.model) - 1, -1, -1):
            beta = self.model[i].backward(later.model[i], beta, output)

    def alphasum(self):
        self.norm = 0.0
        for i in range(len(self.model)):
            self.norm += self.model[i].alphasum()
        return self.norm

    def normalpha(self):
        self.alphasum()
        for i in range(len(self.model)):
            self.model[i].normalpha(self.norm)

    def normbeta(self):
        for i in range(len(self.model)):
            self.model[i].normbeta(self.norm)

    def pass_accmodel(self, accmodel, prev, output):
        for i in range(len(self.model)):
            self.model[i].pass_accmodel(
                accmodel, prev.model[i], output, self.norm)

    def update(self, modelpool):
        for i in range(len(self.model)):
            self.model[i].update(modelpool)


class Trellis:
    """
    The class of trellis

    Members:
    -------
    stage: [] of baseform
        the stages of baseforms of the training word
    """

    def __init__(self, baseform, data, word=''):
        self.word = word
        self.baseform = baseform
        self.data = data
        self.stage = []
        for i in range(len(data) + 1):
            self.stage.append(deepcopy(baseform))

    def forward(self):
        # initilize alpha at stage 0
        self.stage[0].model[0].alpha[0] = 1.0
        for i in range(len(self.data)):
            prev = self.stage[i]
            self.stage[i + 1].forward(prev, Fenon.cvtname2id(self.data[i]))
            self.stage[i + 1].normalpha()

    def backward(self):
        # initilize beta at the last stage
        for i in range(len(self.stage[-1].model)):
            for j in range(len(self.stage[-1].model[i].beta)):
                self.stage[-1].model[i].beta[j] = 1.0 / self.stage[-1].norm
        for i in range(len(self.data) - 1, -1, -1):
            later = self.stage[i + 1]
            self.stage[i].backward(later, Fenon.cvtname2id(self.data[i]))
            self.stage[i].normbeta()

    def pass_accmodel(self, accmodel):
        for i in range(len(self.data)):
            prev = self.stage[i]
            output = Fenon.cvtname2id(self.data[i])
            self.stage[i + 1].pass_accmodel(accmodel, prev, output)

    def update(self, modelpool):
        for i in range(len(self.stage)):
            self.stage[i].update(modelpool)

    def getalp(self):
        lp = 0.0
        for i in range(len(self.stage)):
            lp += math.log(self.stage[i].norm)
        lp /= len(self.data)
        return lp


class Trainer:
    """
    The trainer of fenonic baseforms

    Members:
    -------
    modelpool: [] of fenones and silence
        fenones and silence
    training_data: [] of training data
        the training data
    training_trellis: [] of training trellis
        the trellis of the training data
    devdata: [] of test data
        the test data
    """

    def __init__(self):
        pass

    def read_fenones(self, filename):
        self.modelpool = []
        fin = open(filename, 'r')
        lines = fin.readlines()
        for i in range(1, len(lines)):
            name = lines[i][0:2]
            self.modelpool.append(Fenon(name))
        self.modelpool.append(Silence())
        fin.close()

    def pick_fenonic_baseforms(self, scr, pts, lbl):
        scr, pts, lbl = open(scr, 'r'), open(pts, 'r'), open(lbl, 'r')
        scrlines, ptslines, lbllines = scr.readlines()[1:], pts.readlines()[
            1:], lbl.readlines()[1:]
        self.fenonic_baseforms_fenones = {}
        for i in range(len(scrlines)):
            srcline = scrlines[i][:-1]
            if srcline in self.fenonic_baseforms_fenones:
                continue
            s, e = ptslines[i].split(' ')[0:2]
            s, e = int(s), int(e) - 1
            lblline = lbllines[i][:-1].split(' ')[s:e]
            self.fenonic_baseforms_fenones[srcline] = lblline
        scr.close()
        pts.close()
        lbl.close()

    def read_training_data(self, scr, lbl):
        self.training_data = []
        scr, lbl = open(scr, 'r'), open(lbl, 'r')
        scrlines, lbllines = scr.readlines()[1:], lbl.readlines()[1:]
        for i in range(len(scrlines)):
            srcline = scrlines[i][:-1]
            lblline = lbllines[i][:-1]
            lblline = lblline.split(' ')[:-1]
            self.training_data.append([srcline, lblline])

    def build_baseforms(self):
        self.baseforms = {}
        for word in self.fenonic_baseforms_fenones:
            self.baseforms[word] = Baseform(word)
            self.baseforms[word].build(
                self.fenonic_baseforms_fenones[word], self.modelpool)

    def init_modelpool(self):
        for i in range(len(self.modelpool)):
            self.modelpool[i].init_prob()

    def init_training_trellis(self):
        self.training_trellis = []
        for i in range(len(self.training_data)):
            sys.stdout.write('\r[init_training_trellis] {:d}/{:d}'.format(
                i, len(self.training_data)))
            sys.stdout.flush()
            word, data = self.training_data[i][0:2]
            self.training_trellis.append(
                Trellis(self.baseforms[word], data, word))
        print ' done'

    def forward(self):
        for i in range(len(self.training_trellis)):
            self.training_trellis[i].forward()

    def backward(self):
        for i in range(len(self.training_trellis)):
            self.training_trellis[i].backward()

    def update_modelpool(self):
        # copy format from modelpool
        accmodel = deepcopy(self.modelpool)
        # set all to zero
        for i in range(len(accmodel)):
            if accmodel[i].id < 256:
                accmodel[i].trans = [0.0] * 3
                accmodel[i].emiss = []
                for j in range(3):
                    accmodel[i].emiss.append([0.0] * 256)
            else:
                accmodel[i].trans = [0.0] * 12
                accmodel[i].emiss = []
                for j in range(12):
                    accmodel[i].emiss.append([0.0] * 256)
        for i in range(len(self.training_trellis)):
            self.training_trellis[i].pass_accmodel(accmodel)
        self.accmodel = deepcopy(accmodel)
        for i in range(len(self.modelpool)):
            if self.modelpool[i].id < 256:
                trans_array = np.array(accmodel[i].trans)
                trans_array_sum = trans_array.sum()
                if trans_array_sum > 0:
                    trans_array = trans_array / trans_array.sum()
                    self.modelpool[i].trans = trans_array.tolist()
            else:
                self.modelpool[i].trans[0] = accmodel[i].trans[
                    0] / (accmodel[i].trans[0] + accmodel[i].trans[5])
                self.modelpool[i].trans[1] = accmodel[i].trans[
                    1] / (accmodel[i].trans[1] + accmodel[i].trans[2])
                self.modelpool[i].trans[2] = accmodel[i].trans[
                    2] / (accmodel[i].trans[1] + accmodel[i].trans[2])
                self.modelpool[i].trans[3] = accmodel[i].trans[
                    3] / (accmodel[i].trans[3] + accmodel[i].trans[4])
                self.modelpool[i].trans[4] = accmodel[i].trans[
                    4] / (accmodel[i].trans[3] + accmodel[i].trans[4])
                self.modelpool[i].trans[5] = accmodel[i].trans[
                    5] / (accmodel[i].trans[0] + accmodel[i].trans[5])
                self.modelpool[i].trans[6] = accmodel[i].trans[
                    6] / (accmodel[i].trans[6] + accmodel[i].trans[7])
                self.modelpool[i].trans[7] = accmodel[i].trans[
                    7] / (accmodel[i].trans[6] + accmodel[i].trans[7])
                self.modelpool[i].trans[8] = accmodel[i].trans[
                    8] / (accmodel[i].trans[8] + accmodel[i].trans[9])
                self.modelpool[i].trans[9] = accmodel[i].trans[
                    9] / (accmodel[i].trans[8] + accmodel[i].trans[9])
                self.modelpool[i].trans[10] = accmodel[i].trans[
                    10] / (accmodel[i].trans[10] + accmodel[i].trans[11])
                self.modelpool[i].trans[11] = accmodel[i].trans[
                    11] / (accmodel[i].trans[10] + accmodel[i].trans[11])
            for j in range(len(accmodel[i].emiss)):
                emiss_array = np.array(accmodel[i].emiss[j])
                emiss_array_sum = emiss_array.sum()
                if emiss_array_sum > 0:
                    emiss_array = emiss_array / emiss_array.sum()
                    self.modelpool[i].emiss[j] = emiss_array.tolist()

    def update_trellis(self):
        for i in range(len(self.training_trellis)):
            self.training_trellis[i].update(self.modelpool)

    def getalp(self):
        alp = []
        for i in range(len(self.training_trellis)):
            alp.append(self.training_trellis[i].getalp())
        return alp

    def save_model(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.modelpool, f)

    def load_model(self, filename):
        with open(filename, 'rb') as f:
            self.modelpool = pickle.load(f)

    def infer(self, data):
        test_trellis = []
        word_list = []
        for word in self.modelpool:
            word_list.append(word)
            test_trellis.append(Trellis(self.baseforms[word], data))
        for i in range(len(test_trellis)):
            test_trellis[i].forward()
        alp_list = []
        for i in range(len(test_trellis)):
            alp_list.append(test_trellis[i].getalp())
        alp_np = np.array(alp_list)
        alp_md = 0.5 * (np.max(alp_np) + np.min(alp_np))
        alp_np = alp_np - alp_md
        alp_np = alp_np * len(data)
        ret_np = np.exp(alp_np)
        ret_np = ret_np / ret_np.sum()
        del test_trellis
        return word_list, ret_np.tolist()

    def test(self):
        acc = 0.0
        for i in range(len(self.training_data)):
            sys.stdout.write('\r[test] {:d}/{:d}'.format(
                i, len(self.training_data)))
            sys.stdout.flush()
            word, data = self.training_data[i][0:2]
            word_list, probs = self.infer(data)
            inferred = word_list[np.argmax(np.array(probs))]
            if word == inferred:
                acc += 1
        print ' {:3f}'.format(acc / len(self.training_data))
