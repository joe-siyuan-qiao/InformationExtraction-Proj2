"""
The implementation of the fenonic baseforms for isolated word recognition
"""

from copy import deepcopy


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
        self.trans = [0.8, 0.1, 0.1]
        self.emiss = [[0.5 / 255] * 256] * 3
        for i in range(3):
            self.emiss[i][self.id] = 0.5
        self.alpha = [0.0] * 2
        self.beta = [0.0] * 2

    def forward(self, prev, alpha, output):
        self.alpha[0] = alpha + prev.alpha[0] * \
            self.trans[1] * self.emiss[1][output]
        self.alpha[1] = prev.alpha[0] * self.trans[0] * \
            self.emiss[0][output] + self.alpha[0] * self.trans[2]
        return self.alpha[1]

    def alphasum(self):
        return self.alpha[0] + self.alpha[1]

    def normalpha(self, norm):
        self.alpha[0] /= norm
        self.alpha[1] /= norm

    def zeroalpha(self):
        self.alpha[0] = 0.0
        self.alpha[1] = 0.0

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
        self.trans = [0.5] * 12
        self.emiss = [[1. / 256] * 256] * 12
        self.alpha = [0.0] * 7
        self.beta = [0.0] * 7

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

    def alphasum(self):
        retsum = 0.0
        for i in range(7):
            retsum += self.alpha[i]
        return retsum

    def normalpha(self, norm):
        for i in range(7):
            self.alpha[i] /= norm

    def zeroalpha(self):
        for i in range(7):
            self.alpha[i] = 0.0


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

    def build(self, fenones, modelpool):
        self.model = [deepcopy(modelpool[256]), ]
        for fenon in fenones:
            self.model.append(modelpool[Fenon.cvtname2id(fenon)])
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

    def alphasum(self):
        self.norm = 0.0
        for i in range(len(self.model)):
            self.norm += self.model[i].alphasum()
        return self.norm

    def normalpha(self):
        self.alphasum()
        for i in range(len(self.model)):
            self.model[i].normalpha(self.norm)


class Trellis:
    """
    The class of trellis

    Members:
    -------
    stage: [] of baseform
        the stages of baseforms of the training word
    """

    def __init__(self, baseform, data):
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

    def init_training_trellis(self):
        self.training_trellis = []
        for i in range(len(self.training_data)):
            word, data = self.training_data[i][0:2]
            self.training_trellis.append(Trellis(self.baseforms[word], data))

    def forward(self):
        for i in range(len(self.training_trellis)):
            self.training_trellis[i].forward()

    def backward(self):
        for i in range(len(self.training_trellis)):
            self.training_trellis[i].backward()
