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
    """

    def __init__(self, name):
        self.name = name
        self.id = Fenon.cvtname2id(name)
        self.trans = [0.8, 0.1, 0.1]
        self.emiss = [[0.5 / 255] * 256] * 3
        for i in range(3):
            self.emiss[i][self.id] = 0.5

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
        self.trans = [1. / 12] * 12
        self.emiss = [[1. / 256] * 256] * 12


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


class Trainer:
    """
    The trainer of fenonic baseforms

    Members:
    -------
    modelpool: [] of fenones and silence
        fenones and silence
    trndata: [] of training data
        the training data
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
            lblline = lblline.split(' ')
            self.training_data.append([srcline, lblline])

    def build_baseforms(self):
        self.baseforms = {}
        for word in self.fenonic_baseforms_fenones:
            self.baseforms[word] = Baseform()
            self.baseforms[word].build(
                self.fenonic_baseforms_fenones[word], self.modelpool)

    def init_trainig_trellis(self):
        self.training_trellis = []
