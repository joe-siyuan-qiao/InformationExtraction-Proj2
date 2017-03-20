"""
The implementation of the fenonic baseforms for isolated word recognition
"""


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

    def __init__(self, name, num_outputs):
        self.name = name
        self.id = Fenon.cvtname2id(name)
        self.trans = [0.8, 0.1, 0.1]
        self.emiss = [[0.5 / 255] * num_outputs] * 3
        for i in range(3):
            self.emiss[i][self.id] = 0.5

    @staticmethod
    def cvtname2id(name):
        s, e = name[0:2]
        return (ord(s) - 65) * 26 + ord(e) - 65
