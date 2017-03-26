import pickle
with open('dev.p', 'rb') as f:
    a = pickle.load(f)

for i in range(len(a) / 3):
    string = ''
    for j in range(3):
        string += '{} & {} & {:.3f} & '.format(a[3*i+j][0], a[3*i+j][1], a[3*i+j][2])
    string = string[:-2] + ' \\\\'
    print string

