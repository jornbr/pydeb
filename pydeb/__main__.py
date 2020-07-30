import argparse
from matplotlib import pyplot
from . import infer

parser = argparse.ArgumentParser()
parser.add_argument('species')
parser.add_argument('--c_T', type=float, default=1.)
args = parser.parse_args()
model = infer.get_model(args.species)
model.report(c_T=args.c_T)
delta_t = 0.1
result = model.simulate(int(model.a_99/delta_t), delta_t, c_T=args.c_T)
fig = model.plotResult(**result)
pyplot.show()
