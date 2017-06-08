import io
import numpy
from matplotlib import pyplot
import argparse

path = 'traits.txt'
parser = argparse.ArgumentParser()
parser.add_argument('path')
args = parser.parse_args()

trait2data = {}
trait2label = {}
with io.open(args.path, 'rU', encoding='utf-8') as f:
    labels = f.readline().rstrip('\n').split('\t')
    traits = []
    for label in labels[2:]:
        trait = label.split(' (', 1)[0]
        trait2label[trait] = label
        traits.append(trait)
    for l in f:
        items = l.rstrip('\n').split('\t')
        for trait, value in zip(traits, items[2:]):
            if value != '':
                trait2data.setdefault(trait, []).append(float(value))

fig = pyplot.figure(figsize=(15, 6))
ax_lin = fig.add_subplot(121)
ax_log = fig.add_subplot(122)
for trait, values in trait2data.items():
    values = numpy.array(values)
    ax_lin.cla()
    ax_lin.hist(values, bins=max(len(values)/5, 10))
    ax_lin.set_title('%s (linear axis)' % trait)
    ax_lin.grid()
    ax_lin.set_xlabel(trait2label[trait])
    logvalues = numpy.log10(values[values>0])
    ax_log.cla()
    ax_log.set_title('%s (log axis) - %i values <= 0 ignored' % (trait, len(values)-len(logvalues)))
    if len(logvalues) > 0:
        logrange = numpy.linspace(logvalues.min(), logvalues.max(), 1+max(10, len(values)/5))
        ax_log.hist(values, bins=10.**logrange)
        ax_log.set_xscale('log')
        ax_log.grid()
        ax_log.set_xlabel(trait2label[trait])
    print 'saving %s...' % trait
    fig.savefig('distribution %s.png' % trait, dpi=150)

    