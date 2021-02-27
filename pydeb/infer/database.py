from __future__ import print_function

import io
import os
from typing import Optional, List, Mapping

import numpy

import pydeb

data_root = os.path.join(os.path.dirname(__file__), '..', 'data')

inverse_transforms = {'logit': lambda x: 1./(1. + numpy.exp(-x)), 'ln': numpy.exp}
transforms = {'logit': lambda x: numpy.log(x / (1 - x)), 'ln': numpy.log}

def read_covariance(path: str):
    with io.open(path, 'rU', encoding='utf-8') as f:
        collabels = f.readline().rstrip('\n').split('\t')[1:]
        cov = numpy.empty((len(collabels), len(collabels)))
        for i, (collabel, l) in enumerate(zip(collabels, f)):
            items = l.rstrip('\n').split('\t')
            rowlabel = items.pop(0)
            assert rowlabel == collabel, 'Row and column labels for variable %i do not match (%s vs. %s)' % (i + i, rowlabel, collabel)
            cov[i, :] = items
        return tuple(collabels), cov

def read_mean(path: str):
    with io.open(path, 'rU', encoding='utf-8') as f:
        rowlabels = []
        mean = []
        for l in f:
            label, value = l.rstrip('\n').split('\t')
            rowlabels.append(label)
            mean.append(float(value))
        return tuple(rowlabels), numpy.array(mean)

class Database(object):
    def __init__(self, root: Optional[str]=None, version: Optional[str]=None):
        if root is None:
            root = os.path.join(os.path.dirname(__file__), 'data')
        self.root = root
        self.version = version or root
        self.AmP: Optional[AmP] = None
        print('Reading database from %s...' % self.root)

        print('  - metadata...', end='')
        features: List[str] = []
        scale_factors: List[float] = []
        self.transforms: Mapping[str, Optional[str]] = {}
        self.distribution_units: Mapping[str, str] = {}
        self.inverse_transforms = {}
        with io.open(os.path.join(self.root, 'features'), 'rU', encoding='utf-8') as f:
            for l in f:
                name, unit = l.rstrip('\n').split('\t')
                transform = None
                scale_factor = 1.
                tf = unit.split(' ', 1)[0]
                distribution_unit = unit
                if tf in {'log10', 'logit'}:
                    transform = {'log10': 'ln'}.get(tf, tf)
                    scale_factor = numpy.log(10.)
                    unit = unit[len(tf) + 1:]
                    distribution_unit = transform
                    if unit != '-':
                        distribution_unit = '%s %s' % (distribution_unit, unit)
                assert unit == pydeb.units[name]
                features.append(name)
                self.transforms[name] = transform
                self.inverse_transforms[name] = inverse_transforms.get(transform, lambda x: x)
                self.distribution_units[name] = distribution_unit.replace('^2', '\u00B2').replace('^3', '\u00B3')
                scale_factors.append(scale_factor)
        self.features = tuple(features)
        self.scale_factors = numpy.array(scale_factors)
        print('done')

        print('  - means and covariances...', end='')
        self.id2stats = {}
        with io.open(os.path.join(self.root, 'mean'), 'rU', encoding='utf-8') as fmean, io.open(os.path.join(self.root, 'cov'), 'rU', encoding='utf-8') as fcov:
            fcov.readline()
            labels_mean = [l.split(' (', 1)[0] for l in fmean.readline().rstrip('\n').split('\t')]
            assert len(labels_mean) == len(features) + 1
            assert tuple(labels_mean[1:]) == tuple(features)
            for lmean, lcov in zip(fmean, fcov):
                items_mean = lmean.rstrip('\n').split('\t')
                items_cov = lcov.rstrip('\n').split('\t')
                assert items_mean[0] == items_cov[0]
                n = len(items_mean) - 1
                assert len(items_cov) == (n*(n+1))/2 + 1, 'Covariance count = %i, expected %i' % (len(items_cov), (n*(n+1))/2 + 1)
                mean = numpy.array(items_mean[1:], dtype=float)
                cov = numpy.empty((n, n))
                k = 1
                for i in range(n):
                    for j in range(i + 1):
                        cov[i, j] = float(items_cov[k])
                        cov[j, i] = cov[i, j]
                        k += 1
                self.id2stats[items_mean[0]] = (mean * self.scale_factors, cov * self.scale_factors[:, numpy.newaxis] * self.scale_factors[numpy.newaxis, :])
        print('done')

        print('  - typical temperatures...', end='')
        self.typical_temperature: Mapping[str, float] = {}
        with io.open(os.path.join(self.root, 'mean_temp'), 'rU', encoding='utf-8') as f:
            f.readline()
            for l in f:
                taxonid, mean_temp = l.rstrip('\n').split('\t')
                if mean_temp != '':
                    self.typical_temperature[taxonid] = float(mean_temp)
        print('done')

        print('  - phylogenetic covariances...', end='')
        collabels, cov = read_covariance(os.path.join(self.root, 'phylocov'))
        assert tuple([l.split(' (', 1)[0] for l in collabels]) == self.features
        self.phylocov = cov * self.scale_factors[:, numpy.newaxis] * self.scale_factors[numpy.newaxis, :]
        print('done')

        print('  - phenotypic covariances...', end='')
        if os.path.isfile(os.path.join(self.root, 'phenocov')):
            collabels, cov = read_covariance(os.path.join(self.root, 'phenocov'))
            assert tuple([l.split(' (', 1)[0] for l in collabels]) == self.features
            self.phenocov = cov * self.scale_factors[:, numpy.newaxis] * self.scale_factors[numpy.newaxis, :]
            print('done')
        else:
            self.phenocov = numpy.zeros_like(self.phylocov)
            print('not found - assuming 0')

        if not os.path.isfile(os.path.join(self.root, 'mean_del_M')):
            return

        print('  - shape coefficients (del_M)...', end='')
        self.id2del_M: Mapping[str, float] = {}
        with io.open(os.path.join(self.root, 'mean_del_M'), 'rU', encoding='utf-8') as fmean, io.open(os.path.join(self.root, 'cov_del_M'), 'rU', encoding='utf-8') as fvar:
            fmean.readline()
            fvar.readline()
            for lmean, lvar in zip(fmean, fvar):
                taxonid1, mean_del_M = lmean.rstrip('\n').split('\t')
                taxonid2, var_del_M = lvar.rstrip('\n').split('\t')
                assert taxonid1 == taxonid2
                self.id2del_M[taxonid1] = float(mean_del_M) * numpy.log(10.), float(var_del_M) * numpy.log(10.)**2
        _, self.phylocov_del_M = read_covariance(os.path.join(self.root, 'phylocov_del_M'))
        self.phylocov_del_M *= numpy.log(10.)**2
        self.transforms['del_M'] = 'ln'
        self.inverse_transforms['del_M'] = numpy.exp
        self.distribution_units['del_M'] = 'ln'

        print('done')

    def get_amp(self):
        if self.AmP is None:
            self.AmP = AmP(self.root)
        return self.AmP

    def get_crossvalidation(self):
        path = os.path.join(self.root, 'cv_errors')
        if not os.path.isfile(path):
            return
        name2errors = {}
        with io.open(path, 'rU', encoding='utf-8') as f:
            labels =  f.readline().rstrip('\n').split('\t')[1:]
            for l in f:
                for name, value in zip(labels, l.rstrip('\n').split('\t')[1:]):
                    if value != '':
                        name2errors.setdefault(name, []).append(float(value))
            for name in list(name2errors.keys()):
                name2errors[name] = numpy.array(name2errors[name])
        return name2errors

class AmP(object):
    def __init__(self, dirpath: str):
        with io.open(os.path.join(dirpath, 'amp_parameters'), 'rU', encoding='utf-8') as fpar, io.open(os.path.join(dirpath, 'amp_properties'), 'rU', encoding='utf-8') as fprop:
            self.parameter_names = [name.rsplit(' (', 1)[0] for name in fpar.readline().rstrip('\n').split('\t')[2:]]
            self.property_names = [name.rsplit(' (', 1)[0] for name in fprop.readline().rstrip('\n').split('\t')[2:]]
            self.names, self.ids, self.parameters, self.properties = [], [], [], []
            for lpar, lprop in zip(fpar, fprop):
                paritems = lpar.rstrip('\n').split('\t')
                propitems = lprop.rstrip('\n').split('\t')
                assert paritems[1] == propitems[1]
                self.names.append(paritems[0])
                self.ids.append(paritems[1])
                self.properties.append(dict([(name, float(value)) for name, value in zip(self.property_names, propitems[2:]) if value != '']))
                self.parameters.append(dict([(name, float(value)) for name, value in zip(self.parameter_names, paritems[2:]) if value != '']))

    def get_models(self, debug: bool=False, precision: float=0.001):
        for taxon_name, amp_parameters, amp_properties in zip(self.names, self.parameters, self.properties):
            #print(taxon_name)
            m = pydeb.Model('stx' if 't_g' in amp_properties else 'abj')
            for name, value in amp_parameters.items():
                #print('- %s = %s' % (name, value))
                setattr(m, name, float(value))
            m.initialize(verbose=debug, precision=precision)
            if not m.valid:
                print('WARNING - %s failed:' % taxon_name)
                m.initialize(precision=precision, verbose=True)
            yield m

    def compare(self, debug: bool=False, precision: float=0.001):
        property_names, results = set(), []
        for taxon_name, taxon_id, m, amp_properties in zip(self.names, self.ids, self.get_models(debug, precision), self.properties):
            comparison = {}
            for name, amp_value in amp_properties.items():
                value = getattr(m, name, None)
                if value is not None:
                    amp_value *= amp_properties['c_T']**(-pydeb.temperature_correction.get(name, 0))
                    property_names.add(name)
                    comparison[name] = (amp_value, value, None if amp_value == 0 else (value/amp_value - 1))
            results.append((taxon_name, taxon_id, m.type, comparison))
        return sorted(property_names), results
