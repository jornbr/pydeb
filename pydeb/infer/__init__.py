import urllib.request
import urllib.parse
import json
import io
import numpy

from .. import model
from . import database

col_version = 'annual-checklist/2019' # 'col' for latest
debber_url = 'https://deb.bolding-bruggeman.com'

class CoLResult(dict):
    def _repr_html_(self):
        return '<table><tr><th style="text-align:left">Catalogue of Life identifier</th><th style="text-align:left">Taxon</th></tr>%s</table>' % ''.join(['<tr><td style="text-align:left">%s</td><td style="text-align:left"><a href="%s" target="_blank">%s</a></td></tr>' % (colid, url, name) for (colid, (name, url)) in self.items()])

def get_entries(name: str, exact: bool=False):
    name = name.lower()
    f = urllib.request.urlopen('http://catalogueoflife.org/%s/webservice?name=%s&response=full&format=json' % (col_version, urllib.parse.quote_plus(name)))
    data = json.load(f)
    results = []
    for entry in data.get('results', []):
        if exact and entry['name'].lower() != name:
            for cn in entry.get('common_names', []):
                if cn['name'].lower() == name:
                    break
            else:
                continue
        results.append(entry.get('accepted_name', entry))
    return results

def get_ids(name: str, exact: bool=False):
    results = CoLResult()
    for entry in get_entries(name, exact):
        results[entry['id']] = (entry['name_html'], entry['url'])
    return results

class ParameterEstimates(object):
    def __init__(self, col_id: str):
        # Retrieve inferences from Debber (returned as tab-separated UTF8 encoded text file)
        self.col_id = col_id
        f = urllib.request.urlopen('%s?id=%s&download=mean' % (debber_url, col_id))
        self.names, self.units, self.transforms, self.mean = [], [], [], []
        for l in io.TextIOWrapper(f, encoding='utf-8'):
            name, value = l.rstrip('\n').split('\t')
            name, units = name[:-1].split(' (', 1)
            parts = units.split(' ', 1)
            transform = None
            if parts[0] in ('logit', 'ln'):
                transform = parts[0]
                units = '-' if len(parts) == 1 else parts[1]
            self.names.append(name)
            self.units.append(units)
            self.transforms.append(transform)
            self.mean.append(float(value))
        self.mean = numpy.array(self.mean)
        self._cov = None

    @property
    def inverse_transforms(self):
        inverse_transforms = {None: lambda x: x, 'ln': numpy.exp, 'logit': lambda x: 1. / (1. + numpy.exp(-x))}
        return [inverse_transforms[t] for t in self.transforms]

    @property
    def cov(self):
        if self._cov is None:
            self._cov = numpy.empty((len(self.names), len(self.names)))
            f = urllib.request.urlopen('%s?id=%s&download=cov' % (debber_url, self.col_id))
            f.readline()
            for i, l in enumerate(io.TextIOWrapper(f, encoding='utf-8')):
                values = l.rstrip('\n').split('\t')
                name = values.pop(0).split(' (', 1)[0]
                assert len(values )== len(self.names)
                assert name == self.names[i], 'Parameter names in mean and covariance files do not match: %s vs. %s' % (self.names[i], name)
                self._cov[i, :] = values
        return self._cov

class Taxon:
    @staticmethod
    def from_name(name: str) -> 'Taxon':
        entries = get_entries(name, exact=True)
        if len(entries) == 0:
            raise Exception('No entries in found in Catalogue of Life with exact name "%s"' % name)
        elif len(entries) > 1:
            raise Exception('Multiple entries (%i) found in Catalogue of Life with exact name "%s"' % (len(entries), name))
        return Taxon(entries[0])

    @staticmethod
    def from_col_id(col_id: str) -> 'Taxon':
        f = urllib.request.urlopen('http://catalogueoflife.org/%s/webservice?id=%s&response=full&format=json' % (col_version, col_id))
        data = json.load(f)
        return Taxon(data['results'][0])

    def __init__(self, entry):
        self.col_id = entry['id']
        self.name = entry['name']
        self.rank = entry['rank']
        self.classification = entry['classification'] + [entry]

    @property
    def typified_model(self):
        foetus = len(self.classification) >= 3 and self.classification[2]['id'] == '7a4d4854a73e6a4048d013af6416c253'
        if foetus and len(self.classification) >= 4:
            # Filter out egg-laying mammals (Monotremata)
            foetus = self.classification[3]['id'] != '7ba80933a5c268f595f28d7ef689acac'
        return 'stx' if foetus else 'abj'

    @property
    def typical_temperature(self):
        f = urllib.request.urlopen('%s?id=%s&taxonomy_only=1' % (debber_url, self.col_id))
        result = json.load(f)
        return result['typical_temperature']

    @property
    def parameter_estimates(self):
        return ParameterEstimates(self.col_id)

    @property
    def median_parameters(self):
        estimates = self.parameter_estimates
        return dict([(name, it(value)) for name, value, it in zip(estimates.names, estimates.mean, estimates.inverse_transforms)])

    def get_model(self):
        m = model.Model(type=self.typified_model)
        m.col_id = self.col_id
        for name, value in self.median_parameters.items():
            setattr(m, name, value)
        m.initialize()
        if not m.valid:
            raise Exception('Median parameter set is not valid.')
        print('Constructed model for %s %s (typified model %s)' % (self.rank.lower(), self.name, m.type))
        return m
