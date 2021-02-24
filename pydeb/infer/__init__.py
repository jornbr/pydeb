import urllib.request
import urllib.parse
import json
import io
import numpy

from .. import model

col_version = 'annual-checklist/2019' # 'col' for latest
debber_url = 'https://deb.bolding-bruggeman.com'

class CoLResult(dict):
    def _repr_html_(self):
        return '<table><tr><th style="text-align:left">Catalogue of Life identifier</th><th style="text-align:left">Taxon</th></tr>%s</table>' % ''.join(['<tr><td style="text-align:left">%s</td><td style="text-align:left"><a href="%s" target="_blank">%s</a></td></tr>' % (colid, url, name) for (colid, (name, url)) in self.items()])

def get_entries(name: str, exact: bool=False):
    name = name.lower()
    f = urllib.request.urlopen('http://webservice.catalogueoflife.org/%s/webservice?name=%s&response=full&format=json' % (col_version, urllib.parse.quote_plus(name)))
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

def get_typical_temperature(col_id: str):
    f = urllib.request.urlopen('%s?id=%s&taxonomy_only=1' % (debber_url, col_id))
    result = json.load(f)
    return result['typical_temperature']

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

def get_median(col_id: str):
    estimates = ParameterEstimates(col_id)
    return dict([(name, it(value)) for name, value, it in zip(estimates.names, estimates.mean, estimates.inverse_transforms)])

def get_model_by_name(name: str):
    entries = get_entries(name, exact=True)
    if len(entries) == 0:
        raise Exception('No entries in found in Catalogue of Life with exact name "%s"' % name)
    elif len(entries) > 1:
        raise Exception('Multiple entries (%i) found in Catalogue of Life with exact name "%s"' % (len(entries), name))
    return get_model(entries[0])

def get_model_by_id(col_id: str):
    f = urllib.request.urlopen('http://webservice.catalogueoflife.org/%s/webservice?id=%s&response=full&format=json' % (col_version, col_id))
    data = json.load(f)
    return get_model(data['results'][0])

def get_model(entry):
    classification = entry['classification'] + [entry]
    foetus = len(classification) >= 3 and classification[2]['id'] == '7a4d4854a73e6a4048d013af6416c253'
    if foetus and len(classification) >= 4:
        # Filter out egg-laying mammals (Monotremata)
        foetus = classification[3]['id'] != '7ba80933a5c268f595f28d7ef689acac'
    m = model.Model(type='stx' if foetus else 'abj')
    m.col_id = entry['id']
    parameters = get_median(entry['id'])
    for name, value in parameters.items():
        setattr(m, name, value)
    m.initialize()
    if not m.valid:
        raise Exception('Median parameter set is not valid.')
    print('Constructed model for %s %s (typified model %s)' % (entry['rank'].lower(), entry['name'], m.type))
    return m
