from typing import Optional, Union
import urllib.request
import urllib.parse
import json
import io
import numpy

from .. import model
from . import database

col_version = 'annual-checklist/2019' # 'col' for latest
debber_url = 'https://deb.bolding-bruggeman.com'

rank2depth = {'root': -1, 'kingdom': 0, 'phylum': 1, 'class': 2, 'order': 3, 'superfamily': 3.5, 'family': 4, 'genus': 5, 'subgenus': 5.5, 'species': 6, 'infraspecies': 6.5}

# Dummy Catalogue of Life record for the root of the tree of life
root = {'id': 'root', 'name': 'root', 'rank': 'root', 'name_html': 'root'}

class CoLResult(dict):
    def _repr_html_(self):
        return '<table><tr><th style="text-align:left">Catalogue of Life identifier</th><th style="text-align:left">Taxon</th></tr>%s</table>' % ''.join(['<tr><td style="text-align:left">%s</td><td style="text-align:left"><a href="%s" target="_blank">%s</a></td></tr>' % (colid, url, name) for (colid, (name, url)) in self.items()])

def get_entries(name: str, exact: bool=False):
    name = name.lower()
    f = urllib.request.urlopen('http://catalogueoflife.org/%s/webservice?name=%s&response=full&format=json' % (col_version, urllib.parse.quote_plus(name)))
    data = json.load(f)
    results = []
    found_ids = set()
    for entry in data.get('results', []):
        if exact and entry['name'].lower() != name:
            for cn in entry.get('common_names', []):
                if cn['name'].lower() == name:
                    break
            else:
                continue
        entry = entry.get('accepted_name', entry)

        # Avoid duplicates that may have come in by remapping to accepted_name
        if entry['id'] not in found_ids:
            results.append(entry)
            found_ids.add(entry['id'])
    return results

def get_ids(name: str, exact: bool=False):
    results = CoLResult()
    for entry in get_entries(name, exact):
        results[entry['id']] = (entry['name_html'], entry['url'])
    return results

class ParameterEstimates(object):
    def __init__(self, col_id: str, offline_db: Union[database.Database, str, None]=None, add_del_M: bool=False, verbose: bool=False):
        # Retrieve inferences from Debber (returned as tab-separated UTF8 encoded text file)
        self.col_id = col_id
        if offline_db is None:
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
        else:
            if isinstance(offline_db, str):
                offline_db = database.Database(offline_db)

            # Get information on the taxon, including taxonomic classification
            taxon = Taxon.from_col_id(col_id)
            classification = [root] + taxon.classification

            # Get statistics of the taxon's primary parameters
            for found_taxon in classification[::-1]:
                if found_taxon['id'] in offline_db.id2stats:
                    break
            else:
                raise Exception('No ancestor of %s found in result tree.' % (col_id))
            if verbose:
                print('Nearest ancestor found for %s (%s): %s (%s)' % (taxon.name, col_id, found_taxon['name'], found_taxon['id']))
            self.mean, self._cov = offline_db.id2stats[found_taxon['id']]
            self._cov = self._cov + offline_db.phylocov * max(0., rank2depth['species'] - rank2depth[found_taxon['rank'].lower()]) + offline_db.phenocov
            self.names = list(offline_db.features)

            if add_del_M:
                for found_taxon in classification[::-1]:
                    if found_taxon['id'] in offline_db.id2del_M:
                        break
                else:
                    raise Exception('No ancestor of %s found in del_M result tree.' % col_id)
                if verbose:
                    print('Nearest del_M ancestor found for %s (%s): %s (%s)' % (taxon.name, col_id, found_taxon['name'], found_taxon['id']))
                mean_del_M, var_del_M = offline_db.id2del_M[found_taxon['id']]
                newmean = numpy.zeros((self.mean.shape[0] + 1,), dtype=self.mean.dtype)
                newmean[:-1] = self.mean
                newmean[-1] = mean_del_M
                newcov = numpy.zeros((self._cov.shape[0] + 1, self._cov.shape[1] + 1), dtype=self._cov.dtype)
                newcov[:-1, :-1] = self._cov
                newcov[-1, -1] = var_del_M
                self.mean, self._cov = newmean, newcov
                self.names.append('del_M')

            self.units = [model.units[name] for name in self.names]
            self.transforms = [offline_db.transforms[name] for name in self.names]

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

    @property
    def median(self):
        return dict([(name, it(value)) for name, value, it in zip(self.names, self.mean, self.inverse_transforms)])

class Taxon:
    @staticmethod
    def from_name(name: str) -> 'Taxon':
        entries = get_entries(name, exact=True)
        if len(entries) == 0:
            raise Exception('No entries in found in Catalogue of Life with exact name "%s"' % name)
        elif len(entries) > 1:
            raise Exception('Multiple entries (%i) found in Catalogue of Life with exact name "%s":\n%s' % (len(entries), name, '\n'.join(['%s: %s (%s)' % (entry['id'], entry['name'], entry['rank']) for entry in entries])))
        return Taxon(entries[0])

    @staticmethod
    def from_col_id(col_id: str) -> 'Taxon':
        f = urllib.request.urlopen('http://catalogueoflife.org/%s/webservice?id=%s&response=full&format=json' % (col_version, col_id))
        data = json.load(f)
        entry = data['results'][0]
        return Taxon(entry.get('accepted_name', entry))

    def __init__(self, entry):
        self.col_id = entry['id']
        self.name = entry['name']
        self.rank = entry['rank']
        self.classification = entry['classification'] + [entry]

    @property
    def typified_model(self) -> str:
        foetus = len(self.classification) >= 3 and self.classification[2]['id'] == '7a4d4854a73e6a4048d013af6416c253'
        if foetus and len(self.classification) >= 4:
            # Filter out egg-laying mammals (Monotremata)
            foetus = self.classification[3]['id'] != '7ba80933a5c268f595f28d7ef689acac'
        return 'stx' if foetus else 'abj'

    @property
    def typical_temperature(self) -> float:
        f = urllib.request.urlopen('%s?id=%s&taxonomy_only=1' % (debber_url, self.col_id))
        result = json.load(f)
        return result['typical_temperature']

    def infer_parameters(self, offline_db: Union[database.Database, str, None]=None, add_del_M: bool=False, verbose: bool=False) -> ParameterEstimates:
        return ParameterEstimates(self.col_id, offline_db, add_del_M, verbose=verbose)

    def get_model(self) -> model.Model:
        m = model.Model(type=self.typified_model)
        for name, value in self.infer_parameters().median.items():
            setattr(m, name, value)
        m.initialize()
        if not m.valid:
            raise Exception('Median parameter set is not valid.')
        print('Constructed model for %s %s (typified model %s)' % (self.rank.lower(), self.name, m.type))
        return m
