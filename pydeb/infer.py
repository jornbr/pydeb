import urllib.request
import urllib.parse
import json
import io
import numpy

from . import model

class CoLResult(dict):
    def _repr_html_(self):
        return '<table><tr><th style="text-align:left">Catalogue of Life identifier</th><th style="text-align:left">Species</th></tr>%s</table>' % ''.join(['<tr><td style="text-align:left">%s</td><td style="text-align:left"><a href="%s" target="_blank">%s</a></td></tr>' % (colid, url, name) for (colid, (name, url)) in self.items()])

def get_entries(name, exact=False):
    name = name.lower()
    f = urllib.request.urlopen('http://webservice.catalogueoflife.org/col/webservice?name=%s&response=full&format=json' % urllib.parse.quote_plus(name))
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

def get_ids(name, exact=False):
    results = CoLResult()
    for entry in get_entries(name, exact):
        results[entry['id']] = (entry['name_html'], entry['url'])
    return results

def get_median(col_id):
    # Retrieve inferences from Debber (returned as tab-separated UTF8 encoded text file)
    f = urllib.request.urlopen('https://deb.bolding-bruggeman.com?id=%s&download=mean' % col_id)
    parameters = {}
    for l in io.TextIOWrapper(f, encoding='utf-8'):
        name, value = l.rstrip('\n').split('\t')
        name, units = name[:-1].split(' (', 1)
        value = float(value)
        parts = units.split(' ', 1)
        if parts[0] == 'logit':
            value = 1. / (1. + numpy.exp(-value))
            units = '-' if len(parts) == 1 else parts[1]
        elif parts[0] == 'ln':
            value = numpy.exp(value)
            units = '-' if len(parts) == 1 else parts[1]
        parameters[name] = value
    return parameters

def get_model(name):
    entries = get_entries(name, exact=True)
    if len(entries) == 0:
        raise Exception('No entries in found in Catalogue of Life with exact name "%s"' % name)
    elif len(entries) > 1:
        raise Exception('Multiple entries (%i) found in Catalogue of Life with exact name "%s"' % (len(entries), name))
    entry = entries[0]
    classification = entry['classification']
    foetus = len(classification) >= 3 and classification[2]['id'] == '7a4d4854a73e6a4048d013af6416c253'
    if foetus and len(classification) >= 4:
        # Filter out egg-laying mammals (Monotremata)
        foetus = classification[3]['id'] != '7ba80933a5c268f595f28d7ef689acac'
    m = model.Model(type='stx' if foetus else 'abj')
    print('Constructed model for %s (typified model %s)' % (entry['name'], m.type))
    parameters = get_median(entry['id'])
    for name, value in parameters.items():
        setattr(m, name, value)
    return m
