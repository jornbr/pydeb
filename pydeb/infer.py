import urllib.request
import urllib.parse
import json
import io
import numpy
import IPython.display

class CoLResult(dict):
    def _repr_html_(self):
        return '<table><tr><th style="text-align:left">Catalogue of Life identifier</th><th style="text-align:left">Species</th></tr>%s</table>' % ''.join(['<tr><td style="text-align:left">%s</td><td style="text-align:left"><a href="%s" target="_blank">%s</a></td></tr>' % (colid, url, name) for (colid, (name, url)) in self.items()])

def get_ids(name):
    f = urllib.request.urlopen('http://webservice.catalogueoflife.org/col/webservice?name=%s&format=json' % urllib.parse.quote_plus(name))
    data = json.load(f)
    results = CoLResult()
    if 'results' not in data:
        return 'No entries found in Catalogue of Life with name "%s"' % name
    for entry in data['results']:
        entry = entry.get('accepted_name', entry)
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