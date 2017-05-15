from __future__ import print_function

import glob
import io
from HTMLParser import HTMLParser
import os
import sys
import collections
import math

import yaml

sys.path.append(os.path.join(os.path.dirname(__file__), r'C:\Users\jbr\OneDrive\MERP\merp-m6-traits\scripts'))
from worms import col

with open('name_map.yaml', 'rU') as f:
    name_map = yaml.load(f)

name2units = {}

class MyHTMLParser(HTMLParser):
    def __init__(self):
        HTMLParser.__init__(self)
        self.state = 0
        self.params = collections.OrderedDict()
        self.content = None
        self.model = None

    def handle_starttag(self, tag, attrs):
        if self.state == 0 and tag == 'table':
            self.state += 1
        if self.state == 1 and tag == 'tr':
            self.state += 1
            self.content = []
        if self.state == 2 and tag == 'td':
            self.state += 1
        if self.state == 0 and tag == 'h2':
            self.state = 5
        if self.state == 6 and tag == 'a':
            self.state = 7
            self.content = []

    def handle_endtag(self, tag):
        if self.state > 0 and tag == 'table':
            self.state = -1
        if self.state == 3 and tag == 'td':
            self.state -= 1
        if self.state == 2 and tag == 'tr':
            self.state -= 1
            if self.content:
                self.params[self.content[0]] = float(self.content[1])
                if self.content[0] in name2units:
                    assert name2units[self.content[0]] == self.content[2], 'Unit mismatch for %s: %s vs. %s' % (self.content[0], name2units[self.content[0]], self.content[2])
                else:
                    name2units[self.content[0]] = self.content[2].strip()
        if self.state == 7 and tag == 'a':
            self.model = ''.join(self.content)
            self.state = 5
        if self.state == 5 and tag == 'h2':
            self.state = 0

    def handle_data(self, data):
        if self.state == 5 and data.strip() == 'Model:':
            self.state += 1
        if self.state in (3, 7):
            self.content.append(data)

no_tsns = set()
no_cols = set()
results = []
parnames = set()
for path in glob.glob('entries_web/*_par.html'):
    print(path)
    with io.open(path, 'rU') as f:
        parser = MyHTMLParser()
        parser.feed(f.read())
        assert 'p_Am' in parser.params
        assert parser.model is not None
        species, remainder = os.path.basename(path).replace('_', ' ').rsplit(' ', 1)
        col_id = col.getCoLID(species)
        if col_id is None and species in name_map:
            col_id = col.getCoLID(name_map[species])
            assert col_id is not None, 'Alternative name "%s" not found in CoL!' % name_map[species]
        if col_id is None:
            no_cols.add(species)
        #tsn = getTSN(species)
        #if tsn is None:
        #    no_tsns.add(species)
        print('  COL id: %s' % col_id)
        #print('  TSN: %s' % tsn)
        print('  model: %s' % parser.model)
        for name, value in parser.params.items():
            print('  %s: %s' % (name, value))
        if col_id is not None:
            parnames.update(parser.params.keys())
            results.append((species, col_id, parser.params))
parnames = list(parnames)

with io.open('traits.txt', 'w', encoding='utf-8') as f, io.open('traits_log.txt', 'w', encoding='utf-8') as flog:
    f.write(u'Name\tCoL ID\t%s\n' % ('\t'.join(['%s (%s)' % (name, name2units[name]) for name in parnames]),))
    flog.write(u'Name\tCoL ID\t%s\n' % ('\t'.join(['%s (log10 %s)' % (name, name2units[name]) for name in parnames]),))
    for species, col_id, parameters in results:
        f.write(u'%s\t%s' % (species, col_id))
        flog.write(u'%s\t%s' % (species, col_id))
        for name in parnames:
            value = parameters.get(name, u'')
            logvalue = u''
            if isinstance(value, float) and value > 0:
                logvalue = math.log10(value)
            f.write(u'\t%s' % value)
            flog.write(u'\t%s' % logvalue)
        f.write(u'\n')
        flog.write(u'\n')

print('No TSN found for:')
for species in no_tsns:
    print('- %s' % species)
print('No COL id found for:')
for species in no_cols:
    print('- %s' % species)
