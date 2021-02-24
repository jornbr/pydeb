import sys
import argparse

import numpy

import pydeb

def run(interactive: bool=False):
    db = pydeb.infer.database.Database()

    amp = db.get_amp()

    nfailed = 0
    for taxon, model in zip(amp.names, amp.get_models()):
        def compare(name: str, units: str, value_ref: float, value_sim: float, accuracy: float=0.01):
            rel_dif = value_sim / value_ref - 1
            #print('  %s: calculated %.3g %s vs simulated %.3g %s (rel difference = %.6f)' % (name, value_ref, units, value_sim, units, rel_dif))
            if abs(rel_dif) > accuracy:
                print('WARNING for %s: relative difference in simulated %s is %.6f and exceeds threshold %s!' % (taxon, name, rel_dif, accuracy))
                return False
            return True

        print(taxon)
        dt = 0.01 * model.a_b
        n = int(max(365 * 200, model.a_99 * 10) / dt)
        result = model.simulate(n, dt, int(n / 1000))
        assert (numpy.diff(result['L']) >= 0.).all()
        assert (numpy.diff(result['E_H']) >= 0.).all()
        assert (numpy.diff(result['E_R']) >= 0.).all()
        assert (numpy.diff(result['cumR']) >= 0.).all()
        assert (numpy.diff(result['S']) <= 0.).all()
        ok = True
        ok = compare('L_i', 'cm', model.L_i, result['L'][-1]) and ok
        ok = compare('R_i', '#', model.R_i, result['R'][-1]) and ok
        ok = compare('E_m', 'J/cm3', model.E_m, result['E'][-1] / result['L'][-1]**3) and ok
        if not ok:
            print('  a_b: %.3g d' % model.a_b)
            print('  a_99: %.3g d' % model.a_99)
            print('  E0: %.3g J' % model.E_0)
            nfailed += 1

            if interactive:
                import matplotlib.pyplot
                fig = matplotlib.pyplot.figure()
                ax = fig.gca()
                ax.plot(result['t'] / 365, result['L'])
                ax.set_title(taxon)
                ax.grid(True)
                ax.set_xlabel('time (yr)')
                ax.axvline(model.a_99 / 365)
                matplotlib.pyplot.show()

    return nfailed == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    if not run(args.interactive):
        sys.exit(1)
