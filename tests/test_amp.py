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
        assert (numpy.diff(result['N_R']) >= 0.).all()
        assert (numpy.diff(result['N_RS']) >= 0.).all()
        assert (numpy.diff(result['S']) <= 0.).all()
        ok = True
        ok = compare('L_i', 'cm', model.L_i, result['L'][-1]) and ok
        ok = compare('R_i', '#', model.R_i, result['R'][-1]) and ok
        ok = compare('E_m', 'J/cm3', model.E_m, result['E'][-1] / result['L'][-1]**3) and ok
        if not ok:
            print('  a_b: %.3g d' % model.a_b)
            print('  a_99: %.3g d' % model.a_99)
            print('  E_0: %.3g J' % model.E_0)
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

        continue   # for now, do not test python engine (below), as it is slow and we do not have a good metric for judging differences yet

        py_engine = pydeb.engine.create(use_cython=False)
        for name in dir(py_engine):
            if name in pydeb.temperature_correction:
                setattr(py_engine, name, getattr(model, name))
        model.engine = py_engine
        pyresult = model.simulate(n, dt, int(n / 1000))
        for name, values in pyresult.items():
            adiff = abs(values - result[name])
            avalues = abs(values)
            valid = avalues > 1e-12
            rdiff = adiff[valid] / avalues[valid]
            print(name, rdiff.max())
        if interactive:
            import matplotlib.pyplot
            fig = matplotlib.pyplot.figure()
            for i, name in enumerate(pyresult):
                ax = fig.add_subplot(len(pyresult), 1, i + 1)
                ax.plot(result['t'] / 365, result[name])
                ax.plot(pyresult['t'] / 365, pyresult[name])
                ax.set_title(name)
                ax.grid(True)
                ax.set_xlabel('time (yr)')
            matplotlib.pyplot.show()
    print('All test completed successfully' if nfailed == 0 else '%i FAILURES' % nfailed)
    return nfailed == 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--interactive', action='store_true')
    args = parser.parse_args()
    if not run(args.interactive):
        sys.exit(1)
