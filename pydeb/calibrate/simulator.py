from __future__ import print_function
import threading
import timeit

import numpy

from . import likelihood
from .. import model as pydeb

class Sampler(object):
    name = 'MC'
    def __init__(self, parameter_names, mean, cov, inverse_transforms, E_0_ini=None, deb_type='abj'):
        self.parameter_names = parameter_names
        self.mean = mean
        self.cov = cov
        self.inverse_transforms = inverse_transforms
        self.E_0_ini = E_0_ini
        self.deb_type = deb_type
        self.status = 'initializing'

    def create_model(self, x):
        debmodel = pydeb.Model(type=self.deb_type)
        for name, value in zip(self.parameter_names, x):
            setattr(debmodel, name, value)
        debmodel.initialize(self.E_0_ini)
        return debmodel

    def sample(self, n):
        samples = numpy.empty((1000, self.mean.size))
        istep = samples.shape[0]
        count = 0
        while count < n:
            if istep == samples.shape[0]:
                samples = numpy.random.multivariate_normal(self.mean, self.cov, size=samples.shape[0])
                for i, itf in enumerate(self.inverse_transforms):
                    samples[:, i] = itf(samples[:, i])
                istep = 0
            model = self.create_model(samples[istep, :])
            istep += 1
            if model.valid:
                yield model
                count += 1

class MCMCSampler(Sampler):
    name = 'MCMC'
    def __init__(self, parameter_names, mean, cov, inverse_transforms, E_0_ini=None, deb_type='abj'):
        Sampler.__init__(self, parameter_names, mean, cov, inverse_transforms, E_0_ini, deb_type)
        self.likelihood = likelihood.LnLikelihood(deb_type=deb_type, E_0_ini=E_0_ini)
        self.likelihood.add_component(likelihood.Parameters(parameter_names, mean, cov, inverse_transforms))

    def sample(self, n, nburn=None):
        # Adaptive metropolis based on Haario et al. (2001)
        names, mean, cov = self.likelihood.get_prior()

        if nburn is None:
            nburn = int(0.1 * n)
        n = nburn + n

        zeros = numpy.zeros_like(mean)
        steps = numpy.empty((100, zeros.size))
        lncrits = numpy.log(numpy.random.rand(n))
        acceptance = numpy.zeros((n,))
        x = numpy.random.multivariate_normal(mean, cov)
        model, lnl = self.likelihood.calculate(dict(zip(names, x)))

        x_mean = zeros
        samplecov = numpy.zeros_like(cov)
        sqrt_s_d = 2.4 / numpy.sqrt(float(x.size))
        scale = sqrt_s_d

        istep = steps.shape[0]
        for i in range(n):
            for _ in range(100):
                if istep == steps.shape[0]:
                    steps = numpy.random.multivariate_normal(zeros, cov if i < nburn else samplecov, size=steps.shape[0])
                    istep = 0
                x_new = x + steps[istep, :] * (scale if i < nburn else sqrt_s_d)
                istep += 1
                params = dict(zip(names, x_new))
                model_new, lnl_new = self.likelihood.calculate(params)
                if model_new.valid:
                    model_new.params = params
                    break
            else:
                # We obtained 100 invalid DEB models in a row. This is not going to work. Report failure and return.
                yield None

            if lnl_new - lnl > lncrits[i]:
                # accept
                x, lnl, model = x_new, lnl_new, model_new
                acceptance[i] = 1.

            # Update sample mean and covariance (Adaptive Metropolis, Haario et al. 2001)
            x_mean_new = (i * x_mean + x) / (i + 1)
            x_mean2_new = x_mean_new[:, numpy.newaxis] * x_mean_new[numpy.newaxis, :]
            if i > 0:
                x2 = x[:, numpy.newaxis] * x[numpy.newaxis, :]
                samplecov = ((i - 1) * samplecov + x2 - (i + 1) * x_mean2_new) / i +  x_mean2
            x_mean, x_mean2 = x_mean_new, x_mean2_new

            if i >= nburn:
                # Sampling phase (beyond burn-in)
                yield model
            elif i % 100 == 0:
                # Burn-in phase
                self.status = 'Monte-Carlo burn-in (%i of %i)' % (i, nburn)
                if i % 1000 == 0 and i > 0:
                    # update covariance scaling to arrive at optimal acceptance fraction 0.234
                    scale *= 0.5 * (1 + acceptance[i - 999: i + 1].mean() / 0.234)
                    #print('Current acceptance: %.3f, scale = %.3f (%.2f of expected)' % (acceptance[i - 999: i + 1].mean(), scale, scale / sqrt_s_d))
        #print('Mean acceptance (excl. burn-in): %.3f' % acceptance[nburn:].mean())

class EnsembleRunner(threading.Thread):
    selected_properties = ('E_0', 'a_b', 'a_p', 'a_99', 'L_b', 'L_p', 'L_i', 'R_i', 'r_B', 'E_m', 's_M')
    selected_outputs = ('L', 'R', 'S', 'L_w')
    def __init__(self, features, inverse_transforms, mean, cov, sample_size=10000, deb_type='abj', temperature=20, priors=(), t_end=None):
        threading.Thread.__init__(self)
        self.sample_size = sample_size
        self.progress = 0.
        self.status = None
        self.temperature = temperature
        self.t_end = t_end
        self.median_model = pydeb.Model(type=deb_type)
        for name, value, itf in zip(features, mean, inverse_transforms):
            setattr(self.median_model, name, itf(value))
        self.median_model.initialize()
        self.median_result = None
        E_0_ini = None
        if self.median_model.valid:
            self.median_model.c_T = self.median_model.get_temperature_correction(self.temperature)
            E_0_ini = self.median_model.E_0
        else:
            print('WARNING: median model is invalid')
        if priors:
            self.sample_size *= 10
            self.sampler = MCMCSampler(features, mean, cov, inverse_transforms, deb_type=deb_type, E_0_ini=E_0_ini)
            for prior in priors:
                self.sampler.likelihood.add_component(prior)
        else:
            self.sampler = Sampler(features, mean, cov, inverse_transforms, deb_type=deb_type, E_0_ini=E_0_ini)
        self.t = None
        self.results = None
        self.result = None
        self.ensemble = None
        self.nmodels = 0
        self.nresults = 0
        self._out = None
        self._bar = None
        self.start()

    def run(self):
        start_time = timeit.default_timer()

        properties_for_output = set(self.selected_properties)
        if isinstance(self.sampler, MCMCSampler):
            properties_for_output.update(pydeb.primary_parameters)
            properties_for_output.update(self.sampler.parameter_names)
        self.properties = dict([(k, numpy.empty((self.sample_size,))) for k in properties_for_output])
        ensemble = numpy.empty((self.sample_size, len(self.sampler.parameter_names)))

        def sample():
            self.nmodels = 0
            debmodels = []
            for i, model in enumerate(self.sampler.sample(self.sample_size)):
                if model is None:
                    return
                assert model.valid, 'Sampler returned an invalid model.'
                debmodels.append(model)
                model.c_T = model.get_temperature_correction(self.temperature)
                for k, values in self.properties.items():
                    value = getattr(model, k)
                    if k not in pydeb.primary_parameters:
                        value = value * model.c_T**pydeb.temperature_correction[k]
                    values[i] = value
                for ipar, k in enumerate(self.sampler.parameter_names):
                    ensemble[i, ipar] = getattr(model, k)
                if i % 100 == 0:
                    self.update_progress((0.5 * i) / self.sample_size, 'initializing model %i of %i' % (i, self.sample_size))
                    self.nmodels = i + 1
            return debmodels

        debmodels = None
        while debmodels is None:
            debmodels = sample()
        n = self.nmodels = len(debmodels)

        init_end_time = timeit.default_timer()
        print('Time taken for model initialization: %s' % (init_end_time - start_time))

        self.ensemble = self.sampler.parameter_names, ensemble

        # Define time period for simulation based on entire ensemble
        t_end = self.t_end
        if t_end is None:
            a_99_90 = numpy.percentile([model.a_99/model.c_T for model in debmodels], 90)
            a_p_90 = numpy.percentile([model.a_p/model.c_T for model in debmodels], 90)
            t_end = min(max(a_99_90, a_p_90), 365.*200)
        self.t = numpy.linspace(0, t_end, 1000)

        self.results = {}
        for key in self.selected_outputs:
            self.results[key] = numpy.empty((n, len(self.t)), dtype='f4')

        def getResult(model):
            if not hasattr(model, 'outputs'):
                delta_t = max(0.04, model.a_b / model.c_T / 5)
                nt = int(t_end / delta_t)
                nsave = max(1, int(numpy.floor(nt / 1000)))
                result = model.simulate(nt, delta_t, nsave, c_T=model.c_T)
                outputs = {}
                for key in self.selected_outputs:
                    values = model.evaluate(key, c_T=model.c_T, locals=result)
                    outputs[key] = numpy.interp(self.t, result['t'], values)
                model.outputs = outputs
            return model.outputs

        if self.sampler.name == 'MCMC':
            median_pars = dict([(name, numpy.median(self.properties[name])) for name in self.sampler.parameter_names])
            self.median_model = self.median_model.copy(**median_pars)
            self.median_model.initialize()
            self.median_model.c_T = self.median_model.get_temperature_correction(self.temperature)

        if self.median_model.valid:
            self.median_result = getResult(self.median_model)

        for i, model in enumerate(debmodels):
            if i % 100 == 0:
                self.update_progress(0.5 + (0.45 * i) / n, 'simulating with model %i of %i' % (i, n))
            for key, values in getResult(model).items():
                self.results[key][i, :] = values
            if i % 100 == 0:
                self.nresults = i + 1
        self.nresults = n
        self.update_progress(0.95, 'computing statistics')
        sim_end_time = timeit.default_timer()
        print('Time taken for model simulations: %s' % (sim_end_time - init_end_time))
        self.result = self.get_statistics()
        self.update_progress(1., 'done')
        print('Time taken for statistics: %s' % (timeit.default_timer() - sim_end_time))

    def update_progress(self, value, status):
        self.progress = value
        self.status = status
        if self._bar is not None:
            self._bar.value = value
            self._out.value = status

    def get_statistics(self, select=None, percentiles = (2.5, 25, 50, 75, 97.5)):
        if self.result is not None:
            return self.result

        selected_outputs = self.selected_outputs if select is None else set(self.selected_outputs).intersection(select)

        def getStatistics(x):
            bincounts, binbounds = numpy.histogram(x, bins=100)
            stats = dict([('perc%03i' % (10*p), v) for p, v in zip(percentiles, numpy.percentile(x, percentiles))])
            stats['histogram'] = [binbounds[0], binbounds[-1], bincounts]
            return stats

        result = {}
        if self.median_result is not None:
            result['median'] = dict([(key, self.median_result[key]) for key in selected_outputs])
        n = self.nmodels
        if n > 1:
            result['properties'] = dict([(key, getStatistics(values[:n])) for key, values in self.properties.items()])
        n = self.nresults
        if n > 0 and selected_outputs:
            result['t'] = self.t
            for key in selected_outputs:
                percs = numpy.percentile(self.results[key][:n, :], percentiles, axis=0)
                result[key] = [percs[i, :] for i in range(percs.shape[0])]
        return result

    def get_result(self, select=None):
        result = {'status': self.status or self.sampler.status, 'progress': self.progress}
        result.update(self.get_statistics(select))
        return result

    def get_progress_bar(self):
        if self._bar is None:
            import ipywidgets
            self._out = ipywidgets.Text()
            self._bar = ipywidgets.FloatProgress(value=0.0, min=0.0, max=1.0)
        return ipywidgets.VBox([self._bar, self._out])