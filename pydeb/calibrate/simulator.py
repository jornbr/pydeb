from __future__ import print_function
import threading
import timeit
import collections
import functools
from typing import Iterable, Mapping, Sequence, Optional, Any, Tuple, Union

import numpy

from . import likelihood
from .. import model as pydeb

class Sampler(object):
    name = 'MC'
    def __init__(self, parameter_names: Sequence[str], mean, cov, inverse_transforms, model_factory=None):
        self.parameter_names = parameter_names
        self.mean = mean
        self.cov = cov
        self.inverse_transforms = inverse_transforms
        self.model_factory = model_factory
        self.status = 'initializing'

    def create_model(self, x: Iterable[float]) -> pydeb.Model:
        debmodel = self.model_factory()
        for name, value in zip(self.parameter_names, x):
            setattr(debmodel, name, value)
        debmodel.initialize()
        return debmodel

    def sample(self, n: int) -> Iterable[Optional[pydeb.Model]]:
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
    def __init__(self, parameter_names: Sequence[str], mean, cov, inverse_transforms, model_factory=None):
        Sampler.__init__(self, parameter_names, mean, cov, inverse_transforms, model_factory)
        self.likelihood = likelihood.LnLikelihood(model_factory=model_factory)
        self.likelihood.add_component(likelihood.Parameters(parameter_names, mean, cov, inverse_transforms))

    def sample(self, n: int, nburn: Optional[int]=None) -> Iterable[Optional[pydeb.Model]]:
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
                    model_new.lnl = lnl_new
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
    def __init__(self, features: Sequence[str], inverse_transforms, mean, cov, sample_size: int=10000, deb_type: str='abj', temperature: float=20, priors=(), t_end: float=None, selected_properties: Iterable[str]=(), selected_outputs: Iterable[str]=()):
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
        if not self.median_model.valid:
            print('WARNING: median model is invalid')
        model_factory = functools.partial(pydeb.Model, type=deb_type)
        if priors:
            self.sample_size *= 10
            self.sampler = MCMCSampler(features, mean, cov, inverse_transforms, model_factory=model_factory)
            for prior in priors:
                self.sampler.likelihood.add_component(prior)
        else:
            self.sampler = Sampler(features, mean, cov, inverse_transforms, model_factory=model_factory)
        self.t = None
        self.results = None
        self.result = None
        self.ensemble = None
        self.nmodels = 0
        self.nresults = 0
        properties_for_output = list(selected_properties) + list(pydeb.primary_parameters) + list(self.sampler.parameter_names)
        self.properties_for_output = tuple(collections.OrderedDict.fromkeys(properties_for_output))
        self.selected_properties = self.properties_for_output if priors else selected_properties
        self.selected_outputs = selected_outputs
        self._out = None
        self._bar = None
        self.start()

    def run(self):
        start_time = timeit.default_timer()

        ensemble = numpy.empty((self.sample_size, len(self.properties_for_output)))
        self.properties = dict([(k, ensemble[:, i]) for i, k in enumerate(self.properties_for_output)])
        self.c_Ts = numpy.empty((self.sample_size,))

        def sample():
            self.nmodels = 0
            debmodels = []
            for i, model in enumerate(self.sampler.sample(self.sample_size)):
                if model is None:
                    return
                assert model.valid, 'Sampler returned an invalid model.'
                debmodels.append(model)
                self.c_Ts[i] = model.get_temperature_correction(self.temperature)
                ensemble[i, :] = [getattr(model, k) for k in self.properties_for_output]
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

        self.ensemble = ensemble

        # Define time period for simulation based on entire ensemble
        t_end = self.t_end
        if t_end is None:
            a_99_90 = numpy.percentile(numpy.array([model.a_99 for model in debmodels]) / self.c_Ts, 90)
            a_p_90 = numpy.percentile(numpy.array([model.a_p for model in debmodels]) / self.c_Ts, 90)
            t_end = min(max(a_99_90, a_p_90), 365.*200)
        self.t = numpy.linspace(0, t_end, 1000)

        self.results = {}
        for key in self.selected_outputs:
            self.results[key] = numpy.empty((n, len(self.t)), dtype='f4')

        def getResult(model: pydeb.Model, c_T: float):
            if not hasattr(model, 'outputs'):
                delta_t = max(0.04, model.a_b / c_T / 5)
                nt = int(t_end / delta_t)
                nsave = max(1, int(numpy.floor(nt / 1000)))
                result = model.simulate(nt, delta_t, nsave, c_T=c_T)
                outputs = {}
                for key in self.selected_outputs:
                    values = model.evaluate(key, c_T=c_T, locals=result)
                    outputs[key] = numpy.interp(self.t, result['t'], values)
                model.outputs = outputs
            return model.outputs

        if self.sampler.name == 'MCMC':
            median_pars = dict([(name, numpy.median(self.properties[name])) for name in self.sampler.parameter_names])
            self.median_model = self.median_model.copy(**median_pars)
            self.median_model.initialize()

        if self.median_model.valid:
            self.median_result = getResult(self.median_model, self.median_model.get_temperature_correction(self.temperature))

        for i, model in enumerate(debmodels):
            if i % 100 == 0:
                self.update_progress(0.5 + (0.45 * i) / n, 'simulating with model %i of %i' % (i, n))
            for key, values in getResult(model, self.c_Ts[i]).items():
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

    def update_progress(self, value: float, status: str):
        self.progress = value
        self.status = status
        if self._bar is not None:
            self._bar.value = value
            self._out.value = status

    def get_statistics(self, select: Optional[Iterable[str]]=None, percentiles: Iterable[float]=(2.5, 25, 50, 75, 97.5)):
        if self.result is not None:
            return self.result

        selected_outputs = self.selected_outputs if select is None else set(self.selected_outputs).intersection(select)

        def getStatistics(k: str, n: int) -> Mapping[str, Any]:
            x = self.properties[k][:n]
            if k not in pydeb.primary_parameters:
                # Correct for body temperature
                x = x * self.c_Ts[:n]**pydeb.temperature_correction[k]

            bincounts, binbounds = numpy.histogram(x, bins=100)
            stats = dict([('perc%03i' % (10*p), v) for p, v in zip(percentiles, numpy.percentile(x, percentiles))])
            stats['histogram'] = [binbounds[0], binbounds[-1], bincounts]
            return stats

        result = {}
        if self.median_result is not None:
            result['median'] = dict([(key, self.median_result[key]) for key in selected_outputs])
        n = int(self.nmodels)
        if n > 1:
            result['properties'] = dict([(key, getStatistics(key, n)) for key in self.selected_properties])
        n = int(self.nresults)
        if n > 0 and selected_outputs:
            result['t'] = self.t
            for key in selected_outputs:
                percs = numpy.percentile(self.results[key][:n, :], percentiles, axis=0)
                result[key] = [percs[i, :] for i in range(percs.shape[0])]
        return result

    def get_result(self, select: Optional[Iterable[str]]=None) -> Mapping[str, Any]:
        result = {'status': self.status or self.sampler.status, 'progress': self.progress}
        result.update(self.get_statistics(select))
        return result

    def get_ensemble(self, names: Optional[Iterable[str]]=None, temperature_correct: Union[bool, Iterable[bool]]=False) -> Tuple[Iterable[str], numpy.ndarray]:
        if names is None:
            names = self.sampler.parameter_names
        indices = [self.properties_for_output.index(name) for name in names]
        result = self.ensemble[:, indices]
        if numpy.any(temperature_correct):
            temperature_correct = numpy.broadcast_to(temperature_correct, (result.shape[1],))
            result *= self.c_Ts[:, numpy.newaxis]**numpy.where(temperature_correct, [pydeb.temperature_correction[name] for name in names], 0.)
        return names, result

    def get_progress_bar(self):
        if self._bar is None:
            import ipywidgets
            self._out = ipywidgets.Text(value=self.status or self.sampler.status)
            self._bar = ipywidgets.FloatProgress(value=self.progress, min=0.0, max=1.0)
        return ipywidgets.VBox([self._bar, self._out])