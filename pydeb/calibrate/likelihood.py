import sys
import os.path
import collections
from typing import Iterable, Mapping, Optional

import numpy

sys.path.append(os.path.join(os.path.dirname(__file__), 'pydeb'))
from .. import model as pydeb

null_transform = lambda x: x

class Component(object):
    def get_prior(self):
        return (), (), ()

    def configure(self, model, values):
        pass

    def calculate_ln_likelihood(self, model, values):
        return 0

class Parameters(Component):
    def __init__(self, names: Iterable[str], mean, cov, transforms):
        self.names = names
        self.mean = mean
        self.cov = cov
        self.transforms = transforms
        self.invcov = numpy.linalg.inv(self.cov)

    def get_prior(self):
        return self.names, self.mean, self.cov

    def configure(self, model, values):
        for name, value, transform in zip(self.names, values, self.transforms):
            setattr(model, name, transform(value))

    def calculate_ln_likelihood(self, model, values):
        dx = values - self.mean
        return -0.5 * dx.dot(self.invcov).dot(dx)

class ImpliedProperty(Component):
    def __init__(self, expression, value, sd, transform=null_transform, temperature=20):
        self.expression = compile(expression, '<string>', 'eval')
        self.value = value
        self.sd = sd
        self.transform = transform
        self.temperature = temperature

    def calculate_ln_likelihood(self, model, values):
        c_T = model.get_temperature_correction(self.temperature)
        value_mod = model.evaluate(self.expression, c_T=c_T)
        z = (self.transform(value_mod) - self.value) / self.sd
        return -0.5 * z * z

class ExpressionAtSurvival(Component):
    def __init__(self, S, expression, value, sd, transform=null_transform, temperature=20):
        self.expression = compile(expression, '<string>', 'eval')
        self.value = value
        self.sd = sd
        self.transform = transform
        self.temperature = temperature
        self.S = S

    def calculate_ln_likelihood(self, model, values):
        c_T = model.get_temperature_correction(self.temperature)
        result = model.state_at_survival(self.S, c_T=c_T)
        value_mod = model.evaluate(self.expression, c_T=c_T, locals=result)
        z = (self.transform(value_mod) - self.value) / self.sd
        #print(self.transform(value_mod), self.value, self.sd, z, c_T)
        return -0.5 * z * z

class TimeSeries(Component):
    def __init__(self, t, temperature: float=20, offset_reference: Optional[str]=None, offset: float=0., offset_type: str='t'):
        self.t = numpy.array(t)
        self.offset = offset
        self.offset_reference = offset_reference
        assert offset_type in ('t', 'lnE_H')
        self.offset_type = offset_type
        self.temperature = temperature
        self.data = []

    def add_series(self, expression: str, values, sd=None, transform=None):
        assert sd is not None or len(values) > 1, 'Cannot estimate standard deviation with only one observation.'
        assert sd is None or sd > 0, 'Standard deviation must be > 0, or None for it to be estimated (it is %s)' % sd
        expression = {'WM': 'L**3 + E * w_E / mu_E / d_E'}.get(expression, expression)
        expression = compile(expression, '<string>', 'eval')
        if transform is None:
            transform = lambda x: x
        self.data.append((expression, numpy.array(values), None if sd is None else sd * sd, transform))

    def get_prior(self):
        if not isinstance(self.offset, float):
            return ['%s_offset' % self.offset_type], [self.offset[0]], [self.offset[1]**2]
        return Component.get_prior(self)

    def calculate(self, model: pydeb.Model, values):
        # Compute temperature correction factor
        c_T = model.get_temperature_correction(self.temperature)

        # Determine time offset (if any)
        t_offset = 0
        offset = self.offset if isinstance(self.offset, float) else values[0]
        if self.offset_type == 't':
            t_offset = offset
            if self.offset_reference is not None:
                t_offset += model.evaluate(self.offset_reference, c_T=c_T)
        elif self.offset_type == 'lnE_H':
            t_offset = model.age_at_maturity(numpy.exp(offset), c_T=c_T)

        # Time integration
        delta_t = (self.t[-1] + t_offset) / 1000
        nt = int((self.t[-1] + t_offset) / delta_t + 1)
        nsave = max(1, int(numpy.floor(nt / 1000)))
        result = model.simulate(nt, delta_t, nsave, c_T=c_T)

        # Extract requested expression
        t_mod = result['t'] - t_offset
        results = numpy.empty((t_mod.size, len(self.data)))
        for i, (expression, _, _, _) in enumerate(self.data):
            results[:, i] = model.evaluate(expression, c_T=c_T, locals=result)

        return t_mod, results

    def calculate_ln_likelihood(self, model: pydeb.Model, values):
        t_mod, results = self.calculate(model, values)

        lnl = 0
        for i, (_, values, var, tf) in enumerate(self.data):
            # Linearly interpolate model results to time of observations
            values_mod = numpy.interp(self.t, t_mod, results[:, i])

            # Calculate contribution to likelihood (estimate variance if not provided)
            delta2 = (tf(values_mod) - values)**2
            if var is None:
                var = delta2.sum() / (delta2.size - 1)
            lnl += -0.5 * delta2.size * numpy.log(var) - 0.5 * (delta2 / var).sum()
        return lnl

class LnLikelihood(object):
    def __init__(self, deb_type: str='abj', E_0_ini: Optional[float]=None):
        self.deb_type = deb_type
        self.E_0_ini = E_0_ini
        self.components = collections.OrderedDict()

    def add_component(self, component, name=None):
        assert isinstance(component, Component), 'prior information must be of type likelihood.Component'
        if name is None:
            name = '%i' % len(self.components)
        self.components[name] = component

    def get_prior(self):
        names, means, covs = [], [], []
        for component_name, component in self.components.items():
            current_names, current_mean, current_cov = component.get_prior()
            component.global_names = ['%s.%s' % (component_name, n) for n in current_names]
            names.extend(component.global_names)
            means.append(current_mean)
            covs.append(current_cov)
        mean = numpy.empty((len(names),))
        cov = numpy.zeros((len(names), len(names)))
        i = 0
        for current_mean, current_cov in zip(means, covs):
            n = len(current_mean)
            mean[i:i + n] = current_mean
            cov[i:i + n, i:i + n] = current_cov
            i += n
        return names, mean, cov

    def calculate(self, name2value: Mapping[str, float]):
        # Create model object
        model = pydeb.Model(type=self.deb_type)

        # Retrieve parameters specific to each likelihood component
        component_pars = [numpy.array([name2value[n] for n in component.global_names]) for component in self.components.values()]

        # Configure all likelihood components.
        # This also transfer the value of primary parameters to the model object, which must be done before calling model.initialize.
        for component, current_name2value in zip(self.components.values(), component_pars):
            component.configure(model, current_name2value)

        # Initialize the model (determine whether the model is valid, calculate implied properties, etc.)
        model.initialize(self.E_0_ini)

        # Calculate ln likelihood by summing contribution of each component
        lnl = 0
        if model.valid:
            for component, current_name2value in zip(self.components.values(), component_pars):
                lnl += component.calculate_ln_likelihood(model, current_name2value)
        return model, lnl