import collections
from typing import Iterable, List, Optional, Callable, MutableMapping, Sequence, Union

import numpy

from .. import model as pydeb

null_transform = lambda x: x

def merge_statistics(means, covs):
    n = sum([numpy.size(m) for m in means])
    mean = numpy.empty((n,))
    cov = numpy.zeros((n, n))
    i = 0
    for child_mean, child_cov in zip(means, covs):
        n = numpy.size(child_mean)
        mean[i:i + n] = child_mean
        cov[i:i + n, i:i + n] = child_cov
        i += n
    return mean, cov

class Component(object):
    def __init__(self):
        self.prefix = None
        self.children: List[Component] = []
        self.name2path: MutableMapping[str, str] = {}

        self.parameter_names = []
        self.parameter_means = []
        self.parameter_covs = []

    def add_parameters(self, names: Union[str, Iterable[str]], mean, cov):
        if isinstance(names, str):
            names = (names,)
        names = list(names)
        n = len(names)
        mean = numpy.reshape(mean, (n,))
        cov = numpy.reshape(cov, (n, n))
        self.parameter_names.extend(names)
        self.parameter_means.append(mean)
        self.parameter_covs.append(cov)

    def add_external_parameter(self, name: str, path: Optional[str]=None):
        self.name2path[name] = name if path is None else path

    def add_child(self, child: 'Component', name: Optional[str]=None):
        assert isinstance(child, Component), 'prior information must be of type likelihood.Component'
        child.prefix = self.add_prefix(name)
        self.children.append(child)

    def add_prefix(self, name: Optional[str]=None):
        names = []
        if self.prefix:
            names.append(self.prefix)
        if name is not None:
            names.append(name)
        if len(names) > 0:
            return '.'.join(names)

    def get_prior(self):
        self.parameter_mean, self.parameter_cov = merge_statistics(self.parameter_means, self.parameter_covs)
        self.invcov = numpy.linalg.inv(self.parameter_cov)
        return self.parameter_names, self.parameter_mean, self.parameter_cov

    def get_combined_prior(self):
        own_names, own_mean, own_cov = self.get_prior()
        names, means, covs = [self.add_prefix(name) for name in own_names], [own_mean], [own_cov]
        self.name2path.update(zip(own_names, names))
        for child in self.children:
            child_names, child_mean, child_cov = child.get_combined_prior()
            assert len(child_names) == numpy.size(child_mean)
            assert len(child_names)**2 == numpy.size(child_cov)
            names.extend(child_names)
            means.append(child_mean)
            covs.append(child_cov)
        mean, cov = merge_statistics(means, covs)
        return names, mean, cov

    def calculate_ln_likelihood(self, values) -> float:
        x = numpy.array([values[name] for name in self.parameter_names])
        dx = x - self.parameter_mean
        return -0.5 * dx.dot(self.invcov).dot(dx)

    def calculate_combined_ln_likelihood(self, values) -> float:
        for name, path in self.name2path.items():
            values[name] = values[path]
        result = self.calculate_ln_likelihood(values)
        if result is not None:
            for child in self.children:
                result += child.calculate_combined_ln_likelihood(values)
        return result

    def get_locals(self, values):
        return dict([(name, values[path]) for name, path in self.name2path.items()])

class Model(Component):
    def __init__(self, model_factory, names: Sequence[str], mean, cov, transforms):
        Component.__init__(self)
        self.add_parameters(names, mean, cov)
        if transforms is None:
            transforms = [null_transform for name in names]
        assert len(transforms) == len(names), 'Number of transforms (%i) does not match number of names (%i)' % (len(transforms), len(names))
        self.parameter_transforms = transforms
        self.model_factory = model_factory

    def calculate_ln_likelihood(self, values):
        model = self.model_factory()
        for name, transform in zip(self.parameter_names, self.parameter_transforms):
            setattr(model, name, transform(values[name]))

        # Initialize the model (determine whether the model is valid, calculate implied properties, etc.)
        model.initialize()

        values['model'] = model
        if model.valid:
            return Component.calculate_ln_likelihood(self, values)

class ImpliedProperty(Component):
    def __init__(self, expression: str, value: float, sd: float, transform: Callable[[float], float]=null_transform, temperature: float=20.):
        Component.__init__(self)
        self.expression = compile(expression, '<string>', 'eval')
        self.value = value
        self.sd = sd
        self.transform = transform
        self.temperature = temperature

    def calculate_ln_likelihood(self, values) -> float:
        model: pydeb.Model = values['model']
        c_T = model.get_temperature_correction(self.temperature)
        value_mod = model.evaluate(self.expression, c_T=c_T, locals=self.get_locals(values))
        z = (self.transform(value_mod) - self.value) / self.sd
        return -0.5 * z * z

class ExpressionAtSurvival(Component):
    def __init__(self, S: float, temperature: float=20., f: float=1.):
        Component.__init__(self)
        self.S = S
        self.temperature = temperature
        self.f = f
        self.expressions = []

    def add_expression(self, expression: str, value: float, sd: float, transform: Callable[[float], float]=null_transform):
        self.expressions.append((compile(expression, '<string>', 'eval'), value, sd, transform))

    def calculate_ln_likelihood(self, values) -> float:
        model: pydeb.Model = values['model']
        c_T = model.get_temperature_correction(self.temperature)
        result = model.state_at_survival(self.S, c_T=c_T, f=self.f)
        lnl = 0.
        for expression, value, sd, transform in self.expressions:
            value_mod = model.evaluate(expression, c_T=c_T, locals=result)
            z = (transform(value_mod) - value) / sd
            lnl += -0.5 * z * z
        #print(self.transform(value_mod), self.value, self.sd, z, c_T)
        return lnl

class TimeSeries(Component):
    def __init__(self, t, temperature: float=20., f: float=1., offset: float=0.):
        Component.__init__(self)
        self.t = numpy.array(t)
        self.temperature = temperature
        self.f = f
        self.offset = offset
        self.data = []

    def add_series(self, expression: str, values, sd=None, transform: Callable[[float], float]=null_transform):
        assert sd is not None or len(values) > 1, 'Cannot estimate standard deviation with only one observation.'
        assert sd is None or sd > 0, 'Standard deviation must be > 0, or None for it to be estimated (it is %s)' % sd
        expression = {'WM': 'L**3 + E * w_E / mu_E / d_E'}.get(expression, expression)
        expression = compile(expression, '<string>', 'eval')
        values = numpy.array(values)
        assert values.shape == self.t.shape, '%s: shape of values %s does not match shape of times %s' % (expression, values, self.t)
        self.data.append((expression, values, None if sd is None else sd * sd, transform))

    def calculate(self, model: pydeb.Model, values):
        # Compute temperature correction factor
        c_T = model.get_temperature_correction(self.temperature)

        # Determine time offset (if any)
        # Observation times will model times minus the offset
        t_offset = self.offset if isinstance(self.offset, float) else model.evaluate(self.offset, c_T=c_T, locals=self.get_locals(values))

        # Time integration
        delta_t = (self.t[-1] + t_offset) / 1000
        nt = int((self.t[-1] + t_offset) / delta_t + 1)
        nsave = max(1, int(numpy.floor(nt / 1000)))
        result = model.simulate(nt, delta_t, nsave, c_T=c_T, f=self.f)

        # Extract requested expression
        t_mod = result['t'] - t_offset
        results = numpy.empty((t_mod.size, len(self.data)))
        for i, (expression, _, _, _) in enumerate(self.data):
            results[:, i] = model.evaluate(expression, c_T=c_T, locals=result)

        return t_mod, results

    def calculate_ln_likelihood(self, values):
        t_mod, results = self.calculate(values['model'], values)

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

