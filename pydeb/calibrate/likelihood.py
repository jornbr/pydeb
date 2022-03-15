import collections
from typing import Iterable, List, Optional, Callable, MutableMapping, Sequence, Union, Tuple, Mapping, Any

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
        self.parameter_inverse_transforms = []

    def add_parameters(self, names: Union[str, Iterable[str]], mean, cov, inverse_transforms=None):
        """Add parameters to this likelihood component, along with their (normal) prior distribution.
        
        Arguments:

        names: iterable with names of the parameters

        mean: mean of the (normally distributed) parameters (Npar)

        cov: covariance matrix of the (normally distributed) parameters  (Npar x Npar)

        inverse_transforms: functions to transform the normally distributed parameters to their true value.
                            for instance, numpy.exp for a log-normally distributed parameter.
        """
        if isinstance(names, str):
            names = (names,)
        if inverse_transforms is None:
            inverse_transforms = null_transform
        if not isinstance(inverse_transforms, Iterable):
            inverse_transforms = (inverse_transforms,) * len(names)
        assert len(inverse_transforms) == len(names), 'Number of transforms (%i) does not match number of names (%i)' % (len(transforms), len(names))
        names = list(names)
        n = len(names)
        mean = numpy.reshape(mean, (n,))
        cov = numpy.reshape(cov, (n, n))
        self.parameter_names.extend(names)
        self.parameter_means.append(mean)
        self.parameter_covs.append(cov)
        self.parameter_inverse_transforms.extend(inverse_transforms)

    def add_external_parameter(self, name: str, path: Optional[str]=None):
        """Make a parameter defined by another likelihood component available here under a local name."""
        self.name2path[name] = name if path is None else path

    def add_child(self, child: 'Component', name: Optional[str]=None):
        """Add a nested likelihood component."""
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

    def get_prior(self) -> Tuple[Sequence[str], numpy.ndarray, numpy.ndarray]:
        self.parameter_mean, self.parameter_cov = merge_statistics(self.parameter_means, self.parameter_covs)
        self.invcov = numpy.linalg.inv(self.parameter_cov)
        return self.parameter_names, self.parameter_mean, self.parameter_cov

    def get_combined_prior(self) -> Tuple[Sequence[str], numpy.ndarray, numpy.ndarray, Sequence[Callable[[float], float]]]:
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

    def calculate_ln_likelihood(self, values) -> Optional[float]:
        """Calculate the log likelihood contribution of this component.
        
        Return None if the likelihood is not valid, e.g., for a invalid parameter set."""
        return 0.

    def evaluate(self, values) -> Optional[float]:
        """Calculate the log likelihood contribution of this component and any components nested below it.

        Return None if the likelihood is not valid, e.g., for a invalid parameter set.
        """
        x = numpy.array([values[name] for name in self.parameter_names])
        dx = x - self.parameter_mean
        lnl = -0.5 * dx.dot(self.invcov).dot(dx)

        for name, transform in zip(self.parameter_names, self.parameter_inverse_transforms):
            values[name] = transform(values[name])
        for name, path in self.name2path.items():
            values[name] = values[path]

        own_lnl = self.calculate_ln_likelihood(values)
        if own_lnl is None:
            return

        lnl += own_lnl
        for child in self.children:
            child_lnl = child.evaluate(values)
            if child_lnl is None:
                return
            lnl += child_lnl

        return lnl

    def get_locals(self, values) -> Mapping[str, Any]:
        return dict([(name, values[path]) for name, path in self.name2path.items()])

class Model(Component):
    def __init__(self, model_factory, names: Sequence[str], mean, cov, transforms):
        super().__init__()
        self.add_parameters(names, mean, cov, transforms)
        self.model_factory = model_factory

    def calculate_ln_likelihood(self, values) -> Optional[float]:
        # Create the model and transfer parameter values to it
        model = self.model_factory()
        for name in self.parameter_names:
            setattr(model, name, values[name])

        # Initialize the model (determine whether the model is valid, calculate implied properties, etc.)
        model.initialize()

        values['model'] = model

        # If the model is valid, return the default likelihood contribtion.
        # Otherwise return None to indicate the parameter combianton is invalid.
        if model.valid:
            return super().calculate_ln_likelihood(values)

class ImpliedProperty(Component):
    def __init__(self, expression: str, value: float, sd: float, transform: Callable[[float], float]=null_transform, temperature: float=20.):
        super().__init__()
        self.expression = compile(expression, '<string>', 'eval')
        self.value = value
        self.sd = sd
        self.transform = transform
        self.temperature = temperature

    def calculate_ln_likelihood(self, values) -> Optional[float]:
        model: pydeb.Model = values['model']
        c_T = model.get_temperature_correction(self.temperature)
        value_mod = model.evaluate(self.expression, c_T=c_T, locals=self.get_locals(values))
        z = (self.transform(value_mod) - self.value) / self.sd
        return -0.5 * z * z

class ExpressionAtEvent(Component):
    def __init__(self, S_crit: Optional[float]=None, E_H_crit: Optional[float]=None, temperature: float=20., **kwargs):
        super().__init__()
        self.S_crit = S_crit
        self.E_H_crit = E_H_crit
        self.temperature = temperature
        self.kwargs = kwargs
        self.expressions = []

    def add_expression(self, expression: str, value: float, sd: float, transform: Callable[[float], float]=null_transform):
        self.expressions.append((compile(expression, '<string>', 'eval'), value, sd, transform))

    def calculate_ln_likelihood(self, values) -> float:
        model: pydeb.Model = values['model']
        c_T = model.get_temperature_correction(self.temperature)
        result = model.state_at_event(S_crit=self.S_crit, E_H_crit=self.E_H_crit, **self.kwargs)
        lnl = 0.
        for expression, value, sd, transform in self.expressions:
            value_mod = model.evaluate(expression, c_T=c_T, locals=result)
            z = (transform(value_mod) - value) / sd
            lnl += -0.5 * z * z
        #print(self.transform(value_mod), self.value, self.sd, z, c_T)
        return lnl

class ExpressionAtSurvival(ExpressionAtEvent):
    def __init__(self, S: float, **kwargs):
        super().__init__(S_crit=S, **kwargs)

class TimeSeries(Component):
    def __init__(self, t, temperature: float=20., before_simulate=None, t_end: Optional[float]=None, **kwargs):
        super().__init__()
        self.t = numpy.array(t)
        assert self.t.ndim <= 1, 'Time has shape %s, but it must be a scalar or 1-dimensional array' % (self.t.shape,)
        self.temperature = temperature
        self.kwargs = kwargs
        self.before_simulate = before_simulate
        self.data = []

    def add_series(self, expression: str, values, sd=None, transform: Callable[[float], float]=null_transform):
        expression = {'WM': 'L**3 + E * w_E / mu_E / d_E'}.get(expression, expression)
        expression = compile(expression, '<string>', 'eval')
        values = numpy.array(values)
        assert values.shape == self.t.shape, '%s: shape of values %s does not match shape of times %s' % (expression, values, self.t)
        assert sd is not None or values.size > 1, 'Cannot estimate standard deviation with only one observation.'
        if sd is not None:
            sd = numpy.broadcast_to(sd, values.shape)
        assert sd is None or (sd > 0).all(), 'Standard deviation must be > 0, or None for it to be estimated (it is %s)' % sd
        self.data.append((expression, values, None if sd is None else sd * sd, transform))

    def calculate(self, model: pydeb.Model, values):
        loc = self.get_locals(values)

        # Compute temperature correction factor
        c_T = model.get_temperature_correction(self.temperature)

        kwargs = self.kwargs.copy()
        kwargs['times'] = self.t
        kwargs['c_T'] = c_T
        if self.before_simulate is not None:
            self.before_simulate(model, kwargs, **loc)

        # Time integration
        result = model.simulate2(**kwargs)

        results = numpy.empty(result['t'].shape + (len(self.data),))
        for i, (expression, _, _, _) in enumerate(self.data):
            results[:, i] = model.evaluate(expression, c_T=c_T, locals=result)

        return results

    def calculate_ln_likelihood(self, values):
        results = self.calculate(values['model'], values)

        lnl = 0.
        for i, (_, values, var, tf) in enumerate(self.data):
            # Linearly interpolate model results to time of observations
            values_mod = results[..., i]

            # Calculate contribution to likelihood (estimate variance if not provided)
            delta2 = (tf(values_mod) - values)**2
            if var is None:
                var = numpy.broadcast_to(delta2.sum() / (delta2.size - 1), delta2.shape)
            lnl += -0.5 * numpy.log(var).sum() - 0.5 * (delta2 / var).sum()
        return lnl

