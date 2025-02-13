import math
from typing import Dict, Any, Mapping, Optional, Tuple, Union, Sequence, Iterable, Callable
import collections.abc
import functools
import itertools

import numpy

from . import engine

primary_parameters = 'T_A', 'p_Am', 'F_m', 'kap_X', 'kap_P', 'v', 'kap', 'kap_R', 'p_M', 'p_T', 'k_J', 'E_G', 'E_Hb', 'E_Hx', 'E_Hj', 'E_Hp', 'h_a', 's_G', 't_0'

entry_parameters = 'del_M',

implied_properties = 'L_b', 'L_p', 'L_i', 'a_b', 'a_p', 'a_99', 'E_0', 'E_m', 'r_B', 'R_i'

long_names = {
    'T_A': 'Arrhenius temperature',
    'p_Am': 'specific assimilation flux',
    'F_m': 'maximum specific searching rate',
    'kap_X': 'digestion efficiency (fraction of food to reserve)',
    'kap_P': 'faecation efficiency (fraction of food to faeces)',
    'v': 'energy conductance',
    'kap': 'allocation fraction to soma',
    'kap_R': 'reproduction efficiency',
    'p_M': 'volume-specific somatic maintenance',
    'p_T': 'surface-specific somatic maintenance',
    'k_J': 'maturity maintenance rate coefficient',
    'E': 'reserve',
    'E_G': 'specific cost for structure',
    'E_H': 'maturity',
    'E_Hb': 'maturity at birth',
    'E_Hj': 'maturity at metamorphosis',
    'E_Hx': 'maturity at weaning/fledgling',
    'E_Hp': 'maturity at puberty',
    'h_a': 'Weibull aging acceleration',
    's_G': 'Gompertz stress coefficient',
    't_0': 'time at start of development',
    'L': 'structural length',
    'L_b': 'structural length at birth',
    'L_j': 'structural length at metamorphosis',
    'L_p': 'structural length at puberty',
    'L_i': 'ultimate structural length',
    'a_b': 'age at birth',
    'a_j': 'age at metamorphosis',
    'a_p': 'age at puberty',
    'a_99': 'age when reaching 99% of ultimate structural length',
    'a_m': 'expected life span',
    'R': 'reproduction rate',
    'R_i': 'ultimate reproduction rate',
    'r_B': 'von Bertalanffy growth rate',
    'E_m': 'reserve capacity',
    'E_0': 'initial reserve',
    's_M': 'acceleration at metamorphosis',
    'del_M': 'shape coefficient (structural : physical length)',
    'S': 'survival',
    'N_R': 'cumulative reproductive output',
    'N_RS': 'expected cumulative reproductive output (accounting for survival)',
    'N_i': 'expected lifetime reproductive output',
}

units = {
    'T_A': 'K',
    'p_Am': 'J/d.cm^2',
    'F_m': 'l/d.cm^2',
    'kap_X': '-',
    'kap_P': '-',
    'v': 'cm/d',
    'kap': '-',
    'kap_R': '-',
    'p_M': 'J/d.cm^3',
    'p_T': 'J/d.cm^2',
    'k_J': '1/d',
    'E': 'J',
    'E_G': 'J/cm^3',
    'E_H': 'J',
    'E_Hb': 'J',
    'E_Hj': 'J',
    'E_Hx': 'J',
    'E_Hp': 'J',
    'h_a': '1/d^2',
    's_G': '-',
    't_0': 'd',
    'L': 'cm',
    'L_b': 'cm',
    'L_j': 'cm',
    'L_p': 'cm',
    'L_i': 'cm',
    'a_b': 'd',
    'a_j': 'd',
    'a_p': 'd',
    'a_99': 'd',
    'a_m': 'd',
    'R': '1/d',
    'R_i': '1/d',
    'r_B': '1/d',
    'E_m': 'J/cm^3',
    'E_0': 'J',
    's_M': '-',
    'del_M': '-',
    'S': '-',
    'N_R': '#',
    'N_RS': '#',
    'N_i': '#',
}

temperature_correction = {
    'T_A': 0,
    'p_Am': 1,
    'F_m': 1,
    'kap_X': 0,
    'kap_P': 0,
    'v': 1,
    'kap': 0,
    'kap_R': 0,
    'p_M': 1,
    'p_T': 1,
    'k_J': 1,
    'E_G': 0,
    'E_Hb': 0,
    'E_Hj': 0,
    'E_Hx': 0,
    'E_Hp': 0,
    'h_a': 2,
    's_G': 0,
    't_0': -1,
    'L_b': 0,
    'L_j': 0,
    'L_p': 0,
    'L_i': 0,
    'a_b': -1,
    'a_j': -1,
    'a_p': -1,
    'a_99': -1,
    'a_S10': -1,
    'a_S01': -1,
    'a_m': -1,
    'R_i': 1,
    'r_B': 1,
    'E_m': 0,
    'E_0': 0,
    's_M': 0,
    'del_M': 0,
    'mu_E': 0,
    'w_E': 0,
    'd_E': 0,
    'N_i': 0,
    'N_R': 0,
    'N_RS': 0,
    'S_b': 0,
    'S_p': 0,
    'S_j': 0,
    'age_at_maturity': None, # None means a function or method that takes c_T as argument
    'state_at_survival': None, # None means a function or method that takes c_T as argument
    'state_at_time': None, # None means a function or method that takes c_T as argument
    'state_at_event': None, # None means a function or method that takes c_T as argument
    'state_at_maturity': None, # None means a function or method that takes c_T as argument
}

typified_models = {'std': 'standard DEB model',
                   'stf': 'with foetal development (no egg)',
                   'stx': 'with foetal development and optional preparation stage',
                   'abj': 'optional acceleration between birth and metamorphosis'
                   }

compound_variables = {
    'L_w': 'L/del_M'
}

dot_symbols = ('p', 'J', 'j', 'k', 'F', 'v', 'r', 'R')
ddot_symbols = ('h',)
greek_symbols = {'kap': 'kappa', 'del': 'delta'}


def symbol2html(symbol: str) -> str:
    original = symbol
    for s, h in greek_symbols.items():
        if symbol == s or symbol.startswith(s + '_'):
            symbol = '&%s;%s' % (h, symbol[len(s):])
    if '_' in symbol:
        parts = symbol.split('_', 1)
        parts[0] = '<i>%s</i>' % parts[0]
        if not parts[1].isdigit():
            parts[1] = '<i>%s</i>' % parts[1]
        symbol = '%s<sub>%s</sub>' % tuple(parts)
    else:
        symbol = '<i>%s</i>' % symbol
    if original in ('F_m', 'p_Am', 'p_T'):
        symbol = '{%s}' % symbol
    if original in ('E_G', 'p_M', 'E_m'):
        symbol = '[%s]' % symbol
    return symbol


def symbol2mathtext(symbol: str) -> str:
    original = symbol
    parts = symbol.split('_', 1)
    if parts[0] in dot_symbols:
        parts[0] = r'\dot{%s}' % parts[0]
    elif parts[0] in ddot_symbols:
        parts[0] = r'\ddot{%s}' % parts[0]
    elif parts[0] in greek_symbols:
        parts[0] = r'\%s' % greek_symbols[parts[0]]
    symbol = parts[0] if len(parts) == 1 else '%s_{%s}' % tuple(parts)
    if original in ('F_m', 'p_Am', 'p_T'):
        symbol = r'\left\{ %s \right\}' % symbol
    if original in ('E_G', 'p_M', 'E_m'):
        symbol = r'\left[ %s \right]' % symbol
    return symbol


class ModelDict(collections.abc.Mapping):
    """Evaluates a expression combining model parameters and implied properties (to be temperature corrected)
    and optionally an additonals "locals" dictionary with additional objects - e.g., model results. The latter
    takes priority if provided. Its contained varables are assumed to *already have been temperature-corrected*."""

    def __init__(self, model: 'Model', c_T: float=1., locals: Mapping[str, Any]={}):
        self.model = model
        self.c_T = c_T
        self.locals = {'c_T': c_T}
        for key in ('log', 'log10', 'exp'):
            self.locals[key] = getattr(numpy, key)
        self.locals.update(locals)

    def __getitem__(self, key: str):
        if key in self.locals:
            return self.locals[key]
        if key in compound_variables:
            return eval(compound_variables[key], {}, self)
        if not hasattr(self.model, key):
            raise KeyError()
        value = getattr(self.model, key)
        assert key in temperature_correction, 'No temperature correction available for %s' % key
        correction = temperature_correction[key]
        if correction is None:
            # This item is a function or bound method that takes c_T as named argument
            value = functools.partial(value, c_T=self.c_T)
        else:
            # This item is a numerical value, e.g., a primary parameter
            value *= self.c_T**correction
        return value

    def __len__(self) -> int:
        return len(frozenset(list(self.locals.keys()) + list(compound_variables.keys()) + dir(self.model)))

    def __iter__(self):
        return frozenset(list(self.locals.keys()) + list(compound_variables.keys()) + dir(self.model)).__iter__()


class Model(object):
    def __init__(self, type: str='abj'):
        self.p_Am = None   # {p_Am}, spec assimilation flux (J/d.cm^2)
        self.v = None      # energy conductance (cm/d)
        self.p_M = None    # [p_M], vol-spec somatic maint, J/d.cm^3
        self.p_T = 0.      # {p_T}, surf-spec somatic maint, J/d.cm^2
        self.kap = None
        self.E_G = None    # [E_G], spec cost for structure
        self.E_Hb = None   # maturity at birth (J)
        self.E_Hp = None   # maturity at puberty (J)
        self.E_Hj = None   # maturity at metamorphosis (J)
        self.k_J = None    # k_J: maturity maint rate coefficient, 1/d
        self.h_a = None    # Weibull aging acceleration (1/d^2)
        self.s_G = None    # Gompertz stress coefficient
        self.kap_R = None  # reproductive efficiency
        self.kap_X = None  # digestion efficiency of food to reserve
        self.T_A = None    # Arrhenius temperature
        self.type = type   # std, abj, stf, stx

        # stf
        # foetal development (rather than egg development)
        # e -> \Inf, p 64, r = v/L

        # stx:
        # foetal development (rather than egg development) that first starts with a preparation stage and then sparks off at a time that is an extra parameter
        # a baby stage (for mammals) just after birth, ended by weaning, where juvenile switches from feeding on milk to solid food at maturity level EHx. Weaning is between birth and puberty.
        # t_0: time at start development - stx model
        # E_Hx: maturity at weaning (J) - stx model

        # ssj:
        # a non-feeding stage between events s and j during the juvenile stage that is initiated at a particular maturity level and lasts a particular time. Substantial metabolically controlled shrinking occur during this period, more than can be explained by starvation.
        # E_Hs: maturity at S2/S3 transition - ssj model
        # t_sj: period of metamorphosis - model ssj

        # sbp
        # growth ceasing at puberty, meaning that the kappa-rule is not operational in adults.

        # abj:
        # - acceleration between birth and metamorphosis (V1-morph)
        # - before and after acceleration: isomorphy
        # Metamorphosis is before puberty and occurs at maturity EHj, which might or might not correspond with changes in morphology. This model is a one-parameter extension of model std.
        # E_Hj: maturity at metam (J) - abp model

        # asj
        # - start of acceleration is delayed till maturity level EHs
        # - before and after acceleration: isomorphy.

        # abp
        # - acceleration between birth and puberty (V1-morph)
        # - before acceleration: isomorphy
        # - after acceleration: no growth, so no kappa-rule

        # hep:
        # - morphological life stages: egg, larva, (sub)imago; functional stages: embryo, juvenile, adult, imago
        # - the embryo still behaves like model std
        # - acceleration starts at birth and ends at puberty
        # - puberty occurs during the larval stage
        # - emergence occurs when reproduction buffer density hits a threshold
        # - the (sub)imago does not grow or allocate to reproduction. It mobilises reserve to match constant (somatic plus maturity) maintenance
        # E_Rj: reproduction buffer density at emergence (J/cm^3) - hep model

        # hex
        # The DEB model for holometabolic insects (and some other hexapods). It characterics are
        # - morphological life stages: egg, larva, (pupa), imago; functional stages: embryo, adult, (pupa), imago
        # - the embryo still behaves like model std
        # - the larval stage accelerates (V1-morph) and behaves as adult, i.e. no maturation, allocation to reproduction
        # - pupation occurs when reproduction buffer density hits a threshold
        # - pupa behaves like an isomorphic embryo of model std, but larval structure rapidly transforms to pupal reserve just after start of pupation
        # - the reproduction buffer remains unchanged during the pupal stage
        # - the imago does not grow or allocate to reproduction. It mobilises reserve to match constant (somatic plus maturity) maintenance
        # E_He: maturity at emergence (J) - hex model
        # kap_V: conversion efficient E -> V -> E - hex model

        # kap_P: digestion efficiency of food to faeces
        # F_m {F_m}, max spec searching rate (l/d.cm^2)
        # s_j: reprod buffer/structure at pupation as fraction of max (-) - hex model

        # pars specific for this entry
        # del_M = L/Lw: shape coefficient (-)

        # derived parameters
        self.E_0 = None
        self.L_b = None
        self.a_b = None
        self.a_99 = None
        self.end_state_ = None
        # L_m = kappa*v*E_m/p_M = kappa*p_Am/p_M [L_m is maximum length in absence of surface=-area-specific maintenance!]
        # z = L_m/L_m_ref with L_m_ref = 1 cm - equal to L_m

        self.initialized = False
        self.valid = False
        self.engine = None

        self.mu_E = 5.5e5  # chemical potential of reserve (J/C-mol)
        self.w_E = 23.9   # dry weight of reserve (g/C_mol)
        self.d_E = 0.21   # specific density of reserve (g DM/cm3)
        # self.WM_per_E = self.w_E / self.mu_E / self.d_E # cm3/J

        self.f2E_0 = {}

    def copy(self, **parameters):
        clone = Model(type=parameters.get('type', self.type))
        for p in primary_parameters + entry_parameters:
            if hasattr(self, p):
                setattr(clone, p, parameters.get(p, getattr(self, p)))
        return clone

    def initialize(self, verbose: bool=False, precision: float=0.001):
        assert self.p_T >= 0., 'p_T has invalid value %s' % self.p_T
        assert self.p_M >= 0., 'p_M has invalid value %s' % self.p_M
        assert self.p_Am >= 0., 'p_Am has invalid value %s' % self.p_Am
        assert self.v >= 0., 'v has invalid value %s' % self.v
        assert self.E_G >= 0., 'E_G has invalid value %s' % self.E_G
        self.valid = False
        self.kap = max(min(self.kap, 1.), 0.)
        self.kap_R = max(min(self.kap_R, 1.), 0.)
        self.kap_X = max(min(self.kap_X, 1.), 0.)
        self.E_Hj = self.E_Hb if self.type in ('stf', 'stx') else max(self.E_Hb, self.E_Hj)
        self.E_Hp = max(self.E_Hb, self.E_Hp)
        self.initialized = True
        self.devel_state_ini = -1 if self.type in ('stf', 'stx') else 1

        kap = self.kap
        v = self.v
        E_G = self.E_G
        E_Hb = self.E_Hb
        k_J = self.k_J
        E_m = self.p_Am / self.v  # defined at f=1
        g = E_G / kap / E_m
        self.k_M = self.p_M / E_G
        L_m = kap * self.p_Am / self.p_M
        L_T = self.p_T / self.p_M
        p_M = self.p_M
        p_T = self.p_T

        assert E_m >= 0
        assert L_m >= 0
        assert self.h_a >= 0
        #assert self.s_G > 0

        E_G_per_kap = E_G/kap
        p_M_per_kap = p_M/kap
        p_T_per_kap = p_T/kap

        self.engine = engine.create()
        for parameter in primary_parameters:
            if hasattr(self.engine, parameter):
                setattr(self.engine, parameter, getattr(self, parameter))
        get_E_0 = self.engine.get_E_0
        get_birth_state = self.engine.get_birth_state
        find_maturity = self.engine.find_maturity
        find_maturity_v1 = self.engine.find_maturity_v1
        find_maturity_foetus = self.engine.find_maturity_foetus

        # Compute maximum catabolic flux (before any acceleration)
        # This flux needs to be able to at least support [= pay maintenance for] maturity at birth.
        L_i_min = L_m - L_T   # not counting acceleration
        if L_i_min < 0:
            if verbose:
                print('Ultimate structural length is negative: %s cm' % (L_i_min,))
            return
        p_C_i = L_i_min * L_i_min * E_m * ((v * E_G_per_kap + p_T_per_kap) + p_M_per_kap * L_i_min) / (E_m + E_G_per_kap)
        if k_J * E_Hb > (1 - kap) * p_C_i:
            if verbose:
                print('Cannot reach maturity of %s J at birth (maximum reachable is %s J).' % (E_Hb, (1 - kap) * p_C_i / k_J))
            return

        delta_t = 1.
        if self.type in ('stf', 'stx'):
            # foetal development
            while 1:
                a_b, L_b, E_0 = find_maturity_foetus(self.E_Hb, delta_t)
                if a_b == -1.:
                    if verbose:
                        print('Unable to determine age/length at birth.')
                    return
                # Stop if we have the time of birth at desired accuracy
                # (assume linear interpolation within Euler time step results in accuracy of 0.1 delta_t)
                if delta_t < a_b * precision * 10:
                    break
                delta_t = a_b * precision * 5
        else:
            # egg development

            # First bracket the initial reserve E_0
            # At least E_Hb / (1 - kap) must have been spent during embryo development to reach E_Hb.
            # In fact, energy expenditure MUST be more because of maturity maintenance and because there
            # needs to be reserve left over at time of hatching. This gives us a lower bound for E_0.
            # The upper bound is found by multiplying the lower bound by 10 until the resulting
            # maturity-at-hatching exceeds the desired maturity-at-birth E_Hb.
            E_0_min = E_0_max = E_Hb / (1 - kap)
            assert get_birth_state(E_0_min, delta_t, 1.)[2] <= E_Hb
            for _ in range(10):
                E_0_max *= 10
                if get_birth_state(E_0_max, delta_t, 1.)[2] >= E_Hb:
                    break
            else:
                if verbose:
                    print('Cannot find valid initial estimate for E_0 (tried up to %s)' % E_0_max)
                return

            #def root(E_0, delta_t):
            #    _, _, E_H = get_birth_state(E_0, delta_t, 1.)
            #    return E_H - E_Hb
            if verbose:
                print('Determining cost of an egg and state at birth...')
            while 1:
                #E_0 = scipy.optimize.brentq(root, E_0_min, E_0_max, rtol=precision, args=(delta_t,))
                #a_b, L_b, _ = get_birth_state(E_0, delta_t, 1.)
                E_0, a_b, L_b = get_E_0(E_0_min, E_0_max, delta_t=delta_t, precision=precision)

                E_0_max = 2 * E_0

                # Stop if we have the time of birth at desired accuracy
                # (assume linear interpolation within Euler time step results in accuracy of 0.1 delta_t)
                if delta_t < a_b * precision * 10:
                    break
                delta_t = a_b * precision * 5

        self.E_0, self.a_b, self.L_b = E_0, a_b, L_b

        self.L_m = L_m
        self.L_T = L_T
        self.E_m = E_m

        # Set time step for integration to metamorphosis and puberty
        delta_t = self._choose_delta_t(precision=precision)

        self.r_B = self.p_M / 3 / (E_m * kap + E_G)  # checked against p52, note f=1
        if L_i_min < self.L_b:
            # shrinking directly after birth
            if verbose:
                print('Shrinking directly after birth (L_i_min < L_b).')
            return
        a_99_max = self.a_b - numpy.log(1 - (0.99*L_i_min - self.L_b)/(L_i_min - self.L_b))/self.r_B

        self.engine.L_b = self.L_b
        self.engine.r_B = self.r_B
        self.engine.L_m = self.L_m
        self.engine.L_T = self.L_T

        # metamorphosis
        if self.E_Hj > self.E_Hb:
            if verbose:
                print('Determining age and length at metamorphosis...')
            self.a_j, self.L_j = find_maturity_v1(self.L_b, self.E_Hb, self.E_Hj, delta_t=delta_t, t_max=min(100*a_99_max, 365*200.), t_ini=self.a_b)
            if self.a_j == -1.:
                if verbose:
                    print('Cannot determine age and length at metamorphosis.')
                return
        else:
            self.a_j, self.L_j = self.a_b, self.L_b
        self.s_M = self.L_j / self.L_b
        self.engine.s_M = self.s_M
        self.L_i = (self.L_m - self.L_T) * self.s_M
        self.a_99 = self.a_j - numpy.log(1 - (0.99 * self.L_i - self.L_j) / (self.L_i - self.L_j)) / self.r_B
        p_C_i = self.L_i * self.L_i * E_m * ((v * E_G_per_kap + p_T_per_kap) * self.s_M + p_M_per_kap * self.L_i) / (E_m + E_G_per_kap)
        if k_J * self.E_Hp > (1 - kap) * p_C_i:
            if verbose:
                print('Cannot reach maturity of %s J at puberty (maximum reachable is %s J).' % (self.E_Hp, (1 - kap) * p_C_i / k_J))
            return
        self.R_i = self.kap_R * ((1 - kap) * p_C_i - k_J * self.E_Hp) / self.E_0

        if verbose:
            print('Determining age and length at puberty...')
        if self.E_Hp >= self.E_Hj:
            # puberty after metamorphosis
            self.a_p, self.L_p = find_maturity(self.L_j, self.E_Hj, self.E_Hp, delta_t=delta_t, s_M=self.s_M, t_max=min(100 * self.a_99, 365 * 200.), t_ini=self.a_j)
        else:
            # puberty before metamorphosis
            self.a_p, self.L_p = find_maturity_v1(self.L_b, self.E_Hb, self.E_Hp, delta_t=delta_t, t_max=self.a_j, t_ini=self.a_b)
        if self.a_p == -1.:
            if verbose:
                print('Cannot determine age and length at puberty.')
            return
        self.valid = True
        self.maturity_states: Dict[float, Tuple[float, float]] = {
            self.E_Hb: (self.a_b, self.L_b),
            self.E_Hj: (self.a_j, self.L_j),
            self.E_Hp: (self.a_p, self.L_p)
        }
        self.f2E_0[1.] = self.E_0

    def _choose_delta_t(self, c_T: float=1., precision: float=0.001, max_delta_t=numpy.inf) -> float:
        return min(max_delta_t, max(precision * 10, self.a_b * precision * 10) * 2 / c_T)

    def age_at_maturity(self, E_H: float, precision: float=0.001, c_T: float=1.) -> float:
        """Get age (time since start of development) at specific maturity value."""
        assert E_H >= 0. and E_H <= self.E_Hp, 'E_H is %s but must take a value between 0 and E_Hp=%s (inclusive)' % (E_H, self.E_Hp)
        if E_H not in self.maturity_states:
            delta_t = self._choose_delta_t(c_T, precision)
            tmax = min(100 * self.a_99, 365 * 200.)
            if E_H < self.E_Hb:
                # before birth
                if self.type in ('stf', 'stx'):
                    a, L, _ = self.engine.find_maturity_foetus(E_H, delta_t=delta_t, t_max=tmax)
                else:
                    a, L = self.engine.find_maturity_egg(E_H, delta_t=delta_t, E_0=self.E_0, t_max=tmax)
            elif E_H < self.E_Hj:
                # before metamophosis (i.e., during acceleration in V1 morph mode)
                a, L = self.engine.find_maturity_v1(self.L_b, self.E_Hb, E_H, delta_t=delta_t, t_max=tmax, t_ini=self.a_b)
            else:
                # after metamorphosis
                a, L = self.engine.find_maturity(self.L_j, self.E_Hj, E_H, delta_t=delta_t, s_M=self.s_M, t_max=tmax, t_ini=self.a_j)
            self.maturity_states[E_H] = (a, L)
        a, L = self.maturity_states[E_H]
        return a / c_T

    def evaluate(self, expression: str, c_T: float=1., locals: Mapping[str, Any]={}):
        """Compute expression that can contain any DEB parameter, trait, as well as additional variables provided with the "locals" argument.
        If a temperature correction factor (c_T) other than 1 is specified, the result will be temperature-corrected accordingly.
        But note that the values in locals are then assumed to already have been temperature corrected!"""
        return eval(expression, {}, ModelDict(self, c_T, locals=locals))

    def get_temperature_correction(self, T: float) -> float:
        """Compute temperature correction factor c_T from specified body temperature (degrees Celsius).
        This is based on the Arrhenius relationship."""
        assert numpy.all(T < 200.), 'Temperature must be given in degrees Celsius'
        return numpy.exp(self.T_A / 293.15 - self.T_A / (273.15 + T))

    def writeFABMConfiguration(self, path: str, name: str='deb', model: str='deb/population'):
        if not self.initialized:
            self.initialize()
        with open(path, 'w') as f:
            f.write('instances:\n')
            f.write('  %s:\n' % name)
            f.write('    model: %s\n' % model)
            f.write('    parameters:\n')
            f.write('      p_Am: %s\n' % self.p_Am)
            f.write('      p_T: %s\n' % self.p_T)
            f.write('      p_M: %s\n' % self.p_M)
            f.write('      E_Hb: %s\n' % self.E_Hb)
            f.write('      E_Hj: %s\n' % self.E_Hj)
            f.write('      E_Hp: %s\n' % self.E_Hp)
            f.write('      v: %s\n' % self.v)
            f.write('      k_J: %s\n' % self.k_J)
            f.write('      kap: %s\n' % self.kap)
            f.write('      kap_R: %s\n' % self.kap_R)
            f.write('      E_G: %s\n' % self.E_G)
            f.write('      h_a: %s\n' % self.h_a)
            f.write('      s_G: %s\n' % self.s_G)
            f.write('      T_A: %s\n' % self.T_A)

            f.write('      L_b: %s\n' % self.L_b)
            f.write('      L_j: %s\n' % self.L_j)
            f.write('      E_0: %s\n' % self.E_0)

    def describe(self, c_T: float=1.):
        """Report implied properties/traits."""
        if not self.initialized:
            self.initialize()
        if not self.valid:
            print('This parameter set is not valid.')
            return
        d = ModelDict(self, c_T)
        print('c_T [temperature correction factor]: %.3f' % c_T)
        print('Primary parameters:')
        for name in primary_parameters:
            print('  %s [%s]: %.4g %s' % (name, long_names[name], eval(name, {}, d), units[name]))
        print('Implied traits:')
        for name in ('E_0', 'r_B', 'a_b', 'a_j', 'a_p', 'a_99', 'E_m', 'L_b', 'L_j', 'L_p', 'L_i', 's_M', 'R_i', 'a_m', 'N_i'):
            print('  %s [%s]: %.4g %s' % (name, long_names[name], eval(name, {}, d), units[name]))

    @property
    def end_state(self) -> Mapping[str, float]:
        if self.end_state_ is None:
            self.end_state_ = self.state_at_survival(S=0.0001)
        return self.end_state_

    a_m = property(lambda self: self.end_state['a'])
    N_i = property(lambda self: self.end_state['N_RS'])
    a_S01 = property(lambda self: self.state_at_survival(0.01)['t'])
    a_S10 = property(lambda self: self.state_at_survival(0.10)['t'])
    S_b = property(lambda self: self.state_at_time(self.a_b)['S'])
    S_p = property(lambda self: self.state_at_time(self.a_p)['S'])
    S_j = property(lambda self: self.state_at_time(self.a_j)['S'])

    def _unpack_state(self, raw: numpy.ndarray, E_0: float, ind: Union[slice, int]=Ellipsis) -> numpy.ndarray:
        assert raw.shape[-1] == 11
        result = {
            't': raw[ind, 0],
            'E': raw[ind, 1],
            'L': raw[ind, 2],
            'E_H': numpy.minimum(raw[ind, 3], self.E_Hp),
            'Q': raw[ind, 4],
            'H': raw[ind, 5],
            'S': raw[ind, 6],
            'N_RS': raw[ind, 7],
            'a': raw[ind, 8],
            's': raw[ind, 9],
            'R': raw[ind, 10],
        }
        result['E_R'] = self.kap_R * (raw[ind, 3] - result['E_H'])
        result['N_R'] = result['E_R'] * (1.0 / E_0)
        return result

    def _get_initial_state(self, f_egg: Optional[float]=None, E_0: Optional[float]=None, y_ini: Optional[Mapping[str, float]]=None) -> numpy.ndarray:
        if y_ini is not None:
            y = numpy.array([y_ini[n] for n in ('E', 'L', 'E_H', 'Q', 'H', 'S', 'N_RS', 'a', 's')])
        else:
            y = numpy.zeros((9,))
            if self.devel_state_ini != -1:
                # initial development stage is not foetus (without reserve) but egg (with reserve)
                y[0] = E_0 if E_0 is not None else self.E_0_at_f(f_egg if f_egg is not None else 1.)
            y[5] = 1.   # S: initially 100% survival
            y[8] = 1.   # s: initial acceleration is 1
        return y

    def state_at_survival(self, S: float, **kwargs) -> Mapping[str, float]:
        """Get the model state at a specified value of the survival function (the probability of individuals surviving, starting at 1 and dropping to 0 over time)"""
        return self.state_at_event(S_crit=S, **kwargs)

    def state_at_maturity(self, E_H: float, **kwargs) -> Mapping[str, float]:
        """Get the model state at a specified maturity value (starting at 0 and increasing to E_Hp)"""
        return self.state_at_event(E_H_crit=E_H, **kwargs)

    def state_at_event(self, S_crit: Optional[float]=None, E_H_crit: Optional[float]=None, c_T: float=1., f: float=1., delta_t: Optional[float]=None, t_max: float=365*100, precision: float=0.001, events: Sequence=(), event_callback=None, **kwargs) -> Mapping[str, float]:
        """Get the model state at a specified value of the survival function (S_crit) or maturity (E_H_crit).
        If both are None, integrate to the specified maximum end time (t_max)"""
        if not self.initialized:
            self.initialize()
        assert self.valid, 'Model parameterisation is not valid'
        assert f >= 0. and f <= 1., 'Invalid functional response f=%s (it must lie between 0 and 1)' % f
        assert S_crit is None or (S_crit > 0. and S_crit <= 1.), 'S_crit is %s but must take a value between 0 (exclusive) and 1 (inclusive)' % S_crit
        assert E_H_crit is None or (E_H_crit >= 0. and E_H_crit <= self.E_Hp), 'E_H_crit is %s but must take a value between 0 and E_Hp=%s (inclusive)' % (E_H_crit, self.E_Hp)
        if delta_t is None:
            delta_t = self._choose_delta_t(c_T, precision)

        overridden_params = set()

        kwargs.setdefault('f_egg', f)
        result = numpy.empty((1, 11))
        y_ini = result[0, 1:-1]
        y_ini[:] = self._get_initial_state(**kwargs)

        t_start = 0.
        E_0 = self.E_0_at_f(f)
        for event_time in itertools.chain(events, [numpy.inf]):
            t_end = min(event_time, t_max)
            duration = t_end - t_start
            n = int(math.ceil(duration / delta_t))
            current_delta_t = duration / n if n > 0 else delta_t
            self.engine.integrate(n, current_delta_t, 0, result, E_0, c_T=c_T, f=f, devel_state_ini=self.devel_state_ini, S_crit=S_crit or -1., E_H_crit=E_H_crit or -1., y_ini=y_ini)
            if (S_crit is not None and result[0, 6] <= S_crit) or (E_H_crit is not None and result[0, 3] >= E_H_crit) or t_end == t_max:
                result[0, 0] += t_start
                break
            t_start = t_end
            c_T, f, E_0, params = event_callback(event_time, self, y_ini, c_T, f, E_0)
            for p, v in params.items():
                overridden_params.add(p)
                setattr(self.engine, p, v)
        for p in overridden_params:
            setattr(self.engine, p, getattr(self, p))
        return self._unpack_state(result, E_0, 0)

    def _get_parameters_as_string(self) -> str:
        return ', '.join(['%s=%s' % (n, getattr(self, n)) for n in primary_parameters])

    def E_0_at_f(self, f) -> Optional[float]:
        if not self.initialized:
            self.initialize()
        assert self.valid, 'Model parameterisation is not valid'
        if self.devel_state_ini != 1:
            f = 1.
        if f not in self.f2E_0:
            L_i = max(0., f * self.kap * self.p_Am / self.p_M - self.p_T / self.p_M)
            E_m = f * self.E_m
            p_C_i = L_i * L_i * E_m * (self.v * self.E_G + self.p_M * L_i + self.p_T) / (self.kap * E_m + self.E_G)
            if self.k_J * self.E_Hb > (1 - self.kap) * p_C_i:
                # Cannot support maturity at birth at this low value for the functional response
                self.f2E_0[f] = None
            else:
                E_0_min = self.E_Hb / (1. - self.kap)
                precision = 0.001
                delta_t = 10. * precision * self.a_b
                state = self.engine.get_birth_state(1.1*self.E_0, delta_t, f)
                assert state[2] > self.E_Hb, 'Expected E_Hb=%s at f=%s to be larger than original E_Hb=%s at f=1 (both using original E_0=%s) - %s. Parameterization: %s' % (state[2], f, self.E_Hb, self.E_0, state, self._get_parameters_as_string())
                self.f2E_0[f] = self.engine.get_E_0(E_0_min, 1.1*self.E_0, delta_t=delta_t, precision=precision, f=f)[0]
                self.f2E_0[f] = min(self.f2E_0[f], self.E_0)
        return self.f2E_0[f]

    def state_at_time(self, t: float, **kwargs) -> Mapping[str, numpy.ndarray]:
        return self.state_at_event(t_max=t, **kwargs)

    def simulate(self, n: int, delta_t: float, nsave: int=1, c_T: float=1., f: float=1., **kwargs) -> Mapping[str, numpy.ndarray]:
        if not self.initialized:
            self.initialize()
        assert self.valid, 'Model parameterisation is not valid'
        assert f >= 0. and f <= 1., 'Invalid functional response f=%s (it must lie between 0 and 1)' % f
        assert c_T > 0., 'Invalid temperature correction factor c_T=%s (it must be larger than 0)' % c_T

        kwargs.setdefault('f_egg', f)
        y_ini = self._get_initial_state(**kwargs)
        result = numpy.empty((n // nsave + 1, 11))
        self.engine.integrate(n, delta_t, nsave, result, self.E_0_at_f(f), c_T=c_T, f=f, devel_state_ini=self.devel_state_ini, y_ini=y_ini)
        return self._unpack_state(result, self.E_0_at_f(f))

    def simulate2(self, times: Union[float, Sequence], c_T: float=1., f: float=1., delta_t: Optional[float]=None, precision: float=0.001, events: Sequence=(), event_callback=None, **kwargs):
        assert (numpy.diff(times) > 0).all(), 'Time must be monotonically increasing'
        assert (len(events) == 0 and event_callback is None) or (len(events) > 0 and event_callback is not None), 'Arguments events and event_callback must either both be provided, or not at all.'
        if delta_t is None:
            delta_t = self._choose_delta_t(c_T, precision)

        single_time = numpy.ndim(times) == 0
        if single_time:
            times = numpy.reshape(times, -1)

        events = list(events)
        events.append(numpy.inf)

        kwargs.setdefault('f_egg', f)
        y_ini = self._get_initial_state(**kwargs)
        t_start = 0.

        result = numpy.empty((len(times), 11))
        E_0s = numpy.empty((len(times),))
        itime = 0
        overridden_params = set()
        E_0 = self.E_0_at_f(f)
        for event_time in events:
            assert itime < len(times), 'Next save index %i is not valid.' % itime
            assert t_start <= event_time, 'Start time %s should not exceed event time %s' % (t_start, event_time)
            while itime < len(times):
                t_end = min(times[itime], event_time)
                duration = t_end - t_start
                n = int(numpy.ceil(duration / delta_t))
                current_delta_t = delta_t if n == 0 else duration / n
                self.engine.integrate(n, current_delta_t, 0, result[itime:itime+1, :], E_0, c_T=c_T, f=f, devel_state_ini=self.devel_state_ini, y_ini=y_ini)
                #print(event_time, t_start, t_end, self.engine.p_T, n, result[itime, 1])
                y_ini = result[itime, 1:-1]
                result[itime, 0] += t_start
                t_start = t_end
                if times[itime] > event_time:
                    # We have now integrated till the event time and te next save time lies beyond - stop.
                    break
                E_0s[itime] = E_0
                itime += 1
            if itime == len(times):
                break
            c_T, f, E_0, params = event_callback(event_time, self, y_ini, c_T, f, E_0)
            #print('res', c_T, f, E_0, params)
            for p, v in params.items():
                overridden_params.add(p)
                setattr(self.engine, p, v)
        for p in overridden_params:
            setattr(self.engine, p, getattr(self, p))
        if single_time:
            result = result[-1, :]
        return self._unpack_state(result, E_0s)

    def plot_result(self, t: numpy.ndarray, L: numpy.ndarray, S: numpy.ndarray, R: numpy.ndarray, E_H: numpy.ndarray, c_T: float=1., **kwargs):
        from matplotlib import pyplot
        fig = kwargs.get('fig', None)
        if fig is None:
            fig = pyplot.figure()

        time_scale, time_unit = 1., 'd'
        if t[-1] > 3*365:
            time_scale, time_unit = 1. / 365., 'yr'
            t *= time_scale
        ax = fig.add_subplot(411)
        ijuv = E_H.searchsorted(self.E_Hb)
        imet = E_H.searchsorted(self.E_Hj)
        ipub = E_H.searchsorted(self.E_Hp)
        ax.plot(t[:ijuv], L[:ijuv], '-b')
        ax.plot(t[ijuv:imet], L[ijuv:imet], '-', color='purple')
        ax.plot(t[imet:ipub], L[imet:ipub], '-r')
        ax.plot(t[ipub:], L[ipub:], '-k')
        ax.set_xlim(0, t[-1])
        ax.set_title('structural length')
        ax.set_ylabel('length (cm)')
        #ax.plot(t, L, '-b')
        ax.grid()

        ax = fig.add_subplot(412)
        ax.set_title('maturity')
        ax.axhline(self.E_Hb, color='b', linestyle='--', linewidth=1.)
        ax.axvline(time_scale * self.a_b / c_T, color='b', linestyle='--', linewidth=1.)
        if self.E_Hj > self.E_Hb:
            ax.axhline(self.E_Hj, color='purple', linestyle='--', linewidth=1.)
            ax.axvline(time_scale * self.a_j / c_T, color='purple', linestyle='--', linewidth=1.)
        ax.axhline(self.E_Hp, color='r', linestyle='--', linewidth=1.)
        ax.axvline(time_scale * self.a_p / c_T, color='r', linestyle='--', linewidth=1.)
        ax.set_ylabel('maturity (J)')
        ax.plot(t, E_H, '-k')
        ax.set_xlim(0, t[-1])
        ax.grid()

        ax = fig.add_subplot(413)
        ax.set_title('reproduction')
        ax.plot(t, R, '-b')
        ax.set_xlim(0, t[-1])
        ax.set_ylabel('reproduction rate (1/d)')
        ax.grid()

        ax = fig.add_subplot(414)
        ax.set_title('survival')
        ax.plot(t, S, '-b')
        ax.set_xlim(0, t[-1])
        ax.set_ylim(0, 1.1)
        ax.set_xlabel('time (%s)' % time_unit)
        ax.set_ylabel('survival (-)')
        ax.grid()

        return fig

def simulate_ensemble(models: Sequence[Model], selected_outputs: Iterable[str], T: Optional[float]=20., times: Optional[float]=None, before_simulate: Optional[Callable]=None, dtype='f4', progress_reporter: Callable[[float, str], None]=lambda value, status: None, **kwargs) -> Mapping[str, numpy.array]:
        # Define time period for simulation based on entire ensemble
        n = len(models)
        c_Ts = numpy.array([model.get_temperature_correction(T) for model in models])
        if times is None:
            a_99_90 = numpy.percentile(numpy.array([model.a_99 for model in models]) / c_Ts, 90)
            a_p_90 = numpy.percentile(numpy.array([model.a_p for model in models]) / c_Ts, 90)
            t_end = min(max(a_99_90, a_p_90), 365.*200)
            times = numpy.linspace(0, t_end, 1000)

        results = {}
        for key in selected_outputs:
            results[key] = numpy.empty((n, times.size), dtype=dtype)

        model2outputs = {}
        def get_result(model: Model, c_T: float):
            if model not in model2outputs:
                current_kwargs = kwargs.copy()
                current_kwargs['times'] = times
                current_kwargs['c_T'] = c_T
                if before_simulate is not None:
                    before_simulate(model, current_kwargs)
                result = model.simulate2(**current_kwargs)
                outputs = {}
                for key in selected_outputs:
                    outputs[key] = model.evaluate(key, c_T=c_T, locals=result)
                model2outputs[model] = outputs
            return model2outputs[model]

        for i, model in enumerate(models):
            if (i + 1) % 100 == 0:
                progress_reporter(i / n, 'simulating with model %i of %i' % (i + 1, n))
            for key, values in get_result(model, c_Ts[i]).items():
                results[key][i, :] = values
        progress_reporter(1., 'simulations complete')
        return results | {'t': times}