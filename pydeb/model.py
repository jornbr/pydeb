from __future__ import print_function

import sys
import os
import math
import urllib

import numpy

import scipy.optimize

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
    'E_G': 'specific cost for structure',
    'E_Hb': 'maturity at birth',
    'E_Hj': 'maturity at metamorphosis',
    'E_Hx': 'maturity at weaning/fledgling',
    'E_Hp': 'maturity at puberty',
    'h_a': 'Weibull aging acceleration',
    's_G': 'Gompertz stress coefficient',
    't_0': 'time at start of development',
    'L_b': 'structural length at birth',
    'L_j': 'structural length at metamorphosis',
    'L_p': 'structural length at puberty',
    'L_i': 'ultimate structural length',
    'a_b': 'age at birth',
    'a_j': 'age at metamorphosis',
    'a_p': 'age at puberty',
    'a_99': 'age when reaching 99% of ultimate structural length',
    'R_i': 'ultimate reproduction rate',
    'r_B': 'von Bertalanffy growth rate',
    'E_m': 'reserve capacity',
    'E_0': 'initial reserve',
    's_M': 'acceleration at metamorphosis',
    'del_M': 'shape coefficient (structural : physical length)',
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
    'E_G': 'J/cm^3',
    'E_Hb': 'J',
    'E_Hj': 'J',
    'E_Hx': 'J',
    'E_Hp': 'J',
    'h_a': '1/d^2',
    's_G': '-',
    't_0': 'd',
    'L_b': 'cm',
    'L_j': 'cm',
    'L_p': 'cm',
    'L_i': 'cm',
    'a_b': 'd',
    'a_j': 'd',
    'a_p': 'd',
    'a_99': 'd',
    'R_i': '1/d',
    'r_B': '1/d',
    'E_m': 'J/cm^3',
    'E_0': 'J',
    's_M': '-',
    'del_M': '-',
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
    'R_i': 1,
    'r_B': 1,
    'E_m': 0,
    'E_0': 0,
    's_M': 0,
    'del_M': 0,
    'mu_E': 0,
    'w_E': 0,
    'd_E': 0
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


def symbol2html(symbol):
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


def symbol2mathtext(symbol):
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


class ModelDict(object):
    """Evaluates a expression combining model parameters and implied properties (to be temperature corrected)
    and optionally an additonals "locals" dictionary with additional objects - e.g., model results. The latter
    takes priority if provided. Its contained varables are assumed to *already have been temperature-corrected*."""

    def __init__(self, model, c_T=1., locals={}):
        self.model = model
        self.c_T = c_T
        self.locals = locals

    def __getitem__(self, key):
        if key in self.locals:
            return self.locals[key]
        if key in compound_variables:
            return eval(compound_variables[key], {}, self)
        if not hasattr(self.model, key):
            raise KeyError()
        return getattr(self.model, key) * self.c_T**temperature_correction[key]

    def __contains__(self, key):
        return key in self.locals or hasattr(self.model, key) or key in compound_variables


class Model(object):
    def __init__(self, type='abj'):
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
        # L_m = kappa*v*E_m/p_M = kappa*p_Am/p_M [L_m is maximum length in absence of surface=-area-specific maintenance!]
        # z = L_m/L_m_ref with L_m_ref = 1 cm - equal to L_m

        self.initialized = False
        self.valid = False
        self.engine = None

        self.mu_E = 5.5e5  # chemical potential of reserve (J/C-mol)
        self.w_E = 23.9   # dry weight of reserve (g/C_mol)
        self.d_E = 0.21   # specific density of reserve (g DM/cm3)
        # self.WM_per_E = self.w_E / self.mu_E / self.d_E # cm3/J

    def copy(self, **parameters):
        clone = Model(type=parameters.get('type', self.type))
        for p in primary_parameters + entry_parameters:
            setattr(clone, p, parameters.get(p, getattr(self, p)))
        return clone

    def initialize(self, E_0_ini=0., verbose=False, precision=0.001):
        assert self.p_T >= 0.
        assert self.p_M >= 0.
        assert self.p_Am >= 0.
        assert self.v >= 0.
        assert self.E_G >= 0.
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
        k_M = self.p_M / E_G
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
        v_E_G_plus_P_T_per_kap = (v*E_G + p_T)/kap
        one_minus_kappa = 1-kap

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
        L_i = L_m - L_T
        p_C_i = L_i * L_i * E_m * ((v * E_G_per_kap + p_T_per_kap) + p_M_per_kap * L_i) / (E_m + E_G_per_kap)
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
            # At least E_Hb / (1 - kap) must have been spent during embryo development to reach E_Hb.
            # In fact, energy expenditure MUST be more because of maturity maintenance and because there
            # needs to be reserve left over at time of hatching.
            E_0_min = E_0_max = E_Hb / (1 - kap)
            assert get_birth_state(E_0_min, delta_t)[2] <= E_Hb
            for _ in range(10):
                E_0_max *= 10
                if get_birth_state(E_0_max, delta_t)[2] >= E_Hb:
                    break
            else:
                if verbose:
                    print('Cannot find valid initial estimate for E_0 (tried up to %s)' % E_0_max)
                return

            #def root(E_0, delta_t):
            #    _, _, E_H = get_birth_state(E_0, delta_t)
            #    return E_H - E_Hb
            if verbose:
                print('Determining cost of an egg and state at birth...')
            while 1:
                #E_0 = scipy.optimize.brentq(root, E_0_min, E_0_max, rtol=precision, args=(delta_t,))
                #a_b, L_b, _ = get_birth_state(E_0, delta_t)
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
        delta_t = max(precision * 10, self.a_b * precision * 10) * 2

        self.r_B = self.p_M/3/(E_m*kap + E_G)  # checked against p52, note f=1
        L_i_min = self.L_m - self.L_T  # not counting acceleration!
        if L_i_min < self.L_b:
            # shrinking directly after birth
            if verbose:
                print('Shrinking directly after birth (L_i_min < L_b).')
            return
        a_99_max = self.a_b - numpy.log(1 - (0.99*L_i_min - self.L_b)/(L_i_min - self.L_b))/self.r_B

        self.engine.E_0 = self.E_0
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
        self.maturity_states = {
            self.E_Hb: (self.a_b, self.L_b),
            self.E_Hj: (self.a_j, self.L_j),
            self.E_Hp: (self.a_p, self.L_p)
        }

    def ageAtMaturity(self, E_H, precision=0.001, c_T=1.):
        if E_H not in self.maturity_states:
            delta_t = max(precision * 10, self.a_b * precision * 10) * 2
            tmax = min(100 * self.a_99, 365 * 200.)
            if E_H < self.E_Hb:
                # before birth
                if self.type in ('stf', 'stx'):
                    a, L, _ = self.engine.find_maturity_foetus(E_H, delta_t=delta_t, t_max=tmax)
                else:
                    a, L = self.engine.find_maturity_egg(E_H, delta_t=delta_t, t_max=tmax)
            elif E_H < self.E_Hj:
                # before metamophosis (i.e., during acceleration in V1 morph mode)
                a, L = self.engine.find_maturity_v1(self.L_b, self.E_Hb, E_H, delta_t=delta_t, t_max=tmax, t_ini=self.a_b)
            else:
                # after metamorphosis
                a, L = self.engine.find_maturity(self.L_j, self.E_Hj, E_H, delta_t=delta_t, s_M=self.s_M, t_max=tmax, t_ini=self.a_j)
            self.maturity_states[E_H] = (a, L)
        a, L = self.maturity_states[E_H]
        return a / c_T

    def evaluate(self, expression, c_T=1., locals={}):
        return eval(expression, {}, ModelDict(self, c_T, locals=locals))

    def getTemperatureCorrection(self, T):
        # T is temperature in Celsius
        assert T < 200., 'Temperature must be given in degrees Celsius'
        return numpy.exp(self.T_A/293.15 - self.T_A/(273.15 + T))

    def writeFABMConfiguration(self, path, name='deb', model='deb/population'):
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

            f.write('      L_b: %s\n' % self.L_b)
            f.write('      L_j: %s\n' % self.L_j)
            f.write('      E_0: %s\n' % self.E_0)

    def report(self, c_T=1.):
        if not self.initialized:
            self.initialize()
        if not self.valid:
            print('This parameter set is not valid.')
            return
        d = ModelDict(self, c_T)
        shown = ('E_0', 'r_B', 'a_b', 'a_j', 'a_p', 'a_99', 'E_m', 'L_b', 'L_j', 'L_p', 'L_i', 's_M', 'R_i')
        for name in shown:
            print('%s [%s]: %.4g %s' % (name, long_names[name], eval(name, {}, d), units[name]))

    def simulate(self, n, delta_t, nsave=1, c_T=1., f=1.):
        if not self.initialized:
            self.initialize()
        if not self.valid:
            return
        t = numpy.linspace(0., n * delta_t, int(n / nsave) + 1)
        assert f >= 0. and f <= 1.
        assert c_T > 0.

        result = numpy.empty((int(n / nsave) + 1, 10))
        self.engine.integrate(n, delta_t, nsave, result, c_T=c_T, f=f, devel_state_ini=self.devel_state_ini)
        return {'t': t, 'E': result[:, 0], 'L': result[:, 1], 'E_H': result[:, 2], 'E_R': result[:, 3], 'S': result[:, 6], 'cumR': result[:, 7], 'a': result[:, 8], 'R': result[:, 9]}

    def plotResult(self, t, L, S, R, E_H, **kwargs):
        from matplotlib import pyplot
        fig = kwargs.get('fig', None)
        if fig is None:
            fig = pyplot.figure()
        ax = fig.add_subplot(411)
        ijuv = E_H.searchsorted(self.E_Hb)
        ipub = E_H.searchsorted(self.E_Hp)
        ax.plot(t[:ijuv], L[:ijuv], '-g')
        ax.plot(t[ijuv:ipub], L[ijuv:ipub], '-b')
        ax.plot(t[ipub:], L[ipub:], '-r')
        ax.set_title('structural length')
        #ax.plot(t, L, '-b')
        ax.grid()

        ax = fig.add_subplot(412)
        ax.set_title('maturity')
        ax.axhline(self.E_Hb, color='g')
        ax.axhline(self.E_Hp, color='b')
        ax.axvline(self.a_b, color='g')
        ax.axvline(self.a_p, color='b')
        ax.plot(t, E_H, '-k')
        ax.grid()

        ax = fig.add_subplot(413)
        ax.set_title('reproduction rate')
        ax.plot(t, R, '-b')
        ax.grid()

        ax = fig.add_subplot(414)
        ax.set_title('survival')
        ax.plot(t, S, '-b')
        ax.grid()

        return fig


def plot(ax, t, values, perc_wide=0.025, perc_narrow=0.25, ylabel=None, title=None, color='k'):
    values.sort()
    n = values.shape[1]
    perc_50 = values[:, int(0.5*n)]
    if n > 1:
        if perc_wide is not None:
            perc_wide_l = values[:, int(perc_wide*n)]
            perc_wide_u = values[:, int((1-perc_wide)*n)]
            ax.fill_between(t, perc_wide_l, perc_wide_u, facecolor=color, alpha=0.25)
        perc_narrow_l = values[:, int(perc_narrow*n)]
        perc_narrow_u = values[:, int((1-perc_narrow)*n)]
        ax.fill_between(t, perc_narrow_l, perc_narrow_u, facecolor=color, alpha=0.4)
        color = 'k'
    ax.plot(t, perc_50, color)
    ax.grid(True)
    ax.set_xlabel('time (d)')
    ax.set_xlim(t[0], t[-1])
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)


class HTMLGenerator(object):
    def __init__(self, *models):
        self.models = models

    def initialize(self):
        E_0s = []
        E_0_ini = None
        self.valid_models = []
        for model in self.models:
            model.initialize(E_0_ini)
            if model.valid:
                E_0s.append(model.E_0)
                E_0_ini = numpy.mean(E_0s)
                self.valid_models.append(model)

    def generate(self, workdir, color='k', label='', t_end=None, figsize=(6, 4), dpi=96):
        import matplotlib
        from matplotlib import pyplot
        import matplotlib.ticker
        relworkdir = os.path.relpath(workdir)

        matplotlib.rcParams['font.sans-serif'] = 'Verdana'
        matplotlib.rcParams['font.size'] = 9

        # Collect results for all model instances
        n = len(self.valid_models)
        if t_end is None:
            t_end = min(numpy.sort([model.a_99 for model in self.valid_models])[int(0.9*n)], 365.*200)
        a_b_10 = numpy.sort([model.a_b for model in self.valid_models])[int(0.1*n)]
        delta_t = max(0.04, a_b_10/4)
        nt = int(t_end/delta_t)
        nsave = max(1, int(math.floor(nt/1000)))
        for i, model in enumerate(self.valid_models):
            result = model.simulate(nt, delta_t, nsave)
            t = result['t']
            if i == 0:
                Ls = numpy.empty((len(t), n))
                Ss = numpy.empty((len(t), n))
                Rs = numpy.empty((len(t), n))
            Ls[:, i] = result['L']
            Ss[:, i] = result['S']
            Rs[:, i] = result['R']

        strings = []
        #strings.append('delta_t=%s, nt=%s, nsave=%s' % (delta_t, nt, nsave))

        class Fmt(matplotlib.ticker.LogFormatterMathtext):
            def __call__(self, x, pos=None):
                return matplotlib.ticker.LogFormatterMathtext.__call__(self, 10.**x, pos)

        params = 'E_0', 'a_b', 'a_p', 'L_b', 'L_p', 'L_i', 'R_i'

        if n > 1:
            fig = pyplot.figure(figsize=figsize)
            for i, p in enumerate(params):
                ax = fig.add_subplot(1, len(params), i+1)
                values = numpy.array([getattr(model, p) for model in self.valid_models])
                #ax.boxplot(values, labels=(p,), whis=(10, 90))
                ax.violinplot(numpy.log10(values), showmedians=True)
                ax.set_xticks(())
                ax.set_title(p)
                ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator())
                ax.yaxis.set_major_formatter(Fmt())
                # ax.set_yscale('log')
            fig.tight_layout()
            fig.savefig(os.path.join(workdir, 'boxplots.png'), dpi=dpi)
            strings.append('<img alt="violin plots of derived life history parameters" src="%s"/><br>' % urllib.pathname2url('%s/boxplots.png' % relworkdir))
        else:
            strings.append('<table>')
            strings.append('<thead><tr><th>parameter</th><th>value</th></tr></thead>')
            strings.append('<tbody>')
            for i, p in enumerate(params):
                strings.append('  <tr><td>%s</td><td>%.3g</td></tr>' % (p, getattr(self.valid_models[0], p)))
            strings.append('</tbody>')
            strings.append('</table>')

        fig = pyplot.figure(figsize=figsize)
        ax = fig.gca()

        ax.cla()
        plot(ax, t, Ls, perc_wide=0.1, title='growth', ylabel='structural length (cm)', color='b')
        ax.set_ylim(0, None)
        fig.tight_layout()
        fig.savefig(os.path.join(workdir, 'tL.png'), dpi=dpi)
        strings.append('<img alt="time series of structural length" src="%s"/><br>' % urllib.pathname2url('%s/tL.png' % relworkdir))

        ax.cla()
        plot(ax, t, Rs, perc_wide=0.1, title='reproduction', ylabel='reproduction rate (#/d)', color='g')
        ax.set_ylim(0, None)
        fig.tight_layout()
        fig.savefig(os.path.join(workdir, 'tR.png'), dpi=dpi)
        strings.append('<img alt="time series of reproduction rate" src="%s"/><br>' % urllib.pathname2url('%s/tR.png' % relworkdir))

        ax.cla()
        plot(ax, t, Ss, perc_wide=0.1, title='survival', ylabel='survival (-)', color='r')
        ax.set_ylim(0., 1.)
        fig.tight_layout()
        fig.savefig(os.path.join(workdir, 'tS.png'), dpi=dpi)
        strings.append('<img alt="time series of survival" src="%s"/><br>' % urllib.pathname2url('%s/tS.png' % relworkdir))

        return '\n'.join(strings)


if __name__ == '__main__':
    model = Model()
    import argparse
    import io
    from matplotlib import pyplot
    import timeit
    parser = argparse.ArgumentParser()
    parser.add_argument('--traits', default='traits.txt')
    parser.add_argument('--c_T', type=float, default=1.)
    parser.add_argument('species')
    args = parser.parse_args()
    with io.open(args.traits, 'rU', encoding='utf-8') as f:
        labels = f.readline().rstrip('\n').split('\t')
        for l in f:
            items = l.rstrip('\n').split('\t')
            if items[0].lower() == args.species.lower():
                par2value = {}
                for name, item in zip(labels[2:], items[2:]):
                    if item != '':
                        par2value[name.split(' (')[0]] = float(item)
                break
    print('Parameters for %s:' % args.species)
    for name, value in par2value.items():
        print('%s: %s' % (name, value))
        setattr(model, name, value)
    model.report(c_T=args.c_T)
    delta_t = 0.1
    start = timeit.default_timer()
    result_c = model.simulate(int(model.a_99/delta_t), delta_t, c_T=args.c_T)
    print('C duration: %s s' % (timeit.default_timer() - start))
    model.cmodel = None
    start = timeit.default_timer()
    result_py = model.simulate(int(model.a_99/delta_t), delta_t, c_T=args.c_T)
    print('Python duration: %s s' % (timeit.default_timer() - start))
    fig = model.plotResult(**result_c)
    model.plotResult(fig=fig, **result_py)
    print('Relative differences (C vs Python):')
    for variable in ('L', 'E', 'E_H', 'R', 'S'):
        L_diff = result_c[variable] - result_py[variable]
        vmin, vmax = min(result_c[variable].min(), result_py[variable].min()), min(result_c[variable].max(), result_py[variable].max())
        print('  %s: %s - %s (range = %s)' % (variable, L_diff.min()/(vmax-vmin), L_diff.max()/(vmax-vmin), vmax-vmin))
    pyplot.show()