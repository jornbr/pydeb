from __future__ import print_function

import sys
import os
import math
import urllib

import numpy

try:
    import cmodel
except ImportError:
    try:
        import pyximport
        pyximport.install(setup_args={'include_dirs': numpy.get_include()})
        from . import cmodel
    except ImportError as e:
        print('WARNING: unable to load Cython verison of model code. Performance will be reduced. Reason: %s' % e)
        cmodel = None

primary_parameters = 'T_A', 'p_Am', 'F_m', 'kap_X', 'kap_P', 'v', 'kap', 'kap_R', 'p_M', 'p_T', 'k_J', 'E_G', 'E_Hb', 'E_Hx', 'E_Hj', 'E_Hp', 'h_a', 's_G', 't_0'

implied_properties = 'L_b', 'L_p', 'L_i', 'a_b', 'a_p', 'a_99', 'E_0', 'E_m', 'r_B', 'R_i'

long_names = {
    'T_A': 'Arrhenius temperature',
    'p_Am': '{p_Am}, specific assimilation flux',
    'F_m': '{F_m}, maximum specific searching rate',
    'kap_X': 'digestion efficiency (fraction of food to reserve)',
    'kap_P': 'faecation efficiency (fraction of food to faeces)',
    'v': 'energy conductance',
    'kap': 'allocation fraction to soma',
    'kap_R': 'reproduction efficiency',
    'p_M': '[p_M], volume-specific somatic maintenance',
    'p_T': 'p_T}, surface-specific somatic maintenance',
    'k_J': 'maturity maintenance rate coefficient',
    'E_G': '[E_G], specific cost for structure',
    'E_Hb': 'maturity at birth',
    'E_Hj': 'maturity at metamorphosis',
    'E_Hx': 'maturity at weaning/fledgling',
    'E_Hp': 'maturity at puberty',
    'h_a': 'Weibull aging acceleration',
    's_G': 'Gompertz stress coefficient',
    't_0': 'time at start of development',
    'L_b': 'structural length at birth',
    'L_p': 'structural length at puberty',
    'L_i': 'ultimate structural length',
    'a_b': 'age at birth',
    'a_p': 'age at puberty',
    'a_99': 'age when reaching 99% of ultimate structural length',
    'R_i': 'ultimate reproduction rate',
    'r_B': 'von Bertalanffy growth rate',
    'E_m': '[E_m], reserve capacity',
    'E_0': 'initial reserve',
    's_M': 'acceleration at metamorphosis',
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
    'L_p': 'cm',
    'L_i': 'cm',
    'a_b': 'd',
    'a_p': 'd',
    'a_99': 'd',
    'R_i': '1/d',
    'r_B': '1/d',
    'E_m': 'J/cm^3',
    'E_0': 'J',
    's_M': '-',
}

class Model(object):
    def __init__(self, type='abj'):
        self.p_Am = None # {p_Am}, spec assimilation flux (J/d.cm^2)
        self.v = None   # energy conductance (cm/d)
        self.p_M = None # [p_M], vol-spec somatic maint, J/d.cm^3 
        self.p_T = 0. # {p_T}, surf-spec somatic maint, J/d.cm^2
        self.kap = None
        self.E_G = None    # [E_G], spec cost for structure
        self.E_Hb = None   # maturity at birth (J)
        self.E_Hp = None   # maturity at puberty (J)
        self.E_Hj = None   # maturity at metamorphosis (J)
        self.k_J = None #k_J: maturity maint rate coefficient, 1/d
        self.h_a = None #Weibull aging acceleration (1/d^2)
        self.s_G = None #Gompertz stress coefficient
        self.kap_R = None # reproductive efficiency
        self.kap_X = None # digestion efficiency of food to reserve
        self.type = type # std, abj, stf, stx

        # stf
        # foetal development (rather than egg development)
        # e -> \Inf, p 64, r = v/L

        # stx:
        # foetal development (rather than egg development) that first starts with a preparation stage and then sparks off at a time that is an extra parameter
        # a baby stage (for mammals) just after birth, ended by weaning, where juvenile switches from feeding on milk to solid food at maturity level EHx. Weaning is between birth and puberty.
        #t_0: time at start development - stx model
        #E_Hx: maturity at weaning (J) - stx model

        # ssj:
        # a non-feeding stage between events s and j during the juvenile stage that is initiated at a particular maturity level and lasts a particular time. Substantial metabolically controlled shrinking occur during this period, more than can be explained by starvation.
        #E_Hs: maturity at S2/S3 transition - ssj model
        #t_sj: period of metamorphosis - model ssj

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
        #E_Rj: reproduction buffer density at emergence (J/cm^3) - hep model

        # hex
        # The DEB model for holometabolic insects (and some other hexapods). It characterics are
        # - morphological life stages: egg, larva, (pupa), imago; functional stages: embryo, adult, (pupa), imago
        # - the embryo still behaves like model std
        # - the larval stage accelerates (V1-morph) and behaves as adult, i.e. no maturation, allocation to reproduction
        # - pupation occurs when reproduction buffer density hits a threshold
        # - pupa behaves like an isomorphic embryo of model std, but larval structure rapidly transforms to pupal reserve just after start of pupation
        # - the reproduction buffer remains unchanged during the pupal stage
        # - the imago does not grow or allocate to reproduction. It mobilises reserve to match constant (somatic plus maturity) maintenance
        #E_He: maturity at emergence (J) - hex model
        #kap_V: conversion efficient E -> V -> E - hex model

        #kap_P: digestion efficiency of food to faeces
        #F_m {F_m}, max spec searching rate (l/d.cm^2)
        #s_j: reprod buffer/structure at pupation as fraction of max (-) - hex model

        # pars specific for this entry
        #del_M = L/Lw: shape coefficient (-)

        # derived parameters
        self.E_0 = None
        self.L_b = None
        self.a_b = None
        self.a_99 = None
        # L_m = kappa*v*E_m/p_M = kappa*p_Am/p_M [L_m is maximum length in absence of surface=-area-specific maintenance!]
        # z = L_m/L_m_ref with L_m_ref = 1 cm - equal to L_m

        self.initialized = False
        self.valid = False
        self.cmodel = None

    def initialize(self, E_0_ini=0., verbose=False):
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
        E_m = self.p_Am/self.v # defined at f=1
        g = E_G/kap/E_m
        k_M = self.p_M/E_G
        L_m = kap*self.p_Am/self.p_M
        L_T = self.p_T/self.p_M
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

        def get_E_0(log10_E_0_left, log10_E_0_right, delta_t):
            import scipy.optimize 
            def error_in_E(log10_E_0):
                if not isinstance(log10_E_0, float):
                    p = float(numpy.asscalar(log10_E_0))
                a_b, E_b, L_b = get_birth_state(10.**log10_E_0, delta_t)
                if a_b == -1:
                    return numpy.inf
                ssq = (E_b/L_b**3 - E_m)**2
                return ssq
            log10_E_0 = scipy.optimize.minimize_scalar(error_in_E, bracket=(log10_E_0_left, log10_E_0_right)).x
            #log10_E_0 = scipy.optimize.fmin(error_in_E, 0.5 * (log10_E_0_left + log10_E_0_right), disp=False) #, initial_simplex=(log10_E_0_left, log10_E_0_right)
            a_b, _, L_b = get_birth_state(10.**log10_E_0, delta_t)
            return log10_E_0, a_b, L_b

        def get_birth_state(E_0, delta_t=1.):
            t, E, L, E_H = 0., float(E_0), 0., 0.
            done = False
            while not done:
                L2 = L*L
                L3 = L*L2
                denom = E + E_G_per_kap*L3
                p_C = E*(v_E_G_plus_P_T_per_kap*L2 + p_M_per_kap*L3)/denom
                dL = (E*v-(p_M_per_kap*L+p_T_per_kap)*L3)/3/denom
                dE = -p_C
                dE_H = one_minus_kappa*p_C - k_J*E_H
                if E_H + delta_t * dE_H > E_Hb:
                    delta_t = (E_Hb - E_H)/dE_H
                    done = True
                E += delta_t * dE
                L += delta_t * dL
                E_H += delta_t * dE_H
                t += delta_t
                if E < 0 or dL < 0:
                    return -1, -1, -1
            return t, E, L

        def find_maturity(L_ini, E_H_ini, E_H_target, delta_t=1., s_M=1., t_max=numpy.inf, t_ini=0.):
            assert E_H_target >= E_H_ini
            exp = math.exp
            r_B = self.r_B

            t = 0.
            E_H = E_H_ini
            L_i = (L_m - L_T)*s_M # f=1
            done = False
            while not done:
                L = (L_i-L_ini)*(1. - exp(-r_B*t)) + L_ini # p 52
                p_C = L*L*E_m*((v*E_G_per_kap + p_T_per_kap)*s_M + p_M_per_kap*L)/(E_m + E_G_per_kap)
                dE_H = (1. - kap)*p_C - k_J*E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H)/dE_H
                    done = True
                E_H += dE_H*delta_t
                t += delta_t
                if t > t_max:
                    return None, None
            L = (L_i - L_ini)*(1. - exp(-r_B*t)) + L_ini # p 52
            return t_ini + t, L

        def find_maturity_v1(L_ini, E_H_ini, E_H_target, delta_t=1., t_max=numpy.inf, t_ini=0.):
            assert E_H_target >= E_H_ini
            exp = math.exp
            V_b = L_ini**3

            # dL = (E*v*L/L_b-p_S_per_kappa*L2*L2)/3/(E+E_G_per_kappa*L3)
            #    = (E_m*v/L_b-p_S_per_kappa)/3/(E_m+E_G_per_kappa) * L
            r = (kap*v/L_ini*E_m - self.p_M - self.p_T)/(E_G + kap*E_m) # specific growth rate of structural VOLUME
            prefactor = (1. - kap)*V_b*E_m*(v*E_G/L_ini + self.p_M + self.p_T/L_ini)/(E_G + kap*E_m)

            if True:
                t = 0.
                E_H = E_H_ini
                done = False
                while not done:
                    dE_H = prefactor*exp(r*t) - k_J*E_H
                    if E_H + delta_t * dE_H > E_H_target:
                        delta_t = (E_H_target - E_H)/dE_H
                        done = True
                    E_H += dE_H*delta_t
                    t += delta_t
                    if t > t_max:
                        return None, None
            else:
                # analytical solution E_H(t):
                C = E_H_ini/prefactor - 1./(k_J+r)
                t = scipy.optimize.minimize_scalar(lambda t: (E_H_target-prefactor*(exp(r*t)/(k_J+r) + C*exp(-k_J*t)))**2, (t_ini, 10*t_ini)).x

            L = L_ini*exp(r/3*t)
            return t_ini + t, L

        if cmodel is not None:
            self.cmodel = cmodel.Model()
            for parameter in primary_parameters:
                if hasattr(self.cmodel, parameter):
                    setattr(self.cmodel, parameter, getattr(self, parameter))
            get_birth_state = self.cmodel.get_birth_state
            find_maturity = self.cmodel.find_maturity
            find_maturity_v1 = self.cmodel.find_maturity_v1
            find_maturity_foetus = self.cmodel.find_maturity_foetus
            get_E_0 = self.cmodel.get_E_0

        if self.type in ('stf', 'stx'):
            # foetal development
            for delta_t in (1., 0.1, 0.01, 0.001):
                birth_state = find_maturity_foetus(self.E_Hb, delta_t)
                if birth_state is None:
                    if verbose:
                        print('Unable to determine age/length at birth.')
                    return
                if delta_t < birth_state[0] * 0.01:
                    break
            self.a_b, self.L_b, self.E_0 = birth_state
        else:
            # egg development
            # At least E_Hb / (1 - kap) must have been spent during embryo development to reach E_Hb.
            # In fact, energy expenditure MUST be more because of maturity maintenance and because there
            # needs to be reserve left over at time of hatching.
            E_0_ini = max(E_0_ini, 2. * E_Hb / (1 - kap))
            for i in range(10):
                if get_birth_state(E_0_ini)[0] >= 0:
                    break
                E_0_ini *= 10
            else:
                if verbose:
                    print('Cannot find valid initial estimate for E_0 (tried up to %s)' % E_0_ini)
                return

            if verbose:
                print('Determining cost of an egg and state at birth...')
            log10_E_0 = numpy.log10(E_0_ini)
            log10_E_0_left, log10_E_0_right = log10_E_0, log10_E_0 + 1
            for delta_t in (1., 0.1, 0.01, 0.001):
                log10_E_0_new, a_b, L_b = get_E_0(log10_E_0_left, log10_E_0_right, delta_t)
                if a_b <= 0:
                    if verbose:
                        print('Unable to determine cost of an egg (E_0).')
                    return
                step = min(abs(log10_E_0_new - log10_E_0)/10, 1.)
                log10_E_0, log10_E_0_left, log10_E_0_right = log10_E_0_new, log10_E_0_new - step, log10_E_0_new + step

                # Stop if we have the time of birth at 1 % accuracy
                if delta_t < a_b * 0.01:
                    break

            self.E_0, self.a_b, self.L_b = 10.**log10_E_0, a_b, L_b

        self.L_m = L_m
        self.L_T = L_T
        self.E_m = E_m

        self.r_B = self.p_M/3/(E_m*kap + E_G) # checked against p52, note f=1
        L_i_min = self.L_m - self.L_T # not counting acceleration!
        if L_i_min < self.L_b:
            # shrinking directly after birth
            if verbose:
                print('Shrinking directly after birth (L_i_min < L_b).')
            return
        a_99_max = self.a_b - numpy.log(1 - (0.99*L_i_min - self.L_b)/(L_i_min - self.L_b))/self.r_B
        if self.cmodel is not None:
            self.cmodel.E_0 = self.E_0
            self.cmodel.L_b = self.L_b
            self.cmodel.r_B = self.r_B
            self.cmodel.L_m = self.L_m
            self.cmodel.L_T = self.L_T

        # metamorphosis
        if self.E_Hj > self.E_Hb:
            if verbose:
                print('Determining age and length at metamorphosis...')
            self.a_j, self.L_j = find_maturity_v1(self.L_b, self.E_Hb, self.E_Hj, delta_t=max(0.01, self.a_b/100), t_max=min(100*a_99_max, 365*200.), t_ini=self.a_b)
            if self.a_j is None:
                if verbose:
                    print('Cannot determine age and length at metamorphosis.')
                return
        else:
            self.a_j, self.L_j = self.a_b, self.L_b
        self.s_M = self.L_j / self.L_b
        if self.cmodel is not None:
            self.cmodel.s_M = self.s_M
        self.L_i = (self.L_m - self.L_T)*self.s_M
        self.a_99 = self.a_j - numpy.log(1 - (0.99*self.L_i - self.L_j)/(self.L_i - self.L_j))/self.r_B
        p_C_i = self.L_i*self.L_i*E_m*((v*E_G_per_kap + p_T_per_kap)*self.s_M + p_M_per_kap*self.L_i)/(E_m + E_G_per_kap)
        if self.k_J * self.E_Hp > (1 - kap) * p_C_i:
            if verbose:
                print('Cannot reach puberty.')
            return
        self.R_i = ((1-kap)*p_C_i - self.k_J*self.E_Hp)*self.kap_R/self.E_0
        max_E_H = (1 - kap) * p_C_i / self.k_J

        if verbose:
            print('Determining age and length at puberty...')
        if self.E_Hp >= self.E_Hj:
            # puberty after metamorphosis
            self.a_p, self.L_p = find_maturity(self.L_j, self.E_Hj, self.E_Hp, delta_t=max(0.01, self.a_b/100), s_M=self.s_M, t_max=min(100*a_99_max, 365*200.), t_ini=self.a_j)
        else:
            # puberty before metamorphosis
            self.a_p, self.L_p = find_maturity_v1(self.L_b, self.E_Hb, self.E_Hp, delta_t=max(0.01, self.a_b/100), t_max=min(100*a_99_max, 365*200.), t_ini=self.a_b)
        if self.a_p is None:
            if verbose:
                print('Cannot determine age and length at puberty.')
            return
        self.valid = True

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
        print('E_0 [cost of an egg]: %s' % self.E_0)
        print('r_B [von Bertalanffy growth rate]: %s' % (c_T*self.r_B))
        print('a_b [age at birth]: %s' % (self.a_b/c_T))
        print('a_j [age at metamorphosis]: %s' % (self.a_j/c_T))
        print('a_p [age at puberty]: %s' % (self.a_p/c_T))
        print('a_99 [age at L = 0.99 L_m]: %s' % (self.a_99/c_T))
        print('[E_m] [reserve capacity]: %s' % self.E_m)
        print('L_b [structural length at birth]: %s' % self.L_b)
        print('L_j [structural length at metamorphosis]: %s' % self.L_j)
        print('L_p [structural length at puberty]: %s' % self.L_p)
        print('L_i [ultimate structural length]: %s' % self.L_i)
        print('s_M [acceleration factor at f=1]: %s' % self.s_M)
        print('R_i [ultimate reproduction rate]: %s' % (self.R_i*c_T))

    def simulate(self, n, delta_t, nsave=1, c_T=1., f=1.):
        if not self.initialized:
            self.initialize()
        if not self.valid:
            return
        t = numpy.linspace(0., n*delta_t, int(n/nsave)+1)
        assert f >= 0. and f <= 1.
        assert c_T > 0.
        if self.cmodel is not None:
            result = self.cmodel.integrate(n, delta_t, nsave=nsave, c_T=c_T, f=f, devel_state_ini=self.devel_state_ini)
            return {'t': t, 'E': result[:, 0], 'L': result[:, 1], 'E_H': result[:, 2], 'E_R': result[:, 3], 'S': result[:, 6], 'cumR': result[:, 7], 'a': result[:, 8], 'R': result[:, 9]}

        kap = self.kap
        v = self.v*c_T
        k_J = self.k_J*c_T
        p_Am = self.p_Am*c_T
        p_M = self.p_M*c_T
        p_T = self.p_T*c_T
        E_G = self.E_G
        E_Hb = self.E_Hb
        E_Hp = self.E_Hp
        s_G = self.s_G
        h_a = self.h_a*c_T*c_T
        E_0 = self.E_0
        kap_R = self.kap_R
        s_M = self.s_M
        L_b = self.L_b

        E_m = p_Am/v
        L_m = kap*p_Am/p_M
        L_m3 = L_m**3
        E_G_per_kap = E_G/kap
        p_M_per_kap = p_M/kap
        p_T_per_kap = p_T/kap
        v_E_G_plus_P_T_per_kap = (v*E_G + p_T)/kap
        one_minus_kap = 1-kap

        def dy(y, t0):
            E, L, E_H, E_R, Q, H, S, cumR, cumt = map(float, y)
            L2 = L*L
            L3 = L*L2
            s = max(1., min(s_M, L/L_b))
            #s = 1.

            # Energy fluxes in J/d
            p_C = E*(v_E_G_plus_P_T_per_kap*s*L2 + p_M_per_kap*L3)/(E + E_G_per_kap*L3)
            p_A = 0. if E_H < E_Hb else p_Am*L2*f*s
            p_R = one_minus_kap*p_C - k_J*E_H # J/d

            # Change in reserve (J), structural length (cm), maturity (J), reproduction buffer (J)
            dE = p_A - p_C
            dL = (E*v*s-(p_M_per_kap*L+p_T_per_kap*s)*L3)/3/(E+E_G_per_kap*L3)
            if E_H < E_Hp:
                dE_H = p_R
                dE_R = 0.
            else:
                dE_H = 0
                dE_R = kap_R * p_R

            # Damage-inducing compounds, damage, survival (0-1) - p 216
            dQ = (Q/L_m3*s_G + h_a)*max(0., p_C)/E_m
            dH = Q
            dS = 0. if L3 <= 0. or S < 1e-16 else -min(1./(delta_t+1e-8), H/L3)*S

            # Cumulative reproduction (#) and life span (d)
            dcumR = S*dE_R/E_0
            dcumt = S

            return numpy.array((dE, dL, dE_H, dE_R, dQ, dH, dS, dcumR, dcumt), dtype=float), dE_R/E_0

        y0 = numpy.array((E_0, 0., 0., 0., 0., 0., 1., 0., 0.))
        result = numpy.empty((t.size, y0.size))
        allR = numpy.empty((t.size,))
        y = y0
        for it in range(n+1):
            if it % nsave == 0:
                result[it/nsave, :] = y
            derivative, R = dy(y, it*delta_t)
            if not numpy.isfinite(derivative).all():
                print('Temporal derivatives contain NaN: %s' % derivative)
                sys.exit(1)
            y += delta_t*derivative
            if it % nsave == 0:
                allR[it/nsave] = R

        return {'t': t, 'E': result[:, 0], 'L': result[:, 1], 'E_H': result[:, 2], 'E_R': result[:, 3], 'S': result[:, 6], 'cumR': result[:, 7], 'a': result[:, 8], 'R': allR}

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

def plot(ax, t, values, perc_wide = 0.025, perc_narrow = 0.25, ylabel=None, title=None, color='k'):
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
                #ax.set_yscale('log')
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