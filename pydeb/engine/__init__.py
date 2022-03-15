import sys
import math
from typing import Mapping

import numpy

try:
    from . import cmodel
except ImportError as e1:
    try:
        import pyximport
        pyximport.install(setup_args={'include_dirs': numpy.get_include()}, language_level='3')
        from . import cmodel
    except ImportError as e2:
        print('WARNING: unable to load Cython verison of model code. Performance will be reduced. Reason:\n%s\n' % (e1, e2))
        cmodel = None

class PyEngine(object):
    def __init__(self):
        self.p_Am = -1.   # {p_Am}, spec assimilation flux (J/d.cm^2)
        self.v = -1.      # energy conductance (cm/d)
        self.p_M = -1.    # [p_M], vol-spec somatic maint, J/d.cm^3
        self.p_T = 0.     # {p_T}, surf-spec somatic maint, J/d.cm^2
        self.kap = -1.
        self.E_G = -1.    # [E_G], spec cost for structure
        self.E_Hb = -1.   # maturity at birth (J)
        self.E_Hp = -1.   # maturity at puberty (J)
        self.E_Hj = -1.   # maturity at metamorphosis (J)
        self.k_J = -1.    # k_J: maturity maint rate coefficient, 1/d
        self.h_a = 0.     # Weibull aging acceleration (1/d^2)
        self.s_G = 0.     # Gompertz stress coefficient
        self.kap_R = -1.  # reproductive efficiency
        self.kap_X = -1.  # digestion efficiency of food to reserve
        self.T_A = -1.    # Arrhenius temperature
        self.type = type  # std, abj, stf, stx

        self.E_0 = -1.
        self.L_b = -1.
        self.s_M = 1.

    def get_birth_state(self, E_0: float, delta_t: float=1.):
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

    def find_maturity(self, L_ini: float, E_H_ini: float, E_H_target: float, delta_t: float=1., s_M: float=1., t_max: float=numpy.inf, t_ini: float=0.):
        assert E_H_target >= E_H_ini
        exp = math.exp
        r_B = self.r_B

        t = 0.
        E_H = E_H_ini
        L_i = (L_m - L_T) * s_M  # f=1
        done = False
        while not done:
            L = (L_i-L_ini)*(1. - exp(-r_B*t)) + L_ini  # p 52
            p_C = L*L*E_m*((v*E_G_per_kap + p_T_per_kap)*s_M + p_M_per_kap*L)/(E_m + E_G_per_kap)
            dE_H = (1. - kap)*p_C - k_J*E_H
            if E_H + delta_t * dE_H > E_H_target:
                delta_t = (E_H_target - E_H)/dE_H
                done = True
            E_H += dE_H*delta_t
            t += delta_t
            if t > t_max:
                return None, None
        L = (L_i - L_ini)*(1. - exp(-r_B*t)) + L_ini  # p 52
        return t_ini + t, L

    def find_maturity_v1(self, L_ini: float, E_H_ini: float, E_H_target: float, delta_t: float=1., t_max: float=numpy.inf, t_ini=0.):
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

    def integrate(self, n: int, delta_t: float, nsave: int, result: numpy.ndarray, c_T: float, f: float, devel_state_ini: int, y_ini=None):
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

        E_m = p_Am / v
        L_m = kap * p_Am / p_M
        L_m3 = L_m**3

        def dy(y, t0: float):
            E, L, E_H, E_R, Q, H, S, N_RS, cumt = map(float, y)
            L2 = L * L
            L3 = L * L2
            s = max(1., min(s_M, L / L_b))

            # Energy fluxes in J/d
            p_C = E * ((v * E_G + p_T) * s * L2 + p_M * L3) / (E_G * L3 + kap * E)
            p_A = 0. if E_H < E_Hb else p_Am * L2 * f * s
            p_R = (1. - kap) * p_C - k_J * E_H  # J/d

            # Change in reserve (J), structural length (cm), maturity (J), reproduction buffer (J)
            dE = p_A - p_C
            dL = (kap * E * v * s - (p_M * L + p_T * s) * L3) / (E_G * L3 + kap * E) / 3.
            if E_H < E_Hp:
                delta_E_H = p_R * delta_t
                frac = 1. if E_H + delta_E_H <= E_Hp else (E_Hp - E_H) / delta_E_H
                dE_H = frac * p_R
                dE_R = kap_R * (1. - frac) * p_R
            else:
                dE_H = 0
                dE_R = kap_R * p_R

            # Damage-inducing compounds, damage, survival (0-1) - p 216
            dQ = (Q / L_m3 * s_G + h_a) * max(0., p_C) / E_m
            dH = Q
            dS = 0. if L3 <= 0. or S < 0. else -min(1. / (delta_t + 1e-8), H / L3) * S

            # Cumulative reproduction (#) and life span (d)
            dcumR = S * dE_R / E_0
            dcumt = S

            return numpy.array((dE, dL, dE_H, dE_R, dQ, dH, dS, dcumR, dcumt), dtype=float), dE_R / E_0

        assert result.shape == (n // nsave + 1, 11)
        y = numpy.array((E_0, 0., 0., 0., 0., 0., 1., 0., 0.))
        if y_ini is not None:
            y[:] = y_ini
        for it in range(n + 1):
            t = it * delta_t
            if it % nsave == 0:
                result[it // nsave, 0] = t
                result[it // nsave, 1:-1] = y
            derivative, R = dy(y, t)
            if not numpy.isfinite(derivative).all():
                raise Exception('Temporal derivatives at time %s contain NaN: %s' % (t, derivative))
            y += delta_t * derivative
            if it % nsave == 0:
                result[it // nsave, -1] = R

def create(use_cython=True):
    if use_cython and cmodel is not None:
        return cmodel.Model()
    return PyEngine()
