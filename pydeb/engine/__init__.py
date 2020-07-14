import numpy

try:
    import cmodel
except ImportError:
    try:
        import pyximport
        pyximport.install(setup_args={'include_dirs': numpy.get_include()}, language_level='3')
        from . import cmodel
    except ImportError as e:
        print('WARNING: unable to load Cython verison of model code. Performance will be reduced. Reason: %s' % e)
        cmodel = None

class PyEngine(object):
    def __init__(self):
        pass

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

    def integrate(n, delta_t, nsave, result, c_T, f, devel_state_ini):
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
            p_R = one_minus_kap*p_C - k_J*E_H  # J/d

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

def create():
    if cmodel is not None:
        return cmodel.Model()
