cimport cython
cimport numpy
from numpy.math cimport INFINITY

from libc.math cimport exp
from optimize cimport Function, optimize

ctypedef numpy.double_t DTYPE_t

numpy.import_array()

cdef class error_in_E(Function):
    cdef double delta_t
    cdef Model model
    cdef double E_m

    @cython.cdivision(True)
    cdef double evaluate(error_in_E self, double x):
        cdef double a_b, E_b, L_b 
        E_0 = 10.**x
        a_b, E_b, L_b = self.model.get_birth_state(E_0, self.delta_t)
        if a_b == -1.:
            return INFINITY
        ssq = (E_b / (L_b * L_b * L_b) - self.E_m)**2
        return ssq

cdef class Model:
    cdef public double v
    cdef public double p_M
    cdef public double p_T
    cdef public double E_G
    cdef public double kap
    cdef public double k_J
    cdef public double p_Am
    cdef public double E_Hb
    cdef public double E_Hj
    cdef public double E_Hp
    cdef public double kap_R
    cdef public double kap_X
    cdef public double h_a
    cdef public double s_G

    cdef public double E_0
    cdef public double L_b
    cdef public double L_m
    cdef public double L_T
    cdef public double r_B
    cdef public double s_M

    @cython.cdivision(True)
    def get_E_0(Model self, double log10_E_0_left, double log10_E_0_right, double delta_t):
        cdef error_in_E func = error_in_E()
        func.model = self
        func.delta_t = delta_t
        func.E_m = self.p_Am/self.v
        log10_E_0 = optimize(func, log10_E_0_left, log10_E_0_right)
        a_b, E_b, L_b = self.get_birth_state(10.**log10_E_0, delta_t)
        return log10_E_0, a_b, L_b

    @cython.cdivision(True)
    cpdef (double, double, double) get_birth_state(Model self, double E_0, double delta_t=1.):
        cdef double t, E, L, E_H
        cdef double dE, dL, dE_H
        cdef double L2, L3, denom, p_C

        cdef int done = 0

        cdef double p_M_per_kap = self.p_M/self.kap
        cdef double p_T_per_kap = self.p_T/self.kap
        cdef double E_G_per_kap = self.E_G/self.kap
        cdef double one_minus_kap = 1. - self.kap
        cdef double v_E_G_plus_P_T_per_kap = (self.v*self.E_G + self.p_T)/self.kap
        cdef double v = self.v
        cdef double k_J = self.k_J
        cdef double E_Hb = self.E_Hb

        cdef double dt = delta_t

        with nogil:
            t, E, L, E_H = 0., E_0, 0., 0.
            while done == 0:
                L2 = L*L
                L3 = L*L2
                denom = E + E_G_per_kap*L3
                p_C = E*(v_E_G_plus_P_T_per_kap*L2 + p_M_per_kap*L3)/denom
                dL = (E*v-(p_M_per_kap*L+p_T_per_kap)*L3)/3/denom
                dE = -p_C
                dE_H = one_minus_kap*p_C - k_J*E_H
                if E_H + dt * dE_H > E_Hb:
                    dt = (E_Hb - E_H)/dE_H
                    done = 1
                t += dt
                E += dt * dE
                L += dt * dL
                E_H += dt * dE_H
                if E < 0 or dL < 0:
                    done = 2
        if done == 1:
            return t, E, L
        return -1, -1, -1

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def integrate(Model self, int n, double delta_t, int nsave, double c_T=1., double f=1., int devel_state_ini=1):
        cdef numpy.npy_intp *dims = [(n/nsave)+1, 10]
        cdef numpy.ndarray[DTYPE_t, ndim=2] result = numpy.PyArray_EMPTY(2, dims, numpy.NPY_DOUBLE, 0)

        cdef double kap, v, k_J, p_Am, p_M, p_T, E_G, E_Hb, E_Hj, E_Hp, s_G, h_a, E_0, kap_R, s_M, L_b
        cdef double E_m, L_m, L_m3, E_G_per_kap, p_M_per_kap, p_T_per_kap, v_E_G_plus_P_T_per_kap, one_minus_kap
        cdef double L2, L3, s, p_C, p_R, denom
        cdef double E, L, E_H, E_R, Q, H, S, cumR, cumt
        cdef int i, isave, devel_state

        kap = self.kap
        v = self.v*c_T
        k_J = self.k_J*c_T
        p_Am = self.p_Am*c_T
        p_M = self.p_M*c_T
        p_T = self.p_T*c_T
        E_G = self.E_G
        E_Hb = self.E_Hb
        E_Hj = self.E_Hj
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
        devel_state = devel_state_ini

        dE_R = 0.
        with nogil:
            E, L, E_H, E_R, Q, H, S, cumR, cumt = E_0, 0., 0., 0., 0., 0., 1., 0., 0.
            for i in range(n+1):
                isave = i/nsave
                if i % nsave == 0:
                    result[isave, 0] = E
                    result[isave, 1] = L
                    result[isave, 2] = E_H
                    result[isave, 3] = E_R
                    result[isave, 4] = Q
                    result[isave, 5] = H
                    result[isave, 6] = S
                    result[isave, 7] = cumR
                    result[isave, 8] = cumt

                L2 = L*L
                L3 = L*L2
                s = max(1., min(s_M, L/L_b))

                # Calculate current p_C (J/d) and update to next L and E
                if (devel_state == -1):
                    # developing foetus - explicit equations for L(t) and E(t)
                    p_C = v_E_G_plus_P_T_per_kap * L2 + p_M_per_kap * L3
                    L = v * ((i + 1) * delta_t) / 3
                    E = L * E_m
                else:
                    denom = E + E_G_per_kap*L3
                    p_C = E*(v_E_G_plus_P_T_per_kap*s*L2 + p_M_per_kap*L3)/denom

                    dE = - p_C
                    if devel_state > 1:
                        # no longer an embryo/foetus - feeding/assimilation is active
                        dE += p_Am * L2 * f * s
                    dL = (E * v * s - (p_M_per_kap * L + p_T_per_kap * s) * L3) / 3 / denom
                    E += delta_t * dE
                    L += delta_t * dL

                # Change in maturity (J) and reproduction buffer (J)
                p_R = one_minus_kap * p_C - k_J * E_H   # J/d
                if devel_state < 3:
                    # maturation: update maturity and development state
                    E_H += delta_t * p_R
                    if (E_H > E_Hp):
                        devel_state = 3   # adult
                    elif (E_H > E_Hb):
                        devel_state = 2   # juvenile (post birth)
                else:
                    # reproduction (cumulative allocation in J/d and average total offspring over lifetime in #)
                    dE_R = kap_R * p_R
                    E_R += delta_t * dE_R
                    cumR += delta_t * S * dE_R / E_0

                # Damage-inducing compounds, damage, survival (0-1) - p 216
                dQ = (Q/L_m3*s_G + h_a)*max(0., p_C)/E_m
                dH = Q
                dS = 0. if L3 <= 0. or S<1e-16 else -min(1./(delta_t+1e-8), H/L3)*S

                # Update state variables related to survival
                Q += delta_t * dQ     # damage inducing compounds (1/d2)
                H += delta_t * dH     # hazard rate (1/d)
                S += delta_t * dS     # survival (-)
                cumt += delta_t * S   # average life span (d)

                # Save diagnostics
                if i % nsave == 0:
                    result[isave, 9] = dE_R/E_0

        return result

    @cython.cdivision(True)
    def find_maturity(Model self, double L_ini, double E_H_ini, double E_H_target, double delta_t=1., double s_M=1., double t_max=365000., double t_ini=0.):
        cdef double r_B, E_m, E_G_per_kap, p_M_per_kap, p_T_per_kap, one_minus_kap, k_J, v_E_G_plus_P_T_per_kap
        cdef double t, E_H
        cdef double L_i
        cdef double L, p_C, dE_H
        cdef int done

        r_B = self.r_B
        E_G_per_kap = self.E_G/self.kap
        p_M_per_kap = self.p_M/self.kap
        p_T_per_kap = self.p_T/self.kap
        v_E_G_plus_P_T_per_kap = (self.v*self.E_G + self.p_T)/self.kap
        E_m = self.p_Am/self.v
        one_minus_kap = 1.-self.kap
        k_J = self.k_J

        t = 0.
        E_H = E_H_ini
        L_range = (self.L_m - self.L_T)*s_M - L_ini  # note L_i at f=1
        with nogil:
            done = 0
            while done == 0:
                L = L_range*(1. - exp(-r_B*t)) + L_ini # p 52
                p_C = L*L*E_m*(v_E_G_plus_P_T_per_kap*s_M + p_M_per_kap*L)/(E_m + E_G_per_kap)
                dE_H = one_minus_kap*p_C - k_J*E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H)/dE_H
                    done = 1
                t += delta_t
                E_H += dE_H*delta_t
                if t > t_max:
                    done = 2
        if done == 1:
            L = L_range*(1. - exp(-r_B*t)) + L_ini # p 52
            return t_ini + t, L
        return None, None

    @cython.cdivision(True)
    def find_maturity_v1(Model self, double L_ini, double E_H_ini, double E_H_target, double delta_t=1., double t_max=365000., double t_ini=0.):
        cdef double E_m, v, kap, p_M, p_T, E_G, V_ini, r, prefactor, k_J
        cdef double t, E_H
        cdef int done
        cdef double dE_H

        E_m = self.p_Am/self.v
        v = self.v
        kap = self.kap
        p_M = self.p_M
        p_T = self.p_T
        E_G = self.E_G
        k_J = self.k_J
        V_ini = L_ini**3

        # dL = (E*v*L/L_b-p_S_per_kappa*L2*L2)/3/(E+E_G_per_kappa*L3)
        #    = (E_m*v/L_b-p_S_per_kappa)/3/(E_m+E_G_per_kappa) * L
        r = (kap*v/L_ini*E_m - p_M - p_T)/(E_G + kap*E_m) # specific growth rate of structural VOLUME
        prefactor = (1. - kap)*V_ini*E_m*(v*E_G/L_ini + p_M + p_T/L_ini)/(E_G + kap*E_m)

        t = 0.
        E_H = self.E_Hb
        with nogil:
            done = 0
            while done == 0:
                dE_H = prefactor*exp(r*t) - k_J*E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H)/dE_H
                    done = 1
                E_H += dE_H*delta_t
                t += delta_t
                if t > t_max:
                    done = 2
        if done == 1:
            L = L_ini*exp(r/3*t)
            return t_ini + t, L
        return None, None

    @cython.cdivision(True)
    def find_maturity_foetus(Model self, double E_H_target, double delta_t=1., double t_max=365000.):
        cdef double k_J, prefactor1, prefactor2
        cdef double t, E_H, E_0
        cdef int done
        cdef double dE_H

        k_J = self.k_J
        prefactor1 = (1. - self.kap) * (self.v * self.E_G + self.p_T) * self.v * self.v / (9 * self.kap)
        prefactor2 = (1. - self.kap) * self.p_M * self.v**3 / (27 * self.kap)

        t = 0.
        E_H = 0.
        with nogil:
            done = 0
            while done == 0:
                dE_H = (prefactor1 + prefactor2 * t) * t * t - k_J * E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H) / dE_H
                    done = 1
                E_H += dE_H * delta_t
                t += delta_t
                if t > t_max:
                    done = 2
        if done == 1:
            E = self.v * self.v * t * t * t / 27 * self.p_Am
            p_C_int = (prefactor1 / 3 + prefactor2 * t / 4) * t * t * t / (1. - self.kap)
            E_0 = p_C_int + E
            return t, self.v * t / 3, E_0
        return None