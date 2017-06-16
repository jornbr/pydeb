import numpy
cimport numpy
from math import exp

cdef class Model:
    cdef public double v
    cdef public double p_M
    cdef public double p_T
    cdef public double E_G
    cdef public double kap
    cdef public double k_J
    cdef public double p_Am
    cdef public double E_Hb
    cdef public double E_Hp
    cdef public double kap_R
    cdef public double h_a
    cdef public double s_G

    cdef public double E_0
    cdef public double L_b
    cdef public double L_m
    cdef public double L_T
    cdef public double r_B
    cdef public double s_M

    def get_birth_state(self, double E_0, double delta_t=1.):
        cdef double t, E, L, E_H
        cdef double dE, dL, dE_H
        cdef double L2, L3, denom, p_C

        cdef int done = 0

        cdef double p_M_per_kap = self.p_M/self.kap
        cdef double p_T_per_kap = self.p_T/self.kap
        cdef double E_G_per_kap = self.E_G/self.kap
        cdef double one_minus_kap = 1. - self.kap
        cdef double v_E_G_plus_P_T_per_kap = (self.v*self.E_G + self.p_T)/self.kap

        cdef double dt = delta_t

        t, E, L, E_H = 0., E_0, 0., 0.
        while done == 0:
            L2 = L*L
            L3 = L*L2
            denom = E + E_G_per_kap*L3
            p_C = E*(v_E_G_plus_P_T_per_kap*L2 + p_M_per_kap*L3)/denom
            dL = (E*self.v-(p_M_per_kap*L+p_T_per_kap)*L3)/3/denom
            dE = -p_C
            dE_H = one_minus_kap*p_C - self.k_J*E_H
            if E_H + dt * dE_H > self.E_Hb:
                dt = (self.E_Hb - E_H)/dE_H
                done = 1
            t += dt
            E += dt * dE
            L += dt * dL
            E_H += dt * dE_H
            if E < 0 or dL < 0:
                return
        return t, E, L

    def integrate(self, int n, double delta_t, c_T=1., f=1.):
        cdef numpy.ndarray[numpy.double_t, ndim=2] result = numpy.zeros([n+1, 10], dtype=numpy.double)

        cdef double kap, v, k_J, p_Am, p_M, p_T, E_G, E_Hb, E_Hp, s_G, h_a, E_0, kap_R, s_M, L_b
        cdef double E_m, L_m, L_m3, E_G_per_kap, p_M_per_kap, p_T_per_kap, v_E_G_plus_P_T_per_kap, one_minus_kap
        cdef double L2, L3, s, p_C, p_A, p_R, denom
        cdef double E, L, E_H, E_R, Q, H, S, cumR, cumt
        cdef int i

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

        E, L, E_H, E_R, Q, H, S, cumR, cumt = E_0, 0., 0., 0., 0., 0., 1., 0., 0.
        for i in range(n+1):
            result[i, 0] = E
            result[i, 1] = L
            result[i, 2] = E_H
            result[i, 3] = E_R
            result[i, 4] = Q
            result[i, 5] = H
            result[i, 6] = S
            result[i, 7] = cumR
            result[i, 8] = cumt

            L2 = L*L
            L3 = L*L2
            s = max(1., min(s_M, L/L_b))

            # Energy fluxes in J/d
            denom = E + E_G_per_kap*L3
            p_C = E*(v_E_G_plus_P_T_per_kap*s*L2 + p_M_per_kap*L3)/denom
            p_A = 0. if E_H < E_Hb else p_Am*L2*f*s
            p_R = one_minus_kap*p_C - k_J*E_H # J/d

            # Change in reserve (J), structural length (cm), maturity (J), reproduction buffer (J)
            dE = p_A - p_C
            dL = (E*v*s-(p_M_per_kap*L+p_T_per_kap*s)*L3)/3/denom
            if E_H < E_Hp:
                dE_H = p_R
                dE_R = 0.
            else:
                dE_H = 0
                dE_R = kap_R * p_R

            # Damage-inducing compounds, damage, survival (0-1) - p 216
            dQ = (Q/L_m3*s_G + h_a)*max(0., p_C)/E_m
            dH = Q
            dS = 0. if L3 <= 0. or S<1e-16 else -min(1./(delta_t+1e-8), H/L3)*S

            # Cumulative reproduction (#) and life span (d)
            dcumR = S*dE_R/E_0
            dcumt = S

            # Update state
            E += delta_t * dE
            L += delta_t * dL
            E_H += delta_t * dE_H
            E_R += delta_t * dE_R
            Q += delta_t * dQ
            H += delta_t * dH
            S += delta_t * dS
            cumR += delta_t * dcumR
            cumt += delta_t * dcumt

            # Save diagnostics
            result[i, 9] = dE_R/E_0

        return result

    def find_maturity(self, double L_ini, double E_H_ini, double E_H_target, double delta_t=1., double s_M=1., double t_max=365000., double t_ini=0.):
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
        L_range = (self.L_m - self.L_T)*self.s_M - L_ini  # note L_i at f=1
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
                return None, None
        L = L_range*(1. - exp(-r_B*t)) + L_ini # p 52
        return t_ini + t, L

    def find_maturity_v1(self, double L_ini, double E_H_ini, double E_H_target, double delta_t=1., double t_max=365000., double t_ini=0.):
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
        done = 0
        while done == 0:
            dE_H = prefactor*exp(r*t) - k_J*E_H
            if E_H + delta_t * dE_H > E_H_target:
                delta_t = (E_H_target - E_H)/dE_H
                done = 1
            E_H += dE_H*delta_t
            t += delta_t
            if t > t_max:
                return None, None

        L = L_ini*exp(r/3*t)
        return t_ini + t, L
