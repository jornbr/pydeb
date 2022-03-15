# cython: language_level=3

cimport cython
from numpy.math cimport INFINITY

from libc.math cimport exp, log10, sqrt, cbrt
from .optimize cimport Function, optimize, brentq

DEF onethird = 0.3333333333333333

@cython.final
cdef class E_Hb_difference(Function):
    cdef double delta_t
    cdef Model model
    cdef double E_Hb
    cdef double f

    @cython.cdivision(True)
    cdef double evaluate(E_Hb_difference self, double x) nogil:
        cdef double E_H
        E_H = self.model.get_birth_state(x, self.delta_t, self.f)[2]
        return E_H - self.E_Hb

@cython.final
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

    cdef public double L_b
    cdef public double L_m
    cdef public double L_T
    cdef public double r_B
    cdef public double s_M

    @cython.cdivision(True)
    def get_E_0(Model self, double E_0_left, double E_0_right, double delta_t, double precision, double f=1.):
        cdef E_Hb_difference func = E_Hb_difference()
        func.model = self
        func.delta_t = delta_t
        func.E_Hb = self.E_Hb
        func.f = f
        E_0 = brentq(func, E_0_left, E_0_right, xtol = 2e-12, rtol=precision, maxiter=100)
        a_b, L_b, _ = self.get_birth_state(E_0, delta_t, f)
        return E_0, a_b, L_b

    @cython.cdivision(True)
    cpdef (double, double, double) get_birth_state(Model self, double E_0, double delta_t, double f) nogil:
        cdef double t, E, L, E_H
        cdef double dE, dL, dE_H
        cdef double L2, L3, invdenom, p_C

        cdef int done = 0

        cdef double p_M_per_kap = self.p_M / self.kap
        cdef double p_T_per_kap = self.p_T / self.kap
        cdef double E_G_per_kap = self.E_G / self.kap
        cdef double one_minus_kap = 1. - self.kap
        cdef double v_E_G_plus_P_T_per_kap = (self.v * self.E_G + self.p_T) / self.kap
        cdef double v = self.v
        cdef double k_J = self.k_J
        cdef double E_m = f * self.p_Am / self.v
        cdef double L_i = f * self.kap * self.p_Am / self.p_M - self.p_T / self.p_M

        t, E, L, E_H = 0., E_0, 0., 0.
        while done == 0:
            L2 = L * L
            L3 = L * L2
            invdenom = 1. / (E + E_G_per_kap * L3)
            p_C = E * (v_E_G_plus_P_T_per_kap * L2 + p_M_per_kap * L3) * invdenom
            dL = (E * v - (p_M_per_kap * L + p_T_per_kap) * L3) * invdenom * onethird
            dE = -p_C
            dE_H = one_minus_kap * p_C - k_J * E_H
            L_new = min(L_i, L + delta_t * dL)  # If L tends to go above L_i, it would shrink later on. E_0 is then too high. We just need to return E_H >> E_Hb
            if E + delta_t * dE < E_m * L_new * L_new * L_new:
                p = p_C / (dL * E_m)
                q = - (E + p_C * L / dL) / E_m
                c1 = -q/2
                c2 = sqrt(q*q/4 + p*p*p/27)
                L_new = cbrt(c1 + c2) - cbrt(c2 - c1)
                delta_t = (L_new - L) / dL
                done = 1
            t += delta_t
            E += delta_t * dE
            L = L_new
            E_H += delta_t * dE_H
        return t, L, E_H

    @cython.cdivision(True)
    @cython.boundscheck(False) # turn off bounds-checking for entire function
    @cython.wraparound(False)  # turn off negative index wrapping for entire function
    def integrate(Model self, int n, double delta_t, int nsave, double [:, ::1] result not None, double E_0, double c_T=1., double f=1., int devel_state_ini=1, double S_crit=-1., double E_H_crit=-1., double [::1] y_ini=None):
        cdef double kap, v, k_J, p_Am, p_M, p_T, E_G, E_Hb, E_Hj, E_Hp, s_G, h_a, inv_E_0, kap_R, inv_delta_t, h_a_per_E_m, s_G_per_L_m3_E_m
        cdef double E_m, L_m, E_G_per_kap, p_M_per_kap, p_T_per_kap, v_E_G_plus_P_T_per_kap, one_minus_kap, p_Am_f
        cdef double L2, L3, s, p_C, invdenom
        cdef double E, L, E_H, E_R, Q, H, S, RSint, cumt
        cdef int i, isave, devel_state, steps_till_save

        kap = self.kap
        v = self.v * c_T * delta_t
        k_J = self.k_J * c_T * delta_t
        p_Am = self.p_Am * c_T * delta_t
        p_M = self.p_M * c_T * delta_t
        p_T = self.p_T * c_T * delta_t
        E_G = self.E_G
        E_Hb = self.E_Hb
        E_Hj = self.E_Hj
        E_Hp = self.E_Hp
        s_G = self.s_G
        h_a = self.h_a * c_T * c_T * delta_t * delta_t
        inv_E_0 = 1. / E_0
        kap_R = self.kap_R
        inv_delta_t = 1. - 1e-8

        E_m = p_Am / v
        L_m = kap * p_Am / p_M
        E_G_per_kap = E_G / kap
        p_M_per_kap = p_M / kap
        p_T_per_kap = p_T / kap
        v_E_G_plus_P_T_per_kap = (v * E_G + p_T) / kap
        s_G_per_L_m3_E_m = s_G / L_m**3 / E_m
        h_a_per_E_m = h_a / E_m
        one_minus_kap = 1 - kap
        devel_state = devel_state_ini
        p_Am_f = p_Am * f

        if E_H_crit == -1.:
            E_H_crit = 2. * E_Hp

        if y_ini is None:
            E, L, E_H, E_R, Q, H, S, RSint, cumt, s = E_0, 0., 0., 0., 0., 0., 1., 0., 0., 1.
            if devel_state == -1:
                E = 0.
        else:
            E, L, E_H, E_R, Q, H, S, RSint, cumt, s = y_ini
            E_R /= kap_R
            Q *= delta_t * delta_t
            H *= delta_t
            RSint /= kap_R * inv_E_0
            cumt /= delta_t
        if E_H >= E_Hp:
            devel_state = 4   # adult
        elif E_H >= E_Hj:
            devel_state = 3   # juvenile after metamorphosis
        elif E_H >= E_Hb:
            devel_state = 2   # juvenile before metamorphosis

        with nogil:
            dE_R = 0.
            isave = 0
            steps_till_save = n if nsave == 0 else 0
            if devel_state == -1:
                # foetal development: no initial reserve, constant increase in length until birth
                dL = onethird * v
            for i in range(n + 1):
                if nsave == 0 and (S < S_crit or E_H >= E_H_crit):
                    steps_till_save = 0
                if steps_till_save == 0:
                    result[isave, 0] = i * delta_t
                    result[isave, 1] = E
                    result[isave, 2] = L
                    result[isave, 3] = E_H
                    result[isave, 4] = kap_R * E_R
                    result[isave, 5] = Q / delta_t / delta_t
                    result[isave, 6] = H / delta_t
                    result[isave, 7] = S
                    result[isave, 8] = kap_R * RSint * inv_E_0
                    result[isave, 9] = cumt * delta_t
                    result[isave, 10] = s

                L2 = L * L
                L3 = L * L2

                # Calculate current p_C (J/d) and update to next L and E
                if (devel_state == -1):
                    # developing foetus - explicit equations for L(t) and E(t)
                    p_C = v_E_G_plus_P_T_per_kap * L2 + p_M_per_kap * L3
                    L += dL
                    E = L * L * L * E_m
                else:
                    invdenom = 1. / (E + E_G_per_kap * L3)
                    p_C = E * (v_E_G_plus_P_T_per_kap * s * L2 + p_M_per_kap * L3) * invdenom

                    dE = -p_C
                    if devel_state > 1:
                        # no longer an embryo/foetus - feeding/assimilation is active
                        dE += p_Am_f * L2 * s
                    dL = (E * v * s - (p_M_per_kap * L + p_T_per_kap * s) * L3) * onethird * invdenom
                    if devel_state == 2:
                        # Between birth and metamorphosis: update acceleration factor
                        s += dL / L * s
                    E = max(0., E + dE)
                    L = max(0., L + dL)

                # Change in maturity (J) and reproduction buffer (J)
                dE_H = one_minus_kap * p_C - k_J * E_H   # J/dt
                E_H += dE_H    # first add all to maturity buffer; anything above E_Hp will be moved to reproduction buffer later

                # Determine whether the increase in maturity triggered a life history event
                if devel_state < 2 and E_H >= E_Hb:      # *** birth ***
                    f_delta_t = (E_H - E_Hb) / dE_H      # fraction of the time step that falls after birth (linear interpolation of E_H)
                    L_b = L - f_delta_t * dL             # linear interpolation to length-at-birth (undo part of the already-applied delta_t)
                    E += f_delta_t * p_Am_f * L_b * L_b  # add assimilation for the fraction of the time step that falls after birth
                    devel_state = 2
                    s = L / L_b                          # update acceleration based on the fraction of the time step between birth and metamorphosis
                if devel_state == 2 and E_H >= E_Hj:     # *** metamorphosis ***
                    f_delta_t = (E_H - E_Hj) / dE_H      # fraction of the time step that falls after metamorphosis (linear interpolation of E_H)
                    L_j = L - f_delta_t * dL             # linear interpolation to length-at-metamorphosis (undo part of the already-applied delta_t)
                    s *= L_j / L                         # correct final acceleration
                    devel_state = 3
                if devel_state == 3 and E_H >= E_Hp:     # *** puberty ***
                    devel_state = 4

                # If this is a mature individual, move all E_H > E_Hp into reproduction buffer
                if devel_state == 4:
                    dE_R = E_H - E_Hp    # allocation to reproduction buffer (J/dt)
                    E_H = E_Hp           # reset E_H to E_Hp
                    E_R += dE_R          # cumulative allocation (J)
                    RSint += S * dE_R    # life time expected allocation (J)

                # Damage-inducing compounds, damage, survival (0-1) - p 216
                dQ = max((Q * s_G_per_L_m3_E_m + h_a_per_E_m) * max(0., p_C), -Q * inv_delta_t)
                dH = Q
                dS = 0. if L3 <= 0. or S < 0. else -min(inv_delta_t, H / L3) * S

                # Update state variables related to survival
                Q += dQ     # damage inducing compounds (1/dt2)
                H += dH     # hazard rate (1/dt) multiplied by structural volume
                S += dS     # survival (-)
                cumt += S   # average life span (dt)

                # Save diagnostics
                if steps_till_save == 0:
                    result[isave, 11] = kap_R * dE_R * inv_E_0 / delta_t
                    if nsave == 0:
                        break
                    isave += 1
                    steps_till_save = nsave

                steps_till_save -= 1

    @cython.cdivision(True)
    cpdef (double, double) find_maturity(Model self, double L_ini, double E_H_ini, double E_H_target, double delta_t=1., double s_M=1., double t_max=365000., double t_ini=0.):
        cdef double r_B, E_m, E_G_per_kap, p_M_per_kap, p_T_per_kap, one_minus_kap, k_J, v_E_G_plus_P_T_per_kap, prefactor
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
        prefactor = E_m / (E_m + E_G_per_kap)
        exp_min_r_B_delta_t = exp(-r_B * delta_t)

        t = 0.
        E_H = E_H_ini
        exp_min_r_B_t = 1.
        L_range = (self.L_m - self.L_T) * s_M - L_ini  # note L_i at f=1
        with nogil:
            done = 0
            while done == 0:
                L = L_range * (1. - exp_min_r_B_t) + L_ini # p 52
                p_C = L * L * prefactor * (v_E_G_plus_P_T_per_kap * s_M + p_M_per_kap * L)
                dE_H = one_minus_kap * p_C - k_J * E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H) / dE_H
                    done = 1
                t += delta_t
                E_H += dE_H * delta_t
                exp_min_r_B_t *= exp_min_r_B_delta_t
                if t > t_max:
                    done = 2
        if done == 1:
            L = L_range * (1. - exp(-r_B * t)) + L_ini # p 52
            return t_ini + t, L
        return -1., -1.

    @cython.cdivision(True)
    cpdef (double, double) find_maturity_v1(Model self, double L_ini, double E_H_ini, double E_H_target, double delta_t=1., double t_max=365000., double t_ini=0.):
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
        exp_r_delta_t = exp(r * delta_t)

        t = 0.
        E_H = E_H_ini
        exp_r_t = 1.
        with nogil:
            done = 0
            while done == 0:
                dE_H = prefactor * exp_r_t - k_J * E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H) / dE_H
                    done = 1
                E_H += dE_H * delta_t
                t += delta_t
                exp_r_t *= exp_r_delta_t
                if t > t_max:
                    done = 2
        if done == 1:
            L = L_ini * exp(r * t * onethird)
            return t_ini + t, L
        return -1., -1.

    @cython.cdivision(True)
    cpdef (double, double, double) find_maturity_foetus(Model self, double E_H_target, double delta_t=1., double t_max=365000.):
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
            done = 0 if E_H < E_H_target else 1
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
        return -1., -1., -1.

    @cython.cdivision(True)
    cpdef (double, double) find_maturity_egg(Model self, double E_H_target, double delta_t, double E_0, double t_max=365000.):
        cdef double t, E, L, E_H
        cdef double dE, dL, dE_H
        cdef double L2, L3, denom, p_C

        cdef int done

        cdef double p_M_per_kap = self.p_M / self.kap
        cdef double p_T_per_kap = self.p_T / self.kap
        cdef double E_G_per_kap = self.E_G / self.kap
        cdef double one_minus_kap = 1. - self.kap
        cdef double v_E_G_plus_P_T_per_kap = (self.v * self.E_G + self.p_T) / self.kap
        cdef double v = self.v
        cdef double k_J = self.k_J
        cdef double E_m = self.p_Am / self.v

        t, E, L, E_H = 0., E_0, 0., 0.
        with nogil:
            done = 0 if E_H < E_H_target else 1
            while done == 0:
                L2 = L * L
                L3 = L * L2
                invdenom = 1. / (E + E_G_per_kap * L3)
                p_C = E * (v_E_G_plus_P_T_per_kap * L2 + p_M_per_kap * L3) * invdenom
                dL = (E * v - (p_M_per_kap * L + p_T_per_kap) * L3) * onethird * invdenom
                dE = -p_C
                dE_H = one_minus_kap * p_C - k_J * E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H) / dE_H
                    done = 1
                t += delta_t
                E += delta_t * dE
                L += delta_t * dL
                E_H += delta_t * dE_H
                if t > t_max:
                    done = 2
        if done == 1:
            return t, L
        return -1., -1.
