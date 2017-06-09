from __future__ import print_function

import os

import numpy
import scipy.integrate

class DEB_model(object):
    def __init__(self):
        self.p_Am = None # {p_Am}, spec assimilation flux (J/d.cm^2)
        self.v = None   # energy conductance (cm/d)
        self.p_M = None # [p_M], vol-spec somatic maint, J/d.cm^3 
        self.p_T = None # {p_T}, surf-spec somatic maint, J/d.cm^2
        self.kap = None
        self.E_G = None    # [E_G], spec cost for structure
        self.E_Hb = None   # maturity at birth (J)
        self.E_Hp = None   # maturity at puberty (J)
        self.k_J = None #k_J: maturity maint rate coefficient, 1/d
        self.h_a = None #Weibull aging acceleration (1/d^2)
        self.s_G = None #Gompertz stress coefficient
        self.kap_R = None # reproductive efficiency
        self.kap_X = None # ???
        self.p_T = 0.

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

        #kap_X: digestion efficiency of food to reserve
        #kap_P: digestion efficiency of food to faeces
        #kap_R: reproduction efficiency
        #F_m {F_m}, max spec searching rate (l/d.cm^2)
        #s_j: reprod buffer/structure at pupation as fraction of max (-) - hex model

        # pars specific for this entry
        self.f = 1.0 # functional response
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

    def initialize(self, E_0_ini=None):
        self.kap = max(min(self.kap, 1.), 0.)
        self.kap_R = max(min(self.kap_R, 1.), 0.)
        self.kap_X = max(min(self.kap_X, 1.), 0.)
        self.E_Hj = max(self.E_Hb, self.E_Hj)
        self.initialized = True
        kap = self.kap
        v = self.v
        E_G = self.E_G
        E_Hb = self.E_Hb
        k_J = self.k_J
        E_m = self.p_Am*self.f/self.v
        g = E_G/kap/E_m
        k_M = self.p_M/E_G
        L_m = kap*self.p_Am/self.p_M
        L_T = self.p_T/self.p_M
        f = self.f

        assert E_m > 0
        assert L_m > 0
        assert self.h_a > 0
        #assert self.s_G > 0

        E_G_per_kappa = E_G/kap
        p_M_per_kappa = self.p_M/kap
        p_T_per_kappa = self.p_T/kap
        v_E_G_plus_P_T_per_kappa = (v*E_G + self.p_T)/kap
        one_minus_kappa = 1-kap

        def error_in_E(p, delta_t):
            if not isinstance(p, float):
                p = float(numpy.asscalar(p))
            E_0 = 10.**p
            state = get_birth_state(E_0, delta_t)
            if state is None:
                return numpy.inf
            a_b, E_b, L_b = state
            ssq = (E_b/L_b**3 - E_m)**2
            #print('E_0 = %.3g J, a_b = %.3f d, L_b = %.3f cm - SSQ = %s' % (E_0, a_b, L_b, ssq))
            return ssq

        def get_birth_state(E_0, delta_t=1.):
            t, E, L, E_H = 0., float(E_0), 0., 0.
            done = False
            while not done:
                #p_C = (v/L*E_G + p_S)/(kappa*E + E_G)
                #dE_H = (1-kappa)*p_C - k_J*E_H
                #dL = (E*v-p_S/kappa*L)/3/(E+E_G/kappa)
                L2 = L*L
                L3 = L*L2
                p_C = E*(v_E_G_plus_P_T_per_kappa*L2 + p_M_per_kappa*L3)/(E + E_G_per_kappa*L3)
                dL = (E*v-(p_M_per_kappa*L+p_T_per_kappa)*L3)/3/(E+E_G_per_kappa*L3)
                dE = -p_C
                dE_H = one_minus_kappa*p_C - k_J*E_H
                if E_H + delta_t * dE_H > E_Hb:
                    delta_t = (E_Hb - E_H)/dE_H
                    done = True
                E += delta_t * dE
                L += delta_t * dL
                E_H += delta_t * dE_H
                t += delta_t
                if E < 0:
                    #print('E_0 = %.3e J: reserve is negative' % E_0)
                    return
                if dL < 0:
                    #print('E_0 = %.3e J: shrinking' % E_0)
                    return
            return t, E, L

        def find_maturity(E_H_target, delta_t=1., s_M=1., t_ini=None, E_H_ini=None, L_ini=None, t_max=numpy.inf):
            exp = numpy.exp
            r_B = self.r_B

            if t_ini is None:
                t_ini = self.a_b
            if E_H_ini is None:
                E_H_ini = self.E_Hb
            if L_ini is None:
                L_ini = self.L_b
            t = 0.
            E_H = E_H_ini
            L_i = (f*L_m - L_T)*s_M
            done = False
            while not done:
                L = (L_i-L_ini)*(1. - exp(-r_B*t)) + L_ini # p 52
                p_C = L*L*E_m*((v*E_G_per_kappa + p_T_per_kappa)*s_M + p_M_per_kappa*L)/(E_m + E_G_per_kappa)
                dE_H = (1. - kap)*p_C - k_J*E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H)/dE_H
                    done = True
                E_H += dE_H*delta_t
                t += delta_t
                if t > t_max:
                    return None, None
            L = (f*s_M*L_m-L_ini)*(1. - exp(-r_B*t)) + L_ini # p 52
            return t + t_ini, L

        def find_maturity_v1(E_H_target, delta_t=1., t_max=numpy.inf):
            exp = numpy.exp
            L_b = self.L_b
            V_b = L_b**3

            # dL = (E*v*L/L_b-p_S_per_kappa*L2*L2)/3/(E+E_G_per_kappa*L3)
            #    = (E_m*v/L_b-p_S_per_kappa)/3/(E_m+E_G_per_kappa) * L
            r = (kap*v/L_b*E_m - self.p_M - self.p_T)/(E_G+kap*E_m) # specific growth of structural VOLUME
            prefactor = V_b*E_m*(v*E_G/L_b + self.p_M + self.p_T/L_b)/(E_G + kap*E_m)

            t = 0.
            E_H = self.E_Hb
            done = False
            while not done:
                p_C = prefactor*exp(r*t)
                dE_H = (1. - kap)*p_C - k_J*E_H
                if E_H + delta_t * dE_H > E_H_target:
                    delta_t = (E_H_target - E_H)/dE_H
                    done = True
                E_H += dE_H*delta_t
                t += delta_t
                if t > t_max:
                    return None, None
            L = L_b*exp(r/3*t)
            return t + self.a_b, L

        import scipy.optimize
        if E_0_ini is None:
            E_0_ini = max(E_Hb, E_G*L_m**3)
        for i in range(10):
            if get_birth_state(E_0_ini) is not None:
                break
            E_0_ini *= 10
        else:
            return

        p = numpy.log10(E_0_ini),
        bracket = (p[0], p[0]+1)
        initial_simplex = None
        for delta_t in (1., 0.1, 0.01):
            if True:
                p_new = scipy.optimize.minimize_scalar(error_in_E, bracket=bracket, args=(delta_t, )).x,
                step = min(abs(p_new[0] - p[0])/10, 1.)
                bracket = (p_new[0] - step, p_new[0] + step,)
            else:
                p_new = scipy.optimize.fmin(error_in_E, p, args=(delta_t, ), disp=False) #, initial_simplex=initial_simplex
                step = min(abs(p_new[0] - p[0])/10, 1.)
                initial_simplex = numpy.array(((p_new[0] - step, ), (p_new[0] + step, )))
            p = p_new
            E_0 = 10.**p[0]
            #print(E_0, get_birth_state(E_0, delta_t=delta_t))
        self.E_0 = E_0
        birth_state = get_birth_state(E_0, delta_t=0.01)
        if birth_state is None:
            return
        self.a_b, Em, self.L_b = birth_state

        self.L_m = L_m
        self.L_T = L_T
        self.E_m = E_m

        self.r_B = self.p_M/3/(self.f*E_m*kap + E_G) # checked against p52
        L_i_min = self.L_m - self.L_T # not counting acceleration!
        if L_i_min < self.L_b:
            # shrinking directly after birth
            return
        a_99_max = self.a_b - numpy.log(1 - (0.99*L_i_min - self.L_b)/(L_i_min - self.L_b))/self.r_B
        self.a_j, self.L_j = find_maturity_v1(self.E_Hj, delta_t=max(0.01, self.a_b/100), t_max=a_99_max*100)
        if self.a_j is None:
            return
        self.s_M = self.L_j/self.L_b
        self.L_i = (self.L_m - self.L_T)*self.s_M
        self.a_p, self.L_p = find_maturity(self.E_Hp, delta_t=max(0.01, self.a_b/100), s_M=self.s_M, t_ini=self.a_j, E_H_ini=self.E_Hj, L_ini=self.L_j, t_max=a_99_max*100)
        if self.a_p is None:
            return
        self.a_99 = self.a_p - numpy.log(1 - (0.99*self.L_i - self.L_p)/(self.L_i - self.L_p))/self.r_B
        p_C_i = self.L_i*self.L_i*E_m*((v*E_G_per_kappa + p_T_per_kappa)*self.s_M + p_M_per_kappa*self.L_i)/(E_m + E_G_per_kappa)
        self.R_i = ((1-kap)*p_C_i - self.k_J*self.E_Hp)*self.kap_R/self.E_0
        self.valid = True

    def report(self, c_T=1.):
        if not self.initialized:
            self.initialize()
        print('E_0 [cost of an egg] = %s' % self.E_0)
        print('r_B [von Bertalanffy growth rate] = %s' % (c_T*self.r_B))
        print('a_b [age at birth] = %s' % (self.a_b/c_T))
        print('a_j [age at metamorphosis] = %s' % (self.a_j/c_T))
        print('a_p [age at puberty] = %s' % (self.a_p/c_T))
        print('a_99 [age at L = 0.99 L_m] = %s' % (self.a_99/c_T))
        print('[E_m] [reserve capacity] = %s' % self.E_m)
        print('L_b [structural length at birth] = %s' % self.L_b)
        print('L_b [structural length at metamorphosis] = %s' % self.L_j)
        print('L_p [structural length at puberty] = %s' % self.L_p)
        print('L_i [ultimate structural length] = %s' % self.L_i)
        print('s_M [acceleration factor at f=1] = %s' % self.s_M)
        print('R_i [ultimate reproduction rate] = %s' % (self.R_i*c_T))

    def simulate(self, t=None, c_T=1.):
        if not self.initialized:
            self.initialize()
        if not self.valid:
            return None, None
        kap = self.kap
        v = self.v*c_T
        k_J = self.k_J*c_T
        p_Am = self.p_Am*c_T
        p_M = self.p_M*c_T
        p_T = self.p_T*c_T
        E_G = self.E_G
        E_Hb = self.E_Hb
        E_Hp = self.E_Hp
        f = self.f
        s_G = self.s_G
        h_a = self.h_a*c_T*c_T
        E_0 = self.E_0
        kap_R = self.kap_R
        s_M = self.s_M
        L_b = self.L_b

        E_m = p_Am/v
        L_m = kap*p_Am/p_M
        L_m3 = L_m**3
        E_G_per_kappa = E_G/kap
        p_M_per_kappa = p_M/kap
        p_T_per_kappa = p_T/kap
        v_E_G_plus_P_T_per_kappa = (v*E_G + p_T)/kap
        one_minus_kappa = 1-kap

        if t is None:
            t = numpy.linspace(0., self.a_99/c_T, 1000)
        dt = t[1] - t[0]

        def dy(y, t0):
            E, L, E_H, E_R, Q, H, S, cumR, cumt = map(float, y)
            L2 = L*L
            L3 = L*L2
            s = max(1., min(s_M, L/L_b))
            #s = 1.

            # Energy fluxes in J/d
            p_C = E*(v_E_G_plus_P_T_per_kappa*s*L2 + p_M_per_kappa*L3)/(E + E_G_per_kappa*L3)
            p_A = 0. if E_H < E_Hb else p_Am*L2*f*s
            p_R = one_minus_kappa*p_C - k_J*E_H # J/d

            # Change in reserve (J), structural length (cm), maturity (J), reproduction buffer (J)
            dE = p_A - p_C
            dL = (E*v*s-(p_M_per_kappa*L+p_T_per_kappa*s)*L3)/3/(E+E_G_per_kappa*L3)
            if E_H < E_Hp:
                dE_H = p_R
                dE_R = 0.
            else:
                dE_H = 0
                dE_R = kap_R * p_R

            # Damage-inducing compounds, damage, survival (0-1) - p 216
            dQ = (Q/L_m3*s_G + h_a)*max(0., p_C)/E_m
            dH = Q
            dS = 0. if L3 <= 0. or S<1e-16 else -min(1./(dt+1e-8), H/L3)*S

            # Cumulative reproduction (#) and life span (d)
            dcumR = S*dE_R/E_0
            dcumt = S

            return numpy.array((dE, dL, dE_H, dE_R, dQ, dH, dS, dcumR, dcumt), dtype=float), dE_R/E_0

        if False:
            import scipy.integrate
            y0 = numpy.array((E_0, 0., 0., 0., 0., 1., 0.))
            it_b = t.searchsorted(self.a_b)
            t_embryo = t[:it_b]
            t_adult = t[it_b:]
            result = scipy.integrate.odeint(dy_embryo, y0, t_embryo)
            L = (f*L_m-self.L_b)*(1 - numpy.exp(-self.r_B*(t_adult - self.a_b))) + self.L_b # p 52
            result2 = numpy.zeros((L.shape[0], result.shape[1]))
            result2[:, 1] = L
            return t, result[:, 1], result[:, -1]
        else:
            y0 = numpy.array((E_0, 0., 0., 0., 0., 0., 1., 0., 0.))
            result = numpy.empty((t.size, y0.size))
            allR = numpy.empty((t.size,))
            result[0, :] = y0
            yold = y0
            for it, curt in enumerate(t[1:]):
                derivative, R = dy(yold, curt)
                yold = yold + dt*derivative
                result[it+1, :] = yold
                allR[it+1] = R
            return t, result[:, 1], result[:, 6], allR, result[:, 2]

    def plotResult(self, t, L, S, R, M):
        from matplotlib import pyplot
        fig = pyplot.figure()
        ax = fig.add_subplot(411)
        ijuv = M.searchsorted(self.E_Hb)
        ipub = M.searchsorted(self.E_Hp)
        ax.plot(t[:ijuv], L[:ijuv], '-g')
        ax.plot(t[ijuv:ipub], L[ijuv:ipub], '-b')
        ax.plot(t[ipub:], L[ipub:], '-r')
        ax.set_title('structural length')
        #ax.plot(t, L, '-b')
        ax.grid()

        ax = fig.add_subplot(412)
        ax.set_title('maturity')
        ax.plot(t, M, '-b')
        ax.grid()

        ax = fig.add_subplot(413)
        ax.set_title('reproduction rate')
        ax.plot(t, R, '-b')
        ax.grid()

        ax = fig.add_subplot(414)
        ax.set_title('survival')
        ax.plot(t, S, '-b')
        ax.grid()

        pyplot.show()

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

    def generate(self, workdir, color='k', label='', t_end=None, figsize=(8, 5)):
        import tempfile
        from matplotlib import pyplot

        # Collect results for all model instances
        if t_end is None:
            t_ends = numpy.sort([model.a_99 for model in self.valid_models])
            t_end = t_ends[int(0.9*len(t_ends))]
        t = numpy.linspace(0., t_end, 1000)
        Ls = numpy.empty((t.shape[0], len(self.valid_models)))
        Ss = numpy.empty((t.shape[0], len(self.valid_models)))
        Rs = numpy.empty((t.shape[0], len(self.valid_models)))
        for i, model in enumerate(self.valid_models):
            dummy_t, Ls[:, i], Ss[:, i], Rs[:, i], M = model.simulate(t)

        strings = []

        fig = pyplot.figure(figsize=figsize)
        params = 'E_0', 'a_b', 'a_p', 'L_b', 'L_p', 'R_i'
        for i, p in enumerate(params):
            ax = fig.add_subplot(1, len(params), i+1)
            values = [getattr(model, p) for model in self.valid_models]
            values = numpy.ma.log10(values)
            ax.boxplot(values, labels=(p,))
        fig.tight_layout()
        handle, name = tempfile.mkstemp(suffix='.png', dir=workdir)
        fig.savefig(os.fdopen(handle, 'wb'), dpi=72)
        strings.append('<img src="work/%s"/><br>' % os.path.basename(name))

        fig = pyplot.figure(figsize=figsize)
        ax = fig.gca()

        ax.cla()
        plot(ax, t, Ls, perc_wide=None, title='growth', ylabel='structural length (cm)', color=color)
        fig.tight_layout()
        handle, name = tempfile.mkstemp(suffix='.png', dir=workdir)
        fig.savefig(os.fdopen(handle, 'wb'), dpi=72)
        strings.append('<img src="work/%s"/><br>' % os.path.basename(name))

        ax.cla()
        plot(ax, t, Rs, perc_wide=None, title='reproduction', ylabel='reproduction rate (#/d)', color=color)
        fig.tight_layout()
        handle, name = tempfile.mkstemp(suffix='.png', dir=workdir)
        fig.savefig(os.fdopen(handle, 'wb'), dpi=72)
        strings.append('<img src="work/%s"/><br>' % os.path.basename(name))

        ax.cla()
        plot(ax, t, Ss, perc_wide=None, title='survival', ylabel='survival (-)', color=color)
        ax.set_ylim(0., 1.)
        fig.tight_layout()
        handle, name = tempfile.mkstemp(suffix='.png', dir=workdir)
        fig.savefig(os.fdopen(handle, 'wb'), dpi=72)
        strings.append('<img src="work/%s"/><br>' % os.path.basename(name))

        return '\n'.join(strings)

#scipy.integrate.
if __name__ == '__main__':
    model = DEB_model()
    import argparse
    import io
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
    result = model.simulate(c_T=args.c_T)
    model.plotResult(*result)