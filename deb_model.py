from __future__ import print_function

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
        self.t_b = None
        # L_m = kappa*v*E_m/p_M = kappa*p_Am/p_M [L_m is maximum length in absence of surface=-area-specific maintenance!]
        # z = L_m/L_m_ref with L_m_ref = 1 cm - equal to L_m

    def get_lb(self):
        kap = self.kap
        v = self.v
        E_G = self.E_G
        p_S = self.p_M
        E_Hb = self.E_Hb
        k_J = self.k_J
        E_m = self.p_Am*self.f/self.v
        g = E_G/kap/E_m
        k_M = self.p_M/E_G
        L_m = kap*self.p_Am/self.p_M

        E_G_per_kappa = E_G/kap
        p_S_per_kappa = p_S/kap
        one_minus_kappa = 1-kap

        def error_in_E(p, delta_t):
            E_0 = 10.**p[0]
            state = get_birth_state(E_0, delta_t)
            if state is None:
                return numpy.inf
            t_b, E_b, L_b = state
            ssq = (E_b/L_b**3 - E_m)**2
            #print('E_0 = %.3g J, t_b = %.3f d, L_b = %.3f cm - SSQ = %s' % (E_0, t_b, L_b, ssq))
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
                p_C = E*(v*L2*E_G_per_kappa + p_S_per_kappa*L3)/(E + E_G_per_kappa*L3)
                dL = (E*v-p_S_per_kappa*L2*L2)/3/(E+E_G_per_kappa*L3)
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

        import scipy.optimize
        p = numpy.log10(E_G*L_m**3),
        initial_simplex = None
        for delta_t in (1., 0.1, 0.01):
            p_new = scipy.optimize.fmin(error_in_E, p, args=(delta_t, ), initial_simplex=initial_simplex, disp=False)
            step = p_new[0] - p[0]
            initial_simplex = numpy.array(((p_new[0] - step/10, ), (p_new[0] + step/10, )))
            p = p_new
            E_0 = 10.**p[0]
            print(E_0, get_birth_state(E_0, delta_t=delta_t))
        self.E_0 = E_0
        self.t_b, Em, self.L_b = get_birth_state(E_0, delta_t=0.01)

    def simulate(self, delta_t=0.01):
        if self.E_0 is None:
            self.get_lb()
        c_T = 1. #2.6831
        kap = self.kap
        v = self.v*c_T
        k_J = self.k_J*c_T
        p_Am = self.p_Am*c_T
        p_M = self.p_M*c_T
        E_G = self.E_G
        E_Hb = self.E_Hb
        E_Hp = self.E_Hp
        f = self.f
        s_G = self.s_G
        h_a = self.h_a*c_T*c_T
        E_0 = self.E_0
        kap_R = self.kap_R

        p_S = p_M
        E_m = p_Am*f/v
        L_m = kap*p_Am/p_M
        L_m3 = L_m**3
        E_G_per_kappa = E_G/kap
        p_S_per_kappa = p_S/kap
        one_minus_kappa = 1-kap

        # NB scaled reserve density e = E/E_m with E_m = p_Am/v
        # scaled length l = L/L_m
        # From book p51: L_m = v/(g*k_M); inserting g = E_G/kappa/E_m -> L_m = v*kappa*E_m/E_G/k_M = kappa*p_Am/p_M
        E, L, E_H, E_R, Q, H, S, cumR, cumt = E_0, 0., 0., 0., 0., 0., 1., 0., 0.
        result = []
        while 1:
            #p_C = (v/L*E_G + p_S)/(kappa*E + E_G)
            #dE_H = (1-kappa)*p_C - k_J*E_H
            #dL = (E*v-p_S/kappa*L)/3/(E+E_G/kappa)
            L2 = L*L
            L3 = L*L2
            p_C = E*(v*L2*E_G_per_kappa + p_S_per_kappa*L3)/(E + E_G_per_kappa*L3)
            dL = (E*v-p_S_per_kappa*L2*L2)/3/(E+E_G_per_kappa*L3)
            dE = -p_C
            if E_H > E_Hb:
                dE += p_Am*L2*f
            if E_H < E_Hp:
                dE_H = one_minus_kappa*p_C - k_J*E_H
                dE_R = 0.
            else:
                dE_H = 0
                dE_R = one_minus_kappa*p_C - k_J*E_H
            dH = Q
            dQ = (Q/L_m3*s_G + h_a)*p_C/E_m
            dcumR = S*kap_R*dE_R/E_0
            if H > 0:
                dS = -H/L3*S
            else:
                dS = 0.
            dcumt = S
            E += delta_t * dE
            L += delta_t * dL
            E_H += delta_t * dE_H
            E_R += delta_t * dE_R
            Q += delta_t * dQ
            H += delta_t * dH
            S += delta_t * dS
            cumR += delta_t * dcumR
            cumt += delta_t * dcumt
            result.append((E, L, E_H, E_R, Q, H, S, cumR, cumt))
            if S < 1e-3:
                break

        return numpy.arange(len(result)) * delta_t, numpy.array(result)

    def plotResult(self, t, result):
        from matplotlib import pyplot
        fig = pyplot.figure()
        ax = fig.add_subplot(411)
        ijuv = result[:, 2].searchsorted(self.E_Hb)
        ipub = result[:, 2].searchsorted(self.E_Hp)
        ax.plot(t[:ijuv], result[:ijuv, 1], '-g')
        ax.plot(t[ijuv:ipub], result[ijuv:ipub, 1], '-b')
        ax.plot(t[ipub:], result[ipub:, 1], '-r')
        ax.grid()

        ax = fig.add_subplot(412)
        ax.set_title('maturity')
        ax.plot(t, result[:, 2], '-b')
        ax.grid()

        ax = fig.add_subplot(413)
        ax.set_title('cumulative reproductive output')
        ax.plot(t, result[:, 7], '-b')
        ax.grid()

        ax = fig.add_subplot(414)
        ax.set_title('survival')
        ax.plot(t, result[:, 6], '-b')
        ax.grid()

        pyplot.show()

#scipy.integrate.
if __name__ == '__main__':
    model = DEB_model()
    import argparse
    import io
    parser = argparse.ArgumentParser()
    parser.add_argument('species')
    args = parser.parse_args()
    with io.open('traits.txt', 'rU', encoding='utf-8') as f:
        labels = f.readline().rstrip('\n').split('\t')
        for l in f:
            items = l.rstrip('\n').split('\t')
            if items[0] == args.species:
                par2value = {}
                for name, item in zip(labels[2:], items[2:]):
                    if item != '':
                        par2value[name.split(' (')[0]] = float(item)
                break
    #   Dromaius novaehollandiae (Emu)
    print('Parameters for %s:' % args.species)
    for name, value in par2value.items():
        print('%s: %s' % (name, value))
        setattr(model, name, value)
    t, result = model.simulate()
    model.plotResult(t, result)