# Brent's method based on scipy.optimize.optimize (minimize_scalar with method='Brent')

# cython: cdivision=True

cimport cython
from libc.math cimport fabs as abs
from numpy.math cimport INFINITY

cdef class Function:
    cdef double evaluate(Function self, double x) nogil:
        return 0


cdef double optimize(Function func, double xa, double xb) nogil:
    cdef double tol = 1.48e-8
    cdef double _mintol = 1.0e-11
    cdef double _cg = 0.3819660
    cdef int maxiter = 500

    cdef double x, u, v, w
    cdef double fx, fu, fv, fw
    cdef double xc
    cdef double fa, fb, fc
    cdef double a, b
    cdef int iter
    cdef double tol1, tol2
    cdef double xmid
    cdef double deltax
    cdef double rat
    cdef double tmp1, tmp2, p, dx_temp

    # set up for optimization
    xa, xb, xc, fa, fb, fc, iter = bracket(func, xa, xb, 110.0, 1000)
    if iter == -1:
        return INFINITY

    #################################
    #BEGIN CORE ALGORITHM
    #################################
    x = w = v = xb
    fw = fv = fx = func.evaluate(x)
    if (xa < xc):
        a = xa
        b = xc
    else:
        a = xc
        b = xa
    deltax = 0.0
    iter = 0
    while (iter < maxiter):
        tol1 = tol * abs(x) + _mintol
        tol2 = 2.0 * tol1
        xmid = 0.5 * (a + b)
        # check for convergence
        if abs(x - xmid) < (tol2 - 0.5 * (b - a)):
            break
        # XXX In the first iteration, rat is only bound in the true case
        # of this conditional. This used to cause an UnboundLocalError
        # (gh-4140). It should be set before the if (but to what?).
        if (abs(deltax) <= tol1):
            if (x >= xmid):
                deltax = a - x       # do a golden section step
            else:
                deltax = b - x
            rat = _cg * deltax
        else:                              # do a parabolic step
            tmp1 = (x - w) * (fx - fv)
            tmp2 = (x - v) * (fx - fw)
            p = (x - v) * tmp2 - (x - w) * tmp1
            tmp2 = 2.0 * (tmp2 - tmp1)
            if (tmp2 > 0.0):
                p = -p
            tmp2 = abs(tmp2)
            dx_temp = deltax
            deltax = rat
            # check parabolic fit
            if ((p > tmp2 * (a - x)) and (p < tmp2 * (b - x)) and
                    (abs(p) < abs(0.5 * tmp2 * dx_temp))):
                rat = p * 1.0 / tmp2        # if parabolic step is useful.
                u = x + rat
                if ((u - a) < tol2 or (b - u) < tol2):
                    if xmid - x >= 0:
                        rat = tol1
                    else:
                        rat = -tol1
            else:
                if (x >= xmid):
                    deltax = a - x  # if it's not do a golden section step
                else:
                    deltax = b - x
                rat = _cg * deltax

        if (abs(rat) < tol1):            # update by at least tol1
            if rat >= 0:
                u = x + tol1
            else:
                u = x - tol1
        else:
            u = x + rat
        fu = func.evaluate(u)      # calculate new output value

        if (fu > fx):                 # if it's bigger than current
            if (u < x):
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):
                v = w
                w = u
                fv = fw
                fw = fu
            elif (fu <= fv) or (v == x) or (v == w):
                v = u
                fv = fu
        else:
            if (u >= x):
                a = x
            else:
                b = x
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        iter += 1
    #################################
    #END CORE ALGORITHM
    #################################

    return x if iter < maxiter else INFINITY


cdef (double, double, double, double, double, double, int) bracket(Function func, double xa, double xb, double grow_limit, int maxiter) nogil:
    cdef double _gold = 1.618034  # golden ratio: (1.0+sqrt(5.0))/2.0
    cdef double _verysmall_num = 1e-21
    cdef double fa, fb, fc
    cdef double xc
    cdef double tmp1, tmp2, val
    cdef int iter
    cdef double denom, w, fw, wlim

    fa = func.evaluate(xa)
    fb = func.evaluate(xb)
    if (fa < fb):                      # Switch so fa > fb
        xa, xb = xb, xa
        fa, fb = fb, fa
    xc = xb + _gold * (xb - xa)
    fc = func.evaluate(xc)
    iter = 0
    while (fc < fb):
        tmp1 = (xb - xa) * (fb - fc)
        tmp2 = (xb - xc) * (fb - fa)
        val = tmp2 - tmp1
        if abs(val) < _verysmall_num:
            denom = 2.0 * _verysmall_num
        else:
            denom = 2.0 * val
        w = xb - ((xb - xc) * tmp2 - (xb - xa) * tmp1) / denom
        wlim = xb + grow_limit * (xc - xb)
        if iter > maxiter:
            iter = -1
            break
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func.evaluate(w)
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                break
            elif (fw > fb):
                xc = w
                fc = fw
                break
            w = xc + _gold * (xc - xb)
            fw = func.evaluate(w)
        elif (w - wlim)*(wlim - xc) >= 0.0:
            w = wlim
            fw = func.evaluate(w)
        elif (w - wlim)*(xc - w) > 0.0:
            fw = func.evaluate(w)
            if (fw < fc):
                xb = xc
                xc = w
                w = xc + _gold * (xc - xb)
                fb = fc
                fc = fw
                fw = func.evaluate(w)
        else:
            w = xc + _gold * (xc - xb)
            fw = func.evaluate(w)
        xa = xb
        xb = xc
        xc = w
        fa = fb
        fb = fc
        fc = fw
    return xa, xb, xc, fa, fb, fc, iter


# From scipy.optimize.brentq (Zeros/brentq.c)
cdef double brentq(Function func, double xa, double xb, double xtol, double rtol, int maxiter) nogil:
    cdef double xpre = xa, xcur = xb
    cdef double xblk = 0., fpre, fcur, fblk = 0., spre = 0., scur = 0., sbis

    # the tolerance is 2*delta
    cdef double delta
    cdef double stry, dpre, dblk
    cdef int iter

    fpre = func.evaluate(xpre)
    fcur = func.evaluate(xcur)
    if fpre*fcur > 0:
        return INFINITY
    if fpre == 0.:
        return xpre
    if fcur == 0.:
        return xcur

    iter = 0
    while iter < maxiter:
        if fpre*fcur < 0:
            xblk = xpre
            fblk = fpre
            spre = scur = xcur - xpre

        if abs(fblk) < abs(fcur):
            xpre = xcur
            xcur = xblk
            xblk = xpre

            fpre = fcur
            fcur = fblk
            fblk = fpre

        delta = (xtol + rtol*abs(xcur))/2
        sbis = (xblk - xcur)/2
        if fcur == 0 or abs(sbis) < delta:
            return xcur

        if abs(spre) > delta and abs(fcur) < abs(fpre):
            if xpre == xblk:
                # interpolate
                stry = -fcur*(xcur - xpre)/(fcur - fpre)
            else:
                # extrapolate
                dpre = (fpre - fcur)/(xpre - xcur)
                dblk = (fblk - fcur)/(xblk - xcur)
                stry = -fcur*(fblk*dblk - fpre*dpre) / (dblk*dpre*(fblk - fpre))
            if 2*abs(stry) < min(abs(spre), 3*abs(sbis) - delta):
                # good short step
                spre = scur
                scur = stry
            else:
                # bisect
                spre = sbis
                scur = sbis
        else:
            # bisect
            spre = sbis
            scur = sbis

        xpre = xcur
        fpre = fcur
        if abs(scur) > delta:
            xcur += scur
        else:
            xcur += delta if sbis > 0 else -delta

        fcur = func.evaluate(xcur)
        iter += 1

    return xcur
