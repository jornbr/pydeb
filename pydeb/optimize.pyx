cimport cython

cdef class Function:
    cdef double evaluate(Function self, double x):
        return 0

@cython.cdivision(True)
cdef (double, double, double, double, double, double) bracket(Function func, double xa, double xb, double grow_limit=110.0, int maxiter=1000):
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
            exit
        iter += 1
        if (w - xc) * (xb - w) > 0.0:
            fw = func.evaluate(w)
            if (fw < fc):
                xa = xb
                xb = w
                fa = fb
                fb = fw
                exit
            elif (fw > fb):
                xc = w
                fc = fw
                exit
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
    if iter == -1:
        raise RuntimeError("Too many iterations.")
    return xa, xb, xc, fa, fb, fc

@cython.cdivision(True)
cdef double optimize(Function func, double xa, double xb):
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
    xa, xb, xc, fa, fb, fc = bracket(func, xa, xb)
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

    return x