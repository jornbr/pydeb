# cython: language_level=3

cdef class Function:
    cdef double evaluate(Function self, double x) nogil

cdef double optimize(Function func, double xa, double xb) nogil
cdef double brentq(Function func, double xa, double xb, double xtol, double rtol, int maxiter) nogil