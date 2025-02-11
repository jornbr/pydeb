# cython: language_level=3

cdef class Function:
    cdef double evaluate(Function self, double x) noexcept nogil

cdef double optimize(Function func, double xa, double xb) noexcept nogil
cdef double brentq(Function func, double xa, double xb, double xtol, double rtol, int maxiter) noexcept nogil