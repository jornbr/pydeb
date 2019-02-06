cdef class Function:
    cdef double evaluate(Function self, double x)

cdef double optimize(Function func, double xa, double xb)
