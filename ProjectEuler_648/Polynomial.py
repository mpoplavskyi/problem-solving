import numpy as np

MAX_OUTPUT = 50


def set_settings(**kwargs):
    if 'MAX_OUTPUT' in kwargs:
        Polynomial.MAX_OUTPUT = kwargs['MAX_OUTPUT']


class Polynomial:
    def __init__(self, coefficients, limit=None, modulo=None):
        self._limit = limit
        self._modulo = modulo
        if self.limit and self._modulo:
            self._coefficients = np.fmod(np.trim_zeros(coefficients, 'b')[:limit + 1], modulo)
        elif self.limit:
            self._coefficients = np.trim_zeros(coefficients, 'b')[:limit + 1]
        elif self.modulo:
            self._coefficients = np.fmod(np.trim_zeros(coefficients, 'b'), modulo)
        else:
            self._coefficients = np.trim_zeros(coefficients, 'b')

    @property
    def coefficients(self):
        return self._coefficients

    @property
    def limit(self):
        return self._limit

    @property
    def modulo(self):
        return self._modulo

    @property
    def degree(self):
        return len(self.coefficients) - 1

    def __repr__(self):
        s = ""
        terms_to_print = min(MAX_OUTPUT, np.count_nonzero(self.coefficients))
        ind = 0
        while terms_to_print > 0:
            if self.coefficients[ind] == 0:
                ind += 1
                continue
            else:
                terms_to_print -= 1
                if self.coefficients[ind] > 0:
                    s += '+'
                s += str(self.coefficients[ind])
                if ind > 0:
                    s += 'x**'+str(ind)
                ind += 1

        if np.count_nonzero(self.coefficients) > MAX_OUTPUT:
            s += '...'

        return s if s[0] != '+' else s[1:]

    def __add__(self, other):
        if self.degree < other.degree:
            return other + self

        _coefficients = self.coefficients
        if self.modulo:
            _coefficients[:other.degree + 1] = \
                np.fmod(_coefficients[:other.degree + 1] + other.coefficients, self.modulo)
        else:
            _coefficients[:other.degree + 1] = _coefficients[:other.degree + 1] + other.coefficients

        return Polynomial(_coefficients, self.limit, self.modulo)

    def __sub__(self, other):
        if self.degree < other.degree:
            return other + self

        _coefficients = self.coefficients
        if self.modulo:
            _coefficients[:other.degree + 1] = \
                np.fmod(_coefficients[:other.degree + 1] - other.coefficients, self.modulo)
        else:
            _coefficients[:other.degree + 1] = _coefficients[:other.degree + 1] - other.coefficients

        return Polynomial(_coefficients, self.limit, self.modulo)

    def __mul__(self, other):
        if isinstance(other, int):
            return Polynomial(self.coefficients*other, self.limit, self.modulo)

        if not isinstance(other, Polynomial):
            raise NotImplementedError('Polynomials can be multiplied by polynomials or numbers only')

        _deg = min(self.limit, self.degree+other.degree)
        _coefficients = np.zeros(_deg + 1, dtype=np.int64)

        for _i in range(self.degree + 1):
            if self.coefficients[_i] == 0:
                continue
            else:
                if self.modulo:
                    _coefficients[:other.degree+_i+1] =  \
                        np.fmod(
                        _coefficients[:other.degree+_i+1] +
                        np.pad(other.coefficients, (_i, 0), constant_values=(0, 0))[: _deg + 1] * self.coefficients[_i],
                        self.modulo)
                else:
                    _coefficients[:other.degree+_i+1] = _coefficients[:other.degree+_i+1] + \
                        np.pad(other.coefficients, (_i, 0), constant_values=(0, 0))[: _deg + 1] * self.coefficients[_i]

        return Polynomial(_coefficients, self.limit, self.modulo)

