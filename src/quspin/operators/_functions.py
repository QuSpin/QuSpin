class function(object):
    def __init__(self, f, args=()):
        self._f = f
        self._args = args

    def conjugate(self):
        return conjugate_function(function(self._f, self._args))

    def __str__(self):
        return self._f.__name__

    def __repr__(self):
        return self.__str__()

    def conj(self):
        return self.conjugate()

    def __eq__(self, other):
        if isinstance(other, function):
            return hash(self) == hash(other)
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __contains__(self, other):
        return self == other

    def _hash_list(self):
        return [hash((self._f, self._args))]

    def __hash__(self):
        hash_list = tuple(self._hash_list())
        return hash(hash_list)

    def __call__(self, *args):
        return self._f(*(args + self._args))

    def __mul__(self, other):
        if self == other:
            return pow_function(self, 2)
        elif other.__contains__(self):
            return other * self
        else:
            return mul_function(self, other)


class comm_binary_function(function):
    def __init__(self, function1, function2):
        if not isinstance(function1, function):
            raise ValueError("func1 must be function object.")

        if not isinstance(function2, function):
            raise ValueError("func2 must be function object.")

        self._function1 = function1
        self._function2 = function2

    def _hash_list(self):
        hash_list = self._function1._hash_list()
        hash_list.extend(self._function2._hash_list())
        hash_list.sort()
        return hash_list

    def __contains__(self, other):
        return (self._function1.__contains__(other)) or (
            self._function2.__contains__(other)
        )


class mul_function(comm_binary_function):
    def __init__(self, *args, **kwargs):
        comm_binary_function.__init__(self, *args, **kwargs)

    def __call__(self, *args):
        return self._function1(*args) * self._function2(*args)

    def __str__(self):
        return "({0} * {1})".format(
            self._function1.__str__(), self._function2.__str__()
        )

    def __mul__(self, other):
        if self._function1.__contains__(other):
            return mul_function(self._function1 * other, self._function2)
        elif self._function2.__contains__(other):
            return mul_function(self._function1, self._function2 * other)
        else:
            return mul_function(self, other)


class conjugate_function(function):
    def __init__(self, function1):
        if not isinstance(function1, function):
            raise ValueError("func1 must be function object.")

        self._function1 = function1

    def __call__(self, *args):
        return self._function1(*args).conjugate()

    def conjugate(self):
        return self._function1

    def __str__(self):
        return "conj({0})".format(self._function1.__str__())

    def _hash_list(self):
        return [hash((1j, self._function1._f, self._function1._args))]

    def __contains__(self, other):
        return self._function1.__contains__(other)


class pow_function(function):
    def __init__(self, function1, p):
        if not isinstance(function1, function):
            raise ValueError("func1 must be function object.")

        self._function1 = function1
        self._p = p

    def __call__(self, *args):
        return self._function1(*args) ** self._p

    def __str__(self):
        return "{0}^{1}".format(self._function1.__str__(), self._p)

    def _hash_list(self):
        return [hash((self._function1._f, self._function1._args, self._p))]

    def __mul__(self, other):
        if self._function1 == other:
            return pow_function(self._function1, self._p + 1)
        else:
            return mul_function(self, other)

    def __contains__(self, other):
        return self._function1.__contains__(other)
