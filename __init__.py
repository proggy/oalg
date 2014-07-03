#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2013-2014 Daniel Jung
# Contact: djungbremen@gmail.com
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the Free
# Software Foundation; either version 2 of the License, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
# more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA.
#
"""Collection of online algorithms. Most of them are online versions of their
offline counterparts in numpy, but because of their online nature, they are
formulated as classes instead of functions.

For further information about online algorithms, please check Wikipedia:
http://en.wikipedia.org/wiki/Online_algorithm

Usage is as follows:

1. In front of an (probably infinite) loop, instantiate the class.
2. Then, within each iteration, add some data to the class using the method
   *add()*.
3. The result of the calculation can be obtained at any point (after the loop
   has been exited, but also within the loop) by calling the instance as a
   function.

Most classes accept scalar values as well as n-dimensional arrays, as well as
many different data types. Check the documentation of the respective class
for further details."""
#
# To do:
# --> write Hist algorithm, an online version of numpy.histogram, which
#     counts up each bin iteratively as more data values come in.
#
# 2012-08-03 - 2013-12-18
# based on ialg, developed from 2012-04-20 until 2012-07-16

import numpy


class Mean(object):
    """Online algorithm to compute the sample arithmetic mean (average), plus
    the corresponding sample variance.

    To be specific, the algorithm calculates an estimator for the sample
    average and an estimator for the sample variance.

    After creating an instance of this class, add values to it using the method
    *add*. After at least one value has been added, the result can be obtained
    by calling the instance or the method *mean*. Sample variance, sample
    standard deviation and sample standard error can be obtained by calling
    the methods *var*, *std* and *stderr*.


    Working principle:

    Besides the sample count N, the class only has to remember two other values
    (scalar or nd-array):

    1. The sum of all added values, sigma.
    2. The sum of all squared values, sigma2.

    With this information, it is easy to calculate the arithmetic mean and
    the sample variance:

        mean = sigma/N

        var  = 1/(N-1)*sigma2-sigma**2/(N*(N-1))"""
    # 2012-08-03 - 2012-09-06

    def __init__(self, shape=None, dtype=None,
                 init_value=None, init_var=None, init_count=None):
        """Initialize the instance. The desired data shape and data type can be
        set, otherwise they are infered from the first added value, or from the
        given initial values. A previously stopped calculation can be continued
        by setting the three needed initial values:

        init_value : sample mean of the old calculation
        init_count : sample count of the old calculation
        init_var   : sample variance of the old calculation"""
        # 2012-08-03 - 2013-06-19

        # check arguments
        if ((init_count is not None and init_count > 0)
                + (init_value is not None)
                + (init_var is not None)) not in (0, 3):
            raise ValueError('if initial values are given, all three types ' +
                             'of information (count, value and variance) ' +
                             'must be given')

        # get initial values
        self._count = int(init_count or 0)
        self._sigma = numpy.array(self._count*numpy.array(init_value))\
            if self._count else None
        self._sigma2 = numpy.array((self._count - 1) * numpy.array(init_var)
                                   + self._sigma ** 2 / self._count) \
            if self._count else None
        init_shape = self._sigma.shape if hasattr(self._sigma, 'shape') \
            else None
        init_dtype = self._sigma.dtype if hasattr(self._sigma, 'dtype') \
            else None

        # get shape and data type
        self._shape = init_shape if shape is None else shape
        self._dtype = dtype or init_dtype

    @property
    def count(self):
        """Return the current sample count."""
        # 2012-08-03 - 2012-08-03
        return self._count

    @property
    def dtype(self):
        """Return data type."""
        # 2012-08-03 - 2012-08-03
        return self._dtype

    @property
    def shape(self):
        """Return data shape."""
        # 2012-08-03 - 2012-08-03
        return self._shape

    @property
    def size(self):
        """Return number of elements of the allowed values."""
        # 2012-08-03 - 2012-08-03
        return numpy.prod(self._shape) \
            if self._shape is not None else None

    @property
    def ndim(self):
        """Return number of dimensions of the allowed values."""
        # 2012-08-03 - 2012-08-03
        return len(self._shape) \
            if self._shape is not None else None

    def add(self, value):
        """Add a value. It can be a scalar or an n-dimensional array, but it
        must be consistent with values that have already been added."""
        # 2012-08-03 - 2012-08-03
        # based on ialg.MeanAlgorithm.add from 2012-02-17
        # and ialg.MeanAlgorithm.check_added from 2011-11-25

        # check the value, adopt shape and data type from it if needed
        if self._shape is None:
            if self._count == 0:
                self._shape = numpy.shape(value)
                self._sigma = numpy.zeros(self._shape, dtype=self._dtype)
                self._sigma2 = numpy.zeros(self._shape, dtype=self._dtype)
            else:
                raise ValueError('illegal value')
        elif self._shape == ():
            if hasattr(value, '__iter__'):
                raise ValueError('scalar value expected')
        else:
            if numpy.array(value).shape != self._shape:
                raise ValueError('illegal shape')
        if self._count == 0 and self._dtype is None:
            self._dtype = numpy.array(value).dtype\
                if hasattr(value, '__iter__') else type(value)
            self._sigma = self._sigma.astype(self._dtype)
            self._sigma2 = self._sigma2.astype(self._dtype)

        # add the new value
        self._sigma += numpy.array(value)
        self._sigma2 += numpy.array(value) ** 2
        self._count += 1

    def mean(self):
        """Calculate the sample arithmetic mean."""
        # 2012-08-03 - 2012-08-03
        if self._count:
            if self._shape == ():
                return (self._sigma / self._count).tolist()
            else:
                return self._sigma / self._count
        else:
            return None

    def __call__(self):
        """Alias for *mean()*."""
        # 2012-08-03 - 2012-08-03
        return self.mean()

    def var(self):
        """Calculate the sample variance."""
        # 2012-08-03 - 2012-08-03
        if self._count:
            if self._shape == ():
                if self._count > 1:
                    return (self._sigma2/(self._count - 1)
                            - self._sigma ** 2 / self._count
                            / (self._count - 1)).tolist()
                else:
                    return 0.
            else:
                if self._count > 1:
                    return self._sigma2 / (self._count - 1) \
                        - self._sigma ** 2 / self._count / (self._count - 1)
                else:
                    return numpy.zeros_like(self._sigma)
        else:
            return None

    def std(self):
        """Calculate the sample standard deviation, which is the square root of
        the sample variance."""
        # 2012-08-03 - 2012-09-06
        var = self.var()
        if var is None:
            return None
        return numpy.sqrt(var)

    def stderr(self):
        """Calculate the sample standard error of the mean, which is the sample
        standard deviation devided by sqrt(N)."""
        # 2012-09-06 - 2012-09-06
        var = self.var()
        if var is None or self._count < 1:
            return None
        return numpy.sqrt(var/self._count)

    def sem(self):
        """Alias for *stderr()*."""
        # 2012-09-06 - 2012-09-06
        return self.stderr()

    def mspair(self):
        """Return the pair of sample mean and sample standard error
        (2-tuple)."""
        # 2012-08-03 - 2013-06-19
        return self.mean(), self.stderr()

    def __repr__(self):
        """Return complete string representation."""
        # 2012-08-03 - 2012-08-03
        args = {}
        if self._shape is not None:
            args['shape'] = self._shape
        if self._dtype:
            args['dtype'] = self._dtype  # .__name__
        if self._count:
            args['init_count'] = self._count
            if self.ndim == 1:
                args['init_value'] = self.mean().tolist()
                args['init_var'] = self.var().tolist()
            else:
                args['init_value'] = self.mean()
                args['init_var'] = self.var()
        return '%s(%s)' \
            % (type(self).__name__, ', '.join('%s=%s'
                                              % (key, args[key])
                                              for key in args.keys()))

    def __str__(self):
        """Return short string representation."""
        # 2012-08-03 - 2012-08-03
        return '<%s instance with %i value%s (shape=%s, dtype=%s)>' \
            % (type(self).__name__, self._count, self._plural(self._count),
                self._shape, self._dtype)  # .__name__

    @staticmethod
    def _plural(number):
        """Returns an empty string is the given number equals 1, otherwise
        "s"."""
        # 2012-08-03 - 2012-08-03
        # copied from cofunc.CoFunc._plural from 2012-07-11
        return '' if number == 1 else 's'

    def ci(self, n=1):
        """Return confidence interval (CI) of order *n*. Two length-N arrays
        are returned, containing the value of the lower and the upper part of
        the CI, respectively, relative to the mean value. In other words, the
        CI stretches from mean-array1 to mean+array2. The width of the CI is
        array1+array2.
        """
        # 2012-10-09 - 2012-10-09
        assert n == 1, 'only n=1 supported right now'
        stderr = self.stderr()
        return stderr, stderr


class gMean(object):
    """Online algorithm to compute the sample geometric mean, plus the
    corresponding variance.

    To be specific, the algorithm calculates an estimator for the sample
    geometric mean and an estimator for the corresponding sample variance.

    After creating an instance of this class, add values to it using the method
    *add()*. After at least one value has been added, the result can be
    obtained by calling the instance or the method *gmean()*. Sample variance,
    sample standard deviation and sample standard error can be obtained by
    calling the methods *var()*, *std()* and *stderr()*.


    Working principle:

    Besides the sample count N, the class only has to remember two other values
    (scalar or nd-array):

    1. The sum of the logarithm of all added values, gamma.
    2. The sum of the squared logarithms of all added values, gamma2.

    With this information, the sample geometric mean and
    the sample variance can be calculated as follows:

        mean = exp(gamma/N)

        var  = exp(2*gamma/N)*(1/(N-1)*gamma2-gamma**2/(N*(N-1)))"""
    # 2012-08-03 - 2013-06-19

    def __init__(self, shape=None, dtype=None,
                 init_value=None, init_var=None, init_count=None):
        """Initialize the instance. The desired data shape and data type can be
        set, otherwise they are infered from the first added value, or from the
        given initial values. A previously stopped calculation can be continued
        by setting the three needed initial values:
        init_value : the result for the sample mean of the old calculation
        init_count : the sample count of the old calculation
        init_var   : sample variance of the old calculation"""
        # 2012-08-03 - 2013-06-19

        # check arguments
        if ((init_count is not None and init_count > 0)
                + (init_value is not None)
                + (init_var is not None)) not in (0, 3):
            raise ValueError('if initial values are given, all three types ' +
                             'of information (count, value, and variance ' +
                             'must be given')

        # get initial values
        self._count = int(init_count or 0)
        self._gamma = numpy.array(self._count
                                  * numpy.log(numpy.array(init_value))) \
            if self._count else None
        self._gamma2 = numpy.array((self._count - 1) * numpy.array(init_var)
                                   / numpy.exp(2 * self._gamma / self._count)
                                   + self._gamma ** 2 / self._count) \
            if self._count else None
        init_shape = self._gamma.shape if hasattr(self._gamma, 'shape') \
            else None
        init_dtype = self._gamma.dtype if hasattr(self._gamma, 'dtype') \
            else None

        # get shape and data type
        self._shape = init_shape if shape is None else shape
        self._dtype = dtype or init_dtype

    @property
    def count(self):
        """Return the current sample count."""
        # 2012-08-03 - 2012-08-03
        return self._count

    @property
    def dtype(self):
        """Return data type."""
        # 2012-08-03 - 2012-08-03
        return self._dtype

    @property
    def shape(self):
        """Return data shape."""
        # 2012-08-03 - 2012-08-03
        return self._shape

    @property
    def size(self):
        """Return number of elements of the allowed values."""
        # 2012-08-03 - 2012-08-03
        return numpy.prod(self._shape) \
            if self._shape is not None else None

    @property
    def ndim(self):
        """Return number of dimensions of the allowed values."""
        # 2012-08-03 - 2012-08-03
        return len(self._shape) \
            if self._shape is not None else None

    def add(self, value):
        """Add a value. It can be a scalar or an n-dimensional array, but it
        must be consistent with values that were already added."""
        # 2012-08-03 - 2012-08-03
        # based on ialg.MeanAlgorithm.add from 2012-02-17
        # and ialg.MeanAlgorithm.check_added from 2011-11-25

        # check if value is zero
        if numpy.any(numpy.array(value) <= 0):
            raise ValueError('unable to add non-positive values')

        # check the value, adopt shape and data type from it if needed
        if self._shape is None:
            if self._count == 0:
                self._shape = numpy.shape(value)
                self._gamma = numpy.zeros(self._shape, dtype=self._dtype)
                self._gamma2 = numpy.zeros(self._shape, dtype=self._dtype)
            else:
                raise ValueError('illegal value')
        elif self._shape == ():
            if hasattr(value, '__iter__'):
                raise ValueError('scalar value expected')
        else:
            if numpy.array(value).shape != self._shape:
                raise ValueError('illegal shape')
        if self._count == 0 and self._dtype is None:
            self._dtype = numpy.array(value).dtype \
                if hasattr(value, '__iter__') else type(value)
            self._gamma = self._gamma.astype(self._dtype)
            self._gamma2 = self._gamma2.astype(self._dtype)

        # add the new value
        self._gamma += numpy.log(numpy.array(value))
        self._gamma2 += numpy.log(numpy.array(value))**2
        self._count += 1

    def mean(self):
        """Calculate the sample geometric mean."""
        # 2012-08-03 - 2013-06-19
        if self._count:
            if self._shape == ():
                return numpy.exp(self._gamma/self._count).tolist()
            else:
                return numpy.exp(self._gamma/self._count)
        else:
            return None

    def __call__(self):
        """Alias for the method *mean()*."""
        # 2012-08-03 - 2012-08-03
        return self.mean()

    def var(self):
        """Calculate the sample variance.

        Note: Because a numerically unstable algorithm for the variance is
        used, sometimes the variance takes small negative values. To circumvent
        this, those values are set to zero.  Tests show that even with a
        million samples, the error is still of the order 1e-10."""
        # 2012-08-03 - 2013-06-19
        if self._count:
            if self._shape == ():
                if self._count > 1:
                    result = numpy.exp(2 * self._gamma / self._count) \
                        * (self._gamma2 / (self._count - 1)
                           - self._gamma ** 2 / self._count
                           / (self._count - 1))
                    return (result).tolist() \
                        if (result).tolist() > 0 else self._dtype(0)
                else:
                    return self._dtype(0)
            else:
                if self._count > 1:
                    result = numpy.exp(2 * self._gamma / self._count) \
                        * (self._gamma2 / (self._count - 1)
                           - self._gamma ** 2 / self._count
                           / (self._count - 1))
                    result[result < 0] = 0  # force non-negative results
                    return result
                else:
                    return numpy.zeros_like(self._gamma)
        else:
            return None

    def std(self):
        """Calculate the sample standard deviation, which is the square root
        of the sample variance."""
        # 2012-08-03 - 2013-06-19
        var = self.var()
        if var is None:
            return None
        return numpy.sqrt(var)

    def stderr(self):
        """Calculate the standard error of the geometric mean.

        According to [Norris1940], the sample standard error is equal to the
        sample standard deviation devided by sqrt(N-1)."""
        # 2012-09-06 - 2013-06-19
        var = self.var()
        if var is None or self._count < 1:
            return None
        #return numpy.exp(numpy.sqrt(var/self._count))
        return self.std() / numpy.sqrt(self._count - 1)

    def sem(self):
        """Alias for *stderr()*."""
        # 2012-09-06 - 2012-09-06
        return self.stderr()

    def mspair(self):
        """Return the pair of mean and standard error (2-tuple)."""
        # 2012-08-03 - 2013-06-19
        return self.mean(), self.stderr()

    def __repr__(self):
        """Return complete string representation."""
        # 2012-08-03 - 2012-08-03
        args = {}
        if self._shape is not None:
            args['shape'] = self._shape
        if self._dtype:
            args['dtype'] = self._dtype.__name__
        if self._count:
            args['init_count'] = self._count
            if self.ndim == 1:
                args['init_value'] = self.mean().tolist()
                args['init_var'] = self.var().tolist()
            else:
                args['init_value'] = self.mean()
                args['init_var'] = self.var()
        return '%s(%s)' \
            % (type(self).__name__, ', '.join('%s=%s'
                                              % (key, args[key])
                                              for key in args.keys()))

    def __str__(self):
        """Return short string representation."""
        # 2012-08-03 - 2012-08-03
        return '<%s instance with %i value%s (shape=%s, dtype=%s)>' \
            % (type(self).__name__, self._count, self._plural(self._count),
                self._shape, self._dtype)  # .__name__

    @staticmethod
    def _plural(number):
        """Returns an empty string is the given number equals 1, otherwise
        "s"."""
        # 2012-08-03 - 2012-08-03
        # copied from cofunc.CoFunc._plural from 2012-07-11
        return '' if number == 1 else 's'

    def ci(self, n=1):
        """Return confidence interval (CI) of order *n*. Two arrays with the
        same shape as the input data are returned, containing the value of the
        lower and the upper part of the CI for every array element,
        respectively, relative to the mean value. In other words, the CI
        stretches from mean-array1 to mean+array2. The width of the CI is
        array1+array2."""
        # 2012-10-09 - 2013-06-19
        assert n == 1, 'only n=1 supported right now'
        stderr = self.stderr()
        return stderr, stderr


#class Hist
