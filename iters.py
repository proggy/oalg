#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright notice
# ----------------
#
# Copyright (C) 2014 Daniel Jung
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
"""Functions for working with iterables."""
# 2014-10-31 - 2014-10-31


import progress
import oalg


def imean(iterable, verbose=False):
    """Incrementally calculate the arithmetic mean of the values of the given
    iterable.  The values may be scalars or array-likes of any shape. If
    *verbose* is *True*, show a progress bar.

    Parameters
    ----------
    iterable : iterable with scalar or array-like elements
        The values to average over. Multidimensional objects are possible.
    verbose : bool
        If *True*, show a progress bar.

    Returns
    -------
    mean : scalar or array-like
        Arithmetic mean of the input values. Will have the same shape as the
        input values.
    stderr : scalar or array-like
        The associated standard error of the mean. Will have the same shape as
        the input values.

    Notes
    -----
    This function makes most sense if the input is not a static list or array
    of values to average over, but an iterable which loads or generates the
    values "on the fly". Otherwise, conventional methods for averaging should
    be used."""
    m = oalg.Mean()
    with progress.Bar(len(iterable), verbose=verbose) as b:
        for item in iterable:
            m.add(item)
            b.step()
    return m.mean(), m.stderr()


def igmean(iterable, verbose=False):
    """Incrementally calculate the geometric mean of the values of the given
    iterable.  The values may be scalars or array-likes of any shape. If
    *verbose* is *True*, show a progress bar.

    Parameters
    ----------
    iterable : iterable with scalar or array-like elements
        The values to average over. Multidimensional objects are possible.
    verbose : bool
        If *True*, show a progress bar.

    Returns
    -------
    gmean : scalar or array-like
        Arithmetic mean of the input values. Will have the same shape as the
        input values.
    stderr : scalar or array-like
        The associated standard error of the geometric mean. Will have the same
        shape as the input values.

    Notes
    -----
    This function makes most sense if the input is not a static list or array
    of values to average over, but an iterable which loads or generates the
    values "on the fly". Otherwise, conventional methods for averaging should
    be used."""
    m = oalg.gMean()
    with progress.Bar(len(iterable), verbose=verbose) as b:
        for item in iterable:
            m.add(item)
            b.step()
    return m.gmean(), m.stderr()
