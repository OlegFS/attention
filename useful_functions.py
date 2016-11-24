import warnings
import os
import time
import sys
import numpy as np
from shutil import rmtree
import progress
#/from matplotlib import pyplot as plt

"""
=================
File manipulation
=================
"""

def ls_full(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

def rmdir(directory):
    try:
        rmtree(directory)
    except OSError:
        pass

def mkpath(d):
    """Make directory and all parent directories."""
    try:
        os.makedirs(d)
    except OSError:
        pass


"""
================
Timing, progress
================
"""

def time_elapsed(initial_time):
    """Prints the time elapsed since an initial time."""
    dt = time.gmtime(time.time() - initial_time)
    if dt.tm_hour > 0:
        str = '%dh%dm%ds' % (dt.tm_hour, dt.tm_min, dt.tm_sec)
    elif dt.tm_min > 0:
        str = '%dm%ds' % (dt.tm_min, dt.tm_sec)
    else:
        str = '%ds' % (dt.tm_sec)
    return str

def time_this(func):
    def new_func(*a, **kw):
        t0 = time.time()
        output = func(*a, **kw)
        print "function '%s' took [%s]" % (func.__name__, time_elapsed(t0))
        return output
    return new_func

progress_dots = progress.dots
progress_numbers = progress.numbers
progress_names = progress_numbers

"""
========
Printing
========
"""

def simple_decorator(decorator):
    """This decorator can be used to turn simple functions
    into well-behaved decorators, so long as the decorators
    are fairly simple. If a decorator expects a function and
    returns a function (no descriptors), and if it doesn't
    modify function attributes or docstring, then it is
    eligible to use this. Simply apply @simple_decorator to
    your decorator and it will automatically preserve the
    docstring and function attributes of functions to which
    it is applied."""
    def new_decorator(f):
        g = decorator(f)
        g.__name__ = f.__name__
        g.__doc__ = f.__doc__
        g.__dict__.update(f.__dict__)
        return g
    # Now a few lines needed to make simple_decorator itself
    # be a well-behaved decorator.
    new_decorator.__name__ = decorator.__name__
    new_decorator.__doc__ = decorator.__doc__
    new_decorator.__dict__.update(decorator.__dict__)
    return new_decorator

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function %s." % func.__name__,
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    new_func.__name__ = func.__name__
    new_func.__doc__ = func.__doc__
    new_func.__dict__.update(func.__dict__)
    return new_func

def print_title(s, pre=True, post=True, title_char='='):
    """Prints the string `s` as a title."""
    n = len(s) + 2
    if pre:
        print ' '
    print title_char * n
    print ' ' + s + ' '
    print title_char * n
    if post:
        print ' '
    sys.stdout.flush()

def print_subtitle(s, **kw):
    """Prints the string `s` as a subtitle."""
    print_title(s, title_char='-', **kw)


"""
================
Useful functions
================
"""

def add_dict(d1, d2):
    """Returns the concatenation of two dicts.

    In particular, it makes a copy of d1, and updates with d2.

    """
    d3 = d1.copy()
    d3.update(d2)
    return d3

def is_even(x):
    """Returns true iff x is even."""
    return (x // 2) * 2 == x

def is_odd(x):
    """Returns true iff x is odd."""
    return (x // 2) * 2 + 1 == x

def nanmean(x, **kw):
    return np.nansum(x, **kw) / np.nansum(x != np.nan, **kw)

