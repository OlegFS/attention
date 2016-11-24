"""Helpful data types.

PrettyFloat
- an extension of `float` so that it prints nicely within a dict.

Bunch
- an improvement on the easy_install Bunch, which ensures that copy()
    returns a Bunch too.

"""

from bunch import Bunch as SuperBunch
import pdb

class Bunch(SuperBunch):

    """Extends Bunch to ensure that copy() returns a Bunch too.

    Also inherits __repr__ from dict, so as to preserve IPython's pretty
    printing of dictionaries.

    """

    def copy(self):
        return self.__class__(super(Bunch, self).copy())

    __repr__ = dict.__repr__


class PrintableBunch(Bunch):

    """Like Bunch, but only prints keys. Useful for data exploration."""

    def __repr__(self):
        k = self.keys()
        k.sort()
        return k.__repr__().replace('[', '<').replace(']', '>')


PBunch = PrintableBunch 
