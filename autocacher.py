class cached( object ):
    """Decorator creating a cached attribute.

    This decorator should be used on a function definition which appears in the definition of a
    class inheriting from 'AutoCacher'. Example:

    >>> class Objective( AutoCacher ):
    >>>    @cached
    >>>    def f( x, y ):
    >>>        return 2 * x + y

    >>> obj = Objective( x = 3, y = 4 )
    >>> print obj.f

    Whenever the x or y attributes change, the cached value of f will be discarded and f will be
    recomputed from the function definition as it is accessed.

    Note: All parameters of the function definition are taken as the names of attributes of the
    class instance. The function definition should therefore not receive a reference to 'self' as
    its first parameter.

    The decorator takes a number of optional keyword arguments:

    - `verbose` (False): if True, every time the attribute is recomputed, the instance method 
        '_caching_callback' is called. This does not happen when the attribute is retrieved from
        the cache.

    - `settable (False)`: if True, the attribute can be manually set using __setattr__ 
        (or via = ). If False, an AttributeError is raised on an attempted set.

    - `cskip`: under some conditions, certain arguments need not be used in evaluation
        of the attribute. This keyword provides a shortcut to prevent the retrieval of these 
        arguments under those conditions. `cskip` should be a list of triples
        `(arg1, value, arg2)` or `(arg1, value, arglist)`, where arguments are input as strings.
        If `arg1 == value`, then `arg2` (or the arguments in `arglist`) are not fetched, and 
        rather the function is evaluated with these latter arguments set to None. 

        Note that in the present implementation, if an argument appears in both the left
        hand side of a triple, and the right hand side of another triple, a ValueError is raised.

    """

    def __init__( self, func = None, verbose = False, settable = False, cskip = None ):
        self.verbose = verbose
        self.settable = settable

        # skipping: check arguments
        if cskip is None:
            cskip = []
        elif isinstance( cskip, tuple ):
            cskip = [cskip]
        elif isinstance( cskip, list ):
            pass
        else:
            raise ValueError('`cskip` should be a tuple or list of tuples')
        self.cskip = cskip if cskip is not None else []
        skip_arg1s = []
        skip_arg2s = []
        for i, cs in enumerate( self.cskip ):
            # cast the last element of each triple as a list
            if isinstance( cs[2], str ):
                self.cskip[i] = (cs[0], cs[1], [cs[2]])
            elif isinstance( cs[2], list ):
                pass
            else:
                raise TypeError('`%s` is not a string or list' % str(cs[2]))
            # accumulate arguments
            skip_arg1s += [cs[0]]
            skip_arg2s += cs[2]
        # check that no arguments appears on the left and right hand side of the skips
        invalid_args = set(skip_arg1s).intersection(skip_arg2s)
        if len(invalid_args) > 0:
            raise ValueError( '`%s` appears on both the LHS and RHS of a conditional skip' 
                    % invalid_args.pop() )

        # call
        if callable( func ):
            self.__call__( func )
        elif func is not None:
            raise TypeError, 'expected callable but got {} instead'.format( type( func ) )

    def __call__( self, func ):
        # imports
        from inspect import getargspec
        from functools import update_wrapper
        # get arguments of the function
        argspec = getargspec( func )
        # check remaining arguments
        if not ( argspec.varargs is None and
                 argspec.keywords is None and
                 argspec.defaults is None ):
            err_str = ( 'cannot handle default or variable arguments in cached ' + 
                'attribute dependencies' )
            raise NotImplementedError( err_str )

        # skipping
        parents = self.parents = argspec.args
        self.conditional_parents = set()
        # convert the triples into quadruples
        for i, cs in enumerate( self.cskip ):
            arg1, val, arglist = cs
            # check that all args in the cskip triple are valid
            args_to_check = arglist + [arg1]
            for a in args_to_check:
                if a not in parents:
                    raise ValueError('argument in cskip not a function argument: %s' % a)
            # add to conditional parents
            self.conditional_parents = self.conditional_parents.union(arglist)
            # indexes
            argidxs = [parents.index(a) for a in arglist]
            # save
            self.cskip[i] = (arg1, val, arglist, argidxs)
        # notes
        self.unconditional_parents = set([
                p for p in parents if p not in self.conditional_parents])
        self.check_for_skip = len(self.conditional_parents) > 0

        # finish wrapper
        self.func = func
        update_wrapper( self, func )
        return self

    def __get__( self, inst, cls ):
        if inst is None:
            return self
        # try get from cache, else compute
        try:
            return inst._cache[ self.__name__ ]
        except KeyError:
            # skip if required
            if self.check_for_skip:
                idxs_to_skip = [
                        cs[-1] for cs in self.cskip if getattr( inst, cs[0] ) == cs[1] ]
                if len( idxs_to_skip ) == 0:
                    idxs_to_skip += [[]]
                idxs_to_skip = reduce( lambda x, y: x+y, idxs_to_skip ) # flatten
                to_get = [True] * len(self.parents)
                for i in idxs_to_skip:
                    to_get[i] = False
                args = [ getattr( inst, a ) if q else None for q, a in zip( to_get, self.parents ) ]
            else:
                args = [ getattr( inst, a ) for a in self.parents ]
            # if we are skipping any arguments, set the others to None
            # evaluate the function
            if self.verbose:
                inst._caching_callback( name = self.__name__, arguments = args )
            value = self.func( *args )
            inst._cache[ self.__name__ ] = value
            return value

    def __set__( self, inst, value ):
        if self.settable:
            inst._cache[ self.__name__ ] = value
            inst._clear_descendants( self.__name__ )
        else:
            raise AttributeError, 'cached attribute {}.{} is not settable'.format(
                inst.__class__.__name__, self.__name__ )

    def __delete__( self, inst ):
        inst._cache.pop( self.__name__, None )
        inst._clear_descendants( self.__name__ )


class coupled( object ):

    def __init__( self, instname, attrname ):
        self.instname = instname
        self.attrname = attrname

    def __get__( self, inst, cls ):
        if inst is None:
            return self
        return getattr( getattr( inst, self.instname ), self.attrname )


class coupling( object ):

    def __get__( self, inst, cls ):
        if inst is None:
            return self
        return self.other

    def __set__( self, inst, other ):
        self.other = other
        cls = type( inst )
        for n in dir( cls ):
            m = getattr( cls, n )
            try:
                if getattr( cls, m.instname ) is not self:
                    continue
            except AttributeError:
                continue
            proxylist = other._proxies.get( m.attrname, [] )
            proxylist.append( ( inst, n ) )
            other._proxies[ m.attrname ] = proxylist


class _MetaAutoCacher( type ):

    def __init__( self, clsname, bases, dct ):
        super( _MetaAutoCacher, self ).__init__( clsname, bases, dct )

        # find all class members that are cached methods, store their parents
        # TODO: is it better to check for subclasses of 'cached' here?
        # there might be name conflicts with other class members that have a 'parents' attribute.
        # otoh, it is considered bad style to rely on explicit inheritance
        parents = {}
        for n in dir( self ):
            m = getattr( self, n )
            try:
                parents[ m.__name__ ] = m.parents
            except AttributeError:
                pass

        # collect descendants for each attribute that at least one cached attribute depends upon
        descendants = {}
        computing = set()
        def collect_descendants( name ):
            try:
                return descendants[ name ]
            except KeyError:
                if name in computing:
                    raise ValueError, 'circular dependency in cached attribute {}'.format( name )
                # direct descendants
                direct = [ c for c, p in parents.iteritems() if name in p ]
                des = set( direct )
                # update set with recursive descendants
                computing.add( name )
                for d in direct:
                    des.update( collect_descendants( d ) )
                computing.remove( name )
                descendants[ name ] = tuple( des )
                return des
        for n in set( p for par in parents.itervalues() for p in par ):
            collect_descendants( n )

        # collect ancestors for each cached attribute
        ancestors = {}
        def collect_ancestors( name ):
            try:
                return ancestors[ name ]
            except KeyError:
                # direct ancestors
                try:
                    direct = parents[ name ]
                except KeyError:
                    return ()
                anc = set( direct )
                # update set with recursive ancestors
                for d in direct:
                    anc.update( collect_ancestors( d ) )
                ancestors[ name ] = tuple( anc )
                return anc
        for n in parents.iterkeys():
            collect_ancestors( n )

        self._descendants = descendants
        self._ancestors = ancestors


class AutoCacher( object ):
    """Class with cached attributes.

    A class inheriting from 'AutoCacher' maintains a cache of attributes defined through the
    decorator 'cached', and keeps track of their dependencies in order to keep the cache up-to-date.

    Whenever a cached attribute is accessed for the first time, its value is computed according
    to its function definition, and then stored in the cache. On subsequent accesses, the cached
    value is returned, unless any of its dependencies (the parameters of the function definition)
    have changed. Example:

    class Objective( AutoCacher ):

       @cached
       def f( x, y ):
           return 2 * x + y

       @cached
       def g( f, y, z ):
           return f / y + z

    obj = Objective( x = 3, y = 4 )

    # this evaluates f and caches the result
    print obj.f
    obj.z = 3
    # this evaluates g using cached f
    print obj.g
    # this clears the cached values of f and g
    obj.x = 8
    # this reevaluates f and g
    print obj.g
    """

    __metaclass__ = _MetaAutoCacher

    def _clear_proxies( self, name ):
        for inst, attr in self._proxies.get( name, () ):
            inst._clear_descendants( attr )

    def _clear_descendants( self, name ):
        self._clear_proxies( name )
        for d in self._descendants.get( name, () ):
            self._cache.pop( d, None )
            self._clear_proxies( d )

    def _update( self, **kwargs ):
        for k, v in kwargs.iteritems():
            setattr( self, k, v )

    def __new__( cls, *args, **kwargs ):
        new = super( AutoCacher, cls ).__new__( cls, *args, **kwargs )
        super( AutoCacher, new ).__setattr__( '_cache', {} )
        super( AutoCacher, new ).__setattr__( '_proxies', {} )
        return new

    def __init__( self, **kwargs ):
        self._update( **kwargs )

    def __setattr__( self, name, value ):
        super( AutoCacher, self ).__setattr__( name, value )
        self._clear_descendants( name )

    def __delattr__( self, name ):
        super( AutoCacher, self ).__delattr__( name )
        self._clear_descendants( name )

    def __getstate__( self ):
        state = self.__dict__.copy()
        state.update( _cache = {} )
        return state

    class _Function( object ):

        def __init__( self, inst, output, *inputs ):
            self.cacher = inst
            self.output = output
            self.inputs = inputs

        def __call__( self, *inputs, **kwargs ):
            args = { k: v for k, v in zip( self.inputs, inputs ) }
            args.update( kwargs )
            self.cacher._update( **args )
            return getattr( self.cacher, self.output )

    def function( self, output, *inputs ):
        """Return a convenience function with defined input and outputs attribute names."""
        return AutoCacher._Function( self, output, *inputs )

    class _CFunction( object ):

        def __init__( self, inst, equal, output, *inputs ):
            self.cacher = inst
            self.equal = equal
            self.output = output
            self.inputs = inputs

        def __call__( self, *inputs, **kwargs ):
            args = { k: v for k, v in zip( self.inputs, inputs ) }
            args.update( kwargs )
            changed = {}
            for k, v in args.iteritems():
                try:
                    curval = getattr( self.cacher, k )
                except AttributeError:
                    changed[ k ] = v
                    continue
                if not self.equal( curval, v ):
                    changed[ k ] = v
            self.cacher._update( **changed )
            return getattr( self.cacher, self.output )

    def cfunction( self, equal, output, *inputs ):
        """Return a comparing convenience function with defined input and outputs attribute names."""
        return AutoCacher._CFunction( self, equal, output, *inputs )

    def clear_cache( self ):
        """Clear the cache: all cached attributes will need to be recomputed on access."""
        self._cache.clear()

    def descendants( self, name ):
        return self._descendants[ name ]

    def ancestors( self, name ):
        return self._ancestors[ name ]



""" Example of skipping """

"""
class X( AutoCacher ):

    @cached( cskip = [ ('a', True, 'b'), ('c', 2, ['d', 'e']) ] )
    def z(a, b, c, d, e, f, g, h, i):
        return (a, b, c, d, e, f, g, h, i)

x = X()
x.a = True
x.b = 'ooga'
x.c = 2
x.d = 'booga'
x.e = 'awooga'
x.f = False
x.g = 'eep'
x.h = True
x.i = 'meep'

print x.z
x.c = 3
print x.z
"""
