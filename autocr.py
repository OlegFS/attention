import weakref
import inspect

from numpy import ndarray, array_equal

from autoreload import *
from autocacher import _MetaAutoCacher, AutoCacher, cached


class MetaAutoCR( _MetaAutoCacher, MetaAutoReloader ):

    def __init__( cls, name, bases, ns ):
        # prepare autocache
        _MetaAutoCacher.__init__( cls, name, bases, ns )

        # prepare autoreload
        cls.__instance_refs__ = []
        f = inspect.currentframe().f_back
        for d in [f.f_locals, f.f_globals]:
            if name in d:
                old_class = d[name]
                for instance in old_class.__instances__():
                    instance.__change_class__(cls)
                    cls.__instance_refs__.append(weakref.ref(instance))
                for subcls in old_class.__subclasses__():
                    newbases = []
                    for base in subcls.__bases__:
                        if base is old_class:
                            newbases.append(cls)
                        else:
                            newbases.append(base)
                    subcls.__bases__ = tuple(newbases)
                break


class AutoCR( AutoReloader, AutoCacher ):

    __metaclass__ = MetaAutoCR

    @property
    def _cache_summary( self ):
        return self.__summarise__( self._cache )

    @property
    def _dict_summary( self ):
        d = self.__dict__.copy()
        del d['_cache']
        del d['_proxies']
        return self.__summarise__( d )

    def __summarise__( self, d ):
        if len( d ) == 0:
            return 
        max_len_k = max([len(str(k)) for k in d.keys()])
        for k in sorted( d.keys() ):
            k_str = str(k)
            k_str += ' ' * (max_len_k - len(k_str)) + ' : '
            v = d[k]
            if isinstance(v, list):
                k_str += '(%d) list' % len(v)
            elif isinstance(v, tuple):
                k_str += '(%d) tuple' % len(v)
            elif isinstance(v, ndarray):
                k_str += str(v.shape) + ' array'
            elif isinstance(v, dict):
                k_str += '(%d) dict' % len(v)
            else:
                k_str += str(v)
            print '    ', k_str

    def plot_cache_graph( self, leaf=None, root=None, filename=None,
            display=True, display_cmd='eog' ):
        # imports
        import pydot
        import subprocess
        from numpy.random import random_integers
        # styles
        cached_shape = 'ellipse'
        ordinary_shape = 'box'
        present_penwidth = 3
        #fontname = 'helvetica-bold'
        fontname = 'DejaVuSans'
        styles = {}
        styles['cached absent'] = {
                'fontname':fontname,
                'shape':cached_shape, 
                'style':'filled', 
                'fillcolor':"#ffeeee"}
        styles['cached present'] = { 
                'fontname':fontname,
                'shape':cached_shape, 
                'style':'filled', 
                'fillcolor':"#ff9999", 
                'penwidth':present_penwidth }
        styles['ordinary absent'] = { 
                'fontname':fontname,
                'shape':ordinary_shape, 
                'style':'filled', 
                'fillcolor':"#eeeeff" }
        styles['ordinary present'] = { 
                'fontname':fontname,
                'shape':ordinary_shape, 
                'style':'filled', 
                'fillcolor':"#9999ff", 
                'penwidth':present_penwidth }
        styles['method'] = { 
                'fontname':fontname,
                'shape':'trapezium', 
                'style':'filled', 
                'margin':0,
                'fillcolor':"#aaccaa", 
                'penwidth':present_penwidth }

        g = pydot.Dot( graph_name='dependencies', rankdir='TB', ranksep=1 )
        g.set_node_defaults( fixedsize='false' )

        # useful
        cls = self.__class__

        # list of all the cachable attributes
        cachable_attrs = self._ancestors.keys()
        # dictionary of parents
        parents = { a : getattr( cls, a ).parents for a in cachable_attrs }
        conditional_parents = { a : getattr( cls, a ).conditional_parents
                for a in cachable_attrs }

        # list of all the attributes
        attrs = set(cachable_attrs)
        for a in cachable_attrs:
            # add its parents
            for p in parents[a]:
                attrs.add(p)

        # node type
        node_types = {}
        for a in attrs:
            if a in cachable_attrs:
                if self._cache.has_key(a):
                    node_types[a] = 'cached present'
                else:
                    node_types[a] = 'cached absent'
            elif ( hasattr( cls, a ) 
                    and getattr(cls, a).__class__.__name__ == (
                        'instancemethod') ):
                node_types[a] = 'method'
            else:
                if self.__dict__.has_key(a) or hasattr( cls, a ):
                    node_types[a] = 'ordinary present'
                else:
                    node_types[a] = 'ordinary absent'

        # restrict to attr of interest
        if leaf is not None:
            cachable_attrs = [a for a in cachable_attrs 
                    if a in self.ancestors( leaf ) or a == leaf]
            attrs = set(cachable_attrs)
            for a in cachable_attrs:
                # add its parents
                for p in parents[a]:
                    attrs.add(p)
        
        if root is not None:
            cachable_attrs = [a for a in cachable_attrs 
                    if a in self.descendants( root ) or a == root]
            attrs = set(cachable_attrs + [root])
            N_extra_parents = { a : len([p for p in v if p not in attrs])
                    for a, v in parents.items() }
            parents = { a : [p for p in v if p in attrs]
                for a, v in parents.items() }
            conditional_parents = { a : [p for p in v if p in attrs]
                for a, v in conditional_parents.items() }

        # create nodes
        nodes = {}
        for a in attrs:
            # label
            al = a
            if root is not None:
                if a in cachable_attrs: 
                    if N_extra_parents[a] > 0:
                        al = '%s  [+%d]' % (a, N_extra_parents[a])
            # create node
            nodes[a] = pydot.Node( name=a, label=al, **styles[node_types[a]] )
            g.add_node( nodes[a] )

        # create edges
        for a in cachable_attrs:
            # what edges have been temporarily silenced
            if len( conditional_parents[a] ) > 0:
                # get the conditional skips
                cskip = getattr( cls, a ).cskip
                # narrow down to known status
                cskip = [ cs for cs in cskip if node_types[cs[0]] in 
                        ['ordinary present', 'cached present'] ]
                # what should be skipped
                to_skip = [ cs[2] for cs in cskip 
                        if getattr( self, cs[0] ) == cs[1] ] + [[]]
                to_skip = reduce( lambda x, y : x+y, to_skip )
            else:
                to_skip = []
            # plot the edges
            for p in parents[a]:
                e = pydot.Edge( nodes[p], nodes[a] )
                if node_types[p] in [
                        'cached present', 'ordinary present', 'method']:
                    try:
                        e.set_penwidth(2)
                    except AttributeError:
                        pass
                else:
                    e.set_color('#888888')
                if p in to_skip:
                    e.set_style('dotted')
                elif p in conditional_parents[a]:
                    e.set_style('dashed')
                g.add_edge( e )

        # save file
        if filename is None:
            filename = '/tmp/cache.%d.png' % random_integers(1e12)

        g.write_png(filename, prog='dot')

        # display
        if display:
            process = subprocess.Popen(
                    "%s %s" % (display_cmd, filename), 
                    shell=True, stdout=subprocess.PIPE)
            return process

    def csetattr( self, attr, val ):
        """ Set the attr to the val, only if it is different. """
        f = self.cfunction( array_equal, attr, attr )
        f(val)

