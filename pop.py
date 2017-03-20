from common import *

import mop

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
DATA STRUCTURE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class PopData( mop.Data ):

    def __init__( self, X, Y, **kw ):
        # initialise as normal
        super( PopData, self ).__init__( X, Y[:, 0], **kw )
        # replace y
        del self.y
        self.Y = Y
        # size
        assert self.T == shape(Y)[0]
        self.M = shape(Y)[1]

    def plot_h( self, *a, **kw ):
        raise NotImplementedError()

    def plot_S( self, *a, **kw ):
        raise NotImplementedError()

    def plot_W( self, *a, **kw ):
        raise NotImplementedError()

    def plot_H( self, S, W, H=None, draw=True, new_figure=True, 
            return_fig=False, title=None, figsize=(12, 8), exponentiate=False ):
        # parse input
        if H is None:
            H = dot(S, W.T)
        p = S.shape[1]
        # create figure
        if new_figure:
            fig = plt.figure( figsize=figsize )
        else:
            fig = plt.gcf()
        # spatial pattern of w^{(i)}
        ax = axw = fig.add_axes([ 0.1, 0.4, 0.2, 0.55 ])
        yy = np.arange( self.M )
        ax.plot( [0, 0], [-1, self.M], 'k-', lw=1 )
        for i in range( p ):
            xx = w_i = W[:, i]
            ax.plot( xx, yy, 'o-', lw=3 )
        # aesthetics
        ax.set_xlabel(r'$w_m^{(i)}$', fontsize=22)
        ax.set_ylim(-1, self.M )
        ax.set_ylabel('unit #', fontsize=16)
        xl = ax.get_xlim()
        ax.set_xlim( xl[1], xl[0] )
        xts = ax.get_xticks()
        xts = [xt for xt in xts if xl[0] <= xt <= xl[1]]
        ax.set_xticks(xts)
        xts = ax.get_xticks().astype(object)
        xts[1:-1] = ' '
        ax.set_xticklabels(xts)
        # time course of s^{(i)}
        ax = axs = fig.add_axes([ 0.4, 0.1, 0.55, 0.2 ])
        xx = np.arange( self.T )
        if exponentiate:
            ax.plot( [-1, self.T], [1, 1], 'k-', lw=1 )
        else:
            ax.plot( [-1, self.T], [0, 0], 'k-', lw=1 )
        for i in range( p ):
            yy = s_i = S[:, i]
            if exponentiate:
                yy = np.exp(yy)
            ax.plot( xx, yy, '-', lw=2 )
        # aesthetics
        ax.set_xlim(0, self.T-1)
        ax.set_xlabel('time bin', fontsize=16)
        if exponentiate:
            ax.set_ylabel(r'$\exp(s_t^{(i)})$', fontsize=22)
        else:
            ax.set_ylabel(r'$s_t^{(i)}$', fontsize=22)
        ax.set_xticks([0, self.T])
        yl = ax.get_ylim()
        yts = ax.get_yticks()
        yts = [yt for yt in yts if yl[0] <= yt <= yl[1]]
        ax.set_yticks(yts)
        yts = ax.get_yticks().astype(object)
        yts[1:-1] = ' '
        ax.set_yticklabels(yts)
        # image plot
        ax = axi = fig.add_axes([ 0.4, 0.4, 0.55, 0.55 ])
        im = exp(H.T) if exponentiate else H.T
        ax.imshow( im, cmap=plt.cm.Greys, alpha=1,
                interpolation='nearest', aspect='auto')
        ax.set_xticks([0, self.T])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # finish
        if title is None:
            title = ' '
        ax.set_title(title, fontsize=20)
        if draw:
            plt.draw()
            plt.show()
        if return_fig:
            return fig

    def plot_y( self, *a, **kw ):
        raise NotImplementedError()

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
POP SOLVERS
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    - Superclass
    - Subclassed here as PopSimple, PopPositive

"""

class PopSolver( mop.Solver ):

    # defaults
    _hyperparameter_names_s = []
    _hyperparameter_names_w = []
    _hyperparameter_names_k = []

    # for dimensionality reduction
    _cutoff_lambda_s = 1e-9
    _cutoff_lambda_w = 1e-12
    _cutoff_lambda_k = 1e-12

    _default_theta_s0 = None
    _default_theta_w0 = None
    _default_theta_k0 = None

    _variable_names = ['k', 's', 'w']

    def __init__( self, data, rank=3, known_G=None, known_H=None, **kw ):
        self.rank = rank
        # parse known G and known H
        if known_G is None and known_H is not None:
            known_G = exp( known_H )
        if known_G is not None and known_H is None:
            known_H = log( known_G )
        self.known_G = known_G
        self.known_H = known_H
        # access to dim reduction methods
        self._reducer = mop.UnitSolver( data=None, empty_copy=True )
        self._reducer.parent = weakref.proxy( self )
        super( PopSolver, self ).__init__( data, **kw )

    def _create_posterior( self, **kw ):
        """ Create a posterior object of the same class as self. 

        This will place a copy of the current cache in the posterior, and
        set the flag `_is_posterior` to be `True`. Any keywords provided will
        also be set as attributes; this will happen *first* so that it
        does not wipe the cache.
        
        The posterior object will contain weak references to the parent.
        
        """
        # create weak references
        data = weakref.proxy( self.data )
        announcer = weakref.proxy( self._announcer )
        parent = weakref.proxy( self )
        # create object
        p = self.__class__( data=None, empty_copy=True, announcer=announcer,
                rank=self.rank, known_G=self.known_G, known_H=self.known_H )
        # label it as a posterior object
        p._is_posterior = True
        p.parent = parent
        # note the training
        p.initial_conditions = self.initial_conditions
        p.training_slices = self.training_slices
        p.testing_slices = self.testing_slices
        # set any theta attributes first
        for k, v in kw.items():
            if k.startswith('theta_'):
                setattr( p, k, v )
        # set any remaining attributes
        for k, v in kw.items():
            if not k.startswith('theta_'):
                setattr( p, k, v )
        # copy the current cache
        p._cache = self._cache.copy()
        return p
        
    """ Useful properties """

    @property
    def N_theta_s( self ):
        """ Number of hyperparameters for `s`. """
        return len( self._hyperparameter_names_s )

    @property
    def N_theta_w( self ):
        """ Number of hyperparameters for `w`. """
        return len( self._hyperparameter_names_w )

    @property
    def N_theta_k( self ):
        """ Number of hyperparameters for `k`. """
        return len( self._hyperparameter_names_k )

    @property
    def M( self ):
        """ Number of neurons """
        return self.data.M

    @cached
    def Y_training( data, _slice_by_training ):
        return _slice_by_training( data.Y )

    @property
    def N_observations( self ):
        return self.T_training * self.M

    @cached
    def abs_freqs_s( T, pad_size ):
        return mop.calc_abs_freqs( T, pad_size )

    # currently relying only on point estimates
    is_point_estimate = True

    """
    ==================
    Initial conditions
    ==================
    """

    @property
    def _grid_search_theta_s_available( self ):
        return hasattr( self, '_grid_search_theta_s_parameters' )
    
    @property
    def _grid_search_theta_w_available( self ):
        return hasattr( self, '_grid_search_theta_w_parameters' )
    
    @property
    def _grid_search_theta_k_available( self ):
        return hasattr( self, '_grid_search_theta_k_parameters' )

    def _parse_initial_conditions( self, initial_conditions ):
        """ Sets up the initial conditions for the model. """
        # begin with a blank slate
        self.initial_conditions = ics = Bunch()
        # run through each variable
        for v in ['s', 'w', 'k']:
            parse_func = getattr( self, '_parse_initial_conditions_' + v )
            ics.update( **parse_func( initial_conditions ) )

    def _parse_initial_conditions_w( self, ic_in ):
        # global defaults
        ics = Bunch()
        ics['w'] = zeros( self.M * self.rank )
        ics['theta_w'] = self._default_theta_w0
        # parse initial conditions: None
        if ic_in == None:
            pass
        # parse initial_conditions: dict
        elif isinstance( ic_in, dict ):
            # pull out what's in the dict
            for k in ics.keys():
                ics[k] = ic_in.get( k, ics[k] )
            # if W is provided
            if ic_in.has_key('W'):
                W = ic_in['W']
                if not W.shape == ( self.M, self.rank ):
                    raise ValueError('initial `W` is the wrong shape.')
                if ic_in.has_key('w'):
                    if not np.array_equal( W.flatten(), ic_in['w'] ): 
                        raise ValueError('provided conflicting `W`/`w`')
                ics['w'] = W.flatten()
        # parse initial_conditions: Solver object
        else: 
            # copy posterior on `w`
            if hasattr( ic_in, 'posterior_w' ):
                ics['w'] = ic_in.posterior_w.w
            # recast values of `theta`
            try:
                ics['theta_w'] = self._recast_theta_w( ic_in )
            except TypeError:
                pass
        # replace any invalid values
        for i in range( self.N_theta_w ):
            if (ics['theta_w'][i] is None) or (ics['theta_w'][i] == np.nan):
                ics['theta_w'] = self._default_theta_w0[i]
        # return
        return ics

    def _parse_initial_conditions_k( self, ic_in ):
        # global defaults
        ics = Bunch()
        ics['k'] = zeros( self.D * self.M )
        ics['theta_k'] = self._default_theta_k0
        # parse initial conditions: None
        if ic_in == None:
            pass
        # parse initial_conditions: dict
        elif isinstance( ic_in, dict ):
            # pull out what's in the dict
            for k in ics.keys():
                ics[k] = ic_in.get( k, ics[k] )
            # if K is provided
            if ic_in.has_key('K'):
                K = ic_in['K']
                if not K.shape == ( self.D, self.M ):
                    raise ValueError('initial `K` is the wrong shape.')
                if ic_in.has_key('k'):
                    if not np.array_equal( K.flatten(), ic_in['k'] ): 
                        raise ValueError('provided conflicting `K`/`k`')
                ics['k'] = K.flatten()
        # parse initial_conditions: Solver object
        else: 
            # copy posterior on `k`
            if hasattr( ic_in, 'posterior_k' ):
                ics['k'] = ic_in.posterior_k.k
            # recast values of `theta`
            try:
                ics['theta_k'] = self._recast_theta_k( ic_in )
            except TypeError:
                pass
        # replace any invalid values
        for i in range( self.N_theta_k ):
            if (ics['theta_k'][i] is None) or (ics['theta_k'][i] == np.nan):
                ics['theta_k'] = self._default_theta_k0[i]
        # return
        return ics

    @property
    def _Prior_s_class( self ):
        """ Returns the superclass that defines the `Ls` or `Cs` method. """
        return self._Prior_v_class( 's' )

    @property
    def _Prior_w_class( self ):
        """ Returns the superclass that defines the `Lw` or `Cw` method. """
        return self._Prior_v_class( 'w' )

    @property
    def _Prior_k_class( self ):
        """ Returns the superclass that defines the `Lk` or `Ck` method. """
        return self._Prior_v_class( 'k' )

    def _recast_theta_s( self, ic ):
        """ Process results of previous solver to determine initial theta_s. """
        return self._recast_theta_v( 's', ic )

    def _recast_theta_w( self, ic ):
        """ Process results of previous solver to determine initial theta_w. """
        return self._recast_theta_v( 'w', ic )

    def _recast_theta_k( self, ic ):
        """ Process results of previous solver to determine initial theta_k. """
        return self._recast_theta_v( 'k', ic )

    def _initialise_theta_s( self ):
        """ If `theta_s` is not set, initialise it. """
        return self._initialise_theta_v( 's' )

    def _initialise_theta_w( self ):
        """ If `theta_w` is not set, initialise it. """
        return self._initialise_theta_v( 'w' )

    def _initialise_theta_k( self ):
        """ If `theta_k` is not set, initialise it. """
        return self._initialise_theta_v( 'k' )

    @property
    def _most_recent_c( self ):
        # get the last estimate
        ph = [ h for h in self._posterior_history if h in ['sw', 's', 'w'] ]
        v = ph[-1]
        p = getattr( self, 'posterior_'+v )
        c = getattr( p, 'c__'+v )
        return c

    def _initialise_s_vec( self ):
        """ If `s_vec` is not set (validly), initialise it. """
        # check current value
        if hasattr( self, 's_vec' ):
            if len( self.s_vec ) == self._required_s_vec_length:
                return
            else:
                del self.s_vec
        # source
        try:
            source = self.posterior_s
        except AttributeError:
            source = self.initial_conditions
        # reproject
        s_vec = self._reproject_to_s_vec( posterior=source )
        # get the last c value
        if len( self._posterior_history ) > 0:
            c = self._most_recent_c
            s_vec[ -self.M: ] = c
        # set it
        self.s_vec = s_vec

    def _initialise_w_vec( self ):
        """ If `w_vec` is not set (validly), initialise it. """
        # check current value
        if hasattr( self, 'w_vec' ):
            if len( self.w_vec ) == self._required_w_vec_length:
                return
            else:
                del self.w_vec
        # source
        try:
            source = self.posterior_w
        except AttributeError:
            source = self.initial_conditions
        # reproject
        w_vec = self._reproject_to_w_vec( posterior=source )
        # get the last c value
        if len( self._posterior_history ) > 0:
            c = self._most_recent_c
            w_vec[ -self.M: ] = c
        # set it
        self.w_vec = w_vec

    def _initialise_k_vec( self ):
        """ If `k_vec` is not set (validly), initialise it. """
        return self._initialise_v_vec( 'k' )

    def _reset_theta_s( self ):
        """ Force reset of `theta_s`. """
        self._reset_theta_v( 's' )

    def _reset_theta_w( self ):
        """ Force reset of `theta_w`. """
        self._reset_theta_v( 'w' )

    def _reset_theta_k( self ):
        """ Force reset of `theta_k`. """
        self._reset_theta_v( 'k' )

    def _reset_s_vec( self ):
        """ Force reset of `s_vec`. """
        self._reset_v_vec( 's' )

    def _reset_w_vec( self ):
        """ Force reset of `w_vec`. """
        self._reset_v_vec( 'w' )

    def _reset_k_vec( self ):
        """ Force reset of `k_vec`. """
        self._reset_v_vec( 'k' )

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    ftol_per_obs = 1e-5

    """ Solving: individual """

    def calc_posterior_s( self, **kw ):
        # degenerate case
        if self.T_star < self.rank:
            self._calc_degenerate_posterior_s()
        # normal case
        else:
            self._calc_posterior_v( 's', **kw )

    def calc_posterior_w( self, **kw ):
        # degenerate case
        if self.M_star < self.rank:
            self._calc_degenerate_posterior_w()
        # normal case
        else:
            self._calc_posterior_v( 'w', **kw )

    def calc_posterior_k( self, *a, **kw ):
        raise NotImplementedError('non-factorial inference on `k` not yet ' +
                'implemented. Use `Factorial_Prior_k` subclass for now.')

    def get_next_theta_s_n( self, **kw ):
        return self._get_next_theta_v_n( 's', **kw )

    def get_next_theta_w_n( self, **kw ):
        return self._get_next_theta_v_n( 'w', **kw )

    def solve_theta_s( self, **kw ):
        return self._solve_theta_v( 's', **kw )

    def solve_theta_w( self, **kw ):
        return self._solve_theta_v( 'w', **kw )

    def solve_theta_k( self, *a, **kw ):
        raise NotImplementedError('non-factorial inference on `theta_k` not '
                + 'yet implemented. Use `Factorial_Prior_K` subclass for now.')

    def _calc_degenerate_posterior_s( self ):
        # NCR: constant term should not be zero...?
        s_vec = zeros( self._required_s_vec_length )
        s_vec[ -self.M: ] = self.c__s
        self._create_and_save_posterior_v( 's', s_vec )

    def _calc_degenerate_posterior_w( self ):
        # NCR: constant term should not be zero...?
        w_vec = zeros( self._required_w_vec_length )
        w_vec[ -self.M: ] = self.c__w
        self._create_and_save_posterior_v( 'w', w_vec )

    """ Derivative checking """

    def _check_LP_derivatives__s( self, **kw ):
        return self._check_LP_derivatives__v( 's', **kw )

    def _check_LP_derivatives__w( self, **kw ):
        return self._check_LP_derivatives__v( 'w', **kw )

    def _check_LE_derivatives__s( self, **kw ):
        return self._check_LE_derivatives__v( 's', **kw )

    def _check_LE_derivatives__w( self, **kw ):
        return self._check_LE_derivatives__v( 'w', **kw )

    """ Plotting """

    def plot_H_i( self, i, posterior=None, draw=True, new_figure=True, 
            return_fig=False, title=None, figsize=(12, 8), col='b' ):
        # parse input
        if posterior is None:
            posterior = self.posterior
        # create figure
        if new_figure:
            fig = plt.figure( figsize=figsize )
        else:
            fig = plt.gcf()
        # spatial pattern of w^{(i)}
        ax = axw = fig.add_axes([ 0.1, 0.4, 0.2, 0.55 ])
        xx = w_i = posterior['W_hat'][:, i]
        yy = np.arange( self.M )
        ax.plot( [0, 0], [-1, self.M], 'k-', lw=1 )
        ax.plot( xx, yy, 'o-', lw=3, color=col )
        # aesthetics
        ax.set_ylim(-1, self.M )
        ax.set_ylabel('unit #', fontsize=16)
        xts = ax.get_xticks().astype(object)
        xts[1:-1] = ' '
        ax.set_xticklabels(xts)
        # time course of s^{(i)}
        ax = axs = fig.add_axes([ 0.4, 0.1, 0.55, 0.2 ])
        xx = np.arange( self.T )
        yy = s_i = posterior['S_hat'][:, i]
        ax.plot( [-1, self.T], [0, 0], 'k-', lw=1 )
        ax.plot( xx, yy, '-', lw=2, color=col )
        # aesthetics
        ax.set_xlim(0, self.T-1)
        ax.set_xlabel('time bin', fontsize=16)
        ax.set_xticks([0, self.T])
        yts = ax.get_yticks().astype(object)
        yts[1:-1] = ' '
        ax.set_yticklabels(yts)
        # image plot
        ax = axi = fig.add_axes([ 0.4, 0.4, 0.55, 0.55 ])
        H_i = s_i[:, na] * w_i[na, :]
        #ax.imshow( posterior.expected_log_G.T, cmap=plt.cm.Greys,
        #        interpolation='nearest', aspect='auto', alpha=1 )
        ax.imshow( H_i.T, cmap=plt.cm.Blues, alpha=1,
                interpolation='nearest', aspect='auto')
        ax.set_xticks([0, self.T])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        # finish
        if title is None:
            title = ' '
        ax.set_title(title, fontsize=20)
        if draw:
            plt.draw()
            plt.show()
        if return_fig:
            return fig

    def plot_H( self, posterior=None, **kw ):
        if posterior is None:
            posterior = self.posterior
        S = posterior['S_hat']
        W = posterior['W_hat']
        H = posterior['MAP_log_G']
        return self.data.plot_H( S, W, H, **kw )
       

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Cross-validation
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    @property
    def N_observations_testing( self ):
        return self.T_testing * self.M

    @cached
    def Y_testing( data, _slice_by_testing ):
        return _slice_by_testing( data.Y )

    """
    =======
    For `s`
    =======
    """

    @cached
    def LL_testing__s( Mu__s, _slice_by_testing, S, posterior_w, Y_testing,
            c__s ):
        Mu, S = _slice_by_testing( Mu__s ), _slice_by_testing(S)
        return (
                -np.sum( Mu ) 
                + np.trace( mdot(S.T, Y_testing, posterior_w.W) )
                + np.sum( dot(Y_testing, c__s) ) )

    @cached
    def LL_training_per_observation__s( LL_training__s, N_observations ):
        return LL_training__s / N_observations

    @cached
    def LL_testing_per_observation__s( LL_testing__s, N_observations_testing ):
        return LL_testing__s / N_observations_testing

    """
    =======
    For `w`
    =======
    """

    @cached
    def LL_testing__w( Mu__w, _slice_by_testing, posterior_s, W, Y_testing, 
            c__w ):
        Mu = _slice_by_testing( Mu__w )
        S = _slice_by_testing( posterior_s.S )
        return (
                -np.sum( Mu ) 
                + np.trace( mdot(S.T, Y_testing, W) )
                + np.sum( dot(Y_testing, c__w) ) )

    @cached
    def LL_training_per_observation__w( LL_training__w, N_observations ):
        return LL_training__w / N_observations

    @cached
    def LL_testing_per_observation__w( LL_testing__w, N_observations_testing ):
        return LL_testing__w / N_observations_testing



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
SIMPLE POP SOLVER
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    - shared hyperparameters for all modulators
    - weights can be positive or negative
    - given theta_w and theta_s, can solve jointly for `s` and `w`

"""

class PopSimple( PopSolver ):

    def _parse_initial_conditions_s( self, ic_in ):
        # global defaults
        ics = Bunch()
        ics['s_vec'] = zeros( self.rank + self.M )
        ics['dims_si'] = zeros( 2*self.T, dtype=bool )
        ics['dims_si'][0] = True
        ics['theta_s'] = self._default_theta_s0
        # parse initial conditions: None
        if ic_in == None:
            pass
        # parse initial_conditions: dict
        elif isinstance( ic_in, dict ):
            # pull out what's in the dict
            for k in ics.keys():
                ics[k] = ic_in.get( k, ics[k] )
            # if `S` is provided
            if ic_in.has_key('S'):
                S = ic_in['S']
                if not S.shape == ( self.T, self.rank ):
                    raise ValueError('initial `S` is the wrong shape.')
                if ic_in.has_key('s_vec'):
                    raise ValueError('provide only one initial `S`/`s_vec`')
                # collapse
                ics['dims_si'] = dims_si = ones( 2*self.T, dtype=bool )
                s_star = self.collapse_S(S, dims_si, padding='flip').flatten()
                c = ics['s_vec'][-self.M:]
                ics['s_vec'] = s_vec = conc([ s_star, c ])
            # if `c` is provided
            if ic_in.has_key('c'):
                c = ic_in['c']
                if not c.shape == ( self.M, ):
                    raise ValueError('initial `c` is the wrong shape.')
                if ic_in.has_key('s_vec'):
                    raise ValueError('provide only one initial `c`/`s_vec`')
                ics['s_vec'][-self.M:] = c

        # parse initial_conditions: Solver object
        else: 
            # copy posterior on `s`
            if hasattr( ic_in, 'posterior_s' ):
                ics['s_vec'] = ic_in.posterior_s.s_vec
                ics['dims_si'] = ic_in.posterior_s.dims_si
            # recast values of `theta`
            try:
                ics['theta_s'] = self._recast_theta_s( ic_in )
            except TypeError:
                pass
        # replace any invalid values
        for i in range( self.N_theta_s ):
            if (ics['theta_s'][i] is None) or (ics['theta_s'][i] == np.nan):
                ics['theta_s'] = self._default_theta_s0[i]
        # return
        return ics

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving for `s`
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    @cached
    def s_star( s_vec, _required_s_star_length ):
        return s_vec[ :_required_s_star_length ]

    @cached
    def c__s( s_vec, _required_s_star_length, M ):
        #assert len(s_vec) == _required_s_star_length + M
        return s_vec[ _required_s_star_length: ]

    @cached
    def _required_s_vec_length( _required_s_star_length, M ):
        return _required_s_star_length + M

    """
    ====================
    Dim reduction on `s`
    ====================
    """

    def collapse_s( self, s, dims, **kw ):
        """ Reduce dimensionality of `s`. Returns `s_star`.

        Arguments:

        - `s`: (Tp) or (Tp x J) float array

        - `dims`: (2T) boolean array, of which fft coefficients to keep.
            Alternatively, `dims` can be a DimReduce object with the
            key `dims_si`.

        This replaces `collapse_h` in the superclass, and operates on 
        flattened S matrices.

        *IMPORTANT*: it is assumed that the time-varying signals `s_t^(i)` 
        are *interlaced*.  For example, if `s` is 1D, it is of the form:

            [ ..., s_t^(0), s_t^(1), ..., s_t^(p-1), s_(t+1)^(0), ... ]

        Returns:

        - `s_star`: (T*p) or (T*p x J) float array, 
            where T* is the number of `True` elements of `dims`. The form
            is determined to be consistent with the input `s`.

        """
        # parse input
        if not isinstance( dims, np.ndarray ):
            dims = dims['dims_si']
        # function to use
        collapse = self._reducer.collapse_h
        sh = shape(s)
        # the first dimension must be Tp
        if not (sh[0] == self.T * self.rank):
            raise ValueError('first dimension of `s` should be Tp.')
        # if we have a (Tp) array, return a (T*P) array
        if len(sh) == 1:
            S = s.reshape( (self.T, self.rank) )
            S_star = collapse( S, dims, **kw )
            s_star = S_star.flatten()
        # if we have a (Tp x J) array, return a (T*p x J) array
        elif len(sh) == 2:
            S = s.reshape( (self.T, self.rank * sh[1]) )
            S_star = collapse( S, dims, **kw )
            T_star = shape( S_star )[0]
            s_star = S_star.reshape( (T_star * self.rank, sh[1]) )
        else:
            raise ValueError('cannot deal with `s` of this shape')
        return s_star

    def collapse_S( self, S, dims, **kw ):
        """ Reduce dimensionality of `S`. Returns `S_star`.

        Arguments:

        - `S`: (T x p) or (T x pJ) float array
        
        - `dims`: (2T) boolean array, of which fft coefficients to keep.
            Alternatively, `dims` can be a DimReduce object with the
            key `dims_si`.

        This replaces `collapse_h` in the superclass, and operates on 
        S matrices.

        Returns:

        - `S_star`: (T* x p) or (T* x pJ) float array, 
            where T* is the number of `True` elements of `dims`. The form
            is determined to be consistent with the input `S`.

        """
        # parse input
        if not isinstance( dims, np.ndarray ):
            dims = dims['dims_si']
        # function to use
        collapse = self._reducer.collapse_h
        sh = shape(S)
        # the first dimension must be T
        if not (sh[0] == self.T):
            raise ValueError('first dimension of `S` should be T.')
        # the number of dims must be 1 or 2
        if len(sh) not in [1, 2]:
            raise NotImplementedError('cannot deal with `S` of this shape')
        return collapse( S, dims, **kw )

    def expand_s( self, s_star, dims ):
        """ Increase dimensionality of `s_star`. Returns `s`.

        Arguments:

        - `s_star`: (T*p) or (T*p x J) float array

        - `dims`: (2T) boolean array, of which fft coefficients are
            being reported in `S_star`.  Alternatively, `dims` can be 
            a DimReduce object with the key `dims_si`.

        This replaces `expand_h` in the superclass, and operates on
        flattened S matrices.

        *IMPORTANT*: it is assumed that the signals `s*_t^(i)` are 
        *interlaced*. For example, if `S_star` is 1D, it is of the form:

            [ ..., s*_t^(0), s*_t^(1), ..., s*_t^(p-1), s*_(t+1)^(0), ... ]

        Since, dimensionality reduction involves a Fourier transform, this
        amounts to asserting that the (reduced) Fourier coefficients of the
        respective s*^(i) are interlaced.

        Returns:

        - `s`: (Tp) or (Tp x J) float array.

        """
        # parse input
        if not isinstance( dims, np.ndarray ):
            dims = dims['dims_si']
        # function to use
        expand = self._reducer.expand_h
        sh = shape(s_star)
        T_star = np.sum(dims)
        # the first dimension must be T*p
        if not (sh[0] == T_star * self.rank):
            raise ValueError('first dimension of `s` should be T*p.')
        # if we have a (T*p) array, return a (Tp) array
        if len(sh) == 1:
            S_star = s_star.reshape( (T_star, self.rank) )
            S = expand( S_star, dims )
            s = S.flatten()
        # if we have a (T*p x J) array, return a (Tp x J) array
        elif len(sh) == 2:
            S_star = s_star.reshape( (T_star, self.rank * sh[1]) )
            S = expand( S_star, dims )
            s = S.reshape( (self.T * self.rank, sh[1]) )
        else:
            raise ValueError('cannot deal with `S_star` of this shape')
        return s

    def expand_S( self, S_star, dims ):
        """ Increase dimensionality of `S_star`. Returns `S`.

        Arguments:

        - `S_star`: (T* x p) or (T* x pJ) float array

        - `dims`: (2T) boolean array, of which fft coefficients are
            being reported in `S_star`.  Alternatively, `dims` can be a 
            DimReduce object with the key `dims_si`.

        This replaces `expand_h` in the superclass, and operates on
        S matrices.

        Returns:

        - `S`: (T x p) or (T x pJ) float array.

        """
        # parse input
        if not isinstance( dims, np.ndarray ):
            dims = dims['dims_si']
        # function to use
        expand = self._reducer.expand_h
        sh = shape(S_star)
        T_star = np.sum(dims)
        # the first dimension must be T*
        if not (sh[0] == T_star):
            raise ValueError('first dimension of `S` should be T.')
        # the number of dims must be 1 or 2
        if len(sh) not in [1, 2]:
            raise NotImplementedError('cannot deal with `S` of this shape')
        return expand( S_star, dims )

    """
    ================
    Log Prior on `s`
    ================
    """

    @cached
    def Lsi( theta_s ):
        raise NotImplementedError('must subclass')

    @cached
    def dLsi_dtheta( theta_s ):
        raise NotImplementedError('must subclass')

    @cached
    def dims_si( Lsi, _cutoff_lambda_s ):
        """ The non-singular dimensions of s^{(i)}, from the prior. """
        return ( Lsi/maxabs(Lsi) > _cutoff_lambda_s )

    @cached
    def T_star( dims_si ):
        return np.sum( dims_si )

    @cached
    def _required_s_star_length( T_star, rank ):
        return T_star * rank

    @cached
    def Lsi_star( Lsi, dims_si ):
        return Lsi[ dims_si ]

    @cached
    def logdet_Cs_star( rank, Lsi_star, T_star, posterior_w ):
        return ( rank * np.sum(np.log(Lsi_star)) 
                + T_star * logdet( posterior_w.Aw_inv ) )

    def _reproject_to_S_star( self, S_star=None, dims_si=None, posterior=None ):
        """ Unprojects and reprojects S_star (to the new dim).

        It is assumed that `S_star` has dimensionality described by `dims_si`.
        It is reprojected according to the current value of `self.dims_si`. 

        Note that this does not change the object's state.

        If P1 is the projection operator for dims_si_source, and P2 is the 
        projection operator for dims_si_target, then this is the equivalent
        of applying P2 . P1^-1 to the matrix S_star.

        """
        if [ S_star, dims_si, posterior ].count(None) == 0:
            raise ValueError('either provide `S_star/dims_s` or `posterior`')
        elif posterior is not None:
            return self._reproject_to_S_star( 
                    S_star=posterior.S_star, dims_si=posterior.dims_si )
        else:
            new_S_star = zeros( (self.T * 2, self.rank) )
            new_S_star[ dims_si, : ] = S_star
            new_S_star = new_S_star[ self.dims_si, : ]
            return new_S_star
    
    def _reproject_to_s_vec( self, s_vec=None, dims_si=None, posterior=None ):
        """ Unprojects and reprojects `s_vec` (to the new dim).

        Can either provide `s_vec` and `dims_si`, *or* a posterior which
        contains these as attributes.

        Note that this does not change the object's state.

        """
        # check inputs
        if [ s_vec, dims_si, posterior ].count(None) == 0:
            raise ValueError('either provide `s_vec/dims_s` or `posterior`')
        # provide a posterior
        elif posterior is not None:
            return self._reproject_to_s_vec( 
                    s_vec=posterior.s_vec, dims_si=posterior.dims_si )
        # provide `s_vec` and `dims_si` values
        else:
            # extract `s_star` and `c`
            T_star = np.sum( dims_si )
            assert len(s_vec) == (T_star * self.rank) + self.M
            s_star = s_vec[ :-self.M ]
            c = s_vec[ -self.M: ]
            # reproject `s_star`
            S_star = s_star.reshape( T_star, self.rank )
            new_s_star = self._reproject_to_S_star( 
                    S_star=S_star, dims_si=dims_si ).flatten()
            # put back together again
            new_s_vec = conc([ new_s_star, c ])
            return new_s_vec


    """
    ============================
    Conditional posterior on `s`
    ============================
    """

    @cached
    def S_star( s_star, T_star, rank ):
        return s_star.reshape( T_star, rank )

    @cached( settable=True )
    def S( S_star, dims_si, expand_S ):
        return expand_S( S_star, dims_si )

    @cached
    def H__s( S, c__s, posterior_w ):
        return dot( S, posterior_w.W.T ) + c__s[na, :]

    @cached
    def G__s( H__s ):
        return exp( H__s )

    @cached
    def expected_FXK_times_known_G( posterior_k, known_G ):
        if known_G is None:
            return posterior_k.expected_FXK
        else:
            return posterior_k.expected_FXK * known_G

    @cached
    def Mu__s( G__s, expected_FXK_times_known_G ):
        return G__s * expected_FXK_times_known_G 

    @cached
    def LL_training__s( Mu__s, _slice_by_training, Y_training, S, posterior_w, 
            c__s ):
        """ Training log likelihood as a function of `s_vec`. """
        Mu, S = _slice_by_training( Mu__s ), _slice_by_training(S)
        return (
                -np.sum( Mu ) 
                + np.trace( dot(S.T, posterior_w.YW_training) )
                + np.sum( dot(Y_training, c__s) ) )

    @cached
    def LPrior__s( posterior_w, S_star, Lsi_star ):
        """ Log prior as a function of `s_vec`. """
        return -0.5 * np.trace( mdot( 
            posterior_w.Aw, S_star.T, S_star / Lsi_star[:, na] ) )

    @cached
    def LP_training__s( LL_training__s, LPrior__s ):
        """ Training log posterior as a function of `s_vec`. """
        return LL_training__s + LPrior__s

    """ Jacobian """

    @cached
    def Resid_training__s( data, Mu__s, _zero_during_testing ):
        return _zero_during_testing( data.Y - Mu__s )

    @cached
    def dLL_training__s( Resid_training__s, dims_si, collapse_S, posterior_w ):
        """ Jacobian of training log likelihood wrt `s_vec`. """
        Resid_star = collapse_S( Resid_training__s, dims_si, padding='zero' )
        dLL_ds = dot( Resid_star, posterior_w.W ).flatten()
        dLL_dc = np.sum( Resid_training__s, axis=0 )
        return conc([ dLL_ds, dLL_dc ])

    @cached
    def dLPrior__s( S_star, posterior_w, Lsi_star, M ):
        """ Jacobian of log prior wrt `s_vec`. """
        dLPrior_ds = (-dot(S_star, posterior_w.Aw) / Lsi_star[:, na]).flatten()
        dLPrior_dc = zeros( M )
        return conc([ dLPrior_ds, dLPrior_dc ])

    @cached
    def dLP_training__s( dLL_training__s, dLPrior__s ):
        """ Jacobian of training log posterior wrt `s_vec`. """
        return dLL_training__s + dLPrior__s 

    """ Hessian """

    @cached
    def d2LP_training__s( Mu__s, _zero_during_testing, T_star, rank, 
            _required_s_star_length, _required_s_vec_length, 
            posterior_w, dims_si, Lsi_star, M, collapse_S ):
        """ Hessian of training log posterior wrt `s_vec`. """

        import sandwich
        # keep training region only
        Mu = _zero_during_testing( Mu__s )
        # create container for Hessian
        len_s_vec = _required_s_vec_length
        len_s_star = _required_s_star_length 
        d2LP = np.zeros( (len_s_vec, len_s_vec) )

        # d2LP / ds^{(i)}.ds^{(j)}
        for i in range(rank):
            for j in range(i, rank):
                # calculate LL component
                Mu_ij = dot( Mu, posterior_w.WiWj[i, j] )
                Mu_ij_padded = np.hstack([ Mu_ij, np.zeros_like(Mu_ij) ])
                yy = np.fft.fft( Mu_ij_padded )
                #F = ( np.sum(dims_si) - 1 ) / 2
                F = ( np.sum(dims_si) ) / 2 # DC TERM INCLUDED
                d2LP_ij = -sandwich.calzino( yy.real, yy.imag, F ) / len(yy)
                #d2LP_ij[ 0, 1: ] /= sqrt(2)
                #d2LP_ij[ 1:, 0 ] *= sqrt(2)
                d2LP_ij = d2LP_ij[ 1:, 1: ] # NO DC
                # add LPrior component
                d2LP_ij[ range(T_star), range(T_star) ] -= ( 
                        posterior_w.Aw[i, j] / Lsi_star )
                # put into container
                idx_i = slice( i, len_s_star, rank )
                idx_j = slice( j, len_s_star, rank )
                d2LP[idx_i, :len_s_star][:len_s_star, idx_j] = d2LP_ij
                # symmetry
                if j > i:
                    d2LP[idx_j, :len_s_star][:len_s_star, idx_i] = d2LP_ij

        # d2LP / dc^2
        diag_idx = len_s_star + np.arange(M)
        d2LP[ diag_idx, diag_idx ] = -np.sum( Mu, axis=0 )

        # d2LP / ds^{(i)}.dc
        for i in range( rank ):
            # calculate
            d2LP_i = -collapse_S( Mu * posterior_w.W[:, i][na, :], dims_si )
            # put into container
            d2LP[i:len_s_star:rank, len_s_star:] = d2LP_i
            d2LP[len_s_star:, i:len_s_star:rank] = d2LP_i.T

        return d2LP

    """
    ===============
    Solution for `s`
    ===============
    """
   
    @cached
    def _negLP_objective__s( LP_training__s ):
        return -LP_training__s

    @cached
    def _negLP_jacobian__s( dLP_training__s ):
        return -dLP_training__s

    @cached
    def _negLP_hessian__s( d2LP_training__s ):
        return -d2LP_training__s

    def _negLP_callback__s( self, **kw ):
        try:
            offset = self.posterior_w.LPrior__w + self.posterior_k.LPrior__k
        except AttributeError:
            offset = 0
        LP_total = self.LP_training__s + offset
        self.announce( 'LP step:  %.3f' % LP_total, n_steps=4, **kw )

    def _preoptimise_s0_vec_for_LP_objective(self, s_vec_0, ftol_per_obs=None):
        """ Trims the high freqs from `s_vec_0` as a pre-optimisation.

        Trimming continues so long as the improvement in the objective is
        greater than `ftol_per_obs * self.N_observations`. The parameter
        `ftol_per_obs`, if not provided, defaults to `self.ftol_per_obs`.

        """
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        # functions
        eq = np.array_equal
        f = self.function( '_negLP_objective__s', 's_vec' )
        # initial value
        best_s_vec_0 = s_vec_0
        best_objective = f( s_vec_0 )
        N_coeffs_removed = 0
        # start removing coefficients
        s_vec_1 = s_vec_0.copy()
        ndims = (len(s_vec_1) - self.M - 1)/2
        offsets = np.arange(ndims-1, -1, -1)
        for i, o in enumerate(offsets):
            s_vec_1[o + ndims + 1] = 0
            s_vec_1[o + 1] = 0
            # if we have decreased the objective, note this solution
            delta_obj = f(s_vec_1) - best_objective
            if delta_obj < 0:
                best_s_vec_0 = s_vec_1.copy()
                best_objective = f(best_s_vec_0)
                N_coeffs_removed = 2 * (i + 1)
            # if no improvement or too small, end here
            if delta_obj > -ftol_per_obs * self.N_observations:
                break
        # try the zero solution, with a constant
        s_vec_1[ :-self.M ] *= 0
        if f(s_vec_1) < best_objective:
            best_s_vec_0 = s_vec_1.copy()
            best_objective = f(best_s_vec_0)
            N_coeffs_removed = len(s_vec_0) - self.M
        # try the zero solution, with zero constant
        s_vec_1 *= 0
        if f(s_vec_1) < best_objective:
            best_s_vec_0 = s_vec_1.copy()
            best_objective = f(best_s_vec_0)
            N_coeffs_removed = len(s_vec_0)
        # try the c__w solution
        try:
            s_vec_1[ -self.M: ] = self.c__w
            if f(s_vec_1) < best_objective:
                best_s_vec_0 = s_vec_1.copy()
                best_objective = f(best_s_vec_0)
                N_coeffs_removed = len(s_vec_0) - self.M
        except AttributeError:
            pass
        # announce
        if N_coeffs_removed > 0:
            self.announce( 'LP step: pre-zeroed %d coefficients of `s`' 
                    % N_coeffs_removed, prefix='calc_posterior_s' )
        return best_s_vec_0

    """
    ===============================
    Posterior_s specific attributes
    ===============================

    These should only be computed (and used) for posterior objects.

    """

    @cached
    def Lambdainv_s_vec( _negLP_hessian__s, _is_posterior ):
        if not _is_posterior:
            raise TypeError('attribute only available for `s` posterior.')
        return _negLP_hessian__s

    @cached
    def Lambdainv_s_star( Lambdainv_s_vec, M ):
        return Lambdainv_s_vec[ :-M, :-M ]

    @cached
    def Lambda_s_star( Lambdainv_s_star ):
        return inv( Lambdainv_s_star )

    @cached
    def RsiT( T_star, dims_si, expand_S ):
        return expand_S( eye(T_star), dims_si )

    @cached
    def diag_Lambda_s( T, rank, expand_S, Lambda_s_star, dims_si, RsiT ):
        raise NotImplementedError()
        """
        diag_Lambda_s = np.zeros( T * rank )
        for i in range( rank ):
            diag_Lambda_s[i::rank] = np.sum( 
                self.expand_S( Lambda_s_star[i::rank, :][:, i::rank], dims_si )
                * RsiT, axis=1 )
        """

    @cached
    def MAP_H__s( H__s ):
        return H__s

    @cached
    def expected_H__s( H__s ):
        return H__s

    @cached
    def MAP_G__s( G__s ):
        return G__s

    @cached( cskip = ('is_point_estimate', True, 'diag_Lambda_s') )
    def expected_G__s( is_point_estimate, G__s, diag_Lambda_s ):
        if is_point_estimate:
            return G__s
        else:
            raise NotImplementedError()

    @cached
    def As( S_star, Lsi_star ):
        return dot( S_star.T, S_star / Lsi_star[:, na] )

    @cached
    def As_inv( As ):
        try:
            return inv( As )
        except np.linalg.LinAlgError:
            tracer()
            return np.nan * As
        
    @cached
    def S_is_degenerate( S ):
        return (S == 0).all()

    @cached
    def logdet_Lambdainv_s_star( Lambdainv_s_star ):
        return logdet( Lambdainv_s_star )

    @cached( cskip = [ ('S_is_degenerate', True, [
        'logdet_Cs_star', 'logdet_Lambdainv_s_star', 'posterior_w', 'As'] )] )
    def evidence_components__s( S_is_degenerate, LL_training__s, 
            logdet_Cs_star, logdet_Lambdainv_s_star, posterior_w, As ):
        if S_is_degenerate:
            return A([ LL_training__s, 0, 0 ])
        else:
            return A([ 
                LL_training__s,
                -0.5 * ( logdet_Cs_star + logdet_Lambdainv_s_star ),
                -0.5 * np.trace( dot(posterior_w.Aw, As) ) ])

    @cached
    def evidence__s( evidence_components__s ):
        return np.sum( evidence_components__s )

    @cached
    def E_star__s( _is_posterior, d2LP_training__s, rank, T_star,
            posterior_w, Lsi_star, _required_s_star_length ):
        """ Hessian of neg log likelihood, for local evidence approx """
        if not _is_posterior:
            raise TypeError('attribute only available for `s` posterior.')
        # start with hessian
        E = d2LP_training__s.copy()
        # only the components wrt `s_star`
        E = E[ :_required_s_star_length, :_required_s_star_length ]
        # remove the LPrior component
        for i in range(rank):
            for j in range(i, rank):
                # retrieve submatrix
                E_ij = E[i::rank, :][:, j::rank]
                # remove LPrior component
                E_ij[ range(T_star), range(T_star) ] += ( 
                        posterior_w.Aw[i, j] / Lsi_star )
                # put into container
                E[i::rank, :][:, j::rank] = E_ij
                # symmetry
                if j > i:
                    E[j::rank, :][:, i::rank] = E_ij
        # make it negative
        return -E

    """ Posterior specific, for use by `w` solver """

    @cached
    def STY_training( S, Y_training, _slice_by_training ):
        return dot( _slice_by_training(S).T, Y_training )

    @cached
    def SiSj( S, rank ):
        SiSj = np.empty( (rank, rank), dtype=object )
        for i in range(rank):
            for j in range(i, rank):
                SiSj[i, j] = S[:, i] * S[:, j]
        return SiSj

    """
    ===========================
    Local evidence for `s`
    ===========================

    At the current `theta_s` (which is a candidate solution, and so is denoted
    `theta_s_c`). Note that `star` here refers to the dim reduction induced 
    by `theta_s_n`, not `theta_s_c`, so dim reduction is not handled by the 
    same attributes as the standard `theta_s`.

    """

    @cached
    def Lsi_c_star( Lsi, posterior_s ):
        """ Freq-domain diagonal cov matrix at candidate theta_s_c. """
        return Lsi[ posterior_s.dims_si ]

    @cached
    def Lsi_c_star_is_singular( Lsi_c_star, _cutoff_lambda_s ):
        """ Is candidate diagonal cov matrix near-singular. """
        cutoff = _cutoff_lambda_s * maxabs(Lsi_c_star)
        return ( Lsi_c_star < cutoff ).any()

    @cached
    def dims_si_c_star( Lsi_c_star, _cutoff_lambda_s, Lsi_c_star_is_singular ):
        """ How to dim reduce from star- to plus-space. """
        if Lsi_c_star_is_singular:
            cutoff = _cutoff_lambda_s * maxabs(Lsi_c_star)
            return ( Lsi_c_star >= cutoff )
        else:
            return np.ones( len(Lsi_c_star), dtype=bool )

    @cached
    def dims_si_c( Lsi_c_star_is_singular, dims_si_c_star, posterior_s ):
        """ How to dim reduce from full- to plus-space. """
        if Lsi_c_star_is_singular:
            dims_si_c = posterior_s.dims_si.copy()
            dims_si_c[ dims_si_c ] = dims_si_c_star
            return dims_si_c
        else:
            return posterior_s.dims_si

    @cached
    def Lsi_c_plus( Lsi_c_star, dims_si_c_star, Lsi_c_star_is_singular ):
        if Lsi_c_star_is_singular:
            return Lsi_c_star[ dims_si_c_star ]
        else:
            return Lsi_c_star

    @cached
    def T_plus( dims_si_c ):
        return np.sum( dims_si_c )

    @cached
    def E_n_plus__s( posterior_s, dims_si_c_star, Lsi_c_star_is_singular, 
            rank ):
        if Lsi_c_star_is_singular:
            idx = np.repeat(dims_si_c_star[:, na], rank, axis=1).flatten()
            return posterior_s.E_star__s[idx, :][:, idx]
        else:
            return posterior_s.E_star__s

    @cached
    def Lambdainv_n_plus__s( posterior_s, dims_si_c_star, rank,
            Lsi_c_star_is_singular ):
        if Lsi_c_star_is_singular:
            idx = np.repeat(dims_si_c_star[:, na], rank, axis=1).flatten()
            return posterior_s.Lambdainv_s_star[idx, :][:, idx]
        else:
            return posterior_s.Lambdainv_s_star

    @cached
    def S_n_plus( posterior_s, dims_si_c_star, Lsi_c_star_is_singular ):
        if Lsi_c_star_is_singular:
            return posterior_s.S_star[ dims_si_c_star, : ]
        else:
            return posterior_s.S_star

    @cached
    def s_n_plus( S_n_plus ):
        return S_n_plus.flatten()

    """ Approximate posterior at candidate theta """

    @cached
    def Cs_c_plus( Lsi_c_plus, posterior_w ):
        return np.kron( diag(Lsi_c_plus), posterior_w.Aw_inv )

    @cached
    def Csinv_c_plus( Lsi_c_plus, posterior_w ):
        return np.kron( diag(1. / Lsi_c_plus), posterior_w.Aw )

    @cached
    def Lambdainv_c_plus__s( E_n_plus__s, Csinv_c_plus ):
        return E_n_plus__s + Csinv_c_plus 

    @cached
    def Lambda_c_plus__s( Lambdainv_c_plus__s ):
        return inv( Lambdainv_c_plus__s )

    @cached
    def s_c_plus( Lambdainv_c_plus__s, Lambdainv_n_plus__s, s_n_plus ):
        try:
            return ldiv( 
                    Lambdainv_c_plus__s, 
                    dot(Lambdainv_n_plus__s, s_n_plus) )
        except np.linalg.LinAlgError:
            tracer()
            return 0 * s_n_plus

    @cached
    def S_c_plus( s_c_plus, T_plus, rank ):
        return s_c_plus.reshape( T_plus, rank )

    @cached
    def S_c( S_c_plus, dims_si_c, expand_S ):
        return expand_S( S_c_plus, dims_si_c )

    @cached
    def H_c__s( S_c, posterior_w, posterior_s ):
        return dot( S_c, posterior_w.W.T ) + posterior_s.c__s[na, :]

    @cached
    def Mu_c__s( H_c__s, expected_FXK_times_known_G ):
        return exp(H_c__s) * expected_FXK_times_known_G

    """ Local evidence at candidate theta """

    @cached
    def local_evidence_components__s( _slice_by_training, Mu_c__s, H_c__s,
            Y_training, Cs_c_plus, E_n_plus__s, T_plus, rank, s_c_plus,
            Csinv_c_plus ):
        # log likelihood
        sli = _slice_by_training
        LL = np.sum( -sli(Mu_c__s) + sli(H_c__s) * Y_training ) 
        if len( s_c_plus ) == 0:
            return LL
        # psi(theta)
        psi = A([ 
            LL,
            -0.5 * logdet( dot(Cs_c_plus, E_n_plus__s) + eye(T_plus * rank) ),
            -0.5 * mdot( s_c_plus, Csinv_c_plus, s_c_plus ) ])
        return psi

    @cached
    def local_evidence__s( local_evidence_components__s ):
        return np.sum( local_evidence_components__s )

    @cached
    def _LE_objective__s( local_evidence__s ):
        return -local_evidence__s

    @cached
    def _LE_jacobian__s( dLsi_dtheta, dims_si_c, data, Mu_c__s, 
            _zero_during_testing, collapse_S, posterior_w, Cs_c_plus, 
            E_n_plus__s, Csinv_c_plus, s_c_plus, Lambda_c_plus__s,
            Lambdainv_c_plus__s, N_theta_s ):
        # project to +-space
        dLsi_dtheta_plus = [ dL[dims_si_c] for dL in dLsi_dtheta ]
        # residuals at candidate solution
        Resid = data.Y - Mu_c__s
        Resid = _zero_during_testing( Resid ) # only training data
        Resid_plus = collapse_S( Resid, dims_si_c, padding='zero' )
        # intermediate quantities( all in +-space )
        dLL_dS_c = dot( Resid_plus, posterior_w.W )
        dLL_ds_c = dLL_dS_c.flatten()
        C_En = dot( Cs_c_plus, E_n_plus__s )
        Cinv_s = dot( Csinv_c_plus, s_c_plus )
        Lam_Cinv = dot( Lambda_c_plus__s, Csinv_c_plus )
        sT_En_minus_Cinv = dot( s_c_plus.T, E_n_plus__s - Csinv_c_plus )
        # calculate for each variable in theta
        dpsi = np.empty( N_theta_s )
        # derivatives wrt `theta_s` variables
        for j in range( N_theta_s ):
            dLsi = dLsi_dtheta_plus[j]
            dCi = np.kron( diag(dLsi), posterior_w.Aw_inv )
            B = mdot( Lam_Cinv, dCi, Csinv_c_plus)
            Bs = ldiv( 
                    Lambdainv_c_plus__s,
                    mdot( Csinv_c_plus, dCi, Cinv_s ) )
            dpsi[j] = sum([
                dot( dLL_ds_c, Bs ),
                -0.5 * np.trace( dot( B, C_En ) ),
                0.5 * dot( sT_En_minus_Cinv, Bs ) ])
        # make negative
        return -dpsi

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving for `w` 
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    Note: currently have *only* implemented version where Cwi is diagonal.

    Refer to commit # b735c56701efe120ed7d3a3fa2836eb2ef5b0f70 for 
    deprecated code.

    """
    
    @cached
    def w_star( w_vec, _required_w_star_length ):
        return w_vec[ :_required_w_star_length ]

    @cached
    def c__w( w_vec, _required_w_star_length ):
        return w_vec[ _required_w_star_length: ]

    @cached
    def _required_w_vec_length( _required_w_star_length, M ):
        return _required_w_star_length + M

    """
    ================
    Log Prior on `w`
    ================
    """

    @cached
    def Cwi( Cwi_is_diagonal ):
        if Cwi_is_diagonal:
            return None
        else:
            raise TypeError('must subclass to define `Cwi` behaviour')

    @cached
    def dCwi_dtheta():
        raise TypeError('must subclass to define `dCk_dtheta` behaviour')

    @cached
    def Lwi( Cwi_is_diagonal, Cwi_eig ):
        if Cwi_is_diagonal:
            raise TypeError('must subclass to define `Lwi` behaviour')
        else:
            raise NotImplementedError()
       
    @cached
    def dLwi_dtheta():
        raise TypeError('must subclass to define `dLwi_dtheta` behaviour')

    @cached
    def dims_wi( Lwi, _cutoff_lambda_w ):
        return ( Lwi/maxabs(Lwi) > _cutoff_lambda_w )

    @cached
    def M_star( dims_wi ):
        return np.sum( dims_wi )

    @cached
    def _required_w_star_length( M_star, rank ):
        return M_star * rank

    @cached
    def Rwi( Cwi_eig, dims_wi, Cwi_is_diagonal ):
        raise NotImplementedError()

    @cached
    def Rwi_is_identity( Cwi_is_diagonal, dims_wi ):
        if Cwi_is_diagonal:
            return np.all(dims_wi) 
        else:
            raise NotImplementedError()

    @cached
    def Lwi_star( Lwi, dims_wi ):
        return Lwi[ dims_wi ]

    @cached
    def Lwi_inv_star( Lwi_star ):
        return 1. / Lwi_star

    @cached
    def logdet_Cw_star( rank, Lwi_star, M_star, posterior_s ):
        return ( rank * np.sum(np.log(Lwi_star)) 
                + M_star * logdet( posterior_s.As_inv ) )

    def _reproject_to_W_star( self, W=None, posterior=None ):
        """ Calculates `W_star` from `W`. Does not change object's state. 
        
        Can either provide `W` *or* a posterior which contains this attribute.
        
        """
        # check inputs
        if [W, posterior].count(None) != 1:
            raise ValueError('either provide `W` or posterior')
        # provide a posterior
        elif posterior is not None:
            return self._reproject_to_W_star( W=posterior.W )
        # provide a `W` value
        elif W is not None:
            if self.Rwi_is_identity:
                return W
            elif self.Cwi_is_diagonal:
                return W[ self.dims_wi, : ]
            else:
                raise NotImplementedError()

    def _reproject_to_w_star( self, w=None, posterior=None ):
        """ Calculates `w_star` from `w`. Does not change object's state. 
        
        Can either provide `w` *or* a posterior which contains this attribute.
        
        """
        # check inputs
        if [w, posterior].count(None) != 1:
            raise ValueError('either provide `w` or posterior')
        # provide a posterior
        elif posterior is not None:
            if hasattr( posterior, 'w' ):
                w = posterior.w
            elif hasattr( posterior, 'W' ):
                w = posterior.W.flatten()
            else:
                raise KeyError('posterior must have `w` or `W`')
            return self._reproject_to_w_star( w=w )
        # provide a `w` value
        elif w is not None:
            W = w.reshape( self.M, self.rank )
            W_star = self._reproject_to_W_star( W=W )
            return W_star.flatten()

    def _reproject_to_w_vec( self, w=None, c=None, posterior=None ):
        # check inputs
        if [w, posterior].count(None) != 1:
            raise ValueError('either provide `w` or posterior')
        # provide a posterior
        elif posterior is not None:
            if hasattr( posterior, 'w' ):
                w = posterior.w
            elif hasattr( posterior, 'W' ):
                w = posterior.W.flatten()
            else:
                raise KeyError('posterior must have `w` or `W`')
            if hasattr( posterior, 'c' ):
                c = posterior.c
            elif hasattr( posterior, 'c__w' ):
                c = posterior.c__w
            elif hasattr( posterior, 'c__sw' ):
                c = posterior.c__sw
            return self._reproject_to_w_vec( w=w, c=c )
        # provide a `w` value
        elif w is not None:
            W = w.reshape( self.M, self.rank )
            W_star = self._reproject_to_W_star( W=W )
            w_star = W_star.flatten()
            if c is None:
                c = zeros( self.M )
            w_vec = conc([ w_star, c ])
            return w_vec

    @cached( cskip = ('Rwi_is_identity', True, 'dims_wi') )
    def STY_star_training( posterior_s, Rwi_is_identity, Cwi_is_diagonal, 
            dims_wi ):
        STY = posterior_s.STY_training # (p x M)
        if Rwi_is_identity:
            return STY
        elif Cwi_is_diagonal:
            return STY[ :, dims_wi ]
        else:
            raise NotImplementedError()

    """
    ============================
    Conditional posterior on `w`
    ============================
    """

    @cached
    def W_star( w_star, M_star, rank ):
        return w_star.reshape( M_star, rank )

    @cached( cskip = ('Rwi_is_identity', True, ['M', 'rank', 'dims_wi']) )
    def W( W_star, Rwi_is_identity, Cwi_is_diagonal, M, rank, dims_wi ):
        if Rwi_is_identity:
            return W_star
        elif Cwi_is_diagonal:
            W = np.zeros( M, rank )
            W[ dims_wi, : ] = W_star
            return W
        else:
            raise NotImplementedError()

    @cached
    def H__w( posterior_s, W, c__w ):
        return dot( posterior_s.S, W.T ) + c__w[na, :]

    @cached
    def G__w( H__w ):
        return exp( H__w )

    @cached
    def Mu__w( G__w, expected_FXK_times_known_G ):
        return G__w * expected_FXK_times_known_G

    @cached
    def LL_training__w( Mu__w, _slice_by_training, STY_star_training, W_star, 
            c__w, Y_training ):
        """ Training log likelihood as a function of `w_vec`. """
        Mu = _slice_by_training( Mu__w )
        return (
                -np.sum( Mu ) 
                + np.trace( dot(STY_star_training, W_star) )
                + np.sum( dot(Y_training, c__w) ) )

    @cached
    def LPrior__w( posterior_s, W_star, Lwi_star ):
        """ Log prior as a function of `w_vec`. """
        return -0.5 * np.trace( 
                mdot( W_star.T, W_star / Lwi_star[:, na], posterior_s.As ) )

    @cached
    def LP_training__w( LL_training__w, LPrior__w ):
        """ Training log posterior as a function of `w_vec`. """
        return LL_training__w + LPrior__w

    """ Jacobian """

    @cached
    def Resid_training__w( data, Mu__w, _zero_during_testing ):
        return _zero_during_testing( data.Y - Mu__w )

    @cached( cskip = ('Rwi_is_identity', True, 'dims_wi') )
    def dLL_training__w( Rwi_is_identity, Cwi_is_diagonal, 
            Resid_training__w, dims_wi, posterior_s ):
        """ Jacobian of training log likelihood wrt `w_vec`. """
        if Rwi_is_identity:
            Resid_star = Resid_training__w
        elif Cwi_is_diagonal:
            Resid_star = Resid_training__w[ dims_wi, : ]
        else:
            raise NotImplementedError()
        dLL_dw = dot( Resid_star.T, posterior_s.S ).flatten()
        dLL_dc = np.sum( Resid_training__w, axis=0 )
        return conc([ dLL_dw, dLL_dc ])

    @cached
    def dLPrior__w( W_star, posterior_s, Lwi_star, M ):
        """ Jacobian of log prior wrt `w_vec`. """
        dLPrior_dw = (-dot(W_star, posterior_s.As) / Lwi_star[:, na]).flatten()
        dLPrior_dc = zeros( M )
        return conc([ dLPrior_dw, dLPrior_dc ])

    @cached
    def dLP_training__w( dLL_training__w, dLPrior__w ):
        """ Jacobian of training log posterior wrt `w_vec`. """
        return dLL_training__w + dLPrior__w

    """ Hessian """

    @cached( cskip = ('Rwi_is_identity', True, 'dims_wi') )
    def d2LP_training__w( Mu__w, _zero_during_testing, M_star, rank, M,
            _required_w_star_length, _required_w_vec_length,
            posterior_s, dims_wi, Lwi_star, Rwi_is_identity, Cwi_is_diagonal ):
        """ Hessian of training log posterior wrt `w_vec`. """
        # keep training region only
        Mu = _zero_during_testing( Mu__w )
        # create container for Hessian
        len_w_vec = _required_w_vec_length
        len_w_star = _required_w_star_length
        d2LP = np.zeros( (len_w_vec, len_w_vec) )

        # d2LP / dw^{(i)}.dw^{(j)}
        for i in range(rank):
            for j in range(i, rank):
                # calculate LL component
                d2LP_ij = -diag( dot( Mu.T, posterior_s.SiSj[i, j] ) )
                # transform
                if Rwi_is_identity:
                    d2LP_ij_star = d2LP_ij
                elif Cwi_is_diagonal:
                    d2LP_ij_star = d2LP_ij[ dims_wi, : ][ :, dims_wi ]
                else:
                    raise NotImplementedError()
                # calculate LPrior component
                d2LP_ij_star[ range(M_star), range(M_star) ] -= (
                    posterior_s.As[i, j] / Lwi_star )
                # put into container
                idx_i = slice( i, len_w_star, rank )
                idx_j = slice( j, len_w_star, rank )
                d2LP[idx_i, :len_w_star][:len_w_star, idx_j] = d2LP_ij_star
                # symmetry
                if j > i:
                    d2LP[idx_j, :len_w_star][:len_w_star, idx_i] = d2LP_ij_star

        # d2LP / dc^2
        diag_idx = len_w_star + np.arange(M)
        d2LP[ diag_idx, diag_idx ] = -np.sum( Mu, axis=0 )

        # d2LP / dw^{(i)}.dc
        MTS = dot( Mu.T, posterior_s.S )
        for i in range(rank):
            if Rwi_is_identity:
                idx_i = np.arange( i, len_w_star, rank )
                idx_j = len_w_star + np.arange(M)
                d2LP[ idx_i, idx_j ] = -MTS[:, i]
                d2LP[ idx_j, idx_i ] = -MTS[:, i].T
            elif Cwi_is_diagonal:
                raise NotImplementedError()
                #idx_i = slice( i, len_w_star, rank )
                #idx_j = slice( len_w_star, len_w_star + M )
                #d2LP[idx_i, idx_j] = diag(-MTS[:, i])[ dims_w, : ]
                #d2LP[idx_j, idx_i] = diag(-MTS[:, i])[ dims_w, : ].T
            else:
                raise NotImplementedError()

        return d2LP

    """
    ===============
    Solution for `w`
    ===============
    """
   
    @cached
    def _negLP_objective__w( LP_training__w ):
        return -LP_training__w

    @cached
    def _negLP_jacobian__w( dLP_training__w ):
        return -dLP_training__w

    @cached
    def _negLP_hessian__w( d2LP_training__w ):
        return -d2LP_training__w

    def _negLP_callback__w( self, **kw ):
        try:
            offset = self.posterior_w.LPrior__w + self.posterior_k.LPrior__k
        except AttributeError:
            offset = 0
        LP_total = self.LP_training__w + offset
        if np.isnan( LP_total ):
            tracer()
        self.announce( 'LP step:  %.3f' % LP_total, n_steps=4, **kw )

    def _preoptimise_w0_vec_for_LP_objective(self, w_vec_0, ftol_per_obs=None):
        """ Zeros `w_vec_0` as necessary, as a pre-optimisation. """
        # parse tolerance
        if ftol_per_obs is None:
            ftol_per_obs = self.ftol_per_obs
        # functions
        eq = np.array_equal
        f = self.function( '_negLP_objective__w', 'w_vec' )
        # initial value
        best_w_vec_0 = w_vec_0
        best_objective = f( w_vec_0 )
        N_coeffs_removed = 0
        # try the zero solution, with a constant
        w_vec_1 = w_vec_0.copy()
        w_vec_1[ :-self.M ] *= 0
        if f(w_vec_1) < best_objective:
            best_w_vec_0 = w_vec_1.copy()
            best_objective = f(best_w_vec_0)
            N_coeffs_removed = len(w_vec_0) - self.M
        # try the zero solution, with zero constant
        w_vec_1 *= 0
        if f(w_vec_1) < best_objective:
            best_w_vec_0 = w_vec_1.copy()
            best_objective = f(best_w_vec_0)
            N_coeffs_removed = len(w_vec_0)
        # try the c__s solution
        try:
            w_vec_1[ -self.M: ] = self.posterior_s.c__s
            if f(w_vec_1) < best_objective:
                best_w_vec_0 = w_vec_1.copy()
                best_objective = f(best_w_vec_0)
                N_coeffs_removed = len(w_vec_0)
        except (AttributeError, ValueError):
            pass
        # announce
        if N_coeffs_removed > 0:
            self.announce( 'LP step: pre-zeroed %d coefficients of `w`' 
                    % N_coeffs_removed, prefix='calc_posterior_w' )
        return best_w_vec_0

    """
    ===============================
    Posterior_w specific attributes
    ===============================

    These should only be computed (and used) for posterior objects.

    """

    @cached
    def Lambdainv_w_vec( _negLP_hessian__w, _is_posterior ):
        if not _is_posterior:
            raise TypeError('attribute only available for `w` posterior.')
        return _negLP_hessian__w

    @cached
    def Lambdainv_w_star( Lambdainv_w_vec, M ):
        return Lambdainv_w_vec[ :-M, :-M ]

    @cached
    def Lambda_w_star( Lambdainv_w_star ):
        return inv( Lambdainv_w_star )

    @cached
    def MAP_H__w( H__w ):
        return H__w

    @cached
    def expected_H__w( H__w ):
        return H__w

    @cached
    def MAP_G__w( G__w ):
        return G__w

    @cached( cskip = ('is_point_estimate', True, 'Lambda_w_star') )
    def expected_G__w( is_point_estimate, G__w, Lambda_w_star ):
        if is_point_estimate:
            return G__w
        else:
            raise NotImplementedError()

    @cached
    def Aw( W_star, Lwi_star ):
        return dot( W_star.T, W_star / Lwi_star[:, na] )

    @cached
    def Aw_inv( Aw ):
        try:
            return inv( Aw )
        except np.linalg.LinAlgError:
            tracer()
            return np.nan * Aw
        
    @cached
    def W_is_degenerate( W ):
        return (W == 0).all()

    @cached
    def logdet_Lambdainv_w_star( Lambdainv_w_star ):
        return logdet( Lambdainv_w_star )

    @cached( cskip = [ ('W_is_degenerate', True, [
        'logdet_Cw_star', 'logdet_Lambdainv_w_star', 'posterior_s', 'Aw'] )] )
    def evidence_components__w( W_is_degenerate, LL_training__w, 
            logdet_Cw_star, logdet_Lambdainv_w_star, posterior_s, Aw ):
        if W_is_degenerate:
            return A([ LL_training__w, 0, 0 ])
        else:
            return A([ 
                LL_training__w,
                -0.5 * ( logdet_Cw_star + logdet_Lambdainv_w_star ),
                -0.5 * np.trace( dot(Aw, posterior_s.As) ) ])

    @cached
    def evidence__w( evidence_components__w ):
        return np.sum( evidence_components__w )

    @cached
    def E_star__w( _is_posterior, d2LP_training__w, rank, M_star,
            posterior_s, Lwi_star, _required_w_star_length ):
        """ Hessian of neg log likelihood, for local evidence approx """
        if not _is_posterior:
            raise TypeError('attribute only available for `w` posterior.')
        # start with hessian
        E = d2LP_training__w.copy()
        # only the components wrt `w_star`
        E = E[ :_required_w_star_length, :_required_w_star_length ]
        # remove the LPrior component
        for i in range(rank):
            for j in range(i, rank):
                # retrieve submatrix
                E_ij = E[i::rank, :][:, j::rank]
                # remove LPrior component
                E_ij[ range(M_star), range(M_star) ] += ( 
                        posterior_s.As[i, j] / Lwi_star )
                # put into container
                E[i::rank, :][:, j::rank] = E_ij
                # symmetry
                if j > i:
                    E[j::rank, :][:, i::rank] = E_ij
        # make it negative
        return -E

    """ Posterior specific, for use by `s` solver """

    @cached
    def YW_training( Y_training, W ):
        return dot( Y_training, W )

    @cached
    def WiWj( W, rank ):
        WiWj = np.empty( (rank, rank), dtype=object )
        for i in range(rank):
            for j in range(i, rank):
                WiWj[i, j] = W[:, i] * W[:, j]
        return WiWj

    """
    ======================
    Local evidence for `w`
    ======================

    At the current `theta_w` (which is a candidate solution, and so is denoted
    `theta_w_c`). Note that `star` here refers to the dim reduction induced 
    by `theta_w_n`, not `theta_w_c`, so dim reduction is not handled by the 
    same attributes as the standard `theta_w`.

    """

    @cached
    def Lwi_c_star( Lwi, posterior_w ):
        """ Freq-domain diagonal cov matrix at candidate theta_w_c. """
        return Lwi[ posterior_w.dims_wi ]

    @cached
    def Lwi_c_star_is_singular( Lwi_c_star, _cutoff_lambda_w ):
        """ Is candidate diagonal cov matrix near-singular. """
        cutoff = _cutoff_lambda_w * maxabs(Lwi_c_star)
        return ( Lwi_c_star < cutoff ).any()

    @cached
    def dims_wi_c_star( Lwi_c_star, _cutoff_lambda_s, Lwi_c_star_is_singular ):
        """ How to dim reduce from star- to plus-space. """
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        else:
            return np.ones( len(Lwi_c_star), dtype=bool )

    @cached
    def dims_wi_c( Lwi_c_star_is_singular, dims_wi_c_star, posterior_w ):
        """ How to dim reduce from full- to plus-space. """
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        else:
            return posterior_w.dims_wi

    @cached
    def Lwi_c_plus( Lwi_c_star, Lwi_c_star_is_singular ):
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        else:
            return Lwi_c_star

    @cached
    def M_plus( dims_wi_c ):
        return np.sum( dims_wi_c )

    @cached
    def E_n_plus__w( posterior_w, Lwi_c_star_is_singular ):
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        else:
            return posterior_w.E_star__w

    @cached
    def Lambdainv_n_plus__w(posterior_w, Lwi_c_star_is_singular):
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        else:
            return posterior_w.Lambdainv_w_star

    @cached
    def W_n_plus( posterior_w, Lwi_c_star_is_singular ):
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        else:
            return posterior_w.W_star

    @cached
    def w_n_plus( W_n_plus ):
        return W_n_plus.flatten()

    """ Approximate posterior at candidate theta """

    @cached
    def Cw_c_plus( Lwi_c_plus, posterior_s ):
        return np.kron( diag(Lwi_c_plus), posterior_s.As_inv )

    @cached
    def Cwinv_c_plus( Lwi_c_plus, posterior_s ):
        return np.kron( diag(1. / Lwi_c_plus), posterior_s.As )

    @cached
    def Lambdainv_c_plus__w( E_n_plus__w, Cwinv_c_plus ):
        return E_n_plus__w + Cwinv_c_plus 

    @cached
    def Lambda_c_plus__w( Lambdainv_c_plus__w ):
        return inv( Lambdainv_c_plus__w )

    @cached
    def w_c_plus( Lambdainv_c_plus__w, Lambdainv_n_plus__w, w_n_plus ):
        try:
            return ldiv( 
                    Lambdainv_c_plus__w, 
                    dot(Lambdainv_n_plus__w, w_n_plus) )
        except np.linalg.LinAlgError:
            tracer()
            return 0 * w_n_plus

    @cached
    def W_c_plus( w_c_plus, M_plus, rank ):
        return w_c_plus.reshape( M_plus, rank )

    @cached
    def W_c( W_c_plus, Cwi_is_diagonal, Lwi_c_star_is_singular,
            M, rank, posterior_w ):
        if not Cwi_is_diagonal:
            raise NotImplementedError()
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        if posterior_w.Rwi_is_identity:
            return W_c_plus
        else:
            W_c = np.zeros( (M, rank) )
            W_c[ posterior_w.dims_wi, : ] = W_c_plus
            return W_c

    @cached
    def H_c__w( posterior_s, W_c, posterior_w ):
        return dot( posterior_s.S, W_c.T ) + posterior_w.c__w[na, :]

    @cached
    def Mu_c__w( H_c__w, expected_FXK_times_known_G ):
        return exp(H_c__w) * expected_FXK_times_known_G

    """ Local evidence at candidate theta """

    @cached
    def local_evidence_components__w( _slice_by_training, Mu_c__w, H_c__w,
            Y_training, Cw_c_plus, E_n_plus__w, M_plus, rank, w_c_plus,
            Cwinv_c_plus ):
        # log likelihood
        sli = _slice_by_training
        LL = np.sum( -sli(Mu_c__w) + sli(H_c__w) * Y_training )
        if len( w_c_plus ) == 0:
            return LL
        # psi(theta)
        psi = A([ 
            LL,
            -0.5 * logdet( dot(Cw_c_plus, E_n_plus__w) + eye(M_plus * rank) ),
            -0.5 * mdot( w_c_plus, Cwinv_c_plus, w_c_plus ) ])
        return psi

    @cached
    def local_evidence__w( local_evidence_components__w ):
        return np.sum( local_evidence_components__w )

    @cached
    def _LE_objective__w( local_evidence__w ):
        return -local_evidence__w

    @cached
    def _LE_jacobian__w( Cwi_is_diagonal, Lwi_c_star_is_singular,
            dLwi_dtheta, dims_wi_c, data, Mu_c__w, 
            _zero_during_testing, posterior_s, Cw_c_plus, 
            E_n_plus__w, Cwinv_c_plus, w_c_plus, Lambda_c_plus__w,
            N_theta_w, Lambdainv_c_plus__w ):
        # implementation
        if not Cwi_is_diagonal:
            raise NotImplementedError()
        if Lwi_c_star_is_singular:
            raise NotImplementedError()
        # project to +-space
        dLwi_dtheta_plus = [ dL[dims_wi_c] for dL in dLwi_dtheta ]
        # residuals at candidate solution
        Resid = data.Y - Mu_c__w
        Resid = _zero_during_testing( Resid ) # only training data
        Resid_plus = Resid[ :, dims_wi_c ]
        # intermediate quantities( all in +-space )
        dLL_dW_c = dot( Resid_plus.T, posterior_s.S )
        dLL_dw_c = dLL_dW_c.flatten()
        C_En = dot( Cw_c_plus, E_n_plus__w )
        Cinv_w = dot( Cwinv_c_plus, w_c_plus )
        Lam_Cinv = dot( Lambda_c_plus__w, Cwinv_c_plus )
        wT_En_minus_Cinv = dot( w_c_plus.T, E_n_plus__w - Cwinv_c_plus )
        # calculate for each variable in theta
        dpsi = np.empty( N_theta_w )
        # derivatives wrt `theta_w` variables
        for j in range( N_theta_w ):
            dLwi = dLwi_dtheta_plus[j]
            dCi = np.kron( diag(dLwi), posterior_s.As_inv )
            B = mdot( Lam_Cinv, dCi, Cwinv_c_plus)
            Bw = ldiv( 
                    Lambdainv_c_plus__w,
                    mdot( Cwinv_c_plus, dCi, Cinv_w ) )
            dpsi[j] = sum([
                dot( dLL_dw_c, Bw ),
                -0.5 * np.trace( dot( B, C_En ) ),
                0.5 * dot( wT_En_minus_Cinv, Bw ) ])
        # make negative
        return -dpsi

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Joint problem for `s` and `w`
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    @property
    def sw_vec( self ):
        return conc([ self.s_star, self.w_vec ])

    @sw_vec.setter
    def sw_vec( self, sw_vec ):
        len_s_star = self._required_s_star_length
        len_w_vec = self._required_w_vec_length
        assert len(sw_vec) == len_s_star + len_w_vec
        # set `s_vec`
        s_star = sw_vec[ :len_s_star ]
        c = sw_vec[ -self.M: ]
        s_vec = conc([ s_star, c ])
        self.csetattr( 's_vec', s_vec )
        # set `w_vec`
        w_vec = sw_vec[ len_s_star: ]
        self.csetattr( 'w_vec', w_vec )

    def _reset_theta_sw( self ):
        self._reset_theta_s()
        self._reset_theta_w()

    def _reset_sw_vec( self ):
        self._reset_s_vec()
        self._reset_w_vec()

    def _initialise_sw_vec( self ):
        self._initialise_s_vec()
        self._initialise_w_vec()

    @property
    def theta_sw( self ):
        return conc([ self.theta_s, self.theta_w ])

    @theta_sw.setter
    def theta_sw( self, theta_sw ):
        theta_s = theta_sw[ :self.N_theta_s ]
        theta_w = theta_sw[ self.N_theta_s: ]
        self.csetattr( 'theta_s', theta_s )
        self.csetattr( 'theta_w', theta_w )


    """
    =============================
    Conditional posterior on `sw`
    =============================
    """

    @cached
    def c__sw( c__w ):
        return c__w

    @cached
    def H__sw( S, W, c__sw ):
        return dot( S, W.T ) + c__sw[na, :]

    @cached
    def G__sw( H__sw ):
        return exp( H__sw )

    @cached
    def Mu__sw( G__sw, expected_FXK_times_known_G ):
        return G__sw * expected_FXK_times_known_G

    @cached
    def LL_training__sw( Mu__sw, _slice_by_training, Y_training, S, W, c__sw ):
        """ Training log likelihood as a function of `sw_vec`. """
        Mu, S = _slice_by_training( Mu__sw ), _slice_by_training( S )
        return ( 
                -np.sum( Mu ) 
                + np.trace( mdot(S.T, Y_training, W) )
                + np.sum( dot(Y_training, c__sw) ) )

    @cached
    def LPrior__sw( S_star, W_star, Lwi_star, Lsi_star ):
        """ Log prior as a function of `sw_vec`. """
        return -0.5 * np.trace( mdot(
            W_star.T, W_star / Lwi_star[:, na],
            S_star.T, S_star / Lsi_star[:, na] ))

    @cached
    def LP_training__sw( LL_training__sw, LPrior__sw ):
        """ Training log posterior as a function of `sw_vec`. """
        return LL_training__sw + LPrior__sw

    """ Jacobian """

    @cached
    def Resid_training__sw( data, Mu__sw, _zero_during_testing ):
        return _zero_during_testing( data.Y - Mu__sw )

    @cached
    def Resid_star__sw( Resid_training__sw, collapse_S, dims_si, 
            Rwi_is_identity, Cwi_is_diagonal ):
        if not Rwi_is_identity or not Cwi_is_diagonal:
            raise NotImplementedError()
        return collapse_S( Resid_training__sw, dims_si, padding='zero' )

    @cached
    def dLL_training__sw( Resid_star__sw, Resid_training__sw, 
            S_star, W, c__sw ):
        """ Jacobian of training log likelihood wrt `sw_vec`. """
        dLL_ds = dot( Resid_star__sw, W ).flatten()
        dLL_dw = dot( Resid_star__sw.T, S_star ).flatten()
        dLL_dc = np.sum( Resid_training__sw, axis=0 )
        return conc([ dLL_ds, dLL_dw, dLL_dc ])

    @cached
    def dLPrior__sw( Lwi_star, Lsi_star, W_star, S_star, M ):
        """ Jacobian of log prior wrt `sw_vec`. """
        LwinvWSTLsinv = dot( 
                W_star / Lwi_star[:, na], S_star.T / Lsi_star[na, :] )
        dLPrior_ds = -dot( LwinvWSTLsinv.T, W_star ).flatten()
        dLPrior_dw = -dot( LwinvWSTLsinv, S_star ).flatten()
        return conc([ dLPrior_ds, dLPrior_dw, zeros(M) ])

    @cached
    def dLP_training__sw( dLL_training__sw, dLPrior__sw ):
        return dLL_training__sw + dLPrior__sw

    """ Hessian """

    @cached
    def d2LP_training__sw( _zero_during_testing, T_star, M_star, M,
            rank, W, S, W_star, S_star, dims_si, Lwi_star, Lsi_star, 
            Rwi_is_identity, Mu__sw, collapse_S, Resid_star__sw ):
        """ Hessian of training log posterior wrt `sw_vec`. """
        import sandwich
        # keep training region only
        Mu = _zero_during_testing( Mu__sw )

        # d2LP / ds^{(i)}.ds^{(j)}
        d2LPs = np.zeros( (T_star * rank, T_star * rank) )
        for i in range(rank):
            for j in range(i, rank):
                # calculate LL component
                Mu_ij = dot( Mu, W[:, i] * W[:, j] )
                Mu_ij_padded = np.hstack([ Mu_ij, np.zeros_like(Mu_ij) ])
                yy = np.fft.fft( Mu_ij_padded )
                #F = ( np.sum(dims_si) - 1 ) / 2
                F = ( np.sum(dims_si) ) / 2 # DC TERM INCLUDED
                d2LPs_ij = -sandwich.calzino( yy.real, yy.imag, F ) / len(yy)
                #d2LPs_ij[ 0, 1: ] /= sqrt(2)
                #d2LPs_ij[ 1:, 0 ] *= sqrt(2)
                d2LPs_ij = d2LPs_ij[ 1:, 1: ] # NO DC
                # add LPrior component
                Aw_ij = dot( W_star[:, i], W_star[:, j] / Lwi_star ) 
                d2LPs_ij[ range(T_star), range(T_star) ] -= (Aw_ij / Lsi_star)
                # put into container
                d2LPs[i::rank, :][:, j::rank] = d2LPs_ij
                # symmetry
                if j > i:
                    d2LPs[j::rank, :][:, i::rank] = d2LPs_ij

        # d2LP / dw^{(i)}.dw^{(j)}
        d2LPw = np.zeros( (M_star * rank, M_star * rank) )
        for i in range(rank):
            for j in range(i, rank):
                # calculate LL component
                d2LPw_ij = -diag( dot( Mu.T, S[:, i] * S[:, j] ) )
                if not Rwi_is_identity:
                    raise NotImplementedError()
                # calculate LPrior component
                As_ij = dot( S_star[:, i], S_star[:, j] / Lsi_star ) 
                d2LPw_ij[ range(M_star), range(M_star) ] -= (As_ij / Lwi_star) 
                # put into container
                d2LPw[i::rank, :][:, j::rank] = d2LPw_ij
                # symmetry
                if j > i:
                    d2LPw[j::rank, :][:, i::rank] = d2LPw_ij

        # d2LP / ds^{(i)}.dw^{(j)}
        d2LPsw = np.zeros( (T_star * rank, M_star * rank) )
        offset = np.zeros( (T_star, M_star) )
        for i in range(rank):
            offset += -( 
                    ( S_star[:, i] / Lsi_star )[:, na] * 
                    ( W_star[:, i] / Lwi_star )[na, :] )
        for i in range(rank):
            for j in range(rank):
                # calculate LL component
                if not Rwi_is_identity:
                    raise NotImplementedError()
                Mu_ij = -Mu * (S[:, j][:, na] * W[:, i][na, :])
                d2LPsw_ij = collapse_S( Mu_ij, dims_si )
                if i == j:
                    d2LPsw_ij += Resid_star__sw
                # calculate LPrior component
                d2LPsw_ij -= ( 
                        ( S_star[:, j] / Lsi_star )[:, na] * 
                        ( W_star[:, i] / Lwi_star )[na, :] )
                if i == j:
                    d2LPsw_ij += offset
                # put into container
                d2LPsw[i::rank, :][:, j::rank] = d2LPsw_ij

        # d2LP / dc^2
        d2LPc = diag( -np.sum( Mu, axis=0 ) )

        # d2LP / ds^{(i)}.dc
        d2LPsc = np.zeros( (T_star * rank, M) )
        for i in range( rank ):
            # calculate
            d2LPsc[i::rank, :] = -collapse_S(Mu * W[:, i][na, :], dims_si)

        # d2LP / dw^{(i)}.dc
        if not Rwi_is_identity:
            raise NotImplementedError()
        d2LPwc = np.zeros( (M_star * rank, M) )
        MTS = dot( Mu.T, S )
        for i in range(rank):
            d2LPwc[i::rank, :][ range(M), range(M) ] = -MTS[:, i]

        # put together
        # [ ss sw sc ]
        # [ ws ww wc ]
        # [ cs cw cc ]
        d2LP = block_diag( d2LPs, d2LPw, d2LPc )
        D0 = slice( None, T_star * rank )
        D1 = slice( T_star * rank, -M )
        D2 = slice( -M, None )
        d2LP[ D0, D1 ] = d2LPsw
        d2LP[ D1, D0 ] = d2LPsw.T
        d2LP[ D0, D2 ] = d2LPsc
        d2LP[ D2, D0 ] = d2LPsc.T
        d2LP[ D1, D2 ] = d2LPwc
        d2LP[ D2, D1 ] = d2LPwc.T
        return d2LP

    """
    =================
    Solution for `sw`
    =================
    """
   
    @cached
    def _negLP_objective__sw( LP_training__sw ):
        return -LP_training__sw

    @cached
    def _negLP_jacobian__sw( dLP_training__sw ):
        return -dLP_training__sw

    @cached
    def _negLP_hessian__sw( d2LP_training__sw ):
        return -d2LP_training__sw

    def _negLP_callback__sw( self, **kw ):
        try:
            offset = self.posterior_k.LPrior__k
        except AttributeError:
            offset = 0
        LP_total = self.LP_training__sw + offset
        self.announce( 'LP step:  %.3f' % LP_total, n_steps=4, **kw )

    """
    ================================
    Posterior_sw specific attributes
    ================================

    These should only be computed (and used) for posterior objects.

    """

    @cached
    def MAP_H__sw( H__sw ):
        return H__sw

    @cached
    def expected_H__sw( H__sw ):
        return H__sw

    @cached
    def MAP_G__sw( G__sw ):
        return G__sw

    @cached
    def expected_G__sw( is_point_estimate, G__sw ):
        if is_point_estimate:
            return G__sw
        else:
            raise NotImplementedError()

    """
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    Solving
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    """

    def calc_posterior_sw( self, **kw ):
        # degenerate case
        if (self.M_star < self.rank) or (self.T_star < self.rank):
            self._calc_degenerate_posterior_sw()
        # normal case
        else:
            self._calc_posterior_v( 'sw', **kw )
        # overwrite posterior_s and posterior_w
        p = self.posterior_sw
        p._is_posterior = False # hack
        p.posterior_s = p
        p.posterior_w = p
        p._is_posterior = True # hack
        self.posterior_s = p
        self.posterior_w = p

    def _calc_degenerate_posterior_sw( self ):
        # NCR: constant term should not be zero...?
        s_star = zeros( self._required_s_star_length )
        w_star = zeros( self._required_w_star_length )
        # BETTER SOLUTION THAN 0?
        #c = zeros( self.M )
        c = self.c__sw
        sw_vec = conc([ s_star, w_star, c ])
        self._create_and_save_posterior_v( 'sw', sw_vec )
        # overwrite posterior_s and posterior_w
        p = self.posterior_sw
        p._is_posterior = False # hack
        p.posterior_s = p
        p.posterior_w = p
        p._is_posterior = True # hack
        self.posterior_s = p
        self.posterior_w = p

    """ Solving: complete. """

    def solve( self, grid_search_theta_s=True, grid_search_theta_w=True,
            grid_search_theta_k=False, verbose=1 ):
        """ Solve for `S/W/c/K` and `theta_s/theta_w/theta_k`. 
        
        Verbosity:
        - 0: none
        - 1: just the major stages
        - 2: theta_s/theta_w/theta_k solvers
        - 3: posterior s/w/k solvers
        - 4: theta_k solvers
        - 5: posterior k solvers
        
        """
        # verbose
        self._announcer.thresh_allow( verbose, 1, 'solve' )
        self._announcer.thresh_allow( verbose, 2, 
                'solve_theta_s', 'solve_theta_w', 'solve_theta_k' )
        self._announcer.thresh_allow( verbose, 3,
                'calc_posterior_s', 'calc_posterior_w', 'calc_posterior_k',
                'calc_posterior_sw' )
        self._announcer.thresh_allow( verbose, 4, 'solve_theta_k_m' )
        for s in self.k_solvers:
            s._announcer.thresh_allow( verbose, 4, 'solve_theta_k' )
            s._announcer.thresh_allow( verbose, 5, 'calc_posterior_k' )

        # extract starting points for first cycle
        self._reset_k_vec()
        self._reset_s_vec()
        self._reset_w_vec()
        # initial estimate of sw
        self.announce( '(0/12) initial estimate of s, w' )
        self.calc_posterior_sw( verbose=None )
        # initial estimate of theta_k, K
        #self.announce( '(1/12) initial estimate of theta_k, k' )
        #self.solve_theta_k( verbose=None )
        # initial estimate of sw
        self.announce( '(2/12) initial re-estimate of s, w')
        self.calc_posterior_sw( verbose=None )
        # solve for theta_s
        self.announce( '(3/12) estimating theta_s, s' )
        self.solve_theta_s( verbose=None, grid_search=True )
        # allowing simple breakout
        if self.posterior_s.S_is_degenerate:
            return self._reset_zero_gain_solution( verbose )
        # solve for theta_w
        self.announce( '(4/12) estimating theta_w, w' )
        self.solve_theta_w( verbose=None, grid_search=True )
        if self.posterior_w.W_is_degenerate:
            return self._reset_zero_gain_solution( verbose )
        # solve for theta_s
        self.announce( '(5/12) estimating theta_s, s' )
        self.solve_theta_s( verbose=None, grid_search=False )
        if self.posterior_s.S_is_degenerate:
            return self._reset_zero_gain_solution( verbose )
        # solve for theta_w
        self.announce( '(6/12) estimating theta_w, w' )
        self.solve_theta_w( verbose=None, grid_search=False )
        if self.posterior_w.W_is_degenerate:
            return self._reset_zero_gain_solution( verbose )
        # re-solve for sw
        self.announce( '(7/12) estimating s & w jointly' )
        self.calc_posterior_sw( verbose=None )
        # solve for theta_k, k
        self.announce( '(8/12) estimating theta_k, k' )
        self.solve_theta_k( verbose=None, grid_search=False )
        # solve for theta_s
        """
        self.announce( '(9/12) estimating theta_s, s' )
        self.solve_theta_s( verbose=None, grid_search=False )
        if self.posterior_s.S_is_degenerate:
            return self._reset_zero_gain_solution( verbose )
        # solve for theta_w
        self.announce( '(10/12) estimating theta_w, w' )
        self.solve_theta_w( 
                verbose=None, grid_search=False )
        if self.posterior_w.W_is_degenerate:
            return self._reset_zero_gain_solution( pk, verbose )
        """
        # re-solve for sw
        self.announce( '(11/12) estimating s & w jointly' )
        self.calc_posterior_sw( verbose=None )
        # re-solve for theta_k, k
        """
        self.announce( '(12/12) estimating theta_k, k' )
        self.solve_theta_k( verbose=None, grid_search=False )
        """
        # restore verbosity
        self._announcer.thresh_allow( verbose, -np.inf, 
                'solve', 'solve_theta_s', 'solve_theta_w', 'solve_theta_k',
                'calc_posterior_s', 'calc_posterior_w', 'calc_posterior_k',
                'calc_posterior_sw' )
        for s in self.k_solvers:
            s._announcer.thresh_allow( verbose, -np.inf, 
                    'solve_theta_k', 'calc_posterior_k' )

    def _reset_zero_gain_solution( self, verbose=None ):
        # set to zero
        self._calc_degenerate_posterior_sw()
        # restore verbosity
        self._announcer.thresh_allow( verbose, -np.inf, 
                'solve', 'solve_theta_s', 'solve_theta_w', 'solve_theta_k',
                'calc_posterior_s', 'calc_posterior_w', 'calc_posterior_k',
                'calc_posterior_sw' )

    """ Derivative checking """

    def _check_LP_derivatives__sw( self, **kw ):
        return self._check_LP_derivatives__v( 'sw', **kw )


    """
    ===========================================
    Cross-validation: for `s` and `w`, jointly
    ===========================================
    """

    @cached
    def LL_testing__sw( Mu__sw, _slice_by_testing, Y_testing, S, W, c__sw ):
        Mu, S = _slice_by_testing( Mu__sw ), _slice_by_testing( S )
        return (
                -np.sum( Mu ) 
                + np.trace( mdot(S.T, Y_testing, W) )
                + np.sum( dot(Y_testing, c__sw) ) )

    @cached
    def LL_training_per_observation__sw( LL_training__sw, N_observations ):
        return LL_training__sw / N_observations

    @cached
    def LL_testing_per_observation__sw(LL_testing__sw, N_observations_testing):
        return LL_testing__sw / N_observations_testing




"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Priors on `S`
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class Prior_s( AutoCR ):

    """ Superclass for priors on `s`. """

    pass


class Lowpass_s( Prior_s ):

    """ Lowpass prior on each `s^{(i)}`. 
    
    As with the ALDf prior, this is defined by a Gaussian power spectrum. 
    Here, the mean is assumed to be zero. The two hyperparameters are the
    standard deviation (`fstd_s`) of the Gaussian, and an additional 
    ridge-like term `rho_s`.

    Note that in the frequency domain, the prior covariance matrix is diagonal.
    For this reason, methods and variables are defined in terms of `Lsi` which 
    is the diagonal of `Csi`.

    For convenience purposes, frequencies are relative to the sampling 
    frequency, i.e. 0 is the DC term, 0.5 is the Nyquist frequency.
    
    """

    # variable names
    _hyperparameter_names_s = [ 'log_fstd_s', 'rho_s' ]
    _bounds_s = [ (-20, -4.5), (-60, 40) ]
    _default_theta_s0 = [ -7, -10 ]

    _grid_search_theta_s_parameters = {
            'bounds':[ [ -16, -7 ], [ -40, 20 ] ], 
            'spacing': [ 'linear', 'linear'], 
            'initial': A([ -9, -10. ]),
            'strategy': '1D two pass', 
            'grid_size': (10, 10) }


    """ Prior on each `s^{(i)}`. """

    @cached
    def Lsi( theta_s, abs_freqs_s ):
        """ Diagonalised prior covariance matrix for `s^{(i)}`. 

        Note that this is defined in rosfft coordinate space, since the
        covariance matrix is diagonalised by the rosfft transform.
        
        """
        log_fs, r = theta_s
        fs = exp(log_fs)
        Lsi = exp( -0.5 / (fs ** 2) * (abs_freqs_s)**2 - r )
        Lsi[0] = 0 # NO DC TERM
        return Lsi

    @cached
    def dLsi_dtheta( theta_s, Lsi, abs_freqs_s ):
        """ Jacobian of the diagonalised prior covariance matrix for `s^{(i)}`.

        Note that this is defined in rosfft coordinate space, since the
        covariance matrix is diagonalised by the rosfft transform.
        
        """
        # parse input
        log_fs, r = theta_s
        fs = exp(log_fs)
        dLsi_dfs = (fs ** -2) * (abs_freqs_s)**2 * Lsi
        # dLsi_dfs[0] = 0 # NO DC TERM - not necessary since Lsi[0] = 0
        dLsi_dr = -Lsi 
        return [ dLsi_dfs, dLsi_dr ]


class Poisson_s( Prior_s ):

    """ No gain. """

    # variable names
    _hyperparameter_names_s = []
    _bounds_s = []
    _default_theta_s0 = []

    @cached
    def Lsi( T ):
        raise TypeError('should not get to this point')

    @cached
    def dLsi_dtheta():
        raise TypeError('should not get to this point')

    def _reproject_to_s_vec( self, *a, **kw ):
        return A([])

    @property
    def _required_s_star_length( self ):
        return 0

    @property
    def _required_w_star_length( self ):
        return 0

    @property
    def H__sw( self ):
        return zeros( (self.T, self.M) )

    @property
    def G__sw( self ):
        return ones( (self.T, self.M) )

    @property
    def MAP_G__sw( self ):
        return self.G__sw

    @property
    def MAP_H__sw( self ):
        return self.H__sw

    @property
    def expected_G__sw( self ):
        return self.G__sw

    @property
    def expected_H__sw( self ):
        return self.H__sw

    def calc_posterior_s( self, **kw ):
        return self._calc_degenerate_posterior_s()

    def calc_posterior_w( self, **kw ):
        return self._calc_degenerate_posterior_w()

    def calc_posterior_sw( self, **kw ):
        # degenerate case
        self._calc_degenerate_posterior_sw()
        # overwrite posterior_s and posterior_w
        p = self.posterior_sw
        p._is_posterior = False # hack
        p.posterior_s = p
        p.posterior_w = p
        p._is_posterior = True # hack
        self.posterior_s = p
        self.posterior_w = p

    @property
    def S_is_degenerate( self ):
        return True

    @property
    def W_is_degenerate( self ):
        return True

    @property
    def S( self ):
        return zeros( (self.T, self.rank) )

    @property
    def W( self ):
        return zeros( (self.M, self.rank) )



"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Priors on `w`
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class Prior_w( AutoCR ):

    """ Superclass for priors on `w`. """

    Cwi_is_diagonal = False


class Diagonal_Prior_w( Prior_w ):

    Cwi_is_diagonal = True


class ML_w( Diagonal_Prior_w ):

    # default variables
    _hyperparameter_names_w = []
    _bounds_w = []
    _default_theta_w0 = []
        
    """ Prior on each `w^{(i)}` """

    @cached
    def Lwi( M ):
        """ (Dummy) prior covariance matrix for `w^{(i)}`. """
        return np.ones( M ) * 1e12


class Ridge_w( Diagonal_Prior_w ):

    # default variables
    _hyperparameter_names_w = [ 'rho_w' ]
    _bounds_w = [ (-80, 20) ]
    _default_theta_w0 = [0.]

    _grid_search_theta_w_parameters = { 
            'bounds':[ [ -70, 10 ] ], 
            'spacing': [ 'linear' ], 
            'initial': A([ 0. ]), 
            'strategy': '1D', 
            'grid_size': (12,) }
        
    """ Prior on `w^{(i)}` """

    @cached
    def Lwi( theta_w, M ):
        """ Diagonal of the prior covariance matrix for `w^{(i)}`. """
        if np.iterable(theta_w):
            rho_w = theta_w[0]
        else:
            rho_w = theta_w
        return exp(-rho_w) * ones( M )

    @cached
    def dLwi_dtheta( Lwi ):
        return [ -Lwi ]
    

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Priors on `k`: Joint
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class Joint_k( AutoCR ):

    """ Solve k jointly, over the whole population, at once. """

    @property
    def k_solvers( self ):
        return []

    """
    ==================
    Initial conditions
    ==================
    """
    
    def _reproject_to_k_vec( self, k=None, posterior=None ):
        """ Calculates `k_star` from `k`. Does not change object's state. 
        
        Can either provide `k` *or* a posterior which contains this attribute.
        
        """
        # check inputs
        if [k, posterior].count(None) != 1:
            raise ValueError('either provide `k` or posterior')
        # provide a posterior
        elif posterior is not None:
            return self._reproject_to_k_vec( k=posterior.k )
        # provide a `k` value
        elif k is not None:
            if self.Rk_is_identity:
                return k
            elif self.Ck_is_diagonal:
                return k[ self.dims_k ]
            else:
                return dot( self.Rk, k )


class ML_k( Joint_k ):

    """ No prior on `k`, i.e. maximum likelihood. """

    # default variables
    _hyperparameter_names_k = []
    _bounds_k = []
    _default_theta_k0 = []
        
    """ Prior on `k` """

    @cached
    def Lk( D, M ):
        """ (Dummy) prior covariance matrix for `K`. """
        return ones( D*M ) * 1e12


class Ridge_k( Joint_k ):

    """ Ridge prior on `k`, with a single hyperparameter for all units.
    
    This prior is defined in terms of a single hyperparameter, `lambda_k`.

    """

    # default variables
    _hyperparameter_names_k = [ 'rho_k' ]
    _bounds_k = [ (-80, 20) ]
    _default_theta_k0 = [-10]
        
    _grid_search_theta_k_parameters = { 
            'bounds':[ [ -70, 10 ] ], 
            'spacing': [ 'linear' ], 
            'initial': A([ -10. ]), 
            'strategy': '1D', 
            'grid_size': (12,) }

    """ Prior on `k` """

    @cached
    def Lk( theta_k, D, M ):
        """ Diagonal of prior covariance matrix for `k`. """
        if np.iterable(theta_k):
            rho_k = theta_k[0]
        else:
            rho_k = theta_k
        return exp( -rho_k ) * ones( D*M )

    @cached
    def dLk_dtheta( Lk ):
        """ Jacobian of diagonal of prior cov matrix on `k`, wrt `theta_k`. """
        return [ -Lk ]


"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Priors on `k`: Factorial
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

class Factorial_k( AutoCR ):

    """ Separate k solution for each unit in the population. 
    
    This is implemented via the attribute `self.k_solvers`. 

    *IMPORTANT* -- this class needs to be subclassed to provide the 
    attribute `_Prior_k_class`. This determines the structure for the
    individual neurons' priors for `k`.
    
    """

    @property
    def _hyperparameter_names_k( self ):
        source_hks = [s._hyperparameter_names_k for s in self.k_solvers]
        final_hks = []
        for i, hks in enumerate(source_hks):
            final_hks += [hk + '_%d' % i for hk in hks]
        return final_hks

    @cached
    def k_solvers( data, M, D, _k_solver_class, training_slices, 
            testing_slices, initial_conditions ):
        # construct data object
        datas = [ mop.Data( data.X, data.Y[:, m],
            normalise=False, whiten=False, add_constant=False )
            for m in range(M) ]
        # prepare initial conditions
        K = initial_conditions.k.reshape( D, M )
        N_theta_ki = len( _k_solver_class._hyperparameter_names_k )
        theta_K = initial_conditions.theta_k.reshape( N_theta_ki, M )
        # construct solvers
        k_solvers = []
        for m in range(M):
            ic = { 'k': K[:, m], 'theta_k':theta_K[:, m] }
            ks = _k_solver_class( datas[m], initial_conditions=ic )
            ks.training_slices = training_slices
            ks.testing_slices = testing_slices
            k_solvers.append(ks)
        return k_solvers

    """ Sugar, for getting/setting from the individual k_solvers """

    @property
    def theta_k( self ):
        return conc([ ks.theta_k for ks in self.k_solvers ])

    @theta_k.setter
    def theta_k( self, new_theta_k ):
        # check input
        if len(new_theta_k) != self.N_theta_k:
            raise ValueError('attempting to set theta_k of wrong length')
        # set the individual values of theta_k
        count = 0
        for m in range( self.M ):
            this_N = self.k_solvers[m].N_theta_k
            this_theta = new_theta_k[ count : count + this_N ]
            self.k_solvers[m].csetattr( 'theta_k', this_theta )
            count += this_N

    @property
    def _required_k_vec_length( self ):
        return sum([ ks._required_k_vec_length for ks in self.k_solvers ])

    @property
    def k_vec( self ):
        return conc([ ks.k_vec for ks in self.k_solvers ])

    @k_vec.setter
    def k_vec( self, new_k_vec ):
        # check input
        if not len(new_k_vec) == self._required_k_vec_length:
            raise ValueError('attempting to set k_vec of wrong length')
        # set the individual values of k_vec
        count = 0
        for m in range( self.M ):
            this_len = self.k_solvers[m]._required_k_vec_length
            this_k_vec = new_k_vec[ count : count + this_len ]
            self.k_solvers[m].csetattr( 'k_vec', this_k_vec )
            count += this_len

    @property
    def K( self ):
        return A([ ks.k for ks in self.k_solvers ]).T

    @property
    def posterior_k( self ):
        solvers = self.k_solvers
        p = Bunch({})
        p.theta_k = conc([ s.posterior_k.theta_k for s in solvers ]) 
        p.K = A([ s.posterior_k.k for s in solvers ]).T
        p.k = p.K.flatten()
        p._is_posterior = True
        p.expected_FXK = A([ 
            s.posterior_k.expected_FXk for s in solvers ]).T
        p.expected_log_FXK = A([ 
            s.posterior_k.expected_log_FXk for s in solvers ]).T
        return p

    @posterior_k.setter
    def posterior_k( self, new_p ):
        eq = np.array_equal
        N_theta_ki = self.k_solvers[0].N_theta_k
        # set each `theta_k`
        for m, s in enumerate( self.k_solvers ):
            if hasattr( new_p, 'theta_k' ):
                start_idx = m * N_theta_ki 
                end_idx = (m + 1) * N_theta_ki
                theta_ki = new_p.theta_k[ start_idx : end_idx ]
                s.posterior_k._is_posterior = False # hack
                s.posterior_k.csetattr( 'theta_k', theta_ki )
                s.posterior_k._is_posterior = True # hack

        # if `k_vec` is provided:
        if hasattr( new_p, 'k_vec' ):
            count = 0
            for m, s in enumerate( self.k_solvers ):
                this_count = s._required_k_vec_length
                this_k_vec = new_p.k_vec[ count : count + this_count ]
                s.posterior_k._is_posterior = False # hack
                s.posterior_k.csetattr( 'k_vec', this_k_vec )
                s.posterior_k._is_posterior = True # hack
                count += this_count
            if not count == len(new_p.k_vec):
                raise IndexError('distribution of `k_vec` failed')

        # if `k` or `K` is provided:
        elif hasattr( new_p, 'k' ) or hasattr( new_p, 'K' ):
            try:
                K = new_p.K
            except AttributeError:
                K = new_p.k.reshape( self.D, self.M )
            for m, s in enumerate( self.k_solvers ):
                this_k_vec = s._reproject_to_k_vec( k=K[:, m] )
                s.posterior_k._is_posterior = False # hack
                s.posterior_k.csetattr( 'k_vec', this_k_vec )
                s.posterior_k._is_posterior = True # hack


    @posterior_k.deleter
    def posterior_k( self ):
        pass

    """
    ==================
    Initial conditions
    ==================
    """

    @property
    def N_theta_k( self ):
        return self.M * len( self._k_solver_class._hyperparameter_names_k )

    @property
    def _default_theta_k0( self ):
        return conc( [ self._k_solver_class._default_theta_k0 ] * self.M )

    def _reproject_to_k_vec( self, k=None, posterior=None ):
        """ Calculates `k_vec` from `k`. Does not change object's state. 
        
        Can either provide `k` *or* a posterior which contains this attribute.
        
        """
        # check inputs
        if [k, posterior].count(None) != 1:
            raise ValueError('either provide `k` or posterior')
        # provide a posterior
        elif posterior is not None:
            return self._reproject_to_k_vec( k=posterior.k )
        # provide a `k` value
        elif k is not None:
            K = k.reshape( (self.D, self.M) )
            return conc([ 
                self.k_solvers[i]._reproject_to_k_vec( K[:, i] ) 
                for i in range(self.M) ])

    def _initialise_k_vec( self ):
        for s in self.k_solvers:
            s._initialise_k_vec()

    def _reset_k_vec( self ):
        for s in self.k_solvers:
            s._reset_k_vec()

    def _initialise_theta_k( self ):
        for s in self.k_solvers:
            s._initialise_theta_k()

    def _reset_theta_k( self ):
        for s in self.k_solvers:
            s._reset_theta_k()

    """
    ==================
    Posterior features
    ==================
    """

    @property
    def expected_FXK( self ):
        return A([ ks.posterior_k.expected_FXk for ks in self.k_solvers ]).T
    
    """
    =======
    Solving
    =======
    """

    def calc_posterior_k( self, verbose=1, **kw ):
        """ Calculate the posterior on `k`, assuming each unit independent.
        
        Verbosity levels:

        - 0 : print nothing
        - 1 : print unit count
        - 2 : print LP status for each
        - None : do nothing
        
        """
        # verbosity
        self._announcer.thresh_allow( verbose, 1, 'calc_posterior_k' )
        self._announcer.thresh_allow( verbose, 2, 'calc_posterior_k_m' )
        for s in self.k_solvers:
            s._announcer.thresh_allow( verbose, 2, 'calc_posterior_k' )
        # retrieve G, H
        if hasattr( self, 'posterior_sw' ):
            H = self.posterior_sw.expected_H__sw
            G = self.posterior_sw.expected_G__sw
        elif hasattr( self, 'posterior_w' ):
            H = self.posterior_w.expected_H__w
            G = self.posterior_w.expected_G__w
        elif hasattr( self, 'posterior_s' ):
            H = self.posterior_s.expected_H__s
            G = self.posterior_s.expected_G__s
        else:
            raise RuntimeError('no posterior on s or w found')
        # add any fixed component
        if self.known_G is not None:
            G = G * self.known_G
            H = H + self.known_H
        # run through each unit in turn
        for i in range( self.M ):
            # announce
            self.announce( 'calculating posterior for unit [%d/%d]' % (
                i, self.M-1 ), prefix='calc_posterior_k_m' )
            # this solver
            s = self.k_solvers[i]
            # set the posterior_h for this unit
            s.posterior_h = Bunch({ 
                'expected_g':G[:, i], 'expected_log_g':H[:, i], 'LPrior__h':0 })
            # set the hyperparameter for this unit
            idx_i = s.N_theta_k * i
            idx_f = idx_i + s.N_theta_k
            theta_k = self.theta_k[ idx_i:idx_f ]
            s.csetattr( 'theta_k', theta_k )
            # calculate
            s.calc_posterior_k( verbose=None, **kw )
        # restore verbosity
        self._announcer.thresh_allow( verbose, -np.inf, 
                'calc_posterior_k', 'calc_posterior_k_m' )
        for s in self.k_solvers:
            s._announcer.thresh_allow( verbose, -np.inf, 'calc_posterior_k' )

    def solve_theta_k( self, verbose=1, **kw ):
        # verbosity
        self._announcer.thresh_allow( verbose, 1, 'solve_theta_k' )
        self._announcer.thresh_allow( verbose, 2, 'solve_theta_k_m' )
        for s in self.k_solvers:
            s._announcer.thresh_allow( verbose, 2, 'solve_theta_k' )
            s._announcer.thresh_allow( verbose, 3, 'calc_posterior_k' )
        # retrieve G, H
        if hasattr( self, 'posterior_sw' ):
            H = self.posterior_sw.expected_H__sw
            G = self.posterior_sw.expected_G__sw
        elif hasattr( self, 'posterior_w' ):
            H = self.posterior_w.expected_H__w
            G = self.posterior_w.expected_G__w
        elif hasattr( self, 'posterior_s' ):
            H = self.posterior_s.expected_H__s
            G = self.posterior_s.expected_G__s
        else:
            raise RuntimeError('no posterior on s or w found')
        # run through each unit in turn
        for i in range( self.M ):
            # announce
            self.announce( 'solving theta_k for unit [%d/%d]' % (
                i, self.M-1 ), prefix='solve_theta_k_m' )
            # this solver
            s = self.k_solvers[i]
            # set the posterior_h for this unit
            s.posterior_h = Bunch({ 
                'expected_g':G[:, i], 'expected_log_g':H[:, i], 'LPrior__h':0 })
            if hasattr( s, 'posterior_k' ):
                s.posterior_k._is_posterior = False # hack
                s.posterior_k.posterior_h = s.posterior_h
                s.posterior_k._is_posterior = True # hack
            # solve
            s.solve_theta_k( verbose=None, **kw )
        # restore verbosity
        self._announcer.thresh_allow( verbose, -np.inf, 
                'solve_theta_k', 'solve_theta_k_m' )
        for s in self.k_solvers:
            s._announcer.thresh_allow( verbose, -np.inf, 
                    'solve_theta_k', 'calc_posterior_k' )


class Factorial_Ridge_k( Factorial_k ):

    _k_solver_class = mop.Ridge


