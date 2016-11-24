from helpers import *
import helpers

np.seterr(all='ignore')



class CueAlone( AutoCR ):

    S = 2

    def __init__( self, d, eps_per_obs=1e-5, max_newton_steps=10, nonlinearity='exp',
            solve=True, verbose=True ):
        self.Y__tn = d.Y__tn
        self.eps_per_obs = eps_per_obs
        self.max_newton_steps = max_newton_steps
        # parse d
        self.Y__tn = d.Y__tn.astype( float )
        self.T = d.N_events
        self.N = d.N_units
        # parse nonlinearity
        self.nonlinearity = nonlinearity
        if nonlinearity not in ['soft', 'exp']:
            self._raise_nonlinearity_error( nonlinearity )
        # initialise and solve
        self.initialise( d )
        if solve:
            self.solve( verbose=verbose )
            
    def _raise_nonlinearity_error( self ):
        err_str = "`nonlinearity` is '%s'; must be 'soft' or 'exp'"
        raise ValueError(err_str % self.nonlinearity)

    @cached
    def eps( eps_per_obs, T, N ):
        return eps_per_obs * T * N

    def initialise( self, d ):
        # extract 
        Y__tn = self.Y__tn
        T, N = self.T, self.N
        # which trial is which 
        S__ts = d.S__ts
        # inital guess for stimulus-driven component
        Nu__sn = ( Y__tn.T.dot(S__ts) / S__ts.sum(axis=0)[None, :] ).T
        Nu__sn[ Nu__sn < 0.01 ] = 0.01
        # save
        if self.nonlinearity == 'exp':
            self.C__sn = log( Nu__sn )
        elif self.nonlinearity == 'soft':
            tracer()
        self.S__ts = S__ts

    """ For a single neuron """

    @cached
    def a__s( z_vec, S ):
        return z_vec[-S:]

    @cached
    def y__t( Y__tn, n ):
        return Y__tn[ :, n ]

    @cached
    def nu__t( S__ts, a__s ):
        return exp( S__ts.dot(a__s) )

    @cached
    def mu__t( nu__t ):
        return nu__t 

    @cached
    def log_mu__t( mu__t ):
        return log( mu__t )

    @cached
    def r__t( y__t, mu__t ):
        r__t = y__t - mu__t
        r__t[ ~np.isfinite(r__t) ] = 0
        return r__t

    """ Objective for neuron n: h(nu) """

    @cached
    def h( mu__t, y__t, log_mu__t ):
        return np.nansum( mu__t - y__t * log_mu__t )

    """ Jacobian: h'(z, nu) """

    @cached
    def dh( r__t, S__ts ):
        # partials
        da = -r__t.dot( S__ts )
        # put together
        return da

    @cached
    def d2h( mu__t, S__ts ):
        # partials
        da2 = diag( mu__t.dot( S__ts ) )
        return da2

    def solve( self, verbose=True ):
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        hessian = self.cfunction( np.array_equal, 'd2h', 'z_vec' )
        # starting values
        last_C__sn = self.C__sn
        C__sn = last_C__sn.copy()
        # run through each cell
        N, S = self.N, self.S
        for n in progress.dots( range(N), 'solving for Z', verbose=verbose ):
            self.n = n
            self.z_vec = C__sn[:, n]
            z_vec = fmin_ncg( objective, self.z_vec, jacobian, fhess=hessian,
                        disp=False, avextol=1e-3 )
            C__sn[ :, n ] = z_vec[-S:]
        # save
        self.C__sn = C__sn

    """ Calculations: overall """

    @cached
    def Mu__tn( C__sn, S__ts ):
        return np.exp( S__ts.dot(C__sn) )

    @cached
    def log_Mu__tn( Mu__tn ):
        return log( Mu__tn )
    
    @cached
    def LL_per_obs( Mu__tn, log_Mu__tn, Y__tn ):
        return np.nansum( -Mu__tn + Y__tn * log_Mu__tn )






"""
==========
Old models
==========
"""


class Naive( CueAlone ):

    """ Superclass for solving """

    def initialise_taus( self, tau ):
        """ Create a list of taus """
        # set up a list
        if np.iterable( tau ):
            taus = np.sort(tau)[::-1]
        else:
            # where to start
            if tau <= 50:
                initial_tau = 100.
            else:
                initial_tau = tau * 10.
            # where to end
            final_tau = tau
            # steps
            N_steps = np.log2( initial_tau / final_tau )
            taus = initial_tau * 0.5 ** np.arange(N_steps)
            taus = conc([ taus, [final_tau] ])
            if final_tau < 1:
                taus = taus[ (taus >= 0.5) | (taus == final_tau) ]
        # save
        self.tau_list = taus.astype(float)
        self.tau = self.tau_list[0]


    def __init__( self, d, tau=1, rank=1, eps_per_obs=1e-5, 
            max_admm_iterations=100, max_newton_steps=10, 
            nonlinearity='exp', nonlinearity_h='exp', 
            G__tn=None, C__sn=None, verbose=True, solve=True,
            save_intermediates=False ):
        # hyperparameters
        self.initialise_taus( tau )
        self.eps_per_obs = eps_per_obs
        self.rank = rank
        self.max_admm_iterations = max_admm_iterations
        self.max_newton_steps = max_newton_steps
        self.save_intermediates = save_intermediates
        if save_intermediates:
            self.intermediate_results = Bunch({})
        # parse d
        self.Y__tn = d.Y__tn.astype( float )
        self.T = d.N_events
        self.N = d.N_units
        # parse gain and cue
        if G__tn is None:
            raise ValueError('must provide `G__tn`')
        if C__sn is None:
            raise ValueError('must provide `C__sn`')
        self.G__tn = G__tn
        self.C__sn = C__sn
        if isinstance( G__tn, int ) or isinstance( G__tn, float ):
            self.G__tn = np.ones( (self.T, self.N) )
            if G__tn != 1:
                self.G__tn *= G__tn
        # parse nonlinearity
        self.nonlinearity = nonlinearity
        if nonlinearity not in ['soft', 'exp']:
            self._raise_nonlinearity_error( nonlinearity )
        self.nonlinearity_h = nonlinearity_h
        if nonlinearity_h not in ['exp', 'soft']:
            raise ValueError("`nonlinearity_h` must be 'soft' or 'exp'")
        # initialise and solve
        self.initialise( d )
        self.N_admm_iterations = 0
        if solve:
            self.solve( verbose=verbose )

    """ Setup """

    def initialise( self, d ):
        # extract 
        Y__tn = self.Y__tn
        C__sn = self.C__sn
        T, N = self.T, self.N
        G__tn = self.G__tn
        # inital guess for stimulus-driven component
        a__n = C__sn.T.dot( A([
            d.is_trial_cue_L.mean(), d.is_trial_cue_R.mean() ]) )
        a__n[ a__n < log(1e-2) ] = log(1e-2)
        nu__n = exp( a__n )
        # initialise latent factor at residuals
        R__tn = Y__tn / ( nu__n[na, :] * G__tn )
        R__tn[ Y__tn == 0 ] = 1.
        if self.nonlinearity == 'exp':
            Z__tn = log( R__tn )
        elif self.nonlinearity == 'soft':
            Z__tn = Finv( R__tn )
        else:
            self._raise_nonlinearity_error()
        Z__tn[ ~np.isfinite(Z__tn) ] = 0
        Z__tn -= Z__tn.mean(axis=0)[na, :]
        # low rank projection
        U__ti, S_ii, V_im = svd( Z__tn, full_matrices=False )
        U__ti = U__ti[ :, :self.rank ]
        S_ii = S_ii[ :self.rank ]
        V_im = V_im[ :self.rank, : ]
        Omega__tn = ( U__ti * S_ii[na, :] ).dot( V_im )
        # Lagrange multiplier
        Lambda__tn = Z__tn - Omega__tn
        # save initial guess
        self.a__n = a__n
        self.Z__tn = Z__tn
        self.Omega__tn = Omega__tn
        self.Lambda__tn = Lambda__tn
        # save last guess
        self.last_a__n = a__n.copy()
        self.last_Z__tn = Z__tn.copy()
        self.last_Omega__tn = Omega__tn.copy()
        self.last_Lambda__tn = Lambda__tn.copy()
        # hyperparameters
        self.rho = 1.

    """ ADMM """

    @cached
    def ADMM_R__tn( Z__tn, Omega__tn ):
        return Z__tn - Omega__tn

    @cached
    def ADMM_S__tn( Omega__tn, last_Omega__tn, rho ):
        return rho * (Omega__tn - last_Omega__tn)

    @cached
    def ADMM_R_frob( ADMM_R__tn ):
        return norm( ADMM_R__tn, 'fro' )

    @cached
    def ADMM_S_frob( ADMM_S__tn ):
        return norm( ADMM_S__tn, 'fro' )

    @cached
    def ADMM_stop( ADMM_R_frob, ADMM_S_frob, eps ):
        return ( ADMM_R_frob <= eps ) and ( ADMM_S_frob <= eps )

    """ For a single neuron """

    @cached
    def nu( a ):
        return exp( a )

    @cached
    def g__t( G__tn, n ):
        return G__tn[ :, n ]

    @cached
    def y__t( Y__tn, n ):
        return Y__tn[ :, n ]

    @cached
    def omega__t( Omega__tn, n ):
        return Omega__tn[ :, n ]

    @cached
    def lambda__t( Lambda__tn, n ):
        return Lambda__tn[ :, n ]

    """ Nonlinearity """

    @cached
    def exp_z__t( z__t ):
        return exp( z__t )

    @cached
    def f__t( exp_z__t, nonlinearity ):
        if nonlinearity == 'exp':
            return exp_z__t
        elif nonlinearity == 'soft':
            return log( 1 + exp_z__t ) / log(2)
        else:
            self._raise_nonlinearity_error()

    @cached
    def df__t( exp_z__t, nonlinearity ):
        if nonlinearity == 'exp':
            return exp_z__t
        elif nonlinearity == 'soft':
            return exp_z__t / (1 + exp_z__t) / log(2)
        else:
            self._raise_nonlinearity_error()

    @cached
    def d2f__t( exp_z__t, nonlinearity ):
        if nonlinearity == 'exp':
            return exp_z__t
        elif nonlinearity == 'soft':
            return exp_z__t / ((1 + exp_z__t)**2) / log(2)
        else:
            self._raise_nonlinearity_error()

    @cached
    def mu__t( nu, f__t ):
        return nu * f__t

    @cached
    def log_mu__t( mu__t ):
        return log( mu__t )
    @cached
    def mu__t( nu, g__t, f__t ):
        return nu * g__t * f__t

    """ Helpers for Jacobian, Hessian """

    @cached( cskip = [ ('nonlinearity', 'exp', ['f__t', 'df__t']) ] )
    def dfonf__t( f__t, df__t, nonlinearity ):
        if nonlinearity == 'exp':
            return 1
        else:
            return df__t / f__t

    @cached( cskip = [ ('nonlinearity', 'exp', ['f__t', 'd2f__t']) ] )
    def d2fonf__t( f__t, d2f__t, nonlinearity ):
        if nonlinearity == 'exp':
            return 1
        else:
            return d2f__t / f__t


    """ Objective for neuron n: h(z, nu) """

    @cached
    def h( mu__t, y__t, log_mu__t, tau, z__t, rho, lambda__t, omega__t ):
        return sum([
            np.nansum( mu__t - y__t * log_mu__t ),
            0.5 * tau * norm( z__t )**2,
            rho * lambda__t.dot( z__t - omega__t ),
            0.5 * rho * norm( z__t - omega__t )**2
        ])

    """ Jacobian: h'(z, nu) """

    @cached
    def dh( r__t, dfonf__t, rho, lambda__t, omega__t, tau, z__t ):
        # partials
        dz = (
            -r__t * dfonf__t +
            rho * (lambda__t - omega__t) +
            (rho + tau) * z__t )
        da = -r__t.sum()
        # put together
        return conc([ dz, [da] ])

    """ Hessian: h''(z, nu) """

    """
    # THESE ARE NOT REQUIRED

    @cached
    def d2h( r__t, d2fonf__t, dfonf__t, y__t, rho, tau, mu__t, T ):
        # partials
        dz2_diag = -r__t * d2fonf__t + y__t * dfonf__t**2 + rho + tau
        dz2_diag[ ~np.isfinite( dz2_diag ) ] = 0
        da2 = mu__t.sum()
        dzda = mu__t * dfonf__t
        # construct
        d2h = np.zeros( (T + 1, T + 1) )
        d2h[ range(T), range(T) ] = dz2_diag
        d2h[ T, T ] = da2
        d2h[ T, :T ] = dzda
        d2h[ :T, T ] = dzda
        return d2h

    @cached
    def d2h_inv( d2h ):
        # blocks of H
        H = d2h
        a = diag(H)[:-1]
        b = H[-1, :-1]
        c = H[-1, -1 ]
        # precomputations
        e = b / a
        d = 1. / ( c - b.dot(e) )
        # container
        Hinv = np.zeros( H.shape )
        T = H.shape[0] - 1
        # start filling in
        Hinv[ :T, :T ] = d * e[na, :] * e[:, na]
        Hinv[ range(T), range(T) ] += 1 / a
        Hinv_12 = -d * e
        Hinv[ :T, -1 ] = Hinv_12
        Hinv[ -1, :T ] = Hinv_12
        Hinv[ -1, -1 ] = d
        return Hinv

    """

    @cached
    def d2h_inv_dh( dh, r__t, dfonf__t, d2fonf__t, y__t, mu__t, rho, tau ):
        # partials for dh
        x = dh[:-1]
        y = dh[-1]
        # partials for d2h
        a = dz2_diag = -r__t * d2fonf__t + y__t * dfonf__t**2 + rho + tau
        a[ ~np.isfinite(a) ] = rho + tau
        b = dzda = mu__t * dfonf__t
        c = da2 = mu__t.sum()
        # precomputations
        e = b / a
        d = 1. / ( c - b.dot(e) )
        f = d * ( e.dot(x) - y )
        # result
        return conc([ x / a + f * e, [-f] ])

    """ Infrastructure """

    @cached
    def z__t( z_vec ):
        return z_vec[:-1]

    @cached
    def a( z_vec ):
        return z_vec[-1]

    """ Calculations: overall """

    @property
    def Z_vec( self ):
        return conc([ self.Z__tn.flatten(), self.a__n.flatten() ])

    @Z_vec.setter
    def Z_vec( self, val ):
        N, T = self.N, self.T
        self.Z__tn = val[ :(T*N) ].reshape( (T, N) )
        self.a__n = val[ (T*N): ]

    @cached
    def exp_Z__tn( Z__tn ):
        return exp( Z__tn )

    @cached
    def F__tn( exp_Z__tn, nonlinearity ):
        if nonlinearity == 'exp':
            return exp_Z__tn
        elif nonlinearity == 'soft':
            return log( 1 + exp_Z__tn ) / log(2)
        else:
            self._raise_nonlinearity_error()

    @cached
    def dF__tn( exp_Z__tn, nonlinearity ):
        if nonlinearity == 'exp':
            return exp_Z__tn
        elif nonlinearity == 'soft':
            return exp_Z__tn / (1 + exp_Z__tn) / log(2)
        else:
            self._raise_nonlinearity_error()

    @cached
    def Mu__tn( nu__n, G__tn, F__tn ):
        return nu__n[na, :] * G__tn * F__tn

    @cached
    def log_Mu__tn( Mu__tn ):
        return log( Mu__tn )

    @cached( cskip = [ ('nonlinearity', 'exp', ['F__tn', 'dF__tn']) ] )
    def dFonF__tn( F__tn, dF__tn, nonlinearity ):
        if nonlinearity == 'exp':
            return 1
        else:
            return dF__tn / F__tn

    @cached
    def R__tn( Y__tn, Mu__tn ):
        R__tn = Y__tn - Mu__tn
        R__tn[ ~np.isfinite(R__tn) ] = 0
        return R__tn

    @cached
    def hall( Mu__tn, Y__tn, log_Mu__tn, tau, Z__tn, rho, Lambda__tn, Omega__tn ):
        return sum([
            np.nansum( Mu__tn - Y__tn * log_Mu__tn ),
            0.5 * tau * norm( Z__tn, 'fro' )**2,
            rho * np.trace( Lambda__tn.T.dot( Z__tn - Omega__tn ) ), 
            0.5 * rho * norm( Z__tn - Omega__tn, 'fro' )**2
        ])

    @cached
    def dhall( R__tn, dFonF__tn, rho, Lambda__tn, Omega__tn, tau, Z__tn ):
        # partials
        dz = (
            -R__tn * dFonF__tn +
            rho * (Lambda__tn - Omega__tn) +
            (rho + tau) * Z__tn ).flatten()
        da = -R__tn.sum(axis=0)
        # put together
        return conc([ dz, da ])

    """ Solving: neuron by neuron """

    def solve_for_next_Z_ncg( self, verbose=True ):
        """ Deprecated. Use Newton steps instead, which are faster. """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        hessian = self.cfunction( np.array_equal, 'd2h', 'z_vec' )
        # starting values
        last_Z__tn = self.Z__tn
        last_a__n = self.a__n
        Z__tn = last_Z__tn.copy()
        a__n = last_a__n.copy()
        # run through each cell
        N = self.N
        for n in progress.dots( range(N), 'solving for Z', verbose=verbose ):
            self.n = n
            self.z_vec = conc([ Z__tn[:, n], [a__n[n]] ])
            z_vec = fmin_ncg( objective, self.z_vec, jacobian, fhess=hessian,
                        disp=False, avextol=1e-3 )
            Z__tn[ :, n ] = z_vec[:-1]
            a__n[ n ] = z_vec[-1]
        # save
        self.last_Z__tn = last_Z__tn
        self.Z__tn = Z__tn
        self.a__n = a__n

    def solve_for_next_Z( self, verbose=True ):
        """ The next primal variable is a local minimum """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        # starting values
        last_Z__tn = self.Z__tn
        last_a__n = self.a__n
        Z__tn = last_Z__tn.copy()
        a__n = last_a__n.copy()
        # run through each cell
        N = self.N
        for n in progress.dots( range(N), 'solving for Z', verbose=verbose ):
            # set the current state to this cell
            self.n = n
            self.z_vec = conc([ Z__tn[:, n], [a__n[n]] ])
            # perform some newton steps
            for repeat_idx in range( self.max_newton_steps ):
                # save where we were before
                old_z_vec = self.z_vec.copy()
                old_h_val = objective( self.z_vec )
                if ~np.isfinite( old_h_val ):
                    # our initial condition is dodgy
                    tracer()
                # make a step
                p_vec = -self.d2h_inv_dh
                alpha = line_search( objective, jacobian, self.z_vec, p_vec )
                # check that we have a valid solution
                new_z_vec = self.z_vec
                new_h_val = objective( self.z_vec )
                if ~np.isfinite( new_h_val ):
                    self.z_vec = old_z_vec
                    break
                if ( ~np.isfinite( new_z_vec ) ).any():
                    # we have ended up in a dodgy place
                    tracer()
                # termination criteria
                if new_h_val - old_h_val > 0:
                    self.z_vec = old_z_vec
                    break
                if new_h_val - old_h_val > -1e-3:
                    break
                if alpha[0] is None:
                    break
            # save the results for this cell
            Z__tn[ :, n ] = self.z_vec[:-1]
            a__n[ n ] = self.z_vec[-1]
        # save overall
        self.last_Z__tn = last_Z__tn
        self.Z__tn = Z__tn
        self.a__n = a__n

    def solve_for_next_Omega( self ):
        """ The next auxiliary variable is an SVD projection """
        self.last_Omega__tn = self.Omega__tn
        U__ti, S_ii, V_im = svd( self.Z__tn, full_matrices=False )
        U__ti = U__ti[ :, :self.rank ]
        S_ii = S_ii[ :self.rank ]
        V_im = V_im[ :self.rank, : ]
        self.Omega__tn = ( U__ti * S_ii[na, :] ).dot( V_im )

    def solve_for_next_Lambda( self ):
        """ The next scaled dual variable """
        self.last_Lambda__tn = self.Lambda__tn
        self.Lambda__tn = self.Lambda__tn + (self.Z__tn - self.Omega__tn)

    def update_rho( self ):
        """ Adjust step size to balance primal and dual """
        ratio = self.ADMM_R_frob / self.ADMM_S_frob
        if ratio > 10:
            self.rho = self.rho * 2.
            self.Lambda__tn = self.Lambda__tn / 2.
        elif ratio < 0.1:
            self.rho = self.rho / 2.
            self.Lambda__tn = self.Lambda__tn * 2.

    def solve( self, verbose=True ):
        # first solve with a large tau
        for tau in self.tau_list:
            if verbose:
                print '*** solving with tau = %d ***' % tau
            self.tau = tau
            self._solve_helper( verbose=verbose )
            if self.save_intermediates:
                self._save_intermediates( tau )

    def _save_intermediates( self, tau ):
        self.intermediate_results[tau] = Bunch({ 
            'Omega__tn' : self.Omega__tn.copy(), 
            'a__n' : self.a__n.copy(),
            'nu__n' : self.nu__n.copy(),
            'LL_per_obs' : self.LL_per_obs })

    def _solve_helper( self, verbose=True ):
        history = []
        for admm_iter in range( self.max_admm_iterations ):
            # update Z
            last_Z__tn = self.Z__tn
            if hasattr( self, 'C__sn' ):
                last_C__sn = self.C__sn
            else:
                last_a__n = self.a__n
            try:
                self.solve_for_next_Z( verbose=False )
            except np.linalg.LinAlgError:
                print '<linalg error, trying ncg>'
                self.Z__tn = last_Z__tn
                if hasattr( self, 'C__sn' ):
                    self.C__sn = last_C__sn
                else:
                    self.a__n = last_a__n
                self.solve_for_next_Z_ncg( verbose=False )
            if ~np.isfinite(self.Z__tn).all():
                Z__tn = self.Z__tn
                Z__tn[ ~np.isfinite(Z__tn) ] = 0
                self.Z__tn = Z__tn
            if hasattr( self, 'C__sn' ):
                if ~np.isfinite( self.C__sn ).all():
                    C__sn = self.C__sn
                    C__sn[ ~np.isfinite( C__sn ) ] = 0
                    self.C__sn = C__sn
            if hasattr( self, 'a__n' ):
                if ~np.isfinite( self.a__n ).all():
                    a__n = self.a__n
                    a__n[ ~np.isfinite( a__n ) ] = 0
                    self.a__n = a__n
            # update Omega
            self.solve_for_next_Omega()
            # update dual variables
            self.solve_for_next_Lambda()
            self.update_rho()
            # add results to history
            self.N_admm_iterations += 1
            current_state = Bunch({
                    'N_admm_iterations' : self.N_admm_iterations,
                    'Z__tn' : self.Z__tn,
                    'Omega__tn' : self.Omega__tn,
                    'Lambda__tn' : self.Lambda__tn,
                    'rho' : self.rho,
                    'R_err' : self.ADMM_R_frob,
                    'S_err' : self.ADMM_S_frob,
                    'total_err' : self.ADMM_R_frob**2 + self.ADMM_S_frob**2 })
            history.append( current_state )
            # verbose
            if verbose:
                self.print_state( current_state )
            # check for convergence
            if self.ADMM_stop:
                break
        # pick the solution with the lowest total error
        best_idx = np.argmin([ h.total_err for h in history ])
        best_state = history[ best_idx ]
        for k in ['N_admm_iterations', 'Z__tn', 'Omega__tn', 'Lambda__tn', 'rho']:
            setattr( self, k, getattr(best_state, k) )
        if verbose:
            print 'best state:'
            self.print_state( best_state )

    def print_state( self, state ):
        s1 = 'step %d:' % state.N_admm_iterations
        s2 = '|R| = %.2f,' % state.R_err
        s3 = '|S| = %.2f,' % state.S_err
        s4 = 'rho = %.1f' % state.rho
        s1 += ' ' * max(12 - len(s1), 0)
        s2 += ' ' * max(15 - len(s2), 0)
        s3 += ' ' * max(15 - len(s3), 0)
        s = s1 + s2 + s3 + s4
        print s

    """ Evaluation """

    @cached
    def nu__n( a__n ):
        return exp( a__n )

    @cached
    def Mu__tn( F__tn, nu__n, G__tn ):
        return nu__n[None, :] * G__tn * F__tn

    @cached
    def LL_per_obs( Mu__tn, Y__tn ):
        log_Mu__tn = log( Mu__tn )
        LL = np.nansum( -Mu__tn + Y__tn * log_Mu__tn )
        return LL / np.isfinite( Y__tn ).sum()




class Naive_Joint( Naive ):

    """ Solve together """

    def solve_for_next_Z( self, verbose=True ):
        """ The next primal variable is a local minimum """
        # construct objective
        objective = self.cfunction( np.array_equal, 'hall', 'Z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dhall', 'Z_vec' )
        # starting values
        last_Z__tn = self.Z__tn.copy()
        last_a__n = self.a__n.copy()
        N, T = self.N, self.T
        # newton steps
        for repeat_idx in range( self.max_newton_steps ):
            # save where we were before
            old_Z_vec = self.Z_vec.copy()
            old_h_val = objective( self.Z_vec )
            # gather the search directions for each cell
            P__tn = np.zeros( (T,N) )
            p__n = np.zeros( N )
            for n in range(N):
                self.n = n
                self.z_vec = conc([ self.Z__tn[:, n], [self.a__n[n]] ])
                P__tn[ :, n ] = -self.d2h_inv_dh[:-1]
                p__n[ n ] = -self.d2h_inv_dh[-1]
            P_vec = conc([ P__tn.flatten(), p__n ])
            # make a step
            alpha = line_search( objective, jacobian, old_Z_vec, P_vec )
            # check that we have a valid solution
            new_Z_vec = self.Z_vec
            new_h_val = objective( self.Z_vec )
            if ~np.isfinite( new_h_val ) or ( new_h_val - old_h_val > 0 ):
                self.Z_vec = old_Z_vec
                break
            # termination criteria
            if new_h_val - old_h_val > -1e-3:
                break
            if alpha[0] is None:
                break
        # save overall
        self.last_Z__tn = last_Z__tn


"""
===================
EXPLICIT CUE MODELS
===================
"""

class Cue( Naive ):

    # number of stimulus conditions
    S = 2

    def initialise( self, d ):
        # extract 
        Y__tn = self.Y__tn
        T, N = self.T, self.N
        G__tn = self.G__tn
        # inital guess for stimulus-driven component
        C__sn, S__ts = self._initialise_A( d )
        self.S__ts = S__ts
        A__tn = S__ts.dot( C__sn )
        Nu__tn = exp( A__tn )
        # initialise latent factor at residuals
        R__tn = Y__tn / ( Nu__tn * G__tn )
        R__tn[ Y__tn == 0 ] = 1.
        if self.nonlinearity == 'exp':
            Z__tn = log( R__tn )
        elif self.nonlinearity == 'soft':
            Z__tn = Finv( R__tn )
        else:
            self._raise_nonlinearity_error()
        Z__tn[ ~np.isfinite(Z__tn) ] = 0
        Z__tn -= Z__tn.mean(axis=0)[na, :]
        # low rank projection
        U__ti, S_ii, V_im = svd( Z__tn, full_matrices=False )
        U__ti = U__ti[ :, :self.rank ]
        S_ii = S_ii[ :self.rank ]
        V_im = V_im[ :self.rank, : ]
        Omega__tn = ( U__ti * S_ii[na, :] ).dot( V_im )
        # Lagrange multiplier
        Lambda__tn = Z__tn - Omega__tn
        # save initial guess
        self.C__sn = C__sn
        self.Z__tn = Z__tn
        self.Omega__tn = Omega__tn
        self.Lambda__tn = Lambda__tn
        # save last guess
        self.last_C__sn = C__sn.copy()
        self.last_Z__tn = Z__tn.copy()
        self.last_Omega__tn = Omega__tn.copy()
        self.last_Lambda__tn = Lambda__tn.copy()
        # hyperparameters
        self.rho = 1.

    def _initialise_A( self, d ):
        """ Initial guess for stim-driven component, and its regressors. """
        # inital guess for stimulus-driven component
        C__sn = self.C__sn.copy()
        C__sn[ C__sn < log(1e-2) ] = log(1e-2)
        # how to convert back to time domain
        S__ts = d.S__ts
        return C__sn, S__ts

    """ For a single neuron """

    @cached
    def nu__t( S__ts, a__s ):
        return exp( S__ts.dot(a__s) )

    @cached
    def mu__t( nu__t, g__t, f__t ):
        return nu__t * g__t * f__t

    @cached
    def dh( r__t, dfonf__t, rho, lambda__t, omega__t, tau, z__t, S__ts ):
        # partials
        dz = (
            -r__t * dfonf__t +
            rho * (lambda__t - omega__t) +
            (rho + tau) * z__t )
        da = -r__t.dot( S__ts )
        # put together
        return conc([ dz, da ])

    @cached
    def d2h( r__t, d2fonf__t, dfonf__t, y__t, rho, tau, mu__t, T, S__ts, S ):
        # partials
        dz2_diag = -r__t * d2fonf__t + y__t * dfonf__t**2 + rho + tau
        dz2_diag[ ~np.isfinite( dz2_diag ) ] = rho + tau
        da2 = diag( mu__t.dot( S__ts ) )
        dzda = ( mu__t * dfonf__t )[:, na] * S__ts
        # construct
        d2h = np.zeros( (T + S, T + S) )
        d2h[ range(T), range(T) ] = dz2_diag
        d2h[ T:, T: ] = da2
        d2h[ T:, :T ] = dzda.T
        d2h[ :T, T: ] = dzda
        return d2h

    @cached
    def d2h_inv( d2h, S ):
        inv = np.linalg.inv
        # blocks of H
        H = d2h
        T = len(H) - S
        a = diag(H)[:-S]
        C = H[:-S, -S:]
        b = diag(H)[-S:]
        # precomputations
        ainv = 1. / a
        AinvC = ainv[:, na] * C
        E = inv( diag(b) - C.T.dot(AinvC) )
        #X = diag(ainv) + AinvC.dot(E).dot(AinvC.T)
        X = AinvC.dot(E).dot(AinvC.T)
        X[ range(T), range(T) ] += ainv
        Y = -AinvC.dot(E)
        U = E
        # start filling in
        Hinv = np.zeros( H.shape )
        Hinv[ :T, :T ] = X
        Hinv[ T:, :T ] = Y.T
        Hinv[ :T, T: ] = Y
        Hinv[ T:, T: ] = U
        return Hinv

    @cached
    def d2h_inv_dh( dh, r__t, d2fonf__t, dfonf__t, y__t, rho, tau, mu__t, S__ts, S ):
        # partials
        a = dz2_diag = -r__t * d2fonf__t + y__t * dfonf__t**2 + rho + tau
        a[ ~np.isfinite( a ) ] = rho + tau
        b = da2 = mu__t.dot( S__ts )
        C = dzda = ( mu__t * dfonf__t )[:, na] * S__ts
        # precomputations
        ainv = 1. / a
        AinvC = ainv[:, na] * C
        Einv = diag(b) - C.T.dot(AinvC)
        # blocks of dh
        x = dh[:-S]
        y = dh[-S:]
        x_on_a = x / a
        f = np.linalg.solve( Einv, C.T.dot(x_on_a) - y )
        # solution
        return conc([ x_on_a + AinvC.dot(f), -f ])

    @cached
    def z__t( z_vec, S ):
        return z_vec[:-S]

    @cached
    def a__s( z_vec, S ):
        return z_vec[-S:]

    @property
    def Z_vec( self ):
        return conc([ self.Z__tn.flatten(), self.C__sn.flatten() ])

    @Z_vec.setter
    def Z_vec( self, val ):
        N, T, S = self.N, self.T, self.S
        self.Z__tn = val[ :(T*N) ].reshape( (T, N) )
        self.C__sn = val[ (T*N): ].reshape( (S, N) )

    @cached
    def Nu__sn( C__sn ):
        return np.exp( C__sn )

    @cached
    def Mu__tn( Nu__sn, S__ts, G__tn, F__tn ):
        return S__ts.dot( Nu__sn ) * G__tn * F__tn

    @cached
    def dhall( R__tn, dFonF__tn, rho, Lambda__tn, Omega__tn, tau, Z__tn, S__ts ):
        # partials
        dz = (
            -R__tn * dFonF__tn +
            rho * (Lambda__tn - Omega__tn) +
            (rho + tau) * Z__tn ).flatten()
        da = -S__ts.T.dot( R__tn ).flatten()
        # put together
        return conc([ dz, da ])

    def solve_for_next_Z( self, verbose=True ):
        #return self.solve_for_next_Z_ncg( verbose=verbose )
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        # starting values
        last_Z__tn = self.Z__tn
        last_C__sn = self.C__sn
        Z__tn = last_Z__tn.copy()
        C__sn = last_C__sn.copy()
        # run through each cell
        N, S = self.N, self.S
        for n in progress.dots( range(N), 'solving for Z', verbose=verbose ):
            # set the current state to this cell
            self.n = n
            self.z_vec = conc([ Z__tn[:, n], C__sn[:, n] ])
            # perform some newton steps
            for repeat_idx in range( self.max_newton_steps ):
                # save where we were before
                old_z_vec = self.z_vec.copy()
                old_h_val = objective( self.z_vec )
                if ~np.isfinite( old_h_val ):
                    # our initial condition is dodgy
                    tracer()
                # make a step
                p_vec = -self.d2h_inv_dh
                alpha = line_search( objective, jacobian, self.z_vec, p_vec )
                # check that we have a valid solution
                new_z_vec = self.z_vec
                new_h_val = objective( self.z_vec )
                if ~np.isfinite( new_h_val ):
                    self.z_vec = old_z_vec
                    break
                if ( ~np.isfinite( new_z_vec ) ).any():
                    # we have ended up in a dodgy place
                    tracer()
                # termination criteria
                if new_h_val - old_h_val > 0:
                    self.z_vec = old_z_vec
                    break
                if new_h_val - old_h_val > -1e-3:
                    break
                if alpha[0] is None:
                    break
            # save the results for this cell
            Z__tn[ :, n ] = self.z_vec[:-S]
            C__sn[ :, n ] = self.z_vec[-S:]
        # save overall
        self.last_Z__tn = last_Z__tn
        self.last_C__sn = last_C__sn
        self.Z__tn = Z__tn
        self.C__sn = C__sn

    def solve_for_next_Z_ncg( self, verbose=True ):
        """ Deprecated. Use Newton steps instead, which are faster. """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        hessian = self.cfunction( np.array_equal, 'd2h', 'z_vec' )
        # starting values
        last_Z__tn = self.Z__tn
        last_C__sn = self.C__sn
        Z__tn = last_Z__tn.copy()
        C__sn = last_C__sn.copy()
        # run through each cell
        N, S = self.N, self.S
        for n in progress.dots( range(N), 'solving for Z', verbose=verbose ):
            self.n = n
            self.z_vec = conc([ Z__tn[:, n], C__sn[:, n] ])
            z_vec = fmin_ncg( objective, self.z_vec, jacobian, fhess=hessian,
                        disp=False, avextol=1e-3 )
            Z__tn[ :, n ] = z_vec[:-S]
            C__sn[ :, n ] = z_vec[-S:]
        # save
        self.last_Z__tn = last_Z__tn
        self.Z__tn = Z__tn
        self.C__sn = C__sn

    """ Evaluation """

    @cached
    def Nu__tn( C__sn, S__ts ):
        return exp( S__ts.dot( C__sn ) )

    @cached
    def Mu__tn( Nu__tn, G__tn, F__tn ):
        return Nu__tn * G__tn * F__tn

    def _save_intermediates( self, tau ):
        self.intermediate_results[tau] = Bunch({ 
            'Omega__tn' : self.Omega__tn.copy(), 
            'C__sn' : self.C__sn.copy(),
            'Nu__sn' : self.Nu__sn.copy(),
            'LL_per_obs' : self.LL_per_obs })

class CueJoint( Cue ):

    def solve_for_next_Z( self, verbose=True ):
        """ The next primal variable is a local minimum """
        # construct objective
        objective = self.cfunction( np.array_equal, 'hall', 'Z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dhall', 'Z_vec' )
        # starting values
        last_Z__tn = self.Z__tn.copy()
        last_C__sn = self.C__sn.copy()
        N, T, S = self.N, self.T, self.S
        # newton steps
        for repeat_idx in range( self.max_newton_steps ):
            # save where we were before
            old_Z_vec = self.Z_vec.copy()
            old_h_val = objective( self.Z_vec )
            # gather the search directions for each cell
            P__tn = np.zeros( (T,N) )
            R__sn = np.zeros( (S,N) )
            for n in range(N):
                self.n = n
                self.z_vec = conc([ self.Z__tn[:, n], self.C__sn[:, n] ])
                P__tn[ :, n ] = -self.d2h_inv_dh[:-S]
                R__sn[ :, n ] = -self.d2h_inv_dh[-S:]
            P_vec = conc([ P__tn.flatten(), R__sn.flatten() ])
            # make a step
            alpha = line_search( objective, jacobian, old_Z_vec, P_vec )
            # check that we have a valid solution
            new_Z_vec = self.Z_vec
            new_h_val = objective( self.Z_vec )
            if ~np.isfinite( new_h_val ) or ( new_h_val - old_h_val > 0 ):
                self.Z_vec = old_Z_vec
                break
            # termination criteria
            if new_h_val - old_h_val > -1e-3:
                break
            if alpha[0] is None:
                break
        # save overall
        self.last_Z__tn = last_Z__tn
        self.last_C__sn = last_C__sn



"""
===================
EXPLICIT SEQ MODELS
===================
"""

class Seq( Cue ):

    S = 22

    def _initialise_A( self, d ):
        """ Initial guess for stim-driven component, and its regressors. """
        # inital guess for stimulus-driven component
        C__sn = np.hstack([ 
            np.repeat( self.C__sn[0, :][:, None], 11, 1 ),
            np.repeat( self.C__sn[1, :][:, None], 11, 1 ) ]).T
        C__sn[ C__sn < log(1e-2) ] = log(1e-2)
        # how to convert back to time domain
        S__ts = np.vstack([ 
            [ (d.is_trial_cue_L & (d.event_idxs_within_trial == i)) 
                for i in range(1, 12) ],
            [ (d.is_trial_cue_R & (d.event_idxs_within_trial == i)) 
                for i in range(1, 12) ] ]).T.astype(float)
        return C__sn, S__ts


"""
===================
NoCue and SharedCue
===================
"""
class NoCue( AutoCR ):

    def __init__( self, d, G__tn, a0__n, solve=True, verbose=False ):
        self.Y__tn = d.Y__tn
        # parse d
        self.Y__tn = d.Y__tn.astype( float )
        self.y__j = self.Y__tn.flatten()
        self.T = d.N_events
        self.N = d.N_units
        self.S__ts = d.S__ts
        # set initial values
        self.G__tn = G__tn
        self.z_vec = a0__n
        # initialise and solve
        if solve:
            self.solve( verbose=verbose )
            
    @cached
    def a__n( z_vec ):
        return z_vec
    
    @cached
    def C__sn( a__n ):
        return A([ a__n, a__n ])
    
    @cached
    def Mu__tn( C__sn, S__ts, G__tn ):
        return np.exp( S__ts.dot(C__sn) ) * G__tn
    
    @cached
    def mu__j( Mu__tn ):
        return Mu__tn.flatten()
    
    @cached
    def log_mu__j( mu__j ):
        return np.log( mu__j )
    
    @cached
    def r__j( y__j, mu__j ):
        r__j = y__j - mu__j
        r__j[ ~np.isfinite(r__j) ] = 0
        return r__j
    
    @cached
    def h( mu__j, y__j, log_mu__j ):
        return np.nansum( mu__j - y__j * log_mu__j )
    
    def solve( self, verbose=True ):
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        z_vec = fmin_l_bfgs_b( objective, self.z_vec, approx_grad=True, disp=False )


class SharedCue( NoCue ):

    def __init__( self, d, G__tn, w__n, a0__n, c0, solve=True, verbose=True ):
        self.Y__tn = d.Y__tn
        # parse d
        self.Y__tn = d.Y__tn.astype( float )
        self.y__j = self.Y__tn.flatten()
        self.T = d.N_events
        self.N = d.N_units
        self.S__ts = d.S__ts
        # set initial values
        self.G__tn = G__tn
        self.w__n = w__n
        self.z_vec = conc([ a0__n, [c0] ])
        # initialise and solve
        if solve:
            self.solve( verbose=verbose )
        
    @cached
    def a__n( z_vec ):
        return z_vec[:-1]
    
    @cached
    def c( z_vec ):
        return z_vec[-1]
    
    @cached
    def C__sn( a__n, w__n, c ):
        return A([ a__n, a__n + c*w__n ])
    

