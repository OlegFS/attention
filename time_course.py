from helpers import *
import helpers

np.seterr(all='ignore')


class TimeCourse( AutoCR ):

    """ Class for solving time course: ADMM on tm / iterate for i """

    def __init__( self, d, Y__tni, omega__i, Omega0__tn, a__ni, G__tn,
            tau=1, rank=1, eps_per_obs=1e-5, 
            max_admm_iterations=100, max_newton_steps=10, 
            verbose=True, solve=True ):
        # hyperparameters
        self.tau = float(tau)
        self.eps_per_obs = eps_per_obs
        self.rank = rank
        self.max_admm_iterations = max_admm_iterations
        self.max_newton_steps = max_newton_steps
        # parse d
        self.T = d.N_events
        self.M = d.N_units
        # parse other inputs
        self.Y__tni = Y__tni
        self.omega__i = omega__i
        self.Omega0__tn = Omega0__tn
        self.a__ni = a__ni
        self.G__tn = G__tn
        # initialise and solve
        self.initialise( d )
        self.N_admm_iterations = 0
        if solve:
            self.solve( verbose=verbose )

    @cached
    def I( omega__i ):
        return len(omega__i)

    @cached
    def eps( eps_per_obs, T, M, I ):
        return eps_per_obs * T * M * I

    """ Setup """

    def initialise( self, d ):
        # extract 
        Y__tni = self.Y__tni
        T, M, I = self.T, self.M, self.I
        G__tn = self.G__tn
        # inital guess for stimulus-driven component
        a__ni = self.a__ni
        nu__ni = np.exp( a__ni )
        # initialise latent factor at last guess
        Z__tn = self.Omega0__tn.copy()
        # low rank projection
        U__ti, S_ii, V_im = svd( Z__tn, full_matrices=False )
        U__ti = U__ti[ :, :self.rank ]
        S_ii = S_ii[ :self.rank ]
        V_im = V_im[ :self.rank, : ]
        Omega__tn = ( U__ti * S_ii[na, :] ).dot( V_im )
        # Lagrange multiplier
        Lambda__tn = Z__tn - Omega__tn
        # save initial guess
        self.a__ni = a__ni
        self.Z__tn = Z__tn
        self.Omega__tn = Omega__tn
        self.Lambda__tn = Lambda__tn
        # save last guess
        self.last_a__ni = a__ni.copy()
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
    def nu__i( a__i ):
        return exp( a__i )

    @cached
    def g__t( G__tn, m ):
        return G__tn[ :, m ]

    @cached
    def y__ti( Y__tni, m ):
        return Y__tni[ :, m, : ]

    @cached
    def omega__t( Omega__tn, m ):
        return Omega__tn[ :, m ]

    @cached
    def lambda__t( Lambda__tn, m ):
        return Lambda__tn[ :, m ]

    """ Nonlinearity """

    @cached
    def f__ti( z__t, omega__i ):
        return exp( z__t[:, na] * omega__i[na, :] )

    @cached
    def df__ti( f__ti, omega__i ):
        return omega__i[na, :] * f__ti

    @cached
    def d2f__ti( f__ti, omega__i ):
        return (omega__i ** 2)[na, :] * f__ti

    @cached
    def mu__ti( nu__i, g__t, f__ti ):
        return nu__i[na, :] * g__t[:, na] * f__ti

    @cached
    def log_mu__ti( mu__ti ):
        return log( mu__ti )

    """ Helpers for Jacobian, Hessian """

    @cached
    def dfonf__ti( omega__i ):
        return omega__i[na, :]

    @cached
    def d2fonf__ti( omega__i ):
        return (omega__i ** 2)[na, :]

    @cached
    def r__ti( y__ti, mu__ti ):
        r__ti = y__ti - mu__ti
        r__ti[ ~np.isfinite(r__ti) ] = 0
        return r__ti

    """ Objective for neuron n: h(z, nu) """

    @cached
    def h( mu__ti, y__ti, log_mu__ti, tau, z__t, rho, lambda__t, omega__t ):
        return sum([
            np.nansum( mu__ti - y__ti * log_mu__ti ),
            0.5 * tau * norm( z__t )**2,
            rho * lambda__t.dot( z__t - omega__t ),
            0.5 * rho * norm( z__t - omega__t )**2
        ])

    """ Jacobian: h'(z, nu) """

    @cached
    def dh( r__ti, dfonf__ti, rho, lambda__t, omega__t, tau, z__t ):
        # partials
        dz = (
            -(r__ti * dfonf__ti).sum( axis=1 ) +
            rho * (lambda__t - omega__t) +
            (rho + tau) * z__t )
        da = -r__ti.sum(axis=0)
        # put together
        return conc([ dz, da ])

    """ Newton step """

    @cached
    def d2h_inv_dh( dh, r__ti, dfonf__ti, d2fonf__ti, y__ti, mu__ti, rho, tau, I ):
        # partials
        a = dz2_diag = (
                -r__ti * d2fonf__ti + y__ti * dfonf__ti**2 ).sum(axis=1)
        a[ ~np.isfinite( a ) ] = 0
        a = a + rho + tau
        b = da2 = mu__ti.sum( axis=0 )
        C = dzda = mu__ti * dfonf__ti
        # precomputations
        ainv = 1. / a
        AinvC = ainv[:, na] * C
        Einv = diag(b) - C.T.dot(AinvC)
        # blocks of dh
        x = dh[:-I]
        y = dh[-I:]
        x_on_a = x / a
        f = np.linalg.solve( Einv, C.T.dot(x_on_a) - y )
        # solution
        return conc([ x_on_a + AinvC.dot(f), -f ])

    """ Infrastructure """

    @cached
    def z__t( z_vec, I ):
        return z_vec[:-I]

    @cached
    def a__i( z_vec, I ):
        return z_vec[-I:]

    """ Solving: neuron by neuron """

    def solve_for_next_Z_bfgs( self, verbose=True ):
        """ Deprecated. Use Newton steps instead, which are faster. """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        objective_2 = lambda x: objective( x.copy() )
        jacobian_2 = lambda x: jacobian( x.copy() )
        # starting values
        last_Z__tn = self.Z__tn
        last_a__ni = self.a__ni
        Z__tn = last_Z__tn.copy()
        a__ni = last_a__ni.copy()
        # run through each cell
        M, I = self.M, self.I
        for m in progress.dots( range(M), 'solving for Z', verbose=verbose ):
            self.m = m
            self.z_vec = conc([ Z__tn[:, m], a__ni[m, :] ])
            z_vec = fmin_l_bfgs_b( 
                    objective_2, self.z_vec, fprime=jacobian_2, 
                    approx_grad=False )
            z_vec = z_vec[0]
            Z__tn[ :, m ] = z_vec[:-I]
            a__ni[ m, : ] = z_vec[-I:]
        # save
        self.last_Z__tn = last_Z__tn
        self.Z__tn = Z__tn
        self.a__ni = a__ni
    
    def solve_for_next_Z( self, verbose=True ):
        """ The next primal variable is a local minimum """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        # starting values
        last_Z__tn = self.Z__tn
        last_a__ni = self.a__ni
        Z__tn = last_Z__tn.copy()
        a__ni = last_a__ni.copy()
        # run through each cell
        M, I = self.M, self.I
        for m in progress.dots( range(M), 'solving for Z', verbose=verbose ):
            # set the current state to this cell
            self.m = m
            self.z_vec = conc([ Z__tn[:, m], a__ni[m, :] ])
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
            Z__tn[ :, m ] = self.z_vec[:-I]
            a__ni[ m, : ] = self.z_vec[-I:]
        # save overall
        self.last_Z__tn = last_Z__tn
        self.Z__tn = Z__tn
        self.a__ni = a__ni

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
        tau = self.tau
        if verbose:
            print '*** solving with tau = %d ***' % tau
        self._solve_helper( verbose=verbose )

    def _solve_helper( self, verbose=True ):
        history = []
        for admm_iter in range( self.max_admm_iterations ):
            # update Z
            last_Z__tn = self.Z__tn
            last_a__ni = self.a__ni
            try:
                self.solve_for_next_Z( verbose=False )
            except np.linalg.LinAlgError:
                print '<linalg error, trying bfgs>'
                self.Z__tn = last_Z__tn
                self.a__ni = last_a__ni
                self.solve_for_next_Z_bfgs( verbose=True )
            if ~np.isfinite(self.Z__tn).all():
                Z__tn = self.Z__tn
                Z__tn[ ~np.isfinite(Z__tn) ] = 0
                self.Z__tn = Z__tn
            if ~np.isfinite( self.a__ni ).all():
                a__ni = self.a__ni
                a__ni[ ~np.isfinite( a__ni ) ] = 0
                self.a__ni = a__ni
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
                    'a__ni' : self.a__ni,
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
        for k in ['N_admm_iterations', 'Z__tn', 'Omega__tn', 'Lambda__tn', 'rho',
                'a__ni']:
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
    def nu__ni( a__ni ):
        return exp( a__ni )

    @cached
    def LL_per_obs( Z__tn, omega__i, a__ni, Y__tni, G__tn, nonlinearity ):
        F__tni = exp( Z__tn[:, :, na] * omega__i[na, na, :] )
        Mu__tni = exp( a__ni[na, :, :] ) * G__tn[:, :, na] * F__tni
        log_Mu__tni = log( Mu__tni )
        LL = np.nansum( -Mu__tni + Y__tni * log_Mu__tni )
        return LL / np.isfinite( Y__tni ).sum()



class TimeCourse__ti_m( AutoCR ):

    """ Class for solving time course: ADMM on ti / iterate for m """

    def __init__( self, d, Y__tni, omega__n, Omega0__ti, a__ni,
            tau=1, rank=1, eps_per_obs=1e-5, 
            max_admm_iterations=100, max_newton_steps=10, 
            verbose=True, solve=True ):
        # hyperparameters
        self.tau = float(tau)
        self.eps_per_obs = eps_per_obs
        self.rank = rank
        self.max_admm_iterations = max_admm_iterations
        self.max_newton_steps = max_newton_steps
        # parse d
        self.T = d.N_events
        self.M = d.N_units
        # parse other inputs
        self.Y__tni = Y__tni
        self.omega__n = omega__n
        self.Omega0__ti = Omega0__ti
        self.a__ni = a__ni
        # parse nonlinearity
        self.G__tn = exp( d.H__tn )
        # initialise and solve
        self.initialise( d )
        self.N_admm_iterations = 0
        if solve:
            self.solve( verbose=verbose )

    @cached
    def I( Y__tni ):
        return shape( Y__tni )[-1]

    @cached
    def eps( eps_per_obs, T, M, I ):
        return eps_per_obs * T * M * I

    """ Setup """

    def initialise( self, d ):
        # extract 
        Y__tni = self.Y__tni
        T, M, I = self.T, self.M, self.I
        G__tn = self.G__tn
        # inital guess for stimulus-driven component
        a__ni = self.a__ni
        nu__ni = np.exp( a__ni )
        # initialise latent factor at last guess
        Z__ti = self.Omega0__ti.copy()
        # low rank projection
        U__ti, S_ii, V_im = svd( Z__ti, full_matrices=False )
        U__ti = U__ti[ :, :self.rank ]
        S_ii = S_ii[ :self.rank ]
        V_im = V_im[ :self.rank, : ]
        Omega__ti = ( U__ti * S_ii[na, :] ).dot( V_im )
        # Lagrange multiplier
        Lambda__ti = Z__ti - Omega__ti
        # save initial guess
        self.a__ni = a__ni
        self.Z__ti = Z__ti
        self.Omega__ti = Omega__ti
        self.Lambda__ti = Lambda__ti
        # save last guess
        self.last_a__ni = a__ni.copy()
        self.last_Z__ti = Z__ti.copy()
        self.last_Omega__ti = Omega__ti.copy()
        self.last_Lambda__ti = Lambda__ti.copy()
        # hyperparameters
        self.rho = 1.

    """ ADMM """

    @cached
    def ADMM_R__ti( Z__ti, Omega__ti ):
        return Z__ti - Omega__ti

    @cached
    def ADMM_S__ti( Omega__ti, last_Omega__ti, rho ):
        return rho * (Omega__ti - last_Omega__ti)

    @cached
    def ADMM_R_frob( ADMM_R__ti ):
        return norm( ADMM_R__ti, 'fro' )

    @cached
    def ADMM_S_frob( ADMM_S__ti ):
        return norm( ADMM_S__ti, 'fro' )

    @cached
    def ADMM_stop( ADMM_R_frob, ADMM_S_frob, eps ):
        return ( ADMM_R_frob <= eps ) and ( ADMM_S_frob <= eps )

    """ For a single time step """

    @cached
    def nu__n( a__n ):
        return exp( a__n )

    @cached
    def y__tn( Y__tni, i ):
        return Y__tni[ :, :, i ]

    @cached
    def omega__t( Omega__ti, i ):
        return Omega__ti[ :, i ]

    @cached
    def lambda__t( Lambda__ti, i ):
        return Lambda__ti[ :, i ]

    """ Nonlinearity """

    @cached
    def f__tn( z__t, omega__n ):
        return exp( z__t[:, na] * omega__n[na, :] )

    @cached
    def df__tn( f__tn, omega__n ):
        return omega__n[na, :] * f__tn

    @cached
    def d2f__tn( f__tn, omega__n ):
        return (omega__n ** 2)[na, :] * f__tn

    @cached
    def mu__tn( nu__n, G__tn, f__tn ):
        return nu__n[na, :] * G__tn * f__tn

    @cached
    def log_mu__tn( mu__tn ):
        return log( mu__tn )

    """ Helpers for Jacobian, Hessian """

    @cached
    def dfonf__tn( omega__n ):
        return omega__n[na, :]

    @cached
    def d2fonf__tn( omega__n ):
        return (omega__n ** 2)[na, :]

    @cached
    def r__tn( y__tn, mu__tn ):
        r__tn = y__tn - mu__tn
        r__tn[ ~np.isfinite(r__tn) ] = 0
        return r__tn

    """ Objective for neuron n: h(z, nu) """

    @cached
    def h( mu__tn, y__tn, log_mu__tn, tau, z__t, rho, lambda__t, omega__t ):
        return sum([
            np.nansum( mu__tn - y__tn * log_mu__tn ),
            0.5 * tau * norm( z__t )**2,
            rho * lambda__t.dot( z__t - omega__t ),
            0.5 * rho * norm( z__t - omega__t )**2
        ])

    """ Jacobian: h'(z, nu) """

    @cached
    def dh( r__tn, dfonf__tn, rho, lambda__t, omega__t, tau, z__t ):
        # partials
        dz = (
            -(r__tn * dfonf__tn).sum( axis=1 ) +
            rho * (lambda__t - omega__t) +
            (rho + tau) * z__t )
        da = -r__tn.sum(axis=0)
        # put together
        return conc([ dz, da ])

    """ Newton step """

    @cached
    def d2h_inv_dh( dh, r__tn, dfonf__tn, d2fonf__tn, y__tn, mu__tn, rho, tau, M ):
        # partials
        a = dz2_diag = (
                -r__tn * d2fonf__tn + y__tn * dfonf__tn**2 ).sum(axis=1)
        a[ ~np.isfinite( a ) ] = 0
        a = a + rho + tau
        b = da2 = mu__tn.sum( axis=0 )
        C = dzda = mu__tn * dfonf__tn
        # precomputations
        ainv = 1. / a
        AinvC = ainv[:, na] * C
        Einv = diag(b) - C.T.dot(AinvC)
        # blocks of dh
        x = dh[:-M]
        y = dh[-M:]
        x_on_a = x / a
        f = np.linalg.solve( Einv, C.T.dot(x_on_a) - y )
        # solution
        return conc([ x_on_a + AinvC.dot(f), -f ])

    """ Infrastructure """

    @cached
    def z__t( z_vec, M ):
        return z_vec[:-M]

    @cached
    def a__n( z_vec, M ):
        return z_vec[-M:]

    """ Solving: neuron by neuron """

    def solve_for_next_Z_bfgs( self, verbose=True ):
        """ Deprecated. Use Newton steps instead, which are faster. """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        objective_2 = lambda x: objective( x.copy() )
        jacobian_2 = lambda x: jacobian( x.copy() )
        # starting values
        last_Z__ti = self.Z__ti
        last_a__ni = self.a__ni
        Z__ti = last_Z__ti.copy()
        a__ni = last_a__ni.copy()
        # run through each cell
        M, I = self.M, self.I
        for i in progress.dots( range(I), 'solving for Z', verbose=verbose ):
            self.i = i
            self.z_vec = conc([ Z__ti[:, i], a__ni[:, i] ])
            z_vec = fmin_l_bfgs_b( 
                    objective_2, self.z_vec, fprime=jacobian_2, 
                    approx_grad=False )
            z_vec = z_vec[0]
            Z__ti[ :, i ] = z_vec[:-M]
            a__ni[ :, i ] = z_vec[-M:]
        # save
        self.last_Z__ti = last_Z__ti
        self.Z__ti = Z__ti
        self.a__ni = a__ni
    
    def solve_for_next_Z( self, verbose=True ):
        """ The next primal variable is a local minimum """
        # construct objective
        objective = self.cfunction( np.array_equal, 'h', 'z_vec' )
        jacobian = self.cfunction( np.array_equal, 'dh', 'z_vec' )
        # starting values
        last_Z__ti = self.Z__ti
        last_a__ni = self.a__ni
        Z__ti = last_Z__ti.copy()
        a__ni = last_a__ni.copy()
        # run through each cell
        M, I = self.M, self.I
        for i in progress.dots( range(I), 'solving for Z', verbose=verbose ):
            # set the current state to this cell
            self.i = i
            self.z_vec = conc([ Z__ti[:, i], a__ni[:, i] ])
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
            Z__ti[ :, i ] = self.z_vec[:-M]
            a__ni[ :, i ] = self.z_vec[-M:]
        # save overall
        self.last_Z__ti = last_Z__ti
        self.Z__ti = Z__ti
        self.a__ni = a__ni

    def solve_for_next_Omega( self ):
        """ The next auxiliary variable is an SVD projection """
        self.last_Omega__ti = self.Omega__ti
        U__ti, S_ii, V_im = svd( self.Z__ti, full_matrices=False )
        U__ti = U__ti[ :, :self.rank ]
        S_ii = S_ii[ :self.rank ]
        V_im = V_im[ :self.rank, : ]
        self.Omega__ti = ( U__ti * S_ii[na, :] ).dot( V_im )

    def solve_for_next_Lambda( self ):
        """ The next scaled dual variable """
        self.last_Lambda__ti = self.Lambda__ti
        self.Lambda__ti = self.Lambda__ti + (self.Z__ti - self.Omega__ti)

    def update_rho( self ):
        """ Adjust step size to balance primal and dual """
        ratio = self.ADMM_R_frob / self.ADMM_S_frob
        if ratio > 10:
            self.rho = self.rho * 2.
            self.Lambda__ti = self.Lambda__ti / 2.
        elif ratio < 0.1:
            self.rho = self.rho / 2.
            self.Lambda__ti = self.Lambda__ti * 2.

    def solve( self, verbose=True ):
        tau = self.tau
        if verbose:
            print '*** solving with tau = %d ***' % tau
        self._solve_helper( verbose=verbose )

    def _solve_helper( self, verbose=True ):
        history = []
        for admm_iter in range( self.max_admm_iterations ):
            # update Z
            last_Z__ti = self.Z__ti
            last_a__ni = self.a__ni
            try:
                self.solve_for_next_Z( verbose=False )
            except np.linalg.LinAlgError:
                print '<linalg error, trying bfgs>'
                self.Z__ti = last_Z__ti
                self.a__ni = last_a__ni
                self.solve_for_next_Z_bfgs( verbose=True )
            if ~np.isfinite(self.Z__ti).all():
                Z__ti = self.Z__ti
                Z__ti[ ~np.isfinite(Z__ti) ] = 0
                self.Z__ti = Z__ti
            if ~np.isfinite( self.a__ni ).all():
                a__ni = self.a__ni
                a__ni[ ~np.isfinite( a__ni ) ] = 0
                self.a__ni = a__ni
            # update Omega
            self.solve_for_next_Omega()
            # update dual variables
            self.solve_for_next_Lambda()
            self.update_rho()
            # add results to history
            self.N_admm_iterations += 1
            current_state = Bunch({
                    'N_admm_iterations' : self.N_admm_iterations,
                    'Z__ti' : self.Z__ti,
                    'a__ni' : self.a__ni,
                    'Omega__ti' : self.Omega__ti,
                    'Lambda__ti' : self.Lambda__ti,
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
        for k in ['N_admm_iterations', 'Z__ti', 'Omega__ti', 'Lambda__ti', 'rho',
                'a__ni']:
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
    def nu__ni( a__ni ):
        return exp( a__ni )

    @cached
    def LL_per_obs( Z__ti, omega__n, a__ni, Y__tni, G__tn, nonlinearity ):
        F__tni = exp( Z__ti[:, na, :] * omega__n[na, :, na] )
        Mu__tni = exp( a__ni[na, :, :] ) * G__tn[:, :, na] * F__tni
        log_Mu__tni = log( Mu__tni )
        LL = np.nansum( -Mu__tni + Y__tni * log_Mu__tni )
        return LL / np.isfinite( Y__tni ).sum()



