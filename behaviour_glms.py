from attention import *

class GLM( AutoCR ):
    
    def __init__( self, X__ti, y__t, tau=0., labels=None ):
        self.X__ti = X__ti
        self.y__t = y__t
        self.T, self.N = shape( X__ti )        
        self.tau = tau
        self.labels = labels
        self.initialise_theta()
        self.solve()
        if labels is not None:
            for i in range(self.N):
                setattr( self, labels[i], A([ self.a_i[i], self.e_i[i] ]) )
       
    @property
    def results( self ):
        maxl = max([ len(l) for l in self.labels ])
        for i, l in enumerate( self.labels ):
            if l == 'const':
                continue
            a, e = self.a_i[i], self.e_i[i]
            s = ' '*(maxl + 3 - len(l)) + l + '  :  %+.2f  +/-  %.2f  ' % ( a, e )
            nstds = np.abs(a) / e
            for n in [1.96, 2.58, 3.29]:
                if nstds > n:
                    s += '*'
            print s
            
    def get_results( self ):
        return A([ self.a_i, self.e_i ])


class BehaviourGLM( GLM ):
    
    """ Class for fitting behaviour models. No lapse. """    
        
    def initialise_theta( self ):
        self.theta = np.zeros( self.N )
        
    @cached
    def a_i( theta ):
        return theta
        
    @cached
    def q__t( a_i, X__ti ):
        z__t = X__ti.dot( a_i )
        q = 1. / ( 1 + exp(-z__t) )
        q[ q < 1e-10 ] = 1e-10
        q[ q > 1 - 1e-10 ] = 1 - 1e-10
        return q
    
    @cached
    def negLP( q__t, y__t, tau, a_i ):
        LL = y__t.dot( log(q__t) ) + (1 - y__t).dot( log(1 - q__t) )
        LPrior = -0.5 * tau * a_i.dot(a_i)
        return -( LL + LPrior )
    
    @cached
    def dnegLP( q__t, y__t, X__ti, tau, a_i ):
        dLL = (y__t - q__t).dot( X__ti )
        dLPrior = -tau * a_i
        return -( dLL + dLPrior )
    
    @cached
    def d2negLP( q__t, X__ti, tau ):
        d__t = q__t * (1 - q__t)
        d2LL = -X__ti.T.dot( X__ti * d__t[:, na] )
        d2LPrior = -tau
        return -( d2LL + d2LPrior )
    
    @property
    def bounds( self ):
        return [(None, None)] * self.N
    
    def solve( self ):
        f = self.cfunction( np.array_equal, 'negLP', 'theta' )
        df = self.cfunction( np.array_equal, 'dnegLP', 'theta' )
        g = lambda x: f( x.copy() )
        dg = lambda x: df( x.copy() )
        theta = fmin_l_bfgs_b( g, self.theta, fprime=dg, bounds=self.bounds )[0]
        g( theta )
        
    @cached
    def e_i( d2negLP ):
        return np.sqrt(diag(inv(d2negLP)))
    
    @property
    def ae_i( self ):
        return A([ self.a_i, self.e_i ])
     
    def get_effects( self ):
        # original stim, to revert
        X0 = self.X__ti.copy()
        # run through each variable
        effects = np.zeros( self.N )
        for i in range( self.N ):
            X = X0.copy()
            X[:, i] = 1
            self.X__ti = X
            q1 = self.q__t.mean()
            X[:, i] = 0
            self.X__ti = X
            q2 = self.q__t.mean()
            effects[i] = q1 - q2
        self.X__ti = X0
        effects *= 100
        return effects
    
    @property
    def effects( self ):
        effects = self.get_effects()
        # print
        maxl = max([ len(l) for l in self.labels ])
        for i, l in enumerate( self.labels ):
            if l == 'const':
                continue
            e = effects[i]
            s = ' '*(maxl + 3 - len(l)) + l + '  :  %+.1f %%' % e
            print s 



class BehaviourGLMLapse( BehaviourGLM ):
    
    """ Class for fitting behaviour models. Includes lapse. """
    
    def initialise_theta( self ):
        self.theta = np.zeros(self.N + 1)
        self.theta[-1] = 1.
        
    @cached
    def a_i( theta ):
        return theta[:-1]
    
    @cached
    def gamma( theta ):
        return theta[-1]
        
    @cached
    def q__t( a_i, gamma, X__ti ):
        z__t = X__ti.dot( a_i )
        q = gamma / ( 1 + exp(-z__t) )
        q[ q < 1e-10 ] = 1e-10
        q[ q > 1 - 1e-10 ] = 1 - 1e-10
        return q
    
    @cached
    def dnegLP( q__t, y__t, X__ti, tau, a_i, gamma ):
        dLL_da = ( (y__t - q__t) * (gamma - q__t)/(1 - q__t) ).dot( X__ti ) / gamma
        dLPrior_da = -tau * a_i
        dLL_dgamma = np.sum( (y__t - q__t) / (1 - q__t) ) / gamma
        return -conc([ dLL_da + dLPrior_da, [dLL_dgamma] ])
    
    @property
    def bounds( self ):
        return [(None, None)] * self.N + [(0, 1)]
    
    @cached
    def d2negLP_a( gamma, q__t, y__t, tau, X__ti ):
        gmq = gamma - q__t
        omq = 1 - q__t
        ymq = y__t - q__t
        d__t = gamma**-2 * gmq / omq * ( gmq + ymq - gmq*ymq/omq )
        d2LL_a = -X__ti.T.dot( X__ti * d__t[:, na] )
        d2LPrior_a = -tau
        return -( d2LL_a + d2LPrior_a )
    
    @cached
    def e_i( d2negLP_a ):
        return np.sqrt(diag(inv(d2negLP_a)))


class ModulatorGLM( GLM ):
    
    def initialise_theta( self ):
        pass
    
    def solve( self ):
        tok = np.isfinite( self.X__ti ).all( axis=1 ) & np.isfinite( self.y__t )
        X = self.X__ti[ tok, : ]
        y = self.y__t[ tok ]        
        C = inv( X.T.dot(X) + self.tau * np.eye(self.N)  )
        self.theta = self.a_i = C.dot(X.T).dot(y) 
        self.e_i = np.sqrt(diag(C))


class PoissonGLM( GLM ):

    def __init__( self, X__ti, y__t, h__t=None, tau=0., labels=None ):
        self.h__t = h__t
        self.X__ti = X__ti
        self.y__t = y__t
        self.T, self.N = shape( X__ti )
        self.tau = tau
        self.labels = labels
        self.initialise_theta()
        self.solve()
        if labels is not None:
            for i in range(self.N):
                setattr( self, labels[i], A([ self.a_i[i], self.e_i[i] ]) )

    def initialise_theta( self ):
        self.theta = np.zeros( self.N )

    @cached
    def a_i( theta ):
        return theta

    @cached
    def log_mu__t( h__t, a_i, X__ti ):
        if h__t is None:
            return X__ti.dot( a_i )
        else:
            return X__ti.dot( a_i ) + h__t

    @cached
    def mu__t( log_mu__t ):
        return np.exp( log_mu__t )

    @cached
    def negLP( mu__t, log_mu__t, y__t, tau, a_i ):
        LL = -mu__t.sum() + y__t.dot( log_mu__t )
        LPrior = -0.5 * tau * a_i.dot(a_i)
        return -( LL + LPrior )

    @cached
    def dnegLP( y__t, mu__t, tau, X__ti, a_i ):
        dLL = (y__t - mu__t).dot( X__ti )
        dLPrior = -tau * a_i
        return -( dLL + dLPrior )

    @cached
    def d2negLP( mu__t, X__ti, tau ):
        d2LL = -X__ti.T.dot( X__ti * mu__t[:, na] )
        d2LPrior = -tau
        return -( d2LL + d2LPrior )

    @property
    def bounds( self ):
        return [(None, None)] * self.N
    
    def solve( self ):
        f = self.cfunction( np.array_equal, 'negLP', 'theta' )
        df = self.cfunction( np.array_equal, 'dnegLP', 'theta' )
        g = lambda x: f( x.copy() )
        dg = lambda x: df( x.copy() )
        theta = fmin_l_bfgs_b( g, self.theta, fprime=dg, bounds=self.bounds )[0]
        g( theta )

    @cached
    def e_i( d2negLP ):
        return np.sqrt(diag(inv(d2negLP)))
