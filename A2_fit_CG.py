# modules
from helpers import *
import attention
reload(attention)
from attention import *
import pop 
# core data
"""
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/event_data.pickle'))

event_data.load_attribute('C')

=======
classes
=======
"""

def logdet(X):
    """ Log determinant of a matrix. """
    return np.linalg.slogdet(X)[1]



class PopData( pop.PopData ):                                                                            
    
    def __init__( self, X, Y ):
        super( PopData, self ).__init__( X, Y, add_constant=False )

        
class PopSolver( pop.Factorial_k, pop.Lowpass_s, pop.ML_w, pop.PopSimple ):
    
    _k_solver_class = MopSolver

    @property
    def _max_len_h_star( self ):
        return min([ 1000, self.data.T ])
    
    @cached
    def training_idxs( data ):
        return np.isfinite( data.Y ).all( axis=1 )
    
    @cached
    def has_testing_regions():
        return False
    
    @cached
    def T_training( training_idxs ):
        return training_idxs.sum()
    
    @cached
    def T_testing():
        return 0
    
    def _slice_by_testing( self, sig ):
        return A([])
    
    def _slice_by_training( self, sig ):
        return sig[ self.training_idxs, ... ]
    
    def _zero_during_testing( self, sig ):
        sig = sig.copy()
        sig[ ~self.training_idxs, ... ] = 0
        return sig


"""
==========
how to fit
==========
"""

def fit_CG( d_idx, check=False ):
    """ Fit S_global__tn on top of C__sn """
    # get cell
    d = event_data[ d_idx ]
    # is it done
    attr = 'CG'
    if check:
        return d.can_load_attribute( attr )
    if d.can_load_attribute( attr ):
        print '- already done'
        return
    # group together responses to a single trial
    d2 = d.group_by_trial
    # trial times
    t_ms = d2.event_t_msec
    min_dt = float( np.diff( t_ms ).min() )
    t_idx = np.floor( (t_ms - t_ms.min()) / min_dt ).astype(int)
    # how long is the signal
    T0 = T = t_idx.max() + 1
    T2 = int( pow(2, np.ceil(np.log2(T/2.))) ) * 2
    T3 = int( pow(2, np.ceil(np.log2(T/3.))) ) * 3
    T5 = int( pow(2, np.ceil(np.log2(T/5.))) ) * 5
    T = min(T2, T3, T5)
    # assemble spike counts
    Y__tn = d2.mean_counts_per_event.T
    # padded spike counts
    Y_padded__tn = np.zeros((T, d.N_units)) * np.nan
    Y_padded__tn[ t_idx, : ] = Y__tn
    Y__tn = Y_padded__tn
    # stimulus matrix
    X__ti = np.zeros((T, 2))
    X__ti[ t_idx[ d2.is_trial_cue_L ], 0 ] = 1
    X__ti[ t_idx[ d2.is_trial_cue_R ], 1 ] = 1
    # regression: initial conditions
    K__i = d.C.C__sn.T.flatten()
    K__i[ ~np.isfinite(K__i) ] = -3
    # initial weights
    w__n = np.ones( d.N_units )
    # remove any dodgy cells
    tok__n = (d.Y__tn.mean(axis=0) > 0.01)
    if ~tok__n.all():
        Y__tn = Y__tn[ :, tok__n ]
        w__n = w__n[ tok__n ]
        K__i = d.C.C__sn[ :, tok__n ].T.flatten()
        K__i[ ~np.isfinite(K__i) ] = -3
    # construct model data
    model_data = PopData( X__ti, Y__tn )
    # build model
    initial_conditions = { 'k': K__i, 'theta_s':[-7, 10], 's':np.zeros(T), 'w':w__n }
    model = PopSolver( model_data, initial_conditions=initial_conditions, rank=1 )
    model._grid_search_theta_s_parameters['bounds'] = [[-9, -4], [-40, 20]]
    model._bounds_s = [(-9, -4), (-60, 40)]
    # re-estimate cue conditions
    model.calc_posterior_k()
    for ks in model.k_solvers:
        if np.abs(np.diff(ks.k_vec))[0] > 1:
            ks.k_vec = np.max(ks.k_vec) * A([1, 1])
            ks.calc_posterior_k()
    model.calc_posterior_k()
    # solve for slow drift
    model.solve( verbose=2 )
    clear_output()
    # extract slow global signal
    SG__tn = model.posterior_sw.H__sw[ model.training_idxs, : ]
    # make the right shape
    trials, idxs, inv_idxs = np.unique( d.trial_idxs, return_inverse=True, return_index=True )
    SG__tn = SG__tn[ inv_idxs, : ]
    # check that it is not degenerate
    w__n = np.abs( svd( SG__tn - SG__tn.mean(axis=0)[na, :], full_matrices=False )[2][0, :].T )
    N_sig = (w__n > (w__n.max() / 10)).sum()
    if N_sig < 5:
        SG__tn *= 0
        print '**** DEGENERATE ****'
        tracer()
    if ~tok__n.all():
        S = SG__tn
        SG__tn = np.zeros( (S.shape[0], len(tok__n)) )
        SG__tn[ :, tok__n ] = S
    # done
    if d.can_load_attribute( attr ):
        print '- already done'
        return
    # save
    d.CG = Bunch(d.C.copy())
    d.CG['SG__tn'] = SG__tn
    d.CG = BaseModel( d.CG )
    d.save_attribute(attr)


====
jobs
====

# job list
jobs = []
for d_idx in range( len(event_data) ):
    jobs.append( (fit_CG, d_idx) )

# check all the jobs
checked_jobs = []
for job in progress.dots( jobs, 'checking if jobs are done' ):
    func = job[0]
    args = tuple([ a for a in job[1:] ])
    if not func( *args, check=True ):
        checked_jobs.append(job)
print '%d jobs to do out of %d' % (len(checked_jobs), len(jobs))
jobs = checked_jobs

if __name__ == '__main__':
    # run
    print ' '
    print ' '
    randomised_jobs = [ jobs[i] for i in np.random.permutation(len(jobs)) ]
    for job in progress.numbers( randomised_jobs ):
        func = job[0]
        args = tuple([ a for a in job[1:] ])
        func( *args )
        print ' '
        print ' '
