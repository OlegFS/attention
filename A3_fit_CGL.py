from attention import * 

#### load event data
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/event_data.pickle'))

for a in ['CG', 'C']:
    event_data.load_attribute(a)

"""
==========
How to fit
==========
"""

#### script

def fit_SL_to_unit( d, unit_idx, include_SG=True ):
    if include_SG:
        d.SG__tn = d.CG.SG__tn
    # group together responses to a single trial
    d2 = d.group_by_trial
    # assemble spike counts
    y__t = d2.mean_counts_per_event[unit_idx, :]
    # trial times
    t_ms =  d2.event_t_msec
    min_dt = float( np.diff( t_ms ).min() )
    t_idx = np.floor( (t_ms - t_ms.min()) / min_dt ).astype(int)
    # how long is the signal
    T = t_idx.max() + 1
    T2 = int( pow(2, np.ceil(np.log2(T))) )                                 
    T3 = int( pow(2, np.ceil(np.log2(T/3.))) ) * 3
    T5 = int( pow(2, np.ceil(np.log2(T/5.))) ) * 5
    T = min(T2, T3, T5)
    # padded spike counts
    y_padded = np.zeros(T) * np.nan
    y_padded[ t_idx ] = y__t
    y__t = y_padded
    # construct stimulus matrix
    if include_SG:
        X__ti = np.zeros((T, 3))
        X__ti[ t_idx, 2 ] = d2.SG__tn[ :, unit_idx ]
    else:
        X__ti = np.zeros((T, 2))
    X__ti[ t_idx[ d2.is_trial_cue_L ], 0 ] = 1
    X__ti[ t_idx[ d2.is_trial_cue_R ], 1 ] = 1
    # construct MopData object
    md = MopData( X__ti, y__t )
    k0 = d.CG.C__sn[:, unit_idx]
    if include_SG:
        k0 = conc([ k0, [1] ])
    # construct solver
    ms = MopSolver( md, initial_conditions={'k':k0, 'theta_h':[-7, 10]} )
    ms.solve(verbose = 0)

    # MAP estimates
    theta_h = ms.posterior_h.theta_h
    k = ms.posterior_k.k_vec
    SL__n = ms.posterior_h.expected_log_g[ ms.training_idxs ]
    #log_FXk = ms.posterior_k.expected_log_FXk[ ms.training_idxs ]

    # reconstruct how to put back in to original dataset
    SL__trial = SL__n.copy()
    SL__event = SL__n[ np.unique( d.trial_idxs, return_inverse=True )[1] ]

    # final data structure
    results = {
            'SL__trial':SL__trial, 'SL__event':SL__event, 
            'k':k, 'theta_h':theta_h, 'ms':ms, 'md':md}
    return results


def fit_CGL( d_idx, check=False ):
    # check
    d = event_data[ d_idx ]
    attr = 'CGL'
    if check:
        return d.can_load_attribute( attr )
    if d.can_load_attribute( attr ):
        print '[already done]'
        return None
    # fit models individually
    results = []
    for i in progress.dots( range(d.N_units), 'fitting SL' ):
        r = fit_SL_to_unit( d, i )
        results.append(r)
        # has it been done already
        if d.can_load_attribute( attr ):
            print '[already done]'
            return None
    # save
    CGL = Bunch( d.CG.copy() )
    CGL['C__sn'] = A([ r['k'][:2] for r in results ]).T
    CGL['SG__tn'] = CGL['SG__tn'] * A([ r['k'][-1] for r in results ])[None, :]
    CGL['SL__tn'] = A([ r['SL__event'] for r in results ]).T
    CGL['theta_h__ni'] = A([ r['theta_h'] for r in results ])
    CGL = BaseModel( CGL )
    setattr( d, attr, CGL )
    d.save_attribute( attr, overwrite=None )

def fit_CL( d_idx, check=False ):
    # check
    d = event_data[ d_idx ]
    attr = 'CL'
    if check:
        return d.can_load_attribute( attr )
    if d.can_load_attribute( attr ):
        print '[already done]'
        return None
    # fit models individually
    results = []
    for i in progress.dots( range(d.N_units), 'fitting SL' ):
        r = fit_SL_to_unit( d, i, include_SG=False )
        results.append(r)
        # has it been done already
        if d.can_load_attribute( attr ):
            print '[already done]'
            return None
    # save
    CL = Bunch( d.C.copy() )
    CL['C__sn'] = A([ r['k'] for r in results ]).T
    CL['SL__tn'] = A([ r['SL__event'] for r in results ]).T
    CL['theta_h__ni'] = A([ r['theta_h'] for r in results ])
    CL = BaseModel( CL )
    setattr( d, attr, CL )
    d.save_attribute( attr, overwrite=None )



"""
====
jobs
====
"""

# job list
jobs = []
for d_idx in range( len(event_data) ):
    jobs.append( (fit_CGL, d_idx) )
    jobs.append( (fit_CL, d_idx) )

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
