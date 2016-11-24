from attention import *

"""
=========
Load data
=========
"""

#### load event data
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/event_data.pickle'))
    #event_data = cPickle.load(open('data/sim_event_data.v3.pickle'))


####################


taus = np.logspace(0, 3, 10).astype(int)


def partition_d( d ):
    T, M = d.N_events, d.N_units
    d_training = d.__copy__()
    d_testing = d.__copy__()
    d_training.spike_counts = d_training.spike_counts.copy().astype(float)
    d_testing.spike_counts = d_testing.spike_counts.copy().astype(float)
    mask = np.zeros((M, T), dtype=bool)
    for t in range(T):
        mask[ perm( M )[:M*0.2], t ] = True
    d_training.spike_counts[ mask ] = np.nan
    d_testing.spike_counts[ ~mask ] = np.nan
    return d_training, d_testing


def calc_best_tau( d, attr_to_check, base_model='CGL', mod_model='cue', N_reps=40, 
        preload_LLs=None, **kw ):
    # load stuff for base_model
    if base_model not in ['CGL', 'CG', 'CL', 'C']:
        raise ValueError('unknown base_model')
    d.load_attribute(base_model)
    G__tn = getattr( d, base_model ).G__tn
    C__sn = getattr( d, base_model ).C__sn

    # containers
    LLs = Bunch({})
    if preload_LLs is None:
        for tau in taus:
            LLs[tau] = []
    else:
        # convert back into lists
        for tau in taus:
            LLs[tau] = [l for l in preload_LLs[tau]]
        # don't have to do as many
        N_reps_done = len( preload_LLs.values()[0] )
        N_reps = N_reps - N_reps_done

    # run through
    for _ in progress.numbers( range(N_reps), key = lambda i: 'iteration %d' % i ):
        # partition
        d_training, d_testing = partition_d( d )
        # fit model
        if mod_model == 'cue':
            ModModel = models.Cue
        elif mod_model == 'naive':
            ModModel = models.Naive
        else:
            raise ValueError('unknown mod_model')
        s_training = ModModel( 
                d_training, tau=taus, verbose=False, save_intermediates=True, 
                G__tn=G__tn, C__sn=C__sn, **kw )
        # cross validate
        s_testing = ModModel(
                d_testing, tau=taus, verbose=False, solve=False, 
                G__tn=G__tn, C__sn=C__sn, **kw )
        # check all the tausW
        for tau in taus:
            results = s_training.intermediate_results[ tau ]
            if mod_model == 'naive':
                s_testing.a__n = results.a__n
            elif mod_model == 'cue':
                s_testing.C__sn = results.C__sn
            else:
                raise ValueError('unknown mod_model')
            s_testing.Z__tn = results.Omega__tn
            s_testing.Omega__tn = results.Omega__tn
            LLs[tau].append( s_testing.LL_per_obs )
            # check if we're done
            try:
                d.load_attribute( attr_to_check )
                print '[already done]'
                return None
            except IOError:
                pass
    # package and return
    for tau in taus:
        LLs[tau] = A( LLs[tau] )
    return LLs



#################################
# functions for cross validation
#################################

def announce( d, attr ):
    s = '%s:  %s' % ( attr, d.__repr__() )
    print '='*len(s)
    print s
    print '='*len(s)
    
def f_general( d_idx, base_model, mod_model, rank, check=False, **kw ):
    if mod_model == 'cue':
        attr = 'tau_%sM_%d' % (base_model, rank)
    elif mod_model == 'naive':
        attr = 'tau_%sN_%d' % (base_model, rank)
    else:
        raise ValueError('unknown mod_model')
    d = event_data[ d_idx ]
    if check:
        return d.can_load_attribute(attr)
    announce( d, attr )
    try:
        d.load_attribute( attr )
    except IOError:
        # preload
        preload_attr = attr.replace('tau_', 'tau_10reps_')
        if d.can_load_attribute( preload_attr ):
            d.load_attribute(preload_attr)
            preload_LLs = getattr( d, preload_attr ).LLs
        else:
            preload_LLs = None
        # fit rest
        tau_kw = {'base_model':base_model, 'rank':rank, 'mod_model':mod_model}
        tau_kw.update( **kw )
        results = calc_best_tau( d, attr, preload_LLs=preload_LLs, **tau_kw )
        if (results is not None):
            t = Bunch({'LLs':results})
            t.tau = taus[ A([ np.median(results[tau]) for tau in taus ]).argmax() ]
            setattr( d, attr, t )
            d.save_attribute( attr, overwrite=None )

def f_hemi_general( d_idx, base_model, mod_model, rank, check=False, **kw ):
    if mod_model == 'cue':
        attr = 'tau_%sM_h%d' % (base_model, rank)
    elif mod_model == 'naive':
        attr = 'tau_%sN_h%d' % (base_model, rank)
    else:
        raise ValueError('unknown mod_model')
    d = event_data[ d_idx ]
    if check:
        return d.can_load_attribute(attr)
    announce( d, attr )
    try:
        d.load_attribute( attr )
    except IOError:
        # preload
        preload_attr = attr.replace('tau_', 'tau_10reps_')
        if d.can_load_attribute( preload_attr ):
            d.load_attribute(preload_attr)
            preload_LLs = getattr( d, preload_attr )
        else:
            preload_LLs = Bunch({'LLs_L':None, 'LLs_R':None})
        # fit rest
        dL, dR = d.filter_rf_L, d.filter_rf_R
        tau_kw = {'base_model':base_model, 'rank':rank, 'mod_model':mod_model}
        tau_kw.update( **kw )
        print 'fitting L'
        results_L = calc_best_tau( dL, attr, preload_LLs=preload_LLs.LLs_L, **tau_kw )
        print 'fitting R'
        results_R = calc_best_tau( dR, attr, preload_LLs=preload_LLs.LLs_R, **tau_kw )
        if (results_L is not None) and (results_R is not None):
            t = Bunch({'LLs_L':results_L, 'LLs_R':results_R})
            t.tau_L = taus[ A([ np.median(results_L[tau]) for tau in taus ]).argmax() ]
            t.tau_R = taus[ A([ np.median(results_R[tau]) for tau in taus ]).argmax() ]
            setattr( d, attr, t )
            d.save_attribute( attr, overwrite=None )



"""
====
jobs
====
"""

jobs = []
for d_idx in range( len(event_data) ):
    for base_model in ['CG', 'CGL']:
        for rank in range(1, 11):
            jobs.append( (f_general, d_idx, base_model, 'cue', rank) )
        for rank in range(1, 6):
            jobs.append( (f_hemi_general, d_idx, base_model, 'cue', rank) )

"""
# cue models
jobs = []
for d_idx in range( len(event_data) ):
    for base_model in ['CGL', 'CG', 'CL', 'C']:
        for rank in range(1, 6):
            jobs.append( (f_general, d_idx, base_model, 'cue', rank) )
            jobs.append( (f_hemi_general, d_idx, base_model, 'cue', rank) )

# naive models
for d_idx in range( len(event_data) ):
    for base_model in ['CGL']:
        for rank in range(1, 6):
            jobs.append( (f_general, d_idx, base_model, 'naive', rank) )
            jobs.append( (f_hemi_general, d_idx, base_model, 'naive', rank) )

# models for sim data
if event_data[0].prefix.startswith('sim'):
    jobs = []
    for d_idx in range( len(event_data) ):
        for base_model in ['CGL']:
            for rank in range(1, 11):
                jobs.append( (f_general, d_idx, base_model, 'cue', rank) )

# models for sim v3 data
if event_data[0].prefix.startswith('sim_v3'):
    jobs = []
    target_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
    for d_idx in [0]: #range( len(event_data) ):
        for base_model in ['CGL']:
            for rank in target_ranks:
                jobs.append( (f_general, d_idx, base_model, 'cue', rank) )
"""

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
