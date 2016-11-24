import attention
reload(attention)
from attention import *

from scipy.special import gammaln

def logfactorial(x):
    return gammaln( x+1 )


"""
=========
Load data
=========
"""

use_sim = True
repeat_idx = 15


#### load event data

if not locals().has_key('event_data'):
    if use_sim:
        N_reps = 10
        event_data = cPickle.load(open('data/sim_event_data.v4.repeat.%d.pickle' % repeat_idx))
    else:
        N_reps = 50
        event_data = cPickle.load(open('data/event_data.pickle'))


###############

def fit_cv_base( d_idx, base_model, N_reps=N_reps, check=False ):

    # setup
    d = event_data[ d_idx ]
    Mu_attr = 'cv_%s_Mu__j' % base_model
    LL_attr = 'cv_%s_LL' % base_model

    # check
    if check:
        return d.can_load_attribute(LL_attr)

    # announce
    print '================================='
    print '%s:  %s' % ( LL_attr, d.__repr__() )
    print '================================='

    # has this been done already
    if d.can_load_attribute(LL_attr):
        print '[already done]'
        return None

    # get base model
    if base_model != 'Z':
        d.load_attribute( base_model )
        G__tn = getattr( d, base_model ).G__tn
        C__sn = getattr( d, base_model ).C__sn
        S__ts = d.S__ts
    # get masks
    d.load_attribute('cv_masks__int')

    # container for predictions
    T, M = d.N_events, d.N_units
    J = int( np.floor(M * 0.2) * T )
    Mu__j = np.zeros((N_reps, J))
    LL__j = np.zeros(N_reps)

    # predictions
    if base_model == 'Z':
        Mu__tn = np.repeat( np.mean( d.Y__tn, axis=0 )[None, :], T, 0)
    else:
        Mu__tn = np.exp( S__ts.dot(C__sn) ) * G__tn

    # cycle through repeats
    for rep in range(N_reps) :

        mask__nt = d.cv_masks__int[ rep, :, : ]

        # save fits
        Mu__j[rep, :] = Mu__tn[ mask__nt.T ]

        # LL
        Y__j = d.Y__tn[ mask__nt.T ]
        this_LL = -Mu__j[rep, :] + Y__j * np.log(Mu__j[rep, :]) - logfactorial( Y__j )
        this_LL = np.nansum( this_LL ) / np.isfinite(Y__j).sum()
        LL__j[rep] = this_LL

        # is this done already
        if d.can_load_attribute(LL_attr):
            print '[already done]'
            return None

    # save LL
    setattr( d, LL_attr, LL__j )
    d.save_attribute( LL_attr, overwrite=None )
    # save Mu
    """
    if not use_sim:
        setattr( d, Mu_attr, Mu__j )
        d.save_attribute( Mu_attr, overwrite=None )
    """



def fit_cv_mod( d_idx, rank, base_model, mod_model, hemi, N_reps=N_reps, check=False ):

    # setup
    d = event_data[ d_idx ]
    if mod_model == 'cue':
        if hemi:
            infix_attr = '%sM_h%d' % (base_model, rank)
        else:
            infix_attr = '%sM_%d' % (base_model, rank)
    elif mod_model == 'naive':
        if hemi:
            infix_attr = '%sN_h%d' % (base_model, rank)
        else:
            infix_attr = '%sN_%d' % (base_model, rank)
    else:
        raise ValueError('unknown mod_model')
    Mu_attr = 'cv_%s_Mu__j' % infix_attr
    LL_attr = 'cv_%s_LL' % infix_attr
    tau_attr = 'tau_%s' % infix_attr

    # check
    if check:
        return d.can_load_attribute(LL_attr)

    # announce
    print '================================='
    print '%s:  %s' % ( infix_attr, d.__repr__() )
    print '================================='

    # has this been done already
    if d.can_load_attribute(LL_attr):
        print '[already done]'
        return None

    # get tau
    d.load_attribute( tau_attr )
    if hemi:
        tau_L = getattr( d, tau_attr ).tau_L
        tau_R = getattr( d, tau_attr ).tau_R
    else:
        tau = getattr( d, tau_attr ).tau
    # get base model
    d.load_attribute( base_model )
    G__tn = getattr( d, base_model ).G__tn
    C__sn = getattr( d, base_model ).C__sn
    # get masks
    d.load_attribute('cv_masks__int')

    # container for predictions
    T, M = d.N_events, d.N_units
    J = int( np.floor(M * 0.2) * T )
    Mu__j = np.zeros((N_reps, J))
    LL__j = np.zeros(N_reps)
        
    # cycle through repeats
    for rep in progress.numbers( range(N_reps) ):

        # separate into training and testing
        d_training = d.__copy__()
        d_testing = d.__copy__()
        d_training.spike_counts = d_training.spike_counts.copy().astype(float)
        d_testing.spike_counts = d_testing.spike_counts.copy().astype(float)
        mask__nt = d.cv_masks__int[ rep, :, : ]
        d_training.spike_counts[ mask__nt ] = np.nan
        d_testing.spike_counts[ ~mask__nt ] = np.nan

        # fit
        kw = {'nonlinearity':'exp', 'rank':rank, 'verbose':False, 'G__tn':G__tn, 'C__sn':C__sn}
        if mod_model == 'cue':
            if hemi:
                m = d_training.solve_cue_hemi( tau_L=tau_L, tau_R=tau_R, **kw )
                m = m.all
            else:
                m = d_training.solve_cue( tau=tau, **kw )
            # extract
            Mu__tn = (
                    m.S__ts.dot(m.Nu__sn) * G__tn * exp( m.Omega__tn ) )
        
        elif mod_model == 'naive':
            if hemi:
                m = d_training.solve_naive_hemi( tau_L=tau_L, tau_R=tau_R, **kw )
                m = m.all
            else:
                m = d_training.solve_naive( tau=tau, **kw )
            # extract
            Mu__tn = (
                    m.nu__n[None, :] * G__tn * exp( m.Omega__tn ) )


        # save fits
        Mu__j[rep, :] = Mu__tn[ mask__nt.T ]

        # LL
        Y__tn = d_testing.Y__tn
        Y__j = Y__tn[ mask__nt.T ]
        this_LL = (
                -Mu__j[rep, :] 
                + Y__j * np.log(Mu__j[rep, :])
                - logfactorial( Y__j ) )
        this_LL = np.nansum( this_LL ) / np.isfinite(Y__j).sum()
        LL__j[rep] = this_LL

        # is this done already
        if LL_attr in d.available_attributes:
            print '[already done]'
            return None


    # save LL
    setattr( d, LL_attr, LL__j )
    d.save_attribute( LL_attr, overwrite=None )
    # save Mu
    """
    if not use_sim:
        setattr( d, Mu_attr, Mu__j )
        d.save_attribute( Mu_attr, overwrite=None )
    """




#######################
# jobs
#######################

"""
# base models
jobs = []
for d_idx in range( len(event_data) ):
    for base_model in ['Z', 'C', 'CG', 'CGL']:
        jobs.append((fit_cv_base, d_idx, base_model))

# cue models
for d_idx in range( len(event_data) ):
    base_model = 'CG'
    for rank in range(1, 11):
        jobs.append( (fit_cv_mod, d_idx, rank, base_model, 'cue', False) )
    for rank in range(1, 6):
        jobs.append( (fit_cv_mod, d_idx, rank, base_model, 'cue', True) )
"""


"""
# base models
jobs = []
for d_idx in range( len(event_data) ):
    for base_model in ['CGL', 'CG', 'CL', 'C', 'Z']:
        jobs.append((fit_cv_base, d_idx, base_model))

# cue models
for d_idx in range( len(event_data) ):
    for rank in range(1,6):
        for base_model in ['CGL', 'CG', 'CL', 'C']:
            for hemi in [False, True]:
                jobs.append( (fit_cv_mod, d_idx, rank, base_model, 'cue', hemi) )

# naive models
for d_idx in range( len(event_data) ):
    for rank in range(1,6):
        for base_model in ['CGL']:
            for hemi in [False, True]:
                jobs.append( (fit_cv_mod, d_idx, rank, base_model, 'naive', hemi) )

# models for sim data
if event_data[0].prefix.startswith('sim'):
    jobs = []
    for d_idx in range( len(event_data) ):
        for rank in range(11):
            for base_model in ['CGL']:
                for hemi in [False]:
                    jobs.append( (fit_cv_mod, d_idx, rank, base_model, 'cue', hemi) )
"""

# models for sim data
if event_data[0].prefix.startswith('sim_v4'):
    jobs = []
    target_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
    for d_idx in range( len(event_data) ):
        #for rank in target_ranks:
        #    for base_model in ['CGL']:
        #        for hemi in [False]:
        #            pass
        #            #jobs.append( (fit_cv_mod, d_idx, rank, base_model, 'cue', hemi) )
        jobs.append( (fit_cv_base, d_idx, 'CGL') )
        jobs.append( (fit_cv_base, d_idx, 'Z') )



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
