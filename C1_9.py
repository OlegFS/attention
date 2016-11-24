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
N_reps = 10
version = 4
repeat_idx = 11

#### load event data
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/sim_event_data.v4.repeat.%d.pickle' % repeat_idx))


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


def calc_best_tau( d, attr_to_check, base_model='CGL', mod_model='cue', N_reps=10, **kw ):
    # load stuff for base_model
    if base_model not in ['CGL', 'CG', 'CL', 'C']:
        raise ValueError('unknown base_model')
    d.load_attribute(base_model)
    G__tn = getattr( d, base_model ).G__tn
    C__sn = getattr( d, base_model ).C__sn

    # containers
    LLs = Bunch({})
    for tau in taus:
        LLs[tau] = []

    # run through
    for _ in progress.numbers( 
            range(N_reps), key = lambda i: 'iteration %d' % i ):
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
    
def f_general_tau( d_idx, base_model, mod_model, rank, check=False, **kw ):
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
        tau_kw = {'base_model':base_model, 'rank':rank, 'mod_model':mod_model}
        tau_kw.update( **kw )
        results = calc_best_tau( d, attr, **tau_kw )
        if (results is not None):
            t = Bunch({'LLs':results})
            t.tau = taus[ A([ results[tau].mean() for tau in taus ]).argmax() ]
            setattr( d, attr, t )
            d.save_attribute( attr, overwrite=None )

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

    # return
    return Mu__j



def combined_job( d_idx, target_rank, check=False ):
    if check:
        return fit_cv_mod( d_idx, target_rank, 'CGL', 'cue', False, check=True )
    # first fit tau
    print '*** fitting tau ***'
    f_general_tau( d_idx, 'CGL', 'cue', target_rank )
    # now fit cv
    print '*** fitting cv ***'
    fit_cv_mod( d_idx, target_rank, 'CGL', 'cue', False )


"""
====
jobs
====
"""


# models for sim v3 data
jobs = []
if version == 3:
    target_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 24, 32]
    d_idxs = np.concatenate([ (i + np.arange(9) * 50) for i in range(50) ])
elif version == 4:
    target_ranks = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16]
    d_idxs = np.arange( len(event_data) )


for d_idx in d_idxs:
    for rank in target_ranks:
        jobs.append( (combined_job, d_idx, rank) )


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
