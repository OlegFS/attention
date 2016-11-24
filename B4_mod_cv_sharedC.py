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

event_data = cPickle.load(open('data/event_data.pickle'))

"""
=======
Classes
=======
"""

###############

def fit_cv_mod_sharedC( d_idx, check=False ):

    base_model = 'CGL'
    mod_model = 'cue'
    hemi = True
    rank = 1
    N_reps = 10

    # setup
    d = event_data[ d_idx ]
    infix_attr = '%sM_h1' % base_model
    LL_sharedC_attr = 'cv_%s_sharedC_LL' % infix_attr
    LL_noC_attr = 'cv_%s_noC_LL' % infix_attr
    tau_attr = 'tau_%s' % infix_attr

    # check
    if check:
        return d.can_load_attribute(LL_sharedC_attr)

    # announce
    print '================================='
    print '%s:  %s' % ( LL_sharedC_attr, d.__repr__() )
    print '================================='

    # has this been done already
    if d.can_load_attribute(LL_sharedC_attr):
        print '[already done]'
        return None

    # get tau
    d.load_attribute( tau_attr )
    tau_L = getattr( d, tau_attr ).tau_L
    tau_R = getattr( d, tau_attr ).tau_R
    # get base model
    d.load_attribute( base_model )
    G__tn = getattr( d, base_model ).G__tn
    C__sn = getattr( d, base_model ).C__sn
    # get masks
    d.load_attribute('cv_masks__int')

    # container for predictions
    T, M = d.N_events, d.N_units
    J = int( np.floor(M * 0.2) * T )
    Mu_sharedC__j = np.zeros((N_reps, J))
    Mu_noC__j = np.zeros((N_reps, J))
    LL_sharedC__j = np.zeros(N_reps)
    LL_noC__j = np.zeros(N_reps)
        
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
        print 'fitting Cue models'
        G__tn = getattr( d, base_model ).G__tn
        C__sn = getattr( d, base_model ).C__sn
        kw = {'nonlinearity':'exp', 'rank':rank, 'verbose':False, 'G__tn':G__tn, 'C__sn':C__sn}
        kwL, kwR = d_training._parse_hemi_kw( **kw )
        mL = models.Cue( d_training.filter_rf_L, tau=tau_L, **kwL )
        mR = models.Cue( d_training.filter_rf_R, tau=tau_R, **kwR )

        ########################
        ### SharedCue model: L
        ########################

        print 'fitting SharedCue models'

        # gain and modulator in place
        G__tn = mL.G__tn * exp( mL.Omega__tn )
        # allowed cue direction
        w__n = svd(mL.Omega__tn, full_matrices=False)[2][0, :]
        # default z_vec
        a0__n = mL.C__sn.mean(axis=0)
        c0 = 0
        # fit model
        m2 = models.SharedCue( d.filter_rf_L, G__tn, w__n, a0__n, c0, solve=True )
        Mu_L__tn = m2.Mu__tn

        ### SharedCue model: R

        # gain and modulator in place
        G__tn = mR.G__tn * exp( mR.Omega__tn )
        # allowed cue direction
        w__n = svd(mR.Omega__tn, full_matrices=False)[2][0, :]
        # default z_vec
        a0__n = mR.C__sn.mean(axis=0)
        c0 = 0
        # fit model
        m2 = models.SharedCue( d.filter_rf_R, G__tn, w__n, a0__n, c0, solve=True )
        Mu_R__tn = m2.Mu__tn

        ### SharedCue model: put together

        Mu__tn = np.zeros( (T, M) )
        Mu__tn[ :, d.is_unit_rf_L ] = Mu_L__tn
        Mu__tn[ :, d.is_unit_rf_R ] = Mu_R__tn

        # save fits
        Mu_sharedC__j[rep, :] = Mu__tn[ mask__nt.T ]

        # LL
        Y__tn = d_testing.Y__tn
        Y__j = Y__tn[ mask__nt.T ]
        this_LL = (
                -Mu_sharedC__j[rep, :] 
                + Y__j * np.log(Mu_sharedC__j[rep, :])
                - logfactorial( Y__j ) )
        this_LL = np.nansum( this_LL ) / np.isfinite(Y__j).sum()
        LL_sharedC__j[rep] = this_LL

        ########################
        ### NoCue model: L
        ########################

        print 'fitting NoCue models'

        # gain and modulator in place
        G__tn = mL.G__tn * exp( mL.Omega__tn )
        # allowed cue direction
        w__n = svd(mL.Omega__tn, full_matrices=False)[2][0, :]
        # default z_vec
        a0__n = mL.C__sn.mean(axis=0)
        # fit model
        m2 = models.NoCue( d.filter_rf_L, G__tn, a0__n, solve=True )
        Mu_L__tn = m2.Mu__tn

        ### FixedDirection model: R

        # gain and modulator in place
        G__tn = mR.G__tn * exp( mR.Omega__tn )
        # allowed cue direction
        w__n = svd(mR.Omega__tn, full_matrices=False)[2][0, :]
        # default z_vec
        a0__n = mR.C__sn.mean(axis=0)
        # fit model
        m2 = models.NoCue( d.filter_rf_R, G__tn, a0__n, solve=True )
        Mu_R__tn = m2.Mu__tn

        ### put together
        Mu__tn = np.zeros( (T, M) )
        Mu__tn[ :, d.is_unit_rf_L ] = Mu_L__tn
        Mu__tn[ :, d.is_unit_rf_R ] = Mu_R__tn

        # save fits
        Mu_noC__j[rep, :] = Mu__tn[ mask__nt.T ]

        # LL
        Y__tn = d_testing.Y__tn
        Y__j = Y__tn[ mask__nt.T ]
        this_LL = (
                -Mu_noC__j[rep, :] 
                + Y__j * np.log(Mu_noC__j[rep, :])
                - logfactorial( Y__j ) )
        this_LL = np.nansum( this_LL ) / np.isfinite(Y__j).sum()
        LL_noC__j[rep] = this_LL

        # is this done already
        if LL_sharedC_attr in d.available_attributes:
            print '[already done]'
            return None


    # save LL
    setattr( d, LL_sharedC_attr, LL_sharedC__j )
    d.save_attribute( LL_sharedC_attr, overwrite=None )
    setattr( d, LL_noC_attr, LL_noC__j )
    d.save_attribute( LL_noC_attr, overwrite=None )




#######################
# jobs
#######################


jobs = []
for d_idx in range( len(event_data) ):
    jobs.append( (fit_cv_mod_sharedC, d_idx) )


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
