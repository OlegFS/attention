import attention
reload(attention)
from attention import *


"""
=========
Load data
=========
"""

#### load event data
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/event_data.pickle'))
    #event_data = cPickle.load(open('data/sim_event_data.pickle'))


####################


def fit_naive( d, attr_to_check, **kw ):
    announce( d, attr_to_check )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    results = d.solve_naive( **kw )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    return results

def fit_naive_hemi( d, attr_to_check, **kw ):
    announce( d, attr_to_check )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    results = d.solve_naive_hemi( **kw )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    return results

def fit_cue( d, attr_to_check, **kw ):
    announce( d, attr_to_check )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    results = d.solve_cue( **kw )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    return results

def fit_cue_hemi( d, attr_to_check, **kw ):
    announce( d, attr_to_check )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    results = d.solve_cue_hemi( **kw )
    if attr_to_check in d.available_attributes:
        print '[already done]'
        return None
    return results

def announce( d, attr ):
    print '================================='
    print '%s:  %s' % ( attr, d.__repr__() )
    print '================================='
    
# fitting functions

def f_general( d_idx, rank, base_model, mod_model, check=False ):
    # attribute and recording
    if mod_model == 'cue':
        attr = '%sM_%d' % (base_model, rank)
    elif mod_model == 'naive':
        attr = '%sN_%d' % (base_model, rank)
    d = event_data[ d_idx ]
    if check:
        return d.can_load_attribute( attr )
    # get tau
    tau_attr = 'tau_%s' % attr
    d.load_attribute( tau_attr )
    tau = getattr( d, tau_attr ).tau
    # get base model
    d.load_attribute(base_model)
    G__tn = getattr( d, base_model ).G__tn
    C__sn = getattr( d, base_model ).C__sn
    # fit
    fit_kw = {'tau':tau, 'rank':rank, 'nonlinearity':'exp', 
            'verbose':False, 'G__tn':G__tn, 'C__sn':C__sn }
    if mod_model == 'cue':
        results = fit_cue( d, attr, **fit_kw )
    elif mod_model == 'naive':
        results = fit_naive( d, attr, **fit_kw )
    else:
        raise ValueError('unknown mod_model')
    # save
    if results is not None:
        setattr( d, attr, results )
        d.save_attribute( attr, overwrite=None )

# exponential nonlinearity, hemi

def f_hemi_general( d_idx, rank, base_model, mod_model, check=False ):
    # attribute and recording
    if mod_model == 'cue':
        attr = '%sM_h%d' % (base_model, rank)
    elif mod_model == 'naive':
        attr = '%sN_h%d' % (base_model, rank)
    else:
        raise ValueError('unknown mod_model')
    d = event_data[ d_idx ]
    if check:
        return d.can_load_attribute( attr )
    # get tau
    tau_attr = 'tau_%s' % attr
    d.load_attribute( tau_attr )
    tau_L = getattr( d, tau_attr ).tau_L
    tau_R = getattr( d, tau_attr ).tau_R
    # get base model
    d.load_attribute( base_model )
    G__tn = getattr( d, base_model ).G__tn
    C__sn = getattr( d, base_model ).C__sn
    # fit
    fit_kw = {'tau_L':tau_L, 'tau_R':tau_R, 'rank':rank, 'nonlinearity':'exp', 
            'verbose':False, 'G__tn':G__tn, 'C__sn':C__sn }
    if mod_model == 'cue':
        results = fit_cue_hemi( d, attr, **fit_kw )
    elif mod_model == 'naive':
        results = fit_naive_hemi( d, attr, **fit_kw )
    else:
        raise ValueError('unknown mod_model')
    # save
    if results is not None:
        setattr( d, attr, results )
        d.save_attribute( attr, overwrite=None )



"""
====
jobs
====
"""

# cue models
jobs = []
for func in [f_general, f_hemi_general]:
    for d_idx in range( len(event_data) ):
        for i in range(1, 6):
            for base_model in ['CGL', 'CG', 'CL', 'C']:
                jobs.append((func, d_idx, i, base_model, 'cue'))

# naive models
for func in [f_general, f_hemi_general]:
    for d_idx in range( len(event_data) ):
        for i in range(1, 6):
            for base_model in ['CGL']:
                jobs.append((func, d_idx, i, base_model, 'naive'))

# models for sim data
if event_data[0].prefix.startswith('sim'):
    jobs = []
    for func in [f_general]:
        for d_idx in range( len(event_data) ):
            for i in range(1, 11):
                for base_model in ['CGL']:
                    jobs.append((func, d_idx, i, base_model, 'cue'))


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
