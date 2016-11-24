from attention import * 

#### load event data
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/event_data.pickle'))

for a in ['CG', 'C', 'CGL']:
    event_data.load_attribute(a)

"""
==========
How to fit
==========
"""

#### script

def fit_sharedC( d_idx, base_model, check=False ):

    d = event_data[ d_idx ] 
    
    # model names
    mod_model = '%sM_h1' % base_model
    target_model = '%s_sharedC' % mod_model

    # check
    if check:
        return d.can_load_attribute( target_model )
    if d.can_load_attribute( target_model ):
        print '- already done'

    # load base models
    d.load_attribute([ base_model, mod_model ])
    d.base_model = getattr( d, base_model )
    d.mod_model = getattr( d, mod_model )
    for k in ['L', 'R', 'all']:
        d.mod_model[k] = Model(d.mod_model[k])
    

    # fit
    results = Bunch()

    dL = d.filter_rf_L
    G__tn = dL.base_model.G__tn * np.exp( d.mod_model.L.Omega__tn )
    w__n = d.mod_model.L.w__n
    a0__n = dL.base_model.C__sn.mean(axis=0)
    c0 = 0
    m = models.SharedCue( dL, G__tn, w__n, a0__n, c0 )
    results.L = Bunch({'w__n': m.w__n, 'c':m.c, 'C__sn':m.C__sn})

    dR = d.filter_rf_R
    G__tn = dR.base_model.G__tn * np.exp( d.mod_model.R.Omega__tn )
    w__n = d.mod_model.R.w__n
    a0__n = dR.base_model.C__sn.mean(axis=0)
    c0 = 0
    m = models.SharedCue( dR, G__tn, w__n, a0__n, c0 )
    results.R = Bunch({'w__n': m.w__n, 'c':m.c, 'C__sn':m.C__sn})
    
    # check
    if d.can_load_attribute( target_model ):
        print '- already done'
        
    # save
    setattr( d, target_model, results )
    d.save_attribute( target_model, overwrite=None )



"""
====
jobs
====
"""

# job list
jobs = []
for d_idx in range( len(event_data) ):
    for base_model in ['CGL', 'CG']:
        jobs.append( (fit_sharedC, d_idx, base_model) )

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
