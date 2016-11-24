import attention
reload(attention)
from attention import *

import time_course


"""
=========
Load data
=========
"""

event_data = cPickle.load(open('data/event_data.pickle'))
N_data = len(event_data)


N_passes = 40



def compute_CGM_2_omega__i( d_idx, check=False ):
    # basic
    d = event_data[ d_idx ]
    attr_to_check = 'CGM_2_omega__i_60_260'
    #attr_to_check = 'CGM_2_omega__i_m100_450'
    # check
    if check:
        return d.can_load_attribute( attr_to_check )
    if d.can_load_attribute( attr_to_check ):
        print '[already done]'
        return None
    # get data
    for k in ['CG', 'CGM_2', 'tau_CGM_2']:
        d.load_attribute(k)
    # load spikes
    d.load_attribute('Y__tni')
    d.Y__tni = d.Y__tni[:, :, 16:36].astype(float)
    #d.Y__tni = d.Y__tni[:, :, :56].astype(float)
    I = d.Y__tni.shape[-1]

    # to save in
    results = Bunch()

    # extract data
    Omega__tn = d.CGM_2.Omega__tn
    tau = d.tau_CGM_2.tau
    Y__tni = d.Y__tni
    G__tn = d.CG.G__tn * d.S__ts.dot(d.CGM_2.Nu__sn)

    # a, Omega
    a__ni = log( Y__tni.sum(axis=0) / G__tn.sum(axis=0)[:, None] )
    Omega_squared__tn = Omega__tn ** 2

    # passes
    results = Bunch()
    results.passes = passes = []

    for pass_idx in progress.dots(range(N_passes)):

        # re-estimate omega
        if pass_idx > 0:
            # fit
            r = time_course.TimeCourse( 
                    d, Y__tni, omega_i / norm(omega_i), Omega__tn, 
                    a__ni, G__tn, tau=tau, verbose=False )
            m = Model({'Omega__tn':r.Omega__tn, 'rank':2})
            # extract a, Omega
            a__ni = r.a__ni
            Omega__tn = m.Omega__tn
            Omega_squared__tn = Omega__tn ** 2

        # solve for omega_i
        omega_i = np.zeros(I)
        err_i = np.zeros(I)
        for i in range(I):
            # for this bin
            a__n = a__ni[:, i]
            Y__tn = Y__tni[:, :, i]
            Z = (Y__tn * Omega__tn).sum()
            A__tn = np.exp( a__n[None, :] ) * G__tn
            AO = A__tn * Omega__tn
            AOO = A__tn * Omega_squared__tn
            # objective
            def LL( w ):
                return (-A__tn * np.exp( w * Omega__tn )).sum() + w * Z
            def negLL( w ):
                return -LL(w)
            # jacobian
            def dLL( w ):
                return (-AO * np.exp( w * Omega__tn )).sum() + Z
            def dnegLL( w ):
                return -A([ dLL( w ) ])
            # hessian
            def d2LL( w ):
                return (-AOO * np.exp( w * Omega__tn )).sum()
            def d2negLL( w ):
                return -d2LL( w )
            # newton steps
            w = 0
            for _ in range(2):
                w -= dnegLL(w) / d2negLL(w)
            omega_i[i] = w
            err_i[i] = np.sqrt( 1 / d2negLL(w) )

        # save this result
        passes.append( 
            Bunch({ 'omega_i':omega_i, 'err_i':err_i }) )
            # 'a__ni':a__ni, 'Omega__tn':Omega__tn, 

        if attr_to_check in d.available_attributes:
            print '[already done]'
            return None

    results.update( **passes[-1] )

    # save
    setattr( d, attr_to_check, results )
    d.save_attribute( attr_to_check )

# job list
jobs = []
for d_idx in np.random.permutation( range( N_data ) ):
    for f in [ compute_CGM_2_omega__i ]:
        jobs.append((f, d_idx))


if __name__ == '__main__':
    # run
    for j in progress.numbers( jobs ):
        func, d = j
        func(d)
    pass
