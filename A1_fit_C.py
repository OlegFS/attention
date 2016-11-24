# modules
import attention
reload(attention)
from attention import *
import pop

# core data
if not locals().has_key('event_data'):
    event_data = cPickle.load(open('data/event_data.pickle'))

# fit CueAlone model
for d in progress.numbers( event_data ):
    m = models.CueAlone( d, verbose=False )
    d.C = BaseModel({'C__sn':m.C__sn})
    d.save_attribute('C')
