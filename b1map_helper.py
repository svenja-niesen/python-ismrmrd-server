"""
functions for mapping algorithmen
"""

import numpy as np

def B0Map_complex(Ste_comp, FID_comp, te_diff):
    """ Calculates the B0 Map (with the complex images)

    :param Ste_comp: STE image (complex)
    :param FID_comp: FID image (complex)
    :param te_diff: effective echotime difference in [s]
    :returns: B0-Map
    """    
    phasediff = Ste_comp*np.conj(FID_comp) # phase difference between the 2 signals
    # phasediff = np.sum(phasediff,axis=-1) # coil combination - complex sum
    fmap = np.angle(phasediff)/(2*np.pi*te_diff) # ohne sys.gamma
    
    return fmap











