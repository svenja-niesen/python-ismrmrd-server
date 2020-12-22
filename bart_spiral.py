
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64
import ctypes

from bart import bart
import spiraltraj
from cfft import cfftn, cifftn


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")


# from ismrmdrdtools (wip: import from ismrmrdtools instead)
def calculate_prewhitening(noise, scale_factor=1.0):
    '''Calculates the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, nsamples]``
    :scale_factor: Applied on the noise covariance matrix. Used to
                   adjust for effective noise bandwith and difference in
                   sampling rate between noise calibration and actual measurement:
                   scale_factor = (T_acq_dwell/T_noise_dwell)*NoiseReceiverBandwidthRatio

    :returns w: Prewhitening matrix, ``[coil, coil]``, w*data is prewhitened
    '''
    # from scipy.linalg import sqrtm

    noise = noise.reshape((noise.shape[0], noise.size//noise.shape[0]))

    R = np.cov(noise)
    R /= np.mean(abs(np.diag(R)))
    R[np.diag_indices_from(R)] = abs(R[np.diag_indices_from(R)])
    # R = sqrtm(np.linalg.inv(R))
    R = np.linalg.cholesky(np.linalg.inv(R))

    return R


def remove_os(data, axis=0):
    '''Remove oversampling (assumes os factor 2)
    '''
    cut = slice(data.shape[axis]//4, (data.shape[axis]*3)//4)
    data = np.fft.ifft(data, axis=axis)
    data = np.delete(data, cut, axis=axis)
    data = np.fft.fft(data, axis=axis)
    return data


def apply_prewhitening(data, dmtx):
    '''Apply the noise prewhitening matrix

    :param noise: Input noise data (array or matrix), ``[coil, ...]``
    :param dmtx: Input noise prewhitening matrix

    :returns w_data: Prewhitened data, ``[coil, ...]``,
    '''

    s = data.shape
    return np.asarray(np.asmatrix(dmtx)*np.asmatrix(data.reshape(data.shape[0],data.size//data.shape[0]))).reshape(s)
    


def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier
    
    # logging.debug("OMP_NUM_THREADS set!?: %s" % (os.getenv('OMP_NUM_THREADS'),)) 
    # if os.getenv('OMP_NUM_THREADS') is None:
    #     os.environ["OMP_NUM_THREADS"] = "40"

    try:
        # Disabled due to incompatibility between PyXB and Python 3.8:
        # https://github.com/pabigot/pyxb/issues/123
        # # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a field of view of (%s x %s x %s)mm^3 and a matrix size of (%s x %s x %s)", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    # Continuously parse incoming data parsed from MRD messages
    acqGroup = []
    noiseGroup = []
    waveformGroup = []

    # hard-coded limit of 256 slices (better: use Nslice from protocol)
    acsGroup = [[] for _ in range(256)]
    sensmaps = [None] * 256
    dmtx = None

    try:
        for item in connection:
            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # wip: run noise decorrelation
                if item.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
                    noiseGroup.append(item)
                    continue
                elif len(noiseGroup) > 0 and dmtx is None:
                    noise_data = []
                    for acq in noiseGroup:
                        noise_data.append(acq.data)
                    noise_data = np.concatenate(noise_data, axis=1)
                    # calculate pre-whitening matrix
                    dmtx = calculate_prewhitening(noise_data)
                    del(noise_data)
                
                
                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx)


                acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice])
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup = []

            # ----------------------------------------------------------
            # Image data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Image):
                # just pass along
                connection.send_image(item)
                continue

            # ----------------------------------------------------------
            # Waveform data messages
            # ----------------------------------------------------------
            elif isinstance(item, ismrmrd.Waveform):
                waveformGroup.append(item)

            elif item is None:
                break

            else:
                logging.error("Unsupported data type %s", type(item).__name__)

        # Extract raw ECG waveform data. Basic sorting to make sure that data 
        # is time-ordered, but no additional checking for missing data.
        # ecgData has shape (5 x timepoints)
        if len(waveformGroup) > 0:
            waveformGroup.sort(key = lambda item: item.time_stamp)
            ecgData = [item.data for item in waveformGroup if item.waveform_id == 0]
            ecgData = np.concatenate(ecgData,1)

        # Process any remaining groups of raw or image data.  This can 
        # happen if the trigger condition for these groups are not met.
        # This is also a fallback for handling image data, as the last
        # image in a series is typically not separately flagged.
        if len(acqGroup) > 0:
            logging.info("Processing a group of k-space data (untriggered)")
            if sensmaps[item.idx.slice] is None:
                # run parallel imaging calibration
                sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], metadata, dmtx) 
            image = process_raw(acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice])
            connection.send_image(image)
            acqGroup = []

    finally:
        connection.send_close()


def sort_into_kspace(group, metadata, dmtx=None, zf_around_center=False):
    # initialize k-space
    nc = metadata.acquisitionSystemInformation.receiverChannels

    enc1_min, enc1_max = int(999), int(0)
    enc2_min, enc2_max = int(999), int(0)
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        if enc1 < enc1_min:
            enc1_min = enc1
        if enc1 > enc1_max:
            enc1_max = enc1
        if enc2 < enc2_min:
            enc2_min = enc2
        if enc2 > enc2_max:
            enc2_max = enc2

    nx = 2 * metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.x
    # ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z

    kspace = np.zeros([ny, nz, nc, nx], dtype=group[0].data.dtype)
    counter = np.zeros([ny, nz], dtype=np.uint16)

    logging.debug("nx/ny/nz: %s/%s/%s; enc1 min/max: %s/%s; enc2 min/max:%s/%s, ncol: %s" % (nx, ny, nz, enc1_min, enc1_max, enc2_min, enc2_max, group[0].data.shape[-1]))

    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2

        # in case dim sizes smaller than expected, sort data into k-space center (e.g. for reference scans)
        ncol = acq.data.shape[-1]
        cx = nx // 2
        ccol = ncol // 2
        col = slice(cx - ccol, cx + ccol)

        if zf_around_center:
            cy = ny // 2
            cz = nz // 2

            cenc1 = (enc1_max+1) // 2
            cenc2 = (enc2_max+1) // 2

            # sort data into center k-space (assuming a symmetric acquisition)
            enc1 += cy - cenc1
            enc2 += cz - cenc2
        
        if dmtx is None:
            kspace[enc1, enc2, :, col] += acq.data
        else:
            kspace[enc1, enc2, :, col] += apply_prewhitening(acq.data, dmtx)
        counter[enc1, enc2] += 1

    # support averaging (with or without acquisition weighting)
    kspace /= np.maximum(1, counter[:,:,np.newaxis,np.newaxis])

    # rearrange kspace for bart - target size: (nx, ny, nz, nc)
    kspace = np.transpose(kspace, [3, 0, 1, 2])

    return kspace


def rot(mat, n_intl, rot=2*np.pi):
    # rotate spiral gradient
    # trj is a 2D trajectory or gradient arrays ([2, n_samples]), n_intlv is the number of spiral interleaves
    # returns a new trajectory/gradient array with size [n_intl, 2, n_samples]
    phi = np.linspace(0, rot, n_intl, endpoint=False)

    # rot_mat = np.asarray([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    # new orientation (switch x and y):
    rot_mat = np.asarray([[np.cos(phi), np.sin(phi)], [-np.sin(phi), np.cos(phi)]])
    rot_mat = np.moveaxis(rot_mat,-1,0)

    return rot_mat @ mat


def pcs_to_dcs(grads, patient_position='HFS'):
    """ Convert from patient coordinate system (PCS, physical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    """
    grads = grads.copy()

    # only valid for head first/supine - other orientations see IDEA UserGuide
    if patient_position.upper() == 'HFS':
        grads[:,1] *= -1
        grads[:,2] *= -1
    else:
        raise ValueError

    return grads

def dcs_to_pcs(grads, patient_position='HFS'):
    """ Convert from device coordinate system (DCS, physical) 
        to patient coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    """
    return pcs_to_dcs(grads, patient_position) # same sign switch
    
def gcs_to_pcs(grads, rotmat):
    """ Convert from gradient coordinate system (GCS, logical) 
        to patient coordinate system (DCS, physical)
    """
    return np.matmul(rotmat, grads)

def pcs_to_gcs(grads, rotmat):
    """ Convert from patient coordinate system (PCS, physical) 
        to gradient coordinate system (GCS, logical) 
    """
    return np.matmul(np.linalg.inv(rotmat), grads)

def gcs_to_dcs(grads, rotmat):
    """ Convert from gradient coordinate system (GCS, logical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    Parameters
    ----------
    grads : numpy array [3, intl, samples]
            gradient to be converted
    rotmat: numpy array [3,3]
            rotation matrix from quaternion from Siemens Raw Data header
    Returns
    -------
    grads_cv : numpy.ndarray
               Converted gradient
    """
    grads = grads.copy()

    # rotation from GCS (PHASE,READ,SLICE) to patient coordinate system (PCS)
    grads = gcs_to_pcs(grads, rotmat)
    
    # PCS (SAG,COR,TRA) to DCS (X,Y,Z)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    grads = pcs_to_dcs(grads)
    
    return grads


def dcs_to_gcs(grads, rotmat):
    """ Convert from device coordinate system (DCS, logical) 
        to gradient coordinate system (GCS, physical)
        this is valid for patient orientation head first/supine
    Parameters
    ----------
    grads : numpy array [3, intl, samples]
            gradient to be converted
    rotmat: numpy array [3,3]
            rotation matrix from quaternion from Siemens Raw Data header
    Returns
    -------
    grads_cv : numpy.ndarray
               Converted gradient
    """
    grads = grads.copy()
    
    # DCS (X,Y,Z) to PCS (SAG,COR,TRA)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    grads = dcs_to_pcs(grads)
    
    # PCS (SAG,COR,TRA) to GCS (PHASE,READ,SLICE)
    grads = pcs_to_gcs(grads, rotmat)
    
    return grads


def fov_shift_spiral(sig, trj, shift, matr_sz):
    """ 
    shift field of view of spiral data
    sig:  rawdata [ncha, nsamples]
    trj:    trajectory [3, nsamples]
    # shift:   shift [x_shift, y_shift] in voxel
    shift:   shift [y_shift, x_shift] in voxel
    matr_sz: matrix size of reco
    """

    if (abs(shift[0]) < 1e-2) and (abs(shift[1]) < 1e-2):
        # nothing to do
        return sig
    
    kmax = matr_sz/2
    sig *= np.exp(-1j*(shift[1]*np.pi*trj[0]/kmax+shift[0]*np.pi*trj[1]/kmax))[np.newaxis]

    return sig



def intp_axis(newgrid, oldgrid, data, axis=0):
    # interpolation along an axis (shape of newgrid, oldgrid and data see np.interp)
    tmp = np.moveaxis(data.copy(), axis, 0)
    newshape = (len(newgrid),) + tmp.shape[1:]
    tmp = tmp.reshape((len(oldgrid), -1))
    n_elem = tmp.shape[-1]
    intp_data = np.zeros((len(newgrid), n_elem), dtype=data.dtype)
    for k in range(n_elem):
        intp_data[:, k] = np.interp(newgrid, oldgrid, tmp[:, k])
    intp_data = intp_data.reshape(newshape)
    intp_data = np.moveaxis(intp_data, 0, axis)
    return intp_data


def grad_pred(grad, girf):
    """
    gradient prediction with girf
    
    Parameters:
    ------------
    grad: nominal gradient [interleaves, dims, samples]
    girf:     gradient impulse response function [input dims, output dims (incl k0), samples]
    """
    ndim = grad.shape[1]
    grad_sampl = grad.shape[-1]
    girf_sampl = girf.shape[-1]

    # remove k0 from girf:
    girf = girf[:,1:]

    # zero-fill grad to number of girf samples (add check?)
    grad = np.concatenate([grad.copy(), np.zeros([grad.shape[0], ndim, girf_sampl-grad_sampl])], axis=-1)

    # FFT
    grad = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(grad, axes=-1), axis=-1), axes=-1)

    # apply girf to nominal gradients
    pred_grad = np.zeros_like(grad)
    for dim in range(ndim):
        pred_grad[:,dim]=np.sum(grad*girf[np.newaxis,:ndim,dim,:], axis=1)

    # IFFT
    pred_grad = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pred_grad, axes=-1), axis=-1), axes=-1)
    
    # cut out relevant part
    pred_grad = pred_grad[:,:,:grad_sampl]

    return pred_grad


def trap_from_area(area, ramptime, ftoptime, dt_grad=10e-6):
    """create trapezoidal_gradient with selectable gradient moment
    area in [T/m*s]
    ramptime/ftoptime in [s]
    """
    n_ramp = int(ramptime/dt_grad+0.5)
    n_ftop = int(ftoptime/dt_grad+0.5)
    amp = area/(ftoptime+ramptime)
    ramp = np.arange(0.5, n_ramp)/n_ramp
    while np.ndim(ramp) < np.ndim(amp) + 1:
        ramp = ramp[np.newaxis]
    
    ramp = amp[..., np.newaxis] * ramp
        
    zeros = np.zeros(area.shape + (1,))
    grad = np.concatenate((zeros, ramp, amp[..., np.newaxis]*np.ones(area.shape + (n_ftop,)), ramp[...,::-1], zeros), -1)
    return grad

    
def calc_spiral_traj(ncol, rot_mat, encoding):
    dt_grad = 10e-6
    dt_skope = 1e-6
    gammabar = 42.577e6

    traj_params = {item.name: item.value_ for item in encoding.trajectoryDescription.userParameterLong}
    traj_params.update({item.name: item.value_ for item in encoding.trajectoryDescription.userParameterDouble})

    nx = encoding.encodedSpace.matrixSize.x
    fov = encoding.reconSpace.fieldOfView_mm.x
    res = fov/nx
    spiralType = int(traj_params['spiralType'])
    nitlv = int(traj_params['interleaves'])
    max_amp = traj_params['MaxGradient_mT_per_m']
    min_rise = traj_params['MaxRiseTime_mT_per_m_per_ms']
    dwelltime = 1e-6 * traj_params['dwellTime_us']  # s
    spiralOS = traj_params['fSpiralOS_reinterpret_cast_to_int32']  # int_32
    spiralOS = np.frombuffer(np.uint32(spiralOS), 'float32')[0]

    logging.debug("spiralType = %s, nitlv = %s, fov = %s, res = %s, max_amp = %s, min_rise = %s, " % (spiralType, nitlv, fov, res, max_amp, min_rise))

    grad = spiraltraj.calc_traj(nitlv=nitlv, res=res, fov=fov, max_amp=max_amp, min_rise=min_rise, spiraltype=spiralType)
    grad = np.asarray(grad).T # -> [2, nsamples]

    
    adctime = dwelltime * np.arange(0.5, ncol)

    # Determine start of first spiral gradient & first ADC
    adc_shift = dt_skope/2 # empirically determined, seems to work better (may be there is a delay in the skope measurement?)
    if spiralType > 2:
        # new: small timing fix for double spiral
        # align center of gradient & adc
        grad_totaltime = dt_grad * (grad.shape[-1])
        adc_duration = dwelltime * ncol
        adc_shift += np.round((grad_totaltime - adc_duration)/2., 6)
    
    adctime += adc_shift


    gradshift = 0
    if spiralType == 3 or spiralType == 5: # for double spiral or RIO traj.
        # prepend gradient dephaser
        deph_ru = 1e-6 * traj_params['dephRampUp']  # s
        deph_ft = 1e-6 * traj_params['dephFlatTop']  # s
        area = -dt_grad * np.sum(grad[..., :grad.shape[-1]//2], -1)
        dephaser = trap_from_area(area, deph_ru, deph_ft, dt_grad=dt_grad)

        grad = np.concatenate((dephaser, grad), -1)
        gradshift -= dephaser.shape[-1] * dt_grad

    # add zeros around gradient
    zfills = 8
    grad = np.concatenate((np.zeros([2, zfills]), grad, np.zeros([2, zfills])), axis=-1)
    gradshift -= zfills*dt_grad

    # and finally rotate for each interleave
    if spiralType == 3 and nitlv%2==0:
        grad = rot(grad, nitlv, np.pi)
    else:
        grad = rot(grad, nitlv)

    # add z-dir:
    grad = np.concatenate((grad, np.zeros([grad.shape[0], 1, grad.shape[2]])), axis=1)

    # time vector for interpolation
    gradtime = dt_grad * np.arange(grad.shape[-1]) + gradshift

    # calculate base trajectory from gradient (with proper scaling for bart)
    base_trj =  np.cumsum(grad.real, axis=-1)
    gradtime += dt_grad/2  # account for cumsum

    # proper scaling for bart
    base_trj *= 1e-3 * dt_grad * gammabar * (1e-3 * fov)

    # interpolate trajectory to scanner dwelltime
    base_trj = intp_axis(adctime, gradtime, base_trj, axis=-1) # 2.5us seems to be a useful shift

    np.save(os.path.join(debugFolder, "base_trj.npy"), base_trj)


    ##############################
    ## girf trajectory prediction:
    ##############################

    girf = np.load(os.path.join(dependencyFolder, "girf_10us.npy"))

    # rotation to phys coord system
    pred_grad = gcs_to_dcs(grad, rot_mat)

    # gradient prediction
    pred_grad = grad_pred(pred_grad, girf) 

    # rotate back to logical system
    pred_grad = dcs_to_gcs(pred_grad, rot_mat)

    pred_grad[:, 2] = 0. # set z-gradient to zero, otherwise bart reco crashess

    # time vector for interpolation
    gradtime = dt_grad * np.arange(pred_grad.shape[-1]) + gradshift

    # calculate trajectory 
    pred_trj = np.cumsum(pred_grad.real, axis=-1)
    gradtime += dt_grad/2 # account for cumsum

    # proper scaling for bart
    pred_trj *= 1e-3 * dt_grad * gammabar * (1e-3 * fov)

    # interpolate trajectory to scanner dwelltime
    pred_trj = intp_axis(adctime, gradtime, pred_trj, axis=-1)
    
    np.save(os.path.join(debugFolder, "pred_trj.npy"), pred_trj)

    # pred_trj = np.load(os.path.join(dependencyFolder, "spiralout_meas.npy")
    # pred_trj = np.load(os.path.join(dependencyFolder, "traj_doublespiral_r1_skope.npy")
    # pred_trj = np.transpose(pred_trj, [2, 0, 1])

    # now we can switch x and y dir for correct orientation in FIRE
    pred_trj = pred_trj[:,[1,0,2],:]
    base_trj = base_trj[:,[1,0,2],:]

    ## WIP  
    # return base_trj
    return pred_trj


def sort_spiral_data(group, metadata, dmtx=None, remove_ro_os=True):
    
    def calc_rotmat(acq):
        phase_dir = np.asarray(acq.phase_dir)
        read_dir = np.asarray(acq.read_dir)
        slice_dir = np.asarray(acq.slice_dir)
        return np.round(np.concatenate([phase_dir[:,np.newaxis], read_dir[:,np.newaxis], slice_dir[:,np.newaxis]], axis=1), 6)

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    fov = metadata.encoding[0].reconSpace.fieldOfView_mm.x
    res = fov/nx
    ncol = group[0].number_of_samples

    rot_mat = calc_rotmat(group[0])
    base_trj = calc_spiral_traj(ncol, rot_mat, metadata.encoding[0])

    acq_key = [None] * nz
    sig = list()
    trj = list()
    enc = list()
    for key, acq in enumerate(group):
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2

        kz = enc2 - nz//2
        
        # save one header per partition
        if acq_key[enc2] is None:
            acq_key[enc2] = key

        enc.append([enc1, enc2])
        
        # update 3D dir.
        tmp = base_trj[enc1].copy()
        tmp[-1] = kz * np.ones(tmp.shape[-1])
        trj.append(tmp)

        # and append data after optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
        else:
            sig.append(apply_prewhitening(acq.data, dmtx))

        # apply fov shift
        shift = pcs_to_gcs(np.asarray(acq.position)/res, rot_mat)
        sig[-1] = fov_shift_spiral(sig[-1], tmp, shift, nx)

        # we could remove oversampling here (not really necessary after fov shift)

    np.save(os.path.join(debugFolder, "enc.npy"), enc)
    
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)

    if remove_ro_os:
        # remove os before recon, in order to speed it up (fov shifts have already been applied)
        sig = remove_os(sig, -1)
        trj = trj[:,:,::2]

    # rearrange trj & sig for bart - target size: ??? WIP  --(ncol, enc1_max, nz, nc)
    trj = np.transpose(trj, [1, 2, 0])
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis]

    logging.debug("trj.shape = %s, sig.shape = %s"%(trj.shape, sig.shape))
    
    np.save(os.path.join(debugFolder, "trj.npy"), trj)

    return sig, trj, acq_key


def process_acs(group, metadata, dmtx=None):

    if isinstance(group, list) and len(group)>0:  # assume Cartesian reference scan
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)
    elif isinstance(group, np.ndarray):  # assume that data is already sorted
        data = group
    else:
        return None

    afy = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
    afz = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2
    caipi_delta = 0  # wip

    if data.shape[2]>1 and not afz>1 and not (afy>1 and caipi_delta!=0):
        data = cifftn(data.copy(), [2])
        sensmaps = []
        for p in range(data.shape[2]):
            sensmaps.append(bart(1, 'ecalib -m 1 -I', data[:,:,p:p+1,:]))  # ESPIRiT calibration
        sensmaps = np.asarray(sensmaps)[:,:,:,0,:].transpose([1,2,0,3])
    else:
        sensmaps = bart(1, 'ecalib -m 1 -I', data)  # ESPIRiT calibration
        # sensmaps = bart(1, 'ecalib -m 1 -k 8 -I -r 48', data)  # ESPIRiT calibration

    np.save(os.path.join(debugFolder, "acs.npy"), data)
    np.save(os.path.join(debugFolder, "sensmaps.npy"), sensmaps)
    return sensmaps


def process_raw(group, config, metadata, dmtx=None, sensmaps=None):

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    rNx = metadata.encoding[0].reconSpace.matrixSize.x
    rNy = metadata.encoding[0].reconSpace.matrixSize.y
    rNz = metadata.encoding[0].reconSpace.matrixSize.z

    FOVz = metadata.encoding[0].encodedSpace.fieldOfView_mm.z

    data, trj, acq_key = sort_spiral_data(group, metadata, dmtx)

    logging.debug("Raw data is size %s" % (data.shape,))
    logging.debug("nx,ny,nz %s, %s, %s" % (nx, ny, nz))
    np.save(os.path.join(debugFolder, "raw.npy"), data)
    
    afy = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
    afz = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_2

    userparams = {item.name: item.value_ for item in metadata.userParameters.userParameterLong}
    caipi_delta = userparams['lReorderingShift']
    partition_ft_first = (sensmaps is None or (nz>1 and not afz>1 and not (afy>1 and caipi_delta!=0)))
    force_pics = True


    if partition_ft_first:
        # sort data into partitions
        ctrj = list()
        cdata = list()
        for p in range(nz):
            mask = np.round(trj[2,0,:]+nz//2) == p
            ctrj.append(trj[:,:,mask].copy())
            cdata.append(data[:,:,mask,:].copy())

        # perform ft in partition dir
        cdata = cifftn(np.asarray(cdata), [0])
        ctrj = np.asarray(ctrj)
        ctrj[:,2] = 0
        
        if nz > rNz:  # remove oversampling in slice direction
            cdata = cdata[(nz - rNz)//2:-(nz - rNz)//2]
            ctrj = ctrj[(nz - rNz)//2:-(nz - rNz)//2]

        # perform recon
        data = np.zeros([nx, nx, rNz])
        for p in range(rNz):
            if sensmaps is None:
                tmp = bart(1, 'nufft -i -t -c -d %d:%d:%d'%(nx, nx, 1), ctrj[p], cdata[p]) # nufft

                if force_pics:
                    tmp = cfftn(tmp, [0, 1]) # back to k-space
                    sensmap = process_acs(tmp, metadata)
                    data[:,:,p] = abs(bart(1, 'pics -e -l1 -r 0.001 -i 25 -t', ctrj[p], cdata[p], sensmap))
                else:
                    data[:,:,p] = np.sqrt(np.sum(np.abs(tmp)**2, axis=-1))[:,:,0]        
            else:
                data[:,:,p] = abs(bart(1, 'pics -e -l1 -r 0.001 -i 25 -t', ctrj[p], cdata[p], sensmaps[p]))
    else:
        if sensmaps is None:
            print('error: acs data missing')
            raise ValueError
        data = abs(bart(1, 'pics -e -l1 -r 0.001 -i 25 -t', trj, data, sensmaps))
        # data = abs(bart(1, 'pics -e -l1 -r 0.0001 -i 100 -t', trj, data, sensmaps))

        # make sure that data is at least 3d:
        while np.ndim(data) < 3:
            data = data[..., np.newaxis]

        if nz > rNz:  # remove oversampling in slice direction
            data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(os.path.join(debugFolder, "img_%d.npy"%(group[0].idx.repetition)), data)

    # Normalize and convert to int16
    # save one scaling in 'static' variable
    try:
        process_raw.imascale
    except:
        process_raw.imascale = 0.8 / data.max()
    data *= 32767 * process_raw.imascale
    data = np.around(data)
    data = data.astype(np.int16)

    # Set ISMRMRD Meta Attributes
    meta = ismrmrd.Meta({'DataRole':               'Image',
                         'ImageProcessingHistory': ['FIRE', 'PYTHON'],
                         'WindowCenter':           '16384',
                         'WindowWidth':            '32768'})#,
                        #  'Keep_image_geometry':    1})
    xml = meta.serialize()
    
    images = []
    n_par = data.shape[-1]
    logging.debug("data.shape %s" % (data.shape,))

    # Wrong Sequence or Protocol detected, aborting...
    # Number of slices announced via protocol: 1
    # Delivered slice index of current MDH: 1 (allowed: 0..0)
    # Fazit:
    # image.slice auf partition index setzen bringt nichts

    for par in range(n_par):
        # Format as ISMRMRD image data
        image = ismrmrd.Image.from_array(data[...,par], group[acq_key[par]])
        
        field_of_view_z = FOVz
        if n_par>1:
            field_of_view_z /= rNz
        
        image.field_of_view = (ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                               ctypes.c_float(metadata.encoding[0].reconSpace.fieldOfView_mm.x), 
                               ctypes.c_float(field_of_view_z))

        if n_par>1:
            image.image_index = n_par - par
            image.image_series_index = 1 + group[acq_key[par]].idx.repetition
            # image.user_int[0] = par # is this correct??
            # image.user_int[1] = 1 + par # is this correct??
            image.position[-1] += (n_par//2 - par - 0.5) * FOVz / rNz # funktioniert, muss aber noch richtig angepasst werden (+vorzeichen check!!!)
        else:
            image.image_index = 1 + group[acq_key[par]].idx.repetition
        
        image.attribute_string = xml
        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images
