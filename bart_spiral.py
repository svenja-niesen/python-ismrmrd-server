
import ismrmrd
import os
import itertools
import logging
import numpy as np
import numpy.fft as fft
import base64

from bart import bart
import spiraltraj


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
    
    # M = float(noise.shape[1])
    # dmtx = (1/(M-1))*np.asmatrix(noise)*np.asmatrix(noise).H
    # dmtx = np.linalg.inv(np.linalg.cholesky(dmtx))
    # dmtx = dmtx*np.sqrt(2)*np.sqrt(scale_factor)
    
    # channel-wise scaling correct? (conjugate) transpose ok?
    # return dmtx

    R = np.cov(noise)
    R /= np.mean(abs(np.diag(R)))
    R[np.diag_indices_from(R)] = abs(R[np.diag_indices_from(R)])
    # R = sqrtm(np.linalg.inv(R))
    R = np.linalg.cholesky(np.linalg.inv(R))

    # return R
    return None # still untested and potentially buggy


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
    


# Folder for debug output files
debugFolder = "/tmp/share/debug"

def process(connection, config, metadata):
    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

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
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    continue


                acqGroup.append(item)

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    if sensmaps[item.idx.slice] is None:
                        # run parallel imaging calibration
                        sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx)
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
                sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx) 
            image = process_raw(acqGroup, config, metadata, dmtx, sensmaps[item.idx.slice])
            logging.debug("Sending image to client:\n%s", image)
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

    rot_mat = np.asarray([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
    rot_mat = np.moveaxis(rot_mat,-1,0)

    return rot_mat @ mat


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
    grads_cv = np.matmul(rotmat, grads)
    
    # PCS (SAG,COR,TRA) to DCS (X,Y,Z)
    # only valid for head first/supine - other orientations see IDEA UserGuide
    grads_cv[:,1] *= -1
    grads_cv[:,2] *= -1
    
    return grads_cv


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
    grads[:,1] *= -1
    grads[:,2] *= -1
    
    # PCS (SAG,COR,TRA) to GCS (PHASE,READ,SLICE)
    grads_cv = np.matmul(np.linalg.inv(rotmat), grads)
    
    return grads_cv


def intp_axis(newgrid, oldgrid, data, axis=0):
    # interpolation along an axis (shape of newgrid, oldgrid and data see np.interp)
    temp = np.moveaxis(data.copy(), axis, 0)
    newshape = [len(newgrid)] + list(temp.shape[1:])
    temp = temp.reshape((temp.shape[0], -1))
    intp_data = np.zeros((len(newgrid), temp.shape[-1]), dtype=data.dtype)
    for k in range(temp.shape[-1]):
        intp_data[:, k] = np.interp(newgrid, oldgrid, temp[:, k])
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
    girf = girf.copy()[:,1:]

    # zero-fill grad to number of girf samples (add check?)
    grad = np.concatenate([grad.copy(), np.zeros([grad.shape[0], ndim, girf_sampl-grad_sampl])], axis=-1)

    # FFT
    grad = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(grad, axes=-1), axis=-1), axes=-1)
    print('girf.shape =%s, grad.shape = %s'%(girf.shape, grad.shape))

    # apply girf to nominal gradients
    grad_pred = np.zeros_like(grad)
    for dim in range(ndim):
        grad_pred[:,dim]=np.sum(grad*girf[np.newaxis,:ndim,dim,:], axis=1)

    # IFFT
    grad_pred = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(grad_pred, axes=-1), axis=-1), axes=-1)
    
    # cut out relevant part
    grad_pred = grad_pred[:,:,:grad_sampl]

    return grad_pred


def calc_spiral_traj(ncol, dwelltime, nitlv, res, fov, rot_mat, max_amp, min_rise, spiralType):
    dt_grad = 10e-6
    dt_skope = 1e-6
    gammabar = 42.577e6

    logging.debug("spiralType = %s, nitlv = %s, fov = %s, res = %s, max_amp = %s, min_rise = %s, " % (spiralType, nitlv, fov, res, max_amp, min_rise))

    grad = spiraltraj.calc_traj(nitlv=nitlv, res=res, fov=fov, max_amp=max_amp, min_rise=min_rise, spiraltype=spiralType)
    grad = np.asarray(grad).T # -> [2, nsamples]
    # add zeros around gradient
    grad = np.concatenate((np.zeros([2,1]), grad, np.zeros([2,1])), axis=-1)

    # Determine start of first spiral gradient & first ADC
    adc_shift = 0
    if spiralType > 2:
        # new: small timing fix for double spiral
        # align center of gradient & adc
        grad_totaltime = dt_grad * (grad.shape[-1]-2)
        adc_duration = dwelltime * ncol
        adc_shift = np.round((grad_totaltime - adc_duration)/2., 6)

    # and finally rotate for each interleave
    if spiralType == 3:
        grad = rot(grad, nitlv, np.pi)
    else:
        grad = rot(grad, nitlv)

    # add z-dir:
    grad =  np.concatenate((grad, np.zeros([grad.shape[0], 1, grad.shape[2]])), axis=1)
    

    # WIP: THE FOLLOWING LINE IS ONLY REQUIRED FOR OLD DATA, SWITCH X & Y
    grad = grad[:, [1,0,2], :]

    ##############################
    ## girf trajectory prediction:
    ##############################

    filepath = os.path.dirname(os.path.abspath(__file__))
    girf = np.load(filepath + "/girf/girf_10us.npy")

    # rotation to phys coord system
    pred_grad = dcs_to_gcs(grad, rot_mat)

    # gradient prediction
    pred_grad = grad_pred(pred_grad, girf) 

    # rotate back to logical system
    pred_grad = gcs_to_dcs(pred_grad, rot_mat)
    pred_grad[:, 2] = 0. # set z-gradient to zero, otherwise bart reco crashes

    # calculate k-space trajectory
    pred_trj =  np.cumsum(pred_grad.real, axis=-1)
    if spiralType > 2:
        # add dephasing moment
        # deph_ru = metadata.encoding[0].trajectoryDescription.userParameterLong[2].value_  # us
        # deph_ft = metadata.encoding[0].trajectoryDescription.userParameterLong[3].value_  # us
        pred_trj -= np.sum(grad, axis=-1)[:,:,np.newaxis]/2.
    # proper scaling for bart
    pred_trj *= 1e-3 * dt_grad * gammabar * (1e-3 * fov)

    # interpolate trajectory to scanner dwelltime
    gradtime = dt_grad * np.arange(grad.shape[-1])
    gradtime -= dt_grad # account for zero-fill
    adctime = dwelltime * np.arange(0.5, ncol) + adc_shift    
    adctime_girf = adctime - (dt_grad-dt_skope)/2 # subtract dt_grad/2 for 10us GIRF

    pred_trj = intp_axis(adctime_girf, gradtime, pred_trj, axis=-1)

    # calculate base trajectory from gradient (with proper scaling for bart)
    base_trj =  np.cumsum(grad.real, axis=-1)
    if spiralType > 2:
        # add dephasing moment
        base_trj -= np.sum(grad, axis=-1)[:,:,np.newaxis]/2.
    # proper scaling for bart
    base_trj *= 1e-3 * dt_grad * gammabar * (1e-3 * fov)
    base_trj = intp_axis(adctime, gradtime, base_trj, axis=-1)

    np.save(debugFolder + "/" + "gradtime.npy", gradtime)
    np.save(debugFolder + "/" + "adctime.npy", adctime)
    np.save(debugFolder + "/" + "adctime_girf.npy", adctime_girf)
    np.save(debugFolder + "/" + "base_trj.npy", base_trj)
    np.save(debugFolder + "/" + "pred_trj.npy", pred_trj)

    # now we can switch x and y dir for correct orientation in FIRE
    pred_trj = pred_trj[:,[1,0,2],:]
    base_trj = base_trj[:,[1,0,2],:]

    ## WIP  
    return base_trj
    # return pred_trj


def sort_spiral_data(group, metadata, dmtx=None):

    # spiral_af = metadata.encoding[0].parallelImaging.accelerationFactor.kspace_encoding_step_1
    ncol = group[0].number_of_samples
    nz = metadata.encoding[0].encodedSpace.matrixSize.z    
    spiralType = int(metadata.encoding[0].trajectoryDescription.userParameterLong[1].value_)
    nitlv = int(metadata.encoding[0].trajectoryDescription.userParameterLong[0].value_)
    fov = metadata.encoding[0].reconSpace.fieldOfView_mm.x
    res = fov/metadata.encoding[0].encodedSpace.matrixSize.x
    max_amp = metadata.encoding[0].trajectoryDescription.userParameterDouble[0].value_
    min_rise = metadata.encoding[0].trajectoryDescription.userParameterDouble[1].value_
    dwelltime = 1e-6 * metadata.encoding[0].trajectoryDescription.userParameterDouble[2].value_  # s
    spiralOS = metadata.encoding[0].trajectoryDescription.userParameterLong[4].value_  # int_32
    spiralOS = np.frombuffer(np.uint32(spiralOS), 'float32')[0]

    phase_dir = np.asarray(group[0].phase_dir)
    read_dir = np.asarray(group[0].read_dir)
    slice_dir = np.asarray(group[0].slice_dir)
    
    print("read_dir = [%f, %f, %f], phase_dir = [%f, %f, %f], slice_dir = [%f, %f, %f]"%(read_dir[0], read_dir[1], read_dir[2], phase_dir[0], phase_dir[1], phase_dir[2], slice_dir[0], slice_dir[1], slice_dir[2]))

    rot_mat = np.round(np.concatenate([phase_dir[:,np.newaxis], read_dir[:,np.newaxis], slice_dir[:,np.newaxis]], axis=1), 6)

    base_trj = calc_spiral_traj(ncol, dwelltime, nitlv, res, fov, rot_mat, max_amp, min_rise, spiralType)

    sig = list()
    trj = list()
    enc = list()
    for acq in group:
        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        kz = enc2 - nz//2
        
        enc.append([enc1, enc2])
        
        # update 3D dir.
        tmp = base_trj[enc1]
        tmp[-1] = kz * np.ones(tmp.shape[-1])
        trj.append(tmp)    

        # and append data after optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
        else:
            sig.append(apply_prewhitening(acq.data, dmtx))

    np.save(debugFolder + "/" + "enc.npy", enc)
    
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)

    # rearrange trj & sig for bart - target size: ??? WIP  --(ncol, enc1_max, nz, nc)
    trj = np.transpose(trj, [1, 2, 0])
    sig = np.transpose(sig[np.newaxis], [0, 3, 1, 2])

    np.save(debugFolder + "/" + "trj.npy", trj)
    logging.debug("trj.shape = %s, sig.shape = %s"%(trj.shape, sig.shape))

    return sig, trj


def process_acs(group, config, metadata, dmtx=None):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)
        sensmaps = bart(1, 'ecalib -m 1 -I ', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None


def process_raw(group, config, metadata, dmtx=None, sensmaps=None):

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    rNx = metadata.encoding[0].reconSpace.matrixSize.x
    rNy = metadata.encoding[0].reconSpace.matrixSize.y
    rNz = metadata.encoding[0].reconSpace.matrixSize.z

    data, trj = sort_spiral_data(group, metadata, dmtx)

    logging.debug("Raw data is size %s" % (data.shape,))
    logging.debug("nx,ny,nz %s, %s, %s" % (nx, ny, nz))
    np.save(debugFolder + "/" + "raw.npy", data)
    
    # if sensmaps is None: # assume that this is a fully sampled scan (wip: only use autocalibration region in center k-space)
        # sensmaps = bart(1, 'ecalib -m 1 -I ', data)  # ESPIRiT calibration

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard recon")
        
        # bart nufft with nominal trajectory
        data = bart(1, 'nufft -i -t -d %d:%d:%d'%(nx, nx, nz), trj, data) # nufft
        # data = bart(1, 'nufft -i -t', trj, data) # nufft

        # Sum of squares coil combination
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
    else:
        data = bart(1, 'pics -e -l1 -r 0.001 -i 25 -t', trj, data, sensmaps)
        data = np.abs(data)
    
    if nz > rNz:
        # remove oversampling in slice direction
        data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

    logging.debug("Image data is size %s" % (data.shape,))
    np.save(debugFolder + "/" + "img.npy", data)

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
                         'WindowWidth':            '32768'})
    xml = meta.serialize()

    images = []
    n_par = data.shape[-1]
    for par in range(n_par):
        # Format as ISMRMRD image data
        image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
        image.image_index = par + group[0].idx.repetition * n_par
        image.slice = par
        image.attribute_string = xml
        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images
