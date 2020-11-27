import ismrmrd
import numpy as np
import os
import argparse

""" function to insert protocol into ismrmrd file

The protocol should be saved as an hdf5 file with groups 'hdr' and 'acquisitions'.
Protocol data, which contain a single number (e.g dwelltime) are stored as attributes,
protocol data, which contain arrays (e.g. acquisition data) are stored as datasets.

"""

def insert_prot(prot_file, data_file): 

    #---------------------------
    # Read protocol and Ismrmrd file
    #---------------------------

    prot = ismrmrd.Dataset(prot_file, create_if_needed=False)
    dset = ismrmrd.Dataset(data_file, create_if_needed=False)

    #---------------------------
    # First process the header 
    #---------------------------

    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())
    dset_hdr = ismrmrd.xsd.CreateFromDocument(dset.read_xml_header())

    dset_udbl = dset_hdr.userParameters.userParameterDouble
    prot_udbl = prot_hdr.userParameters.userParameterDouble
    dset_udbl[0].name = prot_udbl[0].name # dwellTime_us
    dset_udbl[0].value_ = prot_udbl[0].value_
    dset_udbl[1].name = prot_udbl[1].name # traj_delay (additional delay of trajectory [s])
    dset_udbl[1].value_ = prot_udbl[1].value_

    dset_e1 = dset_hdr.encoding[0]
    prot_e1 = prot_hdr.encoding[0]
    dset_e1.trajectory = prot_e1.trajectory

    dset_e1.encodedSpace.matrixSize.x = prot_e1.encodedSpace.matrixSize.x
    dset_e1.encodedSpace.matrixSize.y = prot_e1.encodedSpace.matrixSize.y
    dset_e1.encodedSpace.matrixSize.z =  prot_e1.encodedSpace.matrixSize.z
    
    dset_e1.encodedSpace.fieldOfView_mm.x = prot_e1.encodedSpace.fieldOfView_mm.x
    dset_e1.encodedSpace.fieldOfView_mm.y = prot_e1.encodedSpace.fieldOfView_mm.y
    dset_e1.encodedSpace.fieldOfView_mm.z = prot_e1.encodedSpace.fieldOfView_mm.z
    
    dset_e1.reconSpace.matrixSize.x = prot_e1.reconSpace.matrixSize.x
    dset_e1.reconSpace.matrixSize.y = prot_e1.reconSpace.matrixSize.y
    dset_e1.reconSpace.matrixSize.z = prot_e1.reconSpace.matrixSize.z
    
    dset_e1.reconSpace.fieldOfView_mm.x = prot_e1.reconSpace.fieldOfView_mm.x
    dset_e1.reconSpace.fieldOfView_mm.y = prot_e1.reconSpace.fieldOfView_mm.y
    dset_e1.reconSpace.fieldOfView_mm.z = prot_e1.reconSpace.fieldOfView_mm.z

    # write header back to file
    dset.write_xml_header(dset_hdr.toxml())

    #---------------------------
    # Now process all acquisitions
    #---------------------------

    # first check if number of acquisitions is the same in both files
    if not dset.number_of_acquisitions() == prot.number_of_acquisitions():
        raise ValueError('Number of acquisitions in protocol and data file is not the same.')

    for n in range(dset.number_of_acquisitions()):

        prot_acq = prot.read_acquisition(n)
        dset_acq = dset.read_acquisition(n)

        # rotation matrix
        dset_acq.phase_dir[:] = prot_acq.phase_dir[:]
        dset_acq.read_dir[:] = prot_acq.read_dir[:]
        dset_acq.slice_dir[:] = prot_acq.slice_dir[:]

        # flags - WIP: this is not the complete list of flags - if needed, flags can be added
        if prot_acq.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
            dset_acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        if prot_acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
            dset_acq.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)
        if prot_acq.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
            dset_acq.setFlag(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)
        if prot_acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
            dset_acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
        if prot_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
            dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
        if prot_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
            dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)

        # encoding counters
        dset_acq.idx.kspace_encode_step_1 = prot_acq.idx.kspace_encode_step_1
        dset_acq.idx.kspace_encode_step_2 = prot_acq.idx.kspace_encode_step_2
        dset_acq.idx.slice = prot_acq.idx.slice
        dset_acq.idx.contrast = prot_acq.idx.contrast
        dset_acq.idx.phase = prot_acq.idx.phase
        dset_acq.idx.average = prot_acq.idx.average
        dset_acq.idx.repetition = prot_acq.idx.repetition
        dset_acq.idx.set = prot_acq.idx.set
        dset_acq.idx.segment = prot_acq.idx.segment

        # calculate trajectory with GIRF prediction
        dset_acq.resize(trajectory_dimensions=prot_acq.trajectory_dimensions, number_of_samples=dset_acq.number_of_samples, active_channels=dset_acq.active_channels)
        dset_acq.traj[:] = calc_traj(prot_acq, prot_hdr, dset_acq.number_of_samples) # [samples, dims]

        dset.write_acquisition(dset_acq, n)

    dset.close()
    prot.close()

def calc_traj(acq, hdr, ncol):
    """ Calculates the kspace trajectory from any gradient using Girf prediction and interpolates it on the adc raster

        acq: acquisition from hdf5 protocol file
        hdr: header from hdf5 protocol file
    """

    def calc_rotmat(acq):
        phase_dir = np.asarray(acq.phase_dir)
        read_dir = np.asarray(acq.read_dir)
        slice_dir = np.asarray(acq.slice_dir)
        return np.round(np.concatenate([phase_dir[:,np.newaxis], read_dir[:,np.newaxis], slice_dir[:,np.newaxis]], axis=1), 6)

    dt_grad = 10e-6 # [s]
    dt_skope = 1e-6 # [s]
    gammabar = 42.577e6

    grad = np.swapaxes(acq.traj[:],0,1) # [dims, samples] [T/m]
    dims = grad.shape[0]

    fov = hdr.encoding[0].reconSpace.fieldOfView_mm.x
    rotmat = calc_rotmat(acq)
    dwelltime = 1e-6 * hdr.userParameters.userParameterDouble[0].value_
    gradshift = hdr.userParameters.userParameterDouble[1].value_

    # ADC sampling time
    adctime = dwelltime * np.arange(0.5, ncol)

    # add some zeros around gradient for right interpolation
    zeros = 10
    grad = np.concatenate((np.zeros([dims,zeros]), grad, np.zeros([dims,zeros])), axis=1)
    gradshift -= zeros*dt_grad

    # add z-dir for prediction if necessary
    if dims == 2:
        grad = np.concatenate((grad, np.zeros([1, grad.shape[1]])), axis=0)

    ##############################
    ## girf trajectory prediction:
    ##############################

    filepath = os.path.dirname(os.path.abspath(__file__))
    girf = np.load(filepath + "../dependency/girf_10us.npy")

    # rotation to phys coord system
    grad_phys = gcs_to_dcs(grad, rotmat)

    # gradient prediction
    pred_grad = grad_pred(grad_phys, girf)

    # rotate back to logical system
    pred_grad = dcs_to_gcs(pred_grad, rotmat)

    # time vector for interpolation
    gradtime = dt_grad * np.arange(pred_grad.shape[-1]) + gradshift

    # calculate trajectory 
    pred_trj = np.cumsum(pred_grad.real, axis=1)
    gradtime += dt_grad/2 - dt_skope/2 # account for cumsum - WIP: Is the dt_skope/2 delay necessary?? Comnpare to skope trajectory data

    # proper scaling - WIP: use BART scaling, is this also the Ismrmrd scaling???
    pred_trj *= dt_grad * gammabar * (1e-3 * fov)

    # interpolate trajectory to scanner dwelltime
    pred_trj = intp_axis(adctime, gradtime, pred_trj, axis=1)
    
    if dims == 2:
        return np.swapaxes(pred_trj[:2],0,1) # [samples, dims]
    else:
        return np.swapaxes(pred_trj,0,1) # [samples, dims]

def grad_pred(grad, girf):
    """
    gradient prediction with girf
    
    Parameters:
    ------------
    grad: nominal gradient [dims, samples]
    girf: gradient impulse response function [input dims, output dims (incl k0), samples]
    """
    ndim = grad.shape[0]
    grad_sampl = grad.shape[-1]
    girf_sampl = girf.shape[-1]

    # remove k0 from girf:
    girf = girf[:,1:]

    # zero-fill grad to number of girf samples (add check?)
    grad = np.concatenate([grad.copy(), np.zeros([ndim, girf_sampl-grad_sampl])], axis=-1)

    # FFT
    grad = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(grad, axes=-1), axis=-1), axes=-1)

    # apply girf to nominal gradients
    pred_grad = np.zeros_like(grad)
    for dim in range(ndim):
        pred_grad[dim]=np.sum(grad*girf[np.newaxis,:ndim,dim,:], axis=1)

    # IFFT
    pred_grad = np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(pred_grad, axes=-1), axis=-1), axes=-1)
    
    # cut out relevant part
    pred_grad = pred_grad[:,:grad_sampl]

    return pred_grad

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


def pcs_to_dcs(grads, patient_position='HFS'):
    """ Convert from patient coordinate system (PCS, physical) 
        to device coordinate system (DCS, physical)
        this is valid for patient orientation head first/supine
    """
    grads = grads.copy()

    # only valid for head first/supine - other orientations see IDEA UserGuide
    if patient_position.upper() == 'HFS':
        grads[1] *= -1
        grads[2] *= -1
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

class HDF5File(argparse.FileType):
    def __call__(self, string):
        _, ext = os.path.splitext(string)

        if ext == '':
            string = string + '.h5'  # .h5 is default file extension
        else:
            if (str.lower(ext) != '.h5' and str.lower(ext) != '.hdf5'):
                parser.error('hdf5 file %s should have a .h5 extension' % (string))

        returnFile = super(HDF5File, self).__call__(string)
        returnFile.close()
        returnFile = os.path.abspath(returnFile.name)
        return returnFile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="compresses and decompresses Siemens raw data files (.dat)")
       
    parser.add_argument('-p', '--prot_file', type=HDF5File(),
                            help='Protocol file (ISMRMRD)', required=True)
    parser.add_argument('-d', '--data_file', type=HDF5File(),
                            help='Data file (ISMRMRD)', required=True)
    args = parser.parse_args()

    insert_prot(args.prot_file, args.data_file)
