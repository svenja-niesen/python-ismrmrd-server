
import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64

from bart import bart
from PowerGridPy import PowerGridIsmrmrd
from cfft import cfftn, cifftn


# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, metadata):
    
    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    protFolder = os.path.join(dependencyFolder, "pulseq_protocols")
    protFolder_local = "/tmp/local/pulseq_protocols" # Protocols mountpoint (not at the scanner)
    prot_filename = metadata.userParameters.userParameterString[0].value_ # protocol filename from Siemens protocol parameter tFree

    # Check if local protocol folder is available - if not use protFolder (scanner)
    date = prot_filename.split('_')[0] # folder in ParametersProcessedData (=date of seqfile)
    protFolder_loc = os.path.join(protFolder_local, date)
    if os.path.exists(protFolder_loc):
        protFolder = protFolder_loc

    # Insert protocol header
    prot_file = protFolder + "/" + prot_filename
    insert_hdr(prot_file, metadata)

    # define variables for FOV shift
    nsegments = metadata.userParameters.userParameterDouble[2].value_
    matr_sz = metadata.encoding[0].encodedSpace.matrixSize.x
    res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / matr_sz

    # Write temporary ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)
    dset_tmp.write_xml_header(metadata.toxml())

    # Insert Field Map
    fmap_path = dependencyFolder+"/fmap.npy"
    if not os.path.exists(fmap_path):
        raise ValueError("No field map file in dependency folder.")
    fmap = np.load(fmap_path)
    dset_tmp.append_array('FieldMap', fmap) # dimensions in PowerGrid seem to be [slices/nz,ny,nx]

    logging.info("Config: \n%s", config)

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("First encoding is of type '%s', with a matrix size of (%s x %s x %s) and a field of view of (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)

    # Continuously parse incoming data parsed from MRD messages
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    
    acqGroup = [[] for _ in range(n_slc)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    sensmaps = [None] * n_slc
    dmtx = None
    base_trj_ = []

    try:
        for acq_ctr, item in enumerate(connection):

            # ----------------------------------------------------------
            # Raw k-space data messages
            # ----------------------------------------------------------
            if isinstance(item, ismrmrd.Acquisition):

                # insert acquisition protocol
                base_trj = insert_acq(prot_file, item, acq_ctr)
                if base_trj is not None: # base_trj is calculated e.g. for future trajectory comparisons
                    base_trj_.append(base_trj)

                # run noise decorrelation
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
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx) # [nx,ny,nz,nc]

                # apply noise whitening (can be done before concatenation of segments)
                item.data[:] = apply_prewhitening(item.data[:], dmtx)

                if item.idx.segment == 0:
                    acqGroup[item.idx.slice].append(item)
                else:
                    # append data to first segment of ADC group
                    idx_lower = item.idx.segment * item.number_of_samples
                    idx_upper = (item.idx.segment+1) * item.number_of_samples
                    acqGroup[item.idx.slice][-1].data[:,idx_lower:idx_upper] = item.data[:]
                
                # fov shift - not yet working as scanner fov shift has to be deactivated
                # if item.idx.segment == nsegments - 1:
                #     rotmat = calc_rotmat(item)
                #     shift = pcs_to_gcs(np.asarray(item.position), rotmat) / res
                #     unshifted_data = acqGroup[item.idx.slice][-1].data[:]
                #     traj = acqGroup[item.idx.slice][-1].traj[:]
                #     acqGroup[item.idx.slice][-1].data[:] = fov_shift_spiral(unshifted_data, traj, shift, matr_sz)

                # if no refscan, calculate sensitivity maps from raw data
                if sensmaps[item.idx.slice] is None:
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                        sensmaps[item.idx.slice] = sens_from_raw(acqGroup[item.idx.slice])

                # When all acquisitions are processed, write them to file for PowerGrid Reco,
                # which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(dset_tmp, sensmaps, acqGroup)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)

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
        if item is not None:
            logging.info("There was untriggered k-space data that will not get processed.")
            acqGroup = []

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def insert_hdr(prot_file, metadata): 

    #---------------------------
    # Read protocol
    #---------------------------

    try:
        prot = ismrmrd.Dataset(prot_file+'.hdf5', create_if_needed=False)
    except:
        try:
            prot = ismrmrd.Dataset(prot_file+'.h5', create_if_needed=False)
        except:
            raise ValueError('Pulseq protocol file not found.')

    #---------------------------
    # Process the header 
    #---------------------------

    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())

    dset_udbl = metadata.userParameters.userParameterDouble
    prot_udbl = prot_hdr.userParameters.userParameterDouble
    dset_udbl[0].name = prot_udbl[0].name # dwellTime_us
    dset_udbl[0].value_ = prot_udbl[0].value_
    dset_udbl[1].name = prot_udbl[1].name # traj_delay (additional delay of trajectory [s])
    dset_udbl[1].value_ = prot_udbl[1].value_
    dset_udbl[2].name = prot_udbl[2].name # nsegments
    dset_udbl[2].value_ = prot_udbl[2].value_

    dset_e1 = metadata.encoding[0]
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

    dset_e1.encodingLimits.slice.minimum = prot_e1.encodingLimits.slice.minimum
    dset_e1.encodingLimits.slice.maximum = prot_e1.encodingLimits.slice.maximum
    dset_e1.encodingLimits.slice.center = prot_e1.encodingLimits.slice.center

    if prot_e1.encodingLimits.kspace_encoding_step_1 is not None:
        dset_e1.encodingLimits.kspace_encoding_step_1.minimum = prot_e1.encodingLimits.kspace_encoding_step_1.minimum
        dset_e1.encodingLimits.kspace_encoding_step_1.maximum = prot_e1.encodingLimits.kspace_encoding_step_1.maximum
        dset_e1.encodingLimits.kspace_encoding_step_1.center = prot_e1.encodingLimits.kspace_encoding_step_1.center
    if prot_e1.encodingLimits.average is not None:
        dset_e1.encodingLimits.average.minimum = prot_e1.encodingLimits.average.minimum
        dset_e1.encodingLimits.average.maximum = prot_e1.encodingLimits.average.maximum
        dset_e1.encodingLimits.average.center = prot_e1.encodingLimits.average.center
    if prot_e1.encodingLimits.phase is not None:
        dset_e1.encodingLimits.phase.minimum = prot_e1.encodingLimits.phase.minimum
        dset_e1.encodingLimits.phase.maximum = prot_e1.encodingLimits.phase.maximum
        dset_e1.encodingLimits.phase.center = prot_e1.encodingLimits.phase.center
    if prot_e1.encodingLimits.contrast is not None:
        dset_e1.encodingLimits.contrast.minimum = prot_e1.encodingLimits.contrast.minimum
        dset_e1.encodingLimits.contrast.maximum = prot_e1.encodingLimits.contrast.maximum
        dset_e1.encodingLimits.contrast.center = prot_e1.encodingLimits.contrast.center

    prot.close()

def insert_acq(prot_file, dset_acq, acq_ctr):

    #---------------------------
    # Read protocol
    #---------------------------

    try:
        prot = ismrmrd.Dataset(prot_file+'.hdf5', create_if_needed=False)
    except:
        try:
            prot = ismrmrd.Dataset(prot_file+'.h5', create_if_needed=False)
        except:
            raise ValueError('Pulseq protocol file not found.')

    prot_hdr = ismrmrd.xsd.CreateFromDocument(prot.read_xml_header())

    #---------------------------
    # Process acquisition
    #---------------------------

    prot_acq = prot.read_acquisition(acq_ctr)

    # rotation matrix
    dset_acq.phase_dir[:] = prot_acq.phase_dir[:]
    dset_acq.read_dir[:] = prot_acq.read_dir[:]
    dset_acq.slice_dir[:] = prot_acq.slice_dir[:]

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

    # flags - WIP: this is not the complete list of flags - if needed, flags can be added
    if prot_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
        dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_SLICE)
    if prot_acq.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
        dset_acq.setFlag(ismrmrd.ACQ_LAST_IN_REPETITION)
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_NOISE_MEASUREMENT):
        dset_acq.setFlag(ismrmrd.ACQ_IS_NOISE_MEASUREMENT)
        prot.close()
        return
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
        dset_acq.setFlag(ismrmrd.ACQ_IS_PHASECORR_DATA)
        prot.close()
        return
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA):
        dset_acq.setFlag(ismrmrd.ACQ_IS_DUMMYSCAN_DATA)
        prot.close()
        return
    if prot_acq.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
        dset_acq.setFlag(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION)
        prot.close()
        return

    # calculate trajectory with GIRF prediction - trajectory is stored only in first segment
    base_trj = None
    if dset_acq.idx.segment == 0:
        # some parameters from the header
        nsamples = dset_acq.number_of_samples
        nsegments = prot_hdr.userParameters.userParameterDouble[2].value_
        nsamples_full = int(nsamples*nsegments+0.5)
        dwelltime = 1e-6*prot_hdr.userParameters.userParameterDouble[0].value_ # [s]
        t_min = prot_hdr.userParameters.userParameterDouble[3].value_ # [s]
        t_vec = t_min + dwelltime * np.arange(nsamples_full) # time vector for B0 correction

        # save data as it gets corrupted by the resizing, dims are [nc, samples]
        data_tmp = dset_acq.data[:] 

        dset_acq.resize(trajectory_dimensions=4, number_of_samples=nsamples_full, active_channels=dset_acq.active_channels)
        dset_acq.data[:] = np.concatenate((data_tmp, np.zeros([dset_acq.active_channels, nsamples_full - nsamples])), axis=-1) # fill extended part of data with zeros
        pred_trj, base_trj = calc_traj(prot_acq, prot_hdr, nsamples_full) # [samples, dims]
        dset_acq.traj[:,:3] = pred_trj.copy()
        dset_acq.traj[:,3] = t_vec

        prot.close()
    
    return base_trj
 

def calc_traj(acq, hdr, ncol):
    """ Calculates the kspace trajectory from any gradient using Girf prediction and interpolates it on the adc raster

        acq: acquisition from hdf5 protocol file
        hdr: header from hdf5 protocol file
        ncol: number of samples
    """
    
    dt_grad = 10e-6 # [s]
    dt_skope = 1e-6 # [s]
    gammabar = 42.577e6

    grad = np.swapaxes(acq.traj[:],0,1) # [dims, samples] [T/m]
    dims = grad.shape[0]

    fov = hdr.encoding[0].reconSpace.fieldOfView_mm.x
    rotmat = calc_rotmat(acq)
    dwelltime = 1e-6 * hdr.userParameters.userParameterDouble[0].value_
    gradshift = hdr.userParameters.userParameterDouble[1].value_ # delay before trajectory begins

    # ADC sampling time
    adctime = dwelltime * np.arange(0.5, ncol)

    # add some zeros around gradient for right interpolation
    zeros = 10
    grad = np.concatenate((np.zeros([dims,zeros]), grad, np.zeros([dims,zeros])), axis=1)
    gradshift -= zeros*dt_grad

    # time vector for interpolation
    gradtime = dt_grad * np.arange(grad.shape[-1]) + gradshift

    # add z-dir for prediction if necessary
    if dims == 2:
        grad = np.concatenate((grad, np.zeros([1, grad.shape[1]])), axis=0)

    ##############################
    ## girf trajectory prediction:
    ##############################

    filepath = os.path.dirname(os.path.abspath(__file__))
    girf = np.load(os.path.join(dependencyFolder, "girf_10us.npy"))

    # rotation to phys coord system
    grad_phys = gcs_to_dcs(grad, rotmat)

    # gradient prediction
    pred_grad = grad_pred(grad_phys, girf)

    # rotate back to logical system
    pred_grad = dcs_to_gcs(pred_grad, rotmat)

    # calculate trajectory 
    pred_trj = np.cumsum(pred_grad.real, axis=1)
    base_trj = np.cumsum(grad, axis=1)

    # set z-axis if trajectory is two-dimensional
    if dims == 2:
        nz = hdr.encoding[0].encodedSpace.matrixSize.z
        partition = acq.idx.kspace_encode_step_2
        kz = partition - nz//2
        pred_trj[2] =  kz * np.ones(pred_trj.shape[1])

    # account for cumsum (assumes rects for integration, we have triangs) - dt_skope/2 seems to be necessary
    gradtime += dt_grad/2 - dt_skope/2

    # proper scaling - WIP: use BART scaling, is this also the Ismrmrd scaling???
    pred_trj *= dt_grad * gammabar * (1e-3 * fov)
    base_trj *= dt_grad * gammabar * (1e-3 * fov)

    # interpolate trajectory to scanner dwelltime
    base_trj = intp_axis(adctime, gradtime, base_trj, axis=1)
    pred_trj = intp_axis(adctime, gradtime, pred_trj, axis=1)

    # switch array order to [samples, dims]
    pred_trj = np.swapaxes(pred_trj,0,1)
    base_trj = np.swapaxes(base_trj,0,1)

    return pred_trj, base_trj

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

def process_raw(dset_tmp, sensmaps, acqGroup):

    # write sensitivity maps
    sens = np.transpose(np.stack(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx] - only tested for 2D, nx/ny might be changed depending on orientation
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # write acquisitions in slice order (might not be chronological, but that makes no difference)
    for k, slc in enumerate(acqGroup):
        for j, acq in enumerate(slc):
            dset_tmp.append_acquisition(acq)
    dset_tmp.close()

    # process with PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    debug_pg = debugFolder+"/powergrid_tmp"
    if not os.path.exists(debug_pg):
        os.makedirs(debug_pg)
    data = PowerGridIsmrmrd(inFile=tmp_file, outFile=debug_pg+"/img", niter=15, beta=0, timesegs=7, TSInterp='histo')
    shapes = data["shapes"] 
    data = np.asarray(data["img_data"]).reshape(shapes)
    data = np.abs(data)

    # data should have output [Slice, Phase, Echo, Avg, Rep, Nz, Ny, Nx]
    # change to [Avg, Rep, Phase, Echo, Slice, Nz, Ny, Nx] and average
    data = np.transpose(data, [3,4,1,2,0,5,6,7]).mean(axis=0)

    logging.debug("Image data is size %s" % (data.shape,))

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

    # Format as ISMRMRD image data
    for rep in range(data.shape[0]):
        for phs in range(data.shape[1]):
            for echo in range(data.shape[2]):
                for slc in range(data.shape[3]):
                    for nz in range(data.shape[4]):
                        image = ismrmrd.Image.from_array(data[rep,phs,echo,slc,nz], acquisition=acqGroup[slc][0])
                        image.image_index = 1 + slc*data.shape[4] + nz # contains image index (slices/partitions)
                        image.image_series_index = 1 + rep*data.shape[1]*data.shape[2] + phs*data.shape[2] + echo # contains image series index, e.g. different contrasts
                        image.slice = 0 # WIP: test counting slices, contrasts, ... at scanner
                        image.attribute_string = xml
                        images.append(image)

    logging.debug("Image MetaAttributes: %s", xml)
    logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, config, metadata, dmtx=None):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        data = remove_os(data)

        # fov shift - not working atm as fov shifts in sequence are not deactivated
        # rotmat = calc_rotmat(group[0])
        # res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
        # shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
        # data = fov_shift(data, shift)

        data = np.swapaxes(data,0,1) # in Pulseq gre_refscan sequence read and phase are changed, might change this in the sequence
        sensmaps = bart(1, 'ecalib -m 1 -k 8 -I', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

def sens_from_raw(group, metadata, dmtx):
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    data, trj = sort_spiral_data(group, metadata, dmtx)

    sensmaps = bart(1, 'nufft -i -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz), trj, data) # nufft
    sensmaps = cfftn(sensmaps, [0, 1, 2]) # back to k-space
    sensmaps = bart(1, 'ecalib -m 1 -I', sensmaps)  # ESPIRiT calibration
    return sensmaps
    
# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group, metadata):

    sig = list()
    trj = list()
    for acq in group:

        # signal - already fov shifted in insert_prot_ismrmrd
        sig.append(acq.data)

        # trajectory
        traj = np.swapaxes(acq.traj,0,1)[:3] # [dims, samples]
        traj = traj[[1,0,2],:]  # switch x and y dir for correct orientation
        trj.append(traj)
  
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)

    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis]
    
    return sig, trj

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
    
#%%
#########################
# Helper functions
#########################

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

def calc_rotmat(acq):
        phase_dir = np.asarray(acq.phase_dir)
        read_dir = np.asarray(acq.read_dir)
        slice_dir = np.asarray(acq.slice_dir)
        return np.round(np.concatenate([phase_dir[:,np.newaxis], read_dir[:,np.newaxis], slice_dir[:,np.newaxis]], axis=1), 6)

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

# the fov-shift from the Pulseq sequence is not correct 
# tried to reapply it with the predicted trajectory, but also doesnt work properly
def fov_shift_spiral_reapply(acq, base_trj, matr_sz, res):
    """ 
    shift field of view of spiral data
    first undo field of view shift with nominal, then reapply with predicted trajectory

    acq: ISMRMRD acquisition
    base_trj: nominal trajectory
    matr_sz: matrix size
    res: resolution 
    """
    rotmat = calc_rotmat(acq)
    shift = pcs_to_gcs(np.asarray(acq.position), rotmat) / res
    sig = acq.data[:]
    pred_trj = acq.traj[:] # [samples, dims]

    if (abs(shift[0]) < 1e-2) and (abs(shift[1]) < 1e-2):
        # nothing to do
        return sig

    kmax = int(matr_sz/2+0.5)

    # undo FOV shift from nominal traj
    sig *= np.exp(-1j*(shift[0]*np.pi*base_trj[:,0]/kmax-shift[1]*np.pi*base_trj[:,1]/kmax))

    # redo FOV shift with predicted traj
    sig *= np.exp(1j*(shift[0]*np.pi*pred_trj[:,0]/kmax-shift[1]*np.pi*pred_trj[:,1]/kmax))

    return sig
