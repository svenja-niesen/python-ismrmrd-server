
import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64

from bart import bart
from PowerGridPy import PowerGridIsmrmrd
from cfft import cfftn, cifftn

from pulseq_prot import insert_hdr, insert_acq
from reco_helper import calculate_prewhitening, apply_prewhitening, calc_rotmat, pcs_to_gcs, fov_shift_spiral, fov_shift, remove_os

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox and the PowerGrid toolbox

Short Comments on FOV Shifting and rotations:
    - Translational shifts should not be acticvated in the GUI, when using the Pulseq Sequence.
      However in-plane FOV shifts can be selected also without activating FOV positioning and are applied in this reco.

    - Rotations are not yet possible as only the standard rotation matrix for Pulseq is considered and taken from the protocol dile.
      The standard rotation matrix was obtained by simulating predefined gradients.
      To implement rotations selected in the GUI, the correct rotation matrix has to be obtained from the dataset ISMRMRD file coming from the scanner.
      However, that rotation matrix seems to be incorrect, as it switches phase and read gradients. It seems like an additional transformation would have to be applied.

"""

# Folder for sharing data/debugging
shareFolder = "/tmp/share"
debugFolder = os.path.join(shareFolder, "debug")
dependencyFolder = os.path.join(shareFolder, "dependency")

########################
# Main Function
########################

def process(connection, config, metadata):
    
    # Select a slice (only for debugging purposes) - if "None" reconstruct all slices
    slc_sel = None

    # Set this True, if a Skope trajectory is used (protocol file with skope trajectory has to be available)
    skope = False

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    protFolder = os.path.join(dependencyFolder, "pulseq_protocols")
    protFolder_local = "/tmp/local/pulseq_protocols" # Protocols mountpoint (not at the scanner)
    prot_filename = metadata.userParameters.userParameterString[0].value_ # protocol filename from Siemens protocol parameter tFree
    if skope:
        prot_filename += "_skopetraj"

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
                base_trj = insert_acq(prot_file, item, acq_ctr, skope=skope)
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
                    if (item.idx.slice != slc_sel and slc_sel is not None):
                        sensmaps[item.idx.slice] = 0 # calculate sensmap only for selected slice
                        continue
                    acsGroup[item.idx.slice].append(item)
                    continue
                elif sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration (after last calibration scan is acquired/before first imaging scan)
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx) # [nx,ny,nz,nc]

                # Deal with ADC segments
                if item.idx.segment == 0:
                    acqGroup[item.idx.slice].append(item)
                else:
                    # append data to first segment of ADC group
                    idx_lower = item.idx.segment * item.number_of_samples
                    idx_upper = (item.idx.segment+1) * item.number_of_samples
                    acqGroup[item.idx.slice][-1].data[:,idx_lower:idx_upper] = item.data[:]

                if item.idx.segment == nsegments - 1:
                    # Inplane FOV shift and noise whitening
                    rotmat = calc_rotmat(item)
                    shift = pcs_to_gcs(np.asarray(item.position), rotmat) / res
                    if dmtx is None:
                        data = acqGroup[item.idx.slice][-1].data[:]
                    else:
                        data = apply_prewhitening(acqGroup[item.idx.slice][-1].data[:], dmtx)
                    traj = np.swapaxes(acqGroup[item.idx.slice][-1].traj[:,:3],0,1)
                    traj = traj[[1,0,2],:]  # switch x and y dir for correct orientation
                    acqGroup[item.idx.slice][-1].data[:] = fov_shift_spiral(data, traj, shift, matr_sz)

                # if no refscan, calculate sensitivity maps from raw data
                if sensmaps[item.idx.slice] is None:
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                        sensmaps[item.idx.slice] = sens_from_raw(acqGroup[item.idx.slice], metadata)

                # When all acquisitions are processed, write them to file for PowerGrid Reco,
                # which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_MEASUREMENT):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup, metadata, sensmaps, slc_sel)
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

def process_raw(acqGroup, metadata, sensmaps, slc_sel=None):

    # average acquisitions before reco
    avg_before = True 
    if metadata.encoding[0].encodingLimits.contrast.maximum > 0:
        avg_before = False # do not average before in diffusion imaging as this would introduce phase errors

    # Write ISMRMRD file for PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    if os.path.exists(tmp_file):
        os.remove(tmp_file)
    dset_tmp = ismrmrd.Dataset(tmp_file, create_if_needed=True)

    # Write header
    if slc_sel is not None:
        metadata.encoding[0].encodingLimits.slice.maximum = 0
    if avg_before:
        n_avg = metadata.encoding[0].encodingLimits.average.maximum + 1
        metadata.encoding[0].encodingLimits.average.maximum = 0
    dset_tmp.write_xml_header(metadata.toxml())

    # Insert Field Map
    fmap_path = dependencyFolder+"/fmap.npz"
    if not os.path.exists(fmap_path):
        raise ValueError("No field map file in dependency folder. Field map should be .npz file containing the field map and field map regularisation parameters")
    fmap = np.load(fmap_path, allow_pickle=True)
    fmap_data = fmap['fmap']
    if slc_sel is not None:
        fmap_data = fmap_data[slc_sel]

    print("Field Map name:", fmap['name'].item())
    print("Field Map regularisation parameters:",  fmap['params'].item())
    dset_tmp.append_array('FieldMap', fmap_data) # dimensions in PowerGrid seem to be [slices/nz,ny,nx]

    # Insert Sensitivity Maps
    if slc_sel is not None:
        sens = np.transpose(sensmaps[slc_sel], [3,2,1,0])
    else:
        sens = np.transpose(np.stack(sensmaps), [0,4,3,2,1]) # [slices,nc,nz,ny,nx] - only tested for 2D, nx/ny might be changed depending on orientation
    dset_tmp.append_array("SENSEMap", sens.astype(np.complex128))

    # Average acquisition data before reco
    # Assume that averages are acquired in the same order for every slice, contrast, ...
    if avg_before:
        avgData = [[] for _ in range(n_avg)]
        for k, slc in enumerate(acqGroup):
            for j, acq in enumerate(slc):
                avgData[acq.idx.average].append(acq.data[:])
        avgData = np.mean(avgData, axis=0)

    # Insert acquisitions
    avg_ix = 0
    for k, slc in enumerate(acqGroup):
        for j, acq in enumerate(slc):
            if avg_before:
                if acq.idx.average == 0:
                    acq.data[:] = avgData[avg_ix]
                    avg_ix += 1
                else:
                  continue
            if slc_sel is not None:
                if acq.idx.slice != slc_sel:
                    continue
                else:
                    acq.idx.slice = 0
            dset_tmp.append_acquisition(acq)
    dset_tmp.close()

    # Process with PowerGrid
    tmp_file = dependencyFolder+"/PowerGrid_tmpfile.h5"
    debug_pg = debugFolder+"/powergrid_tmp"
    if not os.path.exists(debug_pg):
        os.makedirs(debug_pg)

    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        # histo is buggy for GPU, so use minmax here as temporal interpolator
        data = PowerGridIsmrmrd(inFile=tmp_file, outFile=debug_pg+"/img", timesegs=5, niter=10, beta=0, TSInterp='minmax')
    else:
        data = PowerGridIsmrmrd(inFile=tmp_file, outFile=debug_pg+"/img", timesegs=5, niter=10, beta=0, TSInterp='histo')
    shapes = data["shapes"] 
    data = np.asarray(data["img_data"]).reshape(shapes)
    data = np.abs(data)

    # data should have output [Slice, Phase, Echo/Contrast, Avg, Rep, Nz, Ny, Nx]
    # change to [Avg, Rep, Echo/Contrast, Phase, Slice, Nz, Ny, Nx] and average
    data = np.transpose(data, [3,4,2,1,0,5,6,7]).mean(axis=0)

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
    dsets = []

    # If we have a diffusion dataset, b-value and direction contrasts are stored in contrast index
    # as otherwise we run into problems with the PowerGrid acquisition tracking.
    # We now (in case of diffusion imaging) split the b=0 image from other images and reshape to b-values and directions
    n_bval = metadata.encoding[0].encodingLimits.contrast.center # number of b-values (incl b=0)
    n_dirs = metadata.encoding[0].encodingLimits.phase.center # number of directions
    if n_bval > 0:
        shp = data.shape
        dsets.append(np.expand_dims(data[:,0], 1))
        dsets.append(data[:,1:].reshape(shp[0], n_bval-1, n_dirs, shp[3], shp[4], shp[5], shp[6]))
    else:
        dsets.append(data)

    slc_img_ix = 0
    for data in dsets:
        # Format as ISMRMRD image data
        for rep in range(data.shape[0]):
            for echo in range(data.shape[1]):
                slc_img_ix += 1
                img_ix = 0
                for phs in range(data.shape[2]):
                    for slc in range(data.shape[3]):
                        for nz in range(data.shape[4]):
                            img_ix += 1
                            image = ismrmrd.Image.from_array(data[rep,echo,phs,slc,nz], acquisition=acqGroup[slc][0])
                            image.image_index = img_ix # contains slices/partitions and phases
                            image.image_series_index = slc_img_ix # contains repetitions, contrasts
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

        # fov shift
        rotmat = calc_rotmat(group[0])
        if not rotmat.any(): rotmat = -1*np.eye(3) # compatibility if refscan has no rotmat in protocol
        res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
        shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
        data = fov_shift(data, shift)

        data = np.swapaxes(data,0,1) # in Pulseq gre_refscan sequence read and phase are changed, might change this in the sequence
        if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
            print("Run Espirit on GPU.")
            sensmaps = bart(1, 'ecalib -g -m 1 -k 8 -I', data)  # ESPIRiT calibration
        else:
            print("Run Espirit on CPU.")
            sensmaps = bart(1, 'ecalib -m 1 -k 8 -I', data)  # ESPIRiT calibration

        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps
    else:
        return None

def sens_from_raw(group, metadata):
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    data, trj = sort_spiral_data(group, metadata)

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