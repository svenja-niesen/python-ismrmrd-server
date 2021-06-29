import ismrmrd
import os
import itertools
import logging
import numpy as np
import base64

from bart import bart
from cfft import cfftn, cifftn
from pulseq_prot import insert_hdr, insert_acq, get_ismrmrd_arrays
from reco_helper import calculate_prewhitening, apply_prewhitening, calc_rotmat, pcs_to_gcs, fov_shift_spiral, fov_shift, remove_os, intp_axis
from DreamMap import DREAM_filter_fid, calc_fa

from skimage import filters
import scipy

import numpy.fft as fft

""" Reconstruction of imaging data acquired with the Pulseq Sequence via the FIRE framework
    Reconstruction is done with the BART toolbox
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

def process_spiral(connection, config, metadata):
  
    slc_sel = None

    # Create folder, if necessary
    if not os.path.exists(debugFolder):
        os.makedirs(debugFolder)
        logging.debug("Created folder " + debugFolder + " for debug output files")

    protFolder = os.path.join(dependencyFolder, "pulseq_protocols")
    protFolder_local = "/tmp/local/pulseq_protocols" # Protocols mountpoint (not at the scanner)
    prot_filename = metadata.userParameters.userParameterString[0].value_ # protocol filename from Siemens protocol parameter tFree

    # Check if local protocol folder is available - if not use protFolder (scanner)
    date = prot_filename.split('_')[0] # folder in Protocols (=date of seqfile)
    protFolder_loc = os.path.join(protFolder_local, date)
    if os.path.exists(protFolder_loc):
        protFolder = protFolder_loc

    # Insert protocol header
    prot_file = protFolder + "/" + prot_filename
    insert_hdr(prot_file, metadata)
    
    logging.info("Config: \n%s", config)

    # Check for GPU availability
    if os.environ.get('NVIDIA_VISIBLE_DEVICES') == 'all':
        gpu = True
    else:
        gpu = False

    # Metadata should be MRD formatted header, but may be a string
    # if it failed conversion earlier

    try:
        # logging.info("Metadata: \n%s", metadata.toxml('utf-8'))
        # logging.info("Metadata: \n%s", metadata.serialize())

        logging.info("Incoming dataset contains %d encodings", len(metadata.encoding))
        logging.info("Trajectory type '%s', matrix size (%s x %s x %s), field of view (%s x %s x %s)mm^3", 
            metadata.encoding[0].trajectory, 
            metadata.encoding[0].encodedSpace.matrixSize.x, 
            metadata.encoding[0].encodedSpace.matrixSize.y, 
            metadata.encoding[0].encodedSpace.matrixSize.z, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.x, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.y, 
            metadata.encoding[0].encodedSpace.fieldOfView_mm.z)

    except:
        logging.info("Improperly formatted metadata: \n%s", metadata)
    
    # # Initialize lists for datasets
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1

    acqGroup = [[[] for _ in range(n_slc)] for _ in range(n_contr)]
    noiseGroup = []
    waveformGroup = []

    acsGroup = [[] for _ in range(n_slc)]
    #acsGroup = [[] for _ in range(metadata.encoding[0].encodedSpace.matrixSize.z)] # for mult_refscan
    sensmaps = [None] * n_slc
    #sensmaps = [None] * metadata.encoding[0].encodedSpace.matrixSize.z # for mult_refscan
    #sensmaps3 = np.zeros([metadata.encoding[0].encodedSpace.matrixSize.x,metadata.encoding[0].encodedSpace.matrixSize.y,metadata.encoding[0].encodedSpace.matrixSize.z,32],dtype=complex) # for mult_refscan
    #h = None # for mult_refscan
    old_grid = []
    dmtx = None
    base_trj_ = []
    refscans = [None] * n_slc
    #refscans = [None] * metadata.encoding[0].encodedSpace.matrixSize.z # for mult_refscan
    #refscans3 = np.zeros([metadata.encoding[0].encodedSpace.matrixSize.x,metadata.encoding[0].encodedSpace.matrixSize.y,metadata.encoding[0].encodedSpace.matrixSize.z]) # for mult_refscan
    
    # read protocol arrays
    prot_arrays = get_ismrmrd_arrays(prot_file)

    # for B1 Dream map
    if "dream" in prot_arrays:
        process_raw.imagesets = [None] * n_contr
        process_raw.fid_unfiltered = True
    
    # different contrasts need different scaling
    process_raw.imascale = [None] * n_contr
    
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
                    noiseGroup.clear()
                    
                # skip slices in single slice reconstruction
                if slc_sel is not None and item.idx.slice != slc_sel:
                    continue
                
                # Accumulate all imaging readouts in a group
                if item.is_flag_set(ismrmrd.ACQ_IS_PHASECORR_DATA):
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_DUMMYSCAN_DATA): # skope sync scans
                    continue
                elif item.is_flag_set(ismrmrd.ACQ_IS_PARALLEL_CALIBRATION):
                    acsGroup[item.idx.slice].append(item)
                    if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE):
                        logging.debug("item.idx.slice=%s"%(item.idx.slice,))
                        sensmaps[item.idx.slice],refscans[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx, gpu)
                        logging.debug("sensmaps.shape = %s"%(sensmaps[item.idx.slice].shape,))
                        old_grid.append(item.idx.slice)
                        acsGroup[item.idx.slice].clear()
                    continue
                
                # if h is None:                           # for mult_refscan
                #     for slc in range(len(sensmaps)):
                #         logging.debug("slc=%s"%(slc,))
                #         sensmaps3[:,:,slc,:] = sensmaps[slc][:,:,0,:].copy()
                #         refscans3[:,:,slc] = refscans[slc][:,:,0].copy()
                #     h = 1
                #     sensmaps3=np.flip(sensmaps3,axis=2)
                #     refscans3=np.flip(refscans3,axis=2)
                #     np.save(debugFolder + "/" + "sensmaps3.npy", sensmaps3)
                #     np.save(debugFolder + "/" + "refscans3.npy", refscans3)
                #     sensmaps = []
                #     sensmaps.append(sensmaps3)
                
                if ([elem for elem in sensmaps if type(elem) is not np.ndarray].count(None)!=len(sensmaps)) and ([elem for elem in sensmaps if type(elem) is not np.ndarray].count(None)!=0): # wenn sensmaps elemente enthält, aber auch None, dann interpoliere. Geht nur beim ersten imaging acquisition rein, da danach die sensmaps gefüllt sein sollten
                    logging.info("Interpolation of sensitivity maps")
                    
                    if (n_slc % 2 == 0):
                        logging.debug("begin with odd slices")
                        for i in range(0,n_slc-1,2):
                            sensmaps[i] = sensmaps[i+1].copy()
                    else:
                        for i in range(1,n_slc,2):
                            sensmaps[i] = sensmaps[i-1].copy()
                    np.save(debugFolder + "/" + "sensmaps_copy.npy", sensmaps)
                    
                    # data = np.asarray([elem for elem in sensmaps if type(elem) is np.ndarray]) # sensmaps ohne die None elemente und als Array -> ist data Parameter
                    # np.save(debugFolder + "/" + "sensmaps_old.npy", data)
                    # new_grid = np.arange(0,n_slc)
                    # sensmaps = intp_axis(newgrid=new_grid, oldgrid=np.asarray(old_grid), data=data, axis=0) # Ergebnis als numpy array
                    # np.save(debugFolder + "/" + "sensmaps_new.npy", sensmaps)
                    # sensmaps = list(sensmaps) # umwandlung zurück in liste sensmaps[slices] für if bedingung
                    # old_grid.clear()
                    
                    
                    
                if item.idx.segment == 0:
                    acqGroup[item.idx.contrast][item.idx.slice].append(item)
                else:
                    # append data to first segment of ADC group
                    idx_lower = item.idx.segment * item.number_of_samples
                    idx_upper = (item.idx.segment+1) * item.number_of_samples
                    acqGroup[item.idx.contrast][item.idx.slice][-1].data[:,idx_lower:idx_upper] = item.data[:]

                # When this criteria is met, run process_raw() on the accumulated
                # data, which returns images that are sent back to the client.
                if item.is_flag_set(ismrmrd.ACQ_LAST_IN_SLICE) or item.is_flag_set(ismrmrd.ACQ_LAST_IN_REPETITION):
                    logging.info("Processing a group of k-space data")
                    images = process_raw(acqGroup[item.idx.contrast][item.idx.slice], config, metadata, dmtx, sensmaps[item.idx.slice], gpu, prot_arrays)
                    logging.debug("Sending images to client:\n%s", images)
                    connection.send_image(images)
                    acqGroup[item.idx.contrast][item.idx.slice].clear() # free memory

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
            if len(acqGroup[item.idx.contrast][item.idx.slice]) > 0:
                logging.info("Processing a group of k-space data (untriggered)")
                if sensmaps[item.idx.slice] is None:
                    # run parallel imaging calibration
                    #acsGroup[item.idx.slice].clear() ???
                    sensmaps[item.idx.slice] = process_acs(acsGroup[item.idx.slice], config, metadata, dmtx) 
                image = process_raw(acqGroup[item.idx.contrast][item.idx.slice], config, metadata, dmtx, sensmaps[item.idx.slice])
                logging.debug("Sending image to client:\n%s", image)
                connection.send_image(image)
                acqGroup = []

    finally:
        connection.send_close()

# %%
#########################
# Process Data
#########################

def process_raw(group, config, metadata, dmtx=None, sensmaps=None, gpu=False, prot_arrays=None):

    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    
    rNx = metadata.encoding[0].reconSpace.matrixSize.x
    rNy = metadata.encoding[0].reconSpace.matrixSize.y
    rNz = metadata.encoding[0].reconSpace.matrixSize.z

    data, trj, data_kz, trj_kz = sort_spiral_data(group, metadata, dmtx)

    #logging.debug("Raw data is size %s" % (data.shape,))
    #logging.debug("nx,ny,nz %s, %s, %s" % (nx, ny, nz))
    if group[0].idx.contrast == 0:
        np.save(debugFolder + "/" + "raw_ste.npy", data)
        np.save(debugFolder + "/" + "trj_ste.npy", trj)
    else:
        np.save(debugFolder + "/" + "raw_fid.npy", data)
        np.save(debugFolder + "/" + "trj_fid.npy", trj)
        
    np.save(debugFolder + "/" + "raw_kz.npy", data_kz)
    
    np.save(debugFolder + "/" + "sensmaps3copy.npy", sensmaps)
    
    if 'dream' in prot_arrays :
        if prot_arrays['dream'].size > 2 :
            logging.info("fid_filt avtivated")
            data_fid = data.copy() # for b1 filter
    
    # if sensmaps is None: # assume that this is a fully sampled scan (wip: only use autocalibration region in center k-space)
        # sensmaps = bart(1, 'ecalib -m 1 -I ', data)  # ESPIRiT calibration
    #sensmaps = None
    if gpu:
        nufft_config = 'nufft -g -i -l 0.005 -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -g -m 1 -I'
        pics_config = 'pics -g -S -e -l1 -r 0.001 -i 50 -t'
    else:
        nufft_config = 'nufft -i -l 0.005 -c -t -d %d:%d:%d'%(nx, nx, nz)
        ecalib_config = 'ecalib -m 1 -I -r 20 -k 6'
        pics_config = 'pics -S -e -l1 -r 0.001 -i 100 -t'

    force_pics = False
    if sensmaps is None and force_pics:
        sensmaps = bart(1, nufft_config, trj, data) # nufft
        sensmaps = cfftn(sensmaps, [0, 1, 2]) # back to k-space
        sensmaps = bart(1, ecalib_config, sensmaps)  # ESPIRiT calibration

    if sensmaps is None:
        logging.debug("no pics necessary, just do standard recon")
            
        # bart nufft
        data = bart(1, nufft_config, trj, data) # nufft
        #data,os = bart(2, nufft_config, trj, data)
        
        # Sum of squares coil combination
        data = np.sqrt(np.sum(np.abs(data)**2, axis=-1))
    else:
        logging.debug("data.shape = %s" % (data.shape,))
        logging.debug("sensmaps.shape = %s" % (sensmaps.shape,))
        data = bart(1, pics_config , trj, data, sensmaps)
        data = np.abs(data)
        # make sure that data is at least 3d:
        while np.ndim(data) < 3:
            data = data[..., np.newaxis]
    
    # if group[0].idx.slice == 0 and sensmaps is not None:
    #     np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)

    if nz > rNz:
        # remove oversampling in slice direction
        data = data[:,:,(nz - rNz)//2:-(nz - rNz)//2]

    logging.debug("Image data is size %s" % (data.shape,))
    if group[0].idx.slice == 0:
        if group[0].idx.contrast == 0:
            np.save(debugFolder + "/" + "img_ste.npy", data)
        if group[0].idx.contrast == 1:
            np.save(debugFolder + "/" + "img_fid.npy", data)
    
    # B1 Map calculation (Dream approach)
    if 'dream' in prot_arrays: #dream = ([ste_contr,flip_angle_ste,TR,flip_angle,prepscans,t1])
        dream = prot_arrays['dream']
        n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
        
        process_raw.imagesets[group[0].idx.contrast] = data.copy()
        full_set_check = all(elem is not None for elem in process_raw.imagesets)
        if full_set_check:
            logging.info("B1 map calculation using Dream")
            ste = np.asarray(process_raw.imagesets[int(dream[0])])
            fid = np.asarray(process_raw.imagesets[int(n_contr-1-dream[0])])
            
            # image processing filter for fa-map
            dil = np.zeros(fid.shape)
            for nz in range(fid.shape[-1]):
                # otsu
                val = filters.threshold_otsu(fid[:,:,nz])
                otsu = fid[:,:,nz] > val
                # fill holes
                imfill = scipy.ndimage.morphology.binary_fill_holes(otsu) * 1
                # dilation
                dil[:,:,nz] = scipy.ndimage.morphology.binary_dilation(imfill)
            np.save(debugFolder + "/" + "dil.npy", dil)
            
            if dream.size > 2 :                           # without filter: dream = ([ste_contr,flip_angle_ste])
                logging.info("Global filter approach")
                
                # save unfiltered fid if wanted
                if process_raw.fid_unfiltered:
                    logging.debug("fid_unfiltered to scanner")
                    fid_unfilt = fid.copy()
                    np.save(debugFolder + "/" + "fid_unf.npy", fid_unfilt)
                else:
                    fid_unfilt = None
                
                # Blurring compensation parameters
                alpha = dream[1]     # preparation FA
                tr = dream[2]        # [s]
                beta = dream[3]      # readout FA
                dummies = dream[4]   # number of dummy scans before readout echo train starts
                # T1 estimate:
                t1 = dream[5]        # [s] - approximately Gufi Phantom at 7T
                # TI estimate (the time after DREAM preparation after which each k-space line is acquired):
                ti = 0
                mean_alpha = calc_fa(ste.mean(), fid.mean())
                mean_beta = mean_alpha / alpha * beta
                for i,acq in enumerate(group):
                    ti = tr * (dummies + i) # [s]
                    print(ti)
                    # Global filter:
                    filt = DREAM_filter_fid(mean_alpha, mean_beta, tr, t1, ti)
                    # apply filter:
                    data_fid[:,:,i,:] *= filt
                # reco of fid:
                if sensmaps is None:
                    # bart nufft
                    fid = bart(1, nufft_config, trj, data_fid) # nufft
                    # Sum of squares coil combination
                    fid = np.sqrt(np.sum(np.abs(fid)**2, axis=-1))
                else:
                    fid = bart(1, pics_config , trj, data_fid, sensmaps)
                    fid = np.abs(fid)
                    # make sure that data is at least 3d:
                    while np.ndim(fid) < 3:
                        fid = fid[..., np.newaxis]         
                if nz > rNz:
                    # remove oversampling in slice direction
                    fid = fid[:,:,(nz - rNz)//2:-(nz - rNz)//2]
                # fa map:
                fa_map = calc_fa(abs(ste), abs(fid))
                # fid = fid_filt:
                np.save(debugFolder + "/" + "fid_filt.npy", fid)
                data = fid.copy()
            else:
                fa_map = calc_fa(abs(ste), abs(fid))
                fid_unfilt = None
            
            np.save(debugFolder + "/" + "fa_unf.npy", fa_map)
            fa_map *= dil
            current_refvolt = metadata.userParameters.userParameterDouble[5].value_
            nom_fa = dream[1]
            logging.info("current_refvolt = %sV und nom_fa = %s°^" % (current_refvolt, nom_fa))
            ref_volt = current_refvolt * (nom_fa/fa_map)
            
            np.save(debugFolder + "/" + "fa.npy", fa_map)
            fa_map = np.around(fa_map)
            fa_map = fa_map.astype(np.int16)
            logging.debug("fa map is size %s" % (fa_map.shape,))
            
            ref_volt = np.around(ref_volt)
            ref_volt = ref_volt.astype(np.int16)
            np.save(debugFolder + "/" + "ref_volt.npy", ref_volt)
            logging.debug("ref_volt map is size %s" % (ref_volt.shape,))
            
            process_raw.imagesets = [None] * n_contr # free list
        else:
            fa_map = None
            fid_unfilt = None
            ref_volt = None
    else:
        fa_map = None
        fid_unfilt = None
        ref_volt = None
        logging.info("no dream B1 mapping")
    
    # Normalize and convert to int16
    #save one scaling in 'static' variable
    
    contr = group[0].idx.contrast
    if process_raw.imascale[contr] is None:
        process_raw.imascale[contr] = 0.8 / data.max()
    data *= 32767 * process_raw.imascale[contr]
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
    n_slc = metadata.encoding[0].encodingLimits.slice.maximum + 1
    n_contr = metadata.encoding[0].encodingLimits.contrast.maximum + 1
    logging.debug("n_slc= %s" % (n_slc))
    
    # Format as ISMRMRD image data - WIP: something goes wrong here with indexes
    if n_par > 1:
        for par in range(n_par):
            image = ismrmrd.Image.from_array(data[...,par], acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_par + par # contains image index (slices/partitions)
            image.image_series_index = 1 + group[0].idx.repetition # contains image series index, e.g. different contrasts
            image.slice = 0
            image.attribute_string = xml
            images.append(image)
        
        if fa_map is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(fa_map[...,par].T, acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 2 + group[0].idx.repetition
                image.slice = 0
                image.attribute_string = xml
                images.append(image)
        
        if fid_unfilt is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(fa_map[...,par].T, acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 3 + group[0].idx.repetition
                image.slice = 0
                image.attribute_string = xml
                images.append(image)
        
        if ref_volt is not None:
            for par in range(n_par):
                image = ismrmrd.Image.from_array(fa_map[...,par].T, acquisition=group[0])
                image.image_index = 1 + par
                image.image_series_index = 4 + group[0].idx.repetition
                image.slice = 0
                image.attribute_string = xml
                images.append(image)
        
    else:
        image = ismrmrd.Image.from_array(data[...,0], acquisition=group[0])
        image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice # contains image index (slices/partitions)
        logging.debug("image.image_index= %s" % (image.image_index))
        image.image_series_index = 1 + group[0].idx.repetition # contains image series index, e.g. different contrasts
        image.slice = 0
        image.attribute_string = xml
        images.append(image)
        
        if fa_map is not None:
            image = ismrmrd.Image.from_array(fa_map[...,0].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 2 + group[0].idx.repetition
            image.slice = 0
            image.attribute_string = xml
            images.append(image)
        
        if fid_unfilt is not None:
            image = ismrmrd.Image.from_array(fa_map[...,0].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 3 + group[0].idx.repetition
            image.slice = 0
            image.attribute_string = xml
            images.append(image)
        
        if ref_volt is not None:
            image = ismrmrd.Image.from_array(fa_map[...,0].T, acquisition=group[0])
            image.image_index = 1 + group[0].idx.contrast * n_slc + group[0].idx.slice
            image.image_series_index = 4 + group[0].idx.repetition
            image.slice = 0
            image.attribute_string = xml
            images.append(image)

    # logging.debug("Image MetaAttributes: %s", xml)
    # logging.debug("Image data has size %d and %d slices"%(images[0].data.size, len(images)))

    return images

def process_acs(group, config, metadata, dmtx=None, gpu=False):
    if len(group)>0:
        data = sort_into_kspace(group, metadata, dmtx, zf_around_center=True)
        logging.debug("data.shape=%s"% (data.shape,))
        data = remove_os(data)

        # fov shift
        rotmat = calc_rotmat(group[0])
        if not rotmat.any(): rotmat = -1*np.eye(3) # compatibility if refscan rotmat is not in protocol, this is the standard Pulseq rotation matrix
        res = metadata.encoding[0].encodedSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
        shift = pcs_to_gcs(np.asarray(group[0].position), rotmat) / res
        data = fov_shift(data, shift)

        data = np.swapaxes(data,0,1) # in Pulseq gre_refscan sequence read and phase are changed, might change this in the sequence
        np.save(debugFolder + "/" + "data_sensm.npy", data)
        logging.debug("data.shape=%s"% (data.shape,))
        logging.debug("Referencescan: shape= %s" % (data.shape,))
        ref = fft.fftshift(data, axes=(0, 1, 2))
        ref = fft.ifftn(ref, axes=(0, 1, 2))
        ref = fft.ifftshift(ref, axes=(0, 1, 2))
        # Sum of squares coil combination
        ref = np.sqrt(np.sum(np.abs(ref)**2, axis=-1))
        logging.debug("Referencescan: shape= %s" % (ref.shape,))
        np.save(debugFolder + "/" + "ref.npy", ref)
        
        if gpu:
            sensmaps = bart(1, 'ecalib -g -m 1 -k 4 -I', data)  # ESPIRiT calibration
        else:
            sensmaps = bart(1, 'ecalib -m 1 -k 4 -I', data)  # ESPIRiT calibration
        np.save(debugFolder + "/" + "acs.npy", data)
        np.save(debugFolder + "/" + "sensmaps.npy", sensmaps)
        return sensmaps,ref
    else:
        return None

# %%
#########################
# Sort Data
#########################

def sort_spiral_data(group, metadata, dmtx=None):
    
    nx = metadata.encoding[0].encodedSpace.matrixSize.x
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    res = metadata.encoding[0].reconSpace.fieldOfView_mm.x / metadata.encoding[0].encodedSpace.matrixSize.x
    rot_mat = calc_rotmat(group[0])

    sig = list()
    trj = list()
    enc = list()
    
    # for kz
    trj_kz = [[] for _ in range(nz)]
    sig_kz = [[] for _ in range(nz)]
    
    for acq in group:

        enc1 = acq.idx.kspace_encode_step_1
        enc2 = acq.idx.kspace_encode_step_2
        kz = enc2 - nz//2
        enc.append([enc1, enc2])
        
        # append data after optional prewhitening
        if dmtx is None:
            sig.append(acq.data)
            # for kz
            sig_kz[enc2].append(acq.data)
        else:
            sig.append(apply_prewhitening(acq.data, dmtx))
            # for kz
            sig_kz[enc2].append(apply_prewhitening(acq.data, dmtx))

        # update trajectory - for a BART reco we dont need the time dimension
        traj = np.swapaxes(acq.traj[:,:3],0,1) # [dims, samples]
        traj = traj[[1,0,2],:]  # switch x and y dir for correct orientation in FIRE
        trj.append(traj)
        
        # for kz
        trj_kz[enc2].append(traj)
        
        # fov shift
        shift = pcs_to_gcs(np.asarray(acq.position), rot_mat) / res
        sig[-1] = fov_shift_spiral(sig[-1], trj[-1], shift, nx)
        
    np.save(debugFolder + "/" + "enc.npy", enc)
    
    # convert lists to numpy arrays
    trj = np.asarray(trj) # current size: (nacq, 3, ncol)
    sig = np.asarray(sig) # current size: (nacq, ncha, ncol)
    
    # for kz
    trj_kz = np.asarray(trj_kz) # current size: (nz, nacq, 3, ncol)
    sig_kz = np.asarray(sig_kz) # current size: (nz, nacq, ncha, ncol)
    
    # rearrange trj & sig for bart
    trj = np.transpose(trj, [1, 2, 0]) # [3, ncol, nacq]
    sig = np.transpose(sig, [2, 0, 1])[np.newaxis] # [ncol, nacq, ncha]
    logging.debug("Trajectory shape = %s , Signal Shape = %s "%(trj.shape, sig.shape))
    
    # for kz
    # trj_kz = np.transpose(trj_kz, [0, 2, 3, 1]) # [nz, 3, ncol, nacq]
    # sig_kz = np.transpose(sig_kz, [0, 3, 1, 2])[np.newaxis] # [nz, ncol, nacq, ncha]
    # logging.debug("Trajectory kz shape = %s , Signal kz Shape = %s "%(trj_kz.shape, sig_kz.shape))
    
    np.save(debugFolder + "/" + "trj.npy", trj)
    
    # for kz
    np.save(debugFolder + "/" + "trj_kz.npy", trj_kz)

    return sig, trj, sig_kz, trj_kz

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
    #ny = metadata.encoding[0].encodedSpace.matrixSize.y
    nz = metadata.encoding[0].encodedSpace.matrixSize.z
    #nz = 1 # for mult_refscan

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