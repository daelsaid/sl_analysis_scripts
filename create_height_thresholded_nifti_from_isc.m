function [new] = create_height_thresholded_nifti_from_isc(outputdir,statimg,outputfile_prefix,threshold,brain_mask)
    
    brainmask=brain_mask
    output_dir=outputdir
    stat_img =statimg
    out_fname=outputfile_prefix
    thresholdValue = threshold    %set threshold to use 

    %load image
    V = spm_vol(stat_img);
    %image is 4D, pick first volume
    V=V(1)
    
    %read volume
    unthresh_img = spm_read_vols(V);
    %pick 1/5
    unthresh_tmap = unthresh_img(:,:,:,1);

    %load MNI  152 2mm brain mask 
    V_mask = spm_vol(brainmask);
    Y_mask = spm_read_vols(V_mask);

    % Ensure mask and stat image are the same size
    if ~isequal(V.dim, V_mask.dim)
        error('Mask and image dimensions do not match. Reslice the mask.');
    end

    % Apply the brain mask (set non-brain voxels to zero)
    unthresh_tmap(Y_mask == 0) = 0;

    % Create a binary mask based on the threshold
    binaryMask = unthresh_tmap > thresholdValue;
    
    % Optionally, create an output image that retains the original values
    % for voxels above the threshold and sets others to zero
    thresholdedImage = zeros(size(unthresh_tmap)); % Initialize output image that maintains structure
    thresholdedImage(binaryMask) = unthresh_tmap(binaryMask); % Retain values above threshold
    
    % Save corrected NIfTI image
    [filepath,name,ext] = fileparts(fullfile(output_dir,[out_fname,V.fname]));
    corrected_fname =  fullfile(output_dir,['thresh_' name,ext])
    
    %rename new nifti 
    V.fname = corrected_fname;
    V.mat = spm_vol(brainmask).mat;  % Copy original MNI transformation matrix
    V.dim = spm_vol(brainmask).dim;  % Copy original MNI template voxel dimensions

    new=spm_write_vol(V, thresholdedImage);

    save(fullfile(output_dir, ['thresh_' name '.mat']), 'new', 'thresholdedImage');
    disp(['Saved thresholded NIfTI: ', corrected_fname]);

end