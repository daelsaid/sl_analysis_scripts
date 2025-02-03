% clc; clear; close all;
%add spm path:
addpath('/Users/daelsaid/tools/spm12')

% Set the directory containing thresholded NIfTI images
data_dir = '/Users/daelsaid/scratch/maps/results/0.05/asd';  % Change this!
addpath(data_dir);

%set image and file output
output_dir='/Users/daelsaid/scratch/maps/results/0.05/asd' %make sure this folder exists

% Define the minimum cluster size threshold (Change this dynamically if needed)
min_cluster_threshold = 50;  % Set a default, will update per image

% Initialize arrays for visualization
all_cluster_sizes = [];

% Get a list of all NIfTI files in the folder
nifti_files = dir(fullfile(data_dir, '*.nii'));

% Output file to save cluster sizes
output_file = fullfile(output_dir, 'SL_clusters_and_voxels_w_FWEc_clusters.txt');
fid = fopen(output_file, 'w');
fprintf(fid, 'Filename\tMin_Cluster\tMax_Cluster\tMean_Cluster\tMedian_Cluster\tFWEc_Threshold\tTotal_Voxels\tFWEc_Voxels\n');

for i = 1:length(nifti_files)
    nii_path = fullfile(data_dir, nifti_files(i).name);

    %load niftis
    V = spm_vol(nii_path);
    Y = spm_read_vols(V);
    disp(['Processing: ', nii_path]);
    disp(['Min voxel value: ', num2str(min(Y(:)))]);
    disp(['Max voxel value: ', num2str(max(Y(:)))]);
    disp(['Number of nonzero voxels: ', num2str(nnz(Y))]);
    
    % ------------------ Process Clusters ------------------
    bw_mask = Y; % Keep all significant voxels (positive)
    disp(['bw_mask data type: ', class(bw_mask)]);
    disp(['Min value in bw_mask: ', num2str(min(bw_mask(:)))]);
    disp(['Max value in bw_mask: ', num2str(max(bw_mask(:)))]);
    disp(['Number of nonzero voxels in bw_mask: ', num2str(nnz(bw_mask))]);
    
    [L, numClusters] = spm_bwlabel(bw_mask, 6);  % Identify clusters
    
    if numClusters == 0
        fprintf(fid, '%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%s\n', nifti_files(i).name, 0, 0, 0, 0, 0, 0, 0, 'None');
        fprintf('No clusters found in %s. Skipping file.\n', nifti_files(i).name);
    end
    
    % Get all cluster sizes before correction
    cluster_sizes_before = histcounts(L, 1:numClusters+1);
    fprintf('Cluster sizes BEFORE correction: %s\n', num2str(cluster_sizes_before));
    cluster_sizes = histcounts(L, 1:numClusters+1);
    
    % % Apply minimum cluster size threshold
    % valid_clusters = find(cluster_sizes >= min_cluster_threshold);
    % valid_cluster_mask = ismember(L, valid_clusters);
    % Y = Y .* valid_cluster_mask;  % Remove clusters smaller than threshold
    % k_FWE = round(prctile(cluster_sizes(cluster_sizes >= min_cluster_threshold), 95));  % Round to nearest whole number
    
    % Compute FWE-corrected extent threshold (95th percentile of cluster sizes)
    k_FWE = round(prctile(cluster_sizes, 95));  % Round to nearest whole number
    
    fprintf('FWEc Threshold: %d voxels\n', k_FWE);
    
    % Compute total voxel counts
    total_voxels = nnz(Y);  % Count all nonzero voxels in the image
    FWEc_voxels = sum(cluster_sizes_before(cluster_sizes_before >= k_FWE));  % Count voxels in significant clusters
    
    % Save results to file
    fprintf(fid, '%s\t%d\t%d\t%.2f\t%.2f\t%d\t%d\t%d\t%s\n', nii_path, ...
        min(cluster_sizes_before), max(cluster_sizes_before), ...
        mean(cluster_sizes_before), median(cluster_sizes_before), ...
        k_FWE, total_voxels, FWEc_voxels);
    
    % Print results
    fprintf('Total Voxels: %d, FWEc Threshold: %d voxels, FWEc Voxels Remaining: %d\n', ...
            total_voxels, k_FWE, FWEc_voxels);
    
    % ------------------ Apply FWEc Threshold and Generate New NIfTI ------------------
    % Create a binary mask for significant clusters
    valid_clusters = ismember(L, find(cluster_sizes_before >= k_FWE));  
    Y_corrected = Y .* valid_clusters;  % Retain only significant clusters
    
    % Save corrected NIfTI image
     [filepath,name,ext] = fileparts(nii_path)
    
    corrected_fname = fullfile(filepath, ['FWEc_' name ext]);
    V.fname = corrected_fname;
    spm_write_vol(V, Y_corrected);
    
    % Append cluster sizes for visualization
    all_cluster_sizes = [all_cluster_sizes, cluster_sizes_before];
end

% Close output file
fclose(fid);

fprintf('Processing complete. Results saved to %s\n', output_file);

% ------------------ Plot Cluster Size Distribution ------------------
figure;
subplot(1,2,1);
histogram(all_cluster_sizes, 'BinWidth', 10);
xlabel('Cluster Size (Voxels)');
ylabel('Frequency');
title('Histogram of Cluster Sizes');

subplot(1,2,2);
cdfplot(all_cluster_sizes);
xlabel('Cluster Size (Voxels)');
ylabel('Cumulative Probability');
title('Cumulative Distribution of Cluster Sizes');

fprintf('Cluster size distribution plot generated.\n');

