% time_shift

function time_shift_sl_data(speaker_fmri_data_path,listener_fmri_data_path,output_path)
clear all; close all; clc

% Specify directory with Speaker Pre-processed Data
speaker_fmri_data_dir = speaker_fmri_data_path
% '/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/sl_story2/processed_data/speaker_data';

% Get the first NIfTI file in the speaker data directory
speaker_files = dir(fullfile(speaker_fmri_data_dir, '*.nii'));
if isempty(speaker_files)
    error('No NIfTI files found in the specified speaker data directory.');
end
speaker_fmri_data = fullfile(speaker_fmri_data_dir, speaker_files(1).name); % Use the first NIfTI file found

% Specify directory with Listener Pre-processed Data
listener_fmri_data_dir = listener_fmri_data_path
% '/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/sl_story2_asd/processed_data/listener_data';

% Load fMRI data for speaker
speaker_data = niftiread(speaker_fmri_data); % Load speaker fMRI data

% Get the dimensions of the speaker data
[rows, cols, slices, timepoints] = size(speaker_data);

% Define the range of time shifts (in timepoints)
max_shift = 10; % Maximum time shift
shifts = -max_shift:max_shift; % Time shifts from -max_shift to +max_shift

% Extract base name for output filenames
[~, speaker_name, ~] = fileparts(speaker_fmri_data);

% Get list of all NIfTI files in the listener data directory
listener_files = dir(fullfile(listener_fmri_data_dir, '*.nii'));

% Loop through each time shift
for i = 1:length(shifts)
    shift = shifts(i);

    % Determine the output directory based on the shift
    if shift > 0
        shift_label = sprintf('Speaker_Precedes_Listener_%d', shift);
    elseif shift < 0
        shift_label = sprintf('Listener_Precedes_Speaker_%d', abs(shift));
    else
        shift_label = 'No_Shift';
    end

    % Create output directory based on shift
    % '/oak/stanford/groups/menon/projects/daelsaid/2022_speaker_listener/results/post_processed_wholebrain/sl_story2/processed_data/ts_output'
    output_dir = fullfile(output_path, shift_label);
    if ~exist(output_dir, 'dir')
        mkdir(output_dir);
    end

    % Loop through each listener file
    for j = 1:length(listener_files)
        listener_fmri_data = fullfile(listener_fmri_data_dir, listener_files(j).name); % Full path to the listener file
        listener_data = niftiread(listener_fmri_data); % Load listener fMRI data

        % =====================================
        % Apply time shift to speaker data
        % =====================================

        % "shift > 0" is for Speaker precedes Listener
        if shift > 0
            % Shift the speaker data backward (earlier)
            shifted_speaker_data = speaker_data(:, :, :, 1+shift:end);
            % Keep the listener data constant but remove points at end
            listener_data_shifted = listener_data(:, :, :, 1:end-shift);

            % "shift < 0" is for Listener precedes Speaker
        elseif shift < 0
            % Keep the speaker data constant but remove points at end
            shifted_speaker_data = speaker_data(:, :, :, 1:end+shift);
            % Shift the listener data backward (earlier)
            listener_data_shifted = listener_data(:, :, :, 1-shift:end);

            % No time shift between Speaker and Listener:
        else
            shifted_speaker_data = speaker_data;
            listener_data_shifted = listener_data;
        end

        % Ensure the dimensions match after shifting
        min_timepoints = min(size(shifted_speaker_data, 4), size(listener_data_shifted, 4));
        shifted_speaker_data = shifted_speaker_data(:, :, :, 1:min_timepoints);
        listener_data_shifted = listener_data_shifted(:, :, :, 1:min_timepoints);

        % Convert to int16
        shifted_speaker_data_int16 = int16(shifted_speaker_data);
        listener_data_shifted_int16 = int16(listener_data_shifted);

        %         % Save the new time-shifted data as a .nii
        %         % Define the filename for the NIfTI file
        %         % Remove the shift information from the filenames
        %         speaker_filename = fullfile(output_dir, sprintf('%s_shifted.nii', speaker_name)); % Use original speaker name
        %         listener_filename = fullfile(output_dir, sprintf('%s_shifted.nii', listener_files(j).name)); % Use original listener name

        % Save the new time-shifted data as a .nii
        % Define the filename for the NIfTI file
        speaker_filename = fullfile(output_dir, sprintf('%s_shifted_Speaker.nii', speaker_name)); % Use original speaker name

        % Get the base name of the listener file without the extension
        listener_filename = fullfile(output_dir, sprintf('%s_shifted.nii', listener_files(j).name(1:end-4))); % Use original listener base name

        % ==========================================
        % ==========================================
        %         % USE THESE LINES FOR TESTING ONLY
        %         % The real data will have usable NIfTI metadata structure
        %         % Comment this section out when testing is done
        %
        %         % Create a new NIfTI metadata structure based on the original speaker file
        %         test_data_file = '/scratch/users/daa/SL_TI_Devel/Time_Shift_Test/test_data.gz';
        %         nifti_info = niftiinfo(test_data_file); % Get the original NIfTI info
        %
        %         % Create a new NIfTI metadata structure
        %         %         nifti_info = struct();
        %         nifti_info.Filename = speaker_filename; % Output filename
        %         nifti_info.ImageSize = size(shifted_speaker_data_int16); % Size of the data
        %         %         nifti_info.PixelDimensions = [2.81, 2.35, 2.81, 1]; % Voxel size in mm (example values)
        %         %nifti_info.Datatype = 32; % Set to 32 for float32
        %         nifti_info.Magic = 'n+1'; % NIfTI magic number
        %         nifti_info.QformCode = 1; % Orientation code
        %         nifti_info.SformCode = 1; % Orientation code
        %         nifti_info.QtoXYZ = eye(4); % Affine transformation matrix
        %         nifti_info.Qfac = 1; % Orientation factor

        % Ensure the data type matches the NIfTI header
        %         shifted_speaker_data = single(shifted_speaker_data); % Ensure it's single
        %         listener_data_shifted = single(listener_data_shifted); % Ensure it's single

        % Check the class of the data before writing
        %         disp(['Class of shifted_speaker_data: ', class(shifted_speaker_data)]);
        %         disp(['Class of listener_data_shifted: ', class(listener_data_shifted)]);

        %         % Write the 4D data to NIfTI files
        %         niftiwrite(shifted_speaker_data_int16, speaker_filename, nifti_info);
        %         niftiwrite(listener_data_shifted_int16, listener_filename, nifti_info);
        % ==========================================
        % ==========================================

        % ==========================================
        % ==========================================
        % UNCOMMENT THESE LINES WHEN RUNNING ON REAL DATA

        % Define the NIfTI metadata
        nifti_info = niftiinfo(speaker_fmri_data); % Assuming same metadata for both
        nifti_info.ImageSize = size(shifted_speaker_data);
        % nifti_info.PixelDimensions = [2, 2, 2, 0]; % Voxel size in mm (example values)

%         % Ensure the data type matches the NIfTI header
%         shifted_speaker_data = cast(shifted_speaker_data, 'like', zeros(1, 1, 1, 1, 'single')); % Convert to single
%         listener_data_shifted = cast(listener_data_shifted, 'like', zeros(1, 1, 1, 1, 'single')); % Convert to single

        % Write the 4D data to NIfTI files
        niftiwrite(shifted_speaker_data, speaker_filename, nifti_info);
        niftiwrite(listener_data_shifted, listener_filename, nifti_info);
        % ==========================================
        % ==========================================

    end
end
end