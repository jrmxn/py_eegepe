%% Convert .mat to .set
% Simple script to convert .mat files of the Healthy Brain Network dataset
% to eeglab .set files. This is to enable easier loading into python.
clearvars;
eeglabonpath;  % replace with your own code to add EEGLab to path
% the following should point to a folder that contains subject folders as
% downloaded. e.g. file contents: 'NDARAA306NT2', 'NDARAA504CRN', ...
f = fullfile(getuserdir, 'Cloud/DataPort/2019-02-29_healthy_brain_network/data');  % replace with your data location
%%
cell_f = glob(fullfile(f,'**RestingState.mat'));  % glob function for matlab - avaialble at the mathworks
for ix_cell_f = 1:length(cell_f)
    f_in = cell_f{ix_cell_f};
    f_out = strrep(strrep(f_in, 'mat_format', 'eeglab_format'), '.mat', '');
    if not(exist(fileparts(f_out), 'dir') == 7)
        mkdir(fileparts(f_out));
    end
    tempEEG = load(f_in);
    EEG = tempEEG.EEG;
    for ix_event = 1:length(EEG.event)
        EEG.event(ix_event).latency = EEG.event(ix_event).sample;
        EEG.event(ix_event).type = strtrim(EEG.event(ix_event).type);
    end
    
    EEG = eeg_checkset(EEG);
    pop_saveset(EEG, 'filepath', f_out, 'version', '6');  % version 6 required for MNE
end