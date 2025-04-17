function [right_dist_baseline_corrected, left_dist_baseline_corrected, nodist_baseline_corrected] = preprocessing(eeg,header) %,trigger)
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% LOAD THE DATA %%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

data = eeg;
h = header;

eegdata = data(:,1:64);
eegdata(~any(eegdata,2),:)=[]; % remove rows of all 0 at the end of the matrix

% remove M1,M2,EOG
eegdata(:,32) = []; eegdata(:,19) = []; eegdata(:,13) = [];
h.Label{32,1} = []; h.Label{19,1} = []; h.Label{13,1} = [];


% take the trigger channel 
trigger = data(:,69);
%trigger = trigger.trigger;

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Temporal filtering %%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fs = h.SampleRate;
filt_eeg = tempfilter(eegdata,fs);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%% Spatial filtering %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%eegcar = car(filt_eeg);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%% Trial Indices + Parsing %%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[i8,i7,in] = poparse(trigger);
[right, left, no] = getpotrials(filt_eeg,i8,i7,in);


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% Baseline Correction %%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

right_dist_baseline_corrected = baselinecorrect(right,fs);
left_dist_baseline_corrected = baselinecorrect(left,fs);
nodist_baseline_corrected = baselinecorrect(no,fs);


end