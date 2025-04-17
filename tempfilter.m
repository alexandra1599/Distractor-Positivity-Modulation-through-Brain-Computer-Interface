function filt_eeg = tempfilter(eeg,fs)

        [b1,a1] = butter(4,[1, 30]/(fs/2),'bandpass');

    
    %   filt_eeg = (filtfilt(b1,a1,eeg)); % non causal 
  filt_eeg = (filter(b1,a1,eeg)); % causal


end