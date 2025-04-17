
function [signal8,signal7,signaln7] = getpotrials(eeg,index8,index7,indexn7)

b = 0.2*512;
window = 1*512;
sig7 = []; sig8 = []; sign = [];
 signal8 = [];signal7 = [];signaln7 = [];
 leftElectrodeIndices = [21, 22, 47, 48, 52, 59];
 rightElectrodeIndices = [24, 25, 49, 50, 53, 60];

    for j = 2:1:size(index7,1)-3 %loop over time
   
        
           sig7 = eeg(index7(j)-b:index7(j)+window,leftElectrodeIndices); % get the trial data
           signal7 = cat(3,signal7,sig7); % store all the trials for one channel
    end

     for l = 1:1:size(index8,1)-3 %loop over time
   
           sig8 = eeg(index8(l)-b:index8(l)+window,rightElectrodeIndices); % get the trial data
           signal8 = cat(3,signal8,sig8); % store all the trials for one channel
    end

    for k = 1:1:size(indexn7,1)-3 %loop over time
   
           %if indexn7(j)+window <= size(eeg,1)
            sign = eeg(indexn7(k)-b:indexn7(k)+window,leftElectrodeIndices); % get the trial data
            signaln7 = cat(3,signaln7,sign);
           %end
    end

end