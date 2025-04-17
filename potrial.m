## This function is to get the trials at channels PO7/PO8 where we expect to see the distractor positivity signal

function signal = potrial(eeg,index,type)

if type == 7
    k = 59; %po7
elseif type == 8
    k = 60; %po8
end

b = 0.2*512;
window = 1*512;
sig = []; signal = [];avgch = [];
    for j = 1:1:size(index,2) %loop over time
   
           sig = eeg(index(j)-b:index(j)+window,k); % get the trial data
           signal = cat(2,signal,sig); % store all the trials for one channel
       
    end

%signal = mean(signal,3);

end
