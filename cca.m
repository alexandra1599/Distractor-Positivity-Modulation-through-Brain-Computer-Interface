function spf = cca(eeg, labels)

concat_data = [];
concat_ga = [];
classes = unique(labels);

for iClass = 1:length(classes)
    exClass = classes(iClass);
    exEpochs = eeg(:, :, labels == exClass);
    [n_samples,n_channels,n_trials] = size(exEpochs);        
    exEpochs = permute(exEpochs, [2 1 3]);
    ga_data = mean(exEpochs, 3);
    
    ex_epochs = reshape(exEpochs,[n_channels n_samples*n_trials]); % channel x sample*trial
    ga_data = repmat(ga_data,[1 n_trials]); % channel x sample*trial
    
    concat_data = cat(2, concat_data, ex_epochs);
    concat_ga = cat(2, concat_ga, ga_data);
end
[spf,~] =  canoncorr(concat_data', concat_ga');

end