function signal = car(eeg)

        signal = [];
        avgch = mean(eeg,2); % get the avg channel

        signal = eeg(:,:) - avgch;
end 