%% BCI PROJECT

% BLANCA DATA

%first load the data : EEG 
eeg1 = load("sloaded data/data export/e1_20241015152755_training_DATA.mat");
eeg1 = eeg1.e1_20241015152755_training_DATA;
eeg2 = load("sloaded data/data export/e1_20241015153146_training_DATA.mat");
eeg2 = eeg2.data;
eeg3 = load("sloaded data/data export/e1_20241015153557_training_DATA.mat");
eeg3 = eeg3.data;
eeg4 = load("sloaded data/data export/e1_20241015154006_training_Data.mat");
eeg4 = eeg4.data;
eeg5 = load("sloaded data/data export/e1_20241015154348_training_DATA.mat");
eeg5 = eeg5.data;
eeg6 = load("sloaded data/data export/e1_20241015154811_training_DATA.mat");
eeg6 = eeg6.data;
eeg7 = load("sloaded data/data export/e1_20241015155211_training_DATA.mat");
eeg7 = eeg7.data;
eeg8 = load("sloaded data/data export/e1_20241015155649_training_DATA.mat");
eeg8 = eeg8.data;

% load the header files 
header1 = load("sloaded data/data export/e1_20241015152755_training_HEADER.mat");
header1 = header1.e1_20241015152755_training_HEADER;
header2 = load("sloaded data/data export/e1_20241015153146_training_HEADER.mat");
header2 = header2.header;
header3 = load("sloaded data/data export/e1_20241015153557_training_HEADER.mat");
header3 = header3.header;
header4 = load("sloaded data/data export/e1_20241015154006_training_HEADER.mat");
header4 = header4.header;
header5 = load("sloaded data/data export/e1_20241015154348_training_HEADER.mat");
header5 = header5.header;
header6 = load("sloaded data/data export/e1_20241015154811_training_HEADER.mat");
header6 = header6.header;
header7 = load("sloaded data/data export/e1_20241015155211_training_HEADER.mat");
header7 = header7.header;
header8 = load("sloaded data/data export/e1_20241015155649_training_HEADER.mat");
header8 = header8.header;

%% load information about trial being dist or no dist  0 if no distractor, 1 if distractor 
l1 = importdata('bianca_s1_20241015/e1_20241015152755_training/e1_20241015152755_training.analysis.txt');
l1 = l1(:,2);
l2 = importdata('bianca_s1_20241015/e1_20241015153146_training/e1_20241015153146_training.analysis.txt');
l2 = l2(:,2);
l3 = importdata('bianca_s1_20241015/e1_20241015153557_training/e1_20241015153557_training.analysis.txt');
l3 = l3(:,2);
l4 = importdata('bianca_s1_20241015/e1_20241015154006_training/e1_20241015154006_training.analysis.txt');
l4 = l4(:,2);
l5 = importdata('bianca_s1_20241015/e1_20241015154348_training/e1_20241015154348_training.analysis.txt');
l5 = l5(:,2);
l6 = importdata('bianca_s1_20241015/e1_20241015154811_training/e1_20241015154811_training.analysis.txt');
l6 = l6(:,2);
l7 = importdata('bianca_s1_20241015/e1_20241015155211_training/e1_20241015155211_training.analysis.txt');
l7 = l7(:,2);
l8 = importdata('bianca_s1_20241015/e1_20241015155649_training/e1_20241015155649_training.analysis.txt');
l8 = l8(:,2);

%% ALEX DATA

%first load the data : EEG 
eeg = load("sloaded data/Alex_EEG_Data.mat");
eeg1 = eeg.data;
eeg2 = eeg.data; eeg3 = eeg.data; eeg4 = eeg.data; eeg5 = eeg.eeg5; eeg6 = eeg.eeg6; eeg7 = eeg.eeg7; eeg8 = eeg.eeg8;

% load the header files 
header = load("sloaded data/Alex_EEG_HEADERS.mat");
header1 = header.header; header2 = header.header2; header3 = header.header3; header4 = header.header4; 
header5 = header.header5; header6 = header.header6; header7 = header.header7; header8 = header.header8; 

%%
filt_eeg8(:,10)=[];
figure();
for i=1:1:size(filt_eeg6,2) 
    plot(filt_eeg6(8876:93549,:)+i*50);
    hold on; 
end

%% Channels 1:64 – EEG / Channels 65:68 – EOG / Channel 69 – Trigger channel

% BLANCA
eeg1data = eeg1.data(1:103392,1:64);eeg2data = eeg2.data(1:107488,1:64);eeg3data = eeg3.data(1:104736,1:64);eeg4data = eeg4.data(1:105696,1:64);
eeg5data = eeg5.data(1:109408,1:64);eeg6data = eeg6.data(1:106336,1:64);eeg7data = eeg7.data(1:122144,1:64);eeg8data = eeg8.data(1:107200,1:64);

%remove M1,M2 and EOG channels
% 
eeg1data(:,32) = [];eeg1data(:,19) = [];eeg1data(:,13) = [];eeg2data(:,32) = [];eeg2data(:,19) = [];eeg2data(:,13) = [];
eeg3data(:,32) = [];eeg3data(:,19) = [];eeg3data(:,13) = [];eeg4data(:,32) = [];eeg4data(:,19) = [];eeg4data(:,13) = [];
eeg5data(:,32) = [];eeg5data(:,19) = [];eeg5data(:,13) = [];eeg6data(:,32) = [];eeg6data(:,19) = [];eeg6data(:,13) = [];
eeg7data(:,32) = [];eeg7data(:,19) = [];eeg7data(:,13) = [];eeg8data(:,32) = [];eeg8data(:,19) = [];eeg8data(:,13) = [];

% take the trigger channel
eeg1_trigger = eeg1.data(:,69);eeg2_trigger = eeg2.data(:,69);eeg3_trigger = eeg3.data(:,69);eeg4_trigger = eeg4.data(:,69);
eeg5_trigger = eeg5.data(:,69);eeg6_trigger = eeg6.data(:,69);eeg7_trigger = eeg7.data(:,69);eeg8_trigger = eeg8.data(:,69);

%% ALEX
eeg1data = eeg1(1:105600,1:64);eeg2data = eeg2(1:107424,1:64);eeg3data = eeg3(1:104352,1:64);eeg4data = eeg4(1:104320,1:64);
eeg5data = eeg5(1:103584,1:64);eeg6data = eeg6(1:107616,1:64);eeg7data = eeg7(1:128224,1:64);eeg8data = eeg8(1:107936,1:64);

%remove M1,M2 and EOG channels

% eeg1data(:,32) = [];eeg1data(:,19) = [];eeg1data(:,13) = [];eeg2data(:,32) = [];eeg2data(:,19) = [];eeg2data(:,13) = [];
% eeg3data(:,32) = [];eeg3data(:,19) = [];eeg3data(:,13) = [];eeg4data(:,32) = [];eeg4data(:,19) = [];eeg4data(:,13) = [];
% eeg5data(:,32) = [];eeg5data(:,19) = [];eeg5data(:,13) = [];eeg6data(:,32) = [];eeg6data(:,19) = [];eeg6data(:,13) = [];
% eeg7data(:,32) = [];eeg7data(:,19) = [];eeg7data(:,13) = [];eeg8data(:,32) = [];eeg8data(:,19) = [];eeg8data(:,13) = [];

% take the trigger channel

eeg1_trigger = eeg1(:,69);eeg2_trigger = eeg2(:,69);eeg3_trigger = eeg3(:,69);eeg4_trigger = eeg4(:,69);
eeg5_trigger = eeg5(:,69);eeg6_trigger = eeg6(:,69);eeg7_trigger = eeg7(:,69);eeg8_trigger = eeg8(:,69);


%% temporal filtering : BUTTERWORTH BPF [0.1 30] Hz
fs = 512;
filt_eeg1=[];
filt_eeg1 = tempfilter(eeg1data,fs);%,'butter','nausal');
filt_eeg2 = tempfilter(eeg2data,fs,'butter','nausal');
filt_eeg3 = tempfilter(eeg3data,fs,'butter','nausal');
filt_eeg4 = tempfilter(eeg4data,fs,'butter','causal');
filt_eeg5 = tempfilter(eeg5data,fs,'butter','causal');
filt_eeg6 = tempfilter(eeg6data,fs,'butter','causal');
filt_eeg7 = tempfilter(eeg7data,fs,'butter','causal');
filt_eeg8 = tempfilter(eeg8data,fs,'butter','causal');


%% ALEX
filt_eeg2 = filt_eeg2(11784:96709,:);
eeg2_trigger = eeg2_trigger(11784:96709,:);
filt_eeg3 = filt_eeg3(1:93596,:);
eeg3_trigger = eeg3_trigger(1:93596,:);
filt_eeg4 = filt_eeg4(1:96871,:);
eeg4_trigger = eeg4_trigger(1:96871,:);
filt_eeg5 = filt_eeg5(5795:97770,:);
eeg5_trigger = eeg5_trigger(5795:97770,:);
filt_eeg6 = filt_eeg6(9443:92209,:);
eeg6_trigger = eeg6_trigger(9443:92209,:);
filt_eeg7 = filt_eeg7(29836:116708,:);
eeg7_trigger = eeg7_trigger(29836:116708,:);
filt_eeg8 = filt_eeg8(10954:96569,:);
eeg8_trigger = eeg8_trigger(10954:96569,:);

%% spatial filtering option 1 : CAR

eegcar1 = car(filt_eeg1);
eegcar2 = car(filt_eeg2);
%eegcar2 = eegcar2(11784:96709,:);
eegcar3 = car(filt_eeg3);
eegcar4 = car(filt_eeg4);
eegcar5 = car(filt_eeg5);
eegcar6 = car(filt_eeg6);
eegcar7 = car(filt_eeg7);
eegcar8 = car(filt_eeg8);


%% now extract the indices of trials with and without distractors
% index variables have the index of the start of the trial 

indexd1 = distnodist(eeg1_trigger,"dist");
indexnd1 = distnodist(eeg1_trigger,"no");

indexd2 = distnodist(eeg2_trigger,"dist");
indexnd2 = distnodist(eeg2_trigger,"no");

indexd3 = distnodist(eeg3_trigger,"dist");
indexnd3 = distnodist(eeg3_trigger,"no");

indexd4 = distnodist(eeg4_trigger,"dist");
indexnd4 = distnodist(eeg4_trigger,"no");

indexd5 = distnodist(eeg5_trigger,"dist");
indexnd5 = distnodist(eeg5_trigger,"no");

indexd6 = distnodist(eeg6_trigger,"dist");
 indexnd6 = distnodist(eeg6_trigger,"no");

indexd7 = distnodist(eeg7_trigger,"dist");
indexnd7 = distnodist(eeg7_trigger,"no");

 indexd8 = distnodist(eeg8_trigger,"dist");
 indexnd8 = distnodist(eeg8_trigger,"no");

%% get the signal across all trials // parsing
% get a 3d array of : time x channels x trials 

dist_trialavg1 = parsing(filt_eeg1,indexd1);
nodist_trialavg1 = parsing(filt_eeg1,indexnd1);
dist_trialavg2 = parsing(filt_eeg2,indexd2);
nodist_trialavg2 = parsing(filt_eeg2,indexnd2);
dist_trialavg3 = parsing(filt_eeg3,indexd3);
nodist_trialavg3 = parsing(filt_eeg3,indexnd3);
dist_trialavg4 = parsing(filt_eeg4,indexd4);
nodist_trialavg4 = parsing(filt_eeg4,indexnd4);
dist_trialavg5 = parsing(filt_eeg5,indexd5);
nodist_trialavg5 = parsing(filt_eeg5,indexnd5);
dist_trialavg6 = parsing(filt_eeg6,indexd6);
nodist_trialavg6 = parsing(filt_eeg6,indexnd6);
dist_trialavg7 = parsing(filt_eeg7,indexd7);
nodist_trialavg7 = parsing(filt_eeg7,indexnd7);
dist_trialavg8 = parsing(filt_eeg8,indexd8);
nodist_trialavg8 = parsing(filt_eeg8,indexnd8);

%% Baseline Correction

b1 = baselinecorrect(dist_trialavg1,fs);
nb1 = baselinecorrect(nodist_trialavg1,fs);

b2 = baselinecorrect(dist_trialavg2,fs);
nb2 = baselinecorrect(nodist_trialavg2,fs);

b3 = baselinecorrect(dist_trialavg3,fs);
nb3 = baselinecorrect(nodist_trialavg3,fs);

b4 = baselinecorrect(dist_trialavg4,fs);
nb4 = baselinecorrect(nodist_trialavg4,fs);

b5 = baselinecorrect(dist_trialavg5,fs);
nb5 = baselinecorrect(nodist_trialavg5,fs);

b6 = baselinecorrect(dist_trialavg6,fs);
nb6 = baselinecorrect(nodist_trialavg6,fs);

b7 = baselinecorrect(dist_trialavg7,fs);
nb7 = baselinecorrect(nodist_trialavg7,fs);

b8 = baselinecorrect(dist_trialavg8,fs);
nb8 = baselinecorrect(nodist_trialavg8,fs);


%% get the average trial across all sessions

distractors = cat(3,b1 ,b2 ,b3 ,b4, b5, b6, b7, b8);
nodistractors = cat(3,nb1 ,nb2, nb3, nb4 ,nb5, nb6, nb7, nb8);
avgtriald = mean(distractors,3);
avgtrialnd = mean(nodistractors,3);

% avg all the channels 
avgchd = mean(avgtriald,2);
avgchnd = mean(avgtrialnd,2);

figure();
plot(avgchd,'LineWidth',3);
hold on
plot(avgchnd,'LineWidth',3);
hold on
xline(100,'LineWidth',2)
xlim([0 400]);
% ylim([-10 10]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
legend('Distractor',"No Distractor");
title('Grand Average (all channels) ERP for Distractor vs no distractor for all 8 training sessions')


%% use only PO/P channels 202-left 204-right nd left-left

% parse again to distinguish between the diff distractors

[i81,i71,in71] = poparse(eeg1_trigger); %% get the indices for the different distractors and no distractors
[i82,i72,in72] = poparse(eeg2_trigger); %% get the indices for the different distractors and no distractors
[i83,i73,in73] = poparse(eeg3_trigger); %% get the indices for the different distractors and no distractors
[i84,i74,in74] = poparse(eeg4_trigger); %% get the indices for the different distractors and no distractors
[i85,i75,in75] = poparse(eeg5_trigger); %% get the indices for the different distractors and no distractors
[i86,i76,in76] = poparse(eeg6_trigger); %% get the indices for the different distractors and no distractors
[i87,i77,in77] = poparse(eeg7_trigger); %% get the indices for the different distractors and no distractors
[i88,i78,in78] = poparse(eeg8_trigger); %% get the indices for the different distractors and no distractors


[po81,po71,pn1] = getpotrials(filt_eeg1,i81,i71,in71);
[po82,po72,pn2] = getpotrials(filt_eeg2,i82,i72,in72);
[po83,po73,pn3] = getpotrials(filt_eeg3,i83,i73,in73);
[po84,po74,pn4] = getpotrials(filt_eeg4,i84,i74,in74);
[po85,po75,pn5] = getpotrials(filt_eeg5,i85,i75,in75);
[po86,po76,pn6] = getpotrials(filt_eeg6,i86,i76,in76);
[po87,po77,pn7] = getpotrials(filt_eeg7,i87,i77,in77);
[po88,po78,pn8] = getpotrials(filt_eeg8,i88,i78,in78);


base81 = baselinecorrect(po81,fs);base82 = baselinecorrect(po82,fs);
base83 = baselinecorrect(po83,fs);base84 = baselinecorrect(po84,fs);
base85 = baselinecorrect(po85,fs);base86 = baselinecorrect(po86,fs);
base87 = baselinecorrect(po87,fs);base88 = baselinecorrect(po88,fs);
base71 = baselinecorrect(po71,fs);base72 = baselinecorrect(po72,fs);
base73 = baselinecorrect(po73,fs);base74 = baselinecorrect(po74,fs);
base75 = baselinecorrect(po75,fs);base76 = baselinecorrect(po76,fs);
base77 = baselinecorrect(po77,fs);base78 = baselinecorrect(po78,fs);
basen1 = baselinecorrect(pn1,fs);basen2 = baselinecorrect(pn2,fs);
basen3 = baselinecorrect(pn3,fs);basen4 = baselinecorrect(pn4,fs);
basen5 = baselinecorrect(pn5,fs);basen6 = baselinecorrect(pn6,fs);
basen7 = baselinecorrect(pn7,fs);basen8 = baselinecorrect(pn8,fs);

ch8 = cat(2,base81,base82,base83,base84,base85,base86,base87,base88);
ch7 = cat(2,base71,base72,base73,base74,base75,base76,base77,base78);
chn = cat(2,basen1,basen2,basen3,basen4,basen5,basen6,basen7,basen8);

%%
figure();
plot(mean(ch8,2),'LineWidth',3);
hold on
plot(mean(ch7,2),'LineWidth',3);
hold on
plot(mean(chn,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-10 10]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
legend('Right Contralateral'," Left Contralateral", 'No Distractor');
%title('Grand Average (all channels) ERP for Distractor vs no distractor for all 8 training sessions')


%% ML implementation 

% causal preprocessing 

% LDA Implementation

%  EEG data has dimensions: Time x Channels x Trials
reshapedData = []; XTrain = []; LTrain =[]; XTest = []; LTest = []; YPred = [];
data = []; 

Xd = distractors(300:600,:,:);
X = nodistractors(300:600,:,:);

data = cat(3,Xd,X);

% Reshape the data into a 2D matrix: (Channels * Time) x Trials
[numTimePoints, numChannels, numTrials] = size(data);
reshapedData = reshape(data, numChannels * numTimePoints, numTrials)';


% Create labels for each trial (assuming binary classification for simplicity)
labels = [];
for i = 1:1:size(distractors,3)
    labels(i) = 1; % distractor
end
for i = size(distractors,3)+1:1:2*size(distractors,3)
    labels(i) = 0; % no distractor
end

% Creating the model on training session

%% Split the data into training and test sets 
cv = cvpartition(labels, 'HoldOut', 0.3);
idx = cv.test;

% Separate into training and test sets
XTrain = reshapedData(~idx, :);
LTrain = labels(~idx);
XTest = reshapedData(idx, :);
LTest = labels(idx);

% Train the LDA model
ldaModel = fitcdiscr(XTrain, LTrain);

% Make predictions on the test set
YPred = predict(ldaModel, XTest);

% Evaluate the model
accuracy = sum(YPred == LTest) / length(LTest);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% NOW TEST THIS MODEL ON THE ONLINE SESSIONS




%%
% Set the number of folds for cross-validation
k = 5; % Adjust k as needed

% Create a cross-validation partition
cv = cvpartition(labels, 'KFold', k);

% Initialize accuracy storage
accuracies = zeros(k, 1);

% Perform k-fold cross-validation
for i = 1:k
    % Get training and test indices for this fold
    trainIdx = cv.training(i);
    testIdx = cv.test(i);
    
    % Separate training and test data
    XTrain = reshapedData(trainIdx, :);
    LTrain = labels(trainIdx);
    XTest = reshapedData(testIdx, :);
    LTest = labels(testIdx)';
    
    % Train the LDA model
    ldaModel = fitcdiscr(XTrain, LTrain);
    
    % Make predictions on the test set
    YPred = predict(ldaModel, XTest);
    
    % Calculate accuracy for this fold
    accuracies(i) = sum(YPred == LTest) / length(LTest);
end

% Calculate and display the average accuracy across all folds
averageAccuracy = mean(accuracies);
fprintf('Average Accuracy across %d folds: %.2f%%\n', k, averageAccuracy * 100);


%%
% Separate into training and test sets
XTrain = reshapedData(~idx, :);
LTrain = labels(~idx);
XTest = reshapedData(idx, :);
LTest = labels(idx);

Mdl = fitcdiscr(XTrain,LTrain,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',...
    struct('AcquisitionFunctionName','expected-improvement-plus'))

