%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%% LOAD DATA %%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

prompt = "Subject ID : ";
ID =  input(prompt);

 if ID == 'e1' % BLANCA
    
    % training data
 
     eeg = load("Blanca_data/Blanca_EEG_Training_Data.mat");
     eeg1 = eeg.eeg1; eeg2 = eeg.eeg2; eeg3 = eeg.eeg3; eeg4 = eeg.eeg4;
     eeg5 = eeg.eeg5; eeg6 = eeg.eeg6; eeg7 = eeg.eeg7; eeg8 = eeg.eeg8;
     header = load('Blanca_data/Blanca_EEG_Training_Headers.mat');
     header1 = header.header; header2 = header.header2; header3 = header.header3; header4 = header.header4;
     header5 = header.header5; header6 = header.header6; header7 = header.header7; header8 = header.header8;

    % decoding data - session 1

     eegd = load("Blanca_data/Blanca_EEG_Decoding1_Data.mat");
     eegd1 = eegd.eeg1; eegd2 = eegd.eeg2; eegd3 = eegd.eeg3; eegd4 = eegd.eeg4;
     eegd5 = eegd.eeg5; eegd6 = eegd.eeg6; eegd7 = eegd.eeg7; eegd8 = eegd.eeg8;
     headerd = load('Blanca_data/Blanca_EEG_Decoding1_Headers.mat');
     headerd1 = headerd.header1; headerd2 = headerd.header2; headerd3 = headerd.header3; headerd4 = headerd.header4;
     headerd5 = headerd.header5; headerd6 = headerd.header6; headerd7 = headerd.header7; headerd8 = headerd.header8;


 elseif ID == 'e2' % ALEX

     % training data

     eeg = load("Alex_data/Alex_EEG_Training_Data.mat");
     eeg1 = eeg.eeg1; eeg2 = eeg.eeg2; eeg3 = eeg.eeg3; eeg4 = eeg.eeg4;
     eeg5 = eeg.eeg5; eeg6 = eeg.eeg6; eeg7 = eeg.eeg7; eeg8 = eeg.eeg8;
     header = load('Alex_data/Alex_EEG_Training_Headers.mat');
     header1 = header.header; header2 = header.header2; header3 = header.header3; header4 = header.header4;
     header5 = header.header5; header6 = header.header6; header7 = header.header7; header8 = header.header8;

     % decoding data - session 1

     eegd = load('Alex_data/Alex_EEG_Decoding1_Data.mat');
     eegd1 = eegd.eeg1; eegd2 = eegd.eeg2; eegd3 = eegd.eeg3; eegd4 = eegd.eeg4;
     eegd5 = eegd.eeg5; eegd6 = eegd.eeg6; eegd7 = eegd.eeg7; eegd8 = eegd.eeg8;
     headerd = load('Alex_data/Alex_EEG_Decoding1_Headers.mat');
     headerd1 = headerd.header1; headerd2 = headerd.header2; headerd3 = headerd.header3; headerd4 = headerd.header4;
     headerd5 = headerd.header5; headerd6 = headerd.header6; headerd7 = headerd.header7; headerd8 = headerd.header8;

      % decoding data - session 2

     eegd21 = eegd.eeg9; eegd22 = eegd.eeg10; eegd23 = eegd.eeg11; eegd24 = eegd.eeg12;
     eegd25 = eegd.eeg13; eegd26 = eegd.eeg14; eegd27 = eegd.eeg15; eegd28 = eegd.eeg16;
     headerd21 = headerd.header9; headerd22 = headerd.header10; headerd23 = headerd.header11; headerd24 = headerd.header12;
     headerd25 = headerd.header13; headerd26 = headerd.header14; headerd27 = headerd.header15; headerd28 = headerd.header16;


 elseif ID == 'e3' % ARMAN

     %training data 

     eeg = load('Arman_data/Arman_EEG_Training_Data.mat');
     eeg1 = eeg.eeg1; eeg2 = eeg.eeg2; eeg3 = eeg.eeg3; eeg4 = eeg.eeg4;
     eeg5 = eeg.eeg5; eeg6 = eeg.eeg6; eeg7 = eeg.eeg7; eeg8 = eeg.eeg8;
     header = load('Arman_data/Arman_EEG_Training_Headers.mat');
     header1 = header.header; header2 = header.header2; header3 = header.header3; header4 = header.header4;
     header5 = header.header5; header6 = header.header6; header7 = header.header7; header8 = header.header8;

     eegd = load('Arman_data/Arman_EEG_Decoding_Data.mat');
     eegd1 = eegd.eeg1; eegd2 = eegd.eeg2; eegd3 = eegd.eeg3; eegd4 = eegd.eeg4;
     eegd5 = eegd.eeg5; eegd6 = eegd.eeg6; eegd7 = eegd.eeg7; eegd8 = eegd.eeg8;
     headerd = load('Arman_data/Arman_EEG_Decoding_Headers.mat');
     headerd1 = headerd.header; headerd2 = headerd.header2; headerd3 = headerd.header3; headerd4 = headerd.header4;
     headerd5 = headerd.header5; headerd6 = headerd.header6; headerd7 = headerd.header7; headerd8 = headerd.header8;

 end


%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%% TIME DOMAIN %%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[r1,l1,n1] = preprocessing(eeg1,header1);
[r2,l2,n2] = preprocessing(eeg2,header2);
[r3,l3,n3] = preprocessing(eeg3,header3);
[r4,l4,n4] = preprocessing(eeg4,header4);
[r5,l5,n5] = preprocessing(eeg5,header5);
[r6,l6,n6] = preprocessing(eeg6,header6);
[r7,l7,n7] = preprocessing(eeg7,header7);
[r8,l8,n8] = preprocessing(eeg8,header8);

ch8 = cat(3,r1,r2,r3,r4,r5,r6,r7,r8); %right
ch7 = cat(3,l1,l2,l3,l4,l5,l6,l7,l8); %left
chn = cat(3,n1,n2,n3,n4,n5,n6,n7,n8); %no dist
ch8m = mean(ch8,3);
ch7m = mean(ch7,3);
chnm = mean(chn,3);

distchannel = cat(3,ch8,ch7);
difference = ch8-chn(:,:,1:96) ;%(:,:,1:216);
meandiff = mean(difference,3);

diff = ch8(:,:,1:88)-ch7 ;
mdiff = mean(diff,3);

%% right vs left distractor plot 
figure();
plot(mean(ch8m,2),'LineWidth',3);
hold on
plot(mean(ch7m,2),'LineWidth',3);
hold on
plot(mean(chnm,2),'LineWidth',3);


xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-6 8]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
legend('Right Contralateral'," Left Contralateral", 'No Distractor');

title('Grand Average for offline sessions')


%% dist vs no dist plot
ch = mean(difference,3);

figure();
plot(mean(ch7m,2),'LineWidth',3);
hold on
plot(mean(chnm,2),'LineWidth',3);
hold on 

xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-6 8]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
legend('Distractor', 'No Distractor');  %, 'Difference Wave');

title('Grand Average for offline session')

%% difference plot


figure();
plot(mean(mdiff,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation
ylabel('Amplitude (uV)')
ylim([-5 5]);

title('Difference signal for offline')

%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Online Analysis %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%[ro1,lo1,no1] = preprocessing(eegd1,headerd1); % ,trigger);
[ro2,lo2,no2] = preprocessing(eegd2,headerd2); %,trigger);
[ro3,lo3,no3] = preprocessing(eegd3,headerd3); %,trigger);
[ro4,lo4,no4] = preprocessing(eegd4,headerd4); %,trigger);
[ro5,lo5,no5] = preprocessing(eegd5,headerd5); %,trigger);
[ro6,lo6,no6] = preprocessing(eegd6,headerd6); %,trigger);
[ro7,lo7,no7] = preprocessing(eegd7,headerd7); %,trigger);
[ro8,lo8,no8] = preprocessing(eegd8,headerd8); %,trigger);


cho8 = cat(3,ro2,ro3,ro4,ro5,ro6,ro7,ro8);
cho7 = cat(3,lo2,lo3,lo4,lo5,lo6,lo7,lo8);
chon = cat(3,no2,no3,no4,no5,no6,no7,no8);

cho8mn = mean(cho8,3);
cho7mn = mean(cho7,3);
chonm = mean(chon,3);
distchannelon = cat(3,cho8,cho7);


diffo = cho8(:,:,1:77) - cho7;
mdiffo = mean(diffo,3);

%%
%session 1
figure();
plot(mean(cho8mn,2),'LineWidth',3);
hold on
plot(mean(cho7mn,2),'LineWidth',3);
hold on
plot(mean(chonm,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-6 8]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
%legend('Right Contralateral'," Left Contralateral", 'No Distractor');
title('Grand Average for online session ')


%% dist vs no dist plot
cho = mean(distchannelon,3);

figure();
plot(mean(cho8mn,2),'LineWidth',3);
hold on
plot(mean(chonm,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-6 6]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
legend('Distractor', 'No Distractor');

title('Grand Average ERP with causal filter (ONLINE)')


%% difference plot


figure();
plot(mean(mdiffo,2),'LineWidth',3);
% hold on
% plot(mean(mdiff,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation
ylabel('Amplitude (uV)')
ylim([-5 5]);
legend('Online', 'Offline');
title('Difference signal for online session 1')

%% decoding data - session 2

     eegd21 = eegd.eeg9; eegd22 = eegd.eeg10; eegd23 = eegd.eeg11; eegd24 = eegd.eeg12;
     eegd25 = eegd.eeg13; eegd26 = eegd.eeg14; eegd27 = eegd.eeg15; eegd28 = eegd.eeg16;
     headerd21 = headerd.header9; headerd22 = headerd.header10; headerd23 = headerd.header11; headerd24 = headerd.header12;
     headerd25 = headerd.header13; headerd26 = headerd.header14; headerd27 = headerd.header15; headerd28 = headerd.header16;

     
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% Online Analysis %%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[ron1,lon1,non1] = preprocessing(eegd21,headerd21); %,trigger);
[ron2,lon2,non2] = preprocessing(eegd22,headerd22); %,trigger);
[ron3,lon3,non3] = preprocessing(eegd23,headerd23); %,trigger);
[ron4,lon4,non4] = preprocessing(eegd24,headerd24); %,trigger);
[ron5,lon5,non5] = preprocessing(eegd25,headerd25); %,trigger);
[ron6,lon6,non6] = preprocessing(eegd26,headerd26); %,trigger);
[ron7,lon7,non7] = preprocessing(eegd27,headerd27); %,trigger);
[ron8,lon8,non8] = preprocessing(eegd28,headerd28); %,trigger);

cho8n2 = cat(3,ron1,ron2,ron3,ron4,ron5,ron6,ron7,ron8);
cho7n2 = cat(3,lon1,lon2,lon3,lon4,lon5,lon6,lon7,lon8);
chonn2 = cat(3,non1,non2,non3,non4,non5,non6,non7,non8);
ch8n2 = mean(cho8n2,3);
ch7n2 = mean(cho7n2,3);
chnnm2 = mean(chonn2,3);

distchannelon2 = cat(3,cho8n2,cho7n2);


% diffo2 = distchannelon2-chonn2(:,:,1:184) ;
% mdiffo2 = mean(diffo2,3);
diffo2 = cho8n2(:,:,1:88)-cho7n2 ;
mdiffo2 = mean(diffo2,3);

%% Session 2

figure();
plot(mean(cho8mn,2),'LineWidth',3);
hold on
plot(mean(cho7mn,2),'LineWidth',3);
hold on
plot(mean(chonm,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-6 8]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
%legend('Right Contralateral'," Left Contralateral", 'No Distractor');
title('Grand Average for online session ')


%% dist vs no dist plot
cho2 = mean(distchannelon2,3);

figure();
plot(mean(cho2,2),'LineWidth',3);
hold on
plot(mean(chnnm2,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
ylim([-6 6]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation

ylabel('Amplitude (uV)')
legend('Distractor', 'No Distractor');

title('Grand Average ERP with causal filter (ONLINE)')


%% difference plot

figure();
plot(mean(mdiffo2,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation
ylabel('Amplitude (uV)')
ylim([-5 5]);

title('Difference signal for online session 2')


%% difference plot for all sessions

figure();
plot(mean(mdiff,2),'LineWidth',3);
hold on
plot(mean(mdiffo,2),'LineWidth',3);
hold on
plot(mean(mdiffo2,2),'LineWidth',3);

xline(100,'LineWidth',1)
xlim([0 400]);
xlabel('Time (ms)')
xticks([0 100 200 300 400 500 600]); % Original x values
xticklabels({'-200', '0', '200','400','600','800','1000'}); % New labels corresponding to the transformation
ylabel('Amplitude (uV)')
ylim([-5 5]);
legend('Offline Session','Online Session 1', 'Online Session 2')
title('Difference signal for online session 2')
%% t-test

x = mean(distchannel(:,:,:),3);
y = mean(chn(:,:,1:216),3);

[h,p] = ttest2(x,y);
p1=mean(p);

%% ML implementation 
 XTrain = [];% YPred = [];
 XTest = []; %pred=[]; accuracy =[];
% separate per session
d1 = cat(3, r1,l1); d2 = cat(3, r2,l2); 
d3 = cat(3, r3,l3); d4 = cat(3, r4,l4); 
d5 = cat(3, r5,l5); d6 = cat(3, r6,l6); 
d7 = cat(3, r7,l7); d8 = cat(3, r8,l8); 

%no distractors are in n1,n2,...,n8

% Reshape the data into a 2D matrix: (Channels * Time) x Trials

[numTimePoints, numChannels, numTrials] = size(n1);
[numTimePointsd, numChannelsd, numTrialsd] = size(d1);

reshaped1 = reshape(d1, numTrialsd, numChannelsd * numTimePointsd);
reshaped2 = reshape(d2, numTrialsd, numChannelsd * numTimePointsd);
reshaped3 = reshape(d3, numTrialsd, numChannelsd * numTimePointsd);
reshaped4 = reshape(d4, numTrialsd, numChannelsd * numTimePointsd);
reshaped5 = reshape(d5, numTrialsd, numChannelsd * numTimePointsd);
reshaped6 = reshape(d6, numTrialsd, numChannelsd * numTimePointsd);
reshaped7 = reshape(d7, numTrialsd, numChannelsd * numTimePointsd);
reshaped8 = reshape(d8, numTrialsd, numChannelsd * numTimePointsd);

reshapedn1 = reshape(n1, numTrials, numChannels * numTimePoints);
reshapedn2 = reshape(n2, numTrials, numChannels * numTimePoints);
reshapedn3 = reshape(n3, numTrials, numChannels * numTimePoints);
reshapedn4 = reshape(n4, numTrials, numChannels * numTimePoints);
reshapedn5 = reshape(n5, numTrials, numChannels * numTimePoints);
reshapedn6 = reshape(n6, numTrials, numChannels * numTimePoints);
reshapedn7 = reshape(n7, numTrials, numChannels * numTimePoints);
reshapedn8 = reshape(n8, numTrials, numChannels * numTimePoints);

% Create labels for each trial
labelsd = [];labelsnd=[];
for i = 1:1:7*size(reshaped1,1)
    labelsd(i) = 1; % distractor
end
for i = 1:1:7*size(reshapedn1,1)
    labelsnd(i) = 0; % no distractor
end
labelstrain = cat(2,labelsd,labelsnd);
labelsd1 = [];labelsnd1=[];
for i = 1:1:size(reshaped1,1)
    labelsd1(i) = 1; % distractor
end
for i = 1:1:size(reshapedn1,1)
    labelsnd1(i) = 0; % no distractor
end

labelstest = cat(2,labelsd1,labelsnd1);

XTrain = cat(1,reshaped1,reshaped2,reshaped3,reshaped4,reshaped5,reshaped6,reshaped7,...
     reshapedn1,reshapedn2,reshapedn3,reshapedn4,reshapedn5,reshapedn6,reshapedn7);
XTest = cat(1,reshaped8,reshapedn8);

% Train the LDA model
[decoder,in] = computeDecoder(XTrain,labelstrain,'zscore');

% Make predictions on the test set using the selected features

m = @(x) 1./(1+exp(-decoder.b*(x*decoder.w+decoder.mu))); 
YPred = m(XTest);

%
%[x,y,t,auc,opt] = perfcurve(~LTest,1-YPred,1,'Prior','uniform');
% threshold = 1-t(x==opt(1) & y==opt(2));
pred = (YPred > 0.3);
accuracy = mean(pred == labelstest);
disp(['Validation accuracy: ', num2str(accuracy)]);

%% give in the difference signal 
% 
% diff1 = d1-n1(:,:,1:27); diff2 = d2-n2(:,:,1:27); diff3 = d3-n3(:,:,1:27);
% diff4 = d4-n4(:,:,1:27); diff5 = d5-n5(:,:,1:27); diff6 = d6-n6(:,:,1:27);
% diff7 = d7-n7(:,:,1:27); diff8 = d8-n8(:,:,1:27);
% 
% [numTimePoints, numChannels, numTrials] = size(diff1);
% 
% reshapediff1 = reshape(diff1, numTrials, numChannels * numTimePoints);
% reshapediff2 = reshape(diff2, numTrials, numChannels * numTimePoints);
% reshapediff3 = reshape(diff3, numTrials, numChannels * numTimePoints);
% reshapediff4 = reshape(diff4, numTrials, numChannels * numTimePoints);
% reshapediff5 = reshape(diff5, numTrials, numChannels * numTimePoints);
% reshapediff6 = reshape(diff6, numTrials, numChannels * numTimePoints);
% reshapediff7 = reshape(diff7, numTrials, numChannels * numTimePoints);
% reshapediff8 = reshape(diff8, numTrials, numChannels * numTimePoints);
% 
% labelsdiff = [];
% for i = 1:1:7*size(reshapediff1,1)
%     labelsd(i) = 1; 
% end
%% do it as sessions
k = 8; % number of sessions, number of folds for cross-validation
accuracies =[];
% Perform k-fold cross-validation
for i = 1:1:8
XTrain = []; YPred = []; 
XTest = []; 

    % in each iteration we want to train on 7 folds and test on the remaining 1
    
    if i == 1
        XTrain = cat(1,reshaped2,reshaped3,reshaped4,reshaped5,reshaped6,reshaped7,reshaped8, ...
            reshapedn2,reshapedn3,reshapedn4,reshapedn5,reshapedn6,reshapedn7,reshapedn8);
        XTest = cat(1,reshaped1,reshapedn1);

    elseif i == 2
        XTrain = cat(1,reshaped1,reshaped3,reshaped4,reshaped5,reshaped6,reshaped7,reshaped8, ...
            reshapedn1,reshapedn3,reshapedn4,reshapedn5,reshapedn6,reshapedn7,reshapedn8);
        XTest = cat(1,reshaped2,reshapedn2);

    elseif i == 3
        XTrain = cat(1,reshaped1,reshaped2,reshaped4,reshaped5,reshaped6,reshaped7,reshaped8, ...
            reshapedn1,reshapedn2,reshapedn4,reshapedn5,reshapedn6,reshapedn7,reshapedn8);
        XTest = cat(1,reshaped3,reshapedn3);

    elseif i == 4
        XTrain = cat(1,reshaped1,reshaped2,reshaped3,reshaped5,reshaped6,reshaped7,reshaped8, ...
            reshapedn1,reshapedn2,reshapedn3,reshapedn5,reshapedn6,reshapedn7,reshapedn8);
        XTest = cat(1,reshaped4,reshapedn4);

    elseif i == 5
        XTrain = cat(1,reshaped1,reshaped2,reshaped3,reshaped4,reshaped6,reshaped7,reshaped8, ...
            reshapedn1,reshapedn2,reshapedn3,reshapedn4,reshapedn6,reshapedn7,reshapedn8);
        XTest = cat(1,reshaped5,reshapedn5);

    elseif i == 6
        XTrain = cat(1,reshaped1,reshaped2,reshaped3,reshaped4,reshaped5,reshaped7,reshaped8, ...
            reshapedn1,reshapedn2,reshapedn3,reshapedn4,reshapedn5,reshapedn7,reshapedn8);
        XTest = cat(1,reshaped6,reshapedn6);

    elseif i == 7
        XTrain = cat(1,reshaped1,reshaped2,reshaped3,reshaped4,reshaped5,reshaped6,reshaped8, ...
            reshapedn1,reshapedn2,reshapedn3,reshapedn4,reshapedn5,reshapedn6,reshapedn8);
        XTest = cat(1,reshaped7,reshapedn7);

    elseif i == 8
        XTrain = cat(1,reshaped1,reshaped2,reshaped3,reshaped4,reshaped5,reshaped6,reshaped7, ...
            reshapedn1,reshapedn2,reshapedn3,reshapedn4,reshapedn5,reshapedn6,reshapedn7);
        XTest = cat(1,reshaped8,reshapedn8);
    end
    
   labelsd = [];labelsnd=[];
for i1 = 1:1:7*size(reshaped1,1)
    labelsd(i1) = 1; % distractor
end
for i2 = 1:1:7*size(reshapedn1,1)
    labelsnd(i2) = 0; % no distractor
end

labelstrain = cat(2,labelsd,labelsnd);
% permuted_labelstr = labelstrain(randperm(length(labelstrain)));
% for i = 1:1:1000
% 
% permuted_labelstr = permuted_labelstr(randperm(length(permuted_labelstr)));
% end

labelsd1 = [];labelsnd1=[];
for i3 = 1:1:size(reshaped1,1)
    labelsd1(i3) = 1; % distractor
end
for i4 = 1:1:size(reshapedn1,1)
    labelsnd1(i4) = 0; % no distractor
end

labelstest = cat(2,labelsd1,labelsnd1);
permuted_labelste = labelstest(randperm(length(labelstest)));
% 
% for i = 1:1:1000
% permuted_labelste = permuted_labelste(randperm(length(permuted_labelste)));
% 
% end
 
    
   [decoder,in] = computeDecoder(XTrain,labelstrain,'zscore');


%Make predictions on the test set using the selected features
m = @(x) 1./(1+exp(-decoder.b*(x*decoder.w+decoder.mu))); 
YPred = m(XTest);


%[x,y,t,auc,opt] = perfcurve(~LTest,1-YPred,1,'Prior','uniform');
% threshold = 1-t(x==opt(1) & y==opt(2));
pred = (YPred > 0.8);
  accuracies(i) = mean(pred == labelstest');

   
    
end

% Calculate and display the average accuracy across all folds
averageAccuracy = mean(accuracies);
fprintf('Average Accuracy across %d folds: %.2f%%\n', k, averageAccuracy * 100);

cm = confusionmat(logical(labelstest),pred);
disp(cm);

%% NOW TEST THIS MODEL ON THE ONLINE SESSIONS

% ONLINE SESSION 1
% separate per session
%do1 = cat(3, ro1,lo1);
do2 = cat(3, ro2,lo2); 
do3 = cat(3, ro3,lo3); do4 = cat(3, ro4,lo4); 
do5 = cat(3, ro5,lo5); do6 = cat(3, ro6,lo6); 
do7 = cat(3, ro7,lo7); do8 = cat(3, ro8,lo8); 

%no distractors are in n1,n2,...,n8

% Reshape the data into a 2D matrix: (Channels * Time) x Trials

[numTimePoints, numChannels, numTrials] = size(n2);
[numTimePointsd, numChannelsd, numTrialsd] = size(do2);

%reshapedo1 = reshape(do1, numTrialsd, numChannelsd * numTimePointsd);
reshapedo2 = reshape(do2, numTrialsd, numChannelsd * numTimePointsd);
reshapedo3 = reshape(do3, numTrialsd, numChannelsd * numTimePointsd);
reshapedo4 = reshape(do4, numTrialsd, numChannelsd * numTimePointsd);
reshapedo5 = reshape(do5, numTrialsd, numChannelsd * numTimePointsd);
reshapedo6 = reshape(do6, numTrialsd, numChannelsd * numTimePointsd);
reshapedo7 = reshape(do7, numTrialsd, numChannelsd * numTimePointsd);
reshapedo8 = reshape(do8, numTrialsd, numChannelsd * numTimePointsd);

%reshapedon1 = reshape(no1, numTrials, numChannels * numTimePoints);
reshapedon2 = reshape(no2, numTrials, numChannels * numTimePoints);
reshapedon3 = reshape(no3, numTrials, numChannels * numTimePoints);
reshapedon4 = reshape(no4, numTrials, numChannels * numTimePoints);
reshapedon5 = reshape(no5, numTrials, numChannels * numTimePoints);
reshapedon6 = reshape(no6, numTrials, numChannels * numTimePoints);
reshapedon7 = reshape(no7, numTrials, numChannels * numTimePoints);
reshapedon8 = reshape(no8, numTrials, numChannels * numTimePoints);


% Create labels for each trial
labelsd = [];labelsnd=[];
for i = 1:1:7*size(reshapedo2,1)
    labelsd(i) = 1; % distractor
end
for i = 1:1:7*size(reshapedon2,1)
    labelsnd(i) = 0; % no distractor
end
labels = cat(2,labelsd,labelsnd);
 YPred = [];
XTest = []; pred=[]; accuracy =[];

XTest = cat(1,reshapedo2,reshapedo3,reshapedo4,reshapedo5,reshapedo6,reshapedo7,reshapedo8, ...
    reshapedon2,reshapedon3,reshapedon4,reshapedon5,reshapedon6,reshapedon7,reshapedon8);

% Make predictions on the test set using the selected features
%
m = @(x) 1./(1+exp(-decoder.b*(x*decoder.w+decoder.mu))); 
YPred = m(XTest);


%[x,y,t,auc,opt] = perfcurve(~LTest,1-YPred,1,'Prior','uniform');
% threshold = 1-t(x==opt(1) & y==opt(2));
pred = (YPred > 0.4);
accuracy = mean(pred == labelstest);
disp(['Validation accuracy on session 1 data: ', num2str(accuracy)]);



%% separate per session
%don1 = cat(3, ron1,lon1); 
don2 = cat(3, ron2,lon2); 
don3 = cat(3, ron3,lon3); don4 = cat(3, ron4,lon4); 
don5 = cat(3, ron5,lon5); don6 = cat(3, ron6,lon6); 
don7 = cat(3, ron7,lon7); don8 = cat(3, ron8,lon8); 

%no distractors are in n1,n2,...,n8

% Reshape the data into a 2D matrix: (Channels * Time) x Trials

[numTimePoints, numChannels, numTrials] = size(no2);
[numTimePointsd, numChannelsd, numTrialsd] = size(don2);

%reshapedoo1 = reshape(don1, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo2 = reshape(don2, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo3 = reshape(don3, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo4 = reshape(don4, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo5 = reshape(don5, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo6 = reshape(don6, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo7 = reshape(don7, numTrialsd, numChannelsd * numTimePointsd);
reshapedoo8 = reshape(don8, numTrialsd, numChannelsd * numTimePointsd);

%reshapedoon1 = reshape(non1, numTrials, numChannels * numTimePoints);
reshapedoon2 = reshape(non2, numTrials, numChannels * numTimePoints);
reshapedoon3 = reshape(non3, numTrials, numChannels * numTimePoints);
reshapedoon4 = reshape(non4, numTrials, numChannels * numTimePoints);
reshapedoon5 = reshape(non5, numTrials, numChannels * numTimePoints);
reshapedoon6 = reshape(non6, numTrials, numChannels * numTimePoints);
reshapedoon7 = reshape(non7, numTrials, numChannels * numTimePoints);
reshapedoon8 = reshape(non8, numTrials, numChannels * numTimePoints);


% Create labels for each trial
labelsd = [];labelsnd=[];
for i = 1:1:7*size(reshapedoo2,1)
    labelsd(i) = 1; % distractor
end
for i = 1:1:7*size(reshapedoon2,1)
    labelsnd(i) = 0; % no distractor
end
labels = cat(2,labelsd,labelsnd);
XTrain = []; YPred = [];
XTest = []; pred=[]; accuracy =[];

XTest = cat(1,reshapedoo2,reshapedoo3,reshapedoo4,reshapedoo5,reshapedoo6,reshapedoo7,reshapedoo8, ...
    reshapedoon2,reshapedoon3,reshapedoon4,reshapedoon5,reshapedoon6,reshapedoon7,reshapedoon8);

% Make predictions on the test set using the selected features
%
m = @(x) 1./(1+exp(-decoder.b*(x*decoder.w+decoder.mu))); 
YPred = m(XTest);



%[x,y,t,auc,opt] = perfcurve(~LTest,1-YPred,1,'Prior','uniform');
% threshold = 1-t(x==opt(1) & y==opt(2));
pred = (YPred > 0.4);
accuracy = mean(pred == labelstest);
disp(['Validation accuracy on session 2 data: ', num2str(accuracy)]);


