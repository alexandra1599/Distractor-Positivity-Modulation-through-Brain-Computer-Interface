function [decoder,in] = computeDecoder(traindata, trainlabels, norm)

% Initialize the array of selected electrodes
n_samples = size(traindata,1);
n_trials = size(traindata,3);
n_electrodes = 12; % number of electrodes per trial
cepoch = nan(n_samples,n_electrodes,n_trials);

%trainlabels(trainlabels == 2) = 1;

%% Normalize features , giving am the scaling fdacto
% MIN-MAX 
if norm == 'minmax'
    maxim = max(traindata,[],2);
    minim = min(traindata,[],2);

    normalize = @(x)(x-minim) ./ (maxim - minim);
    cepoch = normalize(traindata);
    in = size(cepoch,2);

% LASSO 
elseif norm == 'lasso1'
    lambdamax = 0.1;
    lambda = logspace(log10(0.001*lambdamax),log10(lambdamax),100);
    cv = fitrlinear(traindata',trainlabels,'ObservationsIn','columns','Lambda',lambda, 'KFold', 5, 'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
    mse = kfoldLoss(cv);
    [~, index] = min(mse);
    m = fitrlinear(traindata',trainlabels,"ObservationsIn","columns", "Lambda",lambda(index),'Learner','leastsquares','Solver','sparsa','Regularization','lasso');
    keepindex = m.Beta~=0;
    in = keepindex;
    cepoch = traindata(:,keepindex);

%ZSCORE
elseif norm == 'zscore'
    stdev = std(traindata);
    avg = mean(traindata,1);
    zscore = @(x) (x-avg) ./ stdev;
    cepoch = zscore(traindata);
    in = size(cepoch,2);
end

%% Classification

m = fitcdiscr(cepoch, trainlabels','Prior','uniform','DiscrimType','pseudolinear');

% Activation Function

w = m.Coeffs(2,1).Linear; % this vector separates the two classes in the feature space
mu = m.Coeffs(2,1).Const; % retrieves the constant term μ for the discriminant function between class 2 and class 1

distance = cepoch*w + mu; % projection along the linear discriminant axis, giving a measure of how close it is to each class

p1 = 0.25; % 25th percentile
p2 = 1-p1; % 75th percentile
bcoeff1 = -log((1-p1)/p1)/prctile(distance,100*p1); %scaling coefficients
bcoeff2 = -log((1-p2)/p2)/prctile(distance,100*p2);

b = (bcoeff1+bcoeff2)/2; % scaling factor for the sigmoid function
decoder.b =  b;
decoder.mu = mu;
decoder.w = w;

end