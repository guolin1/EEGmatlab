function mnn_Data = preprocess_mnn(Data)
%% Description
% mnn_Data = preprocess_mnn(Data)
% conducts multivariate noise normalisation on data following Guggenmos et
% al (2018) Multivariate pattern analysis for MEG: a comparison of
% dissimilarity measures. NeuroImage 173
%
% Covariance matrix is computed folowing Ledoit and Wolf (2003, 2004) function
% CovCor [Copyright (c) 2014, Olivier Ledoit and Michael Wolf]
%
% input:
%       Data: nConditions, nChannels, nTimes, nObservations
%
% output:
%       mnn_Data: nConditions, nChannels, nTimes, nObservations
%
% IMPORTANT: Data input must include all conditions in order for mnn to be
%            non-biased. 
%
% Lawrence December 2019
[nConds, nElecs, nTimes, nObs] = size(Data);
Data = permute(Data,[1 4 2 3]);

disp('Obtaining covariance matrices from ERP for mnn');
% Select epoch time points
for iCond = 1:nConds
    parfor iTime = 1:nTimes
        icov(iCond,iTime,:,:) = covCor(squeeze(Data(iCond,:,:,iTime))); % calculate covariance matrix at each time point
    end
end
sig = squeeze(nanmean(nanmean(icov,2),1)); % average covariance matrices across time, and then across conditions
disp('Normalizing by covariance matrix');
% concatenate condition & trials to make normalisation faster
Data = reshape(Data, [nConds*nObs, nElecs, nTimes]);
parfor iTime = 1:nTimes
    idata = squeeze(Data(:,:,iTime));
    ndata(:,:,iTime) = idata * (sig^(-1/2)); % normalize SOA1 data
end
mnn_Data = reshape(ndata,[nConds,nObs,nElecs,nTimes]);
mnn_Data = permute(mnn_Data, [1 3 4 2]);
end
