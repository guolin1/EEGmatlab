function timefreqdata = timefreq_wavelet(Data,freqrange,freqstep,fwhm_ncycles)
%% Description
% timefreq_wavelet(selData,freqrange,fwhm_ncycles)
% Performs wavelet transform following Cohen MX (2019) "A better way to
% define and describe Morlet wavelets for time-frequency analysis"
% NeuroImage (199)
% Input:
%       Data: nChannels x nTimes x nEpochs 
%       freqrange: lower and upper bound 
%       freqstep: step between each freq search [default: 1]
%       fwhm_ncycles: using number of cycles to define fwhm [default: 2]
% Output:
%       timefreqdata: nChannels x nTimes x nFrequencies x nEpochs
%
% IMPORTANT: Note that Additional processing such as averaging, multivaraite
% noise normalisation, and baseline corrections still need to be considered.
% baseline correction example [after averaging]: log10(data./data_baseline)*10
%
% Lawrence December 2019
%% Script
%  check number of inputs
if nargin < 3
    disp('Setting default frequency step to 1.');
    disp('Setting default fwhm_nyclces to 2.');
    freqstep = 1;
    fwhm_ncycles = 2;
end
% setup parameters
if ~isempty(fwhm_ncycles)
    fwhm = (1000./(freqrange(1):1:freqrange(2)).*fwhm_ncycles)/1000;
else
    fwhm = linspace(1,.2,freqrange(2)-freqrange(1)+1);
end
frex = freqrange(1):freqstep:freqrange(2);
nfrex = length(frex);
wavet = -5:1/256:5;
halfw = floor(length(wavet)/2)+1;

for ichan = 1:size(Data,1)
    ichandata = squeeze(Data(ichan,:,:));
    % run wavelet
    npoints = size(ichandata,1); ntrials = size(ichandata,2);
    nConv = size(ichandata,1)*size(ichandata,2) + length(wavet) - 1;
    dataX = fft(reshape(ichandata,1,[]),nConv);
    parfor fi = 1:nfrex
        % create wavelet
        waveX = fft( exp(2*1i*pi*frex(fi)*wavet).*exp(-(4*log(2)*wavet).^2/fwhm(fi).^2),nConv );
        waveX = waveX./max(waveX); % normalize
        
        % convolve
        as = ifft( waveX.*dataX );
        
        % trim and reshape
        as = reshape(as(halfw:end-halfw+1),npoints,ntrials);
        timefreqdata(ichan,:,fi,:) = abs(as).^2;
    end
end
end
