function timefreqdata = timefreq_fft(Data,timepoints,freqrange,SampRate,timebinsize,glidesize)
%% Description
% timefreq_fft(Data,timepoints,freqrange,SampRate,timebinsize,glidesize)
% Performs FFT on input data
% Input:
%       Data: nChannels x nTimes x nEpochs 
%       timepoints: 1 x nTimes vector specifying time indices (nTimes size
%                   should match nTimes size in Data)
%       freqrange: lower and upper bound 
%       SampRate: sampling rate (default: 256)
%       timebinsize: size of each time bin (default: 200ms)
%       glidesize: gliding window size (default: 10ms)
%
% Output:
%       timefreqdata: nChannels x nTimes x nFrequencies x nEpochs
%
% IMPORTANT: Note that Additional processing such as averaging, multivaraite
% noise normalisation, and baseline corrections still need to be considered.
% baseline correction example [after averaging]: log10(data./data_baseline)*10
%
if nargin < 3
    disp('Setting sampling rate to 256 Hz. Change if this is not correct!');
    disp('Setting timebin size to 200 ms');
    disp('Setting gliding window size to 10 ms');
    SampRate = 256;
    timebinsize = 200;
    glidesize = 10;
end
minTime = min(timepoints); maxTime = max(timepoints);
NFFT = SampRate; % n-point fft <-- fft will zero-pad to match the number of points, this makes frequency resolution consistent (in this case, 1Hz).
f = SampRate/2*linspace(0,1,NFFT/2+1);
SelFrqIdx = f>=freqrange(1) & f<=freqrange(2); % find frequency band indices based on freqrange (e.g., freqrange=[8 12] for alphaband);
for ichan = 1:size(Data,1)
    timebins = minTime:glidesize:(maxTime-timebinsize/2);
    ichandata = squeeze(Data(ichan,:,:));
    % fft for main timeperiod of interest
    parfor i = 1:length(timebins)
        itimebin = timebins(i);
        % obtain time info and data for timebin
        timerange = [(itimebin-timebinsize/2) (itimebin+timebinsize/2)];
        timeIdx = MyInfo.timepoints>timerange(1) & MyInfo.timepoints<timerange(2);
        timeData = ichandata(timeIdx,:);
        % fft setup
        L = size(timeData,1);
        hwin = hann(L); hwin = repmat(hwin,[1,size(ichandata,2)]);
        y = fft(hwin.*timeData,NFFT,1)/L; % conduct FFT across rows (time dimension) and normalize by data length.
        y = y.*conj(y); %abs(y).^2;
        Abspower =y(1:NFFT/2+1,:); % Obtain power & drop mirror image
        timefreqdata(ichan,i,:,:) = Abspower(SelFrqIdx,:);
    end
end
end