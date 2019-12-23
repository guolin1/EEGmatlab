function timefreq_wavelet(selData,freqrange,fwhm_ncycles)
%%
        % setup some parameters
        if ~isempty(fwhm_ncycles)
            fwhm = (1000./(freqrange(1):1:freqrange(2)).*fwhm_ncycles)/1000;
        else
            fwhm = linspace(1,.2,freqrange(2)-freqrange(1)+1);
        end
        frex = freqrange(1):1:freqrange(2);
        nfrex = length(frex);
        wavet = -5:1/256:5;
        halfw = floor(length(wavet)/2)+1;
        
        for ichan = 1:size(selData,1)
            ichandata = squeeze(selData(ichan,:,:));
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
                
                % trim and reshape2 to 40 Hz in
                as = reshape(as(halfw:end-halfw+1),npoints,ntrials);
                timefreqdata(ichan,:,fi,:) = abs(as).^2;
                %                                         % power
                %                                         p = mean( abs(as).^2 ,2);
                %                                         tf(fi,:,4) = 10*log10( p(tidx)/mean(p(bidx(1):bidx(2))) );
            end
        end
    end
