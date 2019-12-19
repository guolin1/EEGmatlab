function eucDistance = cvEuc_pairwise(iidata,jjdata,nConds)
%% Description
% cvEuc(iidata,nConds) for regular RDA
% cvEuc(iidata,jjdata,nConds) for temporal generalization, where jjdata can be ERPs from a different timepoint
% Data is a numConditions x numObservationsx numFeatures array 
% for a specific timepoint, nConds = number of conditions
% cvEuc computes cross-validated euclidian distance (cvEuc), thus cvEuc is unbiased
% where the expected distance is 0 if H0 is true, and expected distance ~= 0 if 
% H0 is false. distance output can be negative given the nature of how it is 
% calculated, which computes the covariance between two independent distance
% calculations. 
% Cross-validation is computed using all pairwise observations across two conditions.
% Equivalient to nChoose2 where n = numObservations per condition.

if nargin<3
    nConds = jjdata;
    for iCond = 1:nConds
        for jCond = iCond:nConds
            iData = squeeze(iidata(iCond,:,:));
            jData = squeeze(iidata(jCond,:,:));
            nFold = size(iData,1);
            dists = zeros(nFold,nFold);
            for icv = 1:nFold-1
                for jcv = icv+1:nFold
                    dist_train = iData(icv, :) - jData(icv,:);
                    dist_test = iData(jcv, :) - jData(jcv,:);
                    % cross validate on different folds
                    dists(icv,jcv) = dot(dist_train,dist_test);
                end
            end
            dists = (dists + dists');
            eucDistance(iCond,jCond) = sum(dists(:))./(nFold*nFold-nFold);
        end
    end
else
    for iCond = 1:nConds
        % needs to be 1:nConds rather than iCond:nConds because these are 
        % different data (i.e., iidata vs jjdata), and so unlike the RSA 
        % above, here, iidata(iCond==2) vs jjdata(iCond==3) is not the same 
        % as iidata(iCond==3) vs jjdata(iCond==2). 
        for jCond = 1:nConds 
            iData = squeeze(iidata(iCond,:,:));
            jData = squeeze(jjdata(jCond,:,:));
            
            nFold = size(iData,1);
            dists = zeros(nFold,nFold);
            for icv = 1:nFold
                for jcv = 1:nFold
                   dist_train = iData(icv,:) - jData(icv,:);
                   dist_test = iData(jcv,:) - jData(jcv,:);
                   % cross validate on different folds
                   dists(icv,jcv) = dot(dist_train,dist_test);
                end
            end
            eucDistance(iCond,jCond) = nanmean(dists(eye(nFold,nFold)==0));
        end
    end
end
end
