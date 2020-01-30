function eucDistance = rda_cvEuc(iidata,jjdata,nConds)
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
% Cross-validation is computed using leave-one-out. I.e., one euclidian distance 
% is calculated on all but one pair of observations (1 from each condition), and 1
% euclidian distance is calculated from the average of all the other observations.
% and cvEuc is the dot product between the distances across features. 
%
% Lawrence December 2019

if nargin<3
    nConds = jjdata;
    for iCond = 1:nConds
        for jCond = iCond:nConds
            iData = squeeze(iidata(iCond,:,:));
            jData = squeeze(iidata(jCond,:,:));
            ijData = cat(1,iData,jData);
            
            nFold = size(ijData,1)/2;
            % create cross validation & group labels
            cvlabels = [1:nFold,1:nFold];
            grouplabels = [ones(size(ijData, 1)/2,1); ones(size(ijData, 1)/2,1).*2];
           
            traindata = ijData(:,:); testdata = ijData(:,:);
            for icv = 1:nFold
                trainfold = traindata(cvlabels~=icv, :);
                testfold = testdata(cvlabels==icv, :);
                trainlabel = grouplabels(cvlabels~=icv);
                testlabel = grouplabels(cvlabels==icv);
                
                % euclidean distance
                dist_train_euc = nanmean(trainfold(trainlabel==1,:),1) - ...
                    nanmean(trainfold(trainlabel==2,:),1);
                
                % validate distance measure on testing data
                dist_test_euc = nanmean(testfold(testlabel==1,:),1) - ...
                    nanmean(testfold(testlabel==2,:),1);
                
                accuracies(icv) = dot(dist_train_euc,dist_test_euc);
            end
            eucDistance(iCond,jCond) = nanmean(accuracies);
        end
    end
else
    for iCond = 1:nConds
        for jCond = iCond:nConds
            iData = cat(1,squeeze(iidata(iCond,:,:)),squeeze(iidata(jCond,:,:)));
            jData = cat(1,squeeze(jjdata(iCond,:,:)),squeeze(jjdata(jCond,:,:)));
            
            nFold = size(iData,1)/2;
            % create cross validation & group labels
            cvlabels = [1:nFold,1:nFold];
            grouplabels = [ones(size(iData, 1)/2,1); ones(size(iData, 1)/2,1).*2];
            
            data1 = iData(:,:); data2 = jData(:,:);
            for icv = 1:nFold
                trainfold = data1(cvlabels~=icv, :);
                testfold = data2(cvlabels==icv, :);
                trainlabel = grouplabels(cvlabels~=icv);
                testlabel = grouplabels(cvlabels==icv);
                
                % euclidean distance
                dist_train_euc = nanmean(trainfold(trainlabel==1,:),1) - ...
                    nanmean(trainfold(trainlabel==2,:),1);
                
                % validate distance measure on testing data
                dist_test_euc = nanmean(testfold(testlabel==1,:),1) - ...
                    nanmean(testfold(testlabel==2,:),1);
                
                accuracies_1(icv) = dot(dist_train_euc,dist_test_euc);
                
                trainfold = data2(cvlabels~=icv, :);
                testfold = data1(cvlabels==icv, :);
                trainlabel = grouplabels(cvlabels~=icv);
                testlabel = grouplabels(cvlabels==icv);
                
                % euclidean distance
                dist_train_euc = nanmean(trainfold(trainlabel==1,:),1) - ...
                    nanmean(trainfold(trainlabel==2,:),1);
                
                % validate distance measure on testing data
                dist_test_euc = nanmean(testfold(testlabel==1,:),1) - ...
                    nanmean(testfold(testlabel==2,:),1);
                
                accuracies_2(icv) = dot(dist_train_euc,dist_test_euc);
            end
            eucDistance(iCond,jCond) = nanmean((accuracies_1 + accuracies_2)./2);
        end
    end
end
end
