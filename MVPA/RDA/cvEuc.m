function eucDistance = cvEuc(iidata,jjdata,nConds)
%% Description
% RSADistance(data,nConds)
% Data is a matrix of data for a specific timepoint, nConds = number of
% conditions
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
