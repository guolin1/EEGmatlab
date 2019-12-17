function [accuracies] = ClassifySVM(data1, data2, option)
%% Description
% [accuracies] = ClassifySVM(data1,data2,option);
% Required package: LIBSVM [https://www.csie.ntu.edu.tw/~cjlin/libsvm/]
%
% Takes two data arrays, one for training & one for testing (cross-decode is
% possible) and classifies categories [1 vs 1] using libsvm via leave one out cross-validation
%
% if data1 == data2, then regular decoding
% if data1 ~= data2, then cross decoding [train on data 1, test on data 2]
% 
% ClassifySVM requires libsvm toolbox
%
% Liblinear
% if option = 1, liblinear is used intead of libsvm 
%			  liblinear apparently works better when number of feature is
%			  extremely large. By default, ClassifySVM uses libSVM.
%
% ClassifySVM automatically standardizes and normalizes data separately at
%       each feature. standardization and normalization are done using the 
%       means & stds obtained from the training data, and the same means 
%       and stds are used to standardize the testing data. 
%
% data requirement [IMPORTANT]: 
% data structure:	[rows] Observations x  [columns] numFeatures
% 				top half rows = category 1; bottom half rows = category 2  
% 				number of observations across categories must be equal
%				Optional: time can be included as the third dimension. 
%						However, if parpool is used, it's better to
%						use parfor on the time dimension outside of 
%						this function.

%% check input
if nargin < 3
    option = 0; %libsvm
end

nFold = size(data1,1)/2;
%% create cross validation & group labels
cvlabels = [1:nFold,1:nFold];
grouplabels = [ones(size(data1, 1)/2,1); ones(size(data1, 1)/2,1).*2];

%% loop through nTimes
nTimes = size(data1,3);
accuracies = NaN(1,nTimes);
for itime = 1:nTimes
    %disp(itime);
    traindata = data1(:,:,itime); testdata = data2(:,:,itime);
    for icv = 1:nFold
        trainfold = traindata(cvlabels~=icv, :);
        testfold = testdata(cvlabels==icv, :);
        % normalize trainfold & testfold
		% Compute mean & standard deviation from the training data [at each feature]
        meanvec = nanmean(trainfold); stdvec = nanstd(trainfold); 
		% Standardizes testing data & training data using the same mean & std. Also threshold all data at +/- 3 
		% to curb the impact of outliers on SVM performance
        trainfold = (trainfold-meanvec)./stdvec; trainfold(trainfold>3) = 3; trainfold(trainfold<-3) = -3;
        testfold = (testfold-meanvec)./stdvec; testfold(testfold>3) = 3; testfold(testfold<-3) = -3;
		% Normalizes between 0 and 1 according to the training data
        trainfold = (trainfold-min(trainfold))./(max(trainfold)-min(trainfold));
        testfold = (testfold-min(trainfold))./(max(trainfold)-min(trainfold));
        
		% Labels are automatically computed given that data structure follows the requirement stated under description.
        trainlabel = grouplabels(cvlabels~=icv);
        testlabel = grouplabels(cvlabels==icv);
        if option %liblinear
            disp(icv);
            model = train(trainlabel, sparse(trainfold), '-q -c 1');
            [predictedlabels, ~, ~] = predict(testlabel, sparse(testfold), model, '-q');
        else
            model = svmtrain(trainlabel, double(trainfold), sprintf('-q -t 0 -c %f', 1));
            [predictedlabels, ~, decVal] = svmpredict(testlabel, double(testfold), model, '-q');
            % w = model.SVs' * model.sv_coef; % compute weight vector
            % DV = abs(decVal)./norm(w); % decision value is the distance between test data & decision boundary
        end
        acc(:,icv) = predictedlabels==testlabel;
        %DV_acc(:,icv) = DV.*acc(:,icv); % decision-value weighted accuracies
    end
    accuracies(itime) = mean(mean(acc));
    %DV_accuracies(itime) = mean(mean(DV_acc));
end
end
