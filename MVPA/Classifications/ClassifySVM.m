function [accuracies] = ClassifySVM(data1, data2, preprocessoption, svmoption, permoption)
%% Description
% [accuracies] = ClassifySVM(data1, data2, preprocessoption, svmoption, permoption);
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
% 
% preprocessoption [standardize(1/0), windsorize(1/0), normalize(1/0)]
%             a vector of 3 values to enable / disable standardization,
%               windsorization, and normalization. Standardization and
%               normalization of both training and testing sets are
%               computed using the parameters (mean, std, max and min) of 
%               the training set. 
%               Default: [1 1 1];
%
% if svmoption = 1, liblinear is used intead of libsvm 
%			  liblinear apparently works better when number of feature is
%			  extremely large. By default, ClassifySVM uses libSVM.
%               Default: [0]
%
% if permoption = 1, permutation is enabled and classification is based on
%               shuffled labels. 
%               Default: [0]
%
% data format [IMPORTANT]: 
% data structure:	[rows] Observations x  [columns] Features
% 				top half rows = category 1; bottom half rows = category 2  
% 				number of observations across categories must be equal
%				Optional: time can be included as the third dimension. 
%						However, if parpool is used, it's better to
%						use parfor on the time dimension outside of 
%						this function.
%
% Lawrence December 2019
%% check input
if nargin < 3
    svmoption = 0; %libsvm
    preprocessoption = [1 1 1];
    permoption = 0;
end

nFold = size(data1,1)/2;
%% create cross validation & group labels
cvlabels = [1:nFold,1:nFold];
grouplabels = [ones(size(data1, 1)/2,1); ones(size(data1, 1)/2,1).*2];
% permute grouplabels if permutation is enabled.
if permoption
    rng('shuffle');
    grouplabels = grouplabels(randperm(length(grouplabels)));
end

%% loop through nTimes
nTimes = size(data1,3);
accuracies = NaN(1,nTimes);
for itime = 1:nTimes
    %disp(itime);
    traindata = data1(:,:,itime); testdata = data2(:,:,itime);
    for icv = 1:nFold
        trainfold = traindata(cvlabels~=icv, :);
        testfold = testdata(cvlabels==icv, :);
	% Feature preprocessing
        if preprocessoption(1)==1 % standardization
            % Compute mean & standard deviation from the training data [at each feature separately]
            meanvec = nanmean(trainfold); stdvec = nanstd(trainfold);
	    % Standardizes testing data & training data using the same mean & std.
	    trainfold = (trainfold-meanvec)./stdvec; 
	    testfold = (testfold-meanvec)./stdvec; 
        end
        if preprocessoption(2) == 1 % windsorization
	    %threshold all data at +/- 3 to curb the impact of outliers on SVM performance
            trainfold(trainfold>3) = 3; testfold(testfold>3) = 3;
            trainfold(trainfold<-3) = -3; testfold(testfold<-3) = -3;
        end
        if preprocessoption(3) == 1 % normalization
            % Normalizes between 0 and 1 according to the training data
            trainfold = (trainfold-min(trainfold))./(max(trainfold)-min(trainfold));
            testfold = (testfold-min(trainfold))./(max(trainfold)-min(trainfold));
        end
	% Labels are automatically computed given that data structure follows the requirement stated under description.
        trainlabel = grouplabels(cvlabels~=icv);
        testlabel = grouplabels(cvlabels==icv);
        if svmoption %liblinear
            disp(icv);
            model = train(trainlabel, sparse(trainfold), '-q -c 1');
            [predictedlabels, ~, ~] = predict(testlabel, sparse(testfold), model, '-q');
        else
            model = svmtrain(trainlabel, double(trainfold), sprintf('-q -t 0 -c %f', 1));
            [predictedlabels, ~, ~] = svmpredict(testlabel, double(testfold), model, '-q'); %third output from svmpredict is decVal, used for calculation of DV (see commented out section below).
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
