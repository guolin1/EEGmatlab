%% Description h = perm_cluster_onesample_NH(data,thresh,nperms)
% returns the cluster corrected outcomes
% data input: nSubs x nDim1 x nDim2
% this follows Nichols & Holmes 2002 approach
% cluster-inducing threshold:
%     a t value at cell ij is considered significant if it's greater than 95% of all
%     sign-based permutation produced t-values.
%     a cluster of t-values is considered significant if that cluster size is
%     greater than 95% of all max cluster size obtained from the sign-based
%     permutation tests.
function [h_corrected, crit_h_size, h_true] = perm_2dcluster_onesample_NH(data, thresh, nperms, tail)
rng('Shuffle');
if tail == -1
    tail = 'left'; % not coded in yet
elseif tail == 0
    tail = 'both'; % not coded in yet
elseif tail == 1
    tail = 'right';
end

nsubs = size(data,1);
nDims1 = size(data,2);
nDims2 = size(data,3);

% true t-test outcome
[~, ~, ~, stats] = ttest(data, zeros(size(data)),'tail', tail);
t_true = stats.tstat;

% permutations
clustersize = zeros(1,nperms);
parfor iperm = 1:nperms
    permdata = data;
    
    % sign based permutation
    selSample = randsample(nsubs, floor(nsubs/2));
    permdata(selSample,:,:) = permdata(selSample,:,:).*-1;
    % t-test
    [~,~,~,stats] = ttest(permdata, zeros(size(data)), 'tail', tail);
    t(iperm,:,:) = stats.tstat;
end
t_threshold = prctile(t,(1-thresh)*100);
h = t>t_threshold | t==t_threshold; % obtain sign-perm h_values for cluster thresholding
h_true = t_true > t_threshold | t_true == t_threshold;

parfor iperm = 1:nperms
    % find largest cluster
    cc = bwconncomp(squeeze(h(iperm,:,:)),4);
    numCells = cellfun(@numel,cc.PixelIdxList); 
    if isempty(numCells)
        numCells = 1;
    end
    clustersize(iperm) = max(numCells);
end

crit_h_size = prctile(clustersize,95);

% find clusters of ones in true_h that meet the 95% largest cluster
% criterion
cc = bwconncomp(squeeze(h_true),4);
numCells = cellfun(@numel,cc.PixelIdxList); 
selClusters = {cc.PixelIdxList{numCells>crit_h_size | numCells == crit_h_size}};
h_corrected = zeros(nDims1,nDims2);
for i = 1:length(selClusters)
    h_corrected(selClusters{i})=1;
end
