%% Description h = perm_cluster_onesample_NH(data,thresh,nperms,tail)
% returns the cluster corrected outcomes
% timeseries data input: nSubs x nTimes
% this follows Nichols & Holmes 2002 approach
% cluster-inducing threshold:
%     a t value at time i is considered significant if it's greater than (1-thresh; typically 95%) of all
%     sign-based permutation produced t-values.
%     a cluster of t-values is considered significant if that cluster is
%     greater than 95% of all max clusters obtained from the sign-based
%     permutation tests.
function [h_corrected, crit_h_size, h_true] = perm_1dcluster_onesample_NH(data, thresh, nperms, tail)
rng('Shuffle');
if tail == -1 % 'left-tailed' tests have not been coded in yet.
    ttail = 'left';
elseif tail == 0
    ttail = 'both';
elseif tail == 1
    ttail = 'right';
end

nsubs = size(data,1);
ntimes = size(data,2);
% true t-test outcome
[~, ~, ~, stats] = ttest(data, zeros(size(data)),'tail', ttail);
t_true = stats.tstat;

% permutations
clustersize = zeros(1,nperms);
parfor iperm = 1:nperms
    permdata = data;
    
    % sign based permutation
    selSample = randsample(nsubs, ceil(nsubs/2));
    permdata(selSample,:) = permdata(selSample,:).*-1;
    % t-test
    [~,~,~,stats] = ttest(permdata, zeros(size(data)), 'tail', ttail);
    t(iperm,:) = stats.tstat;
end
if tail==1 
    t_threshold = prctile(t,(1-thresh)*100);
    h = (t>t_threshold | t==t_threshold); % obtain sign-perm h_values for cluster thresholding
    h_true = t_true > t_threshold | t_true == t_threshold;
elseif tail == 0
    t_threshold = prctile(abs(t),(1-thresh)*100);
    h = (abs(t)>=t_threshold);
    h_true = (abs(t_true)>=t_threshold);
end
parfor iperm = 1:nperms
    % find cluster
    ih = [0,h(iperm,:),0];
    findstart = find(diff(ih)==1);
    findend = find(diff(ih)==-1)-1;
    
    if isempty(findstart)
        continue
    else
        clustersize(iperm) = max(findend-findstart);
    end
end

crit_h_size = prctile(clustersize,95);

% find clusters of ones in true_h that meet the 95% largest cluster
% criterion
h = [0 h_true 0];
findstart = find(diff(h)==1);
findend = find(diff(h)==-1)-1;
clusters = findend-findstart;

nclusters = length(clusters); c = 0;
for i = 1:nclusters
    if clusters(i) > crit_h_size || clusters(i) == crit_h_size
        c = c+1;
        cluster_start(c) = findstart(i);
        cluster_end(c) = findend(i);
    end
end
h_corrected = zeros(1,ntimes);
for i = 1:c
    h_corrected(cluster_start(i):cluster_end(i)) = 1;
end
