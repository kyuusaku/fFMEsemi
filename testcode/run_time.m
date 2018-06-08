function run_time(type, num_samples, num_features)
% estimate running time 
% with different number of training samples or features

%% env
close all;
% clear;
clc;
warning off all;
addpath(genpath('./baselines'));
addpath(genpath('./mmlp'));
addpath('./flann-linux');
% addpath('./flann-mac');
addpath('./framework');
addpath('./fFME');

%%
dataset = type;
save_path = 'result/run_time';
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, dataset);
if ~exist(record_path, 'dir')
    mkdir(record_path);
end

diary(fullfile(record_path, 'log.txt'));

%% common parameters
para.num_classes = 10;
para.iter = 20;
para.type = 'equal';
para.p = 10;% number of labeled data for each class
para.s = 3; % anchor
para.cn = 10;
para.num_anchor = 1000;
para.beta = 1;
para.knn = 3;
save(fullfile(record_path, 'para.mat'), 'para', 'num_samples', 'num_features');
disp(para);

if strcmp(type, 'samples')
    assert(numel(num_features)==1);
    nums = num_samples;
elseif strcmp(type, 'features')
    assert(numel(num_samples)==1);
    nums = num_features;
else
    error('Unknow experimental type of runtime');
end
num = numel(nums);

%% generate samples
samples_file = fullfile(record_path, 'samples.mat');
if ~exist(samples_file, 'file')
    if strcmp(type, 'samples')
        max_num_samples = max(num_samples(:));
        [fea, gnd] = make_classification(max_num_samples, num_features, para.num_classes);
        fea = fea';
        class = unique(gnd)';
        assert(sum(class == 1:numel(class)) == numel(class)); % check gnd
        samples = cell(numel(num_samples), 2);
        labels = cell(numel(num_samples), 1);
        for i = 1 : numel(num_samples)
            sample_inds = randsample(max_num_samples, num_samples(i));
            samples{i,1} = fea(:, sample_inds);
            samples{i,2} = gnd(sample_inds);
            l_tmp = generate_label(samples{i,2}, para);
            labels{i} = l_tmp;
        end
        clear fea gnd;
        save(samples_file, 'samples', 'labels', 'class');
    elseif strcmp(type, 'features')
        samples = cell(numel(num_features), 2);
        labels = cell(numel(num_features), 1);
        for i = 1 : numel(num_features)
            [fea, gnd] = make_classification(num_samples, num_features(i), para.num_classes);
            class = unique(gnd)';
            assert(sum(class == 1:numel(class)) == numel(class)); % check gnd
            samples{i,1} = fea';
            samples{i,2} = gnd;
            l_tmp = generate_label(gnd, para);
            labels{i} = l_tmp;
        end
        clear fea gnd;
        save(samples_file, 'samples', 'labels', 'class');
    else
        error('Unknow experimental type of runtime');
    end        
else
    load(samples_file);
end

%% anchor graph
ags_data = fullfile(record_path, 'ags.mat');
if ~exist(ags_data, 'file')
    ags = cell(num, 5);
    for i = 1 : num
        X_tmp = samples{i,1};

        [~, anchor, kmeans_time] = k_means(X_tmp, para.num_anchor);
        [B, rL, ag_time] = flann_AnchorGraph(X_tmp, anchor, para.s, 1, para.cn);

        ags{i,1} = B;
        ags{i,2} = rL;
        ags{i,3} = ag_time;
        ags{i,4} = kmeans_time;
        ags{i,5} = anchor;

        fprintf('AG: num=%d, kmeans_time=%f, ag_time=%f\n', ...
                nums(i), kmeans_time, ag_time);
    end
    save(ags_data, 'ags');
else
    load(ags_data);
end

%% AGR
agr_data = fullfile(record_path, 'agr.mat');
if ~exist(agr_data, 'file')
    AGR_time = zeros(num, para.iter);
    for i = 1 : num
        Y_tmp = samples{i,2};
        B_tmp = ags{i,1};
        rL_tmp = ags{i,2};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter
            label_ind = find(l_tmp(:,t));
            [~, ~, e, elapsed_time] = AnchorGraphReg(B_tmp, rL_tmp, Y_tmp', label_ind, 0.01);
            AGR_time(i, t) = elapsed_time;
            fprintf('AGR: num=%d, t=%d, time=%f\n', ...
                nums(i), t, AGR_time(i, t));
        end
    end
    save(agr_data, 'AGR_time');
else
    load(agr_data);
end

%% fFME
fFME_data = fullfile(record_path, 'ffme.mat');
if ~exist(fFME_data, 'file')
    p.ul = 1e9;
    p.uu = 0;
    p.mu = 1e-9;
    p.gamma = 1e-9;
    fFME_time = zeros(num, para.iter);
    for i = 1 : num
        X_tmp = samples{i,1};
        Y_tmp = samples{i,2};
        B_tmp = ags{i,1};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter            
            label_ind = find(l_tmp(:,t));
            tic;
            Y = zeros(size(X_tmp, 2), para.num_classes);
            for cc = 1 : para.num_classes
                cc_ind = find(Y_tmp(label_ind) == class(cc));
                Y(label_ind(cc_ind),cc) = 1;
            end
            Y = sparse(Y);
            [~, ~, ~] = fastFME_semi(X_tmp, B_tmp, Y, p, true);
            fFME_time(i, t) = toc;
            fprintf('fFME: num=%d, t=%d, time=%f\n', ...
                nums(i), t, fFME_time(i, t));
        end
    end
    save(fFME_data, 'fFME_time');
else
    load(fFME_data);
end

%% efficient anchor graph
eags_data = fullfile(record_path, 'eags.mat');
if ~exist(eags_data, 'file')
    eags = cell(num, 5);
    for i = 1 : num
        X_tmp = samples{i,1};

        anchor = ags{i,5};
        kmeans_time = ags{i,4};

        tic;
        [Z] = FLAE(anchor', X_tmp', para.knn, para.beta);    
        W=Z'*Z; % Normalized graph Laplacian
        Dt=diag(sum(W).^(-1/2));
        S=Dt*W*Dt;
        rLz=eye(para.num_anchor,para.num_anchor)-S; 
        eag_time = toc;

        eags{i,1} = Z;
        eags{i,2} = rLz;
        eags{i,3} = eag_time;
        eags{i,4} = kmeans_time;
        eags{i,5} = anchor;

        fprintf('EAG: num=%d, kmeans_time=%f, eag_time=%f\n', ...
                nums(i), kmeans_time, eag_time);
    end
    save(eags_data, 'eags');
else
    load(eags_data);
end

clear ags;

%%
eagr_data = fullfile(record_path, 'eagr.mat');
if ~exist(eagr_data, 'file')
    EAGR_time = zeros(num, para.iter);
    for i = 1 : num
        Y_tmp = samples{i,2};
        Z_tmp = eags{i,1};
        rLz_tmp = eags{i,2};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter
            label_ind = find(l_tmp(:,t));
            [~, ~, elapsed_time] = EAGReg(Z_tmp, rLz_tmp, Y_tmp', label_ind, 1);
            EAGR_time(i, t) = elapsed_time;
            fprintf('EAGR: num=%d, t=%d, time=%f\n', ...
                nums(i), t, EAGR_time(i, t));
        end
    end
    save(eagr_data, 'EAGR_time');
else
    load(eagr_data);
end

%% efFME
% efFME_data = fullfile(record_path, 'effme.mat');
% if ~exist(efFME_data, 'file')
%     p.ul = 1e9;
%     p.uu = 0;
%     p.mu = 1e-9;
%     p.gamma = 1e-9;
%     efFME_time = zeros(num, para.iter);
%     for i = 1 : num
%         X_tmp = samples{i,1};
%         Y_tmp = samples{i,2};
%         Z_tmp = eags{i,1};
%         l_tmp = labels{i}{1};
%         for t = 1 : para.iter            
%             label_ind = find(l_tmp(:,t));
%             tic;
%             Y = zeros(size(X_tmp, 2), para.num_classes);
%             for cc = 1 : para.num_classes
%                 cc_ind = find(Y_tmp(label_ind) == class(cc));
%                 Y(label_ind(cc_ind),cc) = 1;
%             end
%             Y = sparse(Y);
%             [~, ~, ~] = fastFME_semi(X_tmp, Z_tmp, Y, p, true);
%             efFME_time(i, t) = toc;
%             fprintf('efFME: num=%d, t=%d, time=%f\n', ...
%                 nums(i), t, efFME_time(i, t));
%         end
%     end
%     save(efFME_data, 'efFME_time');
% else
%     load(efFME_data);
% end

%% r-FME
aFME_data = fullfile(record_path, 'afme.mat');
if ~exist(aFME_data, 'file')
    p.ul = 1e9;
    p.uu = 0;
    p.mu = 1e-9;
    p.gamma = 1e-9;
    aFME_time = zeros(num, para.iter);
    for i = 1 : num
        anchor = eags{i,5};
        Y_tmp = samples{i,2};
        Z_tmp = eags{i,1};
        rLz_tmp = eags{i,2};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter
            label_ind = find(l_tmp(:,t));
            tic;
            Y = zeros(numel(Y_tmp), para.num_classes);
            for cc = 1 : para.num_classes
                cc_ind = find(Y_tmp(label_ind) == class(cc));
                Y(label_ind(cc_ind),cc) = 1;
            end
            Y = sparse(Y);
            [~, ~, ~] = aFME_semi(anchor, Z_tmp, rLz_tmp, Y, p, true);
            aFME_time(i, t) = toc;
            fprintf('aFME: num=%d, t=%d, time=%f\n', ...
                nums(i), t, aFME_time(i, t));
        end
    end
    save(aFME_data, 'aFME_time');
else
    load(aFME_data);
end

clear eags;

%% original laplacian graph
lgs_data = fullfile(record_path, 'lgs.mat');
if ~exist(lgs_data, 'file')
    lgs = cell(num, 12);
    for i = 1 : num
        X_tmp = samples{i,1};
        n = size(X_tmp, 2);

        %tic;S = constructS(X_tmp, 10);s_time=toc;
        [S, s_time] = knn_graph_max(X_tmp, 11);

        s = 1e-5/10;

        tic;
        [ii, jj, ss] = find(S);
        [mm, nn] = size(S);
        s_mean = mean(ss);
        para_t = - full(s_mean) ./ log(s);
        W = sparse(ii, jj, exp(-ss ./ para_t), mm, nn);
        W(isnan(W)) = 0; W(isinf(W)) = 0;
        w_time=toc;

        tic;
        D = spdiags(sum(W, 2), 0, nn, nn);
        L = D - W;
        D(isnan(D)) = 0; D(isinf(D)) = 0;
        L(isnan(L)) = 0; L(isinf(L)) = 0;
        l_time=toc;

        tic;
        alpha = 0.99; % default value
        nD = spdiags(sum(W, 2).^(-0.5), 0, nn, nn);
        nD(isnan(nD)) = 0; nD(isinf(nD)) = 0;
        nL = speye(size(W)) - alpha .* (nD * W * nD);
        nL(isnan(nL)) = 0; nL(isinf(nL)) = 0;
        nl_time=toc;

        [E2, mmlp_gr_time] = knn_graph_min(X_tmp, 11);

        tic;
        [idx1, idx2] = find(W~=0);
        edges = [idx1 idx2 W(idx1+(idx2-1).*n)];
        mtc_gr_time = toc;

        lgs{i,1} = S;
        lgs{i,2} = s_time;
        lgs{i,3} = W;
        lgs{i,4} = w_time;
        lgs{i,5} = L;
        lgs{i,6} = l_time;
        lgs{i,7} = nL;
        lgs{i,8} = nl_time;
        lgs{i,9} = E2;
        lgs{i,10} = mmlp_gr_time;
        lgs{i,11} = edges;
        lgs{i,12} = mtc_gr_time;

        fprintf('LG: num=%d, s_time=%f, w_time=%f, l_time=%f, nl_time=%f, mmlp_gr_time=%f, mtc_gr_time=%f\n', ...
                nums(i), s_time, w_time, l_time, nl_time, mmlp_gr_time, mtc_gr_time);
    end
    save(lgs_data, 'lgs');
else
    load(lgs_data);
end

%% MMLP
mmlp_data = fullfile(record_path, 'mmlp.mat');
if ~exist(mmlp_data, 'file')
    MMLP_time = zeros(num, para.iter);
    for i = 1 : num
        X_tmp = samples{i,1};
        Y_tmp = samples{i,2};
        E_tmp = lgs{i,9};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter
            label_ind = find(l_tmp(:,t));
            [~, e, ~, ~, MMLP_time(i, t), num_iter, num_prop] = mmlp(E_tmp, X_tmp, Y_tmp, label_ind);
            fprintf('MMLP: num=%d, t=%d, time=%f\n', ...
                nums(i), t, MMLP_time(i, t));
            num_iter
            num_prop
            e
        end
    end
    save(mmlp_data, 'MMLP_time');
else
    load(mmlp_data);
end

%% LapRLS\L
laprls_data = fullfile(record_path, 'laprls.mat');
if ~exist(laprls_data, 'file')
    LAPRLS_time = zeros(num, para.iter);
    for i = 1 : num
        gammaA = 1; 
        gammaI = 1;  
        X_tmp = samples{i,1};
        Y_tmp = samples{i,2};
        L_tmp = lgs{i,5};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter
            label_ind = find(l_tmp(:,t));
            tic;
            nFea = size(X_tmp, 1);
            % construct the labeled matrix
            feaLabel = X_tmp(:, label_ind);
            gndLabel = Y_tmp(label_ind);
            classLabel = unique(gndLabel);
            nClass = numel(classLabel);
            nLabel = numel(gndLabel);
            YLabel = zeros(nLabel, nClass);
            for cc = 1 : nClass
                YLabel(gndLabel == classLabel(cc), cc) = 1;
            end
            % compute W
            Xl = bsxfun(@minus, feaLabel, mean(feaLabel,2));
            W = (Xl * Xl' + gammaA * nLabel .* eye(nFea) + ...
                gammaI * nLabel .* (X_tmp * L_tmp * X_tmp')) \ (Xl * YLabel);
            b = 1/nLabel*(sum(YLabel,1)' - W'*(feaLabel*ones(nLabel,1)));

            LAPRLS_time(i, t) = toc;
            fprintf('laprls: num=%d, t=%d, time=%f\n', ...
                nums(i), t, LAPRLS_time(i, t));
        end
    end
    save(laprls_data, 'LAPRLS_time');
else
    load(laprls_data);
end

%% MTC
mtc_data = fullfile(record_path, 'mtc.mat');
if ~exist(mtc_data, 'file')
    MTC_time = zeros(num, para.iter);
    for i = 1 : num
        Y_tmp = samples{i,2};
        e_tmp = lgs{i,11};
        l_tmp = labels{i}{1};
        for t = 1 : para.iter
            label_ind = find(l_tmp(:,t));
            tic;
            % construct Y    
            Y = zeros(numel(Y_tmp), 1)-1;
            Y(label_ind) = Y_tmp(label_ind) - 1;
            % compute F
            F = mtc_matlab(full(e_tmp), numel(Y_tmp), Y, para.num_classes, 0, 1);
            F = F + 1;
            MTC_time(i, t) = toc;
            fprintf('MTC: num=%d, t=%d, time=%f\n', ...
                nums(i), t, MTC_time(i, t));
        end
    end
    clear Y F;
    save(mtc_data, 'MTC_time');
else
    load(mtc_data);
end

    
%%
fme_data = fullfile(record_path, 'fme.mat');
if ~exist(fme_data, 'file')
    FME_time = zeros(num, para.iter);
    try
        p.ul = 1e9;
        p.uu = 0;
        p.mu = 1e-9;
        p.lamda = 1e-9;
        for i = 1 : num
            X_tmp = samples{i,1};
            Y_tmp = samples{i,2};
            L_tmp = lgs{i,5};
            l_tmp = labels{i}{1};
            for t = 1 : para.iter
                label_ind = find(l_tmp(:,t));
                tic;
                Y = zeros(size(X_tmp, 2), para.num_classes);
                for cc = 1 : para.num_classes
                    cc_ind = find(Y_tmp(label_ind) == class(cc));
                    Y(label_ind(cc_ind),cc) = 1;
                end
                Y = sparse(Y);
                [~, ~, ~] = FME_semi(X_tmp, L_tmp, Y, p);
                FME_time(i, t) = toc;
                fprintf('FME: num=%d, t=%d, time=%f\n', ...
                    nums(i), t, FME_time(i, t));
            end
        end
    catch ErrorInfo
        disp(ErrorInfo);
        save(fme_data, 'FME_time');
    end
else
    load(fme_data);
end

%% write result
AGR_time = mean(AGR_time, 2)';
EAGR_time = mean(EAGR_time, 2)';
MTC_time = mean(MTC_time, 2)';
MMLP_time = mean(MMLP_time, 2)';
aFME_time = mean(aFME_time, 2)';
fFME_time = mean(fFME_time, 2)';
LAPRLS_time = mean(LAPRLS_time, 2)';
FME_time = mean(FME_time, 2)';

fileID = fopen(fullfile(record_path, 'run_time.txt'),'w');
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'FME', FME_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'fFME', fFME_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'rFME', aFME_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'AGR', AGR_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'EAGR', EAGR_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'LAPRLS\L', LAPRLS_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'MTC', MTC_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'MMLP', MMLP_time);
fclose(fileID);