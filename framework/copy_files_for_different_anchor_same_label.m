function copy_files_for_different_anchor_same_label(dataset,o,varargin)

% parse inputs
p = parse_inputs();
parse(p,dataset,o,varargin{:});

source_path = ['result/' p.Results.dataset '/semi-a' num2str(1000)];
if ~exist(source_path, 'dir')
    error('Cannot found source path');
end
source_record_path = fullfile(source_path, ['record-' num2str(p.Results.o)]);
if ~exist(source_record_path, 'dir')
    error('Cannot found source record path');
end

save_path = ['result/' p.Results.dataset '/semi-a' num2str(p.Results.anchorNumber)];
if ~exist(save_path, 'dir')
    mkdir(save_path);
end
record_path = fullfile(save_path, ['record-' num2str(p.Results.o)]);
if ~exist(record_path, 'dir')
    mkdir(record_path);
end

source_pca_data = fullfile(source_path, 'pca.mat');
pca_data = fullfile(save_path, 'pca.mat');
copyfile(source_pca_data, pca_data);

source_label_data = fullfile(source_record_path, 'label.mat');
label_data = fullfile(record_path, 'label.mat');
copyfile(source_label_data, label_data);

source_emin_data = fullfile(source_path, 'E_min.mat');
emin_data = fullfile(save_path, 'E_min.mat');
copyfile(source_emin_data, emin_data);

source_emax_data = fullfile(source_path, 'E_max.mat');
emax_data = fullfile(save_path, 'E_max.mat');
copyfile(source_emax_data, emax_data);

source_mmlp_data_para = fullfile(source_record_path, 'result_MMLP_min_para.mat');
mmlp_data_para = fullfile(record_path, 'result_MMLP_min_para.mat');
copyfile(source_mmlp_data_para, mmlp_data_para);

source_mmlp_data_para = fullfile(source_record_path, 'result_MMLP_max_para.mat');
mmlp_data_para = fullfile(record_path, 'result_MMLP_max_para.mat');
copyfile(source_mmlp_data_para, mmlp_data_para);

source_mtc_data_para = fullfile(source_record_path, 'result_MTC_para.mat');
mtc_data_para = fullfile(record_path, 'result_MTC_para.mat');
copyfile(source_mtc_data_para, mtc_data_para);

source_nn_data_para = fullfile(source_record_path, 'result_NN_para.mat');
nn_data_para = fullfile(record_path, 'result_NN_para.mat');
copyfile(source_nn_data_para, nn_data_para);

source_laprls_data2_para_best = fullfile(source_record_path, 'result_LapRLS2_para_best.mat');
laprls_data2_para_best = fullfile(record_path, 'result_LapRLS2_para_best.mat');
copyfile(source_laprls_data2_para_best, laprls_data2_para_best);
