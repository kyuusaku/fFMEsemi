% write run time

dataset = 'number';
save_path = 'result/run_time';
record_path = fullfile(save_path, dataset);

load(fullfile(record_path, 'fme_ver.mat'));
load(fullfile(record_path, 'effme.mat'));
load(fullfile(record_path, 'afme.mat'));
load(fullfile(record_path, 'agr.mat'));
load(fullfile(record_path, 'eagr.mat'));
load(fullfile(record_path, 'mmlp.mat'));
load(fullfile(record_path, 'mtc.mat'));
load(fullfile(record_path, 'laprls.mat'));

aFME_time = mean(aFME_time, 2)';
efFME_time = mean(efFME_time, 2)';
LAPRLS_time = mean(LAPRLS_time, 2)';
MTC_time = mean(MTC_time, 2)';
MMLP_time = mean(MMLP_time, 2)';
EAGR_time = mean(EAGR_time, 2)';
AGR_time = mean(AGR_time, 2)';
FME_time_ver = mean(FME_time_ver, 2)';

fileID = fopen(fullfile(record_path, 'number_run_time.txt'),'w');
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'FME', FME_time_ver);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'fFME', efFME_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'aFME', aFME_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'AGR', AGR_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'EAGR', EAGR_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'LAPRLS', LAPRLS_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'MTC', MTC_time);
fprintf(fileID,'%s & %.4f & %.4f & %.4f & %.4f & %.4f \\\\ \n',...
    'MMLP', MMLP_time);
fclose(fileID);