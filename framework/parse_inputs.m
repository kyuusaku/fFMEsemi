function p = parse_inputs()

p = inputParser;

validDataset = {'norb','aloi','rcv1','mnist630k','cifar10-rgb','covtype', ...
    'usps-large','usps-large-imbalance','mnist-large','mnist-large-imbalance'};
checkDataset = @(x) any(validatestring(x,validDataset));

defaultSystem = 'linux';
validSystem = {'linux','win'};
checkSystem = @(x) any(validatestring(x,validSystem));

defaultParfor = false;
defaultParforNumber = 8;

addRequired(p,'dataset',checkDataset);
addRequired(p,'o',@isnumeric);

addOptional(p,'system',defaultSystem,checkSystem);

addParameter(p,'parfor',defaultParfor,@islogical);
addParameter(p,'parforNumber',defaultParforNumber,@isnumeric);
