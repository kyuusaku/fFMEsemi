function test_parse_inputs(dataset,o,varargin)

p = parse_inputs();
parse(p,dataset,o,varargin{:});
disp(['dataset:' p.Results.dataset]);
disp(['o:' num2str(p.Results.o)]);
disp(['system:' p.Results.system]);
disp(['parfor:' num2str(p.Results.parfor)]);
disp(['parforNumber:' num2str(p.Results.parforNumber)]);