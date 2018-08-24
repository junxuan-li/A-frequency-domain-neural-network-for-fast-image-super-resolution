varargin = {};
run(fullfile(fileparts(mfilename('fullpath')),...
   'matlab', 'vl_setupnn.m')) ;

opts.batchNormalization = false ;
opts.network = [] ;
opts.networkType = 'dagnn' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.networkType ;

opts.expDir = fullfile(vl_rootnn, 'data') ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.dataDir = fullfile(vl_rootnn, 'data') ;

opts.imdbPath = fullfile(opts.expDir, 'VOC2012_360_480_5000_bic_2.mat');

opts.train = struct();
opts = vl_argparse(opts, varargin) ;
opts.train.gpus = [1]; 
opts.train.derOutputs = {'objective',1};

opts.train.continue = true ;
opts.train.solver = [] ;
% --------------------------------------------------------------------
%                                                         Prepare data
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', initNet = @cnn_flow_init ;
  case 'dagnn' 
%     initNet = @cnn_init ;
%     initNet = @cnn_init_lineKernel ;
%     initNet = @cnn_init_fullconv ;
%     initNet = @cnn_init_3dtensor ;
    initNet = @cnn_init_equalnet ;
%     initNet = @cnn_init_lineKernel_sp ;
%     initNet = @cnn_init_fullconv_sp ;
end

if isempty(opts.network)
  net = initNet('batchNormalization', opts.batchNormalization, ...
    'networkType', opts.networkType) ;
else
  net = opts.network ;
  opts.network = [] ;
end
net.conserveMemory = true ;

fprintf('read data\n');
% load(opts.imdbPath);



net.meta.classes.name = arrayfun(@(x)sprintf('%d',x),1:10,'UniformOutput',false) ;
% Meta parameters
images_size = size(imdb.data);
opts.inputSize = images_size(1:2) ;
net.meta.inputSize = images_size(1:2) ;

% --------------------------------------------------------------------
%                                                                Train
% --------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainfn = @cnn_train ;
  case 'dagnn'
      trainfn = @cnn_train_dag ;
end

[net, info] = trainfn(net, imdb, getBatch(opts), ...
  'expDir', opts.expDir, ...
  net.meta.trainOpts, ...
  opts.train, ...
  'train', 1:floor(images_size(end)*9/10), ...
  'val', floor(images_size(end)*9/10)+1:images_size(end)) ;
