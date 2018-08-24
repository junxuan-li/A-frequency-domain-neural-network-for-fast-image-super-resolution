function net = cnn_init(varargin)
opts.batchNormalization = true ;
opts.networkType = 'dagnn' ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ;
opts = vl_argparse(opts, varargin) ;

rng('default');
rng(0) ;

f=1/1000 ;
net = dagnn.DagNN() ;

lastAdded.var = 'input' ;
maxnumlayer = 6;
gua_size = 5;
defaultdepth = 9;
lr_order = 5;
batchSize = 35 ;

lastAdded.depth = defaultdepth ;


function fftConv(name, ksize, depth, varargin)
% Helper function to add a Convolutional + BatchNorm + ReLU
% sequence to the network.
  args.gua_conv = true ;
  args.bias = false ;
  args.batchNorm = false;
  args = vl_argparse(args, varargin) ;
  
  stride = 1 ;
  
  if args.bias, pars = {[name '_fft'], [name '_b']} ; else pars = {[name '_fft']} ; end
  
  % TODO 
  % depth should be in poiscale
  net.addLayer([name  '_fftconv'], ...
               dagnn.poiscale('wsize', [360 480 lastAdded.depth],'hasBias', args.bias), ...
               lastAdded.var, ...
               [name '_fftconv'], ...
               pars) ;
           
           
  
  lastAdded.var = [name '_fftconv'] ;

  
    if args.gua_conv
    net.addLayer([name '_guaconv'] , ...
                 dagnn.Conv('size', [ksize ksize lastAdded.depth 1], ...
                          'stride', stride, ....
                          'pad', (ksize - 1) / 2, ...
                          'hasBias', false, ...
                          'opts', {'cudnnworkspacelimit', opts.cudnnWorkspaceLimit}), ...
                 lastAdded.var, ...
                 [name '_guaconv'], ...
                 [name '_filter']) ;
    lastAdded.var = [name '_guaconv'] ;
    end
    
%     if ~isequal(name,'prediction') && args.batchNorm
%     net.addLayer([name '_bn'], ...
%                dagnn.BatchNorm('numChannels', depth, 'epsilon', 1e-5), ...
%                [name '_fftconv'], ...
%                [name '_bn'], ...
%                {[name '_bn_w'], [name '_bn_b'], [name '_bn_m']}) ;
% 
%     lastAdded.var = [name '_bn'] ;
%     end
  
  lastAdded.depth = depth ;
end

for numlayer = 1:maxnumlayer
fftConv(['conv' num2str(numlayer)], gua_size, defaultdepth, ...
     'gua_conv', true, ...
     'bias', true,...
     'batchNorm', false) ;
end


net.addLayer(['sum_' 'conv'],...
            dagnn.Sum(),...
            {['conv1' '_guaconv'],...
             ['conv2' '_guaconv'],...
             ['conv3' '_guaconv'],...
             ['conv4' '_guaconv'],...
             ['conv5' '_guaconv'],...
             ['conv6' '_guaconv']},...
            ['sum']);
lastAdded.var = ['sum'];

fftConv('prediction', 1, 1, ...
     'gua_conv', true, ...
     'bias', true) ;

net.addLayer('loss', ...
             dagnn.myloss('loss','l2') ,...
             {lastAdded.var, 'label'}, ...
             'objective') ; 

% training parameters
net.meta.trainOpts.numEpochs = 200 ;
net.meta.trainOpts.learningRate = logspace(-lr_order,-lr_order-1, net.meta.trainOpts.numEpochs);
net.meta.trainOpts.batchSize = batchSize ;






% initialize network parameters
net.initParams() ;
% for l = 1:numel(net.layers)
%   if isa(net.layers(l).block, 'dagnn.constconv')
%     pi = net.layers(l).paramIndexes;
%     net.params(pi).learningRate = 0 ;
%     net.params(pi).weightDecay = 0 ;
%   end
% end
% net.params(1).learningRate = 10;
% net.params(3).learningRate = 8;
% net.params(5).learningRate = 6;
% net.params(7).learningRate = 4;
% net.params(9).learningRate = 2;



end
