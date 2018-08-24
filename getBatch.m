
% -------------------------------------------------------------------------
function fn = getBatch(opts)
% -------------------------------------------------------------------------
switch lower(opts.networkType)
  case 'simplenn'
    fn = @(x,y) getSimpleNNBatch(x,y) ;
  case 'dagnn'
    bopts = struct('numGpus', numel(opts.train.gpus)) ;
    fn = @(x,y) getDagNNBatch(bopts,x,y) ;
end
end

% -------------------------------------------------------------------------
function [images, labels] = getSimpleNNBatch(imdb, batch)
% -------------------------------------------------------------------------
images = imdb.data(:,:,batch);
labels = imdb.label(:,:,batch) ;

end

% -------------------------------------------------------------------------
% Normal batch
function inputs = getDagNNBatch(opts, imdb, batch)
% -------------------------------------------------------------------------
label = single(imdb.label(:,:,batch));
image = single(imdb.data(:,:,batch));


image = reshape(image, [size(image,1) size(image,2) 1 numel(batch)]);
label = reshape(label, [size(label,1) size(label,2) 1 numel(batch)]);
res = gather(label);


inputs = {'input', gpuArray(image),  'label', res} ;

end

% -------------------------------------------------------------------------
% 3D FFT batch
% function inputs = getDagNNBatch(opts, imdb, batch)
% % -------------------------------------------------------------------------
% label = single(imdb.label(:,:,:,batch));
% image = single(imdb.data(:,:,:,batch));
% 
% res = gather(label);
% 
% 
% inputs = {'input', gpuArray(image),  'label', res} ;
% end