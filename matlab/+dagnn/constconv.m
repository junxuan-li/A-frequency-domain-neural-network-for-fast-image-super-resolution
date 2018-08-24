classdef constconv < dagnn.Filter
  properties
    size = [0 0 0 0]
    hasBias = true
    opts = {'cuDNN'}
    sigma = 10;
    k_f = 1;
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
      outputs{1} = vl_nnconv(...
        inputs{1}, params{1}, params{2}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      [derInputs{1}, derParams{1}, derParams{2}] = vl_nnconv(...
        inputs{1}, params{1}, params{2}, derOutputs{1}, ...
        'pad', obj.pad, ...
        'stride', obj.stride, ...
        'dilate', obj.dilate, ...
        obj.opts{:}) ;
%     
%       derParams{1} = 0;
%       derParams{2} = 0;
    end

    function kernelSize = getKernelSize(obj)
      kernelSize = obj.size(1:2) ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes = getOutputSizes@dagnn.Filter(obj, inputSizes) ;
      outputSizes{1}(3) = obj.size(4) ;
    end

    function params = initParams(obj)
      % Xavier improved
      %sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
%       if obj.size(1)==1  % prediction layer
%           params{1} = single(ones(obj.size(1),obj.size(2),obj.size(3),obj.size(4)));
%       else
%           params{1} = single(zeros(obj.size(1),obj.size(2),obj.size(3),obj.size(4)));
%           for i = 1:obj.size(3)
%             params{1}(:,:,i,1) = single(fspecial('gaussian', obj.size(1), obj.sigma) );
%           end
%       end
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      params{1} = single(randn(obj.size,'single') * sc) ;
      
      if obj.hasBias
        params{2} = zeros(obj.size(4),1,'single') ;
      end
    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = constconv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
