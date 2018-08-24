classdef SepConv < dagnn.Filter
  properties
    size = [0 0 0 0]  % [conv_m conv_n indepth outdepth]
    hasBias = false
    opts = {'cuDNN'}
  end

  methods
    function outputs = forward(obj, inputs, params)
      if ~obj.hasBias, params{2} = [] ; end
%       in_size = size(inputs{1});
%       outputs{1} = single(zeros(in_size(1),in_size(2),obj.size(4)));
      for i = 1:obj.size(4)
          outputs{1}(:,:,i) = vl_nnconv(...
            inputs{1}(:,:,:,i), params{1}(:,:,:,i), params{2}, ...
            'pad', obj.pad, ...
            'stride', obj.stride, ...
            'dilate', obj.dilate, ...
            obj.opts{:}) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      if ~obj.hasBias, params{2} = [] ; end
      for i = 1:obj.size(4)
          [derInputs{1}(:,:,:,i), derParams{1}(:,:,:,i), derParams{2}] = vl_nnconv(...
            inputs{1}(:,:,:,i), params{1}(:,:,:,i), params{2}, derOutputs{1}(:,:,i), ...
            'pad', obj.pad, ...
            'stride', obj.stride, ...
            'dilate', obj.dilate, ...
            obj.opts{:}) ;
      end
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
      sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      params{1} = randn(obj.size,'single') * sc ;

    end

    function set.size(obj, ksize)
      % make sure that ksize has 4 dimensions
      ksize = [ksize(:)' 1 1 1 1] ;
      obj.size = ksize(1:4) ;
    end

    function obj = SepConv(varargin)
      obj.load(varargin) ;
      % normalize field by implicitly calling setters defined in
      % dagnn.Filter and here
      obj.size = obj.size ;
      obj.stride = obj.stride ;
      obj.pad = obj.pad ;
    end
  end
end
