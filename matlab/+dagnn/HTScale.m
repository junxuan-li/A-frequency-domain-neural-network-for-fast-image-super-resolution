classdef HTScale < dagnn.ElementWise
  properties
    wsize       % [Mrow Ncol depth]
    kernel_size % [kernel_width, kernel_height]
    temparams
    
  end

  methods

    function outputs = forward(obj, inputs, params)
      args = horzcat(inputs, params) ;
      outputs{1} = args{1};
      
      obj.temparams = gpuArray(zeros(obj.wsize(1),obj.wsize(2),obj.wsize(3)));
      global htmatrix;
      for i = 1:obj.wsize(3)
        obj.temparams(:,:,i) =  fftshift(reshape(htmatrix * args{2}(:,i),[obj.wsize(2) obj.wsize(1)]).');
      end
      outputs{1} = bsxfun(@times, outputs{1}, obj.temparams) ;
 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      args = horzcat(inputs, params) ;

      global htmatrix_back;

      tempdargs1 = bsxfun(@times, derOutputs{1}, obj.temparams) ;  %dy dinput
      
      dargs{1} = sum(tempdargs1,3); %dy dinput
      
      dydfilter = bsxfun(@times, derOutputs{1}, args{1}) ; %dy dfilter
      dydfilter_sum = sum(dydfilter, 4) ;
      
      for i = 1:obj.wsize(3)
        ishift_dydf = ifftshift(dydfilter_sum(:,:,i));
        vec_filter = ishift_dydf.';
        dargs{2}(:,i) = htmatrix_back * vec_filter(:);  %dy dparams
      end

      derInputs = dargs(1:numel(inputs)) ;
      derParams = dargs(numel(inputs)+(1:numel(params)));

    end
    
    function params = initParams(obj)
      % Xavier improved
      for i = 1:obj.wsize(3)
          sc = sqrt(2 / prod(obj.kernel_size)) ;
          tmp = randn(obj.kernel_size,'single') * sc * 1e-3 ;
          params{1}(:,i) = tmp(:);
      end
    end
    
    function obj = HTScale(varargin)
      obj.load(varargin) ;

    end

  end
end
