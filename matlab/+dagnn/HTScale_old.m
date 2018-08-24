classdef HTScale_old < dagnn.ElementWise
  properties
    wsize       % [Mrow Ncol]
    kernel_size % [kernel_width, kernel_height], usually 5*5
    
  end

  methods

    function outputs = forward(obj, inputs, params)
      args = horzcat(inputs, params) ;
      outputs{1} = args{1};
      global htmatrix;
      ht_params =  reshape(htmatrix * args{2},[obj.wsize(2) obj.wsize(1)]).';
      outputs{1} = bsxfun(@times, outputs{1}, ht_params) ;
 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      args = horzcat(inputs, params) ;

      global htmatrix;
      global htmatrix_back;
      ht_params =  reshape(htmatrix * args{2},[obj.wsize(2) obj.wsize(1)]).';
      dargs{1} = bsxfun(@times, derOutputs{1}, ht_params) ;  %dy dinput

      dydfilter = bsxfun(@times, derOutputs{1}, args{1}) ; %dy dfilter
      dydfilter_sum = sum(dydfilter, 4) ;

      vec_filter = dydfilter_sum.';
      dargs{2} = htmatrix_back * vec_filter(:);  %dy dparams
      

      derInputs = dargs(1:numel(inputs)) ;

      derParams = dargs(numel(inputs)+(1:numel(params)));

    end
    
    function params = initParams(obj)
      % Xavier improved
      sc = sqrt(2 / prod(obj.kernel_size)) ;
      tmp = randn(obj.kernel_size,'single') * sc * 1e-3 ;
      params{1} = tmp(:);
    end
    
    function obj = HTScale_old(varargin)
      obj.load(varargin) ;
    end

  end
end
