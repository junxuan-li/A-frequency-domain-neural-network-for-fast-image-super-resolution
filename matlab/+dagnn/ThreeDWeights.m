classdef ThreeDWeights < dagnn.ElementWise
  properties
    wsize  % [length height indepth outdepth]
    hasBias = false
  end

  methods

    function outputs = forward(obj, inputs, params)
      args = horzcat(inputs, params) ;

      outputs{1} = bsxfun(@times, args{1}, args{2}) ;
 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      args = horzcat(inputs, params) ;
      sz = [size(args{2}) 1 1 1 1] ;
      sz = sz(1:4) ;
      dargs{1} = bsxfun(@times, derOutputs{1}, args{2}) ;  %dy dinput

      dargs{2} = bsxfun(@times, derOutputs{1}, args{1}) ;  %dy dfilter
      
      for k = find(sz == 1)
        dargs{2} = sum(dargs{2}, k) ;
      end

      derInputs = dargs(1:numel(inputs)) ;
      
      szin = [size(inputs{1}) 1 1 1 1] ;
      szin = szin(1:4) ;
      for k = find( szin == 1 )
          derInputs{1} = sum(derInputs{1},k);
      end
      derParams = dargs(numel(inputs)+(1:numel(params)));

%       derParams{1} = symertric(derParams{1});
    end
    
    function params = initParams(obj)
      % Xavier improved
      %sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
      params{1} = (single(0.1 * random('unif',-1,1, obj.wsize)));
%     params{1} = symertric(params{1});
      if obj.hasBias
        params{2} = zeros(obj.wsize(1),obj.wsize(2),'single');
      end
    end

    function obj = ThreeDWeights(varargin)
      obj.load(varargin) ;
    end
    

  end
end
