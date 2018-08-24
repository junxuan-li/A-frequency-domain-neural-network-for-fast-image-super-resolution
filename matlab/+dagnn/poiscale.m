classdef poiscale < dagnn.ElementWise
  properties
    wsize  % [length height depth]
    hasBias = false
  end

  methods

    function outputs = forward(obj, inputs, params)
      args = horzcat(inputs, params) ;
      if obj.hasBias
        outputs{1} = bsxfun(@plus, args{1}, args{3}) ;
      else
          outputs{1} = args{1};
      end
      outputs{1} = bsxfun(@times, outputs{1}, args{2}) ;
 
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      args = horzcat(inputs, params) ;
      sz = [size(args{2}) 1 1 1 1] ;
      sz = sz(1:4) ;
      dargs{1} = bsxfun(@times, derOutputs{1}, args{2}) ;  %dy dinput
      if obj.hasBias
        IB = bsxfun(@plus, args{1}, args{3}) ;
      else
        IB = args{1};
      end
      dargs{2} = bsxfun(@times, derOutputs{1}, IB) ;  %dy dfilter
      
      for k = find(sz == 1)
        dargs{2} = sum(dargs{2}, k) ;
      end
      if obj.hasBias
        dargs{3} = dargs{1};
        for k = [3 4]
          dargs{3} = sum(dargs{3}, k) ;
        end
        dargs{3} = symertric(dargs{3});
      end
      derInputs = dargs(1:numel(inputs)) ;
      if size(inputs{1},3) == 1
          derInputs{1} = sum(derInputs{1},3);
      end
      derParams = dargs(numel(inputs)+(1:numel(params)));
%       derParams{1} = zerofilter(derParams{1}) ;
      derParams{1} = symertric(derParams{1});
    end
    
    function params = initParams(obj)
      % Xavier improved
      %sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;
%       params{1} = zerofilter(single(0.1 * random('unif',-1,1, obj.wsize)));
        params{1} = (single(0.1 * random('unif',-1,1, obj.wsize)));
%         params{1} = zerofilter(params{1});
        params{1} = symertric(params{1});
      if obj.hasBias
        params{2} = zeros(obj.wsize(1),obj.wsize(2),'single');
      end
    end

    function obj = poiscale(varargin)
      obj.load(varargin) ;
    end
    


  end
end

function out = zerofilter(in)
    out = in;
%     in_size = size(in);
%     for k = 1:in_size(3)
%         if k<=2
%             
%         elseif  (k>2) && (k<=5) 
%             out(floor(in_size(1)*3/8) : floor(in_size(1)*5/8) , floor(in_size(2)*3/8) : floor(in_size(2)*5/8),k,: ) = 0;
%         elseif   (k>5) && (k<=9) 
%             out(floor(in_size(1)*2/8) : floor(in_size(1)*6/8) , floor(in_size(2)*2/8) : floor(in_size(2)*6/8),k,: ) = 0;
%         else
%             out(floor(in_size(1)*1/8) : floor(in_size(1)*7/8) , floor(in_size(2)*1/8) : floor(in_size(2)*7/8),k,: ) = 0;
%         end
%     end
    
end

function out = symertric(in)
    in2 = flip(flip(in,1),2);
    out = (in+in2)./2;
end