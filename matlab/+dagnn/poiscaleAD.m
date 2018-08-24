classdef poiscaleAD < dagnn.ElementWise
  properties
    wsize  % [length height indepth outdepth]
    imsize
    indepth
    outdepth
    hasBias = false
  end

  methods

    function outputs = forward(obj, inputs, params)
      batchsize = size(inputs{1},4);
      inputs{1} = reshape(inputs{1},obj.imsize(1),obj.imsize(2),obj.indepth,1,batchsize);
      args = horzcat(inputs, params) ;
      
      if obj.hasBias
        outputs{1} = bsxfun(@plus, args{1}, args{3}) ;
      else
          outputs{1} = args{1};
      end
      outputs{1} = bsxfun(@times, outputs{1}, args{2}) ;
      if obj.indepth ~= 1 && obj.outdepth ~= 1
          outputs{1} = sum(outputs{1},3);
      elseif obj.outdepth == 1
          outputs{1} = sum(outputs{1},3);
      end
      
      outputs{1} = reshape(outputs{1},obj.imsize(1),obj.imsize(2),obj.outdepth,batchsize);
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      batchsize = size(inputs{1},4);
      inputs{1} = reshape(inputs{1},obj.imsize(1),obj.imsize(2),obj.indepth,1,batchsize);
      args = horzcat(inputs, params) ;
      derOutputs{1} = reshape(derOutputs{1},obj.imsize(1),obj.imsize(2),obj.outdepth,1,batchsize);
      
      
      if obj.indepth ~= 1 && obj.outdepth ~= 1
          derOutTemp = reshape(derOutputs{1},obj.imsize(1),obj.imsize(2),1,obj.outdepth,batchsize);
          dargs{1} = bsxfun(@times, derOutTemp, args{2}) ;  %dy dinput
          dargs{1} = sum(dargs{1},4);
      else
          dargs{1} = bsxfun(@times, derOutputs{1}, args{2}) ;  %dy dinput
      end
      
      if obj.indepth == 1
          dargs{1} = sum(dargs{1},3);
      end
      dargs{1} = reshape(dargs{1},obj.imsize(1),obj.imsize(2),obj.indepth,batchsize); %dy dinput

      
      if obj.hasBias
        IB = bsxfun(@plus, args{1}, args{3}) ;
      else
        IB = args{1};
      end
      if obj.indepth ~= 1 && obj.outdepth ~= 1
          dargs{2} = bsxfun(@times, derOutTemp, IB) ;  %dy dfilter
      else
          dargs{2} = bsxfun(@times, derOutputs{1}, IB) ;  %dy dfilter
      end
      dargs{2} = sum(dargs{2}, 5) ;  %dy dfilter
      
      if obj.hasBias
        dargs{3} = dargs{1};
        dargs{3} = sum(dargs{3}, 4);
      end
      
      derInputs = dargs(1) ;
      derParams = dargs(numel(inputs)+(1:numel(params)));
    end
    
    function params = initParams(obj)
      % Xavier improved
      %sc = sqrt(2 / prod(obj.size(1:3))) ;
      %sc = sqrt(2 / prod(obj.size([1 2 4]))) ;

        params{1} = (single(0.1 * random('unif',-1,1, obj.wsize)));

        if obj.wsize(3)==1
            params{1} = reshape(params{1},obj.wsize(1),obj.wsize(2),obj.wsize(4));
        end
      if obj.hasBias
        params{2} = zeros(obj.wsize(1),obj.wsize(2),obj.wsize(3),'single');
      end
    end

    function obj = poiscaleAD(varargin)
      obj.load(varargin) ;
%       obj.imsize = obj.wsize(1:2);
%       obj.indepth = obj.wsize(3);
%       obj.outdepth = obj.wsize(4);
    end
    


  end
end


function out = symertric(in)
    in2 = flip(flip(in,1),2);
    out = (in+in2)./2;
end