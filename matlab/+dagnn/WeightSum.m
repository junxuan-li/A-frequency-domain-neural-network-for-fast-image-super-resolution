classdef WeightSum < dagnn.ElementWise
  %SUM DagNN sum layer
  %   The SUM layer takes the sum of all its inputs and store the result
  %   as its only output.

  properties (Transient)
    numInputs
  end

  methods
    function outputs = forward(obj, inputs, params)
      obj.numInputs = numel(inputs) ;
      weights = params{1};
      outputs{1} = inputs{1}.*weights(1) ;
      for k = 2:obj.numInputs
        outputs{1} = outputs{1} + inputs{k}.*weights(k) ;
      end
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      weights = params{1};
      for k = 1:obj.numInputs
        derInputs{k} = derOutputs{1}.*weights(k) ;
        temp = derOutputs{1}.*inputs{k};
        derWeights(k) = sum( temp(:) );
      end
      derParams{1} = derWeights ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes)
      outputSizes{1} = inputSizes{1} ;
      for k = 2:numel(inputSizes)
        if all(~isnan(inputSizes{k})) && all(~isnan(outputSizes{1}))
          if ~isequal(inputSizes{k}, outputSizes{1})
            warning('Sum layer: the dimensions of the input variables is not the same.') ;
          end
        end
      end
    end

    function rfs = getReceptiveFields(obj)
      obj.numInputs = numel(obj.net.layers(obj.layerIndex).inputs) ;
      rfs.size = [1 1] ;
      rfs.stride = [1 1] ;
      rfs.offset = [1 1] ;
      rfs = repmat(rfs, obj.numInputs, 1) ;
    end

    function params = initParams(obj)
        temp = single(1 * ones(1,obj.numInputs));
%         initialpar = single([13.7988   -5.0131    0.3576    0.1961    0.2007    0.1269]);
%         for i=1:min(obj.numInputs,6)
%             temp(i) = initialpar(i);
%         end
        params{1} = temp;
    end
    
    function obj = WeightSum(varargin)
      obj.load(varargin) ;
    end
  end
end
