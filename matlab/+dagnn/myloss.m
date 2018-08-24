classdef myloss < dagnn.ElementWise
  properties
    loss = 'l2'
    ignoreAverage = false
    opts = {}
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      if isequal(obj.loss,'l2')
          outputs{1} = vl_nnloss_l2(inputs{1}, inputs{2},[]) ;
      elseif isequal(obj.loss,'l1')
          outputs{1} = vl_nnloss_l1(inputs{1}, inputs{2},[]) ;
      elseif isequal(obj.loss,'expl2')
          outputs{1} = vl_nnloss_expl2(inputs{1}, inputs{2},[]) ;
      elseif isequal(obj.loss,'sqrt')
          outputs{1} = vl_nnloss_sqrt(inputs{1}, inputs{2},[]) ;
      elseif isequal(obj.loss,'l3')
          outputs{1} = vl_nnloss_l3(inputs{1}, inputs{2},[]) ;
      end
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
        if isequal(obj.loss,'l2')
            derInputs{1} = vl_nnloss_l2(inputs{1}, inputs{2}, derOutputs{1}) ;
        elseif isequal(obj.loss,'l1')
            derInputs{1} = vl_nnloss_l1(inputs{1}, inputs{2}, derOutputs{1}) ;
        elseif isequal(obj.loss,'expl2')
            derInputs{1} = vl_nnloss_expl2(inputs{1}, inputs{2}, derOutputs{1}) ;
        elseif isequal(obj.loss,'sqrt')
            derInputs{1} = vl_nnloss_sqrt(inputs{1}, inputs{2}, derOutputs{1}) ;
        elseif isequal(obj.loss,'l3')
            derInputs{1} = vl_nnloss_l3(inputs{1}, inputs{2}, derOutputs{1}) ;
        end
        derInputs{2} = [] ;
        derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = myloss(varargin)
      obj.load(varargin) ;
    end
  end
end
