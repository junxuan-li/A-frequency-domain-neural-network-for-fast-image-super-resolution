function y = vl_nnloss_sqrt(x,c,dzdy,varargin)


if isempty(dzdy) % forward
    delta = sqrt(abs(x-c));
    
    y = sum(delta(:)) ;
    %y = y / (size(x,1) * size(x,2)* size(x,3) * size(x,4)) ;
else % backward

    y = sign(x-c) .* 0.5 .* (1 ./ sqrt(abs(x-c)));

    %y = y / (size(x,1) * size(x,2)*size(x,3) * size(x,4)) ;
end

end

