function y = vl_nnloss_l2(x,c,dzdy,varargin)

if isempty(dzdy)
    delta = (x - c).^2 ;
    y = sum(delta(:)) ;
    %y = y / (size(x,1) * size(x,2)* size(x,3) * size(x,4)) ;
else
    y = 2 * dzdy * (x-c) ;
    %y = y / (size(x,1) * size(x,2)*size(x,3) * size(x,4)) ;
end

end

