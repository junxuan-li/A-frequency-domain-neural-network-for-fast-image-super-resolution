function y = vl_nnloss_expl2(x,c,dzdy,varargin)
afa = 1;
beta = -0.01;
f = fw(size(x),afa,beta);

if isempty(dzdy) % forward
    delta =  f.*((x-c).^2);
    y = sum(delta(:)) ;
    %y = y / (size(x,1) * size(x,2)* size(x,3) * size(x,4)) ;
else % backward
    y = f.* 2 .* dzdy .* (x-c);
    %y = y / (size(x,1) * size(x,2)*size(x,3) * size(x,4)) ;
end

end

function f = fw(w,afa,beta)
f = zeros(w);

for  i = 1:w(1)
    for  j = 1:w(2)
        f(i,j,:,:) = afa * exp(-beta .* max(i,j));

    end
end
end