function im = batchtoim(batch, ori_size)
c = 40;  % cropsize
na = 360;  %network_a
nb = 480;  %network_b

a = ori_size(1);
b = ori_size(2);

m = ceil(b/(nb - 2*c));
n = ceil(a/(na - 2*c));

im_s = single(zeros(a,b));
for j = 1:n
    for i = 1:m
        im_s( 1+(j-1)*(na-2*c):(j)*(na-2*c),...
            1+(i-1)*(nb-2*c):(i)*(nb-2*c))...
            = batch(c+1:na-c , c+1:nb-c , i+m*(j-1));
    end
end

im = im_s(1:a,1:b);

end