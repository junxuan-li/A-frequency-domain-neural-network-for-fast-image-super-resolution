function batch = imtobatch(im)
c = 40;  % cropsize
na = 360;  %network_a
nb = 480;  %network_b

a = size(im,1);
b = size(im,2);

m = ceil(b/(nb - 2*c));
n = ceil(a/(na - 2*c));

p_im1 = padarray(im,[c,c],'symmetric','pre');
p_im = padarray(p_im1, [n*(na-2*c)-a+c, m*(nb-2*c)-b+c],'symmetric','post' );

batch = single(zeros(na,nb,m*n));
for j = 1:n
    for i = 1:m
        batch(:,:,i+m*(j-1)) = p_im(    (j-1)*(na-2*c)+1 : j*(na-2*c)+2*c ,...
                                        (i-1)*(nb-2*c)+1 : i*(nb-2*c)+2*c);
    end
end

end