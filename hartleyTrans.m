function out = hartleyTrans(in, method)

insize = size(in);
factor = 1 ./ sqrt(insize(1)*insize(2));

for i = 1:size(in,3)
    if isequal(method,'t')
        f= fftshift(fft2(in(:,:,i))*factor);
    else
        f= fft2(ifftshift(in(:,:,i)))*factor;
    end
    out(:,:,i) = real(f) - imag(f);
end

end