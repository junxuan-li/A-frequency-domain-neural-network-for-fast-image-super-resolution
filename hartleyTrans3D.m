function out = hartleyTrans3D(in, method)
insize = size(in);


factor = sqrt(1/ (insize(1)*insize(2)*insize(3)) );

for i = 1:size(in,4)
    if isequal(method,'t')

        f= fftshift(fftn(in(:,:,:,i))*factor);
        
    else
        f= fftn(ifftshift(in(:,:,:,i)))*factor;

    end

    out(:,:,:,i) = real(f) - imag(f);

end

end
