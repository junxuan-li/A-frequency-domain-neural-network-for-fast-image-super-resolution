function out = HTmatrix(Mrow,Ncol,kernel_size)
k = kernel_size(1)*kernel_size(2);
out = ones(Mrow*Ncol, k);
for r = 1:Mrow*Ncol
    for c = 1:k
        u = fix((r-1)/Ncol);
        v = mod(r-1,Ncol);
        
        x = mod(c-1,kernel_size(1));
        y = fix((c-1)/kernel_size(1));
        
        f = exp(-1i*2*pi*(x*u/Mrow + y*v/Ncol));
        out(r,c) = real(f) - imag(f);
    
    end

end

end