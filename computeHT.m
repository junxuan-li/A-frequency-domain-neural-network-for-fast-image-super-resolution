function out = computeHT(kernel, targetsize)

uvm = HTmatrix(targetsize(1),targetsize(2),size(kernel));

out1 = uvm * kernel(:);

out = reshape(out1,[targetsize(2) targetsize(1)]).';

end