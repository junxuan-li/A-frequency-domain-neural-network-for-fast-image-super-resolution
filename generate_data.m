function generate_data

im = imread('Lenna.png');
imgray = rgb2gray(im);
imgray = imresize(imgray,[321 481]);
scale = 2;

for k = 1:300
    im = imrotate(imgray, random('unif',0,360),'crop');
    im_lr  = imresize(imresize(im,1/scale),[321 481]);
    dct = single(dct2(im));
    dct_lr = single(dct2(im_lr));
    
    imdb.data(:,:,k) = dct_lr;
    imdb.label(:,:,k) = dct - dct_lr;
end

end