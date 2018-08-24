path = './VOC2012/JPEGImages/';
filetype = '*.jpg';
files = dir([path filetype]);

meansize = [360 480];
numoff = 5000;

% imdb.data = single(zeros(meansize(1),meansize(2),numoff));
% imdb.label = single(zeros(meansize(1),meansize(2),numoff));
% 
for scale_factor=[3 4]
clear imdb

for id = 1:numoff
    filename = fullfile(path,files(id).name);
	ycc = single(rgb2ycbcr(imread(filename)));
    im = ycc(:,:,1);
    if size(im,1)> size(im,2)
        im = im';
    end
    im = imresize(im, meansize);

    lr = imresize(im,1/scale_factor);
    lr_up = imresize(lr,scale_factor); % ht spectrum
    
    label_patch = hartleyTrans(im - lr_up,'t') ;
    data_patch = hartleyTrans(lr_up,'t');
    
    imdb.data(:,:,id) = data_patch;
    imdb.label(:,:,id) = label_patch;

    if mod(id,50) == 0
        display(id);
    end

end
save(['VOC2012_' num2str(meansize(1)) '_' num2str(meansize(2)) '_' num2str(numoff) '_bic_' num2str(scale_factor) '.mat'],'imdb','-v7.3');
% save('bsds200_321_481_200.mat','imdb','-v7.3');
% save('t91_200_266_3640.mat','imdb','-v7.3');
end