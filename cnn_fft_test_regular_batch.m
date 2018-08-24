clear all
addpath testing_building_batch;
run matlab/vl_setupnn ;

% Load a model and upgrade it to MatConvNet current version.
path = './data/Journal_model/';
modelid = '6-5-9-Regular-net-scale-2';
equalnet = 0; % 0 for regular, 1 for equal
model_path = [path modelid '/net-epoch-200.mat'];
net = load(model_path) ;
up_scale = 2;



net = dagnn.DagNN.loadobj(net.net);
net = cnn_imagenet_deploy(net);

if equalnet
    spatial_size = sqrt(size(net.params(1).value,1));
    global htmatrix; 
    htmatrix = HTmatrix(360,480,[spatial_size spatial_size]);
    global htmatrix_back; 
    htmatrix_back = pinv(htmatrix);
    fprintf('compute HTmatrix, p-inv matrix\n')
end

net.mode = 'test' ;
gpuDevice(1)
net.move('gpu') ;


%% read data
filepath = 'testing/Set19/';
imfiles = dir([filepath '*.png']);

totaltime = 0;
for i = 1:numel(imfiles)
impath = [filepath imfiles(i).name];
im = imread(impath);


%% test network

if size(im,3)>1
    im = rgb2ycbcr(im);
    im = im(:, :, 1);
end
im_gnd = single(modcrop(im, up_scale));

im_lr = imresize(im_gnd,1/up_scale);
im_bic= imresize(im_lr,up_scale);

im_bic_batch = imtobatch(im_bic);   % 360*480*batch

im_bic_batch_ht = hartleyTrans(im_bic_batch,'t');

net_input = reshape(im_bic_batch_ht, [size(im_bic_batch_ht,1), size(im_bic_batch_ht,2),  1, size(im_bic_batch_ht,3)]);


inputs = {'input', gpuArray(net_input)} ;


net.eval(inputs) ;

out = gather(net.vars(end).value);
net_output = reshape(out, [size(out,1), size(out,2), size(out,4)]);

res_batch_out = hartleyTrans(net_output,'i');  % 360*480*batch

res_out = batchtoim(res_batch_out, size(im_bic));

im_out = res_out + im_bic;

%% remove border
im_out_shave = shave(uint8(im_out), [up_scale, up_scale]);
im_gnd_shave = shave(uint8(im_gnd), [up_scale, up_scale]);
im_bic_shave = shave(uint8(im_bic), [up_scale, up_scale]);

%% compute PSNR
psnr_bic(i) = compute_psnr(im_gnd_shave,im_bic_shave);
psnr_our(i) = compute_psnr(im_gnd_shave,im_out_shave);
ssim_bic(i) = ssim(im_gnd_shave,im_bic_shave);
ssim_our(i) = ssim(im_gnd_shave,im_out_shave);

%% draw figure
% figure(1)
% subplot(1,3,1)
% imagesc(im_bic);
% title('input');
% 
% subplot(1,3,2)
% imagesc(im_out);
% title('output');
% 
% subplot(1,3,3)
% imagesc(im_g2)
% imagesc(abs(im_gnd-im_bic));
% title('gt res');nd);
% title('gt');
% 
% 
% figure(2)
% subplot(1,2,1)
% imagesc(abs(res_out));
% title('predict res');
% 
% subplot(1,2,2)
% imagesc(abs(im_gnd-im_bic));
% title('gt res');

end



%% show results
fprintf('Model ID:  %s     Testing on %s \n', modelid, filepath);
fprintf('PSNR for Bicubic Interpolation: %f dB\n', mean(psnr_bic));
fprintf('PSNR for Our Reconstruction: %f dB\n', mean(psnr_our));

fprintf('SSIM for Bicubic Interpolation: %f dB\n', mean(ssim_bic));
fprintf('SSIM for Our Reconstruction: %f dB\n', mean(ssim_our));

% fprintf('Total time:    %f   seconds \n', totaltime);

