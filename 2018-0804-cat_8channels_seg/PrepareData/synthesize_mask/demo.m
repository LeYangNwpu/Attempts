clear, clc

cla_folder_path = '/home/yangle/TCyb/dataset/MSRA10K/val_formask/';
img_rp = '/home/yangle/TCyb/dataset/MSRA10K/val_1/';
gt_rp = '/home/yangle/TCyb/dataset/MSRA10K/valannot_1/';

img_set = dir([img_rp, '*.png']);
parfor iimg = 1:length(img_set)
%for iimg = 1:2
    disp(iimg);
    img_name = img_set(iimg).name;
    filename = img_name(1:end-4);
    file_name = num2str((str2double(filename)), '%06d');
    img = imread([img_rp, img_name]);
    gt = imread([gt_rp, filename, '.png']);
    folder_path = [cla_folder_path, file_name, '/'];
    
    % deform segmentation masks
    augment_image_and_mask(gt, folder_path,file_name);
    
end
