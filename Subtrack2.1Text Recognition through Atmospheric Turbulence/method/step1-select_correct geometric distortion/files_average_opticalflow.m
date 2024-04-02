%Image Registration of Degraded Sequences Using Optical Flow
clc
clear
close all;
addpath('mex');

%average the distorted sequences to get the reference
frames_dir = '.\Intermediate results\sharpest_sequences\';
file_dirs = dir(frames_dir);
name = sort_nat({file_dirs.name});
for i = 3:length(file_dirs)
    %Obtain a reference frame by averaging degraded sequences
    img_dir = [frames_dir name{i},'/'];
    imgs = dir([img_dir '*.png']);
    mkdir(['.\Intermediate results\sharpest_registerted_sequences\',name{i}])
    for j = 1:length(imgs)
        img = double(imread([img_dir imgs(j).name]));
        img_frames(:,:,:,j) = img;
    end
    img_average = mean(img_frames,4);
    
    %optical flow-image registration
    % load the ref_image
    ref_img = img_average;
    % set optical flow parameters (see Coarse2FineTwoFrames.m for the definition of the parameters)
    alpha = 0.02;
    ratio = 0.75;
    minWidth = 20;
    nOuterFPIterations = 7;
    nInnerFPIterations = 1;
    nSORIterations = 30;
    
    para = [alpha,ratio,minWidth,nOuterFPIterations,nInnerFPIterations,nSORIterations];
    
    % this is the core part of calling the mexed dll file for computing optical flow
    % it also returns the time that is needed for two-frame estimation
    
    %load the images
    ref_img = ref_img /255;
    for j = 1:length(imgs)
        img = double(imread([img_dir imgs(j).name]));
        img = img/255;
        [vx,vy,warpI2] = Coarse2FineTwoFrames(ref_img,img,para);
        imwrite(uint8(warpI2*255),['.\Intermediate results\sharpest_registerted_sequences\',name{i},'/',num2str(j),'.png'])
    end
end
 




