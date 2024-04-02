
%Image Registration of Degraded Sequences Using Optical Flow

clc
clear
close all;
addpath('mex');
frames_dir = 'G:\UG2+competation_submit\subtrack2.2\code\distorted_sequences\distorted_sequences_mat\';
file_dirs = dir(frames_dir);
name = sort_nat({file_dirs.name});
for i = 3:length(file_dirs)
    %Obtain a reference frame by averaging degraded sequences
    img_dir = [frames_dir name{i},'/'];
    imgs = dir([img_dir '*.mat']);
    img_sum = 0;
    mkdir(['G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\final_mat_float_opticalflow\',name{i}])
    img_frames = zeros(271,271,100);
    for j = 1:length(imgs)
        imgstruct = load([img_dir imgs(j).name]);
        img = imgstruct.matrix;
        img_frames(:,:,j) = img;
    end
    img_average = mean(img_frames,3);
    %Optical Flow
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
    for m = 1:length(imgs)
        imgstruct = load([img_dir imgs(m).name]);
        img = imgstruct.matrix;
        [vx,vy,warpI2] = Coarse2FineTwoFrames(ref_img,img,para);
        save(['G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\final_mat_float_opticalflow\',name{i},'/',num2str(m),'.mat'],'warpI2');
    end
end






