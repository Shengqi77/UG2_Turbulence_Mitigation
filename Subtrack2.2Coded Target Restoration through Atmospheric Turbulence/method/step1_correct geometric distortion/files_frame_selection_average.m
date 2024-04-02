%The sharpest 30 images in the registration sequence are selected and averaged to remove registration errors
clc
clear
close all 
%frame selection
frames_dir = 'G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\final_mat_float_opticalflow\';
file_dirs = dir(frames_dir);
files_name  = sort_nat({file_dirs.name});
for m = 3:length(file_dirs)
    exist_dir = ['G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\final_mat_float_opticalflow_sharpest\',files_name{m},'\'];
    source_dir = [frames_dir,'\',files_name{m},'\'];
    if exist(exist_dir)
        continue
    else
        mkdir(exist_dir);
    end
    imgs = dir([source_dir '*mat']);
    name = sort_nat({imgs.name});
    for i = 1:length(imgs)
        imgstruct = load([source_dir name{i}]);
        temp = imgstruct.warpI2;
%         temp = imread([source_dir name{i}]);
        if size(temp,3) == 3
            temp = rgb2gray(temp);
        end
        stack(:,:,i) = temp;
    end
    
    num_frames = size(stack,3);
    [rows,cols] = size(temp);
    Ygrad = zeros(rows,cols,num_frames);
    
    %compute image gradient
    for i = 1:size(stack,3)
        tmp = stack(:,:,i);
        [Grad,~] = imgradient(tmp);
        Ygrad(:,:,i) = Grad;
    end
    mean_img = mean(stack,3);
    h    = ones(15,15); h = h/sum(h(:));
    score_sharp = zeros(rows,cols,num_frames);
    for k=1:num_frames            
        score_sharp(:,:,k) = imfilter( abs(Ygrad(:,:,k)), h, 'symmetric' );
    end
    score_sharp_sum = sum(sum(score_sharp));
    score_sharp_sum = reshape(score_sharp_sum,[1,size(score_sharp_sum,3)]);
    [score_sharp_sort,idx_sharp] = sort(score_sharp_sum,'descend');
    for sharp = 1:30
        name_sharp = idx_sharp(sharp);
        source_file = [frames_dir,files_name{m},'\',num2str(name_sharp),'.mat'];
        target_file =exist_dir;
        copyfile(source_file,target_file);
    end
end



%Average the selected 30 images
frames_dir = 'G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\final_mat_float_opticalflow_sharpest\';
file_dirs = dir(frames_dir);
name = sort_nat({file_dirs.name});
for t = 3:length(file_dirs)
    img_dir = [frames_dir name{t},'/'];
    imgs = dir([img_dir '*.mat']);
    mkdir('G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\final_png_float_opticalflow_sharpest_average\');
    for m = 1:length(imgs)
        imgstruct = load([img_dir imgs(m).name]);
        img = imgstruct.warpI2;
        img_frames(:,:,m) = img;
    end
    img_average = mean(img_frames,3);
    img_average = img_average * 65536;
    imwrite(uint16(img_average),['G:\UG2+competation_submit\subtrack2.2\code\method\step1_correct geometric distortion\Intermediate results\dry_run_mat_float_opticalflow_sharpest_average\',name{t},'.png']);
end

