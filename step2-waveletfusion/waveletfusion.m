% correct turbulence using region-based fusion (pixel-based is also provided) and roi

clc
close all
clear all

addpath('codes');

% -------------------------------------------------------------------------
% input distorted sequence
% -------------------------------------------------------------------------

frames_dir = 'G:\UG2+competation_submit\subtrack2.1\code\method\step1-select_correct geometric distortion\Intermediate results\sharpest_registerted_sequences\';
file_dirs = dir(frames_dir);
name = sort_nat({file_dirs.name});
mkdir('.\Intermediate results\waveletfusion_result\');
for i = 3:length(file_dirs)
    exist_dir = ['.\Intermediate results\waveletfusion_result\',name{i},'.png'];
    if exist(exist_dir);
        continue
    end
    dirnameroot = [frames_dir name{i},'\'];
    dirname = dirnameroot;% [dirnameroot,'distortion',num2str(dist),'\'];
    extfile = 'png';%'png';
    startFrame = 1;
    totalFrame = 30; %length(dir([dirnameroot,'*.',extfile]));
    
    % -------------------------------------------------------------------------
    % control parameters
    % -------------------------------------------------------------------------
    resizeRatio = 1;
    fusionMethod = 'region';     % 'region', 'pixel'
                                % 'pixel' is faster, 'region' is better for noisy
    refFrameType = 'average';   % 'maxGradient', 'maxHP'
    doFrameSelection = false;   % true if i) speed up, ii) need stabilisation
    selectROI = false;          % true if ROI is significantly shifted between succesive frames
    sharpMethod = 'gradient';   %'gradient' or 'dtcwt'
    doPostprocess = true;      % true - sharpening and denoising;
    doEnhance = true;           % false;
    
    mseGain = 1;
    gradGain1 = 1;
    gradGain2 = 1;
    infoGain2 = 20;
    
    levels = 4;
    
    numFrameRegis = 50;
    maxFrameused = 25;
    clipLimit = 0.002;
    useBigAreaInfo = false;     % true;
    
    
    % -------------------------------------------------------------------------
    % load input sequence
    % -------------------------------------------------------------------------
    %[input, inputU, inputV] = loadInput(dirname, extfile,startFrame, totalFrame, resizeRatio);
    [input, inputU, inputV] = loadInput(dirname, extfile, totalFrame, resizeRatio);


    % -------------------------------------------------------------------------
    % find reference frame
    % -------------------------------------------------------------------------
    avgFrame = findRefFrame(input, refFrameType);
    
    % -------------------------------------------------------------------------
    % select good frames
    % -------------------------------------------------------------------------
    if doFrameSelection
        [height, width, totalFrame] = size(input);
        rangei = 1:height;
        rangej = 1:width;
        rangek = 1:totalFrame;
        [valcostSelect, indcostSelect] = findGoodFrame(avgFrame, input, rangei, rangej, rangek, totalFrame, sharpMethod, 0, ...
        mseGain, gradGain1, 0);
    else
        indcostSelect = 1:totalFrame;
    end
    
    % -------------------------------------------------------------------------
    % select good frames from ROI
    % -------------------------------------------------------------------------
    if selectROI
        [x_pos, y_pos] = selectPoint(avgFrame,2,'click roughly ROI');
        rangei = round(y_pos(1):y_pos(2));
        rangej = round(x_pos(1):x_pos(2));
        [valcostSelect, indcostSelectupdate] = findGoodFrame(input(:,:,indcostSelect(1)),input,...
            rangei, rangej, indcostSelect, numFrameRegis, sharpMethod, 0, ...
            mseGain, gradGain2, infoGain2);
    else
        indcostSelectupdate = indcostSelect;
    end
    if doFrameSelection || selectROI
        halfval = (mseGain+gradGain2)/2; % half of max costSelect
        [~,numFrametoUse] = min(abs(valcostSelect-halfval));
        numFrametoUse = min(numFrametoUse,maxFrameused);
    
        % reduce input to only good selected frames
        input = input(:,:,indcostSelectupdate(1:numFrametoUse));
        if ~isempty(inputU)
            inputU = inputU(:,:,indcostSelectupdate(1:numFrametoUse));
            inputV = inputV(:,:,indcostSelectupdate(1:numFrametoUse));
        end
    end
    % -------------------------------------------------------------------------
    % registration
    % -------------------------------------------------------------------------
    if isempty(inputU)
        [xrest] = Nick_regis(input, input, input, levels);
    else
        [xrest, xrestU, xrestV] = Nick_regis(input, inputU, inputV, levels);
    end

    % -------------------------------------------------------------------------
    % fusion
    % -------------------------------------------------------------------------
    if strcmp(fusionMethod, 'pixel')
        [zrest, zrestsmooth] = Nick_pixel_fuse(xrest, levels);
    else
        [zrest, zrestsmooth, zrestReg] = fuseRegionROI(xrest, levels);
    end
    % -------------------------------------------------------------------------
    % contrast enhancement
    % -------------------------------------------------------------------------
    if doPostprocess
        zrest = postDenoiseSharpen(zrest,levels);
    end
    
    % -------------------------------------------------------------------------
    % contrast enhancement
    % -------------------------------------------------------------------------
    if doEnhance
        zrest = adapthisteq(uint8(zrest), 'clipLimit',clipLimit);
    end
    % -------------------------------------------------------------------------
    % final result
    % -------------------------------------------------------------------------
    if ~isempty(inputU)
        zrest(:,:,2) = mean(inputU,3);
        zrest(:,:,3) = mean(inputV,3);
        zrest = ycbcr2rgb(uint8(zrest));
    end
%     if strcmp(fusionMethod, 'pixel')
    imwrite(uint8(zrest), exist_dir);
end
