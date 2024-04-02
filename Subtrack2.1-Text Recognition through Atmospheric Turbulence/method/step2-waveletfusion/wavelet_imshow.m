clc
clear
close all
img = imread('.\test\scene_35\1.png');
% 读取图像

% 读取图像

% img = imresize(img,[512,512]);
% 将图像转为灰度图像
if size(img, 3) == 3
    img = rgb2gray(img);
end

% 进行3次尺度小波变换
[c,s] = wavedec2(img, 3, 'db4');

ca3 = appcoef2(c, s, 'db4', 3);
ch3 = detcoef2('h', c, s, 3);
cv3 = detcoef2('v', c, s, 3);
cd3 = detcoef2('d', c, s, 3);
c3 = [ca3 ch3; cv3 cd3];
c3 = imresize(c3,[115,115]);
% 
% ca2 = appcoef2(c, s, 'db4', 2);
ch2 = detcoef2('h', c, s, 2);
cv2 = detcoef2('v', c, s, 2);
cd2 = detcoef2('d', c, s, 2);
c2 = [c3 ch2;cv2 cd2];
c2 = imresize(c2,[223,223])

% ca1 = appcoef2(c, s, 'db4', 1);
ch1 = detcoef2('h', c, s, 1);
cv1 = detcoef2('v', c, s, 1);
cd1 = detcoef2('d', c, s, 1);

% 将小波系数合并成一张图像
wimg = [c2 ch1 ; cv1 cd1];

% 显示小波系数图像
figure
colormap gray
imagesc(wimg)
axis off
title('Wavelet Coefficients')

