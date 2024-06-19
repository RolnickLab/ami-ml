% Author: Aditya Jain
% Date: June 19, 2023
% About: Compare the difference in two images

clc
close all
clear

orig = "6987_orig.jpg";
wds = "6987_wds.jpg";

%% Difference calculation

img_orig = imread(orig);
img_wds = imread(wds);
img_diff = img_wds - img_orig;

img_diff_r = img_diff(:, :, 1);
img_diff_g = img_diff(:, :, 2);
img_diff_b = img_diff(:, :, 3);

imwrite(img_diff_r, 'img_diff_r.jpg');
imwrite(img_diff_g, 'img_diff_g.jpg');
imwrite(img_diff_b, 'img_diff_b.jpg');
