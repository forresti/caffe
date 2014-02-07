function [ds, bs, trees] = imgdetect(im, model, thresh, imgFname)
% Wrapper around gdetect.m that computes detections in an image.
%   [ds, bs, trees] = imgdetect(im, model, thresh)
%
% Return values (see gdetect.m)
%
% Arguments
%   im        Input image
%   model     Model to use for detection
%   thresh    Detection threshold (scores must be > thresh)

im = color(im);
%pyra = featpyramid(im, model);
pyra = convnet_featpyramid(imgFname); %Forrest
[ds, bs, trees] = gdetect(pyra, model, thresh);
