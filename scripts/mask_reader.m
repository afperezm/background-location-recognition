% function mask_reader(img_filepath, mask_filepath)
%
% Plots on top of an image the convex hull given by the received
% coordinates.
%
% Input:
%   img_filepath: path to an image file
%   mask_filepath: path to a file of coordinates, where first row is the
%   number of coordinates it has and the other rows is an (n x 2) matrix
%

function mask_reader(img_filepath, mask_filepath)
    fid = fopen(mask_filepath, 'r');
    num_vertices = fscanf(fid, '%d', 1);
    vertices = fscanf(fid, '%f %f\n', [2, num_vertices])';
    fclose(fid);
    im = im2double(imread(img_filepath));
    figure, hold off, imshow(im), hold on, plot(vertices(:,1)', vertices(:,2)', 'r-'), hold off;
end