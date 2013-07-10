% masker(images_dir_name)
%
% This function implements the algorithm for feature selection based
% on the identification of the main plane using found long straight lines
% classified according to automatically detected vanishing points.
%
% The algorithm is run over a set of JPEG images read from a folder.
%
% Input:
%
%	images_dir_name: path to a folder containing files with extension .jpg
%
% Example:
%
%	addpath('../finding-long-straight-lines/')
%	addpath('../vanishing-points/')
%	masker('~/oxford_buildings_dataset/oxbuild_images/')
%

function masker(images_dir_name)

	disp(sprintf('\nComputing masks for JPEG images located in [%s]\n', images_dir_name));

	% Read folder
    images_dir = dir([images_dir_name '/*.jpg']);
    num_query_images = length(images_dir);

	% For each of the JPEG images
    for j=1:num_query_images
		% Read image
        query_image_fname = [images_dir_name filesep images_dir(j).name];
        disp(sprintf('Reading file [%s]\n', query_image_fname));
        im = im2double(rgb2gray(imread(query_image_fname)));

		% Estimating vanishing points for the read image
        disp(sprintf('  Finding long straight lines\n'));
        lines = APPgetLargeConnectedEdges(im, 0.05*length(diag(im(:,:,1))));
        disp(sprintf('  Estimating vanishing points\n')); 
        [vp, inliers] = estimate_vps(lines(:,1:4));

        % Find the inlier set with the lower variance
        disp(sprintf('  Finding the set of lines with the lower angle variance\n')); 
        if(isempty(inliers))
            inlier_set = lines;
        else
            var_coeffs=[];
            %figure(1), hold off;

            for i=1:length(inliers)
                %subplot(1,length(inliers),i);
                %hist(rad2deg(lines(cell2mat(inliers(i)), 5)));
                var_coeff = var(rad2deg(lines(cell2mat(inliers(i)), 5)));
                var_coeffs = [var_coeffs var_coeff];
            end

            [val,idx_best_set] = min(var_coeffs);
            inlier_set = lines(cell2mat(inliers(idx_best_set)), 1:4);
        end

        xx = [inlier_set(:,1) ; inlier_set(:,2)];
        yy = [inlier_set(:,3) ; inlier_set(:,4)];

        disp(sprintf('  Computing convex hull for the found set of lines\n')); 
        k = convhull(xx,yy);

        vertices = [xx(k) yy(k)]';
        disp(sprintf('  Writing [%d] convex hull vertices to a file\n', length(k)));
        fileID = fopen([images_dir_name filesep substring(images_dir(j).name,0,length(images_dir(j).name)-5) '_mask.txt'],'w');
        fprintf(fileID,'%d\n', length(k));
        fprintf(fileID,'%f %f\n', vertices);
        fclose(fileID);
    end
end

