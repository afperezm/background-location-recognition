% display_queries(images_dir_name)
%
% This function plots in an approximate square grid a set of JPEG images read from a folder.
%
% Input:
%
%	images_dir_name: path to a folder containing files with extension .jpg
%
% Example:
%
%	display_queries('../../oxford5k/queries/')
%

function display_queries(images_dir_name)

	images_dir = dir([images_dir_name '/*.jpg']);
	num_query_images = length(images_dir);

	fig_num_rows = ceil(sqrt(num_query_images));
	fig_num_cols = ceil(num_query_images/fig_num_rows);

	figure(1), subplot(fig_num_rows, fig_num_cols, 1);

	for i=1:num_query_images
		query_image_fname = [images_dir_name '/' images_dir(i).name];
		query_image = imread(query_image_fname);
		subplot(fig_num_rows, fig_num_cols, i), imshow(query_image), text(size(query_image,2)/8, -30, sprintf('Query image %02d', i), 'fontsize', 10);
	end

end

