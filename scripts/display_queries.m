% display_queries(num_query_images, list_queries_fname)
%
% This function plots in a approximate square grid a set of images read from a
% plain text file.
%
% Input:
%
%	num_query_images: number of images to be read
%	list_queries_fname: name of the file containing the list of query images
%
% Example:
%
%	display_queries(55, 'list_queries.txt')
%

%function display_queries(num_query_images, list_queries_fname)
function display_queries(images_dir_name)

	images_dir = dir([images_dir_name "/*.jpg"]);
	num_query_images = length(images_dir);

	fig_num_rows = ceil(sqrt(num_query_images));
	fig_num_cols = ceil(num_query_images/fig_num_rows);

	figure(1), subplot(fig_num_rows, fig_num_cols, 1);

	for i=1:num_query_images
		query_image_fname = [images_dir_name "/" images_dir(i).name];
		query_image = imread(query_image_fname);
		subplot(fig_num_rows, fig_num_cols, i), imshow(query_image), text(size(query_image,2)/8, -30, sprintf("Query image %02d", i), "fontsize", 10);
	end

	% 1) Reading list queries file
%	fid = fopen(list_queries_fname,'r');
%	if (fid == -1)
%		error(sprintf('A problem occured while opening file [%s] for reading\n', gt_fname)); 
%	end

	% 2) Plotting query images in grid


%	for i=1:num_query_images
%		query_image_fname = fscanf(fid, '%s', 1);
%		query_image_fname = [query_image_fname(1:end-4) ".thumb.jpg"];
%		query_image = imread(query_image_fname);
%		subplot(fig_num_rows, fig_num_cols, i), imshow(query_image), text(size(query_image,2)/8, -30, sprintf("Query image %02d", i), "fontsize", 10);
%	end

end

