function [recall_rates, precision_rates] = eval_performance(num_query_images, num_candidates, candidates_occurrence_fname)
% [recall_rates, precision_rates] = eval_performance(num_query_images, num_candidates, candidates_occurrence_fname)
%
% This function computes and plots the average precision and recall over all
% the query images while varying the numbers of candidates.
%
% The precision and recall measure in this case how good in average are the set
% of ranked candidates.
%
% For this purpose uses the matrix of occurrences of the right landmark id
% on each of the candidates. The rows of the matrix correspond to the number of
% query images and the columns to the candidates retrieved for that query. Each
% element is 1 if the ground truth landmark id of the (i,j)th candidate is
% the same than that of the ith query image.
%
% Input:
%	num_query_images: number of query images, or what is the same, number of rows
%	of the candidates occurrence matrix.
%	num_query_candidates: number of candidates per query image, or what is the same,
%	number of columns of the candidates occurrence matrix.
%	candidates_occurrence_fname: path to the file containing the binary matrix
%	of candidates occurrences.
%
% Output:
%	recall_rates: array of average recall for a varying number of candidates.
%	precision_rates: array of average precision for a varying number of candidates.
%	votes_mat: matrix with the ids for the highest voted landmarks for each
%	query image for a varying number of candidates.
%
% Example:
%
%	num_query_images = 55;
%	num_candidates = 50;
%	cand_occurr_fname = 'candidates_occurrence.txt';
%	[recall_rates, precision_rates] = eval_performance(num_query_images, num_candidates, candidates_occurrence_fname);
%

% 1) Reading candidate occurrences file
fid = fopen(candidates_occurrence_fname,'r');
if (fid == -1) 
	error(sprintf('A problem occured while opening file [%s] for reading\n', gt_fname)); 
end

candidates_mat = zeros(num_query_images, num_candidates);

% Line format: <query_image_name> <landmark 1> ... <landmark n>
for i=1:num_query_images
	for j=1:num_candidates
		candidates_mat(i, j) = fscanf(fid, '%d', 1);
	end
end

% 4) Computing avg precision and recall
recall_rates=zeros(1,num_candidates);
precision_rates=zeros(1,num_candidates);
% true positives: candidates IN the ranked list labeled with the CORRECT landmark id
% false positives: candidates IN the ranked list labeled with the WRONG landmark id
% false negatives: candidates NOT IN the ranked list labeled with the CORRECT landmark id
% true negatives: candidates NOT IN the ranked list labeled with the WRONG landmark id

num_query_images=1;

for k=1:num_candidates
	for i=1:num_query_images
		tp = sum(candidates_mat(i, 1:k));
		fp = k-tp;
		fn = sum(candidates_mat(i, k+1:end));
		tn = num_candidates-k-fn;
		recall = tp/(tp+fn);
		recall_rates(k)=recall_rates(k)+recall;
		precision = tp/(tp+fp);
		precision_rates(k)=precision_rates(k)+precision;
	end
	recall_rates(k)=recall_rates(k)/num_query_images;
	precision_rates(k)=precision_rates(k)/num_query_images;
end

% Plot avg precision and recall
subplot(3,1,1), plot(1:num_candidates, recall_rates, 'r-o'), xlabel('Number of candidates'), ylabel('Recall');
subplot(3,1,2), plot(1:num_candidates, precision_rates, 'r-o'), xlabel('Number of candidates'), ylabel('1-Precision');
subplot(3,1,3), plot(precision_rates, recall_rates, 'r-o'), xlabel('1-Precision'), ylabel('Recall');

end

