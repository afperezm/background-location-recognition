function [recall_rates, precision_rates, votes_mat] = eval_performance(num_query_images, num_candidates, gt_fname, candidates_fname)
% [recall_rates, precision_rates, votes_mat] = eval_performance
%
% This function computes and plots the average precision and recall
% over all the query images while varying the numbers of candidates.
% The precision and recall measure in this case measure how good in
% average are the set of ranked candidates.
%
% For this purpose it reads and uses the query images ground truth
% labels and the matrix of candidate db images for each query image.
%
% Additionally it computes the voting matrix. The idea behind ranking
% several images at once is to get a more reliable landmark prediction
% by implementing a voting scheme on this ranked list.
%
% Input:
%
% Output:
%	recall_rates: array of average recall for a varying number of candidates
%	precision_rates: array of average precision for a varying number of candidates
%	votes_mat: matrix with the ids for the highest voted landmarks for each query image for a varying number of candidates
%
% Example:
%
% num_query_images = 55;
% num_candidates = 50;
% gt_fname = 'list_gt.txt';
% candidates_fname = 'candidates.txt';
%
% [recall_rates, precision_rates, votes_mat] = eval_performance(num_query_images, num_candidates, gt_fname, candidates_fname)

% 1) Reading queries ground truth file
fid = fopen(gt_fname,'r');

if (fid == -1) 
	error(sprintf('Error opening file [%s] for reading\n', gt_fname)); 
end

query_images = [];

% Line format: <query_image_name> <landmark_id>
for i=1:num_query_images
	% Reading query image name
	query_name = fscanf(fid, '%s', 1);
	% Reading landmark id
	landmark_id = fscanf(fid, '%d', 1);
	query_images = [query_images;[{query_name landmark_id}]];
end

fclose(fid);

% 2) Reading candidates file
fid = fopen(candidates_fname, 'r');

if (fid == -1) 
	error(sprintf('Error opening file [%s] for reading\n', candidates_fname)); 
end

candidates_mat = zeros(num_query_images, num_candidates + 1);

% Line format: <query_image_name> <landmark 1> ... <landmark n>
for i=1:num_query_images
	% Reading query image name
	query_name = fscanf(fid, '%s', 1);
	% Finding corresponding landmark id to the query image name
	landmark_id = query_images(find(ismember(query_images(:,1),query_name)),2);
	candidates_mat(i,1) = cell2mat(landmark_id);
	% Reading ids of candidate landmarks
	for k=1:num_candidates
		candidate = fscanf(fid, '%d', 1);
		candidates_mat(i, k+1) = candidate;
	end
end

fclose(fid);

% 3) Computing voting matrix
votes_mat = zeros(size(candidates_mat,1), size(candidates_mat,2)-1);

% Composing array of landmark ids
bins = unique(cell2mat(query_images(:,2)));

for k=2:num_candidates+1
	for i=1:num_query_images
		% Compute how many votes received each landmark using the varying length ranked list of candidates
		h = hist(candidates_mat(i,2:k), bins);
		[max_votes, max_landmark] = max(h);
		% The classification result is the max voted landmark
		votes_mat(i,k-1) = max_landmark - 1;
	end
end

% 4) Computing avg precision and recall
recall_rates=zeros(1,num_candidates);
precision_rates=zeros(1,num_candidates);
% true positives correspond to candidates in the ranked list labeled with the correct landmark id
% false positives correspond to candidates in the ranked list labeled with the wrong landmark id
% false negatives correspond to candidates not in the ranked list labeled with the correct landmark id
% true negatives correspond to candidates not in the ranked list labeled with the wrong landmark id

for k=1:num_candidates
	for i=1:num_query_images
		q = candidates_mat(i, 1);
		% candidates in the ranked list
		d = candidates_mat(i, 2:k+1);
		tp = sum(q==d);
		fp = sum(q!=d);
		% candidates not in the ranked list
		d = candidates_mat(i, k+2:num_candidates+1);
		fn = sum(q==d);
		tn = sum(q!=d);
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

