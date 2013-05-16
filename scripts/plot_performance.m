function plot_performance(recall_rates, precision_rates, line_format='r-o')
% function plot_performance(recall_rates, precision_rates, line_format='r-o')
%
% This function plots the average precision and recall rates computed previously
% starting from the matrix of right landmark ids occurrences.
%
% Input:
%	recall_rates: average recall values for a varying number of candidates
%	precision_rates: average precision values for a varying number of candidates
%	line_format: format of the ploted curves
%

% Plot avg precision and recall
subplot(3,1,1), hold on, plot(1:length(recall_rates), recall_rates, line_format), hold off, xlabel('Number of candidates'), ylabel('Recall');
subplot(3,1,2), hold on, plot(1:length(precision_rates), precision_rates, line_format), hold off, xlabel('Number of candidates'), ylabel('1-Precision');
subplot(3,1,3), hold on, plot(precision_rates, recall_rates, line_format), hold off, xlabel('1-Precision'), ylabel('Recall');

end
