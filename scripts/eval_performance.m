% Evaluate performance of the recognition pipeline

[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'candidates_occurrence_pregv.txt');
plot_performance_rates(recall_rates, precision_rates, 'r-d')

[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'candidates_occurrence_10_auto_0.8.txt');
plot_performance_rates(recall_rates, precision_rates, 'b-d')

[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'candidates_occurrence_3_100_0.5.txt');
plot_performance_rates(recall_rates, precision_rates, 'g-d')

subplot(3,1,1)

legend({"Pre geometric verification", "Post geometric verification 10 auto 0.8", "Post geometric verification 3 100 0.5"}, "location", "northwest");
