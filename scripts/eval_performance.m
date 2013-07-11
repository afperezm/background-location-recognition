% Evaluate performance of the recognition pipeline

figure(1);
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_pregv.txt');
plot_performance_rates(recall_rates, precision_rates, 'r');
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_postgv.txt');
plot_performance_rates(recall_rates, precision_rates, 'g');
subplot(3,1,1);
legend({'Visual', 'Geometric auto'}, 'location', 'northwest');

h=figure(1);
FN = findall(h,'-property','FontName');
set(FN, 'FontName', 'Verdana');

print -dpng pregv_vs_postgv.png

figure(2);
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_pregv.txt');
plot_performance_rates(recall_rates, precision_rates, 'r');
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_3_100_0.5.txt');
plot_performance_rates(recall_rates, precision_rates, 'b')
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_3_auto_0.5.txt');
plot_performance_rates(recall_rates, precision_rates, 'c')
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_postgv.txt');
plot_performance_rates(recall_rates, precision_rates, 'g')
subplot(3,1,1);
legend({'Visual', 'Geometric fixed', 'Geometric auto', 'Geometric loose'}, 'location', 'northwest');

h=figure(2);
FN = findall(h,'-property','FontName');
set(FN, 'FontName', 'Verdana');

print -dpng pregv_vs_postgv_auto_vs_postgv_fix.png

figure(3);
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_pregv.txt');
plot_performance_rates(recall_rates, precision_rates, 'r')
[recall_rates, precision_rates] = compute_performance_rates(55, 50, 'occurrence_matrix_featsel.txt');
plot_performance_rates(recall_rates, precision_rates, 'g')
subplot(3,1,1);
legend({'Visual', 'Masked'}, 'location', 'northwest');

h=figure(3);
FN = findall(h,'-property','FontName');
set(FN, 'FontName', 'Verdana');

print -dpng pregv_vs_pregv_masked.png

