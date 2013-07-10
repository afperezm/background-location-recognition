% Evaluate performance of the feature selection algorithm comparing performance curves for a successful and an unsuccessful case

[recall_rates, precision_rates] = compute_performance_rates(1, 50, 'occurrence_matrix_christ_church_000179_pregv.txt');
precision_rates(end)
[recall_rates, precision_rates] = compute_performance_rates(1, 50, 'occurrence_matrix_christ_church_000179_featsel.txt');
precision_rates(end)

[recall_rates, precision_rates] = compute_performance_rates(1, 50, 'occurrence_matrix_radcliffe_camera_000519_pregv.txt');
precision_rates(end)
[recall_rates, precision_rates] = compute_performance_rates(1, 50, 'occurrence_matrix_radcliffe_camera_000519_featsel.txt');
precision_rates(end)

figure(3), mask_reader('./queries_with_keys/radcliffe_camera_000519_with_keys.jpg', './queries_masks/radcliffe_camera_000519.mask');
h=figure(3);
FN = findall(h,'-property','LineWidth');
set(FN, 'LineWidth', 4);
print -dpng feature_selection_success.png

figure(4), mask_reader('./queries_with_keys/christ_church_000179_with_keys.jpg', './queries_masks/christ_church_000179.mask');
h=figure(4);
FN = findall(h,'-property','LineWidth');
set(FN, 'LineWidth', 4);
print -dpng feature_selection_failure.png

