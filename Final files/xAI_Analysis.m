%  Project: Advanced Diagnostics for Power Transformers
%  Author:  Kgosietsile Ntsowe | ID: 11409134
%
%  PURPOSE:
%    Quantifying which features drive each classification decision, 
%    and verifying that the model's internal logic aligns with IEC 60599 
%    fault gas theory.
%  IMPLEMENTATION APPROACH:
%    PFI:  Global explanations — ranks features by overall importance
%  INPUTS:
%    pipeline_outputs/X_train_scaled.csv
%    pipeline_outputs/X_test_scaled.csv
%
%  OUTPUTS → xai_outputs/:
%    global_importance.png
%    per_class_importance.png
%    local_explanation_D1.png
%    local_explanation_T3.png
%    feature_importance_table.csv
%    iec_alignment_table.csv

clc; clear; close all;
rng(42);
%% ---SETUP 
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, 'xai_outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end


FEAT_NAMES  = {'H2','CH4','C2H6','C2H4','C2H2','CO','CO2', ...
               '%CH4','%C2H4','%C2H2'};
FEAT_FULL   = {'Hydrogen (H2)','Methane (CH4)','Ethane (C2H6)', ...
               'Ethylene (C2H4)','Acetylene (C2H2)', ...
               'Carbon Monoxide (CO)','Carbon Dioxide (CO2)', ...
               'Duval %CH4','Duval %C2H4','Duval %C2H2'};
CLASS_NAMES = {'Healthy','D1 Discharge','T1 <300C','T2 300-700C','T3 >700C'};
N_CLASSES   = 5;
N_FEAT      = 10;
N_REPS_PFI  = 50;    % Repetitions for stable PFI estimates

%% ---LOAD DATA
train_data = readmatrix(fullfile(script_dir, 'pipeline_outputs','X_train_smote.csv'));
test_data  = readmatrix(fullfile(script_dir, 'pipeline_outputs','X_test_scaled.csv'));

X_train = train_data(:,1:10);  y_train = train_data(:,11);
X_test  = test_data(:, 1:10);  y_test  = test_data(:, 11);
%% ---TRAIN FINAL MODEL
% Train SVM — best configuration
t_svm   = templateSVM('KernelFunction','rbf', ...
                      'BoxConstraint', 100, ... 
                      'KernelScale', 1/sqrt(0.1), ...
                      'Standardize', false);
svm_mdl = fitcecoc(X_train, y_train, 'Learners', t_svm, ...
                   'ClassNames', 0:4, 'Coding', 'onevsone');

% Verify baseline
y_pred_clean = predict(svm_mdl, X_test);
[~, baseline_f1, ~] = compute_metrics(y_test, y_pred_clean, N_CLASSES);
fprintf('SVM baseline Macro F1 = %.4f\n\n', baseline_f1);
%% ---PERMUTATION FEATURE IMPORTANCE
%  For each feature j:
%    1. Shuffle column j in the test set (break feature-label link)
%    2. Predict and compute Macro F1 on shuffled data
%    3. Importance = baseline_F1 - shuffled_F1
%    4. Repeat N_REPS_PFI times, report mean ± std
fprintf('--- Permutation Feature Importance ---\n');
pfi_means = zeros(N_FEAT, 1);
pfi_stds  = zeros(N_FEAT, 1);

for j = 1:N_FEAT
    rep_drops = zeros(N_REPS_PFI, 1);
    for rep = 1:N_REPS_PFI
        rng(rep * 7 + 42);
        X_shuffled      = X_test;
        X_shuffled(:,j) = X_shuffled(randperm(size(X_test,1)), j);
        y_shuffled_pred = predict(svm_mdl, X_shuffled);
        [~, shuffled_f1, ~] = compute_metrics(y_test, y_shuffled_pred, N_CLASSES);
        rep_drops(rep) = baseline_f1 - shuffled_f1;
    end
    pfi_means(j) = mean(rep_drops);
    pfi_stds(j)  = std(rep_drops);
    fprintf('  %-10s: %.4f +/- %.4f\n', FEAT_NAMES{j}, pfi_means(j), pfi_stds(j));
end

[pfi_sorted, sort_idx] = sort(pfi_means, 'descend');
fprintf('\n  Global ranking:\n');
for rank = 1:N_FEAT
    j = sort_idx(rank);
    fprintf('  %2d. %-25s %.4f\n', rank, FEAT_FULL{j}, pfi_means(j));
end
fprintf('\n');

%% ---PER-CLASS PERMUTATION IMPORTANCE
%  Computes the drop in PER-CLASS RECALL when each feature is
%  shuffled.
class_importance = zeros(N_FEAT, N_CLASSES);

for j = 1:N_FEAT
    for c = 1:N_CLASSES
        rep_drops = zeros(30, 1);
        for rep = 1:30
            rng(rep*7+42);
            X_shuffled      = X_test;
            X_shuffled(:,j) = X_shuffled(randperm(size(X_test,1)), j);
            y_pred_sh = predict(svm_mdl, X_shuffled);
            % Per-class recall
            class_mask = (y_test == (c-1));
            if sum(class_mask) > 0
                baseline_recall  = sum(y_pred_clean(class_mask)==(c-1)) / sum(class_mask);
                shuffled_recall  = sum(y_pred_sh(class_mask)==(c-1)) / sum(class_mask);
                rep_drops(rep)   = baseline_recall - shuffled_recall;
            end
        end
        class_importance(j,c) = mean(rep_drops);
    end
end

% Print per-class table
fprintf('  Per-class recall drop (feature shuffled):\n');
fprintf('  %-8s', 'Feature');
for c = 1:N_CLASSES
    fprintf('  %-10s', CLASS_NAMES{c}(1:min(end,8)));
end
fprintf('\n  %s\n', repmat('-',1,68));

for j = 1:N_FEAT
    fprintf('  %-8s', FEAT_NAMES{j});
    for c = 1:N_CLASSES
        fprintf('  %10.4f', class_importance(j,c));
    end
    fprintf('\n');
end
fprintf('\n');

% Print class-specific top features
fprintf('  Top features per class (IEC 60599 alignment check):\n');
iec_expected = {
    'Healthy', 'All features low — baseline state',     '';
    'D1',      'C2H2 + %%C2H2 — arcing signature',      'IEC 60599 Table A.1';
    'T1',      'CH4 + C2H6 — low-temp oil cracking',    'IEC 60599 Table A.1';
    'T2',      'C2H4 + C2H6 — moderate thermal',        'IEC 60599 Table A.1';
    'T3',      'C2H4 + CH4 — high-temp pyrolysis',      'IEC 60599 Table A.1';
};
for c = 1:N_CLASSES
    [~, top3] = sort(class_importance(:,c), 'descend');
    fprintf('  %s: %s, %s, %s\n', CLASS_NAMES{c}, ...
        FEAT_NAMES{top3(1)}, FEAT_NAMES{top3(2)}, FEAT_NAMES{top3(3)});
    fprintf('    Expected (IEC 60599): %s\n', iec_expected{c,2});
end
fprintf('\n');

%% ---ANALYSIS
% Find representative D1 and T3 correctly classified samples
d1_correct = find(y_test == 1 & y_pred_clean == 1);
t3_correct = find(y_test == 4 & y_pred_clean == 4);
t2_wrong   = find(y_test == 3 & y_pred_clean ~= 3);  % misclassified

fprintf('  Available for local explanation:\n');
fprintf('    D1 correct:      %d samples\n', length(d1_correct));
fprintf('    T3 correct:      %d samples\n', length(t3_correct));
fprintf('    T2 misclassified: %d samples\n\n', length(t2_wrong));

% D1 local explanation — manual perturbation
if ~isempty(d1_correct)
    sample_idx = d1_correct(1);
    x_sample   = X_test(sample_idx, :);
    
    local_impact = zeros(N_FEAT, 1);
    for j = 1:N_FEAT
        x_perturbed    = x_sample;
        x_perturbed(j) = 0;   % Zero out feature
        pred_orig  = predict(svm_mdl, x_sample);
        pred_pert  = predict(svm_mdl, x_perturbed);
        local_impact(j) = double(pred_orig ~= pred_pert);
    end
    
    % Show top impactful features
    [~, top_local] = sort(local_impact, 'descend');
    fprintf('  Features whose removal changes the D1 prediction:\n');
    changed = false;
    for j = top_local'
        if local_impact(j) > 0
            fprintf('    %s — removal changes prediction\n', FEAT_NAMES{j});
            changed = true;
        end
    end
    if ~changed
        fprintf('    No single feature removal changes prediction\n');
        fprintf('    (Model decision is robust — requires multiple features to flip)\n');
    end
    fprintf('\n');
end

%% ---FIGURES
model_color  = [0.85 0.33 0.10];  % SVM red
class_colors = [0.35 0.70 0.35;
                0.85 0.33 0.10;
                0.00 0.45 0.74;
                0.93 0.69 0.13;
                0.49 0.18 0.56];

% --- Figure 1: Global Feature Importance  ---
fig1 = figure('Position', [50 50 750 520]);
barh(1:N_FEAT, pfi_sorted, 'FaceColor', model_color, 'EdgeColor', 'none');
hold on;
% Error bars
for i = 1:N_FEAT
    j = sort_idx(i);
    plot([pfi_sorted(i)-pfi_stds(j), pfi_sorted(i)+pfi_stds(j)], ...
         [i i], 'k-', 'LineWidth', 1.5);
end
set(gca, 'YTick', 1:N_FEAT, ...
         'YTickLabel', FEAT_FULL(sort_idx), ...
         'FontSize', 10);
xlabel('Macro F1 Drop (Permutation Importance)', 'FontSize', 12);
title({'Global Feature Importance — SVM Classifier'; ...
       'Mean F1 drop ± std across 50 permutations'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
xline(0, '-k', 'LineWidth', 0.5);
grid on; box on;
% Annotate top features
for i = 1:3
    text(pfi_sorted(i)+0.008, i, sprintf('%.4f', pfi_sorted(i)), ...
        'FontSize', 9, 'Color', [0.2 0.2 0.2], 'VerticalAlignment', 'middle');
end
saveas(fig1, fullfile(out_dir, 'global_importance.png'));

% --- Figure 2: Per-Class Importance Heatmap ---
fig2 = figure('Position', [100 100 720 480]);
imagesc(class_importance');
colormap(hot); cb = colorbar;
cb.Label.String = 'Recall Drop (Permutation Importance)';
caxis([0 1]);

for j = 1:N_FEAT
    for c = 1:N_CLASSES
        v   = class_importance(j,c);
clr = 'black'; 
        if v < 0.5
            clr = 'white'; 
        end
        
        text(j, c, sprintf('%.2f', v), ...
            'HorizontalAlignment','center','Color',clr, ...
            'FontSize',8,'FontWeight','bold');
    end
end

set(gca, 'XTick',1:N_FEAT, 'XTickLabel', FEAT_NAMES, ...
         'XTickLabelRotation', 35, ...
         'YTick',1:N_CLASSES, 'YTickLabel', CLASS_NAMES, ...
         'FontSize', 10);
xlabel('Feature', 'FontSize', 12);
ylabel('Fault Class', 'FontSize', 12);
title({'Per-Class Feature Importance — SVM'; ...
       'Cell value = drop in class recall when feature is shuffled'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
saveas(fig2, fullfile(out_dir, 'per_class_importance.png'));

% --- Figure 3: Grouped bar — top 5 features per class ---
fig3 = figure('Position', [100 100 950 520]);
top5_global = sort_idx(1:5);
top5_data   = class_importance(top5_global, :)';  % [5 classes x 5 features]

b = bar(top5_data, 'grouped');
for c = 1:N_CLASSES
    b(c).FaceColor = class_colors(c,:);
end

legend(CLASS_NAMES, 'Location', 'northeast', 'FontSize', 9);
set(gca, 'XTick', 1:5, ...
         'XTickLabel', FEAT_FULL(top5_global), ...
         'XTickLabelRotation', 15, 'FontSize', 10);
ylabel('Recall Drop (Permutation Importance)', 'FontSize', 12);
title({'Class-Specific Importance of Top 5 Global Features'; ...
       'Shows which features matter most for each fault type'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
ylim([0 1.05]); grid on; box on;
saveas(fig3, fullfile(out_dir, 'class_feature_importance.png'));

% --- Figure 4: IEC 60599 Alignment Summary ---
fig4 = figure('Position', [100 100 850 480]);
% Radar/spider chart approximation using bar groups
% Shows expected vs actual top feature per class
iec_primary_feat = [3, 10, 3, 4, 4];  % C2H6,  %C2H2, C2H6, C2H4, C2H4
% Indices into FEAT_NAMES for IEC-expected primary features per class
%   Healthy=C2H6(3), D1=%C2H2(10), T1=C2H6(3), T2=C2H4(4), T3=C2H4(4)

actual_ranks = zeros(1, N_CLASSES);
for c = 1:N_CLASSES
    [~, ranked] = sort(class_importance(:,c), 'descend');
    actual_ranks(c) = find(ranked == iec_primary_feat(c), 1);
end

bar_colors_align = zeros(N_CLASSES, 3);
for c = 1:N_CLASSES
    if actual_ranks(c) <= 2
        bar_colors_align(c,:) = [0.35 0.70 0.35];   % green — aligned
    elseif actual_ranks(c) <= 3
        bar_colors_align(c,:) = [0.93 0.69 0.13];   % amber — close
    else
        bar_colors_align(c,:) = [0.85 0.33 0.10];   % red — misaligned
    end
end

b4 = bar(actual_ranks, 'FaceColor', 'flat');
b4.CData = bar_colors_align;
yline(2, '--k', 'Top-2 threshold', 'LineWidth', 1.5, 'FontSize', 9);
set(gca, 'XTickLabel', CLASS_NAMES, 'FontSize', 11);
ylabel('Rank of IEC 60599 Primary Feature', 'FontSize', 12);
title({'IEC 60599 Alignment — Is the Expected Primary Feature Top-Ranked?'; ...
       'Lower rank = better alignment (1 = top feature by model)'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
ylim([0 5]); grid on; box on;
for c = 1:N_CLASSES
    text(c, actual_ranks(c)+0.15, sprintf('Rank %d', actual_ranks(c)), ...
        'HorizontalAlignment','center','FontSize',10,'FontWeight','bold');
end
saveas(fig4, fullfile(out_dir, 'iec_alignment.png'));

%% ---TABLES
rows = {};
for rank = 1:N_FEAT
    j = sort_idx(rank);
    rows(end+1,:) = {rank, FEAT_NAMES{j}, FEAT_FULL{j}, ...
                     pfi_means(j), pfi_stds(j)};
end
fi_table = cell2table(rows, 'VariableNames', ...
    {'Rank','Feature_Code','Feature_Name','PFI_Mean','PFI_Std'});
writetable(fi_table, fullfile(out_dir, 'feature_importance_table.csv'));

% IEC alignment table
iec_rows = {};
for c = 1:N_CLASSES
    [~, ranked] = sort(class_importance(:,c), 'descend');
    top1 = FEAT_NAMES{ranked(1)};
    top2 = FEAT_NAMES{ranked(2)};
    top3 = FEAT_NAMES{ranked(3)};
    iec_rows(end+1,:) = {CLASS_NAMES{c}, top1, top2, top3, ...
                          iec_expected_gas(c), actual_ranks(c)};
end
iec_table = cell2table(iec_rows, 'VariableNames', ...
    {'Class','Top1_Feature','Top2_Feature','Top3_Feature', ...
     'IEC_Expected','IEC_Feature_Rank'});
writetable(iec_table, fullfile(out_dir, 'iec_alignment_table.csv'));
%% ---SUMMARY
fprintf('  GLOBAL FEATURE IMPORTANCE RANKING:\n');
for rank = 1:N_FEAT
    j = sort_idx(rank);
    fprintf('  %2d. %-12s %.4f\n', rank, FEAT_NAMES{j}, pfi_means(j));
end

fprintf('\n  PER-CLASS TOP FEATURES:\n');
for c = 1:N_CLASSES
    [~, top] = sort(class_importance(:,c), 'descend');
    fprintf('  %-14s: %s, %s, %s\n', CLASS_NAMES{c}, ...
        FEAT_NAMES{top(1)}, FEAT_NAMES{top(2)}, FEAT_NAMES{top(3)});
end
%% ---LOCAL FUNCTIONS
function [accuracy, macro_f1, per_class] = compute_metrics(y_true, y_pred, n_classes)
    accuracy  = mean(y_true == y_pred);
    per_class = zeros(n_classes, 3);
    for c = 0:(n_classes-1)
        tp = sum((y_pred==c)&(y_true==c));
        fp = sum((y_pred==c)&(y_true~=c));
        fn = sum((y_pred~=c)&(y_true==c));
        if (tp+fp)==0,p=0;else,p=tp/(tp+fp);end
        if (tp+fn)==0,r=0;else,r=tp/(tp+fn);end
        if (p+r)==0,  f=0;else,f=2*p*r/(p+r);end
        per_class(c+1,:) = [p,r,f];
    end
    macro_f1 = mean(per_class(:,3));
end

function gas_name = iec_expected_gas(class_idx)
    iec_map = {'All low','C2H2 + %%C2H2','C2H6 + CH4','C2H4 + C2H6','C2H4 + CH4'};
    gas_name = iec_map{class_idx};
end
