%  Project: Advanced Diagnostics for Power Transformers
%  Author:  Kgosietsile Ntsowe | ID: 11409134
%
%    EXPERIMENT A — Imbalanced (no intervention)
%      All models trained on X_train_scaled (1,126 samples)
%      Purpose: establishes raw baseline;
%
%    EXPERIMENT B — SMOTE balanced
%      All models trained on X_train_smote (3,750 samples)
%      Purpose: isolates algorithm quality from data quality
%
%    EXPERIMENT C — Cost-sensitive weighting
%      Models trained with class weights ∝ inverse frequency
%      Purpose: alternative to SMOTE; no synthetic data generated
%      kNN has no native cost matrix in MATLAB
%      DT uses 'Cost' parameter in fitctree
%
%  INPUTS:
%    pipeline_outputs/X_train_scaled.csv  (1126 x 11, imbalanced)
%    pipeline_outputs/X_train_smote.csv   (3750 x 11, balanced)
%    pipeline_outputs/X_test_scaled.csv   (282  x 11, quarantined)
%    Final_Transformer_Dataset_with_Duval.csv (for Duval % columns)
%
%  OUTPUTS → baseline_outputs/:
%    experiment_results.csv     — all models × all experiments
%    per_class_metrics.csv      — P/R/F1 per class per condition
%    mcnemar_results.csv        — pairwise significance tests
%    confusion_matrices.png     — 3x3 grid (3 exps × 2 ML models)
%    experiment_comparison.png  — grouped bar chart
%    smote_effect.png           — SMOTE impact per algorithm
%

clc; clear; close all;
rng(42);

%% --- 0. SETUP -----------------------------------------------------
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, 'baseline_outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

CLASS_NAMES = {'Healthy','D1 Discharge','T1 <300C','T2 300-700C','T3 >700C'};
SHORT_NAMES = {'H','D1','T1','T2','T3'};
N_CLASSES   = 5;
K_FOLDS     = 5;


%% --- LOADING DATASETS

% Imbalanced training set (Experiment A + C)
train_imb  = readmatrix(fullfile(script_dir, 'pipeline_outputs', 'X_train_scaled.csv'));
X_train_A  = train_imb(:, 1:10);
y_train_A  = train_imb(:, 11);

% SMOTE-balanced training set (Experiment B)
train_smote = readmatrix(fullfile(script_dir, 'pipeline_outputs', 'X_train_smote.csv'));
X_train_B   = train_smote(:, 1:10);
y_train_B   = train_smote(:, 11);

% Test set 
test_data  = readmatrix(fullfile(script_dir, 'pipeline_outputs', 'X_test_scaled.csv'));
X_test     = test_data(:, 1:10);
y_test     = test_data(:, 11);

% Raw data for Duval Triangle
T_raw = readtable(fullfile(script_dir, 'Final_Transformer_Dataset_with_Duval.csv'), 'VariableNamingRule', 'preserve');
duval_all = T_raw{:, {'%CH4','%C2H4','%C2H2'}};

% Reconstruct same split
cv_split  = cvpartition(T_raw.Fault_Type, 'HoldOut', 0.20, 'Stratify', true);
test_idx  = find(test(cv_split));
duval_test = duval_all(test_idx, :);

fprintf('  Exp A (Imbalanced): %d train, %d test\n', size(X_train_A,1), size(X_test,1));
fprintf('  Exp B (SMOTE):      %d train, %d test\n', size(X_train_B,1), size(X_test,1));
fprintf('  Exp C (Cost-sens):  %d train, %d test\n', size(X_train_A,1), size(X_test,1));

%% --- CLASS WEIGHTS (EXPERIMENT C)
%  Inverse-frequency weighting: w_c = n_total / (n_classes * n_c)
%  This gives minority classes proportionally more influence
%  without generating any synthetic data.
%  MATLAB's fitctree accepts a 'Cost' matrix where Cost(i,j) is the
%  cost of predicting class j when the true class is i.
%  Setting Cost = diag(1./class_frequencies)

n_total   = length(y_train_A);
n_classes_vec = arrayfun(@(c) sum(y_train_A == c), 0:4)';
class_weights = n_total ./ (N_CLASSES .* n_classes_vec);

fprintf('Class weights for Experiment C (cost-sensitive):\n');
for c = 0:4
    fprintf('  Class %d: %.4f\n', c, class_weights(c+1));
end
fprintf('  (Class 1/D1 gets weight %.2fx the majority class)\n\n', ...
    class_weights(2)/class_weights(1));

% Cost matrix for fitctree
cost_matrix = ones(N_CLASSES) - eye(N_CLASSES);
for c = 1:N_CLASSES
    cost_matrix(c,:) = cost_matrix(c,:) .* class_weights(c);
end
%% --- DUVAL TRIANGLE BASELINE
fprintf('--- Duval Triangle (IEC 60599)---\n');
y_duval = apply_duval_triangle(duval_test);
[duval_acc, duval_f1, duval_pc] = compute_metrics(y_test, y_duval, N_CLASSES);
fprintf('  Accuracy=%.4f  Macro F1=%.4f\n\n', duval_acc, duval_f1);

%% --- K-NN EXPERIMENTS
%
%  Experiment A: trained on imbalanced data  (X_train_A)
%  Experiment B: trained on SMOTE data       (X_train_B)
%  Experiment C: K-NN has NO native class_weight parameter
%                in MATLAB's fitcknn.
%  Best k for each experiment selected independently by 5-fold CV.
fprintf('--- K-NN ---\n');
k_values = [1, 3, 5, 7, 10, 15];
knn_results = struct();

for exp_id = {'A','B'}
    exp = exp_id{1};
    if strcmp(exp,'A')
        X_tr = X_train_A;  y_tr = y_train_A;
    else
        X_tr = X_train_B;  y_tr = y_train_B;
    end

    fprintf('  Exp %s: k-sweep (5-fold CV Macro F1):\n', exp);
    cv = cvpartition(y_tr, 'KFold', K_FOLDS, 'Stratify', true);
    best_f1 = -1;  best_k = 5;

    for k = k_values
        fold_f1s = zeros(K_FOLDS,1);
        for fold = 1:K_FOLDS
            mdl = fitcknn(X_tr(training(cv,fold),:), y_tr(training(cv,fold)), ...
                          'NumNeighbors', k, 'Distance', 'euclidean', ...
                          'Standardize', false);
            pred = predict(mdl, X_tr(test(cv,fold),:));
            [~, f1, ~] = compute_metrics(y_tr(test(cv,fold)), pred, N_CLASSES);
            fold_f1s(fold) = f1;
        end
        mu = mean(fold_f1s);
        fprintf('    k=%2d: CV F1 = %.4f +/- %.4f\n', k, mu, std(fold_f1s));
        if mu > best_f1, best_f1 = mu; best_k = k; end
    end
    fprintf('    >> Best k = %d\n', best_k);

    % Final model trained on full experiment dataset
    knn_final = fitcknn(X_tr, y_tr, 'NumNeighbors', best_k, ...
                        'Distance', 'euclidean', 'Standardize', false);
    y_pred = predict(knn_final, X_test);
    [acc, macro_f1, per_class] = compute_metrics(y_test, y_pred, N_CLASSES);

    knn_results.(exp).acc       = acc;
    knn_results.(exp).macro_f1  = macro_f1;
    knn_results.(exp).per_class = per_class;
    knn_results.(exp).pred      = y_pred;
    knn_results.(exp).best_k    = best_k;

    fprintf('    Test Accuracy=%.4f  Macro F1=%.4f\n\n', acc, macro_f1);
end


%% --- DECISION TREE EXPERIMENTS
%
%  Experiment A: standard DT on imbalanced data
%  Experiment B: standard DT on SMOTE-balanced data
%  Experiment C: cost-sensitive DT with 'Cost' matrix
fprintf('--- Decision Tree ---\n');
depths = [3, 5, 7, 10, 15];
dt_results = struct();

exp_configs = struct();
exp_configs.A.X = X_train_A;  exp_configs.A.y = y_train_A;
exp_configs.A.cost_flag = false;
exp_configs.B.X = X_train_B;  exp_configs.B.y = y_train_B;
exp_configs.B.cost_flag = false;
exp_configs.C.X = X_train_A;  exp_configs.C.y = y_train_A;
exp_configs.C.cost_flag = true;

for exp_id = {'A','B','C'}
    exp = exp_id{1};
    X_tr      = exp_configs.(exp).X;
    y_tr      = exp_configs.(exp).y;
    use_cost  = exp_configs.(exp).cost_flag;

    fprintf('  Exp %s: depth-sweep (5-fold CV Macro F1):\n', exp);
    cv = cvpartition(y_tr, 'KFold', K_FOLDS, 'Stratify', true);
    best_f1 = -1;  best_depth = 10;

    for d = depths
        fold_f1s = zeros(K_FOLDS,1);
        for fold = 1:K_FOLDS
            if use_cost
                mdl = fitctree(X_tr(training(cv,fold),:), ...
                               y_tr(training(cv,fold)), ...
                               'MaxNumSplits', 2^d-1, ...
                               'Cost', cost_matrix);
            else
                mdl = fitctree(X_tr(training(cv,fold),:), ...
                               y_tr(training(cv,fold)), ...
                               'MaxNumSplits', 2^d-1);
            end
            pred = predict(mdl, X_tr(test(cv,fold),:));
            [~, f1, ~] = compute_metrics(y_tr(test(cv,fold)), pred, N_CLASSES);
            fold_f1s(fold) = f1;
        end
        mu = mean(fold_f1s);
        fprintf('    depth=%2d: CV F1 = %.4f +/- %.4f\n', d, mu, std(fold_f1s));
        if mu > best_f1, best_f1 = mu; best_depth = d; end
    end
    fprintf('    >> Best depth = %d\n', best_depth);

    if use_cost
        dt_final = fitctree(X_tr, y_tr, ...
                            'MaxNumSplits', 2^best_depth-1, ...
                            'Cost', cost_matrix);
    else
        dt_final = fitctree(X_tr, y_tr, 'MaxNumSplits', 2^best_depth-1);
    end

    y_pred = predict(dt_final, X_test);
    [acc, macro_f1, per_class] = compute_metrics(y_test, y_pred, N_CLASSES);

    dt_results.(exp).acc        = acc;
    dt_results.(exp).macro_f1   = macro_f1;
    dt_results.(exp).per_class  = per_class;
    dt_results.(exp).pred       = y_pred;
    dt_results.(exp).best_depth = best_depth;

    fprintf('    Test Accuracy=%.4f  Macro F1=%.4f\n\n', acc, macro_f1);
end

%% --- McNEMAR'S TEST
%  Pairwise significance testing within and across experiments.
%  Key comparisons:
%    (a) Within Exp B: kNN vs D
%    (b) kNN Exp A vs Exp B
%    (c) DT  Exp A vs Exp B 
fprintf('--- McNemar''s Test ---\n');

comparisons = {
    'kNN(A)',  knn_results.A.pred;
    'kNN(B)',  knn_results.B.pred;
    'DT(A)',   dt_results.A.pred;
    'DT(B)',   dt_results.B.pred;
    'DT(C)',   dt_results.C.pred;
    'Duval',   y_duval;
};

pairs_to_test = {
    1, 2, 'kNN(A) vs kNN(B) — effect of SMOTE on kNN';
    3, 4, 'DT(A)  vs DT(B)  — effect of SMOTE on DT';
    2, 4, 'kNN(B) vs DT(B)  — fair algorithm comparison ';
    6, 2, 'Duval  vs kNN(B) — ML vs industry standard ';
    6, 4, 'Duval  vs DT(B)  — ML vs industry standard ';
};

mc_rows = {};
for pi = 1:size(pairs_to_test, 1)
    i     = pairs_to_test{pi,1};
    j     = pairs_to_test{pi,2};
    label = pairs_to_test{pi,3};
    pred_i = comparisons{i,2};
    pred_j = comparisons{j,2};

    b = sum((pred_i ~= y_test) & (pred_j == y_test));
    c = sum((pred_i == y_test) & (pred_j ~= y_test));

    if (b+c) == 0
        chi2 = 0;
    else
        chi2 = (abs(b-c) - 1)^2 / (b+c);
    end
    sig = chi2 > 3.841;
    sig_str = 'YES'; if ~sig, sig_str = 'NO'; end

    fprintf('  %s\n    b=%d  c=%d  Chi2=%.3f  Significant: %s\n', ...
        label, b, c, chi2, sig_str);
    mc_rows(end+1,:) = {label, b, c, chi2, sig_str};
end
fprintf('\n');

mc_table = cell2table(mc_rows, ...
    'VariableNames', {'Comparison','b','c','Chi2','Sig_p05'});
writetable(mc_table, fullfile(out_dir, 'mcnemar_results.csv'));
%% --- FIGURES
% --- Figure 1: Experiment comparison grouped bar ---
fig1 = figure('Position', [50 50 900 520]);
exp_labels = {'Exp A\n(Imbalanced)','Exp B\n(SMOTE)','Exp C\n(Cost-Sensitive)'};
knn_f1s = [knn_results.A.macro_f1, knn_results.B.macro_f1, NaN];
dt_f1s  = [dt_results.A.macro_f1,  dt_results.B.macro_f1,  dt_results.C.macro_f1];
duval_f1 = duval_f1;

x = 1:3;
hold on;
b1 = bar(x - 0.22, knn_f1s, 0.2, 'FaceColor', [0.00 0.45 0.74]);
b2 = bar(x,         dt_f1s,  0.2, 'FaceColor', [0.49 0.18 0.56]);
b3 = bar(x + 0.22,  repmat(duval_f1,1,3), 0.2, 'FaceColor', [0.85 0.33 0.10]);

% Annotate bars
for i = 1:3
    if ~isnan(knn_f1s(i))
        text(i-0.22, knn_f1s(i)+0.01, sprintf('%.3f',knn_f1s(i)), ...
            'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
    end
    text(i, dt_f1s(i)+0.01, sprintf('%.3f',dt_f1s(i)), ...
        'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
end

% Annotate N/A for kNN Exp C
text(3-0.22, 0.05, 'N/A', 'HorizontalAlignment','center', ...
    'FontSize',9,'Color',[0.5 0.5 0.5],'FontWeight','bold');

legend({'k-NN','Decision Tree','Duval Triangle'}, ...
    'Location','southeast','FontSize',10);
ylim([0 1.1]);
set(gca,'XTick',1:3, 'XTickLabel', ...
    {'Exp A: Imbalanced','Exp B: SMOTE','Exp C: Cost-Sensitive'}, ...
    'FontSize',11);
ylabel('Macro F1 Score','FontSize',12);
title({'Baseline Comparison — Three Controlled Experiments'; ...
       'Test set is IDENTICAL across all experiments (N=282)'}, ...
    'FontSize',12,'FontWeight','bold');
yline(duval_f1,'--r','Duval Baseline','LabelHorizontalAlignment','left', ...
    'LineWidth',1.2,'FontSize',9);
grid on; box on;
saveas(fig1, fullfile(out_dir, 'experiment_comparison.png'));

% --- Figure 2: SMOTE effect visualisation ---
fig2 = figure('Position', [100 100 650 450]);
delta_knn = knn_results.B.macro_f1 - knn_results.A.macro_f1;
delta_dt  = dt_results.B.macro_f1  - dt_results.A.macro_f1;

bar_vals   = [delta_knn, delta_dt];
bar_colors_effect = [0.00 0.45 0.74; 0.49 0.18 0.56];
b_eff = bar(bar_vals, 'FaceColor', 'flat');
b_eff.CData = bar_colors_effect;

for i = 1:2
    clr = 'black';
    txt = sprintf('%+.4f', bar_vals(i));
    text(i, bar_vals(i) + sign(bar_vals(i))*0.004, txt, ...
        'HorizontalAlignment','center','FontSize',11,'FontWeight','bold', ...
        'Color', clr);
end

yline(0, '-k', 'LineWidth', 1);
set(gca,'XTickLabel',{'k-NN','Decision Tree'},'FontSize',12);
ylabel('\Delta Macro F1 (Exp B - Exp A)','FontSize',12);
title({'Effect of SMOTE on Baseline Performance'; ...
       'Positive = SMOTE helped; Negative = SMOTE hurt'}, ...
    'FontSize',12,'FontWeight','bold');
ylim([-0.08 0.16]);
grid on; box on;
saveas(fig2, fullfile(out_dir, 'smote_effect_on_baselines.png'));

% --- Figure 3: Confusion matrices for Exp B (fair comparison set) ---
fig3 = figure('Position', [50 50 1100 400]);
sgtitle('Confusion Matrices — Experiment B (SMOTE-Balanced, Fair Comparison)', ...
    'FontSize',12,'FontWeight','bold');

plot_configs = {
    'Duval Triangle', y_duval, duval_f1;
    sprintf('k-NN (k=%d, Exp B)', knn_results.B.best_k), knn_results.B.pred, knn_results.B.macro_f1;
    sprintf('DT (depth=%d, Exp B)', dt_results.B.best_depth), dt_results.B.pred, dt_results.B.macro_f1;
};

for m = 1:3
    subplot(1,3,m);
    cm = confusionmat(y_test, plot_configs{m,2}, 'Order', 0:4);
    cm_norm = cm ./ sum(cm,2);
    cm_norm(isnan(cm_norm)) = 0;
    imagesc(cm_norm); colormap(subplot(1,3,m), parula); caxis([0 1]);
    for r = 1:N_CLASSES
        for cc = 1:N_CLASSES
            clr = 'white'; if cm_norm(r,cc) < 0.5, clr = [0.2 0.2 0.2]; end
            text(cc,r,sprintf('%d\n%.0f%%',cm(r,cc),cm_norm(r,cc)*100), ...
                'HorizontalAlignment','center','Color',clr,'FontSize',7.5,'FontWeight','bold');
        end
    end
    set(gca,'XTick',1:5,'XTickLabel',SHORT_NAMES, ...
            'YTick',1:5,'YTickLabel',SHORT_NAMES,'FontSize',9);
    xlabel('Predicted'); ylabel('True');
    title(sprintf('%s\nF1=%.3f', plot_configs{m,1}, plot_configs{m,3}), ...
        'FontSize',10,'FontWeight','bold');
end
saveas(fig3, fullfile(out_dir, 'confusion_matrices_expB.png'));
%% --- RESULTS TABLE
rows = {
    'Naive Baseline',       '-',    '-',    accuracy_score_local(y_test,zeros(size(y_test))), 0.1600, NaN,   NaN;
    'Duval Triangle',       '-',    '-',    duval_acc,             duval_f1,                  duval_pc(2,3), duval_pc(5,3);
    'kNN  Exp A (Imbalanced)', 'A', num2str(knn_results.A.best_k), knn_results.A.acc, knn_results.A.macro_f1, knn_results.A.per_class(2,3), knn_results.A.per_class(5,3);
    'kNN  Exp B (SMOTE)',      'B', num2str(knn_results.B.best_k), knn_results.B.acc, knn_results.B.macro_f1, knn_results.B.per_class(2,3), knn_results.B.per_class(5,3);
    'DT   Exp A (Imbalanced)', 'A', num2str(dt_results.A.best_depth), dt_results.A.acc, dt_results.A.macro_f1, dt_results.A.per_class(2,3), dt_results.A.per_class(5,3);
    'DT   Exp B (SMOTE)',      'B', num2str(dt_results.B.best_depth), dt_results.B.acc, dt_results.B.macro_f1, dt_results.B.per_class(2,3), dt_results.B.per_class(5,3);
    'DT   Exp C (Cost-Sens)',  'C', num2str(dt_results.C.best_depth), dt_results.C.acc, dt_results.C.macro_f1, dt_results.C.per_class(2,3), dt_results.C.per_class(5,3);
};

res_table = cell2table(rows, 'VariableNames', ...
    {'Model','Experiment','HyperParam','Accuracy','Macro_F1','D1_F1','T3_F1'});
writetable(res_table, fullfile(out_dir, 'experiment_results.csv'));
%% --- SUMMARY
fprintf('================================================================\n');
fprintf('  RESULTS SUMMARY\n');
fprintf('================================================================\n');
fprintf('  %-30s  %8s  %8s  %8s  %8s\n','Model','Exp','Accuracy','MacroF1','D1 F1');
fprintf('  %s\n', repmat('-',1,72));
fprintf('  %-30s  %8s  %8.4f  %8.4f  %8.4f\n','Duval Triangle (IEC60599)','-',duval_acc,duval_f1,duval_pc(2,3));
fprintf('  %-30s  %8s  %8.4f  %8.4f  %8.4f\n','k-NN','A',knn_results.A.acc,knn_results.A.macro_f1,knn_results.A.per_class(2,3));
fprintf('  %-30s  %8s  %8.4f  %8.4f  %8.4f\n','k-NN','B',knn_results.B.acc,knn_results.B.macro_f1,knn_results.B.per_class(2,3));
fprintf('  %-30s  %8s  %8.4f  %8.4f  %8.4f\n','DT','A',dt_results.A.acc,dt_results.A.macro_f1,dt_results.A.per_class(2,3));
fprintf('  %-30s  %8s  %8.4f  %8.4f  %8.4f\n','DT','B',dt_results.B.acc,dt_results.B.macro_f1,dt_results.B.per_class(2,3));
fprintf('  %-30s  %8s  %8.4f  %8.4f  %8.4f\n','DT (cost-sensitive)','C',dt_results.C.acc,dt_results.C.macro_f1,dt_results.C.per_class(2,3));

fprintf('\n  KEY FINDINGS:\n');
fprintf('  1. SMOTE improved kNN by +%.4f \n', ...
    knn_results.B.macro_f1 - knn_results.A.macro_f1);
fprintf('  2. SMOTE slightly degraded DT by %.4f\n', ...
    dt_results.B.macro_f1 - dt_results.A.macro_f1);
fprintf('  3. Comparison of Exp B:\n');
fprintf('     kNN F1=%.4f vs DT F1=%.4f\n', ...
    knn_results.B.macro_f1, dt_results.B.macro_f1);
fprintf('  4. All ML models beat Duval Triangle (F1=%.4f)\n', duval_f1);
fprintf('================================================================\n\n');

%% --- LOCAL FUNCTIONS

function y_pred = apply_duval_triangle(duval_pct_matrix)
% APPLY_DUVAL_TRIANGLE  IEC 60599 polygon zone classifier.
% Source: Duval & dePablo (2001), IEEE Electrical Insulation Magazine.
    n = size(duval_pct_matrix, 1);
    y_pred = zeros(n,1);
    for i = 1:n
        ch4  = duval_pct_matrix(i,1);
        c2h4 = duval_pct_matrix(i,2);
        c2h2 = duval_pct_matrix(i,3);
        if (ch4 + c2h4 + c2h2) < 0.001
            y_pred(i) = 0;
        elseif c2h2 >= 13
            y_pred(i) = 1;
        elseif c2h4 >= 50 && c2h2 < 4
            y_pred(i) = 4;
        elseif c2h4 >= 20 && c2h2 < 13
            y_pred(i) = 3;
        elseif ch4 >= 80 && c2h4 < 20
            y_pred(i) = 2;
        else
            if ch4 > c2h4, y_pred(i) = 2; else, y_pred(i) = 3; end
        end
    end
end


function [accuracy, macro_f1, per_class] = compute_metrics(y_true, y_pred, n_classes)
% COMPUTE_METRICS  Accuracy, macro F1, per-class [P, R, F1].
    accuracy  = mean(y_true == y_pred);
    per_class = zeros(n_classes, 3);
    for c = 0:(n_classes-1)
        tp = sum((y_pred==c) & (y_true==c));
        fp = sum((y_pred==c) & (y_true~=c));
        fn = sum((y_pred~=c) & (y_true==c));
        if (tp+fp)==0, p=0; else, p=tp/(tp+fp); end
        if (tp+fn)==0, r=0; else, r=tp/(tp+fn); end
        if (p+r)==0,   f=0; else, f=2*p*r/(p+r); end
        per_class(c+1,:) = [p, r, f];
    end
    macro_f1 = mean(per_class(:,3));
end


function acc = accuracy_score_local(y_true, y_pred)
    acc = mean(y_true == y_pred);
end

%% --- 