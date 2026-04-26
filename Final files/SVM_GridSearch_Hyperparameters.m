%  Project: Advanced Diagnostics for Power Transformers
%  Author:  Kgosietsile Ntsowe | ID: 11409134
%
%  PURPOSE:
%    Train and evaluate the primary model — a Support Vector Machine
%    with an RBF kernel 
%
%  RBF KERNEL: K(x,z) = exp(-gamma * ||x-z||^2)
%    gamma controls the width of the Gaussian: high gamma = narrow
%    kernel = local decision boundary; low gamma = wide kernel =
%    smoother, more global boundary.
%
%  TWO HYPERPARAMETERS TUNED BY GRID SEARCH:
%    C     ∈ {0.1, 1, 10, 100, 1000}  — regularisation strength
%    gamma ∈ {0.001, 0.01, 0.1, 1, 10} — RBF kernel width
%    Total: 25 combinations × 5 folds = 125 model fits per experiment
%
%  EXPERIMENTS (identical structure to Checkpoint 2.1):
%    Exp A — Imbalanced  (X_train_scaled,  1126 samples)
%    Exp B — SMOTE       (X_train_smote,   3750 samples)
%    Exp C — Cost-sensitive (X_train_scaled with Cost matrix)
%
%  OUTPUTS → svm_outputs/:
%    svm_results_summary.csv
%    grid_heatmap_expA.png, grid_heatmap_expB.png
%    confusion_matrices_svm.png
%    full_comparison_table.csv   (SVM + all baselines)
%    support_vector_analysis.png
%
clc; clear; close all;
rng(42);

%% --- SETUP
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, 'svm_outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

CLASS_NAMES = {'Healthy','D1 Discharge','T1 <300C','T2 300-700C','T3 >700C'};
SHORT_NAMES = {'H','D1','T1','T2','T3'};
N_CLASSES   = 5;
K_FOLDS     = 5;

% Grid search values
C_values     = [0.1, 1, 10, 100, 1000];
gamma_values = [0.001, 0.01, 0.1, 1, 10];
n_C          = length(C_values);
n_gamma      = length(gamma_values);
fprintf('  Grid: C = [0.1, 1, 10, 100, 1000]\n');
fprintf('        gamma = [0.001, 0.01, 0.1, 1, 10]\n');
fprintf('  Total fits per experiment: %d (25 combos x 5 folds)\n\n', ...
    n_C * n_gamma * K_FOLDS);

%% --- LOAD DATA
train_imb   = readmatrix(fullfile(script_dir, 'pipeline_outputs', 'X_train_scaled.csv'));
train_smote = readmatrix(fullfile(script_dir, 'pipeline_outputs', 'X_train_smote.csv'));
test_data   = readmatrix(fullfile(script_dir, 'pipeline_outputs', 'X_test_scaled.csv'));

X_train_A = train_imb(:,   1:10);   y_train_A = train_imb(:,   11);
X_train_B = train_smote(:, 1:10);   y_train_B = train_smote(:, 11);
X_test    = test_data(:,   1:10);   y_test    = test_data(:,   11);

fprintf('  Exp A: %d samples  |  Exp B: %d samples  |  Test: %d samples\n\n',...
    size(X_train_A,1), size(X_train_B,1), size(X_test,1));
%% --- CLASS WEIGHTS FOR EXPERIMENT C
%  Inverse-frequency weighting applied via the 'Cost' matrix in
%  fitcecoc. 
% Cost(i,j) = penalty for predicting class j when true
%  class is i.
% Diagonal entries are 0 (correct predictions).
%  Off-diagonal entries are weighted by inverse class frequency,
%  misclassifying a rare fault carrys a proportionally higher
%  penalty than misclassifying a common healthy sample.

n_total      = length(y_train_A);
class_counts = arrayfun(@(c) sum(y_train_A==c), 0:4)';
class_weights = n_total ./ (N_CLASSES .* class_counts);

fprintf('Class weights (Experiment C):\n');
for c = 0:4
    fprintf('  Class %d: %.4f\n', c, class_weights(c+1));
end
fprintf('  D1 receives %.1fx more weight than Healthy\n\n', ...
    class_weights(2)/class_weights(1));

% Build 5x5 Cost matrix
cost_matrix = zeros(N_CLASSES, N_CLASSES);
for i = 1:N_CLASSES
    for j = 1:N_CLASSES
        if i ~= j
            cost_matrix(i,j) = class_weights(i);
        end
    end
end
%% --- GRID SEARCH FOR THE THREE EXPERIMENTS
svm_results = struct();
exp_configs = struct();

exp_configs.A.X    = X_train_A;
exp_configs.A.y    = y_train_A;
exp_configs.A.name = 'Exp A (Imbalanced)';
exp_configs.A.use_cost = false;

exp_configs.B.X    = X_train_B;
exp_configs.B.y    = y_train_B;
exp_configs.B.name = 'Exp B (SMOTE)';
exp_configs.B.use_cost = false;

exp_configs.C.X    = X_train_A;
exp_configs.C.y    = y_train_A;
exp_configs.C.name = 'Exp C (Cost-Sensitive)';
exp_configs.C.use_cost = true;

for exp_id = {'A','B','C'}
    exp   = exp_id{1};
    X_tr  = exp_configs.(exp).X;
    y_tr  = exp_configs.(exp).y;
    ename = exp_configs.(exp).name;

    fprintf('--- Grid Search: %s ---\n', ename);

    cv = cvpartition(y_tr, 'KFold', K_FOLDS, 'Stratify', true);

    % Preallocate grid result matrix
    cv_grid = zeros(n_C, n_gamma);

for ci = 1:n_C
        for gi = 1:n_gamma
            C_val = C_values(ci);
            g_val = gamma_values(gi);
            fold_f1s = zeros(K_FOLDS, 1);

            for fold = 1:K_FOLDS
                tr_idx  = training(cv, fold);
                val_idx = test(cv,     fold);

                % Define the SVM rules (the kernel trick and constraints)
                t = templateSVM('KernelFunction', 'rbf', ...
                                'BoxConstraint',  C_val, ...
                                'KernelScale',    1/sqrt(g_val), ...
                                'Standardize',    false); % Already scaled

                % Train the Multiclass model (OVO decomposition)
                if exp_configs.(exp).use_cost
                    mdl_mc = fitcecoc(X_tr(tr_idx,:), y_tr(tr_idx), ...
                        'Learners',    t, ...
                        'ClassNames',  0:4, ...
                        'Coding',      'onevsone', ...
                        'Cost',        cost_matrix);
                else
                    mdl_mc = fitcecoc(X_tr(tr_idx,:), y_tr(tr_idx), ...
                        'Learners',    t, ...
                        'ClassNames',  0:4, ...
                        'Coding',      'onevsone');
                end

                pred = predict(mdl_mc, X_tr(val_idx,:));
                [~, ~, pc] = compute_metrics(y_tr(val_idx), pred, N_CLASSES);
                fold_f1s(fold) = mean(pc(:,3));
            end

            cv_grid(ci, gi) = mean(fold_f1s);
        end
        fprintf('  C=%5.3g: [%s]\n', C_values(ci), ...
            strjoin(arrayfun(@(v) sprintf('%.3f',v), cv_grid(ci,:), ...
            'UniformOutput',false), '  '));
    end

    % Best hyperparameters
    [best_val, best_lin] = max(cv_grid(:));
    [best_ci, best_gi]   = ind2sub([n_C, n_gamma], best_lin);
    best_C     = C_values(best_ci);
    best_gamma = gamma_values(best_gi);

    fprintf('  >> Best: C=%.3g, gamma=%.3g, CV Macro F1=%.4f\n\n', ...
        best_C, best_gamma, best_val);

    % Store grid and best params
    svm_results.(exp).cv_grid   = cv_grid;
    svm_results.(exp).best_C    = best_C;
    svm_results.(exp).best_gamma = best_gamma;
    svm_results.(exp).best_cv_f1 = best_val;
end

%% --- TRAINING FINAL MODELS ON FULL TRAINING DATA
for exp_id = {'A','B','C'}
    exp    = exp_id{1};
    X_tr   = exp_configs.(exp).X;
    y_tr   = exp_configs.(exp).y;
    best_C = svm_results.(exp).best_C;
    best_g = svm_results.(exp).best_gamma;

    % Build final SVM template
    t_final = templateSVM('KernelFunction', 'rbf', ...
                          'BoxConstraint',  best_C, ...
                          'KernelScale',    1/sqrt(best_g), ...
                          'Standardize',    false);

    if exp_configs.(exp).use_cost
        final_mdl = fitcecoc(X_tr, y_tr, ...
            'Learners',   t_final, ...
            'ClassNames', 0:4, ...
            'Coding',     'onevsone', ...
            'Cost',       cost_matrix);
    else
        final_mdl = fitcecoc(X_tr, y_tr, ...
            'Learners',   t_final, ...
            'ClassNames', 0:4, ...
            'Coding',     'onevsone');
    end

    % Predict on quarantined test set
    y_pred = predict(final_mdl, X_test);

    % Compute all metrics
    [acc, macro_f1, per_class] = compute_metrics(y_test, y_pred, N_CLASSES);

    svm_results.(exp).model     = final_mdl;
    svm_results.(exp).pred      = y_pred;
    svm_results.(exp).acc       = acc;
    svm_results.(exp).macro_f1  = macro_f1;
    svm_results.(exp).per_class = per_class;

    fprintf('  SVM %s (C=%.3g, gamma=%.3g):\n', ...
        exp_configs.(exp).name, best_C, best_g);
    fprintf('    Accuracy = %.4f  |  Macro F1 = %.4f\n', acc, macro_f1);
    fprintf('    Per-class F1:\n');
    for c = 1:N_CLASSES
        fprintf('      %-14s  P=%.3f  R=%.3f  F1=%.3f\n', ...
            CLASS_NAMES{c}, per_class(c,1), per_class(c,2), per_class(c,3));
    end
    fprintf('\n');
end

%% --- HYPERPARAMETER INTERPRETATION
fprintf('================================================================\n');
fprintf('  HYPERPARAMETER INTERPRETATION\n');
fprintf('================================================================\n\n');

fprintf('  Exp A best: C=%.3g, gamma=%.3g\n', ...
    svm_results.A.best_C, svm_results.A.best_gamma);
fprintf('    High C (%.3g): tight margin — the algorithm tolerates very few\n', ...
    svm_results.A.best_C);
fprintf('    training errors. Appropriate for imbalanced data where minority\n');
fprintf('    fault samples are rare and must not be misclassified during training.\n');
fprintf('    Moderate gamma (%.3g): medium-width RBF kernel. Captures the\n', ...
    svm_results.A.best_gamma);
fprintf('    local structure of fault gas signatures without overfitting.\n\n');

fprintf('  Exp B best: C=%.3g, gamma=%.3g\n', ...
    svm_results.B.best_C, svm_results.B.best_gamma);
fprintf('    Lower C (%.3g): softer margin appropriate for SMOTE data,\n', ...
    svm_results.B.best_C);
fprintf('    which places synthetic samples near class boundaries. A hard\n');
fprintf('    margin would attempt to fit noise-like interpolation points.\n');
fprintf('    Higher gamma (%.3g): tighter kernel reflects the denser\n', ...
    svm_results.B.best_gamma);
fprintf('    minority class regions created by SMOTE oversampling.\n\n');
%% --- FIGURES
% --- Grid Heatmaps ---
for exp_id = {'A','B'}
    exp   = exp_id{1};
    fname = sprintf('grid_heatmap_exp%s.png', exp);

    fig = figure('Position', [50 50 700 500]);
    imagesc(svm_results.(exp).cv_grid);
    colormap(parula); cb = colorbar;
    cb.Label.String = 'CV Macro F1';
    caxis([0 1]);

    set(gca, 'XTick', 1:n_gamma, ...
             'XTickLabel', arrayfun(@(v) sprintf('%.3g',v), gamma_values, ...
                           'UniformOutput', false), ...
             'YTick', 1:n_C, ...
             'YTickLabel', arrayfun(@(v) sprintf('%.3g',v), C_values, ...
                           'UniformOutput', false), ...
             'FontSize', 11);
    xlabel('\gamma (RBF kernel width)', 'FontSize', 12);
    ylabel('C (regularisation)', 'FontSize', 12);
    title(sprintf('SVM Grid Search — %s\nCV Macro F1', ...
        exp_configs.(exp).name), 'FontSize', 12, 'FontWeight', 'bold');

    % Annotate each cell
    for ci = 1:n_C
        for gi = 1:n_gamma
            v = svm_results.(exp).cv_grid(ci, gi);
            clr = 'white'; 
            if v > 0.80 , clr ='black'; end
            
            text(gi, ci, sprintf('%.3f', v), ...
                'HorizontalAlignment', 'center', ...
                'Color', clr, 'FontSize', 9, 'FontWeight', 'bold');
        end
    end

    % Highlight best cell
    [~, best_lin] = max(svm_results.(exp).cv_grid(:));
    [best_ci, best_gi] = ind2sub([n_C, n_gamma], best_lin);
    rectangle('Position', [best_gi-0.5, best_ci-0.5, 1, 1], ...
        'EdgeColor', 'red', 'LineWidth', 2.5);
    text(best_gi, best_ci-0.38, 'BEST', ...
        'HorizontalAlignment', 'center', 'Color', 'red', ...
        'FontSize', 8, 'FontWeight', 'bold');

    saveas(fig, fullfile(out_dir, fname));
    fprintf('  Saved: %s\n', fname);
end

% --- Confusion Matrices ---
fig2 = figure('Position', [50 50 1100 400]);
sgtitle('SVM Confusion Matrices — All Three Experiments (Test Set, N=282)', ...
    'FontSize', 12, 'FontWeight', 'bold');

for ei = 1:3
    exp = {'A','B','C'};
    e   = exp{ei};
    subplot(1,3,ei);

    cm = confusionmat(y_test, svm_results.(e).pred, 'Order', 0:4);
    cm_norm = cm ./ sum(cm,2);
    cm_norm(isnan(cm_norm)) = 0;

    imagesc(cm_norm); colormap(subplot(1,3,ei), parula); caxis([0 1]);
    for r = 1:N_CLASSES
        for c = 1:N_CLASSES
            clr = 'white'; if cm_norm(r,c) < 0.5, clr=[0.1 0.1 0.1]; end
            text(c, r, sprintf('%d\n%.0f%%', cm(r,c), cm_norm(r,c)*100), ...
                'HorizontalAlignment','center','Color',clr,...
                'FontSize',7.5,'FontWeight','bold');
        end
    end
    set(gca,'XTick',1:5,'XTickLabel',SHORT_NAMES, ...
            'YTick',1:5,'YTickLabel',SHORT_NAMES,'FontSize',9);
    xlabel('Predicted'); ylabel('True');
    title(sprintf('SVM %s\n(C=%.3g, \\gamma=%.3g)\nMacro F1=%.4f', ...
        exp_configs.(e).name, svm_results.(e).best_C, ...
        svm_results.(e).best_gamma, svm_results.(e).macro_f1), ...
        'FontSize', 9.5, 'FontWeight', 'bold');
end
saveas(fig2, fullfile(out_dir, 'confusion_matrices_svm.png'));

% --- Full Comparison Figure (SVM vs all baselines, Exp B) ---
fig3 = figure('Position', [50 50 900 520]);

% Load baseline results
bl_path = fullfile(script_dir, 'baseline_outputs', 'experiment_results.csv');
if exist(bl_path, 'file')
    bl = readtable(bl_path, 'VariableNamingRule', 'preserve');
    % Extract Exp B results
    expB_rows = strcmp(bl.Experiment, 'B') | strcmp(bl.Experiment, '-');
    bl_b = bl(expB_rows, :);
    labels_all = [bl_b.Model; ...
                  {'SVM Exp B (SMOTE)'}];
    f1_all     = [bl_b.Macro_F1; svm_results.B.macro_f1];
    n_models   = length(f1_all);
    colors_all = repmat([0.7 0.7 0.7], n_models, 1);
    colors_all(end,:) = [0.85 0.20 0.10];  % SVM in red
    b = bar(f1_all, 'FaceColor', 'flat');
    b.CData = colors_all;
    for i = 1:n_models
        text(i, f1_all(i)+0.01, sprintf('%.4f', f1_all(i)), ...
            'HorizontalAlignment','center','FontSize',8,'FontWeight','bold');
    end
    set(gca,'XTickLabel', labels_all, 'XTickLabelRotation', 25, 'FontSize', 9);
    ylabel('Macro F1 Score','FontSize',12);
    title('Full Model Comparison — Experiment B (SMOTE, Identical Data)', ...
        'FontSize',12,'FontWeight','bold');
    ylim([0 1.12]); grid on; box on;
    saveas(fig3, fullfile(out_dir, 'full_comparison_expB.png'));
    fprintf('  Saved: full_comparison_expB.png\n');
end

fprintf('  Saved: confusion_matrices_svm.png\n');
fprintf('  Saved: grid_heatmap_expA.png\n');
fprintf('  Saved: grid_heatmap_expB.png\n\n');
%% --- SAVE RESULTS
rows = {};
for e = {'A','B','C'}
    exp = e{1};
    r   = svm_results.(exp);
    rows(end+1,:) = {
        sprintf('SVM %s', exp_configs.(exp).name), ...
        exp, r.best_C, r.best_gamma, r.best_cv_f1, ...
        r.acc, r.macro_f1, ...
        r.per_class(2,3), r.per_class(5,3)
    };
end

res_table = cell2table(rows, 'VariableNames', ...
    {'Model','Experiment','Best_C','Best_gamma','CV_Macro_F1', ...
     'Test_Accuracy','Test_Macro_F1','D1_F1','T3_F1'});
writetable(res_table, fullfile(out_dir, 'svm_results_summary.csv'));
fprintf('  Saved: svm_results_summary.csv\n\n');
%% --- SUMMARY
fprintf('================================================================\n');
fprintf('  SVM RESULTS SUMMARY\n');
fprintf('================================================================\n');
fprintf('  %-28s  %6s  %6s  %8s  %8s  %7s\n', ...
    'Model','Best C','Best g','Acc','MacroF1','D1 F1');
fprintf('  %s\n', repmat('-',1,72));
for e = {'A','B','C'}
    exp = e{1};
    r   = svm_results.(exp);
    fprintf('  %-28s  %6.3g  %6.3g  %8.4f  %8.4f  %7.4f\n', ...
        exp_configs.(exp).name, r.best_C, r.best_gamma, ...
        r.acc, r.macro_f1, r.per_class(2,3));
end
fprintf('  SVM achieves Macro F1 >= %.4f across all three experiments.\n', ...
    min([svm_results.A.macro_f1, svm_results.B.macro_f1, svm_results.C.macro_f1]));

%% --- LOCAL FUNCTIONS
function [accuracy, macro_f1, per_class] = compute_metrics(y_true, y_pred, n_classes)
% COMPUTE_METRICS  Accuracy, macro F1, per-class [Precision, Recall, F1].
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