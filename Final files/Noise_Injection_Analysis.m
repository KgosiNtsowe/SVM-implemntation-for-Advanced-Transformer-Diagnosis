%  Project: Advanced Diagnostics for Power Transformers
%  Author:  Kgosietsile Ntsowe | ID: 11409134
%
%  PURPOSE:
%    Simulate real-world DGA sensor calibration drift and measurement
%    uncertainty by injecting Gaussian noise into test set gas
%    readings. Evaluates how each model's performance degrades 
%    testing deployment robustness rather than clean-data accuracy.
%
%   MOTIVATION:
%    DGA sensors in the field are subject to:
%      2%  noise — well-calibrated laboratory GC analyser
%      5%  noise — typical online DGA monitor (IEC 60567)
%     10%  noise — online monitor with calibration drift
%     20%  noise — degraded/fouled sensor, long recalibration interval
%     30%  noise — severe drift, approaching end-of-life sensor
%    Source: IEEE C57.104-2019 and IEC 60567:2011 accuracy tolerances
%
%    Gaussian noise N(0, sigma * std_gas) added to raw ppm values
%    of the 7 gas columns in the test set ONLY .
%    Values clipped to >= 0 .
%    Duval triangle percentages held fixed — noise is applied at the
%    sensor level before the Duval computation, simulating the case
%    where the DGA system recomputes Duval from the noisy readings.
%    This is conservative in reality Duval %
%    would also shift, producing additional degradation.
%    MODELS ARE NEVER RETRAINED — only the test input is perturbed.
%    20 repetitions per noise level to average over random seeds.
%
%  INPUTS:
%    pipeline_outputs/X_train_scaled.csv
%    pipeline_outputs/X_test_scaled.csv
%    Final_Transformer_Dataset_with_Duval.csv  (for raw ppm values)
%
%  OUTPUTS → noise_outputs/:
%    degradation_curves.png
%    d1_recall_under_noise.png
%    noise_results_table.csv
%    d1_recall_table.csv

clc; clear; close all;
rng(42);

%% ---SETUP 
script_dir = fileparts(mfilename('fullpath'));
out_dir = fullfile(script_dir, 'noise_outputs');
if ~exist(out_dir, 'dir'), mkdir(out_dir); end

N_CLASSES    = 5;
N_REPS       = 20;    % Repetitions per noise level
NOISE_LEVELS = [0.0, 0.02, 0.05, 0.10, 0.20, 0.30];
NOISE_LABELS = {'Clean','2%','5%','10%','20%','30%'};
N_LEVELS     = length(NOISE_LEVELS);

CLASS_NAMES  = {'Healthy','D1 Discharge','T1 <300C','T2 300-700C','T3 >700C'};
SHORT_NAMES  = {'H','D1','T1','T2','T3'};
fprintf('  Noise levels: 0%%, 2%%, 5%%, 10%%, 20%%, 30%% of gas column std\n');
fprintf('  Repetitions per level: %d\n', N_REPS);
%% ---LOAD DATA
% Scaled pipeline outputs (for extracting scaler params)
train_data = readmatrix(fullfile(script_dir, 'pipeline_outputs','X_train_smote.csv'));
test_data  = readmatrix(fullfile(script_dir, 'pipeline_outputs','X_test_scaled.csv'));


X_train_sc = train_data(:, 1:10);  y_train = train_data(:, 11);
X_test_sc  = test_data(:,  1:10);  y_test  = test_data(:,  11);

% Raw dataset — needed for unscaled ppm values and gas stds
T_raw     = readtable(fullfile(script_dir, 'Final_Transformer_Dataset_with_Duval.csv'), ...
    'VariableNamingRule', 'preserve');
gas_names = {'Hydrogen','Methane','Ethane','Ethylene','Acetylene', ...
             'Carbon Monoxide','Carbon Dioxide'};
% Reconstruct same split indices as initial 
cv_split  = cvpartition(T_raw.Fault_Type, 'HoldOut', 0.20, 'Stratify', true);
train_idx = find(training(cv_split));
test_idx  = find(test(cv_split));

X_tr_raw    = T_raw{train_idx, gas_names};   % Raw ppm — training
X_te_raw    = T_raw{test_idx,  gas_names};   % Raw ppm — test
X_te_duval  = T_raw{test_idx,  {'%CH4','%C2H4','%C2H2'}};
y_test_raw  = T_raw.Fault_Type(test_idx);


% Gas column standard deviations
gas_stds = std(double(X_tr_raw));

% Scaler parameters — median and IQR from training
scaler_params = readtable(fullfile(script_dir, 'pipeline_outputs','scaler_params.csv'));
scaler_center = scaler_params.Center_Median';
scaler_scale  = scaler_params.Scale_IQR';

fprintf('  Train: %d  Test: %d\n', length(train_idx), length(test_idx));
fprintf('  Gas column stds (training raw ppm):\n');
for i = 1:7
    fprintf('    %-22s: %.2f ppm\n', gas_names{i}, gas_stds(i));
end
fprintf('\n');
%% ---TRAIN FINAL MODELS
% SVM — best config
t_svm = templateSVM('KernelFunction', 'rbf', ...
                    'BoxConstraint',  100, ...
                    'KernelScale',    1/sqrt(0.1), ...
                    'Standardize',    false);
svm_mdl = fitcecoc(X_train_sc, y_train, 'Learners', t_svm, ...
                   'ClassNames', 0:4, 'Coding', 'onevsone');

% kNN — Load optimal k from results
bl_path = fullfile(script_dir, 'baseline_outputs', 'experiment_results.csv');
best_k = 5; % Safe default fallback

if exist(bl_path, 'file')
    bl = readtable(bl_path, 'VariableNamingRule', 'preserve');
    knn_row = bl(strcmp(bl.Experiment, 'B') & contains(bl.Model, 'kNN'), :);
    
    if ~isempty(knn_row)
        try
            % Safely cast any format (categorical, cell, char) to string, then double
            parsed_k = str2double(string(knn_row.HyperParam(1)));
            
            % Verify it is a valid, positive number
            if ~isnan(parsed_k) && parsed_k > 0
                best_k = round(parsed_k); % Force it to be a clean integer
                fprintf('  >> Loaded optimal kNN parameter: k = %d\n', best_k);
            else
                fprintf('  >> WARNING: Extracted k was NaN. Defaulting to k = 5\n');
            end
        catch
            fprintf('  >> WARNING: Could not parse k. Defaulting to k = 5\n');
        end
    end
end
knn_mdl = fitcknn(X_train_sc, y_train, 'NumNeighbors', best_k, ...
                  'Distance', 'euclidean', 'Standardize', false);


% Decision Tree — depth 10
dt_mdl  = fitctree(X_train_sc, y_train, 'MaxNumSplits', 2^10-1);

% Verify clean baseline (0% noise)
pred_clean = predict(svm_mdl, X_test_sc);
[~, baseline_f1, ~] = compute_metrics(y_test, pred_clean, N_CLASSES);
fprintf('  SVM clean baseline: Macro F1 = %.4f\n\n', baseline_f1);
%% --- NOISE INJECTION LOOP
%
%  For each noise level sigma_pct:
%    For each repetition rep = 1..N_REPS:
%      1. Generate Gaussian noise: N ~ N(0, sigma_pct * gas_std)
%      2. Add to raw test gas readings, clip to >= 0
%      3. Apply log1p transform to noisy gas values
%      4. Apply robust scaling using TRAINING parameters
%      5. Combine with (fixed) Duval % columns
%      6. Predict with each model
%      7. Compute Macro F1 and D1 recall
%    Average across N_REPS repetitions
% Storage: [3 models x N_LEVELS] means and stds
models_list  = {svm_mdl, knn_mdl, dt_mdl};
model_names  = {'SVM', 'kNN', 'DT'};
n_models     = 3;

mean_f1   = zeros(n_models, N_LEVELS);
std_f1    = zeros(n_models, N_LEVELS);
mean_d1   = zeros(n_models, N_LEVELS);   % D1 recall
std_d1    = zeros(n_models, N_LEVELS);

for li = 1:N_LEVELS
    sigma_pct = NOISE_LEVELS(li);
    fprintf('  Noise level %3.0f%%: ', sigma_pct*100);

    for mi = 1:n_models
        model = models_list{mi};
        rep_f1 = zeros(N_REPS, 1);
        rep_d1 = zeros(N_REPS, 1);

        for rep = 1:N_REPS
            % Set per-rep seed for reproducibility
            rng(rep * 13 + 42);

            % Apply noise to raw gas columns
            if sigma_pct > 0
                noise      = randn(size(X_te_raw)) .* (sigma_pct .* gas_stds);
                X_noisy    = max(double(X_te_raw) + noise, 0);  % clip >= 0
            else
                X_noisy    = double(X_te_raw);
            end

            % Apply log1p transform
            X_noisy_log = log1p(X_noisy);

            % Apply robust scaling using training parameters
            % Gas cols (1:7): (log1p_val - center) / scale
            % Duval cols (8:10): (raw_pct - center) / scale
            X_test_noisy = zeros(size(X_te_raw, 1), 10);
            for col = 1:7
                X_test_noisy(:, col) = (X_noisy_log(:, col) - ...
                    scaler_center(col)) / scaler_scale(col);
            end
            for col = 8:10
                X_test_noisy(:, col) = (double(X_te_duval(:, col-7)) - ...
                    scaler_center(col)) / scaler_scale(col);
            end

            % Predict
            y_pred = predict(model, X_test_noisy);

            % Macro F1
            [~, f1, ~] = compute_metrics(y_test_raw, y_pred, N_CLASSES);
            rep_f1(rep) = f1;

            % D1 Recall (Class 1)
            d1_mask = (y_test_raw == 1);
            if sum(d1_mask) > 0
                rep_d1(rep) = sum(y_pred(d1_mask) == 1) / sum(d1_mask);
            end
        end

        mean_f1(mi, li) = mean(rep_f1);
        std_f1(mi, li)  = std(rep_f1);
        mean_d1(mi, li) = mean(rep_d1);
        std_d1(mi, li)  = std(rep_d1);
    end

    fprintf('SVM=%.4f  kNN=%.4f  DT=%.4f\n', ...
        mean_f1(1,li), mean_f1(2,li), mean_f1(3,li));
end
%% ---RESULTS TABLES
fprintf('\n================================================================\n');
fprintf('  NOISE INJECTION RESULTS — MACRO F1\n');
fprintf('================================================================\n');
fprintf('  %-6s  %7s  %7s  %7s  %7s  %7s  %7s\n', ...
    'Model','Clean','2%','5%','10%','20%','30%');
fprintf('  %s\n', repmat('-', 1, 58));
for mi = 1:n_models
    fprintf('  %-6s', model_names{mi});
    for li = 1:N_LEVELS
        fprintf('  %7.4f', mean_f1(mi,li));
    end
    fprintf('\n');
end

fprintf('\n  ABSOLUTE F1 DROP FROM CLEAN BASELINE:\n');
fprintf('  %-6s  %7s  %7s  %7s  %7s  %7s\n', 'Model','2%','5%','10%','20%','30%');
fprintf('  %s\n', repmat('-', 1, 48));
for mi = 1:n_models
    fprintf('  %-6s', model_names{mi});
    for li = 2:N_LEVELS
        drop = mean_f1(mi,1) - mean_f1(mi,li);
        fprintf('  %+7.4f', -drop);   % negative = drop
    end
    fprintf('\n');
end

fprintf('\n================================================================\n');
fprintf('  D1 RECALL UNDER NOISE (Safety-Critical Class)\n');
fprintf('================================================================\n');
fprintf('  %-6s  %7s  %7s  %7s  %7s  %7s\n', ...
    'Model','Clean','2%','5%','10%','20%','30%');
fprintf('  %s\n', repmat('-', 1, 52));
for mi = 1:n_models
    fprintf('  %-6s', model_names{mi});
    for li = 1:N_LEVELS
        fprintf('  %7.4f', mean_d1(mi,li));
    end
    fprintf('\n');
end
fprintf('  At 10%% noise (typical online DGA monitor), SVM detects %.1f%%\n', ...
    mean_d1(1,4)*100);
fprintf('  of arcing faults vs DT''s %.1f%% — a %.1f%% safety margin.\n\n', ...
    mean_d1(3,4)*100, (mean_d1(1,4)-mean_d1(3,4))*100);
%% ---FIGURES
model_colors = [0.85 0.33 0.10;   % SVM — red
                0.00 0.45 0.74;   % kNN — blue
                0.49 0.18 0.56];  % DT  — purple
noise_pct    = NOISE_LEVELS * 100;

% --- Figure 1: Macro F1 Degradation Curves ---
fig1 = figure('Position', [50 50 900 560]);
hold on;

for mi = 1:n_models
    % Shaded error band
    upper = mean_f1(mi,:) + std_f1(mi,:);
    lower = mean_f1(mi,:) - std_f1(mi,:);


    fill([noise_pct, fliplr(noise_pct)], ...
         [upper, fliplr(lower)], ...
    model_colors(mi,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');

    % Mean curve
    plot(noise_pct, mean_f1(mi,:), '-o', ...
        'Color', model_colors(mi,:), 'LineWidth', 2.5, ...
        'MarkerSize', 7, 'MarkerFaceColor', model_colors(mi,:), ...
        'DisplayName', model_names{mi});
end

% Reference lines for DGA sensor tolerance zones
xregion_fill = fill([0 5 5 0], [0 0 1.05 1.05], [0.9 1.0 0.9], ...
    'FaceAlpha', 0.08, 'EdgeColor', 'none');
xregion_fill2 = fill([5 15 15 5], [0 0 1.05 1.05], [1.0 0.95 0.8], ...
    'FaceAlpha', 0.08, 'EdgeColor', 'none');
xregion_fill3 = fill([15 35 35 15], [0 0 1.05 1.05], [1.0 0.8 0.8], ...
    'FaceAlpha', 0.08, 'EdgeColor', 'none');

text(2.5,  0.28, {'Lab GC'; 'accuracy'}, 'HorizontalAlignment','center', ...
    'FontSize', 8, 'Color', [0.3 0.5 0.3]);
text(10,   0.28, {'Online DGA'; 'monitor'}, 'HorizontalAlignment','center', ...
    'FontSize', 8, 'Color', [0.6 0.5 0.2]);
text(25,   0.28, {'Degraded'; 'sensor'}, 'HorizontalAlignment','center', ...
    'FontSize', 8, 'Color', [0.7 0.3 0.3]);

legend(model_names, 'Location', 'northeast', 'FontSize', 11);
xlabel('Noise Level (% of gas column standard deviation)', 'FontSize', 12);
ylabel('Macro F1 Score', 'FontSize', 12);
title({'Robustness to DGA Sensor Noise — Macro F1 Degradation'; ...
       'Shaded = ±1 std across 20 repetitions'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
xlim([-0.5 31]); ylim([0.25 1.05]);
set(gca, 'XTick', noise_pct, 'FontSize', 11);
grid on; box on;
saveas(fig1, fullfile(out_dir, 'degradation_curves.png'));

% --- Figure 2: D1 Recall Under Noise ---
fig2 = figure('Position', [100 100 900 520]);
hold on;

for mi = 1:n_models
    upper = min(mean_d1(mi,:) + std_d1(mi,:), 1);
    lower = max(mean_d1(mi,:) - std_d1(mi,:), 0);
fill([noise_pct, fliplr(noise_pct)], [upper, fliplr(lower)], ...
         model_colors(mi,:), 'FaceAlpha', 0.15, 'EdgeColor', 'none', ...
         'HandleVisibility', 'off');

    plot(noise_pct, mean_d1(mi,:), '-o', ...
        'Color', model_colors(mi,:), 'LineWidth', 2.5, ...
        'MarkerSize', 7, 'MarkerFaceColor', model_colors(mi,:), ...
        'DisplayName', model_names{mi});
end

% Minimum acceptable D1 recall reference line
yline(0.80, '--k', 'LineWidth', 1.5);
text(30, 0.815, 'Minimum acceptable recall = 0.80', ...
    'HorizontalAlignment', 'right', ...
    'FontSize', 9, 'Color', [0.3 0.3 0.3]);

legend(model_names, 'Location', 'southwest', 'FontSize', 11);
xlabel('Noise Level (% of gas column standard deviation)', 'FontSize', 12);
ylabel('D1 (Arcing Fault) Recall', 'FontSize', 12);
title({'D1 Arcing Fault Recall Under Sensor Noise'; ...
       'Most safety-critical class — false negative = undetected fault'}, ...
    'FontSize', 12, 'FontWeight', 'bold');
xlim([-0.5 31]); ylim([0 1.05]);
set(gca, 'XTick', noise_pct, 'FontSize', 11);
grid on; box on;
saveas(fig2, fullfile(out_dir, 'd1_recall_under_noise.png'));

% --- Figure 3: Heatmap of relative degradation ---
fig3 = figure('Position', [100 100 700 380]);
degradation_pct = zeros(n_models, N_LEVELS-1);
for mi = 1:n_models
    for li = 2:N_LEVELS
        degradation_pct(mi, li-1) = ...
            (mean_f1(mi,1) - mean_f1(mi,li)) / mean_f1(mi,1) * 100;
    end
end
imagesc(degradation_pct);
colormap(flipud(parula)); cb = colorbar;
cb.Label.String = '% F1 drop from clean baseline';
caxis([0 65]);
for mi = 1:n_models
    for li = 1:N_LEVELS-1
        clr = 'white'; if degradation_pct(mi,li) < 30, clr=[0.1 0.1 0.1]; end
        text(li, mi, sprintf('%.1f%%', degradation_pct(mi,li)), ...
            'HorizontalAlignment','center', 'Color', clr, ...
            'FontSize', 10, 'FontWeight', 'bold');
    end
end
set(gca, 'YTick', 1:3, 'YTickLabel', model_names, ...
         'XTick', 1:5, 'XTickLabel', {'2%','5%','10%','20%','30%'}, ...
         'FontSize', 11);
xlabel('Noise Level', 'FontSize', 12);
title('Relative F1 Degradation (% drop from clean baseline)', ...
    'FontSize', 12, 'FontWeight', 'bold');
saveas(fig3, fullfile(out_dir, 'degradation_heatmap.png'));

%% ---SAVE TABLES
noise_pct_labels = {'Clean','pct2','pct5','pct10','pct20','pct30'};
f1_rows = [model_names', num2cell(mean_f1)];
f1_table = cell2table(f1_rows, 'VariableNames', ...
    ['Model', noise_pct_labels]);
writetable(f1_table, fullfile(out_dir, 'noise_results_table.csv'));

d1_rows = [model_names', num2cell(mean_d1)];
d1_table = cell2table(d1_rows, 'VariableNames', ...
    ['Model', noise_pct_labels]);
writetable(d1_table, fullfile(out_dir, 'd1_recall_table.csv'));
%% ---SUMMARY
fprintf('================================================================\n');
fprintf('  SUMMARY\n');
fprintf('================================================================\n');
fprintf('  At 5%% noise (good lab conditions):\n');
fprintf('    SVM  F1=%.4f  D1 recall=%.4f\n', mean_f1(1,3), mean_d1(1,3));
fprintf('    kNN  F1=%.4f  D1 recall=%.4f\n', mean_f1(2,3), mean_d1(2,3));
fprintf('    DT   F1=%.4f  D1 recall=%.4f\n\n', mean_f1(3,3), mean_d1(3,3));
fprintf('  At 10%% noise (typical online DGA monitor):\n');
fprintf('    SVM  F1=%.4f  D1 recall=%.4f\n', mean_f1(1,4), mean_d1(1,4));
fprintf('    kNN  F1=%.4f  D1 recall=%.4f\n', mean_f1(2,4), mean_d1(2,4));
fprintf('    DT   F1=%.4f  D1 recall=%.4f\n\n', mean_f1(3,4), mean_d1(3,4));

%% ---LOCAL FUNCTIONS
function [accuracy, macro_f1, per_class] = compute_metrics(y_true, y_pred, n_classes)
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