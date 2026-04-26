%  Project: Advanced Diagnostics for Power Transformers
%  Author:  Kgosietsile Ntsowe | ID: 11409134
%  PURPOSE: Transform the raw master dataset into a clean, scaled,
%    class-balanced training set and test set ready for model training

%  ORDER:
%    Step 1 → Load & separate features from labels
%    Step 2 → Stratified 80/20 train/test split  [BEFORE any fitting]
%    Step 3 → Log1p transform   (gas columns only, fit-free)
%    Step 4 → Robust scaling    (fit on train, apply to test)
%    Step 5 → SMOTE             (applied to training set only)
%    Step 6 → Integrity checks  (verify no data leakage)
%    Step 7 → Save outputs      (for use in all subsequent scripts)

%    The test set is created FIRST and then set aside completely.
%    The robust scaler is fit ONLY on training data.
%    SMOTE is applied ONLY to training data.
%    The test set is NEVER touched until final model evaluation.

%  OUTPUTS (saved to ./pipeline_outputs/):
%    - X_train_scaled.csv    (scaled, pre-SMOTE — used inside CV)
%    - y_train.csv
%    - X_test_scaled.csv     (scaled with TRAIN parameters — quarantined)
%    - y_test.csv
%    - X_train_smote.csv     (SMOTE-balanced — used for final model)
%    - y_train_smote.csv
%    - scaler_params.csv     (medians + IQRs — needed for deployment)
%    -fig1_class_balance.png
%    -fig2_hydrogen_transform.png
%    -fig3_scaled_features.png

clc; clear; close all;
rng(42);

%% --- 0. OUTPUT DIRECTORY -------------------------------------------
% Find the exact folder where this .m script is currently saved
script_dir = fileparts(mfilename('fullpath'));
% Create a path for the 'audit_outputs' folder perfectly next to the script
out_dir = fullfile(script_dir, 'pipeline_outputs');
% Create the folder if it does not already exist
if ~exist(out_dir, 'dir')
    mkdir(out_dir); 
end

%% --- 1.LOAD DATA AND SEPARATE FEATURES FROM LABELS-------------------------------------------
T = readtable('C:\Users\Kgosi\OneDrive\Desktop\Individual Project\Power transformer data for fault diagnosis\Power transformer data for fault diagnosis\Final_Transformer_Dataset_with_Duval.csv','VariableNamingRule', 'preserve');
gas_idx   = [1 2 3 4 5 6 7];    % Hydrogen through Carbon Dioxide
duval_idx = [9 10 11];           % %CH4, %C2H4, %C2H2
gas_names   = {'Hydrogen','Methane','Ethane','Ethylene','Acetylene',...
               'Carbon Monoxide','Carbon Dioxide'};
duval_names = {'pct_CH4','pct_C2H4','pct_C2H2'};
all_feat_names = [gas_names, duval_names];
% Extract as numeric arrays
X_gas   = T{:, gas_names};         % [1408 x 7]
X_duval = T{:, {'%CH4','%C2H4','%C2H2'}}; % [1408 x 3]

y       = T.Fault_Type;            % [1408 x 1]  integer labels 0–4

X_raw   = [X_gas, X_duval];        % [1408 x 10] full feature matrix
X_raw   = double(X_raw);           % ensure double precision throughout

[n_total, n_feat] = size(X_raw);
n_classes = numel(unique(y));

fprintf('  Loaded: %d samples, %d features, %d classes\n\n', ...
        n_total, n_feat, n_classes);

%% -----------2 — STRATIFIED 80/20 TRAIN/TEST SPLIT----------------------
cv_split = cvpartition(y, 'HoldOut', 0.20, 'Stratify', true);

train_idx = training(cv_split);   % logical index
test_idx  = test(cv_split);

X_train_raw = X_raw(train_idx, :);
y_train     = y(train_idx);
X_test_raw  = X_raw(test_idx,  :);
y_test      = y(test_idx);

fprintf('  Train: %d samples  |  Test: %d samples\n', ...
        sum(train_idx), sum(test_idx));
fprintf('  Stratification check:\n');
fprintf('    %-12s  %6s  %6s\n', 'Class', 'Train', 'Test');
for c = 0:4
    n_tr = sum(y_train == c);
    n_te = sum(y_test  == c);
    fprintf('    Class %-6d  %6d  %6d  (%.1f%% / %.1f%%)\n', c, ...
            n_tr, n_te, ...
            n_tr/sum(train_idx)*100, n_te/sum(test_idx)*100);
end
%% --------3 — LOG1p TRANSFORM------------------------------
X_train_log = X_train_raw;
X_test_log  = X_test_raw;

% Apply log1p to gas columns only (indices 1:7 in 1-indexed MATLAB)
X_train_log(:, 1:7) = log1p(X_train_raw(:, 1:7));
X_test_log(:,  1:7) = log1p(X_test_raw(:,  1:7));

% Verify transform worked correctly
fprintf('    Raw  median: %.4f  |  Log1p median: %.4f\n', ...
        median(X_train_raw(:,5)), median(X_train_log(:,5)));
fprintf('    Raw  max:    %.1f ppm  |  Log1p max:    %.4f\n', ...
        max(X_train_raw(:,5)), max(X_train_log(:,5)));
fprintf('  >> Zero values preserved: log1p(0) = %.4f\n\n', log1p(0));


%% --------------4 — ROBUST SCALING---------------------
[X_train_scaled, scaler_center, scaler_scale] = robust_scale_fit(X_train_log);
X_test_scaled = robust_scale_apply(X_test_log, scaler_center, scaler_scale);

% Save scaler parameters — required for deployment and for test application
scaler_table = table(all_feat_names', scaler_center', scaler_scale', ...
    'VariableNames', {'Feature','Center_Median','Scale_IQR'});
writetable(scaler_table, fullfile(out_dir, 'scaler_params.csv'));

fprintf('  Scaler parameters (fit on training data):\n');
fprintf('    %-22s  %9s  %9s\n', 'Feature', 'Median', 'IQR');
fprintf('    %s\n', repmat('-', 1, 44));
for i = 1:n_feat
    flag = '';
    if scaler_scale(i) == 1.0 && median(X_train_log(:,i)) == 0
        flag = '  [IQR=0 fallback: scale=1]';
    end
    fprintf('    %-22s  %9.4f  %9.4f%s\n', ...
        all_feat_names{i}, scaler_center(i), scaler_scale(i), flag);
end

% Verify: training set should now have median ≈ 0, IQR ≈ 1
fprintf('\n  Post-scaling verification (train set):\n');
fprintf('    %-22s  %9s  %9s\n', 'Feature', 'Median', 'IQR');
for i = 1:n_feat
    med = median(X_train_scaled(:,i));
    q1  = quantile(X_train_scaled(:,i), 0.25);
    q3  = quantile(X_train_scaled(:,i), 0.75);
    fprintf('    %-22s  %9.4f  %9.4f\n', all_feat_names{i}, med, q3-q1);
end
fprintf('\n  >> Test set scaled with TRAIN parameters (not refit).\n\n');



%%  ------5-SMOTE (Synthetic Minority Over-sampling Technique) Chawla et al. (2002)------------
%  LIMITATION & RISK AWARENESS (Extreme Oversampling):
%    Class 1 (D1) only has ~33 training samples, meaning SMOTE will 
%    generate over 700 synthetic points (a >2000% increase). This risks 
%    creating a dense, overlapping feature cluster leading to SVM overfitting.
majority_count = sum(y_train == 0);   % 750
fprintf('  Target count per class: %d (matching majority Class 0)\n', majority_count);
fprintf('  Pre-SMOTE class distribution:\n');
for c = 0:4
    n = sum(y_train == c);
    fprintf('    Class %d: %4d samples\n', c, n);
end

[X_train_smote, y_train_smote] = smote_balance(X_train_scaled, y_train, ...
                                               majority_count, 5, 42);

fprintf('\n  Post-SMOTE class distribution:\n');
for c = 0:4
    n = sum(y_train_smote == c);
    fprintf('    Class %d: %4d samples\n', c, n);
end
fprintf('\n  Total training samples after SMOTE: %d\n', length(y_train_smote));
fprintf('  (was %d before SMOTE — added %d synthetic samples)\n\n', ...
        length(y_train), length(y_train_smote) - length(y_train));


%% -------6—INTEGRITY CHECKS-----------------------------
% Check 1: Test set size unchanged
assert(size(X_test_scaled, 1) == sum(test_idx), ...
    'FAIL: Test set size changed — data leakage risk.');
fprintf('  [PASS] Test set size unchanged: %d samples\n', size(X_test_scaled,1));

% Check 2: SMOTE classes balanced
for c = 0:4
    assert(sum(y_train_smote == c) == majority_count, ...
        'FAIL: SMOTE did not balance class %d correctly.', c);
end
fprintf('  [PASS] All 5 classes balanced to %d samples after SMOTE\n', majority_count);

% Check 3: No NaN or Inf in any output array
assert(~any(isnan(X_train_scaled(:))), 'FAIL: NaN in scaled training data.');
assert(~any(isnan(X_test_scaled(:))),  'FAIL: NaN in scaled test data.');
assert(~any(isnan(X_train_smote(:))), 'FAIL: NaN in SMOTE training data.');
assert(~any(isinf(X_train_scaled(:))), 'FAIL: Inf in scaled training data.');
fprintf('  [PASS] No NaN or Inf values in any output array\n');

% Check 4: Test set values are NOT centred at 0 (confirms not refit)
test_medians = median(X_test_scaled);
n_nonzero = sum(abs(test_medians) > 0.01);
assert(n_nonzero > 0, ...
    'WARN: Test set medians all near 0 — possible accidental refit on test.');
fprintf('  [PASS] Test set medians not forced to 0 (confirms train-only scaling)\n');

% Check 5: SMOTE samples are bounded within training range
train_min = min(X_train_scaled);
train_max = max(X_train_scaled);
smote_only = X_train_smote(length(y_train)+1:end, :);
in_range = all(smote_only >= train_min - 1e-9 & smote_only <= train_max + 1e-9, 2);
assert(all(in_range), 'FAIL: SMOTE generated samples outside training range.');
fprintf('  [PASS] All synthetic SMOTE samples lie within training data range\n');

fprintf('\n  >> ALL INTEGRITY CHECKS PASSED\n\n');

%% -------7—SAVE OUTPUTS AND GENERATE FIGURES
writematrix([X_train_scaled, y_train], ...
    fullfile(out_dir, 'X_train_scaled.csv'));
writematrix([X_test_scaled, y_test], ...
    fullfile(out_dir, 'X_test_scaled.csv'));
writematrix([X_train_smote, y_train_smote], ...
    fullfile(out_dir, 'X_train_smote.csv'));

fprintf('  Saved: X_train_scaled.csv  (%d x %d)\n', size(X_train_scaled));
fprintf('  Saved: X_test_scaled.csv   (%d x %d)\n', size(X_test_scaled));
fprintf('  Saved: X_train_smote.csv   (%d x %d)\n', size(X_train_smote));
fprintf('  Saved: scaler_params.csv\n\n');

%%  FIGURE 1: Class Balance (Before vs After SMOTE)
fig1 = figure('Position', [100 100 700 500]);
class_names = {'Healthy','D1 Disch.','T1 <300C','T2 300-700C','T3 >700C'};

pre_counts  = arrayfun(@(c) sum(y_train==c), 0:4);
post_counts = arrayfun(@(c) sum(y_train_smote==c), 0:4);
b = bar([pre_counts; post_counts]', 'grouped');

% Apply colors: Blue for Pre-SMOTE, Orange for Post-SMOTE
b(1).FaceColor = [0.00 0.45 0.74]; 
b(2).FaceColor = [0.85 0.33 0.10]; 

legend({'Pre-SMOTE','Post-SMOTE'}, 'Location','northwest', 'FontSize',10);
set(gca, 'XTickLabel', class_names, 'FontSize', 10);
title('Class Balance: Before vs After SMOTE', 'FontWeight','bold', 'FontSize', 13);
ylabel('Sample Count', 'FontSize', 11); grid on;

saveas(fig1, fullfile(out_dir, 'fig1_class_balance.png'));

%%  FIGURE 2: Hydrogen Transformation (Raw vs Log1p)
fig2 = figure('Position', [150 150 1000 450]);

% Left Subplot: Raw Hydrogen
subplot(1,2,1);
histogram(X_train_raw(:,1), 30, 'FaceColor', [0.6 0.6 0.6], 'EdgeColor', 'w', 'Normalization', 'probability');
title('Hydrogen: Raw Data', 'FontWeight','bold', 'FontSize', 12);
xlabel('Raw Concentration (ppm)'); ylabel('Probability');
set(gca, 'FontSize', 10); grid on;

% Add Max Value text box
raw_max = max(X_train_raw(:,1));
txt_raw = sprintf('Max Value: %.0f ppm', raw_max);
text(0.50, 0.85, txt_raw, 'Units', 'normalized', 'FontSize', 10, ...
    'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7]);

% Right Subplot: Transformed Hydrogen
subplot(1,2,2);
histogram(X_train_log(:,1), 30, 'FaceColor', [0.00 0.45 0.74], 'EdgeColor', 'w', 'Normalization', 'probability');
title('Hydrogen: ln(1+x) Transformed', 'FontWeight','bold', 'FontSize', 12);
xlabel('ln(1 + ppm)'); ylabel('Probability');
set(gca, 'FontSize', 10); grid on;

% Add Max Value text box
log_max = max(X_train_log(:,1));
txt_log = sprintf('Max Value: %.1f', log_max);
text(0.60, 0.85, txt_log, 'Units', 'normalized', 'FontSize', 10, ...
    'BackgroundColor', 'w', 'EdgeColor', [0.7 0.7 0.7]);

sgtitle('Effect of Log Transform on Hydrogen Distribution', 'FontSize', 14, 'FontWeight', 'bold');
saveas(fig2, fullfile(out_dir, 'fig2_hydrogen_transform.png'));


%%  FIGURE 3: Scaled Features Boxplot
fig3 = figure('Position', [200 200 800 500]);

feat_labels_short = {'H2','CH4','C2H6','C2H4','C2H2','CO','CO2','%CH4','%C2H4','%C2H2'};
boxplot(X_train_scaled, 'Labels', feat_labels_short);
yline(0, '--k', 'LineWidth', 1, 'DisplayName', 'Median = 0');

% Zoom in the Y-axis to show the boxes, letting the extreme outliers run off screen
ylim([-4, 8]); 

title('Scaled Training Features (Robust Scaled)', 'FontWeight','bold', 'FontSize', 13);
ylabel('Scaled Value', 'FontSize', 11); 
xlabel('Feature', 'FontSize', 11);
set(gca, 'FontSize', 10); grid on;

saveas(fig3, fullfile(out_dir, 'fig3_scaled_features.png'));

%% --- FINAL SUMMARY ------------------------------------------------
fprintf('\n');
fprintf('  Training set (pre-SMOTE):  %4d x %d  →  X_train_scaled\n', ...
        size(X_train_scaled,1), n_feat);
fprintf('  Training set (post-SMOTE): %4d x %d  →  X_train_smote\n', ...
        size(X_train_smote,1), n_feat);
fprintf('  Test set (quarantined):    %4d x %d  →  X_test_scaled\n', ...
        size(X_test_scaled,1), n_feat);

%% ================================================================
%  LOCAL FUNCTIONS
% ================================================================
function [X_scaled, center, scale] = robust_scale_fit(X)
% ROBUST_SCALE_FIT  Fit a robust scaler on matrix X.
%
%   Calculates the median and IQR for each feature
%   column. 
%   If IQR == 0 , the scale is set to 1.0 to avoid
%   division by zero
%   The scikit-learn convention.
%
%   INPUTS:
%     X        [n x p] feature matrix (numeric, no NaN)
%
%   OUTPUTS:
%     X_scaled [n x p] scaled matrix: (X - median) / IQR
%     center   [1 x p] median of each column
%     scale    [1 x p] IQR of each column (1.0 if IQR was 0)
    [~, p] = size(X);
    center = median(X);                         % [1 x p]
    q1     = quantile(X, 0.25);                 % [1 x p]
    q3     = quantile(X, 0.75);                 % [1 x p]
    scale  = q3 - q1;                           % [1 x p]

    % Zero-IQR fallback
    scale(scale == 0) = 1.0;

    % Apply: broadcast subtract then divide
    X_scaled = (X - center) ./ scale;
end

function X_scaled = robust_scale_apply(X, center, scale)
% ROBUST_SCALE_APPLY  Apply pre-fit scaler parameters to new data.
%
%   Uses the median and IQR calculated by robust_scale_fit on the
%   TRAINING set. 
%   Must never be called with parameters refit on
%   the data being transformed.
%
%   INPUTS:
%     X        [n x p] feature matrix (same column layout as training)
%     center   [1 x p] from robust_scale_fit
%     scale    [1 x p] from robust_scale_fit
%
%   OUTPUT:
%     X_scaled [n x p] scaled using training parameters

    X_scaled = (X - center) ./ scale;
end

function [X_bal, y_bal] = smote_balance(X, y, target_n, k, seed)
% SMOTE_BALANCE  Synthetic Minority Over-sampling Technique.
%
%   For each minority class, generates (target_n - n_class) synthetic
%   samples by k-NN interpolation between real minority samples.
%
%   ALGORITHM (Chawla et al., 2002):
%     For each minority sample x_i:
%       1. Find its k nearest neighbours in the same class.
%       2. Randomly pick one neighbour x_nn.
%       3. Generate: x_new = x_i + lambda * (x_nn - x_i)
%          where lambda ~ Uniform(0, 1).
%
%   NOTE: The majority class is NOT subsampled. Only minority classes
%   are oversampled. All original samples are preserved in X_bal.

    rng(seed);
    classes   = unique(y);
    X_bal     = X;
    y_bal     = y;

    for c = classes'
        idx_c = find(y == c);
        n_c   = length(idx_c);

        if n_c >= target_n
            continue;   % Majority class — no action needed
        end

        n_needed = target_n - n_c;
        X_c      = X(idx_c, :);     % Real samples of class c

        % Build kNN model within this class only
        % knnsearch returns k+1 neighbours; col 1 is the point itself
        [nn_idx, ~] = knnsearch(X_c, X_c, 'K', k + 1);
        nn_idx = nn_idx(:, 2:end);  % Drop self-match (column 1)

        % Generate synthetic samples
        X_synthetic = zeros(n_needed, size(X, 2));
        for s = 1:n_needed
            % Pick a random real sample from this class
            anchor_pos = randi(n_c);
            x_anchor   = X_c(anchor_pos, :);

            % Pick a random neighbour of that sample
            nn_pos     = randi(k);
            x_nn       = X_c(nn_idx(anchor_pos, nn_pos), :);

            % Interpolate
            lambda          = rand();   % Uniform(0,1)
            X_synthetic(s,:) = x_anchor + lambda * (x_nn - x_anchor);
        end

        % Append synthetic samples
        X_bal = [X_bal; X_synthetic];
        y_bal = [y_bal; repmat(c, n_needed, 1)];

        fprintf('    Class %d: %d real + %d synthetic = %d total\n', ...
                c, n_c, n_needed, target_n);
    end

    fprintf('\n');
end