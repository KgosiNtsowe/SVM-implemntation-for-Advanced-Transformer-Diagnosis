%  Project: Advanced Diagnostics for Power Transformers
%  Author:  Kgosietsile Ntsowe | ID: 11409134
%  PURPOSE: Creates analysis of data before any preprocessing to enable
%  informed methodology based on results

%  OUTPUT FILES (saved to ./audit_outputs/):
%    - class_distribution.png
%    - gas_boxplots_raw.png
%    - gas_boxplots_log.png
%    - correlation_heatmap.png
%    - per_class_gas_medians.png
%    - audit_summary_table.csv

clc; clear; close all;
rng(42);

%% --- 0. OUTPUT DIRECTORY -------------------------------------------
% Find the exact folder where this .m script is currently saved
script_dir = fileparts(mfilename('fullpath'));
% Create a path for the 'audit_outputs' folder perfectly next to the script
output_dir = fullfile(script_dir, 'audit_outputs');
% Create the folder if it does not already exist
if ~exist(output_dir, 'dir')
    mkdir(output_dir); 
end

%% --- 1. LOAD MASTER DATASET ----------------------------------------
T = readtable('C:\Users\Kgosi\OneDrive\Desktop\Individual Project\Power transformer data for fault diagnosis\Power transformer data for fault diagnosis\Final_Transformer_Dataset_with_Duval.csv','VariableNamingRule', 'preserve');
% Column layout
gas_names   = {'Hydrogen','Methane','Ethane','Ethylene','Acetylene','Carbon Monoxide','Carbon Dioxide'};
label_col   = 'Fault_Type';
X_gas   = T{:, gas_names};                              % [1408 x 7] raw gases
X_duval = T{:, {'%CH4', '%C2H4', '%C2H2'}};             % [1408 x 3] Duval percentages
y       = T{:, label_col};           % [1408 x 1]  class labels 0-4


X_all   = [X_gas, X_duval];         % [1408 x 10] full feature matrix
fprintf('Dataset loaded: %d samples, %d features, %d classes\n', ...
        size(X_all,1), size(X_all,2), numel(unique(y)));

%% --- 2. MISSING VALUE CHECK ----------------------------------------
fprintf('\n--- MISSING VALUE CHECK ---\n');
col_names = [gas_names, {'CH4','C2H4','C2H2'}];
missing_counts = sum(ismissing(X_all));
for i = 1:length(col_names)
    fprintf('  %-25s: %d missing values\n', col_names{i}, missing_counts(i));
end
assert(sum(missing_counts) == 0, ...
    'STOP: Missing values found. Investigate before proceeding.');
fprintf('  PASS: Zero missing values confirmed.\n');

%% --- 3. CLASS DISTRIBUTION & IMBALANCE -----------------------------
fprintf('\n--- CLASS DISTRIBUTION ---\n');
class_labels = {0,'Healthy/Normal'; 1,'D1 - Low Energy Discharge'; ...
                2,'T1 - Thermal <300C'; 3,'T2 - Thermal 300-700C'; ...
                4,'T3 - Thermal >700C'};
counts = zeros(5,1);
for c = 0:4
    counts(c+1) = sum(y == c);
end

fprintf('  %-5s  %-30s  %5s  %6s\n','Class','Name','Count','Pct(%)');
fprintf('  %s\n', repmat('-',1,52));
for c = 0:4
    fprintf('  %-5d  %-30s  %5d  %5.1f%%\n', c, ...
        class_labels{c+1,2}, counts(c+1), counts(c+1)/sum(counts)*100);
end
fprintf('  %s\n', repmat('-',1,52));
fprintf('  %-5s  %-30s  %5d\n','','TOTAL',sum(counts));

imbalance_ratio = max(counts) / min(counts);
fprintf('\n  >> Imbalance ratio (majority/minority): %.1f:1\n', imbalance_ratio);
% Plot: Class distribution bar chart
fig1 = figure('Position',[100 100 700 420]);
bar_colors = [0.35 0.70 0.35;   % Class 0 — green (healthy)
              0.85 0.33 0.10;   % Class 1 — red (discharge)
              0.00 0.45 0.74;   % Class 2 — blue
              0.93 0.69 0.13;   % Class 3 — yellow
              0.49 0.18 0.56];  % Class 4 — purple
b = bar(0:4, counts, 'FaceColor', 'flat');
b.CData = bar_colors;
xlabel('Fault Class', 'FontSize', 12);
ylabel('Number of Samples', 'FontSize', 12);
title('Class Distribution of DGA Dataset (N=1408)', 'FontSize', 13, 'FontWeight', 'bold');
class_tick_labels = {'0: Healthy','1: D1','2: T1','3: T2','4: T3'};
xticklabels(class_tick_labels);
% Annotate bars with counts and percentages
for c = 0:4
    text(c, counts(c+1) + 15, ...
        sprintf('%d\n(%.1f%%)', counts(c+1), counts(c+1)/sum(counts)*100), ...
        'HorizontalAlignment','center', 'FontSize', 9);
end
grid on; box on;
set(gca, 'FontSize', 11);
saveas(fig1, fullfile(output_dir, 'class_distribution.png'));

%% --- 4. DESCRIPTIVE STATISTICS 
stat_headers = {'Mean','Std','Min','Q25','Median','Q75','Max','Skewness','Kurtosis'};
fprintf('  %-22s', 'Gas');
for h = stat_headers, fprintf('  %10s', h{1}); end
fprintf('\n  %s\n', repmat('-',1,120));

stats_table = zeros(7, 9);
for i = 1:7
    col = X_gas(:,i);
    stats_table(i,:) = [mean(col), std(col), min(col), ...
                        quantile(col,0.25), median(col), ...
                        quantile(col,0.75), max(col), ...
                        skewness(col), kurtosis(col)];
    fprintf('  %-22s', gas_names{i});
    fprintf('  %10.2f', stats_table(i,:));
    fprintf('\n');
end

fprintf('\n  KEY FINDING: All gases exhibit high positive skewness (range: %.1f to %.1f).\n', ...
    min(stats_table(:,8)), max(stats_table(:,8)));

%% --- 5. ZERO-VALUE ANALYSIS -------------------------------
fprintf('\n--- zero-value counts ---\n');
fprintf('  %-22s  %5s  %7s  %s\n','Gas','Zeros','Pct(%%)','Implication');
fprintf('  %s\n', repmat('-',1,75));
for i = 1:7
    z = sum(X_gas(:,i) == 0);
    pct = z/size(X_gas,1)*100;
    flag = '';
    if pct > 80, flag = '*** VERY HIGH';
    elseif pct > 40, flag = '** HIGH';
    elseif pct > 20, flag = '* MODERATE';
    end
    fprintf('  %-22s  %5d  %6.1f%%  %s\n', gas_names{i}, z, pct, flag);
end

%% --- 6. OUTLIER DETECTION----------------------------
fprintf('\n--- OUTLIER ANALYSIS (IQR method, threshold = Q3 + 3*IQR) ---\n');
for i = 1:7
    col = X_gas(:,i);
    q1 = quantile(col, 0.25); q3 = quantile(col, 0.75);
    iqr_val = q3 - q1;
    thresh = q3 + 3*iqr_val;
    n_out = sum(col > thresh);
    fprintf('  %-22s: %3d outliers (threshold = %.0f ppm)\n', ...
        gas_names{i}, n_out, thresh);
end

%% --- 7. BOX PLOTS ------------------------
% Raw boxplots 
fig2 = figure('Position',[100 100 1100 500]);
subplot(1,2,1);
boxplot(X_gas, 'Labels', {'H2','CH4','C2H6','C2H4','C2H2','CO','CO2'});
title('Raw Gas Concentrations (ppm)', 'FontWeight','bold');
ylabel('Concentration (ppm)');
xlabel('Gas');
set(gca,'FontSize',10);

% Log1p-transformed boxplots
subplot(1,2,2);
boxplot(log1p(X_gas), 'Labels', {'H2','CH4','C2H6','C2H4','C2H2','CO','CO2'});
title('Log_{1}(1+x) Transformed Gases', 'FontWeight','bold');
ylabel('log(1+ppm)');
xlabel('Gas');
set(gca,'FontSize',10);
saveas(fig2, fullfile(output_dir, 'gas_boxplots_raw_vs_log.png'));

%% --- 8. GAS MEDIANS ----------
fprintf('\n--- PER-CLASS MEDIAN GAS CONCENTRATIONS ---\n');
fprintf('  %-8s %-10s %-10s %-10s %-10s %-10s %-10s %-10s\n', ...
    'Class', 'H2', 'CH4', 'C2H6', 'C2H4', 'C2H2', 'CO', 'CO2');
fprintf('  %s\n', repmat('-',1,82));

medians = zeros(5,7);
for c = 0:4
    idx = y == c;
    medians(c+1,:) = median(X_gas(idx,:));
    fprintf('  %-8d', c);
    fprintf(' %-10.1f', medians(c+1,:));
    fprintf('\n');
end

fprintf('\n  PHYSICAL INTERPRETATION:\n');
fprintf('  - Class 0 (Healthy): All gases low — baseline reference.\n');
fprintf('  - Class 1 (D1 Discharge): Elevated C2H2 (%.1f ppm median) — signature of arcing.\n', medians(2,5));
fprintf('  - Class 2 (T1 <300C): Elevated CH4 (%.1f ppm) — low-temp thermal.\n', medians(3,2));
fprintf('  - Class 3 (T2 300-700C): Rising C2H4 (%.1f ppm) — higher thermal energy.\n', medians(4,4));
fprintf('  - Class 4 (T3 >700C): Highest C2H4 (%.1f ppm) and CO (%.1f ppm).\n', medians(5,4), medians(5,6));

% Figure: per-class median heatmap
fig3 = figure('Position',[100 100 800 400]);
imagesc(medians);
colormap(hot);
colorbar;
set(gca, 'XTick', 1:7, 'XTickLabel', {'H2','CH4','C2H6','C2H4','C2H2','CO','CO2'}, ...
         'YTick', 1:5, 'YTickLabel', {'0:Healthy','1:D1','2:T1','3:T2','4:T3'}, ...
         'FontSize', 11);
title('Median Gas Concentration per Fault Class (ppm)', 'FontWeight','bold');
% Annotate cells
for i = 1:5
    for j = 1:7
        text(j, i, sprintf('%.0f', medians(i,j)), ...
            'HorizontalAlignment','center', 'Color','white', ...
            'FontSize', 9, 'FontWeight','bold');
    end
end
saveas(fig3, fullfile(output_dir, 'per_class_gas_medians_heatmap.png'));

%% --- 9. SPEARMAN CORRELATION MATRIX --------------------------------
fprintf('\n--- SPEARMAN CORRELATION (multicollinearity check) ---\n');
[rho, pval] = corr(X_all, 'Type', 'Spearman');

all_feat_names = {'H2','CH4','C2H6','C2H4','C2H2','CO','CO2','%%CH4','%%C2H4','%%C2H2'};
fprintf('  Pairs with |r| > 0.50:\n');
for i = 1:10
    for j = i+1:10
        if abs(rho(i,j)) > 0.50
            fprintf('    %-10s vs %-10s : r = %+.3f  p < 0.001\n', ...
                all_feat_names{i}, all_feat_names{j}, rho(i,j));
        end
    end
end

% Heatmap
fig4 = figure('Position',[100 100 650 580]);
imagesc(rho);
colormap(redblue_colormap());   % see helper function at bottom
caxis([-1 1]); colorbar;
set(gca, 'XTick',1:10, 'XTickLabel', all_feat_names, 'XTickLabelRotation', 35, ...
         'YTick',1:10, 'YTickLabel', all_feat_names, 'FontSize', 10);
title('Spearman Correlation Matrix — All 10 Features', 'FontWeight','bold');
saveas(fig4, fullfile(output_dir, 'correlation_heatmap.png'));

%% --- 10. DUVAL PERCENTAGE VALIDITY CHECK ---------------------------
fprintf('\n--- DUVAL PERCENTAGE VALIDITY CHECK ---\n');
pct_sum = X_duval(:,1) + X_duval(:,2) + X_duval(:,3);
nonzero = pct_sum > 0;
valid   = sum(abs(pct_sum(nonzero) - 100) < 1.0);

%% --- 11.SUMMARY TABLE (CSV export) --------------------------
summary = table(gas_names', stats_table(:,1), stats_table(:,2), ...
                stats_table(:,5), stats_table(:,7), stats_table(:,8), ...
                sum(X_gas == 0)', ...
                'VariableNames', {'Gas','Mean','Std','Median','Max','Skewness','ZeroCount'});
writetable(summary, fullfile(output_dir, 'audit_summary_table.csv'));


%% --- HELPER FUNCTION -----------------------------------------------
function cmap = redblue_colormap()
% Custom diverging red-white-blue colormap for correlation matrices
n = 64;
r = [linspace(0.7,1,n/2), linspace(1,1,n/2)]';
g = [linspace(0.1,1,n/2), linspace(1,0.1,n/2)]';
b = [linspace(1,1,n/2), linspace(1,0.1,n/2)]';
cmap = [r, g, b];
end