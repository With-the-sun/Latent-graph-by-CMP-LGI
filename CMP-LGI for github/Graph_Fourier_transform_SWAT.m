% directed_spectral_analysis_nonnegative_50.m
% Directed graph spectral analysis with nonnegative spectrum using L = I - S
% FIXED: Use ONLY the first 50 columns of X (drop the last column if present)
% - Loads Data.mat (expects X: T x n_all), A_P&ID.mat (A), A_CMP-LGI.mat (A)
% - Selects first 50 columns from X -> n = 50
% - Crops adjacency matrices to the 50x50 principal submatrix
% - Z-score per column
% - Removes self-loops
% - Directed RW-symmetric kernel S = (P + P')/2, with P = D_out^{-1} A
% - Laplacian L = I - S (symmetric, eigenvalues in [0, 2] theoretically)
% - GFT-based spectral energy S_i = mean_t( Xhat_i(t)^2 )
% - Dirichlet energy DE = sum_i lambda_i * S_i
% - Edge-based MSE over directed edges
% - Plots spectra (nonnegative) and prints metrics
%
% Author: <LinS Chen>
% License: MIT
% -------------------------------------------------------------------------

clear; clc; close all;

% --------------------- Configuration ---------------------
n_fixed = 50;             % ONLY use the first 50 columns of X
remove_selfloops = true;
rng('default');

fprintf('Loading data and configuring (n = %d)...\n', n_fixed);

% --------------------- 1) Load data ---------------------
assert(exist('Data.mat','file')==2, 'Data.mat not found.');
Sx = load('Data.mat');
assert(isfield(Sx,'X'), 'Data.mat must contain variable X (T x n).');
Xall = double(Sx.X);                % T x n_all
[T_all, n_all] = size(Xall);

assert(n_all >= n_fixed, ...
    'X has only %d columns, but need at least %d to use the first 50.', n_all, n_fixed);

% Enforce first 50 columns
X = Xall(:, 1:n_fixed);
[T, n] = size(X); %#ok<ASGLU>
fprintf('  Using X: %d rows (time) x %d columns (nodes)\n', size(X,1), n);

% --------------------- 1.1) Load adjacency matrices ---------------------
assert(exist('A_P&ID.mat','file')==2, 'A_P&ID.mat not found.');
Sa1 = load('A_P&ID.mat');
assert(isfield(Sa1,'A'), 'A_P&ID.mat must contain variable A.');
A_prior_full = double(Sa1.A ~= 0);  % ensure binary
fprintf('  A_P&ID: %d x %d (full size)\n', size(A_prior_full,1), size(A_prior_full,2));

assert(exist('A_CMP-LGI.mat','file')==2, 'A_CMP-LGI.mat not found.');
Sa2 = load('A_CMP-LGI.mat');
assert(isfield(Sa2,'A'), 'A_CMP-LGI.mat must contain variable A.');
A_our_full = double(Sa2.A ~= 0);
fprintf('  A_CMP-LGI: %d x %d (full size)\n', size(A_our_full,1), size(A_our_full,2));

% --------------------- 1.2) Crop adjacency to 50 x 50 ---------------------
assert(size(A_prior_full,1) >= n && size(A_prior_full,2) >= n, ...
    'A_P&ID must be at least %dx%d; got %dx%d.', n, n, size(A_prior_full,1), size(A_prior_full,2));
assert(size(A_our_full,1) >= n && size(A_our_full,2) >= n, ...
    'A_CMP-LGI must be at least %dx%d; got %dx%d.', n, n, size(A_our_full,1), size(A_our_full,2));

% Assumption: X(:,1:n) corresponds to nodes 1..n in A; if node order differs,
% replace (1:n,1:n) with a permutation index vector 'perm' such that:
% A_prior = A_prior_full(perm, perm); A_our = A_our_full(perm, perm);
A_prior = A_prior_full(1:n, 1:n);
A_our   = A_our_full(1:n, 1:n);
fprintf('  Cropped A_P&ID and A_CMP-LGI to %d x %d\n', n, n);

clear Sa1 Sa2 Sx A_prior_full A_our_full;

% --------------------- 2) Preprocess signal (z-score per column) ---------------------
fprintf('Preprocessing signal: z-score per column...\n');
mu = mean(X, 1);
sd = std(X, 0, 1);
sd(sd == 0) = 1; % avoid division by zero if column is constant
X = (X - mu) ./ sd;   % T x n (normalized)

% --------------------- 3) Process adjacency (directed, remove self-loops) ---------------------
if remove_selfloops
    fprintf('Removing self-loops...\n');
    A_prior(1:end+1:end) = 0;
    A_our(1:end+1:end)   = 0;
end
% Ensure 0/1
A_prior = double(A_prior ~= 0);
A_our   = double(A_our   ~= 0);

m_prior_dir = nnz(A_prior); % number of directed edges
m_our_dir   = nnz(A_our);
fprintf('  A_prior has %d directed edges\n', m_prior_dir);
fprintf('  A_our has %d directed edges\n', m_our_dir);

% --------------------- 4) Build directed L = I - S and eigendecompose ---------------------
fprintf('Building directed Laplacians (L = I - S)...\n');
[L_prior, V_prior, lambda_prior] = build_dir_rw_symmetric_L(A_prior);
[L_our,   V_our,   lambda_our]   = build_dir_rw_symmetric_L(A_our);

% --------------------- 5) Spectral energy and Dirichlet energy ---------------------
fprintf('Computing spectral energy and Dirichlet energy...\n');
[S_prior, DE_prior] = spectral_energy_and_DE(V_prior, lambda_prior, X);
[S_our,   DE_our]   = spectral_energy_and_DE(V_our,   lambda_our,   X);

% Normalize DE per edge (to compare graphs with different edge counts)
DEpe_prior = DE_prior / max(m_prior_dir, 1);
DEpe_our   = DE_our   / max(m_our_dir,   1);

% --------------------- 6) Edge-based MSE over directed edges ---------------------
fprintf('Computing edge-based MSE...\n');
MSE_prior = compute_mse_edge_directed(X, A_prior);
MSE_our   = compute_mse_edge_directed(X, A_our);

% --------------------- 7) Plot spectra (nonnegative eigenvalues) ---------------------
fprintf('Plotting spectra...\n');
figure('Color','w','Units','normalized','Position',[0.08 0.14 0.84 0.62]);

% Prior graph
subplot(1,2,1);
stem(lambda_prior, S_prior, 'b', 'filled', 'Marker', 'o'); hold on;
plot(lambda_prior, S_prior, 'b-', 'LineWidth', 1.0); hold off; grid on;
xlabel('Eigenvalues of L = I - S (Directed RW-symmetric)');
ylabel('Mean squared GFT coeffs');
title('Directed — A_{prior} (nonnegative spectrum)');
txt1 = sprintf(['DE: %.6g\nEdges(dir): %d\nMSE_{edge}: %.6g\nDE/edge: %.6g'], ...
               DE_prior, m_prior_dir, MSE_prior, DEpe_prior);
text(0.55*max(lambda_prior+eps), 0.85*max([S_prior; eps]), txt1, ...
     'Color', [0.85 0 0], 'FontSize', 10.5, 'FontWeight', 'bold');

% Our graph
subplot(1,2,2);
stem(lambda_our, S_our, 'b', 'filled', 'Marker', 'o'); hold on;
plot(lambda_our, S_our, 'b-', 'LineWidth', 1.0); hold off; grid on;
xlabel('Eigenvalues of L = I - S (Directed RW-symmetric)');
ylabel('Mean squared GFT coeffs');
title('Directed — A_{our} (nonnegative spectrum)');
txt2 = sprintf(['DE: %.6g\nEdges(dir): %d\nMSE_{edge}: %.6g\nDE/edge: %.6g'], ...
               DE_our, m_our_dir, MSE_our, DEpe_our);
text(0.55*max(lambda_our+eps), 0.85*max([S_our; eps]), txt2, ...
     'Color', [0.85 0 0], 'FontSize', 10.5, 'FontWeight', 'bold');

sgtitle('Directed graphs: Nonnegative spectrum of L = I - S, Dirichlet energy, edge-MSE, DE per edge', ...
    'FontSize', 13, 'FontWeight', 'bold');

% --------------------- 8) Console report ---------------------
fprintf('\n-------------------------------------\n');
fprintf('Signals: column-wise z-score; self-loops removed.\n');
fprintf('Directed Laplacian: L = I - S, with S = (P + P'')/2 and P = D_out^{-1} A.\n');
fprintf('Spectrum axis = eigenvalues of L (numerically clipped to be nonnegative).\n\n');

print_report('Directed (L = I - S) — A_{prior}', m_prior_dir, DE_prior, DEpe_prior, MSE_prior);
print_report('Directed (L = I - S) — A_{our}',   m_our_dir,   DE_our,   DEpe_our,   MSE_our);
fprintf('-------------------------------------\n');

% --------------------- Local functions ---------------------
function [L_dir, V, lambda] = build_dir_rw_symmetric_L(A)
% build_dir_rw_symmetric_L
% Construct directed RW-symmetric Laplacian:
%   P = D_out^{-1} A  (row-stochastic, add eps for zero-outdegree)
%   S = (P + P')/2    (symmetric kernel)
%   L = I - S         (symmetric; eigenvalues in [0,2] ideally)
% Return eigenpairs (sorted ascending) and clip tiny negative eigenvalues to 0.

    n = size(A,1);
    dout = sum(A, 2);                   % out-degree
    dout(dout == 0) = eps;              % avoid division by zero
    P = diag(1./dout) * A;              % row-stochastic
    S = 0.5 * (P + P');                 % symmetric kernel
    L_dir = eye(n) - S;                 % Laplacian
    L_dir = (L_dir + L_dir')/2;         % enforce symmetry numerically

    % Eigendecomposition
    [V, D] = eig(L_dir);
    lambda = real(diag(D));
    lambda(lambda < 0) = 0;             % clip tiny negatives due to floating-point errors

    % Sort ascending
    [lambda, idx] = sort(lambda, 'ascend');
    V = V(:, idx);
end

function [S_spec, DE] = spectral_energy_and_DE(V, lambda, X)
% spectral_energy_and_DE
% Compute spectral energy per frequency and Dirichlet energy:
%   Xhat = V' * X'         (n x T) Graph Fourier coefficients over time
%   S_i  = mean(Xhat_i.^2) Mean squared coefficient per frequency
%   DE   = sum(lambda .* S)
%
% Inputs:
%   V      - eigenvectors (n x n), columns are eigenvectors
%   lambda - eigenvalues (n x 1), nonnegative, ascending
%   X      - signal matrix (T x n), z-scored per column
%
% Outputs:
%   S_spec - spectral energy per frequency (n x 1)
%   DE     - Dirichlet energy (scalar)

    Xhat = V' * X.';           % n x T
    S_spec = mean(Xhat.^2, 2); % n x 1
    DE = sum(lambda .* S_spec);
end

function mse = compute_mse_edge_directed(X, A)
% compute_mse_edge_directed
% Edge-based MSE over directed edges:
%   For each directed edge (i -> j), compute mean_t( (x_i - x_j)^2 ),
%   then average over all edges.

    [rows, cols] = find(A);
    m = numel(rows);
    if m == 0
        mse = 0;
        return;
    end

    acc = 0;
    for k = 1:m
        i = rows(k); j = cols(k);
        d = X(:, i) - X(:, j);  % T x 1
        acc = acc + mean(d.^2);
    end
    mse = acc / m;
end

function print_report(tag, m_edges, DE, DEpe, MSEe)
% print_report
% Print a concise summary for one graph.

    fprintf('%s\n', tag);
    fprintf('  Edges (directed) = %d\n', m_edges);
    fprintf('  Dirichlet energy = %.6f\n', DE);
    fprintf('  DE per edge      = %.6g\n', DEpe);
    fprintf('  MSE_edge         = %.6f\n\n', MSEe);
end