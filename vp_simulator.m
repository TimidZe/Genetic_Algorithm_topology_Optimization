% Initilization
close all; clear; clc;
% Parameters settings (don't change!)
Lx = 1.0; Ly = Lx; 
nx = 51; ny = 51; k0 = 1.0; k1 = 500.0; T0 = 300.0; q = 100.0;
x = linspace(0, 1, nx); y = linspace(0, 1, ny); [X, Y] = meshgrid(x, y);

% load your prepared k.xls file
k=readmatrix('k.xls');
% check whether the number of unit cells with high thermal conductivity 
fprintf("The number of unit cells with high thermal conductivity: %d\n", sum(sum(k>250)))
% Plot thermal conductivity distribution
figure('Position', [100, 100, 800, 600]);
contourf(X, Y, flipud(k), 'LineStyle', 'none');
colormap(gca, 'jet'); caxis([k0, k1]);
colorbar('Location', 'eastoutside');

set(gca, 'FontSize', 16); % Font size for axes labels and tick labels
title('Thermal Conductivity Distribution','FontSize', 16);
xlabel('$x$', 'Interpreter', 'latex','FontSize', 18);
ylabel('$y$', 'Interpreter', 'latex','FontSize', 18);
saveas(gca, 'k_distribution.png'); % 'gca' gets the current figure handle

%% Calculate temperature matrix
tolerance = 1.0e-5;
[A, b, T] = calculate_temperature_matrix(Lx, 0.1*Lx, q, T0, nx, ny, k, tolerance);
T_reshaped = reshape(T, [nx, ny]); % Reshape temperature matrix
T_flipped = flipud(T_reshaped.');
T_flipped=k0*(T_flipped-T0)/Lx^2/q;
fprintf("The average dimensionless temperature is: %d\n", mean(T_flipped(:)))

%% Plot dimensionless temperature distribution
figure('Position', [100, 100, 800, 600]);
contourf(X, Y, T_flipped, 50, 'LineStyle', 'none');
colormap(gca, 'jet');
% caxis([0, 0.03]); % Set the colorbar range to 0 to 0.01
colorbar('Location', 'eastoutside');
set(gca, 'FontSize', 16); % Font size for axes labels and tick labels
title('Dimensionless Temperature Distribution','FontSize', 16);
xlabel('$x$ (m)', 'Interpreter', 'latex','FontSize', 18);
ylabel('$y$ (m)', 'Interpreter', 'latex','FontSize', 18);
% Save the plot as a PNG file
saveas(gcf, 't_distribution.png'); % 'gcf' gets the current figure handle

%% Calculate index in a matrix as if it were a vector
function index = indexer(i, j, mesh_x)
    index = (i-1)*mesh_x + j;
end

%% Function to compute directional averages
function averages = compute_directional_averages(matrix)
    [rows, cols] = size(matrix);
    averages = zeros(rows, cols, 4);

    for i = 1:rows
        for j = 1:cols
            if i == 1
                averages(i, j, 1) = matrix(i, j) / 2;
            else
                averages(i, j, 1) = (matrix(i, j) + matrix(i-1, j)) / 2;
            end
            
            if i == rows
                averages(i, j, 2) = matrix(i, j) / 2;
            else
                averages(i, j, 2) = (matrix(i, j) + matrix(i+1, j)) / 2;
            end
            
            if j == 1
                averages(i, j, 3) = matrix(i, j) / 2;
            else
                averages(i, j, 3) = (matrix(i, j) + matrix(i, j-1)) / 2;
            end
            
            if j == cols
                averages(i, j, 4) = matrix(i, j) / 2;
            else
                averages(i, j, 4) = (matrix(i, j) + matrix(i, j+1)) / 2;
            end
        end
    end
end

%% Function to calculate temperature 
function [A, b, T] = calculate_temperature_matrix(L, a, q, T0, mesh_x, mesh_y, given_distribution, epsilon)
    A = sparse(mesh_x*mesh_y, mesh_x*mesh_y);
    b = zeros(mesh_x*mesh_y, 1);
    T = ones(mesh_x*mesh_y, 1) * T0;

    delta_x = L / mesh_x;
    delta_y = L / mesh_y;
    delta_x_square = delta_x^2;
    delta_y_square = delta_y^2;

    start_index = int32((L - a) / 2 / delta_x) + 1;
    end_index = int32((L + a) / 2 / delta_x) + 1;

    converged = false;
    k_averages = compute_directional_averages(given_distribution);

    for i = 1:mesh_x
        for j = 1:mesh_y
            idx = indexer(i, j, mesh_x);

            if i == 1
                A(idx, indexer(i+1, j, mesh_x)) = 1;
            end
            if i == mesh_x
                A(idx, indexer(i-1, j, mesh_x)) = 1;
            end
            if i > 1 && i < mesh_x && j == 1
                A(idx, indexer(i, j+1, mesh_x)) = 1;
            end
            if i > 1 && i < mesh_x && j == mesh_y
                A(idx, indexer(i, j-1, mesh_x)) = 1;
            end
            if i > 1 && i < mesh_x && j > 1 && j < mesh_y
                S = (k_averages(i, j, 2) + k_averages(i, j, 1)) / delta_x_square + (k_averages(i, j, 3) + k_averages(i, j, 4)) / delta_y_square;
                A(idx, indexer(i-1, j, mesh_x)) = k_averages(i, j, 1) / S / delta_x_square;
                A(idx, indexer(i+1, j, mesh_x)) = k_averages(i, j, 2) / S / delta_x_square;
                A(idx, indexer(i, j-1, mesh_x)) = k_averages(i, j, 3) / S / delta_y_square;
                A(idx, indexer(i, j+1, mesh_x)) = k_averages(i, j, 4) / S / delta_y_square;
                b(idx) = q / S;
            end
        end
    end

    while ~converged
        T_old = T;
        T = A * T_old + b;
        for i = start_index:end_index
            T(indexer(mesh_x, i, mesh_x)) = T0;
        end
        max_abs_error = mean(abs(T - T_old));
        if max_abs_error < epsilon
            converged = true;
        end
    end
end