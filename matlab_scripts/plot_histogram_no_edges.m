function plot_histogram_no_edges(data, num_bins)
    % Flatten the 3D tensor into a 1D vector
    data_flattened = data(:);

    % Calculate mean and standard deviation
    mu = mean(data_flattened);
    sigma = std(data_flattened);

    % Calculate bin edges covering ±3 standard deviations from the mean
    bin_edges = linspace(mu - 3*sigma, mu + 3*sigma, num_bins+1);

    % Create a histogram without bin edges
    histogram(data_flattened, 'BinEdges', bin_edges, 'EdgeColor', 'none', 'FaceAlpha', 1);

    xlabel('Value');
    ylabel('Frequency');
    title('Histogram without Bin Edges');

    % Display histogram statistics
    disp('Histogram Statistics:');
    disp(['Mean = ', num2str(mu)]);
    disp(['Standard Deviation = ', num2str(sigma)]);
    disp(['Number of Bins = ', num2str(num_bins)]);
end
