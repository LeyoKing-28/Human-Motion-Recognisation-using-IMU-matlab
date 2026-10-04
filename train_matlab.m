
%% 1. Configuration & Paths
datasetDir = fullfile('dataset', 'harth');
csvFiles = dir(fullfile(datasetDir, '*.csv'));

if isempty(csvFiles)
    error('No CSV files found in %s. Please make sure the dataset is extracted correctly.', datasetDir);
end

fprintf('Found %d subject files in %s.\n', length(csvFiles), datasetDir);

% Split subjects (Leave-One-Subject-Out style division)
% Train subjects: 1 to 15
% Val subjects: 16 to 18
% Test subjects: 19 to 22
trainFiles = csvFiles(1:15);
valFiles = csvFiles(16:18);
testFiles = csvFiles(19:end);

%% 2. Load and Standardize Features
fprintf('\n--- Loading and preprocessing data ---\n');
[trainFeatures, trainLabels] = loadSubjectData(trainFiles, datasetDir);
[valFeatures, valLabels] = loadSubjectData(valFiles, datasetDir);
[testFeatures, testLabels] = loadSubjectData(testFiles, datasetDir);

% Calculate global mean and std from training data for standardization
allTrainFeatures = vertcat(trainFeatures{:});
mu = mean(allTrainFeatures, 1);
sigma = std(allTrainFeatures, 0, 1);

% Apply standardization to all datasets
for i = 1:length(trainFeatures)
    trainFeatures{i} = (trainFeatures{i} - mu) ./ sigma;
end
for i = 1:length(valFeatures)
    valFeatures{i} = (valFeatures{i} - mu) ./ sigma;
end
for i = 1:length(testFeatures)
    testFeatures{i} = (testFeatures{i} - mu) ./ sigma;
end

%% 3. Segment Data into Sliding Windows (window=128, step=64)
windowSize = 128;
stepSize = 64;

fprintf('\n--- Segmenting signals into windows ---\n');
[X_train, Y_train] = segmentIntoWindows(trainFeatures, trainLabels, windowSize, stepSize);
[X_val, Y_val] = segmentIntoWindows(valFeatures, valLabels, windowSize, stepSize);
[X_test, Y_test] = segmentIntoWindows(testFeatures, testLabels, windowSize, stepSize);

fprintf('Train samples: %d\n', length(X_train));
fprintf('Val samples: %d\n', length(X_val));
fprintf('Test samples: %d\n', length(X_test));

%% 4. Define and Train the 1D CNN Model
numFeatures = 6;
sequenceLength = 128;
numClasses = 12;

% Create layer array (using the helper model function)
layers = model_matlab(numFeatures, sequenceLength, numClasses);

% Training options
options = trainingOptions('adam', ...
    'InitialLearnRate', 0.001, ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 50, ...
    'Plots', 'training-progress', ...
    'Verbose', true);

fprintf('\n--- Starting Training in MATLAB ---\n');
% Note: trainNetwork is standard in MATLAB's Deep Learning Toolbox.
% To execute this, run: net = trainNetwork(X_train, Y_train, layers, options);

%% Helper Functions

function [allFeatures, allLabels] = loadSubjectData(filesList, datasetDir)
    % Loads data from subject files and keeps only relevant columns and mapped labels
    allFeatures = cell(length(filesList), 1);
    allLabels = cell(length(filesList), 1);
    
    for i = 1:length(filesList)
        filePath = fullfile(datasetDir, filesList(i).name);
        fprintf('Loading %s...\n', filesList(i).name);
        
        % Read CSV file using readtable
        opts = detectImportOptions(filePath);
        opts.SelectedVariableNames = {'back_x', 'back_y', 'back_z', 'thigh_x', 'thigh_y', 'thigh_z', 'label'};
        tbl = readtable(filePath, opts);
        
        % Clean rows containing NaNs
        cleanRows = ~any(isnan(tbl{:, :}), 2);
        tbl = tbl(cleanRows, :);
        
        features = tbl{:, 1:6};
        labels = tbl.label;
        
        % Map label values to categories
        % Original label values: [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140]
        % We map them to indices: [1, 2, 3, 4, 5, 6, 7, 8,  9,  10, 11,  12]
        mappedLabels = zeros(size(labels));
        originalLabels = [1, 2, 3, 4, 5, 6, 7, 8, 13, 14, 130, 140];
        
        for k = 1:length(originalLabels)
            mappedLabels(labels == originalLabels(k)) = k;
        end
        
        % Filter out any samples that were not matched (e.g. label=0 or invalid ones)
        validIdx = mappedLabels > 0;
        allFeatures{i} = features(validIdx, :);
        allLabels{i} = mappedLabels(validIdx);
    end
end

function [X_cells, Y_cat] = segmentIntoWindows(featuresCell, labelsCell, windowSize, stepSize)
    % Segments features into cell array and labels into categorical array
    X_cells = {};
    Y_vals = [];
    
    for s = 1:length(featuresCell)
        features = featuresCell{s};
        labels = labelsCell{s};
        nSamples = size(features, 1);
        
        for i = 1:stepSize:(nSamples - windowSize + 1)
            % Extract window and transpose to (numFeatures-by-windowSize)
            window = features(i : i + windowSize - 1, :)';
            
            % Set window label to the label of the center element
            label = labels(i + floor(windowSize / 2));
            
            X_cells{end+1, 1} = window; %#ok<AGROW>
            Y_vals(end+1, 1) = label; %#ok<AGROW>
        end
    end
    
    % Convert labels to categorical array for classification
    classNames = {
        'Walking', ...
        'Running', ...
        'Shuffling', ...
        'Stairs (ascending)', ...
        'Stairs (descending)', ...
        'Standing', ...
        'Sitting', ...
        'Lying', ...
        'Cycling (sit)', ...
        'Cycling (stand)', ...
        'Cycling (sit, inactive)', ...
        'Cycling (stand, inactive)'
    };
    Y_cat = categorical(Y_vals, 1:12, classNames);
end
