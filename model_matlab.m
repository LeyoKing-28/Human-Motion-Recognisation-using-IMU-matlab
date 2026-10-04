function layers = model_matlab(numFeatures, sequenceLength, numClasses)
    % MODEL_MATLAB Creates a 1D CNN layer array for Human Activity Recognition (HAR)
    %
    % This model is designed for 1D time-series sensor data classification
    % (e.g., from body-worn IMUs, accelerometers, gyroscopes).
    %
    % Inputs:
    %   numFeatures    - Number of input features/channels (default: 9)
    %   sequenceLength - Length of the input sequence window (default: 128)
    %   numClasses     - Number of activity classes to predict (default: 6)
    %
    % Output:
    %   layers         - A MATLAB layer array representing the 1D CNN model.
    %
    % Example usage:
    %   netLayers = model_matlab(9, 128, 6);
    %   analyzeNetwork(netLayers); % Inspect model architecture in MATLAB

    % Set default parameters if not provided
    if nargin < 1 || isempty(numFeatures)
        numFeatures = 9;
    end
    if nargin < 2 || isempty(sequenceLength)
        sequenceLength = 128;
    end
    if nargin < 3 || isempty(numClasses)
        numClasses = 6;
    end

    layers = [
        % Input Layer
        % Expects a sequence input of dimension numFeatures.
        % The input sequence dimension during training will be numFeatures-by-sequenceLength.
        sequenceInputLayer(numFeatures, 'Name', 'input', 'MinLength', sequenceLength)
        
        % First Convolutional block
        convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
        reluLayer('Name', 'relu1')
        
        % Second Convolutional block
        convolution1dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        reluLayer('Name', 'relu2')
        
        % Dropout layer to prevent overfitting
        dropoutLayer(0.5, 'Name', 'dropout')
        
        % Max Pooling layer to downsample temporal dimensions
        maxPooling1dLayer(2, 'Stride', 2, 'Name', 'maxpool')
        
        % Flatten the 1D feature maps for the fully connected layers
        flattenLayer('Name', 'flatten')
        
        % First Dense / Fully Connected layer
        fullyConnectedLayer(100, 'Name', 'fc1')
        reluLayer('Name', 'relu3')
        
        % Output classification layers
        fullyConnectedLayer(numClasses, 'Name', 'fc2')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classoutput')
    ];
end
