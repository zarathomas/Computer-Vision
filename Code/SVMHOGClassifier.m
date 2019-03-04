% Load extracted faces from directory
% This produces a facedatabase with a 1x53 image datastore structure
imds = imageDatastore('C:\Users\zarat\OneDrive\Msc Data Science\Computer Vision\Face Detection\Face Detection 5','IncludeSubfolders',true,'LabelSource',...
    'foldernames');

imds.ReadSize = numpartitions(imds)
imds.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

% Partition the data into training and test sets (70%,30%) respectively
[trainingSet,testSet] = splitEachLabel(imds,0.7,'randomize');

%%
img = readimage(trainingSet, 2);
% Extract HOG features and HOG visualization
[hog_10x10, vis10x10] = extractHOGFeatures(img,'CellSize',[10 10]);

%%
cellSize = [10 10];
hogFeatureSize = length(hog_10x10);
%%
numImages = numel(trainingSet.Files);
trainingFeatures = zeros(numImages, hogFeatureSize, 'single');
%%
% Loop over the trainingSet and extract HOG features from each image. A
% similar procedure will be used to extract features from the testSet.
for i = 1:numImages
    img = readimage(trainingSet, i);
    
    img = rgb2gray(img);
    
    % Apply pre-processing steps
    img = imbinarize(img);
    
    trainingFeatures(i, :) = extractHOGFeatures(img, 'CellSize', cellSize);  
end

% Get labels for each image.a
trainingLabels = trainingSet.Labels;

%% fitcecoc uses SVM learners and a 'One-vs-One' encoding scheme.
%pool = parpool; % Invoke workers for parallelization 
t = templateSVM('KernelFunction','gaussian','BoxConstraint',100, 'KernelScale', 100)
options = statset('UseParallel',true); %Used for Parallelization of work
SVMHOGclassifier = fitcecoc(trainingFeatures, trainingLabels,'Learners',t, 'Options',options);

%% Extract HOG features from the test set. The procedure is similar to what
% was shown earlier and is encapsulated as a helper function for brevity.
[testFeatures, testLabels] = helperExtractHOGFeaturesFromImageSet(testSet, hogFeatureSize, cellSize);
%%
% Make class predictions using the test features.
predictedLabels = predict(SVMHOGclassifier, testFeatures);

% Tabulate the results using a confusion matrix.
confMat = confusionmat(testLabels, predictedLabels);

%% Produces accuracies of Model against validation set 
accuracy = mean(predictedLabels == testSet.Labels);