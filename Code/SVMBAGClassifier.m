%% Bag of Features 
%%
imds = imageDatastore('C:\Users\zarat\OneDrive\Msc Data Science\Computer Vision\Face Detection\Face Detection 5','IncludeSubfolders',true,'LabelSource',...
    'foldernames');

[trainingSet,testSet] = splitEachLabel(imds,0.7,'randomize');

% This is accomplished with a single call to bagOfFeatures function, which: 
% extracts SURF features from all images in all image categories
% constructs the visual vocabulary by reducing the number of features through 
% quantization of feature space using K-means clustering

%%
%t = templateSVM('KernelFunction','gaussian','BoxConstraint',100, 'KernelScale', 100)
t = templateSVM('KernelFunction','linear')
%Used for Parallelization of work
bag = bagOfFeatures(trainingSet);
categoryClassifier = trainImageCategoryClassifier(trainingSet,bag,'LearnerOptions',t);
%%
confMatrix = evaluate(categoryClassifier,testSet)

%% Mean accuracy of classification
mean(diag(confMatrix))

filename1 = 'ConfusionMatrix_linear.xlsx';
xlswrite(filename1,confMatrix,1);
