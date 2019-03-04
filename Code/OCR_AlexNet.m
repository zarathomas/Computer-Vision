%%
imds = imageDatastore('C:\Users\zarat\OneDrive\Msc Data Science\Computer Vision\Face Detection\OCR012','IncludeSubfolders',true,'LabelSource',...
    'foldernames');
imds.ReadSize = numpartitions(imds)
imds.ReadFcn = @(loc)imresize(imread(loc),[227,227]);
%%
[imdsTrain,imdsTest] = splitEachLabel(imds,0.7,'randomized');
%% Alex Net 
net = alexnet; 
net.Layers
layersTransfer = net.Layers(1:end-3)
inputSize = net.Layers(1).InputSize
layersTransfer = net.Layers(1:end-3);
numClasses = numel(categories(imdsTrain.Labels))
%%
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];
%% 
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter( ...
    'RandXReflection',true, ...
    'RandXTranslation',pixelRange, ...
    'RandYTranslation',pixelRange);
augimdsTrain = augmentedImageDatastore(inputSize(1:2),imdsTrain, ...
    'DataAugmentation',imageAugmenter);
%%
augimdsTest = augmentedImageDatastore(inputSize(1:2),imdsTest);
%%
options = trainingOptions('sgdm', ...
    'MiniBatchSize',10, ...
    'MaxEpochs',6, ...
    'InitialLearnRate',1e-3, ...
    'ValidationData',augimdsTest, ...
    'ValidationFrequency',3, ...
    'ValidationPatience',Inf, ...
    'Verbose',false, ...
    'Plots','training-progress',...
    'ExecutionEnvironment','parallel');
%% Train a 54 class SVM classifier including unknown individuals from group photos 
OCRClassifier = trainNetwork(augimdsTrain,layers,options);
%% Predict the labels for the test set 
[YPred,scores] = classify(OCRClassifier,augimdsTest);
%%
idx = randperm(numel(imdsTest.Files),4);
figure
for i = 1:4
    subplot(2,2,i)
    I = readimage(imdsTest,idx(i));
    imshow(I)
    label = YPred(idx(i));
    title(string(label));
end

%% Produces accuracies of AlexNet Model against Test set 
YTest = imdsTest.Labels;
accuracy = mean(YPred == YTest)