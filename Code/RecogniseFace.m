function [ P ] = RecogniseFace1(I, featureType, classifierName)
    % Given image I, a featureType and a classifierName, the function will 
    % output a Nx3 matrix where each row represents the ID, central x location, 
    % central y location and emotion of the person identified in an image.
    % The featureType and classifierName are the tools used to carry out face
    % recognition.

    % All detected faces in image I are saved in P.

    P = [];

    % This face detector uses the Viola-Jones algorithm. 
    FaceDetector = vision.CascadeObjectDetector();
    % Increase the merge threshold from the default of 4 to avoid false 
    % positive face detection 
    FaceDetector.MergeThreshold = 6;
    bbox = step(FaceDetector, I);

    N = size(bbox,1);

    % Loops through the bounding boxes of detected faces and extracts the 
    % faces where they are eventually saved.
    for i=1:N
        faceNum = i;
        % Extract the ith face
        a = bbox(i, 1);
        b = bbox(i, 2);
        c = a+bbox(i, 3);
        d = b+bbox(i, 4);
        F = I(b:d, a:c, :);
        % create a directory to store the faces once they have been
        % cropped
        mkdir face; 
        filename = strcat('face/' ,num2str(faceNum),'.jpg');

        imwrite(F, filename);
        F = imresize(F,[227 227]);
        % Use the (a,c) and (b, d) coordinates of the bounding box to 
        % determine the central face region (x and y coordinates) 
        % of the person detected
        x = int32((a + c)/2);
        y = int32((b + d)/2);

        % Load emotionClassifier
        load('EmotionClassifier.mat');
        
        % SVM using Histogram of Gradients Features
       % if isequal(classifierName, 'SVM') && isequal(featureType, 'HOG')
            %load SVMHOGClassifier.mat;
            %features = extractHOGFeatures(F);
            %id = predict(SVMHOGClassifier, features);
            %emotion = classify(EmotionClassifier,F);
            
        % SVM using Bag of Features
        if isequal(classifierName, 'SVM') && isequal(featureType, 'BAG')
           load('SVMBAGClassifier.mat');
           id = predict(categoryClassifier, F);
           emotion = classify(EmotionClassifier,F);
                
        % KNN using Histogram of Gradient Features
        elseif isequal(classifierName, 'KNN') && isequal(featureType, 'HOG')
            load('KNNHOGClassifier.mat');
            features = extractHOGFeatures(F);
            id = predict(KNNHOGClassifier, features);
            emotion = classify(EmotionClassifier,F);
            
        % AlexNet
        elseif isequal(classifierName, 'ALEX') && isequal(featureType, 'NIL')
            load('AlexNet.mat');
            id = classify(AlexNet,F);
            emotion = classify(EmotionClassifier,F);
        else
            disp('Please choose the correct classifier and feature type');
            return
        end
        
        if emotion == 'happy'
            e = 0;
        elseif emotion == 'sad'
            e = 1;
        elseif emotion == 'surprised'
            e = 2;
        else 
            e = 3;
        end
        e = int32(e)
        P = [P; int32(id), x,y,e];
    end
end