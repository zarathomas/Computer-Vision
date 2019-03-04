function [label,scores] = ImageSegmentation_NoClipboard(I)
%%
I = imresize(I,[1333 1000]);
%%
grad = (double(imdilate(I, ones(4))) - double(I)); % extract edges
gradSum = sum(grad, 3);
bw = edge(gradSum, 'Canny');
%%
joined = imdilate(bw, ones(6));
X = label2rgb(bwlabel(joined));
%%
[BW,maskedRGBImage] = createMask(X);
%%
    I=X;
    redChannel = I(:, :, 1);
    greenChannel = I(:, :, 2);
    blueChannel = I(:, :, 3);

    binaryImage = blueChannel > 210;
    binaryImage = redChannel > 200;
    binaryImage = greenChannel > 190;

% Identify individual blobs by seeing which pixels are connected to each other.
% Each group of connected pixels will be given a label, a number, to identify it and distinguish it from the other blobs.
% Do connected components labeling with either bwlabel() or bwconncomp().

    labeledImage = bwlabel(BW, 4);     % Label each blob so we can make measurements of it
    stats = regionprops(logical(BW), 'Area', 'Solidity');
    ind = ([stats.Solidity] > 0.5);
    L = bwlabel(BW);
    result = ismember(L, find(ind));
%%
    K = ones(10,10);
    J1 = imopen(result, K);
    J2 = imclose(J1, K);
%%
% Get all the blob properties.  Can only pass in originalImage in version R2008a and later.
    H = vision.BlobAnalysis('AreaOutputPort', true, 'CentroidOutputPort', true, 'BoundingBoxOutputPort', true,'OrientationOutputPort', true);
    [Area, Centroid, BBox, Orientation] = step(H, J1);
    BBox=(double(BBox))
    Size = size(BBox)
%
    st = regionprops(J1, 'BoundingBox','Area','Solidity');
        for k = 1:Size(1)
        %if (BBox(i,3)/BBox(i,4)) > 1.39 && (BBox(i,3)/BBox(i,4))< 1.41
            if (BBox(k,3)/BBox(k,4)) > 0.3 && (BBox(k,3)/BBox(k,4))< 2.2 && st(k).Area > 20000 && st(k).Area < 250000;
                out = imcrop(I,BBox(k,:));
                Szz = size(out);
                if mean2(out) > 160 && Szz(1)>100
                    out2 = imcrop(I,BBox(k,:));	
                    out = out2;
                else
                end
            else
            end
        end 
%%
load('OCRClassifier.mat')
net = OCRClassifier;
im = imresize(out,[227 227]);
[label,scores] = classify(net,im)
end
        