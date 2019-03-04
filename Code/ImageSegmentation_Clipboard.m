function [label,scores] = ImageSegmentation_Clipboard(I)
redChannel = I(:, :, 1);
greenChannel = I(:, :, 2);
blueChannel = I(:, :, 3);

binaryImage = blueChannel > 210;
binaryImage = redChannel > 200;
binaryImage = greenChannel > 190;

% Identify individual blobs by seeing which pixels are connected to each other.
% Each group of connected pixels will be given a label, a number, to identify it and distinguish it from the other blobs.
% Do connected components labeling with either bwlabel() or bwconncomp().

labeledImage = bwlabel(binaryImage, 8);     % Label each blob so we can make measurements of it
stats = regionprops(logical(binaryImage), 'Area', 'Solidity');
ind = ([stats.Solidity] > 0.5);
L = bwlabel(binaryImage);
result = ismember(L, find(ind));
%%
se = strel('disk',8);
J2 = imopen(result, se);
%%
% Get all the blob properties.  Can only pass in originalImage in version R2008a and later.
H = vision.BlobAnalysis('AreaOutputPort', true, 'CentroidOutputPort', true, 'BoundingBoxOutputPort', true);
[Area, Centroid, BBox] = step(H, J2);
BBox=(double(BBox))
Size = size(BBox)
st = regionprops(J2, 'BoundingBox','Area');
for k = 1:Size(1)
	if (BBox(k,3)/BBox(k,4)) > 1 && (BBox(k,3)/BBox(k,4))< 1.7 && st(k).Area > 150000;
    	out = imcrop(I,BBox(k,:));
        Szz = size(out);
        if mean2(out) > 160 && Szz(1)>100; 
        else
        end
    end
end

%%
load('OCRClassifier.mat')
net = OCRClassifier;
im = imresize(out,[227 227]);
[label,scores] = classify(net,im)

end

