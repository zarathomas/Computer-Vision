function [id] = detectNum(filename)

I = ReadInput(filename);

[label1,scores1]=ImageSegmentation_Clipboard(I);
[label2,scores2]=ImageSegmentation_Clipboard(I);

if max(scores1) > max(scores2)
    id = label1;
else 
    id = label2;
end

end