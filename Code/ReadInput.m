function [I] = ReadVideo(filename)
        try 
            I = imread(filename);
        end
            
        if exist('I','var')==1
            try 
                v=VideoReader(filename);
            end
        else 
        end
        
        if exist('v','var')==1
            for img = 1:v.NumberOfFrames;
                I = read(v,img);
                bbox = step(faceDetector,I);
                if isempty(bbox) == 1
                    I = imrotate(I,270);
                    bbox = step(faceDetector,I)
                else
                end
            end
        else
        end
end
        
        