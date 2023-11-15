function compositeImg = CompositeImages(bg, fg, fgA, pos)
% CompositeImages(bg, fg, fgA, pos) Pastes the fg image over the bg image at 
%                                   position pos by using the fgA alpha mask.
%                                   Note that this mask is in the original
%                                   coordinates of the fg image. Pos is a
%                                   2D (y,x) coordinate
%
% Created by Randolph Voorhies (voorhies at usc dot edu)

%Copy the background image to the composite image
compositeImg=bg;

%Determine the maximum and minimum dimensions of the overlay given the sizes
%of both images, and the requested pasting position
minFGx = round(max(1, 1-pos(2)));                       %Minimum foreground x coordinate
minFGy = round(max(1, 1-pos(1)));                       %Minimum foreground y coordinate
maxFGx = round(min(size(bg,2) - pos(2), size(fg,2)));   %Maximum foreground x coordinate
maxFGy = round(min(size(bg,1) - pos(1), size(fg,1)));   %Maximum foreground x coordinate

minBGx = round(minFGx+pos(2));
minBGy = round(minFGy+pos(1));
maxBGx = round(maxFGx+pos(2));
maxBGy = round(maxFGy+pos(1));

%Convert the alpha mask to 0.0 - 1.0 scale and replicate it across the R G 
%and B channels
fgAd        = double(fgA)/255.0;
fgAd(:,:,2) = fgAd(:,:,1);
fgAd(:,:,3) = fgAd(:,:,1);

%Crop the alpha mask to fit in the background image
fgAd = fgAd(minFGy:maxFGy, minFGx:maxFGx,:);

%Prepare the chunk of the background that will have the foreground overlayed on
%top of it by removing pixels respective of the alpha mask
bg_underlay = bg(minBGy:maxBGy, minBGx:maxBGx,:);
bg_underlay = uint8(double(bg_underlay) .* (1-fgAd));

%Prepare the foreground by removing pixels respective of the alpha mask
fg_overlay =  uint8(double(fg(minFGy:maxFGy, minFGx:maxFGx, :)) .* double(fgAd));

%Set the requested chunk of the image to the sum of the background underlay and the
%foreground overlay
compositeImg(minBGy:maxBGy, minBGx:maxBGx,:) = bg_underlay + fg_overlay;





