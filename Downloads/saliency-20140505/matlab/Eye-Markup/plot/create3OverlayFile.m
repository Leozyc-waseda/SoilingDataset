function create3OverlayFile(bgimg,ovrimg,ovrimg2,trans,trans1)
%create3overlay('TOOL_HEADER-mod1.png','../../data/processed/freeviewMaps/TOOL_HEADER-mod1-CIO-SM000000.png','../../data/processed/freeviewMaps/TOOL_HEADER-mod1-fixmap.png',.6,.4);
%creates an overlay of an image and a map or two maps.
%The image must be RGB.  If two maps the first map's colormap will be set
%to jet.  
%bgimg = imread(bgimg);

%ovrimg = double(imread(ovrimg));
%overimg = ovrimg./max(ovrimg(:));

%ovrimg2 = double(imread(ovrimg2));
%ovrimg2 = ovrimg2./max(ovrimg2(:));

adata = ones(size(ovrimg)).*trans;
f = find(ovrimg < .01*max(ovrimg(:)));
adata(f) = 0;
a1data = ones(size(ovrimg2)).*trans1;
f = find(ovrimg2 < .05*max(ovrimg2(:)));
a1data(f) = 0;

image(bgimg);axis image;axis off;
hold on;
ovrc = ovrimg2;
ovrimg = ind2rgb(ovrimg,jet(256));
ovrimg2 = ind2rgb(ovrimg2,hot(256));
imh = image(ovrimg);set(imh,'AlphaData',adata);
%imh1 = image(ovrimg2);set(imh1,'AlphaData',a1data);

imh1c = contour(ovrc,5,'linewidth',1.5);colormap(hot);
%imh1c = contourf(ovrc,[0:25:255]);colormap(hot);
