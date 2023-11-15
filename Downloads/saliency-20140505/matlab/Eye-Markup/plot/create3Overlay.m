function create3Overlay(bgimg,ovrimg,ovrimg2,trans,trans1)
%creates an overlay of an image and a map or two maps.
%The image must be RGB.  If two maps the first map's colormap will be set
%to jet.  

adata = ones(size(ovrimg)).*trans;
f = find(ovrimg < 10^(-1) );
adata(f) = 0;
a1data = ones(size(ovrimg2)).*trans1;
f = find(ovrimg2 < 10^(-1) );
a1data(f) = 0;

image(bgimg);axis image;axis off;
hold on;
ovrimg = ind2rgb(ovrimg,jet(256));
imh = image(ovrimg);set(imh,'AlphaData',adata);
imh1 = imagesc(ovrimg2);colormap(hot);set(imh1,'AlphaData',a1data);
contour(ovrimg2,6);