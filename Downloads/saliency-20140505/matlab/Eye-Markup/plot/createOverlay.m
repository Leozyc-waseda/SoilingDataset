function createOverlay(bgimg,ovrimg,trans,colmap)
%creates an overlay of an image and a map or two maps.
%The image must be RGB.  If two maps the first map's colormap will be set
%to jet.  

adata = ones(size(ovrimg)).*trans;
f = find(ovrimg < 10^(-1) );
adata(f) = 0;
if (ndims(bgimg) <3)
bgimg = ind2rgb(bgimg,jet(256));
end
image(bgimg);axis image;axis off;
hold on;
imh = imagesc(ovrimg);colormap(colmap);
set(imh,'AlphaData',adata);
