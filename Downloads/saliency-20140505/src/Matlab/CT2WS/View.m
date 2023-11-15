function View(wsOutput)
    figure;
    imagesc(wsOutput.signals.values)
    colormap(gray)
    axis equal;