function dat = plotstats(datfile, epsfile)
% dat = plotstats()

% $HeadURL: svn://isvn.usc.edu/software/invt/trunk/saliency/devscripts/profiling/plotstats.m $
% $Id: plotstats.m 6938 2006-08-04 23:05:48Z rjpeters $

    dat = load(datfile);

    mrows=3;
    ncols=3;

    clf;

    pos = get(gcf, 'Position');
    set(gcf, 'Position', [pos(1:2) 1200 1000]);
    set(gcf, 'DefaultAxesFontSize', 7);
    set(gcf, 'DefaultTextInterpreter', 'none');
    set(gcf, 'PaperPositionMode', 'auto');

    mid = size(dat,2) / 2;

    subplotrc(mrows,ncols,1,1);

    titlespecs = { 'FontSize', 9, 'FontWeight', 'bold' };

    syms = { '-o', '-*', '-v', '-s', '-d', '-^' };

    memlegend = {
        sprintf('[t0] total memory O(n^%.1f)', logslope(dat(:,3), dat(:,1))),
        sprintf('[t0] image memory O(n^%.1f)', logslope(dat(:,3), dat(:,2))),
        sprintf('[t15] total memory O(n^%.1f)', logslope(dat(:,3), dat(:,mid+1))),
        sprintf('[t15] image memory O(n^%.1f)', logslope(dat(:,3), dat(:,mid+2)))
                };

    plot(dat(:,3), dat(:,1), syms{1}, ...
         dat(:,3), dat(:,2), syms{2}, ...
         dat(:,3), dat(:,mid+1), syms{3}, ...
         dat(:,3), dat(:,mid+2), syms{4});
    xlabel('image size (kPixels)');
    ylabel('memory usage (MB)');
    legend(memlegend, 2);
    h=title('memory usage as a function of image size');
    set(h, titlespecs{:});

    subplotrc(mrows,ncols,1,2);

    loglog(dat(:,3), dat(:,1), syms{1}, ...
           dat(:,3), dat(:,2), syms{2}, ...
           dat(:,3), dat(:,mid+1), syms{3}, ...
           dat(:,3), dat(:,mid+2), syms{4});
    xlabel('image size (kPixels)');
    ylabel('memory usage (MB)');
    legend(memlegend, 2);
    axis tight;
    h=title('memory usage as a function of image size (log/log scale)');
    set(h, titlespecs{:});

    subplotrc(mrows,ncols,1,3);

    semilogx(dat(:,3), 1024 .* dat(:,2) ./ dat(:,3), syms{1}, ...
             dat(:,3), 1024 .* dat(:,mid+2) ./ dat(:,3), syms{2});
    xlabel('image size (kPixels)');
    ylabel('memory usage per image size (bytes/pixel)');
    ax=axis;
    ax(4) = ax(4)*1.1;
    axis(ax);
    set(gca, 'XLim', [min(dat(:,3)) max(dat(:,3))]);
    legend(memlegend(1:2), 2);
    h=title({'memory usage per pixel,' 'as a function of image size (semilogx)'});
    set(h, titlespecs{:});

    cpulegend = {
        sprintf('[t0] CPU O(n^%.1f)', logslope(dat(:,3), dat(:,7))),
        sprintf('[t0] real O(n^%.1f)', logslope(dat(:,3), dat(:,8))),
        sprintf('[t15] CPU O(n^%.1f)', logslope(dat(:,3), dat(:,mid+7))),
        sprintf('[t15] real O(n^%.1f)', logslope(dat(:,3), dat(:,mid+8)))
                };

    subplotrc(mrows,ncols,2,1);

    plot(dat(:,3), dat(:,7) / 50, syms{1}, ...
         dat(:,3), dat(:,8) / 50, syms{2}, ...
         dat(:,3), dat(:,mid+7) / 50, syms{3}, ...
         dat(:,3), dat(:,mid+8) / 50, syms{4});
    xlabel('image size (kPixels)');
    ylabel('time (s/frame)');
    legend(cpulegend, 2);
    h=title('CPU usage as a function of image size');
    set(h, titlespecs{:});


    subplotrc(mrows,ncols,2,2);

    loglog(dat(:,3), dat(:,7) / 50, syms{1}, ...
           dat(:,3), dat(:,8) / 50, syms{2}, ...
           dat(:,3), dat(:,mid+7) / 50, syms{3}, ...
           dat(:,3), dat(:,mid+8) / 50, syms{4});
    xlabel('image size (kPixels)');
    ylabel('time (s/frame)');
    legend(cpulegend, 2);
    axis tight;
    h=title('CPU usage as a function of image size (log/log scale)');
    set(h, titlespecs{:});


    subplotrc(mrows,ncols,2,3);

    semilogx(dat(:,3), dat(:,7) ./ (dat(:,3)*1024) / 50, syms{1}, ...
             dat(:,3), dat(:,8) ./ (dat(:,3)*1024) / 50, syms{2}, ...
             dat(:,3), dat(:,mid+7) ./ (dat(:,3)*1024) / 50, syms{3}, ...
             dat(:,3), dat(:,mid+8) ./ (dat(:,3)*1024) / 50, syms{4});
    xlabel('image size (kPixels)');
    ylabel('time (s/pixel/frame)');
    set(gca, 'XLim', [min(dat(:,3)) max(dat(:,3))]);
    legend(cpulegend, 2);
    h=title({'CPU usage per pixel per frame,' 'as a function of image size (semilogx)'});
    set(h, titlespecs{:});

    proflegend = {
        sprintf('maxnorm O(n^%.1f)', logslope(dat(:,3), dat(:,10))),
        sprintf('oriented pyramid O(n^%.1f)', logslope(dat(:,3), dat(:,11))),
        sprintf('low-pass filters O(n^%.1f)', logslope(dat(:,3), dat(:,12))),
        sprintf('evolve O(n^%.1f)', logslope(dat(:,3), dat(:,13))),
        sprintf('motion pyramid O(n^%.1f)', logslope(dat(:,3), dat(:,14))),
        sprintf('everything else O(n^%.1f)', logslope(dat(:,3), dat(:,14)))
                 };

    subplotrc(mrows,ncols,3,1);
    plot(dat(:,3), dat(:,10) / 50, syms{1}, ...
         dat(:,3), dat(:,11) / 50, syms{2}, ...
         dat(:,3), dat(:,12) / 50, syms{3}, ...
         dat(:,3), dat(:,13) / 50, syms{4}, ...
         dat(:,3), dat(:,14) / 50, syms{5}, ...
         dat(:,3), dat(:,15) / 50, syms{6});
    xlabel('image size (kPixels)');
    ylabel('time (s/frame)');
    h=legend(proflegend, 2);
    h=title({'CPU usage in peak functions,' 'as a function of image size'});
    set(h, titlespecs{:});

    subplotrc(mrows,ncols,3,2);
    loglog(dat(:,3), dat(:,10) / 50, syms{1}, ...
           dat(:,3), dat(:,11) / 50, syms{2}, ...
           dat(:,3), dat(:,12) / 50, syms{3}, ...
           dat(:,3), dat(:,13) / 50, syms{4}, ...
           dat(:,3), dat(:,14) / 50, syms{5}, ...
           dat(:,3), dat(:,15) / 50, syms{6});
    xlabel('image size (kPixels)');
    ylabel('time (s/frame)');
    h=legend(proflegend, 2);
    axis tight;
    h=title({'CPU usage in peak functions,' 'as a function of image size (log/log scale)'});
    set(h, titlespecs{:});

    subplotrc(mrows,ncols,3,3);
    semilogx(dat(:,3), dat(:,10) ./ (dat(:,3)*1024) / 50, syms{1}, ...
             dat(:,3), dat(:,11) ./ (dat(:,3)*1024) / 50, syms{2}, ...
             dat(:,3), dat(:,12) ./ (dat(:,3)*1024) / 50, syms{3}, ...
             dat(:,3), dat(:,13) ./ (dat(:,3)*1024) / 50, syms{4}, ...
             dat(:,3), dat(:,14) ./ (dat(:,3)*1024) / 50, syms{5}, ...
             dat(:,3), dat(:,15) ./ (dat(:,3)*1024) / 50, syms{6});
    xlabel('image size (kPixels)');
    ylabel('time (s/pixel/frame)');
    set(gca, 'XLim', [min(dat(:,3)) max(dat(:,3))]);
    h=legend(proflegend, 2);
    h=title({'CPU usage per pixel in peak functions,' 'as a function of image size (semilogx)'});
    set(h, titlespecs{:});

    print('-depsc2', epsfile);


function m = logslope(x, y)
    logx = log10(x);
    logy = log10(y);
    diffy = logy(1) - logy(6);
    diffx = logx(1) - logx(6);
    m1 = diffy / diffx;

    m = lsqnonneg(logx(1:6) - logx(1), logy(1:6) - logy(1));
