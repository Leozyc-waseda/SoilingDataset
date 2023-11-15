function [bin1,bin2,center,rint,corr] = hist_norm(X1,X2,N)

% Find the bounds for the histogram
low_bound   = min([min(min(X1)) min(min(X2))]);
high_bound  = max([max(max(X1)) max(max(X2))]);

interval = abs(high_bound - low_bound) / (N - 1);

bin1 = zeros(1,N);
bin2 = zeros(1,N);

cur_bin = 1;

X1size = size(X1,1);
X2size = size(X2,1);

% Bin samples
for i = low_bound:interval:high_bound
    for j = 1:X1size
        if ((X1(j,1) >= i) && (X1(j,1) < i + interval))
            bin1(1,cur_bin) = bin1(1,cur_bin) + 1;
        end
    end
    for j = 1:X2size
        if ((X2(j,1) >= i) && (X2(j,1) < i + interval))
            bin2(1,cur_bin) = bin2(1,cur_bin) + 1;
        end
    end
    center(1,cur_bin) = i + interval/2;
    cur_bin = cur_bin + 1;
end
     
% normalize the "populations"

hist_norm = X1size/X2size;

bin2 = bin2 * hist_norm;

rint = sum(sum(min(bin1,bin2)));
corr = rint/X1size;