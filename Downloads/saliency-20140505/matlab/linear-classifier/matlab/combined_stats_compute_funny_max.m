function fmax = combined_stats_compute_funny_max(hi_set, low_set, type, inv)

global DEBUG;

Min = min([min(min(hi_set)) min(min(low_set))]);
Max = max([max(max(hi_set)) max(max(low_set))]);

% normalize sets
n_hi_set  = (hi_set  - Min)  ./ (Max  - Min);
% normalize and INVERT the low set
if (strcmp(inv,'yes'))
    n_low_set = 1 - (low_set - Min) ./ (Max - Min);
else
    n_low_set = (low_set - Min) ./ (Max - Min);
end

if (strcmp(type,'max'))
    fmax = max(n_hi_set,n_low_set);
elseif (strcmp(type,'min'))
    fmax = min(n_hi_set,n_low_set);
else
    error('Unknown type given for funny max op');
end

DEBUG.n_low_set = n_low_set;
DEBUG.n_hi_set  = n_hi_set;
DEBUG.fmax      = fmax;