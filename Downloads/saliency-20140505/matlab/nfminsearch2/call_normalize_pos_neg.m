function X = call_normalize_pos_neg(Y)

Y(1:11,:)  = normalize_pos_neg(Y(1:11,:));
Y(12:22,:) = normalize_pos_neg(Y(12:22,:));
Y(23:33,:) = normalize_pos_neg(Y(23:33,:));
Y(34:44,:) = normalize_pos_neg(Y(34:44,:));

%if we have extra scaling factors normalize them too
if size(Y,1) > 45
    Y(45:48,:) = normalize_abs(Y(45:48,:));
end

X = Y;