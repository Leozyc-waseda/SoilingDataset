% Take in the subject data and match it against the surprise data

function [Class,class_idx] = linear_model_get_class(S_TFRAME, S_SAMPLE, S_VALUE, TFRAME, SAMPLE)

Class = zeros(size(SAMPLE,1),size(SAMPLE,2));

for i = 1:size(S_TFRAME,1)
    class_idx(S_TFRAME(i,1),S_SAMPLE(i,1)) = S_VALUE(i,1);
end

for i = 1:size(SAMPLE,1)
    Class(i,1) = class_idx(TFRAME(i,1),SAMPLE(i,1));
end
