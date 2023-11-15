function NEW_CLASS = linear_model_get_new_class(CLASS,conf)

NEW_CLASS = zeros(size(CLASS,1),1);

for i=1:size(CLASS,1) 
    if     CLASS(i,1) < conf.hardBound
        NEW_CLASS(i,1) = 1;
    elseif CLASS(i,1) < conf.easyBound
        NEW_CLASS(i,1) = 2;
    else
        NEW_CLASS(i,1) = 3;
    end
end