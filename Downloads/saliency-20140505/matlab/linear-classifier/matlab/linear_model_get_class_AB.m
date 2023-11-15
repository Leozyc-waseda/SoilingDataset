% Take in the subject data and match it against the surprise data

function [Class,class_idx,Tframed,tframed_idx,S_ABType] = linear_model_get_class_AB(S_TFRAME_1, S_TFRAME_2, S_VALUE, S_OFFSET, S_CONDITION, ABType, ...
                                                                                    TFRAME_1, TFRAME_2, S_SAMPLE, SAMPLE)

% NOTE:
% ABType 3 = stim_AB_Anims-Trans
% AbType 4 = stim_AB_Trans-Anims


Class    = zeros(size(SAMPLE,1),size(SAMPLE,2));
Tframed  = zeros(size(SAMPLE,1),size(SAMPLE,2));
S_ABType = zeros(size(SAMPLE,1),size(SAMPLE,2));

for i = 1:size(S_TFRAME_1,1)
    if(S_TFRAME_1(i,1) ~= 0 && S_TFRAME_2(i,1) ~= 0)
        if(strcmp('Anims-Trans',S_CONDITION(i,1)))
            S_ABType(i,1) = 3;
        elseif(strcmp('Trans-Anims',S_CONDITION(i,1)))
            S_ABType(i,1) = 4;
        else
            error('Unknown AB type parsed from file, got \"%s\"',S_CONDITION(i,1));
        end
        class_idx(  S_TFRAME_1(i,1),S_TFRAME_2(i,1),S_SAMPLE(i,1),S_ABType(i,1)) = S_VALUE(i,1);
        tframed_idx(S_TFRAME_1(i,1),S_TFRAME_2(i,1),S_SAMPLE(i,1),S_ABType(i,1)) = S_OFFSET(i,1);
    end
end

for i = 1:size(SAMPLE,1)
    if(TFRAME_1(i,1) ~= 0 && TFRAME_2(i,1) ~= 0)
        Class(i,1)   = class_idx(  TFRAME_1(i,1),TFRAME_2(i,1),SAMPLE(i,1),ABType(i,1));
        Tframed(i,1) = tframed_idx(TFRAME_1(i,1),TFRAME_2(i,1),SAMPLE(i,1),ABType(i,1));
    end
end
