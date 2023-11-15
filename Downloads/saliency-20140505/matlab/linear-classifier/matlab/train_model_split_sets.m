function [TR_CLASS,TR_SET,TR_SET_MEMBER,TR_SWITCH] = train_model_split_sets(SAMPLE,TFRAME,CLASS,SETS)

dprint('Splitting sets into training/testing pairs');

% assign each sequence to one of two classes, but balance the set based on
% target class to keep them balanced
% resize data structures

% copy class, add one so that we avoid zero address
if(min(min(CLASS))) == 0
    TR_CLASS      = CLASS + 1;
else
    TR_CLASS      = CLASS;
end

% Create membership array
TR_SET_MEMBER = zeros(size(TR_CLASS,1),1) - 1;
% Membership array based on what sample / frame in? 
TR_SET        = zeros(max(max(SAMPLE)),max(max(TFRAME))) - 1;
% How many classes do we have, create that many classes
TR_SWITCH     = zeros(max(max(TR_CLASS)),1) - 1;

% create two sets
if SETS == 2
    ltswitch = 0;
    for i=1:size(TR_CLASS,1)
        % New image sequence set?
        if TR_SET(SAMPLE(i,1),TFRAME(i,1)) == -1
            % switch the switch itself, init to class on a rolling basis
            % Thus, this class has never been set. So we set it for the
            % first time here. 
            if TR_SWITCH(TR_CLASS(i,1)) == -1
                TR_SWITCH(TR_CLASS(i,1)) = ltswitch;
                if ltswitch == 0
                    ltswitch = 1;
                else
                    ltswitch = 0;
                end
            end
            % place in one of two data sets keeping class target balanced
            TR_SET(SAMPLE(i,1),TFRAME(i,1)) = TR_SWITCH(TR_CLASS(i,1));
            % switch to other class to keep balance
            % Once a class has had it's initial switch set, we then
            % can just set it on its own
            if TR_SWITCH(TR_CLASS(i,1)) == 1
                TR_SWITCH(TR_CLASS(i,1)) = 0;
            else
                TR_SWITCH(TR_CLASS(i,1)) = 1;
            end
        end
        TR_SET_MEMBER(i,:) = TR_SET(SAMPLE(i,1),TFRAME(i,1));
    end
elseif SETS > 2
    ltswitch = 0;

    for i=1:size(TR_CLASS,1)
        % New image sequence set?
        if TR_SET(SAMPLE(i,1),TFRAME(i,1)) == -1
            % switch the switch itself, init to class on a rolling basis
            % Thus, this class has never been set. So we set it for the
            % first time here. 
            if TR_SWITCH(TR_CLASS(i,1)) == -1
                TR_SWITCH(TR_CLASS(i,1)) = ltswitch;
                if ltswitch == (SETS - 1)
                    ltswitch = 0;
                else
                    ltswitch = ltswitch + 1;
                end
            end
            % place in one of two data sets keeping class target balanced
            TR_SET(SAMPLE(i,1),TFRAME(i,1)) = TR_SWITCH(TR_CLASS(i,1));
            % switch to other class to keep balance
            % Once a class has had it's initial switch set, we then
            % can just set it on its own
            if TR_SWITCH(TR_CLASS(i,1)) == (SETS - 1)
                TR_SWITCH(TR_CLASS(i,1)) = 0;
            else
                TR_SWITCH(TR_CLASS(i,1)) = TR_SWITCH(TR_CLASS(i,1)) + 1;
            end
        end
        TR_SET_MEMBER(i,:) = TR_SET(SAMPLE(i,1),TFRAME(i,1));
    end
else
    error('Number of sets must be equal to or greater than 2');
end