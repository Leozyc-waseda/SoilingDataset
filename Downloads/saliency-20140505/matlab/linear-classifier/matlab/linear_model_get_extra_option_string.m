function [RSVP_extra_args,Options] = linear_model_get_extra_option_string(X)

Options = linear_model_get_optim_set();

e_arg = ['\t\"'];

for i=1:Options.numOn
    e_arg = [e_arg '--' Options.switch{Options.idx(1,i)} '=' num2str(X(1,i)) ' '];
end

RSVP_extra_args = [e_arg '\". \n'];
        