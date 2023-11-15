function Options = linear_model_get_optim_set();

Options.meta_log = '/lab/mundhenk/linear-classifier/log/train.log';
Options.prec_in_log = '/lab/mundhenk/linear-classifier/log/in_train_prec.log';
Options.prec_out_log = '/lab/mundhenk/linear-classifier/log/out_train_prec.log';

Options.size  = 6;
% Turn on some options in training
Options.on    = [0 0 1 1 1 1];
Options.numOn = sum(sum(Options.on));

Options.desc{1}   = ['SingleChannelSurpriseUpdFac'];
Options.switch{1} = ['surprise-updfac'];
Options.ideal{1}  = 0.7;
Options.lowbnd{1} = 0.5;
Options.upbnd{1}  = 1.0;

Options.desc{2}   = ['SingleChannelSurpriseNeighUpdFac'];
Options.switch{2} = ['surprise-neighupdfac'];
Options.ideal{2}  = 0.7;
Options.lowbnd{2} = 0.5;
Options.upbnd{2} = 1.0;

Options.desc{3}   = ['SingleChannelSurpriseSLfac'];
Options.switch{3} = ['surprise-slfac'];
%Options.ideal{3}  = 0.1;
%Options.ideal{3}  = 0.0505697697525615622038408503158279927447438240051269531250000000;
%Options.ideal{3}  = -0.1711924565549051424628856921117403544485569000244140625000000000;
%Options.ideal{3}  = -0.1702652083259778570401010711066192016005516052246093750000000000;
Options.ideal{3}  = -0.1063055478951265842013640394725371152162551879882812500000000000;
Options.lowbnd{3} = 0.001;
Options.upbnd{3}  = 1.0;

Options.desc{4}   = ['SingleChannelSurpriseSSfac'];
Options.switch{4} = ['surprise-ssfac'];
%Options.ideal{4}  = 1.0;
%Options.ideal{4}  = 0.9985112578317147935536013392265886068344116210937500000000000000;
%Options.ideal{4}  = 0.1851035868840598119788865005830302834510803222656250000000000000;
%Options.ideal{4}  = 0.1892484850178392241648595017977640964090824127197265625000000000;
Options.ideal{4}  = 0.5723246142099625011212538083782419562339782714843750000000000000;
Options.lowbnd{4} = 0.001;
Options.upbnd{4}  = 1.0;

Options.desc{5}   = ['SingleChannelSurpriseNeighSigma'];
Options.switch{5} = ['surprise-neighsig'];
%Options.ideal{5}  = 0.5;
%Options.ideal{5}  =	0.5331846490086566969779369173920713365077972412109375000000000000;
%Options.ideal{5}  = 1.7521361387560441258415266929659992456436157226562500000000000000;
%Options.ideal{5}  = 1.7618858712427374335618424083804711699485778808593750000000000000;
Options.ideal{5}  = 0.5061156308856045171751247835345566272735595703125000000000000000;
Options.lowbnd{5} = 0.1;
Options.upbnd{5}  = 6.0;

Options.desc{6}   = ['SingleChannelSurpriseLocSigma'];
Options.switch{6} = ['surprise-locsig'];
%Options.ideal{6}  = 3.0;
%Options.ideal{6}  =	3.0092449875899673905621511948993429541587829589843750000000000000;
%Options.ideal{6}  = 1.6248059042198723656014180960482917726039886474609375000000000000;
%Options.ideal{6}  = 1.6494454112557210834211218752898275852203369140625000000000000000;
Options.ideal{6}  = 1.2008054804327541464914475000114180147647857666015625000000000000;
Options.lowbnd{6} = 0.1;
Options.upbnd{6}  = 6.0;

Options.ub    = zeros(1,Options.numOn);
Options.lb    = zeros(1,Options.numOn);
Options.start = zeros(1,Options.numOn);
Options.idx   = zeros(1,Options.numOn);

i = 1;
for j=1:Options.size
    if Options.on(1,j)
        Options.ub(1,i)    = Options.upbnd{j};
        Options.lb(1,i)    = Options.lowbnd{j};
        Options.start(1,i) = Options.ideal{j};
        Options.idx(1,i)   = j;
        i = i + 1;
    end
end