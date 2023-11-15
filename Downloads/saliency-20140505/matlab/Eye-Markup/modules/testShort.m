function toofast = testShort(evs, args)
toofast = ([evs.type]==1 & [evs.pvel] - 15 * [evs.amp]>=100);
if ~any(toofast)
    return;
end

figure(2);
set(gcf,'Position', [6 550 556 421]); % should be the same figure all the time
clf(gcf);

flds = {'amp','pvel','interval'};
plotMainSeq3(evs, args,flds);
hold on;
plot3([evs(toofast).(flds{1})],[evs(toofast).(flds{2})],[evs(toofast).(flds{3})],'g*','MarkerSize',10);
hold off;
zlabel('time (ms)'); ylabel('pvel (deg/s)'); xlabel('amp (deg)');
axis([0 25 0 400 0 400]);
%{
subplot(2,1,1);
flds = {'amp','pvel'};
plotMainSeq(evs, args,flds);
hold on;
plot([evs(toofast).(flds{1})],[evs(toofast).(flds{2})],'g+','MarkerSize',10);
hold off;
xlabel('amplitude (degrees)'); ylabel('pvel (deg/s)');

subplot(2,1,2);
flds = {'amp', 'interval'};
plotMainSeq(evs([evs.type]~=0), args, flds);
hold on;
plot([evs(toofast).(flds{1})],[evs(toofast).(flds{2})],'g+','MarkerSize',10);
hold off;
xlabel('amplitude (degrees)'); ylabel('interval (ms)');

subplot(2,2,3);
flds = {'amp','avel'};
plotMainSeq(evs, args, flds);
hold on;
plot([evs(toofast).(flds{1})],[evs(toofast).(flds{2})],'g+','MarkerSize',10);
hold off;
xlabel('amplitude (degrees)'); ylabel('avel (deg/s)');

subplot(2,2,4);
flds = {'time_to_prev_saccade','time_to_next_saccade'};
plotMainSeq(evs, args, flds);
hold on;
plot([evs(toofast).(flds{1})],[evs(toofast).(flds{2})],'g+','MarkerSize',10);
hold off;
xlabel('time\_prev (ms)'); ylabel('time\_next (ms)');
%}
end