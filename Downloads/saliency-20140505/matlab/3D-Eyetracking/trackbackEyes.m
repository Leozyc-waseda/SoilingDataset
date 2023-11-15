function [el,er] =trackbackEyes(pl,pr,R1,a,b)
%%this function track backs the recorded projections of LEDs together with
%%the informtion about the LED plate and finds location of both eyes
%%note that it is assumed that LED plate is parallel to calibration plate
%%so by knowing the location of R1 and size of plate we can find out the
%%location of other LEDs
%%parmeters : pl projection of left eye on the calibration plate (2x2 matrix)
%% pr projection of right eye on the calibration plate
%% R1 location of LED#1 
%% a distance between LED #1 and LED #2
%% b distance between LED #1 and LED #4
%% el location of left eye
%% er location of right eye

el = zeros(1,3);
er = zeros(1,3);
R = [R1(1) R1(1)+a R1(1)+a R1(1) ; R1(2) R1(2) R1(2)-b R1(2)-b ; R1(3) R1(3) R1(3) R1(3)];

for ii = 1:4
    for jj=ii+1:4
        el  = el+ findVergence(pl(:,ii)',pl(:,jj)',R(:,ii)',R(:,jj)');
        er  = er+ findVergence(pr(:,ii)',pr(:,jj)',R(:,ii)',R(:,jj)');
    end
    
end
el = el/6;
er = er/6;
