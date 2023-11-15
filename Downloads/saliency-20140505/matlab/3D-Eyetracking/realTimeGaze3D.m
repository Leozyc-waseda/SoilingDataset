function realTimeGaze3D(in,le,re,mvname)
%% realTimeGaze3D(in,le,re,mvname)
%%this function takes in gaze data in 3D space along location of left and
%% right eye and generates a gif movie to visualize the eye movement during
%% the task.
%% prameters: in , a matrix with four columns (time, x , y z) 
%%            le , the position of left eye (vector)
%%            re , the position of right eye
%%            mvname , the base name for mvname
%% an alternative commented out line in the code exists that can be used
%% for generating mpeg movie


fdirout= [getenv('HOME'),'/tmp'];
fbaseout='mvtmp';

%upvec = [cos(90*(pi/180)),cos(60*(pi/180)),cos(130*(pi/180))];
%set(gca,'CameraUpVector',upvec)
m=max(in);



[r ,c] = size(in);




for ii=1:5:r
    
    plot3([0 0 50 50 0 ],[0 90 90 0 0 ],[0 0 0 0 0],'b' )
    grid on;
    box on;
    hold on
    %here we draw eyes by drawing two spheres
    [sx,sy,sz]=sphere;
    surf(sx + le(1), sy +le(2) , sz + le(3));
    surf(sx + re(1), sy +re(2) , sz + re(3));
    set(gca,'CameraViewAngleMode','manual');
    set(gca,'CameraPosition',[-50 150 250]);
    
    set(gca,'CameraUpVector',[-1 0 0 ]);
    axis([0 50 0 90 -50  (le(3)+re(3))/2 0 1]);
    if in(ii,4) >= 0
        plot3([in(ii,2)],[in(ii,3)],[in(ii,4)],'og');
    else
        plot3([in(ii,2)],[in(ii,3)],[in(ii,4)],'or');
    end
    
    plot3([le(1),in(ii,2),re(1)],[le(2),in(ii,3),re(2)],[le(3),in(ii,4),re(3)],'Color',[0.2 0.2 0.2]);
    text(in(ii,2)+1,in(ii,3)+1,in(ii,4),int2str(round((le(3)+re(3))/2 - in(ii,4))));
    
    %here we draw the screen
    sx=[0 50 ; 0 50];sy=[0 0 ; 90 90];sz=zeros(2,2);
    surf(sx,sy,sz);
    alpha(0.3);
    daspect([1 1 1]);
    camproj('perspective');

    % now let's wrtie the image into a file for later use, here we use png
    % format
    filename = [fdirout, '/', fbaseout, int2str(100000+ii), '.png'];
    print('-dpng',filename);
    hold off
end

mycommand = ['convert -quality 100 -delay 5 ',fdirout, '/',fbaseout,'*.png ',mvname,'.gif' ]
unix(mycommand);

%this line can be included for making mpeg movies
%mycommand = ['convert -quality 100 -delay 5 ',fdirout, '/',fbaseout,'*.png ',mvname,'.mpg' ]; unix(mycommand);

mycommand = ['rm -f ',fdirout,'/',fbaseout,'*.png']
unix(mycommand);
