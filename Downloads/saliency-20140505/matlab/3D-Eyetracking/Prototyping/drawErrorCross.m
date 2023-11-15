function drawErrorCross(p,er)
%%this function draws an error 3D cross at a given p with er dimensions


plot3(p(1),p(2),p(3),'.g');
plot3([p(1)-er(1), p(1)+er(1)],[p(2) , p(2)],[p(3) , p(3)],'Color',[p(3)/160+0.5,0.4,0.6]);
plot3([p(1), p(1)],[p(2)-er(2) , p(2)+er(2)],[p(3) , p(3)],'Color',[p(3)/160+0.5,0.4,0.6]);
plot3([p(1), p(1)],[p(2) , p(2)],[p(3)-er(3) , p(3)+er(3)],'Color',[p(3)/160+0.5,0.4,0.6]);
