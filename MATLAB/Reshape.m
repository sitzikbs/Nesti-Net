function Reshape(ax, r, theta, phi, c0)
if nargin < 5 
    c0 = [0, 0, 0];
end

CamX=r*sind(theta)*sind(phi)  + c0(1); CamY=r*cosd(theta) + c0(2); CamZ=r*sind(theta)*cosd(phi) + c0(3);
UpX=-cosd(theta)*sind(phi);UpY=sind(theta);UpZ=-cosd(theta)*cosd(phi);
set(ax,'CameraPosition',[CamX,CamY,CamZ],'CameraTarget',c0,...
    'CameraUpVector',[UpX,UpY,UpZ]);
end