num_frames = 60;
imageA = imread('ball.jpg', 'jpg');
maskA = imread('ball_mask.jpg', 'jpg');
background = imread('beach.jpg', 'jpg');
images = uint8(zeros(size(background,1),size(background,2),3,num_frames));
interpol = 'cubic'; % 'nearest' 'linear'

imageA = imageA-maskA;

video = VideoWriter('transf_beach.avi');
open(video);

for i = 1:num_frames
    % Scale the ball
    sx = 1/4;
    sy = 1/4;

    % Uncomment the following for the ball to travel towards the horizon
%     scf = 4+(i-1)*100/num_frames;
%     sx = 1/scf;
%     sy = 1/scf;

    sform=[  sx  0 0
             0  sy 0 
             0   0 1 ];
    sf = affine2d(sform);
    [imageB, RB] = imwarp(imageA, sf, 'Interp', interpol);
    maskB = imwarp(maskA, sf, 'Interp', interpol);

    % Translation of image at 0,0
    tx = - size(imageB,2)/2;
    ty = - size(imageB,1)/2;
    tform=[  1    0  0
             0    1  0 
             tx  ty  1 ];
    tf = affine2d(tform);
    [imageC, RC] = imwarp(imageB, RB, tf, 'FillValues', 0, 'Interp', interpol);
    maskC = imwarp(maskB, RB, tf, 'FillValues', 255, 'Interp', interpol);

    % Rotate
    theta = (i/num_frames)*2 * 2*pi;
    xform=[  cos(theta)  sin(theta) 0
            -sin(theta)  cos(theta) 0
                  0           0     1 ];
    af = affine2d(xform);
    [imageD, RD] = imwarp(imageC, RC, af, 'Interp', interpol);
    maskD = imwarp(maskC, RC, af, 'FillValues', 255, 'Interp', interpol); % white background

    % Translation of image
    tx = 400 + (i-1) * size(background,2)/num_frames;
    posCos = cos(4*tx/size(background,2)*2*pi);
    end_pos = size(background,1)-size(imageA,1)/4;
    ty = end_pos - abs(end_pos*posCos)/exp(tx/size(background,2)*4);

    % Uncomment the following for the ball to travel towards the horizon
%     tsform = projective2d([1  1 0.00275;
%                            0  1    0;
%                            0  0    1]);
%     [tx,ty] = transformPointsForward(tsform,tx,ty);

    ty = ceil(ty + RD.YWorldLimits(1));
    tx = ceil(tx + RD.XWorldLimits(1));

    tform=[  1    0  0
             0    1  0 
             tx  ty  1 ];
    tf = affine2d(tform);
    [imageF, RF] = imwarp(imageD, RD, tf, 'Interp', interpol);
    maskF = imwarp(maskD, RD, tf, 'FillValues', 255, 'Interp', interpol); % white background

    % after translation to 0,0 world coordinates become negative -> cropping needed
    Fsize = size(imageF);
    if (tx<0)
        imageF = imageF(:,-tx:Fsize(2),:);
        maskF = maskF(:,-tx:Fsize(2),:);
        tx=0;
    end
    if (ty<0)
        imageF = imageF(-ty:size(imageF,1),:,:);
        maskF = maskF(-ty:Fsize(1),:,:);
        ty=0;
    end
    
    % prepadding
    imageI = padarray(imageF,[ceil(ty) ceil(tx) 0], 0, 'pre');
    maskI = padarray(maskF,[ceil(ty) ceil(tx) 0], 255, 'pre');
    
    % cropping
    if size(imageI,1) > size(background,1)
        imageI = imageI(1:size(background,1),:,:);
        maskI = maskI(1:size(background,1),:,:);
        padX = 0;
    else
        padX = size(background,1)-size(imageI,1);
    end
    if size(imageI,2) > size(background,2)
        imageI = imageI(:,1:size(background,2),:);
        maskI = maskI(:,1:size(background,2),:);
        padY = 0;
    else
        padY = size(background,2)-size(imageI,2);
    end

    % postpadding        
    imageI = padarray(imageI,[padX padY 0], 0, 'post');
    maskI = padarray(maskI,[padX padY 0], 255, 'post');

    maskI = maskI/255; % make it binary
    imageI = background.*maskI + imageI.*(1-maskI);
    images(:,:,:,i) = imageI;
    writeVideo(video, imageI); %write the image to file
end

close(video);
implay(images);
