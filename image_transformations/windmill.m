num_frames = 90;

background = imread('windmill_back.jpeg');
imageA = imread('windmill.png');
maskA = imread('windmill_mask.png', 'BackgroundColor', [1 1 1]); % BackgroundColor fixes some 'border' pixels
% 4D tensor to store images
images = uint8(zeros(size(background,1),size(background,2),3,num_frames));
interpolation_method = 'nearest'; % 'cubic' 'linear'

imageA = imageA-maskA;

video = VideoWriter('transf_windmill_nearest.avi');
open(video);

for i = 1:num_frames
    % Scaling - no scale applied for this problem
    sx = 1;
    sy = 1;
    s_tform=[  sx  0 0
               0  sy 0 
               0   0 1 ];
    sf = affine2d(s_tform);
    [imageB, RB] = imwarp(imageA, sf, 'Interp', interpolation_method);
    maskB = imwarp(maskA, sf, 'Interp', interpolation_method);

    % Rotation - 0-2pi -> one loop
    theta = (i/num_frames) * 2*pi;
    r_tform=[  cos(theta)  sin(theta) 0
              -sin(theta)  cos(theta) 0
                   0           0      1 ];
    af = affine2d(r_tform);
    [imageC, RC] = imwarp(imageB, RB, af, 'Interp', interpolation_method);
    maskC = imwarp(maskB, RB, af, 'FillValues', 255, 'Interp', interpolation_method); % white background

    % Translation
    tx = size(background,1)/2 - size(imageC,1)/2;
    ty = size(background,2)/2 - size(imageC,2)/2;
    t_tform=[  1    0  0
               0    1  0 
               tx  ty  1 ];
    tf = affine2d(t_tform);
    [imageD, RD] = imwarp(imageC, tf, 'Interp', interpolation_method);
    maskD = imwarp(maskC, tf, 'Interp', interpolation_method);

    padX = ceil(RD.XWorldLimits(1));
    padY = ceil(RD.YWorldLimits(1));
    I = padarray(imageD,[padX padY 0], 0, 'pre');
    maskI = padarray(maskD,[padX padY 0], 255, 'pre'); 

    padX = size(background,1) - size(I,1);  
    padY = size(background,2) - size(I,2);
    I = padarray(I,[ padX padY 0], 0, 'post');
    maskI = padarray(maskI,[padX padY 0], 255, 'post');

    maskI = maskI/255; % convert to binary
    I = background.*maskI + I.*(1-maskI);

    images(:,:,:,i) = I;
    writeVideo(video, I); %write the image to file
end

close(video);
implay(images);
