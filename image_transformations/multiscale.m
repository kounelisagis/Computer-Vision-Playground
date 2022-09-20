ball = im2double(imread('ball.jpg'));
repeats = 5;
start = 1;

for k=1:repeats
    scale = 2^(1-k);
    tf = [scale  0   0
           0  scale 0
           0    0   1];
    tf = affine2d(tf);
    A = imwarp(ball,tf);

    % translate
    tf=[  1   0   0
          0   1   0 
          0 start 1 ];
    tf = affine2d(tf);
    [B,RB] = imwarp(A, tf);

    x_start = ceil(RB.XWorldLimits(1));
    x_end = ceil(RB.XWorldLimits(2)-1);
    y_start = ceil(RB.YWorldLimits(1));
    y_end = ceil(RB.YWorldLimits(2)-1);

    final_img(x_start:x_end, y_start:y_end,:) = B;

    start = start + size(B,2);
end

imwrite(final_img, 'multiscale.png')
imshow(final_img)
