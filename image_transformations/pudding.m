image = imread('pudding.png');

num_frames = 100;
speed = 3;
amplitude = 0.3;

video = VideoWriter('sheared_pudding.avi');
open(video);

for frame_num = 1:num_frames
    shy = 0;
    shx = amplitude * sin(speed * (frame_num / num_frames) * 2*pi);
    t_matrix=[ 1   shy 0
              shx   1  0 
              128  128 1 ];
    transform = affine2d(t_matrix);

    centerOutput = affineOutputView(size(image)*2, transform, 'BoundsStyle', 'CenterOutput');
    curr_img = imwarp(image, transform, 'OutputView', centerOutput);
    curr_frame = im2frame(curr_img);
    writeVideo(video, curr_frame);
end

close(video);
implay('sheared_pudding.avi');
