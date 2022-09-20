apple_img = im2double(imread('photos/apple.jpg'));
orange_img = im2double(imread('photos/orange.jpg'));

[M, N, ~] = size(apple_img);
level = 5;

apple_lap = genPyr(apple_img, 'lap', level);
orange_lap = genPyr(orange_img, 'lap', level);

m1 = [ones(M, N/2) zeros(M, N/2)];
m1_pyramid = genPyr(m1, 'gauss', level);

B = cell(1,level);
for p = 1:level
	[Mp, Np, ~] = size(apple_lap{p});
	m1_resized = imresize(m1, [Mp Np]);
	m2_resized = imresize(1-m1, [Mp Np]);
	B{p} = apple_lap{p}.*m1_resized + orange_lap{p}.*m2_resized;
end

B_reconstructed = pyrReconstruct(B);
montage(B_reconstructed,'size',[1 NaN]);
