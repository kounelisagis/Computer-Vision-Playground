apple_img = im2double(imread('photos/apple.jpg'));
orange_img = im2double(imread('photos/orange.jpg'));

[M, N, ~] = size(apple_img);
level = 5;

apple_lap = genPyr(apple_img, 'lap', level);
orange_lap = genPyr(orange_img, 'lap', level);

m1 = [ones(M, N/2) zeros(M, N/2)];

B = cell(1,level);
for p = 1:level
	[Mp, Np, ~] = size(apple_lap{p});
	maskap = imresize(m1, [Mp Np]);
	maskbp = imresize(1-m1, [Mp Np]);
	B{p} = apple_lap{p}.*maskap + orange_lap{p}.*maskbp;
end

B_reconstructed = pyrReconstruct(B);
montage(B_reconstructed,'size',[1 NaN]);
