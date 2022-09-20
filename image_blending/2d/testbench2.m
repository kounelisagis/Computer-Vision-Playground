woman_img = im2double(imread('photos/woman.png'));
hand_img = im2double(imread('photos/hand.png'));
hand_img = imresize(hand_img, [size(woman_img,1) size(woman_img, 2)]);

[M, N, ~] = size(woman_img);
level = 5;

woman_lap = genPyr(woman_img, 'lap', level);
hand_lap = genPyr(hand_img, 'lap', level);

m1 = zeros(M, N);
m1(90:120,70:140,:) = 1;

B = cell(1,level);
for p = 1:level
	[Mp, Np, ~] = size(woman_lap{p});
	m1_resized = imresize(m1, [Mp Np]);
	m2_resized = imresize(1-m1, [Mp Np]);
	B{p} = woman_lap{p}.*m1_resized + hand_lap{p}.*m2_resized;
end

B_reconstructed = pyrReconstruct(B);
montage(B_reconstructed,'size',[1 NaN]);
