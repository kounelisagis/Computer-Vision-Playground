p200 = im2double(imread('photos/P200.jpg'));
[M, N, ~] = size(p200);

bench = im2double(imread('photos/bench.jpg'));
bench_mask = im2double(imread('masks/bench.jpg'));

cat = im2double(imread('photos/cat.jpg'));
cat_mask = im2double(imread('masks/cat.jpg'));

dog1 = im2double(imread('photos/dog1.jpg'));
dog1_mask = im2double(imread('masks/dog1.jpg'));

dog2 = im2double(imread('photos/dog2.jpg'));
dog2_mask = im2double(imread('masks/dog2.jpg'));

humanoid = im2double(imread('photos/humanoid.jpeg'));
humanoid_mask = im2double(imread('masks/humanoid.jpeg'));

bench = imresize(bench, [M, N]);
cat = imresize(cat, [M, N]);
dog1 = imresize(dog1,[M, N]);
dog2 = imresize(dog2, [M, N]);
humanoid = imresize(humanoid, [M, N]);

bench_mask = imresize(bench_mask, [M, N]);
cat_mask = imresize(cat_mask, [M, N]);
dog1_mask = imresize(dog1_mask, [M, N]);
dog2_mask = imresize(dog2_mask, [M, N]);
humanoid_mask = imresize(humanoid_mask, [M, N]);

level = 5;

p200_pyramid = genPyr(p200,'lap',level);
bench_pyramid = genPyr(bench,'lap',level);
cat_pyramid = genPyr(cat,'lap',level);
dog1_pyramid = genPyr(dog1,'lap',level);
dog2_pyramid = genPyr(dog2,'lap',level);
humanoid_pyramid = genPyr(humanoid,'lap',level);


% bench
gaussian1 = blend_images(p200_pyramid, bench_pyramid, 1-bench_mask, bench_mask, level);
laplacian1 = genPyr(gaussian1{1},'lap',level);

% cat
gaussian2 = blend_images(laplacian1, cat_pyramid, 1-cat_mask, cat_mask, level);
laplacian2 = genPyr(gaussian2{1},'lap',level);

% dog1
gaussian3 = blend_images(laplacian2, dog1_pyramid, 1-dog1_mask, dog1_mask, level);
laplacian3 = genPyr(gaussian3{1},'lap',level);

% dog2
gaussian4 = blend_images(laplacian3, dog2_pyramid, 1-dog2_mask, dog2_mask, level);
laplacian4 = genPyr(gaussian4{1},'lap',level);

% humanoid
gaussian5 = blend_images(laplacian4, humanoid_pyramid, 1-humanoid_mask, humanoid_mask, level);
laplacian5 = genPyr(gaussian5{1},'lap',level);

imgo = pyrReconstruct(laplacian5);
figure,imshow(imgo{1})


function [B_reconstructed] = blend_images(g_pyr1, g_pyr2, m1, m2, level)
    B = cell(1,level);
    for i = 1:level
	    [Mp, Np, ~] = size(g_pyr1{i});
	    m1 = imresize(m1,[Mp Np]);
	    m2 = imresize(m2,[Mp Np]);
	    B{i} = g_pyr1{i}.*m1 + g_pyr2{i}.*m2;
    end
    
    B_reconstructed = pyrReconstruct(B);
end
