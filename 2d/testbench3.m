photo_names = {'photos/P200.jpg', 'photos/bench.jpg', 'photos/cat.jpg', 'photos/dog1.jpg', 'photos/dog2.jpg', 'photos/tiago.jpg'};

% read photos
photos = cell(1, length(photo_names));
for k=1:length(photo_names)
    photos{k} = im2double(imread(photo_names{k}));
end


% resize all
[M, N, ~] = size(photos{1});
for k=2:length(photos)
    photos{k} = imresize(photos{k}, [M N]);
end

% draw masks - don't draw for the background
masks = cell(1, length(photo_names));
for k=2:length(photo_names)
    x = imread(photo_names{k});
    imshow(x);
    free_hand = drawassisted();
    mask = free_hand.createMask();
    masks{k} = mask;
end
close

level = 5;

% create pyramids
photos_pyramids = cell(1, length(photos));
for k=1:length(photos)
    photos_pyramids{k} = genPyr(photos{k}, 'lap', level);
end

masks_pyramids = cell(1, length(masks));
for k=2:length(masks)
    masks_pyramids{k} = genPyr(masks{k}, 'gauss', level);
    % gauss and lap options give different sizes of pyramids - fix
    for p=1:level
        [M, N, ~] = size(photos_pyramids{k}{p});
        masks_pyramids{k}{p} = imresize(masks_pyramids{k}{p}, [M N]);
    end
end


B = cell(1,level);
for p = 1:level
    B{p} = zeros(size(photos_pyramids{1}{p}));
    available_mask = ones(size(masks_pyramids{2}{p}));
    for k=length(photos):-1:2
        mask_temp = and(masks_pyramids{k}{p}, available_mask);
        available_mask = available_mask - mask_temp;
	    B{p} = B{p} + photos_pyramids{k}{p}.*mask_temp;
    end
    % use first pic as background
    B{p} = B{p} + photos_pyramids{1}{p}.*available_mask;
end

imgo = pyrReconstruct(B);
imgo_p = genPyr(imgo{1},'lap',level);
figure,imshow(imgo{1})
