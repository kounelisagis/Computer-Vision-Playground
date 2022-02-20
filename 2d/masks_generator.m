photos = {'dog1.jpg','dog2.jpg','cat.jpg','bench.jpg','humanoid.jpeg'};

masks = cell(length(photos),1);

for i = length(photos):length(photos)
    figure
    imshow(imread(strcat('photos/', photos{i})));
    roi = images.roi.AssistedFreehand;
    draw(roi);
    masks{i} = createMask(roi, photo);
end

for i = length(photos):length(photos)
    imwrite(masks{i}, strcat('masks/', photos{i}));
end
