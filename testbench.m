N = 256;
x = atan(-5*pi:10*pi/(N-1):5*pi)';


pyramid_height = 5;
g_pyramid = cell(pyramid_height,1);
g_pyramid{1} = x;

for i = 2:pyramid_height
    x = find_next_pyramid_layer(x);
    g_pyramid{i} = x;
end

l_pyramid = gaussian_to_laplacian(g_pyramid, pyramid_height);
g_pyramid_hat = laplacian_to_gaussian(l_pyramid, pyramid_height);


for i = 1:pyramid_height
    immse(g_pyramid_hat{i}, g_pyramid{i})
end
