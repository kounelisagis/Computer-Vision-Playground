N = 256;
x = atan(-5*pi:10*pi/(N-1):5*pi)';


pyramid_height = 5;

g_pyramid = create_gaussian_pyramid(x, pyramid_height);
l_pyramid = gaussian_to_laplacian(g_pyramid, pyramid_height);
g_pyramid_hat = laplacian_to_gaussian(l_pyramid, pyramid_height);


sum_immse = 0;
for i = 1:pyramid_height
    sum_immse = sum_immse + immse(g_pyramid_hat{i}, g_pyramid{i});
end
