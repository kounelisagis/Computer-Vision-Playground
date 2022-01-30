function l_pyramid = gaussian_to_laplacian(g_pyramid, pyramid_height)

    l_pyramid = cell(pyramid_height, 1);

    for i = 1:pyramid_height-1
        g_curr = g_pyramid{i};
        g_next = g_pyramid{i+1};
        g_next_interpolated = interp(g_next, 2);
        l_pyramid{i} = g_curr-g_next_interpolated;
    end

    l_pyramid{end} = g_pyramid{end};
end

