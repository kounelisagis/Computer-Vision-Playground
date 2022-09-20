function g_pyramid = laplacian_to_gaussian(l_pyramid, pyramid_height)

    g_pyramid = cell(pyramid_height, 1);

    g_pyramid{pyramid_height} = l_pyramid{pyramid_height};

    for i = pyramid_height-1:-1:1
        l_curr = l_pyramid{i};
        g_prev = g_pyramid{i+1};
        g_prev_interpolated = interp(g_prev, 2);
        g_pyramid{i} = l_curr+g_prev_interpolated;
    end

end

