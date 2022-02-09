function g_pyramid = create_gaussian_pyramid(x, pyramid_height)
    g_pyramid = cell(pyramid_height,1);
    g_pyramid{1} = x;
    
    for i = 2:pyramid_height
        g_pyramid{i} = find_next_gaussian_pyramid_layer(g_pyramid{i-1});
    end
end


function x_hat = find_next_gaussian_pyramid_layer(x)
    
    N = length(x);
    T = toeplitz((1/16)*e(N, 1), h(N)');
    y = T*x;

    D = NaN(N/2, N/2);
    for r = 1:N/2
        D(r,1:N/2) = e(N/2, r)';
    end
    
    K = kron(D,e(2, 1)');
    x_hat = K*y;
end


function res = e(N, i)
    assert(1 <= i && i <= N);
    res = [zeros(i-1, 1); 1; zeros(N-i, 1)];
end


function res = h(N)
    res = [(1/16)*[1 4 6 4 1]'; zeros(N-5, 1)];
end
