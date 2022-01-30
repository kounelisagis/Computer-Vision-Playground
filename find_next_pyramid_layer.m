function x_hat = find_next_pyramid_layer(x)
    
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
    res = [zeros(i-1, 1); 1; zeros(N-i, 1)];
end


function res = h(N)
    res = [(1/16)*[1 4 6 4 1]'; zeros(N-5, 1)];
end
