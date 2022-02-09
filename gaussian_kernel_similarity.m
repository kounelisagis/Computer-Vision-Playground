[H,W] = freqz((1/16)*[1 4 6 4 1]', 1, 1000);

hold on

plot(W,abs(H))

mean = 0; sigma = 1.33; sz = 5;
x = linspace(-sz / 2, sz / 2, sz);
gaussFilter = (1/sqrt(2*pi*sigma))*exp(-(x-mean).^ 2 / (2 * sigma ^ 2));
gaussFilter = gaussFilter / sum (gaussFilter); % normalize -> sum = 1

[H,W] = freqz(gaussFilter, 1, 1000);
plot(W,abs(H))

legend({'h','Gaussian Kernel'},'Location','southwest')

hold off
