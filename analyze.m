clear;

load("../solution.mat")

%stats
tiledlayout(2,2);

nexttile
histogram(x);
title('x');

nexttile
histogram(e)
title('e');

nexttile
histogram(ricci)
title('ricci');

nexttile
riemann( abs(riemann) < 1e-5) = nan; %get rid of pointless exact zeros
histogram(riemann)
title('riemann');

return;

%% visualize fields with color
scatter3( x(:,1), x(:,3), x(:,4), 100, e(:,2,3), 'filled' );
pbaspect([1 1 1]);
colorbar();
%clim([-1 1]);
colormap jet