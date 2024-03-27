%{
This MATLAB script aims to visualize the output of a neural network trying
to solve the Einstein Field Equations.
%}

%change this to your personal export_fig download
addpath("C:\Users\wowne\Downloads\export_fig\");

clear;
load("../network_output/tetradnet_test.mat");
%load("../network_output/tetradnet_finetuned_test.mat");

%% Look at the loss history
figure(1);
clf;

semilogy( 1:numel(loss), loss, "linewidth", 3, "color", "black" );
xlabel("epoch", 'interpreter', 'latex');
ylabel("$\mathcal{L}_{\textrm{Einstein}}$", 'interpreter', 'latex');
xticks([ 64 ,128,64*3])
xline([64, 128], 'linestyle', '--', 'color', [1,1,1]/2, "linewidth", 3 );
xlim([1, 3*64]);
set(gcf, "color", "w");
set(gca, "fontsize", 12);

export_fig('figures/loss_history.pdf', '-pdf', '-nocrop', gcf);

%% Histogram of Ricci
clf
num_bins = 64;
plot_histogram_no_edges(ricci, num_bins)
set(gcf, "color", "w");
export_fig('figures/ricci_histogram.pdf', '-dpdf', '-nocrop', gcf);


%% Histogram of Riemann
clf
num_bins = 64;
riemann2 = riemann(:);
riemann2( abs(riemann2) < 1e-5) = []; %delete values that are zero by symmetry
plot_histogram_no_edges(riemann2, num_bins)
set(gcf, "color", "w");
export_fig('figures/riemann_histogram.pdf', '-dpdf', '-nocrop', gcf);


%%
nexttile

semilogy(1:N, cond_e);
title('cond_e');


histogram(e)
title('e');

nexttile

ricci_scale = max(max(max(abs(ricci))));
nbins = 64;

histogram(ricci, nbins, "BinEdges", linspace(-ricci_scale, ricci_scale, nbins) )
title('ricci');

nexttile
riemann( abs(riemann) < 1e-4) = nan; %get rid of pointless exact zeros
histogram(riemann, nbins, "BinEdges", 10*linspace(-ricci_scale, ricci_scale, nbins) )
title('riemann');

squeeze( w1(1,:,:,1)./w2(1,:,:,1) )
squeeze(ricci(1,:,:))
return

%%
[x,e] = find_linear_transformation(x,e);

%% visualize fields with color

figure(2)
clf

N = size(e, 1);
g = zeros(N, 4, 4);
for i = 1:N
  e2 = squeeze( e(i,:,:) );
  g(i,:,:) = e2*diag([-1,1,1,1])*e2';
end

tl = tiledlayout(4,4);

for i = 1:4
for j = 1:4
nexttile
ms = 30;
scatter3( x(:,2), x(:,3), x(:,4), 30, e(:,i,j), 'filled' );
pbaspect([1 1 1]);
%colorbar();
%clim([-1 1]);
xlabel("x^1", "rotation", 0);
ylabel("x^2", "rotation", 0);
zlabel("x^3", "rotation", 0);
xticks([-1 1]);
yticks(xticks);
zticks(xticks);
%title( ("g_{" + (i-1)) + (j-1) + "}" );
clim([-2 2]);
colorbar()
colormap bluewhitered
title(tl, "metric components $g_{\mu\nu}$", "interpreter", "latex", "fontsize", 30);
set(gcf, 'color', 'w');
drawnow
%saveas(gcf, ""+i+""+j+".png");
end
end

set( tl, "Padding", "compact" );
set( tl, "TileSpacing", "compact" );



function [x,e] = find_linear_transformation(x,e)
  %{
  PURPOSE:
  General Relativity is Diffeomorphism invariant. Coordinates are
  arbitrary. This script aims to 
  %}
N = size(e, 1);
g = zeros(N, 4, 4);
for i = 1:N
  e2 = squeeze( e(i,:,:) );
  g(i,:,:) = e2*diag([-1,1,1,1])*e2';
end

g0 = squeeze(mean(g));
[V,D] = eigs(g0);

[~,I] = sort(diag(D));

D2 = zeros(4,4);
V2 = zeros(4,4);
V = V*sqrt(abs(D));
for i = 1:4
  D2(i,i) = sign(D(I(i), I(i)));
  V2(:,i) = V(:,I(i));
end
D2;
V = V*sqrt(abs(D));

g0 - V*D*V';
g0 - V2*D2*V2';

%Transform x and e with a linear transformation
x = x*V2;
N = size(e,1);
for i = 1:N
  e(i,:,:) = V2\squeeze(e(i,:,:));
end
end