ng = 1.56;
dn = 0.15;
no = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. - dn./2;
ne = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. + dn./2;
% ng = sqrt((2.*no.^2+ne.^2)./3)
alpha = 60./2;
k0 = 2.*pi./532e-9;
pb = 2 .* 2.*pi./(2.*k0.*ng.*cos(deg2rad(alpha)));

wls = linspace(400, 700, 301).*1e-9;
% thetas = linspace(0, 60, 61);
% phis = linspace(0, 359, 360);
nn = 1;
DERl = [];
DERr = [];
DETl = [];
DETr = [];

lay1 = PVG(2e-6, pb, alpha, 1, no, ne);
lay2 = PVG(2e-6, pb, alpha*1.1, 1, no, ne);
for i = 1:length(wls)
    pb = 2 .* 2.*pi./(2.*k0.*ng.*cos(deg2rad(alpha)));
    
    rcwa = RCWA(ng, ng, {lay1, lay2}, nn, [1, 1j], [1, -1j]);
    [derl, derr, detl, detr] = rcwa.solve(0, 0, wls(i), 1, 1j);
    DERl(i, :) = derl;
    DERr(i, :) = derr;
    DETl(i, :) = detl;
    DETr(i, :) = detr;
end

DERl = DERl.';
DERr = DERr.';
DETl = DETl.';
DETr = DETr.';

xxs = wls;
nn = nn + 1;
ax1 = subplot(2, 1, 1);
hold(ax1, 'on');
plot(ax1, xxs, DERl(nn, :), 'r');
plot(ax1, xxs, DERl(nn-1, :), 'g');
plot(ax1, xxs, DERl(nn+1, :), 'b');
plot(ax1, xxs, DERr(nn, :), 'r--');
plot(ax1, xxs, DERr(nn-1, :), 'g--');
plot(ax1, xxs, DERr(nn+1, :), 'b--');
plot(ax1, xxs, DERl(nn+1, :)./(DERl(nn+1, :)+DETl(nn, :)), 'k');
hold(ax1, 'off');
% plt.plot(wls,  DERr[nn]+DERl[nn], 'y');
ax2 = subplot(2, 1, 2);
hold(ax2, 'on');
plot(ax2, xxs,  DETl(nn, :), 'r');
plot(ax2, xxs,  DETl(nn-1, :), 'g');
plot(ax2, xxs,  DETl(nn+1, :), 'b');
plot(ax2, xxs,  DETr(nn, :), 'r--');
plot(ax2, xxs,  DETr(nn-1, :), 'g--');
plot(ax2, xxs,  DETr(nn+1, :), 'b--');
hold(ax2, 'off');
% plt.subplot(211);
% thetam, phim = meshgrid(thetas, phis);
% plt.pcolormesh(thetam./60.*cos(deg2rad(phim)), thetam./60.*sin(deg2rad(phim)), DERl[:, :, nn+1].T);
% % plt.plot(wls,  DERr[nn]+DERl[nn], 'y');
% plt.subplot(212);
% plt.pcolormesh(thetam./60.*cos(deg2rad(phim)), thetam./60.*sin(deg2rad(phim)), DERl[:, :, nn+1].T./(DERl[:, :, nn+1].T+DETl[:, :, nn].T));
% plt.show();