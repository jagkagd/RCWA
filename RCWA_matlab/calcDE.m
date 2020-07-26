ng = 1.58;
dn = 0.2;
no = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. - dn./2;
ne = (dn + 2.*sqrt(-2.*dn.^2 + 9.*ng.^2))./6. + dn./2;
% ng = sqrt((2.*no.^2+ne.^2)./3)
alpha = rad2deg(asin(1/ng*sin(deg2rad(20))))./2;
k0 = 2.*pi./940e-9;
Kx = -2*k0*ng*sin(deg2rad(alpha))*cos(deg2rad(alpha));
Kz = -2*k0*ng*sin(deg2rad(alpha))*sin(deg2rad(alpha));
%ds = 0:0.01:2;
%wls = (400:700)*1e-9;
wl = 940e-9;
thetas = -10:10;
nn = 1;
DERl = [];
DERr = [];
DETl = [];
DETr = [];
angs1 = [];
angs2 = [];
angs3 = [];

for i = 1:length(thetas)
    theta = rad2deg(asin(sin(deg2rad(thetas(i)))/ng));
    lay = PVG2(wl*0.5/dn, Kx, Kz, -1, no, ne);
    rcwa = RCWA(ng, ng, lay, nn, [1, 1j], [1, -1j]);
    [derl, derr, detl, detr, rcwa] = rcwa.solve(0, theta, wl, 1, 1j);
    DERl(i, :) = derl;
    DERr(i, :) = derr;
    DETl(i, :) = detl;
    DETr(i, :) = detr;
    angs1(i) = rad2deg(asin(sin(atan2(rcwa.k3iv(1, 1), rcwa.k3iv(1, 3)))*ng));
    angs2(i) = rad2deg(asin(sin(atan2(rcwa.k3iv(2, 1), rcwa.k3iv(2, 3)))*ng));
    angs3(i) = rad2deg(asin(sin(atan2(rcwa.k3iv(3, 1), rcwa.k3iv(3, 3)))*ng));

end

DERl = DERl.';
DERr = DERr.';
DETl = DETl.';
DETr = DETr.';

xxs = thetas;
nn = nn + 1;
ax1 = subplot(2, 1, 1);
figure(1);
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
% plot(ax2, xxs,  DETr(nn-1, :)./(DETr(nn-1, :) + DETl(nn, :)), 'k');
hold(ax2, 'off');
figure(2);
plot(xxs, angs1, xxs, angs2, xxs, angs3)
% plt.subplot(211);
% thetam, phim = meshgrid(thetas, phis);
% plt.pcolormesh(thetam./60.*cos(deg2rad(phim)), thetam./60.*sin(deg2rad(phim)), DERl[:, :, nn+1].T);
% % plt.plot(wls,  DERr[nn]+DERl[nn], 'y');
% plt.subplot(212);
% plt.pcolormesh(thetam./60.*cos(deg2rad(phim)), thetam./60.*sin(deg2rad(phim)), DERl[:, :, nn+1].T./(DERl[:, :, nn+1].T+DETl[:, :, nn].T));
% plt.show();