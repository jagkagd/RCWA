classdef PVG < Layer
properties
    pitch, period, delta, chi, K
    no, ne, eo, ee
end
methods
    function obj = PVG(d, pitch, delta, chi, no, ne)
        obj = obj@Layer(d);
        obj.pitch = pitch; % LC rotate 2pi (m)
        obj.period = pitch ./ 2; % period of grating, LC rotate pi (m)
        obj.delta = deg2rad(delta); % angle (rad) between z axis
        obj.chi = chi; % chirality
        obj.K = 2.*pi./(obj.period);
        obj.Kx = obj.K.*sin(obj.delta);
        obj.Kz = obj.K.*cos(obj.delta);
        % obj.px = 2.*pi/obj.Kx
        % obj.pz = 2.*pi/obj.Kz
        obj.no = no;
        obj.ne = ne;
        obj.eo = no.^2;
        obj.ee = ne.^2;
        obj.Kv = [obj.Kx, 0, obj.Kz];
    end

    function res = eps(obj, p, q, n)
        [eo, ee, delta, chi] = deal(obj.eo, obj.ee, obj.delta, obj.chi);

        exx = @(Kr) eo + ((ee - eo).*cos(delta).^2.*cos((chi.*Kr)./2.).^2.*(1 - cos((chi.*Kr)./2.).^2.*sin(delta).^2))./(cos(delta).^2.*cos((chi.*Kr)./2.).^2 + sin((chi.*Kr)./2.).^2);
        exy = @(Kr) ((ee - eo).*cos(delta).*cos((chi.*Kr)./2.).*(1 - cos((chi.*Kr)./2.).^2.*sin(delta).^2).*sin((chi.*Kr)./2.))./(cos(delta).^2.*cos((chi.*Kr)./2.).^2 + sin((chi.*Kr)./2.).^2);
        exz = @(Kr) -((ee - eo).*cos(delta).*cos((chi.*Kr)./2.).^2.*sin(delta).*sqrt(1 - cos((chi.*Kr)./2.).^2.*sin(delta).^2))./sqrt(cos(delta).^2.*cos((chi.*Kr)./2.).^2 + sin((chi.*Kr)./2.).^2);
        eyy = @(Kr) eo + ((ee - eo).*(1 - cos((chi.*Kr)./2.).^2.*sin(delta).^2).*sin((chi.*Kr)./2.).^2)./(cos(delta).^2.*cos((chi.*Kr)./2.).^2 + sin((chi.*Kr)./2.).^2);
        eyz = @(Kr) -((ee - eo).*cos((chi.*Kr)./2.).*sin(delta).*sqrt(1 - cos((chi.*Kr)./2.).^2.*sin(delta).^2).*sin((chi.*Kr)./2.))./sqrt(cos(delta).^2.*cos((chi.*Kr)./2.).^2 + sin((chi.*Kr)./2.).^2);
        ezz = @(Kr) eo + (ee - eo).*cos((chi.*Kr)./2.).^2.*sin(delta).^2;
        es = {{exx, exy, exz}, {exy, eyy, eyz}, {exz, eyz, ezz}};

        ejkn = @(Kr) 1./(2.*pi) .* exp(-1.j.*n.*Kr);

        res = integral(@(Kr) es{p}{q}(Kr).*ejkn(Kr), -pi, pi);
    end
end
end
