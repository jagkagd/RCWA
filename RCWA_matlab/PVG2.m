classdef PVG2 < Layer
properties
    chi, no, ne, eo, ee
end
methods
    function obj = PVG2(d, Kx, Kz, chi, no, ne)
        obj = obj@Layer(d);
        obj.chi = chi; % chirality
        obj.no = no;
        obj.ne = ne;
        obj.eo = no.^2;
        obj.ee = ne.^2;
        obj.Kx = Kx;
        obj.Kz = Kz;
        obj.Kv = [obj.Kx, 0, obj.Kz];
    end

    function res = eps(obj, p, q, n)
        [eo, ee, chi] = deal(obj.eo, obj.ee, obj.chi);
        iif = @(varargin) varargin{2*find([varargin{1:2:end}], 1, 'first')}();
        exx = @(n) iif(abs(n) == 1, (ee-eo)./4, n == 0, (ee+eo)./2, true, 0);
        exy = @(n) iif(n == 1, -chi.*1j.*(ee-eo)./4, n == -1, chi.*1j.*(ee-eo)./4, true, 0);
        exz = @(n) 0;
        eyy = @(n) iif(abs(n) == 1, (eo-ee)./4, n == 0, (ee+eo)./2, true, 0);
        eyz = @(n) 0;
        ezz = @(n) iif(n == 0, eo, true, 0);
        eps = {{exx, exy, exz}, {exy, eyy, eyz}, {exz, eyz, ezz}};
        res = eps{p}{q}(n);
    end
end
end
