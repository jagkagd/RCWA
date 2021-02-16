classdef PVG_dp < PVG
properties
    pitch, period, delta, chi, K
    no, ne, eo, ee
end
methods
    function obj = PVG(d, pitch, delta, chi, no, ne)
        
        obj.pitch = pitch; % LC rotate 2pi (m)
        obj.period = pitch ./ 2; % period of grating, LC rotate pi (m)
        obj.delta = deg2rad(delta); % angle (rad) between z axis
        obj.K = 2.*pi./(obj.period);
        Kx = obj.K.*sin(obj.delta);
        Kz = obj.K.*cos(obj.delta);
        obj = obj@Layer(d, Kx, Kz, chi, no, ne);
    end
end
end
