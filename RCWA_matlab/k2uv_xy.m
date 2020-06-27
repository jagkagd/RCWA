function res = k2uv_xy(k)
    kx = k(1);
    ky = k(2);
    kz = k(3);
    kk = sqrt(kx.^2 + ky.^2 + kz.^2);
    ktan = sqrt(kx.^2 + ky.^2);
    if abs(ktan) ~= 0.
        res = [
            kz/kk .* kx/ktan, -ky/ktan;
            kz/kk .* ky/ktan,  kx/ktan
        ];
    else
        res = [
            kz/kk, 0;
            0, 1
        ];
    end
end
