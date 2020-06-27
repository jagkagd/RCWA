function res = eulerMatrix(angs, i, j)
    innerF = {
        {@(a, b, c) cos(a).*cos(b).*cos(c)-sin(a).*sin(c), @(a, b, c) -cos(c).*sin(a)-cos(a).*cos(b).*sin(c), @(a, b, c) cos(a).*sin(b)},
        {@(a, b, c) cos(b).*cos(c).*sin(a)+cos(a).*sin(c), @(a, b, c)  cos(a).*cos(c)-cos(b).*sin(a).*sin(c), @(a, b, c) sin(a).*sin(b)},
        {@(a, b, c) -cos(c).*sin(b),                       @(a, b, c)  sin(b).*sin(c),                        @(a, b, c) cos(b)}
    };
    eulerMatrixElem = innerF{i}{j};
    res = eulerMatrixElem(angs{:});
end
