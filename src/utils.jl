function mswaprows!(state, x, y, j, f1, f2)
    @inbounds temp = state[x]
    @inbounds state[x, j] = state[y, j] * f2
    @inbounds state[y, j] = temp * f1
end

function mswaprows!(state, x, y, j)
    @inbounds temp = state[x]
    @inbounds state[x, j] = state[y, j]
    @inbounds state[y, j] = temp
end

function mu1rows!(state, x, y, j, a, b, c, d)
    @inbounds w = state[x, j]
    @inbounds v = state[y, j]
    @inbounds state[x, j] = a * w + b * v
    @inbounds state[y, j] = c * w + d * v
end
