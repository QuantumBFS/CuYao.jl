function cudiv(x::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_x, ceil(Int, x/threads_x)
end

function cudiv(x::Int, y::Int)
    max_threads = 256
    threads_x = min(max_threads, x)
    threads_y = min(max_threads ÷ threads_x, y)
    threads = (threads_x, threads_y)
    blocks = ceil.(Int, (x, y) ./ threads)
    threads, blocks
end

piecewise(state::AbstractVector, inds) = state
piecewise(state::AbstractMatrix, inds) = @inbounds view(state,:,inds[2])

function un_kernel(ctrl, U::AbstractMatrix, locs_raw)
    @inline function kernel(state, inds)
        x = inds[1]-1
        ctrl(x) && unrows!(piecewise(state, inds), x + locs_raw, U)
    end
end

function u1_kernel(ctrl, step::Int, a, b, c, d)
    @inline function kernel(state, inds)
        i = inds[1]
        if ctrl(i-1)
            u1rows!(piecewise(state, inds), i, i+step, a, b, c, d)
        end
    end
end

function u1pm_kernel(ctrl, step::Int, c, b)
    @inline function kernel(state, inds)
        x = inds[1]-1
        ctrl(x) && swaprows!(piecewise(state, inds), x+1, x+step+1, c, b)
    end
end

function u1diag_kernel(mask, a, d)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= testany(i-1, mask) ? d : a
    end
end

################ Specific kernels ##################
function x_kernel(mask, c)
    @inline function kernel(state, inds)
        i = inds[1]
        b = i-1
        c(b) && swaprows!(piecewise(state, inds), i, flip(b, mask) + 1)
    end
end

function y_kernel(mask, c, factor, bit_parity)
    @inline function kernel(state, inds)
        i = inds[1]
        b = i-1
        if c(b)
            i_ = flip(b, mask) + 1
            factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
            factor2 = factor1*bit_parity
            swaprows!(piecewise(state, inds), i, i_, factor2, factor1)
        end
    end
end

function z_kernel(mask)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= count_ones((i-1)&mask)%2==1 ? -1 : 1
    end
end

function zlike_kernel(mask, d)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= d ^ count_ones((i-1)&mask)
        #piecewise(state, inds)[i] *= CUDAnative.pow(d, count_ones((i-1)&mask))
    end
end

function cdg_kernel(ctrl, a, d)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= ctrl(i-1) ? d : a
    end
end

function cy_kernel(mask, c)
    @inline function kernel(state, inds)
        i = inds[1]
        b = i-1
        c(b) && swaprows!(piecewise(state, inds), i, flip(b, mask) + 1, im, -im)
    end
end

"""
build a simple kernel function from functions like f(::VecOrMat, inds).
"""
@generated function simple_kernel(func, state::AbstractArray{T, N}) where {T, N}
    if N == 1
        ex = :(ind = (blockIdx().x-1) * blockDim().x + threadIdx().x)
    else
        ex = :(ind = ((blockIdx().x-1) * blockDim().x + threadIdx().x,
        (blockIdx().y-1) * blockDim().y + threadIdx().y))
    end
    :($ex; func(state, ind); nothing)
end
