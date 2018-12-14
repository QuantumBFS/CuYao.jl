@inline function un_kernel(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    U = (all(diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm)) |> staticize
    MM = size(U0, 1)
    locked_bits = (cbits..., locs...)
    locked_vals = (cvals..., zeros(Int, M)...)
    locs_raw = [i+1 for i in itercontrol(nbit, setdiff(1:nbit, locs), zeros(Int, nbit-M))] |> staticize
    ctrl = controller(locked_bits, locked_vals)

    @inline function kernel(state, inds)
        x = inds[1]-1
        ctrl(x) && unrows!(piecewise(state, inds), x+locs_raw, U)
    end
end

@inline function u1_kernel(U1::SDMatrix, ibit::Int)
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ctrl = controller(ibit, 0)

    @inline function kernel(state, inds)
        i = inds[1]
        if ctrl(i-1)
            u1rows!(piecewise(state, inds), i, i+step, a, b, c, d)
        end
    end
end
u1_kernel(U1::SDSparseMatrixCSC, ibit::Int) = u1_kernel(U1|>Matrix, ibit)

@inline function u1_kernel(U1::SDPermMatrix, ibit::Int)
    U1.perm[1] == 1 && return u1_kernel(Diagonal(U1.vals), ibit)

    mask = bmask(ibit)
    b, c = U1.vals[1], U1.vals[2]
    step = 1<<(ibit-1)
    ctrl = controller(ibit, 0)

    @inline function kernel(state, inds)
        x = inds[1]-1
        ctrl(x) && swaprows!(piecewise(state, inds), x+1, x+step+1, c, b)
    end
end
u1_kernel(U1::IMatrix, ibit::Int) = (state, inds) -> nothing

@inline function u1_kernel(U1::SDDiagonal, ibit::Int)
    a, d = U1.diag
    u1diag_kernel(ibit, a, d)
end

@inline function u1diag_kernel(ibit::Int, a, d)
    mask = bmask(ibit)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= testany(i-1, mask) ? d : a
    end
end

################ Specific kernels ##################
x_kernel(bits::Ints) = cx_kernel((), (), bits::Ints)
@inline function cx_kernel(cbits, cvals, bits::Ints)
    ctrl = controller((cbits..., bits[1]), (cvals..., 0))
    mask = bmask(bits...)
    #function kernel(state, inds)
    @inline function kernel(state, inds)
        i = inds[1]
        b = i-1
        ctrl(b) && swaprows!(piecewise(state, inds), i, flip(b, mask) + 1)
        return
    end
end

@inline function y_kernel(bits::Ints)
    mask = bmask(Int, bits...)
    ctrl = controller(bits[1], 0)
    bit_parity = length(bits)%2 == 0 ? 1 : -1
    factor = (-im)^length(bits)
    @inline function kernel(state, inds)
        i = inds[1]
        b = i-1
        if ctrl(b)
            i_ = flip(b, mask) + 1
            factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
            factor2 = factor1*bit_parity
            swaprows!(piecewise(state, inds), i, i_, factor2, factor1)
        end
    end
end

@inline function z_kernel(bits::Ints)
    mask = bmask(bits...)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= count_ones((i-1)&mask)%2==1 ? -1 : 1
        return
    end
end

@inline function zlike_kernel(bits::Ints, d)
    mask = bmask(bits...)
    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= d ^ count_ones((i-1)&mask)
        #piecewise(state, inds)[i] *= CUDAnative.pow(d, count_ones((i-1)&mask))
    end
end

@inline function cdg_kernel(cbits, cvals, ibit, a, d)
    ctrl = controller((cbits..., ibit), (cvals..., 1))

    @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= ctrl(i-1) ? d : a
    end
end

@inline function cy_kernel(cbits, cvals, bit::Int)
    ctrl = controller((cbits..., bit), (cvals..., 0))
    mask = bmask(bit)
    @inline function kernel(state, inds)
        i = inds[1]
        b = i-1
        ctrl(b) && swaprows!(piecewise(state, inds), i, flip(b, mask) + 1, im, -im)
    end
end
for (G, FACTOR) in zip([:z, :s, :t, :sdag, :tdag], [:(-1), :(im), :($(exp(im*π/4))), :(-im), :($(exp(-im*π/4)))])
    KERNEL = Symbol(G, :_kernel)
    CKERNEL = Symbol(:c, KERNEL)
    if G != :z
        @eval $KERNEL(bits::Ints) = zlike_kernel(bits, $FACTOR)
    end
    @eval $KERNEL(bit::Int) = u1diag_kernel(bit, 1, $FACTOR)
    @eval $CKERNEL(cbits, cvals, ibit::Int) = cdg_kernel(cbits, cvals, ibit, 1, $FACTOR)
end

"""
build a simple kernel function from functions like f(::VecOrMat, inds).
"""
@generated function simple_kernel(func, state::AbstractArray{T, N}) where {T, N}
    if N == 1
        ex = :(inds = (blockIdx().x-1) * blockDim().x + threadIdx().x)
    else
        ex = :(inds = ((blockIdx().x-1) * blockDim().x + threadIdx().x,
                       (blockIdx().y-1) * blockDim().y + threadIdx().y); inds[2]<=size(state, 2) || return nothing)
    end
    ex = :($ex; func(state, inds); nothing)
    ex
end
