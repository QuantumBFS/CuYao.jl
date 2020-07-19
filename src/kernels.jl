@inline function un_kernel(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    U = (all(TupleTools.diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm)) |> staticize
    MM = size(U0, 1)
    locked_bits = [cbits..., locs...]
    locked_vals = [cvals..., zeros(Int, M)...]
    locs_raw = [i+1 for i in itercontrol(nbit, setdiff(1:nbit, locs), zeros(Int, nbit-M))] |> staticize
    configs = itercontrol(nbit, locked_bits, locked_vals)

    length(configs), @inline function kernel(state, inds)
        x = @inbounds configs[inds[1]]
        unrows!(piecewise(state, inds), x .+ locs_raw, U)
    end
end

@inline function u1_kernel(nbit::Int, U1::AbstractMatrix, ibit::Int)
    a, c, b, d = U1
    step = 1<<(ibit-1)
    configs = itercontrol(nbit, [ibit], [0])

    length(configs), @inline function kernel(state, inds)
        i = @inbounds configs[inds[1]]+1
        u1rows!(piecewise(state, inds), i, i+step, a, b, c, d)
    end
end
function u1_kernel(nbit::Int, U1::SDSparseMatrixCSC, ibit::Int)
    u1_kernel(nbit, U1|>Matrix, ibit)
end

@inline function u1_kernel(nbit::Int, U1::SDPermMatrix, ibit::Int)
    if U1.perm[1] == 1
        return u1_kernel(nbit, Diagonal(U1.vals), ibit)
    end

    mask = bmask(ibit)
    b, c = U1.vals[1], U1.vals[2]
    step = 1<<(ibit-1)
    configs = itercontrol(nbit, [ibit], [0])

    1<<(nbit-1), function kernel(state, inds)
        x = @inbounds configs[inds[1]] + 1
        swaprows!(piecewise(state, inds), x, x+step, c, b)
    end
end

@inline function u1_kernel(nbit::Int, U1::SDDiagonal, ibit::Int)
    a, d = U1.diag
    u1diag_kernel(nbit,ibit, a, d)
end

@inline function u1diag_kernel(nbit::Int, ibit::Int, a, d)
    mask = bmask(ibit)
    1<<nbit, @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= anyone(i-1, mask) ? d : a
    end
end

################ Specific kernels ##################
x_kernel(nbit::Int, bits::Ints) = cx_kernel(nbit::Int, (), (), bits::Ints)
cx_kernel(nbit::Int, cbits, cvals, loc::Int) = cx_kernel(nbit::Int, cbits, cvals, (loc,))
@inline function cx_kernel(nbit::Int, cbits, cvals, bits::Ints)
    configs = itercontrol(nbit, [cbits..., bits[1]], [cvals..., 0])
    mask = bmask(bits...)
    length(configs), @inline function kernel(state, inds)
        b = @inbounds configs[inds[1]]
        swaprows!(piecewise(state, inds), b+1, flip(b, mask) + 1)
        return
    end
end

@inline function pswap_kernel(nbit::Int, m::Int, n::Int, theta::Real)
    mask1 = bmask(m)
    mask2 = bmask(n)
    mask12 = mask1|mask2
    a, c, b_, d = mat(Rx(theta))
    e = exp(-im/2*theta)
    configs = itercontrol(nbit, [m,n], [0,0])
    length(configs), @inline function kernel(state, inds)
        @inbounds x = configs[inds[1]]
        piecewise(state, inds)[x+1] *= e
        piecewise(state, inds)[x⊻mask12+1] *= e
        y = x ⊻ mask2
        u1rows!(piecewise(state, inds), y+1, y⊻mask12+1, a, b_, c, d)
    end
end

@inline function y_kernel(nbit::Int, bits::Ints)
    mask = bmask(Int, bits...)
    configs = itercontrol(nbit, [bits[1]], [0])
    bit_parity = length(bits)%2 == 0 ? 1 : -1
    factor = (-im)^length(bits)
    length(configs), @inline function kernel(state, inds)
        b = @inbounds configs[inds[1]]
        i_ = flip(b, mask) + 1
        factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
        factor2 = factor1*bit_parity
        swaprows!(piecewise(state, inds), b+1, i_, factor2, factor1)
    end
end

@inline function z_kernel(nbit::Int,bits::Ints)
    mask = bmask(bits...)
    1<<nbit,@inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= count_ones((i-1)&mask)%2==1 ? -1 : 1
        return
    end
end

@inline function zlike_kernel(nbit::Int,bits::Ints, d::Union{ComplexF32, ComplexF64, Float64, Float32})
    mask = bmask(Int32, bits...)
    1<<nbit,@inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= CUDA.pow(d, bit_count(Int32(i-1)&mask))
        return
    end
end

@inline function cdg_kernel(nbit::Int, cbits, cvals, ibit, a, d)
    ctrl = controller((cbits..., ibit), (cvals..., 1))

    1<<nbit, @inline function kernel(state, inds)
        i = inds[1]
        piecewise(state, inds)[i] *= ctrl(i-1) ? d : a
    end
end

@inline function cy_kernel(nbit, cbits, cvals, bit::Int)
    configs = itercontrol(nbit, [cbits..., bit], [cvals..., 0])
    mask = bmask(bit)
    length(configs), @inline function kernel(state, inds)
        b = @inbounds configs[inds[1]]
        swaprows!(piecewise(state, inds), b+1, flip(b, mask) + 1, im, -im)
    end
end

for (G, FACTOR) in zip([:z, :s, :t, :sdag, :tdag], [:(-one(Int32)), :(1f0im), :($(exp(im*π/4))), :(-1f0im), :($(exp(-im*π/4)))])
    KERNEL = Symbol(G, :_kernel)
    CKERNEL = Symbol(:c, KERNEL)
    if G != :z
        @eval $KERNEL(nbit::Int, bits::Ints) = zlike_kernel(nbit, bits, $FACTOR)
    end
    @eval $KERNEL(nbit::Int, bit::Int) = u1diag_kernel(nbit::Int, bit, one($FACTOR), $FACTOR)
    @eval $CKERNEL(nbit::Int, cbits, cvals, ibit::Int) = cdg_kernel(nbit::Int, cbits, cvals, ibit, one($FACTOR), $FACTOR)
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
