export piecewise, cudiv
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

piecewise(stt::AbstractVector, inds) = stt
piecewise(stt::AbstractMatrix, inds) = @inbounds view(stt,:,inds[2])

function un_kernel(nbit::Int, cbits::NTuple{C, Int}, cvals::NTuple{C, Int}, U0::AbstractMatrix, locs::NTuple{M, Int}) where {C, M}
    U = (all(diff(locs).>0) ? U0 : reorder(U0, collect(locs)|>sortperm)) |> staticize
    MM = size(U0, 1)
    locked_bits = (cbits..., locs...)
    locked_vals = (cvals..., zeros(Int, M)...)
    locs_raw = [i+1 for i in itercontrol(nbit, setdiff(1:nbit, locs), zeros(Int, nbit-M))] |> staticize
    ctrl = controller(locked_bits, locked_vals)

    function kernel(stt, inds)
        x = inds[1]-1
        ctrl(x) && unrows!(piecewise(stt, inds), x+locs_raw, U)
    end
end

function u1_kernel(U1::SDMatrix, ibit::Int)
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ctrl = controller(ibit, 0)

    function kernel(stt, inds)
        i = inds[1]
        if ctrl(i-1)
            u1rows!(piecewise(stt, inds), i, i+step, a, b, c, d)
        end
    end
end
u1_kernel(U1::SDSparseMatrixCSC, ibit::Int) = u1_kernel(U1|>Matrix, ibit)

function u1_kernel(U1::SDPermMatrix, ibit::Int)
    U1.perm[1] == 1 && return u1_kernel(Diagonal(U1.vals), ibit)

    mask = bmask(ibit)
    b, c = U1.vals[1], U1.vals[2]
    step = 1<<(ibit-1)
    ctrl = controller(ibit, 0)

    function kernel(stt, inds)
        x = inds[1]-1
        ctrl(x) && swaprows!(piecewise(stt, inds), x+1, x+step+1, c, b)
    end
end
u1_kernel(U1::IMatrix, ibit::Int) = (stt, inds) -> nothing

function u1_kernel(U1::SDDiagonal, ibit::Int)
    a, d = U1.diag
    u1diag_kernel(ibit, a, d)
end

function u1diag_kernel(ibit::Int, a, d)
    mask = bmask(ibit)
    function kernel(stt, inds)
        i = inds[1]
        piecewise(stt, inds)[i] *= testany(i-1, mask) ? d : a
    end
end

################ Specific kernels ##################
x_kernel(bits::Ints) = cx_kernel((), (), bits::Ints)
function cx_kernel(cbits, cvals, bits::Ints)
    ctrl = controller((cbits..., bits[1]), (cvals..., 0))
    mask = bmask(bits...)
    #function kernel(stt, inds)
    function kernel(stt, inds)
        i = inds[1]
        b = i-1
        ctrl(b) && swaprows!(piecewise(stt, inds), i, flip(b, mask) + 1)
        return
    end
end

function y_kernel(bits::Ints)
    mask = bmask(Int, bits...)
    ctrl = controller(bits[1], 0)
    bit_parity = length(bits)%2 == 0 ? 1 : -1
    factor = (-im)^length(bits)
    function kernel(stt, inds)
        i = inds[1]
        b = i-1
        if ctrl(b)
            i_ = flip(b, mask) + 1
            factor1 = count_ones(b&mask)%2 == 1 ? -factor : factor
            factor2 = factor1*bit_parity
            swaprows!(piecewise(stt, inds), i, i_, factor2, factor1)
        end
    end
end

function z_kernel(bits::Ints)
    mask = bmask(bits...)
    function kernel(stt, inds)
        i = inds[1]
        piecewise(stt, inds)[i] *= count_ones((i-1)&mask)%2==1 ? -1 : 1
        return
    end
end

function zlike_kernel(bits::Ints, d)
    mask = bmask(bits...)
    function kernel(stt, inds)
        i = inds[1]
        piecewise(stt, inds)[i] *= d ^ count_ones((i-1)&mask)
        #piecewise(stt, inds)[i] *= CUDAnative.pow(d, count_ones((i-1)&mask))
    end
end

function cdg_kernel(cbits, cvals, ibit, a, d)
    ctrl = controller((cbits..., ibit), (cvals..., 1))

    function kernel(stt, inds)
        i = inds[1]
        piecewise(stt, inds)[i] *= ctrl(i-1) ? d : a
    end
end

function cy_kernel(cbits, cvals, bit::Int)
    ctrl = controller((cbits..., bit), (cvals..., 0))
    mask = bmask(bit)
    function kernel(stt, inds)
        i = inds[1]
        b = i-1
        ctrl(b) && swaprows!(piecewise(stt, inds), i, flip(b, mask) + 1, im, -im)
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
@generated function simple_kernel(func, stt::AbstractArray{T, N}) where {T, N}
    if N == 1
        ex = :(inds = (blockIdx().x-1) * blockDim().x + threadIdx().x)
    else
        ex = :(inds = ((blockIdx().x-1) * blockDim().x + threadIdx().x,
        (blockIdx().y-1) * blockDim().y + threadIdx().y))
    end
    ex = :($ex; func(stt, inds); nothing)
    ex
end
