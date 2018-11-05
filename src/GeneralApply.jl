###################### unapply! ############################
function _unapply!(state::CuVector, U::AbstractMatrix, locs_raw::SDVector, ic::IterControl)
    function kernel(state, U, locs_raw)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds unrows!(state, ic[i] + locs_raw, U)
        return
    end
    X, Y = cudiv(length(ic))
    @cuda threads=X blocks=Y kernel(state, U, locs_raw)
    state
end

function _unapply!(state::CuMatrix, U::AbstractMatrix, locs_raw::SDVector, ic::IterControl)
    function kernel(state, U, locs_raw)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        @inbounds unrows!(view(state,:,j), ic[i] + locs_raw, U)
        return
    end
    X, Y = cudiv(length(ic), size(state, 2))
    @cuda threads=X blocks=Y kernel(state, U, locs_raw)
    state
end

#=
nbit = 6
N = 1<<nbit
LOC1 = SVector{2}([0, 1])
vn = randn(ComplexF32, N, 3)
unapply!(vn |> cu, mat(H), (3,)) |> Matrix ≈ unapply!(vn |> copy, mat(H), (3,))
=#

for MT in [:CuMatrix, :CuVector]
@eval function _unapply!(state::$MT, U::SDSparseMatrixCSC, locs_raw::SDVector, ic::IterControl)
    _unapply!(state, SMatrix{size(U, 1), size(U,2)}(U), locs_raw, ic)
end
end
#=
    wsz = (i==0 ? size(U, 1) : size(state, i) for i in ndims(state))
    function kernel(state, U, locs_raw)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        work = similar(state, wsz...)
        locs = locs_raw+ic[i]
        unrows!(state, locs, U, work)
        return
    end
    X, Y = cudiv(length(ic))
    @cuda threads=X blocks=Y kernel(state, U, locs_raw)
    state
end
=#

################## General U1 apply! ###################
# diagonal
diag_widget_1(mask, a, d) = (v, b)->testany(b, mask) ? d*v : a*v
function u1apply!(state::CuVecOrMat, U1::SDDiagonal, ibit::Int)
    mask = bmask(ibit)
    a, d = U1.diag
    state .= diag_widget_1(mask, a, d).(state, basis(state))
end

u1apply!(state::CuVecOrMat, U1::SDSparseMatrixCSC, ibit::Int) = u1apply!(state, U1|>Matrix, ibit)
function u1apply!(state::CuVector, U1::SDMatrix, ibit::Int)
    # reorder a unirary matrix.
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, a, b, c, d)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds x = ic[i]+1
        u1rows!(state, x, x+step, a, b, c, d)
        return
    end
    X, Y = cudiv(length(ic))
    @cuda threads=X blocks=Y kernel(state, ic, step, a, b, c, d)
    state
end

function u1apply!(state::CuMatrix, U1::SDMatrix, ibit::Int)
    # reorder a unirary matrix.
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, step, a, b, c, d)
        ic = itercontrol(nactive(state), [ibit], [0])
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        @inbounds x = ic[i]+1
        @inbounds u1rows!(view(state, :, j), x, x+step, a, b, c, d)
        return
    end
    X, Y = cudiv(length(ic), size(state, 2))
    @cuda threads=X blocks=Y kernel(state, step, a, b, c, d)
    state
end

function u1apply!(state::CuVector{T}, U1::SDPermMatrix, ibit::Int) where T
    U1.perm[1] == 1 && return u1apply!(state, Diagonal(U1), ibit)

    mask = bmask(ibit)
    b, c = T(U1.vals[1]), T(U1.vals[2])
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, c, b)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        @inbounds j = ic[i] + 1
        swaprows!(state, j, j+step, c, b)
        return
    end
    X, Y = cudiv(length(ic))
    @cuda threads=X blocks=Y kernel(state, ic, step, c, b)
    state
end

function u1apply!(state::CuMatrix{T}, U1::SDPermMatrix, ibit::Int) where T
    U1.perm[1] == 1 && return u1apply!(state, Diagonal(U1), ibit)

    mask = bmask(ibit)
    b, c = T(U1.vals[1]), T(U1.vals[2])
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, c, b)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = (blockIdx().y-1) * blockDim().y + threadIdx().y
        @inbounds x = ic[i] + 1
        @inbounds swaprows!(view(state, :,j), x, x+step, c, b)
        return
    end
    X, Y = cudiv(length(ic), size(state, 2))
    @cuda threads=X blocks=Y kernel(state, ic, step, c, b)
    state
end

u1apply!(state::CuVecOrMat, U1::IMatrix, ibit::Int) = state
