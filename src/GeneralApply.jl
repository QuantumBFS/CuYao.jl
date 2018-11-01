###################### unapply! ############################
function _unapply!(state::CuVecOrMat, U::AbstractMatrix, locs_raw::SDVector, ic::IterControl)
    function kernel(state, U, locs_raw)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        unrows!(state, ic[i] + locs_raw, U)
        return
    end
    @cuda blocks=length(ic) kernel(state, U, locs_raw)
    state
end

function _unapply!(state::CuVecOrMat, U::SDSparseMatrixCSC, locs_raw::SDVector, ic::IterControl)
    return _unapply!(state, SMatrix{size(U, 1), size(U,2)}(U), locs_raw, ic)
    wsz = (i==0 ? size(U, 1) : size(state, i) for i in ndims(state))
    function kernel(state, U, locs_raw)
        work = similar(state, wsz...)
        locs = locs_raw+ic[i]
        unrows!(state, locs, U, work)
        return
    end
    @cuda blocks=length(ic) kernel(state, U, locs_raw)
    state
end

################## General U1 apply! ###################
# diagonal
diag_widget_1(mask, a, d) = (v, b)->testany(b, mask) ? d*v : a*v
function u1apply!(state::CuVecOrMat, U1::SDDiagonal, ibit::Int)
    mask = bmask(ibit)
    a, d = U1.diag
    state .= diag_widget_1(mask, a, d).(state, basis(state))
end

function u1apply!(state::CuVecOrMat, U1::AbstractMatrix, ibit::Int)
    # reorder a unirary matrix.
    a, c, b, d = U1
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, a, b, c, d)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        x = ic[i]+1
        u1rows!(state, x, x+step, a, b, c, d)
        return
    end
    @cuda blocks=length(ic) kernel(state, ic, step, a, b, c, d)
    state
end

function u1apply!(state::CuVecOrMat{T}, U1::SDPermMatrix, ibit::Int) where T
    U1.perm[1] == 1 && return u1apply!(state, Diagonal(U1), ibit)

    mask = bmask(ibit)
    b, c = T(U1.vals[1]), T(U1.vals[2])
    step = 1<<(ibit-1)
    ic = itercontrol(nactive(state), [ibit], [0])
    function kernel(state, ic, step, c, b)
        i = (blockIdx().x-1) * blockDim().x + threadIdx().x
        j = ic[i] + 1
        swaprows!(state, j, j+step, c, b)
        return
    end
    @cuda blocks=length(ic) kernel(state, ic, step, c, b)
    state
end

u1apply!(state::CuVecOrMat, U1::IMatrix, ibit::Int) = state
