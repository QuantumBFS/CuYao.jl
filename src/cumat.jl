function YaoBlocks._apply_fallback!(r::GPUReg{B,T}, b::AbstractBlock) where {B,T}
    YaoBlocks._check_size(r, b)
    csp = mat(T, b) |> load_mat_tocuda
    state = YaoArrayRegister.matvec(r.state)
    mul!(state, csp, copy(state))
    return r
end

function cumat(::Type{T},c::CachedBlock) where T
    if !iscached(c.server, c.content)
        m = mat(T,c.content)
        m = load_mat_tocuda(m)
        push!(c.server, m, c.content)
        return m
    end
    return pull(c)
end

function YaoBlocks.apply!(r::GPUReg{B,T}, c::CachedBlock) where {B,T}
    csp = cumat(T,c)
    state = YaoBlocks.matvec(r.state)
    mul!(state, csp, copy(state))
    r
end
