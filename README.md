# CuYao.jl

Yao.jl with CUDA native!

Under progress! To start
```
add Yao#Design4GPU
julia test/gpuapplys.jl
julia test/GPUReg.jl
```

Supported Gate List
- [x] general U(N) gate
- [x] general U(1) gate
- [x] better X, Y, Z gate
- [ ] better T, S gate
- [x] better control gates

Supported Operations
- [x] measure!, measure_reset!, measure_remove!, select
- [x] addbit!
- [ ] expect
- [ ] focus!, relax!
