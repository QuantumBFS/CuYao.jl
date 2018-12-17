# CuYao.jl

Yao.jl with CUDA native!

Under progress! To start
```julia console
]add Yao#master

using Yao, CuYao
cureg = rand_state(9, 1000) |> cu
cureg |> put(9, 2=>Z)
measure!(cureg |> addbit(1) |> focus!(4,1,3)) |> relax!
```
to run tests
```bash
julia test/runtests.jl
```

Supported Gate List
- [x] general U(N) gate
- [x] general U(1) gate
- [x] better X, Y, Z gate
- [x] better T, S gate
- [x] better control gates
- [x] BP diff blocks

Supported Register Operations
- [x] measure!, measure_reset!, measure_remove!, select
- [x] addbit!
- [x] insert_qubit!
- [x] focus!, relax!
- [x] join
- [ ] density_matrix

- [x] statistic functional diff blocks
- [x] expect for statistic functional
