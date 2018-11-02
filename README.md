# CuYao.jl

Yao.jl with CUDA native!

Under progress! To start
```julia console
]add Yao#Design4GPU

using Yao, CuYao
cureg = rand_state(16) |> cu
cureg |> put(16, 2=>Z)
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
- [ ] better T, S gate # replys on merge of Boosted T, S gates.
- [x] better control gates
- [x] BP diff blocks

Supported Operations
- [x] measure!, measure_reset!, measure_remove!, select
- [x] addbit!
- [x] expect
- [x] focus!, relax!
