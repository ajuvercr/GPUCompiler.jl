# implementation of the GPUCompiler interfaces for generating BPF code

## target

export BPFCompilerTarget

struct BPFCompilerTarget <: AbstractCompilerTarget
end

llvm_triple(::BPFCompilerTarget) = Sys.MACHINE # TODO: "bpf-bpf-bpf"
llvm_datalayout(::BPFCompilerTarget) = "e-m:e-p:64:64-i64:64-n32:64-S128"

function llvm_machine(target::BPFCompilerTarget)
    triple = llvm_triple(target)
    t = Target(;triple=triple)

    cpu = ""
    feat = ""
    tm = TargetMachine(t, triple, cpu, feat)
    asm_verbosity!(tm, true)

    return tm
end


## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{BPFCompilerTarget}) = "bpf"

const bpf_intrinsics = () # TODO
isintrinsic(::CompilerJob{BPFCompilerTarget}, fn::String) = in(fn, bpf_intrinsics)

function prepare_execution!(job::CompilerJob{BPFCompilerTarget}, mod::LLVM.Module)
    triple!(mod, "bpf-bpf-bpf")
end
