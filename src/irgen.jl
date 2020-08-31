# LLVM IR generation

function irgen(@nospecialize(job::CompilerJob), method_instance::Core.MethodInstance, world)
    entry, mod = @timeit_debug to "emission" compile_method_instance(job, method_instance, world)
    ctx = context(mod)

    # clean up incompatibilities
    @timeit_debug to "clean-up" begin
        for llvmf in functions(mod)
            # only occurs in debug builds
            delete!(function_attributes(llvmf), EnumAttribute("sspstrong", 0, ctx))

            if VERSION < v"1.5.0-DEV.393"
                # make function names safe for ptxas
                llvmfn = LLVM.name(llvmf)
                if !isdeclaration(llvmf)
                    llvmfn′ = safe_name(llvmfn)
                    if llvmfn != llvmfn′
                        LLVM.name!(llvmf, llvmfn′)
                        llvmfn = llvmfn′
                    end
                end
            end

            if Sys.iswindows()
                personality!(llvmf, nothing)
            end
        end

        # remove the exception-handling personality function
        if Sys.iswindows() && "__julia_personality" in functions(mod)
            llvmf = functions(mod)["__julia_personality"]
            @compiler_assert isempty(uses(llvmf)) job
            unsafe_delete!(mod, llvmf)
        end
    end

    # target-specific processing
    process_module!(job, mod)

    # sanitize function names
    # FIXME: Julia should do this, but apparently fails (see maleadt/LLVM.jl#201)
    for f in functions(mod)
        LLVM.isintrinsic(f) && continue
        llvmfn = LLVM.name(f)
        startswith(llvmfn, "julia.") && continue # Julia intrinsics
        startswith(llvmfn, "llvm.") && continue # unofficial LLVM intrinsics
        llvmfn′ = safe_name(llvmfn)
        if llvmfn != llvmfn′
            @assert !haskey(functions(mod), llvmfn′)
            LLVM.name!(f, llvmfn′)
        end
    end

    # rename the entry point
    if job.source.name !== nothing
        LLVM.name!(entry, safe_name(string("julia_", job.source.name)))
    end

    # promote entry-points to kernels and mangle its name
    if job.source.kernel
        entry = promote_kernel!(job, mod, entry)
        LLVM.name!(entry, mangle_call(entry, job.source.tt))
    end

    # minimal required optimization
    @timeit_debug to "rewrite" ModulePassManager() do pm
        global current_job
        current_job = job

        linkage!(entry, LLVM.API.LLVMExternalLinkage)
        internalize!(pm, [LLVM.name(entry)])

        can_throw(job) || add!(pm, ModulePass("LowerThrow", lower_throw!))

        add_lowering_passes!(job, pm)

        run!(pm, mod)

        # NOTE: if an optimization is missing, try scheduling an entirely new optimization
        # to see which passes need to be added to the target-specific list
        #     LLVM.clopts("-print-after-all", "-filter-print-funcs=$(LLVM.name(entry))")
        #     ModulePassManager() do pm
        #         add_library_info!(pm, triple(mod))
        #         add_transform_info!(pm, tm)
        #         PassManagerBuilder() do pmb
        #             populate!(pm, pmb)
        #         end
        #         run!(pm, mod)
        #     end
    end

    return mod, entry
end


## name mangling

# we generate function names that look like C++ functions, because many NVIDIA tools
# support them, e.g., grouping different instantiations of the same kernel together.

function mangle_param(t, substitutions)
    t == Nothing && return "v"

    if isa(t, DataType) || isa(t, Core.Function)
        tn = safe_name(t)

        # handle substitutions
        sub = findfirst(isequal(tn), substitutions)
        if sub === nothing
            str = "$(length(tn))$tn"
            push!(substitutions, tn)
        elseif sub == 1
            str = "S_"
        else
            str = "S$(sub-2)_"
        end

        # encode typevars as template parameters
        if !isempty(t.parameters)
            str *= "I"
            for t in t.parameters
                str *= mangle_param(t, substitutions)
            end
            str *= "E"
        end

        str
    elseif isa(t, Integer)
        "Li$(t)E"
    else
        tn = safe_name(t)
        "$(length(tn))$tn"
    end
end

function mangle_call(f, tt)
    fn = safe_name(f)
    str = "_Z$(length(fn))$fn"

    substitutions = String[]
    for t in tt.parameters
        str *= mangle_param(t, substitutions)
    end

    return str
end

# make names safe for ptxas
safe_name(fn::String) = replace(fn, r"[^A-Za-z0-9_]"=>"_")
safe_name(f::Union{Core.Function,DataType}) = safe_name(String(nameof(f)))
safe_name(f::LLVM.Function) = safe_name(LLVM.name(f))
safe_name(x) = safe_name(repr(x))


## exception handling

# this pass lowers `jl_throw` and friends to GPU-compatible exceptions.
# this isn't strictly necessary, but has a couple of advantages:
# - we can kill off unused exception arguments that otherwise would allocate or invoke
# - we can fake debug information (lacking a stack unwinder)
#
# once we have thorough inference (ie. discarding `@nospecialize` and thus supporting
# exception arguments) and proper debug info to unwind the stack, this pass can go.
function lower_throw!(mod::LLVM.Module)
    job = current_job::CompilerJob
    ctx = context(mod)
    changed = false
    @timeit_debug to "lower throw" begin

    throw_functions = [
        # unsupported runtime functions that are used to throw specific exceptions
        "jl_throw"                      => "exception",
        "jl_error"                      => "error",
        "jl_too_few_args"               => "too few arguments exception",
        "jl_too_many_args"              => "too many arguments exception",
        "jl_type_error"                 => "type error",
        "jl_type_error_rt"              => "type error",
        "jl_undefined_var_error"        => "undefined variable error",
        "jl_bounds_error"               => "bounds error",
        "jl_bounds_error_v"             => "bounds error",
        "jl_bounds_error_int"           => "bounds error",
        "jl_bounds_error_tuple_int"     => "bounds error",
        "jl_bounds_error_unboxed_int"   => "bounds error",
        "jl_bounds_error_ints"          => "bounds error",
        "jl_eof_error"                  => "EOF error",
        # Julia-level exceptions that use unsupported inputs like interpolated strings
        r"julia_throw_exp_domainerror_\d+"      => "DomainError",
        r"julia_throw_complex_domainerror_\d+"  => "DomainError"
    ]

    for f in functions(mod)
        fn = LLVM.name(f)
        for (throw_fn, name) in throw_functions
            occursin(throw_fn, fn) || continue

            for use in uses(f)
                call = user(use)::LLVM.CallInst

                # replace the throw with a PTX-compatible exception
                let builder = Builder(ctx)
                    position!(builder, call)
                    emit_exception!(builder, name, call)
                    dispose(builder)
                end

                # remove the call
                call_args = collect(operands(call))[1:end-1] # last arg is function itself
                unsafe_delete!(LLVM.parent(call), call)

                # HACK: kill the exceptions' unused arguments
                for arg in call_args
                    # peek through casts
                    if isa(arg, LLVM.AddrSpaceCastInst)
                        cast = arg
                        arg = first(operands(cast))
                        isempty(uses(cast)) && unsafe_delete!(LLVM.parent(cast), cast)
                    end

                    if isa(arg, LLVM.Instruction) && isempty(uses(arg))
                        unsafe_delete!(LLVM.parent(arg), arg)
                    end
                end

                changed = true
            end

            @compiler_assert isempty(uses(f)) job
            break
         end
     end

    end
    return changed
end

# report an exception in a GPU-compatible manner
#
# the exact behavior depends on the debug level. in all cases, a `trap` will be emitted, On
# debug level 1, the exception name will be printed, and on debug level 2 the individual
# stack frames (as recovered from the LLVM debug information) will be printed as well.
function emit_exception!(builder, name, inst)
    job = current_job::CompilerJob
    bb = position(builder)
    fun = LLVM.parent(bb)
    mod = LLVM.parent(fun)
    ctx = context(mod)

    # report the exception
    if Base.JLOptions().debug_level >= 1
        name = globalstring_ptr!(builder, name, "exception")
        if Base.JLOptions().debug_level == 1
            call!(builder, Runtime.get(:report_exception), [name])
        else
            call!(builder, Runtime.get(:report_exception_name), [name])
        end
    end

    # report each frame
    if Base.JLOptions().debug_level >= 2
        rt = Runtime.get(:report_exception_frame)
        ft = convert(LLVM.FunctionType, rt, ctx)
        bt = backtrace(inst)
        for (i,frame) in enumerate(bt)
            idx = ConstantInt(parameters(ft)[1], i)
            func = globalstring_ptr!(builder, String(frame.func), "di_func")
            file = globalstring_ptr!(builder, String(frame.file), "di_file")
            line = ConstantInt(parameters(ft)[4], frame.line)
            call!(builder, rt, [idx, func, file, line])
        end
    end

    # signal the exception
    call!(builder, Runtime.get(:signal_exception))

    emit_trap!(job, builder, mod, inst)
end

function emit_trap!(@nospecialize(job::CompilerJob), builder, mod, inst)
    ctx = context(mod)
    trap = if haskey(functions(mod), "llvm.trap")
        functions(mod)["llvm.trap"]
    else
        LLVM.Function(mod, "llvm.trap", LLVM.FunctionType(LLVM.VoidType(ctx)))
    end
    call!(builder, trap)
end


## kernel promotion

# promote a function to a kernel
function promote_kernel!(@nospecialize(job::CompilerJob), mod::LLVM.Module, kernel::LLVM.Function)
    # pass non-opaque pointer arguments by value (this improves performance,
    # and is mandated by certain back-ends like SPIR-V). only do so for values
    # that aren't a Julia pointer, so we ca still pass those directly.
    kernel_ft = eltype(llvmtype(kernel)::LLVM.PointerType)::LLVM.FunctionType
    kernel_sig = Base.signature_type(job.source.f, job.source.tt)::Type
    kernel_types = filter(dt->!isghosttype(dt) &&
                              (VERSION < v"1.5.0-DEV.581" || !Core.Compiler.isconstType(dt)),
                          [kernel_sig.parameters...])
    @compiler_assert length(kernel_types) == length(parameters(kernel_ft)) job
    for (i, (param_ft,arg_typ)) in enumerate(zip(parameters(kernel_ft), kernel_types))
        if param_ft isa LLVM.PointerType && issized(eltype(param_ft)) &&
           !(arg_typ <: Ptr) && !(VERSION >= v"1.5-" && arg_typ <: Core.LLVMPtr)
            push!(parameter_attributes(kernel, i), EnumAttribute("byval"))
        end
    end

    # target-specific processing
    kernel = process_kernel!(job, mod, kernel)

    return kernel
end
