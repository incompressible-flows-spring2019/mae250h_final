using FastGaussQuadrature
using Serialization

import Base: *, \

export plan_laplacian, plan_laplacian!, Laplacian

const GL_NODES, GL_WEIGHTS = gausslegendre(100)
const LGF_DIR  = joinpath(@__DIR__, "cache")
const LGF_FILE = joinpath(LGF_DIR, "lgftable.dat")
const GAMMA = MathConstants.γ


function load_lgf(N)
    if isfile(LGF_FILE)
        G = deserialize(open(LGF_FILE, "r"))
        if size(G,1) ≥ N
            return G
        end
    end
    build_lgf(N)
end

function build_lgf(N)
    @info "Building and caching LGF table"

    # interpret N as number of cells in half domain + 1

    # L = 2(N-1)
    # x0, y0 = 0.0, 0.0
    # Δx = 1.0
    # xc = 0.5L; yc = 0.5L
    # ψ∞(x,y) = log.(x.^2+y.^2)/(4π)
    #
    # gL(y,t) = ψ∞(-0.5L,y-0.5L)
    # gR(y,t) = ψ∞(0.5L, y-0.5L)
    # gB(x,t) = ψ∞(x-0.5L,-0.5L)
    # gT(x,t) = ψ∞(x-0.5L, 0.5L)
    #
    # params = ScalarDirichletParameters(Δx,x0,y0,gL,gR,gB,gT)
    # gfull = NodeData(2N-1,2N-1)
    # rhs = NodeData(gfull)
    # icent = jcent = N
    # rhs[icent,jcent] = 1
    # apply_bc!(gfull,0,params)
    # f̃ = rhs - laplacian(gfull)
    # gfull .= f̃
    # poisson_dirichlet_fft!(gfull)
    # apply_bc!(gfull,0,params)
    #
    # g = zeros(N, N)
    # for y in 0:N-1, x in 0:y
    #   g[x+1,y+1] = gfull[x+icent,y+jcent]
    # end

    g = zeros(N, N)
    for y in 0:N-1, x in 0:y
        g[x+1,y+1] = lgf(x,y)
    end

    G = Symmetric(g)
    mkpath(LGF_DIR)

    serialize(open(LGF_FILE,"w"),G)
    G
end

quadgauss(f::Function) = dot(GL_WEIGHTS, f(GL_NODES))

function lgf(i, j)
    if i == j ==0
        return 0.0
    elseif i ≥ j
        v = quadgauss() do x
            if x == -1
                return sqrt(2)abs(i)
            else
                t = (x .+ 1)./2
                return 0.5real((1 .-
                                ( (t.-sqrt(1im))./(t.+sqrt(1im)) ).^(j.+abs(i)).*
                                ( (t.+sqrt(-1im))./(t.-sqrt(-1im)) ).^(j.-abs(i)) ))./t
            end

        end
        return 0.5v/pi
    else
        return lgf(j,i)
    end

end

const LGF_TABLE = load_lgf(1024)

"""
    plan_laplacian(dims::Tuple,[with_inverse=false],[fftw_flags=FFTW.ESTIMATE],
                          [dx=1.0])

Constructor to set up an operator for evaluating the discrete Laplacian on
nodal data of dimension `dims`. If the optional keyword
`with_inverse` is set to `true`, then it also sets up the inverse Laplacian
(the lattice Green's function, LGF). These can then be applied, respectively, with
`*` and `\\` operations on data of the appropriate size. The optional parameter
`dx` is used in adjusting the uniform value of the LGF to match the behavior
of the continuous analog at large distances; this is set to 1.0 by default.

Instead of the first argument, one can also supply `w::NodeData` to specify the
size of the domain.

# Example

```jldoctest
julia> w = NodeData(5,5);

julia> w[3,3] = 1.0;

julia> L = plan_laplacian(5,5;with_inverse=true)
Discrete Laplacian (and inverse) on a (nx = 6, ny = 6) grid with spacing 1.0

julia> s = L\\w
6×6 NodeData{5,5}:
 0.424413  0.38662   0.36338   0.38662   0.424413  0.462207
 0.38662   0.31831   0.25      0.31831   0.38662   0.440376
 0.36338   0.25      0.0       0.25      0.36338   0.430281
 0.38662   0.31831   0.25      0.31831   0.38662   0.440376
 0.424413  0.38662   0.36338   0.38662   0.424413  0.462207
 0.462207  0.440376  0.430281  0.440376  0.462207  0.488075

julia> L*s ≈ w
true
```
"""
function plan_laplacian end

"""
    plan_laplacian!(dims::Tuple,[with_inverse=false],[fftw_flags=FFTW.ESTIMATE],
                          [dx=1.0])

Same as [`plan_laplacian`](@ref), but operates in-place on data.
"""
function plan_laplacian! end

struct Laplacian{NX, NY, R, DX, inplace}
    conv::Union{CircularConvolution{NX, NY},Nothing}
end



for (lf,inplace) in ((:plan_laplacian,false),
                     (:plan_laplacian!,true))
    @eval function $lf(dims::Tuple{Int,Int};
                   with_inverse = false, fftw_flags = FFTW.ESTIMATE, dx = 1.0)
        NX, NY = dims
        if !with_inverse
            return Laplacian{NX, NY, false, dx, $inplace}(nothing)
        end

        G = view(LGF_TABLE, 1:NX, 1:NY)
        Laplacian{NX, NY, true, dx, $inplace}(CircularConvolution(G, fftw_flags))
    end

    @eval function $lf(nx::Int, ny::Int;
        with_inverse = false, fftw_flags = FFTW.ESTIMATE, dx = 1.0)
        $lf((nx, ny), with_inverse = with_inverse, fftw_flags = fftw_flags, dx = dx)
    end

    @eval function $lf(nodes::NodeData{NX,NY};
        with_inverse = false, fftw_flags = FFTW.ESTIMATE, dx = 1.0) where {NX,NY}
        $lf((NX+1,NY+1), with_inverse = with_inverse, fftw_flags = fftw_flags, dx = dx)
    end
end



function Base.show(io::IO, L::Laplacian{NX, NY, R, DX, inplace}) where {NX, NY, R, DX, inplace}
    nodedims = "(nx = $NX, ny = $NY)"
    inverse = R ? " (and inverse)" : ""
    isinplace = inplace ? " in-place" : ""
    print(io, "Discrete$isinplace Laplacian$inverse on a $nodedims grid with spacing $DX")
end

mul!(out::NodeData{NX,NY}, L::Laplacian, s::NodeData{NX,NY}) where {NX,NY} = laplacian!(out, s)
*(L::Laplacian{MX,MY,R,DX,false}, s::NodeData{NX,NY}) where {MX,MY,R,DX,NX,NY} =
      laplacian(s)
function (*)(L::Laplacian{MX,MY,R,DX,true}, s::NodeData{NX,NY}) where {MX,MY,R,DX,NX,NY}
    laplacian!(s,deepcopy(s))
end

function ldiv!(out::NodeData{NX, NY},
                   L::Laplacian{MX, MY, true, DX, inplace},
                   s::NodeData{NX, NY}) where {NX, NY, MX, MY, DX, inplace}

    mul!(out.data, L.conv, s.data)

    # Adjust the behavior at large distance to match continuous kernel
    #out.data .-= (sum(s.data)/2π)*(GAMMA+log(8)/2-log(DX))
    out
end

\(L::Laplacian{MX,MY,R,DX,false},s::NodeData{NX,NY}) where {MX,MY,R,DX,NX,NY} =
  ldiv!(NodeData(s), L, s)

\(L::Laplacian{MX,MY,R,DX,true},s::NodeData{NX,NY}) where {MX,MY,R,DX,NX,NY} =
  ldiv!(s, L, deepcopy(s))
