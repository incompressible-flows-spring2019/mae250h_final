using SpecialFunctions

export plan_intfact, plan_intfact!, IntFact, intfact


intfact(x, y,a) = exp(-4a)besseli(x,2a)besseli(y,2a)

# Integrating factor

"""
    plan_intfact(t::Real,dims::Tuple,[fftw_flags=FFTW.ESTIMATE])

Constructor to set up an operator for evaluating the integrating factor with
real-valued dimensionless time parameter `t`. This can then be applied with the `*` operation on
data of the appropriate size.

The `dims` argument can be replaced with `w::NodeData` to specify the size of the
domain.

# Example

```jldoctest
julia> w = NodeData(6,6);

julia> w[4,4] = 1.0;

julia> H = plan_intfact(1.0,w)
Integrating factor with parameter 1.0 on a (nx = 7, ny = 7) grid

julia> H*w
7×7 NodeData{6,6}:
 0.000828935  0.00268447  0.00619787  …  0.00619787  0.00268447  0.000828935
 0.00268447   0.00869352  0.0200715      0.0200715   0.00869352  0.00268447
 0.00619787   0.0200715   0.0463409      0.0463409   0.0200715   0.00619787
 0.00888233   0.028765    0.0664124      0.0664124   0.028765    0.00888233
 0.00619787   0.0200715   0.0463409      0.0463409   0.0200715   0.00619787
 0.00268447   0.00869352  0.0200715   …  0.0200715   0.00869352  0.00268447
 0.000828935  0.00268447  0.00619787     0.00619787  0.00268447  0.000828935
```
"""
function plan_intfact end

"""
    plan_intfact!(a::Real,dims::Tuple,[fftw_flags=FFTW.ESTIMATE])

Same as [`plan_intfact`](@ref), but the resulting operator performs an in-place
operation on data.
"""
function plan_intfact! end


struct IntFact{NX, NY, a, inplace}
    conv::Union{CircularConvolution{NX, NY},Nothing}
end

for (lf,inplace) in ((:plan_intfact,false),
                     (:plan_intfact!,true))

    @eval function $lf(a::Real,dims::Tuple{Int,Int};fftw_flags = FFTW.ESTIMATE)
        NX, NY = dims

        if a == 0
          return IntFact{NX, NY, 0.0, $inplace}(nothing)
        end

        #qtab = [intfact(x, y, a) for x in 0:NX-1, y in 0:NY-1]
        Nmax = 0
        while abs(intfact(Nmax,0,a)) > eps(Float64)
          Nmax += 1
        end
        qtab = [max(x,y) <= Nmax ? intfact(x, y, a) : 0.0 for x in 0:NX-1, y in 0:NY-1]
        IntFact{NX, NY, a, $inplace}(CircularConvolution(qtab, fftw_flags))
      end

      @eval $lf(a::Real,nodes::NodeData{NX,NY}; fftw_flags = FFTW.ESTIMATE) where {NX,NY} =
          $lf(a,(NX+1,NY+1), fftw_flags = fftw_flags)


end


function Base.show(io::IO, E::IntFact{NX, NY, a, inplace}) where {NX, NY, a, inplace}
    nodedims = "(nx = $NX, ny = $NY)"
    isinplace = inplace ? "In-place integrating factor" : "Integrating factor"
    print(io, "$isinplace with parameter $a on a $nodedims grid")
end

function mul!(out::NodeData{NX, NY},
                   E::IntFact{MX, MY, a, inplace},
                   s::NodeData{NX, NY}) where {NX, NY, MX, MY, a, inplace}

    mul!(out.data, E.conv, s.data)
    out
end

function mul!(out::NodeData{NX, NY},
                   E::IntFact{MX, MY, 0.0, inplace},
                   s::NodeData{NX, NY}) where {NX, NY, MX, MY, inplace}
    out .= deepcopy(s)
end

*(E::IntFact{MX,MY,a,false},s::NodeData{NX,NY}) where {MX,MY,a, NX,NY} =
  mul!(NodeData(s), E, s)

*(E::IntFact{MX,MY,a,true},s::NodeData{NX,NY}) where {MX,MY,a, NX,NY} =
    mul!(s, E, deepcopy(s))


# Identity

struct Identity end


(*)(::Identity,s::GridData) = s
