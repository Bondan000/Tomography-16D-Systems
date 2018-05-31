#The lattice

d = 4;
v = Matrix{Any}(zeros(d,d));
h = Matrix{Any}(zeros(d,d));
p = Matrix{Float64}(d,d);
x = Matrix{Float64}(d,d);
y = Matrix{Float64}(d,d);
Φ = Matrix{Float64}(d,d);



for i = 1:d
    h[i,d] = eye(d)[i,:];
end

#=for k = d:-1:1
    v[k,k] = (x[k,k] + y[k,k]im)*h[k,k];
    
    for j = k-1:-1:1 
        v[j,k] = (x[j,k] + y[j,k]im) * sqrt(complex(p[j,k])) * h[j,k] + sqrt(complex(1-p[j,k])) * v[j+1,k];
        h[j,k-1] = (x[j,k] + y[j,k]im) * sqrt(complex(1-p[j,k])) * h[j,k] - sqrt(complex(p[j,k])) * v[j+1,k];
        println("k:",k,"\t","j:",j)
    end
end=#

function H(m::Int, n::Int)
    for k = d:-1:1
        v[k,k] = (x[k,k] + y[k,k]im)*h[k,k];
        
        for j = k-1:-1:1 
            v[j,k] = (x[j,k] + y[j,k]im) * sqrt(complex(p[j,k])) * h[j,k] + sqrt(complex(1-p[j,k])) * v[j+1,k];
            h[j,k-1] = (x[j,k] + y[j,k]im) * sqrt(complex(1-p[j,k])) * h[j,k] - sqrt(complex(p[j,k])) * v[j+1,k];
        end
    end
    return (h[m,n])
end


#Ist = transpose(v);

#-----------------------------------------------------------------------------------------------------
#DFT

m(n::Int) = 1/sqrt(n)*[exp(2*pi*im*(r-1)*(s-1)/n) for r in 1:n, s in 1:n];

#Soll = m(d);

#------------------------------------------------------------------------------------------------------
#The solution of the algorithm

Denom(j::Int,k::Int) = dot(eye(d)[j,:], H(j,k)) * (j==1 ? 1.0 : prod(sqrt(complex(1-p[l,k])) for l = 1:j-1));

Num(j::Int,k::Int) = j==1 ? 0.0 : sum((x[i,k] + y[i,k]im) * dot(eye(d)[j,:], H(i,k)) * sqrt(complex(p[i,k])) * (i==1 ? 1.0 : prod(sqrt(complex(1-p[l,k])) for l = 1:i-1)) for i = 1:j-1);

Z(j::Int,k::Int) = (m(d)[j,k] - Num(j,k)) / Denom(j,k);

#=function Angle(Re,Im,j,k)
    if abs(exp(asin(Float32(Im))*im) - (Re + Im*im)) < 10.0^-12
        Φ[j,k] = asin(Float32(Im));
    else
        Φ[j,k] = -asin(Float32(Im)) - pi;
    end
end=#
#Angle(Re,Im,j,k) = abs(exp(asin(Float32(Im))*im) - (Re + Float32(Im)*im)) < 10.0^-12 ? Φ[j,k] = asin(Float32(Im)) : Φ[j,k] = -asin(Float32(Im)) - pi;
Angle(Re::Float16,Im::Float16,j::Int,k::Int) = abs(exp(asin(Im)*im) - (Re + Im*im)) < 10.0^-12 ? Φ[j,k] = asin(Im) : Φ[j,k] = -asin(Im) - pi;

#-----------------------------------------------------------------------------------------------------
#Evaluation

@time for k = d:-1:1
    for j = 1:k-1
        p[j,k] = abs2(Z(j,k));
        x[j,k] = real(Z(j,k)/sqrt(complex(p[j,k])));
        y[j,k] = imag(Z(j,k)/sqrt(complex(p[j,k])));
        Angle(Float16(x[j,k]),Float16(y[j,k]),j,k);
        println("p(", j, ",", k, ") = ", p[j,k], "  Φ(", j, ",", k, ") = ", Φ[j,k]*180/pi);
    end

    x[k,k] = real(Z(k,k));
    y[k,k] = imag(Z(k,k));
    Angle(Float16(x[k,k]),Float16(y[k,k]),k,k);
    println("Φ(", k, ",", k, ") = ", Φ[k,k]*180/pi);
end

#------------------------------------------------------------------------------------------------------
