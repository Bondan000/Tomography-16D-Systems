using Base.Iterators, LinearAlgebra, Optim, Plotly;
import DelimitedFiles.readdlm;

NDimensions = Int64(4);
#Define the single qubit measurements, R is the state with "-i".
H = Float64[1, 0];
V = Float64[0, 1];
statevars = [H, V];

Id = Matrix{Float64}(I, 2, 2);
Hx = [1 1; 1 -1]/sqrt(2);
Hy = [1 1; im -im]/sqrt(2);
Trafovars = [Id, Hx, Hy];

Trafos = [kron(l,k) for k in Trafovars, l in Trafovars];
StateArray = [kron(l,k) for k in statevars, l in statevars];
StateMeasured = [];

for a in 1:length(Trafos)
    for b in 1:NDimensions
        push!(StateMeasured,Trafos[a]*StateArray[b])
    end
end

D2Array = readdlm("D:\\myJulia\\two_qubit_tomo_measurement_2.txt", skipstart=2)[1:end,34:37];
Data = vec(transpose(D2Array));
SigmaData = sqrt.(Data);


function BadnessPolynomial!(t)
    pM = [t[1] 0 0 0;
        t[5] + im*t[6]  t[2] 0 0;
        t[11] + im*t[12] t[7] + im*t[8] t[3] 0;
        t[15] + im*t[16] t[13] + im*t[14] t[9] + im*t[10] t[4]];

    pMd = conj(transpose(pM));

    global GeneralDM = pMd*pM;
    Prediction(DM, State) = sum(conj(transpose(State))*DM*State);
    
    BD = sum((Prediction(GeneralDM, StateMeasured[i]) - Data[i])^2/Prediction(GeneralDM, StateMeasured[i]) for i=1:length(StateMeasured));
    return real(BD)
end

Bell = (kron(H,V)-kron(V,H))/sqrt(2);
IdealDM = reshape(kron(Bell, conj(Bell)),(NDimensions,NDimensions)) 

t0 = vec(IdealDM);
@time res = optimize(BadnessPolynomial!, t0, NelderMead(), Optim.Options(g_tol = 10e-10, iterations = 20000));
#@time res = optimize(BadnessPolynomial!, t0, LBFGS(), Optim.Options(iterations = 20000)) #Requires a function and gradient (will be approximated if omitted)
#inner_optimizer = GradientDescent()
#lower = fill(-1.0, NDimensions);
#upper = fill(1.0, NDimensions);
#@time res = optimize(BadnessPolynomial!, lower, upper, t0, Fminbox(inner_optimizer), Optim.Options(iterations = 50000)) #Box constrained minimization
temp = Optim.minimizer(res);

MaxLikDM = GeneralDM/tr(GeneralDM)

Fidelity = tr(sqrt(sqrt(MaxLikDM)*IdealDM*sqrt(MaxLikDM)))^2;
Purity = tr(MaxLikDM*MaxLikDM);
LinearEntropy = NDimensions/(NDimensions - 1)*(1 - Purity);


function plotme3d(dta)
    x = [];
    y = [];
    re_z = [];
    im_z = [];
    for i = 1:size(dta)[1]
        for j = 1:size(dta)[1]
            push!(x, i);
            push!(y, j);
            push!(re_z, real(dta[i,j]));
            push!(im_z, imag(dta[i,j]));
        end
    end

    trace1 = scatter3d(x = x, y = y, z = re_z, mode = "markers", marker = attr(size = 5), name = "Real part");
    trace2 = scatter3d(x = x, y = y, z = im_z, mode = "markers", marker = attr(size = 5), name = "Imaginary part");
    layout = Layout(xaxis = attr(tickvals=[1,2,3,4], ticktext = ["HH", "HV", "VH", "VV "]),
            yaxis = attr(tickvals=[1,2,3,4], ticktext = ["HH", "HV", "VH", "VV "]));
    #plot([trace1, trace2])
    p1 = Plotly.plot(trace1, layout);
    p2 = Plotly.plot(trace2, layout);
    [p1, p2]
end

plotme3d(MaxLikDM)
