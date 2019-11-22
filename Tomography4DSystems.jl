using Base.Iterators, LinearAlgebra, Optim;
import DelimitedFiles.readdlm;

n = 4;
H = [1, 0];
V = [0, 1];
Dia = [1/sqrt(2), 1/sqrt(2)];
A = [1/sqrt(2), -1/sqrt(2)];
R = [1/sqrt(2), -im/sqrt(2)];
L = [1/sqrt(2), im/sqrt(2)];
statevars = [H, V, Dia, A, R, L];

HadamardMatrix_2 = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)];
PauliMatrix_1 = [0 1; 1 0];

Trafo = kron(HadamardMatrix_2 .* Matrix{Float64}(I, 2,2), PauliMatrix_1 * HadamardMatrix_2);
NDimensions = 4;

D2Array = readdlm("D:\\Binto\\ddd\\Julia\\Tomography_Mathematica_to_Julia\\measurement_bases_full_edit_measurement_7.txt", skipstart=2)[1:end,2:5];
Data = collect(flatten(D2Array));
SigmaData = sqrt.(Data);

function BadnessPolynomial!(t)
    pM = [t[1] 0 0 0;
        t[5] + im*t[6]  t[2] 0 0;
        t[11] + im*t[12] t[7] + im*t[8] t[3] 0;
        t[15] + im*t[16] t[13] + im*t[14] t[9] + im*t[10] t[4]];

    pMd = [t[1] t[5] + im*t[6] t[11] + im*t[12] t[15] + im*t[16];
        0 t[2] t[7] + im*t[8] t[13] + im*t[14];
        0 0 t[3] t[9] + im*t[10];
        0 0 0 t[4]];
    
    global GeneralDM = pMd * pM;
    Prediction(DM, State) = sum(conj(State) .* DM .* State);
    StateMeasured = [kron(i,j) for i in statevars, j in statevars];
    #global PredictionPart = [sum(conj(StateMeasured[i]) .* GeneralDM .* StateMeasured[i]) for i = 1:length(StateMeasured)];
    PredictionPart = [];
    for i = 1:length(StateMeasured)
        push!(PredictionPart, Prediction(GeneralDM, StateMeasured[i]));
    end
    return BD = real(sum((PredictionPart[i] - Data[i])^2 /PredictionPart[i] for i=1:length(StateMeasured)))
end
lower = fill(-5.0, 16);
upper = fill(5.0, 16);
t0 = fill(0.05, 16);
#t0 = randn(16)
#@time res = optimize(BadnessPolynomial!, t0, NelderMead(), Optim.Options(g_tol = 10e-10, iterations = 20000))
#@time res = optimize(BadnessPolynomial!, t0, LBFGS(), Optim.Options(iterations = 20000)) #Requires a function and gradient (will be approximated if omitted)
inner_optimizer = GradientDescent()
@time res = optimize(BadnessPolynomial!, lower, upper, t0, Fminbox(inner_optimizer), Optim.Options(iterations = 50000)) #Box constrained minimization
temp = Optim.minimizer(res);

MaxLikDM = GeneralDM/tr(GeneralDM)

n1 = 0;
n2 = 6;
n3 = 4;
n4 = 4;
