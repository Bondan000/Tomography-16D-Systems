using Base.Iterators, LinearAlgebra;
#using DataFrames;
using JuMP;
import DelimitedFiles.readdlm;

#Create a JuMP model
model = Model();

#Here is the tensor product definition
#CircleTimes = kron;
tp(m1, m2) = partition(flatten(m1 .* m2),2);

n=4;
#@variable(model, Im)
#@NLconstraint(model, Im^2 == -1)
#Define the single qubit measurements, R is the state with "-i"..
H = [1, 0];
V = [0, 1];
Dia = [1/sqrt(2), 1/sqrt(2)];
A = [1/sqrt(2), -1/sqrt(2)];
R = [1/sqrt(2), -im/sqrt(2)];
#R = [1/sqrt(2), -Im/sqrt(2)];
L = [1/sqrt(2), im/sqrt(2)];
#L = [1/sqrt(2), Im/sqrt(2)];

HadamardMatrix_2 = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)];
PauliMatrix_1 = [0 1; 1 0];

Trafo = kron(HadamardMatrix_2 .* Matrix{Float64}(I, 2,2), PauliMatrix_1 * HadamardMatrix_2);
NDimensions = 4;

#@variable(model, t[1:16])
@variable(model, t_Re[1:16])
@variable(model, t_Im[1:16])

pM = [t_Re[1] 0 0 0;
    t_Re[5] + t_Im[6]  t_Re[2] 0 0;
    t_Re[11] + t_Im[12] t_Re[7] + t_Im[8] t_Re[3] 0;
    t_Re[15] + t_Im[16] t_Re[13] + t_Im[14] t_Re[9] + t_Im[10] t_Re[4]];

pMd = [t_Re[1] t_Re[5] + t_Im[6] t_Re[11] + t_Im[12] t_Re[15] + t_Im[16];
    0 t_Re[2] t_Re[7] + t_Im[8] t_Re[13] + t_Im[14];
    0 0 t_Re[3] t_Re[9] + t_Im[10];
    0 0 0 t_Re[4]];

#=
pM = [t[1] 0 0 0;
    t[5] + Im*t[6]  t[2] 0 0;
    t[11] + Im*t[12] t[7] + Im*t[8] t[3] 0;
    t[15] + Im*t[16] t[13] + Im*t[14] t[9] + Im*t[10] t[4]];

pMd = [t[1] t[5] + Im*t[6] t[11] + Im*t[12] t[15] + Im*t[16];
    0 t[2] t[7] + Im*t[8] t[13] + Im*t[14];
    0 0 t[3] t[9] + Im*t[10];
    0 0 0 t[4]];
=#

GeneralDM = pMd*pM;
Prediction(DM, State) = State' .* DM .* State;

D2Array = readdlm("D:\\Binto\\ddd\\Julia\\Tomography_Mathematica_to_Julia\\measurement_bases_full_edit_measurement_7.txt", skipstart=2)[1:end,2:5];
Data = collect(flatten(D2Array));
SigmaData = sqrt.(Data);

statevars = [H, V, Dia, A, R, L];

#StateMeasured = collect(partition(flatten([kron(i,j) for i in statevars, j in statevars]),4));
StateMeasured = [kron(i,j) for i in statevars, j in statevars];

PredictionPart = [];
@time for i = 1:length(StateMeasured)
    push!(PredictionPart, Prediction(GeneralDM, real(StateMeasured[i])));
end

BadnessPolynomial = sum((PredictionPart[i] - Data[i])^2 /PredictionPart[i] for i=1:length(StateMeasured));