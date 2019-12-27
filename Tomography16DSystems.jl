
#Maximum Likelihood Estimation for 16 dimensional systems

using Base.Iterators, LinearAlgebra, Optim, Plotly;
import DelimitedFiles.readdlm;

#Here is the tensor product definition
#CircleTimes = kron;
tp(m1, m2) = partition(flatten(m1 .* m2),2);

n = Int64(16);
#Define the single qubit measurements, R is the state with "-i".
H = Float64[1, 0];
V = Float64[0, 1];
Dia = Float64[1/sqrt(2), 1/sqrt(2)];
A = Float64[1/sqrt(2), -1/sqrt(2)];
R = ComplexF64[1/sqrt(2), -im/sqrt(2)];
L = ComplexF64[1/sqrt(2), im/sqrt(2)];
statevars = [H, V, Dia, A, R, L];

HadamardMatrix_2 = Float64[1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)];
PauliMatrix_1 = Float64[0 1; 1 0];

Trafo = kron(HadamardMatrix_2, Matrix{Float64}(I, 2,2), Matrix{Float64}(I, 2,2), PauliMatrix_1 * HadamardMatrix_2);
NDimensions = 16;

#Here are the specific states and data measured in the tomography experiment
D2Array = readdlm("D:\\Binto\\ddd\\Julia\\Tomography_Mathematica_to_Julia\\measurement_bases_full_edit_measurement_7.txt", skipstart=2)[1:end,2:17];
Data = collect(flatten(D2Array));
SigmaData = sqrt.(Data);

function BadnessPolynomial!(t)
    pM = [t[1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        t[17] + im*t[18] t[2] 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
        t[47] + im*t[48] t[19] + im*t[20] t[3] 0 0 0 0 0 0 0 0 0 0 0 0 0;
        t[75] + im*t[76] t[49] + im*t[50] t[21] + im*t[22] t[4] 0 0 0 0 0 0 0 0 0 0 0 0;
        t[101] + im*t[102] t[77] + im*t[78] t[51] + im*t[52] t[23] + im*t[24] t[5] 0 0 0 0 0 0 0 0 0 0 0;
        t[125] + im*t[126] t[103] + im*t[104] t[79] + im*t[80] t[53] + im*t[54] t[25] + im*t[26] t[6] 0 0 0 0 0 0 0 0 0 0;
        t[147] + im*t[148] t[127] + im*t[128] t[105] + im*t[106] t[81] + im*t[82] t[55] + im*t[56] t[27] + im*t[28] t[7] 0 0 0 0 0 0 0 0 0;
        t[167] + im*t[168] t[149] + im*t[150] t[129] + im*t[130] t[107] + im*t[108] t[83] + im*t[84] t[57] + im*t[58] t[29] + im*t[30] t[8] 0 0 0 0 0 0 0 0;
        t[185] + im*t[186] t[169] + im*t[170] t[151] + im*t[152] t[131] + im*t[132] t[109] + im*t[110] t[85] + im*t[86] t[59] + im*t[60] t[31] + im*t[32] t[9] 0 0 0 0 0 0 0;
        t[201] + im*t[202] t[187] + im*t[188] t[171] + im*t[172] t[153] + im*t[154] t[133] + im*t[134] t[111] + im*t[112] t[87] + im*t[88] t[61] + im*t[62] t[33] + im*t[34] t[10] 0 0 0 0 0 0;
        t[215] + im*t[216] t[203] + im*t[204] t[189] + im*t[190] t[173] + im*t[174] t[155] + im*t[156] t[135] + im*t[136] t[113] + im*t[114] t[89] + im*t[90] t[63] + im*t[64] t[35] + im*t[36] t[11] 0 0 0 0 0;
        t[227] + im*t[228] t[217] + im*t[218] t[205] + im*t[206] t[191] + im*t[192] t[175] + im*t[176] t[157] + im*t[158] t[137] + im*t[138] t[115] + im*t[116] t[91] + im*t[92] t[65] + im*t[66] t[37] + im*t[38] t[12] 0 0 0 0;
        t[237] + im*t[238] t[229] + im*t[230] t[219] + im*t[220] t[207] + im*t[208] t[193] + im*t[194] t[177] + im*t[178] t[159] + im*t[160] t[139] + im*t[140] t[117] + im*t[118] t[93] + im*t[94] t[67] + im*t[68] t[39] + im*t[40] t[13] 0 0 0;
        t[245] + im*t[246] t[239] + im*t[240] t[231] + im*t[232] t[221] + im*t[222] t[209] + im*t[210] t[195] + im*t[196] t[179] + im*t[180] t[161] + im*t[162] t[141] + im*t[142] t[119] + im*t[120] t[95] + im*t[96] t[69] + im*t[70] t[41] + im*t[42] t[14] 0 0;
        t[251] + im*t[252] t[247] + im*t[248] t[241] + im*t[242] t[233] + im*t[234] t[223] + im*t[224] t[211] + im*t[212] t[197] + im*t[198] t[181] + im*t[182] t[163] + im*t[164] t[143] + im*t[144] t[121] + im*t[122] t[97] + im*t[98] t[71] + im*t[72] t[43] + im*t[44] t[15] 0;
        t[255] + im*t[256] t[253] + im*t[254] t[249] + im*t[250] t[243] + im*t[244] t[235] + im*t[236] t[225] + im*t[226] t[213] + im*t[214] t[199] + im*t[200] t[183] + im*t[184] t[165] + im*t[166] t[145] + im*t[146] t[123] + im*t[124] t[99] + im*t[100] t[73] + im*t[74] t[45] + im*t[46] t[16]];

    pMd = [t[1] t[17] - im*t[18] t[47] - im*t[48] t[75] - im*t[76] t[101] - im*t[102] t[125] - im*t[126] t[147] - im*t[148] t[167] - im*t[168] t[185] - im*t[186] t[201] - im*t[202] t[215] - im*t[216] t[227] - im*t[228] t[237] - im*t[238] t[245] - im*t[246] t[251] - im*t[252] t[255] - im*t[256];
        0 t[2] t[19] - im*t[20] t[49] - im*t[50] t[77] - im*t[78] t[103] - im*t[104] t[127] - im*t[128] t[149] - im*t[150] t[169] - im*t[170] t[187] - im*t[188] t[203] - im*t[204] t[217] - im*t[218] t[229] - im*t[230] t[239] - im*t[240] t[247] - im*t[248] t[253] - im*t[254];
        0 0 t[3] t[21] - im*t[22] t[51] - im*t[52] t[79] - im*t[80] t[105] - im*t[106] t[129] - im*t[130] t[151] - im*t[152] t[171] - im*t[172] t[189] - im*t[190] t[205] - im*t[206] t[219] - im*t[220] t[231] - im*t[232] t[241] - im*t[242] t[249] - im*t[250];
        0 0 0 t[4] t[23] - im*t[24] t[53] - im*t[54] t[81] - im*t[82] t[107] - im*t[108] t[131] - im*t[132] t[153] - im*t[154] t[173] - im*t[174] t[191] - im*t[192] t[207] - im*t[208] t[221] - im*t[222] t[233] - im*t[234] t[243] - im*t[244];
        0 0 0 0 t[5] t[25] - im*t[26] t[55] - im*t[56] t[83] - im*t[84] t[109] - im*t[110] t[133] - im*t[134] t[155] - im*t[156] t[175] - im*t[176] t[193] - im*t[194] t[209] - im*t[210] t[223] - im*t[224] t[235] - im*t[236];
        0 0 0 0 0 t[6] t[27] - im*t[28] t[57] - im*t[58] t[85] - im*t[86] t[111] - im*t[112] t[135] - im*t[136] t[157] - im*t[158] t[177] - im*t[178] t[195] - im*t[196] t[211] - im*t[212] t[225] - im*t[226];
        0 0 0 0 0 0 t[7] t[29] - im*t[30] t[59] - im*t[60] t[87] - im*t[88] t[113] - im*t[114] t[137] - im*t[138] t[159] - im*t[160] t[179] - im*t[180] t[197] - im*t[198] t[213] - im*t[214];
        0 0 0 0 0 0 0 t[8] t[31] - im*t[32] t[61] - im*t[62] t[89] - im*t[90] t[115] - im*t[116] t[139] - im*t[140] t[161] - im*t[162] t[181] - im*t[182] t[199] - im*t[200];
        0 0 0 0 0 0 0 0 t[9] t[33] - im*t[34] t[63] - im*t[64] t[91] - im*t[92] t[117] - im*t[118] t[141] - im*t[142] t[163] - im*t[164] t[183] - im*t[184];
        0 0 0 0 0 0 0 0 0 t[10] t[35] - im*t[36] t[65] - im*t[66] t[93] - im*t[94] t[119] - im*t[120] t[143] - im*t[144] t[165] - im*t[166];
        0 0 0 0 0 0 0 0 0 0 t[11] t[37] - im*t[38] t[67] - im*t[68] t[95] - im*t[96] t[121] - im*t[122] t[145] - im*t[146];
        0 0 0 0 0 0 0 0 0 0 0 t[12] t[39] - im*t[40] t[69] - im*t[70] t[97] - im*t[98] t[123] - im*t[124];
        0 0 0 0 0 0 0 0 0 0 0 0 t[13] t[41] - im*t[42] t[71] - im*t[72] t[99] - im*t[100];
        0 0 0 0 0 0 0 0 0 0 0 0 0 t[14] t[43] - im*t[44] t[73] - im*t[74];
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 t[15] t[45] - im*t[46];
        0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 t[16]];

    global GeneralDM = pMd * pM;
    Prediction(DM, State) = sum(conj(State) .* DM .* State);

    StateMeasured = [kron(i,j,k,l) for i in statevars, j in statevars, k in statevars, l in statevars];

    PredictionPart = [];
    for i = 1:length(StateMeasured)
        push!(PredictionPart, Prediction(GeneralDM, StateMeasured[i]));
    end
    BD = sum((PredictionPart[i] - Data[i])^2 /PredictionPart[i] for i=1:length(StateMeasured));
    return real(BD)
end

lower = fill(-5.0, 256);
upper = fill(5.0, 256);
t0 = fill(0.05, 256); #initial points

#took  92secs for 20000 iterations! reached max iterations!
@time res = optimize(BadnessPolynomial!, t0, NelderMead(), Optim.Options(iterations = 10000))
#took 77secs for 20000 iterations! reached max iterations!
#@time res = optimize(BadnessPolynomial!, t0, SimulatedAnnealing(), Optim.Options(iterations = 20000))
#took 10mins for 20000 iterations! objective increased between iterations!
#@time res = optimize(BadnessPolynomial!, t0, LBFGS(), Optim.Options(iterations = 20000))
#took 15mins! solution in 6 iterations!
#inner_optimizer = GradientDescent()
#@time res = optimize(BadnessPolynomial!, lower, upper, t0, Fminbox(inner_optimizer), Optim.Options(iterations = 10)) #Box constrained minimization
#temp = Optim.minimizer(res);

MaxLikDM = GeneralDM/tr(GeneralDM);
n1 = 0;
n2 = 6;
n3 = 4;
n4 = 4;

IdealDM = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 1/2 0.0 0.0 -(1/2) 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 -(1/2) 0.0 0.0 1/2 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0];

IdealDMIm = [0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0;
    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0];

Fidelity = tr(MaxLikDM.*IdealDM);
Fidelity2 = tr((MaxLikDM^1/2 .* IdealDM .* MaxLikDM^1/2)^1/2)^2;
Purity = tr(MaxLikDM.*MaxLikDM);
LinearEntropy = n/(n - 1)*(1 - Purity);

#=function plotme3d(dta)
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
    p1 = plot([scatter3d(x = x, y = y, z = re_z, mode = "markers", marker = attr(size = 5), name = "Real part")])
    p2 = plot([scatter3d(x = x, y = y, z = im_z, mode = "markers", marker = attr(size = 5), name = "Imaginary part")])
    [p1, p2]
end =#

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
    layout = Layout(xaxis = attr(tickvals=[1,7,10,16], ticktext = ["HHHH", "HVVH", "VHHV", "VVVV"]),
            yaxis = attr(tickvals=[1,7,10,16], ticktext = ["HHHH", "HVVH", "VHHV", "VVVV"]));
    #plot([trace1, trace2])
    p1 = plot(trace1, layout);
    p2 = plot(trace2, layout);
    [p1, p2]
end


plotme3d(MaxLikDM)
plotme3d(IdealDM)
