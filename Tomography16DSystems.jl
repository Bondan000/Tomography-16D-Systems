
#Maximum Likelihood Estimation for 16 dimensional systems

using Base.Iterators, LinearAlgebra;
#using DataFrames;
using JuMP;
import DelimitedFiles.readdlm;

#Create a JuMP model
model = Model();

#Here is the tensor product definition
#CircleTimes = kron;
tp(m1, m2) = partition(flatten(m1 .* m2),2);

n=16;
#Define the single qubit measurements, R is the state with "-i"..
H = [1, 0];
V = [0, 1];
Dia = [1/sqrt(2), 1/sqrt(2)];
A = [1/sqrt(2), -1/sqrt(2)];
R = [1/sqrt(2), -im/sqrt(2)];
L = [1/sqrt(2), im/sqrt(2)];

HadamardMatrix_2 = [1/sqrt(2) 1/sqrt(2); 1/sqrt(2) -1/sqrt(2)];
PauliMatrix_1 = [0 1; 1 0];

Trafo = kron(HadamardMatrix_2, Matrix{Float64}(I, 2,2), Matrix{Float64}(I, 2,2), PauliMatrix_1 * HadamardMatrix_2);
NDimensions = 16;

#=
t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,t30,
t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,t42,t43,t44,t45,t46,t47,t48,t49,t50,t51,t52,t53,t54,t55,t56,t57,t58,t59,t60,
t61,t62,t63,t64,t65,t66,t67,t68,t69,t70,t71,t72,t73,t74,t75,t76,t77,t78,t79,t80,t81,t82,t83,t84,t85,t86,t87,t88,t89,t90,
t91,t92,t93,t94,t95,t96,t97,t98,t99,t100,t101,t102,t103,t104,t105,t106,t107,t108,t109,t110,t111,t112,t113,t114,t115,t116,t117,t118,t119,t120,
t121,t122,t123,t124,t125,t126,t127,t128,t129,t130,t131,t132,t133,t134,t135,t136,t137,t138,t139,t140,t141,t142,t143,t144,t145,t146,t147,t148,t149,t150,
t151,t152,t153,t154,t155,t156,t157,t158,t159,t160,t161,t162,t163,t164,t165,t166,t167,t168,t169,t170,t171,t172,t173,t174,t175,t176,t177,t178,t179,t180,
t181,t182,t183,t184,t185,t186,t187,t188,t189,t190,t191,t192,t193,t194,t195,t196,t197,t198,t199,t200,t201,t202,t203,t204,t205,t206,t207,t208,t209,t210,
t211,t212,t213,t214,t215,t216,t217,t218,t219,t220,t221,t222,t223,t224,t225,t226,t227,t228,t229,t230,t231,t232,t233,t234,t235,t236,t237,t238,t239,t240,
t241,t242,t243,t244,t245,t246,t247,t248,t249,t250,t251,t252,t253,t254,t255,t256 = ones(Float64, 256);

pM = [t1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
   t17 + im*t18 t2 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
   t47 + im*t48 t19 + im*t20 t3 0 0 0 0 0 0 0 0 0 0 0 0 0;
   t75 + im*t76 t49 + im*t50 t21 + im*t22 t4 0 0 0 0 0 0 0 0 0 0 0 0;
   t101 + im*t102 t77 + im*t78 t51 + im*t52 t23 + im*t24 t5 0 0 0 0 0 0 0 0 0 0 0;
   t125 + im*t126 t103 + im*t104 t79 + im*t80 t53 + im*t54 t25 + im*t26 t6 0 0 0 0 0 0 0 0 0 0;
   t147 + im*t148 t127 + im*t128 t105 + im*t106 t81 + im*t82 t55 + im*t56 t27 + im*t28 t7 0 0 0 0 0 0 0 0 0;
   t167 + im*t168 t149 + im*t150 t129 + im*t130 t107 + im*t108 t83 + im*t84 t57 + im*t58 t29 + im*t30 t8 0 0 0 0 0 0 0 0;
   t185 + im*t186 t169 + im*t170 t151 + im*t152 t131 + im*t132 t109 + im*t110 t85 + im*t86 t59 + im*t60 t31 + im*t32 t9 0 0 0 0 0 0 0;
   t201 + im*t202 t187 + im*t188 t171 + im*t172 t153 + im*t154 t133 + im*t134 t111 + im*t112 t87 + im*t88 t61 + im*t62 t33 + im*t34 t10 0 0 0 0 0 0;
   t215 + im*t216 t203 + im*t204 t189 + im*t190 t173 + im*t174 t155 + im*t156 t135 + im*t136 t113 + im*t114 t89 + im*t90 t63 + im*t64 t35 + im*t36 t11 0 0 0 0 0;
   t227 + im*t228 t217 + im*t218 t205 + im*t206 t191 + im*t192 t175 + im*t176 t157 + im*t158 t137 + im*t138 t115 + im*t116 t91 + im*t92 t65 + im*t66 t37 + im*t38 t12 0 0 0 0;
   t237 + im*t238 t229 + im*t230 t219 + im*t220 t207 + im*t208 t193 + im*t194 t177 + im*t178 t159 + im*t160 t139 + im*t140 t117 + im*t118 t93 + im*t94 t67 + im*t68 t39 + im*t40 t13 0 0 0;
   t245 + im*t246 t239 + im*t240 t231 + im*t232 t221 + im*t222 t209 + im*t210 t195 + im*t196 t179 + im*t180 t161 + im*t162 t141 + im*t142 t119 + im*t120 t95 + im*t96 t69 + im*t70 t41 + im*t42 t14 0 0;
   t251 + im*t252 t247 + im*t248 t241 + im*t242 t233 + im*t234 t223 + im*t224 t211 + im*t212 t197 + im*t198 t181 + im*t182 t163 + im*t164 t143 + im*t144 t121 + im*t122 t97 + im*t98 t71 + im*t72 t43 + im*t44 t15 0;
   t255 + im*t256 t253 + im*t254 t249 + im*t250 t243 + im*t244 t235 + im*t236 t225 + im*t226 t213 + im*t214 t199 + im*t200 t183 + im*t184 t165 + im*t166 t145 + im*t146 t123 + im*t124 t99 + im*t100 t73 + im*t74 t45 + im*t46 t16];

pMd = [t1 t17 - im*t18 t47 - im*t48 t75 - im*t76 t101 - im*t102 t125 - im*t126 t147 - im*t148 t167 - im*t168 t185 - im*t186 t201 - im*t202 t215 - im*t216 t227 - im*t228 t237 - im*t238 t245 - im*t246 t251 - im*t252 t255 - im*t256;
    0 t2 t19 - im*t20 t49 - im*t50 t77 - im*t78 t103 - im*t104 t127 - im*t128 t149 - im*t150 t169 - im*t170 t187 - im*t188 t203 - im*t204 t217 - im*t218 t229 - im*t230 t239 - im*t240 t247 - im*t248 t253 - im*t254;
    0 0 t3 t21 - im*t22 t51 - im*t52 t79 - im*t80 t105 - im*t106 t129 - im*t130 t151 - im*t152 t171 - im*t172 t189 - im*t190 t205 - im*t206 t219 - im*t220 t231 - im*t232 t241 - im*t242 t249 - im*t250;
    0 0 0 t4 t23 - im*t24 t53 - im*t54 t81 - im*t82 t107 - im*t108 t131 - im*t132 t153 - im*t154 t173 - im*t174 t191 - im*t192 t207 - im*t208 t221 - im*t222 t233 - im*t234 t243 - im*t244;
    0 0 0 0 t5 t25 - im*t26 t55 - im*t56 t83 - im*t84 t109 - im*t110 t133 - im*t134 t155 - im*t156 t175 - im*t176 t193 - im*t194 t209 - im*t210 t223 - im*t224 t235 - im*t236;
    0 0 0 0 0 t6 t27 - im*t28 t57 - im*t58 t85 - im*t86 t111 - im*t112 t135 - im*t136 t157 - im*t158 t177 - im*t178 t195 - im*t196 t211 - im*t212 t225 - im*t226;
    0 0 0 0 0 0 t7 t29 - im*t30 t59 - im*t60 t87 - im*t88 t113 - im*t114 t137 - im*t138 t159 - im*t160 t179 - im*t180 t197 - im*t198 t213 - im*t214;
    0 0 0 0 0 0 0 t8 t31 - im*t32 t61 - im*t62 t89 - im*t90 t115 - im*t116 t139 - im*t140 t161 - im*t162 t181 - im*t182 t199 - im*t200;
    0 0 0 0 0 0 0 0 t9 t33 - im*t34 t63 - im*t64 t91 - im*t92 t117 - im*t118 t141 - im*t142 t163 - im*t164 t183 - im*t184;
    0 0 0 0 0 0 0 0 0 t10 t35 - im*t36 t65 - im*t66 t93 - im*t94 t119 - im*t120 t143 - im*t144 t165 - im*t166;
    0 0 0 0 0 0 0 0 0 0 t11 t37 - im*t38 t67 - im*t68 t95 - im*t96 t121 - im*t122 t145 - im*t146;
    0 0 0 0 0 0 0 0 0 0 0 t12 t39 - im*t40 t69 - im*t70 t97 - im*t98 t123 - im*t124;
    0 0 0 0 0 0 0 0 0 0 0 0 t13 t41 - im*t42 t71 - im*t72 t99 - im*t100;
    0 0 0 0 0 0 0 0 0 0 0 0 0 t14 t43 - im*t44 t73 - im*t74;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 t15 t45 - im*t46;
    0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 t16];

tvector = string.("t", collect(1:16^2));
=#
t=0;
@variable(model, (-10.0)<= t[1:256] <= (10.0));

@expression(model, begin
    
pM = @NLexpression(model, [t[1] 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0;
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
   t[255] + im*t[256] t[253] + im*t[254] t[249] + im*t[250] t[243] + im*t[244] t[235] + im*t[236] t[225] + im*t[226] t[213] + im*t[214] t[199] + im*t[200] t[183] + im*t[184] t[165] + im*t[166] t[145] + im*t[146] t[123] + im*t[124] t[99] + im*t[100] t[73] + im*t[74] t[45] + im*t[46] t[16]];)

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
end)
GeneralDM = pMd*pM;
Prediction(DM, State) = State' * DM * State;

#Here are the specific states and data measured in the tomography experiment

#test = readdlm("D:\\Binto\\ddd\\Julia\\Tomography_Mathematica_to_Julia\\measurement_bases_full_edit_measurement_7.txt", skipstart=2);
D2Array = readdlm("D:\\Binto\\ddd\\Julia\\Tomography_Mathematica_to_Julia\\measurement_bases_full_edit_measurement_7.txt", skipstart=2)[1:end,2:17];
Data = collect(flatten(D2Array));
SigmaData = sqrt.(Data);

#=StateMeasured = [kron(H, H, H, H), kron(H, H, H, V), kron(H, H, V, H), kron(H, H, V, V), kron(H, V, H, H), kron(H, V, H, V),
                kron(H, V, V, H), kron(H, V, V, V), kron(V, H, H, H), kron(V, H, H, V), kron(V, H, V, H), kron(V, H, V, V),
                kron(V, V, H, H), kron(V, V, H, V), kron(V, V, V, H), kron(V, V, V, V), kron(H, H, H, Dia), kron(H, H, H, A),
                kron(H, H, V, Dia), kron(H, H, V, A), kron(H, V, H, Dia), kron(H, V, H, A), kron(H, V, V, Dia), kron(H, V, V, A),
                kron(V, H, H, Dia), kron(V, H, H, A), kron(V, H, V, Dia), kron(V, H, V, A), kron(V, V, H, Dia), kron(V, V, H, A),
                kron(V, V, V, Dia), kron(V, V, V, A), kron(H, H, Dia, H), kron(H, H, Dia, V), kron(H, H, A, H), kron(H, H, A, V),
                kron(H, V, Dia, H), kron(H, V, Dia, V), kron(H, V, A, H), kron(H, V, A, V), kron(V, H, Dia, H), kron(V, H, Dia, V),
                kron(V, H, A, H), kron(V, H, A, V), kron(V, V, Dia, H), kron(V, V, Dia, V), kron(V, V, A, H), kron(V, V, A, V),
                kron(H, H, Dia, Dia), kron(H, H, Dia, A), kron(H, H, A, Dia), kron(H, H, A, A), kron(H, V, Dia, Dia), kron(H, V, Dia, A),
                kron(H, V, A, Dia), kron(H, V, A, A), kron(V, H, Dia, Dia), kron(V, H, Dia, A), kron(V, H, A, Dia), kron(V, H, A, A),
                kron(V, V, Dia, Dia), kron(V, V, Dia, A), kron(V, V, A, Dia), kron(V, V, A, A), kron(H, H, H, R), kron(H, H, H, L),
                kron(H, H, V, R), kron(H, H, V, L), kron(H, V, H, R), kron(H, V, H, L), kron(H, V, V, R), kron(H, V, V, L),
                kron(V, H, H, R), kron(V, H, H, L), kron(V, H, V, R), kron(V, H, V, L), kron(V, V, H, R), kron(V, V, H, L),
                kron(V, V, V, R), kron(V, V, V, L), kron(H, H, R, H), kron(H, H, R, V), kron(H, H, L, H), kron(H, H, L, V),
                kron(H, V, R, H), kron(H, V, R, V), kron(H, V, L, H), kron(H, V, L, V), kron(V, H, R, H), kron(V, H, R, V),
                kron(V, H, L, H), kron(V, H, L, V), kron(V, V, R, H), kron(V, V, R, V), kron(V, V, L, H), kron(V, V, L, V),
                kron(H, H, R, R), kron(H, H, R, L), kron(H, H, L, R), kron(H, H, L, L), kron(H, V, R, R), kron(H, V, R, L),
                kron(H, V, L, R), kron(H, V, L, L), kron(V, H, R, R), kron(V, H, R, L), kron(V, H, L, R), kron(V, H, L, L),
                kron(V, V, R, R), kron(V, V, R, L), kron(V, V, L, R), kron(V, V, L, L), kron(H, H, Dia, R), kron(H, H, Dia, L),
                kron(H, H, A, R), kron(H, H, A, L), kron(H, V, Dia, R), kron(H, V, Dia, L), kron(H, V, A, R), kron(H, V, A, L),
                kron(V, H, Dia, R), kron(V, H, Dia, L), kron(V, H, A, R), kron(V, H, A, L), kron(V, V, Dia, R), kron(V, V, Dia, L),
                kron(V, V, A, R), kron(V, V, A, L), kron(H, H, R, Dia), kron(H, H, R, A), kron(H, H, L, Dia), kron(H, H, L, A),
                kron(H, V, R, Dia), kron(H, V, R, A), kron(H, V, L, Dia), kron(H, V, L, A), kron(V, H, R, Dia), kron(V, H, R, A),
                kron(V, H, L, Dia), kron(V, H, L, A), kron(V, V, R, Dia), kron(V, V, R, A), kron(V, V, L, Dia), kron(V, V, L, A),
                kron(H, Dia, H, H), kron(H, Dia, H, V), kron(H, Dia, V, H), kron(H, Dia, V, V), kron(H, A, H, H), kron(H, A, H, V),
                kron(H, A, V, H), kron(H, A, V, V), kron(V, Dia, H, H), kron(V, Dia, H, V), kron(V, Dia, V, H), kron(V, Dia, V, V),
                kron(V, A, H, H), kron(V, A, H, V), kron(V, A, V, H), kron(V, A, V, V), kron(H, Dia, H, Dia), kron(H, Dia, H, A),
                kron(H, Dia, V, Dia), kron(H, Dia, V, A), kron(H, A, H, Dia), kron(H, A, H, A), kron(H, A, V, Dia), kron(H, A, V, A),
                kron(V, Dia, H, Dia), kron(V, Dia, H, A), kron(V, Dia, V, Dia), kron(V, Dia, V, A), kron(V, A, H, Dia), kron(V, A, H, A),
                kron(V, A, V, Dia), kron(V, A, V, A), kron(H, Dia, Dia, H), kron(H, Dia, Dia, V), kron(H, Dia, A, H), kron(H, Dia, A, V),
                kron(H, A, Dia, H), kron(H, A, Dia, V), kron(H, A, A, H), kron(H, A, A, V), kron(V, Dia, Dia, H), kron(V, Dia, Dia, V),
                kron(V, Dia, A, H), kron(V, Dia, A, V), kron(V, A, Dia, H), kron(V, A, Dia, V), kron(V, A, A, H), kron(V, A, A, V),
                kron(H, Dia, Dia, Dia), kron(H, Dia, Dia, A), kron(H, Dia, A, Dia), kron(H, Dia, A, A), kron(H, A, Dia, Dia), kron(H, A, Dia, A),
                kron(H, A, A, Dia), kron(H, A, A, A), kron(V, Dia, Dia, Dia), kron(V, Dia, Dia, A), kron(V, Dia, A, Dia), kron(V, Dia, A, A),
                kron(V, A, Dia, Dia), kron(V, A, Dia, A), kron(V, A, A, Dia), kron(V, A, A, A), kron(H, Dia, H, R), kron(H, Dia, H, L),
                kron(H, Dia, V, R), kron(H, Dia, V, L), kron(H, A, H, R), kron(H, A, H, L), kron(H, A, V, R), kron(H, A, V, L),
                kron(V, Dia, H, R), kron(V, Dia, H, L), kron(V, Dia, V, R), kron(V, Dia, V, L), kron(V, A, H, R), kron(V, A, H, L),
                kron(V, A, V, R), kron(V, A, V, L), kron(H, Dia, R, H), kron(H, Dia, R, V), kron(H, Dia, L, H), kron(H, Dia, L, V),
                kron(H, A, R, H), kron(H, A, R, V), kron(H, A, L, H), kron(H, A, L, V), kron(V, Dia, R, H), kron(V, Dia, R, V),
                kron(V, Dia, L, H), kron(V, Dia, L, V), kron(V, A, R, H), kron(V, A, R, V), kron(V, A, L, H), kron(V, A, L, V),
                kron(H, Dia, R, R), kron(H, Dia, R, L), kron(H, Dia, L, R), kron(H, Dia, L, L), kron(H, A, R, R), kron(H, A, R, L),
                kron(H, A, L, R), kron(H, A, L, L), kron(V, Dia, R, R), kron(V, Dia, R, L), kron(V, Dia, L, R), kron(V, Dia, L, L),
                kron(V, A, R, R), kron(V, A, R, L), kron(V, A, L, R), kron(V, A, L, L), kron(H, Dia, Dia, R), kron(H, Dia, Dia, L),
                kron(H, Dia, A, R), kron(H, Dia, A, L), kron(H, A, Dia, R), kron(H, A, Dia, L), kron(H, A, A, R), kron(H, A, A, L),
                kron(V, Dia, Dia, R), kron(V, Dia, Dia, L), kron(V, Dia, A, R), kron(V, Dia, A, L), kron(V, A, Dia, R), kron(V, A, Dia, L),
                kron(V, A, A, R), kron(V, A, A, L), kron(H, Dia, R, Dia), kron(H, Dia, R, A), kron(H, Dia, L, Dia), kron(H, Dia, L, A),
                kron(H, A, R, Dia), kron(H, A, R, A), kron(H, A, L, Dia), kron(H, A, L, A), kron(V, Dia, R, Dia), kron(V, Dia, R, A),
                kron(V, Dia, L, Dia), kron(V, Dia, L, A), kron(V, A, R, Dia), kron(V, A, R, A), kron(V, A, L, Dia), kron(V, A, L, A),
                kron(Dia, H, H, H), kron(Dia, H, H, V), kron(Dia, H, V, H), kron(Dia, H, V, V), kron(Dia, V, H, H), kron(Dia, V, H, V),
                kron(Dia, V, V, H), kron(Dia, V, V, V), kron(A, H, H, H), kron(A, H, H, V), kron(A, H, V, H), kron(A, H, V, V),
                kron(A, V, H, H), kron(A, V, H, V), kron(A, V, V, H), kron(A, V, V, V), kron(Dia, H, H, Dia), kron(Dia, H, H, A),
                kron(Dia, H, V, Dia), kron(Dia, H, V, A), kron(Dia, V, H, Dia), kron(Dia, V, H, A), kron(Dia, V, V, Dia), kron(Dia, V, V, A),
                kron(A, H, H, Dia), kron(A, H, H, A), kron(A, H, V, Dia), kron(A, H, V, A), kron(A, V, H, Dia), kron(A, V, H, A),
                kron(A, V, V, Dia), kron(A, V, V, A), kron(Dia, H, Dia, H), kron(Dia, H, Dia, V), kron(Dia, H, A, H), kron(Dia, H, A, V),
                kron(Dia, V, Dia, H), kron(Dia, V, Dia, V), kron(Dia, V, A, H), kron(Dia, V, A, V), kron(A, H, Dia, H), kron(A, H, Dia, V),
                kron(A, H, A, H), kron(A, H, A, V), kron(A, V, Dia, H), kron(A, V, Dia, V), kron(A, V, A, H), kron(A, V, A, V),
                kron(Dia, H, Dia, Dia), kron(Dia, H, Dia, A), kron(Dia, H, A, Dia), kron(Dia, H, A, A), kron(Dia, V, Dia, Dia), kron(Dia, V, Dia, A),
                kron(Dia, V, A, Dia), kron(Dia, V, A, A), kron(A, H, Dia, Dia), kron(A, H, Dia, A), kron(A, H, A, Dia), kron(A, H, A, A),
                kron(A, V, Dia, Dia), kron(A, V, Dia, A), kron(A, V, A, Dia), kron(A, V, A, A), kron(Dia, H, H, R), kron(Dia, H, H, L),
                kron(Dia, H, V, R), kron(Dia, H, V, L), kron(Dia, V, H, R), kron(Dia, V, H, L), kron(Dia, V, V, R), kron(Dia, V, V, L),
                kron(A, H, H, R), kron(A, H, H, L), kron(A, H, V, R), kron(A, H, V, L), kron(A, V, H, R), kron(A, V, H, L),
                kron(A, V, V, R), kron(A, V, V, L), kron(Dia, H, R, H), kron(Dia, H, R, V), kron(Dia, H, L, H), kron(Dia, H, L, V),
                kron(Dia, V, R, H), kron(Dia, V, R, V), kron(Dia, V, L, H), kron(Dia, V, L, V), kron(A, H, R, H), kron(A, H, R, V),
                kron(A, H, L, H), kron(A, H, L, V), kron(A, V, R, H), kron(A, V, R, V), kron(A, V, L, H), kron(A, V, L, V),
                kron(Dia, H, R, R), kron(Dia, H, R, L), kron(Dia, H, L, R), kron(Dia, H, L, L), kron(Dia, V, R, R), kron(Dia, V, R, L),
                kron(Dia, V, L, R), kron(Dia, V, L, L), kron(A, H, R, R), kron(A, H, R, L), kron(A, H, L, R), kron(A, H, L, L),
                kron(A, V, R, R), kron(A, V, R, L), kron(A, V, L, R), kron(A, V, L, L), kron(Dia, H, Dia, R), kron(Dia, H, Dia, L),
                kron(Dia, H, A, R), kron(Dia, H, A, L), kron(Dia, V, Dia, R), kron(Dia, V, Dia, L), kron(Dia, V, A, R), kron(Dia, V, A, L),
                kron(A, H, Dia, R), kron(A, H, Dia, L), kron(A, H, A, R), kron(A, H, A, L), kron(A, V, Dia, R), kron(A, V, Dia, L),
                kron(A, V, A, R), kron(A, V, A, L), kron(Dia, H, R, Dia), kron(Dia, H, R, A), kron(Dia, H, L, Dia), kron(Dia, H, L, A),
                kron(Dia, V, R, Dia), kron(Dia, V, R, A), kron(Dia, V, L, Dia), kron(Dia, V, L, A), kron(A, H, R, Dia), kron(A, H, R, A),
                kron(A, H, L, Dia), kron(A, H, L, A), kron(A, V, R, Dia), kron(A, V, R, A), kron(A, V, L, Dia), kron(A, V, L, A),
                kron(Dia, Dia, H, H), kron(Dia, Dia, H, V), kron(Dia, Dia, V, H), kron(Dia, Dia, V, V), kron(Dia, A, H, H), kron(Dia, A, H, V),
                kron(Dia, A, V, H), kron(Dia, A, V, V), kron(A, Dia, H, H), kron(A, Dia, H, V), kron(A, Dia, V, H), kron(A, Dia, V, V),
                kron(A, A, H, H), kron(A, A, H, V), kron(A, A, V, H), kron(A, A, V, V), kron(Dia, Dia, H, Dia), kron(Dia, Dia, H, A),
                kron(Dia, Dia, V, Dia), kron(Dia, Dia, V, A), kron(Dia, A, H, Dia), kron(Dia, A, H, A), kron(Dia, A, V, Dia), kron(Dia, A, V, A),
                kron(A, Dia, H, Dia), kron(A, Dia, H, A), kron(A, Dia, V, Dia), kron(A, Dia, V, A), kron(A, A, H, Dia), kron(A, A, H, A),
                kron(A, A, V, Dia), kron(A, A, V, A), kron(Dia, Dia, Dia, H), kron(Dia, Dia, Dia, V), kron(Dia, Dia, A, H), kron(Dia, Dia, A, V),
                kron(Dia, A, Dia, H), kron(Dia, A, Dia, V), kron(Dia, A, A, H), kron(Dia, A, A, V), kron(A, Dia, Dia, H), kron(A, Dia, Dia, V),
                kron(A, Dia, A, H), kron(A, Dia, A, V), kron(A, A, Dia, H), kron(A, A, Dia, V), kron(A, A, A, H), kron(A, A, A, V),
                kron(Dia, Dia, Dia, Dia), kron(Dia, Dia, Dia, A), kron(Dia, Dia, A, Dia), kron(Dia, Dia, A, A), kron(Dia, A, Dia, Dia),
                kron(Dia, A, Dia, A), kron(Dia, A, A, Dia), kron(Dia, A, A, A), kron(A, Dia, Dia, Dia), kron(A, Dia, Dia, A), kron(A, Dia, A, Dia),
                kron(A, Dia, A, A), kron(A, A, Dia, Dia), kron(A, A, Dia, A), kron(A, A, A, Dia), kron(A, A, A, A), kron(Dia, Dia, H, R),
                kron(Dia, Dia, H, L), kron(Dia, Dia, V, R), kron(Dia, Dia, V, L), kron(Dia, A, H, R), kron(Dia, A, H, L), kron(Dia, A, V, R),
                kron(Dia, A, V, L), kron(A, Dia, H, R), kron(A, Dia, H, L), kron(A, Dia, V, R), kron(A, Dia, V, L), kron(A, A, H, R), kron(A, A, H, L),
                kron(A, A, V, R), kron(A, A, V, L), kron(Dia, Dia, R, H), kron(Dia, Dia, R, V), kron(Dia, Dia, L, H), kron(Dia, Dia, L, V), kron(Dia, A, R, H),
                kron(Dia, A, R, V), kron(Dia, A, L, H), kron(Dia, A, L, V), kron(A, Dia, R, H), kron(A, Dia, R, V), kron(A, Dia, L, H),
                kron(A, Dia, L, V), kron(A, A, R, H), kron(A, A, R, V), kron(A, A, L, H), kron(A, A, L, V), kron(Dia, Dia, R, R),
                kron(Dia, Dia, R, L), kron(Dia, Dia, L, R), kron(Dia, Dia, L, L), kron(Dia, A, R, R), kron(Dia, A, R, L), kron(Dia, A, L, R),
                kron(Dia, A, L, L), kron(A, Dia, R, R), kron(A, Dia, R, L), kron(A, Dia, L, R), kron(A, Dia, L, L), kron(A, A, R, R),
                kron(A, A, R, L), kron(A, A, L, R), kron(A, A, L, L), kron(Dia, Dia, Dia, R), kron(Dia, Dia, Dia, L), kron(Dia, Dia, A, R),
                kron(Dia, Dia, A, L), kron(Dia, A, Dia, R), kron(Dia, A, Dia, L), kron(Dia, A, A, R), kron(Dia, A, A, L), 
                kron(A, Dia, Dia, R), kron(A, Dia, Dia, L), kron(A, Dia, A, R), kron(A, Dia, A, L), kron(A, A, Dia, R), kron(A, A, Dia, L), 
                kron(A, A, A, R), kron(A, A, A, L), kron(Dia, Dia, R, Dia), kron(Dia, Dia, R, A), kron(Dia, Dia, L, Dia), kron(Dia, Dia, L, A),
                kron(Dia, A, R, Dia), kron(Dia, A, R, A), kron(Dia, A, L, Dia), kron(Dia, A, L, A), kron(A, Dia, R, Dia), kron(A, Dia, R, A), 
                kron(A, Dia, L, Dia), kron(A, Dia, L, A), kron(A, A, R, Dia), kron(A, A, R, A), kron(A, A, L, Dia), kron(A, A, L, A), 
                kron(H, R, H, H), kron(H, R, H, V), kron(H, R, V, H), kron(H, R, V, V), kron(H, L, H, H), kron(H, L, H, V), 
                kron(H, L, V, H), kron(H, L, V, V), kron(V, R, H, H), kron(V, R, H, V), kron(V, R, V, H), kron(V, R, V, V), 
                kron(V, L, H, H), kron(V, L, H, V), kron(V, L, V, H), kron(V, L, V, V), kron(H, R, H, Dia), kron(H, R, H, A), 
                kron(H, R, V, Dia), kron(H, R, V, A), kron(H, L, H, Dia), kron(H, L, H, A), kron(H, L, V, Dia), kron(H, L, V, A), 
                kron(V, R, H, Dia), kron(V, R, H, A), kron(V, R, V, Dia), kron(V, R, V, A), kron(V, L, H, Dia), kron(V, L, H, A), 
                kron(V, L, V, Dia), kron(V, L, V, A), kron(H, R, Dia, H), kron(H, R, Dia, V), kron(H, R, A, H), kron(H, R, A, V), 
                kron(H, L, Dia, H), kron(H, L, Dia, V), kron(H, L, A, H), kron(H, L, A, V), kron(V, R, Dia, H), kron(V, R, Dia, V), 
                kron(V, R, A, H), kron(V, R, A, V), kron(V, L, Dia, H), kron(V, L, Dia, V), kron(V, L, A, H), kron(V, L, A, V), 
                kron(H, R, Dia, Dia), kron(H, R, Dia, A), kron(H, R, A, Dia), kron(H, R, A, A), kron(H, L, Dia, Dia), kron(H, L, Dia, A), 
                kron(H, L, A, Dia), kron(H, L, A, A), kron(V, R, Dia, Dia), kron(V, R, Dia, A), kron(V, R, A, Dia), kron(V, R, A, A), 
                kron(V, L, Dia, Dia), kron(V, L, Dia, A), kron(V, L, A, Dia), kron(V, L, A, A), kron(H, R, H, R), kron(H, R, H, L), 
                kron(H, R, V, R), kron(H, R, V, L), kron(H, L, H, R), kron(H, L, H, L), kron(H, L, V, R), kron(H, L, V, L), 
                kron(V, R, H, R), kron(V, R, H, L), kron(V, R, V, R), kron(V, R, V, L), kron(V, L, H, R), kron(V, L, H, L), 
                kron(V, L, V, R), kron(V, L, V, L), kron(H, R, R, H), kron(H, R, R, V), kron(H, R, L, H), kron(H, R, L, V), 
                kron(H, L, R, H), kron(H, L, R, V), kron(H, L, L, H), kron(H, L, L, V), kron(V, R, R, H), kron(V, R, R, V), 
                kron(V, R, L, H), kron(V, R, L, V), kron(V, L, R, H), kron(V, L, R, V), kron(V, L, L, H), kron(V, L, L, V), 
                kron(H, R, R, R), kron(H, R, R, L), kron(H, R, L, R), kron(H, R, L, L), kron(H, L, R, R), kron(H, L, R, L), 
                kron(H, L, L, R), kron(H, L, L, L), kron(V, R, R, R), kron(V, R, R, L), kron(V, R, L, R), kron(V, R, L, L), 
                kron(V, L, R, R), kron(V, L, R, L), kron(V, L, L, R), kron(V, L, L, L), kron(H, R, Dia, R), kron(H, R, Dia, L), 
                kron(H, R, A, R), kron(H, R, A, L), kron(H, L, Dia, R), kron(H, L, Dia, L), kron(H, L, A, R), kron(H, L, A, L), 
                kron(V, R, Dia, R), kron(V, R, Dia, L), kron(V, R, A, R), kron(V, R, A, L), kron(V, L, Dia, R), kron(V, L, Dia, L), 
                kron(V, L, A, R), kron(V, L, A, L), kron(H, R, R, Dia), kron(H, R, R, A), kron(H, R, L, Dia), kron(H, R, L, A), 
                kron(H, L, R, Dia), kron(H, L, R, A), kron(H, L, L, Dia), kron(H, L, L, A), kron(V, R, R, Dia), kron(V, R, R, A), 
                kron(V, R, L, Dia), kron(V, R, L, A), kron(V, L, R, Dia), kron(V, L, R, A), kron(V, L, L, Dia), kron(V, L, L, A), 
                kron(R, H, H, H), kron(R, H, H, V), kron(R, H, V, H), kron(R, H, V, V), kron(R, V, H, H), kron(R, V, H, V), 
                kron(R, V, V, H), kron(R, V, V, V), kron(L, H, H, H), kron(L, H, H, V), kron(L, H, V, H), kron(L, H, V, V), 
                kron(L, V, H, H), kron(L, V, H, V), kron(L, V, V, H), kron(L, V, V, V), kron(R, H, H, Dia), kron(R, H, H, A), 
                kron(R, H, V, Dia), kron(R, H, V, A), kron(R, V, H, Dia), kron(R, V, H, A), kron(R, V, V, Dia), kron(R, V, V, A), 
                kron(L, H, H, Dia), kron(L, H, H, A), kron(L, H, V, Dia), kron(L, H, V, A), kron(L, V, H, Dia), kron(L, V, H, A), 
                kron(L, V, V, Dia), kron(L, V, V, A), kron(R, H, Dia, H), kron(R, H, Dia, V), kron(R, H, A, H), kron(R, H, A, V), 
                kron(R, V, Dia, H), kron(R, V, Dia, V), kron(R, V, A, H), kron(R, V, A, V), kron(L, H, Dia, H), kron(L, H, Dia, V), 
                kron(L, H, A, H), kron(L, H, A, V), kron(L, V, Dia, H), kron(L, V, Dia, V), kron(L, V, A, H), kron(L, V, A, V), 
                kron(R, H, Dia, Dia), kron(R, H, Dia, A), kron(R, H, A, Dia), kron(R, H, A, A), kron(R, V, Dia, Dia), kron(R, V, Dia, A), 
                kron(R, V, A, Dia), kron(R, V, A, A), kron(L, H, Dia, Dia), kron(L, H, Dia, A), kron(L, H, A, Dia), kron(L, H, A, A), 
                kron(L, V, Dia, Dia), kron(L, V, Dia, A), kron(L, V, A, Dia), kron(L, V, A, A), kron(R, H, H, R), kron(R, H, H, L), 
                kron(R, H, V, R), kron(R, H, V, L), kron(R, V, H, R), kron(R, V, H, L), kron(R, V, V, R), kron(R, V, V, L), 
                kron(L, H, H, R), kron(L, H, H, L), kron(L, H, V, R), kron(L, H, V, L), kron(L, V, H, R), kron(L, V, H, L), 
                kron(L, V, V, R), kron(L, V, V, L), kron(R, H, R, H), kron(R, H, R, V), kron(R, H, L, H), kron(R, H, L, V), 
                kron(R, V, R, H), kron(R, V, R, V), kron(R, V, L, H), kron(R, V, L, V), kron(L, H, R, H), kron(L, H, R, V), 
                kron(L, H, L, H), kron(L, H, L, V), kron(L, V, R, H), kron(L, V, R, V), kron(L, V, L, H), kron(L, V, L, V), 
                kron(R, H, R, R), kron(R, H, R, L), kron(R, H, L, R), kron(R, H, L, L), kron(R, V, R, R), kron(R, V, R, L), 
                kron(R, V, L, R), kron(R, V, L, L), kron(L, H, R, R), kron(L, H, R, L), kron(L, H, L, R), kron(L, H, L, L), 
                kron(L, V, R, R), kron(L, V, R, L), kron(L, V, L, R), kron(L, V, L, L), kron(R, H, Dia, R), kron(R, H, Dia, L), 
                kron(R, H, A, R), kron(R, H, A, L), kron(R, V, Dia, R), kron(R, V, Dia, L), kron(R, V, A, R), kron(R, V, A, L), 
                kron(L, H, Dia, R), kron(L, H, Dia, L), kron(L, H, A, R), kron(L, H, A, L), kron(L, V, Dia, R), kron(L, V, Dia, L), 
                kron(L, V, A, R), kron(L, V, A, L), kron(R, H, R, Dia), kron(R, H, R, A), kron(R, H, L, Dia), kron(R, H, L, A), 
                kron(R, V, R, Dia), kron(R, V, R, A), kron(R, V, L, Dia), kron(R, V, L, A), kron(L, H, R, Dia), kron(L, H, R, A), 
                kron(L, H, L, Dia), kron(L, H, L, A), kron(L, V, R, Dia), kron(L, V, R, A), kron(L, V, L, Dia), kron(L, V, L, A), 
                kron(R, R, H, H), kron(R, R, H, V), kron(R, R, V, H), kron(R, R, V, V), kron(R, L, H, H), kron(R, L, H, V), 
                kron(R, L, V, H), kron(R, L, V, V), kron(L, R, H, H), kron(L, R, H, V), kron(L, R, V, H), kron(L, R, V, V), 
                kron(L, L, H, H), kron(L, L, H, V), kron(L, L, V, H), kron(L, L, V, V), kron(R, R, H, Dia), kron(R, R, H, A), 
                kron(R, R, V, Dia), kron(R, R, V, A), kron(R, L, H, Dia), kron(R, L, H, A), kron(R, L, V, Dia), kron(R, L, V, A), 
                kron(L, R, H, Dia), kron(L, R, H, A), kron(L, R, V, Dia), kron(L, R, V, A), kron(L, L, H, Dia), kron(L, L, H, A), 
                kron(L, L, V, Dia), kron(L, L, V, A), kron(R, R, Dia, H), kron(R, R, Dia, V), kron(R, R, A, H), kron(R, R, A, V), 
                kron(R, L, Dia, H), kron(R, L, Dia, V), kron(R, L, A, H), kron(R, L, A, V), kron(L, R, Dia, H), kron(L, R, Dia, V), 
                kron(L, R, A, H), kron(L, R, A, V), kron(L, L, Dia, H), kron(L, L, Dia, V), kron(L, L, A, H), kron(L, L, A, V), 
                kron(R, R, Dia, Dia), kron(R, R, Dia, A), kron(R, R, A, Dia), kron(R, R, A, A), kron(R, L, Dia, Dia), kron(R, L, Dia, A), 
                kron(R, L, A, Dia), kron(R, L, A, A), kron(L, R, Dia, Dia), kron(L, R, Dia, A), kron(L, R, A, Dia), kron(L, R, A, A), 
                kron(L, L, Dia, Dia), kron(L, L, Dia, A), kron(L, L, A, Dia), kron(L, L, A, A), kron(R, R, H, R), kron(R, R, H, L), 
                kron(R, R, V, R), kron(R, R, V, L), kron(R, L, H, R), kron(R, L, H, L), kron(R, L, V, R), kron(R, L, V, L), 
                kron(L, R, H, R), kron(L, R, H, L), kron(L, R, V, R), kron(L, R, V, L), kron(L, L, H, R), kron(L, L, H, L), 
                kron(L, L, V, R), kron(L, L, V, L), kron(R, R, R, H), kron(R, R, R, V), kron(R, R, L, H), kron(R, R, L, V), 
                kron(R, L, R, H), kron(R, L, R, V), kron(R, L, L, H), kron(R, L, L, V), kron(L, R, R, H), kron(L, R, R, V), 
                kron(L, R, L, H), kron(L, R, L, V), kron(L, L, R, H), kron(L, L, R, V), kron(L, L, L, H), kron(L, L, L, V), 
                kron(R, R, R, R), kron(R, R, R, L), kron(R, R, L, R), kron(R, R, L, L), kron(R, L, R, R), kron(R, L, R, L), 
                kron(R, L, L, R), kron(R, L, L, L), kron(L, R, R, R), kron(L, R, R, L), kron(L, R, L, R), kron(L, R, L, L), 
                kron(L, L, R, R), kron(L, L, R, L), kron(L, L, L, R), kron(L, L, L, L), kron(R, R, Dia, R), kron(R, R, Dia, L), 
                kron(R, R, A, R), kron(R, R, A, L), kron(R, L, Dia, R), kron(R, L, Dia, L), kron(R, L, A, R), kron(R, L, A, L), 
                kron(L, R, Dia, R), kron(L, R, Dia, L), kron(L, R, A, R), kron(L, R, A, L), kron(L, L, Dia, R), kron(L, L, Dia, L), 
                kron(L, L, A, R), kron(L, L, A, L), kron(R, R, R, Dia), kron(R, R, R, A), kron(R, R, L, Dia), kron(R, R, L, A), 
                kron(R, L, R, Dia), kron(R, L, R, A), kron(R, L, L, Dia), kron(R, L, L, A), kron(L, R, R, Dia), kron(L, R, R, A), 
                kron(L, R, L, Dia), kron(L, R, L, A), kron(L, L, R, Dia), kron(L, L, R, A), kron(L, L, L, Dia), kron(L, L, L, A), 
                kron(Dia, R, H, H), kron(Dia, R, H, V), kron(Dia, R, V, H), kron(Dia, R, V, V), kron(Dia, L, H, H), kron(Dia, L, H, V), 
                kron(Dia, L, V, H), kron(Dia, L, V, V), kron(A, R, H, H), kron(A, R, H, V), kron(A, R, V, H), kron(A, R, V, V), 
                kron(A, L, H, H), kron(A, L, H, V), kron(A, L, V, H), kron(A, L, V, V), kron(Dia, R, H, Dia), kron(Dia, R, H, A), 
                kron(Dia, R, V, Dia), kron(Dia, R, V, A), kron(Dia, L, H, Dia), kron(Dia, L, H, A), kron(Dia, L, V, Dia), kron(Dia, L, V, A), 
                kron(A, R, H, Dia), kron(A, R, H, A), kron(A, R, V, Dia), kron(A, R, V, A), kron(A, L, H, Dia), kron(A, L, H, A), 
                kron(A, L, V, Dia), kron(A, L, V, A), kron(Dia, R, Dia, H), kron(Dia, R, Dia, V), kron(Dia, R, A, H), kron(Dia, R, A, V), 
                kron(Dia, L, Dia, H), kron(Dia, L, Dia, V), kron(Dia, L, A, H), kron(Dia, L, A, V), kron(A, R, Dia, H), kron(A, R, Dia, V), 
                kron(A, R, A, H), kron(A, R, A, V), kron(A, L, Dia, H), kron(A, L, Dia, V), kron(A, L, A, H), kron(A, L, A, V), 
                kron(Dia, R, Dia, Dia), kron(Dia, R, Dia, A), kron(Dia, R, A, Dia), kron(Dia, R, A, A), kron(Dia, L, Dia, Dia), kron(Dia, L, Dia, A), 
                kron(Dia, L, A, Dia), kron(Dia, L, A, A), kron(A, R, Dia, Dia), kron(A, R, Dia, A), kron(A, R, A, Dia), kron(A, R, A, A), 
                kron(A, L, Dia, Dia), kron(A, L, Dia, A), kron(A, L, A, Dia), kron(A, L, A, A), kron(Dia, R, H, R), kron(Dia, R, H, L), 
                kron(Dia, R, V, R), kron(Dia, R, V, L), kron(Dia, L, H, R), kron(Dia, L, H, L), kron(Dia, L, V, R), kron(Dia, L, V, L), 
                kron(A, R, H, R), kron(A, R, H, L), kron(A, R, V, R), kron(A, R, V, L), kron(A, L, H, R), kron(A, L, H, L), 
                kron(A, L, V, R), kron(A, L, V, L), kron(Dia, R, R, H), kron(Dia, R, R, V), kron(Dia, R, L, H), kron(Dia, R, L, V), 
                kron(Dia, L, R, H), kron(Dia, L, R, V), kron(Dia, L, L, H), kron(Dia, L, L, V), kron(A, R, R, H), kron(A, R, R, V), 
                kron(A, R, L, H), kron(A, R, L, V), kron(A, L, R, H), kron(A, L, R, V), kron(A, L, L, H), kron(A, L, L, V), 
                kron(Dia, R, R, R), kron(Dia, R, R, L), kron(Dia, R, L, R), kron(Dia, R, L, L), kron(Dia, L, R, R), kron(Dia, L, R, L), 
                kron(Dia, L, L, R), kron(Dia, L, L, L), kron(A, R, R, R), kron(A, R, R, L), kron(A, R, L, R), kron(A, R, L, L), 
                kron(A, L, R, R), kron(A, L, R, L), kron(A, L, L, R), kron(A, L, L, L), kron(Dia, R, Dia, R), kron(Dia, R, Dia, L), 
                kron(Dia, R, A, R), kron(Dia, R, A, L), kron(Dia, L, Dia, R), kron(Dia, L, Dia, L), kron(Dia, L, A, R), kron(Dia, L, A, L), 
                kron(A, R, Dia, R), kron(A, R, Dia, L), kron(A, R, A, R), kron(A, R, A, L), kron(A, L, Dia, R), kron(A, L, Dia, L), 
                kron(A, L, A, R), kron(A, L, A, L), kron(Dia, R, R, Dia), kron(Dia, R, R, A), kron(Dia, R, L, Dia), kron(Dia, R, L, A), 
                kron(Dia, L, R, Dia), kron(Dia, L, R, A), kron(Dia, L, L, Dia), kron(Dia, L, L, A), kron(A, R, R, Dia), kron(A, R, R, A), 
                kron(A, R, L, Dia), kron(A, R, L, A), kron(A, L, R, Dia), kron(A, L, R, A), kron(A, L, L, Dia), kron(A, L, L, A), 
                kron(R, Dia, H, H), kron(R, Dia, H, V), kron(R, Dia, V, H), kron(R, Dia, V, V), kron(R, A, H, H), kron(R, A, H, V), 
                kron(R, A, V, H), kron(R, A, V, V), kron(L, Dia, H, H), kron(L, Dia, H, V), kron(L, Dia, V, H), kron(L, Dia, V, V), 
                kron(L, A, H, H), kron(L, A, H, V), kron(L, A, V, H), kron(L, A, V, V), kron(R, Dia, H, Dia), kron(R, Dia, H, A), 
                kron(R, Dia, V, Dia), kron(R, Dia, V, A), kron(R, A, H, Dia), kron(R, A, H, A), kron(R, A, V, Dia), kron(R, A, V, A), 
                kron(L, Dia, H, Dia), kron(L, Dia, H, A), kron(L, Dia, V, Dia), kron(L, Dia, V, A), kron(L, A, H, Dia), kron(L, A, H, A), 
                kron(L, A, V, Dia), kron(L, A, V, A), kron(R, Dia, Dia, H), kron(R, Dia, Dia, V), kron(R, Dia, A, H), kron(R, Dia, A, V), 
                kron(R, A, Dia, H), kron(R, A, Dia, V), kron(R, A, A, H), kron(R, A, A, V), kron(L, Dia, Dia, H), kron(L, Dia, Dia, V), 
                kron(L, Dia, A, H), kron(L, Dia, A, V), kron(L, A, Dia, H), kron(L, A, Dia, V), kron(L, A, A, H), kron(L, A, A, V), 
                kron(R, Dia, Dia, Dia), kron(R, Dia, Dia, A), kron(R, Dia, A, Dia), kron(R, Dia, A, A), kron(R, A, Dia, Dia), kron(R, A, Dia, A), 
                kron(R, A, A, Dia), kron(R, A, A, A), kron(L, Dia, Dia, Dia), kron(L, Dia, Dia, A), kron(L, Dia, A, Dia), kron(L, Dia, A, A), 
                kron(L, A, Dia, Dia), kron(L, A, Dia, A), kron(L, A, A, Dia), kron(L, A, A, A), kron(R, Dia, H, R), kron(R, Dia, H, L), 
                kron(R, Dia, V, R), kron(R, Dia, V, L), kron(R, A, H, R), kron(R, A, H, L), kron(R, A, V, R), kron(R, A, V, L), 
                kron(L, Dia, H, R), kron(L, Dia, H, L), kron(L, Dia, V, R), kron(L, Dia, V, L), kron(L, A, H, R), kron(L, A, H, L), 
                kron(L, A, V, R), kron(L, A, V, L), kron(R, Dia, R, H), kron(R, Dia, R, V), kron(R, Dia, L, H), kron(R, Dia, L, V), 
                kron(R, A, R, H), kron(R, A, R, V), kron(R, A, L, H), kron(R, A, L, V), kron(L, Dia, R, H), kron(L, Dia, R, V), 
                kron(L, Dia, L, H), kron(L, Dia, L, V), kron(L, A, R, H), kron(L, A, R, V), kron(L, A, L, H), kron(L, A, L, V), 
                kron(R, Dia, R, R), kron(R, Dia, R, L), kron(R, Dia, L, R), kron(R, Dia, L, L), kron(R, A, R, R), kron(R, A, R, L), 
                kron(R, A, L, R), kron(R, A, L, L), kron(L, Dia, R, R), kron(L, Dia, R, L), kron(L, Dia, L, R), kron(L, Dia, L, L), 
                kron(L, A, R, R), kron(L, A, R, L), kron(L, A, L, R), kron(L, A, L, L), kron(R, Dia, Dia, R), kron(R, Dia, Dia, L), 
                kron(R, Dia, A, R), kron(R, Dia, A, L), kron(R, A, Dia, R), kron(R, A, Dia, L), kron(R, A, A, R), kron(R, A, A, L), 
                kron(L, Dia, Dia, R), kron(L, Dia, Dia, L), kron(L, Dia, A, R), kron(L, Dia, A, L), kron(L, A, Dia, R), kron(L, A, Dia, L), 
                kron(L, A, A, R), kron(L, A, A, L), kron(R, Dia, R, Dia), kron(R, Dia, R, A), kron(R, Dia, L, Dia), kron(R, Dia, L, A), 
                kron(R, A, R, Dia), kron(R, A, R, A), kron(R, A, L, Dia), kron(R, A, L, A), kron(L, Dia, R, Dia), kron(L, Dia, R, A), 
                kron(L, Dia, L, Dia), kron(L, Dia, L, A), kron(L, A, R, Dia), kron(L, A, R, A), kron(L, A, L, Dia), kron(L, A, L, A)];=#
statevars = [H, V, Dia, A, R, L];
StateMeasured = [kron(i,j,k,l) for i in statevars, j in statevars, k in statevars, l in statevars];

PredictionPart = [];
@time for i = 1:length(StateMeasured)
    push!(PredictionPart, Prediction(GeneralDM, StateMeasured[i]));
end

BadnessPolynomial = sum((PredictionPart[i] - Data[i])^2 /PredictionPart[i] for i=1:length(StateMeasured));

temp = [];
