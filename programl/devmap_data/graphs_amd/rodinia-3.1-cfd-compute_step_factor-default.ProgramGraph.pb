

[external]
@allocaB6
4
	full_text'
%
#%5 = alloca %struct.FLOAT3, align 8
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #4
4truncB+
)
	full_text

%7 = trunc i64 %6 to i32
"i64B

	full_text


i64 %6
3icmpB+
)
	full_text

%8 = icmp slt i32 %7, %3
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %51
 i1B

	full_text	

i1 %8
0shl8B'
%
	full_text

%10 = shl i64 %6, 32
$i648B

	full_text


i64 %6
9ashr8B/
-
	full_text 

%11 = ashr exact i64 %10, 32
%i648B

	full_text
	
i64 %10
\getelementptr8BI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %0, i64 %11
%i648B

	full_text
	
i64 %11
Lload8BB
@
	full_text3
1
/%13 = load float, float* %12, align 4, !tbaa !8
+float*8B

	full_text


float* %12
4add8B+
)
	full_text

%14 = add nsw i32 %7, %3
$i328B

	full_text


i32 %7
6sext8B,
*
	full_text

%15 = sext i32 %14 to i64
%i328B

	full_text
	
i32 %14
\getelementptr8BI
G
	full_text:
8
6%16 = getelementptr inbounds float, float* %0, i64 %15
%i648B

	full_text
	
i64 %15
Lload8BB
@
	full_text3
1
/%17 = load float, float* %16, align 4, !tbaa !8
+float*8B

	full_text


float* %16
]insertelement8BJ
H
	full_text;
9
7%18 = insertelement <2 x float> undef, float %17, i32 0
)float8B

	full_text

	float %17
/shl8B&
$
	full_text

%19 = shl i32 %3, 1
5add8B,
*
	full_text

%20 = add nsw i32 %19, %7
%i328B

	full_text
	
i32 %19
$i328B

	full_text


i32 %7
6sext8B,
*
	full_text

%21 = sext i32 %20 to i64
%i328B

	full_text
	
i32 %20
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %0, i64 %21
%i648B

	full_text
	
i64 %21
Lload8BB
@
	full_text3
1
/%23 = load float, float* %22, align 4, !tbaa !8
+float*8B

	full_text


float* %22
[insertelement8BH
F
	full_text9
7
5%24 = insertelement <2 x float> %18, float %23, i32 1
5<2 x float>8B"
 
	full_text

<2 x float> %18
)float8B

	full_text

	float %23
3mul8B*
(
	full_text

%25 = mul nsw i32 %3, 3
5add8B,
*
	full_text

%26 = add nsw i32 %25, %7
%i328B

	full_text
	
i32 %25
$i328B

	full_text


i32 %7
6sext8B,
*
	full_text

%27 = sext i32 %26 to i64
%i328B

	full_text
	
i32 %26
\getelementptr8BI
G
	full_text:
8
6%28 = getelementptr inbounds float, float* %0, i64 %27
%i648B

	full_text
	
i64 %27
Lload8BB
@
	full_text3
1
/%29 = load float, float* %28, align 4, !tbaa !8
+float*8B

	full_text


float* %28
/shl8B&
$
	full_text

%30 = shl i32 %3, 2
5add8B,
*
	full_text

%31 = add nsw i32 %30, %7
%i328B

	full_text
	
i32 %30
$i328B

	full_text


i32 %7
6sext8B,
*
	full_text

%32 = sext i32 %31 to i64
%i328B

	full_text
	
i32 %31
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %0, i64 %32
%i648B

	full_text
	
i64 %32
Lload8BB
@
	full_text3
1
/%34 = load float, float* %33, align 4, !tbaa !8
+float*8B

	full_text


float* %33
Gbitcast8B:
8
	full_text+
)
'%35 = bitcast %struct.FLOAT3* %5 to i8*
,struct*8B

	full_text


struct* %5
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 12, i8* nonnull %35) #5
%i8*8B

	full_text
	
i8* %35
~call8Bt
r
	full_texte
c
acall void @compute_velocity(float %13, <2 x float> %24, float %29, %struct.FLOAT3* nonnull %5) #5
)float8B

	full_text

	float %13
5<2 x float>8B"
 
	full_text

<2 x float> %24
)float8B

	full_text

	float %29
,struct*8B

	full_text


struct* %5
Pbitcast8BC
A
	full_text4
2
0%36 = bitcast %struct.FLOAT3* %5 to <2 x float>*
,struct*8B

	full_text


struct* %5
Nload8BD
B
	full_text5
3
1%37 = load <2 x float>, <2 x float>* %36, align 8
7<2 x float>*8B#
!
	full_text

<2 x float>* %36
sgetelementptr8B`
^
	full_textQ
O
M%38 = getelementptr inbounds %struct.FLOAT3, %struct.FLOAT3* %5, i64 0, i32 2
,struct*8B

	full_text


struct* %5
Bload8B8
6
	full_text)
'
%%39 = load float, float* %38, align 8
+float*8B

	full_text


float* %38
_call8BU
S
	full_textF
D
B%40 = call float @compute_speed_sqd(<2 x float> %37, float %39) #5
5<2 x float>8B"
 
	full_text

<2 x float> %37
)float8B

	full_text

	float %39
ccall8BY
W
	full_textJ
H
F%41 = call float @compute_pressure(float %13, float %34, float %40) #5
)float8B

	full_text

	float %13
)float8B

	full_text

	float %34
)float8B

	full_text

	float %40
^call8BT
R
	full_textE
C
A%42 = call float @compute_speed_of_sound(float %13, float %41) #5
)float8B

	full_text

	float %13
)float8B

	full_text

	float %41
\getelementptr8BI
G
	full_text:
8
6%43 = getelementptr inbounds float, float* %1, i64 %11
%i648B

	full_text
	
i64 %11
Lload8BB
@
	full_text3
1
/%44 = load float, float* %43, align 4, !tbaa !8
+float*8B

	full_text


float* %43
Ecall8B;
9
	full_text,
*
(%45 = call float @_Z4sqrtf(float %44) #4
)float8B

	full_text

	float %44
Ecall8B;
9
	full_text,
*
(%46 = call float @_Z4sqrtf(float %40) #4
)float8B

	full_text

	float %40
6fadd8B,
*
	full_text

%47 = fadd float %42, %46
)float8B

	full_text

	float %42
)float8B

	full_text

	float %46
6fmul8B,
*
	full_text

%48 = fmul float %45, %47
)float8B

	full_text

	float %45
)float8B

	full_text

	float %47
Lfdiv8BB
@
	full_text3
1
/%49 = fdiv float 5.000000e-01, %48, !fpmath !12
)float8B

	full_text

	float %48
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %2, i64 %11
%i648B

	full_text
	
i64 %11
Lstore8BA
?
	full_text2
0
.store float %49, float* %50, align 4, !tbaa !8
)float8B

	full_text

	float %49
+float*8B

	full_text


float* %50
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 12, i8* nonnull %35) #5
%i8*8B

	full_text
	
i8* %35
'br8B

	full_text

br label %51
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
$i648B

	full_text


i64 32
7<2 x float>8B$
"
	full_text

<2 x float> undef
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 3
$i648B

	full_text


i64 12
2float8B%
#
	full_text

float 5.000000e-01       
 		                        !" !! #$ ## %& %' %% (( )* )+ )) ,- ,, ./ .. 01 00 22 34 35 33 67 66 89 88 :; :: <= << >? >> @A @B @C @D @@ EF EE GH GG IJ II KL KK MN MO MM PQ PR PS PP TU TV TT WX WW YZ YY [\ [[ ]^ ]] _` _a __ bc bd bb ef ee gh gg ij ik ii lm ll np Wq gr r r r (r 2s s s !s .s 8    
	             "! $ &# '( * +) -, /. 12 4 53 76 98 ; =< ? A% B0 C D FE H JI LG NK O Q: RM S UP V XW ZY \M ^T `] a[ c_ db f he jg k< m 	 on o ww xx vv o {{ uu yy tt zz> tt >] zz ]l {{ l[ zz [T yy TP xx P uu @ vv @M ww M| 	| } ~ I   %	? 2	? I? 	? 	? (? >? l? e"
compute_step_factor"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
compute_velocity"
compute_speed_sqd"
compute_pressure"
compute_speed_of_sound"

_Z4sqrtf"
llvm.lifetime.end.p0i8*?
&rodinia-3.1-cfd-compute_step_factor.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02?
 
transfer_bytes_log1p
I??A

wgsize
?

transfer_bytes
???

devmap_label
 

wgsize_log1p
I??A