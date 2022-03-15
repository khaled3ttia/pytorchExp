

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #3
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
2addB+
)
	full_text

%11 = add nsw i32 %2, -1
5icmpB-
+
	full_text

%12 = icmp sgt i32 %11, %5
#i32B

	full_text
	
i32 %11
2addB+
)
	full_text

%13 = add nsw i32 %3, -1
6icmpB.
,
	full_text

%14 = icmp sgt i32 %13, %10
#i32B

	full_text
	
i32 %13
#i32B

	full_text
	
i32 %10
/andB(
&
	full_text

%15 = and i1 %12, %14
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %14
9brB3
1
	full_text$
"
 br i1 %15, label %16, label %103
!i1B

	full_text


i1 %15
4add8B+
)
	full_text

%17 = add nsw i32 %4, -1
7icmp8B-
+
	full_text

%18 = icmp sgt i32 %17, %8
%i328B

	full_text
	
i32 %17
$i328B

	full_text


i32 %8
5icmp8B+
)
	full_text

%19 = icmp sgt i32 %5, 0
1and8B(
&
	full_text

%20 = and i1 %19, %18
#i18B

	full_text


i1 %19
#i18B

	full_text


i1 %18
6icmp8B,
*
	full_text

%21 = icmp sgt i32 %10, 0
%i328B

	full_text
	
i32 %10
1and8B(
&
	full_text

%22 = and i1 %20, %21
#i18B

	full_text


i1 %20
#i18B

	full_text


i1 %21
5icmp8B+
)
	full_text

%23 = icmp sgt i32 %8, 0
$i328B

	full_text


i32 %8
1and8B(
&
	full_text

%24 = and i1 %23, %22
#i18B

	full_text


i1 %23
#i18B

	full_text


i1 %22
;br8B3
1
	full_text$
"
 br i1 %24, label %25, label %103
#i18B

	full_text


i1 %24
4add8B+
)
	full_text

%26 = add nsw i32 %5, -1
4mul8B+
)
	full_text

%27 = mul nsw i32 %4, %3
6mul8B-
+
	full_text

%28 = mul nsw i32 %26, %27
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %27
5add8B,
*
	full_text

%29 = add nsw i32 %10, -1
%i328B

	full_text
	
i32 %10
5mul8B,
*
	full_text

%30 = mul nsw i32 %29, %4
%i328B

	full_text
	
i32 %29
6add8B-
+
	full_text

%31 = add nsw i32 %30, %28
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %28
4add8B+
)
	full_text

%32 = add nsw i32 %8, -1
$i328B

	full_text


i32 %8
6add8B-
+
	full_text

%33 = add nsw i32 %31, %32
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %32
6sext8B,
*
	full_text

%34 = sext i32 %33 to i64
%i328B

	full_text
	
i32 %33
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %0, i64 %34
%i648B

	full_text
	
i64 %34
Lload8BB
@
	full_text3
1
/%36 = load float, float* %35, align 4, !tbaa !9
+float*8B

	full_text


float* %35
3add8B*
(
	full_text

%37 = add nsw i32 %5, 1
6mul8B-
+
	full_text

%38 = mul nsw i32 %37, %27
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %27
6add8B-
+
	full_text

%39 = add nsw i32 %30, %38
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %38
6add8B-
+
	full_text

%40 = add nsw i32 %39, %32
%i328B

	full_text
	
i32 %39
%i328B

	full_text
	
i32 %32
6sext8B,
*
	full_text

%41 = sext i32 %40 to i64
%i328B

	full_text
	
i32 %40
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %0, i64 %41
%i648B

	full_text
	
i64 %41
Lload8BB
@
	full_text3
1
/%43 = load float, float* %42, align 4, !tbaa !9
+float*8B

	full_text


float* %42
?fmul8B5
3
	full_text&
$
"%44 = fmul float %43, 4.000000e+00
)float8B

	full_text

	float %43
ncall8Bd
b
	full_textU
S
Q%45 = tail call float @llvm.fmuladd.f32(float %36, float 2.000000e+00, float %44)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %44
ncall8Bd
b
	full_textU
S
Q%46 = tail call float @llvm.fmuladd.f32(float %36, float 5.000000e+00, float %45)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %45
ncall8Bd
b
	full_textU
S
Q%47 = tail call float @llvm.fmuladd.f32(float %43, float 7.000000e+00, float %46)
)float8B

	full_text

	float %43
)float8B

	full_text

	float %46
ocall8Be
c
	full_textV
T
R%48 = tail call float @llvm.fmuladd.f32(float %36, float -8.000000e+00, float %47)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %47
ncall8Bd
b
	full_textU
S
Q%49 = tail call float @llvm.fmuladd.f32(float %43, float 1.000000e+01, float %48)
)float8B

	full_text

	float %43
)float8B

	full_text

	float %48
5mul8B,
*
	full_text

%50 = mul nsw i32 %27, %5
%i328B

	full_text
	
i32 %27
1add8B(
&
	full_text

%51 = add i32 %50, %8
%i328B

	full_text
	
i32 %50
$i328B

	full_text


i32 %8
2add8B)
'
	full_text

%52 = add i32 %51, %30
%i328B

	full_text
	
i32 %51
%i328B

	full_text
	
i32 %30
6sext8B,
*
	full_text

%53 = sext i32 %52 to i64
%i328B

	full_text
	
i32 %52
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %0, i64 %53
%i648B

	full_text
	
i64 %53
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !9
+float*8B

	full_text


float* %54
ocall8Be
c
	full_textV
T
R%56 = tail call float @llvm.fmuladd.f32(float %55, float -3.000000e+00, float %49)
)float8B

	full_text

	float %55
)float8B

	full_text

	float %49
5mul8B,
*
	full_text

%57 = mul nsw i32 %10, %4
%i328B

	full_text
	
i32 %10
2add8B)
'
	full_text

%58 = add i32 %51, %57
%i328B

	full_text
	
i32 %51
%i328B

	full_text
	
i32 %57
6sext8B,
*
	full_text

%59 = sext i32 %58 to i64
%i328B

	full_text
	
i32 %58
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %0, i64 %59
%i648B

	full_text
	
i64 %59
Lload8BB
@
	full_text3
1
/%61 = load float, float* %60, align 4, !tbaa !9
+float*8B

	full_text


float* %60
ncall8Bd
b
	full_textU
S
Q%62 = tail call float @llvm.fmuladd.f32(float %61, float 6.000000e+00, float %56)
)float8B

	full_text

	float %61
)float8B

	full_text

	float %56
4add8B+
)
	full_text

%63 = add nsw i32 %10, 1
%i328B

	full_text
	
i32 %10
5mul8B,
*
	full_text

%64 = mul nsw i32 %63, %4
%i328B

	full_text
	
i32 %63
2add8B)
'
	full_text

%65 = add i32 %51, %64
%i328B

	full_text
	
i32 %51
%i328B

	full_text
	
i32 %64
6sext8B,
*
	full_text

%66 = sext i32 %65 to i64
%i328B

	full_text
	
i32 %65
\getelementptr8BI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %0, i64 %66
%i648B

	full_text
	
i64 %66
Lload8BB
@
	full_text3
1
/%68 = load float, float* %67, align 4, !tbaa !9
+float*8B

	full_text


float* %67
ocall8Be
c
	full_textV
T
R%69 = tail call float @llvm.fmuladd.f32(float %68, float -9.000000e+00, float %62)
)float8B

	full_text

	float %68
)float8B

	full_text

	float %62
3add8B*
(
	full_text

%70 = add nsw i32 %8, 1
$i328B

	full_text


i32 %8
6add8B-
+
	full_text

%71 = add nsw i32 %31, %70
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %70
6sext8B,
*
	full_text

%72 = sext i32 %71 to i64
%i328B

	full_text
	
i32 %71
\getelementptr8BI
G
	full_text:
8
6%73 = getelementptr inbounds float, float* %0, i64 %72
%i648B

	full_text
	
i64 %72
Lload8BB
@
	full_text3
1
/%74 = load float, float* %73, align 4, !tbaa !9
+float*8B

	full_text


float* %73
ncall8Bd
b
	full_textU
S
Q%75 = tail call float @llvm.fmuladd.f32(float %74, float 2.000000e+00, float %69)
)float8B

	full_text

	float %74
)float8B

	full_text

	float %69
6add8B-
+
	full_text

%76 = add nsw i32 %39, %70
%i328B

	full_text
	
i32 %39
%i328B

	full_text
	
i32 %70
6sext8B,
*
	full_text

%77 = sext i32 %76 to i64
%i328B

	full_text
	
i32 %76
\getelementptr8BI
G
	full_text:
8
6%78 = getelementptr inbounds float, float* %0, i64 %77
%i648B

	full_text
	
i64 %77
Lload8BB
@
	full_text3
1
/%79 = load float, float* %78, align 4, !tbaa !9
+float*8B

	full_text


float* %78
ncall8Bd
b
	full_textU
S
Q%80 = tail call float @llvm.fmuladd.f32(float %79, float 4.000000e+00, float %75)
)float8B

	full_text

	float %79
)float8B

	full_text

	float %75
2add8B)
'
	full_text

%81 = add i32 %70, %28
%i328B

	full_text
	
i32 %70
%i328B

	full_text
	
i32 %28
2add8B)
'
	full_text

%82 = add i32 %81, %57
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %57
6sext8B,
*
	full_text

%83 = sext i32 %82 to i64
%i328B

	full_text
	
i32 %82
\getelementptr8BI
G
	full_text:
8
6%84 = getelementptr inbounds float, float* %0, i64 %83
%i648B

	full_text
	
i64 %83
Lload8BB
@
	full_text3
1
/%85 = load float, float* %84, align 4, !tbaa !9
+float*8B

	full_text


float* %84
ncall8Bd
b
	full_textU
S
Q%86 = tail call float @llvm.fmuladd.f32(float %85, float 5.000000e+00, float %80)
)float8B

	full_text

	float %85
)float8B

	full_text

	float %80
2add8B)
'
	full_text

%87 = add i32 %70, %38
%i328B

	full_text
	
i32 %70
%i328B

	full_text
	
i32 %38
2add8B)
'
	full_text

%88 = add i32 %87, %57
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %57
6sext8B,
*
	full_text

%89 = sext i32 %88 to i64
%i328B

	full_text
	
i32 %88
\getelementptr8BI
G
	full_text:
8
6%90 = getelementptr inbounds float, float* %0, i64 %89
%i648B

	full_text
	
i64 %89
Lload8BB
@
	full_text3
1
/%91 = load float, float* %90, align 4, !tbaa !9
+float*8B

	full_text


float* %90
ncall8Bd
b
	full_textU
S
Q%92 = tail call float @llvm.fmuladd.f32(float %91, float 7.000000e+00, float %86)
)float8B

	full_text

	float %91
)float8B

	full_text

	float %86
2add8B)
'
	full_text

%93 = add i32 %81, %64
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %64
6sext8B,
*
	full_text

%94 = sext i32 %93 to i64
%i328B

	full_text
	
i32 %93
\getelementptr8BI
G
	full_text:
8
6%95 = getelementptr inbounds float, float* %0, i64 %94
%i648B

	full_text
	
i64 %94
Lload8BB
@
	full_text3
1
/%96 = load float, float* %95, align 4, !tbaa !9
+float*8B

	full_text


float* %95
ocall8Be
c
	full_textV
T
R%97 = tail call float @llvm.fmuladd.f32(float %96, float -8.000000e+00, float %92)
)float8B

	full_text

	float %96
)float8B

	full_text

	float %92
2add8B)
'
	full_text

%98 = add i32 %87, %64
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %64
6sext8B,
*
	full_text

%99 = sext i32 %98 to i64
%i328B

	full_text
	
i32 %98
]getelementptr8BJ
H
	full_text;
9
7%100 = getelementptr inbounds float, float* %0, i64 %99
%i648B

	full_text
	
i64 %99
Nload8BD
B
	full_text5
3
1%101 = load float, float* %100, align 4, !tbaa !9
,float*8B

	full_text

float* %100
pcall8Bf
d
	full_textW
U
S%102 = tail call float @llvm.fmuladd.f32(float %101, float 1.000000e+01, float %97)
*float8B

	full_text


float %101
)float8B

	full_text

	float %97
(br8B 

	full_text

br label %108
1mul8B(
&
	full_text

%104 = mul i32 %5, %3
4add8B+
)
	full_text

%105 = add i32 %104, %10
&i328B

	full_text


i32 %104
%i328B

	full_text
	
i32 %10
3mul8B*
(
	full_text

%106 = mul i32 %105, %4
&i328B

	full_text


i32 %105
3add8B*
(
	full_text

%107 = add i32 %106, %8
&i328B

	full_text


i32 %106
$i328B

	full_text


i32 %8
(br8B 

	full_text

br label %108
Gphi8B>
<
	full_text/
-
+%109 = phi i32 [ %107, %103 ], [ %58, %25 ]
&i328B

	full_text


i32 %107
%i328B

	full_text
	
i32 %58
Rphi8BI
G
	full_text:
8
6%110 = phi float [ 0.000000e+00, %103 ], [ %102, %25 ]
*float8B

	full_text


float %102
8sext8B.
,
	full_text

%111 = sext i32 %109 to i64
&i328B

	full_text


i32 %109
^getelementptr8BK
I
	full_text<
:
8%112 = getelementptr inbounds float, float* %1, i64 %111
&i648B

	full_text


i64 %111
Nstore8BC
A
	full_text4
2
0store float %110, float* %112, align 4, !tbaa !9
*float8B

	full_text


float %110
,float*8B

	full_text

float* %112
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %2
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
3float8B&
$
	full_text

float -8.000000e+00
2float8B%
#
	full_text

float 4.000000e+00
2float8B%
#
	full_text

float 5.000000e+00
2float8B%
#
	full_text

float 7.000000e+00
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 -1
2float8B%
#
	full_text

float 1.000000e+01
3float8B&
$
	full_text

float -3.000000e+00
2float8B%
#
	full_text

float 6.000000e+00
2float8B%
#
	full_text

float 0.000000e+00
2float8B%
#
	full_text

float 2.000000e+00
3float8B&
$
	full_text

float -9.000000e+00
#i328B

	full_text	

i32 1       	  

                      !    "# "$ "" %& %' (( )* )+ )) ,- ,, ./ .. 01 02 00 34 33 56 57 55 89 88 :; :: <= << >> ?@ ?A ?? BC BD BB EF EG EE HI HH JK JJ LM LL NO NN PQ PR PP ST SU SS VW VX VV YZ Y[ YY \] \^ \\ _` __ ab ac aa de df dd gh gg ij ii kl kk mn mo mm pq pp rs rt rr uv uu wx ww yz yy {| {} {{ ~ ~~ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ á
à áá âä ââ ãå ã
ç ãã éè éé êë ê
í êê ìî ìì ï
ñ ïï óò óó ôö ô
õ ôô úù ú
û úú ü† üü °
¢ °° £§ ££ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ ÆÆ ∞
± ∞∞ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ ΩΩ ø
¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ Â
Á ÂÂ ËÍ È
Î ÈÈ Ï
Ì ÏÏ ÓÔ ÓÓ 
Ò  ÚÛ Ú
Ù ÚÚ ı	ˆ ˆ ˆ 'ˆ >	ˆ _ˆ ﬂ˜ ˜ (	˜ .	˜ p
˜ Ä
˜ „¯ :¯ J¯ i¯ w¯ á¯ ï¯ °¯ ∞¯ ø¯ À¯ ◊˘ 
	˘ (
˘ ﬂ˙ ˚    	
             !  # $" &' *( + -, /. 1) 2 40 63 75 98 ;: => @( A. C? DB F3 GE IH KJ ML O< QN R< TP UL WS X< ZV [L ]Y ^( `_ b ca e. fd hg ji lk n\ o qa sp tr vu xw zy |m } ~ Åa ÉÄ ÑÇ ÜÖ àá äâ å{ ç è0 ëé íê îì ñï òó öã õB ùé ûú †ü ¢° §£ ¶ô ßé ©) ™® ¨p ≠´ ØÆ ±∞ ≥≤ µ• ∂é ∏? π∑ ªp º∫ æΩ ¿ø ¬¡ ƒ¥ ≈® «Ä »∆  … ÃÀ ŒÕ –√ —∑ ”Ä ‘“ ÷’ ÿ◊ ⁄Ÿ ‹œ ›ﬂ · ‚‡ ‰„ Ê ÁÂ Ír Î€ ÌÈ ÔÓ ÒÏ Û Ù  ﬂ% '% ﬂË Èﬁ È ˝˝ ı ¸¸ ¸¸ ã ˝˝ ãS ˝˝ SY ˝˝ Yœ ˝˝ œm ˝˝ m€ ˝˝ €V ˝˝ V¥ ˝˝ ¥\ ˝˝ \ ¸¸ • ˝˝ •P ˝˝ P{ ˝˝ {ô ˝˝ ô√ ˝˝ √	˛ Y
˛ œ	ˇ N
ˇ •	Ä S
Ä ¥	Å V
Å √Ç 	Ç 	Ç 	Ç  	É 	É 
	É 	É '	É ,	É 3	Ñ \
Ñ €	Ö m	Ü {á Ï	à P
à ô
â ãä 	ä >	ä ~
ä é"
Convolution3D_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*∞
7polybench-gpu-1.0-3DConvolution-Convolution3D_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize
Ä

transfer_bytes
ÄÄÄ@

wgsize_log1p
D∏ïA

devmap_label

 
transfer_bytes_log1p
D∏ïA