

[external]
KcallBC
A
	full_text4
2
0%3 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%4 = trunc i64 %3 to i32
"i64B

	full_text


i64 %3
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 1) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
McallBE
C
	full_text6
4
2%7 = tail call i64 @_Z15get_global_sizej(i32 0) #3
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
-mulB&
$
	full_text

%9 = mul i32 %8, %6
"i32B

	full_text


i32 %8
"i32B

	full_text


i32 %6
.addB'
%
	full_text

%10 = add i32 %9, %4
"i32B

	full_text


i32 %9
"i32B

	full_text


i32 %4
2icmpB*
(
	full_text

%11 = icmp eq i32 %4, 0
"i32B

	full_text


i32 %4
8brB2
0
	full_text#
!
br i1 %11, label %87, label %12
!i1B

	full_text


i1 %11
Pcall8BF
D
	full_text7
5
3%13 = tail call i64 @_Z15get_global_sizej(i32 1) #3
8trunc8B-
+
	full_text

%14 = trunc i64 %13 to i32
%i648B

	full_text
	
i64 %13
0add8B'
%
	full_text

%15 = add i32 %8, -1
$i328B

	full_text


i32 %8
7icmp8B-
+
	full_text

%16 = icmp ugt i32 %15, %4
%i328B

	full_text
	
i32 %15
$i328B

	full_text


i32 %4
4icmp8B*
(
	full_text

%17 = icmp ne i32 %6, 0
$i328B

	full_text


i32 %6
1and8B(
&
	full_text

%18 = and i1 %17, %16
#i18B

	full_text


i1 %17
#i18B

	full_text


i1 %16
1add8B(
&
	full_text

%19 = add i32 %14, -1
%i328B

	full_text
	
i32 %14
7icmp8B-
+
	full_text

%20 = icmp ugt i32 %19, %6
%i328B

	full_text
	
i32 %19
$i328B

	full_text


i32 %6
1and8B(
&
	full_text

%21 = and i1 %18, %20
#i18B

	full_text


i1 %18
#i18B

	full_text


i1 %20
:br8B2
0
	full_text#
!
br i1 %21, label %22, label %87
#i18B

	full_text


i1 %21
5add8B,
*
	full_text

%23 = add nsw i32 %10, -1
%i328B

	full_text
	
i32 %10
1sub8B(
&
	full_text

%24 = sub i32 %23, %8
%i328B

	full_text
	
i32 %23
$i328B

	full_text


i32 %8
6zext8B,
*
	full_text

%25 = zext i32 %24 to i64
%i328B

	full_text
	
i32 %24
bgetelementptr8BO
M
	full_text@
>
<%26 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %25
%i648B

	full_text
	
i64 %25
Cbitcast8B6
4
	full_text'
%
#%27 = bitcast <4 x i8>* %26 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %26
Hload8B>
<
	full_text/
-
+%28 = load i32, i32* %27, align 4, !tbaa !9
'i32*8B

	full_text


i32* %27
]call8BS
Q
	full_textD
B
@%29 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %28) #3
%i328B

	full_text
	
i32 %28
1sub8B(
&
	full_text

%30 = sub i32 %10, %8
%i328B

	full_text
	
i32 %10
$i328B

	full_text


i32 %8
6zext8B,
*
	full_text

%31 = zext i32 %30 to i64
%i328B

	full_text
	
i32 %30
bgetelementptr8BO
M
	full_text@
>
<%32 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %31
%i648B

	full_text
	
i64 %31
Cbitcast8B6
4
	full_text'
%
#%33 = bitcast <4 x i8>* %32 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %32
Hload8B>
<
	full_text/
-
+%34 = load i32, i32* %33, align 4, !tbaa !9
'i32*8B

	full_text


i32* %33
]call8BS
Q
	full_textD
B
@%35 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %34) #3
%i328B

	full_text
	
i32 %34
4add8B+
)
	full_text

%36 = add nsw i32 %10, 1
%i328B

	full_text
	
i32 %10
1sub8B(
&
	full_text

%37 = sub i32 %36, %8
%i328B

	full_text
	
i32 %36
$i328B

	full_text


i32 %8
6zext8B,
*
	full_text

%38 = zext i32 %37 to i64
%i328B

	full_text
	
i32 %37
bgetelementptr8BO
M
	full_text@
>
<%39 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %38
%i648B

	full_text
	
i64 %38
Cbitcast8B6
4
	full_text'
%
#%40 = bitcast <4 x i8>* %39 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %39
Hload8B>
<
	full_text/
-
+%41 = load i32, i32* %40, align 4, !tbaa !9
'i32*8B

	full_text


i32* %40
]call8BS
Q
	full_textD
B
@%42 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %41) #3
%i328B

	full_text
	
i32 %41
6sext8B,
*
	full_text

%43 = sext i32 %23 to i64
%i328B

	full_text
	
i32 %23
bgetelementptr8BO
M
	full_text@
>
<%44 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %43
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%45 = bitcast <4 x i8>* %44 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %44
Hload8B>
<
	full_text/
-
+%46 = load i32, i32* %45, align 4, !tbaa !9
'i32*8B

	full_text


i32* %45
]call8BS
Q
	full_textD
B
@%47 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %46) #3
%i328B

	full_text
	
i32 %46
6sext8B,
*
	full_text

%48 = sext i32 %10 to i64
%i328B

	full_text
	
i32 %10
6sext8B,
*
	full_text

%49 = sext i32 %36 to i64
%i328B

	full_text
	
i32 %36
bgetelementptr8BO
M
	full_text@
>
<%50 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %49
%i648B

	full_text
	
i64 %49
Cbitcast8B6
4
	full_text'
%
#%51 = bitcast <4 x i8>* %50 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %50
Hload8B>
<
	full_text/
-
+%52 = load i32, i32* %51, align 4, !tbaa !9
'i32*8B

	full_text


i32* %51
]call8BS
Q
	full_textD
B
@%53 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %52) #3
%i328B

	full_text
	
i32 %52
1add8B(
&
	full_text

%54 = add i32 %23, %8
%i328B

	full_text
	
i32 %23
$i328B

	full_text


i32 %8
6zext8B,
*
	full_text

%55 = zext i32 %54 to i64
%i328B

	full_text
	
i32 %54
bgetelementptr8BO
M
	full_text@
>
<%56 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %55
%i648B

	full_text
	
i64 %55
Cbitcast8B6
4
	full_text'
%
#%57 = bitcast <4 x i8>* %56 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %56
Hload8B>
<
	full_text/
-
+%58 = load i32, i32* %57, align 4, !tbaa !9
'i32*8B

	full_text


i32* %57
]call8BS
Q
	full_textD
B
@%59 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %58) #3
%i328B

	full_text
	
i32 %58
1add8B(
&
	full_text

%60 = add i32 %10, %8
%i328B

	full_text
	
i32 %10
$i328B

	full_text


i32 %8
6zext8B,
*
	full_text

%61 = zext i32 %60 to i64
%i328B

	full_text
	
i32 %60
bgetelementptr8BO
M
	full_text@
>
<%62 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %61
%i648B

	full_text
	
i64 %61
Cbitcast8B6
4
	full_text'
%
#%63 = bitcast <4 x i8>* %62 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %62
Hload8B>
<
	full_text/
-
+%64 = load i32, i32* %63, align 4, !tbaa !9
'i32*8B

	full_text


i32* %63
]call8BS
Q
	full_textD
B
@%65 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %64) #3
%i328B

	full_text
	
i32 %64
1add8B(
&
	full_text

%66 = add i32 %36, %8
%i328B

	full_text
	
i32 %36
$i328B

	full_text


i32 %8
6zext8B,
*
	full_text

%67 = zext i32 %66 to i64
%i328B

	full_text
	
i32 %66
bgetelementptr8BO
M
	full_text@
>
<%68 = getelementptr inbounds <4 x i8>, <4 x i8>* %0, i64 %67
%i648B

	full_text
	
i64 %67
Cbitcast8B6
4
	full_text'
%
#%69 = bitcast <4 x i8>* %68 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %68
Hload8B>
<
	full_text/
-
+%70 = load i32, i32* %69, align 4, !tbaa !9
'i32*8B

	full_text


i32* %69
]call8BS
Q
	full_textD
B
@%71 = tail call <4 x float> @_Z14convert_float4Dv4_h(i32 %70) #3
%i328B

	full_text
	
i32 %70
—call8B∆
√
	full_textµ
≤
Ø%72 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %35, <4 x float> <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>, <4 x float> %29)
5<4 x float>8B"
 
	full_text

<4 x float> %35
5<4 x float>8B"
 
	full_text

<4 x float> %29
<fadd8B2
0
	full_text#
!
%73 = fadd <4 x float> %72, %42
5<4 x float>8B"
 
	full_text

<4 x float> %72
5<4 x float>8B"
 
	full_text

<4 x float> %42
<fsub8B2
0
	full_text#
!
%74 = fsub <4 x float> %73, %59
5<4 x float>8B"
 
	full_text

<4 x float> %73
5<4 x float>8B"
 
	full_text

<4 x float> %59
’call8B 
«
	full_textπ
∂
≥%75 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %65, <4 x float> <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>, <4 x float> %74)
5<4 x float>8B"
 
	full_text

<4 x float> %65
5<4 x float>8B"
 
	full_text

<4 x float> %74
<fsub8B2
0
	full_text#
!
%76 = fsub <4 x float> %75, %71
5<4 x float>8B"
 
	full_text

<4 x float> %75
5<4 x float>8B"
 
	full_text

<4 x float> %71
<fsub8B2
0
	full_text#
!
%77 = fsub <4 x float> %29, %42
5<4 x float>8B"
 
	full_text

<4 x float> %29
5<4 x float>8B"
 
	full_text

<4 x float> %42
—call8B∆
√
	full_textµ
≤
Ø%78 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %47, <4 x float> <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>, <4 x float> %77)
5<4 x float>8B"
 
	full_text

<4 x float> %47
5<4 x float>8B"
 
	full_text

<4 x float> %77
’call8B 
«
	full_textπ
∂
≥%79 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %53, <4 x float> <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>, <4 x float> %78)
5<4 x float>8B"
 
	full_text

<4 x float> %53
5<4 x float>8B"
 
	full_text

<4 x float> %78
<fadd8B2
0
	full_text#
!
%80 = fadd <4 x float> %79, %59
5<4 x float>8B"
 
	full_text

<4 x float> %79
5<4 x float>8B"
 
	full_text

<4 x float> %59
<fsub8B2
0
	full_text#
!
%81 = fsub <4 x float> %80, %71
5<4 x float>8B"
 
	full_text

<4 x float> %80
5<4 x float>8B"
 
	full_text

<4 x float> %71
ncall8Bd
b
	full_textU
S
Q%82 = tail call <4 x float> @_Z5hypotDv4_fS_(<4 x float> %76, <4 x float> %81) #3
5<4 x float>8B"
 
	full_text

<4 x float> %76
5<4 x float>8B"
 
	full_text

<4 x float> %81
òfdiv8Bç
ä
	full_text}
{
y%83 = fdiv <4 x float> %82, <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>, !fpmath !12
5<4 x float>8B"
 
	full_text

<4 x float> %82
]call8BS
Q
	full_textD
B
@%84 = tail call i32 @_Z14convert_uchar4Dv4_f(<4 x float> %83) #3
5<4 x float>8B"
 
	full_text

<4 x float> %83
bgetelementptr8BO
M
	full_text@
>
<%85 = getelementptr inbounds <4 x i8>, <4 x i8>* %1, i64 %48
%i648B

	full_text
	
i64 %48
Cbitcast8B6
4
	full_text'
%
#%86 = bitcast <4 x i8>* %85 to i32*
1	<4 x i8>*8B 

	full_text

<4 x i8>* %85
Hstore8B=
;
	full_text.
,
*store i32 %84, i32* %86, align 4, !tbaa !9
%i328B

	full_text
	
i32 %84
'i32*8B

	full_text


i32* %86
'br8B

	full_text

br label %87
$ret8B

	full_text


ret void
0	<4 x i8>*8B

	full_text

<4 x i8>* %1
0	<4 x i8>*8B

	full_text

<4 x i8>* %0
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
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 1
Ü<4 x float>8Bs
q
	full_textd
b
`<4 x float> <float -2.000000e+00, float -2.000000e+00, float -2.000000e+00, float -2.000000e+00>
#i328B

	full_text	

i32 0
Ç<4 x float>8Bo
m
	full_text`
^
\<4 x float> <float 2.000000e+00, float 2.000000e+00, float 2.000000e+00, float 2.000000e+00>       	  
 
 

                     !" !! #$ #% ## &' &( && )* ), ++ -. -/ -- 01 00 23 22 45 44 67 66 89 88 :; :< :: => == ?@ ?? AB AA CD CC EF EE GH GG IJ IK II LM LL NO NN PQ PP RS RR TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bb de dd fg ff hi hh jk jj lm ln ll op oo qr qq st ss uv uu wx ww yz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê èè ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏
π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¡ ∏¬ 2¬ ?¬ N¬ X¬ d¬ q¬ ~¬ ã   	  
             "! $ % '# (& * ,+ . /- 10 32 54 76 9 ; <: >= @? BA DC F HG J KI ML ON QP SR U+ WV YX [Z ]\ _ aG cb ed gf ih k+ m nl po rq ts vu x z {y }| ~ ÅÄ ÉÇ ÖG á àÜ äâ åã éç êè íE î8 ïì óT òñ öw õÑ ùô ûú †ë °8 £T §^ ¶¢ ßj ©• ™® ¨w ≠´ Øë ∞ü ≤Æ ≥± µ¥ ∑` π∏ ª∂ Ω∫ æ ¿ ) +) ¿ø ¿ «« √√ ≈≈ ∆∆ ƒƒ »» ¿w ≈≈ wE ≈≈ E® ∆∆ ®ì ∆∆ ìú ∆∆ ú8 ≈≈ 8^ ≈≈ ^Ñ ≈≈ Ñë ≈≈ ë∂ «« ∂ ƒƒ  ƒƒ ± »» ± √√ T ≈≈ Tj ≈≈ j √√ • ∆∆ •	… 	… !	… +    	  G
À ú
À ®Ã Ã 	Ã 	Ã 
Õ ì
Õ •
Õ ¥"
sobel_filter"
_Z13get_global_idj"
_Z15get_global_sizej"
_Z14convert_float4Dv4_h"
llvm.fmuladd.v4f32"
_Z14convert_uchar4Dv4_f"
_Z5hypotDv4_fS_*è
SobelFilter_Kernels.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å
 
transfer_bytes_log1p
¿ÂhA

wgsize
Ä

transfer_bytes
ÄÄÄ

wgsize_log1p
¿ÂhA

devmap_label
