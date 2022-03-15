

[external]
KcallBC
A
	full_text4
2
0%6 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
%8 = icmp ult i32 %7, %3
"i32B

	full_text


i32 %7
6brB0
.
	full_text!

br i1 %8, label %9, label %18
 i1B

	full_text	

i1 %8
4icmp8B*
(
	full_text

%10 = icmp eq i32 %2, 0
Pcall8BF
D
	full_text7
5
3%11 = tail call i64 @_Z15get_global_sizej(i32 0) #3
5zext8B+
)
	full_text

%12 = zext i32 %2 to i64
5add8B,
*
	full_text

%13 = add nsw i64 %12, -1
%i648B

	full_text
	
i64 %12
0and8B'
%
	full_text

%14 = and i64 %12, 3
%i648B

	full_text
	
i64 %12
6icmp8B,
*
	full_text

%15 = icmp ult i64 %13, 3
%i648B

	full_text
	
i64 %13
6sub8B-
+
	full_text

%16 = sub nsw i64 %12, %14
%i648B

	full_text
	
i64 %12
%i648B

	full_text
	
i64 %14
5icmp8B+
)
	full_text

%17 = icmp eq i64 %14, 0
%i648B

	full_text
	
i64 %14
'br8B

	full_text

br label %19
$ret8B

	full_text


ret void
Bphi8B9
7
	full_text*
(
&%20 = phi i32 [ %7, %9 ], [ %49, %44 ]
$i328B

	full_text


i32 %7
%i328B

	full_text
	
i32 %49
Bphi8B9
7
	full_text*
(
&%21 = phi i64 [ %6, %9 ], [ %48, %44 ]
$i648B

	full_text


i64 %6
%i648B

	full_text
	
i64 %48
1mul8B(
&
	full_text

%22 = mul i32 %20, %2
%i328B

	full_text
	
i32 %20
6zext8B,
*
	full_text

%23 = zext i32 %22 to i64
%i328B

	full_text
	
i32 %22
\getelementptr8BI
G
	full_text:
8
6%24 = getelementptr inbounds float, float* %0, i64 %23
%i648B

	full_text
	
i64 %23
:br8B2
0
	full_text#
!
br i1 %10, label %44, label %25
#i18B

	full_text


i1 %10
:br8B2
0
	full_text#
!
br i1 %15, label %27, label %26
#i18B

	full_text


i1 %15
'br8B

	full_text

br label %51
Hphi8B?
=
	full_text0
.
,%28 = phi float [ undef, %25 ], [ %77, %51 ]
)float8B

	full_text

	float %77
Bphi8B9
7
	full_text*
(
&%29 = phi i64 [ 0, %25 ], [ %78, %51 ]
%i648B

	full_text
	
i64 %78
Ophi8BF
D
	full_text7
5
3%30 = phi float [ 0.000000e+00, %25 ], [ %77, %51 ]
)float8B

	full_text

	float %77
:br8B2
0
	full_text#
!
br i1 %17, label %44, label %31
#i18B

	full_text


i1 %17
'br8B

	full_text

br label %32
Dphi8B;
9
	full_text,
*
(%33 = phi i64 [ %41, %32 ], [ %29, %31 ]
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %29
Fphi8B=
;
	full_text.
,
*%34 = phi float [ %40, %32 ], [ %30, %31 ]
)float8B

	full_text

	float %40
)float8B

	full_text

	float %30
Dphi8B;
9
	full_text,
*
(%35 = phi i64 [ %42, %32 ], [ %14, %31 ]
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %14
]getelementptr8BJ
H
	full_text;
9
7%36 = getelementptr inbounds float, float* %24, i64 %33
+float*8B

	full_text


float* %24
%i648B

	full_text
	
i64 %33
Lload8BB
@
	full_text3
1
/%37 = load float, float* %36, align 4, !tbaa !8
+float*8B

	full_text


float* %36
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %33
%i648B

	full_text
	
i64 %33
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !8
+float*8B

	full_text


float* %38
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %37, float %39, float %34)
)float8B

	full_text

	float %37
)float8B

	full_text

	float %39
)float8B

	full_text

	float %34
8add8B/
-
	full_text 

%41 = add nuw nsw i64 %33, 1
%i648B

	full_text
	
i64 %33
1add8B(
&
	full_text

%42 = add i64 %35, -1
%i648B

	full_text
	
i64 %35
5icmp8B+
)
	full_text

%43 = icmp eq i64 %42, 0
%i648B

	full_text
	
i64 %42
Jbr8BB
@
	full_text3
1
/br i1 %43, label %44, label %32, !llvm.loop !12
#i18B

	full_text


i1 %43
]phi8	BT
R
	full_textE
C
A%45 = phi float [ 0.000000e+00, %19 ], [ %28, %27 ], [ %40, %32 ]
)float8	B

	full_text

	float %28
)float8	B

	full_text

	float %40
9and8	B0
.
	full_text!

%46 = and i64 %21, 4294967295
%i648	B

	full_text
	
i64 %21
\getelementptr8	BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %4, i64 %46
%i648	B

	full_text
	
i64 %46
Lstore8	BA
?
	full_text2
0
.store float %45, float* %47, align 4, !tbaa !8
)float8	B

	full_text

	float %45
+float*8	B

	full_text


float* %47
2add8	B)
'
	full_text

%48 = add i64 %11, %46
%i648	B

	full_text
	
i64 %11
%i648	B

	full_text
	
i64 %46
8trunc8	B-
+
	full_text

%49 = trunc i64 %48 to i32
%i648	B

	full_text
	
i64 %48
7icmp8	B-
+
	full_text

%50 = icmp ult i32 %49, %3
%i328	B

	full_text
	
i32 %49
:br8	B2
0
	full_text#
!
br i1 %50, label %19, label %18
#i18	B

	full_text


i1 %50
Bphi8
B9
7
	full_text*
(
&%52 = phi i64 [ 0, %26 ], [ %78, %51 ]
%i648
B

	full_text
	
i64 %78
Ophi8
BF
D
	full_text7
5
3%53 = phi float [ 0.000000e+00, %26 ], [ %77, %51 ]
)float8
B

	full_text

	float %77
Dphi8
B;
9
	full_text,
*
(%54 = phi i64 [ %16, %26 ], [ %79, %51 ]
%i648
B

	full_text
	
i64 %16
%i648
B

	full_text
	
i64 %79
]getelementptr8
BJ
H
	full_text;
9
7%55 = getelementptr inbounds float, float* %24, i64 %52
+float*8
B

	full_text


float* %24
%i648
B

	full_text
	
i64 %52
Lload8
BB
@
	full_text3
1
/%56 = load float, float* %55, align 4, !tbaa !8
+float*8
B

	full_text


float* %55
\getelementptr8
BI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %1, i64 %52
%i648
B

	full_text
	
i64 %52
Lload8
BB
@
	full_text3
1
/%58 = load float, float* %57, align 4, !tbaa !8
+float*8
B

	full_text


float* %57
ecall8
B[
Y
	full_textL
J
H%59 = tail call float @llvm.fmuladd.f32(float %56, float %58, float %53)
)float8
B

	full_text

	float %56
)float8
B

	full_text

	float %58
)float8
B

	full_text

	float %53
.or8
B&
$
	full_text

%60 = or i64 %52, 1
%i648
B

	full_text
	
i64 %52
]getelementptr8
BJ
H
	full_text;
9
7%61 = getelementptr inbounds float, float* %24, i64 %60
+float*8
B

	full_text


float* %24
%i648
B

	full_text
	
i64 %60
Lload8
BB
@
	full_text3
1
/%62 = load float, float* %61, align 4, !tbaa !8
+float*8
B

	full_text


float* %61
\getelementptr8
BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %1, i64 %60
%i648
B

	full_text
	
i64 %60
Lload8
BB
@
	full_text3
1
/%64 = load float, float* %63, align 4, !tbaa !8
+float*8
B

	full_text


float* %63
ecall8
B[
Y
	full_textL
J
H%65 = tail call float @llvm.fmuladd.f32(float %62, float %64, float %59)
)float8
B

	full_text

	float %62
)float8
B

	full_text

	float %64
)float8
B

	full_text

	float %59
.or8
B&
$
	full_text

%66 = or i64 %52, 2
%i648
B

	full_text
	
i64 %52
]getelementptr8
BJ
H
	full_text;
9
7%67 = getelementptr inbounds float, float* %24, i64 %66
+float*8
B

	full_text


float* %24
%i648
B

	full_text
	
i64 %66
Lload8
BB
@
	full_text3
1
/%68 = load float, float* %67, align 4, !tbaa !8
+float*8
B

	full_text


float* %67
\getelementptr8
BI
G
	full_text:
8
6%69 = getelementptr inbounds float, float* %1, i64 %66
%i648
B

	full_text
	
i64 %66
Lload8
BB
@
	full_text3
1
/%70 = load float, float* %69, align 4, !tbaa !8
+float*8
B

	full_text


float* %69
ecall8
B[
Y
	full_textL
J
H%71 = tail call float @llvm.fmuladd.f32(float %68, float %70, float %65)
)float8
B

	full_text

	float %68
)float8
B

	full_text

	float %70
)float8
B

	full_text

	float %65
.or8
B&
$
	full_text

%72 = or i64 %52, 3
%i648
B

	full_text
	
i64 %52
]getelementptr8
BJ
H
	full_text;
9
7%73 = getelementptr inbounds float, float* %24, i64 %72
+float*8
B

	full_text


float* %24
%i648
B

	full_text
	
i64 %72
Lload8
BB
@
	full_text3
1
/%74 = load float, float* %73, align 4, !tbaa !8
+float*8
B

	full_text


float* %73
\getelementptr8
BI
G
	full_text:
8
6%75 = getelementptr inbounds float, float* %1, i64 %72
%i648
B

	full_text
	
i64 %72
Lload8
BB
@
	full_text3
1
/%76 = load float, float* %75, align 4, !tbaa !8
+float*8
B

	full_text


float* %75
ecall8
B[
Y
	full_textL
J
H%77 = tail call float @llvm.fmuladd.f32(float %74, float %76, float %71)
)float8
B

	full_text

	float %74
)float8
B

	full_text

	float %76
)float8
B

	full_text

	float %71
4add8
B+
)
	full_text

%78 = add nsw i64 %52, 4
%i648
B

	full_text
	
i64 %52
1add8
B(
&
	full_text

%79 = add i64 %54, -4
%i648
B

	full_text
	
i64 %54
5icmp8
B+
)
	full_text

%80 = icmp eq i64 %79, 0
%i648
B

	full_text
	
i64 %79
:br8
B2
0
	full_text#
!
br i1 %80, label %27, label %51
#i18
B

	full_text


i1 %80
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %4
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
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 0
2float8B%
#
	full_text

float 0.000000e+00
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 -4
#i648B

	full_text	

i64 3
+float8B

	full_text

float undef
#i328B

	full_text	

i32 0
,i648B!

	full_text

i64 4294967295       		 

                     !    "# "" $% $' &* )) +, ++ -. -- /0 /3 24 22 56 57 55 89 8: 88 ;< ;= ;; >? >> @A @@ BC BB DE DF DG DD HI HH JK JJ LM LL NO NQ PR PP ST SS UV UU WX WY WW Z[ Z\ ZZ ]^ ]] _` __ ab ad cc ef ee gh gi gg jk jl jj mn mm op oo qr qq st su sv ss wx ww yz y{ yy |} || ~ ~~ ÄÅ ÄÄ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá ÜÜ àâ à
ä àà ãå ãã ç
é çç èê èè ëí ë
ì ë
î ëë ïñ ïï óò ó
ô óó öõ öö ú
ù úú ûü ûû †° †
¢ †
£ †† §• §§ ¶ß ¶¶ ®© ®® ™´ ™¨ ¨ 
	¨ 	≠ 	≠ _Æ "Ø @Ø oØ ~Ø çØ ú∞ U   
 
  
    ]  Z   !  # % '† *§ ,† . 0H 3+ 4D 6- 7J 9 :" <2 =; ?2 A@ C> EB F5 G2 I8 KJ ML O) QD R TS VP XU Y	 [S \Z ^] `_ b§ d† f h¶ i" kc lj nc po rm tq ue vc x" zw {y }w ~ Å| ÉÄ Ñs Öc á" âÜ äà åÜ éç êã íè ìÇ îc ñ" òï ôó õï ùú üö °û ¢ë £c •g ß¶ ©® ´   $ P$ &a a & )& (/ P/ 1( c1 2™ )™ cN PN 2  ≤≤ ±± ≥≥	 ≥≥ 	D ≤≤ D ±± Ç ≤≤ Çë ≤≤ ë† ≤≤ †s ≤≤ s	¥ 	¥ J	µ µ +	µ Lµ c
µ ®∂ -∂ P∂ e
∑ §	∏ H	∏ w
π Ü
∫ ¶	ª 	ª 
ª ïº )Ω 	Ω Ω 		æ S"
MatVecMulUncoalesced1"
_Z13get_global_idj"
llvm.fmuladd.f32"
_Z15get_global_sizej*¶
-nvidia-4.2-MatVecMul-MatVecMulUncoalesced1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ç

wgsize_log1p
√9üA

transfer_bytes	
∞ìÄ“

devmap_label

 
transfer_bytes_log1p
√9üA

wgsize
Ä