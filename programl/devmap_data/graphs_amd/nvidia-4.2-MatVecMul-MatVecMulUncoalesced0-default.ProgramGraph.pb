
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

br i1 %8, label %9, label %73
 i1B

	full_text	

i1 %8
0mul8B'
%
	full_text

%10 = mul i32 %7, %2
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%11 = zext i32 %10 to i64
%i328B

	full_text
	
i32 %10
\getelementptr8BI
G
	full_text:
8
6%12 = getelementptr inbounds float, float* %0, i64 %11
%i648B

	full_text
	
i64 %11
4icmp8B*
(
	full_text

%13 = icmp eq i32 %2, 0
:br8B2
0
	full_text#
!
br i1 %13, label %39, label %14
#i18B

	full_text


i1 %13
5zext8B+
)
	full_text

%15 = zext i32 %2 to i64
5add8B,
*
	full_text

%16 = add nsw i64 %15, -1
%i648B

	full_text
	
i64 %15
0and8B'
%
	full_text

%17 = and i64 %15, 3
%i648B

	full_text
	
i64 %15
6icmp8B,
*
	full_text

%18 = icmp ult i64 %16, 3
%i648B

	full_text
	
i64 %16
:br8B2
0
	full_text#
!
br i1 %18, label %21, label %19
#i18B

	full_text


i1 %18
6sub8B-
+
	full_text

%20 = sub nsw i64 %15, %17
%i648B

	full_text
	
i64 %15
%i648B

	full_text
	
i64 %17
'br8B

	full_text

br label %43
Hphi8B?
=
	full_text0
.
,%22 = phi float [ undef, %14 ], [ %69, %43 ]
)float8B

	full_text

	float %69
Bphi8B9
7
	full_text*
(
&%23 = phi i64 [ 0, %14 ], [ %70, %43 ]
%i648B

	full_text
	
i64 %70
Ophi8BF
D
	full_text7
5
3%24 = phi float [ 0.000000e+00, %14 ], [ %69, %43 ]
)float8B

	full_text

	float %69
5icmp8B+
)
	full_text

%25 = icmp eq i64 %17, 0
%i648B

	full_text
	
i64 %17
:br8B2
0
	full_text#
!
br i1 %25, label %39, label %26
#i18B

	full_text


i1 %25
'br8B

	full_text

br label %27
Dphi8B;
9
	full_text,
*
(%28 = phi i64 [ %23, %26 ], [ %36, %27 ]
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %36
Fphi8B=
;
	full_text.
,
*%29 = phi float [ %24, %26 ], [ %35, %27 ]
)float8B

	full_text

	float %24
)float8B

	full_text

	float %35
Dphi8B;
9
	full_text,
*
(%30 = phi i64 [ %17, %26 ], [ %37, %27 ]
%i648B

	full_text
	
i64 %17
%i648B

	full_text
	
i64 %37
]getelementptr8BJ
H
	full_text;
9
7%31 = getelementptr inbounds float, float* %12, i64 %28
+float*8B

	full_text


float* %12
%i648B

	full_text
	
i64 %28
Lload8BB
@
	full_text3
1
/%32 = load float, float* %31, align 4, !tbaa !8
+float*8B

	full_text


float* %31
\getelementptr8BI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %1, i64 %28
%i648B

	full_text
	
i64 %28
Lload8BB
@
	full_text3
1
/%34 = load float, float* %33, align 4, !tbaa !8
+float*8B

	full_text


float* %33
ecall8B[
Y
	full_textL
J
H%35 = tail call float @llvm.fmuladd.f32(float %32, float %34, float %29)
)float8B

	full_text

	float %32
)float8B

	full_text

	float %34
)float8B

	full_text

	float %29
8add8B/
-
	full_text 

%36 = add nuw nsw i64 %28, 1
%i648B

	full_text
	
i64 %28
1add8B(
&
	full_text

%37 = add i64 %30, -1
%i648B

	full_text
	
i64 %30
5icmp8B+
)
	full_text

%38 = icmp eq i64 %37, 0
%i648B

	full_text
	
i64 %37
Jbr8BB
@
	full_text3
1
/br i1 %38, label %39, label %27, !llvm.loop !12
#i18B

	full_text


i1 %38
\phi8BS
Q
	full_textD
B
@%40 = phi float [ 0.000000e+00, %9 ], [ %22, %21 ], [ %35, %27 ]
)float8B

	full_text

	float %22
)float8B

	full_text

	float %35
8and8B/
-
	full_text 

%41 = and i64 %6, 4294967295
$i648B

	full_text


i64 %6
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %4, i64 %41
%i648B

	full_text
	
i64 %41
Lstore8BA
?
	full_text2
0
.store float %40, float* %42, align 4, !tbaa !8
)float8B

	full_text

	float %40
+float*8B

	full_text


float* %42
'br8B

	full_text

br label %73
Bphi8B9
7
	full_text*
(
&%44 = phi i64 [ 0, %19 ], [ %70, %43 ]
%i648B

	full_text
	
i64 %70
Ophi8BF
D
	full_text7
5
3%45 = phi float [ 0.000000e+00, %19 ], [ %69, %43 ]
)float8B

	full_text

	float %69
Dphi8B;
9
	full_text,
*
(%46 = phi i64 [ %20, %19 ], [ %71, %43 ]
%i648B

	full_text
	
i64 %20
%i648B

	full_text
	
i64 %71
]getelementptr8BJ
H
	full_text;
9
7%47 = getelementptr inbounds float, float* %12, i64 %44
+float*8B

	full_text


float* %12
%i648B

	full_text
	
i64 %44
Lload8BB
@
	full_text3
1
/%48 = load float, float* %47, align 4, !tbaa !8
+float*8B

	full_text


float* %47
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %1, i64 %44
%i648B

	full_text
	
i64 %44
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !8
+float*8B

	full_text


float* %49
ecall8B[
Y
	full_textL
J
H%51 = tail call float @llvm.fmuladd.f32(float %48, float %50, float %45)
)float8B

	full_text

	float %48
)float8B

	full_text

	float %50
)float8B

	full_text

	float %45
.or8B&
$
	full_text

%52 = or i64 %44, 1
%i648B

	full_text
	
i64 %44
]getelementptr8BJ
H
	full_text;
9
7%53 = getelementptr inbounds float, float* %12, i64 %52
+float*8B

	full_text


float* %12
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !8
+float*8B

	full_text


float* %53
\getelementptr8BI
G
	full_text:
8
6%55 = getelementptr inbounds float, float* %1, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%56 = load float, float* %55, align 4, !tbaa !8
+float*8B

	full_text


float* %55
ecall8B[
Y
	full_textL
J
H%57 = tail call float @llvm.fmuladd.f32(float %54, float %56, float %51)
)float8B

	full_text

	float %54
)float8B

	full_text

	float %56
)float8B

	full_text

	float %51
.or8B&
$
	full_text

%58 = or i64 %44, 2
%i648B

	full_text
	
i64 %44
]getelementptr8BJ
H
	full_text;
9
7%59 = getelementptr inbounds float, float* %12, i64 %58
+float*8B

	full_text


float* %12
%i648B

	full_text
	
i64 %58
Lload8BB
@
	full_text3
1
/%60 = load float, float* %59, align 4, !tbaa !8
+float*8B

	full_text


float* %59
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %1, i64 %58
%i648B

	full_text
	
i64 %58
Lload8BB
@
	full_text3
1
/%62 = load float, float* %61, align 4, !tbaa !8
+float*8B

	full_text


float* %61
ecall8B[
Y
	full_textL
J
H%63 = tail call float @llvm.fmuladd.f32(float %60, float %62, float %57)
)float8B

	full_text

	float %60
)float8B

	full_text

	float %62
)float8B

	full_text

	float %57
.or8B&
$
	full_text

%64 = or i64 %44, 3
%i648B

	full_text
	
i64 %44
]getelementptr8BJ
H
	full_text;
9
7%65 = getelementptr inbounds float, float* %12, i64 %64
+float*8B

	full_text


float* %12
%i648B

	full_text
	
i64 %64
Lload8BB
@
	full_text3
1
/%66 = load float, float* %65, align 4, !tbaa !8
+float*8B

	full_text


float* %65
\getelementptr8BI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %1, i64 %64
%i648B

	full_text
	
i64 %64
Lload8BB
@
	full_text3
1
/%68 = load float, float* %67, align 4, !tbaa !8
+float*8B

	full_text


float* %67
ecall8B[
Y
	full_textL
J
H%69 = tail call float @llvm.fmuladd.f32(float %66, float %68, float %63)
)float8B

	full_text

	float %66
)float8B

	full_text

	float %68
)float8B

	full_text

	float %63
4add8B+
)
	full_text

%70 = add nsw i64 %44, 4
%i648B

	full_text
	
i64 %44
1add8B(
&
	full_text

%71 = add i64 %46, -4
%i648B

	full_text
	
i64 %46
5icmp8B+
)
	full_text

%72 = icmp eq i64 %71, 0
%i648B

	full_text
	
i64 %71
:br8B2
0
	full_text#
!
br i1 %72, label %21, label %43
#i18B

	full_text


i1 %72
$ret8	B

	full_text


ret void
*float*8
B

	full_text

	float* %0
$i328
B

	full_text


i32 %2
*float*8
B

	full_text

	float* %4
*float*8
B

	full_text

	float* %1
$i328
B

	full_text


i32 %3
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
#i328
B

	full_text	

i32 0
+float8
B

	full_text

float undef
,i648
B!

	full_text

i64 4294967295
#i648
B

	full_text	

i64 3
#i648
B

	full_text	

i64 2
#i648
B

	full_text	

i64 1
$i648
B

	full_text


i64 -4
$i648
B

	full_text


i64 -1
#i648
B

	full_text	

i64 0
2float8
B%
#
	full_text

float 0.000000e+00
#i648
B

	full_text	

i64 4      	  
 

                   !    "# "" $% $$ &' &* )+ )) ,- ,. ,, /0 /1 // 23 24 22 56 55 78 77 9: 99 ;< ;= ;> ;; ?@ ?? AB AA CD CC EF EH GI GG JK JJ LM LL NO NP NN QS RR TU TT VW VX VV YZ Y[ YY \] \\ ^_ ^^ `a `` bc bd be bb fg ff hi hj hh kl kk mn mm op oo qr qs qt qq uv uu wx wy ww z{ zz |} || ~ ~~ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê è
ë è
í èè ìî ìì ïñ ïï óò óó ôö ôú 	ù ù ù û Lü 7ü ^ü mü |ü ã	†     	 
        è ì !è # %$ '  *? +" -; . 0A 1 3) 42 6) 87 :5 <9 =, >) @/ BA DC F H; I KJ MG OL Pì Sè U Wï X ZR [Y ]R _^ a\ c` dT eR g if jh lf nm pk ro sb tR v xu yw {u }| z Å~ Çq ÉR Ö áÑ àÜ äÑ åã éâ êç ëÄ íR îV ñï òó ö  õ G Q õ  & G& ( R( )ô ô RE GE ) õ °° ¢¢; ¢¢ ;q ¢¢ qÄ ¢¢ Äè ¢¢ èb ¢¢ b °° £ 	£ § 	• J	¶ 	¶ 
¶ Ñ	ß u	® ?	® f
© ï	™ 	™ A´  	´ $	´ C´ R
´ ó¨ "¨ G¨ T
≠ ì"
MatVecMulUncoalesced0"
_Z13get_global_idj"
llvm.fmuladd.f32*¶
-nvidia-4.2-MatVecMul-MatVecMulUncoalesced0.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ç
 
transfer_bytes_log1p
√9üA

transfer_bytes	
∞ìÄ“

wgsize
Ä

wgsize_log1p
√9üA

devmap_label
