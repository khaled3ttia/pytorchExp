

[external]
KcallBC
A
	full_text4
2
0%5 = tail call i64 @_Z13get_global_idj(i32 0) #3
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
1addB*
(
	full_text

%7 = add nsw i32 %2, -1
3icmpB+
)
	full_text

%8 = icmp sgt i32 %7, %6
"i32B

	full_text


i32 %7
"i32B

	full_text


i32 %6
6brB0
.
	full_text!

br i1 %8, label %9, label %81
 i1B

	full_text	

i1 %8
4mul8B+
)
	full_text

%10 = mul nsw i32 %6, %2
$i328B

	full_text


i32 %6
5add8B,
*
	full_text

%11 = add nsw i32 %10, %6
%i328B

	full_text
	
i32 %10
$i328B

	full_text


i32 %6
6sext8B,
*
	full_text

%12 = sext i32 %11 to i64
%i328B

	full_text
	
i32 %11
\getelementptr8BI
G
	full_text:
8
6%13 = getelementptr inbounds float, float* %0, i64 %12
%i648B

	full_text
	
i64 %12
Ustore8BJ
H
	full_text;
9
7store float 1.000000e+00, float* %13, align 4, !tbaa !9
+float*8B

	full_text


float* %13
3add8B*
(
	full_text

%14 = add nsw i32 %6, 1
$i328B

	full_text


i32 %6
7icmp8B-
+
	full_text

%15 = icmp slt i32 %14, %2
%i328B

	full_text
	
i32 %14
:br8B2
0
	full_text#
!
br i1 %15, label %16, label %81
#i18B

	full_text


i1 %15
5icmp8B+
)
	full_text

%17 = icmp sgt i32 %3, 0
5sext8B+
)
	full_text

%18 = sext i32 %2 to i64
0shl8B'
%
	full_text

%19 = shl i64 %5, 32
$i648B

	full_text


i64 %5
9ashr8B/
-
	full_text 

%20 = ashr exact i64 %19, 32
%i648B

	full_text
	
i64 %19
4add8B+
)
	full_text

%21 = add nsw i64 %20, 1
%i648B

	full_text
	
i64 %20
6sext8B,
*
	full_text

%22 = sext i32 %10 to i64
%i328B

	full_text
	
i32 %10
5zext8B+
)
	full_text

%23 = zext i32 %3 to i64
0and8B'
%
	full_text

%24 = and i64 %23, 1
%i648B

	full_text
	
i64 %23
4icmp8B*
(
	full_text

%25 = icmp eq i32 %3, 1
6sub8B-
+
	full_text

%26 = sub nsw i64 %23, %24
%i648B

	full_text
	
i64 %23
%i648B

	full_text
	
i64 %24
5icmp8B+
)
	full_text

%27 = icmp eq i64 %24, 0
%i648B

	full_text
	
i64 %24
'br8B

	full_text

br label %28
Dphi8B;
9
	full_text,
*
(%29 = phi i64 [ %21, %16 ], [ %78, %71 ]
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %78
6add8B-
+
	full_text

%30 = add nsw i64 %29, %22
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %22
\getelementptr8BI
G
	full_text:
8
6%31 = getelementptr inbounds float, float* %0, i64 %30
%i648B

	full_text
	
i64 %30
:br8B2
0
	full_text#
!
br i1 %17, label %32, label %71
#i18B

	full_text


i1 %17
Lload8BB
@
	full_text3
1
/%33 = load float, float* %31, align 4, !tbaa !9
+float*8B

	full_text


float* %31
:br8B2
0
	full_text#
!
br i1 %25, label %59, label %34
#i18B

	full_text


i1 %25
'br8B

	full_text

br label %35
Fphi8B=
;
	full_text.
,
*%36 = phi float [ %33, %34 ], [ %55, %35 ]
)float8B

	full_text

	float %33
)float8B

	full_text

	float %55
Bphi8B9
7
	full_text*
(
&%37 = phi i64 [ 0, %34 ], [ %56, %35 ]
%i648B

	full_text
	
i64 %56
Dphi8B;
9
	full_text,
*
(%38 = phi i64 [ %26, %34 ], [ %57, %35 ]
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %57
6mul8B-
+
	full_text

%39 = mul nsw i64 %37, %18
%i648B

	full_text
	
i64 %37
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%40 = add nsw i64 %39, %20
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %20
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %1, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !9
+float*8B

	full_text


float* %41
6add8B-
+
	full_text

%43 = add nsw i64 %39, %29
%i648B

	full_text
	
i64 %39
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %1, i64 %43
%i648B

	full_text
	
i64 %43
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !9
+float*8B

	full_text


float* %44
ecall8B[
Y
	full_textL
J
H%46 = tail call float @llvm.fmuladd.f32(float %42, float %45, float %36)
)float8B

	full_text

	float %42
)float8B

	full_text

	float %45
)float8B

	full_text

	float %36
Lstore8BA
?
	full_text2
0
.store float %46, float* %31, align 4, !tbaa !9
)float8B

	full_text

	float %46
+float*8B

	full_text


float* %31
.or8B&
$
	full_text

%47 = or i64 %37, 1
%i648B

	full_text
	
i64 %37
6mul8B-
+
	full_text

%48 = mul nsw i64 %47, %18
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%49 = add nsw i64 %48, %20
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %20
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %1, i64 %49
%i648B

	full_text
	
i64 %49
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !9
+float*8B

	full_text


float* %50
6add8B-
+
	full_text

%52 = add nsw i64 %48, %29
%i648B

	full_text
	
i64 %48
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%53 = getelementptr inbounds float, float* %1, i64 %52
%i648B

	full_text
	
i64 %52
Lload8BB
@
	full_text3
1
/%54 = load float, float* %53, align 4, !tbaa !9
+float*8B

	full_text


float* %53
ecall8B[
Y
	full_textL
J
H%55 = tail call float @llvm.fmuladd.f32(float %51, float %54, float %46)
)float8B

	full_text

	float %51
)float8B

	full_text

	float %54
)float8B

	full_text

	float %46
Lstore8BA
?
	full_text2
0
.store float %55, float* %31, align 4, !tbaa !9
)float8B

	full_text

	float %55
+float*8B

	full_text


float* %31
4add8B+
)
	full_text

%56 = add nsw i64 %37, 2
%i648B

	full_text
	
i64 %37
1add8B(
&
	full_text

%57 = add i64 %38, -2
%i648B

	full_text
	
i64 %38
5icmp8B+
)
	full_text

%58 = icmp eq i64 %57, 0
%i648B

	full_text
	
i64 %57
:br8B2
0
	full_text#
!
br i1 %58, label %59, label %35
#i18B

	full_text


i1 %58
Fphi8B=
;
	full_text.
,
*%60 = phi float [ %33, %32 ], [ %55, %35 ]
)float8B

	full_text

	float %33
)float8B

	full_text

	float %55
Bphi8B9
7
	full_text*
(
&%61 = phi i64 [ 0, %32 ], [ %56, %35 ]
%i648B

	full_text
	
i64 %56
:br8B2
0
	full_text#
!
br i1 %27, label %71, label %62
#i18B

	full_text


i1 %27
6mul8B-
+
	full_text

%63 = mul nsw i64 %61, %18
%i648B

	full_text
	
i64 %61
%i648B

	full_text
	
i64 %18
6add8B-
+
	full_text

%64 = add nsw i64 %63, %20
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %20
\getelementptr8BI
G
	full_text:
8
6%65 = getelementptr inbounds float, float* %1, i64 %64
%i648B

	full_text
	
i64 %64
Lload8BB
@
	full_text3
1
/%66 = load float, float* %65, align 4, !tbaa !9
+float*8B

	full_text


float* %65
6add8B-
+
	full_text

%67 = add nsw i64 %63, %29
%i648B

	full_text
	
i64 %63
%i648B

	full_text
	
i64 %29
\getelementptr8BI
G
	full_text:
8
6%68 = getelementptr inbounds float, float* %1, i64 %67
%i648B

	full_text
	
i64 %67
Lload8BB
@
	full_text3
1
/%69 = load float, float* %68, align 4, !tbaa !9
+float*8B

	full_text


float* %68
ecall8B[
Y
	full_textL
J
H%70 = tail call float @llvm.fmuladd.f32(float %66, float %69, float %60)
)float8B

	full_text

	float %66
)float8B

	full_text

	float %69
)float8B

	full_text

	float %60
Lstore8BA
?
	full_text2
0
.store float %70, float* %31, align 4, !tbaa !9
)float8B

	full_text

	float %70
+float*8B

	full_text


float* %31
'br8B

	full_text

br label %71
@bitcast8	B3
1
	full_text$
"
 %72 = bitcast float* %31 to i32*
+float*8	B

	full_text


float* %31
Hload8	B>
<
	full_text/
-
+%73 = load i32, i32* %72, align 4, !tbaa !9
'i32*8	B

	full_text


i32* %72
6mul8	B-
+
	full_text

%74 = mul nsw i64 %29, %18
%i648	B

	full_text
	
i64 %29
%i648	B

	full_text
	
i64 %18
6add8	B-
+
	full_text

%75 = add nsw i64 %74, %20
%i648	B

	full_text
	
i64 %74
%i648	B

	full_text
	
i64 %20
\getelementptr8	BI
G
	full_text:
8
6%76 = getelementptr inbounds float, float* %0, i64 %75
%i648	B

	full_text
	
i64 %75
@bitcast8	B3
1
	full_text$
"
 %77 = bitcast float* %76 to i32*
+float*8	B

	full_text


float* %76
Hstore8	B=
;
	full_text.
,
*store i32 %73, i32* %77, align 4, !tbaa !9
%i328	B

	full_text
	
i32 %73
'i32*8	B

	full_text


i32* %77
0add8	B'
%
	full_text

%78 = add i64 %29, 1
%i648	B

	full_text
	
i64 %29
8trunc8	B-
+
	full_text

%79 = trunc i64 %78 to i32
%i648	B

	full_text
	
i64 %78
6icmp8	B,
*
	full_text

%80 = icmp eq i32 %79, %2
%i328	B

	full_text
	
i32 %79
:br8	B2
0
	full_text#
!
br i1 %80, label %81, label %28
#i18	B

	full_text


i1 %80
$ret8
B

	full_text


ret void
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %1
$i328B
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
2float8B%
#
	full_text

float 1.000000e+00
#i648B

	full_text	

i64 1
$i328B

	full_text


i32 -1
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0       	  

                      !" !! #$ ## %% &' && (( )* )+ )) ,- ,, .0 /1 // 23 24 22 56 55 78 7: 99 ;< ;? >@ >> AB AA CD CE CC FG FH FF IJ IK II LM LL NO NN PQ PR PP ST SS UV UU WX WY WZ WW [\ [] [[ ^_ ^^ `a `b `` cd ce cc fg ff hi hh jk jl jj mn mm op oo qr qs qt qq uv uw uu xy xx z{ zz |} || ~ ~Å Ä
Ç ÄÄ É
Ñ ÉÉ ÖÜ Öà á
â áá äã ä
å ää ç
é çç èê èè ëí ë
ì ëë î
ï îî ñó ññ òô ò
ö ò
õ òò úù ú
û úú ü° †† ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™
´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µµ ∑∏ ∑∫ ∫ 5∫ ™ª 	ª 
	ª ª 
ª µº Lº Sº fº mº çº îΩ Ω %Ω (    	 
            "
 $% '% *& +& -! 0± 1/ 3# 42 6 85 :( <9 ?q @x B) Dz EA G HF J KI ML OF Q/ RP TS VN XU Y> ZW \5 ]A _^ a b` d ec gf i` k/ lj nm ph ro sW tq v5 wA yC {z }| 9 Åq Çx Ñ, ÜÉ à âá ã åä éç êá í/ ìë ïî óè ôñ öÄ õò ù5 û5 °† £/ • ¶§ ® ©ß ´™ ≠¢ Ø¨ ∞/ ≤± ¥≥ ∂µ ∏ 
 π  π. /7 97 †; Ä; =∑ π∑ /Ö †Ö á= >ü †~ Ä~ > øø ææ π ææ q øø qò øø òW øø W¿ 	¡ !	¡ &	¡ ^
¡ ±	¬ 	√ z	ƒ x	≈ 	≈ 	∆ 	∆ (	« ,« A	« |« É» 	» "
corr_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*•
,polybench-gpu-1.0-correlation-corr_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize_log1p
"§äA

devmap_label


transfer_bytes
êÄÉ
 
transfer_bytes_log1p
"§äA

wgsize
Ä