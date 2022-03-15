

[external]
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 1) #3
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
5icmpB-
+
	full_text

%14 = icmp slt i32 %13, %3
#i32B

	full_text
	
i32 %13
5icmpB-
+
	full_text

%15 = icmp slt i32 %11, %4
#i32B

	full_text
	
i32 %11
/andB(
&
	full_text

%16 = and i1 %15, %14
!i1B

	full_text


i1 %15
!i1B

	full_text


i1 %14
8brB2
0
	full_text#
!
br i1 %16, label %17, label %74
!i1B

	full_text


i1 %16
5mul8B,
*
	full_text

%18 = mul nsw i32 %13, %4
%i328B

	full_text
	
i32 %13
6add8B-
+
	full_text

%19 = add nsw i32 %18, %11
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %11
6sext8B,
*
	full_text

%20 = sext i32 %19 to i64
%i328B

	full_text
	
i32 %19
\getelementptr8BI
G
	full_text:
8
6%21 = getelementptr inbounds float, float* %0, i64 %20
%i648B

	full_text
	
i64 %20
Ustore8BJ
H
	full_text;
9
7store float 0.000000e+00, float* %21, align 4, !tbaa !9
+float*8B

	full_text


float* %21
5icmp8B+
)
	full_text

%22 = icmp sgt i32 %5, 0
:br8B2
0
	full_text#
!
br i1 %22, label %23, label %74
#i18B

	full_text


i1 %22
5mul8B,
*
	full_text

%24 = mul nsw i32 %13, %5
%i328B

	full_text
	
i32 %13
5sext8B+
)
	full_text

%25 = sext i32 %4 to i64
1shl8B(
&
	full_text

%26 = shl i64 %10, 32
%i648B

	full_text
	
i64 %10
9ashr8B/
-
	full_text 

%27 = ashr exact i64 %26, 32
%i648B

	full_text
	
i64 %26
6sext8B,
*
	full_text

%28 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
5zext8B+
)
	full_text

%29 = zext i32 %5 to i64
0and8B'
%
	full_text

%30 = and i64 %29, 1
%i648B

	full_text
	
i64 %29
4icmp8B*
(
	full_text

%31 = icmp eq i32 %5, 1
:br8B2
0
	full_text#
!
br i1 %31, label %60, label %32
#i18B

	full_text


i1 %31
6sub8B-
+
	full_text

%33 = sub nsw i64 %29, %30
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %30
'br8B

	full_text

br label %34
Ophi8BF
D
	full_text7
5
3%35 = phi float [ 0.000000e+00, %32 ], [ %56, %34 ]
)float8B

	full_text

	float %56
Bphi8B9
7
	full_text*
(
&%36 = phi i64 [ 0, %32 ], [ %57, %34 ]
%i648B

	full_text
	
i64 %57
Dphi8B;
9
	full_text,
*
(%37 = phi i64 [ %33, %32 ], [ %58, %34 ]
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %58
6add8B-
+
	full_text

%38 = add nsw i64 %36, %28
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %28
\getelementptr8BI
G
	full_text:
8
6%39 = getelementptr inbounds float, float* %1, i64 %38
%i648B

	full_text
	
i64 %38
Lload8BB
@
	full_text3
1
/%40 = load float, float* %39, align 4, !tbaa !9
+float*8B

	full_text


float* %39
5fmul8B+
)
	full_text

%41 = fmul float %40, %7
)float8B

	full_text

	float %40
6mul8B-
+
	full_text

%42 = mul nsw i64 %36, %25
%i648B

	full_text
	
i64 %36
%i648B

	full_text
	
i64 %25
6add8B-
+
	full_text

%43 = add nsw i64 %42, %27
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %27
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %2, i64 %43
%i648B

	full_text
	
i64 %43
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !9
+float*8B

	full_text


float* %44
ecall8B[
Y
	full_textL
J
H%46 = tail call float @llvm.fmuladd.f32(float %41, float %45, float %35)
)float8B

	full_text

	float %41
)float8B

	full_text

	float %45
)float8B

	full_text

	float %35
Lstore8BA
?
	full_text2
0
.store float %46, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %46
+float*8B

	full_text


float* %21
.or8B&
$
	full_text

%47 = or i64 %36, 1
%i648B

	full_text
	
i64 %36
6add8B-
+
	full_text

%48 = add nsw i64 %47, %28
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %28
\getelementptr8BI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %1, i64 %48
%i648B

	full_text
	
i64 %48
Lload8BB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !9
+float*8B

	full_text


float* %49
5fmul8B+
)
	full_text

%51 = fmul float %50, %7
)float8B

	full_text

	float %50
6mul8B-
+
	full_text

%52 = mul nsw i64 %47, %25
%i648B

	full_text
	
i64 %47
%i648B

	full_text
	
i64 %25
6add8B-
+
	full_text

%53 = add nsw i64 %52, %27
%i648B

	full_text
	
i64 %52
%i648B

	full_text
	
i64 %27
\getelementptr8BI
G
	full_text:
8
6%54 = getelementptr inbounds float, float* %2, i64 %53
%i648B

	full_text
	
i64 %53
Lload8BB
@
	full_text3
1
/%55 = load float, float* %54, align 4, !tbaa !9
+float*8B

	full_text


float* %54
ecall8B[
Y
	full_textL
J
H%56 = tail call float @llvm.fmuladd.f32(float %51, float %55, float %46)
)float8B

	full_text

	float %51
)float8B

	full_text

	float %55
)float8B

	full_text

	float %46
Lstore8BA
?
	full_text2
0
.store float %56, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %56
+float*8B

	full_text


float* %21
4add8B+
)
	full_text

%57 = add nsw i64 %36, 2
%i648B

	full_text
	
i64 %36
1add8B(
&
	full_text

%58 = add i64 %37, -2
%i648B

	full_text
	
i64 %37
5icmp8B+
)
	full_text

%59 = icmp eq i64 %58, 0
%i648B

	full_text
	
i64 %58
:br8B2
0
	full_text#
!
br i1 %59, label %60, label %34
#i18B

	full_text


i1 %59
Ophi8BF
D
	full_text7
5
3%61 = phi float [ 0.000000e+00, %23 ], [ %56, %34 ]
)float8B

	full_text

	float %56
Bphi8B9
7
	full_text*
(
&%62 = phi i64 [ 0, %23 ], [ %57, %34 ]
%i648B

	full_text
	
i64 %57
5icmp8B+
)
	full_text

%63 = icmp eq i64 %30, 0
%i648B

	full_text
	
i64 %30
:br8B2
0
	full_text#
!
br i1 %63, label %74, label %64
#i18B

	full_text


i1 %63
6add8B-
+
	full_text

%65 = add nsw i64 %62, %28
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %28
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %1, i64 %65
%i648B

	full_text
	
i64 %65
Lload8BB
@
	full_text3
1
/%67 = load float, float* %66, align 4, !tbaa !9
+float*8B

	full_text


float* %66
5fmul8B+
)
	full_text

%68 = fmul float %67, %7
)float8B

	full_text

	float %67
6mul8B-
+
	full_text

%69 = mul nsw i64 %62, %25
%i648B

	full_text
	
i64 %62
%i648B

	full_text
	
i64 %25
6add8B-
+
	full_text

%70 = add nsw i64 %69, %27
%i648B

	full_text
	
i64 %69
%i648B

	full_text
	
i64 %27
\getelementptr8BI
G
	full_text:
8
6%71 = getelementptr inbounds float, float* %2, i64 %70
%i648B

	full_text
	
i64 %70
Lload8BB
@
	full_text3
1
/%72 = load float, float* %71, align 4, !tbaa !9
+float*8B

	full_text


float* %71
ecall8B[
Y
	full_textL
J
H%73 = tail call float @llvm.fmuladd.f32(float %68, float %72, float %61)
)float8B

	full_text

	float %68
)float8B

	full_text

	float %72
)float8B

	full_text

	float %61
Lstore8BA
?
	full_text2
0
.store float %73, float* %21, align 4, !tbaa !9
)float8B

	full_text

	float %73
+float*8B

	full_text


float* %21
'br8B

	full_text

br label %74
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
*float*8B

	full_text

	float* %1
$i328B

	full_text


i32 %5
(float8B

	full_text


float %7
*float*8B

	full_text

	float* %2
$i328B

	full_text


i32 %3
$i328B

	full_text


i32 %4
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
#i648B

	full_text	

i64 1
$i648B

	full_text


i64 -2
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 2
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0
2float8B%
#
	full_text

float 0.000000e+00        	
 		                       !" !! #$ ## %& %% '' () (( ** +, +. -/ -- 02 11 34 33 56 57 55 89 8: 88 ;< ;; => == ?@ ?? AB AC AA DE DF DD GH GG IJ II KL KM KN KK OP OQ OO RS RR TU TV TT WX WW YZ YY [\ [[ ]^ ]_ ]] `a `b `` cd cc ef ee gh gi gj gg kl km kk no nn pq pp rs rr tu tw vv xy xx z{ zz |} | ~	Ä ~~ Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá äã ä
å ää ç
é çç èê èè ëí ë
ì ë
î ëë ïñ ï
ó ïï òö õ ;õ Wõ Åú 	ú ú 'ú *	ù ?	ù [
ù Öû Gû cû ç	ü 	† 		† †      
	            "! $ &' )* ,' .( /g 2n 4- 6p 73 9% :8 <; >= @3 B  CA E# FD HG J? LI M1 NK P Q3 SR U% VT XW ZY \R ^  _] a# b` dc f[ he iK jg l m3 o5 qp sr ug wn y( {z }x % Ä~ ÇÅ ÑÉ Üx à  âá ã# åä éç êÖ íè ìv îë ñ ó  ô  ô+ v+ -| ô| ~0 1ò ôt vt 1 ô °° ¢¢ °°  °° g ¢¢ gK ¢¢ Kë ¢¢ ë	£ (	£ R	§ p• 	• *¶ 3	¶ r¶ x	¶ z	ß n	® !	® #© 	© ™ ™ 1™ v"
mm2_kernel1"
_Z13get_global_idj"
llvm.fmuladd.f32*ù
$polybench-gpu-1.0-2mm-mm2_kernel1.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize
Ä

wgsize_log1p
≥ıëA
 
transfer_bytes_log1p
≥ıëA

transfer_bytes
ÄÄÄ(

devmap_label
