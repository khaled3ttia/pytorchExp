
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
5icmpB-
+
	full_text

%11 = icmp slt i32 %10, %5
#i32B

	full_text
	
i32 %10
4icmpB,
*
	full_text

%12 = icmp slt i32 %8, %5
"i32B

	full_text


i32 %8
/andB(
&
	full_text

%13 = and i1 %12, %11
!i1B

	full_text


i1 %12
!i1B

	full_text


i1 %11
8brB2
0
	full_text#
!
br i1 %13, label %14, label %69
!i1B

	full_text


i1 %13
5mul8B,
*
	full_text

%15 = mul nsw i32 %10, %5
%i328B

	full_text
	
i32 %10
5add8B,
*
	full_text

%16 = add nsw i32 %15, %8
%i328B

	full_text
	
i32 %15
$i328B

	full_text


i32 %8
6sext8B,
*
	full_text

%17 = sext i32 %16 to i64
%i328B

	full_text
	
i32 %16
\getelementptr8BI
G
	full_text:
8
6%18 = getelementptr inbounds float, float* %1, i64 %17
%i648B

	full_text
	
i64 %17
Lload8BB
@
	full_text3
1
/%19 = load float, float* %18, align 4, !tbaa !9
+float*8B

	full_text


float* %18
5fmul8B+
)
	full_text

%20 = fmul float %19, %3
)float8B

	full_text

	float %19
Lstore8BA
?
	full_text2
0
.store float %20, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %20
+float*8B

	full_text


float* %18
5icmp8B+
)
	full_text

%21 = icmp sgt i32 %4, 0
:br8B2
0
	full_text#
!
br i1 %21, label %22, label %69
#i18B

	full_text


i1 %21
5mul8B,
*
	full_text

%23 = mul nsw i32 %10, %4
%i328B

	full_text
	
i32 %10
4mul8B+
)
	full_text

%24 = mul nsw i32 %8, %4
$i328B

	full_text


i32 %8
6sext8B,
*
	full_text

%25 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
6sext8B,
*
	full_text

%26 = sext i32 %23 to i64
%i328B

	full_text
	
i32 %23
5zext8B+
)
	full_text

%27 = zext i32 %4 to i64
0and8B'
%
	full_text

%28 = and i64 %27, 1
%i648B

	full_text
	
i64 %27
4icmp8B*
(
	full_text

%29 = icmp eq i32 %4, 1
:br8B2
0
	full_text#
!
br i1 %29, label %56, label %30
#i18B

	full_text


i1 %29
6sub8B-
+
	full_text

%31 = sub nsw i64 %27, %28
%i648B

	full_text
	
i64 %27
%i648B

	full_text
	
i64 %28
'br8B

	full_text

br label %32
Fphi8B=
;
	full_text.
,
*%33 = phi float [ %20, %30 ], [ %52, %32 ]
)float8B

	full_text

	float %20
)float8B

	full_text

	float %52
Bphi8B9
7
	full_text*
(
&%34 = phi i64 [ 0, %30 ], [ %53, %32 ]
%i648B

	full_text
	
i64 %53
Dphi8B;
9
	full_text,
*
(%35 = phi i64 [ %31, %30 ], [ %54, %32 ]
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %54
6add8B-
+
	full_text

%36 = add nsw i64 %34, %26
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%37 = getelementptr inbounds float, float* %0, i64 %36
%i648B

	full_text
	
i64 %36
Lload8BB
@
	full_text3
1
/%38 = load float, float* %37, align 4, !tbaa !9
+float*8B

	full_text


float* %37
5fmul8B+
)
	full_text

%39 = fmul float %38, %2
)float8B

	full_text

	float %38
6add8B-
+
	full_text

%40 = add nsw i64 %34, %25
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %0, i64 %40
%i648B

	full_text
	
i64 %40
Lload8BB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !9
+float*8B

	full_text


float* %41
ecall8B[
Y
	full_textL
J
H%43 = tail call float @llvm.fmuladd.f32(float %39, float %42, float %33)
)float8B

	full_text

	float %39
)float8B

	full_text

	float %42
)float8B

	full_text

	float %33
Lstore8BA
?
	full_text2
0
.store float %43, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %43
+float*8B

	full_text


float* %18
.or8B&
$
	full_text

%44 = or i64 %34, 1
%i648B

	full_text
	
i64 %34
6add8B-
+
	full_text

%45 = add nsw i64 %44, %26
%i648B

	full_text
	
i64 %44
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%46 = getelementptr inbounds float, float* %0, i64 %45
%i648B

	full_text
	
i64 %45
Lload8BB
@
	full_text3
1
/%47 = load float, float* %46, align 4, !tbaa !9
+float*8B

	full_text


float* %46
5fmul8B+
)
	full_text

%48 = fmul float %47, %2
)float8B

	full_text

	float %47
6add8B-
+
	full_text

%49 = add nsw i64 %44, %25
%i648B

	full_text
	
i64 %44
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%50 = getelementptr inbounds float, float* %0, i64 %49
%i648B

	full_text
	
i64 %49
Lload8BB
@
	full_text3
1
/%51 = load float, float* %50, align 4, !tbaa !9
+float*8B

	full_text


float* %50
ecall8B[
Y
	full_textL
J
H%52 = tail call float @llvm.fmuladd.f32(float %48, float %51, float %43)
)float8B

	full_text

	float %48
)float8B

	full_text

	float %51
)float8B

	full_text

	float %43
Lstore8BA
?
	full_text2
0
.store float %52, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %52
+float*8B

	full_text


float* %18
4add8B+
)
	full_text

%53 = add nsw i64 %34, 2
%i648B

	full_text
	
i64 %34
1add8B(
&
	full_text

%54 = add i64 %35, -2
%i648B

	full_text
	
i64 %35
5icmp8B+
)
	full_text

%55 = icmp eq i64 %54, 0
%i648B

	full_text
	
i64 %54
:br8B2
0
	full_text#
!
br i1 %55, label %56, label %32
#i18B

	full_text


i1 %55
Fphi8B=
;
	full_text.
,
*%57 = phi float [ %20, %22 ], [ %52, %32 ]
)float8B

	full_text

	float %20
)float8B

	full_text

	float %52
Bphi8B9
7
	full_text*
(
&%58 = phi i64 [ 0, %22 ], [ %53, %32 ]
%i648B

	full_text
	
i64 %53
5icmp8B+
)
	full_text

%59 = icmp eq i64 %28, 0
%i648B

	full_text
	
i64 %28
:br8B2
0
	full_text#
!
br i1 %59, label %69, label %60
#i18B

	full_text


i1 %59
6add8B-
+
	full_text

%61 = add nsw i64 %58, %26
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %0, i64 %61
%i648B

	full_text
	
i64 %61
Lload8BB
@
	full_text3
1
/%63 = load float, float* %62, align 4, !tbaa !9
+float*8B

	full_text


float* %62
5fmul8B+
)
	full_text

%64 = fmul float %63, %2
)float8B

	full_text

	float %63
6add8B-
+
	full_text

%65 = add nsw i64 %58, %25
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %25
\getelementptr8BI
G
	full_text:
8
6%66 = getelementptr inbounds float, float* %0, i64 %65
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
ecall8B[
Y
	full_textL
J
H%68 = tail call float @llvm.fmuladd.f32(float %64, float %67, float %57)
)float8B

	full_text

	float %64
)float8B

	full_text

	float %67
)float8B

	full_text

	float %57
Lstore8BA
?
	full_text2
0
.store float %68, float* %18, align 4, !tbaa !9
)float8B

	full_text

	float %68
+float*8B

	full_text


float* %18
'br8B

	full_text

br label %69
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
(float8B

	full_text


float %3
(float8B

	full_text


float %2
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %5
*float*8B

	full_text

	float* %1
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
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 2        	
 		                        !" !$ ## %& %% '( '' )* )) ++ ,- ,, .. /0 /2 13 11 46 57 55 89 88 :; :< :: => =? == @A @@ BC BB DE DD FG FH FF IJ II KL KK MN MO MP MM QR QS QQ TU TT VW VX VV YZ YY [\ [[ ]^ ]] _` _a __ bc bb de dd fg fh fi ff jk jl jj mn mm op oo qr qq st sv uw uu xy xx z{ zz |} | ~	Ä ~~ Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá ä
ã ää åç åå éè é
ê é
ë éé íì í
î íí ïó  	ó #	ó %ó +ó .	ò 	ô D	ô ]
ô Öö @ö Iö Yö bö Åö ä	õ 	õ 		õ ú     
	              " $ &% (# *+ -. 0+ 2, 3 6f 7m 91 ;o <8 >) ?= A@ CB E8 G' HF JI LD NK O5 PM R S8 UT W) XV ZY \[ ^T `' a_ cb e] gd hM if k l8 n: po rq t vf wm y, {z }x ) Ä~ ÇÅ ÑÉ Üx à' âá ãä çÖ èå êu ëé ì î  ñ! #! ñ/ u/ 1| ñ| ~4 5ï ñs us 5 ûû ñ ùùM ûû Mf ûû fé ûû é ùù  ùù ü 	ü .† 8	† q† x	† z	° o	¢ ,	¢ T£ 	£  	§ m"
syrk_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*û
%polybench-gpu-1.0-syrk-syrk_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize_log1p
A

wgsize
Ä

transfer_bytes
ÄÄÄ

devmap_label

 
transfer_bytes_log1p
A