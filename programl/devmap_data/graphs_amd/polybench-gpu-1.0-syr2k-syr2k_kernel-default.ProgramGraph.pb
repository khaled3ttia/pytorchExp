

[external]
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 0) #3
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 1) #3
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
5icmpB-
+
	full_text

%12 = icmp slt i32 %11, %6
#i32B

	full_text
	
i32 %11
4icmpB,
*
	full_text

%13 = icmp slt i32 %9, %6
"i32B

	full_text


i32 %9
/andB(
&
	full_text

%14 = and i1 %13, %12
!i1B

	full_text


i1 %13
!i1B

	full_text


i1 %12
8brB2
0
	full_text#
!
br i1 %14, label %15, label %91
!i1B

	full_text


i1 %14
5mul8B,
*
	full_text

%16 = mul nsw i32 %11, %6
%i328B

	full_text
	
i32 %11
5add8B,
*
	full_text

%17 = add nsw i32 %16, %9
%i328B

	full_text
	
i32 %16
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%18 = sext i32 %17 to i64
%i328B

	full_text
	
i32 %17
\getelementptr8BI
G
	full_text:
8
6%19 = getelementptr inbounds float, float* %2, i64 %18
%i648B

	full_text
	
i64 %18
Lload8BB
@
	full_text3
1
/%20 = load float, float* %19, align 4, !tbaa !9
+float*8B

	full_text


float* %19
5fmul8B+
)
	full_text

%21 = fmul float %20, %4
)float8B

	full_text

	float %20
Lstore8BA
?
	full_text2
0
.store float %21, float* %19, align 4, !tbaa !9
)float8B

	full_text

	float %21
+float*8B

	full_text


float* %19
5icmp8B+
)
	full_text

%22 = icmp sgt i32 %5, 0
:br8B2
0
	full_text#
!
br i1 %22, label %23, label %91
#i18B

	full_text


i1 %22
5mul8B,
*
	full_text

%24 = mul nsw i32 %11, %5
%i328B

	full_text
	
i32 %11
4mul8B+
)
	full_text

%25 = mul nsw i32 %9, %5
$i328B

	full_text


i32 %9
6sext8B,
*
	full_text

%26 = sext i32 %25 to i64
%i328B

	full_text
	
i32 %25
6sext8B,
*
	full_text

%27 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
5zext8B+
)
	full_text

%28 = zext i32 %5 to i64
0and8B'
%
	full_text

%29 = and i64 %28, 1
%i648B

	full_text
	
i64 %28
4icmp8B*
(
	full_text

%30 = icmp eq i32 %5, 1
:br8B2
0
	full_text#
!
br i1 %30, label %71, label %31
#i18B

	full_text


i1 %30
6sub8B-
+
	full_text

%32 = sub nsw i64 %28, %29
%i648B

	full_text
	
i64 %28
%i648B

	full_text
	
i64 %29
'br8B

	full_text

br label %33
Fphi8B=
;
	full_text.
,
*%34 = phi float [ %21, %31 ], [ %67, %33 ]
)float8B

	full_text

	float %21
)float8B

	full_text

	float %67
Bphi8B9
7
	full_text*
(
&%35 = phi i64 [ 0, %31 ], [ %68, %33 ]
%i648B

	full_text
	
i64 %68
Dphi8B;
9
	full_text,
*
(%36 = phi i64 [ %32, %31 ], [ %69, %33 ]
%i648B

	full_text
	
i64 %32
%i648B

	full_text
	
i64 %69
6add8B-
+
	full_text

%37 = add nsw i64 %35, %27
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %27
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %0, i64 %37
%i648B

	full_text
	
i64 %37
Lload8BB
@
	full_text3
1
/%39 = load float, float* %38, align 4, !tbaa !9
+float*8B

	full_text


float* %38
5fmul8B+
)
	full_text

%40 = fmul float %39, %3
)float8B

	full_text

	float %39
6add8B-
+
	full_text

%41 = add nsw i64 %35, %26
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%42 = getelementptr inbounds float, float* %1, i64 %41
%i648B

	full_text
	
i64 %41
Lload8BB
@
	full_text3
1
/%43 = load float, float* %42, align 4, !tbaa !9
+float*8B

	full_text


float* %42
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %1, i64 %37
%i648B

	full_text
	
i64 %37
Lload8BB
@
	full_text3
1
/%45 = load float, float* %44, align 4, !tbaa !9
+float*8B

	full_text


float* %44
5fmul8B+
)
	full_text

%46 = fmul float %45, %3
)float8B

	full_text

	float %45
\getelementptr8BI
G
	full_text:
8
6%47 = getelementptr inbounds float, float* %0, i64 %41
%i648B

	full_text
	
i64 %41
Lload8BB
@
	full_text3
1
/%48 = load float, float* %47, align 4, !tbaa !9
+float*8B

	full_text


float* %47
6fmul8B,
*
	full_text

%49 = fmul float %46, %48
)float8B

	full_text

	float %46
)float8B

	full_text

	float %48
ecall8B[
Y
	full_textL
J
H%50 = tail call float @llvm.fmuladd.f32(float %40, float %43, float %49)
)float8B

	full_text

	float %40
)float8B

	full_text

	float %43
)float8B

	full_text

	float %49
6fadd8B,
*
	full_text

%51 = fadd float %34, %50
)float8B

	full_text

	float %34
)float8B

	full_text

	float %50
Lstore8BA
?
	full_text2
0
.store float %51, float* %19, align 4, !tbaa !9
)float8B

	full_text

	float %51
+float*8B

	full_text


float* %19
.or8B&
$
	full_text

%52 = or i64 %35, 1
%i648B

	full_text
	
i64 %35
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
6%54 = getelementptr inbounds float, float* %0, i64 %53
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
5fmul8B+
)
	full_text

%56 = fmul float %55, %3
)float8B

	full_text

	float %55
6add8B-
+
	full_text

%57 = add nsw i64 %52, %26
%i648B

	full_text
	
i64 %52
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %1, i64 %57
%i648B

	full_text
	
i64 %57
Lload8BB
@
	full_text3
1
/%59 = load float, float* %58, align 4, !tbaa !9
+float*8B

	full_text


float* %58
\getelementptr8BI
G
	full_text:
8
6%60 = getelementptr inbounds float, float* %1, i64 %53
%i648B

	full_text
	
i64 %53
Lload8BB
@
	full_text3
1
/%61 = load float, float* %60, align 4, !tbaa !9
+float*8B

	full_text


float* %60
5fmul8B+
)
	full_text

%62 = fmul float %61, %3
)float8B

	full_text

	float %61
\getelementptr8BI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %0, i64 %57
%i648B

	full_text
	
i64 %57
Lload8BB
@
	full_text3
1
/%64 = load float, float* %63, align 4, !tbaa !9
+float*8B

	full_text


float* %63
6fmul8B,
*
	full_text

%65 = fmul float %62, %64
)float8B

	full_text

	float %62
)float8B

	full_text

	float %64
ecall8B[
Y
	full_textL
J
H%66 = tail call float @llvm.fmuladd.f32(float %56, float %59, float %65)
)float8B

	full_text

	float %56
)float8B

	full_text

	float %59
)float8B

	full_text

	float %65
6fadd8B,
*
	full_text

%67 = fadd float %51, %66
)float8B

	full_text

	float %51
)float8B

	full_text

	float %66
Lstore8BA
?
	full_text2
0
.store float %67, float* %19, align 4, !tbaa !9
)float8B

	full_text

	float %67
+float*8B

	full_text


float* %19
4add8B+
)
	full_text

%68 = add nsw i64 %35, 2
%i648B

	full_text
	
i64 %35
1add8B(
&
	full_text

%69 = add i64 %36, -2
%i648B

	full_text
	
i64 %36
5icmp8B+
)
	full_text

%70 = icmp eq i64 %69, 0
%i648B

	full_text
	
i64 %69
:br8B2
0
	full_text#
!
br i1 %70, label %71, label %33
#i18B

	full_text


i1 %70
Fphi8B=
;
	full_text.
,
*%72 = phi float [ %21, %23 ], [ %67, %33 ]
)float8B

	full_text

	float %21
)float8B

	full_text

	float %67
Bphi8B9
7
	full_text*
(
&%73 = phi i64 [ 0, %23 ], [ %68, %33 ]
%i648B

	full_text
	
i64 %68
5icmp8B+
)
	full_text

%74 = icmp eq i64 %29, 0
%i648B

	full_text
	
i64 %29
:br8B2
0
	full_text#
!
br i1 %74, label %91, label %75
#i18B

	full_text


i1 %74
6add8B-
+
	full_text

%76 = add nsw i64 %73, %27
%i648B

	full_text
	
i64 %73
%i648B

	full_text
	
i64 %27
\getelementptr8BI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %0, i64 %76
%i648B

	full_text
	
i64 %76
Lload8BB
@
	full_text3
1
/%78 = load float, float* %77, align 4, !tbaa !9
+float*8B

	full_text


float* %77
5fmul8B+
)
	full_text

%79 = fmul float %78, %3
)float8B

	full_text

	float %78
6add8B-
+
	full_text

%80 = add nsw i64 %73, %26
%i648B

	full_text
	
i64 %73
%i648B

	full_text
	
i64 %26
\getelementptr8BI
G
	full_text:
8
6%81 = getelementptr inbounds float, float* %1, i64 %80
%i648B

	full_text
	
i64 %80
Lload8BB
@
	full_text3
1
/%82 = load float, float* %81, align 4, !tbaa !9
+float*8B

	full_text


float* %81
\getelementptr8BI
G
	full_text:
8
6%83 = getelementptr inbounds float, float* %1, i64 %76
%i648B

	full_text
	
i64 %76
Lload8BB
@
	full_text3
1
/%84 = load float, float* %83, align 4, !tbaa !9
+float*8B

	full_text


float* %83
5fmul8B+
)
	full_text

%85 = fmul float %84, %3
)float8B

	full_text

	float %84
\getelementptr8BI
G
	full_text:
8
6%86 = getelementptr inbounds float, float* %0, i64 %80
%i648B

	full_text
	
i64 %80
Lload8BB
@
	full_text3
1
/%87 = load float, float* %86, align 4, !tbaa !9
+float*8B

	full_text


float* %86
6fmul8B,
*
	full_text

%88 = fmul float %85, %87
)float8B

	full_text

	float %85
)float8B

	full_text

	float %87
ecall8B[
Y
	full_textL
J
H%89 = tail call float @llvm.fmuladd.f32(float %79, float %82, float %88)
)float8B

	full_text

	float %79
)float8B

	full_text

	float %82
)float8B

	full_text

	float %88
6fadd8B,
*
	full_text

%90 = fadd float %72, %89
)float8B

	full_text

	float %72
)float8B

	full_text

	float %89
Lstore8BA
?
	full_text2
0
.store float %90, float* %19, align 4, !tbaa !9
)float8B

	full_text

	float %90
+float*8B

	full_text


float* %19
'br8B

	full_text

br label %91
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %2
(float8B

	full_text


float %3
$i328B

	full_text


i32 %6
(float8B

	full_text


float %4
$i328B

	full_text


i32 %5
*float*8B

	full_text

	float* %0
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
#i648B

	full_text	

i64 2
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
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1        	
 		                        !" !$ ## %& %% '( '' )* )) ++ ,- ,, .. /0 /2 13 11 46 57 55 89 88 :; :< :: => =? == @A @@ BC BB DE DD FG FH FF IJ II KL KK MN MM OP OO QR QQ ST SS UV UU WX WY WW Z[ Z\ Z] ZZ ^_ ^` ^^ ab ac aa de dd fg fh ff ij ii kl kk mn mm op oq oo rs rr tu tt vw vv xy xx z{ zz |} || ~ ~~ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü ÉÉ áà á
â áá äã ä
å ää çé çç èê èè ëí ëë ìî ìñ ï
ó ïï ò
ô òò öõ öö úù úü û
† ûû °
¢ °° £§ ££ •¶ •• ß® ß
© ßß ™
´ ™™ ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞∞ ≤≥ ≤≤ ¥
µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ª
æ ªª ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈« 	» D	» Q	» m	» z
» •
» ≤	… 	… 		… 	  À  	À #	À %À +À .Ã @Ã SÃ iÃ |Ã °Ã ¥Õ IÕ MÕ rÕ vÕ ™Õ Æ    
	              " $ &% (# *+ -. 0+ 2, 3 6á 7ç 91 ;è <8 >) ?= A@ CB E8 G' HF JI L= NM PO RF TS VQ XU YD [K \W ]5 _Z `^ b c8 ed g) hf ji lk nd p' qo sr uf wv yx {o }| z Å~ Çm Ñt ÖÄ Ü^ àÉ âá ã å8 é: êè íë î ñá óç ô, õö ùò ü) †û ¢° §£ ¶ò ®' ©ß ´™ ≠û ØÆ ±∞ ≥ß µ¥ ∑≤ π∂ ∫• º¨ Ω∏ æï ¿ª ¡ø √ ƒ  ∆! #! ∆/ ï/ 1ú ∆ú û4 5≈ ∆ì ïì 5 ŒŒ œœ ∆ ŒŒ  ŒŒ ª œœ ªZ œœ ZÉ œœ É
– ç
— è	“ ,	“ d” 8
” ë” ò
” ö‘ 	‘  ’ 	’ ."
syr2k_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*†
'polybench-gpu-1.0-syr2k-syr2k_kernel.clu
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

devmap_label

 
transfer_bytes_log1p
áﬂçA

transfer_bytes
ÄÄÄ

wgsize_log1p
áﬂçA