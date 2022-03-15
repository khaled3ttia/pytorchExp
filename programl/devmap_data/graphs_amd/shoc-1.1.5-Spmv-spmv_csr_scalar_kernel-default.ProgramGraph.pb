
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
3icmpB+
)
	full_text

%9 = icmp slt i32 %8, %4
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %93
 i1B

	full_text	

i1 %9
0shl8B'
%
	full_text

%11 = shl i64 %7, 32
$i648B

	full_text


i64 %7
9ashr8B/
-
	full_text 

%12 = ashr exact i64 %11, 32
%i648B

	full_text
	
i64 %11
Xgetelementptr8BE
C
	full_text6
4
2%13 = getelementptr inbounds i32, i32* %3, i64 %12
%i648B

	full_text
	
i64 %12
Hload8B>
<
	full_text/
-
+%14 = load i32, i32* %13, align 4, !tbaa !8
'i32*8B

	full_text


i32* %13
9add8B0
.
	full_text!

%15 = add i64 %11, 4294967296
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%16 = ashr exact i64 %15, 32
%i648B

	full_text
	
i64 %15
Xgetelementptr8BE
C
	full_text6
4
2%17 = getelementptr inbounds i32, i32* %3, i64 %16
%i648B

	full_text
	
i64 %16
Hload8B>
<
	full_text/
-
+%18 = load i32, i32* %17, align 4, !tbaa !8
'i32*8B

	full_text


i32* %17
8icmp8B.
,
	full_text

%19 = icmp slt i32 %14, %18
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %18
:br8B2
0
	full_text#
!
br i1 %19, label %20, label %50
#i18B

	full_text


i1 %19
6sext8B,
*
	full_text

%21 = sext i32 %14 to i64
%i328B

	full_text
	
i32 %14
6sext8B,
*
	full_text

%22 = sext i32 %18 to i64
%i328B

	full_text
	
i32 %18
6sub8B-
+
	full_text

%23 = sub nsw i64 %22, %21
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %21
5add8B,
*
	full_text

%24 = add nsw i64 %22, -1
%i648B

	full_text
	
i64 %22
6sub8B-
+
	full_text

%25 = sub nsw i64 %24, %21
%i648B

	full_text
	
i64 %24
%i648B

	full_text
	
i64 %21
0and8B'
%
	full_text

%26 = and i64 %23, 3
%i648B

	full_text
	
i64 %23
5icmp8B+
)
	full_text

%27 = icmp eq i64 %26, 0
%i648B

	full_text
	
i64 %26
:br8B2
0
	full_text#
!
br i1 %27, label %44, label %28
#i18B

	full_text


i1 %27
'br8B

	full_text

br label %29
Dphi8B;
9
	full_text,
*
(%30 = phi i64 [ %21, %28 ], [ %41, %29 ]
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %41
Ophi8BF
D
	full_text7
5
3%31 = phi float [ 0.000000e+00, %28 ], [ %40, %29 ]
)float8B

	full_text

	float %40
Dphi8B;
9
	full_text,
*
(%32 = phi i64 [ %26, %28 ], [ %42, %29 ]
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %42
Xgetelementptr8BE
C
	full_text6
4
2%33 = getelementptr inbounds i32, i32* %2, i64 %30
%i648B

	full_text
	
i64 %30
Hload8B>
<
	full_text/
-
+%34 = load i32, i32* %33, align 4, !tbaa !8
'i32*8B

	full_text


i32* %33
\getelementptr8BI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %0, i64 %30
%i648B

	full_text
	
i64 %30
Mload8BC
A
	full_text4
2
0%36 = load float, float* %35, align 4, !tbaa !12
+float*8B

	full_text


float* %35
6sext8B,
*
	full_text

%37 = sext i32 %34 to i64
%i328B

	full_text
	
i32 %34
\getelementptr8BI
G
	full_text:
8
6%38 = getelementptr inbounds float, float* %1, i64 %37
%i648B

	full_text
	
i64 %37
Mload8BC
A
	full_text4
2
0%39 = load float, float* %38, align 4, !tbaa !12
+float*8B

	full_text


float* %38
ecall8B[
Y
	full_textL
J
H%40 = tail call float @llvm.fmuladd.f32(float %36, float %39, float %31)
)float8B

	full_text

	float %36
)float8B

	full_text

	float %39
)float8B

	full_text

	float %31
4add8B+
)
	full_text

%41 = add nsw i64 %30, 1
%i648B

	full_text
	
i64 %30
1add8B(
&
	full_text

%42 = add i64 %32, -1
%i648B

	full_text
	
i64 %32
5icmp8B+
)
	full_text

%43 = icmp eq i64 %42, 0
%i648B

	full_text
	
i64 %42
Jbr8BB
@
	full_text3
1
/br i1 %43, label %44, label %29, !llvm.loop !14
#i18B

	full_text


i1 %43
Hphi8B?
=
	full_text0
.
,%45 = phi float [ undef, %20 ], [ %40, %29 ]
)float8B

	full_text

	float %40
Dphi8B;
9
	full_text,
*
(%46 = phi i64 [ %21, %20 ], [ %41, %29 ]
%i648B

	full_text
	
i64 %21
%i648B

	full_text
	
i64 %41
Ophi8BF
D
	full_text7
5
3%47 = phi float [ 0.000000e+00, %20 ], [ %40, %29 ]
)float8B

	full_text

	float %40
6icmp8B,
*
	full_text

%48 = icmp ult i64 %25, 3
%i648B

	full_text
	
i64 %25
:br8B2
0
	full_text#
!
br i1 %48, label %50, label %49
#i18B

	full_text


i1 %48
'br8B

	full_text

br label %53
]phi8BT
R
	full_textE
C
A%51 = phi float [ 0.000000e+00, %10 ], [ %45, %44 ], [ %90, %53 ]
)float8B

	full_text

	float %45
)float8B

	full_text

	float %90
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %5, i64 %12
%i648B

	full_text
	
i64 %12
Mstore8BB
@
	full_text3
1
/store float %51, float* %52, align 4, !tbaa !12
)float8B

	full_text

	float %51
+float*8B

	full_text


float* %52
'br8B

	full_text

br label %93
Dphi8B;
9
	full_text,
*
(%54 = phi i64 [ %46, %49 ], [ %91, %53 ]
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %91
Fphi8B=
;
	full_text.
,
*%55 = phi float [ %47, %49 ], [ %90, %53 ]
)float8B

	full_text

	float %47
)float8B

	full_text

	float %90
Xgetelementptr8BE
C
	full_text6
4
2%56 = getelementptr inbounds i32, i32* %2, i64 %54
%i648B

	full_text
	
i64 %54
Hload8B>
<
	full_text/
-
+%57 = load i32, i32* %56, align 4, !tbaa !8
'i32*8B

	full_text


i32* %56
\getelementptr8BI
G
	full_text:
8
6%58 = getelementptr inbounds float, float* %0, i64 %54
%i648B

	full_text
	
i64 %54
Mload8BC
A
	full_text4
2
0%59 = load float, float* %58, align 4, !tbaa !12
+float*8B

	full_text


float* %58
6sext8B,
*
	full_text

%60 = sext i32 %57 to i64
%i328B

	full_text
	
i32 %57
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %1, i64 %60
%i648B

	full_text
	
i64 %60
Mload8BC
A
	full_text4
2
0%62 = load float, float* %61, align 4, !tbaa !12
+float*8B

	full_text


float* %61
ecall8B[
Y
	full_textL
J
H%63 = tail call float @llvm.fmuladd.f32(float %59, float %62, float %55)
)float8B

	full_text

	float %59
)float8B

	full_text

	float %62
)float8B

	full_text

	float %55
4add8B+
)
	full_text

%64 = add nsw i64 %54, 1
%i648B

	full_text
	
i64 %54
Xgetelementptr8BE
C
	full_text6
4
2%65 = getelementptr inbounds i32, i32* %2, i64 %64
%i648B

	full_text
	
i64 %64
Hload8B>
<
	full_text/
-
+%66 = load i32, i32* %65, align 4, !tbaa !8
'i32*8B

	full_text


i32* %65
\getelementptr8BI
G
	full_text:
8
6%67 = getelementptr inbounds float, float* %0, i64 %64
%i648B

	full_text
	
i64 %64
Mload8BC
A
	full_text4
2
0%68 = load float, float* %67, align 4, !tbaa !12
+float*8B

	full_text


float* %67
6sext8B,
*
	full_text

%69 = sext i32 %66 to i64
%i328B

	full_text
	
i32 %66
\getelementptr8BI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %1, i64 %69
%i648B

	full_text
	
i64 %69
Mload8BC
A
	full_text4
2
0%71 = load float, float* %70, align 4, !tbaa !12
+float*8B

	full_text


float* %70
ecall8B[
Y
	full_textL
J
H%72 = tail call float @llvm.fmuladd.f32(float %68, float %71, float %63)
)float8B

	full_text

	float %68
)float8B

	full_text

	float %71
)float8B

	full_text

	float %63
4add8B+
)
	full_text

%73 = add nsw i64 %54, 2
%i648B

	full_text
	
i64 %54
Xgetelementptr8BE
C
	full_text6
4
2%74 = getelementptr inbounds i32, i32* %2, i64 %73
%i648B

	full_text
	
i64 %73
Hload8B>
<
	full_text/
-
+%75 = load i32, i32* %74, align 4, !tbaa !8
'i32*8B

	full_text


i32* %74
\getelementptr8BI
G
	full_text:
8
6%76 = getelementptr inbounds float, float* %0, i64 %73
%i648B

	full_text
	
i64 %73
Mload8BC
A
	full_text4
2
0%77 = load float, float* %76, align 4, !tbaa !12
+float*8B

	full_text


float* %76
6sext8B,
*
	full_text

%78 = sext i32 %75 to i64
%i328B

	full_text
	
i32 %75
\getelementptr8BI
G
	full_text:
8
6%79 = getelementptr inbounds float, float* %1, i64 %78
%i648B

	full_text
	
i64 %78
Mload8BC
A
	full_text4
2
0%80 = load float, float* %79, align 4, !tbaa !12
+float*8B

	full_text


float* %79
ecall8B[
Y
	full_textL
J
H%81 = tail call float @llvm.fmuladd.f32(float %77, float %80, float %72)
)float8B

	full_text

	float %77
)float8B

	full_text

	float %80
)float8B

	full_text

	float %72
4add8B+
)
	full_text

%82 = add nsw i64 %54, 3
%i648B

	full_text
	
i64 %54
Xgetelementptr8BE
C
	full_text6
4
2%83 = getelementptr inbounds i32, i32* %2, i64 %82
%i648B

	full_text
	
i64 %82
Hload8B>
<
	full_text/
-
+%84 = load i32, i32* %83, align 4, !tbaa !8
'i32*8B

	full_text


i32* %83
\getelementptr8BI
G
	full_text:
8
6%85 = getelementptr inbounds float, float* %0, i64 %82
%i648B

	full_text
	
i64 %82
Mload8BC
A
	full_text4
2
0%86 = load float, float* %85, align 4, !tbaa !12
+float*8B

	full_text


float* %85
6sext8B,
*
	full_text

%87 = sext i32 %84 to i64
%i328B

	full_text
	
i32 %84
\getelementptr8BI
G
	full_text:
8
6%88 = getelementptr inbounds float, float* %1, i64 %87
%i648B

	full_text
	
i64 %87
Mload8BC
A
	full_text4
2
0%89 = load float, float* %88, align 4, !tbaa !12
+float*8B

	full_text


float* %88
ecall8B[
Y
	full_textL
J
H%90 = tail call float @llvm.fmuladd.f32(float %86, float %89, float %81)
)float8B

	full_text

	float %86
)float8B

	full_text

	float %89
)float8B

	full_text

	float %81
4add8B+
)
	full_text

%91 = add nsw i64 %54, 4
%i648B

	full_text
	
i64 %54
7icmp8B-
+
	full_text

%92 = icmp eq i64 %91, %22
%i648B

	full_text
	
i64 %91
%i648B

	full_text
	
i64 %22
:br8B2
0
	full_text#
!
br i1 %92, label %50, label %53
#i18B

	full_text


i1 %92
$ret8	B

	full_text


ret void
*float*8
B

	full_text

	float* %5
$i328
B

	full_text


i32 %4
&i32*8
B

	full_text
	
i32* %3
*float*8
B

	full_text

	float* %0
&i32*8
B

	full_text
	
i32* %2
*float*8
B
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
$i648
B

	full_text


i64 32
#i648
B

	full_text	

i64 2
#i328
B

	full_text	

i32 0
#i648
B

	full_text	

i64 0
,i648
B!

	full_text

i64 4294967296
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
i64 4
+float8
B

	full_text

float undef
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
i64 1
2float8
B%
#
	full_text

float 0.000000e+00      	  
 

                      !" !# !! $% $$ &' &( && )* )) +, ++ -. -1 02 00 34 33 56 57 55 89 88 :; :: <= << >? >> @A @@ BC BB DE DD FG FH FI FF JK JJ LM LL NO NN PQ PS RR TU TV TT WX WW YZ YY [\ [_ ^` ^^ ab aa cd ce cc fh gi gg jk jl jj mn mm op oo qr qq st ss uv uu wx ww yz yy {| {} {~ {{ Ä  Å
Ç ÅÅ ÉÑ ÉÉ Ö
Ü ÖÖ áà áá âä ââ ã
å ãã çé çç èê è
ë è
í èè ìî ìì ï
ñ ïï óò óó ô
ö ôô õú õõ ùû ùù ü
† üü °¢ °° £§ £
• £
¶ ££ ß® ßß ©
™ ©© ´¨ ´´ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥
¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑
∫ ∑∑ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿√ a	ƒ ≈ ≈ ∆ <∆ q∆ Ö∆ ô∆ ≠« 8« m« Å« ï« ©» B» w» ã» ü» ≥    	 
             " # %$ ' (! *) ,+ . 1J 2F 4) 6L 70 98 ;0 =< ?: A@ CB E> GD H3 I0 K5 ML ON QF S UJ VF X& ZY \R _∑ `
 b^ da eT hª iW k∑ lg nm pg rq to vu xw zs |y }j ~g Ä ÇÅ Ñ ÜÖ àÉ äâ åã éá êç ë{ íg îì ñï òì öô úó ûù †ü ¢õ §° •è ¶g ®ß ™© ¨ß Æ≠ ∞´ ≤± ¥≥ ∂Ø ∏µ π£ ∫g ºª æ øΩ ¡  ¬  ^- R- /f ¬[ ^[ ]/ 0] gP RP 0¿ ^¿ g ¬ ……   {    {∑    ∑ …… F    F£    £è    è	À 	À 
	À 
Ã ìÕ 	Œ +	Œ N	œ 	– )	– Y
– ß
— ª“ R	” $	” L	‘ J	‘ ’ 3’ W’ ^"
spmv_csr_scalar_kernel"
_Z13get_global_idj"
llvm.fmuladd.f32*¢
)shoc-1.1.5-Spmv-spmv_csr_scalar_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize
Ä
 
transfer_bytes_log1p
¿$hA

wgsize_log1p
¿$hA

devmap_label
 

transfer_bytes
Ùçz