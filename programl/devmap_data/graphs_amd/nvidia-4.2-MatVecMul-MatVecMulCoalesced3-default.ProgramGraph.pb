

[external]
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
%9 = icmp ult i32 %8, %3
"i32B

	full_text


i32 %8
7brB1
/
	full_text"
 
br i1 %9, label %10, label %46
 i1B

	full_text	

i1 %9
Mcall8BC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_local_idj(i32 0) #4
8trunc8B-
+
	full_text

%12 = trunc i64 %11 to i32
%i648B

	full_text
	
i64 %11
7icmp8B-
+
	full_text

%13 = icmp ult i32 %12, %2
%i328B

	full_text
	
i32 %12
\getelementptr8BI
G
	full_text:
8
6%14 = getelementptr inbounds float, float* %5, i64 %11
%i648B

	full_text
	
i64 %11
1and8B(
&
	full_text

%15 = and i32 %12, 31
%i328B

	full_text
	
i32 %12
0shl8B'
%
	full_text

%16 = shl i64 %11, 1
%i648B

	full_text
	
i64 %11
\getelementptr8BI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %5, i64 %16
%i648B

	full_text
	
i64 %16
6zext8B,
*
	full_text

%18 = zext i32 %15 to i64
%i328B

	full_text
	
i32 %15
4sub8B+
)
	full_text

%19 = sub nsw i64 0, %18
%i648B

	full_text
	
i64 %18
]getelementptr8BJ
H
	full_text;
9
7%20 = getelementptr inbounds float, float* %17, i64 %19
+float*8B

	full_text


float* %17
%i648B

	full_text
	
i64 %19
\getelementptr8BI
G
	full_text:
8
6%21 = getelementptr inbounds float, float* %20, i64 32
+float*8B

	full_text


float* %20
\getelementptr8BI
G
	full_text:
8
6%22 = getelementptr inbounds float, float* %20, i64 16
+float*8B

	full_text


float* %20
[getelementptr8BH
F
	full_text9
7
5%23 = getelementptr inbounds float, float* %20, i64 8
+float*8B

	full_text


float* %20
[getelementptr8BH
F
	full_text9
7
5%24 = getelementptr inbounds float, float* %20, i64 4
+float*8B

	full_text


float* %20
[getelementptr8BH
F
	full_text9
7
5%25 = getelementptr inbounds float, float* %20, i64 2
+float*8B

	full_text


float* %20
[getelementptr8BH
F
	full_text9
7
5%26 = getelementptr inbounds float, float* %20, i64 1
+float*8B

	full_text


float* %20
5icmp8B+
)
	full_text

%27 = icmp eq i32 %15, 0
%i328B

	full_text
	
i32 %15
2lshr8B(
&
	full_text

%28 = lshr i64 %11, 5
%i648B

	full_text
	
i64 %11
\getelementptr8BI
G
	full_text:
8
6%29 = getelementptr inbounds float, float* %5, i64 %28
%i648B

	full_text
	
i64 %28
[getelementptr8BH
F
	full_text9
7
5%30 = getelementptr inbounds float, float* %14, i64 4
+float*8B

	full_text


float* %14
[getelementptr8BH
F
	full_text9
7
5%31 = getelementptr inbounds float, float* %14, i64 2
+float*8B

	full_text


float* %14
[getelementptr8BH
F
	full_text9
7
5%32 = getelementptr inbounds float, float* %14, i64 1
+float*8B

	full_text


float* %14
5icmp8B+
)
	full_text

%33 = icmp eq i64 %11, 0
%i648B

	full_text
	
i64 %11
?bitcast8B2
0
	full_text#
!
%34 = bitcast float* %5 to i32*
Ocall8BE
C
	full_text6
4
2%35 = tail call i64 @_Z14get_local_sizej(i32 0) #4
2lshr8B(
&
	full_text

%36 = lshr i64 %35, 1
%i648B

	full_text
	
i64 %35
8icmp8B.
,
	full_text

%37 = icmp ult i64 %11, %36
%i648B

	full_text
	
i64 %11
%i648B

	full_text
	
i64 %36
2lshr8B(
&
	full_text

%38 = lshr i64 %35, 6
%i648B

	full_text
	
i64 %35
8trunc8B-
+
	full_text

%39 = trunc i64 %38 to i32
%i648B

	full_text
	
i64 %38
2lshr8B(
&
	full_text

%40 = lshr i64 %35, 7
%i648B

	full_text
	
i64 %35
9and8B0
.
	full_text!

%41 = and i64 %40, 2147483647
%i648B

	full_text
	
i64 %40
8icmp8B.
,
	full_text

%42 = icmp ult i64 %11, %41
%i648B

	full_text
	
i64 %11
%i648B

	full_text
	
i64 %41
6icmp8B,
*
	full_text

%43 = icmp ugt i32 %39, 7
%i328B

	full_text
	
i32 %39
6icmp8B,
*
	full_text

%44 = icmp ugt i32 %39, 3
%i328B

	full_text
	
i32 %39
6icmp8B,
*
	full_text

%45 = icmp ugt i32 %39, 1
%i328B

	full_text
	
i32 %39
'br8B

	full_text

br label %47
$ret8B

	full_text


ret void
Ephi8B<
:
	full_text-
+
)%48 = phi i32 [ %8, %10 ], [ %119, %115 ]
$i328B

	full_text


i32 %8
&i328B

	full_text


i32 %119
Ephi8B<
:
	full_text-
+
)%49 = phi i64 [ %7, %10 ], [ %118, %115 ]
$i648B

	full_text


i64 %7
&i648B

	full_text


i64 %118
1mul8B(
&
	full_text

%50 = mul i32 %48, %2
%i328B

	full_text
	
i32 %48
6zext8B,
*
	full_text

%51 = zext i32 %50 to i64
%i328B

	full_text
	
i32 %50
\getelementptr8BI
G
	full_text:
8
6%52 = getelementptr inbounds float, float* %0, i64 %51
%i648B

	full_text
	
i64 %51
:br8B2
0
	full_text#
!
br i1 %13, label %53, label %54
#i18B

	full_text


i1 %13
'br8B

	full_text

br label %56
Ophi8BF
D
	full_text7
5
3%55 = phi float [ 0.000000e+00, %47 ], [ %64, %56 ]
)float8B

	full_text

	float %64
Lstore8BA
?
	full_text2
0
.store float %55, float* %14, align 4, !tbaa !8
)float8B

	full_text

	float %55
+float*8B

	full_text


float* %14
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
:br8B2
0
	full_text#
!
br i1 %37, label %68, label %88
#i18B

	full_text


i1 %37
Ophi8BF
D
	full_text7
5
3%57 = phi float [ %64, %56 ], [ 0.000000e+00, %53 ]
)float8B

	full_text

	float %64
Dphi8B;
9
	full_text,
*
(%58 = phi i64 [ %65, %56 ], [ %11, %53 ]
%i648B

	full_text
	
i64 %65
%i648B

	full_text
	
i64 %11
9and8B0
.
	full_text!

%59 = and i64 %58, 4294967295
%i648B

	full_text
	
i64 %58
]getelementptr8BJ
H
	full_text;
9
7%60 = getelementptr inbounds float, float* %52, i64 %59
+float*8B

	full_text


float* %52
%i648B

	full_text
	
i64 %59
Lload8BB
@
	full_text3
1
/%61 = load float, float* %60, align 4, !tbaa !8
+float*8B

	full_text


float* %60
\getelementptr8BI
G
	full_text:
8
6%62 = getelementptr inbounds float, float* %1, i64 %59
%i648B

	full_text
	
i64 %59
Lload8BB
@
	full_text3
1
/%63 = load float, float* %62, align 4, !tbaa !8
+float*8B

	full_text


float* %62
ecall8B[
Y
	full_textL
J
H%64 = tail call float @llvm.fmuladd.f32(float %61, float %63, float %57)
)float8B

	full_text

	float %61
)float8B

	full_text

	float %63
)float8B

	full_text

	float %57
2add8B)
'
	full_text

%65 = add i64 %35, %59
%i648B

	full_text
	
i64 %35
%i648B

	full_text
	
i64 %59
8trunc8B-
+
	full_text

%66 = trunc i64 %65 to i32
%i648B

	full_text
	
i64 %65
7icmp8B-
+
	full_text

%67 = icmp ult i32 %66, %2
%i328B

	full_text
	
i32 %66
:br8B2
0
	full_text#
!
br i1 %67, label %56, label %54
#i18B

	full_text


i1 %67
Uload8BK
I
	full_text<
:
8%69 = load volatile float, float* %21, align 4, !tbaa !8
+float*8B

	full_text


float* %21
Uload8BK
I
	full_text<
:
8%70 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
6fadd8B,
*
	full_text

%71 = fadd float %69, %70
)float8B

	full_text

	float %69
)float8B

	full_text

	float %70
Ustore8BJ
H
	full_text;
9
7store volatile float %71, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %71
+float*8B

	full_text


float* %20
Uload8BK
I
	full_text<
:
8%72 = load volatile float, float* %22, align 4, !tbaa !8
+float*8B

	full_text


float* %22
Uload8BK
I
	full_text<
:
8%73 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
6fadd8B,
*
	full_text

%74 = fadd float %72, %73
)float8B

	full_text

	float %72
)float8B

	full_text

	float %73
Ustore8BJ
H
	full_text;
9
7store volatile float %74, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %74
+float*8B

	full_text


float* %20
Uload8BK
I
	full_text<
:
8%75 = load volatile float, float* %23, align 4, !tbaa !8
+float*8B

	full_text


float* %23
Uload8BK
I
	full_text<
:
8%76 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
6fadd8B,
*
	full_text

%77 = fadd float %75, %76
)float8B

	full_text

	float %75
)float8B

	full_text

	float %76
Ustore8BJ
H
	full_text;
9
7store volatile float %77, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %77
+float*8B

	full_text


float* %20
Uload8BK
I
	full_text<
:
8%78 = load volatile float, float* %24, align 4, !tbaa !8
+float*8B

	full_text


float* %24
Uload8BK
I
	full_text<
:
8%79 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
6fadd8B,
*
	full_text

%80 = fadd float %78, %79
)float8B

	full_text

	float %78
)float8B

	full_text

	float %79
Ustore8BJ
H
	full_text;
9
7store volatile float %80, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %80
+float*8B

	full_text


float* %20
Uload8BK
I
	full_text<
:
8%81 = load volatile float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
Uload8BK
I
	full_text<
:
8%82 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
6fadd8B,
*
	full_text

%83 = fadd float %81, %82
)float8B

	full_text

	float %81
)float8B

	full_text

	float %82
Ustore8BJ
H
	full_text;
9
7store volatile float %83, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %83
+float*8B

	full_text


float* %20
Uload8BK
I
	full_text<
:
8%84 = load volatile float, float* %26, align 4, !tbaa !8
+float*8B

	full_text


float* %26
Uload8BK
I
	full_text<
:
8%85 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
6fadd8B,
*
	full_text

%86 = fadd float %84, %85
)float8B

	full_text

	float %84
)float8B

	full_text

	float %85
Ustore8BJ
H
	full_text;
9
7store volatile float %86, float* %20, align 4, !tbaa !8
)float8B

	full_text

	float %86
+float*8B

	full_text


float* %20
Uload8BK
I
	full_text<
:
8%87 = load volatile float, float* %20, align 4, !tbaa !8
+float*8B

	full_text


float* %20
'br8B

	full_text

br label %88
Ophi8BF
D
	full_text7
5
3%89 = phi float [ %87, %68 ], [ 0.000000e+00, %54 ]
)float8B

	full_text

	float %87
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
:br8B2
0
	full_text#
!
br i1 %27, label %90, label %91
#i18B

	full_text


i1 %27
Lstore8	BA
?
	full_text2
0
.store float %89, float* %29, align 4, !tbaa !8
)float8	B

	full_text

	float %89
+float*8	B

	full_text


float* %29
'br8	B

	full_text

br label %91
Bcall8
B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
;br8
B3
1
	full_text$
"
 br i1 %42, label %92, label %107
#i18
B

	full_text


i1 %42
:br8B2
0
	full_text#
!
br i1 %43, label %93, label %97
#i18B

	full_text


i1 %43
Uload8BK
I
	full_text<
:
8%94 = load volatile float, float* %30, align 4, !tbaa !8
+float*8B

	full_text


float* %30
Uload8BK
I
	full_text<
:
8%95 = load volatile float, float* %14, align 4, !tbaa !8
+float*8B

	full_text


float* %14
6fadd8B,
*
	full_text

%96 = fadd float %94, %95
)float8B

	full_text

	float %94
)float8B

	full_text

	float %95
Ustore8BJ
H
	full_text;
9
7store volatile float %96, float* %14, align 4, !tbaa !8
)float8B

	full_text

	float %96
+float*8B

	full_text


float* %14
'br8B

	full_text

br label %98
;br8B3
1
	full_text$
"
 br i1 %44, label %98, label %102
#i18B

	full_text


i1 %44
Uload8BK
I
	full_text<
:
8%99 = load volatile float, float* %31, align 4, !tbaa !8
+float*8B

	full_text


float* %31
Vload8BL
J
	full_text=
;
9%100 = load volatile float, float* %14, align 4, !tbaa !8
+float*8B

	full_text


float* %14
8fadd8B.
,
	full_text

%101 = fadd float %99, %100
)float8B

	full_text

	float %99
*float8B

	full_text


float %100
Vstore8BK
I
	full_text<
:
8store volatile float %101, float* %14, align 4, !tbaa !8
*float8B

	full_text


float %101
+float*8B

	full_text


float* %14
(br8B 

	full_text

br label %103
<br8B4
2
	full_text%
#
!br i1 %45, label %103, label %107
#i18B

	full_text


i1 %45
Vload8BL
J
	full_text=
;
9%104 = load volatile float, float* %32, align 4, !tbaa !8
+float*8B

	full_text


float* %32
Vload8BL
J
	full_text=
;
9%105 = load volatile float, float* %14, align 4, !tbaa !8
+float*8B

	full_text


float* %14
9fadd8B/
-
	full_text 

%106 = fadd float %104, %105
*float8B

	full_text


float %104
*float8B

	full_text


float %105
Vstore8BK
I
	full_text<
:
8store volatile float %106, float* %14, align 4, !tbaa !8
*float8B

	full_text


float %106
+float*8B

	full_text


float* %14
(br8B 

	full_text

br label %107
<br8B4
2
	full_text%
#
!br i1 %33, label %110, label %108
#i18B

	full_text


i1 %33
:and8B1
/
	full_text"
 
%109 = and i64 %49, 4294967295
%i648B

	full_text
	
i64 %49
(br8B 

	full_text

br label %115
Iload8B?
=
	full_text0
.
,%111 = load i32, i32* %34, align 4, !tbaa !8
'i32*8B

	full_text


i32* %34
:and8B1
/
	full_text"
 
%112 = and i64 %49, 4294967295
%i648B

	full_text
	
i64 %49
^getelementptr8BK
I
	full_text<
:
8%113 = getelementptr inbounds float, float* %4, i64 %112
&i648B

	full_text


i64 %112
Bbitcast8B5
3
	full_text&
$
"%114 = bitcast float* %113 to i32*
,float*8B

	full_text

float* %113
Jstore8B?
=
	full_text0
.
,store i32 %111, i32* %114, align 4, !tbaa !8
&i328B

	full_text


i32 %111
(i32*8B

	full_text

	i32* %114
(br8B 

	full_text

br label %115
Iphi8B@
>
	full_text1
/
-%116 = phi i64 [ %109, %108 ], [ %112, %110 ]
&i648B

	full_text


i64 %109
&i648B

	full_text


i64 %112
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
Pcall8BF
D
	full_text7
5
3%117 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
5add8B,
*
	full_text

%118 = add i64 %117, %116
&i648B

	full_text


i64 %117
&i648B

	full_text


i64 %116
:trunc8B/
-
	full_text 

%119 = trunc i64 %118 to i32
&i648B

	full_text


i64 %118
9icmp8B/
-
	full_text 

%120 = icmp ult i32 %119, %3
&i328B

	full_text


i32 %119
;br8B3
1
	full_text$
"
 br i1 %120, label %47, label %46
$i18B

	full_text
	
i1 %120
*float*8B

	full_text

	float* %5
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %4
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %1
*float*8B
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
#i648B

	full_text	

i64 8
#i328B

	full_text	

i32 7
#i648B

	full_text	

i64 0
,i648B!

	full_text

i64 2147483647
$i648B

	full_text


i64 16
,i648B!

	full_text

i64 4294967295
2float8B%
#
	full_text

float 0.000000e+00
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 31
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 5
#i648B

	full_text	

i64 7
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 6       	
 		                       !    "# "" $% $$ &' && () (( *+ ** ,- ,, ./ .. 01 00 23 22 45 44 66 77 89 88 :; :< :: => == ?@ ?? AB AA CD CC EF EG EE HI HH JK JJ LM LL NQ PR PP ST SU SS VW VV XY XX Z[ ZZ \] \` __ ab ac aa dd ef eh gg ij ik ii lm ll no np nn qr qq st ss uv uu wx wy wz ww {| {} {{ ~ ~~ ÄÅ ÄÄ ÇÉ ÇÖ ÑÑ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã éè éé êë êê íì í
î íí ïñ ï
ó ïï òô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬ƒ √√ ≈≈ ∆« ∆… »
  »» ÀÃ ÕŒ Õ– œ“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €› ‹ﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÍ ÈÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ı˜ ˆ˘ ¯¯ ˙¸ ˚˚ ˝˛ ˝˝ ˇ
Ä ˇˇ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üà á
â áá ää ãã åç å
é åå èê èè ëí ëë ìî ìï ï ï ,ï 6	ñ 	ñ V
ñ Äó ˇ	ò 
ò ëô sö Z    
	  	          ! # % ' ) +* - / 1 3 57 9 ;8 <7 >= @7 BA D FC G? I? K? M Qè R Tå UP WV YX [ ]w `_ b c: fw h{ j ki mZ ol pn rl ts vq xu yg z7 |l }{ ~ ÅÄ É Ö áÑ âÜ äà å ç è ëé ìê îí ñ ó  ô õò ùö ûú † °" £ •¢ ß§ ®¶ ™ ´$ ≠ Ø¨ ±Æ ≤∞ ¥ µ& ∑ π∂ ª∏ º∫ æ ø ¡¿ ƒ( «√ …,  E ŒH –. “ ‘— ÷” ◊’ Ÿ ⁄J ›0 ﬂ ·ﬁ „‡ ‰‚ Ê ÁL Í2 Ï ÓÎ Ì ÒÔ Û Ù4 ˜S ˘6 ¸S ˛˝ Äˇ Ç˚ ÑÅ Ö¯ à˝ âã çá éå êè íë î  ON P\ ^\ _^ ge Ñe √Ç gÇ _¬ √∆ »∆ ÃÀ ÃÕ œÕ ˆœ —œ ‹ˆ ˚ˆ ¯€ ﬁ‹ ﬁ‹ ÈÜ á˙ áË ÎÈ ÎÈ ˆì Pì Oı ˆ õõ ùù úú O ûû üü ††ä üü ä õõ 7 ûû 7 úú ≈ üü ≈Ã üü Ãd üü dw ùù wã †† ã	°  	¢ H£ 	£ 4	§ C	• 	¶ l
¶ ¯
¶ ˝ß _	ß g
ß √	® L® d® ≈® Ã® ä	© 	™ 	´ $	´ 0	¨ J	≠ *	Æ AØ Ø 	Ø (Ø 7Ø ã	∞ "	∞ .	± 	± &	± 2	± 8	≤ ="
MatVecMulCoalesced3"
_Z12get_group_idj"
_Z12get_local_idj"
llvm.fmuladd.f32"
_Z14get_local_sizej"
_Z7barrierj"
_Z14get_num_groupsj*§
+nvidia-4.2-MatVecMul-MatVecMulCoalesced3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ç

devmap_label


wgsize
Ä
 
transfer_bytes_log1p
√9üA

wgsize_log1p
√9üA

transfer_bytes	
∞ìÄ“