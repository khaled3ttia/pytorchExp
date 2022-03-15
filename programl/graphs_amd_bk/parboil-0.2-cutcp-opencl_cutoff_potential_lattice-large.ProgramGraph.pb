

[external]
3sextB+
)
	full_text

%12 = sext i32 %3 to i64
fgetelementptrBU
S
	full_textF
D
B%13 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %12
#i64B

	full_text
	
i64 %12
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_local_idj(i32 2) #3
McallBE
C
	full_text6
4
2%15 = tail call i64 @_Z14get_local_sizej(i32 1) #3
0mulB)
'
	full_text

%16 = mul i64 %15, %14
#i64B

	full_text
	
i64 %15
#i64B

	full_text
	
i64 %14
KcallBC
A
	full_text4
2
0%17 = tail call i64 @_Z12get_local_idj(i32 1) #3
0addB)
'
	full_text

%18 = add i64 %16, %17
#i64B

	full_text
	
i64 %16
#i64B

	full_text
	
i64 %17
McallBE
C
	full_text6
4
2%19 = tail call i64 @_Z14get_local_sizej(i32 0) #3
0mulB)
'
	full_text

%20 = mul i64 %18, %19
#i64B

	full_text
	
i64 %18
#i64B

	full_text
	
i64 %19
KcallBC
A
	full_text4
2
0%21 = tail call i64 @_Z12get_local_idj(i32 0) #3
0addB)
'
	full_text

%22 = add i64 %20, %21
#i64B

	full_text
	
i64 %20
#i64B

	full_text
	
i64 %21
3sextB+
)
	full_text

%23 = sext i32 %8 to i64
McallBE
C
	full_text6
4
2%24 = tail call i64 @_Z14get_num_groupsj(i32 1) #3
0mulB)
'
	full_text

%25 = mul i64 %24, %23
#i64B

	full_text
	
i64 %24
#i64B

	full_text
	
i64 %23
KcallBC
A
	full_text4
2
0%26 = tail call i64 @_Z12get_group_idj(i32 1) #3
0addB)
'
	full_text

%27 = add i64 %25, %26
#i64B

	full_text
	
i64 %25
#i64B

	full_text
	
i64 %26
McallBE
C
	full_text6
4
2%28 = tail call i64 @_Z14get_num_groupsj(i32 0) #3
0lshrB(
&
	full_text

%29 = lshr i64 %28, 2
#i64B

	full_text
	
i64 %28
0mulB)
'
	full_text

%30 = mul i64 %29, %27
#i64B

	full_text
	
i64 %29
#i64B

	full_text
	
i64 %27
KcallBC
A
	full_text4
2
0%31 = tail call i64 @_Z12get_group_idj(i32 0) #3
0lshrB(
&
	full_text

%32 = lshr i64 %31, 2
#i64B

	full_text
	
i64 %31
0addB)
'
	full_text

%33 = add i64 %30, %32
#i64B

	full_text
	
i64 %30
#i64B

	full_text
	
i64 %32
.shlB'
%
	full_text

%34 = shl i64 %33, 9
#i64B

	full_text
	
i64 %33
ZgetelementptrBI
G
	full_text:
8
6%35 = getelementptr inbounds float, float* %7, i64 %34
#i64B

	full_text
	
i64 %34
.andB'
%
	full_text

%36 = and i64 %31, 3
#i64B

	full_text
	
i64 %31
6shlB/
-
	full_text 

%37 = shl nuw nsw i64 %36, 7
#i64B

	full_text
	
i64 %36
[getelementptrBJ
H
	full_text;
9
7%38 = getelementptr inbounds float, float* %35, i64 %37
)float*B

	full_text


float* %35
#i64B

	full_text
	
i64 %37
.shlB'
%
	full_text

%39 = shl i64 %32, 3
#i64B

	full_text
	
i64 %32
0addB)
'
	full_text

%40 = add i64 %39, %21
#i64B

	full_text
	
i64 %39
#i64B

	full_text
	
i64 %21
:uitofpB0
.
	full_text!

%41 = uitofp i64 %40 to float
#i64B

	full_text
	
i64 %40
3fmulB+
)
	full_text

%42 = fmul float %41, %4
'floatB

	full_text

	float %41
.shlB'
%
	full_text

%43 = shl i64 %26, 3
#i64B

	full_text
	
i64 %26
0addB)
'
	full_text

%44 = add i64 %43, %17
#i64B

	full_text
	
i64 %43
#i64B

	full_text
	
i64 %17
:uitofpB0
.
	full_text!

%45 = uitofp i64 %44 to float
#i64B

	full_text
	
i64 %44
3fmulB+
)
	full_text

%46 = fmul float %45, %4
'floatB

	full_text

	float %45
1shlB*
(
	full_text

%47 = shl nsw i32 %8, 3
4sextB,
*
	full_text

%48 = sext i32 %47 to i64
#i32B

	full_text
	
i32 %47
6shlB/
-
	full_text 

%49 = shl nuw nsw i64 %36, 1
#i64B

	full_text
	
i64 %36
.orB(
&
	full_text

%50 = or i64 %49, %48
#i64B

	full_text
	
i64 %49
#i64B

	full_text
	
i64 %48
0addB)
'
	full_text

%51 = add i64 %50, %14
#i64B

	full_text
	
i64 %50
#i64B

	full_text
	
i64 %14
:uitofpB0
.
	full_text!

%52 = uitofp i64 %51 to float
#i64B

	full_text
	
i64 %51
3fmulB+
)
	full_text

%53 = fmul float %52, %4
'floatB

	full_text

	float %52
,orB&
$
	full_text

%54 = or i64 %39, 4
#i64B

	full_text
	
i64 %39
:uitofpB0
.
	full_text!

%55 = uitofp i64 %54 to float
#i64B

	full_text
	
i64 %54
3fmulB+
)
	full_text

%56 = fmul float %55, %4
'floatB

	full_text

	float %55
=fmulB5
3
	full_text&
$
"%57 = fmul float %56, 2.500000e-01
'floatB

	full_text

	float %56
IcallBA
?
	full_text2
0
.%58 = tail call float @_Z5floorf(float %57) #3
'floatB

	full_text

	float %57
:fptosiB0
.
	full_text!

%59 = fptosi float %58 to i32
'floatB

	full_text

	float %58
,orB&
$
	full_text

%60 = or i64 %43, 4
#i64B

	full_text
	
i64 %43
:uitofpB0
.
	full_text!

%61 = uitofp i64 %60 to float
#i64B

	full_text
	
i64 %60
3fmulB+
)
	full_text

%62 = fmul float %61, %4
'floatB

	full_text

	float %61
=fmulB5
3
	full_text&
$
"%63 = fmul float %62, 2.500000e-01
'floatB

	full_text

	float %62
IcallBA
?
	full_text2
0
.%64 = tail call float @_Z5floorf(float %63) #3
'floatB

	full_text

	float %63
:fptosiB0
.
	full_text!

%65 = fptosi float %64 to i32
'floatB

	full_text

	float %64
,orB&
$
	full_text

%66 = or i32 %47, 4
#i32B

	full_text
	
i32 %47
:sitofpB0
.
	full_text!

%67 = sitofp i32 %66 to float
#i32B

	full_text
	
i32 %66
3fmulB+
)
	full_text

%68 = fmul float %67, %4
'floatB

	full_text

	float %67
=fmulB5
3
	full_text&
$
"%69 = fmul float %68, 2.500000e-01
'floatB

	full_text

	float %68
IcallBA
?
	full_text2
0
.%70 = tail call float @_Z5floorf(float %69) #3
'floatB

	full_text

	float %69
:fptosiB0
.
	full_text!

%71 = fptosi float %70 to i32
'floatB

	full_text

	float %70
EloadB=
;
	full_text.
,
*%72 = load i32, i32* %9, align 4, !tbaa !9
4icmpB,
*
	full_text

%73 = icmp sgt i32 %72, 0
#i32B

	full_text
	
i32 %72
9brB3
1
	full_text$
"
 br i1 %73, label %74, label %126
!i1B

	full_text


i1 %73
6sext8B,
*
	full_text

%75 = sext i32 %72 to i64
%i328B

	full_text
	
i32 %72
'br8B

	full_text

br label %76
Dphi8B;
9
	full_text,
*
(%77 = phi i64 [ 0, %74 ], [ %124, %123 ]
&i648B

	full_text


i64 %124
Qphi8BH
F
	full_text9
7
5%78 = phi float [ 0.000000e+00, %74 ], [ %120, %123 ]
*float8B

	full_text


float %120
egetelementptr8BR
P
	full_textC
A
?%79 = getelementptr inbounds <4 x i32>, <4 x i32>* %10, i64 %77
%i648B

	full_text
	
i64 %77
Kload8BA
?
	full_text2
0
.%80 = load <4 x i32>, <4 x i32>* %79, align 16
3
<4 x i32>*8B!

	full_text

<4 x i32>* %79
Pextractelement8B<
:
	full_text-
+
)%81 = extractelement <4 x i32> %80, i64 0
1	<4 x i32>8B 

	full_text

<4 x i32> %80
6add8B-
+
	full_text

%82 = add nsw i32 %81, %59
%i328B

	full_text
	
i32 %81
%i328B

	full_text
	
i32 %59
Pextractelement8B<
:
	full_text-
+
)%83 = extractelement <4 x i32> %80, i64 1
1	<4 x i32>8B 

	full_text

<4 x i32> %80
6add8B-
+
	full_text

%84 = add nsw i32 %83, %65
%i328B

	full_text
	
i32 %83
%i328B

	full_text
	
i32 %65
Pextractelement8B<
:
	full_text-
+
)%85 = extractelement <4 x i32> %80, i64 2
1	<4 x i32>8B 

	full_text

<4 x i32> %80
6add8B-
+
	full_text

%86 = add nsw i32 %85, %71
%i328B

	full_text
	
i32 %85
%i328B

	full_text
	
i32 %71
5mul8B,
*
	full_text

%87 = mul nsw i32 %86, %1
%i328B

	full_text
	
i32 %86
6add8B-
+
	full_text

%88 = add nsw i32 %84, %87
%i328B

	full_text
	
i32 %84
%i328B

	full_text
	
i32 %87
5mul8B,
*
	full_text

%89 = mul nsw i32 %88, %0
%i328B

	full_text
	
i32 %88
6add8B-
+
	full_text

%90 = add nsw i32 %82, %89
%i328B

	full_text
	
i32 %82
%i328B

	full_text
	
i32 %89
4shl8B+
)
	full_text

%91 = shl nsw i32 %90, 3
%i328B

	full_text
	
i32 %90
6sext8B,
*
	full_text

%92 = sext i32 %91 to i64
%i328B

	full_text
	
i32 %91
igetelementptr8BV
T
	full_textG
E
C%93 = getelementptr inbounds <4 x float>, <4 x float>* %13, i64 %92
7<4 x float>*8B#
!
	full_text

<4 x float>* %13
%i648B

	full_text
	
i64 %92
'br8B

	full_text

br label %94
Dphi8B;
9
	full_text,
*
(%95 = phi i64 [ 0, %76 ], [ %121, %119 ]
&i648B

	full_text


i64 %121
Hphi8B?
=
	full_text0
.
,%96 = phi float [ %78, %76 ], [ %120, %119 ]
)float8B

	full_text

	float %78
*float8B

	full_text


float %120
igetelementptr8BV
T
	full_textG
E
C%97 = getelementptr inbounds <4 x float>, <4 x float>* %93, i64 %95
7<4 x float>*8B#
!
	full_text

<4 x float>* %93
%i648B

	full_text
	
i64 %95
Oload8BE
C
	full_text6
4
2%98 = load <4 x float>, <4 x float>* %97, align 16
7<4 x float>*8B#
!
	full_text

<4 x float>* %97
Rextractelement8B>
<
	full_text/
-
+%99 = extractelement <4 x float> %98, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %98
Dfcmp8B:
8
	full_text+
)
'%100 = fcmp une float %99, 0.000000e+00
)float8B

	full_text

	float %99
=br8B5
3
	full_text&
$
"br i1 %100, label %101, label %119
$i18B

	full_text
	
i1 %100
Sextractelement8B?
=
	full_text0
.
,%102 = extractelement <4 x float> %98, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %98
8fsub8B.
,
	full_text

%103 = fsub float %102, %42
*float8B

	full_text


float %102
)float8B

	full_text

	float %42
Sextractelement8B?
=
	full_text0
.
,%104 = extractelement <4 x float> %98, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %98
8fsub8B.
,
	full_text

%105 = fsub float %104, %46
*float8B

	full_text


float %104
)float8B

	full_text

	float %46
Sextractelement8B?
=
	full_text0
.
,%106 = extractelement <4 x float> %98, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %98
8fsub8B.
,
	full_text

%107 = fsub float %106, %53
*float8B

	full_text


float %106
)float8B

	full_text

	float %53
9fmul8B/
-
	full_text 

%108 = fmul float %105, %105
*float8B

	full_text


float %105
*float8B

	full_text


float %105
icall8B_
]
	full_textP
N
L%109 = tail call float @llvm.fmuladd.f32(float %103, float %103, float %108)
*float8B

	full_text


float %103
*float8B

	full_text


float %103
*float8B

	full_text


float %108
icall8B_
]
	full_textP
N
L%110 = tail call float @llvm.fmuladd.f32(float %107, float %107, float %109)
*float8B

	full_text


float %107
*float8B

	full_text


float %107
*float8B

	full_text


float %109
;fcmp8B1
/
	full_text"
 
%111 = fcmp olt float %110, %5
*float8B

	full_text


float %110
=br8B5
3
	full_text&
$
"br i1 %111, label %112, label %119
$i18B

	full_text
	
i1 %111
Bfsub8B8
6
	full_text)
'
%%113 = fsub float -0.000000e+00, %110
*float8B

	full_text


float %110
ocall8Be
c
	full_textV
T
R%114 = tail call float @llvm.fmuladd.f32(float %113, float %6, float 1.000000e+00)
*float8B

	full_text


float %113
Mcall8BC
A
	full_text4
2
0%115 = tail call float @_Z5rsqrtf(float %110) #3
*float8B

	full_text


float %110
8fmul8B.
,
	full_text

%116 = fmul float %99, %115
)float8B

	full_text

	float %99
*float8B

	full_text


float %115
9fmul8B/
-
	full_text 

%117 = fmul float %114, %116
*float8B

	full_text


float %114
*float8B

	full_text


float %116
hcall8B^
\
	full_textO
M
K%118 = tail call float @llvm.fmuladd.f32(float %117, float %114, float %96)
*float8B

	full_text


float %117
*float8B

	full_text


float %114
)float8B

	full_text

	float %96
(br8B 

	full_text

br label %119
Xphi8BO
M
	full_text@
>
<%120 = phi float [ %118, %112 ], [ %96, %101 ], [ %96, %94 ]
*float8B

	full_text


float %118
)float8B

	full_text

	float %96
)float8B

	full_text

	float %96
9add8B0
.
	full_text!

%121 = add nuw nsw i64 %95, 1
%i648B

	full_text
	
i64 %95
7icmp8B-
+
	full_text

%122 = icmp eq i64 %121, 8
&i648B

	full_text


i64 %121
<br8B4
2
	full_text%
#
!br i1 %122, label %123, label %94
$i18B

	full_text
	
i1 %122
9add8B0
.
	full_text!

%124 = add nuw nsw i64 %77, 1
%i648B

	full_text
	
i64 %77
:icmp8B0
.
	full_text!

%125 = icmp slt i64 %124, %75
&i648B

	full_text


i64 %124
%i648B

	full_text
	
i64 %75
<br8B4
2
	full_text%
#
!br i1 %125, label %76, label %126
$i18B

	full_text
	
i1 %125
Rphi8BI
G
	full_text:
8
6%127 = phi float [ 0.000000e+00, %11 ], [ %120, %123 ]
*float8B

	full_text


float %120
2shl8B)
'
	full_text

%128 = shl i64 %22, 32
%i648B

	full_text
	
i64 %22
;ashr8B1
/
	full_text"
 
%129 = ashr exact i64 %128, 32
&i648B

	full_text


i64 %128
_getelementptr8BL
J
	full_text=
;
9%130 = getelementptr inbounds float, float* %38, i64 %129
+float*8B

	full_text


float* %38
&i648B

	full_text


i64 %129
Ostore8BD
B
	full_text5
3
1store float %127, float* %130, align 4, !tbaa !13
*float8B

	full_text


float %127
,float*8B

	full_text

float* %130
$ret8B

	full_text


ret void
$i328	B

	full_text


i32 %3
(float8	B

	full_text


float %4
3
<4 x i32>*8	B!

	full_text

<4 x i32>* %10
$i328	B

	full_text


i32 %8
&i32*8	B

	full_text
	
i32* %9
$i328	B

	full_text


i32 %1
(float8	B

	full_text


float %5
(float8	B

	full_text


float %6
$i328	B

	full_text


i32 %0
6<4 x float>*8	B"
 
	full_text

<4 x float>* %2
*float*8	B

	full_text

	float* %7
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
-; undefined function B

	full_text

 
#i648	B

	full_text	

i64 4
#i328	B

	full_text	

i32 0
#i328	B

	full_text	

i32 3
#i648	B

	full_text	

i64 3
#i648	B

	full_text	

i64 7
2float8	B%
#
	full_text

float 2.500000e-01
#i648	B

	full_text	

i64 1
3float8	B&
$
	full_text

float -0.000000e+00
#i648	B

	full_text	

i64 2
#i328	B

	full_text	

i32 4
#i648	B

	full_text	

i64 9
2float8	B%
#
	full_text

float 0.000000e+00
#i648	B

	full_text	

i64 0
2float8	B%
#
	full_text

float 1.000000e+00
#i648	B

	full_text	

i64 8
$i648	B

	full_text


i64 32
#i328	B

	full_text	

i32 2
#i328	B

	full_text	

i32 1        		 
 
 

                      !" !# !! $$ %& %% '( ') '' *+ ** ,- ,, ./ .. 01 00 23 24 22 56 55 78 79 77 :; :: <= << >? >> @A @B @@ CD CC EF EE GG HI HH JK JJ LM LN LL OP OQ OO RS RR TU TT VW VV XY XX Z[ ZZ \] \\ ^_ ^^ `a `` bc bb de dd fg ff hi hh jk jj lm ll no nn pq pp rs rr tu tt vw vv xy xx zz {| {{ }~ }Ä  Å
É ÇÇ Ñ
Ö ÑÑ Ü
á ÜÜ àâ àà äã ää åç å
é åå èê èè ëí ë
ì ëë îï îî ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü ûû †° †
¢ †† £§ ££ •¶ •• ß® ß
© ßß ™
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π∫ πº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À  
Ã    ÕŒ Õ
œ Õ
– ÕÕ —“ —
” —
‘ —— ’÷ ’’ ◊ÿ ◊
⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á Â
Ë ÂÂ ÈÎ Í
Ï Í
Ì ÍÍ ÓÔ ÓÓ Ò  ÚÛ Úı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà 	â <	â E	â T	â Z	â f	â rä Üã ã Gå z
ç ô
é ’
è €
ê ûë í ,    	 
           " #$ &! (% )' +* -$ /. 1, 30 4% 65 8 97 ;: = ?> A	 B@ DC FG I. KJ MH NL P QO SR U5 WV YX [Z ]\ _^ a> cb ed gf ih kj mG on qp sr ut wv yz |{ ~z ÄÙ ÉÍ ÖÇ áÜ âà ãä ç` éà êè íl ìà ïî óx òñ öë úô ùõ üå °û ¢† §£ ¶ ®• ©Ó ¨Ñ ÆÍ Øß ±´ ≤∞ ¥≥ ∂µ ∏∑ ∫≥ ºª æ< ø≥ ¡¿ √E ƒ≥ ∆≈ »T …¬ À¬ ÃΩ ŒΩ œ  –« “« ”Õ ‘— ÷’ ÿ— ⁄Ÿ ‹— ﬁµ ‡› ·€ „ﬂ ‰‚ Ê€ Á≠ ËÂ Î≠ Ï≠ Ì´ ÔÓ Ò ÛÇ ıÙ ˜ ¯ˆ ˙Í ¸ ˛˝ Ä2 Çˇ É˚ ÖÅ Ü} } ˚Å Ç™ ´π ªπ Í◊ Ÿ◊ ÍÚ ÙÚ ´È Í˘ Ç˘ ˚ ìì ïï ôô îî ññ á óó òò ïï  ìì $ ññ $ îî — òò —Â òò Â îî  ìì Õ òò Õj óó j€ òò €	 ìì 	 ïï ^ óó ^› ôô › ññ v óó v	ö V	ö bõ õ õ õ $	õ {	ú G
ú £	ù .	ù 5	ù >
ù µ	û 0	ü \	ü h	ü t	† J
† è
† ¿
† Ó
† Ù° Ÿ	¢ 	¢ %
¢ î
¢ ≈	£ n	§ *• Ñ
• ∑• ˚¶ Ç
¶ ä¶ ´
¶ ª
ß €
® 
© ˝
© ˇ™ ´ ´ 	´ ´ "!
opencl_cutoff_potential_lattice"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z14get_num_groupsj"
_Z12get_group_idj"
	_Z5floorf"
llvm.fmuladd.f32"
	_Z5rsqrtf*õ
"opencl_cutoff_potential_lattice.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å
 
transfer_bytes_log1p
«åA

devmap_label


transfer_bytes
¥¶ñ

wgsize_log1p
«åA

wgsize
Ä