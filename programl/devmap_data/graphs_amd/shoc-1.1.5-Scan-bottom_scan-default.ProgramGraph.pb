

[external]
cstoreBZ
X
	full_textK
I
Gstore float 0.000000e+00, float* @bottom_scan.s_seed, align 4, !tbaa !8
DbitcastB9
7
	full_text*
(
&%6 = bitcast float* %0 to <4 x float>*
DbitcastB9
7
	full_text*
(
&%7 = bitcast float* %2 to <4 x float>*
.sdivB&
$
	full_text

%8 = sdiv i32 %3, 4
2sextB*
(
	full_text

%9 = sext i32 %8 to i64
"i32B

	full_text


i32 %8
McallBE
C
	full_text6
4
2%10 = tail call i64 @_Z14get_num_groupsj(i32 0) #4
1udivB)
'
	full_text

%11 = udiv i64 %9, %10
"i64B

	full_text


i64 %9
#i64B

	full_text
	
i64 %10
KcallBC
A
	full_text4
2
0%12 = tail call i64 @_Z12get_group_idj(i32 0) #4
/shlB(
&
	full_text

%13 = shl i64 %11, 32
#i64B

	full_text
	
i64 %11
7ashrB/
-
	full_text 

%14 = ashr exact i64 %13, 32
#i64B

	full_text
	
i64 %13
0mulB)
'
	full_text

%15 = mul i64 %14, %12
#i64B

	full_text
	
i64 %14
#i64B

	full_text
	
i64 %12
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
/addB(
&
	full_text

%17 = add i64 %10, -1
#i64B

	full_text
	
i64 %10
5icmpB-
+
	full_text

%18 = icmp eq i64 %12, %17
#i64B

	full_text
	
i64 %12
#i64B

	full_text
	
i64 %17
6truncB-
+
	full_text

%19 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
4addB-
+
	full_text

%20 = add nsw i32 %16, %19
#i32B

	full_text
	
i32 %16
#i32B

	full_text
	
i32 %19
AselectB7
5
	full_text(
&
$%21 = select i1 %18, i32 %8, i32 %20
!i1B

	full_text


i1 %18
"i32B

	full_text


i32 %8
#i32B

	full_text
	
i32 %20
KcallBC
A
	full_text4
2
0%22 = tail call i64 @_Z12get_local_idj(i32 0) #4
6icmpB.
,
	full_text

%23 = icmp ugt i32 %21, %16
#i32B

	full_text
	
i32 %21
#i32B

	full_text
	
i32 %16
9brB3
1
	full_text$
"
 br i1 %23, label %24, label %113
!i1B

	full_text


i1 %23
\getelementptr8BI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %1, i64 %12
%i648B

	full_text
	
i64 %12
9and8B0
.
	full_text!

%26 = and i64 %15, 4294967295
%i648B

	full_text
	
i64 %15
2add8B)
'
	full_text

%27 = add i64 %26, %22
%i648B

	full_text
	
i64 %26
%i648B

	full_text
	
i64 %22
Lload8BB
@
	full_text3
1
/%28 = load float, float* %25, align 4, !tbaa !8
+float*8B

	full_text


float* %25
8trunc8B-
+
	full_text

%29 = trunc i64 %27 to i32
%i648B

	full_text
	
i64 %27
8icmp8B.
,
	full_text

%30 = icmp sgt i32 %21, %29
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %29
:br8B2
0
	full_text#
!
br i1 %30, label %31, label %36
#i18B

	full_text


i1 %30
1shl8B(
&
	full_text

%32 = shl i64 %27, 32
%i648B

	full_text
	
i64 %27
9ashr8B/
-
	full_text 

%33 = ashr exact i64 %32, 32
%i648B

	full_text
	
i64 %32
hgetelementptr8BU
S
	full_textF
D
B%34 = getelementptr inbounds <4 x float>, <4 x float>* %6, i64 %33
6<4 x float>*8B"
 
	full_text

<4 x float>* %6
%i648B

	full_text
	
i64 %33
Zload8BP
N
	full_textA
?
=%35 = load <4 x float>, <4 x float>* %34, align 16, !tbaa !12
7<4 x float>*8B#
!
	full_text

<4 x float>* %34
'br8B

	full_text

br label %36
Xphi8BO
M
	full_text@
>
<%37 = phi <4 x float> [ %35, %31 ], [ zeroinitializer, %24 ]
5<4 x float>8B"
 
	full_text

<4 x float> %35
Rextractelement8B>
<
	full_text/
-
+%38 = extractelement <4 x float> %37, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %37
Rextractelement8B>
<
	full_text/
-
+%39 = extractelement <4 x float> %37, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %37
6fadd8B,
*
	full_text

%40 = fadd float %39, %38
)float8B

	full_text

	float %39
)float8B

	full_text

	float %38
Rextractelement8B>
<
	full_text/
-
+%41 = extractelement <4 x float> %37, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %37
6fadd8B,
*
	full_text

%42 = fadd float %41, %40
)float8B

	full_text

	float %41
)float8B

	full_text

	float %40
Rextractelement8B>
<
	full_text/
-
+%43 = extractelement <4 x float> %37, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %37
6fadd8B,
*
	full_text

%44 = fadd float %43, %42
)float8B

	full_text

	float %43
)float8B

	full_text

	float %42
`call8BV
T
	full_textG
E
C%45 = tail call float @scanLocalMem(float %44, float* %4, i32 1) #5
)float8B

	full_text

	float %44
6fadd8B,
*
	full_text

%46 = fadd float %28, %45
)float8B

	full_text

	float %28
)float8B

	full_text

	float %45
6fadd8B,
*
	full_text

%47 = fadd float %46, %44
)float8B

	full_text

	float %46
)float8B

	full_text

	float %44
:br8B2
0
	full_text#
!
br i1 %30, label %48, label %59
#i18B

	full_text


i1 %30
6fadd8B,
*
	full_text

%49 = fadd float %38, %46
)float8B

	full_text

	float %38
)float8B

	full_text

	float %46
]insertelement8BJ
H
	full_text;
9
7%50 = insertelement <4 x float> undef, float %49, i64 0
)float8B

	full_text

	float %49
6fadd8B,
*
	full_text

%51 = fadd float %40, %46
)float8B

	full_text

	float %40
)float8B

	full_text

	float %46
[insertelement8BH
F
	full_text9
7
5%52 = insertelement <4 x float> %50, float %51, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %50
)float8B

	full_text

	float %51
6fadd8B,
*
	full_text

%53 = fadd float %46, %42
)float8B

	full_text

	float %46
)float8B

	full_text

	float %42
[insertelement8BH
F
	full_text9
7
5%54 = insertelement <4 x float> %52, float %53, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %52
)float8B

	full_text

	float %53
[insertelement8BH
F
	full_text9
7
5%55 = insertelement <4 x float> %54, float %47, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %54
)float8B

	full_text

	float %47
1shl8B(
&
	full_text

%56 = shl i64 %27, 32
%i648B

	full_text
	
i64 %27
9ashr8B/
-
	full_text 

%57 = ashr exact i64 %56, 32
%i648B

	full_text
	
i64 %56
hgetelementptr8BU
S
	full_textF
D
B%58 = getelementptr inbounds <4 x float>, <4 x float>* %7, i64 %57
6<4 x float>*8B"
 
	full_text

<4 x float>* %7
%i648B

	full_text
	
i64 %57
Zstore8BO
M
	full_text@
>
<store <4 x float> %55, <4 x float>* %58, align 16, !tbaa !12
5<4 x float>8B"
 
	full_text

<4 x float> %55
7<4 x float>*8B#
!
	full_text

<4 x float>* %58
'br8B

	full_text

br label %59
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
Ocall8BE
C
	full_text6
4
2%60 = tail call i64 @_Z14get_local_sizej(i32 0) #4
1add8B(
&
	full_text

%61 = add i64 %60, -1
%i648B

	full_text
	
i64 %60
7icmp8B-
+
	full_text

%62 = icmp eq i64 %22, %61
%i648B

	full_text
	
i64 %22
%i648B

	full_text
	
i64 %61
:br8B2
0
	full_text#
!
br i1 %62, label %63, label %64
#i18B

	full_text


i1 %62
\store8BQ
O
	full_textB
@
>store float %47, float* @bottom_scan.s_seed, align 4, !tbaa !8
)float8B

	full_text

	float %47
'br8B

	full_text

br label %64
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
2add8B)
'
	full_text

%65 = add i64 %60, %26
%i648B

	full_text
	
i64 %60
%i648B

	full_text
	
i64 %26
8trunc8B-
+
	full_text

%66 = trunc i64 %65 to i32
%i648B

	full_text
	
i64 %65
8icmp8B.
,
	full_text

%67 = icmp ugt i32 %21, %66
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %66
;br8B3
1
	full_text$
"
 br i1 %67, label %68, label %113
#i18B

	full_text


i1 %67
'br8B

	full_text

br label %69
Ephi8	B<
:
	full_text-
+
)%70 = phi i64 [ %74, %108 ], [ %27, %68 ]
%i648	B

	full_text
	
i64 %74
%i648	B

	full_text
	
i64 %27
Fphi8	B=
;
	full_text.
,
*%71 = phi i64 [ %110, %108 ], [ %65, %68 ]
&i648	B

	full_text


i64 %110
%i648	B

	full_text
	
i64 %65
1shl8	B(
&
	full_text

%72 = shl i64 %70, 32
%i648	B

	full_text
	
i64 %70
9ashr8	B/
-
	full_text 

%73 = ashr exact i64 %72, 32
%i648	B

	full_text
	
i64 %72
2add8	B)
'
	full_text

%74 = add i64 %60, %73
%i648	B

	full_text
	
i64 %60
%i648	B

	full_text
	
i64 %73
\load8	BR
P
	full_textC
A
?%75 = load float, float* @bottom_scan.s_seed, align 4, !tbaa !8
8trunc8	B-
+
	full_text

%76 = trunc i64 %74 to i32
%i648	B

	full_text
	
i64 %74
8icmp8	B.
,
	full_text

%77 = icmp sgt i32 %21, %76
%i328	B

	full_text
	
i32 %21
%i328	B

	full_text
	
i32 %76
:br8	B2
0
	full_text#
!
br i1 %77, label %78, label %83
#i18	B

	full_text


i1 %77
1shl8
B(
&
	full_text

%79 = shl i64 %74, 32
%i648
B

	full_text
	
i64 %74
9ashr8
B/
-
	full_text 

%80 = ashr exact i64 %79, 32
%i648
B

	full_text
	
i64 %79
hgetelementptr8
BU
S
	full_textF
D
B%81 = getelementptr inbounds <4 x float>, <4 x float>* %6, i64 %80
6<4 x float>*8
B"
 
	full_text

<4 x float>* %6
%i648
B

	full_text
	
i64 %80
Zload8
BP
N
	full_textA
?
=%82 = load <4 x float>, <4 x float>* %81, align 16, !tbaa !12
7<4 x float>*8
B#
!
	full_text

<4 x float>* %81
'br8
B

	full_text

br label %83
Xphi8BO
M
	full_text@
>
<%84 = phi <4 x float> [ %82, %78 ], [ zeroinitializer, %69 ]
5<4 x float>8B"
 
	full_text

<4 x float> %82
Rextractelement8B>
<
	full_text/
-
+%85 = extractelement <4 x float> %84, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %84
Rextractelement8B>
<
	full_text/
-
+%86 = extractelement <4 x float> %84, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %84
6fadd8B,
*
	full_text

%87 = fadd float %86, %85
)float8B

	full_text

	float %86
)float8B

	full_text

	float %85
Rextractelement8B>
<
	full_text/
-
+%88 = extractelement <4 x float> %84, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %84
6fadd8B,
*
	full_text

%89 = fadd float %88, %87
)float8B

	full_text

	float %88
)float8B

	full_text

	float %87
Rextractelement8B>
<
	full_text/
-
+%90 = extractelement <4 x float> %84, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %84
6fadd8B,
*
	full_text

%91 = fadd float %90, %89
)float8B

	full_text

	float %90
)float8B

	full_text

	float %89
`call8BV
T
	full_textG
E
C%92 = tail call float @scanLocalMem(float %91, float* %4, i32 1) #5
)float8B

	full_text

	float %91
6fadd8B,
*
	full_text

%93 = fadd float %75, %92
)float8B

	full_text

	float %75
)float8B

	full_text

	float %92
6fadd8B,
*
	full_text

%94 = fadd float %93, %91
)float8B

	full_text

	float %93
)float8B

	full_text

	float %91
;br8B3
1
	full_text$
"
 br i1 %77, label %95, label %106
#i18B

	full_text


i1 %77
6fadd8B,
*
	full_text

%96 = fadd float %85, %93
)float8B

	full_text

	float %85
)float8B

	full_text

	float %93
]insertelement8BJ
H
	full_text;
9
7%97 = insertelement <4 x float> undef, float %96, i64 0
)float8B

	full_text

	float %96
6fadd8B,
*
	full_text

%98 = fadd float %87, %93
)float8B

	full_text

	float %87
)float8B

	full_text

	float %93
[insertelement8BH
F
	full_text9
7
5%99 = insertelement <4 x float> %97, float %98, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %97
)float8B

	full_text

	float %98
7fadd8B-
+
	full_text

%100 = fadd float %93, %89
)float8B

	full_text

	float %93
)float8B

	full_text

	float %89
]insertelement8BJ
H
	full_text;
9
7%101 = insertelement <4 x float> %99, float %100, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %99
*float8B

	full_text


float %100
]insertelement8BJ
H
	full_text;
9
7%102 = insertelement <4 x float> %101, float %94, i64 3
6<4 x float>8B#
!
	full_text

<4 x float> %101
)float8B

	full_text

	float %94
2shl8B)
'
	full_text

%103 = shl i64 %74, 32
%i648B

	full_text
	
i64 %74
;ashr8B1
/
	full_text"
 
%104 = ashr exact i64 %103, 32
&i648B

	full_text


i64 %103
jgetelementptr8BW
U
	full_textH
F
D%105 = getelementptr inbounds <4 x float>, <4 x float>* %7, i64 %104
6<4 x float>*8B"
 
	full_text

<4 x float>* %7
&i648B

	full_text


i64 %104
\store8BQ
O
	full_textB
@
>store <4 x float> %102, <4 x float>* %105, align 16, !tbaa !12
6<4 x float>8B#
!
	full_text

<4 x float> %102
8<4 x float>*8B$
"
	full_text

<4 x float>* %105
(br8B 

	full_text

br label %106
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
<br8B4
2
	full_text%
#
!br i1 %62, label %107, label %108
#i18B

	full_text


i1 %62
\store8BQ
O
	full_textB
@
>store float %94, float* @bottom_scan.s_seed, align 4, !tbaa !8
)float8B

	full_text

	float %94
(br8B 

	full_text

br label %108
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #6
:and8B1
/
	full_text"
 
%109 = and i64 %71, 4294967295
%i648B

	full_text
	
i64 %71
4add8B+
)
	full_text

%110 = add i64 %60, %109
%i648B

	full_text
	
i64 %60
&i648B

	full_text


i64 %109
:trunc8B/
-
	full_text 

%111 = trunc i64 %110 to i32
&i648B

	full_text


i64 %110
:icmp8B0
.
	full_text!

%112 = icmp ugt i32 %21, %111
%i328B

	full_text
	
i32 %21
&i328B

	full_text


i32 %111
Lbr8BD
B
	full_text5
3
1br i1 %112, label %69, label %113, !llvm.loop !13
$i18B

	full_text
	
i1 %112
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %0
$i328B

	full_text


i32 %3
*float*8B

	full_text

	float* %2
*float*8B

	full_text

	float* %4
*float*8B
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
,i648B!

	full_text

i64 4294967295
A<4 x float>8B.
,
	full_text

<4 x float> zeroinitializer
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 2
2float8B%
#
	full_text

float 0.000000e+00
7<4 x float>8B$
"
	full_text

<4 x float> undef
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 0
hfloat*8BZ
X
	full_textK
I
G@bottom_scan.s_seed = internal unnamed_addr global float undef, align 4
#i328B

	full_text	

i32 4
$i648B

	full_text


i64 -1       	 
                        ! "  ## $% $& $$ '( '* )) +, ++ -. -/ -- 01 00 23 22 45 46 44 78 7: 99 ;< ;; => =? == @A @@ BD CC EF EE GH GG IJ IK II LM LL NO NP NN QR QQ ST SU SS VW VV XY XZ XX [\ [] [[ ^_ ^a `b `` cd cc ef eg ee hi hj hh kl km kk no np nn qr qs qq tu tt vw vv xy xz xx {| {} {{ ~ ÄÄ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá Üâ àà äã åç å
é åå èê èè ëí ë
ì ëë îï îò ó
ô óó öõ ö
ú öö ùû ùù ü† üü °¢ °
£ °° §§ •¶ •• ß® ß
© ßß ™´ ™≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —‘ ”
’ ”” ÷
◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ ÛÙ Ûˆ ıı ˜¯ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÜ á à 	â V
â …ä )  	 
              ! " % &$ ( * ,+ .# /) 1- 3 52 64 8- :9 < >; ?= A@ DC FC HG JE KC ML OI PC RQ TN US W0 YV ZX \S ]4 _E aX b` dI fX gc ie jX lN mh ok pn r[ s- ut w yv zq |x }Ä Ç# ÑÅ ÖÉ á[ âÄ ç+ éå ê íè ìë ï° ò- ô˚ õå úó ûù †Ä ¢ü £° ¶ ®• ©ß ´° ≠¨ Ø ±Æ ≤∞ ¥≥ ∑∂ π∂ ª∫ Ω∏ æ∂ ¿ø ¬º √∂ ≈ƒ «¡ »∆  § Ã… ÕÀ œ∆ –ß “∏ ‘À ’” ◊º ŸÀ ⁄÷ ‹ÿ ›À ﬂ¡ ‡€ ‚ﬁ „· ÂŒ Ê° ËÁ Í ÏÈ Ì‰ ÔÎ É ÙŒ ˆö ˙Ä ¸˘ ˝˚ ˇ Å˛ ÇÄ Ñ' )' Ö7 97 CB C^ `^ ~ Ü àÜ ãä ãî ñî Öñ ó™ ¨™ ∂µ ∂— ”— ÚÒ ÚÛ ıÛ ¯˜ ¯É óÉ Ö ãã Ö éé èè åå çç êê èè Ä êê Ä… éé …Ú èè Ú åå ã èè ãV éé V ãã # çç #¯ èè ¯	ë +
ë ˘	í C
í ∂	ì E	ì c
ì ∏
ì ÷	î G	î h
î ∫
î €	ï Vï ï ã
ï …ï Úï ¯	ñ 	ñ 	ñ 9	ñ ;	ñ t	ñ v
ñ ù
ñ ü
ñ ¨
ñ Æ
ñ Á
ñ È	ó L	ó n
ó ø
ó ·ò ô cô ÷	ö Q	ö q
ö ƒ
ö ‰õ õ õ #õ Ä	ú 
ú àú §
ú ı	ù 	û 
û Å"
bottom_scan"
_Z14get_num_groupsj"
_Z12get_group_idj"
_Z12get_local_idj"
scanLocalMem"
_Z7barrierj"
_Z14get_local_sizej*ó
shoc-1.1.5-Scan-bottom_scan.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

devmap_label


wgsize
Ä

transfer_bytes
ÄÜÄ
 
transfer_bytes_log1p
cA

wgsize_log1p
cA