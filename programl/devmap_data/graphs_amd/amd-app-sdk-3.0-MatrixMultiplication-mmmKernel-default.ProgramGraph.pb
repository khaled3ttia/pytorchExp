
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
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #3
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
/lshrB'
%
	full_text

%10 = lshr i32 %4, 2
2icmpB*
(
	full_text

%11 = icmp eq i32 %3, 0
-shlB&
$
	full_text

%12 = shl i32 %9, 2
"i32B

	full_text


i32 %9
8brB2
0
	full_text#
!
br i1 %11, label %13, label %17
!i1B

	full_text


i1 %11
.or8B&
$
	full_text

%14 = or i32 %12, 1
%i328B

	full_text
	
i32 %12
.or8B&
$
	full_text

%15 = or i32 %12, 2
%i328B

	full_text
	
i32 %12
.or8B&
$
	full_text

%16 = or i32 %12, 3
%i328B

	full_text
	
i32 %12
'br8B

	full_text

br label %27
1lshr8B'
%
	full_text

%18 = lshr i32 %3, 2
2mul8B)
'
	full_text

%19 = mul i32 %12, %18
%i328B

	full_text
	
i32 %12
%i328B

	full_text
	
i32 %18
.or8B&
$
	full_text

%20 = or i32 %12, 1
%i328B

	full_text
	
i32 %12
2mul8B)
'
	full_text

%21 = mul i32 %20, %18
%i328B

	full_text
	
i32 %20
%i328B

	full_text
	
i32 %18
.or8B&
$
	full_text

%22 = or i32 %12, 2
%i328B

	full_text
	
i32 %12
2mul8B)
'
	full_text

%23 = mul i32 %22, %18
%i328B

	full_text
	
i32 %22
%i328B

	full_text
	
i32 %18
.or8B&
$
	full_text

%24 = or i32 %12, 3
%i328B

	full_text
	
i32 %12
2mul8B)
'
	full_text

%25 = mul i32 %24, %18
%i328B

	full_text
	
i32 %24
%i328B

	full_text
	
i32 %18
5zext8B+
)
	full_text

%26 = zext i32 %3 to i64
'br8B

	full_text

br label %51
Dphi8B;
9
	full_text,
*
(%28 = phi i32 [ %16, %13 ], [ %24, %51 ]
%i328B

	full_text
	
i32 %16
%i328B

	full_text
	
i32 %24
Dphi8B;
9
	full_text,
*
(%29 = phi i32 [ %15, %13 ], [ %22, %51 ]
%i328B

	full_text
	
i32 %15
%i328B

	full_text
	
i32 %22
Dphi8B;
9
	full_text,
*
(%30 = phi i32 [ %14, %13 ], [ %20, %51 ]
%i328B

	full_text
	
i32 %14
%i328B

	full_text
	
i32 %20
Yphi8BP
N
	full_textA
?
=%31 = phi <4 x float> [ zeroinitializer, %13 ], [ %245, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %245
Yphi8BP
N
	full_textA
?
=%32 = phi <4 x float> [ zeroinitializer, %13 ], [ %213, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %213
Yphi8BP
N
	full_textA
?
=%33 = phi <4 x float> [ zeroinitializer, %13 ], [ %181, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %181
Yphi8BP
N
	full_textA
?
=%34 = phi <4 x float> [ zeroinitializer, %13 ], [ %149, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %149
2mul8B)
'
	full_text

%35 = mul i32 %12, %10
%i328B

	full_text
	
i32 %12
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%36 = add i32 %35, %7
%i328B

	full_text
	
i32 %35
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%37 = zext i32 %36 to i64
%i328B

	full_text
	
i32 %36
hgetelementptr8BU
S
	full_textF
D
B%38 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %37
%i648B

	full_text
	
i64 %37
Ystore8BN
L
	full_text?
=
;store <4 x float> %34, <4 x float>* %38, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %34
7<4 x float>*8B#
!
	full_text

<4 x float>* %38
2mul8B)
'
	full_text

%39 = mul i32 %30, %10
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%40 = add i32 %39, %7
%i328B

	full_text
	
i32 %39
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%41 = zext i32 %40 to i64
%i328B

	full_text
	
i32 %40
hgetelementptr8BU
S
	full_textF
D
B%42 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %41
%i648B

	full_text
	
i64 %41
Ystore8BN
L
	full_text?
=
;store <4 x float> %33, <4 x float>* %42, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %33
7<4 x float>*8B#
!
	full_text

<4 x float>* %42
2mul8B)
'
	full_text

%43 = mul i32 %29, %10
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%44 = add i32 %43, %7
%i328B

	full_text
	
i32 %43
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%45 = zext i32 %44 to i64
%i328B

	full_text
	
i32 %44
hgetelementptr8BU
S
	full_textF
D
B%46 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %45
%i648B

	full_text
	
i64 %45
Ystore8BN
L
	full_text?
=
;store <4 x float> %32, <4 x float>* %46, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %32
7<4 x float>*8B#
!
	full_text

<4 x float>* %46
2mul8B)
'
	full_text

%47 = mul i32 %28, %10
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%48 = add i32 %47, %7
%i328B

	full_text
	
i32 %47
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%49 = zext i32 %48 to i64
%i328B

	full_text
	
i32 %48
hgetelementptr8BU
S
	full_textF
D
B%50 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %49
%i648B

	full_text
	
i64 %49
Ystore8BN
L
	full_text?
=
;store <4 x float> %31, <4 x float>* %50, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %31
7<4 x float>*8B#
!
	full_text

<4 x float>* %50
$ret8B

	full_text


ret void
Cphi8B:
8
	full_text+
)
'%52 = phi i64 [ 0, %17 ], [ %246, %51 ]
&i648B

	full_text


i64 %246
Yphi8BP
N
	full_textA
?
=%53 = phi <4 x float> [ zeroinitializer, %17 ], [ %149, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %149
Yphi8BP
N
	full_textA
?
=%54 = phi <4 x float> [ zeroinitializer, %17 ], [ %181, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %181
Yphi8BP
N
	full_textA
?
=%55 = phi <4 x float> [ zeroinitializer, %17 ], [ %213, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %213
Yphi8BP
N
	full_textA
?
=%56 = phi <4 x float> [ zeroinitializer, %17 ], [ %245, %51 ]
6<4 x float>8B#
!
	full_text

<4 x float> %245
8lshr8B.
,
	full_text

%57 = lshr exact i64 %52, 2
%i648B

	full_text
	
i64 %52
8trunc8B-
+
	full_text

%58 = trunc i64 %57 to i32
%i648B

	full_text
	
i64 %57
2add8B)
'
	full_text

%59 = add i32 %19, %58
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %58
6zext8B,
*
	full_text

%60 = zext i32 %59 to i64
%i328B

	full_text
	
i32 %59
hgetelementptr8BU
S
	full_textF
D
B%61 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %60
%i648B

	full_text
	
i64 %60
Yload8BO
M
	full_text@
>
<%62 = load <4 x float>, <4 x float>* %61, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %61
2add8B)
'
	full_text

%63 = add i32 %21, %58
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %58
6zext8B,
*
	full_text

%64 = zext i32 %63 to i64
%i328B

	full_text
	
i32 %63
hgetelementptr8BU
S
	full_textF
D
B%65 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %64
%i648B

	full_text
	
i64 %64
Yload8BO
M
	full_text@
>
<%66 = load <4 x float>, <4 x float>* %65, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %65
2add8B)
'
	full_text

%67 = add i32 %23, %58
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %58
6zext8B,
*
	full_text

%68 = zext i32 %67 to i64
%i328B

	full_text
	
i32 %67
hgetelementptr8BU
S
	full_textF
D
B%69 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %68
%i648B

	full_text
	
i64 %68
Yload8BO
M
	full_text@
>
<%70 = load <4 x float>, <4 x float>* %69, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %69
2add8B)
'
	full_text

%71 = add i32 %25, %58
%i328B

	full_text
	
i32 %25
%i328B

	full_text
	
i32 %58
6zext8B,
*
	full_text

%72 = zext i32 %71 to i64
%i328B

	full_text
	
i32 %71
hgetelementptr8BU
S
	full_textF
D
B%73 = getelementptr inbounds <4 x float>, <4 x float>* %0, i64 %72
%i648B

	full_text
	
i64 %72
Yload8BO
M
	full_text@
>
<%74 = load <4 x float>, <4 x float>* %73, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %73
8trunc8B-
+
	full_text

%75 = trunc i64 %52 to i32
%i648B

	full_text
	
i64 %52
2mul8B)
'
	full_text

%76 = mul i32 %10, %75
%i328B

	full_text
	
i32 %10
%i328B

	full_text
	
i32 %75
1add8B(
&
	full_text

%77 = add i32 %76, %7
%i328B

	full_text
	
i32 %76
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%78 = zext i32 %77 to i64
%i328B

	full_text
	
i32 %77
hgetelementptr8BU
S
	full_textF
D
B%79 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %78
%i648B

	full_text
	
i64 %78
Yload8BO
M
	full_text@
>
<%80 = load <4 x float>, <4 x float>* %79, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %79
8trunc8B-
+
	full_text

%81 = trunc i64 %52 to i32
%i648B

	full_text
	
i64 %52
.or8B&
$
	full_text

%82 = or i32 %81, 1
%i328B

	full_text
	
i32 %81
2mul8B)
'
	full_text

%83 = mul i32 %82, %10
%i328B

	full_text
	
i32 %82
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%84 = add i32 %83, %7
%i328B

	full_text
	
i32 %83
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%85 = zext i32 %84 to i64
%i328B

	full_text
	
i32 %84
hgetelementptr8BU
S
	full_textF
D
B%86 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %85
%i648B

	full_text
	
i64 %85
Yload8BO
M
	full_text@
>
<%87 = load <4 x float>, <4 x float>* %86, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %86
8trunc8B-
+
	full_text

%88 = trunc i64 %52 to i32
%i648B

	full_text
	
i64 %52
.or8B&
$
	full_text

%89 = or i32 %88, 2
%i328B

	full_text
	
i32 %88
2mul8B)
'
	full_text

%90 = mul i32 %89, %10
%i328B

	full_text
	
i32 %89
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%91 = add i32 %90, %7
%i328B

	full_text
	
i32 %90
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%92 = zext i32 %91 to i64
%i328B

	full_text
	
i32 %91
hgetelementptr8BU
S
	full_textF
D
B%93 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %92
%i648B

	full_text
	
i64 %92
Yload8BO
M
	full_text@
>
<%94 = load <4 x float>, <4 x float>* %93, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %93
8trunc8B-
+
	full_text

%95 = trunc i64 %52 to i32
%i648B

	full_text
	
i64 %52
.or8B&
$
	full_text

%96 = or i32 %95, 3
%i328B

	full_text
	
i32 %95
2mul8B)
'
	full_text

%97 = mul i32 %96, %10
%i328B

	full_text
	
i32 %96
%i328B

	full_text
	
i32 %10
1add8B(
&
	full_text

%98 = add i32 %97, %7
%i328B

	full_text
	
i32 %97
$i328B

	full_text


i32 %7
6zext8B,
*
	full_text

%99 = zext i32 %98 to i64
%i328B

	full_text
	
i32 %98
igetelementptr8BV
T
	full_textG
E
C%100 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %99
%i648B

	full_text
	
i64 %99
[load8BQ
O
	full_textB
@
>%101 = load <4 x float>, <4 x float>* %100, align 16, !tbaa !9
8<4 x float>*8B$
"
	full_text

<4 x float>* %100
Sextractelement8B?
=
	full_text0
.
,%102 = extractelement <4 x float> %62, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %62
Sextractelement8B?
=
	full_text0
.
,%103 = extractelement <4 x float> %80, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %80
Sextractelement8B?
=
	full_text0
.
,%104 = extractelement <4 x float> %62, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %62
Sextractelement8B?
=
	full_text0
.
,%105 = extractelement <4 x float> %87, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %87
9fmul8B/
-
	full_text 

%106 = fmul float %104, %105
*float8B

	full_text


float %104
*float8B

	full_text


float %105
icall8B_
]
	full_textP
N
L%107 = tail call float @llvm.fmuladd.f32(float %102, float %103, float %106)
*float8B

	full_text


float %102
*float8B

	full_text


float %103
*float8B

	full_text


float %106
Sextractelement8B?
=
	full_text0
.
,%108 = extractelement <4 x float> %62, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %62
Sextractelement8B?
=
	full_text0
.
,%109 = extractelement <4 x float> %94, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %94
icall8B_
]
	full_textP
N
L%110 = tail call float @llvm.fmuladd.f32(float %108, float %109, float %107)
*float8B

	full_text


float %108
*float8B

	full_text


float %109
*float8B

	full_text


float %107
Sextractelement8B?
=
	full_text0
.
,%111 = extractelement <4 x float> %62, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %62
Textractelement8B@
>
	full_text1
/
-%112 = extractelement <4 x float> %101, i64 0
6<4 x float>8B#
!
	full_text

<4 x float> %101
icall8B_
]
	full_textP
N
L%113 = tail call float @llvm.fmuladd.f32(float %111, float %112, float %110)
*float8B

	full_text


float %111
*float8B

	full_text


float %112
*float8B

	full_text


float %110
Sextractelement8B?
=
	full_text0
.
,%114 = extractelement <4 x float> %53, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %53
9fadd8B/
-
	full_text 

%115 = fadd float %114, %113
*float8B

	full_text


float %114
*float8B

	full_text


float %113
_insertelement8BL
J
	full_text=
;
9%116 = insertelement <4 x float> undef, float %115, i64 0
*float8B

	full_text


float %115
Sextractelement8B?
=
	full_text0
.
,%117 = extractelement <4 x float> %80, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %80
Sextractelement8B?
=
	full_text0
.
,%118 = extractelement <4 x float> %87, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %87
9fmul8B/
-
	full_text 

%119 = fmul float %104, %118
*float8B

	full_text


float %104
*float8B

	full_text


float %118
icall8B_
]
	full_textP
N
L%120 = tail call float @llvm.fmuladd.f32(float %102, float %117, float %119)
*float8B

	full_text


float %102
*float8B

	full_text


float %117
*float8B

	full_text


float %119
Sextractelement8B?
=
	full_text0
.
,%121 = extractelement <4 x float> %94, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %94
icall8B_
]
	full_textP
N
L%122 = tail call float @llvm.fmuladd.f32(float %108, float %121, float %120)
*float8B

	full_text


float %108
*float8B

	full_text


float %121
*float8B

	full_text


float %120
Textractelement8B@
>
	full_text1
/
-%123 = extractelement <4 x float> %101, i64 1
6<4 x float>8B#
!
	full_text

<4 x float> %101
icall8B_
]
	full_textP
N
L%124 = tail call float @llvm.fmuladd.f32(float %111, float %123, float %122)
*float8B

	full_text


float %111
*float8B

	full_text


float %123
*float8B

	full_text


float %122
Sextractelement8B?
=
	full_text0
.
,%125 = extractelement <4 x float> %53, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %53
9fadd8B/
-
	full_text 

%126 = fadd float %125, %124
*float8B

	full_text


float %125
*float8B

	full_text


float %124
^insertelement8BK
I
	full_text<
:
8%127 = insertelement <4 x float> %116, float %126, i64 1
6<4 x float>8B#
!
	full_text

<4 x float> %116
*float8B

	full_text


float %126
Sextractelement8B?
=
	full_text0
.
,%128 = extractelement <4 x float> %80, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %80
Sextractelement8B?
=
	full_text0
.
,%129 = extractelement <4 x float> %87, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %87
9fmul8B/
-
	full_text 

%130 = fmul float %104, %129
*float8B

	full_text


float %104
*float8B

	full_text


float %129
icall8B_
]
	full_textP
N
L%131 = tail call float @llvm.fmuladd.f32(float %102, float %128, float %130)
*float8B

	full_text


float %102
*float8B

	full_text


float %128
*float8B

	full_text


float %130
Sextractelement8B?
=
	full_text0
.
,%132 = extractelement <4 x float> %94, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %94
icall8B_
]
	full_textP
N
L%133 = tail call float @llvm.fmuladd.f32(float %108, float %132, float %131)
*float8B

	full_text


float %108
*float8B

	full_text


float %132
*float8B

	full_text


float %131
Textractelement8B@
>
	full_text1
/
-%134 = extractelement <4 x float> %101, i64 2
6<4 x float>8B#
!
	full_text

<4 x float> %101
icall8B_
]
	full_textP
N
L%135 = tail call float @llvm.fmuladd.f32(float %111, float %134, float %133)
*float8B

	full_text


float %111
*float8B

	full_text


float %134
*float8B

	full_text


float %133
Sextractelement8B?
=
	full_text0
.
,%136 = extractelement <4 x float> %53, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %53
9fadd8B/
-
	full_text 

%137 = fadd float %136, %135
*float8B

	full_text


float %136
*float8B

	full_text


float %135
^insertelement8BK
I
	full_text<
:
8%138 = insertelement <4 x float> %127, float %137, i64 2
6<4 x float>8B#
!
	full_text

<4 x float> %127
*float8B

	full_text


float %137
Sextractelement8B?
=
	full_text0
.
,%139 = extractelement <4 x float> %80, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %80
Sextractelement8B?
=
	full_text0
.
,%140 = extractelement <4 x float> %87, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %87
9fmul8B/
-
	full_text 

%141 = fmul float %104, %140
*float8B

	full_text


float %104
*float8B

	full_text


float %140
icall8B_
]
	full_textP
N
L%142 = tail call float @llvm.fmuladd.f32(float %102, float %139, float %141)
*float8B

	full_text


float %102
*float8B

	full_text


float %139
*float8B

	full_text


float %141
Sextractelement8B?
=
	full_text0
.
,%143 = extractelement <4 x float> %94, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %94
icall8B_
]
	full_textP
N
L%144 = tail call float @llvm.fmuladd.f32(float %108, float %143, float %142)
*float8B

	full_text


float %108
*float8B

	full_text


float %143
*float8B

	full_text


float %142
Textractelement8B@
>
	full_text1
/
-%145 = extractelement <4 x float> %101, i64 3
6<4 x float>8B#
!
	full_text

<4 x float> %101
icall8B_
]
	full_textP
N
L%146 = tail call float @llvm.fmuladd.f32(float %111, float %145, float %144)
*float8B

	full_text


float %111
*float8B

	full_text


float %145
*float8B

	full_text


float %144
Sextractelement8B?
=
	full_text0
.
,%147 = extractelement <4 x float> %53, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %53
9fadd8B/
-
	full_text 

%148 = fadd float %147, %146
*float8B

	full_text


float %147
*float8B

	full_text


float %146
^insertelement8BK
I
	full_text<
:
8%149 = insertelement <4 x float> %138, float %148, i64 3
6<4 x float>8B#
!
	full_text

<4 x float> %138
*float8B

	full_text


float %148
Sextractelement8B?
=
	full_text0
.
,%150 = extractelement <4 x float> %66, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %66
Sextractelement8B?
=
	full_text0
.
,%151 = extractelement <4 x float> %66, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %66
9fmul8B/
-
	full_text 

%152 = fmul float %151, %105
*float8B

	full_text


float %151
*float8B

	full_text


float %105
icall8B_
]
	full_textP
N
L%153 = tail call float @llvm.fmuladd.f32(float %150, float %103, float %152)
*float8B

	full_text


float %150
*float8B

	full_text


float %103
*float8B

	full_text


float %152
Sextractelement8B?
=
	full_text0
.
,%154 = extractelement <4 x float> %66, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %66
icall8B_
]
	full_textP
N
L%155 = tail call float @llvm.fmuladd.f32(float %154, float %109, float %153)
*float8B

	full_text


float %154
*float8B

	full_text


float %109
*float8B

	full_text


float %153
Sextractelement8B?
=
	full_text0
.
,%156 = extractelement <4 x float> %66, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %66
icall8B_
]
	full_textP
N
L%157 = tail call float @llvm.fmuladd.f32(float %156, float %112, float %155)
*float8B

	full_text


float %156
*float8B

	full_text


float %112
*float8B

	full_text


float %155
Sextractelement8B?
=
	full_text0
.
,%158 = extractelement <4 x float> %54, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %54
9fadd8B/
-
	full_text 

%159 = fadd float %158, %157
*float8B

	full_text


float %158
*float8B

	full_text


float %157
_insertelement8BL
J
	full_text=
;
9%160 = insertelement <4 x float> undef, float %159, i64 0
*float8B

	full_text


float %159
9fmul8B/
-
	full_text 

%161 = fmul float %151, %118
*float8B

	full_text


float %151
*float8B

	full_text


float %118
icall8B_
]
	full_textP
N
L%162 = tail call float @llvm.fmuladd.f32(float %150, float %117, float %161)
*float8B

	full_text


float %150
*float8B

	full_text


float %117
*float8B

	full_text


float %161
icall8B_
]
	full_textP
N
L%163 = tail call float @llvm.fmuladd.f32(float %154, float %121, float %162)
*float8B

	full_text


float %154
*float8B

	full_text


float %121
*float8B

	full_text


float %162
icall8B_
]
	full_textP
N
L%164 = tail call float @llvm.fmuladd.f32(float %156, float %123, float %163)
*float8B

	full_text


float %156
*float8B

	full_text


float %123
*float8B

	full_text


float %163
Sextractelement8B?
=
	full_text0
.
,%165 = extractelement <4 x float> %54, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %54
9fadd8B/
-
	full_text 

%166 = fadd float %165, %164
*float8B

	full_text


float %165
*float8B

	full_text


float %164
^insertelement8BK
I
	full_text<
:
8%167 = insertelement <4 x float> %160, float %166, i64 1
6<4 x float>8B#
!
	full_text

<4 x float> %160
*float8B

	full_text


float %166
9fmul8B/
-
	full_text 

%168 = fmul float %151, %129
*float8B

	full_text


float %151
*float8B

	full_text


float %129
icall8B_
]
	full_textP
N
L%169 = tail call float @llvm.fmuladd.f32(float %150, float %128, float %168)
*float8B

	full_text


float %150
*float8B

	full_text


float %128
*float8B

	full_text


float %168
icall8B_
]
	full_textP
N
L%170 = tail call float @llvm.fmuladd.f32(float %154, float %132, float %169)
*float8B

	full_text


float %154
*float8B

	full_text


float %132
*float8B

	full_text


float %169
icall8B_
]
	full_textP
N
L%171 = tail call float @llvm.fmuladd.f32(float %156, float %134, float %170)
*float8B

	full_text


float %156
*float8B

	full_text


float %134
*float8B

	full_text


float %170
Sextractelement8B?
=
	full_text0
.
,%172 = extractelement <4 x float> %54, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %54
9fadd8B/
-
	full_text 

%173 = fadd float %172, %171
*float8B

	full_text


float %172
*float8B

	full_text


float %171
^insertelement8BK
I
	full_text<
:
8%174 = insertelement <4 x float> %167, float %173, i64 2
6<4 x float>8B#
!
	full_text

<4 x float> %167
*float8B

	full_text


float %173
9fmul8B/
-
	full_text 

%175 = fmul float %151, %140
*float8B

	full_text


float %151
*float8B

	full_text


float %140
icall8B_
]
	full_textP
N
L%176 = tail call float @llvm.fmuladd.f32(float %150, float %139, float %175)
*float8B

	full_text


float %150
*float8B

	full_text


float %139
*float8B

	full_text


float %175
icall8B_
]
	full_textP
N
L%177 = tail call float @llvm.fmuladd.f32(float %154, float %143, float %176)
*float8B

	full_text


float %154
*float8B

	full_text


float %143
*float8B

	full_text


float %176
icall8B_
]
	full_textP
N
L%178 = tail call float @llvm.fmuladd.f32(float %156, float %145, float %177)
*float8B

	full_text


float %156
*float8B

	full_text


float %145
*float8B

	full_text


float %177
Sextractelement8B?
=
	full_text0
.
,%179 = extractelement <4 x float> %54, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %54
9fadd8B/
-
	full_text 

%180 = fadd float %179, %178
*float8B

	full_text


float %179
*float8B

	full_text


float %178
^insertelement8BK
I
	full_text<
:
8%181 = insertelement <4 x float> %174, float %180, i64 3
6<4 x float>8B#
!
	full_text

<4 x float> %174
*float8B

	full_text


float %180
Sextractelement8B?
=
	full_text0
.
,%182 = extractelement <4 x float> %70, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %70
Sextractelement8B?
=
	full_text0
.
,%183 = extractelement <4 x float> %70, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %70
9fmul8B/
-
	full_text 

%184 = fmul float %183, %105
*float8B

	full_text


float %183
*float8B

	full_text


float %105
icall8B_
]
	full_textP
N
L%185 = tail call float @llvm.fmuladd.f32(float %182, float %103, float %184)
*float8B

	full_text


float %182
*float8B

	full_text


float %103
*float8B

	full_text


float %184
Sextractelement8B?
=
	full_text0
.
,%186 = extractelement <4 x float> %70, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %70
icall8B_
]
	full_textP
N
L%187 = tail call float @llvm.fmuladd.f32(float %186, float %109, float %185)
*float8B

	full_text


float %186
*float8B

	full_text


float %109
*float8B

	full_text


float %185
Sextractelement8B?
=
	full_text0
.
,%188 = extractelement <4 x float> %70, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %70
icall8B_
]
	full_textP
N
L%189 = tail call float @llvm.fmuladd.f32(float %188, float %112, float %187)
*float8B

	full_text


float %188
*float8B

	full_text


float %112
*float8B

	full_text


float %187
Sextractelement8B?
=
	full_text0
.
,%190 = extractelement <4 x float> %55, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %55
9fadd8B/
-
	full_text 

%191 = fadd float %190, %189
*float8B

	full_text


float %190
*float8B

	full_text


float %189
_insertelement8BL
J
	full_text=
;
9%192 = insertelement <4 x float> undef, float %191, i64 0
*float8B

	full_text


float %191
9fmul8B/
-
	full_text 

%193 = fmul float %183, %118
*float8B

	full_text


float %183
*float8B

	full_text


float %118
icall8B_
]
	full_textP
N
L%194 = tail call float @llvm.fmuladd.f32(float %182, float %117, float %193)
*float8B

	full_text


float %182
*float8B

	full_text


float %117
*float8B

	full_text


float %193
icall8B_
]
	full_textP
N
L%195 = tail call float @llvm.fmuladd.f32(float %186, float %121, float %194)
*float8B

	full_text


float %186
*float8B

	full_text


float %121
*float8B

	full_text


float %194
icall8B_
]
	full_textP
N
L%196 = tail call float @llvm.fmuladd.f32(float %188, float %123, float %195)
*float8B

	full_text


float %188
*float8B

	full_text


float %123
*float8B

	full_text


float %195
Sextractelement8B?
=
	full_text0
.
,%197 = extractelement <4 x float> %55, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %55
9fadd8B/
-
	full_text 

%198 = fadd float %197, %196
*float8B

	full_text


float %197
*float8B

	full_text


float %196
^insertelement8BK
I
	full_text<
:
8%199 = insertelement <4 x float> %192, float %198, i64 1
6<4 x float>8B#
!
	full_text

<4 x float> %192
*float8B

	full_text


float %198
9fmul8B/
-
	full_text 

%200 = fmul float %183, %129
*float8B

	full_text


float %183
*float8B

	full_text


float %129
icall8B_
]
	full_textP
N
L%201 = tail call float @llvm.fmuladd.f32(float %182, float %128, float %200)
*float8B

	full_text


float %182
*float8B

	full_text


float %128
*float8B

	full_text


float %200
icall8B_
]
	full_textP
N
L%202 = tail call float @llvm.fmuladd.f32(float %186, float %132, float %201)
*float8B

	full_text


float %186
*float8B

	full_text


float %132
*float8B

	full_text


float %201
icall8B_
]
	full_textP
N
L%203 = tail call float @llvm.fmuladd.f32(float %188, float %134, float %202)
*float8B

	full_text


float %188
*float8B

	full_text


float %134
*float8B

	full_text


float %202
Sextractelement8B?
=
	full_text0
.
,%204 = extractelement <4 x float> %55, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %55
9fadd8B/
-
	full_text 

%205 = fadd float %204, %203
*float8B

	full_text


float %204
*float8B

	full_text


float %203
^insertelement8BK
I
	full_text<
:
8%206 = insertelement <4 x float> %199, float %205, i64 2
6<4 x float>8B#
!
	full_text

<4 x float> %199
*float8B

	full_text


float %205
9fmul8B/
-
	full_text 

%207 = fmul float %183, %140
*float8B

	full_text


float %183
*float8B

	full_text


float %140
icall8B_
]
	full_textP
N
L%208 = tail call float @llvm.fmuladd.f32(float %182, float %139, float %207)
*float8B

	full_text


float %182
*float8B

	full_text


float %139
*float8B

	full_text


float %207
icall8B_
]
	full_textP
N
L%209 = tail call float @llvm.fmuladd.f32(float %186, float %143, float %208)
*float8B

	full_text


float %186
*float8B

	full_text


float %143
*float8B

	full_text


float %208
icall8B_
]
	full_textP
N
L%210 = tail call float @llvm.fmuladd.f32(float %188, float %145, float %209)
*float8B

	full_text


float %188
*float8B

	full_text


float %145
*float8B

	full_text


float %209
Sextractelement8B?
=
	full_text0
.
,%211 = extractelement <4 x float> %55, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %55
9fadd8B/
-
	full_text 

%212 = fadd float %211, %210
*float8B

	full_text


float %211
*float8B

	full_text


float %210
^insertelement8BK
I
	full_text<
:
8%213 = insertelement <4 x float> %206, float %212, i64 3
6<4 x float>8B#
!
	full_text

<4 x float> %206
*float8B

	full_text


float %212
Sextractelement8B?
=
	full_text0
.
,%214 = extractelement <4 x float> %74, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %74
Sextractelement8B?
=
	full_text0
.
,%215 = extractelement <4 x float> %74, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %74
9fmul8B/
-
	full_text 

%216 = fmul float %215, %105
*float8B

	full_text


float %215
*float8B

	full_text


float %105
icall8B_
]
	full_textP
N
L%217 = tail call float @llvm.fmuladd.f32(float %214, float %103, float %216)
*float8B

	full_text


float %214
*float8B

	full_text


float %103
*float8B

	full_text


float %216
Sextractelement8B?
=
	full_text0
.
,%218 = extractelement <4 x float> %74, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %74
icall8B_
]
	full_textP
N
L%219 = tail call float @llvm.fmuladd.f32(float %218, float %109, float %217)
*float8B

	full_text


float %218
*float8B

	full_text


float %109
*float8B

	full_text


float %217
Sextractelement8B?
=
	full_text0
.
,%220 = extractelement <4 x float> %74, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %74
icall8B_
]
	full_textP
N
L%221 = tail call float @llvm.fmuladd.f32(float %220, float %112, float %219)
*float8B

	full_text


float %220
*float8B

	full_text


float %112
*float8B

	full_text


float %219
Sextractelement8B?
=
	full_text0
.
,%222 = extractelement <4 x float> %56, i64 0
5<4 x float>8B"
 
	full_text

<4 x float> %56
9fadd8B/
-
	full_text 

%223 = fadd float %222, %221
*float8B

	full_text


float %222
*float8B

	full_text


float %221
_insertelement8BL
J
	full_text=
;
9%224 = insertelement <4 x float> undef, float %223, i64 0
*float8B

	full_text


float %223
9fmul8B/
-
	full_text 

%225 = fmul float %215, %118
*float8B

	full_text


float %215
*float8B

	full_text


float %118
icall8B_
]
	full_textP
N
L%226 = tail call float @llvm.fmuladd.f32(float %214, float %117, float %225)
*float8B

	full_text


float %214
*float8B

	full_text


float %117
*float8B

	full_text


float %225
icall8B_
]
	full_textP
N
L%227 = tail call float @llvm.fmuladd.f32(float %218, float %121, float %226)
*float8B

	full_text


float %218
*float8B

	full_text


float %121
*float8B

	full_text


float %226
icall8B_
]
	full_textP
N
L%228 = tail call float @llvm.fmuladd.f32(float %220, float %123, float %227)
*float8B

	full_text


float %220
*float8B

	full_text


float %123
*float8B

	full_text


float %227
Sextractelement8B?
=
	full_text0
.
,%229 = extractelement <4 x float> %56, i64 1
5<4 x float>8B"
 
	full_text

<4 x float> %56
9fadd8B/
-
	full_text 

%230 = fadd float %229, %228
*float8B

	full_text


float %229
*float8B

	full_text


float %228
^insertelement8BK
I
	full_text<
:
8%231 = insertelement <4 x float> %224, float %230, i64 1
6<4 x float>8B#
!
	full_text

<4 x float> %224
*float8B

	full_text


float %230
9fmul8B/
-
	full_text 

%232 = fmul float %215, %129
*float8B

	full_text


float %215
*float8B

	full_text


float %129
icall8B_
]
	full_textP
N
L%233 = tail call float @llvm.fmuladd.f32(float %214, float %128, float %232)
*float8B

	full_text


float %214
*float8B

	full_text


float %128
*float8B

	full_text


float %232
icall8B_
]
	full_textP
N
L%234 = tail call float @llvm.fmuladd.f32(float %218, float %132, float %233)
*float8B

	full_text


float %218
*float8B

	full_text


float %132
*float8B

	full_text


float %233
icall8B_
]
	full_textP
N
L%235 = tail call float @llvm.fmuladd.f32(float %220, float %134, float %234)
*float8B

	full_text


float %220
*float8B

	full_text


float %134
*float8B

	full_text


float %234
Sextractelement8B?
=
	full_text0
.
,%236 = extractelement <4 x float> %56, i64 2
5<4 x float>8B"
 
	full_text

<4 x float> %56
9fadd8B/
-
	full_text 

%237 = fadd float %236, %235
*float8B

	full_text


float %236
*float8B

	full_text


float %235
^insertelement8BK
I
	full_text<
:
8%238 = insertelement <4 x float> %231, float %237, i64 2
6<4 x float>8B#
!
	full_text

<4 x float> %231
*float8B

	full_text


float %237
9fmul8B/
-
	full_text 

%239 = fmul float %215, %140
*float8B

	full_text


float %215
*float8B

	full_text


float %140
icall8B_
]
	full_textP
N
L%240 = tail call float @llvm.fmuladd.f32(float %214, float %139, float %239)
*float8B

	full_text


float %214
*float8B

	full_text


float %139
*float8B

	full_text


float %239
icall8B_
]
	full_textP
N
L%241 = tail call float @llvm.fmuladd.f32(float %218, float %143, float %240)
*float8B

	full_text


float %218
*float8B

	full_text


float %143
*float8B

	full_text


float %240
icall8B_
]
	full_textP
N
L%242 = tail call float @llvm.fmuladd.f32(float %220, float %145, float %241)
*float8B

	full_text


float %220
*float8B

	full_text


float %145
*float8B

	full_text


float %241
Sextractelement8B?
=
	full_text0
.
,%243 = extractelement <4 x float> %56, i64 3
5<4 x float>8B"
 
	full_text

<4 x float> %56
9fadd8B/
-
	full_text 

%244 = fadd float %243, %242
*float8B

	full_text


float %243
*float8B

	full_text


float %242
^insertelement8BK
I
	full_text<
:
8%245 = insertelement <4 x float> %238, float %244, i64 3
6<4 x float>8B#
!
	full_text

<4 x float> %238
*float8B

	full_text


float %244
9add8B0
.
	full_text!

%246 = add nuw nsw i64 %52, 4
%i648B

	full_text
	
i64 %52
:icmp8B0
.
	full_text!

%247 = icmp ult i64 %246, %26
&i648B

	full_text


i64 %246
%i648B

	full_text
	
i64 %26
;br8B3
1
	full_text$
"
 br i1 %247, label %51, label %27
$i18B

	full_text
	
i1 %247
6<4 x float>*8B"
 
	full_text

<4 x float>* %2
$i328B

	full_text


i32 %4
6<4 x float>*8B"
 
	full_text

<4 x float>* %1
6<4 x float>*8B"
 
	full_text

<4 x float>* %0
$i328B
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
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
A<4 x float>8B.
,
	full_text

<4 x float> zeroinitializer
#i328B

	full_text	

i32 0
7<4 x float>8B$
"
	full_text

<4 x float> undef        	
 		                     !  "# "" $% $& $$ '' (* )+ )) ,- ,. ,, /0 /1 // 23 22 45 44 67 66 89 88 :; :< :: => =? == @A @@ BC BB DE DF DD GH GI GG JK JL JJ MN MM OP OO QR QS QQ TU TV TT WX WY WW Z[ ZZ \] \\ ^_ ^` ^^ ab ac aa de df dd gh gg ij ii kl km kk np oo qr qq st ss uv uu wx ww yz yy {| {{ }~ } }} ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ã
å ãã çé çç èê è
ë èè íì íí î
ï îî ñó ññ òô ò
ö òò õú õõ ù
û ùù ü† üü °¢ °° £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ ππ ª
º ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… À
Ã ÀÀ ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €
‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·· „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ ÓÓ Ò  ÚÛ Ú
Ù Ú
ı ÚÚ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ É
Ñ ÉÉ ÖÜ ÖÖ áà áá âä â
ã ââ åç å
é å
è åå êë êê íì í
î í
ï íí ñó ññ òô ò
ö ò
õ òò úù úú ûü û
† ûû °¢ °
£ °° §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑
π ∑
∫ ∑∑ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À  
Ã  
Õ    Œœ ŒŒ –— –
“ –
” –– ‘’ ‘‘ ÷◊ ÷
ÿ ÷
Ÿ ÷÷ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï ÈÈ ÌÓ ÌÌ Ô Ô
Ò Ô
Ú ÔÔ ÛÙ ÛÛ ıˆ ı
˜ ı
¯ ıı ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛
ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü ÉÉ áà á
â á
ä áá ãå ã
ç ã
é ãã èê èè ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú ö
ù öö ûü û
† û
° ûû ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µ
∑ µ
∏ µµ π∫ π
ª π
º ππ Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «« …  …
À …… ÃÕ Ã
Œ Ã
œ ÃÃ –— –– “” “
‘ “
’ ““ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿ
€ ÿÿ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·
‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ Ó
 Ó
Ò ÓÓ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝
Ä ˝˝ ÅÇ Å
É Å
Ñ ÅÅ ÖÜ Ö
á Ö
à ÖÖ âä ââ ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë îï î
ñ î
ó îî òô ò
ö ò
õ òò úù ú
û ú
ü úú †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥≥ µ∂ µ
∑ µ
∏ µµ π∫ ππ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡
√ ¡¡ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …
Ã …… ÕŒ Õ
œ Õ
– ÕÕ —“ —
” —
‘ —— ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰
Ê ‰
Á ‰‰ ËÈ Ë
Í Ë
Î ËË ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜
˙ ˜˜ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé ç
è çç êë êí Bí Oí \í iì î ´î ªî Àî €ï Çï ãï îï ùñ ñ ñ '   
 	 	 	 	  	   	    !	 #" % & *" + - . 0 1à 3• 5¬ 7ﬂ 9	 ; <: > ?= A@ C8 EB F/ H IG K LJ NM P6 RO S, U VT X YW [Z ]4 _\ `) b ca e fd hg j2 li mã pﬂ r¬ t• và xo zy | ~{ } ÅÄ ÉÇ Ö á{ àÜ äâ åã é ê{ ëè ìí ïî ó$ ô{ öò úõ ûù †o ¢ §° •£ ß ®¶ ™© ¨´ Æo ∞Ø ≤± ¥ µ≥ ∑ ∏∂ ∫π ºª æo ¿ø ¬¡ ƒ ≈√ « »∆  … ÃÀ Œo –œ “— ‘ ’” ◊ ÿ÷ ⁄Ÿ ‹€ ﬁÑ ‡≠ ‚Ñ ‰Ω Ê„ ËÂ Èﬂ Î· ÏÁ ÌÑ ÔÕ ÒÓ Û ÙÍ ıÑ ˜› ˘ˆ ˚¯ ¸Ú ˝q ˇ˛ Å˙ ÇÄ Ñ≠ ÜΩ à„ äá ãﬂ çÖ éâ èÕ ëÓ ìê îå ï› óˆ ôñ öí õq ùú üò †É ¢û £≠ •Ω ß„ ©¶ ™ﬂ ¨§ ≠® ÆÕ ∞Ó ≤Ø ≥´ ¥› ∂ˆ ∏µ π± ∫q ºª æ∑ ø° ¡Ω ¬≠ ƒΩ ∆„ »≈ …ﬂ À√ Ã« ÕÕ œÓ —Œ “  ”› ’ˆ ◊‘ ÿ– Ÿq €⁄ ›÷ ﬁ¿ ‡‹ ·ç „ç Â‰ ÁÂ Ë‚ Í· ÎÊ Ïç ÓÌ  ÒÈ Úç ÙÛ ˆ¯ ˜Ô ¯s ˙˘ ¸ı ˝˚ ˇ‰ Åá Ç‚ ÑÖ ÖÄ ÜÌ àê âÉ äÛ åñ çá és êè íã ì˛ ïë ñ‰ ò¶ ô‚ õ§ úó ùÌ üØ †ö °Û £µ §û •s ß¶ ©¢ ™î ¨® ≠‰ Ø≈ ∞‚ ≤√ ≥Æ ¥Ì ∂Œ ∑± ∏Û ∫‘ ªµ ºs æΩ ¿π ¡´ √ø ƒñ ∆ñ »«  Â À≈ Õ· Œ… œñ —– ” ‘Ã ’ñ ◊÷ Ÿ¯ ⁄“ €u ›‹ ﬂÿ ‡ﬁ ‚« ‰á Â≈ ÁÖ Ë„ È– Îê ÏÊ Ì÷ Ôñ Í Òu ÛÚ ıÓ ˆ· ¯Ù ˘« ˚¶ ¸≈ ˛§ ˇ˙ Ä– ÇØ É˝ Ñ÷ Üµ áÅ àu äâ åÖ ç˜ èã ê« í≈ ì≈ ï√ ñë ó– ôŒ öî õ÷ ù‘ ûò üu °† £ú §é ¶¢ ßü ©ü ´™ ≠Â Æ® ∞· ±¨ ≤ü ¥≥ ∂ ∑Ø ∏ü ∫π º¯ Ωµ æw ¿ø ¬ª √¡ ≈™ «á »®  Ö À∆ Ã≥ Œê œ… –π “ñ ”Õ ‘w ÷’ ÿ— Ÿƒ €◊ ‹™ ﬁ¶ ﬂ® ·§ ‚› „≥ ÂØ Ê‡ Áπ Èµ Í‰ Îw ÌÏ ÔË ⁄ ÚÓ Û™ ı≈ ˆ® ¯√ ˘Ù ˙≥ ¸Œ ˝˜ ˛π Ä‘ Å˚ Çw ÑÉ Üˇ áÒ âÖ äo åã é' èç ë   )( oê oê ) n òò óóû òò û˙ òò ˙È òò Èá òò áã òò ã“ òò “ óó ú òò úØ òò Øπ òò πÚ òò ÚÔ òò Ôı òò ı óó ∑ òò ∑Ã òò Ãµ òò µ– òò –Ê òò Ê˝ òò ˝Ó òò Óö òò öµ òò µÕ òò Õ± òò ±í òò í  òò  ± òò ±Å òò Åª òò ª… òò …— òò —‡ òò ‡ò òò ò˜ òò ˜ˇ òò ˇå òò å¢ òò ¢Í òò Í´ òò ´˚ òò ˚÷ òò ÷Ë òò ËÖ òò ÖÉ òò É‰ òò ‰Í òò Íî òò îÿ òò ÿò òò ò	ô y
ô Ó
ô §
ô ¶
ô Ø
ô µ
ô ª
ô ¿
ô Ì
ô ¶
ô ´
ô –
ô â
ô é
ô ≥
ô Ï
ô Ò	ö 	ö 		ö 	ö 	ö 
ö ¡õ 	õ 	õ 
õ ±
ú ˆ
ú √
ú ≈
ú Œ
ú ‘
ú ⁄
ú ﬂ
ú Û
ú Ω
ú ¬
ú ÷
ú †
ú •
ú π
ú É
ú à
ù ã	û 	û "
û —
ü „
ü Ö
ü á
ü ê
ü ñ
ü ú
ü °
ü ‰
ü è
ü î
ü «
ü Ú
ü ˜
ü ™
ü ’
ü ⁄† o
† ﬂ
† ·
† Â
† 
† ¯
† ˛
† É
† ‚
† ˘
† ˛
† ≈
† ‹
† ·
† ®
† ø
† ƒ° 2° 4° 6° 8° q° s° u° w¢ 	¢ £ É£ ˛£ ·£ ƒ"
	mmmKernel"
_Z13get_global_idj"
llvm.fmuladd.f32*ò
MatrixMultiplication_Kernels.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

devmap_label
 

wgsize
@

wgsize_log1p
’◊,A
 
transfer_bytes_log1p
’◊,A

transfer_bytes
ÄÄ