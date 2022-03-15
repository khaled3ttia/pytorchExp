

[external]
JcallBB
@
	full_text3
1
/%6 = tail call i64 @_Z12get_local_idj(i32 0) #4
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
JcallBB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_group_idj(i32 0) #4
5andB.
,
	full_text

%9 = and i64 %8, 4294967295
"i64B

	full_text


i64 %8
egetelementptrBT
R
	full_textE
C
A%10 = getelementptr inbounds <4 x float>, <4 x float>* %1, i64 %9
"i64B

	full_text


i64 %9
WloadBO
M
	full_text@
>
<%11 = load <4 x float>, <4 x float>* %10, align 16, !tbaa !9
5<4 x float>*B#
!
	full_text

<4 x float>* %10
áfsubB
}
	full_textp
n
l%12 = fsub <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %11
3<4 x float>B"
 
	full_text

<4 x float> %11
áfmulB
}
	full_textp
n
l%13 = fmul <4 x float> %11, <float 3.000000e+01, float 3.000000e+01, float 3.000000e+01, float 3.000000e+01>
3<4 x float>B"
 
	full_text

<4 x float> %11
œcallB∆
√
	full_textµ
≤
Ø%14 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %12, <4 x float> <float 5.000000e+00, float 5.000000e+00, float 5.000000e+00, float 5.000000e+00>, <4 x float> %13)
3<4 x float>B"
 
	full_text

<4 x float> %12
3<4 x float>B"
 
	full_text

<4 x float> %13
áfmulB
}
	full_textp
n
l%15 = fmul <4 x float> %11, <float 1.000000e+02, float 1.000000e+02, float 1.000000e+02, float 1.000000e+02>
3<4 x float>B"
 
	full_text

<4 x float> %11
:faddB2
0
	full_text#
!
%16 = fadd <4 x float> %12, %15
3<4 x float>B"
 
	full_text

<4 x float> %12
3<4 x float>B"
 
	full_text

<4 x float> %15
áfmulB
}
	full_textp
n
l%17 = fmul <4 x float> %11, <float 1.000000e+01, float 1.000000e+01, float 1.000000e+01, float 1.000000e+01>
3<4 x float>B"
 
	full_text

<4 x float> %11
œcallB∆
√
	full_textµ
≤
Ø%18 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %12, <4 x float> <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>, <4 x float> %17)
3<4 x float>B"
 
	full_text

<4 x float> %12
3<4 x float>B"
 
	full_text

<4 x float> %17
9sitofpB/
-
	full_text 

%19 = sitofp i32 %0 to float
JfdivBB
@
	full_text3
1
/%20 = fdiv float 1.000000e+00, %19, !fpmath !12
'floatB

	full_text

	float %19
[insertelementBJ
H
	full_text;
9
7%21 = insertelement <4 x float> undef, float %20, i32 0
'floatB

	full_text

	float %20
ushufflevectorBd
b
	full_textU
S
Q%22 = shufflevector <4 x float> %21, <4 x float> undef, <4 x i32> zeroinitializer
3<4 x float>B"
 
	full_text

<4 x float> %21
:fmulB2
0
	full_text#
!
%23 = fmul <4 x float> %22, %18
3<4 x float>B"
 
	full_text

<4 x float> %22
3<4 x float>B"
 
	full_text

<4 x float> %18
XcallBP
N
	full_textA
?
=%24 = tail call <4 x float> @_Z4sqrtDv4_f(<4 x float> %23) #4
3<4 x float>B"
 
	full_text

<4 x float> %23
§fmulBõ
ò
	full_textä
á
Ñ%25 = fmul <4 x float> %24, <float 0x3FD3333340000000, float 0x3FD3333340000000, float 0x3FD3333340000000, float 0x3FD3333340000000>
3<4 x float>B"
 
	full_text

<4 x float> %24
§fmulBõ
ò
	full_textä
á
Ñ%26 = fmul <4 x float> %23, <float 0x3F947AE140000000, float 0x3F947AE140000000, float 0x3F947AE140000000, float 0x3F947AE140000000>
3<4 x float>B"
 
	full_text

<4 x float> %23
WcallBO
M
	full_text@
>
<%27 = tail call <4 x float> @_Z3expDv4_f(<4 x float> %26) #4
3<4 x float>B"
 
	full_text

<4 x float> %26
ñfdivBç
ä
	full_text}
{
y%28 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %27, !fpmath !12
3<4 x float>B"
 
	full_text

<4 x float> %27
WcallBO
M
	full_text@
>
<%29 = tail call <4 x float> @_Z3expDv4_f(<4 x float> %25) #4
3<4 x float>B"
 
	full_text

<4 x float> %25
ñfdivBç
ä
	full_text}
{
y%30 = fdiv <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %29, !fpmath !12
3<4 x float>B"
 
	full_text

<4 x float> %29
:fsubB2
0
	full_text#
!
%31 = fsub <4 x float> %27, %30
3<4 x float>B"
 
	full_text

<4 x float> %27
3<4 x float>B"
 
	full_text

<4 x float> %30
:fsubB2
0
	full_text#
!
%32 = fsub <4 x float> %29, %30
3<4 x float>B"
 
	full_text

<4 x float> %29
3<4 x float>B"
 
	full_text

<4 x float> %30
GfdivB?
=
	full_text0
.
,%33 = fdiv <4 x float> %31, %32, !fpmath !12
3<4 x float>B"
 
	full_text

<4 x float> %31
3<4 x float>B"
 
	full_text

<4 x float> %32
áfsubB
}
	full_textp
n
l%34 = fsub <4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>, %33
3<4 x float>B"
 
	full_text

<4 x float> %33
:fmulB2
0
	full_text#
!
%35 = fmul <4 x float> %28, %33
3<4 x float>B"
 
	full_text

<4 x float> %28
3<4 x float>B"
 
	full_text

<4 x float> %33
:fmulB2
0
	full_text#
!
%36 = fmul <4 x float> %28, %34
3<4 x float>B"
 
	full_text

<4 x float> %28
3<4 x float>B"
 
	full_text

<4 x float> %34
9uitofpB/
-
	full_text 

%37 = uitofp i32 %7 to float
"i32B

	full_text


i32 %7
>fsubB6
4
	full_text'
%
#%38 = fsub float -0.000000e+00, %19
'floatB

	full_text

	float %19
lcallBd
b
	full_textU
S
Q%39 = tail call float @llvm.fmuladd.f32(float %37, float 2.000000e+00, float %38)
'floatB

	full_text

	float %37
'floatB

	full_text

	float %38
[insertelementBJ
H
	full_text;
9
7%40 = insertelement <4 x float> undef, float %39, i32 0
'floatB

	full_text

	float %39
ushufflevectorBd
b
	full_textU
S
Q%41 = shufflevector <4 x float> %40, <4 x float> undef, <4 x i32> zeroinitializer
3<4 x float>B"
 
	full_text

<4 x float> %40
:fmulB2
0
	full_text#
!
%42 = fmul <4 x float> %25, %41
3<4 x float>B"
 
	full_text

<4 x float> %25
3<4 x float>B"
 
	full_text

<4 x float> %41
WcallBO
M
	full_text@
>
<%43 = tail call <4 x float> @_Z3expDv4_f(<4 x float> %42) #4
3<4 x float>B"
 
	full_text

<4 x float> %42
çfsubBÑ
Å
	full_textt
r
p%44 = fsub <4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>, %16
3<4 x float>B"
 
	full_text

<4 x float> %16
}callBu
s
	full_textf
d
b%45 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %14, <4 x float> %43, <4 x float> %44)
3<4 x float>B"
 
	full_text

<4 x float> %14
3<4 x float>B"
 
	full_text

<4 x float> %43
3<4 x float>B"
 
	full_text

<4 x float> %44
PextractelementB>
<
	full_text/
-
+%46 = extractelement <4 x float> %45, i64 0
3<4 x float>B"
 
	full_text

<4 x float> %45
AfcmpB9
7
	full_text*
(
&%47 = fcmp ogt float %46, 0.000000e+00
'floatB

	full_text

	float %46
OselectBE
C
	full_text6
4
2%48 = select i1 %47, float %46, float 0.000000e+00
!i1B

	full_text


i1 %47
'floatB

	full_text

	float %46
6andB/
-
	full_text 

%49 = and i64 %6, 4294967295
"i64B

	full_text


i64 %6
fgetelementptrBU
S
	full_textF
D
B%50 = getelementptr inbounds <4 x float>, <4 x float>* %3, i64 %49
#i64B

	full_text
	
i64 %49
[insertelementBJ
H
	full_text;
9
7%51 = insertelement <4 x float> undef, float %48, i64 0
'floatB

	full_text

	float %48
PextractelementB>
<
	full_text/
-
+%52 = extractelement <4 x float> %45, i64 1
3<4 x float>B"
 
	full_text

<4 x float> %45
AfcmpB9
7
	full_text*
(
&%53 = fcmp ogt float %52, 0.000000e+00
'floatB

	full_text

	float %52
OselectBE
C
	full_text6
4
2%54 = select i1 %53, float %52, float 0.000000e+00
!i1B

	full_text


i1 %53
'floatB

	full_text

	float %52
YinsertelementBH
F
	full_text9
7
5%55 = insertelement <4 x float> %51, float %54, i64 1
3<4 x float>B"
 
	full_text

<4 x float> %51
'floatB

	full_text

	float %54
PextractelementB>
<
	full_text/
-
+%56 = extractelement <4 x float> %45, i64 2
3<4 x float>B"
 
	full_text

<4 x float> %45
AfcmpB9
7
	full_text*
(
&%57 = fcmp ogt float %56, 0.000000e+00
'floatB

	full_text

	float %56
OselectBE
C
	full_text6
4
2%58 = select i1 %57, float %56, float 0.000000e+00
!i1B

	full_text


i1 %57
'floatB

	full_text

	float %56
YinsertelementBH
F
	full_text9
7
5%59 = insertelement <4 x float> %55, float %58, i64 2
3<4 x float>B"
 
	full_text

<4 x float> %55
'floatB

	full_text

	float %58
PextractelementB>
<
	full_text/
-
+%60 = extractelement <4 x float> %45, i64 3
3<4 x float>B"
 
	full_text

<4 x float> %45
AfcmpB9
7
	full_text*
(
&%61 = fcmp ogt float %60, 0.000000e+00
'floatB

	full_text

	float %60
OselectBE
C
	full_text6
4
2%62 = select i1 %61, float %60, float 0.000000e+00
!i1B

	full_text


i1 %61
'floatB

	full_text

	float %60
YinsertelementBH
F
	full_text9
7
5%63 = insertelement <4 x float> %59, float %62, i64 3
3<4 x float>B"
 
	full_text

<4 x float> %59
'floatB

	full_text

	float %62
MstoreBD
B
	full_text5
3
1store <4 x float> %63, <4 x float>* %50, align 16
3<4 x float>B"
 
	full_text

<4 x float> %63
5<4 x float>*B#
!
	full_text

<4 x float>* %50
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
3icmpB+
)
	full_text

%64 = icmp sgt i32 %0, 0
8brB2
0
	full_text#
!
br i1 %64, label %65, label %71
!i1B

	full_text


i1 %64
/add8B&
$
	full_text

%66 = add i64 %6, 1
$i648B

	full_text


i64 %6
9and8B0
.
	full_text!

%67 = and i64 %66, 4294967295
%i648B

	full_text
	
i64 %66
hgetelementptr8BU
S
	full_textF
D
B%68 = getelementptr inbounds <4 x float>, <4 x float>* %3, i64 %67
%i648B

	full_text
	
i64 %67
hgetelementptr8BU
S
	full_textF
D
B%69 = getelementptr inbounds <4 x float>, <4 x float>* %4, i64 %49
%i648B

	full_text
	
i64 %49
hgetelementptr8BU
S
	full_textF
D
B%70 = getelementptr inbounds <4 x float>, <4 x float>* %4, i64 %67
%i648B

	full_text
	
i64 %67
'br8B

	full_text

br label %73
4icmp8B*
(
	full_text

%72 = icmp eq i32 %7, 0
$i328B

	full_text


i32 %7
:br8B2
0
	full_text#
!
br i1 %72, label %92, label %95
#i18B

	full_text


i1 %72
Cphi8B:
8
	full_text+
)
'%74 = phi i32 [ %0, %65 ], [ %90, %89 ]
%i328B

	full_text
	
i32 %90
7icmp8B-
+
	full_text

%75 = icmp ugt i32 %74, %7
%i328B

	full_text
	
i32 %74
$i328B

	full_text


i32 %7
:br8B2
0
	full_text#
!
br i1 %75, label %76, label %81
#i18B

	full_text


i1 %75
Yload8BO
M
	full_text@
>
<%77 = load <4 x float>, <4 x float>* %50, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %50
Yload8BO
M
	full_text@
>
<%78 = load <4 x float>, <4 x float>* %68, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %68
<fmul8B2
0
	full_text#
!
%79 = fmul <4 x float> %36, %78
5<4 x float>8B"
 
	full_text

<4 x float> %36
5<4 x float>8B"
 
	full_text

<4 x float> %78
call8Bu
s
	full_textf
d
b%80 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %35, <4 x float> %77, <4 x float> %79)
5<4 x float>8B"
 
	full_text

<4 x float> %35
5<4 x float>8B"
 
	full_text

<4 x float> %77
5<4 x float>8B"
 
	full_text

<4 x float> %79
Ystore8BN
L
	full_text?
=
;store <4 x float> %80, <4 x float>* %69, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %80
7<4 x float>*8B#
!
	full_text

<4 x float>* %69
'br8B

	full_text

br label %81
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5add8B,
*
	full_text

%82 = add nsw i32 %74, -1
%i328B

	full_text
	
i32 %74
7icmp8B-
+
	full_text

%83 = icmp ugt i32 %82, %7
%i328B

	full_text
	
i32 %82
$i328B

	full_text


i32 %7
:br8B2
0
	full_text#
!
br i1 %83, label %84, label %89
#i18B

	full_text


i1 %83
Yload8BO
M
	full_text@
>
<%85 = load <4 x float>, <4 x float>* %69, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %69
Yload8BO
M
	full_text@
>
<%86 = load <4 x float>, <4 x float>* %70, align 16, !tbaa !9
7<4 x float>*8B#
!
	full_text

<4 x float>* %70
<fmul8B2
0
	full_text#
!
%87 = fmul <4 x float> %36, %86
5<4 x float>8B"
 
	full_text

<4 x float> %36
5<4 x float>8B"
 
	full_text

<4 x float> %86
call8Bu
s
	full_textf
d
b%88 = tail call <4 x float> @llvm.fmuladd.v4f32(<4 x float> %35, <4 x float> %85, <4 x float> %87)
5<4 x float>8B"
 
	full_text

<4 x float> %35
5<4 x float>8B"
 
	full_text

<4 x float> %85
5<4 x float>8B"
 
	full_text

<4 x float> %87
Ystore8BN
L
	full_text?
=
;store <4 x float> %88, <4 x float>* %50, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %88
7<4 x float>*8B#
!
	full_text

<4 x float>* %50
'br8B

	full_text

br label %89
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
5add8B,
*
	full_text

%90 = add nsw i32 %74, -2
%i328B

	full_text
	
i32 %74
6icmp8B,
*
	full_text

%91 = icmp sgt i32 %74, 2
%i328B

	full_text
	
i32 %74
:br8B2
0
	full_text#
!
br i1 %91, label %73, label %71
#i18B

	full_text


i1 %91
Xload8BN
L
	full_text?
=
;%93 = load <4 x float>, <4 x float>* %3, align 16, !tbaa !9
ggetelementptr8BT
R
	full_textE
C
A%94 = getelementptr inbounds <4 x float>, <4 x float>* %2, i64 %9
$i648B

	full_text


i64 %9
Ystore8BN
L
	full_text?
=
;store <4 x float> %93, <4 x float>* %94, align 16, !tbaa !9
5<4 x float>8B"
 
	full_text

<4 x float> %93
7<4 x float>*8B#
!
	full_text

<4 x float>* %94
'br8B

	full_text

br label %95
$ret8	B

	full_text


ret void
6<4 x float>*8
B"
 
	full_text

<4 x float>* %4
6<4 x float>*8
B"
 
	full_text

<4 x float>* %3
$i328
B

	full_text


i32 %0
6<4 x float>*8
B"
 
	full_text

<4 x float>* %1
6<4 x float>*8
B"
 
	full_text

<4 x float>* %2
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
,i648
B!

	full_text

i64 4294967295
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
i64 0
3float8
B&
$
	full_text

float -0.000000e+00
Ü<4 x float>8
Bs
q
	full_textd
b
`<4 x float> <float -0.000000e+00, float -0.000000e+00, float -0.000000e+00, float -0.000000e+00>
$i328
B

	full_text


i32 -1
ú<4 x float>8
Bà
Ö
	full_textx
v
t<4 x float> <float 0x3F947AE140000000, float 0x3F947AE140000000, float 0x3F947AE140000000, float 0x3F947AE140000000>
#i648
B

	full_text	

i64 3
#i328
B

	full_text	

i32 2
7<4 x float>8
B$
"
	full_text

<4 x float> undef
Ç<4 x float>8
Bo
m
	full_text`
^
\<4 x float> <float 5.000000e+00, float 5.000000e+00, float 5.000000e+00, float 5.000000e+00>
Ç<4 x float>8
Bo
m
	full_text`
^
\<4 x float> <float 3.000000e+01, float 3.000000e+01, float 3.000000e+01, float 3.000000e+01>
Ç<4 x float>8
Bo
m
	full_text`
^
\<4 x float> <float 2.500000e-01, float 2.500000e-01, float 2.500000e-01, float 2.500000e-01>
$i328
B

	full_text


i32 -2
ú<4 x float>8
Bà
Ö
	full_textx
v
t<4 x float> <float 0x3FD3333340000000, float 0x3FD3333340000000, float 0x3FD3333340000000, float 0x3FD3333340000000>
#i328
B

	full_text	

i32 1
2float8
B%
#
	full_text

float 2.000000e+00
Ç<4 x float>8
Bo
m
	full_text`
^
\<4 x float> <float 1.000000e+01, float 1.000000e+01, float 1.000000e+01, float 1.000000e+01>
=	<4 x i32>8
B,
*
	full_text

<4 x i32> zeroinitializer
2float8
B%
#
	full_text

float 1.000000e+00
Ç<4 x float>8
Bo
m
	full_text`
^
\<4 x float> <float 1.000000e+02, float 1.000000e+02, float 1.000000e+02, float 1.000000e+02>
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
i64 1
Ç<4 x float>8
Bo
m
	full_text`
^
\<4 x float> <float 1.000000e+00, float 1.000000e+00, float 1.000000e+00, float 1.000000e+00>
#i648
B

	full_text	

i64 2        	
 		                        !" !! #$ #% ## &' && () (( *+ ** ,- ,, ./ .. 01 00 23 22 45 46 44 78 79 77 :; :< :: => == ?@ ?A ?? BC BD BB EF EE GH GG IJ IK II LM LL NO NN PQ PR PP ST SS UV UU WX WY WZ WW [\ [[ ]^ ]] _` _a __ bc bb de dd fg ff hi hh jk jj lm ln ll op oq oo rs rr tu tt vw vx vv yz y{ yy |} || ~ ~~ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ ââ ää ãå ãé çç èê èè ë
í ëë ì
î ìì ï
ñ ïï óô òò öõ ö
ù úú ûü û
† ûû °¢ °§ ££ •¶ •• ß® ß
© ßß ™´ ™
¨ ™
≠ ™™ ÆØ Æ
∞ ÆÆ ±≤ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈
« ≈≈ »…  À    ÃÕ ÃÃ Œœ Œ– —
“ —— ”‘ ”
’ ”” ÷ÿ ìÿ ïŸ dŸ ëŸ –⁄ ⁄ ä⁄ ú€ ‹ —    
	 	   	   	       "! $ %# '& )# +* -, /( 10 3, 52 60 82 94 ;7 <: >. @: A. C= D F HE JG KI ML O( QN RP T V XS YU ZW \[ ^] `[ a cb e_ gW ih kj mh nf pl qW sr ut wr xo zv {W }| ~ Å| Çy ÑÄ ÖÉ ád àä å éç êè íb îè ñ ôò õ  ùú ü †û ¢d §ë ¶B ®• ©? ´£ ¨ß ≠™ Øì ∞ú ¥≥ ∂ ∑µ πì ªï ΩB øº ¿? ¬∫ √æ ƒ¡ ∆d «ú Àú ÕÃ œ “– ‘— ’ã çã òó úö –ö ◊° £° ≤÷ ◊± ≤∏ ∫∏ …» …Œ úŒ ò ﬂﬂ ◊ ›› ·· ﬁﬁ ‡‡ „„ ‚‚ ››  ﬂﬂ â „„ âI ‚‚ I≤ „„ ≤W ﬂﬂ WS ·· S, ·· ,™ ﬂﬂ ™ ﬁﬁ  ﬂﬂ 0 ·· 0¡ ﬂﬂ ¡… „„ …& ‡‡ &	‰ 	‰ b
‰ è	Â ]	Â _	Â j	Â l	Â t	Â v	Â ~
Â Ä	Ê [	Ê fÁ GË U
È ≥	Í *	Î |
Î É
Ï ÃÌ 	Ì !Ì L	Ì NÌ f	Ó 	Ô 	 
Ò  	Ú (Û âÛ ≤Û …	Ù I	ı 	ˆ !	ˆ N˜ 	¯ ˘ ˘ 	˘ 	˘ L
˘ ä
˘ ò	˙ h	˙ o
˙ ç˚ ˚ .˚ 2˚ =	¸ r	¸ y"
binomial_options"
_Z12get_local_idj"
_Z12get_group_idj"
llvm.fmuladd.v4f32"
_Z4sqrtDv4_f"
_Z3expDv4_f"
llvm.fmuladd.f32"
_Z7barrierj*í
BinomialOption_Kernels.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282
 
transfer_bytes_log1p
· Ù@

devmap_label


wgsize_log1p
· Ù@

wgsize
ˇ

transfer_bytes
Ä