

[external]
EallocaB;
9
	full_text,
*
(%8 = alloca [3 x [5 x double]], align 16
HbitcastB=
;
	full_text.
,
*%9 = bitcast [3 x [5 x double]]* %8 to i8*
B[3 x [5 x double]]*B)
'
	full_text

[3 x [5 x double]]* %8
FallocaB<
:
	full_text-
+
)%10 = alloca [2 x [5 x double]], align 16
JbitcastB?
=
	full_text0
.
,%11 = bitcast [2 x [5 x double]]* %10 to i8*
C[2 x [5 x double]]*B*
(
	full_text

[2 x [5 x double]]* %10
FallocaB<
:
	full_text-
+
)%12 = alloca [5 x [5 x double]], align 16
JbitcastB?
=
	full_text0
.
,%13 = bitcast [5 x [5 x double]]* %12 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %12
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 120, i8* nonnull %9) #4
"i8*B

	full_text


i8* %9
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 80, i8* nonnull %11) #4
#i8*B

	full_text
	
i8* %11
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %13) #4
#i8*B

	full_text
	
i8* %13
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 1) #5
.addB'
%
	full_text

%15 = add i64 %14, 1
#i64B

	full_text
	
i64 %14
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
LcallBD
B
	full_text5
3
1%17 = tail call i64 @_Z13get_global_idj(i32 0) #5
.addB'
%
	full_text

%18 = add i64 %17, 1
#i64B

	full_text
	
i64 %17
2addB+
)
	full_text

%19 = add nsw i32 %5, -1
6icmpB.
,
	full_text

%20 = icmp sgt i32 %19, %16
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %16
:brB4
2
	full_text%
#
!br i1 %20, label %21, label %1210
!i1B

	full_text


i1 %20
8trunc8B-
+
	full_text

%22 = trunc i64 %18 to i32
%i648B

	full_text
	
i64 %18
4add8B+
)
	full_text

%23 = add nsw i32 %4, -1
8icmp8B.
,
	full_text

%24 = icmp sgt i32 %23, %22
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %22
<br8B4
2
	full_text%
#
!br i1 %24, label %25, label %1210
#i18B

	full_text


i1 %24
Wbitcast8BJ
H
	full_text;
9
7%26 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %2 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %3 to [65 x [65 x double]]*
1shl8B(
&
	full_text

%29 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%30 = ashr exact i64 %29, 32
%i648B

	full_text
	
i64 %29
1shl8B(
&
	full_text

%31 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
9ashr8B/
-
	full_text 

%32 = ashr exact i64 %31, 32
%i648B

	full_text
	
i64 %31
ôgetelementptr8BÖ
Ç
	full_textu
s
q%33 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 0, i64 %30, i64 %32
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Gbitcast8B:
8
	full_text+
)
'%34 = bitcast [5 x double]* %33 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
Hload8B>
<
	full_text/
-
+%35 = load i64, i64* %34, align 8, !tbaa !8
'i64*8B

	full_text


i64* %34
|getelementptr8Bi
g
	full_textZ
X
V%36 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Gbitcast8B:
8
	full_text+
)
'%37 = bitcast [5 x double]* %36 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
†getelementptr8Bå
â
	full_text|
z
x%38 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 0, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Abitcast8B4
2
	full_text%
#
!%39 = bitcast double* %38 to i64*
-double*8B

	full_text

double* %38
Hload8B>
<
	full_text/
-
+%40 = load i64, i64* %39, align 8, !tbaa !8
'i64*8B

	full_text


i64* %39
Égetelementptr8Bp
n
	full_texta
_
]%41 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%42 = bitcast double* %41 to i64*
-double*8B

	full_text

double* %41
†getelementptr8Bå
â
	full_text|
z
x%43 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 0, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Abitcast8B4
2
	full_text%
#
!%44 = bitcast double* %43 to i64*
-double*8B

	full_text

double* %43
Hload8B>
<
	full_text/
-
+%45 = load i64, i64* %44, align 8, !tbaa !8
'i64*8B

	full_text


i64* %44
Égetelementptr8Bp
n
	full_texta
_
]%46 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%47 = bitcast double* %46 to i64*
-double*8B

	full_text

double* %46
†getelementptr8Bå
â
	full_text|
z
x%48 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 0, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Abitcast8B4
2
	full_text%
#
!%49 = bitcast double* %48 to i64*
-double*8B

	full_text

double* %48
Hload8B>
<
	full_text/
-
+%50 = load i64, i64* %49, align 8, !tbaa !8
'i64*8B

	full_text


i64* %49
Égetelementptr8Bp
n
	full_texta
_
]%51 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%52 = bitcast double* %51 to i64*
-double*8B

	full_text

double* %51
†getelementptr8Bå
â
	full_text|
z
x%53 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 0, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Abitcast8B4
2
	full_text%
#
!%54 = bitcast double* %53 to i64*
-double*8B

	full_text

double* %53
Hload8B>
<
	full_text/
-
+%55 = load i64, i64* %54, align 8, !tbaa !8
'i64*8B

	full_text


i64* %54
Égetelementptr8Bp
n
	full_texta
_
]%56 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%57 = bitcast double* %56 to i64*
-double*8B

	full_text

double* %56
{getelementptr8Bh
f
	full_textY
W
U%58 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Gbitcast8B:
8
	full_text+
)
'%59 = bitcast [5 x double]* %58 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %58
Hstore8B=
;
	full_text.
,
*store i64 %50, i64* %59, align 8, !tbaa !8
%i648B

	full_text
	
i64 %50
'i64*8B

	full_text


i64* %59
?bitcast8B2
0
	full_text#
!
%60 = bitcast i64 %50 to double
%i648B

	full_text
	
i64 %50
ãgetelementptr8Bx
v
	full_texti
g
e%61 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 0, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%62 = load double, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
7fmul8B-
+
	full_text

%63 = fmul double %62, %60
+double8B

	full_text


double %62
+double8B

	full_text


double %60
ãgetelementptr8Bx
v
	full_texti
g
e%64 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 0, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nload8BD
B
	full_text5
3
1%65 = load double, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
?bitcast8B2
0
	full_text#
!
%66 = bitcast i64 %40 to double
%i648B

	full_text
	
i64 %40
7fmul8B-
+
	full_text

%67 = fmul double %63, %66
+double8B

	full_text


double %63
+double8B

	full_text


double %66
Çgetelementptr8Bo
m
	full_text`
^
\%68 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Nstore8BC
A
	full_text4
2
0store double %67, double* %68, align 8, !tbaa !8
+double8B

	full_text


double %67
-double*8B

	full_text

double* %68
?bitcast8B2
0
	full_text#
!
%69 = bitcast i64 %45 to double
%i648B

	full_text
	
i64 %45
7fmul8B-
+
	full_text

%70 = fmul double %63, %69
+double8B

	full_text


double %63
+double8B

	full_text


double %69
Çgetelementptr8Bo
m
	full_text`
^
\%71 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Nstore8BC
A
	full_text4
2
0store double %70, double* %71, align 8, !tbaa !8
+double8B

	full_text


double %70
-double*8B

	full_text

double* %71
?bitcast8B2
0
	full_text#
!
%72 = bitcast i64 %55 to double
%i648B

	full_text
	
i64 %55
7fsub8B-
+
	full_text

%73 = fsub double %72, %65
+double8B

	full_text


double %72
+double8B

	full_text


double %65
@fmul8B6
4
	full_text'
%
#%74 = fmul double %73, 4.000000e-01
+double8B

	full_text


double %73
icall8B_
]
	full_textP
N
L%75 = tail call double @llvm.fmuladd.f64(double %60, double %63, double %74)
+double8B

	full_text


double %60
+double8B

	full_text


double %63
+double8B

	full_text


double %74
Çgetelementptr8Bo
m
	full_text`
^
\%76 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 3
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Nstore8BC
A
	full_text4
2
0store double %75, double* %76, align 8, !tbaa !8
+double8B

	full_text


double %75
-double*8B

	full_text

double* %76
@fmul8B6
4
	full_text'
%
#%77 = fmul double %65, 4.000000e-01
+double8B

	full_text


double %65
Afsub8B7
5
	full_text(
&
$%78 = fsub double -0.000000e+00, %77
+double8B

	full_text


double %77
rcall8Bh
f
	full_textY
W
U%79 = tail call double @llvm.fmuladd.f64(double %72, double 1.400000e+00, double %78)
+double8B

	full_text


double %72
+double8B

	full_text


double %78
7fmul8B-
+
	full_text

%80 = fmul double %63, %79
+double8B

	full_text


double %63
+double8B

	full_text


double %79
Çgetelementptr8Bo
m
	full_text`
^
\%81 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 4
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Nstore8BC
A
	full_text4
2
0store double %80, double* %81, align 8, !tbaa !8
+double8B

	full_text


double %80
-double*8B

	full_text

double* %81
Égetelementptr8Bp
n
	full_texta
_
]%82 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%83 = bitcast double* %82 to i64*
-double*8B

	full_text

double* %82
Égetelementptr8Bp
n
	full_texta
_
]%84 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%85 = bitcast double* %84 to i64*
-double*8B

	full_text

double* %84
Istore8B>
<
	full_text/
-
+store i64 %35, i64* %85, align 16, !tbaa !8
%i648B

	full_text
	
i64 %35
'i64*8B

	full_text


i64* %85
Égetelementptr8Bp
n
	full_texta
_
]%86 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%87 = bitcast double* %86 to i64*
-double*8B

	full_text

double* %86
Hstore8B=
;
	full_text.
,
*store i64 %40, i64* %87, align 8, !tbaa !8
%i648B

	full_text
	
i64 %40
'i64*8B

	full_text


i64* %87
Égetelementptr8Bp
n
	full_texta
_
]%88 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%89 = bitcast double* %88 to i64*
-double*8B

	full_text

double* %88
Istore8B>
<
	full_text/
-
+store i64 %45, i64* %89, align 16, !tbaa !8
%i648B

	full_text
	
i64 %45
'i64*8B

	full_text


i64* %89
Égetelementptr8Bp
n
	full_texta
_
]%90 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%91 = bitcast double* %90 to i64*
-double*8B

	full_text

double* %90
Hstore8B=
;
	full_text.
,
*store i64 %50, i64* %91, align 8, !tbaa !8
%i648B

	full_text
	
i64 %50
'i64*8B

	full_text


i64* %91
Égetelementptr8Bp
n
	full_texta
_
]%92 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Abitcast8B4
2
	full_text%
#
!%93 = bitcast double* %92 to i64*
-double*8B

	full_text

double* %92
Istore8B>
<
	full_text/
-
+store i64 %55, i64* %93, align 16, !tbaa !8
%i648B

	full_text
	
i64 %55
'i64*8B

	full_text


i64* %93
Wbitcast8BJ
H
	full_text;
9
7%94 = bitcast double* %1 to [65 x [65 x [5 x double]]]*
pgetelementptr8B]
[
	full_textN
L
J%95 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
`getelementptr8BM
K
	full_text>
<
:%96 = getelementptr inbounds double, double* %0, i64 21125
Xbitcast8BK
I
	full_text<
:
8%97 = bitcast double* %96 to [65 x [65 x [5 x double]]]*
-double*8B

	full_text

double* %96
ôgetelementptr8BÖ
Ç
	full_textu
s
q%98 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %97, i64 0, i64 %30, i64 %32
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %97
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Gbitcast8B:
8
	full_text+
)
'%99 = bitcast [5 x double]* %98 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %98
Iload8B?
=
	full_text0
.
,%100 = load i64, i64* %99, align 8, !tbaa !8
'i64*8B

	full_text


i64* %99
Istore8B>
<
	full_text/
-
+store i64 %100, i64* %37, align 8, !tbaa !8
&i648B

	full_text


i64 %100
'i64*8B

	full_text


i64* %37
°getelementptr8Bç
ä
	full_text}
{
y%101 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %97, i64 0, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %97
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%102 = bitcast double* %101 to i64*
.double*8B

	full_text

double* %101
Jload8B@
>
	full_text1
/
-%103 = load i64, i64* %102, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %102
Istore8B>
<
	full_text/
-
+store i64 %103, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %103
'i64*8B

	full_text


i64* %42
°getelementptr8Bç
ä
	full_text}
{
y%104 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %97, i64 0, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %97
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%105 = bitcast double* %104 to i64*
.double*8B

	full_text

double* %104
Jload8B@
>
	full_text1
/
-%106 = load i64, i64* %105, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %105
Istore8B>
<
	full_text/
-
+store i64 %106, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %106
'i64*8B

	full_text


i64* %47
°getelementptr8Bç
ä
	full_text}
{
y%107 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %97, i64 0, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %97
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%108 = bitcast double* %107 to i64*
.double*8B

	full_text

double* %107
Jload8B@
>
	full_text1
/
-%109 = load i64, i64* %108, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %108
Istore8B>
<
	full_text/
-
+store i64 %109, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %109
'i64*8B

	full_text


i64* %52
°getelementptr8Bç
ä
	full_text}
{
y%110 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %97, i64 0, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %97
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%111 = bitcast double* %110 to i64*
.double*8B

	full_text

double* %110
Jload8B@
>
	full_text1
/
-%112 = load i64, i64* %111, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %111
Istore8B>
<
	full_text/
-
+store i64 %112, i64* %57, align 8, !tbaa !8
&i648B

	full_text


i64 %112
'i64*8B

	full_text


i64* %57
|getelementptr8Bi
g
	full_textZ
X
V%113 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Ibitcast8B<
:
	full_text-
+
)%114 = bitcast [5 x double]* %113 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %113
Kstore8B@
>
	full_text1
/
-store i64 %109, i64* %114, align 16, !tbaa !8
&i648B

	full_text


i64 %109
(i64*8B

	full_text

	i64* %114
Abitcast8B4
2
	full_text%
#
!%115 = bitcast i64 %109 to double
&i648B

	full_text


i64 %109
`getelementptr8BM
K
	full_text>
<
:%116 = getelementptr inbounds double, double* %3, i64 4225
Tbitcast8BG
E
	full_text8
6
4%117 = bitcast double* %116 to [65 x [65 x double]]*
.double*8B

	full_text

double* %116
çgetelementptr8Bz
x
	full_textk
i
g%118 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %117, i64 0, i64 %30, i64 %32
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %117
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%119 = load double, double* %118, align 8, !tbaa !8
.double*8B

	full_text

double* %118
:fmul8B0
.
	full_text!

%120 = fmul double %119, %115
,double8B

	full_text

double %119
,double8B

	full_text

double %115
`getelementptr8BM
K
	full_text>
<
:%121 = getelementptr inbounds double, double* %2, i64 4225
Tbitcast8BG
E
	full_text8
6
4%122 = bitcast double* %121 to [65 x [65 x double]]*
.double*8B

	full_text

double* %121
çgetelementptr8Bz
x
	full_textk
i
g%123 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %122, i64 0, i64 %30, i64 %32
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %122
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%124 = load double, double* %123, align 8, !tbaa !8
.double*8B

	full_text

double* %123
Abitcast8B4
2
	full_text%
#
!%125 = bitcast i64 %103 to double
&i648B

	full_text


i64 %103
:fmul8B0
.
	full_text!

%126 = fmul double %120, %125
,double8B

	full_text

double %120
,double8B

	full_text

double %125
Égetelementptr8Bp
n
	full_texta
_
]%127 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Pstore8BE
C
	full_text6
4
2store double %126, double* %127, align 8, !tbaa !8
,double8B

	full_text

double %126
.double*8B

	full_text

double* %127
Abitcast8B4
2
	full_text%
#
!%128 = bitcast i64 %106 to double
&i648B

	full_text


i64 %106
:fmul8B0
.
	full_text!

%129 = fmul double %120, %128
,double8B

	full_text

double %120
,double8B

	full_text

double %128
Égetelementptr8Bp
n
	full_texta
_
]%130 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Qstore8BF
D
	full_text7
5
3store double %129, double* %130, align 16, !tbaa !8
,double8B

	full_text

double %129
.double*8B

	full_text

double* %130
Abitcast8B4
2
	full_text%
#
!%131 = bitcast i64 %112 to double
&i648B

	full_text


i64 %112
:fsub8B0
.
	full_text!

%132 = fsub double %131, %124
,double8B

	full_text

double %131
,double8B

	full_text

double %124
Bfmul8B8
6
	full_text)
'
%%133 = fmul double %132, 4.000000e-01
,double8B

	full_text

double %132
mcall8Bc
a
	full_textT
R
P%134 = tail call double @llvm.fmuladd.f64(double %115, double %120, double %133)
,double8B

	full_text

double %115
,double8B

	full_text

double %120
,double8B

	full_text

double %133
Égetelementptr8Bp
n
	full_texta
_
]%135 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 3
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Pstore8BE
C
	full_text6
4
2store double %134, double* %135, align 8, !tbaa !8
,double8B

	full_text

double %134
.double*8B

	full_text

double* %135
Bfmul8B8
6
	full_text)
'
%%136 = fmul double %124, 4.000000e-01
,double8B

	full_text

double %124
Cfsub8B9
7
	full_text*
(
&%137 = fsub double -0.000000e+00, %136
,double8B

	full_text

double %136
ucall8Bk
i
	full_text\
Z
X%138 = tail call double @llvm.fmuladd.f64(double %131, double 1.400000e+00, double %137)
,double8B

	full_text

double %131
,double8B

	full_text

double %137
:fmul8B0
.
	full_text!

%139 = fmul double %120, %138
,double8B

	full_text

double %120
,double8B

	full_text

double %138
Égetelementptr8Bp
n
	full_texta
_
]%140 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 4
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Qstore8BF
D
	full_text7
5
3store double %139, double* %140, align 16, !tbaa !8
,double8B

	full_text

double %139
.double*8B

	full_text

double* %140
:fmul8B0
.
	full_text!

%141 = fmul double %119, %125
,double8B

	full_text

double %119
,double8B

	full_text

double %125
:fmul8B0
.
	full_text!

%142 = fmul double %119, %128
,double8B

	full_text

double %119
,double8B

	full_text

double %128
:fmul8B0
.
	full_text!

%143 = fmul double %119, %131
,double8B

	full_text

double %119
,double8B

	full_text

double %131
8fmul8B.
,
	full_text

%144 = fmul double %62, %66
+double8B

	full_text


double %62
+double8B

	full_text


double %66
8fmul8B.
,
	full_text

%145 = fmul double %62, %69
+double8B

	full_text


double %62
+double8B

	full_text


double %69
8fmul8B.
,
	full_text

%146 = fmul double %62, %72
+double8B

	full_text


double %62
+double8B

	full_text


double %72
:fsub8B0
.
	full_text!

%147 = fsub double %141, %144
,double8B

	full_text

double %141
,double8B

	full_text

double %144
Bfmul8B8
6
	full_text)
'
%%148 = fmul double %147, 6.300000e+01
,double8B

	full_text

double %147
Ñgetelementptr8Bq
o
	full_textb
`
^%149 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 1
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %148, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %148
.double*8B

	full_text

double* %149
:fsub8B0
.
	full_text!

%150 = fsub double %142, %145
,double8B

	full_text

double %142
,double8B

	full_text

double %145
Bfmul8B8
6
	full_text)
'
%%151 = fmul double %150, 6.300000e+01
,double8B

	full_text

double %150
Ñgetelementptr8Bq
o
	full_textb
`
^%152 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 2
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %151, double* %152, align 8, !tbaa !8
,double8B

	full_text

double %151
.double*8B

	full_text

double* %152
9fsub8B/
-
	full_text 

%153 = fsub double %120, %63
,double8B

	full_text

double %120
+double8B

	full_text


double %63
Bfmul8B8
6
	full_text)
'
%%154 = fmul double %153, 8.400000e+01
,double8B

	full_text

double %153
Ñgetelementptr8Bq
o
	full_textb
`
^%155 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 3
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %154, double* %155, align 8, !tbaa !8
,double8B

	full_text

double %154
.double*8B

	full_text

double* %155
:fmul8B0
.
	full_text!

%156 = fmul double %142, %142
,double8B

	full_text

double %142
,double8B

	full_text

double %142
mcall8Bc
a
	full_textT
R
P%157 = tail call double @llvm.fmuladd.f64(double %141, double %141, double %156)
,double8B

	full_text

double %141
,double8B

	full_text

double %141
,double8B

	full_text

double %156
mcall8Bc
a
	full_textT
R
P%158 = tail call double @llvm.fmuladd.f64(double %120, double %120, double %157)
,double8B

	full_text

double %120
,double8B

	full_text

double %120
,double8B

	full_text

double %157
:fmul8B0
.
	full_text!

%159 = fmul double %145, %145
,double8B

	full_text

double %145
,double8B

	full_text

double %145
mcall8Bc
a
	full_textT
R
P%160 = tail call double @llvm.fmuladd.f64(double %144, double %144, double %159)
,double8B

	full_text

double %144
,double8B

	full_text

double %144
,double8B

	full_text

double %159
kcall8Ba
_
	full_textR
P
N%161 = tail call double @llvm.fmuladd.f64(double %63, double %63, double %160)
+double8B

	full_text


double %63
+double8B

	full_text


double %63
,double8B

	full_text

double %160
:fsub8B0
.
	full_text!

%162 = fsub double %158, %161
,double8B

	full_text

double %158
,double8B

	full_text

double %161
8fmul8B.
,
	full_text

%163 = fmul double %63, %63
+double8B

	full_text


double %63
+double8B

	full_text


double %63
Cfsub8B9
7
	full_text*
(
&%164 = fsub double -0.000000e+00, %163
,double8B

	full_text

double %163
mcall8Bc
a
	full_textT
R
P%165 = tail call double @llvm.fmuladd.f64(double %120, double %120, double %164)
,double8B

	full_text

double %120
,double8B

	full_text

double %120
,double8B

	full_text

double %164
Bfmul8B8
6
	full_text)
'
%%166 = fmul double %165, 1.050000e+01
,double8B

	full_text

double %165
{call8Bq
o
	full_textb
`
^%167 = tail call double @llvm.fmuladd.f64(double %162, double 0xC03E3D70A3D70A3B, double %166)
,double8B

	full_text

double %162
,double8B

	full_text

double %166
:fsub8B0
.
	full_text!

%168 = fsub double %143, %146
,double8B

	full_text

double %143
,double8B

	full_text

double %146
{call8Bq
o
	full_textb
`
^%169 = tail call double @llvm.fmuladd.f64(double %168, double 0x405EDEB851EB851E, double %167)
,double8B

	full_text

double %168
,double8B

	full_text

double %167
Ñgetelementptr8Bq
o
	full_textb
`
^%170 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 4
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %169, double* %170, align 8, !tbaa !8
,double8B

	full_text

double %169
.double*8B

	full_text

double* %170
agetelementptr8BN
L
	full_text?
=
;%171 = getelementptr inbounds double, double* %0, i64 42250
Zbitcast8BM
K
	full_text>
<
:%172 = bitcast double* %171 to [65 x [65 x [5 x double]]]*
.double*8B

	full_text

double* %171
õgetelementptr8Bá
Ñ
	full_textw
u
s%173 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %172, i64 0, i64 %30, i64 %32
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %172
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%174 = bitcast [5 x double]* %173 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %173
Jload8B@
>
	full_text1
/
-%175 = load i64, i64* %174, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %174
}getelementptr8Bj
h
	full_text[
Y
W%176 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ibitcast8B<
:
	full_text-
+
)%177 = bitcast [5 x double]* %176 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %176
Kstore8B@
>
	full_text1
/
-store i64 %175, i64* %177, align 16, !tbaa !8
&i648B

	full_text


i64 %175
(i64*8B

	full_text

	i64* %177
¢getelementptr8Bé
ã
	full_text~
|
z%178 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %172, i64 0, i64 %30, i64 %32, i64 1
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %172
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%179 = bitcast double* %178 to i64*
.double*8B

	full_text

double* %178
Jload8B@
>
	full_text1
/
-%180 = load i64, i64* %179, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %179
Ñgetelementptr8Bq
o
	full_textb
`
^%181 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%182 = bitcast double* %181 to i64*
.double*8B

	full_text

double* %181
Jstore8B?
=
	full_text0
.
,store i64 %180, i64* %182, align 8, !tbaa !8
&i648B

	full_text


i64 %180
(i64*8B

	full_text

	i64* %182
¢getelementptr8Bé
ã
	full_text~
|
z%183 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %172, i64 0, i64 %30, i64 %32, i64 2
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %172
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%184 = bitcast double* %183 to i64*
.double*8B

	full_text

double* %183
Jload8B@
>
	full_text1
/
-%185 = load i64, i64* %184, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %184
Ñgetelementptr8Bq
o
	full_textb
`
^%186 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%187 = bitcast double* %186 to i64*
.double*8B

	full_text

double* %186
Kstore8B@
>
	full_text1
/
-store i64 %185, i64* %187, align 16, !tbaa !8
&i648B

	full_text


i64 %185
(i64*8B

	full_text

	i64* %187
¢getelementptr8Bé
ã
	full_text~
|
z%188 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %172, i64 0, i64 %30, i64 %32, i64 3
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %172
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%189 = bitcast double* %188 to i64*
.double*8B

	full_text

double* %188
Jload8B@
>
	full_text1
/
-%190 = load i64, i64* %189, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %189
Ñgetelementptr8Bq
o
	full_textb
`
^%191 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%192 = bitcast double* %191 to i64*
.double*8B

	full_text

double* %191
Jstore8B?
=
	full_text0
.
,store i64 %190, i64* %192, align 8, !tbaa !8
&i648B

	full_text


i64 %190
(i64*8B

	full_text

	i64* %192
¢getelementptr8Bé
ã
	full_text~
|
z%193 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %172, i64 0, i64 %30, i64 %32, i64 4
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %172
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%194 = bitcast double* %193 to i64*
.double*8B

	full_text

double* %193
Jload8B@
>
	full_text1
/
-%195 = load i64, i64* %194, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %194
Ñgetelementptr8Bq
o
	full_textb
`
^%196 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%197 = bitcast double* %196 to i64*
.double*8B

	full_text

double* %196
Kstore8B@
>
	full_text1
/
-store i64 %195, i64* %197, align 16, !tbaa !8
&i648B

	full_text


i64 %195
(i64*8B

	full_text

	i64* %197
Ñgetelementptr8Bq
o
	full_textb
`
^%198 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%199 = bitcast double* %198 to i64*
.double*8B

	full_text

double* %198
Jload8B@
>
	full_text1
/
-%200 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
Ñgetelementptr8Bq
o
	full_textb
`
^%201 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Nbitcast8BA
?
	full_text2
0
.%202 = bitcast [5 x [5 x double]]* %12 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Kstore8B@
>
	full_text1
/
-store i64 %200, i64* %202, align 16, !tbaa !8
&i648B

	full_text


i64 %200
(i64*8B

	full_text

	i64* %202
Jload8B@
>
	full_text1
/
-%203 = load i64, i64* %85, align 16, !tbaa !8
'i64*8B

	full_text


i64* %85
Jstore8B?
=
	full_text0
.
,store i64 %203, i64* %199, align 8, !tbaa !8
&i648B

	full_text


i64 %203
(i64*8B

	full_text

	i64* %199
Iload8B?
=
	full_text0
.
,%204 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
Jstore8B?
=
	full_text0
.
,store i64 %204, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %204
'i64*8B

	full_text


i64* %85
Ñgetelementptr8Bq
o
	full_textb
`
^%205 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%206 = bitcast double* %205 to i64*
.double*8B

	full_text

double* %205
Istore8B>
<
	full_text/
-
+store i64 %175, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %175
'i64*8B

	full_text


i64* %83
Égetelementptr8Bp
n
	full_texta
_
]%207 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%208 = bitcast double* %207 to i64*
.double*8B

	full_text

double* %207
Jload8B@
>
	full_text1
/
-%209 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Égetelementptr8Bp
n
	full_texta
_
]%210 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Mbitcast8B@
>
	full_text1
/
-%211 = bitcast [3 x [5 x double]]* %8 to i64*
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Kstore8B@
>
	full_text1
/
-store i64 %209, i64* %211, align 16, !tbaa !8
&i648B

	full_text


i64 %209
(i64*8B

	full_text

	i64* %211
Égetelementptr8Bp
n
	full_texta
_
]%212 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%213 = bitcast double* %212 to i64*
.double*8B

	full_text

double* %212
Kload8BA
?
	full_text2
0
.%214 = load i64, i64* %213, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %213
Jstore8B?
=
	full_text0
.
,store i64 %214, i64* %208, align 8, !tbaa !8
&i648B

	full_text


i64 %214
(i64*8B

	full_text

	i64* %208
Ñgetelementptr8Bq
o
	full_textb
`
^%215 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 0
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%216 = bitcast double* %215 to i64*
.double*8B

	full_text

double* %215
Jload8B@
>
	full_text1
/
-%217 = load i64, i64* %216, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %216
Nbitcast8BA
?
	full_text2
0
.%218 = bitcast [2 x [5 x double]]* %10 to i64*
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Kstore8B@
>
	full_text1
/
-store i64 %217, i64* %218, align 16, !tbaa !8
&i648B

	full_text


i64 %217
(i64*8B

	full_text

	i64* %218
Ñgetelementptr8Bq
o
	full_textb
`
^%219 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%220 = bitcast double* %219 to i64*
.double*8B

	full_text

double* %219
Jload8B@
>
	full_text1
/
-%221 = load i64, i64* %220, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %220
Ñgetelementptr8Bq
o
	full_textb
`
^%222 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%223 = bitcast double* %222 to i64*
.double*8B

	full_text

double* %222
Jstore8B?
=
	full_text0
.
,store i64 %221, i64* %223, align 8, !tbaa !8
&i648B

	full_text


i64 %221
(i64*8B

	full_text

	i64* %223
Iload8B?
=
	full_text0
.
,%224 = load i64, i64* %87, align 8, !tbaa !8
'i64*8B

	full_text


i64* %87
Jstore8B?
=
	full_text0
.
,store i64 %224, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %224
(i64*8B

	full_text

	i64* %220
Iload8B?
=
	full_text0
.
,%225 = load i64, i64* %42, align 8, !tbaa !8
'i64*8B

	full_text


i64* %42
Istore8B>
<
	full_text/
-
+store i64 %225, i64* %87, align 8, !tbaa !8
&i648B

	full_text


i64 %225
'i64*8B

	full_text


i64* %87
Istore8B>
<
	full_text/
-
+store i64 %180, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %180
'i64*8B

	full_text


i64* %42
Bbitcast8B5
3
	full_text&
$
"%226 = bitcast double* %68 to i64*
-double*8B

	full_text

double* %68
Jload8B@
>
	full_text1
/
-%227 = load i64, i64* %226, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %226
Égetelementptr8Bp
n
	full_texta
_
]%228 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%229 = bitcast double* %228 to i64*
.double*8B

	full_text

double* %228
Jstore8B?
=
	full_text0
.
,store i64 %227, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %227
(i64*8B

	full_text

	i64* %229
Cbitcast8B6
4
	full_text'
%
#%230 = bitcast double* %127 to i64*
.double*8B

	full_text

double* %127
Jload8B@
>
	full_text1
/
-%231 = load i64, i64* %230, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %230
Jstore8B?
=
	full_text0
.
,store i64 %231, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %231
(i64*8B

	full_text

	i64* %226
Cbitcast8B6
4
	full_text'
%
#%232 = bitcast double* %149 to i64*
.double*8B

	full_text

double* %149
Jload8B@
>
	full_text1
/
-%233 = load i64, i64* %232, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %232
Ñgetelementptr8Bq
o
	full_textb
`
^%234 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 1
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%235 = bitcast double* %234 to i64*
.double*8B

	full_text

double* %234
Jstore8B?
=
	full_text0
.
,store i64 %233, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %233
(i64*8B

	full_text

	i64* %235
Ñgetelementptr8Bq
o
	full_textb
`
^%236 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%237 = bitcast double* %236 to i64*
.double*8B

	full_text

double* %236
Jload8B@
>
	full_text1
/
-%238 = load i64, i64* %237, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %237
Ñgetelementptr8Bq
o
	full_textb
`
^%239 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%240 = bitcast double* %239 to i64*
.double*8B

	full_text

double* %239
Kstore8B@
>
	full_text1
/
-store i64 %238, i64* %240, align 16, !tbaa !8
&i648B

	full_text


i64 %238
(i64*8B

	full_text

	i64* %240
Jload8B@
>
	full_text1
/
-%241 = load i64, i64* %89, align 16, !tbaa !8
'i64*8B

	full_text


i64* %89
Jstore8B?
=
	full_text0
.
,store i64 %241, i64* %237, align 8, !tbaa !8
&i648B

	full_text


i64 %241
(i64*8B

	full_text

	i64* %237
Iload8B?
=
	full_text0
.
,%242 = load i64, i64* %47, align 8, !tbaa !8
'i64*8B

	full_text


i64* %47
Jstore8B?
=
	full_text0
.
,store i64 %242, i64* %89, align 16, !tbaa !8
&i648B

	full_text


i64 %242
'i64*8B

	full_text


i64* %89
Istore8B>
<
	full_text/
-
+store i64 %185, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %185
'i64*8B

	full_text


i64* %47
Bbitcast8B5
3
	full_text&
$
"%243 = bitcast double* %71 to i64*
-double*8B

	full_text

double* %71
Jload8B@
>
	full_text1
/
-%244 = load i64, i64* %243, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %243
Égetelementptr8Bp
n
	full_texta
_
]%245 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%246 = bitcast double* %245 to i64*
.double*8B

	full_text

double* %245
Kstore8B@
>
	full_text1
/
-store i64 %244, i64* %246, align 16, !tbaa !8
&i648B

	full_text


i64 %244
(i64*8B

	full_text

	i64* %246
Cbitcast8B6
4
	full_text'
%
#%247 = bitcast double* %130 to i64*
.double*8B

	full_text

double* %130
Kload8BA
?
	full_text2
0
.%248 = load i64, i64* %247, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %247
Jstore8B?
=
	full_text0
.
,store i64 %248, i64* %243, align 8, !tbaa !8
&i648B

	full_text


i64 %248
(i64*8B

	full_text

	i64* %243
Cbitcast8B6
4
	full_text'
%
#%249 = bitcast double* %152 to i64*
.double*8B

	full_text

double* %152
Jload8B@
>
	full_text1
/
-%250 = load i64, i64* %249, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %249
Ñgetelementptr8Bq
o
	full_textb
`
^%251 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 2
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%252 = bitcast double* %251 to i64*
.double*8B

	full_text

double* %251
Kstore8B@
>
	full_text1
/
-store i64 %250, i64* %252, align 16, !tbaa !8
&i648B

	full_text


i64 %250
(i64*8B

	full_text

	i64* %252
Ñgetelementptr8Bq
o
	full_textb
`
^%253 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%254 = bitcast double* %253 to i64*
.double*8B

	full_text

double* %253
Jload8B@
>
	full_text1
/
-%255 = load i64, i64* %254, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %254
Ñgetelementptr8Bq
o
	full_textb
`
^%256 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%257 = bitcast double* %256 to i64*
.double*8B

	full_text

double* %256
Jstore8B?
=
	full_text0
.
,store i64 %255, i64* %257, align 8, !tbaa !8
&i648B

	full_text


i64 %255
(i64*8B

	full_text

	i64* %257
Iload8B?
=
	full_text0
.
,%258 = load i64, i64* %91, align 8, !tbaa !8
'i64*8B

	full_text


i64* %91
Jstore8B?
=
	full_text0
.
,store i64 %258, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %258
(i64*8B

	full_text

	i64* %254
Iload8B?
=
	full_text0
.
,%259 = load i64, i64* %52, align 8, !tbaa !8
'i64*8B

	full_text


i64* %52
Istore8B>
<
	full_text/
-
+store i64 %259, i64* %91, align 8, !tbaa !8
&i648B

	full_text


i64 %259
'i64*8B

	full_text


i64* %91
Istore8B>
<
	full_text/
-
+store i64 %190, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %190
'i64*8B

	full_text


i64* %52
Bbitcast8B5
3
	full_text&
$
"%260 = bitcast double* %76 to i64*
-double*8B

	full_text

double* %76
Jload8B@
>
	full_text1
/
-%261 = load i64, i64* %260, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %260
Égetelementptr8Bp
n
	full_texta
_
]%262 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 3
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%263 = bitcast double* %262 to i64*
.double*8B

	full_text

double* %262
Jstore8B?
=
	full_text0
.
,store i64 %261, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %261
(i64*8B

	full_text

	i64* %263
Cbitcast8B6
4
	full_text'
%
#%264 = bitcast double* %135 to i64*
.double*8B

	full_text

double* %135
Jload8B@
>
	full_text1
/
-%265 = load i64, i64* %264, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %264
Jstore8B?
=
	full_text0
.
,store i64 %265, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %265
(i64*8B

	full_text

	i64* %260
Cbitcast8B6
4
	full_text'
%
#%266 = bitcast double* %155 to i64*
.double*8B

	full_text

double* %155
Jload8B@
>
	full_text1
/
-%267 = load i64, i64* %266, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %266
Ñgetelementptr8Bq
o
	full_textb
`
^%268 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 3
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%269 = bitcast double* %268 to i64*
.double*8B

	full_text

double* %268
Jstore8B?
=
	full_text0
.
,store i64 %267, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %267
(i64*8B

	full_text

	i64* %269
Ñgetelementptr8Bq
o
	full_textb
`
^%270 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%271 = bitcast double* %270 to i64*
.double*8B

	full_text

double* %270
Jload8B@
>
	full_text1
/
-%272 = load i64, i64* %271, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %271
Ñgetelementptr8Bq
o
	full_textb
`
^%273 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%274 = bitcast double* %273 to i64*
.double*8B

	full_text

double* %273
Kstore8B@
>
	full_text1
/
-store i64 %272, i64* %274, align 16, !tbaa !8
&i648B

	full_text


i64 %272
(i64*8B

	full_text

	i64* %274
Jload8B@
>
	full_text1
/
-%275 = load i64, i64* %93, align 16, !tbaa !8
'i64*8B

	full_text


i64* %93
Jstore8B?
=
	full_text0
.
,store i64 %275, i64* %271, align 8, !tbaa !8
&i648B

	full_text


i64 %275
(i64*8B

	full_text

	i64* %271
Iload8B?
=
	full_text0
.
,%276 = load i64, i64* %57, align 8, !tbaa !8
'i64*8B

	full_text


i64* %57
Jstore8B?
=
	full_text0
.
,store i64 %276, i64* %93, align 16, !tbaa !8
&i648B

	full_text


i64 %276
'i64*8B

	full_text


i64* %93
Kload8BA
?
	full_text2
0
.%277 = load i64, i64* %197, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %197
Istore8B>
<
	full_text/
-
+store i64 %277, i64* %57, align 8, !tbaa !8
&i648B

	full_text


i64 %277
'i64*8B

	full_text


i64* %57
Bbitcast8B5
3
	full_text&
$
"%278 = bitcast double* %81 to i64*
-double*8B

	full_text

double* %81
Jload8B@
>
	full_text1
/
-%279 = load i64, i64* %278, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %278
Égetelementptr8Bp
n
	full_texta
_
]%280 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 4
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%281 = bitcast double* %280 to i64*
.double*8B

	full_text

double* %280
Kstore8B@
>
	full_text1
/
-store i64 %279, i64* %281, align 16, !tbaa !8
&i648B

	full_text


i64 %279
(i64*8B

	full_text

	i64* %281
Cbitcast8B6
4
	full_text'
%
#%282 = bitcast double* %140 to i64*
.double*8B

	full_text

double* %140
Kload8BA
?
	full_text2
0
.%283 = load i64, i64* %282, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %282
Jstore8B?
=
	full_text0
.
,store i64 %283, i64* %278, align 8, !tbaa !8
&i648B

	full_text


i64 %283
(i64*8B

	full_text

	i64* %278
Cbitcast8B6
4
	full_text'
%
#%284 = bitcast double* %170 to i64*
.double*8B

	full_text

double* %170
Jload8B@
>
	full_text1
/
-%285 = load i64, i64* %284, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %284
Ñgetelementptr8Bq
o
	full_textb
`
^%286 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 4
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%287 = bitcast double* %286 to i64*
.double*8B

	full_text

double* %286
Kstore8B@
>
	full_text1
/
-store i64 %285, i64* %287, align 16, !tbaa !8
&i648B

	full_text


i64 %285
(i64*8B

	full_text

	i64* %287
agetelementptr8BN
L
	full_text?
=
;%288 = getelementptr inbounds double, double* %0, i64 63375
Zbitcast8BM
K
	full_text>
<
:%289 = bitcast double* %288 to [65 x [65 x [5 x double]]]*
.double*8B

	full_text

double* %288
õgetelementptr8Bá
Ñ
	full_textw
u
s%290 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %289, i64 0, i64 %30, i64 %32
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %289
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%291 = bitcast [5 x double]* %290 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %290
Jload8B@
>
	full_text1
/
-%292 = load i64, i64* %291, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %291
Kstore8B@
>
	full_text1
/
-store i64 %292, i64* %177, align 16, !tbaa !8
&i648B

	full_text


i64 %292
(i64*8B

	full_text

	i64* %177
¢getelementptr8Bé
ã
	full_text~
|
z%293 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %289, i64 0, i64 %30, i64 %32, i64 1
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %289
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%294 = bitcast double* %293 to i64*
.double*8B

	full_text

double* %293
Jload8B@
>
	full_text1
/
-%295 = load i64, i64* %294, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %294
Jstore8B?
=
	full_text0
.
,store i64 %295, i64* %182, align 8, !tbaa !8
&i648B

	full_text


i64 %295
(i64*8B

	full_text

	i64* %182
¢getelementptr8Bé
ã
	full_text~
|
z%296 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %289, i64 0, i64 %30, i64 %32, i64 2
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %289
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%297 = bitcast double* %296 to i64*
.double*8B

	full_text

double* %296
Jload8B@
>
	full_text1
/
-%298 = load i64, i64* %297, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %297
Kstore8B@
>
	full_text1
/
-store i64 %298, i64* %187, align 16, !tbaa !8
&i648B

	full_text


i64 %298
(i64*8B

	full_text

	i64* %187
¢getelementptr8Bé
ã
	full_text~
|
z%299 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %289, i64 0, i64 %30, i64 %32, i64 3
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %289
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%300 = bitcast double* %299 to i64*
.double*8B

	full_text

double* %299
Jload8B@
>
	full_text1
/
-%301 = load i64, i64* %300, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %300
Jstore8B?
=
	full_text0
.
,store i64 %301, i64* %192, align 8, !tbaa !8
&i648B

	full_text


i64 %301
(i64*8B

	full_text

	i64* %192
¢getelementptr8Bé
ã
	full_text~
|
z%302 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %289, i64 0, i64 %30, i64 %32, i64 4
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %289
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%303 = bitcast double* %302 to i64*
.double*8B

	full_text

double* %302
Jload8B@
>
	full_text1
/
-%304 = load i64, i64* %303, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %303
Kstore8B@
>
	full_text1
/
-store i64 %304, i64* %197, align 16, !tbaa !8
&i648B

	full_text


i64 %304
(i64*8B

	full_text

	i64* %197
Kstore8B@
>
	full_text1
/
-store i64 %190, i64* %114, align 16, !tbaa !8
&i648B

	full_text


i64 %190
(i64*8B

	full_text

	i64* %114
Abitcast8B4
2
	full_text%
#
!%305 = bitcast i64 %190 to double
&i648B

	full_text


i64 %190
`getelementptr8BM
K
	full_text>
<
:%306 = getelementptr inbounds double, double* %3, i64 8450
Tbitcast8BG
E
	full_text8
6
4%307 = bitcast double* %306 to [65 x [65 x double]]*
.double*8B

	full_text

double* %306
çgetelementptr8Bz
x
	full_textk
i
g%308 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %307, i64 0, i64 %30, i64 %32
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %307
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%309 = load double, double* %308, align 8, !tbaa !8
.double*8B

	full_text

double* %308
:fmul8B0
.
	full_text!

%310 = fmul double %309, %305
,double8B

	full_text

double %309
,double8B

	full_text

double %305
`getelementptr8BM
K
	full_text>
<
:%311 = getelementptr inbounds double, double* %2, i64 8450
Tbitcast8BG
E
	full_text8
6
4%312 = bitcast double* %311 to [65 x [65 x double]]*
.double*8B

	full_text

double* %311
çgetelementptr8Bz
x
	full_textk
i
g%313 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %312, i64 0, i64 %30, i64 %32
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %312
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%314 = load double, double* %313, align 8, !tbaa !8
.double*8B

	full_text

double* %313
Oload8BE
C
	full_text6
4
2%315 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
:fmul8B0
.
	full_text!

%316 = fmul double %310, %315
,double8B

	full_text

double %310
,double8B

	full_text

double %315
Pstore8BE
C
	full_text6
4
2store double %316, double* %127, align 8, !tbaa !8
,double8B

	full_text

double %316
.double*8B

	full_text

double* %127
Abitcast8B4
2
	full_text%
#
!%317 = bitcast i64 %185 to double
&i648B

	full_text


i64 %185
:fmul8B0
.
	full_text!

%318 = fmul double %310, %317
,double8B

	full_text

double %310
,double8B

	full_text

double %317
Qstore8BF
D
	full_text7
5
3store double %318, double* %130, align 16, !tbaa !8
,double8B

	full_text

double %318
.double*8B

	full_text

double* %130
Abitcast8B4
2
	full_text%
#
!%319 = bitcast i64 %277 to double
&i648B

	full_text


i64 %277
:fsub8B0
.
	full_text!

%320 = fsub double %319, %314
,double8B

	full_text

double %319
,double8B

	full_text

double %314
Bfmul8B8
6
	full_text)
'
%%321 = fmul double %320, 4.000000e-01
,double8B

	full_text

double %320
mcall8Bc
a
	full_textT
R
P%322 = tail call double @llvm.fmuladd.f64(double %305, double %310, double %321)
,double8B

	full_text

double %305
,double8B

	full_text

double %310
,double8B

	full_text

double %321
Pstore8BE
C
	full_text6
4
2store double %322, double* %135, align 8, !tbaa !8
,double8B

	full_text

double %322
.double*8B

	full_text

double* %135
Bfmul8B8
6
	full_text)
'
%%323 = fmul double %314, 4.000000e-01
,double8B

	full_text

double %314
Cfsub8B9
7
	full_text*
(
&%324 = fsub double -0.000000e+00, %323
,double8B

	full_text

double %323
ucall8Bk
i
	full_text\
Z
X%325 = tail call double @llvm.fmuladd.f64(double %319, double 1.400000e+00, double %324)
,double8B

	full_text

double %319
,double8B

	full_text

double %324
:fmul8B0
.
	full_text!

%326 = fmul double %310, %325
,double8B

	full_text

double %310
,double8B

	full_text

double %325
Qstore8BF
D
	full_text7
5
3store double %326, double* %140, align 16, !tbaa !8
,double8B

	full_text

double %326
.double*8B

	full_text

double* %140
:fmul8B0
.
	full_text!

%327 = fmul double %309, %315
,double8B

	full_text

double %309
,double8B

	full_text

double %315
:fmul8B0
.
	full_text!

%328 = fmul double %309, %317
,double8B

	full_text

double %309
,double8B

	full_text

double %317
:fmul8B0
.
	full_text!

%329 = fmul double %309, %319
,double8B

	full_text

double %309
,double8B

	full_text

double %319
Oload8BE
C
	full_text6
4
2%330 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
:fmul8B0
.
	full_text!

%331 = fmul double %119, %330
,double8B

	full_text

double %119
,double8B

	full_text

double %330
Pload8BF
D
	full_text7
5
3%332 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
:fmul8B0
.
	full_text!

%333 = fmul double %119, %332
,double8B

	full_text

double %119
,double8B

	full_text

double %332
Abitcast8B4
2
	full_text%
#
!%334 = bitcast i64 %259 to double
&i648B

	full_text


i64 %259
:fmul8B0
.
	full_text!

%335 = fmul double %119, %334
,double8B

	full_text

double %119
,double8B

	full_text

double %334
Abitcast8B4
2
	full_text%
#
!%336 = bitcast i64 %276 to double
&i648B

	full_text


i64 %276
:fmul8B0
.
	full_text!

%337 = fmul double %119, %336
,double8B

	full_text

double %119
,double8B

	full_text

double %336
:fsub8B0
.
	full_text!

%338 = fsub double %327, %331
,double8B

	full_text

double %327
,double8B

	full_text

double %331
Bfmul8B8
6
	full_text)
'
%%339 = fmul double %338, 6.300000e+01
,double8B

	full_text

double %338
Pstore8BE
C
	full_text6
4
2store double %339, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %339
.double*8B

	full_text

double* %149
:fsub8B0
.
	full_text!

%340 = fsub double %328, %333
,double8B

	full_text

double %328
,double8B

	full_text

double %333
Bfmul8B8
6
	full_text)
'
%%341 = fmul double %340, 6.300000e+01
,double8B

	full_text

double %340
Pstore8BE
C
	full_text6
4
2store double %341, double* %152, align 8, !tbaa !8
,double8B

	full_text

double %341
.double*8B

	full_text

double* %152
:fsub8B0
.
	full_text!

%342 = fsub double %310, %335
,double8B

	full_text

double %310
,double8B

	full_text

double %335
Bfmul8B8
6
	full_text)
'
%%343 = fmul double %342, 8.400000e+01
,double8B

	full_text

double %342
Pstore8BE
C
	full_text6
4
2store double %343, double* %155, align 8, !tbaa !8
,double8B

	full_text

double %343
.double*8B

	full_text

double* %155
:fmul8B0
.
	full_text!

%344 = fmul double %328, %328
,double8B

	full_text

double %328
,double8B

	full_text

double %328
mcall8Bc
a
	full_textT
R
P%345 = tail call double @llvm.fmuladd.f64(double %327, double %327, double %344)
,double8B

	full_text

double %327
,double8B

	full_text

double %327
,double8B

	full_text

double %344
mcall8Bc
a
	full_textT
R
P%346 = tail call double @llvm.fmuladd.f64(double %310, double %310, double %345)
,double8B

	full_text

double %310
,double8B

	full_text

double %310
,double8B

	full_text

double %345
:fmul8B0
.
	full_text!

%347 = fmul double %333, %333
,double8B

	full_text

double %333
,double8B

	full_text

double %333
mcall8Bc
a
	full_textT
R
P%348 = tail call double @llvm.fmuladd.f64(double %331, double %331, double %347)
,double8B

	full_text

double %331
,double8B

	full_text

double %331
,double8B

	full_text

double %347
mcall8Bc
a
	full_textT
R
P%349 = tail call double @llvm.fmuladd.f64(double %335, double %335, double %348)
,double8B

	full_text

double %335
,double8B

	full_text

double %335
,double8B

	full_text

double %348
:fsub8B0
.
	full_text!

%350 = fsub double %346, %349
,double8B

	full_text

double %346
,double8B

	full_text

double %349
:fmul8B0
.
	full_text!

%351 = fmul double %335, %335
,double8B

	full_text

double %335
,double8B

	full_text

double %335
Cfsub8B9
7
	full_text*
(
&%352 = fsub double -0.000000e+00, %351
,double8B

	full_text

double %351
mcall8Bc
a
	full_textT
R
P%353 = tail call double @llvm.fmuladd.f64(double %310, double %310, double %352)
,double8B

	full_text

double %310
,double8B

	full_text

double %310
,double8B

	full_text

double %352
Bfmul8B8
6
	full_text)
'
%%354 = fmul double %353, 1.050000e+01
,double8B

	full_text

double %353
{call8Bq
o
	full_textb
`
^%355 = tail call double @llvm.fmuladd.f64(double %350, double 0xC03E3D70A3D70A3B, double %354)
,double8B

	full_text

double %350
,double8B

	full_text

double %354
:fsub8B0
.
	full_text!

%356 = fsub double %329, %337
,double8B

	full_text

double %329
,double8B

	full_text

double %337
{call8Bq
o
	full_textb
`
^%357 = tail call double @llvm.fmuladd.f64(double %356, double 0x405EDEB851EB851E, double %355)
,double8B

	full_text

double %356
,double8B

	full_text

double %355
Pstore8BE
C
	full_text6
4
2store double %357, double* %170, align 8, !tbaa !8
,double8B

	full_text

double %357
.double*8B

	full_text

double* %170
agetelementptr8BN
L
	full_text?
=
;%358 = getelementptr inbounds double, double* %1, i64 21125
Zbitcast8BM
K
	full_text>
<
:%359 = bitcast double* %358 to [65 x [65 x [5 x double]]]*
.double*8B

	full_text

double* %358
¢getelementptr8Bé
ã
	full_text~
|
z%360 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %359, i64 0, i64 %30, i64 %32, i64 0
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %359
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%361 = load double, double* %360, align 8, !tbaa !8
.double*8B

	full_text

double* %360
Qload8BG
E
	full_text8
6
4%362 = load double, double* %210, align 16, !tbaa !8
.double*8B

	full_text

double* %210
:fsub8B0
.
	full_text!

%363 = fsub double %305, %362
,double8B

	full_text

double %305
,double8B

	full_text

double %362
vcall8Bl
j
	full_text]
[
Y%364 = tail call double @llvm.fmuladd.f64(double %363, double -3.150000e+01, double %361)
,double8B

	full_text

double %363
,double8B

	full_text

double %361
¢getelementptr8Bé
ã
	full_text~
|
z%365 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %359, i64 0, i64 %30, i64 %32, i64 1
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %359
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%366 = load double, double* %365, align 8, !tbaa !8
.double*8B

	full_text

double* %365
Pload8BF
D
	full_text7
5
3%367 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
:fsub8B0
.
	full_text!

%368 = fsub double %316, %367
,double8B

	full_text

double %316
,double8B

	full_text

double %367
vcall8Bl
j
	full_text]
[
Y%369 = tail call double @llvm.fmuladd.f64(double %368, double -3.150000e+01, double %366)
,double8B

	full_text

double %368
,double8B

	full_text

double %366
¢getelementptr8Bé
ã
	full_text~
|
z%370 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %359, i64 0, i64 %30, i64 %32, i64 2
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %359
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%371 = load double, double* %370, align 8, !tbaa !8
.double*8B

	full_text

double* %370
Qload8BG
E
	full_text8
6
4%372 = load double, double* %245, align 16, !tbaa !8
.double*8B

	full_text

double* %245
:fsub8B0
.
	full_text!

%373 = fsub double %318, %372
,double8B

	full_text

double %318
,double8B

	full_text

double %372
vcall8Bl
j
	full_text]
[
Y%374 = tail call double @llvm.fmuladd.f64(double %373, double -3.150000e+01, double %371)
,double8B

	full_text

double %373
,double8B

	full_text

double %371
¢getelementptr8Bé
ã
	full_text~
|
z%375 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %359, i64 0, i64 %30, i64 %32, i64 3
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %359
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%376 = load double, double* %375, align 8, !tbaa !8
.double*8B

	full_text

double* %375
Pload8BF
D
	full_text7
5
3%377 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
:fsub8B0
.
	full_text!

%378 = fsub double %322, %377
,double8B

	full_text

double %322
,double8B

	full_text

double %377
vcall8Bl
j
	full_text]
[
Y%379 = tail call double @llvm.fmuladd.f64(double %378, double -3.150000e+01, double %376)
,double8B

	full_text

double %378
,double8B

	full_text

double %376
¢getelementptr8Bé
ã
	full_text~
|
z%380 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %359, i64 0, i64 %30, i64 %32, i64 4
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %359
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%381 = load double, double* %380, align 8, !tbaa !8
.double*8B

	full_text

double* %380
Qload8BG
E
	full_text8
6
4%382 = load double, double* %280, align 16, !tbaa !8
.double*8B

	full_text

double* %280
:fsub8B0
.
	full_text!

%383 = fsub double %326, %382
,double8B

	full_text

double %326
,double8B

	full_text

double %382
vcall8Bl
j
	full_text]
[
Y%384 = tail call double @llvm.fmuladd.f64(double %383, double -3.150000e+01, double %381)
,double8B

	full_text

double %383
,double8B

	full_text

double %381
Pload8BF
D
	full_text7
5
3%385 = load double, double* %198, align 8, !tbaa !8
.double*8B

	full_text

double* %198
Pload8BF
D
	full_text7
5
3%386 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%387 = tail call double @llvm.fmuladd.f64(double %386, double -2.000000e+00, double %385)
,double8B

	full_text

double %386
,double8B

	full_text

double %385
Oload8BE
C
	full_text6
4
2%388 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%389 = fadd double %387, %388
,double8B

	full_text

double %387
,double8B

	full_text

double %388
{call8Bq
o
	full_textb
`
^%390 = tail call double @llvm.fmuladd.f64(double %389, double 0x40AF020000000001, double %364)
,double8B

	full_text

double %389
,double8B

	full_text

double %364
Pload8BF
D
	full_text7
5
3%391 = load double, double* %234, align 8, !tbaa !8
.double*8B

	full_text

double* %234
:fsub8B0
.
	full_text!

%392 = fsub double %339, %391
,double8B

	full_text

double %339
,double8B

	full_text

double %391
{call8Bq
o
	full_textb
`
^%393 = tail call double @llvm.fmuladd.f64(double %392, double 0x4019333333333334, double %369)
,double8B

	full_text

double %392
,double8B

	full_text

double %369
Pload8BF
D
	full_text7
5
3%394 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
vcall8Bl
j
	full_text]
[
Y%395 = tail call double @llvm.fmuladd.f64(double %330, double -2.000000e+00, double %394)
,double8B

	full_text

double %330
,double8B

	full_text

double %394
:fadd8B0
.
	full_text!

%396 = fadd double %315, %395
,double8B

	full_text

double %315
,double8B

	full_text

double %395
{call8Bq
o
	full_textb
`
^%397 = tail call double @llvm.fmuladd.f64(double %396, double 0x40AF020000000001, double %393)
,double8B

	full_text

double %396
,double8B

	full_text

double %393
Qload8BG
E
	full_text8
6
4%398 = load double, double* %251, align 16, !tbaa !8
.double*8B

	full_text

double* %251
:fsub8B0
.
	full_text!

%399 = fsub double %341, %398
,double8B

	full_text

double %341
,double8B

	full_text

double %398
{call8Bq
o
	full_textb
`
^%400 = tail call double @llvm.fmuladd.f64(double %399, double 0x4019333333333334, double %374)
,double8B

	full_text

double %399
,double8B

	full_text

double %374
Pload8BF
D
	full_text7
5
3%401 = load double, double* %236, align 8, !tbaa !8
.double*8B

	full_text

double* %236
vcall8Bl
j
	full_text]
[
Y%402 = tail call double @llvm.fmuladd.f64(double %332, double -2.000000e+00, double %401)
,double8B

	full_text

double %332
,double8B

	full_text

double %401
:fadd8B0
.
	full_text!

%403 = fadd double %402, %317
,double8B

	full_text

double %402
,double8B

	full_text

double %317
{call8Bq
o
	full_textb
`
^%404 = tail call double @llvm.fmuladd.f64(double %403, double 0x40AF020000000001, double %400)
,double8B

	full_text

double %403
,double8B

	full_text

double %400
Pload8BF
D
	full_text7
5
3%405 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
:fsub8B0
.
	full_text!

%406 = fsub double %343, %405
,double8B

	full_text

double %343
,double8B

	full_text

double %405
{call8Bq
o
	full_textb
`
^%407 = tail call double @llvm.fmuladd.f64(double %406, double 0x4019333333333334, double %379)
,double8B

	full_text

double %406
,double8B

	full_text

double %379
Pload8BF
D
	full_text7
5
3%408 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
vcall8Bl
j
	full_text]
[
Y%409 = tail call double @llvm.fmuladd.f64(double %334, double -2.000000e+00, double %408)
,double8B

	full_text

double %334
,double8B

	full_text

double %408
:fadd8B0
.
	full_text!

%410 = fadd double %409, %305
,double8B

	full_text

double %409
,double8B

	full_text

double %305
{call8Bq
o
	full_textb
`
^%411 = tail call double @llvm.fmuladd.f64(double %410, double 0x40AF020000000001, double %407)
,double8B

	full_text

double %410
,double8B

	full_text

double %407
Qload8BG
E
	full_text8
6
4%412 = load double, double* %286, align 16, !tbaa !8
.double*8B

	full_text

double* %286
:fsub8B0
.
	full_text!

%413 = fsub double %357, %412
,double8B

	full_text

double %357
,double8B

	full_text

double %412
{call8Bq
o
	full_textb
`
^%414 = tail call double @llvm.fmuladd.f64(double %413, double 0x4019333333333334, double %384)
,double8B

	full_text

double %413
,double8B

	full_text

double %384
Pload8BF
D
	full_text7
5
3%415 = load double, double* %270, align 8, !tbaa !8
.double*8B

	full_text

double* %270
vcall8Bl
j
	full_text]
[
Y%416 = tail call double @llvm.fmuladd.f64(double %336, double -2.000000e+00, double %415)
,double8B

	full_text

double %336
,double8B

	full_text

double %415
:fadd8B0
.
	full_text!

%417 = fadd double %416, %319
,double8B

	full_text

double %416
,double8B

	full_text

double %319
{call8Bq
o
	full_textb
`
^%418 = tail call double @llvm.fmuladd.f64(double %417, double 0x40AF020000000001, double %414)
,double8B

	full_text

double %417
,double8B

	full_text

double %414
kcall8Ba
_
	full_textR
P
N%419 = tail call double @_Z3maxdd(double 7.500000e-01, double 7.500000e-01) #5
ccall8BY
W
	full_textJ
H
F%420 = tail call double @_Z3maxdd(double %419, double 1.000000e+00) #5
,double8B

	full_text

double %419
Bfmul8B8
6
	full_text)
'
%%421 = fmul double %420, 2.500000e-01
,double8B

	full_text

double %420
Cfsub8B9
7
	full_text*
(
&%422 = fsub double -0.000000e+00, %421
,double8B

	full_text

double %421
Bfmul8B8
6
	full_text)
'
%%423 = fmul double %388, 4.000000e+00
,double8B

	full_text

double %388
Cfsub8B9
7
	full_text*
(
&%424 = fsub double -0.000000e+00, %423
,double8B

	full_text

double %423
ucall8Bk
i
	full_text\
Z
X%425 = tail call double @llvm.fmuladd.f64(double %386, double 5.000000e+00, double %424)
,double8B

	full_text

double %386
,double8B

	full_text

double %424
Qload8BG
E
	full_text8
6
4%426 = load double, double* %205, align 16, !tbaa !8
.double*8B

	full_text

double* %205
:fadd8B0
.
	full_text!

%427 = fadd double %426, %425
,double8B

	full_text

double %426
,double8B

	full_text

double %425
mcall8Bc
a
	full_textT
R
P%428 = tail call double @llvm.fmuladd.f64(double %422, double %427, double %390)
,double8B

	full_text

double %422
,double8B

	full_text

double %427
,double8B

	full_text

double %390
Pstore8BE
C
	full_text6
4
2store double %428, double* %360, align 8, !tbaa !8
,double8B

	full_text

double %428
.double*8B

	full_text

double* %360
Oload8BE
C
	full_text6
4
2%429 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
Oload8BE
C
	full_text6
4
2%430 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
Bfmul8B8
6
	full_text)
'
%%431 = fmul double %430, 4.000000e+00
,double8B

	full_text

double %430
Cfsub8B9
7
	full_text*
(
&%432 = fsub double -0.000000e+00, %431
,double8B

	full_text

double %431
ucall8Bk
i
	full_text\
Z
X%433 = tail call double @llvm.fmuladd.f64(double %429, double 5.000000e+00, double %432)
,double8B

	full_text

double %429
,double8B

	full_text

double %432
Pload8BF
D
	full_text7
5
3%434 = load double, double* %181, align 8, !tbaa !8
.double*8B

	full_text

double* %181
:fadd8B0
.
	full_text!

%435 = fadd double %434, %433
,double8B

	full_text

double %434
,double8B

	full_text

double %433
mcall8Bc
a
	full_textT
R
P%436 = tail call double @llvm.fmuladd.f64(double %422, double %435, double %397)
,double8B

	full_text

double %422
,double8B

	full_text

double %435
,double8B

	full_text

double %397
Pstore8BE
C
	full_text6
4
2store double %436, double* %365, align 8, !tbaa !8
,double8B

	full_text

double %436
.double*8B

	full_text

double* %365
Pload8BF
D
	full_text7
5
3%437 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
Oload8BE
C
	full_text6
4
2%438 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
Bfmul8B8
6
	full_text)
'
%%439 = fmul double %438, 4.000000e+00
,double8B

	full_text

double %438
Cfsub8B9
7
	full_text*
(
&%440 = fsub double -0.000000e+00, %439
,double8B

	full_text

double %439
ucall8Bk
i
	full_text\
Z
X%441 = tail call double @llvm.fmuladd.f64(double %437, double 5.000000e+00, double %440)
,double8B

	full_text

double %437
,double8B

	full_text

double %440
Qload8BG
E
	full_text8
6
4%442 = load double, double* %186, align 16, !tbaa !8
.double*8B

	full_text

double* %186
:fadd8B0
.
	full_text!

%443 = fadd double %442, %441
,double8B

	full_text

double %442
,double8B

	full_text

double %441
mcall8Bc
a
	full_textT
R
P%444 = tail call double @llvm.fmuladd.f64(double %422, double %443, double %404)
,double8B

	full_text

double %422
,double8B

	full_text

double %443
,double8B

	full_text

double %404
Pstore8BE
C
	full_text6
4
2store double %444, double* %370, align 8, !tbaa !8
,double8B

	full_text

double %444
.double*8B

	full_text

double* %370
Oload8BE
C
	full_text6
4
2%445 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Oload8BE
C
	full_text6
4
2%446 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
Bfmul8B8
6
	full_text)
'
%%447 = fmul double %446, 4.000000e+00
,double8B

	full_text

double %446
Cfsub8B9
7
	full_text*
(
&%448 = fsub double -0.000000e+00, %447
,double8B

	full_text

double %447
ucall8Bk
i
	full_text\
Z
X%449 = tail call double @llvm.fmuladd.f64(double %445, double 5.000000e+00, double %448)
,double8B

	full_text

double %445
,double8B

	full_text

double %448
Pload8BF
D
	full_text7
5
3%450 = load double, double* %191, align 8, !tbaa !8
.double*8B

	full_text

double* %191
:fadd8B0
.
	full_text!

%451 = fadd double %450, %449
,double8B

	full_text

double %450
,double8B

	full_text

double %449
mcall8Bc
a
	full_textT
R
P%452 = tail call double @llvm.fmuladd.f64(double %422, double %451, double %411)
,double8B

	full_text

double %422
,double8B

	full_text

double %451
,double8B

	full_text

double %411
Pstore8BE
C
	full_text6
4
2store double %452, double* %375, align 8, !tbaa !8
,double8B

	full_text

double %452
.double*8B

	full_text

double* %375
Pload8BF
D
	full_text7
5
3%453 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
Oload8BE
C
	full_text6
4
2%454 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
Bfmul8B8
6
	full_text)
'
%%455 = fmul double %454, 4.000000e+00
,double8B

	full_text

double %454
Cfsub8B9
7
	full_text*
(
&%456 = fsub double -0.000000e+00, %455
,double8B

	full_text

double %455
ucall8Bk
i
	full_text\
Z
X%457 = tail call double @llvm.fmuladd.f64(double %453, double 5.000000e+00, double %456)
,double8B

	full_text

double %453
,double8B

	full_text

double %456
Qload8BG
E
	full_text8
6
4%458 = load double, double* %196, align 16, !tbaa !8
.double*8B

	full_text

double* %196
:fadd8B0
.
	full_text!

%459 = fadd double %458, %457
,double8B

	full_text

double %458
,double8B

	full_text

double %457
mcall8Bc
a
	full_textT
R
P%460 = tail call double @llvm.fmuladd.f64(double %422, double %459, double %418)
,double8B

	full_text

double %422
,double8B

	full_text

double %459
,double8B

	full_text

double %418
Pstore8BE
C
	full_text6
4
2store double %460, double* %380, align 8, !tbaa !8
,double8B

	full_text

double %460
.double*8B

	full_text

double* %380
Ñgetelementptr8Bq
o
	full_textb
`
^%461 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %385, double* %461, align 16, !tbaa !8
,double8B

	full_text

double %385
.double*8B

	full_text

double* %461
Pstore8BE
C
	full_text6
4
2store double %386, double* %198, align 8, !tbaa !8
,double8B

	full_text

double %386
.double*8B

	full_text

double* %198
Pstore8BE
C
	full_text6
4
2store double %388, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %388
-double*8B

	full_text

double* %84
Ostore8BD
B
	full_text5
3
1store double %426, double* %82, align 8, !tbaa !8
,double8B

	full_text

double %426
-double*8B

	full_text

double* %82
Jload8B@
>
	full_text1
/
-%462 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Kstore8B@
>
	full_text1
/
-store i64 %462, i64* %211, align 16, !tbaa !8
&i648B

	full_text


i64 %462
(i64*8B

	full_text

	i64* %211
Kload8BA
?
	full_text2
0
.%463 = load i64, i64* %213, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %213
Jstore8B?
=
	full_text0
.
,store i64 %463, i64* %208, align 8, !tbaa !8
&i648B

	full_text


i64 %463
(i64*8B

	full_text

	i64* %208
Jload8B@
>
	full_text1
/
-%464 = load i64, i64* %216, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %216
Kstore8B@
>
	full_text1
/
-store i64 %464, i64* %218, align 16, !tbaa !8
&i648B

	full_text


i64 %464
(i64*8B

	full_text

	i64* %218
Pstore8BE
C
	full_text6
4
2store double %394, double* %222, align 8, !tbaa !8
,double8B

	full_text

double %394
.double*8B

	full_text

double* %222
Pstore8BE
C
	full_text6
4
2store double %429, double* %219, align 8, !tbaa !8
,double8B

	full_text

double %429
.double*8B

	full_text

double* %219
Ostore8BD
B
	full_text5
3
1store double %430, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %430
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %434, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %434
-double*8B

	full_text

double* %41
Jload8B@
>
	full_text1
/
-%465 = load i64, i64* %226, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %226
Jstore8B?
=
	full_text0
.
,store i64 %465, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %465
(i64*8B

	full_text

	i64* %229
Jload8B@
>
	full_text1
/
-%466 = load i64, i64* %230, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %230
Jstore8B?
=
	full_text0
.
,store i64 %466, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %466
(i64*8B

	full_text

	i64* %226
Jload8B@
>
	full_text1
/
-%467 = load i64, i64* %232, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %232
Jstore8B?
=
	full_text0
.
,store i64 %467, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %467
(i64*8B

	full_text

	i64* %235
Qstore8BF
D
	full_text7
5
3store double %401, double* %239, align 16, !tbaa !8
,double8B

	full_text

double %401
.double*8B

	full_text

double* %239
Pstore8BE
C
	full_text6
4
2store double %437, double* %236, align 8, !tbaa !8
,double8B

	full_text

double %437
.double*8B

	full_text

double* %236
Pstore8BE
C
	full_text6
4
2store double %438, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %438
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %442, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %442
-double*8B

	full_text

double* %46
Jload8B@
>
	full_text1
/
-%468 = load i64, i64* %243, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %243
Kstore8B@
>
	full_text1
/
-store i64 %468, i64* %246, align 16, !tbaa !8
&i648B

	full_text


i64 %468
(i64*8B

	full_text

	i64* %246
Kload8BA
?
	full_text2
0
.%469 = load i64, i64* %247, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %247
Jstore8B?
=
	full_text0
.
,store i64 %469, i64* %243, align 8, !tbaa !8
&i648B

	full_text


i64 %469
(i64*8B

	full_text

	i64* %243
Jload8B@
>
	full_text1
/
-%470 = load i64, i64* %249, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %249
Kstore8B@
>
	full_text1
/
-store i64 %470, i64* %252, align 16, !tbaa !8
&i648B

	full_text


i64 %470
(i64*8B

	full_text

	i64* %252
Jload8B@
>
	full_text1
/
-%471 = load i64, i64* %254, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %254
Jstore8B?
=
	full_text0
.
,store i64 %471, i64* %257, align 8, !tbaa !8
&i648B

	full_text


i64 %471
(i64*8B

	full_text

	i64* %257
Pstore8BE
C
	full_text6
4
2store double %445, double* %253, align 8, !tbaa !8
,double8B

	full_text

double %445
.double*8B

	full_text

double* %253
Ostore8BD
B
	full_text5
3
1store double %446, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %446
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %450, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %450
-double*8B

	full_text

double* %51
Jload8B@
>
	full_text1
/
-%472 = load i64, i64* %260, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %260
Jstore8B?
=
	full_text0
.
,store i64 %472, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %472
(i64*8B

	full_text

	i64* %263
Jload8B@
>
	full_text1
/
-%473 = load i64, i64* %264, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %264
Jstore8B?
=
	full_text0
.
,store i64 %473, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %473
(i64*8B

	full_text

	i64* %260
Jload8B@
>
	full_text1
/
-%474 = load i64, i64* %266, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %266
Jstore8B?
=
	full_text0
.
,store i64 %474, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %474
(i64*8B

	full_text

	i64* %269
Jload8B@
>
	full_text1
/
-%475 = load i64, i64* %271, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %271
Kstore8B@
>
	full_text1
/
-store i64 %475, i64* %274, align 16, !tbaa !8
&i648B

	full_text


i64 %475
(i64*8B

	full_text

	i64* %274
Pstore8BE
C
	full_text6
4
2store double %453, double* %270, align 8, !tbaa !8
,double8B

	full_text

double %453
.double*8B

	full_text

double* %270
Pstore8BE
C
	full_text6
4
2store double %454, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %454
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %458, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %458
-double*8B

	full_text

double* %56
Jload8B@
>
	full_text1
/
-%476 = load i64, i64* %278, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %278
Kstore8B@
>
	full_text1
/
-store i64 %476, i64* %281, align 16, !tbaa !8
&i648B

	full_text


i64 %476
(i64*8B

	full_text

	i64* %281
Kload8BA
?
	full_text2
0
.%477 = load i64, i64* %282, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %282
Jstore8B?
=
	full_text0
.
,store i64 %477, i64* %278, align 8, !tbaa !8
&i648B

	full_text


i64 %477
(i64*8B

	full_text

	i64* %278
Jload8B@
>
	full_text1
/
-%478 = load i64, i64* %284, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %284
Kstore8B@
>
	full_text1
/
-store i64 %478, i64* %287, align 16, !tbaa !8
&i648B

	full_text


i64 %478
(i64*8B

	full_text

	i64* %287
agetelementptr8BN
L
	full_text?
=
;%479 = getelementptr inbounds double, double* %0, i64 84500
Zbitcast8BM
K
	full_text>
<
:%480 = bitcast double* %479 to [65 x [65 x [5 x double]]]*
.double*8B

	full_text

double* %479
õgetelementptr8Bá
Ñ
	full_textw
u
s%481 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %480, i64 0, i64 %30, i64 %32
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %480
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%482 = bitcast [5 x double]* %481 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %481
Jload8B@
>
	full_text1
/
-%483 = load i64, i64* %482, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %482
Kstore8B@
>
	full_text1
/
-store i64 %483, i64* %177, align 16, !tbaa !8
&i648B

	full_text


i64 %483
(i64*8B

	full_text

	i64* %177
¢getelementptr8Bé
ã
	full_text~
|
z%484 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %480, i64 0, i64 %30, i64 %32, i64 1
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %480
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%485 = bitcast double* %484 to i64*
.double*8B

	full_text

double* %484
Jload8B@
>
	full_text1
/
-%486 = load i64, i64* %485, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %485
Jstore8B?
=
	full_text0
.
,store i64 %486, i64* %182, align 8, !tbaa !8
&i648B

	full_text


i64 %486
(i64*8B

	full_text

	i64* %182
¢getelementptr8Bé
ã
	full_text~
|
z%487 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %480, i64 0, i64 %30, i64 %32, i64 2
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %480
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%488 = bitcast double* %487 to i64*
.double*8B

	full_text

double* %487
Jload8B@
>
	full_text1
/
-%489 = load i64, i64* %488, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %488
Kstore8B@
>
	full_text1
/
-store i64 %489, i64* %187, align 16, !tbaa !8
&i648B

	full_text


i64 %489
(i64*8B

	full_text

	i64* %187
¢getelementptr8Bé
ã
	full_text~
|
z%490 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %480, i64 0, i64 %30, i64 %32, i64 3
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %480
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%491 = bitcast double* %490 to i64*
.double*8B

	full_text

double* %490
Jload8B@
>
	full_text1
/
-%492 = load i64, i64* %491, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %491
Jstore8B?
=
	full_text0
.
,store i64 %492, i64* %192, align 8, !tbaa !8
&i648B

	full_text


i64 %492
(i64*8B

	full_text

	i64* %192
¢getelementptr8Bé
ã
	full_text~
|
z%493 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %480, i64 0, i64 %30, i64 %32, i64 4
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %480
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%494 = bitcast double* %493 to i64*
.double*8B

	full_text

double* %493
Jload8B@
>
	full_text1
/
-%495 = load i64, i64* %494, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %494
Kstore8B@
>
	full_text1
/
-store i64 %495, i64* %197, align 16, !tbaa !8
&i648B

	full_text


i64 %495
(i64*8B

	full_text

	i64* %197
rgetelementptr8B_
]
	full_textP
N
L%496 = getelementptr inbounds [5 x double], [5 x double]* %113, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %113
Qstore8BF
D
	full_text7
5
3store double %450, double* %496, align 16, !tbaa !8
,double8B

	full_text

double %450
.double*8B

	full_text

double* %496
agetelementptr8BN
L
	full_text?
=
;%497 = getelementptr inbounds double, double* %3, i64 12675
Tbitcast8BG
E
	full_text8
6
4%498 = bitcast double* %497 to [65 x [65 x double]]*
.double*8B

	full_text

double* %497
çgetelementptr8Bz
x
	full_textk
i
g%499 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %498, i64 0, i64 %30, i64 %32
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %498
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%500 = load double, double* %499, align 8, !tbaa !8
.double*8B

	full_text

double* %499
:fmul8B0
.
	full_text!

%501 = fmul double %500, %450
,double8B

	full_text

double %500
,double8B

	full_text

double %450
agetelementptr8BN
L
	full_text?
=
;%502 = getelementptr inbounds double, double* %2, i64 12675
Tbitcast8BG
E
	full_text8
6
4%503 = bitcast double* %502 to [65 x [65 x double]]*
.double*8B

	full_text

double* %502
çgetelementptr8Bz
x
	full_textk
i
g%504 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %503, i64 0, i64 %30, i64 %32
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %503
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%505 = load double, double* %504, align 8, !tbaa !8
.double*8B

	full_text

double* %504
:fmul8B0
.
	full_text!

%506 = fmul double %501, %434
,double8B

	full_text

double %501
,double8B

	full_text

double %434
Pstore8BE
C
	full_text6
4
2store double %506, double* %127, align 8, !tbaa !8
,double8B

	full_text

double %506
.double*8B

	full_text

double* %127
:fmul8B0
.
	full_text!

%507 = fmul double %501, %442
,double8B

	full_text

double %501
,double8B

	full_text

double %442
Qstore8BF
D
	full_text7
5
3store double %507, double* %130, align 16, !tbaa !8
,double8B

	full_text

double %507
.double*8B

	full_text

double* %130
:fsub8B0
.
	full_text!

%508 = fsub double %458, %505
,double8B

	full_text

double %458
,double8B

	full_text

double %505
Bfmul8B8
6
	full_text)
'
%%509 = fmul double %508, 4.000000e-01
,double8B

	full_text

double %508
mcall8Bc
a
	full_textT
R
P%510 = tail call double @llvm.fmuladd.f64(double %450, double %501, double %509)
,double8B

	full_text

double %450
,double8B

	full_text

double %501
,double8B

	full_text

double %509
Pstore8BE
C
	full_text6
4
2store double %510, double* %135, align 8, !tbaa !8
,double8B

	full_text

double %510
.double*8B

	full_text

double* %135
Bfmul8B8
6
	full_text)
'
%%511 = fmul double %505, 4.000000e-01
,double8B

	full_text

double %505
Cfsub8B9
7
	full_text*
(
&%512 = fsub double -0.000000e+00, %511
,double8B

	full_text

double %511
ucall8Bk
i
	full_text\
Z
X%513 = tail call double @llvm.fmuladd.f64(double %458, double 1.400000e+00, double %512)
,double8B

	full_text

double %458
,double8B

	full_text

double %512
:fmul8B0
.
	full_text!

%514 = fmul double %501, %513
,double8B

	full_text

double %501
,double8B

	full_text

double %513
Qstore8BF
D
	full_text7
5
3store double %514, double* %140, align 16, !tbaa !8
,double8B

	full_text

double %514
.double*8B

	full_text

double* %140
:fmul8B0
.
	full_text!

%515 = fmul double %500, %434
,double8B

	full_text

double %500
,double8B

	full_text

double %434
:fmul8B0
.
	full_text!

%516 = fmul double %500, %442
,double8B

	full_text

double %500
,double8B

	full_text

double %442
:fmul8B0
.
	full_text!

%517 = fmul double %500, %458
,double8B

	full_text

double %500
,double8B

	full_text

double %458
Pload8BF
D
	full_text7
5
3%518 = load double, double* %308, align 8, !tbaa !8
.double*8B

	full_text

double* %308
Oload8BE
C
	full_text6
4
2%519 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
:fmul8B0
.
	full_text!

%520 = fmul double %518, %519
,double8B

	full_text

double %518
,double8B

	full_text

double %519
:fmul8B0
.
	full_text!

%521 = fmul double %518, %438
,double8B

	full_text

double %518
,double8B

	full_text

double %438
:fmul8B0
.
	full_text!

%522 = fmul double %518, %446
,double8B

	full_text

double %518
,double8B

	full_text

double %446
:fmul8B0
.
	full_text!

%523 = fmul double %518, %454
,double8B

	full_text

double %518
,double8B

	full_text

double %454
:fsub8B0
.
	full_text!

%524 = fsub double %515, %520
,double8B

	full_text

double %515
,double8B

	full_text

double %520
Bfmul8B8
6
	full_text)
'
%%525 = fmul double %524, 6.300000e+01
,double8B

	full_text

double %524
Pstore8BE
C
	full_text6
4
2store double %525, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %525
.double*8B

	full_text

double* %149
:fsub8B0
.
	full_text!

%526 = fsub double %516, %521
,double8B

	full_text

double %516
,double8B

	full_text

double %521
Bfmul8B8
6
	full_text)
'
%%527 = fmul double %526, 6.300000e+01
,double8B

	full_text

double %526
Pstore8BE
C
	full_text6
4
2store double %527, double* %152, align 8, !tbaa !8
,double8B

	full_text

double %527
.double*8B

	full_text

double* %152
:fsub8B0
.
	full_text!

%528 = fsub double %501, %522
,double8B

	full_text

double %501
,double8B

	full_text

double %522
Bfmul8B8
6
	full_text)
'
%%529 = fmul double %528, 8.400000e+01
,double8B

	full_text

double %528
Pstore8BE
C
	full_text6
4
2store double %529, double* %155, align 8, !tbaa !8
,double8B

	full_text

double %529
.double*8B

	full_text

double* %155
:fmul8B0
.
	full_text!

%530 = fmul double %516, %516
,double8B

	full_text

double %516
,double8B

	full_text

double %516
mcall8Bc
a
	full_textT
R
P%531 = tail call double @llvm.fmuladd.f64(double %515, double %515, double %530)
,double8B

	full_text

double %515
,double8B

	full_text

double %515
,double8B

	full_text

double %530
mcall8Bc
a
	full_textT
R
P%532 = tail call double @llvm.fmuladd.f64(double %501, double %501, double %531)
,double8B

	full_text

double %501
,double8B

	full_text

double %501
,double8B

	full_text

double %531
:fmul8B0
.
	full_text!

%533 = fmul double %521, %521
,double8B

	full_text

double %521
,double8B

	full_text

double %521
mcall8Bc
a
	full_textT
R
P%534 = tail call double @llvm.fmuladd.f64(double %520, double %520, double %533)
,double8B

	full_text

double %520
,double8B

	full_text

double %520
,double8B

	full_text

double %533
mcall8Bc
a
	full_textT
R
P%535 = tail call double @llvm.fmuladd.f64(double %522, double %522, double %534)
,double8B

	full_text

double %522
,double8B

	full_text

double %522
,double8B

	full_text

double %534
:fsub8B0
.
	full_text!

%536 = fsub double %532, %535
,double8B

	full_text

double %532
,double8B

	full_text

double %535
:fmul8B0
.
	full_text!

%537 = fmul double %522, %522
,double8B

	full_text

double %522
,double8B

	full_text

double %522
Cfsub8B9
7
	full_text*
(
&%538 = fsub double -0.000000e+00, %537
,double8B

	full_text

double %537
mcall8Bc
a
	full_textT
R
P%539 = tail call double @llvm.fmuladd.f64(double %501, double %501, double %538)
,double8B

	full_text

double %501
,double8B

	full_text

double %501
,double8B

	full_text

double %538
Bfmul8B8
6
	full_text)
'
%%540 = fmul double %539, 1.050000e+01
,double8B

	full_text

double %539
{call8Bq
o
	full_textb
`
^%541 = tail call double @llvm.fmuladd.f64(double %536, double 0xC03E3D70A3D70A3B, double %540)
,double8B

	full_text

double %536
,double8B

	full_text

double %540
:fsub8B0
.
	full_text!

%542 = fsub double %517, %523
,double8B

	full_text

double %517
,double8B

	full_text

double %523
{call8Bq
o
	full_textb
`
^%543 = tail call double @llvm.fmuladd.f64(double %542, double 0x405EDEB851EB851E, double %541)
,double8B

	full_text

double %542
,double8B

	full_text

double %541
Pstore8BE
C
	full_text6
4
2store double %543, double* %170, align 8, !tbaa !8
,double8B

	full_text

double %543
.double*8B

	full_text

double* %170
agetelementptr8BN
L
	full_text?
=
;%544 = getelementptr inbounds double, double* %1, i64 42250
Zbitcast8BM
K
	full_text>
<
:%545 = bitcast double* %544 to [65 x [65 x [5 x double]]]*
.double*8B

	full_text

double* %544
¢getelementptr8Bé
ã
	full_text~
|
z%546 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %545, i64 0, i64 %30, i64 %32, i64 0
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %545
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%547 = load double, double* %546, align 8, !tbaa !8
.double*8B

	full_text

double* %546
Qload8BG
E
	full_text8
6
4%548 = load double, double* %210, align 16, !tbaa !8
.double*8B

	full_text

double* %210
:fsub8B0
.
	full_text!

%549 = fsub double %450, %548
,double8B

	full_text

double %450
,double8B

	full_text

double %548
vcall8Bl
j
	full_text]
[
Y%550 = tail call double @llvm.fmuladd.f64(double %549, double -3.150000e+01, double %547)
,double8B

	full_text

double %549
,double8B

	full_text

double %547
¢getelementptr8Bé
ã
	full_text~
|
z%551 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %545, i64 0, i64 %30, i64 %32, i64 1
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %545
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%552 = load double, double* %551, align 8, !tbaa !8
.double*8B

	full_text

double* %551
Pload8BF
D
	full_text7
5
3%553 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
:fsub8B0
.
	full_text!

%554 = fsub double %506, %553
,double8B

	full_text

double %506
,double8B

	full_text

double %553
vcall8Bl
j
	full_text]
[
Y%555 = tail call double @llvm.fmuladd.f64(double %554, double -3.150000e+01, double %552)
,double8B

	full_text

double %554
,double8B

	full_text

double %552
¢getelementptr8Bé
ã
	full_text~
|
z%556 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %545, i64 0, i64 %30, i64 %32, i64 2
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %545
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%557 = load double, double* %556, align 8, !tbaa !8
.double*8B

	full_text

double* %556
Qload8BG
E
	full_text8
6
4%558 = load double, double* %245, align 16, !tbaa !8
.double*8B

	full_text

double* %245
:fsub8B0
.
	full_text!

%559 = fsub double %507, %558
,double8B

	full_text

double %507
,double8B

	full_text

double %558
vcall8Bl
j
	full_text]
[
Y%560 = tail call double @llvm.fmuladd.f64(double %559, double -3.150000e+01, double %557)
,double8B

	full_text

double %559
,double8B

	full_text

double %557
¢getelementptr8Bé
ã
	full_text~
|
z%561 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %545, i64 0, i64 %30, i64 %32, i64 3
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %545
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%562 = load double, double* %561, align 8, !tbaa !8
.double*8B

	full_text

double* %561
Pload8BF
D
	full_text7
5
3%563 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
:fsub8B0
.
	full_text!

%564 = fsub double %510, %563
,double8B

	full_text

double %510
,double8B

	full_text

double %563
vcall8Bl
j
	full_text]
[
Y%565 = tail call double @llvm.fmuladd.f64(double %564, double -3.150000e+01, double %562)
,double8B

	full_text

double %564
,double8B

	full_text

double %562
¢getelementptr8Bé
ã
	full_text~
|
z%566 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %545, i64 0, i64 %30, i64 %32, i64 4
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %545
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%567 = load double, double* %566, align 8, !tbaa !8
.double*8B

	full_text

double* %566
Qload8BG
E
	full_text8
6
4%568 = load double, double* %280, align 16, !tbaa !8
.double*8B

	full_text

double* %280
:fsub8B0
.
	full_text!

%569 = fsub double %514, %568
,double8B

	full_text

double %514
,double8B

	full_text

double %568
vcall8Bl
j
	full_text]
[
Y%570 = tail call double @llvm.fmuladd.f64(double %569, double -3.150000e+01, double %567)
,double8B

	full_text

double %569
,double8B

	full_text

double %567
Pload8BF
D
	full_text7
5
3%571 = load double, double* %198, align 8, !tbaa !8
.double*8B

	full_text

double* %198
Pload8BF
D
	full_text7
5
3%572 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%573 = tail call double @llvm.fmuladd.f64(double %572, double -2.000000e+00, double %571)
,double8B

	full_text

double %572
,double8B

	full_text

double %571
Oload8BE
C
	full_text6
4
2%574 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%575 = fadd double %573, %574
,double8B

	full_text

double %573
,double8B

	full_text

double %574
{call8Bq
o
	full_textb
`
^%576 = tail call double @llvm.fmuladd.f64(double %575, double 0x40AF020000000001, double %550)
,double8B

	full_text

double %575
,double8B

	full_text

double %550
Pload8BF
D
	full_text7
5
3%577 = load double, double* %234, align 8, !tbaa !8
.double*8B

	full_text

double* %234
:fsub8B0
.
	full_text!

%578 = fsub double %525, %577
,double8B

	full_text

double %525
,double8B

	full_text

double %577
{call8Bq
o
	full_textb
`
^%579 = tail call double @llvm.fmuladd.f64(double %578, double 0x4019333333333334, double %555)
,double8B

	full_text

double %578
,double8B

	full_text

double %555
Pload8BF
D
	full_text7
5
3%580 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
vcall8Bl
j
	full_text]
[
Y%581 = tail call double @llvm.fmuladd.f64(double %519, double -2.000000e+00, double %580)
,double8B

	full_text

double %519
,double8B

	full_text

double %580
:fadd8B0
.
	full_text!

%582 = fadd double %434, %581
,double8B

	full_text

double %434
,double8B

	full_text

double %581
{call8Bq
o
	full_textb
`
^%583 = tail call double @llvm.fmuladd.f64(double %582, double 0x40AF020000000001, double %579)
,double8B

	full_text

double %582
,double8B

	full_text

double %579
Qload8BG
E
	full_text8
6
4%584 = load double, double* %251, align 16, !tbaa !8
.double*8B

	full_text

double* %251
:fsub8B0
.
	full_text!

%585 = fsub double %527, %584
,double8B

	full_text

double %527
,double8B

	full_text

double %584
{call8Bq
o
	full_textb
`
^%586 = tail call double @llvm.fmuladd.f64(double %585, double 0x4019333333333334, double %560)
,double8B

	full_text

double %585
,double8B

	full_text

double %560
Pload8BF
D
	full_text7
5
3%587 = load double, double* %236, align 8, !tbaa !8
.double*8B

	full_text

double* %236
vcall8Bl
j
	full_text]
[
Y%588 = tail call double @llvm.fmuladd.f64(double %438, double -2.000000e+00, double %587)
,double8B

	full_text

double %438
,double8B

	full_text

double %587
:fadd8B0
.
	full_text!

%589 = fadd double %442, %588
,double8B

	full_text

double %442
,double8B

	full_text

double %588
{call8Bq
o
	full_textb
`
^%590 = tail call double @llvm.fmuladd.f64(double %589, double 0x40AF020000000001, double %586)
,double8B

	full_text

double %589
,double8B

	full_text

double %586
Pload8BF
D
	full_text7
5
3%591 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
:fsub8B0
.
	full_text!

%592 = fsub double %529, %591
,double8B

	full_text

double %529
,double8B

	full_text

double %591
{call8Bq
o
	full_textb
`
^%593 = tail call double @llvm.fmuladd.f64(double %592, double 0x4019333333333334, double %565)
,double8B

	full_text

double %592
,double8B

	full_text

double %565
Pload8BF
D
	full_text7
5
3%594 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
vcall8Bl
j
	full_text]
[
Y%595 = tail call double @llvm.fmuladd.f64(double %446, double -2.000000e+00, double %594)
,double8B

	full_text

double %446
,double8B

	full_text

double %594
:fadd8B0
.
	full_text!

%596 = fadd double %450, %595
,double8B

	full_text

double %450
,double8B

	full_text

double %595
{call8Bq
o
	full_textb
`
^%597 = tail call double @llvm.fmuladd.f64(double %596, double 0x40AF020000000001, double %593)
,double8B

	full_text

double %596
,double8B

	full_text

double %593
Qload8BG
E
	full_text8
6
4%598 = load double, double* %286, align 16, !tbaa !8
.double*8B

	full_text

double* %286
:fsub8B0
.
	full_text!

%599 = fsub double %543, %598
,double8B

	full_text

double %543
,double8B

	full_text

double %598
{call8Bq
o
	full_textb
`
^%600 = tail call double @llvm.fmuladd.f64(double %599, double 0x4019333333333334, double %570)
,double8B

	full_text

double %599
,double8B

	full_text

double %570
Pload8BF
D
	full_text7
5
3%601 = load double, double* %270, align 8, !tbaa !8
.double*8B

	full_text

double* %270
vcall8Bl
j
	full_text]
[
Y%602 = tail call double @llvm.fmuladd.f64(double %454, double -2.000000e+00, double %601)
,double8B

	full_text

double %454
,double8B

	full_text

double %601
:fadd8B0
.
	full_text!

%603 = fadd double %458, %602
,double8B

	full_text

double %458
,double8B

	full_text

double %602
{call8Bq
o
	full_textb
`
^%604 = tail call double @llvm.fmuladd.f64(double %603, double 0x40AF020000000001, double %600)
,double8B

	full_text

double %603
,double8B

	full_text

double %600
Bfmul8B8
6
	full_text)
'
%%605 = fmul double %572, 6.000000e+00
,double8B

	full_text

double %572
vcall8Bl
j
	full_text]
[
Y%606 = tail call double @llvm.fmuladd.f64(double %571, double -4.000000e+00, double %605)
,double8B

	full_text

double %571
,double8B

	full_text

double %605
vcall8Bl
j
	full_text]
[
Y%607 = tail call double @llvm.fmuladd.f64(double %574, double -4.000000e+00, double %606)
,double8B

	full_text

double %574
,double8B

	full_text

double %606
Qload8BG
E
	full_text8
6
4%608 = load double, double* %205, align 16, !tbaa !8
.double*8B

	full_text

double* %205
:fadd8B0
.
	full_text!

%609 = fadd double %608, %607
,double8B

	full_text

double %608
,double8B

	full_text

double %607
mcall8Bc
a
	full_textT
R
P%610 = tail call double @llvm.fmuladd.f64(double %422, double %609, double %576)
,double8B

	full_text

double %422
,double8B

	full_text

double %609
,double8B

	full_text

double %576
Pstore8BE
C
	full_text6
4
2store double %610, double* %546, align 8, !tbaa !8
,double8B

	full_text

double %610
.double*8B

	full_text

double* %546
Oload8BE
C
	full_text6
4
2%611 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
Bfmul8B8
6
	full_text)
'
%%612 = fmul double %611, 6.000000e+00
,double8B

	full_text

double %611
vcall8Bl
j
	full_text]
[
Y%613 = tail call double @llvm.fmuladd.f64(double %580, double -4.000000e+00, double %612)
,double8B

	full_text

double %580
,double8B

	full_text

double %612
Oload8BE
C
	full_text6
4
2%614 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
vcall8Bl
j
	full_text]
[
Y%615 = tail call double @llvm.fmuladd.f64(double %614, double -4.000000e+00, double %613)
,double8B

	full_text

double %614
,double8B

	full_text

double %613
Pload8BF
D
	full_text7
5
3%616 = load double, double* %181, align 8, !tbaa !8
.double*8B

	full_text

double* %181
:fadd8B0
.
	full_text!

%617 = fadd double %616, %615
,double8B

	full_text

double %616
,double8B

	full_text

double %615
mcall8Bc
a
	full_textT
R
P%618 = tail call double @llvm.fmuladd.f64(double %422, double %617, double %583)
,double8B

	full_text

double %422
,double8B

	full_text

double %617
,double8B

	full_text

double %583
Pstore8BE
C
	full_text6
4
2store double %618, double* %551, align 8, !tbaa !8
,double8B

	full_text

double %618
.double*8B

	full_text

double* %551
Pload8BF
D
	full_text7
5
3%619 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
Bfmul8B8
6
	full_text)
'
%%620 = fmul double %619, 6.000000e+00
,double8B

	full_text

double %619
vcall8Bl
j
	full_text]
[
Y%621 = tail call double @llvm.fmuladd.f64(double %587, double -4.000000e+00, double %620)
,double8B

	full_text

double %587
,double8B

	full_text

double %620
Oload8BE
C
	full_text6
4
2%622 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
vcall8Bl
j
	full_text]
[
Y%623 = tail call double @llvm.fmuladd.f64(double %622, double -4.000000e+00, double %621)
,double8B

	full_text

double %622
,double8B

	full_text

double %621
Qload8BG
E
	full_text8
6
4%624 = load double, double* %186, align 16, !tbaa !8
.double*8B

	full_text

double* %186
:fadd8B0
.
	full_text!

%625 = fadd double %624, %623
,double8B

	full_text

double %624
,double8B

	full_text

double %623
mcall8Bc
a
	full_textT
R
P%626 = tail call double @llvm.fmuladd.f64(double %422, double %625, double %590)
,double8B

	full_text

double %422
,double8B

	full_text

double %625
,double8B

	full_text

double %590
Pstore8BE
C
	full_text6
4
2store double %626, double* %556, align 8, !tbaa !8
,double8B

	full_text

double %626
.double*8B

	full_text

double* %556
Oload8BE
C
	full_text6
4
2%627 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Bfmul8B8
6
	full_text)
'
%%628 = fmul double %627, 6.000000e+00
,double8B

	full_text

double %627
vcall8Bl
j
	full_text]
[
Y%629 = tail call double @llvm.fmuladd.f64(double %594, double -4.000000e+00, double %628)
,double8B

	full_text

double %594
,double8B

	full_text

double %628
Oload8BE
C
	full_text6
4
2%630 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
vcall8Bl
j
	full_text]
[
Y%631 = tail call double @llvm.fmuladd.f64(double %630, double -4.000000e+00, double %629)
,double8B

	full_text

double %630
,double8B

	full_text

double %629
Pload8BF
D
	full_text7
5
3%632 = load double, double* %191, align 8, !tbaa !8
.double*8B

	full_text

double* %191
:fadd8B0
.
	full_text!

%633 = fadd double %632, %631
,double8B

	full_text

double %632
,double8B

	full_text

double %631
mcall8Bc
a
	full_textT
R
P%634 = tail call double @llvm.fmuladd.f64(double %422, double %633, double %597)
,double8B

	full_text

double %422
,double8B

	full_text

double %633
,double8B

	full_text

double %597
Pstore8BE
C
	full_text6
4
2store double %634, double* %561, align 8, !tbaa !8
,double8B

	full_text

double %634
.double*8B

	full_text

double* %561
Pload8BF
D
	full_text7
5
3%635 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
Bfmul8B8
6
	full_text)
'
%%636 = fmul double %635, 6.000000e+00
,double8B

	full_text

double %635
vcall8Bl
j
	full_text]
[
Y%637 = tail call double @llvm.fmuladd.f64(double %601, double -4.000000e+00, double %636)
,double8B

	full_text

double %601
,double8B

	full_text

double %636
Oload8BE
C
	full_text6
4
2%638 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
vcall8Bl
j
	full_text]
[
Y%639 = tail call double @llvm.fmuladd.f64(double %638, double -4.000000e+00, double %637)
,double8B

	full_text

double %638
,double8B

	full_text

double %637
Qload8BG
E
	full_text8
6
4%640 = load double, double* %196, align 16, !tbaa !8
.double*8B

	full_text

double* %196
:fadd8B0
.
	full_text!

%641 = fadd double %640, %639
,double8B

	full_text

double %640
,double8B

	full_text

double %639
mcall8Bc
a
	full_textT
R
P%642 = tail call double @llvm.fmuladd.f64(double %422, double %641, double %604)
,double8B

	full_text

double %422
,double8B

	full_text

double %641
,double8B

	full_text

double %604
Pstore8BE
C
	full_text6
4
2store double %642, double* %566, align 8, !tbaa !8
,double8B

	full_text

double %642
.double*8B

	full_text

double* %566
5add8B,
*
	full_text

%643 = add nsw i32 %6, -3
6icmp8B,
*
	full_text

%644 = icmp sgt i32 %6, 6
Abitcast8B4
2
	full_text%
#
!%645 = bitcast double %608 to i64
,double8B

	full_text

double %608
Abitcast8B4
2
	full_text%
#
!%646 = bitcast double %580 to i64
,double8B

	full_text

double %580
Abitcast8B4
2
	full_text%
#
!%647 = bitcast double %611 to i64
,double8B

	full_text

double %611
Abitcast8B4
2
	full_text%
#
!%648 = bitcast double %587 to i64
,double8B

	full_text

double %587
Abitcast8B4
2
	full_text%
#
!%649 = bitcast double %619 to i64
,double8B

	full_text

double %619
Abitcast8B4
2
	full_text%
#
!%650 = bitcast double %594 to i64
,double8B

	full_text

double %594
Abitcast8B4
2
	full_text%
#
!%651 = bitcast double %627 to i64
,double8B

	full_text

double %627
Abitcast8B4
2
	full_text%
#
!%652 = bitcast double %632 to i64
,double8B

	full_text

double %632
Abitcast8B4
2
	full_text%
#
!%653 = bitcast double %601 to i64
,double8B

	full_text

double %601
Abitcast8B4
2
	full_text%
#
!%654 = bitcast double %635 to i64
,double8B

	full_text

double %635
=br8B5
3
	full_text&
$
"br i1 %644, label %655, label %867
$i18B

	full_text
	
i1 %644
Ñgetelementptr8Bq
o
	full_textb
`
^%656 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Jload8B@
>
	full_text1
/
-%657 = load i64, i64* %216, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %216
rgetelementptr8B_
]
	full_textP
N
L%658 = getelementptr inbounds [5 x double], [5 x double]* %113, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %113
8zext8B.
,
	full_text

%659 = zext i32 %643 to i64
&i328B

	full_text


i32 %643
Jload8B@
>
	full_text1
/
-%660 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Jload8B@
>
	full_text1
/
-%661 = load i64, i64* %226, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %226
Jload8B@
>
	full_text1
/
-%662 = load i64, i64* %243, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %243
Jload8B@
>
	full_text1
/
-%663 = load i64, i64* %260, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %260
Jload8B@
>
	full_text1
/
-%664 = load i64, i64* %278, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %278
(br8B 

	full_text

br label %665
Iphi8B@
>
	full_text1
/
-%666 = phi i64 [ %664, %655 ], [ %701, %665 ]
&i648B

	full_text


i64 %664
&i648B

	full_text


i64 %701
Iphi8B@
>
	full_text1
/
-%667 = phi i64 [ %663, %655 ], [ %699, %665 ]
&i648B

	full_text


i64 %663
&i648B

	full_text


i64 %699
Iphi8B@
>
	full_text1
/
-%668 = phi i64 [ %662, %655 ], [ %697, %665 ]
&i648B

	full_text


i64 %662
&i648B

	full_text


i64 %697
Iphi8B@
>
	full_text1
/
-%669 = phi i64 [ %661, %655 ], [ %695, %665 ]
&i648B

	full_text


i64 %661
&i648B

	full_text


i64 %695
Iphi8B@
>
	full_text1
/
-%670 = phi i64 [ %660, %655 ], [ %694, %665 ]
&i648B

	full_text


i64 %660
&i648B

	full_text


i64 %694
Lphi8BC
A
	full_text4
2
0%671 = phi double [ %640, %655 ], [ %851, %665 ]
,double8B

	full_text

double %640
,double8B

	full_text

double %851
Lphi8BC
A
	full_text4
2
0%672 = phi double [ %638, %655 ], [ %671, %665 ]
,double8B

	full_text

double %638
,double8B

	full_text

double %671
Iphi8B@
>
	full_text1
/
-%673 = phi i64 [ %654, %655 ], [ %864, %665 ]
&i648B

	full_text


i64 %654
&i648B

	full_text


i64 %864
Iphi8B@
>
	full_text1
/
-%674 = phi i64 [ %653, %655 ], [ %863, %665 ]
&i648B

	full_text


i64 %653
&i648B

	full_text


i64 %863
Lphi8BC
A
	full_text4
2
0%675 = phi double [ %632, %655 ], [ %844, %665 ]
,double8B

	full_text

double %632
,double8B

	full_text

double %844
Iphi8B@
>
	full_text1
/
-%676 = phi i64 [ %652, %655 ], [ %862, %665 ]
&i648B

	full_text


i64 %652
&i648B

	full_text


i64 %862
Lphi8BC
A
	full_text4
2
0%677 = phi double [ %630, %655 ], [ %842, %665 ]
,double8B

	full_text

double %630
,double8B

	full_text

double %842
Iphi8B@
>
	full_text1
/
-%678 = phi i64 [ %651, %655 ], [ %861, %665 ]
&i648B

	full_text


i64 %651
&i648B

	full_text


i64 %861
Iphi8B@
>
	full_text1
/
-%679 = phi i64 [ %650, %655 ], [ %860, %665 ]
&i648B

	full_text


i64 %650
&i648B

	full_text


i64 %860
Lphi8BC
A
	full_text4
2
0%680 = phi double [ %624, %655 ], [ %836, %665 ]
,double8B

	full_text

double %624
,double8B

	full_text

double %836
Lphi8BC
A
	full_text4
2
0%681 = phi double [ %622, %655 ], [ %680, %665 ]
,double8B

	full_text

double %622
,double8B

	full_text

double %680
Iphi8B@
>
	full_text1
/
-%682 = phi i64 [ %649, %655 ], [ %859, %665 ]
&i648B

	full_text


i64 %649
&i648B

	full_text


i64 %859
Iphi8B@
>
	full_text1
/
-%683 = phi i64 [ %648, %655 ], [ %858, %665 ]
&i648B

	full_text


i64 %648
&i648B

	full_text


i64 %858
Lphi8BC
A
	full_text4
2
0%684 = phi double [ %616, %655 ], [ %829, %665 ]
,double8B

	full_text

double %616
,double8B

	full_text

double %829
Lphi8BC
A
	full_text4
2
0%685 = phi double [ %614, %655 ], [ %684, %665 ]
,double8B

	full_text

double %614
,double8B

	full_text

double %684
Iphi8B@
>
	full_text1
/
-%686 = phi i64 [ %647, %655 ], [ %857, %665 ]
&i648B

	full_text


i64 %647
&i648B

	full_text


i64 %857
Iphi8B@
>
	full_text1
/
-%687 = phi i64 [ %646, %655 ], [ %856, %665 ]
&i648B

	full_text


i64 %646
&i648B

	full_text


i64 %856
Iphi8B@
>
	full_text1
/
-%688 = phi i64 [ %645, %655 ], [ %855, %665 ]
&i648B

	full_text


i64 %645
&i648B

	full_text


i64 %855
Lphi8BC
A
	full_text4
2
0%689 = phi double [ %574, %655 ], [ %788, %665 ]
,double8B

	full_text

double %574
,double8B

	full_text

double %788
Lphi8BC
A
	full_text4
2
0%690 = phi double [ %572, %655 ], [ %689, %665 ]
,double8B

	full_text

double %572
,double8B

	full_text

double %689
Lphi8BC
A
	full_text4
2
0%691 = phi double [ %571, %655 ], [ %690, %665 ]
,double8B

	full_text

double %571
,double8B

	full_text

double %690
Fphi8B=
;
	full_text.
,
*%692 = phi i64 [ 3, %655 ], [ %693, %665 ]
&i648B

	full_text


i64 %693
:add8B1
/
	full_text"
 
%693 = add nuw nsw i64 %692, 1
&i648B

	full_text


i64 %692
Istore8B>
<
	full_text/
-
+store i64 %688, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %688
'i64*8B

	full_text


i64* %83
Kstore8B@
>
	full_text1
/
-store i64 %670, i64* %211, align 16, !tbaa !8
&i648B

	full_text


i64 %670
(i64*8B

	full_text

	i64* %211
Kload8BA
?
	full_text2
0
.%694 = load i64, i64* %213, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %213
Jstore8B?
=
	full_text0
.
,store i64 %687, i64* %223, align 8, !tbaa !8
&i648B

	full_text


i64 %687
(i64*8B

	full_text

	i64* %223
Jstore8B?
=
	full_text0
.
,store i64 %686, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %686
(i64*8B

	full_text

	i64* %220
Jstore8B?
=
	full_text0
.
,store i64 %669, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %669
(i64*8B

	full_text

	i64* %229
Jload8B@
>
	full_text1
/
-%695 = load i64, i64* %230, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %230
Jload8B@
>
	full_text1
/
-%696 = load i64, i64* %232, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %232
Jstore8B?
=
	full_text0
.
,store i64 %696, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %696
(i64*8B

	full_text

	i64* %235
Kstore8B@
>
	full_text1
/
-store i64 %683, i64* %240, align 16, !tbaa !8
&i648B

	full_text


i64 %683
(i64*8B

	full_text

	i64* %240
Jstore8B?
=
	full_text0
.
,store i64 %682, i64* %237, align 8, !tbaa !8
&i648B

	full_text


i64 %682
(i64*8B

	full_text

	i64* %237
Kstore8B@
>
	full_text1
/
-store i64 %668, i64* %246, align 16, !tbaa !8
&i648B

	full_text


i64 %668
(i64*8B

	full_text

	i64* %246
Kload8BA
?
	full_text2
0
.%697 = load i64, i64* %247, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %247
Jload8B@
>
	full_text1
/
-%698 = load i64, i64* %249, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %249
Kstore8B@
>
	full_text1
/
-store i64 %698, i64* %252, align 16, !tbaa !8
&i648B

	full_text


i64 %698
(i64*8B

	full_text

	i64* %252
Jstore8B?
=
	full_text0
.
,store i64 %679, i64* %257, align 8, !tbaa !8
&i648B

	full_text


i64 %679
(i64*8B

	full_text

	i64* %257
Jstore8B?
=
	full_text0
.
,store i64 %678, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %678
(i64*8B

	full_text

	i64* %254
Istore8B>
<
	full_text/
-
+store i64 %676, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %676
'i64*8B

	full_text


i64* %52
Jstore8B?
=
	full_text0
.
,store i64 %667, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %667
(i64*8B

	full_text

	i64* %263
Jload8B@
>
	full_text1
/
-%699 = load i64, i64* %264, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %264
Jload8B@
>
	full_text1
/
-%700 = load i64, i64* %266, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %266
Jstore8B?
=
	full_text0
.
,store i64 %700, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %700
(i64*8B

	full_text

	i64* %269
Kstore8B@
>
	full_text1
/
-store i64 %674, i64* %274, align 16, !tbaa !8
&i648B

	full_text


i64 %674
(i64*8B

	full_text

	i64* %274
Jstore8B?
=
	full_text0
.
,store i64 %673, i64* %271, align 8, !tbaa !8
&i648B

	full_text


i64 %673
(i64*8B

	full_text

	i64* %271
Kstore8B@
>
	full_text1
/
-store i64 %666, i64* %281, align 16, !tbaa !8
&i648B

	full_text


i64 %666
(i64*8B

	full_text

	i64* %281
Kload8BA
?
	full_text2
0
.%701 = load i64, i64* %282, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %282
Jload8B@
>
	full_text1
/
-%702 = load i64, i64* %284, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %284
Kstore8B@
>
	full_text1
/
-store i64 %702, i64* %287, align 16, !tbaa !8
&i648B

	full_text


i64 %702
(i64*8B

	full_text

	i64* %287
:add8B1
/
	full_text"
 
%703 = add nuw nsw i64 %692, 2
&i648B

	full_text


i64 %692
ùgetelementptr8Bâ
Ü
	full_texty
w
u%704 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %703, i64 %30, i64 %32
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %703
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%705 = bitcast [5 x double]* %704 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %704
Jload8B@
>
	full_text1
/
-%706 = load i64, i64* %705, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %705
Kstore8B@
>
	full_text1
/
-store i64 %706, i64* %177, align 16, !tbaa !8
&i648B

	full_text


i64 %706
(i64*8B

	full_text

	i64* %177
•getelementptr8Bë
é
	full_textÄ
~
|%707 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %703, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %703
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%708 = bitcast double* %707 to i64*
.double*8B

	full_text

double* %707
Jload8B@
>
	full_text1
/
-%709 = load i64, i64* %708, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %708
Jstore8B?
=
	full_text0
.
,store i64 %709, i64* %182, align 8, !tbaa !8
&i648B

	full_text


i64 %709
(i64*8B

	full_text

	i64* %182
•getelementptr8Bë
é
	full_textÄ
~
|%710 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %703, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %703
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%711 = bitcast double* %710 to i64*
.double*8B

	full_text

double* %710
Jload8B@
>
	full_text1
/
-%712 = load i64, i64* %711, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %711
Kstore8B@
>
	full_text1
/
-store i64 %712, i64* %187, align 16, !tbaa !8
&i648B

	full_text


i64 %712
(i64*8B

	full_text

	i64* %187
•getelementptr8Bë
é
	full_textÄ
~
|%713 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %703, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %703
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%714 = bitcast double* %713 to i64*
.double*8B

	full_text

double* %713
Jload8B@
>
	full_text1
/
-%715 = load i64, i64* %714, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %714
Jstore8B?
=
	full_text0
.
,store i64 %715, i64* %192, align 8, !tbaa !8
&i648B

	full_text


i64 %715
(i64*8B

	full_text

	i64* %192
•getelementptr8Bë
é
	full_textÄ
~
|%716 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %703, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %703
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%717 = bitcast double* %716 to i64*
.double*8B

	full_text

double* %716
Jload8B@
>
	full_text1
/
-%718 = load i64, i64* %717, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %717
Kstore8B@
>
	full_text1
/
-store i64 %718, i64* %197, align 16, !tbaa !8
&i648B

	full_text


i64 %718
(i64*8B

	full_text

	i64* %197
Qstore8BF
D
	full_text7
5
3store double %675, double* %658, align 16, !tbaa !8
,double8B

	full_text

double %675
.double*8B

	full_text

double* %658
ègetelementptr8B|
z
	full_textm
k
i%719 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %693, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %693
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%720 = load double, double* %719, align 8, !tbaa !8
.double*8B

	full_text

double* %719
:fmul8B0
.
	full_text!

%721 = fmul double %720, %675
,double8B

	full_text

double %720
,double8B

	full_text

double %675
ègetelementptr8B|
z
	full_textm
k
i%722 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %693, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
&i648B

	full_text


i64 %693
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%723 = load double, double* %722, align 8, !tbaa !8
.double*8B

	full_text

double* %722
:fmul8B0
.
	full_text!

%724 = fmul double %721, %684
,double8B

	full_text

double %721
,double8B

	full_text

double %684
Pstore8BE
C
	full_text6
4
2store double %724, double* %127, align 8, !tbaa !8
,double8B

	full_text

double %724
.double*8B

	full_text

double* %127
:fmul8B0
.
	full_text!

%725 = fmul double %721, %680
,double8B

	full_text

double %721
,double8B

	full_text

double %680
Qstore8BF
D
	full_text7
5
3store double %725, double* %130, align 16, !tbaa !8
,double8B

	full_text

double %725
.double*8B

	full_text

double* %130
:fsub8B0
.
	full_text!

%726 = fsub double %671, %723
,double8B

	full_text

double %671
,double8B

	full_text

double %723
Bfmul8B8
6
	full_text)
'
%%727 = fmul double %726, 4.000000e-01
,double8B

	full_text

double %726
mcall8Bc
a
	full_textT
R
P%728 = tail call double @llvm.fmuladd.f64(double %675, double %721, double %727)
,double8B

	full_text

double %675
,double8B

	full_text

double %721
,double8B

	full_text

double %727
Pstore8BE
C
	full_text6
4
2store double %728, double* %135, align 8, !tbaa !8
,double8B

	full_text

double %728
.double*8B

	full_text

double* %135
Bfmul8B8
6
	full_text)
'
%%729 = fmul double %723, 4.000000e-01
,double8B

	full_text

double %723
Cfsub8B9
7
	full_text*
(
&%730 = fsub double -0.000000e+00, %729
,double8B

	full_text

double %729
ucall8Bk
i
	full_text\
Z
X%731 = tail call double @llvm.fmuladd.f64(double %671, double 1.400000e+00, double %730)
,double8B

	full_text

double %671
,double8B

	full_text

double %730
:fmul8B0
.
	full_text!

%732 = fmul double %721, %731
,double8B

	full_text

double %721
,double8B

	full_text

double %731
Qstore8BF
D
	full_text7
5
3store double %732, double* %140, align 16, !tbaa !8
,double8B

	full_text

double %732
.double*8B

	full_text

double* %140
:fmul8B0
.
	full_text!

%733 = fmul double %720, %684
,double8B

	full_text

double %720
,double8B

	full_text

double %684
:fmul8B0
.
	full_text!

%734 = fmul double %720, %680
,double8B

	full_text

double %720
,double8B

	full_text

double %680
:fmul8B0
.
	full_text!

%735 = fmul double %720, %671
,double8B

	full_text

double %720
,double8B

	full_text

double %671
ègetelementptr8B|
z
	full_textm
k
i%736 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %692, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %692
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%737 = load double, double* %736, align 8, !tbaa !8
.double*8B

	full_text

double* %736
:fmul8B0
.
	full_text!

%738 = fmul double %737, %685
,double8B

	full_text

double %737
,double8B

	full_text

double %685
:fmul8B0
.
	full_text!

%739 = fmul double %737, %681
,double8B

	full_text

double %737
,double8B

	full_text

double %681
:fmul8B0
.
	full_text!

%740 = fmul double %737, %677
,double8B

	full_text

double %737
,double8B

	full_text

double %677
:fmul8B0
.
	full_text!

%741 = fmul double %737, %672
,double8B

	full_text

double %737
,double8B

	full_text

double %672
:fsub8B0
.
	full_text!

%742 = fsub double %733, %738
,double8B

	full_text

double %733
,double8B

	full_text

double %738
Bfmul8B8
6
	full_text)
'
%%743 = fmul double %742, 6.300000e+01
,double8B

	full_text

double %742
Pstore8BE
C
	full_text6
4
2store double %743, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %743
.double*8B

	full_text

double* %149
:fsub8B0
.
	full_text!

%744 = fsub double %734, %739
,double8B

	full_text

double %734
,double8B

	full_text

double %739
Bfmul8B8
6
	full_text)
'
%%745 = fmul double %744, 6.300000e+01
,double8B

	full_text

double %744
Pstore8BE
C
	full_text6
4
2store double %745, double* %152, align 8, !tbaa !8
,double8B

	full_text

double %745
.double*8B

	full_text

double* %152
:fsub8B0
.
	full_text!

%746 = fsub double %721, %740
,double8B

	full_text

double %721
,double8B

	full_text

double %740
Bfmul8B8
6
	full_text)
'
%%747 = fmul double %746, 8.400000e+01
,double8B

	full_text

double %746
Pstore8BE
C
	full_text6
4
2store double %747, double* %155, align 8, !tbaa !8
,double8B

	full_text

double %747
.double*8B

	full_text

double* %155
:fmul8B0
.
	full_text!

%748 = fmul double %734, %734
,double8B

	full_text

double %734
,double8B

	full_text

double %734
mcall8Bc
a
	full_textT
R
P%749 = tail call double @llvm.fmuladd.f64(double %733, double %733, double %748)
,double8B

	full_text

double %733
,double8B

	full_text

double %733
,double8B

	full_text

double %748
mcall8Bc
a
	full_textT
R
P%750 = tail call double @llvm.fmuladd.f64(double %721, double %721, double %749)
,double8B

	full_text

double %721
,double8B

	full_text

double %721
,double8B

	full_text

double %749
:fmul8B0
.
	full_text!

%751 = fmul double %739, %739
,double8B

	full_text

double %739
,double8B

	full_text

double %739
mcall8Bc
a
	full_textT
R
P%752 = tail call double @llvm.fmuladd.f64(double %738, double %738, double %751)
,double8B

	full_text

double %738
,double8B

	full_text

double %738
,double8B

	full_text

double %751
mcall8Bc
a
	full_textT
R
P%753 = tail call double @llvm.fmuladd.f64(double %740, double %740, double %752)
,double8B

	full_text

double %740
,double8B

	full_text

double %740
,double8B

	full_text

double %752
:fsub8B0
.
	full_text!

%754 = fsub double %750, %753
,double8B

	full_text

double %750
,double8B

	full_text

double %753
:fmul8B0
.
	full_text!

%755 = fmul double %740, %740
,double8B

	full_text

double %740
,double8B

	full_text

double %740
Cfsub8B9
7
	full_text*
(
&%756 = fsub double -0.000000e+00, %755
,double8B

	full_text

double %755
mcall8Bc
a
	full_textT
R
P%757 = tail call double @llvm.fmuladd.f64(double %721, double %721, double %756)
,double8B

	full_text

double %721
,double8B

	full_text

double %721
,double8B

	full_text

double %756
Bfmul8B8
6
	full_text)
'
%%758 = fmul double %757, 1.050000e+01
,double8B

	full_text

double %757
{call8Bq
o
	full_textb
`
^%759 = tail call double @llvm.fmuladd.f64(double %754, double 0xC03E3D70A3D70A3B, double %758)
,double8B

	full_text

double %754
,double8B

	full_text

double %758
:fsub8B0
.
	full_text!

%760 = fsub double %735, %741
,double8B

	full_text

double %735
,double8B

	full_text

double %741
{call8Bq
o
	full_textb
`
^%761 = tail call double @llvm.fmuladd.f64(double %760, double 0x405EDEB851EB851E, double %759)
,double8B

	full_text

double %760
,double8B

	full_text

double %759
Pstore8BE
C
	full_text6
4
2store double %761, double* %170, align 8, !tbaa !8
,double8B

	full_text

double %761
.double*8B

	full_text

double* %170
•getelementptr8Bë
é
	full_textÄ
~
|%762 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %692, i64 %30, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %692
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%763 = load double, double* %762, align 8, !tbaa !8
.double*8B

	full_text

double* %762
Qload8BG
E
	full_text8
6
4%764 = load double, double* %210, align 16, !tbaa !8
.double*8B

	full_text

double* %210
:fsub8B0
.
	full_text!

%765 = fsub double %675, %764
,double8B

	full_text

double %675
,double8B

	full_text

double %764
vcall8Bl
j
	full_text]
[
Y%766 = tail call double @llvm.fmuladd.f64(double %765, double -3.150000e+01, double %763)
,double8B

	full_text

double %765
,double8B

	full_text

double %763
•getelementptr8Bë
é
	full_textÄ
~
|%767 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %692, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %692
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%768 = load double, double* %767, align 8, !tbaa !8
.double*8B

	full_text

double* %767
Pload8BF
D
	full_text7
5
3%769 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
:fsub8B0
.
	full_text!

%770 = fsub double %724, %769
,double8B

	full_text

double %724
,double8B

	full_text

double %769
vcall8Bl
j
	full_text]
[
Y%771 = tail call double @llvm.fmuladd.f64(double %770, double -3.150000e+01, double %768)
,double8B

	full_text

double %770
,double8B

	full_text

double %768
•getelementptr8Bë
é
	full_textÄ
~
|%772 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %692, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %692
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%773 = load double, double* %772, align 8, !tbaa !8
.double*8B

	full_text

double* %772
Qload8BG
E
	full_text8
6
4%774 = load double, double* %245, align 16, !tbaa !8
.double*8B

	full_text

double* %245
:fsub8B0
.
	full_text!

%775 = fsub double %725, %774
,double8B

	full_text

double %725
,double8B

	full_text

double %774
vcall8Bl
j
	full_text]
[
Y%776 = tail call double @llvm.fmuladd.f64(double %775, double -3.150000e+01, double %773)
,double8B

	full_text

double %775
,double8B

	full_text

double %773
•getelementptr8Bë
é
	full_textÄ
~
|%777 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %692, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %692
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%778 = load double, double* %777, align 8, !tbaa !8
.double*8B

	full_text

double* %777
Pload8BF
D
	full_text7
5
3%779 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
:fsub8B0
.
	full_text!

%780 = fsub double %728, %779
,double8B

	full_text

double %728
,double8B

	full_text

double %779
vcall8Bl
j
	full_text]
[
Y%781 = tail call double @llvm.fmuladd.f64(double %780, double -3.150000e+01, double %778)
,double8B

	full_text

double %780
,double8B

	full_text

double %778
•getelementptr8Bë
é
	full_textÄ
~
|%782 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %692, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %692
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%783 = load double, double* %782, align 8, !tbaa !8
.double*8B

	full_text

double* %782
Qload8BG
E
	full_text8
6
4%784 = load double, double* %280, align 16, !tbaa !8
.double*8B

	full_text

double* %280
:fsub8B0
.
	full_text!

%785 = fsub double %732, %784
,double8B

	full_text

double %732
,double8B

	full_text

double %784
vcall8Bl
j
	full_text]
[
Y%786 = tail call double @llvm.fmuladd.f64(double %785, double -3.150000e+01, double %783)
,double8B

	full_text

double %785
,double8B

	full_text

double %783
vcall8Bl
j
	full_text]
[
Y%787 = tail call double @llvm.fmuladd.f64(double %689, double -2.000000e+00, double %690)
,double8B

	full_text

double %689
,double8B

	full_text

double %690
Oload8BE
C
	full_text6
4
2%788 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%789 = fadd double %787, %788
,double8B

	full_text

double %787
,double8B

	full_text

double %788
{call8Bq
o
	full_textb
`
^%790 = tail call double @llvm.fmuladd.f64(double %789, double 0x40AF020000000001, double %766)
,double8B

	full_text

double %789
,double8B

	full_text

double %766
Pload8BF
D
	full_text7
5
3%791 = load double, double* %234, align 8, !tbaa !8
.double*8B

	full_text

double* %234
:fsub8B0
.
	full_text!

%792 = fsub double %743, %791
,double8B

	full_text

double %743
,double8B

	full_text

double %791
{call8Bq
o
	full_textb
`
^%793 = tail call double @llvm.fmuladd.f64(double %792, double 0x4019333333333334, double %771)
,double8B

	full_text

double %792
,double8B

	full_text

double %771
Pload8BF
D
	full_text7
5
3%794 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
vcall8Bl
j
	full_text]
[
Y%795 = tail call double @llvm.fmuladd.f64(double %685, double -2.000000e+00, double %794)
,double8B

	full_text

double %685
,double8B

	full_text

double %794
:fadd8B0
.
	full_text!

%796 = fadd double %684, %795
,double8B

	full_text

double %684
,double8B

	full_text

double %795
{call8Bq
o
	full_textb
`
^%797 = tail call double @llvm.fmuladd.f64(double %796, double 0x40AF020000000001, double %793)
,double8B

	full_text

double %796
,double8B

	full_text

double %793
Qload8BG
E
	full_text8
6
4%798 = load double, double* %251, align 16, !tbaa !8
.double*8B

	full_text

double* %251
:fsub8B0
.
	full_text!

%799 = fsub double %745, %798
,double8B

	full_text

double %745
,double8B

	full_text

double %798
{call8Bq
o
	full_textb
`
^%800 = tail call double @llvm.fmuladd.f64(double %799, double 0x4019333333333334, double %776)
,double8B

	full_text

double %799
,double8B

	full_text

double %776
Pload8BF
D
	full_text7
5
3%801 = load double, double* %236, align 8, !tbaa !8
.double*8B

	full_text

double* %236
vcall8Bl
j
	full_text]
[
Y%802 = tail call double @llvm.fmuladd.f64(double %681, double -2.000000e+00, double %801)
,double8B

	full_text

double %681
,double8B

	full_text

double %801
:fadd8B0
.
	full_text!

%803 = fadd double %680, %802
,double8B

	full_text

double %680
,double8B

	full_text

double %802
{call8Bq
o
	full_textb
`
^%804 = tail call double @llvm.fmuladd.f64(double %803, double 0x40AF020000000001, double %800)
,double8B

	full_text

double %803
,double8B

	full_text

double %800
Pload8BF
D
	full_text7
5
3%805 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
:fsub8B0
.
	full_text!

%806 = fsub double %747, %805
,double8B

	full_text

double %747
,double8B

	full_text

double %805
{call8Bq
o
	full_textb
`
^%807 = tail call double @llvm.fmuladd.f64(double %806, double 0x4019333333333334, double %781)
,double8B

	full_text

double %806
,double8B

	full_text

double %781
Pload8BF
D
	full_text7
5
3%808 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
vcall8Bl
j
	full_text]
[
Y%809 = tail call double @llvm.fmuladd.f64(double %677, double -2.000000e+00, double %808)
,double8B

	full_text

double %677
,double8B

	full_text

double %808
:fadd8B0
.
	full_text!

%810 = fadd double %675, %809
,double8B

	full_text

double %675
,double8B

	full_text

double %809
{call8Bq
o
	full_textb
`
^%811 = tail call double @llvm.fmuladd.f64(double %810, double 0x40AF020000000001, double %807)
,double8B

	full_text

double %810
,double8B

	full_text

double %807
Qload8BG
E
	full_text8
6
4%812 = load double, double* %286, align 16, !tbaa !8
.double*8B

	full_text

double* %286
:fsub8B0
.
	full_text!

%813 = fsub double %761, %812
,double8B

	full_text

double %761
,double8B

	full_text

double %812
{call8Bq
o
	full_textb
`
^%814 = tail call double @llvm.fmuladd.f64(double %813, double 0x4019333333333334, double %786)
,double8B

	full_text

double %813
,double8B

	full_text

double %786
Pload8BF
D
	full_text7
5
3%815 = load double, double* %270, align 8, !tbaa !8
.double*8B

	full_text

double* %270
vcall8Bl
j
	full_text]
[
Y%816 = tail call double @llvm.fmuladd.f64(double %672, double -2.000000e+00, double %815)
,double8B

	full_text

double %672
,double8B

	full_text

double %815
:fadd8B0
.
	full_text!

%817 = fadd double %671, %816
,double8B

	full_text

double %671
,double8B

	full_text

double %816
{call8Bq
o
	full_textb
`
^%818 = tail call double @llvm.fmuladd.f64(double %817, double 0x40AF020000000001, double %814)
,double8B

	full_text

double %817
,double8B

	full_text

double %814
vcall8Bl
j
	full_text]
[
Y%819 = tail call double @llvm.fmuladd.f64(double %690, double -4.000000e+00, double %691)
,double8B

	full_text

double %690
,double8B

	full_text

double %691
ucall8Bk
i
	full_text\
Z
X%820 = tail call double @llvm.fmuladd.f64(double %689, double 6.000000e+00, double %819)
,double8B

	full_text

double %689
,double8B

	full_text

double %819
vcall8Bl
j
	full_text]
[
Y%821 = tail call double @llvm.fmuladd.f64(double %788, double -4.000000e+00, double %820)
,double8B

	full_text

double %788
,double8B

	full_text

double %820
Qload8BG
E
	full_text8
6
4%822 = load double, double* %205, align 16, !tbaa !8
.double*8B

	full_text

double* %205
:fadd8B0
.
	full_text!

%823 = fadd double %821, %822
,double8B

	full_text

double %821
,double8B

	full_text

double %822
mcall8Bc
a
	full_textT
R
P%824 = tail call double @llvm.fmuladd.f64(double %422, double %823, double %790)
,double8B

	full_text

double %422
,double8B

	full_text

double %823
,double8B

	full_text

double %790
Pstore8BE
C
	full_text6
4
2store double %824, double* %762, align 8, !tbaa !8
,double8B

	full_text

double %824
.double*8B

	full_text

double* %762
Pload8BF
D
	full_text7
5
3%825 = load double, double* %222, align 8, !tbaa !8
.double*8B

	full_text

double* %222
vcall8Bl
j
	full_text]
[
Y%826 = tail call double @llvm.fmuladd.f64(double %794, double -4.000000e+00, double %825)
,double8B

	full_text

double %794
,double8B

	full_text

double %825
ucall8Bk
i
	full_text\
Z
X%827 = tail call double @llvm.fmuladd.f64(double %685, double 6.000000e+00, double %826)
,double8B

	full_text

double %685
,double8B

	full_text

double %826
vcall8Bl
j
	full_text]
[
Y%828 = tail call double @llvm.fmuladd.f64(double %684, double -4.000000e+00, double %827)
,double8B

	full_text

double %684
,double8B

	full_text

double %827
Pload8BF
D
	full_text7
5
3%829 = load double, double* %181, align 8, !tbaa !8
.double*8B

	full_text

double* %181
:fadd8B0
.
	full_text!

%830 = fadd double %828, %829
,double8B

	full_text

double %828
,double8B

	full_text

double %829
mcall8Bc
a
	full_textT
R
P%831 = tail call double @llvm.fmuladd.f64(double %422, double %830, double %797)
,double8B

	full_text

double %422
,double8B

	full_text

double %830
,double8B

	full_text

double %797
Pstore8BE
C
	full_text6
4
2store double %831, double* %767, align 8, !tbaa !8
,double8B

	full_text

double %831
.double*8B

	full_text

double* %767
Qload8BG
E
	full_text8
6
4%832 = load double, double* %239, align 16, !tbaa !8
.double*8B

	full_text

double* %239
vcall8Bl
j
	full_text]
[
Y%833 = tail call double @llvm.fmuladd.f64(double %801, double -4.000000e+00, double %832)
,double8B

	full_text

double %801
,double8B

	full_text

double %832
ucall8Bk
i
	full_text\
Z
X%834 = tail call double @llvm.fmuladd.f64(double %681, double 6.000000e+00, double %833)
,double8B

	full_text

double %681
,double8B

	full_text

double %833
vcall8Bl
j
	full_text]
[
Y%835 = tail call double @llvm.fmuladd.f64(double %680, double -4.000000e+00, double %834)
,double8B

	full_text

double %680
,double8B

	full_text

double %834
Qload8BG
E
	full_text8
6
4%836 = load double, double* %186, align 16, !tbaa !8
.double*8B

	full_text

double* %186
:fadd8B0
.
	full_text!

%837 = fadd double %835, %836
,double8B

	full_text

double %835
,double8B

	full_text

double %836
mcall8Bc
a
	full_textT
R
P%838 = tail call double @llvm.fmuladd.f64(double %422, double %837, double %804)
,double8B

	full_text

double %422
,double8B

	full_text

double %837
,double8B

	full_text

double %804
Pstore8BE
C
	full_text6
4
2store double %838, double* %772, align 8, !tbaa !8
,double8B

	full_text

double %838
.double*8B

	full_text

double* %772
Pload8BF
D
	full_text7
5
3%839 = load double, double* %256, align 8, !tbaa !8
.double*8B

	full_text

double* %256
vcall8Bl
j
	full_text]
[
Y%840 = tail call double @llvm.fmuladd.f64(double %808, double -4.000000e+00, double %839)
,double8B

	full_text

double %808
,double8B

	full_text

double %839
ucall8Bk
i
	full_text\
Z
X%841 = tail call double @llvm.fmuladd.f64(double %677, double 6.000000e+00, double %840)
,double8B

	full_text

double %677
,double8B

	full_text

double %840
Oload8BE
C
	full_text6
4
2%842 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
vcall8Bl
j
	full_text]
[
Y%843 = tail call double @llvm.fmuladd.f64(double %842, double -4.000000e+00, double %841)
,double8B

	full_text

double %842
,double8B

	full_text

double %841
Pload8BF
D
	full_text7
5
3%844 = load double, double* %191, align 8, !tbaa !8
.double*8B

	full_text

double* %191
:fadd8B0
.
	full_text!

%845 = fadd double %843, %844
,double8B

	full_text

double %843
,double8B

	full_text

double %844
mcall8Bc
a
	full_textT
R
P%846 = tail call double @llvm.fmuladd.f64(double %422, double %845, double %811)
,double8B

	full_text

double %422
,double8B

	full_text

double %845
,double8B

	full_text

double %811
Pstore8BE
C
	full_text6
4
2store double %846, double* %777, align 8, !tbaa !8
,double8B

	full_text

double %846
.double*8B

	full_text

double* %777
Qload8BG
E
	full_text8
6
4%847 = load double, double* %273, align 16, !tbaa !8
.double*8B

	full_text

double* %273
vcall8Bl
j
	full_text]
[
Y%848 = tail call double @llvm.fmuladd.f64(double %815, double -4.000000e+00, double %847)
,double8B

	full_text

double %815
,double8B

	full_text

double %847
ucall8Bk
i
	full_text\
Z
X%849 = tail call double @llvm.fmuladd.f64(double %672, double 6.000000e+00, double %848)
,double8B

	full_text

double %672
,double8B

	full_text

double %848
vcall8Bl
j
	full_text]
[
Y%850 = tail call double @llvm.fmuladd.f64(double %671, double -4.000000e+00, double %849)
,double8B

	full_text

double %671
,double8B

	full_text

double %849
Qload8BG
E
	full_text8
6
4%851 = load double, double* %196, align 16, !tbaa !8
.double*8B

	full_text

double* %196
:fadd8B0
.
	full_text!

%852 = fadd double %850, %851
,double8B

	full_text

double %850
,double8B

	full_text

double %851
mcall8Bc
a
	full_textT
R
P%853 = tail call double @llvm.fmuladd.f64(double %422, double %852, double %818)
,double8B

	full_text

double %422
,double8B

	full_text

double %852
,double8B

	full_text

double %818
Pstore8BE
C
	full_text6
4
2store double %853, double* %782, align 8, !tbaa !8
,double8B

	full_text

double %853
.double*8B

	full_text

double* %782
:icmp8B0
.
	full_text!

%854 = icmp eq i64 %693, %659
&i648B

	full_text


i64 %693
&i648B

	full_text


i64 %659
Abitcast8B4
2
	full_text%
#
!%855 = bitcast double %822 to i64
,double8B

	full_text

double %822
Abitcast8B4
2
	full_text%
#
!%856 = bitcast double %794 to i64
,double8B

	full_text

double %794
Abitcast8B4
2
	full_text%
#
!%857 = bitcast double %685 to i64
,double8B

	full_text

double %685
Abitcast8B4
2
	full_text%
#
!%858 = bitcast double %801 to i64
,double8B

	full_text

double %801
Abitcast8B4
2
	full_text%
#
!%859 = bitcast double %681 to i64
,double8B

	full_text

double %681
Abitcast8B4
2
	full_text%
#
!%860 = bitcast double %808 to i64
,double8B

	full_text

double %808
Abitcast8B4
2
	full_text%
#
!%861 = bitcast double %677 to i64
,double8B

	full_text

double %677
Abitcast8B4
2
	full_text%
#
!%862 = bitcast double %844 to i64
,double8B

	full_text

double %844
Abitcast8B4
2
	full_text%
#
!%863 = bitcast double %815 to i64
,double8B

	full_text

double %815
Abitcast8B4
2
	full_text%
#
!%864 = bitcast double %672 to i64
,double8B

	full_text

double %672
=br8B5
3
	full_text&
$
"br i1 %854, label %865, label %665
$i18B

	full_text
	
i1 %854
Qstore8BF
D
	full_text7
5
3store double %691, double* %656, align 16, !tbaa !8
,double8B

	full_text

double %691
.double*8B

	full_text

double* %656
Pstore8BE
C
	full_text6
4
2store double %690, double* %198, align 8, !tbaa !8
,double8B

	full_text

double %690
.double*8B

	full_text

double* %198
Pstore8BE
C
	full_text6
4
2store double %689, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %689
-double*8B

	full_text

double* %84
Jstore8B?
=
	full_text0
.
,store i64 %694, i64* %208, align 8, !tbaa !8
&i648B

	full_text


i64 %694
(i64*8B

	full_text

	i64* %208
Kstore8B@
>
	full_text1
/
-store i64 %657, i64* %218, align 16, !tbaa !8
&i648B

	full_text


i64 %657
(i64*8B

	full_text

	i64* %218
Ostore8BD
B
	full_text5
3
1store double %685, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %685
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %684, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %684
-double*8B

	full_text

double* %41
Jstore8B?
=
	full_text0
.
,store i64 %695, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %695
(i64*8B

	full_text

	i64* %226
Pstore8BE
C
	full_text6
4
2store double %681, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %681
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %680, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %680
-double*8B

	full_text

double* %46
Jstore8B?
=
	full_text0
.
,store i64 %697, i64* %243, align 8, !tbaa !8
&i648B

	full_text


i64 %697
(i64*8B

	full_text

	i64* %243
Ostore8BD
B
	full_text5
3
1store double %677, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %677
-double*8B

	full_text

double* %90
Jstore8B?
=
	full_text0
.
,store i64 %699, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %699
(i64*8B

	full_text

	i64* %260
Pstore8BE
C
	full_text6
4
2store double %672, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %672
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %671, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %671
-double*8B

	full_text

double* %56
Jstore8B?
=
	full_text0
.
,store i64 %701, i64* %278, align 8, !tbaa !8
&i648B

	full_text


i64 %701
(i64*8B

	full_text

	i64* %278
Jload8B@
>
	full_text1
/
-%866 = load i64, i64* %220, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %220
(br8B 

	full_text

br label %867
Kphi8BB
@
	full_text3
1
/%868 = phi double [ %851, %865 ], [ %640, %25 ]
,double8B

	full_text

double %851
,double8B

	full_text

double %640
Kphi8BB
@
	full_text3
1
/%869 = phi double [ %671, %865 ], [ %638, %25 ]
,double8B

	full_text

double %671
,double8B

	full_text

double %638
Hphi8B?
=
	full_text0
.
,%870 = phi i64 [ %864, %865 ], [ %654, %25 ]
&i648B

	full_text


i64 %864
&i648B

	full_text


i64 %654
Hphi8B?
=
	full_text0
.
,%871 = phi i64 [ %863, %865 ], [ %653, %25 ]
&i648B

	full_text


i64 %863
&i648B

	full_text


i64 %653
Kphi8BB
@
	full_text3
1
/%872 = phi double [ %844, %865 ], [ %632, %25 ]
,double8B

	full_text

double %844
,double8B

	full_text

double %632
Hphi8B?
=
	full_text0
.
,%873 = phi i64 [ %862, %865 ], [ %652, %25 ]
&i648B

	full_text


i64 %862
&i648B

	full_text


i64 %652
Kphi8BB
@
	full_text3
1
/%874 = phi double [ %842, %865 ], [ %630, %25 ]
,double8B

	full_text

double %842
,double8B

	full_text

double %630
Hphi8B?
=
	full_text0
.
,%875 = phi i64 [ %861, %865 ], [ %651, %25 ]
&i648B

	full_text


i64 %861
&i648B

	full_text


i64 %651
Hphi8B?
=
	full_text0
.
,%876 = phi i64 [ %860, %865 ], [ %650, %25 ]
&i648B

	full_text


i64 %860
&i648B

	full_text


i64 %650
Kphi8BB
@
	full_text3
1
/%877 = phi double [ %836, %865 ], [ %624, %25 ]
,double8B

	full_text

double %836
,double8B

	full_text

double %624
Kphi8BB
@
	full_text3
1
/%878 = phi double [ %680, %865 ], [ %622, %25 ]
,double8B

	full_text

double %680
,double8B

	full_text

double %622
Hphi8B?
=
	full_text0
.
,%879 = phi i64 [ %859, %865 ], [ %649, %25 ]
&i648B

	full_text


i64 %859
&i648B

	full_text


i64 %649
Hphi8B?
=
	full_text0
.
,%880 = phi i64 [ %858, %865 ], [ %648, %25 ]
&i648B

	full_text


i64 %858
&i648B

	full_text


i64 %648
Kphi8BB
@
	full_text3
1
/%881 = phi double [ %829, %865 ], [ %616, %25 ]
,double8B

	full_text

double %829
,double8B

	full_text

double %616
Kphi8BB
@
	full_text3
1
/%882 = phi double [ %684, %865 ], [ %614, %25 ]
,double8B

	full_text

double %684
,double8B

	full_text

double %614
Hphi8B?
=
	full_text0
.
,%883 = phi i64 [ %857, %865 ], [ %647, %25 ]
&i648B

	full_text


i64 %857
&i648B

	full_text


i64 %647
Hphi8B?
=
	full_text0
.
,%884 = phi i64 [ %866, %865 ], [ %646, %25 ]
&i648B

	full_text


i64 %866
&i648B

	full_text


i64 %646
Hphi8B?
=
	full_text0
.
,%885 = phi i64 [ %855, %865 ], [ %645, %25 ]
&i648B

	full_text


i64 %855
&i648B

	full_text


i64 %645
5add8B,
*
	full_text

%886 = add nsw i32 %6, -2
Jload8B@
>
	full_text1
/
-%887 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
Kstore8B@
>
	full_text1
/
-store i64 %887, i64* %202, align 16, !tbaa !8
&i648B

	full_text


i64 %887
(i64*8B

	full_text

	i64* %202
Jload8B@
>
	full_text1
/
-%888 = load i64, i64* %85, align 16, !tbaa !8
'i64*8B

	full_text


i64* %85
Jstore8B?
=
	full_text0
.
,store i64 %888, i64* %199, align 8, !tbaa !8
&i648B

	full_text


i64 %888
(i64*8B

	full_text

	i64* %199
Iload8B?
=
	full_text0
.
,%889 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
Jstore8B?
=
	full_text0
.
,store i64 %889, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %889
'i64*8B

	full_text


i64* %85
Istore8B>
<
	full_text/
-
+store i64 %885, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %885
'i64*8B

	full_text


i64* %83
Jload8B@
>
	full_text1
/
-%890 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Kstore8B@
>
	full_text1
/
-store i64 %890, i64* %211, align 16, !tbaa !8
&i648B

	full_text


i64 %890
(i64*8B

	full_text

	i64* %211
Kload8BA
?
	full_text2
0
.%891 = load i64, i64* %213, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %213
Jstore8B?
=
	full_text0
.
,store i64 %891, i64* %208, align 8, !tbaa !8
&i648B

	full_text


i64 %891
(i64*8B

	full_text

	i64* %208
Jload8B@
>
	full_text1
/
-%892 = load i64, i64* %216, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %216
Kstore8B@
>
	full_text1
/
-store i64 %892, i64* %218, align 16, !tbaa !8
&i648B

	full_text


i64 %892
(i64*8B

	full_text

	i64* %218
Jstore8B?
=
	full_text0
.
,store i64 %884, i64* %223, align 8, !tbaa !8
&i648B

	full_text


i64 %884
(i64*8B

	full_text

	i64* %223
Jstore8B?
=
	full_text0
.
,store i64 %883, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %883
(i64*8B

	full_text

	i64* %220
Ostore8BD
B
	full_text5
3
1store double %882, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %882
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %881, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %881
-double*8B

	full_text

double* %41
Jload8B@
>
	full_text1
/
-%893 = load i64, i64* %226, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %226
Jstore8B?
=
	full_text0
.
,store i64 %893, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %893
(i64*8B

	full_text

	i64* %229
Jload8B@
>
	full_text1
/
-%894 = load i64, i64* %230, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %230
Jstore8B?
=
	full_text0
.
,store i64 %894, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %894
(i64*8B

	full_text

	i64* %226
Jload8B@
>
	full_text1
/
-%895 = load i64, i64* %232, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %232
Jstore8B?
=
	full_text0
.
,store i64 %895, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %895
(i64*8B

	full_text

	i64* %235
Kstore8B@
>
	full_text1
/
-store i64 %880, i64* %240, align 16, !tbaa !8
&i648B

	full_text


i64 %880
(i64*8B

	full_text

	i64* %240
Jstore8B?
=
	full_text0
.
,store i64 %879, i64* %237, align 8, !tbaa !8
&i648B

	full_text


i64 %879
(i64*8B

	full_text

	i64* %237
Pstore8BE
C
	full_text6
4
2store double %878, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %878
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %877, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %877
-double*8B

	full_text

double* %46
Jload8B@
>
	full_text1
/
-%896 = load i64, i64* %243, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %243
Kstore8B@
>
	full_text1
/
-store i64 %896, i64* %246, align 16, !tbaa !8
&i648B

	full_text


i64 %896
(i64*8B

	full_text

	i64* %246
Kload8BA
?
	full_text2
0
.%897 = load i64, i64* %247, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %247
Jstore8B?
=
	full_text0
.
,store i64 %897, i64* %243, align 8, !tbaa !8
&i648B

	full_text


i64 %897
(i64*8B

	full_text

	i64* %243
Jload8B@
>
	full_text1
/
-%898 = load i64, i64* %249, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %249
Kstore8B@
>
	full_text1
/
-store i64 %898, i64* %252, align 16, !tbaa !8
&i648B

	full_text


i64 %898
(i64*8B

	full_text

	i64* %252
Jstore8B?
=
	full_text0
.
,store i64 %876, i64* %257, align 8, !tbaa !8
&i648B

	full_text


i64 %876
(i64*8B

	full_text

	i64* %257
Jstore8B?
=
	full_text0
.
,store i64 %875, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %875
(i64*8B

	full_text

	i64* %254
Ostore8BD
B
	full_text5
3
1store double %874, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %874
-double*8B

	full_text

double* %90
Istore8B>
<
	full_text/
-
+store i64 %873, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %873
'i64*8B

	full_text


i64* %52
Jload8B@
>
	full_text1
/
-%899 = load i64, i64* %260, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %260
Jstore8B?
=
	full_text0
.
,store i64 %899, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %899
(i64*8B

	full_text

	i64* %263
Jload8B@
>
	full_text1
/
-%900 = load i64, i64* %264, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %264
Jstore8B?
=
	full_text0
.
,store i64 %900, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %900
(i64*8B

	full_text

	i64* %260
Jload8B@
>
	full_text1
/
-%901 = load i64, i64* %266, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %266
Jstore8B?
=
	full_text0
.
,store i64 %901, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %901
(i64*8B

	full_text

	i64* %269
Kstore8B@
>
	full_text1
/
-store i64 %871, i64* %274, align 16, !tbaa !8
&i648B

	full_text


i64 %871
(i64*8B

	full_text

	i64* %274
Jstore8B?
=
	full_text0
.
,store i64 %870, i64* %271, align 8, !tbaa !8
&i648B

	full_text


i64 %870
(i64*8B

	full_text

	i64* %271
Pstore8BE
C
	full_text6
4
2store double %869, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %869
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %868, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %868
-double*8B

	full_text

double* %56
Jload8B@
>
	full_text1
/
-%902 = load i64, i64* %278, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %278
Kstore8B@
>
	full_text1
/
-store i64 %902, i64* %281, align 16, !tbaa !8
&i648B

	full_text


i64 %902
(i64*8B

	full_text

	i64* %281
Kload8BA
?
	full_text2
0
.%903 = load i64, i64* %282, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %282
Jstore8B?
=
	full_text0
.
,store i64 %903, i64* %278, align 8, !tbaa !8
&i648B

	full_text


i64 %903
(i64*8B

	full_text

	i64* %278
Jload8B@
>
	full_text1
/
-%904 = load i64, i64* %284, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %284
Kstore8B@
>
	full_text1
/
-store i64 %904, i64* %287, align 16, !tbaa !8
&i648B

	full_text


i64 %904
(i64*8B

	full_text

	i64* %287
5add8B,
*
	full_text

%905 = add nsw i32 %6, -1
8sext8B.
,
	full_text

%906 = sext i32 %905 to i64
&i328B

	full_text


i32 %905
ùgetelementptr8Bâ
Ü
	full_texty
w
u%907 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %906, i64 %30, i64 %32
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%908 = bitcast [5 x double]* %907 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %907
Jload8B@
>
	full_text1
/
-%909 = load i64, i64* %908, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %908
Kstore8B@
>
	full_text1
/
-store i64 %909, i64* %177, align 16, !tbaa !8
&i648B

	full_text


i64 %909
(i64*8B

	full_text

	i64* %177
•getelementptr8Bë
é
	full_textÄ
~
|%910 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %906, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%911 = bitcast double* %910 to i64*
.double*8B

	full_text

double* %910
Jload8B@
>
	full_text1
/
-%912 = load i64, i64* %911, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %911
Jstore8B?
=
	full_text0
.
,store i64 %912, i64* %182, align 8, !tbaa !8
&i648B

	full_text


i64 %912
(i64*8B

	full_text

	i64* %182
•getelementptr8Bë
é
	full_textÄ
~
|%913 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %906, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%914 = bitcast double* %913 to i64*
.double*8B

	full_text

double* %913
Jload8B@
>
	full_text1
/
-%915 = load i64, i64* %914, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %914
Kstore8B@
>
	full_text1
/
-store i64 %915, i64* %187, align 16, !tbaa !8
&i648B

	full_text


i64 %915
(i64*8B

	full_text

	i64* %187
•getelementptr8Bë
é
	full_textÄ
~
|%916 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %906, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%917 = bitcast double* %916 to i64*
.double*8B

	full_text

double* %916
Jload8B@
>
	full_text1
/
-%918 = load i64, i64* %917, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %917
Jstore8B?
=
	full_text0
.
,store i64 %918, i64* %192, align 8, !tbaa !8
&i648B

	full_text


i64 %918
(i64*8B

	full_text

	i64* %192
•getelementptr8Bë
é
	full_textÄ
~
|%919 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %906, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%920 = bitcast double* %919 to i64*
.double*8B

	full_text

double* %919
Jload8B@
>
	full_text1
/
-%921 = load i64, i64* %920, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %920
Kstore8B@
>
	full_text1
/
-store i64 %921, i64* %197, align 16, !tbaa !8
&i648B

	full_text


i64 %921
(i64*8B

	full_text

	i64* %197
rgetelementptr8B_
]
	full_textP
N
L%922 = getelementptr inbounds [5 x double], [5 x double]* %113, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %113
Qstore8BF
D
	full_text7
5
3store double %872, double* %922, align 16, !tbaa !8
,double8B

	full_text

double %872
.double*8B

	full_text

double* %922
8sext8B.
,
	full_text

%923 = sext i32 %886 to i64
&i328B

	full_text


i32 %886
ègetelementptr8B|
z
	full_textm
k
i%924 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %923, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%925 = load double, double* %924, align 8, !tbaa !8
.double*8B

	full_text

double* %924
:fmul8B0
.
	full_text!

%926 = fmul double %925, %872
,double8B

	full_text

double %925
,double8B

	full_text

double %872
ègetelementptr8B|
z
	full_textm
k
i%927 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %923, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%928 = load double, double* %927, align 8, !tbaa !8
.double*8B

	full_text

double* %927
:fmul8B0
.
	full_text!

%929 = fmul double %926, %881
,double8B

	full_text

double %926
,double8B

	full_text

double %881
Pstore8BE
C
	full_text6
4
2store double %929, double* %127, align 8, !tbaa !8
,double8B

	full_text

double %929
.double*8B

	full_text

double* %127
:fmul8B0
.
	full_text!

%930 = fmul double %926, %877
,double8B

	full_text

double %926
,double8B

	full_text

double %877
Qstore8BF
D
	full_text7
5
3store double %930, double* %130, align 16, !tbaa !8
,double8B

	full_text

double %930
.double*8B

	full_text

double* %130
:fsub8B0
.
	full_text!

%931 = fsub double %868, %928
,double8B

	full_text

double %868
,double8B

	full_text

double %928
Bfmul8B8
6
	full_text)
'
%%932 = fmul double %931, 4.000000e-01
,double8B

	full_text

double %931
mcall8Bc
a
	full_textT
R
P%933 = tail call double @llvm.fmuladd.f64(double %872, double %926, double %932)
,double8B

	full_text

double %872
,double8B

	full_text

double %926
,double8B

	full_text

double %932
Pstore8BE
C
	full_text6
4
2store double %933, double* %135, align 8, !tbaa !8
,double8B

	full_text

double %933
.double*8B

	full_text

double* %135
Bfmul8B8
6
	full_text)
'
%%934 = fmul double %928, 4.000000e-01
,double8B

	full_text

double %928
Cfsub8B9
7
	full_text*
(
&%935 = fsub double -0.000000e+00, %934
,double8B

	full_text

double %934
ucall8Bk
i
	full_text\
Z
X%936 = tail call double @llvm.fmuladd.f64(double %868, double 1.400000e+00, double %935)
,double8B

	full_text

double %868
,double8B

	full_text

double %935
:fmul8B0
.
	full_text!

%937 = fmul double %926, %936
,double8B

	full_text

double %926
,double8B

	full_text

double %936
Qstore8BF
D
	full_text7
5
3store double %937, double* %140, align 16, !tbaa !8
,double8B

	full_text

double %937
.double*8B

	full_text

double* %140
:fmul8B0
.
	full_text!

%938 = fmul double %925, %881
,double8B

	full_text

double %925
,double8B

	full_text

double %881
:fmul8B0
.
	full_text!

%939 = fmul double %925, %877
,double8B

	full_text

double %925
,double8B

	full_text

double %877
:fmul8B0
.
	full_text!

%940 = fmul double %925, %868
,double8B

	full_text

double %925
,double8B

	full_text

double %868
8sext8B.
,
	full_text

%941 = sext i32 %643 to i64
&i328B

	full_text


i32 %643
ègetelementptr8B|
z
	full_textm
k
i%942 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %941, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %941
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%943 = load double, double* %942, align 8, !tbaa !8
.double*8B

	full_text

double* %942
:fmul8B0
.
	full_text!

%944 = fmul double %943, %882
,double8B

	full_text

double %943
,double8B

	full_text

double %882
:fmul8B0
.
	full_text!

%945 = fmul double %943, %878
,double8B

	full_text

double %943
,double8B

	full_text

double %878
:fmul8B0
.
	full_text!

%946 = fmul double %943, %874
,double8B

	full_text

double %943
,double8B

	full_text

double %874
:fmul8B0
.
	full_text!

%947 = fmul double %943, %869
,double8B

	full_text

double %943
,double8B

	full_text

double %869
:fsub8B0
.
	full_text!

%948 = fsub double %938, %944
,double8B

	full_text

double %938
,double8B

	full_text

double %944
Bfmul8B8
6
	full_text)
'
%%949 = fmul double %948, 6.300000e+01
,double8B

	full_text

double %948
Pstore8BE
C
	full_text6
4
2store double %949, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %949
.double*8B

	full_text

double* %149
:fsub8B0
.
	full_text!

%950 = fsub double %939, %945
,double8B

	full_text

double %939
,double8B

	full_text

double %945
Bfmul8B8
6
	full_text)
'
%%951 = fmul double %950, 6.300000e+01
,double8B

	full_text

double %950
Pstore8BE
C
	full_text6
4
2store double %951, double* %152, align 8, !tbaa !8
,double8B

	full_text

double %951
.double*8B

	full_text

double* %152
:fsub8B0
.
	full_text!

%952 = fsub double %926, %946
,double8B

	full_text

double %926
,double8B

	full_text

double %946
Bfmul8B8
6
	full_text)
'
%%953 = fmul double %952, 8.400000e+01
,double8B

	full_text

double %952
Pstore8BE
C
	full_text6
4
2store double %953, double* %155, align 8, !tbaa !8
,double8B

	full_text

double %953
.double*8B

	full_text

double* %155
:fmul8B0
.
	full_text!

%954 = fmul double %939, %939
,double8B

	full_text

double %939
,double8B

	full_text

double %939
mcall8Bc
a
	full_textT
R
P%955 = tail call double @llvm.fmuladd.f64(double %938, double %938, double %954)
,double8B

	full_text

double %938
,double8B

	full_text

double %938
,double8B

	full_text

double %954
mcall8Bc
a
	full_textT
R
P%956 = tail call double @llvm.fmuladd.f64(double %926, double %926, double %955)
,double8B

	full_text

double %926
,double8B

	full_text

double %926
,double8B

	full_text

double %955
:fmul8B0
.
	full_text!

%957 = fmul double %945, %945
,double8B

	full_text

double %945
,double8B

	full_text

double %945
mcall8Bc
a
	full_textT
R
P%958 = tail call double @llvm.fmuladd.f64(double %944, double %944, double %957)
,double8B

	full_text

double %944
,double8B

	full_text

double %944
,double8B

	full_text

double %957
mcall8Bc
a
	full_textT
R
P%959 = tail call double @llvm.fmuladd.f64(double %946, double %946, double %958)
,double8B

	full_text

double %946
,double8B

	full_text

double %946
,double8B

	full_text

double %958
:fsub8B0
.
	full_text!

%960 = fsub double %956, %959
,double8B

	full_text

double %956
,double8B

	full_text

double %959
:fmul8B0
.
	full_text!

%961 = fmul double %946, %946
,double8B

	full_text

double %946
,double8B

	full_text

double %946
Cfsub8B9
7
	full_text*
(
&%962 = fsub double -0.000000e+00, %961
,double8B

	full_text

double %961
mcall8Bc
a
	full_textT
R
P%963 = tail call double @llvm.fmuladd.f64(double %926, double %926, double %962)
,double8B

	full_text

double %926
,double8B

	full_text

double %926
,double8B

	full_text

double %962
Bfmul8B8
6
	full_text)
'
%%964 = fmul double %963, 1.050000e+01
,double8B

	full_text

double %963
{call8Bq
o
	full_textb
`
^%965 = tail call double @llvm.fmuladd.f64(double %960, double 0xC03E3D70A3D70A3B, double %964)
,double8B

	full_text

double %960
,double8B

	full_text

double %964
:fsub8B0
.
	full_text!

%966 = fsub double %940, %947
,double8B

	full_text

double %940
,double8B

	full_text

double %947
{call8Bq
o
	full_textb
`
^%967 = tail call double @llvm.fmuladd.f64(double %966, double 0x405EDEB851EB851E, double %965)
,double8B

	full_text

double %966
,double8B

	full_text

double %965
Pstore8BE
C
	full_text6
4
2store double %967, double* %170, align 8, !tbaa !8
,double8B

	full_text

double %967
.double*8B

	full_text

double* %170
•getelementptr8Bë
é
	full_textÄ
~
|%968 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %941, i64 %30, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %941
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%969 = load double, double* %968, align 8, !tbaa !8
.double*8B

	full_text

double* %968
Qload8BG
E
	full_text8
6
4%970 = load double, double* %210, align 16, !tbaa !8
.double*8B

	full_text

double* %210
:fsub8B0
.
	full_text!

%971 = fsub double %872, %970
,double8B

	full_text

double %872
,double8B

	full_text

double %970
vcall8Bl
j
	full_text]
[
Y%972 = tail call double @llvm.fmuladd.f64(double %971, double -3.150000e+01, double %969)
,double8B

	full_text

double %971
,double8B

	full_text

double %969
•getelementptr8Bë
é
	full_textÄ
~
|%973 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %941, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %941
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%974 = load double, double* %973, align 8, !tbaa !8
.double*8B

	full_text

double* %973
Pload8BF
D
	full_text7
5
3%975 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
:fsub8B0
.
	full_text!

%976 = fsub double %929, %975
,double8B

	full_text

double %929
,double8B

	full_text

double %975
vcall8Bl
j
	full_text]
[
Y%977 = tail call double @llvm.fmuladd.f64(double %976, double -3.150000e+01, double %974)
,double8B

	full_text

double %976
,double8B

	full_text

double %974
•getelementptr8Bë
é
	full_textÄ
~
|%978 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %941, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %941
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%979 = load double, double* %978, align 8, !tbaa !8
.double*8B

	full_text

double* %978
Qload8BG
E
	full_text8
6
4%980 = load double, double* %245, align 16, !tbaa !8
.double*8B

	full_text

double* %245
:fsub8B0
.
	full_text!

%981 = fsub double %930, %980
,double8B

	full_text

double %930
,double8B

	full_text

double %980
vcall8Bl
j
	full_text]
[
Y%982 = tail call double @llvm.fmuladd.f64(double %981, double -3.150000e+01, double %979)
,double8B

	full_text

double %981
,double8B

	full_text

double %979
•getelementptr8Bë
é
	full_textÄ
~
|%983 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %941, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %941
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%984 = load double, double* %983, align 8, !tbaa !8
.double*8B

	full_text

double* %983
Pload8BF
D
	full_text7
5
3%985 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
:fsub8B0
.
	full_text!

%986 = fsub double %933, %985
,double8B

	full_text

double %933
,double8B

	full_text

double %985
vcall8Bl
j
	full_text]
[
Y%987 = tail call double @llvm.fmuladd.f64(double %986, double -3.150000e+01, double %984)
,double8B

	full_text

double %986
,double8B

	full_text

double %984
•getelementptr8Bë
é
	full_textÄ
~
|%988 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %941, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %941
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%989 = load double, double* %988, align 8, !tbaa !8
.double*8B

	full_text

double* %988
Qload8BG
E
	full_text8
6
4%990 = load double, double* %280, align 16, !tbaa !8
.double*8B

	full_text

double* %280
:fsub8B0
.
	full_text!

%991 = fsub double %937, %990
,double8B

	full_text

double %937
,double8B

	full_text

double %990
vcall8Bl
j
	full_text]
[
Y%992 = tail call double @llvm.fmuladd.f64(double %991, double -3.150000e+01, double %989)
,double8B

	full_text

double %991
,double8B

	full_text

double %989
Pload8BF
D
	full_text7
5
3%993 = load double, double* %198, align 8, !tbaa !8
.double*8B

	full_text

double* %198
Pload8BF
D
	full_text7
5
3%994 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%995 = tail call double @llvm.fmuladd.f64(double %994, double -2.000000e+00, double %993)
,double8B

	full_text

double %994
,double8B

	full_text

double %993
Oload8BE
C
	full_text6
4
2%996 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%997 = fadd double %995, %996
,double8B

	full_text

double %995
,double8B

	full_text

double %996
{call8Bq
o
	full_textb
`
^%998 = tail call double @llvm.fmuladd.f64(double %997, double 0x40AF020000000001, double %972)
,double8B

	full_text

double %997
,double8B

	full_text

double %972
Pload8BF
D
	full_text7
5
3%999 = load double, double* %234, align 8, !tbaa !8
.double*8B

	full_text

double* %234
;fsub8B1
/
	full_text"
 
%1000 = fsub double %949, %999
,double8B

	full_text

double %949
,double8B

	full_text

double %999
}call8Bs
q
	full_textd
b
`%1001 = tail call double @llvm.fmuladd.f64(double %1000, double 0x4019333333333334, double %977)
-double8B

	full_text

double %1000
,double8B

	full_text

double %977
Qload8BG
E
	full_text8
6
4%1002 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
xcall8Bn
l
	full_text_
]
[%1003 = tail call double @llvm.fmuladd.f64(double %882, double -2.000000e+00, double %1002)
,double8B

	full_text

double %882
-double8B

	full_text

double %1002
<fadd8B2
0
	full_text#
!
%1004 = fadd double %881, %1003
,double8B

	full_text

double %881
-double8B

	full_text

double %1003
~call8Bt
r
	full_texte
c
a%1005 = tail call double @llvm.fmuladd.f64(double %1004, double 0x40AF020000000001, double %1001)
-double8B

	full_text

double %1004
-double8B

	full_text

double %1001
Rload8BH
F
	full_text9
7
5%1006 = load double, double* %251, align 16, !tbaa !8
.double*8B

	full_text

double* %251
<fsub8B2
0
	full_text#
!
%1007 = fsub double %951, %1006
,double8B

	full_text

double %951
-double8B

	full_text

double %1006
}call8Bs
q
	full_textd
b
`%1008 = tail call double @llvm.fmuladd.f64(double %1007, double 0x4019333333333334, double %982)
-double8B

	full_text

double %1007
,double8B

	full_text

double %982
Qload8BG
E
	full_text8
6
4%1009 = load double, double* %236, align 8, !tbaa !8
.double*8B

	full_text

double* %236
xcall8Bn
l
	full_text_
]
[%1010 = tail call double @llvm.fmuladd.f64(double %878, double -2.000000e+00, double %1009)
,double8B

	full_text

double %878
-double8B

	full_text

double %1009
<fadd8B2
0
	full_text#
!
%1011 = fadd double %877, %1010
,double8B

	full_text

double %877
-double8B

	full_text

double %1010
~call8Bt
r
	full_texte
c
a%1012 = tail call double @llvm.fmuladd.f64(double %1011, double 0x40AF020000000001, double %1008)
-double8B

	full_text

double %1011
-double8B

	full_text

double %1008
Qload8BG
E
	full_text8
6
4%1013 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
<fsub8B2
0
	full_text#
!
%1014 = fsub double %953, %1013
,double8B

	full_text

double %953
-double8B

	full_text

double %1013
}call8Bs
q
	full_textd
b
`%1015 = tail call double @llvm.fmuladd.f64(double %1014, double 0x4019333333333334, double %987)
-double8B

	full_text

double %1014
,double8B

	full_text

double %987
Qload8BG
E
	full_text8
6
4%1016 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
xcall8Bn
l
	full_text_
]
[%1017 = tail call double @llvm.fmuladd.f64(double %874, double -2.000000e+00, double %1016)
,double8B

	full_text

double %874
-double8B

	full_text

double %1016
<fadd8B2
0
	full_text#
!
%1018 = fadd double %872, %1017
,double8B

	full_text

double %872
-double8B

	full_text

double %1017
~call8Bt
r
	full_texte
c
a%1019 = tail call double @llvm.fmuladd.f64(double %1018, double 0x40AF020000000001, double %1015)
-double8B

	full_text

double %1018
-double8B

	full_text

double %1015
Rload8BH
F
	full_text9
7
5%1020 = load double, double* %286, align 16, !tbaa !8
.double*8B

	full_text

double* %286
<fsub8B2
0
	full_text#
!
%1021 = fsub double %967, %1020
,double8B

	full_text

double %967
-double8B

	full_text

double %1020
}call8Bs
q
	full_textd
b
`%1022 = tail call double @llvm.fmuladd.f64(double %1021, double 0x4019333333333334, double %992)
-double8B

	full_text

double %1021
,double8B

	full_text

double %992
Qload8BG
E
	full_text8
6
4%1023 = load double, double* %270, align 8, !tbaa !8
.double*8B

	full_text

double* %270
xcall8Bn
l
	full_text_
]
[%1024 = tail call double @llvm.fmuladd.f64(double %869, double -2.000000e+00, double %1023)
,double8B

	full_text

double %869
-double8B

	full_text

double %1023
<fadd8B2
0
	full_text#
!
%1025 = fadd double %868, %1024
,double8B

	full_text

double %868
-double8B

	full_text

double %1024
~call8Bt
r
	full_texte
c
a%1026 = tail call double @llvm.fmuladd.f64(double %1025, double 0x40AF020000000001, double %1022)
-double8B

	full_text

double %1025
-double8B

	full_text

double %1022
Rload8BH
F
	full_text9
7
5%1027 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
xcall8Bn
l
	full_text_
]
[%1028 = tail call double @llvm.fmuladd.f64(double %993, double -4.000000e+00, double %1027)
,double8B

	full_text

double %993
-double8B

	full_text

double %1027
wcall8Bm
k
	full_text^
\
Z%1029 = tail call double @llvm.fmuladd.f64(double %994, double 6.000000e+00, double %1028)
,double8B

	full_text

double %994
-double8B

	full_text

double %1028
xcall8Bn
l
	full_text_
]
[%1030 = tail call double @llvm.fmuladd.f64(double %996, double -4.000000e+00, double %1029)
,double8B

	full_text

double %996
-double8B

	full_text

double %1029
ocall8Be
c
	full_textV
T
R%1031 = tail call double @llvm.fmuladd.f64(double %422, double %1030, double %998)
,double8B

	full_text

double %422
-double8B

	full_text

double %1030
,double8B

	full_text

double %998
Qstore8BF
D
	full_text7
5
3store double %1031, double* %968, align 8, !tbaa !8
-double8B

	full_text

double %1031
.double*8B

	full_text

double* %968
Qload8BG
E
	full_text8
6
4%1032 = load double, double* %222, align 8, !tbaa !8
.double*8B

	full_text

double* %222
ycall8Bo
m
	full_text`
^
\%1033 = tail call double @llvm.fmuladd.f64(double %1002, double -4.000000e+00, double %1032)
-double8B

	full_text

double %1002
-double8B

	full_text

double %1032
Pload8BF
D
	full_text7
5
3%1034 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
xcall8Bn
l
	full_text_
]
[%1035 = tail call double @llvm.fmuladd.f64(double %1034, double 6.000000e+00, double %1033)
-double8B

	full_text

double %1034
-double8B

	full_text

double %1033
Pload8BF
D
	full_text7
5
3%1036 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
ycall8Bo
m
	full_text`
^
\%1037 = tail call double @llvm.fmuladd.f64(double %1036, double -4.000000e+00, double %1035)
-double8B

	full_text

double %1036
-double8B

	full_text

double %1035
pcall8Bf
d
	full_textW
U
S%1038 = tail call double @llvm.fmuladd.f64(double %422, double %1037, double %1005)
,double8B

	full_text

double %422
-double8B

	full_text

double %1037
-double8B

	full_text

double %1005
Qstore8BF
D
	full_text7
5
3store double %1038, double* %973, align 8, !tbaa !8
-double8B

	full_text

double %1038
.double*8B

	full_text

double* %973
Rload8BH
F
	full_text9
7
5%1039 = load double, double* %239, align 16, !tbaa !8
.double*8B

	full_text

double* %239
ycall8Bo
m
	full_text`
^
\%1040 = tail call double @llvm.fmuladd.f64(double %1009, double -4.000000e+00, double %1039)
-double8B

	full_text

double %1009
-double8B

	full_text

double %1039
Qload8BG
E
	full_text8
6
4%1041 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
xcall8Bn
l
	full_text_
]
[%1042 = tail call double @llvm.fmuladd.f64(double %1041, double 6.000000e+00, double %1040)
-double8B

	full_text

double %1041
-double8B

	full_text

double %1040
Pload8BF
D
	full_text7
5
3%1043 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
ycall8Bo
m
	full_text`
^
\%1044 = tail call double @llvm.fmuladd.f64(double %1043, double -4.000000e+00, double %1042)
-double8B

	full_text

double %1043
-double8B

	full_text

double %1042
pcall8Bf
d
	full_textW
U
S%1045 = tail call double @llvm.fmuladd.f64(double %422, double %1044, double %1012)
,double8B

	full_text

double %422
-double8B

	full_text

double %1044
-double8B

	full_text

double %1012
Qstore8BF
D
	full_text7
5
3store double %1045, double* %978, align 8, !tbaa !8
-double8B

	full_text

double %1045
.double*8B

	full_text

double* %978
Qload8BG
E
	full_text8
6
4%1046 = load double, double* %256, align 8, !tbaa !8
.double*8B

	full_text

double* %256
ycall8Bo
m
	full_text`
^
\%1047 = tail call double @llvm.fmuladd.f64(double %1016, double -4.000000e+00, double %1046)
-double8B

	full_text

double %1016
-double8B

	full_text

double %1046
Pload8BF
D
	full_text7
5
3%1048 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
xcall8Bn
l
	full_text_
]
[%1049 = tail call double @llvm.fmuladd.f64(double %1048, double 6.000000e+00, double %1047)
-double8B

	full_text

double %1048
-double8B

	full_text

double %1047
Pload8BF
D
	full_text7
5
3%1050 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
ycall8Bo
m
	full_text`
^
\%1051 = tail call double @llvm.fmuladd.f64(double %1050, double -4.000000e+00, double %1049)
-double8B

	full_text

double %1050
-double8B

	full_text

double %1049
pcall8Bf
d
	full_textW
U
S%1052 = tail call double @llvm.fmuladd.f64(double %422, double %1051, double %1019)
,double8B

	full_text

double %422
-double8B

	full_text

double %1051
-double8B

	full_text

double %1019
Qstore8BF
D
	full_text7
5
3store double %1052, double* %983, align 8, !tbaa !8
-double8B

	full_text

double %1052
.double*8B

	full_text

double* %983
Rload8BH
F
	full_text9
7
5%1053 = load double, double* %273, align 16, !tbaa !8
.double*8B

	full_text

double* %273
ycall8Bo
m
	full_text`
^
\%1054 = tail call double @llvm.fmuladd.f64(double %1023, double -4.000000e+00, double %1053)
-double8B

	full_text

double %1023
-double8B

	full_text

double %1053
Qload8BG
E
	full_text8
6
4%1055 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
xcall8Bn
l
	full_text_
]
[%1056 = tail call double @llvm.fmuladd.f64(double %1055, double 6.000000e+00, double %1054)
-double8B

	full_text

double %1055
-double8B

	full_text

double %1054
Pload8BF
D
	full_text7
5
3%1057 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
ycall8Bo
m
	full_text`
^
\%1058 = tail call double @llvm.fmuladd.f64(double %1057, double -4.000000e+00, double %1056)
-double8B

	full_text

double %1057
-double8B

	full_text

double %1056
pcall8Bf
d
	full_textW
U
S%1059 = tail call double @llvm.fmuladd.f64(double %422, double %1058, double %1026)
,double8B

	full_text

double %422
-double8B

	full_text

double %1058
-double8B

	full_text

double %1026
Qstore8BF
D
	full_text7
5
3store double %1059, double* %988, align 8, !tbaa !8
-double8B

	full_text

double %1059
.double*8B

	full_text

double* %988
Ögetelementptr8Br
p
	full_textc
a
_%1060 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Rstore8BG
E
	full_text8
6
4store double %993, double* %1060, align 16, !tbaa !8
,double8B

	full_text

double %993
/double*8B 

	full_text

double* %1060
Pstore8BE
C
	full_text6
4
2store double %994, double* %198, align 8, !tbaa !8
,double8B

	full_text

double %994
.double*8B

	full_text

double* %198
Pstore8BE
C
	full_text6
4
2store double %996, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %996
-double*8B

	full_text

double* %84
Lload8BB
@
	full_text3
1
/%1061 = load i64, i64* %206, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %206
Jstore8B?
=
	full_text0
.
,store i64 %1061, i64* %83, align 8, !tbaa !8
'i648B

	full_text

	i64 %1061
'i64*8B

	full_text


i64* %83
Kload8BA
?
	full_text2
0
.%1062 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Lstore8BA
?
	full_text2
0
.store i64 %1062, i64* %211, align 16, !tbaa !8
'i648B

	full_text

	i64 %1062
(i64*8B

	full_text

	i64* %211
Lload8BB
@
	full_text3
1
/%1063 = load i64, i64* %213, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %213
Kstore8B@
>
	full_text1
/
-store i64 %1063, i64* %208, align 8, !tbaa !8
'i648B

	full_text

	i64 %1063
(i64*8B

	full_text

	i64* %208
Kload8BA
?
	full_text2
0
.%1064 = load i64, i64* %216, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %216
Lstore8BA
?
	full_text2
0
.store i64 %1064, i64* %218, align 16, !tbaa !8
'i648B

	full_text

	i64 %1064
(i64*8B

	full_text

	i64* %218
Qstore8BF
D
	full_text7
5
3store double %1002, double* %222, align 8, !tbaa !8
-double8B

	full_text

double %1002
.double*8B

	full_text

double* %222
Qstore8BF
D
	full_text7
5
3store double %1034, double* %219, align 8, !tbaa !8
-double8B

	full_text

double %1034
.double*8B

	full_text

double* %219
Pstore8BE
C
	full_text6
4
2store double %1036, double* %86, align 8, !tbaa !8
-double8B

	full_text

double %1036
-double*8B

	full_text

double* %86
Kload8BA
?
	full_text2
0
.%1065 = load i64, i64* %182, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %182
Jstore8B?
=
	full_text0
.
,store i64 %1065, i64* %42, align 8, !tbaa !8
'i648B

	full_text

	i64 %1065
'i64*8B

	full_text


i64* %42
Kload8BA
?
	full_text2
0
.%1066 = load i64, i64* %226, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %226
Kstore8B@
>
	full_text1
/
-store i64 %1066, i64* %229, align 8, !tbaa !8
'i648B

	full_text

	i64 %1066
(i64*8B

	full_text

	i64* %229
Kload8BA
?
	full_text2
0
.%1067 = load i64, i64* %230, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %230
Kstore8B@
>
	full_text1
/
-store i64 %1067, i64* %226, align 8, !tbaa !8
'i648B

	full_text

	i64 %1067
(i64*8B

	full_text

	i64* %226
Kload8BA
?
	full_text2
0
.%1068 = load i64, i64* %232, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %232
Kstore8B@
>
	full_text1
/
-store i64 %1068, i64* %235, align 8, !tbaa !8
'i648B

	full_text

	i64 %1068
(i64*8B

	full_text

	i64* %235
Rstore8BG
E
	full_text8
6
4store double %1009, double* %239, align 16, !tbaa !8
-double8B

	full_text

double %1009
.double*8B

	full_text

double* %239
Qstore8BF
D
	full_text7
5
3store double %1041, double* %236, align 8, !tbaa !8
-double8B

	full_text

double %1041
.double*8B

	full_text

double* %236
Qstore8BF
D
	full_text7
5
3store double %1043, double* %88, align 16, !tbaa !8
-double8B

	full_text

double %1043
-double*8B

	full_text

double* %88
Lload8BB
@
	full_text3
1
/%1069 = load i64, i64* %187, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %187
Jstore8B?
=
	full_text0
.
,store i64 %1069, i64* %47, align 8, !tbaa !8
'i648B

	full_text

	i64 %1069
'i64*8B

	full_text


i64* %47
Kload8BA
?
	full_text2
0
.%1070 = load i64, i64* %243, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %243
Lstore8BA
?
	full_text2
0
.store i64 %1070, i64* %246, align 16, !tbaa !8
'i648B

	full_text

	i64 %1070
(i64*8B

	full_text

	i64* %246
Lload8BB
@
	full_text3
1
/%1071 = load i64, i64* %247, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %247
Kstore8B@
>
	full_text1
/
-store i64 %1071, i64* %243, align 8, !tbaa !8
'i648B

	full_text

	i64 %1071
(i64*8B

	full_text

	i64* %243
Kload8BA
?
	full_text2
0
.%1072 = load i64, i64* %249, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %249
Lstore8BA
?
	full_text2
0
.store i64 %1072, i64* %252, align 16, !tbaa !8
'i648B

	full_text

	i64 %1072
(i64*8B

	full_text

	i64* %252
Qstore8BF
D
	full_text7
5
3store double %1016, double* %256, align 8, !tbaa !8
-double8B

	full_text

double %1016
.double*8B

	full_text

double* %256
Qstore8BF
D
	full_text7
5
3store double %1048, double* %253, align 8, !tbaa !8
-double8B

	full_text

double %1048
.double*8B

	full_text

double* %253
Pstore8BE
C
	full_text6
4
2store double %1050, double* %90, align 8, !tbaa !8
-double8B

	full_text

double %1050
-double*8B

	full_text

double* %90
Kload8BA
?
	full_text2
0
.%1073 = load i64, i64* %192, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %192
Jstore8B?
=
	full_text0
.
,store i64 %1073, i64* %52, align 8, !tbaa !8
'i648B

	full_text

	i64 %1073
'i64*8B

	full_text


i64* %52
Kload8BA
?
	full_text2
0
.%1074 = load i64, i64* %260, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %260
Kstore8B@
>
	full_text1
/
-store i64 %1074, i64* %263, align 8, !tbaa !8
'i648B

	full_text

	i64 %1074
(i64*8B

	full_text

	i64* %263
Kload8BA
?
	full_text2
0
.%1075 = load i64, i64* %264, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %264
Kstore8B@
>
	full_text1
/
-store i64 %1075, i64* %260, align 8, !tbaa !8
'i648B

	full_text

	i64 %1075
(i64*8B

	full_text

	i64* %260
Kload8BA
?
	full_text2
0
.%1076 = load i64, i64* %266, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %266
Kstore8B@
>
	full_text1
/
-store i64 %1076, i64* %269, align 8, !tbaa !8
'i648B

	full_text

	i64 %1076
(i64*8B

	full_text

	i64* %269
Rstore8BG
E
	full_text8
6
4store double %1023, double* %273, align 16, !tbaa !8
-double8B

	full_text

double %1023
.double*8B

	full_text

double* %273
Qstore8BF
D
	full_text7
5
3store double %1055, double* %270, align 8, !tbaa !8
-double8B

	full_text

double %1055
.double*8B

	full_text

double* %270
Qstore8BF
D
	full_text7
5
3store double %1057, double* %92, align 16, !tbaa !8
-double8B

	full_text

double %1057
-double*8B

	full_text

double* %92
Lload8BB
@
	full_text3
1
/%1077 = load i64, i64* %197, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %197
Jstore8B?
=
	full_text0
.
,store i64 %1077, i64* %57, align 8, !tbaa !8
'i648B

	full_text

	i64 %1077
'i64*8B

	full_text


i64* %57
Kload8BA
?
	full_text2
0
.%1078 = load i64, i64* %278, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %278
Lstore8BA
?
	full_text2
0
.store i64 %1078, i64* %281, align 16, !tbaa !8
'i648B

	full_text

	i64 %1078
(i64*8B

	full_text

	i64* %281
Lload8BB
@
	full_text3
1
/%1079 = load i64, i64* %282, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %282
Kstore8B@
>
	full_text1
/
-store i64 %1079, i64* %278, align 8, !tbaa !8
'i648B

	full_text

	i64 %1079
(i64*8B

	full_text

	i64* %278
Kload8BA
?
	full_text2
0
.%1080 = load i64, i64* %284, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %284
Lstore8BA
?
	full_text2
0
.store i64 %1080, i64* %287, align 16, !tbaa !8
'i648B

	full_text

	i64 %1080
(i64*8B

	full_text

	i64* %287
Lstore8BA
?
	full_text2
0
.store i64 %1073, i64* %114, align 16, !tbaa !8
'i648B

	full_text

	i64 %1073
(i64*8B

	full_text

	i64* %114
Cbitcast8B6
4
	full_text'
%
#%1081 = bitcast i64 %1073 to double
'i648B

	full_text

	i64 %1073
êgetelementptr8B}
{
	full_textn
l
j%1082 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %906, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1083 = load double, double* %1082, align 8, !tbaa !8
/double*8B 

	full_text

double* %1082
=fmul8B3
1
	full_text$
"
 %1084 = fmul double %1083, %1081
-double8B

	full_text

double %1083
-double8B

	full_text

double %1081
êgetelementptr8B}
{
	full_textn
l
j%1085 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %906, i64 %30, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
&i648B

	full_text


i64 %906
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1086 = load double, double* %1085, align 8, !tbaa !8
/double*8B 

	full_text

double* %1085
Cbitcast8B6
4
	full_text'
%
#%1087 = bitcast i64 %1065 to double
'i648B

	full_text

	i64 %1065
=fmul8B3
1
	full_text$
"
 %1088 = fmul double %1084, %1087
-double8B

	full_text

double %1084
-double8B

	full_text

double %1087
Qstore8BF
D
	full_text7
5
3store double %1088, double* %127, align 8, !tbaa !8
-double8B

	full_text

double %1088
.double*8B

	full_text

double* %127
Cbitcast8B6
4
	full_text'
%
#%1089 = bitcast i64 %1069 to double
'i648B

	full_text

	i64 %1069
=fmul8B3
1
	full_text$
"
 %1090 = fmul double %1084, %1089
-double8B

	full_text

double %1084
-double8B

	full_text

double %1089
Rstore8BG
E
	full_text8
6
4store double %1090, double* %130, align 16, !tbaa !8
-double8B

	full_text

double %1090
.double*8B

	full_text

double* %130
Cbitcast8B6
4
	full_text'
%
#%1091 = bitcast i64 %1077 to double
'i648B

	full_text

	i64 %1077
=fsub8B3
1
	full_text$
"
 %1092 = fsub double %1091, %1086
-double8B

	full_text

double %1091
-double8B

	full_text

double %1086
Dfmul8B:
8
	full_text+
)
'%1093 = fmul double %1092, 4.000000e-01
-double8B

	full_text

double %1092
qcall8Bg
e
	full_textX
V
T%1094 = tail call double @llvm.fmuladd.f64(double %1081, double %1084, double %1093)
-double8B

	full_text

double %1081
-double8B

	full_text

double %1084
-double8B

	full_text

double %1093
Qstore8BF
D
	full_text7
5
3store double %1094, double* %135, align 8, !tbaa !8
-double8B

	full_text

double %1094
.double*8B

	full_text

double* %135
Dfmul8B:
8
	full_text+
)
'%1095 = fmul double %1086, 4.000000e-01
-double8B

	full_text

double %1086
Efsub8B;
9
	full_text,
*
(%1096 = fsub double -0.000000e+00, %1095
-double8B

	full_text

double %1095
xcall8Bn
l
	full_text_
]
[%1097 = tail call double @llvm.fmuladd.f64(double %1091, double 1.400000e+00, double %1096)
-double8B

	full_text

double %1091
-double8B

	full_text

double %1096
=fmul8B3
1
	full_text$
"
 %1098 = fmul double %1084, %1097
-double8B

	full_text

double %1084
-double8B

	full_text

double %1097
Rstore8BG
E
	full_text8
6
4store double %1098, double* %140, align 16, !tbaa !8
-double8B

	full_text

double %1098
.double*8B

	full_text

double* %140
=fmul8B3
1
	full_text$
"
 %1099 = fmul double %1083, %1087
-double8B

	full_text

double %1083
-double8B

	full_text

double %1087
=fmul8B3
1
	full_text$
"
 %1100 = fmul double %1083, %1089
-double8B

	full_text

double %1083
-double8B

	full_text

double %1089
=fmul8B3
1
	full_text$
"
 %1101 = fmul double %1083, %1091
-double8B

	full_text

double %1083
-double8B

	full_text

double %1091
Qload8BG
E
	full_text8
6
4%1102 = load double, double* %924, align 8, !tbaa !8
.double*8B

	full_text

double* %924
=fmul8B3
1
	full_text$
"
 %1103 = fmul double %1102, %1036
-double8B

	full_text

double %1102
-double8B

	full_text

double %1036
=fmul8B3
1
	full_text$
"
 %1104 = fmul double %1102, %1043
-double8B

	full_text

double %1102
-double8B

	full_text

double %1043
=fmul8B3
1
	full_text$
"
 %1105 = fmul double %1102, %1050
-double8B

	full_text

double %1102
-double8B

	full_text

double %1050
=fmul8B3
1
	full_text$
"
 %1106 = fmul double %1102, %1057
-double8B

	full_text

double %1102
-double8B

	full_text

double %1057
=fsub8B3
1
	full_text$
"
 %1107 = fsub double %1099, %1103
-double8B

	full_text

double %1099
-double8B

	full_text

double %1103
Dfmul8B:
8
	full_text+
)
'%1108 = fmul double %1107, 6.300000e+01
-double8B

	full_text

double %1107
Qstore8BF
D
	full_text7
5
3store double %1108, double* %149, align 8, !tbaa !8
-double8B

	full_text

double %1108
.double*8B

	full_text

double* %149
=fsub8B3
1
	full_text$
"
 %1109 = fsub double %1100, %1104
-double8B

	full_text

double %1100
-double8B

	full_text

double %1104
Dfmul8B:
8
	full_text+
)
'%1110 = fmul double %1109, 6.300000e+01
-double8B

	full_text

double %1109
Qstore8BF
D
	full_text7
5
3store double %1110, double* %152, align 8, !tbaa !8
-double8B

	full_text

double %1110
.double*8B

	full_text

double* %152
=fsub8B3
1
	full_text$
"
 %1111 = fsub double %1084, %1105
-double8B

	full_text

double %1084
-double8B

	full_text

double %1105
Dfmul8B:
8
	full_text+
)
'%1112 = fmul double %1111, 8.400000e+01
-double8B

	full_text

double %1111
Qstore8BF
D
	full_text7
5
3store double %1112, double* %155, align 8, !tbaa !8
-double8B

	full_text

double %1112
.double*8B

	full_text

double* %155
=fmul8B3
1
	full_text$
"
 %1113 = fmul double %1100, %1100
-double8B

	full_text

double %1100
-double8B

	full_text

double %1100
qcall8Bg
e
	full_textX
V
T%1114 = tail call double @llvm.fmuladd.f64(double %1099, double %1099, double %1113)
-double8B

	full_text

double %1099
-double8B

	full_text

double %1099
-double8B

	full_text

double %1113
qcall8Bg
e
	full_textX
V
T%1115 = tail call double @llvm.fmuladd.f64(double %1084, double %1084, double %1114)
-double8B

	full_text

double %1084
-double8B

	full_text

double %1084
-double8B

	full_text

double %1114
=fmul8B3
1
	full_text$
"
 %1116 = fmul double %1104, %1104
-double8B

	full_text

double %1104
-double8B

	full_text

double %1104
qcall8Bg
e
	full_textX
V
T%1117 = tail call double @llvm.fmuladd.f64(double %1103, double %1103, double %1116)
-double8B

	full_text

double %1103
-double8B

	full_text

double %1103
-double8B

	full_text

double %1116
qcall8Bg
e
	full_textX
V
T%1118 = tail call double @llvm.fmuladd.f64(double %1105, double %1105, double %1117)
-double8B

	full_text

double %1105
-double8B

	full_text

double %1105
-double8B

	full_text

double %1117
=fsub8B3
1
	full_text$
"
 %1119 = fsub double %1115, %1118
-double8B

	full_text

double %1115
-double8B

	full_text

double %1118
=fmul8B3
1
	full_text$
"
 %1120 = fmul double %1105, %1105
-double8B

	full_text

double %1105
-double8B

	full_text

double %1105
Efsub8B;
9
	full_text,
*
(%1121 = fsub double -0.000000e+00, %1120
-double8B

	full_text

double %1120
qcall8Bg
e
	full_textX
V
T%1122 = tail call double @llvm.fmuladd.f64(double %1084, double %1084, double %1121)
-double8B

	full_text

double %1084
-double8B

	full_text

double %1084
-double8B

	full_text

double %1121
Dfmul8B:
8
	full_text+
)
'%1123 = fmul double %1122, 1.050000e+01
-double8B

	full_text

double %1122
~call8Bt
r
	full_texte
c
a%1124 = tail call double @llvm.fmuladd.f64(double %1119, double 0xC03E3D70A3D70A3B, double %1123)
-double8B

	full_text

double %1119
-double8B

	full_text

double %1123
=fsub8B3
1
	full_text$
"
 %1125 = fsub double %1101, %1106
-double8B

	full_text

double %1101
-double8B

	full_text

double %1106
~call8Bt
r
	full_texte
c
a%1126 = tail call double @llvm.fmuladd.f64(double %1125, double 0x405EDEB851EB851E, double %1124)
-double8B

	full_text

double %1125
-double8B

	full_text

double %1124
Qstore8BF
D
	full_text7
5
3store double %1126, double* %170, align 8, !tbaa !8
-double8B

	full_text

double %1126
.double*8B

	full_text

double* %170
¶getelementptr8Bí
è
	full_textÅ

}%1127 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %923, i64 %30, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1128 = load double, double* %1127, align 8, !tbaa !8
/double*8B 

	full_text

double* %1127
Rload8BH
F
	full_text9
7
5%1129 = load double, double* %210, align 16, !tbaa !8
.double*8B

	full_text

double* %210
=fsub8B3
1
	full_text$
"
 %1130 = fsub double %1081, %1129
-double8B

	full_text

double %1081
-double8B

	full_text

double %1129
ycall8Bo
m
	full_text`
^
\%1131 = tail call double @llvm.fmuladd.f64(double %1130, double -3.150000e+01, double %1128)
-double8B

	full_text

double %1130
-double8B

	full_text

double %1128
¶getelementptr8Bí
è
	full_textÅ

}%1132 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %923, i64 %30, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1133 = load double, double* %1132, align 8, !tbaa !8
/double*8B 

	full_text

double* %1132
Qload8BG
E
	full_text8
6
4%1134 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
=fsub8B3
1
	full_text$
"
 %1135 = fsub double %1088, %1134
-double8B

	full_text

double %1088
-double8B

	full_text

double %1134
ycall8Bo
m
	full_text`
^
\%1136 = tail call double @llvm.fmuladd.f64(double %1135, double -3.150000e+01, double %1133)
-double8B

	full_text

double %1135
-double8B

	full_text

double %1133
¶getelementptr8Bí
è
	full_textÅ

}%1137 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %923, i64 %30, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1138 = load double, double* %1137, align 8, !tbaa !8
/double*8B 

	full_text

double* %1137
Rload8BH
F
	full_text9
7
5%1139 = load double, double* %245, align 16, !tbaa !8
.double*8B

	full_text

double* %245
=fsub8B3
1
	full_text$
"
 %1140 = fsub double %1090, %1139
-double8B

	full_text

double %1090
-double8B

	full_text

double %1139
ycall8Bo
m
	full_text`
^
\%1141 = tail call double @llvm.fmuladd.f64(double %1140, double -3.150000e+01, double %1138)
-double8B

	full_text

double %1140
-double8B

	full_text

double %1138
¶getelementptr8Bí
è
	full_textÅ

}%1142 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %923, i64 %30, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1143 = load double, double* %1142, align 8, !tbaa !8
/double*8B 

	full_text

double* %1142
Qload8BG
E
	full_text8
6
4%1144 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
=fsub8B3
1
	full_text$
"
 %1145 = fsub double %1094, %1144
-double8B

	full_text

double %1094
-double8B

	full_text

double %1144
ycall8Bo
m
	full_text`
^
\%1146 = tail call double @llvm.fmuladd.f64(double %1145, double -3.150000e+01, double %1143)
-double8B

	full_text

double %1145
-double8B

	full_text

double %1143
¶getelementptr8Bí
è
	full_textÅ

}%1147 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %923, i64 %30, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
&i648B

	full_text


i64 %923
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1148 = load double, double* %1147, align 8, !tbaa !8
/double*8B 

	full_text

double* %1147
Cbitcast8B6
4
	full_text'
%
#%1149 = bitcast i64 %1078 to double
'i648B

	full_text

	i64 %1078
=fsub8B3
1
	full_text$
"
 %1150 = fsub double %1098, %1149
-double8B

	full_text

double %1098
-double8B

	full_text

double %1149
ycall8Bo
m
	full_text`
^
\%1151 = tail call double @llvm.fmuladd.f64(double %1150, double -3.150000e+01, double %1148)
-double8B

	full_text

double %1150
-double8B

	full_text

double %1148
Qload8BG
E
	full_text8
6
4%1152 = load double, double* %198, align 8, !tbaa !8
.double*8B

	full_text

double* %198
Qload8BG
E
	full_text8
6
4%1153 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
ycall8Bo
m
	full_text`
^
\%1154 = tail call double @llvm.fmuladd.f64(double %1153, double -2.000000e+00, double %1152)
-double8B

	full_text

double %1153
-double8B

	full_text

double %1152
Pload8BF
D
	full_text7
5
3%1155 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
=fadd8B3
1
	full_text$
"
 %1156 = fadd double %1154, %1155
-double8B

	full_text

double %1154
-double8B

	full_text

double %1155
~call8Bt
r
	full_texte
c
a%1157 = tail call double @llvm.fmuladd.f64(double %1156, double 0x40AF020000000001, double %1131)
-double8B

	full_text

double %1156
-double8B

	full_text

double %1131
Qload8BG
E
	full_text8
6
4%1158 = load double, double* %234, align 8, !tbaa !8
.double*8B

	full_text

double* %234
=fsub8B3
1
	full_text$
"
 %1159 = fsub double %1108, %1158
-double8B

	full_text

double %1108
-double8B

	full_text

double %1158
~call8Bt
r
	full_texte
c
a%1160 = tail call double @llvm.fmuladd.f64(double %1159, double 0x4019333333333334, double %1136)
-double8B

	full_text

double %1159
-double8B

	full_text

double %1136
Qload8BG
E
	full_text8
6
4%1161 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
ycall8Bo
m
	full_text`
^
\%1162 = tail call double @llvm.fmuladd.f64(double %1036, double -2.000000e+00, double %1161)
-double8B

	full_text

double %1036
-double8B

	full_text

double %1161
=fadd8B3
1
	full_text$
"
 %1163 = fadd double %1162, %1087
-double8B

	full_text

double %1162
-double8B

	full_text

double %1087
~call8Bt
r
	full_texte
c
a%1164 = tail call double @llvm.fmuladd.f64(double %1163, double 0x40AF020000000001, double %1160)
-double8B

	full_text

double %1163
-double8B

	full_text

double %1160
Rload8BH
F
	full_text9
7
5%1165 = load double, double* %251, align 16, !tbaa !8
.double*8B

	full_text

double* %251
=fsub8B3
1
	full_text$
"
 %1166 = fsub double %1110, %1165
-double8B

	full_text

double %1110
-double8B

	full_text

double %1165
~call8Bt
r
	full_texte
c
a%1167 = tail call double @llvm.fmuladd.f64(double %1166, double 0x4019333333333334, double %1141)
-double8B

	full_text

double %1166
-double8B

	full_text

double %1141
Qload8BG
E
	full_text8
6
4%1168 = load double, double* %236, align 8, !tbaa !8
.double*8B

	full_text

double* %236
ycall8Bo
m
	full_text`
^
\%1169 = tail call double @llvm.fmuladd.f64(double %1043, double -2.000000e+00, double %1168)
-double8B

	full_text

double %1043
-double8B

	full_text

double %1168
=fadd8B3
1
	full_text$
"
 %1170 = fadd double %1169, %1089
-double8B

	full_text

double %1169
-double8B

	full_text

double %1089
~call8Bt
r
	full_texte
c
a%1171 = tail call double @llvm.fmuladd.f64(double %1170, double 0x40AF020000000001, double %1167)
-double8B

	full_text

double %1170
-double8B

	full_text

double %1167
Qload8BG
E
	full_text8
6
4%1172 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
=fsub8B3
1
	full_text$
"
 %1173 = fsub double %1112, %1172
-double8B

	full_text

double %1112
-double8B

	full_text

double %1172
~call8Bt
r
	full_texte
c
a%1174 = tail call double @llvm.fmuladd.f64(double %1173, double 0x4019333333333334, double %1146)
-double8B

	full_text

double %1173
-double8B

	full_text

double %1146
Qload8BG
E
	full_text8
6
4%1175 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
ycall8Bo
m
	full_text`
^
\%1176 = tail call double @llvm.fmuladd.f64(double %1050, double -2.000000e+00, double %1175)
-double8B

	full_text

double %1050
-double8B

	full_text

double %1175
=fadd8B3
1
	full_text$
"
 %1177 = fadd double %1176, %1081
-double8B

	full_text

double %1176
-double8B

	full_text

double %1081
~call8Bt
r
	full_texte
c
a%1178 = tail call double @llvm.fmuladd.f64(double %1177, double 0x40AF020000000001, double %1174)
-double8B

	full_text

double %1177
-double8B

	full_text

double %1174
Rload8BH
F
	full_text9
7
5%1179 = load double, double* %286, align 16, !tbaa !8
.double*8B

	full_text

double* %286
=fsub8B3
1
	full_text$
"
 %1180 = fsub double %1126, %1179
-double8B

	full_text

double %1126
-double8B

	full_text

double %1179
~call8Bt
r
	full_texte
c
a%1181 = tail call double @llvm.fmuladd.f64(double %1180, double 0x4019333333333334, double %1151)
-double8B

	full_text

double %1180
-double8B

	full_text

double %1151
Qload8BG
E
	full_text8
6
4%1182 = load double, double* %270, align 8, !tbaa !8
.double*8B

	full_text

double* %270
ycall8Bo
m
	full_text`
^
\%1183 = tail call double @llvm.fmuladd.f64(double %1057, double -2.000000e+00, double %1182)
-double8B

	full_text

double %1057
-double8B

	full_text

double %1182
=fadd8B3
1
	full_text$
"
 %1184 = fadd double %1183, %1091
-double8B

	full_text

double %1183
-double8B

	full_text

double %1091
~call8Bt
r
	full_texte
c
a%1185 = tail call double @llvm.fmuladd.f64(double %1184, double 0x40AF020000000001, double %1181)
-double8B

	full_text

double %1184
-double8B

	full_text

double %1181
Rload8BH
F
	full_text9
7
5%1186 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
ycall8Bo
m
	full_text`
^
\%1187 = tail call double @llvm.fmuladd.f64(double %1152, double -4.000000e+00, double %1186)
-double8B

	full_text

double %1152
-double8B

	full_text

double %1186
xcall8Bn
l
	full_text_
]
[%1188 = tail call double @llvm.fmuladd.f64(double %1153, double 5.000000e+00, double %1187)
-double8B

	full_text

double %1153
-double8B

	full_text

double %1187
pcall8Bf
d
	full_textW
U
S%1189 = tail call double @llvm.fmuladd.f64(double %422, double %1188, double %1157)
,double8B

	full_text

double %422
-double8B

	full_text

double %1188
-double8B

	full_text

double %1157
Rstore8BG
E
	full_text8
6
4store double %1189, double* %1127, align 8, !tbaa !8
-double8B

	full_text

double %1189
/double*8B 

	full_text

double* %1127
Qload8BG
E
	full_text8
6
4%1190 = load double, double* %222, align 8, !tbaa !8
.double*8B

	full_text

double* %222
ycall8Bo
m
	full_text`
^
\%1191 = tail call double @llvm.fmuladd.f64(double %1161, double -4.000000e+00, double %1190)
-double8B

	full_text

double %1161
-double8B

	full_text

double %1190
Pload8BF
D
	full_text7
5
3%1192 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
xcall8Bn
l
	full_text_
]
[%1193 = tail call double @llvm.fmuladd.f64(double %1192, double 5.000000e+00, double %1191)
-double8B

	full_text

double %1192
-double8B

	full_text

double %1191
pcall8Bf
d
	full_textW
U
S%1194 = tail call double @llvm.fmuladd.f64(double %422, double %1193, double %1164)
,double8B

	full_text

double %422
-double8B

	full_text

double %1193
-double8B

	full_text

double %1164
Rstore8BG
E
	full_text8
6
4store double %1194, double* %1132, align 8, !tbaa !8
-double8B

	full_text

double %1194
/double*8B 

	full_text

double* %1132
Rload8BH
F
	full_text9
7
5%1195 = load double, double* %239, align 16, !tbaa !8
.double*8B

	full_text

double* %239
ycall8Bo
m
	full_text`
^
\%1196 = tail call double @llvm.fmuladd.f64(double %1168, double -4.000000e+00, double %1195)
-double8B

	full_text

double %1168
-double8B

	full_text

double %1195
Qload8BG
E
	full_text8
6
4%1197 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
xcall8Bn
l
	full_text_
]
[%1198 = tail call double @llvm.fmuladd.f64(double %1197, double 5.000000e+00, double %1196)
-double8B

	full_text

double %1197
-double8B

	full_text

double %1196
pcall8Bf
d
	full_textW
U
S%1199 = tail call double @llvm.fmuladd.f64(double %422, double %1198, double %1171)
,double8B

	full_text

double %422
-double8B

	full_text

double %1198
-double8B

	full_text

double %1171
Rstore8BG
E
	full_text8
6
4store double %1199, double* %1137, align 8, !tbaa !8
-double8B

	full_text

double %1199
/double*8B 

	full_text

double* %1137
Qload8BG
E
	full_text8
6
4%1200 = load double, double* %256, align 8, !tbaa !8
.double*8B

	full_text

double* %256
ycall8Bo
m
	full_text`
^
\%1201 = tail call double @llvm.fmuladd.f64(double %1175, double -4.000000e+00, double %1200)
-double8B

	full_text

double %1175
-double8B

	full_text

double %1200
Pload8BF
D
	full_text7
5
3%1202 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
xcall8Bn
l
	full_text_
]
[%1203 = tail call double @llvm.fmuladd.f64(double %1202, double 5.000000e+00, double %1201)
-double8B

	full_text

double %1202
-double8B

	full_text

double %1201
pcall8Bf
d
	full_textW
U
S%1204 = tail call double @llvm.fmuladd.f64(double %422, double %1203, double %1178)
,double8B

	full_text

double %422
-double8B

	full_text

double %1203
-double8B

	full_text

double %1178
Rstore8BG
E
	full_text8
6
4store double %1204, double* %1142, align 8, !tbaa !8
-double8B

	full_text

double %1204
/double*8B 

	full_text

double* %1142
Rload8BH
F
	full_text9
7
5%1205 = load double, double* %273, align 16, !tbaa !8
.double*8B

	full_text

double* %273
ycall8Bo
m
	full_text`
^
\%1206 = tail call double @llvm.fmuladd.f64(double %1182, double -4.000000e+00, double %1205)
-double8B

	full_text

double %1182
-double8B

	full_text

double %1205
Qload8BG
E
	full_text8
6
4%1207 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
xcall8Bn
l
	full_text_
]
[%1208 = tail call double @llvm.fmuladd.f64(double %1207, double 5.000000e+00, double %1206)
-double8B

	full_text

double %1207
-double8B

	full_text

double %1206
pcall8Bf
d
	full_textW
U
S%1209 = tail call double @llvm.fmuladd.f64(double %422, double %1208, double %1185)
,double8B

	full_text

double %422
-double8B

	full_text

double %1208
-double8B

	full_text

double %1185
Rstore8BG
E
	full_text8
6
4store double %1209, double* %1147, align 8, !tbaa !8
-double8B

	full_text

double %1209
/double*8B 

	full_text

double* %1147
)br8B!

	full_text

br label %1210
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %13) #4
%i8*8B

	full_text
	
i8* %13
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 80, i8* nonnull %11) #4
%i8*8B

	full_text
	
i8* %11
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 120, i8* nonnull %9) #4
$i8*8B

	full_text


i8* %9
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %6
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
%i648B

	full_text
	
i64 120
4double8B&
$
	full_text

double 1.400000e+00
$i648B

	full_text


i64 80
5double8B'
%
	full_text

double -2.000000e+00
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 1
'i648B

	full_text

	i64 63375
'i648B

	full_text

	i64 21125
&i648B

	full_text


i64 4225
5double8B'
%
	full_text

double -3.150000e+01
:double8B,
*
	full_text

double 0x405EDEB851EB851E
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
4double8B&
$
	full_text

double 4.000000e-01
#i648B

	full_text	

i64 3
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 7.500000e-01
4double8B&
$
	full_text

double 5.000000e+00
$i648B

	full_text


i64 32
$i328B

	full_text


i32 -2
&i648B

	full_text


i64 8450
5double8B'
%
	full_text

double -4.000000e+00
%i648B

	full_text
	
i64 200
:double8B,
*
	full_text

double 0xC03E3D70A3D70A3B
:double8B,
*
	full_text

double 0x40AF020000000001
'i648B

	full_text

	i64 84500
#i328B

	full_text	

i32 1
4double8B&
$
	full_text

double 1.050000e+01
:double8B,
*
	full_text

double 0x4019333333333334
#i328B

	full_text	

i32 6
'i648B

	full_text

	i64 42250
4double8B&
$
	full_text

double 8.400000e+01
4double8B&
$
	full_text

double 6.300000e+01
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 4.000000e+00
4double8B&
$
	full_text

double 2.500000e-01
'i648B

	full_text

	i64 12675
#i648B

	full_text	

i64 2
$i328B

	full_text


i32 -3
4double8B&
$
	full_text

double 6.000000e+00
4double8B&
$
	full_text

double 1.000000e+00       	  
 

                       !" !# !! $% $& '' (( )* )) +, ++ -. -- /0 // 12 13 14 11 56 55 78 77 9: 99 ;< ;; => =? =@ == AB AA CD CC EF EE GH GG IJ IK IL II MN MM OP OO QR QQ ST SS UV UW UX UU YZ YY [\ [[ ]^ ]] _` __ ab ac ad aa ef ee gh gg ij ii kl kk mn mm op oo qr qs qq tu tt vw vx vy vv z{ zz |} |~ || Ä 	Å 	Ç  ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá äã ää åç å
é åå èê èè ëí ë
ì ëë îï îî ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü ûû †° †
¢ †
£ †† §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂﬂ ‡· ‡‡ ‚‚ „‰ „„ ÂÊ Â
Á Â
Ë ÂÂ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û  Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá Ü
à Ü
â ÜÜ äã ää åç åå éè é
ê éé ëí ë
ì ë
î ëë ïñ ïï óò óó ôö ô
õ ôô úù úú ûü ûû †° †
¢ †† £§ ££ •• ¶ß ¶¶ ®© ®
™ ®
´ ®® ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±± ≤≥ ≤≤ ¥µ ¥
∂ ¥
∑ ¥¥ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”” ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä â
ã ââ åç åå éè éé êë ê
í êê ìî ì
ï ìì ñó ññ òô òò öõ ö
ú öö ùû ù
ü ùù †° †
¢ †
£ †† §• §
¶ §
ß §§ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π
∫ ππ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œœ –— –– “” “
‘ “
’ ““ ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·
‰ ·· ÂÊ ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û  Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ ÉÉ ÖÜ ÖÖ áà áá âä ââ ãå ã
ç ãã éè é
ê é
ë éé íì íí îï îî ñó ññ òô òò öõ ö
ú öö ùû ùù ü† üü °¢ °° £§ ££ •¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ää åç åå éè éé êë ê
í êê ìî ìì ïñ ïï óò óó ôö ôô õú õõ ùû ù
ü ùù †° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À    ÃÕ ÃÃ Œœ ŒŒ –— –– “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ ÖÖ áà áá âä ââ ãå ã
ç ãã éè éé êë ê
í êê ìî ìì ïñ ï
ó ïï òô òò öõ ö
ú öö ùû ùù ü† üü °¢ °° £§ ££ •¶ •
ß •• ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±± ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫∫ ªº ªª Ωæ Ω
ø Ω
¿ ΩΩ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »
À »» ÃÕ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï ÈÈ ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˘ ˙˚ ˙˙ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÖ Üá ÜÜ àâ à
ä à
ã àà åç åå éè éé êë ê
í êê ìî ì
ï ìì ñó ññ òô ò
ö òò õú õ
ù õõ ûü ûû †° †
¢ †† £§ ££ •¶ •
ß •
® •• ©™ ©
´ ©© ¨≠ ¨¨ Æ
Ø ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «« …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û Ò
Ù ÒÒ ıˆ ı
˜ ı
¯ ıı ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ Ñ
Ü ÑÑ áà á
â áá ä
ã ää åç å
é å
è åå êë êê íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûû ü† üü °¢ °
£ °
§ °° •¶ •• ß® ßß ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ Ω
ø Ω
¿ ΩΩ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ Ÿ
‹ ŸŸ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ Ä	Å	 Ä	
Ç	 Ä	Ä	 É	Ñ	 É	
Ö	 É	É	 Ü	á	 Ü	
à	 Ü	Ü	 â	ä	 â	â	 ã	å	 ã	
ç	 ã	ã	 é	è	 é	
ê	 é	é	 ë	í	 ë	ë	 ì	î	 ì	
ï	 ì	ì	 ñ	ó	 ñ	
ò	 ñ	ñ	 ô	ö	 ô	
õ	 ô	ô	 ú	ù	 ú	ú	 û	ü	 û	
†	 û	û	 °	¢	 °	
£	 °	°	 §	•	 §	§	 ¶	ß	 ¶	
®	 ¶	¶	 ©	™	 ©	
´	 ©	©	 ¨	≠	 ¨	
Æ	 ¨	¨	 Ø	∞	 Ø	Ø	 ±	≤	 ±	
≥	 ±	±	 ¥	µ	 ¥	
∂	 ¥	¥	 ∑	∏	 ∑	∑	 π	∫	 π	
ª	 π	π	 º	Ω	 º	
æ	 º	º	 ø	¿	 ø	
¡	 ø	ø	 ¬	¬	 √	ƒ	 √	√	 ≈	∆	 ≈	≈	 «	
»	 «	«	 …	 	 …	…	 À	
Ã	 À	À	 Õ	Œ	 Õ	
œ	 Õ	Õ	 –	—	 –	–	 “	”	 “	
‘	 “	“	 ’	÷	 ’	
◊	 ’	
ÿ	 ’	’	 Ÿ	⁄	 Ÿ	
€	 Ÿ	Ÿ	 ‹	›	 ‹	‹	 ﬁ	ﬂ	 ﬁ	ﬁ	 ‡	·	 ‡	‡	 ‚	
„	 ‚	‚	 ‰	Â	 ‰	
Ê	 ‰	‰	 Á	Ë	 Á	Á	 È	Í	 È	
Î	 È	È	 Ï	Ì	 Ï	
Ó	 Ï	
Ô	 Ï	Ï	 	Ò	 	
Ú	 		 Û	Ù	 Û	Û	 ı	ˆ	 ı	ı	 ˜	¯	 ˜	˜	 ˘	
˙	 ˘	˘	 ˚	¸	 ˚	
˝	 ˚	˚	 ˛	ˇ	 ˛	˛	 Ä
Å
 Ä

Ç
 Ä
Ä
 É
Ñ
 É

Ö
 É

Ü
 É
É
 á
à
 á

â
 á
á
 ä
ã
 ä
ä
 å
ç
 å
å
 é
è
 é
é
 ê

ë
 ê
ê
 í
ì
 í

î
 í
í
 ï
ñ
 ï
ï
 ó
ò
 ó

ô
 ó
ó
 ö
õ
 ö

ú
 ö

ù
 ö
ö
 û
ü
 û

†
 û
û
 °
¢
 °
°
 £
§
 £
£
 •
¶
 •
•
 ß

®
 ß
ß
 ©
™
 ©

´
 ©
©
 ¨
≠
 ¨
¨
 Æ
Ø
 Æ

∞
 Æ
Æ
 ±
≤
 ±

≥
 ±

¥
 ±
±
 µ
∂
 µ

∑
 µ
µ
 ∏
π
 ∏
∏
 ∫
ª
 ∫

º
 ∫
∫
 Ω
æ
 Ω

ø
 Ω
Ω
 ¿
¡
 ¿

¬
 ¿
¿
 √
ƒ
 √

≈
 √
√
 ∆
«
 ∆
∆
 »
…
 »

 
 »
»
 À
Ã
 À
À
 Õ
Œ
 Õ

œ
 Õ
Õ
 –
—
 –
–
 “
”
 “

‘
 “
“
 ’
÷
 ’

◊
 ’
’
 ÿ
Ÿ
 ÿ

⁄
 ÿ
ÿ
 €
‹
 €

›
 €
€
 ﬁ
ﬂ
 ﬁ

‡
 ﬁ
ﬁ
 ·
‚
 ·
·
 „
‰
 „

Â
 „
„
 Ê
Á
 Ê
Ê
 Ë
È
 Ë

Í
 Ë
Ë
 Î
Ï
 Î
Î
 Ì
Ó
 Ì

Ô
 Ì
Ì
 
Ò
 

Ú
 

 Û
Ù
 Û

ı
 Û
Û
 ˆ
˜
 ˆ

¯
 ˆ
ˆ
 ˘
˙
 ˘

˚
 ˘
˘
 ¸
˝
 ¸
¸
 ˛
ˇ
 ˛

Ä ˛
˛
 ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ãã çé ç
è çç êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü ûû †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈≈ ∆« ∆∆ »… »
  »
À »» ÃÕ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï ÈÈ ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ Ù
˜ ÙÙ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÑ ÖÜ ÖÖ áà á
â á
ä áá ãå ãã çé ç
è çç êê ëí ëë ìî ì
ï ì
ñ ìì óò óó ôö ô
õ ôô úù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®® ™´ ™
¨ ™
≠ ™™ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «« …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù Ú
ı ÚÚ ˆ˜ ˆ
¯ ˆ
˘ ˆˆ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝
Ä ˝˝ ÅÇ Å
É Å
Ñ ÅÅ ÖÜ Ö
á ÖÖ àâ à
ä àà ã
å ãã çé ç
è ç
ê çç ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú üü †° †† ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø æ
¿ æ
¡ ææ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ Ã
œ ÃÃ –— –– “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ää åç å
é åå èê è
ë èè íì íí îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –
“ –
” –– ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í Ë
Î ËË ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ Ñ
Ü ÑÑ áà áá âä ââ ãå ã
ç ãã éè éé êë ê
í êê ìî ìì ïñ ï
ó ïï òô ò
ö ò
õ òò úù ú
û úú ü† üü °¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥
∂ ¥¥ ∑∑ ∏∏ π∫ ππ ªº ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «» «« …  …… ÀÃ ÀÀ ÕŒ Õ– œœ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «« …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É Å
Ñ Å
Ö ÅÅ Üá ÜÜ àâ àà äã ä
å ää çé ç
è ç
ê ç
ë çç íì íí îï îî ñó ñ
ò ññ ôö ô
õ ô
ú ô
ù ôô ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •
® •
© •• ™´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±
µ ±± ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿
√ ¿
ƒ ¿¿ ≈∆ ≈≈ «» «
… ««  À  
Ã  
Õ  
Œ    œ– œœ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚
Â ‚‚ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ Î
Ï ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇ
É ˇˇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±
≥ ±
¥ ±± µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ º
ø ºº ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆
« ∆∆ »… »
  »
À »» ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄
› ⁄
ﬁ ⁄⁄ ﬂ‡ ﬂﬂ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï È
Ì ÈÈ ÓÔ ÓÓ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯
˚ ¯
¸ ¯¯ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â á
ä á
ã áá åç åå éè éé êë ê
í êê ìî ì
ï ìì ñó ñ
ò ñ
ô ñ
ö ññ õú õõ ùû ùù ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà á
â áá äã ä
å ä
ç ää éè é
ê éé ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù úú ûü û
† ûû °¢ °
£ °
§ °° •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «« …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —
‘ —— ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í Ë
Î ËË ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ ÑÑ Üá Üâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏∏ ∫º ª
Ω ªª æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÒ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ã
ç ãã éè éé êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú ü† üü °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇˇ ÄÅ ÄÄ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà áá âä ââ ãå ã
ç ãã éè é
ê é
ë é
í éé ìî ìì ïñ ïï óò ó
ô óó öõ ö
ú ö
ù ö
û öö ü† üü °¢ °° £§ £
• ££ ¶ß ¶
® ¶
© ¶
™ ¶¶ ´¨ ´´ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤
µ ≤
∂ ≤≤ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ªª æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈
» ≈
… ≈≈  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œ
” œœ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË Á
È Á
Í ÁÁ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ 
Ò  ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à Ü
â Ü
ä ÜÜ ãå ãã çé ç
è çç êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù úú ûü û
† ûû °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥
∑ ¥¥ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ ºº ø¿ ø
¡ ø
¬ øø √ƒ √
≈ √
∆ √√ «» «
… ««  À  
Ã    Õ
Œ ÕÕ œ– œ
— œ
“ œœ ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·
‰ ·
Â ·· ÊÁ ÊÊ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û 
Ù  ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇ
É ˇˇ ÑÖ ÑÑ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã éè é
ê é
ë é
í éé ìî ìì ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ù
ü ù
† ù
° ùù ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã ââ åç å
é åå èê è
ë èè íì í
î í
ï íí ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü ûû †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®
™ ®
´ ®® ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ªª æø æ
¿ æ
¡ ææ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œœ —“ —
” —— ‘’ ‘
÷ ‘
◊ ‘‘ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË Á
È ÁÁ ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ãã çé ç
è çç êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü ûû †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã â
å â
ç ââ éè éé êë ê
í êê ìî ì
ï ì
ñ ì
ó ìì òô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±
¥ ±± µ∂ µ
∑ µµ ∏π ∏∏ ∫
ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜
˙ ˜˜ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à Ü
â ÜÜ äã ä
å ää çé ç
è çç ê
ë êê íì í
î í
ï íí ñó ññ òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §
ß §
® §§ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥
∂ ≥
∑ ≥≥ ∏π ∏∏ ∫ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬
≈ ¬
∆ ¬¬ «» «« …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —
‘ —
’ —— ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡
‰ ‡‡ ÂÊ ÂÂ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ “
’ ““ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „
Ê „„ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ Ù
˜ ÙÙ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á Ö
à ÖÖ âä â
ã ââ åç åå éè é
ê éé ëí ëë ìî ì
ï ìì ñó ñ
ò ñ
ô ññ öõ ö
ú öö ù
ü ûû †
° †† ¢
£ ¢¢ §• &• ‚• œ• ∫• ≈¶ '¶ ±¶ Ö¶ êß ﬂß ûß ü® (® •® ˘® Ñ© ™ ∑™ ∏™ Ò™ ˇ´     	            " #! % *) , .- 0& 2+ 3/ 41 65 8 :9 <& >+ ?/ @= BA D FE H& J+ K/ LI NM P RQ T& V+ W/ XU ZY \ ^] `& b+ c/ da fe h ji l nm p[ ro s[ u( w+ x/ yv {z }t ~' Ä+ Å/ Ç ÑC Ü| àÖ â ãá çä éO ê| íè ì ïë óî òg öô úÉ ùõ üt °| ¢û £ •† ß§ ®É ™© ¨ô Æ´ Ø| ±≠ ≤ ¥∞ ∂≥ ∑ π∏ ª Ωº ø7 ¡æ ¬ ƒ√ ∆C »≈ … À  ÕO œÃ – “— ‘[ ÷” ◊ Ÿÿ €g ›⁄ ﬁ9 ·‚ ‰„ Ê+ Á/ ËÂ ÍÈ ÏÎ Ó; Ô„ Ò+ Ú/ Û ıÙ ˜ˆ ˘G ˙„ ¸+ ˝/ ˛˚ Äˇ ÇÅ ÑS Ö„ á+ à/ âÜ ãä çå è_ ê„ í+ ì/ îë ñï òó ök õ ùú üå °û ¢å §• ß¶ ©+ ™/ ´® ≠¨ Ø£ ∞± ≥≤ µ+ ∂/ ∑¥ πˆ ªÆ Ω∫ æ ¿º ¬ø √Å ≈Æ «ƒ »  ∆ Ã… Õó œŒ —∏ “– ‘£ ÷Æ ◊” ÿ ⁄’ ‹Ÿ ›∏ ﬂﬁ ·Œ „‡ ‰Æ Ê‚ Á ÈÂ ÎË Ï¨ Ó∫ Ô¨ Òƒ Ú¨ ÙŒ ız ˜Ö ¯z ˙è ˚z ˝ô ˛Ì Äˆ Åˇ É ÖÇ áÑ à ä˘ ãâ ç èå ëé íÆ î| ïì ó ôñ õò ú û üÌ °Ì ¢ù £Æ •Æ ¶† ß˘ ©˘ ™ˆ ¨ˆ ≠® Æ| ∞| ±´ ≤§ ¥Ø µ| ∑| ∏∂ ∫Æ ºÆ Ωπ æª ¿≥ ¬ø √Û ≈¸ ∆ƒ »¡ … À« Õ  Œœ —– ”+ ‘/ ’“ ◊÷ Ÿ €⁄ ›ÿ ﬂ‹ ‡– ‚+ „/ ‰· ÊÂ Ë ÍÈ ÏÁ ÓÎ Ô– Ò+ Ú/ Û ıÙ ˜ ˘¯ ˚ˆ ˝˙ ˛– Ä+ Å/ Çˇ ÑÉ Ü àá äÖ åâ ç– è+ ê/ ëé ìí ï óñ ôî õò ú ûù †ü ¢ § ¶° ®• ©æ ´™ ≠ü Æ∫ ∞Ø ≤æ ≥ µ¥ ∑ÿ π∫ ∫ ºª æΩ ¿ ¬ ƒø ∆√ « …» À  ÕÃ œΩ – “— ‘” ÷ ÿ’ ⁄◊ € ›‹ ﬂﬁ · „‚ Â‡ Á‰ Ë≈ ÍÈ Ïﬁ ÌG ÔÓ Ò≈ ÚÁ ÙG ıä ˜ˆ ˘ ˚˙ ˝¯ ˇ¸ Äø ÇÅ ÑÉ Üˆ áÑ âà ã çå èä ëé í îì ñï ò öô úó ûõ üÃ °† £ï §S ¶• ®Ã ©ˆ ´S ¨î Æ≠ ∞ ≤± ¥Ø ∂≥ ∑… π∏ ª∫ Ω≠ æé ¿ø ¬ ƒ√ ∆¡ »≈ … À  ÕÃ œ —– ”Œ ’“ ÷” ÿ◊ ⁄Ã €_ ›‹ ﬂ” ‡Ö ‚_ „§ Â‰ Á ÈË ÎÊ ÌÍ ÓŸ Ô ÚÒ Ù‰ ıò ˜ˆ ˘ ˚˙ ˝¯ ˇ¸ Ä ÇÅ ÑÉ Ü àá äÖ åâ ç⁄ èé ëÉ ík îì ñ⁄ óò ôò õk ú≥ ûù † ¢° §ü ¶£ ßË ©® ´™ ≠ù Æ  ∞Ø ≤ ¥≥ ∂± ∏µ π∫ ºª æ+ ø/ ¿Ω ¬¡ ƒ√ ∆‹ «ª …+  / À» ÕÃ œŒ —Î “ª ‘+ ’/ ÷” ÿ◊ ⁄Ÿ ‹˙ ›ª ﬂ+ ‡/ ·ﬁ „‚ Â‰ Áâ Ëª Í+ Î/ ÏÈ ÓÌ Ô Úò ÛÖ ıû ˆÖ ¯˘ ˚˙ ˝+ ˛/ ˇ¸ ÅÄ É˜ ÑÖ áÜ â+ ä/ ãà çE èÇ ëé íê îø ïˆ óÇ ôñ öò ú… ùò üû °å ¢† §˜ ¶Ç ß£ ®• ™Ÿ ´å ≠¨ Øû ±Æ ≤Ç ¥∞ µ≥ ∑Ë ∏Ä ∫é ªÄ Ωñ æÄ ¿û ¡√ √¨ ≈¬ ∆  »¨  « À‹ Õ¨ œÃ –ì “¨ ‘— ’π ◊ƒ ÿ÷ ⁄Ÿ ‹Ñ ›º ﬂ… ‡ﬁ ‚· ‰é ÂÇ ÁŒ ËÊ ÍÈ Ïò Ìº Ôº π Úπ ÛÓ ÙÇ ˆÇ ˜Ò ¯… ˙… ˚ƒ ˝ƒ ˛˘ ˇŒ ÅŒ Ç¸ Éı ÖÄ ÜŒ àŒ âá ãÇ çÇ éä èå ëÑ ìê îø ñ” óï ôí öò ú  ùû †ü ¢+ £/ §° ¶¡ ®˜ ™ß ´© ≠• Æü ∞+ ±/ ≤Ø ¥˙ ∂ê ∏µ π∑ ª≥ ºü æ+ ø/ ¿Ω ¬± ƒò ∆√ «≈ …¡  ü Ã+ Õ/ ŒÀ –Ë “• ‘— ’” ◊œ ÿü ⁄+ €/ ‹Ÿ ﬁ° ‡≥ ‚ﬂ „· Â› Êù Ëº ÍÈ ÏÁ Ì‡ ÔÎ ÒÓ Ú Ù¨ ıå ˜Ÿ ˘ˆ ˙¯ ¸∫ ˝‹ ˇ¬ Å	˛ Ç	é Ñ	Ä	 Ö	É	 á	˚ à	√ ä	· å	â	 ç	ã	 è	» ê	ì í	« î	ë	 ï	ì	 ó	ñ ò	ñ	 ö	é	 õ	˙ ù	È ü	ú	 †	û	 ¢	÷ £	  •	Ã ß	§	 ®	¶	 ™	˜ ´	©	 ≠	°	 Æ	≥ ∞	ò ≤	Ø	 ≥	±	 µ	‰ ∂	Å ∏	— ∫	∑	 ª	π	 Ω	û æ	º	 ¿	¥	 ¡	¬	 ƒ	√	 ∆	≈	 »	Ó  	…	 Ã	È Œ	À	 œ	¥ —	–	 ”	Õ	 ‘	«	 ÷	“	 ◊	Û ÿ	’	 ⁄	° €	√ ›	E ﬂ	ﬁ	 ·	‡	 „	‹	 Â	‚	 Ê	È Ë	Á	 Í	‰	 Î	«	 Ì	È	 Ó	Ü	 Ô	Ï	 Ò	Ø Ú	  Ù	Q ˆ	ı	 ¯	˜	 ˙	Û	 ¸	˘	 ˝	¯ ˇ	˛	 Å
˚	 Ç
«	 Ñ
Ä
 Ö
ô	 Ü
É
 à
Ω â
— ã
] ç
å
 è
é
 ë
ä
 ì
ê
 î
á ñ
ï
 ò
í
 ô
«	 õ
ó
 ú
¨	 ù
ö
 ü
À †
ÿ ¢
i §
£
 ¶
•
 ®
°
 ™
ß
 ´
ñ ≠
¨
 Ø
©
 ∞
«	 ≤
Æ
 ≥
ø	 ¥
±
 ∂
Ÿ ∑
 π
Á ª
∏
 º
È æ
ù ø
Ó ¡
º ¬
–	 ƒ
∏ ≈
Ω «
∆
 …
√  
  Ã
À
 Œ
Ω œ
” —
–
 ”
◊ ‘
˛ ÷
‚ ◊
‹	 Ÿ
‹ ⁄
ﬁ	 ‹
√ ›
Á	 ﬂ
E ‡
ˆ ‚
·
 ‰
¸ Â
Å Á
Ê
 È
ˆ Í
à Ï
Î
 Ó
é Ô
ë	 Ò
ô Ú
Û	 Ù
ì ı
ı	 ˜
  ¯
˛	 ˙
Q ˚
≠ ˝
¸
 ˇ
≥ Ä∏ ÇÅ Ñ≠ Öø áÜ â≈ äÃ åã é“ èä
 ë  íå
 î— ïï
 ó] ò‰ öô úÍ ùÔ üû °‰ ¢ˆ §£ ¶¸ ßÉ ©® ´â ¨°
 ÆÅ Ø£
 ±ÿ ≤¨
 ¥i µù ∑∂ π£ ∫® ºª æù øØ ¡¿ √µ ƒ≈ «∆ …+  / À» ÕÃ œŒ —‹ “∆ ‘+ ’/ ÷” ÿ◊ ⁄Ÿ ‹Î ›∆ ﬂ+ ‡/ ·ﬁ „‚ Â‰ Á˙ Ë∆ Í+ Î/ ÏÈ ÓÌ Ô Úâ Û∆ ı+ ˆ/ ˜Ù ˘¯ ˚˙ ˝ò ˛ú Äï
 Çˇ ÉÑ ÜÖ à+ â/ äá åã éï
 èê íë î+ ï/ ñì òç öÁ	 õô ùø ûç †˛	 °ü £… §¨
 ¶ó ß• ©ï
 ´ç ¨® ≠™ ØŸ ∞ó ≤± ¥¨
 ∂≥ ∑ç πµ ∫∏ ºË Ωã øÁ	 ¿ã ¬˛	 √ã ≈¨
 ∆¸ »√  « Ã… Õ« œı	 –« “å
 ”« ’£
 ÷æ ÿÀ Ÿ◊ €⁄ ›Ñ ﬁ¡ ‡Œ ·ﬂ „‚ Âé Êç Ë— ÈÁ ÎÍ Ìò Ó¡ ¡ Òæ Ûæ ÙÔ ıç ˜ç ¯Ú ˘Œ ˚Œ ¸À ˛À ˇ˙ Ä— Ç— É˝ Ñˆ ÜÅ á— â— äà åç éç èã êç íÖ îë ïƒ ó‘ òñ öì õô ù  ûü °† £+ §/ •¢ ß¡ ©ï
 ´® ¨™ Æ¶ Ø† ±+ ≤/ ≥∞ µ˙ ∑ô π∂ ∫∏ º¥ Ω† ø+ ¿/ ¡æ √± ≈ü «ƒ »∆  ¬ À† Õ+ Œ/ œÃ —Ë ”™ ’“ ÷‘ ÿ– Ÿ† €+ ‹/ ›⁄ ﬂ° ·∏ „‡ ‰‚ Êﬁ Áù Èº ÎÍ ÌË Ó‡ Ï ÚÔ ÛÒ ı≠ ˆå ¯⁄ ˙˜ ˚˘ ˝ª ˛‹ Ä… Çˇ ÉÁ	 ÖÅ ÜÑ à¸ â√ ã‚ çä éå ê… ëì ìı	 ïí ñ˛	 òî ôó õè ú˙ ûÍ †ù °ü £◊ §  ¶å
 ®• ©ï
 ´ß ¨™ Æ¢ Ø≥ ±ô ≥∞ ¥≤ ∂Â ∑Å π£
 ª∏ º¨
 æ∫ øΩ ¡µ ¬Í ƒË ∆√ «Ô …≈  ¥ ÃÀ Œ» œ«	 —Õ “Ù ”– ’¢ ÷√ ÿ◊ ⁄ˇ ‹Ÿ ›E ﬂﬁ ·€ ‚È ‰„ Ê‡ Á«	 ÈÂ Íá ÎË Ì∞ Ó  Ô Úí ÙÒ ıQ ˜ˆ ˘Û ˙¯ ¸˚ ˛¯ ˇ«	 Å˝ Çö ÉÄ Öæ Ü— àá ä• åâ ç] èé ëã íá îì ñê ó«	 ôï ö≠ õò ùÃ ûÿ †ü ¢∏ §° •i ß¶ ©£ ™ñ ¨´ Æ® Ø«	 ±≠ ≤¿ ≥∞ µ⁄ ∂À ∫ˇ º◊ æí ¿Ô ¬• ƒá ∆ì »∏  ü Ã∏ Œ –” “ú ‘∑ ÷Ω ÿˆ ⁄≠ ‹‰ ﬁù ‡ﬂ „¯ ‰› ÊË Á€ È’ ÍŸ Ï≈ Ì◊ Ô∫ ´ Ú„ Û¶ ıÒ ˆÀ ¯Ñ ˘… ˚Ç ¸ì ˛Ã ˇ« ÅÄ Çé Ñ« Ö≈ á˛ à√ ä¸ ã˚ ç≥ éˆ êå ë¡ ì˙ îø ñ¯ ó„ ôú öﬁ úò ùΩ üˆ †ª ¢Ù £π •Ú ¶Ô ®® ©Í ´ß ¨Ë Æ™ Ø≤ ±∞ ≥§ µ∫ ∂Ó ∏√ π  ª° Ω‰ æû ¿ﬁ ¡Î √¸ ƒÅ ∆à »«  é Àï Õõ Œí –ï —Ë ”≥ ‘∏ ÷ø ÿ◊ ⁄≈ €â ›“ ﬁÜ ‡Ã ·Ä „_ ‰Â ÊÍ ÁÔ Èˆ ÎÍ Ì¸ Ó˙ â Ò˜ ÛÉ Ù‚ ˆ£ ˜® ˘Ø ˚˙ ˝µ ˛∞ Ä& Çˇ É+ Ñ/ ÖÅ áÜ âà ã‹ å& éˇ è+ ê/ ëç ìí ïî óÎ ò& öˇ õ+ ú/ ùô üû °† £˙ §& ¶ˇ ß+ ®/ ©• ´™ ≠¨ Øâ ∞& ≤ˇ ≥+ ¥/ µ± ∑∂ π∏ ªò º˝ æ” ø( ¡≤ ¬+ √/ ƒ¿ ∆≈ »˝ …' À≤ Ã+ Õ/ Œ  –« “ò ”— ’ø ÷« ÿå Ÿ◊ €… ‹Ò ﬁœ ﬂ› ·˝ „« ‰‡ Â‚ ÁŸ Ëœ ÍÈ ÏÒ ÓÎ Ô« ÒÌ Ú ÙË ı≈ ˜ò ¯≈ ˙å ˚≈ ˝Ò ˛( Ä∞ Å+ Ç/ Éˇ ÖÑ áõ àÑ äè ãÑ çÉ éÑ êÙ ëˆ ìÜ îí ñï òÑ ô˘ õâ úö ûù †é °« £å §¢ ¶• ®ò ©˘ ´˘ ¨ˆ Æˆ Ø™ ∞« ≤« ≥≠ ¥â ∂â ∑Ü πÜ ∫µ ªå Ωå æ∏ ø± ¡º ¬å ƒå ≈√ «« …«  ∆ À» Õ¿ œÃ –¸ “è ”— ’Œ ÷‘ ÿ  Ÿﬂ €∞ ‹+ ›/ ﬁ⁄ ‡¡ ‚˝ ‰· Â„ Áﬂ Ëﬂ Í∞ Î+ Ï/ ÌÈ Ô˙ Ò— Û ÙÚ ˆÓ ˜ﬂ ˘∞ ˙+ ˚/ ¸¯ ˛± Ä◊ Çˇ ÉÅ Ö˝ Üﬂ à∞ â+ ä/ ãá çË è‚ ëé íê îå ïﬂ ó∞ ò+ ô/ öñ ú° û †ù °ü £õ §ß ¶™ ß‡ ©• ´® ¨™ ÆÊ Øå ±ï ≥∞ ¥≤ ∂ı ∑‹ πõ ª∏ ºò æ∫ øΩ ¡µ ¬√ ƒù ∆√ «≈ …Ñ  ì Ãè ŒÀ œå —Õ “– ‘» ’˙ ◊• Ÿ÷ ⁄ÿ ‹ì ›  ﬂÉ ·ﬁ ‚˝ ‰‡ Â„ Á€ Ë≥ Í‘ ÏÈ ÌÎ Ô¢ Å ÚÙ ÙÒ ıÒ ˜Û ¯ˆ ˙Ó ˚™ ˝≠ ˛ß Ä¸ Å® Éˇ Ñ¥ ÜÇ àÖ â«	 ãá å≠ çä è⁄ ê‚ í∏ îë ïõ óì òò öñ õÈ ùô üú †«	 ¢û £¿ §° ¶È ßô ©À ´® ¨è Æ™ Øå ±≠ ≤¯ ¥∞ ∂≥ ∑«	 πµ ∫” ª∏ Ω¯ æ– ¿ﬁ ¬ø √É ≈¡ ∆] »«  ƒ Àá Õ… œÃ –«	 “Œ ”Ê ‘— ÷á ◊á ŸÒ €ÿ ‹Ù ﬁ⁄ ﬂÒ ·› ‚ñ ‰‡ Ê„ Á«	 ÈÂ Í˘ ÎË Ìñ Ó≤ ’ ÒÖ Û∏ ıõ ˜À ˘è ˚ﬁ ˝É ˇÃ ÅÒ ÉÙ ÖÔ á≠ âœ ä™ åù çß èº ê∫ íΩ ì— ï◊ ñõ ò√ ôò õE ú≈ ûˆ üè °  ¢å §Q •’ ß≠ ®É ™— ´Ë ≠‰ ÆÙ ∞ÿ ±Ò ≥i ¥¯ ∂ù ∑ﬁ π„ º´ ΩÒ ø¶ ¿Ñ ¬À √Ç ≈… ∆Ã »ì …Ä À« Ã« Œé œ˛ —≈ “¸ ‘√ ’≥ ◊˚ ÿå ⁄ˆ €˙ ›¡ ﬁ¯ ‡ø ·ú „„ ‰ò Êﬁ Áˆ ÈΩ Í∏ Ïª ÌÚ Ôπ ü ÛÚ ı• ˆæ ¯˜ ˙ü ˚∫ ˝¸ ˇæ ÄÓ Ç∫ ÉΩ ÖÑ á√ à  äâ åΩ ç” èé ë◊ íÎ î‰ ïË óﬁ òÂ ö√ õ‚ ùE ûˆ †ü ¢¸ £Å •§ ßˆ ®à ™© ¨é ≠ﬂ Øõ ∞‹ ≤ï ≥Ÿ µ  ∂÷ ∏Q π≠ ª∫ Ω≥ æ∏ ¿ø ¬≠ √ø ≈ƒ «≈ »”  “ À– ÕÃ ŒÕ –— —  ”_ ‘‰ ÷’ ÿÍ ŸÔ €⁄ ›‰ ﬁˆ ‡ﬂ ‚¸ „ƒ Ââ Ê¡ ËÉ Èæ Îÿ Ïª Ói Ôù Ò Û£ Ù® ˆı ¯ù ˘Ø ˚˙ ˝µ ˛ˇ Å& ÉÄ Ñ+ Ö/ ÜÇ àá äâ å‹ ç& èÄ ê+ ë/ íé îì ñï òÎ ô& õÄ ú+ ù/ ûö †ü ¢° §˙ •& ßÄ ®+ ©/ ™¶ ¨´ Æ≠ ∞â ±& ≥Ä ¥+ µ/ ∂≤ ∏∑ ∫π ºò Ωú ø« ¡æ ¬Ò ƒ( ∆√ «+ »/ …≈ À  Õ« Œ' –√ —+ “/ ”œ ’Ã ◊‚ ÿ÷ ⁄ø €Ã ›÷ ﬁ‹ ‡… ·ª „‘ ‰‚ Ê« ËÃ ÈÂ ÍÁ ÏŸ Ì‘ ÔÓ Òª Û ÙÃ ˆÚ ˜ı ˘Ë ˙  ¸‚ ˝  ˇ÷ Ä  Çª É∑ Ö( áÑ à+ â/ äÜ åã éÂ èã ëŸ íã îÕ ïã óæ ò˚ öç õô ùú üÑ †˛ ¢ê £° •§ ßé ®Ã ™ì ´© ≠¨ Øò ∞˛ ≤˛ ≥˚ µ˚ ∂± ∑Ã πÃ ∫¥ ªê Ωê æç ¿ç ¡º ¬ì ƒì ≈ø ∆∏ »√ …ì Àì Ã  ŒÃ –Ã —Õ “œ ‘« ÷” ◊Å Ÿñ ⁄ÿ ‹’ ›€ ﬂ  ‡ﬂ ‚Ñ „+ ‰/ Â· Á¡ È« ÎË ÏÍ ÓÊ Ôﬂ ÒÑ Ú+ Û/ Ù ˆ˙ ¯÷ ˙˜ ˚˘ ˝ı ˛ﬂ ÄÑ Å+ Ç/ Éˇ Ö± á‹ âÜ äà åÑ çﬂ èÑ ê+ ë/ íé îË ñÁ òï ôó õì úﬂ ûÑ ü+ †/ °ù £° •ı ß§ ®¶ ™¢ ´ù ≠º ØÆ ±¨ ≤‡ ¥∞ ∂≥ ∑µ πÌ ∫å ºú æª øΩ ¡¸ ¬‹ ƒÂ ∆√ «‚ …≈  » Ã¿ Õ√ œ§ —Œ “– ‘ã ’ì ◊Ÿ Ÿ÷ ⁄÷ ‹ÿ ›€ ﬂ” ‡˙ ‚¨ ‰· Â„ Áö Ë  ÍÕ ÏÈ Ì« ÔÎ Ó ÚÊ Û≥ ı€ ˜Ù ¯ˆ ˙© ˚Å ˝æ ˇ¸ Äª Ç˛ ÉÅ Ö˘ Ü£ à¨ äá ãÆ çâ é≥ êå ë«	 ìè î∏ ïí ó· ò‚ ö√ úô ù√ üû °õ ¢E §£ ¶† ß«	 ©• ™À ´® ≠ Æô ∞÷ ≤Ø ≥  µ¥ ∑± ∏Q ∫π º∂ Ω«	 øª ¿ﬁ ¡æ √ˇ ƒ– ∆È »≈ …— À  Õ« Œ] –œ “Ã ”«	 ’— ÷Ò ◊‘ Ÿé ⁄á ‹¸ ﬁ€ ﬂÿ ·‡ „› ‰i ÊÂ Ë‚ È«	 ÎÁ ÏÑ ÌÍ Ôù  Ú¨ ÙÒ ıÆ ˜ù ¯≥ ˙º ˚∂ ˝¸ ˇ∫ ÄΩ ÇÅ Ñ√ Ö  áÜ âΩ ä” åã é◊ è√ ë‚ íû î‹ ï£ ó√ òÎ öô úG ùˆ üû °¸ ¢Å §£ ¶ˆ ßà ©® ´é ¨÷ Æô Ø¥ ±ì ≤π ¥  µ˙ ∑∂ πS ∫≠ ºª æ≥ ø∏ ¡¿ √≠ ƒø ∆≈ »≈ …È À– Ã  Œ  œœ —— “â ‘” ÷_ ◊‰ Ÿÿ €Í ‹Ô ﬁ› ‡‰ ·ˆ „‚ Â¸ Ê¸ Ëá È‡ ÎÅ ÏÂ Óÿ Ôò Ò Ûk Ùù ˆı ¯£ ˘® ˚˙ ˝ù ˛Ø Äˇ Çµ É” Öû Ü” à( äÄ ã+ å/ çâ èé ëá í' îÄ ï+ ñ/ óì ôô õê ùö ûú †ø °∂ £ê •¢ ¶§ ®… © ´™ ≠ò Æ¨ ∞á ≤ê ≥Ø ¥± ∂Ÿ ∑ò π∏ ª™ Ω∫ æê ¿º ¡ø √Ë ƒé ∆ö «é …¢  é Ã™ Õ≈ œŒ —£ “Œ ‘π ’Œ ◊œ ÿŒ ⁄Â €≈ ›– ﬁ‹ ‡ﬂ ‚Ñ „» Â” Ê‰ ËÁ Íé Îê Ì÷ ÓÏ Ô Úò Û» ı» ˆ≈ ¯≈ ˘Ù ˙ê ¸ê ˝˜ ˛” Ä” Å– É– Ñˇ Ö÷ á÷ àÇ â˚ ãÜ å÷ é÷ èç ëê ìê îê ïí óä ôñ öÀ úŸ ùõ üò †û ¢  £ﬂ •√ ¶+ ß/ ®§ ™¡ ¨á Æ´ Ø≠ ±© ≤ﬂ ¥√ µ+ ∂/ ∑≥ π˙ ªú Ω∫ æº ¿∏ ¡ﬂ √√ ƒ+ ≈/ ∆¬ »±  § Ã… ÕÀ œ« –ﬂ “√ ”+ ‘/ ’— ◊Ë Ÿ± €ÿ ‹⁄ ﬁ÷ ﬂﬂ ·√ ‚+ „/ ‰‡ Êı Ëø ÍÁ ÎÈ ÌÂ Óù º ÚÒ ÙÔ ı‡ ˜Û ˘ˆ ˙¯ ¸∞ ˝å ˇﬂ Å˛ ÇÄ Ñø Ö‹ á£ âÜ äà åö çã èÉ ê√ íÁ îë ïì óŒ òì öπ úô ùõ ü¢ †û ¢ñ £˙ •Ô ß§ ®¶ ™› ´  ≠œ Ø¨ ∞Æ ≤á ≥± µ© ∂≥ ∏û ∫∑ ªπ ΩÏ æÅ ¿Â ¬ø √¡ ≈™ ∆ƒ »º …£ ÀÔ Õ  ŒÒ –Ã —«	 ”œ ‘˚ ’“ ◊§ ÿ‚ ⁄Ü ‹Ÿ ›√ ﬂﬁ ·€ ‚«	 ‰‡ Âé Ê„ Ë≥ Èô Îô ÌÍ Ó  Ô ÚÏ Û«	 ıÒ ˆ° ˜Ù ˘¬ ˙– ¸¨ ˛˚ ˇ— ÅÄ É˝ Ñ«	 ÜÇ á¥ àÖ ä— ãá çø èå êÿ íë îé ï«	 óì ò« ôñ õ‡ ú ü ° £  û$ &$ ûÕ œÕ ª· ‚ù ûÜ àÜ ‚∫ ª § ¨¨ ÆÆ ∞∞ ≠≠ ØØê ÆÆ ê‰	 ÆÆ ‰	ƒ ÆÆ ƒ¬	 ØØ ¬	Ú ÆÆ ÚÁ ÆÆ Áœ ÆÆ œµ ÆÆ µæ ÆÆ æ€ ÆÆ €Õ	 ÆÆ Õ	Î ÆÆ Î” ÆÆ ”µ ÆÆ µÉ ÆÆ É† ∞∞ †Ú ÆÆ ÚÄ ÆÆ ÄÁ ÆÆ ÁÌ ÆÆ Ì´ ÆÆ ´∏ ÆÆ ∏« ÆÆ «ç ÆÆ ç» ÆÆ »˝ ÆÆ ˝« ÆÆ «ô	 ÆÆ ô	Î ÆÆ Î¿ ÆÆ ¿Ê ÆÆ Êª ÆÆ ª‡ ÆÆ ‡π	 ÆÆ π	≈ ÆÆ ≈í ÆÆ í… ÆÆ …¥ ÆÆ ¥
 ¨¨ 
› ÆÆ › ≠≠ ò ÆÆ òÂ ÆÆ Â± ÆÆ ±‡ ÆÆ ‡å ÆÆ å¥ ÆÆ ¥˚	 ÆÆ ˚	÷ ÆÆ ÷±
 ÆÆ ±
ß ÆÆ ß∫ ÆÆ ∫¿ ÆÆ ¿⁄ ÆÆ ⁄≠ ÆÆ ≠≈ ÆÆ ≈õ ÆÆ õû ÆÆ û’ ÆÆ ’€ ÆÆ €é ÆÆ é‘ ÆÆ ‘ã ÆÆ ãí
 ÆÆ í
Ñ ÆÆ Ñœ ÆÆ œì	 ÆÆ ì	‰ ÆÆ ‰ö ÆÆ ö™ ÆÆ ™≠ ÆÆ ≠• ÆÆ •› ÆÆ ›» ÆÆ »ö ÆÆ öÏ ÆÆ Ï°	 ÆÆ °	Ï ÆÆ ÏÒ ÆÆ Òè ÆÆ èÍ ÆÆ Í¨ ÆÆ ¨º ÆÆ º≠ ÆÆ ≠É
 ÆÆ É
ı ÆÆ ı∫ ÆÆ ∫– ÆÆ –© ÆÆ ©± ÆÆ ±‘ ÆÆ ‘¡ ÆÆ ¡° ÆÆ °Ü ÆÆ Ü‡ ÆÆ ‡Œ ÆÆ Œ— ÆÆ —◊ ÆÆ ◊Ë ÆÆ Ë® ÆÆ ®€ ÆÆ €≠ ÆÆ ≠” ÆÆ ”Ï	 ÆÆ Ï	ì ÆÆ ì˛ ÆÆ ˛Ù ÆÆ Ù“ ÆÆ “© ÆÆ ©ö
 ÆÆ ö
∫ ÆÆ ∫Ï ÆÆ Ï¢ ÆÆ ¢À ÆÆ À∞ ÆÆ ∞√ ÆÆ √∞ ÆÆ ∞» ÆÆ »å ÆÆ åﬁ ÆÆ ﬁî ÆÆ î¸ ÆÆ ¸€ ÆÆ €ı ÆÆ ı∞ ÆÆ ∞‚ ÆÆ ‚ä ÆÆ ä˚ ÆÆ ˚Ì ÆÆ Ì¡ ÆÆ ¡∂ ÆÆ ∂ˆ ÆÆ ˆé ÆÆ é˚ ÆÆ ˚† ÆÆ †Ç ÆÆ Ç± ÆÆ ±∞ ÆÆ ∞ÿ ÆÆ ÿè ÆÆ è˘ ÆÆ ˘í ÆÆ íø ÆÆ øØ ÆÆ Øñ ÆÆ ñ’ ÆÆ ’˝ ÆÆ ˝¡ ÆÆ ¡µ ÆÆ µ£ ÆÆ £Ç ÆÆ Ç„ ÆÆ „Ê ÆÆ Ê» ÆÆ »Ä ÆÆ Ä≠ ÆÆ ≠® ÆÆ ®¿ ÆÆ ¿ì ÆÆ ì§ ÆÆ §ª ÆÆ ªã ÆÆ ã©
 ÆÆ ©
Ò ÆÆ Ò√	 ØØ √	ø	 ÆÆ ø	¸ ÆÆ ¸ñ ÆÆ ñì ÆÆ ì¸ ÆÆ ¸Õ ÆÆ Õ’	 ÆÆ ’	á ÆÆ á— ÆÆ —à ÆÆ à› ÆÆ ›â ÆÆ âÃ ÆÆ Ã ¨¨ ò ÆÆ ò† ÆÆ †• ÆÆ •º ÆÆ º¸ ÆÆ ¸∞ ÆÆ ∞« ÆÆ «¶	 ÆÆ ¶	Û ÆÆ Ûí ÆÆ í∏ ÆÆ ∏° ÆÆ °ì ÆÆ ìÛ ÆÆ Û¨	 ÆÆ ¨	• ÆÆ • ¨¨ Û ÆÆ ÛÆ ÆÆ Æõ ÆÆ õÅ ÆÆ Åò ÆÆ òñ ÆÆ ñ… ÆÆ …‚ ÆÆ ‚∏ ÆÆ ∏ô ÆÆ ô¢ ÆÆ ¢Ç ÆÆ ÇÖ ÆÆ ÖÄ	 ÆÆ Ä	Ó ÆÆ Ó≠ ÆÆ ≠™ ÆÆ ™Ã ÆÆ Ãô ÆÆ ôÑ ÆÆ ÑÙ ÆÆ Ù‚ ÆÆ ‚ ≠≠ Ë ÆÆ ËÅ ÆÆ Å‡ ÆÆ ‡ª ÆÆ ª¥	 ÆÆ ¥	¯ ÆÆ ¯Ò ÆÆ ÒÊ ÆÆ Ê˜ ÆÆ ˜˘ ÆÆ ˘ø ÆÆ ø† ÆÆ †∏ ÆÆ ∏ˇ ÆÆ ˇÜ	 ÆÆ Ü	¢ ∞∞ ¢º ÆÆ ºé	 ÆÆ é	˚ ÆÆ ˚Û ÆÆ Ûû ∞∞ ûŒ ÆÆ Œ± 
± ¢
≤ ≠
≤ ‚
≤ ∞
≤ µ
≤ Ì
≤ Ú
≤ º≥ ≥ †
¥ Î
¥ Ä	
¥ ì	
¥ ¶	
¥ π	
¥ Ï
¥ Å
¥ î
¥ ß
¥ ∫
¥ •
¥ ∫
¥ Õ
¥ ‡
¥ Û
¥ ∞
¥ ≈
¥ ÿ
¥ Î
¥ ˛
¥ Û
¥ à
¥ õ
¥ Æ
¥ ¡µ 	∂ 	∂ 	∂ =	∂ E	∂ m
∂ ä
∂ ä
∂ î
∂ §
∂ ≥
∂ √
∂ 
∂ ø
∂ Ñ
∂ Ñ
∂ é
∂ ò
∂  
∂ ·
∂ È
∂ ù
∂ ª
∂ —
∂ ‹
∂ ‹
∂ ‚
∂ ˙
∂ å
∂ ì
∂  
∂ Å
∂ »
∂ Ø
∂ ”
∂ ∞
∂ ≤
∂ ç
∂ È
∂ é
∂ 
∂ ≥
∑ ∫
∏ ‚
∏ û
π •
π ±
∫ ¨
∫ ∫
∫ »
∫ ÷
∫ ‰
∫ ≠
∫ ª
∫ …
∫ ◊
∫ Â
∫ Ê
∫ ı
∫ Ñ
∫ ì
∫ ¢
∫ Ì
∫ ¸
∫ ã
∫ ö
∫ ©
∫ ∞
∫ ø
∫ Œ
∫ ›
∫ Ï
ª «
ª ò
ª ô
ª ‘
ª €
ª û	º 	º  
º ˇ	Ω 1	Ω 9	Ω =	Ω E	Ω I	Ω Q	Ω U	Ω ]	Ω a	Ω i	Ω m	Ω v	Ω 
Ω ä
Ω î
Ω §
Ω ≥
Ω ∏
Ω ∏
Ω º
Ω º
Ω √
Ω  
Ω —
Ω ÿ
Ω ‡
Ω ‡
Ω Â
Ω 
Ω ˚
Ω Ü
Ω ë
Ω ú
Ω ®
Ω ¥
Ω ø
Ω …
Ω Ÿ
Ω Ë
Ω Ñ
Ω é
Ω ò
Ω  
Ω “
Ω ⁄
Ω ·
Ω È
Ω 
Ω ¯
Ω ˇ
Ω á
Ω é
Ω ñ
Ω ù
Ω ù
Ω £
Ω £
Ω £
Ω ¥
Ω ¥
Ω ª
Ω ª
Ω ¡
Ω ¡
Ω ¡
Ω »
Ω »
Ω —
Ω —
Ω ‹
Ω ‚
Ω ‚
Ω ˙
Ω ˙
Ω å
Ω å
Ω ì
Ω ô
Ω ô
Ω ±
Ω ±
Ω √
Ω √
Ω  
Ω –
Ω –
Ω Ë
Ω Ë
Ω ˙
Ω ˙
Ω Å
Ω á
Ω á
Ω °
Ω °
Ω ≥
Ω ≥
Ω Ω
Ω »
Ω ”
Ω ﬁ
Ω È
Ω ¸
Ω à
Ω °
Ω °
Ω Ø
Ω Ω
Ω À
Ω Ÿ
Ω ∏

Ω ∏

Ω ∏

Ω »
Ω ”
Ω ﬁ
Ω È
Ω Ù
Ω ˇ
Ω ˇ
Ω á
Ω ì
Ω ¢
Ω ¢
Ω ∞
Ω æ
Ω Ã
Ω ⁄
Ω œ
Ω œ
Ω œ
Ω ”
Ω ”
Ω ⁄
Ω æ
Ω æ
Ω ·
Ω Ò
Ω Ò
Ω Ò
Ω §
æ û
æ ©
æ ”
æ ﬁ
æ £
æ ¨
æ ®
æ ±
æ ‡
æ È
æ Â
æ Ó
æ Ø
æ ∏	ø 9	ø E	ø Q	ø U	ø ]	ø ]	ø i
ø §
ø ∏
ø —
ø Ü
ø Ÿ
ø ò
ø ˇ
ø á
ø  
ø –
ø Ë
ø ˙
ø ﬁ
ø À
ø È
ø Ãø ∞
ø •
ø á
ø ¶
ø é
ø —¿ ´¿ ‡¿ π¿ Æ¿ ä¿ «	¿ À	¿ ‚	¿ ˘	¿ ê
¿ ß
¿ ≥¿ ã¿ Î¿ ∆¿ ¿ Õ¿ ∫¿ ê¡ ¬	
¡ ¬	
¬ Õ	
¬ ‰	
¬ ˚	
¬ í

¬ ©

¬ œ
¬ ‡
¬ Ò
¬ Ç
¬ ì	√ )	√ +	√ -	√ /
ƒ Ò
≈ ˘
≈ Ö
∆ ≈
∆ »
∆ €
∆ ‡
∆ Û
∆ ¯
∆ ã
∆ ê
∆ £
∆ ®
∆ ¸
∆ Ç
∆ ì
∆ ô
∆ ™
∆ ∞
∆ ¡
∆ …
∆ ⁄
∆ ‡
∆ â
∆ è
∆ õ
∆ •
∆ ±
∆ ª
∆ «
∆ —
∆ ›
∆ Á
∆ Ã
∆ €
∆ Ï
∆ ˝
∆ é« « û
» ¡
» í
» ì
» Œ
» ’
» ò
… Û
… Ü	
… ô	
… ¨	
… ø	
… Ù
… á
… ö
… ≠
… ¿
… ≠
… ¿
… ”
… Ê
… ˘
… ∏
… À
… ﬁ
… Ò
… Ñ
… ˚
… é
… °
… ¥
… «
  ≈À À À À 
Ã ø
Ã ê
Ã ë
Ã Ã
Ã ”
Ã ñ
Õ ˚
Õ é	
Õ °	
Õ ¥	
Õ ¸
Õ è
Õ ¢
Õ µ
Õ µ
Õ »
Õ €
Õ Ó
Õ ¿
Õ ”
Õ Ê
Õ ˘
Õ É
Õ ñ
Õ ©
Õ º
Œ ∏
œ œ
œ ü
– ñ
– È
– Í
– •
– ¨
– Ô
— Ç
— å
— Ÿ
— ·
— ⁄
— ‚
— ï
— ù
— ú
— §
— ﬂ
— Á	“ a	“ i
“ ≥
“ ÿ
“ ë
“ Ë
“  
“ ⁄
“ È
“ ¯
“ á
“ é
“ ñ
“ ñ
“ ¥
“ Å
“ á
“ °
“ ≥
“ È
“ Ÿ
“ Ù
“ ⁄
“ ±
“ ñ
“ ≤
“ ù
“ ‡
” …	
” ‡	
” ˜	
” é

” •

‘ ≈	
’ Ñ
’ ê	÷ I	÷ Q
÷ î
÷ º
÷ √
÷  
÷  
÷ —
÷ ÿ
÷ ˚
÷ ú
÷ ø
÷ …
÷ …
÷ Ÿ
÷ Ë
÷ é
÷ 
÷ ¯
÷ »
÷ ì
÷ ô
÷ ±
÷ √
÷ ”
÷ Ω
÷ ﬁ
÷ æ
÷ ˇ
÷ ô
÷ ¯
÷ ö
÷ ˇ
÷ ¬
◊ ∑
ÿ √
ÿ Ÿ
ÿ Ò
ÿ â
ÿ °
ÿ ˇ
ÿ ñ
ÿ ≠
ÿ ƒ
ÿ ›
ÿ å
ÿ †
ÿ ∂
ÿ Ã
ÿ ‚
Ÿ √	"
rhsz"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z3maxdd"
llvm.lifetime.end.p0i8*á
npb-LU-rhsz.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

wgsize
>

transfer_bytes
òì…

devmap_label

 
transfer_bytes_log1p
⁄açA

wgsize_log1p
⁄açA