
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
%19 = add nsw i32 %6, -1
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
!br i1 %20, label %21, label %1186
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
%23 = add nsw i32 %5, -1
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
!br i1 %24, label %25, label %1186
#i18B

	full_text


i1 %24
Wbitcast8BJ
H
	full_text;
9
7%26 = bitcast double* %0 to [33 x [33 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%27 = bitcast double* %2 to [33 x [33 x double]]*
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %3 to [33 x [33 x double]]*
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
getelementptr8B}
{
	full_textn
l
j%33 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Nbitcast8BA
?
	full_text2
0
.%34 = bitcast [33 x [5 x double]]* %33 to i64*
G[33 x [5 x double]]*8B+
)
	full_text

[33 x [5 x double]]* %33
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
 getelementptr8BŒ
‰
	full_text|
z
x%38 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 0, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
ƒgetelementptr8Bp
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
 getelementptr8BŒ
‰
	full_text|
z
x%43 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 0, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
ƒgetelementptr8Bp
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
 getelementptr8BŒ
‰
	full_text|
z
x%48 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 0, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
ƒgetelementptr8Bp
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
 getelementptr8BŒ
‰
	full_text|
z
x%53 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 0, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
ƒgetelementptr8Bp
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
*store i64 %40, i64* %59, align 8, !tbaa !8
%i648B

	full_text
	
i64 %40
'i64*8B

	full_text


i64* %59
?bitcast8B2
0
	full_text#
!
%60 = bitcast i64 %40 to double
%i648B

	full_text
	
i64 %40
‹getelementptr8Bx
v
	full_texti
g
e%61 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 0
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
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
‹getelementptr8Bx
v
	full_texti
g
e%64 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 0
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
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
%66 = bitcast i64 %55 to double
%i648B

	full_text
	
i64 %55
7fsub8B-
+
	full_text

%67 = fsub double %66, %65
+double8B

	full_text


double %66
+double8B

	full_text


double %65
@fmul8B6
4
	full_text'
%
#%68 = fmul double %67, 4.000000e-01
+double8B

	full_text


double %67
icall8B_
]
	full_textP
N
L%69 = tail call double @llvm.fmuladd.f64(double %60, double %63, double %68)
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


double %68
‚getelementptr8Bo
m
	full_text`
^
\%70 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Nstore8BC
A
	full_text4
2
0store double %69, double* %70, align 8, !tbaa !8
+double8B

	full_text


double %69
-double*8B

	full_text

double* %70
?bitcast8B2
0
	full_text#
!
%71 = bitcast i64 %45 to double
%i648B

	full_text
	
i64 %45
7fmul8B-
+
	full_text

%72 = fmul double %63, %71
+double8B

	full_text


double %63
+double8B

	full_text


double %71
‚getelementptr8Bo
m
	full_text`
^
\%73 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Nstore8BC
A
	full_text4
2
0store double %72, double* %73, align 8, !tbaa !8
+double8B

	full_text


double %72
-double*8B

	full_text

double* %73
?bitcast8B2
0
	full_text#
!
%74 = bitcast i64 %50 to double
%i648B

	full_text
	
i64 %50
7fmul8B-
+
	full_text

%75 = fmul double %63, %74
+double8B

	full_text


double %63
+double8B

	full_text


double %74
‚getelementptr8Bo
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
U%79 = tail call double @llvm.fmuladd.f64(double %66, double 1.400000e+00, double %78)
+double8B

	full_text


double %66
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
‚getelementptr8Bo
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
ƒgetelementptr8Bp
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
ƒgetelementptr8Bp
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
ƒgetelementptr8Bp
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
ƒgetelementptr8Bp
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
ƒgetelementptr8Bp
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
ƒgetelementptr8Bp
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
7%94 = bitcast double* %1 to [33 x [33 x [5 x double]]]*
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
™getelementptr8B…
‚
	full_textu
s
q%96 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
'%97 = bitcast [5 x double]* %96 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %96
Hload8B>
<
	full_text/
-
+%98 = load i64, i64* %97, align 8, !tbaa !8
'i64*8B

	full_text


i64* %97
Hstore8B=
;
	full_text.
,
*store i64 %98, i64* %37, align 8, !tbaa !8
%i648B

	full_text
	
i64 %98
'i64*8B

	full_text


i64* %37
 getelementptr8BŒ
‰
	full_text|
z
x%99 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 1, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
Bbitcast8B5
3
	full_text&
$
"%100 = bitcast double* %99 to i64*
-double*8B

	full_text

double* %99
Jload8B@
>
	full_text1
/
-%101 = load i64, i64* %100, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %100
Istore8B>
<
	full_text/
-
+store i64 %101, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %101
'i64*8B

	full_text


i64* %42
¡getelementptr8B
Š
	full_text}
{
y%102 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 1, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%103 = bitcast double* %102 to i64*
.double*8B

	full_text

double* %102
Jload8B@
>
	full_text1
/
-%104 = load i64, i64* %103, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %103
Istore8B>
<
	full_text/
-
+store i64 %104, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %104
'i64*8B

	full_text


i64* %47
¡getelementptr8B
Š
	full_text}
{
y%105 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 1, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%106 = bitcast double* %105 to i64*
.double*8B

	full_text

double* %105
Jload8B@
>
	full_text1
/
-%107 = load i64, i64* %106, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %106
Istore8B>
<
	full_text/
-
+store i64 %107, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %107
'i64*8B

	full_text


i64* %52
¡getelementptr8B
Š
	full_text}
{
y%108 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 1, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%109 = bitcast double* %108 to i64*
.double*8B

	full_text

double* %108
Jload8B@
>
	full_text1
/
-%110 = load i64, i64* %109, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %109
Istore8B>
<
	full_text/
-
+store i64 %110, i64* %57, align 8, !tbaa !8
&i648B

	full_text


i64 %110
'i64*8B

	full_text


i64* %57
|getelementptr8Bi
g
	full_textZ
X
V%111 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Ibitcast8B<
:
	full_text-
+
)%112 = bitcast [5 x double]* %111 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
Kstore8B@
>
	full_text1
/
-store i64 %101, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %101
(i64*8B

	full_text

	i64* %112
Abitcast8B4
2
	full_text%
#
!%113 = bitcast i64 %101 to double
&i648B

	full_text


i64 %101
Œgetelementptr8By
w
	full_textj
h
f%114 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 1
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
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
3%115 = load double, double* %114, align 8, !tbaa !8
.double*8B

	full_text

double* %114
:fmul8B0
.
	full_text!

%116 = fmul double %115, %113
,double8B

	full_text

double %115
,double8B

	full_text

double %113
Œgetelementptr8By
w
	full_textj
h
f%117 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 1
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
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
3%118 = load double, double* %117, align 8, !tbaa !8
.double*8B

	full_text

double* %117
Abitcast8B4
2
	full_text%
#
!%119 = bitcast i64 %110 to double
&i648B

	full_text


i64 %110
:fsub8B0
.
	full_text!

%120 = fsub double %119, %118
,double8B

	full_text

double %119
,double8B

	full_text

double %118
Bfmul8B8
6
	full_text)
'
%%121 = fmul double %120, 4.000000e-01
,double8B

	full_text

double %120
mcall8Bc
a
	full_textT
R
P%122 = tail call double @llvm.fmuladd.f64(double %113, double %116, double %121)
,double8B

	full_text

double %113
,double8B

	full_text

double %116
,double8B

	full_text

double %121
ƒgetelementptr8Bp
n
	full_texta
_
]%123 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Pstore8BE
C
	full_text6
4
2store double %122, double* %123, align 8, !tbaa !8
,double8B

	full_text

double %122
.double*8B

	full_text

double* %123
Abitcast8B4
2
	full_text%
#
!%124 = bitcast i64 %104 to double
&i648B

	full_text


i64 %104
:fmul8B0
.
	full_text!

%125 = fmul double %116, %124
,double8B

	full_text

double %116
,double8B

	full_text

double %124
ƒgetelementptr8Bp
n
	full_texta
_
]%126 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Qstore8BF
D
	full_text7
5
3store double %125, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %125
.double*8B

	full_text

double* %126
Abitcast8B4
2
	full_text%
#
!%127 = bitcast i64 %107 to double
&i648B

	full_text


i64 %107
:fmul8B0
.
	full_text!

%128 = fmul double %116, %127
,double8B

	full_text

double %116
,double8B

	full_text

double %127
ƒgetelementptr8Bp
n
	full_texta
_
]%129 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 3
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Pstore8BE
C
	full_text6
4
2store double %128, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %128
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%130 = fmul double %118, 4.000000e-01
,double8B

	full_text

double %118
Cfsub8B9
7
	full_text*
(
&%131 = fsub double -0.000000e+00, %130
,double8B

	full_text

double %130
ucall8Bk
i
	full_text\
Z
X%132 = tail call double @llvm.fmuladd.f64(double %119, double 1.400000e+00, double %131)
,double8B

	full_text

double %119
,double8B

	full_text

double %131
:fmul8B0
.
	full_text!

%133 = fmul double %116, %132
,double8B

	full_text

double %116
,double8B

	full_text

double %132
ƒgetelementptr8Bp
n
	full_texta
_
]%134 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 4
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Qstore8BF
D
	full_text7
5
3store double %133, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %133
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%135 = fmul double %115, %124
,double8B

	full_text

double %115
,double8B

	full_text

double %124
:fmul8B0
.
	full_text!

%136 = fmul double %115, %127
,double8B

	full_text

double %115
,double8B

	full_text

double %127
:fmul8B0
.
	full_text!

%137 = fmul double %115, %119
,double8B

	full_text

double %115
,double8B

	full_text

double %119
8fmul8B.
,
	full_text

%138 = fmul double %62, %71
+double8B

	full_text


double %62
+double8B

	full_text


double %71
8fmul8B.
,
	full_text

%139 = fmul double %62, %74
+double8B

	full_text


double %62
+double8B

	full_text


double %74
8fmul8B.
,
	full_text

%140 = fmul double %62, %66
+double8B

	full_text


double %62
+double8B

	full_text


double %66
9fsub8B/
-
	full_text 

%141 = fsub double %116, %63
,double8B

	full_text

double %116
+double8B

	full_text


double %63
Hfmul8B>
<
	full_text/
-
+%142 = fmul double %141, 0x4045555555555555
,double8B

	full_text

double %141
„getelementptr8Bq
o
	full_textb
`
^%143 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 1
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %142, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %142
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%144 = fsub double %135, %138
,double8B

	full_text

double %135
,double8B

	full_text

double %138
Bfmul8B8
6
	full_text)
'
%%145 = fmul double %144, 3.200000e+01
,double8B

	full_text

double %144
„getelementptr8Bq
o
	full_textb
`
^%146 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 2
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %145, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %145
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%147 = fsub double %136, %139
,double8B

	full_text

double %136
,double8B

	full_text

double %139
Bfmul8B8
6
	full_text)
'
%%148 = fmul double %147, 3.200000e+01
,double8B

	full_text

double %147
„getelementptr8Bq
o
	full_textb
`
^%149 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 3
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
:fmul8B0
.
	full_text!

%150 = fmul double %135, %135
,double8B

	full_text

double %135
,double8B

	full_text

double %135
mcall8Bc
a
	full_textT
R
P%151 = tail call double @llvm.fmuladd.f64(double %116, double %116, double %150)
,double8B

	full_text

double %116
,double8B

	full_text

double %116
,double8B

	full_text

double %150
mcall8Bc
a
	full_textT
R
P%152 = tail call double @llvm.fmuladd.f64(double %136, double %136, double %151)
,double8B

	full_text

double %136
,double8B

	full_text

double %136
,double8B

	full_text

double %151
:fmul8B0
.
	full_text!

%153 = fmul double %138, %138
,double8B

	full_text

double %138
,double8B

	full_text

double %138
kcall8Ba
_
	full_textR
P
N%154 = tail call double @llvm.fmuladd.f64(double %63, double %63, double %153)
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

double %153
mcall8Bc
a
	full_textT
R
P%155 = tail call double @llvm.fmuladd.f64(double %139, double %139, double %154)
,double8B

	full_text

double %139
,double8B

	full_text

double %139
,double8B

	full_text

double %154
:fsub8B0
.
	full_text!

%156 = fsub double %152, %155
,double8B

	full_text

double %152
,double8B

	full_text

double %155
8fmul8B.
,
	full_text

%157 = fmul double %63, %63
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
&%158 = fsub double -0.000000e+00, %157
,double8B

	full_text

double %157
mcall8Bc
a
	full_textT
R
P%159 = tail call double @llvm.fmuladd.f64(double %116, double %116, double %158)
,double8B

	full_text

double %116
,double8B

	full_text

double %116
,double8B

	full_text

double %158
Hfmul8B>
<
	full_text/
-
+%160 = fmul double %159, 0x4015555555555555
,double8B

	full_text

double %159
{call8Bq
o
	full_textb
`
^%161 = tail call double @llvm.fmuladd.f64(double %156, double 0xC02EB851EB851EB6, double %160)
,double8B

	full_text

double %156
,double8B

	full_text

double %160
:fsub8B0
.
	full_text!

%162 = fsub double %137, %140
,double8B

	full_text

double %137
,double8B

	full_text

double %140
{call8Bq
o
	full_textb
`
^%163 = tail call double @llvm.fmuladd.f64(double %162, double 0x404F5C28F5C28F5B, double %161)
,double8B

	full_text

double %162
,double8B

	full_text

double %161
„getelementptr8Bq
o
	full_textb
`
^%164 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 4
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %163, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %163
.double*8B

	full_text

double* %164
šgetelementptr8B†
ƒ
	full_textv
t
r%165 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
)%166 = bitcast [5 x double]* %165 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %165
Jload8B@
>
	full_text1
/
-%167 = load i64, i64* %166, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %166
}getelementptr8Bj
h
	full_text[
Y
W%168 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ibitcast8B<
:
	full_text-
+
)%169 = bitcast [5 x double]* %168 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %168
Kstore8B@
>
	full_text1
/
-store i64 %167, i64* %169, align 16, !tbaa !8
&i648B

	full_text


i64 %167
(i64*8B

	full_text

	i64* %169
¡getelementptr8B
Š
	full_text}
{
y%170 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 2, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%171 = bitcast double* %170 to i64*
.double*8B

	full_text

double* %170
Jload8B@
>
	full_text1
/
-%172 = load i64, i64* %171, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %171
„getelementptr8Bq
o
	full_textb
`
^%173 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%174 = bitcast double* %173 to i64*
.double*8B

	full_text

double* %173
Jstore8B?
=
	full_text0
.
,store i64 %172, i64* %174, align 8, !tbaa !8
&i648B

	full_text


i64 %172
(i64*8B

	full_text

	i64* %174
¡getelementptr8B
Š
	full_text}
{
y%175 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 2, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%176 = bitcast double* %175 to i64*
.double*8B

	full_text

double* %175
Jload8B@
>
	full_text1
/
-%177 = load i64, i64* %176, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %176
„getelementptr8Bq
o
	full_textb
`
^%178 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%179 = bitcast double* %178 to i64*
.double*8B

	full_text

double* %178
Kstore8B@
>
	full_text1
/
-store i64 %177, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %177
(i64*8B

	full_text

	i64* %179
¡getelementptr8B
Š
	full_text}
{
y%180 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 2, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%181 = bitcast double* %180 to i64*
.double*8B

	full_text

double* %180
Jload8B@
>
	full_text1
/
-%182 = load i64, i64* %181, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %181
„getelementptr8Bq
o
	full_textb
`
^%183 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%184 = bitcast double* %183 to i64*
.double*8B

	full_text

double* %183
Jstore8B?
=
	full_text0
.
,store i64 %182, i64* %184, align 8, !tbaa !8
&i648B

	full_text


i64 %182
(i64*8B

	full_text

	i64* %184
¡getelementptr8B
Š
	full_text}
{
y%185 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 2, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%186 = bitcast double* %185 to i64*
.double*8B

	full_text

double* %185
Jload8B@
>
	full_text1
/
-%187 = load i64, i64* %186, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %186
„getelementptr8Bq
o
	full_textb
`
^%188 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%189 = bitcast double* %188 to i64*
.double*8B

	full_text

double* %188
Kstore8B@
>
	full_text1
/
-store i64 %187, i64* %189, align 16, !tbaa !8
&i648B

	full_text


i64 %187
(i64*8B

	full_text

	i64* %189
„getelementptr8Bq
o
	full_textb
`
^%190 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%191 = bitcast double* %190 to i64*
.double*8B

	full_text

double* %190
Jload8B@
>
	full_text1
/
-%192 = load i64, i64* %191, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %191
„getelementptr8Bq
o
	full_textb
`
^%193 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Nbitcast8BA
?
	full_text2
0
.%194 = bitcast [5 x [5 x double]]* %12 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Kstore8B@
>
	full_text1
/
-store i64 %192, i64* %194, align 16, !tbaa !8
&i648B

	full_text


i64 %192
(i64*8B

	full_text

	i64* %194
Jload8B@
>
	full_text1
/
-%195 = load i64, i64* %85, align 16, !tbaa !8
'i64*8B

	full_text


i64* %85
Jstore8B?
=
	full_text0
.
,store i64 %195, i64* %191, align 8, !tbaa !8
&i648B

	full_text


i64 %195
(i64*8B

	full_text

	i64* %191
Iload8B?
=
	full_text0
.
,%196 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
Jstore8B?
=
	full_text0
.
,store i64 %196, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %196
'i64*8B

	full_text


i64* %85
„getelementptr8Bq
o
	full_textb
`
^%197 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%198 = bitcast double* %197 to i64*
.double*8B

	full_text

double* %197
Istore8B>
<
	full_text/
-
+store i64 %167, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %167
'i64*8B

	full_text


i64* %83
ƒgetelementptr8Bp
n
	full_texta
_
]%199 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%200 = bitcast double* %199 to i64*
.double*8B

	full_text

double* %199
Jload8B@
>
	full_text1
/
-%201 = load i64, i64* %200, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %200
ƒgetelementptr8Bp
n
	full_texta
_
]%202 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Mbitcast8B@
>
	full_text1
/
-%203 = bitcast [3 x [5 x double]]* %8 to i64*
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Kstore8B@
>
	full_text1
/
-store i64 %201, i64* %203, align 16, !tbaa !8
&i648B

	full_text


i64 %201
(i64*8B

	full_text

	i64* %203
ƒgetelementptr8Bp
n
	full_texta
_
]%204 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%205 = bitcast double* %204 to i64*
.double*8B

	full_text

double* %204
Kload8BA
?
	full_text2
0
.%206 = load i64, i64* %205, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %205
Jstore8B?
=
	full_text0
.
,store i64 %206, i64* %200, align 8, !tbaa !8
&i648B

	full_text


i64 %206
(i64*8B

	full_text

	i64* %200
„getelementptr8Bq
o
	full_textb
`
^%207 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 0
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
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
Nbitcast8BA
?
	full_text2
0
.%210 = bitcast [2 x [5 x double]]* %10 to i64*
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Kstore8B@
>
	full_text1
/
-store i64 %209, i64* %210, align 16, !tbaa !8
&i648B

	full_text


i64 %209
(i64*8B

	full_text

	i64* %210
„getelementptr8Bq
o
	full_textb
`
^%211 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%212 = bitcast double* %211 to i64*
.double*8B

	full_text

double* %211
Jload8B@
>
	full_text1
/
-%213 = load i64, i64* %212, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %212
„getelementptr8Bq
o
	full_textb
`
^%214 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%215 = bitcast double* %214 to i64*
.double*8B

	full_text

double* %214
Jstore8B?
=
	full_text0
.
,store i64 %213, i64* %215, align 8, !tbaa !8
&i648B

	full_text


i64 %213
(i64*8B

	full_text

	i64* %215
Iload8B?
=
	full_text0
.
,%216 = load i64, i64* %87, align 8, !tbaa !8
'i64*8B

	full_text


i64* %87
Jstore8B?
=
	full_text0
.
,store i64 %216, i64* %212, align 8, !tbaa !8
&i648B

	full_text


i64 %216
(i64*8B

	full_text

	i64* %212
Iload8B?
=
	full_text0
.
,%217 = load i64, i64* %42, align 8, !tbaa !8
'i64*8B

	full_text


i64* %42
Istore8B>
<
	full_text/
-
+store i64 %217, i64* %87, align 8, !tbaa !8
&i648B

	full_text


i64 %217
'i64*8B

	full_text


i64* %87
Istore8B>
<
	full_text/
-
+store i64 %172, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %172
'i64*8B

	full_text


i64* %42
Bbitcast8B5
3
	full_text&
$
"%218 = bitcast double* %70 to i64*
-double*8B

	full_text

double* %70
Jload8B@
>
	full_text1
/
-%219 = load i64, i64* %218, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %218
ƒgetelementptr8Bp
n
	full_texta
_
]%220 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%221 = bitcast double* %220 to i64*
.double*8B

	full_text

double* %220
Jstore8B?
=
	full_text0
.
,store i64 %219, i64* %221, align 8, !tbaa !8
&i648B

	full_text


i64 %219
(i64*8B

	full_text

	i64* %221
Cbitcast8B6
4
	full_text'
%
#%222 = bitcast double* %123 to i64*
.double*8B

	full_text

double* %123
Jload8B@
>
	full_text1
/
-%223 = load i64, i64* %222, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %222
Jstore8B?
=
	full_text0
.
,store i64 %223, i64* %218, align 8, !tbaa !8
&i648B

	full_text


i64 %223
(i64*8B

	full_text

	i64* %218
Cbitcast8B6
4
	full_text'
%
#%224 = bitcast double* %143 to i64*
.double*8B

	full_text

double* %143
Jload8B@
>
	full_text1
/
-%225 = load i64, i64* %224, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %224
„getelementptr8Bq
o
	full_textb
`
^%226 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 1
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%227 = bitcast double* %226 to i64*
.double*8B

	full_text

double* %226
Jstore8B?
=
	full_text0
.
,store i64 %225, i64* %227, align 8, !tbaa !8
&i648B

	full_text


i64 %225
(i64*8B

	full_text

	i64* %227
„getelementptr8Bq
o
	full_textb
`
^%228 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%229 = bitcast double* %228 to i64*
.double*8B

	full_text

double* %228
Jload8B@
>
	full_text1
/
-%230 = load i64, i64* %229, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %229
„getelementptr8Bq
o
	full_textb
`
^%231 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%232 = bitcast double* %231 to i64*
.double*8B

	full_text

double* %231
Kstore8B@
>
	full_text1
/
-store i64 %230, i64* %232, align 16, !tbaa !8
&i648B

	full_text


i64 %230
(i64*8B

	full_text

	i64* %232
Jload8B@
>
	full_text1
/
-%233 = load i64, i64* %89, align 16, !tbaa !8
'i64*8B

	full_text


i64* %89
Jstore8B?
=
	full_text0
.
,store i64 %233, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %233
(i64*8B

	full_text

	i64* %229
Iload8B?
=
	full_text0
.
,%234 = load i64, i64* %47, align 8, !tbaa !8
'i64*8B

	full_text


i64* %47
Jstore8B?
=
	full_text0
.
,store i64 %234, i64* %89, align 16, !tbaa !8
&i648B

	full_text


i64 %234
'i64*8B

	full_text


i64* %89
Istore8B>
<
	full_text/
-
+store i64 %177, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %177
'i64*8B

	full_text


i64* %47
Bbitcast8B5
3
	full_text&
$
"%235 = bitcast double* %73 to i64*
-double*8B

	full_text

double* %73
Jload8B@
>
	full_text1
/
-%236 = load i64, i64* %235, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %235
ƒgetelementptr8Bp
n
	full_texta
_
]%237 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%238 = bitcast double* %237 to i64*
.double*8B

	full_text

double* %237
Kstore8B@
>
	full_text1
/
-store i64 %236, i64* %238, align 16, !tbaa !8
&i648B

	full_text


i64 %236
(i64*8B

	full_text

	i64* %238
Cbitcast8B6
4
	full_text'
%
#%239 = bitcast double* %126 to i64*
.double*8B

	full_text

double* %126
Kload8BA
?
	full_text2
0
.%240 = load i64, i64* %239, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %239
Jstore8B?
=
	full_text0
.
,store i64 %240, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %240
(i64*8B

	full_text

	i64* %235
Cbitcast8B6
4
	full_text'
%
#%241 = bitcast double* %146 to i64*
.double*8B

	full_text

double* %146
Jload8B@
>
	full_text1
/
-%242 = load i64, i64* %241, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %241
„getelementptr8Bq
o
	full_textb
`
^%243 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 2
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%244 = bitcast double* %243 to i64*
.double*8B

	full_text

double* %243
Kstore8B@
>
	full_text1
/
-store i64 %242, i64* %244, align 16, !tbaa !8
&i648B

	full_text


i64 %242
(i64*8B

	full_text

	i64* %244
„getelementptr8Bq
o
	full_textb
`
^%245 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%246 = bitcast double* %245 to i64*
.double*8B

	full_text

double* %245
Jload8B@
>
	full_text1
/
-%247 = load i64, i64* %246, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %246
„getelementptr8Bq
o
	full_textb
`
^%248 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%249 = bitcast double* %248 to i64*
.double*8B

	full_text

double* %248
Jstore8B?
=
	full_text0
.
,store i64 %247, i64* %249, align 8, !tbaa !8
&i648B

	full_text


i64 %247
(i64*8B

	full_text

	i64* %249
Iload8B?
=
	full_text0
.
,%250 = load i64, i64* %91, align 8, !tbaa !8
'i64*8B

	full_text


i64* %91
Jstore8B?
=
	full_text0
.
,store i64 %250, i64* %246, align 8, !tbaa !8
&i648B

	full_text


i64 %250
(i64*8B

	full_text

	i64* %246
Iload8B?
=
	full_text0
.
,%251 = load i64, i64* %52, align 8, !tbaa !8
'i64*8B

	full_text


i64* %52
Istore8B>
<
	full_text/
-
+store i64 %251, i64* %91, align 8, !tbaa !8
&i648B

	full_text


i64 %251
'i64*8B

	full_text


i64* %91
Istore8B>
<
	full_text/
-
+store i64 %182, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %182
'i64*8B

	full_text


i64* %52
Bbitcast8B5
3
	full_text&
$
"%252 = bitcast double* %76 to i64*
-double*8B

	full_text

double* %76
Jload8B@
>
	full_text1
/
-%253 = load i64, i64* %252, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %252
ƒgetelementptr8Bp
n
	full_texta
_
]%254 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 3
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%255 = bitcast double* %254 to i64*
.double*8B

	full_text

double* %254
Jstore8B?
=
	full_text0
.
,store i64 %253, i64* %255, align 8, !tbaa !8
&i648B

	full_text


i64 %253
(i64*8B

	full_text

	i64* %255
Cbitcast8B6
4
	full_text'
%
#%256 = bitcast double* %129 to i64*
.double*8B

	full_text

double* %129
Jload8B@
>
	full_text1
/
-%257 = load i64, i64* %256, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %256
Jstore8B?
=
	full_text0
.
,store i64 %257, i64* %252, align 8, !tbaa !8
&i648B

	full_text


i64 %257
(i64*8B

	full_text

	i64* %252
Cbitcast8B6
4
	full_text'
%
#%258 = bitcast double* %149 to i64*
.double*8B

	full_text

double* %149
Jload8B@
>
	full_text1
/
-%259 = load i64, i64* %258, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %258
„getelementptr8Bq
o
	full_textb
`
^%260 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 3
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%261 = bitcast double* %260 to i64*
.double*8B

	full_text

double* %260
Jstore8B?
=
	full_text0
.
,store i64 %259, i64* %261, align 8, !tbaa !8
&i648B

	full_text


i64 %259
(i64*8B

	full_text

	i64* %261
„getelementptr8Bq
o
	full_textb
`
^%262 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%263 = bitcast double* %262 to i64*
.double*8B

	full_text

double* %262
Jload8B@
>
	full_text1
/
-%264 = load i64, i64* %263, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %263
„getelementptr8Bq
o
	full_textb
`
^%265 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%266 = bitcast double* %265 to i64*
.double*8B

	full_text

double* %265
Kstore8B@
>
	full_text1
/
-store i64 %264, i64* %266, align 16, !tbaa !8
&i648B

	full_text


i64 %264
(i64*8B

	full_text

	i64* %266
Jload8B@
>
	full_text1
/
-%267 = load i64, i64* %93, align 16, !tbaa !8
'i64*8B

	full_text


i64* %93
Jstore8B?
=
	full_text0
.
,store i64 %267, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %267
(i64*8B

	full_text

	i64* %263
Iload8B?
=
	full_text0
.
,%268 = load i64, i64* %57, align 8, !tbaa !8
'i64*8B

	full_text


i64* %57
Jstore8B?
=
	full_text0
.
,store i64 %268, i64* %93, align 16, !tbaa !8
&i648B

	full_text


i64 %268
'i64*8B

	full_text


i64* %93
Kload8BA
?
	full_text2
0
.%269 = load i64, i64* %189, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %189
Istore8B>
<
	full_text/
-
+store i64 %269, i64* %57, align 8, !tbaa !8
&i648B

	full_text


i64 %269
'i64*8B

	full_text


i64* %57
Bbitcast8B5
3
	full_text&
$
"%270 = bitcast double* %81 to i64*
-double*8B

	full_text

double* %81
Jload8B@
>
	full_text1
/
-%271 = load i64, i64* %270, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %270
ƒgetelementptr8Bp
n
	full_texta
_
]%272 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 4
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%273 = bitcast double* %272 to i64*
.double*8B

	full_text

double* %272
Kstore8B@
>
	full_text1
/
-store i64 %271, i64* %273, align 16, !tbaa !8
&i648B

	full_text


i64 %271
(i64*8B

	full_text

	i64* %273
Cbitcast8B6
4
	full_text'
%
#%274 = bitcast double* %134 to i64*
.double*8B

	full_text

double* %134
Kload8BA
?
	full_text2
0
.%275 = load i64, i64* %274, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %274
Jstore8B?
=
	full_text0
.
,store i64 %275, i64* %270, align 8, !tbaa !8
&i648B

	full_text


i64 %275
(i64*8B

	full_text

	i64* %270
Cbitcast8B6
4
	full_text'
%
#%276 = bitcast double* %164 to i64*
.double*8B

	full_text

double* %164
Jload8B@
>
	full_text1
/
-%277 = load i64, i64* %276, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %276
„getelementptr8Bq
o
	full_textb
`
^%278 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 4
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%279 = bitcast double* %278 to i64*
.double*8B

	full_text

double* %278
Kstore8B@
>
	full_text1
/
-store i64 %277, i64* %279, align 16, !tbaa !8
&i648B

	full_text


i64 %277
(i64*8B

	full_text

	i64* %279
šgetelementptr8B†
ƒ
	full_textv
t
r%280 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
)%281 = bitcast [5 x double]* %280 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %280
Jload8B@
>
	full_text1
/
-%282 = load i64, i64* %281, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %281
Kstore8B@
>
	full_text1
/
-store i64 %282, i64* %169, align 16, !tbaa !8
&i648B

	full_text


i64 %282
(i64*8B

	full_text

	i64* %169
¡getelementptr8B
Š
	full_text}
{
y%283 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 3, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%284 = bitcast double* %283 to i64*
.double*8B

	full_text

double* %283
Jload8B@
>
	full_text1
/
-%285 = load i64, i64* %284, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %284
Jstore8B?
=
	full_text0
.
,store i64 %285, i64* %174, align 8, !tbaa !8
&i648B

	full_text


i64 %285
(i64*8B

	full_text

	i64* %174
¡getelementptr8B
Š
	full_text}
{
y%286 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 3, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%287 = bitcast double* %286 to i64*
.double*8B

	full_text

double* %286
Jload8B@
>
	full_text1
/
-%288 = load i64, i64* %287, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %287
Kstore8B@
>
	full_text1
/
-store i64 %288, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %288
(i64*8B

	full_text

	i64* %179
¡getelementptr8B
Š
	full_text}
{
y%289 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 3, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%290 = bitcast double* %289 to i64*
.double*8B

	full_text

double* %289
Jload8B@
>
	full_text1
/
-%291 = load i64, i64* %290, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %290
Jstore8B?
=
	full_text0
.
,store i64 %291, i64* %184, align 8, !tbaa !8
&i648B

	full_text


i64 %291
(i64*8B

	full_text

	i64* %184
¡getelementptr8B
Š
	full_text}
{
y%292 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 3, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%293 = bitcast double* %292 to i64*
.double*8B

	full_text

double* %292
Jload8B@
>
	full_text1
/
-%294 = load i64, i64* %293, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %293
Kstore8B@
>
	full_text1
/
-store i64 %294, i64* %189, align 16, !tbaa !8
&i648B

	full_text


i64 %294
(i64*8B

	full_text

	i64* %189
Iload8B?
=
	full_text0
.
,%295 = load i64, i64* %42, align 8, !tbaa !8
'i64*8B

	full_text


i64* %42
Kstore8B@
>
	full_text1
/
-store i64 %295, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %295
(i64*8B

	full_text

	i64* %112
Abitcast8B4
2
	full_text%
#
!%296 = bitcast i64 %295 to double
&i648B

	full_text


i64 %295
Œgetelementptr8By
w
	full_textj
h
f%297 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 2
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
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
3%298 = load double, double* %297, align 8, !tbaa !8
.double*8B

	full_text

double* %297
:fmul8B0
.
	full_text!

%299 = fmul double %298, %296
,double8B

	full_text

double %298
,double8B

	full_text

double %296
Œgetelementptr8By
w
	full_textj
h
f%300 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 2
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
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
3%301 = load double, double* %300, align 8, !tbaa !8
.double*8B

	full_text

double* %300
Abitcast8B4
2
	full_text%
#
!%302 = bitcast i64 %269 to double
&i648B

	full_text


i64 %269
:fsub8B0
.
	full_text!

%303 = fsub double %302, %301
,double8B

	full_text

double %302
,double8B

	full_text

double %301
Bfmul8B8
6
	full_text)
'
%%304 = fmul double %303, 4.000000e-01
,double8B

	full_text

double %303
mcall8Bc
a
	full_textT
R
P%305 = tail call double @llvm.fmuladd.f64(double %296, double %299, double %304)
,double8B

	full_text

double %296
,double8B

	full_text

double %299
,double8B

	full_text

double %304
Pstore8BE
C
	full_text6
4
2store double %305, double* %123, align 8, !tbaa !8
,double8B

	full_text

double %305
.double*8B

	full_text

double* %123
Abitcast8B4
2
	full_text%
#
!%306 = bitcast i64 %177 to double
&i648B

	full_text


i64 %177
:fmul8B0
.
	full_text!

%307 = fmul double %299, %306
,double8B

	full_text

double %299
,double8B

	full_text

double %306
Qstore8BF
D
	full_text7
5
3store double %307, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %307
.double*8B

	full_text

double* %126
Abitcast8B4
2
	full_text%
#
!%308 = bitcast i64 %182 to double
&i648B

	full_text


i64 %182
:fmul8B0
.
	full_text!

%309 = fmul double %299, %308
,double8B

	full_text

double %299
,double8B

	full_text

double %308
Pstore8BE
C
	full_text6
4
2store double %309, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %309
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%310 = fmul double %301, 4.000000e-01
,double8B

	full_text

double %301
Cfsub8B9
7
	full_text*
(
&%311 = fsub double -0.000000e+00, %310
,double8B

	full_text

double %310
ucall8Bk
i
	full_text\
Z
X%312 = tail call double @llvm.fmuladd.f64(double %302, double 1.400000e+00, double %311)
,double8B

	full_text

double %302
,double8B

	full_text

double %311
:fmul8B0
.
	full_text!

%313 = fmul double %299, %312
,double8B

	full_text

double %299
,double8B

	full_text

double %312
Qstore8BF
D
	full_text7
5
3store double %313, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %313
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%314 = fmul double %298, %306
,double8B

	full_text

double %298
,double8B

	full_text

double %306
:fmul8B0
.
	full_text!

%315 = fmul double %298, %308
,double8B

	full_text

double %298
,double8B

	full_text

double %308
:fmul8B0
.
	full_text!

%316 = fmul double %298, %302
,double8B

	full_text

double %298
,double8B

	full_text

double %302
Oload8BE
C
	full_text6
4
2%317 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
:fmul8B0
.
	full_text!

%318 = fmul double %115, %317
,double8B

	full_text

double %115
,double8B

	full_text

double %317
Pload8BF
D
	full_text7
5
3%319 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
:fmul8B0
.
	full_text!

%320 = fmul double %115, %319
,double8B

	full_text

double %115
,double8B

	full_text

double %319
Abitcast8B4
2
	full_text%
#
!%321 = bitcast i64 %251 to double
&i648B

	full_text


i64 %251
:fmul8B0
.
	full_text!

%322 = fmul double %115, %321
,double8B

	full_text

double %115
,double8B

	full_text

double %321
Abitcast8B4
2
	full_text%
#
!%323 = bitcast i64 %268 to double
&i648B

	full_text


i64 %268
:fmul8B0
.
	full_text!

%324 = fmul double %115, %323
,double8B

	full_text

double %115
,double8B

	full_text

double %323
:fsub8B0
.
	full_text!

%325 = fsub double %299, %318
,double8B

	full_text

double %299
,double8B

	full_text

double %318
Hfmul8B>
<
	full_text/
-
+%326 = fmul double %325, 0x4045555555555555
,double8B

	full_text

double %325
Pstore8BE
C
	full_text6
4
2store double %326, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %326
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%327 = fsub double %314, %320
,double8B

	full_text

double %314
,double8B

	full_text

double %320
Bfmul8B8
6
	full_text)
'
%%328 = fmul double %327, 3.200000e+01
,double8B

	full_text

double %327
Pstore8BE
C
	full_text6
4
2store double %328, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %328
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%329 = fsub double %315, %322
,double8B

	full_text

double %315
,double8B

	full_text

double %322
Bfmul8B8
6
	full_text)
'
%%330 = fmul double %329, 3.200000e+01
,double8B

	full_text

double %329
Pstore8BE
C
	full_text6
4
2store double %330, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %330
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%331 = fmul double %314, %314
,double8B

	full_text

double %314
,double8B

	full_text

double %314
mcall8Bc
a
	full_textT
R
P%332 = tail call double @llvm.fmuladd.f64(double %299, double %299, double %331)
,double8B

	full_text

double %299
,double8B

	full_text

double %299
,double8B

	full_text

double %331
mcall8Bc
a
	full_textT
R
P%333 = tail call double @llvm.fmuladd.f64(double %315, double %315, double %332)
,double8B

	full_text

double %315
,double8B

	full_text

double %315
,double8B

	full_text

double %332
:fmul8B0
.
	full_text!

%334 = fmul double %320, %320
,double8B

	full_text

double %320
,double8B

	full_text

double %320
mcall8Bc
a
	full_textT
R
P%335 = tail call double @llvm.fmuladd.f64(double %318, double %318, double %334)
,double8B

	full_text

double %318
,double8B

	full_text

double %318
,double8B

	full_text

double %334
mcall8Bc
a
	full_textT
R
P%336 = tail call double @llvm.fmuladd.f64(double %322, double %322, double %335)
,double8B

	full_text

double %322
,double8B

	full_text

double %322
,double8B

	full_text

double %335
:fsub8B0
.
	full_text!

%337 = fsub double %333, %336
,double8B

	full_text

double %333
,double8B

	full_text

double %336
:fmul8B0
.
	full_text!

%338 = fmul double %318, %318
,double8B

	full_text

double %318
,double8B

	full_text

double %318
Cfsub8B9
7
	full_text*
(
&%339 = fsub double -0.000000e+00, %338
,double8B

	full_text

double %338
mcall8Bc
a
	full_textT
R
P%340 = tail call double @llvm.fmuladd.f64(double %299, double %299, double %339)
,double8B

	full_text

double %299
,double8B

	full_text

double %299
,double8B

	full_text

double %339
Hfmul8B>
<
	full_text/
-
+%341 = fmul double %340, 0x4015555555555555
,double8B

	full_text

double %340
{call8Bq
o
	full_textb
`
^%342 = tail call double @llvm.fmuladd.f64(double %337, double 0xC02EB851EB851EB6, double %341)
,double8B

	full_text

double %337
,double8B

	full_text

double %341
:fsub8B0
.
	full_text!

%343 = fsub double %316, %324
,double8B

	full_text

double %316
,double8B

	full_text

double %324
{call8Bq
o
	full_textb
`
^%344 = tail call double @llvm.fmuladd.f64(double %343, double 0x404F5C28F5C28F5B, double %342)
,double8B

	full_text

double %343
,double8B

	full_text

double %342
Pstore8BE
C
	full_text6
4
2store double %344, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %344
.double*8B

	full_text

double* %164
¡getelementptr8B
Š
	full_text}
{
y%345 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 1, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%346 = load double, double* %345, align 8, !tbaa !8
.double*8B

	full_text

double* %345
Qload8BG
E
	full_text8
6
4%347 = load double, double* %202, align 16, !tbaa !8
.double*8B

	full_text

double* %202
:fsub8B0
.
	full_text!

%348 = fsub double %296, %347
,double8B

	full_text

double %296
,double8B

	full_text

double %347
vcall8Bl
j
	full_text]
[
Y%349 = tail call double @llvm.fmuladd.f64(double %348, double -1.600000e+01, double %346)
,double8B

	full_text

double %348
,double8B

	full_text

double %346
¡getelementptr8B
Š
	full_text}
{
y%350 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 1, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%351 = load double, double* %350, align 8, !tbaa !8
.double*8B

	full_text

double* %350
Pload8BF
D
	full_text7
5
3%352 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
:fsub8B0
.
	full_text!

%353 = fsub double %305, %352
,double8B

	full_text

double %305
,double8B

	full_text

double %352
vcall8Bl
j
	full_text]
[
Y%354 = tail call double @llvm.fmuladd.f64(double %353, double -1.600000e+01, double %351)
,double8B

	full_text

double %353
,double8B

	full_text

double %351
¡getelementptr8B
Š
	full_text}
{
y%355 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 1, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%356 = load double, double* %355, align 8, !tbaa !8
.double*8B

	full_text

double* %355
Qload8BG
E
	full_text8
6
4%357 = load double, double* %237, align 16, !tbaa !8
.double*8B

	full_text

double* %237
:fsub8B0
.
	full_text!

%358 = fsub double %307, %357
,double8B

	full_text

double %307
,double8B

	full_text

double %357
vcall8Bl
j
	full_text]
[
Y%359 = tail call double @llvm.fmuladd.f64(double %358, double -1.600000e+01, double %356)
,double8B

	full_text

double %358
,double8B

	full_text

double %356
¡getelementptr8B
Š
	full_text}
{
y%360 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 1, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
Pload8BF
D
	full_text7
5
3%362 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
:fsub8B0
.
	full_text!

%363 = fsub double %309, %362
,double8B

	full_text

double %309
,double8B

	full_text

double %362
vcall8Bl
j
	full_text]
[
Y%364 = tail call double @llvm.fmuladd.f64(double %363, double -1.600000e+01, double %361)
,double8B

	full_text

double %363
,double8B

	full_text

double %361
¡getelementptr8B
Š
	full_text}
{
y%365 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 1, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
Qload8BG
E
	full_text8
6
4%367 = load double, double* %272, align 16, !tbaa !8
.double*8B

	full_text

double* %272
:fsub8B0
.
	full_text!

%368 = fsub double %313, %367
,double8B

	full_text

double %313
,double8B

	full_text

double %367
vcall8Bl
j
	full_text]
[
Y%369 = tail call double @llvm.fmuladd.f64(double %368, double -1.600000e+01, double %366)
,double8B

	full_text

double %368
,double8B

	full_text

double %366
Pload8BF
D
	full_text7
5
3%370 = load double, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
Pload8BF
D
	full_text7
5
3%371 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%372 = tail call double @llvm.fmuladd.f64(double %371, double -2.000000e+00, double %370)
,double8B

	full_text

double %371
,double8B

	full_text

double %370
Oload8BE
C
	full_text6
4
2%373 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%374 = fadd double %372, %373
,double8B

	full_text

double %372
,double8B

	full_text

double %373
ucall8Bk
i
	full_text\
Z
X%375 = tail call double @llvm.fmuladd.f64(double %374, double 7.680000e+02, double %349)
,double8B

	full_text

double %374
,double8B

	full_text

double %349
Pload8BF
D
	full_text7
5
3%376 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
:fsub8B0
.
	full_text!

%377 = fsub double %326, %376
,double8B

	full_text

double %326
,double8B

	full_text

double %376
ucall8Bk
i
	full_text\
Z
X%378 = tail call double @llvm.fmuladd.f64(double %377, double 3.200000e+00, double %354)
,double8B

	full_text

double %377
,double8B

	full_text

double %354
Pload8BF
D
	full_text7
5
3%379 = load double, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
vcall8Bl
j
	full_text]
[
Y%380 = tail call double @llvm.fmuladd.f64(double %317, double -2.000000e+00, double %379)
,double8B

	full_text

double %317
,double8B

	full_text

double %379
:fadd8B0
.
	full_text!

%381 = fadd double %380, %296
,double8B

	full_text

double %380
,double8B

	full_text

double %296
ucall8Bk
i
	full_text\
Z
X%382 = tail call double @llvm.fmuladd.f64(double %381, double 7.680000e+02, double %378)
,double8B

	full_text

double %381
,double8B

	full_text

double %378
Qload8BG
E
	full_text8
6
4%383 = load double, double* %243, align 16, !tbaa !8
.double*8B

	full_text

double* %243
:fsub8B0
.
	full_text!

%384 = fsub double %328, %383
,double8B

	full_text

double %328
,double8B

	full_text

double %383
ucall8Bk
i
	full_text\
Z
X%385 = tail call double @llvm.fmuladd.f64(double %384, double 3.200000e+00, double %359)
,double8B

	full_text

double %384
,double8B

	full_text

double %359
Pload8BF
D
	full_text7
5
3%386 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
vcall8Bl
j
	full_text]
[
Y%387 = tail call double @llvm.fmuladd.f64(double %319, double -2.000000e+00, double %386)
,double8B

	full_text

double %319
,double8B

	full_text

double %386
:fadd8B0
.
	full_text!

%388 = fadd double %387, %306
,double8B

	full_text

double %387
,double8B

	full_text

double %306
ucall8Bk
i
	full_text\
Z
X%389 = tail call double @llvm.fmuladd.f64(double %388, double 7.680000e+02, double %385)
,double8B

	full_text

double %388
,double8B

	full_text

double %385
Pload8BF
D
	full_text7
5
3%390 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
:fsub8B0
.
	full_text!

%391 = fsub double %330, %390
,double8B

	full_text

double %330
,double8B

	full_text

double %390
ucall8Bk
i
	full_text\
Z
X%392 = tail call double @llvm.fmuladd.f64(double %391, double 3.200000e+00, double %364)
,double8B

	full_text

double %391
,double8B

	full_text

double %364
Pload8BF
D
	full_text7
5
3%393 = load double, double* %245, align 8, !tbaa !8
.double*8B

	full_text

double* %245
vcall8Bl
j
	full_text]
[
Y%394 = tail call double @llvm.fmuladd.f64(double %321, double -2.000000e+00, double %393)
,double8B

	full_text

double %321
,double8B

	full_text

double %393
:fadd8B0
.
	full_text!

%395 = fadd double %394, %308
,double8B

	full_text

double %394
,double8B

	full_text

double %308
ucall8Bk
i
	full_text\
Z
X%396 = tail call double @llvm.fmuladd.f64(double %395, double 7.680000e+02, double %392)
,double8B

	full_text

double %395
,double8B

	full_text

double %392
Qload8BG
E
	full_text8
6
4%397 = load double, double* %278, align 16, !tbaa !8
.double*8B

	full_text

double* %278
:fsub8B0
.
	full_text!

%398 = fsub double %344, %397
,double8B

	full_text

double %344
,double8B

	full_text

double %397
ucall8Bk
i
	full_text\
Z
X%399 = tail call double @llvm.fmuladd.f64(double %398, double 3.200000e+00, double %369)
,double8B

	full_text

double %398
,double8B

	full_text

double %369
Pload8BF
D
	full_text7
5
3%400 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
vcall8Bl
j
	full_text]
[
Y%401 = tail call double @llvm.fmuladd.f64(double %323, double -2.000000e+00, double %400)
,double8B

	full_text

double %323
,double8B

	full_text

double %400
:fadd8B0
.
	full_text!

%402 = fadd double %401, %302
,double8B

	full_text

double %401
,double8B

	full_text

double %302
ucall8Bk
i
	full_text\
Z
X%403 = tail call double @llvm.fmuladd.f64(double %402, double 7.680000e+02, double %399)
,double8B

	full_text

double %402
,double8B

	full_text

double %399
kcall8Ba
_
	full_textR
P
N%404 = tail call double @_Z3maxdd(double 7.500000e-01, double 7.500000e-01) #5
ccall8BY
W
	full_textJ
H
F%405 = tail call double @_Z3maxdd(double %404, double 1.000000e+00) #5
,double8B

	full_text

double %404
Bfmul8B8
6
	full_text)
'
%%406 = fmul double %405, 2.500000e-01
,double8B

	full_text

double %405
Cfsub8B9
7
	full_text*
(
&%407 = fsub double -0.000000e+00, %406
,double8B

	full_text

double %406
Bfmul8B8
6
	full_text)
'
%%408 = fmul double %373, 4.000000e+00
,double8B

	full_text

double %373
Cfsub8B9
7
	full_text*
(
&%409 = fsub double -0.000000e+00, %408
,double8B

	full_text

double %408
ucall8Bk
i
	full_text\
Z
X%410 = tail call double @llvm.fmuladd.f64(double %371, double 5.000000e+00, double %409)
,double8B

	full_text

double %371
,double8B

	full_text

double %409
Qload8BG
E
	full_text8
6
4%411 = load double, double* %197, align 16, !tbaa !8
.double*8B

	full_text

double* %197
:fadd8B0
.
	full_text!

%412 = fadd double %411, %410
,double8B

	full_text

double %411
,double8B

	full_text

double %410
mcall8Bc
a
	full_textT
R
P%413 = tail call double @llvm.fmuladd.f64(double %407, double %412, double %375)
,double8B

	full_text

double %407
,double8B

	full_text

double %412
,double8B

	full_text

double %375
Pstore8BE
C
	full_text6
4
2store double %413, double* %345, align 8, !tbaa !8
,double8B

	full_text

double %413
.double*8B

	full_text

double* %345
Oload8BE
C
	full_text6
4
2%414 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
Oload8BE
C
	full_text6
4
2%415 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
Bfmul8B8
6
	full_text)
'
%%416 = fmul double %415, 4.000000e+00
,double8B

	full_text

double %415
Cfsub8B9
7
	full_text*
(
&%417 = fsub double -0.000000e+00, %416
,double8B

	full_text

double %416
ucall8Bk
i
	full_text\
Z
X%418 = tail call double @llvm.fmuladd.f64(double %414, double 5.000000e+00, double %417)
,double8B

	full_text

double %414
,double8B

	full_text

double %417
Pload8BF
D
	full_text7
5
3%419 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
:fadd8B0
.
	full_text!

%420 = fadd double %419, %418
,double8B

	full_text

double %419
,double8B

	full_text

double %418
mcall8Bc
a
	full_textT
R
P%421 = tail call double @llvm.fmuladd.f64(double %407, double %420, double %382)
,double8B

	full_text

double %407
,double8B

	full_text

double %420
,double8B

	full_text

double %382
Pstore8BE
C
	full_text6
4
2store double %421, double* %350, align 8, !tbaa !8
,double8B

	full_text

double %421
.double*8B

	full_text

double* %350
Pload8BF
D
	full_text7
5
3%422 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
Oload8BE
C
	full_text6
4
2%423 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
Bfmul8B8
6
	full_text)
'
%%424 = fmul double %423, 4.000000e+00
,double8B

	full_text

double %423
Cfsub8B9
7
	full_text*
(
&%425 = fsub double -0.000000e+00, %424
,double8B

	full_text

double %424
ucall8Bk
i
	full_text\
Z
X%426 = tail call double @llvm.fmuladd.f64(double %422, double 5.000000e+00, double %425)
,double8B

	full_text

double %422
,double8B

	full_text

double %425
Qload8BG
E
	full_text8
6
4%427 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fadd8B0
.
	full_text!

%428 = fadd double %427, %426
,double8B

	full_text

double %427
,double8B

	full_text

double %426
mcall8Bc
a
	full_textT
R
P%429 = tail call double @llvm.fmuladd.f64(double %407, double %428, double %389)
,double8B

	full_text

double %407
,double8B

	full_text

double %428
,double8B

	full_text

double %389
Pstore8BE
C
	full_text6
4
2store double %429, double* %355, align 8, !tbaa !8
,double8B

	full_text

double %429
.double*8B

	full_text

double* %355
Oload8BE
C
	full_text6
4
2%430 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Oload8BE
C
	full_text6
4
2%431 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
Bfmul8B8
6
	full_text)
'
%%432 = fmul double %431, 4.000000e+00
,double8B

	full_text

double %431
Cfsub8B9
7
	full_text*
(
&%433 = fsub double -0.000000e+00, %432
,double8B

	full_text

double %432
ucall8Bk
i
	full_text\
Z
X%434 = tail call double @llvm.fmuladd.f64(double %430, double 5.000000e+00, double %433)
,double8B

	full_text

double %430
,double8B

	full_text

double %433
Pload8BF
D
	full_text7
5
3%435 = load double, double* %183, align 8, !tbaa !8
.double*8B

	full_text

double* %183
:fadd8B0
.
	full_text!

%436 = fadd double %435, %434
,double8B

	full_text

double %435
,double8B

	full_text

double %434
mcall8Bc
a
	full_textT
R
P%437 = tail call double @llvm.fmuladd.f64(double %407, double %436, double %396)
,double8B

	full_text

double %407
,double8B

	full_text

double %436
,double8B

	full_text

double %396
Pstore8BE
C
	full_text6
4
2store double %437, double* %360, align 8, !tbaa !8
,double8B

	full_text

double %437
.double*8B

	full_text

double* %360
Pload8BF
D
	full_text7
5
3%438 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
Oload8BE
C
	full_text6
4
2%439 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
Bfmul8B8
6
	full_text)
'
%%440 = fmul double %439, 4.000000e+00
,double8B

	full_text

double %439
Cfsub8B9
7
	full_text*
(
&%441 = fsub double -0.000000e+00, %440
,double8B

	full_text

double %440
ucall8Bk
i
	full_text\
Z
X%442 = tail call double @llvm.fmuladd.f64(double %438, double 5.000000e+00, double %441)
,double8B

	full_text

double %438
,double8B

	full_text

double %441
Qload8BG
E
	full_text8
6
4%443 = load double, double* %188, align 16, !tbaa !8
.double*8B

	full_text

double* %188
:fadd8B0
.
	full_text!

%444 = fadd double %443, %442
,double8B

	full_text

double %443
,double8B

	full_text

double %442
mcall8Bc
a
	full_textT
R
P%445 = tail call double @llvm.fmuladd.f64(double %407, double %444, double %403)
,double8B

	full_text

double %407
,double8B

	full_text

double %444
,double8B

	full_text

double %403
Pstore8BE
C
	full_text6
4
2store double %445, double* %365, align 8, !tbaa !8
,double8B

	full_text

double %445
.double*8B

	full_text

double* %365
„getelementptr8Bq
o
	full_textb
`
^%446 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %370, double* %446, align 16, !tbaa !8
,double8B

	full_text

double %370
.double*8B

	full_text

double* %446
Pstore8BE
C
	full_text6
4
2store double %371, double* %190, align 8, !tbaa !8
,double8B

	full_text

double %371
.double*8B

	full_text

double* %190
Pstore8BE
C
	full_text6
4
2store double %373, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %373
-double*8B

	full_text

double* %84
Ostore8BD
B
	full_text5
3
1store double %411, double* %82, align 8, !tbaa !8
,double8B

	full_text

double %411
-double*8B

	full_text

double* %82
Jload8B@
>
	full_text1
/
-%447 = load i64, i64* %200, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %200
Kstore8B@
>
	full_text1
/
-store i64 %447, i64* %203, align 16, !tbaa !8
&i648B

	full_text


i64 %447
(i64*8B

	full_text

	i64* %203
Kload8BA
?
	full_text2
0
.%448 = load i64, i64* %205, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %205
Jstore8B?
=
	full_text0
.
,store i64 %448, i64* %200, align 8, !tbaa !8
&i648B

	full_text


i64 %448
(i64*8B

	full_text

	i64* %200
Jload8B@
>
	full_text1
/
-%449 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Kstore8B@
>
	full_text1
/
-store i64 %449, i64* %210, align 16, !tbaa !8
&i648B

	full_text


i64 %449
(i64*8B

	full_text

	i64* %210
Pstore8BE
C
	full_text6
4
2store double %379, double* %214, align 8, !tbaa !8
,double8B

	full_text

double %379
.double*8B

	full_text

double* %214
Pstore8BE
C
	full_text6
4
2store double %414, double* %211, align 8, !tbaa !8
,double8B

	full_text

double %414
.double*8B

	full_text

double* %211
Ostore8BD
B
	full_text5
3
1store double %415, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %415
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %419, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %419
-double*8B

	full_text

double* %41
Jload8B@
>
	full_text1
/
-%450 = load i64, i64* %218, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %218
Jstore8B?
=
	full_text0
.
,store i64 %450, i64* %221, align 8, !tbaa !8
&i648B

	full_text


i64 %450
(i64*8B

	full_text

	i64* %221
Jload8B@
>
	full_text1
/
-%451 = load i64, i64* %222, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %222
Jstore8B?
=
	full_text0
.
,store i64 %451, i64* %218, align 8, !tbaa !8
&i648B

	full_text


i64 %451
(i64*8B

	full_text

	i64* %218
Jload8B@
>
	full_text1
/
-%452 = load i64, i64* %224, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %224
Jstore8B?
=
	full_text0
.
,store i64 %452, i64* %227, align 8, !tbaa !8
&i648B

	full_text


i64 %452
(i64*8B

	full_text

	i64* %227
Qstore8BF
D
	full_text7
5
3store double %386, double* %231, align 16, !tbaa !8
,double8B

	full_text

double %386
.double*8B

	full_text

double* %231
Pstore8BE
C
	full_text6
4
2store double %422, double* %228, align 8, !tbaa !8
,double8B

	full_text

double %422
.double*8B

	full_text

double* %228
Pstore8BE
C
	full_text6
4
2store double %423, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %423
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %427, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %427
-double*8B

	full_text

double* %46
Jload8B@
>
	full_text1
/
-%453 = load i64, i64* %235, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %235
Kstore8B@
>
	full_text1
/
-store i64 %453, i64* %238, align 16, !tbaa !8
&i648B

	full_text


i64 %453
(i64*8B

	full_text

	i64* %238
Kload8BA
?
	full_text2
0
.%454 = load i64, i64* %239, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %239
Jstore8B?
=
	full_text0
.
,store i64 %454, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %454
(i64*8B

	full_text

	i64* %235
Jload8B@
>
	full_text1
/
-%455 = load i64, i64* %241, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %241
Kstore8B@
>
	full_text1
/
-store i64 %455, i64* %244, align 16, !tbaa !8
&i648B

	full_text


i64 %455
(i64*8B

	full_text

	i64* %244
Jload8B@
>
	full_text1
/
-%456 = load i64, i64* %246, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %246
Jstore8B?
=
	full_text0
.
,store i64 %456, i64* %249, align 8, !tbaa !8
&i648B

	full_text


i64 %456
(i64*8B

	full_text

	i64* %249
Pstore8BE
C
	full_text6
4
2store double %430, double* %245, align 8, !tbaa !8
,double8B

	full_text

double %430
.double*8B

	full_text

double* %245
Ostore8BD
B
	full_text5
3
1store double %431, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %431
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %435, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %435
-double*8B

	full_text

double* %51
Jload8B@
>
	full_text1
/
-%457 = load i64, i64* %252, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %252
Jstore8B?
=
	full_text0
.
,store i64 %457, i64* %255, align 8, !tbaa !8
&i648B

	full_text


i64 %457
(i64*8B

	full_text

	i64* %255
Jload8B@
>
	full_text1
/
-%458 = load i64, i64* %256, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %256
Jstore8B?
=
	full_text0
.
,store i64 %458, i64* %252, align 8, !tbaa !8
&i648B

	full_text


i64 %458
(i64*8B

	full_text

	i64* %252
Jload8B@
>
	full_text1
/
-%459 = load i64, i64* %258, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %258
Jstore8B?
=
	full_text0
.
,store i64 %459, i64* %261, align 8, !tbaa !8
&i648B

	full_text


i64 %459
(i64*8B

	full_text

	i64* %261
Jload8B@
>
	full_text1
/
-%460 = load i64, i64* %263, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %263
Kstore8B@
>
	full_text1
/
-store i64 %460, i64* %266, align 16, !tbaa !8
&i648B

	full_text


i64 %460
(i64*8B

	full_text

	i64* %266
Pstore8BE
C
	full_text6
4
2store double %438, double* %262, align 8, !tbaa !8
,double8B

	full_text

double %438
.double*8B

	full_text

double* %262
Pstore8BE
C
	full_text6
4
2store double %439, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %439
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %443, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %443
-double*8B

	full_text

double* %56
Jload8B@
>
	full_text1
/
-%461 = load i64, i64* %270, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %270
Kstore8B@
>
	full_text1
/
-store i64 %461, i64* %273, align 16, !tbaa !8
&i648B

	full_text


i64 %461
(i64*8B

	full_text

	i64* %273
Kload8BA
?
	full_text2
0
.%462 = load i64, i64* %274, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %274
Jstore8B?
=
	full_text0
.
,store i64 %462, i64* %270, align 8, !tbaa !8
&i648B

	full_text


i64 %462
(i64*8B

	full_text

	i64* %270
Jload8B@
>
	full_text1
/
-%463 = load i64, i64* %276, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %276
Kstore8B@
>
	full_text1
/
-store i64 %463, i64* %279, align 16, !tbaa !8
&i648B

	full_text


i64 %463
(i64*8B

	full_text

	i64* %279
šgetelementptr8B†
ƒ
	full_textv
t
r%464 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
)%465 = bitcast [5 x double]* %464 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %464
Jload8B@
>
	full_text1
/
-%466 = load i64, i64* %465, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %465
Kstore8B@
>
	full_text1
/
-store i64 %466, i64* %169, align 16, !tbaa !8
&i648B

	full_text


i64 %466
(i64*8B

	full_text

	i64* %169
¡getelementptr8B
Š
	full_text}
{
y%467 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 4, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%468 = bitcast double* %467 to i64*
.double*8B

	full_text

double* %467
Jload8B@
>
	full_text1
/
-%469 = load i64, i64* %468, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %468
Jstore8B?
=
	full_text0
.
,store i64 %469, i64* %174, align 8, !tbaa !8
&i648B

	full_text


i64 %469
(i64*8B

	full_text

	i64* %174
¡getelementptr8B
Š
	full_text}
{
y%470 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 4, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%471 = bitcast double* %470 to i64*
.double*8B

	full_text

double* %470
Jload8B@
>
	full_text1
/
-%472 = load i64, i64* %471, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %471
Kstore8B@
>
	full_text1
/
-store i64 %472, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %472
(i64*8B

	full_text

	i64* %179
¡getelementptr8B
Š
	full_text}
{
y%473 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 4, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%474 = bitcast double* %473 to i64*
.double*8B

	full_text

double* %473
Jload8B@
>
	full_text1
/
-%475 = load i64, i64* %474, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %474
Jstore8B?
=
	full_text0
.
,store i64 %475, i64* %184, align 8, !tbaa !8
&i648B

	full_text


i64 %475
(i64*8B

	full_text

	i64* %184
¡getelementptr8B
Š
	full_text}
{
y%476 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 4, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
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
#%477 = bitcast double* %476 to i64*
.double*8B

	full_text

double* %476
Jload8B@
>
	full_text1
/
-%478 = load i64, i64* %477, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %477
Kstore8B@
>
	full_text1
/
-store i64 %478, i64* %189, align 16, !tbaa !8
&i648B

	full_text


i64 %478
(i64*8B

	full_text

	i64* %189
rgetelementptr8B_
]
	full_textP
N
L%479 = getelementptr inbounds [5 x double], [5 x double]* %111, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
Qstore8BF
D
	full_text7
5
3store double %419, double* %479, align 16, !tbaa !8
,double8B

	full_text

double %419
.double*8B

	full_text

double* %479
Œgetelementptr8By
w
	full_textj
h
f%480 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 3
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
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
3%481 = load double, double* %480, align 8, !tbaa !8
.double*8B

	full_text

double* %480
:fmul8B0
.
	full_text!

%482 = fmul double %481, %419
,double8B

	full_text

double %481
,double8B

	full_text

double %419
Œgetelementptr8By
w
	full_textj
h
f%483 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 3
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
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
3%484 = load double, double* %483, align 8, !tbaa !8
.double*8B

	full_text

double* %483
:fsub8B0
.
	full_text!

%485 = fsub double %443, %484
,double8B

	full_text

double %443
,double8B

	full_text

double %484
Bfmul8B8
6
	full_text)
'
%%486 = fmul double %485, 4.000000e-01
,double8B

	full_text

double %485
mcall8Bc
a
	full_textT
R
P%487 = tail call double @llvm.fmuladd.f64(double %419, double %482, double %486)
,double8B

	full_text

double %419
,double8B

	full_text

double %482
,double8B

	full_text

double %486
Pstore8BE
C
	full_text6
4
2store double %487, double* %123, align 8, !tbaa !8
,double8B

	full_text

double %487
.double*8B

	full_text

double* %123
:fmul8B0
.
	full_text!

%488 = fmul double %482, %427
,double8B

	full_text

double %482
,double8B

	full_text

double %427
Qstore8BF
D
	full_text7
5
3store double %488, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %488
.double*8B

	full_text

double* %126
:fmul8B0
.
	full_text!

%489 = fmul double %482, %435
,double8B

	full_text

double %482
,double8B

	full_text

double %435
Pstore8BE
C
	full_text6
4
2store double %489, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %489
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%490 = fmul double %484, 4.000000e-01
,double8B

	full_text

double %484
Cfsub8B9
7
	full_text*
(
&%491 = fsub double -0.000000e+00, %490
,double8B

	full_text

double %490
ucall8Bk
i
	full_text\
Z
X%492 = tail call double @llvm.fmuladd.f64(double %443, double 1.400000e+00, double %491)
,double8B

	full_text

double %443
,double8B

	full_text

double %491
:fmul8B0
.
	full_text!

%493 = fmul double %482, %492
,double8B

	full_text

double %482
,double8B

	full_text

double %492
Qstore8BF
D
	full_text7
5
3store double %493, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %493
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%494 = fmul double %481, %427
,double8B

	full_text

double %481
,double8B

	full_text

double %427
:fmul8B0
.
	full_text!

%495 = fmul double %481, %435
,double8B

	full_text

double %481
,double8B

	full_text

double %435
:fmul8B0
.
	full_text!

%496 = fmul double %481, %443
,double8B

	full_text

double %481
,double8B

	full_text

double %443
Pload8BF
D
	full_text7
5
3%497 = load double, double* %297, align 8, !tbaa !8
.double*8B

	full_text

double* %297
:fmul8B0
.
	full_text!

%498 = fmul double %497, %415
,double8B

	full_text

double %497
,double8B

	full_text

double %415
:fmul8B0
.
	full_text!

%499 = fmul double %497, %423
,double8B

	full_text

double %497
,double8B

	full_text

double %423
:fmul8B0
.
	full_text!

%500 = fmul double %497, %431
,double8B

	full_text

double %497
,double8B

	full_text

double %431
:fmul8B0
.
	full_text!

%501 = fmul double %497, %439
,double8B

	full_text

double %497
,double8B

	full_text

double %439
:fsub8B0
.
	full_text!

%502 = fsub double %482, %498
,double8B

	full_text

double %482
,double8B

	full_text

double %498
Hfmul8B>
<
	full_text/
-
+%503 = fmul double %502, 0x4045555555555555
,double8B

	full_text

double %502
Pstore8BE
C
	full_text6
4
2store double %503, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %503
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%504 = fsub double %494, %499
,double8B

	full_text

double %494
,double8B

	full_text

double %499
Bfmul8B8
6
	full_text)
'
%%505 = fmul double %504, 3.200000e+01
,double8B

	full_text

double %504
Pstore8BE
C
	full_text6
4
2store double %505, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %505
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%506 = fsub double %495, %500
,double8B

	full_text

double %495
,double8B

	full_text

double %500
Bfmul8B8
6
	full_text)
'
%%507 = fmul double %506, 3.200000e+01
,double8B

	full_text

double %506
Pstore8BE
C
	full_text6
4
2store double %507, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %507
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%508 = fmul double %494, %494
,double8B

	full_text

double %494
,double8B

	full_text

double %494
mcall8Bc
a
	full_textT
R
P%509 = tail call double @llvm.fmuladd.f64(double %482, double %482, double %508)
,double8B

	full_text

double %482
,double8B

	full_text

double %482
,double8B

	full_text

double %508
mcall8Bc
a
	full_textT
R
P%510 = tail call double @llvm.fmuladd.f64(double %495, double %495, double %509)
,double8B

	full_text

double %495
,double8B

	full_text

double %495
,double8B

	full_text

double %509
:fmul8B0
.
	full_text!

%511 = fmul double %499, %499
,double8B

	full_text

double %499
,double8B

	full_text

double %499
mcall8Bc
a
	full_textT
R
P%512 = tail call double @llvm.fmuladd.f64(double %498, double %498, double %511)
,double8B

	full_text

double %498
,double8B

	full_text

double %498
,double8B

	full_text

double %511
mcall8Bc
a
	full_textT
R
P%513 = tail call double @llvm.fmuladd.f64(double %500, double %500, double %512)
,double8B

	full_text

double %500
,double8B

	full_text

double %500
,double8B

	full_text

double %512
:fsub8B0
.
	full_text!

%514 = fsub double %510, %513
,double8B

	full_text

double %510
,double8B

	full_text

double %513
:fmul8B0
.
	full_text!

%515 = fmul double %498, %498
,double8B

	full_text

double %498
,double8B

	full_text

double %498
Cfsub8B9
7
	full_text*
(
&%516 = fsub double -0.000000e+00, %515
,double8B

	full_text

double %515
mcall8Bc
a
	full_textT
R
P%517 = tail call double @llvm.fmuladd.f64(double %482, double %482, double %516)
,double8B

	full_text

double %482
,double8B

	full_text

double %482
,double8B

	full_text

double %516
Hfmul8B>
<
	full_text/
-
+%518 = fmul double %517, 0x4015555555555555
,double8B

	full_text

double %517
{call8Bq
o
	full_textb
`
^%519 = tail call double @llvm.fmuladd.f64(double %514, double 0xC02EB851EB851EB6, double %518)
,double8B

	full_text

double %514
,double8B

	full_text

double %518
:fsub8B0
.
	full_text!

%520 = fsub double %496, %501
,double8B

	full_text

double %496
,double8B

	full_text

double %501
{call8Bq
o
	full_textb
`
^%521 = tail call double @llvm.fmuladd.f64(double %520, double 0x404F5C28F5C28F5B, double %519)
,double8B

	full_text

double %520
,double8B

	full_text

double %519
Pstore8BE
C
	full_text6
4
2store double %521, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %521
.double*8B

	full_text

double* %164
¡getelementptr8B
Š
	full_text}
{
y%522 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 2, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%523 = load double, double* %522, align 8, !tbaa !8
.double*8B

	full_text

double* %522
Qload8BG
E
	full_text8
6
4%524 = load double, double* %202, align 16, !tbaa !8
.double*8B

	full_text

double* %202
:fsub8B0
.
	full_text!

%525 = fsub double %419, %524
,double8B

	full_text

double %419
,double8B

	full_text

double %524
vcall8Bl
j
	full_text]
[
Y%526 = tail call double @llvm.fmuladd.f64(double %525, double -1.600000e+01, double %523)
,double8B

	full_text

double %525
,double8B

	full_text

double %523
¡getelementptr8B
Š
	full_text}
{
y%527 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 2, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%528 = load double, double* %527, align 8, !tbaa !8
.double*8B

	full_text

double* %527
Pload8BF
D
	full_text7
5
3%529 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
:fsub8B0
.
	full_text!

%530 = fsub double %487, %529
,double8B

	full_text

double %487
,double8B

	full_text

double %529
vcall8Bl
j
	full_text]
[
Y%531 = tail call double @llvm.fmuladd.f64(double %530, double -1.600000e+01, double %528)
,double8B

	full_text

double %530
,double8B

	full_text

double %528
¡getelementptr8B
Š
	full_text}
{
y%532 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 2, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%533 = load double, double* %532, align 8, !tbaa !8
.double*8B

	full_text

double* %532
Qload8BG
E
	full_text8
6
4%534 = load double, double* %237, align 16, !tbaa !8
.double*8B

	full_text

double* %237
:fsub8B0
.
	full_text!

%535 = fsub double %488, %534
,double8B

	full_text

double %488
,double8B

	full_text

double %534
vcall8Bl
j
	full_text]
[
Y%536 = tail call double @llvm.fmuladd.f64(double %535, double -1.600000e+01, double %533)
,double8B

	full_text

double %535
,double8B

	full_text

double %533
¡getelementptr8B
Š
	full_text}
{
y%537 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 2, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%538 = load double, double* %537, align 8, !tbaa !8
.double*8B

	full_text

double* %537
Pload8BF
D
	full_text7
5
3%539 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
:fsub8B0
.
	full_text!

%540 = fsub double %489, %539
,double8B

	full_text

double %489
,double8B

	full_text

double %539
vcall8Bl
j
	full_text]
[
Y%541 = tail call double @llvm.fmuladd.f64(double %540, double -1.600000e+01, double %538)
,double8B

	full_text

double %540
,double8B

	full_text

double %538
¡getelementptr8B
Š
	full_text}
{
y%542 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 2, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
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
3%543 = load double, double* %542, align 8, !tbaa !8
.double*8B

	full_text

double* %542
Qload8BG
E
	full_text8
6
4%544 = load double, double* %272, align 16, !tbaa !8
.double*8B

	full_text

double* %272
:fsub8B0
.
	full_text!

%545 = fsub double %493, %544
,double8B

	full_text

double %493
,double8B

	full_text

double %544
vcall8Bl
j
	full_text]
[
Y%546 = tail call double @llvm.fmuladd.f64(double %545, double -1.600000e+01, double %543)
,double8B

	full_text

double %545
,double8B

	full_text

double %543
Pload8BF
D
	full_text7
5
3%547 = load double, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
Pload8BF
D
	full_text7
5
3%548 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%549 = tail call double @llvm.fmuladd.f64(double %548, double -2.000000e+00, double %547)
,double8B

	full_text

double %548
,double8B

	full_text

double %547
Oload8BE
C
	full_text6
4
2%550 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%551 = fadd double %549, %550
,double8B

	full_text

double %549
,double8B

	full_text

double %550
ucall8Bk
i
	full_text\
Z
X%552 = tail call double @llvm.fmuladd.f64(double %551, double 7.680000e+02, double %526)
,double8B

	full_text

double %551
,double8B

	full_text

double %526
Pload8BF
D
	full_text7
5
3%553 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
:fsub8B0
.
	full_text!

%554 = fsub double %503, %553
,double8B

	full_text

double %503
,double8B

	full_text

double %553
ucall8Bk
i
	full_text\
Z
X%555 = tail call double @llvm.fmuladd.f64(double %554, double 3.200000e+00, double %531)
,double8B

	full_text

double %554
,double8B

	full_text

double %531
Pload8BF
D
	full_text7
5
3%556 = load double, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
vcall8Bl
j
	full_text]
[
Y%557 = tail call double @llvm.fmuladd.f64(double %415, double -2.000000e+00, double %556)
,double8B

	full_text

double %415
,double8B

	full_text

double %556
:fadd8B0
.
	full_text!

%558 = fadd double %419, %557
,double8B

	full_text

double %419
,double8B

	full_text

double %557
ucall8Bk
i
	full_text\
Z
X%559 = tail call double @llvm.fmuladd.f64(double %558, double 7.680000e+02, double %555)
,double8B

	full_text

double %558
,double8B

	full_text

double %555
Qload8BG
E
	full_text8
6
4%560 = load double, double* %243, align 16, !tbaa !8
.double*8B

	full_text

double* %243
:fsub8B0
.
	full_text!

%561 = fsub double %505, %560
,double8B

	full_text

double %505
,double8B

	full_text

double %560
ucall8Bk
i
	full_text\
Z
X%562 = tail call double @llvm.fmuladd.f64(double %561, double 3.200000e+00, double %536)
,double8B

	full_text

double %561
,double8B

	full_text

double %536
Pload8BF
D
	full_text7
5
3%563 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
vcall8Bl
j
	full_text]
[
Y%564 = tail call double @llvm.fmuladd.f64(double %423, double -2.000000e+00, double %563)
,double8B

	full_text

double %423
,double8B

	full_text

double %563
:fadd8B0
.
	full_text!

%565 = fadd double %427, %564
,double8B

	full_text

double %427
,double8B

	full_text

double %564
ucall8Bk
i
	full_text\
Z
X%566 = tail call double @llvm.fmuladd.f64(double %565, double 7.680000e+02, double %562)
,double8B

	full_text

double %565
,double8B

	full_text

double %562
Pload8BF
D
	full_text7
5
3%567 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
:fsub8B0
.
	full_text!

%568 = fsub double %507, %567
,double8B

	full_text

double %507
,double8B

	full_text

double %567
ucall8Bk
i
	full_text\
Z
X%569 = tail call double @llvm.fmuladd.f64(double %568, double 3.200000e+00, double %541)
,double8B

	full_text

double %568
,double8B

	full_text

double %541
Pload8BF
D
	full_text7
5
3%570 = load double, double* %245, align 8, !tbaa !8
.double*8B

	full_text

double* %245
vcall8Bl
j
	full_text]
[
Y%571 = tail call double @llvm.fmuladd.f64(double %431, double -2.000000e+00, double %570)
,double8B

	full_text

double %431
,double8B

	full_text

double %570
:fadd8B0
.
	full_text!

%572 = fadd double %435, %571
,double8B

	full_text

double %435
,double8B

	full_text

double %571
ucall8Bk
i
	full_text\
Z
X%573 = tail call double @llvm.fmuladd.f64(double %572, double 7.680000e+02, double %569)
,double8B

	full_text

double %572
,double8B

	full_text

double %569
Qload8BG
E
	full_text8
6
4%574 = load double, double* %278, align 16, !tbaa !8
.double*8B

	full_text

double* %278
:fsub8B0
.
	full_text!

%575 = fsub double %521, %574
,double8B

	full_text

double %521
,double8B

	full_text

double %574
ucall8Bk
i
	full_text\
Z
X%576 = tail call double @llvm.fmuladd.f64(double %575, double 3.200000e+00, double %546)
,double8B

	full_text

double %575
,double8B

	full_text

double %546
Pload8BF
D
	full_text7
5
3%577 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
vcall8Bl
j
	full_text]
[
Y%578 = tail call double @llvm.fmuladd.f64(double %439, double -2.000000e+00, double %577)
,double8B

	full_text

double %439
,double8B

	full_text

double %577
:fadd8B0
.
	full_text!

%579 = fadd double %443, %578
,double8B

	full_text

double %443
,double8B

	full_text

double %578
ucall8Bk
i
	full_text\
Z
X%580 = tail call double @llvm.fmuladd.f64(double %579, double 7.680000e+02, double %576)
,double8B

	full_text

double %579
,double8B

	full_text

double %576
Bfmul8B8
6
	full_text)
'
%%581 = fmul double %548, 6.000000e+00
,double8B

	full_text

double %548
vcall8Bl
j
	full_text]
[
Y%582 = tail call double @llvm.fmuladd.f64(double %547, double -4.000000e+00, double %581)
,double8B

	full_text

double %547
,double8B

	full_text

double %581
vcall8Bl
j
	full_text]
[
Y%583 = tail call double @llvm.fmuladd.f64(double %550, double -4.000000e+00, double %582)
,double8B

	full_text

double %550
,double8B

	full_text

double %582
Qload8BG
E
	full_text8
6
4%584 = load double, double* %197, align 16, !tbaa !8
.double*8B

	full_text

double* %197
:fadd8B0
.
	full_text!

%585 = fadd double %584, %583
,double8B

	full_text

double %584
,double8B

	full_text

double %583
mcall8Bc
a
	full_textT
R
P%586 = tail call double @llvm.fmuladd.f64(double %407, double %585, double %552)
,double8B

	full_text

double %407
,double8B

	full_text

double %585
,double8B

	full_text

double %552
Pstore8BE
C
	full_text6
4
2store double %586, double* %522, align 8, !tbaa !8
,double8B

	full_text

double %586
.double*8B

	full_text

double* %522
Oload8BE
C
	full_text6
4
2%587 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
Bfmul8B8
6
	full_text)
'
%%588 = fmul double %587, 6.000000e+00
,double8B

	full_text

double %587
vcall8Bl
j
	full_text]
[
Y%589 = tail call double @llvm.fmuladd.f64(double %556, double -4.000000e+00, double %588)
,double8B

	full_text

double %556
,double8B

	full_text

double %588
Oload8BE
C
	full_text6
4
2%590 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
vcall8Bl
j
	full_text]
[
Y%591 = tail call double @llvm.fmuladd.f64(double %590, double -4.000000e+00, double %589)
,double8B

	full_text

double %590
,double8B

	full_text

double %589
Pload8BF
D
	full_text7
5
3%592 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
:fadd8B0
.
	full_text!

%593 = fadd double %592, %591
,double8B

	full_text

double %592
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%594 = tail call double @llvm.fmuladd.f64(double %407, double %593, double %559)
,double8B

	full_text

double %407
,double8B

	full_text

double %593
,double8B

	full_text

double %559
Pstore8BE
C
	full_text6
4
2store double %594, double* %527, align 8, !tbaa !8
,double8B

	full_text

double %594
.double*8B

	full_text

double* %527
Pload8BF
D
	full_text7
5
3%595 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
Bfmul8B8
6
	full_text)
'
%%596 = fmul double %595, 6.000000e+00
,double8B

	full_text

double %595
vcall8Bl
j
	full_text]
[
Y%597 = tail call double @llvm.fmuladd.f64(double %563, double -4.000000e+00, double %596)
,double8B

	full_text

double %563
,double8B

	full_text

double %596
Oload8BE
C
	full_text6
4
2%598 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
vcall8Bl
j
	full_text]
[
Y%599 = tail call double @llvm.fmuladd.f64(double %598, double -4.000000e+00, double %597)
,double8B

	full_text

double %598
,double8B

	full_text

double %597
Qload8BG
E
	full_text8
6
4%600 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fadd8B0
.
	full_text!

%601 = fadd double %600, %599
,double8B

	full_text

double %600
,double8B

	full_text

double %599
mcall8Bc
a
	full_textT
R
P%602 = tail call double @llvm.fmuladd.f64(double %407, double %601, double %566)
,double8B

	full_text

double %407
,double8B

	full_text

double %601
,double8B

	full_text

double %566
Pstore8BE
C
	full_text6
4
2store double %602, double* %532, align 8, !tbaa !8
,double8B

	full_text

double %602
.double*8B

	full_text

double* %532
Oload8BE
C
	full_text6
4
2%603 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Bfmul8B8
6
	full_text)
'
%%604 = fmul double %603, 6.000000e+00
,double8B

	full_text

double %603
vcall8Bl
j
	full_text]
[
Y%605 = tail call double @llvm.fmuladd.f64(double %570, double -4.000000e+00, double %604)
,double8B

	full_text

double %570
,double8B

	full_text

double %604
Oload8BE
C
	full_text6
4
2%606 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
vcall8Bl
j
	full_text]
[
Y%607 = tail call double @llvm.fmuladd.f64(double %606, double -4.000000e+00, double %605)
,double8B

	full_text

double %606
,double8B

	full_text

double %605
Pload8BF
D
	full_text7
5
3%608 = load double, double* %183, align 8, !tbaa !8
.double*8B

	full_text

double* %183
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
P%610 = tail call double @llvm.fmuladd.f64(double %407, double %609, double %573)
,double8B

	full_text

double %407
,double8B

	full_text

double %609
,double8B

	full_text

double %573
Pstore8BE
C
	full_text6
4
2store double %610, double* %537, align 8, !tbaa !8
,double8B

	full_text

double %610
.double*8B

	full_text

double* %537
Pload8BF
D
	full_text7
5
3%611 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
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
Y%613 = tail call double @llvm.fmuladd.f64(double %577, double -4.000000e+00, double %612)
,double8B

	full_text

double %577
,double8B

	full_text

double %612
Oload8BE
C
	full_text6
4
2%614 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
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
Qload8BG
E
	full_text8
6
4%616 = load double, double* %188, align 16, !tbaa !8
.double*8B

	full_text

double* %188
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
P%618 = tail call double @llvm.fmuladd.f64(double %407, double %617, double %580)
,double8B

	full_text

double %407
,double8B

	full_text

double %617
,double8B

	full_text

double %580
Pstore8BE
C
	full_text6
4
2store double %618, double* %542, align 8, !tbaa !8
,double8B

	full_text

double %618
.double*8B

	full_text

double* %542
5add8B,
*
	full_text

%619 = add nsw i32 %4, -3
6icmp8B,
*
	full_text

%620 = icmp sgt i32 %4, 6
Abitcast8B4
2
	full_text%
#
!%621 = bitcast double %584 to i64
,double8B

	full_text

double %584
Abitcast8B4
2
	full_text%
#
!%622 = bitcast double %556 to i64
,double8B

	full_text

double %556
Abitcast8B4
2
	full_text%
#
!%623 = bitcast double %587 to i64
,double8B

	full_text

double %587
Abitcast8B4
2
	full_text%
#
!%624 = bitcast double %592 to i64
,double8B

	full_text

double %592
Abitcast8B4
2
	full_text%
#
!%625 = bitcast double %563 to i64
,double8B

	full_text

double %563
Abitcast8B4
2
	full_text%
#
!%626 = bitcast double %595 to i64
,double8B

	full_text

double %595
Abitcast8B4
2
	full_text%
#
!%627 = bitcast double %570 to i64
,double8B

	full_text

double %570
Abitcast8B4
2
	full_text%
#
!%628 = bitcast double %603 to i64
,double8B

	full_text

double %603
Abitcast8B4
2
	full_text%
#
!%629 = bitcast double %577 to i64
,double8B

	full_text

double %577
Abitcast8B4
2
	full_text%
#
!%630 = bitcast double %611 to i64
,double8B

	full_text

double %611
=br8B5
3
	full_text&
$
"br i1 %620, label %631, label %843
$i18B

	full_text
	
i1 %620
„getelementptr8Bq
o
	full_textb
`
^%632 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Jload8B@
>
	full_text1
/
-%633 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
rgetelementptr8B_
]
	full_textP
N
L%634 = getelementptr inbounds [5 x double], [5 x double]* %111, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
8zext8B.
,
	full_text

%635 = zext i32 %619 to i64
&i328B

	full_text


i32 %619
Jload8B@
>
	full_text1
/
-%636 = load i64, i64* %200, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %200
Jload8B@
>
	full_text1
/
-%637 = load i64, i64* %218, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %218
Jload8B@
>
	full_text1
/
-%638 = load i64, i64* %235, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %235
Jload8B@
>
	full_text1
/
-%639 = load i64, i64* %252, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %252
Jload8B@
>
	full_text1
/
-%640 = load i64, i64* %270, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %270
(br8B 

	full_text

br label %641
Iphi8B@
>
	full_text1
/
-%642 = phi i64 [ %640, %631 ], [ %677, %641 ]
&i648B

	full_text


i64 %640
&i648B

	full_text


i64 %677
Iphi8B@
>
	full_text1
/
-%643 = phi i64 [ %639, %631 ], [ %675, %641 ]
&i648B

	full_text


i64 %639
&i648B

	full_text


i64 %675
Iphi8B@
>
	full_text1
/
-%644 = phi i64 [ %638, %631 ], [ %673, %641 ]
&i648B

	full_text


i64 %638
&i648B

	full_text


i64 %673
Iphi8B@
>
	full_text1
/
-%645 = phi i64 [ %637, %631 ], [ %671, %641 ]
&i648B

	full_text


i64 %637
&i648B

	full_text


i64 %671
Iphi8B@
>
	full_text1
/
-%646 = phi i64 [ %636, %631 ], [ %670, %641 ]
&i648B

	full_text


i64 %636
&i648B

	full_text


i64 %670
Lphi8BC
A
	full_text4
2
0%647 = phi double [ %616, %631 ], [ %827, %641 ]
,double8B

	full_text

double %616
,double8B

	full_text

double %827
Lphi8BC
A
	full_text4
2
0%648 = phi double [ %614, %631 ], [ %647, %641 ]
,double8B

	full_text

double %614
,double8B

	full_text

double %647
Iphi8B@
>
	full_text1
/
-%649 = phi i64 [ %630, %631 ], [ %840, %641 ]
&i648B

	full_text


i64 %630
&i648B

	full_text


i64 %840
Iphi8B@
>
	full_text1
/
-%650 = phi i64 [ %629, %631 ], [ %839, %641 ]
&i648B

	full_text


i64 %629
&i648B

	full_text


i64 %839
Lphi8BC
A
	full_text4
2
0%651 = phi double [ %608, %631 ], [ %820, %641 ]
,double8B

	full_text

double %608
,double8B

	full_text

double %820
Lphi8BC
A
	full_text4
2
0%652 = phi double [ %606, %631 ], [ %651, %641 ]
,double8B

	full_text

double %606
,double8B

	full_text

double %651
Iphi8B@
>
	full_text1
/
-%653 = phi i64 [ %628, %631 ], [ %838, %641 ]
&i648B

	full_text


i64 %628
&i648B

	full_text


i64 %838
Iphi8B@
>
	full_text1
/
-%654 = phi i64 [ %627, %631 ], [ %837, %641 ]
&i648B

	full_text


i64 %627
&i648B

	full_text


i64 %837
Lphi8BC
A
	full_text4
2
0%655 = phi double [ %600, %631 ], [ %813, %641 ]
,double8B

	full_text

double %600
,double8B

	full_text

double %813
Lphi8BC
A
	full_text4
2
0%656 = phi double [ %598, %631 ], [ %655, %641 ]
,double8B

	full_text

double %598
,double8B

	full_text

double %655
Iphi8B@
>
	full_text1
/
-%657 = phi i64 [ %626, %631 ], [ %836, %641 ]
&i648B

	full_text


i64 %626
&i648B

	full_text


i64 %836
Iphi8B@
>
	full_text1
/
-%658 = phi i64 [ %625, %631 ], [ %835, %641 ]
&i648B

	full_text


i64 %625
&i648B

	full_text


i64 %835
Lphi8BC
A
	full_text4
2
0%659 = phi double [ %592, %631 ], [ %806, %641 ]
,double8B

	full_text

double %592
,double8B

	full_text

double %806
Iphi8B@
>
	full_text1
/
-%660 = phi i64 [ %624, %631 ], [ %834, %641 ]
&i648B

	full_text


i64 %624
&i648B

	full_text


i64 %834
Lphi8BC
A
	full_text4
2
0%661 = phi double [ %590, %631 ], [ %804, %641 ]
,double8B

	full_text

double %590
,double8B

	full_text

double %804
Iphi8B@
>
	full_text1
/
-%662 = phi i64 [ %623, %631 ], [ %833, %641 ]
&i648B

	full_text


i64 %623
&i648B

	full_text


i64 %833
Iphi8B@
>
	full_text1
/
-%663 = phi i64 [ %622, %631 ], [ %832, %641 ]
&i648B

	full_text


i64 %622
&i648B

	full_text


i64 %832
Iphi8B@
>
	full_text1
/
-%664 = phi i64 [ %621, %631 ], [ %831, %641 ]
&i648B

	full_text


i64 %621
&i648B

	full_text


i64 %831
Lphi8BC
A
	full_text4
2
0%665 = phi double [ %550, %631 ], [ %764, %641 ]
,double8B

	full_text

double %550
,double8B

	full_text

double %764
Lphi8BC
A
	full_text4
2
0%666 = phi double [ %548, %631 ], [ %665, %641 ]
,double8B

	full_text

double %548
,double8B

	full_text

double %665
Lphi8BC
A
	full_text4
2
0%667 = phi double [ %547, %631 ], [ %666, %641 ]
,double8B

	full_text

double %547
,double8B

	full_text

double %666
Fphi8B=
;
	full_text.
,
*%668 = phi i64 [ 3, %631 ], [ %669, %641 ]
&i648B

	full_text


i64 %669
:add8B1
/
	full_text"
 
%669 = add nuw nsw i64 %668, 1
&i648B

	full_text


i64 %668
Istore8B>
<
	full_text/
-
+store i64 %664, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %664
'i64*8B

	full_text


i64* %83
Kstore8B@
>
	full_text1
/
-store i64 %646, i64* %203, align 16, !tbaa !8
&i648B

	full_text


i64 %646
(i64*8B

	full_text

	i64* %203
Kload8BA
?
	full_text2
0
.%670 = load i64, i64* %205, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %205
Jstore8B?
=
	full_text0
.
,store i64 %663, i64* %215, align 8, !tbaa !8
&i648B

	full_text


i64 %663
(i64*8B

	full_text

	i64* %215
Jstore8B?
=
	full_text0
.
,store i64 %662, i64* %212, align 8, !tbaa !8
&i648B

	full_text


i64 %662
(i64*8B

	full_text

	i64* %212
Istore8B>
<
	full_text/
-
+store i64 %660, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %660
'i64*8B

	full_text


i64* %42
Jstore8B?
=
	full_text0
.
,store i64 %645, i64* %221, align 8, !tbaa !8
&i648B

	full_text


i64 %645
(i64*8B

	full_text

	i64* %221
Jload8B@
>
	full_text1
/
-%671 = load i64, i64* %222, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %222
Jload8B@
>
	full_text1
/
-%672 = load i64, i64* %224, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %224
Jstore8B?
=
	full_text0
.
,store i64 %672, i64* %227, align 8, !tbaa !8
&i648B

	full_text


i64 %672
(i64*8B

	full_text

	i64* %227
Kstore8B@
>
	full_text1
/
-store i64 %658, i64* %232, align 16, !tbaa !8
&i648B

	full_text


i64 %658
(i64*8B

	full_text

	i64* %232
Jstore8B?
=
	full_text0
.
,store i64 %657, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %657
(i64*8B

	full_text

	i64* %229
Kstore8B@
>
	full_text1
/
-store i64 %644, i64* %238, align 16, !tbaa !8
&i648B

	full_text


i64 %644
(i64*8B

	full_text

	i64* %238
Kload8BA
?
	full_text2
0
.%673 = load i64, i64* %239, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %239
Jload8B@
>
	full_text1
/
-%674 = load i64, i64* %241, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %241
Kstore8B@
>
	full_text1
/
-store i64 %674, i64* %244, align 16, !tbaa !8
&i648B

	full_text


i64 %674
(i64*8B

	full_text

	i64* %244
Jstore8B?
=
	full_text0
.
,store i64 %654, i64* %249, align 8, !tbaa !8
&i648B

	full_text


i64 %654
(i64*8B

	full_text

	i64* %249
Jstore8B?
=
	full_text0
.
,store i64 %653, i64* %246, align 8, !tbaa !8
&i648B

	full_text


i64 %653
(i64*8B

	full_text

	i64* %246
Jstore8B?
=
	full_text0
.
,store i64 %643, i64* %255, align 8, !tbaa !8
&i648B

	full_text


i64 %643
(i64*8B

	full_text

	i64* %255
Jload8B@
>
	full_text1
/
-%675 = load i64, i64* %256, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %256
Jload8B@
>
	full_text1
/
-%676 = load i64, i64* %258, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %258
Jstore8B?
=
	full_text0
.
,store i64 %676, i64* %261, align 8, !tbaa !8
&i648B

	full_text


i64 %676
(i64*8B

	full_text

	i64* %261
Kstore8B@
>
	full_text1
/
-store i64 %650, i64* %266, align 16, !tbaa !8
&i648B

	full_text


i64 %650
(i64*8B

	full_text

	i64* %266
Jstore8B?
=
	full_text0
.
,store i64 %649, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %649
(i64*8B

	full_text

	i64* %263
Kstore8B@
>
	full_text1
/
-store i64 %642, i64* %273, align 16, !tbaa !8
&i648B

	full_text


i64 %642
(i64*8B

	full_text

	i64* %273
Kload8BA
?
	full_text2
0
.%677 = load i64, i64* %274, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %274
Jload8B@
>
	full_text1
/
-%678 = load i64, i64* %276, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %276
Kstore8B@
>
	full_text1
/
-store i64 %678, i64* %279, align 16, !tbaa !8
&i648B

	full_text


i64 %678
(i64*8B

	full_text

	i64* %279
:add8B1
/
	full_text"
 
%679 = add nuw nsw i64 %668, 2
&i648B

	full_text


i64 %668
getelementptr8B‰
†
	full_texty
w
u%680 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %679
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %679
Ibitcast8B<
:
	full_text-
+
)%681 = bitcast [5 x double]* %680 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %680
Jload8B@
>
	full_text1
/
-%682 = load i64, i64* %681, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %681
Kstore8B@
>
	full_text1
/
-store i64 %682, i64* %169, align 16, !tbaa !8
&i648B

	full_text


i64 %682
(i64*8B

	full_text

	i64* %169
¥getelementptr8B‘

	full_text€
~
|%683 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %679, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %679
Cbitcast8B6
4
	full_text'
%
#%684 = bitcast double* %683 to i64*
.double*8B

	full_text

double* %683
Jload8B@
>
	full_text1
/
-%685 = load i64, i64* %684, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %684
Jstore8B?
=
	full_text0
.
,store i64 %685, i64* %174, align 8, !tbaa !8
&i648B

	full_text


i64 %685
(i64*8B

	full_text

	i64* %174
¥getelementptr8B‘

	full_text€
~
|%686 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %679, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %679
Cbitcast8B6
4
	full_text'
%
#%687 = bitcast double* %686 to i64*
.double*8B

	full_text

double* %686
Jload8B@
>
	full_text1
/
-%688 = load i64, i64* %687, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %687
Kstore8B@
>
	full_text1
/
-store i64 %688, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %688
(i64*8B

	full_text

	i64* %179
¥getelementptr8B‘

	full_text€
~
|%689 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %679, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %679
Cbitcast8B6
4
	full_text'
%
#%690 = bitcast double* %689 to i64*
.double*8B

	full_text

double* %689
Jload8B@
>
	full_text1
/
-%691 = load i64, i64* %690, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %690
Jstore8B?
=
	full_text0
.
,store i64 %691, i64* %184, align 8, !tbaa !8
&i648B

	full_text


i64 %691
(i64*8B

	full_text

	i64* %184
¥getelementptr8B‘

	full_text€
~
|%692 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %679, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %679
Cbitcast8B6
4
	full_text'
%
#%693 = bitcast double* %692 to i64*
.double*8B

	full_text

double* %692
Jload8B@
>
	full_text1
/
-%694 = load i64, i64* %693, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %693
Kstore8B@
>
	full_text1
/
-store i64 %694, i64* %189, align 16, !tbaa !8
&i648B

	full_text


i64 %694
(i64*8B

	full_text

	i64* %189
Qstore8BF
D
	full_text7
5
3store double %659, double* %634, align 16, !tbaa !8
,double8B

	full_text

double %659
.double*8B

	full_text

double* %634
getelementptr8B|
z
	full_textm
k
i%695 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 %669
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %669
Pload8BF
D
	full_text7
5
3%696 = load double, double* %695, align 8, !tbaa !8
.double*8B

	full_text

double* %695
:fmul8B0
.
	full_text!

%697 = fmul double %696, %659
,double8B

	full_text

double %696
,double8B

	full_text

double %659
getelementptr8B|
z
	full_textm
k
i%698 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 %669
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %669
Pload8BF
D
	full_text7
5
3%699 = load double, double* %698, align 8, !tbaa !8
.double*8B

	full_text

double* %698
:fsub8B0
.
	full_text!

%700 = fsub double %647, %699
,double8B

	full_text

double %647
,double8B

	full_text

double %699
Bfmul8B8
6
	full_text)
'
%%701 = fmul double %700, 4.000000e-01
,double8B

	full_text

double %700
mcall8Bc
a
	full_textT
R
P%702 = tail call double @llvm.fmuladd.f64(double %659, double %697, double %701)
,double8B

	full_text

double %659
,double8B

	full_text

double %697
,double8B

	full_text

double %701
Pstore8BE
C
	full_text6
4
2store double %702, double* %123, align 8, !tbaa !8
,double8B

	full_text

double %702
.double*8B

	full_text

double* %123
:fmul8B0
.
	full_text!

%703 = fmul double %697, %655
,double8B

	full_text

double %697
,double8B

	full_text

double %655
Qstore8BF
D
	full_text7
5
3store double %703, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %703
.double*8B

	full_text

double* %126
:fmul8B0
.
	full_text!

%704 = fmul double %697, %651
,double8B

	full_text

double %697
,double8B

	full_text

double %651
Pstore8BE
C
	full_text6
4
2store double %704, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %704
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%705 = fmul double %699, 4.000000e-01
,double8B

	full_text

double %699
Cfsub8B9
7
	full_text*
(
&%706 = fsub double -0.000000e+00, %705
,double8B

	full_text

double %705
ucall8Bk
i
	full_text\
Z
X%707 = tail call double @llvm.fmuladd.f64(double %647, double 1.400000e+00, double %706)
,double8B

	full_text

double %647
,double8B

	full_text

double %706
:fmul8B0
.
	full_text!

%708 = fmul double %697, %707
,double8B

	full_text

double %697
,double8B

	full_text

double %707
Qstore8BF
D
	full_text7
5
3store double %708, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %708
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%709 = fmul double %696, %655
,double8B

	full_text

double %696
,double8B

	full_text

double %655
:fmul8B0
.
	full_text!

%710 = fmul double %696, %651
,double8B

	full_text

double %696
,double8B

	full_text

double %651
:fmul8B0
.
	full_text!

%711 = fmul double %696, %647
,double8B

	full_text

double %696
,double8B

	full_text

double %647
getelementptr8B|
z
	full_textm
k
i%712 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 %668
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %668
Pload8BF
D
	full_text7
5
3%713 = load double, double* %712, align 8, !tbaa !8
.double*8B

	full_text

double* %712
:fmul8B0
.
	full_text!

%714 = fmul double %713, %661
,double8B

	full_text

double %713
,double8B

	full_text

double %661
:fmul8B0
.
	full_text!

%715 = fmul double %713, %656
,double8B

	full_text

double %713
,double8B

	full_text

double %656
:fmul8B0
.
	full_text!

%716 = fmul double %713, %652
,double8B

	full_text

double %713
,double8B

	full_text

double %652
:fmul8B0
.
	full_text!

%717 = fmul double %713, %648
,double8B

	full_text

double %713
,double8B

	full_text

double %648
:fsub8B0
.
	full_text!

%718 = fsub double %697, %714
,double8B

	full_text

double %697
,double8B

	full_text

double %714
Hfmul8B>
<
	full_text/
-
+%719 = fmul double %718, 0x4045555555555555
,double8B

	full_text

double %718
Pstore8BE
C
	full_text6
4
2store double %719, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %719
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%720 = fsub double %709, %715
,double8B

	full_text

double %709
,double8B

	full_text

double %715
Bfmul8B8
6
	full_text)
'
%%721 = fmul double %720, 3.200000e+01
,double8B

	full_text

double %720
Pstore8BE
C
	full_text6
4
2store double %721, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %721
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%722 = fsub double %710, %716
,double8B

	full_text

double %710
,double8B

	full_text

double %716
Bfmul8B8
6
	full_text)
'
%%723 = fmul double %722, 3.200000e+01
,double8B

	full_text

double %722
Pstore8BE
C
	full_text6
4
2store double %723, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %723
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%724 = fmul double %709, %709
,double8B

	full_text

double %709
,double8B

	full_text

double %709
mcall8Bc
a
	full_textT
R
P%725 = tail call double @llvm.fmuladd.f64(double %697, double %697, double %724)
,double8B

	full_text

double %697
,double8B

	full_text

double %697
,double8B

	full_text

double %724
mcall8Bc
a
	full_textT
R
P%726 = tail call double @llvm.fmuladd.f64(double %710, double %710, double %725)
,double8B

	full_text

double %710
,double8B

	full_text

double %710
,double8B

	full_text

double %725
:fmul8B0
.
	full_text!

%727 = fmul double %715, %715
,double8B

	full_text

double %715
,double8B

	full_text

double %715
mcall8Bc
a
	full_textT
R
P%728 = tail call double @llvm.fmuladd.f64(double %714, double %714, double %727)
,double8B

	full_text

double %714
,double8B

	full_text

double %714
,double8B

	full_text

double %727
mcall8Bc
a
	full_textT
R
P%729 = tail call double @llvm.fmuladd.f64(double %716, double %716, double %728)
,double8B

	full_text

double %716
,double8B

	full_text

double %716
,double8B

	full_text

double %728
:fsub8B0
.
	full_text!

%730 = fsub double %726, %729
,double8B

	full_text

double %726
,double8B

	full_text

double %729
:fmul8B0
.
	full_text!

%731 = fmul double %714, %714
,double8B

	full_text

double %714
,double8B

	full_text

double %714
Cfsub8B9
7
	full_text*
(
&%732 = fsub double -0.000000e+00, %731
,double8B

	full_text

double %731
mcall8Bc
a
	full_textT
R
P%733 = tail call double @llvm.fmuladd.f64(double %697, double %697, double %732)
,double8B

	full_text

double %697
,double8B

	full_text

double %697
,double8B

	full_text

double %732
Hfmul8B>
<
	full_text/
-
+%734 = fmul double %733, 0x4015555555555555
,double8B

	full_text

double %733
{call8Bq
o
	full_textb
`
^%735 = tail call double @llvm.fmuladd.f64(double %730, double 0xC02EB851EB851EB6, double %734)
,double8B

	full_text

double %730
,double8B

	full_text

double %734
:fsub8B0
.
	full_text!

%736 = fsub double %711, %717
,double8B

	full_text

double %711
,double8B

	full_text

double %717
{call8Bq
o
	full_textb
`
^%737 = tail call double @llvm.fmuladd.f64(double %736, double 0x404F5C28F5C28F5B, double %735)
,double8B

	full_text

double %736
,double8B

	full_text

double %735
Pstore8BE
C
	full_text6
4
2store double %737, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %737
.double*8B

	full_text

double* %164
¥getelementptr8B‘

	full_text€
~
|%738 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %668, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %668
Pload8BF
D
	full_text7
5
3%739 = load double, double* %738, align 8, !tbaa !8
.double*8B

	full_text

double* %738
Qload8BG
E
	full_text8
6
4%740 = load double, double* %202, align 16, !tbaa !8
.double*8B

	full_text

double* %202
:fsub8B0
.
	full_text!

%741 = fsub double %659, %740
,double8B

	full_text

double %659
,double8B

	full_text

double %740
vcall8Bl
j
	full_text]
[
Y%742 = tail call double @llvm.fmuladd.f64(double %741, double -1.600000e+01, double %739)
,double8B

	full_text

double %741
,double8B

	full_text

double %739
¥getelementptr8B‘

	full_text€
~
|%743 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %668, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %668
Pload8BF
D
	full_text7
5
3%744 = load double, double* %743, align 8, !tbaa !8
.double*8B

	full_text

double* %743
Pload8BF
D
	full_text7
5
3%745 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
:fsub8B0
.
	full_text!

%746 = fsub double %702, %745
,double8B

	full_text

double %702
,double8B

	full_text

double %745
vcall8Bl
j
	full_text]
[
Y%747 = tail call double @llvm.fmuladd.f64(double %746, double -1.600000e+01, double %744)
,double8B

	full_text

double %746
,double8B

	full_text

double %744
¥getelementptr8B‘

	full_text€
~
|%748 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %668, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %668
Pload8BF
D
	full_text7
5
3%749 = load double, double* %748, align 8, !tbaa !8
.double*8B

	full_text

double* %748
Qload8BG
E
	full_text8
6
4%750 = load double, double* %237, align 16, !tbaa !8
.double*8B

	full_text

double* %237
:fsub8B0
.
	full_text!

%751 = fsub double %703, %750
,double8B

	full_text

double %703
,double8B

	full_text

double %750
vcall8Bl
j
	full_text]
[
Y%752 = tail call double @llvm.fmuladd.f64(double %751, double -1.600000e+01, double %749)
,double8B

	full_text

double %751
,double8B

	full_text

double %749
¥getelementptr8B‘

	full_text€
~
|%753 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %668, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %668
Pload8BF
D
	full_text7
5
3%754 = load double, double* %753, align 8, !tbaa !8
.double*8B

	full_text

double* %753
Pload8BF
D
	full_text7
5
3%755 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
:fsub8B0
.
	full_text!

%756 = fsub double %704, %755
,double8B

	full_text

double %704
,double8B

	full_text

double %755
vcall8Bl
j
	full_text]
[
Y%757 = tail call double @llvm.fmuladd.f64(double %756, double -1.600000e+01, double %754)
,double8B

	full_text

double %756
,double8B

	full_text

double %754
¥getelementptr8B‘

	full_text€
~
|%758 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %668, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %668
Pload8BF
D
	full_text7
5
3%759 = load double, double* %758, align 8, !tbaa !8
.double*8B

	full_text

double* %758
Qload8BG
E
	full_text8
6
4%760 = load double, double* %272, align 16, !tbaa !8
.double*8B

	full_text

double* %272
:fsub8B0
.
	full_text!

%761 = fsub double %708, %760
,double8B

	full_text

double %708
,double8B

	full_text

double %760
vcall8Bl
j
	full_text]
[
Y%762 = tail call double @llvm.fmuladd.f64(double %761, double -1.600000e+01, double %759)
,double8B

	full_text

double %761
,double8B

	full_text

double %759
vcall8Bl
j
	full_text]
[
Y%763 = tail call double @llvm.fmuladd.f64(double %665, double -2.000000e+00, double %666)
,double8B

	full_text

double %665
,double8B

	full_text

double %666
Oload8BE
C
	full_text6
4
2%764 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%765 = fadd double %763, %764
,double8B

	full_text

double %763
,double8B

	full_text

double %764
ucall8Bk
i
	full_text\
Z
X%766 = tail call double @llvm.fmuladd.f64(double %765, double 7.680000e+02, double %742)
,double8B

	full_text

double %765
,double8B

	full_text

double %742
Pload8BF
D
	full_text7
5
3%767 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
:fsub8B0
.
	full_text!

%768 = fsub double %719, %767
,double8B

	full_text

double %719
,double8B

	full_text

double %767
ucall8Bk
i
	full_text\
Z
X%769 = tail call double @llvm.fmuladd.f64(double %768, double 3.200000e+00, double %747)
,double8B

	full_text

double %768
,double8B

	full_text

double %747
Pload8BF
D
	full_text7
5
3%770 = load double, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
vcall8Bl
j
	full_text]
[
Y%771 = tail call double @llvm.fmuladd.f64(double %661, double -2.000000e+00, double %770)
,double8B

	full_text

double %661
,double8B

	full_text

double %770
:fadd8B0
.
	full_text!

%772 = fadd double %659, %771
,double8B

	full_text

double %659
,double8B

	full_text

double %771
ucall8Bk
i
	full_text\
Z
X%773 = tail call double @llvm.fmuladd.f64(double %772, double 7.680000e+02, double %769)
,double8B

	full_text

double %772
,double8B

	full_text

double %769
Qload8BG
E
	full_text8
6
4%774 = load double, double* %243, align 16, !tbaa !8
.double*8B

	full_text

double* %243
:fsub8B0
.
	full_text!

%775 = fsub double %721, %774
,double8B

	full_text

double %721
,double8B

	full_text

double %774
ucall8Bk
i
	full_text\
Z
X%776 = tail call double @llvm.fmuladd.f64(double %775, double 3.200000e+00, double %752)
,double8B

	full_text

double %775
,double8B

	full_text

double %752
Pload8BF
D
	full_text7
5
3%777 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
vcall8Bl
j
	full_text]
[
Y%778 = tail call double @llvm.fmuladd.f64(double %656, double -2.000000e+00, double %777)
,double8B

	full_text

double %656
,double8B

	full_text

double %777
:fadd8B0
.
	full_text!

%779 = fadd double %655, %778
,double8B

	full_text

double %655
,double8B

	full_text

double %778
ucall8Bk
i
	full_text\
Z
X%780 = tail call double @llvm.fmuladd.f64(double %779, double 7.680000e+02, double %776)
,double8B

	full_text

double %779
,double8B

	full_text

double %776
Pload8BF
D
	full_text7
5
3%781 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
:fsub8B0
.
	full_text!

%782 = fsub double %723, %781
,double8B

	full_text

double %723
,double8B

	full_text

double %781
ucall8Bk
i
	full_text\
Z
X%783 = tail call double @llvm.fmuladd.f64(double %782, double 3.200000e+00, double %757)
,double8B

	full_text

double %782
,double8B

	full_text

double %757
Pload8BF
D
	full_text7
5
3%784 = load double, double* %245, align 8, !tbaa !8
.double*8B

	full_text

double* %245
vcall8Bl
j
	full_text]
[
Y%785 = tail call double @llvm.fmuladd.f64(double %652, double -2.000000e+00, double %784)
,double8B

	full_text

double %652
,double8B

	full_text

double %784
:fadd8B0
.
	full_text!

%786 = fadd double %651, %785
,double8B

	full_text

double %651
,double8B

	full_text

double %785
ucall8Bk
i
	full_text\
Z
X%787 = tail call double @llvm.fmuladd.f64(double %786, double 7.680000e+02, double %783)
,double8B

	full_text

double %786
,double8B

	full_text

double %783
Qload8BG
E
	full_text8
6
4%788 = load double, double* %278, align 16, !tbaa !8
.double*8B

	full_text

double* %278
:fsub8B0
.
	full_text!

%789 = fsub double %737, %788
,double8B

	full_text

double %737
,double8B

	full_text

double %788
ucall8Bk
i
	full_text\
Z
X%790 = tail call double @llvm.fmuladd.f64(double %789, double 3.200000e+00, double %762)
,double8B

	full_text

double %789
,double8B

	full_text

double %762
Pload8BF
D
	full_text7
5
3%791 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
vcall8Bl
j
	full_text]
[
Y%792 = tail call double @llvm.fmuladd.f64(double %648, double -2.000000e+00, double %791)
,double8B

	full_text

double %648
,double8B

	full_text

double %791
:fadd8B0
.
	full_text!

%793 = fadd double %647, %792
,double8B

	full_text

double %647
,double8B

	full_text

double %792
ucall8Bk
i
	full_text\
Z
X%794 = tail call double @llvm.fmuladd.f64(double %793, double 7.680000e+02, double %790)
,double8B

	full_text

double %793
,double8B

	full_text

double %790
vcall8Bl
j
	full_text]
[
Y%795 = tail call double @llvm.fmuladd.f64(double %666, double -4.000000e+00, double %667)
,double8B

	full_text

double %666
,double8B

	full_text

double %667
ucall8Bk
i
	full_text\
Z
X%796 = tail call double @llvm.fmuladd.f64(double %665, double 6.000000e+00, double %795)
,double8B

	full_text

double %665
,double8B

	full_text

double %795
vcall8Bl
j
	full_text]
[
Y%797 = tail call double @llvm.fmuladd.f64(double %764, double -4.000000e+00, double %796)
,double8B

	full_text

double %764
,double8B

	full_text

double %796
Qload8BG
E
	full_text8
6
4%798 = load double, double* %197, align 16, !tbaa !8
.double*8B

	full_text

double* %197
:fadd8B0
.
	full_text!

%799 = fadd double %797, %798
,double8B

	full_text

double %797
,double8B

	full_text

double %798
mcall8Bc
a
	full_textT
R
P%800 = tail call double @llvm.fmuladd.f64(double %407, double %799, double %766)
,double8B

	full_text

double %407
,double8B

	full_text

double %799
,double8B

	full_text

double %766
Pstore8BE
C
	full_text6
4
2store double %800, double* %738, align 8, !tbaa !8
,double8B

	full_text

double %800
.double*8B

	full_text

double* %738
Pload8BF
D
	full_text7
5
3%801 = load double, double* %214, align 8, !tbaa !8
.double*8B

	full_text

double* %214
vcall8Bl
j
	full_text]
[
Y%802 = tail call double @llvm.fmuladd.f64(double %770, double -4.000000e+00, double %801)
,double8B

	full_text

double %770
,double8B

	full_text

double %801
ucall8Bk
i
	full_text\
Z
X%803 = tail call double @llvm.fmuladd.f64(double %661, double 6.000000e+00, double %802)
,double8B

	full_text

double %661
,double8B

	full_text

double %802
Oload8BE
C
	full_text6
4
2%804 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
vcall8Bl
j
	full_text]
[
Y%805 = tail call double @llvm.fmuladd.f64(double %804, double -4.000000e+00, double %803)
,double8B

	full_text

double %804
,double8B

	full_text

double %803
Pload8BF
D
	full_text7
5
3%806 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
:fadd8B0
.
	full_text!

%807 = fadd double %805, %806
,double8B

	full_text

double %805
,double8B

	full_text

double %806
mcall8Bc
a
	full_textT
R
P%808 = tail call double @llvm.fmuladd.f64(double %407, double %807, double %773)
,double8B

	full_text

double %407
,double8B

	full_text

double %807
,double8B

	full_text

double %773
Pstore8BE
C
	full_text6
4
2store double %808, double* %743, align 8, !tbaa !8
,double8B

	full_text

double %808
.double*8B

	full_text

double* %743
Qload8BG
E
	full_text8
6
4%809 = load double, double* %231, align 16, !tbaa !8
.double*8B

	full_text

double* %231
vcall8Bl
j
	full_text]
[
Y%810 = tail call double @llvm.fmuladd.f64(double %777, double -4.000000e+00, double %809)
,double8B

	full_text

double %777
,double8B

	full_text

double %809
ucall8Bk
i
	full_text\
Z
X%811 = tail call double @llvm.fmuladd.f64(double %656, double 6.000000e+00, double %810)
,double8B

	full_text

double %656
,double8B

	full_text

double %810
vcall8Bl
j
	full_text]
[
Y%812 = tail call double @llvm.fmuladd.f64(double %655, double -4.000000e+00, double %811)
,double8B

	full_text

double %655
,double8B

	full_text

double %811
Qload8BG
E
	full_text8
6
4%813 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fadd8B0
.
	full_text!

%814 = fadd double %812, %813
,double8B

	full_text

double %812
,double8B

	full_text

double %813
mcall8Bc
a
	full_textT
R
P%815 = tail call double @llvm.fmuladd.f64(double %407, double %814, double %780)
,double8B

	full_text

double %407
,double8B

	full_text

double %814
,double8B

	full_text

double %780
Pstore8BE
C
	full_text6
4
2store double %815, double* %748, align 8, !tbaa !8
,double8B

	full_text

double %815
.double*8B

	full_text

double* %748
Pload8BF
D
	full_text7
5
3%816 = load double, double* %248, align 8, !tbaa !8
.double*8B

	full_text

double* %248
vcall8Bl
j
	full_text]
[
Y%817 = tail call double @llvm.fmuladd.f64(double %784, double -4.000000e+00, double %816)
,double8B

	full_text

double %784
,double8B

	full_text

double %816
ucall8Bk
i
	full_text\
Z
X%818 = tail call double @llvm.fmuladd.f64(double %652, double 6.000000e+00, double %817)
,double8B

	full_text

double %652
,double8B

	full_text

double %817
vcall8Bl
j
	full_text]
[
Y%819 = tail call double @llvm.fmuladd.f64(double %651, double -4.000000e+00, double %818)
,double8B

	full_text

double %651
,double8B

	full_text

double %818
Pload8BF
D
	full_text7
5
3%820 = load double, double* %183, align 8, !tbaa !8
.double*8B

	full_text

double* %183
:fadd8B0
.
	full_text!

%821 = fadd double %819, %820
,double8B

	full_text

double %819
,double8B

	full_text

double %820
mcall8Bc
a
	full_textT
R
P%822 = tail call double @llvm.fmuladd.f64(double %407, double %821, double %787)
,double8B

	full_text

double %407
,double8B

	full_text

double %821
,double8B

	full_text

double %787
Pstore8BE
C
	full_text6
4
2store double %822, double* %753, align 8, !tbaa !8
,double8B

	full_text

double %822
.double*8B

	full_text

double* %753
Qload8BG
E
	full_text8
6
4%823 = load double, double* %265, align 16, !tbaa !8
.double*8B

	full_text

double* %265
vcall8Bl
j
	full_text]
[
Y%824 = tail call double @llvm.fmuladd.f64(double %791, double -4.000000e+00, double %823)
,double8B

	full_text

double %791
,double8B

	full_text

double %823
ucall8Bk
i
	full_text\
Z
X%825 = tail call double @llvm.fmuladd.f64(double %648, double 6.000000e+00, double %824)
,double8B

	full_text

double %648
,double8B

	full_text

double %824
vcall8Bl
j
	full_text]
[
Y%826 = tail call double @llvm.fmuladd.f64(double %647, double -4.000000e+00, double %825)
,double8B

	full_text

double %647
,double8B

	full_text

double %825
Qload8BG
E
	full_text8
6
4%827 = load double, double* %188, align 16, !tbaa !8
.double*8B

	full_text

double* %188
:fadd8B0
.
	full_text!

%828 = fadd double %826, %827
,double8B

	full_text

double %826
,double8B

	full_text

double %827
mcall8Bc
a
	full_textT
R
P%829 = tail call double @llvm.fmuladd.f64(double %407, double %828, double %794)
,double8B

	full_text

double %407
,double8B

	full_text

double %828
,double8B

	full_text

double %794
Pstore8BE
C
	full_text6
4
2store double %829, double* %758, align 8, !tbaa !8
,double8B

	full_text

double %829
.double*8B

	full_text

double* %758
:icmp8B0
.
	full_text!

%830 = icmp eq i64 %669, %635
&i648B

	full_text


i64 %669
&i648B

	full_text


i64 %635
Abitcast8B4
2
	full_text%
#
!%831 = bitcast double %798 to i64
,double8B

	full_text

double %798
Abitcast8B4
2
	full_text%
#
!%832 = bitcast double %770 to i64
,double8B

	full_text

double %770
Abitcast8B4
2
	full_text%
#
!%833 = bitcast double %661 to i64
,double8B

	full_text

double %661
Abitcast8B4
2
	full_text%
#
!%834 = bitcast double %806 to i64
,double8B

	full_text

double %806
Abitcast8B4
2
	full_text%
#
!%835 = bitcast double %777 to i64
,double8B

	full_text

double %777
Abitcast8B4
2
	full_text%
#
!%836 = bitcast double %656 to i64
,double8B

	full_text

double %656
Abitcast8B4
2
	full_text%
#
!%837 = bitcast double %784 to i64
,double8B

	full_text

double %784
Abitcast8B4
2
	full_text%
#
!%838 = bitcast double %652 to i64
,double8B

	full_text

double %652
Abitcast8B4
2
	full_text%
#
!%839 = bitcast double %791 to i64
,double8B

	full_text

double %791
Abitcast8B4
2
	full_text%
#
!%840 = bitcast double %648 to i64
,double8B

	full_text

double %648
=br8B5
3
	full_text&
$
"br i1 %830, label %841, label %641
$i18B

	full_text
	
i1 %830
Qstore8BF
D
	full_text7
5
3store double %667, double* %632, align 16, !tbaa !8
,double8B

	full_text

double %667
.double*8B

	full_text

double* %632
Pstore8BE
C
	full_text6
4
2store double %666, double* %190, align 8, !tbaa !8
,double8B

	full_text

double %666
.double*8B

	full_text

double* %190
Pstore8BE
C
	full_text6
4
2store double %665, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %665
-double*8B

	full_text

double* %84
Jstore8B?
=
	full_text0
.
,store i64 %670, i64* %200, align 8, !tbaa !8
&i648B

	full_text


i64 %670
(i64*8B

	full_text

	i64* %200
Kstore8B@
>
	full_text1
/
-store i64 %633, i64* %210, align 16, !tbaa !8
&i648B

	full_text


i64 %633
(i64*8B

	full_text

	i64* %210
Ostore8BD
B
	full_text5
3
1store double %661, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %661
-double*8B

	full_text

double* %86
Jstore8B?
=
	full_text0
.
,store i64 %671, i64* %218, align 8, !tbaa !8
&i648B

	full_text


i64 %671
(i64*8B

	full_text

	i64* %218
Pstore8BE
C
	full_text6
4
2store double %656, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %656
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %655, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %655
-double*8B

	full_text

double* %46
Jstore8B?
=
	full_text0
.
,store i64 %673, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %673
(i64*8B

	full_text

	i64* %235
Ostore8BD
B
	full_text5
3
1store double %652, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %652
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %651, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %651
-double*8B

	full_text

double* %51
Jstore8B?
=
	full_text0
.
,store i64 %675, i64* %252, align 8, !tbaa !8
&i648B

	full_text


i64 %675
(i64*8B

	full_text

	i64* %252
Pstore8BE
C
	full_text6
4
2store double %648, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %648
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %647, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %647
-double*8B

	full_text

double* %56
Jstore8B?
=
	full_text0
.
,store i64 %677, i64* %270, align 8, !tbaa !8
&i648B

	full_text


i64 %677
(i64*8B

	full_text

	i64* %270
Jload8B@
>
	full_text1
/
-%842 = load i64, i64* %212, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %212
(br8B 

	full_text

br label %843
Kphi8BB
@
	full_text3
1
/%844 = phi double [ %827, %841 ], [ %616, %25 ]
,double8B

	full_text

double %827
,double8B

	full_text

double %616
Kphi8BB
@
	full_text3
1
/%845 = phi double [ %647, %841 ], [ %614, %25 ]
,double8B

	full_text

double %647
,double8B

	full_text

double %614
Hphi8B?
=
	full_text0
.
,%846 = phi i64 [ %840, %841 ], [ %630, %25 ]
&i648B

	full_text


i64 %840
&i648B

	full_text


i64 %630
Hphi8B?
=
	full_text0
.
,%847 = phi i64 [ %839, %841 ], [ %629, %25 ]
&i648B

	full_text


i64 %839
&i648B

	full_text


i64 %629
Kphi8BB
@
	full_text3
1
/%848 = phi double [ %820, %841 ], [ %608, %25 ]
,double8B

	full_text

double %820
,double8B

	full_text

double %608
Kphi8BB
@
	full_text3
1
/%849 = phi double [ %651, %841 ], [ %606, %25 ]
,double8B

	full_text

double %651
,double8B

	full_text

double %606
Hphi8B?
=
	full_text0
.
,%850 = phi i64 [ %838, %841 ], [ %628, %25 ]
&i648B

	full_text


i64 %838
&i648B

	full_text


i64 %628
Hphi8B?
=
	full_text0
.
,%851 = phi i64 [ %837, %841 ], [ %627, %25 ]
&i648B

	full_text


i64 %837
&i648B

	full_text


i64 %627
Kphi8BB
@
	full_text3
1
/%852 = phi double [ %813, %841 ], [ %600, %25 ]
,double8B

	full_text

double %813
,double8B

	full_text

double %600
Kphi8BB
@
	full_text3
1
/%853 = phi double [ %655, %841 ], [ %598, %25 ]
,double8B

	full_text

double %655
,double8B

	full_text

double %598
Hphi8B?
=
	full_text0
.
,%854 = phi i64 [ %836, %841 ], [ %626, %25 ]
&i648B

	full_text


i64 %836
&i648B

	full_text


i64 %626
Hphi8B?
=
	full_text0
.
,%855 = phi i64 [ %835, %841 ], [ %625, %25 ]
&i648B

	full_text


i64 %835
&i648B

	full_text


i64 %625
Kphi8BB
@
	full_text3
1
/%856 = phi double [ %806, %841 ], [ %592, %25 ]
,double8B

	full_text

double %806
,double8B

	full_text

double %592
Hphi8B?
=
	full_text0
.
,%857 = phi i64 [ %834, %841 ], [ %624, %25 ]
&i648B

	full_text


i64 %834
&i648B

	full_text


i64 %624
Kphi8BB
@
	full_text3
1
/%858 = phi double [ %804, %841 ], [ %590, %25 ]
,double8B

	full_text

double %804
,double8B

	full_text

double %590
Hphi8B?
=
	full_text0
.
,%859 = phi i64 [ %833, %841 ], [ %623, %25 ]
&i648B

	full_text


i64 %833
&i648B

	full_text


i64 %623
Hphi8B?
=
	full_text0
.
,%860 = phi i64 [ %842, %841 ], [ %622, %25 ]
&i648B

	full_text


i64 %842
&i648B

	full_text


i64 %622
Hphi8B?
=
	full_text0
.
,%861 = phi i64 [ %831, %841 ], [ %621, %25 ]
&i648B

	full_text


i64 %831
&i648B

	full_text


i64 %621
5add8B,
*
	full_text

%862 = add nsw i32 %4, -2
Jload8B@
>
	full_text1
/
-%863 = load i64, i64* %191, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %191
Kstore8B@
>
	full_text1
/
-store i64 %863, i64* %194, align 16, !tbaa !8
&i648B

	full_text


i64 %863
(i64*8B

	full_text

	i64* %194
Jload8B@
>
	full_text1
/
-%864 = load i64, i64* %85, align 16, !tbaa !8
'i64*8B

	full_text


i64* %85
Jstore8B?
=
	full_text0
.
,store i64 %864, i64* %191, align 8, !tbaa !8
&i648B

	full_text


i64 %864
(i64*8B

	full_text

	i64* %191
Iload8B?
=
	full_text0
.
,%865 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
Jstore8B?
=
	full_text0
.
,store i64 %865, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %865
'i64*8B

	full_text


i64* %85
Istore8B>
<
	full_text/
-
+store i64 %861, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %861
'i64*8B

	full_text


i64* %83
Jload8B@
>
	full_text1
/
-%866 = load i64, i64* %200, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %200
Kstore8B@
>
	full_text1
/
-store i64 %866, i64* %203, align 16, !tbaa !8
&i648B

	full_text


i64 %866
(i64*8B

	full_text

	i64* %203
Kload8BA
?
	full_text2
0
.%867 = load i64, i64* %205, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %205
Jstore8B?
=
	full_text0
.
,store i64 %867, i64* %200, align 8, !tbaa !8
&i648B

	full_text


i64 %867
(i64*8B

	full_text

	i64* %200
Jload8B@
>
	full_text1
/
-%868 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Kstore8B@
>
	full_text1
/
-store i64 %868, i64* %210, align 16, !tbaa !8
&i648B

	full_text


i64 %868
(i64*8B

	full_text

	i64* %210
Jstore8B?
=
	full_text0
.
,store i64 %860, i64* %215, align 8, !tbaa !8
&i648B

	full_text


i64 %860
(i64*8B

	full_text

	i64* %215
Jstore8B?
=
	full_text0
.
,store i64 %859, i64* %212, align 8, !tbaa !8
&i648B

	full_text


i64 %859
(i64*8B

	full_text

	i64* %212
Ostore8BD
B
	full_text5
3
1store double %858, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %858
-double*8B

	full_text

double* %86
Istore8B>
<
	full_text/
-
+store i64 %857, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %857
'i64*8B

	full_text


i64* %42
Jload8B@
>
	full_text1
/
-%869 = load i64, i64* %218, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %218
Jstore8B?
=
	full_text0
.
,store i64 %869, i64* %221, align 8, !tbaa !8
&i648B

	full_text


i64 %869
(i64*8B

	full_text

	i64* %221
Jload8B@
>
	full_text1
/
-%870 = load i64, i64* %222, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %222
Jstore8B?
=
	full_text0
.
,store i64 %870, i64* %218, align 8, !tbaa !8
&i648B

	full_text


i64 %870
(i64*8B

	full_text

	i64* %218
Jload8B@
>
	full_text1
/
-%871 = load i64, i64* %224, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %224
Jstore8B?
=
	full_text0
.
,store i64 %871, i64* %227, align 8, !tbaa !8
&i648B

	full_text


i64 %871
(i64*8B

	full_text

	i64* %227
Kstore8B@
>
	full_text1
/
-store i64 %855, i64* %232, align 16, !tbaa !8
&i648B

	full_text


i64 %855
(i64*8B

	full_text

	i64* %232
Jstore8B?
=
	full_text0
.
,store i64 %854, i64* %229, align 8, !tbaa !8
&i648B

	full_text


i64 %854
(i64*8B

	full_text

	i64* %229
Pstore8BE
C
	full_text6
4
2store double %853, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %853
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %852, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %852
-double*8B

	full_text

double* %46
Jload8B@
>
	full_text1
/
-%872 = load i64, i64* %235, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %235
Kstore8B@
>
	full_text1
/
-store i64 %872, i64* %238, align 16, !tbaa !8
&i648B

	full_text


i64 %872
(i64*8B

	full_text

	i64* %238
Kload8BA
?
	full_text2
0
.%873 = load i64, i64* %239, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %239
Jstore8B?
=
	full_text0
.
,store i64 %873, i64* %235, align 8, !tbaa !8
&i648B

	full_text


i64 %873
(i64*8B

	full_text

	i64* %235
Jload8B@
>
	full_text1
/
-%874 = load i64, i64* %241, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %241
Kstore8B@
>
	full_text1
/
-store i64 %874, i64* %244, align 16, !tbaa !8
&i648B

	full_text


i64 %874
(i64*8B

	full_text

	i64* %244
Jstore8B?
=
	full_text0
.
,store i64 %851, i64* %249, align 8, !tbaa !8
&i648B

	full_text


i64 %851
(i64*8B

	full_text

	i64* %249
Jstore8B?
=
	full_text0
.
,store i64 %850, i64* %246, align 8, !tbaa !8
&i648B

	full_text


i64 %850
(i64*8B

	full_text

	i64* %246
Ostore8BD
B
	full_text5
3
1store double %849, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %849
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %848, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %848
-double*8B

	full_text

double* %51
Jload8B@
>
	full_text1
/
-%875 = load i64, i64* %252, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %252
Jstore8B?
=
	full_text0
.
,store i64 %875, i64* %255, align 8, !tbaa !8
&i648B

	full_text


i64 %875
(i64*8B

	full_text

	i64* %255
Jload8B@
>
	full_text1
/
-%876 = load i64, i64* %256, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %256
Jstore8B?
=
	full_text0
.
,store i64 %876, i64* %252, align 8, !tbaa !8
&i648B

	full_text


i64 %876
(i64*8B

	full_text

	i64* %252
Jload8B@
>
	full_text1
/
-%877 = load i64, i64* %258, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %258
Jstore8B?
=
	full_text0
.
,store i64 %877, i64* %261, align 8, !tbaa !8
&i648B

	full_text


i64 %877
(i64*8B

	full_text

	i64* %261
Kstore8B@
>
	full_text1
/
-store i64 %847, i64* %266, align 16, !tbaa !8
&i648B

	full_text


i64 %847
(i64*8B

	full_text

	i64* %266
Jstore8B?
=
	full_text0
.
,store i64 %846, i64* %263, align 8, !tbaa !8
&i648B

	full_text


i64 %846
(i64*8B

	full_text

	i64* %263
Pstore8BE
C
	full_text6
4
2store double %845, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %845
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %844, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %844
-double*8B

	full_text

double* %56
Jload8B@
>
	full_text1
/
-%878 = load i64, i64* %270, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %270
Kstore8B@
>
	full_text1
/
-store i64 %878, i64* %273, align 16, !tbaa !8
&i648B

	full_text


i64 %878
(i64*8B

	full_text

	i64* %273
Kload8BA
?
	full_text2
0
.%879 = load i64, i64* %274, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %274
Jstore8B?
=
	full_text0
.
,store i64 %879, i64* %270, align 8, !tbaa !8
&i648B

	full_text


i64 %879
(i64*8B

	full_text

	i64* %270
Jload8B@
>
	full_text1
/
-%880 = load i64, i64* %276, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %276
Kstore8B@
>
	full_text1
/
-store i64 %880, i64* %279, align 16, !tbaa !8
&i648B

	full_text


i64 %880
(i64*8B

	full_text

	i64* %279
5add8B,
*
	full_text

%881 = add nsw i32 %4, -1
8sext8B.
,
	full_text

%882 = sext i32 %881 to i64
&i328B

	full_text


i32 %881
getelementptr8B‰
†
	full_texty
w
u%883 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %882
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Ibitcast8B<
:
	full_text-
+
)%884 = bitcast [5 x double]* %883 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %883
Jload8B@
>
	full_text1
/
-%885 = load i64, i64* %884, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %884
Kstore8B@
>
	full_text1
/
-store i64 %885, i64* %169, align 16, !tbaa !8
&i648B

	full_text


i64 %885
(i64*8B

	full_text

	i64* %169
¥getelementptr8B‘

	full_text€
~
|%886 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %882, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Cbitcast8B6
4
	full_text'
%
#%887 = bitcast double* %886 to i64*
.double*8B

	full_text

double* %886
Jload8B@
>
	full_text1
/
-%888 = load i64, i64* %887, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %887
Jstore8B?
=
	full_text0
.
,store i64 %888, i64* %174, align 8, !tbaa !8
&i648B

	full_text


i64 %888
(i64*8B

	full_text

	i64* %174
¥getelementptr8B‘

	full_text€
~
|%889 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %882, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Cbitcast8B6
4
	full_text'
%
#%890 = bitcast double* %889 to i64*
.double*8B

	full_text

double* %889
Jload8B@
>
	full_text1
/
-%891 = load i64, i64* %890, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %890
Kstore8B@
>
	full_text1
/
-store i64 %891, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %891
(i64*8B

	full_text

	i64* %179
¥getelementptr8B‘

	full_text€
~
|%892 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %882, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Cbitcast8B6
4
	full_text'
%
#%893 = bitcast double* %892 to i64*
.double*8B

	full_text

double* %892
Jload8B@
>
	full_text1
/
-%894 = load i64, i64* %893, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %893
Jstore8B?
=
	full_text0
.
,store i64 %894, i64* %184, align 8, !tbaa !8
&i648B

	full_text


i64 %894
(i64*8B

	full_text

	i64* %184
¥getelementptr8B‘

	full_text€
~
|%895 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %26, i64 %30, i64 %32, i64 %882, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Cbitcast8B6
4
	full_text'
%
#%896 = bitcast double* %895 to i64*
.double*8B

	full_text

double* %895
Jload8B@
>
	full_text1
/
-%897 = load i64, i64* %896, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %896
Kstore8B@
>
	full_text1
/
-store i64 %897, i64* %189, align 16, !tbaa !8
&i648B

	full_text


i64 %897
(i64*8B

	full_text

	i64* %189
rgetelementptr8B_
]
	full_textP
N
L%898 = getelementptr inbounds [5 x double], [5 x double]* %111, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
Qstore8BF
D
	full_text7
5
3store double %856, double* %898, align 16, !tbaa !8
,double8B

	full_text

double %856
.double*8B

	full_text

double* %898
8sext8B.
,
	full_text

%899 = sext i32 %862 to i64
&i328B

	full_text


i32 %862
getelementptr8B|
z
	full_textm
k
i%900 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 %899
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Pload8BF
D
	full_text7
5
3%901 = load double, double* %900, align 8, !tbaa !8
.double*8B

	full_text

double* %900
:fmul8B0
.
	full_text!

%902 = fmul double %901, %856
,double8B

	full_text

double %901
,double8B

	full_text

double %856
getelementptr8B|
z
	full_textm
k
i%903 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 %899
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Pload8BF
D
	full_text7
5
3%904 = load double, double* %903, align 8, !tbaa !8
.double*8B

	full_text

double* %903
:fsub8B0
.
	full_text!

%905 = fsub double %844, %904
,double8B

	full_text

double %844
,double8B

	full_text

double %904
Bfmul8B8
6
	full_text)
'
%%906 = fmul double %905, 4.000000e-01
,double8B

	full_text

double %905
mcall8Bc
a
	full_textT
R
P%907 = tail call double @llvm.fmuladd.f64(double %856, double %902, double %906)
,double8B

	full_text

double %856
,double8B

	full_text

double %902
,double8B

	full_text

double %906
Pstore8BE
C
	full_text6
4
2store double %907, double* %123, align 8, !tbaa !8
,double8B

	full_text

double %907
.double*8B

	full_text

double* %123
:fmul8B0
.
	full_text!

%908 = fmul double %902, %852
,double8B

	full_text

double %902
,double8B

	full_text

double %852
Qstore8BF
D
	full_text7
5
3store double %908, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %908
.double*8B

	full_text

double* %126
:fmul8B0
.
	full_text!

%909 = fmul double %902, %848
,double8B

	full_text

double %902
,double8B

	full_text

double %848
Pstore8BE
C
	full_text6
4
2store double %909, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %909
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%910 = fmul double %904, 4.000000e-01
,double8B

	full_text

double %904
Cfsub8B9
7
	full_text*
(
&%911 = fsub double -0.000000e+00, %910
,double8B

	full_text

double %910
ucall8Bk
i
	full_text\
Z
X%912 = tail call double @llvm.fmuladd.f64(double %844, double 1.400000e+00, double %911)
,double8B

	full_text

double %844
,double8B

	full_text

double %911
:fmul8B0
.
	full_text!

%913 = fmul double %902, %912
,double8B

	full_text

double %902
,double8B

	full_text

double %912
Qstore8BF
D
	full_text7
5
3store double %913, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %913
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%914 = fmul double %901, %852
,double8B

	full_text

double %901
,double8B

	full_text

double %852
:fmul8B0
.
	full_text!

%915 = fmul double %901, %848
,double8B

	full_text

double %901
,double8B

	full_text

double %848
:fmul8B0
.
	full_text!

%916 = fmul double %901, %844
,double8B

	full_text

double %901
,double8B

	full_text

double %844
8sext8B.
,
	full_text

%917 = sext i32 %619 to i64
&i328B

	full_text


i32 %619
getelementptr8B|
z
	full_textm
k
i%918 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 %917
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %917
Pload8BF
D
	full_text7
5
3%919 = load double, double* %918, align 8, !tbaa !8
.double*8B

	full_text

double* %918
:fmul8B0
.
	full_text!

%920 = fmul double %919, %858
,double8B

	full_text

double %919
,double8B

	full_text

double %858
:fmul8B0
.
	full_text!

%921 = fmul double %919, %853
,double8B

	full_text

double %919
,double8B

	full_text

double %853
:fmul8B0
.
	full_text!

%922 = fmul double %919, %849
,double8B

	full_text

double %919
,double8B

	full_text

double %849
:fmul8B0
.
	full_text!

%923 = fmul double %919, %845
,double8B

	full_text

double %919
,double8B

	full_text

double %845
:fsub8B0
.
	full_text!

%924 = fsub double %902, %920
,double8B

	full_text

double %902
,double8B

	full_text

double %920
Hfmul8B>
<
	full_text/
-
+%925 = fmul double %924, 0x4045555555555555
,double8B

	full_text

double %924
Pstore8BE
C
	full_text6
4
2store double %925, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %925
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%926 = fsub double %914, %921
,double8B

	full_text

double %914
,double8B

	full_text

double %921
Bfmul8B8
6
	full_text)
'
%%927 = fmul double %926, 3.200000e+01
,double8B

	full_text

double %926
Pstore8BE
C
	full_text6
4
2store double %927, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %927
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%928 = fsub double %915, %922
,double8B

	full_text

double %915
,double8B

	full_text

double %922
Bfmul8B8
6
	full_text)
'
%%929 = fmul double %928, 3.200000e+01
,double8B

	full_text

double %928
Pstore8BE
C
	full_text6
4
2store double %929, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %929
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%930 = fmul double %914, %914
,double8B

	full_text

double %914
,double8B

	full_text

double %914
mcall8Bc
a
	full_textT
R
P%931 = tail call double @llvm.fmuladd.f64(double %902, double %902, double %930)
,double8B

	full_text

double %902
,double8B

	full_text

double %902
,double8B

	full_text

double %930
mcall8Bc
a
	full_textT
R
P%932 = tail call double @llvm.fmuladd.f64(double %915, double %915, double %931)
,double8B

	full_text

double %915
,double8B

	full_text

double %915
,double8B

	full_text

double %931
:fmul8B0
.
	full_text!

%933 = fmul double %921, %921
,double8B

	full_text

double %921
,double8B

	full_text

double %921
mcall8Bc
a
	full_textT
R
P%934 = tail call double @llvm.fmuladd.f64(double %920, double %920, double %933)
,double8B

	full_text

double %920
,double8B

	full_text

double %920
,double8B

	full_text

double %933
mcall8Bc
a
	full_textT
R
P%935 = tail call double @llvm.fmuladd.f64(double %922, double %922, double %934)
,double8B

	full_text

double %922
,double8B

	full_text

double %922
,double8B

	full_text

double %934
:fsub8B0
.
	full_text!

%936 = fsub double %932, %935
,double8B

	full_text

double %932
,double8B

	full_text

double %935
:fmul8B0
.
	full_text!

%937 = fmul double %920, %920
,double8B

	full_text

double %920
,double8B

	full_text

double %920
Cfsub8B9
7
	full_text*
(
&%938 = fsub double -0.000000e+00, %937
,double8B

	full_text

double %937
mcall8Bc
a
	full_textT
R
P%939 = tail call double @llvm.fmuladd.f64(double %902, double %902, double %938)
,double8B

	full_text

double %902
,double8B

	full_text

double %902
,double8B

	full_text

double %938
Hfmul8B>
<
	full_text/
-
+%940 = fmul double %939, 0x4015555555555555
,double8B

	full_text

double %939
{call8Bq
o
	full_textb
`
^%941 = tail call double @llvm.fmuladd.f64(double %936, double 0xC02EB851EB851EB6, double %940)
,double8B

	full_text

double %936
,double8B

	full_text

double %940
:fsub8B0
.
	full_text!

%942 = fsub double %916, %923
,double8B

	full_text

double %916
,double8B

	full_text

double %923
{call8Bq
o
	full_textb
`
^%943 = tail call double @llvm.fmuladd.f64(double %942, double 0x404F5C28F5C28F5B, double %941)
,double8B

	full_text

double %942
,double8B

	full_text

double %941
Pstore8BE
C
	full_text6
4
2store double %943, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %943
.double*8B

	full_text

double* %164
¥getelementptr8B‘

	full_text€
~
|%944 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %917, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %917
Pload8BF
D
	full_text7
5
3%945 = load double, double* %944, align 8, !tbaa !8
.double*8B

	full_text

double* %944
Qload8BG
E
	full_text8
6
4%946 = load double, double* %202, align 16, !tbaa !8
.double*8B

	full_text

double* %202
:fsub8B0
.
	full_text!

%947 = fsub double %856, %946
,double8B

	full_text

double %856
,double8B

	full_text

double %946
vcall8Bl
j
	full_text]
[
Y%948 = tail call double @llvm.fmuladd.f64(double %947, double -1.600000e+01, double %945)
,double8B

	full_text

double %947
,double8B

	full_text

double %945
¥getelementptr8B‘

	full_text€
~
|%949 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %917, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %917
Pload8BF
D
	full_text7
5
3%950 = load double, double* %949, align 8, !tbaa !8
.double*8B

	full_text

double* %949
Pload8BF
D
	full_text7
5
3%951 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
:fsub8B0
.
	full_text!

%952 = fsub double %907, %951
,double8B

	full_text

double %907
,double8B

	full_text

double %951
vcall8Bl
j
	full_text]
[
Y%953 = tail call double @llvm.fmuladd.f64(double %952, double -1.600000e+01, double %950)
,double8B

	full_text

double %952
,double8B

	full_text

double %950
¥getelementptr8B‘

	full_text€
~
|%954 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %917, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %917
Pload8BF
D
	full_text7
5
3%955 = load double, double* %954, align 8, !tbaa !8
.double*8B

	full_text

double* %954
Qload8BG
E
	full_text8
6
4%956 = load double, double* %237, align 16, !tbaa !8
.double*8B

	full_text

double* %237
:fsub8B0
.
	full_text!

%957 = fsub double %908, %956
,double8B

	full_text

double %908
,double8B

	full_text

double %956
vcall8Bl
j
	full_text]
[
Y%958 = tail call double @llvm.fmuladd.f64(double %957, double -1.600000e+01, double %955)
,double8B

	full_text

double %957
,double8B

	full_text

double %955
¥getelementptr8B‘

	full_text€
~
|%959 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %917, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %917
Pload8BF
D
	full_text7
5
3%960 = load double, double* %959, align 8, !tbaa !8
.double*8B

	full_text

double* %959
Pload8BF
D
	full_text7
5
3%961 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
:fsub8B0
.
	full_text!

%962 = fsub double %909, %961
,double8B

	full_text

double %909
,double8B

	full_text

double %961
vcall8Bl
j
	full_text]
[
Y%963 = tail call double @llvm.fmuladd.f64(double %962, double -1.600000e+01, double %960)
,double8B

	full_text

double %962
,double8B

	full_text

double %960
¥getelementptr8B‘

	full_text€
~
|%964 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %917, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %917
Pload8BF
D
	full_text7
5
3%965 = load double, double* %964, align 8, !tbaa !8
.double*8B

	full_text

double* %964
Qload8BG
E
	full_text8
6
4%966 = load double, double* %272, align 16, !tbaa !8
.double*8B

	full_text

double* %272
:fsub8B0
.
	full_text!

%967 = fsub double %913, %966
,double8B

	full_text

double %913
,double8B

	full_text

double %966
vcall8Bl
j
	full_text]
[
Y%968 = tail call double @llvm.fmuladd.f64(double %967, double -1.600000e+01, double %965)
,double8B

	full_text

double %967
,double8B

	full_text

double %965
Pload8BF
D
	full_text7
5
3%969 = load double, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
Pload8BF
D
	full_text7
5
3%970 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%971 = tail call double @llvm.fmuladd.f64(double %970, double -2.000000e+00, double %969)
,double8B

	full_text

double %970
,double8B

	full_text

double %969
Oload8BE
C
	full_text6
4
2%972 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%973 = fadd double %971, %972
,double8B

	full_text

double %971
,double8B

	full_text

double %972
ucall8Bk
i
	full_text\
Z
X%974 = tail call double @llvm.fmuladd.f64(double %973, double 7.680000e+02, double %948)
,double8B

	full_text

double %973
,double8B

	full_text

double %948
Pload8BF
D
	full_text7
5
3%975 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
:fsub8B0
.
	full_text!

%976 = fsub double %925, %975
,double8B

	full_text

double %925
,double8B

	full_text

double %975
ucall8Bk
i
	full_text\
Z
X%977 = tail call double @llvm.fmuladd.f64(double %976, double 3.200000e+00, double %953)
,double8B

	full_text

double %976
,double8B

	full_text

double %953
Pload8BF
D
	full_text7
5
3%978 = load double, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
vcall8Bl
j
	full_text]
[
Y%979 = tail call double @llvm.fmuladd.f64(double %858, double -2.000000e+00, double %978)
,double8B

	full_text

double %858
,double8B

	full_text

double %978
:fadd8B0
.
	full_text!

%980 = fadd double %856, %979
,double8B

	full_text

double %856
,double8B

	full_text

double %979
ucall8Bk
i
	full_text\
Z
X%981 = tail call double @llvm.fmuladd.f64(double %980, double 7.680000e+02, double %977)
,double8B

	full_text

double %980
,double8B

	full_text

double %977
Qload8BG
E
	full_text8
6
4%982 = load double, double* %243, align 16, !tbaa !8
.double*8B

	full_text

double* %243
:fsub8B0
.
	full_text!

%983 = fsub double %927, %982
,double8B

	full_text

double %927
,double8B

	full_text

double %982
ucall8Bk
i
	full_text\
Z
X%984 = tail call double @llvm.fmuladd.f64(double %983, double 3.200000e+00, double %958)
,double8B

	full_text

double %983
,double8B

	full_text

double %958
Pload8BF
D
	full_text7
5
3%985 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
vcall8Bl
j
	full_text]
[
Y%986 = tail call double @llvm.fmuladd.f64(double %853, double -2.000000e+00, double %985)
,double8B

	full_text

double %853
,double8B

	full_text

double %985
:fadd8B0
.
	full_text!

%987 = fadd double %852, %986
,double8B

	full_text

double %852
,double8B

	full_text

double %986
ucall8Bk
i
	full_text\
Z
X%988 = tail call double @llvm.fmuladd.f64(double %987, double 7.680000e+02, double %984)
,double8B

	full_text

double %987
,double8B

	full_text

double %984
Pload8BF
D
	full_text7
5
3%989 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
:fsub8B0
.
	full_text!

%990 = fsub double %929, %989
,double8B

	full_text

double %929
,double8B

	full_text

double %989
ucall8Bk
i
	full_text\
Z
X%991 = tail call double @llvm.fmuladd.f64(double %990, double 3.200000e+00, double %963)
,double8B

	full_text

double %990
,double8B

	full_text

double %963
Pload8BF
D
	full_text7
5
3%992 = load double, double* %245, align 8, !tbaa !8
.double*8B

	full_text

double* %245
vcall8Bl
j
	full_text]
[
Y%993 = tail call double @llvm.fmuladd.f64(double %849, double -2.000000e+00, double %992)
,double8B

	full_text

double %849
,double8B

	full_text

double %992
:fadd8B0
.
	full_text!

%994 = fadd double %848, %993
,double8B

	full_text

double %848
,double8B

	full_text

double %993
ucall8Bk
i
	full_text\
Z
X%995 = tail call double @llvm.fmuladd.f64(double %994, double 7.680000e+02, double %991)
,double8B

	full_text

double %994
,double8B

	full_text

double %991
Qload8BG
E
	full_text8
6
4%996 = load double, double* %278, align 16, !tbaa !8
.double*8B

	full_text

double* %278
:fsub8B0
.
	full_text!

%997 = fsub double %943, %996
,double8B

	full_text

double %943
,double8B

	full_text

double %996
ucall8Bk
i
	full_text\
Z
X%998 = tail call double @llvm.fmuladd.f64(double %997, double 3.200000e+00, double %968)
,double8B

	full_text

double %997
,double8B

	full_text

double %968
Pload8BF
D
	full_text7
5
3%999 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
wcall8Bm
k
	full_text^
\
Z%1000 = tail call double @llvm.fmuladd.f64(double %845, double -2.000000e+00, double %999)
,double8B

	full_text

double %845
,double8B

	full_text

double %999
<fadd8B2
0
	full_text#
!
%1001 = fadd double %844, %1000
,double8B

	full_text

double %844
-double8B

	full_text

double %1000
wcall8Bm
k
	full_text^
\
Z%1002 = tail call double @llvm.fmuladd.f64(double %1001, double 7.680000e+02, double %998)
-double8B

	full_text

double %1001
,double8B

	full_text

double %998
Rload8BH
F
	full_text9
7
5%1003 = load double, double* %193, align 16, !tbaa !8
.double*8B

	full_text

double* %193
xcall8Bn
l
	full_text_
]
[%1004 = tail call double @llvm.fmuladd.f64(double %969, double -4.000000e+00, double %1003)
,double8B

	full_text

double %969
-double8B

	full_text

double %1003
wcall8Bm
k
	full_text^
\
Z%1005 = tail call double @llvm.fmuladd.f64(double %970, double 6.000000e+00, double %1004)
,double8B

	full_text

double %970
-double8B

	full_text

double %1004
xcall8Bn
l
	full_text_
]
[%1006 = tail call double @llvm.fmuladd.f64(double %972, double -4.000000e+00, double %1005)
,double8B

	full_text

double %972
-double8B

	full_text

double %1005
ocall8Be
c
	full_textV
T
R%1007 = tail call double @llvm.fmuladd.f64(double %407, double %1006, double %974)
,double8B

	full_text

double %407
-double8B

	full_text

double %1006
,double8B

	full_text

double %974
Qstore8BF
D
	full_text7
5
3store double %1007, double* %944, align 8, !tbaa !8
-double8B

	full_text

double %1007
.double*8B

	full_text

double* %944
Qload8BG
E
	full_text8
6
4%1008 = load double, double* %214, align 8, !tbaa !8
.double*8B

	full_text

double* %214
xcall8Bn
l
	full_text_
]
[%1009 = tail call double @llvm.fmuladd.f64(double %978, double -4.000000e+00, double %1008)
,double8B

	full_text

double %978
-double8B

	full_text

double %1008
Pload8BF
D
	full_text7
5
3%1010 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
xcall8Bn
l
	full_text_
]
[%1011 = tail call double @llvm.fmuladd.f64(double %1010, double 6.000000e+00, double %1009)
-double8B

	full_text

double %1010
-double8B

	full_text

double %1009
Pload8BF
D
	full_text7
5
3%1012 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
ycall8Bo
m
	full_text`
^
\%1013 = tail call double @llvm.fmuladd.f64(double %1012, double -4.000000e+00, double %1011)
-double8B

	full_text

double %1012
-double8B

	full_text

double %1011
ocall8Be
c
	full_textV
T
R%1014 = tail call double @llvm.fmuladd.f64(double %407, double %1013, double %981)
,double8B

	full_text

double %407
-double8B

	full_text

double %1013
,double8B

	full_text

double %981
Qstore8BF
D
	full_text7
5
3store double %1014, double* %949, align 8, !tbaa !8
-double8B

	full_text

double %1014
.double*8B

	full_text

double* %949
Rload8BH
F
	full_text9
7
5%1015 = load double, double* %231, align 16, !tbaa !8
.double*8B

	full_text

double* %231
xcall8Bn
l
	full_text_
]
[%1016 = tail call double @llvm.fmuladd.f64(double %985, double -4.000000e+00, double %1015)
,double8B

	full_text

double %985
-double8B

	full_text

double %1015
Qload8BG
E
	full_text8
6
4%1017 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
xcall8Bn
l
	full_text_
]
[%1018 = tail call double @llvm.fmuladd.f64(double %1017, double 6.000000e+00, double %1016)
-double8B

	full_text

double %1017
-double8B

	full_text

double %1016
Pload8BF
D
	full_text7
5
3%1019 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
ycall8Bo
m
	full_text`
^
\%1020 = tail call double @llvm.fmuladd.f64(double %1019, double -4.000000e+00, double %1018)
-double8B

	full_text

double %1019
-double8B

	full_text

double %1018
ocall8Be
c
	full_textV
T
R%1021 = tail call double @llvm.fmuladd.f64(double %407, double %1020, double %988)
,double8B

	full_text

double %407
-double8B

	full_text

double %1020
,double8B

	full_text

double %988
Qstore8BF
D
	full_text7
5
3store double %1021, double* %954, align 8, !tbaa !8
-double8B

	full_text

double %1021
.double*8B

	full_text

double* %954
Qload8BG
E
	full_text8
6
4%1022 = load double, double* %248, align 8, !tbaa !8
.double*8B

	full_text

double* %248
xcall8Bn
l
	full_text_
]
[%1023 = tail call double @llvm.fmuladd.f64(double %992, double -4.000000e+00, double %1022)
,double8B

	full_text

double %992
-double8B

	full_text

double %1022
Pload8BF
D
	full_text7
5
3%1024 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
xcall8Bn
l
	full_text_
]
[%1025 = tail call double @llvm.fmuladd.f64(double %1024, double 6.000000e+00, double %1023)
-double8B

	full_text

double %1024
-double8B

	full_text

double %1023
Pload8BF
D
	full_text7
5
3%1026 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
ycall8Bo
m
	full_text`
^
\%1027 = tail call double @llvm.fmuladd.f64(double %1026, double -4.000000e+00, double %1025)
-double8B

	full_text

double %1026
-double8B

	full_text

double %1025
ocall8Be
c
	full_textV
T
R%1028 = tail call double @llvm.fmuladd.f64(double %407, double %1027, double %995)
,double8B

	full_text

double %407
-double8B

	full_text

double %1027
,double8B

	full_text

double %995
Qstore8BF
D
	full_text7
5
3store double %1028, double* %959, align 8, !tbaa !8
-double8B

	full_text

double %1028
.double*8B

	full_text

double* %959
Rload8BH
F
	full_text9
7
5%1029 = load double, double* %265, align 16, !tbaa !8
.double*8B

	full_text

double* %265
xcall8Bn
l
	full_text_
]
[%1030 = tail call double @llvm.fmuladd.f64(double %999, double -4.000000e+00, double %1029)
,double8B

	full_text

double %999
-double8B

	full_text

double %1029
Qload8BG
E
	full_text8
6
4%1031 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
xcall8Bn
l
	full_text_
]
[%1032 = tail call double @llvm.fmuladd.f64(double %1031, double 6.000000e+00, double %1030)
-double8B

	full_text

double %1031
-double8B

	full_text

double %1030
Pload8BF
D
	full_text7
5
3%1033 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
ycall8Bo
m
	full_text`
^
\%1034 = tail call double @llvm.fmuladd.f64(double %1033, double -4.000000e+00, double %1032)
-double8B

	full_text

double %1033
-double8B

	full_text

double %1032
pcall8Bf
d
	full_textW
U
S%1035 = tail call double @llvm.fmuladd.f64(double %407, double %1034, double %1002)
,double8B

	full_text

double %407
-double8B

	full_text

double %1034
-double8B

	full_text

double %1002
Qstore8BF
D
	full_text7
5
3store double %1035, double* %964, align 8, !tbaa !8
-double8B

	full_text

double %1035
.double*8B

	full_text

double* %964
…getelementptr8Br
p
	full_textc
a
_%1036 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Rstore8BG
E
	full_text8
6
4store double %969, double* %1036, align 16, !tbaa !8
,double8B

	full_text

double %969
/double*8B 

	full_text

double* %1036
Pstore8BE
C
	full_text6
4
2store double %970, double* %190, align 8, !tbaa !8
,double8B

	full_text

double %970
.double*8B

	full_text

double* %190
Pstore8BE
C
	full_text6
4
2store double %972, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %972
-double*8B

	full_text

double* %84
Lload8BB
@
	full_text3
1
/%1037 = load i64, i64* %198, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %198
Jstore8B?
=
	full_text0
.
,store i64 %1037, i64* %83, align 8, !tbaa !8
'i648B

	full_text

	i64 %1037
'i64*8B

	full_text


i64* %83
Kload8BA
?
	full_text2
0
.%1038 = load i64, i64* %200, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %200
Lstore8BA
?
	full_text2
0
.store i64 %1038, i64* %203, align 16, !tbaa !8
'i648B

	full_text

	i64 %1038
(i64*8B

	full_text

	i64* %203
Lload8BB
@
	full_text3
1
/%1039 = load i64, i64* %205, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %205
Kstore8B@
>
	full_text1
/
-store i64 %1039, i64* %200, align 8, !tbaa !8
'i648B

	full_text

	i64 %1039
(i64*8B

	full_text

	i64* %200
Kload8BA
?
	full_text2
0
.%1040 = load i64, i64* %208, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %208
Lstore8BA
?
	full_text2
0
.store i64 %1040, i64* %210, align 16, !tbaa !8
'i648B

	full_text

	i64 %1040
(i64*8B

	full_text

	i64* %210
Pstore8BE
C
	full_text6
4
2store double %978, double* %214, align 8, !tbaa !8
,double8B

	full_text

double %978
.double*8B

	full_text

double* %214
Qstore8BF
D
	full_text7
5
3store double %1010, double* %211, align 8, !tbaa !8
-double8B

	full_text

double %1010
.double*8B

	full_text

double* %211
Pstore8BE
C
	full_text6
4
2store double %1012, double* %86, align 8, !tbaa !8
-double8B

	full_text

double %1012
-double*8B

	full_text

double* %86
Kload8BA
?
	full_text2
0
.%1041 = load i64, i64* %174, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %174
Jstore8B?
=
	full_text0
.
,store i64 %1041, i64* %42, align 8, !tbaa !8
'i648B

	full_text

	i64 %1041
'i64*8B

	full_text


i64* %42
Kload8BA
?
	full_text2
0
.%1042 = load i64, i64* %218, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %218
Kstore8B@
>
	full_text1
/
-store i64 %1042, i64* %221, align 8, !tbaa !8
'i648B

	full_text

	i64 %1042
(i64*8B

	full_text

	i64* %221
Kload8BA
?
	full_text2
0
.%1043 = load i64, i64* %222, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %222
Kstore8B@
>
	full_text1
/
-store i64 %1043, i64* %218, align 8, !tbaa !8
'i648B

	full_text

	i64 %1043
(i64*8B

	full_text

	i64* %218
Kload8BA
?
	full_text2
0
.%1044 = load i64, i64* %224, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %224
Kstore8B@
>
	full_text1
/
-store i64 %1044, i64* %227, align 8, !tbaa !8
'i648B

	full_text

	i64 %1044
(i64*8B

	full_text

	i64* %227
Qstore8BF
D
	full_text7
5
3store double %985, double* %231, align 16, !tbaa !8
,double8B

	full_text

double %985
.double*8B

	full_text

double* %231
Qstore8BF
D
	full_text7
5
3store double %1017, double* %228, align 8, !tbaa !8
-double8B

	full_text

double %1017
.double*8B

	full_text

double* %228
Qstore8BF
D
	full_text7
5
3store double %1019, double* %88, align 16, !tbaa !8
-double8B

	full_text

double %1019
-double*8B

	full_text

double* %88
Lload8BB
@
	full_text3
1
/%1045 = load i64, i64* %179, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %179
Jstore8B?
=
	full_text0
.
,store i64 %1045, i64* %47, align 8, !tbaa !8
'i648B

	full_text

	i64 %1045
'i64*8B

	full_text


i64* %47
Kload8BA
?
	full_text2
0
.%1046 = load i64, i64* %235, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %235
Lstore8BA
?
	full_text2
0
.store i64 %1046, i64* %238, align 16, !tbaa !8
'i648B

	full_text

	i64 %1046
(i64*8B

	full_text

	i64* %238
Lload8BB
@
	full_text3
1
/%1047 = load i64, i64* %239, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %239
Kstore8B@
>
	full_text1
/
-store i64 %1047, i64* %235, align 8, !tbaa !8
'i648B

	full_text

	i64 %1047
(i64*8B

	full_text

	i64* %235
Kload8BA
?
	full_text2
0
.%1048 = load i64, i64* %241, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %241
Lstore8BA
?
	full_text2
0
.store i64 %1048, i64* %244, align 16, !tbaa !8
'i648B

	full_text

	i64 %1048
(i64*8B

	full_text

	i64* %244
Pstore8BE
C
	full_text6
4
2store double %992, double* %248, align 8, !tbaa !8
,double8B

	full_text

double %992
.double*8B

	full_text

double* %248
Qstore8BF
D
	full_text7
5
3store double %1024, double* %245, align 8, !tbaa !8
-double8B

	full_text

double %1024
.double*8B

	full_text

double* %245
Pstore8BE
C
	full_text6
4
2store double %1026, double* %90, align 8, !tbaa !8
-double8B

	full_text

double %1026
-double*8B

	full_text

double* %90
Kload8BA
?
	full_text2
0
.%1049 = load i64, i64* %184, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %184
Jstore8B?
=
	full_text0
.
,store i64 %1049, i64* %52, align 8, !tbaa !8
'i648B

	full_text

	i64 %1049
'i64*8B

	full_text


i64* %52
Kload8BA
?
	full_text2
0
.%1050 = load i64, i64* %252, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %252
Kstore8B@
>
	full_text1
/
-store i64 %1050, i64* %255, align 8, !tbaa !8
'i648B

	full_text

	i64 %1050
(i64*8B

	full_text

	i64* %255
Kload8BA
?
	full_text2
0
.%1051 = load i64, i64* %256, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %256
Kstore8B@
>
	full_text1
/
-store i64 %1051, i64* %252, align 8, !tbaa !8
'i648B

	full_text

	i64 %1051
(i64*8B

	full_text

	i64* %252
Kload8BA
?
	full_text2
0
.%1052 = load i64, i64* %258, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %258
Kstore8B@
>
	full_text1
/
-store i64 %1052, i64* %261, align 8, !tbaa !8
'i648B

	full_text

	i64 %1052
(i64*8B

	full_text

	i64* %261
Qstore8BF
D
	full_text7
5
3store double %999, double* %265, align 16, !tbaa !8
,double8B

	full_text

double %999
.double*8B

	full_text

double* %265
Qstore8BF
D
	full_text7
5
3store double %1031, double* %262, align 8, !tbaa !8
-double8B

	full_text

double %1031
.double*8B

	full_text

double* %262
Qstore8BF
D
	full_text7
5
3store double %1033, double* %92, align 16, !tbaa !8
-double8B

	full_text

double %1033
-double*8B

	full_text

double* %92
Lload8BB
@
	full_text3
1
/%1053 = load i64, i64* %189, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %189
Jstore8B?
=
	full_text0
.
,store i64 %1053, i64* %57, align 8, !tbaa !8
'i648B

	full_text

	i64 %1053
'i64*8B

	full_text


i64* %57
Kload8BA
?
	full_text2
0
.%1054 = load i64, i64* %270, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %270
Lstore8BA
?
	full_text2
0
.store i64 %1054, i64* %273, align 16, !tbaa !8
'i648B

	full_text

	i64 %1054
(i64*8B

	full_text

	i64* %273
Lload8BB
@
	full_text3
1
/%1055 = load i64, i64* %274, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %274
Kstore8B@
>
	full_text1
/
-store i64 %1055, i64* %270, align 8, !tbaa !8
'i648B

	full_text

	i64 %1055
(i64*8B

	full_text

	i64* %270
Kload8BA
?
	full_text2
0
.%1056 = load i64, i64* %276, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %276
Lstore8BA
?
	full_text2
0
.store i64 %1056, i64* %279, align 16, !tbaa !8
'i648B

	full_text

	i64 %1056
(i64*8B

	full_text

	i64* %279
Lstore8BA
?
	full_text2
0
.store i64 %1041, i64* %112, align 16, !tbaa !8
'i648B

	full_text

	i64 %1041
(i64*8B

	full_text

	i64* %112
Cbitcast8B6
4
	full_text'
%
#%1057 = bitcast i64 %1041 to double
'i648B

	full_text

	i64 %1041
getelementptr8B}
{
	full_textn
l
j%1058 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %28, i64 %30, i64 %32, i64 %882
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %28
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Rload8BH
F
	full_text9
7
5%1059 = load double, double* %1058, align 8, !tbaa !8
/double*8B 

	full_text

double* %1058
=fmul8B3
1
	full_text$
"
 %1060 = fmul double %1059, %1057
-double8B

	full_text

double %1059
-double8B

	full_text

double %1057
getelementptr8B}
{
	full_textn
l
j%1061 = getelementptr inbounds [33 x [33 x double]], [33 x [33 x double]]* %27, i64 %30, i64 %32, i64 %882
I[33 x [33 x double]]*8B,
*
	full_text

[33 x [33 x double]]* %27
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %882
Rload8BH
F
	full_text9
7
5%1062 = load double, double* %1061, align 8, !tbaa !8
/double*8B 

	full_text

double* %1061
Cbitcast8B6
4
	full_text'
%
#%1063 = bitcast i64 %1053 to double
'i648B

	full_text

	i64 %1053
=fsub8B3
1
	full_text$
"
 %1064 = fsub double %1063, %1062
-double8B

	full_text

double %1063
-double8B

	full_text

double %1062
Dfmul8B:
8
	full_text+
)
'%1065 = fmul double %1064, 4.000000e-01
-double8B

	full_text

double %1064
qcall8Bg
e
	full_textX
V
T%1066 = tail call double @llvm.fmuladd.f64(double %1057, double %1060, double %1065)
-double8B

	full_text

double %1057
-double8B

	full_text

double %1060
-double8B

	full_text

double %1065
Qstore8BF
D
	full_text7
5
3store double %1066, double* %123, align 8, !tbaa !8
-double8B

	full_text

double %1066
.double*8B

	full_text

double* %123
Cbitcast8B6
4
	full_text'
%
#%1067 = bitcast i64 %1045 to double
'i648B

	full_text

	i64 %1045
=fmul8B3
1
	full_text$
"
 %1068 = fmul double %1060, %1067
-double8B

	full_text

double %1060
-double8B

	full_text

double %1067
Rstore8BG
E
	full_text8
6
4store double %1068, double* %126, align 16, !tbaa !8
-double8B

	full_text

double %1068
.double*8B

	full_text

double* %126
Cbitcast8B6
4
	full_text'
%
#%1069 = bitcast i64 %1049 to double
'i648B

	full_text

	i64 %1049
=fmul8B3
1
	full_text$
"
 %1070 = fmul double %1060, %1069
-double8B

	full_text

double %1060
-double8B

	full_text

double %1069
Qstore8BF
D
	full_text7
5
3store double %1070, double* %129, align 8, !tbaa !8
-double8B

	full_text

double %1070
.double*8B

	full_text

double* %129
Dfmul8B:
8
	full_text+
)
'%1071 = fmul double %1062, 4.000000e-01
-double8B

	full_text

double %1062
Efsub8B;
9
	full_text,
*
(%1072 = fsub double -0.000000e+00, %1071
-double8B

	full_text

double %1071
xcall8Bn
l
	full_text_
]
[%1073 = tail call double @llvm.fmuladd.f64(double %1063, double 1.400000e+00, double %1072)
-double8B

	full_text

double %1063
-double8B

	full_text

double %1072
=fmul8B3
1
	full_text$
"
 %1074 = fmul double %1060, %1073
-double8B

	full_text

double %1060
-double8B

	full_text

double %1073
Rstore8BG
E
	full_text8
6
4store double %1074, double* %134, align 16, !tbaa !8
-double8B

	full_text

double %1074
.double*8B

	full_text

double* %134
=fmul8B3
1
	full_text$
"
 %1075 = fmul double %1059, %1067
-double8B

	full_text

double %1059
-double8B

	full_text

double %1067
=fmul8B3
1
	full_text$
"
 %1076 = fmul double %1059, %1069
-double8B

	full_text

double %1059
-double8B

	full_text

double %1069
=fmul8B3
1
	full_text$
"
 %1077 = fmul double %1059, %1063
-double8B

	full_text

double %1059
-double8B

	full_text

double %1063
Qload8BG
E
	full_text8
6
4%1078 = load double, double* %900, align 8, !tbaa !8
.double*8B

	full_text

double* %900
=fmul8B3
1
	full_text$
"
 %1079 = fmul double %1078, %1012
-double8B

	full_text

double %1078
-double8B

	full_text

double %1012
=fmul8B3
1
	full_text$
"
 %1080 = fmul double %1078, %1019
-double8B

	full_text

double %1078
-double8B

	full_text

double %1019
=fmul8B3
1
	full_text$
"
 %1081 = fmul double %1078, %1026
-double8B

	full_text

double %1078
-double8B

	full_text

double %1026
=fmul8B3
1
	full_text$
"
 %1082 = fmul double %1078, %1033
-double8B

	full_text

double %1078
-double8B

	full_text

double %1033
=fsub8B3
1
	full_text$
"
 %1083 = fsub double %1060, %1079
-double8B

	full_text

double %1060
-double8B

	full_text

double %1079
Jfmul8B@
>
	full_text1
/
-%1084 = fmul double %1083, 0x4045555555555555
-double8B

	full_text

double %1083
Qstore8BF
D
	full_text7
5
3store double %1084, double* %143, align 8, !tbaa !8
-double8B

	full_text

double %1084
.double*8B

	full_text

double* %143
=fsub8B3
1
	full_text$
"
 %1085 = fsub double %1075, %1080
-double8B

	full_text

double %1075
-double8B

	full_text

double %1080
Dfmul8B:
8
	full_text+
)
'%1086 = fmul double %1085, 3.200000e+01
-double8B

	full_text

double %1085
Qstore8BF
D
	full_text7
5
3store double %1086, double* %146, align 8, !tbaa !8
-double8B

	full_text

double %1086
.double*8B

	full_text

double* %146
=fsub8B3
1
	full_text$
"
 %1087 = fsub double %1076, %1081
-double8B

	full_text

double %1076
-double8B

	full_text

double %1081
Dfmul8B:
8
	full_text+
)
'%1088 = fmul double %1087, 3.200000e+01
-double8B

	full_text

double %1087
Qstore8BF
D
	full_text7
5
3store double %1088, double* %149, align 8, !tbaa !8
-double8B

	full_text

double %1088
.double*8B

	full_text

double* %149
=fmul8B3
1
	full_text$
"
 %1089 = fmul double %1075, %1075
-double8B

	full_text

double %1075
-double8B

	full_text

double %1075
qcall8Bg
e
	full_textX
V
T%1090 = tail call double @llvm.fmuladd.f64(double %1060, double %1060, double %1089)
-double8B

	full_text

double %1060
-double8B

	full_text

double %1060
-double8B

	full_text

double %1089
qcall8Bg
e
	full_textX
V
T%1091 = tail call double @llvm.fmuladd.f64(double %1076, double %1076, double %1090)
-double8B

	full_text

double %1076
-double8B

	full_text

double %1076
-double8B

	full_text

double %1090
=fmul8B3
1
	full_text$
"
 %1092 = fmul double %1080, %1080
-double8B

	full_text

double %1080
-double8B

	full_text

double %1080
qcall8Bg
e
	full_textX
V
T%1093 = tail call double @llvm.fmuladd.f64(double %1079, double %1079, double %1092)
-double8B

	full_text

double %1079
-double8B

	full_text

double %1079
-double8B

	full_text

double %1092
qcall8Bg
e
	full_textX
V
T%1094 = tail call double @llvm.fmuladd.f64(double %1081, double %1081, double %1093)
-double8B

	full_text

double %1081
-double8B

	full_text

double %1081
-double8B

	full_text

double %1093
=fsub8B3
1
	full_text$
"
 %1095 = fsub double %1091, %1094
-double8B

	full_text

double %1091
-double8B

	full_text

double %1094
=fmul8B3
1
	full_text$
"
 %1096 = fmul double %1079, %1079
-double8B

	full_text

double %1079
-double8B

	full_text

double %1079
Efsub8B;
9
	full_text,
*
(%1097 = fsub double -0.000000e+00, %1096
-double8B

	full_text

double %1096
qcall8Bg
e
	full_textX
V
T%1098 = tail call double @llvm.fmuladd.f64(double %1060, double %1060, double %1097)
-double8B

	full_text

double %1060
-double8B

	full_text

double %1060
-double8B

	full_text

double %1097
Jfmul8B@
>
	full_text1
/
-%1099 = fmul double %1098, 0x4015555555555555
-double8B

	full_text

double %1098
~call8Bt
r
	full_texte
c
a%1100 = tail call double @llvm.fmuladd.f64(double %1095, double 0xC02EB851EB851EB6, double %1099)
-double8B

	full_text

double %1095
-double8B

	full_text

double %1099
=fsub8B3
1
	full_text$
"
 %1101 = fsub double %1077, %1082
-double8B

	full_text

double %1077
-double8B

	full_text

double %1082
~call8Bt
r
	full_texte
c
a%1102 = tail call double @llvm.fmuladd.f64(double %1101, double 0x404F5C28F5C28F5B, double %1100)
-double8B

	full_text

double %1101
-double8B

	full_text

double %1100
Qstore8BF
D
	full_text7
5
3store double %1102, double* %164, align 8, !tbaa !8
-double8B

	full_text

double %1102
.double*8B

	full_text

double* %164
¦getelementptr8B’

	full_text

}%1103 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %899, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Rload8BH
F
	full_text9
7
5%1104 = load double, double* %1103, align 8, !tbaa !8
/double*8B 

	full_text

double* %1103
Rload8BH
F
	full_text9
7
5%1105 = load double, double* %202, align 16, !tbaa !8
.double*8B

	full_text

double* %202
=fsub8B3
1
	full_text$
"
 %1106 = fsub double %1057, %1105
-double8B

	full_text

double %1057
-double8B

	full_text

double %1105
ycall8Bo
m
	full_text`
^
\%1107 = tail call double @llvm.fmuladd.f64(double %1106, double -1.600000e+01, double %1104)
-double8B

	full_text

double %1106
-double8B

	full_text

double %1104
¦getelementptr8B’

	full_text

}%1108 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %899, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Rload8BH
F
	full_text9
7
5%1109 = load double, double* %1108, align 8, !tbaa !8
/double*8B 

	full_text

double* %1108
Qload8BG
E
	full_text8
6
4%1110 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
=fsub8B3
1
	full_text$
"
 %1111 = fsub double %1066, %1110
-double8B

	full_text

double %1066
-double8B

	full_text

double %1110
ycall8Bo
m
	full_text`
^
\%1112 = tail call double @llvm.fmuladd.f64(double %1111, double -1.600000e+01, double %1109)
-double8B

	full_text

double %1111
-double8B

	full_text

double %1109
¦getelementptr8B’

	full_text

}%1113 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %899, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Rload8BH
F
	full_text9
7
5%1114 = load double, double* %1113, align 8, !tbaa !8
/double*8B 

	full_text

double* %1113
Rload8BH
F
	full_text9
7
5%1115 = load double, double* %237, align 16, !tbaa !8
.double*8B

	full_text

double* %237
=fsub8B3
1
	full_text$
"
 %1116 = fsub double %1068, %1115
-double8B

	full_text

double %1068
-double8B

	full_text

double %1115
ycall8Bo
m
	full_text`
^
\%1117 = tail call double @llvm.fmuladd.f64(double %1116, double -1.600000e+01, double %1114)
-double8B

	full_text

double %1116
-double8B

	full_text

double %1114
¦getelementptr8B’

	full_text

}%1118 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %899, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Rload8BH
F
	full_text9
7
5%1119 = load double, double* %1118, align 8, !tbaa !8
/double*8B 

	full_text

double* %1118
Qload8BG
E
	full_text8
6
4%1120 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
=fsub8B3
1
	full_text$
"
 %1121 = fsub double %1070, %1120
-double8B

	full_text

double %1070
-double8B

	full_text

double %1120
ycall8Bo
m
	full_text`
^
\%1122 = tail call double @llvm.fmuladd.f64(double %1121, double -1.600000e+01, double %1119)
-double8B

	full_text

double %1121
-double8B

	full_text

double %1119
¦getelementptr8B’

	full_text

}%1123 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %94, i64 %30, i64 %32, i64 %899, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %32
&i648B

	full_text


i64 %899
Rload8BH
F
	full_text9
7
5%1124 = load double, double* %1123, align 8, !tbaa !8
/double*8B 

	full_text

double* %1123
Cbitcast8B6
4
	full_text'
%
#%1125 = bitcast i64 %1054 to double
'i648B

	full_text

	i64 %1054
=fsub8B3
1
	full_text$
"
 %1126 = fsub double %1074, %1125
-double8B

	full_text

double %1074
-double8B

	full_text

double %1125
ycall8Bo
m
	full_text`
^
\%1127 = tail call double @llvm.fmuladd.f64(double %1126, double -1.600000e+01, double %1124)
-double8B

	full_text

double %1126
-double8B

	full_text

double %1124
Qload8BG
E
	full_text8
6
4%1128 = load double, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
Qload8BG
E
	full_text8
6
4%1129 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
ycall8Bo
m
	full_text`
^
\%1130 = tail call double @llvm.fmuladd.f64(double %1129, double -2.000000e+00, double %1128)
-double8B

	full_text

double %1129
-double8B

	full_text

double %1128
Pload8BF
D
	full_text7
5
3%1131 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
=fadd8B3
1
	full_text$
"
 %1132 = fadd double %1130, %1131
-double8B

	full_text

double %1130
-double8B

	full_text

double %1131
xcall8Bn
l
	full_text_
]
[%1133 = tail call double @llvm.fmuladd.f64(double %1132, double 7.680000e+02, double %1107)
-double8B

	full_text

double %1132
-double8B

	full_text

double %1107
Qload8BG
E
	full_text8
6
4%1134 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
=fsub8B3
1
	full_text$
"
 %1135 = fsub double %1084, %1134
-double8B

	full_text

double %1084
-double8B

	full_text

double %1134
xcall8Bn
l
	full_text_
]
[%1136 = tail call double @llvm.fmuladd.f64(double %1135, double 3.200000e+00, double %1112)
-double8B

	full_text

double %1135
-double8B

	full_text

double %1112
Qload8BG
E
	full_text8
6
4%1137 = load double, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
ycall8Bo
m
	full_text`
^
\%1138 = tail call double @llvm.fmuladd.f64(double %1012, double -2.000000e+00, double %1137)
-double8B

	full_text

double %1012
-double8B

	full_text

double %1137
=fadd8B3
1
	full_text$
"
 %1139 = fadd double %1138, %1057
-double8B

	full_text

double %1138
-double8B

	full_text

double %1057
xcall8Bn
l
	full_text_
]
[%1140 = tail call double @llvm.fmuladd.f64(double %1139, double 7.680000e+02, double %1136)
-double8B

	full_text

double %1139
-double8B

	full_text

double %1136
Rload8BH
F
	full_text9
7
5%1141 = load double, double* %243, align 16, !tbaa !8
.double*8B

	full_text

double* %243
=fsub8B3
1
	full_text$
"
 %1142 = fsub double %1086, %1141
-double8B

	full_text

double %1086
-double8B

	full_text

double %1141
xcall8Bn
l
	full_text_
]
[%1143 = tail call double @llvm.fmuladd.f64(double %1142, double 3.200000e+00, double %1117)
-double8B

	full_text

double %1142
-double8B

	full_text

double %1117
Qload8BG
E
	full_text8
6
4%1144 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
ycall8Bo
m
	full_text`
^
\%1145 = tail call double @llvm.fmuladd.f64(double %1019, double -2.000000e+00, double %1144)
-double8B

	full_text

double %1019
-double8B

	full_text

double %1144
=fadd8B3
1
	full_text$
"
 %1146 = fadd double %1145, %1067
-double8B

	full_text

double %1145
-double8B

	full_text

double %1067
xcall8Bn
l
	full_text_
]
[%1147 = tail call double @llvm.fmuladd.f64(double %1146, double 7.680000e+02, double %1143)
-double8B

	full_text

double %1146
-double8B

	full_text

double %1143
Qload8BG
E
	full_text8
6
4%1148 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
=fsub8B3
1
	full_text$
"
 %1149 = fsub double %1088, %1148
-double8B

	full_text

double %1088
-double8B

	full_text

double %1148
xcall8Bn
l
	full_text_
]
[%1150 = tail call double @llvm.fmuladd.f64(double %1149, double 3.200000e+00, double %1122)
-double8B

	full_text

double %1149
-double8B

	full_text

double %1122
Qload8BG
E
	full_text8
6
4%1151 = load double, double* %245, align 8, !tbaa !8
.double*8B

	full_text

double* %245
ycall8Bo
m
	full_text`
^
\%1152 = tail call double @llvm.fmuladd.f64(double %1026, double -2.000000e+00, double %1151)
-double8B

	full_text

double %1026
-double8B

	full_text

double %1151
=fadd8B3
1
	full_text$
"
 %1153 = fadd double %1152, %1069
-double8B

	full_text

double %1152
-double8B

	full_text

double %1069
xcall8Bn
l
	full_text_
]
[%1154 = tail call double @llvm.fmuladd.f64(double %1153, double 7.680000e+02, double %1150)
-double8B

	full_text

double %1153
-double8B

	full_text

double %1150
Rload8BH
F
	full_text9
7
5%1155 = load double, double* %278, align 16, !tbaa !8
.double*8B

	full_text

double* %278
=fsub8B3
1
	full_text$
"
 %1156 = fsub double %1102, %1155
-double8B

	full_text

double %1102
-double8B

	full_text

double %1155
xcall8Bn
l
	full_text_
]
[%1157 = tail call double @llvm.fmuladd.f64(double %1156, double 3.200000e+00, double %1127)
-double8B

	full_text

double %1156
-double8B

	full_text

double %1127
Qload8BG
E
	full_text8
6
4%1158 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
ycall8Bo
m
	full_text`
^
\%1159 = tail call double @llvm.fmuladd.f64(double %1033, double -2.000000e+00, double %1158)
-double8B

	full_text

double %1033
-double8B

	full_text

double %1158
=fadd8B3
1
	full_text$
"
 %1160 = fadd double %1159, %1063
-double8B

	full_text

double %1159
-double8B

	full_text

double %1063
xcall8Bn
l
	full_text_
]
[%1161 = tail call double @llvm.fmuladd.f64(double %1160, double 7.680000e+02, double %1157)
-double8B

	full_text

double %1160
-double8B

	full_text

double %1157
Rload8BH
F
	full_text9
7
5%1162 = load double, double* %193, align 16, !tbaa !8
.double*8B

	full_text

double* %193
ycall8Bo
m
	full_text`
^
\%1163 = tail call double @llvm.fmuladd.f64(double %1128, double -4.000000e+00, double %1162)
-double8B

	full_text

double %1128
-double8B

	full_text

double %1162
xcall8Bn
l
	full_text_
]
[%1164 = tail call double @llvm.fmuladd.f64(double %1129, double 5.000000e+00, double %1163)
-double8B

	full_text

double %1129
-double8B

	full_text

double %1163
pcall8Bf
d
	full_textW
U
S%1165 = tail call double @llvm.fmuladd.f64(double %407, double %1164, double %1133)
,double8B

	full_text

double %407
-double8B

	full_text

double %1164
-double8B

	full_text

double %1133
Rstore8BG
E
	full_text8
6
4store double %1165, double* %1103, align 8, !tbaa !8
-double8B

	full_text

double %1165
/double*8B 

	full_text

double* %1103
Qload8BG
E
	full_text8
6
4%1166 = load double, double* %214, align 8, !tbaa !8
.double*8B

	full_text

double* %214
ycall8Bo
m
	full_text`
^
\%1167 = tail call double @llvm.fmuladd.f64(double %1137, double -4.000000e+00, double %1166)
-double8B

	full_text

double %1137
-double8B

	full_text

double %1166
Pload8BF
D
	full_text7
5
3%1168 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
xcall8Bn
l
	full_text_
]
[%1169 = tail call double @llvm.fmuladd.f64(double %1168, double 5.000000e+00, double %1167)
-double8B

	full_text

double %1168
-double8B

	full_text

double %1167
pcall8Bf
d
	full_textW
U
S%1170 = tail call double @llvm.fmuladd.f64(double %407, double %1169, double %1140)
,double8B

	full_text

double %407
-double8B

	full_text

double %1169
-double8B

	full_text

double %1140
Rstore8BG
E
	full_text8
6
4store double %1170, double* %1108, align 8, !tbaa !8
-double8B

	full_text

double %1170
/double*8B 

	full_text

double* %1108
Rload8BH
F
	full_text9
7
5%1171 = load double, double* %231, align 16, !tbaa !8
.double*8B

	full_text

double* %231
ycall8Bo
m
	full_text`
^
\%1172 = tail call double @llvm.fmuladd.f64(double %1144, double -4.000000e+00, double %1171)
-double8B

	full_text

double %1144
-double8B

	full_text

double %1171
Qload8BG
E
	full_text8
6
4%1173 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
xcall8Bn
l
	full_text_
]
[%1174 = tail call double @llvm.fmuladd.f64(double %1173, double 5.000000e+00, double %1172)
-double8B

	full_text

double %1173
-double8B

	full_text

double %1172
pcall8Bf
d
	full_textW
U
S%1175 = tail call double @llvm.fmuladd.f64(double %407, double %1174, double %1147)
,double8B

	full_text

double %407
-double8B

	full_text

double %1174
-double8B

	full_text

double %1147
Rstore8BG
E
	full_text8
6
4store double %1175, double* %1113, align 8, !tbaa !8
-double8B

	full_text

double %1175
/double*8B 

	full_text

double* %1113
Qload8BG
E
	full_text8
6
4%1176 = load double, double* %248, align 8, !tbaa !8
.double*8B

	full_text

double* %248
ycall8Bo
m
	full_text`
^
\%1177 = tail call double @llvm.fmuladd.f64(double %1151, double -4.000000e+00, double %1176)
-double8B

	full_text

double %1151
-double8B

	full_text

double %1176
Pload8BF
D
	full_text7
5
3%1178 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
xcall8Bn
l
	full_text_
]
[%1179 = tail call double @llvm.fmuladd.f64(double %1178, double 5.000000e+00, double %1177)
-double8B

	full_text

double %1178
-double8B

	full_text

double %1177
pcall8Bf
d
	full_textW
U
S%1180 = tail call double @llvm.fmuladd.f64(double %407, double %1179, double %1154)
,double8B

	full_text

double %407
-double8B

	full_text

double %1179
-double8B

	full_text

double %1154
Rstore8BG
E
	full_text8
6
4store double %1180, double* %1118, align 8, !tbaa !8
-double8B

	full_text

double %1180
/double*8B 

	full_text

double* %1118
Rload8BH
F
	full_text9
7
5%1181 = load double, double* %265, align 16, !tbaa !8
.double*8B

	full_text

double* %265
ycall8Bo
m
	full_text`
^
\%1182 = tail call double @llvm.fmuladd.f64(double %1158, double -4.000000e+00, double %1181)
-double8B

	full_text

double %1158
-double8B

	full_text

double %1181
Qload8BG
E
	full_text8
6
4%1183 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
xcall8Bn
l
	full_text_
]
[%1184 = tail call double @llvm.fmuladd.f64(double %1183, double 5.000000e+00, double %1182)
-double8B

	full_text

double %1183
-double8B

	full_text

double %1182
pcall8Bf
d
	full_textW
U
S%1185 = tail call double @llvm.fmuladd.f64(double %407, double %1184, double %1161)
,double8B

	full_text

double %407
-double8B

	full_text

double %1184
-double8B

	full_text

double %1161
Rstore8BG
E
	full_text8
6
4store double %1185, double* %1123, align 8, !tbaa !8
-double8B

	full_text

double %1185
/double*8B 

	full_text

double* %1123
)br8B!

	full_text

br label %1186
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


double* %3
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %1
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
	
i64 200
5double8B'
%
	full_text

double -4.000000e+00
:double8B,
*
	full_text

double 0xC02EB851EB851EB6
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0x4045555555555555
$i328B

	full_text


i32 -2
%i648B

	full_text
	
i64 120
5double8B'
%
	full_text

double -2.000000e+00
#i328B

	full_text	

i32 6
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 1.400000e+00
#i648B

	full_text	

i64 3
4double8B&
$
	full_text

double 2.500000e-01
$i328B

	full_text


i32 -1
:double8B,
*
	full_text

double 0x4015555555555555
:double8B,
*
	full_text

double 0x404F5C28F5C28F5B
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
4double8B&
$
	full_text

double 7.680000e+02
$i648B

	full_text


i64 80
4double8B&
$
	full_text

double 3.200000e+01
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 4
5double8B'
%
	full_text

double -1.600000e+01
$i328B

	full_text


i32 -3
4double8B&
$
	full_text

double 4.000000e-01
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 1.000000e+00
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 4.000000e+00
4double8B&
$
	full_text

double 3.200000e+00
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
4double8B&
$
	full_text

double 6.000000e+00       	  
 

                       !" !# !! $% $& '' (( )* )) +, ++ -. -- /0 // 12 13 14 11 56 55 78 77 9: 99 ;< ;; => =? =@ == AB AA CD CC EF EE GH GG IJ IK IL II MN MM OP OO QR QQ ST SS UV UW UX UU YZ YY [\ [[ ]^ ]] _` __ ab ac ad aa ef ee gh gg ij ii kl kk mn mm op oo qr qs qq tu tt vw vx vy vv z{ zz |} |~ || € 	 	‚  ƒ„ ƒƒ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ Œ
 Œ
 ŒŒ ‘  ’“ ’
” ’’ •– •• —˜ —
™ —— š› šš œ œ
 œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «
¬ «« ­® ­
¯ ­­ °± °
² °° ³´ ³³ µ¶ µ
· µµ ¸¹ ¸¸ º» ºº ¼½ ¼¼ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ ÚÚ Üİ Ü
Ş ÜÜ ßß àá àà âã â
ä â
å ââ æç ææ èé èè êë ê
ì êê íî í
ï í
ğ íí ñò ññ óô óó õö õ
÷ õõ øù ø
ú ø
û øø üı üü şÿ şş € €
‚ €€ ƒ„ ƒ
… ƒ
† ƒƒ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹
 ‹‹  
 
‘  ’“ ’’ ”• ”” –— –
˜ –– ™š ™™ ›œ ››  
Ÿ   ¡    ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «
® «« ¯° ¯¯ ±² ±± ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸
º ¸
» ¸¸ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ ĞĞ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×
Ø ×× ÙÚ Ù
Û ÙÙ Üİ Ü
Ş ÜÜ ßà ßß áâ á
ã áá äå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï íí ğñ ğ
ò ğğ óô ó
õ óó ö÷ ö
ø öö ùú ùù ûü ûû ış ı
ÿ ıı € €
‚ €€ ƒ„ ƒƒ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ     ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —
š —— ›œ ›
 ›
 ›› Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦
¨ ¦
© ¦¦ ª« ª
¬ ªª ­® ­
¯ ­­ °
± °° ²³ ²
´ ²
µ ²² ¶· ¶¶ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È Æ
É ÆÆ ÊË ÊÊ ÌÍ ÌÌ ÎÏ ÎÎ ĞÑ ĞĞ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× Õ
Ø ÕÕ ÙÚ ÙÙ ÛÜ ÛÛ İŞ İİ ßà ßß áâ á
ã áá äå ä
æ ä
ç ää èé èè êë êê ìí ìì îï îî ğñ ğ
ò ğğ óô ó
õ ó
ö óó ÷ø ÷÷ ùú ùù ûü ûû ış ıı ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †† ˆ‰ ˆˆ Š‹ ŠŠ Œ ŒŒ  
  ‘’ ‘‘ “” ““ •– •• —˜ —— ™š ™™ ›œ ›
 ›› Ÿ   ¡  
¢    £¤ ££ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±± ³´ ³³ µ¶ µµ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ ĞĞ ÒÓ ÒÒ ÔÕ ÔÔ Ö× ÖÖ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İŞ İİ ßà ß
á ßß âã ââ äå ä
æ ää çè ç
é çç êë êê ìí ìì îï îî ğñ ğğ òó ò
ô òò õö õõ ÷ø ÷÷ ùú ù
û ùù üı üü şÿ şş € €€ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹‹     ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™™ ›œ ›
 ›› Ÿ 
   ¡¢ ¡¡ £¤ ££ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®® °± °
² °° ³´ ³³ µ¶ µµ ·¸ ·· ¹º ¹¹ »¼ »
½ »» ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ ĞĞ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ ÚÚ Üİ ÜÜ Şß ŞŞ àá à
â àà ãä ãã åæ åå çè ç
é çç êë êê ìí ìì îï îî ğñ ğğ òó ò
ô òò õö õõ ÷ø ÷÷ ùú ùù ûü ûû ış ıı ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ  
  ‘’ ‘‘ “” ““ •– •• —˜ —— ™š ™
› ™™ œ œœ Ÿ   ¡  
¢    £¤ ££ ¥¦ ¥¥ §¨ §§ ©ª ©© «¬ «
­ «« ®¯ ®
° ®
± ®® ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹
¼ ¹¹ ½¾ ½½ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ ÈÈ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏĞ Ï
Ñ Ï
Ò ÏÏ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü Ú
İ ÚÚ Şß ŞŞ àá àà âã â
ä ââ åæ åå çè ç
é çç êë êê ìí ì
î ì
ï ìì ğñ ğğ òó ò
ô òò õö õ
÷ õ
ø õõ ùú ùù ûü ûû ış ı
ÿ ıı € €€ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹  
  ‘’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™™ ›
œ ››  
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ ÎÎ ĞÑ Ğ
Ò ĞĞ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ Û
İ ÛÛ Şß Ş
à Ş
á ŞŞ âã â
ä â
å ââ æç æ
è ææ éê é
ë é
ì éé íî í
ï í
ğ íí ñò ñ
ó ññ ôõ ô
ö ôô ÷
ø ÷÷ ùú ù
û ù
ü ùù ış ıı ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹
 ‹‹   ‘’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™
› ™
œ ™™   Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §
ª §§ «¬ «« ­® ­­ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µ
¸ µµ ¹º ¹¹ »¼ »» ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å Ã
Æ ÃÃ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ Î
Ğ ÎÎ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá àà âã â
ä ââ åæ å
ç åå èé èè êë ê
ì êê íî í
ï íí ğñ ğ
ò ğğ óô óó õö õ
÷ õõ øù ø
ú øø ûü ûû ış ı
ÿ ıı €		 €	
‚	 €	€	 ƒ	„	 ƒ	
…	 ƒ	ƒ	 †	‡	 †	†	 ˆ	‰	 ˆ	
Š	 ˆ	ˆ	 ‹	Œ	 ‹	
	 ‹	‹	 		 		 	‘	 	
’	 		 “	”	 “	
•	 “	“	 –	—	 –	
˜	 –	–	 ™	š	 ™	™	 ›	œ	 ›	
	 ›	›	 	Ÿ	 	
 	 		 ¡	¢	 ¡	¡	 £	¤	 £	
¥	 £	£	 ¦	§	 ¦	
¨	 ¦	¦	 ©	ª	 ©	
«	 ©	©	 ¬	¬	 ­	®	 ­	­	 ¯	°	 ¯	¯	 ±	
²	 ±	±	 ³	´	 ³	³	 µ	
¶	 µ	µ	 ·	¸	 ·	
¹	 ·	·	 º	»	 º	º	 ¼	½	 ¼	
¾	 ¼	¼	 ¿	À	 ¿	
Á	 ¿	
Â	 ¿	¿	 Ã	Ä	 Ã	
Å	 Ã	Ã	 Æ	Ç	 Æ	Æ	 È	É	 È	È	 Ê	Ë	 Ê	Ê	 Ì	
Í	 Ì	Ì	 Î	Ï	 Î	
Ğ	 Î	Î	 Ñ	Ò	 Ñ	Ñ	 Ó	Ô	 Ó	
Õ	 Ó	Ó	 Ö	×	 Ö	
Ø	 Ö	
Ù	 Ö	Ö	 Ú	Û	 Ú	
Ü	 Ú	Ú	 İ	Ş	 İ	İ	 ß	à	 ß	ß	 á	â	 á	á	 ã	
ä	 ã	ã	 å	æ	 å	
ç	 å	å	 è	é	 è	è	 ê	ë	 ê	
ì	 ê	ê	 í	î	 í	
ï	 í	
ğ	 í	í	 ñ	ò	 ñ	
ó	 ñ	ñ	 ô	õ	 ô	ô	 ö	÷	 ö	ö	 ø	ù	 ø	ø	 ú	
û	 ú	ú	 ü	ı	 ü	
ş	 ü	ü	 ÿ	€
 ÿ	ÿ	 
‚
 

ƒ
 

 „
…
 „

†
 „

‡
 „
„
 ˆ
‰
 ˆ

Š
 ˆ
ˆ
 ‹
Œ
 ‹
‹
 

 

 

 

 ‘

’
 ‘
‘
 “
”
 “

•
 “
“
 –
—
 –
–
 ˜
™
 ˜

š
 ˜
˜
 ›
œ
 ›


 ›


 ›
›
 Ÿ
 
 Ÿ

¡
 Ÿ
Ÿ
 ¢
£
 ¢
¢
 ¤
¥
 ¤

¦
 ¤
¤
 §
¨
 §

©
 §
§
 ª
«
 ª

¬
 ª
ª
 ­
®
 ­

¯
 ­
­
 °
±
 °
°
 ²
³
 ²

´
 ²
²
 µ
¶
 µ
µ
 ·
¸
 ·

¹
 ·
·
 º
»
 º
º
 ¼
½
 ¼

¾
 ¼
¼
 ¿
À
 ¿

Á
 ¿
¿
 Â
Ã
 Â

Ä
 Â
Â
 Å
Æ
 Å

Ç
 Å
Å
 È
É
 È

Ê
 È
È
 Ë
Ì
 Ë
Ë
 Í
Î
 Í

Ï
 Í
Í
 Ğ
Ñ
 Ğ
Ğ
 Ò
Ó
 Ò

Ô
 Ò
Ò
 Õ
Ö
 Õ
Õ
 ×
Ø
 ×

Ù
 ×
×
 Ú
Û
 Ú

Ü
 Ú
Ú
 İ
Ş
 İ

ß
 İ
İ
 à
á
 à

â
 à
à
 ã
ä
 ã

å
 ã
ã
 æ
ç
 æ
æ
 è
é
 è

ê
 è
è
 ë
ì
 ë
ë
 í
î
 í

ï
 í
í
 ğ
ñ
 ğ
ğ
 ò
ó
 ò

ô
 ò
ò
 õ
ö
 õ
õ
 ÷
ø
 ÷

ù
 ÷
÷
 ú
û
 ú

ü
 ú
ú
 ı
ş
 ı

ÿ
 ı
ı
 € €
‚ €€ ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ    
‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› š
œ šš  
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯
² ¯¯ ³´ ³³ µ¶ µµ ·¸ ·
¹ ·· º» º
¼ º
½ ºº ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç Å
È ÅÅ ÉÊ ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ Ğ
Ò Ğ
Ó ĞĞ ÔÕ ÔÔ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ Û
İ Û
Ş ÛÛ ßà ßß áâ áá ãä ã
å ãã æç ææ èé è
ê èè ëì ë
í ë
î ëë ïğ ïï ñò ñ
ó ññ ôõ ô
ö ô
÷ ôô øù øø úû ú
ü úú ış ıı ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
 ŒŒ  
‘  ’“ ’’ ”
• ”” –— –
˜ –– ™š ™
› ™™ œ œ
 œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ Î
Ğ ÎÎ ÑÒ Ñ
Ó Ñ
Ô ÑÑ ÕÖ Õ
× Õ
Ø ÕÕ ÙÚ Ù
Û ÙÙ Üİ Ü
Ş Ü
ß ÜÜ àá à
â à
ã àà äå ä
æ ää çè ç
é çç ê
ë êê ìí ì
î ì
ï ìì ğñ ğğ òó ò
ô òò õö õ
÷ õõ øù ø
ú øø ûü û
ı ûû şÿ ş
€ ş
 şş ‚ƒ ‚‚ „… „„ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
 Œ
 ŒŒ ‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› š
œ š
 šš Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®® °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶
¹ ¶¶ º» ºº ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ Ğ
Ò ĞĞ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ ÛÛ İŞ İ
ß İİ àá à
â àà ãä ã
å ãã æç ææ èé è
ê èè ëì ë
í ëë îï îî ğñ ğ
ò ğğ óô ó
õ óó ö÷ ö
ø öö ùú ùù ûü û
ı ûû şÿ ş
€ şş ‚  ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ ŒŒ  
  ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
 œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬
¯ ¬¬ °± °
² °° ³´ ³³ µ¶ µµ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ ÏĞ Ï
Ñ ÏÏ ÒÓ ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×× ÙÚ Ù
Û ÙÙ Üİ Ü
Ş Ü
ß ÜÜ àá à
â àà ãä ãã åæ åå çè ç
é çç êë êê ìí ì
î ìì ïğ ïï ñò ñ
ó ññ ôõ ô
ö ô
÷ ôô øù ø
ú øø ûü ûû ış ıı ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ Œ
 Œ
 ŒŒ ‘ 
’  ““ ”” •– •• —˜ —— ™š ™™ ›œ ››   Ÿ  ŸŸ ¡¢ ¡¡ £¤ ££ ¥¦ ¥¥ §¨ §§ ©ª ©¬ «« ­® ­­ ¯° ¯¯ ±² ±± ³´ ³³ µ¶ µµ ·¸ ·· ¹º ¹¹ »¼ »» ½¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ĞÑ Ğ
Ò ĞĞ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ Üİ Ü
Ş ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ğ îî ñò ñ
ó ññ ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ış ı
ÿ ıı € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ
 ŒŒ   ‘ 
’  “” “
• ““ –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› Ÿ 
   ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´´ ¶· ¶¶ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ Î
Ğ ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ ÛÛ İŞ İ
ß İ
à İ
á İİ âã ââ äå ää æç æ
è ææ éê é
ë é
ì é
í éé îï îî ğñ ğğ òó ò
ô òò õö õ
÷ õ
ø õ
ù õõ úû úú üı üü şÿ ş
€ şş ‚ 
ƒ 
„ 
…  †‡ †† ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ  
 
 
‘  ’“ ’’ ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
 œ
Ÿ œ
  œœ ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦
© ¦
ª ¦¦ «¬ «« ­® ­
¯ ­­ °± °° ²³ ²
´ ²
µ ²² ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏĞ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
İ Û
Ş Û
ß ÛÛ àá àà âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ğ îî ñò ññ óô ó
õ óó ö÷ ö
ø öö ùú ùù ûü û
ı ûû şÿ ş
€ şş ‚  ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰
Œ ‰‰  
 
  ‘’ ‘
“ ‘‘ ”• ”
– ”
— ”” ˜™ ˜
š ˜
› ˜˜ œ œ
 œœ Ÿ  Ÿ
¡ ŸŸ ¢
£ ¢¢ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶
¹ ¶
º ¶¶ »¼ »» ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç Å
È Å
É ÅÅ ÊË ÊÊ ÌÍ ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö Ô
× Ô
Ø ÔÔ ÙÚ ÙÙ ÛÜ ÛÛ İŞ İ
ß İİ àá à
â àà ãä ã
å ã
æ ã
ç ãã èé èè êë êê ìí ì
î ìì ïğ ï
ñ ïï òó ò
ô ò
õ ò
ö òò ÷ø ÷÷ ùú ùù ûü û
ı ûû şÿ ş
€ şş ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ ŒŒ  
  ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
 œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏĞ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
İ ÛÛ Şß Ş
à ŞŞ áâ áá ãä ã
å ãã æç æ
è æ
é ææ êë ê
ì êê íî íí ïğ ï
ñ ïï òó ò
ô òò õö õõ ÷ø ÷
ù ÷÷ úû úú üı ü
ş üü ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹  
  ‘’ ‘‘ “” “
• ““ –— –
˜ –
™ –– š› š
œ šš   Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­
° ­­ ±² ±
³ ±± ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ ÎÎ ĞÑ ĞĞ ÒÓ ÒÒ ÔÕ ÔÔ Ö× ÖÖ ØÙ ØØ ÚÛ ÚÚ Üİ ÜÜ Şß ŞŞ àá àà âã âå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï íí ğñ ğ
ò ğğ óô ó
õ óó ö÷ ö
ø öö ùú ù
û ùù üı ü
ş üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹  
  ‘’ ‘
“ ‘‘ ”• ”” –˜ —
™ —— š› š
œ šš  
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÍ ÎÏ ÎÎ ĞÑ Ğ
Ò ĞĞ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá àà âã â
ä ââ åæ åå çè ç
é çç êë êê ìí ì
î ìì ïğ ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù ø
ú øø ûü ûû ış ı
ÿ ıı € €€ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ  
  ‘ 
’  “” “
• ““ –— –– ˜™ ˜
š ˜˜ ›œ ››  
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±± ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÛ Üİ ÜÜ Şß Ş
à Ş
á Ş
â ŞŞ ãä ãã åæ åå çè ç
é çç êë ê
ì ê
í ê
î êê ïğ ïï ñò ññ óô ó
õ óó ö÷ ö
ø ö
ù ö
ú öö ûü ûû ış ıı ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚
† ‚‚ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹
 ‹‹  
 
‘ 
’  “” ““ •– •• —˜ —
™ —— š› šš œ œ
 œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡
¤ ¡
¥ ¡¡ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «
® «
¯ «« °± °° ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·
º ·· »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË ÊÊ Ì
Í ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá àà âã â
ä â
å â
æ ââ çè çç éê é
ë éé ìí ì
î ìì ïğ ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù øø úû ú
ü úú ış ı
ÿ ıı € €€ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ  
  ‘ 
’ 
“  ”• ”
– ”
— ”” ˜™ ˜
š ˜˜ ›œ ›
 ›
 ›› Ÿ  Ÿ
¡ Ÿ
¢ ŸŸ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©
ª ©© «¬ «
­ «
® «« ¯° ¯¯ ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½
¿ ½
À ½
Á ½½ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î Ì
Ï Ì
Ğ ÌÌ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
İ Û
Ş Û
ß ÛÛ àá àà âã ââ äå ä
æ ää çè ç
é çç êë ê
ì ê
í ê
î êê ïğ ïï ñò ññ óô ó
õ óó ö÷ ö
ø öö ùú ù
û ù
ü ù
ı ùù şÿ şş € €€ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆˆ Š‹ ŠŠ Œ Œ
 ŒŒ   ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™
› ™™ œ œ
 œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ĞÑ ĞĞ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá à
â àà ãä ãã åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ğ î
ñ îî òó ò
ô òò õö õõ ÷ø ÷
ù ÷÷ úû úú üı ü
ş üü ÿ€ ÿÿ ‚ 
ƒ  „… „
† „
‡ „„ ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹  
  ‘  ’“ ’
” ’’ •– •• —˜ —
™ —— š› š
œ š
 šš Ÿ 
   ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °
² °
³ °° ´µ ´
¶ ´´ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È Æ
É ÆÆ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏĞ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İŞ İİ ßà ß
á ßß âã ââ äå ä
æ ää çè çç éê é
ë éé ìí ì
î ìì ïğ ï
ñ ïï òó ò
ô òò õö õõ ÷ø ÷
ù ÷÷ úû úú üı ü
ş üü ÿ€ ÿÿ ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
 ŒŒ  
‘  ’“ ’’ ”• ”
– ”” —˜ —— ™š ™
› ™™ œ œœ Ÿ 
   ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ ÛÛ İŞ İ
ß İİ àá à
â àà ãä ãã åæ å
ç å
è å
é åå êë êê ìí ì
î ìì ïğ ï
ñ ï
ò ï
ó ïï ôõ ôô ö÷ öö øù ø
ú øø ûü ûû ış ı
ÿ ı
€ ıı ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ ŒŒ  
  ‘’ ‘
“ ‘‘ ”• ”” –
— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› Ÿ 
   ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ Ğ
Ò ĞĞ ÓÔ Ó
Õ Ó
Ö ÓÓ ×Ø ×
Ù ×
Ú ×× ÛÜ Û
İ ÛÛ Şß Ş
à Ş
á ŞŞ âã â
ä â
å ââ æç æ
è ææ éê é
ë éé ì
í ìì îï î
ğ î
ñ îî òó òò ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ış ı
ÿ ıı € €
‚ €
ƒ €
„ €€ …† …… ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ Œ
 ŒŒ  
‘ 
’ 
“  ”• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› Ÿ 
  
¡ 
¢  £¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­
° ­
± ­­ ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼
¿ ¼
À ¼¼ ÁÂ ÁÁ ÃÄ ÃÃ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ ÏĞ Ï
Ñ ÏÏ ÒÓ ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ ÚÚ Üİ Ü
Ş ÜÜ ßà ß
á ßß âã ââ äå ä
æ ää çè ç
é çç êë ê
ì êê íî íí ïğ ï
ñ ïï òó ò
ô òò õö õõ ÷ø ÷
ù ÷÷ úû ú
ü úú ış ı
ÿ ıı € €€ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ  
  ‘ 
’  “” ““ •– •
— •• ˜™ ˜
š ˜˜ ›œ ››  
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®
± ®® ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿
Â ¿¿ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ĞÑ Ğ
Ò Ğ
Ó ĞĞ ÔÕ Ô
Ö ÔÔ ×Ø ×× ÙÚ Ù
Û ÙÙ Üİ ÜÜ Şß Ş
à ŞŞ áâ á
ã á
ä áá åæ å
ç åå èé èè êë ê
ì êê íî íí ïğ ï
ñ ïï òó ò
ô ò
õ òò ö÷ ö
ø öö ù
û úú ü
ı üü ş
ÿ şş € (‚ ƒ  „ “„ ”„ Í„ Û… '† &‡ ß   	            " #! % *) , .- 0& 2+ 3/ 41 65 8 :9 <& >+ ?/ @= BA D FE H& J+ K/ LI NM P RQ T& V+ W/ XU ZY \ ^] `& b+ c/ da fe h ji l nm pC ro sC u( w+ x/ yv {z }t ~' €+ / ‚ „g †… ˆƒ ‰‡ ‹t | Š  ‘Œ “ ”O –| ˜• ™ ›— š [  | ¢Ÿ £ ¥¡ §¤ ¨ƒ ª© ¬… ®« ¯| ±­ ² ´° ¶³ · ¹¸ » ½¼ ¿7 Á¾ Â ÄÃ ÆC ÈÅ É ËÊ ÍO ÏÌ Ğ ÒÑ Ô[ ÖÓ × ÙØ Ûg İÚ Ş9 á& ã+ ä/ åâ çæ éè ë; ì& î+ ï/ ğí òñ ôó öG ÷& ù+ ú/ ûø ıü ÿş S ‚& „+ …/ †ƒ ˆ‡ Š‰ Œ_ & + / ‘ “’ •” —k ˜ š™ œó › Ÿó ¡( £+ ¤/ ¥¢ §¦ ©  ª' ¬+ ­/ ®« °” ²± ´¯ µ³ ·  ¹¨ º¶ » ½¸ ¿¼ Àş Â¨ ÄÁ Å ÇÃ ÉÆ Ê‰ Ì¨ ÎË Ï ÑÍ ÓĞ Ô¯ ÖÕ Ø± Ú× Û¨ İÙ Ş àÜ âß ã¦ åÁ æ¦ èË é¦ ë± ìz î• ïz ñŸ òz ô… õ¨ ÷| øö ú üù şû ÿä í ‚€ „ †ƒ ˆ… ‰ç ‹ğ ŒŠ   ’ “ä •ä –¨ ˜¨ ™” šç œç — í  í ¡| £| ¤Ÿ ¥ğ §ğ ¨¢ ©› «¦ ¬| ®| ¯­ ±¨ ³¨ ´° µ² ·ª ¹¶ ºê ¼ó ½» ¿¸ À Â¾ ÄÁ Å& Ç+ È/ ÉÆ ËÊ Í ÏÎ ÑÌ ÓĞ Ô& Ö+ ×/ ØÕ ÚÙ Ü Şİ àÛ âß ã& å+ æ/ çä éè ë íì ïê ñî ò& ô+ õ/ öó ø÷ ú üû şù €ı & ƒ+ „/ …‚ ‡† ‰ ‹Š ˆ Œ  ’‘ ”“ – ˜ š• œ™ ¾ Ÿ ¡“ ¢º ¤£ ¦¾ § ©¨ «Ì ­º ® °¯ ²± ´ ¶ ¸³ º· » ½¼ ¿¾ ÁÀ Ã± Ä ÆÅ ÈÇ Ê ÌÉ ÎË Ï ÑĞ ÓÒ Õ ×Ö ÙÔ ÛØ ÜÅ Şİ àÒ áG ãâ åÅ æÛ èG é ëê í ïî ñì óğ ô¼ öõ ø÷ úê ûû ıü ÿ € ƒş …‚ † ˆ‡ Š‰ Œ  ‹ ’ “Ì •” —‰ ˜S š™ œÌ ê ŸS  š ¢¡ ¤ ¦¥ ¨£ ª§ «Æ ­¬ ¯® ±¡ ²… ´³ ¶ ¸· ºµ ¼¹ ½ ¿¾ ÁÀ Ã ÅÄ ÇÂ ÉÆ ÊÓ ÌË ÎÀ Ï_ ÑĞ ÓÓ Ôù Ö_ ×¤ ÙØ Û İÜ ßÚ áŞ âĞ äã æå èØ é ëê í ïî ñì óğ ô öõ ø÷ ú üû şù €ı Ú ƒ‚ …÷ †k ˆ‡ ŠÚ ‹Œ Œ k ³ ’‘ ” –• ˜“ š— ›ß œ Ÿ ¡‘ ¢Á ¤£ ¦ ¨§ ª¥ ¬© ­& ¯+ °/ ±® ³² µ´ ·Ğ ¸& º+ »/ ¼¹ ¾½ À¿ Âß Ã& Å+ Æ/ ÇÄ ÉÈ ËÊ Íî Î& Ğ+ Ñ/ ÒÏ ÔÓ ÖÕ Øı Ù& Û+ Ü/ İÚ ßŞ áà ãŒ äG æå è› éå ë( í+ î/ ïì ñğ óê ô' ö+ ÷/ øõ úŒ üû şù ÿı ê ƒò „€ …‚ ‡¼ ˆê Šò Œ‰ ‹ Æ ù ’ò ”‘ •“ —Ğ ˜ù š™ œû › Ÿò ¡ ¢  ¤ß ¥ğ §‰ ¨ğ ª‘ «ğ ­û ®Ã °¦ ²¯ ³Ê µ¦ ·´ ¸Ğ º¦ ¼¹ ½‡ ¿¦ Á¾ Âò Ä± ÅÃ ÇÆ Éû Ê¦ Ì¶ ÍË ÏÎ Ñ… Ò© Ô» ÕÓ ×Ö Ù Ú¦ Ü¦ İò ßò àÛ á© ã© äŞ å¶ ç¶ è± ê± ëæ ì» î» ïé ğâ òí ó± õ± öô øò úò û÷ üù şñ €ı ¬ ƒÀ „‚ †ÿ ‡… ‰Á Šß Œ+ / ‹ µ ’ê ”‘ •“ — ˜ß š+ ›/ œ™ î  ‚ ¢Ÿ £¡ ¥ ¦ß ¨+ ©/ ª§ ¬¥ ®‹ °­ ±¯ ³« ´ß ¶+ ·/ ¸µ ºÜ ¼“ ¾» ¿½ Á¹ Âß Ä+ Å/ ÆÃ È• Ê  ÌÉ ÍË ÏÇ Ğ‘ Ò¼ ÔÓ ÖÑ ×à ÙÕ ÛØ ÜÚ Ş– ß€ áÆ ãà äâ æ¤ çĞ é¯ ëè ìê îê ïí ñå ò· ôÎ öó ÷õ ù² ú‡ ü´ şû ÿı 	‰ ‚	€	 „	ø …	î ‡	Ö ‰	†	 Š	ˆ	 Œ	À 	¾ 	¹ ‘		 ’		 ”	‘ •	“	 —	‹	 ˜	§ š	… œ	™	 	›	 Ÿ	Î  	õ ¢	¾ ¤	¡	 ¥	£	 §	û ¨	¦	 ª		 «	¬	 ®	­	 °	¯	 ²	Ø ´	³	 ¶	Ó ¸	µ	 ¹	¨ »	º	 ½	·	 ¾	±	 À	¼	 Á	İ Â	¿	 Ä	‹ Å	Ã Ç	E É	È	 Ë	Ê	 Í	Æ	 Ï	Ì	 Ğ	İ Ò	Ñ	 Ô	Î	 Õ	±	 ×	Ó	 Ø	ğ Ù	Ö	 Û	™ Ü	Ê Ş	Q à	ß	 â	á	 ä	İ	 æ	ã	 ç	ì é	è	 ë	å	 ì	±	 î	ê	 ï	ƒ	 ğ	í	 ò	§ ó	Ñ õ	] ÷	ö	 ù	ø	 û	ô	 ı	ú	 ş	û €
ÿ	 ‚
ü	 ƒ
±	 …

 †
–	 ‡
„
 ‰
µ Š
Ø Œ
i 

 

 ’
‹
 ”
‘
 •
Š —
–
 ™
“
 š
±	 œ
˜
 
©	 
›
  
Ã ¡
 £
Ñ ¥
¢
 ¦
Ó ¨
‘ ©
Ø «
¼ ¬
º	 ®
¸ ¯
± ±
°
 ³
· ´
¾ ¶
µ
 ¸
± ¹
Ç »
º
 ½
Ë ¾
è À
Ö Á
Æ	 Ã
Ğ Ä
È	 Æ
Ã Ç
Ñ	 É
E Ê
ê Ì
Ë
 Î
ğ Ï
õ Ñ
Ğ
 Ó
ê Ô
ü Ö
Õ
 Ø
‚ Ù
û Û
 Ü
İ	 Ş
‡ ß
ß	 á
Ê â
è	 ä
Q å
¡ ç
æ
 é
§ ê
¬ ì
ë
 î
¡ ï
³ ñ
ğ
 ó
¹ ô
À ö
õ
 ø
Æ ù
ô	 û
¾ ü
ö	 ş
Ñ ÿ
ÿ	 ] ‚Ø „ƒ †Ş ‡ã ‰ˆ ‹Ø Œê  ğ ‘÷ “’ •ı –‹
 ˜õ ™
 ›Ø œ–
 i Ÿ‘ ¡  £— ¤œ ¦¥ ¨‘ ©£ «ª ­© ®& °+ ±/ ²¯ ´³ ¶µ ¸Ğ ¹& »+ ¼/ ½º ¿¾ ÁÀ Ãß Ä& Æ+ Ç/ ÈÅ ÊÉ ÌË Îî Ï& Ñ+ Ò/ ÓĞ ÕÔ ×Ö Ùı Ú& Ü+ İ/ ŞÛ àß âá äŒ å™ çÑ	 éæ ê( ì+ í/ îë ğï òÑ	 ó' õ+ ö/ ÷ô ù–
 ûø üú şÑ	 €ñ ı ‚ÿ „¼ …ñ ‡è	 ˆ† ŠÆ ‹ñ ÿ	 Œ Ğ ‘ø “’ •–
 —” ˜ñ š– ›™ ß ï  è	 ¡ï £ÿ	 ¤ï ¦–
 §ì ©¨ «È	 ¬¨ ®ß	 ¯¨ ±ö	 ²¨ ´
 µñ ·ª ¸¶ º¹ ¼û ½Ÿ ¿­ À¾ ÂÁ Ä… Å¢ Ç° ÈÆ ÊÉ Ì ÍŸ ÏŸ Ğñ Òñ ÓÎ Ô¢ Ö¢ ×Ñ Ø­ Ú­ Ûª İª ŞÙ ß° á° âÜ ãÕ åà æª èª éç ëñ íñ îê ïì ñä óğ ô¥ ö³ ÷õ ùò úø üÁ ıß ÿ+ €/ ş ƒµ …Ñ	 ‡„ ˆ† Š‚ ‹ß + / Œ ‘î “ÿ •’ –” ˜ ™ß ›+ œ/ š Ÿ¥ ¡† £  ¤¢ ¦ §ß ©+ ª/ «¨ ­Ü ¯Œ ±® ²° ´¬ µß ·+ ¸/ ¹¶ »• ½™ ¿¼ À¾ Âº Ã‘ Å¼ ÇÆ ÉÄ Êà ÌÈ ÎË ÏÍ Ñ‰ Ò€ Ô¹ ÖÓ ×Õ Ù— ÚĞ ÜÈ	 ŞÛ ßÑ	 áİ âà äØ å· çÁ éæ êè ì¥ í‡ ïß	 ñî òè	 ôğ õó ÷ë øî úÉ üù ıû ÿ³ €¾ ‚ö	 „ …ÿ	 ‡ƒ ˆ† Šş ‹§ ø Œ  ’Á “õ •
 —” ˜–
 š– ›™ ‘ Æ  Ä ¢Ÿ £Ë ¥¡ ¦¨ ¨§ ª¤ «±	 ­© ®Ğ ¯¬ ±ş ²Ã ´³ ¶Û ¸µ ¹E »º ½· ¾İ À¿ Â¼ Ã±	 ÅÁ Æã ÇÄ ÉŒ ÊÊ ÌË Îî ĞÍ ÑQ ÓÒ ÕÏ Öì Ø× ÚÔ Û±	 İÙ Şö ßÜ áš âÑ äã æ èå é] ëê íç îû ğï òì ó±	 õñ ö‰ ÷ô ù¨ úØ üû ş” €ı i ƒ‚ …ÿ †Š ˆ‡ Š„ ‹±	 ‰ œ Œ ‘¶ ’§ –Û ˜³ š¿ œî Ë   ¢ã ¤” ¦û ¨” ª ¬Ç ®™ °“ ²± ´ê ¶¡ ¸Ø º‘ ¼» ¿Ô À¹ ÂÄ Ã· Å´ Æµ È¤ É³ Ë– Ì‡ Î¿ Ï‚ ÑÍ Ò§ Ôà Õ¥ ×Ş Øï Ú¨ Ûê İÙ Ş£ àÜ á¡ ãÚ ä× æ‘ çÒ éå êŸ ìØ í ïÖ ğ¿ òú ó› õÔ öº øõ ù™ ûÒ ü— şĞ ÿ• Î ‚Ë „„ …Æ ‡ƒ ˆÄ Š† ‹ Œ € ‘º ’Ê ”· •¾ —ı ™Ø šú œÒ ô ŸG  Ç ¢ğ £õ ¥ü §¦ ©‚ ªî ¬ ­ë ¯‰ °Ä ²§ ³¬ µ³ ·¶ ¹¹ ºâ ¼Æ ½ß ¿À ÀÁ ÂŞ Ãã Åê ÇÆ Éğ ÊÖ Ìı ÍÓ Ï÷ Ğ¾ Ò— Óœ Õ£ ×Ö Ù© ÚŒ Ü& Ş+ ß/ àÛ áİ ãâ åä çĞ è& ê+ ë/ ìÛ íé ïî ñğ óß ô& ö+ ÷/ øÛ ùõ ûú ıü ÿî €& ‚+ ƒ/ „Û … ‡† ‰ˆ ‹ı Œ& + / Û ‘ “’ •” —Œ ˜ñ š¯ ›( + / Ÿ  œ ¢¡ ¤ñ ¥' §+ ¨/ © ª¦ ¬Í ®« ¯­ ±ñ ³£ ´° µ² ·¼ ¸£ ºå »¹ ½Æ ¾£ ÀÙ Á¿ ÃĞ Ä« ÆÅ ÈÍ ÊÇ Ë£ ÍÉ ÎÌ Ğß Ñ¡ Óå Ô¡ ÖÙ ×¡ ÙÍ Ú( Ü+ İ/ ŞŒ ßÛ áà ã÷ äà æè çà éÜ êà ìĞ í£ ïâ ğî òñ ôû õÒ ÷å øö úù ü… ıÕ ÿè €ş ‚ „ …Ò ‡Ò ˆ£ Š£ ‹† ŒÕ Õ ‰ å ’å “â •â –‘ —è ™è š” › ˜ â  â ¡Ÿ ££ ¥£ ¦¢ §¤ ©œ «¨ ¬Ø ®ë ¯­ ±ª ²° ´Á µß ·+ ¸/ ¹Œ º¶ ¼µ ¾ñ À½ Á¿ Ã» Äß Æ+ Ç/ ÈŒ ÉÅ Ëî Í² ÏÌ ĞÎ ÒÊ Óß Õ+ Ö/ ×Œ ØÔ Ú¥ Ü¹ ŞÛ ßİ áÙ âß ä+ å/ æŒ çã éÜ ë¿ íê îì ğè ñß ó+ ô/ õŒ öò ø• úÌ üù ıû ÿ÷ €ƒ ‚† ƒà … ‡„ ˆ† ŠÂ ‹€ ñ Œ  ’Ñ “Ğ •÷ —” ˜ñ š– ›™ ‘ ·  ù ¢Ÿ £¡ ¥à ¦‡ ¨è ª§ «å ­© ®¬ °¤ ±î ³ µ² ¶´ ¸ï ¹¾ »Ü ½º ¾Ù À¼ Á¿ Ã· Ä§ Æ° ÈÅ ÉÇ Ëş Ìõ ÎĞ ĞÍ ÑÍ ÓÏ ÔÒ ÖÊ ×† Ù‰ Úƒ ÜØ İ„ ßÛ à¨ âŞ äá å±	 çã è‰ éæ ë¶ ìÖ î” ğí ñ÷ óï ôE öõ øò ùİ û÷ ıú ş±	 €ü œ ‚ÿ „Å … ‡§ ‰† Šè Œˆ å ‹ ì ’ ”‘ •±	 —“ ˜¯ ™– ›Ô œÄ º   ¡Ü £Ÿ ¤Ù ¦¢ §û ©¥ «¨ ¬±	 ®ª ¯Â °­ ²ã ³û µÍ ·´ ¸Ğ º¶ »Í ½¹ ¾Š À¼ Â¿ Ã±	 ÅÁ ÆÕ ÇÄ Éò Ê Ì± Íá Ï” Ñ÷ Óú Õ§ ×è Ùº ÛÜ İÍ ßĞ áË ã‰ å« æ† è‘ éƒ ë¼ ì– î± ï­ ñË ò÷ ôÃ õ¤ ÷ê øè úÊ ûå ıQ ş´ €¡ Ü ƒÑ „Ù †] ‡Ä ‰Ø ŠĞ ŒØ Í i Ô ’‘ “Ò •¿ ˜‡ ™Í ›‚ œà § ŸŞ ¡¥ ¢¨ ¤ï ¥Ù §ê ¨Ü ª£ «Ú ­¡ ®‘ °× ±å ³Ò ´Ø ¶Ÿ ·Ö ¹ ºú ¼¿ ½Ô ¿› Àõ Âº ÃÒ Å™ Æ” È— ÉÎ Ë• Ì“ ÏÎ Ñ™ Ò¾ ÔÓ Ö“ ×º ÙØ Û¾ ÜÊ Şº ß± áà ã· ä¾ æå è± éÇ ëê íË îÇ ğØ ñÄ óÒ ôÁ öÃ ÷¾ ùG úê üû şğ ÿõ € ƒê „ü †… ˆ‚ ‰¸ ‹ Œµ ‰ ² ‘Ê ’¯ ”Q •¡ —– ™§ š¬ œ› ¡ Ÿ³ ¡  £¹ ¤¬ ¦Æ §© ©À ª¦ ¬Ñ ­£ ¯] °Ø ²± ´Ş µã ·¶ ¹Ø ºê ¼» ¾ğ ¿  Áı Â Ä÷ Åš ÇØ È— Êi Ë‘ ÍÌ Ï— Ğœ ÒÑ Ô‘ Õ£ ×Ö Ù© ÚÛ İ& ß+ à/ áÜ âŞ äã æå èĞ é& ë+ ì/ íÜ îê ğï òñ ôß õ& ÷+ ø/ ùÜ úö üû şı €î & ƒ+ „/ …Ü †‚ ˆ‡ Š‰ Œı & + / ‘Ü ’ ”“ –• ˜Œ ™™ ›» š Í  ( ¢+ £/ ¤Ÿ ¥¡ §¦ ©» ª' ¬+ ­/ ®Ÿ ¯« ±— ³° ´² ¶» ¸¨ ¹µ º· ¼¼ ½¨ ¿¯ À¾ ÂÆ Ã¨ Å£ ÆÄ ÈĞ É° ËÊ Í— ÏÌ Ğ¨ ÒÎ ÓÑ Õß Ö¦ Ø¯ Ù¦ Û£ Ü¦ Ş— ß“ á( ã+ ä/ åà æâ èç êÁ ëç í² îç ğ¦ ñç óš ô¨ öé ÷õ ùø ûû ü× şì ÿı € ƒ… „Ú †ï ‡… ‰ˆ ‹ Œ× × ¨ ‘¨ ’ “Ú •Ú – —ì ™ì šé œé ˜ ï  ï ¡› ¢” ¤Ÿ ¥é §é ¨¦ ª¨ ¬¨ ­© ®« °£ ²¯ ³İ µò ¶´ ¸± ¹· »Á ¼ß ¾+ ¿/ Àà Á½ Ãµ Å» ÇÄ ÈÆ ÊÂ Ëß Í+ Î/ Ïà ĞÌ Òî Ô· ÖÓ ×Õ ÙÑ Úß Ü+ İ/ Şà ßÛ á¥ ã¾ åâ æä èà éß ë+ ì/ íà îê ğÜ òÄ ôñ õó ÷ï øß ú+ û/ üà ıù ÿ• Ñ ƒ€ „‚ †ş ‡‘ ‰¼ ‹Š ˆ à Œ ’ “‘ •É –€ ˜ø š— ›™ Ø Ğ  Á ¢Ÿ £» ¥¡ ¦¤ ¨œ ©· «€ ­ª ®¬ °ç ±‡ ³² µ² ¶¯ ¸´ ¹· »¯ ¼î ¾ˆ À½ Á¿ Ãö Ä¾ Æ¦ ÈÅ É£ ËÇ ÌÊ ÎÂ Ï§ Ñ· ÓĞ ÔÒ Ö… ×õ Ùš ÛØ Ü— ŞÚ ßİ áÕ â— äˆ æã çŠ éå ê ìè í±	 ïë ğ” ñî ó½ ôÖ öŸ øõ ùÃ ûú ı÷ şE €ÿ ‚ü ƒ±	 … †§ ‡„ ‰Ì Š Œ² ‹ Ê ‘ “ ”Q –• ˜’ ™±	 ›— œº š ŸÛ  Ä ¢Å ¤¡ ¥Ñ §¦ ©£ ª] ¬« ®¨ ¯±	 ±­ ²Í ³° µê ¶û ¸Ø º· »Ø ½¼ ¿¹ Ài ÂÁ Ä¾ Å±	 ÇÃ Èà ÉÆ Ëù Ì Îˆ ĞÍ ÑŠ Ó‘ Ô Ö¼ ×ª ÙØ Ûº Ü± Şİ à· á¾ ãâ å± æÇ èç êË ëŸ íÖ îú ğĞ ñÿ óÃ ôß öõ øG ùê ûú ığ şõ €ÿ ‚ê ƒü …„ ‡‚ ˆ² Š ‹ ‡ • Ê ‘î “’ •S –¡ ˜— š§ ›¬ œ Ÿ¡  ³ ¢¡ ¤¹ ¥Å §Ä ¨¦ ª¾ «« ­Ñ ®ı °¯ ²_ ³Ø µ´ ·Ş ¸ã º¹ ¼Ø ½ê ¿¾ Áğ ÂØ Äû Å¼ Çõ ÈÁ ÊØ ËŒ ÍÌ Ïk Ğ‘ ÒÑ Ô— Õœ ×Ö Ù‘ Ú£ ÜÛ Ş© ßõ á› âõ ä( æ+ ç/ èÜ éå ëê íã î' ğ+ ñ/ òÜ óï õÌ ÷ö ùô úø üã şì ÿû €ı ‚¼ ƒ’ …ì ‡„ ˆ† ŠÆ ‹¯ ì Œ  ’Ğ “ô •” —ö ™– šì œ˜ › Ÿß  ê ¢„ £ê ¥Œ ¦ê ¨ö ©¡ «ª ­ÿ ®ª °• ±ª ³« ´ª ¶Á ·ì ¹¬ º¸ ¼» ¾û ¿¡ Á¯ ÂÀ ÄÃ Æ… Ç¤ É² ÊÈ ÌË Î Ï¡ Ñ¡ Òì Ôì ÕĞ Ö¤ Ø¤ ÙÓ Ú¯ Ü¯ İ¬ ß¬ àÛ á² ã² äŞ å× çâ è¬ ê¬ ëé íì ïì ğì ñî óæ õò ö§ øµ ù÷ ûô üú şÁ ÿß + ‚/ ƒŸ „€ †µ ˆã Š‡ ‹‰ … ß + ‘/ ’Ÿ “ •î —ı ™– š˜ œ” ß Ÿ+  / ¡Ÿ ¢ ¤¥ ¦† ¨¥ ©§ «£ ¬ß ®+ ¯/ °Ÿ ±­ ³Ü µ ·´ ¸¶ º² »ß ½+ ¾/ ¿Ÿ À¼ ÂÑ Ä› ÆÃ ÇÅ ÉÁ Ê‘ Ì¼ ÎÍ ĞË Ñà ÓÏ ÕÒ ÖÔ ØŒ Ù€ Û» İÚ ŞÜ à› áĞ ãÿ åâ æä èã éç ëß ì· îÃ ğí ñï óª ô‡ ö• øõ ù÷ û„ üú şò ÿî Ë ƒ€ „‚ †¹ ‡¾ ‰« ‹ˆ ŒŠ Œ  ‘… ’§ ”ú –“ —• ™È šõ œÁ › Ÿ ¡ö ¢  ¤˜ ¥— §Ë ©¦ ªÍ ¬¨ ­±	 ¯« °× ±® ³€ ´Ö ¶â ¸µ ¹Ã »º ½· ¾±	 À¼ Áê Â¿ Ä Å Çõ ÉÆ ÊÊ ÌË ÎÈ Ï±	 ÑÍ Òı ÓĞ Õ ÖÄ Øˆ Ú× ÛÑ İÜ ßÙ à±	 âŞ ã äá æ­ çû é› ëè ìØ îí ğê ñ±	 óï ô£ õò ÷¼ ø û ı ÿ  ú$ &$ ú© «© —½ ¾ù úâ äâ ¾– — ‹‹ ŒŒ ˆˆ ŠŠ ‰‰ €á ŠŠ á÷ ŠŠ ÷ä ŠŠ ä£ ŠŠ £	 ŠŠ 	– ŠŠ –ü ŒŒ üÄ ŠŠ Äã ŠŠ ãŒ ŠŠ Œÿ ŠŠ ÿ¤ ŠŠ ¤² ŠŠ ²¯ ŠŠ ¯¾ ŠŠ ¾… ŠŠ …œ ŠŠ œŞ ŠŠ Şé ŠŠ éØ ŠŠ Ø¬	 ‹‹ ¬	 ŠŠ Õ ŠŠ Õ“
 ŠŠ “
›
 ŠŠ ›
Â ŠŠ Â¬ ŠŠ ¬´ ŠŠ ´Ñ ŠŠ ÑÓ ŠŠ Ó° ŠŠ °˜ ŠŠ ˜ê ŠŠ êà ŠŠ àÜ ŠŠ Ü– ŠŠ –… ŠŠ …— ŠŠ —– ŠŠ –÷ ŠŠ ÷« ŠŠ «î ŠŠ î‰ ŠŠ ‰­ ŠŠ ­	 ŠŠ 	Ù ŠŠ Ù· ŠŠ ·¹ ŠŠ ¹ò ŠŠ òÛ ŠŠ ÛÁ ŠŠ Áï ŠŠ ï¿ ŠŠ ¿… ŠŠ …ª ŠŠ ªò ŠŠ òú ŠŠ úÿ ŠŠ ÿÕ ŠŠ ÕÑ ŠŠ Ñ© ŠŠ ©« ŠŠ «È ŠŠ È” ŠŠ ”ƒ	 ŠŠ ƒ	ü ŠŠ üŒ ŠŠ Œâ ŠŠ â ŠŠ œ ŠŠ œÍ ŠŠ ÍŸ ŠŠ Ÿ¯ ŠŠ ¯› ŠŠ ›Ä ŠŠ Ä ‰‰ ­ ŠŠ ­í	 ŠŠ í	ê ŠŠ êş ŠŠ şƒ ŠŠ ƒç ŠŠ ç¾ ŠŠ ¾– ŠŠ –’ ŠŠ ’Ã ŠŠ Ã ŠŠ · ŠŠ ·¥ ŠŠ ¥£ ŠŠ £ˆ ŠŠ ˆğ ŠŠ ğ± ŠŠ ±Â ŠŠ Â² ŠŠ ²¤ ŠŠ ¤— ŠŠ —‹ ŠŠ ‹Ï ŠŠ Ï˜ ŠŠ ˜Õ ŠŠ ÕŞ ŠŠ Ş¦ ŠŠ ¦î ŠŠ î÷ ŠŠ ÷à ŠŠ à¿	 ŠŠ ¿	Ï ŠŠ Ï„ ŠŠ „ì ŠŠ ìë ŠŠ ë¶ ŠŠ ¶É ŠŠ Éš ŠŠ š‘ ŠŠ ‘å ŠŠ åŠ ŠŠ ŠÎ	 ŠŠ Î	å ŠŠ å” ŠŠ ”ğ ŠŠ ğí ŠŠ í¸ ŠŠ ¸ô ŠŠ ôÉ ŠŠ Éü	 ŠŠ ü	Â ŠŠ ÂÖ	 ŠŠ Ö	Ø ŠŠ ØÜ ŠŠ Ü› ŠŠ ›İ ŠŠ İ ŠŠ  ŠŠ ı ŠŠ ıï ŠŠ ï° ŠŠ °Î ŠŠ Î¤ ŠŠ ¤¹ ŠŠ ¹¨ ŠŠ ¨
 ˆˆ 
ø ŠŠ øİ ŠŠ İò ŠŠ ò ‰‰ ¸ ŠŠ ¸è ŠŠ è® ŠŠ ®à ŠŠ àş ŒŒ şï ŠŠ ï ŠŠ ß ŠŠ ßĞ ŠŠ Ğ·	 ŠŠ ·	 ˆˆ × ŠŠ ×À ŠŠ À‹	 ŠŠ ‹	¡ ŠŠ ¡Ç ŠŠ Ç¼ ŠŠ ¼ö ŠŠ öŞ ŠŠ Şö ŠŠ öŒ ŠŠ Œ­	 ‹‹ ­	Õ ŠŠ Õ¢ ŠŠ ¢Ï ŠŠ Ï× ŠŠ ×Í ŠŠ Í¢ ŠŠ ¢ª ŠŠ ª– ŠŠ –„
 ŠŠ „
¡ ŠŠ ¡ç ŠŠ ç„ ŠŠ „©	 ŠŠ ©	‘ ŠŠ ‘ÿ ŠŠ ÿ ˆˆ º ŠŠ ºÚ ŠŠ Ú ŠŠ ú ŒŒ úÆ ŠŠ ÆŞ ŠŠ Ş‰ ŠŠ ‰¥ ŠŠ ¥Ê ŠŠ Êÿ ŠŠ ÿ· ŠŠ ·ı ŠŠ ı‰ ŠŠ ‰— ŠŠ —Œ ŠŠ Œı ŠŠ ı˜ ŠŠ ˜Î ŠŠ ÎÈ ŠŠ È” ŠŠ ”‰ ŠŠ ‰ê ŠŠ ê¼ ŠŠ ¼œ ŠŠ œ¹ ŠŠ ¹–	 ŠŠ –	Ù ŠŠ Ùë ŠŠ ëÈ ŠŠ È¼ ŠŠ ¼ô ŠŠ ô§ ŠŠ §å	 ŠŠ å	² ŠŠ ²ø ŠŠ ø£	 ŠŠ £	¤ ŠŠ ¤ò ŠŠ òÔ ŠŠ Ô­ ŠŠ ­· ŠŠ ·‚ ŠŠ ‚æ ŠŠ æş ŠŠ ş³ ŠŠ ³· ŠŠ ·ù ŠŠ ùŸ ŠŠ Ÿ¨ ŠŠ ¨¼ ŠŠ ¼ ŠŠ Ø ŠŠ Øì ŠŠ ìâ ŠŠ â ŠŠ Ğ ŠŠ Ğ› ŠŠ ›  ú
 ¡
 ¤
 ·
 ¼
 Ï
 Ô
 ç
 ì
 ÿ
 „
 Ø
 Ş
 ï
 ÷
 ˆ
 
 Ÿ
 ¥
 ¶
 ¼
 å
 ë
 ÷
 
 
 —
 £
 ­
 ¹
 Ã
 ¨
 ·
 È
 Ù
 ê
 ¸
 ÿ
 ò
 ª
 ±
 ô    
‘ ù
‘ Æ
‘ ¹
‘ ñ
‘ ø
‘ »
’ Í“ 
“ ş
” Õ
” ê
” ı
” 	
” £	
” È
” İ
” ğ
” ƒ
” –
” 
” –
” ©
” ¼
” Ï
” Œ
” ¡
” ´
” Ç
” Ú
” Ï
” ä
” ÷
” Š
” 
• ”	– 	– 	– =	– E	– m
– 
– 
– š
– ¤
– ³
– Ã
– â
– í
– í
– ø
– ƒ
– 
– ¢
– «
– ¼
– û
– û
– …
– 
– Á
– Õ
– İ
– ‘
– ¯
– Å
– Ğ
– Ğ
– Ö
– î
– €
– ‡
– ¾
– õ
– ¹
– ‹
– ™
– ™
– §
– µ
– Ã
– º
– Œ
– 
– é
– Å
– ê
– Ì
– 
— ­
— Ù
— 
— –
— É
— Î
— ˜	˜ 9	˜ E	˜ Q	˜ U	˜ ]	˜ ]	˜ i
˜ ¤
˜ ¸
˜ Ñ
˜ ƒ
˜ Ğ
˜ 
˜ ó
˜ û
˜ ¾
˜ Ä
˜ Ü
˜ î
˜ ®
˜ ¹
˜ Ä
˜ Ï
˜ Ï
˜ Ú
˜ µ
˜ Ğ
˜ ë
˜ ô
˜ ¨˜ Œ
˜ 
˜ ã
˜ ‚
˜ ê
˜ ­
™ ¯		š 	š  
š Û
› ¶
› ı
› ğ
› ¨
› ¯
› ò
œ ¾
œ …
œ ø
œ °
œ ·
œ ú	 9	 =	 E	 I	 Q	 U	 ]	 a	 i	 m	 v	 
 
 š
 ¤
 ³
 ¸
 ¸
 ¼
 ¼
 Ã
 Ê
 Ñ
 Ø
 à
 à
 ™
 ¼
 Æ
 Ğ
 ß
 û
 …
 
 Á
 Î
 İ
 ì
 û
 Š
 ‘
 ‘
 —
 —
 —
 ¨
 ¨
 ¯
 ¯
 µ
 µ
 µ
 ¼
 ¼
 Å
 Å
 Ğ
 Ö
 Ö
 î
 î
 €
 €
 ‡
 
 
 ¥
 ¥
 ·
 ·
 ¾
 Ä
 Ä
 Ü
 Ü
 î
 î
 õ
 û
 û
 •
 •
 §
 §
 ‹
 ¢

 ¢

 ¢

 æ
 æ
 ş
 «
 «
 «
 ¯
 ¯
 ¶
 š
 š
 ½
 Í
 Í
 Í
 €	 I	 Q
 š
 ¼
 Ã
 Ê
 Ê
 Ñ
 Ø
 ø
 ™
 ¼
 Æ
 Æ
 Ğ
 ß
 …
 Æ
 Õ
 ä
 ä
 ì
 ó
 ‚
 ¼
 ‡
 
 ¥
 ·
 Ä
 ì
 õ
 §
 Å
 ş
 Œ
 š
 š
 ¨
 ¶
 Û
 õ
 Ô
 ö
 Û
 
Ÿ İ
Ÿ ğ
Ÿ ƒ	
Ÿ –	
Ÿ ©	
Ÿ Ğ
Ÿ ã
Ÿ ö
Ÿ ‰
Ÿ œ
Ÿ ‰
Ÿ œ
Ÿ ¯
Ÿ Â
Ÿ Õ
Ÿ ”
Ÿ §
Ÿ º
Ÿ Í
Ÿ à
Ÿ ×
Ÿ ê
Ÿ ı
Ÿ 
Ÿ £    ü
¡ ƒ
¡ 
¡ Î
¡ Ö
¡ Á
¡ É
¡ ù
¡ 
¡ €
¡ ˆ
¡ Ã
¡ Ë¢ 	£ a	£ i
£ ³
£ Ø
£ 
£ ß
£ Á
£ Î
£ İ
£ ì
£ û
£ ‚
£ Š
£ Š
£ ¨
£ õ
£ û
£ •
£ §
£ Ú
£ Ã
£ ¯
£ º
£ Å
£ Ğ
£ Û
£ Û
£ ¶
£ 
£ ò
£ 
£ ù
£ ¼
¤ –
¤ ¤
¤ ²
¤ À
¤ Î
¤ ‰
¤ —
¤ ¥
¤ ³
¤ Á
¤ Â
¤ Ñ
¤ à
¤ ï
¤ ş
¤ É
¤ Ø
¤ ç
¤ ö
¤ …
¤ Œ
¤ ›
¤ ª
¤ ¹
¤ È
¥ “
¦ Š
¦ ©
¦ ¶
¦ Õ
¦ €
¦ ™
¦ ı
¦ ’
¦ °
¦ Å
¦ µ
¦ Ê
¦ û
¦ ”	§ )	§ +	§ -	§ /
¨ ­	© «© ×© °© ›© ÷© ±	© µ	© Ì	© ã	© ú	© ‘
© ”© ê© Ç© ¢© Ì© ©© –© ì
ª ³	
ª Ê	
ª á	
ª ø	
ª 

« å
« ø
« ‹	
« 	
« Ø
« ë
« ş
« ‘
« ‘
« ¤
« ·
« Ê
« œ
« ¯
« Â
« Õ
« ß
« ò
« …
« ˜¬ ¬	
¬ ¬	
­ ·	
­ Î	
­ å	
­ ü	
­ “

­ «
­ ¼
­ Í
­ Ş
­ ï
® Ÿ
® µ
® Í
® å
® ı
® Û
® ò
® ‹
® ¢
® ¹
® è
® ü
® ’
® ¨
® ¾"
rhsx"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z3maxdd"
llvm.lifetime.end.p0i8*‡
npb-LU-rhsx.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02€

devmap_label
 

transfer_bytes
°°ƒ
 
transfer_bytes_log1p
ŠzA

wgsize
>

wgsize_log1p
ŠzA