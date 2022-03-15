
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
!br i1 %20, label %21, label %1179
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
!br i1 %24, label %25, label %1179
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
q%33 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 0, i64 %32
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
x%38 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 0, i64 %32, i64 1
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
x%43 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 0, i64 %32, i64 2
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
x%48 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 0, i64 %32, i64 3
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
x%53 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 0, i64 %32, i64 4
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
*store i64 %45, i64* %59, align 8, !tbaa !8
%i648B

	full_text
	
i64 %45
'i64*8B

	full_text


i64* %59
?bitcast8B2
0
	full_text#
!
%60 = bitcast i64 %45 to double
%i648B

	full_text
	
i64 %45
ãgetelementptr8Bx
v
	full_texti
g
e%61 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 0, i64 %32
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
e%64 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 0, i64 %32
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
%69 = bitcast i64 %55 to double
%i648B

	full_text
	
i64 %55
7fsub8B-
+
	full_text

%70 = fsub double %69, %65
+double8B

	full_text


double %69
+double8B

	full_text


double %65
@fmul8B6
4
	full_text'
%
#%71 = fmul double %70, 4.000000e-01
+double8B

	full_text


double %70
icall8B_
]
	full_textP
N
L%72 = tail call double @llvm.fmuladd.f64(double %60, double %63, double %71)
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


double %71
Çgetelementptr8Bo
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
U%79 = tail call double @llvm.fmuladd.f64(double %69, double 1.400000e+00, double %78)
+double8B

	full_text


double %69
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
ôgetelementptr8BÖ
Ç
	full_textu
s
q%96 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 1, i64 %32
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
†getelementptr8Bå
â
	full_text|
z
x%99 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 1, i64 %32, i64 1
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
°getelementptr8Bç
ä
	full_text}
{
y%102 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 1, i64 %32, i64 2
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
°getelementptr8Bç
ä
	full_text}
{
y%105 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 1, i64 %32, i64 3
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
°getelementptr8Bç
ä
	full_text}
{
y%108 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 1, i64 %32, i64 4
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
-store i64 %104, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %104
(i64*8B

	full_text

	i64* %112
Abitcast8B4
2
	full_text%
#
!%113 = bitcast i64 %104 to double
&i648B

	full_text


i64 %104
ågetelementptr8By
w
	full_textj
h
f%114 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 1, i64 %32
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
ågetelementptr8By
w
	full_textj
h
f%117 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 1, i64 %32
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
!%119 = bitcast i64 %101 to double
&i648B

	full_text


i64 %101
:fmul8B0
.
	full_text!

%120 = fmul double %116, %119
,double8B

	full_text

double %116
,double8B

	full_text

double %119
Égetelementptr8Bp
n
	full_texta
_
]%121 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Pstore8BE
C
	full_text6
4
2store double %120, double* %121, align 8, !tbaa !8
,double8B

	full_text

double %120
.double*8B

	full_text

double* %121
Abitcast8B4
2
	full_text%
#
!%122 = bitcast i64 %110 to double
&i648B

	full_text


i64 %110
:fsub8B0
.
	full_text!

%123 = fsub double %122, %118
,double8B

	full_text

double %122
,double8B

	full_text

double %118
Bfmul8B8
6
	full_text)
'
%%124 = fmul double %123, 4.000000e-01
,double8B

	full_text

double %123
mcall8Bc
a
	full_textT
R
P%125 = tail call double @llvm.fmuladd.f64(double %113, double %116, double %124)
,double8B

	full_text

double %113
,double8B

	full_text

double %116
,double8B

	full_text

double %124
Égetelementptr8Bp
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
Égetelementptr8Bp
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
X%132 = tail call double @llvm.fmuladd.f64(double %122, double 1.400000e+00, double %131)
,double8B

	full_text

double %122
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
Égetelementptr8Bp
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

%135 = fmul double %115, %119
,double8B

	full_text

double %115
,double8B

	full_text

double %119
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

%137 = fmul double %115, %122
,double8B

	full_text

double %115
,double8B

	full_text

double %122
8fmul8B.
,
	full_text

%138 = fmul double %62, %66
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

%140 = fmul double %62, %69
+double8B

	full_text


double %62
+double8B

	full_text


double %69
:fsub8B0
.
	full_text!

%141 = fsub double %135, %138
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
%%142 = fmul double %141, 6.300000e+01
,double8B

	full_text

double %141
Ñgetelementptr8Bq
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
9fsub8B/
-
	full_text 

%144 = fsub double %116, %63
,double8B

	full_text

double %116
+double8B

	full_text


double %63
Bfmul8B8
6
	full_text)
'
%%145 = fmul double %144, 8.400000e+01
,double8B

	full_text

double %144
Ñgetelementptr8Bq
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
%%148 = fmul double %147, 6.300000e+01
,double8B

	full_text

double %147
Ñgetelementptr8Bq
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

%150 = fmul double %116, %116
,double8B

	full_text

double %116
,double8B

	full_text

double %116
mcall8Bc
a
	full_textT
R
P%151 = tail call double @llvm.fmuladd.f64(double %135, double %135, double %150)
,double8B

	full_text

double %135
,double8B

	full_text

double %135
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
8fmul8B.
,
	full_text

%153 = fmul double %63, %63
+double8B

	full_text


double %63
+double8B

	full_text


double %63
mcall8Bc
a
	full_textT
R
P%154 = tail call double @llvm.fmuladd.f64(double %138, double %138, double %153)
,double8B

	full_text

double %138
,double8B

	full_text

double %138
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
Cfsub8B9
7
	full_text*
(
&%157 = fsub double -0.000000e+00, %153
,double8B

	full_text

double %153
mcall8Bc
a
	full_textT
R
P%158 = tail call double @llvm.fmuladd.f64(double %116, double %116, double %157)
,double8B

	full_text

double %116
,double8B

	full_text

double %116
,double8B

	full_text

double %157
Bfmul8B8
6
	full_text)
'
%%159 = fmul double %158, 1.050000e+01
,double8B

	full_text

double %158
{call8Bq
o
	full_textb
`
^%160 = tail call double @llvm.fmuladd.f64(double %156, double 0xC03E3D70A3D70A3B, double %159)
,double8B

	full_text

double %156
,double8B

	full_text

double %159
:fsub8B0
.
	full_text!

%161 = fsub double %137, %140
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
^%162 = tail call double @llvm.fmuladd.f64(double %161, double 0x405EDEB851EB851E, double %160)
,double8B

	full_text

double %161
,double8B

	full_text

double %160
Ñgetelementptr8Bq
o
	full_textb
`
^%163 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 4
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Pstore8BE
C
	full_text6
4
2store double %162, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %162
.double*8B

	full_text

double* %163
ögetelementptr8BÜ
É
	full_textv
t
r%164 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 2, i64 %32
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
Ibitcast8B<
:
	full_text-
+
)%165 = bitcast [5 x double]* %164 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %164
Jload8B@
>
	full_text1
/
-%166 = load i64, i64* %165, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %165
}getelementptr8Bj
h
	full_text[
Y
W%167 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ibitcast8B<
:
	full_text-
+
)%168 = bitcast [5 x double]* %167 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %167
Kstore8B@
>
	full_text1
/
-store i64 %166, i64* %168, align 16, !tbaa !8
&i648B

	full_text


i64 %166
(i64*8B

	full_text

	i64* %168
°getelementptr8Bç
ä
	full_text}
{
y%169 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 2, i64 %32, i64 1
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
Cbitcast8B6
4
	full_text'
%
#%170 = bitcast double* %169 to i64*
.double*8B

	full_text

double* %169
Jload8B@
>
	full_text1
/
-%171 = load i64, i64* %170, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %170
Ñgetelementptr8Bq
o
	full_textb
`
^%172 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%173 = bitcast double* %172 to i64*
.double*8B

	full_text

double* %172
Jstore8B?
=
	full_text0
.
,store i64 %171, i64* %173, align 8, !tbaa !8
&i648B

	full_text


i64 %171
(i64*8B

	full_text

	i64* %173
°getelementptr8Bç
ä
	full_text}
{
y%174 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 2, i64 %32, i64 2
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
Cbitcast8B6
4
	full_text'
%
#%175 = bitcast double* %174 to i64*
.double*8B

	full_text

double* %174
Jload8B@
>
	full_text1
/
-%176 = load i64, i64* %175, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %175
Ñgetelementptr8Bq
o
	full_textb
`
^%177 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%178 = bitcast double* %177 to i64*
.double*8B

	full_text

double* %177
Kstore8B@
>
	full_text1
/
-store i64 %176, i64* %178, align 16, !tbaa !8
&i648B

	full_text


i64 %176
(i64*8B

	full_text

	i64* %178
°getelementptr8Bç
ä
	full_text}
{
y%179 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 2, i64 %32, i64 3
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
Cbitcast8B6
4
	full_text'
%
#%180 = bitcast double* %179 to i64*
.double*8B

	full_text

double* %179
Jload8B@
>
	full_text1
/
-%181 = load i64, i64* %180, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %180
Ñgetelementptr8Bq
o
	full_textb
`
^%182 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%183 = bitcast double* %182 to i64*
.double*8B

	full_text

double* %182
Jstore8B?
=
	full_text0
.
,store i64 %181, i64* %183, align 8, !tbaa !8
&i648B

	full_text


i64 %181
(i64*8B

	full_text

	i64* %183
°getelementptr8Bç
ä
	full_text}
{
y%184 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 2, i64 %32, i64 4
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
Cbitcast8B6
4
	full_text'
%
#%185 = bitcast double* %184 to i64*
.double*8B

	full_text

double* %184
Jload8B@
>
	full_text1
/
-%186 = load i64, i64* %185, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %185
Ñgetelementptr8Bq
o
	full_textb
`
^%187 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%188 = bitcast double* %187 to i64*
.double*8B

	full_text

double* %187
Kstore8B@
>
	full_text1
/
-store i64 %186, i64* %188, align 16, !tbaa !8
&i648B

	full_text


i64 %186
(i64*8B

	full_text

	i64* %188
Ñgetelementptr8Bq
o
	full_textb
`
^%189 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%190 = bitcast double* %189 to i64*
.double*8B

	full_text

double* %189
Jload8B@
>
	full_text1
/
-%191 = load i64, i64* %190, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %190
Ñgetelementptr8Bq
o
	full_textb
`
^%192 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Nbitcast8BA
?
	full_text2
0
.%193 = bitcast [5 x [5 x double]]* %12 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Kstore8B@
>
	full_text1
/
-store i64 %191, i64* %193, align 16, !tbaa !8
&i648B

	full_text


i64 %191
(i64*8B

	full_text

	i64* %193
Jload8B@
>
	full_text1
/
-%194 = load i64, i64* %85, align 16, !tbaa !8
'i64*8B

	full_text


i64* %85
Jstore8B?
=
	full_text0
.
,store i64 %194, i64* %190, align 8, !tbaa !8
&i648B

	full_text


i64 %194
(i64*8B

	full_text

	i64* %190
Iload8B?
=
	full_text0
.
,%195 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
Jstore8B?
=
	full_text0
.
,store i64 %195, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %195
'i64*8B

	full_text


i64* %85
Ñgetelementptr8Bq
o
	full_textb
`
^%196 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 0
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
Istore8B>
<
	full_text/
-
+store i64 %166, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %166
'i64*8B

	full_text


i64* %83
Égetelementptr8Bp
n
	full_texta
_
]%198 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 1, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
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
Égetelementptr8Bp
n
	full_texta
_
]%201 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Mbitcast8B@
>
	full_text1
/
-%202 = bitcast [3 x [5 x double]]* %8 to i64*
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
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
Égetelementptr8Bp
n
	full_texta
_
]%203 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 2, i64 0
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%204 = bitcast double* %203 to i64*
.double*8B

	full_text

double* %203
Kload8BA
?
	full_text2
0
.%205 = load i64, i64* %204, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %204
Jstore8B?
=
	full_text0
.
,store i64 %205, i64* %199, align 8, !tbaa !8
&i648B

	full_text


i64 %205
(i64*8B

	full_text

	i64* %199
Ñgetelementptr8Bq
o
	full_textb
`
^%206 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 1, i64 0
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%207 = bitcast double* %206 to i64*
.double*8B

	full_text

double* %206
Jload8B@
>
	full_text1
/
-%208 = load i64, i64* %207, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %207
Nbitcast8BA
?
	full_text2
0
.%209 = bitcast [2 x [5 x double]]* %10 to i64*
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Kstore8B@
>
	full_text1
/
-store i64 %208, i64* %209, align 16, !tbaa !8
&i648B

	full_text


i64 %208
(i64*8B

	full_text

	i64* %209
Ñgetelementptr8Bq
o
	full_textb
`
^%210 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%211 = bitcast double* %210 to i64*
.double*8B

	full_text

double* %210
Jload8B@
>
	full_text1
/
-%212 = load i64, i64* %211, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %211
Ñgetelementptr8Bq
o
	full_textb
`
^%213 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%214 = bitcast double* %213 to i64*
.double*8B

	full_text

double* %213
Jstore8B?
=
	full_text0
.
,store i64 %212, i64* %214, align 8, !tbaa !8
&i648B

	full_text


i64 %212
(i64*8B

	full_text

	i64* %214
Iload8B?
=
	full_text0
.
,%215 = load i64, i64* %87, align 8, !tbaa !8
'i64*8B

	full_text


i64* %87
Jstore8B?
=
	full_text0
.
,store i64 %215, i64* %211, align 8, !tbaa !8
&i648B

	full_text


i64 %215
(i64*8B

	full_text

	i64* %211
Iload8B?
=
	full_text0
.
,%216 = load i64, i64* %42, align 8, !tbaa !8
'i64*8B

	full_text


i64* %42
Istore8B>
<
	full_text/
-
+store i64 %216, i64* %87, align 8, !tbaa !8
&i648B

	full_text


i64 %216
'i64*8B

	full_text


i64* %87
Istore8B>
<
	full_text/
-
+store i64 %171, i64* %42, align 8, !tbaa !8
&i648B

	full_text


i64 %171
'i64*8B

	full_text


i64* %42
Bbitcast8B5
3
	full_text&
$
"%217 = bitcast double* %68 to i64*
-double*8B

	full_text

double* %68
Jload8B@
>
	full_text1
/
-%218 = load i64, i64* %217, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %217
Égetelementptr8Bp
n
	full_texta
_
]%219 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 1
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%220 = bitcast double* %219 to i64*
.double*8B

	full_text

double* %219
Jstore8B?
=
	full_text0
.
,store i64 %218, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %218
(i64*8B

	full_text

	i64* %220
Cbitcast8B6
4
	full_text'
%
#%221 = bitcast double* %121 to i64*
.double*8B

	full_text

double* %121
Jload8B@
>
	full_text1
/
-%222 = load i64, i64* %221, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %221
Jstore8B?
=
	full_text0
.
,store i64 %222, i64* %217, align 8, !tbaa !8
&i648B

	full_text


i64 %222
(i64*8B

	full_text

	i64* %217
Cbitcast8B6
4
	full_text'
%
#%223 = bitcast double* %143 to i64*
.double*8B

	full_text

double* %143
Jload8B@
>
	full_text1
/
-%224 = load i64, i64* %223, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %223
Ñgetelementptr8Bq
o
	full_textb
`
^%225 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 1
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%226 = bitcast double* %225 to i64*
.double*8B

	full_text

double* %225
Jstore8B?
=
	full_text0
.
,store i64 %224, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %224
(i64*8B

	full_text

	i64* %226
Ñgetelementptr8Bq
o
	full_textb
`
^%227 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%228 = bitcast double* %227 to i64*
.double*8B

	full_text

double* %227
Jload8B@
>
	full_text1
/
-%229 = load i64, i64* %228, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %228
Ñgetelementptr8Bq
o
	full_textb
`
^%230 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%231 = bitcast double* %230 to i64*
.double*8B

	full_text

double* %230
Kstore8B@
>
	full_text1
/
-store i64 %229, i64* %231, align 16, !tbaa !8
&i648B

	full_text


i64 %229
(i64*8B

	full_text

	i64* %231
Jload8B@
>
	full_text1
/
-%232 = load i64, i64* %89, align 16, !tbaa !8
'i64*8B

	full_text


i64* %89
Jstore8B?
=
	full_text0
.
,store i64 %232, i64* %228, align 8, !tbaa !8
&i648B

	full_text


i64 %232
(i64*8B

	full_text

	i64* %228
Iload8B?
=
	full_text0
.
,%233 = load i64, i64* %47, align 8, !tbaa !8
'i64*8B

	full_text


i64* %47
Jstore8B?
=
	full_text0
.
,store i64 %233, i64* %89, align 16, !tbaa !8
&i648B

	full_text


i64 %233
'i64*8B

	full_text


i64* %89
Istore8B>
<
	full_text/
-
+store i64 %176, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %176
'i64*8B

	full_text


i64* %47
Bbitcast8B5
3
	full_text&
$
"%234 = bitcast double* %73 to i64*
-double*8B

	full_text

double* %73
Jload8B@
>
	full_text1
/
-%235 = load i64, i64* %234, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %234
Égetelementptr8Bp
n
	full_texta
_
]%236 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 2
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%237 = bitcast double* %236 to i64*
.double*8B

	full_text

double* %236
Kstore8B@
>
	full_text1
/
-store i64 %235, i64* %237, align 16, !tbaa !8
&i648B

	full_text


i64 %235
(i64*8B

	full_text

	i64* %237
Cbitcast8B6
4
	full_text'
%
#%238 = bitcast double* %126 to i64*
.double*8B

	full_text

double* %126
Kload8BA
?
	full_text2
0
.%239 = load i64, i64* %238, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %238
Jstore8B?
=
	full_text0
.
,store i64 %239, i64* %234, align 8, !tbaa !8
&i648B

	full_text


i64 %239
(i64*8B

	full_text

	i64* %234
Cbitcast8B6
4
	full_text'
%
#%240 = bitcast double* %146 to i64*
.double*8B

	full_text

double* %146
Jload8B@
>
	full_text1
/
-%241 = load i64, i64* %240, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %240
Ñgetelementptr8Bq
o
	full_textb
`
^%242 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 2
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%243 = bitcast double* %242 to i64*
.double*8B

	full_text

double* %242
Kstore8B@
>
	full_text1
/
-store i64 %241, i64* %243, align 16, !tbaa !8
&i648B

	full_text


i64 %241
(i64*8B

	full_text

	i64* %243
Ñgetelementptr8Bq
o
	full_textb
`
^%244 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%245 = bitcast double* %244 to i64*
.double*8B

	full_text

double* %244
Jload8B@
>
	full_text1
/
-%246 = load i64, i64* %245, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %245
Ñgetelementptr8Bq
o
	full_textb
`
^%247 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%248 = bitcast double* %247 to i64*
.double*8B

	full_text

double* %247
Jstore8B?
=
	full_text0
.
,store i64 %246, i64* %248, align 8, !tbaa !8
&i648B

	full_text


i64 %246
(i64*8B

	full_text

	i64* %248
Iload8B?
=
	full_text0
.
,%249 = load i64, i64* %91, align 8, !tbaa !8
'i64*8B

	full_text


i64* %91
Jstore8B?
=
	full_text0
.
,store i64 %249, i64* %245, align 8, !tbaa !8
&i648B

	full_text


i64 %249
(i64*8B

	full_text

	i64* %245
Iload8B?
=
	full_text0
.
,%250 = load i64, i64* %52, align 8, !tbaa !8
'i64*8B

	full_text


i64* %52
Istore8B>
<
	full_text/
-
+store i64 %250, i64* %91, align 8, !tbaa !8
&i648B

	full_text


i64 %250
'i64*8B

	full_text


i64* %91
Istore8B>
<
	full_text/
-
+store i64 %181, i64* %52, align 8, !tbaa !8
&i648B

	full_text


i64 %181
'i64*8B

	full_text


i64* %52
Bbitcast8B5
3
	full_text&
$
"%251 = bitcast double* %76 to i64*
-double*8B

	full_text

double* %76
Jload8B@
>
	full_text1
/
-%252 = load i64, i64* %251, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %251
Égetelementptr8Bp
n
	full_texta
_
]%253 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 3
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%254 = bitcast double* %253 to i64*
.double*8B

	full_text

double* %253
Jstore8B?
=
	full_text0
.
,store i64 %252, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %252
(i64*8B

	full_text

	i64* %254
Cbitcast8B6
4
	full_text'
%
#%255 = bitcast double* %129 to i64*
.double*8B

	full_text

double* %129
Jload8B@
>
	full_text1
/
-%256 = load i64, i64* %255, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %255
Jstore8B?
=
	full_text0
.
,store i64 %256, i64* %251, align 8, !tbaa !8
&i648B

	full_text


i64 %256
(i64*8B

	full_text

	i64* %251
Cbitcast8B6
4
	full_text'
%
#%257 = bitcast double* %149 to i64*
.double*8B

	full_text

double* %149
Jload8B@
>
	full_text1
/
-%258 = load i64, i64* %257, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %257
Ñgetelementptr8Bq
o
	full_textb
`
^%259 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 3
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%260 = bitcast double* %259 to i64*
.double*8B

	full_text

double* %259
Jstore8B?
=
	full_text0
.
,store i64 %258, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %258
(i64*8B

	full_text

	i64* %260
Ñgetelementptr8Bq
o
	full_textb
`
^%261 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%262 = bitcast double* %261 to i64*
.double*8B

	full_text

double* %261
Jload8B@
>
	full_text1
/
-%263 = load i64, i64* %262, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %262
Ñgetelementptr8Bq
o
	full_textb
`
^%264 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Cbitcast8B6
4
	full_text'
%
#%265 = bitcast double* %264 to i64*
.double*8B

	full_text

double* %264
Kstore8B@
>
	full_text1
/
-store i64 %263, i64* %265, align 16, !tbaa !8
&i648B

	full_text


i64 %263
(i64*8B

	full_text

	i64* %265
Jload8B@
>
	full_text1
/
-%266 = load i64, i64* %93, align 16, !tbaa !8
'i64*8B

	full_text


i64* %93
Jstore8B?
=
	full_text0
.
,store i64 %266, i64* %262, align 8, !tbaa !8
&i648B

	full_text


i64 %266
(i64*8B

	full_text

	i64* %262
Iload8B?
=
	full_text0
.
,%267 = load i64, i64* %57, align 8, !tbaa !8
'i64*8B

	full_text


i64* %57
Jstore8B?
=
	full_text0
.
,store i64 %267, i64* %93, align 16, !tbaa !8
&i648B

	full_text


i64 %267
'i64*8B

	full_text


i64* %93
Kload8BA
?
	full_text2
0
.%268 = load i64, i64* %188, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %188
Istore8B>
<
	full_text/
-
+store i64 %268, i64* %57, align 8, !tbaa !8
&i648B

	full_text


i64 %268
'i64*8B

	full_text


i64* %57
Bbitcast8B5
3
	full_text&
$
"%269 = bitcast double* %81 to i64*
-double*8B

	full_text

double* %81
Jload8B@
>
	full_text1
/
-%270 = load i64, i64* %269, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %269
Égetelementptr8Bp
n
	full_texta
_
]%271 = getelementptr inbounds [3 x [5 x double]], [3 x [5 x double]]* %8, i64 0, i64 0, i64 4
D[3 x [5 x double]]*8B)
'
	full_text

[3 x [5 x double]]* %8
Cbitcast8B6
4
	full_text'
%
#%272 = bitcast double* %271 to i64*
.double*8B

	full_text

double* %271
Kstore8B@
>
	full_text1
/
-store i64 %270, i64* %272, align 16, !tbaa !8
&i648B

	full_text


i64 %270
(i64*8B

	full_text

	i64* %272
Cbitcast8B6
4
	full_text'
%
#%273 = bitcast double* %134 to i64*
.double*8B

	full_text

double* %134
Kload8BA
?
	full_text2
0
.%274 = load i64, i64* %273, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %273
Jstore8B?
=
	full_text0
.
,store i64 %274, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %274
(i64*8B

	full_text

	i64* %269
Cbitcast8B6
4
	full_text'
%
#%275 = bitcast double* %163 to i64*
.double*8B

	full_text

double* %163
Jload8B@
>
	full_text1
/
-%276 = load i64, i64* %275, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %275
Ñgetelementptr8Bq
o
	full_textb
`
^%277 = getelementptr inbounds [2 x [5 x double]], [2 x [5 x double]]* %10, i64 0, i64 0, i64 4
E[2 x [5 x double]]*8B*
(
	full_text

[2 x [5 x double]]* %10
Cbitcast8B6
4
	full_text'
%
#%278 = bitcast double* %277 to i64*
.double*8B

	full_text

double* %277
Kstore8B@
>
	full_text1
/
-store i64 %276, i64* %278, align 16, !tbaa !8
&i648B

	full_text


i64 %276
(i64*8B

	full_text

	i64* %278
ögetelementptr8BÜ
É
	full_textv
t
r%279 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 3, i64 %32
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
Ibitcast8B<
:
	full_text-
+
)%280 = bitcast [5 x double]* %279 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %279
Jload8B@
>
	full_text1
/
-%281 = load i64, i64* %280, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %280
Kstore8B@
>
	full_text1
/
-store i64 %281, i64* %168, align 16, !tbaa !8
&i648B

	full_text


i64 %281
(i64*8B

	full_text

	i64* %168
°getelementptr8Bç
ä
	full_text}
{
y%282 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 3, i64 %32, i64 1
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
Cbitcast8B6
4
	full_text'
%
#%283 = bitcast double* %282 to i64*
.double*8B

	full_text

double* %282
Jload8B@
>
	full_text1
/
-%284 = load i64, i64* %283, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %283
Jstore8B?
=
	full_text0
.
,store i64 %284, i64* %173, align 8, !tbaa !8
&i648B

	full_text


i64 %284
(i64*8B

	full_text

	i64* %173
°getelementptr8Bç
ä
	full_text}
{
y%285 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 3, i64 %32, i64 2
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
Cbitcast8B6
4
	full_text'
%
#%286 = bitcast double* %285 to i64*
.double*8B

	full_text

double* %285
Jload8B@
>
	full_text1
/
-%287 = load i64, i64* %286, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %286
Kstore8B@
>
	full_text1
/
-store i64 %287, i64* %178, align 16, !tbaa !8
&i648B

	full_text


i64 %287
(i64*8B

	full_text

	i64* %178
°getelementptr8Bç
ä
	full_text}
{
y%288 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 3, i64 %32, i64 3
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
Cbitcast8B6
4
	full_text'
%
#%289 = bitcast double* %288 to i64*
.double*8B

	full_text

double* %288
Jload8B@
>
	full_text1
/
-%290 = load i64, i64* %289, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %289
Jstore8B?
=
	full_text0
.
,store i64 %290, i64* %183, align 8, !tbaa !8
&i648B

	full_text


i64 %290
(i64*8B

	full_text

	i64* %183
°getelementptr8Bç
ä
	full_text}
{
y%291 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 3, i64 %32, i64 4
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
Cbitcast8B6
4
	full_text'
%
#%292 = bitcast double* %291 to i64*
.double*8B

	full_text

double* %291
Jload8B@
>
	full_text1
/
-%293 = load i64, i64* %292, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %292
Kstore8B@
>
	full_text1
/
-store i64 %293, i64* %188, align 16, !tbaa !8
&i648B

	full_text


i64 %293
(i64*8B

	full_text

	i64* %188
Kstore8B@
>
	full_text1
/
-store i64 %176, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %176
(i64*8B

	full_text

	i64* %112
Abitcast8B4
2
	full_text%
#
!%294 = bitcast i64 %176 to double
&i648B

	full_text


i64 %176
ågetelementptr8By
w
	full_textj
h
f%295 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 2, i64 %32
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
Pload8BF
D
	full_text7
5
3%296 = load double, double* %295, align 8, !tbaa !8
.double*8B

	full_text

double* %295
:fmul8B0
.
	full_text!

%297 = fmul double %296, %294
,double8B

	full_text

double %296
,double8B

	full_text

double %294
ågetelementptr8By
w
	full_textj
h
f%298 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 2, i64 %32
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
Pload8BF
D
	full_text7
5
3%299 = load double, double* %298, align 8, !tbaa !8
.double*8B

	full_text

double* %298
Oload8BE
C
	full_text6
4
2%300 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
:fmul8B0
.
	full_text!

%301 = fmul double %297, %300
,double8B

	full_text

double %297
,double8B

	full_text

double %300
Pstore8BE
C
	full_text6
4
2store double %301, double* %121, align 8, !tbaa !8
,double8B

	full_text

double %301
.double*8B

	full_text

double* %121
Abitcast8B4
2
	full_text%
#
!%302 = bitcast i64 %268 to double
&i648B

	full_text


i64 %268
:fsub8B0
.
	full_text!

%303 = fsub double %302, %299
,double8B

	full_text

double %302
,double8B

	full_text

double %299
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
P%305 = tail call double @llvm.fmuladd.f64(double %294, double %297, double %304)
,double8B

	full_text

double %294
,double8B

	full_text

double %297
,double8B

	full_text

double %304
Qstore8BF
D
	full_text7
5
3store double %305, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %305
.double*8B

	full_text

double* %126
Abitcast8B4
2
	full_text%
#
!%306 = bitcast i64 %181 to double
&i648B

	full_text


i64 %181
:fmul8B0
.
	full_text!

%307 = fmul double %297, %306
,double8B

	full_text

double %297
,double8B

	full_text

double %306
Pstore8BE
C
	full_text6
4
2store double %307, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %307
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%308 = fmul double %299, 4.000000e-01
,double8B

	full_text

double %299
Cfsub8B9
7
	full_text*
(
&%309 = fsub double -0.000000e+00, %308
,double8B

	full_text

double %308
ucall8Bk
i
	full_text\
Z
X%310 = tail call double @llvm.fmuladd.f64(double %302, double 1.400000e+00, double %309)
,double8B

	full_text

double %302
,double8B

	full_text

double %309
:fmul8B0
.
	full_text!

%311 = fmul double %297, %310
,double8B

	full_text

double %297
,double8B

	full_text

double %310
Qstore8BF
D
	full_text7
5
3store double %311, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %311
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%312 = fmul double %296, %300
,double8B

	full_text

double %296
,double8B

	full_text

double %300
:fmul8B0
.
	full_text!

%313 = fmul double %296, %306
,double8B

	full_text

double %296
,double8B

	full_text

double %306
:fmul8B0
.
	full_text!

%314 = fmul double %296, %302
,double8B

	full_text

double %296
,double8B

	full_text

double %302
Oload8BE
C
	full_text6
4
2%315 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
:fmul8B0
.
	full_text!

%316 = fmul double %115, %315
,double8B

	full_text

double %115
,double8B

	full_text

double %315
Pload8BF
D
	full_text7
5
3%317 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
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
Abitcast8B4
2
	full_text%
#
!%319 = bitcast i64 %250 to double
&i648B

	full_text


i64 %250
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
!%321 = bitcast i64 %267 to double
&i648B

	full_text


i64 %267
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
:fsub8B0
.
	full_text!

%323 = fsub double %312, %316
,double8B

	full_text

double %312
,double8B

	full_text

double %316
Bfmul8B8
6
	full_text)
'
%%324 = fmul double %323, 6.300000e+01
,double8B

	full_text

double %323
Pstore8BE
C
	full_text6
4
2store double %324, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %324
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%325 = fsub double %297, %318
,double8B

	full_text

double %297
,double8B

	full_text

double %318
Bfmul8B8
6
	full_text)
'
%%326 = fmul double %325, 8.400000e+01
,double8B

	full_text

double %325
Pstore8BE
C
	full_text6
4
2store double %326, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %326
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%327 = fsub double %313, %320
,double8B

	full_text

double %313
,double8B

	full_text

double %320
Bfmul8B8
6
	full_text)
'
%%328 = fmul double %327, 6.300000e+01
,double8B

	full_text

double %327
Pstore8BE
C
	full_text6
4
2store double %328, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %328
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%329 = fmul double %297, %297
,double8B

	full_text

double %297
,double8B

	full_text

double %297
mcall8Bc
a
	full_textT
R
P%330 = tail call double @llvm.fmuladd.f64(double %312, double %312, double %329)
,double8B

	full_text

double %312
,double8B

	full_text

double %312
,double8B

	full_text

double %329
mcall8Bc
a
	full_textT
R
P%331 = tail call double @llvm.fmuladd.f64(double %313, double %313, double %330)
,double8B

	full_text

double %313
,double8B

	full_text

double %313
,double8B

	full_text

double %330
:fmul8B0
.
	full_text!

%332 = fmul double %318, %318
,double8B

	full_text

double %318
,double8B

	full_text

double %318
mcall8Bc
a
	full_textT
R
P%333 = tail call double @llvm.fmuladd.f64(double %316, double %316, double %332)
,double8B

	full_text

double %316
,double8B

	full_text

double %316
,double8B

	full_text

double %332
mcall8Bc
a
	full_textT
R
P%334 = tail call double @llvm.fmuladd.f64(double %320, double %320, double %333)
,double8B

	full_text

double %320
,double8B

	full_text

double %320
,double8B

	full_text

double %333
:fsub8B0
.
	full_text!

%335 = fsub double %331, %334
,double8B

	full_text

double %331
,double8B

	full_text

double %334
Cfsub8B9
7
	full_text*
(
&%336 = fsub double -0.000000e+00, %332
,double8B

	full_text

double %332
mcall8Bc
a
	full_textT
R
P%337 = tail call double @llvm.fmuladd.f64(double %297, double %297, double %336)
,double8B

	full_text

double %297
,double8B

	full_text

double %297
,double8B

	full_text

double %336
Bfmul8B8
6
	full_text)
'
%%338 = fmul double %337, 1.050000e+01
,double8B

	full_text

double %337
{call8Bq
o
	full_textb
`
^%339 = tail call double @llvm.fmuladd.f64(double %335, double 0xC03E3D70A3D70A3B, double %338)
,double8B

	full_text

double %335
,double8B

	full_text

double %338
:fsub8B0
.
	full_text!

%340 = fsub double %314, %322
,double8B

	full_text

double %314
,double8B

	full_text

double %322
{call8Bq
o
	full_textb
`
^%341 = tail call double @llvm.fmuladd.f64(double %340, double 0x405EDEB851EB851E, double %339)
,double8B

	full_text

double %340
,double8B

	full_text

double %339
Pstore8BE
C
	full_text6
4
2store double %341, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %341
.double*8B

	full_text

double* %163
°getelementptr8Bç
ä
	full_text}
{
y%342 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 1, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%343 = load double, double* %342, align 8, !tbaa !8
.double*8B

	full_text

double* %342
Qload8BG
E
	full_text8
6
4%344 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
:fsub8B0
.
	full_text!

%345 = fsub double %294, %344
,double8B

	full_text

double %294
,double8B

	full_text

double %344
vcall8Bl
j
	full_text]
[
Y%346 = tail call double @llvm.fmuladd.f64(double %345, double -3.150000e+01, double %343)
,double8B

	full_text

double %345
,double8B

	full_text

double %343
°getelementptr8Bç
ä
	full_text}
{
y%347 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 1, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%348 = load double, double* %347, align 8, !tbaa !8
.double*8B

	full_text

double* %347
Pload8BF
D
	full_text7
5
3%349 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
:fsub8B0
.
	full_text!

%350 = fsub double %301, %349
,double8B

	full_text

double %301
,double8B

	full_text

double %349
vcall8Bl
j
	full_text]
[
Y%351 = tail call double @llvm.fmuladd.f64(double %350, double -3.150000e+01, double %348)
,double8B

	full_text

double %350
,double8B

	full_text

double %348
°getelementptr8Bç
ä
	full_text}
{
y%352 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 1, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%353 = load double, double* %352, align 8, !tbaa !8
.double*8B

	full_text

double* %352
Qload8BG
E
	full_text8
6
4%354 = load double, double* %236, align 16, !tbaa !8
.double*8B

	full_text

double* %236
:fsub8B0
.
	full_text!

%355 = fsub double %305, %354
,double8B

	full_text

double %305
,double8B

	full_text

double %354
vcall8Bl
j
	full_text]
[
Y%356 = tail call double @llvm.fmuladd.f64(double %355, double -3.150000e+01, double %353)
,double8B

	full_text

double %355
,double8B

	full_text

double %353
°getelementptr8Bç
ä
	full_text}
{
y%357 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 1, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%358 = load double, double* %357, align 8, !tbaa !8
.double*8B

	full_text

double* %357
Pload8BF
D
	full_text7
5
3%359 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
:fsub8B0
.
	full_text!

%360 = fsub double %307, %359
,double8B

	full_text

double %307
,double8B

	full_text

double %359
vcall8Bl
j
	full_text]
[
Y%361 = tail call double @llvm.fmuladd.f64(double %360, double -3.150000e+01, double %358)
,double8B

	full_text

double %360
,double8B

	full_text

double %358
°getelementptr8Bç
ä
	full_text}
{
y%362 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 1, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%363 = load double, double* %362, align 8, !tbaa !8
.double*8B

	full_text

double* %362
Qload8BG
E
	full_text8
6
4%364 = load double, double* %271, align 16, !tbaa !8
.double*8B

	full_text

double* %271
:fsub8B0
.
	full_text!

%365 = fsub double %311, %364
,double8B

	full_text

double %311
,double8B

	full_text

double %364
vcall8Bl
j
	full_text]
[
Y%366 = tail call double @llvm.fmuladd.f64(double %365, double -3.150000e+01, double %363)
,double8B

	full_text

double %365
,double8B

	full_text

double %363
Pload8BF
D
	full_text7
5
3%367 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
Pload8BF
D
	full_text7
5
3%368 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%369 = tail call double @llvm.fmuladd.f64(double %368, double -2.000000e+00, double %367)
,double8B

	full_text

double %368
,double8B

	full_text

double %367
Oload8BE
C
	full_text6
4
2%370 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%371 = fadd double %369, %370
,double8B

	full_text

double %369
,double8B

	full_text

double %370
{call8Bq
o
	full_textb
`
^%372 = tail call double @llvm.fmuladd.f64(double %371, double 0x40A7418000000001, double %346)
,double8B

	full_text

double %371
,double8B

	full_text

double %346
Pload8BF
D
	full_text7
5
3%373 = load double, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
:fsub8B0
.
	full_text!

%374 = fsub double %324, %373
,double8B

	full_text

double %324
,double8B

	full_text

double %373
{call8Bq
o
	full_textb
`
^%375 = tail call double @llvm.fmuladd.f64(double %374, double 0x4019333333333334, double %351)
,double8B

	full_text

double %374
,double8B

	full_text

double %351
Pload8BF
D
	full_text7
5
3%376 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
vcall8Bl
j
	full_text]
[
Y%377 = tail call double @llvm.fmuladd.f64(double %315, double -2.000000e+00, double %376)
,double8B

	full_text

double %315
,double8B

	full_text

double %376
:fadd8B0
.
	full_text!

%378 = fadd double %300, %377
,double8B

	full_text

double %300
,double8B

	full_text

double %377
{call8Bq
o
	full_textb
`
^%379 = tail call double @llvm.fmuladd.f64(double %378, double 0x40A7418000000001, double %375)
,double8B

	full_text

double %378
,double8B

	full_text

double %375
Qload8BG
E
	full_text8
6
4%380 = load double, double* %242, align 16, !tbaa !8
.double*8B

	full_text

double* %242
:fsub8B0
.
	full_text!

%381 = fsub double %326, %380
,double8B

	full_text

double %326
,double8B

	full_text

double %380
{call8Bq
o
	full_textb
`
^%382 = tail call double @llvm.fmuladd.f64(double %381, double 0x4019333333333334, double %356)
,double8B

	full_text

double %381
,double8B

	full_text

double %356
Pload8BF
D
	full_text7
5
3%383 = load double, double* %227, align 8, !tbaa !8
.double*8B

	full_text

double* %227
vcall8Bl
j
	full_text]
[
Y%384 = tail call double @llvm.fmuladd.f64(double %317, double -2.000000e+00, double %383)
,double8B

	full_text

double %317
,double8B

	full_text

double %383
:fadd8B0
.
	full_text!

%385 = fadd double %384, %294
,double8B

	full_text

double %384
,double8B

	full_text

double %294
{call8Bq
o
	full_textb
`
^%386 = tail call double @llvm.fmuladd.f64(double %385, double 0x40A7418000000001, double %382)
,double8B

	full_text

double %385
,double8B

	full_text

double %382
Pload8BF
D
	full_text7
5
3%387 = load double, double* %259, align 8, !tbaa !8
.double*8B

	full_text

double* %259
:fsub8B0
.
	full_text!

%388 = fsub double %328, %387
,double8B

	full_text

double %328
,double8B

	full_text

double %387
{call8Bq
o
	full_textb
`
^%389 = tail call double @llvm.fmuladd.f64(double %388, double 0x4019333333333334, double %361)
,double8B

	full_text

double %388
,double8B

	full_text

double %361
Pload8BF
D
	full_text7
5
3%390 = load double, double* %244, align 8, !tbaa !8
.double*8B

	full_text

double* %244
vcall8Bl
j
	full_text]
[
Y%391 = tail call double @llvm.fmuladd.f64(double %319, double -2.000000e+00, double %390)
,double8B

	full_text

double %319
,double8B

	full_text

double %390
:fadd8B0
.
	full_text!

%392 = fadd double %391, %306
,double8B

	full_text

double %391
,double8B

	full_text

double %306
{call8Bq
o
	full_textb
`
^%393 = tail call double @llvm.fmuladd.f64(double %392, double 0x40A7418000000001, double %389)
,double8B

	full_text

double %392
,double8B

	full_text

double %389
Qload8BG
E
	full_text8
6
4%394 = load double, double* %277, align 16, !tbaa !8
.double*8B

	full_text

double* %277
:fsub8B0
.
	full_text!

%395 = fsub double %341, %394
,double8B

	full_text

double %341
,double8B

	full_text

double %394
{call8Bq
o
	full_textb
`
^%396 = tail call double @llvm.fmuladd.f64(double %395, double 0x4019333333333334, double %366)
,double8B

	full_text

double %395
,double8B

	full_text

double %366
Pload8BF
D
	full_text7
5
3%397 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
vcall8Bl
j
	full_text]
[
Y%398 = tail call double @llvm.fmuladd.f64(double %321, double -2.000000e+00, double %397)
,double8B

	full_text

double %321
,double8B

	full_text

double %397
:fadd8B0
.
	full_text!

%399 = fadd double %398, %302
,double8B

	full_text

double %398
,double8B

	full_text

double %302
{call8Bq
o
	full_textb
`
^%400 = tail call double @llvm.fmuladd.f64(double %399, double 0x40A7418000000001, double %396)
,double8B

	full_text

double %399
,double8B

	full_text

double %396
kcall8Ba
_
	full_textR
P
N%401 = tail call double @_Z3maxdd(double 7.500000e-01, double 7.500000e-01) #5
ccall8BY
W
	full_textJ
H
F%402 = tail call double @_Z3maxdd(double %401, double 1.000000e+00) #5
,double8B

	full_text

double %401
Bfmul8B8
6
	full_text)
'
%%403 = fmul double %402, 2.500000e-01
,double8B

	full_text

double %402
Cfsub8B9
7
	full_text*
(
&%404 = fsub double -0.000000e+00, %403
,double8B

	full_text

double %403
Bfmul8B8
6
	full_text)
'
%%405 = fmul double %370, 4.000000e+00
,double8B

	full_text

double %370
Cfsub8B9
7
	full_text*
(
&%406 = fsub double -0.000000e+00, %405
,double8B

	full_text

double %405
ucall8Bk
i
	full_text\
Z
X%407 = tail call double @llvm.fmuladd.f64(double %368, double 5.000000e+00, double %406)
,double8B

	full_text

double %368
,double8B

	full_text

double %406
Qload8BG
E
	full_text8
6
4%408 = load double, double* %196, align 16, !tbaa !8
.double*8B

	full_text

double* %196
:fadd8B0
.
	full_text!

%409 = fadd double %408, %407
,double8B

	full_text

double %408
,double8B

	full_text

double %407
mcall8Bc
a
	full_textT
R
P%410 = tail call double @llvm.fmuladd.f64(double %404, double %409, double %372)
,double8B

	full_text

double %404
,double8B

	full_text

double %409
,double8B

	full_text

double %372
Pstore8BE
C
	full_text6
4
2store double %410, double* %342, align 8, !tbaa !8
,double8B

	full_text

double %410
.double*8B

	full_text

double* %342
Oload8BE
C
	full_text6
4
2%411 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
Oload8BE
C
	full_text6
4
2%412 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
Bfmul8B8
6
	full_text)
'
%%413 = fmul double %412, 4.000000e+00
,double8B

	full_text

double %412
Cfsub8B9
7
	full_text*
(
&%414 = fsub double -0.000000e+00, %413
,double8B

	full_text

double %413
ucall8Bk
i
	full_text\
Z
X%415 = tail call double @llvm.fmuladd.f64(double %411, double 5.000000e+00, double %414)
,double8B

	full_text

double %411
,double8B

	full_text

double %414
Pload8BF
D
	full_text7
5
3%416 = load double, double* %172, align 8, !tbaa !8
.double*8B

	full_text

double* %172
:fadd8B0
.
	full_text!

%417 = fadd double %416, %415
,double8B

	full_text

double %416
,double8B

	full_text

double %415
mcall8Bc
a
	full_textT
R
P%418 = tail call double @llvm.fmuladd.f64(double %404, double %417, double %379)
,double8B

	full_text

double %404
,double8B

	full_text

double %417
,double8B

	full_text

double %379
Pstore8BE
C
	full_text6
4
2store double %418, double* %347, align 8, !tbaa !8
,double8B

	full_text

double %418
.double*8B

	full_text

double* %347
Pload8BF
D
	full_text7
5
3%419 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
Oload8BE
C
	full_text6
4
2%420 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
Bfmul8B8
6
	full_text)
'
%%421 = fmul double %420, 4.000000e+00
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
ucall8Bk
i
	full_text\
Z
X%423 = tail call double @llvm.fmuladd.f64(double %419, double 5.000000e+00, double %422)
,double8B

	full_text

double %419
,double8B

	full_text

double %422
Qload8BG
E
	full_text8
6
4%424 = load double, double* %177, align 16, !tbaa !8
.double*8B

	full_text

double* %177
:fadd8B0
.
	full_text!

%425 = fadd double %424, %423
,double8B

	full_text

double %424
,double8B

	full_text

double %423
mcall8Bc
a
	full_textT
R
P%426 = tail call double @llvm.fmuladd.f64(double %404, double %425, double %386)
,double8B

	full_text

double %404
,double8B

	full_text

double %425
,double8B

	full_text

double %386
Pstore8BE
C
	full_text6
4
2store double %426, double* %352, align 8, !tbaa !8
,double8B

	full_text

double %426
.double*8B

	full_text

double* %352
Oload8BE
C
	full_text6
4
2%427 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Oload8BE
C
	full_text6
4
2%428 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
Bfmul8B8
6
	full_text)
'
%%429 = fmul double %428, 4.000000e+00
,double8B

	full_text

double %428
Cfsub8B9
7
	full_text*
(
&%430 = fsub double -0.000000e+00, %429
,double8B

	full_text

double %429
ucall8Bk
i
	full_text\
Z
X%431 = tail call double @llvm.fmuladd.f64(double %427, double 5.000000e+00, double %430)
,double8B

	full_text

double %427
,double8B

	full_text

double %430
Pload8BF
D
	full_text7
5
3%432 = load double, double* %182, align 8, !tbaa !8
.double*8B

	full_text

double* %182
:fadd8B0
.
	full_text!

%433 = fadd double %432, %431
,double8B

	full_text

double %432
,double8B

	full_text

double %431
mcall8Bc
a
	full_textT
R
P%434 = tail call double @llvm.fmuladd.f64(double %404, double %433, double %393)
,double8B

	full_text

double %404
,double8B

	full_text

double %433
,double8B

	full_text

double %393
Pstore8BE
C
	full_text6
4
2store double %434, double* %357, align 8, !tbaa !8
,double8B

	full_text

double %434
.double*8B

	full_text

double* %357
Pload8BF
D
	full_text7
5
3%435 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
Oload8BE
C
	full_text6
4
2%436 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
Bfmul8B8
6
	full_text)
'
%%437 = fmul double %436, 4.000000e+00
,double8B

	full_text

double %436
Cfsub8B9
7
	full_text*
(
&%438 = fsub double -0.000000e+00, %437
,double8B

	full_text

double %437
ucall8Bk
i
	full_text\
Z
X%439 = tail call double @llvm.fmuladd.f64(double %435, double 5.000000e+00, double %438)
,double8B

	full_text

double %435
,double8B

	full_text

double %438
Qload8BG
E
	full_text8
6
4%440 = load double, double* %187, align 16, !tbaa !8
.double*8B

	full_text

double* %187
:fadd8B0
.
	full_text!

%441 = fadd double %440, %439
,double8B

	full_text

double %440
,double8B

	full_text

double %439
mcall8Bc
a
	full_textT
R
P%442 = tail call double @llvm.fmuladd.f64(double %404, double %441, double %400)
,double8B

	full_text

double %404
,double8B

	full_text

double %441
,double8B

	full_text

double %400
Pstore8BE
C
	full_text6
4
2store double %442, double* %362, align 8, !tbaa !8
,double8B

	full_text

double %442
.double*8B

	full_text

double* %362
Ñgetelementptr8Bq
o
	full_textb
`
^%443 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %367, double* %443, align 16, !tbaa !8
,double8B

	full_text

double %367
.double*8B

	full_text

double* %443
Pstore8BE
C
	full_text6
4
2store double %368, double* %189, align 8, !tbaa !8
,double8B

	full_text

double %368
.double*8B

	full_text

double* %189
Pstore8BE
C
	full_text6
4
2store double %370, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %370
-double*8B

	full_text

double* %84
Ostore8BD
B
	full_text5
3
1store double %408, double* %82, align 8, !tbaa !8
,double8B

	full_text

double %408
-double*8B

	full_text

double* %82
Jload8B@
>
	full_text1
/
-%444 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
Kstore8B@
>
	full_text1
/
-store i64 %444, i64* %202, align 16, !tbaa !8
&i648B

	full_text


i64 %444
(i64*8B

	full_text

	i64* %202
Kload8BA
?
	full_text2
0
.%445 = load i64, i64* %204, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %204
Jstore8B?
=
	full_text0
.
,store i64 %445, i64* %199, align 8, !tbaa !8
&i648B

	full_text


i64 %445
(i64*8B

	full_text

	i64* %199
Jload8B@
>
	full_text1
/
-%446 = load i64, i64* %207, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %207
Kstore8B@
>
	full_text1
/
-store i64 %446, i64* %209, align 16, !tbaa !8
&i648B

	full_text


i64 %446
(i64*8B

	full_text

	i64* %209
Pstore8BE
C
	full_text6
4
2store double %376, double* %213, align 8, !tbaa !8
,double8B

	full_text

double %376
.double*8B

	full_text

double* %213
Pstore8BE
C
	full_text6
4
2store double %411, double* %210, align 8, !tbaa !8
,double8B

	full_text

double %411
.double*8B

	full_text

double* %210
Ostore8BD
B
	full_text5
3
1store double %412, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %412
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %416, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %416
-double*8B

	full_text

double* %41
Jload8B@
>
	full_text1
/
-%447 = load i64, i64* %217, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %217
Jstore8B?
=
	full_text0
.
,store i64 %447, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %447
(i64*8B

	full_text

	i64* %220
Jload8B@
>
	full_text1
/
-%448 = load i64, i64* %221, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %221
Jstore8B?
=
	full_text0
.
,store i64 %448, i64* %217, align 8, !tbaa !8
&i648B

	full_text


i64 %448
(i64*8B

	full_text

	i64* %217
Jload8B@
>
	full_text1
/
-%449 = load i64, i64* %223, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %223
Jstore8B?
=
	full_text0
.
,store i64 %449, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %449
(i64*8B

	full_text

	i64* %226
Qstore8BF
D
	full_text7
5
3store double %383, double* %230, align 16, !tbaa !8
,double8B

	full_text

double %383
.double*8B

	full_text

double* %230
Pstore8BE
C
	full_text6
4
2store double %419, double* %227, align 8, !tbaa !8
,double8B

	full_text

double %419
.double*8B

	full_text

double* %227
Pstore8BE
C
	full_text6
4
2store double %420, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %420
-double*8B

	full_text

double* %88
Ostore8BD
B
	full_text5
3
1store double %424, double* %46, align 8, !tbaa !8
,double8B

	full_text

double %424
-double*8B

	full_text

double* %46
Jload8B@
>
	full_text1
/
-%450 = load i64, i64* %234, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %234
Kstore8B@
>
	full_text1
/
-store i64 %450, i64* %237, align 16, !tbaa !8
&i648B

	full_text


i64 %450
(i64*8B

	full_text

	i64* %237
Kload8BA
?
	full_text2
0
.%451 = load i64, i64* %238, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %238
Jstore8B?
=
	full_text0
.
,store i64 %451, i64* %234, align 8, !tbaa !8
&i648B

	full_text


i64 %451
(i64*8B

	full_text

	i64* %234
Jload8B@
>
	full_text1
/
-%452 = load i64, i64* %240, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %240
Kstore8B@
>
	full_text1
/
-store i64 %452, i64* %243, align 16, !tbaa !8
&i648B

	full_text


i64 %452
(i64*8B

	full_text

	i64* %243
Jload8B@
>
	full_text1
/
-%453 = load i64, i64* %245, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %245
Jstore8B?
=
	full_text0
.
,store i64 %453, i64* %248, align 8, !tbaa !8
&i648B

	full_text


i64 %453
(i64*8B

	full_text

	i64* %248
Pstore8BE
C
	full_text6
4
2store double %427, double* %244, align 8, !tbaa !8
,double8B

	full_text

double %427
.double*8B

	full_text

double* %244
Ostore8BD
B
	full_text5
3
1store double %428, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %428
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %432, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %432
-double*8B

	full_text

double* %51
Jload8B@
>
	full_text1
/
-%454 = load i64, i64* %251, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %251
Jstore8B?
=
	full_text0
.
,store i64 %454, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %454
(i64*8B

	full_text

	i64* %254
Jload8B@
>
	full_text1
/
-%455 = load i64, i64* %255, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %255
Jstore8B?
=
	full_text0
.
,store i64 %455, i64* %251, align 8, !tbaa !8
&i648B

	full_text


i64 %455
(i64*8B

	full_text

	i64* %251
Jload8B@
>
	full_text1
/
-%456 = load i64, i64* %257, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %257
Jstore8B?
=
	full_text0
.
,store i64 %456, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %456
(i64*8B

	full_text

	i64* %260
Jload8B@
>
	full_text1
/
-%457 = load i64, i64* %262, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %262
Kstore8B@
>
	full_text1
/
-store i64 %457, i64* %265, align 16, !tbaa !8
&i648B

	full_text


i64 %457
(i64*8B

	full_text

	i64* %265
Pstore8BE
C
	full_text6
4
2store double %435, double* %261, align 8, !tbaa !8
,double8B

	full_text

double %435
.double*8B

	full_text

double* %261
Pstore8BE
C
	full_text6
4
2store double %436, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %436
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %440, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %440
-double*8B

	full_text

double* %56
Jload8B@
>
	full_text1
/
-%458 = load i64, i64* %269, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %269
Kstore8B@
>
	full_text1
/
-store i64 %458, i64* %272, align 16, !tbaa !8
&i648B

	full_text


i64 %458
(i64*8B

	full_text

	i64* %272
Kload8BA
?
	full_text2
0
.%459 = load i64, i64* %273, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %273
Jstore8B?
=
	full_text0
.
,store i64 %459, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %459
(i64*8B

	full_text

	i64* %269
Jload8B@
>
	full_text1
/
-%460 = load i64, i64* %275, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %275
Kstore8B@
>
	full_text1
/
-store i64 %460, i64* %278, align 16, !tbaa !8
&i648B

	full_text


i64 %460
(i64*8B

	full_text

	i64* %278
ögetelementptr8BÜ
É
	full_textv
t
r%461 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 4, i64 %32
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
Ibitcast8B<
:
	full_text-
+
)%462 = bitcast [5 x double]* %461 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %461
Jload8B@
>
	full_text1
/
-%463 = load i64, i64* %462, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %462
Kstore8B@
>
	full_text1
/
-store i64 %463, i64* %168, align 16, !tbaa !8
&i648B

	full_text


i64 %463
(i64*8B

	full_text

	i64* %168
°getelementptr8Bç
ä
	full_text}
{
y%464 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 4, i64 %32, i64 1
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
Cbitcast8B6
4
	full_text'
%
#%465 = bitcast double* %464 to i64*
.double*8B

	full_text

double* %464
Jload8B@
>
	full_text1
/
-%466 = load i64, i64* %465, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %465
Jstore8B?
=
	full_text0
.
,store i64 %466, i64* %173, align 8, !tbaa !8
&i648B

	full_text


i64 %466
(i64*8B

	full_text

	i64* %173
°getelementptr8Bç
ä
	full_text}
{
y%467 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 4, i64 %32, i64 2
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
Kstore8B@
>
	full_text1
/
-store i64 %469, i64* %178, align 16, !tbaa !8
&i648B

	full_text


i64 %469
(i64*8B

	full_text

	i64* %178
°getelementptr8Bç
ä
	full_text}
{
y%470 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 4, i64 %32, i64 3
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
Jstore8B?
=
	full_text0
.
,store i64 %472, i64* %183, align 8, !tbaa !8
&i648B

	full_text


i64 %472
(i64*8B

	full_text

	i64* %183
°getelementptr8Bç
ä
	full_text}
{
y%473 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 4, i64 %32, i64 4
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
Kstore8B@
>
	full_text1
/
-store i64 %475, i64* %188, align 16, !tbaa !8
&i648B

	full_text


i64 %475
(i64*8B

	full_text

	i64* %188
rgetelementptr8B_
]
	full_textP
N
L%476 = getelementptr inbounds [5 x double], [5 x double]* %111, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
Qstore8BF
D
	full_text7
5
3store double %424, double* %476, align 16, !tbaa !8
,double8B

	full_text

double %424
.double*8B

	full_text

double* %476
ågetelementptr8By
w
	full_textj
h
f%477 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 3, i64 %32
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
Pload8BF
D
	full_text7
5
3%478 = load double, double* %477, align 8, !tbaa !8
.double*8B

	full_text

double* %477
:fmul8B0
.
	full_text!

%479 = fmul double %478, %424
,double8B

	full_text

double %478
,double8B

	full_text

double %424
ågetelementptr8By
w
	full_textj
h
f%480 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 3, i64 %32
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

%482 = fmul double %479, %416
,double8B

	full_text

double %479
,double8B

	full_text

double %416
Pstore8BE
C
	full_text6
4
2store double %482, double* %121, align 8, !tbaa !8
,double8B

	full_text

double %482
.double*8B

	full_text

double* %121
:fsub8B0
.
	full_text!

%483 = fsub double %440, %481
,double8B

	full_text

double %440
,double8B

	full_text

double %481
Bfmul8B8
6
	full_text)
'
%%484 = fmul double %483, 4.000000e-01
,double8B

	full_text

double %483
mcall8Bc
a
	full_textT
R
P%485 = tail call double @llvm.fmuladd.f64(double %424, double %479, double %484)
,double8B

	full_text

double %424
,double8B

	full_text

double %479
,double8B

	full_text

double %484
Qstore8BF
D
	full_text7
5
3store double %485, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %485
.double*8B

	full_text

double* %126
:fmul8B0
.
	full_text!

%486 = fmul double %479, %432
,double8B

	full_text

double %479
,double8B

	full_text

double %432
Pstore8BE
C
	full_text6
4
2store double %486, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %486
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%487 = fmul double %481, 4.000000e-01
,double8B

	full_text

double %481
Cfsub8B9
7
	full_text*
(
&%488 = fsub double -0.000000e+00, %487
,double8B

	full_text

double %487
ucall8Bk
i
	full_text\
Z
X%489 = tail call double @llvm.fmuladd.f64(double %440, double 1.400000e+00, double %488)
,double8B

	full_text

double %440
,double8B

	full_text

double %488
:fmul8B0
.
	full_text!

%490 = fmul double %479, %489
,double8B

	full_text

double %479
,double8B

	full_text

double %489
Qstore8BF
D
	full_text7
5
3store double %490, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %490
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%491 = fmul double %478, %416
,double8B

	full_text

double %478
,double8B

	full_text

double %416
:fmul8B0
.
	full_text!

%492 = fmul double %478, %432
,double8B

	full_text

double %478
,double8B

	full_text

double %432
:fmul8B0
.
	full_text!

%493 = fmul double %478, %440
,double8B

	full_text

double %478
,double8B

	full_text

double %440
Pload8BF
D
	full_text7
5
3%494 = load double, double* %295, align 8, !tbaa !8
.double*8B

	full_text

double* %295
:fmul8B0
.
	full_text!

%495 = fmul double %494, %412
,double8B

	full_text

double %494
,double8B

	full_text

double %412
:fmul8B0
.
	full_text!

%496 = fmul double %494, %420
,double8B

	full_text

double %494
,double8B

	full_text

double %420
:fmul8B0
.
	full_text!

%497 = fmul double %494, %428
,double8B

	full_text

double %494
,double8B

	full_text

double %428
:fmul8B0
.
	full_text!

%498 = fmul double %494, %436
,double8B

	full_text

double %494
,double8B

	full_text

double %436
:fsub8B0
.
	full_text!

%499 = fsub double %491, %495
,double8B

	full_text

double %491
,double8B

	full_text

double %495
Bfmul8B8
6
	full_text)
'
%%500 = fmul double %499, 6.300000e+01
,double8B

	full_text

double %499
Pstore8BE
C
	full_text6
4
2store double %500, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %500
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%501 = fsub double %479, %496
,double8B

	full_text

double %479
,double8B

	full_text

double %496
Bfmul8B8
6
	full_text)
'
%%502 = fmul double %501, 8.400000e+01
,double8B

	full_text

double %501
Pstore8BE
C
	full_text6
4
2store double %502, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %502
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%503 = fsub double %492, %497
,double8B

	full_text

double %492
,double8B

	full_text

double %497
Bfmul8B8
6
	full_text)
'
%%504 = fmul double %503, 6.300000e+01
,double8B

	full_text

double %503
Pstore8BE
C
	full_text6
4
2store double %504, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %504
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%505 = fmul double %479, %479
,double8B

	full_text

double %479
,double8B

	full_text

double %479
mcall8Bc
a
	full_textT
R
P%506 = tail call double @llvm.fmuladd.f64(double %491, double %491, double %505)
,double8B

	full_text

double %491
,double8B

	full_text

double %491
,double8B

	full_text

double %505
mcall8Bc
a
	full_textT
R
P%507 = tail call double @llvm.fmuladd.f64(double %492, double %492, double %506)
,double8B

	full_text

double %492
,double8B

	full_text

double %492
,double8B

	full_text

double %506
:fmul8B0
.
	full_text!

%508 = fmul double %496, %496
,double8B

	full_text

double %496
,double8B

	full_text

double %496
mcall8Bc
a
	full_textT
R
P%509 = tail call double @llvm.fmuladd.f64(double %495, double %495, double %508)
,double8B

	full_text

double %495
,double8B

	full_text

double %495
,double8B

	full_text

double %508
mcall8Bc
a
	full_textT
R
P%510 = tail call double @llvm.fmuladd.f64(double %497, double %497, double %509)
,double8B

	full_text

double %497
,double8B

	full_text

double %497
,double8B

	full_text

double %509
:fsub8B0
.
	full_text!

%511 = fsub double %507, %510
,double8B

	full_text

double %507
,double8B

	full_text

double %510
Cfsub8B9
7
	full_text*
(
&%512 = fsub double -0.000000e+00, %508
,double8B

	full_text

double %508
mcall8Bc
a
	full_textT
R
P%513 = tail call double @llvm.fmuladd.f64(double %479, double %479, double %512)
,double8B

	full_text

double %479
,double8B

	full_text

double %479
,double8B

	full_text

double %512
Bfmul8B8
6
	full_text)
'
%%514 = fmul double %513, 1.050000e+01
,double8B

	full_text

double %513
{call8Bq
o
	full_textb
`
^%515 = tail call double @llvm.fmuladd.f64(double %511, double 0xC03E3D70A3D70A3B, double %514)
,double8B

	full_text

double %511
,double8B

	full_text

double %514
:fsub8B0
.
	full_text!

%516 = fsub double %493, %498
,double8B

	full_text

double %493
,double8B

	full_text

double %498
{call8Bq
o
	full_textb
`
^%517 = tail call double @llvm.fmuladd.f64(double %516, double 0x405EDEB851EB851E, double %515)
,double8B

	full_text

double %516
,double8B

	full_text

double %515
Pstore8BE
C
	full_text6
4
2store double %517, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %517
.double*8B

	full_text

double* %163
°getelementptr8Bç
ä
	full_text}
{
y%518 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 2, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%519 = load double, double* %518, align 8, !tbaa !8
.double*8B

	full_text

double* %518
Qload8BG
E
	full_text8
6
4%520 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
:fsub8B0
.
	full_text!

%521 = fsub double %424, %520
,double8B

	full_text

double %424
,double8B

	full_text

double %520
vcall8Bl
j
	full_text]
[
Y%522 = tail call double @llvm.fmuladd.f64(double %521, double -3.150000e+01, double %519)
,double8B

	full_text

double %521
,double8B

	full_text

double %519
°getelementptr8Bç
ä
	full_text}
{
y%523 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 2, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%524 = load double, double* %523, align 8, !tbaa !8
.double*8B

	full_text

double* %523
Pload8BF
D
	full_text7
5
3%525 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
:fsub8B0
.
	full_text!

%526 = fsub double %482, %525
,double8B

	full_text

double %482
,double8B

	full_text

double %525
vcall8Bl
j
	full_text]
[
Y%527 = tail call double @llvm.fmuladd.f64(double %526, double -3.150000e+01, double %524)
,double8B

	full_text

double %526
,double8B

	full_text

double %524
°getelementptr8Bç
ä
	full_text}
{
y%528 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 2, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%529 = load double, double* %528, align 8, !tbaa !8
.double*8B

	full_text

double* %528
Qload8BG
E
	full_text8
6
4%530 = load double, double* %236, align 16, !tbaa !8
.double*8B

	full_text

double* %236
:fsub8B0
.
	full_text!

%531 = fsub double %485, %530
,double8B

	full_text

double %485
,double8B

	full_text

double %530
vcall8Bl
j
	full_text]
[
Y%532 = tail call double @llvm.fmuladd.f64(double %531, double -3.150000e+01, double %529)
,double8B

	full_text

double %531
,double8B

	full_text

double %529
°getelementptr8Bç
ä
	full_text}
{
y%533 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 2, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%534 = load double, double* %533, align 8, !tbaa !8
.double*8B

	full_text

double* %533
Pload8BF
D
	full_text7
5
3%535 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
:fsub8B0
.
	full_text!

%536 = fsub double %486, %535
,double8B

	full_text

double %486
,double8B

	full_text

double %535
vcall8Bl
j
	full_text]
[
Y%537 = tail call double @llvm.fmuladd.f64(double %536, double -3.150000e+01, double %534)
,double8B

	full_text

double %536
,double8B

	full_text

double %534
°getelementptr8Bç
ä
	full_text}
{
y%538 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 2, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
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
3%539 = load double, double* %538, align 8, !tbaa !8
.double*8B

	full_text

double* %538
Qload8BG
E
	full_text8
6
4%540 = load double, double* %271, align 16, !tbaa !8
.double*8B

	full_text

double* %271
:fsub8B0
.
	full_text!

%541 = fsub double %490, %540
,double8B

	full_text

double %490
,double8B

	full_text

double %540
vcall8Bl
j
	full_text]
[
Y%542 = tail call double @llvm.fmuladd.f64(double %541, double -3.150000e+01, double %539)
,double8B

	full_text

double %541
,double8B

	full_text

double %539
Pload8BF
D
	full_text7
5
3%543 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
Pload8BF
D
	full_text7
5
3%544 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%545 = tail call double @llvm.fmuladd.f64(double %544, double -2.000000e+00, double %543)
,double8B

	full_text

double %544
,double8B

	full_text

double %543
Oload8BE
C
	full_text6
4
2%546 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%547 = fadd double %545, %546
,double8B

	full_text

double %545
,double8B

	full_text

double %546
{call8Bq
o
	full_textb
`
^%548 = tail call double @llvm.fmuladd.f64(double %547, double 0x40A7418000000001, double %522)
,double8B

	full_text

double %547
,double8B

	full_text

double %522
Pload8BF
D
	full_text7
5
3%549 = load double, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
:fsub8B0
.
	full_text!

%550 = fsub double %500, %549
,double8B

	full_text

double %500
,double8B

	full_text

double %549
{call8Bq
o
	full_textb
`
^%551 = tail call double @llvm.fmuladd.f64(double %550, double 0x4019333333333334, double %527)
,double8B

	full_text

double %550
,double8B

	full_text

double %527
Pload8BF
D
	full_text7
5
3%552 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
vcall8Bl
j
	full_text]
[
Y%553 = tail call double @llvm.fmuladd.f64(double %412, double -2.000000e+00, double %552)
,double8B

	full_text

double %412
,double8B

	full_text

double %552
:fadd8B0
.
	full_text!

%554 = fadd double %416, %553
,double8B

	full_text

double %416
,double8B

	full_text

double %553
{call8Bq
o
	full_textb
`
^%555 = tail call double @llvm.fmuladd.f64(double %554, double 0x40A7418000000001, double %551)
,double8B

	full_text

double %554
,double8B

	full_text

double %551
Qload8BG
E
	full_text8
6
4%556 = load double, double* %242, align 16, !tbaa !8
.double*8B

	full_text

double* %242
:fsub8B0
.
	full_text!

%557 = fsub double %502, %556
,double8B

	full_text

double %502
,double8B

	full_text

double %556
{call8Bq
o
	full_textb
`
^%558 = tail call double @llvm.fmuladd.f64(double %557, double 0x4019333333333334, double %532)
,double8B

	full_text

double %557
,double8B

	full_text

double %532
Pload8BF
D
	full_text7
5
3%559 = load double, double* %227, align 8, !tbaa !8
.double*8B

	full_text

double* %227
vcall8Bl
j
	full_text]
[
Y%560 = tail call double @llvm.fmuladd.f64(double %420, double -2.000000e+00, double %559)
,double8B

	full_text

double %420
,double8B

	full_text

double %559
:fadd8B0
.
	full_text!

%561 = fadd double %424, %560
,double8B

	full_text

double %424
,double8B

	full_text

double %560
{call8Bq
o
	full_textb
`
^%562 = tail call double @llvm.fmuladd.f64(double %561, double 0x40A7418000000001, double %558)
,double8B

	full_text

double %561
,double8B

	full_text

double %558
Pload8BF
D
	full_text7
5
3%563 = load double, double* %259, align 8, !tbaa !8
.double*8B

	full_text

double* %259
:fsub8B0
.
	full_text!

%564 = fsub double %504, %563
,double8B

	full_text

double %504
,double8B

	full_text

double %563
{call8Bq
o
	full_textb
`
^%565 = tail call double @llvm.fmuladd.f64(double %564, double 0x4019333333333334, double %537)
,double8B

	full_text

double %564
,double8B

	full_text

double %537
Pload8BF
D
	full_text7
5
3%566 = load double, double* %244, align 8, !tbaa !8
.double*8B

	full_text

double* %244
vcall8Bl
j
	full_text]
[
Y%567 = tail call double @llvm.fmuladd.f64(double %428, double -2.000000e+00, double %566)
,double8B

	full_text

double %428
,double8B

	full_text

double %566
:fadd8B0
.
	full_text!

%568 = fadd double %432, %567
,double8B

	full_text

double %432
,double8B

	full_text

double %567
{call8Bq
o
	full_textb
`
^%569 = tail call double @llvm.fmuladd.f64(double %568, double 0x40A7418000000001, double %565)
,double8B

	full_text

double %568
,double8B

	full_text

double %565
Qload8BG
E
	full_text8
6
4%570 = load double, double* %277, align 16, !tbaa !8
.double*8B

	full_text

double* %277
:fsub8B0
.
	full_text!

%571 = fsub double %517, %570
,double8B

	full_text

double %517
,double8B

	full_text

double %570
{call8Bq
o
	full_textb
`
^%572 = tail call double @llvm.fmuladd.f64(double %571, double 0x4019333333333334, double %542)
,double8B

	full_text

double %571
,double8B

	full_text

double %542
Pload8BF
D
	full_text7
5
3%573 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
vcall8Bl
j
	full_text]
[
Y%574 = tail call double @llvm.fmuladd.f64(double %436, double -2.000000e+00, double %573)
,double8B

	full_text

double %436
,double8B

	full_text

double %573
:fadd8B0
.
	full_text!

%575 = fadd double %440, %574
,double8B

	full_text

double %440
,double8B

	full_text

double %574
{call8Bq
o
	full_textb
`
^%576 = tail call double @llvm.fmuladd.f64(double %575, double 0x40A7418000000001, double %572)
,double8B

	full_text

double %575
,double8B

	full_text

double %572
Bfmul8B8
6
	full_text)
'
%%577 = fmul double %544, 6.000000e+00
,double8B

	full_text

double %544
vcall8Bl
j
	full_text]
[
Y%578 = tail call double @llvm.fmuladd.f64(double %543, double -4.000000e+00, double %577)
,double8B

	full_text

double %543
,double8B

	full_text

double %577
vcall8Bl
j
	full_text]
[
Y%579 = tail call double @llvm.fmuladd.f64(double %546, double -4.000000e+00, double %578)
,double8B

	full_text

double %546
,double8B

	full_text

double %578
Qload8BG
E
	full_text8
6
4%580 = load double, double* %196, align 16, !tbaa !8
.double*8B

	full_text

double* %196
:fadd8B0
.
	full_text!

%581 = fadd double %580, %579
,double8B

	full_text

double %580
,double8B

	full_text

double %579
mcall8Bc
a
	full_textT
R
P%582 = tail call double @llvm.fmuladd.f64(double %404, double %581, double %548)
,double8B

	full_text

double %404
,double8B

	full_text

double %581
,double8B

	full_text

double %548
Pstore8BE
C
	full_text6
4
2store double %582, double* %518, align 8, !tbaa !8
,double8B

	full_text

double %582
.double*8B

	full_text

double* %518
Oload8BE
C
	full_text6
4
2%583 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
Bfmul8B8
6
	full_text)
'
%%584 = fmul double %583, 6.000000e+00
,double8B

	full_text

double %583
vcall8Bl
j
	full_text]
[
Y%585 = tail call double @llvm.fmuladd.f64(double %552, double -4.000000e+00, double %584)
,double8B

	full_text

double %552
,double8B

	full_text

double %584
Oload8BE
C
	full_text6
4
2%586 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
vcall8Bl
j
	full_text]
[
Y%587 = tail call double @llvm.fmuladd.f64(double %586, double -4.000000e+00, double %585)
,double8B

	full_text

double %586
,double8B

	full_text

double %585
Pload8BF
D
	full_text7
5
3%588 = load double, double* %172, align 8, !tbaa !8
.double*8B

	full_text

double* %172
:fadd8B0
.
	full_text!

%589 = fadd double %588, %587
,double8B

	full_text

double %588
,double8B

	full_text

double %587
mcall8Bc
a
	full_textT
R
P%590 = tail call double @llvm.fmuladd.f64(double %404, double %589, double %555)
,double8B

	full_text

double %404
,double8B

	full_text

double %589
,double8B

	full_text

double %555
Pstore8BE
C
	full_text6
4
2store double %590, double* %523, align 8, !tbaa !8
,double8B

	full_text

double %590
.double*8B

	full_text

double* %523
Pload8BF
D
	full_text7
5
3%591 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
Bfmul8B8
6
	full_text)
'
%%592 = fmul double %591, 6.000000e+00
,double8B

	full_text

double %591
vcall8Bl
j
	full_text]
[
Y%593 = tail call double @llvm.fmuladd.f64(double %559, double -4.000000e+00, double %592)
,double8B

	full_text

double %559
,double8B

	full_text

double %592
Oload8BE
C
	full_text6
4
2%594 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
vcall8Bl
j
	full_text]
[
Y%595 = tail call double @llvm.fmuladd.f64(double %594, double -4.000000e+00, double %593)
,double8B

	full_text

double %594
,double8B

	full_text

double %593
Qload8BG
E
	full_text8
6
4%596 = load double, double* %177, align 16, !tbaa !8
.double*8B

	full_text

double* %177
:fadd8B0
.
	full_text!

%597 = fadd double %596, %595
,double8B

	full_text

double %596
,double8B

	full_text

double %595
mcall8Bc
a
	full_textT
R
P%598 = tail call double @llvm.fmuladd.f64(double %404, double %597, double %562)
,double8B

	full_text

double %404
,double8B

	full_text

double %597
,double8B

	full_text

double %562
Pstore8BE
C
	full_text6
4
2store double %598, double* %528, align 8, !tbaa !8
,double8B

	full_text

double %598
.double*8B

	full_text

double* %528
Oload8BE
C
	full_text6
4
2%599 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Bfmul8B8
6
	full_text)
'
%%600 = fmul double %599, 6.000000e+00
,double8B

	full_text

double %599
vcall8Bl
j
	full_text]
[
Y%601 = tail call double @llvm.fmuladd.f64(double %566, double -4.000000e+00, double %600)
,double8B

	full_text

double %566
,double8B

	full_text

double %600
Oload8BE
C
	full_text6
4
2%602 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
vcall8Bl
j
	full_text]
[
Y%603 = tail call double @llvm.fmuladd.f64(double %602, double -4.000000e+00, double %601)
,double8B

	full_text

double %602
,double8B

	full_text

double %601
Pload8BF
D
	full_text7
5
3%604 = load double, double* %182, align 8, !tbaa !8
.double*8B

	full_text

double* %182
:fadd8B0
.
	full_text!

%605 = fadd double %604, %603
,double8B

	full_text

double %604
,double8B

	full_text

double %603
mcall8Bc
a
	full_textT
R
P%606 = tail call double @llvm.fmuladd.f64(double %404, double %605, double %569)
,double8B

	full_text

double %404
,double8B

	full_text

double %605
,double8B

	full_text

double %569
Pstore8BE
C
	full_text6
4
2store double %606, double* %533, align 8, !tbaa !8
,double8B

	full_text

double %606
.double*8B

	full_text

double* %533
Pload8BF
D
	full_text7
5
3%607 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
Bfmul8B8
6
	full_text)
'
%%608 = fmul double %607, 6.000000e+00
,double8B

	full_text

double %607
vcall8Bl
j
	full_text]
[
Y%609 = tail call double @llvm.fmuladd.f64(double %573, double -4.000000e+00, double %608)
,double8B

	full_text

double %573
,double8B

	full_text

double %608
Oload8BE
C
	full_text6
4
2%610 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
vcall8Bl
j
	full_text]
[
Y%611 = tail call double @llvm.fmuladd.f64(double %610, double -4.000000e+00, double %609)
,double8B

	full_text

double %610
,double8B

	full_text

double %609
Qload8BG
E
	full_text8
6
4%612 = load double, double* %187, align 16, !tbaa !8
.double*8B

	full_text

double* %187
:fadd8B0
.
	full_text!

%613 = fadd double %612, %611
,double8B

	full_text

double %612
,double8B

	full_text

double %611
mcall8Bc
a
	full_textT
R
P%614 = tail call double @llvm.fmuladd.f64(double %404, double %613, double %576)
,double8B

	full_text

double %404
,double8B

	full_text

double %613
,double8B

	full_text

double %576
Pstore8BE
C
	full_text6
4
2store double %614, double* %538, align 8, !tbaa !8
,double8B

	full_text

double %614
.double*8B

	full_text

double* %538
5add8B,
*
	full_text

%615 = add nsw i32 %5, -3
6icmp8B,
*
	full_text

%616 = icmp sgt i32 %5, 6
Abitcast8B4
2
	full_text%
#
!%617 = bitcast double %580 to i64
,double8B

	full_text

double %580
Abitcast8B4
2
	full_text%
#
!%618 = bitcast double %552 to i64
,double8B

	full_text

double %552
Abitcast8B4
2
	full_text%
#
!%619 = bitcast double %583 to i64
,double8B

	full_text

double %583
Abitcast8B4
2
	full_text%
#
!%620 = bitcast double %559 to i64
,double8B

	full_text

double %559
Abitcast8B4
2
	full_text%
#
!%621 = bitcast double %591 to i64
,double8B

	full_text

double %591
Abitcast8B4
2
	full_text%
#
!%622 = bitcast double %596 to i64
,double8B

	full_text

double %596
Abitcast8B4
2
	full_text%
#
!%623 = bitcast double %566 to i64
,double8B

	full_text

double %566
Abitcast8B4
2
	full_text%
#
!%624 = bitcast double %599 to i64
,double8B

	full_text

double %599
Abitcast8B4
2
	full_text%
#
!%625 = bitcast double %573 to i64
,double8B

	full_text

double %573
Abitcast8B4
2
	full_text%
#
!%626 = bitcast double %607 to i64
,double8B

	full_text

double %607
=br8B5
3
	full_text&
$
"br i1 %616, label %627, label %838
$i18B

	full_text
	
i1 %616
Ñgetelementptr8Bq
o
	full_textb
`
^%628 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Jload8B@
>
	full_text1
/
-%629 = load i64, i64* %207, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %207
rgetelementptr8B_
]
	full_textP
N
L%630 = getelementptr inbounds [5 x double], [5 x double]* %111, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
8zext8B.
,
	full_text

%631 = zext i32 %615 to i64
&i328B

	full_text


i32 %615
Jload8B@
>
	full_text1
/
-%632 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
Jload8B@
>
	full_text1
/
-%633 = load i64, i64* %217, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %217
Jload8B@
>
	full_text1
/
-%634 = load i64, i64* %234, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %234
Jload8B@
>
	full_text1
/
-%635 = load i64, i64* %251, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %251
Jload8B@
>
	full_text1
/
-%636 = load i64, i64* %269, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %269
(br8B 

	full_text

br label %637
Iphi8B@
>
	full_text1
/
-%638 = phi i64 [ %636, %627 ], [ %673, %637 ]
&i648B

	full_text


i64 %636
&i648B

	full_text


i64 %673
Iphi8B@
>
	full_text1
/
-%639 = phi i64 [ %635, %627 ], [ %671, %637 ]
&i648B

	full_text


i64 %635
&i648B

	full_text


i64 %671
Iphi8B@
>
	full_text1
/
-%640 = phi i64 [ %634, %627 ], [ %669, %637 ]
&i648B

	full_text


i64 %634
&i648B

	full_text


i64 %669
Iphi8B@
>
	full_text1
/
-%641 = phi i64 [ %633, %627 ], [ %667, %637 ]
&i648B

	full_text


i64 %633
&i648B

	full_text


i64 %667
Iphi8B@
>
	full_text1
/
-%642 = phi i64 [ %632, %627 ], [ %666, %637 ]
&i648B

	full_text


i64 %632
&i648B

	full_text


i64 %666
Lphi8BC
A
	full_text4
2
0%643 = phi double [ %612, %627 ], [ %822, %637 ]
,double8B

	full_text

double %612
,double8B

	full_text

double %822
Lphi8BC
A
	full_text4
2
0%644 = phi double [ %610, %627 ], [ %643, %637 ]
,double8B

	full_text

double %610
,double8B

	full_text

double %643
Iphi8B@
>
	full_text1
/
-%645 = phi i64 [ %626, %627 ], [ %835, %637 ]
&i648B

	full_text


i64 %626
&i648B

	full_text


i64 %835
Iphi8B@
>
	full_text1
/
-%646 = phi i64 [ %625, %627 ], [ %834, %637 ]
&i648B

	full_text


i64 %625
&i648B

	full_text


i64 %834
Lphi8BC
A
	full_text4
2
0%647 = phi double [ %604, %627 ], [ %815, %637 ]
,double8B

	full_text

double %604
,double8B

	full_text

double %815
Lphi8BC
A
	full_text4
2
0%648 = phi double [ %602, %627 ], [ %647, %637 ]
,double8B

	full_text

double %602
,double8B

	full_text

double %647
Iphi8B@
>
	full_text1
/
-%649 = phi i64 [ %624, %627 ], [ %833, %637 ]
&i648B

	full_text


i64 %624
&i648B

	full_text


i64 %833
Iphi8B@
>
	full_text1
/
-%650 = phi i64 [ %623, %627 ], [ %832, %637 ]
&i648B

	full_text


i64 %623
&i648B

	full_text


i64 %832
Lphi8BC
A
	full_text4
2
0%651 = phi double [ %596, %627 ], [ %808, %637 ]
,double8B

	full_text

double %596
,double8B

	full_text

double %808
Iphi8B@
>
	full_text1
/
-%652 = phi i64 [ %622, %627 ], [ %831, %637 ]
&i648B

	full_text


i64 %622
&i648B

	full_text


i64 %831
Lphi8BC
A
	full_text4
2
0%653 = phi double [ %594, %627 ], [ %806, %637 ]
,double8B

	full_text

double %594
,double8B

	full_text

double %806
Iphi8B@
>
	full_text1
/
-%654 = phi i64 [ %621, %627 ], [ %830, %637 ]
&i648B

	full_text


i64 %621
&i648B

	full_text


i64 %830
Iphi8B@
>
	full_text1
/
-%655 = phi i64 [ %620, %627 ], [ %829, %637 ]
&i648B

	full_text


i64 %620
&i648B

	full_text


i64 %829
Lphi8BC
A
	full_text4
2
0%656 = phi double [ %588, %627 ], [ %800, %637 ]
,double8B

	full_text

double %588
,double8B

	full_text

double %800
Lphi8BC
A
	full_text4
2
0%657 = phi double [ %586, %627 ], [ %656, %637 ]
,double8B

	full_text

double %586
,double8B

	full_text

double %656
Iphi8B@
>
	full_text1
/
-%658 = phi i64 [ %619, %627 ], [ %828, %637 ]
&i648B

	full_text


i64 %619
&i648B

	full_text


i64 %828
Iphi8B@
>
	full_text1
/
-%659 = phi i64 [ %618, %627 ], [ %827, %637 ]
&i648B

	full_text


i64 %618
&i648B

	full_text


i64 %827
Iphi8B@
>
	full_text1
/
-%660 = phi i64 [ %617, %627 ], [ %826, %637 ]
&i648B

	full_text


i64 %617
&i648B

	full_text


i64 %826
Lphi8BC
A
	full_text4
2
0%661 = phi double [ %546, %627 ], [ %759, %637 ]
,double8B

	full_text

double %546
,double8B

	full_text

double %759
Lphi8BC
A
	full_text4
2
0%662 = phi double [ %544, %627 ], [ %661, %637 ]
,double8B

	full_text

double %544
,double8B

	full_text

double %661
Lphi8BC
A
	full_text4
2
0%663 = phi double [ %543, %627 ], [ %662, %637 ]
,double8B

	full_text

double %543
,double8B

	full_text

double %662
Fphi8B=
;
	full_text.
,
*%664 = phi i64 [ 3, %627 ], [ %665, %637 ]
&i648B

	full_text


i64 %665
:add8B1
/
	full_text"
 
%665 = add nuw nsw i64 %664, 1
&i648B

	full_text


i64 %664
Istore8B>
<
	full_text/
-
+store i64 %660, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %660
'i64*8B

	full_text


i64* %83
Kstore8B@
>
	full_text1
/
-store i64 %642, i64* %202, align 16, !tbaa !8
&i648B

	full_text


i64 %642
(i64*8B

	full_text

	i64* %202
Kload8BA
?
	full_text2
0
.%666 = load i64, i64* %204, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %204
Jstore8B?
=
	full_text0
.
,store i64 %659, i64* %214, align 8, !tbaa !8
&i648B

	full_text


i64 %659
(i64*8B

	full_text

	i64* %214
Jstore8B?
=
	full_text0
.
,store i64 %658, i64* %211, align 8, !tbaa !8
&i648B

	full_text


i64 %658
(i64*8B

	full_text

	i64* %211
Jstore8B?
=
	full_text0
.
,store i64 %641, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %641
(i64*8B

	full_text

	i64* %220
Jload8B@
>
	full_text1
/
-%667 = load i64, i64* %221, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %221
Jload8B@
>
	full_text1
/
-%668 = load i64, i64* %223, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %223
Jstore8B?
=
	full_text0
.
,store i64 %668, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %668
(i64*8B

	full_text

	i64* %226
Kstore8B@
>
	full_text1
/
-store i64 %655, i64* %231, align 16, !tbaa !8
&i648B

	full_text


i64 %655
(i64*8B

	full_text

	i64* %231
Jstore8B?
=
	full_text0
.
,store i64 %654, i64* %228, align 8, !tbaa !8
&i648B

	full_text


i64 %654
(i64*8B

	full_text

	i64* %228
Istore8B>
<
	full_text/
-
+store i64 %652, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %652
'i64*8B

	full_text


i64* %47
Kstore8B@
>
	full_text1
/
-store i64 %640, i64* %237, align 16, !tbaa !8
&i648B

	full_text


i64 %640
(i64*8B

	full_text

	i64* %237
Kload8BA
?
	full_text2
0
.%669 = load i64, i64* %238, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %238
Jload8B@
>
	full_text1
/
-%670 = load i64, i64* %240, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %240
Kstore8B@
>
	full_text1
/
-store i64 %670, i64* %243, align 16, !tbaa !8
&i648B

	full_text


i64 %670
(i64*8B

	full_text

	i64* %243
Jstore8B?
=
	full_text0
.
,store i64 %650, i64* %248, align 8, !tbaa !8
&i648B

	full_text


i64 %650
(i64*8B

	full_text

	i64* %248
Jstore8B?
=
	full_text0
.
,store i64 %649, i64* %245, align 8, !tbaa !8
&i648B

	full_text


i64 %649
(i64*8B

	full_text

	i64* %245
Jstore8B?
=
	full_text0
.
,store i64 %639, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %639
(i64*8B

	full_text

	i64* %254
Jload8B@
>
	full_text1
/
-%671 = load i64, i64* %255, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %255
Jload8B@
>
	full_text1
/
-%672 = load i64, i64* %257, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %257
Jstore8B?
=
	full_text0
.
,store i64 %672, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %672
(i64*8B

	full_text

	i64* %260
Kstore8B@
>
	full_text1
/
-store i64 %646, i64* %265, align 16, !tbaa !8
&i648B

	full_text


i64 %646
(i64*8B

	full_text

	i64* %265
Jstore8B?
=
	full_text0
.
,store i64 %645, i64* %262, align 8, !tbaa !8
&i648B

	full_text


i64 %645
(i64*8B

	full_text

	i64* %262
Kstore8B@
>
	full_text1
/
-store i64 %638, i64* %272, align 16, !tbaa !8
&i648B

	full_text


i64 %638
(i64*8B

	full_text

	i64* %272
Kload8BA
?
	full_text2
0
.%673 = load i64, i64* %273, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %273
Jload8B@
>
	full_text1
/
-%674 = load i64, i64* %275, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %275
Kstore8B@
>
	full_text1
/
-store i64 %674, i64* %278, align 16, !tbaa !8
&i648B

	full_text


i64 %674
(i64*8B

	full_text

	i64* %278
:add8B1
/
	full_text"
 
%675 = add nuw nsw i64 %664, 2
&i648B

	full_text


i64 %664
ùgetelementptr8Bâ
Ü
	full_texty
w
u%676 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %675, i64 %32
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %675
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%677 = bitcast [5 x double]* %676 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %676
Jload8B@
>
	full_text1
/
-%678 = load i64, i64* %677, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %677
Kstore8B@
>
	full_text1
/
-store i64 %678, i64* %168, align 16, !tbaa !8
&i648B

	full_text


i64 %678
(i64*8B

	full_text

	i64* %168
•getelementptr8Bë
é
	full_textÄ
~
|%679 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %675, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %675
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%680 = bitcast double* %679 to i64*
.double*8B

	full_text

double* %679
Jload8B@
>
	full_text1
/
-%681 = load i64, i64* %680, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %680
Jstore8B?
=
	full_text0
.
,store i64 %681, i64* %173, align 8, !tbaa !8
&i648B

	full_text


i64 %681
(i64*8B

	full_text

	i64* %173
•getelementptr8Bë
é
	full_textÄ
~
|%682 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %675, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %675
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%683 = bitcast double* %682 to i64*
.double*8B

	full_text

double* %682
Jload8B@
>
	full_text1
/
-%684 = load i64, i64* %683, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %683
Kstore8B@
>
	full_text1
/
-store i64 %684, i64* %178, align 16, !tbaa !8
&i648B

	full_text


i64 %684
(i64*8B

	full_text

	i64* %178
•getelementptr8Bë
é
	full_textÄ
~
|%685 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %675, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %675
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%686 = bitcast double* %685 to i64*
.double*8B

	full_text

double* %685
Jload8B@
>
	full_text1
/
-%687 = load i64, i64* %686, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %686
Jstore8B?
=
	full_text0
.
,store i64 %687, i64* %183, align 8, !tbaa !8
&i648B

	full_text


i64 %687
(i64*8B

	full_text

	i64* %183
•getelementptr8Bë
é
	full_textÄ
~
|%688 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %675, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %675
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%689 = bitcast double* %688 to i64*
.double*8B

	full_text

double* %688
Jload8B@
>
	full_text1
/
-%690 = load i64, i64* %689, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %689
Kstore8B@
>
	full_text1
/
-store i64 %690, i64* %188, align 16, !tbaa !8
&i648B

	full_text


i64 %690
(i64*8B

	full_text

	i64* %188
Qstore8BF
D
	full_text7
5
3store double %651, double* %630, align 16, !tbaa !8
,double8B

	full_text

double %651
.double*8B

	full_text

double* %630
ègetelementptr8B|
z
	full_textm
k
i%691 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 %665, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %665
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%692 = load double, double* %691, align 8, !tbaa !8
.double*8B

	full_text

double* %691
:fmul8B0
.
	full_text!

%693 = fmul double %692, %651
,double8B

	full_text

double %692
,double8B

	full_text

double %651
ègetelementptr8B|
z
	full_textm
k
i%694 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 %665, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %665
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%695 = load double, double* %694, align 8, !tbaa !8
.double*8B

	full_text

double* %694
:fmul8B0
.
	full_text!

%696 = fmul double %693, %656
,double8B

	full_text

double %693
,double8B

	full_text

double %656
Pstore8BE
C
	full_text6
4
2store double %696, double* %121, align 8, !tbaa !8
,double8B

	full_text

double %696
.double*8B

	full_text

double* %121
:fsub8B0
.
	full_text!

%697 = fsub double %643, %695
,double8B

	full_text

double %643
,double8B

	full_text

double %695
Bfmul8B8
6
	full_text)
'
%%698 = fmul double %697, 4.000000e-01
,double8B

	full_text

double %697
mcall8Bc
a
	full_textT
R
P%699 = tail call double @llvm.fmuladd.f64(double %651, double %693, double %698)
,double8B

	full_text

double %651
,double8B

	full_text

double %693
,double8B

	full_text

double %698
Qstore8BF
D
	full_text7
5
3store double %699, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %699
.double*8B

	full_text

double* %126
:fmul8B0
.
	full_text!

%700 = fmul double %693, %647
,double8B

	full_text

double %693
,double8B

	full_text

double %647
Pstore8BE
C
	full_text6
4
2store double %700, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %700
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%701 = fmul double %695, 4.000000e-01
,double8B

	full_text

double %695
Cfsub8B9
7
	full_text*
(
&%702 = fsub double -0.000000e+00, %701
,double8B

	full_text

double %701
ucall8Bk
i
	full_text\
Z
X%703 = tail call double @llvm.fmuladd.f64(double %643, double 1.400000e+00, double %702)
,double8B

	full_text

double %643
,double8B

	full_text

double %702
:fmul8B0
.
	full_text!

%704 = fmul double %693, %703
,double8B

	full_text

double %693
,double8B

	full_text

double %703
Qstore8BF
D
	full_text7
5
3store double %704, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %704
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%705 = fmul double %692, %656
,double8B

	full_text

double %692
,double8B

	full_text

double %656
:fmul8B0
.
	full_text!

%706 = fmul double %692, %647
,double8B

	full_text

double %692
,double8B

	full_text

double %647
:fmul8B0
.
	full_text!

%707 = fmul double %692, %643
,double8B

	full_text

double %692
,double8B

	full_text

double %643
ègetelementptr8B|
z
	full_textm
k
i%708 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 %664, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %664
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%709 = load double, double* %708, align 8, !tbaa !8
.double*8B

	full_text

double* %708
:fmul8B0
.
	full_text!

%710 = fmul double %709, %657
,double8B

	full_text

double %709
,double8B

	full_text

double %657
:fmul8B0
.
	full_text!

%711 = fmul double %709, %653
,double8B

	full_text

double %709
,double8B

	full_text

double %653
:fmul8B0
.
	full_text!

%712 = fmul double %709, %648
,double8B

	full_text

double %709
,double8B

	full_text

double %648
:fmul8B0
.
	full_text!

%713 = fmul double %709, %644
,double8B

	full_text

double %709
,double8B

	full_text

double %644
:fsub8B0
.
	full_text!

%714 = fsub double %705, %710
,double8B

	full_text

double %705
,double8B

	full_text

double %710
Bfmul8B8
6
	full_text)
'
%%715 = fmul double %714, 6.300000e+01
,double8B

	full_text

double %714
Pstore8BE
C
	full_text6
4
2store double %715, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %715
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%716 = fsub double %693, %711
,double8B

	full_text

double %693
,double8B

	full_text

double %711
Bfmul8B8
6
	full_text)
'
%%717 = fmul double %716, 8.400000e+01
,double8B

	full_text

double %716
Pstore8BE
C
	full_text6
4
2store double %717, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %717
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%718 = fsub double %706, %712
,double8B

	full_text

double %706
,double8B

	full_text

double %712
Bfmul8B8
6
	full_text)
'
%%719 = fmul double %718, 6.300000e+01
,double8B

	full_text

double %718
Pstore8BE
C
	full_text6
4
2store double %719, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %719
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%720 = fmul double %693, %693
,double8B

	full_text

double %693
,double8B

	full_text

double %693
mcall8Bc
a
	full_textT
R
P%721 = tail call double @llvm.fmuladd.f64(double %705, double %705, double %720)
,double8B

	full_text

double %705
,double8B

	full_text

double %705
,double8B

	full_text

double %720
mcall8Bc
a
	full_textT
R
P%722 = tail call double @llvm.fmuladd.f64(double %706, double %706, double %721)
,double8B

	full_text

double %706
,double8B

	full_text

double %706
,double8B

	full_text

double %721
:fmul8B0
.
	full_text!

%723 = fmul double %711, %711
,double8B

	full_text

double %711
,double8B

	full_text

double %711
mcall8Bc
a
	full_textT
R
P%724 = tail call double @llvm.fmuladd.f64(double %710, double %710, double %723)
,double8B

	full_text

double %710
,double8B

	full_text

double %710
,double8B

	full_text

double %723
mcall8Bc
a
	full_textT
R
P%725 = tail call double @llvm.fmuladd.f64(double %712, double %712, double %724)
,double8B

	full_text

double %712
,double8B

	full_text

double %712
,double8B

	full_text

double %724
:fsub8B0
.
	full_text!

%726 = fsub double %722, %725
,double8B

	full_text

double %722
,double8B

	full_text

double %725
Cfsub8B9
7
	full_text*
(
&%727 = fsub double -0.000000e+00, %723
,double8B

	full_text

double %723
mcall8Bc
a
	full_textT
R
P%728 = tail call double @llvm.fmuladd.f64(double %693, double %693, double %727)
,double8B

	full_text

double %693
,double8B

	full_text

double %693
,double8B

	full_text

double %727
Bfmul8B8
6
	full_text)
'
%%729 = fmul double %728, 1.050000e+01
,double8B

	full_text

double %728
{call8Bq
o
	full_textb
`
^%730 = tail call double @llvm.fmuladd.f64(double %726, double 0xC03E3D70A3D70A3B, double %729)
,double8B

	full_text

double %726
,double8B

	full_text

double %729
:fsub8B0
.
	full_text!

%731 = fsub double %707, %713
,double8B

	full_text

double %707
,double8B

	full_text

double %713
{call8Bq
o
	full_textb
`
^%732 = tail call double @llvm.fmuladd.f64(double %731, double 0x405EDEB851EB851E, double %730)
,double8B

	full_text

double %731
,double8B

	full_text

double %730
Pstore8BE
C
	full_text6
4
2store double %732, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %732
.double*8B

	full_text

double* %163
•getelementptr8Bë
é
	full_textÄ
~
|%733 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %664, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %664
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%734 = load double, double* %733, align 8, !tbaa !8
.double*8B

	full_text

double* %733
Qload8BG
E
	full_text8
6
4%735 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
:fsub8B0
.
	full_text!

%736 = fsub double %651, %735
,double8B

	full_text

double %651
,double8B

	full_text

double %735
vcall8Bl
j
	full_text]
[
Y%737 = tail call double @llvm.fmuladd.f64(double %736, double -3.150000e+01, double %734)
,double8B

	full_text

double %736
,double8B

	full_text

double %734
•getelementptr8Bë
é
	full_textÄ
~
|%738 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %664, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %664
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%739 = load double, double* %738, align 8, !tbaa !8
.double*8B

	full_text

double* %738
Pload8BF
D
	full_text7
5
3%740 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
:fsub8B0
.
	full_text!

%741 = fsub double %696, %740
,double8B

	full_text

double %696
,double8B

	full_text

double %740
vcall8Bl
j
	full_text]
[
Y%742 = tail call double @llvm.fmuladd.f64(double %741, double -3.150000e+01, double %739)
,double8B

	full_text

double %741
,double8B

	full_text

double %739
•getelementptr8Bë
é
	full_textÄ
~
|%743 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %664, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %664
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%744 = load double, double* %743, align 8, !tbaa !8
.double*8B

	full_text

double* %743
Qload8BG
E
	full_text8
6
4%745 = load double, double* %236, align 16, !tbaa !8
.double*8B

	full_text

double* %236
:fsub8B0
.
	full_text!

%746 = fsub double %699, %745
,double8B

	full_text

double %699
,double8B

	full_text

double %745
vcall8Bl
j
	full_text]
[
Y%747 = tail call double @llvm.fmuladd.f64(double %746, double -3.150000e+01, double %744)
,double8B

	full_text

double %746
,double8B

	full_text

double %744
•getelementptr8Bë
é
	full_textÄ
~
|%748 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %664, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %664
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%749 = load double, double* %748, align 8, !tbaa !8
.double*8B

	full_text

double* %748
Pload8BF
D
	full_text7
5
3%750 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
:fsub8B0
.
	full_text!

%751 = fsub double %700, %750
,double8B

	full_text

double %700
,double8B

	full_text

double %750
vcall8Bl
j
	full_text]
[
Y%752 = tail call double @llvm.fmuladd.f64(double %751, double -3.150000e+01, double %749)
,double8B

	full_text

double %751
,double8B

	full_text

double %749
•getelementptr8Bë
é
	full_textÄ
~
|%753 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %664, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %664
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%754 = load double, double* %753, align 8, !tbaa !8
.double*8B

	full_text

double* %753
Qload8BG
E
	full_text8
6
4%755 = load double, double* %271, align 16, !tbaa !8
.double*8B

	full_text

double* %271
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
Y%757 = tail call double @llvm.fmuladd.f64(double %756, double -3.150000e+01, double %754)
,double8B

	full_text

double %756
,double8B

	full_text

double %754
vcall8Bl
j
	full_text]
[
Y%758 = tail call double @llvm.fmuladd.f64(double %661, double -2.000000e+00, double %662)
,double8B

	full_text

double %661
,double8B

	full_text

double %662
Oload8BE
C
	full_text6
4
2%759 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%760 = fadd double %758, %759
,double8B

	full_text

double %758
,double8B

	full_text

double %759
{call8Bq
o
	full_textb
`
^%761 = tail call double @llvm.fmuladd.f64(double %760, double 0x40A7418000000001, double %737)
,double8B

	full_text

double %760
,double8B

	full_text

double %737
Pload8BF
D
	full_text7
5
3%762 = load double, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
:fsub8B0
.
	full_text!

%763 = fsub double %715, %762
,double8B

	full_text

double %715
,double8B

	full_text

double %762
{call8Bq
o
	full_textb
`
^%764 = tail call double @llvm.fmuladd.f64(double %763, double 0x4019333333333334, double %742)
,double8B

	full_text

double %763
,double8B

	full_text

double %742
Pload8BF
D
	full_text7
5
3%765 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
vcall8Bl
j
	full_text]
[
Y%766 = tail call double @llvm.fmuladd.f64(double %657, double -2.000000e+00, double %765)
,double8B

	full_text

double %657
,double8B

	full_text

double %765
:fadd8B0
.
	full_text!

%767 = fadd double %656, %766
,double8B

	full_text

double %656
,double8B

	full_text

double %766
{call8Bq
o
	full_textb
`
^%768 = tail call double @llvm.fmuladd.f64(double %767, double 0x40A7418000000001, double %764)
,double8B

	full_text

double %767
,double8B

	full_text

double %764
Qload8BG
E
	full_text8
6
4%769 = load double, double* %242, align 16, !tbaa !8
.double*8B

	full_text

double* %242
:fsub8B0
.
	full_text!

%770 = fsub double %717, %769
,double8B

	full_text

double %717
,double8B

	full_text

double %769
{call8Bq
o
	full_textb
`
^%771 = tail call double @llvm.fmuladd.f64(double %770, double 0x4019333333333334, double %747)
,double8B

	full_text

double %770
,double8B

	full_text

double %747
Pload8BF
D
	full_text7
5
3%772 = load double, double* %227, align 8, !tbaa !8
.double*8B

	full_text

double* %227
vcall8Bl
j
	full_text]
[
Y%773 = tail call double @llvm.fmuladd.f64(double %653, double -2.000000e+00, double %772)
,double8B

	full_text

double %653
,double8B

	full_text

double %772
:fadd8B0
.
	full_text!

%774 = fadd double %651, %773
,double8B

	full_text

double %651
,double8B

	full_text

double %773
{call8Bq
o
	full_textb
`
^%775 = tail call double @llvm.fmuladd.f64(double %774, double 0x40A7418000000001, double %771)
,double8B

	full_text

double %774
,double8B

	full_text

double %771
Pload8BF
D
	full_text7
5
3%776 = load double, double* %259, align 8, !tbaa !8
.double*8B

	full_text

double* %259
:fsub8B0
.
	full_text!

%777 = fsub double %719, %776
,double8B

	full_text

double %719
,double8B

	full_text

double %776
{call8Bq
o
	full_textb
`
^%778 = tail call double @llvm.fmuladd.f64(double %777, double 0x4019333333333334, double %752)
,double8B

	full_text

double %777
,double8B

	full_text

double %752
Pload8BF
D
	full_text7
5
3%779 = load double, double* %244, align 8, !tbaa !8
.double*8B

	full_text

double* %244
vcall8Bl
j
	full_text]
[
Y%780 = tail call double @llvm.fmuladd.f64(double %648, double -2.000000e+00, double %779)
,double8B

	full_text

double %648
,double8B

	full_text

double %779
:fadd8B0
.
	full_text!

%781 = fadd double %647, %780
,double8B

	full_text

double %647
,double8B

	full_text

double %780
{call8Bq
o
	full_textb
`
^%782 = tail call double @llvm.fmuladd.f64(double %781, double 0x40A7418000000001, double %778)
,double8B

	full_text

double %781
,double8B

	full_text

double %778
Qload8BG
E
	full_text8
6
4%783 = load double, double* %277, align 16, !tbaa !8
.double*8B

	full_text

double* %277
:fsub8B0
.
	full_text!

%784 = fsub double %732, %783
,double8B

	full_text

double %732
,double8B

	full_text

double %783
{call8Bq
o
	full_textb
`
^%785 = tail call double @llvm.fmuladd.f64(double %784, double 0x4019333333333334, double %757)
,double8B

	full_text

double %784
,double8B

	full_text

double %757
Pload8BF
D
	full_text7
5
3%786 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
vcall8Bl
j
	full_text]
[
Y%787 = tail call double @llvm.fmuladd.f64(double %644, double -2.000000e+00, double %786)
,double8B

	full_text

double %644
,double8B

	full_text

double %786
:fadd8B0
.
	full_text!

%788 = fadd double %643, %787
,double8B

	full_text

double %643
,double8B

	full_text

double %787
{call8Bq
o
	full_textb
`
^%789 = tail call double @llvm.fmuladd.f64(double %788, double 0x40A7418000000001, double %785)
,double8B

	full_text

double %788
,double8B

	full_text

double %785
vcall8Bl
j
	full_text]
[
Y%790 = tail call double @llvm.fmuladd.f64(double %662, double -4.000000e+00, double %663)
,double8B

	full_text

double %662
,double8B

	full_text

double %663
ucall8Bk
i
	full_text\
Z
X%791 = tail call double @llvm.fmuladd.f64(double %661, double 6.000000e+00, double %790)
,double8B

	full_text

double %661
,double8B

	full_text

double %790
vcall8Bl
j
	full_text]
[
Y%792 = tail call double @llvm.fmuladd.f64(double %759, double -4.000000e+00, double %791)
,double8B

	full_text

double %759
,double8B

	full_text

double %791
Qload8BG
E
	full_text8
6
4%793 = load double, double* %196, align 16, !tbaa !8
.double*8B

	full_text

double* %196
:fadd8B0
.
	full_text!

%794 = fadd double %792, %793
,double8B

	full_text

double %792
,double8B

	full_text

double %793
mcall8Bc
a
	full_textT
R
P%795 = tail call double @llvm.fmuladd.f64(double %404, double %794, double %761)
,double8B

	full_text

double %404
,double8B

	full_text

double %794
,double8B

	full_text

double %761
Pstore8BE
C
	full_text6
4
2store double %795, double* %733, align 8, !tbaa !8
,double8B

	full_text

double %795
.double*8B

	full_text

double* %733
Pload8BF
D
	full_text7
5
3%796 = load double, double* %213, align 8, !tbaa !8
.double*8B

	full_text

double* %213
vcall8Bl
j
	full_text]
[
Y%797 = tail call double @llvm.fmuladd.f64(double %765, double -4.000000e+00, double %796)
,double8B

	full_text

double %765
,double8B

	full_text

double %796
ucall8Bk
i
	full_text\
Z
X%798 = tail call double @llvm.fmuladd.f64(double %657, double 6.000000e+00, double %797)
,double8B

	full_text

double %657
,double8B

	full_text

double %797
vcall8Bl
j
	full_text]
[
Y%799 = tail call double @llvm.fmuladd.f64(double %656, double -4.000000e+00, double %798)
,double8B

	full_text

double %656
,double8B

	full_text

double %798
Pload8BF
D
	full_text7
5
3%800 = load double, double* %172, align 8, !tbaa !8
.double*8B

	full_text

double* %172
:fadd8B0
.
	full_text!

%801 = fadd double %799, %800
,double8B

	full_text

double %799
,double8B

	full_text

double %800
mcall8Bc
a
	full_textT
R
P%802 = tail call double @llvm.fmuladd.f64(double %404, double %801, double %768)
,double8B

	full_text

double %404
,double8B

	full_text

double %801
,double8B

	full_text

double %768
Pstore8BE
C
	full_text6
4
2store double %802, double* %738, align 8, !tbaa !8
,double8B

	full_text

double %802
.double*8B

	full_text

double* %738
Qload8BG
E
	full_text8
6
4%803 = load double, double* %230, align 16, !tbaa !8
.double*8B

	full_text

double* %230
vcall8Bl
j
	full_text]
[
Y%804 = tail call double @llvm.fmuladd.f64(double %772, double -4.000000e+00, double %803)
,double8B

	full_text

double %772
,double8B

	full_text

double %803
ucall8Bk
i
	full_text\
Z
X%805 = tail call double @llvm.fmuladd.f64(double %653, double 6.000000e+00, double %804)
,double8B

	full_text

double %653
,double8B

	full_text

double %804
Oload8BE
C
	full_text6
4
2%806 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
vcall8Bl
j
	full_text]
[
Y%807 = tail call double @llvm.fmuladd.f64(double %806, double -4.000000e+00, double %805)
,double8B

	full_text

double %806
,double8B

	full_text

double %805
Qload8BG
E
	full_text8
6
4%808 = load double, double* %177, align 16, !tbaa !8
.double*8B

	full_text

double* %177
:fadd8B0
.
	full_text!

%809 = fadd double %807, %808
,double8B

	full_text

double %807
,double8B

	full_text

double %808
mcall8Bc
a
	full_textT
R
P%810 = tail call double @llvm.fmuladd.f64(double %404, double %809, double %775)
,double8B

	full_text

double %404
,double8B

	full_text

double %809
,double8B

	full_text

double %775
Pstore8BE
C
	full_text6
4
2store double %810, double* %743, align 8, !tbaa !8
,double8B

	full_text

double %810
.double*8B

	full_text

double* %743
Pload8BF
D
	full_text7
5
3%811 = load double, double* %247, align 8, !tbaa !8
.double*8B

	full_text

double* %247
vcall8Bl
j
	full_text]
[
Y%812 = tail call double @llvm.fmuladd.f64(double %779, double -4.000000e+00, double %811)
,double8B

	full_text

double %779
,double8B

	full_text

double %811
ucall8Bk
i
	full_text\
Z
X%813 = tail call double @llvm.fmuladd.f64(double %648, double 6.000000e+00, double %812)
,double8B

	full_text

double %648
,double8B

	full_text

double %812
vcall8Bl
j
	full_text]
[
Y%814 = tail call double @llvm.fmuladd.f64(double %647, double -4.000000e+00, double %813)
,double8B

	full_text

double %647
,double8B

	full_text

double %813
Pload8BF
D
	full_text7
5
3%815 = load double, double* %182, align 8, !tbaa !8
.double*8B

	full_text

double* %182
:fadd8B0
.
	full_text!

%816 = fadd double %814, %815
,double8B

	full_text

double %814
,double8B

	full_text

double %815
mcall8Bc
a
	full_textT
R
P%817 = tail call double @llvm.fmuladd.f64(double %404, double %816, double %782)
,double8B

	full_text

double %404
,double8B

	full_text

double %816
,double8B

	full_text

double %782
Pstore8BE
C
	full_text6
4
2store double %817, double* %748, align 8, !tbaa !8
,double8B

	full_text

double %817
.double*8B

	full_text

double* %748
Qload8BG
E
	full_text8
6
4%818 = load double, double* %264, align 16, !tbaa !8
.double*8B

	full_text

double* %264
vcall8Bl
j
	full_text]
[
Y%819 = tail call double @llvm.fmuladd.f64(double %786, double -4.000000e+00, double %818)
,double8B

	full_text

double %786
,double8B

	full_text

double %818
ucall8Bk
i
	full_text\
Z
X%820 = tail call double @llvm.fmuladd.f64(double %644, double 6.000000e+00, double %819)
,double8B

	full_text

double %644
,double8B

	full_text

double %819
vcall8Bl
j
	full_text]
[
Y%821 = tail call double @llvm.fmuladd.f64(double %643, double -4.000000e+00, double %820)
,double8B

	full_text

double %643
,double8B

	full_text

double %820
Qload8BG
E
	full_text8
6
4%822 = load double, double* %187, align 16, !tbaa !8
.double*8B

	full_text

double* %187
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
P%824 = tail call double @llvm.fmuladd.f64(double %404, double %823, double %789)
,double8B

	full_text

double %404
,double8B

	full_text

double %823
,double8B

	full_text

double %789
Pstore8BE
C
	full_text6
4
2store double %824, double* %753, align 8, !tbaa !8
,double8B

	full_text

double %824
.double*8B

	full_text

double* %753
:icmp8B0
.
	full_text!

%825 = icmp eq i64 %665, %631
&i648B

	full_text


i64 %665
&i648B

	full_text


i64 %631
Abitcast8B4
2
	full_text%
#
!%826 = bitcast double %793 to i64
,double8B

	full_text

double %793
Abitcast8B4
2
	full_text%
#
!%827 = bitcast double %765 to i64
,double8B

	full_text

double %765
Abitcast8B4
2
	full_text%
#
!%828 = bitcast double %657 to i64
,double8B

	full_text

double %657
Abitcast8B4
2
	full_text%
#
!%829 = bitcast double %772 to i64
,double8B

	full_text

double %772
Abitcast8B4
2
	full_text%
#
!%830 = bitcast double %653 to i64
,double8B

	full_text

double %653
Abitcast8B4
2
	full_text%
#
!%831 = bitcast double %808 to i64
,double8B

	full_text

double %808
Abitcast8B4
2
	full_text%
#
!%832 = bitcast double %779 to i64
,double8B

	full_text

double %779
Abitcast8B4
2
	full_text%
#
!%833 = bitcast double %648 to i64
,double8B

	full_text

double %648
Abitcast8B4
2
	full_text%
#
!%834 = bitcast double %786 to i64
,double8B

	full_text

double %786
Abitcast8B4
2
	full_text%
#
!%835 = bitcast double %644 to i64
,double8B

	full_text

double %644
=br8B5
3
	full_text&
$
"br i1 %825, label %836, label %637
$i18B

	full_text
	
i1 %825
Qstore8BF
D
	full_text7
5
3store double %663, double* %628, align 16, !tbaa !8
,double8B

	full_text

double %663
.double*8B

	full_text

double* %628
Pstore8BE
C
	full_text6
4
2store double %662, double* %189, align 8, !tbaa !8
,double8B

	full_text

double %662
.double*8B

	full_text

double* %189
Pstore8BE
C
	full_text6
4
2store double %661, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %661
-double*8B

	full_text

double* %84
Jstore8B?
=
	full_text0
.
,store i64 %666, i64* %199, align 8, !tbaa !8
&i648B

	full_text


i64 %666
(i64*8B

	full_text

	i64* %199
Kstore8B@
>
	full_text1
/
-store i64 %629, i64* %209, align 16, !tbaa !8
&i648B

	full_text


i64 %629
(i64*8B

	full_text

	i64* %209
Ostore8BD
B
	full_text5
3
1store double %657, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %657
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %656, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %656
-double*8B

	full_text

double* %41
Jstore8B?
=
	full_text0
.
,store i64 %667, i64* %217, align 8, !tbaa !8
&i648B

	full_text


i64 %667
(i64*8B

	full_text

	i64* %217
Pstore8BE
C
	full_text6
4
2store double %653, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %653
-double*8B

	full_text

double* %88
Jstore8B?
=
	full_text0
.
,store i64 %669, i64* %234, align 8, !tbaa !8
&i648B

	full_text


i64 %669
(i64*8B

	full_text

	i64* %234
Ostore8BD
B
	full_text5
3
1store double %648, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %648
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %647, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %647
-double*8B

	full_text

double* %51
Jstore8B?
=
	full_text0
.
,store i64 %671, i64* %251, align 8, !tbaa !8
&i648B

	full_text


i64 %671
(i64*8B

	full_text

	i64* %251
Pstore8BE
C
	full_text6
4
2store double %644, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %644
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %643, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %643
-double*8B

	full_text

double* %56
Jstore8B?
=
	full_text0
.
,store i64 %673, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %673
(i64*8B

	full_text

	i64* %269
Jload8B@
>
	full_text1
/
-%837 = load i64, i64* %211, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %211
(br8B 

	full_text

br label %838
Kphi8BB
@
	full_text3
1
/%839 = phi double [ %822, %836 ], [ %612, %25 ]
,double8B

	full_text

double %822
,double8B

	full_text

double %612
Kphi8BB
@
	full_text3
1
/%840 = phi double [ %643, %836 ], [ %610, %25 ]
,double8B

	full_text

double %643
,double8B

	full_text

double %610
Hphi8B?
=
	full_text0
.
,%841 = phi i64 [ %835, %836 ], [ %626, %25 ]
&i648B

	full_text


i64 %835
&i648B

	full_text


i64 %626
Hphi8B?
=
	full_text0
.
,%842 = phi i64 [ %834, %836 ], [ %625, %25 ]
&i648B

	full_text


i64 %834
&i648B

	full_text


i64 %625
Kphi8BB
@
	full_text3
1
/%843 = phi double [ %815, %836 ], [ %604, %25 ]
,double8B

	full_text

double %815
,double8B

	full_text

double %604
Kphi8BB
@
	full_text3
1
/%844 = phi double [ %647, %836 ], [ %602, %25 ]
,double8B

	full_text

double %647
,double8B

	full_text

double %602
Hphi8B?
=
	full_text0
.
,%845 = phi i64 [ %833, %836 ], [ %624, %25 ]
&i648B

	full_text


i64 %833
&i648B

	full_text


i64 %624
Hphi8B?
=
	full_text0
.
,%846 = phi i64 [ %832, %836 ], [ %623, %25 ]
&i648B

	full_text


i64 %832
&i648B

	full_text


i64 %623
Kphi8BB
@
	full_text3
1
/%847 = phi double [ %808, %836 ], [ %596, %25 ]
,double8B

	full_text

double %808
,double8B

	full_text

double %596
Hphi8B?
=
	full_text0
.
,%848 = phi i64 [ %831, %836 ], [ %622, %25 ]
&i648B

	full_text


i64 %831
&i648B

	full_text


i64 %622
Kphi8BB
@
	full_text3
1
/%849 = phi double [ %806, %836 ], [ %594, %25 ]
,double8B

	full_text

double %806
,double8B

	full_text

double %594
Hphi8B?
=
	full_text0
.
,%850 = phi i64 [ %830, %836 ], [ %621, %25 ]
&i648B

	full_text


i64 %830
&i648B

	full_text


i64 %621
Hphi8B?
=
	full_text0
.
,%851 = phi i64 [ %829, %836 ], [ %620, %25 ]
&i648B

	full_text


i64 %829
&i648B

	full_text


i64 %620
Kphi8BB
@
	full_text3
1
/%852 = phi double [ %800, %836 ], [ %588, %25 ]
,double8B

	full_text

double %800
,double8B

	full_text

double %588
Kphi8BB
@
	full_text3
1
/%853 = phi double [ %656, %836 ], [ %586, %25 ]
,double8B

	full_text

double %656
,double8B

	full_text

double %586
Hphi8B?
=
	full_text0
.
,%854 = phi i64 [ %828, %836 ], [ %619, %25 ]
&i648B

	full_text


i64 %828
&i648B

	full_text


i64 %619
Hphi8B?
=
	full_text0
.
,%855 = phi i64 [ %837, %836 ], [ %618, %25 ]
&i648B

	full_text


i64 %837
&i648B

	full_text


i64 %618
Hphi8B?
=
	full_text0
.
,%856 = phi i64 [ %826, %836 ], [ %617, %25 ]
&i648B

	full_text


i64 %826
&i648B

	full_text


i64 %617
5add8B,
*
	full_text

%857 = add nsw i32 %5, -2
Jload8B@
>
	full_text1
/
-%858 = load i64, i64* %190, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %190
Kstore8B@
>
	full_text1
/
-store i64 %858, i64* %193, align 16, !tbaa !8
&i648B

	full_text


i64 %858
(i64*8B

	full_text

	i64* %193
Jload8B@
>
	full_text1
/
-%859 = load i64, i64* %85, align 16, !tbaa !8
'i64*8B

	full_text


i64* %85
Jstore8B?
=
	full_text0
.
,store i64 %859, i64* %190, align 8, !tbaa !8
&i648B

	full_text


i64 %859
(i64*8B

	full_text

	i64* %190
Iload8B?
=
	full_text0
.
,%860 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
Jstore8B?
=
	full_text0
.
,store i64 %860, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %860
'i64*8B

	full_text


i64* %85
Istore8B>
<
	full_text/
-
+store i64 %856, i64* %83, align 8, !tbaa !8
&i648B

	full_text


i64 %856
'i64*8B

	full_text


i64* %83
Jload8B@
>
	full_text1
/
-%861 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
Kstore8B@
>
	full_text1
/
-store i64 %861, i64* %202, align 16, !tbaa !8
&i648B

	full_text


i64 %861
(i64*8B

	full_text

	i64* %202
Kload8BA
?
	full_text2
0
.%862 = load i64, i64* %204, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %204
Jstore8B?
=
	full_text0
.
,store i64 %862, i64* %199, align 8, !tbaa !8
&i648B

	full_text


i64 %862
(i64*8B

	full_text

	i64* %199
Jload8B@
>
	full_text1
/
-%863 = load i64, i64* %207, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %207
Kstore8B@
>
	full_text1
/
-store i64 %863, i64* %209, align 16, !tbaa !8
&i648B

	full_text


i64 %863
(i64*8B

	full_text

	i64* %209
Jstore8B?
=
	full_text0
.
,store i64 %855, i64* %214, align 8, !tbaa !8
&i648B

	full_text


i64 %855
(i64*8B

	full_text

	i64* %214
Jstore8B?
=
	full_text0
.
,store i64 %854, i64* %211, align 8, !tbaa !8
&i648B

	full_text


i64 %854
(i64*8B

	full_text

	i64* %211
Ostore8BD
B
	full_text5
3
1store double %853, double* %86, align 8, !tbaa !8
,double8B

	full_text

double %853
-double*8B

	full_text

double* %86
Ostore8BD
B
	full_text5
3
1store double %852, double* %41, align 8, !tbaa !8
,double8B

	full_text

double %852
-double*8B

	full_text

double* %41
Jload8B@
>
	full_text1
/
-%864 = load i64, i64* %217, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %217
Jstore8B?
=
	full_text0
.
,store i64 %864, i64* %220, align 8, !tbaa !8
&i648B

	full_text


i64 %864
(i64*8B

	full_text

	i64* %220
Jload8B@
>
	full_text1
/
-%865 = load i64, i64* %221, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %221
Jstore8B?
=
	full_text0
.
,store i64 %865, i64* %217, align 8, !tbaa !8
&i648B

	full_text


i64 %865
(i64*8B

	full_text

	i64* %217
Jload8B@
>
	full_text1
/
-%866 = load i64, i64* %223, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %223
Jstore8B?
=
	full_text0
.
,store i64 %866, i64* %226, align 8, !tbaa !8
&i648B

	full_text


i64 %866
(i64*8B

	full_text

	i64* %226
Kstore8B@
>
	full_text1
/
-store i64 %851, i64* %231, align 16, !tbaa !8
&i648B

	full_text


i64 %851
(i64*8B

	full_text

	i64* %231
Jstore8B?
=
	full_text0
.
,store i64 %850, i64* %228, align 8, !tbaa !8
&i648B

	full_text


i64 %850
(i64*8B

	full_text

	i64* %228
Pstore8BE
C
	full_text6
4
2store double %849, double* %88, align 16, !tbaa !8
,double8B

	full_text

double %849
-double*8B

	full_text

double* %88
Istore8B>
<
	full_text/
-
+store i64 %848, i64* %47, align 8, !tbaa !8
&i648B

	full_text


i64 %848
'i64*8B

	full_text


i64* %47
Jload8B@
>
	full_text1
/
-%867 = load i64, i64* %234, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %234
Kstore8B@
>
	full_text1
/
-store i64 %867, i64* %237, align 16, !tbaa !8
&i648B

	full_text


i64 %867
(i64*8B

	full_text

	i64* %237
Kload8BA
?
	full_text2
0
.%868 = load i64, i64* %238, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %238
Jstore8B?
=
	full_text0
.
,store i64 %868, i64* %234, align 8, !tbaa !8
&i648B

	full_text


i64 %868
(i64*8B

	full_text

	i64* %234
Jload8B@
>
	full_text1
/
-%869 = load i64, i64* %240, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %240
Kstore8B@
>
	full_text1
/
-store i64 %869, i64* %243, align 16, !tbaa !8
&i648B

	full_text


i64 %869
(i64*8B

	full_text

	i64* %243
Jstore8B?
=
	full_text0
.
,store i64 %846, i64* %248, align 8, !tbaa !8
&i648B

	full_text


i64 %846
(i64*8B

	full_text

	i64* %248
Jstore8B?
=
	full_text0
.
,store i64 %845, i64* %245, align 8, !tbaa !8
&i648B

	full_text


i64 %845
(i64*8B

	full_text

	i64* %245
Ostore8BD
B
	full_text5
3
1store double %844, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %844
-double*8B

	full_text

double* %90
Ostore8BD
B
	full_text5
3
1store double %843, double* %51, align 8, !tbaa !8
,double8B

	full_text

double %843
-double*8B

	full_text

double* %51
Jload8B@
>
	full_text1
/
-%870 = load i64, i64* %251, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %251
Jstore8B?
=
	full_text0
.
,store i64 %870, i64* %254, align 8, !tbaa !8
&i648B

	full_text


i64 %870
(i64*8B

	full_text

	i64* %254
Jload8B@
>
	full_text1
/
-%871 = load i64, i64* %255, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %255
Jstore8B?
=
	full_text0
.
,store i64 %871, i64* %251, align 8, !tbaa !8
&i648B

	full_text


i64 %871
(i64*8B

	full_text

	i64* %251
Jload8B@
>
	full_text1
/
-%872 = load i64, i64* %257, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %257
Jstore8B?
=
	full_text0
.
,store i64 %872, i64* %260, align 8, !tbaa !8
&i648B

	full_text


i64 %872
(i64*8B

	full_text

	i64* %260
Kstore8B@
>
	full_text1
/
-store i64 %842, i64* %265, align 16, !tbaa !8
&i648B

	full_text


i64 %842
(i64*8B

	full_text

	i64* %265
Jstore8B?
=
	full_text0
.
,store i64 %841, i64* %262, align 8, !tbaa !8
&i648B

	full_text


i64 %841
(i64*8B

	full_text

	i64* %262
Pstore8BE
C
	full_text6
4
2store double %840, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %840
-double*8B

	full_text

double* %92
Ostore8BD
B
	full_text5
3
1store double %839, double* %56, align 8, !tbaa !8
,double8B

	full_text

double %839
-double*8B

	full_text

double* %56
Jload8B@
>
	full_text1
/
-%873 = load i64, i64* %269, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %269
Kstore8B@
>
	full_text1
/
-store i64 %873, i64* %272, align 16, !tbaa !8
&i648B

	full_text


i64 %873
(i64*8B

	full_text

	i64* %272
Kload8BA
?
	full_text2
0
.%874 = load i64, i64* %273, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %273
Jstore8B?
=
	full_text0
.
,store i64 %874, i64* %269, align 8, !tbaa !8
&i648B

	full_text


i64 %874
(i64*8B

	full_text

	i64* %269
Jload8B@
>
	full_text1
/
-%875 = load i64, i64* %275, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %275
Kstore8B@
>
	full_text1
/
-store i64 %875, i64* %278, align 16, !tbaa !8
&i648B

	full_text


i64 %875
(i64*8B

	full_text

	i64* %278
5add8B,
*
	full_text

%876 = add nsw i32 %5, -1
8sext8B.
,
	full_text

%877 = sext i32 %876 to i64
&i328B

	full_text


i32 %876
ùgetelementptr8Bâ
Ü
	full_texty
w
u%878 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %877, i64 %32
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Ibitcast8B<
:
	full_text-
+
)%879 = bitcast [5 x double]* %878 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %878
Jload8B@
>
	full_text1
/
-%880 = load i64, i64* %879, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %879
Kstore8B@
>
	full_text1
/
-store i64 %880, i64* %168, align 16, !tbaa !8
&i648B

	full_text


i64 %880
(i64*8B

	full_text

	i64* %168
•getelementptr8Bë
é
	full_textÄ
~
|%881 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %877, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%882 = bitcast double* %881 to i64*
.double*8B

	full_text

double* %881
Jload8B@
>
	full_text1
/
-%883 = load i64, i64* %882, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %882
Jstore8B?
=
	full_text0
.
,store i64 %883, i64* %173, align 8, !tbaa !8
&i648B

	full_text


i64 %883
(i64*8B

	full_text

	i64* %173
•getelementptr8Bë
é
	full_textÄ
~
|%884 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %877, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%885 = bitcast double* %884 to i64*
.double*8B

	full_text

double* %884
Jload8B@
>
	full_text1
/
-%886 = load i64, i64* %885, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %885
Kstore8B@
>
	full_text1
/
-store i64 %886, i64* %178, align 16, !tbaa !8
&i648B

	full_text


i64 %886
(i64*8B

	full_text

	i64* %178
•getelementptr8Bë
é
	full_textÄ
~
|%887 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %877, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%888 = bitcast double* %887 to i64*
.double*8B

	full_text

double* %887
Jload8B@
>
	full_text1
/
-%889 = load i64, i64* %888, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %888
Jstore8B?
=
	full_text0
.
,store i64 %889, i64* %183, align 8, !tbaa !8
&i648B

	full_text


i64 %889
(i64*8B

	full_text

	i64* %183
•getelementptr8Bë
é
	full_textÄ
~
|%890 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %26, i64 %30, i64 %877, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %26
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%891 = bitcast double* %890 to i64*
.double*8B

	full_text

double* %890
Jload8B@
>
	full_text1
/
-%892 = load i64, i64* %891, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %891
Kstore8B@
>
	full_text1
/
-store i64 %892, i64* %188, align 16, !tbaa !8
&i648B

	full_text


i64 %892
(i64*8B

	full_text

	i64* %188
rgetelementptr8B_
]
	full_textP
N
L%893 = getelementptr inbounds [5 x double], [5 x double]* %111, i64 0, i64 0
:[5 x double]*8B%
#
	full_text

[5 x double]* %111
Qstore8BF
D
	full_text7
5
3store double %847, double* %893, align 16, !tbaa !8
,double8B

	full_text

double %847
.double*8B

	full_text

double* %893
8sext8B.
,
	full_text

%894 = sext i32 %857 to i64
&i328B

	full_text


i32 %857
ègetelementptr8B|
z
	full_textm
k
i%895 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 %894, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%896 = load double, double* %895, align 8, !tbaa !8
.double*8B

	full_text

double* %895
:fmul8B0
.
	full_text!

%897 = fmul double %896, %847
,double8B

	full_text

double %896
,double8B

	full_text

double %847
ègetelementptr8B|
z
	full_textm
k
i%898 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 %894, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%899 = load double, double* %898, align 8, !tbaa !8
.double*8B

	full_text

double* %898
:fmul8B0
.
	full_text!

%900 = fmul double %897, %852
,double8B

	full_text

double %897
,double8B

	full_text

double %852
Pstore8BE
C
	full_text6
4
2store double %900, double* %121, align 8, !tbaa !8
,double8B

	full_text

double %900
.double*8B

	full_text

double* %121
:fsub8B0
.
	full_text!

%901 = fsub double %839, %899
,double8B

	full_text

double %839
,double8B

	full_text

double %899
Bfmul8B8
6
	full_text)
'
%%902 = fmul double %901, 4.000000e-01
,double8B

	full_text

double %901
mcall8Bc
a
	full_textT
R
P%903 = tail call double @llvm.fmuladd.f64(double %847, double %897, double %902)
,double8B

	full_text

double %847
,double8B

	full_text

double %897
,double8B

	full_text

double %902
Qstore8BF
D
	full_text7
5
3store double %903, double* %126, align 16, !tbaa !8
,double8B

	full_text

double %903
.double*8B

	full_text

double* %126
:fmul8B0
.
	full_text!

%904 = fmul double %897, %843
,double8B

	full_text

double %897
,double8B

	full_text

double %843
Pstore8BE
C
	full_text6
4
2store double %904, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %904
.double*8B

	full_text

double* %129
Bfmul8B8
6
	full_text)
'
%%905 = fmul double %899, 4.000000e-01
,double8B

	full_text

double %899
Cfsub8B9
7
	full_text*
(
&%906 = fsub double -0.000000e+00, %905
,double8B

	full_text

double %905
ucall8Bk
i
	full_text\
Z
X%907 = tail call double @llvm.fmuladd.f64(double %839, double 1.400000e+00, double %906)
,double8B

	full_text

double %839
,double8B

	full_text

double %906
:fmul8B0
.
	full_text!

%908 = fmul double %897, %907
,double8B

	full_text

double %897
,double8B

	full_text

double %907
Qstore8BF
D
	full_text7
5
3store double %908, double* %134, align 16, !tbaa !8
,double8B

	full_text

double %908
.double*8B

	full_text

double* %134
:fmul8B0
.
	full_text!

%909 = fmul double %896, %852
,double8B

	full_text

double %896
,double8B

	full_text

double %852
:fmul8B0
.
	full_text!

%910 = fmul double %896, %843
,double8B

	full_text

double %896
,double8B

	full_text

double %843
:fmul8B0
.
	full_text!

%911 = fmul double %896, %839
,double8B

	full_text

double %896
,double8B

	full_text

double %839
8sext8B.
,
	full_text

%912 = sext i32 %615 to i64
&i328B

	full_text


i32 %615
ègetelementptr8B|
z
	full_textm
k
i%913 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 %912, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %912
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%914 = load double, double* %913, align 8, !tbaa !8
.double*8B

	full_text

double* %913
:fmul8B0
.
	full_text!

%915 = fmul double %914, %853
,double8B

	full_text

double %914
,double8B

	full_text

double %853
:fmul8B0
.
	full_text!

%916 = fmul double %914, %849
,double8B

	full_text

double %914
,double8B

	full_text

double %849
:fmul8B0
.
	full_text!

%917 = fmul double %914, %844
,double8B

	full_text

double %914
,double8B

	full_text

double %844
:fmul8B0
.
	full_text!

%918 = fmul double %914, %840
,double8B

	full_text

double %914
,double8B

	full_text

double %840
:fsub8B0
.
	full_text!

%919 = fsub double %909, %915
,double8B

	full_text

double %909
,double8B

	full_text

double %915
Bfmul8B8
6
	full_text)
'
%%920 = fmul double %919, 6.300000e+01
,double8B

	full_text

double %919
Pstore8BE
C
	full_text6
4
2store double %920, double* %143, align 8, !tbaa !8
,double8B

	full_text

double %920
.double*8B

	full_text

double* %143
:fsub8B0
.
	full_text!

%921 = fsub double %897, %916
,double8B

	full_text

double %897
,double8B

	full_text

double %916
Bfmul8B8
6
	full_text)
'
%%922 = fmul double %921, 8.400000e+01
,double8B

	full_text

double %921
Pstore8BE
C
	full_text6
4
2store double %922, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %922
.double*8B

	full_text

double* %146
:fsub8B0
.
	full_text!

%923 = fsub double %910, %917
,double8B

	full_text

double %910
,double8B

	full_text

double %917
Bfmul8B8
6
	full_text)
'
%%924 = fmul double %923, 6.300000e+01
,double8B

	full_text

double %923
Pstore8BE
C
	full_text6
4
2store double %924, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %924
.double*8B

	full_text

double* %149
:fmul8B0
.
	full_text!

%925 = fmul double %897, %897
,double8B

	full_text

double %897
,double8B

	full_text

double %897
mcall8Bc
a
	full_textT
R
P%926 = tail call double @llvm.fmuladd.f64(double %909, double %909, double %925)
,double8B

	full_text

double %909
,double8B

	full_text

double %909
,double8B

	full_text

double %925
mcall8Bc
a
	full_textT
R
P%927 = tail call double @llvm.fmuladd.f64(double %910, double %910, double %926)
,double8B

	full_text

double %910
,double8B

	full_text

double %910
,double8B

	full_text

double %926
:fmul8B0
.
	full_text!

%928 = fmul double %916, %916
,double8B

	full_text

double %916
,double8B

	full_text

double %916
mcall8Bc
a
	full_textT
R
P%929 = tail call double @llvm.fmuladd.f64(double %915, double %915, double %928)
,double8B

	full_text

double %915
,double8B

	full_text

double %915
,double8B

	full_text

double %928
mcall8Bc
a
	full_textT
R
P%930 = tail call double @llvm.fmuladd.f64(double %917, double %917, double %929)
,double8B

	full_text

double %917
,double8B

	full_text

double %917
,double8B

	full_text

double %929
:fsub8B0
.
	full_text!

%931 = fsub double %927, %930
,double8B

	full_text

double %927
,double8B

	full_text

double %930
Cfsub8B9
7
	full_text*
(
&%932 = fsub double -0.000000e+00, %928
,double8B

	full_text

double %928
mcall8Bc
a
	full_textT
R
P%933 = tail call double @llvm.fmuladd.f64(double %897, double %897, double %932)
,double8B

	full_text

double %897
,double8B

	full_text

double %897
,double8B

	full_text

double %932
Bfmul8B8
6
	full_text)
'
%%934 = fmul double %933, 1.050000e+01
,double8B

	full_text

double %933
{call8Bq
o
	full_textb
`
^%935 = tail call double @llvm.fmuladd.f64(double %931, double 0xC03E3D70A3D70A3B, double %934)
,double8B

	full_text

double %931
,double8B

	full_text

double %934
:fsub8B0
.
	full_text!

%936 = fsub double %911, %918
,double8B

	full_text

double %911
,double8B

	full_text

double %918
{call8Bq
o
	full_textb
`
^%937 = tail call double @llvm.fmuladd.f64(double %936, double 0x405EDEB851EB851E, double %935)
,double8B

	full_text

double %936
,double8B

	full_text

double %935
Pstore8BE
C
	full_text6
4
2store double %937, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %937
.double*8B

	full_text

double* %163
•getelementptr8Bë
é
	full_textÄ
~
|%938 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %912, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %912
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%939 = load double, double* %938, align 8, !tbaa !8
.double*8B

	full_text

double* %938
Qload8BG
E
	full_text8
6
4%940 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
:fsub8B0
.
	full_text!

%941 = fsub double %847, %940
,double8B

	full_text

double %847
,double8B

	full_text

double %940
vcall8Bl
j
	full_text]
[
Y%942 = tail call double @llvm.fmuladd.f64(double %941, double -3.150000e+01, double %939)
,double8B

	full_text

double %941
,double8B

	full_text

double %939
•getelementptr8Bë
é
	full_textÄ
~
|%943 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %912, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %912
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%944 = load double, double* %943, align 8, !tbaa !8
.double*8B

	full_text

double* %943
Pload8BF
D
	full_text7
5
3%945 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
:fsub8B0
.
	full_text!

%946 = fsub double %900, %945
,double8B

	full_text

double %900
,double8B

	full_text

double %945
vcall8Bl
j
	full_text]
[
Y%947 = tail call double @llvm.fmuladd.f64(double %946, double -3.150000e+01, double %944)
,double8B

	full_text

double %946
,double8B

	full_text

double %944
•getelementptr8Bë
é
	full_textÄ
~
|%948 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %912, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %912
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%949 = load double, double* %948, align 8, !tbaa !8
.double*8B

	full_text

double* %948
Qload8BG
E
	full_text8
6
4%950 = load double, double* %236, align 16, !tbaa !8
.double*8B

	full_text

double* %236
:fsub8B0
.
	full_text!

%951 = fsub double %903, %950
,double8B

	full_text

double %903
,double8B

	full_text

double %950
vcall8Bl
j
	full_text]
[
Y%952 = tail call double @llvm.fmuladd.f64(double %951, double -3.150000e+01, double %949)
,double8B

	full_text

double %951
,double8B

	full_text

double %949
•getelementptr8Bë
é
	full_textÄ
~
|%953 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %912, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %912
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%954 = load double, double* %953, align 8, !tbaa !8
.double*8B

	full_text

double* %953
Pload8BF
D
	full_text7
5
3%955 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
:fsub8B0
.
	full_text!

%956 = fsub double %904, %955
,double8B

	full_text

double %904
,double8B

	full_text

double %955
vcall8Bl
j
	full_text]
[
Y%957 = tail call double @llvm.fmuladd.f64(double %956, double -3.150000e+01, double %954)
,double8B

	full_text

double %956
,double8B

	full_text

double %954
•getelementptr8Bë
é
	full_textÄ
~
|%958 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %912, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %912
%i648B

	full_text
	
i64 %32
Pload8BF
D
	full_text7
5
3%959 = load double, double* %958, align 8, !tbaa !8
.double*8B

	full_text

double* %958
Qload8BG
E
	full_text8
6
4%960 = load double, double* %271, align 16, !tbaa !8
.double*8B

	full_text

double* %271
:fsub8B0
.
	full_text!

%961 = fsub double %908, %960
,double8B

	full_text

double %908
,double8B

	full_text

double %960
vcall8Bl
j
	full_text]
[
Y%962 = tail call double @llvm.fmuladd.f64(double %961, double -3.150000e+01, double %959)
,double8B

	full_text

double %961
,double8B

	full_text

double %959
Pload8BF
D
	full_text7
5
3%963 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
Pload8BF
D
	full_text7
5
3%964 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%965 = tail call double @llvm.fmuladd.f64(double %964, double -2.000000e+00, double %963)
,double8B

	full_text

double %964
,double8B

	full_text

double %963
Oload8BE
C
	full_text6
4
2%966 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
:fadd8B0
.
	full_text!

%967 = fadd double %965, %966
,double8B

	full_text

double %965
,double8B

	full_text

double %966
{call8Bq
o
	full_textb
`
^%968 = tail call double @llvm.fmuladd.f64(double %967, double 0x40A7418000000001, double %942)
,double8B

	full_text

double %967
,double8B

	full_text

double %942
Pload8BF
D
	full_text7
5
3%969 = load double, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
:fsub8B0
.
	full_text!

%970 = fsub double %920, %969
,double8B

	full_text

double %920
,double8B

	full_text

double %969
{call8Bq
o
	full_textb
`
^%971 = tail call double @llvm.fmuladd.f64(double %970, double 0x4019333333333334, double %947)
,double8B

	full_text

double %970
,double8B

	full_text

double %947
Pload8BF
D
	full_text7
5
3%972 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
vcall8Bl
j
	full_text]
[
Y%973 = tail call double @llvm.fmuladd.f64(double %853, double -2.000000e+00, double %972)
,double8B

	full_text

double %853
,double8B

	full_text

double %972
:fadd8B0
.
	full_text!

%974 = fadd double %852, %973
,double8B

	full_text

double %852
,double8B

	full_text

double %973
{call8Bq
o
	full_textb
`
^%975 = tail call double @llvm.fmuladd.f64(double %974, double 0x40A7418000000001, double %971)
,double8B

	full_text

double %974
,double8B

	full_text

double %971
Qload8BG
E
	full_text8
6
4%976 = load double, double* %242, align 16, !tbaa !8
.double*8B

	full_text

double* %242
:fsub8B0
.
	full_text!

%977 = fsub double %922, %976
,double8B

	full_text

double %922
,double8B

	full_text

double %976
{call8Bq
o
	full_textb
`
^%978 = tail call double @llvm.fmuladd.f64(double %977, double 0x4019333333333334, double %952)
,double8B

	full_text

double %977
,double8B

	full_text

double %952
Pload8BF
D
	full_text7
5
3%979 = load double, double* %227, align 8, !tbaa !8
.double*8B

	full_text

double* %227
vcall8Bl
j
	full_text]
[
Y%980 = tail call double @llvm.fmuladd.f64(double %849, double -2.000000e+00, double %979)
,double8B

	full_text

double %849
,double8B

	full_text

double %979
:fadd8B0
.
	full_text!

%981 = fadd double %847, %980
,double8B

	full_text

double %847
,double8B

	full_text

double %980
{call8Bq
o
	full_textb
`
^%982 = tail call double @llvm.fmuladd.f64(double %981, double 0x40A7418000000001, double %978)
,double8B

	full_text

double %981
,double8B

	full_text

double %978
Pload8BF
D
	full_text7
5
3%983 = load double, double* %259, align 8, !tbaa !8
.double*8B

	full_text

double* %259
:fsub8B0
.
	full_text!

%984 = fsub double %924, %983
,double8B

	full_text

double %924
,double8B

	full_text

double %983
{call8Bq
o
	full_textb
`
^%985 = tail call double @llvm.fmuladd.f64(double %984, double 0x4019333333333334, double %957)
,double8B

	full_text

double %984
,double8B

	full_text

double %957
Pload8BF
D
	full_text7
5
3%986 = load double, double* %244, align 8, !tbaa !8
.double*8B

	full_text

double* %244
vcall8Bl
j
	full_text]
[
Y%987 = tail call double @llvm.fmuladd.f64(double %844, double -2.000000e+00, double %986)
,double8B

	full_text

double %844
,double8B

	full_text

double %986
:fadd8B0
.
	full_text!

%988 = fadd double %843, %987
,double8B

	full_text

double %843
,double8B

	full_text

double %987
{call8Bq
o
	full_textb
`
^%989 = tail call double @llvm.fmuladd.f64(double %988, double 0x40A7418000000001, double %985)
,double8B

	full_text

double %988
,double8B

	full_text

double %985
Qload8BG
E
	full_text8
6
4%990 = load double, double* %277, align 16, !tbaa !8
.double*8B

	full_text

double* %277
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
{call8Bq
o
	full_textb
`
^%992 = tail call double @llvm.fmuladd.f64(double %991, double 0x4019333333333334, double %962)
,double8B

	full_text

double %991
,double8B

	full_text

double %962
Pload8BF
D
	full_text7
5
3%993 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
vcall8Bl
j
	full_text]
[
Y%994 = tail call double @llvm.fmuladd.f64(double %840, double -2.000000e+00, double %993)
,double8B

	full_text

double %840
,double8B

	full_text

double %993
:fadd8B0
.
	full_text!

%995 = fadd double %839, %994
,double8B

	full_text

double %839
,double8B

	full_text

double %994
{call8Bq
o
	full_textb
`
^%996 = tail call double @llvm.fmuladd.f64(double %995, double 0x40A7418000000001, double %992)
,double8B

	full_text

double %995
,double8B

	full_text

double %992
Qload8BG
E
	full_text8
6
4%997 = load double, double* %192, align 16, !tbaa !8
.double*8B

	full_text

double* %192
vcall8Bl
j
	full_text]
[
Y%998 = tail call double @llvm.fmuladd.f64(double %963, double -4.000000e+00, double %997)
,double8B

	full_text

double %963
,double8B

	full_text

double %997
ucall8Bk
i
	full_text\
Z
X%999 = tail call double @llvm.fmuladd.f64(double %964, double 6.000000e+00, double %998)
,double8B

	full_text

double %964
,double8B

	full_text

double %998
wcall8Bm
k
	full_text^
\
Z%1000 = tail call double @llvm.fmuladd.f64(double %966, double -4.000000e+00, double %999)
,double8B

	full_text

double %966
,double8B

	full_text

double %999
ocall8Be
c
	full_textV
T
R%1001 = tail call double @llvm.fmuladd.f64(double %404, double %1000, double %968)
,double8B

	full_text

double %404
-double8B

	full_text

double %1000
,double8B

	full_text

double %968
Qstore8BF
D
	full_text7
5
3store double %1001, double* %938, align 8, !tbaa !8
-double8B

	full_text

double %1001
.double*8B

	full_text

double* %938
Qload8BG
E
	full_text8
6
4%1002 = load double, double* %213, align 8, !tbaa !8
.double*8B

	full_text

double* %213
xcall8Bn
l
	full_text_
]
[%1003 = tail call double @llvm.fmuladd.f64(double %972, double -4.000000e+00, double %1002)
,double8B

	full_text

double %972
-double8B

	full_text

double %1002
Pload8BF
D
	full_text7
5
3%1004 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
xcall8Bn
l
	full_text_
]
[%1005 = tail call double @llvm.fmuladd.f64(double %1004, double 6.000000e+00, double %1003)
-double8B

	full_text

double %1004
-double8B

	full_text

double %1003
Pload8BF
D
	full_text7
5
3%1006 = load double, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
ycall8Bo
m
	full_text`
^
\%1007 = tail call double @llvm.fmuladd.f64(double %1006, double -4.000000e+00, double %1005)
-double8B

	full_text

double %1006
-double8B

	full_text

double %1005
ocall8Be
c
	full_textV
T
R%1008 = tail call double @llvm.fmuladd.f64(double %404, double %1007, double %975)
,double8B

	full_text

double %404
-double8B

	full_text

double %1007
,double8B

	full_text

double %975
Qstore8BF
D
	full_text7
5
3store double %1008, double* %943, align 8, !tbaa !8
-double8B

	full_text

double %1008
.double*8B

	full_text

double* %943
Rload8BH
F
	full_text9
7
5%1009 = load double, double* %230, align 16, !tbaa !8
.double*8B

	full_text

double* %230
xcall8Bn
l
	full_text_
]
[%1010 = tail call double @llvm.fmuladd.f64(double %979, double -4.000000e+00, double %1009)
,double8B

	full_text

double %979
-double8B

	full_text

double %1009
Qload8BG
E
	full_text8
6
4%1011 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
xcall8Bn
l
	full_text_
]
[%1012 = tail call double @llvm.fmuladd.f64(double %1011, double 6.000000e+00, double %1010)
-double8B

	full_text

double %1011
-double8B

	full_text

double %1010
Pload8BF
D
	full_text7
5
3%1013 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
ycall8Bo
m
	full_text`
^
\%1014 = tail call double @llvm.fmuladd.f64(double %1013, double -4.000000e+00, double %1012)
-double8B

	full_text

double %1013
-double8B

	full_text

double %1012
ocall8Be
c
	full_textV
T
R%1015 = tail call double @llvm.fmuladd.f64(double %404, double %1014, double %982)
,double8B

	full_text

double %404
-double8B

	full_text

double %1014
,double8B

	full_text

double %982
Qstore8BF
D
	full_text7
5
3store double %1015, double* %948, align 8, !tbaa !8
-double8B

	full_text

double %1015
.double*8B

	full_text

double* %948
Qload8BG
E
	full_text8
6
4%1016 = load double, double* %247, align 8, !tbaa !8
.double*8B

	full_text

double* %247
xcall8Bn
l
	full_text_
]
[%1017 = tail call double @llvm.fmuladd.f64(double %986, double -4.000000e+00, double %1016)
,double8B

	full_text

double %986
-double8B

	full_text

double %1016
Pload8BF
D
	full_text7
5
3%1018 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
xcall8Bn
l
	full_text_
]
[%1019 = tail call double @llvm.fmuladd.f64(double %1018, double 6.000000e+00, double %1017)
-double8B

	full_text

double %1018
-double8B

	full_text

double %1017
Pload8BF
D
	full_text7
5
3%1020 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
ycall8Bo
m
	full_text`
^
\%1021 = tail call double @llvm.fmuladd.f64(double %1020, double -4.000000e+00, double %1019)
-double8B

	full_text

double %1020
-double8B

	full_text

double %1019
ocall8Be
c
	full_textV
T
R%1022 = tail call double @llvm.fmuladd.f64(double %404, double %1021, double %989)
,double8B

	full_text

double %404
-double8B

	full_text

double %1021
,double8B

	full_text

double %989
Qstore8BF
D
	full_text7
5
3store double %1022, double* %953, align 8, !tbaa !8
-double8B

	full_text

double %1022
.double*8B

	full_text

double* %953
Rload8BH
F
	full_text9
7
5%1023 = load double, double* %264, align 16, !tbaa !8
.double*8B

	full_text

double* %264
xcall8Bn
l
	full_text_
]
[%1024 = tail call double @llvm.fmuladd.f64(double %993, double -4.000000e+00, double %1023)
,double8B

	full_text

double %993
-double8B

	full_text

double %1023
Qload8BG
E
	full_text8
6
4%1025 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
xcall8Bn
l
	full_text_
]
[%1026 = tail call double @llvm.fmuladd.f64(double %1025, double 6.000000e+00, double %1024)
-double8B

	full_text

double %1025
-double8B

	full_text

double %1024
Pload8BF
D
	full_text7
5
3%1027 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
ycall8Bo
m
	full_text`
^
\%1028 = tail call double @llvm.fmuladd.f64(double %1027, double -4.000000e+00, double %1026)
-double8B

	full_text

double %1027
-double8B

	full_text

double %1026
ocall8Be
c
	full_textV
T
R%1029 = tail call double @llvm.fmuladd.f64(double %404, double %1028, double %996)
,double8B

	full_text

double %404
-double8B

	full_text

double %1028
,double8B

	full_text

double %996
Qstore8BF
D
	full_text7
5
3store double %1029, double* %958, align 8, !tbaa !8
-double8B

	full_text

double %1029
.double*8B

	full_text

double* %958
Ögetelementptr8Br
p
	full_textc
a
_%1030 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Rstore8BG
E
	full_text8
6
4store double %963, double* %1030, align 16, !tbaa !8
,double8B

	full_text

double %963
/double*8B 

	full_text

double* %1030
Pstore8BE
C
	full_text6
4
2store double %964, double* %189, align 8, !tbaa !8
,double8B

	full_text

double %964
.double*8B

	full_text

double* %189
Pstore8BE
C
	full_text6
4
2store double %966, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %966
-double*8B

	full_text

double* %84
Lload8BB
@
	full_text3
1
/%1031 = load i64, i64* %197, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %197
Jstore8B?
=
	full_text0
.
,store i64 %1031, i64* %83, align 8, !tbaa !8
'i648B

	full_text

	i64 %1031
'i64*8B

	full_text


i64* %83
Kload8BA
?
	full_text2
0
.%1032 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
Lstore8BA
?
	full_text2
0
.store i64 %1032, i64* %202, align 16, !tbaa !8
'i648B

	full_text

	i64 %1032
(i64*8B

	full_text

	i64* %202
Lload8BB
@
	full_text3
1
/%1033 = load i64, i64* %204, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %204
Kstore8B@
>
	full_text1
/
-store i64 %1033, i64* %199, align 8, !tbaa !8
'i648B

	full_text

	i64 %1033
(i64*8B

	full_text

	i64* %199
Kload8BA
?
	full_text2
0
.%1034 = load i64, i64* %207, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %207
Lstore8BA
?
	full_text2
0
.store i64 %1034, i64* %209, align 16, !tbaa !8
'i648B

	full_text

	i64 %1034
(i64*8B

	full_text

	i64* %209
Pstore8BE
C
	full_text6
4
2store double %972, double* %213, align 8, !tbaa !8
,double8B

	full_text

double %972
.double*8B

	full_text

double* %213
Qstore8BF
D
	full_text7
5
3store double %1004, double* %210, align 8, !tbaa !8
-double8B

	full_text

double %1004
.double*8B

	full_text

double* %210
Pstore8BE
C
	full_text6
4
2store double %1006, double* %86, align 8, !tbaa !8
-double8B

	full_text

double %1006
-double*8B

	full_text

double* %86
Kload8BA
?
	full_text2
0
.%1035 = load i64, i64* %173, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %173
Jstore8B?
=
	full_text0
.
,store i64 %1035, i64* %42, align 8, !tbaa !8
'i648B

	full_text

	i64 %1035
'i64*8B

	full_text


i64* %42
Kload8BA
?
	full_text2
0
.%1036 = load i64, i64* %217, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %217
Kstore8B@
>
	full_text1
/
-store i64 %1036, i64* %220, align 8, !tbaa !8
'i648B

	full_text

	i64 %1036
(i64*8B

	full_text

	i64* %220
Kload8BA
?
	full_text2
0
.%1037 = load i64, i64* %221, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %221
Kstore8B@
>
	full_text1
/
-store i64 %1037, i64* %217, align 8, !tbaa !8
'i648B

	full_text

	i64 %1037
(i64*8B

	full_text

	i64* %217
Kload8BA
?
	full_text2
0
.%1038 = load i64, i64* %223, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %223
Kstore8B@
>
	full_text1
/
-store i64 %1038, i64* %226, align 8, !tbaa !8
'i648B

	full_text

	i64 %1038
(i64*8B

	full_text

	i64* %226
Qstore8BF
D
	full_text7
5
3store double %979, double* %230, align 16, !tbaa !8
,double8B

	full_text

double %979
.double*8B

	full_text

double* %230
Qstore8BF
D
	full_text7
5
3store double %1011, double* %227, align 8, !tbaa !8
-double8B

	full_text

double %1011
.double*8B

	full_text

double* %227
Qstore8BF
D
	full_text7
5
3store double %1013, double* %88, align 16, !tbaa !8
-double8B

	full_text

double %1013
-double*8B

	full_text

double* %88
Lload8BB
@
	full_text3
1
/%1039 = load i64, i64* %178, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %178
Jstore8B?
=
	full_text0
.
,store i64 %1039, i64* %47, align 8, !tbaa !8
'i648B

	full_text

	i64 %1039
'i64*8B

	full_text


i64* %47
Kload8BA
?
	full_text2
0
.%1040 = load i64, i64* %234, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %234
Lstore8BA
?
	full_text2
0
.store i64 %1040, i64* %237, align 16, !tbaa !8
'i648B

	full_text

	i64 %1040
(i64*8B

	full_text

	i64* %237
Lload8BB
@
	full_text3
1
/%1041 = load i64, i64* %238, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %238
Kstore8B@
>
	full_text1
/
-store i64 %1041, i64* %234, align 8, !tbaa !8
'i648B

	full_text

	i64 %1041
(i64*8B

	full_text

	i64* %234
Kload8BA
?
	full_text2
0
.%1042 = load i64, i64* %240, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %240
Lstore8BA
?
	full_text2
0
.store i64 %1042, i64* %243, align 16, !tbaa !8
'i648B

	full_text

	i64 %1042
(i64*8B

	full_text

	i64* %243
Pstore8BE
C
	full_text6
4
2store double %986, double* %247, align 8, !tbaa !8
,double8B

	full_text

double %986
.double*8B

	full_text

double* %247
Qstore8BF
D
	full_text7
5
3store double %1018, double* %244, align 8, !tbaa !8
-double8B

	full_text

double %1018
.double*8B

	full_text

double* %244
Pstore8BE
C
	full_text6
4
2store double %1020, double* %90, align 8, !tbaa !8
-double8B

	full_text

double %1020
-double*8B

	full_text

double* %90
Kload8BA
?
	full_text2
0
.%1043 = load i64, i64* %183, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %183
Jstore8B?
=
	full_text0
.
,store i64 %1043, i64* %52, align 8, !tbaa !8
'i648B

	full_text

	i64 %1043
'i64*8B

	full_text


i64* %52
Kload8BA
?
	full_text2
0
.%1044 = load i64, i64* %251, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %251
Kstore8B@
>
	full_text1
/
-store i64 %1044, i64* %254, align 8, !tbaa !8
'i648B

	full_text

	i64 %1044
(i64*8B

	full_text

	i64* %254
Kload8BA
?
	full_text2
0
.%1045 = load i64, i64* %255, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %255
Kstore8B@
>
	full_text1
/
-store i64 %1045, i64* %251, align 8, !tbaa !8
'i648B

	full_text

	i64 %1045
(i64*8B

	full_text

	i64* %251
Kload8BA
?
	full_text2
0
.%1046 = load i64, i64* %257, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %257
Kstore8B@
>
	full_text1
/
-store i64 %1046, i64* %260, align 8, !tbaa !8
'i648B

	full_text

	i64 %1046
(i64*8B

	full_text

	i64* %260
Qstore8BF
D
	full_text7
5
3store double %993, double* %264, align 16, !tbaa !8
,double8B

	full_text

double %993
.double*8B

	full_text

double* %264
Qstore8BF
D
	full_text7
5
3store double %1025, double* %261, align 8, !tbaa !8
-double8B

	full_text

double %1025
.double*8B

	full_text

double* %261
Qstore8BF
D
	full_text7
5
3store double %1027, double* %92, align 16, !tbaa !8
-double8B

	full_text

double %1027
-double*8B

	full_text

double* %92
Lload8BB
@
	full_text3
1
/%1047 = load i64, i64* %188, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %188
Jstore8B?
=
	full_text0
.
,store i64 %1047, i64* %57, align 8, !tbaa !8
'i648B

	full_text

	i64 %1047
'i64*8B

	full_text


i64* %57
Kload8BA
?
	full_text2
0
.%1048 = load i64, i64* %269, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %269
Lstore8BA
?
	full_text2
0
.store i64 %1048, i64* %272, align 16, !tbaa !8
'i648B

	full_text

	i64 %1048
(i64*8B

	full_text

	i64* %272
Lload8BB
@
	full_text3
1
/%1049 = load i64, i64* %273, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %273
Kstore8B@
>
	full_text1
/
-store i64 %1049, i64* %269, align 8, !tbaa !8
'i648B

	full_text

	i64 %1049
(i64*8B

	full_text

	i64* %269
Kload8BA
?
	full_text2
0
.%1050 = load i64, i64* %275, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %275
Lstore8BA
?
	full_text2
0
.store i64 %1050, i64* %278, align 16, !tbaa !8
'i648B

	full_text

	i64 %1050
(i64*8B

	full_text

	i64* %278
Lstore8BA
?
	full_text2
0
.store i64 %1039, i64* %112, align 16, !tbaa !8
'i648B

	full_text

	i64 %1039
(i64*8B

	full_text

	i64* %112
Cbitcast8B6
4
	full_text'
%
#%1051 = bitcast i64 %1039 to double
'i648B

	full_text

	i64 %1039
êgetelementptr8B}
{
	full_textn
l
j%1052 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %30, i64 %877, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1053 = load double, double* %1052, align 8, !tbaa !8
/double*8B 

	full_text

double* %1052
=fmul8B3
1
	full_text$
"
 %1054 = fmul double %1053, %1051
-double8B

	full_text

double %1053
-double8B

	full_text

double %1051
êgetelementptr8B}
{
	full_textn
l
j%1055 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %27, i64 %30, i64 %877, i64 %32
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %27
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %877
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1056 = load double, double* %1055, align 8, !tbaa !8
/double*8B 

	full_text

double* %1055
Cbitcast8B6
4
	full_text'
%
#%1057 = bitcast i64 %1035 to double
'i648B

	full_text

	i64 %1035
=fmul8B3
1
	full_text$
"
 %1058 = fmul double %1054, %1057
-double8B

	full_text

double %1054
-double8B

	full_text

double %1057
Qstore8BF
D
	full_text7
5
3store double %1058, double* %121, align 8, !tbaa !8
-double8B

	full_text

double %1058
.double*8B

	full_text

double* %121
Cbitcast8B6
4
	full_text'
%
#%1059 = bitcast i64 %1047 to double
'i648B

	full_text

	i64 %1047
=fsub8B3
1
	full_text$
"
 %1060 = fsub double %1059, %1056
-double8B

	full_text

double %1059
-double8B

	full_text

double %1056
Dfmul8B:
8
	full_text+
)
'%1061 = fmul double %1060, 4.000000e-01
-double8B

	full_text

double %1060
qcall8Bg
e
	full_textX
V
T%1062 = tail call double @llvm.fmuladd.f64(double %1051, double %1054, double %1061)
-double8B

	full_text

double %1051
-double8B

	full_text

double %1054
-double8B

	full_text

double %1061
Rstore8BG
E
	full_text8
6
4store double %1062, double* %126, align 16, !tbaa !8
-double8B

	full_text

double %1062
.double*8B

	full_text

double* %126
Cbitcast8B6
4
	full_text'
%
#%1063 = bitcast i64 %1043 to double
'i648B

	full_text

	i64 %1043
=fmul8B3
1
	full_text$
"
 %1064 = fmul double %1054, %1063
-double8B

	full_text

double %1054
-double8B

	full_text

double %1063
Qstore8BF
D
	full_text7
5
3store double %1064, double* %129, align 8, !tbaa !8
-double8B

	full_text

double %1064
.double*8B

	full_text

double* %129
Dfmul8B:
8
	full_text+
)
'%1065 = fmul double %1056, 4.000000e-01
-double8B

	full_text

double %1056
Efsub8B;
9
	full_text,
*
(%1066 = fsub double -0.000000e+00, %1065
-double8B

	full_text

double %1065
xcall8Bn
l
	full_text_
]
[%1067 = tail call double @llvm.fmuladd.f64(double %1059, double 1.400000e+00, double %1066)
-double8B

	full_text

double %1059
-double8B

	full_text

double %1066
=fmul8B3
1
	full_text$
"
 %1068 = fmul double %1054, %1067
-double8B

	full_text

double %1054
-double8B

	full_text

double %1067
Rstore8BG
E
	full_text8
6
4store double %1068, double* %134, align 16, !tbaa !8
-double8B

	full_text

double %1068
.double*8B

	full_text

double* %134
=fmul8B3
1
	full_text$
"
 %1069 = fmul double %1053, %1057
-double8B

	full_text

double %1053
-double8B

	full_text

double %1057
=fmul8B3
1
	full_text$
"
 %1070 = fmul double %1053, %1063
-double8B

	full_text

double %1053
-double8B

	full_text

double %1063
=fmul8B3
1
	full_text$
"
 %1071 = fmul double %1053, %1059
-double8B

	full_text

double %1053
-double8B

	full_text

double %1059
Qload8BG
E
	full_text8
6
4%1072 = load double, double* %895, align 8, !tbaa !8
.double*8B

	full_text

double* %895
=fmul8B3
1
	full_text$
"
 %1073 = fmul double %1072, %1006
-double8B

	full_text

double %1072
-double8B

	full_text

double %1006
=fmul8B3
1
	full_text$
"
 %1074 = fmul double %1072, %1013
-double8B

	full_text

double %1072
-double8B

	full_text

double %1013
=fmul8B3
1
	full_text$
"
 %1075 = fmul double %1072, %1020
-double8B

	full_text

double %1072
-double8B

	full_text

double %1020
=fmul8B3
1
	full_text$
"
 %1076 = fmul double %1072, %1027
-double8B

	full_text

double %1072
-double8B

	full_text

double %1027
=fsub8B3
1
	full_text$
"
 %1077 = fsub double %1069, %1073
-double8B

	full_text

double %1069
-double8B

	full_text

double %1073
Dfmul8B:
8
	full_text+
)
'%1078 = fmul double %1077, 6.300000e+01
-double8B

	full_text

double %1077
Qstore8BF
D
	full_text7
5
3store double %1078, double* %143, align 8, !tbaa !8
-double8B

	full_text

double %1078
.double*8B

	full_text

double* %143
=fsub8B3
1
	full_text$
"
 %1079 = fsub double %1054, %1074
-double8B

	full_text

double %1054
-double8B

	full_text

double %1074
Dfmul8B:
8
	full_text+
)
'%1080 = fmul double %1079, 8.400000e+01
-double8B

	full_text

double %1079
Qstore8BF
D
	full_text7
5
3store double %1080, double* %146, align 8, !tbaa !8
-double8B

	full_text

double %1080
.double*8B

	full_text

double* %146
=fsub8B3
1
	full_text$
"
 %1081 = fsub double %1070, %1075
-double8B

	full_text

double %1070
-double8B

	full_text

double %1075
Dfmul8B:
8
	full_text+
)
'%1082 = fmul double %1081, 6.300000e+01
-double8B

	full_text

double %1081
Qstore8BF
D
	full_text7
5
3store double %1082, double* %149, align 8, !tbaa !8
-double8B

	full_text

double %1082
.double*8B

	full_text

double* %149
=fmul8B3
1
	full_text$
"
 %1083 = fmul double %1054, %1054
-double8B

	full_text

double %1054
-double8B

	full_text

double %1054
qcall8Bg
e
	full_textX
V
T%1084 = tail call double @llvm.fmuladd.f64(double %1069, double %1069, double %1083)
-double8B

	full_text

double %1069
-double8B

	full_text

double %1069
-double8B

	full_text

double %1083
qcall8Bg
e
	full_textX
V
T%1085 = tail call double @llvm.fmuladd.f64(double %1070, double %1070, double %1084)
-double8B

	full_text

double %1070
-double8B

	full_text

double %1070
-double8B

	full_text

double %1084
=fmul8B3
1
	full_text$
"
 %1086 = fmul double %1074, %1074
-double8B

	full_text

double %1074
-double8B

	full_text

double %1074
qcall8Bg
e
	full_textX
V
T%1087 = tail call double @llvm.fmuladd.f64(double %1073, double %1073, double %1086)
-double8B

	full_text

double %1073
-double8B

	full_text

double %1073
-double8B

	full_text

double %1086
qcall8Bg
e
	full_textX
V
T%1088 = tail call double @llvm.fmuladd.f64(double %1075, double %1075, double %1087)
-double8B

	full_text

double %1075
-double8B

	full_text

double %1075
-double8B

	full_text

double %1087
=fsub8B3
1
	full_text$
"
 %1089 = fsub double %1085, %1088
-double8B

	full_text

double %1085
-double8B

	full_text

double %1088
Efsub8B;
9
	full_text,
*
(%1090 = fsub double -0.000000e+00, %1086
-double8B

	full_text

double %1086
qcall8Bg
e
	full_textX
V
T%1091 = tail call double @llvm.fmuladd.f64(double %1054, double %1054, double %1090)
-double8B

	full_text

double %1054
-double8B

	full_text

double %1054
-double8B

	full_text

double %1090
Dfmul8B:
8
	full_text+
)
'%1092 = fmul double %1091, 1.050000e+01
-double8B

	full_text

double %1091
~call8Bt
r
	full_texte
c
a%1093 = tail call double @llvm.fmuladd.f64(double %1089, double 0xC03E3D70A3D70A3B, double %1092)
-double8B

	full_text

double %1089
-double8B

	full_text

double %1092
=fsub8B3
1
	full_text$
"
 %1094 = fsub double %1071, %1076
-double8B

	full_text

double %1071
-double8B

	full_text

double %1076
~call8Bt
r
	full_texte
c
a%1095 = tail call double @llvm.fmuladd.f64(double %1094, double 0x405EDEB851EB851E, double %1093)
-double8B

	full_text

double %1094
-double8B

	full_text

double %1093
Qstore8BF
D
	full_text7
5
3store double %1095, double* %163, align 8, !tbaa !8
-double8B

	full_text

double %1095
.double*8B

	full_text

double* %163
¶getelementptr8Bí
è
	full_textÅ

}%1096 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %894, i64 %32, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1097 = load double, double* %1096, align 8, !tbaa !8
/double*8B 

	full_text

double* %1096
Rload8BH
F
	full_text9
7
5%1098 = load double, double* %201, align 16, !tbaa !8
.double*8B

	full_text

double* %201
=fsub8B3
1
	full_text$
"
 %1099 = fsub double %1051, %1098
-double8B

	full_text

double %1051
-double8B

	full_text

double %1098
ycall8Bo
m
	full_text`
^
\%1100 = tail call double @llvm.fmuladd.f64(double %1099, double -3.150000e+01, double %1097)
-double8B

	full_text

double %1099
-double8B

	full_text

double %1097
¶getelementptr8Bí
è
	full_textÅ

}%1101 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %894, i64 %32, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1102 = load double, double* %1101, align 8, !tbaa !8
/double*8B 

	full_text

double* %1101
Qload8BG
E
	full_text8
6
4%1103 = load double, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
=fsub8B3
1
	full_text$
"
 %1104 = fsub double %1058, %1103
-double8B

	full_text

double %1058
-double8B

	full_text

double %1103
ycall8Bo
m
	full_text`
^
\%1105 = tail call double @llvm.fmuladd.f64(double %1104, double -3.150000e+01, double %1102)
-double8B

	full_text

double %1104
-double8B

	full_text

double %1102
¶getelementptr8Bí
è
	full_textÅ

}%1106 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %894, i64 %32, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1107 = load double, double* %1106, align 8, !tbaa !8
/double*8B 

	full_text

double* %1106
Rload8BH
F
	full_text9
7
5%1108 = load double, double* %236, align 16, !tbaa !8
.double*8B

	full_text

double* %236
=fsub8B3
1
	full_text$
"
 %1109 = fsub double %1062, %1108
-double8B

	full_text

double %1062
-double8B

	full_text

double %1108
ycall8Bo
m
	full_text`
^
\%1110 = tail call double @llvm.fmuladd.f64(double %1109, double -3.150000e+01, double %1107)
-double8B

	full_text

double %1109
-double8B

	full_text

double %1107
¶getelementptr8Bí
è
	full_textÅ

}%1111 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %894, i64 %32, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1112 = load double, double* %1111, align 8, !tbaa !8
/double*8B 

	full_text

double* %1111
Cbitcast8B6
4
	full_text'
%
#%1113 = bitcast i64 %1044 to double
'i648B

	full_text

	i64 %1044
=fsub8B3
1
	full_text$
"
 %1114 = fsub double %1064, %1113
-double8B

	full_text

double %1064
-double8B

	full_text

double %1113
ycall8Bo
m
	full_text`
^
\%1115 = tail call double @llvm.fmuladd.f64(double %1114, double -3.150000e+01, double %1112)
-double8B

	full_text

double %1114
-double8B

	full_text

double %1112
¶getelementptr8Bí
è
	full_textÅ

}%1116 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %94, i64 %30, i64 %894, i64 %32, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %94
%i648B

	full_text
	
i64 %30
&i648B

	full_text


i64 %894
%i648B

	full_text
	
i64 %32
Rload8BH
F
	full_text9
7
5%1117 = load double, double* %1116, align 8, !tbaa !8
/double*8B 

	full_text

double* %1116
Cbitcast8B6
4
	full_text'
%
#%1118 = bitcast i64 %1048 to double
'i648B

	full_text

	i64 %1048
=fsub8B3
1
	full_text$
"
 %1119 = fsub double %1068, %1118
-double8B

	full_text

double %1068
-double8B

	full_text

double %1118
ycall8Bo
m
	full_text`
^
\%1120 = tail call double @llvm.fmuladd.f64(double %1119, double -3.150000e+01, double %1117)
-double8B

	full_text

double %1119
-double8B

	full_text

double %1117
Qload8BG
E
	full_text8
6
4%1121 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
Qload8BG
E
	full_text8
6
4%1122 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
ycall8Bo
m
	full_text`
^
\%1123 = tail call double @llvm.fmuladd.f64(double %1122, double -2.000000e+00, double %1121)
-double8B

	full_text

double %1122
-double8B

	full_text

double %1121
Pload8BF
D
	full_text7
5
3%1124 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
=fadd8B3
1
	full_text$
"
 %1125 = fadd double %1123, %1124
-double8B

	full_text

double %1123
-double8B

	full_text

double %1124
~call8Bt
r
	full_texte
c
a%1126 = tail call double @llvm.fmuladd.f64(double %1125, double 0x40A7418000000001, double %1100)
-double8B

	full_text

double %1125
-double8B

	full_text

double %1100
Qload8BG
E
	full_text8
6
4%1127 = load double, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
=fsub8B3
1
	full_text$
"
 %1128 = fsub double %1078, %1127
-double8B

	full_text

double %1078
-double8B

	full_text

double %1127
~call8Bt
r
	full_texte
c
a%1129 = tail call double @llvm.fmuladd.f64(double %1128, double 0x4019333333333334, double %1105)
-double8B

	full_text

double %1128
-double8B

	full_text

double %1105
Qload8BG
E
	full_text8
6
4%1130 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
ycall8Bo
m
	full_text`
^
\%1131 = tail call double @llvm.fmuladd.f64(double %1006, double -2.000000e+00, double %1130)
-double8B

	full_text

double %1006
-double8B

	full_text

double %1130
=fadd8B3
1
	full_text$
"
 %1132 = fadd double %1131, %1057
-double8B

	full_text

double %1131
-double8B

	full_text

double %1057
~call8Bt
r
	full_texte
c
a%1133 = tail call double @llvm.fmuladd.f64(double %1132, double 0x40A7418000000001, double %1129)
-double8B

	full_text

double %1132
-double8B

	full_text

double %1129
Rload8BH
F
	full_text9
7
5%1134 = load double, double* %242, align 16, !tbaa !8
.double*8B

	full_text

double* %242
=fsub8B3
1
	full_text$
"
 %1135 = fsub double %1080, %1134
-double8B

	full_text

double %1080
-double8B

	full_text

double %1134
~call8Bt
r
	full_texte
c
a%1136 = tail call double @llvm.fmuladd.f64(double %1135, double 0x4019333333333334, double %1110)
-double8B

	full_text

double %1135
-double8B

	full_text

double %1110
Qload8BG
E
	full_text8
6
4%1137 = load double, double* %227, align 8, !tbaa !8
.double*8B

	full_text

double* %227
ycall8Bo
m
	full_text`
^
\%1138 = tail call double @llvm.fmuladd.f64(double %1013, double -2.000000e+00, double %1137)
-double8B

	full_text

double %1013
-double8B

	full_text

double %1137
=fadd8B3
1
	full_text$
"
 %1139 = fadd double %1138, %1051
-double8B

	full_text

double %1138
-double8B

	full_text

double %1051
~call8Bt
r
	full_texte
c
a%1140 = tail call double @llvm.fmuladd.f64(double %1139, double 0x40A7418000000001, double %1136)
-double8B

	full_text

double %1139
-double8B

	full_text

double %1136
Qload8BG
E
	full_text8
6
4%1141 = load double, double* %259, align 8, !tbaa !8
.double*8B

	full_text

double* %259
=fsub8B3
1
	full_text$
"
 %1142 = fsub double %1082, %1141
-double8B

	full_text

double %1082
-double8B

	full_text

double %1141
~call8Bt
r
	full_texte
c
a%1143 = tail call double @llvm.fmuladd.f64(double %1142, double 0x4019333333333334, double %1115)
-double8B

	full_text

double %1142
-double8B

	full_text

double %1115
Qload8BG
E
	full_text8
6
4%1144 = load double, double* %244, align 8, !tbaa !8
.double*8B

	full_text

double* %244
ycall8Bo
m
	full_text`
^
\%1145 = tail call double @llvm.fmuladd.f64(double %1020, double -2.000000e+00, double %1144)
-double8B

	full_text

double %1020
-double8B

	full_text

double %1144
=fadd8B3
1
	full_text$
"
 %1146 = fadd double %1145, %1063
-double8B

	full_text

double %1145
-double8B

	full_text

double %1063
~call8Bt
r
	full_texte
c
a%1147 = tail call double @llvm.fmuladd.f64(double %1146, double 0x40A7418000000001, double %1143)
-double8B

	full_text

double %1146
-double8B

	full_text

double %1143
Rload8BH
F
	full_text9
7
5%1148 = load double, double* %277, align 16, !tbaa !8
.double*8B

	full_text

double* %277
=fsub8B3
1
	full_text$
"
 %1149 = fsub double %1095, %1148
-double8B

	full_text

double %1095
-double8B

	full_text

double %1148
~call8Bt
r
	full_texte
c
a%1150 = tail call double @llvm.fmuladd.f64(double %1149, double 0x4019333333333334, double %1120)
-double8B

	full_text

double %1149
-double8B

	full_text

double %1120
Qload8BG
E
	full_text8
6
4%1151 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
ycall8Bo
m
	full_text`
^
\%1152 = tail call double @llvm.fmuladd.f64(double %1027, double -2.000000e+00, double %1151)
-double8B

	full_text

double %1027
-double8B

	full_text

double %1151
=fadd8B3
1
	full_text$
"
 %1153 = fadd double %1152, %1059
-double8B

	full_text

double %1152
-double8B

	full_text

double %1059
~call8Bt
r
	full_texte
c
a%1154 = tail call double @llvm.fmuladd.f64(double %1153, double 0x40A7418000000001, double %1150)
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
5%1155 = load double, double* %192, align 16, !tbaa !8
.double*8B

	full_text

double* %192
ycall8Bo
m
	full_text`
^
\%1156 = tail call double @llvm.fmuladd.f64(double %1121, double -4.000000e+00, double %1155)
-double8B

	full_text

double %1121
-double8B

	full_text

double %1155
xcall8Bn
l
	full_text_
]
[%1157 = tail call double @llvm.fmuladd.f64(double %1122, double 5.000000e+00, double %1156)
-double8B

	full_text

double %1122
-double8B

	full_text

double %1156
pcall8Bf
d
	full_textW
U
S%1158 = tail call double @llvm.fmuladd.f64(double %404, double %1157, double %1126)
,double8B

	full_text

double %404
-double8B

	full_text

double %1157
-double8B

	full_text

double %1126
Rstore8BG
E
	full_text8
6
4store double %1158, double* %1096, align 8, !tbaa !8
-double8B

	full_text

double %1158
/double*8B 

	full_text

double* %1096
Qload8BG
E
	full_text8
6
4%1159 = load double, double* %213, align 8, !tbaa !8
.double*8B

	full_text

double* %213
ycall8Bo
m
	full_text`
^
\%1160 = tail call double @llvm.fmuladd.f64(double %1130, double -4.000000e+00, double %1159)
-double8B

	full_text

double %1130
-double8B

	full_text

double %1159
Pload8BF
D
	full_text7
5
3%1161 = load double, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
xcall8Bn
l
	full_text_
]
[%1162 = tail call double @llvm.fmuladd.f64(double %1161, double 5.000000e+00, double %1160)
-double8B

	full_text

double %1161
-double8B

	full_text

double %1160
pcall8Bf
d
	full_textW
U
S%1163 = tail call double @llvm.fmuladd.f64(double %404, double %1162, double %1133)
,double8B

	full_text

double %404
-double8B

	full_text

double %1162
-double8B

	full_text

double %1133
Rstore8BG
E
	full_text8
6
4store double %1163, double* %1101, align 8, !tbaa !8
-double8B

	full_text

double %1163
/double*8B 

	full_text

double* %1101
Rload8BH
F
	full_text9
7
5%1164 = load double, double* %230, align 16, !tbaa !8
.double*8B

	full_text

double* %230
ycall8Bo
m
	full_text`
^
\%1165 = tail call double @llvm.fmuladd.f64(double %1137, double -4.000000e+00, double %1164)
-double8B

	full_text

double %1137
-double8B

	full_text

double %1164
Qload8BG
E
	full_text8
6
4%1166 = load double, double* %88, align 16, !tbaa !8
-double*8B

	full_text

double* %88
xcall8Bn
l
	full_text_
]
[%1167 = tail call double @llvm.fmuladd.f64(double %1166, double 5.000000e+00, double %1165)
-double8B

	full_text

double %1166
-double8B

	full_text

double %1165
pcall8Bf
d
	full_textW
U
S%1168 = tail call double @llvm.fmuladd.f64(double %404, double %1167, double %1140)
,double8B

	full_text

double %404
-double8B

	full_text

double %1167
-double8B

	full_text

double %1140
Rstore8BG
E
	full_text8
6
4store double %1168, double* %1106, align 8, !tbaa !8
-double8B

	full_text

double %1168
/double*8B 

	full_text

double* %1106
Qload8BG
E
	full_text8
6
4%1169 = load double, double* %247, align 8, !tbaa !8
.double*8B

	full_text

double* %247
ycall8Bo
m
	full_text`
^
\%1170 = tail call double @llvm.fmuladd.f64(double %1144, double -4.000000e+00, double %1169)
-double8B

	full_text

double %1144
-double8B

	full_text

double %1169
Pload8BF
D
	full_text7
5
3%1171 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
xcall8Bn
l
	full_text_
]
[%1172 = tail call double @llvm.fmuladd.f64(double %1171, double 5.000000e+00, double %1170)
-double8B

	full_text

double %1171
-double8B

	full_text

double %1170
pcall8Bf
d
	full_textW
U
S%1173 = tail call double @llvm.fmuladd.f64(double %404, double %1172, double %1147)
,double8B

	full_text

double %404
-double8B

	full_text

double %1172
-double8B

	full_text

double %1147
Rstore8BG
E
	full_text8
6
4store double %1173, double* %1111, align 8, !tbaa !8
-double8B

	full_text

double %1173
/double*8B 

	full_text

double* %1111
Rload8BH
F
	full_text9
7
5%1174 = load double, double* %264, align 16, !tbaa !8
.double*8B

	full_text

double* %264
ycall8Bo
m
	full_text`
^
\%1175 = tail call double @llvm.fmuladd.f64(double %1151, double -4.000000e+00, double %1174)
-double8B

	full_text

double %1151
-double8B

	full_text

double %1174
Qload8BG
E
	full_text8
6
4%1176 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
xcall8Bn
l
	full_text_
]
[%1177 = tail call double @llvm.fmuladd.f64(double %1176, double 5.000000e+00, double %1175)
-double8B

	full_text

double %1176
-double8B

	full_text

double %1175
pcall8Bf
d
	full_textW
U
S%1178 = tail call double @llvm.fmuladd.f64(double %404, double %1177, double %1154)
,double8B

	full_text

double %404
-double8B

	full_text

double %1177
-double8B

	full_text

double %1154
Rstore8BG
E
	full_text8
6
4store double %1178, double* %1116, align 8, !tbaa !8
-double8B

	full_text

double %1178
/double*8B 

	full_text

double* %1116
)br8B!

	full_text

br label %1179
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


double* %1
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
,double*8B

	full_text


double* %0
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


double* %3
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
#i328B

	full_text	

i32 1
5double8B'
%
	full_text

double -4.000000e+00
$i328B

	full_text


i32 -2
4double8B&
$
	full_text

double 1.050000e+01
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
4double8B&
$
	full_text

double 1.400000e+00
#i328B

	full_text	

i32 6
:double8B,
*
	full_text

double 0x40A7418000000001
4double8B&
$
	full_text

double 6.300000e+01
$i328B

	full_text


i32 -3
:double8B,
*
	full_text

double 0xC03E3D70A3D70A3B
%i648B

	full_text
	
i64 200
$i648B

	full_text


i64 32
:double8B,
*
	full_text

double 0x4019333333333334
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 4.000000e-01
4double8B&
$
	full_text

double 4.000000e+00
4double8B&
$
	full_text

double 7.500000e-01
%i648B

	full_text
	
i64 120
4double8B&
$
	full_text

double 2.500000e-01
4double8B&
$
	full_text

double 5.000000e+00
4double8B&
$
	full_text

double 6.000000e+00
5double8B'
%
	full_text

double -3.150000e+01
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 80
5double8B'
%
	full_text

double -0.000000e+00
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 3
4double8B&
$
	full_text

double 1.000000e+00
5double8B'
%
	full_text

double -2.000000e+00
4double8B&
$
	full_text

double 8.400000e+01
:double8B,
*
	full_text

double 0x405EDEB851EB851E       	  
 

                       !" !# !! $% $& '' (( )* )) +, ++ -. -- /0 // 12 13 14 11 56 55 78 77 9: 99 ;< ;; => =? =@ == AB AA CD CC EF EE GH GG IJ IK IL II MN MM OP OO QR QQ ST SS UV UW UX UU YZ YY [\ [[ ]^ ]] _` __ ab ac ad aa ef ee gh gg ij ii kl kk mn mm op oo qr qs qq tu tt vw vx vy vv z{ zz |} |~ || Ä 	Å 	Ç  ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá äã ää åç å
é åå èê èè ëí ë
ì ëë îï îî ñó ñ
ò ñ
ô ññ öõ öö úù ú
û úú ü† üü °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂﬂ ‡· ‡‡ ‚„ ‚
‰ ‚
Â ‚‚ ÊÁ ÊÊ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô Ì
 ÌÌ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯
˚ ¯¯ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ã
ç ãã éè é
ê é
ë éé íì íí îï îî ñó ñ
ò ññ ôö ôô õú õõ ùû ù
ü ùù †° †† ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬
≈ ¬¬ ∆« ∆∆ »… »
  »» ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’’ ◊
ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ ÖÖ áà á
â áá äã ä
å ää çé çç èê èè ëí ë
ì ëë îï î
ñ îî óò ó
ô ó
ö óó õú õ
ù õ
û õõ ü† ü
° üü ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶
® ¶
© ¶¶ ™´ ™
¨ ™™ ≠
Æ ≠≠ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √
∆ √√ «» «« …  …… ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ “
’ ““ ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·
‰ ·· ÂÊ ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û  Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ ÉÉ ÖÜ ÖÖ áà áá âä ââ ãå ã
ç ãã éè éé êë êê íì íí îï îî ñó ññ òô ò
ö òò õú õõ ùû ù
ü ùù †° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ßß ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »»  À  
Ã    ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá ÜÜ àâ àà äã ää åç åå éè é
ê éé ëí ëë ìî ì
ï ìì ñó ññ òô ò
ö òò õú õ
ù õõ ûü ûû †° †† ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ ΩΩ ø¿ øø ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À  
Ã    ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯¯ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ã
ç ãã éè éé êë êê íì íí îï îî ñó ñ
ò ññ ôö ôô õú õõ ùû ù
ü ùù †° †† ¢£ ¢¢ §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂
π ∂∂ ∫ª ∫∫ ºΩ ºº æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈≈ «» «« …  …
À …… ÃÕ Ã
Œ Ã
œ ÃÃ –— –– “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË Á
È Á
Í ÁÁ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û  Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ Ö
á Ö
à ÖÖ âä â
ã ââ åç åå éè é
ê éé ëí ë
ì ëë îï îî ñ
ó ññ òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ Ÿ
‹ ŸŸ ›ﬁ ›
ﬂ ›
‡ ›› ·‚ ·
„ ·· ‰Â ‰
Ê ‰
Á ‰‰ ËÈ Ë
Í Ë
Î ËË ÏÌ Ï
Ó ÏÏ Ô
 ÔÔ ÒÚ Ò
Û Ò
Ù ÒÒ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ã
ç ãã éè é
ê éé ëí ë
ì ë
î ëë ïñ ïï óò óó ôö ô
õ ôô úù ú
û úú ü† ü
° ü
¢ üü £§ ££ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ª
æ ªª ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ Ä	Å	 Ä	
Ç	 Ä	Ä	 É	Ñ	 É	
Ö	 É	É	 Ü	á	 Ü	Ü	 à	â	 à	
ä	 à	à	 ã	å	 ã	
ç	 ã	ã	 é	è	 é	
ê	 é	é	 ë	í	 ë	ë	 ì	î	 ì	
ï	 ì	ì	 ñ	ó	 ñ	
ò	 ñ	ñ	 ô	ö	 ô	ô	 õ	ú	 õ	
ù	 õ	õ	 û	ü	 û	
†	 û	û	 °	¢	 °	
£	 °	°	 §	§	 •	¶	 •	•	 ß	®	 ß	ß	 ©	
™	 ©	©	 ´	¨	 ´	´	 ≠	
Æ	 ≠	≠	 Ø	∞	 Ø	
±	 Ø	Ø	 ≤	≥	 ≤	≤	 ¥	µ	 ¥	
∂	 ¥	¥	 ∑	∏	 ∑	
π	 ∑	
∫	 ∑	∑	 ª	º	 ª	
Ω	 ª	ª	 æ	ø	 æ	æ	 ¿	¡	 ¿	¿	 ¬	√	 ¬	¬	 ƒ	
≈	 ƒ	ƒ	 ∆	«	 ∆	
»	 ∆	∆	 …	 	 …	…	 À	Ã	 À	
Õ	 À	À	 Œ	œ	 Œ	
–	 Œ	
—	 Œ	Œ	 “	”	 “	
‘	 “	“	 ’	÷	 ’	’	 ◊	ÿ	 ◊	◊	 Ÿ	⁄	 Ÿ	Ÿ	 €	
‹	 €	€	 ›	ﬁ	 ›	
ﬂ	 ›	›	 ‡	·	 ‡	‡	 ‚	„	 ‚	
‰	 ‚	‚	 Â	Ê	 Â	
Á	 Â	
Ë	 Â	Â	 È	Í	 È	
Î	 È	È	 Ï	Ì	 Ï	Ï	 Ó	Ô	 Ó	Ó	 	Ò	 		 Ú	
Û	 Ú	Ú	 Ù	ı	 Ù	
ˆ	 Ù	Ù	 ˜	¯	 ˜	˜	 ˘	˙	 ˘	
˚	 ˘	˘	 ¸	˝	 ¸	
˛	 ¸	
ˇ	 ¸	¸	 Ä
Å
 Ä

Ç
 Ä
Ä
 É
Ñ
 É
É
 Ö
Ü
 Ö
Ö
 á
à
 á
á
 â

ä
 â
â
 ã
å
 ã

ç
 ã
ã
 é
è
 é
é
 ê
ë
 ê

í
 ê
ê
 ì
î
 ì

ï
 ì

ñ
 ì
ì
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
ö
 ú
ù
 ú

û
 ú
ú
 ü
†
 ü

°
 ü
ü
 ¢
£
 ¢

§
 ¢
¢
 •
¶
 •

ß
 •
•
 ®
©
 ®
®
 ™
´
 ™

¨
 ™
™
 ≠
Æ
 ≠
≠
 Ø
∞
 Ø

±
 Ø
Ø
 ≤
≥
 ≤
≤
 ¥
µ
 ¥

∂
 ¥
¥
 ∑
∏
 ∑

π
 ∑
∑
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
√
 ≈
∆
 ≈

«
 ≈
≈
 »
…
 »
»
  
À
  

Ã
  
 
 Õ
Œ
 Õ
Õ
 œ
–
 œ

—
 œ
œ
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
ﬁ
 ‡
·
 ‡

‚
 ‡
‡
 „
‰
 „
„
 Â
Ê
 Â

Á
 Â
Â
 Ë
È
 Ë
Ë
 Í
Î
 Í

Ï
 Í
Í
 Ì
Ó
 Ì
Ì
 Ô

 Ô

Ò
 Ô
Ô
 Ú
Û
 Ú

Ù
 Ú
Ú
 ı
ˆ
 ı

˜
 ı
ı
 ¯
˘
 ¯

˙
 ¯
¯
 ˚
¸
 ˚
˚
 ˝
˛
 ˝

ˇ
 ˝
˝
 ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà á
â áá äã ää åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô òò öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢¢ §• §
¶ §§ ß® ß
© ß
™ ßß ´¨ ´´ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤
µ ≤≤ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø Ω
¿ ΩΩ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »
À »» ÃÕ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”
÷ ”” ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „
Ê „„ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô ÏÏ Ò  ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝
Ä ˝˝ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ää å
ç åå éè é
ê éé ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …
Ã …… ÕŒ Õ
œ Õ
– ÕÕ —“ —
” —— ‘’ ‘
÷ ‘
◊ ‘‘ ÿŸ ÿ
⁄ ÿ
€ ÿÿ ‹› ‹
ﬁ ‹‹ ﬂ
‡ ﬂﬂ ·‚ ·
„ ·
‰ ·· ÂÊ ÂÂ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É Å
Ñ ÅÅ ÖÜ ÖÖ áà áá âä â
ã ââ åç å
é åå èê è
ë è
í èè ìî ìì ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ù
ü ù
† ùù °¢ °° £§ ££ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »»  À  
Ã    ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä ââ ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë îï îî ñó ñ
ò ññ ôö ô
õ ôô úù úú ûü û
† ûû °¢ °
£ °
§ °° •¶ •
ß •• ®© ®® ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ π
ª π
º ππ Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «« …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —
‘ —— ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï ÈÈ ÌÓ Ì
Ô ÌÌ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É Å
Ñ ÅÅ ÖÜ Ö
á ÖÖ àà ââ äã ää åç åå éè éé êë êê íì íí îï îî ñó ññ òô òò öõ öö úù úú ûü û° †† ¢£ ¢¢ §• §§ ¶ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞∞ ≤¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ Å
Ç ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé ç
è çç êë ê
í êê ìî ì
ï ìì ñó ññ òô òò öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” “
‘ “
’ “
÷ ““ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁ
‚ ﬁﬁ „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ ÍÎ Í
Ï Í
Ì Í
Ó ÍÍ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆ
˘ ˆ
˙ ˆˆ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà áá âä ââ ãå ã
ç ãã éè é
ê éé ëí ë
ì ë
î ë
ï ëë ñó ññ òô ò
ö òò õú õ
ù õ
û õ
ü õõ †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´´ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ º
Ω ºº æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –
“ –
” –
‘ –– ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛
Å ˛˛ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à ÜÜ âä â
ã â
å ââ çé ç
è ç
ê çç ëí ë
ì ëë î
ï îî ñó ñ
ò ñ
ô ññ öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®
´ ®
¨ ®® ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑
∫ ∑
ª ∑∑ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆
… ∆
  ∆∆ ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’
ÿ ’
Ÿ ’’ ⁄€ ⁄⁄ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰
Á ‰
Ë ‰‰ ÈÍ ÈÈ ÎÏ ÎÎ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿ
€ ÿÿ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò Ô
Ú ÔÔ ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä à
ã àà åç å
é åå èê èè ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ öö úù ú
û úú ü† ü
° ü
¢ üü £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂
π ∂∂ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆∆ »… »»  À    ÃÕ ÃÃ Œœ ŒŒ –— –– “” ““ ‘’ ‘◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá ÜÜ àä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº øø ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ àâ àà äã ä
å ää çé çç èê è
ë èè íì íí îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À  
Ã    ÕÕ Œœ ŒŒ –— –
“ –
” –
‘ –– ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹
ﬂ ‹
‡ ‹‹ ·‚ ·· „‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í Ë
Î Ë
Ï ËË ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ Ù
˜ Ù
¯ ÙÙ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç Ä
É Ä
Ñ ÄÄ ÖÜ ÖÖ áà áá âä â
ã ââ åç åå éè é
ê éé ëí ëë ìî ì
ï ì
ñ ì
ó ìì òô òò öõ ö
ú öö ùû ù
ü ù
† ù
° ùù ¢£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ ºº æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” ““ ‘’ ‘
÷ ‘
◊ ‘
ÿ ‘‘ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à Ü
â ÜÜ äã ä
å ää çé ç
è ç
ê çç ëí ë
ì ë
î ëë ïñ ï
ó ïï ò
ô òò öõ ö
ú ö
ù öö ûü ûû †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨
Æ ¨
Ø ¨
∞ ¨¨ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ª
æ ª
ø ªª ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À  
Ã  
Õ  
Œ    œ– œœ —“ —— ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ Ÿ
‹ Ÿ
› ŸŸ ﬁﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í Ë
Î Ë
Ï ËË ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã éè éé êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü û
† ûû °¢ °° £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «« …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›
‡ ›› ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá Ü
à ÜÜ âä â
ã â
å ââ çé ç
è çç êë êê íì í
î íí ïñ ïï óò ó
ô óó öõ öö úù ú
û úú ü† ü
° ü
¢ üü £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µ
∏ µµ π∫ π
ª ππ ºΩ ºº æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «« …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ãã çé ç
è çç êë êê íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü ûû †° †
¢ †† £§ ££ •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” ““ ‘’ ‘
÷ ‘
◊ ‘
ÿ ‘‘ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁ
‚ ﬁﬁ „‰ „„ ÂÊ ÂÂ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı Ù
ˆ Ù
˜ ÙÙ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â áá äã ä
å ää çé ç
è çç êë ê
í êê ìî ì
ï ìì ñó ñ
ò ññ ôö ôô õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬
≈ ¬¬ ∆« ∆
» ∆
… ∆∆  À  
Ã    ÕŒ Õ
œ Õ
– ÕÕ —“ —
” —
‘ —— ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô Ï
 ÏÏ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚
˛ ˚
ˇ ˚˚ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ä
å ä
ç ä
é ää èê èè ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ô
ú ô
ù ôô ûü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®
´ ®
¨ ®® ≠Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ªª æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆∆ »… »
  »» ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà áá âä â
ã ââ åç å
é åå èê è
ë èè íì íí îï î
ñ îî óò ó
ô óó öõ ö
ú ö
ù öö ûü û
† ûû °¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ º
ø ºº ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À  
Ã    ÕŒ Õ
œ Õ
– ÕÕ —“ —
” —— ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚‚ Â
Á ÊÊ Ë
È ËË Í
Î ÍÍ ÏÌ ﬂÓ Ô àÔ âÔ øÔ Õ &Ò  Ú 'Û (   	            " #! % *) , .- 0& 2+ 3/ 41 65 8 :9 <& >+ ?/ @= BA D FE H& J+ K/ LI NM P RQ T& V+ W/ XU ZY \ ^] `& b+ c/ da fe h ji l nm pO ro sO u( w+ x/ yv {z }t ~' Ä+ Å/ Ç ÑC Ü| àÖ â ãá çä ég êè íÉ ìë ït ó| òî ô õñ ùö û[ †| ¢ü £ •° ß§ ®É ™© ¨è Æ´ Ø| ±≠ ≤ ¥∞ ∂≥ ∑ π∏ ª Ωº ø7 ¡æ ¬ ƒ√ ∆C »≈ … À  ÕO œÃ – “— ‘[ ÷” ◊ Ÿÿ €g ›⁄ ﬁ9 ·& „+ ‰/ Â‚ ÁÊ ÈË Î; Ï& Ó+ Ô/ Ì ÚÒ ÙÛ ˆG ˜& ˘+ ˙/ ˚¯ ˝¸ ˇ˛ ÅS Ç& Ñ+ Ö/ ÜÉ àá äâ å_ ç& è+ ê/ ëé ìí ïî ók ò öô ú˛ ûõ ü˛ °( £+ §/ •¢ ß¶ ©† ™' ¨+ ≠/ Æ´ ∞Û ≤® ¥± µ ∑≥ π∂ ∫î ºª æØ øΩ ¡† √® ƒ¿ ≈ «¬ …∆  â Ã® ŒÀ œ —Õ ”– ‘Ø ÷’ ÿª ⁄◊ €® ›Ÿ ﬁ ‡‹ ‚ﬂ „¶ Â± Ê¶ ËÀ È¶ Îª Ïz ÓÖ Ôz Òü Úz Ùè ı‰ ˜Ì ¯ˆ ˙ ¸˘ ˛˚ ˇ® Å| ÇÄ Ñ ÜÉ àÖ âÁ ã åä é êç íè ì® ï® ñ‰ ò‰ ôî öÁ úÁ ùó û| †| °Ì £Ì §ü • ß ®¢ ©õ ´¶ ¨ü Æ® ∞® ±≠ ≤Ø ¥™ ∂≥ ∑Í πÛ ∫∏ ºµ Ω øª ¡æ ¬& ƒ+ ≈/ ∆√ »«   ÃÀ Œ… –Õ —& ”+ ‘/ ’“ ◊÷ Ÿ €⁄ ›ÿ ﬂ‹ ‡& ‚+ „/ ‰· ÊÂ Ë ÍÈ ÏÁ ÓÎ Ô& Ò+ Ú/ Û ıÙ ˜ ˘¯ ˚ˆ ˝˙ ˛& Ä+ Å/ Çˇ ÑÉ Ü àá äÖ åâ ç èé ëê ì ï óí ôñ öæ úõ ûê ü∫ °† £æ § ¶• ®… ™∫ ´ ≠¨ ØÆ ± ≥ µ∞ ∑¥ ∏ ∫π ºª æΩ ¿Æ ¡ √¬ ≈ƒ « …∆ À» Ã ŒÕ –œ “ ‘” ÷— ÿ’ Ÿ≈ €⁄ ›œ ﬁG ‡ﬂ ‚≈ „ÿ ÂG Êä ËÁ Í ÏÎ ÓÈ Ì Ò∂ ÛÚ ıÙ ˜Á ¯˚ ˙˘ ¸ ˛˝ Ä˚ Çˇ É ÖÑ áÜ â ãä çà èå êÃ íë îÜ ïS óñ ôÃ öÁ úS ùö üû ° £¢ •† ß§ ®∆ ™© ¨´ Æû ØÖ ±∞ ≥ µ¥ ∑≤ π∂ ∫ ºª æΩ ¿ ¬¡ ƒø ∆√ «” …» ÀΩ Ã_ ŒÕ –” —ˆ ”_ ‘§ ÷’ ÿ ⁄Ÿ ‹◊ ﬁ€ ﬂ– ·‡ „‚ Â’ Êè ËÁ Í ÏÎ ÓÈ Ì Ò ÛÚ ıÙ ˜ ˘¯ ˚ˆ ˝˙ ˛⁄ Äˇ ÇÙ Ék ÖÑ á⁄ àâ äâ åk ç≥ èé ë ìí ïê óî òﬂ öô úõ ûé üæ °† £ •§ ß¢ ©¶ ™& ¨+ ≠/ Æ´ ∞Ø ≤± ¥Õ µ& ∑+ ∏/ π∂ ª∫ Ωº ø‹ ¿& ¬+ √/ ƒ¡ ∆≈ »«  Î À& Õ+ Œ/ œÃ —– ”“ ’˙ ÷& ÿ+ Ÿ/ ⁄◊ ‹€ ﬁ› ‡â ·Á „õ ‰Á Ê( Ë+ È/ ÍÁ ÏÎ ÓÂ Ô' Ò+ Ú/ Û ıE ˜Ì ˘ˆ ˙¯ ¸∂ ˝â ˇ˛ ÅÙ ÇÄ ÑÂ ÜÌ áÉ àÖ ä∆ ãˆ çÌ èå êé í– ìÙ ïî ó˛ ôñ öÌ úò ùõ üﬂ †Î ¢ˆ £Î •å ¶Î ®˛ ©√ ´¶ ≠™ Æ  ∞¶ ≤Ø ≥Õ µ¶ ∑¥ ∏Ñ ∫¶ ºπ Ω° ø¨ ¿æ ¬¡ ƒ˚ ≈Ì «± »∆  … ÃÖ Õ§ œ∂ –Œ “— ‘è ’Ì ◊Ì ÿ° ⁄° €÷ ‹§ ﬁ§ ﬂŸ ‡± ‚± „¨ Â¨ Ê· Á∂ È∂ Í‰ Î› ÌË Ó· Ì ÚÌ ÛÔ ÙÒ ˆÏ ¯ı ˘ß ˚ª ¸˙ ˛˜ ˇ˝ Åæ Çﬂ Ñ+ Ö/ ÜÉ à≤ äÂ åâ çã èá êﬂ í+ ì/ îë ñÎ ò¯ öó õô ùï ûﬂ †+ °/ ¢ü §¢ ¶Ö ®• ©ß ´£ ¨ﬂ Æ+ Ø/ ∞≠ ≤Ÿ ¥é ∂≥ ∑µ π± ∫ﬂ º+ Ω/ æª ¿í ¬õ ƒ¡ ≈√ «ø »é  º ÃÀ Œ… œ‡ —Õ ”– ‘“ ÷é ◊˝ Ÿ¡ €ÿ ‹⁄ ﬁú ﬂÕ ·™ „‡ ‰ˆ Ê‚ ÁÂ È› Í¥ Ï… ÓÎ ÔÌ Ò™ ÚÑ ÙØ ˆÛ ˜ı ˘Â ˙¯ ¸ ˝Î ˇ— Å	˛ Ç	Ä	 Ñ	∏ Ö	ª á	¥ â	Ü	 ä	à	 å	å ç	ã	 è	É	 ê	§ í	˝ î	ë	 ï	ì	 ó	∆ ò	Ú ö	π ú	ô	 ù	õ	 ü	˛ †	û	 ¢	ñ	 £	§	 ¶	•	 ®	ß	 ™	– ¨	´	 Æ	À ∞	≠	 ±	• ≥	≤	 µ	Ø	 ∂	©	 ∏	¥	 π	’ ∫	∑	 º	É Ω	√ ø	E ¡	¿	 √	¬	 ≈	æ	 «	ƒ	 »	⁄  	…	 Ã	∆	 Õ	©	 œ	À	 –	Ë —	Œ	 ”	ë ‘	  ÷	Q ÿ	◊	 ⁄	Ÿ	 ‹	’	 ﬁ	€	 ﬂ	È ·	‡	 „	›	 ‰	©	 Ê	‚	 Á	˚ Ë	Â	 Í	ü Î	— Ì	] Ô	Ó	 Ò		 Û	Ï	 ı	Ú	 ˆ	¯ ¯	˜	 ˙	Ù	 ˚	©	 ˝	˘	 ˛	é	 ˇ	¸	 Å
≠ Ç
ÿ Ñ
i Ü
Ö
 à
á
 ä
É
 å
â
 ç
á è
é
 ë
ã
 í
©	 î
ê
 ï
°	 ñ
ì
 ò
ª ô
 õ
… ù
ö
 û
À †
é °
– £
º §
≤	 ¶
∏ ß
Æ ©
®
 ´
¥ ¨
ª Æ
≠
 ∞
Æ ±
ƒ ≥
≤
 µ
» ∂
‡ ∏
” π
æ	 ª
Õ º
¿	 æ
√ ø
…	 ¡
E ¬
Á ƒ
√
 ∆
Ì «
Ú …
»
 À
Á Ã
˘ Œ
Õ
 –
ˇ —
Û ”
ä ‘
’	 ÷
Ñ ◊
◊	 Ÿ
  ⁄
‡	 ‹
Q ›
û ﬂ
ﬁ
 ·
§ ‚
© ‰
„
 Ê
û Á
∞ È
Ë
 Î
∂ Ï
Ω Ó
Ì
 
√ Ò
Ï	 Û
ª Ù
Ó	 ˆ
— ˜
˜	 ˘
] ˙
’ ¸
˚
 ˛
€ ˇ
‡ ÅÄ É’ ÑÁ ÜÖ àÌ âÙ ãä ç˙ éÉ
 êÚ ëÖ
 ìÿ îé
 ñi óé ôò õî úô ûù †é °† £¢ •¶ ¶& ®+ ©/ ™ß ¨´ Æ≠ ∞Õ ±& ≥+ ¥/ µ≤ ∑∂ π∏ ª‹ º& æ+ ø/ ¿Ω ¬¡ ƒ√ ∆Î «& …+  / À» ÕÃ œŒ —˙ “& ‘+ ’/ ÷” ÿ◊ ⁄Ÿ ‹â ›ô ﬂ‡	 ·ﬁ ‚( ‰+ Â/ Ê„ ËÁ Í‡	 Î' Ì+ Ó/ ÔÏ ÒÈ Û…	 ÙÚ ˆ∂ ˜é
 ˘ ˙¯ ¸‡	 ˛È ˇ˚ Ä˝ Ç∆ ÉÈ Ö˜	 ÜÑ à– â ãä çé
 èå êÈ íé ìë ïﬂ ñÁ ò…	 ôÁ õ˜	 úÁ ûé
 üÁ °† £¿	 §† ¶◊	 ß† ©Ó	 ™† ¨Ö
 ≠ó Ø¢ ∞Æ ≤± ¥˚ µÈ ∑• ∏∂ ∫π ºÖ Ωö ø® ¿æ ¬¡ ƒè ≈È «È »ó  ó À∆ Ãö Œö œ… –• “• ”¢ ’¢ ÷— ◊® Ÿ® ⁄‘ €Õ ›ÿ ﬁ— ‡È ‚È „ﬂ ‰· Ê‹ ËÂ Èù Î´ ÏÍ ÓÁ ÔÌ Òæ Úﬂ Ù+ ı/ ˆÛ ¯≤ ˙‡	 ¸˘ ˝˚ ˇ˜ Äﬂ Ç+ É/ ÑÅ ÜÎ àÚ äá ãâ çÖ éﬂ ê+ ë/ íè î¢ ñ˝ òï ôó õì úﬂ û+ ü/ †ù ¢Ÿ §Ñ ¶£ ß• ©° ™ﬂ ¨+ ≠/ Æ´ ∞í ≤ë ¥± µ≥ ∑Ø ∏é ∫º ºª æπ ø‡ ¡Ω √¿ ƒ¬ ∆˛ «˝ …± À» Ã  Œå œÕ —¿	 ”– ‘…	 ÷“ ◊’ ŸÕ ⁄¥ ‹π ﬁ€ ﬂ› ·ö ‚Ñ ‰◊	 Ê„ Á‡	 ÈÂ ÍË Ï‡ ÌÎ Ô¡ ÒÓ Ú Ù® ıª ˜Ó	 ˘ˆ ˙˜	 ¸¯ ˝˚ ˇÛ Ä§ ÇÌ ÑÅ ÖÉ á∂ àÚ äÖ
 åâ çé
 èã êé íÜ ìª ïπ óî ò¿ öñ õ• ùú üô †©	 ¢û £≈ §° ¶Û ß√ ©® ´– ≠™ ÆE ∞Ø ≤¨ ≥⁄ µ¥ ∑± ∏©	 ∫∂ ªÿ ºπ æÅ ø  ¡¿ √„ ≈¬ ∆Q »«  ƒ ÀÈ ÕÃ œ… –©	 “Œ ”Î ‘— ÷è ◊— Ÿÿ €ˆ ›⁄ ﬁ] ‡ﬂ ‚‹ „¯ Â‰ Á· Ë©	 ÍÊ Î˛ ÏÈ Óù Ôÿ Ò Ûâ ıÚ ˆi ¯˜ ˙Ù ˚á ˝¸ ˇ˘ Ä©	 Ç˛ Éë ÑÅ Ü´ áú ã– ç® è„ ë¿ ìÃ ïˆ óÿ ôâ õ ùâ ü °ƒ £ô •à ßÆ ©Á ´û ≠’ Øé ±∞ ¥… µÆ ∑π ∏¨ ∫© ª™ Ωñ æ® ¿ã ¡¸ √± ƒ˜ ∆¬ «ú …“  ö Ã– Õ‰ œö –ﬂ “Œ ”ò ’Œ ÷ñ ÿÃ ŸÃ €É ‹î ﬁ  ﬂ« ·˛ ‚í ‰» Âê Á∆ Ë¥ ÍÍ ÎØ ÌÈ Óé ƒ Òå Û¬ Ùä ˆ¿ ˜¿ ˘ˆ ˙ª ¸¯ ˝π ˇ˚ ÄÉ ÇÅ Ñı Ü∫ áø â¥ äª åÚ é’ èÔ ëœ íº îÌ ïÚ ó˘ ôò õˇ úÊ ûå ü„ °Ü ¢› §S •π ß§ ®© ™∞ ¨´ Æ∂ Ø◊ ±√ ≤‘ ¥Ω µ∂ ∑€ ∏‡ ∫Á ºª æÌ øÀ ¡˙ ¬» ƒÙ ≈≥ «î »ô  † ÃÀ Œ¶ œÅ —& ”+ ‘– ’/ ÷“ ÿ◊ ⁄Ÿ ‹Õ ›& ﬂ+ ‡– ·/ ‚ﬁ ‰„ ÊÂ Ë‹ È& Î+ Ï– Ì/ ÓÍ Ô ÚÒ ÙÎ ı& ˜+ ¯– ˘/ ˙ˆ ¸˚ ˛˝ Ä˙ Å& É+ Ñ– Ö/ ÜÇ àá äâ åâ ç⁄ è§ ê( í+ ìÉ î/ ïë óñ ô⁄ ö' ú+ ùÉ û/ üõ °ò £È §¢ ¶∂ ß¬ ©† ™® ¨⁄ Æò Ø´ ∞≠ ≤∆ ≥ò µŒ ∂¥ ∏– π† ª∫ Ω¬ øº ¿ò ¬æ √¡ ≈ﬂ ∆ñ »È …ñ ÀŒ Ãñ Œ¬ œ( —+ “Å ”/ ‘– ÷’ ÿÏ Ÿ’ €‡ ‹’ ﬁ— ﬂ’ ·≈ ‚« ‰◊ Â„ ÁÊ È˚ Íò Ï⁄ ÌÎ ÔÓ ÒÖ Ú  Ù› ıÛ ˜ˆ ˘è ˙ò ¸ò ˝« ˇ« Ä˚ Å  É  Ñ˛ Ö⁄ á⁄ à◊ ä◊ ãÜ å› é› èâ êÇ íç ìÜ ïò óò òî ôñ õë ùö ûÕ †‡ °ü £ú §¢ ¶æ ßﬂ ©+ ™Å ´/ ¨® Æ≤ ∞⁄ ≤Ø ≥± µ≠ ∂ﬂ ∏+ πÅ ∫/ ª∑ ΩÎ ø¢ ¡æ ¬¿ ƒº ≈ﬂ «+ »Å …/  ∆ Ã¢ Œ≠ –Õ —œ ”À ‘ﬂ ÷+ ◊Å ÿ/ Ÿ’ €Ÿ ›¥ ﬂ‹ ‡ﬁ ‚⁄ „ﬂ Â+ ÊÅ Á/ Ë‰ Íí Ï¡ ÓÎ ÔÌ ÒÈ Ú¯ Ù˚ ı‡ ˜Û ˘ˆ ˙¯ ¸¥ ˝˝ ˇÊ Å˛ ÇÄ Ñ√ ÖÕ áÏ âÜ äÈ åà çã èÉ ê¥ íÓ îë ïì ó“ òÑ ö‡ úô ù⁄ üõ †û ¢ñ £Î •ˆ ß§ ®¶ ™· ´ª ≠— Ø¨ ∞Œ ≤Æ ≥± µ© ∂§ ∏¢ ∫∑ ªπ Ω æÚ ¿≈ ¬ø √¬ ≈¡ ∆ƒ »º …˚ À˛ Ã¯ Œ  œˆ —Õ “• ‘– ÷” ◊©	 Ÿ’ ⁄˚ €ÿ ›® ﬁ” ‡Ü ‚ﬂ „Ï Â· ÊÈ Ë‰ È⁄ ÎÁ ÌÍ Ó©	 Ï Òé ÚÔ Ù∑ ıä ˜ô ˘ˆ ˙‡ ¸¯ ˝Q ˇ˛ Å˚ ÇÈ ÑÄ ÜÉ á©	 âÖ ä° ãà ç∆ é¡ ê¨ íè ì— ïë ñŒ òî ô¯ õó ùö û©	 †ú °¥ ¢ü §’ •¯ ßø ©¶ ™≈ ¨® ≠¬ Ø´ ∞á ≤Æ ¥± µ©	 ∑≥ ∏« π∂ ª‰ ºÉ æ¶ ø” ¡Ü √Ï ≈ô «‡ …É À¨ Õ— œø —≈ ”Ω ’˛ ◊† ÿ˚ ⁄é €¯ ›º ﬁã ‡Æ ·¢ „» ‰Ï Ê√ ÁÈ ÈE Íñ ÏÁ Ì‡ Ô  © Úû Û— ı— ˆŒ ¯] ˘π ˚’ ¸≈ ˛ÿ ˇ¬ Åi Ç… Ñé Öœ á± ä¸ ã¬ ç˜ é“ êú ë– ìö îö ñ‰ óŒ ôﬂ öŒ úò ùÃ üñ †É ¢Ã £  •î ¶˛ ®« ©» ´í ¨∆ Æê ØÍ ±¥ ≤È ¥Ø µƒ ∑é ∏Ü ∫å ª¿ Ωä æê ¡¿ √ñ ƒæ ∆≈ »ê …∫ À  Õæ Œº –∫ —Æ ”“ ’¥ ÷ª ÿ◊ ⁄Æ €ƒ ›‹ ﬂ» ‡π ‚’ „∂ Âœ Ê≥ Ë√ È∞ ÎE ÏÁ ÓÌ Ì ÒÚ ÛÚ ıÁ ˆ˘ ¯˜ ˙ˇ ˚≠ ˝å ˛™ ÄÜ Åß É  Ñ§ ÜS áû âà ã§ å© éç êû ë∞ ìí ï∂ ñû ò√ ôõ õΩ úò û— üï °] ¢’ §£ ¶€ ß‡ ©® ´’ ¨Á Æ≠ ∞Ì ±í ≥˙ ¥è ∂Ù ∑å πÿ ∫â ºi Ωé øæ ¡î ¬ô ƒ√ ∆é «† …» À¶ ÃÕ œ& —+ “Œ ”/ ‘– ÷’ ÿ◊ ⁄Õ €& ›+ ﬁŒ ﬂ/ ‡‹ ‚· ‰„ Ê‹ Á& È+ ÍŒ Î/ ÏË ÓÌ Ô ÚÎ Û& ı+ ˆŒ ˜/ ¯Ù ˙˘ ¸˚ ˛˙ ˇ& Å+ ÇŒ É/ ÑÄ ÜÖ àá äâ ãô ç° èå êø í( î+ ïë ñ/ óì ôò õ° ú' û+ üë †/ °ù £ö •∞ ¶§ ®∂ ©â ´¢ ¨™ Æ° ∞ö ±≠ ≤Ø ¥∆ µö ∑ï ∏∂ ∫– ª¢ Ωº øâ ¡æ ¬ö ƒ¿ ≈√ «ﬂ »ò  ∞ Àò Õï Œò –â —à ”( ’+ ÷“ ◊/ ÿ‘ ⁄Ÿ ‹≥ ›Ÿ ﬂß ‡Ÿ ‚ò „Ÿ Âå Ê… Ë€ ÈÁ ÎÍ Ì˚ Óö ﬁ ÒÔ ÛÚ ıÖ ˆÃ ¯· ˘˜ ˚˙ ˝è ˛ö Äö Å… É… Ñˇ ÖÃ áÃ àÇ âﬁ ãﬁ å€ é€ èä ê· í· ìç îÜ ñë óä ôö õö úò ùö üï °û ¢œ §‰ •£ ß† ®¶ ™æ ´ﬂ ≠+ Æ“ Ø/ ∞¨ ≤≤ ¥° ∂≥ ∑µ π± ∫ﬂ º+ Ω“ æ/ øª ¡Î √§ ≈¬ ∆ƒ »¿ …ﬂ À+ Ã“ Õ/ Œ  –¢ “Ø ‘— ’” ◊œ ÿﬂ ⁄+ €“ ‹/ ›Ÿ ﬂŸ ·∂ „‡ ‰‚ Êﬁ Áﬂ È+ Í“ Î/ ÏË Óí √ ÚÔ ÛÒ ıÌ ˆé ¯º ˙˘ ¸˜ ˝‡ ˇ˚ Å˛ ÇÄ Ñ∏ Ö˝ áÍ âÜ äà å« çÕ è≥ ëé í∞ îê ïì óã ò¥ öÚ úô ùõ ü÷ †Ñ ¢ß §° •° ß£ ®¶ ™û ´Î ≠˙ Ø¨ ∞Æ ≤Â ≥ª µò ∑¥ ∏ï ∫∂ ªπ Ω± æ§ ¿¶ ¬ø √¡ ≈Ù ∆Ú »å  « Àâ Õ… ŒÃ –ƒ —î ”˜ ’“ ÷˘ ÿ‘ Ÿ˛ €◊ ‹©	 ﬁ⁄ ﬂÉ ‡› ‚¨ „” Âé Á‰ Ë√ ÍÈ ÏÊ ÌE ÔÓ ÒÎ Ú©	 Ù ıñ ˆÛ ¯ª ˘ä ˚° ˝˙ ˛  Äˇ Ç¸ ÉQ ÖÑ áÅ à©	 äÜ ã© åâ é  è¡ ë¥ ìê î— ñï òí ô] õö ùó û©	 †ú °º ¢ü §Ÿ •¯ ß« ©¶ ™ÿ ¨´ Æ® Øi ±∞ ≥≠ ¥©	 ∂≤ ∑œ ∏µ ∫Ë ª Ω˜ øº ¿˘ ¬é √˛ ≈º ∆ß »«  ∫ ÀÆ ÕÃ œ¥ –ª “— ‘Æ ’ƒ ◊÷ Ÿ» ⁄é ‹” ›È ﬂÕ ‡Ó ‚√ „‹ Â‰ ÁG ËÁ ÍÈ ÏÌ ÌÚ ÔÓ ÒÁ Ú˘ ÙÛ ˆˇ ˜° ˘ä ˙ˇ ¸Ñ ˝Ñ ˇ  ÄÎ ÇÅ ÑS Öû áÜ â§ ä© åã éû è∞ ëê ì∂ î¥ ñ¡ óï ôª öö ú— ù˙ üû °_ ¢’ §£ ¶€ ß‡ ©® ´’ ¨Á Æ≠ ∞Ì ±« ≥¯ ¥´ ∂Ú ∑∞ πÿ ∫â ºª æk øé ¡¿ √î ƒô ∆≈ »é …† À  Õ¶ ŒÅ –õ —Å ”( ’+ ÷Œ ◊/ ÿ‘ ⁄Ÿ ‹“ ›' ﬂ+ ‡Œ ·/ ‚ﬁ ‰‰ Ê€ ËÂ ÈÁ Î∂ Ïª ÓÌ „ ÒÔ Û“ ı€ ˆÚ ˜Ù ˘∆ ˙û ¸€ ˛˚ ˇ˝ Å– Ç„ ÑÉ ÜÌ àÖ â€ ãá åä éﬂ èŸ ëÂ íŸ î˚ ïŸ óÌ òì öô úÓ ùô üÑ †ô ¢ö £ô •∞ ¶ê ®õ ©ß ´™ ≠˚ Æ€ ∞û ±Ø ≥≤ µÖ ∂ì ∏° π∑ ª∫ Ωè æ€ ¿€ ¡ê √ê ƒø ≈ì «ì »¬ …û Àû Ãõ Œõ œ  –° “° ”Õ ‘∆ ÷— ◊  Ÿ€ €€ ‹ÿ ›⁄ ﬂ’ ·ﬁ ‚ñ ‰§ Â„ Á‡ ËÊ Íæ Îﬂ Ì+ Óë Ô/ Ï Ú≤ Ù“ ˆÛ ˜ı ˘Ò ˙ﬂ ¸+ ˝ë ˛/ ˇ˚ ÅÎ ÉÁ ÖÇ ÜÑ àÄ âﬂ ã+ åë ç/ éä ê¢ íÙ îë ïì óè òﬂ ö+ õë ú/ ùô ü£ °˝ £† §¢ ¶û ßﬂ ©+ ™ë ´/ ¨® Æ¿ ∞ä ≤Ø ≥± µ≠ ∂é ∏º ∫π º∑ Ω‡ øª ¡æ ¬¿ ƒ¯ ≈˝ «™ …∆  » Ãá ÕÕ œÓ —Œ “– ‘Â ’” ◊À ÿ¥ ⁄≤ ‹Ÿ ›€ ﬂñ ‡Ñ ‚Ñ ‰· Â„ Á“ ËÊ Íﬁ ÎÎ Ì∫ ÔÏ Ó Ú• Ûª ıö ˜Ù ¯ˆ ˙˚ ˚˘ ˝Ò ˛§ ÄÊ Çˇ ÉÅ Ö¥ ÜÚ à∞ äá ãâ çÌ éå êÑ ëî ì∑ ïí ñπ òî ô©	 õó ú√ ùö üÏ †” ¢Œ §° •√ ß¶ ©£ ™©	 ¨® ≠÷ Æ´ ∞˚ ±ä ≥· µ≤ ∂  ∏∑ ∫¥ ª©	 Ωπ æÈ øº ¡ä ¬¡ ƒÙ ∆√ «— …» À≈ Ã©	 Œ  œ¸ –Õ “ô ”¯ ’á ◊‘ ÿÿ ⁄Ÿ ‹÷ ›©	 ﬂ€ ‡è ·ﬁ „® ‰ Á È Î  Ê$ &$ Êû †û â≤ ≥Â Ê‘ ÷‘ ≥à â ÙÙ Ï ıı ˆˆ ˜˜ ¯¯â ˆˆ â⁄ ˆˆ ⁄™ ˆˆ ™ú ˆˆ úé ˆˆ é® ˆˆ ®è ˆˆ èØ ˆˆ Ø∏ ˆˆ ∏ô ˆˆ ôç ˆˆ ç– ˆˆ –◊ ˆˆ ◊· ˆˆ ·⁄ ˆˆ ⁄• ˆˆ •¬ ˆˆ ¬ÿ ˆˆ ÿé ˆˆ éÅ ˆˆ ÅÇ ˆˆ Ç¸ ˆˆ ¸± ˆˆ ±› ˆˆ ›È ˆˆ Èº ˆˆ º´ ˆˆ ´Î ˆˆ Î≠ ˆˆ ≠Ñ ˆˆ Ñ® ˆˆ ®∆ ˆˆ ∆˜ ˆˆ ˜‡ ˆˆ ‡Ë ¯¯ Ëê ˆˆ êò ˆˆ òë ˆˆ ë› ˆˆ ›˚ ˆˆ ˚ƒ ˆˆ ƒ∑	 ˆˆ ∑	ë ˆˆ ë¯ ˆˆ ¯Å ˆˆ Å‚ ˆˆ ‚ú ˆˆ úö ˆˆ ö‘ ˆˆ ‘ö ˆˆ ö¬ ˆˆ ¬ì
 ˆˆ ì
€ ˆˆ €Õ ˆˆ Õâ ˆˆ â¸	 ˆˆ ¸	  ˆˆ  ´ ˆˆ ´‘ ˆˆ ‘Ê ˆˆ ÊÕ ˆˆ Õ÷ ˆˆ ÷∂ ˆˆ ∂ñ ˆˆ ñµ ˆˆ µº ˆˆ º÷ ˆˆ ÷¢ ˆˆ ¢… ˆˆ …Ù ˆˆ Ù ÙÙ Æ ˆˆ ÆÂ	 ˆˆ Â	’ ˆˆ ’Ê ˆˆ ÊÈ ˆˆ ÈÙ	 ˆˆ Ù	î ˆˆ îπ ˆˆ πÍ ¯¯ Í· ˆˆ ·ã
 ˆˆ ã
í ˆˆ íå ˆˆ åâ ˆˆ â√ ˆˆ √± ˆˆ ±ƒ ˆˆ ƒó ˆˆ óÖ ˆˆ Ö¥ ˆˆ ¥ñ ˆˆ ñÁ ˆˆ Á ˆˆ ∂ ˆˆ ∂Ù ˆˆ Ùñ ˆˆ ñ… ˆˆ …— ˆˆ —¯ ˆˆ ¯É ˆˆ É£ ˆˆ £Ë ˆˆ Ë¥ ˆˆ ¥ﬁ ˆˆ ﬁµ ˆˆ µ° ˆˆ ° ˆˆ ≈ ˆˆ ≈« ˆˆ «‰ ˆˆ ‰Ÿ ˆˆ Ÿœ ˆˆ œ“ ˆˆ “Û ˆˆ Û≤ ˆˆ ≤Ü ˆˆ Ü£ ˆˆ £ü ˆˆ üı ˆˆ ıÉ ˆˆ Éó ˆˆ ó÷ ˆˆ ÷¨ ˆˆ ¨Ê ¯¯ Ê® ˆˆ ®Ä ˆˆ Ä¶ ˆˆ ¶õ ˆˆ õÌ ˆˆ Ìã ˆˆ ãÿ ˆˆ ÿ  ˆˆ  Î ˆˆ Î¸ ˆˆ ¸Û ˆˆ Û“ ˆˆ “√ ˆˆ √Û ˆˆ ÛÒ ˆˆ Òÿ ˆˆ ÿŒ	 ˆˆ Œ	Á ˆˆ Á∂ ˆˆ ∂˝ ˆˆ ˝˛ ˆˆ ˛° ˆˆ °ö ˆˆ ö≈ ˆˆ ≈à	 ˆˆ à	∆	 ˆˆ ∆	À ˆˆ Àæ ˆˆ æØ	 ˆˆ Ø	Â ˆˆ Â‹ ˆˆ ‹á ˆˆ áÕ ˆˆ ÕÆ ˆˆ Æ† ˆˆ †à ˆˆ à ÙÙ „ ˆˆ „ñ ˆˆ ñ˚ ˆˆ ˚… ˆˆ …º ˆˆ º ıı á ˆˆ á© ˆˆ ©Ç ˆˆ Ç˚ ˆˆ ˚Ü ˆˆ Üó ˆˆ ó‡ ˆˆ ‡
 ÙÙ 
ñ ˆˆ ñü ˆˆ ü– ˆˆ –à ˆˆ à§	 ˜˜ §	π ˆˆ πË ˆˆ Ëñ	 ˆˆ ñ	· ˆˆ ·é	 ˆˆ é	‰ ˆˆ ‰Â ˆˆ Âõ	 ˆˆ õ	® ˆˆ ®› ˆˆ ›Ÿ ˆˆ Ÿõ ˆˆ õ· ˆˆ ·Õ ˆˆ ÕÒ ˆˆ Òé ˆˆ é˛ ˆˆ ˛¯ ˆˆ ¯≠ ˆˆ ≠ç ˆˆ çã ˆˆ ãÉ	 ˆˆ É	ª ˆˆ ª— ˆˆ —¥ ˆˆ ¥ú ˆˆ úû ˆˆ û ıı Ô ˆˆ Ôˆ ˆˆ ˆ ˆˆ î ˆˆ î∆ ˆˆ ∆« ˆˆ «∏ ˆˆ ∏ñ ˆˆ ñ¿ ˆˆ ¿•	 ˜˜ •	Ø ˆˆ ØÜ ˆˆ ÜÙ ˆˆ Ù© ˆˆ ©ª ˆˆ ªﬁ ˆˆ ﬁ¡ ˆˆ ¡›	 ˆˆ ›	¶ ˆˆ ¶˘ ˆˆ ˘¢ ˆˆ ¢¥ ˆˆ ¥≠ ˆˆ ≠˚ ˆˆ ˚Õ ˆˆ Õ˛ ˆˆ ˛°	 ˆˆ °	ó ˆˆ óë ˆˆ ëÕ ˆˆ ÕΩ ˆˆ Ω˝ ˆˆ ˝˘ ˘ ˘ ˘ 
˙ ñ
˙ ô
˙ ¨
˙ ±
˙ ƒ
˙ …
˙ ‹
˙ ·
˙ Ù
˙ ˘
˙  
˙ –
˙ ·
˙ Á
˙ ¯
˙ Ä
˙ ë
˙ ó
˙ ®
˙ Æ
˙ ‘
˙ ⁄
˙ Ê
˙ 
˙ ¸
˙ Ü
˙ í
˙ ú
˙ ®
˙ ≤
˙ î
˙ £
˙ ¥
˙ ≈
˙ ÷
˚ ø
¸ ≥
¸ ı
¸ Â
¸ ö
¸ û
¸ ﬁ˝ 	˛ 1	˛ 9	˛ =	˛ E	˛ I	˛ Q	˛ U	˛ ]	˛ a	˛ i	˛ m	˛ v	˛ 
˛ ä
˛ ö
˛ §
˛ ≥
˛ ∏
˛ ∏
˛ º
˛ º
˛ √
˛  
˛ —
˛ ÿ
˛ ‡
˛ ‡
˛ ô
˛ ∂
˛ ∆
˛ –
˛ ﬂ
˛ ˚
˛ Ö
˛ è
˛ æ
˛ À
˛ ⁄
˛ È
˛ ¯
˛ á
˛ é
˛ é
˛ î
˛ î
˛ î
˛ •
˛ •
˛ ¨
˛ ¨
˛ ≤
˛ ≤
˛ ≤
˛ π
˛ π
˛ ¬
˛ ¬
˛ Õ
˛ ”
˛ ”
˛ Î
˛ Î
˛ ˝
˛ ˝
˛ Ñ
˛ ä
˛ ä
˛ ¢
˛ ¢
˛ ¥
˛ ¥
˛ ª
˛ ¡
˛ ¡
˛ Ÿ
˛ Ÿ
˛ Î
˛ Î
˛ Ú
˛ ¯
˛ ¯
˛ í
˛ í
˛ §
˛ §
˛ É
˛ ö

˛ ö

˛ ö

˛ ﬁ
˛ ﬁ
˛ Û
˛ †
˛ †
˛ †
˛ §
˛ §
˛ ®
˛ å
˛ å
˛ ¨
˛ º
˛ º
˛ º
˛ Ï
ˇ ≠
ˇ Ÿ
ˇ ò
ˇ é
ˇ æ
ˇ ¿
ˇ á
Ä â
Å ’
Å Ë
Å ˚
Å é	
Å °	
Å ≈
Å ÿ
Å Î
Å ˛
Å ë
Å ˚
Å é
Å °
Å ¥
Å «
Å É
Å ñ
Å ©
Å º
Å œ
Å √
Å ÷
Å È
Å ¸
Å è
Ç ˘
Ç ç
Ç ¡
Ç —
Ç ±
Ç ¡
Ç Ê
Ç ˆ
Ç Í
Ç ˙
Ç ™
Ç ∫
É à
Ñ µ
Ñ ˜
Ñ Á
Ñ ú
Ñ †
Ñ ‡Ö Ö Ê	Ü )	Ü +	Ü -	Ü /
á ›
á 
á É	
á ñ	
á Õ
á ‡
á Û
á Ü
á É
á ñ
á ©
á º
á ã
á û
á ±
á ƒ
á À
á ﬁ
á Ò
á Ñ	à 	à 	à =	à E	à m
à ä
à ä
à ö
à §
à ≥
à √
à ‚
à Ì
à Ì
à ¯
à É
à é
à ¢
à ´
à ∂
à ˚
à ˚
à Ö
à è
à æ
à “
à ⁄
à é
à ¨
à ¬
à Õ
à Õ
à ”
à Î
à ˝
à Ñ
à ª
à Ú
à ∂
à É
à ë
à ë
à ü
à ≠
à ª
à ≤
à Å
à É
à ﬁ
à ∑
à ‹
à ª
à ˚	â I	â Q
â ö
â º
â √
â  
â  
â —
â ÿ
â ¯
â ô
â ∂
â ∆
â ∆
â –
â ﬂ
â Ö
â √
â “
â ·
â ·
â È
â 
â ˇ
â π
â Ñ
â ä
â ¢
â ¥
â ¡
â Á
â 
â ü
â Ω
â Û
â Å
â è
â è
â ù
â ´
â –
â Í
â ∆
â Ë
â  
â ä
ä î
ä ©
ä ¿
ä ’
ä É
ä î
ä ˚
ä ä
ä ´
ä ∫
ä ≠
ä º
ä Ú
ä É
ã ´	
ã ¬	
ã Ÿ	
ã 	
ã á
å §	
å §	ç 
ç Í
é ß	
è Ø	
è ∆	
è ›	
è Ù	
è ã

è ó
è ®
è π
è  
è €
ê î
ê ™
ê ¬
ê ⁄
ê Ú
ê Õ
ê ‰
ê ˚
ê î
ê ´
ê ◊
ê Î
ê Å
ê ó
ê ≠
ë é
ë ú
ë ™
ë ∏
ë ∆
ë ˛
ë å
ë ö
ë ®
ë ∂
ë ¥
ë √
ë “
ë ·
ë 
ë ∏
ë «
ë ÷
ë Â
ë Ù
ë ¯
ë á
ë ñ
ë •
ë ¥	í a	í i
í ≥
í ÿ
í é
í ﬂ
í æ
í À
í ⁄
í È
í ¯
í ˇ
í á
í á
í •
í Ú
í ¯
í í
í §
í ◊
í ª
í ß
í ≤
í Ω
í »
í ”
í ”
í ´
í Ç
í ‰
í Ä
í Ë
í ®ì ì Ëî ´î ◊î ≠î ñî Ôî ©	î ≠	î ƒ	î €	î Ú	î â
î åî ﬂî ºî îî æî òî Öî ÿ	ï 	ï  
ï Õ	ñ 9	ñ E	ñ Q	ñ U	ñ ]	ñ ]	ñ i
ñ §
ñ ∏
ñ —
ñ É
ñ –
ñ è
ñ 
ñ ¯
ñ ª
ñ ¡
ñ Ÿ
ñ Î
ñ ´
ñ ∂
ñ ¡
ñ Ã
ñ Ã
ñ ◊
ñ ≠
ñ »
ñ „
ñ Ï
ñ ùñ Å
ñ ˆ
ñ ’
ñ Ù
ñ Ÿ
ñ ô
ó •	
ò Õ
ò ‚
ò ı
ò à	
ò õ	
ò Ω
ò “
ò Â
ò ¯
ò ã
ò Û
ò à
ò õ
ò Æ
ò ¡
ò ˚
ò ê
ò £
ò ∂
ò …
ò ª
ò –
ò „
ò ˆ
ò â
ô É
ô …
ô π
ô Ó
ô Ú
ô ≤
ö ª
ö ˝
ö Ì
ö ¢
ö ¶
ö Ê"
rhsy"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z3maxdd"
llvm.lifetime.end.p0i8*á
npb-LU-rhsy.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

devmap_label

 
transfer_bytes_log1p
⁄açA

transfer_bytes
òì…

wgsize
>

wgsize_log1p
⁄açA