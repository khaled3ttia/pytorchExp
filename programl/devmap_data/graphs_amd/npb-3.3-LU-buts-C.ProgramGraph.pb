

[external]
FallocaB<
:
	full_text-
+
)%11 = alloca [5 x [5 x double]], align 16
FallocaB<
:
	full_text-
+
)%12 = alloca [5 x [5 x double]], align 16
FallocaB<
:
	full_text-
+
)%13 = alloca [5 x [5 x double]], align 16
FallocaB<
:
	full_text-
+
)%14 = alloca [5 x [5 x double]], align 16
FallocaB<
:
	full_text-
+
)%15 = alloca [5 x [5 x double]], align 16
@allocaB6
4
	full_text'
%
#%16 = alloca [5 x double], align 16
JbitcastB?
=
	full_text0
.
,%17 = bitcast [5 x [5 x double]]* %11 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %11
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %17) #4
#i8*B

	full_text
	
i8* %17
JbitcastB?
=
	full_text0
.
,%18 = bitcast [5 x [5 x double]]* %12 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %12
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %18) #4
#i8*B

	full_text
	
i8* %18
JbitcastB?
=
	full_text0
.
,%19 = bitcast [5 x [5 x double]]* %13 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %13
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %19) #4
#i8*B

	full_text
	
i8* %19
JbitcastB?
=
	full_text0
.
,%20 = bitcast [5 x [5 x double]]* %14 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %14
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %20) #4
#i8*B

	full_text
	
i8* %20
JbitcastB?
=
	full_text0
.
,%21 = bitcast [5 x [5 x double]]* %15 to i8*
C[5 x [5 x double]]*B*
(
	full_text

[5 x [5 x double]]* %15
[callBS
Q
	full_textD
B
@call void @llvm.lifetime.start.p0i8(i64 200, i8* nonnull %21) #4
#i8*B

	full_text
	
i8* %21
DbitcastB9
7
	full_text*
(
&%22 = bitcast [5 x double]* %16 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %16
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %22) #4
#i8*B

	full_text
	
i8* %22
LcallBD
B
	full_text5
3
1%23 = tail call i64 @_Z13get_global_idj(i32 1) #5
3sextB+
)
	full_text

%24 = sext i32 %8 to i64
2addB+
)
	full_text

%25 = add nsw i64 %24, 1
#i64B

	full_text
	
i64 %24
0addB)
'
	full_text

%26 = add i64 %25, %23
#i64B

	full_text
	
i64 %25
#i64B

	full_text
	
i64 %23
6truncB-
+
	full_text

%27 = trunc i64 %26 to i32
#i64B

	full_text
	
i64 %26
LcallBD
B
	full_text5
3
1%28 = tail call i64 @_Z13get_global_idj(i32 0) #5
3sextB+
)
	full_text

%29 = sext i32 %9 to i64
2addB+
)
	full_text

%30 = add nsw i64 %29, 1
#i64B

	full_text
	
i64 %29
0addB)
'
	full_text

%31 = add i64 %30, %28
#i64B

	full_text
	
i64 %30
#i64B

	full_text
	
i64 %28
3sextB+
)
	full_text

%32 = sext i32 %7 to i64
2addB+
)
	full_text

%33 = add nsw i64 %32, 1
#i64B

	full_text
	
i64 %32
4subB-
+
	full_text

%34 = sub nsw i64 %33, %24
#i64B

	full_text
	
i64 %33
#i64B

	full_text
	
i64 %24
4subB-
+
	full_text

%35 = sub nsw i64 %34, %29
#i64B

	full_text
	
i64 %34
#i64B

	full_text
	
i64 %29
0subB)
'
	full_text

%36 = sub i64 %35, %23
#i64B

	full_text
	
i64 %35
#i64B

	full_text
	
i64 %23
0subB)
'
	full_text

%37 = sub i64 %36, %28
#i64B

	full_text
	
i64 %36
#i64B

	full_text
	
i64 %28
6truncB-
+
	full_text

%38 = trunc i64 %37 to i32
#i64B

	full_text
	
i64 %37
2addB+
)
	full_text

%39 = add nsw i32 %4, -1
6icmpB.
,
	full_text

%40 = icmp sgt i32 %39, %27
#i32B

	full_text
	
i32 %39
#i32B

	full_text
	
i32 %27
9brB3
1
	full_text$
"
 br i1 %40, label %41, label %915
!i1B

	full_text


i1 %40
8trunc8B-
+
	full_text

%42 = trunc i64 %31 to i32
%i648B

	full_text
	
i64 %31
4add8B+
)
	full_text

%43 = add nsw i32 %5, -1
8icmp8B.
,
	full_text

%44 = icmp sgt i32 %43, %42
%i328B

	full_text
	
i32 %43
%i328B

	full_text
	
i32 %42
6icmp8B,
*
	full_text

%45 = icmp sgt i32 %38, 0
%i328B

	full_text
	
i32 %38
1and8B(
&
	full_text

%46 = and i1 %44, %45
#i18B

	full_text


i1 %44
#i18B

	full_text


i1 %45
4add8B+
)
	full_text

%47 = add nsw i32 %6, -1
8icmp8B.
,
	full_text

%48 = icmp sgt i32 %47, %38
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %38
1and8B(
&
	full_text

%49 = and i1 %48, %46
#i18B

	full_text


i1 %48
#i18B

	full_text


i1 %46
;br8B3
1
	full_text$
"
 br i1 %49, label %50, label %915
#i18B

	full_text


i1 %49
Ybitcast8BL
J
	full_text=
;
9%51 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
Ybitcast8BL
J
	full_text=
;
9%52 = bitcast double* %1 to [163 x [163 x [5 x double]]]*
Sbitcast8BF
D
	full_text7
5
3%53 = bitcast double* %2 to [163 x [163 x double]]*
Sbitcast8BF
D
	full_text7
5
3%54 = bitcast double* %3 to [163 x [163 x double]]*
1shl8B(
&
	full_text

%55 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
9ashr8B/
-
	full_text 

%56 = ashr exact i64 %55, 32
%i648B

	full_text
	
i64 %55
1shl8B(
&
	full_text

%57 = shl i64 %31, 32
%i648B

	full_text
	
i64 %31
9ashr8B/
-
	full_text 

%58 = ashr exact i64 %57, 32
%i648B

	full_text
	
i64 %57
1shl8B(
&
	full_text

%59 = shl i64 %37, 32
%i648B

	full_text
	
i64 %37
9ashr8B/
-
	full_text 

%60 = ashr exact i64 %59, 32
%i648B

	full_text
	
i64 %59
‘getelementptr8B~
|
	full_texto
m
k%61 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %54, i64 %56, i64 %58, i64 %60
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
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

%63 = fmul double %62, %62
+double8B

	full_text


double %62
+double8B

	full_text


double %62
7fmul8B-
+
	full_text

%64 = fmul double %62, %63
+double8B

	full_text


double %62
+double8B

	full_text


double %63
ƒgetelementptr8Bp
n
	full_texta
_
]%65 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
^store8BS
Q
	full_textD
B
@store double 0x410FA45800000002, double* %65, align 16, !tbaa !8
-double*8B

	full_text

double* %65
ƒgetelementptr8Bp
n
	full_texta
_
]%66 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %66, align 8, !tbaa !8
-double*8B

	full_text

double* %66
ƒgetelementptr8Bp
n
	full_texta
_
]%67 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %67, align 16, !tbaa !8
-double*8B

	full_text

double* %67
ƒgetelementptr8Bp
n
	full_texta
_
]%68 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
ƒgetelementptr8Bp
n
	full_texta
_
]%69 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %69, align 16, !tbaa !8
-double*8B

	full_text

double* %69
@fmul8B6
4
	full_text'
%
#%70 = fmul double %63, 1.000000e-01
+double8B

	full_text


double %63
¨getelementptr8B”
‘
	full_textƒ
€
~%71 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Nload8BD
B
	full_text5
3
1%72 = load double, double* %71, align 8, !tbaa !8
-double*8B

	full_text

double* %71
7fmul8B-
+
	full_text

%73 = fmul double %70, %72
+double8B

	full_text


double %70
+double8B

	full_text


double %72
Ffmul8B<
:
	full_text-
+
)%74 = fmul double %73, 0xC115183555555556
+double8B

	full_text


double %73
ƒgetelementptr8Bp
n
	full_texta
_
]%75 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Nstore8BC
A
	full_text4
2
0store double %74, double* %75, align 8, !tbaa !8
+double8B

	full_text


double %74
-double*8B

	full_text

double* %75
@fmul8B6
4
	full_text'
%
#%76 = fmul double %62, 4.000000e-01
+double8B

	full_text


double %62
call8Bw
u
	full_texth
f
d%77 = tail call double @llvm.fmuladd.f64(double %76, double 0x40F5183555555556, double 1.000000e+00)
+double8B

	full_text


double %76
Ffadd8B<
:
	full_text-
+
)%78 = fadd double %77, 0x410FA45000000002
+double8B

	full_text


double %77
ƒgetelementptr8Bp
n
	full_texta
_
]%79 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Nstore8BC
A
	full_text4
2
0store double %78, double* %79, align 8, !tbaa !8
+double8B

	full_text


double %78
-double*8B

	full_text

double* %79
ƒgetelementptr8Bp
n
	full_texta
_
]%80 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
ƒgetelementptr8Bp
n
	full_texta
_
]%81 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %81, align 8, !tbaa !8
-double*8B

	full_text

double* %81
ƒgetelementptr8Bp
n
	full_texta
_
]%82 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %82, align 8, !tbaa !8
-double*8B

	full_text

double* %82
¨getelementptr8B”
‘
	full_textƒ
€
~%83 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Nload8BD
B
	full_text5
3
1%84 = load double, double* %83, align 8, !tbaa !8
-double*8B

	full_text

double* %83
7fmul8B-
+
	full_text

%85 = fmul double %70, %84
+double8B

	full_text


double %70
+double8B

	full_text


double %84
Ffmul8B<
:
	full_text-
+
)%86 = fmul double %85, 0xC115183555555556
+double8B

	full_text


double %85
ƒgetelementptr8Bp
n
	full_texta
_
]%87 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Ostore8BD
B
	full_text5
3
1store double %86, double* %87, align 16, !tbaa !8
+double8B

	full_text


double %86
-double*8B

	full_text

double* %87
ƒgetelementptr8Bp
n
	full_texta
_
]%88 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %88, align 8, !tbaa !8
-double*8B

	full_text

double* %88
ƒgetelementptr8Bp
n
	full_texta
_
]%89 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Ostore8BD
B
	full_text5
3
1store double %78, double* %89, align 16, !tbaa !8
+double8B

	full_text


double %78
-double*8B

	full_text

double* %89
ƒgetelementptr8Bp
n
	full_texta
_
]%90 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
ƒgetelementptr8Bp
n
	full_texta
_
]%91 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %91, align 16, !tbaa !8
-double*8B

	full_text

double* %91
¨getelementptr8B”
‘
	full_textƒ
€
~%92 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Nload8BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8B

	full_text

double* %92
7fmul8B-
+
	full_text

%94 = fmul double %70, %93
+double8B

	full_text


double %70
+double8B

	full_text


double %93
Ffmul8B<
:
	full_text-
+
)%95 = fmul double %94, 0xC115183555555556
+double8B

	full_text


double %94
ƒgetelementptr8Bp
n
	full_texta
_
]%96 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Nstore8BC
A
	full_text4
2
0store double %95, double* %96, align 8, !tbaa !8
+double8B

	full_text


double %95
-double*8B

	full_text

double* %96
ƒgetelementptr8Bp
n
	full_texta
_
]%97 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %97, align 8, !tbaa !8
-double*8B

	full_text

double* %97
ƒgetelementptr8Bp
n
	full_texta
_
]%98 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
ƒgetelementptr8Bp
n
	full_texta
_
]%99 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Nstore8BC
A
	full_text4
2
0store double %78, double* %99, align 8, !tbaa !8
+double8B

	full_text


double %78
-double*8B

	full_text

double* %99
„getelementptr8Bq
o
	full_textb
`
^%100 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %100, align 8, !tbaa !8
.double*8B

	full_text

double* %100
8fmul8B.
,
	full_text

%101 = fmul double %72, %72
+double8B

	full_text


double %72
+double8B

	full_text


double %72
8fmul8B.
,
	full_text

%102 = fmul double %84, %84
+double8B

	full_text


double %84
+double8B

	full_text


double %84
Hfmul8B>
<
	full_text/
-
+%103 = fmul double %102, 0xC0B9C936F46508DE
,double8B

	full_text

double %102
{call8Bq
o
	full_textb
`
^%104 = tail call double @llvm.fmuladd.f64(double %101, double 0xC0B9C936F46508DF, double %103)
,double8B

	full_text

double %101
,double8B

	full_text

double %103
8fmul8B.
,
	full_text

%105 = fmul double %93, %93
+double8B

	full_text


double %93
+double8B

	full_text


double %93
{call8Bq
o
	full_textb
`
^%106 = tail call double @llvm.fmuladd.f64(double %105, double 0xC0B9C936F46508DF, double %104)
,double8B

	full_text

double %105
,double8B

	full_text

double %104
Gfmul8B=
;
	full_text.
,
*%107 = fmul double %63, 0x40CDC4C624DD2F1B
+double8B

	full_text


double %63
©getelementptr8B•
’
	full_text„

%108 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%109 = load double, double* %108, align 8, !tbaa !8
.double*8B

	full_text

double* %108
:fmul8B0
.
	full_text!

%110 = fmul double %107, %109
,double8B

	full_text

double %107
,double8B

	full_text

double %109
lcall8Bb
`
	full_textS
Q
O%111 = tail call double @llvm.fmuladd.f64(double %106, double %64, double %110)
,double8B

	full_text

double %106
+double8B

	full_text


double %64
,double8B

	full_text

double %110
Cfmul8B9
7
	full_text*
(
&%112 = fmul double %111, -4.000000e+00
,double8B

	full_text

double %111
„getelementptr8Bq
o
	full_textb
`
^%113 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Qstore8BF
D
	full_text7
5
3store double %112, double* %113, align 16, !tbaa !8
,double8B

	full_text

double %112
.double*8B

	full_text

double* %113
Gfmul8B=
;
	full_text.
,
*%114 = fmul double %63, 0xC0D9C936F46508DF
+double8B

	full_text


double %63
9fmul8B/
-
	full_text 

%115 = fmul double %114, %72
,double8B

	full_text

double %114
+double8B

	full_text


double %72
„getelementptr8Bq
o
	full_textb
`
^%116 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Pstore8BE
C
	full_text6
4
2store double %115, double* %116, align 8, !tbaa !8
,double8B

	full_text

double %115
.double*8B

	full_text

double* %116
Gfmul8B=
;
	full_text.
,
*%117 = fmul double %63, 0xC0D9C936F46508DE
+double8B

	full_text


double %63
9fmul8B/
-
	full_text 

%118 = fmul double %117, %84
,double8B

	full_text

double %117
+double8B

	full_text


double %84
„getelementptr8Bq
o
	full_textb
`
^%119 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Qstore8BF
D
	full_text7
5
3store double %118, double* %119, align 16, !tbaa !8
,double8B

	full_text

double %118
.double*8B

	full_text

double* %119
9fmul8B/
-
	full_text 

%120 = fmul double %114, %93
,double8B

	full_text

double %114
+double8B

	full_text


double %93
„getelementptr8Bq
o
	full_textb
`
^%121 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
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
‚call8Bx
v
	full_texti
g
e%122 = tail call double @llvm.fmuladd.f64(double %62, double 0x40EDC4C624DD2F1B, double 1.000000e+00)
+double8B

	full_text


double %62
Hfadd8B>
<
	full_text/
-
+%123 = fadd double %122, 0x410FA45000000002
,double8B

	full_text

double %122
„getelementptr8Bq
o
	full_textb
`
^%124 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Qstore8BF
D
	full_text7
5
3store double %123, double* %124, align 16, !tbaa !8
,double8B

	full_text

double %123
.double*8B

	full_text

double* %124
:add8B1
/
	full_text"
 
%125 = add i64 %59, 4294967296
%i648B

	full_text
	
i64 %59
;ashr8B1
/
	full_text"
 
%126 = ashr exact i64 %125, 32
&i648B

	full_text


i64 %125
”getelementptr8B€
~
	full_textq
o
m%127 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %54, i64 %56, i64 %58, i64 %126
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %54
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%128 = load double, double* %127, align 8, !tbaa !8
.double*8B

	full_text

double* %127
:fmul8B0
.
	full_text!

%129 = fmul double %128, %128
,double8B

	full_text

double %128
,double8B

	full_text

double %128
:fmul8B0
.
	full_text!

%130 = fmul double %128, %129
,double8B

	full_text

double %128
,double8B

	full_text

double %129
„getelementptr8Bq
o
	full_textb
`
^%131 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
_store8BT
R
	full_textE
C
Astore double 0xC0E2FC3000000001, double* %131, align 16, !tbaa !8
.double*8B

	full_text

double* %131
„getelementptr8Bq
o
	full_textb
`
^%132 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 1.610000e+02, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
„getelementptr8Bq
o
	full_textb
`
^%133 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %133, align 16, !tbaa !8
.double*8B

	full_text

double* %133
„getelementptr8Bq
o
	full_textb
`
^%134 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
„getelementptr8Bq
o
	full_textb
`
^%135 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %135, align 16, !tbaa !8
.double*8B

	full_text

double* %135
«getelementptr8B—
”
	full_text†
ƒ
€%136 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %126, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%137 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
:fmul8B0
.
	full_text!

%138 = fmul double %128, %137
,double8B

	full_text

double %128
,double8B

	full_text

double %137
Cfsub8B9
7
	full_text*
(
&%139 = fsub double -0.000000e+00, %138
,double8B

	full_text

double %138
”getelementptr8B€
~
	full_textq
o
m%140 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %53, i64 %56, i64 %58, i64 %126
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %53
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%141 = load double, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
Bfmul8B8
6
	full_text)
'
%%142 = fmul double %141, 4.000000e-01
,double8B

	full_text

double %141
:fmul8B0
.
	full_text!

%143 = fmul double %128, %142
,double8B

	full_text

double %128
,double8B

	full_text

double %142
mcall8Bc
a
	full_textT
R
P%144 = tail call double @llvm.fmuladd.f64(double %139, double %138, double %143)
,double8B

	full_text

double %139
,double8B

	full_text

double %138
,double8B

	full_text

double %143
Hfmul8B>
<
	full_text/
-
+%145 = fmul double %129, 0xBFC1111111111111
,double8B

	full_text

double %129
:fmul8B0
.
	full_text!

%146 = fmul double %145, %137
,double8B

	full_text

double %145
,double8B

	full_text

double %137
Hfmul8B>
<
	full_text/
-
+%147 = fmul double %146, 0x40E9504000000001
,double8B

	full_text

double %146
Cfsub8B9
7
	full_text*
(
&%148 = fsub double -0.000000e+00, %147
,double8B

	full_text

double %147
ucall8Bk
i
	full_text\
Z
X%149 = tail call double @llvm.fmuladd.f64(double %144, double 1.610000e+02, double %148)
,double8B

	full_text

double %144
,double8B

	full_text

double %148
„getelementptr8Bq
o
	full_textb
`
^%150 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %149, double* %150, align 8, !tbaa !8
,double8B

	full_text

double %149
.double*8B

	full_text

double* %150
Bfmul8B8
6
	full_text)
'
%%151 = fmul double %138, 1.600000e+00
,double8B

	full_text

double %138
Hfmul8B>
<
	full_text/
-
+%152 = fmul double %128, 0x3FC1111111111111
,double8B

	full_text

double %128
Hfmul8B>
<
	full_text/
-
+%153 = fmul double %152, 0x40E9504000000001
,double8B

	full_text

double %152
Cfsub8B9
7
	full_text*
(
&%154 = fsub double -0.000000e+00, %153
,double8B

	full_text

double %153
ucall8Bk
i
	full_text\
Z
X%155 = tail call double @llvm.fmuladd.f64(double %151, double 1.610000e+02, double %154)
,double8B

	full_text

double %151
,double8B

	full_text

double %154
Hfadd8B>
<
	full_text/
-
+%156 = fadd double %155, 0xC0E2FC3000000001
,double8B

	full_text

double %155
„getelementptr8Bq
o
	full_textb
`
^%157 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %156, double* %157, align 8, !tbaa !8
,double8B

	full_text

double %156
.double*8B

	full_text

double* %157
«getelementptr8B—
”
	full_text†
ƒ
€%158 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %126, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%159 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
:fmul8B0
.
	full_text!

%160 = fmul double %128, %159
,double8B

	full_text

double %128
,double8B

	full_text

double %159
Cfmul8B9
7
	full_text*
(
&%161 = fmul double %160, -4.000000e-01
,double8B

	full_text

double %160
Bfmul8B8
6
	full_text)
'
%%162 = fmul double %161, 1.610000e+02
,double8B

	full_text

double %161
„getelementptr8Bq
o
	full_textb
`
^%163 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
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
«getelementptr8B—
”
	full_text†
ƒ
€%164 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %126, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%165 = load double, double* %164, align 8, !tbaa !8
.double*8B

	full_text

double* %164
:fmul8B0
.
	full_text!

%166 = fmul double %128, %165
,double8B

	full_text

double %128
,double8B

	full_text

double %165
Cfmul8B9
7
	full_text*
(
&%167 = fmul double %166, -4.000000e-01
,double8B

	full_text

double %166
Bfmul8B8
6
	full_text)
'
%%168 = fmul double %167, 1.610000e+02
,double8B

	full_text

double %167
„getelementptr8Bq
o
	full_textb
`
^%169 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %168, double* %169, align 8, !tbaa !8
,double8B

	full_text

double %168
.double*8B

	full_text

double* %169
„getelementptr8Bq
o
	full_textb
`
^%170 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 6.440000e+01, double* %170, align 8, !tbaa !8
.double*8B

	full_text

double* %170
:fmul8B0
.
	full_text!

%171 = fmul double %137, %159
,double8B

	full_text

double %137
,double8B

	full_text

double %159
:fmul8B0
.
	full_text!

%172 = fmul double %129, %171
,double8B

	full_text

double %129
,double8B

	full_text

double %171
Cfsub8B9
7
	full_text*
(
&%173 = fsub double -0.000000e+00, %172
,double8B

	full_text

double %172
Cfmul8B9
7
	full_text*
(
&%174 = fmul double %129, -1.000000e-01
,double8B

	full_text

double %129
:fmul8B0
.
	full_text!

%175 = fmul double %174, %159
,double8B

	full_text

double %174
,double8B

	full_text

double %159
Hfmul8B>
<
	full_text/
-
+%176 = fmul double %175, 0x40E9504000000001
,double8B

	full_text

double %175
Cfsub8B9
7
	full_text*
(
&%177 = fsub double -0.000000e+00, %176
,double8B

	full_text

double %176
ucall8Bk
i
	full_text\
Z
X%178 = tail call double @llvm.fmuladd.f64(double %173, double 1.610000e+02, double %177)
,double8B

	full_text

double %173
,double8B

	full_text

double %177
„getelementptr8Bq
o
	full_textb
`
^%179 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %178, double* %179, align 16, !tbaa !8
,double8B

	full_text

double %178
.double*8B

	full_text

double* %179
Bfmul8B8
6
	full_text)
'
%%180 = fmul double %160, 1.610000e+02
,double8B

	full_text

double %160
„getelementptr8Bq
o
	full_textb
`
^%181 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %180, double* %181, align 8, !tbaa !8
,double8B

	full_text

double %180
.double*8B

	full_text

double* %181
Bfmul8B8
6
	full_text)
'
%%182 = fmul double %128, 1.000000e-01
,double8B

	full_text

double %128
Hfmul8B>
<
	full_text/
-
+%183 = fmul double %182, 0x40E9504000000001
,double8B

	full_text

double %182
Cfsub8B9
7
	full_text*
(
&%184 = fsub double -0.000000e+00, %183
,double8B

	full_text

double %183
ucall8Bk
i
	full_text\
Z
X%185 = tail call double @llvm.fmuladd.f64(double %138, double 1.610000e+02, double %184)
,double8B

	full_text

double %138
,double8B

	full_text

double %184
Hfadd8B>
<
	full_text/
-
+%186 = fadd double %185, 0xC0E2FC3000000001
,double8B

	full_text

double %185
„getelementptr8Bq
o
	full_textb
`
^%187 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %186, double* %187, align 16, !tbaa !8
,double8B

	full_text

double %186
.double*8B

	full_text

double* %187
„getelementptr8Bq
o
	full_textb
`
^%188 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %188, align 8, !tbaa !8
.double*8B

	full_text

double* %188
„getelementptr8Bq
o
	full_textb
`
^%189 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %189, align 16, !tbaa !8
.double*8B

	full_text

double* %189
:fmul8B0
.
	full_text!

%190 = fmul double %137, %165
,double8B

	full_text

double %137
,double8B

	full_text

double %165
:fmul8B0
.
	full_text!

%191 = fmul double %129, %190
,double8B

	full_text

double %129
,double8B

	full_text

double %190
Cfsub8B9
7
	full_text*
(
&%192 = fsub double -0.000000e+00, %191
,double8B

	full_text

double %191
:fmul8B0
.
	full_text!

%193 = fmul double %174, %165
,double8B

	full_text

double %174
,double8B

	full_text

double %165
Hfmul8B>
<
	full_text/
-
+%194 = fmul double %193, 0x40E9504000000001
,double8B

	full_text

double %193
Cfsub8B9
7
	full_text*
(
&%195 = fsub double -0.000000e+00, %194
,double8B

	full_text

double %194
ucall8Bk
i
	full_text\
Z
X%196 = tail call double @llvm.fmuladd.f64(double %192, double 1.610000e+02, double %195)
,double8B

	full_text

double %192
,double8B

	full_text

double %195
„getelementptr8Bq
o
	full_textb
`
^%197 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %196, double* %197, align 8, !tbaa !8
,double8B

	full_text

double %196
.double*8B

	full_text

double* %197
Bfmul8B8
6
	full_text)
'
%%198 = fmul double %166, 1.610000e+02
,double8B

	full_text

double %166
„getelementptr8Bq
o
	full_textb
`
^%199 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %198, double* %199, align 8, !tbaa !8
,double8B

	full_text

double %198
.double*8B

	full_text

double* %199
„getelementptr8Bq
o
	full_textb
`
^%200 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %200, align 8, !tbaa !8
.double*8B

	full_text

double* %200
„getelementptr8Bq
o
	full_textb
`
^%201 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %186, double* %201, align 8, !tbaa !8
,double8B

	full_text

double %186
.double*8B

	full_text

double* %201
„getelementptr8Bq
o
	full_textb
`
^%202 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %202, align 8, !tbaa !8
.double*8B

	full_text

double* %202
«getelementptr8B—
”
	full_text†
ƒ
€%203 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %126, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%204 = load double, double* %203, align 8, !tbaa !8
.double*8B

	full_text

double* %203
Bfmul8B8
6
	full_text)
'
%%205 = fmul double %204, 1.400000e+00
,double8B

	full_text

double %204
Cfsub8B9
7
	full_text*
(
&%206 = fsub double -0.000000e+00, %205
,double8B

	full_text

double %205
ucall8Bk
i
	full_text\
Z
X%207 = tail call double @llvm.fmuladd.f64(double %141, double 8.000000e-01, double %206)
,double8B

	full_text

double %141
,double8B

	full_text

double %206
:fmul8B0
.
	full_text!

%208 = fmul double %129, %137
,double8B

	full_text

double %129
,double8B

	full_text

double %137
:fmul8B0
.
	full_text!

%209 = fmul double %208, %207
,double8B

	full_text

double %208
,double8B

	full_text

double %207
Hfmul8B>
<
	full_text/
-
+%210 = fmul double %130, 0x3FB00AEC33E1F670
,double8B

	full_text

double %130
:fmul8B0
.
	full_text!

%211 = fmul double %137, %137
,double8B

	full_text

double %137
,double8B

	full_text

double %137
Hfmul8B>
<
	full_text/
-
+%212 = fmul double %130, 0xBFB89374BC6A7EF8
,double8B

	full_text

double %130
:fmul8B0
.
	full_text!

%213 = fmul double %159, %159
,double8B

	full_text

double %159
,double8B

	full_text

double %159
:fmul8B0
.
	full_text!

%214 = fmul double %212, %213
,double8B

	full_text

double %212
,double8B

	full_text

double %213
Cfsub8B9
7
	full_text*
(
&%215 = fsub double -0.000000e+00, %214
,double8B

	full_text

double %214
mcall8Bc
a
	full_textT
R
P%216 = tail call double @llvm.fmuladd.f64(double %210, double %211, double %215)
,double8B

	full_text

double %210
,double8B

	full_text

double %211
,double8B

	full_text

double %215
:fmul8B0
.
	full_text!

%217 = fmul double %165, %165
,double8B

	full_text

double %165
,double8B

	full_text

double %165
Cfsub8B9
7
	full_text*
(
&%218 = fsub double -0.000000e+00, %212
,double8B

	full_text

double %212
mcall8Bc
a
	full_textT
R
P%219 = tail call double @llvm.fmuladd.f64(double %218, double %217, double %216)
,double8B

	full_text

double %218
,double8B

	full_text

double %217
,double8B

	full_text

double %216
Hfmul8B>
<
	full_text/
-
+%220 = fmul double %129, 0x3FC916872B020C49
,double8B

	full_text

double %129
Cfsub8B9
7
	full_text*
(
&%221 = fsub double -0.000000e+00, %220
,double8B

	full_text

double %220
mcall8Bc
a
	full_textT
R
P%222 = tail call double @llvm.fmuladd.f64(double %221, double %204, double %219)
,double8B

	full_text

double %221
,double8B

	full_text

double %204
,double8B

	full_text

double %219
Hfmul8B>
<
	full_text/
-
+%223 = fmul double %222, 0x40E9504000000001
,double8B

	full_text

double %222
Cfsub8B9
7
	full_text*
(
&%224 = fsub double -0.000000e+00, %223
,double8B

	full_text

double %223
ucall8Bk
i
	full_text\
Z
X%225 = tail call double @llvm.fmuladd.f64(double %209, double 1.610000e+02, double %224)
,double8B

	full_text

double %209
,double8B

	full_text

double %224
„getelementptr8Bq
o
	full_textb
`
^%226 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %225, double* %226, align 16, !tbaa !8
,double8B

	full_text

double %225
.double*8B

	full_text

double* %226
:fmul8B0
.
	full_text!

%227 = fmul double %128, %204
,double8B

	full_text

double %128
,double8B

	full_text

double %204
:fmul8B0
.
	full_text!

%228 = fmul double %128, %141
,double8B

	full_text

double %128
,double8B

	full_text

double %141
mcall8Bc
a
	full_textT
R
P%229 = tail call double @llvm.fmuladd.f64(double %211, double %129, double %228)
,double8B

	full_text

double %211
,double8B

	full_text

double %129
,double8B

	full_text

double %228
Bfmul8B8
6
	full_text)
'
%%230 = fmul double %229, 4.000000e-01
,double8B

	full_text

double %229
Cfsub8B9
7
	full_text*
(
&%231 = fsub double -0.000000e+00, %230
,double8B

	full_text

double %230
ucall8Bk
i
	full_text\
Z
X%232 = tail call double @llvm.fmuladd.f64(double %227, double 1.400000e+00, double %231)
,double8B

	full_text

double %227
,double8B

	full_text

double %231
Hfmul8B>
<
	full_text/
-
+%233 = fmul double %129, 0xC0A96187D9C54A68
,double8B

	full_text

double %129
:fmul8B0
.
	full_text!

%234 = fmul double %233, %137
,double8B

	full_text

double %233
,double8B

	full_text

double %137
Cfsub8B9
7
	full_text*
(
&%235 = fsub double -0.000000e+00, %234
,double8B

	full_text

double %234
ucall8Bk
i
	full_text\
Z
X%236 = tail call double @llvm.fmuladd.f64(double %232, double 1.610000e+02, double %235)
,double8B

	full_text

double %232
,double8B

	full_text

double %235
„getelementptr8Bq
o
	full_textb
`
^%237 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %236, double* %237, align 8, !tbaa !8
,double8B

	full_text

double %236
.double*8B

	full_text

double* %237
Cfmul8B9
7
	full_text*
(
&%238 = fmul double %171, -4.000000e-01
,double8B

	full_text

double %171
:fmul8B0
.
	full_text!

%239 = fmul double %129, %238
,double8B

	full_text

double %129
,double8B

	full_text

double %238
Hfmul8B>
<
	full_text/
-
+%240 = fmul double %129, 0xC0B370D4FDF3B645
,double8B

	full_text

double %129
:fmul8B0
.
	full_text!

%241 = fmul double %240, %159
,double8B

	full_text

double %240
,double8B

	full_text

double %159
Cfsub8B9
7
	full_text*
(
&%242 = fsub double -0.000000e+00, %241
,double8B

	full_text

double %241
ucall8Bk
i
	full_text\
Z
X%243 = tail call double @llvm.fmuladd.f64(double %239, double 1.610000e+02, double %242)
,double8B

	full_text

double %239
,double8B

	full_text

double %242
„getelementptr8Bq
o
	full_textb
`
^%244 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %243, double* %244, align 16, !tbaa !8
,double8B

	full_text

double %243
.double*8B

	full_text

double* %244
Cfmul8B9
7
	full_text*
(
&%245 = fmul double %190, -4.000000e-01
,double8B

	full_text

double %190
:fmul8B0
.
	full_text!

%246 = fmul double %129, %245
,double8B

	full_text

double %129
,double8B

	full_text

double %245
:fmul8B0
.
	full_text!

%247 = fmul double %240, %165
,double8B

	full_text

double %240
,double8B

	full_text

double %165
Cfsub8B9
7
	full_text*
(
&%248 = fsub double -0.000000e+00, %247
,double8B

	full_text

double %247
ucall8Bk
i
	full_text\
Z
X%249 = tail call double @llvm.fmuladd.f64(double %246, double 1.610000e+02, double %248)
,double8B

	full_text

double %246
,double8B

	full_text

double %248
„getelementptr8Bq
o
	full_textb
`
^%250 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %249, double* %250, align 8, !tbaa !8
,double8B

	full_text

double %249
.double*8B

	full_text

double* %250
Bfmul8B8
6
	full_text)
'
%%251 = fmul double %138, 1.400000e+00
,double8B

	full_text

double %138
Hfmul8B>
<
	full_text/
-
+%252 = fmul double %128, 0x40C3D884189374BD
,double8B

	full_text

double %128
Cfsub8B9
7
	full_text*
(
&%253 = fsub double -0.000000e+00, %252
,double8B

	full_text

double %252
ucall8Bk
i
	full_text\
Z
X%254 = tail call double @llvm.fmuladd.f64(double %251, double 1.610000e+02, double %253)
,double8B

	full_text

double %251
,double8B

	full_text

double %253
Hfadd8B>
<
	full_text/
-
+%255 = fadd double %254, 0xC0E2FC3000000001
,double8B

	full_text

double %254
„getelementptr8Bq
o
	full_textb
`
^%256 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %255, double* %256, align 16, !tbaa !8
,double8B

	full_text

double %255
.double*8B

	full_text

double* %256
:add8B1
/
	full_text"
 
%257 = add i64 %57, 4294967296
%i648B

	full_text
	
i64 %57
;ashr8B1
/
	full_text"
 
%258 = ashr exact i64 %257, 32
&i648B

	full_text


i64 %257
”getelementptr8B€
~
	full_textq
o
m%259 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %54, i64 %56, i64 %258, i64 %60
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %54
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%260 = load double, double* %259, align 8, !tbaa !8
.double*8B

	full_text

double* %259
:fmul8B0
.
	full_text!

%261 = fmul double %260, %260
,double8B

	full_text

double %260
,double8B

	full_text

double %260
:fmul8B0
.
	full_text!

%262 = fmul double %260, %261
,double8B

	full_text

double %260
,double8B

	full_text

double %261
„getelementptr8Bq
o
	full_textb
`
^%263 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
_store8BT
R
	full_textE
C
Astore double 0xC0E2FC3000000001, double* %263, align 16, !tbaa !8
.double*8B

	full_text

double* %263
„getelementptr8Bq
o
	full_textb
`
^%264 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %264, align 8, !tbaa !8
.double*8B

	full_text

double* %264
„getelementptr8Bq
o
	full_textb
`
^%265 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ystore8BN
L
	full_text?
=
;store double 1.610000e+02, double* %265, align 16, !tbaa !8
.double*8B

	full_text

double* %265
„getelementptr8Bq
o
	full_textb
`
^%266 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %266, align 8, !tbaa !8
.double*8B

	full_text

double* %266
„getelementptr8Bq
o
	full_textb
`
^%267 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %267, align 16, !tbaa !8
.double*8B

	full_text

double* %267
«getelementptr8B—
”
	full_text†
ƒ
€%268 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %258, i64 %60, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%269 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
«getelementptr8B—
”
	full_text†
ƒ
€%270 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %258, i64 %60, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%271 = load double, double* %270, align 8, !tbaa !8
.double*8B

	full_text

double* %270
:fmul8B0
.
	full_text!

%272 = fmul double %269, %271
,double8B

	full_text

double %269
,double8B

	full_text

double %271
:fmul8B0
.
	full_text!

%273 = fmul double %261, %272
,double8B

	full_text

double %261
,double8B

	full_text

double %272
Cfsub8B9
7
	full_text*
(
&%274 = fsub double -0.000000e+00, %273
,double8B

	full_text

double %273
Cfmul8B9
7
	full_text*
(
&%275 = fmul double %261, -1.000000e-01
,double8B

	full_text

double %261
:fmul8B0
.
	full_text!

%276 = fmul double %275, %269
,double8B

	full_text

double %275
,double8B

	full_text

double %269
Hfmul8B>
<
	full_text/
-
+%277 = fmul double %276, 0x40E9504000000001
,double8B

	full_text

double %276
Cfsub8B9
7
	full_text*
(
&%278 = fsub double -0.000000e+00, %277
,double8B

	full_text

double %277
ucall8Bk
i
	full_text\
Z
X%279 = tail call double @llvm.fmuladd.f64(double %274, double 1.610000e+02, double %278)
,double8B

	full_text

double %274
,double8B

	full_text

double %278
„getelementptr8Bq
o
	full_textb
`
^%280 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %279, double* %280, align 8, !tbaa !8
,double8B

	full_text

double %279
.double*8B

	full_text

double* %280
:fmul8B0
.
	full_text!

%281 = fmul double %260, %271
,double8B

	full_text

double %260
,double8B

	full_text

double %271
Bfmul8B8
6
	full_text)
'
%%282 = fmul double %260, 1.000000e-01
,double8B

	full_text

double %260
Hfmul8B>
<
	full_text/
-
+%283 = fmul double %282, 0x40E9504000000001
,double8B

	full_text

double %282
Cfsub8B9
7
	full_text*
(
&%284 = fsub double -0.000000e+00, %283
,double8B

	full_text

double %283
ucall8Bk
i
	full_text\
Z
X%285 = tail call double @llvm.fmuladd.f64(double %281, double 1.610000e+02, double %284)
,double8B

	full_text

double %281
,double8B

	full_text

double %284
Hfadd8B>
<
	full_text/
-
+%286 = fadd double %285, 0xC0E2FC3000000001
,double8B

	full_text

double %285
„getelementptr8Bq
o
	full_textb
`
^%287 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %286, double* %287, align 8, !tbaa !8
,double8B

	full_text

double %286
.double*8B

	full_text

double* %287
:fmul8B0
.
	full_text!

%288 = fmul double %260, %269
,double8B

	full_text

double %260
,double8B

	full_text

double %269
Bfmul8B8
6
	full_text)
'
%%289 = fmul double %288, 1.610000e+02
,double8B

	full_text

double %288
„getelementptr8Bq
o
	full_textb
`
^%290 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %289, double* %290, align 8, !tbaa !8
,double8B

	full_text

double %289
.double*8B

	full_text

double* %290
„getelementptr8Bq
o
	full_textb
`
^%291 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %291, align 8, !tbaa !8
.double*8B

	full_text

double* %291
„getelementptr8Bq
o
	full_textb
`
^%292 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %292, align 8, !tbaa !8
.double*8B

	full_text

double* %292
Cfsub8B9
7
	full_text*
(
&%293 = fsub double -0.000000e+00, %281
,double8B

	full_text

double %281
”getelementptr8B€
~
	full_textq
o
m%294 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %53, i64 %56, i64 %258, i64 %60
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %53
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%295 = load double, double* %294, align 8, !tbaa !8
.double*8B

	full_text

double* %294
:fmul8B0
.
	full_text!

%296 = fmul double %260, %295
,double8B

	full_text

double %260
,double8B

	full_text

double %295
Bfmul8B8
6
	full_text)
'
%%297 = fmul double %296, 4.000000e-01
,double8B

	full_text

double %296
mcall8Bc
a
	full_textT
R
P%298 = tail call double @llvm.fmuladd.f64(double %293, double %281, double %297)
,double8B

	full_text

double %293
,double8B

	full_text

double %281
,double8B

	full_text

double %297
Hfmul8B>
<
	full_text/
-
+%299 = fmul double %261, 0xBFC1111111111111
,double8B

	full_text

double %261
:fmul8B0
.
	full_text!

%300 = fmul double %299, %271
,double8B

	full_text

double %299
,double8B

	full_text

double %271
Hfmul8B>
<
	full_text/
-
+%301 = fmul double %300, 0x40E9504000000001
,double8B

	full_text

double %300
Cfsub8B9
7
	full_text*
(
&%302 = fsub double -0.000000e+00, %301
,double8B

	full_text

double %301
ucall8Bk
i
	full_text\
Z
X%303 = tail call double @llvm.fmuladd.f64(double %298, double 1.610000e+02, double %302)
,double8B

	full_text

double %298
,double8B

	full_text

double %302
„getelementptr8Bq
o
	full_textb
`
^%304 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %303, double* %304, align 16, !tbaa !8
,double8B

	full_text

double %303
.double*8B

	full_text

double* %304
Cfmul8B9
7
	full_text*
(
&%305 = fmul double %288, -4.000000e-01
,double8B

	full_text

double %288
Bfmul8B8
6
	full_text)
'
%%306 = fmul double %305, 1.610000e+02
,double8B

	full_text

double %305
„getelementptr8Bq
o
	full_textb
`
^%307 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %306, double* %307, align 8, !tbaa !8
,double8B

	full_text

double %306
.double*8B

	full_text

double* %307
Bfmul8B8
6
	full_text)
'
%%308 = fmul double %281, 1.600000e+00
,double8B

	full_text

double %281
Hfmul8B>
<
	full_text/
-
+%309 = fmul double %260, 0x3FC1111111111111
,double8B

	full_text

double %260
Hfmul8B>
<
	full_text/
-
+%310 = fmul double %309, 0x40E9504000000001
,double8B

	full_text

double %309
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
X%312 = tail call double @llvm.fmuladd.f64(double %308, double 1.610000e+02, double %311)
,double8B

	full_text

double %308
,double8B

	full_text

double %311
Hfadd8B>
<
	full_text/
-
+%313 = fadd double %312, 0xC0E2FC3000000001
,double8B

	full_text

double %312
„getelementptr8Bq
o
	full_textb
`
^%314 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %313, double* %314, align 16, !tbaa !8
,double8B

	full_text

double %313
.double*8B

	full_text

double* %314
«getelementptr8B—
”
	full_text†
ƒ
€%315 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %258, i64 %60, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%316 = load double, double* %315, align 8, !tbaa !8
.double*8B

	full_text

double* %315
:fmul8B0
.
	full_text!

%317 = fmul double %260, %316
,double8B

	full_text

double %260
,double8B

	full_text

double %316
Cfmul8B9
7
	full_text*
(
&%318 = fmul double %317, -4.000000e-01
,double8B

	full_text

double %317
Bfmul8B8
6
	full_text)
'
%%319 = fmul double %318, 1.610000e+02
,double8B

	full_text

double %318
„getelementptr8Bq
o
	full_textb
`
^%320 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %319, double* %320, align 8, !tbaa !8
,double8B

	full_text

double %319
.double*8B

	full_text

double* %320
„getelementptr8Bq
o
	full_textb
`
^%321 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ystore8BN
L
	full_text?
=
;store double 6.440000e+01, double* %321, align 16, !tbaa !8
.double*8B

	full_text

double* %321
:fmul8B0
.
	full_text!

%322 = fmul double %271, %316
,double8B

	full_text

double %271
,double8B

	full_text

double %316
:fmul8B0
.
	full_text!

%323 = fmul double %261, %322
,double8B

	full_text

double %261
,double8B

	full_text

double %322
Cfsub8B9
7
	full_text*
(
&%324 = fsub double -0.000000e+00, %323
,double8B

	full_text

double %323
:fmul8B0
.
	full_text!

%325 = fmul double %275, %316
,double8B

	full_text

double %275
,double8B

	full_text

double %316
Hfmul8B>
<
	full_text/
-
+%326 = fmul double %325, 0x40E9504000000001
,double8B

	full_text

double %325
Cfsub8B9
7
	full_text*
(
&%327 = fsub double -0.000000e+00, %326
,double8B

	full_text

double %326
ucall8Bk
i
	full_text\
Z
X%328 = tail call double @llvm.fmuladd.f64(double %324, double 1.610000e+02, double %327)
,double8B

	full_text

double %324
,double8B

	full_text

double %327
„getelementptr8Bq
o
	full_textb
`
^%329 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %328, double* %329, align 8, !tbaa !8
,double8B

	full_text

double %328
.double*8B

	full_text

double* %329
„getelementptr8Bq
o
	full_textb
`
^%330 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %330, align 8, !tbaa !8
.double*8B

	full_text

double* %330
Bfmul8B8
6
	full_text)
'
%%331 = fmul double %317, 1.610000e+02
,double8B

	full_text

double %317
„getelementptr8Bq
o
	full_textb
`
^%332 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %331, double* %332, align 8, !tbaa !8
,double8B

	full_text

double %331
.double*8B

	full_text

double* %332
„getelementptr8Bq
o
	full_textb
`
^%333 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %286, double* %333, align 8, !tbaa !8
,double8B

	full_text

double %286
.double*8B

	full_text

double* %333
„getelementptr8Bq
o
	full_textb
`
^%334 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %334, align 8, !tbaa !8
.double*8B

	full_text

double* %334
«getelementptr8B—
”
	full_text†
ƒ
€%335 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %56, i64 %258, i64 %60, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%336 = load double, double* %335, align 8, !tbaa !8
.double*8B

	full_text

double* %335
Bfmul8B8
6
	full_text)
'
%%337 = fmul double %336, 1.400000e+00
,double8B

	full_text

double %336
Cfsub8B9
7
	full_text*
(
&%338 = fsub double -0.000000e+00, %337
,double8B

	full_text

double %337
ucall8Bk
i
	full_text\
Z
X%339 = tail call double @llvm.fmuladd.f64(double %295, double 8.000000e-01, double %338)
,double8B

	full_text

double %295
,double8B

	full_text

double %338
:fmul8B0
.
	full_text!

%340 = fmul double %261, %271
,double8B

	full_text

double %261
,double8B

	full_text

double %271
:fmul8B0
.
	full_text!

%341 = fmul double %340, %339
,double8B

	full_text

double %340
,double8B

	full_text

double %339
Hfmul8B>
<
	full_text/
-
+%342 = fmul double %262, 0x3FB89374BC6A7EF8
,double8B

	full_text

double %262
:fmul8B0
.
	full_text!

%343 = fmul double %269, %269
,double8B

	full_text

double %269
,double8B

	full_text

double %269
Hfmul8B>
<
	full_text/
-
+%344 = fmul double %262, 0xBFB00AEC33E1F670
,double8B

	full_text

double %262
:fmul8B0
.
	full_text!

%345 = fmul double %271, %271
,double8B

	full_text

double %271
,double8B

	full_text

double %271
:fmul8B0
.
	full_text!

%346 = fmul double %344, %345
,double8B

	full_text

double %344
,double8B

	full_text

double %345
Cfsub8B9
7
	full_text*
(
&%347 = fsub double -0.000000e+00, %346
,double8B

	full_text

double %346
mcall8Bc
a
	full_textT
R
P%348 = tail call double @llvm.fmuladd.f64(double %342, double %343, double %347)
,double8B

	full_text

double %342
,double8B

	full_text

double %343
,double8B

	full_text

double %347
:fmul8B0
.
	full_text!

%349 = fmul double %316, %316
,double8B

	full_text

double %316
,double8B

	full_text

double %316
mcall8Bc
a
	full_textT
R
P%350 = tail call double @llvm.fmuladd.f64(double %342, double %349, double %348)
,double8B

	full_text

double %342
,double8B

	full_text

double %349
,double8B

	full_text

double %348
Hfmul8B>
<
	full_text/
-
+%351 = fmul double %261, 0x3FC916872B020C49
,double8B

	full_text

double %261
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
P%353 = tail call double @llvm.fmuladd.f64(double %352, double %336, double %350)
,double8B

	full_text

double %352
,double8B

	full_text

double %336
,double8B

	full_text

double %350
Hfmul8B>
<
	full_text/
-
+%354 = fmul double %353, 0x40E9504000000001
,double8B

	full_text

double %353
Cfsub8B9
7
	full_text*
(
&%355 = fsub double -0.000000e+00, %354
,double8B

	full_text

double %354
ucall8Bk
i
	full_text\
Z
X%356 = tail call double @llvm.fmuladd.f64(double %341, double 1.610000e+02, double %355)
,double8B

	full_text

double %341
,double8B

	full_text

double %355
„getelementptr8Bq
o
	full_textb
`
^%357 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %356, double* %357, align 16, !tbaa !8
,double8B

	full_text

double %356
.double*8B

	full_text

double* %357
Cfmul8B9
7
	full_text*
(
&%358 = fmul double %272, -4.000000e-01
,double8B

	full_text

double %272
:fmul8B0
.
	full_text!

%359 = fmul double %261, %358
,double8B

	full_text

double %261
,double8B

	full_text

double %358
Hfmul8B>
<
	full_text/
-
+%360 = fmul double %261, 0xC0B370D4FDF3B645
,double8B

	full_text

double %261
:fmul8B0
.
	full_text!

%361 = fmul double %360, %269
,double8B

	full_text

double %360
,double8B

	full_text

double %269
Cfsub8B9
7
	full_text*
(
&%362 = fsub double -0.000000e+00, %361
,double8B

	full_text

double %361
ucall8Bk
i
	full_text\
Z
X%363 = tail call double @llvm.fmuladd.f64(double %359, double 1.610000e+02, double %362)
,double8B

	full_text

double %359
,double8B

	full_text

double %362
„getelementptr8Bq
o
	full_textb
`
^%364 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %363, double* %364, align 8, !tbaa !8
,double8B

	full_text

double %363
.double*8B

	full_text

double* %364
:fmul8B0
.
	full_text!

%365 = fmul double %260, %336
,double8B

	full_text

double %260
,double8B

	full_text

double %336
:fmul8B0
.
	full_text!

%366 = fmul double %261, %345
,double8B

	full_text

double %261
,double8B

	full_text

double %345
mcall8Bc
a
	full_textT
R
P%367 = tail call double @llvm.fmuladd.f64(double %295, double %260, double %366)
,double8B

	full_text

double %295
,double8B

	full_text

double %260
,double8B

	full_text

double %366
Bfmul8B8
6
	full_text)
'
%%368 = fmul double %367, 4.000000e-01
,double8B

	full_text

double %367
Cfsub8B9
7
	full_text*
(
&%369 = fsub double -0.000000e+00, %368
,double8B

	full_text

double %368
ucall8Bk
i
	full_text\
Z
X%370 = tail call double @llvm.fmuladd.f64(double %365, double 1.400000e+00, double %369)
,double8B

	full_text

double %365
,double8B

	full_text

double %369
Hfmul8B>
<
	full_text/
-
+%371 = fmul double %261, 0xC0A96187D9C54A68
,double8B

	full_text

double %261
:fmul8B0
.
	full_text!

%372 = fmul double %371, %271
,double8B

	full_text

double %371
,double8B

	full_text

double %271
Cfsub8B9
7
	full_text*
(
&%373 = fsub double -0.000000e+00, %372
,double8B

	full_text

double %372
ucall8Bk
i
	full_text\
Z
X%374 = tail call double @llvm.fmuladd.f64(double %370, double 1.610000e+02, double %373)
,double8B

	full_text

double %370
,double8B

	full_text

double %373
„getelementptr8Bq
o
	full_textb
`
^%375 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %374, double* %375, align 16, !tbaa !8
,double8B

	full_text

double %374
.double*8B

	full_text

double* %375
Cfmul8B9
7
	full_text*
(
&%376 = fmul double %322, -4.000000e-01
,double8B

	full_text

double %322
:fmul8B0
.
	full_text!

%377 = fmul double %261, %376
,double8B

	full_text

double %261
,double8B

	full_text

double %376
:fmul8B0
.
	full_text!

%378 = fmul double %360, %316
,double8B

	full_text

double %360
,double8B

	full_text

double %316
Cfsub8B9
7
	full_text*
(
&%379 = fsub double -0.000000e+00, %378
,double8B

	full_text

double %378
ucall8Bk
i
	full_text\
Z
X%380 = tail call double @llvm.fmuladd.f64(double %377, double 1.610000e+02, double %379)
,double8B

	full_text

double %377
,double8B

	full_text

double %379
„getelementptr8Bq
o
	full_textb
`
^%381 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %380, double* %381, align 8, !tbaa !8
,double8B

	full_text

double %380
.double*8B

	full_text

double* %381
Bfmul8B8
6
	full_text)
'
%%382 = fmul double %281, 1.400000e+00
,double8B

	full_text

double %281
Hfmul8B>
<
	full_text/
-
+%383 = fmul double %260, 0x40C3D884189374BD
,double8B

	full_text

double %260
Cfsub8B9
7
	full_text*
(
&%384 = fsub double -0.000000e+00, %383
,double8B

	full_text

double %383
ucall8Bk
i
	full_text\
Z
X%385 = tail call double @llvm.fmuladd.f64(double %382, double 1.610000e+02, double %384)
,double8B

	full_text

double %382
,double8B

	full_text

double %384
Hfadd8B>
<
	full_text/
-
+%386 = fadd double %385, 0xC0E2FC3000000001
,double8B

	full_text

double %385
„getelementptr8Bq
o
	full_textb
`
^%387 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %386, double* %387, align 16, !tbaa !8
,double8B

	full_text

double %386
.double*8B

	full_text

double* %387
:add8B1
/
	full_text"
 
%388 = add i64 %55, 4294967296
%i648B

	full_text
	
i64 %55
;ashr8B1
/
	full_text"
 
%389 = ashr exact i64 %388, 32
&i648B

	full_text


i64 %388
”getelementptr8B€
~
	full_textq
o
m%390 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %54, i64 %389, i64 %58, i64 %60
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %54
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%391 = load double, double* %390, align 8, !tbaa !8
.double*8B

	full_text

double* %390
:fmul8B0
.
	full_text!

%392 = fmul double %391, %391
,double8B

	full_text

double %391
,double8B

	full_text

double %391
:fmul8B0
.
	full_text!

%393 = fmul double %391, %392
,double8B

	full_text

double %391
,double8B

	full_text

double %392
„getelementptr8Bq
o
	full_textb
`
^%394 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
_store8BT
R
	full_textE
C
Astore double 0xC0E9504000000001, double* %394, align 16, !tbaa !8
.double*8B

	full_text

double* %394
„getelementptr8Bq
o
	full_textb
`
^%395 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %395, align 8, !tbaa !8
.double*8B

	full_text

double* %395
„getelementptr8Bq
o
	full_textb
`
^%396 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %396, align 16, !tbaa !8
.double*8B

	full_text

double* %396
„getelementptr8Bq
o
	full_textb
`
^%397 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 1.610000e+02, double* %397, align 8, !tbaa !8
.double*8B

	full_text

double* %397
„getelementptr8Bq
o
	full_textb
`
^%398 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %398, align 16, !tbaa !8
.double*8B

	full_text

double* %398
«getelementptr8B—
”
	full_text†
ƒ
€%399 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %389, i64 %58, i64 %60, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%400 = load double, double* %399, align 8, !tbaa !8
.double*8B

	full_text

double* %399
«getelementptr8B—
”
	full_text†
ƒ
€%401 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %389, i64 %58, i64 %60, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%402 = load double, double* %401, align 8, !tbaa !8
.double*8B

	full_text

double* %401
:fmul8B0
.
	full_text!

%403 = fmul double %400, %402
,double8B

	full_text

double %400
,double8B

	full_text

double %402
:fmul8B0
.
	full_text!

%404 = fmul double %392, %403
,double8B

	full_text

double %392
,double8B

	full_text

double %403
Cfsub8B9
7
	full_text*
(
&%405 = fsub double -0.000000e+00, %404
,double8B

	full_text

double %404
Cfmul8B9
7
	full_text*
(
&%406 = fmul double %392, -1.000000e-01
,double8B

	full_text

double %392
:fmul8B0
.
	full_text!

%407 = fmul double %406, %400
,double8B

	full_text

double %406
,double8B

	full_text

double %400
Hfmul8B>
<
	full_text/
-
+%408 = fmul double %407, 0x40E9504000000001
,double8B

	full_text

double %407
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
X%410 = tail call double @llvm.fmuladd.f64(double %405, double 1.610000e+02, double %409)
,double8B

	full_text

double %405
,double8B

	full_text

double %409
„getelementptr8Bq
o
	full_textb
`
^%411 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %410, double* %411, align 8, !tbaa !8
,double8B

	full_text

double %410
.double*8B

	full_text

double* %411
:fmul8B0
.
	full_text!

%412 = fmul double %391, %402
,double8B

	full_text

double %391
,double8B

	full_text

double %402
Hfmul8B>
<
	full_text/
-
+%413 = fmul double %391, 0x40B4403333333334
,double8B

	full_text

double %391
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
X%415 = tail call double @llvm.fmuladd.f64(double %412, double 1.610000e+02, double %414)
,double8B

	full_text

double %412
,double8B

	full_text

double %414
Hfadd8B>
<
	full_text/
-
+%416 = fadd double %415, 0xC0E9504000000001
,double8B

	full_text

double %415
„getelementptr8Bq
o
	full_textb
`
^%417 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %416, double* %417, align 8, !tbaa !8
,double8B

	full_text

double %416
.double*8B

	full_text

double* %417
„getelementptr8Bq
o
	full_textb
`
^%418 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %418, align 8, !tbaa !8
.double*8B

	full_text

double* %418
:fmul8B0
.
	full_text!

%419 = fmul double %391, %400
,double8B

	full_text

double %391
,double8B

	full_text

double %400
Bfmul8B8
6
	full_text)
'
%%420 = fmul double %419, 1.610000e+02
,double8B

	full_text

double %419
„getelementptr8Bq
o
	full_textb
`
^%421 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %420, double* %421, align 8, !tbaa !8
,double8B

	full_text

double %420
.double*8B

	full_text

double* %421
„getelementptr8Bq
o
	full_textb
`
^%422 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %422, align 8, !tbaa !8
.double*8B

	full_text

double* %422
«getelementptr8B—
”
	full_text†
ƒ
€%423 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %389, i64 %58, i64 %60, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%424 = load double, double* %423, align 8, !tbaa !8
.double*8B

	full_text

double* %423
:fmul8B0
.
	full_text!

%425 = fmul double %402, %424
,double8B

	full_text

double %402
,double8B

	full_text

double %424
:fmul8B0
.
	full_text!

%426 = fmul double %392, %425
,double8B

	full_text

double %392
,double8B

	full_text

double %425
Cfsub8B9
7
	full_text*
(
&%427 = fsub double -0.000000e+00, %426
,double8B

	full_text

double %426
:fmul8B0
.
	full_text!

%428 = fmul double %406, %424
,double8B

	full_text

double %406
,double8B

	full_text

double %424
Hfmul8B>
<
	full_text/
-
+%429 = fmul double %428, 0x40E9504000000001
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
X%431 = tail call double @llvm.fmuladd.f64(double %427, double 1.610000e+02, double %430)
,double8B

	full_text

double %427
,double8B

	full_text

double %430
„getelementptr8Bq
o
	full_textb
`
^%432 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %431, double* %432, align 16, !tbaa !8
,double8B

	full_text

double %431
.double*8B

	full_text

double* %432
„getelementptr8Bq
o
	full_textb
`
^%433 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %433, align 8, !tbaa !8
.double*8B

	full_text

double* %433
Bfmul8B8
6
	full_text)
'
%%434 = fmul double %391, 1.000000e-01
,double8B

	full_text

double %391
Hfmul8B>
<
	full_text/
-
+%435 = fmul double %434, 0x40E9504000000001
,double8B

	full_text

double %434
Cfsub8B9
7
	full_text*
(
&%436 = fsub double -0.000000e+00, %435
,double8B

	full_text

double %435
ucall8Bk
i
	full_text\
Z
X%437 = tail call double @llvm.fmuladd.f64(double %412, double 1.610000e+02, double %436)
,double8B

	full_text

double %412
,double8B

	full_text

double %436
Hfadd8B>
<
	full_text/
-
+%438 = fadd double %437, 0xC0E9504000000001
,double8B

	full_text

double %437
„getelementptr8Bq
o
	full_textb
`
^%439 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %438, double* %439, align 16, !tbaa !8
,double8B

	full_text

double %438
.double*8B

	full_text

double* %439
:fmul8B0
.
	full_text!

%440 = fmul double %391, %424
,double8B

	full_text

double %391
,double8B

	full_text

double %424
Bfmul8B8
6
	full_text)
'
%%441 = fmul double %440, 1.610000e+02
,double8B

	full_text

double %440
„getelementptr8Bq
o
	full_textb
`
^%442 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %441, double* %442, align 8, !tbaa !8
,double8B

	full_text

double %441
.double*8B

	full_text

double* %442
„getelementptr8Bq
o
	full_textb
`
^%443 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %443, align 16, !tbaa !8
.double*8B

	full_text

double* %443
Cfsub8B9
7
	full_text*
(
&%444 = fsub double -0.000000e+00, %412
,double8B

	full_text

double %412
”getelementptr8B€
~
	full_textq
o
m%445 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %53, i64 %389, i64 %58, i64 %60
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %53
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%446 = load double, double* %445, align 8, !tbaa !8
.double*8B

	full_text

double* %445
:fmul8B0
.
	full_text!

%447 = fmul double %391, %446
,double8B

	full_text

double %391
,double8B

	full_text

double %446
Bfmul8B8
6
	full_text)
'
%%448 = fmul double %447, 4.000000e-01
,double8B

	full_text

double %447
mcall8Bc
a
	full_textT
R
P%449 = tail call double @llvm.fmuladd.f64(double %444, double %412, double %448)
,double8B

	full_text

double %444
,double8B

	full_text

double %412
,double8B

	full_text

double %448
Hfmul8B>
<
	full_text/
-
+%450 = fmul double %392, 0xBFC1111111111111
,double8B

	full_text

double %392
:fmul8B0
.
	full_text!

%451 = fmul double %450, %402
,double8B

	full_text

double %450
,double8B

	full_text

double %402
Hfmul8B>
<
	full_text/
-
+%452 = fmul double %451, 0x40E9504000000001
,double8B

	full_text

double %451
Cfsub8B9
7
	full_text*
(
&%453 = fsub double -0.000000e+00, %452
,double8B

	full_text

double %452
ucall8Bk
i
	full_text\
Z
X%454 = tail call double @llvm.fmuladd.f64(double %449, double 1.610000e+02, double %453)
,double8B

	full_text

double %449
,double8B

	full_text

double %453
„getelementptr8Bq
o
	full_textb
`
^%455 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %454, double* %455, align 8, !tbaa !8
,double8B

	full_text

double %454
.double*8B

	full_text

double* %455
Cfmul8B9
7
	full_text*
(
&%456 = fmul double %419, -4.000000e-01
,double8B

	full_text

double %419
Bfmul8B8
6
	full_text)
'
%%457 = fmul double %456, 1.610000e+02
,double8B

	full_text

double %456
„getelementptr8Bq
o
	full_textb
`
^%458 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %457, double* %458, align 8, !tbaa !8
,double8B

	full_text

double %457
.double*8B

	full_text

double* %458
Cfmul8B9
7
	full_text*
(
&%459 = fmul double %440, -4.000000e-01
,double8B

	full_text

double %440
Bfmul8B8
6
	full_text)
'
%%460 = fmul double %459, 1.610000e+02
,double8B

	full_text

double %459
„getelementptr8Bq
o
	full_textb
`
^%461 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %460, double* %461, align 8, !tbaa !8
,double8B

	full_text

double %460
.double*8B

	full_text

double* %461
Hfmul8B>
<
	full_text/
-
+%462 = fmul double %391, 0x3FC1111111111111
,double8B

	full_text

double %391
Hfmul8B>
<
	full_text/
-
+%463 = fmul double %462, 0x40E9504000000001
,double8B

	full_text

double %462
Cfsub8B9
7
	full_text*
(
&%464 = fsub double -0.000000e+00, %463
,double8B

	full_text

double %463
ucall8Bk
i
	full_text\
Z
X%465 = tail call double @llvm.fmuladd.f64(double %412, double 2.576000e+02, double %464)
,double8B

	full_text

double %412
,double8B

	full_text

double %464
Hfadd8B>
<
	full_text/
-
+%466 = fadd double %465, 0xC0E9504000000001
,double8B

	full_text

double %465
„getelementptr8Bq
o
	full_textb
`
^%467 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %466, double* %467, align 8, !tbaa !8
,double8B

	full_text

double %466
.double*8B

	full_text

double* %467
„getelementptr8Bq
o
	full_textb
`
^%468 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 6.440000e+01, double* %468, align 8, !tbaa !8
.double*8B

	full_text

double* %468
«getelementptr8B—
”
	full_text†
ƒ
€%469 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %52, i64 %389, i64 %58, i64 %60, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %52
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%470 = load double, double* %469, align 8, !tbaa !8
.double*8B

	full_text

double* %469
Bfmul8B8
6
	full_text)
'
%%471 = fmul double %470, 1.400000e+00
,double8B

	full_text

double %470
Cfsub8B9
7
	full_text*
(
&%472 = fsub double -0.000000e+00, %471
,double8B

	full_text

double %471
ucall8Bk
i
	full_text\
Z
X%473 = tail call double @llvm.fmuladd.f64(double %446, double 8.000000e-01, double %472)
,double8B

	full_text

double %446
,double8B

	full_text

double %472
:fmul8B0
.
	full_text!

%474 = fmul double %392, %402
,double8B

	full_text

double %392
,double8B

	full_text

double %402
:fmul8B0
.
	full_text!

%475 = fmul double %474, %473
,double8B

	full_text

double %474
,double8B

	full_text

double %473
Hfmul8B>
<
	full_text/
-
+%476 = fmul double %393, 0x3FB89374BC6A7EF8
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%477 = fmul double %400, %400
,double8B

	full_text

double %400
,double8B

	full_text

double %400
Hfmul8B>
<
	full_text/
-
+%478 = fmul double %393, 0xBFB89374BC6A7EF8
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%479 = fmul double %424, %424
,double8B

	full_text

double %424
,double8B

	full_text

double %424
:fmul8B0
.
	full_text!

%480 = fmul double %478, %479
,double8B

	full_text

double %478
,double8B

	full_text

double %479
Cfsub8B9
7
	full_text*
(
&%481 = fsub double -0.000000e+00, %480
,double8B

	full_text

double %480
mcall8Bc
a
	full_textT
R
P%482 = tail call double @llvm.fmuladd.f64(double %476, double %477, double %481)
,double8B

	full_text

double %476
,double8B

	full_text

double %477
,double8B

	full_text

double %481
Hfmul8B>
<
	full_text/
-
+%483 = fmul double %393, 0x3FB00AEC33E1F670
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%484 = fmul double %402, %402
,double8B

	full_text

double %402
,double8B

	full_text

double %402
mcall8Bc
a
	full_textT
R
P%485 = tail call double @llvm.fmuladd.f64(double %483, double %484, double %482)
,double8B

	full_text

double %483
,double8B

	full_text

double %484
,double8B

	full_text

double %482
Hfmul8B>
<
	full_text/
-
+%486 = fmul double %392, 0x3FC916872B020C49
,double8B

	full_text

double %392
Cfsub8B9
7
	full_text*
(
&%487 = fsub double -0.000000e+00, %486
,double8B

	full_text

double %486
mcall8Bc
a
	full_textT
R
P%488 = tail call double @llvm.fmuladd.f64(double %487, double %470, double %485)
,double8B

	full_text

double %487
,double8B

	full_text

double %470
,double8B

	full_text

double %485
Hfmul8B>
<
	full_text/
-
+%489 = fmul double %488, 0x40E9504000000001
,double8B

	full_text

double %488
Cfsub8B9
7
	full_text*
(
&%490 = fsub double -0.000000e+00, %489
,double8B

	full_text

double %489
ucall8Bk
i
	full_text\
Z
X%491 = tail call double @llvm.fmuladd.f64(double %475, double 1.610000e+02, double %490)
,double8B

	full_text

double %475
,double8B

	full_text

double %490
„getelementptr8Bq
o
	full_textb
`
^%492 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %491, double* %492, align 16, !tbaa !8
,double8B

	full_text

double %491
.double*8B

	full_text

double* %492
Cfmul8B9
7
	full_text*
(
&%493 = fmul double %403, -4.000000e-01
,double8B

	full_text

double %403
:fmul8B0
.
	full_text!

%494 = fmul double %392, %493
,double8B

	full_text

double %392
,double8B

	full_text

double %493
Hfmul8B>
<
	full_text/
-
+%495 = fmul double %392, 0xC0B370D4FDF3B645
,double8B

	full_text

double %392
:fmul8B0
.
	full_text!

%496 = fmul double %495, %400
,double8B

	full_text

double %495
,double8B

	full_text

double %400
Cfsub8B9
7
	full_text*
(
&%497 = fsub double -0.000000e+00, %496
,double8B

	full_text

double %496
ucall8Bk
i
	full_text\
Z
X%498 = tail call double @llvm.fmuladd.f64(double %494, double 1.610000e+02, double %497)
,double8B

	full_text

double %494
,double8B

	full_text

double %497
„getelementptr8Bq
o
	full_textb
`
^%499 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %498, double* %499, align 8, !tbaa !8
,double8B

	full_text

double %498
.double*8B

	full_text

double* %499
Cfmul8B9
7
	full_text*
(
&%500 = fmul double %425, -4.000000e-01
,double8B

	full_text

double %425
:fmul8B0
.
	full_text!

%501 = fmul double %392, %500
,double8B

	full_text

double %392
,double8B

	full_text

double %500
:fmul8B0
.
	full_text!

%502 = fmul double %495, %424
,double8B

	full_text

double %495
,double8B

	full_text

double %424
Cfsub8B9
7
	full_text*
(
&%503 = fsub double -0.000000e+00, %502
,double8B

	full_text

double %502
ucall8Bk
i
	full_text\
Z
X%504 = tail call double @llvm.fmuladd.f64(double %501, double 1.610000e+02, double %503)
,double8B

	full_text

double %501
,double8B

	full_text

double %503
„getelementptr8Bq
o
	full_textb
`
^%505 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %504, double* %505, align 16, !tbaa !8
,double8B

	full_text

double %504
.double*8B

	full_text

double* %505
:fmul8B0
.
	full_text!

%506 = fmul double %391, %470
,double8B

	full_text

double %391
,double8B

	full_text

double %470
:fmul8B0
.
	full_text!

%507 = fmul double %392, %484
,double8B

	full_text

double %392
,double8B

	full_text

double %484
mcall8Bc
a
	full_textT
R
P%508 = tail call double @llvm.fmuladd.f64(double %446, double %391, double %507)
,double8B

	full_text

double %446
,double8B

	full_text

double %391
,double8B

	full_text

double %507
Bfmul8B8
6
	full_text)
'
%%509 = fmul double %508, 4.000000e-01
,double8B

	full_text

double %508
Cfsub8B9
7
	full_text*
(
&%510 = fsub double -0.000000e+00, %509
,double8B

	full_text

double %509
ucall8Bk
i
	full_text\
Z
X%511 = tail call double @llvm.fmuladd.f64(double %506, double 1.400000e+00, double %510)
,double8B

	full_text

double %506
,double8B

	full_text

double %510
Hfmul8B>
<
	full_text/
-
+%512 = fmul double %392, 0xC0A96187D9C54A68
,double8B

	full_text

double %392
:fmul8B0
.
	full_text!

%513 = fmul double %512, %402
,double8B

	full_text

double %512
,double8B

	full_text

double %402
Cfsub8B9
7
	full_text*
(
&%514 = fsub double -0.000000e+00, %513
,double8B

	full_text

double %513
ucall8Bk
i
	full_text\
Z
X%515 = tail call double @llvm.fmuladd.f64(double %511, double 1.610000e+02, double %514)
,double8B

	full_text

double %511
,double8B

	full_text

double %514
„getelementptr8Bq
o
	full_textb
`
^%516 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %515, double* %516, align 8, !tbaa !8
,double8B

	full_text

double %515
.double*8B

	full_text

double* %516
Bfmul8B8
6
	full_text)
'
%%517 = fmul double %412, 1.400000e+00
,double8B

	full_text

double %412
Hfmul8B>
<
	full_text/
-
+%518 = fmul double %391, 0x40C3D884189374BD
,double8B

	full_text

double %391
Cfsub8B9
7
	full_text*
(
&%519 = fsub double -0.000000e+00, %518
,double8B

	full_text

double %518
ucall8Bk
i
	full_text\
Z
X%520 = tail call double @llvm.fmuladd.f64(double %517, double 1.610000e+02, double %519)
,double8B

	full_text

double %517
,double8B

	full_text

double %519
Hfadd8B>
<
	full_text/
-
+%521 = fadd double %520, 0xC0E9504000000001
,double8B

	full_text

double %520
„getelementptr8Bq
o
	full_textb
`
^%522 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %521, double* %522, align 16, !tbaa !8
,double8B

	full_text

double %521
.double*8B

	full_text

double* %522
«getelementptr8B—
”
	full_text†
ƒ
€%523 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %389, i64 %58, i64 %60, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%524 = load double, double* %523, align 8, !tbaa !8
.double*8B

	full_text

double* %523
«getelementptr8B—
”
	full_text†
ƒ
€%525 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %389, i64 %58, i64 %60, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%526 = load double, double* %525, align 8, !tbaa !8
.double*8B

	full_text

double* %525
«getelementptr8B—
”
	full_text†
ƒ
€%527 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %389, i64 %58, i64 %60, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%528 = load double, double* %527, align 8, !tbaa !8
.double*8B

	full_text

double* %527
«getelementptr8B—
”
	full_text†
ƒ
€%529 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %389, i64 %58, i64 %60, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%530 = load double, double* %529, align 8, !tbaa !8
.double*8B

	full_text

double* %529
«getelementptr8B—
”
	full_text†
ƒ
€%531 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %389, i64 %58, i64 %60, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
&i648B

	full_text


i64 %389
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%532 = load double, double* %531, align 8, !tbaa !8
.double*8B

	full_text

double* %531
Qload8BG
E
	full_text8
6
4%533 = load double, double* %394, align 16, !tbaa !8
.double*8B

	full_text

double* %394
Pload8BF
D
	full_text7
5
3%534 = load double, double* %395, align 8, !tbaa !8
.double*8B

	full_text

double* %395
:fmul8B0
.
	full_text!

%535 = fmul double %534, %526
,double8B

	full_text

double %534
,double8B

	full_text

double %526
mcall8Bc
a
	full_textT
R
P%536 = tail call double @llvm.fmuladd.f64(double %533, double %524, double %535)
,double8B

	full_text

double %533
,double8B

	full_text

double %524
,double8B

	full_text

double %535
Qload8BG
E
	full_text8
6
4%537 = load double, double* %396, align 16, !tbaa !8
.double*8B

	full_text

double* %396
mcall8Bc
a
	full_textT
R
P%538 = tail call double @llvm.fmuladd.f64(double %537, double %528, double %536)
,double8B

	full_text

double %537
,double8B

	full_text

double %528
,double8B

	full_text

double %536
Pload8BF
D
	full_text7
5
3%539 = load double, double* %397, align 8, !tbaa !8
.double*8B

	full_text

double* %397
mcall8Bc
a
	full_textT
R
P%540 = tail call double @llvm.fmuladd.f64(double %539, double %530, double %538)
,double8B

	full_text

double %539
,double8B

	full_text

double %530
,double8B

	full_text

double %538
Qload8BG
E
	full_text8
6
4%541 = load double, double* %398, align 16, !tbaa !8
.double*8B

	full_text

double* %398
mcall8Bc
a
	full_textT
R
P%542 = tail call double @llvm.fmuladd.f64(double %541, double %532, double %540)
,double8B

	full_text

double %541
,double8B

	full_text

double %532
,double8B

	full_text

double %540
Bfmul8B8
6
	full_text)
'
%%543 = fmul double %542, 1.200000e+00
,double8B

	full_text

double %542
qgetelementptr8B^
\
	full_textO
M
K%544 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Pload8BF
D
	full_text7
5
3%545 = load double, double* %411, align 8, !tbaa !8
.double*8B

	full_text

double* %411
Pload8BF
D
	full_text7
5
3%546 = load double, double* %417, align 8, !tbaa !8
.double*8B

	full_text

double* %417
:fmul8B0
.
	full_text!

%547 = fmul double %546, %526
,double8B

	full_text

double %546
,double8B

	full_text

double %526
mcall8Bc
a
	full_textT
R
P%548 = tail call double @llvm.fmuladd.f64(double %545, double %524, double %547)
,double8B

	full_text

double %545
,double8B

	full_text

double %524
,double8B

	full_text

double %547
Pload8BF
D
	full_text7
5
3%549 = load double, double* %418, align 8, !tbaa !8
.double*8B

	full_text

double* %418
mcall8Bc
a
	full_textT
R
P%550 = tail call double @llvm.fmuladd.f64(double %549, double %528, double %548)
,double8B

	full_text

double %549
,double8B

	full_text

double %528
,double8B

	full_text

double %548
Pload8BF
D
	full_text7
5
3%551 = load double, double* %421, align 8, !tbaa !8
.double*8B

	full_text

double* %421
mcall8Bc
a
	full_textT
R
P%552 = tail call double @llvm.fmuladd.f64(double %551, double %530, double %550)
,double8B

	full_text

double %551
,double8B

	full_text

double %530
,double8B

	full_text

double %550
Pload8BF
D
	full_text7
5
3%553 = load double, double* %422, align 8, !tbaa !8
.double*8B

	full_text

double* %422
mcall8Bc
a
	full_textT
R
P%554 = tail call double @llvm.fmuladd.f64(double %553, double %532, double %552)
,double8B

	full_text

double %553
,double8B

	full_text

double %532
,double8B

	full_text

double %552
Bfmul8B8
6
	full_text)
'
%%555 = fmul double %554, 1.200000e+00
,double8B

	full_text

double %554
qgetelementptr8B^
\
	full_textO
M
K%556 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qload8BG
E
	full_text8
6
4%557 = load double, double* %432, align 16, !tbaa !8
.double*8B

	full_text

double* %432
Pload8BF
D
	full_text7
5
3%558 = load double, double* %433, align 8, !tbaa !8
.double*8B

	full_text

double* %433
:fmul8B0
.
	full_text!

%559 = fmul double %558, %526
,double8B

	full_text

double %558
,double8B

	full_text

double %526
mcall8Bc
a
	full_textT
R
P%560 = tail call double @llvm.fmuladd.f64(double %557, double %524, double %559)
,double8B

	full_text

double %557
,double8B

	full_text

double %524
,double8B

	full_text

double %559
Qload8BG
E
	full_text8
6
4%561 = load double, double* %439, align 16, !tbaa !8
.double*8B

	full_text

double* %439
mcall8Bc
a
	full_textT
R
P%562 = tail call double @llvm.fmuladd.f64(double %561, double %528, double %560)
,double8B

	full_text

double %561
,double8B

	full_text

double %528
,double8B

	full_text

double %560
Pload8BF
D
	full_text7
5
3%563 = load double, double* %442, align 8, !tbaa !8
.double*8B

	full_text

double* %442
mcall8Bc
a
	full_textT
R
P%564 = tail call double @llvm.fmuladd.f64(double %563, double %530, double %562)
,double8B

	full_text

double %563
,double8B

	full_text

double %530
,double8B

	full_text

double %562
Qload8BG
E
	full_text8
6
4%565 = load double, double* %443, align 16, !tbaa !8
.double*8B

	full_text

double* %443
mcall8Bc
a
	full_textT
R
P%566 = tail call double @llvm.fmuladd.f64(double %565, double %532, double %564)
,double8B

	full_text

double %565
,double8B

	full_text

double %532
,double8B

	full_text

double %564
Bfmul8B8
6
	full_text)
'
%%567 = fmul double %566, 1.200000e+00
,double8B

	full_text

double %566
qgetelementptr8B^
\
	full_textO
M
K%568 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %567, double* %568, align 16, !tbaa !8
,double8B

	full_text

double %567
.double*8B

	full_text

double* %568
Pload8BF
D
	full_text7
5
3%569 = load double, double* %455, align 8, !tbaa !8
.double*8B

	full_text

double* %455
Pload8BF
D
	full_text7
5
3%570 = load double, double* %458, align 8, !tbaa !8
.double*8B

	full_text

double* %458
:fmul8B0
.
	full_text!

%571 = fmul double %570, %526
,double8B

	full_text

double %570
,double8B

	full_text

double %526
mcall8Bc
a
	full_textT
R
P%572 = tail call double @llvm.fmuladd.f64(double %569, double %524, double %571)
,double8B

	full_text

double %569
,double8B

	full_text

double %524
,double8B

	full_text

double %571
Pload8BF
D
	full_text7
5
3%573 = load double, double* %461, align 8, !tbaa !8
.double*8B

	full_text

double* %461
mcall8Bc
a
	full_textT
R
P%574 = tail call double @llvm.fmuladd.f64(double %573, double %528, double %572)
,double8B

	full_text

double %573
,double8B

	full_text

double %528
,double8B

	full_text

double %572
Pload8BF
D
	full_text7
5
3%575 = load double, double* %467, align 8, !tbaa !8
.double*8B

	full_text

double* %467
mcall8Bc
a
	full_textT
R
P%576 = tail call double @llvm.fmuladd.f64(double %575, double %530, double %574)
,double8B

	full_text

double %575
,double8B

	full_text

double %530
,double8B

	full_text

double %574
Pload8BF
D
	full_text7
5
3%577 = load double, double* %468, align 8, !tbaa !8
.double*8B

	full_text

double* %468
mcall8Bc
a
	full_textT
R
P%578 = tail call double @llvm.fmuladd.f64(double %577, double %532, double %576)
,double8B

	full_text

double %577
,double8B

	full_text

double %532
,double8B

	full_text

double %576
Bfmul8B8
6
	full_text)
'
%%579 = fmul double %578, 1.200000e+00
,double8B

	full_text

double %578
qgetelementptr8B^
\
	full_textO
M
K%580 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Pstore8BE
C
	full_text6
4
2store double %579, double* %580, align 8, !tbaa !8
,double8B

	full_text

double %579
.double*8B

	full_text

double* %580
:fmul8B0
.
	full_text!

%581 = fmul double %498, %526
,double8B

	full_text

double %498
,double8B

	full_text

double %526
mcall8Bc
a
	full_textT
R
P%582 = tail call double @llvm.fmuladd.f64(double %491, double %524, double %581)
,double8B

	full_text

double %491
,double8B

	full_text

double %524
,double8B

	full_text

double %581
mcall8Bc
a
	full_textT
R
P%583 = tail call double @llvm.fmuladd.f64(double %504, double %528, double %582)
,double8B

	full_text

double %504
,double8B

	full_text

double %528
,double8B

	full_text

double %582
mcall8Bc
a
	full_textT
R
P%584 = tail call double @llvm.fmuladd.f64(double %515, double %530, double %583)
,double8B

	full_text

double %515
,double8B

	full_text

double %530
,double8B

	full_text

double %583
mcall8Bc
a
	full_textT
R
P%585 = tail call double @llvm.fmuladd.f64(double %521, double %532, double %584)
,double8B

	full_text

double %521
,double8B

	full_text

double %532
,double8B

	full_text

double %584
Bfmul8B8
6
	full_text)
'
%%586 = fmul double %585, 1.200000e+00
,double8B

	full_text

double %585
qgetelementptr8B^
\
	full_textO
M
K%587 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %586, double* %587, align 16, !tbaa !8
,double8B

	full_text

double %586
.double*8B

	full_text

double* %587
«getelementptr8B—
”
	full_text†
ƒ
€%588 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %258, i64 %60, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%589 = load double, double* %588, align 8, !tbaa !8
.double*8B

	full_text

double* %588
«getelementptr8B—
”
	full_text†
ƒ
€%590 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %126, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%591 = load double, double* %590, align 8, !tbaa !8
.double*8B

	full_text

double* %590
«getelementptr8B—
”
	full_text†
ƒ
€%592 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %258, i64 %60, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%593 = load double, double* %592, align 8, !tbaa !8
.double*8B

	full_text

double* %592
«getelementptr8B—
”
	full_text†
ƒ
€%594 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %126, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%595 = load double, double* %594, align 8, !tbaa !8
.double*8B

	full_text

double* %594
«getelementptr8B—
”
	full_text†
ƒ
€%596 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %258, i64 %60, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%597 = load double, double* %596, align 8, !tbaa !8
.double*8B

	full_text

double* %596
«getelementptr8B—
”
	full_text†
ƒ
€%598 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %126, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%599 = load double, double* %598, align 8, !tbaa !8
.double*8B

	full_text

double* %598
«getelementptr8B—
”
	full_text†
ƒ
€%600 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %258, i64 %60, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%601 = load double, double* %600, align 8, !tbaa !8
.double*8B

	full_text

double* %600
«getelementptr8B—
”
	full_text†
ƒ
€%602 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %126, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%603 = load double, double* %602, align 8, !tbaa !8
.double*8B

	full_text

double* %602
«getelementptr8B—
”
	full_text†
ƒ
€%604 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %258, i64 %60, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %258
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%605 = load double, double* %604, align 8, !tbaa !8
.double*8B

	full_text

double* %604
«getelementptr8B—
”
	full_text†
ƒ
€%606 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %126, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
&i648B

	full_text


i64 %126
Pload8BF
D
	full_text7
5
3%607 = load double, double* %606, align 8, !tbaa !8
.double*8B

	full_text

double* %606
Qload8BG
E
	full_text8
6
4%608 = load double, double* %263, align 16, !tbaa !8
.double*8B

	full_text

double* %263
Qload8BG
E
	full_text8
6
4%609 = load double, double* %131, align 16, !tbaa !8
.double*8B

	full_text

double* %131
:fmul8B0
.
	full_text!

%610 = fmul double %609, %591
,double8B

	full_text

double %609
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%611 = tail call double @llvm.fmuladd.f64(double %608, double %589, double %610)
,double8B

	full_text

double %608
,double8B

	full_text

double %589
,double8B

	full_text

double %610
Pload8BF
D
	full_text7
5
3%612 = load double, double* %264, align 8, !tbaa !8
.double*8B

	full_text

double* %264
mcall8Bc
a
	full_textT
R
P%613 = tail call double @llvm.fmuladd.f64(double %612, double %593, double %611)
,double8B

	full_text

double %612
,double8B

	full_text

double %593
,double8B

	full_text

double %611
Pload8BF
D
	full_text7
5
3%614 = load double, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
mcall8Bc
a
	full_textT
R
P%615 = tail call double @llvm.fmuladd.f64(double %614, double %595, double %613)
,double8B

	full_text

double %614
,double8B

	full_text

double %595
,double8B

	full_text

double %613
Qload8BG
E
	full_text8
6
4%616 = load double, double* %265, align 16, !tbaa !8
.double*8B

	full_text

double* %265
mcall8Bc
a
	full_textT
R
P%617 = tail call double @llvm.fmuladd.f64(double %616, double %597, double %615)
,double8B

	full_text

double %616
,double8B

	full_text

double %597
,double8B

	full_text

double %615
Qload8BG
E
	full_text8
6
4%618 = load double, double* %133, align 16, !tbaa !8
.double*8B

	full_text

double* %133
mcall8Bc
a
	full_textT
R
P%619 = tail call double @llvm.fmuladd.f64(double %618, double %599, double %617)
,double8B

	full_text

double %618
,double8B

	full_text

double %599
,double8B

	full_text

double %617
Pload8BF
D
	full_text7
5
3%620 = load double, double* %266, align 8, !tbaa !8
.double*8B

	full_text

double* %266
mcall8Bc
a
	full_textT
R
P%621 = tail call double @llvm.fmuladd.f64(double %620, double %601, double %619)
,double8B

	full_text

double %620
,double8B

	full_text

double %601
,double8B

	full_text

double %619
Pload8BF
D
	full_text7
5
3%622 = load double, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
mcall8Bc
a
	full_textT
R
P%623 = tail call double @llvm.fmuladd.f64(double %622, double %603, double %621)
,double8B

	full_text

double %622
,double8B

	full_text

double %603
,double8B

	full_text

double %621
Qload8BG
E
	full_text8
6
4%624 = load double, double* %267, align 16, !tbaa !8
.double*8B

	full_text

double* %267
mcall8Bc
a
	full_textT
R
P%625 = tail call double @llvm.fmuladd.f64(double %624, double %605, double %623)
,double8B

	full_text

double %624
,double8B

	full_text

double %605
,double8B

	full_text

double %623
Qload8BG
E
	full_text8
6
4%626 = load double, double* %135, align 16, !tbaa !8
.double*8B

	full_text

double* %135
mcall8Bc
a
	full_textT
R
P%627 = tail call double @llvm.fmuladd.f64(double %626, double %607, double %625)
,double8B

	full_text

double %626
,double8B

	full_text

double %607
,double8B

	full_text

double %625
ucall8Bk
i
	full_text\
Z
X%628 = tail call double @llvm.fmuladd.f64(double %627, double 1.200000e+00, double %543)
,double8B

	full_text

double %627
,double8B

	full_text

double %543
Qstore8BF
D
	full_text7
5
3store double %628, double* %544, align 16, !tbaa !8
,double8B

	full_text

double %628
.double*8B

	full_text

double* %544
Pload8BF
D
	full_text7
5
3%629 = load double, double* %280, align 8, !tbaa !8
.double*8B

	full_text

double* %280
Pload8BF
D
	full_text7
5
3%630 = load double, double* %150, align 8, !tbaa !8
.double*8B

	full_text

double* %150
:fmul8B0
.
	full_text!

%631 = fmul double %630, %591
,double8B

	full_text

double %630
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%632 = tail call double @llvm.fmuladd.f64(double %629, double %589, double %631)
,double8B

	full_text

double %629
,double8B

	full_text

double %589
,double8B

	full_text

double %631
Pload8BF
D
	full_text7
5
3%633 = load double, double* %287, align 8, !tbaa !8
.double*8B

	full_text

double* %287
mcall8Bc
a
	full_textT
R
P%634 = tail call double @llvm.fmuladd.f64(double %633, double %593, double %632)
,double8B

	full_text

double %633
,double8B

	full_text

double %593
,double8B

	full_text

double %632
Pload8BF
D
	full_text7
5
3%635 = load double, double* %157, align 8, !tbaa !8
.double*8B

	full_text

double* %157
mcall8Bc
a
	full_textT
R
P%636 = tail call double @llvm.fmuladd.f64(double %635, double %595, double %634)
,double8B

	full_text

double %635
,double8B

	full_text

double %595
,double8B

	full_text

double %634
Pload8BF
D
	full_text7
5
3%637 = load double, double* %290, align 8, !tbaa !8
.double*8B

	full_text

double* %290
mcall8Bc
a
	full_textT
R
P%638 = tail call double @llvm.fmuladd.f64(double %637, double %597, double %636)
,double8B

	full_text

double %637
,double8B

	full_text

double %597
,double8B

	full_text

double %636
Pload8BF
D
	full_text7
5
3%639 = load double, double* %163, align 8, !tbaa !8
.double*8B

	full_text

double* %163
mcall8Bc
a
	full_textT
R
P%640 = tail call double @llvm.fmuladd.f64(double %639, double %599, double %638)
,double8B

	full_text

double %639
,double8B

	full_text

double %599
,double8B

	full_text

double %638
Pload8BF
D
	full_text7
5
3%641 = load double, double* %291, align 8, !tbaa !8
.double*8B

	full_text

double* %291
mcall8Bc
a
	full_textT
R
P%642 = tail call double @llvm.fmuladd.f64(double %641, double %601, double %640)
,double8B

	full_text

double %641
,double8B

	full_text

double %601
,double8B

	full_text

double %640
Pload8BF
D
	full_text7
5
3%643 = load double, double* %169, align 8, !tbaa !8
.double*8B

	full_text

double* %169
mcall8Bc
a
	full_textT
R
P%644 = tail call double @llvm.fmuladd.f64(double %643, double %603, double %642)
,double8B

	full_text

double %643
,double8B

	full_text

double %603
,double8B

	full_text

double %642
Pload8BF
D
	full_text7
5
3%645 = load double, double* %292, align 8, !tbaa !8
.double*8B

	full_text

double* %292
mcall8Bc
a
	full_textT
R
P%646 = tail call double @llvm.fmuladd.f64(double %645, double %605, double %644)
,double8B

	full_text

double %645
,double8B

	full_text

double %605
,double8B

	full_text

double %644
Pload8BF
D
	full_text7
5
3%647 = load double, double* %170, align 8, !tbaa !8
.double*8B

	full_text

double* %170
mcall8Bc
a
	full_textT
R
P%648 = tail call double @llvm.fmuladd.f64(double %647, double %607, double %646)
,double8B

	full_text

double %647
,double8B

	full_text

double %607
,double8B

	full_text

double %646
ucall8Bk
i
	full_text\
Z
X%649 = tail call double @llvm.fmuladd.f64(double %648, double 1.200000e+00, double %555)
,double8B

	full_text

double %648
,double8B

	full_text

double %555
Pstore8BE
C
	full_text6
4
2store double %649, double* %556, align 8, !tbaa !8
,double8B

	full_text

double %649
.double*8B

	full_text

double* %556
Qload8BG
E
	full_text8
6
4%650 = load double, double* %304, align 16, !tbaa !8
.double*8B

	full_text

double* %304
Qload8BG
E
	full_text8
6
4%651 = load double, double* %179, align 16, !tbaa !8
.double*8B

	full_text

double* %179
:fmul8B0
.
	full_text!

%652 = fmul double %651, %591
,double8B

	full_text

double %651
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%653 = tail call double @llvm.fmuladd.f64(double %650, double %589, double %652)
,double8B

	full_text

double %650
,double8B

	full_text

double %589
,double8B

	full_text

double %652
Pload8BF
D
	full_text7
5
3%654 = load double, double* %307, align 8, !tbaa !8
.double*8B

	full_text

double* %307
mcall8Bc
a
	full_textT
R
P%655 = tail call double @llvm.fmuladd.f64(double %654, double %593, double %653)
,double8B

	full_text

double %654
,double8B

	full_text

double %593
,double8B

	full_text

double %653
Pload8BF
D
	full_text7
5
3%656 = load double, double* %181, align 8, !tbaa !8
.double*8B

	full_text

double* %181
mcall8Bc
a
	full_textT
R
P%657 = tail call double @llvm.fmuladd.f64(double %656, double %595, double %655)
,double8B

	full_text

double %656
,double8B

	full_text

double %595
,double8B

	full_text

double %655
Qload8BG
E
	full_text8
6
4%658 = load double, double* %314, align 16, !tbaa !8
.double*8B

	full_text

double* %314
mcall8Bc
a
	full_textT
R
P%659 = tail call double @llvm.fmuladd.f64(double %658, double %597, double %657)
,double8B

	full_text

double %658
,double8B

	full_text

double %597
,double8B

	full_text

double %657
Qload8BG
E
	full_text8
6
4%660 = load double, double* %187, align 16, !tbaa !8
.double*8B

	full_text

double* %187
mcall8Bc
a
	full_textT
R
P%661 = tail call double @llvm.fmuladd.f64(double %660, double %599, double %659)
,double8B

	full_text

double %660
,double8B

	full_text

double %599
,double8B

	full_text

double %659
Pload8BF
D
	full_text7
5
3%662 = load double, double* %320, align 8, !tbaa !8
.double*8B

	full_text

double* %320
mcall8Bc
a
	full_textT
R
P%663 = tail call double @llvm.fmuladd.f64(double %662, double %601, double %661)
,double8B

	full_text

double %662
,double8B

	full_text

double %601
,double8B

	full_text

double %661
Pload8BF
D
	full_text7
5
3%664 = load double, double* %188, align 8, !tbaa !8
.double*8B

	full_text

double* %188
mcall8Bc
a
	full_textT
R
P%665 = tail call double @llvm.fmuladd.f64(double %664, double %603, double %663)
,double8B

	full_text

double %664
,double8B

	full_text

double %603
,double8B

	full_text

double %663
Qload8BG
E
	full_text8
6
4%666 = load double, double* %321, align 16, !tbaa !8
.double*8B

	full_text

double* %321
mcall8Bc
a
	full_textT
R
P%667 = tail call double @llvm.fmuladd.f64(double %666, double %605, double %665)
,double8B

	full_text

double %666
,double8B

	full_text

double %605
,double8B

	full_text

double %665
Qload8BG
E
	full_text8
6
4%668 = load double, double* %189, align 16, !tbaa !8
.double*8B

	full_text

double* %189
mcall8Bc
a
	full_textT
R
P%669 = tail call double @llvm.fmuladd.f64(double %668, double %607, double %667)
,double8B

	full_text

double %668
,double8B

	full_text

double %607
,double8B

	full_text

double %667
ucall8Bk
i
	full_text\
Z
X%670 = tail call double @llvm.fmuladd.f64(double %669, double 1.200000e+00, double %567)
,double8B

	full_text

double %669
,double8B

	full_text

double %567
Qstore8BF
D
	full_text7
5
3store double %670, double* %568, align 16, !tbaa !8
,double8B

	full_text

double %670
.double*8B

	full_text

double* %568
Pload8BF
D
	full_text7
5
3%671 = load double, double* %329, align 8, !tbaa !8
.double*8B

	full_text

double* %329
Pload8BF
D
	full_text7
5
3%672 = load double, double* %197, align 8, !tbaa !8
.double*8B

	full_text

double* %197
:fmul8B0
.
	full_text!

%673 = fmul double %672, %591
,double8B

	full_text

double %672
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%674 = tail call double @llvm.fmuladd.f64(double %671, double %589, double %673)
,double8B

	full_text

double %671
,double8B

	full_text

double %589
,double8B

	full_text

double %673
Pload8BF
D
	full_text7
5
3%675 = load double, double* %330, align 8, !tbaa !8
.double*8B

	full_text

double* %330
mcall8Bc
a
	full_textT
R
P%676 = tail call double @llvm.fmuladd.f64(double %675, double %593, double %674)
,double8B

	full_text

double %675
,double8B

	full_text

double %593
,double8B

	full_text

double %674
Pload8BF
D
	full_text7
5
3%677 = load double, double* %199, align 8, !tbaa !8
.double*8B

	full_text

double* %199
mcall8Bc
a
	full_textT
R
P%678 = tail call double @llvm.fmuladd.f64(double %677, double %595, double %676)
,double8B

	full_text

double %677
,double8B

	full_text

double %595
,double8B

	full_text

double %676
Pload8BF
D
	full_text7
5
3%679 = load double, double* %332, align 8, !tbaa !8
.double*8B

	full_text

double* %332
mcall8Bc
a
	full_textT
R
P%680 = tail call double @llvm.fmuladd.f64(double %679, double %597, double %678)
,double8B

	full_text

double %679
,double8B

	full_text

double %597
,double8B

	full_text

double %678
Pload8BF
D
	full_text7
5
3%681 = load double, double* %200, align 8, !tbaa !8
.double*8B

	full_text

double* %200
mcall8Bc
a
	full_textT
R
P%682 = tail call double @llvm.fmuladd.f64(double %681, double %599, double %680)
,double8B

	full_text

double %681
,double8B

	full_text

double %599
,double8B

	full_text

double %680
Pload8BF
D
	full_text7
5
3%683 = load double, double* %333, align 8, !tbaa !8
.double*8B

	full_text

double* %333
mcall8Bc
a
	full_textT
R
P%684 = tail call double @llvm.fmuladd.f64(double %683, double %601, double %682)
,double8B

	full_text

double %683
,double8B

	full_text

double %601
,double8B

	full_text

double %682
Pload8BF
D
	full_text7
5
3%685 = load double, double* %201, align 8, !tbaa !8
.double*8B

	full_text

double* %201
mcall8Bc
a
	full_textT
R
P%686 = tail call double @llvm.fmuladd.f64(double %685, double %603, double %684)
,double8B

	full_text

double %685
,double8B

	full_text

double %603
,double8B

	full_text

double %684
Pload8BF
D
	full_text7
5
3%687 = load double, double* %334, align 8, !tbaa !8
.double*8B

	full_text

double* %334
mcall8Bc
a
	full_textT
R
P%688 = tail call double @llvm.fmuladd.f64(double %687, double %605, double %686)
,double8B

	full_text

double %687
,double8B

	full_text

double %605
,double8B

	full_text

double %686
Pload8BF
D
	full_text7
5
3%689 = load double, double* %202, align 8, !tbaa !8
.double*8B

	full_text

double* %202
mcall8Bc
a
	full_textT
R
P%690 = tail call double @llvm.fmuladd.f64(double %689, double %607, double %688)
,double8B

	full_text

double %689
,double8B

	full_text

double %607
,double8B

	full_text

double %688
ucall8Bk
i
	full_text\
Z
X%691 = tail call double @llvm.fmuladd.f64(double %690, double 1.200000e+00, double %579)
,double8B

	full_text

double %690
,double8B

	full_text

double %579
Pstore8BE
C
	full_text6
4
2store double %691, double* %580, align 8, !tbaa !8
,double8B

	full_text

double %691
.double*8B

	full_text

double* %580
Qload8BG
E
	full_text8
6
4%692 = load double, double* %587, align 16, !tbaa !8
.double*8B

	full_text

double* %587
Qload8BG
E
	full_text8
6
4%693 = load double, double* %357, align 16, !tbaa !8
.double*8B

	full_text

double* %357
Qload8BG
E
	full_text8
6
4%694 = load double, double* %226, align 16, !tbaa !8
.double*8B

	full_text

double* %226
:fmul8B0
.
	full_text!

%695 = fmul double %694, %591
,double8B

	full_text

double %694
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%696 = tail call double @llvm.fmuladd.f64(double %693, double %589, double %695)
,double8B

	full_text

double %693
,double8B

	full_text

double %589
,double8B

	full_text

double %695
Pload8BF
D
	full_text7
5
3%697 = load double, double* %364, align 8, !tbaa !8
.double*8B

	full_text

double* %364
mcall8Bc
a
	full_textT
R
P%698 = tail call double @llvm.fmuladd.f64(double %697, double %593, double %696)
,double8B

	full_text

double %697
,double8B

	full_text

double %593
,double8B

	full_text

double %696
Pload8BF
D
	full_text7
5
3%699 = load double, double* %237, align 8, !tbaa !8
.double*8B

	full_text

double* %237
mcall8Bc
a
	full_textT
R
P%700 = tail call double @llvm.fmuladd.f64(double %699, double %595, double %698)
,double8B

	full_text

double %699
,double8B

	full_text

double %595
,double8B

	full_text

double %698
Qload8BG
E
	full_text8
6
4%701 = load double, double* %375, align 16, !tbaa !8
.double*8B

	full_text

double* %375
mcall8Bc
a
	full_textT
R
P%702 = tail call double @llvm.fmuladd.f64(double %701, double %597, double %700)
,double8B

	full_text

double %701
,double8B

	full_text

double %597
,double8B

	full_text

double %700
Qload8BG
E
	full_text8
6
4%703 = load double, double* %244, align 16, !tbaa !8
.double*8B

	full_text

double* %244
mcall8Bc
a
	full_textT
R
P%704 = tail call double @llvm.fmuladd.f64(double %703, double %599, double %702)
,double8B

	full_text

double %703
,double8B

	full_text

double %599
,double8B

	full_text

double %702
Pload8BF
D
	full_text7
5
3%705 = load double, double* %381, align 8, !tbaa !8
.double*8B

	full_text

double* %381
mcall8Bc
a
	full_textT
R
P%706 = tail call double @llvm.fmuladd.f64(double %705, double %601, double %704)
,double8B

	full_text

double %705
,double8B

	full_text

double %601
,double8B

	full_text

double %704
Pload8BF
D
	full_text7
5
3%707 = load double, double* %250, align 8, !tbaa !8
.double*8B

	full_text

double* %250
mcall8Bc
a
	full_textT
R
P%708 = tail call double @llvm.fmuladd.f64(double %707, double %603, double %706)
,double8B

	full_text

double %707
,double8B

	full_text

double %603
,double8B

	full_text

double %706
Qload8BG
E
	full_text8
6
4%709 = load double, double* %387, align 16, !tbaa !8
.double*8B

	full_text

double* %387
mcall8Bc
a
	full_textT
R
P%710 = tail call double @llvm.fmuladd.f64(double %709, double %605, double %708)
,double8B

	full_text

double %709
,double8B

	full_text

double %605
,double8B

	full_text

double %708
Qload8BG
E
	full_text8
6
4%711 = load double, double* %256, align 16, !tbaa !8
.double*8B

	full_text

double* %256
mcall8Bc
a
	full_textT
R
P%712 = tail call double @llvm.fmuladd.f64(double %711, double %607, double %710)
,double8B

	full_text

double %711
,double8B

	full_text

double %607
,double8B

	full_text

double %710
ucall8Bk
i
	full_text\
Z
X%713 = tail call double @llvm.fmuladd.f64(double %712, double 1.200000e+00, double %692)
,double8B

	full_text

double %712
,double8B

	full_text

double %692
Qstore8BF
D
	full_text7
5
3store double %713, double* %587, align 16, !tbaa !8
,double8B

	full_text

double %713
.double*8B

	full_text

double* %587
Nbitcast8BA
?
	full_text2
0
.%714 = bitcast [5 x [5 x double]]* %14 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Kload8BA
?
	full_text2
0
.%715 = load i64, i64* %714, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %714
Nbitcast8BA
?
	full_text2
0
.%716 = bitcast [5 x [5 x double]]* %15 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Kstore8B@
>
	full_text1
/
-store i64 %715, i64* %716, align 16, !tbaa !8
&i648B

	full_text


i64 %715
(i64*8B

	full_text

	i64* %716
Bbitcast8B5
3
	full_text&
$
"%717 = bitcast double* %66 to i64*
-double*8B

	full_text

double* %66
Jload8B@
>
	full_text1
/
-%718 = load i64, i64* %717, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %717
„getelementptr8Bq
o
	full_textb
`
^%719 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%720 = bitcast double* %719 to i64*
.double*8B

	full_text

double* %719
Jstore8B?
=
	full_text0
.
,store i64 %718, i64* %720, align 8, !tbaa !8
&i648B

	full_text


i64 %718
(i64*8B

	full_text

	i64* %720
Bbitcast8B5
3
	full_text&
$
"%721 = bitcast double* %67 to i64*
-double*8B

	full_text

double* %67
Kload8BA
?
	full_text2
0
.%722 = load i64, i64* %721, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %721
„getelementptr8Bq
o
	full_textb
`
^%723 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%724 = bitcast double* %723 to i64*
.double*8B

	full_text

double* %723
Kstore8B@
>
	full_text1
/
-store i64 %722, i64* %724, align 16, !tbaa !8
&i648B

	full_text


i64 %722
(i64*8B

	full_text

	i64* %724
Bbitcast8B5
3
	full_text&
$
"%725 = bitcast double* %68 to i64*
-double*8B

	full_text

double* %68
Jload8B@
>
	full_text1
/
-%726 = load i64, i64* %725, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %725
„getelementptr8Bq
o
	full_textb
`
^%727 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%728 = bitcast double* %727 to i64*
.double*8B

	full_text

double* %727
Jstore8B?
=
	full_text0
.
,store i64 %726, i64* %728, align 8, !tbaa !8
&i648B

	full_text


i64 %726
(i64*8B

	full_text

	i64* %728
Bbitcast8B5
3
	full_text&
$
"%729 = bitcast double* %69 to i64*
-double*8B

	full_text

double* %69
Kload8BA
?
	full_text2
0
.%730 = load i64, i64* %729, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %729
„getelementptr8Bq
o
	full_textb
`
^%731 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%732 = bitcast double* %731 to i64*
.double*8B

	full_text

double* %731
Kstore8B@
>
	full_text1
/
-store i64 %730, i64* %732, align 16, !tbaa !8
&i648B

	full_text


i64 %730
(i64*8B

	full_text

	i64* %732
Bbitcast8B5
3
	full_text&
$
"%733 = bitcast double* %75 to i64*
-double*8B

	full_text

double* %75
Jload8B@
>
	full_text1
/
-%734 = load i64, i64* %733, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %733
}getelementptr8Bj
h
	full_text[
Y
W%735 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%736 = bitcast [5 x double]* %735 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %735
Jstore8B?
=
	full_text0
.
,store i64 %734, i64* %736, align 8, !tbaa !8
&i648B

	full_text


i64 %734
(i64*8B

	full_text

	i64* %736
Bbitcast8B5
3
	full_text&
$
"%737 = bitcast double* %79 to i64*
-double*8B

	full_text

double* %79
Jload8B@
>
	full_text1
/
-%738 = load i64, i64* %737, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %737
„getelementptr8Bq
o
	full_textb
`
^%739 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%740 = bitcast double* %739 to i64*
.double*8B

	full_text

double* %739
Jstore8B?
=
	full_text0
.
,store i64 %738, i64* %740, align 8, !tbaa !8
&i648B

	full_text


i64 %738
(i64*8B

	full_text

	i64* %740
Bbitcast8B5
3
	full_text&
$
"%741 = bitcast double* %80 to i64*
-double*8B

	full_text

double* %80
Jload8B@
>
	full_text1
/
-%742 = load i64, i64* %741, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %741
„getelementptr8Bq
o
	full_textb
`
^%743 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%744 = bitcast double* %743 to i64*
.double*8B

	full_text

double* %743
Jstore8B?
=
	full_text0
.
,store i64 %742, i64* %744, align 8, !tbaa !8
&i648B

	full_text


i64 %742
(i64*8B

	full_text

	i64* %744
Oload8BE
C
	full_text6
4
2%745 = load double, double* %81, align 8, !tbaa !8
-double*8B

	full_text

double* %81
„getelementptr8Bq
o
	full_textb
`
^%746 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%747 = load double, double* %82, align 8, !tbaa !8
-double*8B

	full_text

double* %82
„getelementptr8Bq
o
	full_textb
`
^%748 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Bbitcast8B5
3
	full_text&
$
"%749 = bitcast double* %87 to i64*
-double*8B

	full_text

double* %87
Kload8BA
?
	full_text2
0
.%750 = load i64, i64* %749, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %749
}getelementptr8Bj
h
	full_text[
Y
W%751 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%752 = bitcast [5 x double]* %751 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %751
Kstore8B@
>
	full_text1
/
-store i64 %750, i64* %752, align 16, !tbaa !8
&i648B

	full_text


i64 %750
(i64*8B

	full_text

	i64* %752
Bbitcast8B5
3
	full_text&
$
"%753 = bitcast double* %88 to i64*
-double*8B

	full_text

double* %88
Jload8B@
>
	full_text1
/
-%754 = load i64, i64* %753, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %753
„getelementptr8Bq
o
	full_textb
`
^%755 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%756 = bitcast double* %755 to i64*
.double*8B

	full_text

double* %755
Jstore8B?
=
	full_text0
.
,store i64 %754, i64* %756, align 8, !tbaa !8
&i648B

	full_text


i64 %754
(i64*8B

	full_text

	i64* %756
Pload8BF
D
	full_text7
5
3%757 = load double, double* %89, align 16, !tbaa !8
-double*8B

	full_text

double* %89
„getelementptr8Bq
o
	full_textb
`
^%758 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%759 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
„getelementptr8Bq
o
	full_textb
`
^%760 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%761 = load double, double* %91, align 16, !tbaa !8
-double*8B

	full_text

double* %91
„getelementptr8Bq
o
	full_textb
`
^%762 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Bbitcast8B5
3
	full_text&
$
"%763 = bitcast double* %96 to i64*
-double*8B

	full_text

double* %96
Jload8B@
>
	full_text1
/
-%764 = load i64, i64* %763, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %763
}getelementptr8Bj
h
	full_text[
Y
W%765 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%766 = bitcast [5 x double]* %765 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %765
Jstore8B?
=
	full_text0
.
,store i64 %764, i64* %766, align 8, !tbaa !8
&i648B

	full_text


i64 %764
(i64*8B

	full_text

	i64* %766
Oload8BE
C
	full_text6
4
2%767 = load double, double* %97, align 8, !tbaa !8
-double*8B

	full_text

double* %97
„getelementptr8Bq
o
	full_textb
`
^%768 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%769 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
„getelementptr8Bq
o
	full_textb
`
^%770 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%771 = load double, double* %99, align 8, !tbaa !8
-double*8B

	full_text

double* %99
„getelementptr8Bq
o
	full_textb
`
^%772 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%773 = load double, double* %100, align 8, !tbaa !8
.double*8B

	full_text

double* %100
„getelementptr8Bq
o
	full_textb
`
^%774 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%775 = bitcast double* %113 to i64*
.double*8B

	full_text

double* %113
Kload8BA
?
	full_text2
0
.%776 = load i64, i64* %775, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %775
}getelementptr8Bj
h
	full_text[
Y
W%777 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%778 = bitcast [5 x double]* %777 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %777
Kstore8B@
>
	full_text1
/
-store i64 %776, i64* %778, align 16, !tbaa !8
&i648B

	full_text


i64 %776
(i64*8B

	full_text

	i64* %778
Pload8BF
D
	full_text7
5
3%779 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
„getelementptr8Bq
o
	full_textb
`
^%780 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%781 = load double, double* %119, align 16, !tbaa !8
.double*8B

	full_text

double* %119
„getelementptr8Bq
o
	full_textb
`
^%782 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%783 = load double, double* %121, align 8, !tbaa !8
.double*8B

	full_text

double* %121
„getelementptr8Bq
o
	full_textb
`
^%784 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%785 = load double, double* %124, align 16, !tbaa !8
.double*8B

	full_text

double* %124
„getelementptr8Bq
o
	full_textb
`
^%786 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
„getelementptr8Bq
o
	full_textb
`
^%787 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%788 = load double, double* %787, align 16, !tbaa !8
.double*8B

	full_text

double* %787
Bfdiv8B8
6
	full_text)
'
%%789 = fdiv double 1.000000e+00, %788
,double8B

	full_text

double %788
„getelementptr8Bq
o
	full_textb
`
^%790 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%791 = load double, double* %790, align 8, !tbaa !8
.double*8B

	full_text

double* %790
:fmul8B0
.
	full_text!

%792 = fmul double %789, %791
,double8B

	full_text

double %789
,double8B

	full_text

double %791
Abitcast8B4
2
	full_text%
#
!%793 = bitcast i64 %738 to double
&i648B

	full_text


i64 %738
Pload8BF
D
	full_text7
5
3%794 = load double, double* %719, align 8, !tbaa !8
.double*8B

	full_text

double* %719
Cfsub8B9
7
	full_text*
(
&%795 = fsub double -0.000000e+00, %792
,double8B

	full_text

double %792
mcall8Bc
a
	full_textT
R
P%796 = tail call double @llvm.fmuladd.f64(double %795, double %794, double %793)
,double8B

	full_text

double %795
,double8B

	full_text

double %794
,double8B

	full_text

double %793
Pstore8BE
C
	full_text6
4
2store double %796, double* %739, align 8, !tbaa !8
,double8B

	full_text

double %796
.double*8B

	full_text

double* %739
Abitcast8B4
2
	full_text%
#
!%797 = bitcast i64 %742 to double
&i648B

	full_text


i64 %742
Qload8BG
E
	full_text8
6
4%798 = load double, double* %723, align 16, !tbaa !8
.double*8B

	full_text

double* %723
mcall8Bc
a
	full_textT
R
P%799 = tail call double @llvm.fmuladd.f64(double %795, double %798, double %797)
,double8B

	full_text

double %795
,double8B

	full_text

double %798
,double8B

	full_text

double %797
Pstore8BE
C
	full_text6
4
2store double %799, double* %743, align 8, !tbaa !8
,double8B

	full_text

double %799
.double*8B

	full_text

double* %743
Pload8BF
D
	full_text7
5
3%800 = load double, double* %727, align 8, !tbaa !8
.double*8B

	full_text

double* %727
mcall8Bc
a
	full_textT
R
P%801 = tail call double @llvm.fmuladd.f64(double %795, double %800, double %745)
,double8B

	full_text

double %795
,double8B

	full_text

double %800
,double8B

	full_text

double %745
Pstore8BE
C
	full_text6
4
2store double %801, double* %746, align 8, !tbaa !8
,double8B

	full_text

double %801
.double*8B

	full_text

double* %746
Qload8BG
E
	full_text8
6
4%802 = load double, double* %731, align 16, !tbaa !8
.double*8B

	full_text

double* %731
mcall8Bc
a
	full_textT
R
P%803 = tail call double @llvm.fmuladd.f64(double %795, double %802, double %747)
,double8B

	full_text

double %795
,double8B

	full_text

double %802
,double8B

	full_text

double %747
Pstore8BE
C
	full_text6
4
2store double %803, double* %748, align 8, !tbaa !8
,double8B

	full_text

double %803
.double*8B

	full_text

double* %748
Pload8BF
D
	full_text7
5
3%804 = load double, double* %556, align 8, !tbaa !8
.double*8B

	full_text

double* %556
Qload8BG
E
	full_text8
6
4%805 = load double, double* %544, align 16, !tbaa !8
.double*8B

	full_text

double* %544
Cfsub8B9
7
	full_text*
(
&%806 = fsub double -0.000000e+00, %805
,double8B

	full_text

double %805
mcall8Bc
a
	full_textT
R
P%807 = tail call double @llvm.fmuladd.f64(double %806, double %792, double %804)
,double8B

	full_text

double %806
,double8B

	full_text

double %792
,double8B

	full_text

double %804
Pstore8BE
C
	full_text6
4
2store double %807, double* %556, align 8, !tbaa !8
,double8B

	full_text

double %807
.double*8B

	full_text

double* %556
„getelementptr8Bq
o
	full_textb
`
^%808 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%809 = load double, double* %808, align 16, !tbaa !8
.double*8B

	full_text

double* %808
:fmul8B0
.
	full_text!

%810 = fmul double %789, %809
,double8B

	full_text

double %789
,double8B

	full_text

double %809
Abitcast8B4
2
	full_text%
#
!%811 = bitcast i64 %754 to double
&i648B

	full_text


i64 %754
Cfsub8B9
7
	full_text*
(
&%812 = fsub double -0.000000e+00, %810
,double8B

	full_text

double %810
mcall8Bc
a
	full_textT
R
P%813 = tail call double @llvm.fmuladd.f64(double %812, double %794, double %811)
,double8B

	full_text

double %812
,double8B

	full_text

double %794
,double8B

	full_text

double %811
Pstore8BE
C
	full_text6
4
2store double %813, double* %755, align 8, !tbaa !8
,double8B

	full_text

double %813
.double*8B

	full_text

double* %755
mcall8Bc
a
	full_textT
R
P%814 = tail call double @llvm.fmuladd.f64(double %812, double %798, double %757)
,double8B

	full_text

double %812
,double8B

	full_text

double %798
,double8B

	full_text

double %757
mcall8Bc
a
	full_textT
R
P%815 = tail call double @llvm.fmuladd.f64(double %812, double %800, double %759)
,double8B

	full_text

double %812
,double8B

	full_text

double %800
,double8B

	full_text

double %759
mcall8Bc
a
	full_textT
R
P%816 = tail call double @llvm.fmuladd.f64(double %812, double %802, double %761)
,double8B

	full_text

double %812
,double8B

	full_text

double %802
,double8B

	full_text

double %761
Qload8BG
E
	full_text8
6
4%817 = load double, double* %568, align 16, !tbaa !8
.double*8B

	full_text

double* %568
mcall8Bc
a
	full_textT
R
P%818 = tail call double @llvm.fmuladd.f64(double %806, double %810, double %817)
,double8B

	full_text

double %806
,double8B

	full_text

double %810
,double8B

	full_text

double %817
Abitcast8B4
2
	full_text%
#
!%819 = bitcast i64 %764 to double
&i648B

	full_text


i64 %764
:fmul8B0
.
	full_text!

%820 = fmul double %789, %819
,double8B

	full_text

double %789
,double8B

	full_text

double %819
Cfsub8B9
7
	full_text*
(
&%821 = fsub double -0.000000e+00, %820
,double8B

	full_text

double %820
mcall8Bc
a
	full_textT
R
P%822 = tail call double @llvm.fmuladd.f64(double %821, double %794, double %767)
,double8B

	full_text

double %821
,double8B

	full_text

double %794
,double8B

	full_text

double %767
Pstore8BE
C
	full_text6
4
2store double %822, double* %768, align 8, !tbaa !8
,double8B

	full_text

double %822
.double*8B

	full_text

double* %768
mcall8Bc
a
	full_textT
R
P%823 = tail call double @llvm.fmuladd.f64(double %821, double %798, double %769)
,double8B

	full_text

double %821
,double8B

	full_text

double %798
,double8B

	full_text

double %769
mcall8Bc
a
	full_textT
R
P%824 = tail call double @llvm.fmuladd.f64(double %821, double %800, double %771)
,double8B

	full_text

double %821
,double8B

	full_text

double %800
,double8B

	full_text

double %771
mcall8Bc
a
	full_textT
R
P%825 = tail call double @llvm.fmuladd.f64(double %821, double %802, double %773)
,double8B

	full_text

double %821
,double8B

	full_text

double %802
,double8B

	full_text

double %773
Pload8BF
D
	full_text7
5
3%826 = load double, double* %580, align 8, !tbaa !8
.double*8B

	full_text

double* %580
mcall8Bc
a
	full_textT
R
P%827 = tail call double @llvm.fmuladd.f64(double %806, double %820, double %826)
,double8B

	full_text

double %806
,double8B

	full_text

double %820
,double8B

	full_text

double %826
Abitcast8B4
2
	full_text%
#
!%828 = bitcast i64 %776 to double
&i648B

	full_text


i64 %776
:fmul8B0
.
	full_text!

%829 = fmul double %789, %828
,double8B

	full_text

double %789
,double8B

	full_text

double %828
Cfsub8B9
7
	full_text*
(
&%830 = fsub double -0.000000e+00, %829
,double8B

	full_text

double %829
mcall8Bc
a
	full_textT
R
P%831 = tail call double @llvm.fmuladd.f64(double %830, double %794, double %779)
,double8B

	full_text

double %830
,double8B

	full_text

double %794
,double8B

	full_text

double %779
Pstore8BE
C
	full_text6
4
2store double %831, double* %780, align 8, !tbaa !8
,double8B

	full_text

double %831
.double*8B

	full_text

double* %780
mcall8Bc
a
	full_textT
R
P%832 = tail call double @llvm.fmuladd.f64(double %830, double %798, double %781)
,double8B

	full_text

double %830
,double8B

	full_text

double %798
,double8B

	full_text

double %781
mcall8Bc
a
	full_textT
R
P%833 = tail call double @llvm.fmuladd.f64(double %830, double %800, double %783)
,double8B

	full_text

double %830
,double8B

	full_text

double %800
,double8B

	full_text

double %783
mcall8Bc
a
	full_textT
R
P%834 = tail call double @llvm.fmuladd.f64(double %830, double %802, double %785)
,double8B

	full_text

double %830
,double8B

	full_text

double %802
,double8B

	full_text

double %785
Qload8BG
E
	full_text8
6
4%835 = load double, double* %587, align 16, !tbaa !8
.double*8B

	full_text

double* %587
mcall8Bc
a
	full_textT
R
P%836 = tail call double @llvm.fmuladd.f64(double %806, double %829, double %835)
,double8B

	full_text

double %806
,double8B

	full_text

double %829
,double8B

	full_text

double %835
Bfdiv8B8
6
	full_text)
'
%%837 = fdiv double 1.000000e+00, %796
,double8B

	full_text

double %796
:fmul8B0
.
	full_text!

%838 = fmul double %837, %813
,double8B

	full_text

double %837
,double8B

	full_text

double %813
Cfsub8B9
7
	full_text*
(
&%839 = fsub double -0.000000e+00, %838
,double8B

	full_text

double %838
mcall8Bc
a
	full_textT
R
P%840 = tail call double @llvm.fmuladd.f64(double %839, double %799, double %814)
,double8B

	full_text

double %839
,double8B

	full_text

double %799
,double8B

	full_text

double %814
Qstore8BF
D
	full_text7
5
3store double %840, double* %758, align 16, !tbaa !8
,double8B

	full_text

double %840
.double*8B

	full_text

double* %758
mcall8Bc
a
	full_textT
R
P%841 = tail call double @llvm.fmuladd.f64(double %839, double %801, double %815)
,double8B

	full_text

double %839
,double8B

	full_text

double %801
,double8B

	full_text

double %815
Pstore8BE
C
	full_text6
4
2store double %841, double* %760, align 8, !tbaa !8
,double8B

	full_text

double %841
.double*8B

	full_text

double* %760
mcall8Bc
a
	full_textT
R
P%842 = tail call double @llvm.fmuladd.f64(double %839, double %803, double %816)
,double8B

	full_text

double %839
,double8B

	full_text

double %803
,double8B

	full_text

double %816
Qstore8BF
D
	full_text7
5
3store double %842, double* %762, align 16, !tbaa !8
,double8B

	full_text

double %842
.double*8B

	full_text

double* %762
Cfsub8B9
7
	full_text*
(
&%843 = fsub double -0.000000e+00, %807
,double8B

	full_text

double %807
mcall8Bc
a
	full_textT
R
P%844 = tail call double @llvm.fmuladd.f64(double %843, double %838, double %818)
,double8B

	full_text

double %843
,double8B

	full_text

double %838
,double8B

	full_text

double %818
:fmul8B0
.
	full_text!

%845 = fmul double %837, %822
,double8B

	full_text

double %837
,double8B

	full_text

double %822
Cfsub8B9
7
	full_text*
(
&%846 = fsub double -0.000000e+00, %845
,double8B

	full_text

double %845
mcall8Bc
a
	full_textT
R
P%847 = tail call double @llvm.fmuladd.f64(double %846, double %799, double %823)
,double8B

	full_text

double %846
,double8B

	full_text

double %799
,double8B

	full_text

double %823
Pstore8BE
C
	full_text6
4
2store double %847, double* %770, align 8, !tbaa !8
,double8B

	full_text

double %847
.double*8B

	full_text

double* %770
mcall8Bc
a
	full_textT
R
P%848 = tail call double @llvm.fmuladd.f64(double %846, double %801, double %824)
,double8B

	full_text

double %846
,double8B

	full_text

double %801
,double8B

	full_text

double %824
mcall8Bc
a
	full_textT
R
P%849 = tail call double @llvm.fmuladd.f64(double %846, double %803, double %825)
,double8B

	full_text

double %846
,double8B

	full_text

double %803
,double8B

	full_text

double %825
mcall8Bc
a
	full_textT
R
P%850 = tail call double @llvm.fmuladd.f64(double %843, double %845, double %827)
,double8B

	full_text

double %843
,double8B

	full_text

double %845
,double8B

	full_text

double %827
:fmul8B0
.
	full_text!

%851 = fmul double %837, %831
,double8B

	full_text

double %837
,double8B

	full_text

double %831
Cfsub8B9
7
	full_text*
(
&%852 = fsub double -0.000000e+00, %851
,double8B

	full_text

double %851
mcall8Bc
a
	full_textT
R
P%853 = tail call double @llvm.fmuladd.f64(double %852, double %799, double %832)
,double8B

	full_text

double %852
,double8B

	full_text

double %799
,double8B

	full_text

double %832
Qstore8BF
D
	full_text7
5
3store double %853, double* %782, align 16, !tbaa !8
,double8B

	full_text

double %853
.double*8B

	full_text

double* %782
mcall8Bc
a
	full_textT
R
P%854 = tail call double @llvm.fmuladd.f64(double %852, double %801, double %833)
,double8B

	full_text

double %852
,double8B

	full_text

double %801
,double8B

	full_text

double %833
mcall8Bc
a
	full_textT
R
P%855 = tail call double @llvm.fmuladd.f64(double %852, double %803, double %834)
,double8B

	full_text

double %852
,double8B

	full_text

double %803
,double8B

	full_text

double %834
mcall8Bc
a
	full_textT
R
P%856 = tail call double @llvm.fmuladd.f64(double %843, double %851, double %836)
,double8B

	full_text

double %843
,double8B

	full_text

double %851
,double8B

	full_text

double %836
Bfdiv8B8
6
	full_text)
'
%%857 = fdiv double 1.000000e+00, %840
,double8B

	full_text

double %840
:fmul8B0
.
	full_text!

%858 = fmul double %857, %847
,double8B

	full_text

double %857
,double8B

	full_text

double %847
Cfsub8B9
7
	full_text*
(
&%859 = fsub double -0.000000e+00, %858
,double8B

	full_text

double %858
mcall8Bc
a
	full_textT
R
P%860 = tail call double @llvm.fmuladd.f64(double %859, double %841, double %848)
,double8B

	full_text

double %859
,double8B

	full_text

double %841
,double8B

	full_text

double %848
Pstore8BE
C
	full_text6
4
2store double %860, double* %772, align 8, !tbaa !8
,double8B

	full_text

double %860
.double*8B

	full_text

double* %772
mcall8Bc
a
	full_textT
R
P%861 = tail call double @llvm.fmuladd.f64(double %859, double %842, double %849)
,double8B

	full_text

double %859
,double8B

	full_text

double %842
,double8B

	full_text

double %849
Pstore8BE
C
	full_text6
4
2store double %861, double* %774, align 8, !tbaa !8
,double8B

	full_text

double %861
.double*8B

	full_text

double* %774
Cfsub8B9
7
	full_text*
(
&%862 = fsub double -0.000000e+00, %844
,double8B

	full_text

double %844
mcall8Bc
a
	full_textT
R
P%863 = tail call double @llvm.fmuladd.f64(double %862, double %858, double %850)
,double8B

	full_text

double %862
,double8B

	full_text

double %858
,double8B

	full_text

double %850
:fmul8B0
.
	full_text!

%864 = fmul double %857, %853
,double8B

	full_text

double %857
,double8B

	full_text

double %853
Cfsub8B9
7
	full_text*
(
&%865 = fsub double -0.000000e+00, %864
,double8B

	full_text

double %864
mcall8Bc
a
	full_textT
R
P%866 = tail call double @llvm.fmuladd.f64(double %865, double %841, double %854)
,double8B

	full_text

double %865
,double8B

	full_text

double %841
,double8B

	full_text

double %854
Pstore8BE
C
	full_text6
4
2store double %866, double* %784, align 8, !tbaa !8
,double8B

	full_text

double %866
.double*8B

	full_text

double* %784
mcall8Bc
a
	full_textT
R
P%867 = tail call double @llvm.fmuladd.f64(double %865, double %842, double %855)
,double8B

	full_text

double %865
,double8B

	full_text

double %842
,double8B

	full_text

double %855
mcall8Bc
a
	full_textT
R
P%868 = tail call double @llvm.fmuladd.f64(double %862, double %864, double %856)
,double8B

	full_text

double %862
,double8B

	full_text

double %864
,double8B

	full_text

double %856
Bfdiv8B8
6
	full_text)
'
%%869 = fdiv double 1.000000e+00, %860
,double8B

	full_text

double %860
:fmul8B0
.
	full_text!

%870 = fmul double %869, %866
,double8B

	full_text

double %869
,double8B

	full_text

double %866
Cfsub8B9
7
	full_text*
(
&%871 = fsub double -0.000000e+00, %870
,double8B

	full_text

double %870
mcall8Bc
a
	full_textT
R
P%872 = tail call double @llvm.fmuladd.f64(double %871, double %861, double %867)
,double8B

	full_text

double %871
,double8B

	full_text

double %861
,double8B

	full_text

double %867
Qstore8BF
D
	full_text7
5
3store double %872, double* %786, align 16, !tbaa !8
,double8B

	full_text

double %872
.double*8B

	full_text

double* %786
Cfsub8B9
7
	full_text*
(
&%873 = fsub double -0.000000e+00, %863
,double8B

	full_text

double %863
mcall8Bc
a
	full_textT
R
P%874 = tail call double @llvm.fmuladd.f64(double %873, double %870, double %868)
,double8B

	full_text

double %873
,double8B

	full_text

double %870
,double8B

	full_text

double %868
:fdiv8B0
.
	full_text!

%875 = fdiv double %874, %872
,double8B

	full_text

double %874
,double8B

	full_text

double %872
Qstore8BF
D
	full_text7
5
3store double %875, double* %587, align 16, !tbaa !8
,double8B

	full_text

double %875
.double*8B

	full_text

double* %587
Cfsub8B9
7
	full_text*
(
&%876 = fsub double -0.000000e+00, %861
,double8B

	full_text

double %861
mcall8Bc
a
	full_textT
R
P%877 = tail call double @llvm.fmuladd.f64(double %876, double %875, double %863)
,double8B

	full_text

double %876
,double8B

	full_text

double %875
,double8B

	full_text

double %863
:fdiv8B0
.
	full_text!

%878 = fdiv double %877, %860
,double8B

	full_text

double %877
,double8B

	full_text

double %860
Pstore8BE
C
	full_text6
4
2store double %878, double* %580, align 8, !tbaa !8
,double8B

	full_text

double %878
.double*8B

	full_text

double* %580
Cfsub8B9
7
	full_text*
(
&%879 = fsub double -0.000000e+00, %841
,double8B

	full_text

double %841
mcall8Bc
a
	full_textT
R
P%880 = tail call double @llvm.fmuladd.f64(double %879, double %878, double %844)
,double8B

	full_text

double %879
,double8B

	full_text

double %878
,double8B

	full_text

double %844
Cfsub8B9
7
	full_text*
(
&%881 = fsub double -0.000000e+00, %842
,double8B

	full_text

double %842
mcall8Bc
a
	full_textT
R
P%882 = tail call double @llvm.fmuladd.f64(double %881, double %875, double %880)
,double8B

	full_text

double %881
,double8B

	full_text

double %875
,double8B

	full_text

double %880
:fdiv8B0
.
	full_text!

%883 = fdiv double %882, %840
,double8B

	full_text

double %882
,double8B

	full_text

double %840
Qstore8BF
D
	full_text7
5
3store double %883, double* %568, align 16, !tbaa !8
,double8B

	full_text

double %883
.double*8B

	full_text

double* %568
Cfsub8B9
7
	full_text*
(
&%884 = fsub double -0.000000e+00, %799
,double8B

	full_text

double %799
mcall8Bc
a
	full_textT
R
P%885 = tail call double @llvm.fmuladd.f64(double %884, double %883, double %807)
,double8B

	full_text

double %884
,double8B

	full_text

double %883
,double8B

	full_text

double %807
Cfsub8B9
7
	full_text*
(
&%886 = fsub double -0.000000e+00, %801
,double8B

	full_text

double %801
mcall8Bc
a
	full_textT
R
P%887 = tail call double @llvm.fmuladd.f64(double %886, double %878, double %885)
,double8B

	full_text

double %886
,double8B

	full_text

double %878
,double8B

	full_text

double %885
Cfsub8B9
7
	full_text*
(
&%888 = fsub double -0.000000e+00, %803
,double8B

	full_text

double %803
mcall8Bc
a
	full_textT
R
P%889 = tail call double @llvm.fmuladd.f64(double %888, double %875, double %887)
,double8B

	full_text

double %888
,double8B

	full_text

double %875
,double8B

	full_text

double %887
:fdiv8B0
.
	full_text!

%890 = fdiv double %889, %796
,double8B

	full_text

double %889
,double8B

	full_text

double %796
Pstore8BE
C
	full_text6
4
2store double %890, double* %556, align 8, !tbaa !8
,double8B

	full_text

double %890
.double*8B

	full_text

double* %556
Cfsub8B9
7
	full_text*
(
&%891 = fsub double -0.000000e+00, %794
,double8B

	full_text

double %794
mcall8Bc
a
	full_textT
R
P%892 = tail call double @llvm.fmuladd.f64(double %891, double %890, double %805)
,double8B

	full_text

double %891
,double8B

	full_text

double %890
,double8B

	full_text

double %805
Cfsub8B9
7
	full_text*
(
&%893 = fsub double -0.000000e+00, %798
,double8B

	full_text

double %798
mcall8Bc
a
	full_textT
R
P%894 = tail call double @llvm.fmuladd.f64(double %893, double %883, double %892)
,double8B

	full_text

double %893
,double8B

	full_text

double %883
,double8B

	full_text

double %892
Cfsub8B9
7
	full_text*
(
&%895 = fsub double -0.000000e+00, %800
,double8B

	full_text

double %800
mcall8Bc
a
	full_textT
R
P%896 = tail call double @llvm.fmuladd.f64(double %895, double %878, double %894)
,double8B

	full_text

double %895
,double8B

	full_text

double %878
,double8B

	full_text

double %894
Cfsub8B9
7
	full_text*
(
&%897 = fsub double -0.000000e+00, %802
,double8B

	full_text

double %802
mcall8Bc
a
	full_textT
R
P%898 = tail call double @llvm.fmuladd.f64(double %897, double %875, double %896)
,double8B

	full_text

double %897
,double8B

	full_text

double %875
,double8B

	full_text

double %896
:fdiv8B0
.
	full_text!

%899 = fdiv double %898, %788
,double8B

	full_text

double %898
,double8B

	full_text

double %788
Qstore8BF
D
	full_text7
5
3store double %899, double* %544, align 16, !tbaa !8
,double8B

	full_text

double %899
.double*8B

	full_text

double* %544
©getelementptr8B•
’
	full_text„

%900 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%901 = load double, double* %900, align 8, !tbaa !8
.double*8B

	full_text

double* %900
:fsub8B0
.
	full_text!

%902 = fsub double %901, %899
,double8B

	full_text

double %901
,double8B

	full_text

double %899
Pstore8BE
C
	full_text6
4
2store double %902, double* %900, align 8, !tbaa !8
,double8B

	full_text

double %902
.double*8B

	full_text

double* %900
©getelementptr8B•
’
	full_text„

%903 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%904 = load double, double* %903, align 8, !tbaa !8
.double*8B

	full_text

double* %903
:fsub8B0
.
	full_text!

%905 = fsub double %904, %890
,double8B

	full_text

double %904
,double8B

	full_text

double %890
Pstore8BE
C
	full_text6
4
2store double %905, double* %903, align 8, !tbaa !8
,double8B

	full_text

double %905
.double*8B

	full_text

double* %903
©getelementptr8B•
’
	full_text„

%906 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%907 = load double, double* %906, align 8, !tbaa !8
.double*8B

	full_text

double* %906
:fsub8B0
.
	full_text!

%908 = fsub double %907, %883
,double8B

	full_text

double %907
,double8B

	full_text

double %883
Pstore8BE
C
	full_text6
4
2store double %908, double* %906, align 8, !tbaa !8
,double8B

	full_text

double %908
.double*8B

	full_text

double* %906
©getelementptr8B•
’
	full_text„

%909 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%910 = load double, double* %909, align 8, !tbaa !8
.double*8B

	full_text

double* %909
:fsub8B0
.
	full_text!

%911 = fsub double %910, %878
,double8B

	full_text

double %910
,double8B

	full_text

double %878
Pstore8BE
C
	full_text6
4
2store double %911, double* %909, align 8, !tbaa !8
,double8B

	full_text

double %911
.double*8B

	full_text

double* %909
©getelementptr8B•
’
	full_text„

%912 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %58
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%913 = load double, double* %912, align 8, !tbaa !8
.double*8B

	full_text

double* %912
:fsub8B0
.
	full_text!

%914 = fsub double %913, %875
,double8B

	full_text

double %913
,double8B

	full_text

double %875
Pstore8BE
C
	full_text6
4
2store double %914, double* %912, align 8, !tbaa !8
,double8B

	full_text

double %914
.double*8B

	full_text

double* %912
(br8B 

	full_text

br label %915
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %22) #4
%i8*8B

	full_text
	
i8* %22
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %21) #4
%i8*8B

	full_text
	
i8* %21
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %20) #4
%i8*8B

	full_text
	
i8* %20
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %19) #4
%i8*8B

	full_text
	
i8* %19
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %18) #4
%i8*8B

	full_text
	
i8* %18
[call8BQ
O
	full_textB
@
>call void @llvm.lifetime.end.p0i8(i64 200, i8* nonnull %17) #4
%i8*8B

	full_text
	
i8* %17
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %9
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %7
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %8
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
,i648B!

	full_text

i64 4294967296
4double8B&
$
	full_text

double 1.600000e+00
5double8B'
%
	full_text

double -1.000000e-01
:double8B,
*
	full_text

double 0xC0B370D4FDF3B645
4double8B&
$
	full_text

double 1.610000e+02
:double8B,
*
	full_text

double 0x3FB89374BC6A7EF8
:double8B,
*
	full_text

double 0x410FA45000000002
#i648B

	full_text	

i64 4
:double8B,
*
	full_text

double 0xBFC1111111111111
4double8B&
$
	full_text

double 1.400000e+00
:double8B,
*
	full_text

double 0x3FB00AEC33E1F670
%i648B

	full_text
	
i64 200
:double8B,
*
	full_text

double 0x40E9504000000001
$i648B

	full_text


i64 32
:double8B,
*
	full_text

double 0xBFB89374BC6A7EF8
:double8B,
*
	full_text

double 0xBFB00AEC33E1F670
#i648B

	full_text	

i64 2
:double8B,
*
	full_text

double 0xC115183555555556
#i328B

	full_text	

i32 1
4double8B&
$
	full_text

double 1.000000e+00
:double8B,
*
	full_text

double 0xC0E2FC3000000001
5double8B'
%
	full_text

double -4.000000e-01
4double8B&
$
	full_text

double 2.576000e+02
:double8B,
*
	full_text

double 0x40C3D884189374BD
:double8B,
*
	full_text

double 0xC0B9C936F46508DF
#i648B

	full_text	

i64 3
4double8B&
$
	full_text

double 1.000000e-01
4double8B&
$
	full_text

double 0.000000e+00
:double8B,
*
	full_text

double 0xC0E9504000000001
4double8B&
$
	full_text

double 6.440000e+01
:double8B,
*
	full_text

double 0x410FA45800000002
:double8B,
*
	full_text

double 0xC0A96187D9C54A68
4double8B&
$
	full_text

double 8.000000e-01
:double8B,
*
	full_text

double 0xC0D9C936F46508DE
#i648B

	full_text	

i64 1
:double8B,
*
	full_text

double 0x40EDC4C624DD2F1B
:double8B,
*
	full_text

double 0x40CDC4C624DD2F1B
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 40
$i328B

	full_text


i32 -1
4double8B&
$
	full_text

double 4.000000e-01
:double8B,
*
	full_text

double 0x40F5183555555556
:double8B,
*
	full_text

double 0xC0B9C936F46508DE
:double8B,
*
	full_text

double 0xC0D9C936F46508DF
#i648B

	full_text	

i64 0
5double8B'
%
	full_text

double -0.000000e+00
:double8B,
*
	full_text

double 0x3FC1111111111111
:double8B,
*
	full_text

double 0x40B4403333333334
:double8B,
*
	full_text

double 0x3FC916872B020C49
4double8B&
$
	full_text

double 1.200000e+00
5double8B'
%
	full_text

double -4.000000e+00        	
 		                         !" !! #$ #% ## &' && (( )) *+ ** ,- ,. ,, // 01 00 23 24 22 56 57 55 89 8: 88 ;< ;= ;; >? >> @@ AB AC AA DE DG FF HH IJ IK II LM LL NO NP NN QQ RS RT RR UV UW UU XY XZ [[ \\ ]] ^_ ^^ `a `` bc bb de dd fg ff hi hh jk jl jm jn jj op oo qr qs qq tu tv tt wx ww yz yy {| {{ }~ }} €  
‚  ƒ„ ƒƒ …
† …… ‡ˆ ‡‡ ‰
Š ‰‰ ‹Œ ‹‹ Ž 
 
 
‘  ’“ ’’ ”• ”
– ”” —˜ —— ™š ™™ ›œ ›
 ›› žŸ žž  ¡    ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «
¬ «« ­® ­­ ¯
° ¯¯ ±² ±± ³
´ ³³ µ¶ µ
· µ
¸ µ
¹ µµ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ È
É ÈÈ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ ÏÏ Ñ
Ò ÑÑ ÓÔ ÓÓ Õ
Ö ÕÕ ×Ø ×
Ù ×
Ú ×
Û ×× ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ãä ãã åæ å
ç åå èé èè ê
ë êê ìí ìì î
ï îî ðñ ðð òó ò
ô òò õö õõ ÷
ø ÷÷ ùú ù
û ùù üý ü
þ üü ÿ€ ÿÿ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ Œ
Ž Œ
 Œ
 ŒŒ ‘’ ‘‘ “” “
• ““ –— –
˜ –
™ –– š› šš œ œœ žŸ ž
  žž ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿¿ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ ÈÈ ÊË Ê
Ì Ê
Í Ê
Î ÊÊ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×× Ù
Ú ÙÙ ÛÜ ÛÛ Ý
Þ ÝÝ ßà ßß á
â áá ãä ãã å
æ åå çè çç é
ê éé ëì ë
í ë
î ë
ï ëë ðñ ðð òó ò
ô òò õ
ö õõ ÷ø ÷
ù ÷
ú ÷
û ÷÷ üý üü þÿ þþ € €
‚ €€ ƒ„ ƒ
… ƒ
† ƒƒ ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ Ž
 ŽŽ ‘ 
’  “” ““ •– •
— •• ˜™ ˜˜ š› šš œ œœ ž
Ÿ žž  ¡  
¢    £¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ª
­ ª
® ªª ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½
À ½
Á ½½ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ ÐÐ Ò
Ó ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× Ú
Û ÚÚ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ã
ä ãã åæ å
ç åå èé èè êë ê
ì êê íî íí ïð ïï ñò ñ
ó ññ ôõ ôô ö÷ öö ø
ù øø úû ú
ü úú ýþ ýý ÿ€ ÿÿ ‚ 
ƒ  „… „„ †
‡ †† ˆ‰ ˆˆ Š
‹ ŠŠ Œ Œ
Ž ŒŒ  
‘  ’
“ ’’ ”• ”
– ”” —˜ —— ™
š ™™ ›œ ›
 ›› žŸ žž  ¡  
¢    £¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬
­ ¬¬ ®¯ ®® °± °
² °° ³´ ³³ µ
¶ µµ ·¸ ·
¹ ·
º ·
» ·· ¼½ ¼¼ ¾¿ ¾¾ À
Á ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ ÐÐ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ Ø
Ù ØØ ÚÛ Ú
Ü Ú
Ý ÚÚ Þß Þ
à ÞÞ á
â áá ãä ã
å ã
æ ãã çè çç é
ê éé ëì ë
í ë
î ëë ïð ïï ñ
ò ññ óô ó
õ óó ö÷ öö øù ø
ú øø ûü û
ý ûû þÿ þ
€ þþ ‚ 
ƒ 
„  …† …… ‡
ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ Ž Ž
 ŽŽ ‘
’ ‘‘ “” “
• ““ –— –– ˜™ ˜
š ˜˜ ›œ ›› ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥
¦ ¥¥ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´µ ´
¶ ´´ ·
¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ ÃÃ Å
Æ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ ÌÌ ÎÏ Î
Ð ÎÎ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× Õ
Ø Õ
Ù ÕÕ ÚÛ ÚÚ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã ââ ä
å ää æç ææ è
é èè êë êê ì
í ìì îï îî ð
ñ ðð òó òò ô
õ ôô ö÷ ö
ø ö
ù ö
ú öö ûü ûû ýþ ý
ÿ ý
€ ý
 ýý ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š
‹ ŠŠ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “
” ““ •– •
— •• ˜™ ˜˜ š› š
œ šš ž 
Ÿ   ¡    ¢£ ¢¢ ¤
¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «« ­® ­
¯ ­­ °± °
² °° ³´ ³³ µ¶ µµ ·¸ ·
¹ ·· º» ºº ¼
½ ¼¼ ¾¿ ¾¾ À
Á ÀÀ Â
Ã ÂÂ ÄÅ Ä
Æ Ä
Ç Ä
È ÄÄ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ ÎÎ ÐÑ Ð
Ò Ð
Ó ÐÐ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ Û
Ü ÛÛ ÝÞ Ý
ß ÝÝ àá àà âã â
ä ââ åæ åå çè çç éê éé ëì ë
í ëë îï îî ðñ ðð òó òò ô
õ ôô ö÷ ö
ø öö ùú ùù ûü ûû ýþ ý
ÿ ýý € €
‚ €
ƒ €
„ €€ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ ŒŒ Ž ŽŽ ‘ 
’  “” ““ •
– •• —˜ —
™ —— š› š
œ šš 
ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤
¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «
­ «« ®¯ ®® °
± °° ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾¾ À
Á ÀÀ ÂÃ Â
Ä Â
Å Â
Æ ÂÂ ÇÈ ÇÇ ÉÊ ÉÉ Ë
Ì ËË ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ ÛÛ ÝÞ Ý
ß ÝÝ àá à
â àà ã
ä ãã åæ å
ç å
è åå éê é
ë éé ìí ì
î ì
ï ìì ðñ ðð ò
ó òò ôõ ô
ö ô
÷ ôô øù øø ú
û úú üý ü
þ üü ÿ€ ÿÿ ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž
 ŽŽ ‘ 
’  “” ““ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ ž
  ž
¡ žž ¢£ ¢¢ ¤
¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «
­ «« ®
¯ ®® °± °
² °° ³´ ³³ µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½½ À
Á ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ ÌÌ Î
Ï ÎÎ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ ÚÚ ÜÝ ÜÜ Þß Þ
à Þ
á Þ
â ÞÞ ãä ãã åæ å
ç åå èé è
ê èè ëì ëë í
î íí ïð ïï ñ
ò ññ óô óó õ
ö õõ ÷ø ÷÷ ù
ú ùù ûü ûû ý
þ ýý ÿ€	 ÿ
	 ÿ
‚	 ÿ
ƒ	 ÿÿ „	…	 „	„	 †	‡	 †	
ˆ	 †	
‰	 †	
Š	 †	†	 ‹	Œ	 ‹	‹	 	Ž	 	
	 		 	‘	 	
’	 		 “	
”	 “	“	 •	–	 •	•	 —	˜	 —	
™	 —	—	 š	›	 š	š	 œ	
	 œ	œ	 ž	Ÿ	 ž	
 	 ž	ž	 ¡	¢	 ¡	¡	 £	¤	 £	
¥	 £	£	 ¦	§	 ¦	
¨	 ¦	¦	 ©	ª	 ©	©	 «	
¬	 «	«	 ­	®	 ­	
¯	 ­	­	 °	±	 °	°	 ²	³	 ²	²	 ´	µ	 ´	
¶	 ´	´	 ·	¸	 ·	·	 ¹	
º	 ¹	¹	 »	¼	 »	
½	 »	»	 ¾	¿	 ¾	¾	 À	Á	 À	À	 Â	Ã	 Â	
Ä	 Â	Â	 Å	Æ	 Å	Å	 Ç	
È	 Ç	Ç	 É	Ê	 É	
Ë	 É	
Ì	 É	
Í	 É	É	 Î	Ï	 Î	Î	 Ð	Ñ	 Ð	
Ò	 Ð	Ð	 Ó	Ô	 Ó	
Õ	 Ó	Ó	 Ö	
×	 Ö	Ö	 Ø	Ù	 Ø	
Ú	 Ø	Ø	 Û	Ü	 Û	Û	 Ý	
Þ	 Ý	Ý	 ß	à	 ß	
á	 ß	ß	 â	ã	 â	â	 ä	å	 ä	
æ	 ä	ä	 ç	è	 ç	ç	 é	
ê	 é	é	 ë	ì	 ë	ë	 í	î	 í	í	 ï	
ð	 ï	ï	 ñ	ò	 ñ	
ó	 ñ	ñ	 ô	õ	 ô	ô	 ö	÷	 ö	ö	 ø	ù	 ø	
ú	 ø	ø	 û	ü	 û	
ý	 û	û	 þ	ÿ	 þ	þ	 €

 €
€
 ‚
ƒ
 ‚

„
 ‚
‚
 …
†
 …
…
 ‡

ˆ
 ‡
‡
 ‰

Š
 ‰
‰
 ‹
Œ
 ‹


 ‹

Ž
 ‹


 ‹
‹
 
‘
 

 ’
“
 ’

”
 ’
’
 •
–
 •
•
 —
˜
 —

™
 —

š
 —
—
 ›
œ
 ›
›
 
ž
 

Ÿ
 

  
¡
  
 
 ¢

£
 ¢
¢
 ¤
¥
 ¤

¦
 ¤
¤
 §
¨
 §
§
 ©
ª
 ©

«
 ©
©
 ¬
­
 ¬
¬
 ®
¯
 ®
®
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
·
 ¹
º
 ¹
¹
 »
¼
 »

½
 »
»
 ¾
¿
 ¾
¾
 À
Á
 À
À
 Â

Ã
 Â
Â
 Ä
Å
 Ä

Æ
 Ä
Ä
 Ç
È
 Ç
Ç
 É
Ê
 É
É
 Ë
Ì
 Ë

Í
 Ë
Ë
 Î
Ï
 Î
Î
 Ð

Ñ
 Ð
Ð
 Ò
Ó
 Ò

Ô
 Ò

Õ
 Ò

Ö
 Ò
Ò
 ×
Ø
 ×
×
 Ù
Ú
 Ù
Ù
 Û

Ü
 Û
Û
 Ý
Þ
 Ý

ß
 Ý
Ý
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
 ð
ñ
 ð

ò
 ð
ð
 ó

ô
 ó
ó
 õ
ö
 õ

÷
 õ

ø
 õ
õ
 ù
ú
 ù
ù
 û
ü
 û

ý
 û
û
 þ
ÿ
 þ

€ þ

 þ
þ
 ‚ƒ ‚‚ „
… „„ †‡ †
ˆ †
‰ †† Š‹ ŠŠ Œ
 ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –– ˜™ ˜
š ˜˜ ›œ ›› ž 
Ÿ   
¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²
³ ²² ´µ ´
¶ ´´ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä Â
Å ÂÂ ÆÇ ÆÆ È
É ÈÈ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏÐ Ï
Ñ ÏÏ Ò
Ó ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß ÞÞ à
á àà âã â
ä ââ åæ åå çè çç éê é
ë éé ìí ì
î ì
ï ì
ð ìì ñò ññ óô ó
õ ó
ö ó
÷ óó øù øø úû ú
ü ú
ý ú
þ úú ÿ€ ÿÿ ‚ 
ƒ 
„ 
…  †‡ †† ˆ‰ ˆ
Š ˆ
‹ ˆ
Œ ˆˆ Ž    ‘’ ‘‘ “” “
• ““ –— –
˜ –
™ –– š› šš œ œ
ž œ
Ÿ œœ  ¡    ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®® °± °° ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·
º ·· »¼ »» ½¾ ½
¿ ½
À ½½ ÁÂ ÁÁ ÃÄ Ã
Å Ã
Æ ÃÃ ÇÈ ÇÇ ÉÊ É
Ë É
Ì ÉÉ ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú Ø
Û ØØ ÜÝ ÜÜ Þß Þ
à Þ
á ÞÞ âã ââ äå ä
æ ä
ç ää èé èè êë ê
ì ê
í êê îï îî ðñ ðð òó ò
ô òò õö õõ ÷ø ÷÷ ùú ù
û ùù üý ü
þ ü
ÿ üü € €€ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †† ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž Ž
 Ž
‘ ŽŽ ’“ ’’ ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
ž œ
Ÿ œœ  ¡  
¢  
£    ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®® °± °
² °° ³´ ³
µ ³
¶ ³
· ³³ ¸¹ ¸¸ º» º
¼ º
½ º
¾ ºº ¿À ¿¿ ÁÂ Á
Ã Á
Ä Á
Å ÁÁ ÆÇ ÆÆ ÈÉ È
Ê È
Ë È
Ì ÈÈ ÍÎ ÍÍ ÏÐ Ï
Ñ Ï
Ò Ï
Ó ÏÏ ÔÕ ÔÔ Ö× Ö
Ø Ö
Ù Ö
Ú ÖÖ ÛÜ ÛÛ ÝÞ Ý
ß Ý
à Ý
á ÝÝ âã ââ äå ä
æ ä
ç ä
è ää éê éé ëì ë
í ë
î ë
ï ëë ðñ ðð òó ò
ô ò
õ ò
ö òò ÷ø ÷÷ ùú ùù ûü ûû ýþ ý
ÿ ýý € €
‚ €
ƒ €€ „… „„ †‡ †
ˆ †
‰ †† Š‹ ŠŠ Œ Œ
Ž Œ
 ŒŒ ‘  ’“ ’
” ’
• ’’ –— –– ˜™ ˜
š ˜
› ˜˜ œ œœ žŸ ž
  ž
¡ žž ¢£ ¢¢ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨¨ ª« ª
¬ ª
­ ªª ®¯ ®® °± °
² °
³ °° ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» ºº ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ Á
Ã Á
Ä ÁÁ ÅÆ ÅÅ ÇÈ Ç
É Ç
Ê ÇÇ ËÌ ËË ÍÎ Í
Ï Í
Ð ÍÍ ÑÒ ÑÑ ÓÔ Ó
Õ Ó
Ö ÓÓ ×Ø ×× ÙÚ Ù
Û Ù
Ü ÙÙ ÝÞ ÝÝ ßà ß
á ß
â ßß ãä ãã åæ å
ç å
è åå éê éé ëì ë
í ë
î ëë ïð ïï ñò ñ
ó ñ
ô ññ õö õ
÷ õõ øù ø
ú øø ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †† ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž Ž
 Ž
‘ ŽŽ ’“ ’’ ”• ”
– ”
— ”” ˜™ ˜˜ š› š
œ š
 šš žŸ žž  ¡  
¢  
£    ¤¥ ¤¤ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬­ ¬
® ¬
¯ ¬¬ °± °° ²³ ²
´ ²
µ ²² ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å Ã
Æ ÃÃ ÇÈ ÇÇ ÉÊ É
Ë É
Ì ÉÉ ÍÎ ÍÍ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ ÓÓ ÕÖ Õ
× Õ
Ø ÕÕ ÙÚ ÙÙ ÛÜ Û
Ý Û
Þ ÛÛ ßà ßß áâ á
ã á
ä áá åæ åå çè ç
é ç
ê çç ëì ëë íî í
ï í
ð íí ñò ññ óô ó
õ ó
ö óó ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ýý ÿ€ ÿÿ ‚  ƒ„ ƒ
… ƒƒ †‡ †
ˆ †
‰ †† Š‹ ŠŠ Œ Œ
Ž Œ
 ŒŒ ‘  ’“ ’
” ’
• ’’ –— –– ˜™ ˜
š ˜
› ˜˜ œ œœ žŸ ž
  ž
¡ žž ¢£ ¢¢ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨¨ ª« ª
¬ ª
­ ªª ®¯ ®® °± °
² °
³ °° ´µ ´´ ¶· ¶
¸ ¶
¹ ¶¶ º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ ËË ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× ÖÖ ØÙ ØØ ÚÛ ÚÚ ÜÝ Ü
Þ ÜÜ ßà ßß áâ áá ãä ãã åæ åå çè ç
é çç êë êê ìí ìì îï îî ðñ ðð òó ò
ô òò õö õõ ÷ø ÷÷ ùú ùù ûü ûû ýþ ý
ÿ ýý € €€ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž    ‘’ ‘‘ “” “
• ““ –— –– ˜™ ˜˜ š› šš œ œœ žŸ žž  ¡    ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «« ­® ­­ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» ºº ¼½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× ÙÚ ÙÙ ÛÜ ÛÛ ÝÞ ÝÝ ßà ßß áâ áá ãä ã
å ãã æç ææ èé èè êë êê ìí ìì îï îî ðñ ðð òó òò ôõ ôô ö÷ öö øù øø ú
û úú üý üü þÿ þþ € €
‚ €€ ƒ„ ƒƒ …† …… ‡
ˆ ‡‡ ‰Š ‰
‹ ‰
Œ ‰‰ Ž 
  ‘  ’“ ’’ ”• ”
– ”
— ”” ˜™ ˜
š ˜˜ ›œ ›› ž 
Ÿ 
   ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦
¨ ¦
© ¦¦ ª« ª
¬ ªª ­® ­­ ¯° ¯¯ ±
² ±± ³´ ³
µ ³
¶ ³³ ·¸ ·
¹ ·· º» ºº ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ Ã
Ä ÃÃ ÅÆ Å
Ç Å
È ÅÅ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î Ì
Ï ÌÌ ÐÑ Ð
Ò Ð
Ó ÐÐ ÔÕ Ô
Ö Ô
× ÔÔ ØÙ ØØ ÚÛ Ú
Ü Ú
Ý ÚÚ Þß ÞÞ àá à
â àà ã
ä ãã åæ å
ç å
è åå éê é
ë éé ìí ì
î ì
ï ìì ðñ ð
ò ð
ó ðð ôõ ô
ö ô
÷ ôô øù øø úû ú
ü ú
ý úú þÿ þþ € €
‚ €€ ƒ
„ ƒƒ …† …
‡ …
ˆ …… ‰Š ‰
‹ ‰‰ Œ Œ
Ž Œ
 ŒŒ ‘ 
’ 
“  ”• ”
– ”
— ”” ˜™ ˜˜ š› š
œ š
 šš ž
Ÿ žž  ¡  
¢    £
¤ ££ ¥¦ ¥
§ ¥
¨ ¥¥ ©ª ©
« ©© ¬­ ¬
® ¬
¯ ¬¬ °± °
² °° ³´ ³
µ ³
¶ ³³ ·¸ ·
¹ ·· º
» ºº ¼½ ¼
¾ ¼
¿ ¼¼ ÀÁ À
Â ÀÀ Ã
Ä ÃÃ ÅÆ Å
Ç Å
È ÅÅ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î Ì
Ï ÌÌ ÐÑ Ð
Ò Ð
Ó ÐÐ ÔÕ Ô
Ö Ô
× ÔÔ ØÙ Ø
Ú ØØ Û
Ü ÛÛ ÝÞ Ý
ß Ý
à ÝÝ áâ á
ã áá äå ä
æ ä
ç ää èé è
ê è
ë èè ìí ì
î ì
ï ìì ð
ñ ðð òó ò
ô òò õ
ö õõ ÷ø ÷
ù ÷
ú ÷÷ ûü û
ý ûû þÿ þ
€ þ
 þþ ‚ƒ ‚
„ ‚‚ …
† …… ‡ˆ ‡
‰ ‡
Š ‡‡ ‹Œ ‹
 ‹‹ Ž
 ŽŽ ‘ 
’ 
“  ”• ”
– ”” —˜ —
™ —
š —— ›œ ›
 ›
ž ›› Ÿ
  ŸŸ ¡¢ ¡
£ ¡¡ ¤
¥ ¤¤ ¦§ ¦
¨ ¦
© ¦¦ ª« ª
¬ ªª ­
® ­­ ¯° ¯
± ¯
² ¯¯ ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹
º ¹¹ »¼ »
½ »
¾ »» ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ Å
Æ ÅÅ ÇÈ Ç
É Ç
Ê ÇÇ Ë
Ì ËË ÍÎ Í
Ï Í
Ð ÍÍ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×
Ø ×× ÙÚ Ù
Û Ù
Ü ÙÙ Ý
Þ ÝÝ ßà ß
á ß
â ßß ã
ä ãã åæ å
ç å
è åå éê é
ë éé ìí ì
î ìì ï
ð ïï ñò ñ
ó ñ
ô ññ õ
ö õõ ÷ø ÷
ù ÷
ú ÷÷ û
ü ûû ýþ ý
ÿ ý
€ ýý 
‚  ƒ„ ƒ
… ƒ
† ƒƒ ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
 
 
‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› š
œ š
 š
ž šš Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §
ª §
« §§ ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´
· ´
¸ ´´ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã Á
Ä Á
Å ÁÁ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË Î
Ð ÏÏ Ñ
Ò ÑÑ Ó
Ô ÓÓ Õ
Ö ÕÕ ×
Ø ×× Ù
Ú ÙÙ ÛÜ ZÝ [Þ ]ß Hà @á )â Qã /ä \å    
            "! $ %# ') +* -( ./ 10 3  42 6) 75 9 :8 <( =; ?@ B& CA E, GH JF K> MI OL PQ S> TR VN WU Y# _^ a, cb e; gf i] k` ld mh nj po ro so uq v xw z |{ ~ € ‚ „ƒ † ˆ‡ Šq Œ[ Ž` d h ‘ “‹ •’ –” ˜ š— œ™ o Ÿž ¡  £ ¥¢ §¤ ¨ ª© ¬ ®­ ° ²± ´[ ¶` ·d ¸h ¹µ »‹ ½º ¾¼ À Â¿ ÄÁ Å ÇÆ É Ë¢ ÍÊ Î ÐÏ Ò ÔÓ Ö[ Ø` Ùd Úh Û× Ý‹ ßÜ àÞ â äá æã ç éè ë íì ï ñ¢ óð ô öõ ø’ ú’ ûº ýº þü €ù ‚ÿ ƒÜ …Ü †„ ˆ ‰q ‹[ ` Žd h Œ ’Š ”‘ •‡ —t ˜“ ™– › š Ÿœ  q ¢¡ ¤’ ¥ §£ ©¦ ªq ¬« ®º ¯ ±­ ³° ´¡ ¶Ü · ¹µ »¸ ¼o ¾½ À Â¿ ÄÁ Åf ÇÆ É] Ë` Ìd ÍÈ ÎÊ ÐÏ ÒÏ ÓÏ ÕÑ Ö Ø× Ú ÜÛ Þ àß â äã æ èç ê[ ì` íd îÈ ïë ñÏ óð ôò ö\ ø` ùd úÈ û÷ ýü ÿÏ þ ‚õ „ò …€ †Ñ ˆ‡ Šð ‹‰ Œ ƒ ‘Ž ’ ” –“ —ò ™Ï ›š œ Ÿ˜ ¡ž ¢  ¤ ¦£ ¨¥ ©[ «` ¬d ­È ®ª °Ï ²¯ ³± µ´ · ¹¶ »¸ ¼[ ¾` ¿d ÀÈ Á½ ÃÏ ÅÂ ÆÄ ÈÇ Ê ÌÉ ÎË Ï ÑÐ Óð Õ¯ ÖÑ ØÔ Ù× ÛÑ ÝÜ ß¯ àÞ âá äÚ æã ç éå ëè ì± î ðí òï óÏ õô ÷ö ùò ûø üú þ €ý ‚ÿ ƒ …„ ‡ ‰ˆ ‹ð Â ŽÑ Œ ‘ “Ü •Â –” ˜— š’ œ™  Ÿ› ¡ž ¢Ä ¤ ¦£ ¨¥ © «ª ­ ¯ý ±® ² ´³ ¶[ ¸` ¹d ºÈ »· ½¼ ¿¾ Áü ÃÀ ÄÑ Æð ÇÅ ÉÂ ÊÔ Ìð Îð ÏÔ Ñ¯ Ó¯ ÔÐ ÖÒ ×Õ ÙË ÛÍ ÜØ ÝÂ ßÂ àÐ âá äÞ åÚ æÑ èç êé ì¼ íã îë ðï òÈ ôñ õ ÷ó ùö úÏ ü¼ ýÏ ÿü €Í ‚Ñ ƒþ „ †… ˆû Š‡ ‹Ñ Œ ð Ž ’‰ ”‘ • —“ ™– šÔ œÑ ž› ŸÑ ¡  £¯ ¤¢ ¦ ¨¥ © «§ ­ª ®Œ °Ñ ²¯ ³  µÂ ¶´ ¸± º· » ½¹ ¿¼ Àò ÂÏ ÄÃ ÆÁ ÈÅ ÉÇ Ë ÍÊ ÏÌ Ðb ÒÑ Ô] Ö` ×Ó Øh ÙÕ ÛÚ ÝÚ ÞÚ àÜ á ãâ å çæ é ëê í ïî ñ óò õ[ ÷` øÓ ùh úö ü[ þ` ÿÓ €h ý ƒû …‚ †Ü ˆ„ ‰‡ ‹Ü Œ û Ž ’‘ ”Š –“ — ™• ›˜ œÚ ž‚ ŸÚ ¡  £¢ ¥ §¤ ¨¦ ª ¬© ®« ¯Ú ±û ²° ´ ¶³ ¸µ ¹ »º ½ ¿¾ Á Ã\ Å` ÆÓ Çh ÈÄ ÊÚ ÌÉ ÍË ÏÂ Ñ ÒÎ ÓÜ ÕÔ ×‚ ØÖ ÚÙ ÜÐ ÞÛ ß áÝ ãà ä° æå è êç ìé í ïÚ ñð óò õî ÷ô øö ú üù þû ÿ[ ` ‚Ó ƒh „€ †Ú ˆ… ‰‡ ‹Š  Œ ‘Ž ’ ”“ –‚ ˜… ™Ü ›— œš žŒ  … ¡Ÿ £¢ ¥ §¤ ¨ ª¦ ¬© ­ ¯® ±‡ ³ µ² ·´ ¸ º© ¼¹ ½ ¿¾ Á[ Ã` ÄÓ Åh ÆÂ ÈÇ ÊÉ ÌÉ ÎË ÏÜ Ñ‚ ÒÐ ÔÍ Õß ×û Ùû Úß Ü‚ Þ‚ ßÛ áÝ âà äÖ æØ çã è… ê… ëÖ íé îå ïÜ ñð óò õÇ öì ÷ô ùø ûÓ ýú þ €ü ‚ÿ ƒ„ …Ü ‡„ ˆÜ Š‰ Œû ‹ † ‘Ž ’ ” –“ —Ú ™Ç šÜ œÝ É ŸÚ  › ¡ž £¢ ¥˜ §¤ ¨Ü ª© ¬‚ ­« ¯¦ ±® ² ´° ¶³ ·— ¹Ü »¸ ¼‰ ¾… ¿½ Áº ÃÀ Ä ÆÂ ÈÅ É ËÚ ÍÌ ÏÊ ÑÎ ÒÐ Ô ÖÓ ØÕ Ù^ ÛÚ Ý] ßÜ àd áh âÞ äã æã çã éå ê ìë î ðï ò ôó ö ø÷ ú üû þ[ €	Ü 	d ‚	h ƒ	ÿ …	[ ‡	Ü ˆ	d ‰	h Š	†	 Œ	„	 Ž	‹	 	å ‘		 ’		 ”	å –	•	 ˜	„	 ™	—	 ›	š	 	“	 Ÿ	œ	  	 ¢	ž	 ¤	¡	 ¥	ã §	‹	 ¨	ã ª	©	 ¬	¦	 ®	«	 ¯	­	 ±	 ³	°	 µ	²	 ¶	 ¸	·	 º	ã ¼	„	 ½	»	 ¿	 Á	¾	 Ã	À	 Ä	 Æ	Å	 È	[ Ê	Ü Ë	d Ì	h Í	É	 Ï	‹	 Ñ	Î	 Ò	å Ô	Ð	 Õ	Ó	 ×	•	 Ù	Î	 Ú	Ø	 Ü	Û	 Þ	Ö	 à	Ý	 á	 ã	ß	 å	â	 æ	 è	ç	 ê	ã ì	ë	 î	í	 ð	¦	 ò	ï	 ó	ñ	 õ	 ÷	ô	 ù	ö	 ú	ã ü	Î	 ý	û	 ÿ	 
þ	 ƒ
€
 „
 †
…
 ˆ
¦	 Š
\ Œ
Ü 
d Ž
h 
‹
 ‘
ã “

 ”
’
 –
‰
 ˜
¦	 ™
•
 š
å œ
›
 ž
‹	 Ÿ

 ¡
 
 £
—
 ¥
¢
 ¦
 ¨
¤
 ª
§
 «
»	 ­
¬
 ¯
 ±
®
 ³
°
 ´
û	 ¶
µ
 ¸
 º
·
 ¼
¹
 ½
ã ¿
¾
 Á
À
 Ã
¦	 Å
Â
 Æ
Ä
 È
 Ê
Ç
 Ì
É
 Í
 Ï
Î
 Ñ
[ Ó
Ü Ô
d Õ
h Ö
Ò
 Ø
×
 Ú
Ù
 Ü

 Þ
Û
 ß
å á
‹	 â
à
 ä
Ý
 å
è ç
„	 é
„	 ê
è ì
Î	 î
Î	 ï
ë
 ñ
í
 ò
ð
 ô
æ
 ö
è
 ÷
ó
 ø
è ú
‹	 ü
‹	 ý
ù
 ÿ
û
 €õ
 å ƒ‚ …„ ‡×
 ˆþ
 ‰† ‹Š ã
 Œ  ’Ž ”‘ •	 —å ™– šå œ› ž„	 Ÿ ¡˜ £  ¤ ¦¢ ¨¥ ©Ð	 «å ­ª ®› °Î	 ±¯ ³¬ µ² ¶ ¸´ º· »ã ½×
 ¾å Àû
 Á
 Ãã Ä¿ ÅÂ ÇÆ É¼ ËÈ Ìå ÎÍ Ð‹	 ÑÏ ÓÊ ÕÒ Ö ØÔ Ú× Û¦	 Ýã ßÞ áÜ ãà äâ æ èå êç ëZ íÜ îd ïh ðì òZ ôÜ õd öh ÷ó ùZ ûÜ üd ýh þú €Z ‚Ü ƒd „h … ‡Z ‰Ü Šd ‹h Œˆ Žë ï ’‘ ”ø • —ñ ˜“ ™ó ›š ÿ ž– Ÿ÷ ¡  £† ¤œ ¥û §¦ © ª¢ «¨ ­ ¯¡	 ±²	 ³² µø ¶° ¸ñ ¹´ º·	 ¼» ¾ÿ ¿· ÀÀ	 ÂÁ Ä† Å½ ÆÅ	 ÈÇ Ê ËÃ ÌÉ Î Ðâ	 Òç	 ÔÓ Öø ×Ñ Ùñ ÚÕ Ûö	 ÝÜ ßÿ àØ á€
 ãâ å† æÞ ç…
 éè ë ìä íê ï ñî óð ô§
 ö°
 ø÷ úø ûõ ýñ þù ÿ¹
 € ƒÿ „ü …É
 ‡† ‰† Š‚ ‹Î
 Œ  ˆ ‘Ž “ •’ —” ˜¢ šø ›Ž ñ ž™ Ÿ´ ¡ÿ ¢œ £Ô ¥† ¦  §å © ª¤ «¨ ­ ¯¬ ±® ²Z ´` µÓ ¶h ·³ ¹Z »` ¼d ½È ¾º ÀZ Â` ÃÓ Äh ÅÁ ÇZ É` Êd ËÈ ÌÈ ÎZ Ð` ÑÓ Òh ÓÏ ÕZ ×` Ød ÙÈ ÚÖ ÜZ Þ` ßÓ àh áÝ ãZ å` æd çÈ èä êZ ì` íÓ îh ïë ñZ ó` ôd õÈ öò øâ ú× üû þ¿ ÿù ¸ ‚ý ƒæ …„ ‡Æ ˆ€ ‰Û ‹Š Í Ž† ê ‘ “Ô ”Œ •ß —– ™Û š’ ›î œ Ÿâ  ˜ ¡ã £¢ ¥é ¦ž §ò ©¨ «ð ¬¤ ­ç ¯® ±÷ ²ª ³° µ¬ ¶´ ¸® ¹˜ »“ ½¼ ¿¿ Àº Â¸ Ã¾ Ä« ÆÅ ÈÆ ÉÁ Ê¥ ÌË ÎÍ ÏÇ Ðµ ÒÑ ÔÔ ÕÍ Ö¸ Ø× ÚÛ ÛÓ Üº ÞÝ àâ áÙ âË äã æé çß è¾ êé ìð íå îÐ ðï ò÷ óë ôñ öÍ ÷õ ùÏ úà üè þý €¿ û ƒ¸ „ÿ …é ‡† ‰Æ Š‚ ‹ï Œ Í ˆ ‘û “’ •Ô –Ž —ÿ ™˜ ›Û œ” Ž Ÿž ¡â ¢š £„ ¥¤ §é ¨  ©“ «ª ­ð ®¦ ¯ˆ ±° ³÷ ´¬ µ² ·î ¸¶ ºð »© ½ž ¿¾ Á¿ Â¼ Ä¸ ÅÀ Æ® ÈÇ ÊÆ ËÃ Ì¥ ÎÍ ÐÍ ÑÉ Ò´ ÔÓ ÖÔ ×Ï Øª ÚÙ ÜÛ ÝÕ Þ¹ àß ââ ãÛ ä® æå èé éá ê¾ ìë îð ïç ð³ òñ ô÷ õí öó ø’ ù÷ û” ü® þÿ €ö ‚ „¿ …ÿ ‡¸ ˆƒ ‰“ ‹Š Æ Ž† – ‘ “Í ”Œ •³ —– ™Ô š’ ›ª œ ŸÛ  ˜ ¡Å £¢ ¥â ¦ž §¼ ©¨ «é ¬¤ ­Õ ¯® ±ð ²ª ³Ì µ´ ·÷ ¸° ¹¶ »ý ¼º ¾® ¿ ÁÀ Ã ÅÂ ÇÄ È{ ÊÉ Ì ÎÍ ÐË ÒÏ Ó ÕÔ × ÙØ ÛÖ ÝÚ Þƒ àß â äã æá èå é‡ ëê í ïî ñì óð ô™ öõ ø úù ü÷ þû ÿ¤ € ƒ …„ ‡‚ ‰† Š© Œ‹ Ž  ’ ”‘ •­ — ™± › Á Ÿž ¡ £¢ ¥  §¤ ¨Æ ª© ¬ ®­ °« ²¯ ³Ê µ ·Ï ¹ »Ó ½ ¿ã ÁÀ Ã ÅÄ ÇÂ ÉÆ Êè Ì Îì Ð Òð Ô Öõ Ø Úœ ÜÛ Þ àß âÝ äá å¦ ç é° ë í¸ ï ñÁ ó õ ÷ö ùø û ýü ÿú þ ‚‚ „Í †€ ˆ‡ Š… ‹ƒ Œ‰ Ž„  ‘Ø “‡ •’ – —” ™ šã œ‡ ž› Ÿ–   ¢˜ £î ¥‡ §¤ ¨š ©¦ «œ ¬Ï ®® °¯ ²± ´€ µ­ ¶³ ¸Ï ¹ »º ½ú ¿¼ À« Â¾ ÄÃ Æ… ÇÁ ÈÅ Ê­ ËÃ Í’ Î´ ÏÃ Ñ› Ò¸ ÓÃ Õ¤ Ö¼ ×ð Ù± Û¾ ÜØ ÝÂ ßú áÞ âà äã æ… çË èå êÍ ëã í’ îÏ ïã ñ› òÓ óã õ¤ ö× ÷” ù± ûà üø ýÝ ÿú þ ‚€ „ƒ †… ‡æ ˆ… Šè ‹ƒ ’ Žê ƒ ‘› ’î “ƒ •¤ –ò —® ™± ›€ œ˜ ‰ Ÿž ¡Å ¢  ¤£ ¦” §Ì ¨¥ ª¶ «£ ­ ®Ð ¯¬ ±º ²£ ´¦ µÔ ¶³ ¸¾ ¹³ »º ½  ¾Ú ¿ž Áå ÂÀ ÄÃ Æ” Çì ÈÅ ÊÑ ËÃ Í Îð ÏÃ Ñ¦ Òô Óº ÕÀ Öú ×ž Ù… ÚØ ÜÛ Þ” ßŒ àÝ âì ãÛ å æ çÛ é¦ ê” ëº íØ îš ï¥ ñð óÅ ôò öõ ø¬ ùÌ ú÷ üÕ ýõ ÿ³ €Ð þ ƒÙ „¼ †… ˆò ‰Ô Šð ŒÝ ‹ Ž ‘¬ ’ä “ •ð –Ž ˜³ ™è š… œ‹ ì ž÷  Ÿ ¢ £¡ ¥¤ §þ ¨— ©¦ «ô ¬‡ ®­ °¡ ±› ²¯ ´¦ µ³ ·® ¸þ º¹ ¼³ ½‡ ¾» À÷ Á¿ Ã” Ä¬ ÆÅ È¿ É¼ Ê³ ÌË Î³ ÏÇ ÐÍ Ò¥ ÓÑ Õð Ö” Ø× ÚÑ Û³ Ü ÞÝ à¿ áÙ â¦ äã æ³ çß èå ê‰ ëé íÏ î… ðï òé ó¯ ô’ öõ øÑ ùñ ú› üû þ¿ ÿ÷ €¤ ‚ „³ …ý †ƒ ˆø ‰‡ ‹® ŒZ Ž` d h ‘ “’ •‡ –” ˜ ™Z ›` œd h žš  Ÿ ¢é £¡ ¥š ¦Z ¨` ©d ªh «§ ­¬ ¯Ñ °® ²§ ³Z µ` ¶d ·h ¸´ º¹ ¼¿ ½» ¿´ ÀZ Â` Ãd Äh ÅÁ ÇÆ É³ ÊÈ ÌÁ Í Ð Ò Ô Ö Ø ÚD FD ÏX ZX ÏÎ Ï Û ææ éé èè ççÐ èè Ð“ èè “ ææ Ô èè Ôó èè óƒ èè ƒì èè ìŒ èè Œˆ èè ˆÕ èè ÕØ èè Ø° èè °˜ èè ˜ä èè äð èè ðñ èè ñÑ éé Ñ‡ èè ‡ ææ þ
 èè þ
ß èè ßŽ èè Ž¤ èè ¤¶ èè ¶÷ èè ÷Ç èè ÇÐ èè Ð× éé ×€ èè €ô èè ô( çç (Ð èè Ð¤ èè ¤Œ èè ŒÞ èè ÞŽ èè Ž¢ èè ¢—
 èè —
¢ èè ¢Ã èè Ãë èè ë’ èè ’° èè °å èè å° èè °Å èè Å¬ èè ¬ ææ œ èè œ  èè  ² èè ²ª èè ªÙ èè Ù» èè »â èè â èè ¨ èè ¨Í èè Íü èè üÚ èè Ú èè ž èè žþ èè þß	 èè ß	É èè Éó èè óÍ èè Í¶ èè ¶Â èè Â½ èè ½” èè ”¯ èè ¯Ç èè Çå èè å” èè ”Ç èè Ç çç ¦ èè ¦Â èè Âñ èè ñ³ èè ³Ô èè Ôý èè ý ææ  èè Ä
 èè Ä
¦ èè ¦· èè ·Ù èè Ùí èè íú èè úÁ èè Á… èè …‰ èè ‰¼ èè ¼ èè Ù éé ÙÅ èè Å– èè –¨ èè ¨ä èè äô èè ôì èè ì´ èè ´— èè —¬ èè ¬¦ èè ¦ƒ èè ƒ† èè †  èè  ã èè ãö èè öÏ éé Ï› èè ›º èè º	 ææ 	‰ èè ‰ž	 èè ž	ç èè ç´ èè ´ß èè ßÌ èè ÌÝ èè Ý èè  ææ ë èè ëá èè á¤
 èè ¤
Ý
 èè Ý
å èè å• èè •ì èè ì¥ èè ¥Ý èè ÝÊ èè ÊÂ èè Âš èè šú èè úœ èè œŽ èè Ž‚ èè ‚Û èè Û† èè †÷ èè ÷‡ èè ‡ž èè žñ	 èè ñ	 èè › èè ›Ó éé Ó¹ èè ¹Õ éé Õˆ èè ˆ– èè –ž èè ž¤ èè ¤§ èè §¦ èè ¦å èè å  èè  õ
 èè õ
’ èè ’ü èè üÍ èè Í‚ èè ‚ê èè êÓ èè ÓÃ èè ÃÏ èè ÏÚ èè Ú” èè ”½ èè ½É èè ÉŒ èè Œ­	 èè ­	å èè å÷ èè ÷¦ èè ¦¦ èè ¦³ èè ³ª èè ª† èè †Ô èè Ôè èè è èè Ì èè Ìš èè š˜ èè ˜Ð èè Ð  èè  õ èè õ
ê Æ
ê Ñ
ê Ú
ë ˜
ë î
ì Ü
ì Œ
ì •	
í  
í ‰
í ›î Ý
î 
î  
î ¶
î É
î å
î í
î ú
î ›
î £
î ó
î “
î §
î ¹
î Çî ì
î •
î ¦
î ³
î Ý
î ç
î ö
î Œ
î ¦
î ²
î ü
î 
î °
î Â
î Ðî ù
î ž	
î ­	
î ¾	
î ß	
î ñ	
î þ	
î ¤

î ®

î ·

î Ž
î ¢
î ´
î Ô
î â
ï Ö
ï æ

ð ¢
ð ¿
ñ ‡
ñ ±
ñ Ó
ñ õ
ñ Œ
ñ œ
ñ ¦
ñ °
ñ ¸
ñ Á
ñ Á
ñ ç
ñ Ð
ñ ˆ
ñ ³
ñ ·
ñ ö
ñ –
ñ ª
ñ ¼
ñ Ì
ñ Ì
ñ ò
ñ ¾
ñ “
ñ ¾
ñ Â
ñ ÿ
ñ “
ñ ³
ñ Å
ñ Õ
ñ Õ
ñ û
ñ Å	
ñ …

ñ Î

ñ Ò

ñ ‘
ñ ¥
ñ ·
ñ ×
ñ ç
ñ ç
ñ ˆ
ñ ®
ñ ë
ñ ò
ñ î
ñ œ
ñ ¾
ñ Ù
ñ ß
ñ è
ñ ì
ñ ð
ñ ô
ñ ô
ñ Á
ò ‡
ò Ô
ò ›

ó ¾
ó ‰
ó Á
ó É
ó ¦
ó Ê
ó Ù

ó Ê
ó Ü
ô Ë
ô ù
õ 	õ õ õ õ õ Ñõ Óõ Õõ ×õ Ù
ö Œ
ö œ
ö á
ö ö
ö —
ö ï
ö ‘
ö ¢
ö Ù
ö ò
ö ¢
ö ø
ö š	
ö Û	
ö í	
ö  

ö À

ö Š	÷ ^	÷ `	÷ b	÷ d	÷ f	÷ h
÷ È
÷ Ó
÷ Ü
ø Ð
ø ë

ù Û	ú 
ú ©
ú µ
ú Á
ú Æ
ú Ê
ú Ê
ú Ï
ú Ó
ú ì
ú °
ú ß
ú ª
ú ¸
ú è
ú ï
ú ÿ
ú ÿ
ú „
ú ˆ
ú ª
ú ª
ú ê
ú ý
ú µ
ú à
ú é
ú û
ú û
ú Ž
ú “
ú ´
ú ³
ú ó
ú ·	
ú É	
ú â	
ú ç	
ú ö	
ú ö	
ú €

ú …

ú ¹

ú ·
ú ú
ú ð
ú Ï
ú Ö
ú Ø
ú 
ú ¢
ú ­
ú ¶
ú ¶
ú º
ú ¾
ú Ñ
ú ì
ú º
ú §
û —
û ¿
û áü ü ü ü ü ü ü 
ý  
ý ½ý úý žý ðý Ÿþ Ù
þ £
þ ý
þ Êþ ä
þ ©
þ ù
þ Ó
ÿ ´
ÿ Ç
ÿ ›
ÿ ¯
ÿ å
ÿ Š
ÿ „
ÿ ¸
ÿ ¬

ÿ µ

ÿ –
ÿ ª
€ Ä

 Ã
 Ì
 Þ
‚ 
‚ ‡
ƒ ƒ
ƒ ­
ƒ Ï
ƒ ×
ƒ ã
ƒ è
ƒ ì
ƒ ð
ƒ ð
ƒ õ
ƒ ¸
ƒ ã
ƒ ½
ƒ Ë
ƒ „
ƒ ž
ƒ ¥
ƒ ª
ƒ ®
ƒ ®
ƒ ³
ƒ ¼
ƒ î
ƒ º
ƒ €
ƒ Ž
ƒ ©
ƒ ®
ƒ ´
ƒ ¹
ƒ ¹
ƒ ¾
ƒ Å
ƒ ÷
ƒ †	
ƒ À	
ƒ €

ƒ §

ƒ °

ƒ ¹

ƒ É

ƒ É

ƒ Î

ƒ ×
ƒ 
ƒ ”
ƒ Ý
ƒ ä
ƒ ã
ƒ ˜
ƒ º
ƒ Ä
ƒ Í
ƒ Ñ
ƒ Õ
ƒ Õ
ƒ Ù
ƒ ð
ƒ ´
„ ‹
„ ô
„  
„ ë	… }… … …… ‰… «… ¯… ³… È… Ñ… Õ… ê… î… ÷… á… å… é… †… Š… ¬… µ… è… ð… ô… ¼… À… °… À… ñ… õ… ý… ¹	… Ç	… é	… ‡
† í
† °	
† ô	
† Ç

† å‡ Ò‡ •‡ Ð
ˆ y
‰ Œ
‰ ©
‰ Í
Š Â
Š Í
Š Ý

‹ «	Œ !	Œ *	Œ 0	Œ {
Œ 
Œ ™
Œ ¤
Œ ¤
Œ ©
Œ ­
Œ ±
Œ Æ
Œ è
Œ ¦
Œ Û
Œ ë
Œ “
Œ ¥
Œ ¥
Œ ¸
Œ Ë
Œ Ð
Œ ï
Œ ¥
Œ –
Œ æ
Œ ö
Œ ˜
Œ «
Œ «
Œ µ
Œ º
Œ ¾
Œ é
Œ ®
Œ “
Œ ï
Œ ÿ
Œ ¡	
Œ ²	
Œ ²	
Œ ·	
Œ À	
Œ Å	
Œ ç	
Œ °

Œ ¥
Œ ó
Œ Ï
Œ Á
Œ È
Œ Í
Œ ù
Œ „
Œ „
Œ 
Œ ˜
Œ œ
Œ ­
Œ Í
Œ è
Œ ü
Œ š
 ½
Ž Š (	 L  Ï	‘ @	‘ H	‘ Q
’ ž
’ þ
’ …
’ Î
’ ¢
’ •

’ Æ
“  
” ÿ
• ¡	– w	– w	– w	– {	– {	– 	– 
– ƒ
– ƒ
– ‡
– ‡
– ™
– ™
– ¤
– ©
– ­
– ±
– Á
– Á
– Æ
– Ê
– Ï
– Ó
– ã
– ã
– è
– ì
– ð
– õ
– œ
– œ
– ¦
– °
– ¸
– Á
– ×
– ×
– ×
– Û
– Û
– ß
– ß
– ã
– ã
– ç
– ç
– “
– “
– ¥
– ¸
– Ë
– Ð
– è
– è
– ï
– ÿ
– „
– ˆ
– ž
– ž
– ¥
– ª
– ®
– ³
– ö
– ö
– –
– ª
– ¼
– Ì
– â
– â
– â
– æ
– æ
– ê
– ê
– î
– î
– ò
– ò
– ˜
– ˜
– «
– µ
– º
– ¾
– à
– à
– é
– û
– Ž
– “
– ©
– ©
– ®
– ´
– ¹
– ¾
– ÿ
– ÿ
– “
– ³
– Å
– Õ
– ë
– ë
– ë
– ï
– ï
– ó
– ó
– ÷
– ÷
– û
– û
– ¡	
– ¡	
– ²	
– ·	
– À	
– Å	
– â	
– â	
– ç	
– ö	
– €

– …

– §

– §

– °

– ¹

– É

– Î

– ‘
– ‘
– ¥
– ·
– ×
– ç
– ì
– ®
– ®
– Ï
– ð
– ”
– ®
– ³
– º
– Í
– Í
– Ø
– Ø
– ã
– ã
– î
– î
– ù
– „
– 
– ˜
– œ
– ¢
– ­
– ¶
– º
– ¾
– Ä
– Í
– Ñ
– Õ
– Ù
– ß
– è
– ì
– ð
– ô
– ö
– ö
– ö
– ü
– ü
– º
– º
– — õ— Ž— ž— Ú— ã— ø— ’— ™— À— Ø— á— é— ñ— ‡— ‘— ¥— ·— Å— Š— “— ¤— Â— Û— ô— — ¤— Ë— ã— ò— ú— Ž— ¤— ®— À— Î— “	— œ	— «	— Ö	— Ý	— ï	— ‰
— ¢
— Â
— Û
— ó
— „— Œ—  — ²— È— Ò— à— ‡— ±— Ã— ã— ƒ— £— º— Ã— Û— õ— …— Ž— ¤— ­— ¹— Å— Ë— ×— Ý— ã— ï— õ— û— 
˜ š
˜ ð
˜ ¾

™ ©	
š ç
š ð
š ‚
› ¬
› Í
› î
› ’
› ¬
› ´
› õ
› ¶
› ÷
› º
œ š"
buts"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*‡
npb-LU-buts.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

transfer_bytes	
Øº¶è
 
transfer_bytes_log1p
	Œ£A

wgsize_log1p
	Œ£A

devmap_label
 

wgsize
<