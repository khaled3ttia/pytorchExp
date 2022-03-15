
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
 br i1 %40, label %41, label %916
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
 br i1 %49, label %50, label %916
#i18B

	full_text


i1 %49
Wbitcast8BJ
H
	full_text;
9
7%51 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%52 = bitcast double* %1 to [65 x [65 x [5 x double]]]*
Qbitcast8BD
B
	full_text5
3
1%53 = bitcast double* %2 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%54 = bitcast double* %3 to [65 x [65 x double]]*
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
getelementptr8Bz
x
	full_textk
i
g%61 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %54, i64 %56, i64 %58, i64 %60
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %54
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
@store double 0x40E3616000000001, double* %65, align 16, !tbaa !8
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
¢getelementptr8BŽ
‹
	full_text~
|
z%71 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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
Afmul8B7
5
	full_text(
&
$%74 = fmul double %73, -5.292000e+04
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
{call8Bq
o
	full_textb
`
^%77 = tail call double @llvm.fmuladd.f64(double %76, double 1.323000e+04, double 1.000000e+00)
+double8B

	full_text


double %76
Ffadd8B<
:
	full_text-
+
)%78 = fadd double %77, 0x40E3614000000001
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
¢getelementptr8BŽ
‹
	full_text~
|
z%83 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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
Afmul8B7
5
	full_text(
&
$%86 = fmul double %85, -5.292000e+04
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
¢getelementptr8BŽ
‹
	full_text~
|
z%92 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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
)%95 = fmul double %94, 0xC0E9D70000000001
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
call8Bw
u
	full_texth
f
d%99 = tail call double @llvm.fmuladd.f64(double %76, double 0x40C9D70000000001, double 1.000000e+00)
+double8B

	full_text


double %76
Gfadd8B=
;
	full_text.
,
*%100 = fadd double %99, 0x40E3614000000001
+double8B

	full_text


double %99
„getelementptr8Bq
o
	full_textb
`
^%101 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Pstore8BE
C
	full_text6
4
2store double %100, double* %101, align 8, !tbaa !8
,double8B

	full_text

double %100
.double*8B

	full_text

double* %101
„getelementptr8Bq
o
	full_textb
`
^%102 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %102, align 8, !tbaa !8
.double*8B

	full_text

double* %102
8fmul8B.
,
	full_text

%103 = fmul double %72, %72
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

%104 = fmul double %84, %84
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
+%105 = fmul double %104, 0xC08F962D0E560417
,double8B

	full_text

double %104
{call8Bq
o
	full_textb
`
^%106 = tail call double @llvm.fmuladd.f64(double %103, double 0xC08F962D0E560417, double %105)
,double8B

	full_text

double %103
,double8B

	full_text

double %105
8fmul8B.
,
	full_text

%107 = fmul double %93, %93
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
^%108 = tail call double @llvm.fmuladd.f64(double %107, double 0xC08F962D0E560417, double %106)
,double8B

	full_text

double %107
,double8B

	full_text

double %106
Gfmul8B=
;
	full_text.
,
*%109 = fmul double %63, 0x40A23B8B43958106
+double8B

	full_text


double %63
£getelementptr8B
Œ
	full_text
}
{%110 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %60, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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
3%111 = load double, double* %110, align 8, !tbaa !8
.double*8B

	full_text

double* %110
:fmul8B0
.
	full_text!

%112 = fmul double %109, %111
,double8B

	full_text

double %109
,double8B

	full_text

double %111
lcall8Bb
`
	full_textS
Q
O%113 = tail call double @llvm.fmuladd.f64(double %108, double %64, double %112)
,double8B

	full_text

double %108
+double8B

	full_text


double %64
,double8B

	full_text

double %112
Cfmul8B9
7
	full_text*
(
&%114 = fmul double %113, -4.000000e+00
,double8B

	full_text

double %113
„getelementptr8Bq
o
	full_textb
`
^%115 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Qstore8BF
D
	full_text7
5
3store double %114, double* %115, align 16, !tbaa !8
,double8B

	full_text

double %114
.double*8B

	full_text

double* %115
Gfmul8B=
;
	full_text.
,
*%116 = fmul double %63, 0xC0AF962D0E560417
+double8B

	full_text


double %63
9fmul8B/
-
	full_text 

%117 = fmul double %116, %72
,double8B

	full_text

double %116
+double8B

	full_text


double %72
„getelementptr8Bq
o
	full_textb
`
^%118 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Pstore8BE
C
	full_text6
4
2store double %117, double* %118, align 8, !tbaa !8
,double8B

	full_text

double %117
.double*8B

	full_text

double* %118
9fmul8B/
-
	full_text 

%119 = fmul double %116, %84
,double8B

	full_text

double %116
+double8B

	full_text


double %84
„getelementptr8Bq
o
	full_textb
`
^%120 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Qstore8BF
D
	full_text7
5
3store double %119, double* %120, align 16, !tbaa !8
,double8B

	full_text

double %119
.double*8B

	full_text

double* %120
9fmul8B/
-
	full_text 

%121 = fmul double %116, %93
,double8B

	full_text

double %116
+double8B

	full_text


double %93
„getelementptr8Bq
o
	full_textb
`
^%122 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Pstore8BE
C
	full_text6
4
2store double %121, double* %122, align 8, !tbaa !8
,double8B

	full_text

double %121
.double*8B

	full_text

double* %122
‚call8Bx
v
	full_texti
g
e%123 = tail call double @llvm.fmuladd.f64(double %62, double 0x40C23B8B43958106, double 1.000000e+00)
+double8B

	full_text


double %62
Hfadd8B>
<
	full_text/
-
+%124 = fadd double %123, 0x40E3614000000001
,double8B

	full_text

double %123
„getelementptr8Bq
o
	full_textb
`
^%125 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %14, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Qstore8BF
D
	full_text7
5
3store double %124, double* %125, align 16, !tbaa !8
,double8B

	full_text

double %124
.double*8B

	full_text

double* %125
:add8B1
/
	full_text"
 
%126 = add i64 %59, 4294967296
%i648B

	full_text
	
i64 %59
;ashr8B1
/
	full_text"
 
%127 = ashr exact i64 %126, 32
&i648B

	full_text


i64 %126
getelementptr8B|
z
	full_textm
k
i%128 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %54, i64 %56, i64 %58, i64 %127
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %54
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


i64 %127
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
:fmul8B0
.
	full_text!

%130 = fmul double %129, %129
,double8B

	full_text

double %129
,double8B

	full_text

double %129
:fmul8B0
.
	full_text!

%131 = fmul double %129, %130
,double8B

	full_text

double %129
,double8B

	full_text

double %130
„getelementptr8Bq
o
	full_textb
`
^%132 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
_store8BT
R
	full_textE
C
Astore double 0xC0B7418000000001, double* %132, align 16, !tbaa !8
.double*8B

	full_text

double* %132
„getelementptr8Bq
o
	full_textb
`
^%133 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 6.300000e+01, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
„getelementptr8Bq
o
	full_textb
`
^%134 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %134, align 16, !tbaa !8
.double*8B

	full_text

double* %134
„getelementptr8Bq
o
	full_textb
`
^%135 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %135, align 8, !tbaa !8
.double*8B

	full_text

double* %135
„getelementptr8Bq
o
	full_textb
`
^%136 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %136, align 16, !tbaa !8
.double*8B

	full_text

double* %136
¥getelementptr8B‘
Ž
	full_text€
~
|%137 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %127, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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


i64 %127
Pload8BF
D
	full_text7
5
3%138 = load double, double* %137, align 8, !tbaa !8
.double*8B

	full_text

double* %137
:fmul8B0
.
	full_text!

%139 = fmul double %129, %138
,double8B

	full_text

double %129
,double8B

	full_text

double %138
Cfsub8B9
7
	full_text*
(
&%140 = fsub double -0.000000e+00, %139
,double8B

	full_text

double %139
getelementptr8B|
z
	full_textm
k
i%141 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %53, i64 %56, i64 %58, i64 %127
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %53
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


i64 %127
Pload8BF
D
	full_text7
5
3%142 = load double, double* %141, align 8, !tbaa !8
.double*8B

	full_text

double* %141
Bfmul8B8
6
	full_text)
'
%%143 = fmul double %142, 4.000000e-01
,double8B

	full_text

double %142
:fmul8B0
.
	full_text!

%144 = fmul double %129, %143
,double8B

	full_text

double %129
,double8B

	full_text

double %143
mcall8Bc
a
	full_textT
R
P%145 = tail call double @llvm.fmuladd.f64(double %140, double %139, double %144)
,double8B

	full_text

double %140
,double8B

	full_text

double %139
,double8B

	full_text

double %144
Hfmul8B>
<
	full_text/
-
+%146 = fmul double %130, 0xBFC1111111111111
,double8B

	full_text

double %130
:fmul8B0
.
	full_text!

%147 = fmul double %146, %138
,double8B

	full_text

double %146
,double8B

	full_text

double %138
Hfmul8B>
<
	full_text/
-
+%148 = fmul double %147, 0x40BF020000000001
,double8B

	full_text

double %147
Cfsub8B9
7
	full_text*
(
&%149 = fsub double -0.000000e+00, %148
,double8B

	full_text

double %148
ucall8Bk
i
	full_text\
Z
X%150 = tail call double @llvm.fmuladd.f64(double %145, double 6.300000e+01, double %149)
,double8B

	full_text

double %145
,double8B

	full_text

double %149
„getelementptr8Bq
o
	full_textb
`
^%151 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %150, double* %151, align 8, !tbaa !8
,double8B

	full_text

double %150
.double*8B

	full_text

double* %151
Bfmul8B8
6
	full_text)
'
%%152 = fmul double %139, 1.600000e+00
,double8B

	full_text

double %139
Hfmul8B>
<
	full_text/
-
+%153 = fmul double %129, 0x3FC1111111111111
,double8B

	full_text

double %129
Hfmul8B>
<
	full_text/
-
+%154 = fmul double %153, 0x40BF020000000001
,double8B

	full_text

double %153
Cfsub8B9
7
	full_text*
(
&%155 = fsub double -0.000000e+00, %154
,double8B

	full_text

double %154
ucall8Bk
i
	full_text\
Z
X%156 = tail call double @llvm.fmuladd.f64(double %152, double 6.300000e+01, double %155)
,double8B

	full_text

double %152
,double8B

	full_text

double %155
Hfadd8B>
<
	full_text/
-
+%157 = fadd double %156, 0xC0B7418000000001
,double8B

	full_text

double %156
„getelementptr8Bq
o
	full_textb
`
^%158 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %157, double* %158, align 8, !tbaa !8
,double8B

	full_text

double %157
.double*8B

	full_text

double* %158
¥getelementptr8B‘
Ž
	full_text€
~
|%159 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %127, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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


i64 %127
Pload8BF
D
	full_text7
5
3%160 = load double, double* %159, align 8, !tbaa !8
.double*8B

	full_text

double* %159
:fmul8B0
.
	full_text!

%161 = fmul double %129, %160
,double8B

	full_text

double %129
,double8B

	full_text

double %160
Cfmul8B9
7
	full_text*
(
&%162 = fmul double %161, -4.000000e-01
,double8B

	full_text

double %161
Bfmul8B8
6
	full_text)
'
%%163 = fmul double %162, 6.300000e+01
,double8B

	full_text

double %162
„getelementptr8Bq
o
	full_textb
`
^%164 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
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
¥getelementptr8B‘
Ž
	full_text€
~
|%165 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %127, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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


i64 %127
Pload8BF
D
	full_text7
5
3%166 = load double, double* %165, align 8, !tbaa !8
.double*8B

	full_text

double* %165
:fmul8B0
.
	full_text!

%167 = fmul double %129, %166
,double8B

	full_text

double %129
,double8B

	full_text

double %166
Cfmul8B9
7
	full_text*
(
&%168 = fmul double %167, -4.000000e-01
,double8B

	full_text

double %167
Bfmul8B8
6
	full_text)
'
%%169 = fmul double %168, 6.300000e+01
,double8B

	full_text

double %168
„getelementptr8Bq
o
	full_textb
`
^%170 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
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
„getelementptr8Bq
o
	full_textb
`
^%171 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
^store8BS
Q
	full_textD
B
@store double 0x4039333333333334, double* %171, align 8, !tbaa !8
.double*8B

	full_text

double* %171
:fmul8B0
.
	full_text!

%172 = fmul double %138, %160
,double8B

	full_text

double %138
,double8B

	full_text

double %160
:fmul8B0
.
	full_text!

%173 = fmul double %130, %172
,double8B

	full_text

double %130
,double8B

	full_text

double %172
Cfsub8B9
7
	full_text*
(
&%174 = fsub double -0.000000e+00, %173
,double8B

	full_text

double %173
Cfmul8B9
7
	full_text*
(
&%175 = fmul double %130, -1.000000e-01
,double8B

	full_text

double %130
:fmul8B0
.
	full_text!

%176 = fmul double %175, %160
,double8B

	full_text

double %175
,double8B

	full_text

double %160
Hfmul8B>
<
	full_text/
-
+%177 = fmul double %176, 0x40BF020000000001
,double8B

	full_text

double %176
Cfsub8B9
7
	full_text*
(
&%178 = fsub double -0.000000e+00, %177
,double8B

	full_text

double %177
ucall8Bk
i
	full_text\
Z
X%179 = tail call double @llvm.fmuladd.f64(double %174, double 6.300000e+01, double %178)
,double8B

	full_text

double %174
,double8B

	full_text

double %178
„getelementptr8Bq
o
	full_textb
`
^%180 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %179, double* %180, align 16, !tbaa !8
,double8B

	full_text

double %179
.double*8B

	full_text

double* %180
Bfmul8B8
6
	full_text)
'
%%181 = fmul double %161, 6.300000e+01
,double8B

	full_text

double %161
„getelementptr8Bq
o
	full_textb
`
^%182 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %181, double* %182, align 8, !tbaa !8
,double8B

	full_text

double %181
.double*8B

	full_text

double* %182
Bfmul8B8
6
	full_text)
'
%%183 = fmul double %129, 1.000000e-01
,double8B

	full_text

double %129
Hfmul8B>
<
	full_text/
-
+%184 = fmul double %183, 0x40BF020000000001
,double8B

	full_text

double %183
Cfsub8B9
7
	full_text*
(
&%185 = fsub double -0.000000e+00, %184
,double8B

	full_text

double %184
ucall8Bk
i
	full_text\
Z
X%186 = tail call double @llvm.fmuladd.f64(double %139, double 6.300000e+01, double %185)
,double8B

	full_text

double %139
,double8B

	full_text

double %185
Hfadd8B>
<
	full_text/
-
+%187 = fadd double %186, 0xC0B7418000000001
,double8B

	full_text

double %186
„getelementptr8Bq
o
	full_textb
`
^%188 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %187, double* %188, align 16, !tbaa !8
,double8B

	full_text

double %187
.double*8B

	full_text

double* %188
„getelementptr8Bq
o
	full_textb
`
^%189 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
„getelementptr8Bq
o
	full_textb
`
^%190 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %190, align 16, !tbaa !8
.double*8B

	full_text

double* %190
:fmul8B0
.
	full_text!

%191 = fmul double %138, %166
,double8B

	full_text

double %138
,double8B

	full_text

double %166
:fmul8B0
.
	full_text!

%192 = fmul double %130, %191
,double8B

	full_text

double %130
,double8B

	full_text

double %191
Cfsub8B9
7
	full_text*
(
&%193 = fsub double -0.000000e+00, %192
,double8B

	full_text

double %192
:fmul8B0
.
	full_text!

%194 = fmul double %175, %166
,double8B

	full_text

double %175
,double8B

	full_text

double %166
Hfmul8B>
<
	full_text/
-
+%195 = fmul double %194, 0x40BF020000000001
,double8B

	full_text

double %194
Cfsub8B9
7
	full_text*
(
&%196 = fsub double -0.000000e+00, %195
,double8B

	full_text

double %195
ucall8Bk
i
	full_text\
Z
X%197 = tail call double @llvm.fmuladd.f64(double %193, double 6.300000e+01, double %196)
,double8B

	full_text

double %193
,double8B

	full_text

double %196
„getelementptr8Bq
o
	full_textb
`
^%198 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %197, double* %198, align 8, !tbaa !8
,double8B

	full_text

double %197
.double*8B

	full_text

double* %198
Bfmul8B8
6
	full_text)
'
%%199 = fmul double %167, 6.300000e+01
,double8B

	full_text

double %167
„getelementptr8Bq
o
	full_textb
`
^%200 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %199, double* %200, align 8, !tbaa !8
,double8B

	full_text

double %199
.double*8B

	full_text

double* %200
„getelementptr8Bq
o
	full_textb
`
^%201 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %201, align 8, !tbaa !8
.double*8B

	full_text

double* %201
„getelementptr8Bq
o
	full_textb
`
^%202 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %187, double* %202, align 8, !tbaa !8
,double8B

	full_text

double %187
.double*8B

	full_text

double* %202
„getelementptr8Bq
o
	full_textb
`
^%203 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %203, align 8, !tbaa !8
.double*8B

	full_text

double* %203
¥getelementptr8B‘
Ž
	full_text€
~
|%204 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %58, i64 %127, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
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


i64 %127
Pload8BF
D
	full_text7
5
3%205 = load double, double* %204, align 8, !tbaa !8
.double*8B

	full_text

double* %204
Bfmul8B8
6
	full_text)
'
%%206 = fmul double %205, 1.400000e+00
,double8B

	full_text

double %205
Cfsub8B9
7
	full_text*
(
&%207 = fsub double -0.000000e+00, %206
,double8B

	full_text

double %206
ucall8Bk
i
	full_text\
Z
X%208 = tail call double @llvm.fmuladd.f64(double %142, double 8.000000e-01, double %207)
,double8B

	full_text

double %142
,double8B

	full_text

double %207
:fmul8B0
.
	full_text!

%209 = fmul double %130, %138
,double8B

	full_text

double %130
,double8B

	full_text

double %138
:fmul8B0
.
	full_text!

%210 = fmul double %209, %208
,double8B

	full_text

double %209
,double8B

	full_text

double %208
Hfmul8B>
<
	full_text/
-
+%211 = fmul double %131, 0x3FB00AEC33E1F670
,double8B

	full_text

double %131
:fmul8B0
.
	full_text!

%212 = fmul double %138, %138
,double8B

	full_text

double %138
,double8B

	full_text

double %138
Hfmul8B>
<
	full_text/
-
+%213 = fmul double %131, 0xBFB89374BC6A7EF8
,double8B

	full_text

double %131
:fmul8B0
.
	full_text!

%214 = fmul double %160, %160
,double8B

	full_text

double %160
,double8B

	full_text

double %160
:fmul8B0
.
	full_text!

%215 = fmul double %213, %214
,double8B

	full_text

double %213
,double8B

	full_text

double %214
Cfsub8B9
7
	full_text*
(
&%216 = fsub double -0.000000e+00, %215
,double8B

	full_text

double %215
mcall8Bc
a
	full_textT
R
P%217 = tail call double @llvm.fmuladd.f64(double %211, double %212, double %216)
,double8B

	full_text

double %211
,double8B

	full_text

double %212
,double8B

	full_text

double %216
:fmul8B0
.
	full_text!

%218 = fmul double %166, %166
,double8B

	full_text

double %166
,double8B

	full_text

double %166
Cfsub8B9
7
	full_text*
(
&%219 = fsub double -0.000000e+00, %213
,double8B

	full_text

double %213
mcall8Bc
a
	full_textT
R
P%220 = tail call double @llvm.fmuladd.f64(double %219, double %218, double %217)
,double8B

	full_text

double %219
,double8B

	full_text

double %218
,double8B

	full_text

double %217
Hfmul8B>
<
	full_text/
-
+%221 = fmul double %130, 0x3FC916872B020C49
,double8B

	full_text

double %130
Cfsub8B9
7
	full_text*
(
&%222 = fsub double -0.000000e+00, %221
,double8B

	full_text

double %221
mcall8Bc
a
	full_textT
R
P%223 = tail call double @llvm.fmuladd.f64(double %222, double %205, double %220)
,double8B

	full_text

double %222
,double8B

	full_text

double %205
,double8B

	full_text

double %220
Hfmul8B>
<
	full_text/
-
+%224 = fmul double %223, 0x40BF020000000001
,double8B

	full_text

double %223
Cfsub8B9
7
	full_text*
(
&%225 = fsub double -0.000000e+00, %224
,double8B

	full_text

double %224
ucall8Bk
i
	full_text\
Z
X%226 = tail call double @llvm.fmuladd.f64(double %210, double 6.300000e+01, double %225)
,double8B

	full_text

double %210
,double8B

	full_text

double %225
„getelementptr8Bq
o
	full_textb
`
^%227 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %226, double* %227, align 16, !tbaa !8
,double8B

	full_text

double %226
.double*8B

	full_text

double* %227
:fmul8B0
.
	full_text!

%228 = fmul double %129, %205
,double8B

	full_text

double %129
,double8B

	full_text

double %205
:fmul8B0
.
	full_text!

%229 = fmul double %129, %142
,double8B

	full_text

double %129
,double8B

	full_text

double %142
mcall8Bc
a
	full_textT
R
P%230 = tail call double @llvm.fmuladd.f64(double %212, double %130, double %229)
,double8B

	full_text

double %212
,double8B

	full_text

double %130
,double8B

	full_text

double %229
Bfmul8B8
6
	full_text)
'
%%231 = fmul double %230, 4.000000e-01
,double8B

	full_text

double %230
Cfsub8B9
7
	full_text*
(
&%232 = fsub double -0.000000e+00, %231
,double8B

	full_text

double %231
ucall8Bk
i
	full_text\
Z
X%233 = tail call double @llvm.fmuladd.f64(double %228, double 1.400000e+00, double %232)
,double8B

	full_text

double %228
,double8B

	full_text

double %232
Hfmul8B>
<
	full_text/
-
+%234 = fmul double %130, 0xC07F172B020C49B9
,double8B

	full_text

double %130
:fmul8B0
.
	full_text!

%235 = fmul double %234, %138
,double8B

	full_text

double %234
,double8B

	full_text

double %138
Cfsub8B9
7
	full_text*
(
&%236 = fsub double -0.000000e+00, %235
,double8B

	full_text

double %235
ucall8Bk
i
	full_text\
Z
X%237 = tail call double @llvm.fmuladd.f64(double %233, double 6.300000e+01, double %236)
,double8B

	full_text

double %233
,double8B

	full_text

double %236
„getelementptr8Bq
o
	full_textb
`
^%238 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %237, double* %238, align 8, !tbaa !8
,double8B

	full_text

double %237
.double*8B

	full_text

double* %238
Cfmul8B9
7
	full_text*
(
&%239 = fmul double %172, -4.000000e-01
,double8B

	full_text

double %172
:fmul8B0
.
	full_text!

%240 = fmul double %130, %239
,double8B

	full_text

double %130
,double8B

	full_text

double %239
Hfmul8B>
<
	full_text/
-
+%241 = fmul double %130, 0xC087D0624DD2F1A9
,double8B

	full_text

double %130
:fmul8B0
.
	full_text!

%242 = fmul double %241, %160
,double8B

	full_text

double %241
,double8B

	full_text

double %160
Cfsub8B9
7
	full_text*
(
&%243 = fsub double -0.000000e+00, %242
,double8B

	full_text

double %242
ucall8Bk
i
	full_text\
Z
X%244 = tail call double @llvm.fmuladd.f64(double %240, double 6.300000e+01, double %243)
,double8B

	full_text

double %240
,double8B

	full_text

double %243
„getelementptr8Bq
o
	full_textb
`
^%245 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %244, double* %245, align 16, !tbaa !8
,double8B

	full_text

double %244
.double*8B

	full_text

double* %245
Cfmul8B9
7
	full_text*
(
&%246 = fmul double %191, -4.000000e-01
,double8B

	full_text

double %191
:fmul8B0
.
	full_text!

%247 = fmul double %130, %246
,double8B

	full_text

double %130
,double8B

	full_text

double %246
:fmul8B0
.
	full_text!

%248 = fmul double %241, %166
,double8B

	full_text

double %241
,double8B

	full_text

double %166
Cfsub8B9
7
	full_text*
(
&%249 = fsub double -0.000000e+00, %248
,double8B

	full_text

double %248
ucall8Bk
i
	full_text\
Z
X%250 = tail call double @llvm.fmuladd.f64(double %247, double 6.300000e+01, double %249)
,double8B

	full_text

double %247
,double8B

	full_text

double %249
„getelementptr8Bq
o
	full_textb
`
^%251 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Pstore8BE
C
	full_text6
4
2store double %250, double* %251, align 8, !tbaa !8
,double8B

	full_text

double %250
.double*8B

	full_text

double* %251
Bfmul8B8
6
	full_text)
'
%%252 = fmul double %139, 1.400000e+00
,double8B

	full_text

double %139
Hfmul8B>
<
	full_text/
-
+%253 = fmul double %129, 0x40984F645A1CAC08
,double8B

	full_text

double %129
Cfsub8B9
7
	full_text*
(
&%254 = fsub double -0.000000e+00, %253
,double8B

	full_text

double %253
ucall8Bk
i
	full_text\
Z
X%255 = tail call double @llvm.fmuladd.f64(double %252, double 6.300000e+01, double %254)
,double8B

	full_text

double %252
,double8B

	full_text

double %254
Hfadd8B>
<
	full_text/
-
+%256 = fadd double %255, 0xC0B7418000000001
,double8B

	full_text

double %255
„getelementptr8Bq
o
	full_textb
`
^%257 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %11, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %11
Qstore8BF
D
	full_text7
5
3store double %256, double* %257, align 16, !tbaa !8
,double8B

	full_text

double %256
.double*8B

	full_text

double* %257
:add8B1
/
	full_text"
 
%258 = add i64 %57, 4294967296
%i648B

	full_text
	
i64 %57
;ashr8B1
/
	full_text"
 
%259 = ashr exact i64 %258, 32
&i648B

	full_text


i64 %258
getelementptr8B|
z
	full_textm
k
i%260 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %54, i64 %56, i64 %259, i64 %60
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %54
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%261 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
:fmul8B0
.
	full_text!

%262 = fmul double %261, %261
,double8B

	full_text

double %261
,double8B

	full_text

double %261
:fmul8B0
.
	full_text!

%263 = fmul double %261, %262
,double8B

	full_text

double %261
,double8B

	full_text

double %262
„getelementptr8Bq
o
	full_textb
`
^%264 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
_store8BT
R
	full_textE
C
Astore double 0xC0B7418000000001, double* %264, align 16, !tbaa !8
.double*8B

	full_text

double* %264
„getelementptr8Bq
o
	full_textb
`
^%265 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %265, align 8, !tbaa !8
.double*8B

	full_text

double* %265
„getelementptr8Bq
o
	full_textb
`
^%266 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ystore8BN
L
	full_text?
=
;store double 6.300000e+01, double* %266, align 16, !tbaa !8
.double*8B

	full_text

double* %266
„getelementptr8Bq
o
	full_textb
`
^%267 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %267, align 8, !tbaa !8
.double*8B

	full_text

double* %267
„getelementptr8Bq
o
	full_textb
`
^%268 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %268, align 16, !tbaa !8
.double*8B

	full_text

double* %268
¥getelementptr8B‘
Ž
	full_text€
~
|%269 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %259, i64 %60, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%270 = load double, double* %269, align 8, !tbaa !8
.double*8B

	full_text

double* %269
¥getelementptr8B‘
Ž
	full_text€
~
|%271 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %259, i64 %60, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%272 = load double, double* %271, align 8, !tbaa !8
.double*8B

	full_text

double* %271
:fmul8B0
.
	full_text!

%273 = fmul double %270, %272
,double8B

	full_text

double %270
,double8B

	full_text

double %272
:fmul8B0
.
	full_text!

%274 = fmul double %262, %273
,double8B

	full_text

double %262
,double8B

	full_text

double %273
Cfsub8B9
7
	full_text*
(
&%275 = fsub double -0.000000e+00, %274
,double8B

	full_text

double %274
Cfmul8B9
7
	full_text*
(
&%276 = fmul double %262, -1.000000e-01
,double8B

	full_text

double %262
:fmul8B0
.
	full_text!

%277 = fmul double %276, %270
,double8B

	full_text

double %276
,double8B

	full_text

double %270
Hfmul8B>
<
	full_text/
-
+%278 = fmul double %277, 0x40BF020000000001
,double8B

	full_text

double %277
Cfsub8B9
7
	full_text*
(
&%279 = fsub double -0.000000e+00, %278
,double8B

	full_text

double %278
ucall8Bk
i
	full_text\
Z
X%280 = tail call double @llvm.fmuladd.f64(double %275, double 6.300000e+01, double %279)
,double8B

	full_text

double %275
,double8B

	full_text

double %279
„getelementptr8Bq
o
	full_textb
`
^%281 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %280, double* %281, align 8, !tbaa !8
,double8B

	full_text

double %280
.double*8B

	full_text

double* %281
:fmul8B0
.
	full_text!

%282 = fmul double %261, %272
,double8B

	full_text

double %261
,double8B

	full_text

double %272
Bfmul8B8
6
	full_text)
'
%%283 = fmul double %261, 1.000000e-01
,double8B

	full_text

double %261
Hfmul8B>
<
	full_text/
-
+%284 = fmul double %283, 0x40BF020000000001
,double8B

	full_text

double %283
Cfsub8B9
7
	full_text*
(
&%285 = fsub double -0.000000e+00, %284
,double8B

	full_text

double %284
ucall8Bk
i
	full_text\
Z
X%286 = tail call double @llvm.fmuladd.f64(double %282, double 6.300000e+01, double %285)
,double8B

	full_text

double %282
,double8B

	full_text

double %285
Hfadd8B>
<
	full_text/
-
+%287 = fadd double %286, 0xC0B7418000000001
,double8B

	full_text

double %286
„getelementptr8Bq
o
	full_textb
`
^%288 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %287, double* %288, align 8, !tbaa !8
,double8B

	full_text

double %287
.double*8B

	full_text

double* %288
:fmul8B0
.
	full_text!

%289 = fmul double %261, %270
,double8B

	full_text

double %261
,double8B

	full_text

double %270
Bfmul8B8
6
	full_text)
'
%%290 = fmul double %289, 6.300000e+01
,double8B

	full_text

double %289
„getelementptr8Bq
o
	full_textb
`
^%291 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %290, double* %291, align 8, !tbaa !8
,double8B

	full_text

double %290
.double*8B

	full_text

double* %291
„getelementptr8Bq
o
	full_textb
`
^%292 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 1
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
„getelementptr8Bq
o
	full_textb
`
^%293 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %293, align 8, !tbaa !8
.double*8B

	full_text

double* %293
Cfsub8B9
7
	full_text*
(
&%294 = fsub double -0.000000e+00, %282
,double8B

	full_text

double %282
getelementptr8B|
z
	full_textm
k
i%295 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %53, i64 %56, i64 %259, i64 %60
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %53
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
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

%297 = fmul double %261, %296
,double8B

	full_text

double %261
,double8B

	full_text

double %296
Bfmul8B8
6
	full_text)
'
%%298 = fmul double %297, 4.000000e-01
,double8B

	full_text

double %297
mcall8Bc
a
	full_textT
R
P%299 = tail call double @llvm.fmuladd.f64(double %294, double %282, double %298)
,double8B

	full_text

double %294
,double8B

	full_text

double %282
,double8B

	full_text

double %298
Hfmul8B>
<
	full_text/
-
+%300 = fmul double %262, 0xBFC1111111111111
,double8B

	full_text

double %262
:fmul8B0
.
	full_text!

%301 = fmul double %300, %272
,double8B

	full_text

double %300
,double8B

	full_text

double %272
Hfmul8B>
<
	full_text/
-
+%302 = fmul double %301, 0x40BF020000000001
,double8B

	full_text

double %301
Cfsub8B9
7
	full_text*
(
&%303 = fsub double -0.000000e+00, %302
,double8B

	full_text

double %302
ucall8Bk
i
	full_text\
Z
X%304 = tail call double @llvm.fmuladd.f64(double %299, double 6.300000e+01, double %303)
,double8B

	full_text

double %299
,double8B

	full_text

double %303
„getelementptr8Bq
o
	full_textb
`
^%305 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %304, double* %305, align 16, !tbaa !8
,double8B

	full_text

double %304
.double*8B

	full_text

double* %305
Cfmul8B9
7
	full_text*
(
&%306 = fmul double %289, -4.000000e-01
,double8B

	full_text

double %289
Bfmul8B8
6
	full_text)
'
%%307 = fmul double %306, 6.300000e+01
,double8B

	full_text

double %306
„getelementptr8Bq
o
	full_textb
`
^%308 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %307, double* %308, align 8, !tbaa !8
,double8B

	full_text

double %307
.double*8B

	full_text

double* %308
Bfmul8B8
6
	full_text)
'
%%309 = fmul double %282, 1.600000e+00
,double8B

	full_text

double %282
Hfmul8B>
<
	full_text/
-
+%310 = fmul double %261, 0x3FC1111111111111
,double8B

	full_text

double %261
Hfmul8B>
<
	full_text/
-
+%311 = fmul double %310, 0x40BF020000000001
,double8B

	full_text

double %310
Cfsub8B9
7
	full_text*
(
&%312 = fsub double -0.000000e+00, %311
,double8B

	full_text

double %311
ucall8Bk
i
	full_text\
Z
X%313 = tail call double @llvm.fmuladd.f64(double %309, double 6.300000e+01, double %312)
,double8B

	full_text

double %309
,double8B

	full_text

double %312
Hfadd8B>
<
	full_text/
-
+%314 = fadd double %313, 0xC0B7418000000001
,double8B

	full_text

double %313
„getelementptr8Bq
o
	full_textb
`
^%315 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %314, double* %315, align 16, !tbaa !8
,double8B

	full_text

double %314
.double*8B

	full_text

double* %315
¥getelementptr8B‘
Ž
	full_text€
~
|%316 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %259, i64 %60, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%317 = load double, double* %316, align 8, !tbaa !8
.double*8B

	full_text

double* %316
:fmul8B0
.
	full_text!

%318 = fmul double %261, %317
,double8B

	full_text

double %261
,double8B

	full_text

double %317
Cfmul8B9
7
	full_text*
(
&%319 = fmul double %318, -4.000000e-01
,double8B

	full_text

double %318
Bfmul8B8
6
	full_text)
'
%%320 = fmul double %319, 6.300000e+01
,double8B

	full_text

double %319
„getelementptr8Bq
o
	full_textb
`
^%321 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %320, double* %321, align 8, !tbaa !8
,double8B

	full_text

double %320
.double*8B

	full_text

double* %321
„getelementptr8Bq
o
	full_textb
`
^%322 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
_store8BT
R
	full_textE
C
Astore double 0x4039333333333334, double* %322, align 16, !tbaa !8
.double*8B

	full_text

double* %322
:fmul8B0
.
	full_text!

%323 = fmul double %272, %317
,double8B

	full_text

double %272
,double8B

	full_text

double %317
:fmul8B0
.
	full_text!

%324 = fmul double %262, %323
,double8B

	full_text

double %262
,double8B

	full_text

double %323
Cfsub8B9
7
	full_text*
(
&%325 = fsub double -0.000000e+00, %324
,double8B

	full_text

double %324
:fmul8B0
.
	full_text!

%326 = fmul double %276, %317
,double8B

	full_text

double %276
,double8B

	full_text

double %317
Hfmul8B>
<
	full_text/
-
+%327 = fmul double %326, 0x40BF020000000001
,double8B

	full_text

double %326
Cfsub8B9
7
	full_text*
(
&%328 = fsub double -0.000000e+00, %327
,double8B

	full_text

double %327
ucall8Bk
i
	full_text\
Z
X%329 = tail call double @llvm.fmuladd.f64(double %325, double 6.300000e+01, double %328)
,double8B

	full_text

double %325
,double8B

	full_text

double %328
„getelementptr8Bq
o
	full_textb
`
^%330 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %329, double* %330, align 8, !tbaa !8
,double8B

	full_text

double %329
.double*8B

	full_text

double* %330
„getelementptr8Bq
o
	full_textb
`
^%331 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %331, align 8, !tbaa !8
.double*8B

	full_text

double* %331
Bfmul8B8
6
	full_text)
'
%%332 = fmul double %318, 6.300000e+01
,double8B

	full_text

double %318
„getelementptr8Bq
o
	full_textb
`
^%333 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %332, double* %333, align 8, !tbaa !8
,double8B

	full_text

double %332
.double*8B

	full_text

double* %333
„getelementptr8Bq
o
	full_textb
`
^%334 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %287, double* %334, align 8, !tbaa !8
,double8B

	full_text

double %287
.double*8B

	full_text

double* %334
„getelementptr8Bq
o
	full_textb
`
^%335 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %335, align 8, !tbaa !8
.double*8B

	full_text

double* %335
¥getelementptr8B‘
Ž
	full_text€
~
|%336 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %56, i64 %259, i64 %60, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%337 = load double, double* %336, align 8, !tbaa !8
.double*8B

	full_text

double* %336
Bfmul8B8
6
	full_text)
'
%%338 = fmul double %337, 1.400000e+00
,double8B

	full_text

double %337
Cfsub8B9
7
	full_text*
(
&%339 = fsub double -0.000000e+00, %338
,double8B

	full_text

double %338
ucall8Bk
i
	full_text\
Z
X%340 = tail call double @llvm.fmuladd.f64(double %296, double 8.000000e-01, double %339)
,double8B

	full_text

double %296
,double8B

	full_text

double %339
:fmul8B0
.
	full_text!

%341 = fmul double %262, %272
,double8B

	full_text

double %262
,double8B

	full_text

double %272
:fmul8B0
.
	full_text!

%342 = fmul double %341, %340
,double8B

	full_text

double %341
,double8B

	full_text

double %340
Hfmul8B>
<
	full_text/
-
+%343 = fmul double %263, 0x3FB89374BC6A7EF8
,double8B

	full_text

double %263
:fmul8B0
.
	full_text!

%344 = fmul double %270, %270
,double8B

	full_text

double %270
,double8B

	full_text

double %270
Hfmul8B>
<
	full_text/
-
+%345 = fmul double %263, 0xBFB00AEC33E1F670
,double8B

	full_text

double %263
:fmul8B0
.
	full_text!

%346 = fmul double %272, %272
,double8B

	full_text

double %272
,double8B

	full_text

double %272
:fmul8B0
.
	full_text!

%347 = fmul double %345, %346
,double8B

	full_text

double %345
,double8B

	full_text

double %346
Cfsub8B9
7
	full_text*
(
&%348 = fsub double -0.000000e+00, %347
,double8B

	full_text

double %347
mcall8Bc
a
	full_textT
R
P%349 = tail call double @llvm.fmuladd.f64(double %343, double %344, double %348)
,double8B

	full_text

double %343
,double8B

	full_text

double %344
,double8B

	full_text

double %348
:fmul8B0
.
	full_text!

%350 = fmul double %317, %317
,double8B

	full_text

double %317
,double8B

	full_text

double %317
mcall8Bc
a
	full_textT
R
P%351 = tail call double @llvm.fmuladd.f64(double %343, double %350, double %349)
,double8B

	full_text

double %343
,double8B

	full_text

double %350
,double8B

	full_text

double %349
Hfmul8B>
<
	full_text/
-
+%352 = fmul double %262, 0x3FC916872B020C49
,double8B

	full_text

double %262
Cfsub8B9
7
	full_text*
(
&%353 = fsub double -0.000000e+00, %352
,double8B

	full_text

double %352
mcall8Bc
a
	full_textT
R
P%354 = tail call double @llvm.fmuladd.f64(double %353, double %337, double %351)
,double8B

	full_text

double %353
,double8B

	full_text

double %337
,double8B

	full_text

double %351
Hfmul8B>
<
	full_text/
-
+%355 = fmul double %354, 0x40BF020000000001
,double8B

	full_text

double %354
Cfsub8B9
7
	full_text*
(
&%356 = fsub double -0.000000e+00, %355
,double8B

	full_text

double %355
ucall8Bk
i
	full_text\
Z
X%357 = tail call double @llvm.fmuladd.f64(double %342, double 6.300000e+01, double %356)
,double8B

	full_text

double %342
,double8B

	full_text

double %356
„getelementptr8Bq
o
	full_textb
`
^%358 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %357, double* %358, align 16, !tbaa !8
,double8B

	full_text

double %357
.double*8B

	full_text

double* %358
Cfmul8B9
7
	full_text*
(
&%359 = fmul double %273, -4.000000e-01
,double8B

	full_text

double %273
:fmul8B0
.
	full_text!

%360 = fmul double %262, %359
,double8B

	full_text

double %262
,double8B

	full_text

double %359
Hfmul8B>
<
	full_text/
-
+%361 = fmul double %262, 0xC087D0624DD2F1A9
,double8B

	full_text

double %262
:fmul8B0
.
	full_text!

%362 = fmul double %361, %270
,double8B

	full_text

double %361
,double8B

	full_text

double %270
Cfsub8B9
7
	full_text*
(
&%363 = fsub double -0.000000e+00, %362
,double8B

	full_text

double %362
ucall8Bk
i
	full_text\
Z
X%364 = tail call double @llvm.fmuladd.f64(double %360, double 6.300000e+01, double %363)
,double8B

	full_text

double %360
,double8B

	full_text

double %363
„getelementptr8Bq
o
	full_textb
`
^%365 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %364, double* %365, align 8, !tbaa !8
,double8B

	full_text

double %364
.double*8B

	full_text

double* %365
:fmul8B0
.
	full_text!

%366 = fmul double %261, %337
,double8B

	full_text

double %261
,double8B

	full_text

double %337
:fmul8B0
.
	full_text!

%367 = fmul double %262, %346
,double8B

	full_text

double %262
,double8B

	full_text

double %346
mcall8Bc
a
	full_textT
R
P%368 = tail call double @llvm.fmuladd.f64(double %296, double %261, double %367)
,double8B

	full_text

double %296
,double8B

	full_text

double %261
,double8B

	full_text

double %367
Bfmul8B8
6
	full_text)
'
%%369 = fmul double %368, 4.000000e-01
,double8B

	full_text

double %368
Cfsub8B9
7
	full_text*
(
&%370 = fsub double -0.000000e+00, %369
,double8B

	full_text

double %369
ucall8Bk
i
	full_text\
Z
X%371 = tail call double @llvm.fmuladd.f64(double %366, double 1.400000e+00, double %370)
,double8B

	full_text

double %366
,double8B

	full_text

double %370
Hfmul8B>
<
	full_text/
-
+%372 = fmul double %262, 0xC07F172B020C49B9
,double8B

	full_text

double %262
:fmul8B0
.
	full_text!

%373 = fmul double %372, %272
,double8B

	full_text

double %372
,double8B

	full_text

double %272
Cfsub8B9
7
	full_text*
(
&%374 = fsub double -0.000000e+00, %373
,double8B

	full_text

double %373
ucall8Bk
i
	full_text\
Z
X%375 = tail call double @llvm.fmuladd.f64(double %371, double 6.300000e+01, double %374)
,double8B

	full_text

double %371
,double8B

	full_text

double %374
„getelementptr8Bq
o
	full_textb
`
^%376 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %375, double* %376, align 16, !tbaa !8
,double8B

	full_text

double %375
.double*8B

	full_text

double* %376
Cfmul8B9
7
	full_text*
(
&%377 = fmul double %323, -4.000000e-01
,double8B

	full_text

double %323
:fmul8B0
.
	full_text!

%378 = fmul double %262, %377
,double8B

	full_text

double %262
,double8B

	full_text

double %377
:fmul8B0
.
	full_text!

%379 = fmul double %361, %317
,double8B

	full_text

double %361
,double8B

	full_text

double %317
Cfsub8B9
7
	full_text*
(
&%380 = fsub double -0.000000e+00, %379
,double8B

	full_text

double %379
ucall8Bk
i
	full_text\
Z
X%381 = tail call double @llvm.fmuladd.f64(double %378, double 6.300000e+01, double %380)
,double8B

	full_text

double %378
,double8B

	full_text

double %380
„getelementptr8Bq
o
	full_textb
`
^%382 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Pstore8BE
C
	full_text6
4
2store double %381, double* %382, align 8, !tbaa !8
,double8B

	full_text

double %381
.double*8B

	full_text

double* %382
Bfmul8B8
6
	full_text)
'
%%383 = fmul double %282, 1.400000e+00
,double8B

	full_text

double %282
Hfmul8B>
<
	full_text/
-
+%384 = fmul double %261, 0x40984F645A1CAC08
,double8B

	full_text

double %261
Cfsub8B9
7
	full_text*
(
&%385 = fsub double -0.000000e+00, %384
,double8B

	full_text

double %384
ucall8Bk
i
	full_text\
Z
X%386 = tail call double @llvm.fmuladd.f64(double %383, double 6.300000e+01, double %385)
,double8B

	full_text

double %383
,double8B

	full_text

double %385
Hfadd8B>
<
	full_text/
-
+%387 = fadd double %386, 0xC0B7418000000001
,double8B

	full_text

double %386
„getelementptr8Bq
o
	full_textb
`
^%388 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %12, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %12
Qstore8BF
D
	full_text7
5
3store double %387, double* %388, align 16, !tbaa !8
,double8B

	full_text

double %387
.double*8B

	full_text

double* %388
:add8B1
/
	full_text"
 
%389 = add i64 %55, 4294967296
%i648B

	full_text
	
i64 %55
;ashr8B1
/
	full_text"
 
%390 = ashr exact i64 %389, 32
&i648B

	full_text


i64 %389
getelementptr8B|
z
	full_textm
k
i%391 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %54, i64 %390, i64 %58, i64 %60
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %54
&i648B

	full_text


i64 %390
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
3%392 = load double, double* %391, align 8, !tbaa !8
.double*8B

	full_text

double* %391
:fmul8B0
.
	full_text!

%393 = fmul double %392, %392
,double8B

	full_text

double %392
,double8B

	full_text

double %392
:fmul8B0
.
	full_text!

%394 = fmul double %392, %393
,double8B

	full_text

double %392
,double8B

	full_text

double %393
„getelementptr8Bq
o
	full_textb
`
^%395 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
_store8BT
R
	full_textE
C
Astore double 0xC0BF020000000001, double* %395, align 16, !tbaa !8
.double*8B

	full_text

double* %395
„getelementptr8Bq
o
	full_textb
`
^%396 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %396, align 8, !tbaa !8
.double*8B

	full_text

double* %396
„getelementptr8Bq
o
	full_textb
`
^%397 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %397, align 16, !tbaa !8
.double*8B

	full_text

double* %397
„getelementptr8Bq
o
	full_textb
`
^%398 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 6.300000e+01, double* %398, align 8, !tbaa !8
.double*8B

	full_text

double* %398
„getelementptr8Bq
o
	full_textb
`
^%399 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %399, align 16, !tbaa !8
.double*8B

	full_text

double* %399
¥getelementptr8B‘
Ž
	full_text€
~
|%400 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %390, i64 %58, i64 %60, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
&i648B

	full_text


i64 %390
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
3%401 = load double, double* %400, align 8, !tbaa !8
.double*8B

	full_text

double* %400
¥getelementptr8B‘
Ž
	full_text€
~
|%402 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %390, i64 %58, i64 %60, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
&i648B

	full_text


i64 %390
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
3%403 = load double, double* %402, align 8, !tbaa !8
.double*8B

	full_text

double* %402
:fmul8B0
.
	full_text!

%404 = fmul double %401, %403
,double8B

	full_text

double %401
,double8B

	full_text

double %403
:fmul8B0
.
	full_text!

%405 = fmul double %393, %404
,double8B

	full_text

double %393
,double8B

	full_text

double %404
Cfsub8B9
7
	full_text*
(
&%406 = fsub double -0.000000e+00, %405
,double8B

	full_text

double %405
Cfmul8B9
7
	full_text*
(
&%407 = fmul double %393, -1.000000e-01
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%408 = fmul double %407, %401
,double8B

	full_text

double %407
,double8B

	full_text

double %401
Hfmul8B>
<
	full_text/
-
+%409 = fmul double %408, 0x40BF020000000001
,double8B

	full_text

double %408
Cfsub8B9
7
	full_text*
(
&%410 = fsub double -0.000000e+00, %409
,double8B

	full_text

double %409
ucall8Bk
i
	full_text\
Z
X%411 = tail call double @llvm.fmuladd.f64(double %406, double 6.300000e+01, double %410)
,double8B

	full_text

double %406
,double8B

	full_text

double %410
„getelementptr8Bq
o
	full_textb
`
^%412 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %411, double* %412, align 8, !tbaa !8
,double8B

	full_text

double %411
.double*8B

	full_text

double* %412
:fmul8B0
.
	full_text!

%413 = fmul double %392, %403
,double8B

	full_text

double %392
,double8B

	full_text

double %403
Hfmul8B>
<
	full_text/
-
+%414 = fmul double %392, 0x4088CE6666666668
,double8B

	full_text

double %392
Cfsub8B9
7
	full_text*
(
&%415 = fsub double -0.000000e+00, %414
,double8B

	full_text

double %414
ucall8Bk
i
	full_text\
Z
X%416 = tail call double @llvm.fmuladd.f64(double %413, double 6.300000e+01, double %415)
,double8B

	full_text

double %413
,double8B

	full_text

double %415
Hfadd8B>
<
	full_text/
-
+%417 = fadd double %416, 0xC0BF020000000001
,double8B

	full_text

double %416
„getelementptr8Bq
o
	full_textb
`
^%418 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %417, double* %418, align 8, !tbaa !8
,double8B

	full_text

double %417
.double*8B

	full_text

double* %418
„getelementptr8Bq
o
	full_textb
`
^%419 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %419, align 8, !tbaa !8
.double*8B

	full_text

double* %419
:fmul8B0
.
	full_text!

%420 = fmul double %392, %401
,double8B

	full_text

double %392
,double8B

	full_text

double %401
Bfmul8B8
6
	full_text)
'
%%421 = fmul double %420, 6.300000e+01
,double8B

	full_text

double %420
„getelementptr8Bq
o
	full_textb
`
^%422 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %421, double* %422, align 8, !tbaa !8
,double8B

	full_text

double %421
.double*8B

	full_text

double* %422
„getelementptr8Bq
o
	full_textb
`
^%423 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %423, align 8, !tbaa !8
.double*8B

	full_text

double* %423
¥getelementptr8B‘
Ž
	full_text€
~
|%424 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %390, i64 %58, i64 %60, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
&i648B

	full_text


i64 %390
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
3%425 = load double, double* %424, align 8, !tbaa !8
.double*8B

	full_text

double* %424
:fmul8B0
.
	full_text!

%426 = fmul double %403, %425
,double8B

	full_text

double %403
,double8B

	full_text

double %425
:fmul8B0
.
	full_text!

%427 = fmul double %393, %426
,double8B

	full_text

double %393
,double8B

	full_text

double %426
Cfsub8B9
7
	full_text*
(
&%428 = fsub double -0.000000e+00, %427
,double8B

	full_text

double %427
:fmul8B0
.
	full_text!

%429 = fmul double %407, %425
,double8B

	full_text

double %407
,double8B

	full_text

double %425
Hfmul8B>
<
	full_text/
-
+%430 = fmul double %429, 0x40BF020000000001
,double8B

	full_text

double %429
Cfsub8B9
7
	full_text*
(
&%431 = fsub double -0.000000e+00, %430
,double8B

	full_text

double %430
ucall8Bk
i
	full_text\
Z
X%432 = tail call double @llvm.fmuladd.f64(double %428, double 6.300000e+01, double %431)
,double8B

	full_text

double %428
,double8B

	full_text

double %431
„getelementptr8Bq
o
	full_textb
`
^%433 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %432, double* %433, align 16, !tbaa !8
,double8B

	full_text

double %432
.double*8B

	full_text

double* %433
„getelementptr8Bq
o
	full_textb
`
^%434 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %434, align 8, !tbaa !8
.double*8B

	full_text

double* %434
Bfmul8B8
6
	full_text)
'
%%435 = fmul double %392, 1.000000e-01
,double8B

	full_text

double %392
Hfmul8B>
<
	full_text/
-
+%436 = fmul double %435, 0x40BF020000000001
,double8B

	full_text

double %435
Cfsub8B9
7
	full_text*
(
&%437 = fsub double -0.000000e+00, %436
,double8B

	full_text

double %436
ucall8Bk
i
	full_text\
Z
X%438 = tail call double @llvm.fmuladd.f64(double %413, double 6.300000e+01, double %437)
,double8B

	full_text

double %413
,double8B

	full_text

double %437
Hfadd8B>
<
	full_text/
-
+%439 = fadd double %438, 0xC0BF020000000001
,double8B

	full_text

double %438
„getelementptr8Bq
o
	full_textb
`
^%440 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %439, double* %440, align 16, !tbaa !8
,double8B

	full_text

double %439
.double*8B

	full_text

double* %440
:fmul8B0
.
	full_text!

%441 = fmul double %392, %425
,double8B

	full_text

double %392
,double8B

	full_text

double %425
Bfmul8B8
6
	full_text)
'
%%442 = fmul double %441, 6.300000e+01
,double8B

	full_text

double %441
„getelementptr8Bq
o
	full_textb
`
^%443 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %442, double* %443, align 8, !tbaa !8
,double8B

	full_text

double %442
.double*8B

	full_text

double* %443
„getelementptr8Bq
o
	full_textb
`
^%444 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %444, align 16, !tbaa !8
.double*8B

	full_text

double* %444
Cfsub8B9
7
	full_text*
(
&%445 = fsub double -0.000000e+00, %413
,double8B

	full_text

double %413
getelementptr8B|
z
	full_textm
k
i%446 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %53, i64 %390, i64 %58, i64 %60
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %53
&i648B

	full_text


i64 %390
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
3%447 = load double, double* %446, align 8, !tbaa !8
.double*8B

	full_text

double* %446
:fmul8B0
.
	full_text!

%448 = fmul double %392, %447
,double8B

	full_text

double %392
,double8B

	full_text

double %447
Bfmul8B8
6
	full_text)
'
%%449 = fmul double %448, 4.000000e-01
,double8B

	full_text

double %448
mcall8Bc
a
	full_textT
R
P%450 = tail call double @llvm.fmuladd.f64(double %445, double %413, double %449)
,double8B

	full_text

double %445
,double8B

	full_text

double %413
,double8B

	full_text

double %449
Hfmul8B>
<
	full_text/
-
+%451 = fmul double %393, 0xBFC1111111111111
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%452 = fmul double %451, %403
,double8B

	full_text

double %451
,double8B

	full_text

double %403
Hfmul8B>
<
	full_text/
-
+%453 = fmul double %452, 0x40BF020000000001
,double8B

	full_text

double %452
Cfsub8B9
7
	full_text*
(
&%454 = fsub double -0.000000e+00, %453
,double8B

	full_text

double %453
ucall8Bk
i
	full_text\
Z
X%455 = tail call double @llvm.fmuladd.f64(double %450, double 6.300000e+01, double %454)
,double8B

	full_text

double %450
,double8B

	full_text

double %454
„getelementptr8Bq
o
	full_textb
`
^%456 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %455, double* %456, align 8, !tbaa !8
,double8B

	full_text

double %455
.double*8B

	full_text

double* %456
Cfmul8B9
7
	full_text*
(
&%457 = fmul double %420, -4.000000e-01
,double8B

	full_text

double %420
Bfmul8B8
6
	full_text)
'
%%458 = fmul double %457, 6.300000e+01
,double8B

	full_text

double %457
„getelementptr8Bq
o
	full_textb
`
^%459 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %458, double* %459, align 8, !tbaa !8
,double8B

	full_text

double %458
.double*8B

	full_text

double* %459
Cfmul8B9
7
	full_text*
(
&%460 = fmul double %441, -4.000000e-01
,double8B

	full_text

double %441
Bfmul8B8
6
	full_text)
'
%%461 = fmul double %460, 6.300000e+01
,double8B

	full_text

double %460
„getelementptr8Bq
o
	full_textb
`
^%462 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %461, double* %462, align 8, !tbaa !8
,double8B

	full_text

double %461
.double*8B

	full_text

double* %462
Hfmul8B>
<
	full_text/
-
+%463 = fmul double %392, 0x3FC1111111111111
,double8B

	full_text

double %392
Hfmul8B>
<
	full_text/
-
+%464 = fmul double %463, 0x40BF020000000001
,double8B

	full_text

double %463
Cfsub8B9
7
	full_text*
(
&%465 = fsub double -0.000000e+00, %464
,double8B

	full_text

double %464
{call8Bq
o
	full_textb
`
^%466 = tail call double @llvm.fmuladd.f64(double %413, double 0x4059333333333334, double %465)
,double8B

	full_text

double %413
,double8B

	full_text

double %465
Hfadd8B>
<
	full_text/
-
+%467 = fadd double %466, 0xC0BF020000000001
,double8B

	full_text

double %466
„getelementptr8Bq
o
	full_textb
`
^%468 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %467, double* %468, align 8, !tbaa !8
,double8B

	full_text

double %467
.double*8B

	full_text

double* %468
„getelementptr8Bq
o
	full_textb
`
^%469 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
^store8BS
Q
	full_textD
B
@store double 0x4039333333333334, double* %469, align 8, !tbaa !8
.double*8B

	full_text

double* %469
¥getelementptr8B‘
Ž
	full_text€
~
|%470 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %52, i64 %390, i64 %58, i64 %60, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %52
&i648B

	full_text


i64 %390
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
3%471 = load double, double* %470, align 8, !tbaa !8
.double*8B

	full_text

double* %470
Bfmul8B8
6
	full_text)
'
%%472 = fmul double %471, 1.400000e+00
,double8B

	full_text

double %471
Cfsub8B9
7
	full_text*
(
&%473 = fsub double -0.000000e+00, %472
,double8B

	full_text

double %472
ucall8Bk
i
	full_text\
Z
X%474 = tail call double @llvm.fmuladd.f64(double %447, double 8.000000e-01, double %473)
,double8B

	full_text

double %447
,double8B

	full_text

double %473
:fmul8B0
.
	full_text!

%475 = fmul double %393, %403
,double8B

	full_text

double %393
,double8B

	full_text

double %403
:fmul8B0
.
	full_text!

%476 = fmul double %475, %474
,double8B

	full_text

double %475
,double8B

	full_text

double %474
Hfmul8B>
<
	full_text/
-
+%477 = fmul double %394, 0x3FB89374BC6A7EF8
,double8B

	full_text

double %394
:fmul8B0
.
	full_text!

%478 = fmul double %401, %401
,double8B

	full_text

double %401
,double8B

	full_text

double %401
Hfmul8B>
<
	full_text/
-
+%479 = fmul double %394, 0xBFB89374BC6A7EF8
,double8B

	full_text

double %394
:fmul8B0
.
	full_text!

%480 = fmul double %425, %425
,double8B

	full_text

double %425
,double8B

	full_text

double %425
:fmul8B0
.
	full_text!

%481 = fmul double %479, %480
,double8B

	full_text

double %479
,double8B

	full_text

double %480
Cfsub8B9
7
	full_text*
(
&%482 = fsub double -0.000000e+00, %481
,double8B

	full_text

double %481
mcall8Bc
a
	full_textT
R
P%483 = tail call double @llvm.fmuladd.f64(double %477, double %478, double %482)
,double8B

	full_text

double %477
,double8B

	full_text

double %478
,double8B

	full_text

double %482
Hfmul8B>
<
	full_text/
-
+%484 = fmul double %394, 0x3FB00AEC33E1F670
,double8B

	full_text

double %394
:fmul8B0
.
	full_text!

%485 = fmul double %403, %403
,double8B

	full_text

double %403
,double8B

	full_text

double %403
mcall8Bc
a
	full_textT
R
P%486 = tail call double @llvm.fmuladd.f64(double %484, double %485, double %483)
,double8B

	full_text

double %484
,double8B

	full_text

double %485
,double8B

	full_text

double %483
Hfmul8B>
<
	full_text/
-
+%487 = fmul double %393, 0x3FC916872B020C49
,double8B

	full_text

double %393
Cfsub8B9
7
	full_text*
(
&%488 = fsub double -0.000000e+00, %487
,double8B

	full_text

double %487
mcall8Bc
a
	full_textT
R
P%489 = tail call double @llvm.fmuladd.f64(double %488, double %471, double %486)
,double8B

	full_text

double %488
,double8B

	full_text

double %471
,double8B

	full_text

double %486
Hfmul8B>
<
	full_text/
-
+%490 = fmul double %489, 0x40BF020000000001
,double8B

	full_text

double %489
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
X%492 = tail call double @llvm.fmuladd.f64(double %476, double 6.300000e+01, double %491)
,double8B

	full_text

double %476
,double8B

	full_text

double %491
„getelementptr8Bq
o
	full_textb
`
^%493 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %492, double* %493, align 16, !tbaa !8
,double8B

	full_text

double %492
.double*8B

	full_text

double* %493
Cfmul8B9
7
	full_text*
(
&%494 = fmul double %404, -4.000000e-01
,double8B

	full_text

double %404
:fmul8B0
.
	full_text!

%495 = fmul double %393, %494
,double8B

	full_text

double %393
,double8B

	full_text

double %494
Hfmul8B>
<
	full_text/
-
+%496 = fmul double %393, 0xC087D0624DD2F1A9
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%497 = fmul double %496, %401
,double8B

	full_text

double %496
,double8B

	full_text

double %401
Cfsub8B9
7
	full_text*
(
&%498 = fsub double -0.000000e+00, %497
,double8B

	full_text

double %497
ucall8Bk
i
	full_text\
Z
X%499 = tail call double @llvm.fmuladd.f64(double %495, double 6.300000e+01, double %498)
,double8B

	full_text

double %495
,double8B

	full_text

double %498
„getelementptr8Bq
o
	full_textb
`
^%500 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %499, double* %500, align 8, !tbaa !8
,double8B

	full_text

double %499
.double*8B

	full_text

double* %500
Cfmul8B9
7
	full_text*
(
&%501 = fmul double %426, -4.000000e-01
,double8B

	full_text

double %426
:fmul8B0
.
	full_text!

%502 = fmul double %393, %501
,double8B

	full_text

double %393
,double8B

	full_text

double %501
:fmul8B0
.
	full_text!

%503 = fmul double %496, %425
,double8B

	full_text

double %496
,double8B

	full_text

double %425
Cfsub8B9
7
	full_text*
(
&%504 = fsub double -0.000000e+00, %503
,double8B

	full_text

double %503
ucall8Bk
i
	full_text\
Z
X%505 = tail call double @llvm.fmuladd.f64(double %502, double 6.300000e+01, double %504)
,double8B

	full_text

double %502
,double8B

	full_text

double %504
„getelementptr8Bq
o
	full_textb
`
^%506 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %505, double* %506, align 16, !tbaa !8
,double8B

	full_text

double %505
.double*8B

	full_text

double* %506
:fmul8B0
.
	full_text!

%507 = fmul double %392, %471
,double8B

	full_text

double %392
,double8B

	full_text

double %471
:fmul8B0
.
	full_text!

%508 = fmul double %393, %485
,double8B

	full_text

double %393
,double8B

	full_text

double %485
mcall8Bc
a
	full_textT
R
P%509 = tail call double @llvm.fmuladd.f64(double %447, double %392, double %508)
,double8B

	full_text

double %447
,double8B

	full_text

double %392
,double8B

	full_text

double %508
Bfmul8B8
6
	full_text)
'
%%510 = fmul double %509, 4.000000e-01
,double8B

	full_text

double %509
Cfsub8B9
7
	full_text*
(
&%511 = fsub double -0.000000e+00, %510
,double8B

	full_text

double %510
ucall8Bk
i
	full_text\
Z
X%512 = tail call double @llvm.fmuladd.f64(double %507, double 1.400000e+00, double %511)
,double8B

	full_text

double %507
,double8B

	full_text

double %511
Hfmul8B>
<
	full_text/
-
+%513 = fmul double %393, 0xC07F172B020C49B9
,double8B

	full_text

double %393
:fmul8B0
.
	full_text!

%514 = fmul double %513, %403
,double8B

	full_text

double %513
,double8B

	full_text

double %403
Cfsub8B9
7
	full_text*
(
&%515 = fsub double -0.000000e+00, %514
,double8B

	full_text

double %514
ucall8Bk
i
	full_text\
Z
X%516 = tail call double @llvm.fmuladd.f64(double %512, double 6.300000e+01, double %515)
,double8B

	full_text

double %512
,double8B

	full_text

double %515
„getelementptr8Bq
o
	full_textb
`
^%517 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Pstore8BE
C
	full_text6
4
2store double %516, double* %517, align 8, !tbaa !8
,double8B

	full_text

double %516
.double*8B

	full_text

double* %517
Bfmul8B8
6
	full_text)
'
%%518 = fmul double %413, 1.400000e+00
,double8B

	full_text

double %413
Hfmul8B>
<
	full_text/
-
+%519 = fmul double %392, 0x40984F645A1CAC08
,double8B

	full_text

double %392
Cfsub8B9
7
	full_text*
(
&%520 = fsub double -0.000000e+00, %519
,double8B

	full_text

double %519
ucall8Bk
i
	full_text\
Z
X%521 = tail call double @llvm.fmuladd.f64(double %518, double 6.300000e+01, double %520)
,double8B

	full_text

double %518
,double8B

	full_text

double %520
Hfadd8B>
<
	full_text/
-
+%522 = fadd double %521, 0xC0BF020000000001
,double8B

	full_text

double %521
„getelementptr8Bq
o
	full_textb
`
^%523 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %13, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %13
Qstore8BF
D
	full_text7
5
3store double %522, double* %523, align 16, !tbaa !8
,double8B

	full_text

double %522
.double*8B

	full_text

double* %523
¥getelementptr8B‘
Ž
	full_text€
~
|%524 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %390, i64 %58, i64 %60, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
&i648B

	full_text


i64 %390
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
3%525 = load double, double* %524, align 8, !tbaa !8
.double*8B

	full_text

double* %524
¥getelementptr8B‘
Ž
	full_text€
~
|%526 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %390, i64 %58, i64 %60, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
&i648B

	full_text


i64 %390
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
3%527 = load double, double* %526, align 8, !tbaa !8
.double*8B

	full_text

double* %526
¥getelementptr8B‘
Ž
	full_text€
~
|%528 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %390, i64 %58, i64 %60, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
&i648B

	full_text


i64 %390
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
3%529 = load double, double* %528, align 8, !tbaa !8
.double*8B

	full_text

double* %528
¥getelementptr8B‘
Ž
	full_text€
~
|%530 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %390, i64 %58, i64 %60, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
&i648B

	full_text


i64 %390
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
3%531 = load double, double* %530, align 8, !tbaa !8
.double*8B

	full_text

double* %530
¥getelementptr8B‘
Ž
	full_text€
~
|%532 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %390, i64 %58, i64 %60, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
&i648B

	full_text


i64 %390
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
3%533 = load double, double* %532, align 8, !tbaa !8
.double*8B

	full_text

double* %532
Qload8BG
E
	full_text8
6
4%534 = load double, double* %395, align 16, !tbaa !8
.double*8B

	full_text

double* %395
Pload8BF
D
	full_text7
5
3%535 = load double, double* %396, align 8, !tbaa !8
.double*8B

	full_text

double* %396
:fmul8B0
.
	full_text!

%536 = fmul double %535, %527
,double8B

	full_text

double %535
,double8B

	full_text

double %527
mcall8Bc
a
	full_textT
R
P%537 = tail call double @llvm.fmuladd.f64(double %534, double %525, double %536)
,double8B

	full_text

double %534
,double8B

	full_text

double %525
,double8B

	full_text

double %536
Qload8BG
E
	full_text8
6
4%538 = load double, double* %397, align 16, !tbaa !8
.double*8B

	full_text

double* %397
mcall8Bc
a
	full_textT
R
P%539 = tail call double @llvm.fmuladd.f64(double %538, double %529, double %537)
,double8B

	full_text

double %538
,double8B

	full_text

double %529
,double8B

	full_text

double %537
Pload8BF
D
	full_text7
5
3%540 = load double, double* %398, align 8, !tbaa !8
.double*8B

	full_text

double* %398
mcall8Bc
a
	full_textT
R
P%541 = tail call double @llvm.fmuladd.f64(double %540, double %531, double %539)
,double8B

	full_text

double %540
,double8B

	full_text

double %531
,double8B

	full_text

double %539
Qload8BG
E
	full_text8
6
4%542 = load double, double* %399, align 16, !tbaa !8
.double*8B

	full_text

double* %399
mcall8Bc
a
	full_textT
R
P%543 = tail call double @llvm.fmuladd.f64(double %542, double %533, double %541)
,double8B

	full_text

double %542
,double8B

	full_text

double %533
,double8B

	full_text

double %541
Bfmul8B8
6
	full_text)
'
%%544 = fmul double %543, 1.200000e+00
,double8B

	full_text

double %543
qgetelementptr8B^
\
	full_textO
M
K%545 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Pload8BF
D
	full_text7
5
3%546 = load double, double* %412, align 8, !tbaa !8
.double*8B

	full_text

double* %412
Pload8BF
D
	full_text7
5
3%547 = load double, double* %418, align 8, !tbaa !8
.double*8B

	full_text

double* %418
:fmul8B0
.
	full_text!

%548 = fmul double %547, %527
,double8B

	full_text

double %547
,double8B

	full_text

double %527
mcall8Bc
a
	full_textT
R
P%549 = tail call double @llvm.fmuladd.f64(double %546, double %525, double %548)
,double8B

	full_text

double %546
,double8B

	full_text

double %525
,double8B

	full_text

double %548
Pload8BF
D
	full_text7
5
3%550 = load double, double* %419, align 8, !tbaa !8
.double*8B

	full_text

double* %419
mcall8Bc
a
	full_textT
R
P%551 = tail call double @llvm.fmuladd.f64(double %550, double %529, double %549)
,double8B

	full_text

double %550
,double8B

	full_text

double %529
,double8B

	full_text

double %549
Pload8BF
D
	full_text7
5
3%552 = load double, double* %422, align 8, !tbaa !8
.double*8B

	full_text

double* %422
mcall8Bc
a
	full_textT
R
P%553 = tail call double @llvm.fmuladd.f64(double %552, double %531, double %551)
,double8B

	full_text

double %552
,double8B

	full_text

double %531
,double8B

	full_text

double %551
Pload8BF
D
	full_text7
5
3%554 = load double, double* %423, align 8, !tbaa !8
.double*8B

	full_text

double* %423
mcall8Bc
a
	full_textT
R
P%555 = tail call double @llvm.fmuladd.f64(double %554, double %533, double %553)
,double8B

	full_text

double %554
,double8B

	full_text

double %533
,double8B

	full_text

double %553
Bfmul8B8
6
	full_text)
'
%%556 = fmul double %555, 1.200000e+00
,double8B

	full_text

double %555
qgetelementptr8B^
\
	full_textO
M
K%557 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qload8BG
E
	full_text8
6
4%558 = load double, double* %433, align 16, !tbaa !8
.double*8B

	full_text

double* %433
Pload8BF
D
	full_text7
5
3%559 = load double, double* %434, align 8, !tbaa !8
.double*8B

	full_text

double* %434
:fmul8B0
.
	full_text!

%560 = fmul double %559, %527
,double8B

	full_text

double %559
,double8B

	full_text

double %527
mcall8Bc
a
	full_textT
R
P%561 = tail call double @llvm.fmuladd.f64(double %558, double %525, double %560)
,double8B

	full_text

double %558
,double8B

	full_text

double %525
,double8B

	full_text

double %560
Qload8BG
E
	full_text8
6
4%562 = load double, double* %440, align 16, !tbaa !8
.double*8B

	full_text

double* %440
mcall8Bc
a
	full_textT
R
P%563 = tail call double @llvm.fmuladd.f64(double %562, double %529, double %561)
,double8B

	full_text

double %562
,double8B

	full_text

double %529
,double8B

	full_text

double %561
Pload8BF
D
	full_text7
5
3%564 = load double, double* %443, align 8, !tbaa !8
.double*8B

	full_text

double* %443
mcall8Bc
a
	full_textT
R
P%565 = tail call double @llvm.fmuladd.f64(double %564, double %531, double %563)
,double8B

	full_text

double %564
,double8B

	full_text

double %531
,double8B

	full_text

double %563
Qload8BG
E
	full_text8
6
4%566 = load double, double* %444, align 16, !tbaa !8
.double*8B

	full_text

double* %444
mcall8Bc
a
	full_textT
R
P%567 = tail call double @llvm.fmuladd.f64(double %566, double %533, double %565)
,double8B

	full_text

double %566
,double8B

	full_text

double %533
,double8B

	full_text

double %565
Bfmul8B8
6
	full_text)
'
%%568 = fmul double %567, 1.200000e+00
,double8B

	full_text

double %567
qgetelementptr8B^
\
	full_textO
M
K%569 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %568, double* %569, align 16, !tbaa !8
,double8B

	full_text

double %568
.double*8B

	full_text

double* %569
Pload8BF
D
	full_text7
5
3%570 = load double, double* %456, align 8, !tbaa !8
.double*8B

	full_text

double* %456
Pload8BF
D
	full_text7
5
3%571 = load double, double* %459, align 8, !tbaa !8
.double*8B

	full_text

double* %459
:fmul8B0
.
	full_text!

%572 = fmul double %571, %527
,double8B

	full_text

double %571
,double8B

	full_text

double %527
mcall8Bc
a
	full_textT
R
P%573 = tail call double @llvm.fmuladd.f64(double %570, double %525, double %572)
,double8B

	full_text

double %570
,double8B

	full_text

double %525
,double8B

	full_text

double %572
Pload8BF
D
	full_text7
5
3%574 = load double, double* %462, align 8, !tbaa !8
.double*8B

	full_text

double* %462
mcall8Bc
a
	full_textT
R
P%575 = tail call double @llvm.fmuladd.f64(double %574, double %529, double %573)
,double8B

	full_text

double %574
,double8B

	full_text

double %529
,double8B

	full_text

double %573
Pload8BF
D
	full_text7
5
3%576 = load double, double* %468, align 8, !tbaa !8
.double*8B

	full_text

double* %468
mcall8Bc
a
	full_textT
R
P%577 = tail call double @llvm.fmuladd.f64(double %576, double %531, double %575)
,double8B

	full_text

double %576
,double8B

	full_text

double %531
,double8B

	full_text

double %575
Pload8BF
D
	full_text7
5
3%578 = load double, double* %469, align 8, !tbaa !8
.double*8B

	full_text

double* %469
mcall8Bc
a
	full_textT
R
P%579 = tail call double @llvm.fmuladd.f64(double %578, double %533, double %577)
,double8B

	full_text

double %578
,double8B

	full_text

double %533
,double8B

	full_text

double %577
Bfmul8B8
6
	full_text)
'
%%580 = fmul double %579, 1.200000e+00
,double8B

	full_text

double %579
qgetelementptr8B^
\
	full_textO
M
K%581 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Pstore8BE
C
	full_text6
4
2store double %580, double* %581, align 8, !tbaa !8
,double8B

	full_text

double %580
.double*8B

	full_text

double* %581
:fmul8B0
.
	full_text!

%582 = fmul double %499, %527
,double8B

	full_text

double %499
,double8B

	full_text

double %527
mcall8Bc
a
	full_textT
R
P%583 = tail call double @llvm.fmuladd.f64(double %492, double %525, double %582)
,double8B

	full_text

double %492
,double8B

	full_text

double %525
,double8B

	full_text

double %582
mcall8Bc
a
	full_textT
R
P%584 = tail call double @llvm.fmuladd.f64(double %505, double %529, double %583)
,double8B

	full_text

double %505
,double8B

	full_text

double %529
,double8B

	full_text

double %583
mcall8Bc
a
	full_textT
R
P%585 = tail call double @llvm.fmuladd.f64(double %516, double %531, double %584)
,double8B

	full_text

double %516
,double8B

	full_text

double %531
,double8B

	full_text

double %584
mcall8Bc
a
	full_textT
R
P%586 = tail call double @llvm.fmuladd.f64(double %522, double %533, double %585)
,double8B

	full_text

double %522
,double8B

	full_text

double %533
,double8B

	full_text

double %585
Bfmul8B8
6
	full_text)
'
%%587 = fmul double %586, 1.200000e+00
,double8B

	full_text

double %586
qgetelementptr8B^
\
	full_textO
M
K%588 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %587, double* %588, align 16, !tbaa !8
,double8B

	full_text

double %587
.double*8B

	full_text

double* %588
¥getelementptr8B‘
Ž
	full_text€
~
|%589 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %259, i64 %60, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%590 = load double, double* %589, align 8, !tbaa !8
.double*8B

	full_text

double* %589
¥getelementptr8B‘
Ž
	full_text€
~
|%591 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %127, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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


i64 %127
Pload8BF
D
	full_text7
5
3%592 = load double, double* %591, align 8, !tbaa !8
.double*8B

	full_text

double* %591
¥getelementptr8B‘
Ž
	full_text€
~
|%593 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %259, i64 %60, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%594 = load double, double* %593, align 8, !tbaa !8
.double*8B

	full_text

double* %593
¥getelementptr8B‘
Ž
	full_text€
~
|%595 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %127, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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


i64 %127
Pload8BF
D
	full_text7
5
3%596 = load double, double* %595, align 8, !tbaa !8
.double*8B

	full_text

double* %595
¥getelementptr8B‘
Ž
	full_text€
~
|%597 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %259, i64 %60, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%598 = load double, double* %597, align 8, !tbaa !8
.double*8B

	full_text

double* %597
¥getelementptr8B‘
Ž
	full_text€
~
|%599 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %127, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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


i64 %127
Pload8BF
D
	full_text7
5
3%600 = load double, double* %599, align 8, !tbaa !8
.double*8B

	full_text

double* %599
¥getelementptr8B‘
Ž
	full_text€
~
|%601 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %259, i64 %60, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%602 = load double, double* %601, align 8, !tbaa !8
.double*8B

	full_text

double* %601
¥getelementptr8B‘
Ž
	full_text€
~
|%603 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %127, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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


i64 %127
Pload8BF
D
	full_text7
5
3%604 = load double, double* %603, align 8, !tbaa !8
.double*8B

	full_text

double* %603
¥getelementptr8B‘
Ž
	full_text€
~
|%605 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %259, i64 %60, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
%i648B

	full_text
	
i64 %56
&i648B

	full_text


i64 %259
%i648B

	full_text
	
i64 %60
Pload8BF
D
	full_text7
5
3%606 = load double, double* %605, align 8, !tbaa !8
.double*8B

	full_text

double* %605
¥getelementptr8B‘
Ž
	full_text€
~
|%607 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %127, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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


i64 %127
Pload8BF
D
	full_text7
5
3%608 = load double, double* %607, align 8, !tbaa !8
.double*8B

	full_text

double* %607
Qload8BG
E
	full_text8
6
4%609 = load double, double* %264, align 16, !tbaa !8
.double*8B

	full_text

double* %264
Qload8BG
E
	full_text8
6
4%610 = load double, double* %132, align 16, !tbaa !8
.double*8B

	full_text

double* %132
:fmul8B0
.
	full_text!

%611 = fmul double %610, %592
,double8B

	full_text

double %610
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%612 = tail call double @llvm.fmuladd.f64(double %609, double %590, double %611)
,double8B

	full_text

double %609
,double8B

	full_text

double %590
,double8B

	full_text

double %611
Pload8BF
D
	full_text7
5
3%613 = load double, double* %265, align 8, !tbaa !8
.double*8B

	full_text

double* %265
mcall8Bc
a
	full_textT
R
P%614 = tail call double @llvm.fmuladd.f64(double %613, double %594, double %612)
,double8B

	full_text

double %613
,double8B

	full_text

double %594
,double8B

	full_text

double %612
Pload8BF
D
	full_text7
5
3%615 = load double, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
mcall8Bc
a
	full_textT
R
P%616 = tail call double @llvm.fmuladd.f64(double %615, double %596, double %614)
,double8B

	full_text

double %615
,double8B

	full_text

double %596
,double8B

	full_text

double %614
Qload8BG
E
	full_text8
6
4%617 = load double, double* %266, align 16, !tbaa !8
.double*8B

	full_text

double* %266
mcall8Bc
a
	full_textT
R
P%618 = tail call double @llvm.fmuladd.f64(double %617, double %598, double %616)
,double8B

	full_text

double %617
,double8B

	full_text

double %598
,double8B

	full_text

double %616
Qload8BG
E
	full_text8
6
4%619 = load double, double* %134, align 16, !tbaa !8
.double*8B

	full_text

double* %134
mcall8Bc
a
	full_textT
R
P%620 = tail call double @llvm.fmuladd.f64(double %619, double %600, double %618)
,double8B

	full_text

double %619
,double8B

	full_text

double %600
,double8B

	full_text

double %618
Pload8BF
D
	full_text7
5
3%621 = load double, double* %267, align 8, !tbaa !8
.double*8B

	full_text

double* %267
mcall8Bc
a
	full_textT
R
P%622 = tail call double @llvm.fmuladd.f64(double %621, double %602, double %620)
,double8B

	full_text

double %621
,double8B

	full_text

double %602
,double8B

	full_text

double %620
Pload8BF
D
	full_text7
5
3%623 = load double, double* %135, align 8, !tbaa !8
.double*8B

	full_text

double* %135
mcall8Bc
a
	full_textT
R
P%624 = tail call double @llvm.fmuladd.f64(double %623, double %604, double %622)
,double8B

	full_text

double %623
,double8B

	full_text

double %604
,double8B

	full_text

double %622
Qload8BG
E
	full_text8
6
4%625 = load double, double* %268, align 16, !tbaa !8
.double*8B

	full_text

double* %268
mcall8Bc
a
	full_textT
R
P%626 = tail call double @llvm.fmuladd.f64(double %625, double %606, double %624)
,double8B

	full_text

double %625
,double8B

	full_text

double %606
,double8B

	full_text

double %624
Qload8BG
E
	full_text8
6
4%627 = load double, double* %136, align 16, !tbaa !8
.double*8B

	full_text

double* %136
mcall8Bc
a
	full_textT
R
P%628 = tail call double @llvm.fmuladd.f64(double %627, double %608, double %626)
,double8B

	full_text

double %627
,double8B

	full_text

double %608
,double8B

	full_text

double %626
ucall8Bk
i
	full_text\
Z
X%629 = tail call double @llvm.fmuladd.f64(double %628, double 1.200000e+00, double %544)
,double8B

	full_text

double %628
,double8B

	full_text

double %544
Qstore8BF
D
	full_text7
5
3store double %629, double* %545, align 16, !tbaa !8
,double8B

	full_text

double %629
.double*8B

	full_text

double* %545
Pload8BF
D
	full_text7
5
3%630 = load double, double* %281, align 8, !tbaa !8
.double*8B

	full_text

double* %281
Pload8BF
D
	full_text7
5
3%631 = load double, double* %151, align 8, !tbaa !8
.double*8B

	full_text

double* %151
:fmul8B0
.
	full_text!

%632 = fmul double %631, %592
,double8B

	full_text

double %631
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%633 = tail call double @llvm.fmuladd.f64(double %630, double %590, double %632)
,double8B

	full_text

double %630
,double8B

	full_text

double %590
,double8B

	full_text

double %632
Pload8BF
D
	full_text7
5
3%634 = load double, double* %288, align 8, !tbaa !8
.double*8B

	full_text

double* %288
mcall8Bc
a
	full_textT
R
P%635 = tail call double @llvm.fmuladd.f64(double %634, double %594, double %633)
,double8B

	full_text

double %634
,double8B

	full_text

double %594
,double8B

	full_text

double %633
Pload8BF
D
	full_text7
5
3%636 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
mcall8Bc
a
	full_textT
R
P%637 = tail call double @llvm.fmuladd.f64(double %636, double %596, double %635)
,double8B

	full_text

double %636
,double8B

	full_text

double %596
,double8B

	full_text

double %635
Pload8BF
D
	full_text7
5
3%638 = load double, double* %291, align 8, !tbaa !8
.double*8B

	full_text

double* %291
mcall8Bc
a
	full_textT
R
P%639 = tail call double @llvm.fmuladd.f64(double %638, double %598, double %637)
,double8B

	full_text

double %638
,double8B

	full_text

double %598
,double8B

	full_text

double %637
Pload8BF
D
	full_text7
5
3%640 = load double, double* %164, align 8, !tbaa !8
.double*8B

	full_text

double* %164
mcall8Bc
a
	full_textT
R
P%641 = tail call double @llvm.fmuladd.f64(double %640, double %600, double %639)
,double8B

	full_text

double %640
,double8B

	full_text

double %600
,double8B

	full_text

double %639
Pload8BF
D
	full_text7
5
3%642 = load double, double* %292, align 8, !tbaa !8
.double*8B

	full_text

double* %292
mcall8Bc
a
	full_textT
R
P%643 = tail call double @llvm.fmuladd.f64(double %642, double %602, double %641)
,double8B

	full_text

double %642
,double8B

	full_text

double %602
,double8B

	full_text

double %641
Pload8BF
D
	full_text7
5
3%644 = load double, double* %170, align 8, !tbaa !8
.double*8B

	full_text

double* %170
mcall8Bc
a
	full_textT
R
P%645 = tail call double @llvm.fmuladd.f64(double %644, double %604, double %643)
,double8B

	full_text

double %644
,double8B

	full_text

double %604
,double8B

	full_text

double %643
Pload8BF
D
	full_text7
5
3%646 = load double, double* %293, align 8, !tbaa !8
.double*8B

	full_text

double* %293
mcall8Bc
a
	full_textT
R
P%647 = tail call double @llvm.fmuladd.f64(double %646, double %606, double %645)
,double8B

	full_text

double %646
,double8B

	full_text

double %606
,double8B

	full_text

double %645
Pload8BF
D
	full_text7
5
3%648 = load double, double* %171, align 8, !tbaa !8
.double*8B

	full_text

double* %171
mcall8Bc
a
	full_textT
R
P%649 = tail call double @llvm.fmuladd.f64(double %648, double %608, double %647)
,double8B

	full_text

double %648
,double8B

	full_text

double %608
,double8B

	full_text

double %647
ucall8Bk
i
	full_text\
Z
X%650 = tail call double @llvm.fmuladd.f64(double %649, double 1.200000e+00, double %556)
,double8B

	full_text

double %649
,double8B

	full_text

double %556
Pstore8BE
C
	full_text6
4
2store double %650, double* %557, align 8, !tbaa !8
,double8B

	full_text

double %650
.double*8B

	full_text

double* %557
Qload8BG
E
	full_text8
6
4%651 = load double, double* %305, align 16, !tbaa !8
.double*8B

	full_text

double* %305
Qload8BG
E
	full_text8
6
4%652 = load double, double* %180, align 16, !tbaa !8
.double*8B

	full_text

double* %180
:fmul8B0
.
	full_text!

%653 = fmul double %652, %592
,double8B

	full_text

double %652
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%654 = tail call double @llvm.fmuladd.f64(double %651, double %590, double %653)
,double8B

	full_text

double %651
,double8B

	full_text

double %590
,double8B

	full_text

double %653
Pload8BF
D
	full_text7
5
3%655 = load double, double* %308, align 8, !tbaa !8
.double*8B

	full_text

double* %308
mcall8Bc
a
	full_textT
R
P%656 = tail call double @llvm.fmuladd.f64(double %655, double %594, double %654)
,double8B

	full_text

double %655
,double8B

	full_text

double %594
,double8B

	full_text

double %654
Pload8BF
D
	full_text7
5
3%657 = load double, double* %182, align 8, !tbaa !8
.double*8B

	full_text

double* %182
mcall8Bc
a
	full_textT
R
P%658 = tail call double @llvm.fmuladd.f64(double %657, double %596, double %656)
,double8B

	full_text

double %657
,double8B

	full_text

double %596
,double8B

	full_text

double %656
Qload8BG
E
	full_text8
6
4%659 = load double, double* %315, align 16, !tbaa !8
.double*8B

	full_text

double* %315
mcall8Bc
a
	full_textT
R
P%660 = tail call double @llvm.fmuladd.f64(double %659, double %598, double %658)
,double8B

	full_text

double %659
,double8B

	full_text

double %598
,double8B

	full_text

double %658
Qload8BG
E
	full_text8
6
4%661 = load double, double* %188, align 16, !tbaa !8
.double*8B

	full_text

double* %188
mcall8Bc
a
	full_textT
R
P%662 = tail call double @llvm.fmuladd.f64(double %661, double %600, double %660)
,double8B

	full_text

double %661
,double8B

	full_text

double %600
,double8B

	full_text

double %660
Pload8BF
D
	full_text7
5
3%663 = load double, double* %321, align 8, !tbaa !8
.double*8B

	full_text

double* %321
mcall8Bc
a
	full_textT
R
P%664 = tail call double @llvm.fmuladd.f64(double %663, double %602, double %662)
,double8B

	full_text

double %663
,double8B

	full_text

double %602
,double8B

	full_text

double %662
Pload8BF
D
	full_text7
5
3%665 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
mcall8Bc
a
	full_textT
R
P%666 = tail call double @llvm.fmuladd.f64(double %665, double %604, double %664)
,double8B

	full_text

double %665
,double8B

	full_text

double %604
,double8B

	full_text

double %664
Qload8BG
E
	full_text8
6
4%667 = load double, double* %322, align 16, !tbaa !8
.double*8B

	full_text

double* %322
mcall8Bc
a
	full_textT
R
P%668 = tail call double @llvm.fmuladd.f64(double %667, double %606, double %666)
,double8B

	full_text

double %667
,double8B

	full_text

double %606
,double8B

	full_text

double %666
Qload8BG
E
	full_text8
6
4%669 = load double, double* %190, align 16, !tbaa !8
.double*8B

	full_text

double* %190
mcall8Bc
a
	full_textT
R
P%670 = tail call double @llvm.fmuladd.f64(double %669, double %608, double %668)
,double8B

	full_text

double %669
,double8B

	full_text

double %608
,double8B

	full_text

double %668
ucall8Bk
i
	full_text\
Z
X%671 = tail call double @llvm.fmuladd.f64(double %670, double 1.200000e+00, double %568)
,double8B

	full_text

double %670
,double8B

	full_text

double %568
Qstore8BF
D
	full_text7
5
3store double %671, double* %569, align 16, !tbaa !8
,double8B

	full_text

double %671
.double*8B

	full_text

double* %569
Pload8BF
D
	full_text7
5
3%672 = load double, double* %330, align 8, !tbaa !8
.double*8B

	full_text

double* %330
Pload8BF
D
	full_text7
5
3%673 = load double, double* %198, align 8, !tbaa !8
.double*8B

	full_text

double* %198
:fmul8B0
.
	full_text!

%674 = fmul double %673, %592
,double8B

	full_text

double %673
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%675 = tail call double @llvm.fmuladd.f64(double %672, double %590, double %674)
,double8B

	full_text

double %672
,double8B

	full_text

double %590
,double8B

	full_text

double %674
Pload8BF
D
	full_text7
5
3%676 = load double, double* %331, align 8, !tbaa !8
.double*8B

	full_text

double* %331
mcall8Bc
a
	full_textT
R
P%677 = tail call double @llvm.fmuladd.f64(double %676, double %594, double %675)
,double8B

	full_text

double %676
,double8B

	full_text

double %594
,double8B

	full_text

double %675
Pload8BF
D
	full_text7
5
3%678 = load double, double* %200, align 8, !tbaa !8
.double*8B

	full_text

double* %200
mcall8Bc
a
	full_textT
R
P%679 = tail call double @llvm.fmuladd.f64(double %678, double %596, double %677)
,double8B

	full_text

double %678
,double8B

	full_text

double %596
,double8B

	full_text

double %677
Pload8BF
D
	full_text7
5
3%680 = load double, double* %333, align 8, !tbaa !8
.double*8B

	full_text

double* %333
mcall8Bc
a
	full_textT
R
P%681 = tail call double @llvm.fmuladd.f64(double %680, double %598, double %679)
,double8B

	full_text

double %680
,double8B

	full_text

double %598
,double8B

	full_text

double %679
Pload8BF
D
	full_text7
5
3%682 = load double, double* %201, align 8, !tbaa !8
.double*8B

	full_text

double* %201
mcall8Bc
a
	full_textT
R
P%683 = tail call double @llvm.fmuladd.f64(double %682, double %600, double %681)
,double8B

	full_text

double %682
,double8B

	full_text

double %600
,double8B

	full_text

double %681
Pload8BF
D
	full_text7
5
3%684 = load double, double* %334, align 8, !tbaa !8
.double*8B

	full_text

double* %334
mcall8Bc
a
	full_textT
R
P%685 = tail call double @llvm.fmuladd.f64(double %684, double %602, double %683)
,double8B

	full_text

double %684
,double8B

	full_text

double %602
,double8B

	full_text

double %683
Pload8BF
D
	full_text7
5
3%686 = load double, double* %202, align 8, !tbaa !8
.double*8B

	full_text

double* %202
mcall8Bc
a
	full_textT
R
P%687 = tail call double @llvm.fmuladd.f64(double %686, double %604, double %685)
,double8B

	full_text

double %686
,double8B

	full_text

double %604
,double8B

	full_text

double %685
Pload8BF
D
	full_text7
5
3%688 = load double, double* %335, align 8, !tbaa !8
.double*8B

	full_text

double* %335
mcall8Bc
a
	full_textT
R
P%689 = tail call double @llvm.fmuladd.f64(double %688, double %606, double %687)
,double8B

	full_text

double %688
,double8B

	full_text

double %606
,double8B

	full_text

double %687
Pload8BF
D
	full_text7
5
3%690 = load double, double* %203, align 8, !tbaa !8
.double*8B

	full_text

double* %203
mcall8Bc
a
	full_textT
R
P%691 = tail call double @llvm.fmuladd.f64(double %690, double %608, double %689)
,double8B

	full_text

double %690
,double8B

	full_text

double %608
,double8B

	full_text

double %689
ucall8Bk
i
	full_text\
Z
X%692 = tail call double @llvm.fmuladd.f64(double %691, double 1.200000e+00, double %580)
,double8B

	full_text

double %691
,double8B

	full_text

double %580
Pstore8BE
C
	full_text6
4
2store double %692, double* %581, align 8, !tbaa !8
,double8B

	full_text

double %692
.double*8B

	full_text

double* %581
Qload8BG
E
	full_text8
6
4%693 = load double, double* %588, align 16, !tbaa !8
.double*8B

	full_text

double* %588
Qload8BG
E
	full_text8
6
4%694 = load double, double* %358, align 16, !tbaa !8
.double*8B

	full_text

double* %358
Qload8BG
E
	full_text8
6
4%695 = load double, double* %227, align 16, !tbaa !8
.double*8B

	full_text

double* %227
:fmul8B0
.
	full_text!

%696 = fmul double %695, %592
,double8B

	full_text

double %695
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%697 = tail call double @llvm.fmuladd.f64(double %694, double %590, double %696)
,double8B

	full_text

double %694
,double8B

	full_text

double %590
,double8B

	full_text

double %696
Pload8BF
D
	full_text7
5
3%698 = load double, double* %365, align 8, !tbaa !8
.double*8B

	full_text

double* %365
mcall8Bc
a
	full_textT
R
P%699 = tail call double @llvm.fmuladd.f64(double %698, double %594, double %697)
,double8B

	full_text

double %698
,double8B

	full_text

double %594
,double8B

	full_text

double %697
Pload8BF
D
	full_text7
5
3%700 = load double, double* %238, align 8, !tbaa !8
.double*8B

	full_text

double* %238
mcall8Bc
a
	full_textT
R
P%701 = tail call double @llvm.fmuladd.f64(double %700, double %596, double %699)
,double8B

	full_text

double %700
,double8B

	full_text

double %596
,double8B

	full_text

double %699
Qload8BG
E
	full_text8
6
4%702 = load double, double* %376, align 16, !tbaa !8
.double*8B

	full_text

double* %376
mcall8Bc
a
	full_textT
R
P%703 = tail call double @llvm.fmuladd.f64(double %702, double %598, double %701)
,double8B

	full_text

double %702
,double8B

	full_text

double %598
,double8B

	full_text

double %701
Qload8BG
E
	full_text8
6
4%704 = load double, double* %245, align 16, !tbaa !8
.double*8B

	full_text

double* %245
mcall8Bc
a
	full_textT
R
P%705 = tail call double @llvm.fmuladd.f64(double %704, double %600, double %703)
,double8B

	full_text

double %704
,double8B

	full_text

double %600
,double8B

	full_text

double %703
Pload8BF
D
	full_text7
5
3%706 = load double, double* %382, align 8, !tbaa !8
.double*8B

	full_text

double* %382
mcall8Bc
a
	full_textT
R
P%707 = tail call double @llvm.fmuladd.f64(double %706, double %602, double %705)
,double8B

	full_text

double %706
,double8B

	full_text

double %602
,double8B

	full_text

double %705
Pload8BF
D
	full_text7
5
3%708 = load double, double* %251, align 8, !tbaa !8
.double*8B

	full_text

double* %251
mcall8Bc
a
	full_textT
R
P%709 = tail call double @llvm.fmuladd.f64(double %708, double %604, double %707)
,double8B

	full_text

double %708
,double8B

	full_text

double %604
,double8B

	full_text

double %707
Qload8BG
E
	full_text8
6
4%710 = load double, double* %388, align 16, !tbaa !8
.double*8B

	full_text

double* %388
mcall8Bc
a
	full_textT
R
P%711 = tail call double @llvm.fmuladd.f64(double %710, double %606, double %709)
,double8B

	full_text

double %710
,double8B

	full_text

double %606
,double8B

	full_text

double %709
Qload8BG
E
	full_text8
6
4%712 = load double, double* %257, align 16, !tbaa !8
.double*8B

	full_text

double* %257
mcall8Bc
a
	full_textT
R
P%713 = tail call double @llvm.fmuladd.f64(double %712, double %608, double %711)
,double8B

	full_text

double %712
,double8B

	full_text

double %608
,double8B

	full_text

double %711
ucall8Bk
i
	full_text\
Z
X%714 = tail call double @llvm.fmuladd.f64(double %713, double 1.200000e+00, double %693)
,double8B

	full_text

double %713
,double8B

	full_text

double %693
Qstore8BF
D
	full_text7
5
3store double %714, double* %588, align 16, !tbaa !8
,double8B

	full_text

double %714
.double*8B

	full_text

double* %588
Nbitcast8BA
?
	full_text2
0
.%715 = bitcast [5 x [5 x double]]* %14 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %14
Kload8BA
?
	full_text2
0
.%716 = load i64, i64* %715, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %715
Nbitcast8BA
?
	full_text2
0
.%717 = bitcast [5 x [5 x double]]* %15 to i64*
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Kstore8B@
>
	full_text1
/
-store i64 %716, i64* %717, align 16, !tbaa !8
&i648B

	full_text


i64 %716
(i64*8B

	full_text

	i64* %717
Bbitcast8B5
3
	full_text&
$
"%718 = bitcast double* %66 to i64*
-double*8B

	full_text

double* %66
Jload8B@
>
	full_text1
/
-%719 = load i64, i64* %718, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %718
„getelementptr8Bq
o
	full_textb
`
^%720 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%721 = bitcast double* %720 to i64*
.double*8B

	full_text

double* %720
Jstore8B?
=
	full_text0
.
,store i64 %719, i64* %721, align 8, !tbaa !8
&i648B

	full_text


i64 %719
(i64*8B

	full_text

	i64* %721
Bbitcast8B5
3
	full_text&
$
"%722 = bitcast double* %67 to i64*
-double*8B

	full_text

double* %67
Kload8BA
?
	full_text2
0
.%723 = load i64, i64* %722, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %722
„getelementptr8Bq
o
	full_textb
`
^%724 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%725 = bitcast double* %724 to i64*
.double*8B

	full_text

double* %724
Kstore8B@
>
	full_text1
/
-store i64 %723, i64* %725, align 16, !tbaa !8
&i648B

	full_text


i64 %723
(i64*8B

	full_text

	i64* %725
Bbitcast8B5
3
	full_text&
$
"%726 = bitcast double* %68 to i64*
-double*8B

	full_text

double* %68
Jload8B@
>
	full_text1
/
-%727 = load i64, i64* %726, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %726
„getelementptr8Bq
o
	full_textb
`
^%728 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%729 = bitcast double* %728 to i64*
.double*8B

	full_text

double* %728
Jstore8B?
=
	full_text0
.
,store i64 %727, i64* %729, align 8, !tbaa !8
&i648B

	full_text


i64 %727
(i64*8B

	full_text

	i64* %729
Bbitcast8B5
3
	full_text&
$
"%730 = bitcast double* %69 to i64*
-double*8B

	full_text

double* %69
Kload8BA
?
	full_text2
0
.%731 = load i64, i64* %730, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %730
„getelementptr8Bq
o
	full_textb
`
^%732 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%733 = bitcast double* %732 to i64*
.double*8B

	full_text

double* %732
Kstore8B@
>
	full_text1
/
-store i64 %731, i64* %733, align 16, !tbaa !8
&i648B

	full_text


i64 %731
(i64*8B

	full_text

	i64* %733
Bbitcast8B5
3
	full_text&
$
"%734 = bitcast double* %75 to i64*
-double*8B

	full_text

double* %75
Jload8B@
>
	full_text1
/
-%735 = load i64, i64* %734, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %734
}getelementptr8Bj
h
	full_text[
Y
W%736 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%737 = bitcast [5 x double]* %736 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %736
Jstore8B?
=
	full_text0
.
,store i64 %735, i64* %737, align 8, !tbaa !8
&i648B

	full_text


i64 %735
(i64*8B

	full_text

	i64* %737
Bbitcast8B5
3
	full_text&
$
"%738 = bitcast double* %79 to i64*
-double*8B

	full_text

double* %79
Jload8B@
>
	full_text1
/
-%739 = load i64, i64* %738, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %738
„getelementptr8Bq
o
	full_textb
`
^%740 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%741 = bitcast double* %740 to i64*
.double*8B

	full_text

double* %740
Jstore8B?
=
	full_text0
.
,store i64 %739, i64* %741, align 8, !tbaa !8
&i648B

	full_text


i64 %739
(i64*8B

	full_text

	i64* %741
Bbitcast8B5
3
	full_text&
$
"%742 = bitcast double* %80 to i64*
-double*8B

	full_text

double* %80
Jload8B@
>
	full_text1
/
-%743 = load i64, i64* %742, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %742
„getelementptr8Bq
o
	full_textb
`
^%744 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%745 = bitcast double* %744 to i64*
.double*8B

	full_text

double* %744
Jstore8B?
=
	full_text0
.
,store i64 %743, i64* %745, align 8, !tbaa !8
&i648B

	full_text


i64 %743
(i64*8B

	full_text

	i64* %745
Oload8BE
C
	full_text6
4
2%746 = load double, double* %81, align 8, !tbaa !8
-double*8B

	full_text

double* %81
„getelementptr8Bq
o
	full_textb
`
^%747 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%748 = load double, double* %82, align 8, !tbaa !8
-double*8B

	full_text

double* %82
„getelementptr8Bq
o
	full_textb
`
^%749 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Bbitcast8B5
3
	full_text&
$
"%750 = bitcast double* %87 to i64*
-double*8B

	full_text

double* %87
Kload8BA
?
	full_text2
0
.%751 = load i64, i64* %750, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %750
}getelementptr8Bj
h
	full_text[
Y
W%752 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%753 = bitcast [5 x double]* %752 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %752
Kstore8B@
>
	full_text1
/
-store i64 %751, i64* %753, align 16, !tbaa !8
&i648B

	full_text


i64 %751
(i64*8B

	full_text

	i64* %753
Bbitcast8B5
3
	full_text&
$
"%754 = bitcast double* %88 to i64*
-double*8B

	full_text

double* %88
Jload8B@
>
	full_text1
/
-%755 = load i64, i64* %754, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %754
„getelementptr8Bq
o
	full_textb
`
^%756 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%757 = bitcast double* %756 to i64*
.double*8B

	full_text

double* %756
Jstore8B?
=
	full_text0
.
,store i64 %755, i64* %757, align 8, !tbaa !8
&i648B

	full_text


i64 %755
(i64*8B

	full_text

	i64* %757
Pload8BF
D
	full_text7
5
3%758 = load double, double* %89, align 16, !tbaa !8
-double*8B

	full_text

double* %89
„getelementptr8Bq
o
	full_textb
`
^%759 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%760 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
„getelementptr8Bq
o
	full_textb
`
^%761 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%762 = load double, double* %91, align 16, !tbaa !8
-double*8B

	full_text

double* %91
„getelementptr8Bq
o
	full_textb
`
^%763 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Bbitcast8B5
3
	full_text&
$
"%764 = bitcast double* %96 to i64*
-double*8B

	full_text

double* %96
Jload8B@
>
	full_text1
/
-%765 = load i64, i64* %764, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %764
}getelementptr8Bj
h
	full_text[
Y
W%766 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%767 = bitcast [5 x double]* %766 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %766
Jstore8B?
=
	full_text0
.
,store i64 %765, i64* %767, align 8, !tbaa !8
&i648B

	full_text


i64 %765
(i64*8B

	full_text

	i64* %767
Oload8BE
C
	full_text6
4
2%768 = load double, double* %97, align 8, !tbaa !8
-double*8B

	full_text

double* %97
„getelementptr8Bq
o
	full_textb
`
^%769 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Oload8BE
C
	full_text6
4
2%770 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
„getelementptr8Bq
o
	full_textb
`
^%771 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%772 = load double, double* %101, align 8, !tbaa !8
.double*8B

	full_text

double* %101
„getelementptr8Bq
o
	full_textb
`
^%773 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%774 = load double, double* %102, align 8, !tbaa !8
.double*8B

	full_text

double* %102
„getelementptr8Bq
o
	full_textb
`
^%775 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 3, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Cbitcast8B6
4
	full_text'
%
#%776 = bitcast double* %115 to i64*
.double*8B

	full_text

double* %115
Kload8BA
?
	full_text2
0
.%777 = load i64, i64* %776, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %776
}getelementptr8Bj
h
	full_text[
Y
W%778 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Ibitcast8B<
:
	full_text-
+
)%779 = bitcast [5 x double]* %778 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %778
Kstore8B@
>
	full_text1
/
-store i64 %777, i64* %779, align 16, !tbaa !8
&i648B

	full_text


i64 %777
(i64*8B

	full_text

	i64* %779
Pload8BF
D
	full_text7
5
3%780 = load double, double* %118, align 8, !tbaa !8
.double*8B

	full_text

double* %118
„getelementptr8Bq
o
	full_textb
`
^%781 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 1
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%782 = load double, double* %120, align 16, !tbaa !8
.double*8B

	full_text

double* %120
„getelementptr8Bq
o
	full_textb
`
^%783 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 2
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%784 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
„getelementptr8Bq
o
	full_textb
`
^%785 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 3
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%786 = load double, double* %125, align 16, !tbaa !8
.double*8B

	full_text

double* %125
„getelementptr8Bq
o
	full_textb
`
^%787 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 4, i64 4
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
„getelementptr8Bq
o
	full_textb
`
^%788 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 0, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%789 = load double, double* %788, align 16, !tbaa !8
.double*8B

	full_text

double* %788
Bfdiv8B8
6
	full_text)
'
%%790 = fdiv double 1.000000e+00, %789
,double8B

	full_text

double %789
„getelementptr8Bq
o
	full_textb
`
^%791 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 1, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Pload8BF
D
	full_text7
5
3%792 = load double, double* %791, align 8, !tbaa !8
.double*8B

	full_text

double* %791
:fmul8B0
.
	full_text!

%793 = fmul double %790, %792
,double8B

	full_text

double %790
,double8B

	full_text

double %792
Abitcast8B4
2
	full_text%
#
!%794 = bitcast i64 %739 to double
&i648B

	full_text


i64 %739
Pload8BF
D
	full_text7
5
3%795 = load double, double* %720, align 8, !tbaa !8
.double*8B

	full_text

double* %720
Cfsub8B9
7
	full_text*
(
&%796 = fsub double -0.000000e+00, %793
,double8B

	full_text

double %793
mcall8Bc
a
	full_textT
R
P%797 = tail call double @llvm.fmuladd.f64(double %796, double %795, double %794)
,double8B

	full_text

double %796
,double8B

	full_text

double %795
,double8B

	full_text

double %794
Pstore8BE
C
	full_text6
4
2store double %797, double* %740, align 8, !tbaa !8
,double8B

	full_text

double %797
.double*8B

	full_text

double* %740
Abitcast8B4
2
	full_text%
#
!%798 = bitcast i64 %743 to double
&i648B

	full_text


i64 %743
Qload8BG
E
	full_text8
6
4%799 = load double, double* %724, align 16, !tbaa !8
.double*8B

	full_text

double* %724
mcall8Bc
a
	full_textT
R
P%800 = tail call double @llvm.fmuladd.f64(double %796, double %799, double %798)
,double8B

	full_text

double %796
,double8B

	full_text

double %799
,double8B

	full_text

double %798
Pstore8BE
C
	full_text6
4
2store double %800, double* %744, align 8, !tbaa !8
,double8B

	full_text

double %800
.double*8B

	full_text

double* %744
Pload8BF
D
	full_text7
5
3%801 = load double, double* %728, align 8, !tbaa !8
.double*8B

	full_text

double* %728
mcall8Bc
a
	full_textT
R
P%802 = tail call double @llvm.fmuladd.f64(double %796, double %801, double %746)
,double8B

	full_text

double %796
,double8B

	full_text

double %801
,double8B

	full_text

double %746
Pstore8BE
C
	full_text6
4
2store double %802, double* %747, align 8, !tbaa !8
,double8B

	full_text

double %802
.double*8B

	full_text

double* %747
Qload8BG
E
	full_text8
6
4%803 = load double, double* %732, align 16, !tbaa !8
.double*8B

	full_text

double* %732
mcall8Bc
a
	full_textT
R
P%804 = tail call double @llvm.fmuladd.f64(double %796, double %803, double %748)
,double8B

	full_text

double %796
,double8B

	full_text

double %803
,double8B

	full_text

double %748
Pstore8BE
C
	full_text6
4
2store double %804, double* %749, align 8, !tbaa !8
,double8B

	full_text

double %804
.double*8B

	full_text

double* %749
Pload8BF
D
	full_text7
5
3%805 = load double, double* %557, align 8, !tbaa !8
.double*8B

	full_text

double* %557
Qload8BG
E
	full_text8
6
4%806 = load double, double* %545, align 16, !tbaa !8
.double*8B

	full_text

double* %545
Cfsub8B9
7
	full_text*
(
&%807 = fsub double -0.000000e+00, %806
,double8B

	full_text

double %806
mcall8Bc
a
	full_textT
R
P%808 = tail call double @llvm.fmuladd.f64(double %807, double %793, double %805)
,double8B

	full_text

double %807
,double8B

	full_text

double %793
,double8B

	full_text

double %805
Pstore8BE
C
	full_text6
4
2store double %808, double* %557, align 8, !tbaa !8
,double8B

	full_text

double %808
.double*8B

	full_text

double* %557
„getelementptr8Bq
o
	full_textb
`
^%809 = getelementptr inbounds [5 x [5 x double]], [5 x [5 x double]]* %15, i64 0, i64 2, i64 0
E[5 x [5 x double]]*8B*
(
	full_text

[5 x [5 x double]]* %15
Qload8BG
E
	full_text8
6
4%810 = load double, double* %809, align 16, !tbaa !8
.double*8B

	full_text

double* %809
:fmul8B0
.
	full_text!

%811 = fmul double %790, %810
,double8B

	full_text

double %790
,double8B

	full_text

double %810
Abitcast8B4
2
	full_text%
#
!%812 = bitcast i64 %755 to double
&i648B

	full_text


i64 %755
Cfsub8B9
7
	full_text*
(
&%813 = fsub double -0.000000e+00, %811
,double8B

	full_text

double %811
mcall8Bc
a
	full_textT
R
P%814 = tail call double @llvm.fmuladd.f64(double %813, double %795, double %812)
,double8B

	full_text

double %813
,double8B

	full_text

double %795
,double8B

	full_text

double %812
Pstore8BE
C
	full_text6
4
2store double %814, double* %756, align 8, !tbaa !8
,double8B

	full_text

double %814
.double*8B

	full_text

double* %756
mcall8Bc
a
	full_textT
R
P%815 = tail call double @llvm.fmuladd.f64(double %813, double %799, double %758)
,double8B

	full_text

double %813
,double8B

	full_text

double %799
,double8B

	full_text

double %758
mcall8Bc
a
	full_textT
R
P%816 = tail call double @llvm.fmuladd.f64(double %813, double %801, double %760)
,double8B

	full_text

double %813
,double8B

	full_text

double %801
,double8B

	full_text

double %760
mcall8Bc
a
	full_textT
R
P%817 = tail call double @llvm.fmuladd.f64(double %813, double %803, double %762)
,double8B

	full_text

double %813
,double8B

	full_text

double %803
,double8B

	full_text

double %762
Qload8BG
E
	full_text8
6
4%818 = load double, double* %569, align 16, !tbaa !8
.double*8B

	full_text

double* %569
mcall8Bc
a
	full_textT
R
P%819 = tail call double @llvm.fmuladd.f64(double %807, double %811, double %818)
,double8B

	full_text

double %807
,double8B

	full_text

double %811
,double8B

	full_text

double %818
Abitcast8B4
2
	full_text%
#
!%820 = bitcast i64 %765 to double
&i648B

	full_text


i64 %765
:fmul8B0
.
	full_text!

%821 = fmul double %790, %820
,double8B

	full_text

double %790
,double8B

	full_text

double %820
Cfsub8B9
7
	full_text*
(
&%822 = fsub double -0.000000e+00, %821
,double8B

	full_text

double %821
mcall8Bc
a
	full_textT
R
P%823 = tail call double @llvm.fmuladd.f64(double %822, double %795, double %768)
,double8B

	full_text

double %822
,double8B

	full_text

double %795
,double8B

	full_text

double %768
Pstore8BE
C
	full_text6
4
2store double %823, double* %769, align 8, !tbaa !8
,double8B

	full_text

double %823
.double*8B

	full_text

double* %769
mcall8Bc
a
	full_textT
R
P%824 = tail call double @llvm.fmuladd.f64(double %822, double %799, double %770)
,double8B

	full_text

double %822
,double8B

	full_text

double %799
,double8B

	full_text

double %770
mcall8Bc
a
	full_textT
R
P%825 = tail call double @llvm.fmuladd.f64(double %822, double %801, double %772)
,double8B

	full_text

double %822
,double8B

	full_text

double %801
,double8B

	full_text

double %772
mcall8Bc
a
	full_textT
R
P%826 = tail call double @llvm.fmuladd.f64(double %822, double %803, double %774)
,double8B

	full_text

double %822
,double8B

	full_text

double %803
,double8B

	full_text

double %774
Pload8BF
D
	full_text7
5
3%827 = load double, double* %581, align 8, !tbaa !8
.double*8B

	full_text

double* %581
mcall8Bc
a
	full_textT
R
P%828 = tail call double @llvm.fmuladd.f64(double %807, double %821, double %827)
,double8B

	full_text

double %807
,double8B

	full_text

double %821
,double8B

	full_text

double %827
Abitcast8B4
2
	full_text%
#
!%829 = bitcast i64 %777 to double
&i648B

	full_text


i64 %777
:fmul8B0
.
	full_text!

%830 = fmul double %790, %829
,double8B

	full_text

double %790
,double8B

	full_text

double %829
Cfsub8B9
7
	full_text*
(
&%831 = fsub double -0.000000e+00, %830
,double8B

	full_text

double %830
mcall8Bc
a
	full_textT
R
P%832 = tail call double @llvm.fmuladd.f64(double %831, double %795, double %780)
,double8B

	full_text

double %831
,double8B

	full_text

double %795
,double8B

	full_text

double %780
Pstore8BE
C
	full_text6
4
2store double %832, double* %781, align 8, !tbaa !8
,double8B

	full_text

double %832
.double*8B

	full_text

double* %781
mcall8Bc
a
	full_textT
R
P%833 = tail call double @llvm.fmuladd.f64(double %831, double %799, double %782)
,double8B

	full_text

double %831
,double8B

	full_text

double %799
,double8B

	full_text

double %782
mcall8Bc
a
	full_textT
R
P%834 = tail call double @llvm.fmuladd.f64(double %831, double %801, double %784)
,double8B

	full_text

double %831
,double8B

	full_text

double %801
,double8B

	full_text

double %784
mcall8Bc
a
	full_textT
R
P%835 = tail call double @llvm.fmuladd.f64(double %831, double %803, double %786)
,double8B

	full_text

double %831
,double8B

	full_text

double %803
,double8B

	full_text

double %786
Qload8BG
E
	full_text8
6
4%836 = load double, double* %588, align 16, !tbaa !8
.double*8B

	full_text

double* %588
mcall8Bc
a
	full_textT
R
P%837 = tail call double @llvm.fmuladd.f64(double %807, double %830, double %836)
,double8B

	full_text

double %807
,double8B

	full_text

double %830
,double8B

	full_text

double %836
Bfdiv8B8
6
	full_text)
'
%%838 = fdiv double 1.000000e+00, %797
,double8B

	full_text

double %797
:fmul8B0
.
	full_text!

%839 = fmul double %838, %814
,double8B

	full_text

double %838
,double8B

	full_text

double %814
Cfsub8B9
7
	full_text*
(
&%840 = fsub double -0.000000e+00, %839
,double8B

	full_text

double %839
mcall8Bc
a
	full_textT
R
P%841 = tail call double @llvm.fmuladd.f64(double %840, double %800, double %815)
,double8B

	full_text

double %840
,double8B

	full_text

double %800
,double8B

	full_text

double %815
Qstore8BF
D
	full_text7
5
3store double %841, double* %759, align 16, !tbaa !8
,double8B

	full_text

double %841
.double*8B

	full_text

double* %759
mcall8Bc
a
	full_textT
R
P%842 = tail call double @llvm.fmuladd.f64(double %840, double %802, double %816)
,double8B

	full_text

double %840
,double8B

	full_text

double %802
,double8B

	full_text

double %816
Pstore8BE
C
	full_text6
4
2store double %842, double* %761, align 8, !tbaa !8
,double8B

	full_text

double %842
.double*8B

	full_text

double* %761
mcall8Bc
a
	full_textT
R
P%843 = tail call double @llvm.fmuladd.f64(double %840, double %804, double %817)
,double8B

	full_text

double %840
,double8B

	full_text

double %804
,double8B

	full_text

double %817
Qstore8BF
D
	full_text7
5
3store double %843, double* %763, align 16, !tbaa !8
,double8B

	full_text

double %843
.double*8B

	full_text

double* %763
Cfsub8B9
7
	full_text*
(
&%844 = fsub double -0.000000e+00, %808
,double8B

	full_text

double %808
mcall8Bc
a
	full_textT
R
P%845 = tail call double @llvm.fmuladd.f64(double %844, double %839, double %819)
,double8B

	full_text

double %844
,double8B

	full_text

double %839
,double8B

	full_text

double %819
:fmul8B0
.
	full_text!

%846 = fmul double %838, %823
,double8B

	full_text

double %838
,double8B

	full_text

double %823
Cfsub8B9
7
	full_text*
(
&%847 = fsub double -0.000000e+00, %846
,double8B

	full_text

double %846
mcall8Bc
a
	full_textT
R
P%848 = tail call double @llvm.fmuladd.f64(double %847, double %800, double %824)
,double8B

	full_text

double %847
,double8B

	full_text

double %800
,double8B

	full_text

double %824
Pstore8BE
C
	full_text6
4
2store double %848, double* %771, align 8, !tbaa !8
,double8B

	full_text

double %848
.double*8B

	full_text

double* %771
mcall8Bc
a
	full_textT
R
P%849 = tail call double @llvm.fmuladd.f64(double %847, double %802, double %825)
,double8B

	full_text

double %847
,double8B

	full_text

double %802
,double8B

	full_text

double %825
mcall8Bc
a
	full_textT
R
P%850 = tail call double @llvm.fmuladd.f64(double %847, double %804, double %826)
,double8B

	full_text

double %847
,double8B

	full_text

double %804
,double8B

	full_text

double %826
mcall8Bc
a
	full_textT
R
P%851 = tail call double @llvm.fmuladd.f64(double %844, double %846, double %828)
,double8B

	full_text

double %844
,double8B

	full_text

double %846
,double8B

	full_text

double %828
:fmul8B0
.
	full_text!

%852 = fmul double %838, %832
,double8B

	full_text

double %838
,double8B

	full_text

double %832
Cfsub8B9
7
	full_text*
(
&%853 = fsub double -0.000000e+00, %852
,double8B

	full_text

double %852
mcall8Bc
a
	full_textT
R
P%854 = tail call double @llvm.fmuladd.f64(double %853, double %800, double %833)
,double8B

	full_text

double %853
,double8B

	full_text

double %800
,double8B

	full_text

double %833
Qstore8BF
D
	full_text7
5
3store double %854, double* %783, align 16, !tbaa !8
,double8B

	full_text

double %854
.double*8B

	full_text

double* %783
mcall8Bc
a
	full_textT
R
P%855 = tail call double @llvm.fmuladd.f64(double %853, double %802, double %834)
,double8B

	full_text

double %853
,double8B

	full_text

double %802
,double8B

	full_text

double %834
mcall8Bc
a
	full_textT
R
P%856 = tail call double @llvm.fmuladd.f64(double %853, double %804, double %835)
,double8B

	full_text

double %853
,double8B

	full_text

double %804
,double8B

	full_text

double %835
mcall8Bc
a
	full_textT
R
P%857 = tail call double @llvm.fmuladd.f64(double %844, double %852, double %837)
,double8B

	full_text

double %844
,double8B

	full_text

double %852
,double8B

	full_text

double %837
Bfdiv8B8
6
	full_text)
'
%%858 = fdiv double 1.000000e+00, %841
,double8B

	full_text

double %841
:fmul8B0
.
	full_text!

%859 = fmul double %858, %848
,double8B

	full_text

double %858
,double8B

	full_text

double %848
Cfsub8B9
7
	full_text*
(
&%860 = fsub double -0.000000e+00, %859
,double8B

	full_text

double %859
mcall8Bc
a
	full_textT
R
P%861 = tail call double @llvm.fmuladd.f64(double %860, double %842, double %849)
,double8B

	full_text

double %860
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
2store double %861, double* %773, align 8, !tbaa !8
,double8B

	full_text

double %861
.double*8B

	full_text

double* %773
mcall8Bc
a
	full_textT
R
P%862 = tail call double @llvm.fmuladd.f64(double %860, double %843, double %850)
,double8B

	full_text

double %860
,double8B

	full_text

double %843
,double8B

	full_text

double %850
Pstore8BE
C
	full_text6
4
2store double %862, double* %775, align 8, !tbaa !8
,double8B

	full_text

double %862
.double*8B

	full_text

double* %775
Cfsub8B9
7
	full_text*
(
&%863 = fsub double -0.000000e+00, %845
,double8B

	full_text

double %845
mcall8Bc
a
	full_textT
R
P%864 = tail call double @llvm.fmuladd.f64(double %863, double %859, double %851)
,double8B

	full_text

double %863
,double8B

	full_text

double %859
,double8B

	full_text

double %851
:fmul8B0
.
	full_text!

%865 = fmul double %858, %854
,double8B

	full_text

double %858
,double8B

	full_text

double %854
Cfsub8B9
7
	full_text*
(
&%866 = fsub double -0.000000e+00, %865
,double8B

	full_text

double %865
mcall8Bc
a
	full_textT
R
P%867 = tail call double @llvm.fmuladd.f64(double %866, double %842, double %855)
,double8B

	full_text

double %866
,double8B

	full_text

double %842
,double8B

	full_text

double %855
Pstore8BE
C
	full_text6
4
2store double %867, double* %785, align 8, !tbaa !8
,double8B

	full_text

double %867
.double*8B

	full_text

double* %785
mcall8Bc
a
	full_textT
R
P%868 = tail call double @llvm.fmuladd.f64(double %866, double %843, double %856)
,double8B

	full_text

double %866
,double8B

	full_text

double %843
,double8B

	full_text

double %856
mcall8Bc
a
	full_textT
R
P%869 = tail call double @llvm.fmuladd.f64(double %863, double %865, double %857)
,double8B

	full_text

double %863
,double8B

	full_text

double %865
,double8B

	full_text

double %857
Bfdiv8B8
6
	full_text)
'
%%870 = fdiv double 1.000000e+00, %861
,double8B

	full_text

double %861
:fmul8B0
.
	full_text!

%871 = fmul double %870, %867
,double8B

	full_text

double %870
,double8B

	full_text

double %867
Cfsub8B9
7
	full_text*
(
&%872 = fsub double -0.000000e+00, %871
,double8B

	full_text

double %871
mcall8Bc
a
	full_textT
R
P%873 = tail call double @llvm.fmuladd.f64(double %872, double %862, double %868)
,double8B

	full_text

double %872
,double8B

	full_text

double %862
,double8B

	full_text

double %868
Qstore8BF
D
	full_text7
5
3store double %873, double* %787, align 16, !tbaa !8
,double8B

	full_text

double %873
.double*8B

	full_text

double* %787
Cfsub8B9
7
	full_text*
(
&%874 = fsub double -0.000000e+00, %864
,double8B

	full_text

double %864
mcall8Bc
a
	full_textT
R
P%875 = tail call double @llvm.fmuladd.f64(double %874, double %871, double %869)
,double8B

	full_text

double %874
,double8B

	full_text

double %871
,double8B

	full_text

double %869
:fdiv8B0
.
	full_text!

%876 = fdiv double %875, %873
,double8B

	full_text

double %875
,double8B

	full_text

double %873
Qstore8BF
D
	full_text7
5
3store double %876, double* %588, align 16, !tbaa !8
,double8B

	full_text

double %876
.double*8B

	full_text

double* %588
Cfsub8B9
7
	full_text*
(
&%877 = fsub double -0.000000e+00, %862
,double8B

	full_text

double %862
mcall8Bc
a
	full_textT
R
P%878 = tail call double @llvm.fmuladd.f64(double %877, double %876, double %864)
,double8B

	full_text

double %877
,double8B

	full_text

double %876
,double8B

	full_text

double %864
:fdiv8B0
.
	full_text!

%879 = fdiv double %878, %861
,double8B

	full_text

double %878
,double8B

	full_text

double %861
Pstore8BE
C
	full_text6
4
2store double %879, double* %581, align 8, !tbaa !8
,double8B

	full_text

double %879
.double*8B

	full_text

double* %581
Cfsub8B9
7
	full_text*
(
&%880 = fsub double -0.000000e+00, %842
,double8B

	full_text

double %842
mcall8Bc
a
	full_textT
R
P%881 = tail call double @llvm.fmuladd.f64(double %880, double %879, double %845)
,double8B

	full_text

double %880
,double8B

	full_text

double %879
,double8B

	full_text

double %845
Cfsub8B9
7
	full_text*
(
&%882 = fsub double -0.000000e+00, %843
,double8B

	full_text

double %843
mcall8Bc
a
	full_textT
R
P%883 = tail call double @llvm.fmuladd.f64(double %882, double %876, double %881)
,double8B

	full_text

double %882
,double8B

	full_text

double %876
,double8B

	full_text

double %881
:fdiv8B0
.
	full_text!

%884 = fdiv double %883, %841
,double8B

	full_text

double %883
,double8B

	full_text

double %841
Qstore8BF
D
	full_text7
5
3store double %884, double* %569, align 16, !tbaa !8
,double8B

	full_text

double %884
.double*8B

	full_text

double* %569
Cfsub8B9
7
	full_text*
(
&%885 = fsub double -0.000000e+00, %800
,double8B

	full_text

double %800
mcall8Bc
a
	full_textT
R
P%886 = tail call double @llvm.fmuladd.f64(double %885, double %884, double %808)
,double8B

	full_text

double %885
,double8B

	full_text

double %884
,double8B

	full_text

double %808
Cfsub8B9
7
	full_text*
(
&%887 = fsub double -0.000000e+00, %802
,double8B

	full_text

double %802
mcall8Bc
a
	full_textT
R
P%888 = tail call double @llvm.fmuladd.f64(double %887, double %879, double %886)
,double8B

	full_text

double %887
,double8B

	full_text

double %879
,double8B

	full_text

double %886
Cfsub8B9
7
	full_text*
(
&%889 = fsub double -0.000000e+00, %804
,double8B

	full_text

double %804
mcall8Bc
a
	full_textT
R
P%890 = tail call double @llvm.fmuladd.f64(double %889, double %876, double %888)
,double8B

	full_text

double %889
,double8B

	full_text

double %876
,double8B

	full_text

double %888
:fdiv8B0
.
	full_text!

%891 = fdiv double %890, %797
,double8B

	full_text

double %890
,double8B

	full_text

double %797
Pstore8BE
C
	full_text6
4
2store double %891, double* %557, align 8, !tbaa !8
,double8B

	full_text

double %891
.double*8B

	full_text

double* %557
Cfsub8B9
7
	full_text*
(
&%892 = fsub double -0.000000e+00, %795
,double8B

	full_text

double %795
mcall8Bc
a
	full_textT
R
P%893 = tail call double @llvm.fmuladd.f64(double %892, double %891, double %806)
,double8B

	full_text

double %892
,double8B

	full_text

double %891
,double8B

	full_text

double %806
Cfsub8B9
7
	full_text*
(
&%894 = fsub double -0.000000e+00, %799
,double8B

	full_text

double %799
mcall8Bc
a
	full_textT
R
P%895 = tail call double @llvm.fmuladd.f64(double %894, double %884, double %893)
,double8B

	full_text

double %894
,double8B

	full_text

double %884
,double8B

	full_text

double %893
Cfsub8B9
7
	full_text*
(
&%896 = fsub double -0.000000e+00, %801
,double8B

	full_text

double %801
mcall8Bc
a
	full_textT
R
P%897 = tail call double @llvm.fmuladd.f64(double %896, double %879, double %895)
,double8B

	full_text

double %896
,double8B

	full_text

double %879
,double8B

	full_text

double %895
Cfsub8B9
7
	full_text*
(
&%898 = fsub double -0.000000e+00, %803
,double8B

	full_text

double %803
mcall8Bc
a
	full_textT
R
P%899 = tail call double @llvm.fmuladd.f64(double %898, double %876, double %897)
,double8B

	full_text

double %898
,double8B

	full_text

double %876
,double8B

	full_text

double %897
:fdiv8B0
.
	full_text!

%900 = fdiv double %899, %789
,double8B

	full_text

double %899
,double8B

	full_text

double %789
Qstore8BF
D
	full_text7
5
3store double %900, double* %545, align 16, !tbaa !8
,double8B

	full_text

double %900
.double*8B

	full_text

double* %545
£getelementptr8B
Œ
	full_text
}
{%901 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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
3%902 = load double, double* %901, align 8, !tbaa !8
.double*8B

	full_text

double* %901
:fsub8B0
.
	full_text!

%903 = fsub double %902, %900
,double8B

	full_text

double %902
,double8B

	full_text

double %900
Pstore8BE
C
	full_text6
4
2store double %903, double* %901, align 8, !tbaa !8
,double8B

	full_text

double %903
.double*8B

	full_text

double* %901
£getelementptr8B
Œ
	full_text
}
{%904 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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
3%905 = load double, double* %904, align 8, !tbaa !8
.double*8B

	full_text

double* %904
:fsub8B0
.
	full_text!

%906 = fsub double %905, %891
,double8B

	full_text

double %905
,double8B

	full_text

double %891
Pstore8BE
C
	full_text6
4
2store double %906, double* %904, align 8, !tbaa !8
,double8B

	full_text

double %906
.double*8B

	full_text

double* %904
£getelementptr8B
Œ
	full_text
}
{%907 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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
3%908 = load double, double* %907, align 8, !tbaa !8
.double*8B

	full_text

double* %907
:fsub8B0
.
	full_text!

%909 = fsub double %908, %884
,double8B

	full_text

double %908
,double8B

	full_text

double %884
Pstore8BE
C
	full_text6
4
2store double %909, double* %907, align 8, !tbaa !8
,double8B

	full_text

double %909
.double*8B

	full_text

double* %907
£getelementptr8B
Œ
	full_text
}
{%910 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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
3%911 = load double, double* %910, align 8, !tbaa !8
.double*8B

	full_text

double* %910
:fsub8B0
.
	full_text!

%912 = fsub double %911, %879
,double8B

	full_text

double %911
,double8B

	full_text

double %879
Pstore8BE
C
	full_text6
4
2store double %912, double* %910, align 8, !tbaa !8
,double8B

	full_text

double %912
.double*8B

	full_text

double* %910
£getelementptr8B
Œ
	full_text
}
{%913 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %51, i64 %56, i64 %58, i64 %60, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %51
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
3%914 = load double, double* %913, align 8, !tbaa !8
.double*8B

	full_text

double* %913
:fsub8B0
.
	full_text!

%915 = fsub double %914, %876
,double8B

	full_text

double %914
,double8B

	full_text

double %876
Pstore8BE
C
	full_text6
4
2store double %915, double* %913, align 8, !tbaa !8
,double8B

	full_text

double %915
.double*8B

	full_text

double* %913
(br8B 

	full_text

br label %916
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


double* %3
,double*8B

	full_text


double* %1
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
$i328B

	full_text


i32 %8
$i328B

	full_text


i32 %5
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %9
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
:double8B,
*
	full_text

double 0xBFB00AEC33E1F670
5double8B'
%
	full_text

double -4.000000e+00
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 -1
:double8B,
*
	full_text

double 0xBFC1111111111111
#i648B

	full_text	

i64 1
:double8B,
*
	full_text

double 0x3FB89374BC6A7EF8
4double8B&
$
	full_text

double 1.600000e+00
:double8B,
*
	full_text

double 0x4059333333333334
4double8B&
$
	full_text

double 0.000000e+00
:double8B,
*
	full_text

double 0x40A23B8B43958106
$i648B

	full_text


i64 40
:double8B,
*
	full_text

double 0x40E3616000000001
4double8B&
$
	full_text

double 1.323000e+04
:double8B,
*
	full_text

double 0xC08F962D0E560417
:double8B,
*
	full_text

double 0xC0E9D70000000001
:double8B,
*
	full_text

double 0x40BF020000000001
5double8B'
%
	full_text

double -1.000000e-01
:double8B,
*
	full_text

double 0x40E3614000000001
5double8B'
%
	full_text

double -4.000000e-01
:double8B,
*
	full_text

double 0x40984F645A1CAC08
:double8B,
*
	full_text

double 0x3FB00AEC33E1F670
4double8B&
$
	full_text

double 1.400000e+00
#i648B

	full_text	

i64 3
:double8B,
*
	full_text

double 0xC0B7418000000001
#i648B

	full_text	

i64 2
:double8B,
*
	full_text

double 0x3FC916872B020C49
:double8B,
*
	full_text

double 0xC07F172B020C49B9
:double8B,
*
	full_text

double 0xBFB89374BC6A7EF8
:double8B,
*
	full_text

double 0x4088CE6666666668
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 1.000000e-01
#i648B

	full_text	

i64 4
:double8B,
*
	full_text

double 0xC0AF962D0E560417
#i328B

	full_text	

i32 0
5double8B'
%
	full_text

double -5.292000e+04
:double8B,
*
	full_text

double 0x4039333333333334
:double8B,
*
	full_text

double 0x40C9D70000000001
:double8B,
*
	full_text

double 0x3FC1111111111111
#i648B

	full_text	

i64 0
:double8B,
*
	full_text

double 0xC087D0624DD2F1A9
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 1.000000e+00
4double8B&
$
	full_text

double 1.200000e+00
4double8B&
$
	full_text

double 4.000000e-01
:double8B,
*
	full_text

double 0xC0BF020000000001
:double8B,
*
	full_text

double 0x40C23B8B43958106
%i648B

	full_text
	
i64 200
4double8B&
$
	full_text

double 6.300000e+01
,i648B!

	full_text

i64 4294967296
4double8B&
$
	full_text

double 8.000000e-01        	
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
ï îî ðñ ðð òó òò ôõ ôô ö÷ ö
ø öö ùú ùù û
ü ûû ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž ŽŽ ‘ 
’ 
“ 
”  •– •• —˜ —
™ —— š› š
œ š
 šš žŸ žž  ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ ÁÁ ÃÄ ÃÃ ÅÆ Å
Ç ÅÅ ÈÉ ÈÈ ÊË ÊÊ ÌÍ Ì
Î Ì
Ï Ì
Ð ÌÌ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ Û
Ü ÛÛ ÝÞ ÝÝ ß
à ßß áâ áá ã
ä ãã åæ åå ç
è çç éê éé ë
ì ëë íî í
ï í
ð í
ñ íí òó òò ôõ ô
ö ôô ÷
ø ÷÷ ùú ù
û ù
ü ù
ý ùù þÿ þþ € €€ ‚ƒ ‚
„ ‚‚ …† …
‡ …
ˆ …… ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž ŽŽ 
‘  ’“ ’
” ’’ •– •• —˜ —
™ —— š› šš œ œœ žŸ žž  
¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬
¯ ¬
° ¬¬ ±² ±± ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿
Â ¿
Ã ¿¿ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ ËË ÍÎ ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ ÒÒ Ô
Õ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ Ü
Ý ÜÜ Þß ÞÞ àá à
â àà ãä ãã å
æ åå çè ç
é çç êë êê ìí ì
î ìì ïð ïï ñò ññ óô ó
õ óó ö÷ öö øù øø ú
û úú üý ü
þ üü ÿ€ ÿÿ ‚  ƒ„ ƒ
… ƒƒ †‡ †† ˆ
‰ ˆˆ Š‹ ŠŠ Œ
 ŒŒ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”
• ”” –— –
˜ –– ™š ™™ ›
œ ›› ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬¬ ®
¯ ®® °± °° ²³ ²
´ ²² µ¶ µµ ·
¸ ·· ¹º ¹
» ¹
¼ ¹
½ ¹¹ ¾¿ ¾¾ ÀÁ ÀÀ Â
Ã ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ ÒÒ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× Ú
Û ÚÚ ÜÝ Ü
Þ Ü
ß ÜÜ àá à
â àà ã
ä ãã åæ å
ç å
è åå éê éé ë
ì ëë íî í
ï í
ð íí ñò ññ ó
ô óó õö õ
÷ õõ øù øø úû ú
ü úú ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒ
… ƒ
† ƒƒ ‡ˆ ‡‡ ‰
Š ‰‰ ‹Œ ‹
 ‹‹ Ž ŽŽ ‘ 
’  “
” ““ •– •
— •• ˜™ ˜˜ š› š
œ šš ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §
¨ §§ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®® ±² ±± ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹
º ¹¹ »¼ »
½ »» ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ ÎÎ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×
Ú ×
Û ×× ÜÝ ÜÜ Þß Þ
à ÞÞ áâ á
ã áá äå ää æ
ç ææ èé èè ê
ë êê ìí ìì î
ï îî ðñ ðð ò
ó òò ôõ ôô ö
÷ öö øù ø
ú ø
û ø
ü øø ýþ ýý ÿ€ ÿ
 ÿ
‚ ÿ
ƒ ÿÿ „… „„ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ
 ŒŒ Ž ŽŽ ‘ 
’  “” ““ •
– •• —˜ —
™ —— š› šš œ œ
ž œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤¤ ¦
§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­­ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µµ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾
¿ ¾¾ ÀÁ ÀÀ Â
Ã ÂÂ Ä
Å ÄÄ ÆÇ Æ
È Æ
É Æ
Ê ÆÆ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ ÐÐ ÒÓ Ò
Ô Ò
Õ ÒÒ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ ÛÛ Ý
Þ ÝÝ ßà ß
á ßß âã ââ äå ä
æ ää çè çç éê éé ëì ëë íî í
ï íí ðñ ðð òó òò ôõ ôô ö
÷ öö øù ø
ú øø ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚
† ‚‚ ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ Ž ŽŽ ‘  ’“ ’
” ’’ •– •• —
˜ —— ™š ™
› ™™ œ œ
ž œœ Ÿ
  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦
§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °° ²
³ ²² ´µ ´´ ¶· ¶¶ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½
¿ ½½ ÀÁ ÀÀ Â
Ã ÂÂ ÄÅ Ä
Æ Ä
Ç Ä
È ÄÄ ÉÊ ÉÉ ËÌ ËË Í
Î ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ ØØ ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ßà ß
á ßß âã â
ä ââ å
æ åå çè ç
é ç
ê çç ëì ë
í ëë îï î
ð î
ñ îî òó òò ô
õ ôô ö÷ ö
ø ö
ù öö úû úú ü
ý üü þÿ þ
€ þþ ‚  ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž 
  
‘  ’“ ’
” ’’ •– •• —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢  
£    ¤¥ ¤¤ ¦
§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °
± °° ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ Â
Ã ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ ÎÎ Ð
Ñ ÐÐ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß ÞÞ àá à
â à
ã à
ä àà åæ åå çè ç
é çç êë ê
ì êê íî íí ï
ð ïï ñò ññ ó
ô óó õö õõ ÷
ø ÷÷ ùú ùù û
ü ûû ýþ ýý ÿ
€	 ÿÿ 	‚	 	
ƒ	 	
„	 	
…	 		 †	‡	 †	†	 ˆ	‰	 ˆ	
Š	 ˆ	
‹	 ˆ	
Œ	 ˆ	ˆ	 	Ž	 		 		 	
‘	 		 ’	“	 ’	
”	 ’	’	 •	
–	 •	•	 —	˜	 —	—	 ™	š	 ™	
›	 ™	™	 œ		 œ	œ	 ž	
Ÿ	 ž	ž	  	¡	  	
¢	  	 	 £	¤	 £	£	 ¥	¦	 ¥	
§	 ¥	¥	 ¨	©	 ¨	
ª	 ¨	¨	 «	¬	 «	«	 ­	
®	 ­	­	 ¯	°	 ¯	
±	 ¯	¯	 ²	³	 ²	²	 ´	µ	 ´	´	 ¶	·	 ¶	
¸	 ¶	¶	 ¹	º	 ¹	¹	 »	
¼	 »	»	 ½	¾	 ½	
¿	 ½	½	 À	Á	 À	À	 Â	Ã	 Â	Â	 Ä	Å	 Ä	
Æ	 Ä	Ä	 Ç	È	 Ç	Ç	 É	
Ê	 É	É	 Ë	Ì	 Ë	
Í	 Ë	
Î	 Ë	
Ï	 Ë	Ë	 Ð	Ñ	 Ð	Ð	 Ò	Ó	 Ò	
Ô	 Ò	Ò	 Õ	Ö	 Õ	
×	 Õ	Õ	 Ø	
Ù	 Ø	Ø	 Ú	Û	 Ú	
Ü	 Ú	Ú	 Ý	Þ	 Ý	Ý	 ß	
à	 ß	ß	 á	â	 á	
ã	 á	á	 ä	å	 ä	ä	 æ	ç	 æ	
è	 æ	æ	 é	ê	 é	é	 ë	
ì	 ë	ë	 í	î	 í	í	 ï	ð	 ï	ï	 ñ	
ò	 ñ	ñ	 ó	ô	 ó	
õ	 ó	ó	 ö	÷	 ö	ö	 ø	ù	 ø	ø	 ú	û	 ú	
ü	 ú	ú	 ý	þ	 ý	
ÿ	 ý	ý	 €

 €
€
 ‚
ƒ
 ‚
‚
 „
…
 „

†
 „
„
 ‡
ˆ
 ‡
‡
 ‰

Š
 ‰
‰
 ‹

Œ
 ‹
‹
 
Ž
 


 


 

‘
 

 ’
“
 ’
’
 ”
•
 ”

–
 ”
”
 —
˜
 —
—
 ™
š
 ™

›
 ™

œ
 ™
™
 
ž
 

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

¥
 ¤
¤
 ¦
§
 ¦

¨
 ¦
¦
 ©
ª
 ©
©
 «
¬
 «

­
 «
«
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
²
 ´
µ
 ´

¶
 ´
´
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
»
 ½
¾
 ½

¿
 ½
½
 À
Á
 À
À
 Â
Ã
 Â
Â
 Ä

Å
 Ä
Ä
 Æ
Ç
 Æ

È
 Æ
Æ
 É
Ê
 É
É
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
 Ð
Ñ
 Ð
Ð
 Ò

Ó
 Ò
Ò
 Ô
Õ
 Ô

Ö
 Ô

×
 Ô

Ø
 Ô
Ô
 Ù
Ú
 Ù
Ù
 Û
Ü
 Û
Û
 Ý

Þ
 Ý
Ý
 ß
à
 ß

á
 ß
ß
 â
ã
 â

ä
 â
â
 å
æ
 å

ç
 å
å
 è
é
 è
è
 ê
ë
 ê

ì
 ê
ê
 í
î
 í
í
 ï
ð
 ï

ñ
 ï
ï
 ò
ó
 ò

ô
 ò
ò
 õ

ö
 õ
õ
 ÷
ø
 ÷

ù
 ÷

ú
 ÷
÷
 û
ü
 û
û
 ý
þ
 ý

ÿ
 ý
ý
 € €
‚ €
ƒ €€ „… „„ †
‡ †† ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž
 ŽŽ ‘ 
’  “” ““ •– •
— •• ˜™ ˜˜ š› š
œ šš ž  Ÿ  Ÿ
¡ ŸŸ ¢
£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±± ´
µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ ÈÈ Ê
Ë ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ Ô
Õ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ ÛÜ Û
Ý ÛÛ Þß ÞÞ àá àà â
ã ââ äå ä
æ ää çè çç éê éé ëì ë
í ëë îï î
ð î
ñ î
ò îî óô óó õö õ
÷ õ
ø õ
ù õõ úû úú üý ü
þ ü
ÿ ü
€ üü ‚  ƒ„ ƒ
… ƒ
† ƒ
‡ ƒƒ ˆ‰ ˆˆ Š‹ Š
Œ Š
 Š
Ž ŠŠ   ‘’ ‘‘ “” ““ •– •
— •• ˜™ ˜
š ˜
› ˜˜ œ œœ žŸ ž
  ž
¡ žž ¢£ ¢¢ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨¨ ª« ª
¬ ª
­ ªª ®¯ ®® °± °° ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹
¼ ¹¹ ½¾ ½½ ¿À ¿
Á ¿
Â ¿¿ ÃÄ ÃÃ ÅÆ Å
Ç Å
È ÅÅ ÉÊ ÉÉ ËÌ Ë
Í Ë
Î ËË ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü Ú
Ý ÚÚ Þß ÞÞ àá à
â à
ã àà äå ää æç æ
è æ
é ææ êë êê ìí ì
î ì
ï ìì ðñ ðð òó òò ôõ ô
ö ôô ÷ø ÷÷ ùú ùù ûü û
ý ûû þÿ þ
€ þ
 þþ ‚ƒ ‚‚ „… „
† „
‡ „„ ˆ‰ ˆˆ Š‹ Š
Œ Š
 ŠŠ Ž ŽŽ ‘ 
’ 
“  ”• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ ž
  ž
¡ žž ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦
¨ ¦
© ¦¦ ª« ª
¬ ª
­ ªª ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µ
· µ
¸ µ
¹ µµ º» ºº ¼½ ¼
¾ ¼
¿ ¼
À ¼¼ ÁÂ ÁÁ ÃÄ Ã
Å Ã
Æ Ã
Ç ÃÃ ÈÉ ÈÈ ÊË Ê
Ì Ê
Í Ê
Î ÊÊ ÏÐ ÏÏ ÑÒ Ñ
Ó Ñ
Ô Ñ
Õ ÑÑ Ö× ÖÖ ØÙ Ø
Ú Ø
Û Ø
Ü ØØ ÝÞ ÝÝ ßà ß
á ß
â ß
ã ßß äå ää æç æ
è æ
é æ
ê ææ ëì ëë íî í
ï í
ð í
ñ íí òó òò ôõ ô
ö ô
÷ ô
ø ôô ùú ùù ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †† ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž Ž
 Ž
‘ ŽŽ ’“ ’’ ”• ”
– ”
— ”” ˜™ ˜˜ š› š
œ š
 šš žŸ žž  ¡  
¢  
£    ¤¥ ¤¤ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬­ ¬
® ¬
¯ ¬¬ °± °° ²³ ²
´ ²
µ ²² ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å Ã
Æ ÃÃ ÇÈ ÇÇ ÉÊ É
Ë É
Ì ÉÉ ÍÎ ÍÍ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ ÓÓ ÕÖ Õ
× Õ
Ø ÕÕ ÙÚ ÙÙ ÛÜ Û
Ý Û
Þ ÛÛ ßà ßß áâ á
ã á
ä áá åæ åå çè ç
é ç
ê çç ëì ëë íî í
ï í
ð íí ñò ññ óô ó
õ ó
ö óó ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ýý ÿ€ ÿÿ ‚ 
ƒ  „… „
† „
‡ „„ ˆ‰ ˆˆ Š‹ Š
Œ Š
 ŠŠ Ž ŽŽ ‘ 
’ 
“  ”• ”” –— –
˜ –
™ –– š› šš œ œ
ž œ
Ÿ œœ  ¡    ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®
° ®
± ®® ²³ ²² ´µ ´
¶ ´
· ´´ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç Å
È ÅÅ ÉÊ ÉÉ ËÌ Ë
Í Ë
Î ËË ÏÐ ÏÏ ÑÒ Ñ
Ó Ñ
Ô ÑÑ ÕÖ ÕÕ ×Ø ×
Ù ×
Ú ×× ÛÜ ÛÛ ÝÞ Ý
ß Ý
à ÝÝ áâ áá ãä ã
å ã
æ ãã çè çç éê é
ë é
ì éé íî íí ïð ï
ñ ï
ò ïï óô óó õö õ
÷ õ
ø õõ ùú ù
û ùù üý ü
þ üü ÿ€ ÿÿ ‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž Ž
 Ž
‘ ŽŽ ’“ ’’ ”• ”
– ”
— ”” ˜™ ˜˜ š› š
œ š
 šš žŸ žž  ¡  
¢  
£    ¤¥ ¤¤ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬­ ¬
® ¬
¯ ¬¬ °± °° ²³ ²
´ ²
µ ²² ¶· ¶¶ ¸¹ ¸
º ¸
» ¸¸ ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ ØØ ÚÛ ÚÚ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ãä ãã åæ åå çè çç éê é
ë éé ìí ìì îï îî ðñ ðð òó òò ôõ ô
ö ôô ÷ø ÷÷ ùú ùù ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ Ž    ‘’ ‘‘ “” ““ •– •
— •• ˜™ ˜˜ š› šš œ œœ žŸ žž  ¡    ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­­ ¯° ¯¯ ±² ±± ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸¸ º» ºº ¼½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× ÙÚ ÙÙ ÛÜ ÛÛ ÝÞ ÝÝ ßà ßß áâ áá ãä ãã åæ å
ç åå èé èè êë êê ìí ìì îï îî ðñ ðð òó òò ôõ ôô ö÷ öö øù øø úû úú ü
ý üü þÿ þþ € €€ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡‡ ‰
Š ‰‰ ‹Œ ‹
 ‹
Ž ‹‹  
‘  ’“ ’’ ”• ”” –— –
˜ –
™ –– š› š
œ šš ž  Ÿ  Ÿ
¡ Ÿ
¢ ŸŸ £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±± ³
´ ³³ µ¶ µ
· µ
¸ µµ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ ÃÃ Å
Æ ÅÅ ÇÈ Ç
É Ç
Ê ÇÇ ËÌ Ë
Í ËË ÎÏ Î
Ð Î
Ñ ÎÎ ÒÓ Ò
Ô Ò
Õ ÒÒ Ö× Ö
Ø Ö
Ù ÖÖ ÚÛ ÚÚ ÜÝ Ü
Þ Ü
ß ÜÜ àá àà âã â
ä ââ å
æ åå çè ç
é ç
ê çç ëì ë
í ëë îï î
ð î
ñ îî òó ò
ô ò
õ òò ö÷ ö
ø ö
ù öö úû úú üý ü
þ ü
ÿ üü € €€ ‚ƒ ‚
„ ‚‚ …
† …… ‡ˆ ‡
‰ ‡
Š ‡‡ ‹Œ ‹
 ‹‹ Ž Ž
 Ž
‘ ŽŽ ’“ ’
” ’
• ’’ –— –
˜ –
™ –– š› šš œ œ
ž œ
Ÿ œœ  
¡    ¢£ ¢
¤ ¢¢ ¥
¦ ¥¥ §¨ §
© §
ª §§ «¬ «
­ «« ®¯ ®
° ®
± ®® ²³ ²
´ ²² µ¶ µ
· µ
¸ µµ ¹º ¹
» ¹¹ ¼
½ ¼¼ ¾¿ ¾
À ¾
Á ¾¾ ÂÃ Â
Ä ÂÂ Å
Æ ÅÅ ÇÈ Ç
É Ç
Ê ÇÇ ËÌ Ë
Í ËË ÎÏ Î
Ð Î
Ñ ÎÎ ÒÓ Ò
Ô Ò
Õ ÒÒ Ö× Ö
Ø Ö
Ù ÖÖ ÚÛ Ú
Ü ÚÚ Ý
Þ ÝÝ ßà ß
á ß
â ßß ãä ã
å ãã æç æ
è æ
é ææ êë ê
ì ê
í êê îï î
ð î
ñ îî ò
ó òò ôõ ô
ö ôô ÷
ø ÷÷ ùú ù
û ù
ü ùù ýþ ý
ÿ ýý € €
‚ €
ƒ €€ „… „
† „„ ‡
ˆ ‡‡ ‰Š ‰
‹ ‰
Œ ‰‰ Ž 
  
‘  ’“ ’
” ’
• ’’ –— –
˜ –– ™š ™
› ™
œ ™™ ž 
Ÿ 
   ¡
¢ ¡¡ £¤ £
¥ ££ ¦
§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬
® ¬¬ ¯
° ¯¯ ±² ±
³ ±
´ ±± µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »
¼ »» ½¾ ½
¿ ½
À ½½ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ Ç
È ÇÇ ÉÊ É
Ë É
Ì ÉÉ Í
Î ÍÍ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ Ù
Ú ÙÙ ÛÜ Û
Ý Û
Þ ÛÛ ß
à ßß áâ á
ã á
ä áá å
æ åå çè ç
é ç
ê çç ëì ë
í ëë îï î
ð îî ñ
ò ññ óô ó
õ ó
ö óó ÷
ø ÷÷ ùú ù
û ù
ü ùù ý
þ ýý ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ
„ ƒƒ …† …
‡ …
ˆ …… ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ  
‘ 
’ 
“  ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
ž œ
Ÿ œ
  œœ ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©
¬ ©
­ ©© ®¯ ®® °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶
¹ ¶
º ¶¶ »¼ »» ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å Ã
Æ Ã
Ç ÃÃ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ Ð
Ò ÑÑ Ó
Ô ÓÓ Õ
Ö ÕÕ ×
Ø ×× Ù
Ú ÙÙ Û
Ü ÛÛ ÝÞ ]ß [à Qá /â  ã Hä \å @æ Zç )  
            "! $ %# ') +* -( ./ 10 3  42 6) 75 9 :8 <( =; ?@ B& CA E, GH JF K> MI OL PQ S> TR VN WU Y# _^ a, cb e; gf i] k` ld mh nj po ro so uq v xw z |{ ~ € ‚ „ƒ † ˆ‡ Šq Œ[ Ž` d h ‘ “‹ •’ –” ˜ š— œ™ o Ÿž ¡  £ ¥¢ §¤ ¨ ª© ¬ ®­ ° ²± ´[ ¶` ·d ¸h ¹µ »‹ ½º ¾¼ À Â¿ ÄÁ Å ÇÆ É Ë¢ ÍÊ Î ÐÏ Ò ÔÓ Ö[ Ø` Ùd Úh Û× Ý‹ ßÜ àÞ â äá æã ç éè ë íì ïž ñð ó õò ÷ô ø úù ü’ þ’ ÿº º ‚€ „ý †ƒ ‡Ü ‰Ü Šˆ Œ… q [ ‘` ’d “h ” –Ž ˜• ™‹ ›t œ— š Ÿ ¡ž £  ¤q ¦¥ ¨’ © «§ ­ª ®¥ °º ± ³¯ µ² ¶¥ ¸Ü ¹ »· ½º ¾o À¿ Â ÄÁ ÆÃ Çf ÉÈ Ë] Í` Îd ÏÊ ÐÌ ÒÑ ÔÑ ÕÑ ×Ó Ø ÚÙ Ü ÞÝ à âá ä æå è êé ì[ î` ïd ðÊ ñí óÑ õò öô ø\ ú` ûd üÊ ýù ÿþ Ñ ƒ€ „÷ †ô ‡‚ ˆÓ Š‰ Œò ‹ Ž ‘… “ ” –’ ˜• ™ô ›Ñ œ Ÿž ¡š £  ¤¢ ¦ ¨¥ ª§ «[ ­` ®d ¯Ê °¬ ²Ñ ´± µ³ ·¶ ¹ »¸ ½º ¾[ À` Ád ÂÊ Ã¿ ÅÑ ÇÄ ÈÆ ÊÉ Ì ÎË ÐÍ Ñ ÓÒ Õò ×± ØÓ ÚÖ ÛÙ ÝÓ ßÞ á± âà äã æÜ èå é ëç íê î³ ð òï ôñ õÑ ÷ö ùø ûô ýú þü € ‚ÿ „ … ‡† ‰ ‹Š ò Ä Ó ’Ž “‘ •Þ —Ä ˜– š™ œ” ž› Ÿ ¡ £  ¤Æ ¦ ¨¥ ª§ « ­¬ ¯ ±ÿ ³° ´ ¶µ ¸[ º` »d ¼Ê ½¹ ¿¾ ÁÀ Ãþ ÅÂ ÆÓ Èò ÉÇ ËÄ ÌÖ Îò Ðò ÑÖ Ó± Õ± ÖÒ ØÔ Ù× ÛÍ ÝÏ ÞÚ ßÄ áÄ âÒ äã æà çÜ èÓ êé ìë î¾ ïå ðí òñ ôÊ öó ÷ ùõ ûø üÑ þ¾ ÿÑ þ ‚Ï „Ó …€ †ƒ ˆ‡ Šý Œ‰ Ó Ž ‘ò ’ ”‹ –“ — ™• ›˜ œÖ žÓ   ¡Ó £¢ ¥± ¦¤ ¨Ÿ ª§ « ­© ¯¬ °Ž ²Ó ´± µ¢ ·Ä ¸¶ º³ ¼¹ ½ ¿» Á¾ Âô ÄÑ ÆÅ ÈÃ ÊÇ ËÉ Í ÏÌ ÑÎ Òb ÔÓ Ö] Ø` ÙÕ Úh Û× ÝÜ ßÜ àÜ âÞ ã åä ç éè ë íì ï ñð ó õô ÷[ ù` úÕ ûh üø þ[ €` Õ ‚h ƒÿ …ý ‡„ ˆÞ Š† ‹‰ Þ Ž ‘ý ’ ”“ –Œ ˜• ™ ›— š žÜ  „ ¡Ü £¢ ¥¤ §Ÿ ©¦ ª¨ ¬ ®« °­ ±Ü ³ý ´² ¶ ¸µ º· » ½¼ ¿ ÁÀ ÃŸ Å\ Ç` ÈÕ Éh ÊÆ ÌÜ ÎË ÏÍ ÑÄ ÓŸ ÔÐ ÕÞ ×Ö Ù„ ÚØ ÜÛ ÞÒ àÝ á ãß åâ æ² èç ê ìé îë ïŸ ñÜ óò õô ÷ð ùö úø ü þû €ý [ ƒ` „Õ …h †‚ ˆÜ Š‡ ‹‰ Œ  ‘Ž “ ” –• ˜„ š‡ ›Þ ™ žœ  Ž ¢‡ £¡ ¥¤ §Ÿ ©¦ ª ¬¨ ®« ¯ ±° ³‰ µ ·´ ¹¶ º ¼« ¾» ¿ ÁÀ Ã[ Å` ÆÕ Çh ÈÄ ÊÉ ÌË ÎË ÐÍ ÑÞ Ó„ ÔÒ ÖÏ ×á Ùý Ûý Üá Þ„ à„ áÝ ãß äâ æØ èÚ éå ê‡ ì‡ íØ ïë ðç ñÞ óò õô ÷É øî ùö ûú ýÕ ÿü € ‚þ „ …† ‡Þ ‰† ŠÞ Œ‹ Žý  ‘ˆ “ ” –’ ˜• ™Ü ›É œÞ žß ŸË ¡Ü ¢ £  ¥¤ §š ©¦ ªÞ ¬« ®„ ¯­ ±¨ ³° ´ ¶² ¸µ ¹™ »Þ ½º ¾‹ À‡ Á¿ Ã¼ ÅÂ Æ ÈÄ ÊÇ ËŸ ÍÜ ÏÎ ÑÌ ÓÐ ÔÒ Ö ØÕ Ú× Û^ ÝÜ ß] áÞ âd ãh äà æå èå éå ëç ì îí ð òñ ô öõ ø úù ü þý €	[ ‚	Þ ƒ	d „	h …		 ‡	[ ‰	Þ Š	d ‹	h Œ	ˆ	 Ž	†	 		 ‘	ç “		 ”	’	 –	ç ˜	—	 š	†	 ›	™	 	œ	 Ÿ	•	 ¡	ž	 ¢	 ¤	 	 ¦	£	 §	å ©		 ª	å ¬	«	 ®	¨	 °	­	 ±	¯	 ³	 µ	²	 ·	´	 ¸	 º	¹	 ¼	å ¾	†	 ¿	½	 Á	 Ã	À	 Å	Â	 Æ	 È	Ç	 Ê	[ Ì	Þ Í	d Î	h Ï	Ë	 Ñ		 Ó	Ð	 Ô	ç Ö	Ò	 ×	Õ	 Ù	—	 Û	Ð	 Ü	Ú	 Þ	Ý	 à	Ø	 â	ß	 ã	 å	á	 ç	ä	 è	 ê	é	 ì	å î	í	 ð	ï	 ò	¨	 ô	ñ	 õ	ó	 ÷	 ù	ö	 û	ø	 ü	å þ	Ð	 ÿ	ý	 
 ƒ
€
 …
‚
 †
 ˆ
‡
 Š
¨	 Œ
\ Ž
Þ 
d 
h ‘

 “
å •
’
 –
”
 ˜
‹
 š
¨	 ›
—
 œ
ç ž

  
	 ¡
Ÿ
 £
¢
 ¥
™
 §
¤
 ¨
 ª
¦
 ¬
©
 ­
½	 ¯
®
 ±
 ³
°
 µ
²
 ¶
ý	 ¸
·
 º
 ¼
¹
 ¾
»
 ¿
å Á
À
 Ã
Â
 Å
¨	 Ç
Ä
 È
Æ
 Ê
 Ì
É
 Î
Ë
 Ï
 Ñ
Ð
 Ó
[ Õ
Þ Ö
d ×
h Ø
Ô
 Ú
Ù
 Ü
Û
 Þ
’
 à
Ý
 á
ç ã
	 ä
â
 æ
ß
 ç
ê é
†	 ë
†	 ì
ê î
Ð	 ð
Ð	 ñ
í
 ó
ï
 ô
ò
 ö
è
 ø
ê
 ù
õ
 ú
ê ü
	 þ
	 ÿ
û
 ý
 ‚÷
 ƒç …„ ‡† ‰Ù
 Š€ ‹ˆ Œ å
 ‘Ž ’ ” –“ —	 ™ç ›˜ œç ž  †	 ¡Ÿ £š ¥¢ ¦ ¨¤ ª§ «Ò	 ­ç ¯¬ ° ²Ð	 ³± µ® ·´ ¸ º¶ ¼¹ ½å ¿Ù
 Àç Âý
 Ã’
 Åå ÆÁ ÇÄ ÉÈ Ë¾ ÍÊ Îç ÐÏ Ò	 ÓÑ ÕÌ ×Ô Ø ÚÖ ÜÙ Ý¨	 ßå áà ãÞ åâ æä è êç ìé íZ ïÞ ðd ñh òî ôZ öÞ ÷d øh ùõ ûZ ýÞ þd ÿh €ü ‚Z „Þ …d †h ‡ƒ ‰Z ‹Þ Œd h ŽŠ í ’ñ ”“ –ú —‘ ™ó š• ›õ œ Ÿ  ˜ ¡ù £¢ ¥ˆ ¦ž §ý ©¨ « ¬¤ ­ª ¯ ±£	 ³´	 µ´ ·ú ¸² ºó »¶ ¼¹	 ¾½ À Á¹ ÂÂ	 ÄÃ Æˆ Ç¿ ÈÇ	 ÊÉ Ì ÍÅ ÎË Ð Òä	 Ôé	 ÖÕ Øú ÙÓ Ûó Ü× Ýø	 ßÞ á âÚ ã‚
 åä çˆ èà é‡
 ëê í îæ ïì ñ óð õò ö©
 ø²
 úù üú ý÷ ÿó €û »
 ƒ‚ … †þ ‡Ë
 ‰ˆ ‹ˆ Œ„ Ð
 Ž ‘ ’Š “ • —” ™– š¤ œú  Ÿó  › ¡¶ £ ¤ž ¥Ö §ˆ ¨¢ ©ç « ¬¦ ­ª ¯ ±® ³° ´Z ¶` ·Õ ¸h ¹µ »Z ½` ¾d ¿Ê À¼ ÂZ Ä` ÅÕ Æh ÇÃ ÉZ Ë` Ìd ÍÊ ÎÊ ÐZ Ò` ÓÕ Ôh ÕÑ ×Z Ù` Úd ÛÊ ÜØ ÞZ à` áÕ âh ãß åZ ç` èd éÊ êæ ìZ î` ïÕ ðh ñí óZ õ` öd ÷Ê øô úä üÙ þý €Á û ƒº „ÿ …è ‡† ‰È Š‚ ‹Ý Œ Ï ˆ ‘ì “’ •Ö –Ž —á ™˜ ›Ý œ” ð Ÿž ¡ä ¢š £å ¥¤ §ë ¨  ©ô «ª ­ò ®¦ ¯é ±° ³ù ´¬ µ² ·® ¸¶ º° »š ½• ¿¾ ÁÁ Â¼ Äº ÅÀ Æ­ ÈÇ ÊÈ ËÃ Ì§ ÎÍ ÐÏ ÑÉ Ò· ÔÓ ÖÖ ×Ï Øº ÚÙ ÜÝ ÝÕ Þ¼ àß âä ãÛ äÍ æå èë éá êÀ ìë îò ïç ðÒ òñ ôù õí öó øÏ ù÷ ûÑ üâ þê €ÿ ‚Á ƒý …º † ‡ë ‰ˆ ‹È Œ„ ñ Ž ‘Ï ’Š “ý •” —Ö ˜ ™ ›š Ý ž– Ÿ ¡  £ä ¤œ ¥† §¦ ©ë ª¢ «• ­¬ ¯ò °¨ ±Š ³² µù ¶® ·´ ¹ð º¸ ¼ò ½« ¿  ÁÀ ÃÁ Ä¾ Æº ÇÂ È° ÊÉ ÌÈ ÍÅ Î§ ÐÏ ÒÏ ÓË Ô¶ ÖÕ ØÖ ÙÑ Ú¬ ÜÛ ÞÝ ß× à» âá ää åÝ æ° èç êë ëã ìÀ îí ðò ñé òµ ôó öù ÷ï øõ ú” ûù ý– þ° € ‚ø „ƒ †Á ‡ ‰º Š… ‹• Œ È ˆ ‘˜ “’ •Ï –Ž —µ ™˜ ›Ö œ” ¬ Ÿž ¡Ý ¢š £Ç ¥¤ §ä ¨  ©¾ «ª ­ë ®¦ ¯× ±° ³ò ´¬ µÎ ·¶ ¹ù º² »¸ ½ÿ ¾¼ À° Á ÃÂ Å ÇÄ ÉÆ Ê{ ÌË Î ÐÏ ÒÍ ÔÑ Õ ×Ö Ù ÛÚ ÝØ ßÜ àƒ âá ä æå èã êç ë‡ íì ï ñð óî õò ö™ ø÷ ú üû þù €ý ¤ ƒ‚ … ‡† ‰„ ‹ˆ Œ© Ž  ’‘ ” –“ —­ ™ ›±  ŸÁ ¡  £ ¥¤ §¢ ©¦ ªÆ ¬« ® °¯ ²­ ´± µÊ · ¹Ï » ½Ó ¿ Áã ÃÂ Å ÇÆ ÉÄ ËÈ Ìè Î Ðì Ò Ôô Ö Øù Ú Ü  ÞÝ à âá äß æã çª é ë² í ïº ñ óÃ õ ÷ ùø ûú ý ÿþ ü ƒ€ „„ †Ï ˆ‚ Š‰ Œ‡ … Ž‹ † ‘ “Ú •‰ —” ˜’ ™– ›‘ œå ž‰   ¡˜ ¢Ÿ ¤š ¥ð §‰ ©¦ ªœ «¨ ­ž ®Ñ °° ²± ´³ ¶‚ ·¯ ¸µ ºÑ » ½¼ ¿ü Á¾ Â­ ÄÀ ÆÅ È‡ ÉÃ ÊÇ Ì¯ ÍÅ Ï” Ð¶ ÑÅ Ó Ôº ÕÅ ×¦ Ø¾ Ùò Û³ ÝÀ ÞÚ ßÄ áü ãà äâ æå è‡ éÍ êç ìÏ íå ï” ðÑ ñå ó ôÕ õå ÷¦ øÙ ù– û³ ýâ þú ÿß ü ƒ€ „‚ †… ˆ‡ ‰è Š‡ Œê … ” ì ‘… “ ”ð •… —¦ ˜ô ™° ›³ ‚ žš Ÿ‹ ¡  £Ç ¤¢ ¦¥ ¨– ©Î ª§ ¬¸ ­¥ ¯Ÿ °Ò ±® ³¼ ´¥ ¶¨ ·Ö ¸µ ºÀ »µ ½¼ ¿¢ ÀÜ Á  Ãç ÄÂ ÆÅ È– Éî ÊÇ ÌÓ ÍÅ ÏŸ Ðò ÑÅ Ó¨ Ôö Õ¼ ×Â Øü Ù  Û‡ ÜÚ ÞÝ à– áŽ âß äî åÝ çŸ è’ éÝ ë¨ ì– í¼ ïÚ ðœ ñ§ óò õÇ öô ø÷ ú® ûÎ üù þ× ÿ÷ µ ‚Ò ƒ€ …Û †¾ ˆ‡ Šô ‹Ö Œò Žß  ‘ “® ”æ •’ —ò ˜ šµ ›ê œ‡ ž Ÿî  ù ¢¡ ¤’ ¥£ §¦ ©€ ª™ «¨ ­ö ®‰ °¯ ²£ ³ ´± ¶¨ ·µ ¹° º€ ¼» ¾µ ¿‰ À½ Âù ÃÁ Å– Æ® ÈÇ ÊÁ Ë¾ Ìµ ÎÍ Ðµ ÑÉ ÒÏ Ô§ ÕÓ ×ò Ø– ÚÙ ÜÓ Ýµ ÞŸ àß âÁ ãÛ ä¨ æå èµ éá êç ì‹ íë ïÑ ð‡ òñ ôë õ± ö” ø÷ úÓ ûó ü þý €Á ù ‚¦ „ƒ †µ ‡ÿ ˆ… Šú ‹‰ ° ŽZ ` ‘d ’h “ •” —‰ ˜– š ›Z ` žd Ÿh  œ ¢¡ ¤ë ¥£ §œ ¨Z ª` «d ¬h ­© ¯® ±Ó ²° ´© µZ ·` ¸d ¹h º¶ ¼» ¾Á ¿½ Á¶ ÂZ Ä` Åd Æh ÇÃ ÉÈ Ëµ ÌÊ ÎÃ Ï Ò Ô Ö Ø Ú ÜD FD ÑX ZX ÑÐ Ñ ëë éé Ý êê èè• êê •— êê —á	 êê á	á êê áå êê åË êê Ëˆ êê ˆà êê àŽ êê Ž² êê ²¨ êê ¨ó êê óˆ êê ˆÝ êê ÝÇ êê Ç¢ êê ¢ò êê ò® êê ® êê ç êê çÌ êê Ì¿ êê ¿¸ êê ¸õ êê õÎ êê Î¾ êê ¾µ êê µ¹ êê ¹î êê îÒ êê Ò¶ êê ¶É êê É¯	 êê ¯	ß êê ßç êê çÜ êê Üù êê ù… êê …Ó ëë Ó’ êê ’é êê éí êê í¤ êê ¤ã êê ã¬ êê ¬– êê –¬ êê ¬× êê ×ü êê ü‹ êê ‹‡ êê ‡æ êê æ¨ êê ¨œ êê œ˜ êê ˜þ êê þÉ êê ÉÃ êê Ãž êê žù êê ùš êê šË êê Ë  êê  ¦ êê ¦¸ êê ¸µ êê µ¦ êê ¦„ êê „Ö êê ÖÇ êê Çœ êê œß êê ßê êê ê‰ êê ‰™ êê ™ª êê ª  êê  ç êê ç’ êê ’€ êê €¼ êê ¼– êê –® êê ®î êê î êê  êê á êê á éé Ö êê ÖÕ ëë Õ” êê ”Ü êê Ü èè … êê …Ÿ êê Ÿ¿ êê ¿Î êê Îª êê ªÿ êê ÿ‚ êê ‚ êê ” êê ”Ï êê Ï( éé (² êê ²Õ êê Õð êê ðþ êê þÄ êê ÄÉ êê Éç êê çó	 êê ó	Û êê Ûü êê ü èè ’ êê ’¦
 êê ¦
Ö êê Ö¦ êê ¦ èè ¨ êê ¨ä êê ä	 èè 	© êê ©Š êê ŠŽ êê Žî êê îæ êê æˆ êê ˆ€ êê €Ù ëë ÙÏ êê ÏÛ ëë Ûç êê çÏ êê Ï¶ êê ¶í êê íŽ êê ŽÄ êê Äõ êê õÒ êê ÒÆ
 êê Æ
ö êê ö‹ êê ‹Ú êê Úƒ êê ƒ êê ï êê ï´ êê ´¢ êê ¢» êê »¨ êê ¨‹ êê ‹Ò êê Ò– êê –× ëë ×ó êê óÑ ëë Ñš êê š„ êê „Å êê ÅÛ êê Û¤ êê ¤ èè ø êê ø èè ² êê ²š êê š’ êê ’ö êê ö÷ êê ÷ 	 êê  	ß
 êê ß
… êê …¨ êê ¨Å êê ÅÑ êê Ñ¢ êê ¢  êê  ¨ êê ¨± êê ±÷
 êê ÷
½ êê ½Ä êê Äì êê ìŠ êê Š  êê  ž êê žù êê ù™
 êê ™
Ò êê Ò§ êê §
ì Ý
í žî î î î î î î 	ï @	ï H	ï Q
ð ‰
ð Ö
ð 
	ñ !	ñ *	ñ 0	ñ {
ñ 
ñ ™
ñ ¤
ñ ¤
ñ ©
ñ ­
ñ ±
ñ Æ
ñ è
ñ ª
ñ Ý
ñ í
ñ •
ñ §
ñ §
ñ º
ñ Í
ñ Ò
ñ ñ
ñ §
ñ ˜
ñ è
ñ ø
ñ š
ñ ­
ñ ­
ñ ·
ñ ¼
ñ À
ñ ë
ñ °
ñ •
ñ ñ
ñ 	
ñ £	
ñ ´	
ñ ´	
ñ ¹	
ñ Â	
ñ Ç	
ñ é	
ñ ²

ñ §
ñ õ
ñ Ñ
ñ Ã
ñ Ê
ñ Ï
ñ û
ñ †
ñ †
ñ ‘
ñ š
ñ ž
ñ ¯
ñ Ï
ñ ê
ñ þ
ñ œ
ò Ø
ò è

ó š
ó ð
ô Æ
õ }õ õ …õ ‰õ «õ ¯õ ³õ Èõ Ñõ Õõ êõ îõ ûõ ãõ çõ ëõ ˆõ Œõ ®õ ·õ êõ òõ öõ ¾õ Âõ ²õ Âõ óõ ÷õ ÿõ »	õ É	õ ë	õ ‰

ö Ž÷ ÷ Ñø y
ù  
ú ƒ
ú …
ú ‹
û á
ü Ž
ü ž
ü ã
ü ø
ü ™
ü ñ
ü “
ü ¤
ü Û
ü ô
ü ¤
ü ú
ü œ	
ü Ý	
ü ï	
ü ¢

ü Â

ü Œ
ý Þ
ý Ž
ý —	
þ ¢
þ ò
þ Á
ÿ ¶
ÿ É
ÿ 
ÿ ±
ÿ ç
ÿ Œ
ÿ †
ÿ º
ÿ ®

ÿ ·

ÿ ˜
ÿ ¬
€ Å
€ Î
€ à
 Í
 û

‚ À
‚ ‹
‚ Ã
‚ Ë
‚ ¨
‚ Ì
‚ Û

‚ Ì
‚ Þ
ƒ ƒ
ƒ ­
ƒ Ï
ƒ ×
ƒ ã
ƒ è
ƒ ì
ƒ ô
ƒ ô
ƒ ù
ƒ º
ƒ å
ƒ ¿
ƒ Í
ƒ †
ƒ  
ƒ §
ƒ ¬
ƒ °
ƒ °
ƒ µ
ƒ ¾
ƒ ð
ƒ ¼
ƒ ‚
ƒ 
ƒ «
ƒ °
ƒ ¶
ƒ »
ƒ »
ƒ À
ƒ Ç
ƒ ù
ƒ ˆ	
ƒ Â	
ƒ ‚

ƒ ©

ƒ ²

ƒ »

ƒ Ë

ƒ Ë

ƒ Ð

ƒ Ù
ƒ ƒ
ƒ –
ƒ ß
ƒ æ
ƒ å
ƒ š
ƒ ¼
ƒ Æ
ƒ Ï
ƒ Ó
ƒ ×
ƒ ×
ƒ Û
ƒ ò
ƒ ¶„ Û
„ ¥
„ ÿ
„ Ì„ æ
„ «
„ û
„ Õ	… 
… ©
… µ
… Á
… Æ
… Ê
… Ê
… Ï
… Ó
… ì
… ²
… á
… ¬
… º
… ê
… ñ
… 
… 
… †
… Š
… ¬
… ¬
… ì
… ÿ
… ·
… â
… ë
… ý
… ý
… 
… •
… ¶
… µ
… õ
… ¹	
… Ë	
… ä	
… é	
… ø	
… ø	
… ‚

… ‡

… »

… ¹
… ü
… ò
… Ñ
… Ø
… Ú
… ‘
… ¤
… ¯
… ¸
… ¸
… ¼
… À
… Ó
… î
… ¼
… ©
† é
† ò
† „
‡ Ž
‡ «
‡ Ï
ˆ Ò
ˆ í

‰ «	Š ÷Š Š  Š ÜŠ åŠ úŠ ”Š ›Š ÂŠ ÚŠ ãŠ ëŠ óŠ ‰Š “Š §Š ¹Š ÇŠ ŒŠ •Š ¦Š ÄŠ ÝŠ öŠ ŸŠ ¦Š ÍŠ åŠ ôŠ üŠ Š ¦Š °Š ÂŠ ÐŠ •	Š ž	Š ­	Š Ø	Š ß	Š ñ	Š ‹
Š ¤
Š Ä
Š Ý
Š õ
Š †Š ŽŠ ¢Š ´Š ÊŠ ÔŠ âŠ ‰Š ³Š ÅŠ åŠ …Š ¥Š ¼Š ÅŠ ÝŠ ÷Š ‡Š Š ¦Š ¯Š »Š ÇŠ ÍŠ ÙŠ ßŠ åŠ ñŠ ÷Š ýŠ ƒ
‹ ‹
‹ ö
‹ ¢
‹ í	
Œ ‡
Œ ±
Œ Ó
Œ ù
Œ 
Œ  
Œ ª
Œ ²
Œ º
Œ Ã
Œ Ã
Œ é
Œ Ò
Œ Š
Œ µ
Œ ¹
Œ ø
Œ ˜
Œ ¬
Œ ¾
Œ Î
Œ Î
Œ ô
Œ À
Œ •
Œ À
Œ Ä
Œ 
Œ •
Œ µ
Œ Ç
Œ ×
Œ ×
Œ ý
Œ Ç	
Œ ‡

Œ Ð

Œ Ô

Œ “
Œ §
Œ ¹
Œ Ù
Œ é
Œ é
Œ Š
Œ °
Œ í
Œ ô
Œ ð
Œ ž
Œ À
Œ Û
Œ á
Œ ê
Œ î
Œ ò
Œ ö
Œ ö
Œ Ã
 ¥Ž (	Ž L
 —
 ¿ Ô — Ò

‘ ð
’ œ
’ ò
’ À
	“ w	“ w	“ w	“ {	“ {	“ 	“ 
“ ƒ
“ ƒ
“ ‡
“ ‡
“ ™
“ ™
“ ¤
“ ©
“ ­
“ ±
“ Á
“ Á
“ Æ
“ Ê
“ Ï
“ Ó
“ ã
“ ã
“ è
“ ì
“ ô
“ ù
“  
“  
“ ª
“ ²
“ º
“ Ã
“ Ù
“ Ù
“ Ù
“ Ý
“ Ý
“ á
“ á
“ å
“ å
“ é
“ é
“ •
“ •
“ §
“ º
“ Í
“ Ò
“ ê
“ ê
“ ñ
“ 
“ †
“ Š
“  
“  
“ §
“ ¬
“ °
“ µ
“ ø
“ ø
“ ˜
“ ¬
“ ¾
“ Î
“ ä
“ ä
“ ä
“ è
“ è
“ ì
“ ì
“ ð
“ ð
“ ô
“ ô
“ š
“ š
“ ­
“ ·
“ ¼
“ À
“ â
“ â
“ ë
“ ý
“ 
“ •
“ «
“ «
“ °
“ ¶
“ »
“ À
“ 
“ 
“ •
“ µ
“ Ç
“ ×
“ í
“ í
“ í
“ ñ
“ ñ
“ õ
“ õ
“ ù
“ ù
“ ý
“ ý
“ £	
“ £	
“ ´	
“ ¹	
“ Â	
“ Ç	
“ ä	
“ ä	
“ é	
“ ø	
“ ‚

“ ‡

“ ©

“ ©

“ ²

“ »

“ Ë

“ Ð

“ “
“ “
“ §
“ ¹
“ Ù
“ é
“ î
“ °
“ °
“ Ñ
“ ò
“ –
“ °
“ µ
“ ¼
“ Ï
“ Ï
“ Ú
“ Ú
“ å
“ å
“ ð
“ ð
“ û
“ †
“ ‘
“ š
“ ž
“ ¤
“ ¯
“ ¸
“ ¼
“ À
“ Æ
“ Ï
“ Ó
“ ×
“ Û
“ á
“ ê
“ î
“ ò
“ ö
“ ø
“ ø
“ ø
“ þ
“ þ
“ ¼
“ ¼
“ 
” ¢
” ‹
” 	• ^	• `	• b	• d	• f	• h
• Ê
• Õ
• Þ
–  
– ð
– ¿– ü–  – ò– ¡
— ®
— Ï
— ð
— ”
— ®
— ¶
— ÷
— ¸
— ù
— ¼
˜ ž
˜ €
˜ ‡
˜ Ð
˜ ¤
˜ —

˜ È™ ï
™ ²	
™ ö	
™ É

™ ç
š ¿› 	› › › › › Ó› Õ› ×› Ù› Ûœ ß
œ ’
œ ¢
œ ¸
œ Ë
œ ç
œ ï
œ ü
œ 
œ ¥
œ õ
œ •
œ ©
œ »
œ Éœ î
œ —
œ ¨
œ µ
œ ß
œ é
œ ø
œ Ž
œ ¨
œ ´
œ þ
œ ’
œ ²
œ Ä
œ Òœ û
œ  	
œ ¯	
œ À	
œ á	
œ ó	
œ €

œ ¦

œ °

œ ¹

œ 
œ ¤
œ ¶
œ Ö
œ ä
 È
 Ó
 Ü
ž Ä
ž Ï
ž ß
"
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
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282€

wgsize_log1p
½aA

transfer_bytes
¨ÿÈ

devmap_label
 

wgsize
5
 
transfer_bytes_log1p
½aA