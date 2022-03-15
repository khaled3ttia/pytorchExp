

[external]
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 1) #3
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
1%17 = tail call i64 @_Z13get_global_idj(i32 0) #3
.addB'
%
	full_text

%18 = add i64 %17, 1
#i64B

	full_text
	
i64 %17
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
6icmpB.
,
	full_text

%20 = icmp sgt i32 %16, %11
#i32B

	full_text
	
i32 %16
5icmpB-
+
	full_text

%21 = icmp sgt i32 %19, %9
#i32B

	full_text
	
i32 %19
-orB'
%
	full_text

%22 = or i1 %20, %21
!i1B

	full_text


i1 %20
!i1B

	full_text


i1 %21
9brB3
1
	full_text$
"
 br i1 %22, label %737, label %23
!i1B

	full_text


i1 %22
Sbitcast8BF
D
	full_text7
5
3%24 = bitcast double* %0 to [103 x [103 x double]]*
Sbitcast8BF
D
	full_text7
5
3%25 = bitcast double* %2 to [103 x [103 x double]]*
5add8B,
*
	full_text

%26 = add nsw i32 %16, -1
%i328B

	full_text
	
i32 %16
5mul8B,
*
	full_text

%27 = mul nsw i32 %26, %9
%i328B

	full_text
	
i32 %26
5add8B,
*
	full_text

%28 = add nsw i32 %19, -1
%i328B

	full_text
	
i32 %19
6add8B-
+
	full_text

%29 = add nsw i32 %28, %27
%i328B

	full_text
	
i32 %28
%i328B

	full_text
	
i32 %27
6mul8B-
+
	full_text

%30 = mul nsw i32 %29, 102
%i328B

	full_text
	
i32 %29
6sext8B,
*
	full_text

%31 = sext i32 %30 to i64
%i328B

	full_text
	
i32 %30
^getelementptr8BK
I
	full_text<
:
8%32 = getelementptr inbounds double, double* %4, i64 %31
%i648B

	full_text
	
i64 %31
2mul8B)
'
	full_text

%33 = mul i32 %29, 515
%i328B

	full_text
	
i32 %29
6sext8B,
*
	full_text

%34 = sext i32 %33 to i64
%i328B

	full_text
	
i32 %33
^getelementptr8BK
I
	full_text<
:
8%35 = getelementptr inbounds double, double* %6, i64 %34
%i648B

	full_text
	
i64 %34
Jbitcast8B=
;
	full_text.
,
*%36 = bitcast double* %35 to [5 x double]*
-double*8B

	full_text

double* %35
^getelementptr8BK
I
	full_text<
:
8%37 = getelementptr inbounds double, double* %7, i64 %34
%i648B

	full_text
	
i64 %34
Jbitcast8B=
;
	full_text.
,
*%38 = bitcast double* %37 to [5 x double]*
-double*8B

	full_text

double* %37
^getelementptr8BK
I
	full_text<
:
8%39 = getelementptr inbounds double, double* %8, i64 %34
%i648B

	full_text
	
i64 %34
Jbitcast8B=
;
	full_text.
,
*%40 = bitcast double* %39 to [5 x double]*
-double*8B

	full_text

double* %39
4add8B+
)
	full_text

%41 = add nsw i32 %10, 1
6sext8B,
*
	full_text

%42 = sext i32 %41 to i64
%i328B

	full_text
	
i32 %41
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %37, align 8, !tbaa !8
-double*8B

	full_text

double* %37
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
rgetelementptr8B_
]
	full_textP
N
L%43 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %42, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
rgetelementptr8B_
]
	full_textP
N
L%44 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %42, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %44, align 8, !tbaa !8
-double*8B

	full_text

double* %44
rgetelementptr8B_
]
	full_textP
N
L%45 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %42, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %45, align 8, !tbaa !8
-double*8B

	full_text

double* %45
]getelementptr8BJ
H
	full_text;
9
7%46 = getelementptr inbounds double, double* %35, i64 1
-double*8B

	full_text

double* %35
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
]getelementptr8BJ
H
	full_text;
9
7%47 = getelementptr inbounds double, double* %37, i64 1
-double*8B

	full_text

double* %37
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
]getelementptr8BJ
H
	full_text;
9
7%48 = getelementptr inbounds double, double* %39, i64 1
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
rgetelementptr8B_
]
	full_textP
N
L%49 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %42, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
rgetelementptr8B_
]
	full_textP
N
L%50 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %42, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
rgetelementptr8B_
]
	full_textP
N
L%51 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %42, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
]getelementptr8BJ
H
	full_text;
9
7%52 = getelementptr inbounds double, double* %35, i64 2
-double*8B

	full_text

double* %35
]getelementptr8BJ
H
	full_text;
9
7%53 = getelementptr inbounds double, double* %37, i64 2
-double*8B

	full_text

double* %37
]getelementptr8BJ
H
	full_text;
9
7%54 = getelementptr inbounds double, double* %39, i64 2
-double*8B

	full_text

double* %39
rgetelementptr8B_
]
	full_textP
N
L%55 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %42, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
%i648B

	full_text
	
i64 %42
rgetelementptr8B_
]
	full_textP
N
L%56 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %42, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
%i648B

	full_text
	
i64 %42
rgetelementptr8B_
]
	full_textP
N
L%57 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %42, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %42
]getelementptr8BJ
H
	full_text;
9
7%58 = getelementptr inbounds double, double* %35, i64 3
-double*8B

	full_text

double* %35
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
]getelementptr8BJ
H
	full_text;
9
7%59 = getelementptr inbounds double, double* %37, i64 3
-double*8B

	full_text

double* %37
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %59, align 8, !tbaa !8
-double*8B

	full_text

double* %59
]getelementptr8BJ
H
	full_text;
9
7%60 = getelementptr inbounds double, double* %39, i64 3
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
rgetelementptr8B_
]
	full_textP
N
L%61 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %42, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
rgetelementptr8B_
]
	full_textP
N
L%62 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %42, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
rgetelementptr8B_
]
	full_textP
N
L%63 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %42, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
]getelementptr8BJ
H
	full_text;
9
7%64 = getelementptr inbounds double, double* %35, i64 4
-double*8B

	full_text

double* %35
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
]getelementptr8BJ
H
	full_text;
9
7%65 = getelementptr inbounds double, double* %37, i64 4
-double*8B

	full_text

double* %37
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %65, align 8, !tbaa !8
-double*8B

	full_text

double* %65
]getelementptr8BJ
H
	full_text;
9
7%66 = getelementptr inbounds double, double* %39, i64 4
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %66, align 8, !tbaa !8
-double*8B

	full_text

double* %66
rgetelementptr8B_
]
	full_textP
N
L%67 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %42, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %67, align 8, !tbaa !8
-double*8B

	full_text

double* %67
rgetelementptr8B_
]
	full_textP
N
L%68 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %42, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
rgetelementptr8B_
]
	full_textP
N
L%69 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %42, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %42
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %69, align 8, !tbaa !8
-double*8B

	full_text

double* %69
Sbitcast8BF
D
	full_text7
5
3%70 = bitcast double* %1 to [103 x [103 x double]]*
Ybitcast8BL
J
	full_text=
;
9%71 = bitcast double* %3 to [103 x [103 x [5 x double]]]*
^getelementptr8BK
I
	full_text<
:
8%72 = getelementptr inbounds double, double* %5, i64 %31
%i648B

	full_text
	
i64 %31
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %57, align 8, !tbaa !8
-double*8B

	full_text

double* %57
1shl8B(
&
	full_text

%73 = shl i64 %15, 32
%i648B

	full_text
	
i64 %15
9ashr8B/
-
	full_text 

%74 = ashr exact i64 %73, 32
%i648B

	full_text
	
i64 %73
1shl8B(
&
	full_text

%75 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
9ashr8B/
-
	full_text 

%76 = ashr exact i64 %75, 32
%i648B

	full_text
	
i64 %75
getelementptr8B|
z
	full_textm
k
i%77 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 0, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Nload8BD
B
	full_text5
3
1%78 = load double, double* %77, align 8, !tbaa !8
-double*8B

	full_text

double* %77
@fmul8B6
4
	full_text'
%
#%79 = fmul double %78, 1.000000e-01
+double8B

	full_text


double %78
getelementptr8B|
z
	full_textm
k
i%80 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 0, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Nload8BD
B
	full_text5
3
1%81 = load double, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
Nstore8BC
A
	full_text4
2
0store double %81, double* %32, align 8, !tbaa !8
+double8B

	full_text


double %81
-double*8B

	full_text

double* %32
call8Bw
u
	full_texth
f
d%82 = tail call double @llvm.fmuladd.f64(double %79, double 0x3FF5555555555555, double 7.500000e-01)
+double8B

	full_text


double %79
call8Bw
u
	full_texth
f
d%83 = tail call double @llvm.fmuladd.f64(double %79, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
+double8B

	full_text


double %79
;fcmp8B1
/
	full_text"
 
%84 = fcmp ogt double %82, %83
+double8B

	full_text


double %82
+double8B

	full_text


double %83
Jselect8B>
<
	full_text/
-
+%85 = select i1 %84, double %82, double %83
#i18B

	full_text


i1 %84
+double8B

	full_text


double %82
+double8B

	full_text


double %83
@fadd8B6
4
	full_text'
%
#%86 = fadd double %79, 7.500000e-01
+double8B

	full_text


double %79
Dfcmp8B:
8
	full_text+
)
'%87 = fcmp ogt double %86, 7.500000e-01
+double8B

	full_text


double %86
Sselect8BG
E
	full_text8
6
4%88 = select i1 %87, double %86, double 7.500000e-01
#i18B

	full_text


i1 %87
+double8B

	full_text


double %86
;fcmp8B1
/
	full_text"
 
%89 = fcmp ogt double %85, %88
+double8B

	full_text


double %85
+double8B

	full_text


double %88
Jselect8B>
<
	full_text/
-
+%90 = select i1 %89, double %85, double %88
#i18B

	full_text


i1 %89
+double8B

	full_text


double %85
+double8B

	full_text


double %88
Nstore8BC
A
	full_text4
2
0store double %90, double* %72, align 8, !tbaa !8
+double8B

	full_text


double %90
-double*8B

	full_text

double* %72
getelementptr8B|
z
	full_textm
k
i%91 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 0, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Nload8BD
B
	full_text5
3
1%92 = load double, double* %91, align 8, !tbaa !8
-double*8B

	full_text

double* %91
getelementptr8B|
z
	full_textm
k
i%93 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 1, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Nload8BD
B
	full_text5
3
1%94 = load double, double* %93, align 8, !tbaa !8
-double*8B

	full_text

double* %93
@fmul8B6
4
	full_text'
%
#%95 = fmul double %94, 1.000000e-01
+double8B

	full_text


double %94
getelementptr8B|
z
	full_textm
k
i%96 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 1, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Nload8BD
B
	full_text5
3
1%97 = load double, double* %96, align 8, !tbaa !8
-double*8B

	full_text

double* %96
]getelementptr8BJ
H
	full_text;
9
7%98 = getelementptr inbounds double, double* %32, i64 1
-double*8B

	full_text

double* %32
Nstore8BC
A
	full_text4
2
0store double %97, double* %98, align 8, !tbaa !8
+double8B

	full_text


double %97
-double*8B

	full_text

double* %98
call8Bw
u
	full_texth
f
d%99 = tail call double @llvm.fmuladd.f64(double %95, double 0x3FF5555555555555, double 7.500000e-01)
+double8B

	full_text


double %95
‚call8Bx
v
	full_texti
g
e%100 = tail call double @llvm.fmuladd.f64(double %95, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
+double8B

	full_text


double %95
=fcmp8B3
1
	full_text$
"
 %101 = fcmp ogt double %99, %100
+double8B

	full_text


double %99
,double8B

	full_text

double %100
Mselect8BA
?
	full_text2
0
.%102 = select i1 %101, double %99, double %100
$i18B

	full_text
	
i1 %101
+double8B

	full_text


double %99
,double8B

	full_text

double %100
Afadd8B7
5
	full_text(
&
$%103 = fadd double %95, 7.500000e-01
+double8B

	full_text


double %95
Ffcmp8B<
:
	full_text-
+
)%104 = fcmp ogt double %103, 7.500000e-01
,double8B

	full_text

double %103
Vselect8BJ
H
	full_text;
9
7%105 = select i1 %104, double %103, double 7.500000e-01
$i18B

	full_text
	
i1 %104
,double8B

	full_text

double %103
>fcmp8B4
2
	full_text%
#
!%106 = fcmp ogt double %102, %105
,double8B

	full_text

double %102
,double8B

	full_text

double %105
Nselect8BB
@
	full_text3
1
/%107 = select i1 %106, double %102, double %105
$i18B

	full_text
	
i1 %106
,double8B

	full_text

double %102
,double8B

	full_text

double %105
^getelementptr8BK
I
	full_text<
:
8%108 = getelementptr inbounds double, double* %72, i64 1
-double*8B

	full_text

double* %72
Pstore8BE
C
	full_text6
4
2store double %107, double* %108, align 8, !tbaa !8
,double8B

	full_text

double %107
.double*8B

	full_text

double* %108
getelementptr8B}
{
	full_textn
l
j%109 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 1, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%110 = load double, double* %109, align 8, !tbaa !8
.double*8B

	full_text

double* %109
getelementptr8B}
{
	full_textn
l
j%111 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 2, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%112 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
Bfmul8B8
6
	full_text)
'
%%113 = fmul double %112, 1.000000e-01
,double8B

	full_text

double %112
getelementptr8B}
{
	full_textn
l
j%114 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 2, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%115 = load double, double* %114, align 8, !tbaa !8
.double*8B

	full_text

double* %114
^getelementptr8BK
I
	full_text<
:
8%116 = getelementptr inbounds double, double* %32, i64 2
-double*8B

	full_text

double* %32
Pstore8BE
C
	full_text6
4
2store double %115, double* %116, align 8, !tbaa !8
,double8B

	full_text

double %115
.double*8B

	full_text

double* %116
ƒcall8By
w
	full_textj
h
f%117 = tail call double @llvm.fmuladd.f64(double %113, double 0x3FF5555555555555, double 7.500000e-01)
,double8B

	full_text

double %113
ƒcall8By
w
	full_textj
h
f%118 = tail call double @llvm.fmuladd.f64(double %113, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
,double8B

	full_text

double %113
>fcmp8B4
2
	full_text%
#
!%119 = fcmp ogt double %117, %118
,double8B

	full_text

double %117
,double8B

	full_text

double %118
Nselect8BB
@
	full_text3
1
/%120 = select i1 %119, double %117, double %118
$i18B

	full_text
	
i1 %119
,double8B

	full_text

double %117
,double8B

	full_text

double %118
Bfadd8B8
6
	full_text)
'
%%121 = fadd double %113, 7.500000e-01
,double8B

	full_text

double %113
Ffcmp8B<
:
	full_text-
+
)%122 = fcmp ogt double %121, 7.500000e-01
,double8B

	full_text

double %121
Vselect8BJ
H
	full_text;
9
7%123 = select i1 %122, double %121, double 7.500000e-01
$i18B

	full_text
	
i1 %122
,double8B

	full_text

double %121
>fcmp8B4
2
	full_text%
#
!%124 = fcmp ogt double %120, %123
,double8B

	full_text

double %120
,double8B

	full_text

double %123
Nselect8BB
@
	full_text3
1
/%125 = select i1 %124, double %120, double %123
$i18B

	full_text
	
i1 %124
,double8B

	full_text

double %120
,double8B

	full_text

double %123
^getelementptr8BK
I
	full_text<
:
8%126 = getelementptr inbounds double, double* %72, i64 2
-double*8B

	full_text

double* %72
Pstore8BE
C
	full_text6
4
2store double %125, double* %126, align 8, !tbaa !8
,double8B

	full_text

double %125
.double*8B

	full_text

double* %126
getelementptr8B}
{
	full_textn
l
j%127 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 2, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%128 = load double, double* %127, align 8, !tbaa !8
.double*8B

	full_text

double* %127
^getelementptr8BK
I
	full_text<
:
8%129 = getelementptr inbounds double, double* %35, i64 5
-double*8B

	full_text

double* %35
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %129, align 8, !tbaa !8
.double*8B

	full_text

double* %129
Afmul8B7
5
	full_text(
&
$%130 = fmul double %90, 1.020100e+01
+double8B

	full_text


double %90
Cfsub8B9
7
	full_text*
(
&%131 = fsub double -0.000000e+00, %130
,double8B

	full_text

double %130
ucall8Bk
i
	full_text\
Z
X%132 = tail call double @llvm.fmuladd.f64(double %81, double -5.050000e-02, double %131)
+double8B

	full_text


double %81
,double8B

	full_text

double %131
_getelementptr8BL
J
	full_text=
;
9%133 = getelementptr inbounds double, double* %129, i64 1
.double*8B

	full_text

double* %129
Pstore8BE
C
	full_text6
4
2store double %132, double* %133, align 8, !tbaa !8
,double8B

	full_text

double %132
.double*8B

	full_text

double* %133
}call8Bs
q
	full_textd
b
`%134 = tail call double @llvm.fmuladd.f64(double %107, double 2.040200e+01, double 1.000000e+00)
,double8B

	full_text

double %107
Bfadd8B8
6
	full_text)
'
%%135 = fadd double %134, 1.250000e-03
,double8B

	full_text

double %134
_getelementptr8BL
J
	full_text=
;
9%136 = getelementptr inbounds double, double* %129, i64 2
.double*8B

	full_text

double* %129
Pstore8BE
C
	full_text6
4
2store double %135, double* %136, align 8, !tbaa !8
,double8B

	full_text

double %135
.double*8B

	full_text

double* %136
Bfmul8B8
6
	full_text)
'
%%137 = fmul double %125, 1.020100e+01
,double8B

	full_text

double %125
Cfsub8B9
7
	full_text*
(
&%138 = fsub double -0.000000e+00, %137
,double8B

	full_text

double %137
ucall8Bk
i
	full_text\
Z
X%139 = tail call double @llvm.fmuladd.f64(double %115, double 5.050000e-02, double %138)
,double8B

	full_text

double %115
,double8B

	full_text

double %138
Cfadd8B9
7
	full_text*
(
&%140 = fadd double %139, -1.000000e-03
,double8B

	full_text

double %139
_getelementptr8BL
J
	full_text=
;
9%141 = getelementptr inbounds double, double* %129, i64 3
.double*8B

	full_text

double* %129
Pstore8BE
C
	full_text6
4
2store double %140, double* %141, align 8, !tbaa !8
,double8B

	full_text

double %140
.double*8B

	full_text

double* %141
_getelementptr8BL
J
	full_text=
;
9%142 = getelementptr inbounds double, double* %129, i64 4
.double*8B

	full_text

double* %129
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
^getelementptr8BK
I
	full_text<
:
8%143 = getelementptr inbounds double, double* %37, i64 5
-double*8B

	full_text

double* %37
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %143, align 8, !tbaa !8
.double*8B

	full_text

double* %143
ucall8Bk
i
	full_text\
Z
X%144 = tail call double @llvm.fmuladd.f64(double %92, double -5.050000e-02, double %132)
+double8B

	full_text


double %92
,double8B

	full_text

double %132
_getelementptr8BL
J
	full_text=
;
9%145 = getelementptr inbounds double, double* %143, i64 1
.double*8B

	full_text

double* %143
Pstore8BE
C
	full_text6
4
2store double %144, double* %145, align 8, !tbaa !8
,double8B

	full_text

double %144
.double*8B

	full_text

double* %145
_getelementptr8BL
J
	full_text=
;
9%146 = getelementptr inbounds double, double* %143, i64 2
.double*8B

	full_text

double* %143
Pstore8BE
C
	full_text6
4
2store double %135, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %135
.double*8B

	full_text

double* %146
ucall8Bk
i
	full_text\
Z
X%147 = tail call double @llvm.fmuladd.f64(double %128, double 5.050000e-02, double %140)
,double8B

	full_text

double %128
,double8B

	full_text

double %140
_getelementptr8BL
J
	full_text=
;
9%148 = getelementptr inbounds double, double* %143, i64 3
.double*8B

	full_text

double* %143
Pstore8BE
C
	full_text6
4
2store double %147, double* %148, align 8, !tbaa !8
,double8B

	full_text

double %147
.double*8B

	full_text

double* %148
_getelementptr8BL
J
	full_text=
;
9%149 = getelementptr inbounds double, double* %143, i64 4
.double*8B

	full_text

double* %143
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %149, align 8, !tbaa !8
.double*8B

	full_text

double* %149
^getelementptr8BK
I
	full_text<
:
8%150 = getelementptr inbounds double, double* %39, i64 5
-double*8B

	full_text

double* %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %150, align 8, !tbaa !8
.double*8B

	full_text

double* %150
tcall8Bj
h
	full_text[
Y
W%151 = tail call double @llvm.fmuladd.f64(double %92, double 5.050000e-02, double %132)
+double8B

	full_text


double %92
,double8B

	full_text

double %132
_getelementptr8BL
J
	full_text=
;
9%152 = getelementptr inbounds double, double* %150, i64 1
.double*8B

	full_text

double* %150
Pstore8BE
C
	full_text6
4
2store double %151, double* %152, align 8, !tbaa !8
,double8B

	full_text

double %151
.double*8B

	full_text

double* %152
_getelementptr8BL
J
	full_text=
;
9%153 = getelementptr inbounds double, double* %150, i64 2
.double*8B

	full_text

double* %150
Pstore8BE
C
	full_text6
4
2store double %135, double* %153, align 8, !tbaa !8
,double8B

	full_text

double %135
.double*8B

	full_text

double* %153
vcall8Bl
j
	full_text]
[
Y%154 = tail call double @llvm.fmuladd.f64(double %128, double -5.050000e-02, double %140)
,double8B

	full_text

double %128
,double8B

	full_text

double %140
_getelementptr8BL
J
	full_text=
;
9%155 = getelementptr inbounds double, double* %150, i64 3
.double*8B

	full_text

double* %150
Pstore8BE
C
	full_text6
4
2store double %154, double* %155, align 8, !tbaa !8
,double8B

	full_text

double %154
.double*8B

	full_text

double* %155
_getelementptr8BL
J
	full_text=
;
9%156 = getelementptr inbounds double, double* %150, i64 4
.double*8B

	full_text

double* %150
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
getelementptr8B}
{
	full_textn
l
j%157 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 3, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%158 = load double, double* %157, align 8, !tbaa !8
.double*8B

	full_text

double* %157
Bfmul8B8
6
	full_text)
'
%%159 = fmul double %158, 1.000000e-01
,double8B

	full_text

double %158
getelementptr8B}
{
	full_textn
l
j%160 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 3, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%161 = load double, double* %160, align 8, !tbaa !8
.double*8B

	full_text

double* %160
^getelementptr8BK
I
	full_text<
:
8%162 = getelementptr inbounds double, double* %32, i64 3
-double*8B

	full_text

double* %32
Pstore8BE
C
	full_text6
4
2store double %161, double* %162, align 8, !tbaa !8
,double8B

	full_text

double %161
.double*8B

	full_text

double* %162
ƒcall8By
w
	full_textj
h
f%163 = tail call double @llvm.fmuladd.f64(double %159, double 0x3FF5555555555555, double 7.500000e-01)
,double8B

	full_text

double %159
ƒcall8By
w
	full_textj
h
f%164 = tail call double @llvm.fmuladd.f64(double %159, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
,double8B

	full_text

double %159
>fcmp8B4
2
	full_text%
#
!%165 = fcmp ogt double %163, %164
,double8B

	full_text

double %163
,double8B

	full_text

double %164
Nselect8BB
@
	full_text3
1
/%166 = select i1 %165, double %163, double %164
$i18B

	full_text
	
i1 %165
,double8B

	full_text

double %163
,double8B

	full_text

double %164
Bfadd8B8
6
	full_text)
'
%%167 = fadd double %159, 7.500000e-01
,double8B

	full_text

double %159
Ffcmp8B<
:
	full_text-
+
)%168 = fcmp ogt double %167, 7.500000e-01
,double8B

	full_text

double %167
Vselect8BJ
H
	full_text;
9
7%169 = select i1 %168, double %167, double 7.500000e-01
$i18B

	full_text
	
i1 %168
,double8B

	full_text

double %167
>fcmp8B4
2
	full_text%
#
!%170 = fcmp ogt double %166, %169
,double8B

	full_text

double %166
,double8B

	full_text

double %169
Nselect8BB
@
	full_text3
1
/%171 = select i1 %170, double %166, double %169
$i18B

	full_text
	
i1 %170
,double8B

	full_text

double %166
,double8B

	full_text

double %169
^getelementptr8BK
I
	full_text<
:
8%172 = getelementptr inbounds double, double* %72, i64 3
-double*8B

	full_text

double* %72
Pstore8BE
C
	full_text6
4
2store double %171, double* %172, align 8, !tbaa !8
,double8B

	full_text

double %171
.double*8B

	full_text

double* %172
getelementptr8B}
{
	full_textn
l
j%173 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 3, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%174 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
_getelementptr8BL
J
	full_text=
;
9%175 = getelementptr inbounds double, double* %35, i64 10
-double*8B

	full_text

double* %35
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %175, align 8, !tbaa !8
.double*8B

	full_text

double* %175
Bfmul8B8
6
	full_text)
'
%%176 = fmul double %107, 1.020100e+01
,double8B

	full_text

double %107
Cfsub8B9
7
	full_text*
(
&%177 = fsub double -0.000000e+00, %176
,double8B

	full_text

double %176
ucall8Bk
i
	full_text\
Z
X%178 = tail call double @llvm.fmuladd.f64(double %97, double -5.050000e-02, double %177)
+double8B

	full_text


double %97
,double8B

	full_text

double %177
Cfadd8B9
7
	full_text*
(
&%179 = fadd double %178, -1.000000e-03
,double8B

	full_text

double %178
_getelementptr8BL
J
	full_text=
;
9%180 = getelementptr inbounds double, double* %175, i64 1
.double*8B

	full_text

double* %175
Pstore8BE
C
	full_text6
4
2store double %179, double* %180, align 8, !tbaa !8
,double8B

	full_text

double %179
.double*8B

	full_text

double* %180
}call8Bs
q
	full_textd
b
`%181 = tail call double @llvm.fmuladd.f64(double %125, double 2.040200e+01, double 1.000000e+00)
,double8B

	full_text

double %125
Bfadd8B8
6
	full_text)
'
%%182 = fadd double %181, 1.500000e-03
,double8B

	full_text

double %181
_getelementptr8BL
J
	full_text=
;
9%183 = getelementptr inbounds double, double* %175, i64 2
.double*8B

	full_text

double* %175
Pstore8BE
C
	full_text6
4
2store double %182, double* %183, align 8, !tbaa !8
,double8B

	full_text

double %182
.double*8B

	full_text

double* %183
Bfmul8B8
6
	full_text)
'
%%184 = fmul double %171, 1.020100e+01
,double8B

	full_text

double %171
Cfsub8B9
7
	full_text*
(
&%185 = fsub double -0.000000e+00, %184
,double8B

	full_text

double %184
ucall8Bk
i
	full_text\
Z
X%186 = tail call double @llvm.fmuladd.f64(double %161, double 5.050000e-02, double %185)
,double8B

	full_text

double %161
,double8B

	full_text

double %185
Cfadd8B9
7
	full_text*
(
&%187 = fadd double %186, -1.000000e-03
,double8B

	full_text

double %186
_getelementptr8BL
J
	full_text=
;
9%188 = getelementptr inbounds double, double* %175, i64 3
.double*8B

	full_text

double* %175
Pstore8BE
C
	full_text6
4
2store double %187, double* %188, align 8, !tbaa !8
,double8B

	full_text

double %187
.double*8B

	full_text

double* %188
_getelementptr8BL
J
	full_text=
;
9%189 = getelementptr inbounds double, double* %175, i64 4
.double*8B

	full_text

double* %175
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
_getelementptr8BL
J
	full_text=
;
9%190 = getelementptr inbounds double, double* %37, i64 10
-double*8B

	full_text

double* %37
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %190, align 8, !tbaa !8
.double*8B

	full_text

double* %190
vcall8Bl
j
	full_text]
[
Y%191 = tail call double @llvm.fmuladd.f64(double %110, double -5.050000e-02, double %179)
,double8B

	full_text

double %110
,double8B

	full_text

double %179
_getelementptr8BL
J
	full_text=
;
9%192 = getelementptr inbounds double, double* %190, i64 1
.double*8B

	full_text

double* %190
Pstore8BE
C
	full_text6
4
2store double %191, double* %192, align 8, !tbaa !8
,double8B

	full_text

double %191
.double*8B

	full_text

double* %192
_getelementptr8BL
J
	full_text=
;
9%193 = getelementptr inbounds double, double* %190, i64 2
.double*8B

	full_text

double* %190
Pstore8BE
C
	full_text6
4
2store double %182, double* %193, align 8, !tbaa !8
,double8B

	full_text

double %182
.double*8B

	full_text

double* %193
ucall8Bk
i
	full_text\
Z
X%194 = tail call double @llvm.fmuladd.f64(double %174, double 5.050000e-02, double %187)
,double8B

	full_text

double %174
,double8B

	full_text

double %187
_getelementptr8BL
J
	full_text=
;
9%195 = getelementptr inbounds double, double* %190, i64 3
.double*8B

	full_text

double* %190
Pstore8BE
C
	full_text6
4
2store double %194, double* %195, align 8, !tbaa !8
,double8B

	full_text

double %194
.double*8B

	full_text

double* %195
_getelementptr8BL
J
	full_text=
;
9%196 = getelementptr inbounds double, double* %190, i64 4
.double*8B

	full_text

double* %190
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %196, align 8, !tbaa !8
.double*8B

	full_text

double* %196
_getelementptr8BL
J
	full_text=
;
9%197 = getelementptr inbounds double, double* %39, i64 10
-double*8B

	full_text

double* %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %197, align 8, !tbaa !8
.double*8B

	full_text

double* %197
ucall8Bk
i
	full_text\
Z
X%198 = tail call double @llvm.fmuladd.f64(double %110, double 5.050000e-02, double %179)
,double8B

	full_text

double %110
,double8B

	full_text

double %179
_getelementptr8BL
J
	full_text=
;
9%199 = getelementptr inbounds double, double* %197, i64 1
.double*8B

	full_text

double* %197
Pstore8BE
C
	full_text6
4
2store double %198, double* %199, align 8, !tbaa !8
,double8B

	full_text

double %198
.double*8B

	full_text

double* %199
_getelementptr8BL
J
	full_text=
;
9%200 = getelementptr inbounds double, double* %197, i64 2
.double*8B

	full_text

double* %197
Pstore8BE
C
	full_text6
4
2store double %182, double* %200, align 8, !tbaa !8
,double8B

	full_text

double %182
.double*8B

	full_text

double* %200
vcall8Bl
j
	full_text]
[
Y%201 = tail call double @llvm.fmuladd.f64(double %174, double -5.050000e-02, double %187)
,double8B

	full_text

double %174
,double8B

	full_text

double %187
_getelementptr8BL
J
	full_text=
;
9%202 = getelementptr inbounds double, double* %197, i64 3
.double*8B

	full_text

double* %197
Pstore8BE
C
	full_text6
4
2store double %201, double* %202, align 8, !tbaa !8
,double8B

	full_text

double %201
.double*8B

	full_text

double* %202
_getelementptr8BL
J
	full_text=
;
9%203 = getelementptr inbounds double, double* %197, i64 4
.double*8B

	full_text

double* %197
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %203, align 8, !tbaa !8
.double*8B

	full_text

double* %203
7icmp8B-
+
	full_text

%204 = icmp slt i32 %12, 7
2add8B)
'
	full_text

%205 = add i32 %12, -3
=br8B5
3
	full_text&
$
"br i1 %204, label %265, label %206
$i18B

	full_text
	
i1 %204
8zext8B.
,
	full_text

%207 = zext i32 %205 to i64
&i328B

	full_text


i32 %205
(br8B 

	full_text

br label %208
Fphi8B=
;
	full_text.
,
*%209 = phi i64 [ %216, %208 ], [ 3, %206 ]
&i648B

	full_text


i64 %216
Lphi8BC
A
	full_text4
2
0%210 = phi double [ %234, %208 ], [ %174, %206 ]
,double8B

	full_text

double %234
,double8B

	full_text

double %174
Lphi8BC
A
	full_text4
2
0%211 = phi double [ %210, %208 ], [ %128, %206 ]
,double8B

	full_text

double %210
,double8B

	full_text

double %128
Lphi8BC
A
	full_text4
2
0%212 = phi double [ %231, %208 ], [ %171, %206 ]
,double8B

	full_text

double %231
,double8B

	full_text

double %171
Lphi8BC
A
	full_text4
2
0%213 = phi double [ %212, %208 ], [ %125, %206 ]
,double8B

	full_text

double %212
,double8B

	full_text

double %125
Lphi8BC
A
	full_text4
2
0%214 = phi double [ %221, %208 ], [ %161, %206 ]
,double8B

	full_text

double %221
,double8B

	full_text

double %161
Lphi8BC
A
	full_text4
2
0%215 = phi double [ %214, %208 ], [ %115, %206 ]
,double8B

	full_text

double %214
,double8B

	full_text

double %115
:add8B1
/
	full_text"
 
%216 = add nuw nsw i64 %209, 1
&i648B

	full_text


i64 %209
”getelementptr8B€
~
	full_textq
o
m%217 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 %216, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %216
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%218 = load double, double* %217, align 8, !tbaa !8
.double*8B

	full_text

double* %217
Bfmul8B8
6
	full_text)
'
%%219 = fmul double %218, 1.000000e-01
,double8B

	full_text

double %218
”getelementptr8B€
~
	full_textq
o
m%220 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 %216, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %216
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%221 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
agetelementptr8BN
L
	full_text?
=
;%222 = getelementptr inbounds double, double* %32, i64 %216
-double*8B

	full_text

double* %32
&i648B

	full_text


i64 %216
Pstore8BE
C
	full_text6
4
2store double %221, double* %222, align 8, !tbaa !8
,double8B

	full_text

double %221
.double*8B

	full_text

double* %222
ƒcall8By
w
	full_textj
h
f%223 = tail call double @llvm.fmuladd.f64(double %219, double 0x3FF5555555555555, double 7.500000e-01)
,double8B

	full_text

double %219
ƒcall8By
w
	full_textj
h
f%224 = tail call double @llvm.fmuladd.f64(double %219, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
,double8B

	full_text

double %219
>fcmp8B4
2
	full_text%
#
!%225 = fcmp ogt double %223, %224
,double8B

	full_text

double %223
,double8B

	full_text

double %224
Nselect8BB
@
	full_text3
1
/%226 = select i1 %225, double %223, double %224
$i18B

	full_text
	
i1 %225
,double8B

	full_text

double %223
,double8B

	full_text

double %224
Bfadd8B8
6
	full_text)
'
%%227 = fadd double %219, 7.500000e-01
,double8B

	full_text

double %219
Ffcmp8B<
:
	full_text-
+
)%228 = fcmp ogt double %227, 7.500000e-01
,double8B

	full_text

double %227
Vselect8BJ
H
	full_text;
9
7%229 = select i1 %228, double %227, double 7.500000e-01
$i18B

	full_text
	
i1 %228
,double8B

	full_text

double %227
>fcmp8B4
2
	full_text%
#
!%230 = fcmp ogt double %226, %229
,double8B

	full_text

double %226
,double8B

	full_text

double %229
Nselect8BB
@
	full_text3
1
/%231 = select i1 %230, double %226, double %229
$i18B

	full_text
	
i1 %230
,double8B

	full_text

double %226
,double8B

	full_text

double %229
agetelementptr8BN
L
	full_text?
=
;%232 = getelementptr inbounds double, double* %72, i64 %216
-double*8B

	full_text

double* %72
&i648B

	full_text


i64 %216
Pstore8BE
C
	full_text6
4
2store double %231, double* %232, align 8, !tbaa !8
,double8B

	full_text

double %231
.double*8B

	full_text

double* %232
”getelementptr8B€
~
	full_textq
o
m%233 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 %216, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %216
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%234 = load double, double* %233, align 8, !tbaa !8
.double*8B

	full_text

double* %233
tgetelementptr8Ba
_
	full_textR
P
N%235 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %209, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %209
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %235, align 8, !tbaa !8
.double*8B

	full_text

double* %235
Bfmul8B8
6
	full_text)
'
%%236 = fmul double %213, 1.020100e+01
,double8B

	full_text

double %213
Cfsub8B9
7
	full_text*
(
&%237 = fsub double -0.000000e+00, %236
,double8B

	full_text

double %236
vcall8Bl
j
	full_text]
[
Y%238 = tail call double @llvm.fmuladd.f64(double %215, double -5.050000e-02, double %237)
,double8B

	full_text

double %215
,double8B

	full_text

double %237
Cfadd8B9
7
	full_text*
(
&%239 = fadd double %238, -1.000000e-03
,double8B

	full_text

double %238
tgetelementptr8Ba
_
	full_textR
P
N%240 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %209, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %239, double* %240, align 8, !tbaa !8
,double8B

	full_text

double %239
.double*8B

	full_text

double* %240
}call8Bs
q
	full_textd
b
`%241 = tail call double @llvm.fmuladd.f64(double %212, double 2.040200e+01, double 1.000000e+00)
,double8B

	full_text

double %212
Bfadd8B8
6
	full_text)
'
%%242 = fadd double %241, 1.500000e-03
,double8B

	full_text

double %241
tgetelementptr8Ba
_
	full_textR
P
N%243 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %209, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %242, double* %243, align 8, !tbaa !8
,double8B

	full_text

double %242
.double*8B

	full_text

double* %243
Bfmul8B8
6
	full_text)
'
%%244 = fmul double %231, 1.020100e+01
,double8B

	full_text

double %231
Cfsub8B9
7
	full_text*
(
&%245 = fsub double -0.000000e+00, %244
,double8B

	full_text

double %244
ucall8Bk
i
	full_text\
Z
X%246 = tail call double @llvm.fmuladd.f64(double %221, double 5.050000e-02, double %245)
,double8B

	full_text

double %221
,double8B

	full_text

double %245
Cfadd8B9
7
	full_text*
(
&%247 = fadd double %246, -1.000000e-03
,double8B

	full_text

double %246
tgetelementptr8Ba
_
	full_textR
P
N%248 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %209, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %247, double* %248, align 8, !tbaa !8
,double8B

	full_text

double %247
.double*8B

	full_text

double* %248
tgetelementptr8Ba
_
	full_textR
P
N%249 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %209, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %209
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %249, align 8, !tbaa !8
.double*8B

	full_text

double* %249
tgetelementptr8Ba
_
	full_textR
P
N%250 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %209, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %209
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %250, align 8, !tbaa !8
.double*8B

	full_text

double* %250
vcall8Bl
j
	full_text]
[
Y%251 = tail call double @llvm.fmuladd.f64(double %211, double -5.050000e-02, double %239)
,double8B

	full_text

double %211
,double8B

	full_text

double %239
tgetelementptr8Ba
_
	full_textR
P
N%252 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %209, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %251, double* %252, align 8, !tbaa !8
,double8B

	full_text

double %251
.double*8B

	full_text

double* %252
tgetelementptr8Ba
_
	full_textR
P
N%253 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %209, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %242, double* %253, align 8, !tbaa !8
,double8B

	full_text

double %242
.double*8B

	full_text

double* %253
ucall8Bk
i
	full_text\
Z
X%254 = tail call double @llvm.fmuladd.f64(double %234, double 5.050000e-02, double %247)
,double8B

	full_text

double %234
,double8B

	full_text

double %247
tgetelementptr8Ba
_
	full_textR
P
N%255 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %209, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %254, double* %255, align 8, !tbaa !8
,double8B

	full_text

double %254
.double*8B

	full_text

double* %255
tgetelementptr8Ba
_
	full_textR
P
N%256 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %209, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %209
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %256, align 8, !tbaa !8
.double*8B

	full_text

double* %256
tgetelementptr8Ba
_
	full_textR
P
N%257 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %209, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %209
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %257, align 8, !tbaa !8
.double*8B

	full_text

double* %257
ucall8Bk
i
	full_text\
Z
X%258 = tail call double @llvm.fmuladd.f64(double %211, double 5.050000e-02, double %239)
,double8B

	full_text

double %211
,double8B

	full_text

double %239
tgetelementptr8Ba
_
	full_textR
P
N%259 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %209, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %258, double* %259, align 8, !tbaa !8
,double8B

	full_text

double %258
.double*8B

	full_text

double* %259
tgetelementptr8Ba
_
	full_textR
P
N%260 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %209, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %242, double* %260, align 8, !tbaa !8
,double8B

	full_text

double %242
.double*8B

	full_text

double* %260
vcall8Bl
j
	full_text]
[
Y%261 = tail call double @llvm.fmuladd.f64(double %234, double -5.050000e-02, double %247)
,double8B

	full_text

double %234
,double8B

	full_text

double %247
tgetelementptr8Ba
_
	full_textR
P
N%262 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %209, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %209
Pstore8BE
C
	full_text6
4
2store double %261, double* %262, align 8, !tbaa !8
,double8B

	full_text

double %261
.double*8B

	full_text

double* %262
tgetelementptr8Ba
_
	full_textR
P
N%263 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %209, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %209
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %263, align 8, !tbaa !8
.double*8B

	full_text

double* %263
:icmp8B0
.
	full_text!

%264 = icmp eq i64 %216, %207
&i648B

	full_text


i64 %216
&i648B

	full_text


i64 %207
=br8B5
3
	full_text&
$
"br i1 %264, label %265, label %208
$i18B

	full_text
	
i1 %264
Kphi8BB
@
	full_text3
1
/%266 = phi double [ %115, %23 ], [ %214, %208 ]
,double8B

	full_text

double %115
,double8B

	full_text

double %214
Kphi8BB
@
	full_text3
1
/%267 = phi double [ %161, %23 ], [ %221, %208 ]
,double8B

	full_text

double %161
,double8B

	full_text

double %221
Kphi8BB
@
	full_text3
1
/%268 = phi double [ %125, %23 ], [ %212, %208 ]
,double8B

	full_text

double %125
,double8B

	full_text

double %212
Kphi8BB
@
	full_text3
1
/%269 = phi double [ %171, %23 ], [ %231, %208 ]
,double8B

	full_text

double %171
,double8B

	full_text

double %231
Kphi8BB
@
	full_text3
1
/%270 = phi double [ %128, %23 ], [ %210, %208 ]
,double8B

	full_text

double %128
,double8B

	full_text

double %210
Kphi8BB
@
	full_text3
1
/%271 = phi double [ %174, %23 ], [ %234, %208 ]
,double8B

	full_text

double %174
,double8B

	full_text

double %234
6add8B-
+
	full_text

%272 = add nsw i32 %12, -2
8sext8B.
,
	full_text

%273 = sext i32 %272 to i64
&i328B

	full_text


i32 %272
”getelementptr8B€
~
	full_textq
o
m%274 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 %273, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %273
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%275 = load double, double* %274, align 8, !tbaa !8
.double*8B

	full_text

double* %274
Bfmul8B8
6
	full_text)
'
%%276 = fmul double %275, 1.000000e-01
,double8B

	full_text

double %275
”getelementptr8B€
~
	full_textq
o
m%277 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 %273, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %273
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%278 = load double, double* %277, align 8, !tbaa !8
.double*8B

	full_text

double* %277
agetelementptr8BN
L
	full_text?
=
;%279 = getelementptr inbounds double, double* %32, i64 %273
-double*8B

	full_text

double* %32
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %278, double* %279, align 8, !tbaa !8
,double8B

	full_text

double %278
.double*8B

	full_text

double* %279
ƒcall8By
w
	full_textj
h
f%280 = tail call double @llvm.fmuladd.f64(double %276, double 0x3FF5555555555555, double 7.500000e-01)
,double8B

	full_text

double %276
ƒcall8By
w
	full_textj
h
f%281 = tail call double @llvm.fmuladd.f64(double %276, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
,double8B

	full_text

double %276
>fcmp8B4
2
	full_text%
#
!%282 = fcmp ogt double %280, %281
,double8B

	full_text

double %280
,double8B

	full_text

double %281
Nselect8BB
@
	full_text3
1
/%283 = select i1 %282, double %280, double %281
$i18B

	full_text
	
i1 %282
,double8B

	full_text

double %280
,double8B

	full_text

double %281
Bfadd8B8
6
	full_text)
'
%%284 = fadd double %276, 7.500000e-01
,double8B

	full_text

double %276
Ffcmp8B<
:
	full_text-
+
)%285 = fcmp ogt double %284, 7.500000e-01
,double8B

	full_text

double %284
Vselect8BJ
H
	full_text;
9
7%286 = select i1 %285, double %284, double 7.500000e-01
$i18B

	full_text
	
i1 %285
,double8B

	full_text

double %284
>fcmp8B4
2
	full_text%
#
!%287 = fcmp ogt double %283, %286
,double8B

	full_text

double %283
,double8B

	full_text

double %286
Nselect8BB
@
	full_text3
1
/%288 = select i1 %287, double %283, double %286
$i18B

	full_text
	
i1 %287
,double8B

	full_text

double %283
,double8B

	full_text

double %286
agetelementptr8BN
L
	full_text?
=
;%289 = getelementptr inbounds double, double* %72, i64 %273
-double*8B

	full_text

double* %72
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %288, double* %289, align 8, !tbaa !8
,double8B

	full_text

double %288
.double*8B

	full_text

double* %289
”getelementptr8B€
~
	full_textq
o
m%290 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 %273, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %273
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%291 = load double, double* %290, align 8, !tbaa !8
.double*8B

	full_text

double* %290
8sext8B.
,
	full_text

%292 = sext i32 %205 to i64
&i328B

	full_text


i32 %205
tgetelementptr8Ba
_
	full_textR
P
N%293 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %292, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %292
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %293, align 8, !tbaa !8
.double*8B

	full_text

double* %293
Bfmul8B8
6
	full_text)
'
%%294 = fmul double %268, 1.020100e+01
,double8B

	full_text

double %268
Cfsub8B9
7
	full_text*
(
&%295 = fsub double -0.000000e+00, %294
,double8B

	full_text

double %294
vcall8Bl
j
	full_text]
[
Y%296 = tail call double @llvm.fmuladd.f64(double %266, double -5.050000e-02, double %295)
,double8B

	full_text

double %266
,double8B

	full_text

double %295
Cfadd8B9
7
	full_text*
(
&%297 = fadd double %296, -1.000000e-03
,double8B

	full_text

double %296
tgetelementptr8Ba
_
	full_textR
P
N%298 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %292, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %297, double* %298, align 8, !tbaa !8
,double8B

	full_text

double %297
.double*8B

	full_text

double* %298
}call8Bs
q
	full_textd
b
`%299 = tail call double @llvm.fmuladd.f64(double %269, double 2.040200e+01, double 1.000000e+00)
,double8B

	full_text

double %269
Bfadd8B8
6
	full_text)
'
%%300 = fadd double %299, 1.500000e-03
,double8B

	full_text

double %299
tgetelementptr8Ba
_
	full_textR
P
N%301 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %292, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %300, double* %301, align 8, !tbaa !8
,double8B

	full_text

double %300
.double*8B

	full_text

double* %301
Bfmul8B8
6
	full_text)
'
%%302 = fmul double %288, 1.020100e+01
,double8B

	full_text

double %288
Cfsub8B9
7
	full_text*
(
&%303 = fsub double -0.000000e+00, %302
,double8B

	full_text

double %302
ucall8Bk
i
	full_text\
Z
X%304 = tail call double @llvm.fmuladd.f64(double %278, double 5.050000e-02, double %303)
,double8B

	full_text

double %278
,double8B

	full_text

double %303
Cfadd8B9
7
	full_text*
(
&%305 = fadd double %304, -1.000000e-03
,double8B

	full_text

double %304
tgetelementptr8Ba
_
	full_textR
P
N%306 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %292, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %305, double* %306, align 8, !tbaa !8
,double8B

	full_text

double %305
.double*8B

	full_text

double* %306
tgetelementptr8Ba
_
	full_textR
P
N%307 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %292, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %292
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %307, align 8, !tbaa !8
.double*8B

	full_text

double* %307
tgetelementptr8Ba
_
	full_textR
P
N%308 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %292, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %292
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %308, align 8, !tbaa !8
.double*8B

	full_text

double* %308
vcall8Bl
j
	full_text]
[
Y%309 = tail call double @llvm.fmuladd.f64(double %270, double -5.050000e-02, double %297)
,double8B

	full_text

double %270
,double8B

	full_text

double %297
tgetelementptr8Ba
_
	full_textR
P
N%310 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %292, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %309, double* %310, align 8, !tbaa !8
,double8B

	full_text

double %309
.double*8B

	full_text

double* %310
tgetelementptr8Ba
_
	full_textR
P
N%311 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %292, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %300, double* %311, align 8, !tbaa !8
,double8B

	full_text

double %300
.double*8B

	full_text

double* %311
ucall8Bk
i
	full_text\
Z
X%312 = tail call double @llvm.fmuladd.f64(double %291, double 5.050000e-02, double %305)
,double8B

	full_text

double %291
,double8B

	full_text

double %305
tgetelementptr8Ba
_
	full_textR
P
N%313 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %292, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %312, double* %313, align 8, !tbaa !8
,double8B

	full_text

double %312
.double*8B

	full_text

double* %313
tgetelementptr8Ba
_
	full_textR
P
N%314 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %292, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %292
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %314, align 8, !tbaa !8
.double*8B

	full_text

double* %314
tgetelementptr8Ba
_
	full_textR
P
N%315 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %292, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %292
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %315, align 8, !tbaa !8
.double*8B

	full_text

double* %315
ucall8Bk
i
	full_text\
Z
X%316 = tail call double @llvm.fmuladd.f64(double %270, double 5.050000e-02, double %297)
,double8B

	full_text

double %270
,double8B

	full_text

double %297
tgetelementptr8Ba
_
	full_textR
P
N%317 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %292, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %316, double* %317, align 8, !tbaa !8
,double8B

	full_text

double %316
.double*8B

	full_text

double* %317
tgetelementptr8Ba
_
	full_textR
P
N%318 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %292, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %300, double* %318, align 8, !tbaa !8
,double8B

	full_text

double %300
.double*8B

	full_text

double* %318
vcall8Bl
j
	full_text]
[
Y%319 = tail call double @llvm.fmuladd.f64(double %291, double -5.050000e-02, double %305)
,double8B

	full_text

double %291
,double8B

	full_text

double %305
tgetelementptr8Ba
_
	full_textR
P
N%320 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %292, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %292
Pstore8BE
C
	full_text6
4
2store double %319, double* %320, align 8, !tbaa !8
,double8B

	full_text

double %319
.double*8B

	full_text

double* %320
tgetelementptr8Ba
_
	full_textR
P
N%321 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %292, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %292
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %321, align 8, !tbaa !8
.double*8B

	full_text

double* %321
6add8B-
+
	full_text

%322 = add nsw i32 %12, -1
8sext8B.
,
	full_text

%323 = sext i32 %322 to i64
&i328B

	full_text


i32 %322
”getelementptr8B€
~
	full_textq
o
m%324 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %70, i64 %74, i64 %323, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %70
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %323
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%325 = load double, double* %324, align 8, !tbaa !8
.double*8B

	full_text

double* %324
Bfmul8B8
6
	full_text)
'
%%326 = fmul double %325, 1.000000e-01
,double8B

	full_text

double %325
”getelementptr8B€
~
	full_textq
o
m%327 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %24, i64 %74, i64 %323, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %24
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %323
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%328 = load double, double* %327, align 8, !tbaa !8
.double*8B

	full_text

double* %327
agetelementptr8BN
L
	full_text?
=
;%329 = getelementptr inbounds double, double* %32, i64 %323
-double*8B

	full_text

double* %32
&i648B

	full_text


i64 %323
Pstore8BE
C
	full_text6
4
2store double %328, double* %329, align 8, !tbaa !8
,double8B

	full_text

double %328
.double*8B

	full_text

double* %329
ƒcall8By
w
	full_textj
h
f%330 = tail call double @llvm.fmuladd.f64(double %326, double 0x3FF5555555555555, double 7.500000e-01)
,double8B

	full_text

double %326
ƒcall8By
w
	full_textj
h
f%331 = tail call double @llvm.fmuladd.f64(double %326, double 0x3FFF5C28F5C28F5B, double 7.500000e-01)
,double8B

	full_text

double %326
>fcmp8B4
2
	full_text%
#
!%332 = fcmp ogt double %330, %331
,double8B

	full_text

double %330
,double8B

	full_text

double %331
Nselect8BB
@
	full_text3
1
/%333 = select i1 %332, double %330, double %331
$i18B

	full_text
	
i1 %332
,double8B

	full_text

double %330
,double8B

	full_text

double %331
Bfadd8B8
6
	full_text)
'
%%334 = fadd double %326, 7.500000e-01
,double8B

	full_text

double %326
Ffcmp8B<
:
	full_text-
+
)%335 = fcmp ogt double %334, 7.500000e-01
,double8B

	full_text

double %334
Vselect8BJ
H
	full_text;
9
7%336 = select i1 %335, double %334, double 7.500000e-01
$i18B

	full_text
	
i1 %335
,double8B

	full_text

double %334
>fcmp8B4
2
	full_text%
#
!%337 = fcmp ogt double %333, %336
,double8B

	full_text

double %333
,double8B

	full_text

double %336
Nselect8BB
@
	full_text3
1
/%338 = select i1 %337, double %333, double %336
$i18B

	full_text
	
i1 %337
,double8B

	full_text

double %333
,double8B

	full_text

double %336
agetelementptr8BN
L
	full_text?
=
;%339 = getelementptr inbounds double, double* %72, i64 %323
-double*8B

	full_text

double* %72
&i648B

	full_text


i64 %323
Pstore8BE
C
	full_text6
4
2store double %338, double* %339, align 8, !tbaa !8
,double8B

	full_text

double %338
.double*8B

	full_text

double* %339
”getelementptr8B€
~
	full_textq
o
m%340 = getelementptr inbounds [103 x [103 x double]], [103 x [103 x double]]* %25, i64 %74, i64 %323, i64 %76
M[103 x [103 x double]]*8B.
,
	full_text

[103 x [103 x double]]* %25
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %323
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%341 = load double, double* %340, align 8, !tbaa !8
.double*8B

	full_text

double* %340
tgetelementptr8Ba
_
	full_textR
P
N%342 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %273, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %273
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %342, align 8, !tbaa !8
.double*8B

	full_text

double* %342
Bfmul8B8
6
	full_text)
'
%%343 = fmul double %269, 1.020100e+01
,double8B

	full_text

double %269
Cfsub8B9
7
	full_text*
(
&%344 = fsub double -0.000000e+00, %343
,double8B

	full_text

double %343
vcall8Bl
j
	full_text]
[
Y%345 = tail call double @llvm.fmuladd.f64(double %267, double -5.050000e-02, double %344)
,double8B

	full_text

double %267
,double8B

	full_text

double %344
Cfadd8B9
7
	full_text*
(
&%346 = fadd double %345, -1.000000e-03
,double8B

	full_text

double %345
tgetelementptr8Ba
_
	full_textR
P
N%347 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %273, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %346, double* %347, align 8, !tbaa !8
,double8B

	full_text

double %346
.double*8B

	full_text

double* %347
}call8Bs
q
	full_textd
b
`%348 = tail call double @llvm.fmuladd.f64(double %288, double 2.040200e+01, double 1.000000e+00)
,double8B

	full_text

double %288
Bfadd8B8
6
	full_text)
'
%%349 = fadd double %348, 1.250000e-03
,double8B

	full_text

double %348
tgetelementptr8Ba
_
	full_textR
P
N%350 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %273, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %349, double* %350, align 8, !tbaa !8
,double8B

	full_text

double %349
.double*8B

	full_text

double* %350
Bfmul8B8
6
	full_text)
'
%%351 = fmul double %338, 1.020100e+01
,double8B

	full_text

double %338
Cfsub8B9
7
	full_text*
(
&%352 = fsub double -0.000000e+00, %351
,double8B

	full_text

double %351
ucall8Bk
i
	full_text\
Z
X%353 = tail call double @llvm.fmuladd.f64(double %328, double 5.050000e-02, double %352)
,double8B

	full_text

double %328
,double8B

	full_text

double %352
tgetelementptr8Ba
_
	full_textR
P
N%354 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %273, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %353, double* %354, align 8, !tbaa !8
,double8B

	full_text

double %353
.double*8B

	full_text

double* %354
tgetelementptr8Ba
_
	full_textR
P
N%355 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %273, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %273
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %355, align 8, !tbaa !8
.double*8B

	full_text

double* %355
tgetelementptr8Ba
_
	full_textR
P
N%356 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %273, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %273
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %356, align 8, !tbaa !8
.double*8B

	full_text

double* %356
vcall8Bl
j
	full_text]
[
Y%357 = tail call double @llvm.fmuladd.f64(double %271, double -5.050000e-02, double %346)
,double8B

	full_text

double %271
,double8B

	full_text

double %346
tgetelementptr8Ba
_
	full_textR
P
N%358 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %273, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %357, double* %358, align 8, !tbaa !8
,double8B

	full_text

double %357
.double*8B

	full_text

double* %358
tgetelementptr8Ba
_
	full_textR
P
N%359 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %273, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %349, double* %359, align 8, !tbaa !8
,double8B

	full_text

double %349
.double*8B

	full_text

double* %359
ucall8Bk
i
	full_text\
Z
X%360 = tail call double @llvm.fmuladd.f64(double %341, double 5.050000e-02, double %353)
,double8B

	full_text

double %341
,double8B

	full_text

double %353
tgetelementptr8Ba
_
	full_textR
P
N%361 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %273, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %360, double* %361, align 8, !tbaa !8
,double8B

	full_text

double %360
.double*8B

	full_text

double* %361
tgetelementptr8Ba
_
	full_textR
P
N%362 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %273, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %273
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %362, align 8, !tbaa !8
.double*8B

	full_text

double* %362
tgetelementptr8Ba
_
	full_textR
P
N%363 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %273, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %273
Xstore8BM
K
	full_text>
<
:store double 2.500000e-04, double* %363, align 8, !tbaa !8
.double*8B

	full_text

double* %363
ucall8Bk
i
	full_text\
Z
X%364 = tail call double @llvm.fmuladd.f64(double %271, double 5.050000e-02, double %346)
,double8B

	full_text

double %271
,double8B

	full_text

double %346
tgetelementptr8Ba
_
	full_textR
P
N%365 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %273, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %364, double* %365, align 8, !tbaa !8
,double8B

	full_text

double %364
.double*8B

	full_text

double* %365
tgetelementptr8Ba
_
	full_textR
P
N%366 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %273, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %349, double* %366, align 8, !tbaa !8
,double8B

	full_text

double %349
.double*8B

	full_text

double* %366
vcall8Bl
j
	full_text]
[
Y%367 = tail call double @llvm.fmuladd.f64(double %341, double -5.050000e-02, double %353)
,double8B

	full_text

double %341
,double8B

	full_text

double %353
tgetelementptr8Ba
_
	full_textR
P
N%368 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %273, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %273
Pstore8BE
C
	full_text6
4
2store double %367, double* %368, align 8, !tbaa !8
,double8B

	full_text

double %367
.double*8B

	full_text

double* %368
tgetelementptr8Ba
_
	full_textR
P
N%369 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %273, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %273
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %369, align 8, !tbaa !8
.double*8B

	full_text

double* %369
Oload8BE
C
	full_text6
4
2%370 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
Oload8BE
C
	full_text6
4
2%371 = load double, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
Pload8BF
D
	full_text7
5
3%372 = load double, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
Pload8BF
D
	full_text7
5
3%373 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
getelementptr8B
‡
	full_textz
x
v%374 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 0, i64 %76
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Ibitcast8B<
:
	full_text-
+
)%375 = bitcast [5 x double]* %374 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %374
Jload8B@
>
	full_text1
/
-%376 = load i64, i64* %375, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %375
¦getelementptr8B’

	full_text

}%377 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 0, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%378 = bitcast double* %377 to i64*
.double*8B

	full_text

double* %377
Jload8B@
>
	full_text1
/
-%379 = load i64, i64* %378, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %378
¦getelementptr8B’

	full_text

}%380 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 0, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%381 = bitcast double* %380 to i64*
.double*8B

	full_text

double* %380
Jload8B@
>
	full_text1
/
-%382 = load i64, i64* %381, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %381
getelementptr8B
‡
	full_textz
x
v%383 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 1, i64 %76
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Ibitcast8B<
:
	full_text-
+
)%384 = bitcast [5 x double]* %383 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %383
Jload8B@
>
	full_text1
/
-%385 = load i64, i64* %384, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %384
¦getelementptr8B’

	full_text

}%386 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 1, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%387 = bitcast double* %386 to i64*
.double*8B

	full_text

double* %386
Jload8B@
>
	full_text1
/
-%388 = load i64, i64* %387, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %387
¦getelementptr8B’

	full_text

}%389 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 1, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%390 = bitcast double* %389 to i64*
.double*8B

	full_text

double* %389
Jload8B@
>
	full_text1
/
-%391 = load i64, i64* %390, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %390
7icmp8B-
+
	full_text

%392 = icmp slt i32 %12, 3
=br8B5
3
	full_text&
$
"br i1 %392, label %461, label %393
$i18B

	full_text
	
i1 %392
8zext8B.
,
	full_text

%394 = zext i32 %272 to i64
&i328B

	full_text


i32 %272
(br8B 

	full_text

br label %395
Iphi8B@
>
	full_text1
/
-%396 = phi i64 [ %460, %395 ], [ %391, %393 ]
&i648B

	full_text


i64 %460
&i648B

	full_text


i64 %391
Iphi8B@
>
	full_text1
/
-%397 = phi i64 [ %459, %395 ], [ %388, %393 ]
&i648B

	full_text


i64 %459
&i648B

	full_text


i64 %388
Iphi8B@
>
	full_text1
/
-%398 = phi i64 [ %458, %395 ], [ %385, %393 ]
&i648B

	full_text


i64 %458
&i648B

	full_text


i64 %385
Iphi8B@
>
	full_text1
/
-%399 = phi i64 [ %457, %395 ], [ %382, %393 ]
&i648B

	full_text


i64 %457
&i648B

	full_text


i64 %382
Iphi8B@
>
	full_text1
/
-%400 = phi i64 [ %456, %395 ], [ %379, %393 ]
&i648B

	full_text


i64 %456
&i648B

	full_text


i64 %379
Iphi8B@
>
	full_text1
/
-%401 = phi i64 [ %455, %395 ], [ %376, %393 ]
&i648B

	full_text


i64 %455
&i648B

	full_text


i64 %376
Fphi8B=
;
	full_text.
,
*%402 = phi i64 [ %407, %395 ], [ 0, %393 ]
&i648B

	full_text


i64 %407
Lphi8BC
A
	full_text4
2
0%403 = phi double [ %452, %395 ], [ %371, %393 ]
,double8B

	full_text

double %452
,double8B

	full_text

double %371
Lphi8BC
A
	full_text4
2
0%404 = phi double [ %426, %395 ], [ %370, %393 ]
,double8B

	full_text

double %426
,double8B

	full_text

double %370
Lphi8BC
A
	full_text4
2
0%405 = phi double [ %453, %395 ], [ %373, %393 ]
,double8B

	full_text

double %453
,double8B

	full_text

double %373
Lphi8BC
A
	full_text4
2
0%406 = phi double [ %448, %395 ], [ %372, %393 ]
,double8B

	full_text

double %448
,double8B

	full_text

double %372
:add8B1
/
	full_text"
 
%407 = add nuw nsw i64 %402, 1
&i648B

	full_text


i64 %402
tgetelementptr8Ba
_
	full_textR
P
N%408 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %402, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %402
Pload8BF
D
	full_text7
5
3%409 = load double, double* %408, align 8, !tbaa !8
.double*8B

	full_text

double* %408
Bfdiv8B8
6
	full_text)
'
%%410 = fdiv double 1.000000e+00, %404
,double8B

	full_text

double %404
:fmul8B0
.
	full_text!

%411 = fmul double %410, %403
,double8B

	full_text

double %410
,double8B

	full_text

double %403
tgetelementptr8Ba
_
	full_textR
P
N%412 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %402, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %402
Pstore8BE
C
	full_text6
4
2store double %411, double* %412, align 8, !tbaa !8
,double8B

	full_text

double %411
.double*8B

	full_text

double* %412
:fmul8B0
.
	full_text!

%413 = fmul double %410, %409
,double8B

	full_text

double %410
,double8B

	full_text

double %409
Pstore8BE
C
	full_text6
4
2store double %413, double* %408, align 8, !tbaa !8
,double8B

	full_text

double %413
.double*8B

	full_text

double* %408
Abitcast8B4
2
	full_text%
#
!%414 = bitcast i64 %401 to double
&i648B

	full_text


i64 %401
:fmul8B0
.
	full_text!

%415 = fmul double %410, %414
,double8B

	full_text

double %410
,double8B

	full_text

double %414
«getelementptr8B—
”
	full_text†
ƒ
€%416 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %402, i64 %76, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %402
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %415, double* %416, align 8, !tbaa !8
,double8B

	full_text

double %415
.double*8B

	full_text

double* %416
Abitcast8B4
2
	full_text%
#
!%417 = bitcast i64 %400 to double
&i648B

	full_text


i64 %400
:fmul8B0
.
	full_text!

%418 = fmul double %410, %417
,double8B

	full_text

double %410
,double8B

	full_text

double %417
«getelementptr8B—
”
	full_text†
ƒ
€%419 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %402, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %402
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %418, double* %419, align 8, !tbaa !8
,double8B

	full_text

double %418
.double*8B

	full_text

double* %419
Abitcast8B4
2
	full_text%
#
!%420 = bitcast i64 %399 to double
&i648B

	full_text


i64 %399
:fmul8B0
.
	full_text!

%421 = fmul double %410, %420
,double8B

	full_text

double %410
,double8B

	full_text

double %420
«getelementptr8B—
”
	full_text†
ƒ
€%422 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %402, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %402
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %421, double* %422, align 8, !tbaa !8
,double8B

	full_text

double %421
.double*8B

	full_text

double* %422
tgetelementptr8Ba
_
	full_textR
P
N%423 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %407, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %407
Pload8BF
D
	full_text7
5
3%424 = load double, double* %423, align 8, !tbaa !8
.double*8B

	full_text

double* %423
Cfsub8B9
7
	full_text*
(
&%425 = fsub double -0.000000e+00, %406
,double8B

	full_text

double %406
mcall8Bc
a
	full_textT
R
P%426 = tail call double @llvm.fmuladd.f64(double %425, double %411, double %405)
,double8B

	full_text

double %425
,double8B

	full_text

double %411
,double8B

	full_text

double %405
tgetelementptr8Ba
_
	full_textR
P
N%427 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %407, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %407
Pstore8BE
C
	full_text6
4
2store double %426, double* %427, align 8, !tbaa !8
,double8B

	full_text

double %426
.double*8B

	full_text

double* %427
Abitcast8B4
2
	full_text%
#
!%428 = bitcast i64 %398 to double
&i648B

	full_text


i64 %398
mcall8Bc
a
	full_textT
R
P%429 = tail call double @llvm.fmuladd.f64(double %425, double %415, double %428)
,double8B

	full_text

double %425
,double8B

	full_text

double %415
,double8B

	full_text

double %428
Abitcast8B4
2
	full_text%
#
!%430 = bitcast i64 %397 to double
&i648B

	full_text


i64 %397
mcall8Bc
a
	full_textT
R
P%431 = tail call double @llvm.fmuladd.f64(double %425, double %418, double %430)
,double8B

	full_text

double %425
,double8B

	full_text

double %418
,double8B

	full_text

double %430
Abitcast8B4
2
	full_text%
#
!%432 = bitcast i64 %396 to double
&i648B

	full_text


i64 %396
mcall8Bc
a
	full_textT
R
P%433 = tail call double @llvm.fmuladd.f64(double %425, double %421, double %432)
,double8B

	full_text

double %425
,double8B

	full_text

double %421
,double8B

	full_text

double %432
:add8B1
/
	full_text"
 
%434 = add nuw nsw i64 %402, 2
&i648B

	full_text


i64 %402
tgetelementptr8Ba
_
	full_textR
P
N%435 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %434, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%436 = load double, double* %435, align 8, !tbaa !8
.double*8B

	full_text

double* %435
tgetelementptr8Ba
_
	full_textR
P
N%437 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %434, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%438 = load double, double* %437, align 8, !tbaa !8
.double*8B

	full_text

double* %437
tgetelementptr8Ba
_
	full_textR
P
N%439 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %434, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%440 = load double, double* %439, align 8, !tbaa !8
.double*8B

	full_text

double* %439
«getelementptr8B—
”
	full_text†
ƒ
€%441 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %434, i64 %76, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %434
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%442 = load double, double* %441, align 8, !tbaa !8
.double*8B

	full_text

double* %441
«getelementptr8B—
”
	full_text†
ƒ
€%443 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %434, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %434
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%444 = load double, double* %443, align 8, !tbaa !8
.double*8B

	full_text

double* %443
«getelementptr8B—
”
	full_text†
ƒ
€%445 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %434, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %434
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%446 = load double, double* %445, align 8, !tbaa !8
.double*8B

	full_text

double* %445
Cfsub8B9
7
	full_text*
(
&%447 = fsub double -0.000000e+00, %436
,double8B

	full_text

double %436
mcall8Bc
a
	full_textT
R
P%448 = tail call double @llvm.fmuladd.f64(double %447, double %411, double %438)
,double8B

	full_text

double %447
,double8B

	full_text

double %411
,double8B

	full_text

double %438
Pstore8BE
C
	full_text6
4
2store double %448, double* %437, align 8, !tbaa !8
,double8B

	full_text

double %448
.double*8B

	full_text

double* %437
mcall8Bc
a
	full_textT
R
P%449 = tail call double @llvm.fmuladd.f64(double %447, double %415, double %442)
,double8B

	full_text

double %447
,double8B

	full_text

double %415
,double8B

	full_text

double %442
mcall8Bc
a
	full_textT
R
P%450 = tail call double @llvm.fmuladd.f64(double %447, double %418, double %444)
,double8B

	full_text

double %447
,double8B

	full_text

double %418
,double8B

	full_text

double %444
mcall8Bc
a
	full_textT
R
P%451 = tail call double @llvm.fmuladd.f64(double %447, double %421, double %446)
,double8B

	full_text

double %447
,double8B

	full_text

double %421
,double8B

	full_text

double %446
mcall8Bc
a
	full_textT
R
P%452 = tail call double @llvm.fmuladd.f64(double %425, double %413, double %424)
,double8B

	full_text

double %425
,double8B

	full_text

double %413
,double8B

	full_text

double %424
mcall8Bc
a
	full_textT
R
P%453 = tail call double @llvm.fmuladd.f64(double %447, double %413, double %440)
,double8B

	full_text

double %447
,double8B

	full_text

double %413
,double8B

	full_text

double %440
:icmp8B0
.
	full_text!

%454 = icmp eq i64 %407, %394
&i648B

	full_text


i64 %407
&i648B

	full_text


i64 %394
Abitcast8B4
2
	full_text%
#
!%455 = bitcast double %429 to i64
,double8B

	full_text

double %429
Abitcast8B4
2
	full_text%
#
!%456 = bitcast double %431 to i64
,double8B

	full_text

double %431
Abitcast8B4
2
	full_text%
#
!%457 = bitcast double %433 to i64
,double8B

	full_text

double %433
Abitcast8B4
2
	full_text%
#
!%458 = bitcast double %449 to i64
,double8B

	full_text

double %449
Abitcast8B4
2
	full_text%
#
!%459 = bitcast double %450 to i64
,double8B

	full_text

double %450
Abitcast8B4
2
	full_text%
#
!%460 = bitcast double %451 to i64
,double8B

	full_text

double %451
=br8B5
3
	full_text&
$
"br i1 %454, label %461, label %395
$i18B

	full_text
	
i1 %454
Iphi8B@
>
	full_text1
/
-%462 = phi i64 [ %391, %265 ], [ %460, %395 ]
&i648B

	full_text


i64 %391
&i648B

	full_text


i64 %460
Iphi8B@
>
	full_text1
/
-%463 = phi i64 [ %388, %265 ], [ %459, %395 ]
&i648B

	full_text


i64 %388
&i648B

	full_text


i64 %459
Iphi8B@
>
	full_text1
/
-%464 = phi i64 [ %385, %265 ], [ %458, %395 ]
&i648B

	full_text


i64 %385
&i648B

	full_text


i64 %458
Iphi8B@
>
	full_text1
/
-%465 = phi i64 [ %382, %265 ], [ %457, %395 ]
&i648B

	full_text


i64 %382
&i648B

	full_text


i64 %457
Iphi8B@
>
	full_text1
/
-%466 = phi i64 [ %379, %265 ], [ %456, %395 ]
&i648B

	full_text


i64 %379
&i648B

	full_text


i64 %456
Iphi8B@
>
	full_text1
/
-%467 = phi i64 [ %376, %265 ], [ %455, %395 ]
&i648B

	full_text


i64 %376
&i648B

	full_text


i64 %455
Lphi8BC
A
	full_text4
2
0%468 = phi double [ %372, %265 ], [ %448, %395 ]
,double8B

	full_text

double %372
,double8B

	full_text

double %448
Lphi8BC
A
	full_text4
2
0%469 = phi double [ %373, %265 ], [ %453, %395 ]
,double8B

	full_text

double %373
,double8B

	full_text

double %453
Lphi8BC
A
	full_text4
2
0%470 = phi double [ %370, %265 ], [ %426, %395 ]
,double8B

	full_text

double %370
,double8B

	full_text

double %426
Lphi8BC
A
	full_text4
2
0%471 = phi double [ %371, %265 ], [ %452, %395 ]
,double8B

	full_text

double %371
,double8B

	full_text

double %452
Pload8BF
D
	full_text7
5
3%472 = load double, double* %355, align 8, !tbaa !8
.double*8B

	full_text

double* %355
Bfdiv8B8
6
	full_text)
'
%%473 = fdiv double 1.000000e+00, %470
,double8B

	full_text

double %470
:fmul8B0
.
	full_text!

%474 = fmul double %473, %471
,double8B

	full_text

double %473
,double8B

	full_text

double %471
Pstore8BE
C
	full_text6
4
2store double %474, double* %354, align 8, !tbaa !8
,double8B

	full_text

double %474
.double*8B

	full_text

double* %354
:fmul8B0
.
	full_text!

%475 = fmul double %473, %472
,double8B

	full_text

double %473
,double8B

	full_text

double %472
Pstore8BE
C
	full_text6
4
2store double %475, double* %355, align 8, !tbaa !8
,double8B

	full_text

double %475
.double*8B

	full_text

double* %355
Abitcast8B4
2
	full_text%
#
!%476 = bitcast i64 %467 to double
&i648B

	full_text


i64 %467
:fmul8B0
.
	full_text!

%477 = fmul double %473, %476
,double8B

	full_text

double %473
,double8B

	full_text

double %476
«getelementptr8B—
”
	full_text†
ƒ
€%478 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %273, i64 %76, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %273
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %477, double* %478, align 8, !tbaa !8
,double8B

	full_text

double %477
.double*8B

	full_text

double* %478
Abitcast8B4
2
	full_text%
#
!%479 = bitcast i64 %466 to double
&i648B

	full_text


i64 %466
:fmul8B0
.
	full_text!

%480 = fmul double %473, %479
,double8B

	full_text

double %473
,double8B

	full_text

double %479
«getelementptr8B—
”
	full_text†
ƒ
€%481 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %273, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %273
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %480, double* %481, align 8, !tbaa !8
,double8B

	full_text

double %480
.double*8B

	full_text

double* %481
Abitcast8B4
2
	full_text%
#
!%482 = bitcast i64 %465 to double
&i648B

	full_text


i64 %465
:fmul8B0
.
	full_text!

%483 = fmul double %473, %482
,double8B

	full_text

double %473
,double8B

	full_text

double %482
«getelementptr8B—
”
	full_text†
ƒ
€%484 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %273, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %273
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %483, double* %484, align 8, !tbaa !8
,double8B

	full_text

double %483
.double*8B

	full_text

double* %484
tgetelementptr8Ba
_
	full_textR
P
N%485 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %323, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %323
Pload8BF
D
	full_text7
5
3%486 = load double, double* %485, align 8, !tbaa !8
.double*8B

	full_text

double* %485
Cfsub8B9
7
	full_text*
(
&%487 = fsub double -0.000000e+00, %468
,double8B

	full_text

double %468
mcall8Bc
a
	full_textT
R
P%488 = tail call double @llvm.fmuladd.f64(double %487, double %474, double %469)
,double8B

	full_text

double %487
,double8B

	full_text

double %474
,double8B

	full_text

double %469
tgetelementptr8Ba
_
	full_textR
P
N%489 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %323, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %323
Pstore8BE
C
	full_text6
4
2store double %488, double* %489, align 8, !tbaa !8
,double8B

	full_text

double %488
.double*8B

	full_text

double* %489
mcall8Bc
a
	full_textT
R
P%490 = tail call double @llvm.fmuladd.f64(double %487, double %475, double %486)
,double8B

	full_text

double %487
,double8B

	full_text

double %475
,double8B

	full_text

double %486
Pstore8BE
C
	full_text6
4
2store double %490, double* %485, align 8, !tbaa !8
,double8B

	full_text

double %490
.double*8B

	full_text

double* %485
Abitcast8B4
2
	full_text%
#
!%491 = bitcast i64 %464 to double
&i648B

	full_text


i64 %464
mcall8Bc
a
	full_textT
R
P%492 = tail call double @llvm.fmuladd.f64(double %487, double %477, double %491)
,double8B

	full_text

double %487
,double8B

	full_text

double %477
,double8B

	full_text

double %491
Abitcast8B4
2
	full_text%
#
!%493 = bitcast i64 %463 to double
&i648B

	full_text


i64 %463
mcall8Bc
a
	full_textT
R
P%494 = tail call double @llvm.fmuladd.f64(double %487, double %480, double %493)
,double8B

	full_text

double %487
,double8B

	full_text

double %480
,double8B

	full_text

double %493
Abitcast8B4
2
	full_text%
#
!%495 = bitcast i64 %462 to double
&i648B

	full_text


i64 %462
mcall8Bc
a
	full_textT
R
P%496 = tail call double @llvm.fmuladd.f64(double %487, double %483, double %495)
,double8B

	full_text

double %487
,double8B

	full_text

double %483
,double8B

	full_text

double %495
Bfdiv8B8
6
	full_text)
'
%%497 = fdiv double 1.000000e+00, %488
,double8B

	full_text

double %488
:fmul8B0
.
	full_text!

%498 = fmul double %497, %492
,double8B

	full_text

double %497
,double8B

	full_text

double %492
«getelementptr8B—
”
	full_text†
ƒ
€%499 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %323, i64 %76, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %323
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %498, double* %499, align 8, !tbaa !8
,double8B

	full_text

double %498
.double*8B

	full_text

double* %499
:fmul8B0
.
	full_text!

%500 = fmul double %497, %494
,double8B

	full_text

double %497
,double8B

	full_text

double %494
«getelementptr8B—
”
	full_text†
ƒ
€%501 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %323, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %323
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %500, double* %501, align 8, !tbaa !8
,double8B

	full_text

double %500
.double*8B

	full_text

double* %501
:fmul8B0
.
	full_text!

%502 = fmul double %497, %496
,double8B

	full_text

double %497
,double8B

	full_text

double %496
«getelementptr8B—
”
	full_text†
ƒ
€%503 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %323, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %323
%i648B

	full_text
	
i64 %76
Pstore8BE
C
	full_text6
4
2store double %502, double* %503, align 8, !tbaa !8
,double8B

	full_text

double %502
.double*8B

	full_text

double* %503
Oload8BE
C
	full_text6
4
2%504 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
Oload8BE
C
	full_text6
4
2%505 = load double, double* %59, align 8, !tbaa !8
-double*8B

	full_text

double* %59
Pload8BF
D
	full_text7
5
3%506 = load double, double* %145, align 8, !tbaa !8
.double*8B

	full_text

double* %145
Pload8BF
D
	full_text7
5
3%507 = load double, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
Oload8BE
C
	full_text6
4
2%508 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
Oload8BE
C
	full_text6
4
2%509 = load double, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
Pload8BF
D
	full_text7
5
3%510 = load double, double* %152, align 8, !tbaa !8
.double*8B

	full_text

double* %152
Pload8BF
D
	full_text7
5
3%511 = load double, double* %153, align 8, !tbaa !8
.double*8B

	full_text

double* %153
¦getelementptr8B’

	full_text

}%512 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 0, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%513 = bitcast double* %512 to i64*
.double*8B

	full_text

double* %512
Jload8B@
>
	full_text1
/
-%514 = load i64, i64* %513, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %513
¦getelementptr8B’

	full_text

}%515 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 0, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%516 = bitcast double* %515 to i64*
.double*8B

	full_text

double* %515
Jload8B@
>
	full_text1
/
-%517 = load i64, i64* %516, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %516
¦getelementptr8B’

	full_text

}%518 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 1, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%519 = bitcast double* %518 to i64*
.double*8B

	full_text

double* %518
Jload8B@
>
	full_text1
/
-%520 = load i64, i64* %519, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %519
¦getelementptr8B’

	full_text

}%521 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 1, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
%i648B

	full_text
	
i64 %76
Cbitcast8B6
4
	full_text'
%
#%522 = bitcast double* %521 to i64*
.double*8B

	full_text

double* %521
Jload8B@
>
	full_text1
/
-%523 = load i64, i64* %522, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %522
=br8B5
3
	full_text&
$
"br i1 %392, label %605, label %524
$i18B

	full_text
	
i1 %392
8zext8B.
,
	full_text

%525 = zext i32 %272 to i64
&i328B

	full_text


i32 %272
(br8B 

	full_text

br label %526
Iphi8	B@
>
	full_text1
/
-%527 = phi i64 [ %604, %526 ], [ %523, %524 ]
&i648	B

	full_text


i64 %604
&i648	B

	full_text


i64 %523
Iphi8	B@
>
	full_text1
/
-%528 = phi i64 [ %603, %526 ], [ %517, %524 ]
&i648	B

	full_text


i64 %603
&i648	B

	full_text


i64 %517
Iphi8	B@
>
	full_text1
/
-%529 = phi i64 [ %602, %526 ], [ %520, %524 ]
&i648	B

	full_text


i64 %602
&i648	B

	full_text


i64 %520
Iphi8	B@
>
	full_text1
/
-%530 = phi i64 [ %601, %526 ], [ %514, %524 ]
&i648	B

	full_text


i64 %601
&i648	B

	full_text


i64 %514
Fphi8	B=
;
	full_text.
,
*%531 = phi i64 [ %540, %526 ], [ 0, %524 ]
&i648	B

	full_text


i64 %540
Lphi8	BC
A
	full_text4
2
0%532 = phi double [ %597, %526 ], [ %510, %524 ]
,double8	B

	full_text

double %597
,double8	B

	full_text

double %510
Lphi8	BC
A
	full_text4
2
0%533 = phi double [ %598, %526 ], [ %511, %524 ]
,double8	B

	full_text

double %598
,double8	B

	full_text

double %511
Lphi8	BC
A
	full_text4
2
0%534 = phi double [ %583, %526 ], [ %508, %524 ]
,double8	B

	full_text

double %583
,double8	B

	full_text

double %508
Lphi8	BC
A
	full_text4
2
0%535 = phi double [ %585, %526 ], [ %509, %524 ]
,double8	B

	full_text

double %585
,double8	B

	full_text

double %509
Lphi8	BC
A
	full_text4
2
0%536 = phi double [ %568, %526 ], [ %506, %524 ]
,double8	B

	full_text

double %568
,double8	B

	full_text

double %506
Lphi8	BC
A
	full_text4
2
0%537 = phi double [ %569, %526 ], [ %507, %524 ]
,double8	B

	full_text

double %569
,double8	B

	full_text

double %507
Lphi8	BC
A
	full_text4
2
0%538 = phi double [ %554, %526 ], [ %504, %524 ]
,double8	B

	full_text

double %554
,double8	B

	full_text

double %504
Lphi8	BC
A
	full_text4
2
0%539 = phi double [ %556, %526 ], [ %505, %524 ]
,double8	B

	full_text

double %556
,double8	B

	full_text

double %505
:add8	B1
/
	full_text"
 
%540 = add nuw nsw i64 %531, 1
&i648	B

	full_text


i64 %531
:add8	B1
/
	full_text"
 
%541 = add nuw nsw i64 %531, 2
&i648	B

	full_text


i64 %531
tgetelementptr8	Ba
_
	full_textR
P
N%542 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %531, i64 4
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %531
Pload8	BF
D
	full_text7
5
3%543 = load double, double* %542, align 8, !tbaa !8
.double*8	B

	full_text

double* %542
Bfdiv8	B8
6
	full_text)
'
%%544 = fdiv double 1.000000e+00, %538
,double8	B

	full_text

double %538
:fmul8	B0
.
	full_text!

%545 = fmul double %539, %544
,double8	B

	full_text

double %539
,double8	B

	full_text

double %544
tgetelementptr8	Ba
_
	full_textR
P
N%546 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %531, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %531
Pstore8	BE
C
	full_text6
4
2store double %545, double* %546, align 8, !tbaa !8
,double8	B

	full_text

double %545
.double*8	B

	full_text

double* %546
:fmul8	B0
.
	full_text!

%547 = fmul double %544, %543
,double8	B

	full_text

double %544
,double8	B

	full_text

double %543
Pstore8	BE
C
	full_text6
4
2store double %547, double* %542, align 8, !tbaa !8
,double8	B

	full_text

double %547
.double*8	B

	full_text

double* %542
Abitcast8	B4
2
	full_text%
#
!%548 = bitcast i64 %530 to double
&i648	B

	full_text


i64 %530
:fmul8	B0
.
	full_text!

%549 = fmul double %544, %548
,double8	B

	full_text

double %544
,double8	B

	full_text

double %548
«getelementptr8	B—
”
	full_text†
ƒ
€%550 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %531, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8	B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648	B

	full_text
	
i64 %74
&i648	B

	full_text


i64 %531
%i648	B

	full_text
	
i64 %76
Pstore8	BE
C
	full_text6
4
2store double %549, double* %550, align 8, !tbaa !8
,double8	B

	full_text

double %549
.double*8	B

	full_text

double* %550
tgetelementptr8	Ba
_
	full_textR
P
N%551 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %540, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %540
Pload8	BF
D
	full_text7
5
3%552 = load double, double* %551, align 8, !tbaa !8
.double*8	B

	full_text

double* %551
Cfsub8	B9
7
	full_text*
(
&%553 = fsub double -0.000000e+00, %536
,double8	B

	full_text

double %536
mcall8	Bc
a
	full_textT
R
P%554 = tail call double @llvm.fmuladd.f64(double %553, double %545, double %537)
,double8	B

	full_text

double %553
,double8	B

	full_text

double %545
,double8	B

	full_text

double %537
tgetelementptr8	Ba
_
	full_textR
P
N%555 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %540, i64 2
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %540
Pstore8	BE
C
	full_text6
4
2store double %554, double* %555, align 8, !tbaa !8
,double8	B

	full_text

double %554
.double*8	B

	full_text

double* %555
mcall8	Bc
a
	full_textT
R
P%556 = tail call double @llvm.fmuladd.f64(double %553, double %547, double %552)
,double8	B

	full_text

double %553
,double8	B

	full_text

double %547
,double8	B

	full_text

double %552
Abitcast8	B4
2
	full_text%
#
!%557 = bitcast i64 %529 to double
&i648	B

	full_text


i64 %529
mcall8	Bc
a
	full_textT
R
P%558 = tail call double @llvm.fmuladd.f64(double %553, double %549, double %557)
,double8	B

	full_text

double %553
,double8	B

	full_text

double %549
,double8	B

	full_text

double %557
tgetelementptr8	Ba
_
	full_textR
P
N%559 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %541, i64 0
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %541
Pload8	BF
D
	full_text7
5
3%560 = load double, double* %559, align 8, !tbaa !8
.double*8	B

	full_text

double* %559
tgetelementptr8	Ba
_
	full_textR
P
N%561 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %541, i64 1
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %541
Pload8	BF
D
	full_text7
5
3%562 = load double, double* %561, align 8, !tbaa !8
.double*8	B

	full_text

double* %561
tgetelementptr8	Ba
_
	full_textR
P
N%563 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %541, i64 2
9[5 x double]*8	B$
"
	full_text

[5 x double]* %38
&i648	B

	full_text


i64 %541
Pload8	BF
D
	full_text7
5
3%564 = load double, double* %563, align 8, !tbaa !8
.double*8	B

	full_text

double* %563
«getelementptr8	B—
”
	full_text†
ƒ
€%565 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %541, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8	B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648	B

	full_text
	
i64 %74
&i648	B

	full_text


i64 %541
%i648	B

	full_text
	
i64 %76
Pload8	BF
D
	full_text7
5
3%566 = load double, double* %565, align 8, !tbaa !8
.double*8	B

	full_text

double* %565
Cfsub8	B9
7
	full_text*
(
&%567 = fsub double -0.000000e+00, %560
,double8	B

	full_text

double %560
mcall8	Bc
a
	full_textT
R
P%568 = tail call double @llvm.fmuladd.f64(double %567, double %545, double %562)
,double8	B

	full_text

double %567
,double8	B

	full_text

double %545
,double8	B

	full_text

double %562
Pstore8	BE
C
	full_text6
4
2store double %568, double* %561, align 8, !tbaa !8
,double8	B

	full_text

double %568
.double*8	B

	full_text

double* %561
mcall8	Bc
a
	full_textT
R
P%569 = tail call double @llvm.fmuladd.f64(double %567, double %547, double %564)
,double8	B

	full_text

double %567
,double8	B

	full_text

double %547
,double8	B

	full_text

double %564
mcall8	Bc
a
	full_textT
R
P%570 = tail call double @llvm.fmuladd.f64(double %567, double %549, double %566)
,double8	B

	full_text

double %567
,double8	B

	full_text

double %549
,double8	B

	full_text

double %566
tgetelementptr8	Ba
_
	full_textR
P
N%571 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %531, i64 4
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %531
Pload8	BF
D
	full_text7
5
3%572 = load double, double* %571, align 8, !tbaa !8
.double*8	B

	full_text

double* %571
Bfdiv8	B8
6
	full_text)
'
%%573 = fdiv double 1.000000e+00, %534
,double8	B

	full_text

double %534
:fmul8	B0
.
	full_text!

%574 = fmul double %535, %573
,double8	B

	full_text

double %535
,double8	B

	full_text

double %573
tgetelementptr8	Ba
_
	full_textR
P
N%575 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %531, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %531
Pstore8	BE
C
	full_text6
4
2store double %574, double* %575, align 8, !tbaa !8
,double8	B

	full_text

double %574
.double*8	B

	full_text

double* %575
:fmul8	B0
.
	full_text!

%576 = fmul double %573, %572
,double8	B

	full_text

double %573
,double8	B

	full_text

double %572
Pstore8	BE
C
	full_text6
4
2store double %576, double* %571, align 8, !tbaa !8
,double8	B

	full_text

double %576
.double*8	B

	full_text

double* %571
Abitcast8	B4
2
	full_text%
#
!%577 = bitcast i64 %528 to double
&i648	B

	full_text


i64 %528
:fmul8	B0
.
	full_text!

%578 = fmul double %573, %577
,double8	B

	full_text

double %573
,double8	B

	full_text

double %577
«getelementptr8	B—
”
	full_text†
ƒ
€%579 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %531, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8	B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648	B

	full_text
	
i64 %74
&i648	B

	full_text


i64 %531
%i648	B

	full_text
	
i64 %76
Pstore8	BE
C
	full_text6
4
2store double %578, double* %579, align 8, !tbaa !8
,double8	B

	full_text

double %578
.double*8	B

	full_text

double* %579
tgetelementptr8	Ba
_
	full_textR
P
N%580 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %540, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %540
Pload8	BF
D
	full_text7
5
3%581 = load double, double* %580, align 8, !tbaa !8
.double*8	B

	full_text

double* %580
Cfsub8	B9
7
	full_text*
(
&%582 = fsub double -0.000000e+00, %532
,double8	B

	full_text

double %532
mcall8	Bc
a
	full_textT
R
P%583 = tail call double @llvm.fmuladd.f64(double %582, double %574, double %533)
,double8	B

	full_text

double %582
,double8	B

	full_text

double %574
,double8	B

	full_text

double %533
tgetelementptr8	Ba
_
	full_textR
P
N%584 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %540, i64 2
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %540
Pstore8	BE
C
	full_text6
4
2store double %583, double* %584, align 8, !tbaa !8
,double8	B

	full_text

double %583
.double*8	B

	full_text

double* %584
mcall8	Bc
a
	full_textT
R
P%585 = tail call double @llvm.fmuladd.f64(double %582, double %576, double %581)
,double8	B

	full_text

double %582
,double8	B

	full_text

double %576
,double8	B

	full_text

double %581
Abitcast8	B4
2
	full_text%
#
!%586 = bitcast i64 %527 to double
&i648	B

	full_text


i64 %527
mcall8	Bc
a
	full_textT
R
P%587 = tail call double @llvm.fmuladd.f64(double %582, double %578, double %586)
,double8	B

	full_text

double %582
,double8	B

	full_text

double %578
,double8	B

	full_text

double %586
tgetelementptr8	Ba
_
	full_textR
P
N%588 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %541, i64 0
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %541
Pload8	BF
D
	full_text7
5
3%589 = load double, double* %588, align 8, !tbaa !8
.double*8	B

	full_text

double* %588
tgetelementptr8	Ba
_
	full_textR
P
N%590 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %541, i64 1
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %541
Pload8	BF
D
	full_text7
5
3%591 = load double, double* %590, align 8, !tbaa !8
.double*8	B

	full_text

double* %590
tgetelementptr8	Ba
_
	full_textR
P
N%592 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %541, i64 2
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %541
Pload8	BF
D
	full_text7
5
3%593 = load double, double* %592, align 8, !tbaa !8
.double*8	B

	full_text

double* %592
«getelementptr8	B—
”
	full_text†
ƒ
€%594 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %541, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8	B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648	B

	full_text
	
i64 %74
&i648	B

	full_text


i64 %541
%i648	B

	full_text
	
i64 %76
Pload8	BF
D
	full_text7
5
3%595 = load double, double* %594, align 8, !tbaa !8
.double*8	B

	full_text

double* %594
Cfsub8	B9
7
	full_text*
(
&%596 = fsub double -0.000000e+00, %589
,double8	B

	full_text

double %589
mcall8	Bc
a
	full_textT
R
P%597 = tail call double @llvm.fmuladd.f64(double %596, double %574, double %591)
,double8	B

	full_text

double %596
,double8	B

	full_text

double %574
,double8	B

	full_text

double %591
Pstore8	BE
C
	full_text6
4
2store double %597, double* %590, align 8, !tbaa !8
,double8	B

	full_text

double %597
.double*8	B

	full_text

double* %590
mcall8	Bc
a
	full_textT
R
P%598 = tail call double @llvm.fmuladd.f64(double %596, double %576, double %593)
,double8	B

	full_text

double %596
,double8	B

	full_text

double %576
,double8	B

	full_text

double %593
mcall8	Bc
a
	full_textT
R
P%599 = tail call double @llvm.fmuladd.f64(double %596, double %578, double %595)
,double8	B

	full_text

double %596
,double8	B

	full_text

double %578
,double8	B

	full_text

double %595
:icmp8	B0
.
	full_text!

%600 = icmp eq i64 %540, %525
&i648	B

	full_text


i64 %540
&i648	B

	full_text


i64 %525
Abitcast8	B4
2
	full_text%
#
!%601 = bitcast double %558 to i64
,double8	B

	full_text

double %558
Abitcast8	B4
2
	full_text%
#
!%602 = bitcast double %570 to i64
,double8	B

	full_text

double %570
Abitcast8	B4
2
	full_text%
#
!%603 = bitcast double %587 to i64
,double8	B

	full_text

double %587
Abitcast8	B4
2
	full_text%
#
!%604 = bitcast double %599 to i64
,double8	B

	full_text

double %599
=br8	B5
3
	full_text&
$
"br i1 %600, label %605, label %526
$i18	B

	full_text
	
i1 %600
Iphi8
B@
>
	full_text1
/
-%606 = phi i64 [ %523, %461 ], [ %604, %526 ]
&i648
B

	full_text


i64 %523
&i648
B

	full_text


i64 %604
Iphi8
B@
>
	full_text1
/
-%607 = phi i64 [ %517, %461 ], [ %603, %526 ]
&i648
B

	full_text


i64 %517
&i648
B

	full_text


i64 %603
Iphi8
B@
>
	full_text1
/
-%608 = phi i64 [ %520, %461 ], [ %602, %526 ]
&i648
B

	full_text


i64 %520
&i648
B

	full_text


i64 %602
Iphi8
B@
>
	full_text1
/
-%609 = phi i64 [ %514, %461 ], [ %601, %526 ]
&i648
B

	full_text


i64 %514
&i648
B

	full_text


i64 %601
Lphi8
BC
A
	full_text4
2
0%610 = phi double [ %505, %461 ], [ %556, %526 ]
,double8
B

	full_text

double %505
,double8
B

	full_text

double %556
Lphi8
BC
A
	full_text4
2
0%611 = phi double [ %504, %461 ], [ %554, %526 ]
,double8
B

	full_text

double %504
,double8
B

	full_text

double %554
Lphi8
BC
A
	full_text4
2
0%612 = phi double [ %507, %461 ], [ %569, %526 ]
,double8
B

	full_text

double %507
,double8
B

	full_text

double %569
Lphi8
BC
A
	full_text4
2
0%613 = phi double [ %506, %461 ], [ %568, %526 ]
,double8
B

	full_text

double %506
,double8
B

	full_text

double %568
Lphi8
BC
A
	full_text4
2
0%614 = phi double [ %509, %461 ], [ %585, %526 ]
,double8
B

	full_text

double %509
,double8
B

	full_text

double %585
Lphi8
BC
A
	full_text4
2
0%615 = phi double [ %508, %461 ], [ %583, %526 ]
,double8
B

	full_text

double %508
,double8
B

	full_text

double %583
Lphi8
BC
A
	full_text4
2
0%616 = phi double [ %511, %461 ], [ %598, %526 ]
,double8
B

	full_text

double %511
,double8
B

	full_text

double %598
Lphi8
BC
A
	full_text4
2
0%617 = phi double [ %510, %461 ], [ %597, %526 ]
,double8
B

	full_text

double %510
,double8
B

	full_text

double %597
Pload8
BF
D
	full_text7
5
3%618 = load double, double* %362, align 8, !tbaa !8
.double*8
B

	full_text

double* %362
Bfdiv8
B8
6
	full_text)
'
%%619 = fdiv double 1.000000e+00, %611
,double8
B

	full_text

double %611
:fmul8
B0
.
	full_text!

%620 = fmul double %610, %619
,double8
B

	full_text

double %610
,double8
B

	full_text

double %619
Pstore8
BE
C
	full_text6
4
2store double %620, double* %361, align 8, !tbaa !8
,double8
B

	full_text

double %620
.double*8
B

	full_text

double* %361
:fmul8
B0
.
	full_text!

%621 = fmul double %619, %618
,double8
B

	full_text

double %619
,double8
B

	full_text

double %618
Pstore8
BE
C
	full_text6
4
2store double %621, double* %362, align 8, !tbaa !8
,double8
B

	full_text

double %621
.double*8
B

	full_text

double* %362
Abitcast8
B4
2
	full_text%
#
!%622 = bitcast i64 %609 to double
&i648
B

	full_text


i64 %609
:fmul8
B0
.
	full_text!

%623 = fmul double %619, %622
,double8
B

	full_text

double %619
,double8
B

	full_text

double %622
«getelementptr8
B—
”
	full_text†
ƒ
€%624 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %273, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8
B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648
B

	full_text
	
i64 %74
&i648
B

	full_text


i64 %273
%i648
B

	full_text
	
i64 %76
Pstore8
BE
C
	full_text6
4
2store double %623, double* %624, align 8, !tbaa !8
,double8
B

	full_text

double %623
.double*8
B

	full_text

double* %624
tgetelementptr8
Ba
_
	full_textR
P
N%625 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %323, i64 3
9[5 x double]*8
B$
"
	full_text

[5 x double]* %38
&i648
B

	full_text


i64 %323
Pload8
BF
D
	full_text7
5
3%626 = load double, double* %625, align 8, !tbaa !8
.double*8
B

	full_text

double* %625
Cfsub8
B9
7
	full_text*
(
&%627 = fsub double -0.000000e+00, %613
,double8
B

	full_text

double %613
mcall8
Bc
a
	full_textT
R
P%628 = tail call double @llvm.fmuladd.f64(double %627, double %620, double %612)
,double8
B

	full_text

double %627
,double8
B

	full_text

double %620
,double8
B

	full_text

double %612
tgetelementptr8
Ba
_
	full_textR
P
N%629 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %323, i64 2
9[5 x double]*8
B$
"
	full_text

[5 x double]* %38
&i648
B

	full_text


i64 %323
Pstore8
BE
C
	full_text6
4
2store double %628, double* %629, align 8, !tbaa !8
,double8
B

	full_text

double %628
.double*8
B

	full_text

double* %629
mcall8
Bc
a
	full_textT
R
P%630 = tail call double @llvm.fmuladd.f64(double %627, double %621, double %626)
,double8
B

	full_text

double %627
,double8
B

	full_text

double %621
,double8
B

	full_text

double %626
Pstore8
BE
C
	full_text6
4
2store double %630, double* %625, align 8, !tbaa !8
,double8
B

	full_text

double %630
.double*8
B

	full_text

double* %625
Abitcast8
B4
2
	full_text%
#
!%631 = bitcast i64 %608 to double
&i648
B

	full_text


i64 %608
mcall8
Bc
a
	full_textT
R
P%632 = tail call double @llvm.fmuladd.f64(double %627, double %623, double %631)
,double8
B

	full_text

double %627
,double8
B

	full_text

double %623
,double8
B

	full_text

double %631
Pload8
BF
D
	full_text7
5
3%633 = load double, double* %369, align 8, !tbaa !8
.double*8
B

	full_text

double* %369
Bfdiv8
B8
6
	full_text)
'
%%634 = fdiv double 1.000000e+00, %615
,double8
B

	full_text

double %615
:fmul8
B0
.
	full_text!

%635 = fmul double %614, %634
,double8
B

	full_text

double %614
,double8
B

	full_text

double %634
Pstore8
BE
C
	full_text6
4
2store double %635, double* %368, align 8, !tbaa !8
,double8
B

	full_text

double %635
.double*8
B

	full_text

double* %368
:fmul8
B0
.
	full_text!

%636 = fmul double %634, %633
,double8
B

	full_text

double %634
,double8
B

	full_text

double %633
Pstore8
BE
C
	full_text6
4
2store double %636, double* %369, align 8, !tbaa !8
,double8
B

	full_text

double %636
.double*8
B

	full_text

double* %369
Abitcast8
B4
2
	full_text%
#
!%637 = bitcast i64 %607 to double
&i648
B

	full_text


i64 %607
:fmul8
B0
.
	full_text!

%638 = fmul double %634, %637
,double8
B

	full_text

double %634
,double8
B

	full_text

double %637
«getelementptr8
B—
”
	full_text†
ƒ
€%639 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %273, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8
B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648
B

	full_text
	
i64 %74
&i648
B

	full_text


i64 %273
%i648
B

	full_text
	
i64 %76
Pstore8
BE
C
	full_text6
4
2store double %638, double* %639, align 8, !tbaa !8
,double8
B

	full_text

double %638
.double*8
B

	full_text

double* %639
tgetelementptr8
Ba
_
	full_textR
P
N%640 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %323, i64 3
9[5 x double]*8
B$
"
	full_text

[5 x double]* %40
&i648
B

	full_text


i64 %323
Pload8
BF
D
	full_text7
5
3%641 = load double, double* %640, align 8, !tbaa !8
.double*8
B

	full_text

double* %640
Cfsub8
B9
7
	full_text*
(
&%642 = fsub double -0.000000e+00, %617
,double8
B

	full_text

double %617
mcall8
Bc
a
	full_textT
R
P%643 = tail call double @llvm.fmuladd.f64(double %642, double %635, double %616)
,double8
B

	full_text

double %642
,double8
B

	full_text

double %635
,double8
B

	full_text

double %616
tgetelementptr8
Ba
_
	full_textR
P
N%644 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %323, i64 2
9[5 x double]*8
B$
"
	full_text

[5 x double]* %40
&i648
B

	full_text


i64 %323
Pstore8
BE
C
	full_text6
4
2store double %643, double* %644, align 8, !tbaa !8
,double8
B

	full_text

double %643
.double*8
B

	full_text

double* %644
mcall8
Bc
a
	full_textT
R
P%645 = tail call double @llvm.fmuladd.f64(double %642, double %636, double %641)
,double8
B

	full_text

double %642
,double8
B

	full_text

double %636
,double8
B

	full_text

double %641
Pstore8
BE
C
	full_text6
4
2store double %645, double* %640, align 8, !tbaa !8
,double8
B

	full_text

double %645
.double*8
B

	full_text

double* %640
Abitcast8
B4
2
	full_text%
#
!%646 = bitcast i64 %606 to double
&i648
B

	full_text


i64 %606
mcall8
Bc
a
	full_textT
R
P%647 = tail call double @llvm.fmuladd.f64(double %642, double %638, double %646)
,double8
B

	full_text

double %642
,double8
B

	full_text

double %638
,double8
B

	full_text

double %646
:fdiv8
B0
.
	full_text!

%648 = fdiv double %632, %628
,double8
B

	full_text

double %632
,double8
B

	full_text

double %628
«getelementptr8
B—
”
	full_text†
ƒ
€%649 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %323, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8
B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648
B

	full_text
	
i64 %74
&i648
B

	full_text


i64 %323
%i648
B

	full_text
	
i64 %76
Pstore8
BE
C
	full_text6
4
2store double %648, double* %649, align 8, !tbaa !8
,double8
B

	full_text

double %648
.double*8
B

	full_text

double* %649
:fdiv8
B0
.
	full_text!

%650 = fdiv double %647, %643
,double8
B

	full_text

double %647
,double8
B

	full_text

double %643
«getelementptr8
B—
”
	full_text†
ƒ
€%651 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %323, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8
B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648
B

	full_text
	
i64 %74
&i648
B

	full_text


i64 %323
%i648
B

	full_text
	
i64 %76
Pstore8
BE
C
	full_text6
4
2store double %650, double* %651, align 8, !tbaa !8
,double8
B

	full_text

double %650
.double*8
B

	full_text

double* %651
Pload8
BF
D
	full_text7
5
3%652 = load double, double* %354, align 8, !tbaa !8
.double*8
B

	full_text

double* %354
Cfsub8
B9
7
	full_text*
(
&%653 = fsub double -0.000000e+00, %652
,double8
B

	full_text

double %652
Cbitcast8
B6
4
	full_text'
%
#%654 = bitcast double* %499 to i64*
.double*8
B

	full_text

double* %499
Jload8
B@
>
	full_text1
/
-%655 = load i64, i64* %654, align 8, !tbaa !8
(i64*8
B

	full_text

	i64* %654
Pload8
BF
D
	full_text7
5
3%656 = load double, double* %478, align 8, !tbaa !8
.double*8
B

	full_text

double* %478
Abitcast8
B4
2
	full_text%
#
!%657 = bitcast i64 %655 to double
&i648
B

	full_text


i64 %655
mcall8
Bc
a
	full_textT
R
P%658 = tail call double @llvm.fmuladd.f64(double %653, double %657, double %656)
,double8
B

	full_text

double %653
,double8
B

	full_text

double %657
,double8
B

	full_text

double %656
Pstore8
BE
C
	full_text6
4
2store double %658, double* %478, align 8, !tbaa !8
,double8
B

	full_text

double %658
.double*8
B

	full_text

double* %478
Cbitcast8
B6
4
	full_text'
%
#%659 = bitcast double* %501 to i64*
.double*8
B

	full_text

double* %501
Jload8
B@
>
	full_text1
/
-%660 = load i64, i64* %659, align 8, !tbaa !8
(i64*8
B

	full_text

	i64* %659
Pload8
BF
D
	full_text7
5
3%661 = load double, double* %481, align 8, !tbaa !8
.double*8
B

	full_text

double* %481
Abitcast8
B4
2
	full_text%
#
!%662 = bitcast i64 %660 to double
&i648
B

	full_text


i64 %660
mcall8
Bc
a
	full_textT
R
P%663 = tail call double @llvm.fmuladd.f64(double %653, double %662, double %661)
,double8
B

	full_text

double %653
,double8
B

	full_text

double %662
,double8
B

	full_text

double %661
Pstore8
BE
C
	full_text6
4
2store double %663, double* %481, align 8, !tbaa !8
,double8
B

	full_text

double %663
.double*8
B

	full_text

double* %481
Cbitcast8
B6
4
	full_text'
%
#%664 = bitcast double* %503 to i64*
.double*8
B

	full_text

double* %503
Jload8
B@
>
	full_text1
/
-%665 = load i64, i64* %664, align 8, !tbaa !8
(i64*8
B

	full_text

	i64* %664
Pload8
BF
D
	full_text7
5
3%666 = load double, double* %484, align 8, !tbaa !8
.double*8
B

	full_text

double* %484
Abitcast8
B4
2
	full_text%
#
!%667 = bitcast i64 %665 to double
&i648
B

	full_text


i64 %665
mcall8
Bc
a
	full_textT
R
P%668 = tail call double @llvm.fmuladd.f64(double %653, double %667, double %666)
,double8
B

	full_text

double %653
,double8
B

	full_text

double %667
,double8
B

	full_text

double %666
Pstore8
BE
C
	full_text6
4
2store double %668, double* %484, align 8, !tbaa !8
,double8
B

	full_text

double %668
.double*8
B

	full_text

double* %484
Pload8
BF
D
	full_text7
5
3%669 = load double, double* %624, align 8, !tbaa !8
.double*8
B

	full_text

double* %624
Pload8
BF
D
	full_text7
5
3%670 = load double, double* %361, align 8, !tbaa !8
.double*8
B

	full_text

double* %361
Cfsub8
B9
7
	full_text*
(
&%671 = fsub double -0.000000e+00, %670
,double8
B

	full_text

double %670
mcall8
Bc
a
	full_textT
R
P%672 = tail call double @llvm.fmuladd.f64(double %671, double %648, double %669)
,double8
B

	full_text

double %671
,double8
B

	full_text

double %648
,double8
B

	full_text

double %669
Pstore8
BE
C
	full_text6
4
2store double %672, double* %624, align 8, !tbaa !8
,double8
B

	full_text

double %672
.double*8
B

	full_text

double* %624
Pload8
BF
D
	full_text7
5
3%673 = load double, double* %639, align 8, !tbaa !8
.double*8
B

	full_text

double* %639
Pload8
BF
D
	full_text7
5
3%674 = load double, double* %368, align 8, !tbaa !8
.double*8
B

	full_text

double* %368
Cfsub8
B9
7
	full_text*
(
&%675 = fsub double -0.000000e+00, %674
,double8
B

	full_text

double %674
mcall8
Bc
a
	full_textT
R
P%676 = tail call double @llvm.fmuladd.f64(double %675, double %650, double %673)
,double8
B

	full_text

double %675
,double8
B

	full_text

double %650
,double8
B

	full_text

double %673
Pstore8
BE
C
	full_text6
4
2store double %676, double* %639, align 8, !tbaa !8
,double8
B

	full_text

double %676
.double*8
B

	full_text

double* %639
7icmp8
B-
+
	full_text

%677 = icmp sgt i32 %12, 2
=br8
B5
3
	full_text&
$
"br i1 %677, label %678, label %737
$i18
B

	full_text
	
i1 %677
(br8B 

	full_text

br label %679
Lphi8BC
A
	full_text4
2
0%680 = phi double [ %731, %679 ], [ %676, %678 ]
,double8B

	full_text

double %731
,double8B

	full_text

double %676
Lphi8BC
A
	full_text4
2
0%681 = phi double [ %680, %679 ], [ %650, %678 ]
,double8B

	full_text

double %680
,double8B

	full_text

double %650
Lphi8BC
A
	full_text4
2
0%682 = phi double [ %721, %679 ], [ %672, %678 ]
,double8B

	full_text

double %721
,double8B

	full_text

double %672
Lphi8BC
A
	full_text4
2
0%683 = phi double [ %682, %679 ], [ %648, %678 ]
,double8B

	full_text

double %682
,double8B

	full_text

double %648
Lphi8BC
A
	full_text4
2
0%684 = phi double [ %711, %679 ], [ %668, %678 ]
,double8B

	full_text

double %711
,double8B

	full_text

double %668
Iphi8B@
>
	full_text1
/
-%685 = phi i64 [ %736, %679 ], [ %665, %678 ]
&i648B

	full_text


i64 %736
&i648B

	full_text


i64 %665
Lphi8BC
A
	full_text4
2
0%686 = phi double [ %706, %679 ], [ %663, %678 ]
,double8B

	full_text

double %706
,double8B

	full_text

double %663
Iphi8B@
>
	full_text1
/
-%687 = phi i64 [ %735, %679 ], [ %660, %678 ]
&i648B

	full_text


i64 %735
&i648B

	full_text


i64 %660
Lphi8BC
A
	full_text4
2
0%688 = phi double [ %701, %679 ], [ %658, %678 ]
,double8B

	full_text

double %701
,double8B

	full_text

double %658
Iphi8B@
>
	full_text1
/
-%689 = phi i64 [ %734, %679 ], [ %655, %678 ]
&i648B

	full_text


i64 %734
&i648B

	full_text


i64 %655
Iphi8B@
>
	full_text1
/
-%690 = phi i64 [ %732, %679 ], [ %292, %678 ]
&i648B

	full_text


i64 %732
&i648B

	full_text


i64 %292
tgetelementptr8Ba
_
	full_textR
P
N%691 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %690, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %690
Pload8BF
D
	full_text7
5
3%692 = load double, double* %691, align 8, !tbaa !8
.double*8B

	full_text

double* %691
tgetelementptr8Ba
_
	full_textR
P
N%693 = getelementptr inbounds [5 x double], [5 x double]* %36, i64 %690, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %36
&i648B

	full_text


i64 %690
Pload8BF
D
	full_text7
5
3%694 = load double, double* %693, align 8, !tbaa !8
.double*8B

	full_text

double* %693
Cfsub8B9
7
	full_text*
(
&%695 = fsub double -0.000000e+00, %692
,double8B

	full_text

double %692
Cfsub8B9
7
	full_text*
(
&%696 = fsub double -0.000000e+00, %694
,double8B

	full_text

double %694
«getelementptr8B—
”
	full_text†
ƒ
€%697 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %690, i64 %76, i64 0
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %690
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%698 = load double, double* %697, align 8, !tbaa !8
.double*8B

	full_text

double* %697
mcall8Bc
a
	full_textT
R
P%699 = tail call double @llvm.fmuladd.f64(double %695, double %688, double %698)
,double8B

	full_text

double %695
,double8B

	full_text

double %688
,double8B

	full_text

double %698
Abitcast8B4
2
	full_text%
#
!%700 = bitcast i64 %689 to double
&i648B

	full_text


i64 %689
mcall8Bc
a
	full_textT
R
P%701 = tail call double @llvm.fmuladd.f64(double %696, double %700, double %699)
,double8B

	full_text

double %696
,double8B

	full_text

double %700
,double8B

	full_text

double %699
Pstore8BE
C
	full_text6
4
2store double %701, double* %697, align 8, !tbaa !8
,double8B

	full_text

double %701
.double*8B

	full_text

double* %697
«getelementptr8B—
”
	full_text†
ƒ
€%702 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %690, i64 %76, i64 1
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %690
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%703 = load double, double* %702, align 8, !tbaa !8
.double*8B

	full_text

double* %702
mcall8Bc
a
	full_textT
R
P%704 = tail call double @llvm.fmuladd.f64(double %695, double %686, double %703)
,double8B

	full_text

double %695
,double8B

	full_text

double %686
,double8B

	full_text

double %703
Abitcast8B4
2
	full_text%
#
!%705 = bitcast i64 %687 to double
&i648B

	full_text


i64 %687
mcall8Bc
a
	full_textT
R
P%706 = tail call double @llvm.fmuladd.f64(double %696, double %705, double %704)
,double8B

	full_text

double %696
,double8B

	full_text

double %705
,double8B

	full_text

double %704
Pstore8BE
C
	full_text6
4
2store double %706, double* %702, align 8, !tbaa !8
,double8B

	full_text

double %706
.double*8B

	full_text

double* %702
«getelementptr8B—
”
	full_text†
ƒ
€%707 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %690, i64 %76, i64 2
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %690
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%708 = load double, double* %707, align 8, !tbaa !8
.double*8B

	full_text

double* %707
mcall8Bc
a
	full_textT
R
P%709 = tail call double @llvm.fmuladd.f64(double %695, double %684, double %708)
,double8B

	full_text

double %695
,double8B

	full_text

double %684
,double8B

	full_text

double %708
Abitcast8B4
2
	full_text%
#
!%710 = bitcast i64 %685 to double
&i648B

	full_text


i64 %685
mcall8Bc
a
	full_textT
R
P%711 = tail call double @llvm.fmuladd.f64(double %696, double %710, double %709)
,double8B

	full_text

double %696
,double8B

	full_text

double %710
,double8B

	full_text

double %709
Pstore8BE
C
	full_text6
4
2store double %711, double* %707, align 8, !tbaa !8
,double8B

	full_text

double %711
.double*8B

	full_text

double* %707
«getelementptr8B—
”
	full_text†
ƒ
€%712 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %690, i64 %76, i64 3
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %690
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%713 = load double, double* %712, align 8, !tbaa !8
.double*8B

	full_text

double* %712
tgetelementptr8Ba
_
	full_textR
P
N%714 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %690, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %690
Pload8BF
D
	full_text7
5
3%715 = load double, double* %714, align 8, !tbaa !8
.double*8B

	full_text

double* %714
Cfsub8B9
7
	full_text*
(
&%716 = fsub double -0.000000e+00, %715
,double8B

	full_text

double %715
mcall8Bc
a
	full_textT
R
P%717 = tail call double @llvm.fmuladd.f64(double %716, double %682, double %713)
,double8B

	full_text

double %716
,double8B

	full_text

double %682
,double8B

	full_text

double %713
tgetelementptr8Ba
_
	full_textR
P
N%718 = getelementptr inbounds [5 x double], [5 x double]* %38, i64 %690, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %38
&i648B

	full_text


i64 %690
Pload8BF
D
	full_text7
5
3%719 = load double, double* %718, align 8, !tbaa !8
.double*8B

	full_text

double* %718
Cfsub8B9
7
	full_text*
(
&%720 = fsub double -0.000000e+00, %719
,double8B

	full_text

double %719
mcall8Bc
a
	full_textT
R
P%721 = tail call double @llvm.fmuladd.f64(double %720, double %683, double %717)
,double8B

	full_text

double %720
,double8B

	full_text

double %683
,double8B

	full_text

double %717
Pstore8BE
C
	full_text6
4
2store double %721, double* %712, align 8, !tbaa !8
,double8B

	full_text

double %721
.double*8B

	full_text

double* %712
«getelementptr8B—
”
	full_text†
ƒ
€%722 = getelementptr inbounds [103 x [103 x [5 x double]]], [103 x [103 x [5 x double]]]* %71, i64 %74, i64 %690, i64 %76, i64 4
Y[103 x [103 x [5 x double]]]*8B4
2
	full_text%
#
![103 x [103 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %74
&i648B

	full_text


i64 %690
%i648B

	full_text
	
i64 %76
Pload8BF
D
	full_text7
5
3%723 = load double, double* %722, align 8, !tbaa !8
.double*8B

	full_text

double* %722
tgetelementptr8Ba
_
	full_textR
P
N%724 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %690, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %690
Pload8BF
D
	full_text7
5
3%725 = load double, double* %724, align 8, !tbaa !8
.double*8B

	full_text

double* %724
Cfsub8B9
7
	full_text*
(
&%726 = fsub double -0.000000e+00, %725
,double8B

	full_text

double %725
mcall8Bc
a
	full_textT
R
P%727 = tail call double @llvm.fmuladd.f64(double %726, double %680, double %723)
,double8B

	full_text

double %726
,double8B

	full_text

double %680
,double8B

	full_text

double %723
tgetelementptr8Ba
_
	full_textR
P
N%728 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %690, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %690
Pload8BF
D
	full_text7
5
3%729 = load double, double* %728, align 8, !tbaa !8
.double*8B

	full_text

double* %728
Cfsub8B9
7
	full_text*
(
&%730 = fsub double -0.000000e+00, %729
,double8B

	full_text

double %729
mcall8Bc
a
	full_textT
R
P%731 = tail call double @llvm.fmuladd.f64(double %730, double %681, double %727)
,double8B

	full_text

double %730
,double8B

	full_text

double %681
,double8B

	full_text

double %727
Pstore8BE
C
	full_text6
4
2store double %731, double* %722, align 8, !tbaa !8
,double8B

	full_text

double %731
.double*8B

	full_text

double* %722
7add8B.
,
	full_text

%732 = add nsw i64 %690, -1
&i648B

	full_text


i64 %690
8icmp8B.
,
	full_text

%733 = icmp sgt i64 %690, 0
&i648B

	full_text


i64 %690
Abitcast8B4
2
	full_text%
#
!%734 = bitcast double %688 to i64
,double8B

	full_text

double %688
Abitcast8B4
2
	full_text%
#
!%735 = bitcast double %686 to i64
,double8B

	full_text

double %686
Abitcast8B4
2
	full_text%
#
!%736 = bitcast double %684 to i64
,double8B

	full_text

double %684
=br8B5
3
	full_text&
$
"br i1 %733, label %679, label %737
$i18B

	full_text
	
i1 %733
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
%i328B

	full_text
	
i32 %11
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %8
,double*8B

	full_text


double* %6
%i328B

	full_text
	
i32 %10
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %9
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %7
%i328B

	full_text
	
i32 %12
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
4double8B&
$
	full_text

double 1.250000e-03
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 32
5double8B'
%
	full_text

double -0.000000e+00
$i328B

	full_text


i32 -3
5double8B'
%
	full_text

double -5.050000e-02
#i328B

	full_text	

i32 2
:double8B,
*
	full_text

double 0x3FF5555555555555
4double8B&
$
	full_text

double 1.500000e-03
4double8B&
$
	full_text

double 2.500000e-04
$i328B

	full_text


i32 -1
4double8B&
$
	full_text

double 1.000000e-01
$i328B

	full_text


i32 -2
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 3
5double8B'
%
	full_text

double -1.000000e-03
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 5
%i328B

	full_text
	
i32 102
4double8B&
$
	full_text

double 2.040200e+01
4double8B&
$
	full_text

double 1.020100e+01
$i648B

	full_text


i64 10
#i648B

	full_text	

i64 3
4double8B&
$
	full_text

double 5.050000e-02
#i328B

	full_text	

i32 0
%i328B

	full_text
	
i32 515
4double8B&
$
	full_text

double 1.000000e+00
#i328B

	full_text	

i32 7
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 1
4double8B&
$
	full_text

double 7.500000e-01
4double8B&
$
	full_text

double 0.000000e+00
#i648B

	full_text	

i64 0
:double8B,
*
	full_text

double 0x3FFF5C28F5C28F5B        	
 		                       !" !! #$ ## %& %% '( '' )* )) +, ++ -. -- /0 // 12 11 34 33 55 67 66 89 88 :; :: <= << >? >@ >> AB AA CD CE CC FG FF HI HJ HH KL KK MN MM OP OO QR QQ ST SS UV UU WX WW YZ Y[ YY \] \\ ^_ ^` ^^ ab aa cd ce cc fg ff hi hh jk jj lm ll no np nn qr qs qq tu tv tt wx ww yz yy {| {{ }~ }} €  
‚  ƒ„ ƒ
… ƒƒ †
‡ †† ‰ 
  ‹
 ‹‹  
  
‘  ’“ ’’ ”
• ”” –— –– 
™  ›  
   
   ΅
Ά ΅΅ £¤ £
¥ ££ ¦
§ ¦¦ ¨© ¨
ª ¨¨ «
¬ «« ­­ ®® ―
° ―― ±
² ±± ³
΄ ³³ µ
¶ µµ ·
Έ ·· Ή
Ί ΉΉ »
Ό »» ½Ύ ½½ Ώΐ ΏΏ ΑΒ ΑΑ ΓΔ ΓΓ ΕΖ Ε
Η Ε
Θ ΕΕ ΙΚ ΙΙ ΛΜ ΛΛ ΝΞ Ν
Ο Ν
Π ΝΝ ΡÒ ΡΡ ΣΤ Σ
Υ ΣΣ ΦΧ ΦΦ ΨΩ ΨΨ ΪΫ Ϊ
ά ΪΪ έή έ
ί έ
ΰ έέ αβ αα γδ γγ εζ ε
η εε θι θ
κ θθ λμ λ
ν λ
ξ λλ οπ ο
ρ οο ςσ ς
τ ς
υ ςς φχ φφ ψω ψ
ϊ ψ
ϋ ψψ όύ όό ώÿ ώώ € €
‚ €
ƒ €€ „… „„ †‡ †† ‰ 
  ‹ ‹‹    
‘  ’“ ’
” ’
• ’’ –— –– ™  › 
   
   ΅  
Ά  
£    ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©
« ©
¬ ©© ­® ­­ ―° ―
± ―
² ―― ³΄ ³³ µ¶ µµ ·Έ ·
Ή ·
Ί ·· »Ό »» ½Ύ ½½ Ώΐ Ώ
Α ΏΏ ΒΓ ΒΒ ΔΕ ΔΔ ΖΗ Ζ
Θ ΖΖ ΙΚ Ι
Λ Ι
Μ ΙΙ ΝΞ ΝΝ ΟΠ ΟΟ ΡÒ Ρ
Σ ΡΡ ΤΥ Τ
Φ ΤΤ ΧΨ Χ
Ω Χ
Ϊ ΧΧ Ϋά ΫΫ έή έ
ί έέ ΰα ΰ
β ΰ
γ ΰΰ δε δδ ζη ζζ θ
ι θθ κλ κκ μ
ν μμ ξο ξ
π ξξ ρς ρρ στ σ
υ σσ φχ φφ ψω ψψ ϊϋ ϊϊ όύ ό
ώ όό ÿ€ ÿÿ 
‚  ƒ„ ƒ
… ƒƒ †‡ †† ‰  ‹ 
    
  ‘’ ‘‘ “
” ““ •– •
— •• ™  › 
      
΅  Ά£ Ά
¤ ΆΆ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬
­ ¬¬ ®― ®® °
± °° ²³ ²
΄ ²² µ¶ µµ ·Έ ·
Ή ·· Ί» ΊΊ Ό½ Ό
Ύ ΌΌ Ώΐ Ώ
Α ΏΏ ΒΓ ΒΒ ΔΕ Δ
Ζ ΔΔ ΗΘ ΗΗ Ι
Κ ΙΙ ΛΜ Λ
Ν Λ
Ξ ΛΛ ΟΠ ΟΟ ΡÒ ΡΡ ΣΤ Σ
Υ Σ
Φ ΣΣ ΧΨ ΧΧ ΩΪ ΩΩ Ϋά Ϋ
έ ΫΫ ήί ήή ΰα ΰΰ βγ β
δ ββ εζ ε
η ε
θ εε ικ ιι λμ λλ νξ ν
ο νν πρ π
ς ππ στ σ
υ σ
φ σσ χψ χχ ωϊ ω
ϋ ωω όύ ό
ώ ό
ÿ όό € €€ ‚ƒ ‚‚ „
… „„ †‡ †† 
‰  ‹ 
      ‘’ ‘
“ ‘‘ ”• ”” –— –– ™  › 
    
   ΅Ά ΅
£ ΅΅ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­
® ­­ ―° ―― ±
² ±± ³΄ ³
µ ³³ ¶· ¶¶ ΈΉ Έ
Ί ΈΈ »Ό »» ½Ύ ½
Ώ ½½ ΐΑ ΐ
Β ΐΐ ΓΔ ΓΓ ΕΖ Ε
Η ΕΕ ΘΙ ΘΘ Κ
Λ ΚΚ ΜΝ ΜΜ Ξ
Ο ΞΞ ΠΡ Π
Ò ΠΠ ΣΤ ΣΣ ΥΦ Υ
Χ ΥΥ ΨΩ ΨΨ ΪΫ Ϊ
ά ΪΪ έή έ
ί έέ ΰα ΰΰ βγ β
δ ββ εζ εε η
θ ηη ιι κκ λμ λξ νν ορ ππ ςσ ς
τ ςς υφ υ
χ υυ ψω ψ
ϊ ψψ ϋό ϋ
ύ ϋϋ ώÿ ώ
€ ώώ ‚ 
ƒ  „… „„ †‡ †
 †
‰ †
 †† ‹ ‹‹    
‘ 
’ 
“  ”• ”” –— –
 –– ™ ™
› ™™      ΅  
Ά    £¤ £
¥ £
¦ ££ §¨ §§ ©ª ©© «¬ «
­ «« ®― ®
° ®® ±² ±
³ ±
΄ ±± µ¶ µ
· µµ ΈΉ Έ
Ί ΈΈ »Ό »
½ »
Ύ »
Ώ »» ΐΑ ΐΐ ΒΓ Β
Δ ΒΒ Ε
Ζ ΕΕ ΗΘ ΗΗ Ι
Κ ΙΙ ΛΜ Λ
Ν ΛΛ ΞΟ ΞΞ ΠΡ Π
Ò ΠΠ ΣΤ Σ
Υ ΣΣ ΦΧ ΦΦ ΨΩ ΨΨ ΪΫ Ϊ
ά ΪΪ έή έ
ί έέ ΰα ΰΰ β
γ ββ δε δ
ζ δδ ηθ ηη ικ ι
λ ιι μν μ
ξ μμ οπ ο
ρ οο ς
σ ςς τυ τ
φ ττ χ
ψ χχ ωϊ ω
ϋ ωω όύ ό
ώ όό ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ‰ 
  ‹ ‹
 ‹‹  
  ‘’ ‘
“ ‘‘ ”
• ”” –— –
 –– ™
 ™™ › ›
 ››  
   ΅Ά ΅
£ ΅΅ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ªª ­® ­
― ­­ °± °
² °° ³΄ ³
µ ³³ ¶
· ¶¶ ΈΉ Έ
Ί ΈΈ »Ό »Ύ ½
Ώ ½½ ΐΑ ΐ
Β ΐΐ ΓΔ Γ
Ε ΓΓ ΖΗ Ζ
Θ ΖΖ ΙΚ Ι
Λ ΙΙ ΜΝ Μ
Ξ ΜΜ ΟΟ ΠΡ ΠΠ ÒΣ Ò
Τ Ò
Υ Ò
Φ ÒÒ ΧΨ ΧΧ ΩΪ ΩΩ Ϋά Ϋ
έ Ϋ
ή Ϋ
ί ΫΫ ΰα ΰΰ βγ β
δ ββ εζ ε
η εε θι θθ κλ κκ μν μ
ξ μμ οπ ο
ρ ο
ς οο στ σσ υφ υυ χψ χ
ω χχ ϊϋ ϊ
ό ϊϊ ύώ ύ
ÿ ύ
€ ύύ ‚ 
ƒ  „… „
† „„ ‡ ‡
‰ ‡
 ‡
‹ ‡‡     ‘ 
’  “
” ““ •– •• —
 —— ™ ™
› ™™    
   ΅Ά ΅
£ ΅΅ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®― ®® °
± °° ²³ ²
΄ ²² µ¶ µµ ·Έ ·
Ή ·· Ί» Ί
Ό ΊΊ ½Ύ ½
Ώ ½½ ΐ
Α ΐΐ ΒΓ Β
Δ ΒΒ Ε
Ζ ΕΕ ΗΘ Η
Ι ΗΗ ΚΛ Κ
Μ ΚΚ ΝΞ Ν
Ο ΝΝ ΠΡ Π
Ò ΠΠ ΣΤ Σ
Υ ΣΣ ΦΧ Φ
Ψ ΦΦ ΩΪ Ω
Ϋ ΩΩ άέ ά
ή άά ίΰ ί
α ίί β
γ ββ δε δ
ζ δδ η
θ ηη ικ ι
λ ιι μν μ
ξ μμ οπ ο
ρ οο ςσ ς
τ ςς υφ υ
χ υυ ψω ψ
ϊ ψψ ϋό ϋ
ύ ϋϋ ώÿ ώ
€ ώώ ‚ 
ƒ  „
… „„ †† ‡ ‡‡ ‰ ‰
‹ ‰
 ‰
 ‰‰   ‘  ’“ ’
” ’
• ’
– ’’ — —— ™ ™
› ™™  
     ΅Ά ΅΅ £¤ £
¥ ££ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬­ ¬¬ ®― ®
° ®® ±² ±
³ ±± ΄µ ΄
¶ ΄
· ΄΄ ΈΉ Έ
Ί ΈΈ »Ό »
½ »» ΎΏ Ύ
ΐ Ύ
Α Ύ
Β ΎΎ ΓΔ ΓΓ ΕΖ Ε
Η ΕΕ Θ
Ι ΘΘ ΚΛ ΚΚ Μ
Ν ΜΜ ΞΟ Ξ
Π ΞΞ ΡÒ ΡΡ ΣΤ Σ
Υ ΣΣ ΦΧ Φ
Ψ ΦΦ ΩΪ ΩΩ Ϋά ΫΫ έή έ
ί έέ ΰα ΰ
β ΰΰ γδ γγ ε
ζ εε ηθ η
ι ηη κλ κ
μ κκ νξ ν
ο νν πρ π
ς ππ σ
τ σσ υφ υ
χ υυ ψ
ω ψψ ϊϋ ϊ
ό ϊϊ ύώ ύ
ÿ ύύ €		 €	
‚	 €	€	 ƒ	„	 ƒ	
…	 ƒ	ƒ	 †	‡	 †	
	 †	†	 ‰		 ‰	
‹	 ‰	‰	 		 	
	 		 		 	
‘	 		 ’	“	 ’	
”	 ’	’	 •	
–	 •	•	 —		 —	
™	 —	—	 	
›	 		 		 	
	 		 	 	 	
΅	 		 Ά	£	 Ά	
¤	 Ά	Ά	 ¥	¦	 ¥	
§	 ¥	¥	 ¨	©	 ¨	
ª	 ¨	¨	 «	¬	 «	
­	 «	«	 ®	―	 ®	
°	 ®	®	 ±	²	 ±	
³	 ±	±	 ΄	µ	 ΄	
¶	 ΄	΄	 ·	
Έ	 ·	·	 Ή	Ί	 Ή	Ή	 »	Ό	 »	»	 ½	Ύ	 ½	½	 Ώ	ΐ	 Ώ	Ώ	 Α	Β	 Α	
Γ	 Α	
Δ	 Α	Α	 Ε	Ζ	 Ε	Ε	 Η	Θ	 Η	Η	 Ι	Κ	 Ι	
Λ	 Ι	
Μ	 Ι	Ι	 Ν	Ξ	 Ν	Ν	 Ο	Π	 Ο	Ο	 Ρ	Ò	 Ρ	
Σ	 Ρ	
Τ	 Ρ	Ρ	 Υ	Φ	 Υ	Υ	 Χ	Ψ	 Χ	Χ	 Ω	Ϊ	 Ω	
Ϋ	 Ω	
ά	 Ω	Ω	 έ	ή	 έ	έ	 ί	ΰ	 ί	ί	 α	β	 α	
γ	 α	
δ	 α	α	 ε	ζ	 ε	ε	 η	θ	 η	η	 ι	κ	 ι	
λ	 ι	
μ	 ι	ι	 ν	ξ	 ν	ν	 ο	π	 ο	ο	 ρ	ρ	 ς	σ	 ς	υ	 τ	τ	 φ	ψ	 χ	
ω	 χ	χ	 ϊ	ϋ	 ϊ	
ό	 ϊ	ϊ	 ύ	ώ	 ύ	
ÿ	 ύ	ύ	 €

 €

‚
 €
€
 ƒ
„
 ƒ

…
 ƒ
ƒ
 †
‡
 †


 †
†
 ‰

 ‰
‰
 ‹

 ‹


 ‹
‹
 

 


 

 ‘
’
 ‘

“
 ‘
‘
 ”
•
 ”

–
 ”
”
 —

 —
—
 ™

 ™

›
 ™
™
 

 

 


 

  
΅
  

Ά
  
 
 £
¤
 £

¥
 £
£
 ¦
§
 ¦

¨
 ¦
¦
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

®
 ¬
¬
 ―
°
 ―
―
 ±
²
 ±

³
 ±
±
 ΄
µ
 ΄

¶
 ΄

·
 ΄

Έ
 ΄
΄
 Ή
Ί
 Ή

»
 Ή
Ή
 Ό
½
 Ό
Ό
 Ύ
Ώ
 Ύ

ΐ
 Ύ
Ύ
 Α
Β
 Α

Γ
 Α

Δ
 Α

Ε
 Α
Α
 Ζ
Η
 Ζ

Θ
 Ζ
Ζ
 Ι
Κ
 Ι
Ι
 Λ
Μ
 Λ

Ν
 Λ
Λ
 Ξ
Ο
 Ξ

Π
 Ξ

Ρ
 Ξ

Ò
 Ξ
Ξ
 Σ
Τ
 Σ

Υ
 Σ
Σ
 Φ
Χ
 Φ

Ψ
 Φ
Φ
 Ω
Ϊ
 Ω
Ω
 Ϋ

ά
 Ϋ
Ϋ
 έ
ή
 έ

ί
 έ

ΰ
 έ
έ
 α
β
 α

γ
 α
α
 δ
ε
 δ

ζ
 δ
δ
 η
θ
 η
η
 ι
κ
 ι

λ
 ι

μ
 ι
ι
 ν
ξ
 ν
ν
 ο
π
 ο

ρ
 ο

ς
 ο
ο
 σ
τ
 σ
σ
 υ
φ
 υ

χ
 υ

ψ
 υ
υ
 ω
ϊ
 ω
ω
 ϋ
ό
 ϋ

ύ
 ϋ
ϋ
 ώ
ÿ
 ώ
ώ
 € €
‚ €€ ƒ„ ƒƒ …† …
‡ …… ‰  ‹ 
 
 
    ‘’ ‘
“ ‘
” ‘
• ‘‘ –— –– ™ 
 
› 
    
   ΅Ά ΅
£ ΅
¤ ΅΅ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬
® ¬
― ¬¬ °± °
² °
³ °° ΄µ ΄
¶ ΄
· ΄΄ ΈΉ Έ
Ί Έ
» ΈΈ Ό½ Ό
Ύ ΌΌ Ώΐ ΏΏ ΑΒ ΑΑ ΓΔ ΓΓ ΕΖ ΕΕ ΗΘ ΗΗ ΙΚ ΙΙ ΛΜ ΛΞ Ν
Ο ΝΝ ΠΡ Π
Ò ΠΠ ΣΤ Σ
Υ ΣΣ ΦΧ Φ
Ψ ΦΦ ΩΪ Ω
Ϋ ΩΩ άέ ά
ή άά ίΰ ί
α ίί βγ β
δ ββ εζ ε
η εε θι θ
κ θθ λμ λλ ν
ξ νν οπ ο
ρ οο ςσ ς
τ ςς υφ υ
χ υυ ψω ψ
ϊ ψψ ϋό ϋϋ ύώ ύ
ÿ ύύ € €
‚ €
ƒ €
„ €€ …† …
‡ …… ‰  ‹ 
   
 
 
‘  ’“ ’
” ’’ •– •• — —
™ —— › 
 
 
    
΅  Ά£ Ά
¤ ΆΆ ¥¦ ¥¥ §
¨ §§ ©ª ©
« ©
¬ ©© ­® ­
― ­­ °± °
² °° ³΄ ³
µ ³
¶ ³³ ·Έ ·
Ή ·· Ί» ΊΊ Ό½ Ό
Ύ Ό
Ώ ΌΌ ΐΑ ΐΐ ΒΓ Β
Δ Β
Ε ΒΒ ΖΗ ΖΖ ΘΙ Θ
Κ Θ
Λ ΘΘ Μ
Ν ΜΜ ΞΟ Ξ
Π ΞΞ ΡÒ Ρ
Σ Ρ
Τ Ρ
Υ ΡΡ ΦΧ Φ
Ψ ΦΦ ΩΪ Ω
Ϋ ΩΩ άέ ά
ή ά
ί ά
ΰ άά αβ α
γ αα δε δ
ζ δδ ηθ η
ι η
κ η
λ ηη μν μ
ξ μμ οπ οο ρς ρρ στ σσ υφ υυ χψ χχ ωϊ ωω ϋό ϋϋ ύώ ύύ ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ„ ƒƒ …† …… ‡ ‡
‰ ‡
 ‡‡ ‹ ‹‹    
‘ 
’  “” ““ •– •• — —
™ —
 —— › ››     Ά ΅΅ £¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ªª ­® ­
― ­­ °± °° ²³ ²
΄ ²² µ¶ µ
· µµ ΈΉ Έ
Ί ΈΈ »Ό »
½ »» ΎΏ Ύ
ΐ ΎΎ ΑΒ Α
Γ ΑΑ ΔΕ Δ
Ζ ΔΔ ΗΘ Η
Ι ΗΗ ΚΛ ΚΚ ΜΝ ΜΜ ΞΟ Ξ
Π ΞΞ ΡÒ ΡΡ Σ
Τ ΣΣ ΥΦ Υ
Χ ΥΥ ΨΩ Ψ
Ϊ ΨΨ Ϋά Ϋ
έ ΫΫ ήί ή
ΰ ήή αβ α
γ αα δε δδ ζη ζ
θ ζζ ικ ι
λ ι
μ ι
ν ιι ξο ξ
π ξξ ρς ρ
σ ρρ τυ ττ φ
χ φφ ψω ψ
ϊ ψ
ϋ ψψ όύ ό
ώ όό ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †† ‰ 
 
‹   
    ‘’ ‘
“ ‘‘ ”• ”” –— –
 –– ™ ™™ › ›
 ›
 ›
 ››  ΅    Ά
£ ΆΆ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨
ª ¨¨ «¬ «
­ «
® «« ―° ―
± ―
² ―― ³΄ ³
µ ³³ ¶· ¶¶ Έ
Ή ΈΈ Ί» Ί
Ό ΊΊ ½Ύ ½
Ώ ½½ ΐΑ ΐ
Β ΐΐ ΓΔ Γ
Ε ΓΓ ΖΗ Ζ
Θ ΖΖ ΙΚ ΙΙ ΛΜ Λ
Ν ΛΛ ΞΟ Ξ
Π Ξ
Ρ Ξ
Ò ΞΞ ΣΤ Σ
Υ ΣΣ ΦΧ Φ
Ψ ΦΦ ΩΪ ΩΩ Ϋ
ά ΫΫ έή έ
ί έ
ΰ έέ αβ α
γ αα δε δ
ζ δδ ηθ η
ι η
κ ηη λμ λλ νξ ν
ο ν
π νν ρς ρ
σ ρρ τυ ττ φχ φ
ψ φφ ωϊ ωω ϋό ϋ
ύ ϋϋ ώÿ ώώ € €
‚ €
ƒ €
„ €€ …† …… ‡
 ‡‡ ‰ ‰
‹ ‰
 ‰‰  
  ‘ 
’ 
“  ”• ”
– ”
— ”” ™ 
  › ››      ΅Ά ΅΅ £¤ £¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «« ®― ®
° ®® ±² ±
³ ±± ΄µ ΄
¶ ΄΄ ·Έ ·
Ή ·· Ί» Ί
Ό ΊΊ ½Ύ ½
Ώ ½½ ΐΑ ΐ
Β ΐΐ ΓΔ Γ
Ε ΓΓ ΖΗ Ζ
Θ ΖΖ ΙΚ ΙΙ Λ
Μ ΛΛ ΝΞ Ν
Ο ΝΝ ΠΡ Π
Ò ΠΠ ΣΤ Σ
Υ ΣΣ ΦΧ Φ
Ψ ΦΦ ΩΪ ΩΩ Ϋά Ϋ
έ ΫΫ ήί ή
ΰ ή
α ή
β ήή γδ γ
ε γγ ζη ζ
θ ζζ ικ ιι λ
μ λλ νξ ν
ο ν
π νν ρς ρ
σ ρρ τυ τ
φ ττ χψ χ
ω χ
ϊ χχ ϋό ϋ
ύ ϋϋ ώÿ ώώ € €
‚ €
ƒ €€ „… „„ †
‡ †† ‰ 
  ‹ ‹
 ‹‹  
  ‘’ ‘
“ ‘‘ ”• ”” –— –
 –– ™ ™
› ™
 ™
 ™™  
   ΅Ά ΅
£ ΅΅ ¤¥ ¤¤ ¦
§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬
® ¬¬ ―° ―
± ―― ²³ ²
΄ ²
µ ²² ¶· ¶
Έ ¶¶ ΉΊ ΉΉ »Ό »
½ »
Ύ »» Ώΐ Ώ
Α ΏΏ ΒΓ Β
Δ Β
Ε Β
Ζ ΒΒ ΗΘ Η
Ι ΗΗ ΚΛ Κ
Μ ΚΚ ΝΞ Ν
Ο Ν
Π Ν
Ρ ΝΝ ÒΣ Ò
Τ ÒÒ ΥΦ ΥΥ Χ
Ψ ΧΧ ΩΪ ΩΩ Ϋά ΫΫ έή έέ ίΰ ίί αβ α
γ α
δ αα εζ ε
η εε θι θθ κλ κκ μν μμ ξο ξξ πρ π
ς π
σ ππ τυ τ
φ ττ χψ χχ ωϊ ωω ϋό ϋϋ ύώ ύύ ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ„ ƒ
… ƒƒ †‡ †† ‰  
‹   
 
  ‘ 
’  “” ““ •– •• —
 —— ™ ™
› ™
 ™™  
     ΅Ά ΅¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ªª ­® ­
― ­­ °± °
² °° ³΄ ³
µ ³³ ¶· ¶
Έ ¶¶ ΉΊ Ή
» ΉΉ Ό½ Ό
Ύ ΌΌ Ώΐ Ώ
Α ΏΏ ΒΓ Β
Δ ΒΒ ΕΖ Ε
Η ΕΕ ΘΙ ΘΘ ΚΛ Κ
Μ ΚΚ ΝΞ ΝΝ Ο
Π ΟΟ Ρ
Ò ΡΡ ΣΤ Σ
Υ Σ
Φ Σ
Χ ΣΣ ΨΩ ΨΨ ΪΫ Ϊ
ά Ϊ
έ ΪΪ ήί ήή ΰα ΰ
β ΰ
γ ΰΰ δε δ
ζ δδ ηθ η
ι η
κ η
λ ηη μν μμ ξο ξ
π ξ
ρ ξξ ςσ ςς τυ τ
φ τ
χ ττ ψω ψ
ϊ ψψ ϋό ϋ
ύ ϋ
ώ ϋ
ÿ ϋϋ € €€ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †† ‰ 
 
‹   
   
‘ 
’ 
“  ”• ”” –— –
 –– ™ ™™ ›
 ››  
 
   ΅Ά ΅
£ ΅΅ ¤¥ ¤¤ ¦
§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬
® ¬¬ ―° ―
± ―
² ―
³ ―― ΄µ ΄΄ ¶· ¶
Έ ¶¶ ΉΊ ΉΉ »
Ό »» ½Ύ ½
Ώ ½
ΐ ½½ ΑΒ Α
Γ ΑΑ ΔΕ ΔΔ Ζ
Η ΖΖ ΘΙ Θ
Κ Θ
Λ ΘΘ ΜΝ Μ
Ξ ΜΜ ΟΠ ΟΟ ΡÒ ΡΡ ΣΤ ΣΣ ΥΦ ΥΥ ΧΨ ΧΧ ΩΪ Ωά έ 	ή ί #ΰ 1α )β 5γ ­	δ 	δ ε ®ζ ―η -θ ιθ κθ Οθ †θ ρ	θ      
 	      	      "! $ &% (' *) ,' .- 0' 21 45 7) 9- ;1 =+ ?6 @> B/ D6 EC G3 I6 JH L) NM P- RQ T1 VU X+ Z6 [Y ]/ _6 `^ b3 d6 ec g) i- k1 m+ o6 p/ r6 s3 u6 v) xw z- |{ ~1 € ‚+ „6 …ƒ ‡/ ‰6  3 6  ‘) “’ •- —– ™1 › + 6   Ά/ ¤6 ¥£ §3 ©6 ª¨ ¬! °h ²j ΄l ¶n Έq Ίt Ό Ύ½ ΐ ΒΑ Δ­ ΖΏ ΗΓ ΘΕ ΚΙ Μ ΞΏ ΟΓ ΠΝ ÒΡ Τ# ΥΛ ΧΛ ΩΦ ΫΨ άΪ ήΦ ίΨ ΰΛ βα δγ ζα ηέ ιε κθ μέ νε ξλ π― ρ σΏ τΓ υς χ­ ωΏ ϊΓ ϋψ ύό ÿ Ώ ‚Γ ƒ€ …# ‡„ ‰† ώ ώ ‹  ‘ “‹ ” •ώ —– ™ ›– ’   ΅’ Ά £― ¥  §¤ ¨ ªΏ «Γ ¬© ®­ °Ώ ±Γ ²― ΄³ ¶ ΈΏ ΉΓ Ί· Ό# Ύ» ΐ½ Αµ Γµ ΕΒ ΗΔ ΘΖ ΚΒ ΛΔ Μµ ΞΝ ΠΟ ÒΝ ΣΙ ΥΡ ΦΤ ΨΙ ΩΡ Ϊ― άΧ ήΫ ί αΏ βΓ γΰ ε) ηζ ιλ λκ νΡ ομ πζ ςξ τρ υ  χφ ωζ ϋψ ύϊ ώΧ €ÿ ‚» „ …ƒ ‡ζ ‰† ‹ ζ  - ’‘ ”φ –ξ —‘ ™• › ‘ ψ   ΅δ £† ¤‘ ¦Ά ¨¥ ©‘ «ª ­1 ―® ±φ ³ξ ΄® ¶² Έµ Ή® »ψ ½Ί Ύδ ΐ† Α® ΓΏ ΕΒ Ζ® ΘΗ Κ­ ΜΏ ΝΓ ΞΛ ΠΟ Ò ΤΏ ΥΓ ΦΣ Ψ# ΪΧ άΩ έΡ ίΡ αή γΰ δβ ζή ηΰ θΡ κι μλ ξι οε ρν ςπ τε υν φ― ψσ ϊχ ϋ ύΏ ώΓ ÿό ) ƒ‚ …  ‡† ‰„ ‹  ‚  ’ “Χ •” —‚ ™– › σ   Χ Ά £΅ ¥‚ §¤ ©¦ ª‚ ¬« ®- °― ²­ ΄ µ― ·³ Ή¶ Ί― Ό– Ύ» Ώ€ Α¤ Β― Δΐ ΖΓ Η― ΙΘ Λ1 ΝΜ Ο­ Ρ ÒΜ ΤΠ ΦΣ ΧΜ Ω– ΫΨ ά€ ή¤ ίΜ αέ γΰ δΜ ζε θι μκ ξ„ ρΐ σ€ τς φδ χ± ωσ ϊψ όΧ ύ” ÿΧ €ώ ‚» ƒπ …­ ‡Ώ „ ‰Γ † ‹  Ώ ‘„ ’Γ “ •# —„ ” – ›   ΅ Ά  ¤ ¥ ¦ ¨§ ª© ¬§ ­£ ―« °® ²£ ³« ΄― ¶„ ·± Ήµ Ί ΌΏ ½„ ΎΓ Ώ» Α+ Γπ ΔΒ Ζϋ ΘΗ Κ ΜΙ ΝΛ Ο+ Ρπ ÒΞ ΤΠ Υψ ΧΦ Ω+ Ϋπ άΨ ήΪ ί± αΰ γ” εβ ζδ θ+ κπ λη νι ξ+ ππ ρο σ/ υπ φτ ψυ ϊΞ ϋ/ ύπ ώω €ό / ƒπ „Ψ †‚ ‡ΐ ‰η / π  ‹ / ’π “‘ •3 —π – υ Ξ 3 π  › Ά £3 ¥π ¦Ψ ¨¤ ©ΐ «η ¬3 ®π ―ª ±­ ²3 ΄π µ³ ·„ Ήν ΊΈ Ό» Ύώ ΏΧ Α” ΒΧ Δψ Εσ Η± Θδ Κς Λ€ Νΐ ΞΟ Ρ­ ΣΏ ΤΠ ΥΓ ΦÒ ΨΧ Ϊ άΏ έΠ ήΓ ίΫ α# γΠ δΰ ζβ ηΩ ιΩ λθ νκ ξμ πθ ρκ ςΩ τσ φυ ψσ ωο ϋχ όϊ ώο ÿχ €― ‚Π ƒύ … † Ώ ‰Π Γ ‹‡ κ + ‘ ’ ”Γ –• ½ — ›™ +    Ά £Ζ ¥¤ §+ © ª¦ ¬¨ ­ύ ―® ±ΰ ³° ΄² ¶+ Έ Ήµ »· Ό+ Ύ Ώ½ Α/ Γ ΔΒ ΖΙ Θ Ι/ Λ ΜΗ ΞΚ Ο/ Ρ Ò¦ ΤΠ Υ Χµ Ψ/ Ϊ ΫΦ έΩ ή/ ΰ αί γ3 ε ζδ θΙ κ λ3 ν ξι πμ ρ3 σ τ¦ φς χ ωµ ϊ3 ό ύψ ÿϋ €3 ‚ ƒ …† ­ Ώ ‹‡ Γ ‰  ‘ “Ώ ”‡ •Γ –’ # ‡ ›— ™    Ά ¤΅ ¥£ § ¨΅ © «ª ­¬ ―ª °¦ ²® ³± µ¦ ¶® ·― Ή‡ Ί΄ ΌΈ ½ ΏΏ ΐ‡ ΑΓ ΒΎ Δ+ ΖΠ ΗΕ ΙΖ ΛΚ Νΐ ΟΜ ΠΞ Ò+ ΤΠ ΥΡ ΧΣ Ψύ ΪΩ ά+ ήΠ ίΫ αέ β΄ δγ ζ— θε ι+ λΠ μη ξκ ο+ ρΠ ςπ τ/ φΠ χυ ωΜ ϋΡ ό/ ώΠ ÿϊ 	ύ ‚	/ „	Π …	Ϋ ‡	ƒ	 	Γ 	η ‹	/ 	Π 	‰	 		 ‘	/ “	Π ”	’	 –	3 	Π ™	—	 ›	Μ 	Ρ 	3  	Π ΅		 £		 ¤	3 ¦	Π §	Ϋ ©	¥	 ª	Γ ¬	η ­	3 ―	Π °	«	 ²	®	 ³	3 µ	Π ¶	΄	 Έ	h Ί	w Ό	ρ Ύ	ϊ ΐ	® Β	Ώ Γ	Γ Δ	Α	 Ζ	Ε	 Θ	® Κ	Ώ Λ	Γ Μ	Ι	 Ξ	Ν	 Π	® Ò	Ώ Σ	Γ Τ	Ρ	 Φ	Υ	 Ψ	® Ϊ	Ώ Ϋ	Γ ά	Ω	 ή	έ	 ΰ	® β	Ώ γ	Γ δ	α	 ζ	ε	 θ	® κ	Ώ λ	Γ μ	ι	 ξ	ν	 π	ρ	 σ	Ο υ	Ι ψ	ο	 ω	Η ϋ	η	 ό	Ε ώ	ί	 ÿ	Γ 
Χ	 ‚
Α „
Ο	 …
Ώ ‡
Η	 
—
 
΄ 
»	 
έ
 
Ή	 
Έ ’
Ώ	 “
΅ •
½	 –
‰
 
+ 
‰
 ›
™
 

 

 ΅
‹
 Ά
+ ¤
‰
 ¥
 
 §
£
 ¨

 ª

 «
©
 ­
™
 ®
†
 °

 ²
―
 ³
® µ
Ώ ¶
‰
 ·
Γ Έ
±
 Ί
΄
 »
ƒ
 ½

 Ώ
Ό
 ΐ
® Β
Ώ Γ
‰
 Δ
Γ Ε
Ύ
 Η
Α
 Θ
€
 Κ

 Μ
Ι
 Ν
® Ο
Ώ Π
‰
 Ρ
Γ Ò
Λ
 Τ
Ξ
 Υ
+ Χ
—
 Ψ
Φ
 Ϊ
”
 ά
Ϋ
 ή
 
 ί
‘
 ΰ
+ β
—
 γ
έ
 ε
α
 ζ
ύ	 θ
Ϋ
 κ
±
 λ
η
 μ
ϊ	 ξ
Ϋ
 π
Ύ
 ρ
ν
 ς
χ	 τ
Ϋ
 φ
Λ
 χ
σ
 ψ
‰
 ϊ
+ ό
ω
 ύ
ϋ
 ÿ
+ ω
 ‚€ „+ †ω
 ‡… ‰® ‹Ώ ω
 Γ  ® ’Ώ “ω
 ”Γ •‘ —® ™Ώ ω
 ›Γ  ώ
   Ά 
 £ƒ ¤΅ ¦€ § ©±
 ª « ­Ύ
 ®– ― ±Λ
 ² ³Ϋ
 µ©
 ¶Ω
 · Ή©
 Ί »—
 ½τ	 Ύι
 ΐο
 Βυ
 Δ¨ Ζ¬ Θ° ΚΌ Μο	 ΞΙ Οη	 ΡΗ Òί	 ΤΕ ΥΧ	 ΧΓ ΨΟ	 ΪΑ ΫΗ	 έΏ ή½	 ΰ΅ αΏ	 γΈ δΉ	 ζέ
 η»	 ι΄ κπ με ξν πθ ρο σκ τν φλ χυ ωπ ϊά όν ώϋ ÿ® Ώ ‚Π ƒΓ „ύ †€ ‡Ω ‰ν ‹ ® Ώ Π Γ ‘ “ ”Φ –ν • ™® ›Ώ Π Γ —   ΅+ £‡ ¤Ά ¦ί ¨§ ªο «β ¬+ ®‡ ―© ±­ ²§ ΄υ µ¥ ¶³ ΈΆ ΉΣ »§ ½ύ ΎΊ ΏΠ Α§ Γ Δΐ ΕΝ Η§ Ι— ΚΖ Λ© ΝΜ ΟΌ Π® ÒΏ Σ‡ ΤΓ ΥΞ ΧΡ ΨΜ ΪΒ Ϋ® έΏ ή‡ ίΓ ΰΩ βά γΜ εΘ ζ® θΏ ι‡ κΓ λδ νη ξj π{ ς τ φl ψ ϊµ όΊ ώ® €Ώ Γ ‚ÿ „ƒ †® Ώ ‰Γ ‡ ‹ ® Ώ ‘Γ ’ ”“ –® Ώ ™Γ — › ρ	  Ο Ά΅ ¥ ¦ ¨ © «• ¬› ®… ―Κ ±‰ ³ϋ ΄ ¶ύ ·έ Ήχ Ίη Όω ½¤ Ώσ ΐ« Βυ Γψ Εο Ζ‚ Θρ Ι° Λ° Ν/ Ο° ΠΞ ÒΔ ΤΗ ΦΣ Χ/ Ω° ΪΥ άΨ έΣ ίΡ ΰή βΞ γ­ εΣ ηδ θ® κΏ λ° μΓ νζ οι π/ ςΚ σρ υΎ χφ ωΥ ϊΑ ϋ/ ύΚ ώψ €ό φ ƒή „τ …ª ‡φ ‰ζ † ‹/ Μ  / ’Μ “‘ •/ —Μ – ® Ώ Μ Γ › ΅ £Ά ¥Υ ¦” §¤ ©‘ ªΆ ¬ή ­™ ®Ά °ζ ±  ²3 ΄° µ³ ·Έ Ή» »Έ Ό3 Ύ° ΏΊ Α½ ΒΈ Δ¶ ΕΓ Η³ Θ§ ΚΈ ΜΙ Ν® ΟΏ Π° ΡΓ ÒΛ ΤΞ Υ3 ΧΚ ΨΦ Ϊ² άΫ ήΊ ίµ ΰ3 βΚ γέ εα ζΫ θΓ ιΩ κ¤ μΫ ξΛ ολ π3 ςΜ σρ υ3 χΜ ψφ ϊ3 όΜ ύϋ ÿ® Ώ ‚Μ ƒΓ „€ †τ ‡ Ί ‹ω ‰ φ ‡ ‘Γ ’ώ “‡ •Λ –… —Κ ™΅  ― ν  ” Ά ¤ ¦΅ § © ª• ¬ ­… ―› °ρ ²‚ ³ο µψ ¶υ Έ« Ήσ »¤ Όω Ύη Ώχ Αέ Βύ Δ Εϋ Η‰ Θ’	 Κ΄ Μ± ΞΛ ΟΝ Ρ	 ÒΛ ΤΙ ΥΣ Χ’	 Ψ® ΪΛ άΩ έ® ίΏ ΰΠ αΓ βΫ δή ε/ η‡ θζ κΊ μλ ξΝ ο· π/ ς‡ σν υρ φλ ψΣ ωι ϊχ όζ ύ« ÿλ Ϋ ‚ώ ƒ΄	 …ΐ ‡½ ‰†  ®	 † „  ’΄	 “¨ •† —” ® Ώ ›Π Γ – ™  3 Ά‡ £΅ ¥Ζ §¦ © ªΓ «3 ­‡ ®¨ °¬ ±¦ ³ ΄¤ µ² ·΅ Έ¥ Ί¦ Ό– ½Ή Ύ€ ΐν Α® ΓΏ Δ‡ ΕΓ ΖΏ ΘΒ Ι» Λ¨ Μ® ΞΏ Ο‡ ΠΓ ΡΚ ΣΝ Τκ ΦΥ ΨΡ ΪΩ ά€ ήΫ ΰΧ βί γέ δα ζ€ ηά ιθ λ νκ οΧ ρξ ςμ σπ υ φη ψχ ϊ όω ώΧ €ύ ϋ ‚ÿ „ …ή ‡	 ‰ ‹ Ώ †  ‘ή ’™ ”®	 –• — Κ ›“ ™ ™   ΆΘ ¥™ ¦¤ ¨Κ ©¨ « ¬ª ®Ώ ― ±ÿ ²Χ ΄ω µτ ·π ΈΥ Ίκ »ΰ ½α ΎΣ ΐΫ ΑΟ Γ Δ+ ΖΒ ΗΕ Ι+ ΛΒ ΜΚ ΞΘ ΠΝ Ò® ΤΏ ΥΒ ΦΓ ΧΣ ΩΟ ΫΌ άΨ έΏ ίΡ αή βΪ γΰ εΣ ζ® θΏ ιΒ κΓ λη νΟ ο¶ πμ ρΉ σΡ υς φξ χτ ωη ϊ® όΏ ύΒ ώΓ ÿϋ Ο ƒ° „€ …³ ‡Ρ ‰† ‚ ‹ ϋ ® Ώ ‘Β ’Γ “ •/ —Β – ™ › ª ”  / ΆΒ £΅ ¥¤ §¦ ©­ ª «¨ ­ ®® °Ώ ±Β ²Γ ³― µ3 ·Β Έ¶ ΊΉ Ό» Ύ¤ Ώ΄ ΐ3 ΒΒ ΓΑ ΕΔ ΗΖ Ι§ Κ½ ΛΘ Ν― ΞΒ ΠΒ ÒΌ Τ¶ Φ° ΨΡ Ϊ Ϋ λ ½λ νς	 Νς	 τ	ο π ¥ ΅φ	 χ	» ½» π΅ £΅ Ϋ£ ¤Λ ΝΛ χ	£ ¤£ ¥£ ¤Ω ¤Ω Ϋ Ϋ ιι κκ κκ ΄ κκ ΄¤ κκ ¤	 κκ 	Φ κκ Φ³ κκ ³π κκ πέ
 κκ έ
Β κκ ΒΛ κκ Λƒ κκ ƒ€ κκ € κκ χ κκ χΔ κκ Δ¨ κκ ¨² κκ ²» κκ »¤ κκ ¤” κκ ”ÿ κκ ÿ κκ  κκ ι
 κκ ι
Ό κκ Ό‹ κκ ‹¨ κκ ¨½ κκ ½ψ κκ ψέ κκ έξ κκ ξ° κκ °Β κκ Β² κκ ²ΰ κκ ΰ΅ κκ ΅™ κκ ™η κκ ηι κκ ιΏ κκ Ώª κκ ª΅ κκ ΅« κκ «‰ κκ ‰‚ κκ ‚¨ κκ ¨κ κκ κο
 κκ ο
 ιι ϊ κκ ϊΈ κκ Έ‚ κκ ‚ κκ ξ κκ ξη κκ ηή κκ ήν κκ νΠ κκ Π¬ κκ ¬ψ κκ ψ κκ ΅ κκ ΅Ξ κκ Ξα κκ α™ κκ ™ΰ κκ ΰτ κκ τ κκ Θ κκ Θ› κκ ›ΐ κκ ΐΦ κκ Φ• κκ •Θ κκ Θ κκ  κκ  κκ  ιι «	 κκ «	Ϊ κκ Ϊ” κκ ”έ κκ έΗ κκ Ηθ κκ θΆ κκ Άω κκ ωΦ κκ Φδ κκ δ² κκ ²‰	 κκ ‰	ν κκ ν κκ Ψ κκ Ψ© κκ ©Ω κκ Ωυ
 κκ υ
― κκ ―φ κκ φ³ κκ ³
λ ψ
λ Ϋ
μ ’
μ –
μ 
μ 
μ £
μ ¨
μ 
μ ª
μ Η
μ «
μ Θ
μ ε
μ ο
μ ‘
μ ³
μ ½
μ ί
μ 
μ π
μ ’	
μ ΄	
μ ™

μ ‡
μ —
μ Ξ
μ ³
μ Ξ
μ €
μ ™
μ Ν
μ Κ
μ ΅
μ ―
μ Α
ν ½
ν Ώ
ν Α
ν Γξ μξ ξ ξ ξ Ιξ βξ —ξ °ξ Μξ εξ Ϋ
ξ ξ §ξ φξ Άξ Ϋξ ‡ξ λξ ¦ξ Χξ ξ —ξ Οξ Ρξ ›ξ ¦ξ »ξ Ζ
ο κ
π ξ
π •
π Ώ
π 
π ³
π έ
π Λ
π ω
π ª
π ™
π Η
π ψ
π Ξ
π ϊ
π «	
ρ  
ς Φ
ς ‹
ς Β
ς ή
ς 
ς θ
ς 
σ –
σ Ψ
σ ¦τ τ ¬τ Ιτ ­τ Κτ ητ Ετ ςτ χτ ”τ ™τ ¶τ “τ Ετ ητ Θτ ψτ 		υ 	υ 
υ †
φ Λ
φ ώ
φ µ
φ Ρ
φ 
φ Ω
φ 
χ Ο	ψ h	ψ j	ψ l	ψ n	ψ q	ψ t
ψ ―
ψ ·
ψ ½
ψ Ϋ
ψ ΰ
ψ ϊ
ψ 
ψ Ί
ψ 
ψ »
ψ Ψ
ψ Ϊ
ψ ‚
ψ ¤
ψ ¨
ψ Π
ψ ς
ψ έ
ψ ƒ	
ψ ¥	
ψ Ρ	
ψ ι	
ψ Ξ

ψ α

ψ ω

ψ …
ψ 
ψ 
ψ ­
ψ η
ψ Μ
ψ ό
ψ –
ψ α
ψ ϋ
ψ ρ
ψ ¬
ψ ϋ
ω ρ	
ϊ †
ϊ 
ϊ ¤
ϊ Ξ
ϊ η
ϊ 
ϊ µ
ϊ Ρ
ϋ Ο
ό ζ
ό ‘
ό ®	ύ 
ώ φ
ώ ”
ώ Φ
ώ ¤
ώ Ω
ÿ κ
ÿ ÿ
ÿ †
ÿ 
ÿ Η
ÿ ΰ
ÿ •
ÿ ®
ÿ Κ
ÿ γ
€ ‚
€ ―
€ Μ	 w	 {	 
 ƒ
 
 
 
 ¥
 Β
 Λ
 Σ
 Ω
 χ
 ό
 ¦
 Γ
 ΰ
 π
 ι
 ‹
 ­
 ·
 Ω
 ϋ
 κ
 	
 ®	
 £

 Φ

 Ά
 ÿ
 
 Ψ
 ι
 ρ
 ›
 ½
 Φ
 ή
 ζ
 ΅
 Β
 Ε
 
 –
 ¶
‚ ƒ
‚ Ά
‚ ²
‚ ΅
‚ ΐ
‚ Π
‚ δ
‚ 
‚ ›
‚ ²
‚ Φ
‚ ι
‚ η
‚ ‰	
‚ 	ƒ 	„ %… ±… ³… µ… ·… Ή… »
… φ
… ”
… Φ
… ¤
… Ω… 
… ν… Μ… Σ… Έ… Λ… †
† ι	‡ 	‡ 	‡ M	‡ Q	‡ U	‡ Y	‡ ^	‡ c
‡ ψ
‡ €
‡ †
‡ ¤
‡ ©
‡ ρ
‡ 
‡ µ
‡ 
‡ ¶
‡ Σ
‡ „
‡ Π
‡ ό
‡ 
‡ 
‡ Κ
‡ μ
‡ Σ
‡ ύ
‡ 	
‡ Ι	
‡ Ω	
‡ α	
‡ α	
‡ ι	
‡ —

‡ Α

‡ €
‡ ‘
‡ 
‡ ά
‡ 
‡ —
‡ Κ
‡ ‘
‡ φ
‡ η 	 5
‰ Φ
‰ Ψ
‰ α
‰ γ
‰ ε
‰ ‹
‰ 
‰ –
‰ 
‰ 
‰ Β
‰ Δ
‰ Ν
‰ Ο
‰ Ρ
‰ ή
‰ ΰ
‰ ι
‰ λ
‰ ν
‰ 
‰ 
‰ §
‰ ©
‰ «
‰ θ
‰ κ
‰ σ
‰ υ
‰ χ
‰ 
‰ ΅
‰ ª
‰ ¬
‰ ® 8 : < A F K O S W \ a f y }  † ‹  ”   ΅ ¦ « θ “ ° „ ± Ξ ΐ β „ σ •	 ·		‹ >	‹ C	‹ H
‹ Ε
‹ Ν
‹ ς
‹ Β
‹ τ
‹ –
‹ 
‹ Β
‹ δ
‹ Ε
‹ υ
‹ —	
‹ Α	
‹ Ι	
‹ Ρ	
‹ ‰

‹ ΄

‹ ϋ

‹ 
‹ €
‹ Ρ
‹ ÿ
‹ ‡
‹ °
‹ 
‹ ρ
‹ Σ
‹ Ρ
 Ψ
 
 Δ
 ΰ
 
 κ
 ΅"	
y_solve"
_Z13get_global_idj"
llvm.fmuladd.f64*
npb-SP-y_solve.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize_log1p
|A
 
transfer_bytes_log1p
|A

devmap_label


wgsize
 

transfer_bytes	
θυσΨ