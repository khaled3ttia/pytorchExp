

[external]
@allocaB6
4
	full_text'
%
#%14 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%15 = alloca [5 x double], align 16
DbitcastB9
7
	full_text*
(
&%16 = bitcast [5 x double]* %15 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %15
DbitcastB9
7
	full_text*
(
&%17 = bitcast [5 x double]* %14 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %14
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %17) #4
#i8*B

	full_text
	
i8* %17
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %16) #4
#i8*B

	full_text
	
i8* %16
LcallBD
B
	full_text5
3
1%18 = tail call i64 @_Z13get_global_idj(i32 1) #5
.addB'
%
	full_text

%19 = add i64 %18, 1
#i64B

	full_text
	
i64 %18
6truncB-
+
	full_text

%20 = trunc i64 %19 to i32
#i64B

	full_text
	
i64 %19
LcallBD
B
	full_text5
3
1%21 = tail call i64 @_Z13get_global_idj(i32 0) #5
.addB'
%
	full_text

%22 = add i64 %21, 1
#i64B

	full_text
	
i64 %21
6truncB-
+
	full_text

%23 = trunc i64 %22 to i32
#i64B

	full_text
	
i64 %22
6icmpB.
,
	full_text

%24 = icmp sgt i32 %20, %10
#i32B

	full_text
	
i32 %20
5icmpB-
+
	full_text

%25 = icmp sgt i32 %23, %9
#i32B

	full_text
	
i32 %23
-orB'
%
	full_text

%26 = or i1 %24, %25
!i1B

	full_text


i1 %24
!i1B

	full_text


i1 %25
9brB3
1
	full_text$
"
 br i1 %26, label %792, label %27
!i1B

	full_text


i1 %26
Qbitcast8BD
B
	full_text5
3
1%28 = bitcast double* %0 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%29 = bitcast double* %2 to [65 x [65 x double]]*
5add8B,
*
	full_text

%30 = add nsw i32 %20, -1
%i328B

	full_text
	
i32 %20
5mul8B,
*
	full_text

%31 = mul nsw i32 %30, %9
%i328B

	full_text
	
i32 %30
5add8B,
*
	full_text

%32 = add nsw i32 %23, -1
%i328B

	full_text
	
i32 %23
6add8B-
+
	full_text

%33 = add nsw i32 %32, %31
%i328B

	full_text
	
i32 %32
%i328B

	full_text
	
i32 %31
4shl8B+
)
	full_text

%34 = shl nsw i32 %33, 6
%i328B

	full_text
	
i32 %33
6sext8B,
*
	full_text

%35 = sext i32 %34 to i64
%i328B

	full_text
	
i32 %34
^getelementptr8BK
I
	full_text<
:
8%36 = getelementptr inbounds double, double* %4, i64 %35
%i648B

	full_text
	
i64 %35
2mul8B)
'
	full_text

%37 = mul i32 %33, 325
%i328B

	full_text
	
i32 %33
6sext8B,
*
	full_text

%38 = sext i32 %37 to i64
%i328B

	full_text
	
i32 %37
^getelementptr8BK
I
	full_text<
:
8%39 = getelementptr inbounds double, double* %6, i64 %38
%i648B

	full_text
	
i64 %38
Jbitcast8B=
;
	full_text.
,
*%40 = bitcast double* %39 to [5 x double]*
-double*8B

	full_text

double* %39
^getelementptr8BK
I
	full_text<
:
8%41 = getelementptr inbounds double, double* %7, i64 %38
%i648B

	full_text
	
i64 %38
Jbitcast8B=
;
	full_text.
,
*%42 = bitcast double* %41 to [5 x double]*
-double*8B

	full_text

double* %41
^getelementptr8BK
I
	full_text<
:
8%43 = getelementptr inbounds double, double* %8, i64 %38
%i648B

	full_text
	
i64 %38
Jbitcast8B=
;
	full_text.
,
*%44 = bitcast double* %43 to [5 x double]*
-double*8B

	full_text

double* %43
4add8B+
)
	full_text

%45 = add nsw i32 %11, 1
6sext8B,
*
	full_text

%46 = sext i32 %45 to i64
%i328B

	full_text
	
i32 %45
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %39, align 8, !tbaa !8
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %41, align 8, !tbaa !8
-double*8B

	full_text

double* %41
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
L%47 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %46, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
rgetelementptr8B_
]
	full_textP
N
L%48 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %46, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
%i648B

	full_text
	
i64 %46
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
L%49 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %46, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
]getelementptr8BJ
H
	full_text;
9
7%50 = getelementptr inbounds double, double* %39, i64 1
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
]getelementptr8BJ
H
	full_text;
9
7%51 = getelementptr inbounds double, double* %41, i64 1
-double*8B

	full_text

double* %41
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
7%52 = getelementptr inbounds double, double* %43, i64 1
-double*8B

	full_text

double* %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
rgetelementptr8B_
]
	full_textP
N
L%53 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %46, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
rgetelementptr8B_
]
	full_textP
N
L%54 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %46, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
rgetelementptr8B_
]
	full_textP
N
L%55 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %46, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
]getelementptr8BJ
H
	full_text;
9
7%56 = getelementptr inbounds double, double* %39, i64 2
-double*8B

	full_text

double* %39
]getelementptr8BJ
H
	full_text;
9
7%57 = getelementptr inbounds double, double* %41, i64 2
-double*8B

	full_text

double* %41
]getelementptr8BJ
H
	full_text;
9
7%58 = getelementptr inbounds double, double* %43, i64 2
-double*8B

	full_text

double* %43
rgetelementptr8B_
]
	full_textP
N
L%59 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %46, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %46
rgetelementptr8B_
]
	full_textP
N
L%60 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %46, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
%i648B

	full_text
	
i64 %46
rgetelementptr8B_
]
	full_textP
N
L%61 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %46, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
%i648B

	full_text
	
i64 %46
]getelementptr8BJ
H
	full_text;
9
7%62 = getelementptr inbounds double, double* %39, i64 3
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
]getelementptr8BJ
H
	full_text;
9
7%63 = getelementptr inbounds double, double* %41, i64 3
-double*8B

	full_text

double* %41
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
7%64 = getelementptr inbounds double, double* %43, i64 3
-double*8B

	full_text

double* %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
rgetelementptr8B_
]
	full_textP
N
L%65 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %46, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %65, align 8, !tbaa !8
-double*8B

	full_text

double* %65
rgetelementptr8B_
]
	full_textP
N
L%66 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %46, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
%i648B

	full_text
	
i64 %46
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
L%67 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %46, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %67, align 8, !tbaa !8
-double*8B

	full_text

double* %67
]getelementptr8BJ
H
	full_text;
9
7%68 = getelementptr inbounds double, double* %39, i64 4
-double*8B

	full_text

double* %39
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
]getelementptr8BJ
H
	full_text;
9
7%69 = getelementptr inbounds double, double* %41, i64 4
-double*8B

	full_text

double* %41
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %69, align 8, !tbaa !8
-double*8B

	full_text

double* %69
]getelementptr8BJ
H
	full_text;
9
7%70 = getelementptr inbounds double, double* %43, i64 4
-double*8B

	full_text

double* %43
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %70, align 8, !tbaa !8
-double*8B

	full_text

double* %70
rgetelementptr8B_
]
	full_textP
N
L%71 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %46, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %71, align 8, !tbaa !8
-double*8B

	full_text

double* %71
rgetelementptr8B_
]
	full_textP
N
L%72 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %46, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %72, align 8, !tbaa !8
-double*8B

	full_text

double* %72
rgetelementptr8B_
]
	full_textP
N
L%73 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %46, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
%i648B

	full_text
	
i64 %46
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %73, align 8, !tbaa !8
-double*8B

	full_text

double* %73
Qbitcast8BD
B
	full_text5
3
1%74 = bitcast double* %1 to [65 x [65 x double]]*
Wbitcast8BJ
H
	full_text;
9
7%75 = bitcast double* %3 to [65 x [65 x [5 x double]]]*
^getelementptr8BK
I
	full_text<
:
8%76 = getelementptr inbounds double, double* %5, i64 %35
%i648B

	full_text
	
i64 %35
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
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %59, align 8, !tbaa !8
-double*8B

	full_text

double* %59
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %60, align 8, !tbaa !8
-double*8B

	full_text

double* %60
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
1shl8B(
&
	full_text

%77 = shl i64 %19, 32
%i648B

	full_text
	
i64 %19
9ashr8B/
-
	full_text 

%78 = ashr exact i64 %77, 32
%i648B

	full_text
	
i64 %77
1shl8B(
&
	full_text

%79 = shl i64 %22, 32
%i648B

	full_text
	
i64 %22
9ashr8B/
-
	full_text 

%80 = ashr exact i64 %79, 32
%i648B

	full_text
	
i64 %79
ãgetelementptr8Bx
v
	full_texti
g
e%81 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %74, i64 0, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %74
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Nload8BD
B
	full_text5
3
1%82 = load double, double* %81, align 8, !tbaa !8
-double*8B

	full_text

double* %81
@fmul8B6
4
	full_text'
%
#%83 = fmul double %82, 1.000000e-01
+double8B

	full_text


double %82
ãgetelementptr8Bx
v
	full_texti
g
e%84 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 0, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Nload8BD
B
	full_text5
3
1%85 = load double, double* %84, align 8, !tbaa !8
-double*8B

	full_text

double* %84
Nstore8BC
A
	full_text4
2
0store double %85, double* %36, align 8, !tbaa !8
+double8B

	full_text


double %85
-double*8B

	full_text

double* %36
Åcall8Bw
u
	full_texth
f
d%86 = tail call double @llvm.fmuladd.f64(double %83, double 0x3FF5555555555555, double 1.000000e+00)
+double8B

	full_text


double %83
Åcall8Bw
u
	full_texth
f
d%87 = tail call double @llvm.fmuladd.f64(double %83, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
+double8B

	full_text


double %83
;fcmp8B1
/
	full_text"
 
%88 = fcmp ogt double %86, %87
+double8B

	full_text


double %86
+double8B

	full_text


double %87
Jselect8B>
<
	full_text/
-
+%89 = select i1 %88, double %86, double %87
#i18B

	full_text


i1 %88
+double8B

	full_text


double %86
+double8B

	full_text


double %87
@fadd8B6
4
	full_text'
%
#%90 = fadd double %83, 1.000000e+00
+double8B

	full_text


double %83
Dfcmp8B:
8
	full_text+
)
'%91 = fcmp ogt double %90, 1.000000e+00
+double8B

	full_text


double %90
Sselect8BG
E
	full_text8
6
4%92 = select i1 %91, double %90, double 1.000000e+00
#i18B

	full_text


i1 %91
+double8B

	full_text


double %90
;fcmp8B1
/
	full_text"
 
%93 = fcmp ogt double %89, %92
+double8B

	full_text


double %89
+double8B

	full_text


double %92
Jselect8B>
<
	full_text/
-
+%94 = select i1 %93, double %89, double %92
#i18B

	full_text


i1 %93
+double8B

	full_text


double %89
+double8B

	full_text


double %92
Nstore8BC
A
	full_text4
2
0store double %94, double* %76, align 8, !tbaa !8
+double8B

	full_text


double %94
-double*8B

	full_text

double* %76
ãgetelementptr8Bx
v
	full_texti
g
e%95 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %29, i64 0, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %29
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Nload8BD
B
	full_text5
3
1%96 = load double, double* %95, align 8, !tbaa !8
-double*8B

	full_text

double* %95
_getelementptr8BL
J
	full_text=
;
9%97 = getelementptr inbounds double, double* %1, i64 4225
Rbitcast8BE
C
	full_text6
4
2%98 = bitcast double* %97 to [65 x [65 x double]]*
-double*8B

	full_text

double* %97
ãgetelementptr8Bx
v
	full_texti
g
e%99 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %98, i64 0, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %98
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Oload8BE
C
	full_text6
4
2%100 = load double, double* %99, align 8, !tbaa !8
-double*8B

	full_text

double* %99
Bfmul8B8
6
	full_text)
'
%%101 = fmul double %100, 1.000000e-01
,double8B

	full_text

double %100
`getelementptr8BM
K
	full_text>
<
:%102 = getelementptr inbounds double, double* %0, i64 4225
Tbitcast8BG
E
	full_text8
6
4%103 = bitcast double* %102 to [65 x [65 x double]]*
.double*8B

	full_text

double* %102
çgetelementptr8Bz
x
	full_textk
i
g%104 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %103, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %103
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%105 = load double, double* %104, align 8, !tbaa !8
.double*8B

	full_text

double* %104
^getelementptr8BK
I
	full_text<
:
8%106 = getelementptr inbounds double, double* %36, i64 1
-double*8B

	full_text

double* %36
Pstore8BE
C
	full_text6
4
2store double %105, double* %106, align 8, !tbaa !8
,double8B

	full_text

double %105
.double*8B

	full_text

double* %106
Écall8By
w
	full_textj
h
f%107 = tail call double @llvm.fmuladd.f64(double %101, double 0x3FF5555555555555, double 1.000000e+00)
,double8B

	full_text

double %101
Écall8By
w
	full_textj
h
f%108 = tail call double @llvm.fmuladd.f64(double %101, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
,double8B

	full_text

double %101
>fcmp8B4
2
	full_text%
#
!%109 = fcmp ogt double %107, %108
,double8B

	full_text

double %107
,double8B

	full_text

double %108
Nselect8BB
@
	full_text3
1
/%110 = select i1 %109, double %107, double %108
$i18B

	full_text
	
i1 %109
,double8B

	full_text

double %107
,double8B

	full_text

double %108
Bfadd8B8
6
	full_text)
'
%%111 = fadd double %101, 1.000000e+00
,double8B

	full_text

double %101
Ffcmp8B<
:
	full_text-
+
)%112 = fcmp ogt double %111, 1.000000e+00
,double8B

	full_text

double %111
Vselect8BJ
H
	full_text;
9
7%113 = select i1 %112, double %111, double 1.000000e+00
$i18B

	full_text
	
i1 %112
,double8B

	full_text

double %111
>fcmp8B4
2
	full_text%
#
!%114 = fcmp ogt double %110, %113
,double8B

	full_text

double %110
,double8B

	full_text

double %113
Nselect8BB
@
	full_text3
1
/%115 = select i1 %114, double %110, double %113
$i18B

	full_text
	
i1 %114
,double8B

	full_text

double %110
,double8B

	full_text

double %113
^getelementptr8BK
I
	full_text<
:
8%116 = getelementptr inbounds double, double* %76, i64 1
-double*8B

	full_text

double* %76
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
`getelementptr8BM
K
	full_text>
<
:%117 = getelementptr inbounds double, double* %2, i64 4225
Tbitcast8BG
E
	full_text8
6
4%118 = bitcast double* %117 to [65 x [65 x double]]*
.double*8B

	full_text

double* %117
çgetelementptr8Bz
x
	full_textk
i
g%119 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %118, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %118
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%120 = load double, double* %119, align 8, !tbaa !8
.double*8B

	full_text

double* %119
`getelementptr8BM
K
	full_text>
<
:%121 = getelementptr inbounds double, double* %1, i64 8450
Tbitcast8BG
E
	full_text8
6
4%122 = bitcast double* %121 to [65 x [65 x double]]*
.double*8B

	full_text

double* %121
çgetelementptr8Bz
x
	full_textk
i
g%123 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %122, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %122
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%124 = load double, double* %123, align 8, !tbaa !8
.double*8B

	full_text

double* %123
Bfmul8B8
6
	full_text)
'
%%125 = fmul double %124, 1.000000e-01
,double8B

	full_text

double %124
`getelementptr8BM
K
	full_text>
<
:%126 = getelementptr inbounds double, double* %0, i64 8450
Tbitcast8BG
E
	full_text8
6
4%127 = bitcast double* %126 to [65 x [65 x double]]*
.double*8B

	full_text

double* %126
çgetelementptr8Bz
x
	full_textk
i
g%128 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %127, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %127
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
^getelementptr8BK
I
	full_text<
:
8%130 = getelementptr inbounds double, double* %36, i64 2
-double*8B

	full_text

double* %36
Pstore8BE
C
	full_text6
4
2store double %129, double* %130, align 8, !tbaa !8
,double8B

	full_text

double %129
.double*8B

	full_text

double* %130
Écall8By
w
	full_textj
h
f%131 = tail call double @llvm.fmuladd.f64(double %125, double 0x3FF5555555555555, double 1.000000e+00)
,double8B

	full_text

double %125
Écall8By
w
	full_textj
h
f%132 = tail call double @llvm.fmuladd.f64(double %125, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
,double8B

	full_text

double %125
>fcmp8B4
2
	full_text%
#
!%133 = fcmp ogt double %131, %132
,double8B

	full_text

double %131
,double8B

	full_text

double %132
Nselect8BB
@
	full_text3
1
/%134 = select i1 %133, double %131, double %132
$i18B

	full_text
	
i1 %133
,double8B

	full_text

double %131
,double8B

	full_text

double %132
Bfadd8B8
6
	full_text)
'
%%135 = fadd double %125, 1.000000e+00
,double8B

	full_text

double %125
Ffcmp8B<
:
	full_text-
+
)%136 = fcmp ogt double %135, 1.000000e+00
,double8B

	full_text

double %135
Vselect8BJ
H
	full_text;
9
7%137 = select i1 %136, double %135, double 1.000000e+00
$i18B

	full_text
	
i1 %136
,double8B

	full_text

double %135
>fcmp8B4
2
	full_text%
#
!%138 = fcmp ogt double %134, %137
,double8B

	full_text

double %134
,double8B

	full_text

double %137
Nselect8BB
@
	full_text3
1
/%139 = select i1 %138, double %134, double %137
$i18B

	full_text
	
i1 %138
,double8B

	full_text

double %134
,double8B

	full_text

double %137
^getelementptr8BK
I
	full_text<
:
8%140 = getelementptr inbounds double, double* %76, i64 2
-double*8B

	full_text

double* %76
Pstore8BE
C
	full_text6
4
2store double %139, double* %140, align 8, !tbaa !8
,double8B

	full_text

double %139
.double*8B

	full_text

double* %140
`getelementptr8BM
K
	full_text>
<
:%141 = getelementptr inbounds double, double* %2, i64 8450
Tbitcast8BG
E
	full_text8
6
4%142 = bitcast double* %141 to [65 x [65 x double]]*
.double*8B

	full_text

double* %141
çgetelementptr8Bz
x
	full_textk
i
g%143 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %142, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %142
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%144 = load double, double* %143, align 8, !tbaa !8
.double*8B

	full_text

double* %143
^getelementptr8BK
I
	full_text<
:
8%145 = getelementptr inbounds double, double* %39, i64 5
-double*8B

	full_text

double* %39
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %145, align 8, !tbaa !8
.double*8B

	full_text

double* %145
Gfmul8B=
;
	full_text.
,
*%146 = fmul double %94, 0x4017D0624DD2F1AB
+double8B

	full_text


double %94
Cfsub8B9
7
	full_text*
(
&%147 = fsub double -0.000000e+00, %146
,double8B

	full_text

double %146
ucall8Bk
i
	full_text\
Z
X%148 = tail call double @llvm.fmuladd.f64(double %85, double -4.725000e-02, double %147)
+double8B

	full_text


double %85
,double8B

	full_text

double %147
_getelementptr8BL
J
	full_text=
;
9%149 = getelementptr inbounds double, double* %145, i64 1
.double*8B

	full_text

double* %145
Pstore8BE
C
	full_text6
4
2store double %148, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %148
.double*8B

	full_text

double* %149
Écall8By
w
	full_textj
h
f%150 = tail call double @llvm.fmuladd.f64(double %115, double 0x4027D0624DD2F1AB, double 1.000000e+00)
,double8B

	full_text

double %115
Bfadd8B8
6
	full_text)
'
%%151 = fadd double %150, 1.875000e-03
,double8B

	full_text

double %150
_getelementptr8BL
J
	full_text=
;
9%152 = getelementptr inbounds double, double* %145, i64 2
.double*8B

	full_text

double* %145
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
Hfmul8B>
<
	full_text/
-
+%153 = fmul double %139, 0x4017D0624DD2F1AB
,double8B

	full_text

double %139
Cfsub8B9
7
	full_text*
(
&%154 = fsub double -0.000000e+00, %153
,double8B

	full_text

double %153
ucall8Bk
i
	full_text\
Z
X%155 = tail call double @llvm.fmuladd.f64(double %129, double 4.725000e-02, double %154)
,double8B

	full_text

double %129
,double8B

	full_text

double %154
Cfadd8B9
7
	full_text*
(
&%156 = fadd double %155, -1.500000e-03
,double8B

	full_text

double %155
_getelementptr8BL
J
	full_text=
;
9%157 = getelementptr inbounds double, double* %145, i64 3
.double*8B

	full_text

double* %145
Pstore8BE
C
	full_text6
4
2store double %156, double* %157, align 8, !tbaa !8
,double8B

	full_text

double %156
.double*8B

	full_text

double* %157
_getelementptr8BL
J
	full_text=
;
9%158 = getelementptr inbounds double, double* %145, i64 4
.double*8B

	full_text

double* %145
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
^getelementptr8BK
I
	full_text<
:
8%159 = getelementptr inbounds double, double* %41, i64 5
-double*8B

	full_text

double* %41
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %159, align 8, !tbaa !8
.double*8B

	full_text

double* %159
ucall8Bk
i
	full_text\
Z
X%160 = tail call double @llvm.fmuladd.f64(double %96, double -4.725000e-02, double %148)
+double8B

	full_text


double %96
,double8B

	full_text

double %148
_getelementptr8BL
J
	full_text=
;
9%161 = getelementptr inbounds double, double* %159, i64 1
.double*8B

	full_text

double* %159
Pstore8BE
C
	full_text6
4
2store double %160, double* %161, align 8, !tbaa !8
,double8B

	full_text

double %160
.double*8B

	full_text

double* %161
_getelementptr8BL
J
	full_text=
;
9%162 = getelementptr inbounds double, double* %159, i64 2
.double*8B

	full_text

double* %159
Pstore8BE
C
	full_text6
4
2store double %151, double* %162, align 8, !tbaa !8
,double8B

	full_text

double %151
.double*8B

	full_text

double* %162
ucall8Bk
i
	full_text\
Z
X%163 = tail call double @llvm.fmuladd.f64(double %144, double 4.725000e-02, double %156)
,double8B

	full_text

double %144
,double8B

	full_text

double %156
_getelementptr8BL
J
	full_text=
;
9%164 = getelementptr inbounds double, double* %159, i64 3
.double*8B

	full_text

double* %159
Pstore8BE
C
	full_text6
4
2store double %163, double* %164, align 8, !tbaa !8
,double8B

	full_text

double %163
.double*8B

	full_text

double* %164
_getelementptr8BL
J
	full_text=
;
9%165 = getelementptr inbounds double, double* %159, i64 4
.double*8B

	full_text

double* %159
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %165, align 8, !tbaa !8
.double*8B

	full_text

double* %165
^getelementptr8BK
I
	full_text<
:
8%166 = getelementptr inbounds double, double* %43, i64 5
-double*8B

	full_text

double* %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %166, align 8, !tbaa !8
.double*8B

	full_text

double* %166
tcall8Bj
h
	full_text[
Y
W%167 = tail call double @llvm.fmuladd.f64(double %96, double 4.725000e-02, double %148)
+double8B

	full_text


double %96
,double8B

	full_text

double %148
_getelementptr8BL
J
	full_text=
;
9%168 = getelementptr inbounds double, double* %166, i64 1
.double*8B

	full_text

double* %166
Pstore8BE
C
	full_text6
4
2store double %167, double* %168, align 8, !tbaa !8
,double8B

	full_text

double %167
.double*8B

	full_text

double* %168
_getelementptr8BL
J
	full_text=
;
9%169 = getelementptr inbounds double, double* %166, i64 2
.double*8B

	full_text

double* %166
Pstore8BE
C
	full_text6
4
2store double %151, double* %169, align 8, !tbaa !8
,double8B

	full_text

double %151
.double*8B

	full_text

double* %169
vcall8Bl
j
	full_text]
[
Y%170 = tail call double @llvm.fmuladd.f64(double %144, double -4.725000e-02, double %156)
,double8B

	full_text

double %144
,double8B

	full_text

double %156
_getelementptr8BL
J
	full_text=
;
9%171 = getelementptr inbounds double, double* %166, i64 3
.double*8B

	full_text

double* %166
Pstore8BE
C
	full_text6
4
2store double %170, double* %171, align 8, !tbaa !8
,double8B

	full_text

double %170
.double*8B

	full_text

double* %171
_getelementptr8BL
J
	full_text=
;
9%172 = getelementptr inbounds double, double* %166, i64 4
.double*8B

	full_text

double* %166
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %172, align 8, !tbaa !8
.double*8B

	full_text

double* %172
agetelementptr8BN
L
	full_text?
=
;%173 = getelementptr inbounds double, double* %1, i64 12675
Tbitcast8BG
E
	full_text8
6
4%174 = bitcast double* %173 to [65 x [65 x double]]*
.double*8B

	full_text

double* %173
çgetelementptr8Bz
x
	full_textk
i
g%175 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %174, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %174
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%176 = load double, double* %175, align 8, !tbaa !8
.double*8B

	full_text

double* %175
Bfmul8B8
6
	full_text)
'
%%177 = fmul double %176, 1.000000e-01
,double8B

	full_text

double %176
agetelementptr8BN
L
	full_text?
=
;%178 = getelementptr inbounds double, double* %0, i64 12675
Tbitcast8BG
E
	full_text8
6
4%179 = bitcast double* %178 to [65 x [65 x double]]*
.double*8B

	full_text

double* %178
çgetelementptr8Bz
x
	full_textk
i
g%180 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %179, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %179
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%181 = load double, double* %180, align 8, !tbaa !8
.double*8B

	full_text

double* %180
^getelementptr8BK
I
	full_text<
:
8%182 = getelementptr inbounds double, double* %36, i64 3
-double*8B

	full_text

double* %36
Pstore8BE
C
	full_text6
4
2store double %181, double* %182, align 8, !tbaa !8
,double8B

	full_text

double %181
.double*8B

	full_text

double* %182
Écall8By
w
	full_textj
h
f%183 = tail call double @llvm.fmuladd.f64(double %177, double 0x3FF5555555555555, double 1.000000e+00)
,double8B

	full_text

double %177
Écall8By
w
	full_textj
h
f%184 = tail call double @llvm.fmuladd.f64(double %177, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
,double8B

	full_text

double %177
>fcmp8B4
2
	full_text%
#
!%185 = fcmp ogt double %183, %184
,double8B

	full_text

double %183
,double8B

	full_text

double %184
Nselect8BB
@
	full_text3
1
/%186 = select i1 %185, double %183, double %184
$i18B

	full_text
	
i1 %185
,double8B

	full_text

double %183
,double8B

	full_text

double %184
Bfadd8B8
6
	full_text)
'
%%187 = fadd double %177, 1.000000e+00
,double8B

	full_text

double %177
Ffcmp8B<
:
	full_text-
+
)%188 = fcmp ogt double %187, 1.000000e+00
,double8B

	full_text

double %187
Vselect8BJ
H
	full_text;
9
7%189 = select i1 %188, double %187, double 1.000000e+00
$i18B

	full_text
	
i1 %188
,double8B

	full_text

double %187
>fcmp8B4
2
	full_text%
#
!%190 = fcmp ogt double %186, %189
,double8B

	full_text

double %186
,double8B

	full_text

double %189
Nselect8BB
@
	full_text3
1
/%191 = select i1 %190, double %186, double %189
$i18B

	full_text
	
i1 %190
,double8B

	full_text

double %186
,double8B

	full_text

double %189
^getelementptr8BK
I
	full_text<
:
8%192 = getelementptr inbounds double, double* %76, i64 3
-double*8B

	full_text

double* %76
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
agetelementptr8BN
L
	full_text?
=
;%193 = getelementptr inbounds double, double* %2, i64 12675
Tbitcast8BG
E
	full_text8
6
4%194 = bitcast double* %193 to [65 x [65 x double]]*
.double*8B

	full_text

double* %193
çgetelementptr8Bz
x
	full_textk
i
g%195 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %194, i64 0, i64 %78, i64 %80
J[65 x [65 x double]]*8B-
+
	full_text

[65 x [65 x double]]* %194
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%196 = load double, double* %195, align 8, !tbaa !8
.double*8B

	full_text

double* %195
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
Hfmul8B>
<
	full_text/
-
+%198 = fmul double %115, 0x4017D0624DD2F1AB
,double8B

	full_text

double %115
Cfsub8B9
7
	full_text*
(
&%199 = fsub double -0.000000e+00, %198
,double8B

	full_text

double %198
vcall8Bl
j
	full_text]
[
Y%200 = tail call double @llvm.fmuladd.f64(double %105, double -4.725000e-02, double %199)
,double8B

	full_text

double %105
,double8B

	full_text

double %199
Cfadd8B9
7
	full_text*
(
&%201 = fadd double %200, -1.500000e-03
,double8B

	full_text

double %200
_getelementptr8BL
J
	full_text=
;
9%202 = getelementptr inbounds double, double* %197, i64 1
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
Écall8By
w
	full_textj
h
f%203 = tail call double @llvm.fmuladd.f64(double %139, double 0x4027D0624DD2F1AB, double 1.000000e+00)
,double8B

	full_text

double %139
Hfadd8B>
<
	full_text/
-
+%204 = fadd double %203, 0x3F626E978D4FDF3C
,double8B

	full_text

double %203
_getelementptr8BL
J
	full_text=
;
9%205 = getelementptr inbounds double, double* %197, i64 2
.double*8B

	full_text

double* %197
Pstore8BE
C
	full_text6
4
2store double %204, double* %205, align 8, !tbaa !8
,double8B

	full_text

double %204
.double*8B

	full_text

double* %205
Hfmul8B>
<
	full_text/
-
+%206 = fmul double %191, 0x4017D0624DD2F1AB
,double8B

	full_text

double %191
Cfsub8B9
7
	full_text*
(
&%207 = fsub double -0.000000e+00, %206
,double8B

	full_text

double %206
ucall8Bk
i
	full_text\
Z
X%208 = tail call double @llvm.fmuladd.f64(double %181, double 4.725000e-02, double %207)
,double8B

	full_text

double %181
,double8B

	full_text

double %207
Cfadd8B9
7
	full_text*
(
&%209 = fadd double %208, -1.500000e-03
,double8B

	full_text

double %208
_getelementptr8BL
J
	full_text=
;
9%210 = getelementptr inbounds double, double* %197, i64 3
.double*8B

	full_text

double* %197
Pstore8BE
C
	full_text6
4
2store double %209, double* %210, align 8, !tbaa !8
,double8B

	full_text

double %209
.double*8B

	full_text

double* %210
_getelementptr8BL
J
	full_text=
;
9%211 = getelementptr inbounds double, double* %197, i64 4
.double*8B

	full_text

double* %197
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
_getelementptr8BL
J
	full_text=
;
9%212 = getelementptr inbounds double, double* %41, i64 10
-double*8B

	full_text

double* %41
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %212, align 8, !tbaa !8
.double*8B

	full_text

double* %212
vcall8Bl
j
	full_text]
[
Y%213 = tail call double @llvm.fmuladd.f64(double %120, double -4.725000e-02, double %201)
,double8B

	full_text

double %120
,double8B

	full_text

double %201
_getelementptr8BL
J
	full_text=
;
9%214 = getelementptr inbounds double, double* %212, i64 1
.double*8B

	full_text

double* %212
Pstore8BE
C
	full_text6
4
2store double %213, double* %214, align 8, !tbaa !8
,double8B

	full_text

double %213
.double*8B

	full_text

double* %214
_getelementptr8BL
J
	full_text=
;
9%215 = getelementptr inbounds double, double* %212, i64 2
.double*8B

	full_text

double* %212
Pstore8BE
C
	full_text6
4
2store double %204, double* %215, align 8, !tbaa !8
,double8B

	full_text

double %204
.double*8B

	full_text

double* %215
ucall8Bk
i
	full_text\
Z
X%216 = tail call double @llvm.fmuladd.f64(double %196, double 4.725000e-02, double %209)
,double8B

	full_text

double %196
,double8B

	full_text

double %209
_getelementptr8BL
J
	full_text=
;
9%217 = getelementptr inbounds double, double* %212, i64 3
.double*8B

	full_text

double* %212
Pstore8BE
C
	full_text6
4
2store double %216, double* %217, align 8, !tbaa !8
,double8B

	full_text

double %216
.double*8B

	full_text

double* %217
_getelementptr8BL
J
	full_text=
;
9%218 = getelementptr inbounds double, double* %212, i64 4
.double*8B

	full_text

double* %212
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %218, align 8, !tbaa !8
.double*8B

	full_text

double* %218
_getelementptr8BL
J
	full_text=
;
9%219 = getelementptr inbounds double, double* %43, i64 10
-double*8B

	full_text

double* %43
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %219, align 8, !tbaa !8
.double*8B

	full_text

double* %219
ucall8Bk
i
	full_text\
Z
X%220 = tail call double @llvm.fmuladd.f64(double %120, double 4.725000e-02, double %201)
,double8B

	full_text

double %120
,double8B

	full_text

double %201
_getelementptr8BL
J
	full_text=
;
9%221 = getelementptr inbounds double, double* %219, i64 1
.double*8B

	full_text

double* %219
Pstore8BE
C
	full_text6
4
2store double %220, double* %221, align 8, !tbaa !8
,double8B

	full_text

double %220
.double*8B

	full_text

double* %221
_getelementptr8BL
J
	full_text=
;
9%222 = getelementptr inbounds double, double* %219, i64 2
.double*8B

	full_text

double* %219
Pstore8BE
C
	full_text6
4
2store double %204, double* %222, align 8, !tbaa !8
,double8B

	full_text

double %204
.double*8B

	full_text

double* %222
vcall8Bl
j
	full_text]
[
Y%223 = tail call double @llvm.fmuladd.f64(double %196, double -4.725000e-02, double %209)
,double8B

	full_text

double %196
,double8B

	full_text

double %209
_getelementptr8BL
J
	full_text=
;
9%224 = getelementptr inbounds double, double* %219, i64 3
.double*8B

	full_text

double* %219
Pstore8BE
C
	full_text6
4
2store double %223, double* %224, align 8, !tbaa !8
,double8B

	full_text

double %223
.double*8B

	full_text

double* %224
_getelementptr8BL
J
	full_text=
;
9%225 = getelementptr inbounds double, double* %219, i64 4
.double*8B

	full_text

double* %219
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %225, align 8, !tbaa !8
.double*8B

	full_text

double* %225
7icmp8B-
+
	full_text

%226 = icmp slt i32 %11, 5
2add8B)
'
	full_text

%227 = add i32 %11, -1
=br8B5
3
	full_text&
$
"br i1 %226, label %287, label %228
$i18B

	full_text
	
i1 %226
8zext8B.
,
	full_text

%229 = zext i32 %227 to i64
&i328B

	full_text


i32 %227
(br8B 

	full_text

br label %230
Fphi8B=
;
	full_text.
,
*%231 = phi i64 [ %238, %230 ], [ 3, %228 ]
&i648B

	full_text


i64 %238
Lphi8BC
A
	full_text4
2
0%232 = phi double [ %256, %230 ], [ %196, %228 ]
,double8B

	full_text

double %256
,double8B

	full_text

double %196
Lphi8BC
A
	full_text4
2
0%233 = phi double [ %232, %230 ], [ %144, %228 ]
,double8B

	full_text

double %232
,double8B

	full_text

double %144
Lphi8BC
A
	full_text4
2
0%234 = phi double [ %253, %230 ], [ %191, %228 ]
,double8B

	full_text

double %253
,double8B

	full_text

double %191
Lphi8BC
A
	full_text4
2
0%235 = phi double [ %234, %230 ], [ %139, %228 ]
,double8B

	full_text

double %234
,double8B

	full_text

double %139
Lphi8BC
A
	full_text4
2
0%236 = phi double [ %243, %230 ], [ %181, %228 ]
,double8B

	full_text

double %243
,double8B

	full_text

double %181
Lphi8BC
A
	full_text4
2
0%237 = phi double [ %236, %230 ], [ %129, %228 ]
,double8B

	full_text

double %236
,double8B

	full_text

double %129
:add8B1
/
	full_text"
 
%238 = add nuw nsw i64 %231, 1
&i648B

	full_text


i64 %231
ègetelementptr8B|
z
	full_textm
k
i%239 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %74, i64 %238, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %74
&i648B

	full_text


i64 %238
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%240 = load double, double* %239, align 8, !tbaa !8
.double*8B

	full_text

double* %239
Bfmul8B8
6
	full_text)
'
%%241 = fmul double %240, 1.000000e-01
,double8B

	full_text

double %240
ègetelementptr8B|
z
	full_textm
k
i%242 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %238, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %238
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%243 = load double, double* %242, align 8, !tbaa !8
.double*8B

	full_text

double* %242
agetelementptr8BN
L
	full_text?
=
;%244 = getelementptr inbounds double, double* %36, i64 %238
-double*8B

	full_text

double* %36
&i648B

	full_text


i64 %238
Pstore8BE
C
	full_text6
4
2store double %243, double* %244, align 8, !tbaa !8
,double8B

	full_text

double %243
.double*8B

	full_text

double* %244
Écall8By
w
	full_textj
h
f%245 = tail call double @llvm.fmuladd.f64(double %241, double 0x3FF5555555555555, double 1.000000e+00)
,double8B

	full_text

double %241
Écall8By
w
	full_textj
h
f%246 = tail call double @llvm.fmuladd.f64(double %241, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
,double8B

	full_text

double %241
>fcmp8B4
2
	full_text%
#
!%247 = fcmp ogt double %245, %246
,double8B

	full_text

double %245
,double8B

	full_text

double %246
Nselect8BB
@
	full_text3
1
/%248 = select i1 %247, double %245, double %246
$i18B

	full_text
	
i1 %247
,double8B

	full_text

double %245
,double8B

	full_text

double %246
Bfadd8B8
6
	full_text)
'
%%249 = fadd double %241, 1.000000e+00
,double8B

	full_text

double %241
Ffcmp8B<
:
	full_text-
+
)%250 = fcmp ogt double %249, 1.000000e+00
,double8B

	full_text

double %249
Vselect8BJ
H
	full_text;
9
7%251 = select i1 %250, double %249, double 1.000000e+00
$i18B

	full_text
	
i1 %250
,double8B

	full_text

double %249
>fcmp8B4
2
	full_text%
#
!%252 = fcmp ogt double %248, %251
,double8B

	full_text

double %248
,double8B

	full_text

double %251
Nselect8BB
@
	full_text3
1
/%253 = select i1 %252, double %248, double %251
$i18B

	full_text
	
i1 %252
,double8B

	full_text

double %248
,double8B

	full_text

double %251
agetelementptr8BN
L
	full_text?
=
;%254 = getelementptr inbounds double, double* %76, i64 %238
-double*8B

	full_text

double* %76
&i648B

	full_text


i64 %238
Pstore8BE
C
	full_text6
4
2store double %253, double* %254, align 8, !tbaa !8
,double8B

	full_text

double %253
.double*8B

	full_text

double* %254
ègetelementptr8B|
z
	full_textm
k
i%255 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %29, i64 %238, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %29
&i648B

	full_text


i64 %238
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%256 = load double, double* %255, align 8, !tbaa !8
.double*8B

	full_text

double* %255
tgetelementptr8Ba
_
	full_textR
P
N%257 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %231, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %231
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %257, align 8, !tbaa !8
.double*8B

	full_text

double* %257
Hfmul8B>
<
	full_text/
-
+%258 = fmul double %235, 0x4017D0624DD2F1AB
,double8B

	full_text

double %235
Cfsub8B9
7
	full_text*
(
&%259 = fsub double -0.000000e+00, %258
,double8B

	full_text

double %258
vcall8Bl
j
	full_text]
[
Y%260 = tail call double @llvm.fmuladd.f64(double %237, double -4.725000e-02, double %259)
,double8B

	full_text

double %237
,double8B

	full_text

double %259
Cfadd8B9
7
	full_text*
(
&%261 = fadd double %260, -1.500000e-03
,double8B

	full_text

double %260
tgetelementptr8Ba
_
	full_textR
P
N%262 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %231, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %231
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
Écall8By
w
	full_textj
h
f%263 = tail call double @llvm.fmuladd.f64(double %234, double 0x4027D0624DD2F1AB, double 1.000000e+00)
,double8B

	full_text

double %234
Hfadd8B>
<
	full_text/
-
+%264 = fadd double %263, 0x3F626E978D4FDF3C
,double8B

	full_text

double %263
tgetelementptr8Ba
_
	full_textR
P
N%265 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %231, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %264, double* %265, align 8, !tbaa !8
,double8B

	full_text

double %264
.double*8B

	full_text

double* %265
Hfmul8B>
<
	full_text/
-
+%266 = fmul double %253, 0x4017D0624DD2F1AB
,double8B

	full_text

double %253
Cfsub8B9
7
	full_text*
(
&%267 = fsub double -0.000000e+00, %266
,double8B

	full_text

double %266
ucall8Bk
i
	full_text\
Z
X%268 = tail call double @llvm.fmuladd.f64(double %243, double 4.725000e-02, double %267)
,double8B

	full_text

double %243
,double8B

	full_text

double %267
Cfadd8B9
7
	full_text*
(
&%269 = fadd double %268, -1.500000e-03
,double8B

	full_text

double %268
tgetelementptr8Ba
_
	full_textR
P
N%270 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %231, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %269, double* %270, align 8, !tbaa !8
,double8B

	full_text

double %269
.double*8B

	full_text

double* %270
tgetelementptr8Ba
_
	full_textR
P
N%271 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %231, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %231
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %271, align 8, !tbaa !8
.double*8B

	full_text

double* %271
tgetelementptr8Ba
_
	full_textR
P
N%272 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %231, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %231
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %272, align 8, !tbaa !8
.double*8B

	full_text

double* %272
vcall8Bl
j
	full_text]
[
Y%273 = tail call double @llvm.fmuladd.f64(double %233, double -4.725000e-02, double %261)
,double8B

	full_text

double %233
,double8B

	full_text

double %261
tgetelementptr8Ba
_
	full_textR
P
N%274 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %231, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %273, double* %274, align 8, !tbaa !8
,double8B

	full_text

double %273
.double*8B

	full_text

double* %274
tgetelementptr8Ba
_
	full_textR
P
N%275 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %231, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %264, double* %275, align 8, !tbaa !8
,double8B

	full_text

double %264
.double*8B

	full_text

double* %275
ucall8Bk
i
	full_text\
Z
X%276 = tail call double @llvm.fmuladd.f64(double %256, double 4.725000e-02, double %269)
,double8B

	full_text

double %256
,double8B

	full_text

double %269
tgetelementptr8Ba
_
	full_textR
P
N%277 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %231, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %276, double* %277, align 8, !tbaa !8
,double8B

	full_text

double %276
.double*8B

	full_text

double* %277
tgetelementptr8Ba
_
	full_textR
P
N%278 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %231, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %231
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %278, align 8, !tbaa !8
.double*8B

	full_text

double* %278
tgetelementptr8Ba
_
	full_textR
P
N%279 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %231, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %231
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %279, align 8, !tbaa !8
.double*8B

	full_text

double* %279
ucall8Bk
i
	full_text\
Z
X%280 = tail call double @llvm.fmuladd.f64(double %233, double 4.725000e-02, double %261)
,double8B

	full_text

double %233
,double8B

	full_text

double %261
tgetelementptr8Ba
_
	full_textR
P
N%281 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %231, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %280, double* %281, align 8, !tbaa !8
,double8B

	full_text

double %280
.double*8B

	full_text

double* %281
tgetelementptr8Ba
_
	full_textR
P
N%282 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %231, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %264, double* %282, align 8, !tbaa !8
,double8B

	full_text

double %264
.double*8B

	full_text

double* %282
vcall8Bl
j
	full_text]
[
Y%283 = tail call double @llvm.fmuladd.f64(double %256, double -4.725000e-02, double %269)
,double8B

	full_text

double %256
,double8B

	full_text

double %269
tgetelementptr8Ba
_
	full_textR
P
N%284 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %231, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %231
Pstore8BE
C
	full_text6
4
2store double %283, double* %284, align 8, !tbaa !8
,double8B

	full_text

double %283
.double*8B

	full_text

double* %284
tgetelementptr8Ba
_
	full_textR
P
N%285 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %231, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %231
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %285, align 8, !tbaa !8
.double*8B

	full_text

double* %285
:icmp8B0
.
	full_text!

%286 = icmp eq i64 %238, %229
&i648B

	full_text


i64 %238
&i648B

	full_text


i64 %229
=br8B5
3
	full_text&
$
"br i1 %286, label %287, label %230
$i18B

	full_text
	
i1 %286
Kphi8BB
@
	full_text3
1
/%288 = phi double [ %129, %27 ], [ %236, %230 ]
,double8B

	full_text

double %129
,double8B

	full_text

double %236
Kphi8BB
@
	full_text3
1
/%289 = phi double [ %181, %27 ], [ %243, %230 ]
,double8B

	full_text

double %181
,double8B

	full_text

double %243
Kphi8BB
@
	full_text3
1
/%290 = phi double [ %139, %27 ], [ %234, %230 ]
,double8B

	full_text

double %139
,double8B

	full_text

double %234
Kphi8BB
@
	full_text3
1
/%291 = phi double [ %191, %27 ], [ %253, %230 ]
,double8B

	full_text

double %191
,double8B

	full_text

double %253
Kphi8BB
@
	full_text3
1
/%292 = phi double [ %144, %27 ], [ %232, %230 ]
,double8B

	full_text

double %144
,double8B

	full_text

double %232
Kphi8BB
@
	full_text3
1
/%293 = phi double [ %196, %27 ], [ %256, %230 ]
,double8B

	full_text

double %196
,double8B

	full_text

double %256
7sext8B-
+
	full_text

%294 = sext i32 %11 to i64
ègetelementptr8B|
z
	full_textm
k
i%295 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %74, i64 %294, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %74
&i648B

	full_text


i64 %294
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%296 = load double, double* %295, align 8, !tbaa !8
.double*8B

	full_text

double* %295
Bfmul8B8
6
	full_text)
'
%%297 = fmul double %296, 1.000000e-01
,double8B

	full_text

double %296
ègetelementptr8B|
z
	full_textm
k
i%298 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %294, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
&i648B

	full_text


i64 %294
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%299 = load double, double* %298, align 8, !tbaa !8
.double*8B

	full_text

double* %298
agetelementptr8BN
L
	full_text?
=
;%300 = getelementptr inbounds double, double* %36, i64 %294
-double*8B

	full_text

double* %36
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %299, double* %300, align 8, !tbaa !8
,double8B

	full_text

double %299
.double*8B

	full_text

double* %300
Écall8By
w
	full_textj
h
f%301 = tail call double @llvm.fmuladd.f64(double %297, double 0x3FF5555555555555, double 1.000000e+00)
,double8B

	full_text

double %297
Écall8By
w
	full_textj
h
f%302 = tail call double @llvm.fmuladd.f64(double %297, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
,double8B

	full_text

double %297
>fcmp8B4
2
	full_text%
#
!%303 = fcmp ogt double %301, %302
,double8B

	full_text

double %301
,double8B

	full_text

double %302
Nselect8BB
@
	full_text3
1
/%304 = select i1 %303, double %301, double %302
$i18B

	full_text
	
i1 %303
,double8B

	full_text

double %301
,double8B

	full_text

double %302
Bfadd8B8
6
	full_text)
'
%%305 = fadd double %297, 1.000000e+00
,double8B

	full_text

double %297
Ffcmp8B<
:
	full_text-
+
)%306 = fcmp ogt double %305, 1.000000e+00
,double8B

	full_text

double %305
Vselect8BJ
H
	full_text;
9
7%307 = select i1 %306, double %305, double 1.000000e+00
$i18B

	full_text
	
i1 %306
,double8B

	full_text

double %305
>fcmp8B4
2
	full_text%
#
!%308 = fcmp ogt double %304, %307
,double8B

	full_text

double %304
,double8B

	full_text

double %307
Nselect8BB
@
	full_text3
1
/%309 = select i1 %308, double %304, double %307
$i18B

	full_text
	
i1 %308
,double8B

	full_text

double %304
,double8B

	full_text

double %307
agetelementptr8BN
L
	full_text?
=
;%310 = getelementptr inbounds double, double* %76, i64 %294
-double*8B

	full_text

double* %76
&i648B

	full_text


i64 %294
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
ègetelementptr8B|
z
	full_textm
k
i%311 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %29, i64 %294, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %29
&i648B

	full_text


i64 %294
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%312 = load double, double* %311, align 8, !tbaa !8
.double*8B

	full_text

double* %311
8sext8B.
,
	full_text

%313 = sext i32 %227 to i64
&i328B

	full_text


i32 %227
tgetelementptr8Ba
_
	full_textR
P
N%314 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %313, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %313
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %314, align 8, !tbaa !8
.double*8B

	full_text

double* %314
Hfmul8B>
<
	full_text/
-
+%315 = fmul double %290, 0x4017D0624DD2F1AB
,double8B

	full_text

double %290
Cfsub8B9
7
	full_text*
(
&%316 = fsub double -0.000000e+00, %315
,double8B

	full_text

double %315
vcall8Bl
j
	full_text]
[
Y%317 = tail call double @llvm.fmuladd.f64(double %288, double -4.725000e-02, double %316)
,double8B

	full_text

double %288
,double8B

	full_text

double %316
Cfadd8B9
7
	full_text*
(
&%318 = fadd double %317, -1.500000e-03
,double8B

	full_text

double %317
tgetelementptr8Ba
_
	full_textR
P
N%319 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %313, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %318, double* %319, align 8, !tbaa !8
,double8B

	full_text

double %318
.double*8B

	full_text

double* %319
Écall8By
w
	full_textj
h
f%320 = tail call double @llvm.fmuladd.f64(double %291, double 0x4027D0624DD2F1AB, double 1.000000e+00)
,double8B

	full_text

double %291
Hfadd8B>
<
	full_text/
-
+%321 = fadd double %320, 0x3F626E978D4FDF3C
,double8B

	full_text

double %320
tgetelementptr8Ba
_
	full_textR
P
N%322 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %313, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %321, double* %322, align 8, !tbaa !8
,double8B

	full_text

double %321
.double*8B

	full_text

double* %322
Hfmul8B>
<
	full_text/
-
+%323 = fmul double %309, 0x4017D0624DD2F1AB
,double8B

	full_text

double %309
Cfsub8B9
7
	full_text*
(
&%324 = fsub double -0.000000e+00, %323
,double8B

	full_text

double %323
ucall8Bk
i
	full_text\
Z
X%325 = tail call double @llvm.fmuladd.f64(double %299, double 4.725000e-02, double %324)
,double8B

	full_text

double %299
,double8B

	full_text

double %324
Cfadd8B9
7
	full_text*
(
&%326 = fadd double %325, -1.500000e-03
,double8B

	full_text

double %325
tgetelementptr8Ba
_
	full_textR
P
N%327 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %313, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %326, double* %327, align 8, !tbaa !8
,double8B

	full_text

double %326
.double*8B

	full_text

double* %327
tgetelementptr8Ba
_
	full_textR
P
N%328 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %313, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %313
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %328, align 8, !tbaa !8
.double*8B

	full_text

double* %328
tgetelementptr8Ba
_
	full_textR
P
N%329 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %313, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %313
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %329, align 8, !tbaa !8
.double*8B

	full_text

double* %329
vcall8Bl
j
	full_text]
[
Y%330 = tail call double @llvm.fmuladd.f64(double %292, double -4.725000e-02, double %318)
,double8B

	full_text

double %292
,double8B

	full_text

double %318
tgetelementptr8Ba
_
	full_textR
P
N%331 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %313, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %330, double* %331, align 8, !tbaa !8
,double8B

	full_text

double %330
.double*8B

	full_text

double* %331
tgetelementptr8Ba
_
	full_textR
P
N%332 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %313, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %321, double* %332, align 8, !tbaa !8
,double8B

	full_text

double %321
.double*8B

	full_text

double* %332
ucall8Bk
i
	full_text\
Z
X%333 = tail call double @llvm.fmuladd.f64(double %312, double 4.725000e-02, double %326)
,double8B

	full_text

double %312
,double8B

	full_text

double %326
tgetelementptr8Ba
_
	full_textR
P
N%334 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %313, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %333, double* %334, align 8, !tbaa !8
,double8B

	full_text

double %333
.double*8B

	full_text

double* %334
tgetelementptr8Ba
_
	full_textR
P
N%335 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %313, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %313
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %335, align 8, !tbaa !8
.double*8B

	full_text

double* %335
tgetelementptr8Ba
_
	full_textR
P
N%336 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %313, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %313
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %336, align 8, !tbaa !8
.double*8B

	full_text

double* %336
ucall8Bk
i
	full_text\
Z
X%337 = tail call double @llvm.fmuladd.f64(double %292, double 4.725000e-02, double %318)
,double8B

	full_text

double %292
,double8B

	full_text

double %318
tgetelementptr8Ba
_
	full_textR
P
N%338 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %313, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %337, double* %338, align 8, !tbaa !8
,double8B

	full_text

double %337
.double*8B

	full_text

double* %338
tgetelementptr8Ba
_
	full_textR
P
N%339 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %313, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %321, double* %339, align 8, !tbaa !8
,double8B

	full_text

double %321
.double*8B

	full_text

double* %339
vcall8Bl
j
	full_text]
[
Y%340 = tail call double @llvm.fmuladd.f64(double %312, double -4.725000e-02, double %326)
,double8B

	full_text

double %312
,double8B

	full_text

double %326
tgetelementptr8Ba
_
	full_textR
P
N%341 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %313, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %313
Pstore8BE
C
	full_text6
4
2store double %340, double* %341, align 8, !tbaa !8
,double8B

	full_text

double %340
.double*8B

	full_text

double* %341
tgetelementptr8Ba
_
	full_textR
P
N%342 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %313, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %313
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %342, align 8, !tbaa !8
.double*8B

	full_text

double* %342
égetelementptr8B{
y
	full_textl
j
h%343 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %74, i64 %46, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %74
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%344 = load double, double* %343, align 8, !tbaa !8
.double*8B

	full_text

double* %343
Bfmul8B8
6
	full_text)
'
%%345 = fmul double %344, 1.000000e-01
,double8B

	full_text

double %344
égetelementptr8B{
y
	full_textl
j
h%346 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %28, i64 %46, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %28
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%347 = load double, double* %346, align 8, !tbaa !8
.double*8B

	full_text

double* %346
`getelementptr8BM
K
	full_text>
<
:%348 = getelementptr inbounds double, double* %36, i64 %46
-double*8B

	full_text

double* %36
%i648B

	full_text
	
i64 %46
Pstore8BE
C
	full_text6
4
2store double %347, double* %348, align 8, !tbaa !8
,double8B

	full_text

double %347
.double*8B

	full_text

double* %348
Écall8By
w
	full_textj
h
f%349 = tail call double @llvm.fmuladd.f64(double %345, double 0x3FF5555555555555, double 1.000000e+00)
,double8B

	full_text

double %345
Écall8By
w
	full_textj
h
f%350 = tail call double @llvm.fmuladd.f64(double %345, double 0x3FFF5C28F5C28F5B, double 1.000000e+00)
,double8B

	full_text

double %345
>fcmp8B4
2
	full_text%
#
!%351 = fcmp ogt double %349, %350
,double8B

	full_text

double %349
,double8B

	full_text

double %350
Nselect8BB
@
	full_text3
1
/%352 = select i1 %351, double %349, double %350
$i18B

	full_text
	
i1 %351
,double8B

	full_text

double %349
,double8B

	full_text

double %350
Bfadd8B8
6
	full_text)
'
%%353 = fadd double %345, 1.000000e+00
,double8B

	full_text

double %345
Ffcmp8B<
:
	full_text-
+
)%354 = fcmp ogt double %353, 1.000000e+00
,double8B

	full_text

double %353
Vselect8BJ
H
	full_text;
9
7%355 = select i1 %354, double %353, double 1.000000e+00
$i18B

	full_text
	
i1 %354
,double8B

	full_text

double %353
>fcmp8B4
2
	full_text%
#
!%356 = fcmp ogt double %352, %355
,double8B

	full_text

double %352
,double8B

	full_text

double %355
Nselect8BB
@
	full_text3
1
/%357 = select i1 %356, double %352, double %355
$i18B

	full_text
	
i1 %356
,double8B

	full_text

double %352
,double8B

	full_text

double %355
`getelementptr8BM
K
	full_text>
<
:%358 = getelementptr inbounds double, double* %76, i64 %46
-double*8B

	full_text

double* %76
%i648B

	full_text
	
i64 %46
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
égetelementptr8B{
y
	full_textl
j
h%359 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %29, i64 %46, i64 %78, i64 %80
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %29
%i648B

	full_text
	
i64 %46
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%360 = load double, double* %359, align 8, !tbaa !8
.double*8B

	full_text

double* %359
tgetelementptr8Ba
_
	full_textR
P
N%361 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %294, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %294
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %361, align 8, !tbaa !8
.double*8B

	full_text

double* %361
Hfmul8B>
<
	full_text/
-
+%362 = fmul double %291, 0x4017D0624DD2F1AB
,double8B

	full_text

double %291
Cfsub8B9
7
	full_text*
(
&%363 = fsub double -0.000000e+00, %362
,double8B

	full_text

double %362
vcall8Bl
j
	full_text]
[
Y%364 = tail call double @llvm.fmuladd.f64(double %289, double -4.725000e-02, double %363)
,double8B

	full_text

double %289
,double8B

	full_text

double %363
Cfadd8B9
7
	full_text*
(
&%365 = fadd double %364, -1.500000e-03
,double8B

	full_text

double %364
tgetelementptr8Ba
_
	full_textR
P
N%366 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %294, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %365, double* %366, align 8, !tbaa !8
,double8B

	full_text

double %365
.double*8B

	full_text

double* %366
Écall8By
w
	full_textj
h
f%367 = tail call double @llvm.fmuladd.f64(double %309, double 0x4027D0624DD2F1AB, double 1.000000e+00)
,double8B

	full_text

double %309
Bfadd8B8
6
	full_text)
'
%%368 = fadd double %367, 1.875000e-03
,double8B

	full_text

double %367
tgetelementptr8Ba
_
	full_textR
P
N%369 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %294, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %368, double* %369, align 8, !tbaa !8
,double8B

	full_text

double %368
.double*8B

	full_text

double* %369
Hfmul8B>
<
	full_text/
-
+%370 = fmul double %357, 0x4017D0624DD2F1AB
,double8B

	full_text

double %357
Cfsub8B9
7
	full_text*
(
&%371 = fsub double -0.000000e+00, %370
,double8B

	full_text

double %370
ucall8Bk
i
	full_text\
Z
X%372 = tail call double @llvm.fmuladd.f64(double %347, double 4.725000e-02, double %371)
,double8B

	full_text

double %347
,double8B

	full_text

double %371
tgetelementptr8Ba
_
	full_textR
P
N%373 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %294, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %372, double* %373, align 8, !tbaa !8
,double8B

	full_text

double %372
.double*8B

	full_text

double* %373
tgetelementptr8Ba
_
	full_textR
P
N%374 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %294, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %294
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %374, align 8, !tbaa !8
.double*8B

	full_text

double* %374
tgetelementptr8Ba
_
	full_textR
P
N%375 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %294, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %294
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %375, align 8, !tbaa !8
.double*8B

	full_text

double* %375
vcall8Bl
j
	full_text]
[
Y%376 = tail call double @llvm.fmuladd.f64(double %293, double -4.725000e-02, double %365)
,double8B

	full_text

double %293
,double8B

	full_text

double %365
tgetelementptr8Ba
_
	full_textR
P
N%377 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %294, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %376, double* %377, align 8, !tbaa !8
,double8B

	full_text

double %376
.double*8B

	full_text

double* %377
tgetelementptr8Ba
_
	full_textR
P
N%378 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %294, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %368, double* %378, align 8, !tbaa !8
,double8B

	full_text

double %368
.double*8B

	full_text

double* %378
ucall8Bk
i
	full_text\
Z
X%379 = tail call double @llvm.fmuladd.f64(double %360, double 4.725000e-02, double %372)
,double8B

	full_text

double %360
,double8B

	full_text

double %372
tgetelementptr8Ba
_
	full_textR
P
N%380 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %294, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %379, double* %380, align 8, !tbaa !8
,double8B

	full_text

double %379
.double*8B

	full_text

double* %380
tgetelementptr8Ba
_
	full_textR
P
N%381 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %294, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %294
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %381, align 8, !tbaa !8
.double*8B

	full_text

double* %381
tgetelementptr8Ba
_
	full_textR
P
N%382 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %294, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %294
Xstore8BM
K
	full_text>
<
:store double 3.750000e-04, double* %382, align 8, !tbaa !8
.double*8B

	full_text

double* %382
ucall8Bk
i
	full_text\
Z
X%383 = tail call double @llvm.fmuladd.f64(double %293, double 4.725000e-02, double %365)
,double8B

	full_text

double %293
,double8B

	full_text

double %365
tgetelementptr8Ba
_
	full_textR
P
N%384 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %294, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %383, double* %384, align 8, !tbaa !8
,double8B

	full_text

double %383
.double*8B

	full_text

double* %384
tgetelementptr8Ba
_
	full_textR
P
N%385 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %294, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %368, double* %385, align 8, !tbaa !8
,double8B

	full_text

double %368
.double*8B

	full_text

double* %385
vcall8Bl
j
	full_text]
[
Y%386 = tail call double @llvm.fmuladd.f64(double %360, double -4.725000e-02, double %372)
,double8B

	full_text

double %360
,double8B

	full_text

double %372
tgetelementptr8Ba
_
	full_textR
P
N%387 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %294, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %294
Pstore8BE
C
	full_text6
4
2store double %386, double* %387, align 8, !tbaa !8
,double8B

	full_text

double %386
.double*8B

	full_text

double* %387
tgetelementptr8Ba
_
	full_textR
P
N%388 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %294, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %294
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %388, align 8, !tbaa !8
.double*8B

	full_text

double* %388
Oload8BE
C
	full_text6
4
2%389 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
Oload8BE
C
	full_text6
4
2%390 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
Pload8BF
D
	full_text7
5
3%391 = load double, double* %149, align 8, !tbaa !8
.double*8B

	full_text

double* %149
Pload8BF
D
	full_text7
5
3%392 = load double, double* %152, align 8, !tbaa !8
.double*8B

	full_text

double* %152
ögetelementptr8BÜ
É
	full_textv
t
r%393 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 0, i64 %78, i64 %80
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Ibitcast8B<
:
	full_text-
+
)%394 = bitcast [5 x double]* %393 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %393
Jload8B@
>
	full_text1
/
-%395 = load i64, i64* %394, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %394
Hbitcast8B;
9
	full_text,
*
(%396 = bitcast [5 x double]* %15 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Kstore8B@
>
	full_text1
/
-store i64 %395, i64* %396, align 16, !tbaa !8
&i648B

	full_text


i64 %395
(i64*8B

	full_text

	i64* %396
°getelementptr8Bç
ä
	full_text}
{
y%397 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 0, i64 %78, i64 %80, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Cbitcast8B6
4
	full_text'
%
#%398 = bitcast double* %397 to i64*
.double*8B

	full_text

double* %397
Jload8B@
>
	full_text1
/
-%399 = load i64, i64* %398, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %398
qgetelementptr8B^
\
	full_textO
M
K%400 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%401 = bitcast double* %400 to i64*
.double*8B

	full_text

double* %400
Jstore8B?
=
	full_text0
.
,store i64 %399, i64* %401, align 8, !tbaa !8
&i648B

	full_text


i64 %399
(i64*8B

	full_text

	i64* %401
°getelementptr8Bç
ä
	full_text}
{
y%402 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 0, i64 %78, i64 %80, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Cbitcast8B6
4
	full_text'
%
#%403 = bitcast double* %402 to i64*
.double*8B

	full_text

double* %402
Jload8B@
>
	full_text1
/
-%404 = load i64, i64* %403, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %403
qgetelementptr8B^
\
	full_textO
M
K%405 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%406 = bitcast double* %405 to i64*
.double*8B

	full_text

double* %405
Kstore8B@
>
	full_text1
/
-store i64 %404, i64* %406, align 16, !tbaa !8
&i648B

	full_text


i64 %404
(i64*8B

	full_text

	i64* %406
agetelementptr8BN
L
	full_text?
=
;%407 = getelementptr inbounds double, double* %3, i64 21125
Zbitcast8BM
K
	full_text>
<
:%408 = bitcast double* %407 to [65 x [65 x [5 x double]]]*
.double*8B

	full_text

double* %407
õgetelementptr8Bá
Ñ
	full_textw
u
s%409 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %408, i64 0, i64 %78, i64 %80
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %408
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Ibitcast8B<
:
	full_text-
+
)%410 = bitcast [5 x double]* %409 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %409
Jload8B@
>
	full_text1
/
-%411 = load i64, i64* %410, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %410
¢getelementptr8Bé
ã
	full_text~
|
z%412 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %408, i64 0, i64 %78, i64 %80, i64 1
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %408
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Cbitcast8B6
4
	full_text'
%
#%413 = bitcast double* %412 to i64*
.double*8B

	full_text

double* %412
Jload8B@
>
	full_text1
/
-%414 = load i64, i64* %413, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %413
¢getelementptr8Bé
ã
	full_text~
|
z%415 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %408, i64 0, i64 %78, i64 %80, i64 2
V[65 x [65 x [5 x double]]]*8B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %408
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Cbitcast8B6
4
	full_text'
%
#%416 = bitcast double* %415 to i64*
.double*8B

	full_text

double* %415
Jload8B@
>
	full_text1
/
-%417 = load i64, i64* %416, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %416
6add8B-
+
	full_text

%418 = add nsw i32 %12, -3
7icmp8B-
+
	full_text

%419 = icmp slt i32 %12, 3
=br8B5
3
	full_text&
$
"br i1 %419, label %420, label %426
$i18B

	full_text
	
i1 %419
6add8B-
+
	full_text

%421 = add nsw i32 %12, -2
qgetelementptr8B^
\
	full_textO
M
K%422 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
qgetelementptr8B^
\
	full_textO
M
K%423 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
qgetelementptr8B^
\
	full_textO
M
K%424 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
qgetelementptr8B^
\
	full_textO
M
K%425 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
(br8B 

	full_text

br label %500
qgetelementptr8B^
\
	full_textO
M
K%427 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
qgetelementptr8B^
\
	full_textO
M
K%428 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
2add8B)
'
	full_text

%429 = add i32 %12, -2
qgetelementptr8B^
\
	full_textO
M
K%430 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
qgetelementptr8B^
\
	full_textO
M
K%431 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
8zext8B.
,
	full_text

%432 = zext i32 %429 to i64
&i328B

	full_text


i32 %429
(br8B 

	full_text

br label %433
Iphi8B@
>
	full_text1
/
-%434 = phi i64 [ %498, %433 ], [ %417, %426 ]
&i648B

	full_text


i64 %498
&i648B

	full_text


i64 %417
Iphi8B@
>
	full_text1
/
-%435 = phi i64 [ %497, %433 ], [ %414, %426 ]
&i648B

	full_text


i64 %497
&i648B

	full_text


i64 %414
Iphi8B@
>
	full_text1
/
-%436 = phi i64 [ %496, %433 ], [ %411, %426 ]
&i648B

	full_text


i64 %496
&i648B

	full_text


i64 %411
Iphi8B@
>
	full_text1
/
-%437 = phi i64 [ %495, %433 ], [ %404, %426 ]
&i648B

	full_text


i64 %495
&i648B

	full_text


i64 %404
Iphi8B@
>
	full_text1
/
-%438 = phi i64 [ %494, %433 ], [ %399, %426 ]
&i648B

	full_text


i64 %494
&i648B

	full_text


i64 %399
Iphi8B@
>
	full_text1
/
-%439 = phi i64 [ %493, %433 ], [ %395, %426 ]
&i648B

	full_text


i64 %493
&i648B

	full_text


i64 %395
Fphi8B=
;
	full_text.
,
*%440 = phi i64 [ %445, %433 ], [ 0, %426 ]
&i648B

	full_text


i64 %445
Lphi8BC
A
	full_text4
2
0%441 = phi double [ %490, %433 ], [ %390, %426 ]
,double8B

	full_text

double %490
,double8B

	full_text

double %390
Lphi8BC
A
	full_text4
2
0%442 = phi double [ %464, %433 ], [ %389, %426 ]
,double8B

	full_text

double %464
,double8B

	full_text

double %389
Lphi8BC
A
	full_text4
2
0%443 = phi double [ %491, %433 ], [ %392, %426 ]
,double8B

	full_text

double %491
,double8B

	full_text

double %392
Lphi8BC
A
	full_text4
2
0%444 = phi double [ %486, %433 ], [ %391, %426 ]
,double8B

	full_text

double %486
,double8B

	full_text

double %391
:add8B1
/
	full_text"
 
%445 = add nuw nsw i64 %440, 1
&i648B

	full_text


i64 %440
tgetelementptr8Ba
_
	full_textR
P
N%446 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %440, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %440
Pload8BF
D
	full_text7
5
3%447 = load double, double* %446, align 8, !tbaa !8
.double*8B

	full_text

double* %446
Bfdiv8B8
6
	full_text)
'
%%448 = fdiv double 1.000000e+00, %442
,double8B

	full_text

double %442
:fmul8B0
.
	full_text!

%449 = fmul double %448, %441
,double8B

	full_text

double %448
,double8B

	full_text

double %441
tgetelementptr8Ba
_
	full_textR
P
N%450 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %440, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %440
Pstore8BE
C
	full_text6
4
2store double %449, double* %450, align 8, !tbaa !8
,double8B

	full_text

double %449
.double*8B

	full_text

double* %450
:fmul8B0
.
	full_text!

%451 = fmul double %448, %447
,double8B

	full_text

double %448
,double8B

	full_text

double %447
Pstore8BE
C
	full_text6
4
2store double %451, double* %446, align 8, !tbaa !8
,double8B

	full_text

double %451
.double*8B

	full_text

double* %446
Abitcast8B4
2
	full_text%
#
!%452 = bitcast i64 %439 to double
&i648B

	full_text


i64 %439
:fmul8B0
.
	full_text!

%453 = fmul double %448, %452
,double8B

	full_text

double %448
,double8B

	full_text

double %452
•getelementptr8Bë
é
	full_textÄ
~
|%454 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %440, i64 %78, i64 %80, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %440
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %453, double* %454, align 8, !tbaa !8
,double8B

	full_text

double %453
.double*8B

	full_text

double* %454
Abitcast8B4
2
	full_text%
#
!%455 = bitcast i64 %438 to double
&i648B

	full_text


i64 %438
:fmul8B0
.
	full_text!

%456 = fmul double %448, %455
,double8B

	full_text

double %448
,double8B

	full_text

double %455
•getelementptr8Bë
é
	full_textÄ
~
|%457 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %440, i64 %78, i64 %80, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %440
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %456, double* %457, align 8, !tbaa !8
,double8B

	full_text

double %456
.double*8B

	full_text

double* %457
Abitcast8B4
2
	full_text%
#
!%458 = bitcast i64 %437 to double
&i648B

	full_text


i64 %437
:fmul8B0
.
	full_text!

%459 = fmul double %448, %458
,double8B

	full_text

double %448
,double8B

	full_text

double %458
•getelementptr8Bë
é
	full_textÄ
~
|%460 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %440, i64 %78, i64 %80, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %440
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %459, double* %460, align 8, !tbaa !8
,double8B

	full_text

double %459
.double*8B

	full_text

double* %460
tgetelementptr8Ba
_
	full_textR
P
N%461 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %445, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %445
Pload8BF
D
	full_text7
5
3%462 = load double, double* %461, align 8, !tbaa !8
.double*8B

	full_text

double* %461
Cfsub8B9
7
	full_text*
(
&%463 = fsub double -0.000000e+00, %444
,double8B

	full_text

double %444
mcall8Bc
a
	full_textT
R
P%464 = tail call double @llvm.fmuladd.f64(double %463, double %449, double %443)
,double8B

	full_text

double %463
,double8B

	full_text

double %449
,double8B

	full_text

double %443
tgetelementptr8Ba
_
	full_textR
P
N%465 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %445, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %445
Pstore8BE
C
	full_text6
4
2store double %464, double* %465, align 8, !tbaa !8
,double8B

	full_text

double %464
.double*8B

	full_text

double* %465
Abitcast8B4
2
	full_text%
#
!%466 = bitcast i64 %436 to double
&i648B

	full_text


i64 %436
mcall8Bc
a
	full_textT
R
P%467 = tail call double @llvm.fmuladd.f64(double %463, double %453, double %466)
,double8B

	full_text

double %463
,double8B

	full_text

double %453
,double8B

	full_text

double %466
Abitcast8B4
2
	full_text%
#
!%468 = bitcast i64 %435 to double
&i648B

	full_text


i64 %435
mcall8Bc
a
	full_textT
R
P%469 = tail call double @llvm.fmuladd.f64(double %463, double %456, double %468)
,double8B

	full_text

double %463
,double8B

	full_text

double %456
,double8B

	full_text

double %468
Abitcast8B4
2
	full_text%
#
!%470 = bitcast i64 %434 to double
&i648B

	full_text


i64 %434
mcall8Bc
a
	full_textT
R
P%471 = tail call double @llvm.fmuladd.f64(double %463, double %459, double %470)
,double8B

	full_text

double %463
,double8B

	full_text

double %459
,double8B

	full_text

double %470
:add8B1
/
	full_text"
 
%472 = add nuw nsw i64 %440, 2
&i648B

	full_text


i64 %440
tgetelementptr8Ba
_
	full_textR
P
N%473 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %472, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %472
Pload8BF
D
	full_text7
5
3%474 = load double, double* %473, align 8, !tbaa !8
.double*8B

	full_text

double* %473
tgetelementptr8Ba
_
	full_textR
P
N%475 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %472, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %472
Pload8BF
D
	full_text7
5
3%476 = load double, double* %475, align 8, !tbaa !8
.double*8B

	full_text

double* %475
tgetelementptr8Ba
_
	full_textR
P
N%477 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %472, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %472
Pload8BF
D
	full_text7
5
3%478 = load double, double* %477, align 8, !tbaa !8
.double*8B

	full_text

double* %477
•getelementptr8Bë
é
	full_textÄ
~
|%479 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %472, i64 %78, i64 %80, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %472
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%480 = load double, double* %479, align 8, !tbaa !8
.double*8B

	full_text

double* %479
•getelementptr8Bë
é
	full_textÄ
~
|%481 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %472, i64 %78, i64 %80, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %472
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%482 = load double, double* %481, align 8, !tbaa !8
.double*8B

	full_text

double* %481
•getelementptr8Bë
é
	full_textÄ
~
|%483 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %472, i64 %78, i64 %80, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %472
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%484 = load double, double* %483, align 8, !tbaa !8
.double*8B

	full_text

double* %483
Cfsub8B9
7
	full_text*
(
&%485 = fsub double -0.000000e+00, %474
,double8B

	full_text

double %474
mcall8Bc
a
	full_textT
R
P%486 = tail call double @llvm.fmuladd.f64(double %485, double %449, double %476)
,double8B

	full_text

double %485
,double8B

	full_text

double %449
,double8B

	full_text

double %476
Pstore8BE
C
	full_text6
4
2store double %486, double* %475, align 8, !tbaa !8
,double8B

	full_text

double %486
.double*8B

	full_text

double* %475
mcall8Bc
a
	full_textT
R
P%487 = tail call double @llvm.fmuladd.f64(double %485, double %453, double %480)
,double8B

	full_text

double %485
,double8B

	full_text

double %453
,double8B

	full_text

double %480
mcall8Bc
a
	full_textT
R
P%488 = tail call double @llvm.fmuladd.f64(double %485, double %456, double %482)
,double8B

	full_text

double %485
,double8B

	full_text

double %456
,double8B

	full_text

double %482
mcall8Bc
a
	full_textT
R
P%489 = tail call double @llvm.fmuladd.f64(double %485, double %459, double %484)
,double8B

	full_text

double %485
,double8B

	full_text

double %459
,double8B

	full_text

double %484
mcall8Bc
a
	full_textT
R
P%490 = tail call double @llvm.fmuladd.f64(double %463, double %451, double %462)
,double8B

	full_text

double %463
,double8B

	full_text

double %451
,double8B

	full_text

double %462
mcall8Bc
a
	full_textT
R
P%491 = tail call double @llvm.fmuladd.f64(double %485, double %451, double %478)
,double8B

	full_text

double %485
,double8B

	full_text

double %451
,double8B

	full_text

double %478
:icmp8B0
.
	full_text!

%492 = icmp eq i64 %445, %432
&i648B

	full_text


i64 %445
&i648B

	full_text


i64 %432
Abitcast8B4
2
	full_text%
#
!%493 = bitcast double %467 to i64
,double8B

	full_text

double %467
Abitcast8B4
2
	full_text%
#
!%494 = bitcast double %469 to i64
,double8B

	full_text

double %469
Abitcast8B4
2
	full_text%
#
!%495 = bitcast double %471 to i64
,double8B

	full_text

double %471
Abitcast8B4
2
	full_text%
#
!%496 = bitcast double %487 to i64
,double8B

	full_text

double %487
Abitcast8B4
2
	full_text%
#
!%497 = bitcast double %488 to i64
,double8B

	full_text

double %488
Abitcast8B4
2
	full_text%
#
!%498 = bitcast double %489 to i64
,double8B

	full_text

double %489
=br8B5
3
	full_text&
$
"br i1 %492, label %499, label %433
$i18B

	full_text
	
i1 %492
Qstore8BF
D
	full_text7
5
3store double %453, double* %430, align 16, !tbaa !8
,double8B

	full_text

double %453
.double*8B

	full_text

double* %430
Pstore8BE
C
	full_text6
4
2store double %456, double* %427, align 8, !tbaa !8
,double8B

	full_text

double %456
.double*8B

	full_text

double* %427
Qstore8BF
D
	full_text7
5
3store double %459, double* %428, align 16, !tbaa !8
,double8B

	full_text

double %459
.double*8B

	full_text

double* %428
Qstore8BF
D
	full_text7
5
3store double %467, double* %431, align 16, !tbaa !8
,double8B

	full_text

double %467
.double*8B

	full_text

double* %431
Pstore8BE
C
	full_text6
4
2store double %469, double* %400, align 8, !tbaa !8
,double8B

	full_text

double %469
.double*8B

	full_text

double* %400
Qstore8BF
D
	full_text7
5
3store double %471, double* %405, align 16, !tbaa !8
,double8B

	full_text

double %471
.double*8B

	full_text

double* %405
(br8B 

	full_text

br label %500
Mphi8	BD
B
	full_text5
3
1%501 = phi double* [ %425, %420 ], [ %431, %499 ]
.double*8	B

	full_text

double* %425
.double*8	B

	full_text

double* %431
Mphi8	BD
B
	full_text5
3
1%502 = phi double* [ %424, %420 ], [ %430, %499 ]
.double*8	B

	full_text

double* %424
.double*8	B

	full_text

double* %430
Mphi8	BD
B
	full_text5
3
1%503 = phi double* [ %423, %420 ], [ %428, %499 ]
.double*8	B

	full_text

double* %423
.double*8	B

	full_text

double* %428
Mphi8	BD
B
	full_text5
3
1%504 = phi double* [ %422, %420 ], [ %427, %499 ]
.double*8	B

	full_text

double* %422
.double*8	B

	full_text

double* %427
Iphi8	B@
>
	full_text1
/
-%505 = phi i32 [ %421, %420 ], [ %429, %499 ]
&i328	B

	full_text


i32 %421
&i328	B

	full_text


i32 %429
Iphi8	B@
>
	full_text1
/
-%506 = phi i64 [ %417, %420 ], [ %498, %499 ]
&i648	B

	full_text


i64 %417
&i648	B

	full_text


i64 %498
Iphi8	B@
>
	full_text1
/
-%507 = phi i64 [ %414, %420 ], [ %497, %499 ]
&i648	B

	full_text


i64 %414
&i648	B

	full_text


i64 %497
Iphi8	B@
>
	full_text1
/
-%508 = phi i64 [ %411, %420 ], [ %496, %499 ]
&i648	B

	full_text


i64 %411
&i648	B

	full_text


i64 %496
Iphi8	B@
>
	full_text1
/
-%509 = phi i64 [ %404, %420 ], [ %495, %499 ]
&i648	B

	full_text


i64 %404
&i648	B

	full_text


i64 %495
Iphi8	B@
>
	full_text1
/
-%510 = phi i64 [ %399, %420 ], [ %494, %499 ]
&i648	B

	full_text


i64 %399
&i648	B

	full_text


i64 %494
Iphi8	B@
>
	full_text1
/
-%511 = phi i64 [ %395, %420 ], [ %493, %499 ]
&i648	B

	full_text


i64 %395
&i648	B

	full_text


i64 %493
Lphi8	BC
A
	full_text4
2
0%512 = phi double [ %391, %420 ], [ %486, %499 ]
,double8	B

	full_text

double %391
,double8	B

	full_text

double %486
Lphi8	BC
A
	full_text4
2
0%513 = phi double [ %392, %420 ], [ %491, %499 ]
,double8	B

	full_text

double %392
,double8	B

	full_text

double %491
Lphi8	BC
A
	full_text4
2
0%514 = phi double [ %389, %420 ], [ %464, %499 ]
,double8	B

	full_text

double %389
,double8	B

	full_text

double %464
Lphi8	BC
A
	full_text4
2
0%515 = phi double [ %390, %420 ], [ %490, %499 ]
,double8	B

	full_text

double %390
,double8	B

	full_text

double %490
Hbitcast8	B;
9
	full_text,
*
(%516 = bitcast [5 x double]* %14 to i64*
9[5 x double]*8	B$
"
	full_text

[5 x double]* %14
Cbitcast8	B6
4
	full_text'
%
#%517 = bitcast double* %504 to i64*
.double*8	B

	full_text

double* %504
Cbitcast8	B6
4
	full_text'
%
#%518 = bitcast double* %503 to i64*
.double*8	B

	full_text

double* %503
8sext8	B.
,
	full_text

%519 = sext i32 %505 to i64
&i328	B

	full_text


i32 %505
tgetelementptr8	Ba
_
	full_textR
P
N%520 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %519, i64 4
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %519
Pload8	BF
D
	full_text7
5
3%521 = load double, double* %520, align 8, !tbaa !8
.double*8	B

	full_text

double* %520
Kstore8	B@
>
	full_text1
/
-store i64 %511, i64* %516, align 16, !tbaa !8
&i648	B

	full_text


i64 %511
(i64*8	B

	full_text

	i64* %516
Jstore8	B?
=
	full_text0
.
,store i64 %510, i64* %517, align 8, !tbaa !8
&i648	B

	full_text


i64 %510
(i64*8	B

	full_text

	i64* %517
Kstore8	B@
>
	full_text1
/
-store i64 %509, i64* %518, align 16, !tbaa !8
&i648	B

	full_text


i64 %509
(i64*8	B

	full_text

	i64* %518
Bfdiv8	B8
6
	full_text)
'
%%522 = fdiv double 1.000000e+00, %514
,double8	B

	full_text

double %514
:fmul8	B0
.
	full_text!

%523 = fmul double %522, %515
,double8	B

	full_text

double %522
,double8	B

	full_text

double %515
tgetelementptr8	Ba
_
	full_textR
P
N%524 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %519, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %519
Pstore8	BE
C
	full_text6
4
2store double %523, double* %524, align 8, !tbaa !8
,double8	B

	full_text

double %523
.double*8	B

	full_text

double* %524
:fmul8	B0
.
	full_text!

%525 = fmul double %522, %521
,double8	B

	full_text

double %522
,double8	B

	full_text

double %521
Pstore8	BE
C
	full_text6
4
2store double %525, double* %520, align 8, !tbaa !8
,double8	B

	full_text

double %525
.double*8	B

	full_text

double* %520
Abitcast8	B4
2
	full_text%
#
!%526 = bitcast i64 %511 to double
&i648	B

	full_text


i64 %511
:fmul8	B0
.
	full_text!

%527 = fmul double %522, %526
,double8	B

	full_text

double %522
,double8	B

	full_text

double %526
Pstore8	BE
C
	full_text6
4
2store double %527, double* %502, align 8, !tbaa !8
,double8	B

	full_text

double %527
.double*8	B

	full_text

double* %502
•getelementptr8	Bë
é
	full_textÄ
~
|%528 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %519, i64 %78, i64 %80, i64 0
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648	B

	full_text


i64 %519
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Pstore8	BE
C
	full_text6
4
2store double %527, double* %528, align 8, !tbaa !8
,double8	B

	full_text

double %527
.double*8	B

	full_text

double* %528
Abitcast8	B4
2
	full_text%
#
!%529 = bitcast i64 %510 to double
&i648	B

	full_text


i64 %510
:fmul8	B0
.
	full_text!

%530 = fmul double %522, %529
,double8	B

	full_text

double %522
,double8	B

	full_text

double %529
Pstore8	BE
C
	full_text6
4
2store double %530, double* %504, align 8, !tbaa !8
,double8	B

	full_text

double %530
.double*8	B

	full_text

double* %504
•getelementptr8	Bë
é
	full_textÄ
~
|%531 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %519, i64 %78, i64 %80, i64 1
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648	B

	full_text


i64 %519
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Pstore8	BE
C
	full_text6
4
2store double %530, double* %531, align 8, !tbaa !8
,double8	B

	full_text

double %530
.double*8	B

	full_text

double* %531
Abitcast8	B4
2
	full_text%
#
!%532 = bitcast i64 %509 to double
&i648	B

	full_text


i64 %509
:fmul8	B0
.
	full_text!

%533 = fmul double %522, %532
,double8	B

	full_text

double %522
,double8	B

	full_text

double %532
Pstore8	BE
C
	full_text6
4
2store double %533, double* %503, align 8, !tbaa !8
,double8	B

	full_text

double %533
.double*8	B

	full_text

double* %503
•getelementptr8	Bë
é
	full_textÄ
~
|%534 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %519, i64 %78, i64 %80, i64 2
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648	B

	full_text


i64 %519
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Pstore8	BE
C
	full_text6
4
2store double %533, double* %534, align 8, !tbaa !8
,double8	B

	full_text

double %533
.double*8	B

	full_text

double* %534
6add8	B-
+
	full_text

%535 = add nsw i32 %12, -1
8sext8	B.
,
	full_text

%536 = sext i32 %535 to i64
&i328	B

	full_text


i32 %535
tgetelementptr8	Ba
_
	full_textR
P
N%537 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %536, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %536
Pload8	BF
D
	full_text7
5
3%538 = load double, double* %537, align 8, !tbaa !8
.double*8	B

	full_text

double* %537
Kstore8	B@
>
	full_text1
/
-store i64 %508, i64* %396, align 16, !tbaa !8
&i648	B

	full_text


i64 %508
(i64*8	B

	full_text

	i64* %396
Cfsub8	B9
7
	full_text*
(
&%539 = fsub double -0.000000e+00, %512
,double8	B

	full_text

double %512
mcall8	Bc
a
	full_textT
R
P%540 = tail call double @llvm.fmuladd.f64(double %539, double %523, double %513)
,double8	B

	full_text

double %539
,double8	B

	full_text

double %523
,double8	B

	full_text

double %513
tgetelementptr8	Ba
_
	full_textR
P
N%541 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %536, i64 2
9[5 x double]*8	B$
"
	full_text

[5 x double]* %40
&i648	B

	full_text


i64 %536
Pstore8	BE
C
	full_text6
4
2store double %540, double* %541, align 8, !tbaa !8
,double8	B

	full_text

double %540
.double*8	B

	full_text

double* %541
mcall8	Bc
a
	full_textT
R
P%542 = tail call double @llvm.fmuladd.f64(double %539, double %525, double %538)
,double8	B

	full_text

double %539
,double8	B

	full_text

double %525
,double8	B

	full_text

double %538
Pstore8	BE
C
	full_text6
4
2store double %542, double* %537, align 8, !tbaa !8
,double8	B

	full_text

double %542
.double*8	B

	full_text

double* %537
Abitcast8	B4
2
	full_text%
#
!%543 = bitcast i64 %508 to double
&i648	B

	full_text


i64 %508
mcall8	Bc
a
	full_textT
R
P%544 = tail call double @llvm.fmuladd.f64(double %539, double %527, double %543)
,double8	B

	full_text

double %539
,double8	B

	full_text

double %527
,double8	B

	full_text

double %543
Pstore8	BE
C
	full_text6
4
2store double %544, double* %501, align 8, !tbaa !8
,double8	B

	full_text

double %544
.double*8	B

	full_text

double* %501
Abitcast8	B4
2
	full_text%
#
!%545 = bitcast i64 %507 to double
&i648	B

	full_text


i64 %507
mcall8	Bc
a
	full_textT
R
P%546 = tail call double @llvm.fmuladd.f64(double %539, double %530, double %545)
,double8	B

	full_text

double %539
,double8	B

	full_text

double %530
,double8	B

	full_text

double %545
Pstore8	BE
C
	full_text6
4
2store double %546, double* %400, align 8, !tbaa !8
,double8	B

	full_text

double %546
.double*8	B

	full_text

double* %400
Abitcast8	B4
2
	full_text%
#
!%547 = bitcast i64 %506 to double
&i648	B

	full_text


i64 %506
mcall8	Bc
a
	full_textT
R
P%548 = tail call double @llvm.fmuladd.f64(double %539, double %533, double %547)
,double8	B

	full_text

double %539
,double8	B

	full_text

double %533
,double8	B

	full_text

double %547
Qstore8	BF
D
	full_text7
5
3store double %548, double* %405, align 16, !tbaa !8
,double8	B

	full_text

double %548
.double*8	B

	full_text

double* %405
Bfdiv8	B8
6
	full_text)
'
%%549 = fdiv double 1.000000e+00, %540
,double8	B

	full_text

double %540
:fmul8	B0
.
	full_text!

%550 = fmul double %549, %544
,double8	B

	full_text

double %549
,double8	B

	full_text

double %544
•getelementptr8	Bë
é
	full_textÄ
~
|%551 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %536, i64 %78, i64 %80, i64 0
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648	B

	full_text


i64 %536
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Pstore8	BE
C
	full_text6
4
2store double %550, double* %551, align 8, !tbaa !8
,double8	B

	full_text

double %550
.double*8	B

	full_text

double* %551
:fmul8	B0
.
	full_text!

%552 = fmul double %549, %546
,double8	B

	full_text

double %549
,double8	B

	full_text

double %546
•getelementptr8	Bë
é
	full_textÄ
~
|%553 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %536, i64 %78, i64 %80, i64 1
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648	B

	full_text


i64 %536
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Pstore8	BE
C
	full_text6
4
2store double %552, double* %553, align 8, !tbaa !8
,double8	B

	full_text

double %552
.double*8	B

	full_text

double* %553
:fmul8	B0
.
	full_text!

%554 = fmul double %549, %548
,double8	B

	full_text

double %549
,double8	B

	full_text

double %548
•getelementptr8	Bë
é
	full_textÄ
~
|%555 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %536, i64 %78, i64 %80, i64 2
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648	B

	full_text


i64 %536
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
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
Oload8	BE
C
	full_text6
4
2%556 = load double, double* %57, align 8, !tbaa !8
-double*8	B

	full_text

double* %57
Oload8	BE
C
	full_text6
4
2%557 = load double, double* %63, align 8, !tbaa !8
-double*8	B

	full_text

double* %63
Pload8	BF
D
	full_text7
5
3%558 = load double, double* %161, align 8, !tbaa !8
.double*8	B

	full_text

double* %161
Pload8	BF
D
	full_text7
5
3%559 = load double, double* %162, align 8, !tbaa !8
.double*8	B

	full_text

double* %162
Oload8	BE
C
	full_text6
4
2%560 = load double, double* %58, align 8, !tbaa !8
-double*8	B

	full_text

double* %58
Oload8	BE
C
	full_text6
4
2%561 = load double, double* %64, align 8, !tbaa !8
-double*8	B

	full_text

double* %64
Pload8	BF
D
	full_text7
5
3%562 = load double, double* %168, align 8, !tbaa !8
.double*8	B

	full_text

double* %168
Pload8	BF
D
	full_text7
5
3%563 = load double, double* %169, align 8, !tbaa !8
.double*8	B

	full_text

double* %169
°getelementptr8	Bç
ä
	full_text}
{
y%564 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 0, i64 %78, i64 %80, i64 3
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Cbitcast8	B6
4
	full_text'
%
#%565 = bitcast double* %564 to i64*
.double*8	B

	full_text

double* %564
Jload8	B@
>
	full_text1
/
-%566 = load i64, i64* %565, align 8, !tbaa !8
(i64*8	B

	full_text

	i64* %565
qgetelementptr8	B^
\
	full_textO
M
K%567 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %15
Cbitcast8	B6
4
	full_text'
%
#%568 = bitcast double* %567 to i64*
.double*8	B

	full_text

double* %567
Jstore8	B?
=
	full_text0
.
,store i64 %566, i64* %568, align 8, !tbaa !8
&i648	B

	full_text


i64 %566
(i64*8	B

	full_text

	i64* %568
°getelementptr8	Bç
ä
	full_text}
{
y%569 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 0, i64 %78, i64 %80, i64 4
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Cbitcast8	B6
4
	full_text'
%
#%570 = bitcast double* %569 to i64*
.double*8	B

	full_text

double* %569
Jload8	B@
>
	full_text1
/
-%571 = load i64, i64* %570, align 8, !tbaa !8
(i64*8	B

	full_text

	i64* %570
qgetelementptr8	B^
\
	full_textO
M
K%572 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 4
9[5 x double]*8	B$
"
	full_text

[5 x double]* %15
Cbitcast8	B6
4
	full_text'
%
#%573 = bitcast double* %572 to i64*
.double*8	B

	full_text

double* %572
Kstore8	B@
>
	full_text1
/
-store i64 %571, i64* %573, align 16, !tbaa !8
&i648	B

	full_text


i64 %571
(i64*8	B

	full_text

	i64* %573
¢getelementptr8	Bé
ã
	full_text~
|
z%574 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %408, i64 0, i64 %78, i64 %80, i64 3
V[65 x [65 x [5 x double]]]*8	B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %408
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Cbitcast8	B6
4
	full_text'
%
#%575 = bitcast double* %574 to i64*
.double*8	B

	full_text

double* %574
Jload8	B@
>
	full_text1
/
-%576 = load i64, i64* %575, align 8, !tbaa !8
(i64*8	B

	full_text

	i64* %575
¢getelementptr8	Bé
ã
	full_text~
|
z%577 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %408, i64 0, i64 %78, i64 %80, i64 4
V[65 x [65 x [5 x double]]]*8	B3
1
	full_text$
"
 [65 x [65 x [5 x double]]]* %408
%i648	B

	full_text
	
i64 %78
%i648	B

	full_text
	
i64 %80
Cbitcast8	B6
4
	full_text'
%
#%578 = bitcast double* %577 to i64*
.double*8	B

	full_text

double* %577
Jload8	B@
>
	full_text1
/
-%579 = load i64, i64* %578, align 8, !tbaa !8
(i64*8	B

	full_text

	i64* %578
qgetelementptr8	B^
\
	full_textO
M
K%580 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %14
Cbitcast8	B6
4
	full_text'
%
#%581 = bitcast double* %580 to i64*
.double*8	B

	full_text

double* %580
qgetelementptr8	B^
\
	full_textO
M
K%582 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 4
9[5 x double]*8	B$
"
	full_text

[5 x double]* %14
Cbitcast8	B6
4
	full_text'
%
#%583 = bitcast double* %582 to i64*
.double*8	B

	full_text

double* %582
=br8	B5
3
	full_text&
$
"br i1 %419, label %666, label %584
$i18	B

	full_text
	
i1 %419
8zext8
B.
,
	full_text

%585 = zext i32 %505 to i64
&i328
B

	full_text


i32 %505
(br8
B 

	full_text

br label %586
Iphi8B@
>
	full_text1
/
-%587 = phi i64 [ %664, %586 ], [ %579, %584 ]
&i648B

	full_text


i64 %664
&i648B

	full_text


i64 %579
Iphi8B@
>
	full_text1
/
-%588 = phi i64 [ %663, %586 ], [ %571, %584 ]
&i648B

	full_text


i64 %663
&i648B

	full_text


i64 %571
Iphi8B@
>
	full_text1
/
-%589 = phi i64 [ %662, %586 ], [ %576, %584 ]
&i648B

	full_text


i64 %662
&i648B

	full_text


i64 %576
Iphi8B@
>
	full_text1
/
-%590 = phi i64 [ %661, %586 ], [ %566, %584 ]
&i648B

	full_text


i64 %661
&i648B

	full_text


i64 %566
Fphi8B=
;
	full_text.
,
*%591 = phi i64 [ %600, %586 ], [ 0, %584 ]
&i648B

	full_text


i64 %600
Lphi8BC
A
	full_text4
2
0%592 = phi double [ %657, %586 ], [ %562, %584 ]
,double8B

	full_text

double %657
,double8B

	full_text

double %562
Lphi8BC
A
	full_text4
2
0%593 = phi double [ %658, %586 ], [ %563, %584 ]
,double8B

	full_text

double %658
,double8B

	full_text

double %563
Lphi8BC
A
	full_text4
2
0%594 = phi double [ %643, %586 ], [ %560, %584 ]
,double8B

	full_text

double %643
,double8B

	full_text

double %560
Lphi8BC
A
	full_text4
2
0%595 = phi double [ %645, %586 ], [ %561, %584 ]
,double8B

	full_text

double %645
,double8B

	full_text

double %561
Lphi8BC
A
	full_text4
2
0%596 = phi double [ %628, %586 ], [ %558, %584 ]
,double8B

	full_text

double %628
,double8B

	full_text

double %558
Lphi8BC
A
	full_text4
2
0%597 = phi double [ %629, %586 ], [ %559, %584 ]
,double8B

	full_text

double %629
,double8B

	full_text

double %559
Lphi8BC
A
	full_text4
2
0%598 = phi double [ %614, %586 ], [ %556, %584 ]
,double8B

	full_text

double %614
,double8B

	full_text

double %556
Lphi8BC
A
	full_text4
2
0%599 = phi double [ %616, %586 ], [ %557, %584 ]
,double8B

	full_text

double %616
,double8B

	full_text

double %557
:add8B1
/
	full_text"
 
%600 = add nuw nsw i64 %591, 1
&i648B

	full_text


i64 %591
:add8B1
/
	full_text"
 
%601 = add nuw nsw i64 %591, 2
&i648B

	full_text


i64 %591
tgetelementptr8Ba
_
	full_textR
P
N%602 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %591, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %591
Pload8BF
D
	full_text7
5
3%603 = load double, double* %602, align 8, !tbaa !8
.double*8B

	full_text

double* %602
Bfdiv8B8
6
	full_text)
'
%%604 = fdiv double 1.000000e+00, %598
,double8B

	full_text

double %598
:fmul8B0
.
	full_text!

%605 = fmul double %599, %604
,double8B

	full_text

double %599
,double8B

	full_text

double %604
tgetelementptr8Ba
_
	full_textR
P
N%606 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %591, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %591
Pstore8BE
C
	full_text6
4
2store double %605, double* %606, align 8, !tbaa !8
,double8B

	full_text

double %605
.double*8B

	full_text

double* %606
:fmul8B0
.
	full_text!

%607 = fmul double %604, %603
,double8B

	full_text

double %604
,double8B

	full_text

double %603
Pstore8BE
C
	full_text6
4
2store double %607, double* %602, align 8, !tbaa !8
,double8B

	full_text

double %607
.double*8B

	full_text

double* %602
Abitcast8B4
2
	full_text%
#
!%608 = bitcast i64 %590 to double
&i648B

	full_text


i64 %590
:fmul8B0
.
	full_text!

%609 = fmul double %604, %608
,double8B

	full_text

double %604
,double8B

	full_text

double %608
•getelementptr8Bë
é
	full_textÄ
~
|%610 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %591, i64 %78, i64 %80, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %591
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %609, double* %610, align 8, !tbaa !8
,double8B

	full_text

double %609
.double*8B

	full_text

double* %610
tgetelementptr8Ba
_
	full_textR
P
N%611 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %600, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %600
Pload8BF
D
	full_text7
5
3%612 = load double, double* %611, align 8, !tbaa !8
.double*8B

	full_text

double* %611
Cfsub8B9
7
	full_text*
(
&%613 = fsub double -0.000000e+00, %596
,double8B

	full_text

double %596
mcall8Bc
a
	full_textT
R
P%614 = tail call double @llvm.fmuladd.f64(double %613, double %605, double %597)
,double8B

	full_text

double %613
,double8B

	full_text

double %605
,double8B

	full_text

double %597
tgetelementptr8Ba
_
	full_textR
P
N%615 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %600, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %600
Pstore8BE
C
	full_text6
4
2store double %614, double* %615, align 8, !tbaa !8
,double8B

	full_text

double %614
.double*8B

	full_text

double* %615
mcall8Bc
a
	full_textT
R
P%616 = tail call double @llvm.fmuladd.f64(double %613, double %607, double %612)
,double8B

	full_text

double %613
,double8B

	full_text

double %607
,double8B

	full_text

double %612
Abitcast8B4
2
	full_text%
#
!%617 = bitcast i64 %589 to double
&i648B

	full_text


i64 %589
mcall8Bc
a
	full_textT
R
P%618 = tail call double @llvm.fmuladd.f64(double %613, double %609, double %617)
,double8B

	full_text

double %613
,double8B

	full_text

double %609
,double8B

	full_text

double %617
tgetelementptr8Ba
_
	full_textR
P
N%619 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %601, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %601
Pload8BF
D
	full_text7
5
3%620 = load double, double* %619, align 8, !tbaa !8
.double*8B

	full_text

double* %619
tgetelementptr8Ba
_
	full_textR
P
N%621 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %601, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %601
Pload8BF
D
	full_text7
5
3%622 = load double, double* %621, align 8, !tbaa !8
.double*8B

	full_text

double* %621
tgetelementptr8Ba
_
	full_textR
P
N%623 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %601, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %601
Pload8BF
D
	full_text7
5
3%624 = load double, double* %623, align 8, !tbaa !8
.double*8B

	full_text

double* %623
•getelementptr8Bë
é
	full_textÄ
~
|%625 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %601, i64 %78, i64 %80, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %601
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%626 = load double, double* %625, align 8, !tbaa !8
.double*8B

	full_text

double* %625
Cfsub8B9
7
	full_text*
(
&%627 = fsub double -0.000000e+00, %620
,double8B

	full_text

double %620
mcall8Bc
a
	full_textT
R
P%628 = tail call double @llvm.fmuladd.f64(double %627, double %605, double %622)
,double8B

	full_text

double %627
,double8B

	full_text

double %605
,double8B

	full_text

double %622
Pstore8BE
C
	full_text6
4
2store double %628, double* %621, align 8, !tbaa !8
,double8B

	full_text

double %628
.double*8B

	full_text

double* %621
mcall8Bc
a
	full_textT
R
P%629 = tail call double @llvm.fmuladd.f64(double %627, double %607, double %624)
,double8B

	full_text

double %627
,double8B

	full_text

double %607
,double8B

	full_text

double %624
mcall8Bc
a
	full_textT
R
P%630 = tail call double @llvm.fmuladd.f64(double %627, double %609, double %626)
,double8B

	full_text

double %627
,double8B

	full_text

double %609
,double8B

	full_text

double %626
tgetelementptr8Ba
_
	full_textR
P
N%631 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %591, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %591
Pload8BF
D
	full_text7
5
3%632 = load double, double* %631, align 8, !tbaa !8
.double*8B

	full_text

double* %631
Bfdiv8B8
6
	full_text)
'
%%633 = fdiv double 1.000000e+00, %594
,double8B

	full_text

double %594
:fmul8B0
.
	full_text!

%634 = fmul double %595, %633
,double8B

	full_text

double %595
,double8B

	full_text

double %633
tgetelementptr8Ba
_
	full_textR
P
N%635 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %591, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %591
Pstore8BE
C
	full_text6
4
2store double %634, double* %635, align 8, !tbaa !8
,double8B

	full_text

double %634
.double*8B

	full_text

double* %635
:fmul8B0
.
	full_text!

%636 = fmul double %633, %632
,double8B

	full_text

double %633
,double8B

	full_text

double %632
Pstore8BE
C
	full_text6
4
2store double %636, double* %631, align 8, !tbaa !8
,double8B

	full_text

double %636
.double*8B

	full_text

double* %631
Abitcast8B4
2
	full_text%
#
!%637 = bitcast i64 %588 to double
&i648B

	full_text


i64 %588
:fmul8B0
.
	full_text!

%638 = fmul double %633, %637
,double8B

	full_text

double %633
,double8B

	full_text

double %637
•getelementptr8Bë
é
	full_textÄ
~
|%639 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %591, i64 %78, i64 %80, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %591
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %638, double* %639, align 8, !tbaa !8
,double8B

	full_text

double %638
.double*8B

	full_text

double* %639
tgetelementptr8Ba
_
	full_textR
P
N%640 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %600, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %600
Pload8BF
D
	full_text7
5
3%641 = load double, double* %640, align 8, !tbaa !8
.double*8B

	full_text

double* %640
Cfsub8B9
7
	full_text*
(
&%642 = fsub double -0.000000e+00, %592
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%643 = tail call double @llvm.fmuladd.f64(double %642, double %634, double %593)
,double8B

	full_text

double %642
,double8B

	full_text

double %634
,double8B

	full_text

double %593
tgetelementptr8Ba
_
	full_textR
P
N%644 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %600, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %600
Pstore8BE
C
	full_text6
4
2store double %643, double* %644, align 8, !tbaa !8
,double8B

	full_text

double %643
.double*8B

	full_text

double* %644
mcall8Bc
a
	full_textT
R
P%645 = tail call double @llvm.fmuladd.f64(double %642, double %636, double %641)
,double8B

	full_text

double %642
,double8B

	full_text

double %636
,double8B

	full_text

double %641
Abitcast8B4
2
	full_text%
#
!%646 = bitcast i64 %587 to double
&i648B

	full_text


i64 %587
mcall8Bc
a
	full_textT
R
P%647 = tail call double @llvm.fmuladd.f64(double %642, double %638, double %646)
,double8B

	full_text

double %642
,double8B

	full_text

double %638
,double8B

	full_text

double %646
tgetelementptr8Ba
_
	full_textR
P
N%648 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %601, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %601
Pload8BF
D
	full_text7
5
3%649 = load double, double* %648, align 8, !tbaa !8
.double*8B

	full_text

double* %648
tgetelementptr8Ba
_
	full_textR
P
N%650 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %601, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %601
Pload8BF
D
	full_text7
5
3%651 = load double, double* %650, align 8, !tbaa !8
.double*8B

	full_text

double* %650
tgetelementptr8Ba
_
	full_textR
P
N%652 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %601, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %601
Pload8BF
D
	full_text7
5
3%653 = load double, double* %652, align 8, !tbaa !8
.double*8B

	full_text

double* %652
•getelementptr8Bë
é
	full_textÄ
~
|%654 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %601, i64 %78, i64 %80, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %601
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%655 = load double, double* %654, align 8, !tbaa !8
.double*8B

	full_text

double* %654
Cfsub8B9
7
	full_text*
(
&%656 = fsub double -0.000000e+00, %649
,double8B

	full_text

double %649
mcall8Bc
a
	full_textT
R
P%657 = tail call double @llvm.fmuladd.f64(double %656, double %634, double %651)
,double8B

	full_text

double %656
,double8B

	full_text

double %634
,double8B

	full_text

double %651
Pstore8BE
C
	full_text6
4
2store double %657, double* %650, align 8, !tbaa !8
,double8B

	full_text

double %657
.double*8B

	full_text

double* %650
mcall8Bc
a
	full_textT
R
P%658 = tail call double @llvm.fmuladd.f64(double %656, double %636, double %653)
,double8B

	full_text

double %656
,double8B

	full_text

double %636
,double8B

	full_text

double %653
mcall8Bc
a
	full_textT
R
P%659 = tail call double @llvm.fmuladd.f64(double %656, double %638, double %655)
,double8B

	full_text

double %656
,double8B

	full_text

double %638
,double8B

	full_text

double %655
:icmp8B0
.
	full_text!

%660 = icmp eq i64 %600, %585
&i648B

	full_text


i64 %600
&i648B

	full_text


i64 %585
Abitcast8B4
2
	full_text%
#
!%661 = bitcast double %618 to i64
,double8B

	full_text

double %618
Abitcast8B4
2
	full_text%
#
!%662 = bitcast double %630 to i64
,double8B

	full_text

double %630
Abitcast8B4
2
	full_text%
#
!%663 = bitcast double %647 to i64
,double8B

	full_text

double %647
Abitcast8B4
2
	full_text%
#
!%664 = bitcast double %659 to i64
,double8B

	full_text

double %659
=br8B5
3
	full_text&
$
"br i1 %660, label %665, label %586
$i18B

	full_text
	
i1 %660
Pstore8BE
C
	full_text6
4
2store double %609, double* %580, align 8, !tbaa !8
,double8B

	full_text

double %609
.double*8B

	full_text

double* %580
Pstore8BE
C
	full_text6
4
2store double %618, double* %567, align 8, !tbaa !8
,double8B

	full_text

double %618
.double*8B

	full_text

double* %567
Qstore8BF
D
	full_text7
5
3store double %638, double* %582, align 16, !tbaa !8
,double8B

	full_text

double %638
.double*8B

	full_text

double* %582
Qstore8BF
D
	full_text7
5
3store double %647, double* %572, align 16, !tbaa !8
,double8B

	full_text

double %647
.double*8B

	full_text

double* %572
(br8B 

	full_text

br label %666
Iphi8B@
>
	full_text1
/
-%667 = phi i64 [ %664, %665 ], [ %579, %500 ]
&i648B

	full_text


i64 %664
&i648B

	full_text


i64 %579
Iphi8B@
>
	full_text1
/
-%668 = phi i64 [ %663, %665 ], [ %571, %500 ]
&i648B

	full_text


i64 %663
&i648B

	full_text


i64 %571
Iphi8B@
>
	full_text1
/
-%669 = phi i64 [ %662, %665 ], [ %576, %500 ]
&i648B

	full_text


i64 %662
&i648B

	full_text


i64 %576
Iphi8B@
>
	full_text1
/
-%670 = phi i64 [ %661, %665 ], [ %566, %500 ]
&i648B

	full_text


i64 %661
&i648B

	full_text


i64 %566
Lphi8BC
A
	full_text4
2
0%671 = phi double [ %616, %665 ], [ %557, %500 ]
,double8B

	full_text

double %616
,double8B

	full_text

double %557
Lphi8BC
A
	full_text4
2
0%672 = phi double [ %614, %665 ], [ %556, %500 ]
,double8B

	full_text

double %614
,double8B

	full_text

double %556
Lphi8BC
A
	full_text4
2
0%673 = phi double [ %629, %665 ], [ %559, %500 ]
,double8B

	full_text

double %629
,double8B

	full_text

double %559
Lphi8BC
A
	full_text4
2
0%674 = phi double [ %628, %665 ], [ %558, %500 ]
,double8B

	full_text

double %628
,double8B

	full_text

double %558
Lphi8BC
A
	full_text4
2
0%675 = phi double [ %645, %665 ], [ %561, %500 ]
,double8B

	full_text

double %645
,double8B

	full_text

double %561
Lphi8BC
A
	full_text4
2
0%676 = phi double [ %643, %665 ], [ %560, %500 ]
,double8B

	full_text

double %643
,double8B

	full_text

double %560
Lphi8BC
A
	full_text4
2
0%677 = phi double [ %658, %665 ], [ %563, %500 ]
,double8B

	full_text

double %658
,double8B

	full_text

double %563
Lphi8BC
A
	full_text4
2
0%678 = phi double [ %657, %665 ], [ %562, %500 ]
,double8B

	full_text

double %657
,double8B

	full_text

double %562
tgetelementptr8Ba
_
	full_textR
P
N%679 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %519, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %519
Pload8BF
D
	full_text7
5
3%680 = load double, double* %679, align 8, !tbaa !8
.double*8B

	full_text

double* %679
Jstore8B?
=
	full_text0
.
,store i64 %670, i64* %581, align 8, !tbaa !8
&i648B

	full_text


i64 %670
(i64*8B

	full_text

	i64* %581
Bfdiv8B8
6
	full_text)
'
%%681 = fdiv double 1.000000e+00, %672
,double8B

	full_text

double %672
:fmul8B0
.
	full_text!

%682 = fmul double %671, %681
,double8B

	full_text

double %671
,double8B

	full_text

double %681
tgetelementptr8Ba
_
	full_textR
P
N%683 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %519, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %519
Pstore8BE
C
	full_text6
4
2store double %682, double* %683, align 8, !tbaa !8
,double8B

	full_text

double %682
.double*8B

	full_text

double* %683
:fmul8B0
.
	full_text!

%684 = fmul double %681, %680
,double8B

	full_text

double %681
,double8B

	full_text

double %680
Pstore8BE
C
	full_text6
4
2store double %684, double* %679, align 8, !tbaa !8
,double8B

	full_text

double %684
.double*8B

	full_text

double* %679
Abitcast8B4
2
	full_text%
#
!%685 = bitcast i64 %670 to double
&i648B

	full_text


i64 %670
:fmul8B0
.
	full_text!

%686 = fmul double %681, %685
,double8B

	full_text

double %681
,double8B

	full_text

double %685
•getelementptr8Bë
é
	full_textÄ
~
|%687 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %519, i64 %78, i64 %80, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %519
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %686, double* %687, align 8, !tbaa !8
,double8B

	full_text

double %686
.double*8B

	full_text

double* %687
tgetelementptr8Ba
_
	full_textR
P
N%688 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %536, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %536
Pload8BF
D
	full_text7
5
3%689 = load double, double* %688, align 8, !tbaa !8
.double*8B

	full_text

double* %688
Cfsub8B9
7
	full_text*
(
&%690 = fsub double -0.000000e+00, %674
,double8B

	full_text

double %674
mcall8Bc
a
	full_textT
R
P%691 = tail call double @llvm.fmuladd.f64(double %690, double %682, double %673)
,double8B

	full_text

double %690
,double8B

	full_text

double %682
,double8B

	full_text

double %673
tgetelementptr8Ba
_
	full_textR
P
N%692 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %536, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %536
Pstore8BE
C
	full_text6
4
2store double %691, double* %692, align 8, !tbaa !8
,double8B

	full_text

double %691
.double*8B

	full_text

double* %692
mcall8Bc
a
	full_textT
R
P%693 = tail call double @llvm.fmuladd.f64(double %690, double %684, double %689)
,double8B

	full_text

double %690
,double8B

	full_text

double %684
,double8B

	full_text

double %689
Pstore8BE
C
	full_text6
4
2store double %693, double* %688, align 8, !tbaa !8
,double8B

	full_text

double %693
.double*8B

	full_text

double* %688
Abitcast8B4
2
	full_text%
#
!%694 = bitcast i64 %669 to double
&i648B

	full_text


i64 %669
mcall8Bc
a
	full_textT
R
P%695 = tail call double @llvm.fmuladd.f64(double %690, double %686, double %694)
,double8B

	full_text

double %690
,double8B

	full_text

double %686
,double8B

	full_text

double %694
tgetelementptr8Ba
_
	full_textR
P
N%696 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %519, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %519
Pload8BF
D
	full_text7
5
3%697 = load double, double* %696, align 8, !tbaa !8
.double*8B

	full_text

double* %696
Kstore8B@
>
	full_text1
/
-store i64 %668, i64* %583, align 16, !tbaa !8
&i648B

	full_text


i64 %668
(i64*8B

	full_text

	i64* %583
Bfdiv8B8
6
	full_text)
'
%%698 = fdiv double 1.000000e+00, %676
,double8B

	full_text

double %676
:fmul8B0
.
	full_text!

%699 = fmul double %675, %698
,double8B

	full_text

double %675
,double8B

	full_text

double %698
tgetelementptr8Ba
_
	full_textR
P
N%700 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %519, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %519
Pstore8BE
C
	full_text6
4
2store double %699, double* %700, align 8, !tbaa !8
,double8B

	full_text

double %699
.double*8B

	full_text

double* %700
:fmul8B0
.
	full_text!

%701 = fmul double %698, %697
,double8B

	full_text

double %698
,double8B

	full_text

double %697
Pstore8BE
C
	full_text6
4
2store double %701, double* %696, align 8, !tbaa !8
,double8B

	full_text

double %701
.double*8B

	full_text

double* %696
Abitcast8B4
2
	full_text%
#
!%702 = bitcast i64 %668 to double
&i648B

	full_text


i64 %668
:fmul8B0
.
	full_text!

%703 = fmul double %698, %702
,double8B

	full_text

double %698
,double8B

	full_text

double %702
•getelementptr8Bë
é
	full_textÄ
~
|%704 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %519, i64 %78, i64 %80, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %519
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %703, double* %704, align 8, !tbaa !8
,double8B

	full_text

double %703
.double*8B

	full_text

double* %704
tgetelementptr8Ba
_
	full_textR
P
N%705 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %536, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %536
Pload8BF
D
	full_text7
5
3%706 = load double, double* %705, align 8, !tbaa !8
.double*8B

	full_text

double* %705
Cfsub8B9
7
	full_text*
(
&%707 = fsub double -0.000000e+00, %678
,double8B

	full_text

double %678
mcall8Bc
a
	full_textT
R
P%708 = tail call double @llvm.fmuladd.f64(double %707, double %699, double %677)
,double8B

	full_text

double %707
,double8B

	full_text

double %699
,double8B

	full_text

double %677
tgetelementptr8Ba
_
	full_textR
P
N%709 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %536, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %536
Pstore8BE
C
	full_text6
4
2store double %708, double* %709, align 8, !tbaa !8
,double8B

	full_text

double %708
.double*8B

	full_text

double* %709
mcall8Bc
a
	full_textT
R
P%710 = tail call double @llvm.fmuladd.f64(double %707, double %701, double %706)
,double8B

	full_text

double %707
,double8B

	full_text

double %701
,double8B

	full_text

double %706
Pstore8BE
C
	full_text6
4
2store double %710, double* %705, align 8, !tbaa !8
,double8B

	full_text

double %710
.double*8B

	full_text

double* %705
Abitcast8B4
2
	full_text%
#
!%711 = bitcast i64 %667 to double
&i648B

	full_text


i64 %667
mcall8Bc
a
	full_textT
R
P%712 = tail call double @llvm.fmuladd.f64(double %707, double %703, double %711)
,double8B

	full_text

double %707
,double8B

	full_text

double %703
,double8B

	full_text

double %711
:fdiv8B0
.
	full_text!

%713 = fdiv double %695, %691
,double8B

	full_text

double %695
,double8B

	full_text

double %691
•getelementptr8Bë
é
	full_textÄ
~
|%714 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %536, i64 %78, i64 %80, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %713, double* %714, align 8, !tbaa !8
,double8B

	full_text

double %713
.double*8B

	full_text

double* %714
:fdiv8B0
.
	full_text!

%715 = fdiv double %712, %708
,double8B

	full_text

double %712
,double8B

	full_text

double %708
•getelementptr8Bë
é
	full_textÄ
~
|%716 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %536, i64 %78, i64 %80, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pstore8BE
C
	full_text6
4
2store double %715, double* %716, align 8, !tbaa !8
,double8B

	full_text

double %715
.double*8B

	full_text

double* %716
Pload8BF
D
	full_text7
5
3%717 = load double, double* %524, align 8, !tbaa !8
.double*8B

	full_text

double* %524
Cfsub8B9
7
	full_text*
(
&%718 = fsub double -0.000000e+00, %717
,double8B

	full_text

double %717
Pload8BF
D
	full_text7
5
3%719 = load double, double* %551, align 8, !tbaa !8
.double*8B

	full_text

double* %551
Pstore8BE
C
	full_text6
4
2store double %719, double* %501, align 8, !tbaa !8
,double8B

	full_text

double %719
.double*8B

	full_text

double* %501
Pload8BF
D
	full_text7
5
3%720 = load double, double* %528, align 8, !tbaa !8
.double*8B

	full_text

double* %528
mcall8Bc
a
	full_textT
R
P%721 = tail call double @llvm.fmuladd.f64(double %718, double %719, double %720)
,double8B

	full_text

double %718
,double8B

	full_text

double %719
,double8B

	full_text

double %720
Pstore8BE
C
	full_text6
4
2store double %721, double* %502, align 8, !tbaa !8
,double8B

	full_text

double %721
.double*8B

	full_text

double* %502
Pstore8BE
C
	full_text6
4
2store double %721, double* %528, align 8, !tbaa !8
,double8B

	full_text

double %721
.double*8B

	full_text

double* %528
Pload8BF
D
	full_text7
5
3%722 = load double, double* %553, align 8, !tbaa !8
.double*8B

	full_text

double* %553
Pstore8BE
C
	full_text6
4
2store double %722, double* %400, align 8, !tbaa !8
,double8B

	full_text

double %722
.double*8B

	full_text

double* %400
Pload8BF
D
	full_text7
5
3%723 = load double, double* %531, align 8, !tbaa !8
.double*8B

	full_text

double* %531
mcall8Bc
a
	full_textT
R
P%724 = tail call double @llvm.fmuladd.f64(double %718, double %722, double %723)
,double8B

	full_text

double %718
,double8B

	full_text

double %722
,double8B

	full_text

double %723
Pstore8BE
C
	full_text6
4
2store double %724, double* %504, align 8, !tbaa !8
,double8B

	full_text

double %724
.double*8B

	full_text

double* %504
Pstore8BE
C
	full_text6
4
2store double %724, double* %531, align 8, !tbaa !8
,double8B

	full_text

double %724
.double*8B

	full_text

double* %531
Pload8BF
D
	full_text7
5
3%725 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
Qstore8BF
D
	full_text7
5
3store double %725, double* %405, align 16, !tbaa !8
,double8B

	full_text

double %725
.double*8B

	full_text

double* %405
Pload8BF
D
	full_text7
5
3%726 = load double, double* %534, align 8, !tbaa !8
.double*8B

	full_text

double* %534
mcall8Bc
a
	full_textT
R
P%727 = tail call double @llvm.fmuladd.f64(double %718, double %725, double %726)
,double8B

	full_text

double %718
,double8B

	full_text

double %725
,double8B

	full_text

double %726
Pstore8BE
C
	full_text6
4
2store double %727, double* %503, align 8, !tbaa !8
,double8B

	full_text

double %727
.double*8B

	full_text

double* %503
Pstore8BE
C
	full_text6
4
2store double %727, double* %534, align 8, !tbaa !8
,double8B

	full_text

double %727
.double*8B

	full_text

double* %534
Pstore8BE
C
	full_text6
4
2store double %713, double* %567, align 8, !tbaa !8
,double8B

	full_text

double %713
.double*8B

	full_text

double* %567
Pload8BF
D
	full_text7
5
3%728 = load double, double* %687, align 8, !tbaa !8
.double*8B

	full_text

double* %687
Pload8BF
D
	full_text7
5
3%729 = load double, double* %683, align 8, !tbaa !8
.double*8B

	full_text

double* %683
Cfsub8B9
7
	full_text*
(
&%730 = fsub double -0.000000e+00, %729
,double8B

	full_text

double %729
mcall8Bc
a
	full_textT
R
P%731 = tail call double @llvm.fmuladd.f64(double %730, double %713, double %728)
,double8B

	full_text

double %730
,double8B

	full_text

double %713
,double8B

	full_text

double %728
Pstore8BE
C
	full_text6
4
2store double %731, double* %580, align 8, !tbaa !8
,double8B

	full_text

double %731
.double*8B

	full_text

double* %580
Pstore8BE
C
	full_text6
4
2store double %731, double* %687, align 8, !tbaa !8
,double8B

	full_text

double %731
.double*8B

	full_text

double* %687
Qstore8BF
D
	full_text7
5
3store double %715, double* %572, align 16, !tbaa !8
,double8B

	full_text

double %715
.double*8B

	full_text

double* %572
Pload8BF
D
	full_text7
5
3%732 = load double, double* %704, align 8, !tbaa !8
.double*8B

	full_text

double* %704
Pload8BF
D
	full_text7
5
3%733 = load double, double* %700, align 8, !tbaa !8
.double*8B

	full_text

double* %700
Cfsub8B9
7
	full_text*
(
&%734 = fsub double -0.000000e+00, %733
,double8B

	full_text

double %733
mcall8Bc
a
	full_textT
R
P%735 = tail call double @llvm.fmuladd.f64(double %734, double %715, double %732)
,double8B

	full_text

double %734
,double8B

	full_text

double %715
,double8B

	full_text

double %732
Qstore8BF
D
	full_text7
5
3store double %735, double* %582, align 16, !tbaa !8
,double8B

	full_text

double %735
.double*8B

	full_text

double* %582
Pstore8BE
C
	full_text6
4
2store double %735, double* %704, align 8, !tbaa !8
,double8B

	full_text

double %735
.double*8B

	full_text

double* %704
7icmp8B-
+
	full_text

%736 = icmp sgt i32 %12, 2
=br8B5
3
	full_text&
$
"br i1 %736, label %737, label %792
$i18B

	full_text
	
i1 %736
8sext8B.
,
	full_text

%738 = sext i32 %418 to i64
&i328B

	full_text


i32 %418
(br8B 

	full_text

br label %739
Lphi8BC
A
	full_text4
2
0%740 = phi double [ %735, %737 ], [ %788, %739 ]
,double8B

	full_text

double %735
,double8B

	full_text

double %788
Lphi8BC
A
	full_text4
2
0%741 = phi double [ %715, %737 ], [ %740, %739 ]
,double8B

	full_text

double %715
,double8B

	full_text

double %740
Lphi8BC
A
	full_text4
2
0%742 = phi double [ %731, %737 ], [ %778, %739 ]
,double8B

	full_text

double %731
,double8B

	full_text

double %778
Lphi8BC
A
	full_text4
2
0%743 = phi double [ %713, %737 ], [ %742, %739 ]
,double8B

	full_text

double %713
,double8B

	full_text

double %742
Lphi8BC
A
	full_text4
2
0%744 = phi double [ %727, %737 ], [ %768, %739 ]
,double8B

	full_text

double %727
,double8B

	full_text

double %768
Lphi8BC
A
	full_text4
2
0%745 = phi double [ %725, %737 ], [ %744, %739 ]
,double8B

	full_text

double %725
,double8B

	full_text

double %744
Lphi8BC
A
	full_text4
2
0%746 = phi double [ %724, %737 ], [ %764, %739 ]
,double8B

	full_text

double %724
,double8B

	full_text

double %764
Lphi8BC
A
	full_text4
2
0%747 = phi double [ %722, %737 ], [ %746, %739 ]
,double8B

	full_text

double %722
,double8B

	full_text

double %746
Lphi8BC
A
	full_text4
2
0%748 = phi double [ %721, %737 ], [ %760, %739 ]
,double8B

	full_text

double %721
,double8B

	full_text

double %760
Lphi8BC
A
	full_text4
2
0%749 = phi double [ %719, %737 ], [ %748, %739 ]
,double8B

	full_text

double %719
,double8B

	full_text

double %748
Iphi8B@
>
	full_text1
/
-%750 = phi i64 [ %738, %737 ], [ %789, %739 ]
&i648B

	full_text


i64 %738
&i648B

	full_text


i64 %789
tgetelementptr8Ba
_
	full_textR
P
N%751 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %750, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %750
Pload8BF
D
	full_text7
5
3%752 = load double, double* %751, align 8, !tbaa !8
.double*8B

	full_text

double* %751
tgetelementptr8Ba
_
	full_textR
P
N%753 = getelementptr inbounds [5 x double], [5 x double]* %40, i64 %750, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
&i648B

	full_text


i64 %750
Pload8BF
D
	full_text7
5
3%754 = load double, double* %753, align 8, !tbaa !8
.double*8B

	full_text

double* %753
Cfsub8B9
7
	full_text*
(
&%755 = fsub double -0.000000e+00, %752
,double8B

	full_text

double %752
Cfsub8B9
7
	full_text*
(
&%756 = fsub double -0.000000e+00, %754
,double8B

	full_text

double %754
•getelementptr8Bë
é
	full_textÄ
~
|%757 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %750, i64 %78, i64 %80, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %750
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%758 = load double, double* %757, align 8, !tbaa !8
.double*8B

	full_text

double* %757
mcall8Bc
a
	full_textT
R
P%759 = tail call double @llvm.fmuladd.f64(double %755, double %748, double %758)
,double8B

	full_text

double %755
,double8B

	full_text

double %748
,double8B

	full_text

double %758
mcall8Bc
a
	full_textT
R
P%760 = tail call double @llvm.fmuladd.f64(double %756, double %749, double %759)
,double8B

	full_text

double %756
,double8B

	full_text

double %749
,double8B

	full_text

double %759
Pstore8BE
C
	full_text6
4
2store double %760, double* %757, align 8, !tbaa !8
,double8B

	full_text

double %760
.double*8B

	full_text

double* %757
•getelementptr8Bë
é
	full_textÄ
~
|%761 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %750, i64 %78, i64 %80, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %750
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%762 = load double, double* %761, align 8, !tbaa !8
.double*8B

	full_text

double* %761
mcall8Bc
a
	full_textT
R
P%763 = tail call double @llvm.fmuladd.f64(double %755, double %746, double %762)
,double8B

	full_text

double %755
,double8B

	full_text

double %746
,double8B

	full_text

double %762
mcall8Bc
a
	full_textT
R
P%764 = tail call double @llvm.fmuladd.f64(double %756, double %747, double %763)
,double8B

	full_text

double %756
,double8B

	full_text

double %747
,double8B

	full_text

double %763
Pstore8BE
C
	full_text6
4
2store double %764, double* %761, align 8, !tbaa !8
,double8B

	full_text

double %764
.double*8B

	full_text

double* %761
•getelementptr8Bë
é
	full_textÄ
~
|%765 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %750, i64 %78, i64 %80, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %750
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%766 = load double, double* %765, align 8, !tbaa !8
.double*8B

	full_text

double* %765
mcall8Bc
a
	full_textT
R
P%767 = tail call double @llvm.fmuladd.f64(double %755, double %744, double %766)
,double8B

	full_text

double %755
,double8B

	full_text

double %744
,double8B

	full_text

double %766
mcall8Bc
a
	full_textT
R
P%768 = tail call double @llvm.fmuladd.f64(double %756, double %745, double %767)
,double8B

	full_text

double %756
,double8B

	full_text

double %745
,double8B

	full_text

double %767
Pstore8BE
C
	full_text6
4
2store double %768, double* %765, align 8, !tbaa !8
,double8B

	full_text

double %768
.double*8B

	full_text

double* %765
•getelementptr8Bë
é
	full_textÄ
~
|%769 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %750, i64 %78, i64 %80, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %750
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%770 = load double, double* %769, align 8, !tbaa !8
.double*8B

	full_text

double* %769
tgetelementptr8Ba
_
	full_textR
P
N%771 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %750, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %750
Pload8BF
D
	full_text7
5
3%772 = load double, double* %771, align 8, !tbaa !8
.double*8B

	full_text

double* %771
Cfsub8B9
7
	full_text*
(
&%773 = fsub double -0.000000e+00, %772
,double8B

	full_text

double %772
mcall8Bc
a
	full_textT
R
P%774 = tail call double @llvm.fmuladd.f64(double %773, double %742, double %770)
,double8B

	full_text

double %773
,double8B

	full_text

double %742
,double8B

	full_text

double %770
tgetelementptr8Ba
_
	full_textR
P
N%775 = getelementptr inbounds [5 x double], [5 x double]* %42, i64 %750, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %42
&i648B

	full_text


i64 %750
Pload8BF
D
	full_text7
5
3%776 = load double, double* %775, align 8, !tbaa !8
.double*8B

	full_text

double* %775
Cfsub8B9
7
	full_text*
(
&%777 = fsub double -0.000000e+00, %776
,double8B

	full_text

double %776
mcall8Bc
a
	full_textT
R
P%778 = tail call double @llvm.fmuladd.f64(double %777, double %743, double %774)
,double8B

	full_text

double %777
,double8B

	full_text

double %743
,double8B

	full_text

double %774
Pstore8BE
C
	full_text6
4
2store double %778, double* %769, align 8, !tbaa !8
,double8B

	full_text

double %778
.double*8B

	full_text

double* %769
•getelementptr8Bë
é
	full_textÄ
~
|%779 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %75, i64 %750, i64 %78, i64 %80, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %75
&i648B

	full_text


i64 %750
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %80
Pload8BF
D
	full_text7
5
3%780 = load double, double* %779, align 8, !tbaa !8
.double*8B

	full_text

double* %779
tgetelementptr8Ba
_
	full_textR
P
N%781 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %750, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %750
Pload8BF
D
	full_text7
5
3%782 = load double, double* %781, align 8, !tbaa !8
.double*8B

	full_text

double* %781
Cfsub8B9
7
	full_text*
(
&%783 = fsub double -0.000000e+00, %782
,double8B

	full_text

double %782
mcall8Bc
a
	full_textT
R
P%784 = tail call double @llvm.fmuladd.f64(double %783, double %740, double %780)
,double8B

	full_text

double %783
,double8B

	full_text

double %740
,double8B

	full_text

double %780
tgetelementptr8Ba
_
	full_textR
P
N%785 = getelementptr inbounds [5 x double], [5 x double]* %44, i64 %750, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %44
&i648B

	full_text


i64 %750
Pload8BF
D
	full_text7
5
3%786 = load double, double* %785, align 8, !tbaa !8
.double*8B

	full_text

double* %785
Cfsub8B9
7
	full_text*
(
&%787 = fsub double -0.000000e+00, %786
,double8B

	full_text

double %786
mcall8Bc
a
	full_textT
R
P%788 = tail call double @llvm.fmuladd.f64(double %787, double %741, double %784)
,double8B

	full_text

double %787
,double8B

	full_text

double %741
,double8B

	full_text

double %784
Pstore8BE
C
	full_text6
4
2store double %788, double* %779, align 8, !tbaa !8
,double8B

	full_text

double %788
.double*8B

	full_text

double* %779
7add8B.
,
	full_text

%789 = add nsw i64 %750, -1
&i648B

	full_text


i64 %750
8icmp8B.
,
	full_text

%790 = icmp sgt i64 %750, 0
&i648B

	full_text


i64 %750
=br8B5
3
	full_text&
$
"br i1 %790, label %739, label %791
$i18B

	full_text
	
i1 %790
Pstore8BE
C
	full_text6
4
2store double %748, double* %501, align 8, !tbaa !8
,double8B

	full_text

double %748
.double*8B

	full_text

double* %501
Pstore8BE
C
	full_text6
4
2store double %760, double* %502, align 8, !tbaa !8
,double8B

	full_text

double %760
.double*8B

	full_text

double* %502
Pstore8BE
C
	full_text6
4
2store double %746, double* %400, align 8, !tbaa !8
,double8B

	full_text

double %746
.double*8B

	full_text

double* %400
Qstore8BF
D
	full_text7
5
3store double %744, double* %405, align 16, !tbaa !8
,double8B

	full_text

double %744
.double*8B

	full_text

double* %405
Pstore8BE
C
	full_text6
4
2store double %742, double* %567, align 8, !tbaa !8
,double8B

	full_text

double %742
.double*8B

	full_text

double* %567
Pstore8BE
C
	full_text6
4
2store double %778, double* %580, align 8, !tbaa !8
,double8B

	full_text

double %778
.double*8B

	full_text

double* %580
Qstore8BF
D
	full_text7
5
3store double %740, double* %572, align 16, !tbaa !8
,double8B

	full_text

double %740
.double*8B

	full_text

double* %572
Qstore8BF
D
	full_text7
5
3store double %788, double* %582, align 16, !tbaa !8
,double8B

	full_text

double %788
.double*8B

	full_text

double* %582
Pstore8BE
C
	full_text6
4
2store double %764, double* %504, align 8, !tbaa !8
,double8B

	full_text

double %764
.double*8B

	full_text

double* %504
Pstore8BE
C
	full_text6
4
2store double %768, double* %503, align 8, !tbaa !8
,double8B

	full_text

double %768
.double*8B

	full_text

double* %503
(br8B 

	full_text

br label %792
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %16) #4
%i8*8B

	full_text
	
i8* %16
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %17) #4
%i8*8B

	full_text
	
i8* %17
$ret8B

	full_text


ret void
%i328B

	full_text
	
i32 %11
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %7
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %3
%i328B

	full_text
	
i32 %12
$i328B

	full_text


i32 %9
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %6
%i328B

	full_text
	
i32 %10
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %8
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
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0x4017D0624DD2F1AB
:double8B,
*
	full_text

double 0x4027D0624DD2F1AB
4double8B&
$
	full_text

double 1.000000e+00
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 32
$i648B

	full_text


i64 40
5double8B'
%
	full_text

double -4.725000e-02
$i648B

	full_text


i64 -1
#i328B

	full_text	

i32 0
&i648B

	full_text


i64 8450
#i328B

	full_text	

i32 3
#i328B

	full_text	

i32 6
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 1
:double8B,
*
	full_text

double 0x3F626E978D4FDF3C
#i328B

	full_text	

i32 2
4double8B&
$
	full_text

double 1.000000e-01
'i648B

	full_text

	i64 21125
4double8B&
$
	full_text

double 4.725000e-02
$i328B

	full_text


i32 -3
4double8B&
$
	full_text

double 1.875000e-03
#i328B

	full_text	

i32 5
$i328B

	full_text


i32 -2
:double8B,
*
	full_text

double 0x3FFF5C28F5C28F5B
#i648B

	full_text	

i64 5
5double8B'
%
	full_text

double -0.000000e+00
#i648B

	full_text	

i64 2
:double8B,
*
	full_text

double 0x3FF5555555555555
&i648B

	full_text


i64 4225
4double8B&
$
	full_text

double 0.000000e+00
5double8B'
%
	full_text

double -1.500000e-03
'i648B

	full_text

	i64 12675
4double8B&
$
	full_text

double 3.750000e-04
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 10
%i328B

	full_text
	
i32 325        	
 		                      !    "# "" $% $$ &' &( && )* )) +, ++ -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?? @A @@ BC BB DE DD FG FF HI HJ HH KL KK MN MO MM PQ PP RS RT RR UV UU WX WW YZ YY [\ [[ ]^ ]] _` __ ab aa cd ce cc fg ff hi hj hh kl kk mn mo mm pq pp rs rr tu tt vw vv xy xz xx {| {} {{ ~ ~	Ä ~~ ÅÇ ÅÅ É
Ñ ÉÉ ÖÜ ÖÖ á
à áá âä ââ ã
å ãã çé ç
è çç ê
ë êê íì í
î íí ï
ñ ïï óò ó
ô óó ö
õ öö úù úú û
ü ûû †° †† ¢
£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®
™ ®® ´
¨ ´´ ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ
∂ µµ ∑∑ ∏∏ π
∫ ππ ª
º ªª Ω
æ ΩΩ ø
¿ øø ¡
¬ ¡¡ √
ƒ √√ ≈
∆ ≈≈ «» «« …  …… ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œ
“ œœ ”‘ ”” ’÷ ’’ ◊ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È Á
Í ÁÁ ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ı
¯ ıı ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ ÄÄ ÇÇ ÉÑ ÉÉ ÖÜ Ö
á Ö
à ÖÖ âä ââ ãå ãã çç éè éé êë ê
í ê
ì êê îï îî ñó ññ òô ò
ö òò õú õõ ùû ùù ü† ü
° üü ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ ππ ∫ª ∫∫ ºΩ º
æ º
ø ºº ¿¡ ¿¿ ¬¬ √ƒ √√ ≈∆ ≈
« ≈
» ≈≈ …  …… ÀÃ ÀÀ ÕÕ Œœ ŒŒ –— –
“ –
” –– ‘’ ‘‘ ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚
Â ‚‚ ÊÁ ÊÊ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú 
Û  Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˘ ˙˚ ˙˙ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ ÄÄ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá ÜÜ à
â àà äã ä
å ää çé çç èê è
ë èè íì íí îï îî ñó ññ òô ò
ö òò õú õõ ù
û ùù ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠≠ Ø
∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »
… »»  À    Ã
Õ ÃÃ Œœ Œ
– ŒŒ —“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ Â
Ê ÂÂ ÁÁ ËÈ ËË ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ ÓÓ Ò  ÚÚ ÛÙ ÛÛ ıˆ ı
˜ ı
¯ ıı ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ Ñ
Ü ÑÑ áà á
â á
ä áá ãå ãã çé çç èê è
ë èè íì í
î íí ïñ ï
ó ï
ò ïï ôö ôô õú õ
ù õõ ûû ü† üü °¢ °
£ °
§ °° •¶ •• ß® ßß ©
™ ©© ´¨ ´´ ≠
Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ªª Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬¬ ƒ
≈ ƒƒ ∆« ∆
» ∆∆ …  …… ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “
” ““ ‘’ ‘‘ ÷
◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ áà á
â áá äã ää å
ç åå éé èè êë êì íí îñ ïï óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´
≠ ´
Æ ´
Ø ´´ ∞± ∞∞ ≤≥ ≤≤ ¥µ ¥
∂ ¥
∑ ¥
∏ ¥¥ π∫ ππ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »
À »» ÃÕ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷
Ÿ ÷÷ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡
‰ ‡‡ ÂÊ ÂÂ ÁË Á
È ÁÁ Í
Î ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò 
Ú  ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ á
à áá âä â
ã ââ åç åå éè é
ê éé ëí ë
ì ëë îï î
ñ îî ó
ò óó ôö ô
õ ôô ú
ù úú ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π
∫ ππ ªº ª
Ω ªª æ
ø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €
‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ ÙÙ ıˆ ı
˜ ı
¯ ı
˘ ıı ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛
Å ˛
Ç ˛˛ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ãã çé çç èê è
ë èè íì í
î í
ï íí ñó ññ òô òò öõ ö
ú öö ùû ù
ü ùù †° †
¢ †
£ †† §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™
≠ ™
Æ ™™ Ø∞ ØØ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂
∑ ∂∂ ∏π ∏∏ ∫
ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «« …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —— ”
‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „
‰ „„ ÂÊ Â
Á ÂÂ Ë
È ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ Ö
Ü ÖÖ áà á
â áá ä
ã ää åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß
® ßß ©™ ©
´ ©
¨ ©
≠ ©© ÆØ ÆÆ ∞± ∞∞ ≤≥ ≤
¥ ≤
µ ≤
∂ ≤≤ ∑∏ ∑∑ π∫ π
ª ππ ºΩ º
æ ºº ø¿ øø ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆
… ∆∆  À    ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘
◊ ‘‘ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁ
‚ ﬁﬁ „‰ „„ ÂÊ Â
Á ÂÂ Ë
È ËË ÍÎ ÍÍ Ï
Ì ÏÏ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ Ä	Å	 Ä	
Ç	 Ä	Ä	 É	Ñ	 É	É	 Ö	
Ü	 Ö	Ö	 á	à	 á	
â	 á	á	 ä	ã	 ä	
å	 ä	ä	 ç	é	 ç	
è	 ç	ç	 ê	ë	 ê	
í	 ê	ê	 ì	
î	 ì	ì	 ï	ñ	 ï	
ó	 ï	ï	 ò	
ô	 ò	ò	 ö	õ	 ö	
ú	 ö	ö	 ù	û	 ù	
ü	 ù	ù	 †	°	 †	
¢	 †	†	 £	§	 £	
•	 £	£	 ¶	ß	 ¶	
®	 ¶	¶	 ©	™	 ©	
´	 ©	©	 ¨	≠	 ¨	
Æ	 ¨	¨	 Ø	∞	 Ø	
±	 Ø	Ø	 ≤	≥	 ≤	
¥	 ≤	≤	 µ	
∂	 µ	µ	 ∑	∏	 ∑	
π	 ∑	∑	 ∫	
ª	 ∫	∫	 º	Ω	 º	
æ	 º	º	 ø	¿	 ø	
¡	 ø	ø	 ¬	√	 ¬	
ƒ	 ¬	¬	 ≈	∆	 ≈	
«	 ≈	≈	 »	…	 »	
 	 »	»	 À	Ã	 À	
Õ	 À	À	 Œ	œ	 Œ	
–	 Œ	Œ	 —	“	 —	
”	 —	—	 ‘	’	 ‘	
÷	 ‘	‘	 ◊	
ÿ	 ◊	◊	 Ÿ	⁄	 Ÿ	Ÿ	 €	‹	 €	€	 ›	ﬁ	 ›	›	 ﬂ	‡	 ﬂ	ﬂ	 ·	‚	 ·	
„	 ·	
‰	 ·	·	 Â	Ê	 Â	Â	 Á	Ë	 Á	Á	 È	Í	 È	È	 Î	Ï	 Î	
Ì	 Î	Î	 Ó	Ô	 Ó	
	 Ó	
Ò	 Ó	Ó	 Ú	Û	 Ú	Ú	 Ù	ı	 Ù	Ù	 ˆ	˜	 ˆ	ˆ	 ¯	˘	 ¯	¯	 ˙	˚	 ˙	
¸	 ˙	˙	 ˝	˛	 ˝	
ˇ	 ˝	
Ä
 ˝	˝	 Å
Ç
 Å
Å
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
ä
 â

ã
 â
â
 å
å
 ç
é
 ç
ç
 è
ê
 è

ë
 è

í
 è
è
 ì
î
 ì
ì
 ï
ñ
 ï
ï
 ó
ò
 ó

ô
 ó

ö
 ó
ó
 õ
ú
 õ
õ
 ù
û
 ù
ù
 ü
†
 ü

°
 ü

¢
 ü
ü
 £
§
 £
£
 •
¶
 •
•
 ß
ß
 ®
®
 ©
™
 ©
´
 ¨
≠
 ¨
¨
 Æ
Ø
 Æ
Æ
 ∞
±
 ∞
∞
 ≤
≥
 ≤
≤
 ¥
∂
 µ
µ
 ∑
∏
 ∑
∑
 π
π
 ∫
ª
 ∫
∫
 º
Ω
 º
º
 æ
ø
 æ
æ
 ¿
¬
 ¡

√
 ¡
¡
 ƒ
≈
 ƒ

∆
 ƒ
ƒ
 «
»
 «

…
 «
«
  
À
  

Ã
  
 
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

“
 –
–
 ”
‘
 ”
”
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

È
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
˘
 ˚
¸
 ˚

˝
 ˚
˚
 ˛
ˇ
 ˛

Ä ˛

Å ˛

Ç ˛
˛
 ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ãå ã
ç ã
é ã
è ãã êë ê
í êê ìî ìì ïñ ï
ó ïï òô ò
ö ò
õ ò
ú òò ùû ù
ü ùù †° †
¢ †† £§ ££ •
¶ •• ß® ß
© ß
™ ßß ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ π
ª π
º ππ Ωæ ΩΩ ø¿ ø
¡ ø
¬ øø √ƒ √√ ≈∆ ≈
« ≈≈ »… »»  À  
Ã    ÕŒ ÕÕ œ– œ
— œœ “” ““ ‘’ ‘
÷ ‘
◊ ‘
ÿ ‘‘ Ÿ⁄ ŸŸ €‹ €
› €
ﬁ €
ﬂ €€ ‡· ‡‡ ‚„ ‚
‰ ‚
Â ‚
Ê ‚‚ ÁË ÁÁ È
Í ÈÈ ÎÏ Î
Ì Î
Ó ÎÎ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù Ú
ı ÚÚ ˆ˜ ˆ
¯ ˆ
˘ ˆˆ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛
Ä ˛
Å ˛˛ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à ÜÜ âä ââ ãå ãã çé çç èê èè ëí ëë ìî ìì ïñ ïò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ Ì
Ó ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá Ü
à Ü
â Ü
ä ÜÜ ãå ã
ç ãã éè éé êë ê
í êê ìî ì
ï ìì ñó ñ
ò ñ
ô ñ
ö ññ õú õ
ù õõ ûü ûû †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶
© ¶
™ ¶¶ ´¨ ´
≠ ´´ ÆÆ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂
∏ ∂∂ π
∫ ππ ªº ª
Ω ª
æ ªª ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈
» ≈≈ …  …
À …… ÃÕ ÃÃ Œœ Œ
– Œ
— ŒŒ “” “
‘ ““ ’÷ ’’ ◊ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €
› €€ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰
Ê ‰‰ Á
Ë ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô Ï
 ÏÏ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜
˙ ˜
˚ ˜˜ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà á
â áá äã ää åç åå éè éé êë êê íì íí îï îî ñó ññ òô òò öõ ö
ú ö
ù öö ûü ûû †° †† ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©
¨ ©© ≠Æ ≠≠ Ø∞ ØØ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏∏ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒƒ ∆« ∆∆ »… »»  À    ÃÕ ÃÃ Œœ ŒŒ –— –” ““ ‘÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá Ü
à ÜÜ âä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ïñ ïï óò ó
ô óó öõ ö
ú ö
ù ö
û öö ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •• ß
® ßß ©™ ©
´ ©
¨ ©© ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ π
ª π
º ππ Ωæ Ω
ø ΩΩ ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… ««  À    ÃÕ Ã
Œ Ã
œ Ã
– ÃÃ —“ —— ”
‘ ”” ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹
ﬂ ‹‹ ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰
Ê ‰‰ ÁË ÁÁ È
Í ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇ
É ˇˇ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ää å
ç åå éè é
ê é
ë éé íì í
î íí ïñ ï
ó ïï òô ò
ö ò
õ òò úù úú ûü û
† û
° ûû ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±
¥ ±
µ ±± ∂∑ ∂∂ ∏
π ∏∏ ∫ª ∫
º ∫
Ω ∫∫ æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈
« ≈
» ≈≈ …  …
À …… ÃÕ ÃÃ Œœ ŒŒ –— –– “” ““ ‘’ ‘◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ ÑÖ Ñ
Ü ÑÑ áà á
â áá äã ää åç å
é åå è
ê èè ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †† ¢£ ¢
§ ¢¢ •¶ •
ß •
® •
© •• ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤
≥ ≤≤ ¥µ ¥
∂ ¥
∑ ¥¥ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø æ
¿ æ
¡ ææ ¬√ ¬
ƒ ¬¬ ≈∆ ≈≈ «» «
… «
  «« ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”
‘ ”” ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï È
Ì ÈÈ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯
˙ ¯
˚ ¯¯ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö ÇÇ Üá Ü
à ÜÜ âä ââ ãå ã
ç ã
é ãã èê è
ë èè íì í
î í
ï í
ñ íí óò ó
ô óó öõ ö
ú öö ùû ù
ü ù
† ù
° ùù ¢£ ¢
§ ¢¢ •¶ •• ß
® ßß ©™ ©© ´¨ ´
≠ ´´ ÆØ ÆÆ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” “
‘ “
’ ““ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·· „
‰ „„ ÂÊ Â
Á Â
Ë ÂÂ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı ÙÙ ˆ
˜ ˆˆ ¯˘ ¯
˙ ¯
˚ ¯¯ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÇ ÉÑ ÉÜ ÖÖ áâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ
∂ µµ ∑∏ ∑
π ∑
∫ ∑
ª ∑∑ ºΩ ºº æø æ
¿ æ
¡ ææ ¬√ ¬
ƒ ¬
≈ ¬¬ ∆« ∆
» ∆∆ …  …
À …
Ã …
Õ …… Œœ ŒŒ –— –
“ –
” –– ‘’ ‘
÷ ‘
◊ ‘‘ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €
ﬁ €
ﬂ €€ ‡· ‡‡ ‚„ ‚
‰ ‚
Â ‚‚ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô Ì
 Ì
Ò ÌÌ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚
˛ ˚˚ ˇÄ ˇ
Å ˇˇ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá Ü
à Ü
â ÜÜ äã ä
å ää çé ç
è ç
ê ç
ë çç íì íí îï î
ñ îî óò óó ô
ö ôô õú õ
ù õ
û õõ ü† ü
° üü ¢£ ¢¢ §
• §§ ¶ß ¶
® ¶
© ¶¶ ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ ØØ ±≤ ±¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —
” ““ ‘
’ ‘‘ ÷◊ ?◊ é◊ è◊ Ùÿ πŸ 7⁄ ∑⁄ Ç⁄ ¬⁄ Á€ ∏€ å
‹ ß
‹ ®
‹ ´
‹ π
‹ Æ‹ Ç	› 	› "ﬁ ﬁ çﬁ Õﬁ Úﬂ -‡ 3	· ‚ ‚ π‚ ˘‚ û„ ;    
          !  # %$ '" (& *) ,+ .& 0/ 21 43 61 87 :1 <; >? A3 C7 E; G5 I@ JH L9 N@ OM Q= S@ TR V3 XW Z7 \[ ^; `_ b5 d@ ec g9 i@ jh l= n@ om q3 s7 u; w5 y@ z9 |@ }= @ Ä3 ÇÅ Ñ7 ÜÖ à; äâ å5 é@ èç ë9 ì@ îí ñ= ò@ ôó õ3 ùú ü7 °† £; •§ ß5 ©@ ™® ¨9 Æ@ Ø≠ ±= ≥@ ¥≤ ∂+ ∫r ºt æv ¿x ¬{ ƒ~ ∆ »«   ÃÀ Œ∑ –… —Õ “œ ‘” ÷ ÿ… ŸÕ ⁄◊ ‹€ ﬁ- ﬂ’ ·’ „‡ Â‚ Ê‰ Ë‡ È‚ Í’ ÏÎ ÓÌ Î ÒÁ ÛÔ ÙÚ ˆÁ ˜Ô ¯ı ˙π ˚ ˝… ˛Õ ˇ¸ ÅÇ ÑÉ Ü… áÕ àÖ äâ åç èé ë… íÕ ìê ï- óî ôñ öã úã ûõ †ù °ü £õ §ù •ã ß¶ ©® ´¶ ¨¢ Æ™ Ø≠ ±¢ ≤™ ≥π µ∞ ∑¥ ∏π ª∫ Ω… æÕ øº ¡¬ ƒ√ ∆… «Õ »≈  … ÃÕ œŒ —… “Õ ”– ’- ◊‘ Ÿ÷ ⁄À ‹À ﬁ€ ‡› ·ﬂ „€ ‰› ÂÀ ÁÊ ÈË ÎÊ Ï‚ ÓÍ ÔÌ Ò‚ ÚÍ Ûπ ı ˜Ù ¯˘ ˚˙ ˝… ˛Õ ˇ¸ Å3 ÉÇ Öı áÜ â€ ãà åÇ éä êç ë∞ ìí ïÇ óî ôñ ö úõ û‘ †ù °ü £Ç •¢ ß§ ®Ç ™© ¨7 Æ≠ ∞Ä ≤ä ≥≠ µ± ∑¥ ∏≠ ∫î ºπ ΩÄ ø¢ ¿≠ ¬æ ƒ¡ ≈≠ «∆ …; À  ÕÄ œä –  “Œ ‘— ’  ◊î Ÿ÷ ⁄Ä ‹¢ ›  ﬂ€ ·ﬁ ‚  ‰„ ÊÁ ÈË Î… ÏÕ ÌÍ ÔÓ ÒÚ ÙÛ ˆ… ˜Õ ¯ı ˙- ¸˘ ˛˚ ˇ Å ÉÄ ÖÇ ÜÑ àÄ âÇ ä åã éç êã ëá ìè îí ñá óè òπ öï úô ùû †ü ¢… £Õ §° ¶3 ®ß ™∞ ¨´ Æî ∞≠ ±Ø ≥ß µ≤ ∑¥ ∏ ∫π ºß æª ¿Ω ¡ï √¬ ≈˘ «ƒ »∆  ß Ã… ŒÀ œß —– ”7 ’‘ ◊¿ Ÿ≤ ⁄‘ ‹ÿ ﬁ€ ﬂ‘ ·ª „‡ ‰• Ê… Á‘ ÈÂ ÎË Ï‘ ÓÌ ; ÚÒ Ù¿ ˆ≤ ˜Ò ˘ı ˚¯ ¸Ò ˛ª Ä˝ Å• É… ÑÒ ÜÇ àÖ âÒ ãä çé ëè ì© ñÂ ò• ôó õÄ ú÷ ûï üù ° ¢π §˘ •£ ß‘ ®ï ™∑ ¨© ≠… ÆÕ Ø´ ±∞ ≥ µ© ∂… ∑Õ ∏¥ ∫- º© Ωπ øª ¿≤ ¬≤ ƒ¡ ∆√ «≈ …¡  √ À≤ ÕÃ œŒ —Ã “» ‘– ’” ◊» ÿ– Ÿπ €© ‹÷ ﬁ⁄ ﬂ ·© ‚… „Õ ‰‡ Ê5 Ëï ÈÁ Î† ÌÏ Ô¶ ÒÓ Ú Ù5 ˆï ˜Û ˘ı ˙ù ¸˚ ˛5 Äï Å˝ Éˇ Ñ÷ ÜÖ àπ äá ãâ ç5 èï êå íé ì5 ïï ñî ò9 öï õô ùö üÛ †9 ¢ï £û •° ¶9 ®ï ©˝ ´ß ¨Â Æå Ø9 ±ï ≤≠ ¥∞ µ9 ∑ï ∏∂ ∫= ºï Ωª øö ¡Û ¬= ƒï ≈¿ «√ »=  ï À˝ Õ… ŒÂ –å —= ”ï ‘œ ÷“ ◊= Ÿï ⁄ÿ ‹© ﬁí ﬂ› ·‘ „£ ‰˘ Êπ Á Èù Íï Ï÷ ÌÄ Ôó • ÚÂ Û∑ ˆÙ ˜… ¯Õ ˘ı ˚˙ ˝ ˇÙ Ä… ÅÕ Ç˛ Ñ- ÜÙ áÉ âÖ ä¸ å¸ éã êç ëè ìã îç ï¸ óñ ôò õñ úí ûö üù °í ¢ö £π •Ù ¶† ®§ © ´Ù ¨… ≠Õ Æ™ ∞è ≤5 ¥± µ≥ ∑Ë π∏ ª‚ Ω∫ æº ¿5 ¬± √ø ≈¡ ∆Î »«  5 Ã± Õ… œÀ –† “— ‘É ÷” ◊’ Ÿ5 €± ‹ÿ ﬁ⁄ ﬂ5 ·± ‚‡ ‰9 Ê± ÁÂ ÈÓ Îø Ï9 Ó± ÔÍ ÒÌ Ú9 Ù± ı… ˜Û ¯Ø ˙ÿ ˚9 ˝± ˛˘ Ä¸ Å9 É± ÑÇ Ü= à± âá ãÓ çø é= ê± ëå ìè î= ñ± ó… ôï öØ úÿ ù= ü± †õ ¢û £= •± ¶§ ®∑ ™@ ´… ¨Õ ≠© ØÆ ± ≥@ ¥… µÕ ∂≤ ∏- ∫@ ª∑ Ωπ æ∞ ¿∞ ¬ø ƒ¡ ≈√ «ø »¡ …∞ À  ÕÃ œ  –∆ “Œ ”— ’∆ ÷Œ ◊π Ÿ@ ⁄‘ ‹ÿ › ﬂ@ ‡… ·Õ ‚ﬁ ‰5 ÊÙ ÁÂ ÈÎ ÎÍ ÌÂ ÔÏ Ó Ú5 ÙÙ ıÒ ˜Û ¯† ˙˘ ¸5 ˛Ù ˇ˚ Å	˝ Ç	‘ Ñ	É	 Ü	∑ à	Ö	 â	5 ã	Ù å	á	 é	ä	 è	5 ë	Ù í	ê	 î	9 ñ	Ù ó	ï	 ô	Ò õ	Ò ú	9 û	Ù ü	ö	 °	ù	 ¢	9 §	Ù •	˚ ß	£	 ®	„ ™	á	 ´	9 ≠	Ù Æ	©	 ∞	¨	 ±	9 ≥	Ù ¥	≤	 ∂	= ∏	Ù π	∑	 ª	Ò Ω	Ò æ	= ¿	Ù ¡	º	 √	ø	 ƒ	= ∆	Ù «	˚ …	≈	  	„ Ã	á	 Õ	= œ	Ù –	À	 “	Œ	 ”	= ’	Ù ÷	‘	 ÿ	r ⁄	Å ‹	ç ﬁ	ñ ‡	∏ ‚	… „	Õ ‰	·	 Ê	Â	 Ë	 Í	Á	 Ï	È	 Ì	∏ Ô	… 	Õ Ò	Ó	 Û	Ú	 ı	 ˜	ˆ	 ˘	Ù	 ˚	¯	 ¸	∏ ˛	… ˇ	Õ Ä
˝	 Ç
Å
 Ñ
 Ü
Ö
 à
É
 ä
á
 ã
å
 é
ç
 ê
… ë
Õ í
è
 î
ì
 ñ
ç
 ò
… ô
Õ ö
ó
 ú
õ
 û
ç
 †
… °
Õ ¢
ü
 §
£
 ¶
®
 ™
 ≠
 Ø
 ±
 ≥
 ∂
 ∏
 ª
 Ω
π
 ø
ì ¬
•
 √
ë ≈
ù
 ∆
è »
ï
 …
ç À
É
 Ã
ã Œ
Ù	 œ
â —
Á	 “
·
 ‘
˛ ÷
€	 ◊
ß Ÿ
Ÿ	 ⁄
Ç ‹
ﬂ	 ›
Î ﬂ
›	 ‡
”
 ‚
5 ‰
”
 Â
„
 Á
ÿ
 È
Ë
 Î
’
 Ï
5 Ó
”
 Ô
Í
 Ò
Ì
 Ú
Ë
 Ù
Ê
 ı
Û
 ˜
„
 ¯
–
 ˙
Ë
 ¸
˘
 ˝
∏ ˇ
”
 Ä… ÅÕ Ç˚
 Ñ˛
 ÖÕ
 áË
 âÜ ä∏ å”
 ç… éÕ èà ëã í 
 îË
 ñì ó∏ ô”
 ö… õÕ úï ûò ü5 °·
 ¢† §ﬁ
 ¶• ®Í
 ©€
 ™5 ¨·
 ≠ß Ø´ ∞«
 ≤• ¥˚
 µ± ∂ƒ
 ∏• ∫à ª∑ º¡
 æ• ¿ï ¡Ω ¬”
 ƒ5 ∆√ «≈ …5 À√ Ã  Œ5 –√ —œ ”∏ ’√ ÷… ◊Õ ÿ‘ ⁄∏ ‹√ ›… ﬁÕ ﬂ€ ·∏ „√ ‰… ÂÕ Ê‚ Ë» ÍÈ ÏÍ
 ÌÕ ÓÎ   ÒÈ Û˚
 ÙŸ ıÈ ˜à ¯‡ ˘È ˚ï ¸Á ˝• ˇÛ
 Ä£ ÅÈ ÉÛ
 Ñ“ Ö·
 áæ
 à≥ äπ åø éÚ êˆ í˙ îÜ ñ˚
 ò∫
 ôà õµ
 úï û∑
 ü≥ °º
 ¢π §ˆ	 •ø ßÖ
 ®≤
 ´º
 ¨∞
 Æ∫
 ØÆ
 ±∑
 ≤¨
 ¥µ
 µ´
 ∑π
 ∏•
 ∫ì ªù
 Ωë æï
 ¿è ¡É
 √ç ƒÙ	 ∆ã «Á	 …â  ›	 ÃÎ Õﬂ	 œÇ –Ÿ	 “ß ”€	 ’˛ ÷ ÿ≥ ⁄∞ ‹∂ ﬁ5 ‡› ·ﬂ „» Â◊ Ê≈ ËŸ È¬ Î€ Ï— ÓÌ ‘ Ò5 Û› ÙÔ ˆÚ ˜Ì ˘‚ ˙¯ ¸ﬂ ˝» ˇÌ Å˛ ÇÄ Ñ≠ Ö∏ á› à… âÕ äÄ åÜ ç≈ èÌ ëé íê î≥ ï∏ ó› ò… ôÕ öê úñ ù¬ üÌ °û ¢† §∞ •∏ ß› ®… ©Õ ™† ¨¶ ≠Æ ∞5 ≤Ø ≥± µø ∑È	 ∏À ∫π ºÔ ΩŒ æ5 ¿Ø ¡ª √ø ƒπ ∆¯ «¥ »≈  ± Àø Õπ œÄ –Ã —Œ ”™ ‘º ÷π ÿê Ÿ’ ⁄◊ ‹ˆ	 ›π ﬂπ ·† ‚ﬁ „‡ ÂÖ
 Êª ËÁ ÍŒ Î∏ ÌØ Ó… ÔÕ È ÚÏ ÛÁ ı◊ ˆ∏ ¯Ø ˘… ˙Õ ˚Ù ˝˜ ˛Á Ä‡ Å∏ ÉØ Ñ… ÖÕ Üˇ àÇ ât ãÖ ç¥ èπ ëv ìâ ï— ó÷ ô∏ õ… úÕ ùö üû ° £¢ •† ß§ ®∏ ™… ´Õ ¨© Æ≠ ∞ ≤± ¥Ø ∂≥ ∑ç
 π… ∫Õ ª∏ Ωº øç
 ¡… ¬Õ √¿ ≈ƒ « …» À ÕÃ œ®
 —∂ ”“ ÷∆ ◊– ŸØ ⁄Œ ‹æ ›Ã ﬂ† ‡˚ ‚∫ ‰ñ Â¡ Áò Ëé Íí Îò Ìî Ó’ é Ò‹ Ûê Ù© ˆä ˜≥ ˘å ˙· ¸· ˛9 Ä· Åˇ Éı Ö¯ áÑ à9 ä· ãÜ çâ éÑ êÇ ëè ìˇ îﬁ ñÑ òï ô∏ õ· ú… ùÕ ûó †ö °9 £˚ §¢ ¶Ô ®ß ™Ü ´Ú ¨9 Æ˚ Ø© ±≠ ≤ß ¥è µ• ∂€ ∏ß ∫ó ª∑ º9 æ˝ øΩ ¡9 √˝ ƒ¬ ∆9 »˝ …« À∏ Õ˝ Œ… œÕ –Ã “¿ ‘” ÷Ü ◊≈ ÿ’ ⁄¬ €” ›è ﬁ  ﬂ” ·ó ‚— „= Â· Ê‰ ËÈ ÍÏ ÏÈ Ì= Ô· Î ÚÓ ÛÈ ıÁ ˆÙ ¯‰ ˘ÿ ˚È ˝˙ ˛∏ Ä· Å… ÇÕ É¸ Öˇ Ü= à˚ âá ã„ çå èÎ êÊ ë= ì˚ îé ñí óå ôÙ öä õ’ ùå ü¸ †ú °= £˝ §¢ ¶= ®˝ ©ß ´= ≠˝ Æ¨ ∞∏ ≤˝ ≥… ¥Õ µ± ∑• π∏ ªÎ º™ Ω∫ øß ¿∏ ¬Ù √Ø ƒ∏ ∆¸ «∂ »˚  “ Àπ Õ‡ œû —≈ ”… ’ó ◊» ÿπ ⁄¢ €¸ ›Ã ﬁû ‡± ·“ ‰∆ Â– ÁØ ËŒ Íæ ÎÃ Ì† Ó≥ å Ò© Ûä Ù‹ ˆê ˜’ ˘é ˙ò ¸î ˝é ˇí Ä¡ Çò É∫ Öñ Ü9 à› âá ãÏ ç  éÚ êÔ íè ì9 ï› ñë òî ôè õä úö ûá üÏ °è £† §∏ ¶› ß… ®Õ ©¢ ´• ¨9 ÆØ Ø≠ ±¯ ≥≤ µë ∂ı ∑9 πØ ∫¥ º∏ Ω≤ øö ¿∞ ¡æ √≠ ƒÈ ∆≤ »¢ …≈  = Ã› ÕÀ œÊ —Œ “˛ ‘˚ ÷” ◊= Ÿ› ⁄’ ‹ÿ ›” ﬂŒ ‡ﬁ ‚À „Ê Â” Á‰ Ë∏ Í› Î… ÏÕ ÌÊ ÔÈ = ÚØ ÛÒ ıÑ ˜ˆ ˘’ ˙Å ˚= ˝Ø ˛¯ Ä¸ Åˆ Éﬁ ÑÙ ÖÇ áÒ à„ äˆ åÊ çâ é« ê¥ ë∏ ìØ î… ïÕ ñè òí ôã õ¯ ú∏ ûØ ü… †Õ °ö £ù §Ú ¶• ®Ï ™© ¨™ ≠Ü Øß ±© ≤Æ ≥∞ µ≠ ∂∞ ∏Ü π˜ ª∫ Ωˆ	 æñ ¿ß ¬∫ √ø ƒ¡ ∆≥ «¡ …ñ  Ç ÃÀ ŒÖ
 œ¶ —ß ”À ‘– ’“ ◊∞ ÿ“ ⁄¶ €è ›¢ ﬁ• ‡î ‚· ‰„ Êè Áﬂ ËÂ Í» ÎÂ Ì• Óö ± ÒÈ Ûÿ ıÙ ˜ˆ ˘ö ˙Ú ˚¯ ˝Ã ˛¯ ÄÈ ÅÇ Ñß
 Ü¯ â¶ äö åà çÂ èÜ êè íé ì“ ïÊ ñÀ òî ô¡ õ‘ ú∫ ûö ü∞ °¬ ¢© §† •Ö ß≠ ®5 ™¶ ´© ≠5 Ø¶ ∞Æ ≤¨ ¥± ∂∏ ∏¶ π… ∫Õ ª∑ Ω≥ ø† ¿º ¡µ √£ ƒæ ≈¬ «∑ »∏  ¶ À… ÃÕ Õ… œ≥ —ö “Œ ”µ ’ù ÷– ◊‘ Ÿ… ⁄∏ ‹¶ ›… ﬁÕ ﬂ€ ·≥ „î ‰‡ Âµ Áó Ë‚ ÈÊ Î€ Ï∏ Ó¶ Ô… Õ ÒÌ Û9 ı¶ ˆÙ ¯˜ ˙˘ ¸é ˝Ú ˛9 Ä¶ Åˇ ÉÇ ÖÑ áë à˚ âÜ ãÌ å∏ é¶ è… êÕ ëç ì= ï¶ ñî òó öô úà ùí û= †¶ °ü £¢ •§ ßã ®õ ©¶ ´ç ¨¶ Æ¶ ∞Ø ≤† ¥™ µ¬ ∑≠ ∏ö ∫ˆ	 ªî ΩÖ
 æé ¿¢ ¡Ü √» ƒà ∆± «¶ …Ã  ‘ Ã≥ ÕÊ œ∞ – ” ’ “ ê ‚ê í©
 ´
©
 µ
î ï¥
 ™¿
 ¡
‡ ‚‡ ï– „– “ï óï ¡
É ÖÉ “‘ ’© ™á à‘ ÷‘ ’± à± ≥‚ „— “ ÂÂ ‰‰ ÷ ÁÁ ÊÊ˚ ÊÊ ˚û ÊÊ û¡ ÊÊ ¡œ ÊÊ œ¡ ÊÊ ¡¥ ÊÊ ¥å ÊÊ åá	 ÊÊ á	À	 ÊÊ À	∞ ÊÊ ∞æ ÊÊ æÿ ÊÊ ÿ˘ ÊÊ ˘¬ ÊÊ ¬– ÊÊ –Ü ÊÊ Ü‹ ÊÊ ‹Ê ÊÊ Ê€ ÊÊ € ÂÂ í ÊÊ í© ÊÊ ©ä ÊÊ äø ÊÊ øø ÊÊ ø’ ÊÊ ’‘ ÁÁ ‘Ø ÊÊ Ø ÊÊ ± ÊÊ ±≠ ÊÊ ≠◊ ÊÊ ◊û ÊÊ ûù ÊÊ ù˛ ÊÊ ˛¯ ÊÊ ¯Â ÊÊ Â› ÊÊ ›º ÊÊ ºé ÊÊ éõ ÊÊ õß ÊÊ ß˘ ÊÊ ˘â ÊÊ â« ÊÊ «ı ÊÊ ı	 ‰‰ 	¿ ÊÊ ¿ã ÊÊ ãÇ ÊÊ Ç≈ ÊÊ ≈∫ ÊÊ ∫æ ÊÊ æ˙ ÊÊ ˙Í ÊÊ ÍÇ ÊÊ Ç√ ÊÊ √ ‰‰ ˆ ÊÊ ˆπ ÊÊ ππ ÊÊ π€ ÊÊ €ò ÊÊ òü ÊÊ üõ ÊÊ õ¡ ÊÊ ¡« ÊÊ «Ç ÊÊ Çã ÊÊ ã“ ÊÊ “Ç ÊÊ Ç‘ ÊÊ ‘˚ ÊÊ ˚Â ÊÊ Â ÂÂ º	 ÊÊ º	‚ ÊÊ ‚ö	 ÊÊ ö	π ÊÊ π‡ ÊÊ ‡ª ÊÊ ª‡ ÊÊ ‡æ ÊÊ æÓ ÊÊ ÓÚ ÊÊ ÚÄ ÊÊ ÄŒ ÊÊ Œ≥ ÊÊ ≥‡ ÊÊ ‡¯ ÊÊ ¯¡ ÊÊ ¡‚ ÊÊ ‚¶ ÊÊ ¶“ ÁÁ “’ ÊÊ ’ç ÊÊ ç≥ ÊÊ ≥Î ÊÊ Îõ ÊÊ õ©	 ÊÊ ©	∆ ÊÊ ∆Œ ÊÊ Œ≈ ÊÊ ≈Ë Ë Ë 	Ë ?
È Ü
È õ
È ´
È ¬
È Ï
È Ö
È ∏
È —
È Í
È É	
Í í
Í π
Í ˚
Í «
Í ˘Î ªÎ ΩÎ øÎ ¡Î √Î ≈
Î ‡
Î ‚
Î Î
Î Ì
Î Ô
Î õ
Î ù
Î ¶
Î ®
Î ™
Î €
Î ›
Î Ê
Î Ë
Î Í
Î í
Î Ä
Î Ç
Î ã
Î ç
Î è
Î π
Î ¡
Î √
Î Ã
Î Œ
Î –
Î ˚
Î ã
Î ç
Î ñ
Î ò
Î ö
Î «
Î ø
Î ¡
Î  
Î Ã
Î Œ
Î ˘Î Ë
Î ÌÎ ÁÎ ÑÎ ÈÎ èÎ ”
Ï Å
Ï Ö
Ï â
Ï ç
Ï í
Ï ó
Ï §
Ï ¡
Ï ﬁ
Ï ˚
Ï ô
Ï À
Ï Ë
Ï Ö
Ï ï
Ï é
Ï ∞
Ï “
Ï ⁄
Ï ¸
Ï û
Ï ä	
Ï ¨	
Ï Œ	
Ï Ì

Ï †
Ï Ú
Ï ±
Ï ö
Ï ¢
Ï ∏
Ï »
Ï â
Ï ö
Ï ¢
Ï Ã
Ï Ó
Ï á
Ï î
Ï •
Ï ≠
Ï ÿ
Ï Ò
Ï í
Ï ©
Ï Ì
Ï Ù
Ï î
Ì ú
Ì †
Ì §
Ì ®
Ì ≠
Ì ≤
Ì ©
Ì ∆
Ì „
Ì –
Ì Ì
Ì ä
Ì î
Ì ∂
Ì ÿ
Ì ‡
Ì Ç
Ì §
Ì ê	
Ì ≤	
Ì ‘	
Ì „

Ì ﬂ
Ì ©
Ì ±
Ì ¿
Ì Ã
Ì ˇ
Ì ‰
Ì ˇ
Ì ±
Ì á
Ì À
Ì È
Ì ù
Ì Æ
Ì ˇ
Ì ç
Ì ü
Ó «
Ó …
Ó À
Ó ÕÔ Ô 	Ô “Ô ‘
 ä
 ±
 €
 Ø
 ÿ
 Ç
 
 û
 œ
 º
 Í
 õ
 Ó
 ö	
 À	
Ò ≠Ú 
Û ¬
Û Õ
Û ˘
Ù ®
	ı )	ˆ  	ˆ $
ˆ è
ˆ Æ	˜ 	˜ 	˜ W	˜ [	˜ _	˜ c	˜ h	˜ m
˜ ñ
˜ ¥
˜ ç
˜ ¥
˜ —
˜ ¥
˜ €
˜ ¯
˜ ©
˜ ı
˜ °
˜ √
˜ ¡
˜ Ì
˜ è
˜ Û
˜ ù	
˜ ø	
˜ Ó	
˜ ˆ	
˜ ó

˜ ¨

˜ µ

˜ ·

˜ ã
˜  
˜ €
˜ ñ
˜ ˜
˜ ˚
˜ ¬
˜ ß
˜ …
¯ ª
¯ ˝
¯ …
˘ Ç
˙ ’
˙ ã
˙ À
˙ 
˙ ≤
˙ ¸
˙ ∞
˚ å

¸ ü
¸ æ
¸ Œ
¸ ∆
¸ Â
¸ ı
¸ â
¸ ≠
¸ ¿
¸ ’
¸ ˘
¸ å
¸ á	
¸ ©	
¸ º	
˝ ß

˛ î
˛ ˚
ˇ é
Ä ´

Ä π

Å ‚
Å ù
Å ›
Å Ç
Å √
Å ç
Å ¡
Ç Ç
Ç ≠
Ç  É àÉ ùÉ ≠É ƒÉ ÓÉ áÉ ∫É ”É ÏÉ Ö	É •É ÈÉ πÉ ßÉ ”É åÉ ∏É ≤É ˆÉ ßÉ „É ˆÉ ≥É µÉ ˘É ÑÉ ôÉ §	Ñ r	Ñ t	Ñ v	Ñ x	Ñ {	Ñ ~
Ñ ÷
Ñ Ù
Ñ ñ
Ñ π
Ñ ÷
Ñ Ω
Ñ ‡
Ñ ˝
Ñ ˇ
Ñ ß
Ñ …
Ñ À
Ñ Û
Ñ ï
Ñ ˝
Ñ £	
Ñ ≈	
Ñ ˝	
Ñ Ö

Ñ ü

Ñ Æ

Ñ ∑

Ñ ò
Ñ ´
Ñ √
Ñ œ
Ñ ‚
Ñ ¶
Ñ ø
Ñ Ç
Ñ ˝
Ñ ≠
Ñ «
Ñ í
Ñ ¨
Ñ ∏
Ñ ¸
Ñ €
Ö ‡
Ö õ
Ö €
Ö Ä
Ö ¡
Ö ã
Ö ø
Ü Ç
Ü ç
Ü πá Bá Dá Fá Ká Pá Uá Yá ]á aá fá ká pá Éá áá ãá êá ïá öá ûá ¢á ¶á ´á ∞á µá Ñá Øá Ãá ©á ÷á Ûá „á Öá ßá ì	á µ	á ◊	
à ¢
à ≤
à …
à Û
à å
à ø
à ÿ
à Ò
â Á
â Ú
â ûä ´ä »ä Âä “ä Ôä åä Íä óä úä πä æä €ä ∂ä Ëä ää Ëä ò	ä ∫		ã H	ã M	ã R
ã œ
ã ◊
ã ¸
ã Ö
ã ê
ã º
ã ≈
ã –
ã ¸
ã Í
ã ı
ã °
ã Á
ã ô
ã ª
ã ≥
ã Â
ã á
ã Â
ã ï	
ã ∑	
ã ·	
ã Ó	
ã ˆ	
ã ˝	
ã Ö

ã è

ã ó

ã ü

ã ¨

ã Æ

ã ∞

ã ∞

ã ≤

ã ≤

ã µ

ã ∑

ã ∫

ã ∫

ã º

ã º

ã ”

ã ˛

ã ≈
ã ‘
ã Ü
ã Ï
ã ö
ã ¢
ã ©
ã ±
ã ∏
ã ¿
ã »
ã Ã
ã ·
ã Ω
ã ¢
ã ∑
ã Ø
å ß
å ‘
å Ò	ç /"	
z_solve"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*ä
npb-SP-z_solve.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

wgsize
>

transfer_bytes
∏˘ˆ5

wgsize_log1p
°YîA
 
transfer_bytes_log1p
°YîA

devmap_label
