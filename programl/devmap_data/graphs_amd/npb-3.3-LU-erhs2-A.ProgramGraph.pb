

[external]
KcallBC
A
	full_text4
2
0%7 = tail call i64 @_Z13get_global_idj(i32 1) #3
,addB%
#
	full_text

%8 = add i64 %7, 1
"i64B

	full_text


i64 %7
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
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #3
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
2addB+
)
	full_text

%13 = add nsw i32 %5, -1
5icmpB-
+
	full_text

%14 = icmp sgt i32 %13, %9
#i32B

	full_text
	
i32 %13
"i32B

	full_text


i32 %9
2addB+
)
	full_text

%15 = add nsw i32 %4, -1
6icmpB.
,
	full_text

%16 = icmp sgt i32 %15, %12
#i32B

	full_text
	
i32 %15
#i32B

	full_text
	
i32 %12
/andB(
&
	full_text

%17 = and i1 %14, %16
!i1B

	full_text


i1 %14
!i1B

	full_text


i1 %16
9brB3
1
	full_text$
"
 br i1 %17, label %18, label %639
!i1B

	full_text


i1 %17
Wbitcast8BJ
H
	full_text;
9
7%19 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%20 = bitcast double* %1 to [65 x [65 x [5 x double]]]*
/shl8B&
$
	full_text

%21 = shl i32 %9, 6
$i328B

	full_text


i32 %9
2add8B)
'
	full_text

%22 = add i32 %21, %12
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %12
2mul8B)
'
	full_text

%23 = mul i32 %22, 320
%i328B

	full_text
	
i32 %22
5add8B,
*
	full_text

%24 = add i32 %23, -20800
%i328B

	full_text
	
i32 %23
6sext8B,
*
	full_text

%25 = sext i32 %24 to i64
%i328B

	full_text
	
i32 %24
^getelementptr8BK
I
	full_text<
:
8%26 = getelementptr inbounds double, double* %2, i64 %25
%i648B

	full_text
	
i64 %25
Jbitcast8B=
;
	full_text.
,
*%27 = bitcast double* %26 to [5 x double]*
-double*8B

	full_text

double* %26
5icmp8B+
)
	full_text

%28 = icmp sgt i32 %3, 0
;br8B3
1
	full_text$
"
 br i1 %28, label %29, label %198
#i18B

	full_text


i1 %28
0shl8B'
%
	full_text

%30 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%31 = ashr exact i64 %30, 32
%i648B

	full_text
	
i64 %30
1shl8B(
&
	full_text

%32 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%33 = ashr exact i64 %32, 32
%i648B

	full_text
	
i64 %32
5zext8B+
)
	full_text

%34 = zext i32 %3 to i64
'br8B

	full_text

br label %35
Bphi8B9
7
	full_text*
(
&%36 = phi i64 [ 0, %29 ], [ %73, %35 ]
%i648B

	full_text
	
i64 %73
¢getelementptr8Bé
ã
	full_text~
|
z%37 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %31, i64 %33, i64 %36, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %36
Abitcast8B4
2
	full_text%
#
!%38 = bitcast double* %37 to i64*
-double*8B

	full_text

double* %37
Hload8B>
<
	full_text/
-
+%39 = load i64, i64* %38, align 8, !tbaa !8
'i64*8B

	full_text


i64* %38
kgetelementptr8BX
V
	full_textI
G
E%40 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %36
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %36
Gbitcast8B:
8
	full_text+
)
'%41 = bitcast [5 x double]* %40 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %40
Hstore8B=
;
	full_text.
,
*store i64 %39, i64* %41, align 8, !tbaa !8
%i648B

	full_text
	
i64 %39
'i64*8B

	full_text


i64* %41
Nload8BD
B
	full_text5
3
1%42 = load double, double* %37, align 8, !tbaa !8
-double*8B

	full_text

double* %37
¢getelementptr8Bé
ã
	full_text~
|
z%43 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %31, i64 %33, i64 %36, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %36
Nload8BD
B
	full_text5
3
1%44 = load double, double* %43, align 8, !tbaa !8
-double*8B

	full_text

double* %43
7fdiv8B-
+
	full_text

%45 = fdiv double %42, %44
+double8B

	full_text


double %42
+double8B

	full_text


double %44
¢getelementptr8Bé
ã
	full_text~
|
z%46 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %31, i64 %33, i64 %36, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %36
Nload8BD
B
	full_text5
3
1%47 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
7fmul8B-
+
	full_text

%48 = fmul double %47, %47
+double8B

	full_text


double %47
+double8B

	full_text


double %47
icall8B_
]
	full_textP
N
L%49 = tail call double @llvm.fmuladd.f64(double %42, double %42, double %48)
+double8B

	full_text


double %42
+double8B

	full_text


double %42
+double8B

	full_text


double %48
¢getelementptr8Bé
ã
	full_text~
|
z%50 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %31, i64 %33, i64 %36, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %36
Nload8BD
B
	full_text5
3
1%51 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
icall8B_
]
	full_textP
N
L%52 = tail call double @llvm.fmuladd.f64(double %51, double %51, double %49)
+double8B

	full_text


double %51
+double8B

	full_text


double %51
+double8B

	full_text


double %49
@fmul8B6
4
	full_text'
%
#%53 = fmul double %52, 5.000000e-01
+double8B

	full_text


double %52
7fdiv8B-
+
	full_text

%54 = fdiv double %53, %44
+double8B

	full_text


double %53
+double8B

	full_text


double %44
¢getelementptr8Bé
ã
	full_text~
|
z%55 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %31, i64 %33, i64 %36, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
%i648B

	full_text
	
i64 %31
%i648B

	full_text
	
i64 %33
%i648B

	full_text
	
i64 %36
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
7fsub8B-
+
	full_text

%57 = fsub double %56, %54
+double8B

	full_text


double %56
+double8B

	full_text


double %54
@fmul8B6
4
	full_text'
%
#%58 = fmul double %57, 4.000000e-01
+double8B

	full_text


double %57
icall8B_
]
	full_textP
N
L%59 = tail call double @llvm.fmuladd.f64(double %42, double %45, double %58)
+double8B

	full_text


double %42
+double8B

	full_text


double %45
+double8B

	full_text


double %58
rgetelementptr8B_
]
	full_textP
N
L%60 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %36, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %36
Nstore8BC
A
	full_text4
2
0store double %59, double* %60, align 8, !tbaa !8
+double8B

	full_text


double %59
-double*8B

	full_text

double* %60
Nload8BD
B
	full_text5
3
1%61 = load double, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
7fmul8B-
+
	full_text

%62 = fmul double %45, %61
+double8B

	full_text


double %45
+double8B

	full_text


double %61
rgetelementptr8B_
]
	full_textP
N
L%63 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %36, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %36
Nstore8BC
A
	full_text4
2
0store double %62, double* %63, align 8, !tbaa !8
+double8B

	full_text


double %62
-double*8B

	full_text

double* %63
Nload8BD
B
	full_text5
3
1%64 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
7fmul8B-
+
	full_text

%65 = fmul double %45, %64
+double8B

	full_text


double %45
+double8B

	full_text


double %64
rgetelementptr8B_
]
	full_textP
N
L%66 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %36, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %36
Nstore8BC
A
	full_text4
2
0store double %65, double* %66, align 8, !tbaa !8
+double8B

	full_text


double %65
-double*8B

	full_text

double* %66
Nload8BD
B
	full_text5
3
1%67 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
@fmul8B6
4
	full_text'
%
#%68 = fmul double %54, 4.000000e-01
+double8B

	full_text


double %54
Afsub8B7
5
	full_text(
&
$%69 = fsub double -0.000000e+00, %68
+double8B

	full_text


double %68
rcall8Bh
f
	full_textY
W
U%70 = tail call double @llvm.fmuladd.f64(double %67, double 1.400000e+00, double %69)
+double8B

	full_text


double %67
+double8B

	full_text


double %69
7fmul8B-
+
	full_text

%71 = fmul double %45, %70
+double8B

	full_text


double %45
+double8B

	full_text


double %70
rgetelementptr8B_
]
	full_textP
N
L%72 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %36, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %36
Nstore8BC
A
	full_text4
2
0store double %71, double* %72, align 8, !tbaa !8
+double8B

	full_text


double %71
-double*8B

	full_text

double* %72
8add8B/
-
	full_text 

%73 = add nuw nsw i64 %36, 1
%i648B

	full_text
	
i64 %36
7icmp8B-
+
	full_text

%74 = icmp eq i64 %73, %34
%i648B

	full_text
	
i64 %73
%i648B

	full_text
	
i64 %34
:br8B2
0
	full_text#
!
br i1 %74, label %75, label %35
#i18B

	full_text


i1 %74
4add8B+
)
	full_text

%76 = add nsw i32 %3, -1
5icmp8B+
)
	full_text

%77 = icmp sgt i32 %3, 2
;br8B3
1
	full_text$
"
 br i1 %77, label %78, label %129
#i18B

	full_text


i1 %77
0shl8B'
%
	full_text

%79 = shl i64 %8, 32
$i648B

	full_text


i64 %8
9ashr8B/
-
	full_text 

%80 = ashr exact i64 %79, 32
%i648B

	full_text
	
i64 %79
1shl8B(
&
	full_text

%81 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%82 = ashr exact i64 %81, 32
%i648B

	full_text
	
i64 %81
6zext8B,
*
	full_text

%83 = zext i32 %76 to i64
%i328B

	full_text
	
i32 %76
'br8B

	full_text

br label %84
Bphi8B9
7
	full_text*
(
&%85 = phi i64 [ 1, %78 ], [ %86, %84 ]
%i648B

	full_text
	
i64 %86
8add8B/
-
	full_text 

%86 = add nuw nsw i64 %85, 1
%i648B

	full_text
	
i64 %85
5add8B,
*
	full_text

%87 = add nsw i64 %85, -1
%i648B

	full_text
	
i64 %85
¢getelementptr8Bé
ã
	full_text~
|
z%88 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %80, i64 %82, i64 %85, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %80
%i648B

	full_text
	
i64 %82
%i648B

	full_text
	
i64 %85
Nload8BD
B
	full_text5
3
1%89 = load double, double* %88, align 8, !tbaa !8
-double*8B

	full_text

double* %88
rgetelementptr8B_
]
	full_textP
N
L%90 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %86, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %86
Nload8BD
B
	full_text5
3
1%91 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
rgetelementptr8B_
]
	full_textP
N
L%92 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %87, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %87
Nload8BD
B
	full_text5
3
1%93 = load double, double* %92, align 8, !tbaa !8
-double*8B

	full_text

double* %92
7fsub8B-
+
	full_text

%94 = fsub double %91, %93
+double8B

	full_text


double %91
+double8B

	full_text


double %93
scall8Bi
g
	full_textZ
X
V%95 = tail call double @llvm.fmuladd.f64(double %94, double -3.150000e+01, double %89)
+double8B

	full_text


double %94
+double8B

	full_text


double %89
Nstore8BC
A
	full_text4
2
0store double %95, double* %88, align 8, !tbaa !8
+double8B

	full_text


double %95
-double*8B

	full_text

double* %88
¢getelementptr8Bé
ã
	full_text~
|
z%96 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %80, i64 %82, i64 %85, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %80
%i648B

	full_text
	
i64 %82
%i648B

	full_text
	
i64 %85
Nload8BD
B
	full_text5
3
1%97 = load double, double* %96, align 8, !tbaa !8
-double*8B

	full_text

double* %96
rgetelementptr8B_
]
	full_textP
N
L%98 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %86, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %86
Nload8BD
B
	full_text5
3
1%99 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
sgetelementptr8B`
^
	full_textQ
O
M%100 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %87, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %87
Pload8BF
D
	full_text7
5
3%101 = load double, double* %100, align 8, !tbaa !8
.double*8B

	full_text

double* %100
9fsub8B/
-
	full_text 

%102 = fsub double %99, %101
+double8B

	full_text


double %99
,double8B

	full_text

double %101
ucall8Bk
i
	full_text\
Z
X%103 = tail call double @llvm.fmuladd.f64(double %102, double -3.150000e+01, double %97)
,double8B

	full_text

double %102
+double8B

	full_text


double %97
Ostore8BD
B
	full_text5
3
1store double %103, double* %96, align 8, !tbaa !8
,double8B

	full_text

double %103
-double*8B

	full_text

double* %96
£getelementptr8Bè
å
	full_text
}
{%104 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %80, i64 %82, i64 %85, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %80
%i648B

	full_text
	
i64 %82
%i648B

	full_text
	
i64 %85
Pload8BF
D
	full_text7
5
3%105 = load double, double* %104, align 8, !tbaa !8
.double*8B

	full_text

double* %104
sgetelementptr8B`
^
	full_textQ
O
M%106 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %86, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%107 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
sgetelementptr8B`
^
	full_textQ
O
M%108 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %87, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %87
Pload8BF
D
	full_text7
5
3%109 = load double, double* %108, align 8, !tbaa !8
.double*8B

	full_text

double* %108
:fsub8B0
.
	full_text!

%110 = fsub double %107, %109
,double8B

	full_text

double %107
,double8B

	full_text

double %109
vcall8Bl
j
	full_text]
[
Y%111 = tail call double @llvm.fmuladd.f64(double %110, double -3.150000e+01, double %105)
,double8B

	full_text

double %110
,double8B

	full_text

double %105
Pstore8BE
C
	full_text6
4
2store double %111, double* %104, align 8, !tbaa !8
,double8B

	full_text

double %111
.double*8B

	full_text

double* %104
£getelementptr8Bè
å
	full_text
}
{%112 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %80, i64 %82, i64 %85, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %80
%i648B

	full_text
	
i64 %82
%i648B

	full_text
	
i64 %85
Pload8BF
D
	full_text7
5
3%113 = load double, double* %112, align 8, !tbaa !8
.double*8B

	full_text

double* %112
sgetelementptr8B`
^
	full_textQ
O
M%114 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %86, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%115 = load double, double* %114, align 8, !tbaa !8
.double*8B

	full_text

double* %114
sgetelementptr8B`
^
	full_textQ
O
M%116 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %87, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %87
Pload8BF
D
	full_text7
5
3%117 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
:fsub8B0
.
	full_text!

%118 = fsub double %115, %117
,double8B

	full_text

double %115
,double8B

	full_text

double %117
vcall8Bl
j
	full_text]
[
Y%119 = tail call double @llvm.fmuladd.f64(double %118, double -3.150000e+01, double %113)
,double8B

	full_text

double %118
,double8B

	full_text

double %113
Pstore8BE
C
	full_text6
4
2store double %119, double* %112, align 8, !tbaa !8
,double8B

	full_text

double %119
.double*8B

	full_text

double* %112
£getelementptr8Bè
å
	full_text
}
{%120 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %80, i64 %82, i64 %85, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
%i648B

	full_text
	
i64 %80
%i648B

	full_text
	
i64 %82
%i648B

	full_text
	
i64 %85
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
sgetelementptr8B`
^
	full_textQ
O
M%122 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %86, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %86
Pload8BF
D
	full_text7
5
3%123 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
sgetelementptr8B`
^
	full_textQ
O
M%124 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %87, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
%i648B

	full_text
	
i64 %87
Pload8BF
D
	full_text7
5
3%125 = load double, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
:fsub8B0
.
	full_text!

%126 = fsub double %123, %125
,double8B

	full_text

double %123
,double8B

	full_text

double %125
vcall8Bl
j
	full_text]
[
Y%127 = tail call double @llvm.fmuladd.f64(double %126, double -3.150000e+01, double %121)
,double8B

	full_text

double %126
,double8B

	full_text

double %121
Pstore8BE
C
	full_text6
4
2store double %127, double* %120, align 8, !tbaa !8
,double8B

	full_text

double %127
.double*8B

	full_text

double* %120
8icmp8B.
,
	full_text

%128 = icmp eq i64 %86, %83
%i648B

	full_text
	
i64 %86
%i648B

	full_text
	
i64 %83
<br8B4
2
	full_text%
#
!br i1 %128, label %129, label %84
$i18B

	full_text
	
i1 %128
Fphi8B=
;
	full_text.
,
*%130 = phi i1 [ false, %75 ], [ %77, %84 ]
#i18B

	full_text


i1 %77
6icmp8B,
*
	full_text

%131 = icmp sgt i32 %3, 1
=br8B5
3
	full_text&
$
"br i1 %131, label %132, label %197
$i18B

	full_text
	
i1 %131
1shl8B(
&
	full_text

%133 = shl i64 %8, 32
$i648B

	full_text


i64 %8
;ashr8B1
/
	full_text"
 
%134 = ashr exact i64 %133, 32
&i648B

	full_text


i64 %133
2shl8B)
'
	full_text

%135 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
;ashr8B1
/
	full_text"
 
%136 = ashr exact i64 %135, 32
&i648B

	full_text


i64 %135
6zext8B,
*
	full_text

%137 = zext i32 %3 to i64
(br8B 

	full_text

br label %138
Fphi8	B=
;
	full_text.
,
*%139 = phi i64 [ 1, %132 ], [ %195, %138 ]
&i648	B

	full_text


i64 %195
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%140 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %139, i64 0
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %139
Pload8	BF
D
	full_text7
5
3%141 = load double, double* %140, align 8, !tbaa !8
.double*8	B

	full_text

double* %140
Bfdiv8	B8
6
	full_text)
'
%%142 = fdiv double 1.000000e+00, %141
,double8	B

	full_text

double %141
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%143 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %139, i64 1
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %139
Pload8	BF
D
	full_text7
5
3%144 = load double, double* %143, align 8, !tbaa !8
.double*8	B

	full_text

double* %143
:fmul8	B0
.
	full_text!

%145 = fmul double %142, %144
,double8	B

	full_text

double %142
,double8	B

	full_text

double %144
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%146 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %139, i64 2
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %139
Pload8	BF
D
	full_text7
5
3%147 = load double, double* %146, align 8, !tbaa !8
.double*8	B

	full_text

double* %146
:fmul8	B0
.
	full_text!

%148 = fmul double %142, %147
,double8	B

	full_text

double %142
,double8	B

	full_text

double %147
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%149 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %139, i64 3
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %139
Pload8	BF
D
	full_text7
5
3%150 = load double, double* %149, align 8, !tbaa !8
.double*8	B

	full_text

double* %149
:fmul8	B0
.
	full_text!

%151 = fmul double %142, %150
,double8	B

	full_text

double %142
,double8	B

	full_text

double %150
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%152 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %139, i64 4
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %139
Pload8	BF
D
	full_text7
5
3%153 = load double, double* %152, align 8, !tbaa !8
.double*8	B

	full_text

double* %152
:fmul8	B0
.
	full_text!

%154 = fmul double %142, %153
,double8	B

	full_text

double %142
,double8	B

	full_text

double %153
7add8	B.
,
	full_text

%155 = add nsw i64 %139, -1
&i648	B

	full_text


i64 %139
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%156 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %155, i64 0
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %155
Pload8	BF
D
	full_text7
5
3%157 = load double, double* %156, align 8, !tbaa !8
.double*8	B

	full_text

double* %156
Bfdiv8	B8
6
	full_text)
'
%%158 = fdiv double 1.000000e+00, %157
,double8	B

	full_text

double %157
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%159 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %155, i64 1
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %155
Pload8	BF
D
	full_text7
5
3%160 = load double, double* %159, align 8, !tbaa !8
.double*8	B

	full_text

double* %159
:fmul8	B0
.
	full_text!

%161 = fmul double %158, %160
,double8	B

	full_text

double %158
,double8	B

	full_text

double %160
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%162 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %155, i64 2
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %155
Pload8	BF
D
	full_text7
5
3%163 = load double, double* %162, align 8, !tbaa !8
.double*8	B

	full_text

double* %162
:fmul8	B0
.
	full_text!

%164 = fmul double %158, %163
,double8	B

	full_text

double %158
,double8	B

	full_text

double %163
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%165 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %155, i64 3
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %155
Pload8	BF
D
	full_text7
5
3%166 = load double, double* %165, align 8, !tbaa !8
.double*8	B

	full_text

double* %165
:fmul8	B0
.
	full_text!

%167 = fmul double %158, %166
,double8	B

	full_text

double %158
,double8	B

	full_text

double %166
®getelementptr8	Bî
ë
	full_textÉ
Ä
~%168 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %134, i64 %136, i64 %155, i64 4
U[65 x [65 x [5 x double]]]*8	B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648	B

	full_text


i64 %134
&i648	B

	full_text


i64 %136
&i648	B

	full_text


i64 %155
Pload8	BF
D
	full_text7
5
3%169 = load double, double* %168, align 8, !tbaa !8
.double*8	B

	full_text

double* %168
:fmul8	B0
.
	full_text!

%170 = fmul double %158, %169
,double8	B

	full_text

double %158
,double8	B

	full_text

double %169
:fsub8	B0
.
	full_text!

%171 = fsub double %145, %161
,double8	B

	full_text

double %145
,double8	B

	full_text

double %161
Bfmul8	B8
6
	full_text)
'
%%172 = fmul double %171, 8.400000e+01
,double8	B

	full_text

double %171
tgetelementptr8	Ba
_
	full_textR
P
N%173 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %139, i64 1
9[5 x double]*8	B$
"
	full_text

[5 x double]* %27
&i648	B

	full_text


i64 %139
Pstore8	BE
C
	full_text6
4
2store double %172, double* %173, align 8, !tbaa !8
,double8	B

	full_text

double %172
.double*8	B

	full_text

double* %173
:fsub8	B0
.
	full_text!

%174 = fsub double %148, %164
,double8	B

	full_text

double %148
,double8	B

	full_text

double %164
Bfmul8	B8
6
	full_text)
'
%%175 = fmul double %174, 6.300000e+01
,double8	B

	full_text

double %174
tgetelementptr8	Ba
_
	full_textR
P
N%176 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %139, i64 2
9[5 x double]*8	B$
"
	full_text

[5 x double]* %27
&i648	B

	full_text


i64 %139
Pstore8	BE
C
	full_text6
4
2store double %175, double* %176, align 8, !tbaa !8
,double8	B

	full_text

double %175
.double*8	B

	full_text

double* %176
:fsub8	B0
.
	full_text!

%177 = fsub double %151, %167
,double8	B

	full_text

double %151
,double8	B

	full_text

double %167
Bfmul8	B8
6
	full_text)
'
%%178 = fmul double %177, 6.300000e+01
,double8	B

	full_text

double %177
tgetelementptr8	Ba
_
	full_textR
P
N%179 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %139, i64 3
9[5 x double]*8	B$
"
	full_text

[5 x double]* %27
&i648	B

	full_text


i64 %139
Pstore8	BE
C
	full_text6
4
2store double %178, double* %179, align 8, !tbaa !8
,double8	B

	full_text

double %178
.double*8	B

	full_text

double* %179
:fmul8	B0
.
	full_text!

%180 = fmul double %148, %148
,double8	B

	full_text

double %148
,double8	B

	full_text

double %148
mcall8	Bc
a
	full_textT
R
P%181 = tail call double @llvm.fmuladd.f64(double %145, double %145, double %180)
,double8	B

	full_text

double %145
,double8	B

	full_text

double %145
,double8	B

	full_text

double %180
mcall8	Bc
a
	full_textT
R
P%182 = tail call double @llvm.fmuladd.f64(double %151, double %151, double %181)
,double8	B

	full_text

double %151
,double8	B

	full_text

double %151
,double8	B

	full_text

double %181
:fmul8	B0
.
	full_text!

%183 = fmul double %164, %164
,double8	B

	full_text

double %164
,double8	B

	full_text

double %164
mcall8	Bc
a
	full_textT
R
P%184 = tail call double @llvm.fmuladd.f64(double %161, double %161, double %183)
,double8	B

	full_text

double %161
,double8	B

	full_text

double %161
,double8	B

	full_text

double %183
mcall8	Bc
a
	full_textT
R
P%185 = tail call double @llvm.fmuladd.f64(double %167, double %167, double %184)
,double8	B

	full_text

double %167
,double8	B

	full_text

double %167
,double8	B

	full_text

double %184
:fsub8	B0
.
	full_text!

%186 = fsub double %182, %185
,double8	B

	full_text

double %182
,double8	B

	full_text

double %185
:fmul8	B0
.
	full_text!

%187 = fmul double %161, %161
,double8	B

	full_text

double %161
,double8	B

	full_text

double %161
Cfsub8	B9
7
	full_text*
(
&%188 = fsub double -0.000000e+00, %187
,double8	B

	full_text

double %187
mcall8	Bc
a
	full_textT
R
P%189 = tail call double @llvm.fmuladd.f64(double %145, double %145, double %188)
,double8	B

	full_text

double %145
,double8	B

	full_text

double %145
,double8	B

	full_text

double %188
Bfmul8	B8
6
	full_text)
'
%%190 = fmul double %189, 1.050000e+01
,double8	B

	full_text

double %189
{call8	Bq
o
	full_textb
`
^%191 = tail call double @llvm.fmuladd.f64(double %186, double 0xC03E3D70A3D70A3B, double %190)
,double8	B

	full_text

double %186
,double8	B

	full_text

double %190
:fsub8	B0
.
	full_text!

%192 = fsub double %154, %170
,double8	B

	full_text

double %154
,double8	B

	full_text

double %170
{call8	Bq
o
	full_textb
`
^%193 = tail call double @llvm.fmuladd.f64(double %192, double 0x405EDEB851EB851E, double %191)
,double8	B

	full_text

double %192
,double8	B

	full_text

double %191
tgetelementptr8	Ba
_
	full_textR
P
N%194 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %139, i64 4
9[5 x double]*8	B$
"
	full_text

[5 x double]* %27
&i648	B

	full_text


i64 %139
Pstore8	BE
C
	full_text6
4
2store double %193, double* %194, align 8, !tbaa !8
,double8	B

	full_text

double %193
.double*8	B

	full_text

double* %194
:add8	B1
/
	full_text"
 
%195 = add nuw nsw i64 %139, 1
&i648	B

	full_text


i64 %139
:icmp8	B0
.
	full_text!

%196 = icmp eq i64 %195, %137
&i648	B

	full_text


i64 %195
&i648	B

	full_text


i64 %137
=br8	B5
3
	full_text&
$
"br i1 %196, label %197, label %138
$i18	B

	full_text
	
i1 %196
=br8
B5
3
	full_text&
$
"br i1 %130, label %203, label %198
$i18
B

	full_text
	
i1 %130
1shl8B(
&
	full_text

%199 = shl i64 %8, 32
$i648B

	full_text


i64 %8
;ashr8B1
/
	full_text"
 
%200 = ashr exact i64 %199, 32
&i648B

	full_text


i64 %199
2shl8B)
'
	full_text

%201 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
;ashr8B1
/
	full_text"
 
%202 = ashr exact i64 %201, 32
&i648B

	full_text


i64 %201
(br8B 

	full_text

br label %293
1shl8B(
&
	full_text

%204 = shl i64 %8, 32
$i648B

	full_text


i64 %8
;ashr8B1
/
	full_text"
 
%205 = ashr exact i64 %204, 32
&i648B

	full_text


i64 %204
2shl8B)
'
	full_text

%206 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
;ashr8B1
/
	full_text"
 
%207 = ashr exact i64 %206, 32
&i648B

	full_text


i64 %206
7zext8B-
+
	full_text

%208 = zext i32 %76 to i64
%i328B

	full_text
	
i32 %76
(br8B 

	full_text

br label %209
Fphi8B=
;
	full_text.
,
*%210 = phi i64 [ 1, %203 ], [ %219, %209 ]
&i648B

	full_text


i64 %219
®getelementptr8Bî
ë
	full_textÉ
Ä
~%211 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %205, i64 %207, i64 %210, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%212 = load double, double* %211, align 8, !tbaa !8
.double*8B

	full_text

double* %211
7add8B.
,
	full_text

%213 = add nsw i64 %210, -1
&i648B

	full_text


i64 %210
®getelementptr8Bî
ë
	full_textÉ
Ä
~%214 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %213, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %213
Pload8BF
D
	full_text7
5
3%215 = load double, double* %214, align 8, !tbaa !8
.double*8B

	full_text

double* %214
®getelementptr8Bî
ë
	full_textÉ
Ä
~%216 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %210, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%217 = load double, double* %216, align 8, !tbaa !8
.double*8B

	full_text

double* %216
vcall8Bl
j
	full_text]
[
Y%218 = tail call double @llvm.fmuladd.f64(double %217, double -2.000000e+00, double %215)
,double8B

	full_text

double %217
,double8B

	full_text

double %215
:add8B1
/
	full_text"
 
%219 = add nuw nsw i64 %210, 1
&i648B

	full_text


i64 %210
®getelementptr8Bî
ë
	full_textÉ
Ä
~%220 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %219, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%221 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
:fadd8B0
.
	full_text!

%222 = fadd double %218, %221
,double8B

	full_text

double %218
,double8B

	full_text

double %221
{call8Bq
o
	full_textb
`
^%223 = tail call double @llvm.fmuladd.f64(double %222, double 0x40A7418000000001, double %212)
,double8B

	full_text

double %222
,double8B

	full_text

double %212
Pstore8BE
C
	full_text6
4
2store double %223, double* %211, align 8, !tbaa !8
,double8B

	full_text

double %223
.double*8B

	full_text

double* %211
®getelementptr8Bî
ë
	full_textÉ
Ä
~%224 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %205, i64 %207, i64 %210, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%225 = load double, double* %224, align 8, !tbaa !8
.double*8B

	full_text

double* %224
tgetelementptr8Ba
_
	full_textR
P
N%226 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %219, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%227 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
tgetelementptr8Ba
_
	full_textR
P
N%228 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %210, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%229 = load double, double* %228, align 8, !tbaa !8
.double*8B

	full_text

double* %228
:fsub8B0
.
	full_text!

%230 = fsub double %227, %229
,double8B

	full_text

double %227
,double8B

	full_text

double %229
{call8Bq
o
	full_textb
`
^%231 = tail call double @llvm.fmuladd.f64(double %230, double 0x4019333333333334, double %225)
,double8B

	full_text

double %230
,double8B

	full_text

double %225
®getelementptr8Bî
ë
	full_textÉ
Ä
~%232 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %213, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %213
Pload8BF
D
	full_text7
5
3%233 = load double, double* %232, align 8, !tbaa !8
.double*8B

	full_text

double* %232
®getelementptr8Bî
ë
	full_textÉ
Ä
~%234 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %210, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%235 = load double, double* %234, align 8, !tbaa !8
.double*8B

	full_text

double* %234
vcall8Bl
j
	full_text]
[
Y%236 = tail call double @llvm.fmuladd.f64(double %235, double -2.000000e+00, double %233)
,double8B

	full_text

double %235
,double8B

	full_text

double %233
®getelementptr8Bî
ë
	full_textÉ
Ä
~%237 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %219, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%238 = load double, double* %237, align 8, !tbaa !8
.double*8B

	full_text

double* %237
:fadd8B0
.
	full_text!

%239 = fadd double %236, %238
,double8B

	full_text

double %236
,double8B

	full_text

double %238
{call8Bq
o
	full_textb
`
^%240 = tail call double @llvm.fmuladd.f64(double %239, double 0x40A7418000000001, double %231)
,double8B

	full_text

double %239
,double8B

	full_text

double %231
Pstore8BE
C
	full_text6
4
2store double %240, double* %224, align 8, !tbaa !8
,double8B

	full_text

double %240
.double*8B

	full_text

double* %224
®getelementptr8Bî
ë
	full_textÉ
Ä
~%241 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %205, i64 %207, i64 %210, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%242 = load double, double* %241, align 8, !tbaa !8
.double*8B

	full_text

double* %241
tgetelementptr8Ba
_
	full_textR
P
N%243 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %219, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%244 = load double, double* %243, align 8, !tbaa !8
.double*8B

	full_text

double* %243
tgetelementptr8Ba
_
	full_textR
P
N%245 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %210, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%246 = load double, double* %245, align 8, !tbaa !8
.double*8B

	full_text

double* %245
:fsub8B0
.
	full_text!

%247 = fsub double %244, %246
,double8B

	full_text

double %244
,double8B

	full_text

double %246
{call8Bq
o
	full_textb
`
^%248 = tail call double @llvm.fmuladd.f64(double %247, double 0x4019333333333334, double %242)
,double8B

	full_text

double %247
,double8B

	full_text

double %242
®getelementptr8Bî
ë
	full_textÉ
Ä
~%249 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %213, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %213
Pload8BF
D
	full_text7
5
3%250 = load double, double* %249, align 8, !tbaa !8
.double*8B

	full_text

double* %249
®getelementptr8Bî
ë
	full_textÉ
Ä
~%251 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %210, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%252 = load double, double* %251, align 8, !tbaa !8
.double*8B

	full_text

double* %251
vcall8Bl
j
	full_text]
[
Y%253 = tail call double @llvm.fmuladd.f64(double %252, double -2.000000e+00, double %250)
,double8B

	full_text

double %252
,double8B

	full_text

double %250
®getelementptr8Bî
ë
	full_textÉ
Ä
~%254 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %219, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%255 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
:fadd8B0
.
	full_text!

%256 = fadd double %253, %255
,double8B

	full_text

double %253
,double8B

	full_text

double %255
{call8Bq
o
	full_textb
`
^%257 = tail call double @llvm.fmuladd.f64(double %256, double 0x40A7418000000001, double %248)
,double8B

	full_text

double %256
,double8B

	full_text

double %248
Pstore8BE
C
	full_text6
4
2store double %257, double* %241, align 8, !tbaa !8
,double8B

	full_text

double %257
.double*8B

	full_text

double* %241
®getelementptr8Bî
ë
	full_textÉ
Ä
~%258 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %205, i64 %207, i64 %210, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%259 = load double, double* %258, align 8, !tbaa !8
.double*8B

	full_text

double* %258
tgetelementptr8Ba
_
	full_textR
P
N%260 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %219, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%261 = load double, double* %260, align 8, !tbaa !8
.double*8B

	full_text

double* %260
tgetelementptr8Ba
_
	full_textR
P
N%262 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %210, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%263 = load double, double* %262, align 8, !tbaa !8
.double*8B

	full_text

double* %262
:fsub8B0
.
	full_text!

%264 = fsub double %261, %263
,double8B

	full_text

double %261
,double8B

	full_text

double %263
{call8Bq
o
	full_textb
`
^%265 = tail call double @llvm.fmuladd.f64(double %264, double 0x4019333333333334, double %259)
,double8B

	full_text

double %264
,double8B

	full_text

double %259
®getelementptr8Bî
ë
	full_textÉ
Ä
~%266 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %213, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %213
Pload8BF
D
	full_text7
5
3%267 = load double, double* %266, align 8, !tbaa !8
.double*8B

	full_text

double* %266
®getelementptr8Bî
ë
	full_textÉ
Ä
~%268 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %210, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%269 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
vcall8Bl
j
	full_text]
[
Y%270 = tail call double @llvm.fmuladd.f64(double %269, double -2.000000e+00, double %267)
,double8B

	full_text

double %269
,double8B

	full_text

double %267
®getelementptr8Bî
ë
	full_textÉ
Ä
~%271 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %219, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%272 = load double, double* %271, align 8, !tbaa !8
.double*8B

	full_text

double* %271
:fadd8B0
.
	full_text!

%273 = fadd double %270, %272
,double8B

	full_text

double %270
,double8B

	full_text

double %272
{call8Bq
o
	full_textb
`
^%274 = tail call double @llvm.fmuladd.f64(double %273, double 0x40A7418000000001, double %265)
,double8B

	full_text

double %273
,double8B

	full_text

double %265
Pstore8BE
C
	full_text6
4
2store double %274, double* %258, align 8, !tbaa !8
,double8B

	full_text

double %274
.double*8B

	full_text

double* %258
®getelementptr8Bî
ë
	full_textÉ
Ä
~%275 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %205, i64 %207, i64 %210, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%276 = load double, double* %275, align 8, !tbaa !8
.double*8B

	full_text

double* %275
tgetelementptr8Ba
_
	full_textR
P
N%277 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %219, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%278 = load double, double* %277, align 8, !tbaa !8
.double*8B

	full_text

double* %277
tgetelementptr8Ba
_
	full_textR
P
N%279 = getelementptr inbounds [5 x double], [5 x double]* %27, i64 %210, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %27
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%280 = load double, double* %279, align 8, !tbaa !8
.double*8B

	full_text

double* %279
:fsub8B0
.
	full_text!

%281 = fsub double %278, %280
,double8B

	full_text

double %278
,double8B

	full_text

double %280
{call8Bq
o
	full_textb
`
^%282 = tail call double @llvm.fmuladd.f64(double %281, double 0x4019333333333334, double %276)
,double8B

	full_text

double %281
,double8B

	full_text

double %276
®getelementptr8Bî
ë
	full_textÉ
Ä
~%283 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %213, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %213
Pload8BF
D
	full_text7
5
3%284 = load double, double* %283, align 8, !tbaa !8
.double*8B

	full_text

double* %283
®getelementptr8Bî
ë
	full_textÉ
Ä
~%285 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %210, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %210
Pload8BF
D
	full_text7
5
3%286 = load double, double* %285, align 8, !tbaa !8
.double*8B

	full_text

double* %285
vcall8Bl
j
	full_text]
[
Y%287 = tail call double @llvm.fmuladd.f64(double %286, double -2.000000e+00, double %284)
,double8B

	full_text

double %286
,double8B

	full_text

double %284
®getelementptr8Bî
ë
	full_textÉ
Ä
~%288 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %205, i64 %207, i64 %219, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %205
&i648B

	full_text


i64 %207
&i648B

	full_text


i64 %219
Pload8BF
D
	full_text7
5
3%289 = load double, double* %288, align 8, !tbaa !8
.double*8B

	full_text

double* %288
:fadd8B0
.
	full_text!

%290 = fadd double %287, %289
,double8B

	full_text

double %287
,double8B

	full_text

double %289
{call8Bq
o
	full_textb
`
^%291 = tail call double @llvm.fmuladd.f64(double %290, double 0x40A7418000000001, double %282)
,double8B

	full_text

double %290
,double8B

	full_text

double %282
Pstore8BE
C
	full_text6
4
2store double %291, double* %275, align 8, !tbaa !8
,double8B

	full_text

double %291
.double*8B

	full_text

double* %275
:icmp8B0
.
	full_text!

%292 = icmp eq i64 %219, %208
&i648B

	full_text


i64 %219
&i648B

	full_text


i64 %208
=br8B5
3
	full_text&
$
"br i1 %292, label %293, label %209
$i18B

	full_text
	
i1 %292
Iphi8B@
>
	full_text1
/
-%294 = phi i64 [ %202, %198 ], [ %207, %209 ]
&i648B

	full_text


i64 %202
&i648B

	full_text


i64 %207
Iphi8B@
>
	full_text1
/
-%295 = phi i64 [ %200, %198 ], [ %205, %209 ]
&i648B

	full_text


i64 %200
&i648B

	full_text


i64 %205
kcall8Ba
_
	full_textR
P
N%296 = tail call double @_Z3maxdd(double 7.500000e-01, double 7.500000e-01) #3
ccall8BY
W
	full_textJ
H
F%297 = tail call double @_Z3maxdd(double %296, double 1.000000e+00) #3
,double8B

	full_text

double %296
Bfmul8B8
6
	full_text)
'
%%298 = fmul double %297, 2.500000e-01
,double8B

	full_text

double %297
Cfsub8B9
7
	full_text*
(
&%299 = fsub double -0.000000e+00, %298
,double8B

	full_text

double %298
£getelementptr8Bè
å
	full_text
}
{%300 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 1, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%301 = load double, double* %300, align 8, !tbaa !8
.double*8B

	full_text

double* %300
£getelementptr8Bè
å
	full_text
}
{%302 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 1, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%303 = load double, double* %302, align 8, !tbaa !8
.double*8B

	full_text

double* %302
£getelementptr8Bè
å
	full_text
}
{%304 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 2, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%305 = load double, double* %304, align 8, !tbaa !8
.double*8B

	full_text

double* %304
Bfmul8B8
6
	full_text)
'
%%306 = fmul double %305, 4.000000e+00
,double8B

	full_text

double %305
Cfsub8B9
7
	full_text*
(
&%307 = fsub double -0.000000e+00, %306
,double8B

	full_text

double %306
ucall8Bk
i
	full_text\
Z
X%308 = tail call double @llvm.fmuladd.f64(double %303, double 5.000000e+00, double %307)
,double8B

	full_text

double %303
,double8B

	full_text

double %307
£getelementptr8Bè
å
	full_text
}
{%309 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 3, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%310 = load double, double* %309, align 8, !tbaa !8
.double*8B

	full_text

double* %309
:fadd8B0
.
	full_text!

%311 = fadd double %310, %308
,double8B

	full_text

double %310
,double8B

	full_text

double %308
mcall8Bc
a
	full_textT
R
P%312 = tail call double @llvm.fmuladd.f64(double %299, double %311, double %301)
,double8B

	full_text

double %299
,double8B

	full_text

double %311
,double8B

	full_text

double %301
Pstore8BE
C
	full_text6
4
2store double %312, double* %300, align 8, !tbaa !8
,double8B

	full_text

double %312
.double*8B

	full_text

double* %300
£getelementptr8Bè
å
	full_text
}
{%313 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 2, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%314 = load double, double* %313, align 8, !tbaa !8
.double*8B

	full_text

double* %313
Pload8BF
D
	full_text7
5
3%315 = load double, double* %302, align 8, !tbaa !8
.double*8B

	full_text

double* %302
Pload8BF
D
	full_text7
5
3%316 = load double, double* %304, align 8, !tbaa !8
.double*8B

	full_text

double* %304
Bfmul8B8
6
	full_text)
'
%%317 = fmul double %316, 6.000000e+00
,double8B

	full_text

double %316
vcall8Bl
j
	full_text]
[
Y%318 = tail call double @llvm.fmuladd.f64(double %315, double -4.000000e+00, double %317)
,double8B

	full_text

double %315
,double8B

	full_text

double %317
Pload8BF
D
	full_text7
5
3%319 = load double, double* %309, align 8, !tbaa !8
.double*8B

	full_text

double* %309
vcall8Bl
j
	full_text]
[
Y%320 = tail call double @llvm.fmuladd.f64(double %319, double -4.000000e+00, double %318)
,double8B

	full_text

double %319
,double8B

	full_text

double %318
£getelementptr8Bè
å
	full_text
}
{%321 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 4, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%322 = load double, double* %321, align 8, !tbaa !8
.double*8B

	full_text

double* %321
:fadd8B0
.
	full_text!

%323 = fadd double %322, %320
,double8B

	full_text

double %322
,double8B

	full_text

double %320
mcall8Bc
a
	full_textT
R
P%324 = tail call double @llvm.fmuladd.f64(double %299, double %323, double %314)
,double8B

	full_text

double %299
,double8B

	full_text

double %323
,double8B

	full_text

double %314
Pstore8BE
C
	full_text6
4
2store double %324, double* %313, align 8, !tbaa !8
,double8B

	full_text

double %324
.double*8B

	full_text

double* %313
£getelementptr8Bè
å
	full_text
}
{%325 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 1, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%326 = load double, double* %325, align 8, !tbaa !8
.double*8B

	full_text

double* %325
£getelementptr8Bè
å
	full_text
}
{%327 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 1, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%328 = load double, double* %327, align 8, !tbaa !8
.double*8B

	full_text

double* %327
£getelementptr8Bè
å
	full_text
}
{%329 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 2, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%330 = load double, double* %329, align 8, !tbaa !8
.double*8B

	full_text

double* %329
Bfmul8B8
6
	full_text)
'
%%331 = fmul double %330, 4.000000e+00
,double8B

	full_text

double %330
Cfsub8B9
7
	full_text*
(
&%332 = fsub double -0.000000e+00, %331
,double8B

	full_text

double %331
ucall8Bk
i
	full_text\
Z
X%333 = tail call double @llvm.fmuladd.f64(double %328, double 5.000000e+00, double %332)
,double8B

	full_text

double %328
,double8B

	full_text

double %332
£getelementptr8Bè
å
	full_text
}
{%334 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 3, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%335 = load double, double* %334, align 8, !tbaa !8
.double*8B

	full_text

double* %334
:fadd8B0
.
	full_text!

%336 = fadd double %335, %333
,double8B

	full_text

double %335
,double8B

	full_text

double %333
mcall8Bc
a
	full_textT
R
P%337 = tail call double @llvm.fmuladd.f64(double %299, double %336, double %326)
,double8B

	full_text

double %299
,double8B

	full_text

double %336
,double8B

	full_text

double %326
Pstore8BE
C
	full_text6
4
2store double %337, double* %325, align 8, !tbaa !8
,double8B

	full_text

double %337
.double*8B

	full_text

double* %325
£getelementptr8Bè
å
	full_text
}
{%338 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 2, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%339 = load double, double* %338, align 8, !tbaa !8
.double*8B

	full_text

double* %338
Pload8BF
D
	full_text7
5
3%340 = load double, double* %327, align 8, !tbaa !8
.double*8B

	full_text

double* %327
Pload8BF
D
	full_text7
5
3%341 = load double, double* %329, align 8, !tbaa !8
.double*8B

	full_text

double* %329
Bfmul8B8
6
	full_text)
'
%%342 = fmul double %341, 6.000000e+00
,double8B

	full_text

double %341
vcall8Bl
j
	full_text]
[
Y%343 = tail call double @llvm.fmuladd.f64(double %340, double -4.000000e+00, double %342)
,double8B

	full_text

double %340
,double8B

	full_text

double %342
Pload8BF
D
	full_text7
5
3%344 = load double, double* %334, align 8, !tbaa !8
.double*8B

	full_text

double* %334
vcall8Bl
j
	full_text]
[
Y%345 = tail call double @llvm.fmuladd.f64(double %344, double -4.000000e+00, double %343)
,double8B

	full_text

double %344
,double8B

	full_text

double %343
£getelementptr8Bè
å
	full_text
}
{%346 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 4, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%347 = load double, double* %346, align 8, !tbaa !8
.double*8B

	full_text

double* %346
:fadd8B0
.
	full_text!

%348 = fadd double %347, %345
,double8B

	full_text

double %347
,double8B

	full_text

double %345
mcall8Bc
a
	full_textT
R
P%349 = tail call double @llvm.fmuladd.f64(double %299, double %348, double %339)
,double8B

	full_text

double %299
,double8B

	full_text

double %348
,double8B

	full_text

double %339
Pstore8BE
C
	full_text6
4
2store double %349, double* %338, align 8, !tbaa !8
,double8B

	full_text

double %349
.double*8B

	full_text

double* %338
£getelementptr8Bè
å
	full_text
}
{%350 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 1, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%351 = load double, double* %350, align 8, !tbaa !8
.double*8B

	full_text

double* %350
£getelementptr8Bè
å
	full_text
}
{%352 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 1, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%353 = load double, double* %352, align 8, !tbaa !8
.double*8B

	full_text

double* %352
£getelementptr8Bè
å
	full_text
}
{%354 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 2, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%355 = load double, double* %354, align 8, !tbaa !8
.double*8B

	full_text

double* %354
Bfmul8B8
6
	full_text)
'
%%356 = fmul double %355, 4.000000e+00
,double8B

	full_text

double %355
Cfsub8B9
7
	full_text*
(
&%357 = fsub double -0.000000e+00, %356
,double8B

	full_text

double %356
ucall8Bk
i
	full_text\
Z
X%358 = tail call double @llvm.fmuladd.f64(double %353, double 5.000000e+00, double %357)
,double8B

	full_text

double %353
,double8B

	full_text

double %357
£getelementptr8Bè
å
	full_text
}
{%359 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 3, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%360 = load double, double* %359, align 8, !tbaa !8
.double*8B

	full_text

double* %359
:fadd8B0
.
	full_text!

%361 = fadd double %360, %358
,double8B

	full_text

double %360
,double8B

	full_text

double %358
mcall8Bc
a
	full_textT
R
P%362 = tail call double @llvm.fmuladd.f64(double %299, double %361, double %351)
,double8B

	full_text

double %299
,double8B

	full_text

double %361
,double8B

	full_text

double %351
Pstore8BE
C
	full_text6
4
2store double %362, double* %350, align 8, !tbaa !8
,double8B

	full_text

double %362
.double*8B

	full_text

double* %350
£getelementptr8Bè
å
	full_text
}
{%363 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 2, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%364 = load double, double* %363, align 8, !tbaa !8
.double*8B

	full_text

double* %363
Pload8BF
D
	full_text7
5
3%365 = load double, double* %352, align 8, !tbaa !8
.double*8B

	full_text

double* %352
Pload8BF
D
	full_text7
5
3%366 = load double, double* %354, align 8, !tbaa !8
.double*8B

	full_text

double* %354
Bfmul8B8
6
	full_text)
'
%%367 = fmul double %366, 6.000000e+00
,double8B

	full_text

double %366
vcall8Bl
j
	full_text]
[
Y%368 = tail call double @llvm.fmuladd.f64(double %365, double -4.000000e+00, double %367)
,double8B

	full_text

double %365
,double8B

	full_text

double %367
Pload8BF
D
	full_text7
5
3%369 = load double, double* %359, align 8, !tbaa !8
.double*8B

	full_text

double* %359
vcall8Bl
j
	full_text]
[
Y%370 = tail call double @llvm.fmuladd.f64(double %369, double -4.000000e+00, double %368)
,double8B

	full_text

double %369
,double8B

	full_text

double %368
£getelementptr8Bè
å
	full_text
}
{%371 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 4, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%372 = load double, double* %371, align 8, !tbaa !8
.double*8B

	full_text

double* %371
:fadd8B0
.
	full_text!

%373 = fadd double %372, %370
,double8B

	full_text

double %372
,double8B

	full_text

double %370
mcall8Bc
a
	full_textT
R
P%374 = tail call double @llvm.fmuladd.f64(double %299, double %373, double %364)
,double8B

	full_text

double %299
,double8B

	full_text

double %373
,double8B

	full_text

double %364
Pstore8BE
C
	full_text6
4
2store double %374, double* %363, align 8, !tbaa !8
,double8B

	full_text

double %374
.double*8B

	full_text

double* %363
£getelementptr8Bè
å
	full_text
}
{%375 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 1, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%376 = load double, double* %375, align 8, !tbaa !8
.double*8B

	full_text

double* %375
£getelementptr8Bè
å
	full_text
}
{%377 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 1, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%378 = load double, double* %377, align 8, !tbaa !8
.double*8B

	full_text

double* %377
£getelementptr8Bè
å
	full_text
}
{%379 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 2, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%380 = load double, double* %379, align 8, !tbaa !8
.double*8B

	full_text

double* %379
Bfmul8B8
6
	full_text)
'
%%381 = fmul double %380, 4.000000e+00
,double8B

	full_text

double %380
Cfsub8B9
7
	full_text*
(
&%382 = fsub double -0.000000e+00, %381
,double8B

	full_text

double %381
ucall8Bk
i
	full_text\
Z
X%383 = tail call double @llvm.fmuladd.f64(double %378, double 5.000000e+00, double %382)
,double8B

	full_text

double %378
,double8B

	full_text

double %382
£getelementptr8Bè
å
	full_text
}
{%384 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 3, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%385 = load double, double* %384, align 8, !tbaa !8
.double*8B

	full_text

double* %384
:fadd8B0
.
	full_text!

%386 = fadd double %385, %383
,double8B

	full_text

double %385
,double8B

	full_text

double %383
mcall8Bc
a
	full_textT
R
P%387 = tail call double @llvm.fmuladd.f64(double %299, double %386, double %376)
,double8B

	full_text

double %299
,double8B

	full_text

double %386
,double8B

	full_text

double %376
Pstore8BE
C
	full_text6
4
2store double %387, double* %375, align 8, !tbaa !8
,double8B

	full_text

double %387
.double*8B

	full_text

double* %375
£getelementptr8Bè
å
	full_text
}
{%388 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 2, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%389 = load double, double* %388, align 8, !tbaa !8
.double*8B

	full_text

double* %388
Pload8BF
D
	full_text7
5
3%390 = load double, double* %377, align 8, !tbaa !8
.double*8B

	full_text

double* %377
Pload8BF
D
	full_text7
5
3%391 = load double, double* %379, align 8, !tbaa !8
.double*8B

	full_text

double* %379
Bfmul8B8
6
	full_text)
'
%%392 = fmul double %391, 6.000000e+00
,double8B

	full_text

double %391
vcall8Bl
j
	full_text]
[
Y%393 = tail call double @llvm.fmuladd.f64(double %390, double -4.000000e+00, double %392)
,double8B

	full_text

double %390
,double8B

	full_text

double %392
Pload8BF
D
	full_text7
5
3%394 = load double, double* %384, align 8, !tbaa !8
.double*8B

	full_text

double* %384
vcall8Bl
j
	full_text]
[
Y%395 = tail call double @llvm.fmuladd.f64(double %394, double -4.000000e+00, double %393)
,double8B

	full_text

double %394
,double8B

	full_text

double %393
£getelementptr8Bè
å
	full_text
}
{%396 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 4, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%397 = load double, double* %396, align 8, !tbaa !8
.double*8B

	full_text

double* %396
:fadd8B0
.
	full_text!

%398 = fadd double %397, %395
,double8B

	full_text

double %397
,double8B

	full_text

double %395
mcall8Bc
a
	full_textT
R
P%399 = tail call double @llvm.fmuladd.f64(double %299, double %398, double %389)
,double8B

	full_text

double %299
,double8B

	full_text

double %398
,double8B

	full_text

double %389
Pstore8BE
C
	full_text6
4
2store double %399, double* %388, align 8, !tbaa !8
,double8B

	full_text

double %399
.double*8B

	full_text

double* %388
£getelementptr8Bè
å
	full_text
}
{%400 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 1, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%401 = load double, double* %400, align 8, !tbaa !8
.double*8B

	full_text

double* %400
£getelementptr8Bè
å
	full_text
}
{%402 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 1, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%403 = load double, double* %402, align 8, !tbaa !8
.double*8B

	full_text

double* %402
£getelementptr8Bè
å
	full_text
}
{%404 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 2, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%405 = load double, double* %404, align 8, !tbaa !8
.double*8B

	full_text

double* %404
Bfmul8B8
6
	full_text)
'
%%406 = fmul double %405, 4.000000e+00
,double8B

	full_text

double %405
Cfsub8B9
7
	full_text*
(
&%407 = fsub double -0.000000e+00, %406
,double8B

	full_text

double %406
ucall8Bk
i
	full_text\
Z
X%408 = tail call double @llvm.fmuladd.f64(double %403, double 5.000000e+00, double %407)
,double8B

	full_text

double %403
,double8B

	full_text

double %407
£getelementptr8Bè
å
	full_text
}
{%409 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 3, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%410 = load double, double* %409, align 8, !tbaa !8
.double*8B

	full_text

double* %409
:fadd8B0
.
	full_text!

%411 = fadd double %410, %408
,double8B

	full_text

double %410
,double8B

	full_text

double %408
mcall8Bc
a
	full_textT
R
P%412 = tail call double @llvm.fmuladd.f64(double %299, double %411, double %401)
,double8B

	full_text

double %299
,double8B

	full_text

double %411
,double8B

	full_text

double %401
Pstore8BE
C
	full_text6
4
2store double %412, double* %400, align 8, !tbaa !8
,double8B

	full_text

double %412
.double*8B

	full_text

double* %400
£getelementptr8Bè
å
	full_text
}
{%413 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 2, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%414 = load double, double* %413, align 8, !tbaa !8
.double*8B

	full_text

double* %413
Pload8BF
D
	full_text7
5
3%415 = load double, double* %402, align 8, !tbaa !8
.double*8B

	full_text

double* %402
Pload8BF
D
	full_text7
5
3%416 = load double, double* %404, align 8, !tbaa !8
.double*8B

	full_text

double* %404
Bfmul8B8
6
	full_text)
'
%%417 = fmul double %416, 6.000000e+00
,double8B

	full_text

double %416
vcall8Bl
j
	full_text]
[
Y%418 = tail call double @llvm.fmuladd.f64(double %415, double -4.000000e+00, double %417)
,double8B

	full_text

double %415
,double8B

	full_text

double %417
Pload8BF
D
	full_text7
5
3%419 = load double, double* %409, align 8, !tbaa !8
.double*8B

	full_text

double* %409
vcall8Bl
j
	full_text]
[
Y%420 = tail call double @llvm.fmuladd.f64(double %419, double -4.000000e+00, double %418)
,double8B

	full_text

double %419
,double8B

	full_text

double %418
£getelementptr8Bè
å
	full_text
}
{%421 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 4, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
Pload8BF
D
	full_text7
5
3%422 = load double, double* %421, align 8, !tbaa !8
.double*8B

	full_text

double* %421
:fadd8B0
.
	full_text!

%423 = fadd double %422, %420
,double8B

	full_text

double %422
,double8B

	full_text

double %420
mcall8Bc
a
	full_textT
R
P%424 = tail call double @llvm.fmuladd.f64(double %299, double %423, double %414)
,double8B

	full_text

double %299
,double8B

	full_text

double %423
,double8B

	full_text

double %414
Pstore8BE
C
	full_text6
4
2store double %424, double* %413, align 8, !tbaa !8
,double8B

	full_text

double %424
.double*8B

	full_text

double* %413
5add8B,
*
	full_text

%425 = add nsw i32 %3, -3
6icmp8B,
*
	full_text

%426 = icmp sgt i32 %3, 6
=br8B5
3
	full_text&
$
"br i1 %426, label %427, label %521
$i18B

	full_text
	
i1 %426
8zext8B.
,
	full_text

%428 = zext i32 %425 to i64
&i328B

	full_text


i32 %425
(br8B 

	full_text

br label %429
Fphi8B=
;
	full_text.
,
*%430 = phi i64 [ 3, %427 ], [ %433, %429 ]
&i648B

	full_text


i64 %433
7add8B.
,
	full_text

%431 = add nsw i64 %430, -2
&i648B

	full_text


i64 %430
7add8B.
,
	full_text

%432 = add nsw i64 %430, -1
&i648B

	full_text


i64 %430
:add8B1
/
	full_text"
 
%433 = add nuw nsw i64 %430, 1
&i648B

	full_text


i64 %430
:add8B1
/
	full_text"
 
%434 = add nuw nsw i64 %430, 2
&i648B

	full_text


i64 %430
®getelementptr8Bî
ë
	full_textÉ
Ä
~%435 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %430, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%436 = load double, double* %435, align 8, !tbaa !8
.double*8B

	full_text

double* %435
®getelementptr8Bî
ë
	full_textÉ
Ä
~%437 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %431, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %431
Pload8BF
D
	full_text7
5
3%438 = load double, double* %437, align 8, !tbaa !8
.double*8B

	full_text

double* %437
®getelementptr8Bî
ë
	full_textÉ
Ä
~%439 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %432, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %432
Pload8BF
D
	full_text7
5
3%440 = load double, double* %439, align 8, !tbaa !8
.double*8B

	full_text

double* %439
vcall8Bl
j
	full_text]
[
Y%441 = tail call double @llvm.fmuladd.f64(double %440, double -4.000000e+00, double %438)
,double8B

	full_text

double %440
,double8B

	full_text

double %438
®getelementptr8Bî
ë
	full_textÉ
Ä
~%442 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %430, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%443 = load double, double* %442, align 8, !tbaa !8
.double*8B

	full_text

double* %442
ucall8Bk
i
	full_text\
Z
X%444 = tail call double @llvm.fmuladd.f64(double %443, double 6.000000e+00, double %441)
,double8B

	full_text

double %443
,double8B

	full_text

double %441
®getelementptr8Bî
ë
	full_textÉ
Ä
~%445 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %433, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %433
Pload8BF
D
	full_text7
5
3%446 = load double, double* %445, align 8, !tbaa !8
.double*8B

	full_text

double* %445
vcall8Bl
j
	full_text]
[
Y%447 = tail call double @llvm.fmuladd.f64(double %446, double -4.000000e+00, double %444)
,double8B

	full_text

double %446
,double8B

	full_text

double %444
®getelementptr8Bî
ë
	full_textÉ
Ä
~%448 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %434, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%449 = load double, double* %448, align 8, !tbaa !8
.double*8B

	full_text

double* %448
:fadd8B0
.
	full_text!

%450 = fadd double %447, %449
,double8B

	full_text

double %447
,double8B

	full_text

double %449
mcall8Bc
a
	full_textT
R
P%451 = tail call double @llvm.fmuladd.f64(double %299, double %450, double %436)
,double8B

	full_text

double %299
,double8B

	full_text

double %450
,double8B

	full_text

double %436
Pstore8BE
C
	full_text6
4
2store double %451, double* %435, align 8, !tbaa !8
,double8B

	full_text

double %451
.double*8B

	full_text

double* %435
®getelementptr8Bî
ë
	full_textÉ
Ä
~%452 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %430, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%453 = load double, double* %452, align 8, !tbaa !8
.double*8B

	full_text

double* %452
®getelementptr8Bî
ë
	full_textÉ
Ä
~%454 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %431, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %431
Pload8BF
D
	full_text7
5
3%455 = load double, double* %454, align 8, !tbaa !8
.double*8B

	full_text

double* %454
®getelementptr8Bî
ë
	full_textÉ
Ä
~%456 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %432, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %432
Pload8BF
D
	full_text7
5
3%457 = load double, double* %456, align 8, !tbaa !8
.double*8B

	full_text

double* %456
vcall8Bl
j
	full_text]
[
Y%458 = tail call double @llvm.fmuladd.f64(double %457, double -4.000000e+00, double %455)
,double8B

	full_text

double %457
,double8B

	full_text

double %455
®getelementptr8Bî
ë
	full_textÉ
Ä
~%459 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %430, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%460 = load double, double* %459, align 8, !tbaa !8
.double*8B

	full_text

double* %459
ucall8Bk
i
	full_text\
Z
X%461 = tail call double @llvm.fmuladd.f64(double %460, double 6.000000e+00, double %458)
,double8B

	full_text

double %460
,double8B

	full_text

double %458
®getelementptr8Bî
ë
	full_textÉ
Ä
~%462 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %433, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %433
Pload8BF
D
	full_text7
5
3%463 = load double, double* %462, align 8, !tbaa !8
.double*8B

	full_text

double* %462
vcall8Bl
j
	full_text]
[
Y%464 = tail call double @llvm.fmuladd.f64(double %463, double -4.000000e+00, double %461)
,double8B

	full_text

double %463
,double8B

	full_text

double %461
®getelementptr8Bî
ë
	full_textÉ
Ä
~%465 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %434, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%466 = load double, double* %465, align 8, !tbaa !8
.double*8B

	full_text

double* %465
:fadd8B0
.
	full_text!

%467 = fadd double %464, %466
,double8B

	full_text

double %464
,double8B

	full_text

double %466
mcall8Bc
a
	full_textT
R
P%468 = tail call double @llvm.fmuladd.f64(double %299, double %467, double %453)
,double8B

	full_text

double %299
,double8B

	full_text

double %467
,double8B

	full_text

double %453
Pstore8BE
C
	full_text6
4
2store double %468, double* %452, align 8, !tbaa !8
,double8B

	full_text

double %468
.double*8B

	full_text

double* %452
®getelementptr8Bî
ë
	full_textÉ
Ä
~%469 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %430, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%470 = load double, double* %469, align 8, !tbaa !8
.double*8B

	full_text

double* %469
®getelementptr8Bî
ë
	full_textÉ
Ä
~%471 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %431, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %431
Pload8BF
D
	full_text7
5
3%472 = load double, double* %471, align 8, !tbaa !8
.double*8B

	full_text

double* %471
®getelementptr8Bî
ë
	full_textÉ
Ä
~%473 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %432, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %432
Pload8BF
D
	full_text7
5
3%474 = load double, double* %473, align 8, !tbaa !8
.double*8B

	full_text

double* %473
vcall8Bl
j
	full_text]
[
Y%475 = tail call double @llvm.fmuladd.f64(double %474, double -4.000000e+00, double %472)
,double8B

	full_text

double %474
,double8B

	full_text

double %472
®getelementptr8Bî
ë
	full_textÉ
Ä
~%476 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %430, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%477 = load double, double* %476, align 8, !tbaa !8
.double*8B

	full_text

double* %476
ucall8Bk
i
	full_text\
Z
X%478 = tail call double @llvm.fmuladd.f64(double %477, double 6.000000e+00, double %475)
,double8B

	full_text

double %477
,double8B

	full_text

double %475
®getelementptr8Bî
ë
	full_textÉ
Ä
~%479 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %433, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %433
Pload8BF
D
	full_text7
5
3%480 = load double, double* %479, align 8, !tbaa !8
.double*8B

	full_text

double* %479
vcall8Bl
j
	full_text]
[
Y%481 = tail call double @llvm.fmuladd.f64(double %480, double -4.000000e+00, double %478)
,double8B

	full_text

double %480
,double8B

	full_text

double %478
®getelementptr8Bî
ë
	full_textÉ
Ä
~%482 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %434, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%483 = load double, double* %482, align 8, !tbaa !8
.double*8B

	full_text

double* %482
:fadd8B0
.
	full_text!

%484 = fadd double %481, %483
,double8B

	full_text

double %481
,double8B

	full_text

double %483
mcall8Bc
a
	full_textT
R
P%485 = tail call double @llvm.fmuladd.f64(double %299, double %484, double %470)
,double8B

	full_text

double %299
,double8B

	full_text

double %484
,double8B

	full_text

double %470
Pstore8BE
C
	full_text6
4
2store double %485, double* %469, align 8, !tbaa !8
,double8B

	full_text

double %485
.double*8B

	full_text

double* %469
®getelementptr8Bî
ë
	full_textÉ
Ä
~%486 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %430, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%487 = load double, double* %486, align 8, !tbaa !8
.double*8B

	full_text

double* %486
®getelementptr8Bî
ë
	full_textÉ
Ä
~%488 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %431, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %431
Pload8BF
D
	full_text7
5
3%489 = load double, double* %488, align 8, !tbaa !8
.double*8B

	full_text

double* %488
®getelementptr8Bî
ë
	full_textÉ
Ä
~%490 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %432, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %432
Pload8BF
D
	full_text7
5
3%491 = load double, double* %490, align 8, !tbaa !8
.double*8B

	full_text

double* %490
vcall8Bl
j
	full_text]
[
Y%492 = tail call double @llvm.fmuladd.f64(double %491, double -4.000000e+00, double %489)
,double8B

	full_text

double %491
,double8B

	full_text

double %489
®getelementptr8Bî
ë
	full_textÉ
Ä
~%493 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %430, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%494 = load double, double* %493, align 8, !tbaa !8
.double*8B

	full_text

double* %493
ucall8Bk
i
	full_text\
Z
X%495 = tail call double @llvm.fmuladd.f64(double %494, double 6.000000e+00, double %492)
,double8B

	full_text

double %494
,double8B

	full_text

double %492
®getelementptr8Bî
ë
	full_textÉ
Ä
~%496 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %433, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %433
Pload8BF
D
	full_text7
5
3%497 = load double, double* %496, align 8, !tbaa !8
.double*8B

	full_text

double* %496
vcall8Bl
j
	full_text]
[
Y%498 = tail call double @llvm.fmuladd.f64(double %497, double -4.000000e+00, double %495)
,double8B

	full_text

double %497
,double8B

	full_text

double %495
®getelementptr8Bî
ë
	full_textÉ
Ä
~%499 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %434, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%500 = load double, double* %499, align 8, !tbaa !8
.double*8B

	full_text

double* %499
:fadd8B0
.
	full_text!

%501 = fadd double %498, %500
,double8B

	full_text

double %498
,double8B

	full_text

double %500
mcall8Bc
a
	full_textT
R
P%502 = tail call double @llvm.fmuladd.f64(double %299, double %501, double %487)
,double8B

	full_text

double %299
,double8B

	full_text

double %501
,double8B

	full_text

double %487
Pstore8BE
C
	full_text6
4
2store double %502, double* %486, align 8, !tbaa !8
,double8B

	full_text

double %502
.double*8B

	full_text

double* %486
®getelementptr8Bî
ë
	full_textÉ
Ä
~%503 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %430, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%504 = load double, double* %503, align 8, !tbaa !8
.double*8B

	full_text

double* %503
®getelementptr8Bî
ë
	full_textÉ
Ä
~%505 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %431, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %431
Pload8BF
D
	full_text7
5
3%506 = load double, double* %505, align 8, !tbaa !8
.double*8B

	full_text

double* %505
®getelementptr8Bî
ë
	full_textÉ
Ä
~%507 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %432, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %432
Pload8BF
D
	full_text7
5
3%508 = load double, double* %507, align 8, !tbaa !8
.double*8B

	full_text

double* %507
vcall8Bl
j
	full_text]
[
Y%509 = tail call double @llvm.fmuladd.f64(double %508, double -4.000000e+00, double %506)
,double8B

	full_text

double %508
,double8B

	full_text

double %506
®getelementptr8Bî
ë
	full_textÉ
Ä
~%510 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %430, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %430
Pload8BF
D
	full_text7
5
3%511 = load double, double* %510, align 8, !tbaa !8
.double*8B

	full_text

double* %510
ucall8Bk
i
	full_text\
Z
X%512 = tail call double @llvm.fmuladd.f64(double %511, double 6.000000e+00, double %509)
,double8B

	full_text

double %511
,double8B

	full_text

double %509
®getelementptr8Bî
ë
	full_textÉ
Ä
~%513 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %433, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %433
Pload8BF
D
	full_text7
5
3%514 = load double, double* %513, align 8, !tbaa !8
.double*8B

	full_text

double* %513
vcall8Bl
j
	full_text]
[
Y%515 = tail call double @llvm.fmuladd.f64(double %514, double -4.000000e+00, double %512)
,double8B

	full_text

double %514
,double8B

	full_text

double %512
®getelementptr8Bî
ë
	full_textÉ
Ä
~%516 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %434, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %434
Pload8BF
D
	full_text7
5
3%517 = load double, double* %516, align 8, !tbaa !8
.double*8B

	full_text

double* %516
:fadd8B0
.
	full_text!

%518 = fadd double %515, %517
,double8B

	full_text

double %515
,double8B

	full_text

double %517
mcall8Bc
a
	full_textT
R
P%519 = tail call double @llvm.fmuladd.f64(double %299, double %518, double %504)
,double8B

	full_text

double %299
,double8B

	full_text

double %518
,double8B

	full_text

double %504
Pstore8BE
C
	full_text6
4
2store double %519, double* %503, align 8, !tbaa !8
,double8B

	full_text

double %519
.double*8B

	full_text

double* %503
:icmp8B0
.
	full_text!

%520 = icmp eq i64 %433, %428
&i648B

	full_text


i64 %433
&i648B

	full_text


i64 %428
=br8B5
3
	full_text&
$
"br i1 %520, label %521, label %429
$i18B

	full_text
	
i1 %520
8sext8B.
,
	full_text

%522 = sext i32 %425 to i64
&i328B

	full_text


i32 %425
5add8B,
*
	full_text

%523 = add nsw i32 %3, -5
8sext8B.
,
	full_text

%524 = sext i32 %523 to i64
&i328B

	full_text


i32 %523
5add8B,
*
	full_text

%525 = add nsw i32 %3, -4
8sext8B.
,
	full_text

%526 = sext i32 %525 to i64
&i328B

	full_text


i32 %525
5add8B,
*
	full_text

%527 = add nsw i32 %3, -2
8sext8B.
,
	full_text

%528 = sext i32 %527 to i64
&i328B

	full_text


i32 %527
®getelementptr8Bî
ë
	full_textÉ
Ä
~%529 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %522, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%530 = load double, double* %529, align 8, !tbaa !8
.double*8B

	full_text

double* %529
®getelementptr8Bî
ë
	full_textÉ
Ä
~%531 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %524, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %524
Pload8BF
D
	full_text7
5
3%532 = load double, double* %531, align 8, !tbaa !8
.double*8B

	full_text

double* %531
®getelementptr8Bî
ë
	full_textÉ
Ä
~%533 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %526, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %526
Pload8BF
D
	full_text7
5
3%534 = load double, double* %533, align 8, !tbaa !8
.double*8B

	full_text

double* %533
vcall8Bl
j
	full_text]
[
Y%535 = tail call double @llvm.fmuladd.f64(double %534, double -4.000000e+00, double %532)
,double8B

	full_text

double %534
,double8B

	full_text

double %532
®getelementptr8Bî
ë
	full_textÉ
Ä
~%536 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %522, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%537 = load double, double* %536, align 8, !tbaa !8
.double*8B

	full_text

double* %536
ucall8Bk
i
	full_text\
Z
X%538 = tail call double @llvm.fmuladd.f64(double %537, double 6.000000e+00, double %535)
,double8B

	full_text

double %537
,double8B

	full_text

double %535
®getelementptr8Bî
ë
	full_textÉ
Ä
~%539 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %528, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%540 = load double, double* %539, align 8, !tbaa !8
.double*8B

	full_text

double* %539
vcall8Bl
j
	full_text]
[
Y%541 = tail call double @llvm.fmuladd.f64(double %540, double -4.000000e+00, double %538)
,double8B

	full_text

double %540
,double8B

	full_text

double %538
mcall8Bc
a
	full_textT
R
P%542 = tail call double @llvm.fmuladd.f64(double %299, double %541, double %530)
,double8B

	full_text

double %299
,double8B

	full_text

double %541
,double8B

	full_text

double %530
Pstore8BE
C
	full_text6
4
2store double %542, double* %529, align 8, !tbaa !8
,double8B

	full_text

double %542
.double*8B

	full_text

double* %529
®getelementptr8Bî
ë
	full_textÉ
Ä
~%543 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %528, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%544 = load double, double* %543, align 8, !tbaa !8
.double*8B

	full_text

double* %543
Pload8BF
D
	full_text7
5
3%545 = load double, double* %533, align 8, !tbaa !8
.double*8B

	full_text

double* %533
Pload8BF
D
	full_text7
5
3%546 = load double, double* %536, align 8, !tbaa !8
.double*8B

	full_text

double* %536
vcall8Bl
j
	full_text]
[
Y%547 = tail call double @llvm.fmuladd.f64(double %546, double -4.000000e+00, double %545)
,double8B

	full_text

double %546
,double8B

	full_text

double %545
Pload8BF
D
	full_text7
5
3%548 = load double, double* %539, align 8, !tbaa !8
.double*8B

	full_text

double* %539
ucall8Bk
i
	full_text\
Z
X%549 = tail call double @llvm.fmuladd.f64(double %548, double 5.000000e+00, double %547)
,double8B

	full_text

double %548
,double8B

	full_text

double %547
mcall8Bc
a
	full_textT
R
P%550 = tail call double @llvm.fmuladd.f64(double %299, double %549, double %544)
,double8B

	full_text

double %299
,double8B

	full_text

double %549
,double8B

	full_text

double %544
Pstore8BE
C
	full_text6
4
2store double %550, double* %543, align 8, !tbaa !8
,double8B

	full_text

double %550
.double*8B

	full_text

double* %543
®getelementptr8Bî
ë
	full_textÉ
Ä
~%551 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %522, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%552 = load double, double* %551, align 8, !tbaa !8
.double*8B

	full_text

double* %551
®getelementptr8Bî
ë
	full_textÉ
Ä
~%553 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %524, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %524
Pload8BF
D
	full_text7
5
3%554 = load double, double* %553, align 8, !tbaa !8
.double*8B

	full_text

double* %553
®getelementptr8Bî
ë
	full_textÉ
Ä
~%555 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %526, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %526
Pload8BF
D
	full_text7
5
3%556 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
vcall8Bl
j
	full_text]
[
Y%557 = tail call double @llvm.fmuladd.f64(double %556, double -4.000000e+00, double %554)
,double8B

	full_text

double %556
,double8B

	full_text

double %554
®getelementptr8Bî
ë
	full_textÉ
Ä
~%558 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %522, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%559 = load double, double* %558, align 8, !tbaa !8
.double*8B

	full_text

double* %558
ucall8Bk
i
	full_text\
Z
X%560 = tail call double @llvm.fmuladd.f64(double %559, double 6.000000e+00, double %557)
,double8B

	full_text

double %559
,double8B

	full_text

double %557
®getelementptr8Bî
ë
	full_textÉ
Ä
~%561 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %528, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%562 = load double, double* %561, align 8, !tbaa !8
.double*8B

	full_text

double* %561
vcall8Bl
j
	full_text]
[
Y%563 = tail call double @llvm.fmuladd.f64(double %562, double -4.000000e+00, double %560)
,double8B

	full_text

double %562
,double8B

	full_text

double %560
mcall8Bc
a
	full_textT
R
P%564 = tail call double @llvm.fmuladd.f64(double %299, double %563, double %552)
,double8B

	full_text

double %299
,double8B

	full_text

double %563
,double8B

	full_text

double %552
Pstore8BE
C
	full_text6
4
2store double %564, double* %551, align 8, !tbaa !8
,double8B

	full_text

double %564
.double*8B

	full_text

double* %551
®getelementptr8Bî
ë
	full_textÉ
Ä
~%565 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %528, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%566 = load double, double* %565, align 8, !tbaa !8
.double*8B

	full_text

double* %565
Pload8BF
D
	full_text7
5
3%567 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
Pload8BF
D
	full_text7
5
3%568 = load double, double* %558, align 8, !tbaa !8
.double*8B

	full_text

double* %558
vcall8Bl
j
	full_text]
[
Y%569 = tail call double @llvm.fmuladd.f64(double %568, double -4.000000e+00, double %567)
,double8B

	full_text

double %568
,double8B

	full_text

double %567
Pload8BF
D
	full_text7
5
3%570 = load double, double* %561, align 8, !tbaa !8
.double*8B

	full_text

double* %561
ucall8Bk
i
	full_text\
Z
X%571 = tail call double @llvm.fmuladd.f64(double %570, double 5.000000e+00, double %569)
,double8B

	full_text

double %570
,double8B

	full_text

double %569
mcall8Bc
a
	full_textT
R
P%572 = tail call double @llvm.fmuladd.f64(double %299, double %571, double %566)
,double8B

	full_text

double %299
,double8B

	full_text

double %571
,double8B

	full_text

double %566
Pstore8BE
C
	full_text6
4
2store double %572, double* %565, align 8, !tbaa !8
,double8B

	full_text

double %572
.double*8B

	full_text

double* %565
®getelementptr8Bî
ë
	full_textÉ
Ä
~%573 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %522, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%574 = load double, double* %573, align 8, !tbaa !8
.double*8B

	full_text

double* %573
®getelementptr8Bî
ë
	full_textÉ
Ä
~%575 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %524, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %524
Pload8BF
D
	full_text7
5
3%576 = load double, double* %575, align 8, !tbaa !8
.double*8B

	full_text

double* %575
®getelementptr8Bî
ë
	full_textÉ
Ä
~%577 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %526, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %526
Pload8BF
D
	full_text7
5
3%578 = load double, double* %577, align 8, !tbaa !8
.double*8B

	full_text

double* %577
vcall8Bl
j
	full_text]
[
Y%579 = tail call double @llvm.fmuladd.f64(double %578, double -4.000000e+00, double %576)
,double8B

	full_text

double %578
,double8B

	full_text

double %576
®getelementptr8Bî
ë
	full_textÉ
Ä
~%580 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %522, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%581 = load double, double* %580, align 8, !tbaa !8
.double*8B

	full_text

double* %580
ucall8Bk
i
	full_text\
Z
X%582 = tail call double @llvm.fmuladd.f64(double %581, double 6.000000e+00, double %579)
,double8B

	full_text

double %581
,double8B

	full_text

double %579
®getelementptr8Bî
ë
	full_textÉ
Ä
~%583 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %528, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%584 = load double, double* %583, align 8, !tbaa !8
.double*8B

	full_text

double* %583
vcall8Bl
j
	full_text]
[
Y%585 = tail call double @llvm.fmuladd.f64(double %584, double -4.000000e+00, double %582)
,double8B

	full_text

double %584
,double8B

	full_text

double %582
mcall8Bc
a
	full_textT
R
P%586 = tail call double @llvm.fmuladd.f64(double %299, double %585, double %574)
,double8B

	full_text

double %299
,double8B

	full_text

double %585
,double8B

	full_text

double %574
Pstore8BE
C
	full_text6
4
2store double %586, double* %573, align 8, !tbaa !8
,double8B

	full_text

double %586
.double*8B

	full_text

double* %573
®getelementptr8Bî
ë
	full_textÉ
Ä
~%587 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %528, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%588 = load double, double* %587, align 8, !tbaa !8
.double*8B

	full_text

double* %587
Pload8BF
D
	full_text7
5
3%589 = load double, double* %577, align 8, !tbaa !8
.double*8B

	full_text

double* %577
Pload8BF
D
	full_text7
5
3%590 = load double, double* %580, align 8, !tbaa !8
.double*8B

	full_text

double* %580
vcall8Bl
j
	full_text]
[
Y%591 = tail call double @llvm.fmuladd.f64(double %590, double -4.000000e+00, double %589)
,double8B

	full_text

double %590
,double8B

	full_text

double %589
Pload8BF
D
	full_text7
5
3%592 = load double, double* %583, align 8, !tbaa !8
.double*8B

	full_text

double* %583
ucall8Bk
i
	full_text\
Z
X%593 = tail call double @llvm.fmuladd.f64(double %592, double 5.000000e+00, double %591)
,double8B

	full_text

double %592
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%594 = tail call double @llvm.fmuladd.f64(double %299, double %593, double %588)
,double8B

	full_text

double %299
,double8B

	full_text

double %593
,double8B

	full_text

double %588
Pstore8BE
C
	full_text6
4
2store double %594, double* %587, align 8, !tbaa !8
,double8B

	full_text

double %594
.double*8B

	full_text

double* %587
®getelementptr8Bî
ë
	full_textÉ
Ä
~%595 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %522, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%596 = load double, double* %595, align 8, !tbaa !8
.double*8B

	full_text

double* %595
®getelementptr8Bî
ë
	full_textÉ
Ä
~%597 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %524, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %524
Pload8BF
D
	full_text7
5
3%598 = load double, double* %597, align 8, !tbaa !8
.double*8B

	full_text

double* %597
®getelementptr8Bî
ë
	full_textÉ
Ä
~%599 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %526, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %526
Pload8BF
D
	full_text7
5
3%600 = load double, double* %599, align 8, !tbaa !8
.double*8B

	full_text

double* %599
vcall8Bl
j
	full_text]
[
Y%601 = tail call double @llvm.fmuladd.f64(double %600, double -4.000000e+00, double %598)
,double8B

	full_text

double %600
,double8B

	full_text

double %598
®getelementptr8Bî
ë
	full_textÉ
Ä
~%602 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %522, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%603 = load double, double* %602, align 8, !tbaa !8
.double*8B

	full_text

double* %602
ucall8Bk
i
	full_text\
Z
X%604 = tail call double @llvm.fmuladd.f64(double %603, double 6.000000e+00, double %601)
,double8B

	full_text

double %603
,double8B

	full_text

double %601
®getelementptr8Bî
ë
	full_textÉ
Ä
~%605 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %528, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%606 = load double, double* %605, align 8, !tbaa !8
.double*8B

	full_text

double* %605
vcall8Bl
j
	full_text]
[
Y%607 = tail call double @llvm.fmuladd.f64(double %606, double -4.000000e+00, double %604)
,double8B

	full_text

double %606
,double8B

	full_text

double %604
mcall8Bc
a
	full_textT
R
P%608 = tail call double @llvm.fmuladd.f64(double %299, double %607, double %596)
,double8B

	full_text

double %299
,double8B

	full_text

double %607
,double8B

	full_text

double %596
Pstore8BE
C
	full_text6
4
2store double %608, double* %595, align 8, !tbaa !8
,double8B

	full_text

double %608
.double*8B

	full_text

double* %595
®getelementptr8Bî
ë
	full_textÉ
Ä
~%609 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %528, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%610 = load double, double* %609, align 8, !tbaa !8
.double*8B

	full_text

double* %609
Pload8BF
D
	full_text7
5
3%611 = load double, double* %599, align 8, !tbaa !8
.double*8B

	full_text

double* %599
Pload8BF
D
	full_text7
5
3%612 = load double, double* %602, align 8, !tbaa !8
.double*8B

	full_text

double* %602
vcall8Bl
j
	full_text]
[
Y%613 = tail call double @llvm.fmuladd.f64(double %612, double -4.000000e+00, double %611)
,double8B

	full_text

double %612
,double8B

	full_text

double %611
Pload8BF
D
	full_text7
5
3%614 = load double, double* %605, align 8, !tbaa !8
.double*8B

	full_text

double* %605
ucall8Bk
i
	full_text\
Z
X%615 = tail call double @llvm.fmuladd.f64(double %614, double 5.000000e+00, double %613)
,double8B

	full_text

double %614
,double8B

	full_text

double %613
mcall8Bc
a
	full_textT
R
P%616 = tail call double @llvm.fmuladd.f64(double %299, double %615, double %610)
,double8B

	full_text

double %299
,double8B

	full_text

double %615
,double8B

	full_text

double %610
Pstore8BE
C
	full_text6
4
2store double %616, double* %609, align 8, !tbaa !8
,double8B

	full_text

double %616
.double*8B

	full_text

double* %609
®getelementptr8Bî
ë
	full_textÉ
Ä
~%617 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %522, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%618 = load double, double* %617, align 8, !tbaa !8
.double*8B

	full_text

double* %617
®getelementptr8Bî
ë
	full_textÉ
Ä
~%619 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %524, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %524
Pload8BF
D
	full_text7
5
3%620 = load double, double* %619, align 8, !tbaa !8
.double*8B

	full_text

double* %619
®getelementptr8Bî
ë
	full_textÉ
Ä
~%621 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %526, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %526
Pload8BF
D
	full_text7
5
3%622 = load double, double* %621, align 8, !tbaa !8
.double*8B

	full_text

double* %621
vcall8Bl
j
	full_text]
[
Y%623 = tail call double @llvm.fmuladd.f64(double %622, double -4.000000e+00, double %620)
,double8B

	full_text

double %622
,double8B

	full_text

double %620
®getelementptr8Bî
ë
	full_textÉ
Ä
~%624 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %522, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %522
Pload8BF
D
	full_text7
5
3%625 = load double, double* %624, align 8, !tbaa !8
.double*8B

	full_text

double* %624
ucall8Bk
i
	full_text\
Z
X%626 = tail call double @llvm.fmuladd.f64(double %625, double 6.000000e+00, double %623)
,double8B

	full_text

double %625
,double8B

	full_text

double %623
®getelementptr8Bî
ë
	full_textÉ
Ä
~%627 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %19, i64 %295, i64 %294, i64 %528, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %19
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%628 = load double, double* %627, align 8, !tbaa !8
.double*8B

	full_text

double* %627
vcall8Bl
j
	full_text]
[
Y%629 = tail call double @llvm.fmuladd.f64(double %628, double -4.000000e+00, double %626)
,double8B

	full_text

double %628
,double8B

	full_text

double %626
mcall8Bc
a
	full_textT
R
P%630 = tail call double @llvm.fmuladd.f64(double %299, double %629, double %618)
,double8B

	full_text

double %299
,double8B

	full_text

double %629
,double8B

	full_text

double %618
Pstore8BE
C
	full_text6
4
2store double %630, double* %617, align 8, !tbaa !8
,double8B

	full_text

double %630
.double*8B

	full_text

double* %617
®getelementptr8Bî
ë
	full_textÉ
Ä
~%631 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %20, i64 %295, i64 %294, i64 %528, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %20
&i648B

	full_text


i64 %295
&i648B

	full_text


i64 %294
&i648B

	full_text


i64 %528
Pload8BF
D
	full_text7
5
3%632 = load double, double* %631, align 8, !tbaa !8
.double*8B

	full_text

double* %631
Pload8BF
D
	full_text7
5
3%633 = load double, double* %621, align 8, !tbaa !8
.double*8B

	full_text

double* %621
Pload8BF
D
	full_text7
5
3%634 = load double, double* %624, align 8, !tbaa !8
.double*8B

	full_text

double* %624
vcall8Bl
j
	full_text]
[
Y%635 = tail call double @llvm.fmuladd.f64(double %634, double -4.000000e+00, double %633)
,double8B

	full_text

double %634
,double8B

	full_text

double %633
Pload8BF
D
	full_text7
5
3%636 = load double, double* %627, align 8, !tbaa !8
.double*8B

	full_text

double* %627
ucall8Bk
i
	full_text\
Z
X%637 = tail call double @llvm.fmuladd.f64(double %636, double 5.000000e+00, double %635)
,double8B

	full_text

double %636
,double8B

	full_text

double %635
mcall8Bc
a
	full_textT
R
P%638 = tail call double @llvm.fmuladd.f64(double %299, double %637, double %632)
,double8B

	full_text

double %299
,double8B

	full_text

double %637
,double8B

	full_text

double %632
Pstore8BE
C
	full_text6
4
2store double %638, double* %631, align 8, !tbaa !8
,double8B

	full_text

double %638
.double*8B

	full_text

double* %631
(br8B 

	full_text

br label %639
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %5
$i328B
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
-; undefined function B

	full_text

 
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 4.000000e-01
$i648B

	full_text


i64 -2
$i328B

	full_text


i32 -5
4double8B&
$
	full_text

double 1.050000e+01
:double8B,
*
	full_text

double 0x4019333333333334
4double8B&
$
	full_text

double 8.400000e+01
5double8B'
%
	full_text

double -4.000000e+00
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 1.400000e+00
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 7.500000e-01
:double8B,
*
	full_text

double 0x405EDEB851EB851E
5double8B'
%
	full_text

double -3.150000e+01
#i648B

	full_text	

i64 0
%i18B

	full_text


i1 false
%i328B

	full_text
	
i32 320
#i648B

	full_text	

i64 1
:double8B,
*
	full_text

double 0x40A7418000000001
:double8B,
*
	full_text

double 0xC03E3D70A3D70A3B
4double8B&
$
	full_text

double 6.300000e+01
4double8B&
$
	full_text

double 4.000000e+00
4double8B&
$
	full_text

double 5.000000e+00
(i328B

	full_text


i32 -20800
4double8B&
$
	full_text

double 6.000000e+00
$i328B

	full_text


i32 -4
#i328B

	full_text	

i32 6
$i328B

	full_text


i32 -2
5double8B'
%
	full_text

double -2.000000e+00
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 2
$i648B

	full_text


i64 -1
4double8B&
$
	full_text

double 5.000000e-01
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 2.500000e-01
$i328B

	full_text


i32 -3
4double8B&
$
	full_text

double 1.000000e+00        	
 		                       !" !! #$ ## %& %% '( '' )) *+ *- ,, ./ .. 01 00 23 22 44 57 66 89 8: 8; 8< 88 => == ?@ ?? AB AC AA DE DD FG FH FF IJ II KL KM KN KO KK PQ PP RS RT RR UV UW UX UY UU Z[ ZZ \] \^ \\ _` _a _b __ cd ce cf cg cc hi hh jk jl jm jj no nn pq pr pp st su sv sw ss xy xx z{ z| zz }~ }} Ä 	Å 	Ç  ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä ââ ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë îï îî ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú ü† üü °¢ °° £
§ ££ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂∏ ππ ∫ª ∫Ω ºº æø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆
» «« …  …… ÀÃ ÀÀ ÕŒ Õ
œ Õ
– Õ
— ÕÕ “” ““ ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È Á
Í Á
Î ÁÁ ÏÌ ÏÏ ÓÔ Ó
 ÓÓ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É Å
Ñ Å
Ö ÅÅ Üá ÜÜ àâ à
ä àà ãå ãã çé ç
è çç êë êê íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õ
û õ
ü õõ †° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µ
∏ µ
π µµ ∫ª ∫∫ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
’ ‘‘ ÷÷ ◊ÿ ◊⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·· ‚
‰ „„ ÂÊ Â
Á Â
Ë Â
È ÂÂ ÍÎ ÍÍ Ï
Ì ÏÏ ÓÔ Ó
 Ó
Ò Ó
Ú ÓÓ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯
˚ ¯
¸ ¯¯ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà áá âä â
ã ââ åç å
é å
è å
ê åå ëí ëë ìî ì
ï ìì ñó ññ òô ò
ö ò
õ ò
ú òò ùû ùù ü
† üü °¢ °
£ °
§ °
• °° ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´
Ø ´´ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µ
∏ µ
π µµ ∫ª ∫∫ ºΩ º
æ ºº ø¿ ø
¡ ø
¬ ø
√ øø ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …… ÃÕ ÃÃ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô Ì
 ÌÌ ÒÚ Ò
Û Ò
Ù ÒÒ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯
˚ ¯¯ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Ü
á ÜÜ àâ à
ä à
ã àà åç åå éè é
ê éé ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢• §ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨¨ Æ∞ ØØ ±≤ ±± ≥¥ ≥≥ µ∂ µµ ∑∏ ∑∑ π
ª ∫∫ ºΩ º
æ º
ø º
¿ ºº ¡¬ ¡¡ √ƒ √√ ≈∆ ≈
« ≈
» ≈
… ≈≈  À    ÃÕ Ã
Œ Ã
œ Ã
– ÃÃ —“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿ
€ ÿ
‹ ÿÿ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í Ë
Î Ë
Ï ËË ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇ
É ˇˇ ÑÖ ÑÑ Üá Ü
à Ü
â Ü
ä ÜÜ ãå ãã çé ç
è çç êë ê
í ê
ì ê
î êê ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †
£ †
§ †† •¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑
∫ ∑
ª ∑∑ ºΩ ºº æø æ
¿ æ
¡ æ
¬ ææ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »
À »
Ã »» ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿ
€ ÿ
‹ ÿÿ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰
Ê ‰‰ ÁË ÁÁ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò Ô
Ú Ô
Û ÔÔ Ùı ÙÙ ˆ˜ ˆ
¯ ˆ
˘ ˆ
˙ ˆˆ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç Ä
É Ä
Ñ ÄÄ ÖÜ ÖÖ áà á
â áá äã ä
å ää çé ç
è çç êë ê
í ê
ì ê
î êê ïñ ïï óò ó
ô óó öõ öö úù ú
û úú ü† üü °¢ °
£ °° §• §
¶ §§ ß® ß
© ß
™ ß
´ ßß ¨≠ ¨¨ ÆØ Æ
∞ Æ
± Æ
≤ ÆÆ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏
º ∏∏ Ωæ ΩΩ ø¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ ÀŒ Õ
œ ÕÕ –— –
“ –– ”” ‘’ ‘‘ ÷◊ ÷÷ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰‰ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ ÍÍ ÏÌ ÏÏ Ó
Ô ÓÓ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ãã çé çç èê è
ë èè íì íí îï î
ñ îî óò ó
ô ó
ö óó õú õõ ùû ù
ü ùù †° †
¢ †
£ †† §• §
¶ §§ ß® ß
© ß
™ ßß ´¨ ´´ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ ππ ª
º ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …
Ã …… ÕŒ Õ
œ ÕÕ –— –
“ –
” –– ‘’ ‘‘ ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰
Á ‰‰ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô Ì
 ÌÌ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ Ù
˜ ÙÙ ¯˘ ¯¯ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛˛ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ ÑÑ Üá ÜÜ à
â àà äã ä
å ää çé ç
è ç
ê çç ëí ëë ìî ì
ï ìì ñó ñ
ò ñ
ô ññ öõ ö
ú öö ùû ù
ü ù
† ùù °¢ °° £§ ££ •¶ •• ß® ßß ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫
Ω ∫∫ æø æ
¿ ææ ¡¬ ¡
√ ¡
ƒ ¡¡ ≈∆ ≈≈ «» «
… «
  «« ÀÃ ÀÀ ÕŒ Õ
œ Õ
– ÕÕ —“ —— ”‘ ”” ’
÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „
Â „
Ê „„ ÁË Á
È ÁÁ ÍÎ Í
Ï Í
Ì ÍÍ ÓÔ ÓÓ Ò  ÚÛ ÚÚ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä	 ˛
Å	 ˛˛ Ç	É	 Ç	Ç	 Ñ	Ö	 Ñ	
Ü	 Ñ	Ñ	 á	à	 á	
â	 á	
ä	 á	á	 ã	å	 ã	
ç	 ã	ã	 é	è	 é	
ê	 é	
ë	 é	é	 í	ì	 í	í	 î	ï	 î	
ñ	 î	
ó	 î	î	 ò	ô	 ò	ò	 ö	õ	 ö	
ú	 ö	
ù	 ö	ö	 û	ü	 û	û	 †	°	 †	†	 ¢	
£	 ¢	¢	 §	•	 §	
¶	 §	§	 ß	®	 ß	
©	 ß	
™	 ß	ß	 ´	¨	 ´	´	 ≠	Æ	 ≠	
Ø	 ≠	≠	 ∞	±	 ∞	
≤	 ∞	
≥	 ∞	∞	 ¥	µ	 ¥	
∂	 ¥	¥	 ∑	∏	 ∑	
π	 ∑	
∫	 ∑	∑	 ª	º	 ª	ª	 Ω	æ	 Ω	Ω	 ø	¿	 ø	ø	 ¡	¬	 ¡	¡	 √	ƒ	 √	
≈	 √	√	 ∆	«	 ∆	∆	 »	…	 »	
 	 »	»	 À	Ã	 À	
Õ	 À	
Œ	 À	À	 œ	–	 œ	œ	 —	“	 —	
”	 —	—	 ‘	’	 ‘	
÷	 ‘	
◊	 ‘	‘	 ÿ	Ÿ	 ÿ	
⁄	 ÿ	ÿ	 €	€	 ‹	‹	 ›	ﬁ	 ›	‡	 ﬂ	ﬂ	 ·	
„	 ‚	‚	 ‰	Â	 ‰	‰	 Ê	Á	 Ê	Ê	 Ë	È	 Ë	Ë	 Í	Î	 Í	Í	 Ï	Ì	 Ï	
Ó	 Ï	
Ô	 Ï	
	 Ï	Ï	 Ò	Ú	 Ò	Ò	 Û	Ù	 Û	
ı	 Û	
ˆ	 Û	
˜	 Û	Û	 ¯	˘	 ¯	¯	 ˙	˚	 ˙	
¸	 ˙	
˝	 ˙	
˛	 ˙	˙	 ˇ	Ä
 ˇ	ˇ	 Å
Ç
 Å

É
 Å
Å
 Ñ
Ö
 Ñ

Ü
 Ñ

á
 Ñ

à
 Ñ
Ñ
 â
ä
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

ê
 é

ë
 é

í
 é
é
 ì
î
 ì
ì
 ï
ñ
 ï

ó
 ï
ï
 ò
ô
 ò

ö
 ò

õ
 ò

ú
 ò
ò
 ù
û
 ù
ù
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

•
 ¢
¢
 ¶
ß
 ¶

®
 ¶
¶
 ©
™
 ©

´
 ©

¨
 ©

≠
 ©
©
 Æ
Ø
 Æ
Æ
 ∞
±
 ∞

≤
 ∞

≥
 ∞

¥
 ∞
∞
 µ
∂
 µ
µ
 ∑
∏
 ∑

π
 ∑

∫
 ∑

ª
 ∑
∑
 º
Ω
 º
º
 æ
ø
 æ

¿
 æ
æ
 ¡
¬
 ¡

√
 ¡

ƒ
 ¡

≈
 ¡
¡
 ∆
«
 ∆
∆
 »
…
 »

 
 »
»
 À
Ã
 À

Õ
 À

Œ
 À

œ
 À
À
 –
—
 –
–
 “
”
 “

‘
 “
“
 ’
÷
 ’

◊
 ’

ÿ
 ’

Ÿ
 ’
’
 ⁄
€
 ⁄
⁄
 ‹
›
 ‹

ﬁ
 ‹
‹
 ﬂ
‡
 ﬂ

·
 ﬂ

‚
 ﬂ
ﬂ
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

Ë
 Ê

È
 Ê

Í
 Ê
Ê
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


 Ì

Ò
 Ì
Ì
 Ú
Û
 Ú
Ú
 Ù
ı
 Ù

ˆ
 Ù

˜
 Ù

¯
 Ù
Ù
 ˘
˙
 ˘
˘
 ˚
¸
 ˚

˝
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
 ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä à
ã à
å àà çé çç èê è
ë èè íì í
î í
ï í
ñ íí óò óó ôö ô
õ ôô úù ú
û ú
ü úú †° †
¢ †† £§ £
• £
¶ £
ß ££ ®© ®® ™´ ™
¨ ™
≠ ™
Æ ™™ Ø∞ ØØ ±≤ ±
≥ ±
¥ ±
µ ±± ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ª
æ ª
ø ªª ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈
» ≈
… ≈≈  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œ
” œœ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ Ÿ
‹ ŸŸ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡
‰ ‡‡ ÂÊ ÂÂ ÁË Á
È Á
Í Á
Î ÁÁ ÏÌ ÏÏ ÓÔ Ó
 Ó
Ò Ó
Ú ÓÓ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯
˚ ¯
¸ ¯¯ ˝˛ ˝˝ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ Ç
Ö Ç
Ü ÇÇ áà áá âä â
ã ââ åç å
é å
è å
ê åå ëí ëë ìî ì
ï ìì ñó ñ
ò ñ
ô ññ öõ ö
ú öö ùû ù
ü ùù †° †£ ¢¢ §§ •¶ •• ßß ®© ®® ™™ ´¨ ´´ ≠Æ ≠
Ø ≠
∞ ≠
± ≠≠ ≤≥ ≤≤ ¥µ ¥
∂ ¥
∑ ¥
∏ ¥¥ π∫ ππ ªº ª
Ω ª
æ ª
ø ªª ¿¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈
» ≈
… ≈≈  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œ
” œœ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ Ÿ
‹ ŸŸ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡
‰ ‡‡ ÂÊ ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙
˝ ˙
˛ ˙˙ ˇÄ ˇˇ ÅÇ Å
É Å
Ñ Å
Ö ÅÅ Üá ÜÜ àâ à
ä à
ã à
å àà çé çç èê è
ë èè íì í
î í
ï í
ñ íí óò óó ôö ô
õ ôô úù ú
û ú
ü ú
† úú °¢ °° £§ £
• ££ ¶ß ¶
® ¶
© ¶¶ ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠
∞ ≠
± ≠≠ ≤≥ ≤≤ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «
À «« ÃÕ ÃÃ Œœ Œ
– Œ
— Œ
“ ŒŒ ”‘ ”” ’÷ ’
◊ ’
ÿ ’
Ÿ ’’ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂ
‚ ﬂ
„ ﬂﬂ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î È
Ï È
Ì ÈÈ ÓÔ ÓÓ Ò 
Ú  ÛÙ Û
ı Û
ˆ ÛÛ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙
˝ ˙
˛ ˙˙ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ä
å ää çé ç
è ç
ê çç ëí ë
ì ëë îï î
ñ î
ó î
ò îî ôö ôô õú õ
ù õ
û õ
ü õõ †° †† ¢£ ¢
§ ¢
• ¢
¶ ¢¢ ß® ßß ©™ ©
´ ©© ¨≠ ¨
Æ ¨
Ø ¨
∞ ¨¨ ±≤ ±± ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂
π ∂
∫ ∂∂ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «
À «« ÃÕ ÃÃ Œœ ŒŒ –— –– “” “
‘ ““ ’÷ ’’ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·
‰ ·
Â ·· ÊÁ ÊÊ ËÈ Ë
Í Ë
Î Ë
Ï ËË ÌÓ ÌÌ Ô Ô
Ò Ô
Ú Ô
Û ÔÔ Ùı ÙÙ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘
¸ ˘
˝ ˘˘ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü É
á ÉÉ àâ àà äã ä
å ää çé ç
è ç
ê çç ëí ë
ì ëë îï î
ñ î
ó î
ò îî ôö ôô õú õõ ùû ùù ü† ü
° üü ¢£ ¢¢ §• §
¶ §§ ß® ß
© ß
™ ßß ´¨ ´
≠ ´´ Æ∞ ± ≤ ≥ %¥ µ )µ 4µ ∏µ πµ ÷µ ·µ €	µ ‹	µ §µ ßµ ™    
   	      	    "! $# &% () + -, / 10 3± 7 9. :2 ;6 <8 >= @' B6 CA E? GD H8 J L. M2 N6 OK QI SP T V. W2 X6 YU [Z ]Z ^I `I a\ b d. e2 f6 gc ih kh l_ mj on qP r t. u2 v6 ws yx {p |z ~I ÄR Å} Ç' Ñ6 Ö áÉ àU äR åâ ç' è6 êã íé ìc ïR óî ò' ö6 õñ ùô ûs †p ¢° §ü ¶£ ßR ©• ™' ¨6 ≠® Ø´ ∞6 ≤± ¥4 µ≥ ∑π ª Ωº ø ¡¿ √∏ ≈… »«  « Ã Œæ œ¬ –« —Õ ”' ’… ÷‘ ÿ' ⁄À €Ÿ ›◊ ﬂ‹ ‡ﬁ ‚“ „· ÂÕ Ê Ëæ È¬ Í« ÎÁ Ì' Ô… Ó Ú' ÙÀ ıÛ ˜Ò ˘ˆ ˙¯ ¸Ï ˝˚ ˇÁ Ä Çæ É¬ Ñ« ÖÅ á' â… äà å' éÀ èç ëã ìê îí ñÜ óï ôÅ ö úæ ù¬ û« üõ °' £… §¢ ¶' ®À ©ß ´• ≠™ Æ¨ ∞† ±Ø ≥õ ¥ ∂æ ∑¬ ∏« πµ ª' Ω… æº ¿' ¬À √¡ ≈ø «ƒ »∆  ∫ À… Õµ Œ… –ƒ —œ ”π ’÷ ÿ ⁄Ÿ ‹ ﬁ› ‡ù ‰ Ê€ Áﬂ Ë„ ÈÂ ÎÍ Ì Ô€ ﬂ Ò„ ÚÓ ÙÏ ˆÛ ˜ ˘€ ˙ﬂ ˚„ ¸¯ ˛Ï Ä˝ Å É€ Ñﬂ Ö„ ÜÇ àÏ äá ã ç€ éﬂ è„ êå íÏ îë ï„ ó ô€ öﬂ õñ úò ûù † ¢€ £ﬂ §ñ •° ßü ©¶ ™ ¨€ ≠ﬂ Æñ Ø´ ±ü ≥∞ ¥ ∂€ ∑ﬂ ∏ñ πµ ªü Ω∫ æ ¿€ ¡ﬂ ¬ñ √ø ≈ü «ƒ »ı  ® À… Õ' œ„ –Ã “Œ ”ˇ ’≤ ÷‘ ÿ' ⁄„ €◊ ›Ÿ ﬁâ ‡º ·ﬂ „' Â„ Ê‚ Ë‰ Èˇ Îˇ Ïı Óı ÔÍ â Úâ ÛÌ Ù≤ ˆ≤ ˜® ˘® ˙ı ˚º ˝º ˛¯ ˇÒ Å¸ Ç® Ñ® ÖÉ áı âı äÜ ãà çÄ èå êì í∆ ìë ïé ñ' ò„ ôî õó ú„ ûù †· °ü £‘ • ß¶ © ´™ ≠ ∞Ø ≤ ¥≥ ∂∏ ∏÷ ª Ω± æµ ø∫ ¿º ¬∫ ƒ ∆± «µ »√ …≈ À Õ± Œµ œ∫ –Ã “— ‘  ’∫ ◊ Ÿ± ⁄µ €÷ ‹ÿ ﬁ” ‡› ·ﬂ „¡ ‰‚ Êº Á È± Íµ Î∫ ÏË Ó' ÷ ÒÔ Û' ı∫ ˆÙ ¯Ú ˙˜ ˚˘ ˝Ì ˛ Ä± Åµ Ç√ Éˇ Ö á± àµ â∫ äÜ åã éÑ è ë± íµ ì÷ îê ñç òï ôó õ¸ úö ûË ü °± ¢µ £∫ §† ¶' ®÷ ©ß ´' ≠∫ Æ¨ ∞™ ≤Ø ≥± µ• ∂ ∏± πµ ∫√ ª∑ Ω ø± ¿µ ¡∫ ¬æ ƒ√ ∆º « …±  µ À÷ Ã» Œ≈ –Õ —œ ”¥ ‘“ ÷† ◊ Ÿ± ⁄µ €∫ ‹ÿ ﬁ' ‡÷ ·ﬂ „' Â∫ Ê‰ Ë‚ ÍÁ ÎÈ Ì› Ó ± Òµ Ú√ ÛÔ ı ˜± ¯µ ˘∫ ˙ˆ ¸˚ ˛Ù ˇ Å± Çµ É÷ ÑÄ Ü˝ àÖ âá ãÏ åä éÿ è ë± íµ ì∫ îê ñ' ò÷ ôó õ' ù∫ ûú †ö ¢ü £° •ï ¶ ®± ©µ ™√ ´ß ≠ Ø± ∞µ ±∫ ≤Æ ¥≥ ∂¨ ∑ π± ∫µ ª÷ º∏ æµ ¿Ω ¡ø √§ ƒ¬ ∆ê «÷ …∑  » Ã¨ Œµ œ® —± “” ’‘ ◊÷ Ÿ €– ‹Õ ›⁄ ﬂ ·– ‚Õ „‡ Â Á– ËÕ ÈÊ ÎÍ ÌÏ Ô‰ ÒÓ Ú Ù– ıÕ ˆÛ ¯˜ ˙ ˚ÿ ˝˘ ˛ﬁ ˇ¸ Å⁄ Ç Ñ– ÖÕ ÜÉ à‡ äÊ åã éâ êç ëÛ ìí ïè ñ ò– ôÕ öó úõ ûî üÿ °ù ¢á £† •É ¶ ®– ©Õ ™ß ¨ Æ– ØÕ ∞≠ ≤ ¥– µÕ ∂≥ ∏∑ ∫π º± æª ø ¡– ¬Õ √¿ ≈ƒ «Ω »ÿ  ∆ À´ Ã… Œß œ —– “Õ ”– ’≠ ◊≥ Ÿÿ €÷ ›⁄ ﬁ¿ ‡ﬂ ‚‹ „ Â– ÊÕ Á‰ ÈË Î· Ïÿ ÓÍ Ô‘ Ì Ú– Û ı– ˆÕ ˜Ù ˘ ˚– ¸Õ ˝˙ ˇ Å– ÇÕ ÉÄ ÖÑ áÜ â˛ ãà å é– èÕ êç íë îä ïÿ óì ò¯ ôñ õÙ ú û– üÕ †ù ¢˙ §Ä ¶• ®£ ™ß ´ç ≠¨ Ø© ∞ ≤– ≥Õ ¥± ∂µ ∏Æ πÿ ª∑ º° Ω∫ øù ¿ ¬– √Õ ƒ¡ ∆ »– …Õ  « Ã Œ– œÕ –Õ “— ‘” ÷À ÿ’ Ÿ €– ‹Õ ›⁄ ﬂﬁ ·◊ ‚ÿ ‰‡ Â≈ Ê„ Ë¡ È Î– ÏÕ ÌÍ Ô« ÒÕ ÛÚ ı ˜Ù ¯⁄ ˙˘ ¸ˆ ˝ ˇ– Ä	Õ Å	˛ É	Ç	 Ö	˚ Ü	ÿ à	Ñ	 â	Ó ä	á	 å	Í ç	 è	– ê	Õ ë	é	 ì	 ï	– ñ	Õ ó	î	 ô	 õ	– ú	Õ ù	ö	 ü	û	 °	†	 £	ò	 •	¢	 ¶	 ®	– ©	Õ ™	ß	 ¨	´	 Æ	§	 Ø	ÿ ±	≠	 ≤	í	 ≥	∞	 µ	é	 ∂	 ∏	– π	Õ ∫	∑	 º	î	 æ	ö	 ¿	ø	 ¬	Ω	 ƒ	¡	 ≈	ß	 «	∆	 …	√	  	 Ã	– Õ	Õ Œ	À	 –	œ	 “	»	 ”	ÿ ’	—	 ÷	ª	 ◊	‘	 Ÿ	∑	 ⁄	‹	 ﬁ	€	 ‡	Ë	 „	‚	 Â	‚	 Á	‚	 È	‚	 Î	 Ì	– Ó	Õ Ô	‚	 	Ï	 Ú	 Ù	– ı	Õ ˆ	‰	 ˜	Û	 ˘	 ˚	– ¸	Õ ˝	Ê	 ˛	˙	 Ä
ˇ	 Ç
¯	 É
 Ö
– Ü
Õ á
‚	 à
Ñ
 ä
â
 å
Å
 ç
 è
– ê
Õ ë
Ë	 í
é
 î
ì
 ñ
ã
 ó
 ô
– ö
Õ õ
Í	 ú
ò
 û
ï
 †
ù
 °
ÿ £
ü
 §
Ò	 •
¢
 ß
Ï	 ®
 ™
– ´
Õ ¨
‚	 ≠
©
 Ø
 ±
– ≤
Õ ≥
‰	 ¥
∞
 ∂
 ∏
– π
Õ ∫
Ê	 ª
∑
 Ω
º
 ø
µ
 ¿
 ¬
– √
Õ ƒ
‚	 ≈
¡
 «
∆
 …
æ
  
 Ã
– Õ
Õ Œ
Ë	 œ
À
 —
–
 ”
»
 ‘
 ÷
– ◊
Õ ÿ
Í	 Ÿ
’
 €
“
 ›
⁄
 ﬁ
ÿ ‡
‹
 ·
Æ
 ‚
ﬂ
 ‰
©
 Â
 Á
– Ë
Õ È
‚	 Í
Ê
 Ï
 Ó
– Ô
Õ 
‰	 Ò
Ì
 Û
 ı
– ˆ
Õ ˜
Ê	 ¯
Ù
 ˙
˘
 ¸
Ú
 ˝
 ˇ
– ÄÕ Å‚	 Ç˛
 ÑÉ Ü˚
 á â– äÕ ãË	 åà éç êÖ ë ì– îÕ ïÍ	 ñí òè öó õÿ ùô ûÎ
 üú °Ê
 ¢ §– •Õ ¶‚	 ß£ © ´– ¨Õ ≠‰	 Æ™ ∞ ≤– ≥Õ ¥Ê	 µ± ∑∂ πØ ∫ º– ΩÕ æ‚	 øª ¡¿ √∏ ƒ ∆– «Õ »Ë	 …≈ À  Õ¬ Œ –– —Õ “Í	 ”œ ’Ã ◊‘ ÿÿ ⁄÷ €® ‹Ÿ ﬁ£ ﬂ ·– ‚Õ „‚	 ‰‡ Ê Ë– ÈÕ Í‰	 ÎÁ Ì Ô– Õ ÒÊ	 ÚÓ ÙÛ ˆÏ ˜ ˘– ˙Õ ˚‚	 ¸¯ ˛˝ Äı Å É– ÑÕ ÖË	 ÜÇ àá äˇ ã ç– éÕ èÍ	 êå íâ îë ïÿ óì òÂ ôñ õ‡ úË	 ûﬂ	 üù °€	 £§ ¶ß ©™ ¨ Æ– ØÕ ∞¢ ±≠ ≥ µ– ∂Õ ∑• ∏¥ ∫ º– ΩÕ æ® øª ¡¿ √π ƒ ∆– «Õ »¢ …≈ À  Õ¬ Œ –– —Õ “´ ”œ ’‘ ◊Ã ÿÿ ⁄÷ €≤ ‹Ÿ ﬁ≠ ﬂ ·– ‚Õ „´ ‰‡ Êª Ë≈ ÍÈ ÏÁ Ìœ ÔÓ ÒÎ Úÿ Ù ıÂ ˆÛ ¯‡ ˘ ˚– ¸Õ ˝¢ ˛˙ Ä Ç– ÉÕ Ñ• ÖÅ á â– äÕ ã® åà éç êÜ ë ì– îÕ ï¢ ñí òó öè õ ù– ûÕ ü´ †ú ¢° §ô •ÿ ß£ ®ˇ ©¶ ´˙ ¨ Æ– ØÕ ∞´ ±≠ ≥à µí ∑∂ π¥ ∫ú ºª æ∏ øÿ ¡Ω ¬≤ √¿ ≈≠ ∆ »– …Õ  ¢ À« Õ œ– –Õ —• “Œ ‘ ÷– ◊Õ ÿ® Ÿ’ €⁄ ›” ﬁ ‡– ·Õ ‚¢ „ﬂ Â‰ Á‹ Ë Í– ÎÕ Ï´ ÌÈ ÔÓ ÒÊ Úÿ Ù ıÃ ˆÛ ¯« ˘ ˚– ¸Õ ˝´ ˛˙ Ä’ Çﬂ ÑÉ ÜÅ áÈ âà ãÖ åÿ éä èˇ êç í˙ ì ï– ñÕ ó¢ òî ö ú– ùÕ û• üõ ° £– §Õ •® ¶¢ ®ß ™† ´ ≠– ÆÕ Ø¢ ∞¨ ≤± ¥© µ ∑– ∏Õ π´ ∫∂ ºª æ≥ øÿ ¡Ω ¬ô √¿ ≈î ∆ »– …Õ  ´ À« Õ¢ œ¨ —– ”Œ ‘∂ ÷’ ÿ“ Ÿÿ €◊ ‹Ã ›⁄ ﬂ« ‡ ‚– „Õ ‰¢ Â· Á È– ÍÕ Î• ÏË Ó – ÒÕ Ú® ÛÔ ıÙ ˜Ì ¯ ˙– ˚Õ ¸¢ ˝˘ ˇ˛ Åˆ Ç Ñ– ÖÕ Ü´ áÉ âà ãÄ åÿ éä èÊ êç í· ì ï– ñÕ ó´ òî öÔ ú˘ ûù †õ °É £¢ •ü ¶ÿ ®§ ©ô ™ß ¨î ≠  Ø* ,* ¶5 6Æ Õ∂ ∏∂ 6›	 ﬂ	›	 ¢∫ º∫ ‘·	 ‚	Æ Ø∆ «◊ Ÿ◊ §† ¢† ‚	“ ‘“ «‚ „§ Ø§ ¶¢ §¢ „π ∫À ÕÀ ∫ ∏∏ ∑∑ ∂∂ Øè ∑∑ è◊ ∑∑ ◊Ê ∑∑ Êˆ ∑∑ ˆ˚
 ∑∑ ˚
¬ ∑∑ ¬ ∂∂ î ∑∑ î≈ ∑∑ ≈∏ ∑∑ ∏ñ ∑∑ ñ ∑∑ â ∑∑ â≥ ∑∑ ≥Ì ∑∑ Ìç ∑∑ çç ∑∑ ç‚ ∑∑ ‚Û ∑∑ Û“
 ∑∑ “
• ∑∑ •è ∑∑ è‹ ∑∑ ‹µ ∑∑ µ¸ ∑∑ ¸Ä ∑∑ Ä‘	 ∑∑ ‘	‘ ∏∏ ‘· ∑∑ ·÷ ∑∑ ÷˝ ∑∑ ˝á	 ∑∑ á	© ∑∑ ©… ∑∑ …¢
 ∑∑ ¢
Ω ∑∑ Ωà ∑∑ àä ∑∑ äÌ ∑∑ Ì§ ∑∑ §” ∑∑ ”æ
 ∑∑ æ
ã
 ∑∑ ã
è ∑∑ èü ∑∑ üÖ ∑∑ Ö§ ∑∑ §¥ ∑∑ ¥Ÿ ∑∑ Ÿ_ ∑∑ _‹ ∑∑ ‹î ∑∑ î⁄ ∑∑ ⁄ˆ ∑∑ ˆ„ ∑∑ „»
 ∑∑ »
¬ ∑∑ ¬∫ ∑∑ ∫Î ∑∑ Î¸ ∑∑ ¸¿ ∑∑ ¿© ∑∑ ©∏ ∑∑ ∏” ∏∏ ”¸ ∑∑ ¸Ÿ ∑∑ ŸÅ
 ∑∑ Å
ú ∑∑ ú£ ∑∑ £Ã ∑∑ ÃΩ ∑∑ Ω† ∑∑ †¶ ∑∑ ¶Ï ∑∑ Ï˚ ∑∑ ˚é ∑∑ é¬ ∑∑ ¬˚ ∑∑ ˚Ø ∑∑ Øï ∑∑ ïÖ ∑∑ Ö ∑∑ ä ∑∑ äÃ ∑∑ Ã… ∑∑ …Æ ∑∑ Æj ∑∑ j§	 ∑∑ §	ä ∑∑ ä¯ ∑∑ ¯∞	 ∑∑ ∞	ﬂ
 ∑∑ ﬂ
Û ∑∑ Û◊ ∑∑ ◊Ò ∑∑ Òï
 ∑∑ ï
ä ∑∑ ä· ∑∑ · ∂∂ Ω ∑∑ Ωç ∑∑ çˇ ∑∑ ˇö ∑∑ ö»	 ∑∑ »	ñ ∑∑ ñ“ ∑∑ ““ ∑∑ “ß ∑∑ ß¿ ∑∑ ¿ô ∑∑ ô√	 ∑∑ √	ı ∑∑ ı ∑∑  ∑∑ 	π 	π 
π ∏	∫ s
∫ ´
∫ µ
∫ º
∫ ¡
∫ å
∫ ø
∫ ó
∫ ê
∫ ó
∫ ú
∫ ß
∫ Æ
∫ ∏
∫ ó
∫ ‰
∫ ±
∫ ˛
∫ é	
∫ î	
∫ ö	
∫ ß	
∫ ∑	
∫ À	
∫ À	
∫ ‡
∫ Á
∫ Ó
∫ ¯
∫ Ç
∫ å
∫ ·
∫ Ë
∫ Ô
∫ ˘
∫ É
∫ î	ª }
ª °
º ‰	
Ω §
æ å
ø ¸
ø ¥
ø Ï
ø §
¿ Ã
¡ è
¡ î
¡ ‹
¡ ·
¡ ©
¡ Æ
¡ ˆ
¡ ˚
¡ √	
¡ »	
¡ Å

¡ ï

¡ æ

¡ “

¡ ˚

¡ è
¡ ∏
¡ Ã
¡ ı
¡ â
¡ ¬
¡ ÷
¡ Î
¡ è
¡ £
¡ ∏
¡ ‹
¡ 
¡ Ö
¡ ©
¡ Ω
¡ “
¡ ˆ
¡ ä
¡ ü¬ £¬ Ü¬ ÿ¬ Ó¬ ª¬ à¬ ’¬ ¢	
√ •	ƒ U
ƒ é
ƒ Å
ƒ à
ƒ ç
ƒ ¯
ƒ ´
ƒ Ÿ
ƒ †
ƒ ß
ƒ ¨
ƒ ∑
ƒ æ
ƒ »
ƒ Ê
ƒ É
ƒ ≥
ƒ –
ƒ Ù
ƒ ˙
ƒ Ä
ƒ Ä
ƒ ç
ƒ ù
ƒ ù
ƒ ±
ƒ Õ
ƒ Í
ƒ ö	
ƒ ∑	
ƒ Í	
ƒ Ê

ƒ Ì

ƒ Ù

ƒ ˛

ƒ à
ƒ í
ƒ «
ƒ Œ
ƒ ’
ƒ ﬂ
ƒ È
ƒ ˙≈ ”
≈ ”
∆ î
« ·
« ˚
« ï
« Ø
« …» 6	» K
» Õ
» ‘
» Ÿ
» Â
» ò
» º
» ≈
» Ã
» ÿ
» ⁄
» ‡
» Ê
» Û
» É
» ó
» Ï	
» Û	
» ˙	
» Ñ

» é

» ò

» ≠
» ¥
» ª
» ≈
» œ
» ‡… ‘	  	À 	À 	À 8
À É
À ±À «
À …
À Á
À Ó
À ÛÀ „
À Ó
À °
À Œ
À ùÀ ∫
À ÷
À Ë
À Ô
À Ù
À ˇ
À Ü
À ê
À ⁄
À ‡
À ß
À ß
À ≠
À ≠
À ≥
À ¿
À –
À ‰
À Ù
À ˙
À ¡
À «
À é	
À î	
À Ë	
À ©

À ∞

À ∑

À ¡

À À

À ’

À ˙
À Å
À à
À í
À ú
À ≠
Ã ‚
Ã ö
Ã “
Ã ä
Ã ¬
Õ é
Œ ◊
Œ ‚
œ Ï
œ π
œ Ü
œ ”
œ †	
– 
– Ω
– ä
– ◊
– §	
– 
– Ω
– ä
– ◊
– §	— !
“ ç
“ ⁄
“ ß
“ Ù
“ ¡	
“ ã

“ »

“ Ö
“ ¬
“ ˇ
“ Ã
“ ô
“ Ê
“ ≥
“ Ä
” ß	‘ 
‘ ‹	
’ ™
÷ ”
÷ ç
÷ ≈
÷ ˝
÷ µ◊ 	◊ )	ÿ c
ÿ ô
ÿ õ
ÿ ¢
ÿ ß
ÿ Ç
ÿ µ
ÿ ‰
ÿ ÿ
ÿ ﬂ
ÿ ‰
ÿ Ô
ÿ ˆ
ÿ Ä
ÿ Û
ÿ ¿
ÿ ç
ÿ ¡
ÿ «
ÿ Õ
ÿ ⁄
ÿ ⁄
ÿ Í
ÿ ˛
ÿ ß	ÿ ‚	
ÿ £
ÿ ™
ÿ ±
ÿ ª
ÿ ≈
ÿ œ
ÿ î
ÿ õ
ÿ ¢
ÿ ¨
ÿ ∂
ÿ «
Ÿ π
⁄ À
⁄ ñ
⁄ √
⁄ Ê		€ n‹ 
‹ ÷	› ,	› .	› 0	› 2
› º
› æ
› ¿
› ¬
› Ÿ
› €
› ›
› ﬂ
› ¶
› ®
› ™
› ¨
› Ø
› ±
› ≥
› µ
ﬁ ÷
ﬂ €	‡ Ï‡ ü
‡ ‘"
erhs2"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z3maxdd*à
npb-LU-erhs2.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä
 
transfer_bytes_log1p
ΩaçA

transfer_bytes
®ˇ»

devmap_label
 

wgsize
>

wgsize_log1p
ΩaçA