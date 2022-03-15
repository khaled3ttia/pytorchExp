

[external]
@allocaB6
4
	full_text'
%
#%10 = alloca [5 x double], align 16
DbitcastB9
7
	full_text*
(
&%11 = bitcast [5 x double]* %10 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %10
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %11) #5
#i8*B

	full_text
	
i8* %11
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 1) #6
.addB'
%
	full_text

%13 = add i64 %12, 1
#i64B

	full_text
	
i64 %12
6truncB-
+
	full_text

%14 = trunc i64 %13 to i32
#i64B

	full_text
	
i64 %13
LcallBD
B
	full_text5
3
1%15 = tail call i64 @_Z13get_global_idj(i32 0) #6
.addB'
%
	full_text

%16 = add i64 %15, 1
#i64B

	full_text
	
i64 %15
6truncB-
+
	full_text

%17 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
2addB+
)
	full_text

%18 = add nsw i32 %7, -2
6icmpB.
,
	full_text

%19 = icmp slt i32 %18, %14
#i32B

	full_text
	
i32 %18
#i32B

	full_text
	
i32 %14
2addB+
)
	full_text

%20 = add nsw i32 %6, -2
6icmpB.
,
	full_text

%21 = icmp slt i32 %20, %17
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %17
-orB'
%
	full_text

%22 = or i1 %19, %21
!i1B

	full_text


i1 %19
!i1B

	full_text


i1 %21
9brB3
1
	full_text$
"
 br i1 %22, label %433, label %23
!i1B

	full_text


i1 %22
5add8B,
*
	full_text

%24 = add nsw i32 %14, -1
%i328B

	full_text
	
i32 %14
5mul8B,
*
	full_text

%25 = mul nsw i32 %24, %6
%i328B

	full_text
	
i32 %24
5add8B,
*
	full_text

%26 = add nsw i32 %17, -1
%i328B

	full_text
	
i32 %17
6add8B-
+
	full_text

%27 = add nsw i32 %26, %25
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %25
5mul8B,
*
	full_text

%28 = mul nsw i32 %27, 12
%i328B

	full_text
	
i32 %27
1mul8B(
&
	full_text

%29 = mul i32 %27, 60
%i328B

	full_text
	
i32 %27
Wbitcast8BJ
H
	full_text;
9
7%30 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
6sext8B,
*
	full_text

%31 = sext i32 %29 to i64
%i328B

	full_text
	
i32 %29
^getelementptr8BK
I
	full_text<
:
8%32 = getelementptr inbounds double, double* %1, i64 %31
%i648B

	full_text
	
i64 %31
Jbitcast8B=
;
	full_text.
,
*%33 = bitcast double* %32 to [5 x double]*
-double*8B

	full_text

double* %32
^getelementptr8BK
I
	full_text<
:
8%34 = getelementptr inbounds double, double* %2, i64 %31
%i648B

	full_text
	
i64 %31
Jbitcast8B=
;
	full_text.
,
*%35 = bitcast double* %34 to [5 x double]*
-double*8B

	full_text

double* %34
6sext8B,
*
	full_text

%36 = sext i32 %28 to i64
%i328B

	full_text
	
i32 %28
^getelementptr8BK
I
	full_text<
:
8%37 = getelementptr inbounds double, double* %3, i64 %36
%i648B

	full_text
	
i64 %36
^getelementptr8BK
I
	full_text<
:
8%38 = getelementptr inbounds double, double* %4, i64 %36
%i648B

	full_text
	
i64 %36
=sitofp8B1
/
	full_text"
 
%39 = sitofp i32 %14 to double
%i328B

	full_text
	
i32 %14
Ffmul8B<
:
	full_text-
+
)%40 = fmul double %39, 0x3FB745D1745D1746
+double8B

	full_text


double %39
=sitofp8B1
/
	full_text"
 
%41 = sitofp i32 %17 to double
%i328B

	full_text
	
i32 %17
Ffmul8B<
:
	full_text-
+
)%42 = fmul double %41, 0x3FB745D1745D1746
+double8B

	full_text


double %41
5icmp8B+
)
	full_text

%43 = icmp sgt i32 %8, 0
:br8B2
0
	full_text#
!
br i1 %43, label %46, label %44
#i18B

	full_text


i1 %43
4add8B+
)
	full_text

%45 = add nsw i32 %8, -2
(br8B 

	full_text

br label %100
pgetelementptr8B]
[
	full_textN
L
J%47 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
pgetelementptr8B]
[
	full_textN
L
J%48 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
pgetelementptr8B]
[
	full_textN
L
J%49 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
pgetelementptr8B]
[
	full_textN
L
J%50 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
pgetelementptr8B]
[
	full_textN
L
J%51 = getelementptr inbounds [5 x double], [5 x double]* %10, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %10
5zext8B+
)
	full_text

%52 = zext i32 %8 to i64
'br8B

	full_text

br label %53
Bphi8B9
7
	full_text*
(
&%54 = phi i64 [ 0, %46 ], [ %95, %53 ]
%i648B

	full_text
	
i64 %95
8mul8B/
-
	full_text 

%55 = mul nuw nsw i64 %54, 5
%i648B

	full_text
	
i64 %54
6add8B-
+
	full_text

%56 = add nsw i64 %55, %31
%i648B

	full_text
	
i64 %55
%i648B

	full_text
	
i64 %31
Ugetelementptr8BB
@
	full_text3
1
/%57 = getelementptr double, double* %1, i64 %56
%i648B

	full_text
	
i64 %56
@bitcast8B3
1
	full_text$
"
 %58 = bitcast double* %57 to i8*
-double*8B

	full_text

double* %57
8trunc8B-
+
	full_text

%59 = trunc i64 %54 to i32
%i648B

	full_text
	
i64 %54
=sitofp8B1
/
	full_text"
 
%60 = sitofp i32 %59 to double
%i328B

	full_text
	
i32 %59
Ffmul8B<
:
	full_text-
+
)%61 = fmul double %60, 0x3FB745D1745D1746
+double8B

	full_text


double %60
~call8Bt
r
	full_texte
c
acall void @exact_solution(double %42, double %40, double %61, double* nonnull %47, double* %5) #5
+double8B

	full_text


double %42
+double8B

	full_text


double %40
+double8B

	full_text


double %61
-double*8B

	full_text

double* %47
ucall8Bk
i
	full_text\
Z
Xcall void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %58, i8* align 8 %11, i64 40, i1 false)
%i8*8B

	full_text
	
i8* %58
%i8*8B

	full_text
	
i8* %11
Oload8BE
C
	full_text6
4
2%62 = load double, double* %47, align 16, !tbaa !8
-double*8B

	full_text

double* %47
@fdiv8B6
4
	full_text'
%
#%63 = fdiv double 1.000000e+00, %62
+double8B

	full_text


double %62
Nload8BD
B
	full_text5
3
1%64 = load double, double* %48, align 8, !tbaa !8
-double*8B

	full_text

double* %48
7fmul8B-
+
	full_text

%65 = fmul double %63, %64
+double8B

	full_text


double %63
+double8B

	full_text


double %64
rgetelementptr8B_
]
	full_textP
N
L%66 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %54, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %65, double* %66, align 8, !tbaa !8
+double8B

	full_text


double %65
-double*8B

	full_text

double* %66
Oload8BE
C
	full_text6
4
2%67 = load double, double* %49, align 16, !tbaa !8
-double*8B

	full_text

double* %49
7fmul8B-
+
	full_text

%68 = fmul double %63, %67
+double8B

	full_text


double %63
+double8B

	full_text


double %67
rgetelementptr8B_
]
	full_textP
N
L%69 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %54, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %68, double* %69, align 8, !tbaa !8
+double8B

	full_text


double %68
-double*8B

	full_text

double* %69
Nload8BD
B
	full_text5
3
1%70 = load double, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
7fmul8B-
+
	full_text

%71 = fmul double %63, %70
+double8B

	full_text


double %63
+double8B

	full_text


double %70
rgetelementptr8B_
]
	full_textP
N
L%72 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %54, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %71, double* %72, align 8, !tbaa !8
+double8B

	full_text


double %71
-double*8B

	full_text

double* %72
Oload8BE
C
	full_text6
4
2%73 = load double, double* %51, align 16, !tbaa !8
-double*8B

	full_text

double* %51
7fmul8B-
+
	full_text

%74 = fmul double %63, %73
+double8B

	full_text


double %63
+double8B

	full_text


double %73
rgetelementptr8B_
]
	full_textP
N
L%75 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %54, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %74, double* %75, align 8, !tbaa !8
+double8B

	full_text


double %74
-double*8B

	full_text

double* %75
7fmul8B-
+
	full_text

%76 = fmul double %71, %71
+double8B

	full_text


double %71
+double8B

	full_text


double %71
_getelementptr8BL
J
	full_text=
;
9%77 = getelementptr inbounds double, double* %37, i64 %54
-double*8B

	full_text

double* %37
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %76, double* %77, align 8, !tbaa !8
+double8B

	full_text


double %76
-double*8B

	full_text

double* %77
Nload8BD
B
	full_text5
3
1%78 = load double, double* %66, align 8, !tbaa !8
-double*8B

	full_text

double* %66
dcall8BZ
X
	full_textK
I
G%79 = call double @llvm.fmuladd.f64(double %78, double %78, double %76)
+double8B

	full_text


double %78
+double8B

	full_text


double %78
+double8B

	full_text


double %76
Nload8BD
B
	full_text5
3
1%80 = load double, double* %69, align 8, !tbaa !8
-double*8B

	full_text

double* %69
dcall8BZ
X
	full_textK
I
G%81 = call double @llvm.fmuladd.f64(double %80, double %80, double %79)
+double8B

	full_text


double %80
+double8B

	full_text


double %80
+double8B

	full_text


double %79
rgetelementptr8B_
]
	full_textP
N
L%82 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %54, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %81, double* %82, align 8, !tbaa !8
+double8B

	full_text


double %81
-double*8B

	full_text

double* %82
rgetelementptr8B_
]
	full_textP
N
L%83 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %54, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
%i648B

	full_text
	
i64 %54
Nload8BD
B
	full_text5
3
1%84 = load double, double* %83, align 8, !tbaa !8
-double*8B

	full_text

double* %83
rgetelementptr8B_
]
	full_textP
N
L%85 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %54, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
%i648B

	full_text
	
i64 %54
Nload8BD
B
	full_text5
3
1%86 = load double, double* %85, align 8, !tbaa !8
-double*8B

	full_text

double* %85
7fmul8B-
+
	full_text

%87 = fmul double %80, %86
+double8B

	full_text


double %80
+double8B

	full_text


double %86
dcall8BZ
X
	full_textK
I
G%88 = call double @llvm.fmuladd.f64(double %78, double %84, double %87)
+double8B

	full_text


double %78
+double8B

	full_text


double %84
+double8B

	full_text


double %87
Nload8BD
B
	full_text5
3
1%89 = load double, double* %72, align 8, !tbaa !8
-double*8B

	full_text

double* %72
rgetelementptr8B_
]
	full_textP
N
L%90 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %54, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
%i648B

	full_text
	
i64 %54
Nload8BD
B
	full_text5
3
1%91 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
dcall8BZ
X
	full_textK
I
G%92 = call double @llvm.fmuladd.f64(double %89, double %91, double %88)
+double8B

	full_text


double %89
+double8B

	full_text


double %91
+double8B

	full_text


double %88
@fmul8B6
4
	full_text'
%
#%93 = fmul double %92, 5.000000e-01
+double8B

	full_text


double %92
_getelementptr8BL
J
	full_text=
;
9%94 = getelementptr inbounds double, double* %38, i64 %54
-double*8B

	full_text

double* %38
%i648B

	full_text
	
i64 %54
Nstore8BC
A
	full_text4
2
0store double %93, double* %94, align 8, !tbaa !8
+double8B

	full_text


double %93
-double*8B

	full_text

double* %94
8add8B/
-
	full_text 

%95 = add nuw nsw i64 %54, 1
%i648B

	full_text
	
i64 %54
7icmp8B-
+
	full_text

%96 = icmp eq i64 %95, %52
%i648B

	full_text
	
i64 %95
%i648B

	full_text
	
i64 %52
:br8B2
0
	full_text#
!
br i1 %96, label %97, label %53
#i18B

	full_text


i1 %96
4add8B+
)
	full_text

%98 = add nsw i32 %8, -2
5icmp8B+
)
	full_text

%99 = icmp slt i32 %8, 3
<br8B4
2
	full_text%
#
!br i1 %99, label %100, label %106
#i18B

	full_text


i1 %99
Ephi8B<
:
	full_text-
+
)%101 = phi i32 [ %45, %44 ], [ %98, %97 ]
%i328B

	full_text
	
i32 %45
%i328B

	full_text
	
i32 %98
2shl8B)
'
	full_text

%102 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
;ashr8B1
/
	full_text"
 
%103 = ashr exact i64 %102, 32
&i648B

	full_text


i64 %102
2shl8B)
'
	full_text

%104 = shl i64 %16, 32
%i648B

	full_text
	
i64 %16
;ashr8B1
/
	full_text"
 
%105 = ashr exact i64 %104, 32
&i648B

	full_text


i64 %104
(br8B 

	full_text

br label %271
2shl8B)
'
	full_text

%107 = shl i64 %13, 32
%i648B

	full_text
	
i64 %13
;ashr8B1
/
	full_text"
 
%108 = ashr exact i64 %107, 32
&i648B

	full_text


i64 %107
2shl8B)
'
	full_text

%109 = shl i64 %16, 32
%i648B

	full_text
	
i64 %16
;ashr8B1
/
	full_text"
 
%110 = ashr exact i64 %109, 32
&i648B

	full_text


i64 %109
1add8B(
&
	full_text

%111 = add i32 %8, -1
8zext8B.
,
	full_text

%112 = zext i32 %111 to i64
&i328B

	full_text


i32 %111
(br8B 

	full_text

br label %113
Fphi8B=
;
	full_text.
,
*%114 = phi i64 [ %116, %113 ], [ 1, %106 ]
&i648B

	full_text


i64 %116
7add8B.
,
	full_text

%115 = add nsw i64 %114, -1
&i648B

	full_text


i64 %114
:add8B1
/
	full_text"
 
%116 = add nuw nsw i64 %114, 1
&i648B

	full_text


i64 %114
¨getelementptr8B”
‘
	full_textƒ
€
~%117 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %114, i64 %108, i64 %110, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %108
&i648B

	full_text


i64 %110
Pload8BF
D
	full_text7
5
3%118 = load double, double* %117, align 8, !tbaa !8
.double*8B

	full_text

double* %117
tgetelementptr8Ba
_
	full_textR
P
N%119 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %116, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%120 = load double, double* %119, align 8, !tbaa !8
.double*8B

	full_text

double* %119
tgetelementptr8Ba
_
	full_textR
P
N%121 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %115, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%122 = load double, double* %121, align 8, !tbaa !8
.double*8B

	full_text

double* %121
:fsub8B0
.
	full_text!

%123 = fsub double %120, %122
,double8B

	full_text

double %120
,double8B

	full_text

double %122
qcall8Bg
e
	full_textX
V
T%124 = call double @llvm.fmuladd.f64(double %123, double -5.500000e+00, double %118)
,double8B

	full_text

double %123
,double8B

	full_text

double %118
tgetelementptr8Ba
_
	full_textR
P
N%125 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %116, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%126 = load double, double* %125, align 8, !tbaa !8
.double*8B

	full_text

double* %125
tgetelementptr8Ba
_
	full_textR
P
N%127 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %114, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%128 = load double, double* %127, align 8, !tbaa !8
.double*8B

	full_text

double* %127
qcall8Bg
e
	full_textX
V
T%129 = call double @llvm.fmuladd.f64(double %128, double -2.000000e+00, double %126)
,double8B

	full_text

double %128
,double8B

	full_text

double %126
tgetelementptr8Ba
_
	full_textR
P
N%130 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %115, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%131 = load double, double* %130, align 8, !tbaa !8
.double*8B

	full_text

double* %130
:fadd8B0
.
	full_text!

%132 = fadd double %129, %131
,double8B

	full_text

double %129
,double8B

	full_text

double %131
pcall8Bf
d
	full_textW
U
S%133 = call double @llvm.fmuladd.f64(double %132, double 1.210000e+02, double %124)
,double8B

	full_text

double %132
,double8B

	full_text

double %124
Pstore8BE
C
	full_text6
4
2store double %133, double* %117, align 8, !tbaa !8
,double8B

	full_text

double %133
.double*8B

	full_text

double* %117
¨getelementptr8B”
‘
	full_textƒ
€
~%134 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %114, i64 %108, i64 %110, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %108
&i648B

	full_text


i64 %110
Pload8BF
D
	full_text7
5
3%135 = load double, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
tgetelementptr8Ba
_
	full_textR
P
N%136 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %116, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%137 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
tgetelementptr8Ba
_
	full_textR
P
N%138 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %116, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%139 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
tgetelementptr8Ba
_
	full_textR
P
N%140 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %115, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%141 = load double, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
tgetelementptr8Ba
_
	full_textR
P
N%142 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %115, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%143 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
:fmul8B0
.
	full_text!

%144 = fmul double %141, %143
,double8B

	full_text

double %141
,double8B

	full_text

double %143
Cfsub8B9
7
	full_text*
(
&%145 = fsub double -0.000000e+00, %144
,double8B

	full_text

double %144
hcall8B^
\
	full_textO
M
K%146 = call double @llvm.fmuladd.f64(double %137, double %139, double %145)
,double8B

	full_text

double %137
,double8B

	full_text

double %139
,double8B

	full_text

double %145
qcall8Bg
e
	full_textX
V
T%147 = call double @llvm.fmuladd.f64(double %146, double -5.500000e+00, double %135)
,double8B

	full_text

double %146
,double8B

	full_text

double %135
tgetelementptr8Ba
_
	full_textR
P
N%148 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %116, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%149 = load double, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
tgetelementptr8Ba
_
	full_textR
P
N%150 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %114, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%151 = load double, double* %150, align 8, !tbaa !8
.double*8B

	full_text

double* %150
qcall8Bg
e
	full_textX
V
T%152 = call double @llvm.fmuladd.f64(double %151, double -2.000000e+00, double %149)
,double8B

	full_text

double %151
,double8B

	full_text

double %149
tgetelementptr8Ba
_
	full_textR
P
N%153 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %115, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%154 = load double, double* %153, align 8, !tbaa !8
.double*8B

	full_text

double* %153
:fadd8B0
.
	full_text!

%155 = fadd double %152, %154
,double8B

	full_text

double %152
,double8B

	full_text

double %154
vcall8Bl
j
	full_text]
[
Y%156 = call double @llvm.fmuladd.f64(double %155, double 0x4028333333333334, double %147)
,double8B

	full_text

double %155
,double8B

	full_text

double %147
tgetelementptr8Ba
_
	full_textR
P
N%157 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %114, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%158 = load double, double* %157, align 8, !tbaa !8
.double*8B

	full_text

double* %157
qcall8Bg
e
	full_textX
V
T%159 = call double @llvm.fmuladd.f64(double %158, double -2.000000e+00, double %137)
,double8B

	full_text

double %158
,double8B

	full_text

double %137
:fadd8B0
.
	full_text!

%160 = fadd double %141, %159
,double8B

	full_text

double %141
,double8B

	full_text

double %159
pcall8Bf
d
	full_textW
U
S%161 = call double @llvm.fmuladd.f64(double %160, double 1.210000e+02, double %156)
,double8B

	full_text

double %160
,double8B

	full_text

double %156
Pstore8BE
C
	full_text6
4
2store double %161, double* %134, align 8, !tbaa !8
,double8B

	full_text

double %161
.double*8B

	full_text

double* %134
¨getelementptr8B”
‘
	full_textƒ
€
~%162 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %114, i64 %108, i64 %110, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %108
&i648B

	full_text


i64 %110
Pload8BF
D
	full_text7
5
3%163 = load double, double* %162, align 8, !tbaa !8
.double*8B

	full_text

double* %162
tgetelementptr8Ba
_
	full_textR
P
N%164 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %116, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%165 = load double, double* %164, align 8, !tbaa !8
.double*8B

	full_text

double* %164
Pload8BF
D
	full_text7
5
3%166 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
tgetelementptr8Ba
_
	full_textR
P
N%167 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %115, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%168 = load double, double* %167, align 8, !tbaa !8
.double*8B

	full_text

double* %167
Pload8BF
D
	full_text7
5
3%169 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
:fmul8B0
.
	full_text!

%170 = fmul double %168, %169
,double8B

	full_text

double %168
,double8B

	full_text

double %169
Cfsub8B9
7
	full_text*
(
&%171 = fsub double -0.000000e+00, %170
,double8B

	full_text

double %170
hcall8B^
\
	full_textO
M
K%172 = call double @llvm.fmuladd.f64(double %165, double %166, double %171)
,double8B

	full_text

double %165
,double8B

	full_text

double %166
,double8B

	full_text

double %171
qcall8Bg
e
	full_textX
V
T%173 = call double @llvm.fmuladd.f64(double %172, double -5.500000e+00, double %163)
,double8B

	full_text

double %172
,double8B

	full_text

double %163
tgetelementptr8Ba
_
	full_textR
P
N%174 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %116, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%175 = load double, double* %174, align 8, !tbaa !8
.double*8B

	full_text

double* %174
tgetelementptr8Ba
_
	full_textR
P
N%176 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %114, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%177 = load double, double* %176, align 8, !tbaa !8
.double*8B

	full_text

double* %176
qcall8Bg
e
	full_textX
V
T%178 = call double @llvm.fmuladd.f64(double %177, double -2.000000e+00, double %175)
,double8B

	full_text

double %177
,double8B

	full_text

double %175
tgetelementptr8Ba
_
	full_textR
P
N%179 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %115, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%180 = load double, double* %179, align 8, !tbaa !8
.double*8B

	full_text

double* %179
:fadd8B0
.
	full_text!

%181 = fadd double %178, %180
,double8B

	full_text

double %178
,double8B

	full_text

double %180
vcall8Bl
j
	full_text]
[
Y%182 = call double @llvm.fmuladd.f64(double %181, double 0x4028333333333334, double %173)
,double8B

	full_text

double %181
,double8B

	full_text

double %173
tgetelementptr8Ba
_
	full_textR
P
N%183 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %114, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%184 = load double, double* %183, align 8, !tbaa !8
.double*8B

	full_text

double* %183
qcall8Bg
e
	full_textX
V
T%185 = call double @llvm.fmuladd.f64(double %184, double -2.000000e+00, double %165)
,double8B

	full_text

double %184
,double8B

	full_text

double %165
:fadd8B0
.
	full_text!

%186 = fadd double %168, %185
,double8B

	full_text

double %168
,double8B

	full_text

double %185
pcall8Bf
d
	full_textW
U
S%187 = call double @llvm.fmuladd.f64(double %186, double 1.210000e+02, double %182)
,double8B

	full_text

double %186
,double8B

	full_text

double %182
Pstore8BE
C
	full_text6
4
2store double %187, double* %162, align 8, !tbaa !8
,double8B

	full_text

double %187
.double*8B

	full_text

double* %162
¨getelementptr8B”
‘
	full_textƒ
€
~%188 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %114, i64 %108, i64 %110, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %108
&i648B

	full_text


i64 %110
Pload8BF
D
	full_text7
5
3%189 = load double, double* %188, align 8, !tbaa !8
.double*8B

	full_text

double* %188
Pload8BF
D
	full_text7
5
3%190 = load double, double* %119, align 8, !tbaa !8
.double*8B

	full_text

double* %119
Pload8BF
D
	full_text7
5
3%191 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
tgetelementptr8Ba
_
	full_textR
P
N%192 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %116, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%193 = load double, double* %192, align 8, !tbaa !8
.double*8B

	full_text

double* %192
agetelementptr8BN
L
	full_text?
=
;%194 = getelementptr inbounds double, double* %38, i64 %116
-double*8B

	full_text

double* %38
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%195 = load double, double* %194, align 8, !tbaa !8
.double*8B

	full_text

double* %194
:fsub8B0
.
	full_text!

%196 = fsub double %193, %195
,double8B

	full_text

double %193
,double8B

	full_text

double %195
Bfmul8B8
6
	full_text)
'
%%197 = fmul double %196, 4.000000e-01
,double8B

	full_text

double %196
hcall8B^
\
	full_textO
M
K%198 = call double @llvm.fmuladd.f64(double %190, double %191, double %197)
,double8B

	full_text

double %190
,double8B

	full_text

double %191
,double8B

	full_text

double %197
Pload8BF
D
	full_text7
5
3%199 = load double, double* %121, align 8, !tbaa !8
.double*8B

	full_text

double* %121
Pload8BF
D
	full_text7
5
3%200 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
tgetelementptr8Ba
_
	full_textR
P
N%201 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %115, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%202 = load double, double* %201, align 8, !tbaa !8
.double*8B

	full_text

double* %201
agetelementptr8BN
L
	full_text?
=
;%203 = getelementptr inbounds double, double* %38, i64 %115
-double*8B

	full_text

double* %38
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%204 = load double, double* %203, align 8, !tbaa !8
.double*8B

	full_text

double* %203
:fsub8B0
.
	full_text!

%205 = fsub double %202, %204
,double8B

	full_text

double %202
,double8B

	full_text

double %204
Bfmul8B8
6
	full_text)
'
%%206 = fmul double %205, 4.000000e-01
,double8B

	full_text

double %205
hcall8B^
\
	full_textO
M
K%207 = call double @llvm.fmuladd.f64(double %199, double %200, double %206)
,double8B

	full_text

double %199
,double8B

	full_text

double %200
,double8B

	full_text

double %206
:fsub8B0
.
	full_text!

%208 = fsub double %198, %207
,double8B

	full_text

double %198
,double8B

	full_text

double %207
qcall8Bg
e
	full_textX
V
T%209 = call double @llvm.fmuladd.f64(double %208, double -5.500000e+00, double %189)
,double8B

	full_text

double %208
,double8B

	full_text

double %189
tgetelementptr8Ba
_
	full_textR
P
N%210 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %114, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%211 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
qcall8Bg
e
	full_textX
V
T%212 = call double @llvm.fmuladd.f64(double %211, double -2.000000e+00, double %191)
,double8B

	full_text

double %211
,double8B

	full_text

double %191
:fadd8B0
.
	full_text!

%213 = fadd double %200, %212
,double8B

	full_text

double %200
,double8B

	full_text

double %212
vcall8Bl
j
	full_text]
[
Y%214 = call double @llvm.fmuladd.f64(double %213, double 0x4030222222222222, double %209)
,double8B

	full_text

double %213
,double8B

	full_text

double %209
tgetelementptr8Ba
_
	full_textR
P
N%215 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %114, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%216 = load double, double* %215, align 8, !tbaa !8
.double*8B

	full_text

double* %215
qcall8Bg
e
	full_textX
V
T%217 = call double @llvm.fmuladd.f64(double %216, double -2.000000e+00, double %190)
,double8B

	full_text

double %216
,double8B

	full_text

double %190
:fadd8B0
.
	full_text!

%218 = fadd double %199, %217
,double8B

	full_text

double %199
,double8B

	full_text

double %217
pcall8Bf
d
	full_textW
U
S%219 = call double @llvm.fmuladd.f64(double %218, double 1.210000e+02, double %214)
,double8B

	full_text

double %218
,double8B

	full_text

double %214
Pstore8BE
C
	full_text6
4
2store double %219, double* %188, align 8, !tbaa !8
,double8B

	full_text

double %219
.double*8B

	full_text

double* %188
¨getelementptr8B”
‘
	full_textƒ
€
~%220 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %114, i64 %108, i64 %110, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %114
&i648B

	full_text


i64 %108
&i648B

	full_text


i64 %110
Pload8BF
D
	full_text7
5
3%221 = load double, double* %220, align 8, !tbaa !8
.double*8B

	full_text

double* %220
Pload8BF
D
	full_text7
5
3%222 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
Pload8BF
D
	full_text7
5
3%223 = load double, double* %192, align 8, !tbaa !8
.double*8B

	full_text

double* %192
Pload8BF
D
	full_text7
5
3%224 = load double, double* %194, align 8, !tbaa !8
.double*8B

	full_text

double* %194
Bfmul8B8
6
	full_text)
'
%%225 = fmul double %224, 4.000000e-01
,double8B

	full_text

double %224
Cfsub8B9
7
	full_text*
(
&%226 = fsub double -0.000000e+00, %225
,double8B

	full_text

double %225
pcall8Bf
d
	full_textW
U
S%227 = call double @llvm.fmuladd.f64(double %223, double 1.400000e+00, double %226)
,double8B

	full_text

double %223
,double8B

	full_text

double %226
Pload8BF
D
	full_text7
5
3%228 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
Pload8BF
D
	full_text7
5
3%229 = load double, double* %201, align 8, !tbaa !8
.double*8B

	full_text

double* %201
Pload8BF
D
	full_text7
5
3%230 = load double, double* %203, align 8, !tbaa !8
.double*8B

	full_text

double* %203
Bfmul8B8
6
	full_text)
'
%%231 = fmul double %230, 4.000000e-01
,double8B

	full_text

double %230
Cfsub8B9
7
	full_text*
(
&%232 = fsub double -0.000000e+00, %231
,double8B

	full_text

double %231
pcall8Bf
d
	full_textW
U
S%233 = call double @llvm.fmuladd.f64(double %229, double 1.400000e+00, double %232)
,double8B

	full_text

double %229
,double8B

	full_text

double %232
:fmul8B0
.
	full_text!

%234 = fmul double %228, %233
,double8B

	full_text

double %228
,double8B

	full_text

double %233
Cfsub8B9
7
	full_text*
(
&%235 = fsub double -0.000000e+00, %234
,double8B

	full_text

double %234
hcall8B^
\
	full_textO
M
K%236 = call double @llvm.fmuladd.f64(double %222, double %227, double %235)
,double8B

	full_text

double %222
,double8B

	full_text

double %227
,double8B

	full_text

double %235
qcall8Bg
e
	full_textX
V
T%237 = call double @llvm.fmuladd.f64(double %236, double -5.500000e+00, double %221)
,double8B

	full_text

double %236
,double8B

	full_text

double %221
tgetelementptr8Ba
_
	full_textR
P
N%238 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %116, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%239 = load double, double* %238, align 8, !tbaa !8
.double*8B

	full_text

double* %238
tgetelementptr8Ba
_
	full_textR
P
N%240 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %114, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%241 = load double, double* %240, align 8, !tbaa !8
.double*8B

	full_text

double* %240
qcall8Bg
e
	full_textX
V
T%242 = call double @llvm.fmuladd.f64(double %241, double -2.000000e+00, double %239)
,double8B

	full_text

double %241
,double8B

	full_text

double %239
tgetelementptr8Ba
_
	full_textR
P
N%243 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %115, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%244 = load double, double* %243, align 8, !tbaa !8
.double*8B

	full_text

double* %243
:fadd8B0
.
	full_text!

%245 = fadd double %242, %244
,double8B

	full_text

double %242
,double8B

	full_text

double %244
vcall8Bl
j
	full_text]
[
Y%246 = call double @llvm.fmuladd.f64(double %245, double 0xC0173B645A1CAC07, double %237)
,double8B

	full_text

double %245
,double8B

	full_text

double %237
agetelementptr8BN
L
	full_text?
=
;%247 = getelementptr inbounds double, double* %37, i64 %116
-double*8B

	full_text

double* %37
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%248 = load double, double* %247, align 8, !tbaa !8
.double*8B

	full_text

double* %247
agetelementptr8BN
L
	full_text?
=
;%249 = getelementptr inbounds double, double* %37, i64 %114
-double*8B

	full_text

double* %37
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%250 = load double, double* %249, align 8, !tbaa !8
.double*8B

	full_text

double* %249
qcall8Bg
e
	full_textX
V
T%251 = call double @llvm.fmuladd.f64(double %250, double -2.000000e+00, double %248)
,double8B

	full_text

double %250
,double8B

	full_text

double %248
agetelementptr8BN
L
	full_text?
=
;%252 = getelementptr inbounds double, double* %37, i64 %115
-double*8B

	full_text

double* %37
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%253 = load double, double* %252, align 8, !tbaa !8
.double*8B

	full_text

double* %252
:fadd8B0
.
	full_text!

%254 = fadd double %251, %253
,double8B

	full_text

double %251
,double8B

	full_text

double %253
vcall8Bl
j
	full_text]
[
Y%255 = call double @llvm.fmuladd.f64(double %254, double 0x4000222222222222, double %246)
,double8B

	full_text

double %254
,double8B

	full_text

double %246
tgetelementptr8Ba
_
	full_textR
P
N%256 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %116, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %116
Pload8BF
D
	full_text7
5
3%257 = load double, double* %256, align 8, !tbaa !8
.double*8B

	full_text

double* %256
tgetelementptr8Ba
_
	full_textR
P
N%258 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %114, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%259 = load double, double* %258, align 8, !tbaa !8
.double*8B

	full_text

double* %258
qcall8Bg
e
	full_textX
V
T%260 = call double @llvm.fmuladd.f64(double %259, double -2.000000e+00, double %257)
,double8B

	full_text

double %259
,double8B

	full_text

double %257
tgetelementptr8Ba
_
	full_textR
P
N%261 = getelementptr inbounds [5 x double], [5 x double]* %35, i64 %115, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %35
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%262 = load double, double* %261, align 8, !tbaa !8
.double*8B

	full_text

double* %261
:fadd8B0
.
	full_text!

%263 = fadd double %260, %262
,double8B

	full_text

double %260
,double8B

	full_text

double %262
vcall8Bl
j
	full_text]
[
Y%264 = call double @llvm.fmuladd.f64(double %263, double 0x4037B74BC6A7EF9D, double %255)
,double8B

	full_text

double %263
,double8B

	full_text

double %255
tgetelementptr8Ba
_
	full_textR
P
N%265 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %114, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%266 = load double, double* %265, align 8, !tbaa !8
.double*8B

	full_text

double* %265
qcall8Bg
e
	full_textX
V
T%267 = call double @llvm.fmuladd.f64(double %266, double -2.000000e+00, double %223)
,double8B

	full_text

double %266
,double8B

	full_text

double %223
:fadd8B0
.
	full_text!

%268 = fadd double %229, %267
,double8B

	full_text

double %229
,double8B

	full_text

double %267
pcall8Bf
d
	full_textW
U
S%269 = call double @llvm.fmuladd.f64(double %268, double 1.210000e+02, double %264)
,double8B

	full_text

double %268
,double8B

	full_text

double %264
Pstore8BE
C
	full_text6
4
2store double %269, double* %220, align 8, !tbaa !8
,double8B

	full_text

double %269
.double*8B

	full_text

double* %220
:icmp8B0
.
	full_text!

%270 = icmp eq i64 %116, %112
&i648B

	full_text


i64 %116
&i648B

	full_text


i64 %112
=br8B5
3
	full_text&
$
"br i1 %270, label %271, label %113
$i18B

	full_text
	
i1 %270
Hphi8	B?
=
	full_text0
.
,%272 = phi i32 [ %101, %100 ], [ %98, %113 ]
&i328	B

	full_text


i32 %101
%i328	B

	full_text
	
i32 %98
Iphi8	B@
>
	full_text1
/
-%273 = phi i64 [ %105, %100 ], [ %110, %113 ]
&i648	B

	full_text


i64 %105
&i648	B

	full_text


i64 %110
Iphi8	B@
>
	full_text1
/
-%274 = phi i64 [ %103, %100 ], [ %108, %113 ]
&i648	B

	full_text


i64 %103
&i648	B

	full_text


i64 %108
_getelementptr8	BL
J
	full_text=
;
9%275 = getelementptr inbounds double, double* %0, i64 845
Zbitcast8	BM
K
	full_text>
<
:%276 = bitcast double* %275 to [13 x [13 x [5 x double]]]*
.double*8	B

	full_text

double* %275
^getelementptr8	BK
I
	full_text<
:
8%277 = getelementptr inbounds double, double* %32, i64 5
-double*8	B

	full_text

double* %32
_getelementptr8	BL
J
	full_text=
;
9%278 = getelementptr inbounds double, double* %32, i64 10
-double*8	B

	full_text

double* %32
_getelementptr8	BL
J
	full_text=
;
9%279 = getelementptr inbounds double, double* %32, i64 15
-double*8	B

	full_text

double* %32
`getelementptr8	BM
K
	full_text>
<
:%280 = getelementptr inbounds double, double* %0, i64 1690
Zbitcast8	BM
K
	full_text>
<
:%281 = bitcast double* %280 to [13 x [13 x [5 x double]]]*
.double*8	B

	full_text

double* %280
_getelementptr8	BL
J
	full_text=
;
9%282 = getelementptr inbounds double, double* %32, i64 20
-double*8	B

	full_text

double* %32
¥getelementptr8	B‘
Ž
	full_text€
~
|%283 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %276, i64 0, i64 %274, i64 %273, i64 0
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %276
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%284 = load double, double* %283, align 8, !tbaa !8
.double*8	B

	full_text

double* %283
Pload8	BF
D
	full_text7
5
3%285 = load double, double* %277, align 8, !tbaa !8
.double*8	B

	full_text

double* %277
Pload8	BF
D
	full_text7
5
3%286 = load double, double* %278, align 8, !tbaa !8
.double*8	B

	full_text

double* %278
Bfmul8	B8
6
	full_text)
'
%%287 = fmul double %286, 4.000000e+00
,double8	B

	full_text

double %286
Cfsub8	B9
7
	full_text*
(
&%288 = fsub double -0.000000e+00, %287
,double8	B

	full_text

double %287
pcall8	Bf
d
	full_textW
U
S%289 = call double @llvm.fmuladd.f64(double %285, double 5.000000e+00, double %288)
,double8	B

	full_text

double %285
,double8	B

	full_text

double %288
Pload8	BF
D
	full_text7
5
3%290 = load double, double* %279, align 8, !tbaa !8
.double*8	B

	full_text

double* %279
:fadd8	B0
.
	full_text!

%291 = fadd double %290, %289
,double8	B

	full_text

double %290
,double8	B

	full_text

double %289
qcall8	Bg
e
	full_textX
V
T%292 = call double @llvm.fmuladd.f64(double %291, double -2.500000e-01, double %284)
,double8	B

	full_text

double %291
,double8	B

	full_text

double %284
Pstore8	BE
C
	full_text6
4
2store double %292, double* %283, align 8, !tbaa !8
,double8	B

	full_text

double %292
.double*8	B

	full_text

double* %283
¥getelementptr8	B‘
Ž
	full_text€
~
|%293 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %281, i64 0, i64 %274, i64 %273, i64 0
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %281
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%294 = load double, double* %293, align 8, !tbaa !8
.double*8	B

	full_text

double* %293
Pload8	BF
D
	full_text7
5
3%295 = load double, double* %277, align 8, !tbaa !8
.double*8	B

	full_text

double* %277
Pload8	BF
D
	full_text7
5
3%296 = load double, double* %278, align 8, !tbaa !8
.double*8	B

	full_text

double* %278
Bfmul8	B8
6
	full_text)
'
%%297 = fmul double %296, 6.000000e+00
,double8	B

	full_text

double %296
qcall8	Bg
e
	full_textX
V
T%298 = call double @llvm.fmuladd.f64(double %295, double -4.000000e+00, double %297)
,double8	B

	full_text

double %295
,double8	B

	full_text

double %297
Pload8	BF
D
	full_text7
5
3%299 = load double, double* %279, align 8, !tbaa !8
.double*8	B

	full_text

double* %279
qcall8	Bg
e
	full_textX
V
T%300 = call double @llvm.fmuladd.f64(double %299, double -4.000000e+00, double %298)
,double8	B

	full_text

double %299
,double8	B

	full_text

double %298
Pload8	BF
D
	full_text7
5
3%301 = load double, double* %282, align 8, !tbaa !8
.double*8	B

	full_text

double* %282
:fadd8	B0
.
	full_text!

%302 = fadd double %301, %300
,double8	B

	full_text

double %301
,double8	B

	full_text

double %300
qcall8	Bg
e
	full_textX
V
T%303 = call double @llvm.fmuladd.f64(double %302, double -2.500000e-01, double %294)
,double8	B

	full_text

double %302
,double8	B

	full_text

double %294
Pstore8	BE
C
	full_text6
4
2store double %303, double* %293, align 8, !tbaa !8
,double8	B

	full_text

double %303
.double*8	B

	full_text

double* %293
¥getelementptr8	B‘
Ž
	full_text€
~
|%304 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %276, i64 0, i64 %274, i64 %273, i64 1
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %276
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%305 = load double, double* %304, align 8, !tbaa !8
.double*8	B

	full_text

double* %304
_getelementptr8	BL
J
	full_text=
;
9%306 = getelementptr inbounds double, double* %277, i64 1
.double*8	B

	full_text

double* %277
Pload8	BF
D
	full_text7
5
3%307 = load double, double* %306, align 8, !tbaa !8
.double*8	B

	full_text

double* %306
_getelementptr8	BL
J
	full_text=
;
9%308 = getelementptr inbounds double, double* %278, i64 1
.double*8	B

	full_text

double* %278
Pload8	BF
D
	full_text7
5
3%309 = load double, double* %308, align 8, !tbaa !8
.double*8	B

	full_text

double* %308
Bfmul8	B8
6
	full_text)
'
%%310 = fmul double %309, 4.000000e+00
,double8	B

	full_text

double %309
Cfsub8	B9
7
	full_text*
(
&%311 = fsub double -0.000000e+00, %310
,double8	B

	full_text

double %310
pcall8	Bf
d
	full_textW
U
S%312 = call double @llvm.fmuladd.f64(double %307, double 5.000000e+00, double %311)
,double8	B

	full_text

double %307
,double8	B

	full_text

double %311
_getelementptr8	BL
J
	full_text=
;
9%313 = getelementptr inbounds double, double* %279, i64 1
.double*8	B

	full_text

double* %279
Pload8	BF
D
	full_text7
5
3%314 = load double, double* %313, align 8, !tbaa !8
.double*8	B

	full_text

double* %313
:fadd8	B0
.
	full_text!

%315 = fadd double %314, %312
,double8	B

	full_text

double %314
,double8	B

	full_text

double %312
qcall8	Bg
e
	full_textX
V
T%316 = call double @llvm.fmuladd.f64(double %315, double -2.500000e-01, double %305)
,double8	B

	full_text

double %315
,double8	B

	full_text

double %305
Pstore8	BE
C
	full_text6
4
2store double %316, double* %304, align 8, !tbaa !8
,double8	B

	full_text

double %316
.double*8	B

	full_text

double* %304
¥getelementptr8	B‘
Ž
	full_text€
~
|%317 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %281, i64 0, i64 %274, i64 %273, i64 1
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %281
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%318 = load double, double* %317, align 8, !tbaa !8
.double*8	B

	full_text

double* %317
Pload8	BF
D
	full_text7
5
3%319 = load double, double* %306, align 8, !tbaa !8
.double*8	B

	full_text

double* %306
Pload8	BF
D
	full_text7
5
3%320 = load double, double* %308, align 8, !tbaa !8
.double*8	B

	full_text

double* %308
Bfmul8	B8
6
	full_text)
'
%%321 = fmul double %320, 6.000000e+00
,double8	B

	full_text

double %320
qcall8	Bg
e
	full_textX
V
T%322 = call double @llvm.fmuladd.f64(double %319, double -4.000000e+00, double %321)
,double8	B

	full_text

double %319
,double8	B

	full_text

double %321
Pload8	BF
D
	full_text7
5
3%323 = load double, double* %313, align 8, !tbaa !8
.double*8	B

	full_text

double* %313
qcall8	Bg
e
	full_textX
V
T%324 = call double @llvm.fmuladd.f64(double %323, double -4.000000e+00, double %322)
,double8	B

	full_text

double %323
,double8	B

	full_text

double %322
_getelementptr8	BL
J
	full_text=
;
9%325 = getelementptr inbounds double, double* %282, i64 1
.double*8	B

	full_text

double* %282
Pload8	BF
D
	full_text7
5
3%326 = load double, double* %325, align 8, !tbaa !8
.double*8	B

	full_text

double* %325
:fadd8	B0
.
	full_text!

%327 = fadd double %326, %324
,double8	B

	full_text

double %326
,double8	B

	full_text

double %324
qcall8	Bg
e
	full_textX
V
T%328 = call double @llvm.fmuladd.f64(double %327, double -2.500000e-01, double %318)
,double8	B

	full_text

double %327
,double8	B

	full_text

double %318
Pstore8	BE
C
	full_text6
4
2store double %328, double* %317, align 8, !tbaa !8
,double8	B

	full_text

double %328
.double*8	B

	full_text

double* %317
¥getelementptr8	B‘
Ž
	full_text€
~
|%329 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %276, i64 0, i64 %274, i64 %273, i64 2
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %276
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%330 = load double, double* %329, align 8, !tbaa !8
.double*8	B

	full_text

double* %329
_getelementptr8	BL
J
	full_text=
;
9%331 = getelementptr inbounds double, double* %277, i64 2
.double*8	B

	full_text

double* %277
Pload8	BF
D
	full_text7
5
3%332 = load double, double* %331, align 8, !tbaa !8
.double*8	B

	full_text

double* %331
_getelementptr8	BL
J
	full_text=
;
9%333 = getelementptr inbounds double, double* %278, i64 2
.double*8	B

	full_text

double* %278
Pload8	BF
D
	full_text7
5
3%334 = load double, double* %333, align 8, !tbaa !8
.double*8	B

	full_text

double* %333
Bfmul8	B8
6
	full_text)
'
%%335 = fmul double %334, 4.000000e+00
,double8	B

	full_text

double %334
Cfsub8	B9
7
	full_text*
(
&%336 = fsub double -0.000000e+00, %335
,double8	B

	full_text

double %335
pcall8	Bf
d
	full_textW
U
S%337 = call double @llvm.fmuladd.f64(double %332, double 5.000000e+00, double %336)
,double8	B

	full_text

double %332
,double8	B

	full_text

double %336
_getelementptr8	BL
J
	full_text=
;
9%338 = getelementptr inbounds double, double* %279, i64 2
.double*8	B

	full_text

double* %279
Pload8	BF
D
	full_text7
5
3%339 = load double, double* %338, align 8, !tbaa !8
.double*8	B

	full_text

double* %338
:fadd8	B0
.
	full_text!

%340 = fadd double %339, %337
,double8	B

	full_text

double %339
,double8	B

	full_text

double %337
qcall8	Bg
e
	full_textX
V
T%341 = call double @llvm.fmuladd.f64(double %340, double -2.500000e-01, double %330)
,double8	B

	full_text

double %340
,double8	B

	full_text

double %330
Pstore8	BE
C
	full_text6
4
2store double %341, double* %329, align 8, !tbaa !8
,double8	B

	full_text

double %341
.double*8	B

	full_text

double* %329
¥getelementptr8	B‘
Ž
	full_text€
~
|%342 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %281, i64 0, i64 %274, i64 %273, i64 2
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %281
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%343 = load double, double* %342, align 8, !tbaa !8
.double*8	B

	full_text

double* %342
Pload8	BF
D
	full_text7
5
3%344 = load double, double* %331, align 8, !tbaa !8
.double*8	B

	full_text

double* %331
Pload8	BF
D
	full_text7
5
3%345 = load double, double* %333, align 8, !tbaa !8
.double*8	B

	full_text

double* %333
Bfmul8	B8
6
	full_text)
'
%%346 = fmul double %345, 6.000000e+00
,double8	B

	full_text

double %345
qcall8	Bg
e
	full_textX
V
T%347 = call double @llvm.fmuladd.f64(double %344, double -4.000000e+00, double %346)
,double8	B

	full_text

double %344
,double8	B

	full_text

double %346
Pload8	BF
D
	full_text7
5
3%348 = load double, double* %338, align 8, !tbaa !8
.double*8	B

	full_text

double* %338
qcall8	Bg
e
	full_textX
V
T%349 = call double @llvm.fmuladd.f64(double %348, double -4.000000e+00, double %347)
,double8	B

	full_text

double %348
,double8	B

	full_text

double %347
_getelementptr8	BL
J
	full_text=
;
9%350 = getelementptr inbounds double, double* %282, i64 2
.double*8	B

	full_text

double* %282
Pload8	BF
D
	full_text7
5
3%351 = load double, double* %350, align 8, !tbaa !8
.double*8	B

	full_text

double* %350
:fadd8	B0
.
	full_text!

%352 = fadd double %351, %349
,double8	B

	full_text

double %351
,double8	B

	full_text

double %349
qcall8	Bg
e
	full_textX
V
T%353 = call double @llvm.fmuladd.f64(double %352, double -2.500000e-01, double %343)
,double8	B

	full_text

double %352
,double8	B

	full_text

double %343
Pstore8	BE
C
	full_text6
4
2store double %353, double* %342, align 8, !tbaa !8
,double8	B

	full_text

double %353
.double*8	B

	full_text

double* %342
¥getelementptr8	B‘
Ž
	full_text€
~
|%354 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %276, i64 0, i64 %274, i64 %273, i64 3
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %276
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%355 = load double, double* %354, align 8, !tbaa !8
.double*8	B

	full_text

double* %354
_getelementptr8	BL
J
	full_text=
;
9%356 = getelementptr inbounds double, double* %277, i64 3
.double*8	B

	full_text

double* %277
Pload8	BF
D
	full_text7
5
3%357 = load double, double* %356, align 8, !tbaa !8
.double*8	B

	full_text

double* %356
_getelementptr8	BL
J
	full_text=
;
9%358 = getelementptr inbounds double, double* %278, i64 3
.double*8	B

	full_text

double* %278
Pload8	BF
D
	full_text7
5
3%359 = load double, double* %358, align 8, !tbaa !8
.double*8	B

	full_text

double* %358
Bfmul8	B8
6
	full_text)
'
%%360 = fmul double %359, 4.000000e+00
,double8	B

	full_text

double %359
Cfsub8	B9
7
	full_text*
(
&%361 = fsub double -0.000000e+00, %360
,double8	B

	full_text

double %360
pcall8	Bf
d
	full_textW
U
S%362 = call double @llvm.fmuladd.f64(double %357, double 5.000000e+00, double %361)
,double8	B

	full_text

double %357
,double8	B

	full_text

double %361
_getelementptr8	BL
J
	full_text=
;
9%363 = getelementptr inbounds double, double* %279, i64 3
.double*8	B

	full_text

double* %279
Pload8	BF
D
	full_text7
5
3%364 = load double, double* %363, align 8, !tbaa !8
.double*8	B

	full_text

double* %363
:fadd8	B0
.
	full_text!

%365 = fadd double %364, %362
,double8	B

	full_text

double %364
,double8	B

	full_text

double %362
qcall8	Bg
e
	full_textX
V
T%366 = call double @llvm.fmuladd.f64(double %365, double -2.500000e-01, double %355)
,double8	B

	full_text

double %365
,double8	B

	full_text

double %355
Pstore8	BE
C
	full_text6
4
2store double %366, double* %354, align 8, !tbaa !8
,double8	B

	full_text

double %366
.double*8	B

	full_text

double* %354
¥getelementptr8	B‘
Ž
	full_text€
~
|%367 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %281, i64 0, i64 %274, i64 %273, i64 3
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %281
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%368 = load double, double* %367, align 8, !tbaa !8
.double*8	B

	full_text

double* %367
Pload8	BF
D
	full_text7
5
3%369 = load double, double* %356, align 8, !tbaa !8
.double*8	B

	full_text

double* %356
Pload8	BF
D
	full_text7
5
3%370 = load double, double* %358, align 8, !tbaa !8
.double*8	B

	full_text

double* %358
Bfmul8	B8
6
	full_text)
'
%%371 = fmul double %370, 6.000000e+00
,double8	B

	full_text

double %370
qcall8	Bg
e
	full_textX
V
T%372 = call double @llvm.fmuladd.f64(double %369, double -4.000000e+00, double %371)
,double8	B

	full_text

double %369
,double8	B

	full_text

double %371
Pload8	BF
D
	full_text7
5
3%373 = load double, double* %363, align 8, !tbaa !8
.double*8	B

	full_text

double* %363
qcall8	Bg
e
	full_textX
V
T%374 = call double @llvm.fmuladd.f64(double %373, double -4.000000e+00, double %372)
,double8	B

	full_text

double %373
,double8	B

	full_text

double %372
_getelementptr8	BL
J
	full_text=
;
9%375 = getelementptr inbounds double, double* %282, i64 3
.double*8	B

	full_text

double* %282
Pload8	BF
D
	full_text7
5
3%376 = load double, double* %375, align 8, !tbaa !8
.double*8	B

	full_text

double* %375
:fadd8	B0
.
	full_text!

%377 = fadd double %376, %374
,double8	B

	full_text

double %376
,double8	B

	full_text

double %374
qcall8	Bg
e
	full_textX
V
T%378 = call double @llvm.fmuladd.f64(double %377, double -2.500000e-01, double %368)
,double8	B

	full_text

double %377
,double8	B

	full_text

double %368
Pstore8	BE
C
	full_text6
4
2store double %378, double* %367, align 8, !tbaa !8
,double8	B

	full_text

double %378
.double*8	B

	full_text

double* %367
¥getelementptr8	B‘
Ž
	full_text€
~
|%379 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %276, i64 0, i64 %274, i64 %273, i64 4
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %276
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%380 = load double, double* %379, align 8, !tbaa !8
.double*8	B

	full_text

double* %379
_getelementptr8	BL
J
	full_text=
;
9%381 = getelementptr inbounds double, double* %277, i64 4
.double*8	B

	full_text

double* %277
Pload8	BF
D
	full_text7
5
3%382 = load double, double* %381, align 8, !tbaa !8
.double*8	B

	full_text

double* %381
_getelementptr8	BL
J
	full_text=
;
9%383 = getelementptr inbounds double, double* %278, i64 4
.double*8	B

	full_text

double* %278
Pload8	BF
D
	full_text7
5
3%384 = load double, double* %383, align 8, !tbaa !8
.double*8	B

	full_text

double* %383
Bfmul8	B8
6
	full_text)
'
%%385 = fmul double %384, 4.000000e+00
,double8	B

	full_text

double %384
Cfsub8	B9
7
	full_text*
(
&%386 = fsub double -0.000000e+00, %385
,double8	B

	full_text

double %385
pcall8	Bf
d
	full_textW
U
S%387 = call double @llvm.fmuladd.f64(double %382, double 5.000000e+00, double %386)
,double8	B

	full_text

double %382
,double8	B

	full_text

double %386
_getelementptr8	BL
J
	full_text=
;
9%388 = getelementptr inbounds double, double* %279, i64 4
.double*8	B

	full_text

double* %279
Pload8	BF
D
	full_text7
5
3%389 = load double, double* %388, align 8, !tbaa !8
.double*8	B

	full_text

double* %388
:fadd8	B0
.
	full_text!

%390 = fadd double %389, %387
,double8	B

	full_text

double %389
,double8	B

	full_text

double %387
qcall8	Bg
e
	full_textX
V
T%391 = call double @llvm.fmuladd.f64(double %390, double -2.500000e-01, double %380)
,double8	B

	full_text

double %390
,double8	B

	full_text

double %380
Pstore8	BE
C
	full_text6
4
2store double %391, double* %379, align 8, !tbaa !8
,double8	B

	full_text

double %391
.double*8	B

	full_text

double* %379
¥getelementptr8	B‘
Ž
	full_text€
~
|%392 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %281, i64 0, i64 %274, i64 %273, i64 4
V[13 x [13 x [5 x double]]]*8	B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %281
&i648	B

	full_text


i64 %274
&i648	B

	full_text


i64 %273
Pload8	BF
D
	full_text7
5
3%393 = load double, double* %392, align 8, !tbaa !8
.double*8	B

	full_text

double* %392
Pload8	BF
D
	full_text7
5
3%394 = load double, double* %381, align 8, !tbaa !8
.double*8	B

	full_text

double* %381
Pload8	BF
D
	full_text7
5
3%395 = load double, double* %383, align 8, !tbaa !8
.double*8	B

	full_text

double* %383
Bfmul8	B8
6
	full_text)
'
%%396 = fmul double %395, 6.000000e+00
,double8	B

	full_text

double %395
qcall8	Bg
e
	full_textX
V
T%397 = call double @llvm.fmuladd.f64(double %394, double -4.000000e+00, double %396)
,double8	B

	full_text

double %394
,double8	B

	full_text

double %396
Pload8	BF
D
	full_text7
5
3%398 = load double, double* %388, align 8, !tbaa !8
.double*8	B

	full_text

double* %388
qcall8	Bg
e
	full_textX
V
T%399 = call double @llvm.fmuladd.f64(double %398, double -4.000000e+00, double %397)
,double8	B

	full_text

double %398
,double8	B

	full_text

double %397
_getelementptr8	BL
J
	full_text=
;
9%400 = getelementptr inbounds double, double* %282, i64 4
.double*8	B

	full_text

double* %282
Pload8	BF
D
	full_text7
5
3%401 = load double, double* %400, align 8, !tbaa !8
.double*8	B

	full_text

double* %400
:fadd8	B0
.
	full_text!

%402 = fadd double %401, %399
,double8	B

	full_text

double %401
,double8	B

	full_text

double %399
qcall8	Bg
e
	full_textX
V
T%403 = call double @llvm.fmuladd.f64(double %402, double -2.500000e-01, double %393)
,double8	B

	full_text

double %402
,double8	B

	full_text

double %393
Pstore8	BE
C
	full_text6
4
2store double %403, double* %392, align 8, !tbaa !8
,double8	B

	full_text

double %403
.double*8	B

	full_text

double* %392
6icmp8	B,
*
	full_text

%404 = icmp slt i32 %8, 7
1add8	B(
&
	full_text

%405 = add i32 %8, -3
=br8	B5
3
	full_text&
$
"br i1 %404, label %541, label %406
$i18	B

	full_text
	
i1 %404
8zext8
B.
,
	full_text

%407 = zext i32 %405 to i64
&i328
B

	full_text


i32 %405
(br8
B 

	full_text

br label %408
Fphi8B=
;
	full_text.
,
*%409 = phi i64 [ %422, %408 ], [ 3, %406 ]
&i648B

	full_text


i64 %422
¨getelementptr8B”
‘
	full_textƒ
€
~%410 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %409, i64 %274, i64 %273, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %409
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%411 = load double, double* %410, align 8, !tbaa !8
.double*8B

	full_text

double* %410
7add8B.
,
	full_text

%412 = add nsw i64 %409, -2
&i648B

	full_text


i64 %409
tgetelementptr8Ba
_
	full_textR
P
N%413 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %412, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %412
Pload8BF
D
	full_text7
5
3%414 = load double, double* %413, align 8, !tbaa !8
.double*8B

	full_text

double* %413
7add8B.
,
	full_text

%415 = add nsw i64 %409, -1
&i648B

	full_text


i64 %409
tgetelementptr8Ba
_
	full_textR
P
N%416 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %415, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %415
Pload8BF
D
	full_text7
5
3%417 = load double, double* %416, align 8, !tbaa !8
.double*8B

	full_text

double* %416
qcall8Bg
e
	full_textX
V
T%418 = call double @llvm.fmuladd.f64(double %417, double -4.000000e+00, double %414)
,double8B

	full_text

double %417
,double8B

	full_text

double %414
tgetelementptr8Ba
_
	full_textR
P
N%419 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %409, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %409
Pload8BF
D
	full_text7
5
3%420 = load double, double* %419, align 8, !tbaa !8
.double*8B

	full_text

double* %419
pcall8Bf
d
	full_textW
U
S%421 = call double @llvm.fmuladd.f64(double %420, double 6.000000e+00, double %418)
,double8B

	full_text

double %420
,double8B

	full_text

double %418
:add8B1
/
	full_text"
 
%422 = add nuw nsw i64 %409, 1
&i648B

	full_text


i64 %409
tgetelementptr8Ba
_
	full_textR
P
N%423 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %422, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %422
Pload8BF
D
	full_text7
5
3%424 = load double, double* %423, align 8, !tbaa !8
.double*8B

	full_text

double* %423
qcall8Bg
e
	full_textX
V
T%425 = call double @llvm.fmuladd.f64(double %424, double -4.000000e+00, double %421)
,double8B

	full_text

double %424
,double8B

	full_text

double %421
:add8B1
/
	full_text"
 
%426 = add nuw nsw i64 %409, 2
&i648B

	full_text


i64 %409
tgetelementptr8Ba
_
	full_textR
P
N%427 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %426, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %426
Pload8BF
D
	full_text7
5
3%428 = load double, double* %427, align 8, !tbaa !8
.double*8B

	full_text

double* %427
:fadd8B0
.
	full_text!

%429 = fadd double %425, %428
,double8B

	full_text

double %425
,double8B

	full_text

double %428
qcall8Bg
e
	full_textX
V
T%430 = call double @llvm.fmuladd.f64(double %429, double -2.500000e-01, double %411)
,double8B

	full_text

double %429
,double8B

	full_text

double %411
Pstore8BE
C
	full_text6
4
2store double %430, double* %410, align 8, !tbaa !8
,double8B

	full_text

double %430
.double*8B

	full_text

double* %410
:icmp8B0
.
	full_text!

%431 = icmp eq i64 %422, %407
&i648B

	full_text


i64 %422
&i648B

	full_text


i64 %407
=br8B5
3
	full_text&
$
"br i1 %431, label %432, label %408
$i18B

	full_text
	
i1 %431
=br8B5
3
	full_text&
$
"br i1 %404, label %541, label %434
$i18B

	full_text
	
i1 %404
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %11) #5
%i8*8B

	full_text
	
i8* %11
$ret8B

	full_text


ret void
8zext8B.
,
	full_text

%435 = zext i32 %405 to i64
&i328B

	full_text


i32 %405
(br8B 

	full_text

br label %436
Fphi8B=
;
	full_text.
,
*%437 = phi i64 [ %450, %436 ], [ 3, %434 ]
&i648B

	full_text


i64 %450
¨getelementptr8B”
‘
	full_textƒ
€
~%438 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %437, i64 %274, i64 %273, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %437
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%439 = load double, double* %438, align 8, !tbaa !8
.double*8B

	full_text

double* %438
7add8B.
,
	full_text

%440 = add nsw i64 %437, -2
&i648B

	full_text


i64 %437
tgetelementptr8Ba
_
	full_textR
P
N%441 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %440, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %440
Pload8BF
D
	full_text7
5
3%442 = load double, double* %441, align 8, !tbaa !8
.double*8B

	full_text

double* %441
7add8B.
,
	full_text

%443 = add nsw i64 %437, -1
&i648B

	full_text


i64 %437
tgetelementptr8Ba
_
	full_textR
P
N%444 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %443, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %443
Pload8BF
D
	full_text7
5
3%445 = load double, double* %444, align 8, !tbaa !8
.double*8B

	full_text

double* %444
qcall8Bg
e
	full_textX
V
T%446 = call double @llvm.fmuladd.f64(double %445, double -4.000000e+00, double %442)
,double8B

	full_text

double %445
,double8B

	full_text

double %442
tgetelementptr8Ba
_
	full_textR
P
N%447 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %437, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %437
Pload8BF
D
	full_text7
5
3%448 = load double, double* %447, align 8, !tbaa !8
.double*8B

	full_text

double* %447
pcall8Bf
d
	full_textW
U
S%449 = call double @llvm.fmuladd.f64(double %448, double 6.000000e+00, double %446)
,double8B

	full_text

double %448
,double8B

	full_text

double %446
:add8B1
/
	full_text"
 
%450 = add nuw nsw i64 %437, 1
&i648B

	full_text


i64 %437
tgetelementptr8Ba
_
	full_textR
P
N%451 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %450, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %450
Pload8BF
D
	full_text7
5
3%452 = load double, double* %451, align 8, !tbaa !8
.double*8B

	full_text

double* %451
qcall8Bg
e
	full_textX
V
T%453 = call double @llvm.fmuladd.f64(double %452, double -4.000000e+00, double %449)
,double8B

	full_text

double %452
,double8B

	full_text

double %449
:add8B1
/
	full_text"
 
%454 = add nuw nsw i64 %437, 2
&i648B

	full_text


i64 %437
tgetelementptr8Ba
_
	full_textR
P
N%455 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %454, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %454
Pload8BF
D
	full_text7
5
3%456 = load double, double* %455, align 8, !tbaa !8
.double*8B

	full_text

double* %455
:fadd8B0
.
	full_text!

%457 = fadd double %453, %456
,double8B

	full_text

double %453
,double8B

	full_text

double %456
qcall8Bg
e
	full_textX
V
T%458 = call double @llvm.fmuladd.f64(double %457, double -2.500000e-01, double %439)
,double8B

	full_text

double %457
,double8B

	full_text

double %439
Pstore8BE
C
	full_text6
4
2store double %458, double* %438, align 8, !tbaa !8
,double8B

	full_text

double %458
.double*8B

	full_text

double* %438
:icmp8B0
.
	full_text!

%459 = icmp eq i64 %450, %435
&i648B

	full_text


i64 %450
&i648B

	full_text


i64 %435
=br8B5
3
	full_text&
$
"br i1 %459, label %460, label %436
$i18B

	full_text
	
i1 %459
=br8B5
3
	full_text&
$
"br i1 %404, label %541, label %461
$i18B

	full_text
	
i1 %404
8zext8B.
,
	full_text

%462 = zext i32 %405 to i64
&i328B

	full_text


i32 %405
(br8B 

	full_text

br label %463
Fphi8B=
;
	full_text.
,
*%464 = phi i64 [ %477, %463 ], [ 3, %461 ]
&i648B

	full_text


i64 %477
¨getelementptr8B”
‘
	full_textƒ
€
~%465 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %464, i64 %274, i64 %273, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %464
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%466 = load double, double* %465, align 8, !tbaa !8
.double*8B

	full_text

double* %465
7add8B.
,
	full_text

%467 = add nsw i64 %464, -2
&i648B

	full_text


i64 %464
tgetelementptr8Ba
_
	full_textR
P
N%468 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %467, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %467
Pload8BF
D
	full_text7
5
3%469 = load double, double* %468, align 8, !tbaa !8
.double*8B

	full_text

double* %468
7add8B.
,
	full_text

%470 = add nsw i64 %464, -1
&i648B

	full_text


i64 %464
tgetelementptr8Ba
_
	full_textR
P
N%471 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %470, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %470
Pload8BF
D
	full_text7
5
3%472 = load double, double* %471, align 8, !tbaa !8
.double*8B

	full_text

double* %471
qcall8Bg
e
	full_textX
V
T%473 = call double @llvm.fmuladd.f64(double %472, double -4.000000e+00, double %469)
,double8B

	full_text

double %472
,double8B

	full_text

double %469
tgetelementptr8Ba
_
	full_textR
P
N%474 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %464, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %464
Pload8BF
D
	full_text7
5
3%475 = load double, double* %474, align 8, !tbaa !8
.double*8B

	full_text

double* %474
pcall8Bf
d
	full_textW
U
S%476 = call double @llvm.fmuladd.f64(double %475, double 6.000000e+00, double %473)
,double8B

	full_text

double %475
,double8B

	full_text

double %473
:add8B1
/
	full_text"
 
%477 = add nuw nsw i64 %464, 1
&i648B

	full_text


i64 %464
tgetelementptr8Ba
_
	full_textR
P
N%478 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %477, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %477
Pload8BF
D
	full_text7
5
3%479 = load double, double* %478, align 8, !tbaa !8
.double*8B

	full_text

double* %478
qcall8Bg
e
	full_textX
V
T%480 = call double @llvm.fmuladd.f64(double %479, double -4.000000e+00, double %476)
,double8B

	full_text

double %479
,double8B

	full_text

double %476
:add8B1
/
	full_text"
 
%481 = add nuw nsw i64 %464, 2
&i648B

	full_text


i64 %464
tgetelementptr8Ba
_
	full_textR
P
N%482 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %481, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %481
Pload8BF
D
	full_text7
5
3%483 = load double, double* %482, align 8, !tbaa !8
.double*8B

	full_text

double* %482
:fadd8B0
.
	full_text!

%484 = fadd double %480, %483
,double8B

	full_text

double %480
,double8B

	full_text

double %483
qcall8Bg
e
	full_textX
V
T%485 = call double @llvm.fmuladd.f64(double %484, double -2.500000e-01, double %466)
,double8B

	full_text

double %484
,double8B

	full_text

double %466
Pstore8BE
C
	full_text6
4
2store double %485, double* %465, align 8, !tbaa !8
,double8B

	full_text

double %485
.double*8B

	full_text

double* %465
:icmp8B0
.
	full_text!

%486 = icmp eq i64 %477, %462
&i648B

	full_text


i64 %477
&i648B

	full_text


i64 %462
=br8B5
3
	full_text&
$
"br i1 %486, label %487, label %463
$i18B

	full_text
	
i1 %486
=br8B5
3
	full_text&
$
"br i1 %404, label %541, label %488
$i18B

	full_text
	
i1 %404
8zext8B.
,
	full_text

%489 = zext i32 %405 to i64
&i328B

	full_text


i32 %405
(br8B 

	full_text

br label %490
Fphi8B=
;
	full_text.
,
*%491 = phi i64 [ %504, %490 ], [ 3, %488 ]
&i648B

	full_text


i64 %504
¨getelementptr8B”
‘
	full_textƒ
€
~%492 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %491, i64 %274, i64 %273, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %491
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%493 = load double, double* %492, align 8, !tbaa !8
.double*8B

	full_text

double* %492
7add8B.
,
	full_text

%494 = add nsw i64 %491, -2
&i648B

	full_text


i64 %491
tgetelementptr8Ba
_
	full_textR
P
N%495 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %494, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %494
Pload8BF
D
	full_text7
5
3%496 = load double, double* %495, align 8, !tbaa !8
.double*8B

	full_text

double* %495
7add8B.
,
	full_text

%497 = add nsw i64 %491, -1
&i648B

	full_text


i64 %491
tgetelementptr8Ba
_
	full_textR
P
N%498 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %497, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %497
Pload8BF
D
	full_text7
5
3%499 = load double, double* %498, align 8, !tbaa !8
.double*8B

	full_text

double* %498
qcall8Bg
e
	full_textX
V
T%500 = call double @llvm.fmuladd.f64(double %499, double -4.000000e+00, double %496)
,double8B

	full_text

double %499
,double8B

	full_text

double %496
tgetelementptr8Ba
_
	full_textR
P
N%501 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %491, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %491
Pload8BF
D
	full_text7
5
3%502 = load double, double* %501, align 8, !tbaa !8
.double*8B

	full_text

double* %501
pcall8Bf
d
	full_textW
U
S%503 = call double @llvm.fmuladd.f64(double %502, double 6.000000e+00, double %500)
,double8B

	full_text

double %502
,double8B

	full_text

double %500
:add8B1
/
	full_text"
 
%504 = add nuw nsw i64 %491, 1
&i648B

	full_text


i64 %491
tgetelementptr8Ba
_
	full_textR
P
N%505 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %504, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %504
Pload8BF
D
	full_text7
5
3%506 = load double, double* %505, align 8, !tbaa !8
.double*8B

	full_text

double* %505
qcall8Bg
e
	full_textX
V
T%507 = call double @llvm.fmuladd.f64(double %506, double -4.000000e+00, double %503)
,double8B

	full_text

double %506
,double8B

	full_text

double %503
:add8B1
/
	full_text"
 
%508 = add nuw nsw i64 %491, 2
&i648B

	full_text


i64 %491
tgetelementptr8Ba
_
	full_textR
P
N%509 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %508, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %508
Pload8BF
D
	full_text7
5
3%510 = load double, double* %509, align 8, !tbaa !8
.double*8B

	full_text

double* %509
:fadd8B0
.
	full_text!

%511 = fadd double %507, %510
,double8B

	full_text

double %507
,double8B

	full_text

double %510
qcall8Bg
e
	full_textX
V
T%512 = call double @llvm.fmuladd.f64(double %511, double -2.500000e-01, double %493)
,double8B

	full_text

double %511
,double8B

	full_text

double %493
Pstore8BE
C
	full_text6
4
2store double %512, double* %492, align 8, !tbaa !8
,double8B

	full_text

double %512
.double*8B

	full_text

double* %492
:icmp8B0
.
	full_text!

%513 = icmp eq i64 %504, %489
&i648B

	full_text


i64 %504
&i648B

	full_text


i64 %489
=br8B5
3
	full_text&
$
"br i1 %513, label %514, label %490
$i18B

	full_text
	
i1 %513
=br8B5
3
	full_text&
$
"br i1 %404, label %541, label %515
$i18B

	full_text
	
i1 %404
8zext8B.
,
	full_text

%516 = zext i32 %405 to i64
&i328B

	full_text


i32 %405
(br8B 

	full_text

br label %517
Fphi8B=
;
	full_text.
,
*%518 = phi i64 [ %531, %517 ], [ 3, %515 ]
&i648B

	full_text


i64 %531
¨getelementptr8B”
‘
	full_textƒ
€
~%519 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %518, i64 %274, i64 %273, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %518
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%520 = load double, double* %519, align 8, !tbaa !8
.double*8B

	full_text

double* %519
7add8B.
,
	full_text

%521 = add nsw i64 %518, -2
&i648B

	full_text


i64 %518
tgetelementptr8Ba
_
	full_textR
P
N%522 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %521, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %521
Pload8BF
D
	full_text7
5
3%523 = load double, double* %522, align 8, !tbaa !8
.double*8B

	full_text

double* %522
7add8B.
,
	full_text

%524 = add nsw i64 %518, -1
&i648B

	full_text


i64 %518
tgetelementptr8Ba
_
	full_textR
P
N%525 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %524, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %524
Pload8BF
D
	full_text7
5
3%526 = load double, double* %525, align 8, !tbaa !8
.double*8B

	full_text

double* %525
qcall8Bg
e
	full_textX
V
T%527 = call double @llvm.fmuladd.f64(double %526, double -4.000000e+00, double %523)
,double8B

	full_text

double %526
,double8B

	full_text

double %523
tgetelementptr8Ba
_
	full_textR
P
N%528 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %518, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %518
Pload8BF
D
	full_text7
5
3%529 = load double, double* %528, align 8, !tbaa !8
.double*8B

	full_text

double* %528
pcall8Bf
d
	full_textW
U
S%530 = call double @llvm.fmuladd.f64(double %529, double 6.000000e+00, double %527)
,double8B

	full_text

double %529
,double8B

	full_text

double %527
:add8B1
/
	full_text"
 
%531 = add nuw nsw i64 %518, 1
&i648B

	full_text


i64 %518
tgetelementptr8Ba
_
	full_textR
P
N%532 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %531, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %531
Pload8BF
D
	full_text7
5
3%533 = load double, double* %532, align 8, !tbaa !8
.double*8B

	full_text

double* %532
qcall8Bg
e
	full_textX
V
T%534 = call double @llvm.fmuladd.f64(double %533, double -4.000000e+00, double %530)
,double8B

	full_text

double %533
,double8B

	full_text

double %530
:add8B1
/
	full_text"
 
%535 = add nuw nsw i64 %518, 2
&i648B

	full_text


i64 %518
tgetelementptr8Ba
_
	full_textR
P
N%536 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %535, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %535
Pload8BF
D
	full_text7
5
3%537 = load double, double* %536, align 8, !tbaa !8
.double*8B

	full_text

double* %536
:fadd8B0
.
	full_text!

%538 = fadd double %534, %537
,double8B

	full_text

double %534
,double8B

	full_text

double %537
qcall8Bg
e
	full_textX
V
T%539 = call double @llvm.fmuladd.f64(double %538, double -2.500000e-01, double %520)
,double8B

	full_text

double %538
,double8B

	full_text

double %520
Pstore8BE
C
	full_text6
4
2store double %539, double* %519, align 8, !tbaa !8
,double8B

	full_text

double %539
.double*8B

	full_text

double* %519
:icmp8B0
.
	full_text!

%540 = icmp eq i64 %531, %516
&i648B

	full_text


i64 %531
&i648B

	full_text


i64 %516
=br8B5
3
	full_text&
$
"br i1 %540, label %541, label %517
$i18B

	full_text
	
i1 %540
8sext8B.
,
	full_text

%542 = sext i32 %405 to i64
&i328B

	full_text


i32 %405
5add8B,
*
	full_text

%543 = add nsw i32 %8, -5
8sext8B.
,
	full_text

%544 = sext i32 %543 to i64
&i328B

	full_text


i32 %543
5add8B,
*
	full_text

%545 = add nsw i32 %8, -4
8sext8B.
,
	full_text

%546 = sext i32 %545 to i64
&i328B

	full_text


i32 %545
8sext8B.
,
	full_text

%547 = sext i32 %272 to i64
&i328B

	full_text


i32 %272
¨getelementptr8B”
‘
	full_textƒ
€
~%548 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %542, i64 %274, i64 %273, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %542
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%549 = load double, double* %548, align 8, !tbaa !8
.double*8B

	full_text

double* %548
tgetelementptr8Ba
_
	full_textR
P
N%550 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %544, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %544
Pload8BF
D
	full_text7
5
3%551 = load double, double* %550, align 8, !tbaa !8
.double*8B

	full_text

double* %550
tgetelementptr8Ba
_
	full_textR
P
N%552 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %546, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%553 = load double, double* %552, align 8, !tbaa !8
.double*8B

	full_text

double* %552
qcall8Bg
e
	full_textX
V
T%554 = call double @llvm.fmuladd.f64(double %553, double -4.000000e+00, double %551)
,double8B

	full_text

double %553
,double8B

	full_text

double %551
tgetelementptr8Ba
_
	full_textR
P
N%555 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %542, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %542
Pload8BF
D
	full_text7
5
3%556 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
pcall8Bf
d
	full_textW
U
S%557 = call double @llvm.fmuladd.f64(double %556, double 6.000000e+00, double %554)
,double8B

	full_text

double %556
,double8B

	full_text

double %554
tgetelementptr8Ba
_
	full_textR
P
N%558 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %547, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %547
Pload8BF
D
	full_text7
5
3%559 = load double, double* %558, align 8, !tbaa !8
.double*8B

	full_text

double* %558
qcall8Bg
e
	full_textX
V
T%560 = call double @llvm.fmuladd.f64(double %559, double -4.000000e+00, double %557)
,double8B

	full_text

double %559
,double8B

	full_text

double %557
qcall8Bg
e
	full_textX
V
T%561 = call double @llvm.fmuladd.f64(double %560, double -2.500000e-01, double %549)
,double8B

	full_text

double %560
,double8B

	full_text

double %549
Pstore8BE
C
	full_text6
4
2store double %561, double* %548, align 8, !tbaa !8
,double8B

	full_text

double %561
.double*8B

	full_text

double* %548
¨getelementptr8B”
‘
	full_textƒ
€
~%562 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %547, i64 %274, i64 %273, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %547
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%563 = load double, double* %562, align 8, !tbaa !8
.double*8B

	full_text

double* %562
Pload8BF
D
	full_text7
5
3%564 = load double, double* %552, align 8, !tbaa !8
.double*8B

	full_text

double* %552
Pload8BF
D
	full_text7
5
3%565 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
qcall8Bg
e
	full_textX
V
T%566 = call double @llvm.fmuladd.f64(double %565, double -4.000000e+00, double %564)
,double8B

	full_text

double %565
,double8B

	full_text

double %564
Pload8BF
D
	full_text7
5
3%567 = load double, double* %558, align 8, !tbaa !8
.double*8B

	full_text

double* %558
pcall8Bf
d
	full_textW
U
S%568 = call double @llvm.fmuladd.f64(double %567, double 5.000000e+00, double %566)
,double8B

	full_text

double %567
,double8B

	full_text

double %566
qcall8Bg
e
	full_textX
V
T%569 = call double @llvm.fmuladd.f64(double %568, double -2.500000e-01, double %563)
,double8B

	full_text

double %568
,double8B

	full_text

double %563
Pstore8BE
C
	full_text6
4
2store double %569, double* %562, align 8, !tbaa !8
,double8B

	full_text

double %569
.double*8B

	full_text

double* %562
¨getelementptr8B”
‘
	full_textƒ
€
~%570 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %542, i64 %274, i64 %273, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %542
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%571 = load double, double* %570, align 8, !tbaa !8
.double*8B

	full_text

double* %570
tgetelementptr8Ba
_
	full_textR
P
N%572 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %544, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %544
Pload8BF
D
	full_text7
5
3%573 = load double, double* %572, align 8, !tbaa !8
.double*8B

	full_text

double* %572
tgetelementptr8Ba
_
	full_textR
P
N%574 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %546, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%575 = load double, double* %574, align 8, !tbaa !8
.double*8B

	full_text

double* %574
qcall8Bg
e
	full_textX
V
T%576 = call double @llvm.fmuladd.f64(double %575, double -4.000000e+00, double %573)
,double8B

	full_text

double %575
,double8B

	full_text

double %573
tgetelementptr8Ba
_
	full_textR
P
N%577 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %542, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %542
Pload8BF
D
	full_text7
5
3%578 = load double, double* %577, align 8, !tbaa !8
.double*8B

	full_text

double* %577
pcall8Bf
d
	full_textW
U
S%579 = call double @llvm.fmuladd.f64(double %578, double 6.000000e+00, double %576)
,double8B

	full_text

double %578
,double8B

	full_text

double %576
tgetelementptr8Ba
_
	full_textR
P
N%580 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %547, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %547
Pload8BF
D
	full_text7
5
3%581 = load double, double* %580, align 8, !tbaa !8
.double*8B

	full_text

double* %580
qcall8Bg
e
	full_textX
V
T%582 = call double @llvm.fmuladd.f64(double %581, double -4.000000e+00, double %579)
,double8B

	full_text

double %581
,double8B

	full_text

double %579
qcall8Bg
e
	full_textX
V
T%583 = call double @llvm.fmuladd.f64(double %582, double -2.500000e-01, double %571)
,double8B

	full_text

double %582
,double8B

	full_text

double %571
Pstore8BE
C
	full_text6
4
2store double %583, double* %570, align 8, !tbaa !8
,double8B

	full_text

double %583
.double*8B

	full_text

double* %570
¨getelementptr8B”
‘
	full_textƒ
€
~%584 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %547, i64 %274, i64 %273, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %547
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%585 = load double, double* %584, align 8, !tbaa !8
.double*8B

	full_text

double* %584
Pload8BF
D
	full_text7
5
3%586 = load double, double* %574, align 8, !tbaa !8
.double*8B

	full_text

double* %574
Pload8BF
D
	full_text7
5
3%587 = load double, double* %577, align 8, !tbaa !8
.double*8B

	full_text

double* %577
qcall8Bg
e
	full_textX
V
T%588 = call double @llvm.fmuladd.f64(double %587, double -4.000000e+00, double %586)
,double8B

	full_text

double %587
,double8B

	full_text

double %586
Pload8BF
D
	full_text7
5
3%589 = load double, double* %580, align 8, !tbaa !8
.double*8B

	full_text

double* %580
pcall8Bf
d
	full_textW
U
S%590 = call double @llvm.fmuladd.f64(double %589, double 5.000000e+00, double %588)
,double8B

	full_text

double %589
,double8B

	full_text

double %588
qcall8Bg
e
	full_textX
V
T%591 = call double @llvm.fmuladd.f64(double %590, double -2.500000e-01, double %585)
,double8B

	full_text

double %590
,double8B

	full_text

double %585
Pstore8BE
C
	full_text6
4
2store double %591, double* %584, align 8, !tbaa !8
,double8B

	full_text

double %591
.double*8B

	full_text

double* %584
¨getelementptr8B”
‘
	full_textƒ
€
~%592 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %542, i64 %274, i64 %273, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %542
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%593 = load double, double* %592, align 8, !tbaa !8
.double*8B

	full_text

double* %592
tgetelementptr8Ba
_
	full_textR
P
N%594 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %544, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %544
Pload8BF
D
	full_text7
5
3%595 = load double, double* %594, align 8, !tbaa !8
.double*8B

	full_text

double* %594
tgetelementptr8Ba
_
	full_textR
P
N%596 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %546, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%597 = load double, double* %596, align 8, !tbaa !8
.double*8B

	full_text

double* %596
qcall8Bg
e
	full_textX
V
T%598 = call double @llvm.fmuladd.f64(double %597, double -4.000000e+00, double %595)
,double8B

	full_text

double %597
,double8B

	full_text

double %595
tgetelementptr8Ba
_
	full_textR
P
N%599 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %542, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %542
Pload8BF
D
	full_text7
5
3%600 = load double, double* %599, align 8, !tbaa !8
.double*8B

	full_text

double* %599
pcall8Bf
d
	full_textW
U
S%601 = call double @llvm.fmuladd.f64(double %600, double 6.000000e+00, double %598)
,double8B

	full_text

double %600
,double8B

	full_text

double %598
tgetelementptr8Ba
_
	full_textR
P
N%602 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %547, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %547
Pload8BF
D
	full_text7
5
3%603 = load double, double* %602, align 8, !tbaa !8
.double*8B

	full_text

double* %602
qcall8Bg
e
	full_textX
V
T%604 = call double @llvm.fmuladd.f64(double %603, double -4.000000e+00, double %601)
,double8B

	full_text

double %603
,double8B

	full_text

double %601
qcall8Bg
e
	full_textX
V
T%605 = call double @llvm.fmuladd.f64(double %604, double -2.500000e-01, double %593)
,double8B

	full_text

double %604
,double8B

	full_text

double %593
Pstore8BE
C
	full_text6
4
2store double %605, double* %592, align 8, !tbaa !8
,double8B

	full_text

double %605
.double*8B

	full_text

double* %592
¨getelementptr8B”
‘
	full_textƒ
€
~%606 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %547, i64 %274, i64 %273, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %547
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%607 = load double, double* %606, align 8, !tbaa !8
.double*8B

	full_text

double* %606
Pload8BF
D
	full_text7
5
3%608 = load double, double* %596, align 8, !tbaa !8
.double*8B

	full_text

double* %596
Pload8BF
D
	full_text7
5
3%609 = load double, double* %599, align 8, !tbaa !8
.double*8B

	full_text

double* %599
qcall8Bg
e
	full_textX
V
T%610 = call double @llvm.fmuladd.f64(double %609, double -4.000000e+00, double %608)
,double8B

	full_text

double %609
,double8B

	full_text

double %608
Pload8BF
D
	full_text7
5
3%611 = load double, double* %602, align 8, !tbaa !8
.double*8B

	full_text

double* %602
pcall8Bf
d
	full_textW
U
S%612 = call double @llvm.fmuladd.f64(double %611, double 5.000000e+00, double %610)
,double8B

	full_text

double %611
,double8B

	full_text

double %610
qcall8Bg
e
	full_textX
V
T%613 = call double @llvm.fmuladd.f64(double %612, double -2.500000e-01, double %607)
,double8B

	full_text

double %612
,double8B

	full_text

double %607
Pstore8BE
C
	full_text6
4
2store double %613, double* %606, align 8, !tbaa !8
,double8B

	full_text

double %613
.double*8B

	full_text

double* %606
¨getelementptr8B”
‘
	full_textƒ
€
~%614 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %542, i64 %274, i64 %273, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %542
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%615 = load double, double* %614, align 8, !tbaa !8
.double*8B

	full_text

double* %614
tgetelementptr8Ba
_
	full_textR
P
N%616 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %544, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %544
Pload8BF
D
	full_text7
5
3%617 = load double, double* %616, align 8, !tbaa !8
.double*8B

	full_text

double* %616
tgetelementptr8Ba
_
	full_textR
P
N%618 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %546, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%619 = load double, double* %618, align 8, !tbaa !8
.double*8B

	full_text

double* %618
qcall8Bg
e
	full_textX
V
T%620 = call double @llvm.fmuladd.f64(double %619, double -4.000000e+00, double %617)
,double8B

	full_text

double %619
,double8B

	full_text

double %617
tgetelementptr8Ba
_
	full_textR
P
N%621 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %542, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %542
Pload8BF
D
	full_text7
5
3%622 = load double, double* %621, align 8, !tbaa !8
.double*8B

	full_text

double* %621
pcall8Bf
d
	full_textW
U
S%623 = call double @llvm.fmuladd.f64(double %622, double 6.000000e+00, double %620)
,double8B

	full_text

double %622
,double8B

	full_text

double %620
tgetelementptr8Ba
_
	full_textR
P
N%624 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %547, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %547
Pload8BF
D
	full_text7
5
3%625 = load double, double* %624, align 8, !tbaa !8
.double*8B

	full_text

double* %624
qcall8Bg
e
	full_textX
V
T%626 = call double @llvm.fmuladd.f64(double %625, double -4.000000e+00, double %623)
,double8B

	full_text

double %625
,double8B

	full_text

double %623
qcall8Bg
e
	full_textX
V
T%627 = call double @llvm.fmuladd.f64(double %626, double -2.500000e-01, double %615)
,double8B

	full_text

double %626
,double8B

	full_text

double %615
Pstore8BE
C
	full_text6
4
2store double %627, double* %614, align 8, !tbaa !8
,double8B

	full_text

double %627
.double*8B

	full_text

double* %614
¨getelementptr8B”
‘
	full_textƒ
€
~%628 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %547, i64 %274, i64 %273, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %547
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%629 = load double, double* %628, align 8, !tbaa !8
.double*8B

	full_text

double* %628
Pload8BF
D
	full_text7
5
3%630 = load double, double* %618, align 8, !tbaa !8
.double*8B

	full_text

double* %618
Pload8BF
D
	full_text7
5
3%631 = load double, double* %621, align 8, !tbaa !8
.double*8B

	full_text

double* %621
qcall8Bg
e
	full_textX
V
T%632 = call double @llvm.fmuladd.f64(double %631, double -4.000000e+00, double %630)
,double8B

	full_text

double %631
,double8B

	full_text

double %630
Pload8BF
D
	full_text7
5
3%633 = load double, double* %624, align 8, !tbaa !8
.double*8B

	full_text

double* %624
pcall8Bf
d
	full_textW
U
S%634 = call double @llvm.fmuladd.f64(double %633, double 5.000000e+00, double %632)
,double8B

	full_text

double %633
,double8B

	full_text

double %632
qcall8Bg
e
	full_textX
V
T%635 = call double @llvm.fmuladd.f64(double %634, double -2.500000e-01, double %629)
,double8B

	full_text

double %634
,double8B

	full_text

double %629
Pstore8BE
C
	full_text6
4
2store double %635, double* %628, align 8, !tbaa !8
,double8B

	full_text

double %635
.double*8B

	full_text

double* %628
¨getelementptr8B”
‘
	full_textƒ
€
~%636 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %542, i64 %274, i64 %273, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %542
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%637 = load double, double* %636, align 8, !tbaa !8
.double*8B

	full_text

double* %636
tgetelementptr8Ba
_
	full_textR
P
N%638 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %544, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %544
Pload8BF
D
	full_text7
5
3%639 = load double, double* %638, align 8, !tbaa !8
.double*8B

	full_text

double* %638
tgetelementptr8Ba
_
	full_textR
P
N%640 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %546, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%641 = load double, double* %640, align 8, !tbaa !8
.double*8B

	full_text

double* %640
qcall8Bg
e
	full_textX
V
T%642 = call double @llvm.fmuladd.f64(double %641, double -4.000000e+00, double %639)
,double8B

	full_text

double %641
,double8B

	full_text

double %639
tgetelementptr8Ba
_
	full_textR
P
N%643 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %542, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %542
Pload8BF
D
	full_text7
5
3%644 = load double, double* %643, align 8, !tbaa !8
.double*8B

	full_text

double* %643
pcall8Bf
d
	full_textW
U
S%645 = call double @llvm.fmuladd.f64(double %644, double 6.000000e+00, double %642)
,double8B

	full_text

double %644
,double8B

	full_text

double %642
tgetelementptr8Ba
_
	full_textR
P
N%646 = getelementptr inbounds [5 x double], [5 x double]* %33, i64 %547, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %33
&i648B

	full_text


i64 %547
Pload8BF
D
	full_text7
5
3%647 = load double, double* %646, align 8, !tbaa !8
.double*8B

	full_text

double* %646
qcall8Bg
e
	full_textX
V
T%648 = call double @llvm.fmuladd.f64(double %647, double -4.000000e+00, double %645)
,double8B

	full_text

double %647
,double8B

	full_text

double %645
qcall8Bg
e
	full_textX
V
T%649 = call double @llvm.fmuladd.f64(double %648, double -2.500000e-01, double %637)
,double8B

	full_text

double %648
,double8B

	full_text

double %637
Pstore8BE
C
	full_text6
4
2store double %649, double* %636, align 8, !tbaa !8
,double8B

	full_text

double %649
.double*8B

	full_text

double* %636
¨getelementptr8B”
‘
	full_textƒ
€
~%650 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %30, i64 %547, i64 %274, i64 %273, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %30
&i648B

	full_text


i64 %547
&i648B

	full_text


i64 %274
&i648B

	full_text


i64 %273
Pload8BF
D
	full_text7
5
3%651 = load double, double* %650, align 8, !tbaa !8
.double*8B

	full_text

double* %650
Pload8BF
D
	full_text7
5
3%652 = load double, double* %640, align 8, !tbaa !8
.double*8B

	full_text

double* %640
Pload8BF
D
	full_text7
5
3%653 = load double, double* %643, align 8, !tbaa !8
.double*8B

	full_text

double* %643
qcall8Bg
e
	full_textX
V
T%654 = call double @llvm.fmuladd.f64(double %653, double -4.000000e+00, double %652)
,double8B

	full_text

double %653
,double8B

	full_text

double %652
Pload8BF
D
	full_text7
5
3%655 = load double, double* %646, align 8, !tbaa !8
.double*8B

	full_text

double* %646
pcall8Bf
d
	full_textW
U
S%656 = call double @llvm.fmuladd.f64(double %655, double 5.000000e+00, double %654)
,double8B

	full_text

double %655
,double8B

	full_text

double %654
qcall8Bg
e
	full_textX
V
T%657 = call double @llvm.fmuladd.f64(double %656, double -2.500000e-01, double %651)
,double8B

	full_text

double %656
,double8B

	full_text

double %651
Pstore8BE
C
	full_text6
4
2store double %657, double* %650, align 8, !tbaa !8
,double8B

	full_text

double %657
.double*8B

	full_text

double* %650
(br8B 

	full_text

br label %433
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %7
,double*8B

	full_text


double* %4
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
$i648B

	full_text


i64 32
5double8B'
%
	full_text

double -5.500000e+00
$i328B

	full_text


i32 60
$i328B

	full_text


i32 -2
$i648B

	full_text


i64 40
#i328B

	full_text	

i32 0
5double8B'
%
	full_text

double -2.000000e+00
4double8B&
$
	full_text

double 5.000000e-01
#i648B

	full_text	

i64 5
4double8B&
$
	full_text

double 4.000000e-01
:double8B,
*
	full_text

double 0x4037B74BC6A7EF9D
5double8B'
%
	full_text

double -4.000000e+00
#i648B

	full_text	

i64 2
&i648B

	full_text


i64 1690
$i648B

	full_text


i64 -2
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 -1
#i648B

	full_text	

i64 4
5double8B'
%
	full_text

double -0.000000e+00
5double8B'
%
	full_text

double -2.500000e-01
#i328B

	full_text	

i32 7
%i648B

	full_text
	
i64 845
$i328B

	full_text


i32 -5
#i328B

	full_text	

i32 3
4double8B&
$
	full_text

double 4.000000e+00
$i328B

	full_text


i32 -4
:double8B,
*
	full_text

double 0x4028333333333334
$i648B

	full_text


i64 10
4double8B&
$
	full_text

double 1.400000e+00
$i328B

	full_text


i32 -1
$i328B

	full_text


i32 12
4double8B&
$
	full_text

double 5.000000e+00
#i328B

	full_text	

i32 1
4double8B&
$
	full_text

double 1.210000e+02
%i18B

	full_text


i1 false
:double8B,
*
	full_text

double 0x4030222222222222
$i648B

	full_text


i64 15
:double8B,
*
	full_text

double 0x4000222222222222
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 6.000000e+00
$i328B

	full_text


i32 -3
:double8B,
*
	full_text

double 0x3FB745D1745D1746
4double8B&
$
	full_text

double 1.000000e+00
:double8B,
*
	full_text

double 0xC0173B645A1CAC07
$i648B

	full_text


i64 20        	
 		                       !" !! #$ #% ## &' && () (( ** +, ++ -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?@ ?? AB AA CC DE DF GI HH JK JJ LM LL NO NN PQ PP RR SU TT VW VV XY XZ XX [\ [[ ]^ ]] _` __ ab aa cd cc ef eg eh ei ee jk jl jj mn mm op oo qr qq st su ss vw vx vv yz y{ yy |} || ~ ~	€ ~~ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ  
‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®
° ®
± ®® ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½
¿ ½½ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç Å
È ÅÅ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ ÎÎ ÐÑ Ð
Ò Ð
Ó ÐÐ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áã ää åæ åè ç
é çç êë êê ìí ìì îï îî ðñ ðð òô óó õö õõ ÷ø ÷÷ ùú ùù ûû üý üü þ€ ÿÿ ‚  ƒ„ ƒƒ …† …
‡ …
ˆ …
‰ …… Š‹ ŠŠ Œ Œ
Ž ŒŒ   ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
ž œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·
º ·
» ·· ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ ÐÐ ÒÓ Ò
Ô ÒÒ Õ
Ö ÕÕ ×Ø ×
Ù ×
Ú ×× ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ áá ãä ã
å ãã æç ææ èé è
ê èè ëì ë
í ëë îï îî ðñ ð
ò ðð óô ó
õ óó ö÷ ö
ø öö ùú ùù ûü û
ý ûû þÿ þ
€ þþ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡
‰ ‡
Š ‡
‹ ‡‡ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” ““ •– •
— •• ˜™ ˜˜ š› šš œ œ
ž œœ Ÿ
  ŸŸ ¡¢ ¡
£ ¡
¤ ¡¡ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó Ñ
Ô Ñ
Õ ÑÑ Ö× ÖÖ ØÙ ØØ ÚÛ ÚÚ ÜÝ Ü
Þ ÜÜ ßà ßß áâ á
ã áá äå ää æç æ
è ææ éê éé ëì ë
í ë
î ëë ïð ïï ñò ññ óô ó
õ óó ö÷ öö øù ø
ú øø ûü ûû ýþ ý
ÿ ýý € €€ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ   ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «
® «
¯ «« °± °° ²³ ²² ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º
» ºº ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ ÁÁ ÃÄ ÃÃ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ Ï
Ð ÏÏ ÑÒ Ñ
Ó Ñ
Ô ÑÑ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ ÛÛ ÝÞ Ý
ß ÝÝ àá àà âã â
ä ââ åæ å
ç åå èé èè êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô óó õö õ
÷ õõ øù øø úû ú
ü úú ýþ ý
ÿ ýý € €€ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž 
  ‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜˜ š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ ££ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿¿ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÈ ÉÊ ÉÉ ËÌ ËË ÍÎ Í
Ï Í
Ð ÍÍ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× Ù
Ú ÙÙ ÛÜ Û
Ý ÛÛ Þß ÞÞ àá à
â àà ãä ã
å ãã æç æ
è ææ éê é
ë é
ì éé íî íí ïð ïï ñò ññ óô óó õö õ
÷ õõ øù øø úû ú
ü úú ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž ŽŽ ‘  ’“ ’’ ”• ”” –— –– ˜
™ ˜˜ š› š
œ šš ž  Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ª
­ ªª ®¯ ®® °± °° ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ Ë
Í Ë
Î ËË ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×× ÙÚ ÙÙ Û
Ü ÛÛ ÝÞ Ý
ß ÝÝ àá àà âã ââ äå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï í
ð íí ñò ññ óô óó õö õõ ÷ø ÷÷ ùú ù
û ùù üý üü þÿ þ
€ þþ ‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 Ž
‘ ŽŽ ’“ ’’ ”• ”” –— –– ˜™ ˜˜ š› šš œ œœ ž
Ÿ žž  ¡  
¢    £¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °
³ °° ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó Ñ
Ô ÑÑ ÕÖ ÕÕ ×Ø ×× ÙÚ ÙÙ ÛÜ ÛÛ ÝÞ ÝÝ ßà ßß á
â áá ãä ã
å ãã æç ææ èé èè êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô ó
õ ó
ö óó ÷ø ÷÷ ùú ùù ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”” •• –— –™ ˜˜ šœ ›› ž 
Ÿ 
  
¡  ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «« ­® ­
¯ ­­ °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ ÚÝ Ü
ß ÞÞ àâ áá ãå ää æç æ
è æ
é æ
ê ææ ëì ëë íî íí ïð ï
ñ ïï òó òò ôõ ôô ö÷ ö
ø öö ùú ùù ûü û
ý ûû þÿ þ
€	 þþ 	‚	 		 ƒ	„	 ƒ	
…	 ƒ	ƒ	 †	‡	 †	†	 ˆ	‰	 ˆ	
Š	 ˆ	ˆ	 ‹	Œ	 ‹	‹	 	Ž	 	
	 		 	‘	 		 ’	“	 ’	
”	 ’	’	 •	–	 •	•	 —	˜	 —	
™	 —	—	 š	›	 š	
œ	 š	š	 	ž	 	
Ÿ	 		  	¡	  	
¢	  	 	 £	¤	 £	¦	 ¥	¨	 §	§	 ©	«	 ª	ª	 ¬	­	 ¬	
®	 ¬	
¯	 ¬	
°	 ¬	¬	 ±	²	 ±	±	 ³	´	 ³	³	 µ	¶	 µ	
·	 µ	µ	 ¸	¹	 ¸	¸	 º	»	 º	º	 ¼	½	 ¼	
¾	 ¼	¼	 ¿	À	 ¿	¿	 Á	Â	 Á	
Ã	 Á	Á	 Ä	Å	 Ä	
Æ	 Ä	Ä	 Ç	È	 Ç	Ç	 É	Ê	 É	
Ë	 É	É	 Ì	Í	 Ì	Ì	 Î	Ï	 Î	
Ð	 Î	Î	 Ñ	Ò	 Ñ	Ñ	 Ó	Ô	 Ó	
Õ	 Ó	Ó	 Ö	×	 Ö	Ö	 Ø	Ù	 Ø	
Ú	 Ø	Ø	 Û	Ü	 Û	Û	 Ý	Þ	 Ý	
ß	 Ý	Ý	 à	á	 à	
â	 à	à	 ã	ä	 ã	
å	 ã	ã	 æ	ç	 æ	
è	 æ	æ	 é	ê	 é	ì	 ë	î	 í	í	 ï	ñ	 ð	ð	 ò	ó	 ò	
ô	 ò	
õ	 ò	
ö	 ò	ò	 ÷	ø	 ÷	÷	 ù	ú	 ù	ù	 û	ü	 û	
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
ˆ
 ‡

‰
 ‡
‡
 Š
‹
 Š

Œ
 Š
Š
 
Ž
 

 

 

‘
 

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
›
 ™
™
 œ

 œ
œ
 ž
Ÿ
 ž

 
 ž
ž
 ¡
¢
 ¡
¡
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
¨
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
 ¯
°
 ¯
²
 ±
´
 ³
³
 µ
·
 ¶
¶
 ¸
¹
 ¸

º
 ¸

»
 ¸

¼
 ¸
¸
 ½
¾
 ½
½
 ¿
À
 ¿
¿
 Á
Â
 Á

Ã
 Á
Á
 Ä
Å
 Ä
Ä
 Æ
Ç
 Æ
Æ
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
Ï
 Í
Í
 Ð
Ñ
 Ð

Ò
 Ð
Ð
 Ó
Ô
 Ó
Ó
 Õ
Ö
 Õ

×
 Õ
Õ
 Ø
Ù
 Ø
Ø
 Ú
Û
 Ú

Ü
 Ú
Ú
 Ý
Þ
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
â
 ä
å
 ä

æ
 ä
ä
 ç
è
 ç
ç
 é
ê
 é

ë
 é
é
 ì
í
 ì

î
 ì
ì
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
ö
 õ
ø
 ÷
÷
 ù
ù
 ú
û
 ú
ú
 ü
ü
 ý
þ
 ý
ý
 ÿ
€ ÿ
ÿ
 ‚ 
ƒ 
„ 
…  †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž 
  ‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜˜ š› š
œ šš ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «
® «
¯ «« °± °° ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç Ä
È ÄÄ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ ÎÎ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ ÛÛ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ãã åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ð î
ñ î
ò îî óô óó õö õõ ÷ø ÷÷ ùú ù
û ùù üý üü þÿ þ
€ þþ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡
‰ ‡
Š ‡
‹ ‡‡ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ žž  ¡  
¢    £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±
´ ±
µ ±± ¶· ¶¶ ¸¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì Ê
Í Ê
Î ÊÊ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ ÙÙ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ áá ãä ã
å ãã æç æ
è ææ éê éé ëì ë
í ëë îï î
ð îî ñò ñ
ó ññ ôõ ô
ö ô
÷ ô
ø ôô ùú ùù ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚‚ „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
 
 
‘  ’“ ’’ ”• ”
– ”” —˜ —— ™š ™
› ™™ œ œœ žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·
º ·
» ·· ¼½ ¼¼ ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ *Ñ ¿Ñ ÈÒ CÒ FÒ RÒ ãÒ äÒ ûÒ ”Ò •Ò ù
Ò ü
Ó 7Ô 	Ô Õ -Õ [	Ö e× 1Ø Ù 9    
   	      	    "! $ %# '# )( ,+ .- 0+ 21 4& 65 85 :	 <; > @? BC E I K M O QÜ UT WV Y+ ZX \[ ^T `_ ba dA f= gc hH i] k lH nm pJ ro tq u3 wT xs zv {L }o | €3 ‚T ƒ~ … †N ˆo Š‡ ‹3 T Ž‰ Œ ‘P “o •’ –3 ˜T ™” ›— œ‰ ž‰ Ÿ7 ¡T ¢ ¤  ¥v §¦ ©¦ ª « ­¬ ¯¬ °¨ ±3 ³T ´® ¶² ·/ ¹T º¸ ¼/ ¾T ¿½ Á¬ ÃÀ Ä¦ Æ» ÇÂ ÈŒ Ê/ ÌT ÍË ÏÉ ÑÎ ÒÅ ÓÐ Õ9 ×T ØÔ ÚÖ ÛT ÝÜ ßR àÞ âä æF èã é ëê í ïî ñ ôó ö ø÷ úû ýƒ €ÿ ‚ÿ „* †ÿ ‡õ ˆù ‰… ‹/ ƒ ŽŒ / ’ “‘ • —” ˜– šŠ ›/ ƒ žœ  / ¢ÿ £¡ ¥¤ §Ÿ ¨/ ª «© ­¦ ¯¬ °® ²™ ³± µ… ¶* ¸ÿ ¹õ ºù »· ½/ ¿ƒ À¾ Â3 Äƒ ÅÃ Ç/ É ÊÈ Ì3 Î ÏÍ ÑË ÓÐ ÔÒ ÖÁ ØÆ ÙÕ Ú× Ü¼ Ý3 ßƒ àÞ â3 äÿ åã çæ éá ê3 ì íë ïè ñî òð ôÛ õ/ ÷ÿ øö úù üÁ ýË ÿû €þ ‚ó ƒ …· †* ˆÿ ‰õ Šù ‹‡ / ƒ Ž ’Ã ”/ – —• ™Í ›˜ š žœ  ‘ ¢“ £Ÿ ¤¡ ¦Œ §3 ©ƒ ª¨ ¬3 ®ÿ ¯­ ±° ³« ´3 ¶ ·µ ¹² »¸ ¼º ¾¥ ¿/ Áÿ ÂÀ ÄÃ Æ‘ Ç˜ ÉÅ ÊÈ Ì½ ÍË Ï‡ Ð* Òÿ Óõ Ôù ÕÑ ×Œ ÙÃ Û/ Ýƒ ÞÜ à9 âƒ ãá åß çä èæ êØ ìÚ íé î‘ ðÍ ò/ ô õó ÷9 ù úø üö þû ÿý ï ƒñ „€ …ë ‡‚ ˆ† ŠÖ ‹3 ÿ ŽŒ  ’Ú “ñ •‘ –” ˜‰ ™/ ›ÿ œš ž  Ø ¡ï £Ÿ ¤¢ ¦— §¥ ©Ñ ª* ¬ÿ ­õ ®ù ¯« ±Ã ³Ü µá ·¶ ¹¸ »´ ½º ¾Í Àó Âø ÄÃ ÆÅ ÈÁ ÊÇ Ë¿ ÍÉ ÎÌ Ð² Ò¼ ÓÏ ÔÑ Ö° ×3 Ùƒ ÚØ Ü3 Þÿ ßÝ áà ãÛ ä3 æ çå éâ ëè ìê îÕ ï7 ñƒ òð ô7 öÿ ÷õ ùø ûó ü7 þ ÿý ú ƒ€ „‚ †í ‡3 ‰ƒ Šˆ Œ3 Žÿ  ‘ “‹ ”3 – —• ™’ ›˜ œš ž… Ÿ/ ¡ÿ ¢  ¤£ ¦´ §Á ©¥ ª¨ ¬ ­« ¯« °ƒ ²ü ³± µç ·ã ¸ð ºù »ì ½õ ¾¿ Á- Ã- Å- ÇÈ Ê- ÌÀ Î¼ Ï¹ ÐÍ ÒÂ ÔÄ ÖÕ Ø× ÚÓ ÜÙ ÝÆ ßÞ áÛ âà äÑ åã çÍ èÉ ê¼ ë¹ ìé îÂ ðÄ òñ ôï öó ÷Æ ùø ûõ üË þý €ú ÿ ƒí „‚ †é ‡À ‰¼ Š¹ ‹ˆ Â Ž ‘Ä “’ •” —– ™ ›˜ œÆ ž  Ÿ ¢š £¡ ¥Œ ¦¤ ¨ˆ ©É «¼ ¬¹ ­ª ¯Ž ±’ ³² µ° ·´ ¸ º¹ ¼¶ ½Ë ¿¾ ÁÀ Ã» ÄÂ Æ® ÇÅ Éª ÊÀ Ì¼ Í¹ ÎË ÐÂ ÒÑ ÔÄ ÖÕ Ø× ÚÙ ÜÓ ÞÛ ßÆ áà ãâ åÝ æä èÏ éç ëË ìÉ î¼ ï¹ ðí òÑ ôÕ öõ øó ú÷ ûà ýü ÿù €Ë ‚ „ƒ †þ ‡… ‰ñ Šˆ Œí À ¼ ¹ ‘Ž “Â •” —Ä ™˜ ›š œ Ÿ– ¡ž ¢Æ ¤£ ¦¥ ¨  ©§ «’ ¬ª ®Ž ¯É ±¼ ²¹ ³° µ” ·˜ ¹¸ »¶ ½º ¾£ À¿ Â¼ ÃË ÅÄ ÇÆ ÉÁ ÊÈ Ì´ ÍË Ï° ÐÀ Ò¼ Ó¹ ÔÑ ÖÂ Ø× ÚÄ ÜÛ ÞÝ àß âÙ äá åÆ çæ éè ëã ìê îÕ ïí ñÑ òÉ ô¼ õ¹ öó ø× úÛ üû þù €ý æ ƒ‚ …ÿ †Ë ˆ‡ Š‰ Œ„ ‹ ÷ Ž ’ó “” —• ™½ œ* ž› Ÿ¼  ¹ ¡ £› ¥/ §¤ ¨¦ ª› ¬/ ®« ¯­ ±° ³© ´/ ¶› ·µ ¹¸ »² ¼› ¾/ À½ Á¿ ÃÂ Åº Æ› È/ ÊÇ ËÉ ÍÄ ÏÌ ÐÎ Ò¢ ÓÑ Õ Ö½ Ø˜ Ù× Û” Ý ß• â†	 å* çä è¼ é¹ êæ ìä î/ ðí ñï óä õ/ ÷ô øö úù üò ý/ ÿä €	þ ‚		 „	û …	ä ‡	/ ‰	†	 Š	ˆ	 Œ	‹	 Ž	ƒ	 	ä ‘	/ “		 ”	’	 –		 ˜	•	 ™	—	 ›	ë œ	š	 ž	æ Ÿ	†	 ¡	á ¢	 	 ¤	” ¦	• ¨	Ì	 «	* ­	ª	 ®	¼ ¯	¹ °	¬	 ²	ª	 ´	/ ¶	³	 ·	µ	 ¹	ª	 »	/ ½	º	 ¾	¼	 À	¿	 Â	¸	 Ã	/ Å	ª	 Æ	Ä	 È	Ç	 Ê	Á	 Ë	ª	 Í	/ Ï	Ì	 Ð	Î	 Ò	Ñ	 Ô	É	 Õ	ª	 ×	/ Ù	Ö	 Ú	Ø	 Ü	Ó	 Þ	Û	 ß	Ý	 á	±	 â	à	 ä	¬	 å	Ì	 ç	§	 è	æ	 ê	” ì	• î	’
 ñ	* ó	ð	 ô	¼ õ	¹ ö	ò	 ø	ð	 ú	/ ü	ù	 ý	û	 ÿ	ð	 
/ ƒ
€
 „
‚
 †
…
 ˆ
þ	 ‰
/ ‹
ð	 Œ
Š
 Ž

 
‡
 ‘
ð	 “
/ •
’
 –
”
 ˜
—
 š

 ›
ð	 
/ Ÿ
œ
  
ž
 ¢
™
 ¤
¡
 ¥
£
 §
÷	 ¨
¦
 ª
ò	 «
’
 ­
í	 ®
¬
 °
” ²
• ´
Ø
 ·
* ¹
¶
 º
¼ »
¹ ¼
¸
 ¾
¶
 À
/ Â
¿
 Ã
Á
 Å
¶
 Ç
/ É
Æ
 Ê
È
 Ì
Ë
 Î
Ä
 Ï
/ Ñ
¶
 Ò
Ð
 Ô
Ó
 Ö
Í
 ×
¶
 Ù
/ Û
Ø
 Ü
Ú
 Þ
Ý
 à
Õ
 á
¶
 ã
/ å
â
 æ
ä
 è
ß
 ê
ç
 ë
é
 í
½
 î
ì
 ð
¸
 ñ
Ø
 ó
³
 ô
ò
 ö
• ø
ù
 û
ü
 þ
¶ €* ‚÷
 ƒ¼ „¹ … ‡/ ‰ú
 Šˆ Œ/ Žý
  ‘ “‹ ”/ –÷
 —• ™˜ ›’ œ/ žÿ
 Ÿ ¡  £š ¤¢ ¦† §¥ © ª* ¬ÿ
 ­¼ ®¹ ¯« ± ³• µ´ ·² ¸ º¹ ¼¶ ½» ¿° À¾ Â« Ã* Å÷
 Æ¼ Ç¹ ÈÄ Ê/ Ìú
 ÍË Ï/ Ñý
 ÒÐ ÔÓ ÖÎ ×/ Ù÷
 ÚØ ÜÛ ÞÕ ß/ áÿ
 âà äã æÝ çå éÉ êè ìÄ í* ïÿ
 ð¼ ñ¹ òî ôÐ öØ ø÷ úõ ûà ýü ÿù €þ ‚ó ƒ …î †* ˆ÷
 ‰¼ Š¹ ‹‡ / ú
 Ž ’/ ”ý
 •“ —– ™‘ š/ œ÷
 › Ÿž ¡˜ ¢/ ¤ÿ
 ¥£ §¦ ©  ª¨ ¬Œ ­« ¯‡ °* ²ÿ
 ³¼ ´¹ µ± ·“ ¹› »º ½¸ ¾£ À¿ Â¼ ÃÁ Å¶ ÆÄ È± É* Ë÷
 Ì¼ Í¹ ÎÊ Ð/ Òú
 ÓÑ Õ/ ×ý
 ØÖ ÚÙ ÜÔ Ý/ ß÷
 àÞ âá äÛ å/ çÿ
 èæ êé ìã íë ïÏ ðî òÊ ó* õÿ
 ö¼ ÷¹ øô úÖ üÞ þý €û æ ƒ‚ …ÿ †„ ˆù ‰‡ ‹ô Œ* Ž÷
 ¼ ¹ ‘ “/ •ú
 –” ˜/ šý
 ›™ œ Ÿ—  / ¢÷
 £¡ ¥¤ §ž ¨/ ªÿ
 «© ­¬ ¯¦ °® ²’ ³± µ ¶* ¸ÿ
 ¹¼ º¹ »· ½™ ¿¡ ÁÀ Ã¾ Ä© ÆÅ ÈÂ ÉÇ Ë¼ ÌÊ Î· Ï Þ D HD FS TG çá ãá Tò ¶å çå ó– ÷
– ˜þ ÿÐ Þš ›´ ¶´ ÿÚ ÜÚ ›Ü ÷
Ü áã ä£	 ¥	£	 ä¥	 ÷
¥	 §	©	 ª	é	 ë	é	 ª	ë	 ÷
ë	 í	ï	 ð	¯
 ±
¯
 ð	±
 ÷
±
 ³
µ
 ¶
õ
 ÷
õ
 ¶
 ÚÚ ÜÜ ÛÛ ßß à ÝÝ ÞÞë ÝÝ ë  ÝÝ  í ÝÝ íã ÝÝ ãÝ ÝÝ Ý… ÝÝ … ÛÛ  ÝÝ í ÝÝ í„ ÝÝ „É	 ÝÝ É	þ ÝÝ þ¶ ÝÝ ¶û ÝÝ û ÛÛ ú ÝÝ ú¡ ÝÝ ¡ª ÝÝ ª ÝÝ ² ÝÝ ²Ó	 ÝÝ Ó	ã ÝÝ ã¼ ÝÝ ¼Í
 ÝÝ Í
Á	 ÝÝ Á	Ý ÝÝ Ý¨ ÝÝ ¨’ ÝÝ ’Å ÝÝ Å½ ÝÝ ½è ÝÝ èþ ÝÝ þù ÝÝ ùÇ ÝÝ Ç¦ ÝÝ ¦¤ ÝÝ ¤¼ ÝÝ ¼Ê ÝÝ ÊÞ ÞÞ Þî ÝÝ îâ ÝÝ â« ÝÝ «õ ÝÝ õÐ ÝÝ ÐÄ ÝÝ Äû ÝÝ û‡
 ÝÝ ‡
˜ ÝÝ ˜Û ÝÝ Ûè ÝÝ è  ÝÝ  ¥ ÝÝ ¥« ÝÝ «— ÝÝ —ÿ ÝÝ ÿË ÝÝ ËÕ
 ÝÝ Õ
¶ ÝÝ ¶¼ ÝÝ ¼º ÝÝ ºà	 ÝÝ à	¾ ÝÝ ¾å ÝÝ åÕ ÝÝ ÕÁ ÝÝ Á ÚÚ ± ÝÝ ±ç ÝÝ ç„ ÝÝ „ì
 ÝÝ ì
š ÝÝ šË ÝÝ Ë» ÝÝ »š	 ÝÝ š	™ ÝÝ ™¨ ÝÝ ¨¦
 ÝÝ ¦
e ÜÜ e¥ ÝÝ ¥¥ ÝÝ ¥ƒ	 ÝÝ ƒ	Ä ÝÝ Ä» ÝÝ »‘ ÝÝ ‘¢ ÝÝ ¢Â ÝÝ Âž ÝÝ žÁ ÝÝ ÁÉ ÝÝ É	 ÝÝ 	Å ÝÝ Åÿ ÝÝ ÿ¦ ÝÝ ¦‰ ÝÝ ‰Ñ ÝÝ Ñß
 ÝÝ ß
ë ÝÝ ëˆ ÝÝ ˆ× ÝÝ ×® ÝÝ ®® ÝÝ ®² ÝÝ ²ú ÝÝ úŽ ÝÝ Ž
 ÝÝ 
¥ ÝÝ ¥™
 ÝÝ ™
‚ ÝÝ ‚‚ ÝÝ ‚Ÿ ÝÝ ŸÛ ÝÝ Ûù ÝÝ ùÅ ÝÝ Å± ÝÝ ±j ßß jš ÝÝ šÕ ÝÝ ÕÛ ÝÝ Û‡ ÝÝ ‡ã ÝÝ ãó ÝÝ óÑ ÝÝ Ñ’ ÝÝ ’ ÝÝ 
à ê
à ì
à î
à ð
à ó
à õ
à ÷
à ù
á ™
á Û
á ¥
á ‰
á Õ	â (	ã 	ã 	ã F
ã ãä 	ä jä Þå 	å C
æ ¦
æ è
æ û
æ ²
æ Å
æ ‘
æ Ÿ
æ â
æ ú
æ ’
æ ¥
ç Ô	è V
è Â
é é
é €
é ¸
é Å
ê 
ë õ
ë ú
ë ¶
ë »
ë ù
ë þ
ë ¼
ë Á
ë ÿ
ë „
ë ²
ë Ä
ë û
ë 	
ë Á	
ë Ó	
ë ‡

ë ™

ë Í

ë ß

ë ’
ë ¢
ë ¶
ë Õ
ë å
ë ù
ë ˜
ë ¨
ë ¼
ë Û
ë ë
ë ÿ
ë ž
ë ®
ë Â	ì L
ì 
ì ½
ì ‡
ì Ž
ì •
ì ¨
ì ­
ì µ
ì À
ì Ë
ì Ñ
ì Õ
ì à
ì í
ì 
ì Ç
ì 	
ì ¬	
ì µ	
ì ¼	
ì Ä	
ì Î	
ì Ö	
ì Ø	
ì œ

ì â

ì ‡
ì Ž
ì “
ì ›
ì £
ì ±
í È
î ¤
î í
î ³	
î ù	
î ¿
	ï H	ï H	ï J	ï L	ï N	ï Pï T
ï ²
ï …
ï œ
ï ¡
ï ©
ï Ø
ï Ý
ï å
ï Í
ï Í
ï é
ï é
ï ˆ
ï ª
ï Ë
ï í
ï Ž
ï °
ï Ñ
ï ó
ï 
ï ¦
ï ­
ï µ
ï ¿
ï É
ï 
ï ˆ
ï 
ï •
ï 
ï «
ð 
ð «
ð ô
ð º	
ð €

ð Æ
	ñ P
ñ —
ñ Ü
ñ ó
ñ «
ñ ˆ
ñ 
ñ •
ñ  
ñ Ñ
ñ ×
ñ Û
ñ æ
ñ ó
ñ ‡
ñ ¸

ñ Á

ñ È

ñ Ð

ñ Ú

ñ ä

ñ 
ñ ”
ñ ™
ñ ¡
ñ ©
ñ ·ò Õò Ÿò ºò Çò Ïò Ùò ˜ò Ûò žò á
ó ã
ó ‚
ó ¤
ó Å
ó ç
ó ˆ
ó ª
ó Ë
ó í
ó Ž
ó Ñ
ó š	
ó à	
ó ¦

ó ì

ó ¥
ó ¾
ó è
ó 
ó «
ó Ä
ó î
ó ‡
ó ±
ó Ê
ô ”
õ ¿
ö ù

÷ ä
ø ×
ø –
ø Ù
ø œ
ø ß
ù ü

ú ó
ú ½
û Ä
ü ¼
ü É	ý 	ý !
ý û	þ &
ÿ Û
ÿ š
ÿ Ý
ÿ  
ÿ ã
ÿ »
ÿ þ
ÿ Á
ÿ „
ÿ Ç€ € 
 ±
 
 Ë
 ¥
 «	‚ j
ƒ —
„ Æ
… …	† N
† Œ
† Ë
† Œ
† ‘
† Ã
† Í
† Ñ
† Œ
† š
† Ž
† ”
† ˜
† £
† °
† Ä
† ›
† ä
† ª	
† ð	
† ò	
† û	
† ‚

† Š

† ”

† ž

† ¶

† Ê
† Ñ
† Ö
† Þ
† æ
† ô	‡ 	‡ 	‡ J	‡ v
‡ ¸
‡ Ü
‡ ÿ
‡ ƒ
‡ ·
‡ ¾
‡ È
‡ Þ
‡ ã
‡ ë
‡ ö
‡ ˆ
‡ Ž
‡ ’
‡ 
‡ ª
‡ ¾
‡ ½
‡ æ
‡ ï
‡ ö
‡ þ
‡ †	
‡ ˆ	
‡ ’	
‡ Ì	
‡ ’

‡ Ø

‡ Ä
‡ Ë
‡ Ð
‡ Ø
‡ à
‡ î
ˆ ó
ˆ ´
ˆ ÷
ˆ º
ˆ ý
ˆ º
ˆ ƒ	
ˆ É	
ˆ 

ˆ Õ

ˆ š
ˆ Ý
ˆ  
ˆ ã
ˆ ¦
‰ •	Š =	Š A	Š c‹ o
Œ í
 Ë"

exact_rhs4"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
exact_solution"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8"
llvm.memcpy.p0i8.p0i8.i64*
npb-BT-exact_rhs4.clu
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
<
 
transfer_bytes_log1p
†fA

wgsize_log1p
†fA

transfer_bytes
ø¬n