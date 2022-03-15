

[external]
?allocaB5
3
	full_text&
$
"%8 = alloca [5 x double], align 16
BbitcastB7
5
	full_text(
&
$%9 = bitcast [5 x double]* %8 to i8*
6[5 x double]*B#
!
	full_text

[5 x double]* %8
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %9) #6
"i8*B

	full_text


i8* %9
LcallBD
B
	full_text5
3
1%10 = tail call i64 @_Z13get_global_idj(i32 0) #7
.addB'
%
	full_text

%11 = add i64 %10, 1
#i64B

	full_text
	
i64 %10
KcallBC
A
	full_text4
2
0%12 = tail call i64 @_Z12get_local_idj(i32 0) #7
8mulB1
/
	full_text"
 
%13 = mul i64 %12, 21474836480
#i64B

	full_text
	
i64 %12
7ashrB/
-
	full_text 

%14 = ashr exact i64 %13, 32
#i64B

	full_text
	
i64 %13
\getelementptrBK
I
	full_text<
:
8%15 = getelementptr inbounds double, double* %3, i64 %14
#i64B

	full_text
	
i64 %14
>bitcastB3
1
	full_text$
"
 %16 = bitcast double* %15 to i8*
+double*B

	full_text

double* %15
ccallB[
Y
	full_textL
J
Hcall void @llvm.memset.p0i8.i64(i8* align 8 %16, i8 0, i64 40, i1 false)
#i8*B

	full_text
	
i8* %16
6truncB-
+
	full_text

%17 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
6truncB-
+
	full_text

%18 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
5icmpB-
+
	full_text

%19 = icmp slt i32 %17, %6
#i32B

	full_text
	
i32 %17
8brB2
0
	full_text#
!
br i1 %19, label %20, label %86
!i1B

	full_text


i1 %19
Wbitcast8BJ
H
	full_text;
9
7%21 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
=sitofp8B1
/
	full_text"
 
%22 = sitofp i32 %17 to double
%i328B

	full_text
	
i32 %17
Ffmul8B<
:
	full_text-
+
)%23 = fmul double %22, 0x3FB745D1745D1746
+double8B

	full_text


double %22
5icmp8B+
)
	full_text

%24 = icmp sgt i32 %5, 0
:br8B2
0
	full_text#
!
br i1 %24, label %25, label %86
#i18B

	full_text


i1 %24
5icmp8B+
)
	full_text

%26 = icmp sgt i32 %4, 0
ogetelementptr8B\
Z
	full_textM
K
I%27 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
1shl8B(
&
	full_text

%28 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%29 = ashr exact i64 %28, 32
%i648B

	full_text
	
i64 %28
ogetelementptr8B\
Z
	full_textM
K
I%30 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%31 = getelementptr inbounds double, double* %15, i64 1
-double*8B

	full_text

double* %15
ogetelementptr8B\
Z
	full_textM
K
I%32 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%33 = getelementptr inbounds double, double* %15, i64 2
-double*8B

	full_text

double* %15
ogetelementptr8B\
Z
	full_textM
K
I%34 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%35 = getelementptr inbounds double, double* %15, i64 3
-double*8B

	full_text

double* %15
ogetelementptr8B\
Z
	full_textM
K
I%36 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%37 = getelementptr inbounds double, double* %15, i64 4
-double*8B

	full_text

double* %15
5zext8B+
)
	full_text

%38 = zext i32 %4 to i64
5zext8B+
)
	full_text

%39 = zext i32 %5 to i64
'br8B

	full_text

br label %40
Bphi8B9
7
	full_text*
(
&%41 = phi i64 [ 0, %25 ], [ %84, %83 ]
%i648B

	full_text
	
i64 %84
8trunc8B-
+
	full_text

%42 = trunc i64 %41 to i32
%i648B

	full_text
	
i64 %41
=sitofp8B1
/
	full_text"
 
%43 = sitofp i32 %42 to double
%i328B

	full_text
	
i32 %42
Ffmul8B<
:
	full_text-
+
)%44 = fmul double %43, 0x3FB745D1745D1746
+double8B

	full_text


double %43
:br8B2
0
	full_text#
!
br i1 %26, label %45, label %83
#i18B

	full_text


i1 %26
'br8B

	full_text

br label %46
Bphi8B9
7
	full_text*
(
&%47 = phi i64 [ %81, %46 ], [ 0, %45 ]
%i648B

	full_text
	
i64 %81
8trunc8B-
+
	full_text

%48 = trunc i64 %47 to i32
%i648B

	full_text
	
i64 %47
=sitofp8B1
/
	full_text"
 
%49 = sitofp i32 %48 to double
%i328B

	full_text
	
i32 %48
Ffmul8B<
:
	full_text-
+
)%50 = fmul double %49, 0x3FB745D1745D1746
+double8B

	full_text


double %49
~call8Bt
r
	full_texte
c
acall void @exact_solution(double %50, double %44, double %23, double* nonnull %27, double* %1) #6
+double8B

	full_text


double %50
+double8B

	full_text


double %44
+double8B

	full_text


double %23
-double*8B

	full_text

double* %27
¢getelementptr8B
‹
	full_text~
|
z%51 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %29, i64 %41, i64 %47, i64 0
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
Oload8BE
C
	full_text6
4
2%53 = load double, double* %27, align 16, !tbaa !8
-double*8B

	full_text

double* %27
7fsub8B-
+
	full_text

%54 = fsub double %52, %53
+double8B

	full_text


double %52
+double8B

	full_text


double %53
Nload8BD
B
	full_text5
3
1%55 = load double, double* %15, align 8, !tbaa !8
-double*8B

	full_text

double* %15
dcall8BZ
X
	full_textK
I
G%56 = call double @llvm.fmuladd.f64(double %54, double %54, double %55)
+double8B

	full_text


double %54
+double8B

	full_text


double %54
+double8B

	full_text


double %55
Nstore8BC
A
	full_text4
2
0store double %56, double* %15, align 8, !tbaa !8
+double8B

	full_text


double %56
-double*8B

	full_text

double* %15
¢getelementptr8B
‹
	full_text~
|
z%57 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %29, i64 %41, i64 %47, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%58 = load double, double* %57, align 8, !tbaa !8
-double*8B

	full_text

double* %57
Nload8BD
B
	full_text5
3
1%59 = load double, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
7fsub8B-
+
	full_text

%60 = fsub double %58, %59
+double8B

	full_text


double %58
+double8B

	full_text


double %59
Nload8BD
B
	full_text5
3
1%61 = load double, double* %31, align 8, !tbaa !8
-double*8B

	full_text

double* %31
dcall8BZ
X
	full_textK
I
G%62 = call double @llvm.fmuladd.f64(double %60, double %60, double %61)
+double8B

	full_text


double %60
+double8B

	full_text


double %60
+double8B

	full_text


double %61
Nstore8BC
A
	full_text4
2
0store double %62, double* %31, align 8, !tbaa !8
+double8B

	full_text


double %62
-double*8B

	full_text

double* %31
¢getelementptr8B
‹
	full_text~
|
z%63 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %29, i64 %41, i64 %47, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%64 = load double, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
Oload8BE
C
	full_text6
4
2%65 = load double, double* %32, align 16, !tbaa !8
-double*8B

	full_text

double* %32
7fsub8B-
+
	full_text

%66 = fsub double %64, %65
+double8B

	full_text


double %64
+double8B

	full_text


double %65
Nload8BD
B
	full_text5
3
1%67 = load double, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
dcall8BZ
X
	full_textK
I
G%68 = call double @llvm.fmuladd.f64(double %66, double %66, double %67)
+double8B

	full_text


double %66
+double8B

	full_text


double %66
+double8B

	full_text


double %67
Nstore8BC
A
	full_text4
2
0store double %68, double* %33, align 8, !tbaa !8
+double8B

	full_text


double %68
-double*8B

	full_text

double* %33
¢getelementptr8B
‹
	full_text~
|
z%69 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %29, i64 %41, i64 %47, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%70 = load double, double* %69, align 8, !tbaa !8
-double*8B

	full_text

double* %69
Nload8BD
B
	full_text5
3
1%71 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
7fsub8B-
+
	full_text

%72 = fsub double %70, %71
+double8B

	full_text


double %70
+double8B

	full_text


double %71
Nload8BD
B
	full_text5
3
1%73 = load double, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
dcall8BZ
X
	full_textK
I
G%74 = call double @llvm.fmuladd.f64(double %72, double %72, double %73)
+double8B

	full_text


double %72
+double8B

	full_text


double %72
+double8B

	full_text


double %73
Nstore8BC
A
	full_text4
2
0store double %74, double* %35, align 8, !tbaa !8
+double8B

	full_text


double %74
-double*8B

	full_text

double* %35
¢getelementptr8B
‹
	full_text~
|
z%75 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %21, i64 %29, i64 %41, i64 %47, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %21
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %47
Nload8BD
B
	full_text5
3
1%76 = load double, double* %75, align 8, !tbaa !8
-double*8B

	full_text

double* %75
Oload8BE
C
	full_text6
4
2%77 = load double, double* %36, align 16, !tbaa !8
-double*8B

	full_text

double* %36
7fsub8B-
+
	full_text

%78 = fsub double %76, %77
+double8B

	full_text


double %76
+double8B

	full_text


double %77
Nload8BD
B
	full_text5
3
1%79 = load double, double* %37, align 8, !tbaa !8
-double*8B

	full_text

double* %37
dcall8BZ
X
	full_textK
I
G%80 = call double @llvm.fmuladd.f64(double %78, double %78, double %79)
+double8B

	full_text


double %78
+double8B

	full_text


double %78
+double8B

	full_text


double %79
Nstore8BC
A
	full_text4
2
0store double %80, double* %37, align 8, !tbaa !8
+double8B

	full_text


double %80
-double*8B

	full_text

double* %37
8add8B/
-
	full_text 

%81 = add nuw nsw i64 %47, 1
%i648B

	full_text
	
i64 %47
7icmp8B-
+
	full_text

%82 = icmp eq i64 %81, %38
%i648B

	full_text
	
i64 %81
%i648B

	full_text
	
i64 %38
:br8B2
0
	full_text#
!
br i1 %82, label %83, label %46
#i18B

	full_text


i1 %82
8add8B/
-
	full_text 

%84 = add nuw nsw i64 %41, 1
%i648B

	full_text
	
i64 %41
7icmp8B-
+
	full_text

%85 = icmp eq i64 %84, %39
%i648B

	full_text
	
i64 %84
%i648B

	full_text
	
i64 %39
:br8B2
0
	full_text#
!
br i1 %85, label %86, label %40
#i18B

	full_text


i1 %85
=call8B3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #8
5icmp8B+
)
	full_text

%87 = icmp eq i32 %18, 0
%i328B

	full_text
	
i32 %18
;br8B3
1
	full_text$
"
 br i1 %87, label %88, label %161
#i18B

	full_text


i1 %87
Jcall8B@
>
	full_text1
/
-%89 = call i64 @_Z14get_local_sizej(i32 0) #7
6icmp8B,
*
	full_text

%90 = icmp ugt i64 %89, 1
%i648B

	full_text
	
i64 %89
:br8B2
0
	full_text#
!
br i1 %90, label %98, label %91
#i18B

	full_text


i1 %90
Abitcast8	B4
2
	full_text%
#
!%92 = bitcast double* %15 to i64*
-double*8	B

	full_text

double* %15
Hload8	B>
<
	full_text/
-
+%93 = load i64, i64* %92, align 8, !tbaa !8
'i64*8	B

	full_text


i64* %92
]getelementptr8	BJ
H
	full_text;
9
7%94 = getelementptr inbounds double, double* %15, i64 1
-double*8	B

	full_text

double* %15
]getelementptr8	BJ
H
	full_text;
9
7%95 = getelementptr inbounds double, double* %15, i64 2
-double*8	B

	full_text

double* %15
]getelementptr8	BJ
H
	full_text;
9
7%96 = getelementptr inbounds double, double* %15, i64 3
-double*8	B

	full_text

double* %15
]getelementptr8	BJ
H
	full_text;
9
7%97 = getelementptr inbounds double, double* %15, i64 4
-double*8	B

	full_text

double* %15
(br8	B 

	full_text

br label %135
Nload8
BD
B
	full_text5
3
1%99 = load double, double* %15, align 8, !tbaa !8
-double*8
B

	full_text

double* %15
^getelementptr8
BK
I
	full_text<
:
8%100 = getelementptr inbounds double, double* %15, i64 1
-double*8
B

	full_text

double* %15
Pload8
BF
D
	full_text7
5
3%101 = load double, double* %100, align 8, !tbaa !8
.double*8
B

	full_text

double* %100
^getelementptr8
BK
I
	full_text<
:
8%102 = getelementptr inbounds double, double* %15, i64 2
-double*8
B

	full_text

double* %15
Pload8
BF
D
	full_text7
5
3%103 = load double, double* %102, align 8, !tbaa !8
.double*8
B

	full_text

double* %102
^getelementptr8
BK
I
	full_text<
:
8%104 = getelementptr inbounds double, double* %15, i64 3
-double*8
B

	full_text

double* %15
Pload8
BF
D
	full_text7
5
3%105 = load double, double* %104, align 8, !tbaa !8
.double*8
B

	full_text

double* %104
^getelementptr8
BK
I
	full_text<
:
8%106 = getelementptr inbounds double, double* %15, i64 4
-double*8
B

	full_text

double* %15
Pload8
BF
D
	full_text7
5
3%107 = load double, double* %106, align 8, !tbaa !8
.double*8
B

	full_text

double* %106
(br8
B 

	full_text

br label %108
Kphi8BB
@
	full_text3
1
/%109 = phi double [ %107, %98 ], [ %130, %108 ]
,double8B

	full_text

double %107
,double8B

	full_text

double %130
Kphi8BB
@
	full_text3
1
/%110 = phi double [ %105, %98 ], [ %127, %108 ]
,double8B

	full_text

double %105
,double8B

	full_text

double %127
Kphi8BB
@
	full_text3
1
/%111 = phi double [ %103, %98 ], [ %124, %108 ]
,double8B

	full_text

double %103
,double8B

	full_text

double %124
Kphi8BB
@
	full_text3
1
/%112 = phi double [ %101, %98 ], [ %121, %108 ]
,double8B

	full_text

double %101
,double8B

	full_text

double %121
Jphi8BA
?
	full_text2
0
.%113 = phi double [ %99, %98 ], [ %118, %108 ]
+double8B

	full_text


double %99
,double8B

	full_text

double %118
Ephi8B<
:
	full_text-
+
)%114 = phi i64 [ 1, %98 ], [ %131, %108 ]
&i648B

	full_text


i64 %131
:mul8B1
/
	full_text"
 
%115 = mul nuw nsw i64 %114, 5
&i648B

	full_text


i64 %114
`getelementptr8BM
K
	full_text>
<
:%116 = getelementptr inbounds double, double* %3, i64 %115
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%117 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%118 = fadd double %117, %113
,double8B

	full_text

double %117
,double8B

	full_text

double %113
Ostore8BD
B
	full_text5
3
1store double %118, double* %15, align 8, !tbaa !8
,double8B

	full_text

double %118
-double*8B

	full_text

double* %15
_getelementptr8BL
J
	full_text=
;
9%119 = getelementptr inbounds double, double* %116, i64 1
.double*8B

	full_text

double* %116
Pload8BF
D
	full_text7
5
3%120 = load double, double* %119, align 8, !tbaa !8
.double*8B

	full_text

double* %119
:fadd8B0
.
	full_text!

%121 = fadd double %120, %112
,double8B

	full_text

double %120
,double8B

	full_text

double %112
Pstore8BE
C
	full_text6
4
2store double %121, double* %100, align 8, !tbaa !8
,double8B

	full_text

double %121
.double*8B

	full_text

double* %100
_getelementptr8BL
J
	full_text=
;
9%122 = getelementptr inbounds double, double* %116, i64 2
.double*8B

	full_text

double* %116
Pload8BF
D
	full_text7
5
3%123 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
:fadd8B0
.
	full_text!

%124 = fadd double %123, %111
,double8B

	full_text

double %123
,double8B

	full_text

double %111
Pstore8BE
C
	full_text6
4
2store double %124, double* %102, align 8, !tbaa !8
,double8B

	full_text

double %124
.double*8B

	full_text

double* %102
_getelementptr8BL
J
	full_text=
;
9%125 = getelementptr inbounds double, double* %116, i64 3
.double*8B

	full_text

double* %116
Pload8BF
D
	full_text7
5
3%126 = load double, double* %125, align 8, !tbaa !8
.double*8B

	full_text

double* %125
:fadd8B0
.
	full_text!

%127 = fadd double %126, %110
,double8B

	full_text

double %126
,double8B

	full_text

double %110
Pstore8BE
C
	full_text6
4
2store double %127, double* %104, align 8, !tbaa !8
,double8B

	full_text

double %127
.double*8B

	full_text

double* %104
_getelementptr8BL
J
	full_text=
;
9%128 = getelementptr inbounds double, double* %116, i64 4
.double*8B

	full_text

double* %116
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
:fadd8B0
.
	full_text!

%130 = fadd double %129, %109
,double8B

	full_text

double %129
,double8B

	full_text

double %109
Pstore8BE
C
	full_text6
4
2store double %130, double* %106, align 8, !tbaa !8
,double8B

	full_text

double %130
.double*8B

	full_text

double* %106
:add8B1
/
	full_text"
 
%131 = add nuw nsw i64 %114, 1
&i648B

	full_text


i64 %114
9icmp8B/
-
	full_text 

%132 = icmp eq i64 %131, %89
&i648B

	full_text


i64 %131
%i648B

	full_text
	
i64 %89
=br8B5
3
	full_text&
$
"br i1 %132, label %133, label %108
$i18B

	full_text
	
i1 %132
Abitcast8B4
2
	full_text%
#
!%134 = bitcast double %118 to i64
,double8B

	full_text

double %118
(br8B 

	full_text

br label %135
Kphi8BB
@
	full_text3
1
/%136 = phi double* [ %97, %91 ], [ %106, %133 ]
-double*8B

	full_text

double* %97
.double*8B

	full_text

double* %106
Kphi8BB
@
	full_text3
1
/%137 = phi double* [ %96, %91 ], [ %104, %133 ]
-double*8B

	full_text

double* %96
.double*8B

	full_text

double* %104
Kphi8BB
@
	full_text3
1
/%138 = phi double* [ %95, %91 ], [ %102, %133 ]
-double*8B

	full_text

double* %95
.double*8B

	full_text

double* %102
Kphi8BB
@
	full_text3
1
/%139 = phi double* [ %94, %91 ], [ %100, %133 ]
-double*8B

	full_text

double* %94
.double*8B

	full_text

double* %100
Gphi8B>
<
	full_text/
-
+%140 = phi i64 [ %93, %91 ], [ %134, %133 ]
%i648B

	full_text
	
i64 %93
&i648B

	full_text


i64 %134
Icall8B?
=
	full_text0
.
,%141 = call i64 @_Z12get_group_idj(i32 0) #7
2mul8B)
'
	full_text

%142 = mul i64 %141, 5
&i648B

	full_text


i64 %141
`getelementptr8BM
K
	full_text>
<
:%143 = getelementptr inbounds double, double* %2, i64 %142
&i648B

	full_text


i64 %142
Cbitcast8B6
4
	full_text'
%
#%144 = bitcast double* %143 to i64*
.double*8B

	full_text

double* %143
Jstore8B?
=
	full_text0
.
,store i64 %140, i64* %144, align 8, !tbaa !8
&i648B

	full_text


i64 %140
(i64*8B

	full_text

	i64* %144
Cbitcast8B6
4
	full_text'
%
#%145 = bitcast double* %139 to i64*
.double*8B

	full_text

double* %139
Jload8B@
>
	full_text1
/
-%146 = load i64, i64* %145, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %145
_getelementptr8BL
J
	full_text=
;
9%147 = getelementptr inbounds double, double* %143, i64 1
.double*8B

	full_text

double* %143
Cbitcast8B6
4
	full_text'
%
#%148 = bitcast double* %147 to i64*
.double*8B

	full_text

double* %147
Jstore8B?
=
	full_text0
.
,store i64 %146, i64* %148, align 8, !tbaa !8
&i648B

	full_text


i64 %146
(i64*8B

	full_text

	i64* %148
Cbitcast8B6
4
	full_text'
%
#%149 = bitcast double* %138 to i64*
.double*8B

	full_text

double* %138
Jload8B@
>
	full_text1
/
-%150 = load i64, i64* %149, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %149
_getelementptr8BL
J
	full_text=
;
9%151 = getelementptr inbounds double, double* %143, i64 2
.double*8B

	full_text

double* %143
Cbitcast8B6
4
	full_text'
%
#%152 = bitcast double* %151 to i64*
.double*8B

	full_text

double* %151
Jstore8B?
=
	full_text0
.
,store i64 %150, i64* %152, align 8, !tbaa !8
&i648B

	full_text


i64 %150
(i64*8B

	full_text

	i64* %152
Cbitcast8B6
4
	full_text'
%
#%153 = bitcast double* %137 to i64*
.double*8B

	full_text

double* %137
Jload8B@
>
	full_text1
/
-%154 = load i64, i64* %153, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %153
_getelementptr8BL
J
	full_text=
;
9%155 = getelementptr inbounds double, double* %143, i64 3
.double*8B

	full_text

double* %143
Cbitcast8B6
4
	full_text'
%
#%156 = bitcast double* %155 to i64*
.double*8B

	full_text

double* %155
Jstore8B?
=
	full_text0
.
,store i64 %154, i64* %156, align 8, !tbaa !8
&i648B

	full_text


i64 %154
(i64*8B

	full_text

	i64* %156
Cbitcast8B6
4
	full_text'
%
#%157 = bitcast double* %136 to i64*
.double*8B

	full_text

double* %136
Jload8B@
>
	full_text1
/
-%158 = load i64, i64* %157, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %157
_getelementptr8BL
J
	full_text=
;
9%159 = getelementptr inbounds double, double* %143, i64 4
.double*8B

	full_text

double* %143
Cbitcast8B6
4
	full_text'
%
#%160 = bitcast double* %159 to i64*
.double*8B

	full_text

double* %159
Jstore8B?
=
	full_text0
.
,store i64 %158, i64* %160, align 8, !tbaa !8
&i648B

	full_text


i64 %158
(i64*8B

	full_text

	i64* %160
(br8B 

	full_text

br label %161
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %9) #6
$i8*8B

	full_text


i8* %9
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %6
$i328B

	full_text


i32 %5
,double*8B

	full_text


double* %3
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %2
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
-; undefined function B

	full_text

 
-; undefined function 	B

	full_text

 
-; undefined function 
B

	full_text

 
#i648B

	full_text	

i64 5
#i648B

	full_text	

i64 3
#i328B

	full_text	

i32 1
!i88B

	full_text

i8 0
#i648B

	full_text	

i64 1
-i648B"
 
	full_text

i64 21474836480
%i18B

	full_text


i1 false
:double8B,
*
	full_text

double 0x3FB745D1745D1746
#i648B

	full_text	

i64 2
#i648B

	full_text	

i64 0
$i648B

	full_text


i64 40
#i648B

	full_text	

i64 4
$i648B

	full_text


i64 32
#i328B

	full_text	

i32 0        		 
 

                      !! "# "$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;; << =? >> @A @@ BC BB DE DD FG FJ II KL KK MN MM OP OO QR QS QT QU QQ VW VX VY VZ VV [\ [[ ]^ ]] _` _a __ bc bb de df dg dd hi hj hh kl km kn ko kk pq pp rs rr tu tv tt wx ww yz y{ y| yy }~ } }} € €
‚ €
ƒ €
„ €€ …† …… ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ ŒŒ  
 
‘  ’“ ’
” ’’ •– •
— •
˜ •
™ •• š› šš œ œœ Ÿ 
   ¡¢ ¡¡ £¤ £
¥ £
¦ ££ §¨ §
© §§ ª« ª
¬ ª
­ ª
® ªª ¯° ¯¯ ±² ±± ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸
º ¸
» ¸¸ ¼½ ¼
¾ ¼¼ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ ËÍ ÎÏ ÎÎ ĞÑ ĞÒ ÓÔ ÓÓ ÕÖ ÕØ ×× ÙÚ ÙÙ ÛÜ ÛÛ İŞ İİ ßà ßß áâ áá ãå ää æç ææ èé èè êë êê ìí ìì îï îî ğñ ğğ òó òò ôõ ôô öø ÷
ù ÷÷ úû ú
ü úú ış ı
ÿ ıı € €
‚ €€ ƒ„ ƒ
… ƒƒ †
‡ †† ˆ‰ ˆˆ Š
‹ ŠŠ Œ ŒŒ  
  ‘’ ‘
“ ‘‘ ”• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨¨ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²² ´µ ´´ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ ÁÄ ÃÃ ÅÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏĞ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÕ Ö× ÖÖ Ø
Ù ØØ ÚÛ ÚÚ Üİ Ü
Ş ÜÜ ßà ßß áâ áá ãä ãã åæ åå çè ç
é çç êë êê ìí ìì îï îî ğñ ğğ òó ò
ô òò õö õõ ÷ø ÷÷ ùú ùù ûü ûû ış ı
ÿ ıı € €€ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹
 ŒŒ  	 ‘ !‘ <’ ’ Š“ $“ ;	” Q• Ø   	 
     	      ! # & (' * , . 0 2 4 6 8 :Æ ?> A@ CB E$ G¿ JI LK NM PO RD S T% U W) X> YI ZV \% ^[ `] a c_ e_ fb gd i j l) m> nI ok q+ sp ur v- xt zt {w |y ~-  ) ‚> ƒI „€ †/ ˆ… Š‡ ‹1 ‰ ‰ Œ ‘ “1 ” –) —> ˜I ™• ›3 š Ÿœ  5 ¢ ¤ ¥¡ ¦£ ¨5 © «) ¬> ­I ®ª °7 ²¯ ´± µ9 ·³ ¹³ º¶ »¸ ½9 ¾I À¿ Â; ÃÁ Å> ÇÆ É< ÊÈ Ì ÏÎ ÑÒ ÔÓ Ö Ø× Ú Ü Ş à â å çæ é ëê í ïî ñ óò õô ø¶ ùğ û¬ üì ş¢ ÿè ˜ ‚ä „ …¼ ‡† ‰ˆ ‹Š Œ ƒ  ’ “Š •” —– ™€ š˜ œæ Š Ÿ ¡  £ı ¤¢ ¦ê §Š ©¨ «ª ­ú ®¬ °î ±Š ³² µ´ ·÷ ¸¶ ºò »† ½¼ ¿Ò À¾ Â Äá Çò Èß Êî Ëİ Íê ÎÛ Ğæ ÑÙ ÓÃ ÔÕ ×Ö ÙØ ÛÒ İÚ ŞÏ àß âØ äã æá èå éÌ ëê íØ ïî ñì óğ ôÉ öõ øØ úù ü÷ şû ÿÆ € ƒØ …„ ‡‚ ‰† Š   Í" $" ÍĞ ÒĞ Œ= >Õ äÕ ×F HF Æö ÷ã ÆH IË ÍË >Á ÃÁ ÷‹ ŒÄ ÆÄ IÅ Æ œœ ŸŸ  šš —— ›› –– ™™ ˜˜   —— Í œœ Í –– y šš yŒ ›› Œd šš d¸ šš ¸ šš  ŸŸ Q ™™ QÒ  ÒÕ  Õ£ šš £	 ˜˜ 	
  ˆ
  Ö	¡ 3	¡ 5
¡ •
¡ ß
¡ î
¡ ¨
¡ ù¢ ¢ Í	£ 	¤ 	¤ +	¤ -	¤ k
¤ ¿
¤ Æ
¤ Ó
¤ Û
¤ æ¤ †
¤ ”
¤ ¼
¤ ã	¥ 
	¦ 	§ 	§ D	§ O	¨ /	¨ 1
¨ €
¨ İ
¨ ê
¨ 
¨ î	© %	© %	© +	© /	© 3	© 7© >	© I	© Vª 	ª ª Œ	« 7	« 9
« ª
« á
« ò
« ²
« „	¬ 	¬ '	¬ )­ ­ 		­ !	­ $
­ Î­ Ò­ Õ"

error_norm"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
_Z12get_local_idj"
exact_solution"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.memset.p0i8.i64*
npb-BT-error_norm.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize_log1p
†fA

devmap_label
 

transfer_bytes
ø¬n

wgsize

 
transfer_bytes_log1p
†fA