
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
2addB+
)
	full_text

%19 = add nsw i32 %6, -1
6icmpB.
,
	full_text

%20 = icmp sgt i32 %19, %17
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %17
8brB2
0
	full_text#
!
br i1 %20, label %21, label %83
!i1B

	full_text


i1 %20
Wbitcast8BJ
H
	full_text;
9
7%22 = bitcast double* %0 to [33 x [33 x [5 x double]]]*
5icmp8B+
)
	full_text

%23 = icmp sgt i32 %5, 2
:br8B2
0
	full_text#
!
br i1 %23, label %24, label %83
#i18B

	full_text


i1 %23
4add8B+
)
	full_text

%25 = add nsw i32 %5, -1
4add8B+
)
	full_text

%26 = add nsw i32 %4, -1
5icmp8B+
)
	full_text

%27 = icmp sgt i32 %4, 2
ogetelementptr8B\
Z
	full_textM
K
I%28 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 0
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
1shl8B(
&
	full_text

%29 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%30 = ashr exact i64 %29, 32
%i648B

	full_text
	
i64 %29
ogetelementptr8B\
Z
	full_textM
K
I%31 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 1
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%32 = getelementptr inbounds double, double* %15, i64 1
-double*8B

	full_text

double* %15
ogetelementptr8B\
Z
	full_textM
K
I%33 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 2
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%34 = getelementptr inbounds double, double* %15, i64 2
-double*8B

	full_text

double* %15
ogetelementptr8B\
Z
	full_textM
K
I%35 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 3
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%36 = getelementptr inbounds double, double* %15, i64 3
-double*8B

	full_text

double* %15
ogetelementptr8B\
Z
	full_textM
K
I%37 = getelementptr inbounds [5 x double], [5 x double]* %8, i64 0, i64 4
8[5 x double]*8B#
!
	full_text

[5 x double]* %8
]getelementptr8BJ
H
	full_text;
9
7%38 = getelementptr inbounds double, double* %15, i64 4
-double*8B

	full_text

double* %15
6zext8B,
*
	full_text

%39 = zext i32 %26 to i64
%i328B

	full_text
	
i32 %26
6zext8B,
*
	full_text

%40 = zext i32 %25 to i64
%i328B

	full_text
	
i32 %25
'br8B

	full_text

br label %41
Bphi8B9
7
	full_text*
(
&%42 = phi i64 [ 1, %24 ], [ %81, %80 ]
%i648B

	full_text
	
i64 %81
:br8B2
0
	full_text#
!
br i1 %27, label %43, label %80
#i18B

	full_text


i1 %27
8trunc8B-
+
	full_text

%44 = trunc i64 %42 to i32
%i648B

	full_text
	
i64 %42
'br8B

	full_text

br label %45
Bphi8B9
7
	full_text*
(
&%46 = phi i64 [ 1, %43 ], [ %78, %45 ]
%i648B

	full_text
	
i64 %78
8trunc8B-
+
	full_text

%47 = trunc i64 %46 to i32
%i648B

	full_text
	
i64 %46
lcall8Bb
`
	full_textS
Q
Ocall void @exact(i32 %47, i32 %44, i32 %17, double* nonnull %28, double* %1) #6
%i328B

	full_text
	
i32 %47
%i328B

	full_text
	
i32 %44
%i328B

	full_text
	
i32 %17
-double*8B

	full_text

double* %28
Oload8BE
C
	full_text6
4
2%48 = load double, double* %28, align 16, !tbaa !8
-double*8B

	full_text

double* %28
¢getelementptr8Bé
ã
	full_text~
|
z%49 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %30, i64 %42, i64 %46, i64 0
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%50 = load double, double* %49, align 8, !tbaa !8
-double*8B

	full_text

double* %49
7fsub8B-
+
	full_text

%51 = fsub double %48, %50
+double8B

	full_text


double %48
+double8B

	full_text


double %50
Nload8BD
B
	full_text5
3
1%52 = load double, double* %15, align 8, !tbaa !8
-double*8B

	full_text

double* %15
dcall8BZ
X
	full_textK
I
G%53 = call double @llvm.fmuladd.f64(double %51, double %51, double %52)
+double8B

	full_text


double %51
+double8B

	full_text


double %51
+double8B

	full_text


double %52
Nstore8BC
A
	full_text4
2
0store double %53, double* %15, align 8, !tbaa !8
+double8B

	full_text


double %53
-double*8B

	full_text

double* %15
Nload8BD
B
	full_text5
3
1%54 = load double, double* %31, align 8, !tbaa !8
-double*8B

	full_text

double* %31
¢getelementptr8Bé
ã
	full_text~
|
z%55 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %30, i64 %42, i64 %46, i64 1
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%56 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
7fsub8B-
+
	full_text

%57 = fsub double %54, %56
+double8B

	full_text


double %54
+double8B

	full_text


double %56
Nload8BD
B
	full_text5
3
1%58 = load double, double* %32, align 8, !tbaa !8
-double*8B

	full_text

double* %32
dcall8BZ
X
	full_textK
I
G%59 = call double @llvm.fmuladd.f64(double %57, double %57, double %58)
+double8B

	full_text


double %57
+double8B

	full_text


double %57
+double8B

	full_text


double %58
Nstore8BC
A
	full_text4
2
0store double %59, double* %32, align 8, !tbaa !8
+double8B

	full_text


double %59
-double*8B

	full_text

double* %32
Oload8BE
C
	full_text6
4
2%60 = load double, double* %33, align 16, !tbaa !8
-double*8B

	full_text

double* %33
¢getelementptr8Bé
ã
	full_text~
|
z%61 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %30, i64 %42, i64 %46, i64 2
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%62 = load double, double* %61, align 8, !tbaa !8
-double*8B

	full_text

double* %61
7fsub8B-
+
	full_text

%63 = fsub double %60, %62
+double8B

	full_text


double %60
+double8B

	full_text


double %62
Nload8BD
B
	full_text5
3
1%64 = load double, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
dcall8BZ
X
	full_textK
I
G%65 = call double @llvm.fmuladd.f64(double %63, double %63, double %64)
+double8B

	full_text


double %63
+double8B

	full_text


double %63
+double8B

	full_text


double %64
Nstore8BC
A
	full_text4
2
0store double %65, double* %34, align 8, !tbaa !8
+double8B

	full_text


double %65
-double*8B

	full_text

double* %34
Nload8BD
B
	full_text5
3
1%66 = load double, double* %35, align 8, !tbaa !8
-double*8B

	full_text

double* %35
¢getelementptr8Bé
ã
	full_text~
|
z%67 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %30, i64 %42, i64 %46, i64 3
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%68 = load double, double* %67, align 8, !tbaa !8
-double*8B

	full_text

double* %67
7fsub8B-
+
	full_text

%69 = fsub double %66, %68
+double8B

	full_text


double %66
+double8B

	full_text


double %68
Nload8BD
B
	full_text5
3
1%70 = load double, double* %36, align 8, !tbaa !8
-double*8B

	full_text

double* %36
dcall8BZ
X
	full_textK
I
G%71 = call double @llvm.fmuladd.f64(double %69, double %69, double %70)
+double8B

	full_text


double %69
+double8B

	full_text


double %69
+double8B

	full_text


double %70
Nstore8BC
A
	full_text4
2
0store double %71, double* %36, align 8, !tbaa !8
+double8B

	full_text


double %71
-double*8B

	full_text

double* %36
Oload8BE
C
	full_text6
4
2%72 = load double, double* %37, align 16, !tbaa !8
-double*8B

	full_text

double* %37
¢getelementptr8Bé
ã
	full_text~
|
z%73 = getelementptr inbounds [33 x [33 x [5 x double]]], [33 x [33 x [5 x double]]]* %22, i64 %30, i64 %42, i64 %46, i64 4
U[33 x [33 x [5 x double]]]*8B2
0
	full_text#
!
[33 x [33 x [5 x double]]]* %22
%i648B

	full_text
	
i64 %30
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%74 = load double, double* %73, align 8, !tbaa !8
-double*8B

	full_text

double* %73
7fsub8B-
+
	full_text

%75 = fsub double %72, %74
+double8B

	full_text


double %72
+double8B

	full_text


double %74
Nload8BD
B
	full_text5
3
1%76 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
dcall8BZ
X
	full_textK
I
G%77 = call double @llvm.fmuladd.f64(double %75, double %75, double %76)
+double8B

	full_text


double %75
+double8B

	full_text


double %75
+double8B

	full_text


double %76
Nstore8BC
A
	full_text4
2
0store double %77, double* %38, align 8, !tbaa !8
+double8B

	full_text


double %77
-double*8B

	full_text

double* %38
8add8B/
-
	full_text 

%78 = add nuw nsw i64 %46, 1
%i648B

	full_text
	
i64 %46
7icmp8B-
+
	full_text

%79 = icmp eq i64 %78, %39
%i648B

	full_text
	
i64 %78
%i648B

	full_text
	
i64 %39
:br8B2
0
	full_text#
!
br i1 %79, label %80, label %45
#i18B

	full_text


i1 %79
8add8B/
-
	full_text 

%81 = add nuw nsw i64 %42, 1
%i648B

	full_text
	
i64 %42
7icmp8B-
+
	full_text

%82 = icmp eq i64 %81, %40
%i648B

	full_text
	
i64 %81
%i648B

	full_text
	
i64 %40
:br8B2
0
	full_text#
!
br i1 %82, label %83, label %41
#i18B

	full_text


i1 %82
=call8B3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #8
5icmp8B+
)
	full_text

%84 = icmp eq i32 %18, 0
%i328B

	full_text
	
i32 %18
;br8B3
1
	full_text$
"
 br i1 %84, label %85, label %158
#i18B

	full_text


i1 %84
Jcall8B@
>
	full_text1
/
-%86 = call i64 @_Z14get_local_sizej(i32 0) #7
6icmp8B,
*
	full_text

%87 = icmp ugt i64 %86, 1
%i648B

	full_text
	
i64 %86
:br8B2
0
	full_text#
!
br i1 %87, label %95, label %88
#i18B

	full_text


i1 %87
Abitcast8	B4
2
	full_text%
#
!%89 = bitcast double* %15 to i64*
-double*8	B

	full_text

double* %15
Hload8	B>
<
	full_text/
-
+%90 = load i64, i64* %89, align 8, !tbaa !8
'i64*8	B

	full_text


i64* %89
]getelementptr8	BJ
H
	full_text;
9
7%91 = getelementptr inbounds double, double* %15, i64 1
-double*8	B

	full_text

double* %15
]getelementptr8	BJ
H
	full_text;
9
7%92 = getelementptr inbounds double, double* %15, i64 2
-double*8	B

	full_text

double* %15
]getelementptr8	BJ
H
	full_text;
9
7%93 = getelementptr inbounds double, double* %15, i64 3
-double*8	B

	full_text

double* %15
]getelementptr8	BJ
H
	full_text;
9
7%94 = getelementptr inbounds double, double* %15, i64 4
-double*8	B

	full_text

double* %15
(br8	B 

	full_text

br label %132
Nload8
BD
B
	full_text5
3
1%96 = load double, double* %15, align 8, !tbaa !8
-double*8
B

	full_text

double* %15
]getelementptr8
BJ
H
	full_text;
9
7%97 = getelementptr inbounds double, double* %15, i64 1
-double*8
B

	full_text

double* %15
Nload8
BD
B
	full_text5
3
1%98 = load double, double* %97, align 8, !tbaa !8
-double*8
B

	full_text

double* %97
]getelementptr8
BJ
H
	full_text;
9
7%99 = getelementptr inbounds double, double* %15, i64 2
-double*8
B

	full_text

double* %15
Oload8
BE
C
	full_text6
4
2%100 = load double, double* %99, align 8, !tbaa !8
-double*8
B

	full_text

double* %99
^getelementptr8
BK
I
	full_text<
:
8%101 = getelementptr inbounds double, double* %15, i64 3
-double*8
B

	full_text

double* %15
Pload8
BF
D
	full_text7
5
3%102 = load double, double* %101, align 8, !tbaa !8
.double*8
B

	full_text

double* %101
^getelementptr8
BK
I
	full_text<
:
8%103 = getelementptr inbounds double, double* %15, i64 4
-double*8
B

	full_text

double* %15
Pload8
BF
D
	full_text7
5
3%104 = load double, double* %103, align 8, !tbaa !8
.double*8
B

	full_text

double* %103
(br8
B 

	full_text

br label %105
Kphi8BB
@
	full_text3
1
/%106 = phi double [ %104, %95 ], [ %127, %105 ]
,double8B

	full_text

double %104
,double8B

	full_text

double %127
Kphi8BB
@
	full_text3
1
/%107 = phi double [ %102, %95 ], [ %124, %105 ]
,double8B

	full_text

double %102
,double8B

	full_text

double %124
Kphi8BB
@
	full_text3
1
/%108 = phi double [ %100, %95 ], [ %121, %105 ]
,double8B

	full_text

double %100
,double8B

	full_text

double %121
Jphi8BA
?
	full_text2
0
.%109 = phi double [ %98, %95 ], [ %118, %105 ]
+double8B

	full_text


double %98
,double8B

	full_text

double %118
Jphi8BA
?
	full_text2
0
.%110 = phi double [ %96, %95 ], [ %115, %105 ]
+double8B

	full_text


double %96
,double8B

	full_text

double %115
Ephi8B<
:
	full_text-
+
)%111 = phi i64 [ 1, %95 ], [ %128, %105 ]
&i648B

	full_text


i64 %128
:mul8B1
/
	full_text"
 
%112 = mul nuw nsw i64 %111, 5
&i648B

	full_text


i64 %111
`getelementptr8BM
K
	full_text>
<
:%113 = getelementptr inbounds double, double* %3, i64 %112
&i648B

	full_text


i64 %112
Pload8BF
D
	full_text7
5
3%114 = load double, double* %113, align 8, !tbaa !8
.double*8B

	full_text

double* %113
:fadd8B0
.
	full_text!

%115 = fadd double %114, %110
,double8B

	full_text

double %114
,double8B

	full_text

double %110
Ostore8BD
B
	full_text5
3
1store double %115, double* %15, align 8, !tbaa !8
,double8B

	full_text

double %115
-double*8B

	full_text

double* %15
_getelementptr8BL
J
	full_text=
;
9%116 = getelementptr inbounds double, double* %113, i64 1
.double*8B

	full_text

double* %113
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

%118 = fadd double %117, %109
,double8B

	full_text

double %117
,double8B

	full_text

double %109
Ostore8BD
B
	full_text5
3
1store double %118, double* %97, align 8, !tbaa !8
,double8B

	full_text

double %118
-double*8B

	full_text

double* %97
_getelementptr8BL
J
	full_text=
;
9%119 = getelementptr inbounds double, double* %113, i64 2
.double*8B

	full_text

double* %113
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

%121 = fadd double %120, %108
,double8B

	full_text

double %120
,double8B

	full_text

double %108
Ostore8BD
B
	full_text5
3
1store double %121, double* %99, align 8, !tbaa !8
,double8B

	full_text

double %121
-double*8B

	full_text

double* %99
_getelementptr8BL
J
	full_text=
;
9%122 = getelementptr inbounds double, double* %113, i64 3
.double*8B

	full_text

double* %113
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

%124 = fadd double %123, %107
,double8B

	full_text

double %123
,double8B

	full_text

double %107
Pstore8BE
C
	full_text6
4
2store double %124, double* %101, align 8, !tbaa !8
,double8B

	full_text

double %124
.double*8B

	full_text

double* %101
_getelementptr8BL
J
	full_text=
;
9%125 = getelementptr inbounds double, double* %113, i64 4
.double*8B

	full_text

double* %113
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

%127 = fadd double %126, %106
,double8B

	full_text

double %126
,double8B

	full_text

double %106
Pstore8BE
C
	full_text6
4
2store double %127, double* %103, align 8, !tbaa !8
,double8B

	full_text

double %127
.double*8B

	full_text

double* %103
:add8B1
/
	full_text"
 
%128 = add nuw nsw i64 %111, 1
&i648B

	full_text


i64 %111
9icmp8B/
-
	full_text 

%129 = icmp eq i64 %128, %86
&i648B

	full_text


i64 %128
%i648B

	full_text
	
i64 %86
=br8B5
3
	full_text&
$
"br i1 %129, label %130, label %105
$i18B

	full_text
	
i1 %129
Abitcast8B4
2
	full_text%
#
!%131 = bitcast double %115 to i64
,double8B

	full_text

double %115
(br8B 

	full_text

br label %132
Kphi8BB
@
	full_text3
1
/%133 = phi double* [ %94, %88 ], [ %103, %130 ]
-double*8B

	full_text

double* %94
.double*8B

	full_text

double* %103
Kphi8BB
@
	full_text3
1
/%134 = phi double* [ %93, %88 ], [ %101, %130 ]
-double*8B

	full_text

double* %93
.double*8B

	full_text

double* %101
Jphi8BA
?
	full_text2
0
.%135 = phi double* [ %92, %88 ], [ %99, %130 ]
-double*8B

	full_text

double* %92
-double*8B

	full_text

double* %99
Jphi8BA
?
	full_text2
0
.%136 = phi double* [ %91, %88 ], [ %97, %130 ]
-double*8B

	full_text

double* %91
-double*8B

	full_text

double* %97
Gphi8B>
<
	full_text/
-
+%137 = phi i64 [ %90, %88 ], [ %131, %130 ]
%i648B

	full_text
	
i64 %90
&i648B

	full_text


i64 %131
Icall8B?
=
	full_text0
.
,%138 = call i64 @_Z12get_group_idj(i32 0) #7
2mul8B)
'
	full_text

%139 = mul i64 %138, 5
&i648B

	full_text


i64 %138
`getelementptr8BM
K
	full_text>
<
:%140 = getelementptr inbounds double, double* %2, i64 %139
&i648B

	full_text


i64 %139
Cbitcast8B6
4
	full_text'
%
#%141 = bitcast double* %140 to i64*
.double*8B

	full_text

double* %140
Jstore8B?
=
	full_text0
.
,store i64 %137, i64* %141, align 8, !tbaa !8
&i648B

	full_text


i64 %137
(i64*8B

	full_text

	i64* %141
Cbitcast8B6
4
	full_text'
%
#%142 = bitcast double* %136 to i64*
.double*8B

	full_text

double* %136
Jload8B@
>
	full_text1
/
-%143 = load i64, i64* %142, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %142
_getelementptr8BL
J
	full_text=
;
9%144 = getelementptr inbounds double, double* %140, i64 1
.double*8B

	full_text

double* %140
Cbitcast8B6
4
	full_text'
%
#%145 = bitcast double* %144 to i64*
.double*8B

	full_text

double* %144
Jstore8B?
=
	full_text0
.
,store i64 %143, i64* %145, align 8, !tbaa !8
&i648B

	full_text


i64 %143
(i64*8B

	full_text

	i64* %145
Cbitcast8B6
4
	full_text'
%
#%146 = bitcast double* %135 to i64*
.double*8B

	full_text

double* %135
Jload8B@
>
	full_text1
/
-%147 = load i64, i64* %146, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %146
_getelementptr8BL
J
	full_text=
;
9%148 = getelementptr inbounds double, double* %140, i64 2
.double*8B

	full_text

double* %140
Cbitcast8B6
4
	full_text'
%
#%149 = bitcast double* %148 to i64*
.double*8B

	full_text

double* %148
Jstore8B?
=
	full_text0
.
,store i64 %147, i64* %149, align 8, !tbaa !8
&i648B

	full_text


i64 %147
(i64*8B

	full_text

	i64* %149
Cbitcast8B6
4
	full_text'
%
#%150 = bitcast double* %134 to i64*
.double*8B

	full_text

double* %134
Jload8B@
>
	full_text1
/
-%151 = load i64, i64* %150, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %150
_getelementptr8BL
J
	full_text=
;
9%152 = getelementptr inbounds double, double* %140, i64 3
.double*8B

	full_text

double* %140
Cbitcast8B6
4
	full_text'
%
#%153 = bitcast double* %152 to i64*
.double*8B

	full_text

double* %152
Jstore8B?
=
	full_text0
.
,store i64 %151, i64* %153, align 8, !tbaa !8
&i648B

	full_text


i64 %151
(i64*8B

	full_text

	i64* %153
Cbitcast8B6
4
	full_text'
%
#%154 = bitcast double* %133 to i64*
.double*8B

	full_text

double* %133
Jload8B@
>
	full_text1
/
-%155 = load i64, i64* %154, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %154
_getelementptr8BL
J
	full_text=
;
9%156 = getelementptr inbounds double, double* %140, i64 4
.double*8B

	full_text

double* %140
Cbitcast8B6
4
	full_text'
%
#%157 = bitcast double* %156 to i64*
.double*8B

	full_text

double* %156
Jstore8B?
=
	full_text0
.
,store i64 %155, i64* %157, align 8, !tbaa !8
&i648B

	full_text


i64 %155
(i64*8B

	full_text

	i64* %157
(br8B 

	full_text

br label %158
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


double* %3
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %5
$i328B
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
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 40
#i648B

	full_text	

i64 5
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 1
$i328B

	full_text


i32 -1
!i88B

	full_text

i8 0
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1
-i648B"
 
	full_text

i64 21474836480
#i648B

	full_text	

i64 4
%i18B

	full_text


i1 false
#i328B

	full_text	

i32 2
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 3        		 
 

                     !  " ## $$ %& %% '( '' )* )) +, ++ -. -- /0 // 12 11 34 33 56 55 78 77 9: 99 ;< ;; => == ?A @@ BC BE DD FH GG IJ II KL KM KN KO KK PQ PP RS RT RU RV RR WX WW YZ Y[ YY \] \\ ^_ ^` ^a ^^ bc bd bb ef ee gh gi gj gk gg lm ll no np nn qr qq st su sv ss wx wy ww z{ zz |} |~ | |	Ä || ÅÇ ÅÅ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä à
ã àà åç å
é åå èê èè ëí ë
ì ë
î ë
ï ëë ñó ññ òô ò
ö òò õú õõ ùû ù
ü ù
† ùù °¢ °
£ °° §• §§ ¶ß ¶
® ¶
© ¶
™ ¶¶ ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤
¥ ≤
µ ≤≤ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ªª æø æ¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈« »… »»  À  Ã ÕŒ ÕÕ œ– œ“ —— ”‘ ”” ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬂ ﬁﬁ ‡· ‡‡ ‚„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ ÏÏ ÓÔ ÓÓ Ú Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ Ä
Å ÄÄ ÇÉ ÇÇ Ñ
Ö ÑÑ Üá ÜÜ àâ à
ä àà ãå ã
ç ãã éè éé êë êê íì í
î íí ïñ ï
ó ïï òô òò öõ öö úù ú
û úú ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ ÆÆ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªæ ΩΩ ø¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œœ –— –– “
” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰‰ ÊÁ ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ Ö
á ÜÜ àâ â Ñ	ä Kã å ç “é é "è #è $   	 
     	     ! & (' * , . 0 2 4 6 8 :# <" >¿ A$ C@ Eπ HG JI LD M N% O% Q S) T@ UG VR XP ZW [ ]Y _Y `\ a^ c d+ f h) i@ jG kg me ol p- rn tn uq vs x- y/ { }) ~@ G Ä| Çz ÑÅ Ö1 áÉ âÉ äÜ ãà ç1 é3 ê í) ì@ îG ïë óè ôñ ö5 úò ûò üõ †ù ¢5 £7 • ß) ®@ ©G ™¶ ¨§ Æ´ Ø9 ±≠ ≥≠ ¥∞ µ≤ ∑9 ∏G ∫π º; Ωª ø@ ¡¿ √= ƒ¬ ∆ …» ÀÃ ŒÕ – “— ‘ ÷ ÿ ⁄ ‹ ﬂ ·‡ „ Â‰ Á ÈË Î ÌÏ ÔÓ Ú∞ ÛÍ ı¶ ˆÊ ¯ú ˘‚ ˚í ¸ﬁ ˛à ˇ∂ ÅÄ ÉÇ ÖÑ áÜ â˝ äà å çÑ èé ëê ì˙ îí ñ‡ óÑ ôò õö ù˜ ûú †‰ °Ñ £¢ •§ ßÙ ®¶ ™Ë ´Ñ ≠¨ ØÆ ±Ò ≤∞ ¥Ï µÄ ∑∂ πÃ ∫∏ ºà æ€ ¡Ï ¬Ÿ ƒË ≈◊ «‰ »’  ‡ À” ÕΩ Œœ —– ”“ ’Ã ◊‘ ÿ… ⁄Ÿ ‹“ ﬁ› ‡€ ‚ﬂ „∆ Â‰ Á“ ÈË ÎÊ ÌÍ Ó√ Ô Ú“ ÙÛ ˆÒ ¯ı ˘¿ ˚˙ ˝“ ˇ˛ Å¸ ÉÄ Ñ á  «  "  «  Ã  Ü? @œ ﬁœ —B DB ¿ Ò› ¿F G≈ «≈ @ª Ωª ÒÖ Üæ ¿æ Gø ¿ êê ìì ëë ïï óó ôô ññ òò îî à íí≤ îî ≤Ã óó Ãs îî s ëë 	 íí 	 ôô ^ îî ^à îî àù îî ùK ìì K« ññ «œ òò œÜ ïï Ü êê ö ö 	
ö »ö Ãö œõ 	õ õ Ü
ú Ç
ú –	ù /	ù 1	ù |
ù ◊
ù ‰
ù ò
ù Ëû û «	ü 	ü "	ü #	† 	° %	° %	° +	° /	° 3	° 7	° R	¢ 	¢ +	¢ -¢ @¢ G	¢ g
¢ π
¢ ¿
¢ Õ
¢ ’
¢ ‡¢ Ä
¢ é
¢ ∂
¢ ›	£ 
	§ 7	§ 9
§ ¶
§ €
§ Ï
§ ¨
§ ˛	• 	¶ 	¶ $	ß 	ß '	ß )	® 3	® 5
® ë
® Ÿ
® Ë
® ¢
® Û"
error"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
_Z12get_local_idj"
exact"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8"
_Z7barrierj"
_Z14get_local_sizej"
_Z12get_group_idj"
llvm.memset.p0i8.i64*à
npb-LU-error.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ä

wgsize


wgsize_log1p
äùzA
 
transfer_bytes_log1p
äùzA

devmap_label
 

transfer_bytes
∞∞É