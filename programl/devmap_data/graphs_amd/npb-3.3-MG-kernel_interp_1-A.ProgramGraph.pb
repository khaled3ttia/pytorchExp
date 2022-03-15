

[external]
3sextB+
)
	full_text

%10 = sext i32 %7 to i64
\getelementptrBK
I
	full_text<
:
8%11 = getelementptr inbounds double, double* %0, i64 %10
#i64B

	full_text
	
i64 %10
3sextB+
)
	full_text

%12 = sext i32 %8 to i64
\getelementptrBK
I
	full_text<
:
8%13 = getelementptr inbounds double, double* %0, i64 %12
#i64B

	full_text
	
i64 %12
LcallBD
B
	full_text5
3
1%14 = tail call i64 @_Z13get_global_idj(i32 1) #4
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
KcallBC
A
	full_text4
2
0%16 = tail call i64 @_Z12get_group_idj(i32 0) #4
6truncB-
+
	full_text

%17 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
KcallBC
A
	full_text4
2
0%18 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
.mulB'
%
	full_text

%20 = mul i32 %2, %1
0mulB)
'
	full_text

%21 = mul i32 %20, %15
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %15
2addB+
)
	full_text

%22 = add nsw i32 %17, 1
#i32B

	full_text
	
i32 %17
3mulB,
*
	full_text

%23 = mul nsw i32 %22, %1
#i32B

	full_text
	
i32 %22
0addB)
'
	full_text

%24 = add i32 %21, %19
#i32B

	full_text
	
i32 %21
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%25 = add i32 %24, %23
#i32B

	full_text
	
i32 %24
#i32B

	full_text
	
i32 %23
4sextB,
*
	full_text

%26 = sext i32 %25 to i64
#i32B

	full_text
	
i32 %25
]getelementptrBL
J
	full_text=
;
9%27 = getelementptr inbounds double, double* %11, i64 %26
+double*B

	full_text

double* %11
#i64B

	full_text
	
i64 %26
LloadBD
B
	full_text5
3
1%28 = load double, double* %27, align 8, !tbaa !8
+double*B

	full_text

double* %27
3mulB,
*
	full_text

%29 = mul nsw i32 %17, %1
#i32B

	full_text
	
i32 %17
4addB-
+
	full_text

%30 = add nsw i32 %21, %29
#i32B

	full_text
	
i32 %21
#i32B

	full_text
	
i32 %29
4addB-
+
	full_text

%31 = add nsw i32 %30, %19
#i32B

	full_text
	
i32 %30
#i32B

	full_text
	
i32 %19
4sextB,
*
	full_text

%32 = sext i32 %31 to i64
#i32B

	full_text
	
i32 %31
]getelementptrBL
J
	full_text=
;
9%33 = getelementptr inbounds double, double* %11, i64 %32
+double*B

	full_text

double* %11
#i64B

	full_text
	
i64 %32
LloadBD
B
	full_text5
3
1%34 = load double, double* %33, align 8, !tbaa !8
+double*B

	full_text

double* %33
5faddB-
+
	full_text

%35 = fadd double %28, %34
)doubleB

	full_text


double %28
)doubleB

	full_text


double %34
/shlB(
&
	full_text

%36 = shl i64 %18, 32
#i64B

	full_text
	
i64 %18
7ashrB/
-
	full_text 

%37 = ashr exact i64 %36, 32
#i64B

	full_text
	
i64 %36
ÜgetelementptrBu
s
	full_textf
d
b%38 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_interp_1.z1, i64 0, i64 %37
#i64B

	full_text
	
i64 %37
LstoreBC
A
	full_text4
2
0store double %35, double* %38, align 8, !tbaa !8
)doubleB

	full_text


double %35
+double*B

	full_text

double* %38
2addB+
)
	full_text

%39 = add nsw i32 %15, 1
#i32B

	full_text
	
i32 %15
0mulB)
'
	full_text

%40 = mul i32 %20, %39
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %39
0addB)
'
	full_text

%41 = add i32 %29, %19
#i32B

	full_text
	
i32 %29
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%42 = add i32 %41, %40
#i32B

	full_text
	
i32 %41
#i32B

	full_text
	
i32 %40
4sextB,
*
	full_text

%43 = sext i32 %42 to i64
#i32B

	full_text
	
i32 %42
]getelementptrBL
J
	full_text=
;
9%44 = getelementptr inbounds double, double* %11, i64 %43
+double*B

	full_text

double* %11
#i64B

	full_text
	
i64 %43
LloadBD
B
	full_text5
3
1%45 = load double, double* %44, align 8, !tbaa !8
+double*B

	full_text

double* %44
5faddB-
+
	full_text

%46 = fadd double %34, %45
)doubleB

	full_text


double %34
)doubleB

	full_text


double %45
ÜgetelementptrBu
s
	full_textf
d
b%47 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_interp_1.z2, i64 0, i64 %37
#i64B

	full_text
	
i64 %37
LstoreBC
A
	full_text4
2
0store double %46, double* %47, align 8, !tbaa !8
)doubleB

	full_text


double %46
+double*B

	full_text

double* %47
0addB)
'
	full_text

%48 = add i32 %23, %19
#i32B

	full_text
	
i32 %23
#i32B

	full_text
	
i32 %19
0addB)
'
	full_text

%49 = add i32 %48, %40
#i32B

	full_text
	
i32 %48
#i32B

	full_text
	
i32 %40
4sextB,
*
	full_text

%50 = sext i32 %49 to i64
#i32B

	full_text
	
i32 %49
]getelementptrBL
J
	full_text=
;
9%51 = getelementptr inbounds double, double* %11, i64 %50
+double*B

	full_text

double* %11
#i64B

	full_text
	
i64 %50
LloadBD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
+double*B

	full_text

double* %51
5faddB-
+
	full_text

%53 = fadd double %45, %52
)doubleB

	full_text


double %45
)doubleB

	full_text


double %52
5faddB-
+
	full_text

%54 = fadd double %35, %53
)doubleB

	full_text


double %35
)doubleB

	full_text


double %53
ÜgetelementptrBu
s
	full_textf
d
b%55 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_interp_1.z3, i64 0, i64 %37
#i64B

	full_text
	
i64 %37
LstoreBC
A
	full_text4
2
0store double %54, double* %55, align 8, !tbaa !8
)doubleB

	full_text


double %54
+double*B

	full_text

double* %55
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
2addB+
)
	full_text

%56 = add nsw i32 %1, -1
6icmpB.
,
	full_text

%57 = icmp sgt i32 %56, %19
#i32B

	full_text
	
i32 %56
#i32B

	full_text
	
i32 %19
9brB3
1
	full_text$
"
 br i1 %57, label %58, label %133
!i1B

	full_text


i1 %57
Nload8BD
B
	full_text5
3
1%59 = load double, double* %33, align 8, !tbaa !8
-double*8B

	full_text

double* %33
4shl8B+
)
	full_text

%60 = shl nsw i32 %15, 1
%i328B

	full_text
	
i32 %15
0mul8B'
%
	full_text

%61 = mul i32 %5, %4
2mul8B)
'
	full_text

%62 = mul i32 %61, %60
%i328B

	full_text
	
i32 %61
%i328B

	full_text
	
i32 %60
4shl8B+
)
	full_text

%63 = shl nsw i32 %17, 1
%i328B

	full_text
	
i32 %17
5mul8B,
*
	full_text

%64 = mul nsw i32 %63, %4
%i328B

	full_text
	
i32 %63
6add8B-
+
	full_text

%65 = add nsw i32 %62, %64
%i328B

	full_text
	
i32 %62
%i328B

	full_text
	
i32 %64
4shl8B+
)
	full_text

%66 = shl nsw i32 %19, 1
%i328B

	full_text
	
i32 %19
6add8B-
+
	full_text

%67 = add nsw i32 %65, %66
%i328B

	full_text
	
i32 %65
%i328B

	full_text
	
i32 %66
6sext8B,
*
	full_text

%68 = sext i32 %67 to i64
%i328B

	full_text
	
i32 %67
_getelementptr8BL
J
	full_text=
;
9%69 = getelementptr inbounds double, double* %13, i64 %68
-double*8B

	full_text

double* %13
%i648B

	full_text
	
i64 %68
Nload8BD
B
	full_text5
3
1%70 = load double, double* %69, align 8, !tbaa !8
-double*8B

	full_text

double* %69
7fadd8B-
+
	full_text

%71 = fadd double %59, %70
+double8B

	full_text


double %59
+double8B

	full_text


double %70
Nstore8BC
A
	full_text4
2
0store double %71, double* %69, align 8, !tbaa !8
+double8B

	full_text


double %71
-double*8B

	full_text

double* %69
4add8B+
)
	full_text

%72 = add nsw i32 %31, 1
%i328B

	full_text
	
i32 %31
6sext8B,
*
	full_text

%73 = sext i32 %72 to i64
%i328B

	full_text
	
i32 %72
_getelementptr8BL
J
	full_text=
;
9%74 = getelementptr inbounds double, double* %11, i64 %73
-double*8B

	full_text

double* %11
%i648B

	full_text
	
i64 %73
Nload8BD
B
	full_text5
3
1%75 = load double, double* %74, align 8, !tbaa !8
-double*8B

	full_text

double* %74
7fadd8B-
+
	full_text

%76 = fadd double %59, %75
+double8B

	full_text


double %59
+double8B

	full_text


double %75
.or8B&
$
	full_text

%77 = or i32 %67, 1
%i328B

	full_text
	
i32 %67
6sext8B,
*
	full_text

%78 = sext i32 %77 to i64
%i328B

	full_text
	
i32 %77
_getelementptr8BL
J
	full_text=
;
9%79 = getelementptr inbounds double, double* %13, i64 %78
-double*8B

	full_text

double* %13
%i648B

	full_text
	
i64 %78
Nload8BD
B
	full_text5
3
1%80 = load double, double* %79, align 8, !tbaa !8
-double*8B

	full_text

double* %79
rcall8Bh
f
	full_textY
W
U%81 = tail call double @llvm.fmuladd.f64(double %76, double 5.000000e-01, double %80)
+double8B

	full_text


double %76
+double8B

	full_text


double %80
Nstore8BC
A
	full_text4
2
0store double %81, double* %79, align 8, !tbaa !8
+double8B

	full_text


double %81
-double*8B

	full_text

double* %79
Nload8BD
B
	full_text5
3
1%82 = load double, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
.or8B&
$
	full_text

%83 = or i32 %63, 1
%i328B

	full_text
	
i32 %63
5mul8B,
*
	full_text

%84 = mul nsw i32 %83, %4
%i328B

	full_text
	
i32 %83
2add8B)
'
	full_text

%85 = add i32 %66, %62
%i328B

	full_text
	
i32 %66
%i328B

	full_text
	
i32 %62
2add8B)
'
	full_text

%86 = add i32 %85, %84
%i328B

	full_text
	
i32 %85
%i328B

	full_text
	
i32 %84
6sext8B,
*
	full_text

%87 = sext i32 %86 to i64
%i328B

	full_text
	
i32 %86
_getelementptr8BL
J
	full_text=
;
9%88 = getelementptr inbounds double, double* %13, i64 %87
-double*8B

	full_text

double* %13
%i648B

	full_text
	
i64 %87
Nload8BD
B
	full_text5
3
1%89 = load double, double* %88, align 8, !tbaa !8
-double*8B

	full_text

double* %88
rcall8Bh
f
	full_textY
W
U%90 = tail call double @llvm.fmuladd.f64(double %82, double 5.000000e-01, double %89)
+double8B

	full_text


double %82
+double8B

	full_text


double %89
Nstore8BC
A
	full_text4
2
0store double %90, double* %88, align 8, !tbaa !8
+double8B

	full_text


double %90
-double*8B

	full_text

double* %88
9add8B0
.
	full_text!

%91 = add i64 %36, 4294967296
%i648B

	full_text
	
i64 %36
9ashr8B/
-
	full_text 

%92 = ashr exact i64 %91, 32
%i648B

	full_text
	
i64 %91
àgetelementptr8Bu
s
	full_textf
d
b%93 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_interp_1.z1, i64 0, i64 %92
%i648B

	full_text
	
i64 %92
Nload8BD
B
	full_text5
3
1%94 = load double, double* %93, align 8, !tbaa !8
-double*8B

	full_text

double* %93
7fadd8B-
+
	full_text

%95 = fadd double %82, %94
+double8B

	full_text


double %82
+double8B

	full_text


double %94
4add8B+
)
	full_text

%96 = add nsw i32 %86, 1
%i328B

	full_text
	
i32 %86
6sext8B,
*
	full_text

%97 = sext i32 %96 to i64
%i328B

	full_text
	
i32 %96
_getelementptr8BL
J
	full_text=
;
9%98 = getelementptr inbounds double, double* %13, i64 %97
-double*8B

	full_text

double* %13
%i648B

	full_text
	
i64 %97
Nload8BD
B
	full_text5
3
1%99 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
scall8Bi
g
	full_textZ
X
V%100 = tail call double @llvm.fmuladd.f64(double %95, double 2.500000e-01, double %99)
+double8B

	full_text


double %95
+double8B

	full_text


double %99
Ostore8BD
B
	full_text5
3
1store double %100, double* %98, align 8, !tbaa !8
,double8B

	full_text

double %100
-double*8B

	full_text

double* %98
Oload8BE
C
	full_text6
4
2%101 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
/or8B'
%
	full_text

%102 = or i32 %60, 1
%i328B

	full_text
	
i32 %60
4mul8B+
)
	full_text

%103 = mul i32 %61, %102
%i328B

	full_text
	
i32 %61
&i328B

	full_text


i32 %102
3add8B*
(
	full_text

%104 = add i32 %66, %64
%i328B

	full_text
	
i32 %66
%i328B

	full_text
	
i32 %64
5add8B,
*
	full_text

%105 = add i32 %104, %103
&i328B

	full_text


i32 %104
&i328B

	full_text


i32 %103
8sext8B.
,
	full_text

%106 = sext i32 %105 to i64
&i328B

	full_text


i32 %105
agetelementptr8BN
L
	full_text?
=
;%107 = getelementptr inbounds double, double* %13, i64 %106
-double*8B

	full_text

double* %13
&i648B

	full_text


i64 %106
Pload8BF
D
	full_text7
5
3%108 = load double, double* %107, align 8, !tbaa !8
.double*8B

	full_text

double* %107
ucall8Bk
i
	full_text\
Z
X%109 = tail call double @llvm.fmuladd.f64(double %101, double 5.000000e-01, double %108)
,double8B

	full_text

double %101
,double8B

	full_text

double %108
Pstore8BE
C
	full_text6
4
2store double %109, double* %107, align 8, !tbaa !8
,double8B

	full_text

double %109
.double*8B

	full_text

double* %107
âgetelementptr8Bv
t
	full_textg
e
c%110 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_interp_1.z2, i64 0, i64 %92
%i648B

	full_text
	
i64 %92
Pload8BF
D
	full_text7
5
3%111 = load double, double* %110, align 8, !tbaa !8
.double*8B

	full_text

double* %110
:fadd8B0
.
	full_text!

%112 = fadd double %101, %111
,double8B

	full_text

double %101
,double8B

	full_text

double %111
6add8B-
+
	full_text

%113 = add nsw i32 %105, 1
&i328B

	full_text


i32 %105
8sext8B.
,
	full_text

%114 = sext i32 %113 to i64
&i328B

	full_text


i32 %113
agetelementptr8BN
L
	full_text?
=
;%115 = getelementptr inbounds double, double* %13, i64 %114
-double*8B

	full_text

double* %13
&i648B

	full_text


i64 %114
Pload8BF
D
	full_text7
5
3%116 = load double, double* %115, align 8, !tbaa !8
.double*8B

	full_text

double* %115
ucall8Bk
i
	full_text\
Z
X%117 = tail call double @llvm.fmuladd.f64(double %112, double 2.500000e-01, double %116)
,double8B

	full_text

double %112
,double8B

	full_text

double %116
Pstore8BE
C
	full_text6
4
2store double %117, double* %115, align 8, !tbaa !8
,double8B

	full_text

double %117
.double*8B

	full_text

double* %115
Oload8BE
C
	full_text6
4
2%118 = load double, double* %55, align 8, !tbaa !8
-double*8B

	full_text

double* %55
3add8B*
(
	full_text

%119 = add i32 %84, %66
%i328B

	full_text
	
i32 %84
%i328B

	full_text
	
i32 %66
5add8B,
*
	full_text

%120 = add i32 %119, %103
&i328B

	full_text


i32 %119
&i328B

	full_text


i32 %103
8sext8B.
,
	full_text

%121 = sext i32 %120 to i64
&i328B

	full_text


i32 %120
agetelementptr8BN
L
	full_text?
=
;%122 = getelementptr inbounds double, double* %13, i64 %121
-double*8B

	full_text

double* %13
&i648B

	full_text


i64 %121
Pload8BF
D
	full_text7
5
3%123 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
ucall8Bk
i
	full_text\
Z
X%124 = tail call double @llvm.fmuladd.f64(double %118, double 2.500000e-01, double %123)
,double8B

	full_text

double %118
,double8B

	full_text

double %123
Pstore8BE
C
	full_text6
4
2store double %124, double* %122, align 8, !tbaa !8
,double8B

	full_text

double %124
.double*8B

	full_text

double* %122
âgetelementptr8Bv
t
	full_textg
e
c%125 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_interp_1.z3, i64 0, i64 %92
%i648B

	full_text
	
i64 %92
Pload8BF
D
	full_text7
5
3%126 = load double, double* %125, align 8, !tbaa !8
.double*8B

	full_text

double* %125
:fadd8B0
.
	full_text!

%127 = fadd double %118, %126
,double8B

	full_text

double %118
,double8B

	full_text

double %126
6add8B-
+
	full_text

%128 = add nsw i32 %120, 1
&i328B

	full_text


i32 %120
8sext8B.
,
	full_text

%129 = sext i32 %128 to i64
&i328B

	full_text


i32 %128
agetelementptr8BN
L
	full_text?
=
;%130 = getelementptr inbounds double, double* %13, i64 %129
-double*8B

	full_text

double* %13
&i648B

	full_text


i64 %129
Pload8BF
D
	full_text7
5
3%131 = load double, double* %130, align 8, !tbaa !8
.double*8B

	full_text

double* %130
ucall8Bk
i
	full_text\
Z
X%132 = tail call double @llvm.fmuladd.f64(double %127, double 1.250000e-01, double %131)
,double8B

	full_text

double %127
,double8B

	full_text

double %131
Pstore8BE
C
	full_text6
4
2store double %132, double* %130, align 8, !tbaa !8
,double8B

	full_text

double %132
.double*8B

	full_text

double* %130
(br8B 

	full_text

br label %133
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %8
$i328B

	full_text


i32 %5
$i328B

	full_text


i32 %1
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %0
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
$i648B

	full_text


i64 32
,i648B!

	full_text

i64 4294967296
4double8B&
$
	full_text

double 2.500000e-01
#i328B

	full_text	

i32 1
}[1024 x double]*8Be
c
	full_textV
T
R@kernel_interp_1.z2 = internal unnamed_addr global [1024 x double] undef, align 16
4double8B&
$
	full_text

double 5.000000e-01
4double8B&
$
	full_text

double 1.250000e-01
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
}[1024 x double]*8Be
c
	full_textV
T
R@kernel_interp_1.z1 = internal unnamed_addr global [1024 x double] undef, align 16
}[1024 x double]*8Be
c
	full_textV
T
R@kernel_interp_1.z3 = internal unnamed_addr global [1024 x double] undef, align 16       	  

                       !  "    #$ ## %& %% '( ') '' *+ *, ** -. -- /0 /1 // 23 22 45 46 44 78 77 9: 99 ;< ;; => =? == @A @@ BC BD BB EF EG EE HI HJ HH KL KK MN MO MM PQ PP RS RT RR UV UU WX WY WW Z[ Z\ ZZ ]^ ]_ ]] `a `` bc bd bb ef ee gh gi gg jk jl jj mn mm op oq oo rr ss tu tv tt wx wz yy {| {{ }} ~ ~	Ä ~~ ÅÇ ÅÅ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ àà äã ä
å ää çé çç èê è
ë èè íì íí îï î
ñ îî óò ó
ô óó öõ öö úù úú ûü û
† ûû °¢ °° £§ £
• ££ ¶ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑∏ ∑∑ π∫ ππ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆∆ »… »
  »» ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –– “
” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡‡ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ Ç
É ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ãã çé ç
è çç êë êê íì í
î íí ïñ ï
ó ïï òô òò öõ ö
ú öö ùû ù
ü ùù †° †† ¢£ ¢
§ ¢¢ •¶ •• ß® ß
© ßß ™´ ™
¨ ™™ ≠
Æ ≠≠ Ø∞ ØØ ±≤ ±
≥ ±± ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ªª Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √≈ ∆ « » }	… 	… 	… %… s	  }
  É
  πÀ À    	
            ! "  $ & (% )' + ,* . 0- 1/ 3# 52 6 87 :9 <4 >; ? A C@ D% F GE IB JH L NK OM Q2 SP T9 VR XU Y [ \Z ^B _] a c` db fP he i4 kg l9 nj pm qs u vt x/ z |} { Ä ÇÅ Ñ~ ÜÉ á âÖ ãà åä é êç ëè ìy ïí ñî òè ô* õö ù üú †û ¢y §° •ä ß¶ © ´® ¨™ Æ£ ∞≠ ±Ø ≥™ ¥; ∂Å ∏∑ ∫à º~ Ωª øπ ¿æ ¬ ƒ¡ ≈√ «µ …∆  » Ã√ Õ7 œŒ —– ”“ ’µ ◊‘ ÿæ ⁄Ÿ ‹ ﬁ€ ﬂ› ·÷ „‡ ‰‚ Ê› ÁU È{ Î} ÌÍ Óà É ÒÔ ÛÏ ÙÚ ˆ ¯ı ˘˜ ˚Ë ˝˙ ˛¸ Ä˜ Å– ÉÇ ÖË áÑ àÚ äâ å éã èç ëÜ ìê îí ñç óm ôπ õà úö ûÏ üù ° £† §¢ ¶ò ®• ©ß ´¢ ¨– Æ≠ ∞ò ≤Ø ≥ù µ¥ ∑ π∂ ∫∏ º± æª øΩ ¡∏ ¬w yw ƒ√ ƒ ƒ œœ –– ÕÕ ŒŒ ÃÃí –– íß –– ß¸ –– ¸Ω –– Ωr œœ r
 ÕÕ 
 ÃÃ  ŒŒ » –– »‚ –– ‚Ø –– Ø	— 7	— 9
— –
“ Œ
” ‚
” í
” ß‘ 	‘ 	‘ @‘ r	‘ {
‘ Å
‘ à
‘ ö
‘ ¶
‘ ∑
‘ Ÿ
‘ Í
‘ â
‘ ¥’ U’ Ç
÷ Ø
÷ »
÷ ¸
◊ Ω	ÿ s	Ÿ ;	Ÿ U	Ÿ m
Ÿ “
Ÿ Ç
Ÿ ≠⁄ 
⁄ € ;€ “‹ m‹ ≠"
kernel_interp_1"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f64*í
npb-MG-kernel_interp_1.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize_log1p
îﬁüA

devmap_label


transfer_bytes	
∞ÊÃ„

wgsize
&
 
transfer_bytes_log1p
îﬁüA