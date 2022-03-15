

[external]
3sextB+
)
	full_text

%13 = sext i32 %7 to i64
\getelementptrBK
I
	full_text<
:
8%14 = getelementptr inbounds double, double* %0, i64 %13
#i64B

	full_text
	
i64 %13
KcallBC
A
	full_text4
2
0%15 = tail call i64 @_Z12get_group_idj(i32 0) #4
2addB+
)
	full_text

%16 = add nsw i32 %5, -2
4sextB,
*
	full_text

%17 = sext i32 %16 to i64
#i32B

	full_text
	
i32 %16
2udivB*
(
	full_text

%18 = udiv i64 %15, %17
#i64B

	full_text
	
i64 %15
#i64B

	full_text
	
i64 %17
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
2uremB*
(
	full_text

%21 = urem i64 %15, %17
#i64B

	full_text
	
i64 %15
#i64B

	full_text
	
i64 %17
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
KcallBC
A
	full_text4
2
0%24 = tail call i64 @_Z12get_local_idj(i32 0) #4
.addB'
%
	full_text

%25 = add i64 %24, 1
#i64B

	full_text
	
i64 %24
6truncB-
+
	full_text

%26 = trunc i64 %25 to i32
#i64B

	full_text
	
i64 %25
2shlB+
)
	full_text

%27 = shl nsw i32 %20, 1
#i32B

	full_text
	
i32 %20
4subB-
+
	full_text

%28 = sub nsw i32 %27, %11
#i32B

	full_text
	
i32 %27
2shlB+
)
	full_text

%29 = shl nsw i32 %23, 1
#i32B

	full_text
	
i32 %23
4subB-
+
	full_text

%30 = sub nsw i32 %29, %10
#i32B

	full_text
	
i32 %29
2shlB+
)
	full_text

%31 = shl nsw i32 %26, 1
#i32B

	full_text
	
i32 %26
3subB,
*
	full_text

%32 = sub nsw i32 %31, %9
#i32B

	full_text
	
i32 %31
2addB+
)
	full_text

%33 = add nsw i32 %28, 1
#i32B

	full_text
	
i32 %28
.mulB'
%
	full_text

%34 = mul i32 %2, %1
0mulB)
'
	full_text

%35 = mul i32 %34, %33
#i32B

	full_text
	
i32 %34
#i32B

	full_text
	
i32 %33
3mulB,
*
	full_text

%36 = mul nsw i32 %30, %1
#i32B

	full_text
	
i32 %30
0addB)
'
	full_text

%37 = add i32 %32, %36
#i32B

	full_text
	
i32 %32
#i32B

	full_text
	
i32 %36
0addB)
'
	full_text

%38 = add i32 %37, %35
#i32B

	full_text
	
i32 %37
#i32B

	full_text
	
i32 %35
4sextB,
*
	full_text

%39 = sext i32 %38 to i64
#i32B

	full_text
	
i32 %38
]getelementptrBL
J
	full_text=
;
9%40 = getelementptr inbounds double, double* %14, i64 %39
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %39
LloadBD
B
	full_text5
3
1%41 = load double, double* %40, align 8, !tbaa !8
+double*B

	full_text

double* %40
2addB+
)
	full_text

%42 = add nsw i32 %30, 2
#i32B

	full_text
	
i32 %30
3mulB,
*
	full_text

%43 = mul nsw i32 %42, %1
#i32B

	full_text
	
i32 %42
0addB)
'
	full_text

%44 = add i32 %43, %32
#i32B

	full_text
	
i32 %43
#i32B

	full_text
	
i32 %32
0addB)
'
	full_text

%45 = add i32 %44, %35
#i32B

	full_text
	
i32 %44
#i32B

	full_text
	
i32 %35
4sextB,
*
	full_text

%46 = sext i32 %45 to i64
#i32B

	full_text
	
i32 %45
]getelementptrBL
J
	full_text=
;
9%47 = getelementptr inbounds double, double* %14, i64 %46
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %46
LloadBD
B
	full_text5
3
1%48 = load double, double* %47, align 8, !tbaa !8
+double*B

	full_text

double* %47
5faddB-
+
	full_text

%49 = fadd double %41, %48
)doubleB

	full_text


double %41
)doubleB

	full_text


double %48
0mulB)
'
	full_text

%50 = mul i32 %34, %28
#i32B

	full_text
	
i32 %34
#i32B

	full_text
	
i32 %28
2addB+
)
	full_text

%51 = add nsw i32 %30, 1
#i32B

	full_text
	
i32 %30
3mulB,
*
	full_text

%52 = mul nsw i32 %51, %1
#i32B

	full_text
	
i32 %51
0addB)
'
	full_text

%53 = add i32 %32, %50
#i32B

	full_text
	
i32 %32
#i32B

	full_text
	
i32 %50
0addB)
'
	full_text

%54 = add i32 %53, %52
#i32B

	full_text
	
i32 %53
#i32B

	full_text
	
i32 %52
4sextB,
*
	full_text

%55 = sext i32 %54 to i64
#i32B

	full_text
	
i32 %54
]getelementptrBL
J
	full_text=
;
9%56 = getelementptr inbounds double, double* %14, i64 %55
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %55
LloadBD
B
	full_text5
3
1%57 = load double, double* %56, align 8, !tbaa !8
+double*B

	full_text

double* %56
5faddB-
+
	full_text

%58 = fadd double %49, %57
)doubleB

	full_text


double %49
)doubleB

	full_text


double %57
2addB+
)
	full_text

%59 = add nsw i32 %28, 2
#i32B

	full_text
	
i32 %28
0mulB)
'
	full_text

%60 = mul i32 %34, %59
#i32B

	full_text
	
i32 %34
#i32B

	full_text
	
i32 %59
0addB)
'
	full_text

%61 = add i32 %52, %32
#i32B

	full_text
	
i32 %52
#i32B

	full_text
	
i32 %32
0addB)
'
	full_text

%62 = add i32 %61, %60
#i32B

	full_text
	
i32 %61
#i32B

	full_text
	
i32 %60
4sextB,
*
	full_text

%63 = sext i32 %62 to i64
#i32B

	full_text
	
i32 %62
]getelementptrBL
J
	full_text=
;
9%64 = getelementptr inbounds double, double* %14, i64 %63
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %63
LloadBD
B
	full_text5
3
1%65 = load double, double* %64, align 8, !tbaa !8
+double*B

	full_text

double* %64
5faddB-
+
	full_text

%66 = fadd double %58, %65
)doubleB

	full_text


double %58
)doubleB

	full_text


double %65
4sextB,
*
	full_text

%67 = sext i32 %32 to i64
#i32B

	full_text
	
i32 %32
ƒgetelementptrBr
p
	full_textc
a
_%68 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_rprj3.x1, i64 0, i64 %67
#i64B

	full_text
	
i64 %67
LstoreBC
A
	full_text4
2
0store double %66, double* %68, align 8, !tbaa !8
)doubleB

	full_text


double %66
+double*B

	full_text

double* %68
4addB-
+
	full_text

%69 = add nsw i32 %50, %36
#i32B

	full_text
	
i32 %50
#i32B

	full_text
	
i32 %36
4addB-
+
	full_text

%70 = add nsw i32 %69, %32
#i32B

	full_text
	
i32 %69
#i32B

	full_text
	
i32 %32
4sextB,
*
	full_text

%71 = sext i32 %70 to i64
#i32B

	full_text
	
i32 %70
]getelementptrBL
J
	full_text=
;
9%72 = getelementptr inbounds double, double* %14, i64 %71
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %71
LloadBD
B
	full_text5
3
1%73 = load double, double* %72, align 8, !tbaa !8
+double*B

	full_text

double* %72
0addB)
'
	full_text

%74 = add i32 %37, %60
#i32B

	full_text
	
i32 %37
#i32B

	full_text
	
i32 %60
4sextB,
*
	full_text

%75 = sext i32 %74 to i64
#i32B

	full_text
	
i32 %74
]getelementptrBL
J
	full_text=
;
9%76 = getelementptr inbounds double, double* %14, i64 %75
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %75
LloadBD
B
	full_text5
3
1%77 = load double, double* %76, align 8, !tbaa !8
+double*B

	full_text

double* %76
5faddB-
+
	full_text

%78 = fadd double %73, %77
)doubleB

	full_text


double %73
)doubleB

	full_text


double %77
0addB)
'
	full_text

%79 = add i32 %53, %43
#i32B

	full_text
	
i32 %53
#i32B

	full_text
	
i32 %43
4sextB,
*
	full_text

%80 = sext i32 %79 to i64
#i32B

	full_text
	
i32 %79
]getelementptrBL
J
	full_text=
;
9%81 = getelementptr inbounds double, double* %14, i64 %80
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %80
LloadBD
B
	full_text5
3
1%82 = load double, double* %81, align 8, !tbaa !8
+double*B

	full_text

double* %81
5faddB-
+
	full_text

%83 = fadd double %78, %82
)doubleB

	full_text


double %78
)doubleB

	full_text


double %82
0addB)
'
	full_text

%84 = add i32 %44, %60
#i32B

	full_text
	
i32 %44
#i32B

	full_text
	
i32 %60
4sextB,
*
	full_text

%85 = sext i32 %84 to i64
#i32B

	full_text
	
i32 %84
]getelementptrBL
J
	full_text=
;
9%86 = getelementptr inbounds double, double* %14, i64 %85
+double*B

	full_text

double* %14
#i64B

	full_text
	
i64 %85
LloadBD
B
	full_text5
3
1%87 = load double, double* %86, align 8, !tbaa !8
+double*B

	full_text

double* %86
5faddB-
+
	full_text

%88 = fadd double %83, %87
)doubleB

	full_text


double %83
)doubleB

	full_text


double %87
ƒgetelementptrBr
p
	full_textc
a
_%89 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_rprj3.y1, i64 0, i64 %67
#i64B

	full_text
	
i64 %67
LstoreBC
A
	full_text4
2
0store double %88, double* %89, align 8, !tbaa !8
)doubleB

	full_text


double %88
+double*B

	full_text

double* %89
@callB8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
2addB+
)
	full_text

%90 = add nsw i32 %4, -1
6icmpB.
,
	full_text

%91 = icmp sgt i32 %90, %26
#i32B

	full_text
	
i32 %90
#i32B

	full_text
	
i32 %26
9brB3
1
	full_text$
"
 br i1 %91, label %92, label %168
!i1B

	full_text


i1 %91
5sext8B+
)
	full_text

%93 = sext i32 %8 to i64
^getelementptr8BK
I
	full_text<
:
8%94 = getelementptr inbounds double, double* %0, i64 %93
%i648B

	full_text
	
i64 %93
4add8B+
)
	full_text

%95 = add nsw i32 %70, 1
%i328B

	full_text
	
i32 %70
6sext8B,
*
	full_text

%96 = sext i32 %95 to i64
%i328B

	full_text
	
i32 %95
_getelementptr8BL
J
	full_text=
;
9%97 = getelementptr inbounds double, double* %14, i64 %96
-double*8B

	full_text

double* %14
%i648B

	full_text
	
i64 %96
Nload8BD
B
	full_text5
3
1%98 = load double, double* %97, align 8, !tbaa !8
-double*8B

	full_text

double* %97
4add8B+
)
	full_text

%99 = add nsw i32 %74, 1
%i328B

	full_text
	
i32 %74
7sext8B-
+
	full_text

%100 = sext i32 %99 to i64
%i328B

	full_text
	
i32 %99
agetelementptr8BN
L
	full_text?
=
;%101 = getelementptr inbounds double, double* %14, i64 %100
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %100
Pload8BF
D
	full_text7
5
3%102 = load double, double* %101, align 8, !tbaa !8
.double*8B

	full_text

double* %101
9fadd8B/
-
	full_text 

%103 = fadd double %98, %102
+double8B

	full_text


double %98
,double8B

	full_text

double %102
5add8B,
*
	full_text

%104 = add nsw i32 %79, 1
%i328B

	full_text
	
i32 %79
8sext8B.
,
	full_text

%105 = sext i32 %104 to i64
&i328B

	full_text


i32 %104
agetelementptr8BN
L
	full_text?
=
;%106 = getelementptr inbounds double, double* %14, i64 %105
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %105
Pload8BF
D
	full_text7
5
3%107 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%108 = fadd double %103, %107
,double8B

	full_text

double %103
,double8B

	full_text

double %107
5add8B,
*
	full_text

%109 = add nsw i32 %84, 1
%i328B

	full_text
	
i32 %84
8sext8B.
,
	full_text

%110 = sext i32 %109 to i64
&i328B

	full_text


i32 %109
agetelementptr8BN
L
	full_text?
=
;%111 = getelementptr inbounds double, double* %14, i64 %110
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %110
Pload8BF
D
	full_text7
5
3%112 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%113 = fadd double %108, %112
,double8B

	full_text

double %108
,double8B

	full_text

double %112
5add8B,
*
	full_text

%114 = add nsw i32 %38, 1
%i328B

	full_text
	
i32 %38
8sext8B.
,
	full_text

%115 = sext i32 %114 to i64
&i328B

	full_text


i32 %114
agetelementptr8BN
L
	full_text?
=
;%116 = getelementptr inbounds double, double* %14, i64 %115
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%117 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
5add8B,
*
	full_text

%118 = add nsw i32 %45, 1
%i328B

	full_text
	
i32 %45
8sext8B.
,
	full_text

%119 = sext i32 %118 to i64
&i328B

	full_text


i32 %118
agetelementptr8BN
L
	full_text?
=
;%120 = getelementptr inbounds double, double* %14, i64 %119
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %119
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
:fadd8B0
.
	full_text!

%122 = fadd double %117, %121
,double8B

	full_text

double %117
,double8B

	full_text

double %121
5add8B,
*
	full_text

%123 = add nsw i32 %54, 1
%i328B

	full_text
	
i32 %54
8sext8B.
,
	full_text

%124 = sext i32 %123 to i64
&i328B

	full_text


i32 %123
agetelementptr8BN
L
	full_text?
=
;%125 = getelementptr inbounds double, double* %14, i64 %124
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %124
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

%127 = fadd double %122, %126
,double8B

	full_text

double %122
,double8B

	full_text

double %126
5add8B,
*
	full_text

%128 = add nsw i32 %62, 1
%i328B

	full_text
	
i32 %62
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
;%130 = getelementptr inbounds double, double* %14, i64 %129
-double*8B

	full_text

double* %14
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
:fadd8B0
.
	full_text!

%132 = fadd double %127, %131
,double8B

	full_text

double %127
,double8B

	full_text

double %131
3add8B*
(
	full_text

%133 = add i32 %61, %35
%i328B

	full_text
	
i32 %61
%i328B

	full_text
	
i32 %35
6add8B-
+
	full_text

%134 = add nsw i32 %133, 1
&i328B

	full_text


i32 %133
8sext8B.
,
	full_text

%135 = sext i32 %134 to i64
&i328B

	full_text


i32 %134
agetelementptr8BN
L
	full_text?
=
;%136 = getelementptr inbounds double, double* %14, i64 %135
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %135
Pload8BF
D
	full_text7
5
3%137 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
8sext8B.
,
	full_text

%138 = sext i32 %133 to i64
&i328B

	full_text


i32 %133
agetelementptr8BN
L
	full_text?
=
;%139 = getelementptr inbounds double, double* %14, i64 %138
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %138
Pload8BF
D
	full_text7
5
3%140 = load double, double* %139, align 8, !tbaa !8
.double*8B

	full_text

double* %139
6add8B-
+
	full_text

%141 = add nsw i32 %133, 2
&i328B

	full_text


i32 %133
8sext8B.
,
	full_text

%142 = sext i32 %141 to i64
&i328B

	full_text


i32 %141
agetelementptr8BN
L
	full_text?
=
;%143 = getelementptr inbounds double, double* %14, i64 %142
-double*8B

	full_text

double* %14
&i648B

	full_text


i64 %142
Pload8BF
D
	full_text7
5
3%144 = load double, double* %143, align 8, !tbaa !8
.double*8B

	full_text

double* %143
:fadd8B0
.
	full_text!

%145 = fadd double %140, %144
,double8B

	full_text

double %140
,double8B

	full_text

double %144
:fadd8B0
.
	full_text!

%146 = fadd double %132, %145
,double8B

	full_text

double %132
,double8B

	full_text

double %145
Bfmul8B8
6
	full_text)
'
%%147 = fmul double %146, 2.500000e-01
,double8B

	full_text

double %146
ucall8Bk
i
	full_text\
Z
X%148 = tail call double @llvm.fmuladd.f64(double %137, double 5.000000e-01, double %147)
,double8B

	full_text

double %137
,double8B

	full_text

double %147
Oload8BE
C
	full_text6
4
2%149 = load double, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
5add8B,
*
	full_text

%150 = add nsw i32 %32, 2
%i328B

	full_text
	
i32 %32
8sext8B.
,
	full_text

%151 = sext i32 %150 to i64
&i328B

	full_text


i32 %150
‡getelementptr8Bt
r
	full_texte
c
a%152 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_rprj3.x1, i64 0, i64 %151
&i648B

	full_text


i64 %151
Pload8BF
D
	full_text7
5
3%153 = load double, double* %152, align 8, !tbaa !8
.double*8B

	full_text

double* %152
:fadd8B0
.
	full_text!

%154 = fadd double %149, %153
,double8B

	full_text

double %149
,double8B

	full_text

double %153
:fadd8B0
.
	full_text!

%155 = fadd double %113, %154
,double8B

	full_text

double %113
,double8B

	full_text

double %154
ucall8Bk
i
	full_text\
Z
X%156 = tail call double @llvm.fmuladd.f64(double %155, double 1.250000e-01, double %148)
,double8B

	full_text

double %155
,double8B

	full_text

double %148
Oload8BE
C
	full_text6
4
2%157 = load double, double* %89, align 8, !tbaa !8
-double*8B

	full_text

double* %89
‡getelementptr8Bt
r
	full_texte
c
a%158 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_rprj3.y1, i64 0, i64 %151
&i648B

	full_text


i64 %151
Pload8BF
D
	full_text7
5
3%159 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
:fadd8B0
.
	full_text!

%160 = fadd double %157, %159
,double8B

	full_text

double %157
,double8B

	full_text

double %159
ucall8Bk
i
	full_text\
Z
X%161 = tail call double @llvm.fmuladd.f64(double %160, double 6.250000e-02, double %156)
,double8B

	full_text

double %160
,double8B

	full_text

double %156
6mul8B-
+
	full_text

%162 = mul nsw i32 %20, %5
%i328B

	full_text
	
i32 %20
4add8B+
)
	full_text

%163 = add i32 %162, %23
&i328B

	full_text


i32 %162
%i328B

	full_text
	
i32 %23
3mul8B*
(
	full_text

%164 = mul i32 %163, %4
&i328B

	full_text


i32 %163
8add8B/
-
	full_text 

%165 = add nsw i32 %164, %26
&i328B

	full_text


i32 %164
%i328B

	full_text
	
i32 %26
8sext8B.
,
	full_text

%166 = sext i32 %165 to i64
&i328B

	full_text


i32 %165
agetelementptr8BN
L
	full_text?
=
;%167 = getelementptr inbounds double, double* %94, i64 %166
-double*8B

	full_text

double* %94
&i648B

	full_text


i64 %166
Pstore8BE
C
	full_text6
4
2store double %161, double* %167, align 8, !tbaa !8
,double8B

	full_text

double %161
.double*8B

	full_text

double* %167
(br8B 

	full_text

br label %168
$ret8B

	full_text


ret void
%i328B

	full_text
	
i32 %11
$i328B

	full_text


i32 %1
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %5
%i328B

	full_text
	
i32 %10
$i328B

	full_text


i32 %4
$i328B

	full_text


i32 %2
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %8
$i328B
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
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
$i328B

	full_text


i32 -2
$i328B

	full_text


i32 -1
4double8B&
$
	full_text

double 1.250000e-01
4double8B&
$
	full_text

double 6.250000e-02
z[1024 x double]*8Bb
`
	full_textS
Q
O@kernel_rprj3.x1 = internal unnamed_addr global [1024 x double] undef, align 16
#i328B

	full_text	

i32 2
z[1024 x double]*8Bb
`
	full_textS
Q
O@kernel_rprj3.y1 = internal unnamed_addr global [1024 x double] undef, align 16
4double8B&
$
	full_text

double 2.500000e-01
4double8B&
$
	full_text

double 5.000000e-01
#i648B

	full_text	

i64 1       	 
                         !" !! #$ ## %& %% '( '' )) *+ *, ** -. -- /0 /1 // 23 24 22 56 55 78 79 77 :; :: <= << >? >> @A @B @@ CD CE CC FG FF HI HJ HH KL KK MN MO MM PQ PR PP ST SS UV UU WX WY WW Z[ Z\ ZZ ]^ ]] _` _a __ bc bb de df dd gh gg ij ik ii lm ln ll op oq oo rs rr tu tv tt wx ww yz y{ yy |} || ~ ~~ € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž ŽŽ ‘ 
’  “” ““ •– •
— •• ˜™ ˜˜ š› š
œ šš ž 
Ÿ   ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­­ ¯° ¯
± ¯¯ ²³ ²² ´µ ´
¶ ´´ ·
¸ ·· ¹º ¹
» ¹¹ ¼¼ ½½ ¾¿ ¾
À ¾¾ ÁÂ ÁÃ Ä
Å ÄÄ ÆÇ ÆÆ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ ÛÛ ÝÞ ÝÝ ßà ß
á ßß âã ââ äå ä
æ ää çè çç éê éé ëì ë
í ëë îï îî ðñ ð
ò ðð óô óó õö õõ ÷ø ÷
ù ÷÷ úû úú üý üü þÿ þþ € €
‚ €€ ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆˆ Š‹ ŠŠ Œ Œ
Ž ŒŒ   ‘’ ‘
“ ‘‘ ”• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›› ž 
Ÿ   ¡  
¢    £¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª« ªª ¬­ ¬¬ ®¯ ®
° ®® ±² ±± ³´ ³³ µ¶ µµ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ ÂÂ ÄÅ Ä
Æ ÄÄ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ ËË Í
Î ÍÍ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ ÚÚ Ü
Ý ÜÜ Þß ÞÞ àá à
â àà ãä ã
å ãã æç ææ èé è
ê èè ëì ëë íî í
ï íí ðñ ðð òó ò
ô òò õö õ
÷ õõ ø	ú 	û )	û -	û >	û Uü ü Äý 
ý æ	þ !ÿ ½
ÿ ë€ ) ‚ Ã	ƒ %   	 
             " $# & () +' ,! .% 0- 1/ 3* 42 6 85 97 ;! =< ?> A% B@ D* EC G IF JH L: NK O) Q R! TS V% XP YW [U \Z ^ `] a_ cM eb f h) jg kU m% nl pi qo s ur vt xd zw {% }| y ~ ‚P „- …ƒ ‡% ˆ† Š Œ‰ ‹ / ‘i ’ ” –“ —• ™Ž ›˜ œW ž> Ÿ ¡ £  ¤¢ ¦š ¨¥ ©@ «i ¬ª ® °­ ±¯ ³§ µ² ¶| ¸´ º· »½ ¿ À¾ ÂÃ Å† ÇÆ É ËÈ ÌÊ Î ÐÏ Ò ÔÑ ÕÓ ×Í ÙÖ Ú ÜÛ Þ àÝ áß ãØ åâ æª èç ê ìé íë ïä ñî ò2 ôó ö øõ ù÷ ûC ýü ÿ þ ‚€ „ú †ƒ ‡Z ‰ˆ ‹ Š ŽŒ … ’ “o •” — ™– š˜ œ‘ ž› Ÿl ¡* ¢  ¤£ ¦ ¨¥ ©§ «  ­ ¯¬ °® ²  ´³ ¶ ¸µ ¹· »± ½º ¾ À¼ Á¿ Ãª ÅÂ Æ~ È% ÊÉ ÌË ÎÍ ÐÇ ÒÏ Óð ÕÑ ÖÔ ØÄ Ù· ÛË ÝÜ ßÚ áÞ âà ä× å çæ é êè ìë î ïí ñÄ óð ôã öò ÷Á ÃÁ ùø ù ù …… †† ‡‡ „„ „„ × ‡‡ ×ã ‡‡ ãÄ ‡‡ Ä …… ¼ †† ¼ˆ ˆ 	‰ 	‰ 	‰ #	‰ '	‰ S‰ ¼
‰ Æ
‰ Ï
‰ Û
‰ ç
‰ ó
‰ ü
‰ ˆ
‰ ”
‰ £	Š ~
Š ·
Š Í
Š Ü	‹ 
Œ ½
 ×
Ž ã ~ Í	 <	 g
 ³
 É‘ ·‘ Ü
’ Â
“ Ä	” 	” 	” "
kernel_rprj3"
_Z12get_group_idj"
_Z12get_local_idj"
_Z7barrierj"
llvm.fmuladd.f64*
npb-MG-kernel_rprj3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

transfer_bytes	
°†Íã

wgsize
%
 
transfer_bytes_log1p
˜ÞŸA

wgsize_log1p
˜ÞŸA

devmap_label
