

[external]
KcallBC
A
	full_text4
2
0%8 = tail call i64 @_Z13get_global_idj(i32 1) #4
,addB%
#
	full_text

%9 = add i64 %8, 1
"i64B

	full_text


i64 %8
5truncB,
*
	full_text

%10 = trunc i64 %9 to i32
"i64B

	full_text


i64 %9
KcallBC
A
	full_text4
2
0%11 = tail call i64 @_Z12get_group_idj(i32 0) #4
.addB'
%
	full_text

%12 = add i64 %11, 1
#i64B

	full_text
	
i64 %11
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
KcallBC
A
	full_text4
2
0%14 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
5icmpB-
+
	full_text

%16 = icmp slt i32 %15, %3
#i32B

	full_text
	
i32 %15
8brB2
0
	full_text#
!
br i1 %16, label %17, label %89
!i1B

	full_text


i1 %16
0mul8B'
%
	full_text

%18 = mul i32 %4, %3
2mul8B)
'
	full_text

%19 = mul i32 %18, %10
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %10
5add8B,
*
	full_text

%20 = add nsw i32 %13, -1
%i328B

	full_text
	
i32 %13
5mul8B,
*
	full_text

%21 = mul nsw i32 %20, %4
%i328B

	full_text
	
i32 %20
1add8B(
&
	full_text

%22 = add i32 %19, %6
%i328B

	full_text
	
i32 %19
2add8B)
'
	full_text

%23 = add i32 %22, %21
%i328B

	full_text
	
i32 %22
%i328B

	full_text
	
i32 %21
4add8B+
)
	full_text

%24 = add nsw i32 %13, 1
%i328B

	full_text
	
i32 %13
5mul8B,
*
	full_text

%25 = mul nsw i32 %24, %3
%i328B

	full_text
	
i32 %24
2add8B)
'
	full_text

%26 = add i32 %22, %25
%i328B

	full_text
	
i32 %22
%i328B

	full_text
	
i32 %25
5add8B,
*
	full_text

%27 = add nsw i32 %10, -1
%i328B

	full_text
	
i32 %10
2mul8B)
'
	full_text

%28 = mul i32 %18, %27
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %27
5mul8B,
*
	full_text

%29 = mul nsw i32 %13, %3
%i328B

	full_text
	
i32 %13
1add8B(
&
	full_text

%30 = add i32 %29, %6
%i328B

	full_text
	
i32 %29
2add8B)
'
	full_text

%31 = add i32 %30, %28
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %28
4add8B+
)
	full_text

%32 = add nsw i32 %10, 1
%i328B

	full_text
	
i32 %10
2mul8B)
'
	full_text

%33 = mul i32 %18, %32
%i328B

	full_text
	
i32 %18
%i328B

	full_text
	
i32 %32
2add8B)
'
	full_text

%34 = add i32 %30, %33
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %33
5mul8B,
*
	full_text

%35 = mul nsw i32 %20, %3
%i328B

	full_text
	
i32 %20
1add8B(
&
	full_text

%36 = add i32 %35, %6
%i328B

	full_text
	
i32 %35
2add8B)
'
	full_text

%37 = add i32 %36, %28
%i328B

	full_text
	
i32 %36
%i328B

	full_text
	
i32 %28
1add8B(
&
	full_text

%38 = add i32 %25, %6
%i328B

	full_text
	
i32 %25
2add8B)
'
	full_text

%39 = add i32 %38, %28
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %28
2add8B)
'
	full_text

%40 = add i32 %36, %33
%i328B

	full_text
	
i32 %36
%i328B

	full_text
	
i32 %33
2add8B)
'
	full_text

%41 = add i32 %38, %33
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %33
Ocall8BE
C
	full_text6
4
2%42 = tail call i64 @_Z14get_local_sizej(i32 0) #4
'br8B

	full_text

br label %43
Dphi8B;
9
	full_text,
*
(%44 = phi i32 [ %15, %17 ], [ %87, %43 ]
%i328B

	full_text
	
i32 %15
%i328B

	full_text
	
i32 %87
2add8B)
'
	full_text

%45 = add i32 %23, %44
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%46 = sext i32 %45 to i64
%i328B

	full_text
	
i32 %45
^getelementptr8BK
I
	full_text<
:
8%47 = getelementptr inbounds double, double* %0, i64 %46
%i648B

	full_text
	
i64 %46
Nload8BD
B
	full_text5
3
1%48 = load double, double* %47, align 8, !tbaa !8
-double*8B

	full_text

double* %47
2add8B)
'
	full_text

%49 = add i32 %26, %44
%i328B

	full_text
	
i32 %26
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%50 = sext i32 %49 to i64
%i328B

	full_text
	
i32 %49
^getelementptr8BK
I
	full_text<
:
8%51 = getelementptr inbounds double, double* %0, i64 %50
%i648B

	full_text
	
i64 %50
Nload8BD
B
	full_text5
3
1%52 = load double, double* %51, align 8, !tbaa !8
-double*8B

	full_text

double* %51
7fadd8B-
+
	full_text

%53 = fadd double %48, %52
+double8B

	full_text


double %48
+double8B

	full_text


double %52
2add8B)
'
	full_text

%54 = add i32 %31, %44
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%55 = sext i32 %54 to i64
%i328B

	full_text
	
i32 %54
^getelementptr8BK
I
	full_text<
:
8%56 = getelementptr inbounds double, double* %0, i64 %55
%i648B

	full_text
	
i64 %55
Nload8BD
B
	full_text5
3
1%57 = load double, double* %56, align 8, !tbaa !8
-double*8B

	full_text

double* %56
7fadd8B-
+
	full_text

%58 = fadd double %53, %57
+double8B

	full_text


double %53
+double8B

	full_text


double %57
2add8B)
'
	full_text

%59 = add i32 %34, %44
%i328B

	full_text
	
i32 %34
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%60 = sext i32 %59 to i64
%i328B

	full_text
	
i32 %59
^getelementptr8BK
I
	full_text<
:
8%61 = getelementptr inbounds double, double* %0, i64 %60
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
7fadd8B-
+
	full_text

%63 = fadd double %58, %62
+double8B

	full_text


double %58
+double8B

	full_text


double %62
6sext8B,
*
	full_text

%64 = sext i32 %44 to i64
%i328B

	full_text
	
i32 %44
Ögetelementptr8Br
p
	full_textc
a
_%65 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_psinv.r1, i64 0, i64 %64
%i648B

	full_text
	
i64 %64
Nstore8BC
A
	full_text4
2
0store double %63, double* %65, align 8, !tbaa !8
+double8B

	full_text


double %63
-double*8B

	full_text

double* %65
2add8B)
'
	full_text

%66 = add i32 %37, %44
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%67 = sext i32 %66 to i64
%i328B

	full_text
	
i32 %66
^getelementptr8BK
I
	full_text<
:
8%68 = getelementptr inbounds double, double* %0, i64 %67
%i648B

	full_text
	
i64 %67
Nload8BD
B
	full_text5
3
1%69 = load double, double* %68, align 8, !tbaa !8
-double*8B

	full_text

double* %68
2add8B)
'
	full_text

%70 = add i32 %39, %44
%i328B

	full_text
	
i32 %39
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%71 = sext i32 %70 to i64
%i328B

	full_text
	
i32 %70
^getelementptr8BK
I
	full_text<
:
8%72 = getelementptr inbounds double, double* %0, i64 %71
%i648B

	full_text
	
i64 %71
Nload8BD
B
	full_text5
3
1%73 = load double, double* %72, align 8, !tbaa !8
-double*8B

	full_text

double* %72
7fadd8B-
+
	full_text

%74 = fadd double %69, %73
+double8B

	full_text


double %69
+double8B

	full_text


double %73
2add8B)
'
	full_text

%75 = add i32 %40, %44
%i328B

	full_text
	
i32 %40
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%76 = sext i32 %75 to i64
%i328B

	full_text
	
i32 %75
^getelementptr8BK
I
	full_text<
:
8%77 = getelementptr inbounds double, double* %0, i64 %76
%i648B

	full_text
	
i64 %76
Nload8BD
B
	full_text5
3
1%78 = load double, double* %77, align 8, !tbaa !8
-double*8B

	full_text

double* %77
7fadd8B-
+
	full_text

%79 = fadd double %74, %78
+double8B

	full_text


double %74
+double8B

	full_text


double %78
2add8B)
'
	full_text

%80 = add i32 %41, %44
%i328B

	full_text
	
i32 %41
%i328B

	full_text
	
i32 %44
6sext8B,
*
	full_text

%81 = sext i32 %80 to i64
%i328B

	full_text
	
i32 %80
^getelementptr8BK
I
	full_text<
:
8%82 = getelementptr inbounds double, double* %0, i64 %81
%i648B

	full_text
	
i64 %81
Nload8BD
B
	full_text5
3
1%83 = load double, double* %82, align 8, !tbaa !8
-double*8B

	full_text

double* %82
7fadd8B-
+
	full_text

%84 = fadd double %79, %83
+double8B

	full_text


double %79
+double8B

	full_text


double %83
Ögetelementptr8Br
p
	full_textc
a
_%85 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_psinv.r2, i64 0, i64 %64
%i648B

	full_text
	
i64 %64
Nstore8BC
A
	full_text4
2
0store double %84, double* %85, align 8, !tbaa !8
+double8B

	full_text


double %84
-double*8B

	full_text

double* %85
2add8B)
'
	full_text

%86 = add i64 %42, %64
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %64
8trunc8B-
+
	full_text

%87 = trunc i64 %86 to i32
%i648B

	full_text
	
i64 %86
7icmp8B-
+
	full_text

%88 = icmp slt i32 %87, %3
%i328B

	full_text
	
i32 %87
:br8B2
0
	full_text#
!
br i1 %88, label %43, label %89
#i18B

	full_text


i1 %88
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #5
4add8B+
)
	full_text

%90 = add nsw i32 %15, 1
%i328B

	full_text
	
i32 %15
4add8B+
)
	full_text

%91 = add nsw i32 %3, -1
8icmp8B.
,
	full_text

%92 = icmp slt i32 %90, %91
%i328B

	full_text
	
i32 %90
%i328B

	full_text
	
i32 %91
;br8B3
1
	full_text$
"
 br i1 %92, label %93, label %145
#i18B

	full_text


i1 %92
5mul8B,
*
	full_text

%94 = mul nsw i32 %10, %4
%i328B

	full_text
	
i32 %10
2add8B)
'
	full_text

%95 = add i32 %94, %13
%i328B

	full_text
	
i32 %94
%i328B

	full_text
	
i32 %13
1mul8B(
&
	full_text

%96 = mul i32 %95, %3
%i328B

	full_text
	
i32 %95
\getelementptr8BI
G
	full_text:
8
6%97 = getelementptr inbounds double, double* %2, i64 1
0add8B'
%
	full_text

%98 = add i32 %6, -1
/add8B&
$
	full_text

%99 = add i32 %6, 1
]getelementptr8BJ
H
	full_text;
9
7%100 = getelementptr inbounds double, double* %2, i64 2
Pcall8BF
D
	full_text7
5
3%101 = tail call i64 @_Z14get_local_sizej(i32 0) #4
(br8B 

	full_text

br label %102
Gphi8B>
<
	full_text/
-
+%103 = phi i32 [ %90, %93 ], [ %143, %102 ]
%i328B

	full_text
	
i32 %90
&i328B

	full_text


i32 %143
8add8B/
-
	full_text 

%104 = add nsw i32 %103, %96
&i328B

	full_text


i32 %103
%i328B

	full_text
	
i32 %96
7add8B.
,
	full_text

%105 = add nsw i32 %104, %6
&i328B

	full_text


i32 %104
8sext8B.
,
	full_text

%106 = sext i32 %105 to i64
&i328B

	full_text


i32 %105
`getelementptr8BM
K
	full_text>
<
:%107 = getelementptr inbounds double, double* %1, i64 %106
&i648B

	full_text


i64 %106
Pload8BF
D
	full_text7
5
3%108 = load double, double* %107, align 8, !tbaa !8
.double*8B

	full_text

double* %107
Nload8BD
B
	full_text5
3
1%109 = load double, double* %2, align 8, !tbaa !8
`getelementptr8BM
K
	full_text>
<
:%110 = getelementptr inbounds double, double* %0, i64 %106
&i648B

	full_text


i64 %106
Pload8BF
D
	full_text7
5
3%111 = load double, double* %110, align 8, !tbaa !8
.double*8B

	full_text

double* %110
mcall8Bc
a
	full_textT
R
P%112 = tail call double @llvm.fmuladd.f64(double %109, double %111, double %108)
,double8B

	full_text

double %109
,double8B

	full_text

double %111
,double8B

	full_text

double %108
Oload8BE
C
	full_text6
4
2%113 = load double, double* %97, align 8, !tbaa !8
-double*8B

	full_text

double* %97
4add8B+
)
	full_text

%114 = add i32 %98, %104
%i328B

	full_text
	
i32 %98
&i328B

	full_text


i32 %104
8sext8B.
,
	full_text

%115 = sext i32 %114 to i64
&i328B

	full_text


i32 %114
`getelementptr8BM
K
	full_text>
<
:%116 = getelementptr inbounds double, double* %0, i64 %115
&i648B

	full_text


i64 %115
Pload8BF
D
	full_text7
5
3%117 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
4add8B+
)
	full_text

%118 = add i32 %99, %104
%i328B

	full_text
	
i32 %99
&i328B

	full_text


i32 %104
8sext8B.
,
	full_text

%119 = sext i32 %118 to i64
&i328B

	full_text


i32 %118
`getelementptr8BM
K
	full_text>
<
:%120 = getelementptr inbounds double, double* %0, i64 %119
&i648B

	full_text


i64 %119
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
:fadd8B0
.
	full_text!

%122 = fadd double %117, %121
,double8B

	full_text

double %117
,double8B

	full_text

double %121
8sext8B.
,
	full_text

%123 = sext i32 %103 to i64
&i328B

	full_text


i32 %103
ágetelementptr8Bt
r
	full_texte
c
a%124 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_psinv.r1, i64 0, i64 %123
&i648B

	full_text


i64 %123
Pload8BF
D
	full_text7
5
3%125 = load double, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
:fadd8B0
.
	full_text!

%126 = fadd double %122, %125
,double8B

	full_text

double %122
,double8B

	full_text

double %125
mcall8Bc
a
	full_textT
R
P%127 = tail call double @llvm.fmuladd.f64(double %113, double %126, double %112)
,double8B

	full_text

double %113
,double8B

	full_text

double %126
,double8B

	full_text

double %112
Pload8BF
D
	full_text7
5
3%128 = load double, double* %100, align 8, !tbaa !8
.double*8B

	full_text

double* %100
ágetelementptr8Bt
r
	full_texte
c
a%129 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_psinv.r2, i64 0, i64 %123
&i648B

	full_text


i64 %123
Pload8BF
D
	full_text7
5
3%130 = load double, double* %129, align 8, !tbaa !8
.double*8B

	full_text

double* %129
7add8B.
,
	full_text

%131 = add nsw i32 %103, -1
&i328B

	full_text


i32 %103
8sext8B.
,
	full_text

%132 = sext i32 %131 to i64
&i328B

	full_text


i32 %131
ágetelementptr8Bt
r
	full_texte
c
a%133 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_psinv.r1, i64 0, i64 %132
&i648B

	full_text


i64 %132
Pload8BF
D
	full_text7
5
3%134 = load double, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
:fadd8B0
.
	full_text!

%135 = fadd double %130, %134
,double8B

	full_text

double %130
,double8B

	full_text

double %134
6add8B-
+
	full_text

%136 = add nsw i32 %103, 1
&i328B

	full_text


i32 %103
8sext8B.
,
	full_text

%137 = sext i32 %136 to i64
&i328B

	full_text


i32 %136
ágetelementptr8Bt
r
	full_texte
c
a%138 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_psinv.r1, i64 0, i64 %137
&i648B

	full_text


i64 %137
Pload8BF
D
	full_text7
5
3%139 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
:fadd8B0
.
	full_text!

%140 = fadd double %135, %139
,double8B

	full_text

double %135
,double8B

	full_text

double %139
mcall8Bc
a
	full_textT
R
P%141 = tail call double @llvm.fmuladd.f64(double %128, double %140, double %127)
,double8B

	full_text

double %128
,double8B

	full_text

double %140
,double8B

	full_text

double %127
Pstore8BE
C
	full_text6
4
2store double %141, double* %107, align 8, !tbaa !8
,double8B

	full_text

double %141
.double*8B

	full_text

double* %107
5add8B,
*
	full_text

%142 = add i64 %101, %123
&i648B

	full_text


i64 %101
&i648B

	full_text


i64 %123
:trunc8B/
-
	full_text 

%143 = trunc i64 %142 to i32
&i648B

	full_text


i64 %142
:icmp8B0
.
	full_text!

%144 = icmp sgt i32 %91, %143
%i328B

	full_text
	
i32 %91
&i328B

	full_text


i32 %143
=br8B5
3
	full_text&
$
"br i1 %144, label %102, label %145
$i18B

	full_text
	
i1 %144
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %4
,double*8B

	full_text


double* %2
$i328B

	full_text


i32 %3
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %6
,double*8B
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
-; undefined function B

	full_text

 
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 1
z[1024 x double]*8Bb
`
	full_textS
Q
O@kernel_psinv.r2 = internal unnamed_addr global [1024 x double] undef, align 16
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 2
z[1024 x double]*8Bb
`
	full_textS
Q
O@kernel_psinv.r1 = internal unnamed_addr global [1024 x double] undef, align 16        	
 		                       !" !! #$ #% ## &' && () (* (( +, ++ -. -- /0 /1 // 23 22 45 46 44 78 79 77 :; :: <= << >? >@ >> AB AA CD CE CC FG FH FF IJ IK II LL MO NP NN QR QS QQ TU TT VW VV XY XX Z[ Z\ ZZ ]^ ]] _` __ ab aa cd ce cc fg fh ff ij ii kl kk mn mm op oq oo rs rt rr uv uu wx ww yz yy {| {} {{ ~ ~~ Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ àâ àà ä
ã ää åç åå éè é
ê éé ëí ëë ì
î ìì ïñ ïï óò ó
ô óó öõ ö
ú öö ùû ùù ü
† üü °¢ °° £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©© ´
¨ ´´ ≠Æ ≠≠ Ø∞ Ø
± ØØ ≤
≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫∫ ºΩ ºº æø æ¿ ¡¬ ¡¡ √√ ƒ≈ ƒ
∆ ƒƒ «» «  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –– —— ““ ”” ‘‘ ’◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡
· ‡‡ ‚„ ‚‚ ‰‰ Â
Ê ÂÂ ÁË ÁÁ ÈÍ È
Î È
Ï ÈÈ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ ÚÚ Ù
ı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝
˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Ü
á ÜÜ àâ àà äã ä
å ää çé ç
è ç
ê çç ëí ëë ì
î ìì ïñ ïï óò óó ôö ôô õ
ú õõ ùû ùù ü† ü
° üü ¢£ ¢¢ §• §§ ¶
ß ¶¶ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π∫ π
ª ππ ºΩ ºø 	ø 
ø …¿ –¿ ”¿ ‰	¡ 	¡ 	¡ !	¡ +	¡ :
¡ º¡ √
¡ Œ¬ ‡	√ 	√ -	√ <	√ A√ —√ “
√ ‹ƒ Vƒ _ƒ kƒ wƒ äƒ ìƒ üƒ ´ƒ Âƒ Ùƒ ˝    
     	     	   " $! % ' )& *	 ,+ .- 0( 1 3 52 6- 84 9 ;: =< ?( @! BA D( E< G4 HA J4 K O∫ P RN SQ UT WV Y# [N \Z ^] `_ bX da e/ gN hf ji lk nc pm q7 sN tr vu xw zo |y }N ~ Å{ ÉÄ Ñ> ÜN áÖ âà ãä çC èN êé íë îì ñå òï ôF õN úö ûù †ü ¢ó §° •I ßN ®¶ ™© ¨´ Æ£ ∞≠ ±~ ≥Ø µ≤ ∂L ∏~ π∑ ª∫ Ωº ø ¬¡ ≈√ ∆ƒ »  … Ã	 ÕÀ œ¡ ◊∑ ÿ÷ ⁄Œ €Ÿ ›‹ ﬂﬁ ·‡ „ﬁ ÊÂ Ë‰ ÍÁ Î‚ Ï– Ó— Ÿ ÒÔ ÛÚ ıÙ ˜“ ˘Ÿ ˙¯ ¸˚ ˛˝ Äˆ Çˇ É÷ ÖÑ áÜ âÅ ãà åÌ éä èÈ ê” íÑ îì ñ÷ òó öô úõ ûï †ù °÷ £¢ •§ ß¶ ©ü ´® ¨ë Æ™ Øç ∞≠ ≤‡ ≥‘ µÑ ∂¥ ∏√ ∫∑ ªπ Ω  ¿M N« …« ææ Næ ¿’ ÷º ÷º æ ∆∆ æ ≈≈ »» «« ……    «« ç    çÈ    ÈL »» L ≈≈ ¿ …… ¿‘ »» ‘ ∆∆ ≠    ≠	À 	À &
À √
À —
À ó	Ã 	Ã 
Ã –Õ ≤Õ ìŒ Œ Œ LŒ ‘
œ Ä
œ ≤
œ Ü
œ ì
œ õ
œ ¶– 	– 	– 2– ¿
– ¡
– “
– ¢
— ”“ Ä“ Ü“ õ“ ¶"
kernel_psinv"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z7barrierj"
llvm.fmuladd.f64*è
npb-MG-kernel_psinv.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

wgsize
B

transfer_bytes	
∞ÊÃ„

devmap_label

 
transfer_bytes_log1p
îﬁüA

wgsize_log1p
îﬁüA