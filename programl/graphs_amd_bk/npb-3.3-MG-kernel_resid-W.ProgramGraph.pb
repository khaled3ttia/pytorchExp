

[external]
KcallBC
A
	full_text4
2
0%9 = tail call i64 @_Z13get_global_idj(i32 1) #4
-addB&
$
	full_text

%10 = add i64 %9, 1
"i64B

	full_text


i64 %9
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
KcallBC
A
	full_text4
2
0%12 = tail call i64 @_Z12get_group_idj(i32 0) #4
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
KcallBC
A
	full_text4
2
0%15 = tail call i64 @_Z12get_local_idj(i32 0) #4
6truncB-
+
	full_text

%16 = trunc i64 %15 to i32
#i64B

	full_text
	
i64 %15
5icmpB-
+
	full_text

%17 = icmp slt i32 %16, %4
#i32B

	full_text
	
i32 %16
8brB2
0
	full_text#
!
br i1 %17, label %18, label %89
!i1B

	full_text


i1 %17
0mul8B'
%
	full_text

%19 = mul i32 %5, %4
2mul8B)
'
	full_text

%20 = mul i32 %19, %11
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %11
5add8B,
*
	full_text

%21 = add nsw i32 %14, -1
%i328B

	full_text
	
i32 %14
5mul8B,
*
	full_text

%22 = mul nsw i32 %21, %4
%i328B

	full_text
	
i32 %21
1add8B(
&
	full_text

%23 = add i32 %20, %7
%i328B

	full_text
	
i32 %20
2add8B)
'
	full_text

%24 = add i32 %23, %22
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %22
4add8B+
)
	full_text

%25 = add nsw i32 %14, 1
%i328B

	full_text
	
i32 %14
5mul8B,
*
	full_text

%26 = mul nsw i32 %25, %4
%i328B

	full_text
	
i32 %25
2add8B)
'
	full_text

%27 = add i32 %23, %26
%i328B

	full_text
	
i32 %23
%i328B

	full_text
	
i32 %26
5add8B,
*
	full_text

%28 = add nsw i32 %11, -1
%i328B

	full_text
	
i32 %11
2mul8B)
'
	full_text

%29 = mul i32 %19, %28
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %28
5mul8B,
*
	full_text

%30 = mul nsw i32 %14, %4
%i328B

	full_text
	
i32 %14
1add8B(
&
	full_text

%31 = add i32 %30, %7
%i328B

	full_text
	
i32 %30
2add8B)
'
	full_text

%32 = add i32 %31, %29
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %29
4add8B+
)
	full_text

%33 = add nsw i32 %11, 1
%i328B

	full_text
	
i32 %11
2mul8B)
'
	full_text

%34 = mul i32 %19, %33
%i328B

	full_text
	
i32 %19
%i328B

	full_text
	
i32 %33
2add8B)
'
	full_text

%35 = add i32 %31, %34
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %34
1add8B(
&
	full_text

%36 = add i32 %22, %7
%i328B

	full_text
	
i32 %22
2add8B)
'
	full_text

%37 = add i32 %36, %29
%i328B

	full_text
	
i32 %36
%i328B

	full_text
	
i32 %29
1add8B(
&
	full_text

%38 = add i32 %26, %7
%i328B

	full_text
	
i32 %26
2add8B)
'
	full_text

%39 = add i32 %38, %29
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %29
2add8B)
'
	full_text

%40 = add i32 %36, %34
%i328B

	full_text
	
i32 %36
%i328B

	full_text
	
i32 %34
2add8B)
'
	full_text

%41 = add i32 %38, %34
%i328B

	full_text
	
i32 %38
%i328B

	full_text
	
i32 %34
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
(%44 = phi i32 [ %16, %18 ], [ %87, %43 ]
%i328B

	full_text
	
i32 %16
%i328B

	full_text
	
i32 %87
2add8B)
'
	full_text

%45 = add i32 %24, %44
%i328B

	full_text
	
i32 %24
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
8%47 = getelementptr inbounds double, double* %1, i64 %46
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
%49 = add i32 %27, %44
%i328B

	full_text
	
i32 %27
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
8%51 = getelementptr inbounds double, double* %1, i64 %50
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
%54 = add i32 %32, %44
%i328B

	full_text
	
i32 %32
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
8%56 = getelementptr inbounds double, double* %1, i64 %55
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
%59 = add i32 %35, %44
%i328B

	full_text
	
i32 %35
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
8%61 = getelementptr inbounds double, double* %1, i64 %60
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
_%65 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u1, i64 0, i64 %64
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
8%68 = getelementptr inbounds double, double* %1, i64 %67
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
8%72 = getelementptr inbounds double, double* %1, i64 %71
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
8%77 = getelementptr inbounds double, double* %1, i64 %76
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
8%82 = getelementptr inbounds double, double* %1, i64 %81
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
_%85 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u2, i64 0, i64 %64
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

%88 = icmp slt i32 %87, %4
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
%90 = add nsw i32 %16, 1
%i328B

	full_text
	
i32 %16
4add8B+
)
	full_text

%91 = add nsw i32 %4, -1
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
 br i1 %92, label %93, label %140
#i18B

	full_text


i1 %92
5mul8B,
*
	full_text

%94 = mul nsw i32 %11, %5
%i328B

	full_text
	
i32 %11
2add8B)
'
	full_text

%95 = add i32 %94, %14
%i328B

	full_text
	
i32 %94
%i328B

	full_text
	
i32 %14
1mul8B(
&
	full_text

%96 = mul i32 %95, %4
%i328B

	full_text
	
i32 %95
1add8B(
&
	full_text

%97 = add i32 %96, %7
%i328B

	full_text
	
i32 %96
\getelementptr8BI
G
	full_text:
8
6%98 = getelementptr inbounds double, double* %3, i64 2
\getelementptr8BI
G
	full_text:
8
6%99 = getelementptr inbounds double, double* %3, i64 3
Pcall8BF
D
	full_text7
5
3%100 = tail call i64 @_Z14get_local_sizej(i32 0) #4
(br8B 

	full_text

br label %101
Gphi8B>
<
	full_text/
-
+%102 = phi i32 [ %90, %93 ], [ %138, %101 ]
%i328B

	full_text
	
i32 %90
&i328B

	full_text


i32 %138
4add8B+
)
	full_text

%103 = add i32 %97, %102
%i328B

	full_text
	
i32 %97
&i328B

	full_text


i32 %102
8sext8B.
,
	full_text

%104 = sext i32 %103 to i64
&i328B

	full_text


i32 %103
`getelementptr8BM
K
	full_text>
<
:%105 = getelementptr inbounds double, double* %2, i64 %104
&i648B

	full_text


i64 %104
Pload8BF
D
	full_text7
5
3%106 = load double, double* %105, align 8, !tbaa !8
.double*8B

	full_text

double* %105
Nload8BD
B
	full_text5
3
1%107 = load double, double* %3, align 8, !tbaa !8
`getelementptr8BM
K
	full_text>
<
:%108 = getelementptr inbounds double, double* %1, i64 %104
&i648B

	full_text


i64 %104
Pload8BF
D
	full_text7
5
3%109 = load double, double* %108, align 8, !tbaa !8
.double*8B

	full_text

double* %108
Cfsub8B9
7
	full_text*
(
&%110 = fsub double -0.000000e+00, %107
,double8B

	full_text

double %107
mcall8Bc
a
	full_textT
R
P%111 = tail call double @llvm.fmuladd.f64(double %110, double %109, double %106)
,double8B

	full_text

double %110
,double8B

	full_text

double %109
,double8B

	full_text

double %106
Oload8BE
C
	full_text6
4
2%112 = load double, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
8sext8B.
,
	full_text

%113 = sext i32 %102 to i64
&i328B

	full_text


i32 %102
ágetelementptr8Bt
r
	full_texte
c
a%114 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u2, i64 0, i64 %113
&i648B

	full_text


i64 %113
Pload8BF
D
	full_text7
5
3%115 = load double, double* %114, align 8, !tbaa !8
.double*8B

	full_text

double* %114
7add8B.
,
	full_text

%116 = add nsw i32 %102, -1
&i328B

	full_text


i32 %102
8sext8B.
,
	full_text

%117 = sext i32 %116 to i64
&i328B

	full_text


i32 %116
ágetelementptr8Bt
r
	full_texte
c
a%118 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u1, i64 0, i64 %117
&i648B

	full_text


i64 %117
Pload8BF
D
	full_text7
5
3%119 = load double, double* %118, align 8, !tbaa !8
.double*8B

	full_text

double* %118
:fadd8B0
.
	full_text!

%120 = fadd double %115, %119
,double8B

	full_text

double %115
,double8B

	full_text

double %119
6add8B-
+
	full_text

%121 = add nsw i32 %102, 1
&i328B

	full_text


i32 %102
8sext8B.
,
	full_text

%122 = sext i32 %121 to i64
&i328B

	full_text


i32 %121
ágetelementptr8Bt
r
	full_texte
c
a%123 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u1, i64 0, i64 %122
&i648B

	full_text


i64 %122
Pload8BF
D
	full_text7
5
3%124 = load double, double* %123, align 8, !tbaa !8
.double*8B

	full_text

double* %123
:fadd8B0
.
	full_text!

%125 = fadd double %120, %124
,double8B

	full_text

double %120
,double8B

	full_text

double %124
Cfsub8B9
7
	full_text*
(
&%126 = fsub double -0.000000e+00, %112
,double8B

	full_text

double %112
mcall8Bc
a
	full_textT
R
P%127 = tail call double @llvm.fmuladd.f64(double %126, double %125, double %111)
,double8B

	full_text

double %126
,double8B

	full_text

double %125
,double8B

	full_text

double %111
Oload8BE
C
	full_text6
4
2%128 = load double, double* %99, align 8, !tbaa !8
-double*8B

	full_text

double* %99
ágetelementptr8Bt
r
	full_texte
c
a%129 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u2, i64 0, i64 %117
&i648B

	full_text


i64 %117
Pload8BF
D
	full_text7
5
3%130 = load double, double* %129, align 8, !tbaa !8
.double*8B

	full_text

double* %129
ágetelementptr8Bt
r
	full_texte
c
a%131 = getelementptr inbounds [1024 x double], [1024 x double]* @kernel_resid.u2, i64 0, i64 %122
&i648B

	full_text


i64 %122
Pload8BF
D
	full_text7
5
3%132 = load double, double* %131, align 8, !tbaa !8
.double*8B

	full_text

double* %131
:fadd8B0
.
	full_text!

%133 = fadd double %130, %132
,double8B

	full_text

double %130
,double8B

	full_text

double %132
Cfsub8B9
7
	full_text*
(
&%134 = fsub double -0.000000e+00, %128
,double8B

	full_text

double %128
mcall8Bc
a
	full_textT
R
P%135 = tail call double @llvm.fmuladd.f64(double %134, double %133, double %127)
,double8B

	full_text

double %134
,double8B

	full_text

double %133
,double8B

	full_text

double %127
`getelementptr8BM
K
	full_text>
<
:%136 = getelementptr inbounds double, double* %0, i64 %104
&i648B

	full_text


i64 %104
Pstore8BE
C
	full_text6
4
2store double %135, double* %136, align 8, !tbaa !8
,double8B

	full_text

double %135
.double*8B

	full_text

double* %136
5add8B,
*
	full_text

%137 = add i64 %100, %113
&i648B

	full_text


i64 %100
&i648B

	full_text


i64 %113
:trunc8B/
-
	full_text 

%138 = trunc i64 %137 to i32
&i648B

	full_text


i64 %137
:icmp8B0
.
	full_text!

%139 = icmp sgt i32 %91, %138
%i328B

	full_text
	
i32 %91
&i328B

	full_text


i32 %138
=br8B5
3
	full_text&
$
"br i1 %139, label %101, label %140
$i18B

	full_text
	
i1 %139
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


double* %1
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %5
,double*8B

	full_text


double* %3
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
#i648B

	full_text	

i64 1
#i648B

	full_text	

i64 0
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 2
#i328B

	full_text	

i32 1
5double8B'
%
	full_text

double -0.000000e+00
#i328B

	full_text	

i32 0
z[1024 x double]*8Bb
`
	full_textS
Q
O@kernel_resid.u2 = internal unnamed_addr global [1024 x double] undef, align 16
z[1024 x double]*8Bb
`
	full_textS
Q
O@kernel_resid.u1 = internal unnamed_addr global [1024 x double] undef, align 16        	
 		                       !" !! #$ #% ## &' && () (* (( +, ++ -. -- /0 /1 // 23 22 45 46 44 78 79 77 :; :: <= <> << ?@ ?? AB AC AA DE DF DD GH GI GG JJ KM LN LL OP OQ OO RS RR TU TT VW VV XY XZ XX [\ [[ ]^ ]] _` __ ab ac aa de df dd gh gg ij ii kl kk mn mo mm pq pr pp st ss uv uu wx ww yz y{ yy |} || ~ ~~ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá ÜÜ à
â àà äã ää åç å
é åå èê èè ë
í ëë ìî ìì ïñ ï
ó ïï òô ò
ö òò õú õõ ù
û ùù ü† üü °¢ °
£ °° §• §
¶ §§ ß® ßß ©
™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞
± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫∫ ºΩ ºæ ø¿ øø ¡¡ ¬√ ¬
ƒ ¬¬ ≈∆ ≈» «« …  …
À …… ÃÕ ÃÃ Œœ ŒŒ –– —— ““ ”’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹
› ‹‹ ﬁﬂ ﬁﬁ ‡‡ ·
‚ ·· „‰ „„ Â
Ê ÂÂ ÁË Á
È Á
Í ÁÁ ÎÏ ÎÎ ÌÓ ÌÌ Ô
 ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ ÄÅ ÄÄ Ç
É ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ â
ä ââ ãå ã
ç ã
é ãã èê èè ë
í ëë ìî ìì ï
ñ ïï óò óó ôö ô
õ ôô ú
ù úú ûü û
† û
° ûû ¢
£ ¢¢ §• §
¶ §§ ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø	≤ 	≤ 	≤ 	≤ !	≤ +
≤ ∫≤ ¡
≤ Ã≥ T≥ ]≥ i≥ u≥ à≥ ë≥ ù≥ ©≥ ·¥ ‹µ ¢	∂ 	∂ -	∂ :	∂ ?
∂ Œ∑ 
∑ «∏ –∏ —∏ ‡    
     	     	   " $! % ' )& *	 ,+ .- 0( 1 3 52 6- 84 9 ;: =( >! @? B( C: E4 F? H4 I M∏ N PL QO SR UT W# YL ZX \[ ^] `V b_ c/ eL fd hg ji la nk o7 qL rp ts vu xm zw {L }| y Å~ Ç< ÑL ÖÉ áÜ âà ãA çL éå êè íë îä ñì óD ôL öò úõ ûù †ï ¢ü £G •L ¶§ ®ß ™© ¨° Æ´ Ø| ±≠ ≥∞ ¥J ∂| ∑µ π∏ ª∫ Ω ¿ø √¡ ƒ¬ ∆ »«  	 À… ÕÃ œø ’™ ÷Œ ÿ‘ Ÿ◊ €⁄ ›‹ ﬂ⁄ ‚· ‰‡ ÊÂ Ë„ Èﬁ Í– Ï‘ ÓÌ Ô Ú‘ ÙÛ ˆı ¯˜ ˙Ò ¸˘ ˝‘ ˇ˛ ÅÄ ÉÇ Ö˚ áÑ àÎ äâ åÜ çÁ é— êı íë îÄ ñï òì öó õè ùú üô †ã °⁄ £û •¢ ¶“ ®Ì ©ß ´¡ ≠™ Æ¨ ∞  æK L≈ «≈ ±º Lº æ” ‘Ø ‘Ø ± ± ΩΩ ææ ºº ªª ππ ∫∫Á ææ ÁJ ºº Jã ææ ã ∫∫ û ææ ûæ ΩΩ æ ππ  ªª “ ºº “	ø 	ø 	¿ ~
¿ ∞
¿ Ô
¿ ˜
¿ Ç
¿ ë
¿ ï	¡ 	¡ &
¡ ¡
¡ Û
¬ —
√ –ƒ 	ƒ 	ƒ 2ƒ æ
ƒ ø
ƒ ˛≈ Â≈ â≈ ú∆ ∆ ∆ J∆ “« ∞« Ô« ë« ï» ~» ˜» Ç"
kernel_resid"
_Z13get_global_idj"
_Z12get_group_idj"
_Z12get_local_idj"
_Z14get_local_sizej"
_Z7barrierj"
llvm.fmuladd.f64*è
npb-MG-kernel_resid.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Ä

transfer_bytes
–˚¡
 
transfer_bytes_log1p
,áèA

devmap_label
 

wgsize_log1p
,áèA

wgsize
<