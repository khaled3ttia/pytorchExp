

[external]
LcallBD
B
	full_text5
3
1%11 = tail call i64 @_Z13get_global_idj(i32 0) #2
6truncB-
+
	full_text

%12 = trunc i64 %11 to i32
#i64B

	full_text
	
i64 %11
5icmpB-
+
	full_text

%13 = icmp slt i32 %12, %8
#i32B

	full_text
	
i32 %12
9brB3
1
	full_text$
"
 br i1 %13, label %14, label %158
!i1B

	full_text


i1 %13
Cbitcast8B6
4
	full_text'
%
#%15 = bitcast i32* %4 to [9 x i32]*
Ibitcast8B<
:
	full_text-
+
)%16 = bitcast double* %5 to [9 x double]*
5sext8B+
)
	full_text

%17 = sext i32 %9 to i64
Xgetelementptr8BE
C
	full_text6
4
2%18 = getelementptr inbounds i32, i32* %1, i64 %17
%i648B

	full_text
	
i64 %17
1shl8B(
&
	full_text

%19 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%20 = ashr exact i64 %19, 32
%i648B

	full_text
	
i64 %19
Xgetelementptr8BE
C
	full_text6
4
2%21 = getelementptr inbounds i32, i32* %2, i64 %20
%i648B

	full_text
	
i64 %20
Hload8B>
<
	full_text/
-
+%22 = load i32, i32* %21, align 4, !tbaa !8
'i32*8B

	full_text


i32* %21
9add8B0
.
	full_text!

%23 = add i64 %19, 4294967296
%i648B

	full_text
	
i64 %19
9ashr8B/
-
	full_text 

%24 = ashr exact i64 %23, 32
%i648B

	full_text
	
i64 %23
Xgetelementptr8BE
C
	full_text6
4
2%25 = getelementptr inbounds i32, i32* %2, i64 %24
%i648B

	full_text
	
i64 %24
Hload8B>
<
	full_text/
-
+%26 = load i32, i32* %25, align 4, !tbaa !8
'i32*8B

	full_text


i32* %25
8icmp8B.
,
	full_text

%27 = icmp slt i32 %22, %26
%i328B

	full_text
	
i32 %22
%i328B

	full_text
	
i32 %26
:br8B2
0
	full_text#
!
br i1 %27, label %28, label %38
#i18B

	full_text


i1 %27
6sext8B,
*
	full_text

%29 = sext i32 %22 to i64
%i328B

	full_text
	
i32 %22
'br8B

	full_text

br label %30
Dphi8B;
9
	full_text,
*
(%31 = phi i64 [ %29, %28 ], [ %34, %30 ]
%i648B

	full_text
	
i64 %29
%i648B

	full_text
	
i64 %34
^getelementptr8BK
I
	full_text<
:
8%32 = getelementptr inbounds double, double* %0, i64 %31
%i648B

	full_text
	
i64 %31
Xstore8BM
K
	full_text>
<
:store double 0.000000e+00, double* %32, align 8, !tbaa !12
-double*8B

	full_text

double* %32
Xgetelementptr8BE
C
	full_text6
4
2%33 = getelementptr inbounds i32, i32* %1, i64 %31
%i648B

	full_text
	
i64 %31
Gstore8B<
:
	full_text-
+
)store i32 -1, i32* %33, align 4, !tbaa !8
'i32*8B

	full_text


i32* %33
0add8B'
%
	full_text

%34 = add i64 %31, 1
%i648B

	full_text
	
i64 %31
Hload8B>
<
	full_text/
-
+%35 = load i32, i32* %25, align 4, !tbaa !8
'i32*8B

	full_text


i32* %25
6sext8B,
*
	full_text

%36 = sext i32 %35 to i64
%i328B

	full_text
	
i32 %35
8icmp8B.
,
	full_text

%37 = icmp slt i64 %34, %36
%i648B

	full_text
	
i64 %34
%i648B

	full_text
	
i64 %36
:br8B2
0
	full_text#
!
br i1 %37, label %30, label %38
#i18B

	full_text


i1 %37
Ygetelementptr8BF
D
	full_text7
5
3%39 = getelementptr inbounds i32, i32* %18, i64 %20
'i32*8B

	full_text


i32* %18
%i648B

	full_text
	
i64 %20
Fstore8B;
9
	full_text,
*
(store i32 0, i32* %39, align 4, !tbaa !8
'i32*8B

	full_text


i32* %39
<sitofp8B0
.
	full_text!

%40 = sitofp i32 %8 to double
@fdiv8B6
4
	full_text'
%
#%41 = fdiv double 1.000000e+00, %40
+double8B

	full_text


double %40
acall8BW
U
	full_textH
F
D%42 = tail call double @_Z3powdd(double 1.000000e-01, double %41) #2
+double8B

	full_text


double %41
5icmp8B+
)
	full_text

%43 = icmp sgt i32 %8, 0
;br8B3
1
	full_text$
"
 br i1 %43, label %44, label %158
#i18B

	full_text


i1 %43
9and8B0
.
	full_text!

%45 = and i64 %11, 4294967295
%i648B

	full_text
	
i64 %11
1shl8B(
&
	full_text

%46 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9ashr8B/
-
	full_text 

%47 = ashr exact i64 %46, 32
%i648B

	full_text
	
i64 %46
Xgetelementptr8BE
C
	full_text6
4
2%48 = getelementptr inbounds i32, i32* %2, i64 %47
%i648B

	full_text
	
i64 %47
1shl8B(
&
	full_text

%49 = shl i64 %11, 32
%i648B

	full_text
	
i64 %11
9add8B0
.
	full_text!

%50 = add i64 %49, 4294967296
%i648B

	full_text
	
i64 %49
9ashr8B/
-
	full_text 

%51 = ashr exact i64 %50, 32
%i648B

	full_text
	
i64 %50
Xgetelementptr8BE
C
	full_text6
4
2%52 = getelementptr inbounds i32, i32* %2, i64 %51
%i648B

	full_text
	
i64 %51
Ygetelementptr8BF
D
	full_text7
5
3%53 = getelementptr inbounds i32, i32* %18, i64 %47
'i32*8B

	full_text


i32* %18
%i648B

	full_text
	
i64 %47
5zext8B+
)
	full_text

%54 = zext i32 %8 to i64
'br8B

	full_text

br label %55
Dphi8B;
9
	full_text,
*
(%56 = phi i64 [ 0, %44 ], [ %156, %154 ]
&i648B

	full_text


i64 %156
Rphi8BI
G
	full_text:
8
6%57 = phi double [ 1.000000e+00, %44 ], [ %155, %154 ]
,double8B

	full_text

double %155
Xgetelementptr8BE
C
	full_text6
4
2%58 = getelementptr inbounds i32, i32* %3, i64 %56
%i648B

	full_text
	
i64 %56
Hload8B>
<
	full_text/
-
+%59 = load i32, i32* %58, align 4, !tbaa !8
'i32*8B

	full_text


i32* %58
6icmp8B,
*
	full_text

%60 = icmp sgt i32 %59, 0
%i328B

	full_text
	
i32 %59
;br8B3
1
	full_text$
"
 br i1 %60, label %61, label %154
#i18B

	full_text


i1 %60
7icmp8B-
+
	full_text

%62 = icmp eq i64 %45, %56
%i648B

	full_text
	
i64 %45
%i648B

	full_text
	
i64 %56
'br8B

	full_text

br label %63
Fphi8B=
;
	full_text.
,
*%64 = phi i32 [ %59, %61 ], [ %150, %149 ]
%i328B

	full_text
	
i32 %59
&i328B

	full_text


i32 %150
Dphi8B;
9
	full_text,
*
(%65 = phi i64 [ 0, %61 ], [ %151, %149 ]
&i648B

	full_text


i64 %151
ngetelementptr8B[
Y
	full_textL
J
H%66 = getelementptr inbounds [9 x i32], [9 x i32]* %15, i64 %56, i64 %65
3
[9 x i32]*8B!

	full_text

[9 x i32]* %15
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %65
Hload8B>
<
	full_text/
-
+%67 = load i32, i32* %66, align 4, !tbaa !8
'i32*8B

	full_text


i32* %66
7icmp8B-
+
	full_text

%68 = icmp eq i32 %67, %12
%i328B

	full_text
	
i32 %67
%i328B

	full_text
	
i32 %12
;br8B3
1
	full_text$
"
 br i1 %68, label %69, label %149
#i18B

	full_text


i1 %68
tgetelementptr8	Ba
_
	full_textR
P
N%70 = getelementptr inbounds [9 x double], [9 x double]* %16, i64 %56, i64 %65
9[9 x double]*8	B$
"
	full_text

[9 x double]* %16
%i648	B

	full_text
	
i64 %56
%i648	B

	full_text
	
i64 %65
Oload8	BE
C
	full_text6
4
2%71 = load double, double* %70, align 8, !tbaa !12
-double*8	B

	full_text

double* %70
7fmul8	B-
+
	full_text

%72 = fmul double %57, %71
+double8	B

	full_text


double %57
+double8	B

	full_text


double %71
6icmp8	B,
*
	full_text

%73 = icmp sgt i32 %64, 0
%i328	B

	full_text
	
i32 %64
;br8	B3
1
	full_text$
"
 br i1 %73, label %74, label %149
#i18	B

	full_text


i1 %73
'br8
B

	full_text

br label %75
Dphi8B;
9
	full_text,
*
(%76 = phi i64 [ %145, %139 ], [ 0, %74 ]
&i648B

	full_text


i64 %145
ngetelementptr8B[
Y
	full_textL
J
H%77 = getelementptr inbounds [9 x i32], [9 x i32]* %15, i64 %56, i64 %76
3
[9 x i32]*8B!

	full_text

[9 x i32]* %15
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %76
Hload8B>
<
	full_text/
-
+%78 = load i32, i32* %77, align 4, !tbaa !8
'i32*8B

	full_text


i32* %77
tgetelementptr8Ba
_
	full_textR
P
N%79 = getelementptr inbounds [9 x double], [9 x double]* %16, i64 %56, i64 %76
9[9 x double]*8B$
"
	full_text

[9 x double]* %16
%i648B

	full_text
	
i64 %56
%i648B

	full_text
	
i64 %76
Oload8BE
C
	full_text6
4
2%80 = load double, double* %79, align 8, !tbaa !12
-double*8B

	full_text

double* %79
7fmul8B-
+
	full_text

%81 = fmul double %72, %80
+double8B

	full_text


double %72
+double8B

	full_text


double %80
7icmp8B-
+
	full_text

%82 = icmp eq i32 %78, %12
%i328B

	full_text
	
i32 %78
%i328B

	full_text
	
i32 %12
1and8B(
&
	full_text

%83 = and i1 %62, %82
#i18B

	full_text


i1 %62
#i18B

	full_text


i1 %82
@fadd8B6
4
	full_text'
%
#%84 = fadd double %81, 1.000000e-01
+double8B

	full_text


double %81
Afadd8B7
5
	full_text(
&
$%85 = fadd double %84, -1.200000e+01
+double8B

	full_text


double %84
Jselect8B>
<
	full_text/
-
+%86 = select i1 %83, double %85, double %81
#i18B

	full_text


i1 %83
+double8B

	full_text


double %85
+double8B

	full_text


double %81
Hload8B>
<
	full_text/
-
+%87 = load i32, i32* %48, align 4, !tbaa !8
'i32*8B

	full_text


i32* %48
Hload8B>
<
	full_text/
-
+%88 = load i32, i32* %52, align 4, !tbaa !8
'i32*8B

	full_text


i32* %52
8icmp8B.
,
	full_text

%89 = icmp slt i32 %87, %88
%i328B

	full_text
	
i32 %87
%i328B

	full_text
	
i32 %88
;br8B3
1
	full_text$
"
 br i1 %89, label %90, label %139
#i18B

	full_text


i1 %89
6sext8B,
*
	full_text

%91 = sext i32 %87 to i64
%i328B

	full_text
	
i32 %87
6sext8B,
*
	full_text

%92 = sext i32 %88 to i64
%i328B

	full_text
	
i32 %88
'br8B

	full_text

br label %93
Fphi8B=
;
	full_text.
,
*%94 = phi i64 [ %91, %90 ], [ %135, %134 ]
%i648B

	full_text
	
i64 %91
&i648B

	full_text


i64 %135
Xgetelementptr8BE
C
	full_text6
4
2%95 = getelementptr inbounds i32, i32* %1, i64 %94
%i648B

	full_text
	
i64 %94
Hload8B>
<
	full_text/
-
+%96 = load i32, i32* %95, align 4, !tbaa !8
'i32*8B

	full_text


i32* %95
8icmp8B.
,
	full_text

%97 = icmp sgt i32 %96, %78
%i328B

	full_text
	
i32 %96
%i328B

	full_text
	
i32 %78
;br8B3
1
	full_text$
"
 br i1 %97, label %98, label %124
#i18B

	full_text


i1 %97
8trunc8B-
+
	full_text

%99 = trunc i64 %94 to i32
%i648B

	full_text
	
i64 %94
6add8B-
+
	full_text

%100 = add nsw i32 %88, -2
%i328B

	full_text
	
i32 %88
:icmp8B0
.
	full_text!

%101 = icmp slt i32 %100, %99
&i328B

	full_text


i32 %100
%i328B

	full_text
	
i32 %99
=br8B5
3
	full_text&
$
"br i1 %101, label %122, label %102
$i18B

	full_text
	
i1 %101
8sext8B.
,
	full_text

%103 = sext i32 %100 to i64
&i328B

	full_text


i32 %100
2shl8B)
'
	full_text

%104 = shl i64 %94, 32
%i648B

	full_text
	
i64 %94
;ashr8B1
/
	full_text"
 
%105 = ashr exact i64 %104, 32
&i648B

	full_text


i64 %104
(br8B 

	full_text

br label %106
Iphi8B@
>
	full_text1
/
-%107 = phi i64 [ %120, %119 ], [ %103, %102 ]
&i648B

	full_text


i64 %120
&i648B

	full_text


i64 %103
Zgetelementptr8BG
E
	full_text8
6
4%108 = getelementptr inbounds i32, i32* %1, i64 %107
&i648B

	full_text


i64 %107
Jload8B@
>
	full_text1
/
-%109 = load i32, i32* %108, align 4, !tbaa !8
(i32*8B

	full_text

	i32* %108
9icmp8B/
-
	full_text 

%110 = icmp sgt i32 %109, -1
&i328B

	full_text


i32 %109
=br8B5
3
	full_text&
$
"br i1 %110, label %111, label %119
$i18B

	full_text
	
i1 %110
`getelementptr8BM
K
	full_text>
<
:%112 = getelementptr inbounds double, double* %0, i64 %107
&i648B

	full_text


i64 %107
Cbitcast8B6
4
	full_text'
%
#%113 = bitcast double* %112 to i64*
.double*8B

	full_text

double* %112
Kload8BA
?
	full_text2
0
.%114 = load i64, i64* %113, align 8, !tbaa !12
(i64*8B

	full_text

	i64* %113
6add8B-
+
	full_text

%115 = add nsw i64 %107, 1
&i648B

	full_text


i64 %107
`getelementptr8BM
K
	full_text>
<
:%116 = getelementptr inbounds double, double* %0, i64 %115
&i648B

	full_text


i64 %115
Cbitcast8B6
4
	full_text'
%
#%117 = bitcast double* %116 to i64*
.double*8B

	full_text

double* %116
Kstore8B@
>
	full_text1
/
-store i64 %114, i64* %117, align 8, !tbaa !12
&i648B

	full_text


i64 %114
(i64*8B

	full_text

	i64* %117
Zgetelementptr8BG
E
	full_text8
6
4%118 = getelementptr inbounds i32, i32* %1, i64 %115
&i648B

	full_text


i64 %115
Jstore8B?
=
	full_text0
.
,store i32 %109, i32* %118, align 4, !tbaa !8
&i328B

	full_text


i32 %109
(i32*8B

	full_text

	i32* %118
(br8B 

	full_text

br label %119
7add8B.
,
	full_text

%120 = add nsw i64 %107, -1
&i648B

	full_text


i64 %107
;icmp8B1
/
	full_text"
 
%121 = icmp sgt i64 %107, %105
&i648B

	full_text


i64 %107
&i648B

	full_text


i64 %105
=br8B5
3
	full_text&
$
"br i1 %121, label %106, label %122
$i18B

	full_text
	
i1 %121
Hstore8B=
;
	full_text.
,
*store i32 %78, i32* %95, align 4, !tbaa !8
%i328B

	full_text
	
i32 %78
'i32*8B

	full_text


i32* %95
_getelementptr8BL
J
	full_text=
;
9%123 = getelementptr inbounds double, double* %0, i64 %94
%i648B

	full_text
	
i64 %94
Ystore8BN
L
	full_text?
=
;store double 0.000000e+00, double* %123, align 8, !tbaa !12
.double*8B

	full_text

double* %123
(br8B 

	full_text

br label %139
7icmp8B-
+
	full_text

%125 = icmp eq i32 %96, -1
%i328B

	full_text
	
i32 %96
=br8B5
3
	full_text&
$
"br i1 %125, label %126, label %128
$i18B

	full_text
	
i1 %125
9trunc8B.
,
	full_text

%127 = trunc i64 %94 to i32
%i648B

	full_text
	
i64 %94
Hstore8B=
;
	full_text.
,
*store i32 %78, i32* %95, align 4, !tbaa !8
%i328B

	full_text
	
i32 %78
'i32*8B

	full_text


i32* %95
(br8B 

	full_text

br label %139
8icmp8B.
,
	full_text

%129 = icmp eq i32 %96, %78
%i328B

	full_text
	
i32 %96
%i328B

	full_text
	
i32 %78
=br8B5
3
	full_text&
$
"br i1 %129, label %130, label %134
$i18B

	full_text
	
i1 %129
9trunc8B.
,
	full_text

%131 = trunc i64 %94 to i32
%i648B

	full_text
	
i64 %94
Iload8B?
=
	full_text0
.
,%132 = load i32, i32* %53, align 4, !tbaa !8
'i32*8B

	full_text


i32* %53
6add8B-
+
	full_text

%133 = add nsw i32 %132, 1
&i328B

	full_text


i32 %132
Istore8B>
<
	full_text/
-
+store i32 %133, i32* %53, align 4, !tbaa !8
&i328B

	full_text


i32 %133
'i32*8B

	full_text


i32* %53
(br8B 

	full_text

br label %139
1add8B(
&
	full_text

%135 = add i64 %94, 1
%i648B

	full_text
	
i64 %94
:icmp8B0
.
	full_text!

%136 = icmp slt i64 %135, %92
&i648B

	full_text


i64 %135
%i648B

	full_text
	
i64 %92
<br8B4
2
	full_text%
#
!br i1 %136, label %93, label %137
$i18B

	full_text
	
i1 %136
:trunc8B/
-
	full_text 

%138 = trunc i64 %135 to i32
&i648B

	full_text


i64 %135
(br8B 

	full_text

br label %139
vphi8Bm
k
	full_text^
\
Z%140 = phi i32 [ %131, %130 ], [ %127, %126 ], [ %99, %122 ], [ %87, %75 ], [ %138, %137 ]
&i328B

	full_text


i32 %131
&i328B

	full_text


i32 %127
%i328B

	full_text
	
i32 %99
%i328B

	full_text
	
i32 %87
&i328B

	full_text


i32 %138
8sext8B.
,
	full_text

%141 = sext i32 %140 to i64
&i328B

	full_text


i32 %140
`getelementptr8BM
K
	full_text>
<
:%142 = getelementptr inbounds double, double* %0, i64 %141
&i648B

	full_text


i64 %141
Qload8BG
E
	full_text8
6
4%143 = load double, double* %142, align 8, !tbaa !12
.double*8B

	full_text

double* %142
9fadd8B/
-
	full_text 

%144 = fadd double %86, %143
+double8B

	full_text


double %86
,double8B

	full_text

double %143
Qstore8BF
D
	full_text7
5
3store double %144, double* %142, align 8, !tbaa !12
,double8B

	full_text

double %144
.double*8B

	full_text

double* %142
9add8B0
.
	full_text!

%145 = add nuw nsw i64 %76, 1
%i648B

	full_text
	
i64 %76
Iload8B?
=
	full_text0
.
,%146 = load i32, i32* %58, align 4, !tbaa !8
'i32*8B

	full_text


i32* %58
8sext8B.
,
	full_text

%147 = sext i32 %146 to i64
&i328B

	full_text


i32 %146
;icmp8B1
/
	full_text"
 
%148 = icmp slt i64 %145, %147
&i648B

	full_text


i64 %145
&i648B

	full_text


i64 %147
<br8B4
2
	full_text%
#
!br i1 %148, label %75, label %149
$i18B

	full_text
	
i1 %148
Uphi8BL
J
	full_text=
;
9%150 = phi i32 [ %64, %69 ], [ %64, %63 ], [ %146, %139 ]
%i328B

	full_text
	
i32 %64
%i328B

	full_text
	
i32 %64
&i328B

	full_text


i32 %146
9add8B0
.
	full_text!

%151 = add nuw nsw i64 %65, 1
%i648B

	full_text
	
i64 %65
8sext8B.
,
	full_text

%152 = sext i32 %150 to i64
&i328B

	full_text


i32 %150
;icmp8B1
/
	full_text"
 
%153 = icmp slt i64 %151, %152
&i648B

	full_text


i64 %151
&i648B

	full_text


i64 %152
<br8B4
2
	full_text%
#
!br i1 %153, label %63, label %154
$i18B

	full_text
	
i1 %153
8fmul8B.
,
	full_text

%155 = fmul double %42, %57
+double8B

	full_text


double %42
+double8B

	full_text


double %57
9add8B0
.
	full_text!

%156 = add nuw nsw i64 %56, 1
%i648B

	full_text
	
i64 %56
9icmp8B/
-
	full_text 

%157 = icmp eq i64 %156, %54
&i648B

	full_text


i64 %156
%i648B

	full_text
	
i64 %54
<br8B4
2
	full_text%
#
!br i1 %157, label %158, label %55
$i18B

	full_text
	
i1 %157
$ret8B

	full_text


ret void
&i32*8B

	full_text
	
i32* %3
&i32*8B

	full_text
	
i32* %4
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %5
&i32*8B

	full_text
	
i32* %1
$i328B

	full_text


i32 %9
&i32*8B

	full_text
	
i32* %2
$i328B

	full_text


i32 %8
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
$i648B

	full_text


i64 -1
#i328B

	full_text	

i32 1
$i648B

	full_text


i64 32
4double8B&
$
	full_text

double 0.000000e+00
,i648B!

	full_text

i64 4294967296
$i328B

	full_text


i32 -1
#i648B

	full_text	

i64 1
4double8B&
$
	full_text

double 1.000000e-01
#i648B

	full_text	

i64 0
#i328B

	full_text	

i32 0
4double8B&
$
	full_text

double 1.000000e+00
5double8B'
%
	full_text

double -1.200000e+01
$i328B

	full_text


i32 -2
,i648B!

	full_text

i64 4294967295       		 

                       !  # "" $& %' %% () (( *+ ** ,- ,, ./ .. 01 00 23 22 45 44 67 68 66 9: 9< ;= ;; >? >> @@ AB AA CD CC EE FG FI HH JK JJ LM LL NO NN PQ PP RS RR TU TT VW VV XY XZ XX [[ \^ ]] _` __ ab aa cd cc ef ee gh gj ik ii ln mo mm pq pp rs rt ru rr vw vv xy xz xx {| {~ } }	€ }} ‚  ƒ„ ƒ
… ƒƒ †‡ †† ‰  ‹‹  
 
  ‘’ ‘‘ “” “
• “
– ““ — —— ™ ™
› ™™  
    
΅  Ά£ ΆΆ ¤¥ ¤¤ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬­ ¬¬ ®― ®
° ®® ±² ±΄ ³³ µ¶ µµ ·Ή Έ
Ί ΈΈ »
Ό »» ½Ύ ½½ Ώΐ Ώ
Α ΏΏ ΒΓ ΒΕ ΔΔ ΖΗ ΖΖ ΘΙ Θ
Κ ΘΘ ΛΜ ΛΞ ΝΝ ΟΠ ΟΟ ΡÒ ΡΡ ΣΥ Τ
Φ ΤΤ Χ
Ψ ΧΧ ΩΪ ΩΩ Ϋά ΫΫ έή έ
ΰ ίί αβ αα γδ γγ εζ εε η
θ ηη ικ ιι λμ λ
ν λλ ξ
ο ξξ πρ π
ς ππ συ ττ φχ φ
ψ φφ ωϊ ωό ϋ
ύ ϋϋ ώ
ÿ ώώ €
 €€ ‚„ ƒƒ …† … ‡‡ ‰ ‰
‹ ‰‰  
  ‘ “ ’’ ”• ”” –— –– ™ 
  ›   
   ΅Ά ΅¤ ££ ¥§ ¦
¨ ¦
© ¦
ª ¦
« ¦¦ ¬­ ¬¬ ®
― ®® °± °° ²³ ²
΄ ²² µ¶ µ
· µµ ΈΉ ΈΈ Ί» ΊΊ Ό½ ΌΌ ΎΏ Ύ
ΐ ΎΎ ΑΒ ΑΔ Γ
Ε Γ
Ζ ΓΓ ΗΘ ΗΗ ΙΚ ΙΙ ΛΜ Λ
Ν ΛΛ ΞΟ ΞΡ Π
Ò ΠΠ ΣΤ ΣΣ ΥΦ Υ
Χ ΥΥ ΨΩ ΨΫ aά έ (έ ίέ ηέ ώέ ®ή 	ί ί ,ί »ί Χί ξΰ 
α α α Nα V	β β @β Eβ [   
            ! #" &0 '% )( +% -, /% 1 32 50 74 86 : < =; ?@ BA DE G I KJ ML O QP SR UT W YL ZΣ ^Π `] ba dc fe hH j] kc nΓ oΗ q s] tp ur wv y zx |	 ~] p €} ‚_ „ …m ‡† ‰Έ  ] ‹  ’	 ”] •‹ –“ ƒ — ›‘  i   ΅™ £Ά ¥ §¤ ¨™ ©N «V ­ª ―¬ °® ²ª ΄¬ ¶³ Ή ΊΈ Ό» Ύ½ ΐ‘ ΑΏ ΓΈ Ε¬ ΗΖ ΙΔ ΚΘ ΜΖ ΞΈ ΠΟ Òτ ΥΝ ΦΤ ΨΧ ΪΩ άΫ ήΤ ΰί βα δΤ ζε θη κγ μι νε οΩ ρξ ςΤ υΤ χΡ ψφ ϊ‘ ό» ύΈ ÿώ ½ „ƒ †Έ ‘ » ‹½ ‘  ‘Έ “X •” —– ™X Έ  µ   Ά ¤’ §‡ ¨Δ ©ª ª£ «¦ ­¬ ―® ±¦ ³° ΄² ¶® ·‹ Ήa »Ί ½Έ ΏΌ ΐΎ Βm Δm ΕΊ Ζp ΘΓ ΚΗ ΜΙ ΝΛ ΟC Ρ_ Ò] ΤΣ Φ[ ΧΥ Ω  Ϊ  "  ;$ %F HF Ϊ9 %9 ;\ ]g ig Πl mΨ ΪΨ ]{ }{ Γ  ΓΞ mΞ Π ‹± ³± ¦· ΈΑ ‹Α ΓΒ ΔΒ ƒΛ ϋΛ Ν… ‡… ‚ ¦Σ Τ ¦ ’ έ ίέ τ› ¦΅ Έ΅ £σ τω Τω ϋ¥ ¦ Ϊ γγ δδC δδ C γγ 
ε τ
ζ –	η 	η 	η 	η J	η L	η P	η T
η Ο
η Ρθ *θ €	ι 	ι Rκ .
κ Ϋ
κ ƒ	λ 0
λ ε
λ 
λ Έ
λ Η
λ Σμ C
μ Άν ]ν p
ν ‹ξ ξ >	ξ E	ξ e
ξ †ο Aο _
π ¤
ρ Ζ	ς H"	
makea_3"
_Z13get_global_idj"

_Z3powdd*
npb-CG-makea_3.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02

wgsize
€

transfer_bytes
όδƒ

devmap_label
 

wgsize_log1p
½„A
 
transfer_bytes_log1p
½„A