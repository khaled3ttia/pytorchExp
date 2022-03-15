

[external]
8allocaB.
,
	full_text

%9 = alloca double, align 8
9allocaB/
-
	full_text 

%10 = alloca double, align 8
9allocaB/
-
	full_text 

%11 = alloca double, align 8
BallocaB8
6
	full_text)
'
%%12 = alloca [256 x double], align 16
=bitcastB2
0
	full_text#
!
%13 = bitcast double* %9 to i8*
*double*B

	full_text


double* %9
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %13) #6
#i8*B

	full_text
	
i8* %13
>bitcastB3
1
	full_text$
"
 %14 = bitcast double* %10 to i8*
+double*B

	full_text

double* %10
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %14) #6
#i8*B

	full_text
	
i8* %14
>bitcastB3
1
	full_text$
"
 %15 = bitcast double* %11 to i8*
+double*B

	full_text

double* %11
YcallBQ
O
	full_textB
@
>call void @llvm.lifetime.start.p0i8(i64 8, i8* nonnull %15) #6
#i8*B

	full_text
	
i8* %15
FbitcastB;
9
	full_text,
*
(%16 = bitcast [256 x double]* %12 to i8*
;[256 x double]*B&
$
	full_text

[256 x double]* %12
\callBT
R
	full_textE
C
Acall void @llvm.lifetime.start.p0i8(i64 2048, i8* nonnull %16) #6
#i8*B

	full_text
	
i8* %16
LcallBD
B
	full_text5
3
1%17 = tail call i64 @_Z13get_global_idj(i32 0) #7
KcallBC
A
	full_text4
2
0%18 = tail call i64 @_Z12get_local_idj(i32 0) #7
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
McallBE
C
	full_text6
4
2%20 = tail call i64 @_Z14get_local_sizej(i32 0) #7
6truncB-
+
	full_text

%21 = trunc i64 %20 to i32
#i64B

	full_text
	
i64 %20
/shlB(
&
	full_text

%22 = shl i64 %20, 32
#i64B

	full_text
	
i64 %20
7ashrB/
-
	full_text 

%23 = ashr exact i64 %22, 32
#i64B

	full_text
	
i64 %22
/shlB(
&
	full_text

%24 = shl i64 %18, 32
#i64B

	full_text
	
i64 %18
7ashrB/
-
	full_text 

%25 = ashr exact i64 %24, 32
#i64B

	full_text
	
i64 %24
\getelementptrBK
I
	full_text<
:
8%26 = getelementptr inbounds double, double* %0, i64 %25
#i64B

	full_text
	
i64 %25
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %26, align 8, !tbaa !8
+double*B

	full_text

double* %26
4addB-
+
	full_text

%27 = add nsw i64 %23, %25
#i64B

	full_text
	
i64 %23
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%28 = getelementptr inbounds double, double* %0, i64 %27
#i64B

	full_text
	
i64 %27
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %28, align 8, !tbaa !8
+double*B

	full_text

double* %28
7ashrB/
-
	full_text 

%29 = ashr exact i64 %22, 31
#i64B

	full_text
	
i64 %22
4addB-
+
	full_text

%30 = add nsw i64 %29, %25
#i64B

	full_text
	
i64 %29
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%31 = getelementptr inbounds double, double* %0, i64 %30
#i64B

	full_text
	
i64 %30
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %31, align 8, !tbaa !8
+double*B

	full_text

double* %31
2mulB+
)
	full_text

%32 = mul nsw i64 %23, 3
#i64B

	full_text
	
i64 %23
4addB-
+
	full_text

%33 = add nsw i64 %32, %25
#i64B

	full_text
	
i64 %32
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%34 = getelementptr inbounds double, double* %0, i64 %33
#i64B

	full_text
	
i64 %33
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %34, align 8, !tbaa !8
+double*B

	full_text

double* %34
7ashrB/
-
	full_text 

%35 = ashr exact i64 %22, 30
#i64B

	full_text
	
i64 %22
4addB-
+
	full_text

%36 = add nsw i64 %35, %25
#i64B

	full_text
	
i64 %35
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%37 = getelementptr inbounds double, double* %0, i64 %36
#i64B

	full_text
	
i64 %36
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %37, align 8, !tbaa !8
+double*B

	full_text

double* %37
2mulB+
)
	full_text

%38 = mul nsw i64 %23, 5
#i64B

	full_text
	
i64 %23
4addB-
+
	full_text

%39 = add nsw i64 %38, %25
#i64B

	full_text
	
i64 %38
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%40 = getelementptr inbounds double, double* %0, i64 %39
#i64B

	full_text
	
i64 %39
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %40, align 8, !tbaa !8
+double*B

	full_text

double* %40
2mulB+
)
	full_text

%41 = mul nsw i64 %23, 6
#i64B

	full_text
	
i64 %23
4addB-
+
	full_text

%42 = add nsw i64 %41, %25
#i64B

	full_text
	
i64 %41
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%43 = getelementptr inbounds double, double* %0, i64 %42
#i64B

	full_text
	
i64 %42
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %43, align 8, !tbaa !8
+double*B

	full_text

double* %43
2mulB+
)
	full_text

%44 = mul nsw i64 %23, 7
#i64B

	full_text
	
i64 %23
4addB-
+
	full_text

%45 = add nsw i64 %44, %25
#i64B

	full_text
	
i64 %44
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%46 = getelementptr inbounds double, double* %0, i64 %45
#i64B

	full_text
	
i64 %45
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %46, align 8, !tbaa !8
+double*B

	full_text

double* %46
7ashrB/
-
	full_text 

%47 = ashr exact i64 %22, 29
#i64B

	full_text
	
i64 %22
4addB-
+
	full_text

%48 = add nsw i64 %47, %25
#i64B

	full_text
	
i64 %47
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%49 = getelementptr inbounds double, double* %0, i64 %48
#i64B

	full_text
	
i64 %48
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %49, align 8, !tbaa !8
+double*B

	full_text

double* %49
2mulB+
)
	full_text

%50 = mul nsw i64 %23, 9
#i64B

	full_text
	
i64 %23
4addB-
+
	full_text

%51 = add nsw i64 %50, %25
#i64B

	full_text
	
i64 %50
#i64B

	full_text
	
i64 %25
\getelementptrBK
I
	full_text<
:
8%52 = getelementptr inbounds double, double* %0, i64 %51
#i64B

	full_text
	
i64 %51
UstoreBL
J
	full_text=
;
9store double 0.000000e+00, double* %52, align 8, !tbaa !8
+double*B

	full_text

double* %52
6truncB-
+
	full_text

%53 = trunc i64 %17 to i32
#i64B

	full_text
	
i64 %17
-addB&
$
	full_text

%54 = add i32 %6, 1
0addB)
'
	full_text

%55 = add i32 %54, %53
#i32B

	full_text
	
i32 %54
#i32B

	full_text
	
i32 %53
ZstoreBQ
O
	full_textB
@
>store double 0x41B033C4D7000000, double* %9, align 8, !tbaa !8
*double*B

	full_text


double* %9
KstoreBB
@
	full_text3
1
/store double %7, double* %10, align 8, !tbaa !8
+double*B

	full_text

double* %10
%brB

	full_text

br label %56
Aphi8B8
6
	full_text)
'
%%57 = phi i32 [ 1, %8 ], [ %71, %68 ]
%i328B

	full_text
	
i32 %71
Cphi8B:
8
	full_text+
)
'%58 = phi i32 [ %55, %8 ], [ %59, %68 ]
%i328B

	full_text
	
i32 %55
%i328B

	full_text
	
i32 %59
2sdiv8B(
&
	full_text

%59 = sdiv i32 %58, 2
%i328B

	full_text
	
i32 %58
4shl8B+
)
	full_text

%60 = shl nsw i32 %59, 1
%i328B

	full_text
	
i32 %59
7icmp8B-
+
	full_text

%61 = icmp eq i32 %60, %58
%i328B

	full_text
	
i32 %60
%i328B

	full_text
	
i32 %58
:br8B2
0
	full_text#
!
br i1 %61, label %65, label %62
#i18B

	full_text


i1 %61
Nload8BD
B
	full_text5
3
1%63 = load double, double* %10, align 8, !tbaa !8
-double*8B

	full_text

double* %10
Ycall8BO
M
	full_text@
>
<%64 = call double @randlc(double* nonnull %9, double %63) #6
,double*8B

	full_text


double* %9
+double8B

	full_text


double %63
'br8B

	full_text

br label %65
0add8B'
%
	full_text

%66 = add i32 %58, 1
%i328B

	full_text
	
i32 %58
6icmp8B,
*
	full_text

%67 = icmp ult i32 %66, 3
%i328B

	full_text
	
i32 %66
:br8B2
0
	full_text#
!
br i1 %67, label %73, label %68
#i18B

	full_text


i1 %67
Nload8BD
B
	full_text5
3
1%69 = load double, double* %10, align 8, !tbaa !8
-double*8B

	full_text

double* %10
Zcall8BP
N
	full_textA
?
=%70 = call double @randlc(double* nonnull %10, double %69) #6
-double*8B

	full_text

double* %10
+double8B

	full_text


double %69
8add8B/
-
	full_text 

%71 = add nuw nsw i32 %57, 1
%i328B

	full_text
	
i32 %57
8icmp8B.
,
	full_text

%72 = icmp ult i32 %57, 100
%i328B

	full_text
	
i32 %57
:br8B2
0
	full_text#
!
br i1 %72, label %56, label %73
#i18B

	full_text


i1 %72
@bitcast8B3
1
	full_text$
"
 %74 = bitcast double* %9 to i64*
,double*8B

	full_text


double* %9
Hload8B>
<
	full_text/
-
+%75 = load i64, i64* %74, align 8, !tbaa !8
'i64*8B

	full_text


i64* %74
Abitcast8B4
2
	full_text%
#
!%76 = bitcast double* %11 to i64*
-double*8B

	full_text

double* %11
Hstore8B=
;
	full_text.
,
*store i64 %75, i64* %76, align 8, !tbaa !8
%i648B

	full_text
	
i64 %75
'i64*8B

	full_text


i64* %76
tgetelementptr8Ba
_
	full_textR
P
N%77 = getelementptr inbounds [256 x double], [256 x double]* %12, i64 0, i64 0
=[256 x double]*8B&
$
	full_text

[256 x double]* %12
'br8B

	full_text

br label %78
Rphi8BI
G
	full_text:
8
6%79 = phi double [ 0.000000e+00, %73 ], [ %120, %123 ]
,double8B

	full_text

double %120
Rphi8BI
G
	full_text:
8
6%80 = phi double [ 0.000000e+00, %73 ], [ %119, %123 ]
,double8B

	full_text

double %119
Dphi8B;
9
	full_text,
*
(%81 = phi i32 [ 0, %73 ], [ %124, %123 ]
&i328B

	full_text


i32 %124
call8Bu
s
	full_textf
d
bcall void @vranlc(i32 256, double* nonnull %11, double 0x41D2309CE5400000, double* nonnull %77) #6
-double*8B

	full_text

double* %11
-double*8B

	full_text

double* %77
'br8B

	full_text

br label %82
Dphi8B;
9
	full_text,
*
(%83 = phi i64 [ 0, %78 ], [ %121, %118 ]
&i648B

	full_text


i64 %121
Iphi8B@
>
	full_text1
/
-%84 = phi double [ %79, %78 ], [ %120, %118 ]
+double8B

	full_text


double %79
,double8B

	full_text

double %120
Iphi8B@
>
	full_text1
/
-%85 = phi double [ %80, %78 ], [ %119, %118 ]
+double8B

	full_text


double %80
,double8B

	full_text

double %119
4shl8B+
)
	full_text

%86 = shl nsw i64 %83, 1
%i648B

	full_text
	
i64 %83
vgetelementptr8Bc
a
	full_textT
R
P%87 = getelementptr inbounds [256 x double], [256 x double]* %12, i64 0, i64 %86
=[256 x double]*8B&
$
	full_text

[256 x double]* %12
%i648B

	full_text
	
i64 %86
Oload8BE
C
	full_text6
4
2%88 = load double, double* %87, align 16, !tbaa !8
-double*8B

	full_text

double* %87
wcall8Bm
k
	full_text^
\
Z%89 = call double @llvm.fmuladd.f64(double %88, double 2.000000e+00, double -1.000000e+00)
+double8B

	full_text


double %88
.or8B&
$
	full_text

%90 = or i64 %86, 1
%i648B

	full_text
	
i64 %86
vgetelementptr8Bc
a
	full_textT
R
P%91 = getelementptr inbounds [256 x double], [256 x double]* %12, i64 0, i64 %90
=[256 x double]*8B&
$
	full_text

[256 x double]* %12
%i648B

	full_text
	
i64 %90
Nload8BD
B
	full_text5
3
1%92 = load double, double* %91, align 8, !tbaa !8
-double*8B

	full_text

double* %91
wcall8Bm
k
	full_text^
\
Z%93 = call double @llvm.fmuladd.f64(double %92, double 2.000000e+00, double -1.000000e+00)
+double8B

	full_text


double %92
7fmul8B-
+
	full_text

%94 = fmul double %93, %93
+double8B

	full_text


double %93
+double8B

	full_text


double %93
dcall8BZ
X
	full_textK
I
G%95 = call double @llvm.fmuladd.f64(double %89, double %89, double %94)
+double8B

	full_text


double %89
+double8B

	full_text


double %89
+double8B

	full_text


double %94
Dfcmp8B:
8
	full_text+
)
'%96 = fcmp ugt double %95, 1.000000e+00
+double8B

	full_text


double %95
;br8B3
1
	full_text$
"
 br i1 %96, label %118, label %97
#i18B

	full_text


i1 %96
Fcall8B<
:
	full_text-
+
)%98 = call double @_Z3logd(double %95) #7
+double8B

	full_text


double %95
Afmul8B7
5
	full_text(
&
$%99 = fmul double %98, -2.000000e+00
+double8B

	full_text


double %98
8fdiv8B.
,
	full_text

%100 = fdiv double %99, %95
+double8B

	full_text


double %99
+double8B

	full_text


double %95
Icall8B?
=
	full_text0
.
,%101 = call double @_Z4sqrtd(double %100) #7
,double8B

	full_text

double %100
Ostore8BD
B
	full_text5
3
1store double %101, double* %10, align 8, !tbaa !8
,double8B

	full_text

double %101
-double*8B

	full_text

double* %10
9fmul8B/
-
	full_text 

%102 = fmul double %89, %101
+double8B

	full_text


double %89
,double8B

	full_text

double %101
9fmul8B/
-
	full_text 

%103 = fmul double %93, %101
+double8B

	full_text


double %93
,double8B

	full_text

double %101
Icall8B?
=
	full_text0
.
,%104 = call double @_Z4fabsd(double %102) #7
,double8B

	full_text

double %102
Icall8B?
=
	full_text0
.
,%105 = call double @_Z4fabsd(double %103) #7
,double8B

	full_text

double %103
>fcmp8B4
2
	full_text%
#
!%106 = fcmp ogt double %104, %105
,double8B

	full_text

double %104
,double8B

	full_text

double %105
Nselect8BB
@
	full_text3
1
/%107 = select i1 %106, double %102, double %103
$i18B

	full_text
	
i1 %106
,double8B

	full_text

double %102
,double8B

	full_text

double %103
Icall8B?
=
	full_text0
.
,%108 = call double @_Z4fabsd(double %107) #7
,double8B

	full_text

double %107
?fptosi8B3
1
	full_text$
"
 %109 = fptosi double %108 to i32
,double8B

	full_text

double %108
8mul8B/
-
	full_text 

%110 = mul nsw i32 %109, %21
&i328B

	full_text


i32 %109
%i328B

	full_text
	
i32 %21
8add8B/
-
	full_text 

%111 = add nsw i32 %110, %19
&i328B

	full_text


i32 %110
%i328B

	full_text
	
i32 %19
8sext8B.
,
	full_text

%112 = sext i32 %111 to i64
&i328B

	full_text


i32 %111
`getelementptr8BM
K
	full_text>
<
:%113 = getelementptr inbounds double, double* %0, i64 %112
&i648B

	full_text


i64 %112
Pload8BF
D
	full_text7
5
3%114 = load double, double* %113, align 8, !tbaa !8
.double*8B

	full_text

double* %113
Bfadd8B8
6
	full_text)
'
%%115 = fadd double %114, 1.000000e+00
,double8B

	full_text

double %114
Pstore8BE
C
	full_text6
4
2store double %115, double* %113, align 8, !tbaa !8
,double8B

	full_text

double %115
.double*8B

	full_text

double* %113
9fadd8B/
-
	full_text 

%116 = fadd double %85, %102
+double8B

	full_text


double %85
,double8B

	full_text

double %102
9fadd8B/
-
	full_text 

%117 = fadd double %84, %103
+double8B

	full_text


double %84
,double8B

	full_text

double %103
(br8B 

	full_text

br label %118
Iphi8	B@
>
	full_text1
/
-%119 = phi double [ %116, %97 ], [ %85, %82 ]
,double8	B

	full_text

double %116
+double8	B

	full_text


double %85
Iphi8	B@
>
	full_text1
/
-%120 = phi double [ %117, %97 ], [ %84, %82 ]
,double8	B

	full_text

double %117
+double8	B

	full_text


double %84
9add8	B0
.
	full_text!

%121 = add nuw nsw i64 %83, 1
%i648	B

	full_text
	
i64 %83
9icmp8	B/
-
	full_text 

%122 = icmp eq i64 %121, 128
&i648	B

	full_text


i64 %121
<br8	B4
2
	full_text%
#
!br i1 %122, label %123, label %82
$i18	B

	full_text
	
i1 %122
Mstore8
BB
@
	full_text3
1
/store double %95, double* %9, align 8, !tbaa !8
+double8
B

	full_text


double %95
,double*8
B

	full_text


double* %9
;add8
B2
0
	full_text#
!
%124 = add nuw nsw i32 %81, 128
%i328
B

	full_text
	
i32 %81
;icmp8
B1
/
	full_text"
 
%125 = icmp ult i32 %81, 65408
%i328
B

	full_text
	
i32 %81
<br8
B4
2
	full_text%
#
!br i1 %125, label %78, label %126
$i18
B

	full_text
	
i1 %125
2shl8B)
'
	full_text

%127 = shl i64 %18, 32
%i648B

	full_text
	
i64 %18
;ashr8B1
/
	full_text"
 
%128 = ashr exact i64 %127, 32
&i648B

	full_text


i64 %127
`getelementptr8BM
K
	full_text>
<
:%129 = getelementptr inbounds double, double* %1, i64 %128
&i648B

	full_text


i64 %128
Pstore8BE
C
	full_text6
4
2store double %119, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %119
.double*8B

	full_text

double* %129
`getelementptr8BM
K
	full_text>
<
:%130 = getelementptr inbounds double, double* %2, i64 %128
&i648B

	full_text


i64 %128
Pstore8BE
C
	full_text6
4
2store double %120, double* %130, align 8, !tbaa !8
,double8B

	full_text

double %120
.double*8B

	full_text

double* %130
=call8B3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #8
3lshr8B)
'
	full_text

%131 = lshr i64 %20, 1
%i648B

	full_text
	
i64 %20
:trunc8B/
-
	full_text 

%132 = trunc i64 %131 to i32
&i648B

	full_text


i64 %131
8icmp8B.
,
	full_text

%133 = icmp sgt i32 %132, 0
&i328B

	full_text


i32 %132
(br8B 

	full_text

br label %134
Fphi8B=
;
	full_text.
,
*%135 = phi i64 [ 0, %126 ], [ %155, %154 ]
&i648B

	full_text


i64 %155
=br8B5
3
	full_text&
$
"br i1 %133, label %136, label %154
$i18B

	full_text
	
i1 %133
8mul8B/
-
	full_text 

%137 = mul nsw i64 %135, %23
&i648B

	full_text


i64 %135
%i648B

	full_text
	
i64 %23
8add8B/
-
	full_text 

%138 = add nsw i64 %137, %25
&i648B

	full_text


i64 %137
%i648B

	full_text
	
i64 %25
`getelementptr8BM
K
	full_text>
<
:%139 = getelementptr inbounds double, double* %0, i64 %138
&i648B

	full_text


i64 %138
:trunc8B/
-
	full_text 

%140 = trunc i64 %138 to i32
&i648B

	full_text


i64 %138
(br8B 

	full_text

br label %141
Iphi8B@
>
	full_text1
/
-%142 = phi i32 [ %132, %136 ], [ %152, %151 ]
&i328B

	full_text


i32 %132
&i328B

	full_text


i32 %152
:icmp8B0
.
	full_text!

%143 = icmp sgt i32 %142, %19
&i328B

	full_text


i32 %142
%i328B

	full_text
	
i32 %19
=br8B5
3
	full_text&
$
"br i1 %143, label %144, label %151
$i18B

	full_text
	
i1 %143
9add8B0
.
	full_text!

%145 = add nsw i32 %142, %140
&i328B

	full_text


i32 %142
&i328B

	full_text


i32 %140
8sext8B.
,
	full_text

%146 = sext i32 %145 to i64
&i328B

	full_text


i32 %145
`getelementptr8BM
K
	full_text>
<
:%147 = getelementptr inbounds double, double* %0, i64 %146
&i648B

	full_text


i64 %146
Pload8BF
D
	full_text7
5
3%148 = load double, double* %147, align 8, !tbaa !8
.double*8B

	full_text

double* %147
Pload8BF
D
	full_text7
5
3%149 = load double, double* %139, align 8, !tbaa !8
.double*8B

	full_text

double* %139
:fadd8B0
.
	full_text!

%150 = fadd double %148, %149
,double8B

	full_text

double %148
,double8B

	full_text

double %149
Pstore8BE
C
	full_text6
4
2store double %150, double* %139, align 8, !tbaa !8
,double8B

	full_text

double %150
.double*8B

	full_text

double* %139
(br8B 

	full_text

br label %151
=call8B3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #8
4lshr8B*
(
	full_text

%152 = lshr i32 %142, 1
&i328B

	full_text


i32 %142
7icmp8B-
+
	full_text

%153 = icmp eq i32 %152, 0
&i328B

	full_text


i32 %152
=br8B5
3
	full_text&
$
"br i1 %153, label %154, label %141
$i18B

	full_text
	
i1 %153
:add8B1
/
	full_text"
 
%155 = add nuw nsw i64 %135, 1
&i648B

	full_text


i64 %135
8icmp8B.
,
	full_text

%156 = icmp eq i64 %155, 10
&i648B

	full_text


i64 %155
=br8B5
3
	full_text&
$
"br i1 %156, label %157, label %134
$i18B

	full_text
	
i1 %156
=br8B5
3
	full_text&
$
"br i1 %133, label %158, label %176
$i18B

	full_text
	
i1 %133
(br8B 

	full_text

br label %159
Iphi8B@
>
	full_text1
/
-%160 = phi i32 [ %174, %173 ], [ %132, %158 ]
&i328B

	full_text


i32 %174
&i328B

	full_text


i32 %132
:icmp8B0
.
	full_text!

%161 = icmp sgt i32 %160, %19
&i328B

	full_text


i32 %160
%i328B

	full_text
	
i32 %19
=br8B5
3
	full_text&
$
"br i1 %161, label %162, label %173
$i18B

	full_text
	
i1 %161
8add8B/
-
	full_text 

%163 = add nsw i32 %160, %19
&i328B

	full_text


i32 %160
%i328B

	full_text
	
i32 %19
8sext8B.
,
	full_text

%164 = sext i32 %163 to i64
&i328B

	full_text


i32 %163
`getelementptr8BM
K
	full_text>
<
:%165 = getelementptr inbounds double, double* %1, i64 %164
&i648B

	full_text


i64 %164
Pload8BF
D
	full_text7
5
3%166 = load double, double* %165, align 8, !tbaa !8
.double*8B

	full_text

double* %165
Pload8BF
D
	full_text7
5
3%167 = load double, double* %129, align 8, !tbaa !8
.double*8B

	full_text

double* %129
:fadd8B0
.
	full_text!

%168 = fadd double %166, %167
,double8B

	full_text

double %166
,double8B

	full_text

double %167
Pstore8BE
C
	full_text6
4
2store double %168, double* %129, align 8, !tbaa !8
,double8B

	full_text

double %168
.double*8B

	full_text

double* %129
`getelementptr8BM
K
	full_text>
<
:%169 = getelementptr inbounds double, double* %2, i64 %164
&i648B

	full_text


i64 %164
Pload8BF
D
	full_text7
5
3%170 = load double, double* %169, align 8, !tbaa !8
.double*8B

	full_text

double* %169
Pload8BF
D
	full_text7
5
3%171 = load double, double* %130, align 8, !tbaa !8
.double*8B

	full_text

double* %130
:fadd8B0
.
	full_text!

%172 = fadd double %170, %171
,double8B

	full_text

double %170
,double8B

	full_text

double %171
Pstore8BE
C
	full_text6
4
2store double %172, double* %130, align 8, !tbaa !8
,double8B

	full_text

double %172
.double*8B

	full_text

double* %130
(br8B 

	full_text

br label %173
=call8B3
1
	full_text$
"
 call void @_Z7barrierj(i32 1) #8
4lshr8B*
(
	full_text

%174 = lshr i32 %160, 1
&i328B

	full_text


i32 %160
7icmp8B-
+
	full_text

%175 = icmp eq i32 %174, 0
&i328B

	full_text


i32 %174
=br8B5
3
	full_text&
$
"br i1 %175, label %176, label %159
$i18B

	full_text
	
i1 %175
6icmp8B,
*
	full_text

%177 = icmp eq i32 %19, 0
%i328B

	full_text
	
i32 %19
=br8B5
3
	full_text&
$
"br i1 %177, label %178, label %250
$i18B

	full_text
	
i1 %177
Icall8B?
=
	full_text0
.
,%179 = call i64 @_Z12get_group_idj(i32 0) #7
<mul8B3
1
	full_text$
"
 %180 = mul i64 %179, 42949672960
&i648B

	full_text


i64 %179
;ashr8B1
/
	full_text"
 
%181 = ashr exact i64 %180, 32
&i648B

	full_text


i64 %180
Abitcast8B4
2
	full_text%
#
!%182 = bitcast double* %0 to i64*
Jload8B@
>
	full_text1
/
-%183 = load i64, i64* %182, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %182
`getelementptr8BM
K
	full_text>
<
:%184 = getelementptr inbounds double, double* %3, i64 %181
&i648B

	full_text


i64 %181
Cbitcast8B6
4
	full_text'
%
#%185 = bitcast double* %184 to i64*
.double*8B

	full_text

double* %184
Jstore8B?
=
	full_text0
.
,store i64 %183, i64* %185, align 8, !tbaa !8
&i648B

	full_text


i64 %183
(i64*8B

	full_text

	i64* %185
_getelementptr8BL
J
	full_text=
;
9%186 = getelementptr inbounds double, double* %0, i64 %23
%i648B

	full_text
	
i64 %23
Cbitcast8B6
4
	full_text'
%
#%187 = bitcast double* %186 to i64*
.double*8B

	full_text

double* %186
Jload8B@
>
	full_text1
/
-%188 = load i64, i64* %187, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %187
0or8B(
&
	full_text

%189 = or i64 %181, 1
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%190 = getelementptr inbounds double, double* %3, i64 %189
&i648B

	full_text


i64 %189
Cbitcast8B6
4
	full_text'
%
#%191 = bitcast double* %190 to i64*
.double*8B

	full_text

double* %190
Jstore8B?
=
	full_text0
.
,store i64 %188, i64* %191, align 8, !tbaa !8
&i648B

	full_text


i64 %188
(i64*8B

	full_text

	i64* %191
_getelementptr8BL
J
	full_text=
;
9%192 = getelementptr inbounds double, double* %0, i64 %29
%i648B

	full_text
	
i64 %29
Cbitcast8B6
4
	full_text'
%
#%193 = bitcast double* %192 to i64*
.double*8B

	full_text

double* %192
Jload8B@
>
	full_text1
/
-%194 = load i64, i64* %193, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %193
6add8B-
+
	full_text

%195 = add nsw i64 %181, 2
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%196 = getelementptr inbounds double, double* %3, i64 %195
&i648B

	full_text


i64 %195
Cbitcast8B6
4
	full_text'
%
#%197 = bitcast double* %196 to i64*
.double*8B

	full_text

double* %196
Jstore8B?
=
	full_text0
.
,store i64 %194, i64* %197, align 8, !tbaa !8
&i648B

	full_text


i64 %194
(i64*8B

	full_text

	i64* %197
_getelementptr8BL
J
	full_text=
;
9%198 = getelementptr inbounds double, double* %0, i64 %32
%i648B

	full_text
	
i64 %32
Cbitcast8B6
4
	full_text'
%
#%199 = bitcast double* %198 to i64*
.double*8B

	full_text

double* %198
Jload8B@
>
	full_text1
/
-%200 = load i64, i64* %199, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %199
6add8B-
+
	full_text

%201 = add nsw i64 %181, 3
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%202 = getelementptr inbounds double, double* %3, i64 %201
&i648B

	full_text


i64 %201
Cbitcast8B6
4
	full_text'
%
#%203 = bitcast double* %202 to i64*
.double*8B

	full_text

double* %202
Jstore8B?
=
	full_text0
.
,store i64 %200, i64* %203, align 8, !tbaa !8
&i648B

	full_text


i64 %200
(i64*8B

	full_text

	i64* %203
_getelementptr8BL
J
	full_text=
;
9%204 = getelementptr inbounds double, double* %0, i64 %35
%i648B

	full_text
	
i64 %35
Cbitcast8B6
4
	full_text'
%
#%205 = bitcast double* %204 to i64*
.double*8B

	full_text

double* %204
Jload8B@
>
	full_text1
/
-%206 = load i64, i64* %205, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %205
6add8B-
+
	full_text

%207 = add nsw i64 %181, 4
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%208 = getelementptr inbounds double, double* %3, i64 %207
&i648B

	full_text


i64 %207
Cbitcast8B6
4
	full_text'
%
#%209 = bitcast double* %208 to i64*
.double*8B

	full_text

double* %208
Jstore8B?
=
	full_text0
.
,store i64 %206, i64* %209, align 8, !tbaa !8
&i648B

	full_text


i64 %206
(i64*8B

	full_text

	i64* %209
_getelementptr8BL
J
	full_text=
;
9%210 = getelementptr inbounds double, double* %0, i64 %38
%i648B

	full_text
	
i64 %38
Cbitcast8B6
4
	full_text'
%
#%211 = bitcast double* %210 to i64*
.double*8B

	full_text

double* %210
Jload8B@
>
	full_text1
/
-%212 = load i64, i64* %211, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %211
6add8B-
+
	full_text

%213 = add nsw i64 %181, 5
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%214 = getelementptr inbounds double, double* %3, i64 %213
&i648B

	full_text


i64 %213
Cbitcast8B6
4
	full_text'
%
#%215 = bitcast double* %214 to i64*
.double*8B

	full_text

double* %214
Jstore8B?
=
	full_text0
.
,store i64 %212, i64* %215, align 8, !tbaa !8
&i648B

	full_text


i64 %212
(i64*8B

	full_text

	i64* %215
_getelementptr8BL
J
	full_text=
;
9%216 = getelementptr inbounds double, double* %0, i64 %41
%i648B

	full_text
	
i64 %41
Cbitcast8B6
4
	full_text'
%
#%217 = bitcast double* %216 to i64*
.double*8B

	full_text

double* %216
Jload8B@
>
	full_text1
/
-%218 = load i64, i64* %217, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %217
6add8B-
+
	full_text

%219 = add nsw i64 %181, 6
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%220 = getelementptr inbounds double, double* %3, i64 %219
&i648B

	full_text


i64 %219
Cbitcast8B6
4
	full_text'
%
#%221 = bitcast double* %220 to i64*
.double*8B

	full_text

double* %220
Jstore8B?
=
	full_text0
.
,store i64 %218, i64* %221, align 8, !tbaa !8
&i648B

	full_text


i64 %218
(i64*8B

	full_text

	i64* %221
_getelementptr8BL
J
	full_text=
;
9%222 = getelementptr inbounds double, double* %0, i64 %44
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%223 = bitcast double* %222 to i64*
.double*8B

	full_text

double* %222
Jload8B@
>
	full_text1
/
-%224 = load i64, i64* %223, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %223
6add8B-
+
	full_text

%225 = add nsw i64 %181, 7
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%226 = getelementptr inbounds double, double* %3, i64 %225
&i648B

	full_text


i64 %225
Cbitcast8B6
4
	full_text'
%
#%227 = bitcast double* %226 to i64*
.double*8B

	full_text

double* %226
Jstore8B?
=
	full_text0
.
,store i64 %224, i64* %227, align 8, !tbaa !8
&i648B

	full_text


i64 %224
(i64*8B

	full_text

	i64* %227
_getelementptr8BL
J
	full_text=
;
9%228 = getelementptr inbounds double, double* %0, i64 %47
%i648B

	full_text
	
i64 %47
Cbitcast8B6
4
	full_text'
%
#%229 = bitcast double* %228 to i64*
.double*8B

	full_text

double* %228
Jload8B@
>
	full_text1
/
-%230 = load i64, i64* %229, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %229
6add8B-
+
	full_text

%231 = add nsw i64 %181, 8
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%232 = getelementptr inbounds double, double* %3, i64 %231
&i648B

	full_text


i64 %231
Cbitcast8B6
4
	full_text'
%
#%233 = bitcast double* %232 to i64*
.double*8B

	full_text

double* %232
Jstore8B?
=
	full_text0
.
,store i64 %230, i64* %233, align 8, !tbaa !8
&i648B

	full_text


i64 %230
(i64*8B

	full_text

	i64* %233
_getelementptr8BL
J
	full_text=
;
9%234 = getelementptr inbounds double, double* %0, i64 %50
%i648B

	full_text
	
i64 %50
Cbitcast8B6
4
	full_text'
%
#%235 = bitcast double* %234 to i64*
.double*8B

	full_text

double* %234
Jload8B@
>
	full_text1
/
-%236 = load i64, i64* %235, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %235
6add8B-
+
	full_text

%237 = add nsw i64 %181, 9
&i648B

	full_text


i64 %181
`getelementptr8BM
K
	full_text>
<
:%238 = getelementptr inbounds double, double* %3, i64 %237
&i648B

	full_text


i64 %237
Cbitcast8B6
4
	full_text'
%
#%239 = bitcast double* %238 to i64*
.double*8B

	full_text

double* %238
Jstore8B?
=
	full_text0
.
,store i64 %236, i64* %239, align 8, !tbaa !8
&i648B

	full_text


i64 %236
(i64*8B

	full_text

	i64* %239
Abitcast8B4
2
	full_text%
#
!%240 = bitcast double* %1 to i64*
Jload8B@
>
	full_text1
/
-%241 = load i64, i64* %240, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %240
3shl8B*
(
	full_text

%242 = shl i64 %179, 32
&i648B

	full_text


i64 %179
;ashr8B1
/
	full_text"
 
%243 = ashr exact i64 %242, 32
&i648B

	full_text


i64 %242
`getelementptr8BM
K
	full_text>
<
:%244 = getelementptr inbounds double, double* %4, i64 %243
&i648B

	full_text


i64 %243
Cbitcast8B6
4
	full_text'
%
#%245 = bitcast double* %244 to i64*
.double*8B

	full_text

double* %244
Jstore8B?
=
	full_text0
.
,store i64 %241, i64* %245, align 8, !tbaa !8
&i648B

	full_text


i64 %241
(i64*8B

	full_text

	i64* %245
Abitcast8B4
2
	full_text%
#
!%246 = bitcast double* %2 to i64*
Jload8B@
>
	full_text1
/
-%247 = load i64, i64* %246, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %246
`getelementptr8BM
K
	full_text>
<
:%248 = getelementptr inbounds double, double* %5, i64 %243
&i648B

	full_text


i64 %243
Cbitcast8B6
4
	full_text'
%
#%249 = bitcast double* %248 to i64*
.double*8B

	full_text

double* %248
Jstore8B?
=
	full_text0
.
,store i64 %247, i64* %249, align 8, !tbaa !8
&i648B

	full_text


i64 %247
(i64*8B

	full_text

	i64* %249
(br8B 

	full_text

br label %250
\call8BR
P
	full_textC
A
?call void @llvm.lifetime.end.p0i8(i64 2048, i8* nonnull %16) #6
%i8*8B

	full_text
	
i8* %16
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %15) #6
%i8*8B

	full_text
	
i8* %15
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %14) #6
%i8*8B

	full_text
	
i8* %14
Ycall8BO
M
	full_text@
>
<call void @llvm.lifetime.end.p0i8(i64 8, i8* nonnull %13) #6
%i8*8B

	full_text
	
i8* %13
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %4
,double*8B

	full_text


double* %2
*double8B

	full_text

	double %7
,double*8B

	full_text


double* %1
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %0
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %5
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
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
-; undefined function B

	full_text

 
#i648B

	full_text	

i64 1
'i328B

	full_text

	i32 65408
:double8B,
*
	full_text

double 0x41D2309CE5400000
4double8B&
$
	full_text

double 2.000000e+00
#i328B

	full_text	

i32 2
5double8B'
%
	full_text

double -2.000000e+00
#i648B

	full_text	

i64 3
4double8B&
$
	full_text

double 1.000000e+00
5double8B'
%
	full_text

double -1.000000e+00
#i648B

	full_text	

i64 8
$i648B

	full_text


i64 10
#i648B

	full_text	

i64 7
$i648B

	full_text


i64 32
%i648B

	full_text
	
i64 128
#i648B

	full_text	

i64 4
&i648B

	full_text


i64 2048
-i648B"
 
	full_text

i64 42949672960
$i648B

	full_text


i64 31
%i328B

	full_text
	
i32 100
#i328B

	full_text	

i32 1
:double8B,
*
	full_text

double 0x41B033C4D7000000
#i648B

	full_text	

i64 2
4double8B&
$
	full_text

double 0.000000e+00
#i328B

	full_text	

i32 0
#i648B

	full_text	

i64 5
#i648B

	full_text	

i64 9
%i328B

	full_text
	
i32 256
$i648B

	full_text


i64 30
$i648B

	full_text


i64 29
#i328B

	full_text	

i32 3
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 6
%i328B

	full_text
	
i32 128        	
 		                       !    "# "" $% $$ &' && () (* (( +, ++ -. -- /0 // 12 13 11 45 44 67 66 89 88 :; :< :: => == ?@ ?? AB AA CD CE CC FG FF HI HH JK JJ LM LN LL OP OO QR QQ ST SS UV UW UU XY XX Z[ ZZ \] \\ ^_ ^` ^^ ab aa cd cc ef ee gh gi gg jk jj lm ll no nn pq pr pp st ss uv uu wx ww yy z{ z| zz }~ }} 	€  
ƒ ‚‚ „… „
† „„ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹
 ‹‹  ‘  ’“ ’
” ’’ •— –– ˜™ ˜˜ š› š œœ Ÿ 
   ¡¢ ¡¡ £¤ ££ ¥¦ ¥¨ §§ ©ª ©© «¬ «« ­® ­
¯ ­­ °± °° ²
´ ³³ µ
¶ µµ ·
¸ ·· ¹
º ¹
» ¹¹ ¼
¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ ÌÌ ÎÏ ÎÎ ĞÑ Ğ
Ò ĞĞ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü Ú
İ ÚÚ Şß ŞŞ àá àã ââ äå ää æç æ
è ææ éê éé ëì ë
í ëë îï î
ğ îî ñò ñ
ó ññ ôõ ôô ö÷ öö øù ø
ú øø ûü û
ı û
ş ûû ÿ€ ÿÿ ‚  ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰‰ ‹
Œ ‹‹     ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— šœ ›
 ›› Ÿ 
   ¡¢ ¡¡ £¤ ££ ¥¦ ¥¨ §
© §§ ª« ªª ¬­ ¬¬ ®¯ ®± °° ²³ ²² ´
µ ´´ ¶· ¶
¸ ¶¶ ¹
º ¹¹ »¼ »
½ »» ¾¾ ¿À ¿¿ ÁÂ ÁÁ ÃÄ ÃÃ Å
Ç ÆÆ ÈÉ ÈË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ Ğ
Ñ ĞĞ ÒÓ ÒÒ ÔÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ ÛŞ İ
ß İİ àá àà â
ã ââ äå ää æç ææ èé è
ê èè ëì ë
í ëë îï ğñ ğğ òó òò ôõ ô÷ öö øù øø úû úı ü€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ
 ŒŒ   ‘  ’“ ’
” ’’ •– •
— •• ˜
™ ˜˜ š› šš œ œœ Ÿ 
   ¡¢ ¡
£ ¡¡ ¤¥ ¦§ ¦¦ ¨© ¨¨ ª« ª­ ¬¬ ®¯ ®° ±² ±± ³´ ³³ µµ ¶· ¶¶ ¸
¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿
À ¿¿ ÁÂ ÁÁ ÃÄ ÃÃ ÅÆ ÅÅ Ç
È ÇÇ ÉÊ ÉÉ ËÌ Ë
Í ËË Î
Ï ÎÎ ĞÑ ĞĞ ÒÓ ÒÒ ÔÕ ÔÔ Ö
× ÖÖ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İ
Ş İİ ßà ßß áâ áá ãä ãã å
æ åå çè çç éê é
ë éé ì
í ìì îï îî ğñ ğğ òó òò ô
õ ôô ö÷ öö øù ø
ú øø û
ü ûû ış ıı ÿ€ ÿÿ ‚  ƒ
„ ƒƒ …† …… ‡ˆ ‡
‰ ‡‡ Š
‹ ŠŠ Œ ŒŒ   ‘  ’
“ ’’ ”• ”” –— –
˜ –– ™
š ™™ ›œ ››   Ÿ  ŸŸ ¡
¢ ¡¡ £¤ ££ ¥¦ ¥
§ ¥¥ ¨
© ¨¨ ª« ªª ¬­ ¬¬ ®¯ ®® °
± °° ²³ ²² ´µ ´
¶ ´´ ·
¸ ·· ¹º ¹¹ »¼ »» ½¾ ½½ ¿
À ¿¿ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÆ ÇÈ ÇÇ ÉÊ ÉÉ ËÌ ËË Í
Î ÍÍ ÏĞ ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÔ ÕÖ ÕÕ ×
Ø ×× ÙÚ ÙÙ ÛÜ Û
İ ÛÛ Ş
à ßß á
â áá ã
ä ãã å
æ åå çè Íé ¹é ˜é Ôê ë ´ë Œë Æì ¸ì Çì Öì åì ôì ƒì ’ì ¡ì °ì ¿í $í +í 4í =í Fí Oí Xí aí jí sí ‹í Ğí âí µí ¿í Îí İí ìí ûí Ší ™í ¨í ·î yï ×   
	          !  #" %$ ' )" *( ,+ . 0/ 2" 31 54 7 98 ;" <: >= @ BA D" EC GF I KJ M" NL PO R TS V" WU YX [ ]\ _" `^ ba d fe h" ig kj m on q" rp ts v xy {w | ~ €¡ ƒz …‡ †„ ˆ‡ Š‰ Œ„ ‹  ‘ “ ”„ —– ™˜ ›  Ÿœ  ‚ ¢‚ ¤£ ¦ ¨§ ª ¬© ®« ¯ ± ´› ¶ª ¸ º° »¡ ¾³ À Áµ Ã› Ä½ Æ ÈÅ ÉÇ ËÊ ÍÅ Ï ÑÎ ÒĞ ÔÓ ÖÕ ØÕ ÙÌ ÛÌ Ü× İÚ ßŞ áÚ ãâ åä çÚ èæ êé ì íÌ ïé ğÕ òé óî õñ ÷ô ùö úø üî ıñ şû €ÿ ‚ „ …ƒ ‡ ˆ† Š‰ Œ‹   ’‹ “Â •î –¿ ˜ñ ™” œÂ — Ÿ¿  ½ ¢¡ ¤£ ¦Ú ¨ ©· «· ­¬ ¯ ±° ³² µ› ·´ ¸² º ¼¹ ½ À¿ ÂÁ Äö ÇÃ ÉÆ Ë ÌÊ Î" ÏÍ ÑÍ ÓÁ Öğ ×Õ Ù ÚØ ÜÕ ŞÒ ßİ áà ãâ åĞ çä éæ êè ìĞ íÕ ñğ óò õÆ ÷ö ùø ûÃ ı¦ €Á ÿ ƒ „‚ †ÿ ˆ ‰‡ ‹Š Œ ´ ‘ “ ”’ –´ —Š ™˜ ›¹ š Ÿœ   ¢¹ £ÿ §¦ ©¨ « ­¬ ¯° ²± ´µ ·³ ¹¸ »¶ ½º ¾ À¿ ÂÁ Ä³ ÆÅ ÈÇ ÊÃ ÌÉ Í/ ÏÎ ÑĞ Ó³ ÕÔ ×Ö ÙÒ ÛØ Ü8 Şİ àß â³ äã æå èá êç ëA íì ïî ñ³ óò õô ÷ğ ùö úJ üû şı €³ ‚ „ƒ †ÿ ˆ… ‰S ‹Š Œ ³ ‘ “’ • —” ˜\ š™ œ› ³  Ÿ ¢¡ ¤ ¦£ §e ©¨ «ª ­³ ¯® ±° ³¬ µ² ¶n ¸· º¹ ¼³ ¾½ À¿ Â» ÄÁ ÅÆ È° ÊÉ ÌË ÎÍ ĞÇ ÒÏ ÓÔ ÖË Ø× ÚÕ ÜÙ İ à â	 ä æ ‚ – š §š œ• –² ³¥ ‚¥ §¼ ½à ›à â¥ §¥ ½š ›® ³® °Å ÆÈ ÊÈ öÔ Õú üú ÆÛ İÛ ïü şü ¬î ïô öô Õş ÿ® °® ß… ‡… ¥Ş ß¤ ¥ª ¬ª ÿ öö øø ûû üü ÷÷ ğğ òò ùù úú ç óó ôô ññ õõ ğğ  ğğ Õ öö Õ¹ õõ ¹ ğğ â øø â ññ ’ ôô ’ÿ ùù ÿ¥ úú ¥ óó á üü áå üü åÌ öö Ìö ùù ö ğğ Ú öö Úï úú ïô ùù ôé ÷÷ é òò ° ûû °ß üü ßã üü ã¾ úú ¾ ôô 
ı Å
ı Î
ı ¡
ı ¿
ı ö
ı Å
ş ¬
ÿ ¹
€ Ì
€ Õ
 ‡
‚ ä	ƒ 8
ƒ ã
„ Ş
„ 
… Ì
… Õ† † † 
† ®† á† ã† å
‡ ø	ˆ \
ˆ Ÿ	‰ 	‰ 	‰  	‰ "
‰ °
‰ ²
‰ ³
‰ É
‰ Ë
Š £
‹ òŒ Œ ß
 ±	 /
 £    	 y ‚
 ‰
 –
 ¡ ¾ ï
 ğ ¥
 ¦‘ }
’ Ô“ &“ -“ 6“ ?“ H“ Q“ Z“ c“ l“ u“ ³“ µ” ” ” ” ·
” Ã
” ò
” ¨
” ¬” °	• J
• 	– n
– ½— ¹	˜ A	™ e
š ˜
› °
› °› ½
› Ç
› Ğ› Æ	œ S
œ 
 ª"
embar"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
_Z12get_local_idj"
_Z14get_local_sizej"
randlc"
vranlc"
llvm.fmuladd.f64"

_Z4sqrtd"	
_Z3logd"

_Z4fabsd"
_Z7barrierj"
_Z12get_group_idj"
llvm.lifetime.end.p0i8*ˆ
npb-EP-embar.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02~

wgsize_log1p
“A

transfer_bytes
€0
 
transfer_bytes_log1p
“A

devmap_label


wgsize
@