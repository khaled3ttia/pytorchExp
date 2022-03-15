

[external]
@allocaB6
4
	full_text'
%
#%12 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%13 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%14 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%15 = alloca [5 x double], align 16
@allocaB6
4
	full_text'
%
#%16 = alloca [5 x double], align 16
DbitcastB9
7
	full_text*
(
&%17 = bitcast [5 x double]* %12 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %12
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %17) #4
#i8*B

	full_text
	
i8* %17
DbitcastB9
7
	full_text*
(
&%18 = bitcast [5 x double]* %13 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %13
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %18) #4
#i8*B

	full_text
	
i8* %18
DbitcastB9
7
	full_text*
(
&%19 = bitcast [5 x double]* %14 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %14
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %19) #4
#i8*B

	full_text
	
i8* %19
DbitcastB9
7
	full_text*
(
&%20 = bitcast [5 x double]* %15 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %15
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %20) #4
#i8*B

	full_text
	
i8* %20
DbitcastB9
7
	full_text*
(
&%21 = bitcast [5 x double]* %16 to i8*
7[5 x double]*B$
"
	full_text

[5 x double]* %16
ZcallBR
P
	full_textC
A
?call void @llvm.lifetime.start.p0i8(i64 40, i8* nonnull %21) #4
#i8*B

	full_text
	
i8* %21
dcallB\
Z
	full_textM
K
Icall void @llvm.memset.p0i8.i64(i8* align 16 %21, i8 0, i64 40, i1 false)
#i8*B

	full_text
	
i8* %21
LcallBD
B
	full_text5
3
1%22 = tail call i64 @_Z13get_global_idj(i32 1) #5
.addB'
%
	full_text

%23 = add i64 %22, 1
#i64B

	full_text
	
i64 %22
6truncB-
+
	full_text

%24 = trunc i64 %23 to i32
#i64B

	full_text
	
i64 %23
LcallBD
B
	full_text5
3
1%25 = tail call i64 @_Z13get_global_idj(i32 0) #5
.addB'
%
	full_text

%26 = add i64 %25, 1
#i64B

	full_text
	
i64 %25
2addB+
)
	full_text

%27 = add nsw i32 %9, -2
6icmpB.
,
	full_text

%28 = icmp slt i32 %27, %24
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %24
:brB4
2
	full_text%
#
!br i1 %28, label %1057, label %29
!i1B

	full_text


i1 %28
8trunc8B-
+
	full_text

%30 = trunc i64 %26 to i32
%i648B

	full_text
	
i64 %26
4add8B+
)
	full_text

%31 = add nsw i32 %8, -2
8icmp8B.
,
	full_text

%32 = icmp slt i32 %31, %30
%i328B

	full_text
	
i32 %31
%i328B

	full_text
	
i32 %30
<br8B4
2
	full_text%
#
!br i1 %32, label %1057, label %33
#i18B

	full_text


i1 %32
Qbitcast8BD
B
	full_text5
3
1%34 = bitcast double* %1 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%35 = bitcast double* %2 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%36 = bitcast double* %3 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%37 = bitcast double* %4 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%38 = bitcast double* %5 to [13 x [13 x double]]*
Qbitcast8BD
B
	full_text5
3
1%39 = bitcast double* %6 to [13 x [13 x double]]*
Wbitcast8BJ
H
	full_text;
9
7%40 = bitcast double* %0 to [13 x [13 x [5 x double]]]*
1shl8B(
&
	full_text

%41 = shl i64 %23, 32
%i648B

	full_text
	
i64 %23
9ashr8B/
-
	full_text 

%42 = ashr exact i64 %41, 32
%i648B

	full_text
	
i64 %41
1shl8B(
&
	full_text

%43 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
9ashr8B/
-
	full_text 

%44 = ashr exact i64 %43, 32
%i648B

	full_text
	
i64 %43
™getelementptr8B…
‚
	full_textu
s
q%45 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 0, i64 %42, i64 %44
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Gbitcast8B:
8
	full_text+
)
'%46 = bitcast [5 x double]* %45 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %45
Hload8B>
<
	full_text/
-
+%47 = load i64, i64* %46, align 8, !tbaa !8
'i64*8B

	full_text


i64* %46
pgetelementptr8B]
[
	full_textN
L
J%48 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Gbitcast8B:
8
	full_text+
)
'%49 = bitcast [5 x double]* %12 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Istore8B>
<
	full_text/
-
+store i64 %47, i64* %49, align 16, !tbaa !8
%i648B

	full_text
	
i64 %47
'i64*8B

	full_text


i64* %49
 getelementptr8BŒ
‰
	full_text|
z
x%50 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 0, i64 %42, i64 %44, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%51 = bitcast double* %50 to i64*
-double*8B

	full_text

double* %50
Hload8B>
<
	full_text/
-
+%52 = load i64, i64* %51, align 8, !tbaa !8
'i64*8B

	full_text


i64* %51
pgetelementptr8B]
[
	full_textN
L
J%53 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%54 = bitcast double* %53 to i64*
-double*8B

	full_text

double* %53
Hstore8B=
;
	full_text.
,
*store i64 %52, i64* %54, align 8, !tbaa !8
%i648B

	full_text
	
i64 %52
'i64*8B

	full_text


i64* %54
 getelementptr8BŒ
‰
	full_text|
z
x%55 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 0, i64 %42, i64 %44, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%56 = bitcast double* %55 to i64*
-double*8B

	full_text

double* %55
Hload8B>
<
	full_text/
-
+%57 = load i64, i64* %56, align 8, !tbaa !8
'i64*8B

	full_text


i64* %56
pgetelementptr8B]
[
	full_textN
L
J%58 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%59 = bitcast double* %58 to i64*
-double*8B

	full_text

double* %58
Istore8B>
<
	full_text/
-
+store i64 %57, i64* %59, align 16, !tbaa !8
%i648B

	full_text
	
i64 %57
'i64*8B

	full_text


i64* %59
 getelementptr8BŒ
‰
	full_text|
z
x%60 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 0, i64 %42, i64 %44, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%61 = bitcast double* %60 to i64*
-double*8B

	full_text

double* %60
Hload8B>
<
	full_text/
-
+%62 = load i64, i64* %61, align 8, !tbaa !8
'i64*8B

	full_text


i64* %61
pgetelementptr8B]
[
	full_textN
L
J%63 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%64 = bitcast double* %63 to i64*
-double*8B

	full_text

double* %63
Hstore8B=
;
	full_text.
,
*store i64 %62, i64* %64, align 8, !tbaa !8
%i648B

	full_text
	
i64 %62
'i64*8B

	full_text


i64* %64
 getelementptr8BŒ
‰
	full_text|
z
x%65 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 0, i64 %42, i64 %44, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%66 = bitcast double* %65 to i64*
-double*8B

	full_text

double* %65
Hload8B>
<
	full_text/
-
+%67 = load i64, i64* %66, align 8, !tbaa !8
'i64*8B

	full_text


i64* %66
pgetelementptr8B]
[
	full_textN
L
J%68 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%69 = bitcast double* %68 to i64*
-double*8B

	full_text

double* %68
Istore8B>
<
	full_text/
-
+store i64 %67, i64* %69, align 16, !tbaa !8
%i648B

	full_text
	
i64 %67
'i64*8B

	full_text


i64* %69
^getelementptr8BK
I
	full_text<
:
8%70 = getelementptr inbounds double, double* %0, i64 845
Xbitcast8BK
I
	full_text<
:
8%71 = bitcast double* %70 to [13 x [13 x [5 x double]]]*
-double*8B

	full_text

double* %70
™getelementptr8B…
‚
	full_textu
s
q%72 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %71, i64 0, i64 %42, i64 %44
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Gbitcast8B:
8
	full_text+
)
'%73 = bitcast [5 x double]* %72 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %72
Hload8B>
<
	full_text/
-
+%74 = load i64, i64* %73, align 8, !tbaa !8
'i64*8B

	full_text


i64* %73
pgetelementptr8B]
[
	full_textN
L
J%75 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Gbitcast8B:
8
	full_text+
)
'%76 = bitcast [5 x double]* %13 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Istore8B>
<
	full_text/
-
+store i64 %74, i64* %76, align 16, !tbaa !8
%i648B

	full_text
	
i64 %74
'i64*8B

	full_text


i64* %76
 getelementptr8BŒ
‰
	full_text|
z
x%77 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %71, i64 0, i64 %42, i64 %44, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%78 = bitcast double* %77 to i64*
-double*8B

	full_text

double* %77
Hload8B>
<
	full_text/
-
+%79 = load i64, i64* %78, align 8, !tbaa !8
'i64*8B

	full_text


i64* %78
pgetelementptr8B]
[
	full_textN
L
J%80 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%81 = bitcast double* %80 to i64*
-double*8B

	full_text

double* %80
Hstore8B=
;
	full_text.
,
*store i64 %79, i64* %81, align 8, !tbaa !8
%i648B

	full_text
	
i64 %79
'i64*8B

	full_text


i64* %81
 getelementptr8BŒ
‰
	full_text|
z
x%82 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %71, i64 0, i64 %42, i64 %44, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%83 = bitcast double* %82 to i64*
-double*8B

	full_text

double* %82
Hload8B>
<
	full_text/
-
+%84 = load i64, i64* %83, align 8, !tbaa !8
'i64*8B

	full_text


i64* %83
pgetelementptr8B]
[
	full_textN
L
J%85 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%86 = bitcast double* %85 to i64*
-double*8B

	full_text

double* %85
Istore8B>
<
	full_text/
-
+store i64 %84, i64* %86, align 16, !tbaa !8
%i648B

	full_text
	
i64 %84
'i64*8B

	full_text


i64* %86
 getelementptr8BŒ
‰
	full_text|
z
x%87 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %71, i64 0, i64 %42, i64 %44, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%88 = bitcast double* %87 to i64*
-double*8B

	full_text

double* %87
Hload8B>
<
	full_text/
-
+%89 = load i64, i64* %88, align 8, !tbaa !8
'i64*8B

	full_text


i64* %88
pgetelementptr8B]
[
	full_textN
L
J%90 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%91 = bitcast double* %90 to i64*
-double*8B

	full_text

double* %90
Hstore8B=
;
	full_text.
,
*store i64 %89, i64* %91, align 8, !tbaa !8
%i648B

	full_text
	
i64 %89
'i64*8B

	full_text


i64* %91
 getelementptr8BŒ
‰
	full_text|
z
x%92 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %71, i64 0, i64 %42, i64 %44, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %71
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Abitcast8B4
2
	full_text%
#
!%93 = bitcast double* %92 to i64*
-double*8B

	full_text

double* %92
Hload8B>
<
	full_text/
-
+%94 = load i64, i64* %93, align 8, !tbaa !8
'i64*8B

	full_text


i64* %93
pgetelementptr8B]
[
	full_textN
L
J%95 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%96 = bitcast double* %95 to i64*
-double*8B

	full_text

double* %95
Istore8B>
<
	full_text/
-
+store i64 %94, i64* %96, align 16, !tbaa !8
%i648B

	full_text
	
i64 %94
'i64*8B

	full_text


i64* %96
_getelementptr8BL
J
	full_text=
;
9%97 = getelementptr inbounds double, double* %0, i64 1690
Xbitcast8BK
I
	full_text<
:
8%98 = bitcast double* %97 to [13 x [13 x [5 x double]]]*
-double*8B

	full_text

double* %97
™getelementptr8B…
‚
	full_textu
s
q%99 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %98, i64 0, i64 %42, i64 %44
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %98
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Hbitcast8B;
9
	full_text,
*
(%100 = bitcast [5 x double]* %99 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %99
Jload8B@
>
	full_text1
/
-%101 = load i64, i64* %100, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %100
Hbitcast8B;
9
	full_text,
*
(%102 = bitcast [5 x double]* %14 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Kstore8B@
>
	full_text1
/
-store i64 %101, i64* %102, align 16, !tbaa !8
&i648B

	full_text


i64 %101
(i64*8B

	full_text

	i64* %102
¡getelementptr8B
Š
	full_text}
{
y%103 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %98, i64 0, i64 %42, i64 %44, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %98
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%104 = bitcast double* %103 to i64*
.double*8B

	full_text

double* %103
Jload8B@
>
	full_text1
/
-%105 = load i64, i64* %104, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %104
qgetelementptr8B^
\
	full_textO
M
K%106 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%107 = bitcast double* %106 to i64*
.double*8B

	full_text

double* %106
Jstore8B?
=
	full_text0
.
,store i64 %105, i64* %107, align 8, !tbaa !8
&i648B

	full_text


i64 %105
(i64*8B

	full_text

	i64* %107
¡getelementptr8B
Š
	full_text}
{
y%108 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %98, i64 0, i64 %42, i64 %44, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %98
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%109 = bitcast double* %108 to i64*
.double*8B

	full_text

double* %108
Jload8B@
>
	full_text1
/
-%110 = load i64, i64* %109, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %109
qgetelementptr8B^
\
	full_textO
M
K%111 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%112 = bitcast double* %111 to i64*
.double*8B

	full_text

double* %111
Kstore8B@
>
	full_text1
/
-store i64 %110, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %110
(i64*8B

	full_text

	i64* %112
¡getelementptr8B
Š
	full_text}
{
y%113 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %98, i64 0, i64 %42, i64 %44, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %98
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%114 = bitcast double* %113 to i64*
.double*8B

	full_text

double* %113
Jload8B@
>
	full_text1
/
-%115 = load i64, i64* %114, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %114
qgetelementptr8B^
\
	full_textO
M
K%116 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%117 = bitcast double* %116 to i64*
.double*8B

	full_text

double* %116
Jstore8B?
=
	full_text0
.
,store i64 %115, i64* %117, align 8, !tbaa !8
&i648B

	full_text


i64 %115
(i64*8B

	full_text

	i64* %117
¡getelementptr8B
Š
	full_text}
{
y%118 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %98, i64 0, i64 %42, i64 %44, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %98
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%119 = bitcast double* %118 to i64*
.double*8B

	full_text

double* %118
Jload8B@
>
	full_text1
/
-%120 = load i64, i64* %119, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %119
qgetelementptr8B^
\
	full_textO
M
K%121 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%122 = bitcast double* %121 to i64*
.double*8B

	full_text

double* %121
Kstore8B@
>
	full_text1
/
-store i64 %120, i64* %122, align 16, !tbaa !8
&i648B

	full_text


i64 %120
(i64*8B

	full_text

	i64* %122
Œgetelementptr8By
w
	full_textj
h
f%123 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %34, i64 0, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %34
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%124 = load double, double* %123, align 8, !tbaa !8
.double*8B

	full_text

double* %123
_getelementptr8BL
J
	full_text=
;
9%125 = getelementptr inbounds double, double* %1, i64 169
Tbitcast8BG
E
	full_text8
6
4%126 = bitcast double* %125 to [13 x [13 x double]]*
.double*8B

	full_text

double* %125
getelementptr8Bz
x
	full_textk
i
g%127 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %126, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %126
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%128 = load double, double* %127, align 8, !tbaa !8
.double*8B

	full_text

double* %127
Œgetelementptr8By
w
	full_textj
h
f%129 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %35, i64 0, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %35
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%130 = load double, double* %129, align 8, !tbaa !8
.double*8B

	full_text

double* %129
_getelementptr8BL
J
	full_text=
;
9%131 = getelementptr inbounds double, double* %2, i64 169
Tbitcast8BG
E
	full_text8
6
4%132 = bitcast double* %131 to [13 x [13 x double]]*
.double*8B

	full_text

double* %131
getelementptr8Bz
x
	full_textk
i
g%133 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %132, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %132
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%134 = load double, double* %133, align 8, !tbaa !8
.double*8B

	full_text

double* %133
Œgetelementptr8By
w
	full_textj
h
f%135 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %36, i64 0, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %36
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%136 = load double, double* %135, align 8, !tbaa !8
.double*8B

	full_text

double* %135
_getelementptr8BL
J
	full_text=
;
9%137 = getelementptr inbounds double, double* %3, i64 169
Tbitcast8BG
E
	full_text8
6
4%138 = bitcast double* %137 to [13 x [13 x double]]*
.double*8B

	full_text

double* %137
getelementptr8Bz
x
	full_textk
i
g%139 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %138, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %138
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%140 = load double, double* %139, align 8, !tbaa !8
.double*8B

	full_text

double* %139
Œgetelementptr8By
w
	full_textj
h
f%141 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %37, i64 0, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %37
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%142 = load double, double* %141, align 8, !tbaa !8
.double*8B

	full_text

double* %141
_getelementptr8BL
J
	full_text=
;
9%143 = getelementptr inbounds double, double* %4, i64 169
Tbitcast8BG
E
	full_text8
6
4%144 = bitcast double* %143 to [13 x [13 x double]]*
.double*8B

	full_text

double* %143
getelementptr8Bz
x
	full_textk
i
g%145 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %144, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %144
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%146 = load double, double* %145, align 8, !tbaa !8
.double*8B

	full_text

double* %145
Œgetelementptr8By
w
	full_textj
h
f%147 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %38, i64 0, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %38
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%148 = load double, double* %147, align 8, !tbaa !8
.double*8B

	full_text

double* %147
_getelementptr8BL
J
	full_text=
;
9%149 = getelementptr inbounds double, double* %5, i64 169
Tbitcast8BG
E
	full_text8
6
4%150 = bitcast double* %149 to [13 x [13 x double]]*
.double*8B

	full_text

double* %149
getelementptr8Bz
x
	full_textk
i
g%151 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %150, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %150
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%152 = load double, double* %151, align 8, !tbaa !8
.double*8B

	full_text

double* %151
Œgetelementptr8By
w
	full_textj
h
f%153 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %39, i64 0, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %39
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%154 = load double, double* %153, align 8, !tbaa !8
.double*8B

	full_text

double* %153
_getelementptr8BL
J
	full_text=
;
9%155 = getelementptr inbounds double, double* %6, i64 169
Tbitcast8BG
E
	full_text8
6
4%156 = bitcast double* %155 to [13 x [13 x double]]*
.double*8B

	full_text

double* %155
getelementptr8Bz
x
	full_textk
i
g%157 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %156, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %156
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%158 = load double, double* %157, align 8, !tbaa !8
.double*8B

	full_text

double* %157
qgetelementptr8B^
\
	full_textO
M
K%159 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Hbitcast8B;
9
	full_text,
*
(%160 = bitcast [5 x double]* %16 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Kload8BA
?
	full_text2
0
.%161 = load i64, i64* %160, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %160
Hbitcast8B;
9
	full_text,
*
(%162 = bitcast [5 x double]* %15 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Kstore8B@
>
	full_text1
/
-store i64 %161, i64* %162, align 16, !tbaa !8
&i648B

	full_text


i64 %161
(i64*8B

	full_text

	i64* %162
qgetelementptr8B^
\
	full_textO
M
K%163 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%164 = bitcast double* %163 to i64*
.double*8B

	full_text

double* %163
Jload8B@
>
	full_text1
/
-%165 = load i64, i64* %164, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %164
qgetelementptr8B^
\
	full_textO
M
K%166 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%167 = bitcast double* %166 to i64*
.double*8B

	full_text

double* %166
Jstore8B?
=
	full_text0
.
,store i64 %165, i64* %167, align 8, !tbaa !8
&i648B

	full_text


i64 %165
(i64*8B

	full_text

	i64* %167
qgetelementptr8B^
\
	full_textO
M
K%168 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%169 = bitcast double* %168 to i64*
.double*8B

	full_text

double* %168
Kload8BA
?
	full_text2
0
.%170 = load i64, i64* %169, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %169
qgetelementptr8B^
\
	full_textO
M
K%171 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%172 = bitcast double* %171 to i64*
.double*8B

	full_text

double* %171
Kstore8B@
>
	full_text1
/
-store i64 %170, i64* %172, align 16, !tbaa !8
&i648B

	full_text


i64 %170
(i64*8B

	full_text

	i64* %172
qgetelementptr8B^
\
	full_textO
M
K%173 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%174 = bitcast double* %173 to i64*
.double*8B

	full_text

double* %173
Jload8B@
>
	full_text1
/
-%175 = load i64, i64* %174, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %174
qgetelementptr8B^
\
	full_textO
M
K%176 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%177 = bitcast double* %176 to i64*
.double*8B

	full_text

double* %176
Jstore8B?
=
	full_text0
.
,store i64 %175, i64* %177, align 8, !tbaa !8
&i648B

	full_text


i64 %175
(i64*8B

	full_text

	i64* %177
qgetelementptr8B^
\
	full_textO
M
K%178 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%179 = bitcast double* %178 to i64*
.double*8B

	full_text

double* %178
Kload8BA
?
	full_text2
0
.%180 = load i64, i64* %179, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %179
qgetelementptr8B^
\
	full_textO
M
K%181 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%182 = bitcast double* %181 to i64*
.double*8B

	full_text

double* %181
Kstore8B@
>
	full_text1
/
-store i64 %180, i64* %182, align 16, !tbaa !8
&i648B

	full_text


i64 %180
(i64*8B

	full_text

	i64* %182
Jstore8B?
=
	full_text0
.
,store i64 %47, i64* %160, align 16, !tbaa !8
%i648B

	full_text
	
i64 %47
(i64*8B

	full_text

	i64* %160
Istore8B>
<
	full_text/
-
+store i64 %52, i64* %164, align 8, !tbaa !8
%i648B

	full_text
	
i64 %52
(i64*8B

	full_text

	i64* %164
Jstore8B?
=
	full_text0
.
,store i64 %57, i64* %169, align 16, !tbaa !8
%i648B

	full_text
	
i64 %57
(i64*8B

	full_text

	i64* %169
Istore8B>
<
	full_text/
-
+store i64 %62, i64* %174, align 8, !tbaa !8
%i648B

	full_text
	
i64 %62
(i64*8B

	full_text

	i64* %174
Jstore8B?
=
	full_text0
.
,store i64 %67, i64* %179, align 16, !tbaa !8
%i648B

	full_text
	
i64 %67
(i64*8B

	full_text

	i64* %179
Istore8B>
<
	full_text/
-
+store i64 %74, i64* %49, align 16, !tbaa !8
%i648B

	full_text
	
i64 %74
'i64*8B

	full_text


i64* %49
Hstore8B=
;
	full_text.
,
*store i64 %79, i64* %54, align 8, !tbaa !8
%i648B

	full_text
	
i64 %79
'i64*8B

	full_text


i64* %54
Istore8B>
<
	full_text/
-
+store i64 %84, i64* %59, align 16, !tbaa !8
%i648B

	full_text
	
i64 %84
'i64*8B

	full_text


i64* %59
Hstore8B=
;
	full_text.
,
*store i64 %89, i64* %64, align 8, !tbaa !8
%i648B

	full_text
	
i64 %89
'i64*8B

	full_text


i64* %64
Istore8B>
<
	full_text/
-
+store i64 %94, i64* %69, align 16, !tbaa !8
%i648B

	full_text
	
i64 %94
'i64*8B

	full_text


i64* %69
Jstore8B?
=
	full_text0
.
,store i64 %101, i64* %76, align 16, !tbaa !8
&i648B

	full_text


i64 %101
'i64*8B

	full_text


i64* %76
Istore8B>
<
	full_text/
-
+store i64 %105, i64* %81, align 8, !tbaa !8
&i648B

	full_text


i64 %105
'i64*8B

	full_text


i64* %81
Jstore8B?
=
	full_text0
.
,store i64 %110, i64* %86, align 16, !tbaa !8
&i648B

	full_text


i64 %110
'i64*8B

	full_text


i64* %86
Istore8B>
<
	full_text/
-
+store i64 %115, i64* %91, align 8, !tbaa !8
&i648B

	full_text


i64 %115
'i64*8B

	full_text


i64* %91
Jstore8B?
=
	full_text0
.
,store i64 %120, i64* %96, align 16, !tbaa !8
&i648B

	full_text


i64 %120
'i64*8B

	full_text


i64* %96
`getelementptr8BM
K
	full_text>
<
:%183 = getelementptr inbounds double, double* %0, i64 2535
Zbitcast8BM
K
	full_text>
<
:%184 = bitcast double* %183 to [13 x [13 x [5 x double]]]*
.double*8B

	full_text

double* %183
›getelementptr8B‡
„
	full_textw
u
s%185 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %184, i64 0, i64 %42, i64 %44
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %184
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Ibitcast8B<
:
	full_text-
+
)%186 = bitcast [5 x double]* %185 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %185
Jload8B@
>
	full_text1
/
-%187 = load i64, i64* %186, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %186
Kstore8B@
>
	full_text1
/
-store i64 %187, i64* %102, align 16, !tbaa !8
&i648B

	full_text


i64 %187
(i64*8B

	full_text

	i64* %102
¢getelementptr8BŽ
‹
	full_text~
|
z%188 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %184, i64 0, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %184
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%189 = bitcast double* %188 to i64*
.double*8B

	full_text

double* %188
Jload8B@
>
	full_text1
/
-%190 = load i64, i64* %189, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %189
Jstore8B?
=
	full_text0
.
,store i64 %190, i64* %107, align 8, !tbaa !8
&i648B

	full_text


i64 %190
(i64*8B

	full_text

	i64* %107
¢getelementptr8BŽ
‹
	full_text~
|
z%191 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %184, i64 0, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %184
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%192 = bitcast double* %191 to i64*
.double*8B

	full_text

double* %191
Jload8B@
>
	full_text1
/
-%193 = load i64, i64* %192, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %192
Kstore8B@
>
	full_text1
/
-store i64 %193, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %193
(i64*8B

	full_text

	i64* %112
¢getelementptr8BŽ
‹
	full_text~
|
z%194 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %184, i64 0, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %184
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%195 = bitcast double* %194 to i64*
.double*8B

	full_text

double* %194
Jload8B@
>
	full_text1
/
-%196 = load i64, i64* %195, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %195
Jstore8B?
=
	full_text0
.
,store i64 %196, i64* %117, align 8, !tbaa !8
&i648B

	full_text


i64 %196
(i64*8B

	full_text

	i64* %117
¢getelementptr8BŽ
‹
	full_text~
|
z%197 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %184, i64 0, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %184
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%198 = bitcast double* %197 to i64*
.double*8B

	full_text

double* %197
Jload8B@
>
	full_text1
/
-%199 = load i64, i64* %198, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %198
Kstore8B@
>
	full_text1
/
-store i64 %199, i64* %122, align 16, !tbaa !8
&i648B

	full_text


i64 %199
(i64*8B

	full_text

	i64* %122
_getelementptr8BL
J
	full_text=
;
9%200 = getelementptr inbounds double, double* %1, i64 338
Tbitcast8BG
E
	full_text8
6
4%201 = bitcast double* %200 to [13 x [13 x double]]*
.double*8B

	full_text

double* %200
getelementptr8Bz
x
	full_textk
i
g%202 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %201, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %201
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%203 = load double, double* %202, align 8, !tbaa !8
.double*8B

	full_text

double* %202
_getelementptr8BL
J
	full_text=
;
9%204 = getelementptr inbounds double, double* %2, i64 338
Tbitcast8BG
E
	full_text8
6
4%205 = bitcast double* %204 to [13 x [13 x double]]*
.double*8B

	full_text

double* %204
getelementptr8Bz
x
	full_textk
i
g%206 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %205, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %205
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%207 = load double, double* %206, align 8, !tbaa !8
.double*8B

	full_text

double* %206
_getelementptr8BL
J
	full_text=
;
9%208 = getelementptr inbounds double, double* %3, i64 338
Tbitcast8BG
E
	full_text8
6
4%209 = bitcast double* %208 to [13 x [13 x double]]*
.double*8B

	full_text

double* %208
getelementptr8Bz
x
	full_textk
i
g%210 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %209, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %209
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%211 = load double, double* %210, align 8, !tbaa !8
.double*8B

	full_text

double* %210
_getelementptr8BL
J
	full_text=
;
9%212 = getelementptr inbounds double, double* %4, i64 338
Tbitcast8BG
E
	full_text8
6
4%213 = bitcast double* %212 to [13 x [13 x double]]*
.double*8B

	full_text

double* %212
getelementptr8Bz
x
	full_textk
i
g%214 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %213, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %213
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%215 = load double, double* %214, align 8, !tbaa !8
.double*8B

	full_text

double* %214
_getelementptr8BL
J
	full_text=
;
9%216 = getelementptr inbounds double, double* %5, i64 338
Tbitcast8BG
E
	full_text8
6
4%217 = bitcast double* %216 to [13 x [13 x double]]*
.double*8B

	full_text

double* %216
getelementptr8Bz
x
	full_textk
i
g%218 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %217, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %217
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%219 = load double, double* %218, align 8, !tbaa !8
.double*8B

	full_text

double* %218
_getelementptr8BL
J
	full_text=
;
9%220 = getelementptr inbounds double, double* %6, i64 338
Tbitcast8BG
E
	full_text8
6
4%221 = bitcast double* %220 to [13 x [13 x double]]*
.double*8B

	full_text

double* %220
getelementptr8Bz
x
	full_textk
i
g%222 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %221, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %221
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%223 = load double, double* %222, align 8, !tbaa !8
.double*8B

	full_text

double* %222
_getelementptr8BL
J
	full_text=
;
9%224 = getelementptr inbounds double, double* %7, i64 845
Zbitcast8BM
K
	full_text>
<
:%225 = bitcast double* %224 to [13 x [13 x [5 x double]]]*
.double*8B

	full_text

double* %224
¢getelementptr8BŽ
‹
	full_text~
|
z%226 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %225, i64 0, i64 %42, i64 %44, i64 0
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %225
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%227 = load double, double* %226, align 8, !tbaa !8
.double*8B

	full_text

double* %226
Abitcast8B4
2
	full_text%
#
!%228 = bitcast i64 %101 to double
&i648B

	full_text


i64 %101
@bitcast8B3
1
	full_text$
"
 %229 = bitcast i64 %74 to double
%i648B

	full_text
	
i64 %74
vcall8Bl
j
	full_text]
[
Y%230 = tail call double @llvm.fmuladd.f64(double %229, double -2.000000e+00, double %228)
,double8B

	full_text

double %229
,double8B

	full_text

double %228
@bitcast8B3
1
	full_text$
"
 %231 = bitcast i64 %47 to double
%i648B

	full_text
	
i64 %47
:fadd8B0
.
	full_text!

%232 = fadd double %230, %231
,double8B

	full_text

double %230
,double8B

	full_text

double %231
ucall8Bk
i
	full_text\
Z
X%233 = tail call double @llvm.fmuladd.f64(double %232, double 1.210000e+02, double %227)
,double8B

	full_text

double %232
,double8B

	full_text

double %227
Abitcast8B4
2
	full_text%
#
!%234 = bitcast i64 %115 to double
&i648B

	full_text


i64 %115
@bitcast8B3
1
	full_text$
"
 %235 = bitcast i64 %62 to double
%i648B

	full_text
	
i64 %62
:fsub8B0
.
	full_text!

%236 = fsub double %234, %235
,double8B

	full_text

double %234
,double8B

	full_text

double %235
vcall8Bl
j
	full_text]
[
Y%237 = tail call double @llvm.fmuladd.f64(double %236, double -5.500000e+00, double %233)
,double8B

	full_text

double %236
,double8B

	full_text

double %233
¢getelementptr8BŽ
‹
	full_text~
|
z%238 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %225, i64 0, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %225
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%239 = load double, double* %238, align 8, !tbaa !8
.double*8B

	full_text

double* %238
Abitcast8B4
2
	full_text%
#
!%240 = bitcast i64 %105 to double
&i648B

	full_text


i64 %105
@bitcast8B3
1
	full_text$
"
 %241 = bitcast i64 %79 to double
%i648B

	full_text
	
i64 %79
vcall8Bl
j
	full_text]
[
Y%242 = tail call double @llvm.fmuladd.f64(double %241, double -2.000000e+00, double %240)
,double8B

	full_text

double %241
,double8B

	full_text

double %240
@bitcast8B3
1
	full_text$
"
 %243 = bitcast i64 %52 to double
%i648B

	full_text
	
i64 %52
:fadd8B0
.
	full_text!

%244 = fadd double %242, %243
,double8B

	full_text

double %242
,double8B

	full_text

double %243
ucall8Bk
i
	full_text\
Z
X%245 = tail call double @llvm.fmuladd.f64(double %244, double 1.210000e+02, double %239)
,double8B

	full_text

double %244
,double8B

	full_text

double %239
vcall8Bl
j
	full_text]
[
Y%246 = tail call double @llvm.fmuladd.f64(double %128, double -2.000000e+00, double %203)
,double8B

	full_text

double %128
,double8B

	full_text

double %203
:fadd8B0
.
	full_text!

%247 = fadd double %124, %246
,double8B

	full_text

double %124
,double8B

	full_text

double %246
{call8Bq
o
	full_textb
`
^%248 = tail call double @llvm.fmuladd.f64(double %247, double 0x4028333333333334, double %245)
,double8B

	full_text

double %247
,double8B

	full_text

double %245
:fmul8B0
.
	full_text!

%249 = fmul double %136, %243
,double8B

	full_text

double %136
,double8B

	full_text

double %243
Cfsub8B9
7
	full_text*
(
&%250 = fsub double -0.000000e+00, %249
,double8B

	full_text

double %249
mcall8Bc
a
	full_textT
R
P%251 = tail call double @llvm.fmuladd.f64(double %240, double %211, double %250)
,double8B

	full_text

double %240
,double8B

	full_text

double %211
,double8B

	full_text

double %250
vcall8Bl
j
	full_text]
[
Y%252 = tail call double @llvm.fmuladd.f64(double %251, double -5.500000e+00, double %248)
,double8B

	full_text

double %251
,double8B

	full_text

double %248
¢getelementptr8BŽ
‹
	full_text~
|
z%253 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %225, i64 0, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %225
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%254 = load double, double* %253, align 8, !tbaa !8
.double*8B

	full_text

double* %253
Abitcast8B4
2
	full_text%
#
!%255 = bitcast i64 %110 to double
&i648B

	full_text


i64 %110
@bitcast8B3
1
	full_text$
"
 %256 = bitcast i64 %84 to double
%i648B

	full_text
	
i64 %84
vcall8Bl
j
	full_text]
[
Y%257 = tail call double @llvm.fmuladd.f64(double %256, double -2.000000e+00, double %255)
,double8B

	full_text

double %256
,double8B

	full_text

double %255
@bitcast8B3
1
	full_text$
"
 %258 = bitcast i64 %57 to double
%i648B

	full_text
	
i64 %57
:fadd8B0
.
	full_text!

%259 = fadd double %257, %258
,double8B

	full_text

double %257
,double8B

	full_text

double %258
ucall8Bk
i
	full_text\
Z
X%260 = tail call double @llvm.fmuladd.f64(double %259, double 1.210000e+02, double %254)
,double8B

	full_text

double %259
,double8B

	full_text

double %254
vcall8Bl
j
	full_text]
[
Y%261 = tail call double @llvm.fmuladd.f64(double %134, double -2.000000e+00, double %207)
,double8B

	full_text

double %134
,double8B

	full_text

double %207
:fadd8B0
.
	full_text!

%262 = fadd double %130, %261
,double8B

	full_text

double %130
,double8B

	full_text

double %261
{call8Bq
o
	full_textb
`
^%263 = tail call double @llvm.fmuladd.f64(double %262, double 0x4028333333333334, double %260)
,double8B

	full_text

double %262
,double8B

	full_text

double %260
:fmul8B0
.
	full_text!

%264 = fmul double %136, %258
,double8B

	full_text

double %136
,double8B

	full_text

double %258
Cfsub8B9
7
	full_text*
(
&%265 = fsub double -0.000000e+00, %264
,double8B

	full_text

double %264
mcall8Bc
a
	full_textT
R
P%266 = tail call double @llvm.fmuladd.f64(double %255, double %211, double %265)
,double8B

	full_text

double %255
,double8B

	full_text

double %211
,double8B

	full_text

double %265
vcall8Bl
j
	full_text]
[
Y%267 = tail call double @llvm.fmuladd.f64(double %266, double -5.500000e+00, double %263)
,double8B

	full_text

double %266
,double8B

	full_text

double %263
¢getelementptr8BŽ
‹
	full_text~
|
z%268 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %225, i64 0, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %225
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%269 = load double, double* %268, align 8, !tbaa !8
.double*8B

	full_text

double* %268
Oload8BE
C
	full_text6
4
2%270 = load double, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
vcall8Bl
j
	full_text]
[
Y%271 = tail call double @llvm.fmuladd.f64(double %270, double -2.000000e+00, double %234)
,double8B

	full_text

double %270
,double8B

	full_text

double %234
:fadd8B0
.
	full_text!

%272 = fadd double %271, %235
,double8B

	full_text

double %271
,double8B

	full_text

double %235
ucall8Bk
i
	full_text\
Z
X%273 = tail call double @llvm.fmuladd.f64(double %272, double 1.210000e+02, double %269)
,double8B

	full_text

double %272
,double8B

	full_text

double %269
vcall8Bl
j
	full_text]
[
Y%274 = tail call double @llvm.fmuladd.f64(double %140, double -2.000000e+00, double %211)
,double8B

	full_text

double %140
,double8B

	full_text

double %211
:fadd8B0
.
	full_text!

%275 = fadd double %136, %274
,double8B

	full_text

double %136
,double8B

	full_text

double %274
{call8Bq
o
	full_textb
`
^%276 = tail call double @llvm.fmuladd.f64(double %275, double 0x4030222222222222, double %273)
,double8B

	full_text

double %275
,double8B

	full_text

double %273
:fmul8B0
.
	full_text!

%277 = fmul double %136, %235
,double8B

	full_text

double %136
,double8B

	full_text

double %235
Cfsub8B9
7
	full_text*
(
&%278 = fsub double -0.000000e+00, %277
,double8B

	full_text

double %277
mcall8Bc
a
	full_textT
R
P%279 = tail call double @llvm.fmuladd.f64(double %234, double %211, double %278)
,double8B

	full_text

double %234
,double8B

	full_text

double %211
,double8B

	full_text

double %278
Pload8BF
D
	full_text7
5
3%280 = load double, double* %95, align 16, !tbaa !8
-double*8B

	full_text

double* %95
:fsub8B0
.
	full_text!

%281 = fsub double %280, %223
,double8B

	full_text

double %280
,double8B

	full_text

double %223
Qload8BG
E
	full_text8
6
4%282 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fsub8B0
.
	full_text!

%283 = fsub double %281, %282
,double8B

	full_text

double %281
,double8B

	full_text

double %282
:fadd8B0
.
	full_text!

%284 = fadd double %154, %283
,double8B

	full_text

double %154
,double8B

	full_text

double %283
ucall8Bk
i
	full_text\
Z
X%285 = tail call double @llvm.fmuladd.f64(double %284, double 4.000000e-01, double %279)
,double8B

	full_text

double %284
,double8B

	full_text

double %279
vcall8Bl
j
	full_text]
[
Y%286 = tail call double @llvm.fmuladd.f64(double %285, double -5.500000e+00, double %276)
,double8B

	full_text

double %285
,double8B

	full_text

double %276
¢getelementptr8BŽ
‹
	full_text~
|
z%287 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %225, i64 0, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %225
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%288 = load double, double* %287, align 8, !tbaa !8
.double*8B

	full_text

double* %287
Pload8BF
D
	full_text7
5
3%289 = load double, double* %68, align 16, !tbaa !8
-double*8B

	full_text

double* %68
vcall8Bl
j
	full_text]
[
Y%290 = tail call double @llvm.fmuladd.f64(double %289, double -2.000000e+00, double %280)
,double8B

	full_text

double %289
,double8B

	full_text

double %280
:fadd8B0
.
	full_text!

%291 = fadd double %282, %290
,double8B

	full_text

double %282
,double8B

	full_text

double %290
ucall8Bk
i
	full_text\
Z
X%292 = tail call double @llvm.fmuladd.f64(double %291, double 1.210000e+02, double %288)
,double8B

	full_text

double %291
,double8B

	full_text

double %288
vcall8Bl
j
	full_text]
[
Y%293 = tail call double @llvm.fmuladd.f64(double %146, double -2.000000e+00, double %215)
,double8B

	full_text

double %146
,double8B

	full_text

double %215
:fadd8B0
.
	full_text!

%294 = fadd double %142, %293
,double8B

	full_text

double %142
,double8B

	full_text

double %293
{call8Bq
o
	full_textb
`
^%295 = tail call double @llvm.fmuladd.f64(double %294, double 0xC0273B645A1CAC07, double %292)
,double8B

	full_text

double %294
,double8B

	full_text

double %292
Bfmul8B8
6
	full_text)
'
%%296 = fmul double %140, 2.000000e+00
,double8B

	full_text

double %140
:fmul8B0
.
	full_text!

%297 = fmul double %140, %296
,double8B

	full_text

double %140
,double8B

	full_text

double %296
Cfsub8B9
7
	full_text*
(
&%298 = fsub double -0.000000e+00, %297
,double8B

	full_text

double %297
mcall8Bc
a
	full_textT
R
P%299 = tail call double @llvm.fmuladd.f64(double %211, double %211, double %298)
,double8B

	full_text

double %211
,double8B

	full_text

double %211
,double8B

	full_text

double %298
mcall8Bc
a
	full_textT
R
P%300 = tail call double @llvm.fmuladd.f64(double %136, double %136, double %299)
,double8B

	full_text

double %136
,double8B

	full_text

double %136
,double8B

	full_text

double %299
{call8Bq
o
	full_textb
`
^%301 = tail call double @llvm.fmuladd.f64(double %300, double 0x4000222222222222, double %295)
,double8B

	full_text

double %300
,double8B

	full_text

double %295
Bfmul8B8
6
	full_text)
'
%%302 = fmul double %289, 2.000000e+00
,double8B

	full_text

double %289
:fmul8B0
.
	full_text!

%303 = fmul double %152, %302
,double8B

	full_text

double %152
,double8B

	full_text

double %302
Cfsub8B9
7
	full_text*
(
&%304 = fsub double -0.000000e+00, %303
,double8B

	full_text

double %303
mcall8Bc
a
	full_textT
R
P%305 = tail call double @llvm.fmuladd.f64(double %280, double %219, double %304)
,double8B

	full_text

double %280
,double8B

	full_text

double %219
,double8B

	full_text

double %304
mcall8Bc
a
	full_textT
R
P%306 = tail call double @llvm.fmuladd.f64(double %282, double %148, double %305)
,double8B

	full_text

double %282
,double8B

	full_text

double %148
,double8B

	full_text

double %305
{call8Bq
o
	full_textb
`
^%307 = tail call double @llvm.fmuladd.f64(double %306, double 0x4037B74BC6A7EF9D, double %301)
,double8B

	full_text

double %306
,double8B

	full_text

double %301
Bfmul8B8
6
	full_text)
'
%%308 = fmul double %223, 4.000000e-01
,double8B

	full_text

double %223
Cfsub8B9
7
	full_text*
(
&%309 = fsub double -0.000000e+00, %308
,double8B

	full_text

double %308
ucall8Bk
i
	full_text\
Z
X%310 = tail call double @llvm.fmuladd.f64(double %280, double 1.400000e+00, double %309)
,double8B

	full_text

double %280
,double8B

	full_text

double %309
Bfmul8B8
6
	full_text)
'
%%311 = fmul double %154, 4.000000e-01
,double8B

	full_text

double %154
Cfsub8B9
7
	full_text*
(
&%312 = fsub double -0.000000e+00, %311
,double8B

	full_text

double %311
ucall8Bk
i
	full_text\
Z
X%313 = tail call double @llvm.fmuladd.f64(double %282, double 1.400000e+00, double %312)
,double8B

	full_text

double %282
,double8B

	full_text

double %312
:fmul8B0
.
	full_text!

%314 = fmul double %136, %313
,double8B

	full_text

double %136
,double8B

	full_text

double %313
Cfsub8B9
7
	full_text*
(
&%315 = fsub double -0.000000e+00, %314
,double8B

	full_text

double %314
mcall8Bc
a
	full_textT
R
P%316 = tail call double @llvm.fmuladd.f64(double %310, double %211, double %315)
,double8B

	full_text

double %310
,double8B

	full_text

double %211
,double8B

	full_text

double %315
vcall8Bl
j
	full_text]
[
Y%317 = tail call double @llvm.fmuladd.f64(double %316, double -5.500000e+00, double %307)
,double8B

	full_text

double %316
,double8B

	full_text

double %307
kcall8Ba
_
	full_textR
P
N%318 = tail call double @_Z3maxdd(double 7.500000e-01, double 1.000000e+00) #5
ccall8BY
W
	full_textJ
H
F%319 = tail call double @_Z3maxdd(double 7.500000e-01, double %318) #5
,double8B

	full_text

double %318
Bfmul8B8
6
	full_text)
'
%%320 = fmul double %319, 2.500000e-01
,double8B

	full_text

double %319
Cfsub8B9
7
	full_text*
(
&%321 = fsub double -0.000000e+00, %320
,double8B

	full_text

double %320
Pload8BF
D
	full_text7
5
3%322 = load double, double* %48, align 16, !tbaa !8
-double*8B

	full_text

double* %48
Pload8BF
D
	full_text7
5
3%323 = load double, double* %75, align 16, !tbaa !8
-double*8B

	full_text

double* %75
Bfmul8B8
6
	full_text)
'
%%324 = fmul double %323, 4.000000e+00
,double8B

	full_text

double %323
Cfsub8B9
7
	full_text*
(
&%325 = fsub double -0.000000e+00, %324
,double8B

	full_text

double %324
ucall8Bk
i
	full_text\
Z
X%326 = tail call double @llvm.fmuladd.f64(double %322, double 5.000000e+00, double %325)
,double8B

	full_text

double %322
,double8B

	full_text

double %325
qgetelementptr8B^
\
	full_textO
M
K%327 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Qload8BG
E
	full_text8
6
4%328 = load double, double* %327, align 16, !tbaa !8
.double*8B

	full_text

double* %327
:fadd8B0
.
	full_text!

%329 = fadd double %328, %326
,double8B

	full_text

double %328
,double8B

	full_text

double %326
mcall8Bc
a
	full_textT
R
P%330 = tail call double @llvm.fmuladd.f64(double %321, double %329, double %237)
,double8B

	full_text

double %321
,double8B

	full_text

double %329
,double8B

	full_text

double %237
Pstore8BE
C
	full_text6
4
2store double %330, double* %226, align 8, !tbaa !8
,double8B

	full_text

double %330
.double*8B

	full_text

double* %226
Oload8BE
C
	full_text6
4
2%331 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
Oload8BE
C
	full_text6
4
2%332 = load double, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
Bfmul8B8
6
	full_text)
'
%%333 = fmul double %332, 4.000000e+00
,double8B

	full_text

double %332
Cfsub8B9
7
	full_text*
(
&%334 = fsub double -0.000000e+00, %333
,double8B

	full_text

double %333
ucall8Bk
i
	full_text\
Z
X%335 = tail call double @llvm.fmuladd.f64(double %331, double 5.000000e+00, double %334)
,double8B

	full_text

double %331
,double8B

	full_text

double %334
Pload8BF
D
	full_text7
5
3%336 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%337 = fadd double %336, %335
,double8B

	full_text

double %336
,double8B

	full_text

double %335
mcall8Bc
a
	full_textT
R
P%338 = tail call double @llvm.fmuladd.f64(double %321, double %337, double %252)
,double8B

	full_text

double %321
,double8B

	full_text

double %337
,double8B

	full_text

double %252
Pstore8BE
C
	full_text6
4
2store double %338, double* %238, align 8, !tbaa !8
,double8B

	full_text

double %338
.double*8B

	full_text

double* %238
Pload8BF
D
	full_text7
5
3%339 = load double, double* %58, align 16, !tbaa !8
-double*8B

	full_text

double* %58
Pload8BF
D
	full_text7
5
3%340 = load double, double* %85, align 16, !tbaa !8
-double*8B

	full_text

double* %85
Bfmul8B8
6
	full_text)
'
%%341 = fmul double %340, 4.000000e+00
,double8B

	full_text

double %340
Cfsub8B9
7
	full_text*
(
&%342 = fsub double -0.000000e+00, %341
,double8B

	full_text

double %341
ucall8Bk
i
	full_text\
Z
X%343 = tail call double @llvm.fmuladd.f64(double %339, double 5.000000e+00, double %342)
,double8B

	full_text

double %339
,double8B

	full_text

double %342
Qload8BG
E
	full_text8
6
4%344 = load double, double* %111, align 16, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%345 = fadd double %344, %343
,double8B

	full_text

double %344
,double8B

	full_text

double %343
mcall8Bc
a
	full_textT
R
P%346 = tail call double @llvm.fmuladd.f64(double %321, double %345, double %267)
,double8B

	full_text

double %321
,double8B

	full_text

double %345
,double8B

	full_text

double %267
Pstore8BE
C
	full_text6
4
2store double %346, double* %253, align 8, !tbaa !8
,double8B

	full_text

double %346
.double*8B

	full_text

double* %253
Oload8BE
C
	full_text6
4
2%347 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
Bfmul8B8
6
	full_text)
'
%%348 = fmul double %347, 4.000000e+00
,double8B

	full_text

double %347
Cfsub8B9
7
	full_text*
(
&%349 = fsub double -0.000000e+00, %348
,double8B

	full_text

double %348
ucall8Bk
i
	full_text\
Z
X%350 = tail call double @llvm.fmuladd.f64(double %270, double 5.000000e+00, double %349)
,double8B

	full_text

double %270
,double8B

	full_text

double %349
Pload8BF
D
	full_text7
5
3%351 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%352 = fadd double %351, %350
,double8B

	full_text

double %351
,double8B

	full_text

double %350
mcall8Bc
a
	full_textT
R
P%353 = tail call double @llvm.fmuladd.f64(double %321, double %352, double %286)
,double8B

	full_text

double %321
,double8B

	full_text

double %352
,double8B

	full_text

double %286
Pstore8BE
C
	full_text6
4
2store double %353, double* %268, align 8, !tbaa !8
,double8B

	full_text

double %353
.double*8B

	full_text

double* %268
Bfmul8B8
6
	full_text)
'
%%354 = fmul double %280, 4.000000e+00
,double8B

	full_text

double %280
Cfsub8B9
7
	full_text*
(
&%355 = fsub double -0.000000e+00, %354
,double8B

	full_text

double %354
ucall8Bk
i
	full_text\
Z
X%356 = tail call double @llvm.fmuladd.f64(double %289, double 5.000000e+00, double %355)
,double8B

	full_text

double %289
,double8B

	full_text

double %355
Qload8BG
E
	full_text8
6
4%357 = load double, double* %121, align 16, !tbaa !8
.double*8B

	full_text

double* %121
:fadd8B0
.
	full_text!

%358 = fadd double %357, %356
,double8B

	full_text

double %357
,double8B

	full_text

double %356
mcall8Bc
a
	full_textT
R
P%359 = tail call double @llvm.fmuladd.f64(double %321, double %358, double %317)
,double8B

	full_text

double %321
,double8B

	full_text

double %358
,double8B

	full_text

double %317
Pstore8BE
C
	full_text6
4
2store double %359, double* %287, align 8, !tbaa !8
,double8B

	full_text

double %359
.double*8B

	full_text

double* %287
Xbitcast8BK
I
	full_text<
:
8%360 = bitcast double* %7 to [13 x [13 x [5 x double]]]*
Jstore8B?
=
	full_text0
.
,store i64 %47, i64* %162, align 16, !tbaa !8
%i648B

	full_text
	
i64 %47
(i64*8B

	full_text

	i64* %162
Istore8B>
<
	full_text/
-
+store i64 %52, i64* %167, align 8, !tbaa !8
%i648B

	full_text
	
i64 %52
(i64*8B

	full_text

	i64* %167
Jstore8B?
=
	full_text0
.
,store i64 %57, i64* %172, align 16, !tbaa !8
%i648B

	full_text
	
i64 %57
(i64*8B

	full_text

	i64* %172
Istore8B>
<
	full_text/
-
+store i64 %62, i64* %177, align 8, !tbaa !8
%i648B

	full_text
	
i64 %62
(i64*8B

	full_text

	i64* %177
Jstore8B?
=
	full_text0
.
,store i64 %67, i64* %182, align 16, !tbaa !8
%i648B

	full_text
	
i64 %67
(i64*8B

	full_text

	i64* %182
Jstore8B?
=
	full_text0
.
,store i64 %74, i64* %160, align 16, !tbaa !8
%i648B

	full_text
	
i64 %74
(i64*8B

	full_text

	i64* %160
Istore8B>
<
	full_text/
-
+store i64 %79, i64* %164, align 8, !tbaa !8
%i648B

	full_text
	
i64 %79
(i64*8B

	full_text

	i64* %164
Jstore8B?
=
	full_text0
.
,store i64 %84, i64* %169, align 16, !tbaa !8
%i648B

	full_text
	
i64 %84
(i64*8B

	full_text

	i64* %169
Istore8B>
<
	full_text/
-
+store i64 %89, i64* %174, align 8, !tbaa !8
%i648B

	full_text
	
i64 %89
(i64*8B

	full_text

	i64* %174
Jstore8B?
=
	full_text0
.
,store i64 %94, i64* %179, align 16, !tbaa !8
%i648B

	full_text
	
i64 %94
(i64*8B

	full_text

	i64* %179
Jstore8B?
=
	full_text0
.
,store i64 %101, i64* %49, align 16, !tbaa !8
&i648B

	full_text


i64 %101
'i64*8B

	full_text


i64* %49
Istore8B>
<
	full_text/
-
+store i64 %105, i64* %54, align 8, !tbaa !8
&i648B

	full_text


i64 %105
'i64*8B

	full_text


i64* %54
Jstore8B?
=
	full_text0
.
,store i64 %110, i64* %59, align 16, !tbaa !8
&i648B

	full_text


i64 %110
'i64*8B

	full_text


i64* %59
Istore8B>
<
	full_text/
-
+store i64 %115, i64* %64, align 8, !tbaa !8
&i648B

	full_text


i64 %115
'i64*8B

	full_text


i64* %64
Jstore8B?
=
	full_text0
.
,store i64 %120, i64* %69, align 16, !tbaa !8
&i648B

	full_text


i64 %120
'i64*8B

	full_text


i64* %69
Jstore8B?
=
	full_text0
.
,store i64 %187, i64* %76, align 16, !tbaa !8
&i648B

	full_text


i64 %187
'i64*8B

	full_text


i64* %76
Istore8B>
<
	full_text/
-
+store i64 %190, i64* %81, align 8, !tbaa !8
&i648B

	full_text


i64 %190
'i64*8B

	full_text


i64* %81
Jstore8B?
=
	full_text0
.
,store i64 %193, i64* %86, align 16, !tbaa !8
&i648B

	full_text


i64 %193
'i64*8B

	full_text


i64* %86
Istore8B>
<
	full_text/
-
+store i64 %196, i64* %91, align 8, !tbaa !8
&i648B

	full_text


i64 %196
'i64*8B

	full_text


i64* %91
Jstore8B?
=
	full_text0
.
,store i64 %199, i64* %96, align 16, !tbaa !8
&i648B

	full_text


i64 %199
'i64*8B

	full_text


i64* %96
`getelementptr8BM
K
	full_text>
<
:%361 = getelementptr inbounds double, double* %0, i64 3380
Zbitcast8BM
K
	full_text>
<
:%362 = bitcast double* %361 to [13 x [13 x [5 x double]]]*
.double*8B

	full_text

double* %361
›getelementptr8B‡
„
	full_textw
u
s%363 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %362, i64 0, i64 %42, i64 %44
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %362
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Ibitcast8B<
:
	full_text-
+
)%364 = bitcast [5 x double]* %363 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %363
Jload8B@
>
	full_text1
/
-%365 = load i64, i64* %364, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %364
Kstore8B@
>
	full_text1
/
-store i64 %365, i64* %102, align 16, !tbaa !8
&i648B

	full_text


i64 %365
(i64*8B

	full_text

	i64* %102
¢getelementptr8BŽ
‹
	full_text~
|
z%366 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %362, i64 0, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %362
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%367 = bitcast double* %366 to i64*
.double*8B

	full_text

double* %366
Jload8B@
>
	full_text1
/
-%368 = load i64, i64* %367, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %367
Jstore8B?
=
	full_text0
.
,store i64 %368, i64* %107, align 8, !tbaa !8
&i648B

	full_text


i64 %368
(i64*8B

	full_text

	i64* %107
¢getelementptr8BŽ
‹
	full_text~
|
z%369 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %362, i64 0, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %362
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%370 = bitcast double* %369 to i64*
.double*8B

	full_text

double* %369
Jload8B@
>
	full_text1
/
-%371 = load i64, i64* %370, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %370
Kstore8B@
>
	full_text1
/
-store i64 %371, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %371
(i64*8B

	full_text

	i64* %112
¢getelementptr8BŽ
‹
	full_text~
|
z%372 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %362, i64 0, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %362
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%373 = bitcast double* %372 to i64*
.double*8B

	full_text

double* %372
Jload8B@
>
	full_text1
/
-%374 = load i64, i64* %373, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %373
Jstore8B?
=
	full_text0
.
,store i64 %374, i64* %117, align 8, !tbaa !8
&i648B

	full_text


i64 %374
(i64*8B

	full_text

	i64* %117
¢getelementptr8BŽ
‹
	full_text~
|
z%375 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %362, i64 0, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %362
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%376 = bitcast double* %375 to i64*
.double*8B

	full_text

double* %375
Jload8B@
>
	full_text1
/
-%377 = load i64, i64* %376, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %376
Kstore8B@
>
	full_text1
/
-store i64 %377, i64* %122, align 16, !tbaa !8
&i648B

	full_text


i64 %377
(i64*8B

	full_text

	i64* %122
_getelementptr8BL
J
	full_text=
;
9%378 = getelementptr inbounds double, double* %1, i64 507
Tbitcast8BG
E
	full_text8
6
4%379 = bitcast double* %378 to [13 x [13 x double]]*
.double*8B

	full_text

double* %378
getelementptr8Bz
x
	full_textk
i
g%380 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %379, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %379
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%381 = load double, double* %380, align 8, !tbaa !8
.double*8B

	full_text

double* %380
_getelementptr8BL
J
	full_text=
;
9%382 = getelementptr inbounds double, double* %2, i64 507
Tbitcast8BG
E
	full_text8
6
4%383 = bitcast double* %382 to [13 x [13 x double]]*
.double*8B

	full_text

double* %382
getelementptr8Bz
x
	full_textk
i
g%384 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %383, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %383
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%385 = load double, double* %384, align 8, !tbaa !8
.double*8B

	full_text

double* %384
_getelementptr8BL
J
	full_text=
;
9%386 = getelementptr inbounds double, double* %3, i64 507
Tbitcast8BG
E
	full_text8
6
4%387 = bitcast double* %386 to [13 x [13 x double]]*
.double*8B

	full_text

double* %386
getelementptr8Bz
x
	full_textk
i
g%388 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %387, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %387
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%389 = load double, double* %388, align 8, !tbaa !8
.double*8B

	full_text

double* %388
_getelementptr8BL
J
	full_text=
;
9%390 = getelementptr inbounds double, double* %4, i64 507
Tbitcast8BG
E
	full_text8
6
4%391 = bitcast double* %390 to [13 x [13 x double]]*
.double*8B

	full_text

double* %390
getelementptr8Bz
x
	full_textk
i
g%392 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %391, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %391
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%393 = load double, double* %392, align 8, !tbaa !8
.double*8B

	full_text

double* %392
_getelementptr8BL
J
	full_text=
;
9%394 = getelementptr inbounds double, double* %5, i64 507
Tbitcast8BG
E
	full_text8
6
4%395 = bitcast double* %394 to [13 x [13 x double]]*
.double*8B

	full_text

double* %394
getelementptr8Bz
x
	full_textk
i
g%396 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %395, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %395
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%397 = load double, double* %396, align 8, !tbaa !8
.double*8B

	full_text

double* %396
_getelementptr8BL
J
	full_text=
;
9%398 = getelementptr inbounds double, double* %6, i64 507
Tbitcast8BG
E
	full_text8
6
4%399 = bitcast double* %398 to [13 x [13 x double]]*
.double*8B

	full_text

double* %398
getelementptr8Bz
x
	full_textk
i
g%400 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %399, i64 0, i64 %42, i64 %44
J[13 x [13 x double]]*8B-
+
	full_text

[13 x [13 x double]]* %399
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%401 = load double, double* %400, align 8, !tbaa !8
.double*8B

	full_text

double* %400
`getelementptr8BM
K
	full_text>
<
:%402 = getelementptr inbounds double, double* %7, i64 1690
Zbitcast8BM
K
	full_text>
<
:%403 = bitcast double* %402 to [13 x [13 x [5 x double]]]*
.double*8B

	full_text

double* %402
¢getelementptr8BŽ
‹
	full_text~
|
z%404 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %403, i64 0, i64 %42, i64 %44, i64 0
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %403
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%405 = load double, double* %404, align 8, !tbaa !8
.double*8B

	full_text

double* %404
Abitcast8B4
2
	full_text%
#
!%406 = bitcast i64 %187 to double
&i648B

	full_text


i64 %187
vcall8Bl
j
	full_text]
[
Y%407 = tail call double @llvm.fmuladd.f64(double %228, double -2.000000e+00, double %406)
,double8B

	full_text

double %228
,double8B

	full_text

double %406
:fadd8B0
.
	full_text!

%408 = fadd double %407, %229
,double8B

	full_text

double %407
,double8B

	full_text

double %229
ucall8Bk
i
	full_text\
Z
X%409 = tail call double @llvm.fmuladd.f64(double %408, double 1.210000e+02, double %405)
,double8B

	full_text

double %408
,double8B

	full_text

double %405
Abitcast8B4
2
	full_text%
#
!%410 = bitcast i64 %196 to double
&i648B

	full_text


i64 %196
@bitcast8B3
1
	full_text$
"
 %411 = bitcast i64 %89 to double
%i648B

	full_text
	
i64 %89
:fsub8B0
.
	full_text!

%412 = fsub double %410, %411
,double8B

	full_text

double %410
,double8B

	full_text

double %411
vcall8Bl
j
	full_text]
[
Y%413 = tail call double @llvm.fmuladd.f64(double %412, double -5.500000e+00, double %409)
,double8B

	full_text

double %412
,double8B

	full_text

double %409
¢getelementptr8BŽ
‹
	full_text~
|
z%414 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %403, i64 0, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %403
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%415 = load double, double* %414, align 8, !tbaa !8
.double*8B

	full_text

double* %414
Abitcast8B4
2
	full_text%
#
!%416 = bitcast i64 %190 to double
&i648B

	full_text


i64 %190
vcall8Bl
j
	full_text]
[
Y%417 = tail call double @llvm.fmuladd.f64(double %240, double -2.000000e+00, double %416)
,double8B

	full_text

double %240
,double8B

	full_text

double %416
:fadd8B0
.
	full_text!

%418 = fadd double %417, %241
,double8B

	full_text

double %417
,double8B

	full_text

double %241
ucall8Bk
i
	full_text\
Z
X%419 = tail call double @llvm.fmuladd.f64(double %418, double 1.210000e+02, double %415)
,double8B

	full_text

double %418
,double8B

	full_text

double %415
vcall8Bl
j
	full_text]
[
Y%420 = tail call double @llvm.fmuladd.f64(double %203, double -2.000000e+00, double %381)
,double8B

	full_text

double %203
,double8B

	full_text

double %381
:fadd8B0
.
	full_text!

%421 = fadd double %128, %420
,double8B

	full_text

double %128
,double8B

	full_text

double %420
{call8Bq
o
	full_textb
`
^%422 = tail call double @llvm.fmuladd.f64(double %421, double 0x4028333333333334, double %419)
,double8B

	full_text

double %421
,double8B

	full_text

double %419
:fmul8B0
.
	full_text!

%423 = fmul double %140, %241
,double8B

	full_text

double %140
,double8B

	full_text

double %241
Cfsub8B9
7
	full_text*
(
&%424 = fsub double -0.000000e+00, %423
,double8B

	full_text

double %423
mcall8Bc
a
	full_textT
R
P%425 = tail call double @llvm.fmuladd.f64(double %416, double %389, double %424)
,double8B

	full_text

double %416
,double8B

	full_text

double %389
,double8B

	full_text

double %424
vcall8Bl
j
	full_text]
[
Y%426 = tail call double @llvm.fmuladd.f64(double %425, double -5.500000e+00, double %422)
,double8B

	full_text

double %425
,double8B

	full_text

double %422
¢getelementptr8BŽ
‹
	full_text~
|
z%427 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %403, i64 0, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %403
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%428 = load double, double* %427, align 8, !tbaa !8
.double*8B

	full_text

double* %427
Abitcast8B4
2
	full_text%
#
!%429 = bitcast i64 %193 to double
&i648B

	full_text


i64 %193
vcall8Bl
j
	full_text]
[
Y%430 = tail call double @llvm.fmuladd.f64(double %255, double -2.000000e+00, double %429)
,double8B

	full_text

double %255
,double8B

	full_text

double %429
:fadd8B0
.
	full_text!

%431 = fadd double %430, %256
,double8B

	full_text

double %430
,double8B

	full_text

double %256
ucall8Bk
i
	full_text\
Z
X%432 = tail call double @llvm.fmuladd.f64(double %431, double 1.210000e+02, double %428)
,double8B

	full_text

double %431
,double8B

	full_text

double %428
vcall8Bl
j
	full_text]
[
Y%433 = tail call double @llvm.fmuladd.f64(double %207, double -2.000000e+00, double %385)
,double8B

	full_text

double %207
,double8B

	full_text

double %385
:fadd8B0
.
	full_text!

%434 = fadd double %134, %433
,double8B

	full_text

double %134
,double8B

	full_text

double %433
{call8Bq
o
	full_textb
`
^%435 = tail call double @llvm.fmuladd.f64(double %434, double 0x4028333333333334, double %432)
,double8B

	full_text

double %434
,double8B

	full_text

double %432
:fmul8B0
.
	full_text!

%436 = fmul double %140, %256
,double8B

	full_text

double %140
,double8B

	full_text

double %256
Cfsub8B9
7
	full_text*
(
&%437 = fsub double -0.000000e+00, %436
,double8B

	full_text

double %436
mcall8Bc
a
	full_textT
R
P%438 = tail call double @llvm.fmuladd.f64(double %429, double %389, double %437)
,double8B

	full_text

double %429
,double8B

	full_text

double %389
,double8B

	full_text

double %437
vcall8Bl
j
	full_text]
[
Y%439 = tail call double @llvm.fmuladd.f64(double %438, double -5.500000e+00, double %435)
,double8B

	full_text

double %438
,double8B

	full_text

double %435
¢getelementptr8BŽ
‹
	full_text~
|
z%440 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %403, i64 0, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %403
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%441 = load double, double* %440, align 8, !tbaa !8
.double*8B

	full_text

double* %440
Oload8BE
C
	full_text6
4
2%442 = load double, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
vcall8Bl
j
	full_text]
[
Y%443 = tail call double @llvm.fmuladd.f64(double %442, double -2.000000e+00, double %410)
,double8B

	full_text

double %442
,double8B

	full_text

double %410
:fadd8B0
.
	full_text!

%444 = fadd double %443, %411
,double8B

	full_text

double %443
,double8B

	full_text

double %411
ucall8Bk
i
	full_text\
Z
X%445 = tail call double @llvm.fmuladd.f64(double %444, double 1.210000e+02, double %441)
,double8B

	full_text

double %444
,double8B

	full_text

double %441
vcall8Bl
j
	full_text]
[
Y%446 = tail call double @llvm.fmuladd.f64(double %211, double -2.000000e+00, double %389)
,double8B

	full_text

double %211
,double8B

	full_text

double %389
:fadd8B0
.
	full_text!

%447 = fadd double %140, %446
,double8B

	full_text

double %140
,double8B

	full_text

double %446
{call8Bq
o
	full_textb
`
^%448 = tail call double @llvm.fmuladd.f64(double %447, double 0x4030222222222222, double %445)
,double8B

	full_text

double %447
,double8B

	full_text

double %445
:fmul8B0
.
	full_text!

%449 = fmul double %140, %411
,double8B

	full_text

double %140
,double8B

	full_text

double %411
Cfsub8B9
7
	full_text*
(
&%450 = fsub double -0.000000e+00, %449
,double8B

	full_text

double %449
mcall8Bc
a
	full_textT
R
P%451 = tail call double @llvm.fmuladd.f64(double %410, double %389, double %450)
,double8B

	full_text

double %410
,double8B

	full_text

double %389
,double8B

	full_text

double %450
Pload8BF
D
	full_text7
5
3%452 = load double, double* %95, align 16, !tbaa !8
-double*8B

	full_text

double* %95
:fsub8B0
.
	full_text!

%453 = fsub double %452, %401
,double8B

	full_text

double %452
,double8B

	full_text

double %401
Qload8BG
E
	full_text8
6
4%454 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fsub8B0
.
	full_text!

%455 = fsub double %453, %454
,double8B

	full_text

double %453
,double8B

	full_text

double %454
:fadd8B0
.
	full_text!

%456 = fadd double %158, %455
,double8B

	full_text

double %158
,double8B

	full_text

double %455
ucall8Bk
i
	full_text\
Z
X%457 = tail call double @llvm.fmuladd.f64(double %456, double 4.000000e-01, double %451)
,double8B

	full_text

double %456
,double8B

	full_text

double %451
vcall8Bl
j
	full_text]
[
Y%458 = tail call double @llvm.fmuladd.f64(double %457, double -5.500000e+00, double %448)
,double8B

	full_text

double %457
,double8B

	full_text

double %448
¢getelementptr8BŽ
‹
	full_text~
|
z%459 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %403, i64 0, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %403
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%460 = load double, double* %459, align 8, !tbaa !8
.double*8B

	full_text

double* %459
Pload8BF
D
	full_text7
5
3%461 = load double, double* %68, align 16, !tbaa !8
-double*8B

	full_text

double* %68
vcall8Bl
j
	full_text]
[
Y%462 = tail call double @llvm.fmuladd.f64(double %461, double -2.000000e+00, double %452)
,double8B

	full_text

double %461
,double8B

	full_text

double %452
:fadd8B0
.
	full_text!

%463 = fadd double %454, %462
,double8B

	full_text

double %454
,double8B

	full_text

double %462
ucall8Bk
i
	full_text\
Z
X%464 = tail call double @llvm.fmuladd.f64(double %463, double 1.210000e+02, double %460)
,double8B

	full_text

double %463
,double8B

	full_text

double %460
vcall8Bl
j
	full_text]
[
Y%465 = tail call double @llvm.fmuladd.f64(double %215, double -2.000000e+00, double %393)
,double8B

	full_text

double %215
,double8B

	full_text

double %393
:fadd8B0
.
	full_text!

%466 = fadd double %146, %465
,double8B

	full_text

double %146
,double8B

	full_text

double %465
{call8Bq
o
	full_textb
`
^%467 = tail call double @llvm.fmuladd.f64(double %466, double 0xC0273B645A1CAC07, double %464)
,double8B

	full_text

double %466
,double8B

	full_text

double %464
Bfmul8B8
6
	full_text)
'
%%468 = fmul double %211, 2.000000e+00
,double8B

	full_text

double %211
:fmul8B0
.
	full_text!

%469 = fmul double %211, %468
,double8B

	full_text

double %211
,double8B

	full_text

double %468
Cfsub8B9
7
	full_text*
(
&%470 = fsub double -0.000000e+00, %469
,double8B

	full_text

double %469
mcall8Bc
a
	full_textT
R
P%471 = tail call double @llvm.fmuladd.f64(double %389, double %389, double %470)
,double8B

	full_text

double %389
,double8B

	full_text

double %389
,double8B

	full_text

double %470
mcall8Bc
a
	full_textT
R
P%472 = tail call double @llvm.fmuladd.f64(double %140, double %140, double %471)
,double8B

	full_text

double %140
,double8B

	full_text

double %140
,double8B

	full_text

double %471
{call8Bq
o
	full_textb
`
^%473 = tail call double @llvm.fmuladd.f64(double %472, double 0x4000222222222222, double %467)
,double8B

	full_text

double %472
,double8B

	full_text

double %467
Bfmul8B8
6
	full_text)
'
%%474 = fmul double %461, 2.000000e+00
,double8B

	full_text

double %461
:fmul8B0
.
	full_text!

%475 = fmul double %219, %474
,double8B

	full_text

double %219
,double8B

	full_text

double %474
Cfsub8B9
7
	full_text*
(
&%476 = fsub double -0.000000e+00, %475
,double8B

	full_text

double %475
mcall8Bc
a
	full_textT
R
P%477 = tail call double @llvm.fmuladd.f64(double %452, double %397, double %476)
,double8B

	full_text

double %452
,double8B

	full_text

double %397
,double8B

	full_text

double %476
mcall8Bc
a
	full_textT
R
P%478 = tail call double @llvm.fmuladd.f64(double %454, double %152, double %477)
,double8B

	full_text

double %454
,double8B

	full_text

double %152
,double8B

	full_text

double %477
{call8Bq
o
	full_textb
`
^%479 = tail call double @llvm.fmuladd.f64(double %478, double 0x4037B74BC6A7EF9D, double %473)
,double8B

	full_text

double %478
,double8B

	full_text

double %473
Bfmul8B8
6
	full_text)
'
%%480 = fmul double %401, 4.000000e-01
,double8B

	full_text

double %401
Cfsub8B9
7
	full_text*
(
&%481 = fsub double -0.000000e+00, %480
,double8B

	full_text

double %480
ucall8Bk
i
	full_text\
Z
X%482 = tail call double @llvm.fmuladd.f64(double %452, double 1.400000e+00, double %481)
,double8B

	full_text

double %452
,double8B

	full_text

double %481
Bfmul8B8
6
	full_text)
'
%%483 = fmul double %158, 4.000000e-01
,double8B

	full_text

double %158
Cfsub8B9
7
	full_text*
(
&%484 = fsub double -0.000000e+00, %483
,double8B

	full_text

double %483
ucall8Bk
i
	full_text\
Z
X%485 = tail call double @llvm.fmuladd.f64(double %454, double 1.400000e+00, double %484)
,double8B

	full_text

double %454
,double8B

	full_text

double %484
:fmul8B0
.
	full_text!

%486 = fmul double %140, %485
,double8B

	full_text

double %140
,double8B

	full_text

double %485
Cfsub8B9
7
	full_text*
(
&%487 = fsub double -0.000000e+00, %486
,double8B

	full_text

double %486
mcall8Bc
a
	full_textT
R
P%488 = tail call double @llvm.fmuladd.f64(double %482, double %389, double %487)
,double8B

	full_text

double %482
,double8B

	full_text

double %389
,double8B

	full_text

double %487
vcall8Bl
j
	full_text]
[
Y%489 = tail call double @llvm.fmuladd.f64(double %488, double -5.500000e+00, double %479)
,double8B

	full_text

double %488
,double8B

	full_text

double %479
Qload8BG
E
	full_text8
6
4%490 = load double, double* %159, align 16, !tbaa !8
.double*8B

	full_text

double* %159
Pload8BF
D
	full_text7
5
3%491 = load double, double* %48, align 16, !tbaa !8
-double*8B

	full_text

double* %48
Bfmul8B8
6
	full_text)
'
%%492 = fmul double %491, 6.000000e+00
,double8B

	full_text

double %491
vcall8Bl
j
	full_text]
[
Y%493 = tail call double @llvm.fmuladd.f64(double %490, double -4.000000e+00, double %492)
,double8B

	full_text

double %490
,double8B

	full_text

double %492
Pload8BF
D
	full_text7
5
3%494 = load double, double* %75, align 16, !tbaa !8
-double*8B

	full_text

double* %75
vcall8Bl
j
	full_text]
[
Y%495 = tail call double @llvm.fmuladd.f64(double %494, double -4.000000e+00, double %493)
,double8B

	full_text

double %494
,double8B

	full_text

double %493
Qload8BG
E
	full_text8
6
4%496 = load double, double* %327, align 16, !tbaa !8
.double*8B

	full_text

double* %327
:fadd8B0
.
	full_text!

%497 = fadd double %496, %495
,double8B

	full_text

double %496
,double8B

	full_text

double %495
mcall8Bc
a
	full_textT
R
P%498 = tail call double @llvm.fmuladd.f64(double %321, double %497, double %413)
,double8B

	full_text

double %321
,double8B

	full_text

double %497
,double8B

	full_text

double %413
Pstore8BE
C
	full_text6
4
2store double %498, double* %404, align 8, !tbaa !8
,double8B

	full_text

double %498
.double*8B

	full_text

double* %404
Pload8BF
D
	full_text7
5
3%499 = load double, double* %163, align 8, !tbaa !8
.double*8B

	full_text

double* %163
Oload8BE
C
	full_text6
4
2%500 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
Bfmul8B8
6
	full_text)
'
%%501 = fmul double %500, 6.000000e+00
,double8B

	full_text

double %500
vcall8Bl
j
	full_text]
[
Y%502 = tail call double @llvm.fmuladd.f64(double %499, double -4.000000e+00, double %501)
,double8B

	full_text

double %499
,double8B

	full_text

double %501
Oload8BE
C
	full_text6
4
2%503 = load double, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
vcall8Bl
j
	full_text]
[
Y%504 = tail call double @llvm.fmuladd.f64(double %503, double -4.000000e+00, double %502)
,double8B

	full_text

double %503
,double8B

	full_text

double %502
Pload8BF
D
	full_text7
5
3%505 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%506 = fadd double %505, %504
,double8B

	full_text

double %505
,double8B

	full_text

double %504
mcall8Bc
a
	full_textT
R
P%507 = tail call double @llvm.fmuladd.f64(double %321, double %506, double %426)
,double8B

	full_text

double %321
,double8B

	full_text

double %506
,double8B

	full_text

double %426
Pstore8BE
C
	full_text6
4
2store double %507, double* %414, align 8, !tbaa !8
,double8B

	full_text

double %507
.double*8B

	full_text

double* %414
Qload8BG
E
	full_text8
6
4%508 = load double, double* %168, align 16, !tbaa !8
.double*8B

	full_text

double* %168
Pload8BF
D
	full_text7
5
3%509 = load double, double* %58, align 16, !tbaa !8
-double*8B

	full_text

double* %58
Bfmul8B8
6
	full_text)
'
%%510 = fmul double %509, 6.000000e+00
,double8B

	full_text

double %509
vcall8Bl
j
	full_text]
[
Y%511 = tail call double @llvm.fmuladd.f64(double %508, double -4.000000e+00, double %510)
,double8B

	full_text

double %508
,double8B

	full_text

double %510
Pload8BF
D
	full_text7
5
3%512 = load double, double* %85, align 16, !tbaa !8
-double*8B

	full_text

double* %85
vcall8Bl
j
	full_text]
[
Y%513 = tail call double @llvm.fmuladd.f64(double %512, double -4.000000e+00, double %511)
,double8B

	full_text

double %512
,double8B

	full_text

double %511
Qload8BG
E
	full_text8
6
4%514 = load double, double* %111, align 16, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%515 = fadd double %514, %513
,double8B

	full_text

double %514
,double8B

	full_text

double %513
mcall8Bc
a
	full_textT
R
P%516 = tail call double @llvm.fmuladd.f64(double %321, double %515, double %439)
,double8B

	full_text

double %321
,double8B

	full_text

double %515
,double8B

	full_text

double %439
Pstore8BE
C
	full_text6
4
2store double %516, double* %427, align 8, !tbaa !8
,double8B

	full_text

double %516
.double*8B

	full_text

double* %427
Pload8BF
D
	full_text7
5
3%517 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
Bfmul8B8
6
	full_text)
'
%%518 = fmul double %442, 6.000000e+00
,double8B

	full_text

double %442
vcall8Bl
j
	full_text]
[
Y%519 = tail call double @llvm.fmuladd.f64(double %517, double -4.000000e+00, double %518)
,double8B

	full_text

double %517
,double8B

	full_text

double %518
Oload8BE
C
	full_text6
4
2%520 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
vcall8Bl
j
	full_text]
[
Y%521 = tail call double @llvm.fmuladd.f64(double %520, double -4.000000e+00, double %519)
,double8B

	full_text

double %520
,double8B

	full_text

double %519
Pload8BF
D
	full_text7
5
3%522 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%523 = fadd double %522, %521
,double8B

	full_text

double %522
,double8B

	full_text

double %521
mcall8Bc
a
	full_textT
R
P%524 = tail call double @llvm.fmuladd.f64(double %321, double %523, double %458)
,double8B

	full_text

double %321
,double8B

	full_text

double %523
,double8B

	full_text

double %458
Pstore8BE
C
	full_text6
4
2store double %524, double* %440, align 8, !tbaa !8
,double8B

	full_text

double %524
.double*8B

	full_text

double* %440
Bfmul8B8
6
	full_text)
'
%%525 = fmul double %461, 6.000000e+00
,double8B

	full_text

double %461
vcall8Bl
j
	full_text]
[
Y%526 = tail call double @llvm.fmuladd.f64(double %454, double -4.000000e+00, double %525)
,double8B

	full_text

double %454
,double8B

	full_text

double %525
vcall8Bl
j
	full_text]
[
Y%527 = tail call double @llvm.fmuladd.f64(double %452, double -4.000000e+00, double %526)
,double8B

	full_text

double %452
,double8B

	full_text

double %526
Qload8BG
E
	full_text8
6
4%528 = load double, double* %121, align 16, !tbaa !8
.double*8B

	full_text

double* %121
:fadd8B0
.
	full_text!

%529 = fadd double %528, %527
,double8B

	full_text

double %528
,double8B

	full_text

double %527
mcall8Bc
a
	full_textT
R
P%530 = tail call double @llvm.fmuladd.f64(double %321, double %529, double %489)
,double8B

	full_text

double %321
,double8B

	full_text

double %529
,double8B

	full_text

double %489
Pstore8BE
C
	full_text6
4
2store double %530, double* %459, align 8, !tbaa !8
,double8B

	full_text

double %530
.double*8B

	full_text

double* %459
7icmp8B-
+
	full_text

%531 = icmp slt i32 %10, 7
Abitcast8B4
2
	full_text%
#
!%532 = bitcast double %490 to i64
,double8B

	full_text

double %490
Abitcast8B4
2
	full_text%
#
!%533 = bitcast double %499 to i64
,double8B

	full_text

double %499
Abitcast8B4
2
	full_text%
#
!%534 = bitcast double %508 to i64
,double8B

	full_text

double %508
Abitcast8B4
2
	full_text%
#
!%535 = bitcast double %517 to i64
,double8B

	full_text

double %517
Abitcast8B4
2
	full_text%
#
!%536 = bitcast double %454 to i64
,double8B

	full_text

double %454
Abitcast8B4
2
	full_text%
#
!%537 = bitcast double %461 to i64
,double8B

	full_text

double %461
=br8B5
3
	full_text&
$
"br i1 %531, label %538, label %545
$i18B

	full_text
	
i1 %531
Iload8B?
=
	full_text0
.
,%539 = load i64, i64* %64, align 8, !tbaa !8
'i64*8B

	full_text


i64* %64
Abitcast8B4
2
	full_text%
#
!%540 = bitcast i64 %539 to double
&i648B

	full_text


i64 %539
Jload8B@
>
	full_text1
/
-%541 = load i64, i64* %96, align 16, !tbaa !8
'i64*8B

	full_text


i64* %96
Abitcast8B4
2
	full_text%
#
!%542 = bitcast i64 %541 to double
&i648B

	full_text


i64 %541
6add8B-
+
	full_text

%543 = add nsw i32 %10, -3
qgetelementptr8B^
\
	full_textO
M
K%544 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
(br8B 

	full_text

br label %741
2add8B)
'
	full_text

%546 = add i32 %10, -3
Oload8BE
C
	full_text6
4
2%547 = load double, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
Jload8B@
>
	full_text1
/
-%548 = load i64, i64* %96, align 16, !tbaa !8
'i64*8B

	full_text


i64* %96
qgetelementptr8B^
\
	full_textO
M
K%549 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
qgetelementptr8B^
\
	full_textO
M
K%550 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
qgetelementptr8B^
\
	full_textO
M
K%551 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
qgetelementptr8B^
\
	full_textO
M
K%552 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
8zext8B.
,
	full_text

%553 = zext i32 %546 to i64
&i328B

	full_text


i32 %546
(br8B 

	full_text

br label %554
Lphi8BC
A
	full_text4
2
0%555 = phi double [ %728, %554 ], [ %528, %545 ]
,double8B

	full_text

double %728
,double8B

	full_text

double %528
Lphi8BC
A
	full_text4
2
0%556 = phi double [ %721, %554 ], [ %522, %545 ]
,double8B

	full_text

double %721
,double8B

	full_text

double %522
Lphi8BC
A
	full_text4
2
0%557 = phi double [ %714, %554 ], [ %514, %545 ]
,double8B

	full_text

double %714
,double8B

	full_text

double %514
Lphi8BC
A
	full_text4
2
0%558 = phi double [ %707, %554 ], [ %505, %545 ]
,double8B

	full_text

double %707
,double8B

	full_text

double %505
Lphi8BC
A
	full_text4
2
0%559 = phi double [ %700, %554 ], [ %496, %545 ]
,double8B

	full_text

double %700
,double8B

	full_text

double %496
Iphi8B@
>
	full_text1
/
-%560 = phi i64 [ %738, %554 ], [ %548, %545 ]
&i648B

	full_text


i64 %738
&i648B

	full_text


i64 %548
Lphi8BC
A
	full_text4
2
0%561 = phi double [ %556, %554 ], [ %520, %545 ]
,double8B

	full_text

double %556
,double8B

	full_text

double %520
Lphi8BC
A
	full_text4
2
0%562 = phi double [ %557, %554 ], [ %512, %545 ]
,double8B

	full_text

double %557
,double8B

	full_text

double %512
Lphi8BC
A
	full_text4
2
0%563 = phi double [ %558, %554 ], [ %503, %545 ]
,double8B

	full_text

double %558
,double8B

	full_text

double %503
Lphi8BC
A
	full_text4
2
0%564 = phi double [ %559, %554 ], [ %494, %545 ]
,double8B

	full_text

double %559
,double8B

	full_text

double %494
Iphi8B@
>
	full_text1
/
-%565 = phi i64 [ %737, %554 ], [ %537, %545 ]
&i648B

	full_text


i64 %737
&i648B

	full_text


i64 %537
Lphi8BC
A
	full_text4
2
0%566 = phi double [ %561, %554 ], [ %547, %545 ]
,double8B

	full_text

double %561
,double8B

	full_text

double %547
Lphi8BC
A
	full_text4
2
0%567 = phi double [ %562, %554 ], [ %509, %545 ]
,double8B

	full_text

double %562
,double8B

	full_text

double %509
Lphi8BC
A
	full_text4
2
0%568 = phi double [ %563, %554 ], [ %500, %545 ]
,double8B

	full_text

double %563
,double8B

	full_text

double %500
Lphi8BC
A
	full_text4
2
0%569 = phi double [ %564, %554 ], [ %491, %545 ]
,double8B

	full_text

double %564
,double8B

	full_text

double %491
Iphi8B@
>
	full_text1
/
-%570 = phi i64 [ %736, %554 ], [ %536, %545 ]
&i648B

	full_text


i64 %736
&i648B

	full_text


i64 %536
Iphi8B@
>
	full_text1
/
-%571 = phi i64 [ %735, %554 ], [ %535, %545 ]
&i648B

	full_text


i64 %735
&i648B

	full_text


i64 %535
Iphi8B@
>
	full_text1
/
-%572 = phi i64 [ %734, %554 ], [ %534, %545 ]
&i648B

	full_text


i64 %734
&i648B

	full_text


i64 %534
Iphi8B@
>
	full_text1
/
-%573 = phi i64 [ %733, %554 ], [ %533, %545 ]
&i648B

	full_text


i64 %733
&i648B

	full_text


i64 %533
Iphi8B@
>
	full_text1
/
-%574 = phi i64 [ %732, %554 ], [ %532, %545 ]
&i648B

	full_text


i64 %732
&i648B

	full_text


i64 %532
Fphi8B=
;
	full_text.
,
*%575 = phi i64 [ %604, %554 ], [ 3, %545 ]
&i648B

	full_text


i64 %604
Lphi8BC
A
	full_text4
2
0%576 = phi double [ %577, %554 ], [ %203, %545 ]
,double8B

	full_text

double %577
,double8B

	full_text

double %203
Lphi8BC
A
	full_text4
2
0%577 = phi double [ %606, %554 ], [ %381, %545 ]
,double8B

	full_text

double %606
,double8B

	full_text

double %381
Lphi8BC
A
	full_text4
2
0%578 = phi double [ %579, %554 ], [ %207, %545 ]
,double8B

	full_text

double %579
,double8B

	full_text

double %207
Lphi8BC
A
	full_text4
2
0%579 = phi double [ %608, %554 ], [ %385, %545 ]
,double8B

	full_text

double %608
,double8B

	full_text

double %385
Lphi8BC
A
	full_text4
2
0%580 = phi double [ %616, %554 ], [ %401, %545 ]
,double8B

	full_text

double %616
,double8B

	full_text

double %401
Lphi8BC
A
	full_text4
2
0%581 = phi double [ %580, %554 ], [ %223, %545 ]
,double8B

	full_text

double %580
,double8B

	full_text

double %223
Lphi8BC
A
	full_text4
2
0%582 = phi double [ %614, %554 ], [ %397, %545 ]
,double8B

	full_text

double %614
,double8B

	full_text

double %397
Lphi8BC
A
	full_text4
2
0%583 = phi double [ %582, %554 ], [ %219, %545 ]
,double8B

	full_text

double %582
,double8B

	full_text

double %219
Lphi8BC
A
	full_text4
2
0%584 = phi double [ %612, %554 ], [ %393, %545 ]
,double8B

	full_text

double %612
,double8B

	full_text

double %393
Lphi8BC
A
	full_text4
2
0%585 = phi double [ %584, %554 ], [ %215, %545 ]
,double8B

	full_text

double %584
,double8B

	full_text

double %215
Lphi8BC
A
	full_text4
2
0%586 = phi double [ %610, %554 ], [ %389, %545 ]
,double8B

	full_text

double %610
,double8B

	full_text

double %389
Lphi8BC
A
	full_text4
2
0%587 = phi double [ %586, %554 ], [ %211, %545 ]
,double8B

	full_text

double %586
,double8B

	full_text

double %211
Kstore8B@
>
	full_text1
/
-store i64 %574, i64* %162, align 16, !tbaa !8
&i648B

	full_text


i64 %574
(i64*8B

	full_text

	i64* %162
Jstore8B?
=
	full_text0
.
,store i64 %573, i64* %167, align 8, !tbaa !8
&i648B

	full_text


i64 %573
(i64*8B

	full_text

	i64* %167
Kstore8B@
>
	full_text1
/
-store i64 %572, i64* %172, align 16, !tbaa !8
&i648B

	full_text


i64 %572
(i64*8B

	full_text

	i64* %172
Jstore8B?
=
	full_text0
.
,store i64 %571, i64* %177, align 8, !tbaa !8
&i648B

	full_text


i64 %571
(i64*8B

	full_text

	i64* %177
Kstore8B@
>
	full_text1
/
-store i64 %570, i64* %182, align 16, !tbaa !8
&i648B

	full_text


i64 %570
(i64*8B

	full_text

	i64* %182
Kstore8B@
>
	full_text1
/
-store i64 %565, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %565
(i64*8B

	full_text

	i64* %179
Jstore8B?
=
	full_text0
.
,store i64 %560, i64* %69, align 16, !tbaa !8
&i648B

	full_text


i64 %560
'i64*8B

	full_text


i64* %69
:add8B1
/
	full_text"
 
%588 = add nuw nsw i64 %575, 2
&i648B

	full_text


i64 %575
getelementptr8B‰
†
	full_texty
w
u%589 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %588, i64 %42, i64 %44
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %588
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Ibitcast8B<
:
	full_text-
+
)%590 = bitcast [5 x double]* %589 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %589
Jload8B@
>
	full_text1
/
-%591 = load i64, i64* %590, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %590
Kstore8B@
>
	full_text1
/
-store i64 %591, i64* %102, align 16, !tbaa !8
&i648B

	full_text


i64 %591
(i64*8B

	full_text

	i64* %102
¥getelementptr8B‘
Ž
	full_text€
~
|%592 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %588, i64 %42, i64 %44, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %588
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%593 = bitcast double* %592 to i64*
.double*8B

	full_text

double* %592
Jload8B@
>
	full_text1
/
-%594 = load i64, i64* %593, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %593
Jstore8B?
=
	full_text0
.
,store i64 %594, i64* %107, align 8, !tbaa !8
&i648B

	full_text


i64 %594
(i64*8B

	full_text

	i64* %107
¥getelementptr8B‘
Ž
	full_text€
~
|%595 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %588, i64 %42, i64 %44, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %588
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%596 = bitcast double* %595 to i64*
.double*8B

	full_text

double* %595
Jload8B@
>
	full_text1
/
-%597 = load i64, i64* %596, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %596
Kstore8B@
>
	full_text1
/
-store i64 %597, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %597
(i64*8B

	full_text

	i64* %112
¥getelementptr8B‘
Ž
	full_text€
~
|%598 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %588, i64 %42, i64 %44, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %588
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%599 = bitcast double* %598 to i64*
.double*8B

	full_text

double* %598
Jload8B@
>
	full_text1
/
-%600 = load i64, i64* %599, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %599
Jstore8B?
=
	full_text0
.
,store i64 %600, i64* %117, align 8, !tbaa !8
&i648B

	full_text


i64 %600
(i64*8B

	full_text

	i64* %117
¥getelementptr8B‘
Ž
	full_text€
~
|%601 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %588, i64 %42, i64 %44, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %588
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%602 = bitcast double* %601 to i64*
.double*8B

	full_text

double* %601
Jload8B@
>
	full_text1
/
-%603 = load i64, i64* %602, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %602
Kstore8B@
>
	full_text1
/
-store i64 %603, i64* %122, align 16, !tbaa !8
&i648B

	full_text


i64 %603
(i64*8B

	full_text

	i64* %122
:add8B1
/
	full_text"
 
%604 = add nuw nsw i64 %575, 1
&i648B

	full_text


i64 %575
getelementptr8B|
z
	full_textm
k
i%605 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %34, i64 %604, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %34
&i648B

	full_text


i64 %604
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%606 = load double, double* %605, align 8, !tbaa !8
.double*8B

	full_text

double* %605
getelementptr8B|
z
	full_textm
k
i%607 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %35, i64 %604, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %35
&i648B

	full_text


i64 %604
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%608 = load double, double* %607, align 8, !tbaa !8
.double*8B

	full_text

double* %607
getelementptr8B|
z
	full_textm
k
i%609 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %36, i64 %604, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %36
&i648B

	full_text


i64 %604
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%610 = load double, double* %609, align 8, !tbaa !8
.double*8B

	full_text

double* %609
getelementptr8B|
z
	full_textm
k
i%611 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %37, i64 %604, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %37
&i648B

	full_text


i64 %604
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%612 = load double, double* %611, align 8, !tbaa !8
.double*8B

	full_text

double* %611
getelementptr8B|
z
	full_textm
k
i%613 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %38, i64 %604, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %38
&i648B

	full_text


i64 %604
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%614 = load double, double* %613, align 8, !tbaa !8
.double*8B

	full_text

double* %613
getelementptr8B|
z
	full_textm
k
i%615 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %39, i64 %604, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %39
&i648B

	full_text


i64 %604
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%616 = load double, double* %615, align 8, !tbaa !8
.double*8B

	full_text

double* %615
¦getelementptr8B’

	full_text

}%617 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %575, i64 %42, i64 %44, i64 0
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %575
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%618 = load double, double* %617, align 8, !tbaa !8
.double*8B

	full_text

double* %617
vcall8Bl
j
	full_text]
[
Y%619 = tail call double @llvm.fmuladd.f64(double %564, double -2.000000e+00, double %559)
,double8B

	full_text

double %564
,double8B

	full_text

double %559
:fadd8B0
.
	full_text!

%620 = fadd double %619, %569
,double8B

	full_text

double %619
,double8B

	full_text

double %569
ucall8Bk
i
	full_text\
Z
X%621 = tail call double @llvm.fmuladd.f64(double %620, double 1.210000e+02, double %618)
,double8B

	full_text

double %620
,double8B

	full_text

double %618
:fsub8B0
.
	full_text!

%622 = fsub double %556, %566
,double8B

	full_text

double %556
,double8B

	full_text

double %566
vcall8Bl
j
	full_text]
[
Y%623 = tail call double @llvm.fmuladd.f64(double %622, double -5.500000e+00, double %621)
,double8B

	full_text

double %622
,double8B

	full_text

double %621
¦getelementptr8B’

	full_text

}%624 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %575, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %575
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%625 = load double, double* %624, align 8, !tbaa !8
.double*8B

	full_text

double* %624
vcall8Bl
j
	full_text]
[
Y%626 = tail call double @llvm.fmuladd.f64(double %563, double -2.000000e+00, double %558)
,double8B

	full_text

double %563
,double8B

	full_text

double %558
:fadd8B0
.
	full_text!

%627 = fadd double %626, %568
,double8B

	full_text

double %626
,double8B

	full_text

double %568
ucall8Bk
i
	full_text\
Z
X%628 = tail call double @llvm.fmuladd.f64(double %627, double 1.210000e+02, double %625)
,double8B

	full_text

double %627
,double8B

	full_text

double %625
vcall8Bl
j
	full_text]
[
Y%629 = tail call double @llvm.fmuladd.f64(double %577, double -2.000000e+00, double %606)
,double8B

	full_text

double %577
,double8B

	full_text

double %606
:fadd8B0
.
	full_text!

%630 = fadd double %576, %629
,double8B

	full_text

double %576
,double8B

	full_text

double %629
{call8Bq
o
	full_textb
`
^%631 = tail call double @llvm.fmuladd.f64(double %630, double 0x4028333333333334, double %628)
,double8B

	full_text

double %630
,double8B

	full_text

double %628
:fmul8B0
.
	full_text!

%632 = fmul double %587, %568
,double8B

	full_text

double %587
,double8B

	full_text

double %568
Cfsub8B9
7
	full_text*
(
&%633 = fsub double -0.000000e+00, %632
,double8B

	full_text

double %632
mcall8Bc
a
	full_textT
R
P%634 = tail call double @llvm.fmuladd.f64(double %558, double %610, double %633)
,double8B

	full_text

double %558
,double8B

	full_text

double %610
,double8B

	full_text

double %633
vcall8Bl
j
	full_text]
[
Y%635 = tail call double @llvm.fmuladd.f64(double %634, double -5.500000e+00, double %631)
,double8B

	full_text

double %634
,double8B

	full_text

double %631
¦getelementptr8B’

	full_text

}%636 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %575, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %575
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%637 = load double, double* %636, align 8, !tbaa !8
.double*8B

	full_text

double* %636
vcall8Bl
j
	full_text]
[
Y%638 = tail call double @llvm.fmuladd.f64(double %562, double -2.000000e+00, double %557)
,double8B

	full_text

double %562
,double8B

	full_text

double %557
:fadd8B0
.
	full_text!

%639 = fadd double %638, %567
,double8B

	full_text

double %638
,double8B

	full_text

double %567
ucall8Bk
i
	full_text\
Z
X%640 = tail call double @llvm.fmuladd.f64(double %639, double 1.210000e+02, double %637)
,double8B

	full_text

double %639
,double8B

	full_text

double %637
vcall8Bl
j
	full_text]
[
Y%641 = tail call double @llvm.fmuladd.f64(double %579, double -2.000000e+00, double %608)
,double8B

	full_text

double %579
,double8B

	full_text

double %608
:fadd8B0
.
	full_text!

%642 = fadd double %578, %641
,double8B

	full_text

double %578
,double8B

	full_text

double %641
{call8Bq
o
	full_textb
`
^%643 = tail call double @llvm.fmuladd.f64(double %642, double 0x4028333333333334, double %640)
,double8B

	full_text

double %642
,double8B

	full_text

double %640
:fmul8B0
.
	full_text!

%644 = fmul double %587, %567
,double8B

	full_text

double %587
,double8B

	full_text

double %567
Cfsub8B9
7
	full_text*
(
&%645 = fsub double -0.000000e+00, %644
,double8B

	full_text

double %644
mcall8Bc
a
	full_textT
R
P%646 = tail call double @llvm.fmuladd.f64(double %557, double %610, double %645)
,double8B

	full_text

double %557
,double8B

	full_text

double %610
,double8B

	full_text

double %645
vcall8Bl
j
	full_text]
[
Y%647 = tail call double @llvm.fmuladd.f64(double %646, double -5.500000e+00, double %643)
,double8B

	full_text

double %646
,double8B

	full_text

double %643
¦getelementptr8B’

	full_text

}%648 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %575, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %575
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%649 = load double, double* %648, align 8, !tbaa !8
.double*8B

	full_text

double* %648
vcall8Bl
j
	full_text]
[
Y%650 = tail call double @llvm.fmuladd.f64(double %561, double -2.000000e+00, double %556)
,double8B

	full_text

double %561
,double8B

	full_text

double %556
:fadd8B0
.
	full_text!

%651 = fadd double %566, %650
,double8B

	full_text

double %566
,double8B

	full_text

double %650
ucall8Bk
i
	full_text\
Z
X%652 = tail call double @llvm.fmuladd.f64(double %651, double 1.210000e+02, double %649)
,double8B

	full_text

double %651
,double8B

	full_text

double %649
vcall8Bl
j
	full_text]
[
Y%653 = tail call double @llvm.fmuladd.f64(double %586, double -2.000000e+00, double %610)
,double8B

	full_text

double %586
,double8B

	full_text

double %610
:fadd8B0
.
	full_text!

%654 = fadd double %587, %653
,double8B

	full_text

double %587
,double8B

	full_text

double %653
{call8Bq
o
	full_textb
`
^%655 = tail call double @llvm.fmuladd.f64(double %654, double 0x4030222222222222, double %652)
,double8B

	full_text

double %654
,double8B

	full_text

double %652
:fmul8B0
.
	full_text!

%656 = fmul double %587, %566
,double8B

	full_text

double %587
,double8B

	full_text

double %566
Cfsub8B9
7
	full_text*
(
&%657 = fsub double -0.000000e+00, %656
,double8B

	full_text

double %656
mcall8Bc
a
	full_textT
R
P%658 = tail call double @llvm.fmuladd.f64(double %556, double %610, double %657)
,double8B

	full_text

double %556
,double8B

	full_text

double %610
,double8B

	full_text

double %657
:fsub8B0
.
	full_text!

%659 = fsub double %555, %616
,double8B

	full_text

double %555
,double8B

	full_text

double %616
Qload8BG
E
	full_text8
6
4%660 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fsub8B0
.
	full_text!

%661 = fsub double %659, %660
,double8B

	full_text

double %659
,double8B

	full_text

double %660
:fadd8B0
.
	full_text!

%662 = fadd double %581, %661
,double8B

	full_text

double %581
,double8B

	full_text

double %661
ucall8Bk
i
	full_text\
Z
X%663 = tail call double @llvm.fmuladd.f64(double %662, double 4.000000e-01, double %658)
,double8B

	full_text

double %662
,double8B

	full_text

double %658
vcall8Bl
j
	full_text]
[
Y%664 = tail call double @llvm.fmuladd.f64(double %663, double -5.500000e+00, double %655)
,double8B

	full_text

double %663
,double8B

	full_text

double %655
¦getelementptr8B’

	full_text

}%665 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %575, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %575
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%666 = load double, double* %665, align 8, !tbaa !8
.double*8B

	full_text

double* %665
Pload8BF
D
	full_text7
5
3%667 = load double, double* %68, align 16, !tbaa !8
-double*8B

	full_text

double* %68
vcall8Bl
j
	full_text]
[
Y%668 = tail call double @llvm.fmuladd.f64(double %667, double -2.000000e+00, double %555)
,double8B

	full_text

double %667
,double8B

	full_text

double %555
:fadd8B0
.
	full_text!

%669 = fadd double %660, %668
,double8B

	full_text

double %660
,double8B

	full_text

double %668
ucall8Bk
i
	full_text\
Z
X%670 = tail call double @llvm.fmuladd.f64(double %669, double 1.210000e+02, double %666)
,double8B

	full_text

double %669
,double8B

	full_text

double %666
vcall8Bl
j
	full_text]
[
Y%671 = tail call double @llvm.fmuladd.f64(double %584, double -2.000000e+00, double %612)
,double8B

	full_text

double %584
,double8B

	full_text

double %612
:fadd8B0
.
	full_text!

%672 = fadd double %585, %671
,double8B

	full_text

double %585
,double8B

	full_text

double %671
{call8Bq
o
	full_textb
`
^%673 = tail call double @llvm.fmuladd.f64(double %672, double 0xC0273B645A1CAC07, double %670)
,double8B

	full_text

double %672
,double8B

	full_text

double %670
Bfmul8B8
6
	full_text)
'
%%674 = fmul double %586, 2.000000e+00
,double8B

	full_text

double %586
:fmul8B0
.
	full_text!

%675 = fmul double %586, %674
,double8B

	full_text

double %586
,double8B

	full_text

double %674
Cfsub8B9
7
	full_text*
(
&%676 = fsub double -0.000000e+00, %675
,double8B

	full_text

double %675
mcall8Bc
a
	full_textT
R
P%677 = tail call double @llvm.fmuladd.f64(double %610, double %610, double %676)
,double8B

	full_text

double %610
,double8B

	full_text

double %610
,double8B

	full_text

double %676
mcall8Bc
a
	full_textT
R
P%678 = tail call double @llvm.fmuladd.f64(double %587, double %587, double %677)
,double8B

	full_text

double %587
,double8B

	full_text

double %587
,double8B

	full_text

double %677
{call8Bq
o
	full_textb
`
^%679 = tail call double @llvm.fmuladd.f64(double %678, double 0x4000222222222222, double %673)
,double8B

	full_text

double %678
,double8B

	full_text

double %673
Bfmul8B8
6
	full_text)
'
%%680 = fmul double %667, 2.000000e+00
,double8B

	full_text

double %667
:fmul8B0
.
	full_text!

%681 = fmul double %582, %680
,double8B

	full_text

double %582
,double8B

	full_text

double %680
Cfsub8B9
7
	full_text*
(
&%682 = fsub double -0.000000e+00, %681
,double8B

	full_text

double %681
mcall8Bc
a
	full_textT
R
P%683 = tail call double @llvm.fmuladd.f64(double %555, double %614, double %682)
,double8B

	full_text

double %555
,double8B

	full_text

double %614
,double8B

	full_text

double %682
mcall8Bc
a
	full_textT
R
P%684 = tail call double @llvm.fmuladd.f64(double %660, double %583, double %683)
,double8B

	full_text

double %660
,double8B

	full_text

double %583
,double8B

	full_text

double %683
{call8Bq
o
	full_textb
`
^%685 = tail call double @llvm.fmuladd.f64(double %684, double 0x4037B74BC6A7EF9D, double %679)
,double8B

	full_text

double %684
,double8B

	full_text

double %679
Bfmul8B8
6
	full_text)
'
%%686 = fmul double %616, 4.000000e-01
,double8B

	full_text

double %616
Cfsub8B9
7
	full_text*
(
&%687 = fsub double -0.000000e+00, %686
,double8B

	full_text

double %686
ucall8Bk
i
	full_text\
Z
X%688 = tail call double @llvm.fmuladd.f64(double %555, double 1.400000e+00, double %687)
,double8B

	full_text

double %555
,double8B

	full_text

double %687
Bfmul8B8
6
	full_text)
'
%%689 = fmul double %581, 4.000000e-01
,double8B

	full_text

double %581
Cfsub8B9
7
	full_text*
(
&%690 = fsub double -0.000000e+00, %689
,double8B

	full_text

double %689
ucall8Bk
i
	full_text\
Z
X%691 = tail call double @llvm.fmuladd.f64(double %660, double 1.400000e+00, double %690)
,double8B

	full_text

double %660
,double8B

	full_text

double %690
:fmul8B0
.
	full_text!

%692 = fmul double %587, %691
,double8B

	full_text

double %587
,double8B

	full_text

double %691
Cfsub8B9
7
	full_text*
(
&%693 = fsub double -0.000000e+00, %692
,double8B

	full_text

double %692
mcall8Bc
a
	full_textT
R
P%694 = tail call double @llvm.fmuladd.f64(double %688, double %610, double %693)
,double8B

	full_text

double %688
,double8B

	full_text

double %610
,double8B

	full_text

double %693
vcall8Bl
j
	full_text]
[
Y%695 = tail call double @llvm.fmuladd.f64(double %694, double -5.500000e+00, double %685)
,double8B

	full_text

double %694
,double8B

	full_text

double %685
Qload8BG
E
	full_text8
6
4%696 = load double, double* %552, align 16, !tbaa !8
.double*8B

	full_text

double* %552
vcall8Bl
j
	full_text]
[
Y%697 = tail call double @llvm.fmuladd.f64(double %569, double -4.000000e+00, double %696)
,double8B

	full_text

double %569
,double8B

	full_text

double %696
ucall8Bk
i
	full_text\
Z
X%698 = tail call double @llvm.fmuladd.f64(double %564, double 6.000000e+00, double %697)
,double8B

	full_text

double %564
,double8B

	full_text

double %697
vcall8Bl
j
	full_text]
[
Y%699 = tail call double @llvm.fmuladd.f64(double %559, double -4.000000e+00, double %698)
,double8B

	full_text

double %559
,double8B

	full_text

double %698
Qload8BG
E
	full_text8
6
4%700 = load double, double* %327, align 16, !tbaa !8
.double*8B

	full_text

double* %327
:fadd8B0
.
	full_text!

%701 = fadd double %699, %700
,double8B

	full_text

double %699
,double8B

	full_text

double %700
mcall8Bc
a
	full_textT
R
P%702 = tail call double @llvm.fmuladd.f64(double %321, double %701, double %623)
,double8B

	full_text

double %321
,double8B

	full_text

double %701
,double8B

	full_text

double %623
Pstore8BE
C
	full_text6
4
2store double %702, double* %617, align 8, !tbaa !8
,double8B

	full_text

double %702
.double*8B

	full_text

double* %617
Pload8BF
D
	full_text7
5
3%703 = load double, double* %166, align 8, !tbaa !8
.double*8B

	full_text

double* %166
vcall8Bl
j
	full_text]
[
Y%704 = tail call double @llvm.fmuladd.f64(double %568, double -4.000000e+00, double %703)
,double8B

	full_text

double %568
,double8B

	full_text

double %703
ucall8Bk
i
	full_text\
Z
X%705 = tail call double @llvm.fmuladd.f64(double %563, double 6.000000e+00, double %704)
,double8B

	full_text

double %563
,double8B

	full_text

double %704
vcall8Bl
j
	full_text]
[
Y%706 = tail call double @llvm.fmuladd.f64(double %558, double -4.000000e+00, double %705)
,double8B

	full_text

double %558
,double8B

	full_text

double %705
Pload8BF
D
	full_text7
5
3%707 = load double, double* %106, align 8, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%708 = fadd double %706, %707
,double8B

	full_text

double %706
,double8B

	full_text

double %707
mcall8Bc
a
	full_textT
R
P%709 = tail call double @llvm.fmuladd.f64(double %321, double %708, double %635)
,double8B

	full_text

double %321
,double8B

	full_text

double %708
,double8B

	full_text

double %635
Pstore8BE
C
	full_text6
4
2store double %709, double* %624, align 8, !tbaa !8
,double8B

	full_text

double %709
.double*8B

	full_text

double* %624
Qload8BG
E
	full_text8
6
4%710 = load double, double* %171, align 16, !tbaa !8
.double*8B

	full_text

double* %171
vcall8Bl
j
	full_text]
[
Y%711 = tail call double @llvm.fmuladd.f64(double %567, double -4.000000e+00, double %710)
,double8B

	full_text

double %567
,double8B

	full_text

double %710
ucall8Bk
i
	full_text\
Z
X%712 = tail call double @llvm.fmuladd.f64(double %562, double 6.000000e+00, double %711)
,double8B

	full_text

double %562
,double8B

	full_text

double %711
vcall8Bl
j
	full_text]
[
Y%713 = tail call double @llvm.fmuladd.f64(double %557, double -4.000000e+00, double %712)
,double8B

	full_text

double %557
,double8B

	full_text

double %712
Qload8BG
E
	full_text8
6
4%714 = load double, double* %111, align 16, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%715 = fadd double %713, %714
,double8B

	full_text

double %713
,double8B

	full_text

double %714
mcall8Bc
a
	full_textT
R
P%716 = tail call double @llvm.fmuladd.f64(double %321, double %715, double %647)
,double8B

	full_text

double %321
,double8B

	full_text

double %715
,double8B

	full_text

double %647
Pstore8BE
C
	full_text6
4
2store double %716, double* %636, align 8, !tbaa !8
,double8B

	full_text

double %716
.double*8B

	full_text

double* %636
Pload8BF
D
	full_text7
5
3%717 = load double, double* %176, align 8, !tbaa !8
.double*8B

	full_text

double* %176
vcall8Bl
j
	full_text]
[
Y%718 = tail call double @llvm.fmuladd.f64(double %566, double -4.000000e+00, double %717)
,double8B

	full_text

double %566
,double8B

	full_text

double %717
ucall8Bk
i
	full_text\
Z
X%719 = tail call double @llvm.fmuladd.f64(double %561, double 6.000000e+00, double %718)
,double8B

	full_text

double %561
,double8B

	full_text

double %718
vcall8Bl
j
	full_text]
[
Y%720 = tail call double @llvm.fmuladd.f64(double %556, double -4.000000e+00, double %719)
,double8B

	full_text

double %556
,double8B

	full_text

double %719
Pload8BF
D
	full_text7
5
3%721 = load double, double* %116, align 8, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%722 = fadd double %720, %721
,double8B

	full_text

double %720
,double8B

	full_text

double %721
mcall8Bc
a
	full_textT
R
P%723 = tail call double @llvm.fmuladd.f64(double %321, double %722, double %664)
,double8B

	full_text

double %321
,double8B

	full_text

double %722
,double8B

	full_text

double %664
Pstore8BE
C
	full_text6
4
2store double %723, double* %648, align 8, !tbaa !8
,double8B

	full_text

double %723
.double*8B

	full_text

double* %648
Qload8BG
E
	full_text8
6
4%724 = load double, double* %181, align 16, !tbaa !8
.double*8B

	full_text

double* %181
vcall8Bl
j
	full_text]
[
Y%725 = tail call double @llvm.fmuladd.f64(double %660, double -4.000000e+00, double %724)
,double8B

	full_text

double %660
,double8B

	full_text

double %724
ucall8Bk
i
	full_text\
Z
X%726 = tail call double @llvm.fmuladd.f64(double %667, double 6.000000e+00, double %725)
,double8B

	full_text

double %667
,double8B

	full_text

double %725
vcall8Bl
j
	full_text]
[
Y%727 = tail call double @llvm.fmuladd.f64(double %555, double -4.000000e+00, double %726)
,double8B

	full_text

double %555
,double8B

	full_text

double %726
Qload8BG
E
	full_text8
6
4%728 = load double, double* %121, align 16, !tbaa !8
.double*8B

	full_text

double* %121
:fadd8B0
.
	full_text!

%729 = fadd double %727, %728
,double8B

	full_text

double %727
,double8B

	full_text

double %728
mcall8Bc
a
	full_textT
R
P%730 = tail call double @llvm.fmuladd.f64(double %321, double %729, double %695)
,double8B

	full_text

double %321
,double8B

	full_text

double %729
,double8B

	full_text

double %695
Pstore8BE
C
	full_text6
4
2store double %730, double* %665, align 8, !tbaa !8
,double8B

	full_text

double %730
.double*8B

	full_text

double* %665
:icmp8B0
.
	full_text!

%731 = icmp eq i64 %604, %553
&i648B

	full_text


i64 %604
&i648B

	full_text


i64 %553
Abitcast8B4
2
	full_text%
#
!%732 = bitcast double %569 to i64
,double8B

	full_text

double %569
Abitcast8B4
2
	full_text%
#
!%733 = bitcast double %568 to i64
,double8B

	full_text

double %568
Abitcast8B4
2
	full_text%
#
!%734 = bitcast double %567 to i64
,double8B

	full_text

double %567
Abitcast8B4
2
	full_text%
#
!%735 = bitcast double %566 to i64
,double8B

	full_text

double %566
Abitcast8B4
2
	full_text%
#
!%736 = bitcast double %660 to i64
,double8B

	full_text

double %660
Abitcast8B4
2
	full_text%
#
!%737 = bitcast double %667 to i64
,double8B

	full_text

double %667
Abitcast8B4
2
	full_text%
#
!%738 = bitcast double %555 to i64
,double8B

	full_text

double %555
=br8B5
3
	full_text&
$
"br i1 %731, label %739, label %554
$i18B

	full_text
	
i1 %731
Qstore8BF
D
	full_text7
5
3store double %569, double* %549, align 16, !tbaa !8
,double8B

	full_text

double %569
.double*8B

	full_text

double* %549
Pstore8BE
C
	full_text6
4
2store double %568, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %568
.double*8B

	full_text

double* %163
Qstore8BF
D
	full_text7
5
3store double %567, double* %168, align 16, !tbaa !8
,double8B

	full_text

double %567
.double*8B

	full_text

double* %168
Pstore8BE
C
	full_text6
4
2store double %566, double* %173, align 8, !tbaa !8
,double8B

	full_text

double %566
.double*8B

	full_text

double* %173
Qstore8BF
D
	full_text7
5
3store double %564, double* %550, align 16, !tbaa !8
,double8B

	full_text

double %564
.double*8B

	full_text

double* %550
Ostore8BD
B
	full_text5
3
1store double %563, double* %53, align 8, !tbaa !8
,double8B

	full_text

double %563
-double*8B

	full_text

double* %53
Pstore8BE
C
	full_text6
4
2store double %562, double* %58, align 16, !tbaa !8
,double8B

	full_text

double %562
-double*8B

	full_text

double* %58
Ostore8BD
B
	full_text5
3
1store double %561, double* %63, align 8, !tbaa !8
,double8B

	full_text

double %561
-double*8B

	full_text

double* %63
Qstore8BF
D
	full_text7
5
3store double %559, double* %551, align 16, !tbaa !8
,double8B

	full_text

double %559
.double*8B

	full_text

double* %551
Ostore8BD
B
	full_text5
3
1store double %558, double* %80, align 8, !tbaa !8
,double8B

	full_text

double %558
-double*8B

	full_text

double* %80
Pstore8BE
C
	full_text6
4
2store double %557, double* %85, align 16, !tbaa !8
,double8B

	full_text

double %557
-double*8B

	full_text

double* %85
Ostore8BD
B
	full_text5
3
1store double %556, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %556
-double*8B

	full_text

double* %90
Pstore8BE
C
	full_text6
4
2store double %555, double* %95, align 16, !tbaa !8
,double8B

	full_text

double %555
-double*8B

	full_text

double* %95
Abitcast8B4
2
	full_text%
#
!%740 = bitcast double %561 to i64
,double8B

	full_text

double %561
(br8B 

	full_text

br label %741
Mphi8BD
B
	full_text5
3
1%742 = phi double* [ %544, %538 ], [ %552, %739 ]
.double*8B

	full_text

double* %544
.double*8B

	full_text

double* %552
Iphi8B@
>
	full_text1
/
-%743 = phi i32 [ %543, %538 ], [ %546, %739 ]
&i328B

	full_text


i32 %543
&i328B

	full_text


i32 %546
Lphi8BC
A
	full_text4
2
0%744 = phi double [ %528, %538 ], [ %728, %739 ]
,double8B

	full_text

double %528
,double8B

	full_text

double %728
Lphi8BC
A
	full_text4
2
0%745 = phi double [ %522, %538 ], [ %721, %739 ]
,double8B

	full_text

double %522
,double8B

	full_text

double %721
Lphi8BC
A
	full_text4
2
0%746 = phi double [ %514, %538 ], [ %714, %739 ]
,double8B

	full_text

double %514
,double8B

	full_text

double %714
Lphi8BC
A
	full_text4
2
0%747 = phi double [ %505, %538 ], [ %707, %739 ]
,double8B

	full_text

double %505
,double8B

	full_text

double %707
Lphi8BC
A
	full_text4
2
0%748 = phi double [ %496, %538 ], [ %700, %739 ]
,double8B

	full_text

double %496
,double8B

	full_text

double %700
Lphi8BC
A
	full_text4
2
0%749 = phi double [ %542, %538 ], [ %555, %739 ]
,double8B

	full_text

double %542
,double8B

	full_text

double %555
Iphi8B@
>
	full_text1
/
-%750 = phi i64 [ %541, %538 ], [ %738, %739 ]
&i648B

	full_text


i64 %541
&i648B

	full_text


i64 %738
Lphi8BC
A
	full_text4
2
0%751 = phi double [ %520, %538 ], [ %556, %739 ]
,double8B

	full_text

double %520
,double8B

	full_text

double %556
Lphi8BC
A
	full_text4
2
0%752 = phi double [ %512, %538 ], [ %557, %739 ]
,double8B

	full_text

double %512
,double8B

	full_text

double %557
Lphi8BC
A
	full_text4
2
0%753 = phi double [ %503, %538 ], [ %558, %739 ]
,double8B

	full_text

double %503
,double8B

	full_text

double %558
Lphi8BC
A
	full_text4
2
0%754 = phi double [ %494, %538 ], [ %559, %739 ]
,double8B

	full_text

double %494
,double8B

	full_text

double %559
Iphi8B@
>
	full_text1
/
-%755 = phi i64 [ %537, %538 ], [ %737, %739 ]
&i648B

	full_text


i64 %537
&i648B

	full_text


i64 %737
Lphi8BC
A
	full_text4
2
0%756 = phi double [ %540, %538 ], [ %561, %739 ]
,double8B

	full_text

double %540
,double8B

	full_text

double %561
Iphi8B@
>
	full_text1
/
-%757 = phi i64 [ %539, %538 ], [ %740, %739 ]
&i648B

	full_text


i64 %539
&i648B

	full_text


i64 %740
Lphi8BC
A
	full_text4
2
0%758 = phi double [ %509, %538 ], [ %562, %739 ]
,double8B

	full_text

double %509
,double8B

	full_text

double %562
Lphi8BC
A
	full_text4
2
0%759 = phi double [ %500, %538 ], [ %563, %739 ]
,double8B

	full_text

double %500
,double8B

	full_text

double %563
Lphi8BC
A
	full_text4
2
0%760 = phi double [ %491, %538 ], [ %564, %739 ]
,double8B

	full_text

double %491
,double8B

	full_text

double %564
Iphi8B@
>
	full_text1
/
-%761 = phi i64 [ %536, %538 ], [ %736, %739 ]
&i648B

	full_text


i64 %536
&i648B

	full_text


i64 %736
Iphi8B@
>
	full_text1
/
-%762 = phi i64 [ %535, %538 ], [ %735, %739 ]
&i648B

	full_text


i64 %535
&i648B

	full_text


i64 %735
Iphi8B@
>
	full_text1
/
-%763 = phi i64 [ %534, %538 ], [ %734, %739 ]
&i648B

	full_text


i64 %534
&i648B

	full_text


i64 %734
Iphi8B@
>
	full_text1
/
-%764 = phi i64 [ %533, %538 ], [ %733, %739 ]
&i648B

	full_text


i64 %533
&i648B

	full_text


i64 %733
Iphi8B@
>
	full_text1
/
-%765 = phi i64 [ %532, %538 ], [ %732, %739 ]
&i648B

	full_text


i64 %532
&i648B

	full_text


i64 %732
Lphi8BC
A
	full_text4
2
0%766 = phi double [ %211, %538 ], [ %586, %739 ]
,double8B

	full_text

double %211
,double8B

	full_text

double %586
Lphi8BC
A
	full_text4
2
0%767 = phi double [ %389, %538 ], [ %610, %739 ]
,double8B

	full_text

double %389
,double8B

	full_text

double %610
Lphi8BC
A
	full_text4
2
0%768 = phi double [ %215, %538 ], [ %584, %739 ]
,double8B

	full_text

double %215
,double8B

	full_text

double %584
Lphi8BC
A
	full_text4
2
0%769 = phi double [ %393, %538 ], [ %612, %739 ]
,double8B

	full_text

double %393
,double8B

	full_text

double %612
Lphi8BC
A
	full_text4
2
0%770 = phi double [ %219, %538 ], [ %582, %739 ]
,double8B

	full_text

double %219
,double8B

	full_text

double %582
Lphi8BC
A
	full_text4
2
0%771 = phi double [ %397, %538 ], [ %614, %739 ]
,double8B

	full_text

double %397
,double8B

	full_text

double %614
Lphi8BC
A
	full_text4
2
0%772 = phi double [ %223, %538 ], [ %580, %739 ]
,double8B

	full_text

double %223
,double8B

	full_text

double %580
Lphi8BC
A
	full_text4
2
0%773 = phi double [ %401, %538 ], [ %616, %739 ]
,double8B

	full_text

double %401
,double8B

	full_text

double %616
Lphi8BC
A
	full_text4
2
0%774 = phi double [ %385, %538 ], [ %608, %739 ]
,double8B

	full_text

double %385
,double8B

	full_text

double %608
Lphi8BC
A
	full_text4
2
0%775 = phi double [ %207, %538 ], [ %579, %739 ]
,double8B

	full_text

double %207
,double8B

	full_text

double %579
Lphi8BC
A
	full_text4
2
0%776 = phi double [ %381, %538 ], [ %606, %739 ]
,double8B

	full_text

double %381
,double8B

	full_text

double %606
Lphi8BC
A
	full_text4
2
0%777 = phi double [ %203, %538 ], [ %577, %739 ]
,double8B

	full_text

double %203
,double8B

	full_text

double %577
Kstore8B@
>
	full_text1
/
-store i64 %765, i64* %162, align 16, !tbaa !8
&i648B

	full_text


i64 %765
(i64*8B

	full_text

	i64* %162
Jstore8B?
=
	full_text0
.
,store i64 %764, i64* %167, align 8, !tbaa !8
&i648B

	full_text


i64 %764
(i64*8B

	full_text

	i64* %167
Kstore8B@
>
	full_text1
/
-store i64 %763, i64* %172, align 16, !tbaa !8
&i648B

	full_text


i64 %763
(i64*8B

	full_text

	i64* %172
Jstore8B?
=
	full_text0
.
,store i64 %762, i64* %177, align 8, !tbaa !8
&i648B

	full_text


i64 %762
(i64*8B

	full_text

	i64* %177
Kstore8B@
>
	full_text1
/
-store i64 %761, i64* %182, align 16, !tbaa !8
&i648B

	full_text


i64 %761
(i64*8B

	full_text

	i64* %182
qgetelementptr8B^
\
	full_textO
M
K%778 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %760, double* %778, align 16, !tbaa !8
,double8B

	full_text

double %760
.double*8B

	full_text

double* %778
Pstore8BE
C
	full_text6
4
2store double %759, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %759
.double*8B

	full_text

double* %163
Qstore8BF
D
	full_text7
5
3store double %758, double* %168, align 16, !tbaa !8
,double8B

	full_text

double %758
.double*8B

	full_text

double* %168
Pstore8BE
C
	full_text6
4
2store double %756, double* %173, align 8, !tbaa !8
,double8B

	full_text

double %756
.double*8B

	full_text

double* %173
Kstore8B@
>
	full_text1
/
-store i64 %755, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %755
(i64*8B

	full_text

	i64* %179
qgetelementptr8B^
\
	full_textO
M
K%779 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Qstore8BF
D
	full_text7
5
3store double %754, double* %779, align 16, !tbaa !8
,double8B

	full_text

double %754
.double*8B

	full_text

double* %779
Ostore8BD
B
	full_text5
3
1store double %753, double* %53, align 8, !tbaa !8
,double8B

	full_text

double %753
-double*8B

	full_text

double* %53
Pstore8BE
C
	full_text6
4
2store double %752, double* %58, align 16, !tbaa !8
,double8B

	full_text

double %752
-double*8B

	full_text

double* %58
Ostore8BD
B
	full_text5
3
1store double %751, double* %63, align 8, !tbaa !8
,double8B

	full_text

double %751
-double*8B

	full_text

double* %63
Pstore8BE
C
	full_text6
4
2store double %749, double* %68, align 16, !tbaa !8
,double8B

	full_text

double %749
-double*8B

	full_text

double* %68
qgetelementptr8B^
\
	full_textO
M
K%780 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Qstore8BF
D
	full_text7
5
3store double %748, double* %780, align 16, !tbaa !8
,double8B

	full_text

double %748
.double*8B

	full_text

double* %780
Ostore8BD
B
	full_text5
3
1store double %747, double* %80, align 8, !tbaa !8
,double8B

	full_text

double %747
-double*8B

	full_text

double* %80
Pstore8BE
C
	full_text6
4
2store double %746, double* %85, align 16, !tbaa !8
,double8B

	full_text

double %746
-double*8B

	full_text

double* %85
Ostore8BD
B
	full_text5
3
1store double %745, double* %90, align 8, !tbaa !8
,double8B

	full_text

double %745
-double*8B

	full_text

double* %90
Pstore8BE
C
	full_text6
4
2store double %744, double* %95, align 16, !tbaa !8
,double8B

	full_text

double %744
-double*8B

	full_text

double* %95
6add8B-
+
	full_text

%781 = add nsw i32 %10, -1
8sext8B.
,
	full_text

%782 = sext i32 %781 to i64
&i328B

	full_text


i32 %781
getelementptr8B‰
†
	full_texty
w
u%783 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %782, i64 %42, i64 %44
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Ibitcast8B<
:
	full_text-
+
)%784 = bitcast [5 x double]* %783 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %783
Jload8B@
>
	full_text1
/
-%785 = load i64, i64* %784, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %784
Kstore8B@
>
	full_text1
/
-store i64 %785, i64* %102, align 16, !tbaa !8
&i648B

	full_text


i64 %785
(i64*8B

	full_text

	i64* %102
¥getelementptr8B‘
Ž
	full_text€
~
|%786 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %782, i64 %42, i64 %44, i64 1
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%787 = bitcast double* %786 to i64*
.double*8B

	full_text

double* %786
Jload8B@
>
	full_text1
/
-%788 = load i64, i64* %787, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %787
Jstore8B?
=
	full_text0
.
,store i64 %788, i64* %107, align 8, !tbaa !8
&i648B

	full_text


i64 %788
(i64*8B

	full_text

	i64* %107
¥getelementptr8B‘
Ž
	full_text€
~
|%789 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %782, i64 %42, i64 %44, i64 2
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%790 = bitcast double* %789 to i64*
.double*8B

	full_text

double* %789
Jload8B@
>
	full_text1
/
-%791 = load i64, i64* %790, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %790
Kstore8B@
>
	full_text1
/
-store i64 %791, i64* %112, align 16, !tbaa !8
&i648B

	full_text


i64 %791
(i64*8B

	full_text

	i64* %112
¥getelementptr8B‘
Ž
	full_text€
~
|%792 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %782, i64 %42, i64 %44, i64 3
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%793 = bitcast double* %792 to i64*
.double*8B

	full_text

double* %792
Jload8B@
>
	full_text1
/
-%794 = load i64, i64* %793, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %793
Jstore8B?
=
	full_text0
.
,store i64 %794, i64* %117, align 8, !tbaa !8
&i648B

	full_text


i64 %794
(i64*8B

	full_text

	i64* %117
¥getelementptr8B‘
Ž
	full_text€
~
|%795 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %40, i64 %782, i64 %42, i64 %44, i64 4
U[13 x [13 x [5 x double]]]*8B2
0
	full_text#
!
[13 x [13 x [5 x double]]]* %40
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Cbitcast8B6
4
	full_text'
%
#%796 = bitcast double* %795 to i64*
.double*8B

	full_text

double* %795
Jload8B@
>
	full_text1
/
-%797 = load i64, i64* %796, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %796
Kstore8B@
>
	full_text1
/
-store i64 %797, i64* %122, align 16, !tbaa !8
&i648B

	full_text


i64 %797
(i64*8B

	full_text

	i64* %122
6add8B-
+
	full_text

%798 = add nsw i32 %10, -2
8sext8B.
,
	full_text

%799 = sext i32 %798 to i64
&i328B

	full_text


i32 %798
getelementptr8B|
z
	full_textm
k
i%800 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %34, i64 %799, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %34
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%801 = load double, double* %800, align 8, !tbaa !8
.double*8B

	full_text

double* %800
getelementptr8B|
z
	full_textm
k
i%802 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %35, i64 %799, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %35
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%803 = load double, double* %802, align 8, !tbaa !8
.double*8B

	full_text

double* %802
getelementptr8B|
z
	full_textm
k
i%804 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %36, i64 %799, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %36
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%805 = load double, double* %804, align 8, !tbaa !8
.double*8B

	full_text

double* %804
getelementptr8B|
z
	full_textm
k
i%806 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %37, i64 %799, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %37
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%807 = load double, double* %806, align 8, !tbaa !8
.double*8B

	full_text

double* %806
getelementptr8B|
z
	full_textm
k
i%808 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %38, i64 %799, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %38
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%809 = load double, double* %808, align 8, !tbaa !8
.double*8B

	full_text

double* %808
getelementptr8B|
z
	full_textm
k
i%810 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %39, i64 %799, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %39
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%811 = load double, double* %810, align 8, !tbaa !8
.double*8B

	full_text

double* %810
8sext8B.
,
	full_text

%812 = sext i32 %743 to i64
&i328B

	full_text


i32 %743
¦getelementptr8B’

	full_text

}%813 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %812, i64 %42, i64 %44, i64 0
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %812
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%814 = load double, double* %813, align 8, !tbaa !8
.double*8B

	full_text

double* %813
vcall8Bl
j
	full_text]
[
Y%815 = tail call double @llvm.fmuladd.f64(double %754, double -2.000000e+00, double %748)
,double8B

	full_text

double %754
,double8B

	full_text

double %748
:fadd8B0
.
	full_text!

%816 = fadd double %815, %760
,double8B

	full_text

double %815
,double8B

	full_text

double %760
ucall8Bk
i
	full_text\
Z
X%817 = tail call double @llvm.fmuladd.f64(double %816, double 1.210000e+02, double %814)
,double8B

	full_text

double %816
,double8B

	full_text

double %814
:fsub8B0
.
	full_text!

%818 = fsub double %745, %756
,double8B

	full_text

double %745
,double8B

	full_text

double %756
vcall8Bl
j
	full_text]
[
Y%819 = tail call double @llvm.fmuladd.f64(double %818, double -5.500000e+00, double %817)
,double8B

	full_text

double %818
,double8B

	full_text

double %817
¦getelementptr8B’

	full_text

}%820 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %812, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %812
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%821 = load double, double* %820, align 8, !tbaa !8
.double*8B

	full_text

double* %820
vcall8Bl
j
	full_text]
[
Y%822 = tail call double @llvm.fmuladd.f64(double %753, double -2.000000e+00, double %747)
,double8B

	full_text

double %753
,double8B

	full_text

double %747
:fadd8B0
.
	full_text!

%823 = fadd double %822, %759
,double8B

	full_text

double %822
,double8B

	full_text

double %759
ucall8Bk
i
	full_text\
Z
X%824 = tail call double @llvm.fmuladd.f64(double %823, double 1.210000e+02, double %821)
,double8B

	full_text

double %823
,double8B

	full_text

double %821
vcall8Bl
j
	full_text]
[
Y%825 = tail call double @llvm.fmuladd.f64(double %776, double -2.000000e+00, double %801)
,double8B

	full_text

double %776
,double8B

	full_text

double %801
:fadd8B0
.
	full_text!

%826 = fadd double %777, %825
,double8B

	full_text

double %777
,double8B

	full_text

double %825
{call8Bq
o
	full_textb
`
^%827 = tail call double @llvm.fmuladd.f64(double %826, double 0x4028333333333334, double %824)
,double8B

	full_text

double %826
,double8B

	full_text

double %824
:fmul8B0
.
	full_text!

%828 = fmul double %766, %759
,double8B

	full_text

double %766
,double8B

	full_text

double %759
Cfsub8B9
7
	full_text*
(
&%829 = fsub double -0.000000e+00, %828
,double8B

	full_text

double %828
mcall8Bc
a
	full_textT
R
P%830 = tail call double @llvm.fmuladd.f64(double %747, double %805, double %829)
,double8B

	full_text

double %747
,double8B

	full_text

double %805
,double8B

	full_text

double %829
vcall8Bl
j
	full_text]
[
Y%831 = tail call double @llvm.fmuladd.f64(double %830, double -5.500000e+00, double %827)
,double8B

	full_text

double %830
,double8B

	full_text

double %827
¦getelementptr8B’

	full_text

}%832 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %812, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %812
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%833 = load double, double* %832, align 8, !tbaa !8
.double*8B

	full_text

double* %832
vcall8Bl
j
	full_text]
[
Y%834 = tail call double @llvm.fmuladd.f64(double %752, double -2.000000e+00, double %746)
,double8B

	full_text

double %752
,double8B

	full_text

double %746
:fadd8B0
.
	full_text!

%835 = fadd double %834, %758
,double8B

	full_text

double %834
,double8B

	full_text

double %758
ucall8Bk
i
	full_text\
Z
X%836 = tail call double @llvm.fmuladd.f64(double %835, double 1.210000e+02, double %833)
,double8B

	full_text

double %835
,double8B

	full_text

double %833
vcall8Bl
j
	full_text]
[
Y%837 = tail call double @llvm.fmuladd.f64(double %774, double -2.000000e+00, double %803)
,double8B

	full_text

double %774
,double8B

	full_text

double %803
:fadd8B0
.
	full_text!

%838 = fadd double %775, %837
,double8B

	full_text

double %775
,double8B

	full_text

double %837
{call8Bq
o
	full_textb
`
^%839 = tail call double @llvm.fmuladd.f64(double %838, double 0x4028333333333334, double %836)
,double8B

	full_text

double %838
,double8B

	full_text

double %836
:fmul8B0
.
	full_text!

%840 = fmul double %766, %758
,double8B

	full_text

double %766
,double8B

	full_text

double %758
Cfsub8B9
7
	full_text*
(
&%841 = fsub double -0.000000e+00, %840
,double8B

	full_text

double %840
mcall8Bc
a
	full_textT
R
P%842 = tail call double @llvm.fmuladd.f64(double %746, double %805, double %841)
,double8B

	full_text

double %746
,double8B

	full_text

double %805
,double8B

	full_text

double %841
vcall8Bl
j
	full_text]
[
Y%843 = tail call double @llvm.fmuladd.f64(double %842, double -5.500000e+00, double %839)
,double8B

	full_text

double %842
,double8B

	full_text

double %839
¦getelementptr8B’

	full_text

}%844 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %812, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %812
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%845 = load double, double* %844, align 8, !tbaa !8
.double*8B

	full_text

double* %844
vcall8Bl
j
	full_text]
[
Y%846 = tail call double @llvm.fmuladd.f64(double %751, double -2.000000e+00, double %745)
,double8B

	full_text

double %751
,double8B

	full_text

double %745
:fadd8B0
.
	full_text!

%847 = fadd double %756, %846
,double8B

	full_text

double %756
,double8B

	full_text

double %846
ucall8Bk
i
	full_text\
Z
X%848 = tail call double @llvm.fmuladd.f64(double %847, double 1.210000e+02, double %845)
,double8B

	full_text

double %847
,double8B

	full_text

double %845
vcall8Bl
j
	full_text]
[
Y%849 = tail call double @llvm.fmuladd.f64(double %767, double -2.000000e+00, double %805)
,double8B

	full_text

double %767
,double8B

	full_text

double %805
:fadd8B0
.
	full_text!

%850 = fadd double %766, %849
,double8B

	full_text

double %766
,double8B

	full_text

double %849
{call8Bq
o
	full_textb
`
^%851 = tail call double @llvm.fmuladd.f64(double %850, double 0x4030222222222222, double %848)
,double8B

	full_text

double %850
,double8B

	full_text

double %848
:fmul8B0
.
	full_text!

%852 = fmul double %766, %756
,double8B

	full_text

double %766
,double8B

	full_text

double %756
Cfsub8B9
7
	full_text*
(
&%853 = fsub double -0.000000e+00, %852
,double8B

	full_text

double %852
mcall8Bc
a
	full_textT
R
P%854 = tail call double @llvm.fmuladd.f64(double %745, double %805, double %853)
,double8B

	full_text

double %745
,double8B

	full_text

double %805
,double8B

	full_text

double %853
:fsub8B0
.
	full_text!

%855 = fsub double %744, %811
,double8B

	full_text

double %744
,double8B

	full_text

double %811
Qload8BG
E
	full_text8
6
4%856 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
:fsub8B0
.
	full_text!

%857 = fsub double %855, %856
,double8B

	full_text

double %855
,double8B

	full_text

double %856
:fadd8B0
.
	full_text!

%858 = fadd double %772, %857
,double8B

	full_text

double %772
,double8B

	full_text

double %857
ucall8Bk
i
	full_text\
Z
X%859 = tail call double @llvm.fmuladd.f64(double %858, double 4.000000e-01, double %854)
,double8B

	full_text

double %858
,double8B

	full_text

double %854
vcall8Bl
j
	full_text]
[
Y%860 = tail call double @llvm.fmuladd.f64(double %859, double -5.500000e+00, double %851)
,double8B

	full_text

double %859
,double8B

	full_text

double %851
¦getelementptr8B’

	full_text

}%861 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %812, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %812
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%862 = load double, double* %861, align 8, !tbaa !8
.double*8B

	full_text

double* %861
Pload8BF
D
	full_text7
5
3%863 = load double, double* %68, align 16, !tbaa !8
-double*8B

	full_text

double* %68
vcall8Bl
j
	full_text]
[
Y%864 = tail call double @llvm.fmuladd.f64(double %863, double -2.000000e+00, double %744)
,double8B

	full_text

double %863
,double8B

	full_text

double %744
:fadd8B0
.
	full_text!

%865 = fadd double %856, %864
,double8B

	full_text

double %856
,double8B

	full_text

double %864
ucall8Bk
i
	full_text\
Z
X%866 = tail call double @llvm.fmuladd.f64(double %865, double 1.210000e+02, double %862)
,double8B

	full_text

double %865
,double8B

	full_text

double %862
vcall8Bl
j
	full_text]
[
Y%867 = tail call double @llvm.fmuladd.f64(double %769, double -2.000000e+00, double %807)
,double8B

	full_text

double %769
,double8B

	full_text

double %807
:fadd8B0
.
	full_text!

%868 = fadd double %768, %867
,double8B

	full_text

double %768
,double8B

	full_text

double %867
{call8Bq
o
	full_textb
`
^%869 = tail call double @llvm.fmuladd.f64(double %868, double 0xC0273B645A1CAC07, double %866)
,double8B

	full_text

double %868
,double8B

	full_text

double %866
Bfmul8B8
6
	full_text)
'
%%870 = fmul double %767, 2.000000e+00
,double8B

	full_text

double %767
:fmul8B0
.
	full_text!

%871 = fmul double %767, %870
,double8B

	full_text

double %767
,double8B

	full_text

double %870
Cfsub8B9
7
	full_text*
(
&%872 = fsub double -0.000000e+00, %871
,double8B

	full_text

double %871
mcall8Bc
a
	full_textT
R
P%873 = tail call double @llvm.fmuladd.f64(double %805, double %805, double %872)
,double8B

	full_text

double %805
,double8B

	full_text

double %805
,double8B

	full_text

double %872
mcall8Bc
a
	full_textT
R
P%874 = tail call double @llvm.fmuladd.f64(double %766, double %766, double %873)
,double8B

	full_text

double %766
,double8B

	full_text

double %766
,double8B

	full_text

double %873
{call8Bq
o
	full_textb
`
^%875 = tail call double @llvm.fmuladd.f64(double %874, double 0x4000222222222222, double %869)
,double8B

	full_text

double %874
,double8B

	full_text

double %869
Bfmul8B8
6
	full_text)
'
%%876 = fmul double %863, 2.000000e+00
,double8B

	full_text

double %863
:fmul8B0
.
	full_text!

%877 = fmul double %771, %876
,double8B

	full_text

double %771
,double8B

	full_text

double %876
Cfsub8B9
7
	full_text*
(
&%878 = fsub double -0.000000e+00, %877
,double8B

	full_text

double %877
mcall8Bc
a
	full_textT
R
P%879 = tail call double @llvm.fmuladd.f64(double %744, double %809, double %878)
,double8B

	full_text

double %744
,double8B

	full_text

double %809
,double8B

	full_text

double %878
mcall8Bc
a
	full_textT
R
P%880 = tail call double @llvm.fmuladd.f64(double %856, double %770, double %879)
,double8B

	full_text

double %856
,double8B

	full_text

double %770
,double8B

	full_text

double %879
{call8Bq
o
	full_textb
`
^%881 = tail call double @llvm.fmuladd.f64(double %880, double 0x4037B74BC6A7EF9D, double %875)
,double8B

	full_text

double %880
,double8B

	full_text

double %875
Bfmul8B8
6
	full_text)
'
%%882 = fmul double %811, 4.000000e-01
,double8B

	full_text

double %811
Cfsub8B9
7
	full_text*
(
&%883 = fsub double -0.000000e+00, %882
,double8B

	full_text

double %882
ucall8Bk
i
	full_text\
Z
X%884 = tail call double @llvm.fmuladd.f64(double %744, double 1.400000e+00, double %883)
,double8B

	full_text

double %744
,double8B

	full_text

double %883
Bfmul8B8
6
	full_text)
'
%%885 = fmul double %772, 4.000000e-01
,double8B

	full_text

double %772
Cfsub8B9
7
	full_text*
(
&%886 = fsub double -0.000000e+00, %885
,double8B

	full_text

double %885
ucall8Bk
i
	full_text\
Z
X%887 = tail call double @llvm.fmuladd.f64(double %856, double 1.400000e+00, double %886)
,double8B

	full_text

double %856
,double8B

	full_text

double %886
:fmul8B0
.
	full_text!

%888 = fmul double %766, %887
,double8B

	full_text

double %766
,double8B

	full_text

double %887
Cfsub8B9
7
	full_text*
(
&%889 = fsub double -0.000000e+00, %888
,double8B

	full_text

double %888
mcall8Bc
a
	full_textT
R
P%890 = tail call double @llvm.fmuladd.f64(double %884, double %805, double %889)
,double8B

	full_text

double %884
,double8B

	full_text

double %805
,double8B

	full_text

double %889
vcall8Bl
j
	full_text]
[
Y%891 = tail call double @llvm.fmuladd.f64(double %890, double -5.500000e+00, double %881)
,double8B

	full_text

double %890
,double8B

	full_text

double %881
Pload8BF
D
	full_text7
5
3%892 = load double, double* %742, align 8, !tbaa !8
.double*8B

	full_text

double* %742
Qload8BG
E
	full_text8
6
4%893 = load double, double* %159, align 16, !tbaa !8
.double*8B

	full_text

double* %159
vcall8Bl
j
	full_text]
[
Y%894 = tail call double @llvm.fmuladd.f64(double %893, double -4.000000e+00, double %892)
,double8B

	full_text

double %893
,double8B

	full_text

double %892
Pload8BF
D
	full_text7
5
3%895 = load double, double* %48, align 16, !tbaa !8
-double*8B

	full_text

double* %48
ucall8Bk
i
	full_text\
Z
X%896 = tail call double @llvm.fmuladd.f64(double %895, double 6.000000e+00, double %894)
,double8B

	full_text

double %895
,double8B

	full_text

double %894
Pload8BF
D
	full_text7
5
3%897 = load double, double* %75, align 16, !tbaa !8
-double*8B

	full_text

double* %75
vcall8Bl
j
	full_text]
[
Y%898 = tail call double @llvm.fmuladd.f64(double %897, double -4.000000e+00, double %896)
,double8B

	full_text

double %897
,double8B

	full_text

double %896
mcall8Bc
a
	full_textT
R
P%899 = tail call double @llvm.fmuladd.f64(double %321, double %898, double %819)
,double8B

	full_text

double %321
,double8B

	full_text

double %898
,double8B

	full_text

double %819
Pstore8BE
C
	full_text6
4
2store double %899, double* %813, align 8, !tbaa !8
,double8B

	full_text

double %899
.double*8B

	full_text

double* %813
Pload8BF
D
	full_text7
5
3%900 = load double, double* %166, align 8, !tbaa !8
.double*8B

	full_text

double* %166
Pload8BF
D
	full_text7
5
3%901 = load double, double* %163, align 8, !tbaa !8
.double*8B

	full_text

double* %163
vcall8Bl
j
	full_text]
[
Y%902 = tail call double @llvm.fmuladd.f64(double %901, double -4.000000e+00, double %900)
,double8B

	full_text

double %901
,double8B

	full_text

double %900
Oload8BE
C
	full_text6
4
2%903 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
ucall8Bk
i
	full_text\
Z
X%904 = tail call double @llvm.fmuladd.f64(double %903, double 6.000000e+00, double %902)
,double8B

	full_text

double %903
,double8B

	full_text

double %902
Oload8BE
C
	full_text6
4
2%905 = load double, double* %80, align 8, !tbaa !8
-double*8B

	full_text

double* %80
vcall8Bl
j
	full_text]
[
Y%906 = tail call double @llvm.fmuladd.f64(double %905, double -4.000000e+00, double %904)
,double8B

	full_text

double %905
,double8B

	full_text

double %904
mcall8Bc
a
	full_textT
R
P%907 = tail call double @llvm.fmuladd.f64(double %321, double %906, double %831)
,double8B

	full_text

double %321
,double8B

	full_text

double %906
,double8B

	full_text

double %831
Pstore8BE
C
	full_text6
4
2store double %907, double* %820, align 8, !tbaa !8
,double8B

	full_text

double %907
.double*8B

	full_text

double* %820
Qload8BG
E
	full_text8
6
4%908 = load double, double* %171, align 16, !tbaa !8
.double*8B

	full_text

double* %171
Qload8BG
E
	full_text8
6
4%909 = load double, double* %168, align 16, !tbaa !8
.double*8B

	full_text

double* %168
vcall8Bl
j
	full_text]
[
Y%910 = tail call double @llvm.fmuladd.f64(double %909, double -4.000000e+00, double %908)
,double8B

	full_text

double %909
,double8B

	full_text

double %908
Pload8BF
D
	full_text7
5
3%911 = load double, double* %58, align 16, !tbaa !8
-double*8B

	full_text

double* %58
ucall8Bk
i
	full_text\
Z
X%912 = tail call double @llvm.fmuladd.f64(double %911, double 6.000000e+00, double %910)
,double8B

	full_text

double %911
,double8B

	full_text

double %910
Pload8BF
D
	full_text7
5
3%913 = load double, double* %85, align 16, !tbaa !8
-double*8B

	full_text

double* %85
vcall8Bl
j
	full_text]
[
Y%914 = tail call double @llvm.fmuladd.f64(double %913, double -4.000000e+00, double %912)
,double8B

	full_text

double %913
,double8B

	full_text

double %912
mcall8Bc
a
	full_textT
R
P%915 = tail call double @llvm.fmuladd.f64(double %321, double %914, double %843)
,double8B

	full_text

double %321
,double8B

	full_text

double %914
,double8B

	full_text

double %843
Pstore8BE
C
	full_text6
4
2store double %915, double* %832, align 8, !tbaa !8
,double8B

	full_text

double %915
.double*8B

	full_text

double* %832
Pload8BF
D
	full_text7
5
3%916 = load double, double* %176, align 8, !tbaa !8
.double*8B

	full_text

double* %176
Pload8BF
D
	full_text7
5
3%917 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
vcall8Bl
j
	full_text]
[
Y%918 = tail call double @llvm.fmuladd.f64(double %917, double -4.000000e+00, double %916)
,double8B

	full_text

double %917
,double8B

	full_text

double %916
Oload8BE
C
	full_text6
4
2%919 = load double, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
ucall8Bk
i
	full_text\
Z
X%920 = tail call double @llvm.fmuladd.f64(double %919, double 6.000000e+00, double %918)
,double8B

	full_text

double %919
,double8B

	full_text

double %918
Oload8BE
C
	full_text6
4
2%921 = load double, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
vcall8Bl
j
	full_text]
[
Y%922 = tail call double @llvm.fmuladd.f64(double %921, double -4.000000e+00, double %920)
,double8B

	full_text

double %921
,double8B

	full_text

double %920
mcall8Bc
a
	full_textT
R
P%923 = tail call double @llvm.fmuladd.f64(double %321, double %922, double %860)
,double8B

	full_text

double %321
,double8B

	full_text

double %922
,double8B

	full_text

double %860
Pstore8BE
C
	full_text6
4
2store double %923, double* %844, align 8, !tbaa !8
,double8B

	full_text

double %923
.double*8B

	full_text

double* %844
Qload8BG
E
	full_text8
6
4%924 = load double, double* %181, align 16, !tbaa !8
.double*8B

	full_text

double* %181
vcall8Bl
j
	full_text]
[
Y%925 = tail call double @llvm.fmuladd.f64(double %856, double -4.000000e+00, double %924)
,double8B

	full_text

double %856
,double8B

	full_text

double %924
ucall8Bk
i
	full_text\
Z
X%926 = tail call double @llvm.fmuladd.f64(double %863, double 6.000000e+00, double %925)
,double8B

	full_text

double %863
,double8B

	full_text

double %925
Pload8BF
D
	full_text7
5
3%927 = load double, double* %95, align 16, !tbaa !8
-double*8B

	full_text

double* %95
vcall8Bl
j
	full_text]
[
Y%928 = tail call double @llvm.fmuladd.f64(double %927, double -4.000000e+00, double %926)
,double8B

	full_text

double %927
,double8B

	full_text

double %926
mcall8Bc
a
	full_textT
R
P%929 = tail call double @llvm.fmuladd.f64(double %321, double %928, double %891)
,double8B

	full_text

double %321
,double8B

	full_text

double %928
,double8B

	full_text

double %891
Pstore8BE
C
	full_text6
4
2store double %929, double* %861, align 8, !tbaa !8
,double8B

	full_text

double %929
.double*8B

	full_text

double* %861
qgetelementptr8B^
\
	full_textO
M
K%930 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Qstore8BF
D
	full_text7
5
3store double %760, double* %930, align 16, !tbaa !8
,double8B

	full_text

double %760
.double*8B

	full_text

double* %930
Pstore8BE
C
	full_text6
4
2store double %759, double* %166, align 8, !tbaa !8
,double8B

	full_text

double %759
.double*8B

	full_text

double* %166
Qstore8BF
D
	full_text7
5
3store double %758, double* %171, align 16, !tbaa !8
,double8B

	full_text

double %758
.double*8B

	full_text

double* %171
Jstore8B?
=
	full_text0
.
,store i64 %757, i64* %177, align 8, !tbaa !8
&i648B

	full_text


i64 %757
(i64*8B

	full_text

	i64* %177
Kstore8B@
>
	full_text1
/
-store i64 %755, i64* %182, align 16, !tbaa !8
&i648B

	full_text


i64 %755
(i64*8B

	full_text

	i64* %182
qgetelementptr8B^
\
	full_textO
M
K%931 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %754, double* %931, align 16, !tbaa !8
,double8B

	full_text

double %754
.double*8B

	full_text

double* %931
Pstore8BE
C
	full_text6
4
2store double %753, double* %163, align 8, !tbaa !8
,double8B

	full_text

double %753
.double*8B

	full_text

double* %163
Qstore8BF
D
	full_text7
5
3store double %752, double* %168, align 16, !tbaa !8
,double8B

	full_text

double %752
.double*8B

	full_text

double* %168
Pstore8BE
C
	full_text6
4
2store double %751, double* %173, align 8, !tbaa !8
,double8B

	full_text

double %751
.double*8B

	full_text

double* %173
Kstore8B@
>
	full_text1
/
-store i64 %750, i64* %179, align 16, !tbaa !8
&i648B

	full_text


i64 %750
(i64*8B

	full_text

	i64* %179
qgetelementptr8B^
\
	full_textO
M
K%932 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Qstore8BF
D
	full_text7
5
3store double %748, double* %932, align 16, !tbaa !8
,double8B

	full_text

double %748
.double*8B

	full_text

double* %932
Ostore8BD
B
	full_text5
3
1store double %747, double* %53, align 8, !tbaa !8
,double8B

	full_text

double %747
-double*8B

	full_text

double* %53
Pstore8BE
C
	full_text6
4
2store double %746, double* %58, align 16, !tbaa !8
,double8B

	full_text

double %746
-double*8B

	full_text

double* %58
Ostore8BD
B
	full_text5
3
1store double %745, double* %63, align 8, !tbaa !8
,double8B

	full_text

double %745
-double*8B

	full_text

double* %63
Pstore8BE
C
	full_text6
4
2store double %744, double* %68, align 16, !tbaa !8
,double8B

	full_text

double %744
-double*8B

	full_text

double* %68
Jstore8B?
=
	full_text0
.
,store i64 %785, i64* %76, align 16, !tbaa !8
&i648B

	full_text


i64 %785
'i64*8B

	full_text


i64* %76
Istore8B>
<
	full_text/
-
+store i64 %788, i64* %81, align 8, !tbaa !8
&i648B

	full_text


i64 %788
'i64*8B

	full_text


i64* %81
Jstore8B?
=
	full_text0
.
,store i64 %791, i64* %86, align 16, !tbaa !8
&i648B

	full_text


i64 %791
'i64*8B

	full_text


i64* %86
Istore8B>
<
	full_text/
-
+store i64 %794, i64* %91, align 8, !tbaa !8
&i648B

	full_text


i64 %794
'i64*8B

	full_text


i64* %91
Jstore8B?
=
	full_text0
.
,store i64 %797, i64* %96, align 16, !tbaa !8
&i648B

	full_text


i64 %797
'i64*8B

	full_text


i64* %96
getelementptr8B|
z
	full_textm
k
i%933 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %34, i64 %782, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %34
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%934 = load double, double* %933, align 8, !tbaa !8
.double*8B

	full_text

double* %933
getelementptr8B|
z
	full_textm
k
i%935 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %35, i64 %782, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %35
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%936 = load double, double* %935, align 8, !tbaa !8
.double*8B

	full_text

double* %935
getelementptr8B|
z
	full_textm
k
i%937 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %36, i64 %782, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %36
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%938 = load double, double* %937, align 8, !tbaa !8
.double*8B

	full_text

double* %937
getelementptr8B|
z
	full_textm
k
i%939 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %37, i64 %782, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %37
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%940 = load double, double* %939, align 8, !tbaa !8
.double*8B

	full_text

double* %939
getelementptr8B|
z
	full_textm
k
i%941 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %38, i64 %782, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %38
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%942 = load double, double* %941, align 8, !tbaa !8
.double*8B

	full_text

double* %941
getelementptr8B|
z
	full_textm
k
i%943 = getelementptr inbounds [13 x [13 x double]], [13 x [13 x double]]* %39, i64 %782, i64 %42, i64 %44
I[13 x [13 x double]]*8B,
*
	full_text

[13 x [13 x double]]* %39
&i648B

	full_text


i64 %782
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%944 = load double, double* %943, align 8, !tbaa !8
.double*8B

	full_text

double* %943
¦getelementptr8B’

	full_text

}%945 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %799, i64 %42, i64 %44, i64 0
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%946 = load double, double* %945, align 8, !tbaa !8
.double*8B

	full_text

double* %945
Abitcast8B4
2
	full_text%
#
!%947 = bitcast i64 %785 to double
&i648B

	full_text


i64 %785
vcall8Bl
j
	full_text]
[
Y%948 = tail call double @llvm.fmuladd.f64(double %748, double -2.000000e+00, double %947)
,double8B

	full_text

double %748
,double8B

	full_text

double %947
:fadd8B0
.
	full_text!

%949 = fadd double %948, %754
,double8B

	full_text

double %948
,double8B

	full_text

double %754
ucall8Bk
i
	full_text\
Z
X%950 = tail call double @llvm.fmuladd.f64(double %949, double 1.210000e+02, double %946)
,double8B

	full_text

double %949
,double8B

	full_text

double %946
Abitcast8B4
2
	full_text%
#
!%951 = bitcast i64 %794 to double
&i648B

	full_text


i64 %794
:fsub8B0
.
	full_text!

%952 = fsub double %951, %751
,double8B

	full_text

double %951
,double8B

	full_text

double %751
vcall8Bl
j
	full_text]
[
Y%953 = tail call double @llvm.fmuladd.f64(double %952, double -5.500000e+00, double %950)
,double8B

	full_text

double %952
,double8B

	full_text

double %950
¦getelementptr8B’

	full_text

}%954 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %799, i64 %42, i64 %44, i64 1
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%955 = load double, double* %954, align 8, !tbaa !8
.double*8B

	full_text

double* %954
Abitcast8B4
2
	full_text%
#
!%956 = bitcast i64 %788 to double
&i648B

	full_text


i64 %788
vcall8Bl
j
	full_text]
[
Y%957 = tail call double @llvm.fmuladd.f64(double %747, double -2.000000e+00, double %956)
,double8B

	full_text

double %747
,double8B

	full_text

double %956
:fadd8B0
.
	full_text!

%958 = fadd double %957, %753
,double8B

	full_text

double %957
,double8B

	full_text

double %753
ucall8Bk
i
	full_text\
Z
X%959 = tail call double @llvm.fmuladd.f64(double %958, double 1.210000e+02, double %955)
,double8B

	full_text

double %958
,double8B

	full_text

double %955
vcall8Bl
j
	full_text]
[
Y%960 = tail call double @llvm.fmuladd.f64(double %801, double -2.000000e+00, double %934)
,double8B

	full_text

double %801
,double8B

	full_text

double %934
:fadd8B0
.
	full_text!

%961 = fadd double %776, %960
,double8B

	full_text

double %776
,double8B

	full_text

double %960
{call8Bq
o
	full_textb
`
^%962 = tail call double @llvm.fmuladd.f64(double %961, double 0x4028333333333334, double %959)
,double8B

	full_text

double %961
,double8B

	full_text

double %959
:fmul8B0
.
	full_text!

%963 = fmul double %767, %753
,double8B

	full_text

double %767
,double8B

	full_text

double %753
Cfsub8B9
7
	full_text*
(
&%964 = fsub double -0.000000e+00, %963
,double8B

	full_text

double %963
mcall8Bc
a
	full_textT
R
P%965 = tail call double @llvm.fmuladd.f64(double %956, double %938, double %964)
,double8B

	full_text

double %956
,double8B

	full_text

double %938
,double8B

	full_text

double %964
vcall8Bl
j
	full_text]
[
Y%966 = tail call double @llvm.fmuladd.f64(double %965, double -5.500000e+00, double %962)
,double8B

	full_text

double %965
,double8B

	full_text

double %962
¦getelementptr8B’

	full_text

}%967 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %799, i64 %42, i64 %44, i64 2
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%968 = load double, double* %967, align 8, !tbaa !8
.double*8B

	full_text

double* %967
Abitcast8B4
2
	full_text%
#
!%969 = bitcast i64 %791 to double
&i648B

	full_text


i64 %791
vcall8Bl
j
	full_text]
[
Y%970 = tail call double @llvm.fmuladd.f64(double %746, double -2.000000e+00, double %969)
,double8B

	full_text

double %746
,double8B

	full_text

double %969
:fadd8B0
.
	full_text!

%971 = fadd double %970, %752
,double8B

	full_text

double %970
,double8B

	full_text

double %752
ucall8Bk
i
	full_text\
Z
X%972 = tail call double @llvm.fmuladd.f64(double %971, double 1.210000e+02, double %968)
,double8B

	full_text

double %971
,double8B

	full_text

double %968
vcall8Bl
j
	full_text]
[
Y%973 = tail call double @llvm.fmuladd.f64(double %803, double -2.000000e+00, double %936)
,double8B

	full_text

double %803
,double8B

	full_text

double %936
:fadd8B0
.
	full_text!

%974 = fadd double %774, %973
,double8B

	full_text

double %774
,double8B

	full_text

double %973
{call8Bq
o
	full_textb
`
^%975 = tail call double @llvm.fmuladd.f64(double %974, double 0x4028333333333334, double %972)
,double8B

	full_text

double %974
,double8B

	full_text

double %972
:fmul8B0
.
	full_text!

%976 = fmul double %767, %752
,double8B

	full_text

double %767
,double8B

	full_text

double %752
Cfsub8B9
7
	full_text*
(
&%977 = fsub double -0.000000e+00, %976
,double8B

	full_text

double %976
mcall8Bc
a
	full_textT
R
P%978 = tail call double @llvm.fmuladd.f64(double %969, double %938, double %977)
,double8B

	full_text

double %969
,double8B

	full_text

double %938
,double8B

	full_text

double %977
vcall8Bl
j
	full_text]
[
Y%979 = tail call double @llvm.fmuladd.f64(double %978, double -5.500000e+00, double %975)
,double8B

	full_text

double %978
,double8B

	full_text

double %975
¦getelementptr8B’

	full_text

}%980 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %799, i64 %42, i64 %44, i64 3
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%981 = load double, double* %980, align 8, !tbaa !8
.double*8B

	full_text

double* %980
vcall8Bl
j
	full_text]
[
Y%982 = tail call double @llvm.fmuladd.f64(double %745, double -2.000000e+00, double %951)
,double8B

	full_text

double %745
,double8B

	full_text

double %951
:fadd8B0
.
	full_text!

%983 = fadd double %751, %982
,double8B

	full_text

double %751
,double8B

	full_text

double %982
ucall8Bk
i
	full_text\
Z
X%984 = tail call double @llvm.fmuladd.f64(double %983, double 1.210000e+02, double %981)
,double8B

	full_text

double %983
,double8B

	full_text

double %981
vcall8Bl
j
	full_text]
[
Y%985 = tail call double @llvm.fmuladd.f64(double %805, double -2.000000e+00, double %938)
,double8B

	full_text

double %805
,double8B

	full_text

double %938
:fadd8B0
.
	full_text!

%986 = fadd double %767, %985
,double8B

	full_text

double %767
,double8B

	full_text

double %985
{call8Bq
o
	full_textb
`
^%987 = tail call double @llvm.fmuladd.f64(double %986, double 0x4030222222222222, double %984)
,double8B

	full_text

double %986
,double8B

	full_text

double %984
:fmul8B0
.
	full_text!

%988 = fmul double %767, %751
,double8B

	full_text

double %767
,double8B

	full_text

double %751
Cfsub8B9
7
	full_text*
(
&%989 = fsub double -0.000000e+00, %988
,double8B

	full_text

double %988
mcall8Bc
a
	full_textT
R
P%990 = tail call double @llvm.fmuladd.f64(double %951, double %938, double %989)
,double8B

	full_text

double %951
,double8B

	full_text

double %938
,double8B

	full_text

double %989
Abitcast8B4
2
	full_text%
#
!%991 = bitcast i64 %797 to double
&i648B

	full_text


i64 %797
:fsub8B0
.
	full_text!

%992 = fsub double %991, %944
,double8B

	full_text

double %991
,double8B

	full_text

double %944
:fsub8B0
.
	full_text!

%993 = fsub double %992, %749
,double8B

	full_text

double %992
,double8B

	full_text

double %749
:fadd8B0
.
	full_text!

%994 = fadd double %773, %993
,double8B

	full_text

double %773
,double8B

	full_text

double %993
ucall8Bk
i
	full_text\
Z
X%995 = tail call double @llvm.fmuladd.f64(double %994, double 4.000000e-01, double %990)
,double8B

	full_text

double %994
,double8B

	full_text

double %990
vcall8Bl
j
	full_text]
[
Y%996 = tail call double @llvm.fmuladd.f64(double %995, double -5.500000e+00, double %987)
,double8B

	full_text

double %995
,double8B

	full_text

double %987
¦getelementptr8B’

	full_text

}%997 = getelementptr inbounds [13 x [13 x [5 x double]]], [13 x [13 x [5 x double]]]* %360, i64 %799, i64 %42, i64 %44, i64 4
V[13 x [13 x [5 x double]]]*8B3
1
	full_text$
"
 [13 x [13 x [5 x double]]]* %360
&i648B

	full_text


i64 %799
%i648B

	full_text
	
i64 %42
%i648B

	full_text
	
i64 %44
Pload8BF
D
	full_text7
5
3%998 = load double, double* %997, align 8, !tbaa !8
.double*8B

	full_text

double* %997
vcall8Bl
j
	full_text]
[
Y%999 = tail call double @llvm.fmuladd.f64(double %744, double -2.000000e+00, double %991)
,double8B

	full_text

double %744
,double8B

	full_text

double %991
;fadd8B1
/
	full_text"
 
%1000 = fadd double %749, %999
,double8B

	full_text

double %749
,double8B

	full_text

double %999
wcall8Bm
k
	full_text^
\
Z%1001 = tail call double @llvm.fmuladd.f64(double %1000, double 1.210000e+02, double %998)
-double8B

	full_text

double %1000
,double8B

	full_text

double %998
wcall8Bm
k
	full_text^
\
Z%1002 = tail call double @llvm.fmuladd.f64(double %807, double -2.000000e+00, double %940)
,double8B

	full_text

double %807
,double8B

	full_text

double %940
<fadd8B2
0
	full_text#
!
%1003 = fadd double %769, %1002
,double8B

	full_text

double %769
-double8B

	full_text

double %1002
~call8Bt
r
	full_texte
c
a%1004 = tail call double @llvm.fmuladd.f64(double %1003, double 0xC0273B645A1CAC07, double %1001)
-double8B

	full_text

double %1003
-double8B

	full_text

double %1001
Cfmul8B9
7
	full_text*
(
&%1005 = fmul double %805, 2.000000e+00
,double8B

	full_text

double %805
<fmul8B2
0
	full_text#
!
%1006 = fmul double %805, %1005
,double8B

	full_text

double %805
-double8B

	full_text

double %1005
Efsub8B;
9
	full_text,
*
(%1007 = fsub double -0.000000e+00, %1006
-double8B

	full_text

double %1006
ocall8Be
c
	full_textV
T
R%1008 = tail call double @llvm.fmuladd.f64(double %938, double %938, double %1007)
,double8B

	full_text

double %938
,double8B

	full_text

double %938
-double8B

	full_text

double %1007
ocall8Be
c
	full_textV
T
R%1009 = tail call double @llvm.fmuladd.f64(double %767, double %767, double %1008)
,double8B

	full_text

double %767
,double8B

	full_text

double %767
-double8B

	full_text

double %1008
~call8Bt
r
	full_texte
c
a%1010 = tail call double @llvm.fmuladd.f64(double %1009, double 0x4000222222222222, double %1004)
-double8B

	full_text

double %1009
-double8B

	full_text

double %1004
Cfmul8B9
7
	full_text*
(
&%1011 = fmul double %744, 2.000000e+00
,double8B

	full_text

double %744
<fmul8B2
0
	full_text#
!
%1012 = fmul double %809, %1011
,double8B

	full_text

double %809
-double8B

	full_text

double %1011
Efsub8B;
9
	full_text,
*
(%1013 = fsub double -0.000000e+00, %1012
-double8B

	full_text

double %1012
ocall8Be
c
	full_textV
T
R%1014 = tail call double @llvm.fmuladd.f64(double %991, double %942, double %1013)
,double8B

	full_text

double %991
,double8B

	full_text

double %942
-double8B

	full_text

double %1013
ocall8Be
c
	full_textV
T
R%1015 = tail call double @llvm.fmuladd.f64(double %749, double %771, double %1014)
,double8B

	full_text

double %749
,double8B

	full_text

double %771
-double8B

	full_text

double %1014
~call8Bt
r
	full_texte
c
a%1016 = tail call double @llvm.fmuladd.f64(double %1015, double 0x4037B74BC6A7EF9D, double %1010)
-double8B

	full_text

double %1015
-double8B

	full_text

double %1010
Cfmul8B9
7
	full_text*
(
&%1017 = fmul double %944, 4.000000e-01
,double8B

	full_text

double %944
Efsub8B;
9
	full_text,
*
(%1018 = fsub double -0.000000e+00, %1017
-double8B

	full_text

double %1017
wcall8Bm
k
	full_text^
\
Z%1019 = tail call double @llvm.fmuladd.f64(double %991, double 1.400000e+00, double %1018)
,double8B

	full_text

double %991
-double8B

	full_text

double %1018
Cfmul8B9
7
	full_text*
(
&%1020 = fmul double %773, 4.000000e-01
,double8B

	full_text

double %773
Efsub8B;
9
	full_text,
*
(%1021 = fsub double -0.000000e+00, %1020
-double8B

	full_text

double %1020
wcall8Bm
k
	full_text^
\
Z%1022 = tail call double @llvm.fmuladd.f64(double %749, double 1.400000e+00, double %1021)
,double8B

	full_text

double %749
-double8B

	full_text

double %1021
<fmul8B2
0
	full_text#
!
%1023 = fmul double %767, %1022
,double8B

	full_text

double %767
-double8B

	full_text

double %1022
Efsub8B;
9
	full_text,
*
(%1024 = fsub double -0.000000e+00, %1023
-double8B

	full_text

double %1023
pcall8Bf
d
	full_textW
U
S%1025 = tail call double @llvm.fmuladd.f64(double %1019, double %938, double %1024)
-double8B

	full_text

double %1019
,double8B

	full_text

double %938
-double8B

	full_text

double %1024
ycall8Bo
m
	full_text`
^
\%1026 = tail call double @llvm.fmuladd.f64(double %1025, double -5.500000e+00, double %1016)
-double8B

	full_text

double %1025
-double8B

	full_text

double %1016
Qload8BG
E
	full_text8
6
4%1027 = load double, double* %742, align 8, !tbaa !8
.double*8B

	full_text

double* %742
Rload8BH
F
	full_text9
7
5%1028 = load double, double* %159, align 16, !tbaa !8
.double*8B

	full_text

double* %159
ycall8Bo
m
	full_text`
^
\%1029 = tail call double @llvm.fmuladd.f64(double %1028, double -4.000000e+00, double %1027)
-double8B

	full_text

double %1028
-double8B

	full_text

double %1027
Qload8BG
E
	full_text8
6
4%1030 = load double, double* %48, align 16, !tbaa !8
-double*8B

	full_text

double* %48
xcall8Bn
l
	full_text_
]
[%1031 = tail call double @llvm.fmuladd.f64(double %1030, double 5.000000e+00, double %1029)
-double8B

	full_text

double %1030
-double8B

	full_text

double %1029
ocall8Be
c
	full_textV
T
R%1032 = tail call double @llvm.fmuladd.f64(double %321, double %1031, double %953)
,double8B

	full_text

double %321
-double8B

	full_text

double %1031
,double8B

	full_text

double %953
Qstore8BF
D
	full_text7
5
3store double %1032, double* %945, align 8, !tbaa !8
-double8B

	full_text

double %1032
.double*8B

	full_text

double* %945
Qload8BG
E
	full_text8
6
4%1033 = load double, double* %166, align 8, !tbaa !8
.double*8B

	full_text

double* %166
Qload8BG
E
	full_text8
6
4%1034 = load double, double* %163, align 8, !tbaa !8
.double*8B

	full_text

double* %163
ycall8Bo
m
	full_text`
^
\%1035 = tail call double @llvm.fmuladd.f64(double %1034, double -4.000000e+00, double %1033)
-double8B

	full_text

double %1034
-double8B

	full_text

double %1033
Pload8BF
D
	full_text7
5
3%1036 = load double, double* %53, align 8, !tbaa !8
-double*8B

	full_text

double* %53
xcall8Bn
l
	full_text_
]
[%1037 = tail call double @llvm.fmuladd.f64(double %1036, double 5.000000e+00, double %1035)
-double8B

	full_text

double %1036
-double8B

	full_text

double %1035
ocall8Be
c
	full_textV
T
R%1038 = tail call double @llvm.fmuladd.f64(double %321, double %1037, double %966)
,double8B

	full_text

double %321
-double8B

	full_text

double %1037
,double8B

	full_text

double %966
Qstore8BF
D
	full_text7
5
3store double %1038, double* %954, align 8, !tbaa !8
-double8B

	full_text

double %1038
.double*8B

	full_text

double* %954
Rload8BH
F
	full_text9
7
5%1039 = load double, double* %171, align 16, !tbaa !8
.double*8B

	full_text

double* %171
Rload8BH
F
	full_text9
7
5%1040 = load double, double* %168, align 16, !tbaa !8
.double*8B

	full_text

double* %168
ycall8Bo
m
	full_text`
^
\%1041 = tail call double @llvm.fmuladd.f64(double %1040, double -4.000000e+00, double %1039)
-double8B

	full_text

double %1040
-double8B

	full_text

double %1039
Qload8BG
E
	full_text8
6
4%1042 = load double, double* %58, align 16, !tbaa !8
-double*8B

	full_text

double* %58
xcall8Bn
l
	full_text_
]
[%1043 = tail call double @llvm.fmuladd.f64(double %1042, double 5.000000e+00, double %1041)
-double8B

	full_text

double %1042
-double8B

	full_text

double %1041
ocall8Be
c
	full_textV
T
R%1044 = tail call double @llvm.fmuladd.f64(double %321, double %1043, double %979)
,double8B

	full_text

double %321
-double8B

	full_text

double %1043
,double8B

	full_text

double %979
Qstore8BF
D
	full_text7
5
3store double %1044, double* %967, align 8, !tbaa !8
-double8B

	full_text

double %1044
.double*8B

	full_text

double* %967
Qload8BG
E
	full_text8
6
4%1045 = load double, double* %176, align 8, !tbaa !8
.double*8B

	full_text

double* %176
Qload8BG
E
	full_text8
6
4%1046 = load double, double* %173, align 8, !tbaa !8
.double*8B

	full_text

double* %173
ycall8Bo
m
	full_text`
^
\%1047 = tail call double @llvm.fmuladd.f64(double %1046, double -4.000000e+00, double %1045)
-double8B

	full_text

double %1046
-double8B

	full_text

double %1045
Pload8BF
D
	full_text7
5
3%1048 = load double, double* %63, align 8, !tbaa !8
-double*8B

	full_text

double* %63
xcall8Bn
l
	full_text_
]
[%1049 = tail call double @llvm.fmuladd.f64(double %1048, double 5.000000e+00, double %1047)
-double8B

	full_text

double %1048
-double8B

	full_text

double %1047
ocall8Be
c
	full_textV
T
R%1050 = tail call double @llvm.fmuladd.f64(double %321, double %1049, double %996)
,double8B

	full_text

double %321
-double8B

	full_text

double %1049
,double8B

	full_text

double %996
Qstore8BF
D
	full_text7
5
3store double %1050, double* %980, align 8, !tbaa !8
-double8B

	full_text

double %1050
.double*8B

	full_text

double* %980
Rload8BH
F
	full_text9
7
5%1051 = load double, double* %181, align 16, !tbaa !8
.double*8B

	full_text

double* %181
Rload8BH
F
	full_text9
7
5%1052 = load double, double* %178, align 16, !tbaa !8
.double*8B

	full_text

double* %178
ycall8Bo
m
	full_text`
^
\%1053 = tail call double @llvm.fmuladd.f64(double %1052, double -4.000000e+00, double %1051)
-double8B

	full_text

double %1052
-double8B

	full_text

double %1051
Qload8BG
E
	full_text8
6
4%1054 = load double, double* %68, align 16, !tbaa !8
-double*8B

	full_text

double* %68
xcall8Bn
l
	full_text_
]
[%1055 = tail call double @llvm.fmuladd.f64(double %1054, double 5.000000e+00, double %1053)
-double8B

	full_text

double %1054
-double8B

	full_text

double %1053
pcall8Bf
d
	full_textW
U
S%1056 = tail call double @llvm.fmuladd.f64(double %321, double %1055, double %1026)
,double8B

	full_text

double %321
-double8B

	full_text

double %1055
-double8B

	full_text

double %1026
Qstore8BF
D
	full_text7
5
3store double %1056, double* %997, align 8, !tbaa !8
-double8B

	full_text

double %1056
.double*8B

	full_text

double* %997
)br8B!

	full_text

br label %1057
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %21) #4
%i8*8B

	full_text
	
i8* %21
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %20) #4
%i8*8B

	full_text
	
i8* %20
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %19) #4
%i8*8B

	full_text
	
i8* %19
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %18) #4
%i8*8B

	full_text
	
i8* %18
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %17) #4
%i8*8B

	full_text
	
i8* %17
$ret8B

	full_text


ret void
,double*8	B

	full_text


double* %4
%i328	B

	full_text
	
i32 %10
,double*8	B

	full_text


double* %3
,double*8	B

	full_text


double* %6
,double*8	B

	full_text


double* %7
$i328	B

	full_text


i32 %8
,double*8	B

	full_text


double* %5
$i328	B

	full_text


i32 %9
,double*8	B

	full_text


double* %1
,double*8	B

	full_text


double* %0
,double*8	B
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
#i648	B

	full_text	

i64 2
$i648	B

	full_text


i64 40
#i648	B

	full_text	

i64 0
$i648	B

	full_text


i64 32
5double8	B'
%
	full_text

double -5.500000e+00
:double8	B,
*
	full_text

double 0xC0273B645A1CAC07
%i648	B

	full_text
	
i64 507
:double8	B,
*
	full_text

double 0x4030222222222222
4double8	B&
$
	full_text

double 2.000000e+00
!i88	B

	full_text

i8 0
%i648	B

	full_text
	
i64 338
#i328	B

	full_text	

i32 1
4double8	B&
$
	full_text

double 1.400000e+00
$i328	B

	full_text


i32 -2
5double8	B'
%
	full_text

double -0.000000e+00
#i328	B

	full_text	

i32 7
4double8	B&
$
	full_text

double 1.000000e+00
$i328	B

	full_text


i32 -1
5double8	B'
%
	full_text

double -2.000000e+00
&i648	B

	full_text


i64 1690
5double8	B'
%
	full_text

double -4.000000e+00
4double8	B&
$
	full_text

double 2.500000e-01
&i648	B

	full_text


i64 2535
#i328	B

	full_text	

i32 0
#i648	B

	full_text	

i64 1
:double8	B,
*
	full_text

double 0x4037B74BC6A7EF9D
%i18	B

	full_text


i1 false
4double8	B&
$
	full_text

double 4.000000e-01
4double8	B&
$
	full_text

double 7.500000e-01
4double8	B&
$
	full_text

double 5.000000e+00
:double8	B,
*
	full_text

double 0x4000222222222222
#i648	B

	full_text	

i64 3
%i648	B

	full_text
	
i64 845
%i648	B

	full_text
	
i64 169
&i648	B

	full_text


i64 3380
4double8	B&
$
	full_text

double 4.000000e+00
#i648	B

	full_text	

i64 4
4double8	B&
$
	full_text

double 1.210000e+02
:double8	B,
*
	full_text

double 0x4028333333333334
4double8	B&
$
	full_text

double 6.000000e+00
$i328	B

	full_text


i32 -3       	  
 

                       !! "# "" $$ %& %' %% () (+ ** ,, -. -/ -- 01 02 33 44 55 66 77 88 9: 99 ;< ;; => == ?@ ?? AB AC AD AA EF EE GH GG IJ II KL KK MN MO MM PQ PR PS PP TU TT VW VV XY XX Z[ ZZ \] \^ \\ _` _a _b __ cd cc ef ee gh gg ij ii kl km kk no np nq nn rs rr tu tt vw vv xy xx z{ z| zz }~ } }	€ }} ‚  ƒ„ ƒƒ …† …… ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ ŒŒ Ž   
‘ 
’  “” ““ •– •• —˜ —— ™š ™™ ›œ ›
 ›› žŸ ž
  ž
¡ žž ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­
° ­­ ±² ±± ³´ ³³ µ¶ µµ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼
¿ ¼¼ ÀÁ ÀÀ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ È
Ê ÈÈ ËÌ Ë
Í Ë
Î ËË ÏÐ ÏÏ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÚ ÛÜ ÛÛ ÝÞ Ý
ß Ý
à ÝÝ áâ áá ãä ãã åæ åå çè ç
é çç êë ê
ì ê
í êê îï îî ðñ ðð òó òò ôõ ôô ö÷ ö
ø öö ùú ù
û ù
ü ùù ýþ ýý ÿ€ ÿÿ ‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž ŽŽ ‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —
š —— ›œ ›› ž  Ÿ  ŸŸ ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬¬ ­® ­­ ¯° ¯
± ¯
² ¯¯ ³´ ³³ µ¶ µ
· µ
¸ µµ ¹º ¹¹ »» ¼½ ¼¼ ¾¿ ¾
À ¾
Á ¾¾ ÂÃ ÂÂ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ ÈÈ ÊÊ ËÌ ËË ÍÎ Í
Ï Í
Ð ÍÍ ÑÒ ÑÑ ÓÔ Ó
Õ Ó
Ö ÓÓ ×Ø ×× ÙÙ ÚÛ ÚÚ ÜÝ Ü
Þ Ü
ß ÜÜ àá àà âã â
ä â
å ââ æç ææ èè éê éé ëì ë
í ë
î ëë ïð ïï ñò ñ
ó ñ
ô ññ õö õõ ÷÷ øù øø úû ú
ü ú
ý úú þÿ þþ € €€ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž    ‘’ ‘‘ “” ““ •– •
— •• ˜™ ˜˜ š› šš œ œœ žŸ žž  ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §§ ©ª ©© «¬ «« ­® ­­ ¯° ¯
± ¯¯ ²³ ²² ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ã
å ãã æç æ
è ææ éê é
ë éé ìì íî íí ïð ï
ñ ï
ò ïï óô óó õö õõ ÷ø ÷
ù ÷÷ úû ú
ü ú
ý úú þÿ þþ € €€ ‚ƒ ‚
„ ‚‚ …† …
‡ …
ˆ …… ‰Š ‰‰ ‹Œ ‹‹ Ž 
  ‘ 
’ 
“  ”• ”” –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›
ž ›› Ÿ  ŸŸ ¡¢ ¡¡ £¤ £
¥ ££ ¦¦ §¨ §§ ©ª ©
« ©
¬ ©© ­® ­­ ¯¯ °± °° ²³ ²
´ ²
µ ²² ¶· ¶¶ ¸¸ ¹º ¹¹ »¼ »
½ »
¾ »» ¿À ¿¿ ÁÁ ÂÃ ÂÂ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ ÈÈ ÊÊ ËÌ ËË ÍÎ Í
Ï Í
Ð ÍÍ ÑÒ ÑÑ ÓÓ ÔÕ ÔÔ Ö× Ö
Ø Ö
Ù ÖÖ ÚÛ ÚÚ ÜÜ ÝÞ ÝÝ ßà ß
á ß
â ßß ãä ãã åæ åå çè çç éê é
ë éé ìí ìì îï î
ð îî ñò ñ
ó ññ ôõ ôô ö÷ öö øù ø
ú øø ûü û
ý ûû þÿ þ
€ þ
 þþ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž 
  ‘ 
’  “” “
• ““ –— –
˜ –– ™š ™
› ™™ œ œ
ž œœ Ÿ
  ŸŸ ¡¢ ¡
£ ¡
¤ ¡¡ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µµ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ É
Ê ÉÉ ËÌ Ë
Í Ë
Î ËË ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô Ò
Õ ÒÒ Ö× ÖÖ ØÙ ØØ ÚÛ Ú
Ü ÚÚ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ã
å ãã æç æ
è ææ éê é
ë éé ìí ì
î ìì ï
ð ïï ñò ñ
ó ñ
ô ññ õö õõ ÷ø ÷
ù ÷÷ úû úú üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž ŽŽ ‘ 
’  “” “
• ““ –— –
˜ –– ™š ™
› ™™ œ œ
ž œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §
¨ §§ ©ª ©
« ©
¬ ©© ­® ­
¯ ­
° ­­ ±² ±
³ ±± ´µ ´´ ¶· ¶
¸ ¶¶ ¹
º ¹¹ »¼ »
½ »
¾ »» ¿À ¿
Á ¿
Â ¿¿ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ È
É ÈÈ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ Ï
Ð ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×
Ø ×× ÙÚ Ù
Û Ù
Ü ÙÙ ÝÞ Ý
ß ÝÝ àà á
â áá ãä ãã å
æ åå çè çç éê éé ëì ëë í
î íí ïð ï
ñ ïï òó òò ôõ ôô ö÷ ö
ø öö ùú ù
û ù
ü ùù ýþ ý
ÿ ýý € €€ ‚ƒ ‚‚ „… „„ †
‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž 
  ‘ 
’ 
“  ”• ”
– ”” —˜ —— ™š ™™ ›œ ›› 
ž  Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §
© §
ª §§ «¬ «
­ «« ®¯ ®® °± °° ²
³ ²² ´µ ´
¶ ´´ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼
¿ ¼¼ ÀÁ À
Â ÀÀ ÃÄ ÃÃ Å
Æ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ Ó
Õ ÓÓ ÖÖ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ã
å ãã æç æ
è ææ éê é
ë éé ìí ì
î ìì ïð ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù ø
ú øø ûü û
ý ûû þÿ þ
€ þþ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  ““ ”• ”” –— –
˜ –
™ –– š› šš œ œœ žŸ ž
  žž ¡¢ ¡
£ ¡
¤ ¡¡ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬
¯ ¬¬ °± °° ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·
º ·· »¼ »» ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä Â
Å ÂÂ ÆÇ ÆÆ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÍ ÎÏ ÎÎ ÐÑ Ð
Ò Ð
Ó ÐÐ ÔÕ ÔÔ ÖÖ ×Ø ×× ÙÚ Ù
Û Ù
Ü ÙÙ ÝÞ ÝÝ ßß àá àà âã â
ä â
å ââ æç ææ èè éê éé ëì ë
í ë
î ëë ïð ïï ññ òó òò ôõ ô
ö ô
÷ ôô øù øø úú ûü ûû ýþ ý
ÿ ý
€	 ýý 	‚	 		 ƒ	ƒ	 „	…	 „	„	 †	‡	 †	
ˆ	 †	
‰	 †	†	 Š	‹	 Š	Š	 Œ		 Œ	Œ	 Ž		 Ž	
	 Ž	Ž	 ‘	’	 ‘	
“	 ‘	‘	 ”	•	 ”	
–	 ”	”	 —	˜	 —	—	 ™	š	 ™	™	 ›	œ	 ›	
	 ›	›	 ž	Ÿ	 ž	
 	 ž	ž	 ¡	¢	 ¡	
£	 ¡	
¤	 ¡	¡	 ¥	¦	 ¥	¥	 §	¨	 §	§	 ©	ª	 ©	
«	 ©	©	 ¬	­	 ¬	
®	 ¬	¬	 ¯	°	 ¯	
±	 ¯	¯	 ²	³	 ²	
´	 ²	²	 µ	¶	 µ	
·	 µ	µ	 ¸	¹	 ¸	
º	 ¸	¸	 »	¼	 »	
½	 »	»	 ¾	
¿	 ¾	¾	 À	Á	 À	
Â	 À	
Ã	 À	À	 Ä	Å	 Ä	
Æ	 Ä	Ä	 Ç	È	 Ç	
É	 Ç	
Ê	 Ç	Ç	 Ë	Ì	 Ë	Ë	 Í	Î	 Í	Í	 Ï	Ð	 Ï	
Ñ	 Ï	Ï	 Ò	Ó	 Ò	
Ô	 Ò	Ò	 Õ	Ö	 Õ	
×	 Õ	Õ	 Ø	Ù	 Ø	
Ú	 Ø	Ø	 Û	Ü	 Û	
Ý	 Û	Û	 Þ	ß	 Þ	
à	 Þ	Þ	 á	â	 á	
ã	 á	á	 ä	
å	 ä	ä	 æ	ç	 æ	
è	 æ	
é	 æ	æ	 ê	ë	 ê	
ì	 ê	ê	 í	î	 í	
ï	 í	
ð	 í	í	 ñ	ò	 ñ	ñ	 ó	ô	 ó	ó	 õ	ö	 õ	
÷	 õ	õ	 ø	ù	 ø	
ú	 ø	ø	 û	ü	 û	
ý	 û	û	 þ	ÿ	 þ	
€
 þ	þ	 
‚
 

ƒ
 

 „
…
 „

†
 „
„
 ‡
ˆ
 ‡

‰
 ‡
‡
 Š

‹
 Š
Š
 Œ

 Œ

Ž
 Œ


 Œ
Œ
 
‘
 

 ’
“
 ’

”
 ’
’
 •
–
 •
•
 —
˜
 —

™
 —
—
 š
›
 š

œ
 š
š
 
ž
 

Ÿ
 

  
¡
  

¢
  
 
 £
¤
 £

¥
 £

¦
 £
£
 §
¨
 §
§
 ©
ª
 ©
©
 «
¬
 «

­
 «
«
 ®
¯
 ®

°
 ®
®
 ±
²
 ±

³
 ±
±
 ´
µ
 ´

¶
 ´
´
 ·
¸
 ·

¹
 ·
·
 º
»
 º

¼
 º
º
 ½
¾
 ½
½
 ¿
À
 ¿

Á
 ¿
¿
 Â

Ã
 Â
Â
 Ä
Å
 Ä

Æ
 Ä

Ç
 Ä
Ä
 È
É
 È

Ê
 È

Ë
 È
È
 Ì
Í
 Ì

Î
 Ì
Ì
 Ï
Ð
 Ï
Ï
 Ñ
Ò
 Ñ

Ó
 Ñ
Ñ
 Ô

Õ
 Ô
Ô
 Ö
×
 Ö

Ø
 Ö

Ù
 Ö
Ö
 Ú
Û
 Ú

Ü
 Ú

Ý
 Ú
Ú
 Þ
ß
 Þ

à
 Þ
Þ
 á
â
 á
á
 ã

ä
 ã
ã
 å
æ
 å

ç
 å
å
 è
é
 è
è
 ê

ë
 ê
ê
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

ó
 ò
ò
 ô
õ
 ô

ö
 ô

÷
 ô
ô
 ø
ù
 ø

ú
 ø
ø
 û
ü
 û
û
 ý
þ
 ý
ý
 ÿ
€ ÿ
ÿ
 ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 Ž
‘ ŽŽ ’“ ’
” ’’ •– •• —˜ —— ™š ™™ ›œ ›
 ›› žŸ žž  ¡  
¢    £¤ ££ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±± ³´ ³³ µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä Â
Å ÂÂ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ ËË ÍÎ Í
Ï ÍÍ ÐÑ ÐÐ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü Ú
Ý ÚÚ Þß Þ
à ÞÞ áâ áá ãä ã
å ãã æç æ
è ææ éê éé ëì ë
í ëë îï î
ð î
ñ îî òó ò
ô òò õõ ö÷ öö øù øø úû úú üý üü þÿ þþ € €€ ‚ƒ ‚… „„ †‡ †† ˆ‰ ˆˆ Š‹ ŠŠ ŒŒ Ž   ‘’ ‘‘ “” ““ •– •• —˜ —— ™š ™™ ›œ ›› ž  Ÿ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô ó
õ óó ö÷ ö
ø öö ùú ù
û ùù üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™
› ™
œ ™
 ™™ žŸ žž  ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥
¨ ¥
© ¥¥ ª« ªª ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±
´ ±
µ ±± ¶· ¶¶ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½
À ½
Á ½½ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë É
Ì É
Í ÉÉ ÎÏ ÎÎ ÐÑ ÐÐ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×
Ù ×
Ú ×
Û ×× ÜÝ ÜÜ Þß Þ
à Þ
á Þ
â ÞÞ ãä ãã åæ å
ç å
è å
é åå êë êê ìí ì
î ì
ï ì
ð ìì ñò ññ óô ó
õ ó
ö ó
÷ óó øù øø úû ú
ü ú
ý ú
þ úú ÿ€ ÿÿ ‚ 
ƒ 
„ 
…  †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —
š —
› —— œ œœ žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³
´ ³³ µ¶ µ
· µ
¸ µµ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼
¿ ¼
À ¼¼ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ Ø
Ù ØØ ÚÛ Ú
Ü Ú
Ý ÚÚ Þß Þ
à ÞÞ áâ á
ã á
ä á
å áá æç ææ èé è
ê èè ëì ë
í ëë îï î
ð îî ñò ñ
ó ññ ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ý
þ ýý ÿ€ ÿ
 ÿ
‚ ÿÿ ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”
— ”
˜ ”” ™š ™™ ›œ ›› ž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´
µ ´´ ¶· ¶
¸ ¶
¹ ¶¶ º» º
¼ º
½ ºº ¾¿ ¾
À ¾¾ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ Æ
Ç ÆÆ ÈÉ È
Ê È
Ë ÈÈ ÌÍ Ì
Î Ì
Ï ÌÌ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ Õ
Ö ÕÕ ×Ø ×
Ù ×× ÚÛ ÚÚ Ü
Ý ÜÜ Þß Þ
à ÞÞ áâ á
ã áá ä
å ää æç æ
è æ
é ææ êë ê
ì êê íî íí ïð ï
ñ ïï òó ò
ô òò õö õ
÷ õõ øù øø úû ú
ü úú ýþ ý
ÿ ý
€ ýý ‚ 
ƒ  „… „„ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ   ‘’ ‘
“ ‘‘ ”• ”
– ”
— ”” ˜™ ˜
š ˜˜ ›œ ›› ž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «
® «« ¯° ¯
± ¯¯ ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä Â
Å ÂÂ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ Î
Ð ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û Ù
Ü ÙÙ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ãã åæ åå çè çç éê éé ëì ëë íî íí ïð ïï ñò ñô ó
õ óó ö÷ ö
ø öö ùú ù
û ùù üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› šš œž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ð îî ñò ñ
ó ññ ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ  
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜˜ š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©© «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËË ÌÍ ÌÌ ÎÏ Î
Ð Î
Ñ Î
Ò ÎÎ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü Ú
Ý Ú
Þ ÚÚ ßà ßß áâ áá ãä ã
å ãã æç æ
è æ
é æ
ê ææ ëì ëë íî íí ïð ï
ñ ïï òó ò
ô ò
õ ò
ö òò ÷ø ÷÷ ùú ùù ûü û
ý ûû þÿ þ
€ þ
 þ
‚ þþ ƒ„ ƒƒ …† …… ‡ˆ ‡
‰ ‡‡ ŠŠ ‹Œ ‹‹ Ž 
 
 
‘  ’“ ’’ ”• ”
– ”
— ”
˜ ”” ™š ™™ ›œ ›
 ›
ž ›
Ÿ ››  ¡    ¢£ ¢
¤ ¢
¥ ¢
¦ ¢¢ §¨ §§ ©ª ©
« ©
¬ ©
­ ©© ®¯ ®® °± °
² °
³ °
´ °° µ¶ µµ ·¸ ·· ¹º ¹
» ¹
¼ ¹
½ ¹¹ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ Ï
Ò Ï
Ó ÏÏ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ë
ì ëë íî í
ï í
ð íí ñò ñ
ó ññ ôõ ô
ö ô
÷ ô
ø ôô ùú ùù ûü û
ý ûû þÿ þ
€ þþ ‚ 
ƒ  „… „
† „„ ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  
‘  ’“ ’
” ’
• ’’ –— –
˜ –– ™š ™
› ™
œ ™
 ™™ žŸ žž  ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ
¶ µµ ·¸ ·
¹ ·
º ·· »¼ »
½ »» ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î Ì
Ï Ì
Ð ÌÌ ÑÒ ÑÑ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè çç éê é
ë éé ì
í ìì îï î
ð î
ñ îî òó ò
ô ò
õ òò ö÷ ö
ø öö ùú ùù ûü û
ý ûû þ
ÿ þþ € €
‚ €
ƒ €€ „… „
† „
‡ „„ ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ 
Ž   
‘  ’“ ’’ ”
• ”” –— –
˜ –– ™š ™
› ™™ œ
 œœ žŸ ž
  ž
¡ žž ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®® ±² ±± ³´ ³
µ ³³ ¶· ¶
¸ ¶
¹ ¶¶ º» º
¼ ºº ½¾ ½½ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ ÉÉ ËÌ Ë
Í ËË ÎÏ Î
Ð Î
Ñ ÎÎ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ãä ã
å ãã æç æ
è æ
é ææ êë ê
ì êê íî íí ïð ïï ñò ñ
ó ññ ôõ ôô ö÷ ö
ø öö ùú ùù ûü û
ý ûû þÿ þ
€ þ
 þþ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž   
‘  ’“ ’
” ’
• ’’ –— –
˜ –– ™š ™™ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »» ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
Ý Û
Þ Û
ß ÛÛ àá àà âã â
ä â
å â
æ ââ çè çç éê é
ë é
ì é
í éé îï îî ðñ ð
ò ð
ó ð
ô ðð õö õõ ÷ø ÷
ù ÷
ú ÷
û ÷÷ üý üü þÿ þ
€ þ
 þ
‚ þþ ƒ„ ƒƒ …† …
‡ …
ˆ …
‰ …… Š‹ ŠŠ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —— ™š ™
› ™™ œ œ
ž œœ Ÿ  Ÿ
¡ Ÿ
¢ Ÿ
£ ŸŸ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½
¾ ½½ ¿À ¿
Á ¿
Â ¿¿ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È Æ
É Æ
Ê ÆÆ ËÌ ËË ÍÎ ÍÍ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá ä
å ää æç æ
è æ
é ææ êë ê
ì êê íî í
ï í
ð í
ñ íí òó òò ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰
Š ‰‰ ‹Œ ‹
 ‹
Ž ‹‹   ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢  
£  
¤    ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹¹ »¼ »
½ »» ¾
¿ ¾¾ ÀÁ À
Â À
Ã ÀÀ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ È
Ê ÈÈ ËÌ ËË ÍÎ Í
Ï ÍÍ Ð
Ñ ÐÐ ÒÓ Ò
Ô Ò
Õ ÒÒ Ö× Ö
Ø Ö
Ù ÖÖ ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ß
à ßß áâ á
ã áá äå ää æ
ç ææ èé è
ê èè ëì ë
í ëë î
ï îî ðñ ð
ò ð
ó ðð ôõ ô
ö ôô ÷ø ÷÷ ùú ùù ûü û
ý ûû þÿ þþ € €
‚ €€ ƒ„ ƒ
… ƒ
† ƒƒ ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –
˜ –
™ –– š› š
œ šš ž  Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦
¨ ¦¦ ©ª ©
« ©
¬ ©© ­® ­
¯ ­­ °± °° ²³ ²² ´µ ´
¶ ´´ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼
¿ ¼¼ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ Ó
Õ ÓÓ Ö
Ø ×× Ù
Ú ÙÙ Û
Ü ÛÛ Ý
Þ ÝÝ ß
à ßß áâ 5â Ùâ Áâ èã õã Œã ã Ëã Šä 4ä Êä ¸ä ßå 7å ÷å Óå úæ Üæ Öæ ƒ	ç ,è 6è èè Êè ñé $ê 2ê ¬ê ¦ê Íë 8ë Œë Úë ìë “ì 3ì »ì ¯ì Ö  	 
           ! #$ & '% )" +, .* /- 1 :9 <" >= @8 B; C? DA FE H J LG NK O8 Q; R? SP UT W YX [V ]Z ^8 `; a? b_ dc f hg je li m8 o; p? qn sr u wv yt {x |8 ~; ? €} ‚ „ †… ˆƒ Š‡ ‹Œ Ž ; ‘? ’ ”“ – ˜ š• œ™  Ÿ;  ? ¡ž £¢ ¥ §¦ ©¤ «¨ ¬ ®; ¯? °­ ²± ´ ¶µ ¸³ º· » ½; ¾? ¿¼ ÁÀ Ã ÅÄ ÇÂ ÉÆ Ê Ì; Í? ÎË ÐÏ Ò ÔÓ ÖÑ ØÕ ÙÚ ÜÛ Þ; ß? àÝ âá ä æã èå éÛ ë; ì? íê ïî ñ óò õð ÷ô øÛ ú; û? üù þý € ‚ „ÿ †ƒ ‡Û ‰; Š? ‹ˆ Œ  ‘ “Ž •’ –Û ˜; ™? š— œ› ž  Ÿ ¢ ¤¡ ¥2 §; ¨? ©¦ «¬ ®­ °; ±? ²¯ ´3 ¶; ·? ¸µ º» ½¼ ¿; À? Á¾ Ã4 Å; Æ? ÇÄ ÉÊ ÌË Î; Ï? ÐÍ Ò5 Ô; Õ? ÖÓ ØÙ ÛÚ Ý; Þ? ßÜ á6 ã; ä? åâ çè êé ì; í? îë ð7 ò; ó? ôñ ö÷ ùø û; ü? ýú ÿ  ƒ‚ … ‡„ ‰† Š Œ‹ Ž  ’‘ ” –“ — ™˜ ›š  Ÿž ¡œ £  ¤ ¦¥ ¨§ ª ¬« ®© °­ ± ³² µ´ · ¹¸ »¶ ½º ¾G À‚ ÁV Ã Äe Æš Çt É§ Êƒ Ì´ Í• ÏK Ð¤ ÒZ Ó³ Õi ÖÂ Øx ÙÑ Û‡ Üã Þ™ ßð á¨ âÿ ä· åŽ çÆ è êÕ ëì îí ð; ñ? òï ôó öõ øå ùí û; ü? ýú ÿþ € ƒô „í †; ‡? ˆ… Š‰ Œ‹ Žƒ í ‘; ’? “ •” —– ™’ ší œ; ? ž›  Ÿ ¢¡ ¤¡ ¥¦ ¨§ ª; «? ¬© ®¯ ±° ³; ´? µ² ·¸ º¹ ¼; ½? ¾» ÀÁ ÃÂ Å; Æ? ÇÄ ÉÊ ÌË Î; Ï? ÐÍ ÒÓ ÕÔ ×; Ø? ÙÖ ÛÜ ÞÝ à; á? âß äã æ• èç êå ëG íé ïì ðî òã óŽ õt ÷ô ùö úø üñ ýÝ ÿ; €? þ ƒð …¤ ‡† ‰„ ŠV Œˆ Ž‹  ‘‚ ’³ ”­ •ª —“ ˜– š ›È ‹ žœ  „ ¢¿ £Ÿ ¤¡ ¦™ §Ý ©; ª? «¨ ­ÿ ¯³ ±° ³® ´e ¶² ¸µ ¹· »¬ ¼Â ¾¶ ¿¹ Á½ ÂÀ Äº ÅÈ Çµ ÈÆ Ê® Ì¿ ÍÉ ÎË ÐÃ ÑÝ Ó; Ô? ÕÒ ×v ÙØ Ûô ÜÚ Þö ßÝ áÖ âÑ ä¿ åÈ çã èæ êà ëÈ íö îì ðô ò¿ óï ôÓ öõ øÚ ù² û÷ ýú þõ €ü ÿ ƒñ „‚ †é ‡Ý ‰; Š? ‹ˆ … Ž ‘õ ’ú ” •“ —Œ ˜à šÈ ›× ™ žœ  – ¡Ñ £Ñ ¥¢ ¦¤ ¨¿ ª¿ «§ ¬È ®È ¯© °­ ²Ÿ ³Ž µï ·´ ¸¶ ºõ ¼Ñ ½¹ ¾ú Àæ Á» Â¿ Ä± ÅÚ ÇÆ Éõ ËÈ Ìõ ÎÍ Ðú ÒÏ ÓÈ ÕÑ ÖÔ ØÊ Ú¿ Û× ÜÙ ÞÃ ßà âá äã æI è— êé ìë îç ðí ñ óò õô ÷ï øå úö ûû üù þß ÿX ¦ ƒ‚ …„ ‡€ ‰† Šò Œ‹ Žˆ å ‘ ’¥ “ •þ –g ˜µ š™ œ› ž—   ¡ £¢ ¥Ÿ ¦å ¨¤ ©Ï ª§ ¬¨ ­Ä ¯® ±° ³Ø µ² ¶ ¸· º´ »å ½¹ ¾… ¿¼ ÁÒ Âõ ÄÃ ÆŽ ÈÅ ÉŸ ËÊ ÍÇ Îå ÐÌ ÑÝ ÒÏ Ôˆ ÕG Ø† ÙV Û“ Üe Þ  ßt á­ âƒ äº å• ç‚ è¤ ê ë³ íš îÂ ð§ ñÑ ó´ ôã öK ÷ð ùZ úÿ üi ýŽ ÿx € ‚‡ ƒõ …™ †€ ˆ¨ ‰‹ ‹· Œ– ŽÆ ¡ ‘Õ ’“ •” —; ˜? ™– ›š œ Ÿå  ” ¢; £? ¤¡ ¦¥ ¨§ ªô «” ­; ®? ¯¬ ±° ³² µƒ ¶” ¸; ¹? º· ¼» ¾½ À’ Á” Ã; Ä? ÅÂ ÇÆ ÉÈ Ë¡ ÌÍ ÏÎ Ñ; Ò? ÓÐ ÕÖ Ø× Ú; Û? ÜÙ Þß áà ã; ä? åâ çè êé ì; í? îë ðñ óò õ; ö? ÷ô ùú üû þ; ÿ? €	ý ‚	ƒ	 …	„	 ‡	; ˆ	? ‰	†	 ‹	õ 	å 	Œ	 	Ž	 ’	ç “	‘	 •	Š	 –	– ˜	Â š	—	 œ	™	 	›	 Ÿ	”	  	„	 ¢	; £	? ¤	¡	 ¦	€ ¨	„ ª	§	 «	©	 ­	† ®	¬	 °	¥	 ±	­ ³	Ô ´	³ ¶	²	 ·	µ	 ¹	¯	 º	Ñ ¼	† ½	»	 ¿	§	 Á	æ Â	¾	 Ã	À	 Å	¸	 Æ	„	 È	; É	? Ê	Ç	 Ì	‹ Î	® Ð	Í	 Ñ	Ï	 Ó	° Ô	Ò	 Ö	Ë	 ×	¶ Ù	Ý Ú	Â Ü	Ø	 Ý	Û	 ß	Õ	 à	Ñ â	° ã	á	 å	Í	 ç	æ è	ä	 é	æ	 ë	Þ	 ì	„	 î	; ï	? ð	í	 ò	v ô	ó	 ö	—	 ÷	õ	 ù	™	 ú	ø	 ü	ñ	 ý	¿ ÿ	æ €
Ñ ‚
þ	 ƒ

 …
û	 †
Ñ ˆ
™	 ‰
‡
 ‹
—	 
æ Ž
Š
 
Ó ‘

 “
	 ”
² –
’
 ˜
•
 ™
þ ›
—
 œ
š
 ž
Œ
 Ÿ

 ¡
„
 ¢
„	 ¤
; ¥
? ¦
£
 ¨
… ª
©
 ¬

 ­
•
 ¯
«
 °
®
 ²
§
 ³
È µ
ï ¶
à ¸
´
 ¹
·
 »
±
 ¼
¿ ¾
¿ À
½
 Á
¿
 Ã
æ Å
æ Æ
Â
 Ç
Ñ É
Ñ Ê
Ä
 Ë
È
 Í
º
 Î
©
 Ð
Ñ Ò
Ï
 Ó
Ñ
 Õ

 ×
ø Ø
Ô
 Ù
•
 Û
ï Ü
Ö
 Ý
Ú
 ß
Ì
 à
	 â
á
 ä

 æ
ã
 ç
þ é
è
 ë
•
 í
ê
 î
Ñ ð
ì
 ñ
ï
 ó
å
 õ
æ ö
ò
 ÷
ô
 ù
Þ
 ú
€ ü
I þ
ý
 €û
 ‚ÿ
 ƒ— …„ ‡ ˆò Š‰ Œ† å ‹ ž	 ‘Ž “†	 ”‹ –X ˜— š• œ™ ¦ Ÿž ¡› ¢ò ¤£ ¦  §å ©¥ ªÄ	 «¨ ­¡	 ®˜ °g ²± ´¯ ¶³ ·µ ¹¸ »µ ¼ ¾½ Àº Áå Ã¿ Äê	 ÅÂ ÇÇ	 È¥ Êó	 ÌÉ ÎË ÏÄ ÑÐ ÓÍ Ô ÖÕ ØÒ Ùå Û× Ü 
 ÝÚ ßí	 à©
 â•
 äá å
 çã èŸ êé ìæ íå ïë ðø
 ñî ó£
 ôû
 ÷• ù¯ ûÉ ý•
 ÿ©
 õ ƒx …„ ‡Õ ‰ˆ ‹ Žv ’Õ ” – ˜ š œ žÔ ¡é ¢½ ¤Õ ¥¦ §½ ¨ ª£ «ø ­‰ ®ï °“ ±£ ³Ð ´¦ ¶¸ ·© ¹ž º¬ ¼„ ½í ¿€ À² Â‘ Ãµ Å± Æ¸ È— É» Ëý
 Ìë Îþ Ïé Ñü Òç Ôú Õå ×ø Øã Úö ÛÕ Ýá ß­ àÜ âÔ ãç å¶ æã èÝ éÿ ë	 ìê îÚ ïø ñø òð ôÑ õñ ÷ï øö úÈ ûê ýæ þü €¿ Ù ƒ† „Ö †“ ‡Ó ‰  ŠÐ Œ­ Í º ¾ ’´ “¯ •‡ –Ü ˜8 š— ›; œ? ™ Ÿž ¡  £å ¤8 ¦— §; ¨? ©¥ «ª ­¬ ¯ô °8 ²— ³; ´? µ± ·¶ ¹¸ »ƒ ¼8 ¾— ¿; À? Á½ ÃÂ ÅÄ Ç’ È8 Ê— Ë; Ì? ÍÉ ÏÎ ÑÐ Ó¡ ÔÜ Ö2 ØÕ Ù; Ú? Û× Ý3 ßÕ à; á? âÞ ä4 æÕ ç; è? éå ë5 íÕ î; ï? ðì ò6 ôÕ õ; ö? ÷ó ù7 ûÕ ü; ý? þú €Ö ‚Ü ƒ; „? … ‡» ‰¬ Šˆ ŒÊ ‹ † £ ’Á “‘ •Ž –Ö ˜Ü ™; š? ›— ¸ Ÿ©  ž ¢Ç £¡ ¥œ ¦á ¨Ü ©Þ «§ ¬ª ®¤ ¯ÿ ±Ç ²° ´© ¶ê ·³ ¸µ º­ »Ö ½Ü ¾; ¿? À¼ Âµ Ä¦ ÅÃ ÇÄ ÈÆ ÊÁ Ëç Íã Îä ÐÌ ÑÏ ÓÉ Ôÿ ÖÄ ×Õ Ù¦ Ûê ÜØ ÝÚ ßÒ àÖ âÜ ã; ä? åá ç² é£ êÁ ìè íë ïæ ðü òê óÿ õñ öô øî ùÿ ûÁ üú þ£ €ê ý ‚  „ÿ …² ‡ƒ ‰† Ší Œˆ ‹ ÿ Ž ’÷ “Ö •Ü –; —? ˜” š… œ› ž  Ÿ† ¡ ¢  ¤™ ¥ö §ñ ¨ù ª¦ «© ­£ ®ü °ü ²¯ ³± µê ·ê ¸´ ¹ÿ »ÿ ¼¶ ½º ¿¬ À› Âð ÄÁ ÅÃ Ç  Éø ÊÆ Ë† Íó ÎÈ ÏÌ Ñ¾ Òÿ ÔÓ Ö  ØÕ Ùí ÛÚ Ý† ßÜ àÿ âÞ ãá å× çê èä éæ ëÐ ì› îÊ ðí ñ» óï ô¬ öò ÷ò ùõ ûø üå þú ÿ” €ý ‚ ƒ‘ …Ç ‡„ ˆ¸ Š† ‹© ‰ Žò Œ ’ “å •‘ –¹ —” ™— šž œÄ ž› Ÿµ ¡ ¢¦ ¤  ¥ §£ ©¦ ªå ¬¨ ­Þ ®« °¼ ±« ³Á µ² ¶² ¸´ ¹£ »· ¼ ¾º À½ Áå Ã¿ Ä‘ ÅÂ Çá È¸ Ê† ÌÉ Í› ÏË Ð  ÒÎ ÓŸ ÕÑ ×Ô Øå ÚÖ Ûê ÜÙ Þ” ßÕ á âÊ äÇ æÄ èÁ ê† ì› î  ðà òÊ ô• õÇ ÷‹ øÄ ú˜ ûÁ ý¥ þ» €— ¸ ƒX „µ †g ‡² ‰v Š¬ Œ™ © ¦ ¦ ’µ “£ •Ä –  ˜Ó ™² › ž› ŸŒ ¡ ¢é ¤Ô ¥Õ §½ ¨½ ª¦ «£ ­ ®‰ °ø ±Š ³  ´ˆ ¶ï ·Ð ¹£ º¸ ¼¦ ½ž ¿© À„ Â¬ Ã€ Åí Æ† È² É„ Ëš Ì± Îµ Ï— Ñ¸ Òý
 Ô» Õþ ×ë Øü Úé Ûú Ýç Þø àå áö ãã ä¿ æü çæ éê êÈ ìö íï ïñ ðÑ òð óø õø öÚ øê ù	 ûÿ üÝ þã ÿ¶ ç ‚Ô „Ü …­ ‡á ˆâ Š† ‹ß “ ŽÜ   ‘Ù “­ ”Ö –º — ™Ó ›˜ œÐ ž‹ ŸÍ ¡˜ ¢Ç ¤¥ ¥Ä §´ ¨ ªÁ ¬© ­¾ ¯X °» ²g ³¸ µv ¶² ¸… ¹ »¯ ½º ¾¬ À¦ Á© Ãµ Ä¦ ÆÄ Ç£ ÉÓ ÊË Í8 ÏÌ Ð; Ñ? ÒÎ ÔÓ ÖÕ Øå Ù8 ÛÌ Ü; Ý? ÞÚ àß âá äô å8 çÌ è; é? êæ ìë îí ðƒ ñ8 óÌ ô; õ? öò ø÷ úù ü’ ý8 ÿÌ €; ? ‚þ „ƒ †… ˆ¡ ‰Š Œ2 Ž‹ ; ? ‘ “3 •‹ –; —? ˜” š4 œ‹ ; ž? Ÿ› ¡5 £‹ ¤; ¥? ¦¢ ¨6 ª‹ «; ¬? ­© ¯7 ±‹ ²; ³? ´° ¶  ¸Ö º· »; ¼? ½¹ ¿Á Á¯ ÂÀ ÄÓ ÅÃ Ç¾ È¦ ÊÇ ËÉ ÍÆ ÎÖ Ð· Ñ; Ò? ÓÏ Õ¾ ×¬ ØÖ ÚÐ ÛÙ ÝÔ Þƒ à’ á† ãß äâ æÜ çå éÐ êè ì¬ î  ïë ðí òå óÖ õ· ö; ÷? øô ú» ü© ýû ÿÍ €þ ‚ù ƒý …™ †€ ˆ„ ‰‡ ‹ Œå ŽÍ  ‘© “  ” •’ —Š ˜Ö š· ›; œ? ™ Ÿ¸ ¡¦ ¢Ç ¤  ¥£ §ž ¨è ª  «å ­© ®¬ °¦ ±å ³Ç ´² ¶¦ ¸  ¹µ º£ ¼µ ½² ¿» Á¾ Â÷ ÄÀ ÅÃ Ç· ÈÆ Ê¯ ËÖ Í· Î; Ï? ÐÌ Ò… ÔÓ Ö£ ×¾ ÙÕ ÚØ ÜÑ Ýî ß§ àë âÞ ãá åÛ æè èè êç ëé í  ï  ðì ñå óå ôî õò ÷ä øÓ úô üù ýû ÿ£ ® ‚þ ƒ¾ …ñ †€ ‡„ ‰ö Šµ Œ‹ Ž£  ‘÷ “’ •¾ —” ˜å š– ›™  Ÿ   œ ¡ž £ˆ ¤ ¦€ ¨§ ª¥ «I ­¬ ¯© °— ²± ´® µå ·³ ¸Ì ¹¶ »¹ ¼‘ ¾‹ À¿ Â½ ÃX ÅÄ ÇÁ È¦ ÊÉ ÌÆ Íå ÏË Ðñ ÑÎ ÓÏ Ôž Ö˜ Ø× ÚÕ Ûg ÝÜ ßÙ àµ âá äÞ åå çã è– éæ ëô ì« î¥ ðï òí óv õô ÷ñ øÄ úù üö ýå ÿû €É þ ƒ™ „¸ †¾ ˆ… ‰Ó ‹‡ ŒÓ Ž Š ‘å “ ”¢ •’ —Ì ˜ šÓ œ™ Ð Ÿ‘  Í ¢ž £Ê ¥­ ¦Ä ¨º © «Á ­ª ®¾ °‹ ±» ³˜ ´¸ ¶¥ ·µ ¹´ º ¼¯ ¾» ¿¬ ÁX Â© Äg Å¦ Çv È£ Ê… ËÕ Í™ Îá Ð¨ Ñí Ó· Ôù ÖÆ ×… ÙÕ Ú2 ÜÌ Ý; Þ? ßÛ á3 ãÌ ä; å? æâ è4 êÌ ë; ì? íé ï5 ñÌ ò; ó? ôð ö6 øÌ ù; ú? û÷ ý7 ÿÌ €; ? ‚þ „Ö †‹ ‡; ˆ? ‰… ‹Õ ¯ Œ Ž ’Á “‘ •Š –ù ˜— š¸ ›™ ” žÖ  ‹ ¡; ¢? £Ÿ ¥á §¬ ©¦ ª¨ ¬¾ ­« ¯¤ °’ ²à ³ƒ µ± ¶´ ¸® ¹è »¾ ¼º ¾¦ Àî Á½ Â¿ Ä· ÅÖ Ç‹ È; É? ÊÆ Ìí Î© ÐÍ ÑÏ Ó» ÔÒ ÖË ×™ Ùç Úý ÜØ ÝÛ ßÕ àè â» ãá åÍ çî èä éæ ëÞ ìÖ î‹ ï; ð? ñí ó¦ õ— ö¸ øô ù÷ ûò ü  þî ÿè ý ‚€ „ú …è ‡¸ ˆ† Š— Œî ‰ Ž…  ’ƒ “‘ •² –ú ˜” ™— ›‹ œš žƒ ŸÖ ¡‹ ¢; £? ¤  ¦£ ¨ ©² «§ ¬ª ®¥ ¯§ ±õ ²î ´° µ³ ·­ ¸  º  ¼¹ ½» ¿î Áî Â¾ Ãè Åè ÆÀ ÇÄ É¶ Ê£ Ì® ÎË ÏÍ Ñ Óü ÔÐ Õ² ×ô ØÒ ÙÖ ÛÈ Üƒ ÞÝ à âß ãú åä ç² éæ êè ìè íë ïá ñî òî óð õÚ ö ø€ úù ü÷ ýI ÿþ û ‚å „€ …œ †ƒ ˆ… ‰‘ ‹‹ Œ Š X ’‘ ”Ž •å —“ ˜Ã ™– ›Ÿ œž ž˜  Ÿ ¢ £g ¥¤ §¡ ¨å ª¦ «ê ¬© ®Æ ¯« ±¥ ³² µ° ¶v ¸· º´ »å ½¹ ¾ ¿¼ Áí Â¸ Ä² ÆÅ ÈÃ É… ËÊ ÍÇ Îå ÐÌ Ñô ÒÏ Ô  Õ Ø Ú Ü
 Þ à( ×( *0 ×0 2‚ „‚  Ÿ  Ö ×ñ óñ  œ  òò îî ïï íí á ðð ññå
 ïï å
Š ïï Šº
 ïï º
Ä	 ïï Ä	µ ïï µ¶ ïï ¶
 ïï 
Ù ïï Ùê ïï ê” ïï ”È
 ïï È
Þ ïï Þ¶ ïï ¶ 
 ïï  
™ ïï ™ø
 ïï ø
í ïï íá ïï áæ ïï æÕ	 ïï Õ	Ú
 ïï Ú
Ì ïï ÌÄ ïï Äµ ïï µ¡ ïï ¡¨ ïï ¨Þ ïï Þì
 ïï ì
Î ïï Îá ðð á­ ïï ­ñ ïï ñ· ïï ·û ïï û· ïï ·ñ ïï ñï ïï ï× ïï ×«
 ïï «
û ïï û– ïï –é ïï é¯	 ïï ¯	Ï ïï Ï’ ïï ’ˆ ïï ˆÆ ïï ÆŒ ïï Œ íí ú ïï ú©	 ïï ©	û	 ïï û	ù ïï ùî ïï îÏ ïï ÏŸ ïï Ÿ† ïï †„ ïï „ð ïï ð– ïï –Œ
 ïï Œ
ƒ ïï ƒÈ ïï ÈÑ ïï Ñæ ïï æ‘ ïï ‘± ïï ±® ïï ®÷ ïï ÷Ž ïï Ž¶ ïï ¶´ ïï ´Ã ïï Ã­ ïï ­Ü ïï Üî ïï î íí ã ïï ãÚ ïï Úž ïï ž” ïï ”¸	 ïï ¸	„
 ïï „
¨ ïï ¨ ïï ’ ïï ’±
 ïï ±
Ú ïï Ú ïï Ö
 ïï Ö
û ïï ûƒ ïï ƒ ïï Ø	 ïï Ø	Ö ïï Öœ ïï œŽ	 ïï Ž	Ï	 ïï Ï	 ïï à ïï àË ïï ËÉ ïï ÉÛ ïï ÛŸ ïï Ÿ¥ ïï ¥ÿ ïï ÿ… ïï …Ò ïï Ò  ïï  Ð ïï ÐÞ
 ïï Þ
½ ïï ½õ ïï õŽ ïï Ž– ïï –® ïï ®¦ ïï ¦¿ ïï ¿ý ïï ýº ïï º© ïï ©“ ïï “ï ïï ï  ïï  ž ïï žÒ ïï Ò¼ ïï ¼© ïï ©é ïï é¾ ïï ¾þ ïï þš ïï š íí ‹ ïï ‹„ ïï „Ž ïï ŽÈ ïï È£ ïï £ß ññ ßæ ïï æÛ ññ Ûõ	 ïï õ	å ïï å¦ ïï ¦± ïï ±À ïï Àñ ïï ñÎ ïï Îè ïï èÌ ïï ÌÏ ïï ÏÆ ïï Æ² ïï ²ñ ïï ñý ïï ý¦ ïï ¦ò ïï òÞ ïï Þè ïï è© ïï ©¹ ïï ¹ž	 ïï ž	Ì
 ïï Ì
Ã ïï Ã ïï ” ïï ”ô ïï ôÏ ïï Ï‚ ïï ‚Ú ïï Ú íí § ïï §· ïï ·Õ ïï ÕÇ ïï Ç ïï ê ïï êÖ ïï ÖÞ ïï Þ´
 ïï ´
Ä
 ïï Ä
ö ïï ö“ ïï “Í ïï Í ïï Ú ïï Ú§ ïï §þ	 ïï þ	æ	 ïï æ	» ïï »Â ïï Âà ðð à ïï ´ ïï ´ˆ ïï ˆ íí ˆ ïï ˆ« ïï «À ïï ÀÕ ïï Õæ ïï æ‡ ïï ‡ ïï × ññ ×  ïï  Ù ññ Ù¼ ïï ¼û ïï û‰ ïï ‰¡ ïï ¡¿ ïï ¿Ù ïï Ù¤ ïï ¤Ø ïï Ø²	 ïï ²	¯ ïï ¯€ ïï €© ïï ©£ ïï £î ïï îÆ ïï Æ– ïï –Ã ïï ÃË ïï ËÑ ïï ÑÌ ïï Ì ïï Ã ïï ÃÝ ññ Ýã ïï ã¢ ïï ¢Ê ïï Ê¹ ïï ¹´ ïï ´° ïï ° òò Ž ïï ŽÙ ïï Ùô ïï ôŽ ïï ŽÞ ïï Þñ ïï ñ³ ïï ³ö ïï ö”	 ïï ”	Ò ïï ÒÝ ïï Ýê	 ïï ê	Á ïï Á§ ïï §ß ïï ßË ïï Ëä ïï ä îî ¬ ïï ¬º ïï ºò ïï òº ïï º­ ïï ­† ïï †Þ	 ïï Þ	› ïï ›º ïï ºÂ ïï ÂÇ ïï ÇŠ ïï Š€ ïï €ô
 ïï ô
ˆ ïï ˆ™ ïï ™À	 ïï À	! îî !É ïï Éã ïï ãÌ ïï Ì	ó _	ó g
ó ­
ó µ
ó ù
ó 
ó ˜
ó ž
ó …
ó ¨
ó ¬
ó Ç	
ó —
ó ±
ó ¼
ó æ
ó ô
ó Æô ô ô ô ô 	ô ô ×ô Ùô Ûô Ýô ß	õ A	õ I	õ I	õ P	õ X	õ _	õ g	õ n	õ v	õ }
õ …
õ 
õ —
õ —
õ ž
õ ¦
õ ­
õ µ
õ ¼
õ Ä
õ Ë
õ Ó
õ Ý
õ ê
õ ò
õ ù
õ 
õ ˆ
õ 
õ —
õ Ÿ
õ ¦
õ ¯
õ µ
õ ¾
õ Ä
õ Í
õ Ó
õ Ü
õ â
õ ë
õ ñ
õ ú
õ €
õ €
õ ‹
õ ‘
õ ˜
õ ž
õ ¥
õ «
õ ²
õ ¸
õ ï
õ ú
õ …
õ 
õ ›
õ ©
õ ²
õ »
õ Ä
õ Í
õ Ö
õ ß
õ ß
õ þ
õ ¨
õ Ò
õ ˆ
õ ò
õ ò
õ –
õ ¡
õ ¬
õ ·
õ Â
õ Ð
õ Ù
õ â
õ ë
õ ô
õ ý
õ †	
õ †	
õ ¡	
õ Ç	
õ í	
õ £

õ 
õ 
õ •
õ •
õ —
õ —
õ ™
õ ™
õ ›
õ ›
õ 
õ ˜
õ ˜
õ ©
õ ©
õ º
õ º
õ ¹
õ ™
õ ™
õ ª
õ ª
õ »
õ »
õ …	ö 9	ö ;	ö =	ö ?
÷ û
÷ ¥
÷ Ï
÷ …
÷ Ý
÷ ž	
÷ Ä	
÷ ê	
÷  

÷ ø

÷ ”
÷ ¹
÷ Þ
÷ ‘
÷ ê
÷ Ì
÷ ñ
÷ –
÷ É
÷ ¢
÷ œ
÷ Ã
÷ ê
÷ 
÷ ô
ø Ÿ
ø º

ø ¬
ø ä
ø ¶
ù Í
ù Ö
ù ß
ù è
ù ñ
ù ú
ú é
ú „

ú ÷
ú ¯
ú ƒ
û ¢
û ´
û ½

û Ï

û ¯
û Á
û ç
û ù
û ¹
û Ë	ü 
ý ¦
ý ¯
ý ¸
ý Á
ý Ê
ý Óþ þ þ þ þ þ 
ÿ Ê
ÿ Ñ
ÿ å

ÿ ì

ÿ ×
ÿ Þ
ÿ 
ÿ –
ÿ á
ÿ è	€ $	€ ,
€ Š Ÿ É ï § ¹ È Ï × å í †  ² Å ¾	 ä	 Š
 Â
 Ô
 ã
 ê
 ò
 ³ Ø ý ´ Æ Õ Ü ä ë  µ ì þ  ” œ ½ ä ‰ ¾ Ð ß æ î
‚ õ
ƒ à
„ Ë
… é
… ˆ
… “
… ²
… ½
… Ú
… ã
… 
… ™
… Ž	
… ©	
… ²	
… Ï	
… Ø	
… õ	
… þ	
… «

… ´

… ˆ
… ž
… §
… Ã
… Ì
… è
… ñ
… 
… ¦
… À
… Ö
… ß
… û
… „
…  
… ©
… Õ
… Þ
… Ž
… ¨
… ±
… Ï
… Ø
… ô
… ý
… §
… °
† Ú
† ƒ	
‡ 
‡ †
‡ ›
‡  
‡ µ
‡ º
‡ Í
‡ Ò
‡ ã
‡ æ
‡ ï
‡ õ
‡ †
‡ Œ
‡ 
‡ £
‡ ´
‡ º
‡ Ë
‡ Ñ
‡ ©
‡ ³
‡ Á
‡ Ë
‡ Ù
‡ ã
‡ ñ
‡ û
‡ ‡
‡ 
‡ û
‡ Ž
‡ ¡
‡ ´
‡ Ç
ˆ ã
‰ ìŠ !	‹ 	‹ "	‹ P	‹ X
‹ ž
‹ ¦
‹ ê
‹ ò
‹ ‹
‹ ‘
‹ ú
‹ þ
‹ ¡
‹ ¡	
‹ ¥
‹ Õ
‹ —
‹ Ú
‹ Ï
‹ Ÿ
Œ Ã
Œ Þ

Œ Ð
Œ ˆ
Œ Ú	 
Ž ‚
Ž Æ
Ž Í
Ž 

Ž á

Ž è

Ž Ž
Ž Ó
Ž Ú
Ž Æ
Ž ‹
Ž ’
Ž š
Ž Ý
Ž ä à á
 ï
 ˆ
 Ÿ
 ´
 Ç
 €
 “
 ¦
 ¹
 Ì
‘ ±
‘ Ì

‘ ¾
‘ ö
‘ È	’ n	’ v
’ ¼
’ Ä
’ ˆ
’ 
’ ¥
’ «
’ 
’ Ò
’ ·
’ í	
’ Ü
’ ½
’ á
’ ò
’ ™
’ í
“ Œ
“ Ü
” ¬
” »
” Ê
” Ù
” è
” ÷
• “
– ë
– „
– ›
– °
– Ã	— }
— …
— Ë
— Ó
— —
— Ÿ
— ²
— ¸
— ›
— ˆ
— Â
— £

— É
— ”
— þ
— Ì
—  
˜ ñ
˜ 
˜ º
˜ à
˜ –
˜ ”	
˜ ¯	
˜ Õ	
˜ û	
˜ ±

˜ Ž
˜ ¤
˜ É
˜ î
˜ £
˜ Æ
˜ Ü
˜ 
˜ ¦
˜ Û
˜ ”
˜ ®
˜ Õ
˜ ú
˜ ­
™ ™
™ Ã
™ ¸	
™ Þ	
™ ­
™ Ò
™ å
™ Š
™ ·
™ Þ
š ÿ

š ™
š ³
š Ë
š á
š ò
š ‰
š  
š ·
š Î
š ®
š Æ
š Þ
š ö
š Š
› Œ
› "
compute_rhs5"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z3maxdd"
llvm.lifetime.end.p0i8"
llvm.memset.p0i8.i64*‘
npb-BT-compute_rhs5_S.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

devmap_label
 

wgsize_log1p
†fA

transfer_bytes
ø¬n
 
transfer_bytes_log1p
†fA

wgsize
<