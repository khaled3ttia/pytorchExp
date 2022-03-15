
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
3addB,
*
	full_text

%27 = add nsw i32 %10, -2
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
!br i1 %28, label %1006, label %29
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
!br i1 %32, label %1006, label %33
#i18B

	full_text


i1 %32
Qbitcast8BD
B
	full_text5
3
1%34 = bitcast double* %1 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%35 = bitcast double* %2 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%36 = bitcast double* %3 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%37 = bitcast double* %4 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%38 = bitcast double* %5 to [65 x [65 x double]]*
Qbitcast8BD
B
	full_text5
3
1%39 = bitcast double* %6 to [65 x [65 x double]]*
Wbitcast8BJ
H
	full_text;
9
7%40 = bitcast double* %0 to [65 x [65 x [5 x double]]]*
Wbitcast8BJ
H
	full_text;
9
7%41 = bitcast double* %7 to [65 x [65 x [5 x double]]]*
1shl8B(
&
	full_text

%42 = shl i64 %23, 32
%i648B

	full_text
	
i64 %23
9ashr8B/
-
	full_text 

%43 = ashr exact i64 %42, 32
%i648B

	full_text
	
i64 %42
1shl8B(
&
	full_text

%44 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
9ashr8B/
-
	full_text 

%45 = ashr exact i64 %44, 32
%i648B

	full_text
	
i64 %44
ôgetelementptr8BÖ
Ç
	full_textu
s
q%46 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 0, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Gbitcast8B:
8
	full_text+
)
'%47 = bitcast [5 x double]* %46 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %46
Hload8B>
<
	full_text/
-
+%48 = load i64, i64* %47, align 8, !tbaa !8
'i64*8B

	full_text


i64* %47
pgetelementptr8B]
[
	full_textN
L
J%49 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Gbitcast8B:
8
	full_text+
)
'%50 = bitcast [5 x double]* %12 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Istore8B>
<
	full_text/
-
+store i64 %48, i64* %50, align 16, !tbaa !8
%i648B

	full_text
	
i64 %48
'i64*8B

	full_text


i64* %50
†getelementptr8Bå
â
	full_text|
z
x%51 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 0, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%52 = bitcast double* %51 to i64*
-double*8B

	full_text

double* %51
Hload8B>
<
	full_text/
-
+%53 = load i64, i64* %52, align 8, !tbaa !8
'i64*8B

	full_text


i64* %52
pgetelementptr8B]
[
	full_textN
L
J%54 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%55 = bitcast double* %54 to i64*
-double*8B

	full_text

double* %54
Hstore8B=
;
	full_text.
,
*store i64 %53, i64* %55, align 8, !tbaa !8
%i648B

	full_text
	
i64 %53
'i64*8B

	full_text


i64* %55
†getelementptr8Bå
â
	full_text|
z
x%56 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 0, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%57 = bitcast double* %56 to i64*
-double*8B

	full_text

double* %56
Hload8B>
<
	full_text/
-
+%58 = load i64, i64* %57, align 8, !tbaa !8
'i64*8B

	full_text


i64* %57
pgetelementptr8B]
[
	full_textN
L
J%59 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%60 = bitcast double* %59 to i64*
-double*8B

	full_text

double* %59
Istore8B>
<
	full_text/
-
+store i64 %58, i64* %60, align 16, !tbaa !8
%i648B

	full_text
	
i64 %58
'i64*8B

	full_text


i64* %60
†getelementptr8Bå
â
	full_text|
z
x%61 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 0, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%62 = bitcast double* %61 to i64*
-double*8B

	full_text

double* %61
Hload8B>
<
	full_text/
-
+%63 = load i64, i64* %62, align 8, !tbaa !8
'i64*8B

	full_text


i64* %62
pgetelementptr8B]
[
	full_textN
L
J%64 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%65 = bitcast double* %64 to i64*
-double*8B

	full_text

double* %64
Hstore8B=
;
	full_text.
,
*store i64 %63, i64* %65, align 8, !tbaa !8
%i648B

	full_text
	
i64 %63
'i64*8B

	full_text


i64* %65
†getelementptr8Bå
â
	full_text|
z
x%66 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 0, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%67 = bitcast double* %66 to i64*
-double*8B

	full_text

double* %66
Hload8B>
<
	full_text/
-
+%68 = load i64, i64* %67, align 8, !tbaa !8
'i64*8B

	full_text


i64* %67
pgetelementptr8B]
[
	full_textN
L
J%69 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%70 = bitcast double* %69 to i64*
-double*8B

	full_text

double* %69
Istore8B>
<
	full_text/
-
+store i64 %68, i64* %70, align 16, !tbaa !8
%i648B

	full_text
	
i64 %68
'i64*8B

	full_text


i64* %70
ôgetelementptr8BÖ
Ç
	full_textu
s
q%71 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 1, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Gbitcast8B:
8
	full_text+
)
'%72 = bitcast [5 x double]* %71 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %71
Hload8B>
<
	full_text/
-
+%73 = load i64, i64* %72, align 8, !tbaa !8
'i64*8B

	full_text


i64* %72
pgetelementptr8B]
[
	full_textN
L
J%74 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Gbitcast8B:
8
	full_text+
)
'%75 = bitcast [5 x double]* %13 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Istore8B>
<
	full_text/
-
+store i64 %73, i64* %75, align 16, !tbaa !8
%i648B

	full_text
	
i64 %73
'i64*8B

	full_text


i64* %75
†getelementptr8Bå
â
	full_text|
z
x%76 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 1, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%77 = bitcast double* %76 to i64*
-double*8B

	full_text

double* %76
Hload8B>
<
	full_text/
-
+%78 = load i64, i64* %77, align 8, !tbaa !8
'i64*8B

	full_text


i64* %77
pgetelementptr8B]
[
	full_textN
L
J%79 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%80 = bitcast double* %79 to i64*
-double*8B

	full_text

double* %79
Hstore8B=
;
	full_text.
,
*store i64 %78, i64* %80, align 8, !tbaa !8
%i648B

	full_text
	
i64 %78
'i64*8B

	full_text


i64* %80
†getelementptr8Bå
â
	full_text|
z
x%81 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 1, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%82 = bitcast double* %81 to i64*
-double*8B

	full_text

double* %81
Hload8B>
<
	full_text/
-
+%83 = load i64, i64* %82, align 8, !tbaa !8
'i64*8B

	full_text


i64* %82
pgetelementptr8B]
[
	full_textN
L
J%84 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%85 = bitcast double* %84 to i64*
-double*8B

	full_text

double* %84
Istore8B>
<
	full_text/
-
+store i64 %83, i64* %85, align 16, !tbaa !8
%i648B

	full_text
	
i64 %83
'i64*8B

	full_text


i64* %85
†getelementptr8Bå
â
	full_text|
z
x%86 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 1, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%87 = bitcast double* %86 to i64*
-double*8B

	full_text

double* %86
Hload8B>
<
	full_text/
-
+%88 = load i64, i64* %87, align 8, !tbaa !8
'i64*8B

	full_text


i64* %87
pgetelementptr8B]
[
	full_textN
L
J%89 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%90 = bitcast double* %89 to i64*
-double*8B

	full_text

double* %89
Hstore8B=
;
	full_text.
,
*store i64 %88, i64* %90, align 8, !tbaa !8
%i648B

	full_text
	
i64 %88
'i64*8B

	full_text


i64* %90
†getelementptr8Bå
â
	full_text|
z
x%91 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 1, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Abitcast8B4
2
	full_text%
#
!%92 = bitcast double* %91 to i64*
-double*8B

	full_text

double* %91
Hload8B>
<
	full_text/
-
+%93 = load i64, i64* %92, align 8, !tbaa !8
'i64*8B

	full_text


i64* %92
pgetelementptr8B]
[
	full_textN
L
J%94 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%95 = bitcast double* %94 to i64*
-double*8B

	full_text

double* %94
ôgetelementptr8BÖ
Ç
	full_textu
s
q%96 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 2, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Gbitcast8B:
8
	full_text+
)
'%97 = bitcast [5 x double]* %96 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %96
Hload8B>
<
	full_text/
-
+%98 = load i64, i64* %97, align 8, !tbaa !8
'i64*8B

	full_text


i64* %97
Gbitcast8B:
8
	full_text+
)
'%99 = bitcast [5 x double]* %14 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
°getelementptr8Bç
ä
	full_text}
{
y%100 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 2, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%101 = bitcast double* %100 to i64*
.double*8B

	full_text

double* %100
Jload8B@
>
	full_text1
/
-%102 = load i64, i64* %101, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %101
qgetelementptr8B^
\
	full_textO
M
K%103 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%104 = bitcast double* %103 to i64*
.double*8B

	full_text

double* %103
°getelementptr8Bç
ä
	full_text}
{
y%105 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 2, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%106 = bitcast double* %105 to i64*
.double*8B

	full_text

double* %105
Jload8B@
>
	full_text1
/
-%107 = load i64, i64* %106, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %106
qgetelementptr8B^
\
	full_textO
M
K%108 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%109 = bitcast double* %108 to i64*
.double*8B

	full_text

double* %108
°getelementptr8Bç
ä
	full_text}
{
y%110 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 2, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%111 = bitcast double* %110 to i64*
.double*8B

	full_text

double* %110
Jload8B@
>
	full_text1
/
-%112 = load i64, i64* %111, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %111
qgetelementptr8B^
\
	full_textO
M
K%113 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%114 = bitcast double* %113 to i64*
.double*8B

	full_text

double* %113
°getelementptr8Bç
ä
	full_text}
{
y%115 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 2, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%116 = bitcast double* %115 to i64*
.double*8B

	full_text

double* %115
Jload8B@
>
	full_text1
/
-%117 = load i64, i64* %116, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %116
qgetelementptr8B^
\
	full_textO
M
K%118 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%119 = bitcast double* %118 to i64*
.double*8B

	full_text

double* %118
ågetelementptr8By
w
	full_textj
h
f%120 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 0, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
ågetelementptr8By
w
	full_textj
h
f%122 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 1, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%123 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
ågetelementptr8By
w
	full_textj
h
f%124 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 0, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%125 = load double, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
ågetelementptr8By
w
	full_textj
h
f%126 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 1, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%127 = load double, double* %126, align 8, !tbaa !8
.double*8B

	full_text

double* %126
ågetelementptr8By
w
	full_textj
h
f%128 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 0, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
ågetelementptr8By
w
	full_textj
h
f%130 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 1, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%131 = load double, double* %130, align 8, !tbaa !8
.double*8B

	full_text

double* %130
ågetelementptr8By
w
	full_textj
h
f%132 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 0, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%133 = load double, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
ågetelementptr8By
w
	full_textj
h
f%134 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 1, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%135 = load double, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
ågetelementptr8By
w
	full_textj
h
f%136 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 0, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%137 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
ågetelementptr8By
w
	full_textj
h
f%138 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 1, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%139 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
ågetelementptr8By
w
	full_textj
h
f%140 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 0, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%141 = load double, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
ågetelementptr8By
w
	full_textj
h
f%142 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 1, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%143 = load double, double* %142, align 8, !tbaa !8
.double*8B

	full_text

double* %142
qgetelementptr8B^
\
	full_textO
M
K%144 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Hbitcast8B;
9
	full_text,
*
(%145 = bitcast [5 x double]* %16 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Kload8BA
?
	full_text2
0
.%146 = load i64, i64* %145, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %145
Hbitcast8B;
9
	full_text,
*
(%147 = bitcast [5 x double]* %15 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Kstore8B@
>
	full_text1
/
-store i64 %146, i64* %147, align 16, !tbaa !8
&i648B

	full_text


i64 %146
(i64*8B

	full_text

	i64* %147
qgetelementptr8B^
\
	full_textO
M
K%148 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%149 = bitcast double* %148 to i64*
.double*8B

	full_text

double* %148
Jload8B@
>
	full_text1
/
-%150 = load i64, i64* %149, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %149
qgetelementptr8B^
\
	full_textO
M
K%151 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%152 = bitcast double* %151 to i64*
.double*8B

	full_text

double* %151
Jstore8B?
=
	full_text0
.
,store i64 %150, i64* %152, align 8, !tbaa !8
&i648B

	full_text


i64 %150
(i64*8B

	full_text

	i64* %152
qgetelementptr8B^
\
	full_textO
M
K%153 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%154 = bitcast double* %153 to i64*
.double*8B

	full_text

double* %153
Kload8BA
?
	full_text2
0
.%155 = load i64, i64* %154, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %154
qgetelementptr8B^
\
	full_textO
M
K%156 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%157 = bitcast double* %156 to i64*
.double*8B

	full_text

double* %156
Kstore8B@
>
	full_text1
/
-store i64 %155, i64* %157, align 16, !tbaa !8
&i648B

	full_text


i64 %155
(i64*8B

	full_text

	i64* %157
qgetelementptr8B^
\
	full_textO
M
K%158 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%159 = bitcast double* %158 to i64*
.double*8B

	full_text

double* %158
Jload8B@
>
	full_text1
/
-%160 = load i64, i64* %159, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %159
qgetelementptr8B^
\
	full_textO
M
K%161 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%162 = bitcast double* %161 to i64*
.double*8B

	full_text

double* %161
Jstore8B?
=
	full_text0
.
,store i64 %160, i64* %162, align 8, !tbaa !8
&i648B

	full_text


i64 %160
(i64*8B

	full_text

	i64* %162
qgetelementptr8B^
\
	full_textO
M
K%163 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 4
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
Kload8BA
?
	full_text2
0
.%165 = load i64, i64* %164, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %164
qgetelementptr8B^
\
	full_textO
M
K%166 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 4
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
Kstore8B@
>
	full_text1
/
-store i64 %165, i64* %167, align 16, !tbaa !8
&i648B

	full_text


i64 %165
(i64*8B

	full_text

	i64* %167
Jstore8B?
=
	full_text0
.
,store i64 %48, i64* %145, align 16, !tbaa !8
%i648B

	full_text
	
i64 %48
(i64*8B

	full_text

	i64* %145
Istore8B>
<
	full_text/
-
+store i64 %53, i64* %149, align 8, !tbaa !8
%i648B

	full_text
	
i64 %53
(i64*8B

	full_text

	i64* %149
Jstore8B?
=
	full_text0
.
,store i64 %58, i64* %154, align 16, !tbaa !8
%i648B

	full_text
	
i64 %58
(i64*8B

	full_text

	i64* %154
Istore8B>
<
	full_text/
-
+store i64 %63, i64* %159, align 8, !tbaa !8
%i648B

	full_text
	
i64 %63
(i64*8B

	full_text

	i64* %159
Jstore8B?
=
	full_text0
.
,store i64 %68, i64* %164, align 16, !tbaa !8
%i648B

	full_text
	
i64 %68
(i64*8B

	full_text

	i64* %164
Istore8B>
<
	full_text/
-
+store i64 %73, i64* %50, align 16, !tbaa !8
%i648B

	full_text
	
i64 %73
'i64*8B

	full_text


i64* %50
Hstore8B=
;
	full_text.
,
*store i64 %78, i64* %55, align 8, !tbaa !8
%i648B

	full_text
	
i64 %78
'i64*8B

	full_text


i64* %55
Istore8B>
<
	full_text/
-
+store i64 %83, i64* %60, align 16, !tbaa !8
%i648B

	full_text
	
i64 %83
'i64*8B

	full_text


i64* %60
Hstore8B=
;
	full_text.
,
*store i64 %88, i64* %65, align 8, !tbaa !8
%i648B

	full_text
	
i64 %88
'i64*8B

	full_text


i64* %65
Istore8B>
<
	full_text/
-
+store i64 %93, i64* %70, align 16, !tbaa !8
%i648B

	full_text
	
i64 %93
'i64*8B

	full_text


i64* %70
Istore8B>
<
	full_text/
-
+store i64 %98, i64* %75, align 16, !tbaa !8
%i648B

	full_text
	
i64 %98
'i64*8B

	full_text


i64* %75
Istore8B>
<
	full_text/
-
+store i64 %102, i64* %80, align 8, !tbaa !8
&i648B

	full_text


i64 %102
'i64*8B

	full_text


i64* %80
Jstore8B?
=
	full_text0
.
,store i64 %107, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %107
'i64*8B

	full_text


i64* %85
Istore8B>
<
	full_text/
-
+store i64 %112, i64* %90, align 8, !tbaa !8
&i648B

	full_text


i64 %112
'i64*8B

	full_text


i64* %90
Jstore8B?
=
	full_text0
.
,store i64 %117, i64* %95, align 16, !tbaa !8
&i648B

	full_text


i64 %117
'i64*8B

	full_text


i64* %95
ögetelementptr8BÜ
É
	full_textv
t
r%168 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 3, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Ibitcast8B<
:
	full_text-
+
)%169 = bitcast [5 x double]* %168 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %168
Jload8B@
>
	full_text1
/
-%170 = load i64, i64* %169, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %169
Jstore8B?
=
	full_text0
.
,store i64 %170, i64* %99, align 16, !tbaa !8
&i648B

	full_text


i64 %170
'i64*8B

	full_text


i64* %99
°getelementptr8Bç
ä
	full_text}
{
y%171 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 3, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%172 = bitcast double* %171 to i64*
.double*8B

	full_text

double* %171
Jload8B@
>
	full_text1
/
-%173 = load i64, i64* %172, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %172
Jstore8B?
=
	full_text0
.
,store i64 %173, i64* %104, align 8, !tbaa !8
&i648B

	full_text


i64 %173
(i64*8B

	full_text

	i64* %104
°getelementptr8Bç
ä
	full_text}
{
y%174 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 3, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%175 = bitcast double* %174 to i64*
.double*8B

	full_text

double* %174
Jload8B@
>
	full_text1
/
-%176 = load i64, i64* %175, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %175
Kstore8B@
>
	full_text1
/
-store i64 %176, i64* %109, align 16, !tbaa !8
&i648B

	full_text


i64 %176
(i64*8B

	full_text

	i64* %109
°getelementptr8Bç
ä
	full_text}
{
y%177 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 3, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%178 = bitcast double* %177 to i64*
.double*8B

	full_text

double* %177
Jload8B@
>
	full_text1
/
-%179 = load i64, i64* %178, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %178
Jstore8B?
=
	full_text0
.
,store i64 %179, i64* %114, align 8, !tbaa !8
&i648B

	full_text


i64 %179
(i64*8B

	full_text

	i64* %114
°getelementptr8Bç
ä
	full_text}
{
y%180 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 3, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%181 = bitcast double* %180 to i64*
.double*8B

	full_text

double* %180
Jload8B@
>
	full_text1
/
-%182 = load i64, i64* %181, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %181
Kstore8B@
>
	full_text1
/
-store i64 %182, i64* %119, align 16, !tbaa !8
&i648B

	full_text


i64 %182
(i64*8B

	full_text

	i64* %119
ågetelementptr8By
w
	full_textj
h
f%183 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 2, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%184 = load double, double* %183, align 8, !tbaa !8
.double*8B

	full_text

double* %183
ågetelementptr8By
w
	full_textj
h
f%185 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 2, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%186 = load double, double* %185, align 8, !tbaa !8
.double*8B

	full_text

double* %185
ågetelementptr8By
w
	full_textj
h
f%187 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 2, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%188 = load double, double* %187, align 8, !tbaa !8
.double*8B

	full_text

double* %187
ågetelementptr8By
w
	full_textj
h
f%189 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 2, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%190 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
ågetelementptr8By
w
	full_textj
h
f%191 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 2, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%192 = load double, double* %191, align 8, !tbaa !8
.double*8B

	full_text

double* %191
ågetelementptr8By
w
	full_textj
h
f%193 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 2, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%194 = load double, double* %193, align 8, !tbaa !8
.double*8B

	full_text

double* %193
°getelementptr8Bç
ä
	full_text}
{
y%195 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 1, i64 %45, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%196 = load double, double* %195, align 8, !tbaa !8
.double*8B

	full_text

double* %195
@bitcast8B3
1
	full_text$
"
 %197 = bitcast i64 %98 to double
%i648B

	full_text
	
i64 %98
@bitcast8B3
1
	full_text$
"
 %198 = bitcast i64 %73 to double
%i648B

	full_text
	
i64 %73
vcall8Bl
j
	full_text]
[
Y%199 = tail call double @llvm.fmuladd.f64(double %198, double -2.000000e+00, double %197)
,double8B

	full_text

double %198
,double8B

	full_text

double %197
@bitcast8B3
1
	full_text$
"
 %200 = bitcast i64 %48 to double
%i648B

	full_text
	
i64 %48
:fadd8B0
.
	full_text!

%201 = fadd double %199, %200
,double8B

	full_text

double %199
,double8B

	full_text

double %200
{call8Bq
o
	full_textb
`
^%202 = tail call double @llvm.fmuladd.f64(double %201, double 0x40A7418000000001, double %196)
,double8B

	full_text

double %201
,double8B

	full_text

double %196
Abitcast8B4
2
	full_text%
#
!%203 = bitcast i64 %107 to double
&i648B

	full_text


i64 %107
@bitcast8B3
1
	full_text$
"
 %204 = bitcast i64 %58 to double
%i648B

	full_text
	
i64 %58
:fsub8B0
.
	full_text!

%205 = fsub double %203, %204
,double8B

	full_text

double %203
,double8B

	full_text

double %204
vcall8Bl
j
	full_text]
[
Y%206 = tail call double @llvm.fmuladd.f64(double %205, double -3.150000e+01, double %202)
,double8B

	full_text

double %205
,double8B

	full_text

double %202
°getelementptr8Bç
ä
	full_text}
{
y%207 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 1, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%208 = load double, double* %207, align 8, !tbaa !8
.double*8B

	full_text

double* %207
Abitcast8B4
2
	full_text%
#
!%209 = bitcast i64 %102 to double
&i648B

	full_text


i64 %102
@bitcast8B3
1
	full_text$
"
 %210 = bitcast i64 %78 to double
%i648B

	full_text
	
i64 %78
vcall8Bl
j
	full_text]
[
Y%211 = tail call double @llvm.fmuladd.f64(double %210, double -2.000000e+00, double %209)
,double8B

	full_text

double %210
,double8B

	full_text

double %209
@bitcast8B3
1
	full_text$
"
 %212 = bitcast i64 %53 to double
%i648B

	full_text
	
i64 %53
:fadd8B0
.
	full_text!

%213 = fadd double %211, %212
,double8B

	full_text

double %211
,double8B

	full_text

double %212
{call8Bq
o
	full_textb
`
^%214 = tail call double @llvm.fmuladd.f64(double %213, double 0x40A7418000000001, double %208)
,double8B

	full_text

double %213
,double8B

	full_text

double %208
vcall8Bl
j
	full_text]
[
Y%215 = tail call double @llvm.fmuladd.f64(double %123, double -2.000000e+00, double %184)
,double8B

	full_text

double %123
,double8B

	full_text

double %184
:fadd8B0
.
	full_text!

%216 = fadd double %121, %215
,double8B

	full_text

double %121
,double8B

	full_text

double %215
{call8Bq
o
	full_textb
`
^%217 = tail call double @llvm.fmuladd.f64(double %216, double 0x4078CE6666666667, double %214)
,double8B

	full_text

double %216
,double8B

	full_text

double %214
:fmul8B0
.
	full_text!

%218 = fmul double %125, %212
,double8B

	full_text

double %125
,double8B

	full_text

double %212
Cfsub8B9
7
	full_text*
(
&%219 = fsub double -0.000000e+00, %218
,double8B

	full_text

double %218
mcall8Bc
a
	full_textT
R
P%220 = tail call double @llvm.fmuladd.f64(double %209, double %186, double %219)
,double8B

	full_text

double %209
,double8B

	full_text

double %186
,double8B

	full_text

double %219
vcall8Bl
j
	full_text]
[
Y%221 = tail call double @llvm.fmuladd.f64(double %220, double -3.150000e+01, double %217)
,double8B

	full_text

double %220
,double8B

	full_text

double %217
°getelementptr8Bç
ä
	full_text}
{
y%222 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 1, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%223 = load double, double* %222, align 8, !tbaa !8
.double*8B

	full_text

double* %222
@bitcast8B3
1
	full_text$
"
 %224 = bitcast i64 %83 to double
%i648B

	full_text
	
i64 %83
vcall8Bl
j
	full_text]
[
Y%225 = tail call double @llvm.fmuladd.f64(double %224, double -2.000000e+00, double %203)
,double8B

	full_text

double %224
,double8B

	full_text

double %203
:fadd8B0
.
	full_text!

%226 = fadd double %225, %204
,double8B

	full_text

double %225
,double8B

	full_text

double %204
{call8Bq
o
	full_textb
`
^%227 = tail call double @llvm.fmuladd.f64(double %226, double 0x40A7418000000001, double %223)
,double8B

	full_text

double %226
,double8B

	full_text

double %223
vcall8Bl
j
	full_text]
[
Y%228 = tail call double @llvm.fmuladd.f64(double %127, double -2.000000e+00, double %186)
,double8B

	full_text

double %127
,double8B

	full_text

double %186
:fadd8B0
.
	full_text!

%229 = fadd double %125, %228
,double8B

	full_text

double %125
,double8B

	full_text

double %228
ucall8Bk
i
	full_text\
Z
X%230 = tail call double @llvm.fmuladd.f64(double %229, double 5.292000e+02, double %227)
,double8B

	full_text

double %229
,double8B

	full_text

double %227
:fmul8B0
.
	full_text!

%231 = fmul double %125, %204
,double8B

	full_text

double %125
,double8B

	full_text

double %204
Cfsub8B9
7
	full_text*
(
&%232 = fsub double -0.000000e+00, %231
,double8B

	full_text

double %231
mcall8Bc
a
	full_textT
R
P%233 = tail call double @llvm.fmuladd.f64(double %203, double %186, double %232)
,double8B

	full_text

double %203
,double8B

	full_text

double %186
,double8B

	full_text

double %232
Abitcast8B4
2
	full_text%
#
!%234 = bitcast i64 %117 to double
&i648B

	full_text


i64 %117
:fsub8B0
.
	full_text!

%235 = fsub double %234, %194
,double8B

	full_text

double %234
,double8B

	full_text

double %194
@bitcast8B3
1
	full_text$
"
 %236 = bitcast i64 %68 to double
%i648B

	full_text
	
i64 %68
:fsub8B0
.
	full_text!

%237 = fsub double %235, %236
,double8B

	full_text

double %235
,double8B

	full_text

double %236
:fadd8B0
.
	full_text!

%238 = fadd double %141, %237
,double8B

	full_text

double %141
,double8B

	full_text

double %237
ucall8Bk
i
	full_text\
Z
X%239 = tail call double @llvm.fmuladd.f64(double %238, double 4.000000e-01, double %233)
,double8B

	full_text

double %238
,double8B

	full_text

double %233
vcall8Bl
j
	full_text]
[
Y%240 = tail call double @llvm.fmuladd.f64(double %239, double -3.150000e+01, double %230)
,double8B

	full_text

double %239
,double8B

	full_text

double %230
°getelementptr8Bç
ä
	full_text}
{
y%241 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 1, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%242 = load double, double* %241, align 8, !tbaa !8
.double*8B

	full_text

double* %241
Abitcast8B4
2
	full_text%
#
!%243 = bitcast i64 %112 to double
&i648B

	full_text


i64 %112
@bitcast8B3
1
	full_text$
"
 %244 = bitcast i64 %88 to double
%i648B

	full_text
	
i64 %88
vcall8Bl
j
	full_text]
[
Y%245 = tail call double @llvm.fmuladd.f64(double %244, double -2.000000e+00, double %243)
,double8B

	full_text

double %244
,double8B

	full_text

double %243
Pload8BF
D
	full_text7
5
3%246 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
:fadd8B0
.
	full_text!

%247 = fadd double %245, %246
,double8B

	full_text

double %245
,double8B

	full_text

double %246
{call8Bq
o
	full_textb
`
^%248 = tail call double @llvm.fmuladd.f64(double %247, double 0x40A7418000000001, double %242)
,double8B

	full_text

double %247
,double8B

	full_text

double %242
vcall8Bl
j
	full_text]
[
Y%249 = tail call double @llvm.fmuladd.f64(double %131, double -2.000000e+00, double %188)
,double8B

	full_text

double %131
,double8B

	full_text

double %188
:fadd8B0
.
	full_text!

%250 = fadd double %129, %249
,double8B

	full_text

double %129
,double8B

	full_text

double %249
{call8Bq
o
	full_textb
`
^%251 = tail call double @llvm.fmuladd.f64(double %250, double 0x4078CE6666666667, double %248)
,double8B

	full_text

double %250
,double8B

	full_text

double %248
:fmul8B0
.
	full_text!

%252 = fmul double %125, %246
,double8B

	full_text

double %125
,double8B

	full_text

double %246
Cfsub8B9
7
	full_text*
(
&%253 = fsub double -0.000000e+00, %252
,double8B

	full_text

double %252
mcall8Bc
a
	full_textT
R
P%254 = tail call double @llvm.fmuladd.f64(double %243, double %186, double %253)
,double8B

	full_text

double %243
,double8B

	full_text

double %186
,double8B

	full_text

double %253
vcall8Bl
j
	full_text]
[
Y%255 = tail call double @llvm.fmuladd.f64(double %254, double -3.150000e+01, double %251)
,double8B

	full_text

double %254
,double8B

	full_text

double %251
°getelementptr8Bç
ä
	full_text}
{
y%256 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 1, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%257 = load double, double* %256, align 8, !tbaa !8
.double*8B

	full_text

double* %256
Pload8BF
D
	full_text7
5
3%258 = load double, double* %69, align 16, !tbaa !8
-double*8B

	full_text

double* %69
vcall8Bl
j
	full_text]
[
Y%259 = tail call double @llvm.fmuladd.f64(double %258, double -2.000000e+00, double %234)
,double8B

	full_text

double %258
,double8B

	full_text

double %234
:fadd8B0
.
	full_text!

%260 = fadd double %259, %236
,double8B

	full_text

double %259
,double8B

	full_text

double %236
{call8Bq
o
	full_textb
`
^%261 = tail call double @llvm.fmuladd.f64(double %260, double 0x40A7418000000001, double %257)
,double8B

	full_text

double %260
,double8B

	full_text

double %257
vcall8Bl
j
	full_text]
[
Y%262 = tail call double @llvm.fmuladd.f64(double %135, double -2.000000e+00, double %190)
,double8B

	full_text

double %135
,double8B

	full_text

double %190
:fadd8B0
.
	full_text!

%263 = fadd double %133, %262
,double8B

	full_text

double %133
,double8B

	full_text

double %262
{call8Bq
o
	full_textb
`
^%264 = tail call double @llvm.fmuladd.f64(double %263, double 0xC077D0624DD2F1A9, double %261)
,double8B

	full_text

double %263
,double8B

	full_text

double %261
Bfmul8B8
6
	full_text)
'
%%265 = fmul double %127, 2.000000e+00
,double8B

	full_text

double %127
:fmul8B0
.
	full_text!

%266 = fmul double %127, %265
,double8B

	full_text

double %127
,double8B

	full_text

double %265
Cfsub8B9
7
	full_text*
(
&%267 = fsub double -0.000000e+00, %266
,double8B

	full_text

double %266
mcall8Bc
a
	full_textT
R
P%268 = tail call double @llvm.fmuladd.f64(double %186, double %186, double %267)
,double8B

	full_text

double %186
,double8B

	full_text

double %186
,double8B

	full_text

double %267
mcall8Bc
a
	full_textT
R
P%269 = tail call double @llvm.fmuladd.f64(double %125, double %125, double %268)
,double8B

	full_text

double %125
,double8B

	full_text

double %125
,double8B

	full_text

double %268
ucall8Bk
i
	full_text\
Z
X%270 = tail call double @llvm.fmuladd.f64(double %269, double 6.615000e+01, double %264)
,double8B

	full_text

double %269
,double8B

	full_text

double %264
Bfmul8B8
6
	full_text)
'
%%271 = fmul double %258, 2.000000e+00
,double8B

	full_text

double %258
:fmul8B0
.
	full_text!

%272 = fmul double %139, %271
,double8B

	full_text

double %139
,double8B

	full_text

double %271
Cfsub8B9
7
	full_text*
(
&%273 = fsub double -0.000000e+00, %272
,double8B

	full_text

double %272
mcall8Bc
a
	full_textT
R
P%274 = tail call double @llvm.fmuladd.f64(double %234, double %192, double %273)
,double8B

	full_text

double %234
,double8B

	full_text

double %192
,double8B

	full_text

double %273
mcall8Bc
a
	full_textT
R
P%275 = tail call double @llvm.fmuladd.f64(double %236, double %137, double %274)
,double8B

	full_text

double %236
,double8B

	full_text

double %137
,double8B

	full_text

double %274
{call8Bq
o
	full_textb
`
^%276 = tail call double @llvm.fmuladd.f64(double %275, double 0x40884F645A1CAC08, double %270)
,double8B

	full_text

double %275
,double8B

	full_text

double %270
Bfmul8B8
6
	full_text)
'
%%277 = fmul double %194, 4.000000e-01
,double8B

	full_text

double %194
Cfsub8B9
7
	full_text*
(
&%278 = fsub double -0.000000e+00, %277
,double8B

	full_text

double %277
ucall8Bk
i
	full_text\
Z
X%279 = tail call double @llvm.fmuladd.f64(double %234, double 1.400000e+00, double %278)
,double8B

	full_text

double %234
,double8B

	full_text

double %278
Bfmul8B8
6
	full_text)
'
%%280 = fmul double %141, 4.000000e-01
,double8B

	full_text

double %141
Cfsub8B9
7
	full_text*
(
&%281 = fsub double -0.000000e+00, %280
,double8B

	full_text

double %280
ucall8Bk
i
	full_text\
Z
X%282 = tail call double @llvm.fmuladd.f64(double %236, double 1.400000e+00, double %281)
,double8B

	full_text

double %236
,double8B

	full_text

double %281
:fmul8B0
.
	full_text!

%283 = fmul double %125, %282
,double8B

	full_text

double %125
,double8B

	full_text

double %282
Cfsub8B9
7
	full_text*
(
&%284 = fsub double -0.000000e+00, %283
,double8B

	full_text

double %283
mcall8Bc
a
	full_textT
R
P%285 = tail call double @llvm.fmuladd.f64(double %279, double %186, double %284)
,double8B

	full_text

double %279
,double8B

	full_text

double %186
,double8B

	full_text

double %284
vcall8Bl
j
	full_text]
[
Y%286 = tail call double @llvm.fmuladd.f64(double %285, double -3.150000e+01, double %276)
,double8B

	full_text

double %285
,double8B

	full_text

double %276
kcall8Ba
_
	full_textR
P
N%287 = tail call double @_Z3maxdd(double 7.500000e-01, double 1.000000e+00) #5
ccall8BY
W
	full_textJ
H
F%288 = tail call double @_Z3maxdd(double 7.500000e-01, double %287) #5
,double8B

	full_text

double %287
Bfmul8B8
6
	full_text)
'
%%289 = fmul double %288, 2.500000e-01
,double8B

	full_text

double %288
Cfsub8B9
7
	full_text*
(
&%290 = fsub double -0.000000e+00, %289
,double8B

	full_text

double %289
Pload8BF
D
	full_text7
5
3%291 = load double, double* %49, align 16, !tbaa !8
-double*8B

	full_text

double* %49
Pload8BF
D
	full_text7
5
3%292 = load double, double* %74, align 16, !tbaa !8
-double*8B

	full_text

double* %74
Bfmul8B8
6
	full_text)
'
%%293 = fmul double %292, 4.000000e+00
,double8B

	full_text

double %292
Cfsub8B9
7
	full_text*
(
&%294 = fsub double -0.000000e+00, %293
,double8B

	full_text

double %293
ucall8Bk
i
	full_text\
Z
X%295 = tail call double @llvm.fmuladd.f64(double %291, double 5.000000e+00, double %294)
,double8B

	full_text

double %291
,double8B

	full_text

double %294
qgetelementptr8B^
\
	full_textO
M
K%296 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Qload8BG
E
	full_text8
6
4%297 = load double, double* %296, align 16, !tbaa !8
.double*8B

	full_text

double* %296
:fadd8B0
.
	full_text!

%298 = fadd double %297, %295
,double8B

	full_text

double %297
,double8B

	full_text

double %295
mcall8Bc
a
	full_textT
R
P%299 = tail call double @llvm.fmuladd.f64(double %290, double %298, double %206)
,double8B

	full_text

double %290
,double8B

	full_text

double %298
,double8B

	full_text

double %206
Pstore8BE
C
	full_text6
4
2store double %299, double* %195, align 8, !tbaa !8
,double8B

	full_text

double %299
.double*8B

	full_text

double* %195
Oload8BE
C
	full_text6
4
2%300 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
Oload8BE
C
	full_text6
4
2%301 = load double, double* %79, align 8, !tbaa !8
-double*8B

	full_text

double* %79
Bfmul8B8
6
	full_text)
'
%%302 = fmul double %301, 4.000000e+00
,double8B

	full_text

double %301
Cfsub8B9
7
	full_text*
(
&%303 = fsub double -0.000000e+00, %302
,double8B

	full_text

double %302
ucall8Bk
i
	full_text\
Z
X%304 = tail call double @llvm.fmuladd.f64(double %300, double 5.000000e+00, double %303)
,double8B

	full_text

double %300
,double8B

	full_text

double %303
Pload8BF
D
	full_text7
5
3%305 = load double, double* %103, align 8, !tbaa !8
.double*8B

	full_text

double* %103
:fadd8B0
.
	full_text!

%306 = fadd double %305, %304
,double8B

	full_text

double %305
,double8B

	full_text

double %304
mcall8Bc
a
	full_textT
R
P%307 = tail call double @llvm.fmuladd.f64(double %290, double %306, double %221)
,double8B

	full_text

double %290
,double8B

	full_text

double %306
,double8B

	full_text

double %221
Pstore8BE
C
	full_text6
4
2store double %307, double* %207, align 8, !tbaa !8
,double8B

	full_text

double %307
.double*8B

	full_text

double* %207
Pload8BF
D
	full_text7
5
3%308 = load double, double* %59, align 16, !tbaa !8
-double*8B

	full_text

double* %59
Pload8BF
D
	full_text7
5
3%309 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
Bfmul8B8
6
	full_text)
'
%%310 = fmul double %309, 4.000000e+00
,double8B

	full_text

double %309
Cfsub8B9
7
	full_text*
(
&%311 = fsub double -0.000000e+00, %310
,double8B

	full_text

double %310
ucall8Bk
i
	full_text\
Z
X%312 = tail call double @llvm.fmuladd.f64(double %308, double 5.000000e+00, double %311)
,double8B

	full_text

double %308
,double8B

	full_text

double %311
Qload8BG
E
	full_text8
6
4%313 = load double, double* %108, align 16, !tbaa !8
.double*8B

	full_text

double* %108
:fadd8B0
.
	full_text!

%314 = fadd double %313, %312
,double8B

	full_text

double %313
,double8B

	full_text

double %312
mcall8Bc
a
	full_textT
R
P%315 = tail call double @llvm.fmuladd.f64(double %290, double %314, double %240)
,double8B

	full_text

double %290
,double8B

	full_text

double %314
,double8B

	full_text

double %240
Pstore8BE
C
	full_text6
4
2store double %315, double* %222, align 8, !tbaa !8
,double8B

	full_text

double %315
.double*8B

	full_text

double* %222
Oload8BE
C
	full_text6
4
2%316 = load double, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
Oload8BE
C
	full_text6
4
2%317 = load double, double* %89, align 8, !tbaa !8
-double*8B

	full_text

double* %89
Bfmul8B8
6
	full_text)
'
%%318 = fmul double %317, 4.000000e+00
,double8B

	full_text

double %317
Cfsub8B9
7
	full_text*
(
&%319 = fsub double -0.000000e+00, %318
,double8B

	full_text

double %318
ucall8Bk
i
	full_text\
Z
X%320 = tail call double @llvm.fmuladd.f64(double %316, double 5.000000e+00, double %319)
,double8B

	full_text

double %316
,double8B

	full_text

double %319
Pload8BF
D
	full_text7
5
3%321 = load double, double* %113, align 8, !tbaa !8
.double*8B

	full_text

double* %113
:fadd8B0
.
	full_text!

%322 = fadd double %321, %320
,double8B

	full_text

double %321
,double8B

	full_text

double %320
mcall8Bc
a
	full_textT
R
P%323 = tail call double @llvm.fmuladd.f64(double %290, double %322, double %255)
,double8B

	full_text

double %290
,double8B

	full_text

double %322
,double8B

	full_text

double %255
Pstore8BE
C
	full_text6
4
2store double %323, double* %241, align 8, !tbaa !8
,double8B

	full_text

double %323
.double*8B

	full_text

double* %241
Pload8BF
D
	full_text7
5
3%324 = load double, double* %94, align 16, !tbaa !8
-double*8B

	full_text

double* %94
Bfmul8B8
6
	full_text)
'
%%325 = fmul double %324, 4.000000e+00
,double8B

	full_text

double %324
Cfsub8B9
7
	full_text*
(
&%326 = fsub double -0.000000e+00, %325
,double8B

	full_text

double %325
ucall8Bk
i
	full_text\
Z
X%327 = tail call double @llvm.fmuladd.f64(double %258, double 5.000000e+00, double %326)
,double8B

	full_text

double %258
,double8B

	full_text

double %326
Qload8BG
E
	full_text8
6
4%328 = load double, double* %118, align 16, !tbaa !8
.double*8B

	full_text

double* %118
:fadd8B0
.
	full_text!

%329 = fadd double %328, %327
,double8B

	full_text

double %328
,double8B

	full_text

double %327
mcall8Bc
a
	full_textT
R
P%330 = tail call double @llvm.fmuladd.f64(double %290, double %329, double %286)
,double8B

	full_text

double %290
,double8B

	full_text

double %329
,double8B

	full_text

double %286
Pstore8BE
C
	full_text6
4
2store double %330, double* %256, align 8, !tbaa !8
,double8B

	full_text

double %330
.double*8B

	full_text

double* %256
Jstore8B?
=
	full_text0
.
,store i64 %48, i64* %147, align 16, !tbaa !8
%i648B

	full_text
	
i64 %48
(i64*8B

	full_text

	i64* %147
Istore8B>
<
	full_text/
-
+store i64 %53, i64* %152, align 8, !tbaa !8
%i648B

	full_text
	
i64 %53
(i64*8B

	full_text

	i64* %152
Jstore8B?
=
	full_text0
.
,store i64 %58, i64* %157, align 16, !tbaa !8
%i648B

	full_text
	
i64 %58
(i64*8B

	full_text

	i64* %157
Istore8B>
<
	full_text/
-
+store i64 %63, i64* %162, align 8, !tbaa !8
%i648B

	full_text
	
i64 %63
(i64*8B

	full_text

	i64* %162
Jstore8B?
=
	full_text0
.
,store i64 %68, i64* %167, align 16, !tbaa !8
%i648B

	full_text
	
i64 %68
(i64*8B

	full_text

	i64* %167
Jstore8B?
=
	full_text0
.
,store i64 %73, i64* %145, align 16, !tbaa !8
%i648B

	full_text
	
i64 %73
(i64*8B

	full_text

	i64* %145
Istore8B>
<
	full_text/
-
+store i64 %78, i64* %149, align 8, !tbaa !8
%i648B

	full_text
	
i64 %78
(i64*8B

	full_text

	i64* %149
Jstore8B?
=
	full_text0
.
,store i64 %83, i64* %154, align 16, !tbaa !8
%i648B

	full_text
	
i64 %83
(i64*8B

	full_text

	i64* %154
Istore8B>
<
	full_text/
-
+store i64 %88, i64* %159, align 8, !tbaa !8
%i648B

	full_text
	
i64 %88
(i64*8B

	full_text

	i64* %159
Jstore8B?
=
	full_text0
.
,store i64 %93, i64* %164, align 16, !tbaa !8
%i648B

	full_text
	
i64 %93
(i64*8B

	full_text

	i64* %164
Istore8B>
<
	full_text/
-
+store i64 %98, i64* %50, align 16, !tbaa !8
%i648B

	full_text
	
i64 %98
'i64*8B

	full_text


i64* %50
Istore8B>
<
	full_text/
-
+store i64 %102, i64* %55, align 8, !tbaa !8
&i648B

	full_text


i64 %102
'i64*8B

	full_text


i64* %55
Jstore8B?
=
	full_text0
.
,store i64 %107, i64* %60, align 16, !tbaa !8
&i648B

	full_text


i64 %107
'i64*8B

	full_text


i64* %60
Istore8B>
<
	full_text/
-
+store i64 %112, i64* %65, align 8, !tbaa !8
&i648B

	full_text


i64 %112
'i64*8B

	full_text


i64* %65
Jstore8B?
=
	full_text0
.
,store i64 %117, i64* %70, align 16, !tbaa !8
&i648B

	full_text


i64 %117
'i64*8B

	full_text


i64* %70
Jstore8B?
=
	full_text0
.
,store i64 %170, i64* %75, align 16, !tbaa !8
&i648B

	full_text


i64 %170
'i64*8B

	full_text


i64* %75
Istore8B>
<
	full_text/
-
+store i64 %173, i64* %80, align 8, !tbaa !8
&i648B

	full_text


i64 %173
'i64*8B

	full_text


i64* %80
Jstore8B?
=
	full_text0
.
,store i64 %176, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %176
'i64*8B

	full_text


i64* %85
Istore8B>
<
	full_text/
-
+store i64 %179, i64* %90, align 8, !tbaa !8
&i648B

	full_text


i64 %179
'i64*8B

	full_text


i64* %90
Jstore8B?
=
	full_text0
.
,store i64 %182, i64* %95, align 16, !tbaa !8
&i648B

	full_text


i64 %182
'i64*8B

	full_text


i64* %95
ögetelementptr8BÜ
É
	full_textv
t
r%331 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 4, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Ibitcast8B<
:
	full_text-
+
)%332 = bitcast [5 x double]* %331 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %331
Jload8B@
>
	full_text1
/
-%333 = load i64, i64* %332, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %332
Jstore8B?
=
	full_text0
.
,store i64 %333, i64* %99, align 16, !tbaa !8
&i648B

	full_text


i64 %333
'i64*8B

	full_text


i64* %99
°getelementptr8Bç
ä
	full_text}
{
y%334 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 4, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%335 = bitcast double* %334 to i64*
.double*8B

	full_text

double* %334
Jload8B@
>
	full_text1
/
-%336 = load i64, i64* %335, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %335
Jstore8B?
=
	full_text0
.
,store i64 %336, i64* %104, align 8, !tbaa !8
&i648B

	full_text


i64 %336
(i64*8B

	full_text

	i64* %104
°getelementptr8Bç
ä
	full_text}
{
y%337 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 4, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%338 = bitcast double* %337 to i64*
.double*8B

	full_text

double* %337
Jload8B@
>
	full_text1
/
-%339 = load i64, i64* %338, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %338
Kstore8B@
>
	full_text1
/
-store i64 %339, i64* %109, align 16, !tbaa !8
&i648B

	full_text


i64 %339
(i64*8B

	full_text

	i64* %109
°getelementptr8Bç
ä
	full_text}
{
y%340 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 4, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%341 = bitcast double* %340 to i64*
.double*8B

	full_text

double* %340
Jload8B@
>
	full_text1
/
-%342 = load i64, i64* %341, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %341
Jstore8B?
=
	full_text0
.
,store i64 %342, i64* %114, align 8, !tbaa !8
&i648B

	full_text


i64 %342
(i64*8B

	full_text

	i64* %114
°getelementptr8Bç
ä
	full_text}
{
y%343 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 4, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%344 = bitcast double* %343 to i64*
.double*8B

	full_text

double* %343
Jload8B@
>
	full_text1
/
-%345 = load i64, i64* %344, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %344
Kstore8B@
>
	full_text1
/
-store i64 %345, i64* %119, align 16, !tbaa !8
&i648B

	full_text


i64 %345
(i64*8B

	full_text

	i64* %119
ågetelementptr8By
w
	full_textj
h
f%346 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 3, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%347 = load double, double* %346, align 8, !tbaa !8
.double*8B

	full_text

double* %346
ågetelementptr8By
w
	full_textj
h
f%348 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 3, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%349 = load double, double* %348, align 8, !tbaa !8
.double*8B

	full_text

double* %348
ågetelementptr8By
w
	full_textj
h
f%350 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 3, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%351 = load double, double* %350, align 8, !tbaa !8
.double*8B

	full_text

double* %350
ågetelementptr8By
w
	full_textj
h
f%352 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 3, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%353 = load double, double* %352, align 8, !tbaa !8
.double*8B

	full_text

double* %352
ågetelementptr8By
w
	full_textj
h
f%354 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 3, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%355 = load double, double* %354, align 8, !tbaa !8
.double*8B

	full_text

double* %354
ågetelementptr8By
w
	full_textj
h
f%356 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 3, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%357 = load double, double* %356, align 8, !tbaa !8
.double*8B

	full_text

double* %356
°getelementptr8Bç
ä
	full_text}
{
y%358 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 2, i64 %45, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%359 = load double, double* %358, align 8, !tbaa !8
.double*8B

	full_text

double* %358
Abitcast8B4
2
	full_text%
#
!%360 = bitcast i64 %170 to double
&i648B

	full_text


i64 %170
vcall8Bl
j
	full_text]
[
Y%361 = tail call double @llvm.fmuladd.f64(double %197, double -2.000000e+00, double %360)
,double8B

	full_text

double %197
,double8B

	full_text

double %360
:fadd8B0
.
	full_text!

%362 = fadd double %361, %198
,double8B

	full_text

double %361
,double8B

	full_text

double %198
{call8Bq
o
	full_textb
`
^%363 = tail call double @llvm.fmuladd.f64(double %362, double 0x40A7418000000001, double %359)
,double8B

	full_text

double %362
,double8B

	full_text

double %359
Abitcast8B4
2
	full_text%
#
!%364 = bitcast i64 %176 to double
&i648B

	full_text


i64 %176
:fsub8B0
.
	full_text!

%365 = fsub double %364, %224
,double8B

	full_text

double %364
,double8B

	full_text

double %224
vcall8Bl
j
	full_text]
[
Y%366 = tail call double @llvm.fmuladd.f64(double %365, double -3.150000e+01, double %363)
,double8B

	full_text

double %365
,double8B

	full_text

double %363
°getelementptr8Bç
ä
	full_text}
{
y%367 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 2, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%368 = load double, double* %367, align 8, !tbaa !8
.double*8B

	full_text

double* %367
Abitcast8B4
2
	full_text%
#
!%369 = bitcast i64 %173 to double
&i648B

	full_text


i64 %173
vcall8Bl
j
	full_text]
[
Y%370 = tail call double @llvm.fmuladd.f64(double %209, double -2.000000e+00, double %369)
,double8B

	full_text

double %209
,double8B

	full_text

double %369
:fadd8B0
.
	full_text!

%371 = fadd double %370, %210
,double8B

	full_text

double %370
,double8B

	full_text

double %210
{call8Bq
o
	full_textb
`
^%372 = tail call double @llvm.fmuladd.f64(double %371, double 0x40A7418000000001, double %368)
,double8B

	full_text

double %371
,double8B

	full_text

double %368
vcall8Bl
j
	full_text]
[
Y%373 = tail call double @llvm.fmuladd.f64(double %184, double -2.000000e+00, double %347)
,double8B

	full_text

double %184
,double8B

	full_text

double %347
:fadd8B0
.
	full_text!

%374 = fadd double %123, %373
,double8B

	full_text

double %123
,double8B

	full_text

double %373
{call8Bq
o
	full_textb
`
^%375 = tail call double @llvm.fmuladd.f64(double %374, double 0x4078CE6666666667, double %372)
,double8B

	full_text

double %374
,double8B

	full_text

double %372
:fmul8B0
.
	full_text!

%376 = fmul double %127, %210
,double8B

	full_text

double %127
,double8B

	full_text

double %210
Cfsub8B9
7
	full_text*
(
&%377 = fsub double -0.000000e+00, %376
,double8B

	full_text

double %376
mcall8Bc
a
	full_textT
R
P%378 = tail call double @llvm.fmuladd.f64(double %369, double %349, double %377)
,double8B

	full_text

double %369
,double8B

	full_text

double %349
,double8B

	full_text

double %377
vcall8Bl
j
	full_text]
[
Y%379 = tail call double @llvm.fmuladd.f64(double %378, double -3.150000e+01, double %375)
,double8B

	full_text

double %378
,double8B

	full_text

double %375
°getelementptr8Bç
ä
	full_text}
{
y%380 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 2, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%381 = load double, double* %380, align 8, !tbaa !8
.double*8B

	full_text

double* %380
vcall8Bl
j
	full_text]
[
Y%382 = tail call double @llvm.fmuladd.f64(double %203, double -2.000000e+00, double %364)
,double8B

	full_text

double %203
,double8B

	full_text

double %364
:fadd8B0
.
	full_text!

%383 = fadd double %382, %224
,double8B

	full_text

double %382
,double8B

	full_text

double %224
{call8Bq
o
	full_textb
`
^%384 = tail call double @llvm.fmuladd.f64(double %383, double 0x40A7418000000001, double %381)
,double8B

	full_text

double %383
,double8B

	full_text

double %381
vcall8Bl
j
	full_text]
[
Y%385 = tail call double @llvm.fmuladd.f64(double %186, double -2.000000e+00, double %349)
,double8B

	full_text

double %186
,double8B

	full_text

double %349
:fadd8B0
.
	full_text!

%386 = fadd double %127, %385
,double8B

	full_text

double %127
,double8B

	full_text

double %385
ucall8Bk
i
	full_text\
Z
X%387 = tail call double @llvm.fmuladd.f64(double %386, double 5.292000e+02, double %384)
,double8B

	full_text

double %386
,double8B

	full_text

double %384
:fmul8B0
.
	full_text!

%388 = fmul double %127, %224
,double8B

	full_text

double %127
,double8B

	full_text

double %224
Cfsub8B9
7
	full_text*
(
&%389 = fsub double -0.000000e+00, %388
,double8B

	full_text

double %388
mcall8Bc
a
	full_textT
R
P%390 = tail call double @llvm.fmuladd.f64(double %364, double %349, double %389)
,double8B

	full_text

double %364
,double8B

	full_text

double %349
,double8B

	full_text

double %389
Abitcast8B4
2
	full_text%
#
!%391 = bitcast i64 %182 to double
&i648B

	full_text


i64 %182
:fsub8B0
.
	full_text!

%392 = fsub double %391, %357
,double8B

	full_text

double %391
,double8B

	full_text

double %357
@bitcast8B3
1
	full_text$
"
 %393 = bitcast i64 %93 to double
%i648B

	full_text
	
i64 %93
:fsub8B0
.
	full_text!

%394 = fsub double %392, %393
,double8B

	full_text

double %392
,double8B

	full_text

double %393
:fadd8B0
.
	full_text!

%395 = fadd double %143, %394
,double8B

	full_text

double %143
,double8B

	full_text

double %394
ucall8Bk
i
	full_text\
Z
X%396 = tail call double @llvm.fmuladd.f64(double %395, double 4.000000e-01, double %390)
,double8B

	full_text

double %395
,double8B

	full_text

double %390
vcall8Bl
j
	full_text]
[
Y%397 = tail call double @llvm.fmuladd.f64(double %396, double -3.150000e+01, double %387)
,double8B

	full_text

double %396
,double8B

	full_text

double %387
°getelementptr8Bç
ä
	full_text}
{
y%398 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 2, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%399 = load double, double* %398, align 8, !tbaa !8
.double*8B

	full_text

double* %398
Abitcast8B4
2
	full_text%
#
!%400 = bitcast i64 %179 to double
&i648B

	full_text


i64 %179
vcall8Bl
j
	full_text]
[
Y%401 = tail call double @llvm.fmuladd.f64(double %243, double -2.000000e+00, double %400)
,double8B

	full_text

double %243
,double8B

	full_text

double %400
:fadd8B0
.
	full_text!

%402 = fadd double %401, %244
,double8B

	full_text

double %401
,double8B

	full_text

double %244
{call8Bq
o
	full_textb
`
^%403 = tail call double @llvm.fmuladd.f64(double %402, double 0x40A7418000000001, double %399)
,double8B

	full_text

double %402
,double8B

	full_text

double %399
vcall8Bl
j
	full_text]
[
Y%404 = tail call double @llvm.fmuladd.f64(double %188, double -2.000000e+00, double %351)
,double8B

	full_text

double %188
,double8B

	full_text

double %351
:fadd8B0
.
	full_text!

%405 = fadd double %131, %404
,double8B

	full_text

double %131
,double8B

	full_text

double %404
{call8Bq
o
	full_textb
`
^%406 = tail call double @llvm.fmuladd.f64(double %405, double 0x4078CE6666666667, double %403)
,double8B

	full_text

double %405
,double8B

	full_text

double %403
:fmul8B0
.
	full_text!

%407 = fmul double %127, %244
,double8B

	full_text

double %127
,double8B

	full_text

double %244
Cfsub8B9
7
	full_text*
(
&%408 = fsub double -0.000000e+00, %407
,double8B

	full_text

double %407
mcall8Bc
a
	full_textT
R
P%409 = tail call double @llvm.fmuladd.f64(double %400, double %349, double %408)
,double8B

	full_text

double %400
,double8B

	full_text

double %349
,double8B

	full_text

double %408
vcall8Bl
j
	full_text]
[
Y%410 = tail call double @llvm.fmuladd.f64(double %409, double -3.150000e+01, double %406)
,double8B

	full_text

double %409
,double8B

	full_text

double %406
°getelementptr8Bç
ä
	full_text}
{
y%411 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 2, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%412 = load double, double* %411, align 8, !tbaa !8
.double*8B

	full_text

double* %411
vcall8Bl
j
	full_text]
[
Y%413 = tail call double @llvm.fmuladd.f64(double %234, double -2.000000e+00, double %391)
,double8B

	full_text

double %234
,double8B

	full_text

double %391
:fadd8B0
.
	full_text!

%414 = fadd double %413, %393
,double8B

	full_text

double %413
,double8B

	full_text

double %393
{call8Bq
o
	full_textb
`
^%415 = tail call double @llvm.fmuladd.f64(double %414, double 0x40A7418000000001, double %412)
,double8B

	full_text

double %414
,double8B

	full_text

double %412
vcall8Bl
j
	full_text]
[
Y%416 = tail call double @llvm.fmuladd.f64(double %190, double -2.000000e+00, double %353)
,double8B

	full_text

double %190
,double8B

	full_text

double %353
:fadd8B0
.
	full_text!

%417 = fadd double %135, %416
,double8B

	full_text

double %135
,double8B

	full_text

double %416
{call8Bq
o
	full_textb
`
^%418 = tail call double @llvm.fmuladd.f64(double %417, double 0xC077D0624DD2F1A9, double %415)
,double8B

	full_text

double %417
,double8B

	full_text

double %415
Bfmul8B8
6
	full_text)
'
%%419 = fmul double %186, 2.000000e+00
,double8B

	full_text

double %186
:fmul8B0
.
	full_text!

%420 = fmul double %186, %419
,double8B

	full_text

double %186
,double8B

	full_text

double %419
Cfsub8B9
7
	full_text*
(
&%421 = fsub double -0.000000e+00, %420
,double8B

	full_text

double %420
mcall8Bc
a
	full_textT
R
P%422 = tail call double @llvm.fmuladd.f64(double %349, double %349, double %421)
,double8B

	full_text

double %349
,double8B

	full_text

double %349
,double8B

	full_text

double %421
mcall8Bc
a
	full_textT
R
P%423 = tail call double @llvm.fmuladd.f64(double %127, double %127, double %422)
,double8B

	full_text

double %127
,double8B

	full_text

double %127
,double8B

	full_text

double %422
ucall8Bk
i
	full_text\
Z
X%424 = tail call double @llvm.fmuladd.f64(double %423, double 6.615000e+01, double %418)
,double8B

	full_text

double %423
,double8B

	full_text

double %418
Bfmul8B8
6
	full_text)
'
%%425 = fmul double %234, 2.000000e+00
,double8B

	full_text

double %234
:fmul8B0
.
	full_text!

%426 = fmul double %192, %425
,double8B

	full_text

double %192
,double8B

	full_text

double %425
Cfsub8B9
7
	full_text*
(
&%427 = fsub double -0.000000e+00, %426
,double8B

	full_text

double %426
mcall8Bc
a
	full_textT
R
P%428 = tail call double @llvm.fmuladd.f64(double %391, double %355, double %427)
,double8B

	full_text

double %391
,double8B

	full_text

double %355
,double8B

	full_text

double %427
mcall8Bc
a
	full_textT
R
P%429 = tail call double @llvm.fmuladd.f64(double %393, double %139, double %428)
,double8B

	full_text

double %393
,double8B

	full_text

double %139
,double8B

	full_text

double %428
{call8Bq
o
	full_textb
`
^%430 = tail call double @llvm.fmuladd.f64(double %429, double 0x40884F645A1CAC08, double %424)
,double8B

	full_text

double %429
,double8B

	full_text

double %424
Bfmul8B8
6
	full_text)
'
%%431 = fmul double %357, 4.000000e-01
,double8B

	full_text

double %357
Cfsub8B9
7
	full_text*
(
&%432 = fsub double -0.000000e+00, %431
,double8B

	full_text

double %431
ucall8Bk
i
	full_text\
Z
X%433 = tail call double @llvm.fmuladd.f64(double %391, double 1.400000e+00, double %432)
,double8B

	full_text

double %391
,double8B

	full_text

double %432
Bfmul8B8
6
	full_text)
'
%%434 = fmul double %143, 4.000000e-01
,double8B

	full_text

double %143
Cfsub8B9
7
	full_text*
(
&%435 = fsub double -0.000000e+00, %434
,double8B

	full_text

double %434
ucall8Bk
i
	full_text\
Z
X%436 = tail call double @llvm.fmuladd.f64(double %393, double 1.400000e+00, double %435)
,double8B

	full_text

double %393
,double8B

	full_text

double %435
:fmul8B0
.
	full_text!

%437 = fmul double %127, %436
,double8B

	full_text

double %127
,double8B

	full_text

double %436
Cfsub8B9
7
	full_text*
(
&%438 = fsub double -0.000000e+00, %437
,double8B

	full_text

double %437
mcall8Bc
a
	full_textT
R
P%439 = tail call double @llvm.fmuladd.f64(double %433, double %349, double %438)
,double8B

	full_text

double %433
,double8B

	full_text

double %349
,double8B

	full_text

double %438
vcall8Bl
j
	full_text]
[
Y%440 = tail call double @llvm.fmuladd.f64(double %439, double -3.150000e+01, double %430)
,double8B

	full_text

double %439
,double8B

	full_text

double %430
Qload8BG
E
	full_text8
6
4%441 = load double, double* %144, align 16, !tbaa !8
.double*8B

	full_text

double* %144
Pload8BF
D
	full_text7
5
3%442 = load double, double* %49, align 16, !tbaa !8
-double*8B

	full_text

double* %49
Bfmul8B8
6
	full_text)
'
%%443 = fmul double %442, 6.000000e+00
,double8B

	full_text

double %442
vcall8Bl
j
	full_text]
[
Y%444 = tail call double @llvm.fmuladd.f64(double %441, double -4.000000e+00, double %443)
,double8B

	full_text

double %441
,double8B

	full_text

double %443
Pload8BF
D
	full_text7
5
3%445 = load double, double* %74, align 16, !tbaa !8
-double*8B

	full_text

double* %74
vcall8Bl
j
	full_text]
[
Y%446 = tail call double @llvm.fmuladd.f64(double %445, double -4.000000e+00, double %444)
,double8B

	full_text

double %445
,double8B

	full_text

double %444
Qload8BG
E
	full_text8
6
4%447 = load double, double* %296, align 16, !tbaa !8
.double*8B

	full_text

double* %296
:fadd8B0
.
	full_text!

%448 = fadd double %447, %446
,double8B

	full_text

double %447
,double8B

	full_text

double %446
mcall8Bc
a
	full_textT
R
P%449 = tail call double @llvm.fmuladd.f64(double %290, double %448, double %366)
,double8B

	full_text

double %290
,double8B

	full_text

double %448
,double8B

	full_text

double %366
Pstore8BE
C
	full_text6
4
2store double %449, double* %358, align 8, !tbaa !8
,double8B

	full_text

double %449
.double*8B

	full_text

double* %358
Pload8BF
D
	full_text7
5
3%450 = load double, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
Oload8BE
C
	full_text6
4
2%451 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
Bfmul8B8
6
	full_text)
'
%%452 = fmul double %451, 6.000000e+00
,double8B

	full_text

double %451
vcall8Bl
j
	full_text]
[
Y%453 = tail call double @llvm.fmuladd.f64(double %450, double -4.000000e+00, double %452)
,double8B

	full_text

double %450
,double8B

	full_text

double %452
Oload8BE
C
	full_text6
4
2%454 = load double, double* %79, align 8, !tbaa !8
-double*8B

	full_text

double* %79
vcall8Bl
j
	full_text]
[
Y%455 = tail call double @llvm.fmuladd.f64(double %454, double -4.000000e+00, double %453)
,double8B

	full_text

double %454
,double8B

	full_text

double %453
Pload8BF
D
	full_text7
5
3%456 = load double, double* %103, align 8, !tbaa !8
.double*8B

	full_text

double* %103
:fadd8B0
.
	full_text!

%457 = fadd double %456, %455
,double8B

	full_text

double %456
,double8B

	full_text

double %455
mcall8Bc
a
	full_textT
R
P%458 = tail call double @llvm.fmuladd.f64(double %290, double %457, double %379)
,double8B

	full_text

double %290
,double8B

	full_text

double %457
,double8B

	full_text

double %379
Pstore8BE
C
	full_text6
4
2store double %458, double* %367, align 8, !tbaa !8
,double8B

	full_text

double %458
.double*8B

	full_text

double* %367
Qload8BG
E
	full_text8
6
4%459 = load double, double* %153, align 16, !tbaa !8
.double*8B

	full_text

double* %153
Pload8BF
D
	full_text7
5
3%460 = load double, double* %59, align 16, !tbaa !8
-double*8B

	full_text

double* %59
Bfmul8B8
6
	full_text)
'
%%461 = fmul double %460, 6.000000e+00
,double8B

	full_text

double %460
vcall8Bl
j
	full_text]
[
Y%462 = tail call double @llvm.fmuladd.f64(double %459, double -4.000000e+00, double %461)
,double8B

	full_text

double %459
,double8B

	full_text

double %461
Pload8BF
D
	full_text7
5
3%463 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%464 = tail call double @llvm.fmuladd.f64(double %463, double -4.000000e+00, double %462)
,double8B

	full_text

double %463
,double8B

	full_text

double %462
Qload8BG
E
	full_text8
6
4%465 = load double, double* %108, align 16, !tbaa !8
.double*8B

	full_text

double* %108
:fadd8B0
.
	full_text!

%466 = fadd double %465, %464
,double8B

	full_text

double %465
,double8B

	full_text

double %464
mcall8Bc
a
	full_textT
R
P%467 = tail call double @llvm.fmuladd.f64(double %290, double %466, double %397)
,double8B

	full_text

double %290
,double8B

	full_text

double %466
,double8B

	full_text

double %397
Pstore8BE
C
	full_text6
4
2store double %467, double* %380, align 8, !tbaa !8
,double8B

	full_text

double %467
.double*8B

	full_text

double* %380
Pload8BF
D
	full_text7
5
3%468 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
Oload8BE
C
	full_text6
4
2%469 = load double, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
Bfmul8B8
6
	full_text)
'
%%470 = fmul double %469, 6.000000e+00
,double8B

	full_text

double %469
vcall8Bl
j
	full_text]
[
Y%471 = tail call double @llvm.fmuladd.f64(double %468, double -4.000000e+00, double %470)
,double8B

	full_text

double %468
,double8B

	full_text

double %470
Oload8BE
C
	full_text6
4
2%472 = load double, double* %89, align 8, !tbaa !8
-double*8B

	full_text

double* %89
vcall8Bl
j
	full_text]
[
Y%473 = tail call double @llvm.fmuladd.f64(double %472, double -4.000000e+00, double %471)
,double8B

	full_text

double %472
,double8B

	full_text

double %471
Pload8BF
D
	full_text7
5
3%474 = load double, double* %113, align 8, !tbaa !8
.double*8B

	full_text

double* %113
:fadd8B0
.
	full_text!

%475 = fadd double %474, %473
,double8B

	full_text

double %474
,double8B

	full_text

double %473
mcall8Bc
a
	full_textT
R
P%476 = tail call double @llvm.fmuladd.f64(double %290, double %475, double %410)
,double8B

	full_text

double %290
,double8B

	full_text

double %475
,double8B

	full_text

double %410
Pstore8BE
C
	full_text6
4
2store double %476, double* %398, align 8, !tbaa !8
,double8B

	full_text

double %476
.double*8B

	full_text

double* %398
Qload8BG
E
	full_text8
6
4%477 = load double, double* %163, align 16, !tbaa !8
.double*8B

	full_text

double* %163
Bfmul8B8
6
	full_text)
'
%%478 = fmul double %234, 6.000000e+00
,double8B

	full_text

double %234
vcall8Bl
j
	full_text]
[
Y%479 = tail call double @llvm.fmuladd.f64(double %477, double -4.000000e+00, double %478)
,double8B

	full_text

double %477
,double8B

	full_text

double %478
Pload8BF
D
	full_text7
5
3%480 = load double, double* %94, align 16, !tbaa !8
-double*8B

	full_text

double* %94
vcall8Bl
j
	full_text]
[
Y%481 = tail call double @llvm.fmuladd.f64(double %480, double -4.000000e+00, double %479)
,double8B

	full_text

double %480
,double8B

	full_text

double %479
Qload8BG
E
	full_text8
6
4%482 = load double, double* %118, align 16, !tbaa !8
.double*8B

	full_text

double* %118
:fadd8B0
.
	full_text!

%483 = fadd double %482, %481
,double8B

	full_text

double %482
,double8B

	full_text

double %481
mcall8Bc
a
	full_textT
R
P%484 = tail call double @llvm.fmuladd.f64(double %290, double %483, double %440)
,double8B

	full_text

double %290
,double8B

	full_text

double %483
,double8B

	full_text

double %440
Pstore8BE
C
	full_text6
4
2store double %484, double* %411, align 8, !tbaa !8
,double8B

	full_text

double %484
.double*8B

	full_text

double* %411
6icmp8B,
*
	full_text

%485 = icmp slt i32 %9, 7
Abitcast8B4
2
	full_text%
#
!%486 = bitcast double %441 to i64
,double8B

	full_text

double %441
Abitcast8B4
2
	full_text%
#
!%487 = bitcast double %450 to i64
,double8B

	full_text

double %450
Abitcast8B4
2
	full_text%
#
!%488 = bitcast double %459 to i64
,double8B

	full_text

double %459
Abitcast8B4
2
	full_text%
#
!%489 = bitcast double %468 to i64
,double8B

	full_text

double %468
Abitcast8B4
2
	full_text%
#
!%490 = bitcast double %477 to i64
,double8B

	full_text

double %477
Abitcast8B4
2
	full_text%
#
!%491 = bitcast double %469 to i64
,double8B

	full_text

double %469
Abitcast8B4
2
	full_text%
#
!%492 = bitcast double %480 to i64
,double8B

	full_text

double %480
1add8B(
&
	full_text

%493 = add i32 %9, -3
=br8B5
3
	full_text&
$
"br i1 %485, label %494, label %496
$i18B

	full_text
	
i1 %485
qgetelementptr8B^
\
	full_textO
M
K%495 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
(br8B 

	full_text

br label %691
qgetelementptr8B^
\
	full_textO
M
K%497 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
qgetelementptr8B^
\
	full_textO
M
K%498 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
qgetelementptr8B^
\
	full_textO
M
K%499 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
qgetelementptr8B^
\
	full_textO
M
K%500 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
8zext8B.
,
	full_text

%501 = zext i32 %493 to i64
&i328B

	full_text


i32 %493
(br8B 

	full_text

br label %502
Lphi8BC
A
	full_text4
2
0%503 = phi double [ %678, %502 ], [ %482, %496 ]
,double8B

	full_text

double %678
,double8B

	full_text

double %482
Lphi8BC
A
	full_text4
2
0%504 = phi double [ %670, %502 ], [ %474, %496 ]
,double8B

	full_text

double %670
,double8B

	full_text

double %474
Lphi8BC
A
	full_text4
2
0%505 = phi double [ %663, %502 ], [ %465, %496 ]
,double8B

	full_text

double %663
,double8B

	full_text

double %465
Lphi8BC
A
	full_text4
2
0%506 = phi double [ %656, %502 ], [ %456, %496 ]
,double8B

	full_text

double %656
,double8B

	full_text

double %456
Lphi8BC
A
	full_text4
2
0%507 = phi double [ %649, %502 ], [ %447, %496 ]
,double8B

	full_text

double %649
,double8B

	full_text

double %447
Iphi8B@
>
	full_text1
/
-%508 = phi i64 [ %689, %502 ], [ %492, %496 ]
&i648B

	full_text


i64 %689
&i648B

	full_text


i64 %492
Lphi8BC
A
	full_text4
2
0%509 = phi double [ %504, %502 ], [ %472, %496 ]
,double8B

	full_text

double %504
,double8B

	full_text

double %472
Lphi8BC
A
	full_text4
2
0%510 = phi double [ %505, %502 ], [ %463, %496 ]
,double8B

	full_text

double %505
,double8B

	full_text

double %463
Lphi8BC
A
	full_text4
2
0%511 = phi double [ %506, %502 ], [ %454, %496 ]
,double8B

	full_text

double %506
,double8B

	full_text

double %454
Lphi8BC
A
	full_text4
2
0%512 = phi double [ %507, %502 ], [ %445, %496 ]
,double8B

	full_text

double %507
,double8B

	full_text

double %445
Iphi8B@
>
	full_text1
/
-%513 = phi i64 [ %688, %502 ], [ %117, %496 ]
&i648B

	full_text


i64 %688
&i648B

	full_text


i64 %117
Iphi8B@
>
	full_text1
/
-%514 = phi i64 [ %687, %502 ], [ %491, %496 ]
&i648B

	full_text


i64 %687
&i648B

	full_text


i64 %491
Lphi8BC
A
	full_text4
2
0%515 = phi double [ %510, %502 ], [ %460, %496 ]
,double8B

	full_text

double %510
,double8B

	full_text

double %460
Lphi8BC
A
	full_text4
2
0%516 = phi double [ %511, %502 ], [ %451, %496 ]
,double8B

	full_text

double %511
,double8B

	full_text

double %451
Lphi8BC
A
	full_text4
2
0%517 = phi double [ %512, %502 ], [ %442, %496 ]
,double8B

	full_text

double %512
,double8B

	full_text

double %442
Iphi8B@
>
	full_text1
/
-%518 = phi i64 [ %686, %502 ], [ %490, %496 ]
&i648B

	full_text


i64 %686
&i648B

	full_text


i64 %490
Iphi8B@
>
	full_text1
/
-%519 = phi i64 [ %685, %502 ], [ %489, %496 ]
&i648B

	full_text


i64 %685
&i648B

	full_text


i64 %489
Iphi8B@
>
	full_text1
/
-%520 = phi i64 [ %684, %502 ], [ %488, %496 ]
&i648B

	full_text


i64 %684
&i648B

	full_text


i64 %488
Iphi8B@
>
	full_text1
/
-%521 = phi i64 [ %683, %502 ], [ %487, %496 ]
&i648B

	full_text


i64 %683
&i648B

	full_text


i64 %487
Iphi8B@
>
	full_text1
/
-%522 = phi i64 [ %682, %502 ], [ %486, %496 ]
&i648B

	full_text


i64 %682
&i648B

	full_text


i64 %486
Fphi8B=
;
	full_text.
,
*%523 = phi i64 [ %552, %502 ], [ 3, %496 ]
&i648B

	full_text


i64 %552
Lphi8BC
A
	full_text4
2
0%524 = phi double [ %525, %502 ], [ %184, %496 ]
,double8B

	full_text

double %525
,double8B

	full_text

double %184
Lphi8BC
A
	full_text4
2
0%525 = phi double [ %554, %502 ], [ %347, %496 ]
,double8B

	full_text

double %554
,double8B

	full_text

double %347
Lphi8BC
A
	full_text4
2
0%526 = phi double [ %527, %502 ], [ %186, %496 ]
,double8B

	full_text

double %527
,double8B

	full_text

double %186
Lphi8BC
A
	full_text4
2
0%527 = phi double [ %556, %502 ], [ %349, %496 ]
,double8B

	full_text

double %556
,double8B

	full_text

double %349
Lphi8BC
A
	full_text4
2
0%528 = phi double [ %564, %502 ], [ %357, %496 ]
,double8B

	full_text

double %564
,double8B

	full_text

double %357
Lphi8BC
A
	full_text4
2
0%529 = phi double [ %528, %502 ], [ %194, %496 ]
,double8B

	full_text

double %528
,double8B

	full_text

double %194
Lphi8BC
A
	full_text4
2
0%530 = phi double [ %562, %502 ], [ %355, %496 ]
,double8B

	full_text

double %562
,double8B

	full_text

double %355
Lphi8BC
A
	full_text4
2
0%531 = phi double [ %530, %502 ], [ %192, %496 ]
,double8B

	full_text

double %530
,double8B

	full_text

double %192
Lphi8BC
A
	full_text4
2
0%532 = phi double [ %560, %502 ], [ %353, %496 ]
,double8B

	full_text

double %560
,double8B

	full_text

double %353
Lphi8BC
A
	full_text4
2
0%533 = phi double [ %532, %502 ], [ %190, %496 ]
,double8B

	full_text

double %532
,double8B

	full_text

double %190
Lphi8BC
A
	full_text4
2
0%534 = phi double [ %558, %502 ], [ %351, %496 ]
,double8B

	full_text

double %558
,double8B

	full_text

double %351
Lphi8BC
A
	full_text4
2
0%535 = phi double [ %534, %502 ], [ %188, %496 ]
,double8B

	full_text

double %534
,double8B

	full_text

double %188
Kstore8B@
>
	full_text1
/
-store i64 %522, i64* %147, align 16, !tbaa !8
&i648B

	full_text


i64 %522
(i64*8B

	full_text

	i64* %147
Jstore8B?
=
	full_text0
.
,store i64 %521, i64* %152, align 8, !tbaa !8
&i648B

	full_text


i64 %521
(i64*8B

	full_text

	i64* %152
Kstore8B@
>
	full_text1
/
-store i64 %520, i64* %157, align 16, !tbaa !8
&i648B

	full_text


i64 %520
(i64*8B

	full_text

	i64* %157
Jstore8B?
=
	full_text0
.
,store i64 %519, i64* %162, align 8, !tbaa !8
&i648B

	full_text


i64 %519
(i64*8B

	full_text

	i64* %162
Kstore8B@
>
	full_text1
/
-store i64 %518, i64* %167, align 16, !tbaa !8
&i648B

	full_text


i64 %518
(i64*8B

	full_text

	i64* %167
Jstore8B?
=
	full_text0
.
,store i64 %514, i64* %159, align 8, !tbaa !8
&i648B

	full_text


i64 %514
(i64*8B

	full_text

	i64* %159
Kstore8B@
>
	full_text1
/
-store i64 %513, i64* %164, align 16, !tbaa !8
&i648B

	full_text


i64 %513
(i64*8B

	full_text

	i64* %164
Jstore8B?
=
	full_text0
.
,store i64 %508, i64* %70, align 16, !tbaa !8
&i648B

	full_text


i64 %508
'i64*8B

	full_text


i64* %70
:add8B1
/
	full_text"
 
%536 = add nuw nsw i64 %523, 2
&i648B

	full_text


i64 %523
ùgetelementptr8Bâ
Ü
	full_texty
w
u%537 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %536, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %45
Ibitcast8B<
:
	full_text-
+
)%538 = bitcast [5 x double]* %537 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %537
Jload8B@
>
	full_text1
/
-%539 = load i64, i64* %538, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %538
Jstore8B?
=
	full_text0
.
,store i64 %539, i64* %99, align 16, !tbaa !8
&i648B

	full_text


i64 %539
'i64*8B

	full_text


i64* %99
•getelementptr8Bë
é
	full_textÄ
~
|%540 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %536, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%541 = bitcast double* %540 to i64*
.double*8B

	full_text

double* %540
Jload8B@
>
	full_text1
/
-%542 = load i64, i64* %541, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %541
Jstore8B?
=
	full_text0
.
,store i64 %542, i64* %104, align 8, !tbaa !8
&i648B

	full_text


i64 %542
(i64*8B

	full_text

	i64* %104
•getelementptr8Bë
é
	full_textÄ
~
|%543 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %536, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%544 = bitcast double* %543 to i64*
.double*8B

	full_text

double* %543
Jload8B@
>
	full_text1
/
-%545 = load i64, i64* %544, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %544
Kstore8B@
>
	full_text1
/
-store i64 %545, i64* %109, align 16, !tbaa !8
&i648B

	full_text


i64 %545
(i64*8B

	full_text

	i64* %109
•getelementptr8Bë
é
	full_textÄ
~
|%546 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %536, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%547 = bitcast double* %546 to i64*
.double*8B

	full_text

double* %546
Jload8B@
>
	full_text1
/
-%548 = load i64, i64* %547, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %547
Jstore8B?
=
	full_text0
.
,store i64 %548, i64* %114, align 8, !tbaa !8
&i648B

	full_text


i64 %548
(i64*8B

	full_text

	i64* %114
•getelementptr8Bë
é
	full_textÄ
~
|%549 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %536, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %536
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%550 = bitcast double* %549 to i64*
.double*8B

	full_text

double* %549
Jload8B@
>
	full_text1
/
-%551 = load i64, i64* %550, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %550
Kstore8B@
>
	full_text1
/
-store i64 %551, i64* %119, align 16, !tbaa !8
&i648B

	full_text


i64 %551
(i64*8B

	full_text

	i64* %119
:add8B1
/
	full_text"
 
%552 = add nuw nsw i64 %523, 1
&i648B

	full_text


i64 %523
ègetelementptr8B|
z
	full_textm
k
i%553 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 %552, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %552
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%554 = load double, double* %553, align 8, !tbaa !8
.double*8B

	full_text

double* %553
ègetelementptr8B|
z
	full_textm
k
i%555 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 %552, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %552
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%556 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
ègetelementptr8B|
z
	full_textm
k
i%557 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 %552, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %552
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%558 = load double, double* %557, align 8, !tbaa !8
.double*8B

	full_text

double* %557
ègetelementptr8B|
z
	full_textm
k
i%559 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 %552, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %552
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%560 = load double, double* %559, align 8, !tbaa !8
.double*8B

	full_text

double* %559
ègetelementptr8B|
z
	full_textm
k
i%561 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 %552, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %552
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%562 = load double, double* %561, align 8, !tbaa !8
.double*8B

	full_text

double* %561
ègetelementptr8B|
z
	full_textm
k
i%563 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 %552, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %552
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%564 = load double, double* %563, align 8, !tbaa !8
.double*8B

	full_text

double* %563
•getelementptr8Bë
é
	full_textÄ
~
|%565 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %523, i64 %45, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %523
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%566 = load double, double* %565, align 8, !tbaa !8
.double*8B

	full_text

double* %565
vcall8Bl
j
	full_text]
[
Y%567 = tail call double @llvm.fmuladd.f64(double %512, double -2.000000e+00, double %507)
,double8B

	full_text

double %512
,double8B

	full_text

double %507
:fadd8B0
.
	full_text!

%568 = fadd double %567, %517
,double8B

	full_text

double %567
,double8B

	full_text

double %517
{call8Bq
o
	full_textb
`
^%569 = tail call double @llvm.fmuladd.f64(double %568, double 0x40A7418000000001, double %566)
,double8B

	full_text

double %568
,double8B

	full_text

double %566
:fsub8B0
.
	full_text!

%570 = fsub double %505, %515
,double8B

	full_text

double %505
,double8B

	full_text

double %515
vcall8Bl
j
	full_text]
[
Y%571 = tail call double @llvm.fmuladd.f64(double %570, double -3.150000e+01, double %569)
,double8B

	full_text

double %570
,double8B

	full_text

double %569
•getelementptr8Bë
é
	full_textÄ
~
|%572 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %523, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %523
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%573 = load double, double* %572, align 8, !tbaa !8
.double*8B

	full_text

double* %572
vcall8Bl
j
	full_text]
[
Y%574 = tail call double @llvm.fmuladd.f64(double %511, double -2.000000e+00, double %506)
,double8B

	full_text

double %511
,double8B

	full_text

double %506
:fadd8B0
.
	full_text!

%575 = fadd double %574, %516
,double8B

	full_text

double %574
,double8B

	full_text

double %516
{call8Bq
o
	full_textb
`
^%576 = tail call double @llvm.fmuladd.f64(double %575, double 0x40A7418000000001, double %573)
,double8B

	full_text

double %575
,double8B

	full_text

double %573
vcall8Bl
j
	full_text]
[
Y%577 = tail call double @llvm.fmuladd.f64(double %525, double -2.000000e+00, double %554)
,double8B

	full_text

double %525
,double8B

	full_text

double %554
:fadd8B0
.
	full_text!

%578 = fadd double %524, %577
,double8B

	full_text

double %524
,double8B

	full_text

double %577
{call8Bq
o
	full_textb
`
^%579 = tail call double @llvm.fmuladd.f64(double %578, double 0x4078CE6666666667, double %576)
,double8B

	full_text

double %578
,double8B

	full_text

double %576
:fmul8B0
.
	full_text!

%580 = fmul double %526, %516
,double8B

	full_text

double %526
,double8B

	full_text

double %516
Cfsub8B9
7
	full_text*
(
&%581 = fsub double -0.000000e+00, %580
,double8B

	full_text

double %580
mcall8Bc
a
	full_textT
R
P%582 = tail call double @llvm.fmuladd.f64(double %506, double %556, double %581)
,double8B

	full_text

double %506
,double8B

	full_text

double %556
,double8B

	full_text

double %581
vcall8Bl
j
	full_text]
[
Y%583 = tail call double @llvm.fmuladd.f64(double %582, double -3.150000e+01, double %579)
,double8B

	full_text

double %582
,double8B

	full_text

double %579
•getelementptr8Bë
é
	full_textÄ
~
|%584 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %523, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %523
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%585 = load double, double* %584, align 8, !tbaa !8
.double*8B

	full_text

double* %584
vcall8Bl
j
	full_text]
[
Y%586 = tail call double @llvm.fmuladd.f64(double %510, double -2.000000e+00, double %505)
,double8B

	full_text

double %510
,double8B

	full_text

double %505
:fadd8B0
.
	full_text!

%587 = fadd double %515, %586
,double8B

	full_text

double %515
,double8B

	full_text

double %586
{call8Bq
o
	full_textb
`
^%588 = tail call double @llvm.fmuladd.f64(double %587, double 0x40A7418000000001, double %585)
,double8B

	full_text

double %587
,double8B

	full_text

double %585
vcall8Bl
j
	full_text]
[
Y%589 = tail call double @llvm.fmuladd.f64(double %527, double -2.000000e+00, double %556)
,double8B

	full_text

double %527
,double8B

	full_text

double %556
:fadd8B0
.
	full_text!

%590 = fadd double %526, %589
,double8B

	full_text

double %526
,double8B

	full_text

double %589
ucall8Bk
i
	full_text\
Z
X%591 = tail call double @llvm.fmuladd.f64(double %590, double 5.292000e+02, double %588)
,double8B

	full_text

double %590
,double8B

	full_text

double %588
:fmul8B0
.
	full_text!

%592 = fmul double %526, %515
,double8B

	full_text

double %526
,double8B

	full_text

double %515
Cfsub8B9
7
	full_text*
(
&%593 = fsub double -0.000000e+00, %592
,double8B

	full_text

double %592
mcall8Bc
a
	full_textT
R
P%594 = tail call double @llvm.fmuladd.f64(double %505, double %556, double %593)
,double8B

	full_text

double %505
,double8B

	full_text

double %556
,double8B

	full_text

double %593
:fsub8B0
.
	full_text!

%595 = fsub double %503, %564
,double8B

	full_text

double %503
,double8B

	full_text

double %564
Abitcast8B4
2
	full_text%
#
!%596 = bitcast i64 %513 to double
&i648B

	full_text


i64 %513
:fsub8B0
.
	full_text!

%597 = fsub double %595, %596
,double8B

	full_text

double %595
,double8B

	full_text

double %596
:fadd8B0
.
	full_text!

%598 = fadd double %529, %597
,double8B

	full_text

double %529
,double8B

	full_text

double %597
ucall8Bk
i
	full_text\
Z
X%599 = tail call double @llvm.fmuladd.f64(double %598, double 4.000000e-01, double %594)
,double8B

	full_text

double %598
,double8B

	full_text

double %594
vcall8Bl
j
	full_text]
[
Y%600 = tail call double @llvm.fmuladd.f64(double %599, double -3.150000e+01, double %591)
,double8B

	full_text

double %599
,double8B

	full_text

double %591
•getelementptr8Bë
é
	full_textÄ
~
|%601 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %523, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %523
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%602 = load double, double* %601, align 8, !tbaa !8
.double*8B

	full_text

double* %601
vcall8Bl
j
	full_text]
[
Y%603 = tail call double @llvm.fmuladd.f64(double %509, double -2.000000e+00, double %504)
,double8B

	full_text

double %509
,double8B

	full_text

double %504
Pload8BF
D
	full_text7
5
3%604 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
:fadd8B0
.
	full_text!

%605 = fadd double %603, %604
,double8B

	full_text

double %603
,double8B

	full_text

double %604
{call8Bq
o
	full_textb
`
^%606 = tail call double @llvm.fmuladd.f64(double %605, double 0x40A7418000000001, double %602)
,double8B

	full_text

double %605
,double8B

	full_text

double %602
vcall8Bl
j
	full_text]
[
Y%607 = tail call double @llvm.fmuladd.f64(double %534, double -2.000000e+00, double %558)
,double8B

	full_text

double %534
,double8B

	full_text

double %558
:fadd8B0
.
	full_text!

%608 = fadd double %535, %607
,double8B

	full_text

double %535
,double8B

	full_text

double %607
{call8Bq
o
	full_textb
`
^%609 = tail call double @llvm.fmuladd.f64(double %608, double 0x4078CE6666666667, double %606)
,double8B

	full_text

double %608
,double8B

	full_text

double %606
:fmul8B0
.
	full_text!

%610 = fmul double %526, %604
,double8B

	full_text

double %526
,double8B

	full_text

double %604
Cfsub8B9
7
	full_text*
(
&%611 = fsub double -0.000000e+00, %610
,double8B

	full_text

double %610
mcall8Bc
a
	full_textT
R
P%612 = tail call double @llvm.fmuladd.f64(double %504, double %556, double %611)
,double8B

	full_text

double %504
,double8B

	full_text

double %556
,double8B

	full_text

double %611
vcall8Bl
j
	full_text]
[
Y%613 = tail call double @llvm.fmuladd.f64(double %612, double -3.150000e+01, double %609)
,double8B

	full_text

double %612
,double8B

	full_text

double %609
•getelementptr8Bë
é
	full_textÄ
~
|%614 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %523, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %523
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%615 = load double, double* %614, align 8, !tbaa !8
.double*8B

	full_text

double* %614
Pload8BF
D
	full_text7
5
3%616 = load double, double* %69, align 16, !tbaa !8
-double*8B

	full_text

double* %69
vcall8Bl
j
	full_text]
[
Y%617 = tail call double @llvm.fmuladd.f64(double %616, double -2.000000e+00, double %503)
,double8B

	full_text

double %616
,double8B

	full_text

double %503
:fadd8B0
.
	full_text!

%618 = fadd double %617, %596
,double8B

	full_text

double %617
,double8B

	full_text

double %596
{call8Bq
o
	full_textb
`
^%619 = tail call double @llvm.fmuladd.f64(double %618, double 0x40A7418000000001, double %615)
,double8B

	full_text

double %618
,double8B

	full_text

double %615
vcall8Bl
j
	full_text]
[
Y%620 = tail call double @llvm.fmuladd.f64(double %532, double -2.000000e+00, double %560)
,double8B

	full_text

double %532
,double8B

	full_text

double %560
:fadd8B0
.
	full_text!

%621 = fadd double %533, %620
,double8B

	full_text

double %533
,double8B

	full_text

double %620
{call8Bq
o
	full_textb
`
^%622 = tail call double @llvm.fmuladd.f64(double %621, double 0xC077D0624DD2F1A9, double %619)
,double8B

	full_text

double %621
,double8B

	full_text

double %619
Bfmul8B8
6
	full_text)
'
%%623 = fmul double %527, 2.000000e+00
,double8B

	full_text

double %527
:fmul8B0
.
	full_text!

%624 = fmul double %527, %623
,double8B

	full_text

double %527
,double8B

	full_text

double %623
Cfsub8B9
7
	full_text*
(
&%625 = fsub double -0.000000e+00, %624
,double8B

	full_text

double %624
mcall8Bc
a
	full_textT
R
P%626 = tail call double @llvm.fmuladd.f64(double %556, double %556, double %625)
,double8B

	full_text

double %556
,double8B

	full_text

double %556
,double8B

	full_text

double %625
mcall8Bc
a
	full_textT
R
P%627 = tail call double @llvm.fmuladd.f64(double %526, double %526, double %626)
,double8B

	full_text

double %526
,double8B

	full_text

double %526
,double8B

	full_text

double %626
ucall8Bk
i
	full_text\
Z
X%628 = tail call double @llvm.fmuladd.f64(double %627, double 6.615000e+01, double %622)
,double8B

	full_text

double %627
,double8B

	full_text

double %622
Bfmul8B8
6
	full_text)
'
%%629 = fmul double %616, 2.000000e+00
,double8B

	full_text

double %616
:fmul8B0
.
	full_text!

%630 = fmul double %530, %629
,double8B

	full_text

double %530
,double8B

	full_text

double %629
Cfsub8B9
7
	full_text*
(
&%631 = fsub double -0.000000e+00, %630
,double8B

	full_text

double %630
mcall8Bc
a
	full_textT
R
P%632 = tail call double @llvm.fmuladd.f64(double %503, double %562, double %631)
,double8B

	full_text

double %503
,double8B

	full_text

double %562
,double8B

	full_text

double %631
mcall8Bc
a
	full_textT
R
P%633 = tail call double @llvm.fmuladd.f64(double %596, double %531, double %632)
,double8B

	full_text

double %596
,double8B

	full_text

double %531
,double8B

	full_text

double %632
{call8Bq
o
	full_textb
`
^%634 = tail call double @llvm.fmuladd.f64(double %633, double 0x40884F645A1CAC08, double %628)
,double8B

	full_text

double %633
,double8B

	full_text

double %628
Bfmul8B8
6
	full_text)
'
%%635 = fmul double %564, 4.000000e-01
,double8B

	full_text

double %564
Cfsub8B9
7
	full_text*
(
&%636 = fsub double -0.000000e+00, %635
,double8B

	full_text

double %635
ucall8Bk
i
	full_text\
Z
X%637 = tail call double @llvm.fmuladd.f64(double %503, double 1.400000e+00, double %636)
,double8B

	full_text

double %503
,double8B

	full_text

double %636
Bfmul8B8
6
	full_text)
'
%%638 = fmul double %529, 4.000000e-01
,double8B

	full_text

double %529
Cfsub8B9
7
	full_text*
(
&%639 = fsub double -0.000000e+00, %638
,double8B

	full_text

double %638
ucall8Bk
i
	full_text\
Z
X%640 = tail call double @llvm.fmuladd.f64(double %596, double 1.400000e+00, double %639)
,double8B

	full_text

double %596
,double8B

	full_text

double %639
:fmul8B0
.
	full_text!

%641 = fmul double %526, %640
,double8B

	full_text

double %526
,double8B

	full_text

double %640
Cfsub8B9
7
	full_text*
(
&%642 = fsub double -0.000000e+00, %641
,double8B

	full_text

double %641
mcall8Bc
a
	full_textT
R
P%643 = tail call double @llvm.fmuladd.f64(double %637, double %556, double %642)
,double8B

	full_text

double %637
,double8B

	full_text

double %556
,double8B

	full_text

double %642
vcall8Bl
j
	full_text]
[
Y%644 = tail call double @llvm.fmuladd.f64(double %643, double -3.150000e+01, double %634)
,double8B

	full_text

double %643
,double8B

	full_text

double %634
Qload8BG
E
	full_text8
6
4%645 = load double, double* %500, align 16, !tbaa !8
.double*8B

	full_text

double* %500
vcall8Bl
j
	full_text]
[
Y%646 = tail call double @llvm.fmuladd.f64(double %517, double -4.000000e+00, double %645)
,double8B

	full_text

double %517
,double8B

	full_text

double %645
ucall8Bk
i
	full_text\
Z
X%647 = tail call double @llvm.fmuladd.f64(double %512, double 6.000000e+00, double %646)
,double8B

	full_text

double %512
,double8B

	full_text

double %646
vcall8Bl
j
	full_text]
[
Y%648 = tail call double @llvm.fmuladd.f64(double %507, double -4.000000e+00, double %647)
,double8B

	full_text

double %507
,double8B

	full_text

double %647
Qload8BG
E
	full_text8
6
4%649 = load double, double* %296, align 16, !tbaa !8
.double*8B

	full_text

double* %296
:fadd8B0
.
	full_text!

%650 = fadd double %648, %649
,double8B

	full_text

double %648
,double8B

	full_text

double %649
mcall8Bc
a
	full_textT
R
P%651 = tail call double @llvm.fmuladd.f64(double %290, double %650, double %571)
,double8B

	full_text

double %290
,double8B

	full_text

double %650
,double8B

	full_text

double %571
Pstore8BE
C
	full_text6
4
2store double %651, double* %565, align 8, !tbaa !8
,double8B

	full_text

double %651
.double*8B

	full_text

double* %565
Pload8BF
D
	full_text7
5
3%652 = load double, double* %151, align 8, !tbaa !8
.double*8B

	full_text

double* %151
vcall8Bl
j
	full_text]
[
Y%653 = tail call double @llvm.fmuladd.f64(double %516, double -4.000000e+00, double %652)
,double8B

	full_text

double %516
,double8B

	full_text

double %652
ucall8Bk
i
	full_text\
Z
X%654 = tail call double @llvm.fmuladd.f64(double %511, double 6.000000e+00, double %653)
,double8B

	full_text

double %511
,double8B

	full_text

double %653
vcall8Bl
j
	full_text]
[
Y%655 = tail call double @llvm.fmuladd.f64(double %506, double -4.000000e+00, double %654)
,double8B

	full_text

double %506
,double8B

	full_text

double %654
Pload8BF
D
	full_text7
5
3%656 = load double, double* %103, align 8, !tbaa !8
.double*8B

	full_text

double* %103
:fadd8B0
.
	full_text!

%657 = fadd double %655, %656
,double8B

	full_text

double %655
,double8B

	full_text

double %656
mcall8Bc
a
	full_textT
R
P%658 = tail call double @llvm.fmuladd.f64(double %290, double %657, double %583)
,double8B

	full_text

double %290
,double8B

	full_text

double %657
,double8B

	full_text

double %583
Pstore8BE
C
	full_text6
4
2store double %658, double* %572, align 8, !tbaa !8
,double8B

	full_text

double %658
.double*8B

	full_text

double* %572
Qload8BG
E
	full_text8
6
4%659 = load double, double* %156, align 16, !tbaa !8
.double*8B

	full_text

double* %156
vcall8Bl
j
	full_text]
[
Y%660 = tail call double @llvm.fmuladd.f64(double %515, double -4.000000e+00, double %659)
,double8B

	full_text

double %515
,double8B

	full_text

double %659
ucall8Bk
i
	full_text\
Z
X%661 = tail call double @llvm.fmuladd.f64(double %510, double 6.000000e+00, double %660)
,double8B

	full_text

double %510
,double8B

	full_text

double %660
vcall8Bl
j
	full_text]
[
Y%662 = tail call double @llvm.fmuladd.f64(double %505, double -4.000000e+00, double %661)
,double8B

	full_text

double %505
,double8B

	full_text

double %661
Qload8BG
E
	full_text8
6
4%663 = load double, double* %108, align 16, !tbaa !8
.double*8B

	full_text

double* %108
:fadd8B0
.
	full_text!

%664 = fadd double %662, %663
,double8B

	full_text

double %662
,double8B

	full_text

double %663
mcall8Bc
a
	full_textT
R
P%665 = tail call double @llvm.fmuladd.f64(double %290, double %664, double %600)
,double8B

	full_text

double %290
,double8B

	full_text

double %664
,double8B

	full_text

double %600
Pstore8BE
C
	full_text6
4
2store double %665, double* %584, align 8, !tbaa !8
,double8B

	full_text

double %665
.double*8B

	full_text

double* %584
Pload8BF
D
	full_text7
5
3%666 = load double, double* %161, align 8, !tbaa !8
.double*8B

	full_text

double* %161
vcall8Bl
j
	full_text]
[
Y%667 = tail call double @llvm.fmuladd.f64(double %604, double -4.000000e+00, double %666)
,double8B

	full_text

double %604
,double8B

	full_text

double %666
ucall8Bk
i
	full_text\
Z
X%668 = tail call double @llvm.fmuladd.f64(double %509, double 6.000000e+00, double %667)
,double8B

	full_text

double %509
,double8B

	full_text

double %667
vcall8Bl
j
	full_text]
[
Y%669 = tail call double @llvm.fmuladd.f64(double %504, double -4.000000e+00, double %668)
,double8B

	full_text

double %504
,double8B

	full_text

double %668
Pload8BF
D
	full_text7
5
3%670 = load double, double* %113, align 8, !tbaa !8
.double*8B

	full_text

double* %113
:fadd8B0
.
	full_text!

%671 = fadd double %669, %670
,double8B

	full_text

double %669
,double8B

	full_text

double %670
mcall8Bc
a
	full_textT
R
P%672 = tail call double @llvm.fmuladd.f64(double %290, double %671, double %613)
,double8B

	full_text

double %290
,double8B

	full_text

double %671
,double8B

	full_text

double %613
Pstore8BE
C
	full_text6
4
2store double %672, double* %601, align 8, !tbaa !8
,double8B

	full_text

double %672
.double*8B

	full_text

double* %601
Qload8BG
E
	full_text8
6
4%673 = load double, double* %166, align 16, !tbaa !8
.double*8B

	full_text

double* %166
Qload8BG
E
	full_text8
6
4%674 = load double, double* %163, align 16, !tbaa !8
.double*8B

	full_text

double* %163
vcall8Bl
j
	full_text]
[
Y%675 = tail call double @llvm.fmuladd.f64(double %674, double -4.000000e+00, double %673)
,double8B

	full_text

double %674
,double8B

	full_text

double %673
ucall8Bk
i
	full_text\
Z
X%676 = tail call double @llvm.fmuladd.f64(double %616, double 6.000000e+00, double %675)
,double8B

	full_text

double %616
,double8B

	full_text

double %675
vcall8Bl
j
	full_text]
[
Y%677 = tail call double @llvm.fmuladd.f64(double %503, double -4.000000e+00, double %676)
,double8B

	full_text

double %503
,double8B

	full_text

double %676
Qload8BG
E
	full_text8
6
4%678 = load double, double* %118, align 16, !tbaa !8
.double*8B

	full_text

double* %118
:fadd8B0
.
	full_text!

%679 = fadd double %677, %678
,double8B

	full_text

double %677
,double8B

	full_text

double %678
mcall8Bc
a
	full_textT
R
P%680 = tail call double @llvm.fmuladd.f64(double %290, double %679, double %644)
,double8B

	full_text

double %290
,double8B

	full_text

double %679
,double8B

	full_text

double %644
Pstore8BE
C
	full_text6
4
2store double %680, double* %614, align 8, !tbaa !8
,double8B

	full_text

double %680
.double*8B

	full_text

double* %614
:icmp8B0
.
	full_text!

%681 = icmp eq i64 %552, %501
&i648B

	full_text


i64 %552
&i648B

	full_text


i64 %501
Abitcast8B4
2
	full_text%
#
!%682 = bitcast double %517 to i64
,double8B

	full_text

double %517
Abitcast8B4
2
	full_text%
#
!%683 = bitcast double %516 to i64
,double8B

	full_text

double %516
Abitcast8B4
2
	full_text%
#
!%684 = bitcast double %515 to i64
,double8B

	full_text

double %515
Abitcast8B4
2
	full_text%
#
!%685 = bitcast double %604 to i64
,double8B

	full_text

double %604
Abitcast8B4
2
	full_text%
#
!%686 = bitcast double %674 to i64
,double8B

	full_text

double %674
Abitcast8B4
2
	full_text%
#
!%687 = bitcast double %509 to i64
,double8B

	full_text

double %509
Abitcast8B4
2
	full_text%
#
!%688 = bitcast double %616 to i64
,double8B

	full_text

double %616
Abitcast8B4
2
	full_text%
#
!%689 = bitcast double %503 to i64
,double8B

	full_text

double %503
=br8B5
3
	full_text&
$
"br i1 %681, label %690, label %502
$i18B

	full_text
	
i1 %681
Qstore8BF
D
	full_text7
5
3store double %517, double* %497, align 16, !tbaa !8
,double8B

	full_text

double %517
.double*8B

	full_text

double* %497
Pstore8BE
C
	full_text6
4
2store double %516, double* %148, align 8, !tbaa !8
,double8B

	full_text

double %516
.double*8B

	full_text

double* %148
Qstore8BF
D
	full_text7
5
3store double %515, double* %153, align 16, !tbaa !8
,double8B

	full_text

double %515
.double*8B

	full_text

double* %153
Qstore8BF
D
	full_text7
5
3store double %512, double* %498, align 16, !tbaa !8
,double8B

	full_text

double %512
.double*8B

	full_text

double* %498
Ostore8BD
B
	full_text5
3
1store double %511, double* %54, align 8, !tbaa !8
,double8B

	full_text

double %511
-double*8B

	full_text

double* %54
Pstore8BE
C
	full_text6
4
2store double %510, double* %59, align 16, !tbaa !8
,double8B

	full_text

double %510
-double*8B

	full_text

double* %59
Ostore8BD
B
	full_text5
3
1store double %509, double* %64, align 8, !tbaa !8
,double8B

	full_text

double %509
-double*8B

	full_text

double* %64
Qstore8BF
D
	full_text7
5
3store double %507, double* %499, align 16, !tbaa !8
,double8B

	full_text

double %507
.double*8B

	full_text

double* %499
Ostore8BD
B
	full_text5
3
1store double %506, double* %79, align 8, !tbaa !8
,double8B

	full_text

double %506
-double*8B

	full_text

double* %79
Pstore8BE
C
	full_text6
4
2store double %505, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %505
-double*8B

	full_text

double* %84
Ostore8BD
B
	full_text5
3
1store double %504, double* %89, align 8, !tbaa !8
,double8B

	full_text

double %504
-double*8B

	full_text

double* %89
Pstore8BE
C
	full_text6
4
2store double %503, double* %94, align 16, !tbaa !8
,double8B

	full_text

double %503
-double*8B

	full_text

double* %94
(br8B 

	full_text

br label %691
Mphi8BD
B
	full_text5
3
1%692 = phi double* [ %495, %494 ], [ %500, %690 ]
.double*8B

	full_text

double* %495
.double*8B

	full_text

double* %500
Lphi8BC
A
	full_text4
2
0%693 = phi double [ %482, %494 ], [ %678, %690 ]
,double8B

	full_text

double %482
,double8B

	full_text

double %678
Lphi8BC
A
	full_text4
2
0%694 = phi double [ %474, %494 ], [ %670, %690 ]
,double8B

	full_text

double %474
,double8B

	full_text

double %670
Lphi8BC
A
	full_text4
2
0%695 = phi double [ %465, %494 ], [ %663, %690 ]
,double8B

	full_text

double %465
,double8B

	full_text

double %663
Lphi8BC
A
	full_text4
2
0%696 = phi double [ %456, %494 ], [ %656, %690 ]
,double8B

	full_text

double %456
,double8B

	full_text

double %656
Lphi8BC
A
	full_text4
2
0%697 = phi double [ %447, %494 ], [ %649, %690 ]
,double8B

	full_text

double %447
,double8B

	full_text

double %649
Lphi8BC
A
	full_text4
2
0%698 = phi double [ %480, %494 ], [ %503, %690 ]
,double8B

	full_text

double %480
,double8B

	full_text

double %503
Iphi8B@
>
	full_text1
/
-%699 = phi i64 [ %492, %494 ], [ %689, %690 ]
&i648B

	full_text


i64 %492
&i648B

	full_text


i64 %689
Lphi8BC
A
	full_text4
2
0%700 = phi double [ %472, %494 ], [ %504, %690 ]
,double8B

	full_text

double %472
,double8B

	full_text

double %504
Lphi8BC
A
	full_text4
2
0%701 = phi double [ %463, %494 ], [ %505, %690 ]
,double8B

	full_text

double %463
,double8B

	full_text

double %505
Lphi8BC
A
	full_text4
2
0%702 = phi double [ %454, %494 ], [ %506, %690 ]
,double8B

	full_text

double %454
,double8B

	full_text

double %506
Lphi8BC
A
	full_text4
2
0%703 = phi double [ %445, %494 ], [ %507, %690 ]
,double8B

	full_text

double %445
,double8B

	full_text

double %507
Iphi8B@
>
	full_text1
/
-%704 = phi i64 [ %117, %494 ], [ %688, %690 ]
&i648B

	full_text


i64 %117
&i648B

	full_text


i64 %688
Iphi8B@
>
	full_text1
/
-%705 = phi i64 [ %491, %494 ], [ %687, %690 ]
&i648B

	full_text


i64 %491
&i648B

	full_text


i64 %687
Lphi8BC
A
	full_text4
2
0%706 = phi double [ %460, %494 ], [ %510, %690 ]
,double8B

	full_text

double %460
,double8B

	full_text

double %510
Lphi8BC
A
	full_text4
2
0%707 = phi double [ %451, %494 ], [ %511, %690 ]
,double8B

	full_text

double %451
,double8B

	full_text

double %511
Lphi8BC
A
	full_text4
2
0%708 = phi double [ %442, %494 ], [ %512, %690 ]
,double8B

	full_text

double %442
,double8B

	full_text

double %512
Iphi8B@
>
	full_text1
/
-%709 = phi i64 [ %490, %494 ], [ %686, %690 ]
&i648B

	full_text


i64 %490
&i648B

	full_text


i64 %686
Iphi8B@
>
	full_text1
/
-%710 = phi i64 [ %489, %494 ], [ %685, %690 ]
&i648B

	full_text


i64 %489
&i648B

	full_text


i64 %685
Iphi8B@
>
	full_text1
/
-%711 = phi i64 [ %488, %494 ], [ %684, %690 ]
&i648B

	full_text


i64 %488
&i648B

	full_text


i64 %684
Iphi8B@
>
	full_text1
/
-%712 = phi i64 [ %487, %494 ], [ %683, %690 ]
&i648B

	full_text


i64 %487
&i648B

	full_text


i64 %683
Iphi8B@
>
	full_text1
/
-%713 = phi i64 [ %486, %494 ], [ %682, %690 ]
&i648B

	full_text


i64 %486
&i648B

	full_text


i64 %682
Lphi8BC
A
	full_text4
2
0%714 = phi double [ %188, %494 ], [ %534, %690 ]
,double8B

	full_text

double %188
,double8B

	full_text

double %534
Lphi8BC
A
	full_text4
2
0%715 = phi double [ %351, %494 ], [ %558, %690 ]
,double8B

	full_text

double %351
,double8B

	full_text

double %558
Lphi8BC
A
	full_text4
2
0%716 = phi double [ %190, %494 ], [ %532, %690 ]
,double8B

	full_text

double %190
,double8B

	full_text

double %532
Lphi8BC
A
	full_text4
2
0%717 = phi double [ %353, %494 ], [ %560, %690 ]
,double8B

	full_text

double %353
,double8B

	full_text

double %560
Lphi8BC
A
	full_text4
2
0%718 = phi double [ %192, %494 ], [ %530, %690 ]
,double8B

	full_text

double %192
,double8B

	full_text

double %530
Lphi8BC
A
	full_text4
2
0%719 = phi double [ %355, %494 ], [ %562, %690 ]
,double8B

	full_text

double %355
,double8B

	full_text

double %562
Lphi8BC
A
	full_text4
2
0%720 = phi double [ %194, %494 ], [ %528, %690 ]
,double8B

	full_text

double %194
,double8B

	full_text

double %528
Lphi8BC
A
	full_text4
2
0%721 = phi double [ %357, %494 ], [ %564, %690 ]
,double8B

	full_text

double %357
,double8B

	full_text

double %564
Lphi8BC
A
	full_text4
2
0%722 = phi double [ %349, %494 ], [ %556, %690 ]
,double8B

	full_text

double %349
,double8B

	full_text

double %556
Lphi8BC
A
	full_text4
2
0%723 = phi double [ %186, %494 ], [ %527, %690 ]
,double8B

	full_text

double %186
,double8B

	full_text

double %527
Lphi8BC
A
	full_text4
2
0%724 = phi double [ %347, %494 ], [ %554, %690 ]
,double8B

	full_text

double %347
,double8B

	full_text

double %554
Lphi8BC
A
	full_text4
2
0%725 = phi double [ %184, %494 ], [ %525, %690 ]
,double8B

	full_text

double %184
,double8B

	full_text

double %525
Kstore8B@
>
	full_text1
/
-store i64 %713, i64* %147, align 16, !tbaa !8
&i648B

	full_text


i64 %713
(i64*8B

	full_text

	i64* %147
Jstore8B?
=
	full_text0
.
,store i64 %712, i64* %152, align 8, !tbaa !8
&i648B

	full_text


i64 %712
(i64*8B

	full_text

	i64* %152
Kstore8B@
>
	full_text1
/
-store i64 %711, i64* %157, align 16, !tbaa !8
&i648B

	full_text


i64 %711
(i64*8B

	full_text

	i64* %157
Jstore8B?
=
	full_text0
.
,store i64 %710, i64* %162, align 8, !tbaa !8
&i648B

	full_text


i64 %710
(i64*8B

	full_text

	i64* %162
Kstore8B@
>
	full_text1
/
-store i64 %709, i64* %167, align 16, !tbaa !8
&i648B

	full_text


i64 %709
(i64*8B

	full_text

	i64* %167
qgetelementptr8B^
\
	full_textO
M
K%726 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %708, double* %726, align 16, !tbaa !8
,double8B

	full_text

double %708
.double*8B

	full_text

double* %726
Pstore8BE
C
	full_text6
4
2store double %707, double* %148, align 8, !tbaa !8
,double8B

	full_text

double %707
.double*8B

	full_text

double* %148
Qstore8BF
D
	full_text7
5
3store double %706, double* %153, align 16, !tbaa !8
,double8B

	full_text

double %706
.double*8B

	full_text

double* %153
Jstore8B?
=
	full_text0
.
,store i64 %705, i64* %159, align 8, !tbaa !8
&i648B

	full_text


i64 %705
(i64*8B

	full_text

	i64* %159
Kstore8B@
>
	full_text1
/
-store i64 %704, i64* %164, align 16, !tbaa !8
&i648B

	full_text


i64 %704
(i64*8B

	full_text

	i64* %164
qgetelementptr8B^
\
	full_textO
M
K%727 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Qstore8BF
D
	full_text7
5
3store double %703, double* %727, align 16, !tbaa !8
,double8B

	full_text

double %703
.double*8B

	full_text

double* %727
Ostore8BD
B
	full_text5
3
1store double %702, double* %54, align 8, !tbaa !8
,double8B

	full_text

double %702
-double*8B

	full_text

double* %54
Pstore8BE
C
	full_text6
4
2store double %701, double* %59, align 16, !tbaa !8
,double8B

	full_text

double %701
-double*8B

	full_text

double* %59
Ostore8BD
B
	full_text5
3
1store double %700, double* %64, align 8, !tbaa !8
,double8B

	full_text

double %700
-double*8B

	full_text

double* %64
Pstore8BE
C
	full_text6
4
2store double %698, double* %69, align 16, !tbaa !8
,double8B

	full_text

double %698
-double*8B

	full_text

double* %69
qgetelementptr8B^
\
	full_textO
M
K%728 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Qstore8BF
D
	full_text7
5
3store double %697, double* %728, align 16, !tbaa !8
,double8B

	full_text

double %697
.double*8B

	full_text

double* %728
Ostore8BD
B
	full_text5
3
1store double %696, double* %79, align 8, !tbaa !8
,double8B

	full_text

double %696
-double*8B

	full_text

double* %79
Pstore8BE
C
	full_text6
4
2store double %695, double* %84, align 16, !tbaa !8
,double8B

	full_text

double %695
-double*8B

	full_text

double* %84
Ostore8BD
B
	full_text5
3
1store double %694, double* %89, align 8, !tbaa !8
,double8B

	full_text

double %694
-double*8B

	full_text

double* %89
Pstore8BE
C
	full_text6
4
2store double %693, double* %94, align 16, !tbaa !8
,double8B

	full_text

double %693
-double*8B

	full_text

double* %94
5add8B,
*
	full_text

%729 = add nsw i32 %9, -1
8sext8B.
,
	full_text

%730 = sext i32 %729 to i64
&i328B

	full_text


i32 %729
ùgetelementptr8Bâ
Ü
	full_texty
w
u%731 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %730, i64 %45
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Ibitcast8B<
:
	full_text-
+
)%732 = bitcast [5 x double]* %731 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %731
Jload8B@
>
	full_text1
/
-%733 = load i64, i64* %732, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %732
Jstore8B?
=
	full_text0
.
,store i64 %733, i64* %99, align 16, !tbaa !8
&i648B

	full_text


i64 %733
'i64*8B

	full_text


i64* %99
•getelementptr8Bë
é
	full_textÄ
~
|%734 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %730, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%735 = bitcast double* %734 to i64*
.double*8B

	full_text

double* %734
Jload8B@
>
	full_text1
/
-%736 = load i64, i64* %735, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %735
Jstore8B?
=
	full_text0
.
,store i64 %736, i64* %104, align 8, !tbaa !8
&i648B

	full_text


i64 %736
(i64*8B

	full_text

	i64* %104
•getelementptr8Bë
é
	full_textÄ
~
|%737 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %730, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%738 = bitcast double* %737 to i64*
.double*8B

	full_text

double* %737
Jload8B@
>
	full_text1
/
-%739 = load i64, i64* %738, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %738
Kstore8B@
>
	full_text1
/
-store i64 %739, i64* %109, align 16, !tbaa !8
&i648B

	full_text


i64 %739
(i64*8B

	full_text

	i64* %109
•getelementptr8Bë
é
	full_textÄ
~
|%740 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %730, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%741 = bitcast double* %740 to i64*
.double*8B

	full_text

double* %740
Jload8B@
>
	full_text1
/
-%742 = load i64, i64* %741, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %741
Jstore8B?
=
	full_text0
.
,store i64 %742, i64* %114, align 8, !tbaa !8
&i648B

	full_text


i64 %742
(i64*8B

	full_text

	i64* %114
•getelementptr8Bë
é
	full_textÄ
~
|%743 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %40, i64 %43, i64 %730, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %40
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Cbitcast8B6
4
	full_text'
%
#%744 = bitcast double* %743 to i64*
.double*8B

	full_text

double* %743
Jload8B@
>
	full_text1
/
-%745 = load i64, i64* %744, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %744
Kstore8B@
>
	full_text1
/
-store i64 %745, i64* %119, align 16, !tbaa !8
&i648B

	full_text


i64 %745
(i64*8B

	full_text

	i64* %119
5add8B,
*
	full_text

%746 = add nsw i32 %9, -2
8sext8B.
,
	full_text

%747 = sext i32 %746 to i64
&i328B

	full_text


i32 %746
ègetelementptr8B|
z
	full_textm
k
i%748 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 %747, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%749 = load double, double* %748, align 8, !tbaa !8
.double*8B

	full_text

double* %748
ègetelementptr8B|
z
	full_textm
k
i%750 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 %747, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%751 = load double, double* %750, align 8, !tbaa !8
.double*8B

	full_text

double* %750
ègetelementptr8B|
z
	full_textm
k
i%752 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 %747, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%753 = load double, double* %752, align 8, !tbaa !8
.double*8B

	full_text

double* %752
ègetelementptr8B|
z
	full_textm
k
i%754 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 %747, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%755 = load double, double* %754, align 8, !tbaa !8
.double*8B

	full_text

double* %754
ègetelementptr8B|
z
	full_textm
k
i%756 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 %747, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%757 = load double, double* %756, align 8, !tbaa !8
.double*8B

	full_text

double* %756
ègetelementptr8B|
z
	full_textm
k
i%758 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 %747, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%759 = load double, double* %758, align 8, !tbaa !8
.double*8B

	full_text

double* %758
8sext8B.
,
	full_text

%760 = sext i32 %493 to i64
&i328B

	full_text


i32 %493
•getelementptr8Bë
é
	full_textÄ
~
|%761 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %760, i64 %45, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %760
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%762 = load double, double* %761, align 8, !tbaa !8
.double*8B

	full_text

double* %761
vcall8Bl
j
	full_text]
[
Y%763 = tail call double @llvm.fmuladd.f64(double %703, double -2.000000e+00, double %697)
,double8B

	full_text

double %703
,double8B

	full_text

double %697
:fadd8B0
.
	full_text!

%764 = fadd double %763, %708
,double8B

	full_text

double %763
,double8B

	full_text

double %708
{call8Bq
o
	full_textb
`
^%765 = tail call double @llvm.fmuladd.f64(double %764, double 0x40A7418000000001, double %762)
,double8B

	full_text

double %764
,double8B

	full_text

double %762
:fsub8B0
.
	full_text!

%766 = fsub double %695, %706
,double8B

	full_text

double %695
,double8B

	full_text

double %706
vcall8Bl
j
	full_text]
[
Y%767 = tail call double @llvm.fmuladd.f64(double %766, double -3.150000e+01, double %765)
,double8B

	full_text

double %766
,double8B

	full_text

double %765
•getelementptr8Bë
é
	full_textÄ
~
|%768 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %760, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %760
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%769 = load double, double* %768, align 8, !tbaa !8
.double*8B

	full_text

double* %768
vcall8Bl
j
	full_text]
[
Y%770 = tail call double @llvm.fmuladd.f64(double %702, double -2.000000e+00, double %696)
,double8B

	full_text

double %702
,double8B

	full_text

double %696
:fadd8B0
.
	full_text!

%771 = fadd double %770, %707
,double8B

	full_text

double %770
,double8B

	full_text

double %707
{call8Bq
o
	full_textb
`
^%772 = tail call double @llvm.fmuladd.f64(double %771, double 0x40A7418000000001, double %769)
,double8B

	full_text

double %771
,double8B

	full_text

double %769
vcall8Bl
j
	full_text]
[
Y%773 = tail call double @llvm.fmuladd.f64(double %724, double -2.000000e+00, double %749)
,double8B

	full_text

double %724
,double8B

	full_text

double %749
:fadd8B0
.
	full_text!

%774 = fadd double %725, %773
,double8B

	full_text

double %725
,double8B

	full_text

double %773
{call8Bq
o
	full_textb
`
^%775 = tail call double @llvm.fmuladd.f64(double %774, double 0x4078CE6666666667, double %772)
,double8B

	full_text

double %774
,double8B

	full_text

double %772
:fmul8B0
.
	full_text!

%776 = fmul double %723, %707
,double8B

	full_text

double %723
,double8B

	full_text

double %707
Cfsub8B9
7
	full_text*
(
&%777 = fsub double -0.000000e+00, %776
,double8B

	full_text

double %776
mcall8Bc
a
	full_textT
R
P%778 = tail call double @llvm.fmuladd.f64(double %696, double %751, double %777)
,double8B

	full_text

double %696
,double8B

	full_text

double %751
,double8B

	full_text

double %777
vcall8Bl
j
	full_text]
[
Y%779 = tail call double @llvm.fmuladd.f64(double %778, double -3.150000e+01, double %775)
,double8B

	full_text

double %778
,double8B

	full_text

double %775
•getelementptr8Bë
é
	full_textÄ
~
|%780 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %760, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %760
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%781 = load double, double* %780, align 8, !tbaa !8
.double*8B

	full_text

double* %780
vcall8Bl
j
	full_text]
[
Y%782 = tail call double @llvm.fmuladd.f64(double %701, double -2.000000e+00, double %695)
,double8B

	full_text

double %701
,double8B

	full_text

double %695
:fadd8B0
.
	full_text!

%783 = fadd double %706, %782
,double8B

	full_text

double %706
,double8B

	full_text

double %782
{call8Bq
o
	full_textb
`
^%784 = tail call double @llvm.fmuladd.f64(double %783, double 0x40A7418000000001, double %781)
,double8B

	full_text

double %783
,double8B

	full_text

double %781
vcall8Bl
j
	full_text]
[
Y%785 = tail call double @llvm.fmuladd.f64(double %722, double -2.000000e+00, double %751)
,double8B

	full_text

double %722
,double8B

	full_text

double %751
:fadd8B0
.
	full_text!

%786 = fadd double %723, %785
,double8B

	full_text

double %723
,double8B

	full_text

double %785
ucall8Bk
i
	full_text\
Z
X%787 = tail call double @llvm.fmuladd.f64(double %786, double 5.292000e+02, double %784)
,double8B

	full_text

double %786
,double8B

	full_text

double %784
:fmul8B0
.
	full_text!

%788 = fmul double %723, %706
,double8B

	full_text

double %723
,double8B

	full_text

double %706
Cfsub8B9
7
	full_text*
(
&%789 = fsub double -0.000000e+00, %788
,double8B

	full_text

double %788
mcall8Bc
a
	full_textT
R
P%790 = tail call double @llvm.fmuladd.f64(double %695, double %751, double %789)
,double8B

	full_text

double %695
,double8B

	full_text

double %751
,double8B

	full_text

double %789
:fsub8B0
.
	full_text!

%791 = fsub double %693, %759
,double8B

	full_text

double %693
,double8B

	full_text

double %759
Abitcast8B4
2
	full_text%
#
!%792 = bitcast i64 %704 to double
&i648B

	full_text


i64 %704
:fsub8B0
.
	full_text!

%793 = fsub double %791, %792
,double8B

	full_text

double %791
,double8B

	full_text

double %792
:fadd8B0
.
	full_text!

%794 = fadd double %720, %793
,double8B

	full_text

double %720
,double8B

	full_text

double %793
ucall8Bk
i
	full_text\
Z
X%795 = tail call double @llvm.fmuladd.f64(double %794, double 4.000000e-01, double %790)
,double8B

	full_text

double %794
,double8B

	full_text

double %790
vcall8Bl
j
	full_text]
[
Y%796 = tail call double @llvm.fmuladd.f64(double %795, double -3.150000e+01, double %787)
,double8B

	full_text

double %795
,double8B

	full_text

double %787
•getelementptr8Bë
é
	full_textÄ
~
|%797 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %760, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %760
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%798 = load double, double* %797, align 8, !tbaa !8
.double*8B

	full_text

double* %797
vcall8Bl
j
	full_text]
[
Y%799 = tail call double @llvm.fmuladd.f64(double %700, double -2.000000e+00, double %694)
,double8B

	full_text

double %700
,double8B

	full_text

double %694
Pload8BF
D
	full_text7
5
3%800 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
:fadd8B0
.
	full_text!

%801 = fadd double %799, %800
,double8B

	full_text

double %799
,double8B

	full_text

double %800
{call8Bq
o
	full_textb
`
^%802 = tail call double @llvm.fmuladd.f64(double %801, double 0x40A7418000000001, double %798)
,double8B

	full_text

double %801
,double8B

	full_text

double %798
vcall8Bl
j
	full_text]
[
Y%803 = tail call double @llvm.fmuladd.f64(double %715, double -2.000000e+00, double %753)
,double8B

	full_text

double %715
,double8B

	full_text

double %753
:fadd8B0
.
	full_text!

%804 = fadd double %714, %803
,double8B

	full_text

double %714
,double8B

	full_text

double %803
{call8Bq
o
	full_textb
`
^%805 = tail call double @llvm.fmuladd.f64(double %804, double 0x4078CE6666666667, double %802)
,double8B

	full_text

double %804
,double8B

	full_text

double %802
:fmul8B0
.
	full_text!

%806 = fmul double %723, %800
,double8B

	full_text

double %723
,double8B

	full_text

double %800
Cfsub8B9
7
	full_text*
(
&%807 = fsub double -0.000000e+00, %806
,double8B

	full_text

double %806
mcall8Bc
a
	full_textT
R
P%808 = tail call double @llvm.fmuladd.f64(double %694, double %751, double %807)
,double8B

	full_text

double %694
,double8B

	full_text

double %751
,double8B

	full_text

double %807
vcall8Bl
j
	full_text]
[
Y%809 = tail call double @llvm.fmuladd.f64(double %808, double -3.150000e+01, double %805)
,double8B

	full_text

double %808
,double8B

	full_text

double %805
•getelementptr8Bë
é
	full_textÄ
~
|%810 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %760, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %760
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%811 = load double, double* %810, align 8, !tbaa !8
.double*8B

	full_text

double* %810
Pload8BF
D
	full_text7
5
3%812 = load double, double* %69, align 16, !tbaa !8
-double*8B

	full_text

double* %69
vcall8Bl
j
	full_text]
[
Y%813 = tail call double @llvm.fmuladd.f64(double %812, double -2.000000e+00, double %693)
,double8B

	full_text

double %812
,double8B

	full_text

double %693
:fadd8B0
.
	full_text!

%814 = fadd double %813, %792
,double8B

	full_text

double %813
,double8B

	full_text

double %792
{call8Bq
o
	full_textb
`
^%815 = tail call double @llvm.fmuladd.f64(double %814, double 0x40A7418000000001, double %811)
,double8B

	full_text

double %814
,double8B

	full_text

double %811
vcall8Bl
j
	full_text]
[
Y%816 = tail call double @llvm.fmuladd.f64(double %717, double -2.000000e+00, double %755)
,double8B

	full_text

double %717
,double8B

	full_text

double %755
:fadd8B0
.
	full_text!

%817 = fadd double %716, %816
,double8B

	full_text

double %716
,double8B

	full_text

double %816
{call8Bq
o
	full_textb
`
^%818 = tail call double @llvm.fmuladd.f64(double %817, double 0xC077D0624DD2F1A9, double %815)
,double8B

	full_text

double %817
,double8B

	full_text

double %815
Bfmul8B8
6
	full_text)
'
%%819 = fmul double %722, 2.000000e+00
,double8B

	full_text

double %722
:fmul8B0
.
	full_text!

%820 = fmul double %722, %819
,double8B

	full_text

double %722
,double8B

	full_text

double %819
Cfsub8B9
7
	full_text*
(
&%821 = fsub double -0.000000e+00, %820
,double8B

	full_text

double %820
mcall8Bc
a
	full_textT
R
P%822 = tail call double @llvm.fmuladd.f64(double %751, double %751, double %821)
,double8B

	full_text

double %751
,double8B

	full_text

double %751
,double8B

	full_text

double %821
mcall8Bc
a
	full_textT
R
P%823 = tail call double @llvm.fmuladd.f64(double %723, double %723, double %822)
,double8B

	full_text

double %723
,double8B

	full_text

double %723
,double8B

	full_text

double %822
ucall8Bk
i
	full_text\
Z
X%824 = tail call double @llvm.fmuladd.f64(double %823, double 6.615000e+01, double %818)
,double8B

	full_text

double %823
,double8B

	full_text

double %818
Bfmul8B8
6
	full_text)
'
%%825 = fmul double %812, 2.000000e+00
,double8B

	full_text

double %812
:fmul8B0
.
	full_text!

%826 = fmul double %719, %825
,double8B

	full_text

double %719
,double8B

	full_text

double %825
Cfsub8B9
7
	full_text*
(
&%827 = fsub double -0.000000e+00, %826
,double8B

	full_text

double %826
mcall8Bc
a
	full_textT
R
P%828 = tail call double @llvm.fmuladd.f64(double %693, double %757, double %827)
,double8B

	full_text

double %693
,double8B

	full_text

double %757
,double8B

	full_text

double %827
mcall8Bc
a
	full_textT
R
P%829 = tail call double @llvm.fmuladd.f64(double %792, double %718, double %828)
,double8B

	full_text

double %792
,double8B

	full_text

double %718
,double8B

	full_text

double %828
{call8Bq
o
	full_textb
`
^%830 = tail call double @llvm.fmuladd.f64(double %829, double 0x40884F645A1CAC08, double %824)
,double8B

	full_text

double %829
,double8B

	full_text

double %824
Bfmul8B8
6
	full_text)
'
%%831 = fmul double %759, 4.000000e-01
,double8B

	full_text

double %759
Cfsub8B9
7
	full_text*
(
&%832 = fsub double -0.000000e+00, %831
,double8B

	full_text

double %831
ucall8Bk
i
	full_text\
Z
X%833 = tail call double @llvm.fmuladd.f64(double %693, double 1.400000e+00, double %832)
,double8B

	full_text

double %693
,double8B

	full_text

double %832
Bfmul8B8
6
	full_text)
'
%%834 = fmul double %720, 4.000000e-01
,double8B

	full_text

double %720
Cfsub8B9
7
	full_text*
(
&%835 = fsub double -0.000000e+00, %834
,double8B

	full_text

double %834
ucall8Bk
i
	full_text\
Z
X%836 = tail call double @llvm.fmuladd.f64(double %792, double 1.400000e+00, double %835)
,double8B

	full_text

double %792
,double8B

	full_text

double %835
:fmul8B0
.
	full_text!

%837 = fmul double %723, %836
,double8B

	full_text

double %723
,double8B

	full_text

double %836
Cfsub8B9
7
	full_text*
(
&%838 = fsub double -0.000000e+00, %837
,double8B

	full_text

double %837
mcall8Bc
a
	full_textT
R
P%839 = tail call double @llvm.fmuladd.f64(double %833, double %751, double %838)
,double8B

	full_text

double %833
,double8B

	full_text

double %751
,double8B

	full_text

double %838
vcall8Bl
j
	full_text]
[
Y%840 = tail call double @llvm.fmuladd.f64(double %839, double -3.150000e+01, double %830)
,double8B

	full_text

double %839
,double8B

	full_text

double %830
Pload8BF
D
	full_text7
5
3%841 = load double, double* %692, align 8, !tbaa !8
.double*8B

	full_text

double* %692
Qload8BG
E
	full_text8
6
4%842 = load double, double* %144, align 16, !tbaa !8
.double*8B

	full_text

double* %144
vcall8Bl
j
	full_text]
[
Y%843 = tail call double @llvm.fmuladd.f64(double %842, double -4.000000e+00, double %841)
,double8B

	full_text

double %842
,double8B

	full_text

double %841
Pload8BF
D
	full_text7
5
3%844 = load double, double* %49, align 16, !tbaa !8
-double*8B

	full_text

double* %49
ucall8Bk
i
	full_text\
Z
X%845 = tail call double @llvm.fmuladd.f64(double %844, double 6.000000e+00, double %843)
,double8B

	full_text

double %844
,double8B

	full_text

double %843
Pload8BF
D
	full_text7
5
3%846 = load double, double* %74, align 16, !tbaa !8
-double*8B

	full_text

double* %74
vcall8Bl
j
	full_text]
[
Y%847 = tail call double @llvm.fmuladd.f64(double %846, double -4.000000e+00, double %845)
,double8B

	full_text

double %846
,double8B

	full_text

double %845
mcall8Bc
a
	full_textT
R
P%848 = tail call double @llvm.fmuladd.f64(double %290, double %847, double %767)
,double8B

	full_text

double %290
,double8B

	full_text

double %847
,double8B

	full_text

double %767
Pstore8BE
C
	full_text6
4
2store double %848, double* %761, align 8, !tbaa !8
,double8B

	full_text

double %848
.double*8B

	full_text

double* %761
Pload8BF
D
	full_text7
5
3%849 = load double, double* %151, align 8, !tbaa !8
.double*8B

	full_text

double* %151
Pload8BF
D
	full_text7
5
3%850 = load double, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
vcall8Bl
j
	full_text]
[
Y%851 = tail call double @llvm.fmuladd.f64(double %850, double -4.000000e+00, double %849)
,double8B

	full_text

double %850
,double8B

	full_text

double %849
Oload8BE
C
	full_text6
4
2%852 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
ucall8Bk
i
	full_text\
Z
X%853 = tail call double @llvm.fmuladd.f64(double %852, double 6.000000e+00, double %851)
,double8B

	full_text

double %852
,double8B

	full_text

double %851
Oload8BE
C
	full_text6
4
2%854 = load double, double* %79, align 8, !tbaa !8
-double*8B

	full_text

double* %79
vcall8Bl
j
	full_text]
[
Y%855 = tail call double @llvm.fmuladd.f64(double %854, double -4.000000e+00, double %853)
,double8B

	full_text

double %854
,double8B

	full_text

double %853
mcall8Bc
a
	full_textT
R
P%856 = tail call double @llvm.fmuladd.f64(double %290, double %855, double %779)
,double8B

	full_text

double %290
,double8B

	full_text

double %855
,double8B

	full_text

double %779
Pstore8BE
C
	full_text6
4
2store double %856, double* %768, align 8, !tbaa !8
,double8B

	full_text

double %856
.double*8B

	full_text

double* %768
Qload8BG
E
	full_text8
6
4%857 = load double, double* %156, align 16, !tbaa !8
.double*8B

	full_text

double* %156
Qload8BG
E
	full_text8
6
4%858 = load double, double* %153, align 16, !tbaa !8
.double*8B

	full_text

double* %153
vcall8Bl
j
	full_text]
[
Y%859 = tail call double @llvm.fmuladd.f64(double %858, double -4.000000e+00, double %857)
,double8B

	full_text

double %858
,double8B

	full_text

double %857
Pload8BF
D
	full_text7
5
3%860 = load double, double* %59, align 16, !tbaa !8
-double*8B

	full_text

double* %59
ucall8Bk
i
	full_text\
Z
X%861 = tail call double @llvm.fmuladd.f64(double %860, double 6.000000e+00, double %859)
,double8B

	full_text

double %860
,double8B

	full_text

double %859
Pload8BF
D
	full_text7
5
3%862 = load double, double* %84, align 16, !tbaa !8
-double*8B

	full_text

double* %84
vcall8Bl
j
	full_text]
[
Y%863 = tail call double @llvm.fmuladd.f64(double %862, double -4.000000e+00, double %861)
,double8B

	full_text

double %862
,double8B

	full_text

double %861
mcall8Bc
a
	full_textT
R
P%864 = tail call double @llvm.fmuladd.f64(double %290, double %863, double %796)
,double8B

	full_text

double %290
,double8B

	full_text

double %863
,double8B

	full_text

double %796
Pstore8BE
C
	full_text6
4
2store double %864, double* %780, align 8, !tbaa !8
,double8B

	full_text

double %864
.double*8B

	full_text

double* %780
Pload8BF
D
	full_text7
5
3%865 = load double, double* %161, align 8, !tbaa !8
.double*8B

	full_text

double* %161
vcall8Bl
j
	full_text]
[
Y%866 = tail call double @llvm.fmuladd.f64(double %800, double -4.000000e+00, double %865)
,double8B

	full_text

double %800
,double8B

	full_text

double %865
Oload8BE
C
	full_text6
4
2%867 = load double, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
ucall8Bk
i
	full_text\
Z
X%868 = tail call double @llvm.fmuladd.f64(double %867, double 6.000000e+00, double %866)
,double8B

	full_text

double %867
,double8B

	full_text

double %866
Oload8BE
C
	full_text6
4
2%869 = load double, double* %89, align 8, !tbaa !8
-double*8B

	full_text

double* %89
vcall8Bl
j
	full_text]
[
Y%870 = tail call double @llvm.fmuladd.f64(double %869, double -4.000000e+00, double %868)
,double8B

	full_text

double %869
,double8B

	full_text

double %868
mcall8Bc
a
	full_textT
R
P%871 = tail call double @llvm.fmuladd.f64(double %290, double %870, double %809)
,double8B

	full_text

double %290
,double8B

	full_text

double %870
,double8B

	full_text

double %809
Pstore8BE
C
	full_text6
4
2store double %871, double* %797, align 8, !tbaa !8
,double8B

	full_text

double %871
.double*8B

	full_text

double* %797
Qload8BG
E
	full_text8
6
4%872 = load double, double* %166, align 16, !tbaa !8
.double*8B

	full_text

double* %166
Qload8BG
E
	full_text8
6
4%873 = load double, double* %163, align 16, !tbaa !8
.double*8B

	full_text

double* %163
vcall8Bl
j
	full_text]
[
Y%874 = tail call double @llvm.fmuladd.f64(double %873, double -4.000000e+00, double %872)
,double8B

	full_text

double %873
,double8B

	full_text

double %872
ucall8Bk
i
	full_text\
Z
X%875 = tail call double @llvm.fmuladd.f64(double %812, double 6.000000e+00, double %874)
,double8B

	full_text

double %812
,double8B

	full_text

double %874
Pload8BF
D
	full_text7
5
3%876 = load double, double* %94, align 16, !tbaa !8
-double*8B

	full_text

double* %94
vcall8Bl
j
	full_text]
[
Y%877 = tail call double @llvm.fmuladd.f64(double %876, double -4.000000e+00, double %875)
,double8B

	full_text

double %876
,double8B

	full_text

double %875
mcall8Bc
a
	full_textT
R
P%878 = tail call double @llvm.fmuladd.f64(double %290, double %877, double %840)
,double8B

	full_text

double %290
,double8B

	full_text

double %877
,double8B

	full_text

double %840
Pstore8BE
C
	full_text6
4
2store double %878, double* %810, align 8, !tbaa !8
,double8B

	full_text

double %878
.double*8B

	full_text

double* %810
qgetelementptr8B^
\
	full_textO
M
K%879 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Qstore8BF
D
	full_text7
5
3store double %708, double* %879, align 16, !tbaa !8
,double8B

	full_text

double %708
.double*8B

	full_text

double* %879
Pstore8BE
C
	full_text6
4
2store double %707, double* %151, align 8, !tbaa !8
,double8B

	full_text

double %707
.double*8B

	full_text

double* %151
Qstore8BF
D
	full_text7
5
3store double %706, double* %156, align 16, !tbaa !8
,double8B

	full_text

double %706
.double*8B

	full_text

double* %156
Jstore8B?
=
	full_text0
.
,store i64 %705, i64* %162, align 8, !tbaa !8
&i648B

	full_text


i64 %705
(i64*8B

	full_text

	i64* %162
Kstore8B@
>
	full_text1
/
-store i64 %704, i64* %167, align 16, !tbaa !8
&i648B

	full_text


i64 %704
(i64*8B

	full_text

	i64* %167
qgetelementptr8B^
\
	full_textO
M
K%880 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %703, double* %880, align 16, !tbaa !8
,double8B

	full_text

double %703
.double*8B

	full_text

double* %880
Pstore8BE
C
	full_text6
4
2store double %702, double* %148, align 8, !tbaa !8
,double8B

	full_text

double %702
.double*8B

	full_text

double* %148
Qstore8BF
D
	full_text7
5
3store double %701, double* %153, align 16, !tbaa !8
,double8B

	full_text

double %701
.double*8B

	full_text

double* %153
Pstore8BE
C
	full_text6
4
2store double %700, double* %158, align 8, !tbaa !8
,double8B

	full_text

double %700
.double*8B

	full_text

double* %158
Kstore8B@
>
	full_text1
/
-store i64 %699, i64* %164, align 16, !tbaa !8
&i648B

	full_text


i64 %699
(i64*8B

	full_text

	i64* %164
qgetelementptr8B^
\
	full_textO
M
K%881 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Qstore8BF
D
	full_text7
5
3store double %697, double* %881, align 16, !tbaa !8
,double8B

	full_text

double %697
.double*8B

	full_text

double* %881
Ostore8BD
B
	full_text5
3
1store double %696, double* %54, align 8, !tbaa !8
,double8B

	full_text

double %696
-double*8B

	full_text

double* %54
Pstore8BE
C
	full_text6
4
2store double %695, double* %59, align 16, !tbaa !8
,double8B

	full_text

double %695
-double*8B

	full_text

double* %59
Ostore8BD
B
	full_text5
3
1store double %694, double* %64, align 8, !tbaa !8
,double8B

	full_text

double %694
-double*8B

	full_text

double* %64
Pstore8BE
C
	full_text6
4
2store double %693, double* %69, align 16, !tbaa !8
,double8B

	full_text

double %693
-double*8B

	full_text

double* %69
Jstore8B?
=
	full_text0
.
,store i64 %733, i64* %75, align 16, !tbaa !8
&i648B

	full_text


i64 %733
'i64*8B

	full_text


i64* %75
Istore8B>
<
	full_text/
-
+store i64 %736, i64* %80, align 8, !tbaa !8
&i648B

	full_text


i64 %736
'i64*8B

	full_text


i64* %80
Jstore8B?
=
	full_text0
.
,store i64 %739, i64* %85, align 16, !tbaa !8
&i648B

	full_text


i64 %739
'i64*8B

	full_text


i64* %85
Istore8B>
<
	full_text/
-
+store i64 %742, i64* %90, align 8, !tbaa !8
&i648B

	full_text


i64 %742
'i64*8B

	full_text


i64* %90
Jstore8B?
=
	full_text0
.
,store i64 %745, i64* %95, align 16, !tbaa !8
&i648B

	full_text


i64 %745
'i64*8B

	full_text


i64* %95
ègetelementptr8B|
z
	full_textm
k
i%882 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %34, i64 %43, i64 %730, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %34
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%883 = load double, double* %882, align 8, !tbaa !8
.double*8B

	full_text

double* %882
ègetelementptr8B|
z
	full_textm
k
i%884 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %35, i64 %43, i64 %730, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %35
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%885 = load double, double* %884, align 8, !tbaa !8
.double*8B

	full_text

double* %884
ègetelementptr8B|
z
	full_textm
k
i%886 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %36, i64 %43, i64 %730, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %36
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%887 = load double, double* %886, align 8, !tbaa !8
.double*8B

	full_text

double* %886
ègetelementptr8B|
z
	full_textm
k
i%888 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %37, i64 %43, i64 %730, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %37
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%889 = load double, double* %888, align 8, !tbaa !8
.double*8B

	full_text

double* %888
ègetelementptr8B|
z
	full_textm
k
i%890 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %38, i64 %43, i64 %730, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %38
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%891 = load double, double* %890, align 8, !tbaa !8
.double*8B

	full_text

double* %890
ègetelementptr8B|
z
	full_textm
k
i%892 = getelementptr inbounds [65 x [65 x double]], [65 x [65 x double]]* %39, i64 %43, i64 %730, i64 %45
I[65 x [65 x double]]*8B,
*
	full_text

[65 x [65 x double]]* %39
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %730
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%893 = load double, double* %892, align 8, !tbaa !8
.double*8B

	full_text

double* %892
•getelementptr8Bë
é
	full_textÄ
~
|%894 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %747, i64 %45, i64 0
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%895 = load double, double* %894, align 8, !tbaa !8
.double*8B

	full_text

double* %894
Abitcast8B4
2
	full_text%
#
!%896 = bitcast i64 %733 to double
&i648B

	full_text


i64 %733
vcall8Bl
j
	full_text]
[
Y%897 = tail call double @llvm.fmuladd.f64(double %697, double -2.000000e+00, double %896)
,double8B

	full_text

double %697
,double8B

	full_text

double %896
:fadd8B0
.
	full_text!

%898 = fadd double %897, %703
,double8B

	full_text

double %897
,double8B

	full_text

double %703
{call8Bq
o
	full_textb
`
^%899 = tail call double @llvm.fmuladd.f64(double %898, double 0x40A7418000000001, double %895)
,double8B

	full_text

double %898
,double8B

	full_text

double %895
Abitcast8B4
2
	full_text%
#
!%900 = bitcast i64 %739 to double
&i648B

	full_text


i64 %739
:fsub8B0
.
	full_text!

%901 = fsub double %900, %701
,double8B

	full_text

double %900
,double8B

	full_text

double %701
vcall8Bl
j
	full_text]
[
Y%902 = tail call double @llvm.fmuladd.f64(double %901, double -3.150000e+01, double %899)
,double8B

	full_text

double %901
,double8B

	full_text

double %899
•getelementptr8Bë
é
	full_textÄ
~
|%903 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %747, i64 %45, i64 1
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%904 = load double, double* %903, align 8, !tbaa !8
.double*8B

	full_text

double* %903
Abitcast8B4
2
	full_text%
#
!%905 = bitcast i64 %736 to double
&i648B

	full_text


i64 %736
vcall8Bl
j
	full_text]
[
Y%906 = tail call double @llvm.fmuladd.f64(double %696, double -2.000000e+00, double %905)
,double8B

	full_text

double %696
,double8B

	full_text

double %905
:fadd8B0
.
	full_text!

%907 = fadd double %906, %702
,double8B

	full_text

double %906
,double8B

	full_text

double %702
{call8Bq
o
	full_textb
`
^%908 = tail call double @llvm.fmuladd.f64(double %907, double 0x40A7418000000001, double %904)
,double8B

	full_text

double %907
,double8B

	full_text

double %904
vcall8Bl
j
	full_text]
[
Y%909 = tail call double @llvm.fmuladd.f64(double %749, double -2.000000e+00, double %883)
,double8B

	full_text

double %749
,double8B

	full_text

double %883
:fadd8B0
.
	full_text!

%910 = fadd double %724, %909
,double8B

	full_text

double %724
,double8B

	full_text

double %909
{call8Bq
o
	full_textb
`
^%911 = tail call double @llvm.fmuladd.f64(double %910, double 0x4078CE6666666667, double %908)
,double8B

	full_text

double %910
,double8B

	full_text

double %908
:fmul8B0
.
	full_text!

%912 = fmul double %722, %702
,double8B

	full_text

double %722
,double8B

	full_text

double %702
Cfsub8B9
7
	full_text*
(
&%913 = fsub double -0.000000e+00, %912
,double8B

	full_text

double %912
mcall8Bc
a
	full_textT
R
P%914 = tail call double @llvm.fmuladd.f64(double %905, double %885, double %913)
,double8B

	full_text

double %905
,double8B

	full_text

double %885
,double8B

	full_text

double %913
vcall8Bl
j
	full_text]
[
Y%915 = tail call double @llvm.fmuladd.f64(double %914, double -3.150000e+01, double %911)
,double8B

	full_text

double %914
,double8B

	full_text

double %911
•getelementptr8Bë
é
	full_textÄ
~
|%916 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %747, i64 %45, i64 2
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%917 = load double, double* %916, align 8, !tbaa !8
.double*8B

	full_text

double* %916
vcall8Bl
j
	full_text]
[
Y%918 = tail call double @llvm.fmuladd.f64(double %695, double -2.000000e+00, double %900)
,double8B

	full_text

double %695
,double8B

	full_text

double %900
:fadd8B0
.
	full_text!

%919 = fadd double %701, %918
,double8B

	full_text

double %701
,double8B

	full_text

double %918
{call8Bq
o
	full_textb
`
^%920 = tail call double @llvm.fmuladd.f64(double %919, double 0x40A7418000000001, double %917)
,double8B

	full_text

double %919
,double8B

	full_text

double %917
vcall8Bl
j
	full_text]
[
Y%921 = tail call double @llvm.fmuladd.f64(double %751, double -2.000000e+00, double %885)
,double8B

	full_text

double %751
,double8B

	full_text

double %885
:fadd8B0
.
	full_text!

%922 = fadd double %722, %921
,double8B

	full_text

double %722
,double8B

	full_text

double %921
ucall8Bk
i
	full_text\
Z
X%923 = tail call double @llvm.fmuladd.f64(double %922, double 5.292000e+02, double %920)
,double8B

	full_text

double %922
,double8B

	full_text

double %920
:fmul8B0
.
	full_text!

%924 = fmul double %722, %701
,double8B

	full_text

double %722
,double8B

	full_text

double %701
Cfsub8B9
7
	full_text*
(
&%925 = fsub double -0.000000e+00, %924
,double8B

	full_text

double %924
mcall8Bc
a
	full_textT
R
P%926 = tail call double @llvm.fmuladd.f64(double %900, double %885, double %925)
,double8B

	full_text

double %900
,double8B

	full_text

double %885
,double8B

	full_text

double %925
Abitcast8B4
2
	full_text%
#
!%927 = bitcast i64 %745 to double
&i648B

	full_text


i64 %745
:fsub8B0
.
	full_text!

%928 = fsub double %927, %893
,double8B

	full_text

double %927
,double8B

	full_text

double %893
:fsub8B0
.
	full_text!

%929 = fsub double %928, %698
,double8B

	full_text

double %928
,double8B

	full_text

double %698
:fadd8B0
.
	full_text!

%930 = fadd double %721, %929
,double8B

	full_text

double %721
,double8B

	full_text

double %929
ucall8Bk
i
	full_text\
Z
X%931 = tail call double @llvm.fmuladd.f64(double %930, double 4.000000e-01, double %926)
,double8B

	full_text

double %930
,double8B

	full_text

double %926
vcall8Bl
j
	full_text]
[
Y%932 = tail call double @llvm.fmuladd.f64(double %931, double -3.150000e+01, double %923)
,double8B

	full_text

double %931
,double8B

	full_text

double %923
•getelementptr8Bë
é
	full_textÄ
~
|%933 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %747, i64 %45, i64 3
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%934 = load double, double* %933, align 8, !tbaa !8
.double*8B

	full_text

double* %933
Abitcast8B4
2
	full_text%
#
!%935 = bitcast i64 %742 to double
&i648B

	full_text


i64 %742
vcall8Bl
j
	full_text]
[
Y%936 = tail call double @llvm.fmuladd.f64(double %694, double -2.000000e+00, double %935)
,double8B

	full_text

double %694
,double8B

	full_text

double %935
:fadd8B0
.
	full_text!

%937 = fadd double %936, %700
,double8B

	full_text

double %936
,double8B

	full_text

double %700
{call8Bq
o
	full_textb
`
^%938 = tail call double @llvm.fmuladd.f64(double %937, double 0x40A7418000000001, double %934)
,double8B

	full_text

double %937
,double8B

	full_text

double %934
vcall8Bl
j
	full_text]
[
Y%939 = tail call double @llvm.fmuladd.f64(double %753, double -2.000000e+00, double %887)
,double8B

	full_text

double %753
,double8B

	full_text

double %887
:fadd8B0
.
	full_text!

%940 = fadd double %715, %939
,double8B

	full_text

double %715
,double8B

	full_text

double %939
{call8Bq
o
	full_textb
`
^%941 = tail call double @llvm.fmuladd.f64(double %940, double 0x4078CE6666666667, double %938)
,double8B

	full_text

double %940
,double8B

	full_text

double %938
:fmul8B0
.
	full_text!

%942 = fmul double %722, %700
,double8B

	full_text

double %722
,double8B

	full_text

double %700
Cfsub8B9
7
	full_text*
(
&%943 = fsub double -0.000000e+00, %942
,double8B

	full_text

double %942
mcall8Bc
a
	full_textT
R
P%944 = tail call double @llvm.fmuladd.f64(double %935, double %885, double %943)
,double8B

	full_text

double %935
,double8B

	full_text

double %885
,double8B

	full_text

double %943
vcall8Bl
j
	full_text]
[
Y%945 = tail call double @llvm.fmuladd.f64(double %944, double -3.150000e+01, double %941)
,double8B

	full_text

double %944
,double8B

	full_text

double %941
•getelementptr8Bë
é
	full_textÄ
~
|%946 = getelementptr inbounds [65 x [65 x [5 x double]]], [65 x [65 x [5 x double]]]* %41, i64 %43, i64 %747, i64 %45, i64 4
U[65 x [65 x [5 x double]]]*8B2
0
	full_text#
!
[65 x [65 x [5 x double]]]* %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %747
%i648B

	full_text
	
i64 %45
Pload8BF
D
	full_text7
5
3%947 = load double, double* %946, align 8, !tbaa !8
.double*8B

	full_text

double* %946
vcall8Bl
j
	full_text]
[
Y%948 = tail call double @llvm.fmuladd.f64(double %693, double -2.000000e+00, double %927)
,double8B

	full_text

double %693
,double8B

	full_text

double %927
:fadd8B0
.
	full_text!

%949 = fadd double %698, %948
,double8B

	full_text

double %698
,double8B

	full_text

double %948
{call8Bq
o
	full_textb
`
^%950 = tail call double @llvm.fmuladd.f64(double %949, double 0x40A7418000000001, double %947)
,double8B

	full_text

double %949
,double8B

	full_text

double %947
vcall8Bl
j
	full_text]
[
Y%951 = tail call double @llvm.fmuladd.f64(double %755, double -2.000000e+00, double %889)
,double8B

	full_text

double %755
,double8B

	full_text

double %889
:fadd8B0
.
	full_text!

%952 = fadd double %717, %951
,double8B

	full_text

double %717
,double8B

	full_text

double %951
{call8Bq
o
	full_textb
`
^%953 = tail call double @llvm.fmuladd.f64(double %952, double 0xC077D0624DD2F1A9, double %950)
,double8B

	full_text

double %952
,double8B

	full_text

double %950
Bfmul8B8
6
	full_text)
'
%%954 = fmul double %751, 2.000000e+00
,double8B

	full_text

double %751
:fmul8B0
.
	full_text!

%955 = fmul double %751, %954
,double8B

	full_text

double %751
,double8B

	full_text

double %954
Cfsub8B9
7
	full_text*
(
&%956 = fsub double -0.000000e+00, %955
,double8B

	full_text

double %955
mcall8Bc
a
	full_textT
R
P%957 = tail call double @llvm.fmuladd.f64(double %885, double %885, double %956)
,double8B

	full_text

double %885
,double8B

	full_text

double %885
,double8B

	full_text

double %956
mcall8Bc
a
	full_textT
R
P%958 = tail call double @llvm.fmuladd.f64(double %722, double %722, double %957)
,double8B

	full_text

double %722
,double8B

	full_text

double %722
,double8B

	full_text

double %957
ucall8Bk
i
	full_text\
Z
X%959 = tail call double @llvm.fmuladd.f64(double %958, double 6.615000e+01, double %953)
,double8B

	full_text

double %958
,double8B

	full_text

double %953
Bfmul8B8
6
	full_text)
'
%%960 = fmul double %693, 2.000000e+00
,double8B

	full_text

double %693
:fmul8B0
.
	full_text!

%961 = fmul double %757, %960
,double8B

	full_text

double %757
,double8B

	full_text

double %960
Cfsub8B9
7
	full_text*
(
&%962 = fsub double -0.000000e+00, %961
,double8B

	full_text

double %961
mcall8Bc
a
	full_textT
R
P%963 = tail call double @llvm.fmuladd.f64(double %927, double %891, double %962)
,double8B

	full_text

double %927
,double8B

	full_text

double %891
,double8B

	full_text

double %962
mcall8Bc
a
	full_textT
R
P%964 = tail call double @llvm.fmuladd.f64(double %698, double %719, double %963)
,double8B

	full_text

double %698
,double8B

	full_text

double %719
,double8B

	full_text

double %963
{call8Bq
o
	full_textb
`
^%965 = tail call double @llvm.fmuladd.f64(double %964, double 0x40884F645A1CAC08, double %959)
,double8B

	full_text

double %964
,double8B

	full_text

double %959
Bfmul8B8
6
	full_text)
'
%%966 = fmul double %893, 4.000000e-01
,double8B

	full_text

double %893
Cfsub8B9
7
	full_text*
(
&%967 = fsub double -0.000000e+00, %966
,double8B

	full_text

double %966
ucall8Bk
i
	full_text\
Z
X%968 = tail call double @llvm.fmuladd.f64(double %927, double 1.400000e+00, double %967)
,double8B

	full_text

double %927
,double8B

	full_text

double %967
Bfmul8B8
6
	full_text)
'
%%969 = fmul double %721, 4.000000e-01
,double8B

	full_text

double %721
Cfsub8B9
7
	full_text*
(
&%970 = fsub double -0.000000e+00, %969
,double8B

	full_text

double %969
ucall8Bk
i
	full_text\
Z
X%971 = tail call double @llvm.fmuladd.f64(double %698, double 1.400000e+00, double %970)
,double8B

	full_text

double %698
,double8B

	full_text

double %970
:fmul8B0
.
	full_text!

%972 = fmul double %722, %971
,double8B

	full_text

double %722
,double8B

	full_text

double %971
Cfsub8B9
7
	full_text*
(
&%973 = fsub double -0.000000e+00, %972
,double8B

	full_text

double %972
mcall8Bc
a
	full_textT
R
P%974 = tail call double @llvm.fmuladd.f64(double %968, double %885, double %973)
,double8B

	full_text

double %968
,double8B

	full_text

double %885
,double8B

	full_text

double %973
vcall8Bl
j
	full_text]
[
Y%975 = tail call double @llvm.fmuladd.f64(double %974, double -3.150000e+01, double %965)
,double8B

	full_text

double %974
,double8B

	full_text

double %965
Pload8BF
D
	full_text7
5
3%976 = load double, double* %692, align 8, !tbaa !8
.double*8B

	full_text

double* %692
Qload8BG
E
	full_text8
6
4%977 = load double, double* %144, align 16, !tbaa !8
.double*8B

	full_text

double* %144
vcall8Bl
j
	full_text]
[
Y%978 = tail call double @llvm.fmuladd.f64(double %977, double -4.000000e+00, double %976)
,double8B

	full_text

double %977
,double8B

	full_text

double %976
Pload8BF
D
	full_text7
5
3%979 = load double, double* %49, align 16, !tbaa !8
-double*8B

	full_text

double* %49
ucall8Bk
i
	full_text\
Z
X%980 = tail call double @llvm.fmuladd.f64(double %979, double 5.000000e+00, double %978)
,double8B

	full_text

double %979
,double8B

	full_text

double %978
mcall8Bc
a
	full_textT
R
P%981 = tail call double @llvm.fmuladd.f64(double %290, double %980, double %902)
,double8B

	full_text

double %290
,double8B

	full_text

double %980
,double8B

	full_text

double %902
Pstore8BE
C
	full_text6
4
2store double %981, double* %894, align 8, !tbaa !8
,double8B

	full_text

double %981
.double*8B

	full_text

double* %894
Pload8BF
D
	full_text7
5
3%982 = load double, double* %151, align 8, !tbaa !8
.double*8B

	full_text

double* %151
Pload8BF
D
	full_text7
5
3%983 = load double, double* %148, align 8, !tbaa !8
.double*8B

	full_text

double* %148
vcall8Bl
j
	full_text]
[
Y%984 = tail call double @llvm.fmuladd.f64(double %983, double -4.000000e+00, double %982)
,double8B

	full_text

double %983
,double8B

	full_text

double %982
Oload8BE
C
	full_text6
4
2%985 = load double, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
ucall8Bk
i
	full_text\
Z
X%986 = tail call double @llvm.fmuladd.f64(double %985, double 5.000000e+00, double %984)
,double8B

	full_text

double %985
,double8B

	full_text

double %984
mcall8Bc
a
	full_textT
R
P%987 = tail call double @llvm.fmuladd.f64(double %290, double %986, double %915)
,double8B

	full_text

double %290
,double8B

	full_text

double %986
,double8B

	full_text

double %915
Pstore8BE
C
	full_text6
4
2store double %987, double* %903, align 8, !tbaa !8
,double8B

	full_text

double %987
.double*8B

	full_text

double* %903
Qload8BG
E
	full_text8
6
4%988 = load double, double* %156, align 16, !tbaa !8
.double*8B

	full_text

double* %156
Qload8BG
E
	full_text8
6
4%989 = load double, double* %153, align 16, !tbaa !8
.double*8B

	full_text

double* %153
vcall8Bl
j
	full_text]
[
Y%990 = tail call double @llvm.fmuladd.f64(double %989, double -4.000000e+00, double %988)
,double8B

	full_text

double %989
,double8B

	full_text

double %988
Pload8BF
D
	full_text7
5
3%991 = load double, double* %59, align 16, !tbaa !8
-double*8B

	full_text

double* %59
ucall8Bk
i
	full_text\
Z
X%992 = tail call double @llvm.fmuladd.f64(double %991, double 5.000000e+00, double %990)
,double8B

	full_text

double %991
,double8B

	full_text

double %990
mcall8Bc
a
	full_textT
R
P%993 = tail call double @llvm.fmuladd.f64(double %290, double %992, double %932)
,double8B

	full_text

double %290
,double8B

	full_text

double %992
,double8B

	full_text

double %932
Pstore8BE
C
	full_text6
4
2store double %993, double* %916, align 8, !tbaa !8
,double8B

	full_text

double %993
.double*8B

	full_text

double* %916
Pload8BF
D
	full_text7
5
3%994 = load double, double* %161, align 8, !tbaa !8
.double*8B

	full_text

double* %161
Pload8BF
D
	full_text7
5
3%995 = load double, double* %158, align 8, !tbaa !8
.double*8B

	full_text

double* %158
vcall8Bl
j
	full_text]
[
Y%996 = tail call double @llvm.fmuladd.f64(double %995, double -4.000000e+00, double %994)
,double8B

	full_text

double %995
,double8B

	full_text

double %994
Oload8BE
C
	full_text6
4
2%997 = load double, double* %64, align 8, !tbaa !8
-double*8B

	full_text

double* %64
ucall8Bk
i
	full_text\
Z
X%998 = tail call double @llvm.fmuladd.f64(double %997, double 5.000000e+00, double %996)
,double8B

	full_text

double %997
,double8B

	full_text

double %996
mcall8Bc
a
	full_textT
R
P%999 = tail call double @llvm.fmuladd.f64(double %290, double %998, double %945)
,double8B

	full_text

double %290
,double8B

	full_text

double %998
,double8B

	full_text

double %945
Pstore8BE
C
	full_text6
4
2store double %999, double* %933, align 8, !tbaa !8
,double8B

	full_text

double %999
.double*8B

	full_text

double* %933
Rload8BH
F
	full_text9
7
5%1000 = load double, double* %166, align 16, !tbaa !8
.double*8B

	full_text

double* %166
Rload8BH
F
	full_text9
7
5%1001 = load double, double* %163, align 16, !tbaa !8
.double*8B

	full_text

double* %163
ycall8Bo
m
	full_text`
^
\%1002 = tail call double @llvm.fmuladd.f64(double %1001, double -4.000000e+00, double %1000)
-double8B

	full_text

double %1001
-double8B

	full_text

double %1000
Qload8BG
E
	full_text8
6
4%1003 = load double, double* %69, align 16, !tbaa !8
-double*8B

	full_text

double* %69
xcall8Bn
l
	full_text_
]
[%1004 = tail call double @llvm.fmuladd.f64(double %1003, double 5.000000e+00, double %1002)
-double8B

	full_text

double %1003
-double8B

	full_text

double %1002
ocall8Be
c
	full_textV
T
R%1005 = tail call double @llvm.fmuladd.f64(double %290, double %1004, double %975)
,double8B

	full_text

double %290
-double8B

	full_text

double %1004
,double8B

	full_text

double %975
Qstore8BF
D
	full_text7
5
3store double %1005, double* %946, align 8, !tbaa !8
-double8B

	full_text

double %1005
.double*8B

	full_text

double* %946
)br8B!

	full_text

br label %1006
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
%i328	B

	full_text
	
i32 %10
$i328	B

	full_text


i32 %8
,double*8	B

	full_text


double* %6
,double*8	B

	full_text


double* %0
,double*8	B

	full_text


double* %5
,double*8	B

	full_text


double* %2
$i328	B

	full_text


i32 %9
,double*8	B

	full_text


double* %3
,double*8	B

	full_text


double* %7
,double*8	B

	full_text


double* %1
,double*8	B
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
$i328	B

	full_text


i32 -2
5double8	B'
%
	full_text

double -3.150000e+01
4double8	B&
$
	full_text

double 5.292000e+02
$i648	B

	full_text


i64 40
#i328	B

	full_text	

i32 0
#i648	B

	full_text	

i64 2
4double8	B&
$
	full_text

double 1.400000e+00
:double8	B,
*
	full_text

double 0x40A7418000000001
5double8	B'
%
	full_text

double -4.000000e+00
#i328	B

	full_text	

i32 7
4double8	B&
$
	full_text

double 7.500000e-01
:double8	B,
*
	full_text

double 0xC077D0624DD2F1A9
#i328	B

	full_text	

i32 1
#i648	B

	full_text	

i64 4
4double8	B&
$
	full_text

double 4.000000e-01
$i328	B

	full_text


i32 -1
#i648	B

	full_text	

i64 3
4double8	B&
$
	full_text

double 2.000000e+00
4double8	B&
$
	full_text

double 5.000000e+00
%i18	B

	full_text


i1 false
:double8	B,
*
	full_text

double 0x4078CE6666666667
4double8	B&
$
	full_text

double 2.500000e-01
4double8	B&
$
	full_text

double 6.615000e+01
4double8	B&
$
	full_text

double 1.000000e+00
4double8	B&
$
	full_text

double 6.000000e+00
!i88	B

	full_text

i8 0
:double8	B,
*
	full_text

double 0x40884F645A1CAC08
#i648	B

	full_text	

i64 0
5double8	B'
%
	full_text

double -2.000000e+00
$i648	B

	full_text


i64 32
#i648	B

	full_text	

i64 1
4double8	B&
$
	full_text

double 4.000000e+00
$i328	B

	full_text


i32 -3
5double8	B'
%
	full_text

double -0.000000e+00       	  
 

                       !! "# "" $$ %& %' %% () (+ ** ,, -. -/ -- 01 02 33 44 55 66 77 88 99 :; :: <= << >? >> @A @@ BC BD BE BB FG FF HI HH JK JJ LM LL NO NP NN QR QS QT QQ UV UU WX WW YZ YY [\ [[ ]^ ]_ ]] `a `b `c `` de dd fg ff hi hh jk jj lm ln ll op oq or oo st ss uv uu wx ww yz yy {| {} {{ ~ ~	Ä ~	Å ~~ ÇÉ ÇÇ ÑÖ ÑÑ Üá ÜÜ àâ àà äã ä
å ää çé ç
è ç
ê çç ëí ëë ìî ìì ïñ ïï óò óó ôö ô
õ ôô úù ú
û ú
ü úú †° †† ¢£ ¢¢ §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´
≠ ´
Æ ´´ Ø∞ ØØ ±≤ ±± ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫
Ω ∫∫ æø ææ ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …
Ã …… ÕŒ ÕÕ œ– œœ —“ —— ”‘ ”” ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂ
‚ ﬂﬂ „‰ „„ ÂÊ ÂÂ ÁË ÁÁ ÈÍ ÈÈ ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜
˙ ˜˜ ˚¸ ˚˚ ˝˛ ˝˝ ˇÄ ˇˇ ÅÇ ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä ââ ãå ãã çé çç èê è
ë è
í èè ìî ìì ïñ ï
ó ï
ò ïï ôö ôô õú õ
ù õ
û õõ ü† üü °¢ °
£ °
§ °° •¶ •• ß® ß
© ß
™ ßß ´¨ ´´ ≠Æ ≠
Ø ≠
∞ ≠≠ ±≤ ±± ≥¥ ≥
µ ≥
∂ ≥≥ ∑∏ ∑∑ π∫ π
ª π
º ππ Ωæ ΩΩ ø¿ ø
¡ ø
¬ øø √ƒ √√ ≈∆ ≈
« ≈
» ≈≈ …  …… ÀÃ À
Õ À
Œ ÀÀ œ– œœ —“ —
” —
‘ —— ’÷ ’’ ◊ÿ ◊◊ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›› ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰‰ ÊÁ ÊÊ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô ÔÔ ÒÚ ÒÒ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜˜ ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ ÄÄ ÇÉ ÇÇ ÑÖ ÑÑ Üá Ü
à ÜÜ âä ââ ãå ãã çé çç èê èè ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √
∆ √√ «» «« …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– Œ
— ŒŒ “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ Ÿ
‹ ŸŸ ›ﬁ ›› ﬂ‡ ﬂﬂ ·‚ ·
„ ·· ‰Â ‰
Ê ‰
Á ‰‰ ËÈ ËË ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò Ô
Ú ÔÔ ÛÙ ÛÛ ıˆ ıı ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛˛ ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ ÑÑ Üá Ü
à Ü
â ÜÜ äã ää åç å
é å
è åå êë êê íì í
î í
ï íí ñó ññ òô ò
ö ò
õ òò úù úú ûü û
† û
° ûû ¢£ ¢¢ §• §§ ¶ß ¶¶ ®© ®
™ ®® ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ Ω
ø Ω
¿ ΩΩ ¡¬ ¡¡ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁ
ﬂ ﬁﬁ ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰
Ê ‰‰ ÁË Á
È Á
Í ÁÁ ÎÏ ÎÎ ÌÓ ÌÌ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É ÅÅ Ñ
Ö ÑÑ Üá Ü
à Ü
â ÜÜ äã ää åç å
é åå èê èè ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ù
† ùù °¢ °° £§ ££ •¶ •• ß® ß
© ßß ™´ ™™ ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æ
ø ææ ¿¡ ¿
¬ ¿
√ ¿¿ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «« ÀÃ ÀÀ ÕŒ ÕÕ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ Ê
Á ÊÊ ËÈ Ë
Í Ë
Î ËË ÏÌ Ï
Ó Ï
Ô ÏÏ Ò 
Ú  ÛÙ ÛÛ ıˆ ı
˜ ıı ¯
˘ ¯¯ ˙˚ ˙
¸ ˙
˝ ˙˙ ˛ˇ ˛
Ä ˛
Å ˛˛ ÇÉ Ç
Ñ ÇÇ ÖÜ ÖÖ á
à áá âä â
ã ââ åç åå é
è éé êë ê
í êê ìî ì
ï ìì ñ
ó ññ òô ò
ö ò
õ òò úù ú
û úú üü †
° †† ¢£ ¢¢ §
• §§ ¶ß ¶¶ ®© ®® ™´ ™™ ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ ºº ø¿ øø ¡¬ ¡¡ √ƒ √√ ≈
∆ ≈≈ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œœ ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿÿ ⁄€ ⁄⁄ ‹
› ‹‹ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô ÔÔ ÒÚ ÒÒ Û
Ù ÛÛ ıˆ ı
˜ ıı ¯˘ ¯¯ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝
Ä ˝˝ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Üá ÜÜ à
â àà äã ä
å ää çé çç èê è
ë èè íì í
î í
ï íí ñó ñ
ò ññ ôö ô
õ ôô úù ú
û úú ü† ü
° üü ¢£ ¢
§ ¢¢ •¶ •
ß •• ®© ®
™ ®® ´¨ ´
≠ ´´ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±± ¥µ ¥
∂ ¥¥ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ ŸŸ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡
„ ‡‡ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆ
˘ ˆˆ ˙˚ ˙˙ ¸˝ ¸¸ ˛ˇ ˛
Ä ˛˛ ÅÇ Å
É Å
Ñ ÅÅ ÖÜ ÖÖ áà áá âä â
ã ââ åç å
é å
è åå êë êê íì í
î í
ï íí ñó ññ òô ò
ö ò
õ òò úù úú ûü û
† û
° ûû ¢£ ¢¢ §• §
¶ §
ß §§ ®© ®® ™´ ™
¨ ™
≠ ™™ ÆØ ÆÆ ∞± ∞
≤ ∞
≥ ∞∞ ¥µ ¥¥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡¡ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …
Ã …… ÕŒ ÕÕ œ– œœ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ Ê
Á ÊÊ ËÈ Ë
Í Ë
Î ËË ÏÌ Ï
Ó ÏÏ Ô Ô
Ò Ô
Ú ÔÔ ÛÙ ÛÛ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä	 ˛˛ Å	Ç	 Å	
É	 Å	Å	 Ñ	Ö	 Ñ	
Ü	 Ñ	Ñ	 á	à	 á	
â	 á	á	 ä	
ã	 ä	ä	 å	ç	 å	
é	 å	
è	 å	å	 ê	ë	 ê	ê	 í	ì	 í	
î	 í	í	 ï	ñ	 ï	ï	 ó	ò	 ó	
ô	 ó	ó	 ö	õ	 ö	
ú	 ö	ö	 ù	û	 ù	
ü	 ù	ù	 †	°	 †	
¢	 †	†	 £	§	 £	
•	 £	
¶	 £	£	 ß	®	 ß	ß	 ©	™	 ©	©	 ´	¨	 ´	
≠	 ´	´	 Æ	Ø	 Æ	
∞	 Æ	Æ	 ±	≤	 ±	
≥	 ±	±	 ¥	µ	 ¥	
∂	 ¥	¥	 ∑	∏	 ∑	
π	 ∑	∑	 ∫	ª	 ∫	
º	 ∫	∫	 Ω	æ	 Ω	
ø	 Ω	Ω	 ¿	
¡	 ¿	¿	 ¬	√	 ¬	
ƒ	 ¬	
≈	 ¬	¬	 ∆	«	 ∆	
»	 ∆	∆	 …	 	 …	
À	 …	
Ã	 …	…	 Õ	Œ	 Õ	Õ	 œ	–	 œ	
—	 œ	œ	 “	”	 “	
‘	 “	“	 ’	÷	 ’	
◊	 ’	’	 ÿ	Ÿ	 ÿ	
⁄	 ÿ	ÿ	 €	‹	 €	
›	 €	€	 ﬁ	ﬂ	 ﬁ	
‡	 ﬁ	ﬁ	 ·	‚	 ·	·	 „	‰	 „	
Â	 „	„	 Ê	
Á	 Ê	Ê	 Ë	È	 Ë	
Í	 Ë	
Î	 Ë	Ë	 Ï	Ì	 Ï	
Ó	 Ï	
Ô	 Ï	Ï	 	Ò	 	
Ú	 		 Û	Ù	 Û	Û	 ı	ˆ	 ı	
˜	 ı	ı	 ¯	
˘	 ¯	¯	 ˙	˚	 ˙	
¸	 ˙	
˝	 ˙	˙	 ˛	ˇ	 ˛	
Ä
 ˛	
Å
 ˛	˛	 Ç
É
 Ç

Ñ
 Ç
Ç
 Ö
Ü
 Ö
Ö
 á

à
 á
á
 â
ä
 â

ã
 â
â
 å
ç
 å
å
 é

è
 é
é
 ê
ë
 ê

í
 ê
ê
 ì
î
 ì

ï
 ì
ì
 ñ

ó
 ñ
ñ
 ò
ô
 ò

ö
 ò

õ
 ò
ò
 ú
ù
 ú

û
 ú
ú
 ü
†
 ü
ü
 °
¢
 °
°
 £
§
 £
£
 •
¶
 •

ß
 •
•
 ®
©
 ®
®
 ™
´
 ™

¨
 ™
™
 ≠
Æ
 ≠
≠
 Ø
∞
 Ø

±
 Ø
Ø
 ≤
≥
 ≤

¥
 ≤

µ
 ≤
≤
 ∂
∑
 ∂

∏
 ∂
∂
 π
∫
 π
π
 ª
º
 ª
ª
 Ω
æ
 Ω
Ω
 ø
¿
 ø

¡
 ø
ø
 ¬
√
 ¬
¬
 ƒ
≈
 ƒ

∆
 ƒ
ƒ
 «
»
 «
«
 …
 
 …

À
 …
…
 Ã
Õ
 Ã

Œ
 Ã

œ
 Ã
Ã
 –
—
 –

“
 –
–
 ”
‘
 ”
”
 ’
÷
 ’
’
 ◊
ÿ
 ◊
◊
 Ÿ
⁄
 Ÿ

€
 Ÿ
Ÿ
 ‹
›
 ‹
‹
 ﬁ
ﬂ
 ﬁ

‡
 ﬁ
ﬁ
 ·
‚
 ·
·
 „
‰
 „

Â
 „
„
 Ê
Á
 Ê

Ë
 Ê

È
 Ê
Ê
 Í
Î
 Í

Ï
 Í
Í
 Ì
Ó
 Ì
Ì
 Ô

 Ô
Ô
 Ò
Ú
 Ò
Ò
 Û
Ù
 Û

ı
 Û
Û
 ˆ
˜
 ˆ
ˆ
 ¯
˘
 ¯

˙
 ¯
¯
 ˚
¸
 ˚
˚
 ˝
˛
 ˝

ˇ
 ˝
˝
 ÄÅ Ä
Ç Ä
É ÄÄ ÑÖ Ñ
Ü ÑÑ áà áá âä ââ ãå ã
ç ãã éè éé êë ê
í êê ìî ìì ïñ ï
ó ïï òô ò
ö ò
õ òò úù ú
û úú üü †° †† ¢£ ¢¢ §• §§ ¶ß ¶¶ ®© ®® ™´ ™™ ¨≠ ¨¨ ÆÆ Ø∞ Ø≤ ±± ≥µ ¥¥ ∂∑ ∂∂ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æ¿ ø
¡ øø ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈≈ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄
‹ ⁄⁄ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ ππ ªº ª
Ω ª
æ ª
ø ªª ¿¡ ¿¿ ¬√ ¬¬ ƒ≈ ƒ
∆ ƒƒ «» «
… «
  «
À «« ÃÕ ÃÃ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”
÷ ”
◊ ”” ÿŸ ÿÿ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂ
‚ ﬂ
„ ﬂﬂ ‰Â ‰‰ ÊÁ ÊÊ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó Î
Ô ÎÎ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘˙ ˘
˚ ˘
¸ ˘
˝ ˘˘ ˛ˇ ˛˛ ÄÅ Ä
Ç Ä
É Ä
Ñ ÄÄ ÖÜ ÖÖ áà á
â á
ä á
ã áá åç åå éè é
ê é
ë é
í éé ìî ìì ïñ ï
ó ï
ò ï
ô ïï öõ öö úù ú
û ú
ü ú
† úú °¢ °° £§ £
• £
¶ £
ß ££ ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª π
º π
Ω ππ æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’
÷ ’’ ◊ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁ
· ﬁ
‚ ﬁﬁ „‰ „„ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙
˚ ˙˙ ¸˝ ¸
˛ ¸
ˇ ¸¸ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ë
ì ë
î ë
ï ëë ñó ññ òô ò
ö òò õú õõ ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø
∞ ØØ ±≤ ±
≥ ±
¥ ±± µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏
º ∏∏ Ωæ ΩΩ ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”” ’÷ ’
◊ ’’ ÿ
Ÿ ÿÿ ⁄€ ⁄
‹ ⁄
› ⁄⁄ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚‚ ÂÊ ÂÂ ÁË Á
È ÁÁ Í
Î ÍÍ ÏÌ Ï
Ó Ï
Ô ÏÏ Ò 
Ú 
Û  Ùı Ù
ˆ ÙÙ ˜¯ ˜˜ ˘
˙ ˘˘ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛˛ Ä
Å ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ à
â àà äã ä
å ä
ç ää éè é
ê éé ëí ëë ìî ì
ï ìì ñó ñ
ò ññ ôö ô
õ ôô úù úú ûü û
† ûû °¢ °
£ °
§ °° •¶ •
ß •• ®© ®® ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ ºº ø¿ øø ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒƒ «» «
… ««  À    ÃÕ Ã
Œ ÃÃ œ– œ
— œ
“ œœ ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·· „‰ „
Â „„ ÊÁ Ê
Ë Ê
È ÊÊ ÍÎ Í
Ï ÍÍ ÌÓ ÌÌ Ô ÔÔ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙˙ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇ
Ç ˇˇ ÉÑ É
Ö ÉÉ Üá Ü
à ÜÜ âä ââ ãå ãã çé çç èê èè ëí ëë ìî ìì ïñ ïï óò óó ôö ôú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂
∏ ∂∂ π∫ π
ª ππ ºΩ º
æ ºº ø¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €€ ﬁﬂ ﬁ
‡ ﬁﬁ ·‚ ·
„ ·· ‰Â ‰
Ê ‰‰ ÁË Á
È ÁÁ ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô ÌÌ Ò 
Ú  ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ àâ à
ä àà ãå ã
ç ãã éè é
ê éé ëí ë
ì ëë îï î
ñ îî óò ó
ô óó öõ ö
ú öö ùû ù
ü ùù †° †
¢ †† £§ £
• ££ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨
Æ ¨¨ Ø∞ Ø
± ØØ ≤≥ ≤
¥ ≤≤ µ∂ µµ ∑∏ ∑
π ∑∑ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆∆ »… »
  »» ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊◊ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËË ÈÍ ÈÈ ÎÏ Î
Ì Î
Ó Î
Ô ÎÎ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜
˙ ˜
˚ ˜˜ ¸˝ ¸¸ ˛ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö É
Ü É
á ÉÉ àâ àà äã ää åç å
é åå èê è
ë è
í è
ì èè îï îî ñó ññ òô ò
ö òò õú õ
ù õ
û õ
ü õõ †° †† ¢£ ¢¢ §• §
¶ §§ ßß ®© ®® ™´ ™
¨ ™
≠ ™
Æ ™™ Ø∞ ØØ ±≤ ±
≥ ±
¥ ±
µ ±± ∂∑ ∂∂ ∏π ∏
∫ ∏
ª ∏
º ∏∏ Ωæ ΩΩ ø¿ ø
¡ ø
¬ ø
√ øø ƒ≈ ƒƒ ∆« ∆
» ∆
… ∆
  ∆∆ ÀÃ ÀÀ ÕŒ Õ
œ Õ
– Õ
— ÕÕ “” ““ ‘’ ‘‘ ÷◊ ÷
ÿ ÷
Ÿ ÷
⁄ ÷÷ €‹ €€ ›ﬁ ›
ﬂ ›› ‡· ‡
‚ ‡‡ „‰ „
Â „„ ÊÁ Ê
Ë ÊÊ ÈÍ È
Î ÈÈ ÏÌ Ï
Ó Ï
Ô Ï
 ÏÏ ÒÚ ÒÒ ÛÙ Û
ı ÛÛ ˆ˜ ˆ
¯ ˆˆ ˘˙ ˘
˚ ˘˘ ¸˝ ¸
˛ ¸¸ ˇÄ ˇ
Å ˇˇ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á ÖÖ à
â àà äã ä
å ä
ç ää éè é
ê éé ëí ë
ì ë
î ë
ï ëë ñó ññ òô ò
ö òò õú õ
ù õõ ûü û
† ûû °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠
Æ ≠≠ Ø∞ Ø
± Ø
≤ ØØ ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø æ
¿ ææ ¡¬ ¡
√ ¡¡ ƒ≈ ƒ
∆ ƒ
« ƒ
» ƒƒ …  …… ÀÃ À
Õ ÀÀ Œœ ŒŒ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚
„ ‚‚ ‰Â ‰
Ê ‰
Á ‰‰ ËÈ Ë
Í ËË ÎÏ Î
Ì Î
Ó Î
Ô ÎÎ Ò  ÚÛ ÚÚ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙˙ ˝˛ ˝
ˇ ˝˝ ÄÅ Ä
Ç ÄÄ ÉÑ É
Ö ÉÉ Üá ÜÜ àâ à
ä àà ã
å ãã çé ç
è ç
ê çç ëí ë
ì ë
î ëë ïñ ï
ó ïï òô òò öõ ö
ú öö ù
û ùù ü† ü
° ü
¢ üü £§ £
• £
¶ ££ ß® ß
© ßß ™´ ™™ ¨
≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥
¥ ≥≥ µ∂ µ
∑ µµ ∏π ∏
∫ ∏∏ ª
º ªª Ωæ Ω
ø Ω
¿ ΩΩ ¡¬ ¡
√ ¡¡ ƒ≈ ƒƒ ∆« ∆∆ »… »
  »» ÀÃ ÀÀ ÕŒ Õ
œ ÕÕ –— –– “” “
‘ ““ ’÷ ’
◊ ’
ÿ ’’ Ÿ⁄ Ÿ
€ ŸŸ ‹› ‹‹ ﬁﬂ ﬁﬁ ‡· ‡
‚ ‡‡ „‰ „„ ÂÊ Â
Á ÂÂ ËÈ ËË ÍÎ Í
Ï ÍÍ ÌÓ Ì
Ô Ì
 ÌÌ ÒÚ Ò
Û ÒÒ Ùı ÙÙ ˆ˜ ˆˆ ¯˘ ¯
˙ ¯¯ ˚¸ ˚˚ ˝˛ ˝
ˇ ˝˝ ÄÅ ÄÄ ÇÉ Ç
Ñ ÇÇ ÖÜ Ö
á Ö
à ÖÖ âä â
ã ââ åç åå éè é
ê éé ëí ëë ìî ì
ï ìì ñó ññ òô ò
ö òò õú õ
ù õ
û õõ ü† ü
° üü ¢£ ¢¢ §• §§ ¶ß ¶
® ¶¶ ©™ ©
´ ©© ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±
≥ ±
¥ ±± µ∂ µ
∑ µµ ∏π ∏∏ ∫ª ∫
º ∫∫ Ωæ Ω
ø ΩΩ ¿¡ ¿
¬ ¿¿ √ƒ √
≈ √√ ∆« ∆
» ∆∆ …  …… ÀÃ À
Õ ÀÀ Œœ Œ
– ŒŒ —“ —
” —— ‘’ ‘
÷ ‘‘ ◊ÿ ◊
Ÿ ◊◊ ⁄€ ⁄⁄ ‹› ‹
ﬁ ‹‹ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á ÂÂ ËÈ Ë
Í ËË ÎÏ Î
Ì ÎÎ ÓÔ Ó
 ÓÓ ÒÚ Ò
Û ÒÒ Ùı Ù
ˆ ÙÙ ˜¯ ˜
˘ ˜˜ ˙˚ ˙
¸ ˙
˝ ˙
˛ ˙˙ ˇÄ ˇˇ ÅÇ Å
É Å
Ñ Å
Ö ÅÅ Üá ÜÜ àâ à
ä à
ã à
å àà çé çç èê è
ë è
í è
ì èè îï îî ñó ñ
ò ñ
ô ñ
ö ññ õú õõ ùû ù
ü ù
† ù
° ùù ¢£ ¢¢ §• §
¶ §
ß §
® §§ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂∑ ∂∂ ∏π ∏
∫ ∏∏ ªº ª
Ω ªª æø æ
¿ æ
¡ æ
¬ ææ √ƒ √√ ≈∆ ≈≈ «» «
… ««  À  
Ã    ÕŒ Õ
œ ÕÕ –— –
“ –– ”‘ ”
’ ”” ÷◊ ÷
ÿ ÷÷ Ÿ⁄ Ÿ
€ ŸŸ ‹
› ‹‹ ﬁﬂ ﬁ
‡ ﬁ
· ﬁﬁ ‚„ ‚
‰ ‚‚ ÂÊ Â
Á Â
Ë Â
È ÂÂ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô Ô
Ò ÔÔ ÚÛ Ú
Ù ÚÚ ıˆ ı
˜ ıı ¯˘ ¯
˙ ¯¯ ˚¸ ˚
˝ ˚˚ ˛ˇ ˛
Ä ˛˛ Å
Ç ÅÅ ÉÑ É
Ö É
Ü ÉÉ áà áá âä â
ã ââ åç å
é åå èê è
ë èè íì í
î íí ïñ ï
ó ïï òô ò
ö ò
õ ò
ú òò ùû ùù ü† üü °¢ °
£ °° §• §
¶ §§ ß® ß
© ßß ™´ ™
¨ ™™ ≠Æ ≠
Ø ≠≠ ∞± ∞
≤ ∞∞ ≥¥ ≥
µ ≥≥ ∂
∑ ∂∂ ∏π ∏
∫ ∏
ª ∏∏ ºΩ º
æ ºº ø¿ ø
¡ ø
¬ ø
√ øø ƒ≈ ƒƒ ∆« ∆
» ∆∆ …  …
À …… ÃÕ Ã
Œ ÃÃ œ– œ
— œœ “” “
‘ ““ ’÷ ’
◊ ’’ ÿŸ ÿÿ ⁄€ ⁄
‹ ⁄⁄ ›
ﬁ ›› ﬂ‡ ﬂ
· ﬂ
‚ ﬂﬂ „‰ „
Â „
Ê „„ ÁË Á
È ÁÁ ÍÎ ÍÍ ÏÌ Ï
Ó ÏÏ Ô
 ÔÔ ÒÚ Ò
Û Ò
Ù ÒÒ ıˆ ı
˜ ı
¯ ıı ˘˙ ˘
˚ ˘˘ ¸˝ ¸¸ ˛
ˇ ˛˛ ÄÅ Ä
Ç ÄÄ ÉÑ ÉÉ Ö
Ü ÖÖ áà á
â áá äã ä
å ää ç
é çç èê è
ë è
í èè ìî ì
ï ìì ñó ññ òô òò öõ ö
ú öö ùû ùù ü† ü
° üü ¢£ ¢
§ ¢
• ¢¢ ¶ß ¶
® ¶¶ ©™ ©© ´¨ ´´ ≠Æ ≠
Ø ≠≠ ∞± ∞∞ ≤≥ ≤
¥ ≤≤ µ∂ µ
∑ µ
∏ µµ π∫ π
ª ππ ºΩ ºº æø ææ ¿¡ ¿
¬ ¿¿ √ƒ √√ ≈∆ ≈
« ≈≈ »… »
  »
À »» ÃÕ Ã
Œ ÃÃ œ– œœ —“ —— ”‘ ”
’ ”” ÷◊ ÷÷ ÿŸ ÿ
⁄ ÿÿ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂ
· ﬂﬂ ‚„ ‚‚ ‰Â ‰‰ ÊÁ Ê
Ë ÊÊ ÈÍ ÈÈ ÎÏ Î
Ì ÎÎ ÓÔ Ó
 Ó
Ò ÓÓ ÚÛ Ú
Ù ÚÚ ı
˜ ˆˆ ¯
˘ ¯¯ ˙
˚ ˙˙ ¸
˝ ¸¸ ˛
ˇ ˛˛ ÄÅ $Ç ,É 7Ñ 8Ö 6Ü 3á üá Æá Ëá ßà 4â 9ä 2ã 5  	 
           ! #$ & '% )" +, .* /- 1 ;: =" ?> A8 C< D@ EB GF I K MH OL P8 R< S@ TQ VU X ZY \W ^[ _8 a< b@ c` ed g ih kf mj n8 p< q@ ro ts v xw zu |y }8 < Ä@ Å~ ÉÇ Ö áÜ âÑ ãà å8 é< è@ êç íë î ñ òì öó õ8 ù< û@ üú °† £ •§ ß¢ ©¶ ™8 ¨< ≠@ Æ´ ∞Ø ≤ ¥≥ ∂± ∏µ π8 ª< º@ Ω∫ øæ ¡ √¬ ≈¿ «ƒ »8  < À@ Ã… ŒÕ – “— ‘8 ÷< ◊@ ÿ’ ⁄Ÿ ‹ ﬁ8 ‡< ·@ ‚ﬂ ‰„ Ê ËÁ Í8 Ï< Ì@ ÓÎ Ô Ú ÙÛ ˆ8 ¯< ˘@ ˙˜ ¸˚ ˛ Äˇ Ç8 Ñ< Ö@ ÜÉ àá ä åã é2 ê< ë@ íè î2 ñ< ó@ òï ö3 ú< ù@ ûõ †3 ¢< £@ §° ¶4 ®< ©@ ™ß ¨4 Æ< Ø@ ∞≠ ≤5 ¥< µ@ ∂≥ ∏5 ∫< ª@ ºπ æ6 ¿< ¡@ ¬ø ƒ6 ∆< «@ »≈  7 Ã< Õ@ ŒÀ –7 “< ”@ ‘— ÷ ÿ ⁄Ÿ ‹ ﬁ€ ‡› · „‚ Â‰ Á ÈË ÎÊ ÌÍ Ó Ô ÚÒ Ù ˆı ¯Û ˙˜ ˚ ˝¸ ˇ˛ Å ÉÇ ÖÄ áÑ à äâ åã é êè íç îë ïH óŸ òW ö‰ õf ùÒ ûu †˛ °Ñ £ã §ì ¶L ß¢ ©[ ™± ¨j ≠¿ Øy ∞œ ≤à ≥€ µó ∂Â ∏¶ πÒ ªµ º˝ æƒ øâ ¡” ¬8 ƒ< ≈@ ∆√ »«  … Ã› Õ8 œ< –@ —Œ ”“ ’‘ ◊È ÿ8 ⁄< €@ ‹Ÿ ﬁ› ‡ﬂ ‚ı „8 Â< Ê@ Á‰ ÈË ÎÍ ÌÅ Ó8 < Ò@ ÚÔ ÙÛ ˆı ¯ç ˘2 ˚< ¸@ ˝˙ ˇ3 Å< Ç@ ÉÄ Ö4 á< à@ âÜ ã5 ç< é@ èå ë6 ì< î@ ïí ó7 ô< ö@ õò ù9 ü< †@ °û £€ •ì ß¶ ©§ ™H ¨® Æ´ Ø≠ ±¢ ≤Ò ¥f ∂≥ ∏µ π∑ ª∞ º9 æ< ø@ ¿Ω ¬Â ƒ¢ ∆≈ »√ …W À« Õ  ŒÃ –¡ —ô ”˛ ‘ì ÷“ ◊’ Ÿœ ⁄ü ‹  ›€ ﬂ√ ·Ñ ‚ﬁ „‡ Âÿ Ê9 Ë< È@ ÍÁ Ï± ÓÌ ≥ ÒÔ Ûµ ÙÚ ˆÎ ˜• ˘Ñ ˙ü ¸¯ ˝˚ ˇı Äü Çµ ÉÅ Ö≥ áÑ àÑ ââ ãä çú éÑ êå íè ìœ ïë ñî òÜ ôó õ˛ ú9 û< ü@ †ù ¢˝ §¿ ¶• ®£ ©¸ ´ß ≠™ Æ¨ ∞° ±± ≥ä ¥´ ∂≤ ∑µ πØ ∫ü º™ Ωª ø£ ¡Ñ ¬æ √¿ ≈∏ ∆9 »< …@  « ÃÜ ŒÕ –ä —œ ”è ‘“ ÷À ◊Ω Ÿê ⁄∑ ‹ÿ ›€ ﬂ’ ‡• ‚• ‰· Â„ ÁÑ ÈÑ ÍÊ Îü Ìü ÓË ÔÏ Òﬁ ÚÕ Ù… ˆÛ ˜ı ˘ä ˚ñ ¸¯ ˝è ˇ√ Ä˙ Å˛ É Ñú ÜÖ àä äá ãœ çå èè ëé íü îê ïì óâ ôÑ öñ õò ùÇ ûü °† £¢ •J ßï ©® ´™ ≠¶ Ø¨ ∞ ≤± ¥≥ ∂Æ ∑§ πµ ∫∫ ª∏ Ωû æY ¿§ ¬¡ ƒ√ ∆ø »≈ …Á À  Õ« Œ§ –Ã —‰ “œ ‘Ω ’h ◊≥ Ÿÿ €⁄ ›÷ ﬂ‹ ‡Û ‚· ‰ﬁ Â§ Á„ Ëö ÈÊ ÎÁ Ïw Ó¬ Ô ÚÒ ÙÌ ˆÛ ˜ˇ ˘¯ ˚ı ¸§ ˛˙ ˇƒ Ä˝ Çù É— ÖÑ áÜ âÕ ãà åã éç êä ë§ ìè îú ïí ó« òH ö› õW ùÍ ûf †˜ °u £Ñ §Ñ ¶ë ßì ©Ÿ ™¢ ¨‰ ≠± ØÒ ∞¿ ≤˛ ≥œ µã ∂€ ∏L πÂ ª[ ºÒ æj ø˝ ¡y ¬â ƒà ≈… «ó »‘  ¶ Àﬂ Õµ ŒÍ –ƒ —ı ”” ‘8 ÷< ◊@ ÿ’ ⁄Ÿ ‹€ ﬁ› ﬂ8 ·< ‚@ „‡ Â‰ ÁÊ ÈÈ Í8 Ï< Ì@ ÓÎ Ô ÚÒ Ùı ı8 ˜< ¯@ ˘ˆ ˚˙ ˝¸ ˇÅ Ä8 Ç< É@ ÑÅ ÜÖ àá äç ã2 ç< é@ èå ë3 ì< î@ ïí ó4 ô< ö@ õò ù5 ü< †@ °û £6 •< ¶@ ß§ ©7 ´< ¨@ ≠™ Ø9 ±< ≤@ ≥∞ µ… ∑§ π∂ ∫∏ º¶ Ωª ø¥ ¿ﬂ ¬¡ ƒÌ ≈√ «æ »9  < À@ Ã… Œ‘ –√ “œ ”— ’≈ ÷‘ ÿÕ Ÿ˛ €ê ‹ô ﬁ⁄ ﬂ› ·◊ ‚• ‰≈ Â„ Áœ Èñ ÍÊ ÎË Ì‡ Ó9 < Ò@ ÚÔ Ù≥ ˆ¡ ˜ı ˘Ì ˙¯ ¸Û ˝Ñ ˇñ Ä	• Ç	˛ É	Å	 Ö	˚ Ü	• à	Ì â	á	 ã	¡ ç	ñ é	ä	 è	ı ë	ê	 ì	Æ î	œ ñ	í	 ò	ï	 ô	’ õ	ó	 ú	ö	 û	å	 ü	ù	 °	Ñ	 ¢	9 §	< •	@ ¶	£	 ®	Í ™	£ ¨	©	 ≠	´	 Ø	• ∞	Æ	 ≤	ß	 ≥	ä µ	ú ∂	± ∏	¥	 π	∑	 ª	±	 º	• æ	• ø	Ω	 ¡	©	 √	ñ ƒ	¿	 ≈	¬	 «	∫	 »	9  	< À	@ Ã	…	 Œ	ä –	ê	 —	œ	 ”	ï	 ‘	“	 ÷	Õ	 ◊	ê Ÿ	¢ ⁄	Ω ‹	ÿ	 ›	€	 ﬂ	’	 ‡	Ñ ‚	Ñ ‰	·	 Â	„	 Á	ñ È	ñ Í	Ê	 Î	• Ì	• Ó	Ë	 Ô	Ï	 Ò	ﬁ	 Ú	ä Ù	ñ ˆ	Û	 ˜	ı	 ˘	ê	 ˚	® ¸	¯	 ˝	ï	 ˇ	… Ä
˙	 Å
˛	 É
	 Ñ
Æ Ü
Ö
 à
ê	 ä
á
 ã
’ ç
å
 è
ï	 ë
é
 í
• î
ê
 ï
ì
 ó
â
 ô
ñ ö
ñ
 õ
ò
 ù
Ç
 û
◊ †
J ¢
°
 §
ü
 ¶
£
 ß
ï ©
®
 ´
•
 ¨
± Æ
≠
 ∞
™
 ±
§ ≥
Ø
 ¥
∆ µ
≤
 ∑
∞ ∏
‚ ∫
Y º
ª
 æ
π
 ¿
Ω
 ¡
§ √
¬
 ≈
ø
 ∆
Á »
«
  
ƒ
 À
§ Õ
…
 Œ
Ï œ
Ã
 —
… “
Ô ‘
h ÷
’
 ÿ
”
 ⁄
◊
 €
≥ ›
‹
 ﬂ
Ÿ
 ‡
Û ‚
·
 ‰
ﬁ
 Â
§ Á
„
 Ë
†	 È
Ê
 Î
Ô Ï
¸ Ó
w 
Ô
 Ú
Ì
 Ù
Ò
 ı
¬ ˜
ˆ
 ˘
Û
 ˙
ˇ ¸
˚
 ˛
¯
 ˇ
§ Å˝
 Ç∆	 ÉÄ Ö£	 Üâ àä äá åâ ç— èé ëã íã îì ñê ó§ ôï öú
 õò ù…	 ûü
 °π
 £”
 •Ì
 ßá ©Ô
 ´é ≠ü ∞ ≤ µ ∑ π ªÆ Ω˙ ¿ì ¡· √˚
 ƒ  ∆·
 «≥ …«
  ú Ã≠
 Õó œ¨ –¬ “ˆ
 ”≈ ’‹
 ÷» ÿ¬
 ŸÀ €®
 ‹ï ﬁâ ﬂì ·™ ‚‘ ‰’
 Â◊ Áª
 Ë⁄ Í°
 Îë Ì® Óè ¶ Òç Û§ Ùã ˆ¢ ˜â ˘† ˙˜ ¸Ä ˛˛ ˇ˛ Åê ÇÜ ÑÑ ÖÖ áñ à° äÆ ãâ çú éö ê® ëè ìñ îì ñ¢ óï ôê öå úú ùõ üä †¯ ¢› £ı •Í ¶Ú ®˜ ©Ô ´Ñ ¨Ï Æë Ø‡ ±˛ ≤› ¥ã µŒ ∑à ∏˚ ∫8 º< Ωπ æ@ øª ¡¿ √¬ ≈› ∆8 »< …π  @ À« ÕÃ œŒ —È “8 ‘< ’π ÷@ ◊” Ÿÿ €⁄ ›ı ﬁ8 ‡< ·π ‚@ „ﬂ Â‰ ÁÊ ÈÅ Í8 Ï< Ìπ Ó@ ÔÎ Ò ÛÚ ıç ˆ˚ ¯2 ˙< ˚˜ ¸@ ˝˘ ˇ3 Å< Ç˜ É@ ÑÄ Ü4 à< â˜ ä@ ãá ç5 è< ê˜ ë@ íé î6 ñ< ó˜ ò@ ôï õ7 ù< û˜ ü@ †ú ¢9 §< •˚ ¶@ ß£ ©⁄ ´À ¨™ ÆÈ Ø≠ ±® ≤≈ ¥„ µ≥ ∑∞ ∏9 ∫< ª˚ º@ Ωπ ø◊ ¡» ¬¿ ƒÊ ≈√ «æ »Ä  ˛ À˝ Õ… ŒÃ –∆ —É ”Ê ‘“ ÷» ÿÖ Ÿ’ ⁄◊ ‹œ ›9 ﬂ< ‡˚ ·@ ‚ﬁ ‰‘ Ê≈ Á„ ÈÂ ÍË Ï„ ÌÜ ÔÖ É ÚÓ ÛÒ ıÎ ˆÉ ¯„ ˘˜ ˚≈ ˝Ö ˛˙ ˇø Å° Ç› ÑÄ ÜÉ áå âÖ äà å¸ çã èÙ ê9 í< ì˚ î@ ïë ó— ô¬ ö¸ úò ûõ üù °ñ ¢õ §å •û ß£ ®¶ ™† ´É ≠õ Æ¨ ∞¬ ≤Ö ≥Ø ¥± ∂© ∑9 π< ∫˚ ª@ º∏ æÜ ¿ø ¬ø √¡ ≈É ∆ƒ »Ω …ï Àì Ãò Œ  œÕ —« “Ü ‘Ü ÷” ◊’ ŸÖ €Ö ‹ÿ ›É ﬂÉ ‡⁄ ·ﬁ „– ‰ø Êè ËÂ ÈÁ Îø Ìö ÓÍ ÔÉ Òí ÚÏ Û ı‚ ˆ° ¯˜ ˙ø ¸˘ ˝å ˇ˛ ÅÉ ÉÄ ÑÉ ÜÇ áÖ â˚ ãÖ åà çä èÙ ê∫ íÈ îë ï⁄ óì òÀ öñ õ± ùô üú †§ ¢û £∂ §° ¶£ ßË ©Ê ´® ¨◊ Æ™ Ø» ±≠ ≤Á ¥∞ ∂≥ ∑§ πµ ∫€ ª∏ Ωπ æı ¿„ ¬ø √‘ ≈¡ ∆≈ »ƒ …Û À« Õ  Œ§ –Ã —é “œ ‘ﬁ ’Ç ◊õ Ÿ÷ ⁄— ‹ÿ ›¬ ﬂ€ ‡ˇ ‚ﬁ ‰· Â§ Á„ Ëµ ÈÊ Îë Ïè Óâ Ô ÚÌ Ûø ıÒ ˆø ¯Ù ˘ã ˚˜ ˝˙ ˛§ Ä¸ Åé Çˇ Ñ∏ Ö˜ áº àÈ äÊ å„ éõ êÔ í— îø ñø òÜ öÈ ú¥ ùÊ ü‚ †„ ¢Ô £⁄ •∂ ¶◊ ®Y ©‘ ´h ¨— Æw ØÀ ±∏ ≤» ¥§ µ≈ ∑≥ ∏¬ ∫¬ ªø Ω— æ± ¡∫ ¬ì ƒ˙ ≈˚
 «· »·
    À«
 Õ≥ Œ≠
 –ú —é ”ø ‘¨ ÷ó ◊ˆ
 Ÿ¬ ⁄‹
 ‹≈ ›¬
 ﬂ» ‡®
 ‚À „â Âï Ê™ Ëì È’
 Î‘ Ïª
 Ó◊ Ô°
 Ò⁄ Ú® Ùë ı¶ ˜è ¯§ ˙ç ˚¢ ˝ã ˛† Äâ Åä Éõ Ñú Üå áê âï ä¢ åì çñ èè ê® íö ìú ïâ ñÆ ò° ôñ õÖ úÑ ûÜ üê °˛ ¢˛ §Ä •ˇ ß› ®¸ ™Í ´˘ ≠˜ Æˆ ∞Ñ ±Û ≥ë ¥ ∂ ∏µ πÌ ª‚ ºÍ æÔ øÁ ¡˛ ¬‰ ƒã ≈ «· …∆  ﬁ ÃY Õ€ œh –ÿ “w ”“ ’Ü ÷ ÿœ ⁄◊ €Ã ›§ ﬁ… ‡≥ ·∆ „¬ ‰√ Ê— ÁË Í8 Ï< ÌÈ Ó@ ÔÎ Ò ÛÚ ı› ˆ8 ¯< ˘È ˙@ ˚˜ ˝¸ ˇ˛ ÅÈ Ç8 Ñ< ÖÈ Ü@ áÉ âà ãä çı é8 ê< ëÈ í@ ìè ïî óñ ôÅ ö8 ú< ùÈ û@ üõ °† £¢ •ç ¶ß ©2 ´< ¨® ≠@ Æ™ ∞3 ≤< ≥® ¥@ µ± ∑4 π< ∫® ª@ º∏ æ5 ¿< ¡® ¬@ √ø ≈6 «< »® …@  ∆ Ã7 Œ< œ® –@ —Õ ”Æ ’9 ◊< ÿ‘ Ÿ@ ⁄÷ ‹· ﬁœ ﬂ› · ‚‡ ‰€ Â… ÁÍ ËÊ Í„ Î9 Ì< Ó‘ Ô@ Ï Úﬁ ÙÃ ıÛ ˜Ì ¯ˆ ˙Ò ˚† ˝Ø ˛£ Ä¸ Åˇ É˘ Ñù ÜÌ áÖ âÃ ã∂ åà çä èÇ ê9 í< ì‘ î@ ïë ó€ ô… öÍ úò ùõ üñ †ö ¢∂ £ù •° ¶§ ®û ©ù ´Í ¨™ Æ… ∞∂ ±≠ ≤√ ¥“ µ‰ ∑≥ π∂ ∫î º∏ Ωª øØ ¿æ ¬ß √9 ≈< ∆‘ «@ »ƒ  ÿ Ã∆ Õ¸ œÀ —Œ “– ‘… ’Ö ◊Ω ÿÇ ⁄÷ €Ÿ ›” ﬁù ‡Œ ·ﬂ „∆ Â∂ Ê‚ Á‰ È‹ Í9 Ï< Ì‘ Ó@ ÔÎ ÒÜ ÛÚ ı√ ˆÙ ¯∂ ˘˜ ˚ ¸ã ˛ƒ ˇà Å˝ ÇÄ Ñ˙ Öö áö âÜ äà å∂ é∂ èã êù íù ìç îë ñÉ óÚ ôë õò úö û√ †À °ù ¢∂ §é •ü ¶£ ®ï ©“ ´™ ≠√ Ø¨ ∞î ≤± ¥∂ ∂≥ ∑ù πµ ∫∏ ºÆ æ∂ øª ¿Ω ¬ß √¿ ≈◊ «∆ …ƒ  J ÃÀ Œ» œï —– ”Õ ‘§ ÷“ ◊È ÿ’ ⁄÷ €Ë ›‚ ﬂﬁ ·‹ ‚Y ‰„ Ê‡ Á§ ÈË ÎÂ Ï§ ÓÍ Ôé Ì ÚÏ Ûı ıÔ ˜ˆ ˘Ù ˙h ¸˚ ˛¯ ˇ≥ ÅÄ É˝ Ñ§ ÜÇ á¡ àÖ äë ãÇ çŒ èå êw íë îé ï¬ óñ ôì ö§ úò ùË ûõ †ƒ °è £â •§ ß¢ ®Ú ™¶ ´— ≠¨ Ø© ∞§ ≤Æ ≥¡ ¥± ∂Î ∑ π ª∏ ºÌ æË øÍ ¡ı ¬Á ƒÑ ≈‰ «ë »  · Ã… Õﬁ œ‚ –€ “Ô ”ÿ ’¸ ÷’ ÿã Ÿ €œ ›⁄ ﬁÃ ‡Y ·… „h ‰∆ Êw Á√ ÈÜ ÍÚ Ïó Ì˛ Ô¶ ä Úµ Ûñ ıƒ ˆ¢ ¯” ˘2 ˚< ¸È ˝@ ˛˙ Ä3 Ç< ÉÈ Ñ@ ÖÅ á4 â< äÈ ã@ åà é5 ê< ëÈ í@ ìè ï6 ó< òÈ ô@ öñ ú7 û< üÈ †@ °ù £9 •< ¶® ß@ ®§ ™Ú ¨œ Æ´ Ø≠ ±· ≤∞ ¥© µä ∑∂ π€ ∫∏ º≥ Ω9 ø< ¿® ¡@ ¬æ ƒ˛ ∆Ã »≈ …« Àﬁ Ã  Œ√ œØ —ˇ “† ‘– ’” ◊Õ ÿö ⁄ﬁ €Ÿ ›≈ ﬂÜ ‡‹ ·ﬁ „÷ ‰9 Ê< Á® Ë@ ÈÂ Î… Ì∂ Ó€ Ï ÒÔ ÛÍ Ù∂ ˆÜ ˜ö ˘ı ˙¯ ¸Ú ˝ö ˇ€ Ä˛ Ç∂ ÑÜ ÖÅ Ü¢ àá ä¢ ãâ ç“ éó êå ëè ìÉ îí ñ˚ ó9 ô< ö® õ@ úò ûñ †∆ ¢ü £° •ÿ ¶§ ®ù ©Ω ´ç ¨Ö Æ™ Ø≠ ±ß ≤ö ¥ÿ µ≥ ∑ü πÜ ∫∂ ª∏ Ω∞ æ9 ¿< ¡® ¬@ √ø ≈√ «á »“  ∆ À… Õƒ Œƒ –î —ã ”œ ‘“ ÷Ã ◊∂ Ÿ∂ €ÿ ‹⁄ ﬁÜ ‡Ü ·› ‚ö ‰ö Âﬂ Ê„ Ë’ È√ ÎÀ ÌÍ ÓÏ á Úõ ÛÔ Ù“ ˆë ˜Ò ¯ı ˙Á ˚¢ ˝¸ ˇá Å˛ Çó ÑÉ Ü“ àÖ âö ãá åä éÄ êÜ ëç íè î˘ ï¿ ó◊ ôò õñ úJ ûù †ö °§ £ü §ª •¢ ß§ ®Ë ™‚ ¨´ Æ© ØY ±∞ ≥≠ ¥§ ∂≤ ∑‚ ∏µ ∫æ ªı ΩÔ øæ ¡º ¬h ƒ√ ∆¿ «§ …≈  ï À» ÕÂ ŒÇ –¸ “— ‘œ ’w ◊÷ Ÿ” ⁄§ ‹ÿ ›º ﬁ€ ‡ò ·è „â Â‰ Á‚ ËÜ ÍÈ ÏÊ Ì§ ÔÎ ì ÒÓ Ûø Ù ˜ ˘ ˚
 ˝ ˇ( ˆ( *0 ˆ0 2Ø ±Ø ¥≥ ¿æ øı ˆô õô øø ¿ éé èè åå êê Ä çç ëëØ éé Øﬁ éé ﬁß éé ß” éé ”† éé †Ä éé Ä˘ éé ˘Ç éé Çÿ éé ÿ® éé ®˚ éé ˚¸ éé ¸Ï éé Ï ëë ﬁ éé ﬁõ éé õ‡ éé ‡ò éé ò¯ êê ¯ò éé ò… éé …Ã
 éé Ã
∫ éé ∫Ê éé Ê≤ éé ≤ò éé ò« éé «ò éé òÏ éé ÏÒ éé Ò€ éé €Ê éé ÊÔ éé Ô™ éé ™∆ éé ∆ﬁ éé ﬁÆ éé Æã éé ã∏ éé ∏¬	 éé ¬	◊ éé ◊◊ éé ◊  éé  ¸ éé ¸ åå í éé íÿ éé ÿ¡ éé ¡¿ éé ¿ø
 éé ø
Ë éé Ë‰ éé ‰ô éé ô´	 éé ´	Õ éé ÕÂ éé Âé éé éÜ éé Üò éé òö éé ö÷ éé ÷ï éé ïê éé ê” éé ”ﬁ éé ﬁß éé ß¿ éé ¿∏ éé ∏¥	 éé ¥	œ éé œÏ	 éé Ï	 åå  åå ¶ éé ¶ åå € éé €Ç
 éé Ç
± éé ±Ë éé Ë„ éé „ çç ˙ éé ˙™ éé ™ü éé üœ	 éé œ	ì éé ìÏ éé Ï∏ éé ∏ˆ êê ˆù	 éé ù	« éé «œ éé œæ éé æØ éé Ø∫	 éé ∫	µ éé µ∞ éé ∞˘ éé ˘÷ éé ÷≠ éé ≠∂ éé ∂µ éé µÛ éé Û– éé –≥ éé ≥ƒ éé ƒÒ éé ÒΩ éé Ω’ éé ’æ éé æ⁄ éé ⁄À éé Àä éé äÈ éé Èû éé ûü éé üı éé ıé éé é¯ éé ¯ú
 éé ú
 éé © éé © éé ê éé êﬁ éé ﬁœ éé œ° éé °Ã éé Ã¢ éé ¢∆ éé ∆˛ êê ˛å	 éé å	¯
 éé ¯
£ éé £œ éé œ±	 éé ±	‡ éé ‡— éé —‡ éé ‡ß éé ß’ éé ’¯ éé ¯â
 éé â
¿ éé ¿é éé éä éé äº éé º› éé ›˝ éé ˝ı éé ı£ éé £ˇ éé ˇÛ
 éé Û
∞ éé ∞» éé »ä éé ä˛ éé ˛Ù éé Ù∏ éé ∏â éé â˙	 éé ˙	Ó éé Óú éé úƒ éé ƒÄ éé ÄÊ
 éé Ê
ï éé ï» éé »„ éé „! çç !Æ éé Æ˚ éé ˚Æ éé Æ± éé ±˛ éé ˛ö éé ö∞ éé ∞˛	 éé ˛	ó éé óﬂ éé ﬂ¡ éé ¡© éé ©˝ éé ˝Á éé Áÿ éé ÿÎ éé Î⁄ éé ⁄ÿ éé ÿ≠ éé ≠˙ êê ˙≤ éé ≤è éé è≤
 éé ≤
¡ éé ¡Ç éé Çç éé ç“ éé “Ú éé Úñ éé ñ˙ éé ˙Â éé ÂÕ éé Õ åå « éé «Ö éé Ö† èè †ò
 éé ò
∏ éé ∏ƒ
 éé ƒ
œ éé œÉ éé ÉÙ éé ÙÓ éé ÓË éé Ëì éé ì« éé «‹ éé ‹¸ êê ¸Ï éé Ïê
 éé ê
í éé í€ éé €° éé °‚ éé ‚ı éé ı˝ éé ˝µ éé µË	 éé Ë	Ì éé Ì˛ éé ˛ı éé ı∆ éé ∆ë éé ë¡ éé ¡é éé éı éé ıª éé ª° éé °‰ éé ‰œ éé œü èè üã éé ãﬁ
 éé ﬁ
∆	 éé ∆	≠ éé ≠	 éé 	‚ éé ‚á éé á™ éé ™≈ éé ≈˚ éé ˚†	 éé †	˜ éé ˜ß éé ßì éé ì’ éé ’Ç éé ÇÙ éé ÙÑ	 éé Ñ	•
 éé •
ﬁ	 éé ﬁ	É éé É« éé «∞ éé ∞Ù éé Ù’	 éé ’	“ éé “ÿ	 éé ÿ	™
 éé ™
– éé –Ç éé ÇÊ éé ÊŸ
 éé Ÿ
Î éé ÎÍ éé Í	í $	í ,
í ß
ì ∫
ì ‰
ì ö
ì ƒ
ì ú
ì ∆
ì Ï
ì †	
ì ∆	
ì ú

ì ∂
ì €
ì é
ì µ
ì é
ì È
ì é
ì ¡
ì Ë
ì ¡
ì ª
ì ‚
ì ï
ì º
ì ì
î ˛
î Ñ	
î Ù
î ß
î ˚ï ï ï ï ï 	ï ï ˆï ¯ï ˙ï ¸ï ˛ñ !	ó `	ó h
ó ´
ó ≥
ó ’
ó ﬂ
ó Î
ó Î
ó Û
ó ˜
ó É
ó Ô
ó ı
ó Ÿ
ó ˙
ó Ä
ó Ü
ó å
ó í
ó ò
ó Á
ó Î
ó ∞
ó …
ó Ô
ó Ô
ó £	
ó …	
ó π
ó ”
ó ﬁ
ó É
ó ë
ó Â
ò â
ò ê
ò â

ò ê

ò ˚
ò Ç
ò Æ
ò µ
ò Ä
ò á
ô ∞
ô œ
ô ı
ô Ø
ô ’
ô æ
ô ◊
ô ˚
ô ±	
ô ’	
ô ∞
ô ∆
ô Î
ô †
ô «
ô „
ô ˘
ô û
ô ”
ô ˙
ô ≥
ô Õ
ô Ú
ô ß
ô Ã
ö •

ö ™

ö ø

ö ƒ

ö Ÿ

ö ﬁ

ö Û

ö ¯

ö ã
ö ê
ö ì
ö ô
ö ™
ö ∞
ö ¡
ö «
ö ÿ
ö ﬁ
ö Ò
ö ˜
ö »
ö “
ö ‡
ö Í
ö ¯
ö Ç
ö é
ö ò
ö ¶
ö Æ
ö ö
ö ≠
ö ¿
ö ”
ö Ê
õ üú üú †
ù ﬁ
ù ﬁ	
ù –
ù É
ù ’û û û û û û 	ü ~
ü Ü
ü …
ü —
ü É
ü ã
ü â
ü è
ü Ô
ü «
ü ’
ü ‡
ü Î
ü ˆ
ü Å
ü Å
ü …	
ü Î
ü ∏
ü õ
ü Î
ü ø
† ó
† Ö
† å
† ù	
† Ö

† å

† ã
† ˜
† ˛
† æ
† ™
† ±
† í
† ¸
† É
° Ë	¢ o	¢ w
¢ ∫
¢ ¬
¢ ˜
¢ ˇ
¢ ¸
¢ Ç
¢ √
¢ Œ
¢ Ÿ
¢ ‰
¢ ‰
¢ Ô
¢ ù
¢ ˆ
¢ å
¢ í
¢ ò
¢ û
¢ §
¢ ™
¢ £	
¢ ˚
¢ ﬂ
¢ ë
¢ è
¢ ƒ
¢ ò
£ ·
£ Û
£ ·	
£ Û	
£ ”
£ Â
£ Ü
£ ò
£ ÿ
£ Í
§ Æ
§ «
§ ﬁ
§ ı
§ ä
§ ü
§ ≤
§ ≈
§ ÿ
§ Î	• 
¶ ÿ
¶ ∏
¶ ‡
¶ ∫	
¶ œ
¶ ©
¶ Ç
¶ ‹
¶ ÷
¶ ∞
ß ¢
® 
® 	
® ‚
® ï
® Á
© ü
™ £

™ Ω

™ ◊

™ Ò

™ â
™ ñ
™ ≠
™ ƒ
™ €
™ Ù
™ Õ
™ Â
™ ˝
™ ì
™ ©	´ 
¨ Ç
¨ Ç

¨ Ù
¨ ß
¨ ˘	≠ B	≠ J	≠ J	≠ Q	≠ Y	≠ `	≠ h	≠ o	≠ w	≠ ~
≠ Ü
≠ ï
≠ ï
≠ §
≠ ≥
≠ ¬
≠ —
≠ Á
≠ Û
≠ ˇ
≠ ã
≠ è
≠ õ
≠ ß
≠ ≥
≠ ø
≠ À
≠ ◊
≠ ◊
≠ ‚
≠ Ë
≠ Ô
≠ ı
≠ ¸
≠ Ç
≠ â
≠ è
≠ û
≠ ±
≠ ±
≠ ∞
≠ ±
≠ ±
≠ ¥
≠ ¥
≠ ∂
≠ ∂
≠ ∏
≠ ∏
≠ ∫
≠ ∫
≠ £
≠ µ
≠ µ
≠ ∆
≠ ∆
≠ ◊
≠ ◊
≠ ÷
≠ ∏
≠ ∏
≠ …
≠ …
≠ ⁄
≠ ⁄
≠ §
Æ ®
Æ «
Æ “
Æ Ô
Æ ¯
Æ ß
Æ ≤
Æ œ
Æ ÿ
Æ ∏
Æ —
Æ ⁄
Æ ı
Æ ˛
Æ ´	
Æ ¥	
Æ œ	
Æ ÿ	
Æ ™
Æ ¿
Æ …
Æ Â
Æ Ó
Æ ò
Æ £
Æ ¡
Æ  
Æ ›
Æ Û
Æ ¸
Æ ò
Æ °
Æ À
Æ ÷
Æ Ù
Æ ˝
Æ ≠
Æ «
Æ –
Æ Ï
Æ ı
Æ °
Æ ™
Æ ∆
Æ œ	Ø :	Ø <	Ø >	Ø @	∞ 	∞ "	∞ Q	∞ Y
∞ ç
∞ ú
∞ ú
∞ §
∞ ´
∞ ∫
∞ …
∞ ﬂ
∞ Á
∞ ï
∞ °
∞ ≠
∞ π
∞ ≈
∞ —
∞ ‚
∞ Ë
∞ Œ
∞ û
∞ Ω
∞ Ω
∞ Á
∞ ù
∞ «
∞ ‡
∞ …
∞ «
∞ ˜
∞ π
∞ ˜
∞ Ï
∞ æ
± ™
± √
± ⁄
± Ò
± Ü
≤ Æ≥ ﬁ≥ Ñ≥ æ≥ Ê≥ ¯≥ á≥ é≥ ñ≥ §≥ ¨≥ ≈≥ ‹≥ Û≥ à≥ Ê≥ ä	≥ ¿	≥ Ê	≥ ¯	≥ á
≥ é
≥ ñ
≥ ’≥ ˙≥ Ø≥ ÿ≥ Í≥ ˘≥ Ä≥ à≥ à≥ ≠≥ ‚≥ ã≥ ù≥ ¨≥ ≥≥ ª≥ ‹≥ Å≥ ∂≥ ›≥ Ô≥ ˛≥ Ö≥ ç"
compute_rhs4"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"

_Z3maxdd"
llvm.lifetime.end.p0i8"
llvm.memset.p0i8.i64*ë
npb-BT-compute_rhs4_A.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02Å

transfer_bytes	
ÿîÁò

devmap_label


wgsize_log1p
ùÆúA
 
transfer_bytes_log1p
ùÆúA

wgsize
>