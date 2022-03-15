
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
6icmpB.
,
	full_text

%27 = icmp sgt i32 %24, %10
#i32B

	full_text
	
i32 %24
6truncB-
+
	full_text

%28 = trunc i64 %26 to i32
#i64B

	full_text
	
i64 %26
5icmpB-
+
	full_text

%29 = icmp sgt i32 %28, %9
#i32B

	full_text
	
i32 %28
-orB'
%
	full_text

%30 = or i1 %27, %29
!i1B

	full_text


i1 %27
!i1B

	full_text


i1 %29
9brB3
1
	full_text$
"
 br i1 %30, label %999, label %31
!i1B

	full_text


i1 %30
Ybitcast8BL
J
	full_text=
;
9%32 = bitcast double* %0 to [163 x [163 x [5 x double]]]*
Sbitcast8BF
D
	full_text7
5
3%33 = bitcast double* %1 to [163 x [163 x double]]*
Sbitcast8BF
D
	full_text7
5
3%34 = bitcast double* %2 to [163 x [163 x double]]*
Sbitcast8BF
D
	full_text7
5
3%35 = bitcast double* %3 to [163 x [163 x double]]*
Sbitcast8BF
D
	full_text7
5
3%36 = bitcast double* %4 to [163 x [163 x double]]*
Sbitcast8BF
D
	full_text7
5
3%37 = bitcast double* %5 to [163 x [163 x double]]*
Sbitcast8BF
D
	full_text7
5
3%38 = bitcast double* %6 to [163 x [163 x double]]*
Ybitcast8BL
J
	full_text=
;
9%39 = bitcast double* %7 to [163 x [163 x [5 x double]]]*
1shl8B(
&
	full_text

%40 = shl i64 %23, 32
%i648B

	full_text
	
i64 %23
9ashr8B/
-
	full_text 

%41 = ashr exact i64 %40, 32
%i648B

	full_text
	
i64 %40
1shl8B(
&
	full_text

%42 = shl i64 %26, 32
%i648B

	full_text
	
i64 %26
9ashr8B/
-
	full_text 

%43 = ashr exact i64 %42, 32
%i648B

	full_text
	
i64 %42
•getelementptr8B

	full_textr
p
n%44 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Obitcast8BB
@
	full_text3
1
/%45 = bitcast [163 x [5 x double]]* %44 to i64*
I[163 x [5 x double]]*8B,
*
	full_text

[163 x [5 x double]]* %44
Hload8B>
<
	full_text/
-
+%46 = load i64, i64* %45, align 8, !tbaa !8
'i64*8B

	full_text


i64* %45
pgetelementptr8B]
[
	full_textN
L
J%47 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Gbitcast8B:
8
	full_text+
)
'%48 = bitcast [5 x double]* %12 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Istore8B>
<
	full_text/
-
+store i64 %46, i64* %48, align 16, !tbaa !8
%i648B

	full_text
	
i64 %46
'i64*8B

	full_text


i64* %48
¥getelementptr8B‘
Ž
	full_text€
~
|%49 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 0, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%50 = bitcast double* %49 to i64*
-double*8B

	full_text

double* %49
Hload8B>
<
	full_text/
-
+%51 = load i64, i64* %50, align 8, !tbaa !8
'i64*8B

	full_text


i64* %50
pgetelementptr8B]
[
	full_textN
L
J%52 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%53 = bitcast double* %52 to i64*
-double*8B

	full_text

double* %52
Hstore8B=
;
	full_text.
,
*store i64 %51, i64* %53, align 8, !tbaa !8
%i648B

	full_text
	
i64 %51
'i64*8B

	full_text


i64* %53
¥getelementptr8B‘
Ž
	full_text€
~
|%54 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 0, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%55 = bitcast double* %54 to i64*
-double*8B

	full_text

double* %54
Hload8B>
<
	full_text/
-
+%56 = load i64, i64* %55, align 8, !tbaa !8
'i64*8B

	full_text


i64* %55
pgetelementptr8B]
[
	full_textN
L
J%57 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%58 = bitcast double* %57 to i64*
-double*8B

	full_text

double* %57
Istore8B>
<
	full_text/
-
+store i64 %56, i64* %58, align 16, !tbaa !8
%i648B

	full_text
	
i64 %56
'i64*8B

	full_text


i64* %58
¥getelementptr8B‘
Ž
	full_text€
~
|%59 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 0, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%60 = bitcast double* %59 to i64*
-double*8B

	full_text

double* %59
Hload8B>
<
	full_text/
-
+%61 = load i64, i64* %60, align 8, !tbaa !8
'i64*8B

	full_text


i64* %60
pgetelementptr8B]
[
	full_textN
L
J%62 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%63 = bitcast double* %62 to i64*
-double*8B

	full_text

double* %62
Hstore8B=
;
	full_text.
,
*store i64 %61, i64* %63, align 8, !tbaa !8
%i648B

	full_text
	
i64 %61
'i64*8B

	full_text


i64* %63
¥getelementptr8B‘
Ž
	full_text€
~
|%64 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 0, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%65 = bitcast double* %64 to i64*
-double*8B

	full_text

double* %64
Hload8B>
<
	full_text/
-
+%66 = load i64, i64* %65, align 8, !tbaa !8
'i64*8B

	full_text


i64* %65
pgetelementptr8B]
[
	full_textN
L
J%67 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Abitcast8B4
2
	full_text%
#
!%68 = bitcast double* %67 to i64*
-double*8B

	full_text

double* %67
Istore8B>
<
	full_text/
-
+store i64 %66, i64* %68, align 16, !tbaa !8
%i648B

	full_text
	
i64 %66
'i64*8B

	full_text


i64* %68
getelementptr8B‰
†
	full_texty
w
u%69 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Gbitcast8B:
8
	full_text+
)
'%70 = bitcast [5 x double]* %69 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %69
Hload8B>
<
	full_text/
-
+%71 = load i64, i64* %70, align 8, !tbaa !8
'i64*8B

	full_text


i64* %70
pgetelementptr8B]
[
	full_textN
L
J%72 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Gbitcast8B:
8
	full_text+
)
'%73 = bitcast [5 x double]* %13 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Istore8B>
<
	full_text/
-
+store i64 %71, i64* %73, align 16, !tbaa !8
%i648B

	full_text
	
i64 %71
'i64*8B

	full_text


i64* %73
¥getelementptr8B‘
Ž
	full_text€
~
|%74 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 1, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%75 = bitcast double* %74 to i64*
-double*8B

	full_text

double* %74
Hload8B>
<
	full_text/
-
+%76 = load i64, i64* %75, align 8, !tbaa !8
'i64*8B

	full_text


i64* %75
pgetelementptr8B]
[
	full_textN
L
J%77 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%78 = bitcast double* %77 to i64*
-double*8B

	full_text

double* %77
Hstore8B=
;
	full_text.
,
*store i64 %76, i64* %78, align 8, !tbaa !8
%i648B

	full_text
	
i64 %76
'i64*8B

	full_text


i64* %78
¥getelementptr8B‘
Ž
	full_text€
~
|%79 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 1, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%80 = bitcast double* %79 to i64*
-double*8B

	full_text

double* %79
Hload8B>
<
	full_text/
-
+%81 = load i64, i64* %80, align 8, !tbaa !8
'i64*8B

	full_text


i64* %80
pgetelementptr8B]
[
	full_textN
L
J%82 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%83 = bitcast double* %82 to i64*
-double*8B

	full_text

double* %82
Istore8B>
<
	full_text/
-
+store i64 %81, i64* %83, align 16, !tbaa !8
%i648B

	full_text
	
i64 %81
'i64*8B

	full_text


i64* %83
¥getelementptr8B‘
Ž
	full_text€
~
|%84 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 1, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%85 = bitcast double* %84 to i64*
-double*8B

	full_text

double* %84
Hload8B>
<
	full_text/
-
+%86 = load i64, i64* %85, align 8, !tbaa !8
'i64*8B

	full_text


i64* %85
pgetelementptr8B]
[
	full_textN
L
J%87 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%88 = bitcast double* %87 to i64*
-double*8B

	full_text

double* %87
Hstore8B=
;
	full_text.
,
*store i64 %86, i64* %88, align 8, !tbaa !8
%i648B

	full_text
	
i64 %86
'i64*8B

	full_text


i64* %88
¥getelementptr8B‘
Ž
	full_text€
~
|%89 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 1, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%90 = bitcast double* %89 to i64*
-double*8B

	full_text

double* %89
Hload8B>
<
	full_text/
-
+%91 = load i64, i64* %90, align 8, !tbaa !8
'i64*8B

	full_text


i64* %90
pgetelementptr8B]
[
	full_textN
L
J%92 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Abitcast8B4
2
	full_text%
#
!%93 = bitcast double* %92 to i64*
-double*8B

	full_text

double* %92
getelementptr8B‰
†
	full_texty
w
u%94 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Gbitcast8B:
8
	full_text+
)
'%95 = bitcast [5 x double]* %94 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %94
Hload8B>
<
	full_text/
-
+%96 = load i64, i64* %95, align 8, !tbaa !8
'i64*8B

	full_text


i64* %95
Gbitcast8B:
8
	full_text+
)
'%97 = bitcast [5 x double]* %14 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
¥getelementptr8B‘
Ž
	full_text€
~
|%98 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 2, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Abitcast8B4
2
	full_text%
#
!%99 = bitcast double* %98 to i64*
-double*8B

	full_text

double* %98
Iload8B?
=
	full_text0
.
,%100 = load i64, i64* %99, align 8, !tbaa !8
'i64*8B

	full_text


i64* %99
qgetelementptr8B^
\
	full_textO
M
K%101 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%102 = bitcast double* %101 to i64*
.double*8B

	full_text

double* %101
¦getelementptr8B’

	full_text

}%103 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 2, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%104 = bitcast double* %103 to i64*
.double*8B

	full_text

double* %103
Jload8B@
>
	full_text1
/
-%105 = load i64, i64* %104, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %104
qgetelementptr8B^
\
	full_textO
M
K%106 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%107 = bitcast double* %106 to i64*
.double*8B

	full_text

double* %106
¦getelementptr8B’

	full_text

}%108 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 2, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%109 = bitcast double* %108 to i64*
.double*8B

	full_text

double* %108
Jload8B@
>
	full_text1
/
-%110 = load i64, i64* %109, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %109
qgetelementptr8B^
\
	full_textO
M
K%111 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%112 = bitcast double* %111 to i64*
.double*8B

	full_text

double* %111
¦getelementptr8B’

	full_text

}%113 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 2, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%114 = bitcast double* %113 to i64*
.double*8B

	full_text

double* %113
Jload8B@
>
	full_text1
/
-%115 = load i64, i64* %114, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %114
qgetelementptr8B^
\
	full_textO
M
K%116 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Cbitcast8B6
4
	full_text'
%
#%117 = bitcast double* %116 to i64*
.double*8B

	full_text

double* %116
getelementptr8B}
{
	full_textn
l
j%118 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 0
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%119 = load double, double* %118, align 8, !tbaa !8
.double*8B

	full_text

double* %118
getelementptr8B}
{
	full_textn
l
j%120 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 1
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%121 = load double, double* %120, align 8, !tbaa !8
.double*8B

	full_text

double* %120
getelementptr8B}
{
	full_textn
l
j%122 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 0
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%123 = load double, double* %122, align 8, !tbaa !8
.double*8B

	full_text

double* %122
getelementptr8B}
{
	full_textn
l
j%124 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 1
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%125 = load double, double* %124, align 8, !tbaa !8
.double*8B

	full_text

double* %124
getelementptr8B}
{
	full_textn
l
j%126 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 0
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%127 = load double, double* %126, align 8, !tbaa !8
.double*8B

	full_text

double* %126
getelementptr8B}
{
	full_textn
l
j%128 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 1
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%129 = load double, double* %128, align 8, !tbaa !8
.double*8B

	full_text

double* %128
getelementptr8B}
{
	full_textn
l
j%130 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 0
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%131 = load double, double* %130, align 8, !tbaa !8
.double*8B

	full_text

double* %130
getelementptr8B}
{
	full_textn
l
j%132 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 1
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%133 = load double, double* %132, align 8, !tbaa !8
.double*8B

	full_text

double* %132
getelementptr8B}
{
	full_textn
l
j%134 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 0
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%135 = load double, double* %134, align 8, !tbaa !8
.double*8B

	full_text

double* %134
getelementptr8B}
{
	full_textn
l
j%136 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 1
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%137 = load double, double* %136, align 8, !tbaa !8
.double*8B

	full_text

double* %136
getelementptr8B}
{
	full_textn
l
j%138 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 0
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%139 = load double, double* %138, align 8, !tbaa !8
.double*8B

	full_text

double* %138
getelementptr8B}
{
	full_textn
l
j%140 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 1
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%141 = load double, double* %140, align 8, !tbaa !8
.double*8B

	full_text

double* %140
qgetelementptr8B^
\
	full_textO
M
K%142 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Hbitcast8B;
9
	full_text,
*
(%143 = bitcast [5 x double]* %15 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Kload8BA
?
	full_text2
0
.%144 = load i64, i64* %143, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %143
Hbitcast8B;
9
	full_text,
*
(%145 = bitcast [5 x double]* %16 to i64*
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Kstore8B@
>
	full_text1
/
-store i64 %144, i64* %145, align 16, !tbaa !8
&i648B

	full_text


i64 %144
(i64*8B

	full_text

	i64* %145
qgetelementptr8B^
\
	full_textO
M
K%146 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%147 = bitcast double* %146 to i64*
.double*8B

	full_text

double* %146
Jload8B@
>
	full_text1
/
-%148 = load i64, i64* %147, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %147
qgetelementptr8B^
\
	full_textO
M
K%149 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 1
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%150 = bitcast double* %149 to i64*
.double*8B

	full_text

double* %149
Jstore8B?
=
	full_text0
.
,store i64 %148, i64* %150, align 8, !tbaa !8
&i648B

	full_text


i64 %148
(i64*8B

	full_text

	i64* %150
qgetelementptr8B^
\
	full_textO
M
K%151 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%152 = bitcast double* %151 to i64*
.double*8B

	full_text

double* %151
Kload8BA
?
	full_text2
0
.%153 = load i64, i64* %152, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %152
qgetelementptr8B^
\
	full_textO
M
K%154 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 2
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%155 = bitcast double* %154 to i64*
.double*8B

	full_text

double* %154
Kstore8B@
>
	full_text1
/
-store i64 %153, i64* %155, align 16, !tbaa !8
&i648B

	full_text


i64 %153
(i64*8B

	full_text

	i64* %155
qgetelementptr8B^
\
	full_textO
M
K%156 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%157 = bitcast double* %156 to i64*
.double*8B

	full_text

double* %156
Jload8B@
>
	full_text1
/
-%158 = load i64, i64* %157, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %157
qgetelementptr8B^
\
	full_textO
M
K%159 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 3
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%160 = bitcast double* %159 to i64*
.double*8B

	full_text

double* %159
Jstore8B?
=
	full_text0
.
,store i64 %158, i64* %160, align 8, !tbaa !8
&i648B

	full_text


i64 %158
(i64*8B

	full_text

	i64* %160
qgetelementptr8B^
\
	full_textO
M
K%161 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Cbitcast8B6
4
	full_text'
%
#%162 = bitcast double* %161 to i64*
.double*8B

	full_text

double* %161
Kload8BA
?
	full_text2
0
.%163 = load i64, i64* %162, align 16, !tbaa !8
(i64*8B

	full_text

	i64* %162
qgetelementptr8B^
\
	full_textO
M
K%164 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 4
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Cbitcast8B6
4
	full_text'
%
#%165 = bitcast double* %164 to i64*
.double*8B

	full_text

double* %164
Kstore8B@
>
	full_text1
/
-store i64 %163, i64* %165, align 16, !tbaa !8
&i648B

	full_text


i64 %163
(i64*8B

	full_text

	i64* %165
Jstore8B?
=
	full_text0
.
,store i64 %46, i64* %143, align 16, !tbaa !8
%i648B

	full_text
	
i64 %46
(i64*8B

	full_text

	i64* %143
Istore8B>
<
	full_text/
-
+store i64 %51, i64* %147, align 8, !tbaa !8
%i648B

	full_text
	
i64 %51
(i64*8B

	full_text

	i64* %147
Jstore8B?
=
	full_text0
.
,store i64 %56, i64* %152, align 16, !tbaa !8
%i648B

	full_text
	
i64 %56
(i64*8B

	full_text

	i64* %152
Istore8B>
<
	full_text/
-
+store i64 %61, i64* %157, align 8, !tbaa !8
%i648B

	full_text
	
i64 %61
(i64*8B

	full_text

	i64* %157
Jstore8B?
=
	full_text0
.
,store i64 %66, i64* %162, align 16, !tbaa !8
%i648B

	full_text
	
i64 %66
(i64*8B

	full_text

	i64* %162
Istore8B>
<
	full_text/
-
+store i64 %71, i64* %48, align 16, !tbaa !8
%i648B

	full_text
	
i64 %71
'i64*8B

	full_text


i64* %48
Hstore8B=
;
	full_text.
,
*store i64 %76, i64* %53, align 8, !tbaa !8
%i648B

	full_text
	
i64 %76
'i64*8B

	full_text


i64* %53
Istore8B>
<
	full_text/
-
+store i64 %81, i64* %58, align 16, !tbaa !8
%i648B

	full_text
	
i64 %81
'i64*8B

	full_text


i64* %58
Hstore8B=
;
	full_text.
,
*store i64 %86, i64* %63, align 8, !tbaa !8
%i648B

	full_text
	
i64 %86
'i64*8B

	full_text


i64* %63
Istore8B>
<
	full_text/
-
+store i64 %91, i64* %68, align 16, !tbaa !8
%i648B

	full_text
	
i64 %91
'i64*8B

	full_text


i64* %68
Istore8B>
<
	full_text/
-
+store i64 %96, i64* %73, align 16, !tbaa !8
%i648B

	full_text
	
i64 %96
'i64*8B

	full_text


i64* %73
Istore8B>
<
	full_text/
-
+store i64 %100, i64* %78, align 8, !tbaa !8
&i648B

	full_text


i64 %100
'i64*8B

	full_text


i64* %78
Jstore8B?
=
	full_text0
.
,store i64 %105, i64* %83, align 16, !tbaa !8
&i648B

	full_text


i64 %105
'i64*8B

	full_text


i64* %83
Istore8B>
<
	full_text/
-
+store i64 %110, i64* %88, align 8, !tbaa !8
&i648B

	full_text


i64 %110
'i64*8B

	full_text


i64* %88
Jstore8B?
=
	full_text0
.
,store i64 %115, i64* %93, align 16, !tbaa !8
&i648B

	full_text


i64 %115
'i64*8B

	full_text


i64* %93
žgetelementptr8BŠ
‡
	full_textz
x
v%166 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Ibitcast8B<
:
	full_text-
+
)%167 = bitcast [5 x double]* %166 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %166
Jload8B@
>
	full_text1
/
-%168 = load i64, i64* %167, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %167
Jstore8B?
=
	full_text0
.
,store i64 %168, i64* %97, align 16, !tbaa !8
&i648B

	full_text


i64 %168
'i64*8B

	full_text


i64* %97
¦getelementptr8B’

	full_text

}%169 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 3, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%170 = bitcast double* %169 to i64*
.double*8B

	full_text

double* %169
Jload8B@
>
	full_text1
/
-%171 = load i64, i64* %170, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %170
Jstore8B?
=
	full_text0
.
,store i64 %171, i64* %102, align 8, !tbaa !8
&i648B

	full_text


i64 %171
(i64*8B

	full_text

	i64* %102
¦getelementptr8B’

	full_text

}%172 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 3, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%173 = bitcast double* %172 to i64*
.double*8B

	full_text

double* %172
Jload8B@
>
	full_text1
/
-%174 = load i64, i64* %173, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %173
Kstore8B@
>
	full_text1
/
-store i64 %174, i64* %107, align 16, !tbaa !8
&i648B

	full_text


i64 %174
(i64*8B

	full_text

	i64* %107
¦getelementptr8B’

	full_text

}%175 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 3, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%176 = bitcast double* %175 to i64*
.double*8B

	full_text

double* %175
Jload8B@
>
	full_text1
/
-%177 = load i64, i64* %176, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %176
Jstore8B?
=
	full_text0
.
,store i64 %177, i64* %112, align 8, !tbaa !8
&i648B

	full_text


i64 %177
(i64*8B

	full_text

	i64* %112
¦getelementptr8B’

	full_text

}%178 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 3, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%179 = bitcast double* %178 to i64*
.double*8B

	full_text

double* %178
Jload8B@
>
	full_text1
/
-%180 = load i64, i64* %179, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %179
Kstore8B@
>
	full_text1
/
-store i64 %180, i64* %117, align 16, !tbaa !8
&i648B

	full_text


i64 %180
(i64*8B

	full_text

	i64* %117
getelementptr8B}
{
	full_textn
l
j%181 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 2
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%182 = load double, double* %181, align 8, !tbaa !8
.double*8B

	full_text

double* %181
getelementptr8B}
{
	full_textn
l
j%183 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 2
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%184 = load double, double* %183, align 8, !tbaa !8
.double*8B

	full_text

double* %183
getelementptr8B}
{
	full_textn
l
j%185 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 2
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%186 = load double, double* %185, align 8, !tbaa !8
.double*8B

	full_text

double* %185
getelementptr8B}
{
	full_textn
l
j%187 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 2
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%188 = load double, double* %187, align 8, !tbaa !8
.double*8B

	full_text

double* %187
getelementptr8B}
{
	full_textn
l
j%189 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 2
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%190 = load double, double* %189, align 8, !tbaa !8
.double*8B

	full_text

double* %189
getelementptr8B}
{
	full_textn
l
j%191 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 2
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%192 = load double, double* %191, align 8, !tbaa !8
.double*8B

	full_text

double* %191
¦getelementptr8B’

	full_text

}%193 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 1, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%194 = load double, double* %193, align 8, !tbaa !8
.double*8B

	full_text

double* %193
@bitcast8B3
1
	full_text$
"
 %195 = bitcast i64 %96 to double
%i648B

	full_text
	
i64 %96
@bitcast8B3
1
	full_text$
"
 %196 = bitcast i64 %71 to double
%i648B

	full_text
	
i64 %71
vcall8Bl
j
	full_text]
[
Y%197 = tail call double @llvm.fmuladd.f64(double %196, double -2.000000e+00, double %195)
,double8B

	full_text

double %196
,double8B

	full_text

double %195
@bitcast8B3
1
	full_text$
"
 %198 = bitcast i64 %46 to double
%i648B

	full_text
	
i64 %46
:fadd8B0
.
	full_text!

%199 = fadd double %197, %198
,double8B

	full_text

double %197
,double8B

	full_text

double %198
{call8Bq
o
	full_textb
`
^%200 = tail call double @llvm.fmuladd.f64(double %199, double 0x40D2FC3000000001, double %194)
,double8B

	full_text

double %199
,double8B

	full_text

double %194
Abitcast8B4
2
	full_text%
#
!%201 = bitcast i64 %100 to double
&i648B

	full_text


i64 %100
@bitcast8B3
1
	full_text$
"
 %202 = bitcast i64 %51 to double
%i648B

	full_text
	
i64 %51
:fsub8B0
.
	full_text!

%203 = fsub double %201, %202
,double8B

	full_text

double %201
,double8B

	full_text

double %202
vcall8Bl
j
	full_text]
[
Y%204 = tail call double @llvm.fmuladd.f64(double %203, double -8.050000e+01, double %200)
,double8B

	full_text

double %203
,double8B

	full_text

double %200
¦getelementptr8B’

	full_text

}%205 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 1, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%206 = load double, double* %205, align 8, !tbaa !8
.double*8B

	full_text

double* %205
@bitcast8B3
1
	full_text$
"
 %207 = bitcast i64 %76 to double
%i648B

	full_text
	
i64 %76
vcall8Bl
j
	full_text]
[
Y%208 = tail call double @llvm.fmuladd.f64(double %207, double -2.000000e+00, double %201)
,double8B

	full_text

double %207
,double8B

	full_text

double %201
:fadd8B0
.
	full_text!

%209 = fadd double %208, %202
,double8B

	full_text

double %208
,double8B

	full_text

double %202
{call8Bq
o
	full_textb
`
^%210 = tail call double @llvm.fmuladd.f64(double %209, double 0x40D2FC3000000001, double %206)
,double8B

	full_text

double %209
,double8B

	full_text

double %206
vcall8Bl
j
	full_text]
[
Y%211 = tail call double @llvm.fmuladd.f64(double %121, double -2.000000e+00, double %182)
,double8B

	full_text

double %121
,double8B

	full_text

double %182
:fadd8B0
.
	full_text!

%212 = fadd double %119, %211
,double8B

	full_text

double %119
,double8B

	full_text

double %211
{call8Bq
o
	full_textb
`
^%213 = tail call double @llvm.fmuladd.f64(double %212, double 0x40AB004444444445, double %210)
,double8B

	full_text

double %212
,double8B

	full_text

double %210
:fmul8B0
.
	full_text!

%214 = fmul double %119, %202
,double8B

	full_text

double %119
,double8B

	full_text

double %202
Cfsub8B9
7
	full_text*
(
&%215 = fsub double -0.000000e+00, %214
,double8B

	full_text

double %214
mcall8Bc
a
	full_textT
R
P%216 = tail call double @llvm.fmuladd.f64(double %201, double %182, double %215)
,double8B

	full_text

double %201
,double8B

	full_text

double %182
,double8B

	full_text

double %215
Abitcast8B4
2
	full_text%
#
!%217 = bitcast i64 %115 to double
&i648B

	full_text


i64 %115
:fsub8B0
.
	full_text!

%218 = fsub double %217, %192
,double8B

	full_text

double %217
,double8B

	full_text

double %192
@bitcast8B3
1
	full_text$
"
 %219 = bitcast i64 %66 to double
%i648B

	full_text
	
i64 %66
:fsub8B0
.
	full_text!

%220 = fsub double %218, %219
,double8B

	full_text

double %218
,double8B

	full_text

double %219
:fadd8B0
.
	full_text!

%221 = fadd double %139, %220
,double8B

	full_text

double %139
,double8B

	full_text

double %220
ucall8Bk
i
	full_text\
Z
X%222 = tail call double @llvm.fmuladd.f64(double %221, double 4.000000e-01, double %216)
,double8B

	full_text

double %221
,double8B

	full_text

double %216
vcall8Bl
j
	full_text]
[
Y%223 = tail call double @llvm.fmuladd.f64(double %222, double -8.050000e+01, double %213)
,double8B

	full_text

double %222
,double8B

	full_text

double %213
¦getelementptr8B’

	full_text

}%224 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 1, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%225 = load double, double* %224, align 8, !tbaa !8
.double*8B

	full_text

double* %224
Abitcast8B4
2
	full_text%
#
!%226 = bitcast i64 %105 to double
&i648B

	full_text


i64 %105
@bitcast8B3
1
	full_text$
"
 %227 = bitcast i64 %81 to double
%i648B

	full_text
	
i64 %81
vcall8Bl
j
	full_text]
[
Y%228 = tail call double @llvm.fmuladd.f64(double %227, double -2.000000e+00, double %226)
,double8B

	full_text

double %227
,double8B

	full_text

double %226
@bitcast8B3
1
	full_text$
"
 %229 = bitcast i64 %56 to double
%i648B

	full_text
	
i64 %56
:fadd8B0
.
	full_text!

%230 = fadd double %228, %229
,double8B

	full_text

double %228
,double8B

	full_text

double %229
{call8Bq
o
	full_textb
`
^%231 = tail call double @llvm.fmuladd.f64(double %230, double 0x40D2FC3000000001, double %225)
,double8B

	full_text

double %230
,double8B

	full_text

double %225
vcall8Bl
j
	full_text]
[
Y%232 = tail call double @llvm.fmuladd.f64(double %125, double -2.000000e+00, double %184)
,double8B

	full_text

double %125
,double8B

	full_text

double %184
:fadd8B0
.
	full_text!

%233 = fadd double %123, %232
,double8B

	full_text

double %123
,double8B

	full_text

double %232
{call8Bq
o
	full_textb
`
^%234 = tail call double @llvm.fmuladd.f64(double %233, double 0x40A4403333333334, double %231)
,double8B

	full_text

double %233
,double8B

	full_text

double %231
:fmul8B0
.
	full_text!

%235 = fmul double %119, %229
,double8B

	full_text

double %119
,double8B

	full_text

double %229
Cfsub8B9
7
	full_text*
(
&%236 = fsub double -0.000000e+00, %235
,double8B

	full_text

double %235
mcall8Bc
a
	full_textT
R
P%237 = tail call double @llvm.fmuladd.f64(double %226, double %182, double %236)
,double8B

	full_text

double %226
,double8B

	full_text

double %182
,double8B

	full_text

double %236
vcall8Bl
j
	full_text]
[
Y%238 = tail call double @llvm.fmuladd.f64(double %237, double -8.050000e+01, double %234)
,double8B

	full_text

double %237
,double8B

	full_text

double %234
¦getelementptr8B’

	full_text

}%239 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 1, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%240 = load double, double* %239, align 8, !tbaa !8
.double*8B

	full_text

double* %239
Abitcast8B4
2
	full_text%
#
!%241 = bitcast i64 %110 to double
&i648B

	full_text


i64 %110
@bitcast8B3
1
	full_text$
"
 %242 = bitcast i64 %86 to double
%i648B

	full_text
	
i64 %86
vcall8Bl
j
	full_text]
[
Y%243 = tail call double @llvm.fmuladd.f64(double %242, double -2.000000e+00, double %241)
,double8B

	full_text

double %242
,double8B

	full_text

double %241
Pload8BF
D
	full_text7
5
3%244 = load double, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
:fadd8B0
.
	full_text!

%245 = fadd double %243, %244
,double8B

	full_text

double %243
,double8B

	full_text

double %244
{call8Bq
o
	full_textb
`
^%246 = tail call double @llvm.fmuladd.f64(double %245, double 0x40D2FC3000000001, double %240)
,double8B

	full_text

double %245
,double8B

	full_text

double %240
vcall8Bl
j
	full_text]
[
Y%247 = tail call double @llvm.fmuladd.f64(double %129, double -2.000000e+00, double %186)
,double8B

	full_text

double %129
,double8B

	full_text

double %186
:fadd8B0
.
	full_text!

%248 = fadd double %127, %247
,double8B

	full_text

double %127
,double8B

	full_text

double %247
{call8Bq
o
	full_textb
`
^%249 = tail call double @llvm.fmuladd.f64(double %248, double 0x40A4403333333334, double %246)
,double8B

	full_text

double %248
,double8B

	full_text

double %246
:fmul8B0
.
	full_text!

%250 = fmul double %119, %244
,double8B

	full_text

double %119
,double8B

	full_text

double %244
Cfsub8B9
7
	full_text*
(
&%251 = fsub double -0.000000e+00, %250
,double8B

	full_text

double %250
mcall8Bc
a
	full_textT
R
P%252 = tail call double @llvm.fmuladd.f64(double %241, double %182, double %251)
,double8B

	full_text

double %241
,double8B

	full_text

double %182
,double8B

	full_text

double %251
vcall8Bl
j
	full_text]
[
Y%253 = tail call double @llvm.fmuladd.f64(double %252, double -8.050000e+01, double %249)
,double8B

	full_text

double %252
,double8B

	full_text

double %249
¦getelementptr8B’

	full_text

}%254 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 1, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%255 = load double, double* %254, align 8, !tbaa !8
.double*8B

	full_text

double* %254
Pload8BF
D
	full_text7
5
3%256 = load double, double* %67, align 16, !tbaa !8
-double*8B

	full_text

double* %67
vcall8Bl
j
	full_text]
[
Y%257 = tail call double @llvm.fmuladd.f64(double %256, double -2.000000e+00, double %217)
,double8B

	full_text

double %256
,double8B

	full_text

double %217
:fadd8B0
.
	full_text!

%258 = fadd double %257, %219
,double8B

	full_text

double %257
,double8B

	full_text

double %219
{call8Bq
o
	full_textb
`
^%259 = tail call double @llvm.fmuladd.f64(double %258, double 0x40D2FC3000000001, double %255)
,double8B

	full_text

double %258
,double8B

	full_text

double %255
vcall8Bl
j
	full_text]
[
Y%260 = tail call double @llvm.fmuladd.f64(double %133, double -2.000000e+00, double %188)
,double8B

	full_text

double %133
,double8B

	full_text

double %188
:fadd8B0
.
	full_text!

%261 = fadd double %131, %260
,double8B

	full_text

double %131
,double8B

	full_text

double %260
{call8Bq
o
	full_textb
`
^%262 = tail call double @llvm.fmuladd.f64(double %261, double 0xC0A370D4FDF3B645, double %259)
,double8B

	full_text

double %261
,double8B

	full_text

double %259
Bfmul8B8
6
	full_text)
'
%%263 = fmul double %121, 2.000000e+00
,double8B

	full_text

double %121
:fmul8B0
.
	full_text!

%264 = fmul double %121, %263
,double8B

	full_text

double %121
,double8B

	full_text

double %263
Cfsub8B9
7
	full_text*
(
&%265 = fsub double -0.000000e+00, %264
,double8B

	full_text

double %264
mcall8Bc
a
	full_textT
R
P%266 = tail call double @llvm.fmuladd.f64(double %182, double %182, double %265)
,double8B

	full_text

double %182
,double8B

	full_text

double %182
,double8B

	full_text

double %265
mcall8Bc
a
	full_textT
R
P%267 = tail call double @llvm.fmuladd.f64(double %119, double %119, double %266)
,double8B

	full_text

double %119
,double8B

	full_text

double %119
,double8B

	full_text

double %266
{call8Bq
o
	full_textb
`
^%268 = tail call double @llvm.fmuladd.f64(double %267, double 0x407B004444444445, double %262)
,double8B

	full_text

double %267
,double8B

	full_text

double %262
Bfmul8B8
6
	full_text)
'
%%269 = fmul double %256, 2.000000e+00
,double8B

	full_text

double %256
:fmul8B0
.
	full_text!

%270 = fmul double %137, %269
,double8B

	full_text

double %137
,double8B

	full_text

double %269
Cfsub8B9
7
	full_text*
(
&%271 = fsub double -0.000000e+00, %270
,double8B

	full_text

double %270
mcall8Bc
a
	full_textT
R
P%272 = tail call double @llvm.fmuladd.f64(double %217, double %190, double %271)
,double8B

	full_text

double %217
,double8B

	full_text

double %190
,double8B

	full_text

double %271
mcall8Bc
a
	full_textT
R
P%273 = tail call double @llvm.fmuladd.f64(double %219, double %135, double %272)
,double8B

	full_text

double %219
,double8B

	full_text

double %135
,double8B

	full_text

double %272
{call8Bq
o
	full_textb
`
^%274 = tail call double @llvm.fmuladd.f64(double %273, double 0x40B3D884189374BC, double %268)
,double8B

	full_text

double %273
,double8B

	full_text

double %268
Bfmul8B8
6
	full_text)
'
%%275 = fmul double %192, 4.000000e-01
,double8B

	full_text

double %192
Cfsub8B9
7
	full_text*
(
&%276 = fsub double -0.000000e+00, %275
,double8B

	full_text

double %275
ucall8Bk
i
	full_text\
Z
X%277 = tail call double @llvm.fmuladd.f64(double %217, double 1.400000e+00, double %276)
,double8B

	full_text

double %217
,double8B

	full_text

double %276
Bfmul8B8
6
	full_text)
'
%%278 = fmul double %139, 4.000000e-01
,double8B

	full_text

double %139
Cfsub8B9
7
	full_text*
(
&%279 = fsub double -0.000000e+00, %278
,double8B

	full_text

double %278
ucall8Bk
i
	full_text\
Z
X%280 = tail call double @llvm.fmuladd.f64(double %219, double 1.400000e+00, double %279)
,double8B

	full_text

double %219
,double8B

	full_text

double %279
:fmul8B0
.
	full_text!

%281 = fmul double %119, %280
,double8B

	full_text

double %119
,double8B

	full_text

double %280
Cfsub8B9
7
	full_text*
(
&%282 = fsub double -0.000000e+00, %281
,double8B

	full_text

double %281
mcall8Bc
a
	full_textT
R
P%283 = tail call double @llvm.fmuladd.f64(double %277, double %182, double %282)
,double8B

	full_text

double %277
,double8B

	full_text

double %182
,double8B

	full_text

double %282
vcall8Bl
j
	full_text]
[
Y%284 = tail call double @llvm.fmuladd.f64(double %283, double -8.050000e+01, double %274)
,double8B

	full_text

double %283
,double8B

	full_text

double %274
Pload8BF
D
	full_text7
5
3%285 = load double, double* %47, align 16, !tbaa !8
-double*8B

	full_text

double* %47
Pload8BF
D
	full_text7
5
3%286 = load double, double* %72, align 16, !tbaa !8
-double*8B

	full_text

double* %72
Bfmul8B8
6
	full_text)
'
%%287 = fmul double %286, 4.000000e+00
,double8B

	full_text

double %286
Cfsub8B9
7
	full_text*
(
&%288 = fsub double -0.000000e+00, %287
,double8B

	full_text

double %287
ucall8Bk
i
	full_text\
Z
X%289 = tail call double @llvm.fmuladd.f64(double %285, double 5.000000e+00, double %288)
,double8B

	full_text

double %285
,double8B

	full_text

double %288
qgetelementptr8B^
\
	full_textO
M
K%290 = getelementptr inbounds [5 x double], [5 x double]* %14, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %14
Qload8BG
E
	full_text8
6
4%291 = load double, double* %290, align 16, !tbaa !8
.double*8B

	full_text

double* %290
:fadd8B0
.
	full_text!

%292 = fadd double %291, %289
,double8B

	full_text

double %291
,double8B

	full_text

double %289
vcall8Bl
j
	full_text]
[
Y%293 = tail call double @llvm.fmuladd.f64(double %292, double -2.500000e-01, double %204)
,double8B

	full_text

double %292
,double8B

	full_text

double %204
Pstore8BE
C
	full_text6
4
2store double %293, double* %193, align 8, !tbaa !8
,double8B

	full_text

double %293
.double*8B

	full_text

double* %193
Oload8BE
C
	full_text6
4
2%294 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
Oload8BE
C
	full_text6
4
2%295 = load double, double* %77, align 8, !tbaa !8
-double*8B

	full_text

double* %77
Bfmul8B8
6
	full_text)
'
%%296 = fmul double %295, 4.000000e+00
,double8B

	full_text

double %295
Cfsub8B9
7
	full_text*
(
&%297 = fsub double -0.000000e+00, %296
,double8B

	full_text

double %296
ucall8Bk
i
	full_text\
Z
X%298 = tail call double @llvm.fmuladd.f64(double %294, double 5.000000e+00, double %297)
,double8B

	full_text

double %294
,double8B

	full_text

double %297
Pload8BF
D
	full_text7
5
3%299 = load double, double* %101, align 8, !tbaa !8
.double*8B

	full_text

double* %101
:fadd8B0
.
	full_text!

%300 = fadd double %299, %298
,double8B

	full_text

double %299
,double8B

	full_text

double %298
vcall8Bl
j
	full_text]
[
Y%301 = tail call double @llvm.fmuladd.f64(double %300, double -2.500000e-01, double %223)
,double8B

	full_text

double %300
,double8B

	full_text

double %223
Pstore8BE
C
	full_text6
4
2store double %301, double* %205, align 8, !tbaa !8
,double8B

	full_text

double %301
.double*8B

	full_text

double* %205
Pload8BF
D
	full_text7
5
3%302 = load double, double* %57, align 16, !tbaa !8
-double*8B

	full_text

double* %57
Pload8BF
D
	full_text7
5
3%303 = load double, double* %82, align 16, !tbaa !8
-double*8B

	full_text

double* %82
Bfmul8B8
6
	full_text)
'
%%304 = fmul double %303, 4.000000e+00
,double8B

	full_text

double %303
Cfsub8B9
7
	full_text*
(
&%305 = fsub double -0.000000e+00, %304
,double8B

	full_text

double %304
ucall8Bk
i
	full_text\
Z
X%306 = tail call double @llvm.fmuladd.f64(double %302, double 5.000000e+00, double %305)
,double8B

	full_text

double %302
,double8B

	full_text

double %305
Qload8BG
E
	full_text8
6
4%307 = load double, double* %106, align 16, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%308 = fadd double %307, %306
,double8B

	full_text

double %307
,double8B

	full_text

double %306
vcall8Bl
j
	full_text]
[
Y%309 = tail call double @llvm.fmuladd.f64(double %308, double -2.500000e-01, double %238)
,double8B

	full_text

double %308
,double8B

	full_text

double %238
Pstore8BE
C
	full_text6
4
2store double %309, double* %224, align 8, !tbaa !8
,double8B

	full_text

double %309
.double*8B

	full_text

double* %224
Oload8BE
C
	full_text6
4
2%310 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
Oload8BE
C
	full_text6
4
2%311 = load double, double* %87, align 8, !tbaa !8
-double*8B

	full_text

double* %87
Bfmul8B8
6
	full_text)
'
%%312 = fmul double %311, 4.000000e+00
,double8B

	full_text

double %311
Cfsub8B9
7
	full_text*
(
&%313 = fsub double -0.000000e+00, %312
,double8B

	full_text

double %312
ucall8Bk
i
	full_text\
Z
X%314 = tail call double @llvm.fmuladd.f64(double %310, double 5.000000e+00, double %313)
,double8B

	full_text

double %310
,double8B

	full_text

double %313
Pload8BF
D
	full_text7
5
3%315 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%316 = fadd double %315, %314
,double8B

	full_text

double %315
,double8B

	full_text

double %314
vcall8Bl
j
	full_text]
[
Y%317 = tail call double @llvm.fmuladd.f64(double %316, double -2.500000e-01, double %253)
,double8B

	full_text

double %316
,double8B

	full_text

double %253
Pstore8BE
C
	full_text6
4
2store double %317, double* %239, align 8, !tbaa !8
,double8B

	full_text

double %317
.double*8B

	full_text

double* %239
Pload8BF
D
	full_text7
5
3%318 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
Bfmul8B8
6
	full_text)
'
%%319 = fmul double %318, 4.000000e+00
,double8B

	full_text

double %318
Cfsub8B9
7
	full_text*
(
&%320 = fsub double -0.000000e+00, %319
,double8B

	full_text

double %319
ucall8Bk
i
	full_text\
Z
X%321 = tail call double @llvm.fmuladd.f64(double %256, double 5.000000e+00, double %320)
,double8B

	full_text

double %256
,double8B

	full_text

double %320
Qload8BG
E
	full_text8
6
4%322 = load double, double* %116, align 16, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%323 = fadd double %322, %321
,double8B

	full_text

double %322
,double8B

	full_text

double %321
vcall8Bl
j
	full_text]
[
Y%324 = tail call double @llvm.fmuladd.f64(double %323, double -2.500000e-01, double %284)
,double8B

	full_text

double %323
,double8B

	full_text

double %284
Pstore8BE
C
	full_text6
4
2store double %324, double* %254, align 8, !tbaa !8
,double8B

	full_text

double %324
.double*8B

	full_text

double* %254
Jstore8B?
=
	full_text0
.
,store i64 %46, i64* %145, align 16, !tbaa !8
%i648B

	full_text
	
i64 %46
(i64*8B

	full_text

	i64* %145
Istore8B>
<
	full_text/
-
+store i64 %51, i64* %150, align 8, !tbaa !8
%i648B

	full_text
	
i64 %51
(i64*8B

	full_text

	i64* %150
Jstore8B?
=
	full_text0
.
,store i64 %56, i64* %155, align 16, !tbaa !8
%i648B

	full_text
	
i64 %56
(i64*8B

	full_text

	i64* %155
Istore8B>
<
	full_text/
-
+store i64 %61, i64* %160, align 8, !tbaa !8
%i648B

	full_text
	
i64 %61
(i64*8B

	full_text

	i64* %160
Jstore8B?
=
	full_text0
.
,store i64 %66, i64* %165, align 16, !tbaa !8
%i648B

	full_text
	
i64 %66
(i64*8B

	full_text

	i64* %165
Jstore8B?
=
	full_text0
.
,store i64 %71, i64* %143, align 16, !tbaa !8
%i648B

	full_text
	
i64 %71
(i64*8B

	full_text

	i64* %143
Istore8B>
<
	full_text/
-
+store i64 %76, i64* %147, align 8, !tbaa !8
%i648B

	full_text
	
i64 %76
(i64*8B

	full_text

	i64* %147
Jstore8B?
=
	full_text0
.
,store i64 %81, i64* %152, align 16, !tbaa !8
%i648B

	full_text
	
i64 %81
(i64*8B

	full_text

	i64* %152
Istore8B>
<
	full_text/
-
+store i64 %86, i64* %157, align 8, !tbaa !8
%i648B

	full_text
	
i64 %86
(i64*8B

	full_text

	i64* %157
Jstore8B?
=
	full_text0
.
,store i64 %91, i64* %162, align 16, !tbaa !8
%i648B

	full_text
	
i64 %91
(i64*8B

	full_text

	i64* %162
Istore8B>
<
	full_text/
-
+store i64 %96, i64* %48, align 16, !tbaa !8
%i648B

	full_text
	
i64 %96
'i64*8B

	full_text


i64* %48
Istore8B>
<
	full_text/
-
+store i64 %100, i64* %53, align 8, !tbaa !8
&i648B

	full_text


i64 %100
'i64*8B

	full_text


i64* %53
Jstore8B?
=
	full_text0
.
,store i64 %105, i64* %58, align 16, !tbaa !8
&i648B

	full_text


i64 %105
'i64*8B

	full_text


i64* %58
Istore8B>
<
	full_text/
-
+store i64 %110, i64* %63, align 8, !tbaa !8
&i648B

	full_text


i64 %110
'i64*8B

	full_text


i64* %63
Jstore8B?
=
	full_text0
.
,store i64 %115, i64* %68, align 16, !tbaa !8
&i648B

	full_text


i64 %115
'i64*8B

	full_text


i64* %68
Jstore8B?
=
	full_text0
.
,store i64 %168, i64* %73, align 16, !tbaa !8
&i648B

	full_text


i64 %168
'i64*8B

	full_text


i64* %73
Istore8B>
<
	full_text/
-
+store i64 %171, i64* %78, align 8, !tbaa !8
&i648B

	full_text


i64 %171
'i64*8B

	full_text


i64* %78
Jstore8B?
=
	full_text0
.
,store i64 %174, i64* %83, align 16, !tbaa !8
&i648B

	full_text


i64 %174
'i64*8B

	full_text


i64* %83
Istore8B>
<
	full_text/
-
+store i64 %177, i64* %88, align 8, !tbaa !8
&i648B

	full_text


i64 %177
'i64*8B

	full_text


i64* %88
Jstore8B?
=
	full_text0
.
,store i64 %180, i64* %93, align 16, !tbaa !8
&i648B

	full_text


i64 %180
'i64*8B

	full_text


i64* %93
žgetelementptr8BŠ
‡
	full_textz
x
v%325 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Ibitcast8B<
:
	full_text-
+
)%326 = bitcast [5 x double]* %325 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %325
Jload8B@
>
	full_text1
/
-%327 = load i64, i64* %326, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %326
Jstore8B?
=
	full_text0
.
,store i64 %327, i64* %97, align 16, !tbaa !8
&i648B

	full_text


i64 %327
'i64*8B

	full_text


i64* %97
¦getelementptr8B’

	full_text

}%328 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 4, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%329 = bitcast double* %328 to i64*
.double*8B

	full_text

double* %328
Jload8B@
>
	full_text1
/
-%330 = load i64, i64* %329, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %329
Jstore8B?
=
	full_text0
.
,store i64 %330, i64* %102, align 8, !tbaa !8
&i648B

	full_text


i64 %330
(i64*8B

	full_text

	i64* %102
¦getelementptr8B’

	full_text

}%331 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 4, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%332 = bitcast double* %331 to i64*
.double*8B

	full_text

double* %331
Jload8B@
>
	full_text1
/
-%333 = load i64, i64* %332, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %332
Kstore8B@
>
	full_text1
/
-store i64 %333, i64* %107, align 16, !tbaa !8
&i648B

	full_text


i64 %333
(i64*8B

	full_text

	i64* %107
¦getelementptr8B’

	full_text

}%334 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 4, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%335 = bitcast double* %334 to i64*
.double*8B

	full_text

double* %334
Jload8B@
>
	full_text1
/
-%336 = load i64, i64* %335, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %335
Jstore8B?
=
	full_text0
.
,store i64 %336, i64* %112, align 8, !tbaa !8
&i648B

	full_text


i64 %336
(i64*8B

	full_text

	i64* %112
¦getelementptr8B’

	full_text

}%337 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 4, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Cbitcast8B6
4
	full_text'
%
#%338 = bitcast double* %337 to i64*
.double*8B

	full_text

double* %337
Jload8B@
>
	full_text1
/
-%339 = load i64, i64* %338, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %338
Kstore8B@
>
	full_text1
/
-store i64 %339, i64* %117, align 16, !tbaa !8
&i648B

	full_text


i64 %339
(i64*8B

	full_text

	i64* %117
getelementptr8B}
{
	full_textn
l
j%340 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 3
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%341 = load double, double* %340, align 8, !tbaa !8
.double*8B

	full_text

double* %340
getelementptr8B}
{
	full_textn
l
j%342 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 3
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%343 = load double, double* %342, align 8, !tbaa !8
.double*8B

	full_text

double* %342
getelementptr8B}
{
	full_textn
l
j%344 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 3
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%345 = load double, double* %344, align 8, !tbaa !8
.double*8B

	full_text

double* %344
getelementptr8B}
{
	full_textn
l
j%346 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 3
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%347 = load double, double* %346, align 8, !tbaa !8
.double*8B

	full_text

double* %346
getelementptr8B}
{
	full_textn
l
j%348 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 3
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%349 = load double, double* %348, align 8, !tbaa !8
.double*8B

	full_text

double* %348
getelementptr8B}
{
	full_textn
l
j%350 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 3
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%351 = load double, double* %350, align 8, !tbaa !8
.double*8B

	full_text

double* %350
¦getelementptr8B’

	full_text

}%352 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 2, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%353 = load double, double* %352, align 8, !tbaa !8
.double*8B

	full_text

double* %352
Abitcast8B4
2
	full_text%
#
!%354 = bitcast i64 %168 to double
&i648B

	full_text


i64 %168
vcall8Bl
j
	full_text]
[
Y%355 = tail call double @llvm.fmuladd.f64(double %195, double -2.000000e+00, double %354)
,double8B

	full_text

double %195
,double8B

	full_text

double %354
:fadd8B0
.
	full_text!

%356 = fadd double %355, %196
,double8B

	full_text

double %355
,double8B

	full_text

double %196
{call8Bq
o
	full_textb
`
^%357 = tail call double @llvm.fmuladd.f64(double %356, double 0x40D2FC3000000001, double %353)
,double8B

	full_text

double %356
,double8B

	full_text

double %353
Abitcast8B4
2
	full_text%
#
!%358 = bitcast i64 %171 to double
&i648B

	full_text


i64 %171
:fsub8B0
.
	full_text!

%359 = fsub double %358, %207
,double8B

	full_text

double %358
,double8B

	full_text

double %207
vcall8Bl
j
	full_text]
[
Y%360 = tail call double @llvm.fmuladd.f64(double %359, double -8.050000e+01, double %357)
,double8B

	full_text

double %359
,double8B

	full_text

double %357
¦getelementptr8B’

	full_text

}%361 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 2, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%362 = load double, double* %361, align 8, !tbaa !8
.double*8B

	full_text

double* %361
vcall8Bl
j
	full_text]
[
Y%363 = tail call double @llvm.fmuladd.f64(double %201, double -2.000000e+00, double %358)
,double8B

	full_text

double %201
,double8B

	full_text

double %358
:fadd8B0
.
	full_text!

%364 = fadd double %363, %207
,double8B

	full_text

double %363
,double8B

	full_text

double %207
{call8Bq
o
	full_textb
`
^%365 = tail call double @llvm.fmuladd.f64(double %364, double 0x40D2FC3000000001, double %362)
,double8B

	full_text

double %364
,double8B

	full_text

double %362
vcall8Bl
j
	full_text]
[
Y%366 = tail call double @llvm.fmuladd.f64(double %182, double -2.000000e+00, double %341)
,double8B

	full_text

double %182
,double8B

	full_text

double %341
:fadd8B0
.
	full_text!

%367 = fadd double %121, %366
,double8B

	full_text

double %121
,double8B

	full_text

double %366
{call8Bq
o
	full_textb
`
^%368 = tail call double @llvm.fmuladd.f64(double %367, double 0x40AB004444444445, double %365)
,double8B

	full_text

double %367
,double8B

	full_text

double %365
:fmul8B0
.
	full_text!

%369 = fmul double %121, %207
,double8B

	full_text

double %121
,double8B

	full_text

double %207
Cfsub8B9
7
	full_text*
(
&%370 = fsub double -0.000000e+00, %369
,double8B

	full_text

double %369
mcall8Bc
a
	full_textT
R
P%371 = tail call double @llvm.fmuladd.f64(double %358, double %341, double %370)
,double8B

	full_text

double %358
,double8B

	full_text

double %341
,double8B

	full_text

double %370
Abitcast8B4
2
	full_text%
#
!%372 = bitcast i64 %180 to double
&i648B

	full_text


i64 %180
:fsub8B0
.
	full_text!

%373 = fsub double %372, %351
,double8B

	full_text

double %372
,double8B

	full_text

double %351
@bitcast8B3
1
	full_text$
"
 %374 = bitcast i64 %91 to double
%i648B

	full_text
	
i64 %91
:fsub8B0
.
	full_text!

%375 = fsub double %373, %374
,double8B

	full_text

double %373
,double8B

	full_text

double %374
:fadd8B0
.
	full_text!

%376 = fadd double %141, %375
,double8B

	full_text

double %141
,double8B

	full_text

double %375
ucall8Bk
i
	full_text\
Z
X%377 = tail call double @llvm.fmuladd.f64(double %376, double 4.000000e-01, double %371)
,double8B

	full_text

double %376
,double8B

	full_text

double %371
vcall8Bl
j
	full_text]
[
Y%378 = tail call double @llvm.fmuladd.f64(double %377, double -8.050000e+01, double %368)
,double8B

	full_text

double %377
,double8B

	full_text

double %368
¦getelementptr8B’

	full_text

}%379 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 2, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%380 = load double, double* %379, align 8, !tbaa !8
.double*8B

	full_text

double* %379
Abitcast8B4
2
	full_text%
#
!%381 = bitcast i64 %174 to double
&i648B

	full_text


i64 %174
vcall8Bl
j
	full_text]
[
Y%382 = tail call double @llvm.fmuladd.f64(double %226, double -2.000000e+00, double %381)
,double8B

	full_text

double %226
,double8B

	full_text

double %381
:fadd8B0
.
	full_text!

%383 = fadd double %382, %227
,double8B

	full_text

double %382
,double8B

	full_text

double %227
{call8Bq
o
	full_textb
`
^%384 = tail call double @llvm.fmuladd.f64(double %383, double 0x40D2FC3000000001, double %380)
,double8B

	full_text

double %383
,double8B

	full_text

double %380
vcall8Bl
j
	full_text]
[
Y%385 = tail call double @llvm.fmuladd.f64(double %184, double -2.000000e+00, double %343)
,double8B

	full_text

double %184
,double8B

	full_text

double %343
:fadd8B0
.
	full_text!

%386 = fadd double %125, %385
,double8B

	full_text

double %125
,double8B

	full_text

double %385
{call8Bq
o
	full_textb
`
^%387 = tail call double @llvm.fmuladd.f64(double %386, double 0x40A4403333333334, double %384)
,double8B

	full_text

double %386
,double8B

	full_text

double %384
:fmul8B0
.
	full_text!

%388 = fmul double %121, %227
,double8B

	full_text

double %121
,double8B

	full_text

double %227
Cfsub8B9
7
	full_text*
(
&%389 = fsub double -0.000000e+00, %388
,double8B

	full_text

double %388
mcall8Bc
a
	full_textT
R
P%390 = tail call double @llvm.fmuladd.f64(double %381, double %341, double %389)
,double8B

	full_text

double %381
,double8B

	full_text

double %341
,double8B

	full_text

double %389
vcall8Bl
j
	full_text]
[
Y%391 = tail call double @llvm.fmuladd.f64(double %390, double -8.050000e+01, double %387)
,double8B

	full_text

double %390
,double8B

	full_text

double %387
¦getelementptr8B’

	full_text

}%392 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 2, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%393 = load double, double* %392, align 8, !tbaa !8
.double*8B

	full_text

double* %392
Abitcast8B4
2
	full_text%
#
!%394 = bitcast i64 %177 to double
&i648B

	full_text


i64 %177
vcall8Bl
j
	full_text]
[
Y%395 = tail call double @llvm.fmuladd.f64(double %241, double -2.000000e+00, double %394)
,double8B

	full_text

double %241
,double8B

	full_text

double %394
:fadd8B0
.
	full_text!

%396 = fadd double %395, %242
,double8B

	full_text

double %395
,double8B

	full_text

double %242
{call8Bq
o
	full_textb
`
^%397 = tail call double @llvm.fmuladd.f64(double %396, double 0x40D2FC3000000001, double %393)
,double8B

	full_text

double %396
,double8B

	full_text

double %393
vcall8Bl
j
	full_text]
[
Y%398 = tail call double @llvm.fmuladd.f64(double %186, double -2.000000e+00, double %345)
,double8B

	full_text

double %186
,double8B

	full_text

double %345
:fadd8B0
.
	full_text!

%399 = fadd double %129, %398
,double8B

	full_text

double %129
,double8B

	full_text

double %398
{call8Bq
o
	full_textb
`
^%400 = tail call double @llvm.fmuladd.f64(double %399, double 0x40A4403333333334, double %397)
,double8B

	full_text

double %399
,double8B

	full_text

double %397
:fmul8B0
.
	full_text!

%401 = fmul double %121, %242
,double8B

	full_text

double %121
,double8B

	full_text

double %242
Cfsub8B9
7
	full_text*
(
&%402 = fsub double -0.000000e+00, %401
,double8B

	full_text

double %401
mcall8Bc
a
	full_textT
R
P%403 = tail call double @llvm.fmuladd.f64(double %394, double %341, double %402)
,double8B

	full_text

double %394
,double8B

	full_text

double %341
,double8B

	full_text

double %402
vcall8Bl
j
	full_text]
[
Y%404 = tail call double @llvm.fmuladd.f64(double %403, double -8.050000e+01, double %400)
,double8B

	full_text

double %403
,double8B

	full_text

double %400
¦getelementptr8B’

	full_text

}%405 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 2, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
Pload8BF
D
	full_text7
5
3%406 = load double, double* %405, align 8, !tbaa !8
.double*8B

	full_text

double* %405
vcall8Bl
j
	full_text]
[
Y%407 = tail call double @llvm.fmuladd.f64(double %217, double -2.000000e+00, double %372)
,double8B

	full_text

double %217
,double8B

	full_text

double %372
:fadd8B0
.
	full_text!

%408 = fadd double %407, %374
,double8B

	full_text

double %407
,double8B

	full_text

double %374
{call8Bq
o
	full_textb
`
^%409 = tail call double @llvm.fmuladd.f64(double %408, double 0x40D2FC3000000001, double %406)
,double8B

	full_text

double %408
,double8B

	full_text

double %406
vcall8Bl
j
	full_text]
[
Y%410 = tail call double @llvm.fmuladd.f64(double %188, double -2.000000e+00, double %347)
,double8B

	full_text

double %188
,double8B

	full_text

double %347
:fadd8B0
.
	full_text!

%411 = fadd double %133, %410
,double8B

	full_text

double %133
,double8B

	full_text

double %410
{call8Bq
o
	full_textb
`
^%412 = tail call double @llvm.fmuladd.f64(double %411, double 0xC0A370D4FDF3B645, double %409)
,double8B

	full_text

double %411
,double8B

	full_text

double %409
Bfmul8B8
6
	full_text)
'
%%413 = fmul double %182, 2.000000e+00
,double8B

	full_text

double %182
:fmul8B0
.
	full_text!

%414 = fmul double %182, %413
,double8B

	full_text

double %182
,double8B

	full_text

double %413
Cfsub8B9
7
	full_text*
(
&%415 = fsub double -0.000000e+00, %414
,double8B

	full_text

double %414
mcall8Bc
a
	full_textT
R
P%416 = tail call double @llvm.fmuladd.f64(double %341, double %341, double %415)
,double8B

	full_text

double %341
,double8B

	full_text

double %341
,double8B

	full_text

double %415
mcall8Bc
a
	full_textT
R
P%417 = tail call double @llvm.fmuladd.f64(double %121, double %121, double %416)
,double8B

	full_text

double %121
,double8B

	full_text

double %121
,double8B

	full_text

double %416
{call8Bq
o
	full_textb
`
^%418 = tail call double @llvm.fmuladd.f64(double %417, double 0x407B004444444445, double %412)
,double8B

	full_text

double %417
,double8B

	full_text

double %412
Bfmul8B8
6
	full_text)
'
%%419 = fmul double %217, 2.000000e+00
,double8B

	full_text

double %217
:fmul8B0
.
	full_text!

%420 = fmul double %190, %419
,double8B

	full_text

double %190
,double8B

	full_text

double %419
Cfsub8B9
7
	full_text*
(
&%421 = fsub double -0.000000e+00, %420
,double8B

	full_text

double %420
mcall8Bc
a
	full_textT
R
P%422 = tail call double @llvm.fmuladd.f64(double %372, double %349, double %421)
,double8B

	full_text

double %372
,double8B

	full_text

double %349
,double8B

	full_text

double %421
mcall8Bc
a
	full_textT
R
P%423 = tail call double @llvm.fmuladd.f64(double %374, double %137, double %422)
,double8B

	full_text

double %374
,double8B

	full_text

double %137
,double8B

	full_text

double %422
{call8Bq
o
	full_textb
`
^%424 = tail call double @llvm.fmuladd.f64(double %423, double 0x40B3D884189374BC, double %418)
,double8B

	full_text

double %423
,double8B

	full_text

double %418
Bfmul8B8
6
	full_text)
'
%%425 = fmul double %351, 4.000000e-01
,double8B

	full_text

double %351
Cfsub8B9
7
	full_text*
(
&%426 = fsub double -0.000000e+00, %425
,double8B

	full_text

double %425
ucall8Bk
i
	full_text\
Z
X%427 = tail call double @llvm.fmuladd.f64(double %372, double 1.400000e+00, double %426)
,double8B

	full_text

double %372
,double8B

	full_text

double %426
Bfmul8B8
6
	full_text)
'
%%428 = fmul double %141, 4.000000e-01
,double8B

	full_text

double %141
Cfsub8B9
7
	full_text*
(
&%429 = fsub double -0.000000e+00, %428
,double8B

	full_text

double %428
ucall8Bk
i
	full_text\
Z
X%430 = tail call double @llvm.fmuladd.f64(double %374, double 1.400000e+00, double %429)
,double8B

	full_text

double %374
,double8B

	full_text

double %429
:fmul8B0
.
	full_text!

%431 = fmul double %121, %430
,double8B

	full_text

double %121
,double8B

	full_text

double %430
Cfsub8B9
7
	full_text*
(
&%432 = fsub double -0.000000e+00, %431
,double8B

	full_text

double %431
mcall8Bc
a
	full_textT
R
P%433 = tail call double @llvm.fmuladd.f64(double %427, double %341, double %432)
,double8B

	full_text

double %427
,double8B

	full_text

double %341
,double8B

	full_text

double %432
vcall8Bl
j
	full_text]
[
Y%434 = tail call double @llvm.fmuladd.f64(double %433, double -8.050000e+01, double %424)
,double8B

	full_text

double %433
,double8B

	full_text

double %424
Qload8BG
E
	full_text8
6
4%435 = load double, double* %142, align 16, !tbaa !8
.double*8B

	full_text

double* %142
Pload8BF
D
	full_text7
5
3%436 = load double, double* %47, align 16, !tbaa !8
-double*8B

	full_text

double* %47
Bfmul8B8
6
	full_text)
'
%%437 = fmul double %436, 6.000000e+00
,double8B

	full_text

double %436
vcall8Bl
j
	full_text]
[
Y%438 = tail call double @llvm.fmuladd.f64(double %435, double -4.000000e+00, double %437)
,double8B

	full_text

double %435
,double8B

	full_text

double %437
Pload8BF
D
	full_text7
5
3%439 = load double, double* %72, align 16, !tbaa !8
-double*8B

	full_text

double* %72
vcall8Bl
j
	full_text]
[
Y%440 = tail call double @llvm.fmuladd.f64(double %439, double -4.000000e+00, double %438)
,double8B

	full_text

double %439
,double8B

	full_text

double %438
Qload8BG
E
	full_text8
6
4%441 = load double, double* %290, align 16, !tbaa !8
.double*8B

	full_text

double* %290
:fadd8B0
.
	full_text!

%442 = fadd double %441, %440
,double8B

	full_text

double %441
,double8B

	full_text

double %440
vcall8Bl
j
	full_text]
[
Y%443 = tail call double @llvm.fmuladd.f64(double %442, double -2.500000e-01, double %360)
,double8B

	full_text

double %442
,double8B

	full_text

double %360
Pstore8BE
C
	full_text6
4
2store double %443, double* %352, align 8, !tbaa !8
,double8B

	full_text

double %443
.double*8B

	full_text

double* %352
Pload8BF
D
	full_text7
5
3%444 = load double, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
Oload8BE
C
	full_text6
4
2%445 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
Bfmul8B8
6
	full_text)
'
%%446 = fmul double %445, 6.000000e+00
,double8B

	full_text

double %445
vcall8Bl
j
	full_text]
[
Y%447 = tail call double @llvm.fmuladd.f64(double %444, double -4.000000e+00, double %446)
,double8B

	full_text

double %444
,double8B

	full_text

double %446
Oload8BE
C
	full_text6
4
2%448 = load double, double* %77, align 8, !tbaa !8
-double*8B

	full_text

double* %77
vcall8Bl
j
	full_text]
[
Y%449 = tail call double @llvm.fmuladd.f64(double %448, double -4.000000e+00, double %447)
,double8B

	full_text

double %448
,double8B

	full_text

double %447
Pload8BF
D
	full_text7
5
3%450 = load double, double* %101, align 8, !tbaa !8
.double*8B

	full_text

double* %101
:fadd8B0
.
	full_text!

%451 = fadd double %450, %449
,double8B

	full_text

double %450
,double8B

	full_text

double %449
vcall8Bl
j
	full_text]
[
Y%452 = tail call double @llvm.fmuladd.f64(double %451, double -2.500000e-01, double %378)
,double8B

	full_text

double %451
,double8B

	full_text

double %378
Pstore8BE
C
	full_text6
4
2store double %452, double* %361, align 8, !tbaa !8
,double8B

	full_text

double %452
.double*8B

	full_text

double* %361
Qload8BG
E
	full_text8
6
4%453 = load double, double* %151, align 16, !tbaa !8
.double*8B

	full_text

double* %151
Pload8BF
D
	full_text7
5
3%454 = load double, double* %57, align 16, !tbaa !8
-double*8B

	full_text

double* %57
Bfmul8B8
6
	full_text)
'
%%455 = fmul double %454, 6.000000e+00
,double8B

	full_text

double %454
vcall8Bl
j
	full_text]
[
Y%456 = tail call double @llvm.fmuladd.f64(double %453, double -4.000000e+00, double %455)
,double8B

	full_text

double %453
,double8B

	full_text

double %455
Pload8BF
D
	full_text7
5
3%457 = load double, double* %82, align 16, !tbaa !8
-double*8B

	full_text

double* %82
vcall8Bl
j
	full_text]
[
Y%458 = tail call double @llvm.fmuladd.f64(double %457, double -4.000000e+00, double %456)
,double8B

	full_text

double %457
,double8B

	full_text

double %456
Qload8BG
E
	full_text8
6
4%459 = load double, double* %106, align 16, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%460 = fadd double %459, %458
,double8B

	full_text

double %459
,double8B

	full_text

double %458
vcall8Bl
j
	full_text]
[
Y%461 = tail call double @llvm.fmuladd.f64(double %460, double -2.500000e-01, double %391)
,double8B

	full_text

double %460
,double8B

	full_text

double %391
Pstore8BE
C
	full_text6
4
2store double %461, double* %379, align 8, !tbaa !8
,double8B

	full_text

double %461
.double*8B

	full_text

double* %379
Pload8BF
D
	full_text7
5
3%462 = load double, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
Oload8BE
C
	full_text6
4
2%463 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
Bfmul8B8
6
	full_text)
'
%%464 = fmul double %463, 6.000000e+00
,double8B

	full_text

double %463
vcall8Bl
j
	full_text]
[
Y%465 = tail call double @llvm.fmuladd.f64(double %462, double -4.000000e+00, double %464)
,double8B

	full_text

double %462
,double8B

	full_text

double %464
Oload8BE
C
	full_text6
4
2%466 = load double, double* %87, align 8, !tbaa !8
-double*8B

	full_text

double* %87
vcall8Bl
j
	full_text]
[
Y%467 = tail call double @llvm.fmuladd.f64(double %466, double -4.000000e+00, double %465)
,double8B

	full_text

double %466
,double8B

	full_text

double %465
Pload8BF
D
	full_text7
5
3%468 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%469 = fadd double %468, %467
,double8B

	full_text

double %468
,double8B

	full_text

double %467
vcall8Bl
j
	full_text]
[
Y%470 = tail call double @llvm.fmuladd.f64(double %469, double -2.500000e-01, double %404)
,double8B

	full_text

double %469
,double8B

	full_text

double %404
Pstore8BE
C
	full_text6
4
2store double %470, double* %392, align 8, !tbaa !8
,double8B

	full_text

double %470
.double*8B

	full_text

double* %392
Qload8BG
E
	full_text8
6
4%471 = load double, double* %161, align 16, !tbaa !8
.double*8B

	full_text

double* %161
Bfmul8B8
6
	full_text)
'
%%472 = fmul double %217, 6.000000e+00
,double8B

	full_text

double %217
vcall8Bl
j
	full_text]
[
Y%473 = tail call double @llvm.fmuladd.f64(double %471, double -4.000000e+00, double %472)
,double8B

	full_text

double %471
,double8B

	full_text

double %472
Pload8BF
D
	full_text7
5
3%474 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
vcall8Bl
j
	full_text]
[
Y%475 = tail call double @llvm.fmuladd.f64(double %474, double -4.000000e+00, double %473)
,double8B

	full_text

double %474
,double8B

	full_text

double %473
Qload8BG
E
	full_text8
6
4%476 = load double, double* %116, align 16, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%477 = fadd double %476, %475
,double8B

	full_text

double %476
,double8B

	full_text

double %475
vcall8Bl
j
	full_text]
[
Y%478 = tail call double @llvm.fmuladd.f64(double %477, double -2.500000e-01, double %434)
,double8B

	full_text

double %477
,double8B

	full_text

double %434
Pstore8BE
C
	full_text6
4
2store double %478, double* %405, align 8, !tbaa !8
,double8B

	full_text

double %478
.double*8B

	full_text

double* %405
6icmp8B,
*
	full_text

%479 = icmp slt i32 %8, 5
Abitcast8B4
2
	full_text%
#
!%480 = bitcast double %435 to i64
,double8B

	full_text

double %435
Abitcast8B4
2
	full_text%
#
!%481 = bitcast double %444 to i64
,double8B

	full_text

double %444
Abitcast8B4
2
	full_text%
#
!%482 = bitcast double %453 to i64
,double8B

	full_text

double %453
Abitcast8B4
2
	full_text%
#
!%483 = bitcast double %462 to i64
,double8B

	full_text

double %462
Abitcast8B4
2
	full_text%
#
!%484 = bitcast double %471 to i64
,double8B

	full_text

double %471
Abitcast8B4
2
	full_text%
#
!%485 = bitcast double %463 to i64
,double8B

	full_text

double %463
Abitcast8B4
2
	full_text%
#
!%486 = bitcast double %474 to i64
,double8B

	full_text

double %474
1add8B(
&
	full_text

%487 = add i32 %8, -1
=br8B5
3
	full_text&
$
"br i1 %479, label %488, label %490
$i18B

	full_text
	
i1 %479
qgetelementptr8B^
\
	full_textO
M
K%489 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
(br8B 

	full_text

br label %685
qgetelementptr8B^
\
	full_textO
M
K%491 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
qgetelementptr8B^
\
	full_textO
M
K%492 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
qgetelementptr8B^
\
	full_textO
M
K%493 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
qgetelementptr8B^
\
	full_textO
M
K%494 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
8zext8B.
,
	full_text

%495 = zext i32 %487 to i64
&i328B

	full_text


i32 %487
(br8B 

	full_text

br label %496
Lphi8BC
A
	full_text4
2
0%497 = phi double [ %672, %496 ], [ %476, %490 ]
,double8B

	full_text

double %672
,double8B

	full_text

double %476
Lphi8BC
A
	full_text4
2
0%498 = phi double [ %664, %496 ], [ %468, %490 ]
,double8B

	full_text

double %664
,double8B

	full_text

double %468
Lphi8BC
A
	full_text4
2
0%499 = phi double [ %657, %496 ], [ %459, %490 ]
,double8B

	full_text

double %657
,double8B

	full_text

double %459
Lphi8BC
A
	full_text4
2
0%500 = phi double [ %650, %496 ], [ %450, %490 ]
,double8B

	full_text

double %650
,double8B

	full_text

double %450
Lphi8BC
A
	full_text4
2
0%501 = phi double [ %643, %496 ], [ %441, %490 ]
,double8B

	full_text

double %643
,double8B

	full_text

double %441
Iphi8B@
>
	full_text1
/
-%502 = phi i64 [ %683, %496 ], [ %486, %490 ]
&i648B

	full_text


i64 %683
&i648B

	full_text


i64 %486
Lphi8BC
A
	full_text4
2
0%503 = phi double [ %498, %496 ], [ %466, %490 ]
,double8B

	full_text

double %498
,double8B

	full_text

double %466
Lphi8BC
A
	full_text4
2
0%504 = phi double [ %499, %496 ], [ %457, %490 ]
,double8B

	full_text

double %499
,double8B

	full_text

double %457
Lphi8BC
A
	full_text4
2
0%505 = phi double [ %500, %496 ], [ %448, %490 ]
,double8B

	full_text

double %500
,double8B

	full_text

double %448
Lphi8BC
A
	full_text4
2
0%506 = phi double [ %501, %496 ], [ %439, %490 ]
,double8B

	full_text

double %501
,double8B

	full_text

double %439
Iphi8B@
>
	full_text1
/
-%507 = phi i64 [ %682, %496 ], [ %115, %490 ]
&i648B

	full_text


i64 %682
&i648B

	full_text


i64 %115
Iphi8B@
>
	full_text1
/
-%508 = phi i64 [ %681, %496 ], [ %485, %490 ]
&i648B

	full_text


i64 %681
&i648B

	full_text


i64 %485
Lphi8BC
A
	full_text4
2
0%509 = phi double [ %504, %496 ], [ %454, %490 ]
,double8B

	full_text

double %504
,double8B

	full_text

double %454
Lphi8BC
A
	full_text4
2
0%510 = phi double [ %505, %496 ], [ %445, %490 ]
,double8B

	full_text

double %505
,double8B

	full_text

double %445
Lphi8BC
A
	full_text4
2
0%511 = phi double [ %506, %496 ], [ %436, %490 ]
,double8B

	full_text

double %506
,double8B

	full_text

double %436
Iphi8B@
>
	full_text1
/
-%512 = phi i64 [ %680, %496 ], [ %484, %490 ]
&i648B

	full_text


i64 %680
&i648B

	full_text


i64 %484
Iphi8B@
>
	full_text1
/
-%513 = phi i64 [ %679, %496 ], [ %483, %490 ]
&i648B

	full_text


i64 %679
&i648B

	full_text


i64 %483
Iphi8B@
>
	full_text1
/
-%514 = phi i64 [ %678, %496 ], [ %482, %490 ]
&i648B

	full_text


i64 %678
&i648B

	full_text


i64 %482
Iphi8B@
>
	full_text1
/
-%515 = phi i64 [ %677, %496 ], [ %481, %490 ]
&i648B

	full_text


i64 %677
&i648B

	full_text


i64 %481
Iphi8B@
>
	full_text1
/
-%516 = phi i64 [ %676, %496 ], [ %480, %490 ]
&i648B

	full_text


i64 %676
&i648B

	full_text


i64 %480
Fphi8B=
;
	full_text.
,
*%517 = phi i64 [ %546, %496 ], [ 3, %490 ]
&i648B

	full_text


i64 %546
Lphi8BC
A
	full_text4
2
0%518 = phi double [ %519, %496 ], [ %182, %490 ]
,double8B

	full_text

double %519
,double8B

	full_text

double %182
Lphi8BC
A
	full_text4
2
0%519 = phi double [ %548, %496 ], [ %341, %490 ]
,double8B

	full_text

double %548
,double8B

	full_text

double %341
Lphi8BC
A
	full_text4
2
0%520 = phi double [ %521, %496 ], [ %184, %490 ]
,double8B

	full_text

double %521
,double8B

	full_text

double %184
Lphi8BC
A
	full_text4
2
0%521 = phi double [ %550, %496 ], [ %343, %490 ]
,double8B

	full_text

double %550
,double8B

	full_text

double %343
Lphi8BC
A
	full_text4
2
0%522 = phi double [ %558, %496 ], [ %351, %490 ]
,double8B

	full_text

double %558
,double8B

	full_text

double %351
Lphi8BC
A
	full_text4
2
0%523 = phi double [ %522, %496 ], [ %192, %490 ]
,double8B

	full_text

double %522
,double8B

	full_text

double %192
Lphi8BC
A
	full_text4
2
0%524 = phi double [ %556, %496 ], [ %349, %490 ]
,double8B

	full_text

double %556
,double8B

	full_text

double %349
Lphi8BC
A
	full_text4
2
0%525 = phi double [ %524, %496 ], [ %190, %490 ]
,double8B

	full_text

double %524
,double8B

	full_text

double %190
Lphi8BC
A
	full_text4
2
0%526 = phi double [ %554, %496 ], [ %347, %490 ]
,double8B

	full_text

double %554
,double8B

	full_text

double %347
Lphi8BC
A
	full_text4
2
0%527 = phi double [ %526, %496 ], [ %188, %490 ]
,double8B

	full_text

double %526
,double8B

	full_text

double %188
Lphi8BC
A
	full_text4
2
0%528 = phi double [ %552, %496 ], [ %345, %490 ]
,double8B

	full_text

double %552
,double8B

	full_text

double %345
Lphi8BC
A
	full_text4
2
0%529 = phi double [ %528, %496 ], [ %186, %490 ]
,double8B

	full_text

double %528
,double8B

	full_text

double %186
Kstore8B@
>
	full_text1
/
-store i64 %516, i64* %145, align 16, !tbaa !8
&i648B

	full_text


i64 %516
(i64*8B

	full_text

	i64* %145
Jstore8B?
=
	full_text0
.
,store i64 %515, i64* %150, align 8, !tbaa !8
&i648B

	full_text


i64 %515
(i64*8B

	full_text

	i64* %150
Kstore8B@
>
	full_text1
/
-store i64 %514, i64* %155, align 16, !tbaa !8
&i648B

	full_text


i64 %514
(i64*8B

	full_text

	i64* %155
Jstore8B?
=
	full_text0
.
,store i64 %513, i64* %160, align 8, !tbaa !8
&i648B

	full_text


i64 %513
(i64*8B

	full_text

	i64* %160
Kstore8B@
>
	full_text1
/
-store i64 %512, i64* %165, align 16, !tbaa !8
&i648B

	full_text


i64 %512
(i64*8B

	full_text

	i64* %165
Jstore8B?
=
	full_text0
.
,store i64 %508, i64* %157, align 8, !tbaa !8
&i648B

	full_text


i64 %508
(i64*8B

	full_text

	i64* %157
Kstore8B@
>
	full_text1
/
-store i64 %507, i64* %162, align 16, !tbaa !8
&i648B

	full_text


i64 %507
(i64*8B

	full_text

	i64* %162
Jstore8B?
=
	full_text0
.
,store i64 %502, i64* %68, align 16, !tbaa !8
&i648B

	full_text


i64 %502
'i64*8B

	full_text


i64* %68
:add8B1
/
	full_text"
 
%530 = add nuw nsw i64 %517, 2
&i648B

	full_text


i64 %517
¡getelementptr8B
Š
	full_text}
{
y%531 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %530
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %530
Ibitcast8B<
:
	full_text-
+
)%532 = bitcast [5 x double]* %531 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %531
Jload8B@
>
	full_text1
/
-%533 = load i64, i64* %532, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %532
Jstore8B?
=
	full_text0
.
,store i64 %533, i64* %97, align 16, !tbaa !8
&i648B

	full_text


i64 %533
'i64*8B

	full_text


i64* %97
«getelementptr8B—
”
	full_text†
ƒ
€%534 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %530, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %530
Cbitcast8B6
4
	full_text'
%
#%535 = bitcast double* %534 to i64*
.double*8B

	full_text

double* %534
Jload8B@
>
	full_text1
/
-%536 = load i64, i64* %535, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %535
Jstore8B?
=
	full_text0
.
,store i64 %536, i64* %102, align 8, !tbaa !8
&i648B

	full_text


i64 %536
(i64*8B

	full_text

	i64* %102
«getelementptr8B—
”
	full_text†
ƒ
€%537 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %530, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %530
Cbitcast8B6
4
	full_text'
%
#%538 = bitcast double* %537 to i64*
.double*8B

	full_text

double* %537
Jload8B@
>
	full_text1
/
-%539 = load i64, i64* %538, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %538
Kstore8B@
>
	full_text1
/
-store i64 %539, i64* %107, align 16, !tbaa !8
&i648B

	full_text


i64 %539
(i64*8B

	full_text

	i64* %107
«getelementptr8B—
”
	full_text†
ƒ
€%540 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %530, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %530
Cbitcast8B6
4
	full_text'
%
#%541 = bitcast double* %540 to i64*
.double*8B

	full_text

double* %540
Jload8B@
>
	full_text1
/
-%542 = load i64, i64* %541, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %541
Jstore8B?
=
	full_text0
.
,store i64 %542, i64* %112, align 8, !tbaa !8
&i648B

	full_text


i64 %542
(i64*8B

	full_text

	i64* %112
«getelementptr8B—
”
	full_text†
ƒ
€%543 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %530, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %530
Cbitcast8B6
4
	full_text'
%
#%544 = bitcast double* %543 to i64*
.double*8B

	full_text

double* %543
Jload8B@
>
	full_text1
/
-%545 = load i64, i64* %544, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %544
Kstore8B@
>
	full_text1
/
-store i64 %545, i64* %117, align 16, !tbaa !8
&i648B

	full_text


i64 %545
(i64*8B

	full_text

	i64* %117
:add8B1
/
	full_text"
 
%546 = add nuw nsw i64 %517, 1
&i648B

	full_text


i64 %517
”getelementptr8B€
~
	full_textq
o
m%547 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 %546
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%548 = load double, double* %547, align 8, !tbaa !8
.double*8B

	full_text

double* %547
”getelementptr8B€
~
	full_textq
o
m%549 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 %546
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%550 = load double, double* %549, align 8, !tbaa !8
.double*8B

	full_text

double* %549
”getelementptr8B€
~
	full_textq
o
m%551 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 %546
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%552 = load double, double* %551, align 8, !tbaa !8
.double*8B

	full_text

double* %551
”getelementptr8B€
~
	full_textq
o
m%553 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 %546
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%554 = load double, double* %553, align 8, !tbaa !8
.double*8B

	full_text

double* %553
”getelementptr8B€
~
	full_textq
o
m%555 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 %546
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%556 = load double, double* %555, align 8, !tbaa !8
.double*8B

	full_text

double* %555
”getelementptr8B€
~
	full_textq
o
m%557 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 %546
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %546
Pload8BF
D
	full_text7
5
3%558 = load double, double* %557, align 8, !tbaa !8
.double*8B

	full_text

double* %557
«getelementptr8B—
”
	full_text†
ƒ
€%559 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %517, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %517
Pload8BF
D
	full_text7
5
3%560 = load double, double* %559, align 8, !tbaa !8
.double*8B

	full_text

double* %559
vcall8Bl
j
	full_text]
[
Y%561 = tail call double @llvm.fmuladd.f64(double %506, double -2.000000e+00, double %501)
,double8B

	full_text

double %506
,double8B

	full_text

double %501
:fadd8B0
.
	full_text!

%562 = fadd double %561, %511
,double8B

	full_text

double %561
,double8B

	full_text

double %511
{call8Bq
o
	full_textb
`
^%563 = tail call double @llvm.fmuladd.f64(double %562, double 0x40D2FC3000000001, double %560)
,double8B

	full_text

double %562
,double8B

	full_text

double %560
:fsub8B0
.
	full_text!

%564 = fsub double %500, %510
,double8B

	full_text

double %500
,double8B

	full_text

double %510
vcall8Bl
j
	full_text]
[
Y%565 = tail call double @llvm.fmuladd.f64(double %564, double -8.050000e+01, double %563)
,double8B

	full_text

double %564
,double8B

	full_text

double %563
«getelementptr8B—
”
	full_text†
ƒ
€%566 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %517, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %517
Pload8BF
D
	full_text7
5
3%567 = load double, double* %566, align 8, !tbaa !8
.double*8B

	full_text

double* %566
vcall8Bl
j
	full_text]
[
Y%568 = tail call double @llvm.fmuladd.f64(double %505, double -2.000000e+00, double %500)
,double8B

	full_text

double %505
,double8B

	full_text

double %500
:fadd8B0
.
	full_text!

%569 = fadd double %510, %568
,double8B

	full_text

double %510
,double8B

	full_text

double %568
{call8Bq
o
	full_textb
`
^%570 = tail call double @llvm.fmuladd.f64(double %569, double 0x40D2FC3000000001, double %567)
,double8B

	full_text

double %569
,double8B

	full_text

double %567
vcall8Bl
j
	full_text]
[
Y%571 = tail call double @llvm.fmuladd.f64(double %519, double -2.000000e+00, double %548)
,double8B

	full_text

double %519
,double8B

	full_text

double %548
:fadd8B0
.
	full_text!

%572 = fadd double %518, %571
,double8B

	full_text

double %518
,double8B

	full_text

double %571
{call8Bq
o
	full_textb
`
^%573 = tail call double @llvm.fmuladd.f64(double %572, double 0x40AB004444444445, double %570)
,double8B

	full_text

double %572
,double8B

	full_text

double %570
:fmul8B0
.
	full_text!

%574 = fmul double %518, %510
,double8B

	full_text

double %518
,double8B

	full_text

double %510
Cfsub8B9
7
	full_text*
(
&%575 = fsub double -0.000000e+00, %574
,double8B

	full_text

double %574
mcall8Bc
a
	full_textT
R
P%576 = tail call double @llvm.fmuladd.f64(double %500, double %548, double %575)
,double8B

	full_text

double %500
,double8B

	full_text

double %548
,double8B

	full_text

double %575
:fsub8B0
.
	full_text!

%577 = fsub double %497, %558
,double8B

	full_text

double %497
,double8B

	full_text

double %558
Abitcast8B4
2
	full_text%
#
!%578 = bitcast i64 %507 to double
&i648B

	full_text


i64 %507
:fsub8B0
.
	full_text!

%579 = fsub double %577, %578
,double8B

	full_text

double %577
,double8B

	full_text

double %578
:fadd8B0
.
	full_text!

%580 = fadd double %523, %579
,double8B

	full_text

double %523
,double8B

	full_text

double %579
ucall8Bk
i
	full_text\
Z
X%581 = tail call double @llvm.fmuladd.f64(double %580, double 4.000000e-01, double %576)
,double8B

	full_text

double %580
,double8B

	full_text

double %576
vcall8Bl
j
	full_text]
[
Y%582 = tail call double @llvm.fmuladd.f64(double %581, double -8.050000e+01, double %573)
,double8B

	full_text

double %581
,double8B

	full_text

double %573
«getelementptr8B—
”
	full_text†
ƒ
€%583 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %517, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %517
Pload8BF
D
	full_text7
5
3%584 = load double, double* %583, align 8, !tbaa !8
.double*8B

	full_text

double* %583
vcall8Bl
j
	full_text]
[
Y%585 = tail call double @llvm.fmuladd.f64(double %504, double -2.000000e+00, double %499)
,double8B

	full_text

double %504
,double8B

	full_text

double %499
:fadd8B0
.
	full_text!

%586 = fadd double %585, %509
,double8B

	full_text

double %585
,double8B

	full_text

double %509
{call8Bq
o
	full_textb
`
^%587 = tail call double @llvm.fmuladd.f64(double %586, double 0x40D2FC3000000001, double %584)
,double8B

	full_text

double %586
,double8B

	full_text

double %584
vcall8Bl
j
	full_text]
[
Y%588 = tail call double @llvm.fmuladd.f64(double %521, double -2.000000e+00, double %550)
,double8B

	full_text

double %521
,double8B

	full_text

double %550
:fadd8B0
.
	full_text!

%589 = fadd double %520, %588
,double8B

	full_text

double %520
,double8B

	full_text

double %588
{call8Bq
o
	full_textb
`
^%590 = tail call double @llvm.fmuladd.f64(double %589, double 0x40A4403333333334, double %587)
,double8B

	full_text

double %589
,double8B

	full_text

double %587
:fmul8B0
.
	full_text!

%591 = fmul double %518, %509
,double8B

	full_text

double %518
,double8B

	full_text

double %509
Cfsub8B9
7
	full_text*
(
&%592 = fsub double -0.000000e+00, %591
,double8B

	full_text

double %591
mcall8Bc
a
	full_textT
R
P%593 = tail call double @llvm.fmuladd.f64(double %499, double %548, double %592)
,double8B

	full_text

double %499
,double8B

	full_text

double %548
,double8B

	full_text

double %592
vcall8Bl
j
	full_text]
[
Y%594 = tail call double @llvm.fmuladd.f64(double %593, double -8.050000e+01, double %590)
,double8B

	full_text

double %593
,double8B

	full_text

double %590
«getelementptr8B—
”
	full_text†
ƒ
€%595 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %517, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %517
Pload8BF
D
	full_text7
5
3%596 = load double, double* %595, align 8, !tbaa !8
.double*8B

	full_text

double* %595
vcall8Bl
j
	full_text]
[
Y%597 = tail call double @llvm.fmuladd.f64(double %503, double -2.000000e+00, double %498)
,double8B

	full_text

double %503
,double8B

	full_text

double %498
Pload8BF
D
	full_text7
5
3%598 = load double, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
:fadd8B0
.
	full_text!

%599 = fadd double %597, %598
,double8B

	full_text

double %597
,double8B

	full_text

double %598
{call8Bq
o
	full_textb
`
^%600 = tail call double @llvm.fmuladd.f64(double %599, double 0x40D2FC3000000001, double %596)
,double8B

	full_text

double %599
,double8B

	full_text

double %596
vcall8Bl
j
	full_text]
[
Y%601 = tail call double @llvm.fmuladd.f64(double %528, double -2.000000e+00, double %552)
,double8B

	full_text

double %528
,double8B

	full_text

double %552
:fadd8B0
.
	full_text!

%602 = fadd double %529, %601
,double8B

	full_text

double %529
,double8B

	full_text

double %601
{call8Bq
o
	full_textb
`
^%603 = tail call double @llvm.fmuladd.f64(double %602, double 0x40A4403333333334, double %600)
,double8B

	full_text

double %602
,double8B

	full_text

double %600
:fmul8B0
.
	full_text!

%604 = fmul double %518, %598
,double8B

	full_text

double %518
,double8B

	full_text

double %598
Cfsub8B9
7
	full_text*
(
&%605 = fsub double -0.000000e+00, %604
,double8B

	full_text

double %604
mcall8Bc
a
	full_textT
R
P%606 = tail call double @llvm.fmuladd.f64(double %498, double %548, double %605)
,double8B

	full_text

double %498
,double8B

	full_text

double %548
,double8B

	full_text

double %605
vcall8Bl
j
	full_text]
[
Y%607 = tail call double @llvm.fmuladd.f64(double %606, double -8.050000e+01, double %603)
,double8B

	full_text

double %606
,double8B

	full_text

double %603
«getelementptr8B—
”
	full_text†
ƒ
€%608 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %517, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %517
Pload8BF
D
	full_text7
5
3%609 = load double, double* %608, align 8, !tbaa !8
.double*8B

	full_text

double* %608
Pload8BF
D
	full_text7
5
3%610 = load double, double* %67, align 16, !tbaa !8
-double*8B

	full_text

double* %67
vcall8Bl
j
	full_text]
[
Y%611 = tail call double @llvm.fmuladd.f64(double %610, double -2.000000e+00, double %497)
,double8B

	full_text

double %610
,double8B

	full_text

double %497
:fadd8B0
.
	full_text!

%612 = fadd double %611, %578
,double8B

	full_text

double %611
,double8B

	full_text

double %578
{call8Bq
o
	full_textb
`
^%613 = tail call double @llvm.fmuladd.f64(double %612, double 0x40D2FC3000000001, double %609)
,double8B

	full_text

double %612
,double8B

	full_text

double %609
vcall8Bl
j
	full_text]
[
Y%614 = tail call double @llvm.fmuladd.f64(double %526, double -2.000000e+00, double %554)
,double8B

	full_text

double %526
,double8B

	full_text

double %554
:fadd8B0
.
	full_text!

%615 = fadd double %527, %614
,double8B

	full_text

double %527
,double8B

	full_text

double %614
{call8Bq
o
	full_textb
`
^%616 = tail call double @llvm.fmuladd.f64(double %615, double 0xC0A370D4FDF3B645, double %613)
,double8B

	full_text

double %615
,double8B

	full_text

double %613
Bfmul8B8
6
	full_text)
'
%%617 = fmul double %519, 2.000000e+00
,double8B

	full_text

double %519
:fmul8B0
.
	full_text!

%618 = fmul double %519, %617
,double8B

	full_text

double %519
,double8B

	full_text

double %617
Cfsub8B9
7
	full_text*
(
&%619 = fsub double -0.000000e+00, %618
,double8B

	full_text

double %618
mcall8Bc
a
	full_textT
R
P%620 = tail call double @llvm.fmuladd.f64(double %548, double %548, double %619)
,double8B

	full_text

double %548
,double8B

	full_text

double %548
,double8B

	full_text

double %619
mcall8Bc
a
	full_textT
R
P%621 = tail call double @llvm.fmuladd.f64(double %518, double %518, double %620)
,double8B

	full_text

double %518
,double8B

	full_text

double %518
,double8B

	full_text

double %620
{call8Bq
o
	full_textb
`
^%622 = tail call double @llvm.fmuladd.f64(double %621, double 0x407B004444444445, double %616)
,double8B

	full_text

double %621
,double8B

	full_text

double %616
Bfmul8B8
6
	full_text)
'
%%623 = fmul double %610, 2.000000e+00
,double8B

	full_text

double %610
:fmul8B0
.
	full_text!

%624 = fmul double %524, %623
,double8B

	full_text

double %524
,double8B

	full_text

double %623
Cfsub8B9
7
	full_text*
(
&%625 = fsub double -0.000000e+00, %624
,double8B

	full_text

double %624
mcall8Bc
a
	full_textT
R
P%626 = tail call double @llvm.fmuladd.f64(double %497, double %556, double %625)
,double8B

	full_text

double %497
,double8B

	full_text

double %556
,double8B

	full_text

double %625
mcall8Bc
a
	full_textT
R
P%627 = tail call double @llvm.fmuladd.f64(double %578, double %525, double %626)
,double8B

	full_text

double %578
,double8B

	full_text

double %525
,double8B

	full_text

double %626
{call8Bq
o
	full_textb
`
^%628 = tail call double @llvm.fmuladd.f64(double %627, double 0x40B3D884189374BC, double %622)
,double8B

	full_text

double %627
,double8B

	full_text

double %622
Bfmul8B8
6
	full_text)
'
%%629 = fmul double %558, 4.000000e-01
,double8B

	full_text

double %558
Cfsub8B9
7
	full_text*
(
&%630 = fsub double -0.000000e+00, %629
,double8B

	full_text

double %629
ucall8Bk
i
	full_text\
Z
X%631 = tail call double @llvm.fmuladd.f64(double %497, double 1.400000e+00, double %630)
,double8B

	full_text

double %497
,double8B

	full_text

double %630
Bfmul8B8
6
	full_text)
'
%%632 = fmul double %523, 4.000000e-01
,double8B

	full_text

double %523
Cfsub8B9
7
	full_text*
(
&%633 = fsub double -0.000000e+00, %632
,double8B

	full_text

double %632
ucall8Bk
i
	full_text\
Z
X%634 = tail call double @llvm.fmuladd.f64(double %578, double 1.400000e+00, double %633)
,double8B

	full_text

double %578
,double8B

	full_text

double %633
:fmul8B0
.
	full_text!

%635 = fmul double %518, %634
,double8B

	full_text

double %518
,double8B

	full_text

double %634
Cfsub8B9
7
	full_text*
(
&%636 = fsub double -0.000000e+00, %635
,double8B

	full_text

double %635
mcall8Bc
a
	full_textT
R
P%637 = tail call double @llvm.fmuladd.f64(double %631, double %548, double %636)
,double8B

	full_text

double %631
,double8B

	full_text

double %548
,double8B

	full_text

double %636
vcall8Bl
j
	full_text]
[
Y%638 = tail call double @llvm.fmuladd.f64(double %637, double -8.050000e+01, double %628)
,double8B

	full_text

double %637
,double8B

	full_text

double %628
Qload8BG
E
	full_text8
6
4%639 = load double, double* %494, align 16, !tbaa !8
.double*8B

	full_text

double* %494
vcall8Bl
j
	full_text]
[
Y%640 = tail call double @llvm.fmuladd.f64(double %511, double -4.000000e+00, double %639)
,double8B

	full_text

double %511
,double8B

	full_text

double %639
ucall8Bk
i
	full_text\
Z
X%641 = tail call double @llvm.fmuladd.f64(double %506, double 6.000000e+00, double %640)
,double8B

	full_text

double %506
,double8B

	full_text

double %640
vcall8Bl
j
	full_text]
[
Y%642 = tail call double @llvm.fmuladd.f64(double %501, double -4.000000e+00, double %641)
,double8B

	full_text

double %501
,double8B

	full_text

double %641
Qload8BG
E
	full_text8
6
4%643 = load double, double* %290, align 16, !tbaa !8
.double*8B

	full_text

double* %290
:fadd8B0
.
	full_text!

%644 = fadd double %642, %643
,double8B

	full_text

double %642
,double8B

	full_text

double %643
vcall8Bl
j
	full_text]
[
Y%645 = tail call double @llvm.fmuladd.f64(double %644, double -2.500000e-01, double %565)
,double8B

	full_text

double %644
,double8B

	full_text

double %565
Pstore8BE
C
	full_text6
4
2store double %645, double* %559, align 8, !tbaa !8
,double8B

	full_text

double %645
.double*8B

	full_text

double* %559
Pload8BF
D
	full_text7
5
3%646 = load double, double* %149, align 8, !tbaa !8
.double*8B

	full_text

double* %149
vcall8Bl
j
	full_text]
[
Y%647 = tail call double @llvm.fmuladd.f64(double %510, double -4.000000e+00, double %646)
,double8B

	full_text

double %510
,double8B

	full_text

double %646
ucall8Bk
i
	full_text\
Z
X%648 = tail call double @llvm.fmuladd.f64(double %505, double 6.000000e+00, double %647)
,double8B

	full_text

double %505
,double8B

	full_text

double %647
vcall8Bl
j
	full_text]
[
Y%649 = tail call double @llvm.fmuladd.f64(double %500, double -4.000000e+00, double %648)
,double8B

	full_text

double %500
,double8B

	full_text

double %648
Pload8BF
D
	full_text7
5
3%650 = load double, double* %101, align 8, !tbaa !8
.double*8B

	full_text

double* %101
:fadd8B0
.
	full_text!

%651 = fadd double %649, %650
,double8B

	full_text

double %649
,double8B

	full_text

double %650
vcall8Bl
j
	full_text]
[
Y%652 = tail call double @llvm.fmuladd.f64(double %651, double -2.500000e-01, double %582)
,double8B

	full_text

double %651
,double8B

	full_text

double %582
Pstore8BE
C
	full_text6
4
2store double %652, double* %566, align 8, !tbaa !8
,double8B

	full_text

double %652
.double*8B

	full_text

double* %566
Qload8BG
E
	full_text8
6
4%653 = load double, double* %154, align 16, !tbaa !8
.double*8B

	full_text

double* %154
vcall8Bl
j
	full_text]
[
Y%654 = tail call double @llvm.fmuladd.f64(double %509, double -4.000000e+00, double %653)
,double8B

	full_text

double %509
,double8B

	full_text

double %653
ucall8Bk
i
	full_text\
Z
X%655 = tail call double @llvm.fmuladd.f64(double %504, double 6.000000e+00, double %654)
,double8B

	full_text

double %504
,double8B

	full_text

double %654
vcall8Bl
j
	full_text]
[
Y%656 = tail call double @llvm.fmuladd.f64(double %499, double -4.000000e+00, double %655)
,double8B

	full_text

double %499
,double8B

	full_text

double %655
Qload8BG
E
	full_text8
6
4%657 = load double, double* %106, align 16, !tbaa !8
.double*8B

	full_text

double* %106
:fadd8B0
.
	full_text!

%658 = fadd double %656, %657
,double8B

	full_text

double %656
,double8B

	full_text

double %657
vcall8Bl
j
	full_text]
[
Y%659 = tail call double @llvm.fmuladd.f64(double %658, double -2.500000e-01, double %594)
,double8B

	full_text

double %658
,double8B

	full_text

double %594
Pstore8BE
C
	full_text6
4
2store double %659, double* %583, align 8, !tbaa !8
,double8B

	full_text

double %659
.double*8B

	full_text

double* %583
Pload8BF
D
	full_text7
5
3%660 = load double, double* %159, align 8, !tbaa !8
.double*8B

	full_text

double* %159
vcall8Bl
j
	full_text]
[
Y%661 = tail call double @llvm.fmuladd.f64(double %598, double -4.000000e+00, double %660)
,double8B

	full_text

double %598
,double8B

	full_text

double %660
ucall8Bk
i
	full_text\
Z
X%662 = tail call double @llvm.fmuladd.f64(double %503, double 6.000000e+00, double %661)
,double8B

	full_text

double %503
,double8B

	full_text

double %661
vcall8Bl
j
	full_text]
[
Y%663 = tail call double @llvm.fmuladd.f64(double %498, double -4.000000e+00, double %662)
,double8B

	full_text

double %498
,double8B

	full_text

double %662
Pload8BF
D
	full_text7
5
3%664 = load double, double* %111, align 8, !tbaa !8
.double*8B

	full_text

double* %111
:fadd8B0
.
	full_text!

%665 = fadd double %663, %664
,double8B

	full_text

double %663
,double8B

	full_text

double %664
vcall8Bl
j
	full_text]
[
Y%666 = tail call double @llvm.fmuladd.f64(double %665, double -2.500000e-01, double %607)
,double8B

	full_text

double %665
,double8B

	full_text

double %607
Pstore8BE
C
	full_text6
4
2store double %666, double* %595, align 8, !tbaa !8
,double8B

	full_text

double %666
.double*8B

	full_text

double* %595
Qload8BG
E
	full_text8
6
4%667 = load double, double* %164, align 16, !tbaa !8
.double*8B

	full_text

double* %164
Qload8BG
E
	full_text8
6
4%668 = load double, double* %161, align 16, !tbaa !8
.double*8B

	full_text

double* %161
vcall8Bl
j
	full_text]
[
Y%669 = tail call double @llvm.fmuladd.f64(double %668, double -4.000000e+00, double %667)
,double8B

	full_text

double %668
,double8B

	full_text

double %667
ucall8Bk
i
	full_text\
Z
X%670 = tail call double @llvm.fmuladd.f64(double %610, double 6.000000e+00, double %669)
,double8B

	full_text

double %610
,double8B

	full_text

double %669
vcall8Bl
j
	full_text]
[
Y%671 = tail call double @llvm.fmuladd.f64(double %497, double -4.000000e+00, double %670)
,double8B

	full_text

double %497
,double8B

	full_text

double %670
Qload8BG
E
	full_text8
6
4%672 = load double, double* %116, align 16, !tbaa !8
.double*8B

	full_text

double* %116
:fadd8B0
.
	full_text!

%673 = fadd double %671, %672
,double8B

	full_text

double %671
,double8B

	full_text

double %672
vcall8Bl
j
	full_text]
[
Y%674 = tail call double @llvm.fmuladd.f64(double %673, double -2.500000e-01, double %638)
,double8B

	full_text

double %673
,double8B

	full_text

double %638
Pstore8BE
C
	full_text6
4
2store double %674, double* %608, align 8, !tbaa !8
,double8B

	full_text

double %674
.double*8B

	full_text

double* %608
:icmp8B0
.
	full_text!

%675 = icmp eq i64 %546, %495
&i648B

	full_text


i64 %546
&i648B

	full_text


i64 %495
Abitcast8B4
2
	full_text%
#
!%676 = bitcast double %511 to i64
,double8B

	full_text

double %511
Abitcast8B4
2
	full_text%
#
!%677 = bitcast double %510 to i64
,double8B

	full_text

double %510
Abitcast8B4
2
	full_text%
#
!%678 = bitcast double %509 to i64
,double8B

	full_text

double %509
Abitcast8B4
2
	full_text%
#
!%679 = bitcast double %598 to i64
,double8B

	full_text

double %598
Abitcast8B4
2
	full_text%
#
!%680 = bitcast double %668 to i64
,double8B

	full_text

double %668
Abitcast8B4
2
	full_text%
#
!%681 = bitcast double %503 to i64
,double8B

	full_text

double %503
Abitcast8B4
2
	full_text%
#
!%682 = bitcast double %610 to i64
,double8B

	full_text

double %610
Abitcast8B4
2
	full_text%
#
!%683 = bitcast double %497 to i64
,double8B

	full_text

double %497
=br8B5
3
	full_text&
$
"br i1 %675, label %684, label %496
$i18B

	full_text
	
i1 %675
Qstore8BF
D
	full_text7
5
3store double %511, double* %491, align 16, !tbaa !8
,double8B

	full_text

double %511
.double*8B

	full_text

double* %491
Pstore8BE
C
	full_text6
4
2store double %510, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %510
.double*8B

	full_text

double* %146
Qstore8BF
D
	full_text7
5
3store double %509, double* %151, align 16, !tbaa !8
,double8B

	full_text

double %509
.double*8B

	full_text

double* %151
Qstore8BF
D
	full_text7
5
3store double %506, double* %492, align 16, !tbaa !8
,double8B

	full_text

double %506
.double*8B

	full_text

double* %492
Ostore8BD
B
	full_text5
3
1store double %505, double* %52, align 8, !tbaa !8
,double8B

	full_text

double %505
-double*8B

	full_text

double* %52
Pstore8BE
C
	full_text6
4
2store double %504, double* %57, align 16, !tbaa !8
,double8B

	full_text

double %504
-double*8B

	full_text

double* %57
Ostore8BD
B
	full_text5
3
1store double %503, double* %62, align 8, !tbaa !8
,double8B

	full_text

double %503
-double*8B

	full_text

double* %62
Qstore8BF
D
	full_text7
5
3store double %501, double* %493, align 16, !tbaa !8
,double8B

	full_text

double %501
.double*8B

	full_text

double* %493
Ostore8BD
B
	full_text5
3
1store double %500, double* %77, align 8, !tbaa !8
,double8B

	full_text

double %500
-double*8B

	full_text

double* %77
Pstore8BE
C
	full_text6
4
2store double %499, double* %82, align 16, !tbaa !8
,double8B

	full_text

double %499
-double*8B

	full_text

double* %82
Ostore8BD
B
	full_text5
3
1store double %498, double* %87, align 8, !tbaa !8
,double8B

	full_text

double %498
-double*8B

	full_text

double* %87
Pstore8BE
C
	full_text6
4
2store double %497, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %497
-double*8B

	full_text

double* %92
(br8B 

	full_text

br label %685
Mphi8BD
B
	full_text5
3
1%686 = phi double* [ %489, %488 ], [ %494, %684 ]
.double*8B

	full_text

double* %489
.double*8B

	full_text

double* %494
Lphi8BC
A
	full_text4
2
0%687 = phi double [ %476, %488 ], [ %672, %684 ]
,double8B

	full_text

double %476
,double8B

	full_text

double %672
Lphi8BC
A
	full_text4
2
0%688 = phi double [ %468, %488 ], [ %664, %684 ]
,double8B

	full_text

double %468
,double8B

	full_text

double %664
Lphi8BC
A
	full_text4
2
0%689 = phi double [ %459, %488 ], [ %657, %684 ]
,double8B

	full_text

double %459
,double8B

	full_text

double %657
Lphi8BC
A
	full_text4
2
0%690 = phi double [ %450, %488 ], [ %650, %684 ]
,double8B

	full_text

double %450
,double8B

	full_text

double %650
Lphi8BC
A
	full_text4
2
0%691 = phi double [ %441, %488 ], [ %643, %684 ]
,double8B

	full_text

double %441
,double8B

	full_text

double %643
Lphi8BC
A
	full_text4
2
0%692 = phi double [ %474, %488 ], [ %497, %684 ]
,double8B

	full_text

double %474
,double8B

	full_text

double %497
Iphi8B@
>
	full_text1
/
-%693 = phi i64 [ %486, %488 ], [ %683, %684 ]
&i648B

	full_text


i64 %486
&i648B

	full_text


i64 %683
Lphi8BC
A
	full_text4
2
0%694 = phi double [ %466, %488 ], [ %498, %684 ]
,double8B

	full_text

double %466
,double8B

	full_text

double %498
Lphi8BC
A
	full_text4
2
0%695 = phi double [ %457, %488 ], [ %499, %684 ]
,double8B

	full_text

double %457
,double8B

	full_text

double %499
Lphi8BC
A
	full_text4
2
0%696 = phi double [ %448, %488 ], [ %500, %684 ]
,double8B

	full_text

double %448
,double8B

	full_text

double %500
Lphi8BC
A
	full_text4
2
0%697 = phi double [ %439, %488 ], [ %501, %684 ]
,double8B

	full_text

double %439
,double8B

	full_text

double %501
Iphi8B@
>
	full_text1
/
-%698 = phi i64 [ %115, %488 ], [ %682, %684 ]
&i648B

	full_text


i64 %115
&i648B

	full_text


i64 %682
Iphi8B@
>
	full_text1
/
-%699 = phi i64 [ %485, %488 ], [ %681, %684 ]
&i648B

	full_text


i64 %485
&i648B

	full_text


i64 %681
Lphi8BC
A
	full_text4
2
0%700 = phi double [ %454, %488 ], [ %504, %684 ]
,double8B

	full_text

double %454
,double8B

	full_text

double %504
Lphi8BC
A
	full_text4
2
0%701 = phi double [ %445, %488 ], [ %505, %684 ]
,double8B

	full_text

double %445
,double8B

	full_text

double %505
Lphi8BC
A
	full_text4
2
0%702 = phi double [ %436, %488 ], [ %506, %684 ]
,double8B

	full_text

double %436
,double8B

	full_text

double %506
Iphi8B@
>
	full_text1
/
-%703 = phi i64 [ %484, %488 ], [ %680, %684 ]
&i648B

	full_text


i64 %484
&i648B

	full_text


i64 %680
Iphi8B@
>
	full_text1
/
-%704 = phi i64 [ %483, %488 ], [ %679, %684 ]
&i648B

	full_text


i64 %483
&i648B

	full_text


i64 %679
Iphi8B@
>
	full_text1
/
-%705 = phi i64 [ %482, %488 ], [ %678, %684 ]
&i648B

	full_text


i64 %482
&i648B

	full_text


i64 %678
Iphi8B@
>
	full_text1
/
-%706 = phi i64 [ %481, %488 ], [ %677, %684 ]
&i648B

	full_text


i64 %481
&i648B

	full_text


i64 %677
Iphi8B@
>
	full_text1
/
-%707 = phi i64 [ %480, %488 ], [ %676, %684 ]
&i648B

	full_text


i64 %480
&i648B

	full_text


i64 %676
Lphi8BC
A
	full_text4
2
0%708 = phi double [ %186, %488 ], [ %528, %684 ]
,double8B

	full_text

double %186
,double8B

	full_text

double %528
Lphi8BC
A
	full_text4
2
0%709 = phi double [ %345, %488 ], [ %552, %684 ]
,double8B

	full_text

double %345
,double8B

	full_text

double %552
Lphi8BC
A
	full_text4
2
0%710 = phi double [ %188, %488 ], [ %526, %684 ]
,double8B

	full_text

double %188
,double8B

	full_text

double %526
Lphi8BC
A
	full_text4
2
0%711 = phi double [ %347, %488 ], [ %554, %684 ]
,double8B

	full_text

double %347
,double8B

	full_text

double %554
Lphi8BC
A
	full_text4
2
0%712 = phi double [ %190, %488 ], [ %524, %684 ]
,double8B

	full_text

double %190
,double8B

	full_text

double %524
Lphi8BC
A
	full_text4
2
0%713 = phi double [ %349, %488 ], [ %556, %684 ]
,double8B

	full_text

double %349
,double8B

	full_text

double %556
Lphi8BC
A
	full_text4
2
0%714 = phi double [ %192, %488 ], [ %522, %684 ]
,double8B

	full_text

double %192
,double8B

	full_text

double %522
Lphi8BC
A
	full_text4
2
0%715 = phi double [ %351, %488 ], [ %558, %684 ]
,double8B

	full_text

double %351
,double8B

	full_text

double %558
Lphi8BC
A
	full_text4
2
0%716 = phi double [ %343, %488 ], [ %550, %684 ]
,double8B

	full_text

double %343
,double8B

	full_text

double %550
Lphi8BC
A
	full_text4
2
0%717 = phi double [ %184, %488 ], [ %521, %684 ]
,double8B

	full_text

double %184
,double8B

	full_text

double %521
Lphi8BC
A
	full_text4
2
0%718 = phi double [ %341, %488 ], [ %548, %684 ]
,double8B

	full_text

double %341
,double8B

	full_text

double %548
Lphi8BC
A
	full_text4
2
0%719 = phi double [ %182, %488 ], [ %519, %684 ]
,double8B

	full_text

double %182
,double8B

	full_text

double %519
Kstore8B@
>
	full_text1
/
-store i64 %707, i64* %145, align 16, !tbaa !8
&i648B

	full_text


i64 %707
(i64*8B

	full_text

	i64* %145
Jstore8B?
=
	full_text0
.
,store i64 %706, i64* %150, align 8, !tbaa !8
&i648B

	full_text


i64 %706
(i64*8B

	full_text

	i64* %150
Kstore8B@
>
	full_text1
/
-store i64 %705, i64* %155, align 16, !tbaa !8
&i648B

	full_text


i64 %705
(i64*8B

	full_text

	i64* %155
Jstore8B?
=
	full_text0
.
,store i64 %704, i64* %160, align 8, !tbaa !8
&i648B

	full_text


i64 %704
(i64*8B

	full_text

	i64* %160
Kstore8B@
>
	full_text1
/
-store i64 %703, i64* %165, align 16, !tbaa !8
&i648B

	full_text


i64 %703
(i64*8B

	full_text

	i64* %165
qgetelementptr8B^
\
	full_textO
M
K%720 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Qstore8BF
D
	full_text7
5
3store double %702, double* %720, align 16, !tbaa !8
,double8B

	full_text

double %702
.double*8B

	full_text

double* %720
Pstore8BE
C
	full_text6
4
2store double %701, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %701
.double*8B

	full_text

double* %146
Qstore8BF
D
	full_text7
5
3store double %700, double* %151, align 16, !tbaa !8
,double8B

	full_text

double %700
.double*8B

	full_text

double* %151
Jstore8B?
=
	full_text0
.
,store i64 %699, i64* %157, align 8, !tbaa !8
&i648B

	full_text


i64 %699
(i64*8B

	full_text

	i64* %157
Kstore8B@
>
	full_text1
/
-store i64 %698, i64* %162, align 16, !tbaa !8
&i648B

	full_text


i64 %698
(i64*8B

	full_text

	i64* %162
qgetelementptr8B^
\
	full_textO
M
K%721 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Qstore8BF
D
	full_text7
5
3store double %697, double* %721, align 16, !tbaa !8
,double8B

	full_text

double %697
.double*8B

	full_text

double* %721
Ostore8BD
B
	full_text5
3
1store double %696, double* %52, align 8, !tbaa !8
,double8B

	full_text

double %696
-double*8B

	full_text

double* %52
Pstore8BE
C
	full_text6
4
2store double %695, double* %57, align 16, !tbaa !8
,double8B

	full_text

double %695
-double*8B

	full_text

double* %57
Ostore8BD
B
	full_text5
3
1store double %694, double* %62, align 8, !tbaa !8
,double8B

	full_text

double %694
-double*8B

	full_text

double* %62
Pstore8BE
C
	full_text6
4
2store double %692, double* %67, align 16, !tbaa !8
,double8B

	full_text

double %692
-double*8B

	full_text

double* %67
qgetelementptr8B^
\
	full_textO
M
K%722 = getelementptr inbounds [5 x double], [5 x double]* %13, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %13
Qstore8BF
D
	full_text7
5
3store double %691, double* %722, align 16, !tbaa !8
,double8B

	full_text

double %691
.double*8B

	full_text

double* %722
Ostore8BD
B
	full_text5
3
1store double %690, double* %77, align 8, !tbaa !8
,double8B

	full_text

double %690
-double*8B

	full_text

double* %77
Pstore8BE
C
	full_text6
4
2store double %689, double* %82, align 16, !tbaa !8
,double8B

	full_text

double %689
-double*8B

	full_text

double* %82
Ostore8BD
B
	full_text5
3
1store double %688, double* %87, align 8, !tbaa !8
,double8B

	full_text

double %688
-double*8B

	full_text

double* %87
Pstore8BE
C
	full_text6
4
2store double %687, double* %92, align 16, !tbaa !8
,double8B

	full_text

double %687
-double*8B

	full_text

double* %92
4add8B+
)
	full_text

%723 = add nsw i32 %8, 1
8sext8B.
,
	full_text

%724 = sext i32 %723 to i64
&i328B

	full_text


i32 %723
¡getelementptr8B
Š
	full_text}
{
y%725 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %724
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Ibitcast8B<
:
	full_text-
+
)%726 = bitcast [5 x double]* %725 to i64*
:[5 x double]*8B%
#
	full_text

[5 x double]* %725
Jload8B@
>
	full_text1
/
-%727 = load i64, i64* %726, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %726
Jstore8B?
=
	full_text0
.
,store i64 %727, i64* %97, align 16, !tbaa !8
&i648B

	full_text


i64 %727
'i64*8B

	full_text


i64* %97
«getelementptr8B—
”
	full_text†
ƒ
€%728 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %724, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Cbitcast8B6
4
	full_text'
%
#%729 = bitcast double* %728 to i64*
.double*8B

	full_text

double* %728
Jload8B@
>
	full_text1
/
-%730 = load i64, i64* %729, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %729
Jstore8B?
=
	full_text0
.
,store i64 %730, i64* %102, align 8, !tbaa !8
&i648B

	full_text


i64 %730
(i64*8B

	full_text

	i64* %102
«getelementptr8B—
”
	full_text†
ƒ
€%731 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %724, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Cbitcast8B6
4
	full_text'
%
#%732 = bitcast double* %731 to i64*
.double*8B

	full_text

double* %731
Jload8B@
>
	full_text1
/
-%733 = load i64, i64* %732, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %732
Kstore8B@
>
	full_text1
/
-store i64 %733, i64* %107, align 16, !tbaa !8
&i648B

	full_text


i64 %733
(i64*8B

	full_text

	i64* %107
«getelementptr8B—
”
	full_text†
ƒ
€%734 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %724, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Cbitcast8B6
4
	full_text'
%
#%735 = bitcast double* %734 to i64*
.double*8B

	full_text

double* %734
Jload8B@
>
	full_text1
/
-%736 = load i64, i64* %735, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %735
Jstore8B?
=
	full_text0
.
,store i64 %736, i64* %112, align 8, !tbaa !8
&i648B

	full_text


i64 %736
(i64*8B

	full_text

	i64* %112
«getelementptr8B—
”
	full_text†
ƒ
€%737 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %32, i64 %41, i64 %43, i64 %724, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %32
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Cbitcast8B6
4
	full_text'
%
#%738 = bitcast double* %737 to i64*
.double*8B

	full_text

double* %737
Jload8B@
>
	full_text1
/
-%739 = load i64, i64* %738, align 8, !tbaa !8
(i64*8B

	full_text

	i64* %738
Kstore8B@
>
	full_text1
/
-store i64 %739, i64* %117, align 16, !tbaa !8
&i648B

	full_text


i64 %739
(i64*8B

	full_text

	i64* %117
6sext8B,
*
	full_text

%740 = sext i32 %8 to i64
”getelementptr8B€
~
	full_textq
o
m%741 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 %740
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%742 = load double, double* %741, align 8, !tbaa !8
.double*8B

	full_text

double* %741
”getelementptr8B€
~
	full_textq
o
m%743 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 %740
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%744 = load double, double* %743, align 8, !tbaa !8
.double*8B

	full_text

double* %743
”getelementptr8B€
~
	full_textq
o
m%745 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 %740
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%746 = load double, double* %745, align 8, !tbaa !8
.double*8B

	full_text

double* %745
”getelementptr8B€
~
	full_textq
o
m%747 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 %740
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%748 = load double, double* %747, align 8, !tbaa !8
.double*8B

	full_text

double* %747
”getelementptr8B€
~
	full_textq
o
m%749 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 %740
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%750 = load double, double* %749, align 8, !tbaa !8
.double*8B

	full_text

double* %749
”getelementptr8B€
~
	full_textq
o
m%751 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 %740
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%752 = load double, double* %751, align 8, !tbaa !8
.double*8B

	full_text

double* %751
8sext8B.
,
	full_text

%753 = sext i32 %487 to i64
&i328B

	full_text


i32 %487
«getelementptr8B—
”
	full_text†
ƒ
€%754 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %753, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %753
Pload8BF
D
	full_text7
5
3%755 = load double, double* %754, align 8, !tbaa !8
.double*8B

	full_text

double* %754
vcall8Bl
j
	full_text]
[
Y%756 = tail call double @llvm.fmuladd.f64(double %697, double -2.000000e+00, double %691)
,double8B

	full_text

double %697
,double8B

	full_text

double %691
:fadd8B0
.
	full_text!

%757 = fadd double %756, %702
,double8B

	full_text

double %756
,double8B

	full_text

double %702
{call8Bq
o
	full_textb
`
^%758 = tail call double @llvm.fmuladd.f64(double %757, double 0x40D2FC3000000001, double %755)
,double8B

	full_text

double %757
,double8B

	full_text

double %755
:fsub8B0
.
	full_text!

%759 = fsub double %690, %701
,double8B

	full_text

double %690
,double8B

	full_text

double %701
vcall8Bl
j
	full_text]
[
Y%760 = tail call double @llvm.fmuladd.f64(double %759, double -8.050000e+01, double %758)
,double8B

	full_text

double %759
,double8B

	full_text

double %758
«getelementptr8B—
”
	full_text†
ƒ
€%761 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %753, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %753
Pload8BF
D
	full_text7
5
3%762 = load double, double* %761, align 8, !tbaa !8
.double*8B

	full_text

double* %761
vcall8Bl
j
	full_text]
[
Y%763 = tail call double @llvm.fmuladd.f64(double %696, double -2.000000e+00, double %690)
,double8B

	full_text

double %696
,double8B

	full_text

double %690
:fadd8B0
.
	full_text!

%764 = fadd double %701, %763
,double8B

	full_text

double %701
,double8B

	full_text

double %763
{call8Bq
o
	full_textb
`
^%765 = tail call double @llvm.fmuladd.f64(double %764, double 0x40D2FC3000000001, double %762)
,double8B

	full_text

double %764
,double8B

	full_text

double %762
vcall8Bl
j
	full_text]
[
Y%766 = tail call double @llvm.fmuladd.f64(double %718, double -2.000000e+00, double %742)
,double8B

	full_text

double %718
,double8B

	full_text

double %742
:fadd8B0
.
	full_text!

%767 = fadd double %719, %766
,double8B

	full_text

double %719
,double8B

	full_text

double %766
{call8Bq
o
	full_textb
`
^%768 = tail call double @llvm.fmuladd.f64(double %767, double 0x40AB004444444445, double %765)
,double8B

	full_text

double %767
,double8B

	full_text

double %765
:fmul8B0
.
	full_text!

%769 = fmul double %719, %701
,double8B

	full_text

double %719
,double8B

	full_text

double %701
Cfsub8B9
7
	full_text*
(
&%770 = fsub double -0.000000e+00, %769
,double8B

	full_text

double %769
mcall8Bc
a
	full_textT
R
P%771 = tail call double @llvm.fmuladd.f64(double %690, double %742, double %770)
,double8B

	full_text

double %690
,double8B

	full_text

double %742
,double8B

	full_text

double %770
:fsub8B0
.
	full_text!

%772 = fsub double %687, %752
,double8B

	full_text

double %687
,double8B

	full_text

double %752
Abitcast8B4
2
	full_text%
#
!%773 = bitcast i64 %698 to double
&i648B

	full_text


i64 %698
:fsub8B0
.
	full_text!

%774 = fsub double %772, %773
,double8B

	full_text

double %772
,double8B

	full_text

double %773
:fadd8B0
.
	full_text!

%775 = fadd double %714, %774
,double8B

	full_text

double %714
,double8B

	full_text

double %774
ucall8Bk
i
	full_text\
Z
X%776 = tail call double @llvm.fmuladd.f64(double %775, double 4.000000e-01, double %771)
,double8B

	full_text

double %775
,double8B

	full_text

double %771
vcall8Bl
j
	full_text]
[
Y%777 = tail call double @llvm.fmuladd.f64(double %776, double -8.050000e+01, double %768)
,double8B

	full_text

double %776
,double8B

	full_text

double %768
«getelementptr8B—
”
	full_text†
ƒ
€%778 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %753, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %753
Pload8BF
D
	full_text7
5
3%779 = load double, double* %778, align 8, !tbaa !8
.double*8B

	full_text

double* %778
vcall8Bl
j
	full_text]
[
Y%780 = tail call double @llvm.fmuladd.f64(double %695, double -2.000000e+00, double %689)
,double8B

	full_text

double %695
,double8B

	full_text

double %689
:fadd8B0
.
	full_text!

%781 = fadd double %780, %700
,double8B

	full_text

double %780
,double8B

	full_text

double %700
{call8Bq
o
	full_textb
`
^%782 = tail call double @llvm.fmuladd.f64(double %781, double 0x40D2FC3000000001, double %779)
,double8B

	full_text

double %781
,double8B

	full_text

double %779
vcall8Bl
j
	full_text]
[
Y%783 = tail call double @llvm.fmuladd.f64(double %716, double -2.000000e+00, double %744)
,double8B

	full_text

double %716
,double8B

	full_text

double %744
:fadd8B0
.
	full_text!

%784 = fadd double %717, %783
,double8B

	full_text

double %717
,double8B

	full_text

double %783
{call8Bq
o
	full_textb
`
^%785 = tail call double @llvm.fmuladd.f64(double %784, double 0x40A4403333333334, double %782)
,double8B

	full_text

double %784
,double8B

	full_text

double %782
:fmul8B0
.
	full_text!

%786 = fmul double %719, %700
,double8B

	full_text

double %719
,double8B

	full_text

double %700
Cfsub8B9
7
	full_text*
(
&%787 = fsub double -0.000000e+00, %786
,double8B

	full_text

double %786
mcall8Bc
a
	full_textT
R
P%788 = tail call double @llvm.fmuladd.f64(double %689, double %742, double %787)
,double8B

	full_text

double %689
,double8B

	full_text

double %742
,double8B

	full_text

double %787
vcall8Bl
j
	full_text]
[
Y%789 = tail call double @llvm.fmuladd.f64(double %788, double -8.050000e+01, double %785)
,double8B

	full_text

double %788
,double8B

	full_text

double %785
«getelementptr8B—
”
	full_text†
ƒ
€%790 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %753, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %753
Pload8BF
D
	full_text7
5
3%791 = load double, double* %790, align 8, !tbaa !8
.double*8B

	full_text

double* %790
vcall8Bl
j
	full_text]
[
Y%792 = tail call double @llvm.fmuladd.f64(double %694, double -2.000000e+00, double %688)
,double8B

	full_text

double %694
,double8B

	full_text

double %688
Pload8BF
D
	full_text7
5
3%793 = load double, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
:fadd8B0
.
	full_text!

%794 = fadd double %792, %793
,double8B

	full_text

double %792
,double8B

	full_text

double %793
{call8Bq
o
	full_textb
`
^%795 = tail call double @llvm.fmuladd.f64(double %794, double 0x40D2FC3000000001, double %791)
,double8B

	full_text

double %794
,double8B

	full_text

double %791
vcall8Bl
j
	full_text]
[
Y%796 = tail call double @llvm.fmuladd.f64(double %709, double -2.000000e+00, double %746)
,double8B

	full_text

double %709
,double8B

	full_text

double %746
:fadd8B0
.
	full_text!

%797 = fadd double %708, %796
,double8B

	full_text

double %708
,double8B

	full_text

double %796
{call8Bq
o
	full_textb
`
^%798 = tail call double @llvm.fmuladd.f64(double %797, double 0x40A4403333333334, double %795)
,double8B

	full_text

double %797
,double8B

	full_text

double %795
:fmul8B0
.
	full_text!

%799 = fmul double %719, %793
,double8B

	full_text

double %719
,double8B

	full_text

double %793
Cfsub8B9
7
	full_text*
(
&%800 = fsub double -0.000000e+00, %799
,double8B

	full_text

double %799
mcall8Bc
a
	full_textT
R
P%801 = tail call double @llvm.fmuladd.f64(double %688, double %742, double %800)
,double8B

	full_text

double %688
,double8B

	full_text

double %742
,double8B

	full_text

double %800
vcall8Bl
j
	full_text]
[
Y%802 = tail call double @llvm.fmuladd.f64(double %801, double -8.050000e+01, double %798)
,double8B

	full_text

double %801
,double8B

	full_text

double %798
«getelementptr8B—
”
	full_text†
ƒ
€%803 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %753, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %753
Pload8BF
D
	full_text7
5
3%804 = load double, double* %803, align 8, !tbaa !8
.double*8B

	full_text

double* %803
Pload8BF
D
	full_text7
5
3%805 = load double, double* %67, align 16, !tbaa !8
-double*8B

	full_text

double* %67
vcall8Bl
j
	full_text]
[
Y%806 = tail call double @llvm.fmuladd.f64(double %805, double -2.000000e+00, double %687)
,double8B

	full_text

double %805
,double8B

	full_text

double %687
:fadd8B0
.
	full_text!

%807 = fadd double %806, %773
,double8B

	full_text

double %806
,double8B

	full_text

double %773
{call8Bq
o
	full_textb
`
^%808 = tail call double @llvm.fmuladd.f64(double %807, double 0x40D2FC3000000001, double %804)
,double8B

	full_text

double %807
,double8B

	full_text

double %804
vcall8Bl
j
	full_text]
[
Y%809 = tail call double @llvm.fmuladd.f64(double %711, double -2.000000e+00, double %748)
,double8B

	full_text

double %711
,double8B

	full_text

double %748
:fadd8B0
.
	full_text!

%810 = fadd double %710, %809
,double8B

	full_text

double %710
,double8B

	full_text

double %809
{call8Bq
o
	full_textb
`
^%811 = tail call double @llvm.fmuladd.f64(double %810, double 0xC0A370D4FDF3B645, double %808)
,double8B

	full_text

double %810
,double8B

	full_text

double %808
Bfmul8B8
6
	full_text)
'
%%812 = fmul double %718, 2.000000e+00
,double8B

	full_text

double %718
:fmul8B0
.
	full_text!

%813 = fmul double %718, %812
,double8B

	full_text

double %718
,double8B

	full_text

double %812
Cfsub8B9
7
	full_text*
(
&%814 = fsub double -0.000000e+00, %813
,double8B

	full_text

double %813
mcall8Bc
a
	full_textT
R
P%815 = tail call double @llvm.fmuladd.f64(double %742, double %742, double %814)
,double8B

	full_text

double %742
,double8B

	full_text

double %742
,double8B

	full_text

double %814
mcall8Bc
a
	full_textT
R
P%816 = tail call double @llvm.fmuladd.f64(double %719, double %719, double %815)
,double8B

	full_text

double %719
,double8B

	full_text

double %719
,double8B

	full_text

double %815
{call8Bq
o
	full_textb
`
^%817 = tail call double @llvm.fmuladd.f64(double %816, double 0x407B004444444445, double %811)
,double8B

	full_text

double %816
,double8B

	full_text

double %811
Bfmul8B8
6
	full_text)
'
%%818 = fmul double %805, 2.000000e+00
,double8B

	full_text

double %805
:fmul8B0
.
	full_text!

%819 = fmul double %713, %818
,double8B

	full_text

double %713
,double8B

	full_text

double %818
Cfsub8B9
7
	full_text*
(
&%820 = fsub double -0.000000e+00, %819
,double8B

	full_text

double %819
mcall8Bc
a
	full_textT
R
P%821 = tail call double @llvm.fmuladd.f64(double %687, double %750, double %820)
,double8B

	full_text

double %687
,double8B

	full_text

double %750
,double8B

	full_text

double %820
mcall8Bc
a
	full_textT
R
P%822 = tail call double @llvm.fmuladd.f64(double %773, double %712, double %821)
,double8B

	full_text

double %773
,double8B

	full_text

double %712
,double8B

	full_text

double %821
{call8Bq
o
	full_textb
`
^%823 = tail call double @llvm.fmuladd.f64(double %822, double 0x40B3D884189374BC, double %817)
,double8B

	full_text

double %822
,double8B

	full_text

double %817
Bfmul8B8
6
	full_text)
'
%%824 = fmul double %752, 4.000000e-01
,double8B

	full_text

double %752
Cfsub8B9
7
	full_text*
(
&%825 = fsub double -0.000000e+00, %824
,double8B

	full_text

double %824
ucall8Bk
i
	full_text\
Z
X%826 = tail call double @llvm.fmuladd.f64(double %687, double 1.400000e+00, double %825)
,double8B

	full_text

double %687
,double8B

	full_text

double %825
Bfmul8B8
6
	full_text)
'
%%827 = fmul double %714, 4.000000e-01
,double8B

	full_text

double %714
Cfsub8B9
7
	full_text*
(
&%828 = fsub double -0.000000e+00, %827
,double8B

	full_text

double %827
ucall8Bk
i
	full_text\
Z
X%829 = tail call double @llvm.fmuladd.f64(double %773, double 1.400000e+00, double %828)
,double8B

	full_text

double %773
,double8B

	full_text

double %828
:fmul8B0
.
	full_text!

%830 = fmul double %719, %829
,double8B

	full_text

double %719
,double8B

	full_text

double %829
Cfsub8B9
7
	full_text*
(
&%831 = fsub double -0.000000e+00, %830
,double8B

	full_text

double %830
mcall8Bc
a
	full_textT
R
P%832 = tail call double @llvm.fmuladd.f64(double %826, double %742, double %831)
,double8B

	full_text

double %826
,double8B

	full_text

double %742
,double8B

	full_text

double %831
vcall8Bl
j
	full_text]
[
Y%833 = tail call double @llvm.fmuladd.f64(double %832, double -8.050000e+01, double %823)
,double8B

	full_text

double %832
,double8B

	full_text

double %823
Pload8BF
D
	full_text7
5
3%834 = load double, double* %686, align 8, !tbaa !8
.double*8B

	full_text

double* %686
Qload8BG
E
	full_text8
6
4%835 = load double, double* %142, align 16, !tbaa !8
.double*8B

	full_text

double* %142
vcall8Bl
j
	full_text]
[
Y%836 = tail call double @llvm.fmuladd.f64(double %835, double -4.000000e+00, double %834)
,double8B

	full_text

double %835
,double8B

	full_text

double %834
Pload8BF
D
	full_text7
5
3%837 = load double, double* %47, align 16, !tbaa !8
-double*8B

	full_text

double* %47
ucall8Bk
i
	full_text\
Z
X%838 = tail call double @llvm.fmuladd.f64(double %837, double 6.000000e+00, double %836)
,double8B

	full_text

double %837
,double8B

	full_text

double %836
Pload8BF
D
	full_text7
5
3%839 = load double, double* %72, align 16, !tbaa !8
-double*8B

	full_text

double* %72
vcall8Bl
j
	full_text]
[
Y%840 = tail call double @llvm.fmuladd.f64(double %839, double -4.000000e+00, double %838)
,double8B

	full_text

double %839
,double8B

	full_text

double %838
vcall8Bl
j
	full_text]
[
Y%841 = tail call double @llvm.fmuladd.f64(double %840, double -2.500000e-01, double %760)
,double8B

	full_text

double %840
,double8B

	full_text

double %760
Pstore8BE
C
	full_text6
4
2store double %841, double* %754, align 8, !tbaa !8
,double8B

	full_text

double %841
.double*8B

	full_text

double* %754
Pload8BF
D
	full_text7
5
3%842 = load double, double* %149, align 8, !tbaa !8
.double*8B

	full_text

double* %149
Pload8BF
D
	full_text7
5
3%843 = load double, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
vcall8Bl
j
	full_text]
[
Y%844 = tail call double @llvm.fmuladd.f64(double %843, double -4.000000e+00, double %842)
,double8B

	full_text

double %843
,double8B

	full_text

double %842
Oload8BE
C
	full_text6
4
2%845 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
ucall8Bk
i
	full_text\
Z
X%846 = tail call double @llvm.fmuladd.f64(double %845, double 6.000000e+00, double %844)
,double8B

	full_text

double %845
,double8B

	full_text

double %844
Oload8BE
C
	full_text6
4
2%847 = load double, double* %77, align 8, !tbaa !8
-double*8B

	full_text

double* %77
vcall8Bl
j
	full_text]
[
Y%848 = tail call double @llvm.fmuladd.f64(double %847, double -4.000000e+00, double %846)
,double8B

	full_text

double %847
,double8B

	full_text

double %846
vcall8Bl
j
	full_text]
[
Y%849 = tail call double @llvm.fmuladd.f64(double %848, double -2.500000e-01, double %777)
,double8B

	full_text

double %848
,double8B

	full_text

double %777
Pstore8BE
C
	full_text6
4
2store double %849, double* %761, align 8, !tbaa !8
,double8B

	full_text

double %849
.double*8B

	full_text

double* %761
Qload8BG
E
	full_text8
6
4%850 = load double, double* %154, align 16, !tbaa !8
.double*8B

	full_text

double* %154
Qload8BG
E
	full_text8
6
4%851 = load double, double* %151, align 16, !tbaa !8
.double*8B

	full_text

double* %151
vcall8Bl
j
	full_text]
[
Y%852 = tail call double @llvm.fmuladd.f64(double %851, double -4.000000e+00, double %850)
,double8B

	full_text

double %851
,double8B

	full_text

double %850
Pload8BF
D
	full_text7
5
3%853 = load double, double* %57, align 16, !tbaa !8
-double*8B

	full_text

double* %57
ucall8Bk
i
	full_text\
Z
X%854 = tail call double @llvm.fmuladd.f64(double %853, double 6.000000e+00, double %852)
,double8B

	full_text

double %853
,double8B

	full_text

double %852
Pload8BF
D
	full_text7
5
3%855 = load double, double* %82, align 16, !tbaa !8
-double*8B

	full_text

double* %82
vcall8Bl
j
	full_text]
[
Y%856 = tail call double @llvm.fmuladd.f64(double %855, double -4.000000e+00, double %854)
,double8B

	full_text

double %855
,double8B

	full_text

double %854
vcall8Bl
j
	full_text]
[
Y%857 = tail call double @llvm.fmuladd.f64(double %856, double -2.500000e-01, double %789)
,double8B

	full_text

double %856
,double8B

	full_text

double %789
Pstore8BE
C
	full_text6
4
2store double %857, double* %778, align 8, !tbaa !8
,double8B

	full_text

double %857
.double*8B

	full_text

double* %778
Pload8BF
D
	full_text7
5
3%858 = load double, double* %159, align 8, !tbaa !8
.double*8B

	full_text

double* %159
vcall8Bl
j
	full_text]
[
Y%859 = tail call double @llvm.fmuladd.f64(double %793, double -4.000000e+00, double %858)
,double8B

	full_text

double %793
,double8B

	full_text

double %858
Oload8BE
C
	full_text6
4
2%860 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
ucall8Bk
i
	full_text\
Z
X%861 = tail call double @llvm.fmuladd.f64(double %860, double 6.000000e+00, double %859)
,double8B

	full_text

double %860
,double8B

	full_text

double %859
Oload8BE
C
	full_text6
4
2%862 = load double, double* %87, align 8, !tbaa !8
-double*8B

	full_text

double* %87
vcall8Bl
j
	full_text]
[
Y%863 = tail call double @llvm.fmuladd.f64(double %862, double -4.000000e+00, double %861)
,double8B

	full_text

double %862
,double8B

	full_text

double %861
vcall8Bl
j
	full_text]
[
Y%864 = tail call double @llvm.fmuladd.f64(double %863, double -2.500000e-01, double %802)
,double8B

	full_text

double %863
,double8B

	full_text

double %802
Pstore8BE
C
	full_text6
4
2store double %864, double* %790, align 8, !tbaa !8
,double8B

	full_text

double %864
.double*8B

	full_text

double* %790
Qload8BG
E
	full_text8
6
4%865 = load double, double* %164, align 16, !tbaa !8
.double*8B

	full_text

double* %164
Qload8BG
E
	full_text8
6
4%866 = load double, double* %161, align 16, !tbaa !8
.double*8B

	full_text

double* %161
vcall8Bl
j
	full_text]
[
Y%867 = tail call double @llvm.fmuladd.f64(double %866, double -4.000000e+00, double %865)
,double8B

	full_text

double %866
,double8B

	full_text

double %865
ucall8Bk
i
	full_text\
Z
X%868 = tail call double @llvm.fmuladd.f64(double %805, double 6.000000e+00, double %867)
,double8B

	full_text

double %805
,double8B

	full_text

double %867
Pload8BF
D
	full_text7
5
3%869 = load double, double* %92, align 16, !tbaa !8
-double*8B

	full_text

double* %92
vcall8Bl
j
	full_text]
[
Y%870 = tail call double @llvm.fmuladd.f64(double %869, double -4.000000e+00, double %868)
,double8B

	full_text

double %869
,double8B

	full_text

double %868
vcall8Bl
j
	full_text]
[
Y%871 = tail call double @llvm.fmuladd.f64(double %870, double -2.500000e-01, double %833)
,double8B

	full_text

double %870
,double8B

	full_text

double %833
Pstore8BE
C
	full_text6
4
2store double %871, double* %803, align 8, !tbaa !8
,double8B

	full_text

double %871
.double*8B

	full_text

double* %803
qgetelementptr8B^
\
	full_textO
M
K%872 = getelementptr inbounds [5 x double], [5 x double]* %16, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %16
Qstore8BF
D
	full_text7
5
3store double %702, double* %872, align 16, !tbaa !8
,double8B

	full_text

double %702
.double*8B

	full_text

double* %872
Pstore8BE
C
	full_text6
4
2store double %701, double* %149, align 8, !tbaa !8
,double8B

	full_text

double %701
.double*8B

	full_text

double* %149
Qstore8BF
D
	full_text7
5
3store double %700, double* %154, align 16, !tbaa !8
,double8B

	full_text

double %700
.double*8B

	full_text

double* %154
Jstore8B?
=
	full_text0
.
,store i64 %699, i64* %160, align 8, !tbaa !8
&i648B

	full_text


i64 %699
(i64*8B

	full_text

	i64* %160
Kstore8B@
>
	full_text1
/
-store i64 %698, i64* %165, align 16, !tbaa !8
&i648B

	full_text


i64 %698
(i64*8B

	full_text

	i64* %165
qgetelementptr8B^
\
	full_textO
M
K%873 = getelementptr inbounds [5 x double], [5 x double]* %15, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %15
Qstore8BF
D
	full_text7
5
3store double %697, double* %873, align 16, !tbaa !8
,double8B

	full_text

double %697
.double*8B

	full_text

double* %873
Pstore8BE
C
	full_text6
4
2store double %696, double* %146, align 8, !tbaa !8
,double8B

	full_text

double %696
.double*8B

	full_text

double* %146
Qstore8BF
D
	full_text7
5
3store double %695, double* %151, align 16, !tbaa !8
,double8B

	full_text

double %695
.double*8B

	full_text

double* %151
Pstore8BE
C
	full_text6
4
2store double %694, double* %156, align 8, !tbaa !8
,double8B

	full_text

double %694
.double*8B

	full_text

double* %156
Kstore8B@
>
	full_text1
/
-store i64 %693, i64* %162, align 16, !tbaa !8
&i648B

	full_text


i64 %693
(i64*8B

	full_text

	i64* %162
qgetelementptr8B^
\
	full_textO
M
K%874 = getelementptr inbounds [5 x double], [5 x double]* %12, i64 0, i64 0
9[5 x double]*8B$
"
	full_text

[5 x double]* %12
Qstore8BF
D
	full_text7
5
3store double %691, double* %874, align 16, !tbaa !8
,double8B

	full_text

double %691
.double*8B

	full_text

double* %874
Ostore8BD
B
	full_text5
3
1store double %690, double* %52, align 8, !tbaa !8
,double8B

	full_text

double %690
-double*8B

	full_text

double* %52
Pstore8BE
C
	full_text6
4
2store double %689, double* %57, align 16, !tbaa !8
,double8B

	full_text

double %689
-double*8B

	full_text

double* %57
Ostore8BD
B
	full_text5
3
1store double %688, double* %62, align 8, !tbaa !8
,double8B

	full_text

double %688
-double*8B

	full_text

double* %62
Pstore8BE
C
	full_text6
4
2store double %687, double* %67, align 16, !tbaa !8
,double8B

	full_text

double %687
-double*8B

	full_text

double* %67
Jstore8B?
=
	full_text0
.
,store i64 %727, i64* %73, align 16, !tbaa !8
&i648B

	full_text


i64 %727
'i64*8B

	full_text


i64* %73
Istore8B>
<
	full_text/
-
+store i64 %730, i64* %78, align 8, !tbaa !8
&i648B

	full_text


i64 %730
'i64*8B

	full_text


i64* %78
Jstore8B?
=
	full_text0
.
,store i64 %733, i64* %83, align 16, !tbaa !8
&i648B

	full_text


i64 %733
'i64*8B

	full_text


i64* %83
Istore8B>
<
	full_text/
-
+store i64 %736, i64* %88, align 8, !tbaa !8
&i648B

	full_text


i64 %736
'i64*8B

	full_text


i64* %88
Jstore8B?
=
	full_text0
.
,store i64 %739, i64* %93, align 16, !tbaa !8
&i648B

	full_text


i64 %739
'i64*8B

	full_text


i64* %93
”getelementptr8B€
~
	full_textq
o
m%875 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %33, i64 %41, i64 %43, i64 %724
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %33
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Pload8BF
D
	full_text7
5
3%876 = load double, double* %875, align 8, !tbaa !8
.double*8B

	full_text

double* %875
”getelementptr8B€
~
	full_textq
o
m%877 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %34, i64 %41, i64 %43, i64 %724
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %34
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Pload8BF
D
	full_text7
5
3%878 = load double, double* %877, align 8, !tbaa !8
.double*8B

	full_text

double* %877
”getelementptr8B€
~
	full_textq
o
m%879 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %35, i64 %41, i64 %43, i64 %724
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %35
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Pload8BF
D
	full_text7
5
3%880 = load double, double* %879, align 8, !tbaa !8
.double*8B

	full_text

double* %879
”getelementptr8B€
~
	full_textq
o
m%881 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %36, i64 %41, i64 %43, i64 %724
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %36
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Pload8BF
D
	full_text7
5
3%882 = load double, double* %881, align 8, !tbaa !8
.double*8B

	full_text

double* %881
”getelementptr8B€
~
	full_textq
o
m%883 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %37, i64 %41, i64 %43, i64 %724
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %37
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Pload8BF
D
	full_text7
5
3%884 = load double, double* %883, align 8, !tbaa !8
.double*8B

	full_text

double* %883
”getelementptr8B€
~
	full_textq
o
m%885 = getelementptr inbounds [163 x [163 x double]], [163 x [163 x double]]* %38, i64 %41, i64 %43, i64 %724
M[163 x [163 x double]]*8B.
,
	full_text

[163 x [163 x double]]* %38
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %724
Pload8BF
D
	full_text7
5
3%886 = load double, double* %885, align 8, !tbaa !8
.double*8B

	full_text

double* %885
«getelementptr8B—
”
	full_text†
ƒ
€%887 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %740, i64 0
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%888 = load double, double* %887, align 8, !tbaa !8
.double*8B

	full_text

double* %887
Abitcast8B4
2
	full_text%
#
!%889 = bitcast i64 %727 to double
&i648B

	full_text


i64 %727
vcall8Bl
j
	full_text]
[
Y%890 = tail call double @llvm.fmuladd.f64(double %691, double -2.000000e+00, double %889)
,double8B

	full_text

double %691
,double8B

	full_text

double %889
:fadd8B0
.
	full_text!

%891 = fadd double %890, %697
,double8B

	full_text

double %890
,double8B

	full_text

double %697
{call8Bq
o
	full_textb
`
^%892 = tail call double @llvm.fmuladd.f64(double %891, double 0x40D2FC3000000001, double %888)
,double8B

	full_text

double %891
,double8B

	full_text

double %888
Abitcast8B4
2
	full_text%
#
!%893 = bitcast i64 %730 to double
&i648B

	full_text


i64 %730
:fsub8B0
.
	full_text!

%894 = fsub double %893, %696
,double8B

	full_text

double %893
,double8B

	full_text

double %696
vcall8Bl
j
	full_text]
[
Y%895 = tail call double @llvm.fmuladd.f64(double %894, double -8.050000e+01, double %892)
,double8B

	full_text

double %894
,double8B

	full_text

double %892
«getelementptr8B—
”
	full_text†
ƒ
€%896 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %740, i64 1
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%897 = load double, double* %896, align 8, !tbaa !8
.double*8B

	full_text

double* %896
vcall8Bl
j
	full_text]
[
Y%898 = tail call double @llvm.fmuladd.f64(double %690, double -2.000000e+00, double %893)
,double8B

	full_text

double %690
,double8B

	full_text

double %893
:fadd8B0
.
	full_text!

%899 = fadd double %696, %898
,double8B

	full_text

double %696
,double8B

	full_text

double %898
{call8Bq
o
	full_textb
`
^%900 = tail call double @llvm.fmuladd.f64(double %899, double 0x40D2FC3000000001, double %897)
,double8B

	full_text

double %899
,double8B

	full_text

double %897
vcall8Bl
j
	full_text]
[
Y%901 = tail call double @llvm.fmuladd.f64(double %742, double -2.000000e+00, double %876)
,double8B

	full_text

double %742
,double8B

	full_text

double %876
:fadd8B0
.
	full_text!

%902 = fadd double %718, %901
,double8B

	full_text

double %718
,double8B

	full_text

double %901
{call8Bq
o
	full_textb
`
^%903 = tail call double @llvm.fmuladd.f64(double %902, double 0x40AB004444444445, double %900)
,double8B

	full_text

double %902
,double8B

	full_text

double %900
:fmul8B0
.
	full_text!

%904 = fmul double %718, %696
,double8B

	full_text

double %718
,double8B

	full_text

double %696
Cfsub8B9
7
	full_text*
(
&%905 = fsub double -0.000000e+00, %904
,double8B

	full_text

double %904
mcall8Bc
a
	full_textT
R
P%906 = tail call double @llvm.fmuladd.f64(double %893, double %876, double %905)
,double8B

	full_text

double %893
,double8B

	full_text

double %876
,double8B

	full_text

double %905
Abitcast8B4
2
	full_text%
#
!%907 = bitcast i64 %739 to double
&i648B

	full_text


i64 %739
:fsub8B0
.
	full_text!

%908 = fsub double %907, %886
,double8B

	full_text

double %907
,double8B

	full_text

double %886
:fsub8B0
.
	full_text!

%909 = fsub double %908, %692
,double8B

	full_text

double %908
,double8B

	full_text

double %692
:fadd8B0
.
	full_text!

%910 = fadd double %715, %909
,double8B

	full_text

double %715
,double8B

	full_text

double %909
ucall8Bk
i
	full_text\
Z
X%911 = tail call double @llvm.fmuladd.f64(double %910, double 4.000000e-01, double %906)
,double8B

	full_text

double %910
,double8B

	full_text

double %906
vcall8Bl
j
	full_text]
[
Y%912 = tail call double @llvm.fmuladd.f64(double %911, double -8.050000e+01, double %903)
,double8B

	full_text

double %911
,double8B

	full_text

double %903
«getelementptr8B—
”
	full_text†
ƒ
€%913 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %740, i64 2
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%914 = load double, double* %913, align 8, !tbaa !8
.double*8B

	full_text

double* %913
Abitcast8B4
2
	full_text%
#
!%915 = bitcast i64 %733 to double
&i648B

	full_text


i64 %733
vcall8Bl
j
	full_text]
[
Y%916 = tail call double @llvm.fmuladd.f64(double %689, double -2.000000e+00, double %915)
,double8B

	full_text

double %689
,double8B

	full_text

double %915
:fadd8B0
.
	full_text!

%917 = fadd double %916, %695
,double8B

	full_text

double %916
,double8B

	full_text

double %695
{call8Bq
o
	full_textb
`
^%918 = tail call double @llvm.fmuladd.f64(double %917, double 0x40D2FC3000000001, double %914)
,double8B

	full_text

double %917
,double8B

	full_text

double %914
vcall8Bl
j
	full_text]
[
Y%919 = tail call double @llvm.fmuladd.f64(double %744, double -2.000000e+00, double %878)
,double8B

	full_text

double %744
,double8B

	full_text

double %878
:fadd8B0
.
	full_text!

%920 = fadd double %716, %919
,double8B

	full_text

double %716
,double8B

	full_text

double %919
{call8Bq
o
	full_textb
`
^%921 = tail call double @llvm.fmuladd.f64(double %920, double 0x40A4403333333334, double %918)
,double8B

	full_text

double %920
,double8B

	full_text

double %918
:fmul8B0
.
	full_text!

%922 = fmul double %718, %695
,double8B

	full_text

double %718
,double8B

	full_text

double %695
Cfsub8B9
7
	full_text*
(
&%923 = fsub double -0.000000e+00, %922
,double8B

	full_text

double %922
mcall8Bc
a
	full_textT
R
P%924 = tail call double @llvm.fmuladd.f64(double %915, double %876, double %923)
,double8B

	full_text

double %915
,double8B

	full_text

double %876
,double8B

	full_text

double %923
vcall8Bl
j
	full_text]
[
Y%925 = tail call double @llvm.fmuladd.f64(double %924, double -8.050000e+01, double %921)
,double8B

	full_text

double %924
,double8B

	full_text

double %921
«getelementptr8B—
”
	full_text†
ƒ
€%926 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %740, i64 3
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%927 = load double, double* %926, align 8, !tbaa !8
.double*8B

	full_text

double* %926
Abitcast8B4
2
	full_text%
#
!%928 = bitcast i64 %736 to double
&i648B

	full_text


i64 %736
vcall8Bl
j
	full_text]
[
Y%929 = tail call double @llvm.fmuladd.f64(double %688, double -2.000000e+00, double %928)
,double8B

	full_text

double %688
,double8B

	full_text

double %928
:fadd8B0
.
	full_text!

%930 = fadd double %929, %694
,double8B

	full_text

double %929
,double8B

	full_text

double %694
{call8Bq
o
	full_textb
`
^%931 = tail call double @llvm.fmuladd.f64(double %930, double 0x40D2FC3000000001, double %927)
,double8B

	full_text

double %930
,double8B

	full_text

double %927
vcall8Bl
j
	full_text]
[
Y%932 = tail call double @llvm.fmuladd.f64(double %746, double -2.000000e+00, double %880)
,double8B

	full_text

double %746
,double8B

	full_text

double %880
:fadd8B0
.
	full_text!

%933 = fadd double %709, %932
,double8B

	full_text

double %709
,double8B

	full_text

double %932
{call8Bq
o
	full_textb
`
^%934 = tail call double @llvm.fmuladd.f64(double %933, double 0x40A4403333333334, double %931)
,double8B

	full_text

double %933
,double8B

	full_text

double %931
:fmul8B0
.
	full_text!

%935 = fmul double %718, %694
,double8B

	full_text

double %718
,double8B

	full_text

double %694
Cfsub8B9
7
	full_text*
(
&%936 = fsub double -0.000000e+00, %935
,double8B

	full_text

double %935
mcall8Bc
a
	full_textT
R
P%937 = tail call double @llvm.fmuladd.f64(double %928, double %876, double %936)
,double8B

	full_text

double %928
,double8B

	full_text

double %876
,double8B

	full_text

double %936
vcall8Bl
j
	full_text]
[
Y%938 = tail call double @llvm.fmuladd.f64(double %937, double -8.050000e+01, double %934)
,double8B

	full_text

double %937
,double8B

	full_text

double %934
«getelementptr8B—
”
	full_text†
ƒ
€%939 = getelementptr inbounds [163 x [163 x [5 x double]]], [163 x [163 x [5 x double]]]* %39, i64 %41, i64 %43, i64 %740, i64 4
Y[163 x [163 x [5 x double]]]*8B4
2
	full_text%
#
![163 x [163 x [5 x double]]]* %39
%i648B

	full_text
	
i64 %41
%i648B

	full_text
	
i64 %43
&i648B

	full_text


i64 %740
Pload8BF
D
	full_text7
5
3%940 = load double, double* %939, align 8, !tbaa !8
.double*8B

	full_text

double* %939
vcall8Bl
j
	full_text]
[
Y%941 = tail call double @llvm.fmuladd.f64(double %687, double -2.000000e+00, double %907)
,double8B

	full_text

double %687
,double8B

	full_text

double %907
:fadd8B0
.
	full_text!

%942 = fadd double %692, %941
,double8B

	full_text

double %692
,double8B

	full_text

double %941
{call8Bq
o
	full_textb
`
^%943 = tail call double @llvm.fmuladd.f64(double %942, double 0x40D2FC3000000001, double %940)
,double8B

	full_text

double %942
,double8B

	full_text

double %940
vcall8Bl
j
	full_text]
[
Y%944 = tail call double @llvm.fmuladd.f64(double %748, double -2.000000e+00, double %882)
,double8B

	full_text

double %748
,double8B

	full_text

double %882
:fadd8B0
.
	full_text!

%945 = fadd double %711, %944
,double8B

	full_text

double %711
,double8B

	full_text

double %944
{call8Bq
o
	full_textb
`
^%946 = tail call double @llvm.fmuladd.f64(double %945, double 0xC0A370D4FDF3B645, double %943)
,double8B

	full_text

double %945
,double8B

	full_text

double %943
Bfmul8B8
6
	full_text)
'
%%947 = fmul double %742, 2.000000e+00
,double8B

	full_text

double %742
:fmul8B0
.
	full_text!

%948 = fmul double %742, %947
,double8B

	full_text

double %742
,double8B

	full_text

double %947
Cfsub8B9
7
	full_text*
(
&%949 = fsub double -0.000000e+00, %948
,double8B

	full_text

double %948
mcall8Bc
a
	full_textT
R
P%950 = tail call double @llvm.fmuladd.f64(double %876, double %876, double %949)
,double8B

	full_text

double %876
,double8B

	full_text

double %876
,double8B

	full_text

double %949
mcall8Bc
a
	full_textT
R
P%951 = tail call double @llvm.fmuladd.f64(double %718, double %718, double %950)
,double8B

	full_text

double %718
,double8B

	full_text

double %718
,double8B

	full_text

double %950
{call8Bq
o
	full_textb
`
^%952 = tail call double @llvm.fmuladd.f64(double %951, double 0x407B004444444445, double %946)
,double8B

	full_text

double %951
,double8B

	full_text

double %946
Bfmul8B8
6
	full_text)
'
%%953 = fmul double %687, 2.000000e+00
,double8B

	full_text

double %687
:fmul8B0
.
	full_text!

%954 = fmul double %750, %953
,double8B

	full_text

double %750
,double8B

	full_text

double %953
Cfsub8B9
7
	full_text*
(
&%955 = fsub double -0.000000e+00, %954
,double8B

	full_text

double %954
mcall8Bc
a
	full_textT
R
P%956 = tail call double @llvm.fmuladd.f64(double %907, double %884, double %955)
,double8B

	full_text

double %907
,double8B

	full_text

double %884
,double8B

	full_text

double %955
mcall8Bc
a
	full_textT
R
P%957 = tail call double @llvm.fmuladd.f64(double %692, double %713, double %956)
,double8B

	full_text

double %692
,double8B

	full_text

double %713
,double8B

	full_text

double %956
{call8Bq
o
	full_textb
`
^%958 = tail call double @llvm.fmuladd.f64(double %957, double 0x40B3D884189374BC, double %952)
,double8B

	full_text

double %957
,double8B

	full_text

double %952
Bfmul8B8
6
	full_text)
'
%%959 = fmul double %886, 4.000000e-01
,double8B

	full_text

double %886
Cfsub8B9
7
	full_text*
(
&%960 = fsub double -0.000000e+00, %959
,double8B

	full_text

double %959
ucall8Bk
i
	full_text\
Z
X%961 = tail call double @llvm.fmuladd.f64(double %907, double 1.400000e+00, double %960)
,double8B

	full_text

double %907
,double8B

	full_text

double %960
Bfmul8B8
6
	full_text)
'
%%962 = fmul double %715, 4.000000e-01
,double8B

	full_text

double %715
Cfsub8B9
7
	full_text*
(
&%963 = fsub double -0.000000e+00, %962
,double8B

	full_text

double %962
ucall8Bk
i
	full_text\
Z
X%964 = tail call double @llvm.fmuladd.f64(double %692, double 1.400000e+00, double %963)
,double8B

	full_text

double %692
,double8B

	full_text

double %963
:fmul8B0
.
	full_text!

%965 = fmul double %718, %964
,double8B

	full_text

double %718
,double8B

	full_text

double %964
Cfsub8B9
7
	full_text*
(
&%966 = fsub double -0.000000e+00, %965
,double8B

	full_text

double %965
mcall8Bc
a
	full_textT
R
P%967 = tail call double @llvm.fmuladd.f64(double %961, double %876, double %966)
,double8B

	full_text

double %961
,double8B

	full_text

double %876
,double8B

	full_text

double %966
vcall8Bl
j
	full_text]
[
Y%968 = tail call double @llvm.fmuladd.f64(double %967, double -8.050000e+01, double %958)
,double8B

	full_text

double %967
,double8B

	full_text

double %958
Pload8BF
D
	full_text7
5
3%969 = load double, double* %686, align 8, !tbaa !8
.double*8B

	full_text

double* %686
Qload8BG
E
	full_text8
6
4%970 = load double, double* %142, align 16, !tbaa !8
.double*8B

	full_text

double* %142
vcall8Bl
j
	full_text]
[
Y%971 = tail call double @llvm.fmuladd.f64(double %970, double -4.000000e+00, double %969)
,double8B

	full_text

double %970
,double8B

	full_text

double %969
Pload8BF
D
	full_text7
5
3%972 = load double, double* %47, align 16, !tbaa !8
-double*8B

	full_text

double* %47
ucall8Bk
i
	full_text\
Z
X%973 = tail call double @llvm.fmuladd.f64(double %972, double 5.000000e+00, double %971)
,double8B

	full_text

double %972
,double8B

	full_text

double %971
vcall8Bl
j
	full_text]
[
Y%974 = tail call double @llvm.fmuladd.f64(double %973, double -2.500000e-01, double %895)
,double8B

	full_text

double %973
,double8B

	full_text

double %895
Pstore8BE
C
	full_text6
4
2store double %974, double* %887, align 8, !tbaa !8
,double8B

	full_text

double %974
.double*8B

	full_text

double* %887
Pload8BF
D
	full_text7
5
3%975 = load double, double* %149, align 8, !tbaa !8
.double*8B

	full_text

double* %149
Pload8BF
D
	full_text7
5
3%976 = load double, double* %146, align 8, !tbaa !8
.double*8B

	full_text

double* %146
vcall8Bl
j
	full_text]
[
Y%977 = tail call double @llvm.fmuladd.f64(double %976, double -4.000000e+00, double %975)
,double8B

	full_text

double %976
,double8B

	full_text

double %975
Oload8BE
C
	full_text6
4
2%978 = load double, double* %52, align 8, !tbaa !8
-double*8B

	full_text

double* %52
ucall8Bk
i
	full_text\
Z
X%979 = tail call double @llvm.fmuladd.f64(double %978, double 5.000000e+00, double %977)
,double8B

	full_text

double %978
,double8B

	full_text

double %977
vcall8Bl
j
	full_text]
[
Y%980 = tail call double @llvm.fmuladd.f64(double %979, double -2.500000e-01, double %912)
,double8B

	full_text

double %979
,double8B

	full_text

double %912
Pstore8BE
C
	full_text6
4
2store double %980, double* %896, align 8, !tbaa !8
,double8B

	full_text

double %980
.double*8B

	full_text

double* %896
Qload8BG
E
	full_text8
6
4%981 = load double, double* %154, align 16, !tbaa !8
.double*8B

	full_text

double* %154
Qload8BG
E
	full_text8
6
4%982 = load double, double* %151, align 16, !tbaa !8
.double*8B

	full_text

double* %151
vcall8Bl
j
	full_text]
[
Y%983 = tail call double @llvm.fmuladd.f64(double %982, double -4.000000e+00, double %981)
,double8B

	full_text

double %982
,double8B

	full_text

double %981
Pload8BF
D
	full_text7
5
3%984 = load double, double* %57, align 16, !tbaa !8
-double*8B

	full_text

double* %57
ucall8Bk
i
	full_text\
Z
X%985 = tail call double @llvm.fmuladd.f64(double %984, double 5.000000e+00, double %983)
,double8B

	full_text

double %984
,double8B

	full_text

double %983
vcall8Bl
j
	full_text]
[
Y%986 = tail call double @llvm.fmuladd.f64(double %985, double -2.500000e-01, double %925)
,double8B

	full_text

double %985
,double8B

	full_text

double %925
Pstore8BE
C
	full_text6
4
2store double %986, double* %913, align 8, !tbaa !8
,double8B

	full_text

double %986
.double*8B

	full_text

double* %913
Pload8BF
D
	full_text7
5
3%987 = load double, double* %159, align 8, !tbaa !8
.double*8B

	full_text

double* %159
Pload8BF
D
	full_text7
5
3%988 = load double, double* %156, align 8, !tbaa !8
.double*8B

	full_text

double* %156
vcall8Bl
j
	full_text]
[
Y%989 = tail call double @llvm.fmuladd.f64(double %988, double -4.000000e+00, double %987)
,double8B

	full_text

double %988
,double8B

	full_text

double %987
Oload8BE
C
	full_text6
4
2%990 = load double, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
ucall8Bk
i
	full_text\
Z
X%991 = tail call double @llvm.fmuladd.f64(double %990, double 5.000000e+00, double %989)
,double8B

	full_text

double %990
,double8B

	full_text

double %989
vcall8Bl
j
	full_text]
[
Y%992 = tail call double @llvm.fmuladd.f64(double %991, double -2.500000e-01, double %938)
,double8B

	full_text

double %991
,double8B

	full_text

double %938
Pstore8BE
C
	full_text6
4
2store double %992, double* %926, align 8, !tbaa !8
,double8B

	full_text

double %992
.double*8B

	full_text

double* %926
Qload8BG
E
	full_text8
6
4%993 = load double, double* %164, align 16, !tbaa !8
.double*8B

	full_text

double* %164
Qload8BG
E
	full_text8
6
4%994 = load double, double* %161, align 16, !tbaa !8
.double*8B

	full_text

double* %161
vcall8Bl
j
	full_text]
[
Y%995 = tail call double @llvm.fmuladd.f64(double %994, double -4.000000e+00, double %993)
,double8B

	full_text

double %994
,double8B

	full_text

double %993
Pload8BF
D
	full_text7
5
3%996 = load double, double* %67, align 16, !tbaa !8
-double*8B

	full_text

double* %67
ucall8Bk
i
	full_text\
Z
X%997 = tail call double @llvm.fmuladd.f64(double %996, double 5.000000e+00, double %995)
,double8B

	full_text

double %996
,double8B

	full_text

double %995
vcall8Bl
j
	full_text]
[
Y%998 = tail call double @llvm.fmuladd.f64(double %997, double -2.500000e-01, double %968)
,double8B

	full_text

double %997
,double8B

	full_text

double %968
Pstore8BE
C
	full_text6
4
2store double %998, double* %939, align 8, !tbaa !8
,double8B

	full_text

double %998
.double*8B

	full_text

double* %939
(br8B 

	full_text

br label %999
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %21) #4
%i8*8B

	full_text
	
i8* %21
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %20) #4
%i8*8B

	full_text
	
i8* %20
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %19) #4
%i8*8B

	full_text
	
i8* %19
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %18) #4
%i8*8B

	full_text
	
i8* %18
Zcall8BP
N
	full_textA
?
=call void @llvm.lifetime.end.p0i8(i64 40, i8* nonnull %17) #4
%i8*8B

	full_text
	
i8* %17
$ret8B

	full_text


ret void
,double*8B

	full_text


double* %3
,double*8B

	full_text


double* %6
,double*8B

	full_text


double* %2
,double*8B

	full_text


double* %0
,double*8B

	full_text


double* %4
%i328B

	full_text
	
i32 %10
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %1
$i328B

	full_text


i32 %9
,double*8B

	full_text


double* %5
,double*8B

	full_text


double* %7
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
5double8B'
%
	full_text

double -2.000000e+00
:double8B,
*
	full_text

double 0x40AB004444444445
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 5.000000e+00
4double8B&
$
	full_text

double 4.000000e+00
$i648B

	full_text


i64 40
$i648B

	full_text


i64 32
:double8B,
*
	full_text

double 0x40B3D884189374BC
4double8B&
$
	full_text

double 1.400000e+00
:double8B,
*
	full_text

double 0xC0A370D4FDF3B645
:double8B,
*
	full_text

double 0x407B004444444445
5double8B'
%
	full_text

double -2.500000e-01
4double8B&
$
	full_text

double 4.000000e-01
#i648B

	full_text	

i64 2
:double8B,
*
	full_text

double 0x40D2FC3000000001
:double8B,
*
	full_text

double 0x40A4403333333334
4double8B&
$
	full_text

double 6.000000e+00
5double8B'
%
	full_text

double -4.000000e+00
#i648B

	full_text	

i64 3
5double8B'
%
	full_text

double -0.000000e+00
4double8B&
$
	full_text

double 2.000000e+00
5double8B'
%
	full_text

double -8.050000e+01
#i328B

	full_text	

i32 5
$i328B

	full_text


i32 -1
#i328B

	full_text	

i32 0
#i328B

	full_text	

i32 1
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 1       	  
 

                      !    "# "" $% $$ &' && () (* (( +, +- .. // 00 11 22 33 44 56 55 78 77 9: 99 ;< ;; => =? =@ == AB AA CD CC EF EE GH GG IJ IK II LM LN LO LL PQ PP RS RR TU TT VW VV XY XZ XX [\ [] [^ [[ _` __ ab aa cd cc ef ee gh gi gg jk jl jm jj no nn pq pp rs rr tu tt vw vx vv yz y{ y| yy }~ }} €  ‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆ
‹ ˆˆ Œ ŒŒ Ž ŽŽ ‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —
š —— ›œ ›› ž  Ÿ  ŸŸ ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦
© ¦¦ ª« ªª ¬­ ¬¬ ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µ
· µ
¸ µµ ¹º ¹¹ »¼ »» ½¾ ½½ ¿À ¿¿ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ ÈÈ ÊË ÊÊ ÌÍ ÌÌ ÎÏ ÎÎ ÐÑ Ð
Ò Ð
Ó ÐÐ ÔÕ ÔÔ Ö× ÖÖ ØÙ ØØ ÚÛ Ú
Ü Ú
Ý ÚÚ Þß ÞÞ àá àà âã ââ äå ää æç æ
è æ
é ææ êë êê ìí ìì îï îî ðñ ðð òó ò
ô ò
õ òò ö÷ öö øù øø úû úú üý üü þÿ þ
€ þ
 þþ ‚ƒ ‚‚ „… „„ †‡ †† ˆ‰ ˆˆ Š‹ Š
Œ Š
 ŠŠ Ž ŽŽ ‘ 
’ 
“  ”• ”” –— –
˜ –
™ –– š› šš œ œ
ž œ
Ÿ œœ  ¡    ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦¦ ¨© ¨
ª ¨
« ¨¨ ¬­ ¬¬ ®¯ ®
° ®
± ®® ²³ ²² ´µ ´
¶ ´
· ´´ ¸¹ ¸¸ º» º
¼ º
½ ºº ¾¿ ¾¾ ÀÁ À
Â À
Ã ÀÀ ÄÅ ÄÄ ÆÇ Æ
È Æ
É ÆÆ ÊË ÊÊ ÌÍ Ì
Î Ì
Ï ÌÌ ÐÑ ÐÐ ÒÓ ÒÒ ÔÕ ÔÔ Ö× ÖÖ ØÙ ØØ ÚÛ Ú
Ü ÚÚ ÝÞ ÝÝ ßà ßß áâ áá ãä ãã åæ åå çè ç
é çç êë êê ìí ìì îï îî ðñ ðð òó òò ôõ ô
ö ôô ÷ø ÷÷ ùú ùù ûü ûû ýþ ýý ÿ€ ÿÿ ‚ 
ƒ  „… „„ †‡ †† ˆ‰ ˆˆ Š‹ ŠŠ Œ ŒŒ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾
Á ¾¾ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë É
Ì ÉÉ ÍÎ ÍÍ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö Ô
× ÔÔ ØÙ ØØ ÚÛ ÚÚ ÜÝ Ü
Þ ÜÜ ßà ß
á ß
â ßß ãä ãã åæ åå çè ç
é çç êë ê
ì ê
í êê îï îî ðñ ðð òó ò
ô òò õö õ
÷ õ
ø õõ ùú ùù ûü û
ý û
þ ûû ÿ€ ÿÿ ‚ 
ƒ 
„  …† …… ‡ˆ ‡
‰ ‡
Š ‡‡ ‹Œ ‹‹ Ž 
 
  ‘’ ‘‘ “” “
• “
– ““ —˜ —— ™š ™
› ™
œ ™™ ž  Ÿ  ŸŸ ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸
» ¸¸ ¼½ ¼¼ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ Õ
Ö ÕÕ ×Ø ×
Ù ×
Ú ×× ÛÜ ÛÛ ÝÞ Ý
ß ÝÝ àá àà âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ð î
ñ îî òó òò ôõ ôô ö÷ öö øù ø
ú øø ûü ûû ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ 
  ‘’ ‘
“ ‘
” ‘‘ •– •
— •• ˜™ ˜
š ˜
› ˜˜ œ œœ žŸ žž  ¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹
º ¹¹ »¼ »
½ »
¾ »» ¿À ¿
Á ¿¿ ÂÃ Â
Ä Â
Å ÂÂ ÆÇ ÆÆ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ á
â áá ãä ã
å ã
æ ãã çè ç
é ç
ê çç ëì ë
í ëë îï îî ðñ ð
ò ðð ó
ô óó õö õ
÷ õ
ø õõ ùú ù
û ù
ü ùù ýþ ý
ÿ ýý € €€ ‚
ƒ ‚‚ „… „
† „„ ‡ˆ ‡‡ ‰
Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘
’ ‘‘ “” “
• “
– ““ —˜ —
™ —— š› šš œ œœ žŸ žž  
¡    ¢£ ¢
¤ ¢¢ ¥¦ ¥¥ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²² ´µ ´´ ¶· ¶¶ ¸
¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ ÈÈ ÊË ÊÊ ÌÍ ÌÌ Î
Ï ÎÎ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß ÞÞ àá àà âã ââ ä
å ää æç æ
è ææ éê éé ëì ë
í ëë îï î
ð îî ñò ñ
ó ññ ôõ ôô ö÷ öö ø
ù øø úû ú
ü úú ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ ÈÈ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü Ú
Ý ÚÚ Þß ÞÞ àá àà âã â
ä ââ åæ å
ç å
è åå éê éé ëì ëë íî í
ï íí ðñ ð
ò ð
ó ðð ôõ ôô ö÷ öö øù ø
ú øø ûü û
ý û
þ ûû ÿ€ ÿÿ ‚ 
ƒ 
„  …† …… ‡ˆ ‡
‰ ‡
Š ‡‡ ‹Œ ‹‹ Ž 
 
  ‘’ ‘‘ “” “
• “
– ““ —˜ —— ™š ™
› ™
œ ™™ ž  Ÿ  Ÿ
¡ Ÿ
¢ ŸŸ £¤ ££ ¥¦ ¥¥ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸
» ¸¸ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ Ó
Ô ÓÓ ÕÖ Õ
× Õ
Ø ÕÕ ÙÚ ÙÙ ÛÜ Û
Ý ÛÛ Þß ÞÞ àá à
â àà ãä ã
å ãã æç æ
è ææ éê é
ë éé ìí ì
î ì
ï ìì ðñ ðð òó òò ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ýþ ý
ÿ ýý €		 €	
‚	 €	€	 ƒ	„	 ƒ	
…	 ƒ	ƒ	 †	‡	 †	
ˆ	 †	†	 ‰	
Š	 ‰	‰	 ‹	Œ	 ‹	
	 ‹	
Ž	 ‹	‹	 		 	
‘	 		 ’	“	 ’	
”	 ’	
•	 ’	’	 –	—	 –	–	 ˜	™	 ˜	˜	 š	›	 š	
œ	 š	š	 	ž	 	
Ÿ	 		  	¡	  	
¢	  	 	 £	¤	 £	
¥	 £	£	 ¦	§	 ¦	
¨	 ¦	¦	 ©	ª	 ©	
«	 ©	©	 ¬	­	 ¬	
®	 ¬	¬	 ¯	
°	 ¯	¯	 ±	²	 ±	
³	 ±	
´	 ±	±	 µ	¶	 µ	
·	 µ	µ	 ¸	¹	 ¸	
º	 ¸	
»	 ¸	¸	 ¼	½	 ¼	¼	 ¾	¿	 ¾	
À	 ¾	¾	 Á	Â	 Á	
Ã	 Á	Á	 Ä	Å	 Ä	
Æ	 Ä	Ä	 Ç	È	 Ç	
É	 Ç	Ç	 Ê	Ë	 Ê	
Ì	 Ê	Ê	 Í	Î	 Í	
Ï	 Í	Í	 Ð	Ñ	 Ð	Ð	 Ò	Ó	 Ò	
Ô	 Ò	Ò	 Õ	
Ö	 Õ	Õ	 ×	Ø	 ×	
Ù	 ×	
Ú	 ×	×	 Û	Ü	 Û	
Ý	 Û	
Þ	 Û	Û	 ß	à	 ß	
á	 ß	ß	 â	ã	 â	â	 ä	å	 ä	
æ	 ä	ä	 ç	
è	 ç	ç	 é	ê	 é	
ë	 é	
ì	 é	é	 í	î	 í	
ï	 í	
ð	 í	í	 ñ	ò	 ñ	
ó	 ñ	ñ	 ô	õ	 ô	ô	 ö	
÷	 ö	ö	 ø	ù	 ø	
ú	 ø	ø	 û	ü	 û	û	 ý	
þ	 ý	ý	 ÿ	€
 ÿ	

 ÿ	ÿ	 ‚
ƒ
 ‚

„
 ‚
‚
 …

†
 …
…
 ‡
ˆ
 ‡

‰
 ‡

Š
 ‡
‡
 ‹
Œ
 ‹


 ‹
‹
 Ž

 Ž
Ž
 
‘
 

 ’
“
 ’
’
 ”
•
 ”

–
 ”
”
 —
˜
 —
—
 ™
š
 ™

›
 ™
™
 œ

 œ
œ
 ž
Ÿ
 ž

 
 ž
ž
 ¡
¢
 ¡

£
 ¡
¡
 ¤
¥
 ¤

¦
 ¤
¤
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
«
 ­
®
 ­

¯
 ­
­
 °
±
 °
°
 ²
³
 ²

´
 ²
²
 µ
¶
 µ
µ
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

¿
 ½
½
 À
Á
 À
À
 Â
Ã
 Â
Â
 Ä
Å
 Ä
Ä
 Æ
Ç
 Æ

È
 Æ
Æ
 É
Ê
 É
É
 Ë
Ì
 Ë

Í
 Ë
Ë
 Î
Ï
 Î
Î
 Ð
Ñ
 Ð

Ò
 Ð
Ð
 Ó
Ô
 Ó

Õ
 Ó
Ó
 Ö
×
 Ö

Ø
 Ö
Ö
 Ù
Ú
 Ù
Ù
 Û
Ü
 Û
Û
 Ý
Þ
 Ý
Ý
 ß
à
 ß

á
 ß
ß
 â
ã
 â
â
 ä
å
 ä

æ
 ä
ä
 ç
è
 ç
ç
 é
ê
 é

ë
 é
é
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
ó
 ò
ò
 ô
õ
 ô
ô
 ö
÷
 ö

ø
 ö
ö
 ù
ú
 ù
ù
 û
ü
 û

ý
 û
û
 þ
ÿ
 þ
þ
 € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰‰ Š‹ ŠŠ Œ ŒŒ Ž ŽŽ ‘  ’“ ’’ ”• ”” –— –– ˜˜ ™š ™œ ›› Ÿ žž  ¡    ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦¦ ¨ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ åå çè ç
é çç êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô ó
õ óó ö÷ ö
ø öö ùú ù
û ùù üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ ££ ¥¦ ¥
§ ¥
¨ ¥
© ¥¥ ª« ªª ¬­ ¬¬ ®¯ ®
° ®® ±² ±
³ ±
´ ±
µ ±± ¶· ¶¶ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½
À ½
Á ½½ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë É
Ì É
Í ÉÉ ÎÏ ÎÎ ÐÑ ÐÐ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× Õ
Ø Õ
Ù ÕÕ ÚÛ ÚÚ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ãä ã
å ã
æ ã
ç ãã èé èè êë ê
ì ê
í ê
î êê ïð ïï ñò ñ
ó ñ
ô ñ
õ ññ ö÷ öö øù ø
ú ø
û ø
ü øø ýþ ýý ÿ€ ÿ
 ÿ
‚ ÿ
ƒ ÿÿ „… „„ †‡ †
ˆ †
‰ †
Š †† ‹Œ ‹‹ Ž 
 
 
‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› š
œ šš ž 
Ÿ   ¡  
¢    £¤ £
¥ £
¦ £
§ ££ ¨© ¨¨ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿
À ¿¿ ÁÂ Á
Ã Á
Ä ÁÁ ÅÆ Å
Ç ÅÅ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø Ö
Ù Ö
Ú ÖÖ ÛÜ ÛÛ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ã
å ãã æç æ
è ææ éê é
ë éé ìí ì
î ìì ïð ï
ñ ïï ò
ó òò ôõ ô
ö ô
÷ ôô øù ø
ú øø ûü û
ý û
þ û
ÿ ûû € €€ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ Š
Œ ŠŠ Ž 
  ‘ 
’  “” “
• ““ –— –
˜ –– ™
š ™™ ›œ ›
 ›
ž ›› Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢
¥ ¢
¦ ¢¢ §¨ §§ ©ª ©© «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ Â
Ã ÂÂ ÄÅ Ä
Æ Ä
Ç ÄÄ ÈÉ È
Ê È
Ë ÈÈ ÌÍ Ì
Î ÌÌ ÏÐ ÏÏ ÑÒ Ñ
Ó ÑÑ Ô
Õ ÔÔ Ö× Ö
Ø Ö
Ù ÖÖ ÚÛ Ú
Ü Ú
Ý ÚÚ Þß Þ
à ÞÞ áâ áá ã
ä ãã åæ å
ç åå èé èè ê
ë êê ìí ì
î ìì ïð ï
ñ ïï ò
ó òò ôõ ô
ö ô
÷ ôô øù ø
ú øø ûü ûû ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘‘ “” “
• ““ –— –
˜ –– ™š ™
› ™™ œ œœ žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²² ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½½ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ÐÑ Ð
Ò ÐÐ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ ÝÞ Ý
ß ÝÝ àá àà âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ë
í ëë îï îî ðñ ðð òó òò ôõ ôô ö÷ öö øù øø úû úú üý üü þÿ þ €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ  
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±
³ ±± ´µ ´
¶ ´´ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï íí ðñ ð
ò ðð óô ó
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
™ —— š› šš œ œ
ž œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢
¤ ¢¢ ¥¦ ¥
§ ¥¥ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË Ê
Ì ÊÊ ÍÍ ÎÏ ÎÎ ÐÑ Ð
Ò Ð
Ó Ð
Ô ÐÐ ÕÖ ÕÕ ×Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ Ü
ß Ü
à ÜÜ áâ áá ãä ãã åæ å
ç åå èé è
ê è
ë è
ì èè íî íí ïð ïï ñò ñ
ó ññ ôõ ô
ö ô
÷ ô
ø ôô ùú ùù ûü ûû ýþ ý
ÿ ýý € €
‚ €
ƒ €
„ €€ …† …… ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ ŒŒ Ž 
 
 
‘  ’“ ’’ ”• ”
– ”
— ”
˜ ”” ™š ™™ ›œ ›
 ›
ž ›
Ÿ ››  ¡    ¢£ ¢
¤ ¢
¥ ¢
¦ ¢¢ §¨ §§ ©ª ©
« ©
¬ ©
­ ©© ®¯ ®® °± °
² °
³ °
´ °° µ¶ µµ ·¸ ·· ¹º ¹
» ¹
¼ ¹
½ ¹¹ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ Ï
Ò Ï
Ó ÏÏ ÔÕ ÔÔ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ ÜÝ Ü
Þ ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ë
ì ëë íî í
ï í
ð íí ñò ñ
ó ññ ôõ ôô ö÷ ö
ø öö ùú ù
û ùù üý ü
þ üü ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚
… ‚
† ‚‚ ‡ˆ ‡‡ ‰Š ‰
‹ ‰‰ Œ Œ
Ž ŒŒ  
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›
 ›› ž
Ÿ žž  ¡  
¢  
£    ¤¥ ¤
¦ ¤¤ §¨ §
© §
ª §
« §§ ¬­ ¬¬ ®¯ ®
° ®® ±² ±± ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ Å
Æ ÅÅ ÇÈ Ç
É Ç
Ê ÇÇ ËÌ Ë
Í ËË ÎÏ Î
Ð Î
Ñ Î
Ò ÎÎ ÓÔ ÓÓ ÕÖ ÕÕ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ ÝÞ Ý
ß ÝÝ àá à
â àà ãä ã
å ãã æç æ
è ææ éê éé ëì ë
í ëë î
ï îî ðñ ð
ò ð
ó ðð ôõ ô
ö ô
÷ ôô øù ø
ú øø ûü ûû ýþ ý
ÿ ýý €
 €€ ‚ƒ ‚
„ ‚
… ‚‚ †‡ †
ˆ †
‰ †† Š‹ Š
Œ ŠŠ Ž  
  ‘’ ‘
“ ‘‘ ”• ”” –
— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› ž
Ÿ žž  ¡  
¢  
£    ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©© «¬ «
­ «« ®¯ ®® °± °
² °° ³´ ³³ µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾¾ ÀÁ ÀÀ ÂÃ Â
Ä ÂÂ ÅÆ ÅÅ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×× ÙÚ Ù
Û ÙÙ ÜÝ ÜÜ Þß Þ
à ÞÞ áâ áá ãä ã
å ãã æç æ
è ææ éê é
ë éé ìí ìì îï î
ð îî ñò ññ óô ó
õ óó ö÷ öö øù ø
ú øø ûü û
ý ûû þÿ þ
€ þþ ‚  ƒ„ ƒƒ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹‹ Ž 
  ‘ 
’  “” “
• ““ –— –– ˜™ ˜
š ˜˜ ›œ ›
 ›› žŸ ž
  žž ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §§ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ ÏÏ ÒÓ Ò
Ô ÒÒ ÕÖ Õ
× ÕÕ ØÙ Ø
Ú Ø
Û Ø
Ü ØØ ÝÞ ÝÝ ßà ß
á ß
â ß
ã ßß äå ää æç æ
è æ
é æ
ê ææ ëì ëë íî í
ï í
ð í
ñ íí òó òò ôõ ô
ö ô
÷ ô
ø ôô ùú ùù ûü û
ý û
þ û
ÿ ûû € €€ ‚ƒ ‚
„ ‚
… ‚
† ‚‚ ‡ˆ ‡‡ ‰Š ‰‰ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –– ™š ™
› ™™ œ œ
ž œ
Ÿ œ
  œœ ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸
¹ ¸¸ º» º
¼ º
½ ºº ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ Ã
Å ÃÃ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ Ì
Î ÌÌ ÏÐ Ï
Ñ Ï
Ò Ï
Ó ÏÏ ÔÕ ÔÔ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ Û
Ý ÛÛ Þß Þ
à ÞÞ áâ á
ã áá äå ä
æ ää çè ç
é çç êë ê
ì êê í
î íí ïð ï
ñ ï
ò ïï óô ó
õ óó ö÷ ö
ø ö
ù ö
ú öö ûü ûû ýþ ýý ÿ€ ÿ
 ÿÿ ‚ƒ ‚
„ ‚‚ …† …
‡ …… ˆ‰ ˆ
Š ˆˆ ‹Œ ‹
 ‹‹ Ž Ž
 ŽŽ ‘’ ‘
“ ‘‘ ”
• ”” –— –
˜ –
™ –– š› š
œ šš ž 
Ÿ 
  
¡  ¢£ ¢¢ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶¶ ¸¹ ¸
º ¸¸ »
¼ »» ½¾ ½
¿ ½
À ½½ ÁÂ Á
Ã Á
Ä ÁÁ ÅÆ Å
Ç ÅÅ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ Í
Î ÍÍ ÏÐ Ï
Ñ Ï
Ò ÏÏ ÓÔ Ó
Õ Ó
Ö ÓÓ ×Ø ×
Ù ×× ÚÛ ÚÚ Ü
Ý ÜÜ Þß Þ
à ÞÞ áâ áá ã
ä ãã åæ å
ç åå èé è
ê èè ë
ì ëë íî í
ï í
ð íí ñò ñ
ó ññ ôõ ôô ö÷ öö øù ø
ú øø ûü ûû ýþ ý
ÿ ýý € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ Ž   
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜˜ š› šš œ œ
ž œœ Ÿ  ŸŸ ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ªª ¬­ ¬¬ ®¯ ®
° ®® ±² ±± ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾¾ ÀÁ À
Â ÀÀ ÃÄ ÃÃ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË Î
Ð ÏÏ Ñ
Ò ÑÑ Ó
Ô ÓÓ Õ
Ö ÕÕ ×
Ø ×× ÙÚ 0Û 3Ü /Ý -Þ 1	ß "à ‰à ˜à Íà Œá .	â &ã 2ä 4  	 
          ! #  %$ '" )& *( , 65 8  :9 <- >7 ?; @= BA D F HC JG K- M7 N; OL QP S UT WR YV Z- \7 ]; ^[ `_ b dc fa he i- k7 l; mj on q sr up wt x- z7 {; |y ~} € ‚ „ †ƒ ‡- ‰7 Š; ‹ˆ Œ  ‘ “Ž •’ –- ˜7 ™; š— œ› ž  Ÿ ¢ ¤¡ ¥- §7 ¨; ©¦ «ª ­ ¯® ±¬ ³° ´- ¶7 ·; ¸µ º¹ ¼ ¾½ À» Â¿ Ã- Å7 Æ; ÇÄ ÉÈ Ë ÍÌ Ï- Ñ7 Ò; ÓÐ ÕÔ × Ù- Û7 Ü; ÝÚ ßÞ á ãâ å- ç7 è; éæ ëê í ïî ñ- ó7 ô; õò ÷ö ù ûú ý- ÿ7 €; þ ƒ‚ … ‡† ‰. ‹7 Œ; Š . ‘7 ’; “ •/ —7 ˜; ™– ›/ 7 ž; Ÿœ ¡0 £7 ¤; ¥¢ §0 ©7 ª; «¨ ­1 ¯7 °; ±® ³1 µ7 ¶; ·´ ¹2 »7 ¼; ½º ¿2 Á7 Â; ÃÀ Å3 Ç7 È; ÉÆ Ë3 Í7 Î; ÏÌ Ñ Ó ÕÔ × ÙÖ ÛØ Ü ÞÝ àß â äã æá èå é ëê íì ï ñð óî õò ö ø÷ úù ü þý €û ‚ÿ ƒ …„ ‡† ‰ ‹Š ˆ Œ C ’Ô “R •ß –a ˜ì ™p ›ù œ ž† ŸŽ ¡G ¢ ¤V ¥¬ §e ¨» ªt «Ê ­ƒ ®Ö °’ ±à ³¡ ´ì ¶° ·ø ¹¿ º„ ¼Î ½- ¿7 À; Á¾ ÃÂ ÅÄ ÇØ È- Ê7 Ë; ÌÉ ÎÍ ÐÏ Òä Ó- Õ7 Ö; ×Ô ÙØ ÛÚ Ýð Þ- à7 á; âß äã æå èü é- ë7 ì; íê ïî ñð óˆ ô. ö7 ÷; øõ ú/ ü7 ý; þû €0 ‚7 ƒ; „ †1 ˆ7 ‰; Š‡ Œ2 Ž7 ;  ’3 ”7 •; –“ ˜4 š7 ›; œ™ žÖ  Ž ¢¡ ¤Ÿ ¥C §£ ©¦ ª¨ ¬ ­à ¯R ±® ³° ´² ¶« ·4 ¹7 º; »¸ ½ ¿¾ Á® ÂÀ Ä° ÅÃ Ç¼ È” Êù ËŽ ÍÉ ÎÌ ÐÆ ÑŽ Ó° ÔÒ Ö® Øù ÙÕ Ú„ ÜÛ Þ— ß áÝ ãà äÊ æâ çå é× êè ìÏ í4 ï7 ð; ñî óì õ¬ ÷ö ùô úa üø þû ÿý ò ‚  „ÿ …š ‡ƒ ˆ† Š€ ‹Ž û ŽŒ ô ’ù “ ”‘ –‰ —4 ™7 š; ›˜ ø Ÿ» ¡  £ž ¤÷ ¦¢ ¨¥ ©§ «œ ¬¬ ®… ¯¦ ±­ ²° ´ª µŽ ·¥ ¸¶ ºž ¼ù ½¹ ¾» À³ Á4 Ã7 Ä; ÅÂ Ç ÉÈ ËÛ ÌÊ Îà ÏÍ ÑÆ Ò¸ Ô‹ Õ² ×Ó ØÖ ÚÐ Û” Ý” ßÜ àÞ âù äù åá æŽ èŽ éã êç ìÙ íÈ ïÄ ñî òð ôÛ ö‘ ÷ó øà ú¾ ûõ üù þë ÿ— € ƒÛ …‚ †Ê ˆ‡ Šà Œ‰ Ž ‹ Ž ’„ ”ù •‘ –“ ˜ý ™E › œ Ÿž ¡š £  ¤ ¦¥ ¨§ ª¢ «© ­µ ®¬ °™ ±T ³Ÿ µ´ ·¶ ¹² »¸ ¼â ¾½ Àº Á¿ Ãë ÄÂ Æ¸ Çc É® ËÊ ÍÌ ÏÈ ÑÎ Òî ÔÓ ÖÐ ×Õ Ù• ÚØ Üî Ýr ß½ áà ãâ åÞ çä èú êé ìæ íë ï¿ ðî ò˜ óÌ õô ÷ö ùÈ ûø ü† þý €ú ÿ ƒ— „‚ †Â ‡C ‰Ø ŠR Œå a ò p ’ÿ “ •Œ –Ž ˜Ô ™ ›ß œ¬ žì Ÿ» ¡ù ¢Ê ¤† ¥Ö §G ¨à ªV «ì ­e ®ø °t ±„ ³ƒ ´Ä ¶’ ·Ï ¹¡ ºÚ ¼° ½å ¿¿ Àð ÂÎ Ã- Å7 Æ; ÇÄ ÉÈ ËÊ ÍØ Î- Ð7 Ñ; ÒÏ ÔÓ ÖÕ Øä Ù- Û7 Ü; ÝÚ ßÞ áà ãð ä- æ7 ç; èå êé ìë îü ï- ñ7 ò; óð õô ÷ö ùˆ ú. ü7 ý; þû €/ ‚7 ƒ; „ †0 ˆ7 ‰; Š‡ Œ1 Ž7 ;  ’2 ”7 •; –“ ˜3 š7 ›; œ™ ž4  7 ¡; ¢Ÿ ¤Ä ¦Ÿ ¨¥ ©§ «¡ ¬ª ®£ ¯Ï ±° ³¾ ´² ¶­ ·4 ¹7 º; »¸ ½® ¿° À¾ Â¾ ÃÁ Å¼ Æù Èÿ É” ËÇ ÌÊ ÎÄ Ï” Ñ¾ ÒÐ Ô° Öÿ ×Ó Øð ÚÙ Ü ÝÊ ßÛ áÞ âÐ äà åã çÕ èæ êÍ ë4 í7 î; ïì ñÚ óô õò öô øö ù÷ ûð üÿ þ… ÿ  	ý ‚	€	 „	ú …	” ‡	ö ˆ	†	 Š	ò Œ	ÿ 	‰	 Ž	‹	 	ƒ	 ‘	4 “	7 ”	; •	’	 —	å ™	ž ›	˜	 œ	š	 ž	  Ÿ		 ¡	–	 ¢	… ¤	‹ ¥	¬ §	£	 ¨	¦	 ª	 	 «	” ­	  ®	¬	 °	˜	 ²	ÿ ³	¯	 ´	±	 ¶	©	 ·	4 ¹	7 º	; »	¸	 ½	Û ¿	Ù À	¾	 Â	Þ Ã	Á	 Å	¼	 Æ	‹ È	‘ É	¸ Ë	Ç	 Ì	Ê	 Î	Ä	 Ï	ù Ñ	ù Ó	Ð	 Ô	Ò	 Ö	ÿ Ø	ÿ Ù	Õ	 Ú	” Ü	” Ý	×	 Þ	Û	 à	Í	 á	Û ã	‘ å	â	 æ	ä	 è	Ù ê	— ë	ç	 ì	Þ î	Ä ï	é	 ð	í	 ò	ß	 ó	 õ	ô	 ÷	Ù ù	ö	 ú	Ð ü	û	 þ	Þ €
ý	 
” ƒ
ÿ	 „
‚
 †
ø	 ˆ
ÿ ‰
…
 Š
‡
 Œ
ñ	 
Ò 
E ‘

 “
Ž
 •
’
 –
 ˜
—
 š
”
 ›
¥ 
œ
 Ÿ
™
  
ž
 ¢
µ £
¡
 ¥
Ÿ ¦
Ý ¨
T ª
©
 ¬
§
 ®
«
 ¯
Ÿ ±
°
 ³
­
 ´
â ¶
µ
 ¸
²
 ¹
·
 »
é ¼
º
 ¾
¸ ¿
ê Á
c Ã
Â
 Å
À
 Ç
Ä
 È
® Ê
É
 Ì
Æ
 Í
î Ï
Î
 Ñ
Ë
 Ò
Ð
 Ô
	 Õ
Ó
 ×
ì Ø
÷ Ú
r Ü
Û
 Þ
Ù
 à
Ý
 á
½ ã
â
 å
ß
 æ
ú è
ç
 ê
ä
 ë
é
 í
µ	 î
ì
 ð
’	 ñ
„ ó
Û õ
ò
 ÷
ô
 ø
Ì ú
ù
 ü
ö
 ý
† ÿ
þ
 û
 ‚€ „‹
 …ƒ ‡¸	 ˆŽ
 ‹§
 À
 Ù
 ‘ò
 “Û
 •ù
 —‰ š œ Ÿ ¡ £ ¥˜ §à ªþ
 «È ­ç
 ®² °Î
 ±œ ³µ
 ´† ¶œ
 ·ü ¹– º¬ ¼â
 ½¯ ¿É
 À² Â°
 Ãµ Å—
 Æú È„ Éø Ë” Ì¾ ÎÂ
 ÏÁ Ñ©
 ÒÄ Ô
 Õö ×’ Øô Ú Ûò ÝŽ Þð àŒ áî ãŠ äá æê èù éè ëÿ ìð îÿ ïï ñ… ò‹ ô õó ÷— ø„ ú— ûù ý‘ þý €‘ ÿ ƒ‹ „ö †‹ ‡… ‰… Šâ ŒØ ß å Ü ’ò “Ù •ÿ –Ö ˜Œ ™Ê ›ù œÇ ž† Ÿ¸ ¡ƒ ¢å ¤- ¦7 §; ¨£ ©¥ «ª ­¬ ¯Ø °- ²7 ³; ´£ µ± ·¶ ¹¸ »ä ¼- ¾7 ¿; À£ Á½ ÃÂ ÅÄ Çð È- Ê7 Ë; Ì£ ÍÉ ÏÎ ÑÐ Óü Ô- Ö7 ×; Ø£ ÙÕ ÛÚ ÝÜ ßˆ àå â. ä7 å; æá çã é/ ë7 ì; íá îê ð0 ò7 ó; ôá õñ ÷1 ù7 ú; ûá üø þ2 €7 ; ‚á ƒÿ …3 ‡7 ˆ; ‰á Š† Œ4 Ž7 ; å ‘ “Ä •µ –” ˜Ó ™— ›’ œ² žÐ Ÿ ¡š ¢4 ¤7 ¥; ¦å §£ ©Á «² ¬Ð ®ª ¯­ ±¨ ²ê ´è µç ·³ ¸¶ º° »ç ½Ð ¾¼ À² Âè Ã¿ Ä© Æ‹ ÇÇ ÉÅ ËÈ Ìö ÎÊ ÏÍ ÑÁ ÒÐ Ô¹ Õ4 ×7 Ø; Ùå ÚÖ Ü¾ Þ¯ ßÝ áÍ âà äÛ åð çï èí êæ ëé íã îç ðÍ ñï ó¯ õè öò ÷ô ùì ú4 ü7 ý; þå ÿû » ƒ¬ „÷ †‚ ˆ… ‰‡ ‹€ Œ… Žö ˆ ‘ ’ ”Š •ç —… ˜– š¬ œè ™ ž›  “ ¡4 £7 ¤; ¥å ¦¢ ¨ ª© ¬© ­« ¯È °® ²§ ³ÿ µý ¶‚ ¸´ ¹· »± ¼ê ¾ê À½ Á¿ Ãè Åè ÆÂ Çç Éç ÊÄ ËÈ Íº Î© Ðù ÒÏ ÓÑ Õ© ×„ ØÔ ÙÈ Ûü ÜÖ ÝÚ ßÌ à‹ âá ä© æã çö éè ëÈ íê îç ðì ñï óå õè öò ÷ô ùÞ ú¤ üÓ þû ÿÄ ý ‚µ „€ …¥ ‡ƒ ‰† Šˆ Œ  ‹  ã ’Ð ”‘ •Á —“ ˜² š– ›â ™ Ÿœ  ž ¢Ó £¡ ¥£ ¦ð ¨Í ª§ «¾ ­© ®¯ °¬ ±î ³¯ µ² ¶´ ¸ø ¹· »Ö ¼ý ¾… À½ Á» Ã¿ Ä¬ ÆÂ Çú ÉÅ ËÈ ÌÊ ÎŸ ÏÍ Ñû ÒŠ Ô„ ÖÕ ØÓ Ù© Û× Ü© ÞÚ ß† áÝ ãà äâ æø çå é¢ êá ì¦ íÓ ïÐ ñÍ ó… õÕ ÷» ù© û© ýë ÿÓ ž ‚Ð „Ý …Í ‡ê ˆÄ Š  ‹Á T Ž¾ c ‘» “r ”µ –¢ —² ™Ÿ š¯ œ® ¬ Ÿ½  © ¢Ì £› ¦¤ §þ
 ©à ªç
 ¬È ­Î
 ¯² °µ
 ²œ ³œ
 µ† ¶ù
 ¸© ¹– »ü ¼â
 ¾¬ ¿É
 Á¯ Â°
 Ä² Å—
 Çµ È„ Êú Ë” Íø ÎÂ
 Ð¾ Ñ©
 ÓÁ Ô
 ÖÄ ×’ Ùö Ú Üô ÝŽ ßò àŒ âð ãŠ åî æ… è… é‹ ëö ì‹ îÿ ï‘ ñý ò‘ ôù õ— ÷„ ø— úó û ý‹ þ… €ï ÿ ƒð „ÿ †è ‡ù ‰ê Šä ŒØ á å Þ ’ò “Û •ÿ –Ø ˜Œ ™ ›Õ š žÒ  Ý ¡Ï £ê ¤Ì ¦ù §É ©† ª ¬Æ ®« ¯Ã ±T ²À ´c µ½ ·r ¸· º » ½´ ¿¼ À± ÂŸ Ã® Å® Æ« È½ É¨ ËÌ ÌÍ Ï- Ñ7 Ò; ÓÎ ÔÐ ÖÕ Ø× ÚØ Û- Ý7 Þ; ßÎ àÜ âá äã æä ç- é7 ê; ëÎ ìè îí ðï òð ó- õ7 ö; ÷Î øô úù üû þü ÿ- 7 ‚; ƒÎ „€ †… ˆ‡ Šˆ ‹. Ž7 ; Œ ‘ “/ •7 –; —Œ ˜” š0 œ7 ; žŒ Ÿ› ¡1 £7 ¤; ¥Œ ¦¢ ¨2 ª7 «; ¬Œ ­© ¯3 ±7 ²; ³Œ ´° ¶˜ ¸4 º7 »; ¼· ½¹ ¿Æ Á´ ÂÀ ÄÕ ÅÃ Ç¾ È± ÊÒ ËÉ ÍÆ Î4 Ð7 Ñ; Ò· ÓÏ ÕÃ ×± ØÒ ÚÖ ÛÙ ÝÔ Þ… à’ áˆ ãß äâ æÜ çˆ éÒ êè ì± î’ ïë ð¨ òµ óÉ õñ ÷ô øù úö ûù ýí þü €å 4 ƒ7 „; …· †‚ ˆÀ Š® ‹‰ Ï ŽŒ ‡ ‘ÿ “™ ”‚ –’ —• ™ šˆ œÏ › Ÿ® ¡’ ¢ž £  ¥˜ ¦4 ¨7 ©; ª· «§ ­½ ¯« °÷ ²® ´± µ³ ·¬ ¸ê º  »ç ½¹ ¾¼ À¶ Áˆ Ã± ÄÂ Æ« È’ ÉÅ ÊÇ Ì¿ Í4 Ï7 Ð; Ñ· ÒÎ Ô ÖÕ Ø¨ Ù× Ûô ÜÚ ÞÓ ßð á§ âí äà åã çÝ è… ê… ìé íë ï’ ñ’ òî óˆ õˆ öð ÷ô ùæ úÕ üö þû ÿý ¨ ƒ® „€ …ô ‡ó ˆ‚ ‰† ‹ø Œµ Ž ¨ ’ “ù •” —ô ™– šˆ œ˜ › Ÿ‘ ¡’ ¢ž £  ¥Š ¦¥ ¨Ò ª© ¬§ ­E ¯® ±« ² ´³ ¶° ·µ ¹Ì º¸ ¼¹ ½ã ¿Ý ÁÀ Ã¾ ÄT ÆÅ ÈÂ ÉŸ ËÊ ÍÇ ÎÌ Ðÿ ÑÏ ÓÏ Ôð Öê Ø× ÚÕ Ûc ÝÜ ßÙ à® âá äÞ åã ç¤ èæ ê‚ ëý í± ïì ðr òñ ôî õ½ ÷ö ùó úø üË ýû ÿ§ €Š ‚„ „ƒ † ‡Õ ‰… ŠÌ Œ‹ Žˆ  ‘¤ ’ ”Î • —Õ ™– šÒ œã Ï Ÿð  Ì ¢ÿ £É ¥Œ ¦ ¨Æ ª§ «Ã ­Ý ®À °ê ±½ ³÷ ´º ¶† · ¹´ »¸ ¼± ¾T ¿® Ác Â« Är Å¨ Ç È× Ê’ Ëã Í¡ Îï Ð° Ñû Ó¿ Ô‡ ÖÎ ×. Ù7 Ú; ÛÎ ÜØ Þ/ à7 á; âÎ ãß å0 ç7 è; éÎ êæ ì1 î7 ï; ðÎ ñí ó2 õ7 ö; ÷Î øô ú3 ü7 ý; þÎ ÿû 4 ƒ7 „; …Œ †‚ ˆ× Š´ Œ‰ ‹ Æ Ž ’‡ “ã •” —Ã ˜– š‘ ›4 7 ž; ŸŒ  œ ¢± ¤” ¥Ã §£ ¨¦ ª¡ «’ ­Ý ®… °¬ ±¯ ³© ´… ¶Ã ·µ ¹” »Ý ¼¸ ½‡ ¿¾ Á€ ÂÀ Ä· Åü ÇÃ ÈÆ Êº ËÉ Í² Î4 Ð7 Ñ; ÒŒ ÓÏ Õï ×® ÙÖ ÚØ ÜÀ ÝÛ ßÔ à™ âä ãÿ åá æä èÞ é… ëÀ ìê îÖ ðÝ ñí òï ôç õ4 ÷7 ø; ùŒ úö üû þ« €ý ÿ ƒ½ „‚ †û ‡  ‰ë Šê Œˆ ‹ … … ’½ “‘ •ý —Ý ˜” ™– ›Ž œ4 ž7 Ÿ;  Œ ¡ £¨ ¥¾ ¦· ¨¤ ©§ «¢ ¬§ ®ò ¯ð ±­ ²° ´ª µ’ ·’ ¹¶ º¸ ¼Ý ¾Ý ¿» À… Â… Ã½ ÄÁ Æ³ Ç¨ É® ËÈ ÌÊ Î¾ Ðù ÑÍ Ò· Ôö ÕÏ ÖÓ ØÅ Ù€ ÛÚ Ý¾ ßÜ àü âá ä· æã ç… éå êè ìÞ îÝ ïë ðí ò× ó¥ õÒ ÷ö ùô úE üû þø ÿý ™ ‚€ „‚ …ã ‡Ý ‰ˆ ‹† ŒT Ž Š ‘ “Ì ”’ –œ —ð ™ê ›š ˜ žc  Ÿ ¢œ £¡ ¥ó ¦¤ ¨Ï ©ý «÷ ­¬ ¯ª °r ²± ´® µ³ ·š ¸¶ ºö »Š ½„ ¿¾ Á¼ Â ÄÃ ÆÀ ÇÅ Éñ ÊÈ Ì Í Ð Ò Ô
 Ö Ø+ Ï+ -™ ›™ ž ¥¨ ©Î Ïþ €þ ©¤ ¥ Ù èè çç åå ææÏ çç Ïñ çç ñ› çç ›‹ çç ‹· çç ·— çç —’ çç ’Ì çç Ì¤ çç ¤× çç ×Ê çç Ê£	 çç £	… çç …• çç •ú çç ú‚ çç ‚£ çç £Á çç Á€ çç €ƒ	 çç ƒ	ø çç øå çç åÿ çç ÿ˜ çç ˜ ææ Å çç Å° çç °ô çç ô­ çç ­ÿ	 çç ÿ	Þ çç Þ± çç ±”
 çç ”
ª çç ªÅ çç Åì çç ìƒ çç ƒ† çç †Ð çç ÐÍ çç ÍÏ èè ÏÉ çç É¾ çç ¾  çç  ƒ çç ƒå çç å¢ çç ¢‰ çç ‰¿ çç ¿™
 çç ™
  çç  Ç çç Ç© çç ©‘ çç ‘¶ çç ¶æ çç æî çç îû
 çç û
È çç Èø çç øæ çç æº çç ºË çç ËÐ çç Ð×	 çç ×	ø çç øä
 çç ä
ï çç ïÓ èè Ó× çç ×‚ çç ‚Þ çç Þ„ çç „Ó
 çç Ó
Ý çç Ý­ çç ­ çç Ó çç Ó åå ¤ çç ¤¤ çç ¤× çç ×Ç çç Ç™ çç ™ø	 çç ø	¾	 çç ¾	‹
 çç ‹
¬ çç ¬Õ çç Õß çç ßÇ çç Çð çç ðé	 çç é	– çç –Ü çç Üó çç óª çç ªÆ çç Æ‹ çç ‹ åå Á çç ÁÂ çç Âº çç º® çç ®Þ çç ÞŠ çç Šö
 çç ö
³ çç ³æ çç æˆ çç ˆÛ	 çç Û	˜ çç ˜ý çç ýÝ çç Ý€ çç €« çç «š	 çç š	Ø çç Ø­ çç ­Ä	 çç Ä	 çç  çç Ö çç Ö²
 çç ²
» çç »¸ çç ¸‘ çç ‘ø çç ø¶ çç ¶Â çç Âø çç ø€ çç €” çç ”ª çç ªÄ çç Äè çç èÝ çç Ýœ çç œó çç ó¡ çç ¡ã çç ãÅ çç Å‹ çç ‹‡
 çç ‡
š çç šÓ çç Óµ çç µô çç ôß	 çç ß	ý çç ýî çç îÇ	 çç Ç	ù çç ù³ çç ³Ì çç Ìô çç ôÖ çç Ö‰ çç ‰©	 çç ©	Â çç Â“ çç “û çç û© çç ©ÿ çç ÿ… çç …™ çç ™ƒ çç ƒ¤ çç ¤à çç àß
 çç ß
ý çç ýí	 çç í	È çç Èô çç ôÓ çç Ó× èè ×À çç ÀÐ çç ÐÑ èè ÑÀ çç Àì çç ìŸ çç Ÿ§ çç §Æ
 çç Æ
 	 çç  	– çç –À çç ÀÏ çç Ïæ çç æ¿ çç ¿Ù çç Ùˆ çç ˆÕ èè Õå çç åñ	 çç ñ	µ çç µ åå ‹	 çç ‹	¿ çç ¿á çç áý çç ý¯ çç ¯ç çç çã çç ã­
 çç ­
µ	 çç µ	Í çç Í±	 çç ±	´ çç ´Ä çç ÄÚ çç Ú× çç ×’ çç ’Š çç Ší çç í åå Ï çç Ï“ çç “ì
 çç ì
º çç ºº
 çç º
Í	 çç Í	 çç 	 çç 	‘ çç ‘õ çç õæ çç æÚ çç ÚØ çç Øã çç ã« çç «¬ çç ¬É çç Éé çç éÌ çç Ì åå ® çç ®š çç š³ çç ³ú çç úí çç íø çç øË
 çç Ë
¡
 çç ¡
¡ çç ¡½ çç ½å çç åÆ çç ÆÙ çç ÙÞ çç Þ‚ çç ‚Ì çç Ìü çç ü¬ çç ¬ ææ Ž çç Žë çç ë£ çç £“ çç “ çç ç çç ç¹ çç ¹ë çç ë³ çç ³Š çç Š² çç ²¢ çç ¢« çç «° çç °µ çç µ¹ çç ¹  çç  
é £
é À
é É
é ø
é ƒ
é ¢
é ­
é Ê
é Ó
é §
é ¾
é Ç
é ô
é ý
é š	
é £	
é ¾	
é Ç	
é ”
é ª
é ³
é Ý
é æ
é ‚
é 
é «
é ´
é À
é Ö
é ß
é ‰
é ’
é ®
é ¹
é ×
é à
é ‹
é £
é ¬
é Ø
é á
é ÿ
é ˆ
é ¤
é ­
ê Ï
ê Í
ê ¹
ê å
ê ²	ë y
ë 
ë Ä
ë Ì
ë þ
ë †
ë „
ë Š
ë ê
ë Â
ë Ä
ë Ï
ë Ú
ë å
ë ð
ë ð
ë ¸	
ë Õ
ë ¢
ë €
ë Î
ë 
ì ¢
ì º
ì Ð
ì æ
ì ú
ì ý
ì 
ì ¡
ì ³
ì Å
í ž
í ¶
í Ì
í â
í öî î î î î î Ïî Ñî Óî Õî ×	ï 5	ï 7	ï 9	ï ;
ð ý
ð ñ	
ð Þ
ð Š
ð ×
ñ „
ñ ‹
ñ ø	
ñ ÿ	
ñ å
ñ ì
ñ ‘
ñ ˜
ñ Þ
ñ å
ò Ù
ò Í	
ò º
ò æ
ò ³
ó ë
ó ß	
ó Ì
ó ø
ó Å
ô ¬
ô Â
ô Ø
ô î
ô ‚
ô ¡

ô º

ô Ó

ô ì

ô ƒ
ô ‹
ô ¡
ô ·
ô Í
ô å
ô ¸
ô Ï
ô æ
ô û
ô 
ô €
ô ’
ô ¤
ô ¶
ô È
õ è
õ €
õ ‡
õ æ
õ ô	
õ û	
õ Ð
õ á
õ è
õ ü
õ 
õ ”
õ É
õ Ú
õ á	ö [	ö c
ö ¦
ö ®
ö Ð
ö Ú
ö æ
ö æ
ö î
ö ò
ö þ
ö ê
ö ð
ö Ô
ö õ
ö û
ö 
ö ‡
ö 
ö “
ö î
ö Ú
ö Ÿ
ö ¸
ö ì
ö ì
ö ’	
ö ¸	
ö £
ö ½
ö Ö
ö è
ö ‚
ö Ï
÷ «
÷ Æ
÷ €
÷ ª
÷ Ð
÷ ­
÷ Ä
÷ ú
÷  	
÷ Ä	
÷ š
÷ °
÷ ã
÷ Š
÷ ±
÷ Æ
÷ Ü
÷ 
÷ ¶
÷ Ý
÷ ‘
÷ ©
÷ Þ
÷ …
÷ ª
ø ‰
ø ³
ø ƒ	
ø ©	
ø ì
ø “
ø ˜
ø ¿
ø ç
ø Ž
ù ’

ù «

ù Ä

ù Ý

ù ô

ù €
ù –
ù ¬
ù Â
ù Ú
ù °
ù Ç
ù Þ
ù ó
ù ˆ
ú ”

ú ™

ú ­

ú ²

ú Æ

ú Ë

ú ß

ú ä

ú ö

ú û

ú ý
ú ƒ
ú “
ú ™
ú ©
ú ¯
ú ¿
ú Å
ú ×
ú Ý
ú «
ú µ
ú Â
ú Ì
ú Ù
ú ã
ú î
ú ø
ú …
ú 
ú ø
ú Š
ú œ
ú ®
ú À	û j	û r
û µ
û ½
û ò
û ú
û ÷
û ý
û ¾
û É
û Ô
û ß
û ß
û ê
û ˜
û å
û û
û 
û ‡
û 
û “
û ™
û ’	
û å
û É
û û
û ô
û §
û öü Õü ü ¹ü áü óü ‚ü ‰ü ‘ü  ü ¸ü Îü äü øü Óü ‰	ü ¯	ü Õ	ü ç	ü ö	ü ý	ü …
ü ¿ü òü ™ü Âü Ôü ãü êü òü ëü žü Åü îü €ü ü –ü žü ¸ü íü ”ü »ü Íü Üü ãü ë
ý Ü
ý î
ý Ð	
ý â	
ý ½
ý Ï
ý é
ý û
ý ¶
ý È
þ µ
þ ë
þ •
þ ¿
þ —
þ µ
þ é
þ 	
þ µ	
þ ‹

þ  
þ Ó
þ ø
þ Ÿ
þ ø
þ Ì
þ ÿ
þ ¤
þ Ë
þ ¤
þ ™
þ Ì
þ ó
þ š
þ ñ
ÿ ‰
€ ˜ ‚ ‚ ‚ ‚ ‚ ‚ 
‚ Í	ƒ E	ƒ E	ƒ L	ƒ T	ƒ [	ƒ c	ƒ j	ƒ r	ƒ y
ƒ 
ƒ 
ƒ 
ƒ Ÿ
ƒ ®
ƒ ½
ƒ Ì
ƒ â
ƒ î
ƒ ú
ƒ †
ƒ Š
ƒ –
ƒ ¢
ƒ ®
ƒ º
ƒ Æ
ƒ Ò
ƒ Ò
ƒ Ý
ƒ ã
ƒ ê
ƒ ð
ƒ ÷
ƒ ý
ƒ „
ƒ Š
ƒ ™
ƒ ¥
ƒ ¥
ƒ Ÿ
ƒ ›
ƒ ›
ƒ ž
ƒ ž
ƒ  
ƒ  
ƒ ¢
ƒ ¢
ƒ ¤
ƒ ¤
ƒ 
ƒ š
ƒ š
ƒ «
ƒ «
ƒ ¼
ƒ ¼
ƒ ¹
ƒ –
ƒ –
ƒ §
ƒ §
ƒ ¸
ƒ ¸
ƒ ‚	„ 	„  	„ L	„ T
„ ˆ
„ —
„ —
„ Ÿ
„ ¦
„ µ
„ Ä
„ Ú
„ â
„ 
„ œ
„ ¨
„ ´
„ À
„ Ì
„ Ý
„ ã
„ É
„ ™
„ ¸
„ ¸
„ î
„ ˜
„ Â
„ Ï
„ ¸
„ ±
„ á
„ £
„ Ü
„ Ï
„ œ"
compute_rhs3"
llvm.lifetime.start.p0i8"
_Z13get_global_idj"
llvm.fmuladd.f64"
llvm.lifetime.end.p0i8*
npb-SP-compute_rhs3.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282

wgsize
@

devmap_label

 
transfer_bytes_log1p
ŽªA

wgsize_log1p
ŽªA

transfer_bytes	
°ð¾á