

[external]
3sextB+
)
	full_text

%10 = sext i32 %8 to i64
\getelementptrBK
I
	full_text<
:
8%11 = getelementptr inbounds double, double* %0, i64 %10
#i64B

	full_text
	
i64 %10
LcallBD
B
	full_text5
3
1%12 = tail call i64 @_Z13get_global_idj(i32 0) #3
6truncB-
+
	full_text

%13 = trunc i64 %12 to i32
#i64B

	full_text
	
i64 %12
5icmpB-
+
	full_text

%14 = icmp slt i32 %13, %7
#i32B

	full_text
	
i32 %13
9brB3
1
	full_text$
"
 br i1 %14, label %15, label %163
!i1B

	full_text


i1 %14
Pbitcast8BC
A
	full_text4
2
0%16 = bitcast double* %1 to [10 x [2 x double]]*
Jbitcast8B=
;
	full_text.
,
*%17 = bitcast i32* %2 to [10 x [2 x i32]]*
Jbitcast8B=
;
	full_text.
,
*%18 = bitcast i32* %3 to [10 x [2 x i32]]*
Jbitcast8B=
;
	full_text.
,
*%19 = bitcast i32* %4 to [10 x [2 x i32]]*
1shl8B(
&
	full_text

%20 = shl i64 %12, 32
%i648B

	full_text
	
i64 %12
9ashr8B/
-
	full_text 

%21 = ashr exact i64 %20, 32
%i648B

	full_text
	
i64 %20
�getelementptr8Bt
r
	full_texte
c
a%22 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 0, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %22, align 8, !tbaa !8
-double*8B

	full_text

double* %22
�getelementptr8Bn
l
	full_text_
]
[%23 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 0, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %23, align 4, !tbaa !12
'i32*8B

	full_text


i32* %23
�getelementptr8Bn
l
	full_text_
]
[%24 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 0, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %24, align 4, !tbaa !12
'i32*8B

	full_text


i32* %24
�getelementptr8Bn
l
	full_text_
]
[%25 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 0, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %25, align 4, !tbaa !12
'i32*8B

	full_text


i32* %25
�getelementptr8Bt
r
	full_texte
c
a%26 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 0, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %26, align 8, !tbaa !8
-double*8B

	full_text

double* %26
�getelementptr8Bn
l
	full_text_
]
[%27 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 0, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %27, align 4, !tbaa !12
'i32*8B

	full_text


i32* %27
�getelementptr8Bn
l
	full_text_
]
[%28 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 0, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %28, align 4, !tbaa !12
'i32*8B

	full_text


i32* %28
�getelementptr8Bn
l
	full_text_
]
[%29 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 0, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %29, align 4, !tbaa !12
'i32*8B

	full_text


i32* %29
�getelementptr8Bt
r
	full_texte
c
a%30 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 1, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %30, align 8, !tbaa !8
-double*8B

	full_text

double* %30
�getelementptr8Bn
l
	full_text_
]
[%31 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 1, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %31, align 4, !tbaa !12
'i32*8B

	full_text


i32* %31
�getelementptr8Bn
l
	full_text_
]
[%32 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 1, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %32, align 4, !tbaa !12
'i32*8B

	full_text


i32* %32
�getelementptr8Bn
l
	full_text_
]
[%33 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 1, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %33, align 4, !tbaa !12
'i32*8B

	full_text


i32* %33
�getelementptr8Bt
r
	full_texte
c
a%34 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 1, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %34, align 8, !tbaa !8
-double*8B

	full_text

double* %34
�getelementptr8Bn
l
	full_text_
]
[%35 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 1, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %35, align 4, !tbaa !12
'i32*8B

	full_text


i32* %35
�getelementptr8Bn
l
	full_text_
]
[%36 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 1, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %36, align 4, !tbaa !12
'i32*8B

	full_text


i32* %36
�getelementptr8Bn
l
	full_text_
]
[%37 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 1, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %37, align 4, !tbaa !12
'i32*8B

	full_text


i32* %37
�getelementptr8Bt
r
	full_texte
c
a%38 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 2, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %38, align 8, !tbaa !8
-double*8B

	full_text

double* %38
�getelementptr8Bn
l
	full_text_
]
[%39 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 2, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %39, align 4, !tbaa !12
'i32*8B

	full_text


i32* %39
�getelementptr8Bn
l
	full_text_
]
[%40 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 2, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %40, align 4, !tbaa !12
'i32*8B

	full_text


i32* %40
�getelementptr8Bn
l
	full_text_
]
[%41 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 2, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %41, align 4, !tbaa !12
'i32*8B

	full_text


i32* %41
�getelementptr8Bt
r
	full_texte
c
a%42 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 2, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %42, align 8, !tbaa !8
-double*8B

	full_text

double* %42
�getelementptr8Bn
l
	full_text_
]
[%43 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 2, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %43, align 4, !tbaa !12
'i32*8B

	full_text


i32* %43
�getelementptr8Bn
l
	full_text_
]
[%44 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 2, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %44, align 4, !tbaa !12
'i32*8B

	full_text


i32* %44
�getelementptr8Bn
l
	full_text_
]
[%45 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 2, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %45, align 4, !tbaa !12
'i32*8B

	full_text


i32* %45
�getelementptr8Bt
r
	full_texte
c
a%46 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 3, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %46, align 8, !tbaa !8
-double*8B

	full_text

double* %46
�getelementptr8Bn
l
	full_text_
]
[%47 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 3, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %47, align 4, !tbaa !12
'i32*8B

	full_text


i32* %47
�getelementptr8Bn
l
	full_text_
]
[%48 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 3, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %48, align 4, !tbaa !12
'i32*8B

	full_text


i32* %48
�getelementptr8Bn
l
	full_text_
]
[%49 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 3, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %49, align 4, !tbaa !12
'i32*8B

	full_text


i32* %49
�getelementptr8Bt
r
	full_texte
c
a%50 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 3, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %50, align 8, !tbaa !8
-double*8B

	full_text

double* %50
�getelementptr8Bn
l
	full_text_
]
[%51 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 3, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %51, align 4, !tbaa !12
'i32*8B

	full_text


i32* %51
�getelementptr8Bn
l
	full_text_
]
[%52 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 3, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %52, align 4, !tbaa !12
'i32*8B

	full_text


i32* %52
�getelementptr8Bn
l
	full_text_
]
[%53 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 3, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %53, align 4, !tbaa !12
'i32*8B

	full_text


i32* %53
�getelementptr8Bt
r
	full_texte
c
a%54 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 4, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %54, align 8, !tbaa !8
-double*8B

	full_text

double* %54
�getelementptr8Bn
l
	full_text_
]
[%55 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 4, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %55, align 4, !tbaa !12
'i32*8B

	full_text


i32* %55
�getelementptr8Bn
l
	full_text_
]
[%56 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 4, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %56, align 4, !tbaa !12
'i32*8B

	full_text


i32* %56
�getelementptr8Bn
l
	full_text_
]
[%57 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 4, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %57, align 4, !tbaa !12
'i32*8B

	full_text


i32* %57
�getelementptr8Bt
r
	full_texte
c
a%58 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 4, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %58, align 8, !tbaa !8
-double*8B

	full_text

double* %58
�getelementptr8Bn
l
	full_text_
]
[%59 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 4, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %59, align 4, !tbaa !12
'i32*8B

	full_text


i32* %59
�getelementptr8Bn
l
	full_text_
]
[%60 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 4, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %60, align 4, !tbaa !12
'i32*8B

	full_text


i32* %60
�getelementptr8Bn
l
	full_text_
]
[%61 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 4, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %61, align 4, !tbaa !12
'i32*8B

	full_text


i32* %61
�getelementptr8Bt
r
	full_texte
c
a%62 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 5, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %62, align 8, !tbaa !8
-double*8B

	full_text

double* %62
�getelementptr8Bn
l
	full_text_
]
[%63 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 5, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %63, align 4, !tbaa !12
'i32*8B

	full_text


i32* %63
�getelementptr8Bn
l
	full_text_
]
[%64 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 5, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %64, align 4, !tbaa !12
'i32*8B

	full_text


i32* %64
�getelementptr8Bn
l
	full_text_
]
[%65 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 5, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %65, align 4, !tbaa !12
'i32*8B

	full_text


i32* %65
�getelementptr8Bt
r
	full_texte
c
a%66 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 5, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %66, align 8, !tbaa !8
-double*8B

	full_text

double* %66
�getelementptr8Bn
l
	full_text_
]
[%67 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 5, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %67, align 4, !tbaa !12
'i32*8B

	full_text


i32* %67
�getelementptr8Bn
l
	full_text_
]
[%68 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 5, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %68, align 4, !tbaa !12
'i32*8B

	full_text


i32* %68
�getelementptr8Bn
l
	full_text_
]
[%69 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 5, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %69, align 4, !tbaa !12
'i32*8B

	full_text


i32* %69
�getelementptr8Bt
r
	full_texte
c
a%70 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 6, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %70, align 8, !tbaa !8
-double*8B

	full_text

double* %70
�getelementptr8Bn
l
	full_text_
]
[%71 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 6, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %71, align 4, !tbaa !12
'i32*8B

	full_text


i32* %71
�getelementptr8Bn
l
	full_text_
]
[%72 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 6, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %72, align 4, !tbaa !12
'i32*8B

	full_text


i32* %72
�getelementptr8Bn
l
	full_text_
]
[%73 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 6, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %73, align 4, !tbaa !12
'i32*8B

	full_text


i32* %73
�getelementptr8Bt
r
	full_texte
c
a%74 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 6, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %74, align 8, !tbaa !8
-double*8B

	full_text

double* %74
�getelementptr8Bn
l
	full_text_
]
[%75 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 6, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %75, align 4, !tbaa !12
'i32*8B

	full_text


i32* %75
�getelementptr8Bn
l
	full_text_
]
[%76 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 6, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %76, align 4, !tbaa !12
'i32*8B

	full_text


i32* %76
�getelementptr8Bn
l
	full_text_
]
[%77 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 6, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %77, align 4, !tbaa !12
'i32*8B

	full_text


i32* %77
�getelementptr8Bt
r
	full_texte
c
a%78 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 7, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %78, align 8, !tbaa !8
-double*8B

	full_text

double* %78
�getelementptr8Bn
l
	full_text_
]
[%79 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 7, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %79, align 4, !tbaa !12
'i32*8B

	full_text


i32* %79
�getelementptr8Bn
l
	full_text_
]
[%80 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 7, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %80, align 4, !tbaa !12
'i32*8B

	full_text


i32* %80
�getelementptr8Bn
l
	full_text_
]
[%81 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 7, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %81, align 4, !tbaa !12
'i32*8B

	full_text


i32* %81
�getelementptr8Bt
r
	full_texte
c
a%82 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 7, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %82, align 8, !tbaa !8
-double*8B

	full_text

double* %82
�getelementptr8Bn
l
	full_text_
]
[%83 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 7, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %83, align 4, !tbaa !12
'i32*8B

	full_text


i32* %83
�getelementptr8Bn
l
	full_text_
]
[%84 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 7, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %84, align 4, !tbaa !12
'i32*8B

	full_text


i32* %84
�getelementptr8Bn
l
	full_text_
]
[%85 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 7, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %85, align 4, !tbaa !12
'i32*8B

	full_text


i32* %85
�getelementptr8Bt
r
	full_texte
c
a%86 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 8, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %86, align 8, !tbaa !8
-double*8B

	full_text

double* %86
�getelementptr8Bn
l
	full_text_
]
[%87 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 8, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %87, align 4, !tbaa !12
'i32*8B

	full_text


i32* %87
�getelementptr8Bn
l
	full_text_
]
[%88 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 8, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %88, align 4, !tbaa !12
'i32*8B

	full_text


i32* %88
�getelementptr8Bn
l
	full_text_
]
[%89 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 8, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %89, align 4, !tbaa !12
'i32*8B

	full_text


i32* %89
�getelementptr8Bt
r
	full_texte
c
a%90 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 8, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %90, align 8, !tbaa !8
-double*8B

	full_text

double* %90
�getelementptr8Bn
l
	full_text_
]
[%91 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 8, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %91, align 4, !tbaa !12
'i32*8B

	full_text


i32* %91
�getelementptr8Bn
l
	full_text_
]
[%92 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 8, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %92, align 4, !tbaa !12
'i32*8B

	full_text


i32* %92
�getelementptr8Bn
l
	full_text_
]
[%93 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 8, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %93, align 4, !tbaa !12
'i32*8B

	full_text


i32* %93
�getelementptr8Bt
r
	full_texte
c
a%94 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 9, i64 1
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 0.000000e+00, double* %94, align 8, !tbaa !8
-double*8B

	full_text

double* %94
�getelementptr8Bn
l
	full_text_
]
[%95 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 9, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %95, align 4, !tbaa !12
'i32*8B

	full_text


i32* %95
�getelementptr8Bn
l
	full_text_
]
[%96 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 9, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %96, align 4, !tbaa !12
'i32*8B

	full_text


i32* %96
�getelementptr8Bn
l
	full_text_
]
[%97 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 9, i64 1
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %97, align 4, !tbaa !12
'i32*8B

	full_text


i32* %97
�getelementptr8Bt
r
	full_texte
c
a%98 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21, i64 9, i64 0
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
Wstore8BL
J
	full_text=
;
9store double 1.000000e+00, double* %98, align 8, !tbaa !8
-double*8B

	full_text

double* %98
�getelementptr8Bn
l
	full_text_
]
[%99 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 9, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
Gstore8B<
:
	full_text-
+
)store i32 0, i32* %99, align 4, !tbaa !12
'i32*8B

	full_text


i32* %99
�getelementptr8Bo
m
	full_text`
^
\%100 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 9, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
Hstore8B=
;
	full_text.
,
*store i32 0, i32* %100, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %100
�getelementptr8Bo
m
	full_text`
^
\%101 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 9, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
Hstore8B=
;
	full_text.
,
*store i32 0, i32* %101, align 4, !tbaa !12
(i32*8B

	full_text

	i32* %101
5add8B,
*
	full_text

%102 = add nsw i32 %13, 1
%i328B

	full_text
	
i32 %13
6icmp8B,
*
	full_text

%103 = icmp sgt i32 %6, 2
=br8B5
3
	full_text&
$
"br i1 %103, label %104, label %163
$i18B

	full_text
	
i1 %103
5add8B,
*
	full_text

%105 = add nsw i32 %6, -1
5add8B,
*
	full_text

%106 = add nsw i32 %5, -1
6icmp8B,
*
	full_text

%107 = icmp sgt i32 %5, 2
8icmp8B.
,
	full_text

%108 = icmp sgt i32 %13, -1
%i328B

	full_text
	
i32 %13
5add8B,
*
	full_text

%109 = add nsw i32 %7, -1
;icmp8B1
/
	full_text"
 
%110 = icmp slt i32 %102, %109
&i328B

	full_text


i32 %102
&i328B

	full_text


i32 %109
4and8B+
)
	full_text

%111 = and i1 %108, %110
$i18B

	full_text
	
i1 %108
$i18B

	full_text
	
i1 %110
1mul8B(
&
	full_text

%112 = mul i32 %6, %5
5mul8B,
*
	full_text

%113 = mul i32 %112, %102
&i328B

	full_text


i32 %112
&i328B

	full_text


i32 %102
8sext8B.
,
	full_text

%114 = sext i32 %113 to i64
&i328B

	full_text


i32 %113
agetelementptr8BN
L
	full_text?
=
;%115 = getelementptr inbounds double, double* %11, i64 %114
-double*8B

	full_text

double* %11
&i648B

	full_text


i64 %114
zgetelementptr8Bg
e
	full_textX
V
T%116 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %16, i64 %21
G[10 x [2 x double]]*8B+
)
	full_text

[10 x [2 x double]]* %16
%i648B

	full_text
	
i64 %21
�getelementptr8Bm
k
	full_text^
\
Z%117 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %116, i64 0, i64 0
H[10 x [2 x double]]*8B,
*
	full_text

[10 x [2 x double]]* %116
{getelementptr8Bh
f
	full_textY
W
U%118 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
{getelementptr8Bh
f
	full_textY
W
U%119 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
{getelementptr8Bh
f
	full_textY
W
U%120 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21, i64 0
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
�getelementptr8Bt
r
	full_texte
c
a%121 = getelementptr inbounds [10 x [2 x double]], [10 x [2 x double]]* %116, i64 0, i64 0, i64 0
H[10 x [2 x double]]*8B,
*
	full_text

[10 x [2 x double]]* %116
tgetelementptr8Ba
_
	full_textR
P
N%122 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %17, i64 %21
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %17
%i648B

	full_text
	
i64 %21
�getelementptr8Bn
l
	full_text_
]
[%123 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %122, i64 0, i64 0, i64 0
B[10 x [2 x i32]]*8B)
'
	full_text

[10 x [2 x i32]]* %122
tgetelementptr8Ba
_
	full_textR
P
N%124 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %18, i64 %21
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %18
%i648B

	full_text
	
i64 %21
�getelementptr8Bn
l
	full_text_
]
[%125 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %124, i64 0, i64 0, i64 0
B[10 x [2 x i32]]*8B)
'
	full_text

[10 x [2 x i32]]* %124
tgetelementptr8Ba
_
	full_textR
P
N%126 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %19, i64 %21
A[10 x [2 x i32]]*8B(
&
	full_text

[10 x [2 x i32]]* %19
%i648B

	full_text
	
i64 %21
�getelementptr8Bn
l
	full_text_
]
[%127 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %126, i64 0, i64 0, i64 0
B[10 x [2 x i32]]*8B)
'
	full_text

[10 x [2 x i32]]* %126
zgetelementptr8Bg
e
	full_textX
V
T%128 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %122, i64 0, i64 0
B[10 x [2 x i32]]*8B)
'
	full_text

[10 x [2 x i32]]* %122
zgetelementptr8Bg
e
	full_textX
V
T%129 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %124, i64 0, i64 0
B[10 x [2 x i32]]*8B)
'
	full_text

[10 x [2 x i32]]* %124
zgetelementptr8Bg
e
	full_textX
V
T%130 = getelementptr inbounds [10 x [2 x i32]], [10 x [2 x i32]]* %126, i64 0, i64 0
B[10 x [2 x i32]]*8B)
'
	full_text

[10 x [2 x i32]]* %126
6sext8B,
*
	full_text

%131 = sext i32 %5 to i64
8zext8B.
,
	full_text

%132 = zext i32 %106 to i64
&i328B

	full_text


i32 %106
8zext8B.
,
	full_text

%133 = zext i32 %105 to i64
&i328B

	full_text


i32 %105
(br8B 

	full_text

br label %134
Fphi8B=
;
	full_text.
,
*%135 = phi i64 [ 1, %104 ], [ %161, %160 ]
&i648B

	full_text


i64 %161
=br8B5
3
	full_text&
$
"br i1 %107, label %136, label %160
$i18B

	full_text
	
i1 %107
9mul8B0
.
	full_text!

%137 = mul nsw i64 %135, %131
&i648B

	full_text


i64 %135
&i648B

	full_text


i64 %131
:trunc8B/
-
	full_text 

%138 = trunc i64 %135 to i32
&i648B

	full_text


i64 %135
:trunc8B/
-
	full_text 

%139 = trunc i64 %135 to i32
&i648B

	full_text


i64 %135
(br8B 

	full_text

br label %140
Fphi8B=
;
	full_text.
,
*%141 = phi i64 [ 1, %136 ], [ %158, %157 ]
&i648B

	full_text


i64 %158
=br8B5
3
	full_text&
$
"br i1 %111, label %142, label %157
$i18B

	full_text
	
i1 %111
9add8B0
.
	full_text!

%143 = add nsw i64 %141, %137
&i648B

	full_text


i64 %141
&i648B

	full_text


i64 %137
bgetelementptr8BO
M
	full_text@
>
<%144 = getelementptr inbounds double, double* %115, i64 %143
.double*8B

	full_text

double* %115
&i648B

	full_text


i64 %143
Pload8BF
D
	full_text7
5
3%145 = load double, double* %144, align 8, !tbaa !8
.double*8B

	full_text

double* %144
Oload8BE
C
	full_text6
4
2%146 = load double, double* %22, align 8, !tbaa !8
-double*8B

	full_text

double* %22
>fcmp8B4
2
	full_text%
#
!%147 = fcmp ogt double %145, %146
,double8B

	full_text

double %145
,double8B

	full_text

double %146
=br8B5
3
	full_text&
$
"br i1 %147, label %148, label %151
$i18B

	full_text
	
i1 %147
Ostore8BD
B
	full_text5
3
1store double %145, double* %22, align 8, !tbaa !8
,double8B

	full_text

double %145
-double*8B

	full_text

double* %22
:trunc8B/
-
	full_text 

%149 = trunc i64 %141 to i32
&i648B

	full_text


i64 %141
Jstore8B?
=
	full_text0
.
,store i32 %149, i32* %23, align 4, !tbaa !12
&i328B

	full_text


i32 %149
'i32*8B

	full_text


i32* %23
Jstore8B?
=
	full_text0
.
,store i32 %138, i32* %24, align 4, !tbaa !12
&i328B

	full_text


i32 %138
'i32*8B

	full_text


i32* %24
Jstore8B?
=
	full_text0
.
,store i32 %102, i32* %25, align 4, !tbaa !12
&i328B

	full_text


i32 %102
'i32*8B

	full_text


i32* %25
�call8B�
�
	full_text{
y
wtail call void @bubble([2 x double]* nonnull %117, [2 x i32]* %118, [2 x i32]* %119, [2 x i32]* %120, i32 10, i32 1) #4
:[2 x double]*8B%
#
	full_text

[2 x double]* %117
4
[2 x i32]*8B"
 
	full_text

[2 x i32]* %118
4
[2 x i32]*8B"
 
	full_text

[2 x i32]* %119
4
[2 x i32]*8B"
 
	full_text

[2 x i32]* %120
Pload8BF
D
	full_text7
5
3%150 = load double, double* %144, align 8, !tbaa !8
.double*8B

	full_text

double* %144
(br8B 

	full_text

br label %151
Lphi8BC
A
	full_text4
2
0%152 = phi double [ %150, %148 ], [ %145, %142 ]
,double8B

	full_text

double %150
,double8B

	full_text

double %145
Pload8BF
D
	full_text7
5
3%153 = load double, double* %121, align 8, !tbaa !8
.double*8B

	full_text

double* %121
>fcmp8B4
2
	full_text%
#
!%154 = fcmp olt double %152, %153
,double8B

	full_text

double %152
,double8B

	full_text

double %153
=br8B5
3
	full_text&
$
"br i1 %154, label %155, label %157
$i18B

	full_text
	
i1 %154
Pstore8	BE
C
	full_text6
4
2store double %152, double* %121, align 8, !tbaa !8
,double8	B

	full_text

double %152
.double*8	B

	full_text

double* %121
:trunc8	B/
-
	full_text 

%156 = trunc i64 %141 to i32
&i648	B

	full_text


i64 %141
Kstore8	B@
>
	full_text1
/
-store i32 %156, i32* %123, align 4, !tbaa !12
&i328	B

	full_text


i32 %156
(i32*8	B

	full_text

	i32* %123
Kstore8	B@
>
	full_text1
/
-store i32 %139, i32* %125, align 4, !tbaa !12
&i328	B

	full_text


i32 %139
(i32*8	B

	full_text

	i32* %125
Kstore8	B@
>
	full_text1
/
-store i32 %102, i32* %127, align 4, !tbaa !12
&i328	B

	full_text


i32 %102
(i32*8	B

	full_text

	i32* %127
�call8	B�
�
	full_text{
y
wtail call void @bubble([2 x double]* nonnull %117, [2 x i32]* %128, [2 x i32]* %129, [2 x i32]* %130, i32 10, i32 0) #4
:[2 x double]*8	B%
#
	full_text

[2 x double]* %117
4
[2 x i32]*8	B"
 
	full_text

[2 x i32]* %128
4
[2 x i32]*8	B"
 
	full_text

[2 x i32]* %129
4
[2 x i32]*8	B"
 
	full_text

[2 x i32]* %130
(br8	B 

	full_text

br label %157
:add8
B1
/
	full_text"
 
%158 = add nuw nsw i64 %141, 1
&i648
B

	full_text


i64 %141
:icmp8
B0
.
	full_text!

%159 = icmp eq i64 %158, %132
&i648
B

	full_text


i64 %158
&i648
B

	full_text


i64 %132
=br8
B5
3
	full_text&
$
"br i1 %159, label %160, label %140
$i18
B

	full_text
	
i1 %159
:add8B1
/
	full_text"
 
%161 = add nuw nsw i64 %135, 1
&i648B

	full_text


i64 %135
:icmp8B0
.
	full_text!

%162 = icmp eq i64 %161, %133
&i648B

	full_text


i64 %161
&i648B

	full_text


i64 %133
=br8B5
3
	full_text&
$
"br i1 %162, label %163, label %134
$i18B

	full_text
	
i1 %162
$ret8B

	full_text


ret void
$i328B

	full_text


i32 %7
$i328B

	full_text


i32 %6
,double*8B

	full_text


double* %1
&i32*8B

	full_text
	
i32* %3
$i328B

	full_text


i32 %8
,double*8B

	full_text


double* %0
&i32*8B

	full_text
	
i32* %2
$i328B

	full_text


i32 %5
&i32*8B

	full_text
	
i32* %4
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
#i648B

	full_text	

i64 5
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 2
#i648B

	full_text	

i64 4
4double8B&
$
	full_text

double 1.000000e+00
#i648B

	full_text	

i64 3
#i648B

	full_text	

i64 6
4double8B&
$
	full_text

double 0.000000e+00
$i328B

	full_text


i32 -1
$i328B

	full_text


i32 10
#i648B

	full_text	

i64 0
#i648B

	full_text	

i64 9
#i648B

	full_text	

i64 7
#i648B

	full_text	

i64 1
#i328B

	full_text	

i32 0
$i648B

	full_text


i64 32
#i648B

	full_text	

i64 8
#i648B

	full_text	

i64 2        	
 	                      !    "# "$ "" %& %% '( ') '' *+ ** ,- ,. ,, /0 // 12 13 11 45 44 67 68 66 9: 99 ;< ;= ;; >? >> @A @B @@ CD CC EF EG EE HI HH JK JL JJ MN MM OP OQ OO RS RR TU TV TT WX WW YZ Y[ YY \] \\ ^_ ^` ^^ ab aa cd ce cc fg ff hi hj hh kl kk mn mo mm pq pp rs rt rr uv uu wx wy ww z{ zz |} |~ || 	�  �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �
� �� �
� �� �� �� �� �� �� �� �� �� �� �� �� �
� �� �� �
� �� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �� �� �
� �� �� �� �� �
� �� �� �� �� �
� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �
� �� �� �� �
� �� �� �� �� �� �
� �� �� �� �
� �� �� �
� �� �� �� �� �� �� �
� �� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �
� �
� �� �� �� �� �
� �� �� �� �� �
� �� �� �� �
� �� �� �� �� �
� �� �� �
� �� �� �
� �� �� �
� �
� �
� �� �� �� �� �
� �� �� �� �� �� �
� �� �� �	� � �� �� �� �� � � � � � �� �
� �� ��     
           ! # $" & ( )' + - ., 0 2 31 5 7 86 : < =; ? A B@ D F GE I K LJ N P QO S U VT X Z [Y ] _ `^ b d ec g i jh l n om q s tr v x yw { } ~| � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � � �� � �� � �� �� �� �� �� �� �� � �� � � �� � � � � � � �� � � �� � � �� � � �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� � �� �� �� �� � �� �� � �� � �� �" �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �	 	 �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� � �� � ��� �� �� �� � �� 
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �� *� R� z� �� �� �� �� �� �� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �� � >� f� �� �� �� �� �� �� �
� �
� �
� �
� �
� �
� �	� 	� 	� 	� "	� '	� '	� ,	� ,	� 1	� 1	� 6	� 6	� O	� T	� Y	� ^	� w	� |
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �	� 	� 	� 	� "	� ;	� ;	� @	� @	� E	� E	� J	� J	� O	� T	� Y	� ^	� c	� h	� m	� r
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �
� �� �� �
� �
� �� � �  � %� /� 4� 9� C� H� M� W� \� a� k� p� u� � �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �� �
� �	� 	� 
� �
� �
� �
� �
� �
� �
� �
� �	� c	� h	� m	� r	� w	� |
� �
� �"
kernel_zran3_2"
_Z13get_global_idj"
bubble*�
npb-MG-kernel_zran3_2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282�

wgsize


transfer_bytes	
����

devmap_label


wgsize_log1p
W�A
 
transfer_bytes_log1p
W�A