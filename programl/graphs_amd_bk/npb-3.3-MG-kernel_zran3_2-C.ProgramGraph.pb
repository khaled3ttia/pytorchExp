
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
‚getelementptr8Bo
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
‚getelementptr8Bo
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
Ђgetelementptr8Bm
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
‡getelementptr8Bt
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
Ѓgetelementptr8Bn
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
–call8B‹
€
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
–call8	B‹
€
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
 	                      !    "# "$ "" %& %% '( ') '' *+ ** ,- ,. ,, /0 // 12 13 11 45 44 67 68 66 9: 99 ;< ;= ;; >? >> @A @B @@ CD CC EF EG EE HI HH JK JL JJ MN MM OP OQ OO RS RR TU TV TT WX WW YZ Y[ YY \] \\ ^_ ^` ^^ ab aa cd ce cc fg ff hi hj hh kl kk mn mo mm pq pp rs rt rr uv uu wx wy ww z{ zz |} |~ || 	Ђ  Ѓ‚ Ѓ
ѓ ЃЃ „
… „„ †‡ †
€ †† ‰
Љ ‰‰ ‹Њ ‹
Ќ ‹‹ Ћ
Џ ЋЋ ђ‘ ђ
’ ђђ “
” ““ •– •
— •• 
™  љ› љ
њ љљ ќ
ћ ќќ џ  џ
Ў џџ ў
Ј ўў ¤Ґ ¤
¦ ¤¤ §
Ё §§ ©Є ©
« ©© ¬
­ ¬¬ ®Ї ®
° ®® ±
І ±± іґ і
µ іі ¶
· ¶¶ ё№ ё
є ёё »
ј »» Ѕѕ Ѕ
ї ЅЅ А
Б АА ВГ В
Д ВВ Е
Ж ЕЕ ЗИ З
Й ЗЗ К
Л КК МН М
О ММ П
Р ПП СТ С
У СС Ф
Х ФФ ЦЧ Ц
Ш ЦЦ Щ
Ъ ЩЩ ЫЬ Ы
Э ЫЫ Ю
Я ЮЮ аб а
в аа г
д гг еж е
з ее и
й ии кл к
м кк н
о нн пр п
с пп т
у тт фх ф
ц фф ч
ш чч щъ щ
ы щщ ь
э ьь юя ю
Ђ юю Ѓ
‚ ЃЃ ѓ„ ѓ
… ѓѓ †
‡ †† €‰ €
Љ €€ ‹
Њ ‹‹ ЌЋ Ќ
Џ ЌЌ ђ
‘ ђђ ’“ ’
” ’’ •
– •• — —
™ —— љ
› љљ њќ њ
ћ њњ џ
  џџ Ўў Ў
Ј ЎЎ ¤
Ґ ¤¤ ¦§ ¦
Ё ¦¦ ©
Є ©© «¬ «
­ «« ®
Ї ®® °± °
І °° і
ґ іі µ¶ µ
· µµ ё
№ ёё є» є
ј єє Ѕ
ѕ ЅЅ їА ї
Б її В
Г ВВ ДЕ Д
Ж ДД З
И ЗЗ ЙК Й
Л ЙЙ М
Н ММ ОП О
Р ОО С
Т СС УФ У
Х УУ Ц
Ч ЦЦ ШЩ Ш
Ъ ШШ Ы
Ь ЫЫ ЭЮ Э
Я ЭЭ а
б аа вг в
д вв е
ж ее зи з
й зз к
л кк мн м
о мм п
р пп ст с
у сс ф
х фф цч ц
ш цц щ
ъ щщ ыь ы
э ыы ю
я юю ЂЃ Ђ
‚ ЂЂ ѓ
„ ѓѓ …† …
‡ …… €
‰ €€ Љ‹ Љ
Њ ЉЉ Ќ
Ћ ЌЌ Џђ Џ
‘ ЏЏ ’
“ ’’ ”• ”
– ”” —
 —— ™љ ™
› ™™ њ
ќ њњ ћџ ћ
  ћћ Ў
ў ЎЎ Ј¤ ЈЈ ҐҐ ¦§ ¦Ё ©© ЄЄ «¬ «« ­­ ®Ї ®
° ®® ±І ±
і ±± ґґ µ¶ µ
· µµ ё№ ёё є» є
ј єє Ѕѕ Ѕ
ї ЅЅ АБ АА ВГ В
Д ВВ ЕЖ Е
З ЕЕ ИЙ И
К ИИ ЛМ ЛЛ НО Н
П НН РС РР ТУ Т
Ф ТТ ХЦ ХХ ЧШ Ч
Щ ЧЧ ЪЫ ЪЪ ЬЭ ЬЬ ЮЯ ЮЮ аб аа вв гд гг еж ее з
й ии кл кн м
о мм пр пп ст сс у
х фф цч цщ ш
ъ шш ыь ы
э ыы юя юю ЂЃ ЂЂ ‚ѓ ‚
„ ‚‚ …† …€ ‡
‰ ‡‡ Љ‹ ЉЉ ЊЌ Њ
Ћ ЊЊ Џђ Џ
‘ ЏЏ ’“ ’
” ’’ •– •
— •
 •
™ •• љ› љљ њћ ќ
џ ќќ  Ў    ўЈ ў
¤ ўў Ґ¦ ҐЁ §
© §§ Є« ЄЄ ¬­ ¬
® ¬¬ Ї° Ї
± ЇЇ Іі І
ґ ІІ µ¶ µ
· µ
ё µ
№ µµ єј »» Ѕѕ Ѕ
ї ЅЅ АБ АГ ВВ ДЕ Д
Ж ДД ЗИ З	К К ­Л ҐЛ ЁЛ ґМ Н О П Р С ©С Є
С ґС вТ     
           ! # $" & ( )' + - ., 0 2 31 5 7 86 : < =; ? A B@ D F GE I K LJ N P QO S U VT X Z [Y ] _ `^ b d ec g i jh l n om q s tr v x yw { } ~| Ђ ‚ ѓЃ … ‡ €† Љ Њ Ќ‹ Џ ‘ ’ђ ” – —• ™ › њљ ћ   Ўџ Ј Ґ ¦¤ Ё Є «© ­ Ї °® І ґ µі · № єё ј ѕ їЅ Б Г ДВ Ж И ЙЗ Л Н ОМ Р Т УС Х Ч ШЦ Ъ Ь ЭЫ Я б ва д ж зе й л мк о р сп у х цф ш ъ ыщ э я Ђю ‚ „ …ѓ ‡ ‰ Љ€ Њ Ћ ЏЌ ‘ “ ”’ –  ™— › ќ ћњ   ў ЈЎ Ґ § Ё¦ Є ¬ ­« Ї ± І° ґ ¶ ·µ № » јє ѕ А Бї Г Е ЖД И К ЛЙ Н П РО Т Ф ХУ Ч Щ ЪШ Ь Ю ЯЭ б г дв ж и йз л н ом р т ус х ч шц ъ ь эы я Ѓ ‚Ђ „ † ‡… ‰ ‹ ЊЉ Ћ ђ ‘Џ “ • –”  љ ›™ ќ џ  ћ ў ¤Ґ § ¬Ј Ї­ °« І® іґ ¶Ј ·µ № »ё ј ѕ їЅ Б Г Д Ж З Й КЅ М О ПН С У ФТ Ц Ш ЩЧ ЫН ЭТ ЯЧ б© дЁ жВ йЄ ли нв ои ри т» х± чф щм ъє ьш эы я Ѓю ѓЂ „‚ †ю € ‰ф ‹Љ Ќ Ћп ђ ‘Ј “" ”А –В —Е И ™ы ›љ ћю џЛ Ўќ Ј  ¤ў ¦ќ ЁЛ ©ф «Є ­Р ®с °Х ±Ј іЪ ґА ¶Ь ·Ю ёа №ф ј» ѕг їЅ Би ГВ Ее ЖД И	 	 Й¦ Ё¦ Йз ик мк Ву фЗ ЙЗ иц шц »… ‡… ќА ВА фњ ќҐ §Ґ »є » ФФ Й УУ• ФФ •µ ФФ µ УУ 
Х Ы
Х а
Х е
Х к
Х п
Х ф
Х щ
Х ю
Ц Ј
Ц •
Ч Ґ
Ч Є
Ш і
Ш ё
Ш Ѕ
Ш В
Ш З
Ш М
Ш С
Ш ЦЩ *Щ RЩ zЩ ўЩ КЩ тЩ љЩ ВЩ кЩ ’
Ъ ‹
Ъ ђ
Ъ •
Ъ љ
Ъ џ
Ъ ¤
Ъ ©
Ъ ®
Ы ѓ
Ы €
Ы Ќ
Ы ’
Ы —
Ы њ
Ы Ў
Ы ¦Ь Ь >Ь fЬ ЋЬ ¶Ь ЮЬ †Ь ®Ь ЦЬ ю
Э Ё
Э ©
Э «
Э ­
Ю •
Ю µ	Я 	Я 	Я 	Я "	Я '	Я '	Я ,	Я ,	Я 1	Я 1	Я 6	Я 6	Я O	Я T	Я Y	Я ^	Я w	Я |
Я Ѓ
Я †
Я џ
Я ¤
Я ©
Я ®
Я З
Я М
Я С
Я Ц
Я п
Я ф
Я щ
Я ю
Я —
Я њ
Я Ў
Я ¦
Я ї
Я Д
Я Й
Я О
Я з
Я м
Я с
Я ц
Я Џ
Я ”
Я ™
Я ћ
Я А
Я А
Я В
Я Е
Я И
Я Л
Я Л
Я Л
Я Р
Я Р
Я Р
Я Х
Я Х
Я Х
Я Ъ
Я Ъ
Я Ъ
Я Ь
Я Ь
Я Ю
Я Ю
Я а
Я а
а ы
а Ђ
а …
а Љ
а Џ
а ”
а ™
а ћ
б «
б °
б µ
б є
б ї
б Д
б Й
б О	в 	в 	в 	в "	в ;	в ;	в @	в @	в E	в E	в J	в J	в O	в T	в Y	в ^	в c	в h	в m	в r
в ‹
в ђ
в •
в љ
в і
в ё
в Ѕ
в В
в Ы
в а
в е
в к
в ѓ
в €
в Ќ
в ’
в «
в °
в µ
в є
в У
в Ш
в Э
в в
в ы
в Ђ
в …
в Љв ив ф
в »
в Вг г г  г %г /г 4г 9г Cг Hг Mг Wг \г aг kг pг uг г „г ‰г “г г ќг §г ¬г ±г »г Аг Ег Пг Фг Щг гг иг нг чг ьг Ѓг ‹г ђг •г џг ¤г ©г іг ёг Ѕг Зг Мг Сг Ыг аг ег пг фг щг ѓг €г Ќг —г њг Ў
г µ	д 	д 
е У
е Ш
е Э
е в
е з
е м
е с
е ц	ж c	ж h	ж m	ж r	ж w	ж |
ж Ѓ
ж †"
kernel_zran3_2"
_Z13get_global_idj"
bubble*‘
npb-MG-kernel_zran3_2.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Ѓ

wgsize


transfer_bytes	
РзХч

devmap_label


wgsize_log1p
W°A
 
transfer_bytes_log1p
W°A