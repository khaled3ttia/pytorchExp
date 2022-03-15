

[external]
\getelementptrBK
I
	full_text<
:
8%3 = getelementptr inbounds float, float* %0, i64 614400
\getelementptrBK
I
	full_text<
:
8%4 = getelementptr inbounds float, float* %1, i64 614400
JcallBB
@
	full_text3
1
/%5 = tail call i64 @_Z12get_local_idj(i32 0) #3
4truncB+
)
	full_text

%6 = trunc i64 %5 to i32
"i64B

	full_text


i64 %5
JcallBB
@
	full_text3
1
/%7 = tail call i64 @_Z12get_group_idj(i32 0) #3
4truncB+
)
	full_text

%8 = trunc i64 %7 to i32
"i64B

	full_text


i64 %7
JcallBB
@
	full_text3
1
/%9 = tail call i64 @_Z12get_group_idj(i32 1) #3
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
-shlB&
$
	full_text

%11 = shl i32 %8, 7
"i32B

	full_text


i32 %8
3addB,
*
	full_text

%12 = add nsw i32 %11, %6
#i32B

	full_text
	
i32 %11
"i32B

	full_text


i32 %6
2mulB+
)
	full_text

%13 = mul i32 %10, 15360
#i32B

	full_text
	
i32 %10
4addB-
+
	full_text

%14 = add nsw i32 %12, %13
#i32B

	full_text
	
i32 %12
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%15 = mul nsw i32 %14, 20
#i32B

	full_text
	
i32 %14
4sextB,
*
	full_text

%16 = sext i32 %15 to i64
#i32B

	full_text
	
i32 %15
ZgetelementptrBI
G
	full_text:
8
6%17 = getelementptr inbounds float, float* %3, i64 %16
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %16
JloadBB
@
	full_text3
1
/%18 = load float, float* %17, align 4, !tbaa !8
)float*B

	full_text


float* %17
1addB*
(
	full_text

%19 = add i32 %11, -128
#i32B

	full_text
	
i32 %11
3addB,
*
	full_text

%20 = add nsw i32 %19, %6
#i32B

	full_text
	
i32 %19
"i32B

	full_text


i32 %6
4addB-
+
	full_text

%21 = add nsw i32 %20, %13
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%22 = mul nsw i32 %21, 20
#i32B

	full_text
	
i32 %21
,orB&
$
	full_text

%23 = or i32 %22, 1
#i32B

	full_text
	
i32 %22
4sextB,
*
	full_text

%24 = sext i32 %23 to i64
#i32B

	full_text
	
i32 %23
ZgetelementptrBI
G
	full_text:
8
6%25 = getelementptr inbounds float, float* %3, i64 %24
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %24
JloadBB
@
	full_text3
1
/%26 = load float, float* %25, align 4, !tbaa !8
)float*B

	full_text


float* %25
0addB)
'
	full_text

%27 = add i32 %11, 128
#i32B

	full_text
	
i32 %11
3addB,
*
	full_text

%28 = add nsw i32 %27, %6
#i32B

	full_text
	
i32 %27
"i32B

	full_text


i32 %6
4addB-
+
	full_text

%29 = add nsw i32 %28, %13
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%30 = mul nsw i32 %29, 20
#i32B

	full_text
	
i32 %29
,orB&
$
	full_text

%31 = or i32 %30, 2
#i32B

	full_text
	
i32 %30
4sextB,
*
	full_text

%32 = sext i32 %31 to i64
#i32B

	full_text
	
i32 %31
ZgetelementptrBI
G
	full_text:
8
6%33 = getelementptr inbounds float, float* %3, i64 %32
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %32
JloadBB
@
	full_text3
1
/%34 = load float, float* %33, align 4, !tbaa !8
)float*B

	full_text


float* %33
2addB+
)
	full_text

%35 = add nsw i32 %6, -1
"i32B

	full_text


i32 %6
4addB-
+
	full_text

%36 = add nsw i32 %11, %35
#i32B

	full_text
	
i32 %11
#i32B

	full_text
	
i32 %35
4addB-
+
	full_text

%37 = add nsw i32 %36, %13
#i32B

	full_text
	
i32 %36
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%38 = mul nsw i32 %37, 20
#i32B

	full_text
	
i32 %37
,orB&
$
	full_text

%39 = or i32 %38, 3
#i32B

	full_text
	
i32 %38
4sextB,
*
	full_text

%40 = sext i32 %39 to i64
#i32B

	full_text
	
i32 %39
ZgetelementptrBI
G
	full_text:
8
6%41 = getelementptr inbounds float, float* %3, i64 %40
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %40
JloadBB
@
	full_text3
1
/%42 = load float, float* %41, align 4, !tbaa !8
)float*B

	full_text


float* %41
1addB*
(
	full_text

%43 = add nsw i32 %6, 1
"i32B

	full_text


i32 %6
4addB-
+
	full_text

%44 = add nsw i32 %11, %43
#i32B

	full_text
	
i32 %11
#i32B

	full_text
	
i32 %43
4addB-
+
	full_text

%45 = add nsw i32 %44, %13
#i32B

	full_text
	
i32 %44
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%46 = mul nsw i32 %45, 20
#i32B

	full_text
	
i32 %45
2addB+
)
	full_text

%47 = add nsw i32 %46, 4
#i32B

	full_text
	
i32 %46
4sextB,
*
	full_text

%48 = sext i32 %47 to i64
#i32B

	full_text
	
i32 %47
ZgetelementptrBI
G
	full_text:
8
6%49 = getelementptr inbounds float, float* %3, i64 %48
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %48
JloadBB
@
	full_text3
1
/%50 = load float, float* %49, align 4, !tbaa !8
)float*B

	full_text


float* %49
3addB,
*
	full_text

%51 = add i32 %13, -15360
#i32B

	full_text
	
i32 %13
4addB-
+
	full_text

%52 = add nsw i32 %51, %12
#i32B

	full_text
	
i32 %51
#i32B

	full_text
	
i32 %12
3mulB,
*
	full_text

%53 = mul nsw i32 %52, 20
#i32B

	full_text
	
i32 %52
2addB+
)
	full_text

%54 = add nsw i32 %53, 5
#i32B

	full_text
	
i32 %53
4sextB,
*
	full_text

%55 = sext i32 %54 to i64
#i32B

	full_text
	
i32 %54
ZgetelementptrBI
G
	full_text:
8
6%56 = getelementptr inbounds float, float* %3, i64 %55
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %55
JloadBB
@
	full_text3
1
/%57 = load float, float* %56, align 4, !tbaa !8
)float*B

	full_text


float* %56
2addB+
)
	full_text

%58 = add i32 %13, 15360
#i32B

	full_text
	
i32 %13
4addB-
+
	full_text

%59 = add nsw i32 %58, %12
#i32B

	full_text
	
i32 %58
#i32B

	full_text
	
i32 %12
3mulB,
*
	full_text

%60 = mul nsw i32 %59, 20
#i32B

	full_text
	
i32 %59
2addB+
)
	full_text

%61 = add nsw i32 %60, 6
#i32B

	full_text
	
i32 %60
4sextB,
*
	full_text

%62 = sext i32 %61 to i64
#i32B

	full_text
	
i32 %61
ZgetelementptrBI
G
	full_text:
8
6%63 = getelementptr inbounds float, float* %3, i64 %62
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %62
JloadBB
@
	full_text3
1
/%64 = load float, float* %63, align 4, !tbaa !8
)float*B

	full_text


float* %63
4addB-
+
	full_text

%65 = add nsw i32 %19, %35
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %35
4addB-
+
	full_text

%66 = add nsw i32 %65, %13
#i32B

	full_text
	
i32 %65
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%67 = mul nsw i32 %66, 20
#i32B

	full_text
	
i32 %66
2addB+
)
	full_text

%68 = add nsw i32 %67, 7
#i32B

	full_text
	
i32 %67
4sextB,
*
	full_text

%69 = sext i32 %68 to i64
#i32B

	full_text
	
i32 %68
ZgetelementptrBI
G
	full_text:
8
6%70 = getelementptr inbounds float, float* %3, i64 %69
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %69
JloadBB
@
	full_text3
1
/%71 = load float, float* %70, align 4, !tbaa !8
)float*B

	full_text


float* %70
4addB-
+
	full_text

%72 = add nsw i32 %19, %43
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %43
4addB-
+
	full_text

%73 = add nsw i32 %72, %13
#i32B

	full_text
	
i32 %72
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%74 = mul nsw i32 %73, 20
#i32B

	full_text
	
i32 %73
2addB+
)
	full_text

%75 = add nsw i32 %74, 8
#i32B

	full_text
	
i32 %74
4sextB,
*
	full_text

%76 = sext i32 %75 to i64
#i32B

	full_text
	
i32 %75
ZgetelementptrBI
G
	full_text:
8
6%77 = getelementptr inbounds float, float* %3, i64 %76
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %76
JloadBB
@
	full_text3
1
/%78 = load float, float* %77, align 4, !tbaa !8
)float*B

	full_text


float* %77
4addB-
+
	full_text

%79 = add nsw i32 %27, %35
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %35
4addB-
+
	full_text

%80 = add nsw i32 %79, %13
#i32B

	full_text
	
i32 %79
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%81 = mul nsw i32 %80, 20
#i32B

	full_text
	
i32 %80
2addB+
)
	full_text

%82 = add nsw i32 %81, 9
#i32B

	full_text
	
i32 %81
4sextB,
*
	full_text

%83 = sext i32 %82 to i64
#i32B

	full_text
	
i32 %82
ZgetelementptrBI
G
	full_text:
8
6%84 = getelementptr inbounds float, float* %3, i64 %83
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %83
JloadBB
@
	full_text3
1
/%85 = load float, float* %84, align 4, !tbaa !8
)float*B

	full_text


float* %84
4addB-
+
	full_text

%86 = add nsw i32 %27, %43
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %43
4addB-
+
	full_text

%87 = add nsw i32 %86, %13
#i32B

	full_text
	
i32 %86
#i32B

	full_text
	
i32 %13
3mulB,
*
	full_text

%88 = mul nsw i32 %87, 20
#i32B

	full_text
	
i32 %87
3addB,
*
	full_text

%89 = add nsw i32 %88, 10
#i32B

	full_text
	
i32 %88
4sextB,
*
	full_text

%90 = sext i32 %89 to i64
#i32B

	full_text
	
i32 %89
ZgetelementptrBI
G
	full_text:
8
6%91 = getelementptr inbounds float, float* %3, i64 %90
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %90
JloadBB
@
	full_text3
1
/%92 = load float, float* %91, align 4, !tbaa !8
)float*B

	full_text


float* %91
4addB-
+
	full_text

%93 = add nsw i32 %20, %51
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %51
3mulB,
*
	full_text

%94 = mul nsw i32 %93, 20
#i32B

	full_text
	
i32 %93
3addB,
*
	full_text

%95 = add nsw i32 %94, 11
#i32B

	full_text
	
i32 %94
4sextB,
*
	full_text

%96 = sext i32 %95 to i64
#i32B

	full_text
	
i32 %95
ZgetelementptrBI
G
	full_text:
8
6%97 = getelementptr inbounds float, float* %3, i64 %96
(float*B

	full_text

	float* %3
#i64B

	full_text
	
i64 %96
JloadBB
@
	full_text3
1
/%98 = load float, float* %97, align 4, !tbaa !8
)float*B

	full_text


float* %97
4addB-
+
	full_text

%99 = add nsw i32 %20, %58
#i32B

	full_text
	
i32 %20
#i32B

	full_text
	
i32 %58
4mulB-
+
	full_text

%100 = mul nsw i32 %99, 20
#i32B

	full_text
	
i32 %99
5addB.
,
	full_text

%101 = add nsw i32 %100, 12
$i32B

	full_text


i32 %100
6sextB.
,
	full_text

%102 = sext i32 %101 to i64
$i32B

	full_text


i32 %101
\getelementptrBK
I
	full_text<
:
8%103 = getelementptr inbounds float, float* %3, i64 %102
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %102
LloadBD
B
	full_text5
3
1%104 = load float, float* %103, align 4, !tbaa !8
*float*B

	full_text

float* %103
5addB.
,
	full_text

%105 = add nsw i32 %28, %51
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %51
5mulB.
,
	full_text

%106 = mul nsw i32 %105, 20
$i32B

	full_text


i32 %105
5addB.
,
	full_text

%107 = add nsw i32 %106, 13
$i32B

	full_text


i32 %106
6sextB.
,
	full_text

%108 = sext i32 %107 to i64
$i32B

	full_text


i32 %107
\getelementptrBK
I
	full_text<
:
8%109 = getelementptr inbounds float, float* %3, i64 %108
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %108
LloadBD
B
	full_text5
3
1%110 = load float, float* %109, align 4, !tbaa !8
*float*B

	full_text

float* %109
5addB.
,
	full_text

%111 = add nsw i32 %28, %58
#i32B

	full_text
	
i32 %28
#i32B

	full_text
	
i32 %58
5mulB.
,
	full_text

%112 = mul nsw i32 %111, 20
$i32B

	full_text


i32 %111
5addB.
,
	full_text

%113 = add nsw i32 %112, 14
$i32B

	full_text


i32 %112
6sextB.
,
	full_text

%114 = sext i32 %113 to i64
$i32B

	full_text


i32 %113
\getelementptrBK
I
	full_text<
:
8%115 = getelementptr inbounds float, float* %3, i64 %114
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %114
LloadBD
B
	full_text5
3
1%116 = load float, float* %115, align 4, !tbaa !8
*float*B

	full_text

float* %115
5addB.
,
	full_text

%117 = add nsw i32 %51, %36
#i32B

	full_text
	
i32 %51
#i32B

	full_text
	
i32 %36
5mulB.
,
	full_text

%118 = mul nsw i32 %117, 20
$i32B

	full_text


i32 %117
5addB.
,
	full_text

%119 = add nsw i32 %118, 15
$i32B

	full_text


i32 %118
6sextB.
,
	full_text

%120 = sext i32 %119 to i64
$i32B

	full_text


i32 %119
\getelementptrBK
I
	full_text<
:
8%121 = getelementptr inbounds float, float* %3, i64 %120
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %120
LloadBD
B
	full_text5
3
1%122 = load float, float* %121, align 4, !tbaa !8
*float*B

	full_text

float* %121
5addB.
,
	full_text

%123 = add nsw i32 %58, %36
#i32B

	full_text
	
i32 %58
#i32B

	full_text
	
i32 %36
5mulB.
,
	full_text

%124 = mul nsw i32 %123, 20
$i32B

	full_text


i32 %123
5addB.
,
	full_text

%125 = add nsw i32 %124, 16
$i32B

	full_text


i32 %124
6sextB.
,
	full_text

%126 = sext i32 %125 to i64
$i32B

	full_text


i32 %125
\getelementptrBK
I
	full_text<
:
8%127 = getelementptr inbounds float, float* %3, i64 %126
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %126
LloadBD
B
	full_text5
3
1%128 = load float, float* %127, align 4, !tbaa !8
*float*B

	full_text

float* %127
5addB.
,
	full_text

%129 = add nsw i32 %51, %44
#i32B

	full_text
	
i32 %51
#i32B

	full_text
	
i32 %44
5mulB.
,
	full_text

%130 = mul nsw i32 %129, 20
$i32B

	full_text


i32 %129
5addB.
,
	full_text

%131 = add nsw i32 %130, 17
$i32B

	full_text


i32 %130
6sextB.
,
	full_text

%132 = sext i32 %131 to i64
$i32B

	full_text


i32 %131
\getelementptrBK
I
	full_text<
:
8%133 = getelementptr inbounds float, float* %3, i64 %132
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %132
LloadBD
B
	full_text5
3
1%134 = load float, float* %133, align 4, !tbaa !8
*float*B

	full_text

float* %133
5addB.
,
	full_text

%135 = add nsw i32 %58, %44
#i32B

	full_text
	
i32 %58
#i32B

	full_text
	
i32 %44
5mulB.
,
	full_text

%136 = mul nsw i32 %135, 20
$i32B

	full_text


i32 %135
5addB.
,
	full_text

%137 = add nsw i32 %136, 18
$i32B

	full_text


i32 %136
6sextB.
,
	full_text

%138 = sext i32 %137 to i64
$i32B

	full_text


i32 %137
\getelementptrBK
I
	full_text<
:
8%139 = getelementptr inbounds float, float* %3, i64 %138
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %138
LloadBD
B
	full_text5
3
1%140 = load float, float* %139, align 4, !tbaa !8
*float*B

	full_text

float* %139
4addB-
+
	full_text

%141 = add nsw i32 %15, 19
#i32B

	full_text
	
i32 %15
6sextB.
,
	full_text

%142 = sext i32 %141 to i64
$i32B

	full_text


i32 %141
\getelementptrBK
I
	full_text<
:
8%143 = getelementptr inbounds float, float* %3, i64 %142
(float*B

	full_text

	float* %3
$i64B

	full_text


i64 %142
@bitcastB5
3
	full_text&
$
"%144 = bitcast float* %143 to i32*
*float*B

	full_text

float* %143
HloadB@
>
	full_text1
/
-%145 = load i32, i32* %144, align 4, !tbaa !8
&i32*B

	full_text

	i32* %144
0andB)
'
	full_text

%146 = and i32 %145, 1
$i32B

	full_text


i32 %145
5icmpB-
+
	full_text

%147 = icmp eq i32 %146, 0
$i32B

	full_text


i32 %146
;brB5
3
	full_text&
$
"br i1 %147, label %148, label %299
"i1B

	full_text
	
i1 %147
7fadd8B-
+
	full_text

%149 = fadd float %18, %26
)float8B

	full_text

	float %18
)float8B

	full_text

	float %26
8fadd8B.
,
	full_text

%150 = fadd float %149, %34
*float8B

	full_text


float %149
)float8B

	full_text

	float %34
8fadd8B.
,
	full_text

%151 = fadd float %150, %42
*float8B

	full_text


float %150
)float8B

	full_text

	float %42
8fadd8B.
,
	full_text

%152 = fadd float %151, %50
*float8B

	full_text


float %151
)float8B

	full_text

	float %50
8fadd8B.
,
	full_text

%153 = fadd float %152, %57
*float8B

	full_text


float %152
)float8B

	full_text

	float %57
8fadd8B.
,
	full_text

%154 = fadd float %153, %64
*float8B

	full_text


float %153
)float8B

	full_text

	float %64
8fadd8B.
,
	full_text

%155 = fadd float %154, %71
*float8B

	full_text


float %154
)float8B

	full_text

	float %71
8fadd8B.
,
	full_text

%156 = fadd float %155, %78
*float8B

	full_text


float %155
)float8B

	full_text

	float %78
8fadd8B.
,
	full_text

%157 = fadd float %156, %85
*float8B

	full_text


float %156
)float8B

	full_text

	float %85
8fadd8B.
,
	full_text

%158 = fadd float %157, %92
*float8B

	full_text


float %157
)float8B

	full_text

	float %92
8fadd8B.
,
	full_text

%159 = fadd float %158, %98
*float8B

	full_text


float %158
)float8B

	full_text

	float %98
9fadd8B/
-
	full_text 

%160 = fadd float %159, %104
*float8B

	full_text


float %159
*float8B

	full_text


float %104
9fadd8B/
-
	full_text 

%161 = fadd float %160, %110
*float8B

	full_text


float %160
*float8B

	full_text


float %110
9fadd8B/
-
	full_text 

%162 = fadd float %161, %116
*float8B

	full_text


float %161
*float8B

	full_text


float %116
9fadd8B/
-
	full_text 

%163 = fadd float %162, %122
*float8B

	full_text


float %162
*float8B

	full_text


float %122
9fadd8B/
-
	full_text 

%164 = fadd float %163, %128
*float8B

	full_text


float %163
*float8B

	full_text


float %128
9fadd8B/
-
	full_text 

%165 = fadd float %164, %134
*float8B

	full_text


float %164
*float8B

	full_text


float %134
9fadd8B/
-
	full_text 

%166 = fadd float %165, %140
*float8B

	full_text


float %165
*float8B

	full_text


float %140
7fsub8B-
+
	full_text

%167 = fsub float %42, %50
)float8B

	full_text

	float %42
)float8B

	full_text

	float %50
8fadd8B.
,
	full_text

%168 = fadd float %167, %71
*float8B

	full_text


float %167
)float8B

	full_text

	float %71
8fsub8B.
,
	full_text

%169 = fsub float %168, %78
*float8B

	full_text


float %168
)float8B

	full_text

	float %78
8fadd8B.
,
	full_text

%170 = fadd float %169, %85
*float8B

	full_text


float %169
)float8B

	full_text

	float %85
8fsub8B.
,
	full_text

%171 = fsub float %170, %92
*float8B

	full_text


float %170
)float8B

	full_text

	float %92
9fadd8B/
-
	full_text 

%172 = fadd float %171, %122
*float8B

	full_text


float %171
*float8B

	full_text


float %122
9fadd8B/
-
	full_text 

%173 = fadd float %172, %128
*float8B

	full_text


float %172
*float8B

	full_text


float %128
9fsub8B/
-
	full_text 

%174 = fsub float %173, %134
*float8B

	full_text


float %173
*float8B

	full_text


float %134
9fsub8B/
-
	full_text 

%175 = fsub float %174, %140
*float8B

	full_text


float %174
*float8B

	full_text


float %140
7fsub8B-
+
	full_text

%176 = fsub float %26, %34
)float8B

	full_text

	float %26
)float8B

	full_text

	float %34
8fadd8B.
,
	full_text

%177 = fadd float %176, %71
*float8B

	full_text


float %176
)float8B

	full_text

	float %71
8fadd8B.
,
	full_text

%178 = fadd float %177, %78
*float8B

	full_text


float %177
)float8B

	full_text

	float %78
8fsub8B.
,
	full_text

%179 = fsub float %178, %85
*float8B

	full_text


float %178
)float8B

	full_text

	float %85
8fsub8B.
,
	full_text

%180 = fsub float %179, %92
*float8B

	full_text


float %179
)float8B

	full_text

	float %92
8fadd8B.
,
	full_text

%181 = fadd float %180, %98
*float8B

	full_text


float %180
)float8B

	full_text

	float %98
9fadd8B/
-
	full_text 

%182 = fadd float %181, %104
*float8B

	full_text


float %181
*float8B

	full_text


float %104
9fsub8B/
-
	full_text 

%183 = fsub float %182, %110
*float8B

	full_text


float %182
*float8B

	full_text


float %110
9fsub8B/
-
	full_text 

%184 = fsub float %183, %116
*float8B

	full_text


float %183
*float8B

	full_text


float %116
7fsub8B-
+
	full_text

%185 = fsub float %57, %64
)float8B

	full_text

	float %57
)float8B

	full_text

	float %64
8fadd8B.
,
	full_text

%186 = fadd float %185, %98
*float8B

	full_text


float %185
)float8B

	full_text

	float %98
9fsub8B/
-
	full_text 

%187 = fsub float %186, %104
*float8B

	full_text


float %186
*float8B

	full_text


float %104
9fadd8B/
-
	full_text 

%188 = fadd float %187, %110
*float8B

	full_text


float %187
*float8B

	full_text


float %110
9fsub8B/
-
	full_text 

%189 = fsub float %188, %116
*float8B

	full_text


float %188
*float8B

	full_text


float %116
9fadd8B/
-
	full_text 

%190 = fadd float %189, %122
*float8B

	full_text


float %189
*float8B

	full_text


float %122
9fsub8B/
-
	full_text 

%191 = fsub float %190, %128
*float8B

	full_text


float %190
*float8B

	full_text


float %128
9fadd8B/
-
	full_text 

%192 = fadd float %191, %134
*float8B

	full_text


float %191
*float8B

	full_text


float %134
9fsub8B/
-
	full_text 

%193 = fsub float %192, %140
*float8B

	full_text


float %192
*float8B

	full_text


float %140
Ffdiv8B<
:
	full_text-
+
)%194 = fdiv float %175, %166, !fpmath !12
*float8B

	full_text


float %175
*float8B

	full_text


float %166
Ffdiv8B<
:
	full_text-
+
)%195 = fdiv float %184, %166, !fpmath !12
*float8B

	full_text


float %184
*float8B

	full_text


float %166
Ffdiv8B<
:
	full_text-
+
)%196 = fdiv float %193, %166, !fpmath !12
*float8B

	full_text


float %193
*float8B

	full_text


float %166
2and8B)
'
	full_text

%197 = and i32 %145, 2
&i328B

	full_text


i32 %145
7icmp8B-
+
	full_text

%198 = icmp eq i32 %197, 0
&i328B

	full_text


i32 %197
Zselect8BN
L
	full_text?
=
;%199 = select i1 %198, float %194, float 0x3F747AE140000000
$i18B

	full_text
	
i1 %198
*float8B

	full_text


float %194
Zselect8BN
L
	full_text?
=
;%200 = select i1 %198, float %195, float 0x3F60624DE0000000
$i18B

	full_text
	
i1 %198
*float8B

	full_text


float %195
Tselect8BH
F
	full_text9
7
5%201 = select i1 %198, float %196, float 0.000000e+00
$i18B

	full_text
	
i1 %198
*float8B

	full_text


float %196
9fmul8B/
-
	full_text 

%202 = fmul float %200, %200
*float8B

	full_text


float %200
*float8B

	full_text


float %200
icall8B_
]
	full_textP
N
L%203 = tail call float @llvm.fmuladd.f32(float %199, float %199, float %202)
*float8B

	full_text


float %199
*float8B

	full_text


float %199
*float8B

	full_text


float %202
icall8B_
]
	full_textP
N
L%204 = tail call float @llvm.fmuladd.f32(float %201, float %201, float %203)
*float8B

	full_text


float %201
*float8B

	full_text


float %201
*float8B

	full_text


float %203
zcall8Bp
n
	full_texta
_
]%205 = tail call float @llvm.fmuladd.f32(float %204, float 1.500000e+00, float -1.000000e+00)
*float8B

	full_text


float %204
Gfmul8B=
;
	full_text.
,
*%206 = fmul float %166, 0x3FFF333340000000
*float8B

	full_text


float %166
Gfmul8B=
;
	full_text.
,
*%207 = fmul float %206, 0x3FD5555560000000
*float8B

	full_text


float %206
Bfsub8B8
6
	full_text)
'
%%208 = fsub float -0.000000e+00, %205
*float8B

	full_text


float %205
9fmul8B/
-
	full_text 

%209 = fmul float %207, %208
*float8B

	full_text


float %207
*float8B

	full_text


float %208
vcall8Bl
j
	full_text]
[
Y%210 = tail call float @llvm.fmuladd.f32(float %18, float 0xBFEE666680000000, float %209)
)float8B

	full_text

	float %18
*float8B

	full_text


float %209
Gfmul8B=
;
	full_text.
,
*%211 = fmul float %206, 0x3FAC71C720000000
*float8B

	full_text


float %206
ycall8Bo
m
	full_text`
^
\%212 = tail call float @llvm.fmuladd.f32(float %200, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %200
icall8B_
]
	full_textP
N
L%213 = tail call float @llvm.fmuladd.f32(float %200, float %212, float %208)
*float8B

	full_text


float %200
*float8B

	full_text


float %212
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%214 = fmul float %211, %213
*float8B

	full_text


float %211
*float8B

	full_text


float %213
vcall8Bl
j
	full_text]
[
Y%215 = tail call float @llvm.fmuladd.f32(float %26, float 0xBFEE666680000000, float %214)
)float8B

	full_text

	float %26
*float8B

	full_text


float %214
zcall8Bp
n
	full_texta
_
]%216 = tail call float @llvm.fmuladd.f32(float %200, float 4.500000e+00, float -3.000000e+00)
*float8B

	full_text


float %200
icall8B_
]
	full_textP
N
L%217 = tail call float @llvm.fmuladd.f32(float %200, float %216, float %208)
*float8B

	full_text


float %200
*float8B

	full_text


float %216
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%218 = fmul float %211, %217
*float8B

	full_text


float %211
*float8B

	full_text


float %217
vcall8Bl
j
	full_text]
[
Y%219 = tail call float @llvm.fmuladd.f32(float %34, float 0xBFEE666680000000, float %218)
)float8B

	full_text

	float %34
*float8B

	full_text


float %218
ycall8Bo
m
	full_text`
^
\%220 = tail call float @llvm.fmuladd.f32(float %201, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %201
icall8B_
]
	full_textP
N
L%221 = tail call float @llvm.fmuladd.f32(float %201, float %220, float %208)
*float8B

	full_text


float %201
*float8B

	full_text


float %220
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%222 = fmul float %211, %221
*float8B

	full_text


float %211
*float8B

	full_text


float %221
vcall8Bl
j
	full_text]
[
Y%223 = tail call float @llvm.fmuladd.f32(float %57, float 0xBFEE666680000000, float %222)
)float8B

	full_text

	float %57
*float8B

	full_text


float %222
zcall8Bp
n
	full_texta
_
]%224 = tail call float @llvm.fmuladd.f32(float %201, float 4.500000e+00, float -3.000000e+00)
*float8B

	full_text


float %201
icall8B_
]
	full_textP
N
L%225 = tail call float @llvm.fmuladd.f32(float %201, float %224, float %208)
*float8B

	full_text


float %201
*float8B

	full_text


float %224
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%226 = fmul float %211, %225
*float8B

	full_text


float %211
*float8B

	full_text


float %225
vcall8Bl
j
	full_text]
[
Y%227 = tail call float @llvm.fmuladd.f32(float %64, float 0xBFEE666680000000, float %226)
)float8B

	full_text

	float %64
*float8B

	full_text


float %226
ycall8Bo
m
	full_text`
^
\%228 = tail call float @llvm.fmuladd.f32(float %199, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %199
icall8B_
]
	full_textP
N
L%229 = tail call float @llvm.fmuladd.f32(float %199, float %228, float %208)
*float8B

	full_text


float %199
*float8B

	full_text


float %228
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%230 = fmul float %211, %229
*float8B

	full_text


float %211
*float8B

	full_text


float %229
vcall8Bl
j
	full_text]
[
Y%231 = tail call float @llvm.fmuladd.f32(float %42, float 0xBFEE666680000000, float %230)
)float8B

	full_text

	float %42
*float8B

	full_text


float %230
zcall8Bp
n
	full_texta
_
]%232 = tail call float @llvm.fmuladd.f32(float %199, float 4.500000e+00, float -3.000000e+00)
*float8B

	full_text


float %199
icall8B_
]
	full_textP
N
L%233 = tail call float @llvm.fmuladd.f32(float %199, float %232, float %208)
*float8B

	full_text


float %199
*float8B

	full_text


float %232
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%234 = fmul float %211, %233
*float8B

	full_text


float %211
*float8B

	full_text


float %233
vcall8Bl
j
	full_text]
[
Y%235 = tail call float @llvm.fmuladd.f32(float %50, float 0xBFEE666680000000, float %234)
)float8B

	full_text

	float %50
*float8B

	full_text


float %234
Gfmul8B=
;
	full_text.
,
*%236 = fmul float %206, 0x3F9C71C720000000
*float8B

	full_text


float %206
9fadd8B/
-
	full_text 

%237 = fadd float %200, %201
*float8B

	full_text


float %200
*float8B

	full_text


float %201
ycall8Bo
m
	full_text`
^
\%238 = tail call float @llvm.fmuladd.f32(float %237, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %237
icall8B_
]
	full_textP
N
L%239 = tail call float @llvm.fmuladd.f32(float %237, float %238, float %208)
*float8B

	full_text


float %237
*float8B

	full_text


float %238
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%240 = fmul float %236, %239
*float8B

	full_text


float %236
*float8B

	full_text


float %239
vcall8Bl
j
	full_text]
[
Y%241 = tail call float @llvm.fmuladd.f32(float %98, float 0xBFEE666680000000, float %240)
)float8B

	full_text

	float %98
*float8B

	full_text


float %240
9fsub8B/
-
	full_text 

%242 = fsub float %200, %201
*float8B

	full_text


float %200
*float8B

	full_text


float %201
ycall8Bo
m
	full_text`
^
\%243 = tail call float @llvm.fmuladd.f32(float %242, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %242
icall8B_
]
	full_textP
N
L%244 = tail call float @llvm.fmuladd.f32(float %242, float %243, float %208)
*float8B

	full_text


float %242
*float8B

	full_text


float %243
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%245 = fmul float %236, %244
*float8B

	full_text


float %236
*float8B

	full_text


float %244
wcall8Bm
k
	full_text^
\
Z%246 = tail call float @llvm.fmuladd.f32(float %104, float 0xBFEE666680000000, float %245)
*float8B

	full_text


float %104
*float8B

	full_text


float %245
Bfsub8B8
6
	full_text)
'
%%247 = fsub float -0.000000e+00, %200
*float8B

	full_text


float %200
9fsub8B/
-
	full_text 

%248 = fsub float %201, %200
*float8B

	full_text


float %201
*float8B

	full_text


float %200
ycall8Bo
m
	full_text`
^
\%249 = tail call float @llvm.fmuladd.f32(float %248, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %248
icall8B_
]
	full_textP
N
L%250 = tail call float @llvm.fmuladd.f32(float %248, float %249, float %208)
*float8B

	full_text


float %248
*float8B

	full_text


float %249
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%251 = fmul float %236, %250
*float8B

	full_text


float %236
*float8B

	full_text


float %250
wcall8Bm
k
	full_text^
\
Z%252 = tail call float @llvm.fmuladd.f32(float %110, float 0xBFEE666680000000, float %251)
*float8B

	full_text


float %110
*float8B

	full_text


float %251
9fsub8B/
-
	full_text 

%253 = fsub float %247, %201
*float8B

	full_text


float %247
*float8B

	full_text


float %201
ycall8Bo
m
	full_text`
^
\%254 = tail call float @llvm.fmuladd.f32(float %253, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %253
icall8B_
]
	full_textP
N
L%255 = tail call float @llvm.fmuladd.f32(float %253, float %254, float %208)
*float8B

	full_text


float %253
*float8B

	full_text


float %254
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%256 = fmul float %236, %255
*float8B

	full_text


float %236
*float8B

	full_text


float %255
wcall8Bm
k
	full_text^
\
Z%257 = tail call float @llvm.fmuladd.f32(float %116, float 0xBFEE666680000000, float %256)
*float8B

	full_text


float %116
*float8B

	full_text


float %256
9fadd8B/
-
	full_text 

%258 = fadd float %199, %200
*float8B

	full_text


float %199
*float8B

	full_text


float %200
ycall8Bo
m
	full_text`
^
\%259 = tail call float @llvm.fmuladd.f32(float %258, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %258
icall8B_
]
	full_textP
N
L%260 = tail call float @llvm.fmuladd.f32(float %258, float %259, float %208)
*float8B

	full_text


float %258
*float8B

	full_text


float %259
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%261 = fmul float %236, %260
*float8B

	full_text


float %236
*float8B

	full_text


float %260
vcall8Bl
j
	full_text]
[
Y%262 = tail call float @llvm.fmuladd.f32(float %71, float 0xBFEE666680000000, float %261)
)float8B

	full_text

	float %71
*float8B

	full_text


float %261
9fsub8B/
-
	full_text 

%263 = fsub float %199, %200
*float8B

	full_text


float %199
*float8B

	full_text


float %200
ycall8Bo
m
	full_text`
^
\%264 = tail call float @llvm.fmuladd.f32(float %263, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %263
icall8B_
]
	full_textP
N
L%265 = tail call float @llvm.fmuladd.f32(float %263, float %264, float %208)
*float8B

	full_text


float %263
*float8B

	full_text


float %264
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%266 = fmul float %236, %265
*float8B

	full_text


float %236
*float8B

	full_text


float %265
vcall8Bl
j
	full_text]
[
Y%267 = tail call float @llvm.fmuladd.f32(float %85, float 0xBFEE666680000000, float %266)
)float8B

	full_text

	float %85
*float8B

	full_text


float %266
9fadd8B/
-
	full_text 

%268 = fadd float %199, %201
*float8B

	full_text


float %199
*float8B

	full_text


float %201
ycall8Bo
m
	full_text`
^
\%269 = tail call float @llvm.fmuladd.f32(float %268, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %268
icall8B_
]
	full_textP
N
L%270 = tail call float @llvm.fmuladd.f32(float %268, float %269, float %208)
*float8B

	full_text


float %268
*float8B

	full_text


float %269
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%271 = fmul float %236, %270
*float8B

	full_text


float %236
*float8B

	full_text


float %270
wcall8Bm
k
	full_text^
\
Z%272 = tail call float @llvm.fmuladd.f32(float %122, float 0xBFEE666680000000, float %271)
*float8B

	full_text


float %122
*float8B

	full_text


float %271
9fsub8B/
-
	full_text 

%273 = fsub float %199, %201
*float8B

	full_text


float %199
*float8B

	full_text


float %201
ycall8Bo
m
	full_text`
^
\%274 = tail call float @llvm.fmuladd.f32(float %273, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %273
icall8B_
]
	full_textP
N
L%275 = tail call float @llvm.fmuladd.f32(float %273, float %274, float %208)
*float8B

	full_text


float %273
*float8B

	full_text


float %274
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%276 = fmul float %236, %275
*float8B

	full_text


float %236
*float8B

	full_text


float %275
wcall8Bm
k
	full_text^
\
Z%277 = tail call float @llvm.fmuladd.f32(float %128, float 0xBFEE666680000000, float %276)
*float8B

	full_text


float %128
*float8B

	full_text


float %276
Bfsub8B8
6
	full_text)
'
%%278 = fsub float -0.000000e+00, %199
*float8B

	full_text


float %199
9fsub8B/
-
	full_text 

%279 = fsub float %200, %199
*float8B

	full_text


float %200
*float8B

	full_text


float %199
ycall8Bo
m
	full_text`
^
\%280 = tail call float @llvm.fmuladd.f32(float %279, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %279
icall8B_
]
	full_textP
N
L%281 = tail call float @llvm.fmuladd.f32(float %279, float %280, float %208)
*float8B

	full_text


float %279
*float8B

	full_text


float %280
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%282 = fmul float %236, %281
*float8B

	full_text


float %236
*float8B

	full_text


float %281
vcall8Bl
j
	full_text]
[
Y%283 = tail call float @llvm.fmuladd.f32(float %78, float 0xBFEE666680000000, float %282)
)float8B

	full_text

	float %78
*float8B

	full_text


float %282
9fsub8B/
-
	full_text 

%284 = fsub float %278, %200
*float8B

	full_text


float %278
*float8B

	full_text


float %200
ycall8Bo
m
	full_text`
^
\%285 = tail call float @llvm.fmuladd.f32(float %284, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %284
icall8B_
]
	full_textP
N
L%286 = tail call float @llvm.fmuladd.f32(float %284, float %285, float %208)
*float8B

	full_text


float %284
*float8B

	full_text


float %285
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%287 = fmul float %236, %286
*float8B

	full_text


float %236
*float8B

	full_text


float %286
vcall8Bl
j
	full_text]
[
Y%288 = tail call float @llvm.fmuladd.f32(float %92, float 0xBFEE666680000000, float %287)
)float8B

	full_text

	float %92
*float8B

	full_text


float %287
9fsub8B/
-
	full_text 

%289 = fsub float %201, %199
*float8B

	full_text


float %201
*float8B

	full_text


float %199
ycall8Bo
m
	full_text`
^
\%290 = tail call float @llvm.fmuladd.f32(float %289, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %289
icall8B_
]
	full_textP
N
L%291 = tail call float @llvm.fmuladd.f32(float %289, float %290, float %208)
*float8B

	full_text


float %289
*float8B

	full_text


float %290
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%292 = fmul float %236, %291
*float8B

	full_text


float %236
*float8B

	full_text


float %291
wcall8Bm
k
	full_text^
\
Z%293 = tail call float @llvm.fmuladd.f32(float %134, float 0xBFEE666680000000, float %292)
*float8B

	full_text


float %134
*float8B

	full_text


float %292
9fsub8B/
-
	full_text 

%294 = fsub float %278, %201
*float8B

	full_text


float %278
*float8B

	full_text


float %201
ycall8Bo
m
	full_text`
^
\%295 = tail call float @llvm.fmuladd.f32(float %294, float 4.500000e+00, float 3.000000e+00)
*float8B

	full_text


float %294
icall8B_
]
	full_textP
N
L%296 = tail call float @llvm.fmuladd.f32(float %294, float %295, float %208)
*float8B

	full_text


float %294
*float8B

	full_text


float %295
*float8B

	full_text


float %208
9fmul8B/
-
	full_text 

%297 = fmul float %236, %296
*float8B

	full_text


float %236
*float8B

	full_text


float %296
wcall8Bm
k
	full_text^
\
Z%298 = tail call float @llvm.fmuladd.f32(float %140, float 0xBFEE666680000000, float %297)
*float8B

	full_text


float %140
*float8B

	full_text


float %297
(br8B 

	full_text

br label %299
Hphi8B?
=
	full_text0
.
,%300 = phi float [ %231, %148 ], [ %50, %2 ]
*float8B

	full_text


float %231
)float8B

	full_text

	float %50
Hphi8B?
=
	full_text0
.
,%301 = phi float [ %235, %148 ], [ %42, %2 ]
*float8B

	full_text


float %235
)float8B

	full_text

	float %42
Hphi8B?
=
	full_text0
.
,%302 = phi float [ %223, %148 ], [ %64, %2 ]
*float8B

	full_text


float %223
)float8B

	full_text

	float %64
Hphi8B?
=
	full_text0
.
,%303 = phi float [ %227, %148 ], [ %57, %2 ]
*float8B

	full_text


float %227
)float8B

	full_text

	float %57
Hphi8B?
=
	full_text0
.
,%304 = phi float [ %262, %148 ], [ %92, %2 ]
*float8B

	full_text


float %262
)float8B

	full_text

	float %92
Hphi8B?
=
	full_text0
.
,%305 = phi float [ %283, %148 ], [ %85, %2 ]
*float8B

	full_text


float %283
)float8B

	full_text

	float %85
Hphi8B?
=
	full_text0
.
,%306 = phi float [ %267, %148 ], [ %78, %2 ]
*float8B

	full_text


float %267
)float8B

	full_text

	float %78
Hphi8B?
=
	full_text0
.
,%307 = phi float [ %288, %148 ], [ %71, %2 ]
*float8B

	full_text


float %288
)float8B

	full_text

	float %71
Iphi8B@
>
	full_text1
/
-%308 = phi float [ %241, %148 ], [ %116, %2 ]
*float8B

	full_text


float %241
*float8B

	full_text


float %116
Iphi8B@
>
	full_text1
/
-%309 = phi float [ %246, %148 ], [ %110, %2 ]
*float8B

	full_text


float %246
*float8B

	full_text


float %110
Iphi8B@
>
	full_text1
/
-%310 = phi float [ %252, %148 ], [ %104, %2 ]
*float8B

	full_text


float %252
*float8B

	full_text


float %104
Hphi8B?
=
	full_text0
.
,%311 = phi float [ %257, %148 ], [ %98, %2 ]
*float8B

	full_text


float %257
)float8B

	full_text

	float %98
Iphi8B@
>
	full_text1
/
-%312 = phi float [ %272, %148 ], [ %140, %2 ]
*float8B

	full_text


float %272
*float8B

	full_text


float %140
Iphi8B@
>
	full_text1
/
-%313 = phi float [ %277, %148 ], [ %134, %2 ]
*float8B

	full_text


float %277
*float8B

	full_text


float %134
Iphi8B@
>
	full_text1
/
-%314 = phi float [ %293, %148 ], [ %128, %2 ]
*float8B

	full_text


float %293
*float8B

	full_text


float %128
Iphi8B@
>
	full_text1
/
-%315 = phi float [ %298, %148 ], [ %122, %2 ]
*float8B

	full_text


float %298
*float8B

	full_text


float %122
Hphi8B?
=
	full_text0
.
,%316 = phi float [ %219, %148 ], [ %26, %2 ]
*float8B

	full_text


float %219
)float8B

	full_text

	float %26
Hphi8B?
=
	full_text0
.
,%317 = phi float [ %215, %148 ], [ %34, %2 ]
*float8B

	full_text


float %215
)float8B

	full_text

	float %34
Hphi8B?
=
	full_text0
.
,%318 = phi float [ %210, %148 ], [ %18, %2 ]
*float8B

	full_text


float %210
)float8B

	full_text

	float %18
]getelementptr8BJ
H
	full_text;
9
7%319 = getelementptr inbounds float, float* %4, i64 %16
*float*8B

	full_text

	float* %4
%i648B

	full_text
	
i64 %16
Nstore8BC
A
	full_text4
2
0store float %318, float* %319, align 4, !tbaa !8
*float8B

	full_text


float %318
,float*8B

	full_text

float* %319
/or8B'
%
	full_text

%320 = or i32 %15, 1
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%321 = sext i32 %320 to i64
&i328B

	full_text


i32 %320
^getelementptr8BK
I
	full_text<
:
8%322 = getelementptr inbounds float, float* %4, i64 %321
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %321
Nstore8BC
A
	full_text4
2
0store float %317, float* %322, align 4, !tbaa !8
*float8B

	full_text


float %317
,float*8B

	full_text

float* %322
/or8B'
%
	full_text

%323 = or i32 %15, 2
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%324 = sext i32 %323 to i64
&i328B

	full_text


i32 %323
^getelementptr8BK
I
	full_text<
:
8%325 = getelementptr inbounds float, float* %4, i64 %324
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %324
Nstore8BC
A
	full_text4
2
0store float %316, float* %325, align 4, !tbaa !8
*float8B

	full_text


float %316
,float*8B

	full_text

float* %325
/or8B'
%
	full_text

%326 = or i32 %15, 3
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%327 = sext i32 %326 to i64
&i328B

	full_text


i32 %326
^getelementptr8BK
I
	full_text<
:
8%328 = getelementptr inbounds float, float* %4, i64 %327
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %327
Nstore8BC
A
	full_text4
2
0store float %300, float* %328, align 4, !tbaa !8
*float8B

	full_text


float %300
,float*8B

	full_text

float* %328
5add8B,
*
	full_text

%329 = add nsw i32 %15, 4
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%330 = sext i32 %329 to i64
&i328B

	full_text


i32 %329
^getelementptr8BK
I
	full_text<
:
8%331 = getelementptr inbounds float, float* %4, i64 %330
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %330
Nstore8BC
A
	full_text4
2
0store float %301, float* %331, align 4, !tbaa !8
*float8B

	full_text


float %301
,float*8B

	full_text

float* %331
5add8B,
*
	full_text

%332 = add nsw i32 %15, 5
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%333 = sext i32 %332 to i64
&i328B

	full_text


i32 %332
^getelementptr8BK
I
	full_text<
:
8%334 = getelementptr inbounds float, float* %4, i64 %333
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %333
Nstore8BC
A
	full_text4
2
0store float %302, float* %334, align 4, !tbaa !8
*float8B

	full_text


float %302
,float*8B

	full_text

float* %334
5add8B,
*
	full_text

%335 = add nsw i32 %15, 6
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%336 = sext i32 %335 to i64
&i328B

	full_text


i32 %335
^getelementptr8BK
I
	full_text<
:
8%337 = getelementptr inbounds float, float* %4, i64 %336
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %336
Nstore8BC
A
	full_text4
2
0store float %303, float* %337, align 4, !tbaa !8
*float8B

	full_text


float %303
,float*8B

	full_text

float* %337
5add8B,
*
	full_text

%338 = add nsw i32 %15, 7
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%339 = sext i32 %338 to i64
&i328B

	full_text


i32 %338
^getelementptr8BK
I
	full_text<
:
8%340 = getelementptr inbounds float, float* %4, i64 %339
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %339
Nstore8BC
A
	full_text4
2
0store float %304, float* %340, align 4, !tbaa !8
*float8B

	full_text


float %304
,float*8B

	full_text

float* %340
5add8B,
*
	full_text

%341 = add nsw i32 %15, 8
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%342 = sext i32 %341 to i64
&i328B

	full_text


i32 %341
^getelementptr8BK
I
	full_text<
:
8%343 = getelementptr inbounds float, float* %4, i64 %342
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %342
Nstore8BC
A
	full_text4
2
0store float %305, float* %343, align 4, !tbaa !8
*float8B

	full_text


float %305
,float*8B

	full_text

float* %343
5add8B,
*
	full_text

%344 = add nsw i32 %15, 9
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%345 = sext i32 %344 to i64
&i328B

	full_text


i32 %344
^getelementptr8BK
I
	full_text<
:
8%346 = getelementptr inbounds float, float* %4, i64 %345
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %345
Nstore8BC
A
	full_text4
2
0store float %306, float* %346, align 4, !tbaa !8
*float8B

	full_text


float %306
,float*8B

	full_text

float* %346
6add8B-
+
	full_text

%347 = add nsw i32 %15, 10
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%348 = sext i32 %347 to i64
&i328B

	full_text


i32 %347
^getelementptr8BK
I
	full_text<
:
8%349 = getelementptr inbounds float, float* %4, i64 %348
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %348
Nstore8BC
A
	full_text4
2
0store float %307, float* %349, align 4, !tbaa !8
*float8B

	full_text


float %307
,float*8B

	full_text

float* %349
6add8B-
+
	full_text

%350 = add nsw i32 %15, 11
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%351 = sext i32 %350 to i64
&i328B

	full_text


i32 %350
^getelementptr8BK
I
	full_text<
:
8%352 = getelementptr inbounds float, float* %4, i64 %351
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %351
Nstore8BC
A
	full_text4
2
0store float %308, float* %352, align 4, !tbaa !8
*float8B

	full_text


float %308
,float*8B

	full_text

float* %352
6add8B-
+
	full_text

%353 = add nsw i32 %15, 12
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%354 = sext i32 %353 to i64
&i328B

	full_text


i32 %353
^getelementptr8BK
I
	full_text<
:
8%355 = getelementptr inbounds float, float* %4, i64 %354
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %354
Nstore8BC
A
	full_text4
2
0store float %309, float* %355, align 4, !tbaa !8
*float8B

	full_text


float %309
,float*8B

	full_text

float* %355
6add8B-
+
	full_text

%356 = add nsw i32 %15, 13
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%357 = sext i32 %356 to i64
&i328B

	full_text


i32 %356
^getelementptr8BK
I
	full_text<
:
8%358 = getelementptr inbounds float, float* %4, i64 %357
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %357
Nstore8BC
A
	full_text4
2
0store float %310, float* %358, align 4, !tbaa !8
*float8B

	full_text


float %310
,float*8B

	full_text

float* %358
6add8B-
+
	full_text

%359 = add nsw i32 %15, 14
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%360 = sext i32 %359 to i64
&i328B

	full_text


i32 %359
^getelementptr8BK
I
	full_text<
:
8%361 = getelementptr inbounds float, float* %4, i64 %360
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %360
Nstore8BC
A
	full_text4
2
0store float %311, float* %361, align 4, !tbaa !8
*float8B

	full_text


float %311
,float*8B

	full_text

float* %361
6add8B-
+
	full_text

%362 = add nsw i32 %15, 15
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%363 = sext i32 %362 to i64
&i328B

	full_text


i32 %362
^getelementptr8BK
I
	full_text<
:
8%364 = getelementptr inbounds float, float* %4, i64 %363
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %363
Nstore8BC
A
	full_text4
2
0store float %312, float* %364, align 4, !tbaa !8
*float8B

	full_text


float %312
,float*8B

	full_text

float* %364
6add8B-
+
	full_text

%365 = add nsw i32 %15, 16
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%366 = sext i32 %365 to i64
&i328B

	full_text


i32 %365
^getelementptr8BK
I
	full_text<
:
8%367 = getelementptr inbounds float, float* %4, i64 %366
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %366
Nstore8BC
A
	full_text4
2
0store float %313, float* %367, align 4, !tbaa !8
*float8B

	full_text


float %313
,float*8B

	full_text

float* %367
6add8B-
+
	full_text

%368 = add nsw i32 %15, 17
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%369 = sext i32 %368 to i64
&i328B

	full_text


i32 %368
^getelementptr8BK
I
	full_text<
:
8%370 = getelementptr inbounds float, float* %4, i64 %369
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %369
Nstore8BC
A
	full_text4
2
0store float %314, float* %370, align 4, !tbaa !8
*float8B

	full_text


float %314
,float*8B

	full_text

float* %370
6add8B-
+
	full_text

%371 = add nsw i32 %15, 18
%i328B

	full_text
	
i32 %15
8sext8B.
,
	full_text

%372 = sext i32 %371 to i64
&i328B

	full_text


i32 %371
^getelementptr8BK
I
	full_text<
:
8%373 = getelementptr inbounds float, float* %4, i64 %372
*float*8B

	full_text

	float* %4
&i648B

	full_text


i64 %372
Nstore8BC
A
	full_text4
2
0store float %315, float* %373, align 4, !tbaa !8
*float8B

	full_text


float %315
,float*8B

	full_text

float* %373
$ret8B

	full_text


ret void
*float*8B

	full_text

	float* %1
*float*8B

	full_text

	float* %0
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
$i328B

	full_text


i32 10
3float8B&
$
	full_text

float -1.000000e+00
'i328B

	full_text

	i32 15360
2float8B%
#
	full_text

float 3.000000e+00
#i328B

	full_text	

i32 1
&i328B

	full_text


i32 -128
(i648B

	full_text


i64 614400
$i328B

	full_text


i32 13
$i328B

	full_text


i32 18
2float8B%
#
	full_text

float 4.500000e+00
#i328B

	full_text	

i32 5
$i328B

	full_text


i32 16
2float8B%
#
	full_text

float 0.000000e+00
8float8B+
)
	full_text

float 0x3F9C71C720000000
$i328B

	full_text


i32 -1
%i328B

	full_text
	
i32 128
(i328B

	full_text


i32 -15360
#i328B

	full_text	

i32 9
8float8B+
)
	full_text

float 0x3FD5555560000000
8float8B+
)
	full_text

float 0x3FAC71C720000000
$i328B

	full_text


i32 19
$i328B

	full_text


i32 11
$i328B

	full_text


i32 15
#i328B

	full_text	

i32 4
$i328B

	full_text


i32 12
#i328B

	full_text	

i32 0
$i328B

	full_text


i32 17
8float8B+
)
	full_text

float 0x3FFF333340000000
3float8B&
$
	full_text

float -0.000000e+00
#i328B

	full_text	

i32 8
3float8B&
$
	full_text

float -3.000000e+00
8float8B+
)
	full_text

float 0x3F60624DE0000000
$i328B

	full_text


i32 14
8float8B+
)
	full_text

float 0x3F747AE140000000
2float8B%
#
	full_text

float 1.500000e+00
8float8B+
)
	full_text

float 0xBFEE666680000000
#i328B

	full_text	

i32 2
#i328B

	full_text	

i32 3
#i328B

	full_text	

i32 6
#i328B

	full_text	

i32 7
$i328B

	full_text


i32 20        		 
 

                       !" !# !! $% $& $$ '( '' )* )) +, ++ -. -/ -- 01 00 23 22 45 46 44 78 79 77 :; :: <= << >? >> @A @B @@ CD CC EF EE GH GI GG JK JL JJ MN MM OP OO QR QQ ST SU SS VW VV XY XX Z[ Z\ ZZ ]^ ]_ ]] `a `` bc bb de dd fg fh ff ij ii kl kk mn mo mm pq pp rs rr tu tt vw vx vv yz yy {| {{ }~ } }} € €€ ‚ƒ ‚‚ „… „„ †‡ †
ˆ †† ‰Š ‰‰ ‹Œ ‹
 ‹‹  
  ‘’ ‘‘ “” ““ •– •• —˜ —
™ —— š› šš œ œ
 œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «« ­® ­
¯ ­­ °± °
² °° ³´ ³³ µ¶ µµ ·¸ ·· ¹º ¹
» ¹¹ ¼½ ¼¼ ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ ÆÆ ÈÉ ÈÈ ÊË Ê
Ì ÊÊ ÍÎ ÍÍ ÏĞ Ï
Ñ ÏÏ ÒÓ ÒÒ ÔÕ ÔÔ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ ÛÛ İŞ İ
ß İİ àá àà âã ââ äå ää æç æ
è ææ éê éé ëì ë
í ëë îï îî ğñ ğğ òó òò ôõ ô
ö ôô ÷ø ÷÷ ùú ù
û ùù üı üü şÿ şş € €€ ‚ƒ ‚
„ ‚‚ …† …… ‡ˆ ‡
‰ ‡‡ Š‹ ŠŠ Œ ŒŒ   ‘ 
’  “” ““ •– •
— •• ˜™ ˜˜ š› šš œ œœ Ÿ 
   ¡¢ ¡¡ £¤ £
¥ ££ ¦§ ¦¦ ¨© ¨¨ ª« ªª ¬­ ¬
® ¬¬ ¯° ¯¯ ±² ±
³ ±± ´µ ´´ ¶· ¶¶ ¸¹ ¸¸ º» º
¼ ºº ½¾ ½½ ¿À ¿¿ ÁÂ ÁÁ ÃÄ Ã
Å ÃÃ ÆÇ ÆÆ ÈÉ ÈÈ ÊË ÊÊ ÌÍ ÌÌ ÎÏ ÎÑ Ğ
Ò ĞĞ ÓÔ Ó
Õ ÓÓ Ö× Ö
Ø ÖÖ ÙÚ Ù
Û ÙÙ Üİ Ü
Ş ÜÜ ßà ß
á ßß âã â
ä ââ åæ å
ç åå èé è
ê èè ëì ë
í ëë îï î
ğ îî ñò ñ
ó ññ ôõ ô
ö ôô ÷ø ÷
ù ÷÷ úû ú
ü úú ış ı
ÿ ıı € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †
ˆ †† ‰Š ‰
‹ ‰‰ Œ Œ
 ŒŒ  
‘  ’“ ’
” ’’ •– •
— •• ˜™ ˜
š ˜˜ ›œ ›
 ›› Ÿ 
   ¡¢ ¡
£ ¡¡ ¤¥ ¤
¦ ¤¤ §¨ §
© §§ ª« ª
¬ ªª ­® ­
¯ ­­ °± °
² °° ³´ ³
µ ³³ ¶· ¶
¸ ¶¶ ¹º ¹
» ¹¹ ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ Â
Ä ÂÂ ÅÆ Å
Ç ÅÅ ÈÉ È
Ê ÈÈ ËÌ Ë
Í ËË ÎÏ Î
Ğ ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ Ô
Ö ÔÔ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá àà âã ââ äå ä
æ ää çè ç
é çç êë ê
ì êê íî í
ï íí ğñ ğ
ò ğ
ó ğğ ôõ ô
ö ô
÷ ôô øù øø úû úú üı üü ş
ÿ şş € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆˆ Š‹ Š
Œ Š
 ŠŠ  
  ‘’ ‘
“ ‘‘ ”• ”” –— –
˜ –
™ –– š› š
œ šš  
Ÿ   ¡    ¢£ ¢
¤ ¢
¥ ¢¢ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬¬ ®¯ ®
° ®
± ®® ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸¸ º» º
¼ º
½ ºº ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ ÄÄ ÆÇ Æ
È Æ
É ÆÆ ÊË Ê
Ì ÊÊ ÍÎ Í
Ï ÍÍ ĞÑ ĞĞ ÒÓ Ò
Ô ÒÒ ÕÖ ÕÕ ×Ø ×
Ù ×
Ú ×× ÛÜ Û
İ ÛÛ Şß Ş
à ŞŞ áâ á
ã áá äå ää æç æ
è æ
é ææ êë ê
ì êê íî í
ï íí ğ
ñ ğğ òó ò
ô òò õö õõ ÷ø ÷
ù ÷
ú ÷÷ ûü û
ı ûû şÿ ş
€ şş ‚ 
ƒ  „… „„ †‡ †
ˆ †
‰ †† Š‹ Š
Œ ŠŠ  
  ‘ 
’  “” ““ •– •
— •
˜ •• ™š ™
› ™™ œ œ
 œœ Ÿ  Ÿ
¡ ŸŸ ¢£ ¢¢ ¤¥ ¤
¦ ¤
§ ¤¤ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®
° ®® ±² ±± ³´ ³
µ ³
¶ ³³ ·¸ ·
¹ ·· º» º
¼ ºº ½¾ ½
¿ ½½ ÀÁ ÀÀ ÂÃ Â
Ä Â
Å ÂÂ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ Ì
Í ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ ÑÑ ÓÔ Ó
Õ Ó
Ö ÓÓ ×Ø ×
Ù ×× ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá àà âã â
ä â
å ââ æç æ
è ææ éê é
ë éé ìí ì
î ìì ïğ ïï ñò ñ
ó ñ
ô ññ õö õ
÷ õõ øù ø
ú øø ûü û
ı ûû şÿ şş € €
‚ €
ƒ €€ „… „
† „„ ‡ˆ ‡
‰ ‡‡ ŠŒ ‹
 ‹‹  
  ‘’ ‘
“ ‘‘ ”• ”
– ”” —˜ —
™ —— š› š
œ šš  
Ÿ   ¡  
¢    £¤ £
¥ ££ ¦§ ¦
¨ ¦¦ ©ª ©
« ©© ¬­ ¬
® ¬¬ ¯° ¯
± ¯¯ ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸
º ¸¸ »¼ »
½ »» ¾¿ ¾
À ¾¾ ÁÂ Á
Ã ÁÁ ÄÅ Ä
Æ ÄÄ ÇÈ Ç
É ÇÇ ÊË ÊÊ ÌÍ ÌÌ ÎÏ Î
Ğ ÎÎ ÑÒ Ñ
Ó ÑÑ ÔÕ ÔÔ Ö× ÖÖ ØÙ Ø
Ú ØØ ÛÜ Û
İ ÛÛ Şß ŞŞ àá àà âã â
ä ââ åæ å
ç åå èé èè êë êê ìí ì
î ìì ïğ ï
ñ ïï òó òò ôõ ôô ö÷ ö
ø öö ùú ù
û ùù üı üü şÿ şş € €
‚ €€ ƒ„ ƒ
… ƒƒ †‡ †† ˆ‰ ˆˆ Š‹ Š
Œ ŠŠ  
  ‘  ’“ ’’ ”• ”
– ”” —˜ —
™ —— š› šš œ œœ Ÿ 
   ¡¢ ¡
£ ¡¡ ¤¥ ¤¤ ¦§ ¦¦ ¨© ¨
ª ¨¨ «¬ «
­ «« ®¯ ®® °± °° ²³ ²
´ ²² µ¶ µ
· µµ ¸¹ ¸¸ º» ºº ¼½ ¼
¾ ¼¼ ¿À ¿
Á ¿¿ ÂÃ ÂÂ ÄÅ ÄÄ ÆÇ Æ
È ÆÆ ÉÊ É
Ë ÉÉ ÌÍ ÌÌ ÎÏ ÎÎ ĞÑ Ğ
Ò ĞĞ ÓÔ Ó
Õ ÓÓ Ö× ÖÖ ØÙ ØØ ÚÛ Ú
Ü ÚÚ İŞ İ
ß İİ àá àà âã ââ äå ä
æ ää çè ç
é çç êë êê ìí ìì îï î
ğ îî ñò ñ
ó ññ ôõ ôô ö÷ öö øù ø
ú øø ûü û
ı ûû şÿ €   	    
           " #! % &$ (' *) , .+ /- 1 32 5 64 8 97 ;: =< ? A> B@ D F HE IG K LJ NM PO R TQ US W Y [X \Z ^ _] a` cb e gd hf j lk n om qp sr u wt xv z |{ ~ } € ƒ‚ … ‡„ ˆ† Š ŒE ‹   ’‘ ”“ – ˜• ™— › X œ   ¡Ÿ £¢ ¥¤ § ©¦ ª¨ ¬2 ®E ¯­ ± ²° ´³ ¶µ ¸ º· »¹ ½2 ¿X À¾ Â ÃÁ ÅÄ ÇÆ É ËÈ ÌÊ Î! Ğk ÑÏ ÓÒ ÕÔ × ÙÖ ÚØ Ü! Ş{ ßİ áà ãâ å çä èæ ê4 ìk íë ïî ñğ ó õò öô ø4 ú{ ûù ıü ÿş  ƒ€ „‚ †k ˆG ‰‡ ‹Š Œ  ‘ ’ ”{ –G —• ™˜ ›š  Ÿœ   ¢k ¤Z ¥£ §¦ ©¨ « ­ª ®¬ °{ ²Z ³± µ´ ·¶ ¹ »¸ ¼º ¾ À¿ Â ÄÁ ÅÃ ÇÆ ÉÈ ËÊ ÍÌ Ï Ñ0 ÒĞ ÔC ÕÓ ×V ØÖ Úi ÛÙ İy ŞÜ à‰ áß ãš äâ æ« çå é¼ êè ìÍ íë ïÛ ğî òé óñ õ÷ öô ø… ù÷ û“ üú ş¡ ÿı ¯ ‚€ „½ …V ‡i ˆ† Šš ‹‰ « Œ ¼ ‘ “Í ”’ –“ —• ™¡ š˜ œ¯ › Ÿ½  0 ¢C £¡ ¥š ¦¤ ¨« ©§ «¼ ¬ª ®Í ¯­ ±Û ²° ´é µ³ ·÷ ¸¶ º… »y ½‰ ¾¼ ÀÛ Á¿ Ãé ÄÂ Æ÷ ÇÅ É… ÊÈ Ì“ ÍË Ï¡ ĞÎ Ò¯ ÓÑ Õ½ Ö Øƒ Ù¹ Ûƒ ÜÔ Şƒ ßÈ áà ãâ å× æâ èÚ éâ ëİ ìç îç ïä ñä òí óê õê öğ ÷ô ùƒ ûú ıø ÿü ş ‚ „€ …ú ‡ç ‰ç ‹ˆ Œş † Š 0 ’ “ç •ç —” ˜ş ™† ›– œC š Ÿê ¡ê £  ¤ş ¥† §¢ ¨y ª¦ «ê ­ê ¯¬ °ş ±† ³® ´‰ ¶² ·ä ¹ä »¸ ¼ş ½† ¿º ÀV Â¾ Ãä Åä ÇÄ Èş É† ËÆ Ìi ÎÊ Ïú Ñç Óê ÔÒ ÖÒ ØÕ Ùş ÚĞ Ü× İÛ ßÛ àç âê ãá åá çä èş éĞ ëæ ìé îê ïç ñê óç ôò öò øõ ùş úĞ ü÷ ı÷ ÿû €ğ ‚ê ƒ … ‡„ ˆş ‰Ğ ‹† Œ… Š ä ‘ç ’ ” –“ —ş ˜Ğ š• ›š ™ ä  ç ¡Ÿ £Ÿ ¥¢ ¦ş §Ğ ©¤ ª¼ ¬¨ ­ä ¯ê °® ²® ´± µş ¶Ğ ¸³ ¹“ »· ¼ä ¾ê ¿½ Á½ ÃÀ Äş ÅĞ ÇÂ È¡ ÊÆ Ëä Íç Ïä ĞÎ ÒÎ ÔÑ Õş ÖĞ ØÓ Ù« Û× ÜÌ Şç ßİ áİ ãà äş åĞ çâ èÍ êæ ëê íä îì ğì òï óş ôĞ öñ ÷¯ ùõ úÌ üê ıû ÿû ş ‚ş ƒĞ …€ †½ ˆ„ ‰Á Œi Í V © ’‰ “µ •y –œ ˜Í ™Ú ›¼ œ« « Ÿé ¡š ¢Ş ¤… ¥í §÷ ¨ş ªé « ­Û ®º °½ ±É ³¯ ´ø ¶¡ ·‡ ¹“ º ¼0 ½‘ ¿C Àƒ Â Ã Å ÆÁ ÈÄ É ËÊ Í ÏÌ Ğ¾ ÒÎ Ó ÕÔ × ÙÖ Ú» ÜØ İ ßŞ á ãà ä‹ æâ ç éè ë íê î ğì ñ óò õ ÷ô ø‘ úö û ıü ÿ ş ‚” „€ … ‡† ‰ ‹ˆ Œ— Š  ‘ “ •’ –š ˜” ™ ›š  Ÿœ   ¢ £ ¥¤ § ©¦ ª  ¬¨ ­ ¯® ± ³° ´£ ¶² · ¹¸ » ½º ¾¦ À¼ Á ÃÂ Å ÇÄ È© ÊÆ Ë ÍÌ Ï ÑÎ Ò¬ ÔĞ Õ ×Ö Ù ÛØ Ü¯ ŞÚ ß áà ã åâ æ² èä é ëê í ïì ğµ òî ó õô ÷ ùö ú¸ üø ıÎ ĞÎ ‹Š ‹ ş ‚‚  ƒƒé ƒƒ é– ƒƒ –º ƒƒ º‡ ƒƒ ‡ş ƒƒ şÚ ƒƒ Ú	 ‚‚ 	‘ ƒƒ ‘ğ ƒƒ ğœ ƒƒ œ† ƒƒ †  ƒƒ  ï ƒƒ ïñ ƒƒ ñ€ ƒƒ € ƒƒ © ƒƒ ©ş ƒƒ şŞ ƒƒ Şø ƒƒ øÍ ƒƒ Íí ƒƒ í± ƒƒ ±â ƒƒ âÓ ƒƒ Ó ‚‚ „ ƒƒ „É ƒƒ ÉÁ ƒƒ Áä ƒƒ äô ƒƒ ô” ƒƒ ”æ ƒƒ æ• ƒƒ •“ ƒƒ “® ƒƒ ®¤ ƒƒ ¤ƒ ƒƒ ƒ  Æ ƒƒ Æˆ ƒƒ ˆÄ ƒƒ Ä³ ƒƒ ³ ƒƒ À ƒƒ À¸ ƒƒ ¸õ ƒƒ õµ ƒƒ µŠ ƒƒ Š¢ ƒƒ ¢« ƒƒ «ø ƒƒ øÑ ƒƒ Ñà ƒƒ à× ƒƒ ×÷ ƒƒ ÷¢ ƒƒ ¢Â ƒƒ Âº ƒƒ ºÕ ƒƒ Õ¬ ƒƒ ¬
„ Æ
„ ¤
… ø	† 	† {
‡ ˆ
‡  
‡ ¸
‡ Õ
‡ ä
‡ õ
‡ „
‡ “
‡ ¢
‡ ±
‡ À
‡ Ñ
‡ à
‡ ï
‡ şˆ 		ˆ )	ˆ X
ˆ Ê
ˆ Ê	‰ 	Š 	Š 
‹ ğ
‹ Â
Œ ¶
Œ ô
 ˆ
 ”
  
 ¬
 ¸
 Ä
 Õ
 ä
 õ
 „
 “
 ¢
 ±
 À
 Ñ
 à
 ï
 ş	 r
 ò
 š
 à
 ê
‘ Ğ	’ E	“ 2	” k
• µ
• š
– ü
— †
˜ ¿
™ Ô
™ ®
š Œ
š Ö	› b
› è
œ â
œ ¸  
 Ì
 â
 ¨
 ê
Ÿ ú  ş  ğ  Ì
¡ ¤
¡ 
¢ ”
¢ ¬
¢ Ä
£ ç
¤ ş
¤ Ì
¥ ä
¦ ø
§ ƒ
§ ‘
§ 
§ ©
§ µ
§ Á
§ Í
§ Ş
§ í
§ ş
§ 
§ œ
§ «
§ º
§ É
§ Ú
§ é
§ ø
§ ‡	¨ <
¨ à
¨ Ô	© O
© Ş
ª ‚
ª ü	« 
« “
« †	¬ 	¬ '	¬ :	¬ M	¬ `	¬ p
¬ €
¬ ‘
¬ ¢
¬ ³
¬ Ä
¬ Ò
¬ à
¬ î
¬ ü
¬ Š
¬ ˜
¬ ¦
¬ ´"
performStreamCollide_kernel"
_Z12get_local_idj"
_Z12get_group_idj"
llvm.fmuladd.f32*—
performStreamCollide_kernel.clu
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S128
4
llvm_target_triple

x86_64-apple-macosx10.13.02
 
transfer_bytes_log1p
`A

wgsize
x

transfer_bytes	
€€¼´

wgsize_log1p
`A

devmap_label
