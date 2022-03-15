

[external]
JcallBB
@
	full_text3
1
/%8 = tail call i64 @_Z12get_group_idj(i32 0) #5
4truncB+
)
	full_text

%9 = trunc i64 %8 to i32
"i64B

	full_text


i64 %8
KcallBC
A
	full_text4
2
0%10 = tail call i64 @_Z12get_group_idj(i32 1) #5
6truncB-
+
	full_text

%11 = trunc i64 %10 to i32
#i64B

	full_text
	
i64 %10
McallBE
C
	full_text6
4
2%12 = tail call i64 @_Z14get_num_groupsj(i32 1) #5
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
0%14 = tail call i64 @_Z12get_local_idj(i32 0) #5
6truncB-
+
	full_text

%15 = trunc i64 %14 to i32
#i64B

	full_text
	
i64 %14
KcallBC
A
	full_text4
2
0%16 = tail call i64 @_Z12get_local_idj(i32 1) #5
6truncB-
+
	full_text

%17 = trunc i64 %16 to i32
#i64B

	full_text
	
i64 %16
McallBE
C
	full_text6
4
2%18 = tail call i64 @_Z14get_local_sizej(i32 1) #5
6truncB-
+
	full_text

%19 = trunc i64 %18 to i32
#i64B

	full_text
	
i64 %18
WcallBO
M
	full_text@
>
<%20 = tail call i32 @ToGlobalRow(i32 %9, i32 16, i32 %15) #6
"i32B

	full_text


i32 %9
#i32B

	full_text
	
i32 %15
YcallBQ
O
	full_textB
@
>%21 = tail call i32 @ToGlobalCol(i32 %11, i32 %19, i32 %17) #6
#i32B

	full_text
	
i32 %11
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %17
4mulB-
+
	full_text

%22 = mul nsw i32 %19, %13
#i32B

	full_text
	
i32 %19
#i32B

	full_text
	
i32 %13
2addB+
)
	full_text

%23 = add nsw i32 %22, 2
#i32B

	full_text
	
i32 %22
1sremB)
'
	full_text

%24 = srem i32 %23, %2
#i32B

	full_text
	
i32 %23
3icmpB+
)
	full_text

%25 = icmp eq i32 %24, 0
#i32B

	full_text
	
i32 %24
3subB,
*
	full_text

%26 = sub nsw i32 %2, %24
#i32B

	full_text
	
i32 %24
@selectB6
4
	full_text'
%
#%27 = select i1 %25, i32 0, i32 %26
!i1B

	full_text


i1 %25
#i32B

	full_text
	
i32 %26
0addB)
'
	full_text

%28 = add i32 %27, %22
#i32B

	full_text
	
i32 %27
#i32B

	full_text
	
i32 %22
3addB,
*
	full_text

%29 = add nsw i32 %15, -1
#i32B

	full_text
	
i32 %15
3addB,
*
	full_text

%30 = add nsw i32 %20, -1
#i32B

	full_text
	
i32 %20
%brB

	full_text

br label %33
5icmp8B+
)
	full_text

%32 = icmp eq i32 %17, 0
%i328B

	full_text
	
i32 %17
:br8B2
0
	full_text#
!
br i1 %32, label %48, label %65
#i18B

	full_text


i1 %32
Aphi8B8
6
	full_text)
'
%%34 = phi i32 [ 0, %7 ], [ %46, %33 ]
%i328B

	full_text
	
i32 %46
6add8B-
+
	full_text

%35 = add nsw i32 %29, %34
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %34
_call8BU
S
	full_textF
D
B%36 = tail call i32 @ToFlatHaloedIdx(i32 %35, i32 %17, i32 %19) #6
%i328B

	full_text
	
i32 %35
%i328B

	full_text
	
i32 %17
%i328B

	full_text
	
i32 %19
6add8B-
+
	full_text

%37 = add nsw i32 %30, %34
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %34
_call8BU
S
	full_textF
D
B%38 = tail call i32 @ToFlatHaloedIdx(i32 %37, i32 %21, i32 %28) #6
%i328B

	full_text
	
i32 %37
%i328B

	full_text
	
i32 %21
%i328B

	full_text
	
i32 %28
6sext8B,
*
	full_text

%39 = sext i32 %38 to i64
%i328B

	full_text
	
i32 %38
\getelementptr8BI
G
	full_text:
8
6%40 = getelementptr inbounds float, float* %0, i64 %39
%i648B

	full_text
	
i64 %39
@bitcast8B3
1
	full_text$
"
 %41 = bitcast float* %40 to i32*
+float*8B

	full_text


float* %40
Hload8B>
<
	full_text/
-
+%42 = load i32, i32* %41, align 4, !tbaa !8
'i32*8B

	full_text


i32* %41
6sext8B,
*
	full_text

%43 = sext i32 %36 to i64
%i328B

	full_text
	
i32 %36
\getelementptr8BI
G
	full_text:
8
6%44 = getelementptr inbounds float, float* %6, i64 %43
%i648B

	full_text
	
i64 %43
@bitcast8B3
1
	full_text$
"
 %45 = bitcast float* %44 to i32*
+float*8B

	full_text


float* %44
Hstore8B=
;
	full_text.
,
*store i32 %42, i32* %45, align 4, !tbaa !8
%i328B

	full_text
	
i32 %42
'i32*8B

	full_text


i32* %45
8add8B/
-
	full_text 

%46 = add nuw nsw i32 %34, 1
%i328B

	full_text
	
i32 %34
6icmp8B,
*
	full_text

%47 = icmp eq i32 %46, 18
%i328B

	full_text
	
i32 %46
:br8B2
0
	full_text#
!
br i1 %47, label %31, label %33
#i18B

	full_text


i1 %47
5add8B,
*
	full_text

%49 = add nsw i32 %21, -1
%i328B

	full_text
	
i32 %21
'br8B

	full_text

br label %50
Bphi8B9
7
	full_text*
(
&%51 = phi i32 [ 0, %48 ], [ %63, %50 ]
%i328B

	full_text
	
i32 %63
6add8B-
+
	full_text

%52 = add nsw i32 %29, %51
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %51
^call8BT
R
	full_textE
C
A%53 = tail call i32 @ToFlatHaloedIdx(i32 %52, i32 -1, i32 %19) #6
%i328B

	full_text
	
i32 %52
%i328B

	full_text
	
i32 %19
6add8B-
+
	full_text

%54 = add nsw i32 %30, %51
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %51
_call8BU
S
	full_textF
D
B%55 = tail call i32 @ToFlatHaloedIdx(i32 %54, i32 %49, i32 %28) #6
%i328B

	full_text
	
i32 %54
%i328B

	full_text
	
i32 %49
%i328B

	full_text
	
i32 %28
6sext8B,
*
	full_text

%56 = sext i32 %55 to i64
%i328B

	full_text
	
i32 %55
\getelementptr8BI
G
	full_text:
8
6%57 = getelementptr inbounds float, float* %0, i64 %56
%i648B

	full_text
	
i64 %56
@bitcast8B3
1
	full_text$
"
 %58 = bitcast float* %57 to i32*
+float*8B

	full_text


float* %57
Hload8B>
<
	full_text/
-
+%59 = load i32, i32* %58, align 4, !tbaa !8
'i32*8B

	full_text


i32* %58
6sext8B,
*
	full_text

%60 = sext i32 %53 to i64
%i328B

	full_text
	
i32 %53
\getelementptr8BI
G
	full_text:
8
6%61 = getelementptr inbounds float, float* %6, i64 %60
%i648B

	full_text
	
i64 %60
@bitcast8B3
1
	full_text$
"
 %62 = bitcast float* %61 to i32*
+float*8B

	full_text


float* %61
Hstore8B=
;
	full_text.
,
*store i32 %59, i32* %62, align 4, !tbaa !8
%i328B

	full_text
	
i32 %59
'i32*8B

	full_text


i32* %62
8add8B/
-
	full_text 

%63 = add nuw nsw i32 %51, 1
%i328B

	full_text
	
i32 %51
6icmp8B,
*
	full_text

%64 = icmp eq i32 %63, 18
%i328B

	full_text
	
i32 %63
:br8B2
0
	full_text#
!
br i1 %64, label %86, label %50
#i18B

	full_text


i1 %64
5add8B,
*
	full_text

%66 = add nsw i32 %19, -1
%i328B

	full_text
	
i32 %19
7icmp8B-
+
	full_text

%67 = icmp eq i32 %66, %17
%i328B

	full_text
	
i32 %66
%i328B

	full_text
	
i32 %17
:br8B2
0
	full_text#
!
br i1 %67, label %68, label %86
#i18B

	full_text


i1 %67
4add8B+
)
	full_text

%69 = add nsw i32 %17, 1
%i328B

	full_text
	
i32 %17
4add8B+
)
	full_text

%70 = add nsw i32 %21, 1
%i328B

	full_text
	
i32 %21
'br8B

	full_text

br label %71
Bphi8B9
7
	full_text*
(
&%72 = phi i32 [ 0, %68 ], [ %84, %71 ]
%i328B

	full_text
	
i32 %84
6add8B-
+
	full_text

%73 = add nsw i32 %29, %72
%i328B

	full_text
	
i32 %29
%i328B

	full_text
	
i32 %72
_call8BU
S
	full_textF
D
B%74 = tail call i32 @ToFlatHaloedIdx(i32 %73, i32 %69, i32 %19) #6
%i328B

	full_text
	
i32 %73
%i328B

	full_text
	
i32 %69
%i328B

	full_text
	
i32 %19
6add8B-
+
	full_text

%75 = add nsw i32 %30, %72
%i328B

	full_text
	
i32 %30
%i328B

	full_text
	
i32 %72
_call8BU
S
	full_textF
D
B%76 = tail call i32 @ToFlatHaloedIdx(i32 %75, i32 %70, i32 %28) #6
%i328B

	full_text
	
i32 %75
%i328B

	full_text
	
i32 %70
%i328B

	full_text
	
i32 %28
6sext8B,
*
	full_text

%77 = sext i32 %76 to i64
%i328B

	full_text
	
i32 %76
\getelementptr8BI
G
	full_text:
8
6%78 = getelementptr inbounds float, float* %0, i64 %77
%i648B

	full_text
	
i64 %77
@bitcast8B3
1
	full_text$
"
 %79 = bitcast float* %78 to i32*
+float*8B

	full_text


float* %78
Hload8B>
<
	full_text/
-
+%80 = load i32, i32* %79, align 4, !tbaa !8
'i32*8B

	full_text


i32* %79
6sext8B,
*
	full_text

%81 = sext i32 %74 to i64
%i328B

	full_text
	
i32 %74
\getelementptr8BI
G
	full_text:
8
6%82 = getelementptr inbounds float, float* %6, i64 %81
%i648B

	full_text
	
i64 %81
@bitcast8B3
1
	full_text$
"
 %83 = bitcast float* %82 to i32*
+float*8B

	full_text


float* %82
Hstore8B=
;
	full_text.
,
*store i32 %80, i32* %83, align 4, !tbaa !8
%i328B

	full_text
	
i32 %80
'i32*8B

	full_text


i32* %83
8add8B/
-
	full_text 

%84 = add nuw nsw i32 %72, 1
%i328B

	full_text
	
i32 %72
6icmp8B,
*
	full_text

%85 = icmp eq i32 %84, 18
%i328B

	full_text
	
i32 %84
:br8B2
0
	full_text#
!
br i1 %85, label %86, label %71
#i18B

	full_text


i1 %85
Bcall8B8
6
	full_text)
'
%tail call void @_Z7barrierj(i32 1) #7
4add8B+
)
	full_text

%87 = add nsw i32 %15, 1
%i328B

	full_text
	
i32 %15
4add8B+
)
	full_text

%88 = add nsw i32 %17, 1
%i328B

	full_text
	
i32 %17
5add8B,
*
	full_text

%89 = add nsw i32 %17, -1
%i328B

	full_text
	
i32 %17
'br8B

	full_text

br label %91
$ret8	B

	full_text


ret void
Cphi8
B:
8
	full_text+
)
'%92 = phi i32 [ 0, %86 ], [ %145, %91 ]
&i328
B

	full_text


i32 %145
6add8
B-
+
	full_text

%93 = add nsw i32 %92, %15
%i328
B

	full_text
	
i32 %92
%i328
B

	full_text
	
i32 %15
_call8
BU
S
	full_textF
D
B%94 = tail call i32 @ToFlatHaloedIdx(i32 %93, i32 %17, i32 %19) #6
%i328
B

	full_text
	
i32 %93
%i328
B

	full_text
	
i32 %17
%i328
B

	full_text
	
i32 %19
6add8
B-
+
	full_text

%95 = add nsw i32 %29, %92
%i328
B

	full_text
	
i32 %29
%i328
B

	full_text
	
i32 %92
_call8
BU
S
	full_textF
D
B%96 = tail call i32 @ToFlatHaloedIdx(i32 %95, i32 %17, i32 %19) #6
%i328
B

	full_text
	
i32 %95
%i328
B

	full_text
	
i32 %17
%i328
B

	full_text
	
i32 %19
6add8
B-
+
	full_text

%97 = add nsw i32 %87, %92
%i328
B

	full_text
	
i32 %87
%i328
B

	full_text
	
i32 %92
_call8
BU
S
	full_textF
D
B%98 = tail call i32 @ToFlatHaloedIdx(i32 %97, i32 %17, i32 %19) #6
%i328
B

	full_text
	
i32 %97
%i328
B

	full_text
	
i32 %17
%i328
B

	full_text
	
i32 %19
_call8
BU
S
	full_textF
D
B%99 = tail call i32 @ToFlatHaloedIdx(i32 %93, i32 %88, i32 %19) #6
%i328
B

	full_text
	
i32 %93
%i328
B

	full_text
	
i32 %88
%i328
B

	full_text
	
i32 %19
`call8
BV
T
	full_textG
E
C%100 = tail call i32 @ToFlatHaloedIdx(i32 %93, i32 %89, i32 %19) #6
%i328
B

	full_text
	
i32 %93
%i328
B

	full_text
	
i32 %89
%i328
B

	full_text
	
i32 %19
`call8
BV
T
	full_textG
E
C%101 = tail call i32 @ToFlatHaloedIdx(i32 %95, i32 %88, i32 %19) #6
%i328
B

	full_text
	
i32 %95
%i328
B

	full_text
	
i32 %88
%i328
B

	full_text
	
i32 %19
`call8
BV
T
	full_textG
E
C%102 = tail call i32 @ToFlatHaloedIdx(i32 %97, i32 %88, i32 %19) #6
%i328
B

	full_text
	
i32 %97
%i328
B

	full_text
	
i32 %88
%i328
B

	full_text
	
i32 %19
`call8
BV
T
	full_textG
E
C%103 = tail call i32 @ToFlatHaloedIdx(i32 %95, i32 %89, i32 %19) #6
%i328
B

	full_text
	
i32 %95
%i328
B

	full_text
	
i32 %89
%i328
B

	full_text
	
i32 %19
`call8
BV
T
	full_textG
E
C%104 = tail call i32 @ToFlatHaloedIdx(i32 %97, i32 %89, i32 %19) #6
%i328
B

	full_text
	
i32 %97
%i328
B

	full_text
	
i32 %89
%i328
B

	full_text
	
i32 %19
7sext8
B-
+
	full_text

%105 = sext i32 %94 to i64
%i328
B

	full_text
	
i32 %94
^getelementptr8
BK
I
	full_text<
:
8%106 = getelementptr inbounds float, float* %6, i64 %105
&i648
B

	full_text


i64 %105
Nload8
BD
B
	full_text5
3
1%107 = load float, float* %106, align 4, !tbaa !8
,float*8
B

	full_text

float* %106
7sext8
B-
+
	full_text

%108 = sext i32 %96 to i64
%i328
B

	full_text
	
i32 %96
^getelementptr8
BK
I
	full_text<
:
8%109 = getelementptr inbounds float, float* %6, i64 %108
&i648
B

	full_text


i64 %108
Nload8
BD
B
	full_text5
3
1%110 = load float, float* %109, align 4, !tbaa !8
,float*8
B

	full_text

float* %109
7sext8
B-
+
	full_text

%111 = sext i32 %98 to i64
%i328
B

	full_text
	
i32 %98
^getelementptr8
BK
I
	full_text<
:
8%112 = getelementptr inbounds float, float* %6, i64 %111
&i648
B

	full_text


i64 %111
Nload8
BD
B
	full_text5
3
1%113 = load float, float* %112, align 4, !tbaa !8
,float*8
B

	full_text

float* %112
9fadd8
B/
-
	full_text 

%114 = fadd float %110, %113
*float8
B

	full_text


float %110
*float8
B

	full_text


float %113
7sext8
B-
+
	full_text

%115 = sext i32 %99 to i64
%i328
B

	full_text
	
i32 %99
^getelementptr8
BK
I
	full_text<
:
8%116 = getelementptr inbounds float, float* %6, i64 %115
&i648
B

	full_text


i64 %115
Nload8
BD
B
	full_text5
3
1%117 = load float, float* %116, align 4, !tbaa !8
,float*8
B

	full_text

float* %116
9fadd8
B/
-
	full_text 

%118 = fadd float %114, %117
*float8
B

	full_text


float %114
*float8
B

	full_text


float %117
8sext8
B.
,
	full_text

%119 = sext i32 %100 to i64
&i328
B

	full_text


i32 %100
^getelementptr8
BK
I
	full_text<
:
8%120 = getelementptr inbounds float, float* %6, i64 %119
&i648
B

	full_text


i64 %119
Nload8
BD
B
	full_text5
3
1%121 = load float, float* %120, align 4, !tbaa !8
,float*8
B

	full_text

float* %120
9fadd8
B/
-
	full_text 

%122 = fadd float %118, %121
*float8
B

	full_text


float %118
*float8
B

	full_text


float %121
8sext8
B.
,
	full_text

%123 = sext i32 %101 to i64
&i328
B

	full_text


i32 %101
^getelementptr8
BK
I
	full_text<
:
8%124 = getelementptr inbounds float, float* %6, i64 %123
&i648
B

	full_text


i64 %123
Nload8
BD
B
	full_text5
3
1%125 = load float, float* %124, align 4, !tbaa !8
,float*8
B

	full_text

float* %124
8sext8
B.
,
	full_text

%126 = sext i32 %102 to i64
&i328
B

	full_text


i32 %102
^getelementptr8
BK
I
	full_text<
:
8%127 = getelementptr inbounds float, float* %6, i64 %126
&i648
B

	full_text


i64 %126
Nload8
BD
B
	full_text5
3
1%128 = load float, float* %127, align 4, !tbaa !8
,float*8
B

	full_text

float* %127
9fadd8
B/
-
	full_text 

%129 = fadd float %125, %128
*float8
B

	full_text


float %125
*float8
B

	full_text


float %128
8sext8
B.
,
	full_text

%130 = sext i32 %103 to i64
&i328
B

	full_text


i32 %103
^getelementptr8
BK
I
	full_text<
:
8%131 = getelementptr inbounds float, float* %6, i64 %130
&i648
B

	full_text


i64 %130
Nload8
BD
B
	full_text5
3
1%132 = load float, float* %131, align 4, !tbaa !8
,float*8
B

	full_text

float* %131
9fadd8
B/
-
	full_text 

%133 = fadd float %129, %132
*float8
B

	full_text


float %129
*float8
B

	full_text


float %132
8sext8
B.
,
	full_text

%134 = sext i32 %104 to i64
&i328
B

	full_text


i32 %104
^getelementptr8
BK
I
	full_text<
:
8%135 = getelementptr inbounds float, float* %6, i64 %134
&i648
B

	full_text


i64 %134
Nload8
BD
B
	full_text5
3
1%136 = load float, float* %135, align 4, !tbaa !8
,float*8
B

	full_text

float* %135
9fadd8
B/
-
	full_text 

%137 = fadd float %133, %136
*float8
B

	full_text


float %133
*float8
B

	full_text


float %136
7fmul8
B-
+
	full_text

%138 = fmul float %122, %4
*float8
B

	full_text


float %122
gcall8
B]
[
	full_textN
L
J%139 = tail call float @llvm.fmuladd.f32(float %3, float %107, float %138)
*float8
B

	full_text


float %107
*float8
B

	full_text


float %138
gcall8
B]
[
	full_textN
L
J%140 = tail call float @llvm.fmuladd.f32(float %5, float %137, float %139)
*float8
B

	full_text


float %137
*float8
B

	full_text


float %139
7add8
B.
,
	full_text

%141 = add nsw i32 %92, %20
%i328
B

	full_text
	
i32 %92
%i328
B

	full_text
	
i32 %20
acall8
BW
U
	full_textH
F
D%142 = tail call i32 @ToFlatHaloedIdx(i32 %141, i32 %21, i32 %28) #6
&i328
B

	full_text


i32 %141
%i328
B

	full_text
	
i32 %21
%i328
B

	full_text
	
i32 %28
8sext8
B.
,
	full_text

%143 = sext i32 %142 to i64
&i328
B

	full_text


i32 %142
^getelementptr8
BK
I
	full_text<
:
8%144 = getelementptr inbounds float, float* %1, i64 %143
&i648
B

	full_text


i64 %143
Nstore8
BC
A
	full_text4
2
0store float %140, float* %144, align 4, !tbaa !8
*float8
B

	full_text


float %140
,float*8
B

	full_text

float* %144
9add8
B0
.
	full_text!

%145 = add nuw nsw i32 %92, 1
%i328
B

	full_text
	
i32 %92
8icmp8
B.
,
	full_text

%146 = icmp eq i32 %145, 16
&i328
B

	full_text


i32 %145
;br8
B3
1
	full_text$
"
 br i1 %146, label %90, label %91
$i18
B

	full_text
	
i1 %146
$i328B

	full_text


i32 %2
*float*8B

	full_text

	float* %6
(float8B

	full_text


float %3
(float8B

	full_text


float %4
(float8B

	full_text


float %5
*float*8B

	full_text

	float* %1
*float*8B
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
$i328B

	full_text


i32 16
#i328B

	full_text	

i32 2
$i328B

	full_text


i32 -1
$i328B

	full_text


i32 18
#i328B

	full_text	

i32 1
#i328B

	full_text	

i32 0       	  

                        !" !! #$ ## %& %' %% () (* (( +, ++ -. -- /1 00 23 25 44 67 68 66 9: 9; 9< 99 => =? == @A @B @C @@ DE DD FG FF HI HH JK JJ LM LL NO NN PQ PP RS RT RR UV UU WX WW YZ Y\ [[ ]_ ^^ `a `b `` cd ce cc fg fh ff ij ik il ii mn mm op oo qr qq st ss uv uu wx ww yz yy {| {} {{ ~ ~~ ÄÅ ÄÄ ÇÉ ÇÖ ÑÑ Üá Ü
à ÜÜ âä âå ãã çé çç è
ë êê íì í
î íí ïñ ï
ó ï
ò ïï ôö ô
õ ôô úù ú
û ú
ü úú †° †† ¢
£ ¢¢ §• §§ ¶ß ¶¶ ®© ®® ™
´ ™™ ¨≠ ¨¨ ÆØ Æ
∞ ÆÆ ±≤ ±± ≥¥ ≥≥ µ∂ µ∑ ∏π ∏∏ ∫ª ∫∫ ºΩ ºº æ
¡ ¿¿ ¬√ ¬
ƒ ¬¬ ≈∆ ≈
« ≈
» ≈≈ …  …
À …… ÃÕ Ã
Œ Ã
œ ÃÃ –— –
“ –– ”‘ ”
’ ”
÷ ”” ◊ÿ ◊
Ÿ ◊
⁄ ◊◊ €‹ €
› €
ﬁ €€ ﬂ‡ ﬂ
· ﬂ
‚ ﬂﬂ „‰ „
Â „
Ê „„ ÁË Á
È Á
Í ÁÁ ÎÏ Î
Ì Î
Ó ÎÎ Ô ÔÔ Ò
Ú ÒÒ ÛÙ ÛÛ ıˆ ıı ˜
¯ ˜˜ ˘˙ ˘˘ ˚¸ ˚˚ ˝
˛ ˝˝ ˇÄ ˇˇ ÅÇ Å
É ÅÅ ÑÖ ÑÑ Ü
á ÜÜ àâ àà äã ä
å ää çé çç è
ê èè ëí ëë ìî ì
ï ìì ñó ññ ò
ô òò öõ öö úù úú û
ü ûû †° †† ¢£ ¢
§ ¢¢ •¶ •• ß
® ßß ©™ ©© ´¨ ´
≠ ´´ ÆØ ÆÆ ∞
± ∞∞ ≤≥ ≤≤ ¥µ ¥
∂ ¥¥ ∑∏ ∑∑ π
∫ π
ª ππ º
Ω º
æ ºº ø¿ ø
¡ øø ¬√ ¬
ƒ ¬
≈ ¬¬ ∆« ∆∆ »
… »»  À  
Ã    ÕŒ ÕÕ œ– œœ —“ —	” ” #‘ N‘ w‘ ™‘ Ò‘ ˜‘ ˝‘ Ü‘ è‘ ò‘ û‘ ß‘ ∞’ π
÷ ∑◊ ºÿ »Ÿ FŸ oŸ ¢   	
              " $! &# '% ) * , . 10 3U 5+ 74 86 : ; <- >4 ?= A B( C@ ED GF IH K9 ML ON QJ SP T4 VU XW Z \~ _+ a^ b` d e- g^ hf j[ k( li nm po rq tc vu xw zs |y }^ ~ ÅÄ É ÖÑ á àÜ ä å é± ë+ ìê îí ñã ó ò- öê õô ùç û( üú °† £¢ •§ ßï ©® ´™ ≠¶ Ø¨ ∞ê ≤± ¥≥ ∂ π ª ΩÕ ¡¿ √ ƒ¬ ∆ « »+  ¿ À… Õ Œ œ∏ —¿ “– ‘ ’ ÷¬ ÿ∫ Ÿ ⁄¬ ‹º › ﬁ… ‡∫ · ‚– ‰∫ Â Ê… Ëº È Í– Ïº Ì Ó≈ Ô ÚÒ ÙÃ ˆı ¯˜ ˙” ¸˚ ˛˝ Ä˘ Çˇ É◊ ÖÑ áÜ âÅ ãà å€ éç êè íä îë ïﬂ óñ ôò õ„ ùú üû °ö £† §Á ¶• ®ß ™¢ ¨© ≠Î ØÆ ±∞ ≥´ µ≤ ∂ì ∏Û ∫∑ ª¥ Ωπ æ¿ ¿ ¡ø √ ƒ( ≈¬ «∆ …º À» Ã¿ ŒÕ –œ “/ 4Y 0Y 42 [2 Ñ] ^â ãâ ∑Ç ∑Ç ^è êæ ¿µ ∑µ ê— ø— ¿ ø €€ ﬁﬁ ⁄⁄ ‚‚ ·· ﬂﬂ ‹‹ ›› ‡‡Î ‡‡ Îπ ‚‚ πï ‡‡ ï¬ ‡‡ ¬
 ‹‹ 
º ‚‚ º €€ ú ‡‡ ú∑ ·· ∑c ‡‡ c ⁄⁄ Ã ‡‡ Ã◊ ‡‡ ◊ ⁄⁄ 9 ‡‡ 9@ ‡‡ @€ ‡‡ €„ ‡‡ „ ‹‹  ›› ﬂ ‡‡ ﬂ ﬂﬂ i ‡‡ i ﬁﬁ Á ‡‡ Á≈ ‡‡ ≈” ‡‡ ”	„ 
„ œ	‰ 	Â +	Â -	Â [	Â c
Â Ñ
Â º	Ê W
Ê Ä
Ê ≥Á Á Á Á 	Á U	Á ~
Á ã
Á ç
Á ±Á ∑
Á ∏
Á ∫
Á ÕË Ë 
	Ë !	Ë %	Ë 0Ë 4Ë ^Ë êË ¿"
StencilKernel"
_Z12get_group_idj"
_Z14get_num_groupsj"
_Z12get_local_idj"
_Z14get_local_sizej"
ToGlobalRow"
ToGlobalCol"
ToFlatHaloedIdx"
_Z7barrierj"
llvm.fmuladd.f32*û
%shoc-1.1.5-Stencil2D-StencilKernel.clu
4
llvm_target_triple

x86_64-apple-macosx10.13.0
=
llvm_data_layout)
'
%e-m:o-i64:64-f80:128-n8:16:32:64-S1282Å

wgsize_log1p
∫≤êA

transfer_bytes
ÄÇï"

devmap_label


wgsize
Ä
 
transfer_bytes_log1p
∫≤êA